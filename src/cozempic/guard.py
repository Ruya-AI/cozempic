"""Guard daemon — monitors session size and auto-prunes before compaction.

Runs as a background process that watches the active session JSONL file.
When the file crosses a configurable threshold, it:
1. Extracts team state from the conversation
2. Writes a checkpoint to disk
3. Prunes the session with team-protect
4. Injects team state recovery messages
5. Kills the Claude process and auto-resumes in a new terminal

This prevents auto-compaction from ever triggering, which means
agent team state is never lost.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .executor import run_prescription
from .registry import PRESCRIPTIONS
from .session import find_current_session, load_messages, save_messages
from .team import TeamState, extract_team_state, inject_team_recovery, write_team_checkpoint


# ─── Team-aware pruning ──────────────────────────────────────────────────────

def _is_team_message_by_index(messages, team_indices: set[int]):
    """Build a set of line indices that are team-related."""
    return team_indices


def prune_with_team_protect(
    messages: list,
    rx_name: str = "standard",
    config: dict | None = None,
) -> tuple[list, list, TeamState]:
    """Run a prescription but protect team-related messages from pruning.

    Returns (pruned_messages, strategy_results, team_state).

    Strategy:
    1. Extract team state first
    2. Mark team message indices
    3. Run prescription on non-team messages
    4. Re-insert team messages at their original positions
    5. Inject team recovery messages at the end
    """
    from .team import _is_team_message

    config = config or {}
    strategy_names = PRESCRIPTIONS.get(rx_name, PRESCRIPTIONS["standard"])

    # 1. Extract team state
    team_state = extract_team_state(messages)

    if team_state.is_empty():
        # No team — standard pruning
        new_messages, results = run_prescription(messages, strategy_names, config)
        return new_messages, results, team_state

    # 2. Separate team and non-team messages
    team_messages = []
    non_team_messages = []

    for msg_tuple in messages:
        line_idx, msg_dict, byte_size = msg_tuple
        if _is_team_message(msg_dict):
            team_messages.append(msg_tuple)
        else:
            non_team_messages.append(msg_tuple)

    # 3. Prune only non-team messages
    pruned_non_team, results = run_prescription(non_team_messages, strategy_names, config)

    # 4. Merge back: insert team messages at their original relative positions
    all_messages = list(pruned_non_team) + team_messages
    all_messages.sort(key=lambda m: m[0])  # Sort by original line index

    # 5. Inject team recovery messages at the end
    all_messages = inject_team_recovery(all_messages, team_state)

    return all_messages, results, team_state


# ─── Guard daemon ─────────────────────────────────────────────────────────────

def start_guard(
    cwd: str | None = None,
    threshold_mb: float = 50.0,
    rx_name: str = "standard",
    interval: int = 30,
    auto_reload: bool = True,
    config: dict | None = None,
) -> None:
    """Start the guard daemon.

    Monitors the current session's JSONL file and auto-prunes
    when it crosses the size threshold.

    Args:
        cwd: Working directory for session detection.
        threshold_mb: File size threshold in MB before pruning triggers.
        rx_name: Prescription to apply (gentle, standard, aggressive).
        interval: Check interval in seconds.
        auto_reload: If True, kill Claude and auto-resume after pruning.
        config: Extra config for pruning strategies.
    """
    threshold_bytes = int(threshold_mb * 1024 * 1024)

    # Find the initial session first
    sess = find_current_session(cwd)
    if not sess:
        print("  ERROR: Could not detect current session.", file=sys.stderr)
        print("  Make sure you're running from a directory with a Claude Code project.", file=sys.stderr)
        sys.exit(1)

    session_path = sess["path"]

    print(f"\n  COZEMPIC GUARD")
    print(f"  ═══════════════════════════════════════════════════════════════════")
    print(f"  Session:   {session_path.name}")
    print(f"  Size:      {sess['size'] / 1024 / 1024:.1f}MB")
    print(f"  Threshold: {threshold_mb}MB")
    print(f"  Rx:        {rx_name}")
    print(f"  Interval:  {interval}s")
    print(f"  Reload:    {'yes' if auto_reload else 'no'}")
    print(f"  Team-protect: enabled")
    print(f"\n  Guarding... (Ctrl+C to stop)")
    print()

    prune_count = 0

    try:
        while True:
            time.sleep(interval)

            # Re-check file size
            if not session_path.exists():
                print("  WARNING: Session file disappeared. Stopping guard.")
                break

            current_size = session_path.stat().st_size

            if current_size < threshold_bytes:
                continue

            # Threshold crossed — prune!
            prune_count += 1
            size_mb = current_size / 1024 / 1024
            print(f"  [{_now()}] Threshold crossed: {size_mb:.1f}MB > {threshold_mb}MB")
            print(f"  Pruning (cycle #{prune_count})...")

            result = guard_prune_cycle(
                session_path=session_path,
                rx_name=rx_name,
                config=config,
                auto_reload=auto_reload,
                cwd=cwd or os.getcwd(),
            )

            if result.get("reloading"):
                print(f"  Reload triggered. Guard exiting.")
                break

            print(f"  Pruned: {result['saved_mb']:.1f}MB saved")
            if result.get("team_name"):
                print(f"  Team '{result['team_name']}' state preserved ({result['team_messages']} messages)")
            print()

    except KeyboardInterrupt:
        print("\n  Guard stopped.")


def guard_prune_cycle(
    session_path: Path,
    rx_name: str = "standard",
    config: dict | None = None,
    auto_reload: bool = True,
    cwd: str = "",
) -> dict:
    """Execute a single guard prune cycle.

    Returns dict with: saved_mb, team_name, team_messages, reloading, checkpoint_path
    """
    messages = load_messages(session_path)
    original_bytes = sum(b for _, _, b in messages)

    # Prune with team protection
    pruned_messages, results, team_state = prune_with_team_protect(
        messages, rx_name=rx_name, config=config,
    )

    final_bytes = sum(b for _, _, b in pruned_messages)
    saved_bytes = original_bytes - final_bytes

    # Write checkpoint if team exists
    checkpoint_path = None
    if not team_state.is_empty():
        project_dir = session_path.parent
        checkpoint_path = write_team_checkpoint(team_state, project_dir)

    # Save pruned session
    backup = save_messages(session_path, pruned_messages, create_backup=True)

    result = {
        "saved_mb": saved_bytes / 1024 / 1024,
        "team_name": team_state.team_name,
        "team_messages": team_state.message_count,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "backup_path": str(backup) if backup else None,
        "reloading": False,
    }

    # Trigger reload if configured — kill Claude + resume
    if auto_reload:
        claude_pid = _find_claude_pid()
        if claude_pid:
            _spawn_reload_watcher(claude_pid, cwd)
            result["reloading"] = True
        else:
            print("  WARNING: Could not find Claude PID. Pruned but not reloading.")
            print("  Restart manually: claude --resume")

    return result


# ─── Process management (shared with cli.py reload) ──────────────────────────

def _find_claude_pid() -> int | None:
    """Walk up the process tree to find the Claude Code node process."""
    try:
        pid = os.getpid()
        for _ in range(10):
            result = subprocess.run(
                ["ps", "-o", "ppid=,comm=", "-p", str(pid)],
                capture_output=True, text=True,
            )
            parts = result.stdout.strip().split(None, 1)
            if len(parts) < 2:
                break
            ppid, comm = int(parts[0]), parts[1]
            if "node" in comm.lower() or "claude" in comm.lower():
                return pid
            pid = ppid
    except (ValueError, OSError):
        pass
    ppid = os.getppid()
    if ppid > 1:
        return ppid
    return None


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _spawn_reload_watcher(claude_pid: int, project_dir: str):
    """Spawn a detached watcher that resumes Claude after exit."""
    system = platform.system()

    if system == "Darwin":
        resume_cmd = (
            f"osascript -e 'tell application \"Terminal\" to do script "
            f"\"cd {_shell_quote(project_dir)} && claude --resume\"'"
        )
    elif system == "Linux":
        resume_cmd = (
            f"if command -v gnome-terminal >/dev/null 2>&1; then "
            f"gnome-terminal -- bash -c 'cd {_shell_quote(project_dir)} && claude --resume; exec bash'; "
            f"elif command -v xterm >/dev/null 2>&1; then "
            f"xterm -e 'cd {_shell_quote(project_dir)} && claude --resume' & "
            f"else echo 'No terminal emulator found' >> /tmp/cozempic_guard.log; fi"
        )
    elif system == "Windows":
        resume_cmd = (
            f"start cmd /c \"cd /d {project_dir} && claude --resume\""
        )
    else:
        print(f"  WARNING: Auto-resume not supported on {system}.")
        return

    watcher_script = (
        f"while kill -0 {claude_pid} 2>/dev/null; do sleep 1; done; "
        f"sleep 1; "
        f"{resume_cmd}; "
        f"echo \"$(date): Cozempic guard resumed Claude in {project_dir}\" >> /tmp/cozempic_guard.log"
    )

    subprocess.Popen(
        ["bash", "-c", watcher_script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")
