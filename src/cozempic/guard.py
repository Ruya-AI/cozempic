"""Guard daemon — continuous team checkpointing + emergency prune.

Architecture:
  EVERY interval:  Extract team state → write checkpoint (lightweight, no prune)
  AT threshold:    Prune non-team messages → inject recovery → optionally reload

The checkpoint runs continuously so team state is ALWAYS on disk, regardless
of whether the threshold is ever hit. The threshold prune is the emergency
fallback — not the primary protection mechanism.

Checkpoint triggers:
  1. Every N seconds (guard daemon)
  2. On demand via `cozempic checkpoint` (hook-driven)
  3. At file size threshold (emergency prune)
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .executor import run_prescription
from .helpers import is_ssh_session, shell_quote
from .registry import PRESCRIPTIONS
from .session import find_claude_pid, find_current_session, load_messages, save_messages
from .team import TeamState, extract_team_state, inject_team_recovery, write_team_checkpoint
from .tokens import quick_token_estimate


# ─── Lightweight checkpoint (no prune) ───────────────────────────────────────

def checkpoint_team(
    cwd: str | None = None,
    session_path: Path | None = None,
    quiet: bool = False,
) -> TeamState | None:
    """Extract and save team state from the current session. No pruning.

    This is fast and safe — it only reads the JSONL and writes a checkpoint.
    Designed to be called from hooks, guard daemon, or CLI.

    Returns the extracted TeamState, or None if no session found.
    """
    if session_path is None:
        sess = find_current_session(cwd)
        if not sess:
            if not quiet:
                print("  No active session found.", file=sys.stderr)
            return None
        session_path = sess["path"]

    messages = load_messages(session_path)
    state = extract_team_state(messages)

    if state.is_empty():
        if not quiet:
            print("  No team state detected.")
        return state

    project_dir = session_path.parent
    cp_path = write_team_checkpoint(state, project_dir)

    if not quiet:
        agents = len(state.subagents)
        teammates = len(state.teammates)
        tasks = len(state.tasks)
        parts = []
        if agents:
            parts.append(f"{agents} subagents")
        if teammates:
            parts.append(f"{teammates} teammates")
        if tasks:
            parts.append(f"{tasks} tasks")
        summary = ", ".join(parts) if parts else "empty"
        print(f"  Checkpoint: {summary} → {cp_path.name}")

    return state


# ─── Team-aware pruning ──────────────────────────────────────────────────────

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
    soft_threshold_mb: float | None = None,
    rx_name: str = "standard",
    interval: int = 30,
    auto_reload: bool = True,
    config: dict | None = None,
    reactive: bool = True,
    threshold_tokens: int | None = None,
    soft_threshold_tokens: int | None = None,
) -> None:
    """Start the guard daemon with tiered pruning.

    Three-phase protection:
      1. CHECKPOINT every interval — extract team state, write to disk
      2. SOFT PRUNE at soft threshold — gentle prune, no reload, no disruption
      3. HARD PRUNE at hard threshold — full prune with team-protect + optional reload

    Thresholds can be bytes-based, token-based, or both. When both are set,
    whichever is hit first triggers the action.

    Default soft threshold is 60% of hard threshold if not specified.

    Args:
        cwd: Working directory for session detection.
        threshold_mb: Hard threshold in MB — emergency prune + optional reload.
        soft_threshold_mb: Soft threshold in MB — gentle prune, no reload.
            Defaults to 60% of threshold_mb.
        rx_name: Prescription to apply at hard threshold.
        interval: Check interval in seconds.
        auto_reload: If True, kill Claude and auto-resume after hard prune.
        config: Extra config for pruning strategies.
        threshold_tokens: Hard threshold in tokens (optional, checked alongside bytes).
        soft_threshold_tokens: Soft threshold in tokens (optional, checked alongside bytes).
    """
    hard_threshold_bytes = int(threshold_mb * 1024 * 1024)

    if soft_threshold_mb is None:
        soft_threshold_mb = round(threshold_mb * 0.6, 1)
    soft_threshold_bytes = int(soft_threshold_mb * 1024 * 1024)

    # Find the initial session first
    sess = find_current_session(cwd)
    if not sess:
        print("  ERROR: Could not detect current session.", file=sys.stderr)
        print("  Make sure you're running from a directory with a Claude Code project.", file=sys.stderr)
        sys.exit(1)

    session_path = sess["path"]

    print(f"\n  COZEMPIC GUARD v3")
    print(f"  ═══════════════════════════════════════════════════════════════════")
    print(f"  Session:     {session_path.name}")
    print(f"  Size:        {sess['size'] / 1024 / 1024:.1f}MB")
    print(f"  Soft:        {soft_threshold_mb}MB (gentle prune, no reload)")
    print(f"  Hard:        {threshold_mb}MB (full prune + {'reload' if auto_reload else 'no reload'})")
    if soft_threshold_tokens is not None:
        print(f"  Soft tokens: {soft_threshold_tokens:,}")
    if threshold_tokens is not None:
        print(f"  Hard tokens: {threshold_tokens:,}")
    print(f"  Rx:          gentle (soft) / {rx_name} (hard)")
    print(f"  Interval:    {interval}s")
    print(f"  Team-protect: enabled")
    print(f"  Checkpoint:  continuous (every {interval}s)")
    print(f"  Reactive:    {'enabled' if reactive else 'disabled'}")
    print(f"\n  Guarding... (Ctrl+C to stop)")
    print()

    # Reactive overflow recovery via file watcher
    overflow_watcher = None
    if reactive:
        import threading
        from .overflow import CircuitBreaker, OverflowRecovery
        from .watcher import JsonlWatcher

        breaker = CircuitBreaker(session_id=sess["session_id"])
        recovery = OverflowRecovery(
            session_path, sess["session_id"], cwd or os.getcwd(), breaker,
        )
        overflow_watcher = JsonlWatcher(
            str(session_path), on_growth=recovery.on_file_growth,
        )
        watcher_thread = threading.Thread(
            target=overflow_watcher.start, daemon=True, name="cozempic-watcher",
        )
        watcher_thread.start()

    prune_count = 0
    soft_prune_count = 0
    checkpoint_count = 0
    last_team_hash = ""

    try:
        while True:
            time.sleep(interval)

            # Re-check file exists
            if not session_path.exists():
                print("  WARNING: Session file disappeared. Stopping guard.")
                break

            current_size = session_path.stat().st_size

            # ── Phase 1: Continuous checkpoint ────────────────────────
            state = checkpoint_team(
                session_path=session_path,
                quiet=True,
            )

            # Only log if team state changed
            if state and not state.is_empty():
                team_hash = f"{len(state.subagents)}:{len(state.tasks)}:{state.message_count}"
                if team_hash != last_team_hash:
                    checkpoint_count += 1
                    last_team_hash = team_hash
                    agents = len(state.subagents)
                    tasks = len(state.tasks)
                    size_mb = current_size / 1024 / 1024
                    print(
                        f"  [{_now()}] Checkpoint #{checkpoint_count}: "
                        f"{agents} agents, {tasks} tasks, "
                        f"{state.message_count} msgs "
                        f"({size_mb:.1f}MB)"
                    )

            # ── Token check (fast, from tail of file) ────────────────
            current_tokens = None
            if threshold_tokens is not None or soft_threshold_tokens is not None:
                current_tokens = quick_token_estimate(session_path)

            # ── Phase 3: HARD prune at hard threshold ─────────────────
            hard_bytes_hit = current_size >= hard_threshold_bytes
            hard_tokens_hit = (
                threshold_tokens is not None
                and current_tokens is not None
                and current_tokens >= threshold_tokens
            )
            if hard_bytes_hit or hard_tokens_hit:
                prune_count += 1
                size_mb = current_size / 1024 / 1024
                reason = f"{size_mb:.1f}MB >= {threshold_mb}MB" if hard_bytes_hit else f"{current_tokens:,} tokens >= {threshold_tokens:,}"
                print(f"  [{_now()}] HARD THRESHOLD: {reason}")
                print(f"  Emergency prune with {rx_name} (cycle #{prune_count})...")

                result = guard_prune_cycle(
                    session_path=session_path,
                    rx_name=rx_name,
                    config=config,
                    auto_reload=auto_reload,
                    cwd=cwd or os.getcwd(),
                    session_id=sess["session_id"],
                )

                if result.get("reloading"):
                    print(f"  Reload triggered. Guard exiting.")
                    break

                print(f"  Pruned: {_fmt_prune_result(result)}")
                if result.get("team_name"):
                    print(
                        f"  Team '{result['team_name']}' state preserved "
                        f"({result['team_messages']} messages)"
                    )
                print()

            # ── Phase 2: SOFT prune at soft threshold ─────────────────
            else:
                soft_bytes_hit = current_size >= soft_threshold_bytes
                soft_tokens_hit = (
                    soft_threshold_tokens is not None
                    and current_tokens is not None
                    and current_tokens >= soft_threshold_tokens
                )
                if soft_bytes_hit or soft_tokens_hit:
                    soft_prune_count += 1
                    size_mb = current_size / 1024 / 1024
                    reason = f"{size_mb:.1f}MB >= {soft_threshold_mb}MB" if soft_bytes_hit else f"{current_tokens:,} tokens >= {soft_threshold_tokens:,}"
                    print(f"  [{_now()}] SOFT THRESHOLD: {reason}")
                    print(f"  Gentle prune, no reload (cycle #{soft_prune_count})...")

                    result = guard_prune_cycle(
                        session_path=session_path,
                        rx_name="gentle",
                        config=config,
                        auto_reload=False,  # Never reload on soft prune
                        cwd=cwd or os.getcwd(),
                        session_id=sess["session_id"],
                    )

                    print(f"  Trimmed: {_fmt_prune_result(result)}")
                    if result.get("team_name"):
                        print(
                            f"  Team '{result['team_name']}' state preserved "
                            f"({result['team_messages']} messages)"
                        )
                    print()

    except KeyboardInterrupt:
        # Stop reactive watcher
        if overflow_watcher:
            overflow_watcher.stop()

        # Final checkpoint before exit
        print(f"\n  [{_now()}] Final checkpoint before exit...")
        checkpoint_team(session_path=session_path, quiet=False)
        total_prunes = prune_count + soft_prune_count
        print(
            f"  Guard stopped. {checkpoint_count} checkpoints, "
            f"{soft_prune_count} soft prunes, {prune_count} hard prunes."
        )


def guard_prune_cycle(
    session_path: Path,
    rx_name: str = "standard",
    config: dict | None = None,
    auto_reload: bool = True,
    cwd: str = "",
    session_id: str | None = None,
) -> dict:
    """Execute a single guard prune cycle.

    Returns dict with: saved_mb, team_name, team_messages, reloading, checkpoint_path
    """
    from .tokens import estimate_session_tokens

    messages = load_messages(session_path)
    original_bytes = sum(b for _, _, b in messages)

    # Token estimate before pruning
    pre_te = estimate_session_tokens(messages)

    # Prune with team protection
    pruned_messages, results, team_state = prune_with_team_protect(
        messages, rx_name=rx_name, config=config,
    )

    final_bytes = sum(b for _, _, b in pruned_messages)
    saved_bytes = original_bytes - final_bytes

    # Token estimate after pruning
    post_te = estimate_session_tokens(pruned_messages)

    # Write checkpoint if team exists
    checkpoint_path = None
    if not team_state.is_empty():
        project_dir = session_path.parent
        checkpoint_path = write_team_checkpoint(team_state, project_dir)

    # Save pruned session
    backup = save_messages(session_path, pruned_messages, create_backup=True)

    result = {
        "saved_mb": saved_bytes / 1024 / 1024,
        "original_tokens": pre_te.total,
        "final_tokens": post_te.total,
        "team_name": team_state.team_name,
        "team_messages": team_state.message_count,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "backup_path": str(backup) if backup else None,
        "reloading": False,
    }

    # Trigger reload if configured — kill Claude + resume
    if auto_reload:
        claude_pid = find_claude_pid()
        if claude_pid:
            _spawn_reload_watcher(claude_pid, cwd, session_id=session_id)
            result["reloading"] = True
        else:
            resume_flag = f"--resume {session_id}" if session_id else "--resume"
            print("  WARNING: Could not find Claude PID. Pruned but not reloading.")
            print(f"  Restart manually: claude {resume_flag}")

    return result


def _spawn_reload_watcher(claude_pid: int, project_dir: str, session_id: str | None = None):
    """Spawn a detached watcher that resumes Claude after exit."""
    resume_flag = f"--resume {session_id}" if session_id else "--resume"

    # SSH sessions can't open GUI terminals — skip auto-resume
    if is_ssh_session():
        print(f"  SSH session detected — skipping auto-resume.")
        print(f"  Resume manually: cd {project_dir} && claude {resume_flag}")
        return

    system = platform.system()

    if system == "Darwin":
        resume_cmd = (
            f"osascript -e 'tell application \"Terminal\" to do script "
            f"\"cd {shell_quote(project_dir)} && claude {resume_flag}\"'"
        )
    elif system == "Linux":
        resume_cmd = (
            f"if command -v gnome-terminal >/dev/null 2>&1; then "
            f"gnome-terminal -- bash -c 'cd {shell_quote(project_dir)} && claude {resume_flag}; exec bash'; "
            f"elif command -v xterm >/dev/null 2>&1; then "
            f"xterm -e 'cd {shell_quote(project_dir)} && claude {resume_flag}' & "
            f"else echo 'No terminal emulator found' >> /tmp/cozempic_guard.log; fi"
        )
    elif system == "Windows":
        resume_cmd = (
            f"start cmd /c \"cd /d {project_dir} && claude {resume_flag}\""
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


def _pid_file(cwd: str) -> Path:
    """Return the PID file path for a guard daemon in this project."""
    # Use a hash of the cwd so each project gets its own PID file
    import hashlib
    slug = hashlib.md5(cwd.encode()).hexdigest()[:12]
    return Path("/tmp") / f"cozempic_guard_{slug}.pid"


def _is_guard_running(cwd: str) -> int | None:
    """Check if a guard daemon is already running for this project.

    Returns the PID if running, None otherwise.
    """
    pid_path = _pid_file(cwd)
    if not pid_path.exists():
        return None

    try:
        pid = int(pid_path.read_text().strip())
        # Check if process is actually alive
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        # Stale PID file — clean it up
        pid_path.unlink(missing_ok=True)
        return None


def start_guard_daemon(
    cwd: str | None = None,
    threshold_mb: float = 50.0,
    soft_threshold_mb: float | None = None,
    rx_name: str = "standard",
    interval: int = 30,
    auto_reload: bool = True,
    reactive: bool = True,
    threshold_tokens: int | None = None,
    soft_threshold_tokens: int | None = None,
) -> dict:
    """Start the guard as a background daemon.

    Spawns a detached subprocess running `cozempic guard` with output
    redirected to a log file. Uses a PID file to prevent double-starts.

    Returns dict with: started (bool), pid (int|None), pid_file, log_file,
    already_running (bool).
    """
    cwd = cwd or os.getcwd()

    existing_pid = _is_guard_running(cwd)
    if existing_pid:
        return {
            "started": False,
            "pid": existing_pid,
            "pid_file": str(_pid_file(cwd)),
            "log_file": None,
            "already_running": True,
        }

    import hashlib
    slug = hashlib.md5(cwd.encode()).hexdigest()[:12]
    log_file = Path("/tmp") / f"cozempic_guard_{slug}.log"
    pid_path = _pid_file(cwd)

    # Build the guard command
    cmd_parts = [
        sys.executable, "-m", "cozempic.cli", "guard",
        "--cwd", cwd,
        "--threshold", str(threshold_mb),
        "--interval", str(interval),
        "-rx", rx_name,
    ]
    if soft_threshold_mb is not None:
        cmd_parts.extend(["--soft-threshold", str(soft_threshold_mb)])
    if not auto_reload:
        cmd_parts.append("--no-reload")
    if not reactive:
        cmd_parts.append("--no-reactive")
    if threshold_tokens is not None:
        cmd_parts.extend(["--threshold-tokens", str(threshold_tokens)])
    if soft_threshold_tokens is not None:
        cmd_parts.extend(["--soft-threshold-tokens", str(soft_threshold_tokens)])

    # Spawn detached process
    with open(log_file, "a", encoding="utf-8") as lf:
        from datetime import datetime
        lf.write(f"\n--- Guard daemon started at {datetime.now().isoformat()} ---\n")
        lf.write(f"CWD: {cwd}\n")
        lf.write(f"CMD: {' '.join(cmd_parts)}\n\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd_parts,
            stdout=lf,
            stderr=lf,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=cwd,
        )

    # Write PID file
    pid_path.write_text(str(proc.pid))

    return {
        "started": True,
        "pid": proc.pid,
        "pid_file": str(pid_path),
        "log_file": str(log_file),
        "already_running": False,
    }


def _fmt_prune_result(result: dict) -> str:
    """Format a prune cycle result, leading with tokens if available."""
    orig_tok = result.get("original_tokens")
    final_tok = result.get("final_tokens")
    if orig_tok and final_tok:
        saved_tok = orig_tok - final_tok
        tok_str = f"{saved_tok / 1000:.1f}K" if saved_tok >= 1000 else str(saved_tok)
        pct = f"{saved_tok / orig_tok * 100:.1f}%" if orig_tok > 0 else "0%"
        return f"{tok_str} tokens freed ({pct}), {result['saved_mb']:.1f}MB saved"
    return f"{result['saved_mb']:.1f}MB saved"


def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")
