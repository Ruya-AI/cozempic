"""CLI interface for Cozempic."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

from .diagnosis import diagnose_session
from .doctor import run_doctor
from .executor import execute_actions, run_prescription
from .guard import checkpoint_team, start_guard, start_guard_daemon
from .init import run_init
from .recap import save_recap
from .registry import PRESCRIPTIONS, STRATEGIES
from .helpers import shell_quote
from .session import find_claude_pid, find_current_session, find_sessions, load_messages, project_slug_to_path, resolve_session, save_messages
from .types import PrescriptionResult, StrategyResult

# Ensure all strategies are registered
import cozempic.strategies  # noqa: F401

# Fix Windows stdout/stderr encoding for Unicode characters (box-drawing, emoji)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ─── Formatting ───────────────────────────────────────────────────────────────

def fmt_bytes(b: int) -> str:
    if b < 1024:
        return f"{b}B"
    elif b < 1024 * 1024:
        return f"{b / 1024:.1f}KB"
    else:
        return f"{b / (1024 * 1024):.2f}MB"


def fmt_pct(part: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{part / total * 100:.1f}%"


def print_diagnosis(diag: dict, path: Path):
    total = diag["total_bytes"]
    print(f"\n  Patient: {path.stem}")
    print(f"  Weight:  {fmt_bytes(total)} ({diag['total_messages']} messages)")
    print()

    print("  Vital Signs:")
    print(f"    Progress ticks:     {diag['progress_count']:>6}")
    print(f"    File history snaps: {diag['file_history_count']:>6}")
    print(f"    System reminders:   {diag['reminder_count']:>6}")
    print(f"    Thinking content:   {fmt_bytes(diag['thinking_bytes']):>10} ({fmt_pct(diag['thinking_bytes'], total)})")
    print(f"    Signatures:         {fmt_bytes(diag['signature_bytes']):>10} ({fmt_pct(diag['signature_bytes'], total)})")
    print(f"    Tool results:       {fmt_bytes(diag['tool_result_bytes']):>10} ({fmt_pct(diag['tool_result_bytes'], total)})")
    print()

    print("  Message Type Breakdown:")
    sorted_types = sorted(diag["type_stats"].items(), key=lambda x: x[1]["bytes"], reverse=True)
    for mtype, stats in sorted_types:
        pct = fmt_pct(stats["bytes"], total)
        print(f"    {mtype:<28} {stats['count']:>5} msgs  {fmt_bytes(stats['bytes']):>10}  ({pct})")
    print()

    print("  Top 10 Largest Messages:")
    for size, idx, mtype, pos in diag["largest_messages"][:10]:
        print(f"    Line {idx:<6}  {mtype:<20}  {fmt_bytes(size)}")
    print()


def print_strategy_result(sr: StrategyResult, total_bytes: int):
    saved = sum(a.original_bytes - a.pruned_bytes for a in sr.actions) if sr.actions else 0
    pct = fmt_pct(saved, total_bytes)

    detail_parts = []
    if sr.messages_removed:
        detail_parts.append(f"{sr.messages_removed} removed")
    if sr.messages_replaced:
        detail_parts.append(f"{sr.messages_replaced} modified")
    detail = f" ({', '.join(detail_parts)})" if detail_parts else ""

    print(f"    {sr.strategy_name:<30} {fmt_bytes(saved):>10} saved  ({pct}){detail}  {sr.summary}")


def print_prescription_result(pr: PrescriptionResult):
    saved = pr.original_total_bytes - pr.final_total_bytes
    pct = fmt_pct(saved, pr.original_total_bytes)
    removed = pr.original_message_count - pr.final_message_count
    total_replaced = sum(sr.messages_replaced for sr in pr.strategy_results)

    print(f"\n  Prescription: {pr.prescription_name}")
    print(f"  Before: {fmt_bytes(pr.original_total_bytes)} ({pr.original_message_count} messages)")
    print(f"  After:  {fmt_bytes(pr.final_total_bytes)} ({pr.final_message_count} messages)")
    print(f"  Saved:  {fmt_bytes(saved)} ({pct}) — {removed} removed, {total_replaced} modified")
    print()
    print("  Strategy Results:")
    for sr in pr.strategy_results:
        print_strategy_result(sr, pr.original_total_bytes)
    print()


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_list(args):
    sessions = find_sessions(args.project)
    if not sessions:
        print("No sessions found.")
        return

    print(f"\n  {'Session ID':<40} {'Size':>10} {'Messages':>8} {'Modified':<20} Project")
    print(f"  {'─' * 40} {'─' * 10} {'─' * 8} {'─' * 20} {'─' * 30}")

    for sess in sorted(sessions, key=lambda s: s["size"], reverse=True):
        sid = sess["session_id"]
        if len(sid) > 36:
            sid = sid[:33] + "..."
        print(
            f"  {sid:<40} {fmt_bytes(sess['size']):>10} {sess['lines']:>8}"
            f" {sess['mtime'].strftime('%Y-%m-%d %H:%M'):<20} {sess['project'][-40:]}"
        )
    print()

    total = sum(s["size"] for s in sessions)
    print(f"  Total: {len(sessions)} sessions, {fmt_bytes(total)}")
    print()


def cmd_current(args):
    cwd = args.cwd or None
    match_text = getattr(args, "match", None)
    sess = find_current_session(cwd, match_text=match_text)
    if not sess:
        print("Could not detect current session.", file=sys.stderr)
        print("Make sure you're running from a directory with a Claude Code project.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Current Session:")
    print(f"    ID:      {sess['session_id']}")
    print(f"    Size:    {fmt_bytes(sess['size'])} ({sess['lines']} messages)")
    print(f"    Project: {sess['project']}")
    print(f"    Path:    {sess['path']}")
    print(f"    Modified: {sess['mtime'].strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if args.diagnose:
        messages = load_messages(sess["path"])
        diag = diagnose_session(messages)
        print_diagnosis(diag, sess["path"])

        print("  Estimated Savings by Prescription:")
        for rx_name, strategy_names in PRESCRIPTIONS.items():
            new_msgs, _ = run_prescription(messages, strategy_names, {})
            final_bytes = sum(b for _, _, b in new_msgs)
            total_saved = diag["total_bytes"] - final_bytes
            pct = fmt_pct(total_saved, diag["total_bytes"])
            print(f"    {rx_name:<15} ~{fmt_bytes(total_saved):>10} ({pct})")
        print()


def cmd_diagnose(args):
    path = resolve_session(args.session, getattr(args, "project", None))
    messages = load_messages(path)
    diag = diagnose_session(messages)
    print_diagnosis(diag, path)

    print("  Estimated Savings by Prescription:")
    for rx_name, strategy_names in PRESCRIPTIONS.items():
        new_msgs, _ = run_prescription(messages, strategy_names, {})
        final_bytes = sum(b for _, _, b in new_msgs)
        total_saved = diag["total_bytes"] - final_bytes
        pct = fmt_pct(total_saved, diag["total_bytes"])
        print(f"    {rx_name:<15} ~{fmt_bytes(total_saved):>10} ({pct})")
    print()


def cmd_treat(args):
    path = resolve_session(args.session, getattr(args, "project", None))
    messages = load_messages(path)
    rx_name = args.rx or "standard"

    if rx_name not in PRESCRIPTIONS:
        print(f"Error: Unknown prescription '{rx_name}'. Options: {', '.join(PRESCRIPTIONS)}", file=sys.stderr)
        sys.exit(1)

    strategy_names = PRESCRIPTIONS[rx_name]
    config = {}
    if args.thinking_mode:
        config["thinking_mode"] = args.thinking_mode

    original_bytes = sum(b for _, _, b in messages)
    original_count = len(messages)

    new_messages, strategy_results = run_prescription(messages, strategy_names, config)
    final_bytes = sum(b for _, _, b in new_messages)
    final_count = len(new_messages)

    pr = PrescriptionResult(
        prescription_name=rx_name,
        strategy_results=strategy_results,
        original_total_bytes=original_bytes,
        final_total_bytes=final_bytes,
        original_message_count=original_count,
        final_message_count=final_count,
    )

    print_prescription_result(pr)

    if args.execute:
        backup = save_messages(path, new_messages, create_backup=True)
        print(f"  Treatment applied to {path}")
        if backup:
            print(f"  Backup: {backup}")
        print(f"  Final size: {fmt_bytes(final_bytes)}")
    else:
        print("  DRY RUN — no changes made. Use --execute to apply.")
    print()


def cmd_strategy(args):
    path = resolve_session(args.session, getattr(args, "project", None))
    messages = load_messages(path)

    if args.name not in STRATEGIES:
        print(f"Error: Unknown strategy '{args.name}'.", file=sys.stderr)
        print(f"Available: {', '.join(sorted(STRATEGIES))}", file=sys.stderr)
        sys.exit(1)

    config = {}
    if args.thinking_mode:
        config["thinking_mode"] = args.thinking_mode

    original_bytes = sum(b for _, _, b in messages)
    sr = STRATEGIES[args.name].func(messages, config)

    saved = sum(a.original_bytes - a.pruned_bytes for a in sr.actions)
    print(f"\n  Strategy: {sr.strategy_name}")
    print(f"  Savings: {fmt_bytes(saved)} ({fmt_pct(saved, original_bytes)})")
    print(f"  Actions: {len(sr.actions)} ({sr.messages_removed} removed, {sr.messages_replaced} modified)")
    print(f"  Summary: {sr.summary}")
    print()

    if args.verbose:
        for a in sr.actions[:20]:
            print(f"    Line {a.line_index:<6} {a.action:<8} {fmt_bytes(a.original_bytes):>10} -> {fmt_bytes(a.pruned_bytes):>10}  {a.reason}")
        if len(sr.actions) > 20:
            print(f"    ... and {len(sr.actions) - 20} more actions")
        print()

    if args.execute:
        new_messages = execute_actions(messages, sr.actions)
        backup = save_messages(path, new_messages, create_backup=True)
        final_bytes = sum(b for _, _, b in new_messages)
        print(f"  Applied. Final size: {fmt_bytes(final_bytes)}")
        if backup:
            print(f"  Backup: {backup}")
    else:
        print("  DRY RUN — no changes made. Use --execute to apply.")
    print()


def cmd_reload(args):
    """Treat the current session, then spawn a watcher that auto-resumes Claude."""
    cwd = args.cwd or os.getcwd()
    sess = find_current_session(cwd)
    if not sess:
        print("Could not detect current session.", file=sys.stderr)
        print("Make sure you're running from a directory with a Claude Code project.", file=sys.stderr)
        sys.exit(1)

    # Derive project directory from session slug (more reliable than CWD)
    project_dir = project_slug_to_path(sess["project"])
    if os.path.isdir(project_dir):
        cwd = project_dir

    rx_name = args.rx or "standard"
    if rx_name not in PRESCRIPTIONS:
        print(f"Error: Unknown prescription '{rx_name}'. Options: {', '.join(PRESCRIPTIONS)}", file=sys.stderr)
        sys.exit(1)

    # Step 1: Apply treatment
    path = sess["path"]
    messages = load_messages(path)
    strategy_names = PRESCRIPTIONS[rx_name]
    config = {}
    if args.thinking_mode:
        config["thinking_mode"] = args.thinking_mode

    original_bytes = sum(b for _, _, b in messages)
    original_count = len(messages)

    new_messages, strategy_results = run_prescription(messages, strategy_names, config)
    final_bytes = sum(b for _, _, b in new_messages)
    final_count = len(new_messages)

    pr = PrescriptionResult(
        prescription_name=rx_name,
        strategy_results=strategy_results,
        original_total_bytes=original_bytes,
        final_total_bytes=final_bytes,
        original_message_count=original_count,
        final_message_count=final_count,
    )
    print_prescription_result(pr)

    backup = save_messages(path, new_messages, create_backup=True)
    print(f"  Treatment applied to {path}")
    if backup:
        print(f"  Backup: {backup}")
    print(f"  Final size: {fmt_bytes(final_bytes)}")
    print()

    # Step 2: Generate recap from the pruned messages
    import tempfile
    recap_path = Path(tempfile.gettempdir()) / f"cozempic_recap_{sess['session_id'][:8]}.txt"
    save_recap(new_messages, recap_path)
    print(f"  Recap saved to {recap_path}")

    # Step 3: Find Claude's parent PID and spawn watcher
    claude_pid = find_claude_pid()
    if not claude_pid:
        print("  WARNING: Could not detect Claude Code process.")
        print("  Treatment was applied, but auto-resume watcher was NOT started.")
        print("  Restart Claude manually with: claude --resume")
        return

    _spawn_watcher(claude_pid, cwd, recap_path=recap_path, session_id=sess["session_id"])
    print(f"  Watcher spawned (watching Claude PID {claude_pid}).")
    print(f"  Now type /exit — a new terminal will open with 'claude --resume'.")
    print()


def _spawn_watcher(claude_pid: int, project_dir: str, recap_path: Path | None = None, session_id: str | None = None):
    """Spawn a detached background process that waits for Claude to exit, then resumes."""
    system = platform.system()

    # Build the command sequence: show recap, then launch claude --resume
    recap_cmd = ""
    if recap_path and recap_path.exists():
        recap_cmd = f"cat {shell_quote(str(recap_path))}; echo; "

    # Use session ID for precise resume targeting
    resume_flag = f"--resume {session_id}" if session_id else "--resume"

    if system == "Darwin":
        inner_cmd = f"cd {shell_quote(project_dir)} && {recap_cmd}claude {resume_flag}"
        resume_cmd = (
            f"osascript -e 'tell application \"Terminal\" to do script "
            f"\"{inner_cmd}\"'"
        )
    elif system == "Linux":
        inner_cmd = f"cd {shell_quote(project_dir)} && {recap_cmd}claude {resume_flag}; exec bash"
        resume_cmd = (
            f"if command -v gnome-terminal >/dev/null 2>&1; then "
            f"gnome-terminal -- bash -c '{inner_cmd}'; "
            f"elif command -v xterm >/dev/null 2>&1; then "
            f"xterm -e '{inner_cmd}' & "
            f"else echo 'No terminal emulator found' >> /tmp/cozempic_reload.log; fi"
        )
    else:
        print(f"  WARNING: Auto-resume not supported on {system}.")
        print(f"  Restart manually: cd {project_dir} && claude {resume_flag}")
        return

    watcher_script = (
        f"while kill -0 {claude_pid} 2>/dev/null; do sleep 1; done; "
        f"sleep 1; "
        f"{resume_cmd}; "
        f"echo \"$(date): Cozempic resumed Claude in {project_dir}\" >> /tmp/cozempic_reload.log"
    )

    subprocess.Popen(
        ["bash", "-c", watcher_script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # fully detach from parent
    )


def cmd_checkpoint(args):
    """Save team/agent state from the current session. No pruning."""
    state = checkpoint_team(cwd=args.cwd or os.getcwd())
    if state and not state.is_empty():
        if args.show:
            print()
            print(state.to_recovery_text())
            print()


def cmd_guard(args):
    """Start the guard daemon to prevent compaction-induced state loss."""
    if args.daemon:
        result = start_guard_daemon(
            cwd=args.cwd or os.getcwd(),
            threshold_mb=args.threshold,
            soft_threshold_mb=args.soft_threshold,
            rx_name=args.rx or "standard",
            interval=args.interval,
            auto_reload=not args.no_reload,
            reactive=not args.no_reactive,
        )
        if result["already_running"]:
            print(f"  Guard already running (PID {result['pid']})")
        elif result["started"]:
            print(f"  Guard daemon started (PID {result['pid']})")
            print(f"  Log: {result['log_file']}")
        return

    start_guard(
        cwd=args.cwd or os.getcwd(),
        threshold_mb=args.threshold,
        soft_threshold_mb=args.soft_threshold,
        rx_name=args.rx or "standard",
        interval=args.interval,
        auto_reload=not args.no_reload,
        reactive=not args.no_reactive,
    )


def cmd_doctor(args):
    """Run health checks on Claude Code configuration and sessions."""
    STATUS_ICONS = {
        "ok": "✓",
        "warning": "⚠",
        "issue": "✗",
        "fixed": "→",
    }
    STATUS_COLORS = {
        "ok": "",
        "warning": "",
        "issue": "",
        "fixed": "",
    }

    results = run_doctor(fix=args.fix)

    print("\n  COZEMPIC DOCTOR")
    print("  ═══════════════════════════════════════════════════════════════════")
    print()

    issues = 0
    warnings = 0
    fixed = 0

    for r in results:
        icon = STATUS_ICONS.get(r.status, "?")
        print(f"    {icon} {r.name:<25} [{r.status.upper()}]")
        print(f"      {r.message}")
        if r.fix_description and r.status not in ("ok", "fixed"):
            print(f"      Fix: {r.fix_description}")
        print()

        if r.status == "issue":
            issues += 1
        elif r.status == "warning":
            warnings += 1
        elif r.status == "fixed":
            fixed += 1

    # Summary
    if fixed:
        print(f"  Summary: {fixed} issue(s) fixed")
    elif issues or warnings:
        print(f"  Summary: {issues} issue(s), {warnings} warning(s)")
        if not args.fix:
            print("  Run 'cozempic doctor --fix' to auto-fix where possible.")
    else:
        print("  All clear — no issues found.")
    print()


def cmd_init(args):
    """Wire cozempic hooks and slash command into the current project."""
    project_dir = args.cwd or os.getcwd()

    print(f"\n  COZEMPIC INIT")
    print(f"  ═══════════════════════════════════════════════════════════════════")
    print(f"  Project: {project_dir}")
    print()

    result = run_init(project_dir, skip_slash=args.no_slash_command)

    # Report hooks
    hooks = result["hooks"]
    if hooks["added"]:
        print(f"  Hooks added to {hooks['settings_path']}:")
        for h in hooks["added"]:
            print(f"    + {h}")
        if hooks["backup_path"]:
            print(f"  Backup: {hooks['backup_path']}")
    else:
        print(f"  Hooks: already configured (nothing to add)")

    if hooks["skipped"]:
        for h in hooks["skipped"]:
            print(f"    ~ {h} (already exists, skipped)")

    print()

    # Report slash command
    slash = result["slash_command"]
    if slash.get("updated"):
        print(f"  Slash command: updated → {slash['path']}")
        print(f"  Use /cozempic in any Claude Code session to diagnose and treat.")
    elif slash["installed"]:
        print(f"  Slash command: installed → {slash['path']}")
        print(f"  Use /cozempic in any Claude Code session to diagnose and treat.")
    elif slash["already_existed"]:
        print(f"  Slash command: up-to-date at {slash['path']}")
    elif not args.no_slash_command:
        print(f"  Slash command: source not found (install from git repo to get it)")

    print()

    # Summary: what to do next
    print(f"  Setup complete. Protection is fully automatic:")
    print(f"    - Guard daemon auto-starts on every session (SessionStart hook)")
    print(f"    - Team state checkpointed on every agent event (PostToolUse hooks)")
    print(f"    - Emergency checkpoint before compaction (PreCompact hook)")
    print(f"    - Final checkpoint on session end (Stop hook)")
    print()
    print(f"  Just start Claude Code normally. No second terminal needed.")
    print()


def cmd_formulary(args):
    print("\n  COZEMPIC FORMULARY")
    print("  ═══════════════════════════════════════════════════════════════════")
    print()
    print("  Strategies:")
    print(f"  {'#':<4} {'Name':<30} {'Tier':<12} {'Expected':>10}  Description")
    print(f"  {'─' * 4} {'─' * 30} {'─' * 12} {'─' * 10}  {'─' * 40}")
    for i, (name, info) in enumerate(STRATEGIES.items(), 1):
        print(f"  {i:<4} {name:<30} {info.tier:<12} {info.expected_savings:>10}  {info.description}")
    print()

    print("  Prescriptions:")
    for rx_name, strategy_names in PRESCRIPTIONS.items():
        names = ", ".join(strategy_names)
        print(f"    {rx_name:<15} [{len(strategy_names)} strategies] {names}")
    print()

    print("  Usage:")
    print("    cozempic treat <session> -rx gentle      # Safe, minimal pruning")
    print("    cozempic treat <session> -rx standard     # Recommended (default)")
    print("    cozempic treat <session> -rx aggressive   # Maximum savings")
    print("    cozempic treat <session> --execute        # Apply (default is dry-run)")
    print()


# ─── Parser ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cozempic",
        description="Context weight-loss tool for Claude Code — prune bloated JSONL conversation files",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.5.2")
    sub = parser.add_subparsers(dest="command")

    session_help = "Session ID, UUID prefix, path, or 'current' for auto-detect"

    # list
    p_list = sub.add_parser("list", help="List sessions with sizes")
    p_list.add_argument("--project", help="Filter by project name")

    # current
    p_current = sub.add_parser("current", help="Show current session for this project")
    p_current.add_argument("--cwd", help="Working directory (default: current)")
    p_current.add_argument("--match", help="Text snippet to match against session content (for multi-session disambiguation)")
    p_current.add_argument("--diagnose", "-d", action="store_true", help="Also run diagnosis")

    # diagnose
    p_diag = sub.add_parser("diagnose", help="Analyze bloat sources (read-only)")
    p_diag.add_argument("session", help=session_help)
    p_diag.add_argument("--project", help="Filter by project name")

    # treat
    p_treat = sub.add_parser("treat", help="Run prescription (dry-run by default)")
    p_treat.add_argument("session", help=session_help)
    p_treat.add_argument("-rx", help="Prescription: gentle, standard, aggressive")
    p_treat.add_argument("--execute", action="store_true", help="Apply changes (default is dry-run)")
    p_treat.add_argument("--project", help="Filter by project name")
    p_treat.add_argument("--thinking-mode", choices=["remove", "truncate", "signature-only"], help="Thinking block mode")

    # strategy
    p_strat = sub.add_parser("strategy", help="Run single strategy")
    p_strat.add_argument("name", help="Strategy name")
    p_strat.add_argument("session", help=session_help)
    p_strat.add_argument("--execute", action="store_true", help="Apply changes")
    p_strat.add_argument("--verbose", "-v", action="store_true", help="Show action details")
    p_strat.add_argument("--project", help="Filter by project name")
    p_strat.add_argument("--thinking-mode", choices=["remove", "truncate", "signature-only"])

    # reload
    p_reload = sub.add_parser("reload", help="Treat current session and auto-resume after exit")
    p_reload.add_argument("--cwd", help="Working directory (default: current)")
    p_reload.add_argument("-rx", help="Prescription: gentle, standard, aggressive (default: standard)")
    p_reload.add_argument("--thinking-mode", choices=["remove", "truncate", "signature-only"])

    # checkpoint
    p_cp = sub.add_parser("checkpoint", help="Save team/agent state from the current session (no pruning)")
    p_cp.add_argument("--cwd", help="Working directory (default: current)")
    p_cp.add_argument("--show", action="store_true", help="Print the team state after saving")

    # guard
    p_guard = sub.add_parser("guard", help="Background sentinel — auto-prune before compaction triggers")
    p_guard.add_argument("--cwd", help="Working directory (default: current)")
    p_guard.add_argument("-rx", help="Prescription to apply (default: standard)")
    p_guard.add_argument("--threshold", type=float, default=50.0, help="Hard threshold in MB — full prune + reload (default: 50)")
    p_guard.add_argument("--soft-threshold", type=float, default=None, help="Soft threshold in MB — gentle prune, no reload (default: 60%% of --threshold)")
    p_guard.add_argument("--interval", type=int, default=30, help="Check interval in seconds (default: 30)")
    p_guard.add_argument("--no-reload", action="store_true", help="Prune without auto-reload at hard threshold")
    p_guard.add_argument("--no-reactive", action="store_true", help="Disable reactive overflow recovery (kqueue/polling watcher)")
    p_guard.add_argument("--daemon", action="store_true", help="Run in background (PID file prevents double-starts)")

    # init
    p_init = sub.add_parser("init", help="Auto-wire hooks and slash command into this project")
    p_init.add_argument("--cwd", help="Project directory (default: current)")
    p_init.add_argument("--no-slash-command", action="store_true", help="Skip installing /cozempic slash command")

    # doctor
    p_doctor = sub.add_parser("doctor", help="Check for known Claude Code issues and fix them")
    p_doctor.add_argument("--fix", action="store_true", help="Auto-fix issues where possible")

    # formulary
    sub.add_parser("formulary", help="Show all strategies & prescriptions")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "list": cmd_list,
        "current": cmd_current,
        "diagnose": cmd_diagnose,
        "treat": cmd_treat,
        "strategy": cmd_strategy,
        "reload": cmd_reload,
        "checkpoint": cmd_checkpoint,
        "guard": cmd_guard,
        "init": cmd_init,
        "doctor": cmd_doctor,
        "formulary": cmd_formulary,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
