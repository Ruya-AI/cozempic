"""CLI interface for Cozempic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .diagnosis import diagnose_session
from .executor import execute_actions, run_prescription
from .registry import PRESCRIPTIONS, STRATEGIES
from .session import find_current_session, find_sessions, load_messages, resolve_session, save_messages
from .types import PrescriptionResult, StrategyResult

# Ensure all strategies are registered
import cozempic.strategies  # noqa: F401


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
    sess = find_current_session(cwd)
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
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command")

    session_help = "Session ID, UUID prefix, path, or 'current' for auto-detect"

    # list
    p_list = sub.add_parser("list", help="List sessions with sizes")
    p_list.add_argument("--project", help="Filter by project name")

    # current
    p_current = sub.add_parser("current", help="Show current session for this project")
    p_current.add_argument("--cwd", help="Working directory (default: current)")
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
        "formulary": cmd_formulary,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
