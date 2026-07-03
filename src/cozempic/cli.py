"""CLI interface for Cozempic."""

from __future__ import annotations

import argparse
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .diagnosis import diagnose_session
from .doctor import run_doctor
from .executor import execute_actions, run_prescription
from .guard import checkpoint_team, start_guard, start_guard_daemon
from .helpers import (
    is_ssh_session, shell_quote,
    compile_protect_patterns, tag_pattern_matches, strip_pattern_tags,
)
from .init import run_init
from .recap import save_recap
from .registry import PRESCRIPTIONS, STRATEGIES
from .safety import PruneValidationError
from .session import _PruneLock, PruneConflictError, PruneLockError, find_claude_pid, find_current_session, find_sessions, get_session_cwd, load_messages, load_messages_and_snapshot, project_slug_to_path, resolve_session, save_messages, snapshot_session
from .tokens import estimate_session_tokens, quick_token_estimate, calibrate_ratio
from .types import PrescriptionResult, StrategyResult

# Ensure all strategies are registered
import cozempic.strategies  # noqa: F401


# ─── argparse type= validators ────────────────────────────────────────────
# Kept inline (not a separate module) because they're tiny and argparse-specific.
# Raise argparse.ArgumentTypeError so the error surfaces as a standard
# argparse validation error — user sees "error: argument --threshold:
# must be positive, got -1" with correct exit code 2.


def _positive_int(val: str) -> int:
    """argparse type= for strictly-positive ints. Used for --interval,
    --threshold-tokens, and --soft-threshold-tokens where zero is
    nonsensical (spin loops, always-trigger thresholds)."""
    try:
        n = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{val!r} is not a valid integer")
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {n}")
    if n > 10**12:  # reject an absurd value (e.g. 10**400) that overflows downstream math
        raise argparse.ArgumentTypeError(f"is unreasonably large, got {n}")
    return n


def _positive_float(val: str) -> float:
    """argparse type= for strictly-positive floats. Used for --threshold
    and --soft-threshold (MB thresholds). Rejects NaN and infinity."""
    try:
        f = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{val!r} is not a valid number")
    if not math.isfinite(f):
        raise argparse.ArgumentTypeError(f"must be a finite number, got {f}")
    if f <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {f}")
    return f


def _apply_token_env_overrides(args) -> None:
    """Mirror --context-window / --system-overhead-tokens CLI flags into the env
    vars the token layer reads.

    Uses `is not None` rather than truthiness so an explicit
    --system-overhead-tokens 0 (a legitimate "no overhead" value) is honored
    rather than silently dropped to the 21_000 default. Zero is falsy in Python
    but it is a valid override — e.g. a session with no CLAUDE.md, no MCP
    servers, and no rules files really does have zero system overhead.

    Replaces the duplicate truthiness-gated blocks that existed in both
    cmd_guard() and main(), eliminating the DRY violation and the 0-drop bug
    in a single fix.
    """
    cw = getattr(args, "context_window", None)
    if cw is not None:
        os.environ["COZEMPIC_CONTEXT_WINDOW"] = str(cw)
    soh = getattr(args, "system_overhead_tokens", None)
    if soh is not None:
        os.environ["COZEMPIC_SYSTEM_OVERHEAD_TOKENS"] = str(soh)


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


def fmt_tokens(t: int) -> str:
    if t < 1000:
        return f"{t}"
    elif t < 1_000_000:
        return f"{t / 1000:.1f}K"
    else:
        return f"{t / 1_000_000:.2f}M"


def fmt_context_bar(pct: float, width: int = 20) -> str:
    filled = int(round(pct / 100 * width))
    filled = max(0, min(filled, width))
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}] {pct:.0f}%"


def print_diagnosis(diag: dict, path: Path):
    total = diag["total_bytes"]
    print(f"\n  Patient: {path.stem}")
    print(f"  Weight:  {fmt_bytes(total)} ({diag['total_messages']} messages)")

    te = diag.get("token_estimate")
    if te:
        confidence = f", {te.confidence}" if te.method == "heuristic" else ""
        model_str = f"  Model:   {te.model}" if te.model else ""
        window_str = f"{fmt_tokens(te.context_window)}" if te.context_window != 200_000 else "200K"
        print(f"  Tokens:  {fmt_tokens(te.total)} ({te.method}{confidence})")
        print(f"  Context: {fmt_context_bar(te.context_pct)} of {window_str}")
        if model_str:
            print(model_str)
    print()

    print("  Vital Signs:")
    print(f"    Progress ticks:     {diag['progress_count']:>6}")
    print(f"    File history snaps: {diag['file_history_count']:>6}")
    print(f"    System reminders:   {diag['reminder_count']:>6}")
    print(f"    Thinking content:   {fmt_bytes(diag['thinking_bytes']):>10} ({fmt_pct(diag['thinking_bytes'], total)})")
    print(f"    Signatures:         {fmt_bytes(diag['signature_bytes']):>10} ({fmt_pct(diag['signature_bytes'], total)})")
    print(f"    Tool results:       {fmt_bytes(diag['tool_result_bytes']):>10} ({fmt_pct(diag['tool_result_bytes'], total)})")

    cache = diag.get("cache_stats")
    if cache:
        print(f"    Cache hit rate:     {cache['cache_hit_rate']:>5}%  ({cache['cache_read_tokens']:,} read / {cache['cache_total_tokens']:,} total)")
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
    """Print a single strategy result — only called for strategies that did something."""
    saved = sum(a.original_bytes - a.pruned_bytes for a in sr.actions) if sr.actions else 0
    affected = sr.messages_removed + sr.messages_replaced
    print(f"    {sr.strategy_name:<28} {fmt_bytes(saved):>8}  {affected:>4} msgs")


def print_prescription_result(pr: PrescriptionResult):
    saved = pr.original_total_bytes - pr.final_total_bytes

    print(f"\n  Cozempic — {pr.prescription_name} prescription\n")

    if pr.original_tokens is not None and pr.final_tokens is not None:
        tok_saved = pr.original_tokens - pr.final_tokens
        from .tokens import DEFAULT_CONTEXT_WINDOW
        context_window = pr.context_window or DEFAULT_CONTEXT_WINDOW
        after_pct = round(pr.final_tokens / context_window * 100, 1)
        window_str = fmt_tokens(context_window)
        print(f"  Before   {fmt_tokens(pr.original_tokens):>8} tokens  {fmt_bytes(pr.original_total_bytes):>8}  {pr.original_message_count:,} messages")
        print(f"  After    {fmt_tokens(pr.final_tokens):>8} tokens  {fmt_bytes(pr.final_total_bytes):>8}  {pr.final_message_count:,} messages")
        if tok_saved >= 0 and pr.original_tokens > 0:
            tok_pct = f"{tok_saved / pr.original_tokens * 100:.1f}%"
            print(f"  Saved    {fmt_tokens(tok_saved):>8} tokens ({tok_pct})  {fmt_bytes(saved)} freed")
        else:
            # Tokens appear to have GROWN despite a byte reduction. This is not a
            # real increase: metadata-strip removes the `usage` frames the exact
            # count anchors on, so the post-prune exact count re-anchors on an
            # earlier turn with a different cache_read — making the token delta
            # non-comparable (#105: "-401.6% freed"). Report the reliable byte
            # savings and flag the token delta as unavailable rather than print a
            # nonsensical negative/over-100% value.
            byte_pct = fmt_pct(saved, pr.original_total_bytes)
            print(f"  Saved    {fmt_bytes(saved):>8} ({byte_pct})  — exact token delta n/a (usage metadata pruned; count re-anchored)")
        print(f"  Context  {fmt_context_bar(after_pct)} of {window_str}")
    else:
        byte_pct = fmt_pct(saved, pr.original_total_bytes)
        print(f"  Before   {fmt_bytes(pr.original_total_bytes):>8}  {pr.original_message_count:,} messages")
        print(f"  After    {fmt_bytes(pr.final_total_bytes):>8}  {pr.final_message_count:,} messages")
        print(f"  Saved    {fmt_bytes(saved):>8} ({byte_pct})")

    # Only show strategies that actually did something
    active = [sr for sr in pr.strategy_results if sr.actions]
    if active:
        print(f"\n  What changed:")
        for sr in sorted(active, key=lambda s: sum(a.original_bytes - a.pruned_bytes for a in s.actions), reverse=True):
            print_strategy_result(sr, pr.original_total_bytes)
    print()


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_list(args):
    sessions = find_sessions(args.project)
    if not sessions:
        print("No sessions found.")
        return

    print(f"\n  {'Session ID':<40} {'Size':>10} {'Tokens':>8} {'Messages':>8} {'Modified':<20} Project")
    print(f"  {'─' * 40} {'─' * 10} {'─' * 8} {'─' * 8} {'─' * 20} {'─' * 30}")

    for sess in sorted(sessions, key=lambda s: s["size"], reverse=True):
        sid = sess["session_id"]
        if len(sid) > 36:
            sid = sid[:33] + "..."
        tok = quick_token_estimate(sess["path"])
        tok_str = fmt_tokens(tok) if tok is not None else "—"
        print(
            f"  {sid:<40} {fmt_bytes(sess['size']):>10} {tok_str:>8} {sess['lines']:>8}"
            f" {sess['mtime'].strftime('%Y-%m-%d %H:%M'):<20} {sess['project'][-40:]}"
        )
    print()

    total = sum(s["size"] for s in sessions)
    print(f"  Total: {len(sessions)} sessions, {fmt_bytes(total)}")
    print()


def _auto_heal_session(path):
    """#147: silently repair a torn trailing line on an IDLE (dead) session so a
    user troubleshooting a 'Failed to resume' just by running a cozempic command
    auto-recovers — without ever racing a live write (gated on mtime staleness).
    Best-effort; never raises."""
    try:
        from .session import auto_repair_unresumable
        if path and auto_repair_unresumable(path):
            print(f"  Repaired a torn trailing line — this session can resume again (.torn.bak kept). (#147)")
    except Exception:
        pass


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

    _auto_heal_session(sess["path"])
    from .tokens import detect_context_window, detect_model
    messages_for_model = load_messages(sess["path"])
    context_window = detect_context_window(messages_for_model)
    model = detect_model(messages_for_model)

    tok = quick_token_estimate(sess["path"])
    if tok is not None:
        pct = round(tok / context_window * 100, 1)
        window_str = fmt_tokens(context_window)
        print(f"    Tokens:  {fmt_tokens(tok)} {fmt_context_bar(pct)} of {window_str}")
    if model:
        print(f"    Model:   {model}")

    print(f"    Project: {sess['project']}")
    print(f"    Path:    {sess['path']}")
    print(f"    Modified: {sess['mtime'].strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if args.diagnose:
        messages = load_messages(sess["path"])
        diag = diagnose_session(messages)
        print_diagnosis(diag, sess["path"])

        print("  Estimated Savings by Prescription:")
        # Resolve floor_config once (L-3: avoid per-prescription disk reads in the loop)
        _floor_cfg = load_config().floor
        for rx_name, strategy_names in PRESCRIPTIONS.items():
            try:
                new_msgs, _ = run_prescription(messages, strategy_names, {},
                                               floor_config=_floor_cfg)
            except PruneValidationError as ve:
                # Dry-run estimation: a structural validation failure means the
                # session is already malformed or the strategy would make it
                # unreplayable. Report the check that failed — do not crash.
                check = ve.evidence.get("failed_check", "?")
                print(f"    {rx_name:<15} would be invalid ({check}): {ve.reason}")
                continue
            final_bytes = sum(b for _, _, b in new_msgs)
            total_saved = diag["total_bytes"] - final_bytes
            pct = fmt_pct(total_saved, diag["total_bytes"])
            print(f"    {rx_name:<15} ~{fmt_bytes(total_saved):>10} ({pct})")
        print()


def cmd_diagnose(args):
    path = resolve_session(args.session, getattr(args, "project", None))
    _auto_heal_session(path)
    messages = load_messages(path)
    diag = diagnose_session(messages)
    print_diagnosis(diag, path)

    print("  Estimated Savings by Prescription:")
    # Resolve floor_config once (L-3: avoid per-prescription disk reads in the loop)
    _floor_cfg = load_config().floor
    for rx_name, strategy_names in PRESCRIPTIONS.items():
        try:
            new_msgs, _ = run_prescription(messages, strategy_names, {},
                                           floor_config=_floor_cfg)
        except PruneValidationError as ve:
            # Dry-run estimation: report structural failure instead of crashing.
            check = ve.evidence.get("failed_check", "?")
            print(f"    {rx_name:<15} would be invalid ({check}): {ve.reason}")
            continue
        final_bytes = sum(b for _, _, b in new_msgs)
        total_saved = diag["total_bytes"] - final_bytes
        pct = fmt_pct(total_saved, diag["total_bytes"])
        print(f"    {rx_name:<15} ~{fmt_bytes(total_saved):>10} ({pct})")
    print()


def _compile_protect_patterns_or_exit(args):
    """Compile --protect-pattern regexes from args, or exit(2) with a clean error.
    Returns the compiled list, or None when the flag is unused (#122, @eggrollofchaos)."""
    raw = getattr(args, "protect_pattern", None)
    if not raw:
        return None
    try:
        return compile_protect_patterns(raw)
    except ValueError as e:
        print(f"  Error: {e}", file=sys.stderr)
        sys.exit(2)


_TIER_NAMES = {"gentle", "standard", "aggressive"}


def _emit_prune_receipt(path, pr, *, source, outcome, session_id=None, cwd=None, defer_reason=None):
    """Fire-and-forget a prune receipt (D1 of the dashboard path).

    Doubly exception-isolated (here AND inside emit_receipt) so a receipt can
    never break or defer the prune it records. Called on every outcome —
    committed and deferred.
    """
    try:
        from .metrics import ClaudeMetricsAdapter, TriggerInfo, ValidationInfo
        from .receipts import emit_receipt

        tiers = {
            sr.strategy_name: STRATEGIES[sr.strategy_name].tier
            for sr in pr.strategy_results
            if sr.strategy_name in STRATEGIES
        }
        deferred = outcome != "committed"
        tier = pr.prescription_name if pr.prescription_name in _TIER_NAMES else "custom"
        emit_receipt(
            pr,
            adapter=ClaudeMetricsAdapter(),
            session_id=session_id or (path.stem if path else None),
            transcript_path=str(path) if path else None,
            cwd=cwd,
            trigger=TriggerInfo(
                source=source, tier=tier, prescription=pr.prescription_name,
            ),
            outcome=outcome,
            validation=ValidationInfo(
                passed=not deferred, deferred=deferred, defer_reason=defer_reason,
            ),
            strategy_tiers=tiers,
        )
    except Exception:
        pass


# Prescription ladder, lightest → heaviest. gentle ⊂ standard ⊂ aggressive
# (strict strategy subsets), so a lighter prescription removes strictly less and
# is monotonically MORE likely to pass post-prune validation (e.g. the C3
# conversation-survival check). That monotonicity is what makes graceful
# fall-back sound: try the requested tier, then progressively lighter ones.
_RX_LADDER = ["gentle", "standard", "aggressive"]


def _fallback_order(rx_name):
    """Requested prescription first, then each lighter one (heaviest-safe-first).

    e.g. 'aggressive' → ['aggressive', 'standard', 'gentle']; 'gentle' → ['gentle'].
    A custom (non-ladder) prescription has no lighter sibling → just itself.
    """
    if rx_name not in _RX_LADDER:
        return [rx_name]
    i = _RX_LADDER.index(rx_name)
    return [_RX_LADDER[j] for j in range(i, -1, -1)]


def run_prescription_with_fallback(messages, rx_name, config, *, strict=False):
    """Run ``rx_name``; if it fails post-prune validation (e.g. C3 would wipe the
    conversation), fall back to the heaviest LIGHTER prescription that validates.

    Returns a dict: ``{messages, results, applied_rx, requested_rx, fell_back,
    error}``. On total failure (even the lightest prescription would wipe the
    conversation — a genuinely too-thin / freshly-compacted session), ``messages``
    is None and ``error`` is the last PruneValidationError. ``strict=True`` disables
    fall-back (only the requested tier is tried) for deterministic scripts/CI.
    """
    order = [rx_name] if strict else _fallback_order(rx_name)
    last_err = None
    for rx in order:
        names = PRESCRIPTIONS.get(rx)
        if not names:
            continue
        try:
            new_messages, results = run_prescription(messages, names, config)
            return {"messages": new_messages, "results": results, "applied_rx": rx,
                    "requested_rx": rx_name, "fell_back": rx != rx_name, "error": None}
        except PruneValidationError as exc:
            last_err = exc
    return {"messages": None, "results": None, "applied_rx": None,
            "requested_rx": rx_name, "fell_back": False, "error": last_err}


def cmd_treat(args):
    path = resolve_session(args.session, getattr(args, "project", None), strict=getattr(args, "execute", False))
    # Take snapshot BEFORE load so append-conflict detection in save_messages
    # can correctly identify if Claude wrote new lines mid-prune. Only needed
    # for execute path but cheap to compute always. Read once so the snapshot and
    # messages come from identical bytes (no TOCTOU append-duplication).
    messages, _snap = load_messages_and_snapshot(path)
    snapshot = _snap if getattr(args, "execute", False) else None
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

    # Token estimate before pruning — also capture calibrated ratio before metadata-strip
    pre_te = estimate_session_tokens(messages)
    pre_ratio = calibrate_ratio(messages)

    # --protect-pattern: tag matching messages so the strategies spare them (#122).
    protect_patterns = _compile_protect_patterns_or_exit(args)
    if protect_patterns:
        tag_pattern_matches(messages, protect_patterns)

    try:
        _fb = run_prescription_with_fallback(messages, rx_name, config,
                                             strict=getattr(args, "strict", False))
        if _fb["error"] is not None:
            ve = _fb["error"]
            check = ve.evidence.get("failed_check", "?")
            print(f"  This session can't be pruned safely — every prescription would "
                  f"remove the conversation ({check}: {ve.reason}).", file=sys.stderr)
            print("  Nothing to do (this usually means a very thin / freshly-compacted "
                  "session). The session file was not modified.", file=sys.stderr)
            return
        new_messages, strategy_results = _fb["messages"], _fb["results"]
        if _fb["fell_back"]:
            print(f"  Note: '{_fb['requested_rx']}' would have wiped the conversation here "
                  f"— applied the safe maximum '{_fb['applied_rx']}' instead.")
            rx_name = _fb["applied_rx"]  # reflect what was actually applied downstream
            strategy_names = PRESCRIPTIONS[rx_name]
    finally:
        # Strip the transient tag from EVERY message (crash-safe), so it can never
        # persist into the saved session.
        if protect_patterns:
            strip_pattern_tags(messages)

    if protect_patterns:
        strip_pattern_tags(new_messages)  # belt-and-suspenders if a strategy copied dicts
    final_bytes = sum(b for _, _, b in new_messages)
    final_count = len(new_messages)

    # Token estimate after pruning — pass pre-calibrated ratio so metadata-strip
    # doesn't corrupt the post-treatment count by falling back to raw default
    post_te = estimate_session_tokens(new_messages, pre_calibrated_ratio=pre_ratio)

    pr = PrescriptionResult(
        prescription_name=rx_name,
        strategy_results=strategy_results,
        original_total_bytes=original_bytes,
        final_total_bytes=final_bytes,
        original_message_count=original_count,
        final_message_count=final_count,
        original_tokens=pre_te.total,
        final_tokens=post_te.total,
        token_method=pre_te.method,
        model=pre_te.model,
        context_window=pre_te.context_window,
    )

    print_prescription_result(pr)

    if pre_te.method == "exact" and post_te.method == "heuristic":
        print("  Note: exact usage data was stripped — post-treatment token count is estimated.")

    if args.execute:
        # Check for active background tasks before writing
        from .helpers import find_active_background_tasks
        active_tasks = find_active_background_tasks(messages)
        if active_tasks:
            print(f"  WARNING: {len(active_tasks)} background task(s) in progress:")
            for t in active_tasks[:5]:
                print(f"    - {t['description'][:70] or t['tool_use_id'][:20]}")
            print()
            try:
                answer = input("  Writing may interrupt these tasks. Continue? [y/N] ")
            except EOFError:
                answer = "n"
            if answer.lower() != "y":
                print("  Aborted — wait for tasks to complete or pass --force to override.")
                return

        # Acquire per-session prune lock + pass snapshot for append-conflict detection.
        # Prevents corruption when the guard daemon is mid-prune on the same session.
        try:
            with _PruneLock(path):
                backup = save_messages(path, new_messages, create_backup=True, snapshot=snapshot)
        except PruneLockError:
            _emit_prune_receipt(path, pr, source="manual", outcome="deferred",
                                cwd=os.getcwd(), defer_reason="prune_lock")
            print("  Aborted: another prune cycle (guard daemon) is active. Try again in a few seconds.", file=sys.stderr)
            sys.exit(2)
        except PruneConflictError as exc:
            _emit_prune_receipt(path, pr, source="manual", outcome="deferred",
                                cwd=os.getcwd(), defer_reason="session_changed")
            print(f"  Aborted: session changed mid-prune (Claude wrote new lines). {exc}", file=sys.stderr)
            sys.exit(3)
        _emit_prune_receipt(path, pr, source="manual", outcome="committed", cwd=os.getcwd())
        print(f"  Applied to {path}")
        if backup:
            print(f"  Backup: {backup}")
        print(f"  Final size: {fmt_bytes(final_bytes)}")

        # Track and display lifetime savings
        if pr.original_tokens and pr.final_tokens:
            from .helpers import record_savings, get_savings_line, get_msg_type
            turn_count = sum(1 for _, m, _ in messages
                           if get_msg_type(m) == "user"
                           and isinstance(m.get("message", {}).get("content", ""), str))
            record_savings(
                pr.original_tokens - pr.final_tokens,
                total_tokens=pr.original_tokens,
                turn_count=turn_count,
                session_id=path.stem if path else None,
            )
            savings = get_savings_line()
            if savings:
                print(f"  {savings}")
    else:
        # Show active tasks in dry run too
        from .helpers import find_active_background_tasks
        active_tasks = find_active_background_tasks(messages)
        if active_tasks:
            print(f"  NOTE: {len(active_tasks)} background task(s) active — executing would interrupt them.")
        print("  Dry run — pass --execute to apply.")
    print()


def cmd_strategy(args):
    path = resolve_session(args.session, getattr(args, "project", None), strict=getattr(args, "execute", False))
    # Read once (snapshot + messages from identical bytes) for append-conflict
    # detection on the execute path without a TOCTOU append-duplication window.
    messages, _snap = load_messages_and_snapshot(path)
    snapshot = _snap if getattr(args, "execute", False) else None

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
        # Acquire per-session prune lock + pass snapshot for append-conflict
        # detection. Same protection as cmd_treat.
        try:
            with _PruneLock(path):
                backup = save_messages(path, new_messages, create_backup=True, snapshot=snapshot)
        except PruneLockError:
            print("  Aborted: another prune cycle (guard daemon) is active. Try again in a few seconds.", file=sys.stderr)
            sys.exit(2)
        except PruneConflictError as exc:
            print(f"  Aborted: session changed mid-prune (Claude wrote new lines). {exc}", file=sys.stderr)
            sys.exit(3)
        final_bytes = sum(b for _, _, b in new_messages)
        print(f"  Applied. Final size: {fmt_bytes(final_bytes)}")
        if backup:
            print(f"  Backup: {backup}")
    else:
        print("  Dry run — pass --execute to apply.")
    print()


def cmd_reload(args):
    """Treat the current session, then spawn a watcher that auto-resumes Claude.

    Wave 2: acquires a single-flight `_ReloadLock` to prevent two reload
    pipelines from running concurrently (manual `cozempic reload` + the
    guard daemon's auto-fire at 55%/80% thresholds, or `OverflowRecovery`).
    Without this lock, both spawn watchers, both osascript new terminals,
    both try to `claude --resume` the same session → session conflict.
    """
    from .reload_lock import (
        _ReloadLock, ReloadLockHeld, INIT_CLI_RELOAD, acquire_with_wait,
    )

    cwd = args.cwd or os.getcwd()
    sess = None
    explicit_session = getattr(args, "session", None)
    if explicit_session:
        # User-provided session — path, full UUID, or UUID prefix.
        try:
            path = resolve_session(explicit_session, strict=True)
        except SystemExit:
            raise
        session_id = path.stem
        sess = {"session_id": session_id, "path": path, "project": path.parent.name}
    else:
        sess = find_current_session(cwd, strict=True)
    if not sess:
        print("Could not detect current session.", file=sys.stderr)
        print("Cannot determine session unambiguously — pass one explicitly:", file=sys.stderr)
        print("  cozempic reload --session <uuid-or-path> -rx <prescription>", file=sys.stderr)
        print("Use 'cozempic list' to find the session ID.", file=sys.stderr)
        sys.exit(1)

    # User is taking manual control of the reload — clear any armed-reload
    # sentinel the daemon left so its projected-%/warned state can't go stale and
    # the next arm starts fresh (1.8.22 E; previously clear_armed was never wired).
    try:
        from .guard import clear_armed
        clear_armed(sess["session_id"], sess.get("path"))
    except Exception:
        pass

    # Acquire the single-flight reload lock. Fail-fast by default; opt-in
    # queueing with --wait[=SECS]. The lock is held through treatment +
    # watcher spawn; released when this function returns (the watcher
    # itself is a separate bash process that doesn't need to hold the lock
    # because the actual session JSONL is protected by _PruneLock during
    # save_messages).
    _wait_sec = getattr(args, "wait", None)
    try:
        if _wait_sec is not None:
            _reload_lock = acquire_with_wait(
                sess["session_id"], INIT_CLI_RELOAD, wait_seconds=float(_wait_sec),
            )
        else:
            _reload_lock = _ReloadLock(sess["session_id"], initiator=INIT_CLI_RELOAD)
            _reload_lock.__enter__()
    except ReloadLockHeld as exc:
        print(f"  Aborted: {exc}", file=sys.stderr)
        if exc.wedged:
            # Resolve the actual lock path so the user knows exactly which
            # file to remove. Generic glob hints could wipe unrelated
            # session locks; specificity protects other in-flight sessions.
            from .reload_lock import _lock_path_for, WEDGE_TTL_SECONDS
            lock_path = _lock_path_for(sess["session_id"])
            print(
                f"  The holding process appears wedged (>{WEDGE_TTL_SECONDS}s with no progress).",
                file=sys.stderr,
            )
            print(
                f"  If you're sure it's stuck, clear with: rm {lock_path}",
                file=sys.stderr,
            )
        else:
            print(
                f"  Retry in a few seconds, or pass --wait to queue.",
                file=sys.stderr,
            )
        sys.exit(2)

    # From here on, wrap everything in try/finally so the lock is ALWAYS
    # released on any exit path (return, exception, sys.exit). The lock
    # is held through treat + watcher-spawn; released when this function
    # returns. The bash watcher then polls claude_pid unguarded.
    #
    # KNOWN LIMITATION (late-fire): if a second pipeline (e.g. guard's
    # auto-fire) starts within the sub-second window AFTER we release
    # the lock but BEFORE the user's Claude actually exits, it can
    # acquire the lock and spawn its own watcher. Real-world impact is
    # small because the guard's prune-cycle timer interval (~10s) is
    # much longer than reload spawn latency, making collision rare.
    # Full fix would require the bash watcher to participate in the lock
    # protocol — deferred to a future wave if observed in production.
    try:

        # Resolve the project root directory for the resume cd target.
        # The critical invariant: `claude --resume <id>` must be run from the
        # SAME CWD where the session was originally created, because Claude Code
        # resolves sessions by CWD → project-slug mapping. cd'ing into a
        # subdirectory produces a different slug → "No conversation found."
        #
        # Priority:
        #   1. Sidecar CWD (exact path, recorded by the guard daemon — most
        #      reliable when available)
        #   2. Slug reversal from the session's project directory name. The
        #      JSONL lives at ~/.claude/projects/<slug>/<uuid>.jsonl — the slug
        #      IS the authoritative project identifier. Reversing it gives the
        #      original CWD. This MUST take priority over os.getcwd() because
        #      the user (or Claude) may have cd'd into a subdirectory during
        #      the session. (Known limitation: hyphens in the original path are
        #      ambiguous with separator hyphens in the slug.)
        #   3. os.getcwd() as last resort — only when sidecar is empty AND
        #      slug reversal produces a non-existent directory (hyphen ambiguity).
        # First try: slug reversal from the session's actual project directory.
        # This is the most reliable source because the JSONL path directly
        # encodes where the session was created. It only fails when the original
        # path contained hyphens (ambiguous with the slug separator).
        slug_cwd = project_slug_to_path(sess["project"])
        if os.path.isdir(slug_cwd):
            cwd = slug_cwd
        else:
            # Slug reversal failed (hyphens in path) — try sidecar
            sidecar_cwd = get_session_cwd(sess["session_id"])
            if sidecar_cwd and os.path.isdir(sidecar_cwd):
                cwd = sidecar_cwd
            # else: cwd stays as os.getcwd() — last resort

        rx_name = args.rx or "standard"
        if rx_name not in PRESCRIPTIONS:
            print(f"Error: Unknown prescription '{rx_name}'. Options: {', '.join(PRESCRIPTIONS)}", file=sys.stderr)
            sys.exit(1)

        # Step 1: Apply treatment
        path = sess["path"]
        # Snapshot BEFORE load so append-conflict detection works (Claude may write
        # mid-prune; we need the file state at this exact instant for diff classification).
        messages, snapshot = load_messages_and_snapshot(path)
        strategy_names = PRESCRIPTIONS[rx_name]
        config = {}
        if args.thinking_mode:
            config["thinking_mode"] = args.thinking_mode

        original_bytes = sum(b for _, _, b in messages)
        original_count = len(messages)

        # Token estimate before pruning — capture calibrated ratio before metadata-strip
        pre_te = estimate_session_tokens(messages)
        pre_ratio = calibrate_ratio(messages)

        # --protect-pattern: tag matching messages so the strategies spare them (#122).
        protect_patterns = _compile_protect_patterns_or_exit(args)
        if protect_patterns:
            tag_pattern_matches(messages, protect_patterns)

        try:
            _fb = run_prescription_with_fallback(messages, rx_name, config,
                                                 strict=getattr(args, "strict", False))
            if _fb["error"] is not None:
                ve = _fb["error"]
                check = ve.evidence.get("failed_check", "?")
                print(f"  This session can't be pruned safely — every prescription would "
                      f"remove the conversation ({check}: {ve.reason}).", file=sys.stderr)
                print("  Reload skipped — nothing to do (very thin / freshly-compacted "
                      "session). The session was not modified or terminated.", file=sys.stderr)
                return
            new_messages, strategy_results = _fb["messages"], _fb["results"]
            if _fb["fell_back"]:
                print(f"  Note: '{_fb['requested_rx']}' would have wiped the conversation here "
                      f"— applied the safe maximum '{_fb['applied_rx']}' instead.")
                rx_name = _fb["applied_rx"]
                strategy_names = PRESCRIPTIONS[rx_name]
        finally:
            if protect_patterns:
                strip_pattern_tags(messages)
        if protect_patterns:
            strip_pattern_tags(new_messages)

        final_bytes = sum(b for _, _, b in new_messages)
        final_count = len(new_messages)

        # Token estimate after pruning — pass pre-calibrated ratio
        post_te = estimate_session_tokens(new_messages, pre_calibrated_ratio=pre_ratio)

        pr = PrescriptionResult(
            prescription_name=rx_name,
            strategy_results=strategy_results,
            original_total_bytes=original_bytes,
            final_total_bytes=final_bytes,
            original_message_count=original_count,
            final_message_count=final_count,
            original_tokens=pre_te.total,
            final_tokens=post_te.total,
            token_method=pre_te.method,
            model=pre_te.model,
            context_window=pre_te.context_window,
        )
        print_prescription_result(pr)

        # Acquire per-session prune lock + pass snapshot. Without this, a concurrent
        # guard daemon prune cycle would silently overwrite our pruned output (or
        # vice versa) — exactly the data-loss bug observed in production where
        # cmd_reload raced guard_prune_cycle's auto-fire at the 55% threshold.
        try:
            with _PruneLock(path):
                backup = save_messages(path, new_messages, create_backup=True, snapshot=snapshot)
        except PruneLockError:
            _emit_prune_receipt(path, pr, source="manual", outcome="deferred",
                                session_id=sess["session_id"], cwd=cwd, defer_reason="prune_lock")
            print("  Aborted: another prune cycle (guard daemon) is active. Try again in a few seconds.", file=sys.stderr)
            sys.exit(2)
        except PruneConflictError as exc:
            _emit_prune_receipt(path, pr, source="manual", outcome="deferred",
                                session_id=sess["session_id"], cwd=cwd, defer_reason="session_changed")
            print(f"  Aborted: session changed mid-prune (Claude wrote new lines). {exc}", file=sys.stderr)
            sys.exit(3)
        _emit_prune_receipt(path, pr, source="manual", outcome="committed",
                            session_id=sess["session_id"], cwd=cwd)
        print(f"  Applied to {path}")
        if backup:
            print(f"  Backup: {backup}")
        print(f"  Final size: {fmt_bytes(final_bytes)}")

        # Track lifetime savings
        if pre_te.total and post_te.total:
            from .helpers import record_savings, get_savings_line, get_msg_type
            turn_count = sum(1 for _, m, _ in messages
                           if get_msg_type(m) == "user"
                           and isinstance(m.get("message", {}).get("content", ""), str))
            record_savings(
                pre_te.total - post_te.total,
                total_tokens=pre_te.total,
                turn_count=turn_count,
                session_id=sess["session_id"],
            )
            savings = get_savings_line()
            if savings:
                print(f"  {savings}")
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

        # Auto-send /exit via the best available method
        from .guard import _detect_terminal_env
        term_env = _detect_terminal_env()

        if term_env == "tmux":
            pane = os.environ.get("TMUX_PANE", "")
            import subprocess as sp
            sp.run(["tmux", "send-keys", *(["-t", pane] if pane else []), "/exit", "Enter"],
                   capture_output=True, timeout=5)
            print(f"  Resuming with optimized context...")
        elif term_env == "screen":
            screen_session = os.environ.get("STY", "")
            import subprocess as sp
            sp.run(["screen", "-S", screen_session, "-X", "stuff", "/exit\n"],
                   capture_output=True, timeout=5)
            print(f"  Resuming with optimized context...")
        else:
            print(f"  Type /exit to resume with optimized context.")
        print()
    finally:
        try:
            _reload_lock.__exit__(None, None, None)
        except Exception:
            pass


def _spawn_watcher(claude_pid: int, project_dir: str, recap_path: Path | None = None, session_id: str | None = None):
    """Spawn a detached background process that waits for Claude to exit, then resumes."""
    from .guard import _detect_claude_flags
    resume_flag = f"--resume {session_id}" if session_id else "--resume"
    original_flags = _detect_claude_flags(claude_pid)
    if original_flags:
        resume_flag = f"{original_flags} {resume_flag}"

    # SSH sessions can't open GUI terminals — tell the user to resume manually
    if is_ssh_session():
        print(f"  SSH session detected — auto-resume is not available over SSH.")
        print(f"  After exiting, resume manually:")
        print(f"    cd {project_dir} && claude {resume_flag}")
        return

    system = platform.system()

    # Build the command sequence: show recap, then launch claude --resume
    recap_cmd = ""
    if recap_path and recap_path.exists():
        recap_cmd = f"cat {shell_quote(str(recap_path))}; echo; "

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


def cmd_post_compact(args):
    """Output recovery context after native compaction. Designed for PostCompact hook.

    Reads team-checkpoint.md saved by PreCompact and outputs it to stdout,
    where Claude Code picks it up as hook feedback. Silent when no checkpoint
    exists (solo sessions without agent teams).
    """
    from .team import read_team_checkpoint

    cwd = args.cwd or os.getcwd()

    # Resolve the project dir for this session.
    # strict=True: refuse Strategy 4 (global most-recent fallback) — wrong-project
    # sessions produce wrong-project checkpoints.
    # On None: Path(cwd) points to the local project dir (project-correct; may have
    # no checkpoint file → read_team_checkpoint returns None → silent, which is safe).
    sess = find_current_session(cwd, strict=True)
    project_dir = Path(sess["path"]).parent if sess else Path(cwd)

    # include_global=False: the global ~/.claude/team-checkpoint.md is a cross-project
    # read vector — it holds the last-written checkpoint regardless of project.
    content = read_team_checkpoint(project_dir, include_global=False)
    if content:
        print(content)


def cmd_guard(args):
    """Start the guard daemon to prevent compaction-induced state loss."""
    session_id = args.session or None
    claude_pid = args.claude_pid or find_claude_pid()
    protect_patterns = _compile_protect_patterns_or_exit(args)  # #122

    _apply_token_env_overrides(args)

    if getattr(args, "reload_self", False):
        from .guard import reload_self_daemon
        result = reload_self_daemon(
            cwd=args.cwd or os.getcwd(),
            session_id=session_id,
            threshold_mb=args.threshold,
            soft_threshold_mb=args.soft_threshold,
            rx_name=args.rx or "standard",
            interval=args.interval,
            auto_reload=not args.no_reload,
            reactive=not args.no_reactive,
            threshold_tokens=args.threshold_tokens,
            soft_threshold_tokens=args.soft_threshold_tokens,
            protect_patterns=protect_patterns,
        )
        if result.get("reloaded"):
            print(f"  Guard daemon reloaded (PID {result['old_pid']} → {result['new_pid']})")
        else:
            print(f"  Guard reload skipped: {result.get('reason')}")
        return

    if args.daemon:
        result = start_guard_daemon(
            cwd=args.cwd or os.getcwd(),
            threshold_mb=args.threshold,
            soft_threshold_mb=args.soft_threshold,
            rx_name=args.rx or "standard",
            interval=args.interval,
            auto_reload=not args.no_reload,
            reactive=not args.no_reactive,
            threshold_tokens=args.threshold_tokens,
            soft_threshold_tokens=args.soft_threshold_tokens,
            session_id=session_id,
            claude_pid=claude_pid,
            protect_patterns=protect_patterns,
        )
        # Defensive .get() so a future regression that drops a key from
        # start_guard_daemon's return dict surfaces as a readable message
        # instead of an opaque KeyError traceback.
        if result.get("already_running"):
            print(f"  Guard already running (PID {result.get('pid')})")
        elif result.get("started"):
            print(f"  Guard daemon started (PID {result.get('pid')})")
            print(f"  Log: {result.get('log_file')}")
        else:
            reason = result.get("reason") or "unknown (no started/already_running/reason set)"
            print(f"  Guard daemon failed to start: {reason}")
        return

    start_guard(
        cwd=args.cwd or os.getcwd(),
        threshold_mb=args.threshold,
        soft_threshold_mb=args.soft_threshold,
        rx_name=args.rx or "standard",
        interval=args.interval,
        auto_reload=not args.no_reload,
        reactive=not args.no_reactive,
        threshold_tokens=args.threshold_tokens,
        soft_threshold_tokens=args.soft_threshold_tokens,
        session_id=session_id,
        claude_pid=claude_pid,
        protect_patterns=protect_patterns,
    )


def cmd_guard_watchdog(args):
    """Scan guard logs for stuck reload-loops; report, and optionally arrest them.

    A process safeguard outside the daemon: an OLD or broken guard that fails to
    self-arrest (the f641174c / PilotCC reload-loop) keeps spinning until killed.
    This reads the guard logs and flags the loop signature. ``--fix`` SIGTERMs a
    LIVE looping daemon that has been IDENTITY-VERIFIED as a cozempic guard
    process; without that check, a recycled PID (kernel reuse after hard exit)
    could be an innocent unrelated process. The default mode is report-only.
    """
    import signal
    from .watchdog import scan_guard_logs
    from .guard import _guard_tmp_root

    log_dir = getattr(args, "log_dir", None) or str(_guard_tmp_root())
    loop_trip = getattr(args, "loop_trip", None) or 20
    hits = scan_guard_logs(log_dir, loop_trip=loop_trip)

    print("\n  COZEMPIC GUARD-LOOP WATCHDOG")
    print("  ═══════════════════════════════════════════════════════════════════")
    print(f"  Scanned: {log_dir}")
    if not hits:
        print("  ✓ No stuck guard loops detected.\n")
        return

    arrested = 0
    for h in hits:
        live = "LIVE" if h.pid_alive else "dead/stale"
        print(f"\n  ✗ {h.log_file.name}")
        print(f"      pid={h.pid} ({live})")
        print(f"      {h.report.reason}")
        print(f"      futile={h.report.futile_cycles} total={h.report.total_prune_cycles} "
              f"backoff_max={h.report.max_backoff_s}s exit_seen={h.report.has_exit}")
        if getattr(args, "fix", False) and h.guard_confirmed:
            try:
                os.kill(int(h.pid), signal.SIGTERM)
                print(f"      → SIGTERM sent to looping daemon pid {h.pid}")
                arrested += 1
            except (ProcessLookupError, PermissionError, OSError) as e:
                print(f"      ! could not signal pid {h.pid}: {e}")
        elif getattr(args, "fix", False) and h.pid_alive and h.pid:
            print(f"      → pid {h.pid} is alive but is NOT a cozempic guard "
                  f"(recycled?) — refusing to kill")
        elif h.pid_alive:
            print(f"      → re-run with --fix to terminate this looping daemon "
                  f"(or: kill {h.pid})")

    live_hits = sum(1 for h in hits if h.pid_alive)
    print(f"\n  {len(hits)} stuck loop(s); {live_hits} live"
          + (f", {arrested} arrested" if arrested else "") + ".\n")
    # Non-zero exit so a cron / CI wrapper can alert on a LIVE stuck loop that
    # wasn't arrested.
    if live_hits and not arrested:
        sys.exit(3)


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
            fix_prefix = "      Fix: "
            fix_text = r.fix_description.replace("\n", f"\n{' ' * len(fix_prefix)}")
            print(f"{fix_prefix}{fix_text}")
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


def cmd_uninstall(args):
    """Reverse `cozempic init` — remove hooks, the slash command, and markers.

    Default scope is GLOBAL (~/.claude). Keeps the user's data (savings ledger,
    receipts) unless --purge. Sets the auto-init opt-out so init won't re-wire.
    """
    from .init import preview_uninstall, run_uninstall

    scope = "all" if getattr(args, "all", False) else ("project" if getattr(args, "project", False) else "global")
    purge = getattr(args, "purge", False)

    print("\n  COZEMPIC UNINSTALL")
    print("  ═══════════════════════════════════════════════════════════════════")
    print(f"  Scope: {scope}" + ("  (+ purge data)" if purge else ""))

    if getattr(args, "dry_run", False):
        prev = preview_uninstall(scope, purge)
        print("  Dry run — nothing will be changed.\n")
        print(f"    Hooks would be removed from: {prev['hooks_in'] or '(none)'}")
        print(f"    Slash command (~/.claude/commands/cozempic.md): "
              f"{'remove' if prev['slash_command'] else '(not present / not ours)'}")
        print(f"    Remind counter: {'remove' if prev['remind_counter'] else '(none)'}")
        if purge:
            print(f"    Purge data: {prev['purge_data'] or '(none)'}")
        print()
        return

    if purge:
        print("  --purge will DELETE your savings ledger + receipts (~/.cozempic). This is irreversible.")
        try:
            if input("  Continue? [y/N] ").strip().lower() != "y":
                print("  Aborted.\n")
                return
        except EOFError:
            print("  Aborted (no confirmation).\n")
            return

    result = run_uninstall(scope, purge)
    n_hooks = sum(len(h.get("removed", [])) for h in result["hooks"])
    print(f"  Removed {n_hooks} hook(s) across {len(result['hooks'])} settings file(s).")
    for h in result["hooks"]:
        if h.get("removed"):
            print(f"    - {h['settings_path']}: {', '.join(h['removed'])}"
                  + (f"  (backup: {h['backup_path']})" if h.get("backup_path") else ""))
    sc = result.get("slash_command")
    if sc and sc.get("removed"):
        print(f"  Removed slash command: {sc['path']}"
              + (f"  (backup: {sc['backup_path']})" if sc.get("backup_path") else ""))
    elif sc and sc.get("skipped_foreign"):
        print(f"  Left {sc['path']} in place (not a cozempic command).")
    if result.get("purged"):
        print(f"  Purged data: {', '.join(result['purged'])}")
    if result.get("opt_out_set"):
        print("  Auto-init disabled. Re-run `cozempic init` to reinstall.")
    print()


def cmd_init(args):
    """Wire cozempic hooks and slash command into the current project (or globally)."""
    if getattr(args, "uninstall_global", False):
        # Deprecated alias → `cozempic uninstall --global`.
        print("  Note: `init --uninstall-global` is deprecated; use `cozempic uninstall`.", file=sys.stderr)
        from .init import uninstall_hooks
        result = uninstall_hooks(str(Path.home()))
        print("\n  COZEMPIC INIT — UNINSTALL GLOBAL")
        print("  ═══════════════════════════════════════════════════════════════════")
        if result.get("removed"):
            print(f"  Removed {len(result['removed'])} hook(s) from {result['settings_path']}")
            for h in result["removed"]:
                print(f"    - {h}")
            if result.get("backup_path"):
                print(f"  Backup: {result['backup_path']}")
        else:
            print("  No cozempic hooks found in ~/.claude/settings.json — nothing to remove.")
        # Mark as opted-out so global auto-init doesn't re-fire
        try:
            _GLOBAL_INIT_MARKER.touch()
        except OSError:
            pass
        print()
        return

    if getattr(args, "global_install", False):
        project_dir = str(Path.home())
        scope_label = "GLOBAL (~/.claude/)"
    else:
        project_dir = args.cwd or os.getcwd()
        scope_label = f"Project: {project_dir}"

    print(f"\n  COZEMPIC INIT")
    print(f"  ═══════════════════════════════════════════════════════════════════")
    print(f"  {scope_label}")
    print()

    result = run_init(project_dir, skip_slash=args.no_slash_command)
    if getattr(args, "global_install", False):
        # Mark as initialized so global auto-init doesn't re-fire
        try:
            _GLOBAL_INIT_MARKER.touch()
        except OSError:
            pass

    # Report hooks
    hooks = result["hooks"]
    updated = hooks.get("updated") or []
    if hooks.get("error"):
        print(f"  Hooks: ERROR — {hooks['error']}")
    elif hooks["added"] or updated:
        print(f"  Hooks wired in {hooks['settings_path']}:")
        for h in hooks["added"]:
            print(f"    + {h} (added)")
        for h in updated:
            print(f"    → {h} (refreshed from stale schema)")
        if hooks["backup_path"]:
            print(f"  Backup: {hooks['backup_path']}")
    else:
        print(f"  Hooks: already at current schema (nothing to add or update)")

    if hooks["skipped"]:
        for h in hooks["skipped"]:
            print(f"    ~ {h} (current, skipped)")

    # #158: if cozempic is running from a throwaway env (uvx / uv run), the path
    # baked into the hooks won't exist next session, so the guard daemon can't
    # auto-start. Warn loudly and point at a persistent install.
    if hooks.get("ephemeral"):
        print()
        print("  ⚠ cozempic is running from an EPHEMERAL environment (e.g. `uvx`).")
        print("    The guard daemon will NOT auto-start on the next session — the hook")
        print("    can't find cozempic. Install persistently so it lands on PATH:")
        print("        uv tool install cozempic")
        print("    then re-run `cozempic init`.")

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
    print(f"    - Recovery context after compaction (PostCompact hook)")
    print(f"    - Final checkpoint on session end (Stop hook)")
    print()
    print(f"  Just start Claude Code normally. No second terminal needed.")
    print()


def cmd_self_update(args):
    """Force-upgrade cozempic from PyPI regardless of install method."""
    from .updater import _get_latest_version, _do_upgrade, _version_tuple
    from . import __version__

    latest = _get_latest_version()
    if latest is None:
        print("  Could not reach PyPI.")
        sys.exit(1)

    if _version_tuple(latest) <= _version_tuple(__version__):
        print(f"  Cozempic v{__version__} is already the latest.")
        return

    print(f"  Upgrading {__version__} → {latest}...")
    if _do_upgrade(latest):
        print(f"  Cozempic v{latest} installed. Restart to use the new version.")
    else:
        print(f"  Upgrade failed. Try: pip install --upgrade cozempic")
        sys.exit(1)


_NUDGE_DEFAULT_TIERS = (0.25, 0.55, 0.80)
# A fired tier re-arms only after context drops this far below it (hysteresis), so
# jitter/oscillation around a tier boundary can't re-fire the nudge repeatedly.
_NUDGE_REARM_HYSTERESIS = 0.07


def _build_nudge_message(tier_key: int, pct: float, proj: float | None,
                         inflight: bool = False) -> str:
    """Locked nudge copy. `proj` = projected reduction % from the daemon's armed
    sentinel (omitted if 0/unavailable). `inflight` = harness work (agents/tools)
    is running, so the guard reloads once it finishes — don't promise an idle
    reload then."""
    pct_disp = int(round(pct * 100))
    if tier_key <= 25:
        return (f"✦ Cozempic: context {pct_disp}%. Optional — `/cozempic reload` does a "
                f"lossless prune+resume at any breakpoint (higher fidelity than autocompact).")
    reclaim = f"reclaims ~{int(round(proj))}% by pruning bloat" if proj else "prunes bloat"
    # When agents/tools are in flight, the safe-point gate holds the reload until
    # they finish, so don't tell the user it happens the moment they pause.
    when = ("the guard auto-reloads once your current agents/tools finish"
            if inflight else "the guard auto-reloads once you pause between turns")
    if tier_key <= 55:
        return (f"✦ Cozempic: you're at {pct_disp}% context. A safe reload {reclaim} and "
                f"resumes automatically (conversation preserved). Your call:\n"
                f"    • run `/cozempic reload` now to control the timing, or\n"
                f"    • do nothing — {when}.")
    reclaim80 = (f"reclaims ~{int(round(proj))}% without loss" if proj
                 else "prunes bloat without losing your conversation")
    tail = ("Run `/cozempic reload` now to pick the moment, or it reloads once your "
            "agents/tools finish." if inflight else
            "Run `/cozempic reload` now to pick the moment or wait for the autoreload to kick in.")
    return (f"✦ Cozempic: context {pct_disp}% — approaching the autocompact wall. A reload "
            f"{reclaim80}. {tail}")


def cmd_nudge(args):
    """Stop-hook: emit a non-blocking systemMessage nudging the user to
    `/cozempic reload` at 25/55/80% context, ONCE per tier. Takes NO action — no
    prune, no reload, never blocks the stop, never fed to the model. Always exit 0.
    """
    import json as _json
    if os.environ.get("COZEMPIC_NUDGE_OFF"):
        return
    try:
        payload = _json.load(sys.stdin)
    except Exception:
        return
    if not isinstance(payload, dict):  # valid JSON but wrong shape (list/int/str)
        return
    transcript = payload.get("transcript_path") or ""
    session = payload.get("session_id") or transcript
    if not transcript or not os.path.exists(transcript):
        return
    try:
        from .session import load_messages
        from .tokens import detect_context_window, extract_usage_tokens
        messages = load_messages(Path(transcript))
        usage = extract_usage_tokens(messages)
        window = detect_context_window(messages)
    except Exception:
        return
    if not usage or not window:
        return
    tokens = int(usage.get("total") or 0)
    if tokens <= 0 or window <= 0:
        return
    pct = tokens / window
    # Tiers: prefer the guard's RESOLVED reload-tier fractions (so the nudge fires
    # where the guard actually reloads — tracks a raised --threshold and keeps the
    # FYI aligned with the real SOFT tier); fall back to the defaults. An explicit
    # COZEMPIC_NUDGE_PCTS override always wins.
    tiers = _NUDGE_DEFAULT_TIERS
    try:
        from .session import get_session_nudge_tiers
        _gt = get_session_nudge_tiers(payload.get("session_id") or "")
        if _gt:
            tiers = tuple(_gt)
    except Exception:
        pass
    raw = os.environ.get("COZEMPIC_NUDGE_PCTS")
    if raw:
        try:
            # Keep only finite fractions in (0, 1]; a NaN/inf tier silently disables
            # the nudge (pct >= nan/inf is always False → it never fires).
            _t = tuple(sorted(v for x in raw.split(",") if x.strip()
                              and math.isfinite(v := float(x)) and 0 < v <= 1))
            tiers = _t or tiers
        except Exception:
            pass
    crossed = max((t for t in tiers if pct >= t), default=None)
    tier_key = int(round(crossed * 100)) if crossed is not None else None
    # Once-per-tier latch with re-arm: drop tiers we've fallen BELOW (so they
    # re-fire on a later re-cross), persist that drop in EVERY path, then fire the
    # current tier only if not already latched.
    state_file = Path.home() / ".claude" / "cozempic-metrics" / "nudge-state.json"
    try:
        state = _json.loads(state_file.read_text()) if state_file.exists() else {}
    except Exception:
        state = {}
    if not isinstance(state, dict):  # corrupt/foreign state file → start clean
        state = {}
    sess_state = state.get(session)
    if not isinstance(sess_state, dict):
        sess_state = {}
    try:
        prev_fired = {int(t) for t in sess_state.get("tiers_fired", [])
                      if str(t).lstrip("-").isdigit()}
    except Exception:
        prev_fired = set()
    # Re-arm with HYSTERESIS: keep a tier latched until context drops a clear band
    # below it, so oscillating right around a tier (e.g. 54%↔56%) does NOT re-fire
    # the nudge each time. A tier re-arms only once pct falls below (tier − band).
    fired = {t for t in prev_fired if pct >= (t / 100.0) - _NUDGE_REARM_HYSTERESIS}
    emit = tier_key is not None and tier_key not in fired
    if emit:
        # Latch this tier AND every lower tier — crossing a higher tier means the
        # lower ones were already passed, so a later dip to a lower tier must not
        # fire its now-stale FYI (e.g. 56%→54% should stay silent, not emit 25%).
        fired |= {int(round(t * 100)) for t in tiers if t <= crossed}

    def _persist():
        sess_state["tiers_fired"] = sorted(fired)
        state[session] = sess_state
        try:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(_json.dumps(state))
        except Exception:
            pass

    if not emit:
        if fired != prev_fired:  # a drop occurred — persist the re-arm
            _persist()
        return
    # Real projected reduction from the daemon's armed sentinel (omitted if absent).
    proj = None
    try:
        from .guard import read_armed, mark_armed_warned
        armed = read_armed(session, Path(transcript))
        if armed and isinstance(armed.get("projected_pct"), (int, float)):
            proj = float(armed["projected_pct"])
    except Exception:
        pass
    # If harness work (Workflow / background subagent / open tool call) is in
    # flight, the safe-point gate holds the reload until it finishes — so the copy
    # must not promise an idle reload.
    inflight = False
    try:
        from .guard import detect_in_flight
        d = detect_in_flight(messages)
        inflight = bool(d.get("workflow") or d.get("background")
                        or d.get("agent") or d.get("open_call"))
    except Exception:
        pass
    msg = _build_nudge_message(tier_key, pct, proj, inflight=inflight)
    # First-nudge-only opt-out hint.
    if not sess_state.get("hint_shown"):
        msg += "  ·  silence: COZEMPIC_NUDGE_OFF=1"
        sess_state["hint_shown"] = True
    _persist()  # records the newly-fired tier (and any drop)
    # Only the actionable reload tiers (anything above the lowest/SOFT FYI tier)
    # constitute the WARNING the daemon's warned-before-reload gate waits for.
    # mark_armed_warned upserts the sentinel so the warning isn't lost to an
    # arm/nudge race (the nudge can fire before the daemon's next poll arms).
    if crossed is not None and crossed > min(tiers):
        try:
            mark_armed_warned(session, Path(transcript))
        except Exception:
            pass
    _json.dump({"systemMessage": msg}, sys.stdout)


def cmd_remind(args):
    """Periodic rule reinforcement — outputs active digest rules to stderr.

    Designed for PostToolUse hook. Counts tool calls via a counter file,
    outputs rules every N calls (default 25). Stderr output appears in
    Claude's context, keeping rules in the recency window.
    """
    interval = int(getattr(args, "interval", None) or 25)
    counter_file = Path.home() / ".cozempic_remind_counter"

    # Increment counter
    try:
        count = int(counter_file.read_text().strip()) if counter_file.exists() else 0
    except (ValueError, OSError):
        count = 0
    count += 1
    try:
        counter_file.write_text(str(count))
    except OSError:
        pass

    # Only output every N calls
    if count % interval != 0:
        return

    # Collect rules: digest active rules + CLAUDE.md critical rules
    lines = []

    # 1. Active digest rules. Sanitize r.rule — it is untrusted transcript-derived
    # text and this is emitted into Claude's context via the PostToolUse hook, so a
    # multi-line / markdown-structured rule could inject instructions (sibling of
    # the digest injection fix; the #123 fix-one-miss-the-sibling lesson).
    from .digest import load_digest_store, _sanitize_for_injection
    store = load_digest_store()
    active = store.active_rules()
    if active:
        for r in active[:5]:
            lines.append(f"  [{r.id}|{r.scope}] {_sanitize_for_injection(r.rule)}")

    # 2. Critical CLAUDE.md rules (grep for enforcement markers)
    for candidate in ["CLAUDE.md", ".claude/CLAUDE.md"]:
        p = Path(candidate)
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line_stripped = line.strip()
                    if any(kw in line_stripped.upper() for kw in
                           ["MUST NEVER", "NEVER ", "MUST ALWAYS", "CRITICAL:", "IMPORTANT:"]):
                        if len(line_stripped) > 10 and not line_stripped.startswith("#"):
                            lines.append(f"  {line_stripped[:120]}")
                            if len(lines) >= 8:
                                break
            except OSError:
                pass
            break

    if lines:
        print(f"Cozempic behavioral rules (reminder #{count // interval}):", file=sys.stderr)
        for line in lines[:8]:
            print(line, file=sys.stderr)


def cmd_completions(args):
    """Generate shell completion scripts."""
    from .completion import bash_completion, zsh_completion
    if args.shell == "bash":
        print(bash_completion())
    elif args.shell == "zsh":
        print(zsh_completion())


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


def _digest_session(args):
    """Resolve session path and ID from args.

    Resolution strategy:
      - absent / "current"  → cwd-based auto-detect via find_current_session(cwd);
                              exits 1 with a stderr message if none found.
      - explicit UUID / UUID prefix / file path → resolve_session(session_arg);
                              session_id recovered from resolved path stem.

    Returns (path, session_id, cwd).
    """
    from .session import find_current_session, resolve_session
    cwd = getattr(args, "cwd", None) or os.getcwd()
    session_arg = getattr(args, "session", None)
    if not session_arg or session_arg == "current":
        # Both absent and explicit "current" use cwd-based auto-detection,
        # consistent with how the rest of cli.py resolves the current session.
        # Routing "current" through resolve_session() would invoke process-detection
        # (a different strategy from slug-match) — not what we want here.
        # strict=True: digest flush/inject are write operations — injecting rules
        # into an unrelated session's JSONL is a context-contamination risk.
        sess = find_current_session(cwd, strict=True)
        if not sess:
            # Changed to stderr: consistent with resolve_session error output
            # and all other error paths in cli.py.
            print("No active session found.", file=sys.stderr)
            sys.exit(1)
        return sess["path"], sess.get("session_id", ""), cwd
    # resolve_session handles: explicit path, UUID, UUID prefix
    resolved = resolve_session(session_arg)
    session_id = resolved.stem  # filename stem IS the session UUID
    return resolved, session_id, cwd


def cmd_digest(args):
    from .digest import (
        clear_digest_store, flush_digest,
        load_digest_store, recover_digest,
        save_digest_store, show_digest, update_digest,
    )
    from .session import load_messages, save_messages

    action = getattr(args, "digest_action", "show") or "show"

    if action == "show":
        print(show_digest())

    elif action == "update":
        session_path, session_id, cwd = _digest_session(args)
        messages = load_messages(session_path)
        added, upvoted, rejected = update_digest(
            messages, project_dir=cwd, session_id=session_id,
        )
        print(f"Digest updated: {added} new, {upvoted} reinforced, {rejected} rejected.")

    elif action == "clear":
        clear_digest_store()
        print("Behavioral digest cleared.")

    elif action == "flush":
        session_path, session_id, cwd = _digest_session(args)
        messages = load_messages(session_path)
        added, upvoted, rejected = flush_digest(
            messages, project_dir=cwd, session_id=session_id,
        )
        print(f"Digest flushed: {added} new, {upvoted} reinforced, {rejected} rejected.")

    elif action == "recover":
        cwd = getattr(args, "cwd", None) or os.getcwd()
        synced = recover_digest(project_dir=cwd)
        print(f"Synced {synced} rules to Claude Code memory.")

    elif action == "inject":
        cwd = getattr(args, "cwd", None) or os.getcwd()
        # The SessionStart hook calls `digest inject --session "$TRANSCRIPT"`, and
        # runs as a child of the live Claude process — the one moment we can learn
        # CC's own active session. Record it (keyed by the live Claude PID) so the
        # manual CLI path follows the real transcript instead of guessing from cwd.
        _transcript = getattr(args, "session", None)
        if _transcript and _transcript != "current":
            try:
                from .session import record_active_transcript
                record_active_transcript(_transcript)
            except Exception:
                pass
        store = load_digest_store(cwd)
        if store.is_empty():
            print("No rules to inject.")
            return
        from .digest import sync_to_memdir, _get_memdir
        synced = sync_to_memdir(store, cwd=cwd)
        if synced > 0:
            save_digest_store(store)
            print(f"Synced {synced} active rules to Claude Code memory.")
        elif not store.active_rules():
            # Nothing ACTIVE to write yet (rules activate after 2 occurrences) — the
            # memory-dir state is irrelevant here. The old code mislabeled this as
            # "Could not find Claude Code memory directory".
            print("No active rules to sync yet.")
        elif _get_memdir(cwd) is None:
            # We DO have active rules, but Claude Code hasn't created the memory dir
            # for this project yet — it syncs automatically once the dir exists.
            print("Memory directory not created by Claude Code yet — will sync once it exists.")
        else:
            print("Nothing to sync.")


# ─── Parser ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cozempic",
        description="Context weight-loss tool for Claude Code — prune bloated JSONL conversation files",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.8.39")
    parser.add_argument("--context-window", type=int, default=None, help="Override context window size in tokens (e.g. 1000000 for 1M beta)")
    parser.add_argument("--system-overhead-tokens", type=int, default=None, help="Override system overhead estimate (default: 21000). Increase for heavy rules/MCP configs.")
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
    p_treat.add_argument("--strict", action="store_true", help="Fail (exit 5) if the prescription can't be applied, instead of falling back to a lighter one")
    p_treat.add_argument("--execute", action="store_true", help="Apply changes (default is dry-run)")
    p_treat.add_argument("--project", help="Filter by project name")
    p_treat.add_argument("--thinking-mode", choices=["remove", "truncate", "signature-only"], help="Thinking block mode")
    p_treat.add_argument("--protect-pattern", action="append", metavar="REGEX",
                         help="Protect messages whose text matches REGEX from pruning (repeatable). Runs against every message — avoid catastrophic-backtracking regexes.")

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
    p_reload.add_argument("--strict", action="store_true", help="Fail (exit 5) if the prescription can't be applied, instead of falling back to a lighter one")
    p_reload.add_argument("--thinking-mode", choices=["remove", "truncate", "signature-only"])
    p_reload.add_argument("--session", help="Explicit session ID, UUID prefix, or .jsonl path (bypasses auto-detection)")
    p_reload.add_argument("--protect-pattern", action="append", metavar="REGEX",
                          help="Protect messages whose text matches REGEX from pruning (repeatable).")
    p_reload.add_argument(
        "--wait", nargs="?", const=30, type=int, default=None,
        metavar="SECS",
        help="If another reload is in flight, wait up to SECS for it to finish "
             "(default: 30 if --wait passed with no value). Without --wait, "
             "we fail fast with exit code 2.",
    )

    # checkpoint
    p_cp = sub.add_parser("checkpoint", help="Save team/agent state from the current session (no pruning)")
    p_cp.add_argument("--cwd", help="Working directory (default: current)")
    p_cp.add_argument("--show", action="store_true", help="Print the team state after saving")

    # post-compact
    p_post_compact = sub.add_parser("post-compact", help="Output team state after compaction (for PostCompact hook)")
    p_post_compact.add_argument("--cwd", help="Working directory (default: current)")

    # guard
    p_guard = sub.add_parser("guard", help="Background sentinel — auto-prune before compaction triggers")
    p_guard.add_argument("--cwd", help="Working directory (default: current)")
    p_guard.add_argument("-rx", help="Prescription to apply (default: standard)")
    p_guard.add_argument("--threshold", type=_positive_float, default=50.0, help="Hard threshold in MB — full prune + reload (default: 50)")
    p_guard.add_argument("--soft-threshold", type=_positive_float, default=None, help="Soft threshold in MB — gentle prune, no reload (default: 60%% of --threshold)")
    p_guard.add_argument("--interval", type=_positive_int, default=30, help="Check interval in seconds (default: 30)")
    p_guard.add_argument("--threshold-tokens", type=_positive_int, default=None, help="Hard threshold in tokens (default: 75%% of context window)")
    p_guard.add_argument("--soft-threshold-tokens", type=_positive_int, default=None, help="Soft threshold in tokens (default: 45%% of context window)")
    p_guard.add_argument("--no-reload", action="store_true", help="Prune without auto-reload at hard threshold")
    p_guard.add_argument("--no-reactive", action="store_true", help="Disable reactive overflow recovery (kqueue/polling watcher)")
    p_guard.add_argument("--daemon", action="store_true", help="Run in background (PID file prevents double-starts)")
    p_guard.add_argument("--reload-self", action="store_true", help="Gracefully restart the running daemon for this session (used after upgrading cozempic in place)")
    p_guard.add_argument("--session", help="Explicit session ID or path (bypasses auto-detection)")
    p_guard.add_argument("--claude-pid", type=int, default=None, help=argparse.SUPPRESS)
    p_guard.add_argument("--system-overhead-tokens", type=int, default=None, help="Override system overhead token estimate (default: 21000). Increase for heavy configs with many rules files, MCP servers, or large CLAUDE.md")
    p_guard.add_argument("--protect-pattern", action="append", metavar="REGEX",
                         help="Protect messages whose text matches REGEX from pruning during auto-prunes (repeatable).")

    # init
    p_init = sub.add_parser("init", help="Auto-wire hooks and slash command into this project (or globally with --global)")
    p_init.add_argument("--cwd", help="Project directory (default: current)")
    p_init.add_argument("--no-slash-command", action="store_true", help="Skip installing /cozempic slash command")
    p_init.add_argument("--global", dest="global_install", action="store_true", help="Wire hooks into ~/.claude/settings.json so every Claude Code session in every project is protected")
    p_init.add_argument("--uninstall-global", action="store_true", help="(deprecated) use `cozempic uninstall`")

    p_uninstall = sub.add_parser("uninstall", help="Reverse `cozempic init` — remove hooks, slash command, markers (global by default)")
    p_uninstall.add_argument("--project", action="store_true", help="Uninstall from THIS project's .claude/settings.json (default is global ~/.claude)")
    p_uninstall.add_argument("--all", action="store_true", help="Uninstall from both global and project")
    p_uninstall.add_argument("--purge", action="store_true", help="ALSO delete ~/.cozempic data + savings ledger (irreversible; prompts)")
    p_uninstall.add_argument("--dry-run", action="store_true", help="Show what would be removed without changing anything")

    # doctor
    p_doctor = sub.add_parser("doctor", help="Check for known Claude Code issues and fix them")
    p_doctor.add_argument("--fix", action="store_true", help="Auto-fix issues where possible")

    # guard-watchdog (process safeguard: detect stuck reload-loops in guard logs)
    p_wd = sub.add_parser("guard-watchdog",
                          help="Scan guard logs for stuck reload-loops (report; --fix terminates them)")
    p_wd.add_argument("--fix", action="store_true",
                      help="SIGTERM any LIVE looping guard daemon (default: report only)")
    p_wd.add_argument("--log-dir", help="Directory of guard logs (default: guard tmp root)")
    p_wd.add_argument("--loop-trip", type=_positive_int, default=20,
                      help="Futile-cycle count that flags a stuck loop (default: 20)")

    # formulary
    sub.add_parser("formulary", help="Show all strategies & prescriptions")

    # completions
    p_comp = sub.add_parser("completions", help="Generate shell completion script")
    p_comp.add_argument("shell", choices=["bash", "zsh"], help="Shell type")

    # self-update
    sub.add_parser("self-update", help="Upgrade cozempic to the latest version from PyPI")

    # remind
    p_remind = sub.add_parser("remind", help="Output active behavioral rules (for PostToolUse hook)")
    p_remind.add_argument("--interval", type=_positive_int, default=25, help="Output every N tool calls (default: 25)")

    # nudge (Stop hook — non-blocking context nudge, no action)
    sub.add_parser("nudge", help="Stop-hook: non-blocking 'prune now?' nudge at 25/55/80%% (no action)")

    # digest
    p_digest = sub.add_parser("digest", help="Manage behavioral correction rules")
    p_digest.add_argument("digest_action", nargs="?", default="show",
                          choices=["show", "update", "clear", "flush", "recover", "inject"],
                          help="Action: show (default), update, clear, flush, recover, inject")
    p_digest.add_argument("--session", help=session_help)
    p_digest.add_argument("--cwd", help="Working directory (default: current)")

    p_dash = sub.add_parser("dashboard", help="Generate + open the prune-value dashboard (HTML)")
    p_dash.add_argument("--no-open", action="store_true",
                        help="Write the HTML but don't open a browser")
    p_dash.add_argument("--agent", help="Filter to one agent (e.g. claude, codex)")

    return parser


_SUBCOMMANDS = {
    "list", "current", "diagnose", "treat", "strategy", "reload",
    "checkpoint", "post-compact", "guard", "init", "doctor", "formulary", "completions",
    "digest", "self-update", "remind", "guard-watchdog", "dashboard", "uninstall",
}


def _prescan_argv(argv: list[str]) -> list[str]:
    """Extract global flags from anywhere in argv, setting env vars as a side effect.

    Argparse requires root-level flags before the subcommand name; this lets users
    put --context-window and --system-overhead-tokens anywhere (#13).
    Also strips --no-auto-init (sets COZEMPIC_NO_AUTO_INIT=1 as a side effect) so
    auto-init can opt out per-invocation regardless of where the flag appears.
    Returns cleaned argv with those flags removed so argparse sees the rest normally.
    """
    cleaned: list[str] = []
    i = 0
    sub_seen = False
    while i < len(argv):
        tok = argv[i]
        # --no-auto-init / --no-global-init can appear anywhere (before or after subcommand)
        if tok == "--no-auto-init":
            os.environ["COZEMPIC_NO_AUTO_INIT"] = "1"
            i += 1
            continue
        if tok == "--no-global-init":
            os.environ["COZEMPIC_NO_GLOBAL_INIT"] = "1"
            i += 1
            continue
        if not sub_seen and tok in _SUBCOMMANDS:
            sub_seen = True
            cleaned.append(tok)
            i += 1
            continue
        if sub_seen:
            if tok == "--context-window" and i + 1 < len(argv):
                val = argv[i + 1]
                try:
                    if int(val) <= 0:
                        raise ValueError
                    os.environ["COZEMPIC_CONTEXT_WINDOW"] = val
                except ValueError:
                    print(f"Warning: ignoring invalid --context-window '{val}'", file=sys.stderr)
                i += 2
                continue
            if tok.startswith("--context-window="):
                val = tok.split("=", 1)[1]
                try:
                    if int(val) <= 0:
                        raise ValueError
                    os.environ["COZEMPIC_CONTEXT_WINDOW"] = val
                except ValueError:
                    print(f"Warning: ignoring invalid --context-window '{val}'", file=sys.stderr)
                i += 1
                continue
            if tok == "--system-overhead-tokens" and i + 1 < len(argv):
                val = argv[i + 1]
                try:
                    if int(val) < 0:
                        raise ValueError
                    os.environ["COZEMPIC_SYSTEM_OVERHEAD_TOKENS"] = val
                except ValueError:
                    print(f"Warning: ignoring invalid --system-overhead-tokens '{val}'", file=sys.stderr)
                i += 2
                continue
            if tok.startswith("--system-overhead-tokens="):
                val = tok.split("=", 1)[1]
                try:
                    if int(val) < 0:
                        raise ValueError
                    os.environ["COZEMPIC_SYSTEM_OVERHEAD_TOKENS"] = val
                except ValueError:
                    print(f"Warning: ignoring invalid --system-overhead-tokens '{val}'", file=sys.stderr)
                i += 1
                continue
        cleaned.append(tok)
        i += 1
    return cleaned


# Commands whose STDOUT is a machine-parsed protocol (not human text): the
# auto-updater and auto-init must stay silent / off so nothing prepends to it.
_STDOUT_PROTOCOL_CMDS = frozenset({"nudge"})

_AUTO_INIT_SKIP_CMDS = frozenset({
    "init",          # would loop / shadow user intent
    "completions",   # generates shell completion, no project state needed
    "self-update",   # internal upgrade
    "doctor",        # diagnostic-only; doctor surfaces missing init via its own check
    "nudge",         # Stop-hook protocol command; never mutate state as a side effect
    "guard-watchdog",  # read-only log scan; never mutate project state
    "uninstall",     # the whole point is to REMOVE wiring — must never auto-init
    "dashboard",     # report-only; reads/writes only ~/.cozempic, never project state
})

_GLOBAL_INIT_MARKER = Path.home() / ".cozempic_global_initialized"


def _prompt_with_timeout(msg: str, timeout: int = 30, default: str = "n") -> str:
    """input() wrapped with a hard timeout so we never hang a cozempic invocation.

    Returns `default` on timeout, EOF, or Ctrl-C. TTY-detection should have
    already been done by the caller; this is a belt-and-suspenders guard
    against fooled detection (tmux piping quirks, etc.).
    """
    import signal as _signal

    def _timeout_handler(signum, frame):
        raise TimeoutError

    try:
        prev_handler = _signal.signal(_signal.SIGALRM, _timeout_handler)
    except (ValueError, OSError, AttributeError):
        # SIGALRM unavailable (Windows, non-main thread) — run without timeout
        try:
            return input(msg).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return default

    _signal.alarm(timeout)
    try:
        result = input(msg)
    except (EOFError, KeyboardInterrupt, TimeoutError):
        result = None
    finally:
        # Cancel the alarm FIRST so a race between input() returning and the
        # finally block can't cause SIGALRM to fire during cleanup. Only then
        # restore the previous handler.
        try:
            _signal.alarm(0)
        except (ValueError, OSError):
            pass
        try:
            _signal.signal(_signal.SIGALRM, prev_handler)
        except (ValueError, OSError):
            pass

    if result is None:
        print("", file=sys.stderr)
        return default
    return result.strip().lower()


def _maybe_global_init(argv: list[str]) -> None:
    """Wire cozempic into ~/.claude/settings.json on first cozempic invocation
    on this machine — guarantees protection for every Claude Code session in
    every project, including projects the user has never run cozempic in.

    Bail-outs (in order):
      1. COZEMPIC_NO_GLOBAL_INIT=1 in env (also set by --no-global-init)
      2. Marker file exists (already done once)
      3. Subcommand is in _AUTO_INIT_SKIP_CMDS
      4. ~/.claude/ doesn't exist (Claude Code not installed yet)
      5. Cozempic hooks are already in ~/.claude/settings.json (e.g. plugin
         marketplace install)

    Otherwise: writes hooks (skip slash command — project-level only) and
    prints a one-line opt-out notice to stderr. Touches the marker file so
    we never ask again on this machine.
    """
    if os.environ.get("COZEMPIC_NO_GLOBAL_INIT"):
        return
    if _GLOBAL_INIT_MARKER.exists():
        return

    # --help / -h are pure-info, never trigger init
    if "--help" in argv or "-h" in argv:
        return

    cmd = next((tok for tok in argv if tok in _SUBCOMMANDS), None)
    is_version_check = "--version" in argv

    # Trigger conditions:
    #   - any subcommand not in the skip list, OR
    #   - bare `cozempic --version` (the canonical post-install verification check)
    if cmd in _AUTO_INIT_SKIP_CMDS:
        return
    if cmd is None and not is_version_check:
        return

    home_claude = Path.home() / ".claude"
    if not home_claude.exists():
        return  # Claude Code not yet installed — defer until it is

    if _project_is_cozempic_current(home_claude):
        # Already wired (probably via plugin marketplace install). Mark as done.
        try:
            _GLOBAL_INIT_MARKER.touch()
        except OSError:
            pass
        return

    # Ask the user interactively when both stdin and stderr are TTYs (real terminal).
    # Fall back to silent auto-install for non-interactive contexts (CI, pipelines,
    # Claude Code subprocess invocations) so we never hang waiting for input.
    interactive = sys.stdin.isatty() and sys.stderr.isatty()

    if interactive:
        try:
            print(
                "\n  Cozempic — enable background protection for every Claude Code session?",
                file=sys.stderr,
            )
            print(
                "  Wires hooks into ~/.claude/settings.json. Reverse any time with "
                "`cozempic init --uninstall-global`.",
                file=sys.stderr,
            )
            response = _prompt_with_timeout("  Enable? [Y/n] ", timeout=30, default="n")
        except OSError:
            response = "n"  # I/O error ≠ user consent — decline, don't install

        # Accept common "cancel" synonyms so users who press q/quit/cancel
        # don't accidentally opt IN.
        if response in ("n", "no", "q", "quit", "cancel", "exit", "x"):
            try:
                _GLOBAL_INIT_MARKER.touch()
            except OSError:
                pass
            print(
                "  Skipped. Run `cozempic init --global` later if you change your mind.\n",
                file=sys.stderr,
            )
            return

    try:
        result = run_init(str(Path.home()), skip_slash=True)
    except Exception as exc:
        # Touch the marker even on failure — prevents a DoS loop where every
        # single cozempic invocation re-attempts a failing run_init and spams
        # stderr. User can `rm ~/.cozempic_global_initialized` to retry after
        # addressing the underlying problem (read-only file, permissions, etc).
        try:
            _GLOBAL_INIT_MARKER.touch()
        except OSError:
            pass
        print(
            f"  Cozempic: global init failed ({exc}). Run `cozempic init --global` manually "
            "after fixing; `rm ~/.cozempic_global_initialized` to re-ask on next invocation.",
            file=sys.stderr,
        )
        return

    hooks_result = result.get("hooks", {})
    added = hooks_result.get("added", []) or []
    updated = hooks_result.get("updated", []) or []
    load_error = hooks_result.get("error")

    if load_error:
        # Don't claim we "enabled" anything we didn't actually install. Do NOT
        # touch the marker — let the user retry after `cozempic self-update`.
        print(
            f"  Cozempic: global init FAILED — {load_error}",
            file=sys.stderr,
        )
        return

    try:
        _GLOBAL_INIT_MARKER.touch()
    except OSError:
        pass

    if not (added or updated):
        return

    count_desc = []
    if added:
        count_desc.append(f"{len(added)} new")
    if updated:
        count_desc.append(f"{len(updated)} refreshed")
    summary = ", ".join(count_desc)

    if interactive:
        print(
            f"  Cozempic enabled — {summary} hook(s) wired into ~/.claude/settings.json.",
            file=sys.stderr,
        )
        print(
            "  Disable any time with `cozempic init --uninstall-global` "
            "or COZEMPIC_NO_GLOBAL_INIT=1.\n",
            file=sys.stderr,
        )
    else:
        print(
            f"  Cozempic: protecting every Claude Code session globally "
            f"({summary} hook(s) wired into ~/.claude/settings.json). "
            "Disable with `cozempic init --uninstall-global` or COZEMPIC_NO_GLOBAL_INIT=1.",
            file=sys.stderr,
        )


def _project_is_cozempic_current(claude_dir: Path) -> bool:
    """Predicate: "should we leave this settings dir alone?"

    Returns True iff the settings files (settings.json + settings.local.json)
    already have cozempic hooks AT THE CURRENT SCHEMA VERSION and none are
    stale. Returns False when refresh OR initial install is needed.

    NOTE: This is a "do nothing" predicate, not a "has any cozempic config"
    query. If one file is current and another has stale cozempic hooks, we
    return False so wire_hooks is called and refreshes the stale one.
    """
    import json as _json
    from .init import _is_cozempic_command, HOOK_SCHEMA_MARKER

    any_cozempic_found = False
    for name in ("settings.json", "settings.local.json"):
        p = claude_dir / name
        if not p.exists():
            continue
        try:
            data = _json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue

        hooks = data.get("hooks", {}) or {}
        if not isinstance(hooks, dict):
            continue

        for entries in hooks.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                for h in entry.get("hooks", []) or []:
                    if not isinstance(h, dict):
                        continue
                    cmd = str(h.get("command", ""))
                    if not _is_cozempic_command(cmd):
                        continue
                    any_cozempic_found = True
                    # Any non-current cozempic hook (missing or stale marker)
                    # means a refresh is due.
                    if HOOK_SCHEMA_MARKER not in cmd:
                        return False

    return any_cozempic_found


def _maybe_auto_init(argv: list[str]) -> None:
    """Auto-wire cozempic into the current Claude project on first use.

    Bail-outs (in order):
      1. COZEMPIC_NO_AUTO_INIT=1 in env (also set by --no-auto-init via _prescan_argv)
      2. cwd has no .claude/ directory (not a Claude project)
      3. Subcommand is in _AUTO_INIT_SKIP_CMDS or no subcommand was given
      4. Global hooks are current in ~/.claude/settings.json (local would be redundant)
      5. Cozempic hooks are already wired in this project's settings

    Otherwise: runs init silently and prints a single one-line notice to stderr.
    Failures are non-fatal — the user's original command still runs.
    """
    if os.environ.get("COZEMPIC_NO_AUTO_INIT"):
        return

    claude_dir = Path.cwd() / ".claude"
    if not claude_dir.exists():
        return  # not a Claude project — never modify foreign directories

    cmd = next((tok for tok in argv if tok in _SUBCOMMANDS), None)
    if cmd is None or cmd in _AUTO_INIT_SKIP_CMDS:
        return

    # Skip local init when global hooks are already current (avoids double-firing).
    # Guard: if cwd IS the home dir, home_claude == claude_dir -- fall through
    # to the normal local check instead.
    home_claude = Path.home() / ".claude"
    if home_claude != claude_dir and _project_is_cozempic_current(home_claude):
        # Warn if redundant local hooks are also present.
        if _project_is_cozempic_current(claude_dir):
            print(
                "  Cozempic: local hooks redundant (global hooks active) — "
                "they are harmless but can be removed from this project's "
                ".claude/settings.json.",
                file=sys.stderr,
            )
        return

    if _project_is_cozempic_current(claude_dir):
        return  # already initialized

    try:
        result = run_init(str(Path.cwd()))
    except Exception as exc:
        print(
            f"  Cozempic: auto-init skipped ({exc}). Run `cozempic init` manually to enable background protection.",
            file=sys.stderr,
        )
        return

    hooks_result = result.get("hooks", {})
    load_error = hooks_result.get("error")
    if load_error:
        print(
            f"  Cozempic: auto-init FAILED — {load_error}",
            file=sys.stderr,
        )
        return

    added = hooks_result.get("added", []) or []
    updated = hooks_result.get("updated", []) or []
    if added or updated:
        parts = []
        if added:
            parts.append(f"{len(added)} hook(s) wired")
        if updated:
            parts.append(f"{len(updated)} refreshed from stale schema")
        print(
            f"  Cozempic: auto-initialized this project ({', '.join(parts)}). "
            "Disable with --no-auto-init or COZEMPIC_NO_AUTO_INIT=1.",
            file=sys.stderr,
        )


def cmd_dashboard(args):
    """Generate (and open) the local prune-value dashboard from receipts."""
    from datetime import datetime, timezone

    from .dashboard import aggregate, load_receipts
    from .dashboard.lifetime import load_lifetime
    from .dashboard.render import dashboard_path, render_html, write_dashboard

    receipts = load_receipts()
    agent = getattr(args, "agent", None)
    if agent:
        receipts = [r for r in receipts
                    if isinstance(r.get("agent"), dict) and r["agent"].get("name") == agent]
    # The lifetime ledger is GLOBAL (no agent dimension), so it would contradict
    # an agent-filtered view ("456M reclaimed" next to "no prunes for codex").
    # Show it only on the unfiltered dashboard.
    ledger = None if agent else load_lifetime()
    data = aggregate(receipts)
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    label = "~/.cozempic/receipts" + (f" · agent={agent}" if agent else "")
    html_str = render_html(data, generated_ts=ts, source_label=label, ledger=ledger)
    try:
        path = write_dashboard(html_str)
    except Exception as exc:
        print(f"  Error: could not write dashboard: {exc}", file=sys.stderr)
        sys.exit(1)

    n = data.get("lifetime", {}).get("prunes_total", 0)
    print(f"  Dashboard: {path}")
    if ledger:
        line = f"  Lifetime: {ledger['tokens_saved'] / 1e6:.1f}M tokens reclaimed across {ledger['prune_count']:,} prunes"
        if ledger.get("since"):
            line += f" since {ledger['since']}"
        print(line)
        if ledger.get("session_multiplier_x"):
            print(f"  Sessions ~{ledger['session_multiplier_x']:.2f}x longer "
                  f"(measured across {ledger['sessions']:,} pruned sessions).")
    if n:
        lt = data["lifetime"]
        print(f"  {n} prune(s), {lt.get('committed', 0)} applied across "
              f"{lt.get('sessions', 0)} session(s).")
    elif agent:
        print(f"  No prunes recorded for agent '{agent}'.")
    else:
        print("  No prunes recorded yet — run `cozempic treat --execute` first.")

    if not getattr(args, "no_open", False):
        try:
            import webbrowser

            if not webbrowser.open(dashboard_path().as_uri()):
                print("  (open it manually in a browser)")
        except Exception:
            print("  (open it manually in a browser)")


def main():
    from .updater import maybe_auto_update, ping_install_if_new

    argv = _prescan_argv(sys.argv[1:])
    # The Stop-hook `nudge` emits a machine-parsed JSON protocol on STDOUT
    # ({"systemMessage": ...}). The auto-updater prints upgrade chatter to STDOUT,
    # which would prepend to that JSON and break the hook parser on the ~daily
    # upgrade tick. Skip the updater (and auto-init) entirely for stdout-protocol
    # commands — they still run on the user's next interactive invocation.
    _cmd0 = next((a for a in argv if not a.startswith("-")), None)
    if _cmd0 not in _STDOUT_PROTOCOL_CMDS:
        ping_install_if_new()
        maybe_auto_update()
    _maybe_global_init(argv)
    _maybe_auto_init(argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    # Mirror --context-window / --system-overhead-tokens into env vars using
    # `is not None` (not truthiness) so --system-overhead-tokens 0 is honored.
    _apply_token_env_overrides(args)

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
        "post-compact": cmd_post_compact,
        "guard": cmd_guard,
        "init": cmd_init,
        "uninstall": cmd_uninstall,
        "doctor": cmd_doctor,
        "guard-watchdog": cmd_guard_watchdog,
        "formulary": cmd_formulary,
        "completions": cmd_completions,
        "digest": cmd_digest,
        "self-update": cmd_self_update,
        "remind": cmd_remind,
        "nudge": cmd_nudge,
        "dashboard": cmd_dashboard,
    }

    try:
        commands[args.command](args)
    except ValueError as e:
        # Narrow catch: the `guard` / `reload` subcommands build pidfile
        # paths via `_pid_file_for_session` which raises on malformed
        # session_id. Surface those as a clean one-line error instead of
        # a Python traceback. Re-raise for any other command so bugs
        # aren't silently swallowed.
        if args.command in ("guard", "reload"):
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(2)
        raise


if __name__ == "__main__":
    main()
