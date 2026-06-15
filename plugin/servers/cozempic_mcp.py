#!/usr/bin/env python3
"""Cozempic MCP Server — exposes session diagnostics and treatment as Claude Code tools."""

from __future__ import annotations

import threading

from fastmcp import FastMCP

# Ensure all strategies are registered
import cozempic.strategies  # noqa: F401

mcp = FastMCP("Cozempic")


def _startup_maintenance() -> None:
    """Ping install counter + check for updates at MCP server startup.

    Runs in a background thread so it never blocks tool calls.
    MCP users don't have a TTY so force=True bypasses the isatty() guard.
    """
    try:
        from cozempic.updater import maybe_auto_update, ping_install_if_new
        ping_install_if_new()
        maybe_auto_update(force=True, silent=True)
    except Exception:
        pass


threading.Thread(target=_startup_maintenance, daemon=True, name="cozempic-startup").start()


@mcp.tool()
def diagnose_current() -> str:
    """Diagnose the current Claude Code session for context bloat.

    Returns token count, context window usage %, bloat breakdown by message type,
    and estimated savings for each prescription tier (gentle/standard/aggressive).
    Use this when context is getting heavy or the user asks about session health.
    """
    from cozempic.session import find_current_session, load_messages
    from cozempic.diagnosis import diagnose_session
    from cozempic.tokens import estimate_session_tokens
    from cozempic.registry import PRESCRIPTIONS
    from cozempic.executor import run_prescription

    sess = find_current_session()
    if not sess:
        return "Could not detect current session. Make sure cozempic is installed and you're in a Claude Code project."

    messages = load_messages(sess["path"])
    diag = diagnose_session(messages)
    te = diag.get("token_estimate")

    lines = []
    lines.append(f"Session: {sess['session_id'][:36]}")
    lines.append(f"Size: {diag['total_bytes'] / 1024:.1f}KB ({diag['total_messages']} messages)")

    if te:
        tok_str = f"{te.total / 1000:.1f}K" if te.total >= 1000 else str(te.total)
        lines.append(f"Tokens: {tok_str} ({te.method})")
        window_str = f"{te.context_window // 1000}K" if te.context_window < 1_000_000 else f"{te.context_window // 1_000_000}M"
        lines.append(f"Context: {te.context_pct:.0f}% of {window_str} window")
    lines.append("")

    lines.append("Vital Signs:")
    lines.append(f"  Progress ticks: {diag['progress_count']}")
    lines.append(f"  File history snapshots: {diag['file_history_count']}")
    lines.append(f"  System reminders: {diag['reminder_count']}")
    lines.append(f"  Thinking content: {diag['thinking_bytes'] / 1024:.1f}KB")
    lines.append(f"  Tool results: {diag['tool_result_bytes'] / 1024:.1f}KB")
    lines.append("")

    lines.append("Estimated Savings:")
    for rx_name, strategy_names in PRESCRIPTIONS.items():
        new_msgs, _ = run_prescription(messages, strategy_names, {})
        final_bytes = sum(b for _, _, b in new_msgs)
        saved = diag["total_bytes"] - final_bytes
        pct = saved / diag["total_bytes"] * 100 if diag["total_bytes"] > 0 else 0
        lines.append(f"  {rx_name}: ~{saved / 1024:.1f}KB ({pct:.1f}%)")

    return "\n".join(lines)


@mcp.tool()
def estimate_tokens() -> str:
    """Get the current session's token count and context window usage percentage.

    Fast check — reads only the tail of the session file.
    """
    from cozempic.session import find_current_session, load_messages
    from cozempic.tokens import quick_token_estimate, detect_context_window

    sess = find_current_session()
    if not sess:
        return "Could not detect current session."

    # Detect actual context window from session model
    messages = load_messages(sess["path"])
    context_window = detect_context_window(messages)
    window_str = f"{context_window // 1000}K" if context_window < 1_000_000 else f"{context_window // 1_000_000}M"

    tok = quick_token_estimate(sess["path"], context_window=context_window)
    if tok is None:
        return f"Session: {sess['session_id'][:36]}\nSize: {sess['size'] / 1024:.1f}KB\nTokens: unable to estimate"

    pct = round(tok / context_window * 100, 1)
    tok_str = f"{tok / 1000:.1f}K" if tok >= 1000 else str(tok)
    return f"Session: {sess['session_id'][:36]}\nTokens: {tok_str}\nContext: {pct}% of {window_str} window"


@mcp.tool()
def list_sessions() -> str:
    """List all Claude Code sessions with sizes and token estimates."""
    from cozempic.session import find_sessions
    from cozempic.tokens import quick_token_estimate

    sessions = find_sessions()
    if not sessions:
        return "No sessions found."

    lines = [f"{'Session ID':<38} {'Size':>8} {'Tokens':>8} {'Messages':>6} Project"]
    for sess in sorted(sessions, key=lambda s: s["size"], reverse=True):
        sid = sess["session_id"][:36]
        size_kb = f"{sess['size'] / 1024:.0f}KB"
        tok = quick_token_estimate(sess["path"])
        tok_str = f"{tok / 1000:.1f}K" if tok and tok >= 1000 else (str(tok) if tok else "-")
        proj = sess["project"][-30:]
        lines.append(f"{sid:<38} {size_kb:>8} {tok_str:>8} {sess['lines']:>6} {proj}")

    lines.append(f"\nTotal: {len(sessions)} sessions, {sum(s['size'] for s in sessions) / 1024:.0f}KB")
    return "\n".join(lines)


@mcp.tool()
def treat_session(prescription: str = "standard", execute: bool = False) -> str:
    """Run a pruning prescription on the current session.

    Args:
        prescription: Prescription tier — 'gentle', 'standard', or 'aggressive'.
        execute: If False (default), dry-run only. If True, apply changes with backup.
    """
    from cozempic.session import (
        _PruneLock, PruneConflictError, PruneLockError,
        find_current_session, load_messages, load_messages_and_snapshot, save_messages,
    )
    from cozempic.registry import PRESCRIPTIONS
    from cozempic.executor import run_prescription
    from cozempic.tokens import estimate_session_tokens

    sess = find_current_session(strict=execute)
    if not sess:
        msg = "Could not detect current session."
        if execute:
            msg += " Cannot determine session unambiguously — provide an explicit session ID."
        return msg

    if prescription not in PRESCRIPTIONS:
        return f"Unknown prescription '{prescription}'. Options: {', '.join(PRESCRIPTIONS)}"

    path = sess["path"]
    # Read once: snapshot + messages from identical bytes for append-conflict
    # detection (parallel to cmd_treat/cmd_strategy/cmd_reload in cli.py), without
    # a TOCTOU window where a concurrent append lands in both messages and delta.
    messages, _snap = load_messages_and_snapshot(path)
    snapshot = _snap if execute else None
    strategy_names = PRESCRIPTIONS[prescription]

    original_bytes = sum(b for _, _, b in messages)
    pre_te = estimate_session_tokens(messages)

    new_messages, strategy_results = run_prescription(messages, strategy_names, {})
    final_bytes = sum(b for _, _, b in new_messages)
    post_te = estimate_session_tokens(new_messages)

    saved_bytes = original_bytes - final_bytes
    pct = saved_bytes / original_bytes * 100 if original_bytes > 0 else 0

    lines = []
    lines.append(f"Prescription: {prescription}")

    if pre_te.total and post_te.total:
        tok_saved = pre_te.total - post_te.total
        tok_pct = tok_saved / pre_te.total * 100 if pre_te.total > 0 else 0
        # Use the session's DETECTED context window (1M for current models),
        # not a hardcoded 200K. estimate_session_tokens already computed the
        # percentage against the detected window.
        after_pct = post_te.context_pct
        cw = post_te.context_window
        window_str = f"{cw // 1000}K" if cw < 1_000_000 else f"{cw / 1_000_000:.0f}M"
        pre_str = f"{pre_te.total / 1000:.1f}K" if pre_te.total >= 1000 else str(pre_te.total)
        post_str = f"{post_te.total / 1000:.1f}K" if post_te.total >= 1000 else str(post_te.total)
        tok_saved_str = f"{tok_saved / 1000:.1f}K" if tok_saved >= 1000 else str(tok_saved)
        lines.append(f"Before: {pre_str} tokens ({original_bytes / 1024:.1f}KB, {len(messages)} messages)")
        lines.append(f"After: {post_str} tokens ({final_bytes / 1024:.1f}KB, {len(new_messages)} messages)")
        lines.append(f"Freed: {tok_saved_str} tokens ({tok_pct:.1f}%) — {saved_bytes / 1024:.1f}KB")
        lines.append(f"Context: {after_pct}% of {window_str} window")
    else:
        lines.append(f"Before: {original_bytes / 1024:.1f}KB ({len(messages)} messages)")
        lines.append(f"After: {final_bytes / 1024:.1f}KB ({len(new_messages)} messages)")
        lines.append(f"Saved: {saved_bytes / 1024:.1f}KB ({pct:.1f}%)")

    lines.append("")
    lines.append("Strategy Results:")
    for sr in strategy_results:
        sr_saved = sum(a.original_bytes - a.pruned_bytes for a in sr.actions) if sr.actions else 0
        lines.append(f"  {sr.strategy_name}: {sr_saved / 1024:.1f}KB saved — {sr.summary}")

    if execute:
        # Acquire per-session prune lock + pass snapshot for append-conflict
        # detection. Prevents the same data-loss bug that hit cmd_treat in
        # production (where a guard daemon mid-prune would clobber/be
        # clobbered by the manual treat).
        try:
            with _PruneLock(path):
                backup = save_messages(path, new_messages, create_backup=True, snapshot=snapshot)
        except PruneLockError:
            lines.append("")
            lines.append("Aborted: guard daemon is mid-prune. Try again in a few seconds.")
            return "\n".join(lines)
        except PruneConflictError as exc:
            lines.append("")
            lines.append(f"Aborted: session changed mid-prune (Claude wrote new lines). {exc}")
            return "\n".join(lines)
        lines.append("")
        lines.append(f"Treatment applied to {path.name}")
        if backup:
            lines.append(f"Backup: {backup.name}")
    else:
        lines.append("")
        lines.append("DRY RUN — no changes made. Set execute=True to apply.")

    return "\n".join(lines)


@mcp.tool()
def list_strategies() -> str:
    """List all available cleaning strategies and prescriptions."""
    from cozempic.registry import STRATEGIES, PRESCRIPTIONS

    lines = ["Strategies:"]
    lines.append(f"{'Name':<30} {'Tier':<12} {'Expected':>10}  Description")
    for name, info in STRATEGIES.items():
        lines.append(f"{name:<30} {info.tier:<12} {info.expected_savings:>10}  {info.description}")

    lines.append("")
    lines.append("Prescriptions:")
    for rx_name, strategy_names in PRESCRIPTIONS.items():
        lines.append(f"  {rx_name}: {', '.join(strategy_names)}")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
