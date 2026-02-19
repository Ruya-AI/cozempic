#!/usr/bin/env python3
"""Cozempic MCP Server — exposes session diagnostics and treatment as Claude Code tools."""

from __future__ import annotations

from fastmcp import FastMCP

# Ensure all strategies are registered
import cozempic.strategies  # noqa: F401

mcp = FastMCP("Cozempic")


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
        lines.append(f"Context: {te.context_pct:.0f}% of 200K window")
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
    from cozempic.session import find_current_session
    from cozempic.tokens import quick_token_estimate, DEFAULT_CONTEXT_WINDOW

    sess = find_current_session()
    if not sess:
        return "Could not detect current session."

    tok = quick_token_estimate(sess["path"])
    if tok is None:
        return f"Session: {sess['session_id'][:36]}\nSize: {sess['size'] / 1024:.1f}KB\nTokens: unable to estimate"

    pct = round(tok / DEFAULT_CONTEXT_WINDOW * 100, 1)
    tok_str = f"{tok / 1000:.1f}K" if tok >= 1000 else str(tok)
    return f"Session: {sess['session_id'][:36]}\nTokens: {tok_str}\nContext: {pct}% of 200K window"


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
    from cozempic.session import find_current_session, load_messages, save_messages
    from cozempic.registry import PRESCRIPTIONS
    from cozempic.executor import run_prescription
    from cozempic.tokens import estimate_session_tokens

    sess = find_current_session()
    if not sess:
        return "Could not detect current session."

    if prescription not in PRESCRIPTIONS:
        return f"Unknown prescription '{prescription}'. Options: {', '.join(PRESCRIPTIONS)}"

    path = sess["path"]
    messages = load_messages(path)
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
        from cozempic.tokens import DEFAULT_CONTEXT_WINDOW
        tok_saved = pre_te.total - post_te.total
        tok_pct = tok_saved / pre_te.total * 100 if pre_te.total > 0 else 0
        after_pct = round(post_te.total / DEFAULT_CONTEXT_WINDOW * 100, 1)
        pre_str = f"{pre_te.total / 1000:.1f}K" if pre_te.total >= 1000 else str(pre_te.total)
        post_str = f"{post_te.total / 1000:.1f}K" if post_te.total >= 1000 else str(post_te.total)
        tok_saved_str = f"{tok_saved / 1000:.1f}K" if tok_saved >= 1000 else str(tok_saved)
        lines.append(f"Before: {pre_str} tokens ({original_bytes / 1024:.1f}KB, {len(messages)} messages)")
        lines.append(f"After: {post_str} tokens ({final_bytes / 1024:.1f}KB, {len(new_messages)} messages)")
        lines.append(f"Freed: {tok_saved_str} tokens ({tok_pct:.1f}%) — {saved_bytes / 1024:.1f}KB")
        lines.append(f"Context: {after_pct}% of 200K window")
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
        backup = save_messages(path, new_messages, create_backup=True)
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
