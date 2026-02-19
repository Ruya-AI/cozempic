"""Token estimation for Claude Code session files.

Two methods:
1. Exact — read `usage` from last main-chain assistant message.
2. Heuristic — estimate from content characters when no usage data exists.
"""

from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path

from .helpers import get_content_blocks, get_msg_type, text_of
from .types import Message

# Constants
DEFAULT_CONTEXT_WINDOW = 200_000
SYSTEM_OVERHEAD_TOKENS = 21_000

# Model → context window mapping
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-6": 1_000_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
}

# Chars-per-token defaults (conservative)
CHARS_PER_TOKEN_CODE = 3.5
CHARS_PER_TOKEN_PROSE = 4.0
CHARS_PER_TOKEN_DEFAULT = 3.7  # blended default

TokenEstimate = namedtuple(
    "TokenEstimate", ["total", "context_pct", "method", "confidence", "model", "context_window"]
)


def detect_model(messages: list[Message]) -> str | None:
    """Detect the model from the last main-chain assistant message."""
    for _, msg, _ in reversed(messages):
        if get_msg_type(msg) != "assistant":
            continue
        if msg.get("isSidechain"):
            continue
        inner = msg.get("message", {})
        model = inner.get("model", "")
        if model:
            return model
    return None


def detect_context_window(messages: list[Message]) -> int:
    """Detect the context window size from the session's model.

    Returns the context window in tokens. Falls back to DEFAULT_CONTEXT_WINDOW
    if the model is unknown or undetectable.
    """
    model = detect_model(messages)
    if model:
        # Exact match first
        if model in MODEL_CONTEXT_WINDOWS:
            return MODEL_CONTEXT_WINDOWS[model]
        # Prefix match for versioned model IDs (e.g. claude-opus-4-6-20260101)
        for prefix, window in MODEL_CONTEXT_WINDOWS.items():
            if model.startswith(prefix):
                return window
    return DEFAULT_CONTEXT_WINDOW


def _is_sidechain(msg: dict) -> bool:
    """Check if a message belongs to a sidechain (subagent) conversation."""
    return bool(msg.get("isSidechain"))


def _is_context_message(msg: dict) -> bool:
    """Return True if this message contributes to the context window.

    Excludes: progress ticks, file-history-snapshots, sidechain messages,
    and pure-thinking assistant turns.
    """
    mtype = get_msg_type(msg)

    # Non-context message types
    if mtype in ("progress", "file-history-snapshot"):
        return False

    # Sidechain messages don't count toward main context
    if _is_sidechain(msg):
        return False

    # Assistant messages that are pure thinking (no text/tool_use output)
    if mtype == "assistant":
        blocks = get_content_blocks(msg)
        has_output = any(
            b.get("type") in ("text", "tool_use", "tool_result")
            for b in blocks
        )
        if blocks and not has_output:
            return False

    return True


def extract_usage_tokens(messages: list[Message]) -> dict | None:
    """Extract exact token counts from the last main-chain assistant message.

    Returns dict with keys: input_tokens, output_tokens,
    cache_creation_input_tokens, cache_read_input_tokens, total.
    Returns None if no usage data found.
    """
    # Walk backwards to find the last main-chain assistant with usage
    for _, msg, _ in reversed(messages):
        mtype = get_msg_type(msg)
        if mtype != "assistant":
            continue
        if _is_sidechain(msg):
            continue
        if msg.get("_parse_error"):
            continue

        inner = msg.get("message", {})
        usage = inner.get("usage")
        if not usage or not isinstance(usage, dict):
            continue

        input_tok = usage.get("input_tokens", 0)
        output_tok = usage.get("output_tokens", 0)
        cache_create = usage.get("cache_creation_input_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)

        # The cumulative context size is the sum of all input components
        total = input_tok + cache_create + cache_read

        return {
            "input_tokens": input_tok,
            "output_tokens": output_tok,
            "cache_creation_input_tokens": cache_create,
            "cache_read_input_tokens": cache_read,
            "total": total,
        }

    return None


def _estimate_block_chars(block: dict) -> int:
    """Estimate character count for a content block, excluding thinking."""
    btype = block.get("type", "")

    # Thinking blocks are not counted (they're ephemeral)
    if btype == "thinking":
        return 0

    text = text_of(block)
    if text:
        return len(text)

    # tool_use / tool_result: estimate from JSON serialization
    if btype in ("tool_use", "tool_result"):
        try:
            return len(json.dumps(block, separators=(",", ":")))
        except (TypeError, ValueError):
            return 0

    return 0


def estimate_tokens_heuristic(
    messages: list[Message],
    chars_per_token: float = CHARS_PER_TOKEN_DEFAULT,
) -> tuple[int, dict[str, int]]:
    """Estimate tokens from content characters when no usage data exists.

    Returns (total_tokens, breakdown_by_type) where breakdown maps
    message type to estimated token count.
    """
    total_chars = 0
    breakdown: dict[str, int] = {}

    for _, msg, _ in messages:
        if not _is_context_message(msg):
            continue

        mtype = get_msg_type(msg)
        msg_chars = 0

        blocks = get_content_blocks(msg)
        if blocks:
            for block in blocks:
                msg_chars += _estimate_block_chars(block)
        else:
            # Simple message with string content
            inner = msg.get("message", {})
            content = inner.get("content", "")
            if isinstance(content, str):
                msg_chars = len(content)

        breakdown[mtype] = breakdown.get(mtype, 0) + msg_chars
        total_chars += msg_chars

    total_tokens = int(total_chars / chars_per_token) + SYSTEM_OVERHEAD_TOKENS

    # Convert char breakdown to token breakdown
    token_breakdown = {
        mtype: int(chars / chars_per_token)
        for mtype, chars in breakdown.items()
    }

    return total_tokens, token_breakdown


def estimate_session_tokens(messages: list[Message]) -> TokenEstimate:
    """Estimate session tokens, preferring exact data over heuristic.

    Returns a TokenEstimate namedtuple:
      total: estimated total tokens
      context_pct: percentage of context window used (auto-detected per model)
      method: "exact" or "heuristic"
      confidence: "high" (exact) or "medium" (heuristic)
      model: detected model name or None
      context_window: context window size used for % calculation
    """
    model = detect_model(messages)
    context_window = detect_context_window(messages)

    # Try exact first
    usage = extract_usage_tokens(messages)
    if usage is not None:
        total = usage["total"]
        pct = round(total / context_window * 100, 1)
        return TokenEstimate(
            total=total,
            context_pct=pct,
            method="exact",
            confidence="high",
            model=model,
            context_window=context_window,
        )

    # Fall back to heuristic
    total, _ = estimate_tokens_heuristic(messages)
    pct = round(total / context_window * 100, 1)
    return TokenEstimate(
        total=total,
        context_pct=pct,
        method="heuristic",
        confidence="medium",
        model=model,
        context_window=context_window,
    )


def quick_token_estimate(path: Path) -> int | None:
    """Fast token estimate by reading only the tail of a JSONL file.

    Reads the last 50KB and tries to extract usage from the last assistant
    message. Returns the token total, or None if no usage data found.
    """
    try:
        file_size = path.stat().st_size
        read_size = min(file_size, 50 * 1024)

        with open(path, "rb") as f:
            if file_size > read_size:
                f.seek(file_size - read_size)
            raw = f.read().decode("utf-8", errors="replace")

        # Parse lines from the tail
        lines = raw.strip().split("\n")
        # The first line may be partial if we seeked, skip it
        if file_size > read_size:
            lines = lines[1:]

        # Walk backwards looking for an assistant message with usage
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if get_msg_type(msg) != "assistant":
                continue
            if msg.get("isSidechain"):
                continue

            inner = msg.get("message", {})
            usage = inner.get("usage")
            if not usage or not isinstance(usage, dict):
                continue

            input_tok = usage.get("input_tokens", 0)
            cache_create = usage.get("cache_creation_input_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)
            return input_tok + cache_create + cache_read

    except (OSError, UnicodeDecodeError):
        pass

    return None


def calibrate_ratio(messages: list[Message]) -> float | None:
    """Calculate the actual chars-per-token ratio for a session.

    Requires both exact usage data and content to compare against.
    Returns the ratio, or None if calibration isn't possible.
    """
    usage = extract_usage_tokens(messages)
    if usage is None:
        return None

    exact_tokens = usage["total"]
    if exact_tokens <= SYSTEM_OVERHEAD_TOKENS:
        return None

    # Count content chars (same way as heuristic)
    total_chars = 0
    for _, msg, _ in messages:
        if not _is_context_message(msg):
            continue
        blocks = get_content_blocks(msg)
        if blocks:
            for block in blocks:
                total_chars += _estimate_block_chars(block)
        else:
            inner = msg.get("message", {})
            content = inner.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)

    content_tokens = exact_tokens - SYSTEM_OVERHEAD_TOKENS
    if content_tokens <= 0:
        return None

    return round(total_chars / content_tokens, 2)
