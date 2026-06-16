"""Token estimation for Claude Code session files.

Two methods:
1. Exact — read `usage` from last main-chain assistant message.
2. Heuristic — estimate from content characters when no usage data exists.
"""

from __future__ import annotations

import json
import os
from collections import namedtuple
from pathlib import Path

from .helpers import get_content_blocks, get_msg_type, text_of
from .types import Message

# Constants
DEFAULT_CONTEXT_WINDOW = 1_000_000  # All current Claude models are 1M. Pro plan users can override with COZEMPIC_CONTEXT_WINDOW=200000.
SYSTEM_OVERHEAD_TOKENS = 21_000

# Upper bounds for env-var overrides.
# A context-window override above MAX_CONTEXT_WINDOW can't reflect a real model
# and would silently disable the guard (every pct = total/window rounds toward 0).
# 4M = 4x the current 1M max — generous headroom — above which we reject and
# fall back to model-detected window so the guard keeps firing.
MAX_CONTEXT_WINDOW = 4_000_000
# A system-overhead override above the default context window (1M) is a coarse
# absurdity ceiling: it catches huge-int DoS / fat-finger overrides that would
# zero out usable context, not a per-user-plan guarantee (a Pro user on a 200K
# window can still set overhead=900K; the ceiling is intentionally loose).
# The bound is strict-greater-than (> maximum), so exactly 1M is still accepted.
MAX_SYSTEM_OVERHEAD_TOKENS = DEFAULT_CONTEXT_WINDOW

# 4-tier pruning thresholds as fractions of context window
DEFAULT_SOFT_TOKEN_PCT = 0.25   # 25% — gentle file maintenance, no reload
DEFAULT_HARD1_TOKEN_PCT = 0.55  # 55% — standard prune + reload
DEFAULT_HARD2_TOKEN_PCT = 0.80  # 80% — aggressive prune + reload (emergency)
DEFAULT_HARD_TOKEN_PCT = 0.55   # Alias for backward compat (guard uses this)


def get_system_overhead_tokens() -> int:
    """Get system overhead token estimate, checking env var override.

    Sessions with heavy rules files, MCP servers, and tool schemas can
    have 30K-40K+ tokens of system overhead. The default (21K) is
    conservative for lightweight sessions. Override with
    COZEMPIC_SYSTEM_OVERHEAD_TOKENS env var or --system-overhead-tokens flag.
    """
    from ._validation import parse_env_non_negative_int
    override = parse_env_non_negative_int(
        "COZEMPIC_SYSTEM_OVERHEAD_TOKENS", maximum=MAX_SYSTEM_OVERHEAD_TOKENS
    )
    if override is not None:
        return override
    return SYSTEM_OVERHEAD_TOKENS


def default_token_thresholds(context_window: int = DEFAULT_CONTEXT_WINDOW) -> tuple[int, int]:
    """Compute default hard and soft token thresholds from context window.

    4-tier system:
      Soft (25%):  gentle file maintenance, no reload (preemptive cleanup)
      Hard1 (55%): standard prune + reload (first real prune)
      Hard2 (80%): aggressive prune + reload (emergency, before CC compaction)
      User (90%):  user-triggered aggressive (manual last resort)

    Returns (hard_threshold, soft_threshold) in tokens.
    For backward compat, returns the hard1 (55%) as "hard" and soft (25%) as "soft".
    """
    hard = int(context_window * DEFAULT_HARD1_TOKEN_PCT)
    soft = int(context_window * DEFAULT_SOFT_TOKEN_PCT)
    return hard, soft


def default_token_thresholds_4tier(context_window: int = DEFAULT_CONTEXT_WINDOW) -> tuple[int, int, int]:
    """Compute all 4-tier thresholds. Returns (soft, hard1, hard2) in tokens."""
    soft = int(context_window * DEFAULT_SOFT_TOKEN_PCT)
    hard1 = int(context_window * DEFAULT_HARD1_TOKEN_PCT)
    hard2 = int(context_window * DEFAULT_HARD2_TOKEN_PCT)
    return soft, hard1, hard2

# Every known Haiku generation ships with a 200K context window; revisit if a
# future Haiku launches with a larger window.
HAIKU_CONTEXT_WINDOW = 200_000

# Model → context window mapping
# Claude Code does NOT append "[1m]" to model IDs in the JSONL — the model
# field always contains the base ID (e.g., "claude-opus-4-7"). 1M context is
# the standard for current models on Max plans, so we default 4.5/4.6 to 1M.
# Users on Pro (200K) can override with COZEMPIC_CONTEXT_WINDOW=200000.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Current models — default 1M (standard for Claude Code Max plans)
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-opus-4-5": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-sonnet-4-5": 1_000_000,
    # Haiku — 200K (not available on 1M in Claude Code)
    "claude-haiku-4-5": HAIKU_CONTEXT_WINDOW,
    # Older models — 200K
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": HAIKU_CONTEXT_WINDOW,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": HAIKU_CONTEXT_WINDOW,
}


def get_context_window_override() -> int | None:
    """Check for user override via COZEMPIC_CONTEXT_WINDOW env var.

    Requires a strictly-positive integer. Zero previously hit the
    `if val:` falsy-trap and was silently ignored; negative values
    propagated into `context_pct = total / cw` producing negative
    percentages. Both now emit a stderr warning and fall back to None
    (triggering model-based detection at detect_context_window).
    """
    from ._validation import parse_env_positive_int
    return parse_env_positive_int("COZEMPIC_CONTEXT_WINDOW", maximum=MAX_CONTEXT_WINDOW)

# Chars-per-token defaults, calibrated against live Claude Code JSONL.
# Measured 3.08–3.27 chars/token on real sessions: JSON keys, UUIDs, tool
# arguments and code are far denser than prose, so the old 3.7 blended default
# undercounted the heuristic path by ~15-20%. These are used ONLY by the
# heuristic fallback — sessions with a usage block use the exact recorded
# count (input + cache_creation + cache_read + output), which is authoritative.
CHARS_PER_TOKEN_CODE = 3.0
CHARS_PER_TOKEN_PROSE = 3.5
CHARS_PER_TOKEN_DEFAULT = 3.1  # blended default


def get_chars_per_token() -> float:
    """Resolve the heuristic chars-per-token divisor.

    Honors the ``COZEMPIC_CHARS_PER_TOKEN`` env override (positive float,
    clamped to a sane 1.0–20.0 range); otherwise returns the calibrated
    default. Affects only the heuristic fallback — exact usage-based counts
    ignore it entirely.
    """
    raw = os.environ.get("COZEMPIC_CHARS_PER_TOKEN")
    if raw:
        try:
            val = float(raw)
        except ValueError:
            return CHARS_PER_TOKEN_DEFAULT
        if 1.0 <= val <= 20.0:
            return val
    return CHARS_PER_TOKEN_DEFAULT

TokenEstimate = namedtuple(
    "TokenEstimate", ["total", "context_pct", "method", "confidence", "model", "context_window"]
)


def detect_model(messages: list[Message]) -> str | None:
    """Detect the model from the last main-chain assistant message.

    Skips `<synthetic>` model values — those are injected by Claude Code for
    compaction summaries, system messages, and other non-API-generated entries.
    Keeping them would cause fallback to the wrong context window.
    """
    for _, msg, _ in reversed(messages):
        if get_msg_type(msg) != "assistant":
            continue
        if msg.get("isSidechain"):
            continue
        inner = _inner_dict(msg)
        model = inner.get("model", "")
        if model and model != "<synthetic>":
            return model
    return None


def detect_context_window(messages: list[Message]) -> int:
    """Detect the context window size from the session's model.

    Priority:
    1. COZEMPIC_CONTEXT_WINDOW env var (user override)
    2. Model detection from session data (exact match, then prefix match)
    3. DEFAULT_CONTEXT_WINDOW (1M)

    Match order:
    - "claude-opus-4-6" → 1M (exact match)
    - "claude-opus-4-6-20260301" → 1M (prefix match for versioned IDs)
    - "claude-haiku-9" (unknown Haiku) → 200K (family fallback, not the 1M default)
    - "claude-future-99" (unknown family) → 1M (DEFAULT)

    Claude Code writes the base model ID with no "[1m]" suffix (see the
    MODEL_CONTEXT_WINDOWS note above), so a bracketed ID — should one ever appear
    — is resolved by the same prefix logic
    ("claude-opus-4-6[1m]".startswith("claude-opus-4-6")).
    """
    override = get_context_window_override()
    if override:
        return override

    model = detect_model(messages)
    if model:
        # Exact match, then prefix match for versioned IDs.
        if model in MODEL_CONTEXT_WINDOWS:
            return MODEL_CONTEXT_WINDOWS[model]
        for prefix, window in MODEL_CONTEXT_WINDOWS.items():
            if model.startswith(prefix):
                return window
        # Family fallback for an unknown version: every known Haiku generation is
        # 200K, so an unrecognised Haiku must NOT fall through to the 1M default
        # (a 5x over-estimate that mis-times every guard tier on a real session).
        # Segment-match (split on "-") rather than substring to avoid spurious hits
        # on hypothetical composite names like "claude-opus-haiku-distill".
        # A model that genuinely can't be resolved defaults to 200K — the
        # conservative direction (smaller window → guard fires earlier = safe).
        if "haiku" in model.lower().split("-"):
            return HAIKU_CONTEXT_WINDOW

    return DEFAULT_CONTEXT_WINDOW


def _as_int(value) -> int:
    """Coerce a JSONL `usage` field to a non-negative int, tolerating junk.

    A malformed transcript can carry a present-but-null/string/float usage value
    (e.g. ``"input_tokens": null``). The bare ``.get(k, 0)`` default only covers a
    MISSING key, so ``None + 0`` (or ``"x" + 0``) raises TypeError — and that
    escapes the guard daemon's per-cycle loop (whose only handler is
    KeyboardInterrupt), killing the daemon with no respawn. Coerce defensively:
    bool/None/str/garbage -> 0, float -> int, negative -> 0.
    """
    if isinstance(value, bool):  # bool is an int subclass — treat True/False as 0
        return 0
    if isinstance(value, int):
        return value if value > 0 else 0
    if isinstance(value, float):
        # inf/nan are valid JSON numbers (1e999 -> inf, json accepts NaN/Infinity)
        # and int(inf) raises OverflowError / int(nan) raises ValueError — which
        # would escape into the guard loop. Treat non-finite as 0.
        import math
        if not math.isfinite(value) or value <= 0:
            return 0
        return int(value)
    return 0


def _inner_dict(msg: dict) -> dict:
    """The message's inner dict, or {} if 'message' is missing OR a non-dict.

    `msg.get("message", {})` only defaults a MISSING key — a present-but-non-dict
    "message" (a plain string, which occurs in real JSONL) makes the following
    `.get()` raise AttributeError and (pre-fix) escape into the guard loop. Coerce
    a non-dict to {} so every message-access site is crash-safe."""
    inner = msg.get("message")
    return inner if isinstance(inner, dict) else {}


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
        # isinstance guard: get_content_blocks returns content VERBATIM (the round-3
        # revert that stopped write-path data loss), so a non-dict content element
        # reaches here. Without the guard b.get(...) raises AttributeError, which is
        # UNHANDLED in cmd_treat (aborts the prune → user marches into auto-compaction)
        # and a respawn-storm in the guard cycle (R4 findings is-context-message /
        # nondict-block-elem token crash).
        has_output = any(
            b.get("type") in ("text", "tool_use", "tool_result")
            for b in blocks
            if isinstance(b, dict)
        )
        if blocks and not has_output:
            return False

    return True


def extract_usage_tokens(messages: list[Message]) -> dict | None:
    """Extract exact token counts from the last main-chain assistant message.

    Returns dict with keys: input_tokens, output_tokens,
    cache_creation_input_tokens, cache_read_input_tokens, total.
    Returns None if no usage data found.

    Skips `<synthetic>` model messages — their usage blocks contain all zeros,
    which would make the guard think the session is empty.
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

        inner = _inner_dict(msg)
        # Skip synthetic messages — their usage is all zeros
        if inner.get("model") == "<synthetic>":
            continue
        usage = inner.get("usage")
        if not usage or not isinstance(usage, dict):
            continue

        input_tok = _as_int(usage.get("input_tokens", 0))
        output_tok = _as_int(usage.get("output_tokens", 0))
        cache_create = _as_int(usage.get("cache_creation_input_tokens", 0))
        cache_read = _as_int(usage.get("cache_read_input_tokens", 0))

        # The cumulative context size is the sum of all token components
        total = input_tok + cache_create + cache_read + output_tok

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
    # A content array can legally hold a bare string/number (get_content_blocks now
    # returns elements verbatim to avoid write-path data loss). This read-only path
    # is OUTSIDE the executor's per-strategy isolation, so coerce here.
    if not isinstance(block, dict):
        return len(block) if isinstance(block, str) else len(json.dumps(block, separators=(",", ":")))
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
    chars_per_token: float | None = None,
) -> tuple[int, dict[str, int]]:
    """Estimate tokens from content characters when no usage data exists.

    Returns (total_tokens, breakdown_by_type) where breakdown maps
    message type to estimated token count. When ``chars_per_token`` is not
    given, the calibrated default (env-overridable) is used.
    """
    if chars_per_token is None:
        chars_per_token = get_chars_per_token()
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
            inner = _inner_dict(msg)
            content = inner.get("content", "")
            if isinstance(content, str):
                msg_chars = len(content)

        breakdown[mtype] = breakdown.get(mtype, 0) + msg_chars
        total_chars += msg_chars

    total_tokens = int(total_chars / chars_per_token) + get_system_overhead_tokens()

    # Convert char breakdown to token breakdown
    token_breakdown = {
        mtype: int(chars / chars_per_token)
        for mtype, chars in breakdown.items()
    }

    return total_tokens, token_breakdown


def estimate_session_tokens(
    messages: list[Message],
    pre_calibrated_ratio: float | None = None,
) -> TokenEstimate:
    """Estimate session tokens, preferring exact data over heuristic.

    Args:
        messages: session messages to estimate
        pre_calibrated_ratio: chars-per-token ratio calibrated from a prior
            version of the same session (e.g. before metadata-strip removed
            usage fields). When provided, this is used instead of trying to
            re-calibrate from messages that no longer have usage data.

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

    # Fall back to heuristic — prefer pre-calibrated ratio (survives metadata-strip),
    # then try to calibrate from current messages, then use bare default.
    ratio = pre_calibrated_ratio or calibrate_ratio(messages)
    if ratio is not None:
        total, _ = estimate_tokens_heuristic(messages, chars_per_token=ratio)
    else:
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


def quick_token_estimate(path: Path, context_window: int = DEFAULT_CONTEXT_WINDOW) -> int | None:
    """Fast token estimate by reading the tail of a JSONL file.

    Extracts usage from the last main-chain assistant message. Starts with a
    small tail (50KB for 200K, 100KB for 1M) and GROWS it — up to the whole
    file — when no usage frame is found there. A long trailing run of
    usage-less lines (streaming partials, attachments, queued operations) can
    push the last real ``usage`` block past a fixed window; the old fixed-tail
    read then returned None, blanking ``cozempic current`` AND disabling every
    token-based guard threshold (each gated on ``current_tokens is not None``),
    silently degrading the guard to MB-only thresholds on long sessions.

    Returns the token total, or None only when the file genuinely contains no
    usage data anywhere (e.g. a brand-new or heuristic-only session).
    """
    try:
        file_size = path.stat().st_size
        base = (100 if context_window >= 1_000_000 else 50) * 1024
        # Progressive tail sizes, always ending with the full file so a usage
        # frame is found wherever it sits. Dedup keeps small files cheap.
        sizes: list[int] = []
        for candidate in (base, base * 8, base * 64, file_size):
            s = min(candidate, file_size)
            if s not in sizes:
                sizes.append(s)

        for read_size in sizes:
            with open(path, "rb") as f:
                if file_size > read_size:
                    f.seek(file_size - read_size)
                raw = f.read().decode("utf-8", errors="replace")

            lines = raw.strip().split("\n")
            # The first line may be partial if we seeked into the middle.
            if file_size > read_size:
                lines = lines[1:]

            # Walk backwards looking for an assistant message with usage.
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

                inner = _inner_dict(msg)
                if inner.get("model") == "<synthetic>":
                    continue
                usage = inner.get("usage")
                if not usage or not isinstance(usage, dict):
                    continue

                input_tok = _as_int(usage.get("input_tokens", 0))
                output_tok = _as_int(usage.get("output_tokens", 0))
                cache_create = _as_int(usage.get("cache_creation_input_tokens", 0))
                cache_read = _as_int(usage.get("cache_read_input_tokens", 0))
                return input_tok + cache_create + cache_read + output_tok
            # No usage in this tail — grow to the next (larger) size and retry.

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
    overhead = get_system_overhead_tokens()
    if exact_tokens <= overhead:
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
            inner = _inner_dict(msg)
            content = inner.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)

    content_tokens = exact_tokens - overhead
    if content_tokens <= 0:
        return None

    return round(total_chars / content_tokens, 2)
