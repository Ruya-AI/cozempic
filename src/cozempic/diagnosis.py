"""Session diagnosis — analyze bloat sources in a session."""

from __future__ import annotations

import json
import re

from .helpers import get_dict_blocks, get_msg_type, text_of
from .tokens import estimate_session_tokens, extract_usage_tokens
from .types import Message


def diagnose_session(messages: list[Message]) -> dict:
    """Analyze a session and return bloat breakdown."""
    total_bytes = sum(b for _, _, b in messages)
    total_messages = len(messages)

    type_stats: dict[str, dict] = {}
    largest_messages: list[tuple[int, int, str, int]] = []

    for pos, (idx, msg, size) in enumerate(messages):
        mtype = get_msg_type(msg)
        if mtype not in type_stats:
            type_stats[mtype] = {"count": 0, "bytes": 0}
        type_stats[mtype]["count"] += 1
        type_stats[mtype]["bytes"] += size
        largest_messages.append((size, idx, mtype, pos))

    largest_messages.sort(reverse=True)

    thinking_bytes = 0
    signature_bytes = 0
    tool_result_bytes = 0
    progress_count = 0
    file_history_count = 0
    reminder_count = 0

    reminder_pattern = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)

    for pos, (idx, msg, size) in enumerate(messages):
        mtype = get_msg_type(msg)
        if mtype == "progress":
            progress_count += 1
        if mtype == "file-history-snapshot":
            file_history_count += 1

        for block in get_dict_blocks(msg):  # read-only: dict-only iterator (R5 non-dict leak)
            btype = block.get("type", "")
            if btype == "thinking":
                thinking_bytes += len(json.dumps(block.get("thinking", "")).encode("utf-8"))
                sig = block.get("signature", "")
                if isinstance(sig, str):
                    signature_bytes += len(sig.encode("utf-8", "surrogatepass"))
            elif btype == "tool_result":
                content = block.get("content", "")
                if isinstance(content, str):
                    # surrogatepass (NOT surrogateescape): a content string may carry a
                    # LONE HIGH surrogate (a JSON-escaped \ud83d CC emits) which
                    # surrogateescape can't encode -> crash (R6). surrogatepass encodes
                    # every surrogate; this is only a byte-count estimate so WTF-8 width is fine.
                    tool_result_bytes += len(content.encode("utf-8", "surrogatepass"))
                elif isinstance(content, list):
                    tool_result_bytes += len(json.dumps(content).encode("utf-8"))

            text = text_of(block)
            if text:
                reminder_count += len(reminder_pattern.findall(text))

    token_estimate = estimate_session_tokens(messages)

    # Cache efficiency metrics (#45)
    usage = extract_usage_tokens(messages)
    cache_stats = None
    if usage:
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_create = usage.get("cache_creation_input_tokens", 0)
        cache_total = cache_read + cache_create
        cache_hit_rate = round(cache_read / cache_total * 100, 1) if cache_total > 0 else 0.0
        cache_stats = {
            "cache_read_tokens": cache_read,
            "cache_creation_tokens": cache_create,
            "cache_total_tokens": cache_total,
            "cache_hit_rate": cache_hit_rate,
        }

    return {
        "total_bytes": total_bytes,
        "total_messages": total_messages,
        "type_stats": type_stats,
        "largest_messages": largest_messages[:10],
        "thinking_bytes": thinking_bytes,
        "signature_bytes": signature_bytes,
        "tool_result_bytes": tool_result_bytes,
        "progress_count": progress_count,
        "file_history_count": file_history_count,
        "reminder_count": reminder_count,
        "token_estimate": token_estimate,
        "cache_stats": cache_stats,
    }
