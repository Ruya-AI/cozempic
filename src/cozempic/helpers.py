"""Shared helper functions for message inspection and manipulation."""

from __future__ import annotations

import copy
import json


def msg_bytes(msg: dict) -> int:
    """Calculate the serialized byte size of a message."""
    return len(json.dumps(msg, separators=(",", ":")).encode("utf-8"))


def get_msg_type(msg: dict) -> str:
    """Get the type field from a message."""
    return msg.get("type", "unknown")


def get_content_blocks(msg: dict) -> list[dict]:
    """Extract content blocks from a message's inner message object."""
    m = msg.get("message", {})
    content = m.get("content", [])
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return []


def content_block_bytes(block: dict) -> int:
    """Calculate the serialized byte size of a content block."""
    return len(json.dumps(block, separators=(",", ":")).encode("utf-8"))


def set_content_blocks(msg: dict, blocks: list[dict]) -> dict:
    """Return a deep copy of msg with content blocks replaced."""
    msg = copy.deepcopy(msg)
    if "message" in msg:
        msg["message"]["content"] = blocks
    return msg


def shell_quote(s: str) -> str:
    """Single-quote a string for shell use."""
    return "'" + s.replace("'", "'\\''") + "'"


def text_of(block: dict) -> str:
    """Get the text content of a content block, handling all block types."""
    result = block.get("text", "") or block.get("thinking", "") or block.get("content", "")
    if isinstance(result, list):
        return " ".join(
            sub.get("text", "") for sub in result if isinstance(sub, dict)
        )
    if not isinstance(result, str):
        return ""
    return result
