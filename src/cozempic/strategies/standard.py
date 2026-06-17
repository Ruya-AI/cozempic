"""Standard-tier strategies: recommended pruning with cross-message correlation."""

from __future__ import annotations

import hashlib
import json
import re

from ..helpers import get_content_blocks, get_msg_type, hashable_str, is_protected, msg_bytes, set_content_blocks, text_of
from ..registry import strategy
from ..types import Message, PruneAction, StrategyResult
from ._config import coerce_choice, coerce_non_negative_int, coerce_ordered_pair

_THINKING_MODES: tuple[str, ...] = ("remove", "truncate", "signature-only")


@strategy("thinking-blocks", "Truncate or remove thinking/signature blocks", "standard", "2-5%")
def strategy_thinking_blocks(messages: list[Message], config: dict) -> StrategyResult:
    """Remove or truncate thinking blocks and signatures from assistant messages.

    Modes (via config['thinking_mode']):
        'remove'         - Remove thinking blocks entirely (default)
        'truncate'       - Keep first 200 chars of thinking
        'signature-only' - Only strip signature fields
    """
    mode = coerce_choice(config, "thinking_mode", _THINKING_MODES, default="remove")
    actions: list[PruneAction] = []
    total_orig = sum(b for _, _, b in messages)
    total_pruned = 0
    replaced = 0

    for pos, (idx, msg, size) in enumerate(messages):
        if is_protected(msg):
            continue
        if get_msg_type(msg) != "assistant":
            continue

        blocks = get_content_blocks(msg)
        if not blocks:
            continue

        new_blocks = []
        changed = False
        for block in blocks:
            btype = block.get("type", "")
            if btype == "thinking":
                changed = True
                if mode == "remove":
                    continue
                elif mode == "truncate":
                    thinking = block.get("thinking", "")
                    new_block = {k: v for k, v in block.items() if k != "signature"}
                    if len(thinking) > 200:
                        new_block["thinking"] = thinking[:200] + "...[truncated]"
                    new_blocks.append(new_block)
                elif mode == "signature-only":
                    new_block = {k: v for k, v in block.items() if k != "signature"}
                    new_blocks.append(new_block)
                    changed = new_block != block
            else:
                if "signature" in block:
                    changed = True
                    new_blocks.append({k: v for k, v in block.items() if k != "signature"})
                else:
                    new_blocks.append(block)

        if changed:
            new_msg = set_content_blocks(msg, new_blocks)
            new_size = msg_bytes(new_msg)
            saved = size - new_size
            if saved > 0:
                actions.append(PruneAction(
                    line_index=idx,
                    action="replace",
                    reason=f"thinking-blocks ({mode})",
                    original_bytes=size,
                    pruned_bytes=new_size,
                    replacement=new_msg,
                ))
                total_pruned += saved
                replaced += 1

    return StrategyResult(
        strategy_name="thinking-blocks",
        actions=actions,
        original_bytes=total_orig,
        pruned_bytes=total_pruned,
        messages_affected=replaced,
        messages_removed=0,
        messages_replaced=replaced,
        summary=f"Processed {replaced} thinking blocks (mode={mode})",
    )


@strategy("tool-output-trim", "Trim large tool_result blocks (>8KB or >100 lines)", "standard", "1-8%")
def strategy_tool_output_trim(messages: list[Message], config: dict) -> StrategyResult:
    """Trim oversized tool results while preserving structure."""
    max_bytes = coerce_non_negative_int(config, "tool_output_max_bytes", default=8192)
    max_lines = coerce_non_negative_int(config, "tool_output_max_lines", default=100)

    actions: list[PruneAction] = []
    total_orig = sum(b for _, _, b in messages)
    total_pruned = 0
    replaced = 0

    # T2.3: Collect tool IDs already summarized by microcompact — don't trim those
    compacted_tool_ids: set[str] = set()
    for _, msg, _ in messages:
        if msg.get("type") == "system" and msg.get("subtype") == "microcompact_boundary":
            _ctids = msg.get("compactedToolIds", [])
            for tid in (_ctids if isinstance(_ctids, list) else []):
                if isinstance(tid, str):  # unhashable/non-str element -> skip (R6 crash class)
                    compacted_tool_ids.add(tid)

    for pos, (idx, msg, size) in enumerate(messages):
        if is_protected(msg):
            continue
        blocks = get_content_blocks(msg)
        if not blocks:
            continue

        new_blocks = []
        changed = False
        for block in blocks:
            if not isinstance(block, dict):  # non-dict element preserved, never .get()'d (R6)
                new_blocks.append(block)
                continue
            if block.get("type") == "tool_result":
                # Skip tool results already microcompacted
                tool_use_id = hashable_str(block.get("tool_use_id"))
                if tool_use_id and tool_use_id in compacted_tool_ids:
                    new_blocks.append(block)
                    continue
                content = block.get("content", "")
                if isinstance(content, str):
                    content_bytes = len(content.encode("utf-8"))
                    content_lines = content.count("\n") + 1
                    if content_bytes > max_bytes or content_lines > max_lines:
                        lines = content.split("\n")
                        if len(lines) > max_lines:
                            keep = max_lines // 2
                            trimmed = (
                                lines[:keep]
                                + [f"\n... [{len(lines) - max_lines} lines trimmed by cozempic] ...\n"]
                                + lines[-keep:]
                            )
                            new_content = "\n".join(trimmed)
                        else:
                            half = max_bytes // 2
                            new_content = (
                                content[:half]
                                + f"\n... [{content_bytes - max_bytes} bytes trimmed by cozempic] ...\n"
                                + content[-half:]
                            )
                        new_blocks.append({**block, "content": new_content})
                        changed = True
                        continue
                elif isinstance(content, list):
                    block_json = json.dumps(content, separators=(",", ":"))
                    if len(block_json.encode("utf-8")) > max_bytes:
                        trimmed_content = []
                        for sub in content:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                text = sub.get("text", "")
                                if isinstance(text, str) and len(text.encode("utf-8", "surrogateescape")) > max_bytes:
                                    half = max_bytes // 2
                                    sub = {**sub, "text": text[:half] + "\n...[trimmed by cozempic]...\n" + text[-half:]}
                            trimmed_content.append(sub)
                        new_blocks.append({**block, "content": trimmed_content})
                        changed = True
                        continue
            new_blocks.append(block)

        if changed:
            new_msg = set_content_blocks(msg, new_blocks)
            new_size = msg_bytes(new_msg)
            saved = size - new_size
            if saved > 0:
                actions.append(PruneAction(
                    line_index=idx,
                    action="replace",
                    reason="tool-output-trim",
                    original_bytes=size,
                    pruned_bytes=new_size,
                    replacement=new_msg,
                ))
                total_pruned += saved
                replaced += 1

    return StrategyResult(
        strategy_name="tool-output-trim",
        actions=actions,
        original_bytes=total_orig,
        pruned_bytes=total_pruned,
        messages_affected=replaced,
        messages_removed=0,
        messages_replaced=replaced,
        summary=f"Trimmed {replaced} oversized tool outputs",
    )


@strategy("stale-reads", "Remove file reads superseded by later edits", "standard", "0.5-2%")
def strategy_stale_reads(messages: list[Message], config: dict) -> StrategyResult:
    """If a file was read and then later edited/written, the read result is stale."""
    actions: list[PruneAction] = []
    total_orig = sum(b for _, _, b in messages)
    total_pruned = 0

    file_events: dict[str, list[tuple[int, str, int]]] = {}

    for pos, (idx, msg, size) in enumerate(messages):
        if is_protected(msg):
            continue
        for block in get_content_blocks(msg):
            if block.get("type") == "tool_use":
                tool_name = block.get("name", "")
                tool_input = block.get("input", {})
                if tool_name in ("Read", "read"):
                    fp = tool_input.get("file_path", "")
                    if fp:
                        file_events.setdefault(fp, []).append((pos, "read", idx))
                elif tool_name in ("Edit", "edit", "Write", "write"):
                    fp = tool_input.get("file_path", "")
                    if fp:
                        file_events.setdefault(fp, []).append((pos, "edit", idx))

    stale_read_positions: set[int] = set()
    for fp, events in file_events.items():
        events.sort(key=lambda x: x[0])
        for i, (pos, etype, idx) in enumerate(events):
            if etype == "read":
                for j in range(i + 1, len(events)):
                    if events[j][1] == "edit":
                        stale_read_positions.add(pos)
                        break

    for pos, (idx, msg, size) in enumerate(messages):
        if pos not in stale_read_positions:
            continue
        for block in get_content_blocks(msg):
            if block.get("type") == "tool_use" and block.get("name") in ("Read", "read"):
                tool_use_id = block.get("id", "")
                if not tool_use_id:
                    continue
                for fpos in range(pos + 1, min(pos + 5, len(messages))):
                    fidx, fmsg, fsize = messages[fpos]
                    for fb in get_content_blocks(fmsg):
                        if fb.get("type") == "tool_result" and fb.get("tool_use_id") == tool_use_id:
                            content = fb.get("content", "")
                            if isinstance(content, str) and len(content) > 500:
                                new_fb = {**fb, "content": "[stale read - file was later edited, trimmed by cozempic]"}
                                new_blocks = []
                                did_replace = False
                                for ob in get_content_blocks(fmsg):
                                    if ob.get("type") == "tool_result" and ob.get("tool_use_id") == tool_use_id and not did_replace:
                                        new_blocks.append(new_fb)
                                        did_replace = True
                                    else:
                                        new_blocks.append(ob)
                                new_msg = set_content_blocks(fmsg, new_blocks)
                                new_size = msg_bytes(new_msg)
                                saved = fsize - new_size
                                if saved > 0:
                                    actions.append(PruneAction(
                                        line_index=fidx,
                                        action="replace",
                                        reason="stale-read (file later edited)",
                                        original_bytes=fsize,
                                        pruned_bytes=new_size,
                                        replacement=new_msg,
                                    ))
                                    total_pruned += saved

    replaced = len(actions)
    return StrategyResult(
        strategy_name="stale-reads",
        actions=actions,
        original_bytes=total_orig,
        pruned_bytes=total_pruned,
        messages_affected=replaced,
        messages_removed=0,
        messages_replaced=replaced,
        summary=f"Trimmed {replaced} stale file read results",
    )


@strategy("system-reminder-dedup", "Deduplicate repeated <system-reminder> tags", "standard", "0.1-3%")
def strategy_system_reminder_dedup(messages: list[Message], config: dict) -> StrategyResult:
    """Remove duplicate system-reminder content, keeping only the first occurrence."""
    actions: list[PruneAction] = []
    total_orig = sum(b for _, _, b in messages)
    total_pruned = 0
    replaced = 0

    reminder_pattern = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
    seen_hashes: set[str] = set()

    for pos, (idx, msg, size) in enumerate(messages):
        if is_protected(msg):
            continue
        blocks = get_content_blocks(msg)
        if not blocks:
            continue

        new_blocks = []
        changed = False
        for block in blocks:
            if block.get("type") in ("text", "tool_result"):
                text = block.get("text", "") or (block.get("content", "") if isinstance(block.get("content"), str) else "")
                if not text:
                    new_blocks.append(block)
                    continue

                reminders = reminder_pattern.findall(text)
                if reminders:
                    new_text = text
                    for reminder in reminders:
                        h = hashlib.md5(reminder.encode()).hexdigest()
                        if h in seen_hashes:
                            new_text = new_text.replace(reminder, "")
                            changed = True
                        else:
                            seen_hashes.add(h)

                    if changed:
                        new_text = re.sub(r"\n{3,}", "\n\n", new_text).strip()
                        if block.get("type") == "text":
                            new_blocks.append({**block, "text": new_text})
                        elif block.get("type") == "tool_result" and isinstance(block.get("content"), str):
                            new_blocks.append({**block, "content": new_text})
                        else:
                            new_blocks.append(block)
                    else:
                        new_blocks.append(block)
                else:
                    new_blocks.append(block)
            else:
                new_blocks.append(block)

        if changed:
            new_msg = set_content_blocks(msg, new_blocks)
            new_size = msg_bytes(new_msg)
            saved = size - new_size
            if saved > 0:
                actions.append(PruneAction(
                    line_index=idx,
                    action="replace",
                    reason="system-reminder-dedup",
                    original_bytes=size,
                    pruned_bytes=new_size,
                    replacement=new_msg,
                ))
                total_pruned += saved
                replaced += 1

    return StrategyResult(
        strategy_name="system-reminder-dedup",
        actions=actions,
        original_bytes=total_orig,
        pruned_bytes=total_pruned,
        messages_affected=replaced,
        messages_removed=0,
        messages_replaced=replaced,
        summary=f"Deduped system-reminders in {replaced} messages ({len(seen_hashes)} unique)",
    )


@strategy("tool-result-age", "Compact old tool results by age — minify mid-age, stub old", "standard", "10-40%")
def strategy_tool_result_age(messages: list[Message], config: dict) -> StrategyResult:
    """Three-tier age-based tool result compaction.

    Tool results decay in value exponentially. A file read from 50 turns ago
    is stale (the file has been edited since) and wastes tokens. Claude can
    always re-read if needed.

    Tiers (configurable via config):
      Recent (0 to mid_age turns):  untouched
      Mid-age (mid_age to old_age): minify JSON, strip diff context lines
      Old (old_age+):               replace content with compact stub

    Research: JetBrains validated observation masking on SWE-bench — matched
    LLM summarization quality at zero compute cost.
    """
    mid_age, old_age = coerce_ordered_pair(
        config, "tool_result_mid_age", "tool_result_old_age", defaults=(15, 40)
    )

    actions: list[PruneAction] = []
    total_orig = sum(b for _, _, b in messages)
    total_pruned = 0
    replaced = 0

    # Count actual user prompts (not tool_result wrappers which also have type="user")
    def _is_user_prompt(msg: dict) -> bool:
        if get_msg_type(msg) != "user":
            return False
        content = msg.get("message", {}).get("content", "")
        # tool_result messages have list content with tool_result blocks
        if isinstance(content, list):
            return not any(b.get("type") == "tool_result" for b in content if isinstance(b, dict))
        return isinstance(content, str)

    total_turns = sum(1 for _, msg, _ in messages if _is_user_prompt(msg))

    # Build turn index: for each message position, how many user prompts precede it?
    turn_count = 0
    msg_turn: list[int] = []
    for _, msg, _ in messages:
        if _is_user_prompt(msg):
            turn_count += 1
        msg_turn.append(turn_count)

    for pos, (idx, msg, size) in enumerate(messages):
        if is_protected(msg):
            continue

        blocks = get_content_blocks(msg)
        if not blocks:
            continue

        has_tool_result = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in blocks)
        if not has_tool_result:
            continue

        turns_ago = total_turns - msg_turn[pos]

        if turns_ago < mid_age:
            continue  # Recent — keep verbatim

        new_blocks = []
        changed = False

        for block in blocks:
            if block.get("type") != "tool_result":
                new_blocks.append(block)
                continue

            content = block.get("content", "")
            if not isinstance(content, str) or len(content) < 100:
                new_blocks.append(block)
                continue

            tool_use_id = block.get("tool_use_id", "")

            if turns_ago >= old_age:
                # OLD: replace with compact stub
                stub = _build_stub(block, blocks, messages, pos)
                new_blocks.append({**block, "content": stub})
                changed = True
            else:
                # MID-AGE: minify content
                compacted = _minify_tool_content(content)
                if compacted != content:
                    new_blocks.append({**block, "content": compacted})
                    changed = True
                else:
                    new_blocks.append(block)

        if changed:
            new_msg = set_content_blocks(msg, new_blocks)
            new_size = msg_bytes(new_msg)
            saved = size - new_size
            if saved > 0:
                actions.append(PruneAction(
                    line_index=idx,
                    action="replace",
                    reason=f"tool-result-age ({turns_ago} turns ago)",
                    original_bytes=size,
                    pruned_bytes=new_size,
                    replacement=new_msg,
                ))
                total_pruned += saved
                replaced += 1

    return StrategyResult(
        strategy_name="tool-result-age",
        actions=actions,
        original_bytes=total_orig,
        pruned_bytes=total_pruned,
        messages_affected=replaced,
        messages_removed=0,
        messages_replaced=replaced,
        summary=f"Compacted {replaced} old tool results ({total_pruned / 1024:.0f}KB saved)",
    )


def _build_stub(block: dict, all_blocks: list[dict], messages: list[Message], pos: int) -> str:
    """Build a compact stub for an old tool result."""
    content = block.get("content", "")
    tool_use_id = block.get("tool_use_id", "")
    content_len = len(content)
    line_count = content.count("\n") + 1 if content else 0

    # Try to find the matching tool_use to get tool name and input
    tool_name = ""
    tool_path = ""
    for search_pos in range(max(0, pos - 10), pos + 1):
        if search_pos >= len(messages):
            break
        _, search_msg, _ = messages[search_pos]
        for b in get_content_blocks(search_msg):
            if b.get("type") == "tool_use" and b.get("id") == tool_use_id:
                tool_name = b.get("name", "")
                tool_input = b.get("input", {})
                tool_path = (
                    tool_input.get("file_path", "")
                    or tool_input.get("path", "")
                    or tool_input.get("pattern", "")
                    or tool_input.get("command", "")[:80]
                )
                break

    parts = ["[cozempic"]
    if tool_name:
        parts.append(f": {tool_name}")
    if tool_path:
        parts.append(f" {tool_path}")
    parts.append(f" — {line_count} lines, {content_len / 1024:.1f}KB]")

    return "".join(parts)


def _minify_tool_content(content: str) -> str:
    """Minify mid-age tool result content: strip JSON whitespace, collapse diff context."""
    # Try JSON minification first
    try:
        parsed = json.loads(content)
        minified = json.dumps(parsed, separators=(",", ":"))
        if len(minified) < len(content) * 0.85:  # Only if meaningful savings
            return minified
    except (json.JSONDecodeError, TypeError):
        pass

    # Collapse diff context lines — but ONLY for a GENUINE unified diff. The gate
    # requires the unified-diff ENVELOPE (a `--- `/`+++ ` file-header pair or a
    # `diff ` command line) AND a real hunk header. A lone coincidental `@@ … @@`
    # line in non-diff output (a git-log fragment, CI text, an indented config
    # block) no longer triggers collapse — and even past the gate,
    # _collapse_diff_context only collapses context lines INSIDE a hunk, so it can
    # never wholesale-destroy indented non-diff content (the audit P1).
    if "\0" not in content and _looks_like_unified_diff(content):
        collapsed = _collapse_diff_context(content)
        if collapsed != content:
            return collapsed

    return content


# A real unified-diff hunk header, anchored at line start: "@@ -12,7 +12,9 @@".
_UNIFIED_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.M)
# The file-header pair every unified diff carries: a "--- …" line immediately
# followed by a "+++ …" line.
_DIFF_FILE_HDR_RE = re.compile(r"^--- .*\n\+\+\+ ", re.M)


def _looks_like_unified_diff(content: str) -> bool:
    """True only for content with the unified-diff envelope (so we never collapse
    non-diff output that merely contains a hunk-shaped line)."""
    if not _UNIFIED_HUNK_RE.search(content):
        return False
    if _DIFF_FILE_HDR_RE.search(content):
        return True
    return content.startswith("diff ") or "\ndiff --git " in content or "\ndiff " in content


def _collapse_diff_context(diff_text: str) -> str:
    """Strip unchanged context lines from unified diffs, keep +/- and headers.

    Context lines (" "-prefixed) are collapsed ONLY while inside a hunk (after an
    `@@` header). Lines before the first hunk, and any non-hunk content, are kept
    verbatim — so a stray indented block cannot be destroyed even if the gate
    passed on a real diff that also contains trailing prose."""
    lines = diff_text.split("\n")
    result = []
    context_run = 0
    in_hunk = False

    def _flush():
        nonlocal context_run
        if context_run > 0:
            result.append(f"  [...{context_run} unchanged lines...]")
            context_run = 0

    for line in lines:
        if line.startswith("@@"):
            _flush()
            in_hunk = True
            result.append(line)
        elif line.startswith(("diff ", "---", "+++", "+", "-")):
            _flush()
            result.append(line)
        elif in_hunk and line.startswith(" "):
            context_run += 1
        else:
            # A non-context line: we're no longer inside a hunk's body. Reset
            # in_hunk so indented content AFTER the hunk (e.g. a git-log-p second
            # commit's message body, or trailing prose) is kept verbatim and never
            # collapsed (the audit P1 — in_hunk was set but never reset).
            in_hunk = False
            _flush()
            result.append(line)

    _flush()
    collapsed = "\n".join(result)
    return collapsed if len(collapsed) < len(diff_text) else diff_text
