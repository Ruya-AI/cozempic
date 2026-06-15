"""Gentle-tier strategies: safe, minimal pruning."""

from __future__ import annotations

import copy

from ..helpers import get_msg_type, is_protected, msg_bytes
from ..registry import strategy
from ..types import Message, PruneAction, StrategyResult


def _no_op(name: str, total_orig: int, summary: str) -> StrategyResult:
    """Return a no-op StrategyResult."""
    return StrategyResult(name, [], total_orig, 0, 0, 0, 0, summary)


@strategy("compact-summary-collapse", "Remove all pre-compaction messages (already in the summary)", "gentle", "85-95%")
def strategy_compact_summary_collapse(messages: list[Message], config: dict) -> StrategyResult:
    """Remove everything before the last compact_boundary.

    After compaction, all pre-boundary messages are already summarized in the
    isCompactSummary message. Claude Code discards them at load time for files
    >5MB; this strategy does it proactively at any size.

    Safety: skips if hasPreservedSegment is True (boundary preserves pre-content).
    Keeps metadata singletons that only appear before the boundary (re-anchor).
    """
    total_orig = sum(b for _, _, b in messages)

    # Find last compact_boundary
    last_boundary_pos = -1
    last_boundary_msg = None
    for pos, (idx, msg, _) in enumerate(messages):
        if msg.get("type") == "system" and msg.get("subtype") == "compact_boundary":
            last_boundary_pos = pos
            last_boundary_msg = msg

    if last_boundary_pos < 0:
        return _no_op("compact-summary-collapse", total_orig, "No compact_boundary found")

    if last_boundary_msg.get("hasPreservedSegment"):
        return _no_op("compact-summary-collapse", total_orig, "Skipped (hasPreservedSegment=True)")

    # Metadata singletons: keep if they only appear before the boundary.
    # "permission-mode" MUST be here — it carries the dangerouslySkipPermissions
    # bootstrap state; losing the last occurrence silently breaks session resume.
    _META_TYPES = {
        "last-prompt", "pr-link", "custom-title", "ai-title",
        "attribution-snapshot", "permission-mode",
    }
    post_meta_types = {msg.get("type") for _, msg, _ in messages[last_boundary_pos:]}

    actions: list[PruneAction] = []
    total_pruned = 0
    removed = 0

    for pos, (idx, msg, size) in enumerate(messages[:last_boundary_pos]):
        if is_protected(msg):
            continue
        # Keep metadata singletons that won't appear after boundary
        if msg.get("type") in _META_TYPES and msg.get("type") not in post_meta_types:
            continue
        actions.append(PruneAction(
            line_index=idx, action="remove",
            reason="compact-summary-collapse (pre-boundary, already summarized)",
            original_bytes=size, pruned_bytes=0,
        ))
        total_pruned += size
        removed += 1

    return StrategyResult(
        strategy_name="compact-summary-collapse",
        actions=actions,
        original_bytes=total_orig,
        pruned_bytes=total_pruned,
        messages_affected=removed,
        messages_removed=removed,
        messages_replaced=0,
        summary=f"Collapsed {removed} pre-compaction messages ({total_pruned / 1024:.0f}KB)",
    )


@strategy("attribution-snapshot-strip", "Strip attribution-snapshot metadata entries", "gentle", "0-2%")
def strategy_attribution_snapshot_strip(messages: list[Message], config: dict) -> StrategyResult:
    """Remove AttributionSnapshotMessage entries — pure UI metadata, never sent to API."""
    actions: list[PruneAction] = []
    total_orig = sum(b for _, _, b in messages)
    total_pruned = 0

    for idx, msg, size in messages:
        if is_protected(msg):
            continue
        if msg.get("type") == "attribution-snapshot":
            actions.append(PruneAction(idx, "remove", "attribution-snapshot-strip", size, 0))
            total_pruned += size

    removed = len(actions)
    return StrategyResult(
        strategy_name="attribution-snapshot-strip",
        actions=actions,
        original_bytes=total_orig,
        pruned_bytes=total_pruned,
        messages_affected=removed,
        messages_removed=removed,
        messages_replaced=0,
        summary=f"Stripped {removed} attribution-snapshot entries",
    )


@strategy("progress-collapse", "Collapse consecutive and isolated progress tick messages", "gentle", "40-48%")
def strategy_progress_collapse(messages: list[Message], config: dict) -> StrategyResult:
    """Remove progress messages (hook_progress, bash_progress, etc.).

    Consecutive runs: all but the last are removed (preserves one summary tick per run).
    Isolated ticks (run length 1): removed entirely — they carry no conversational value.
    """
    actions: list[PruneAction] = []
    total_orig = sum(b for _, _, b in messages)
    total_pruned = 0

    i = 0
    while i < len(messages):
        idx, msg, size = messages[i]
        if get_msg_type(msg) == "progress" and not is_protected(msg):
            run_start = i
            run_end = i + 1
            while run_end < len(messages) and get_msg_type(messages[run_end][1]) == "progress" and not is_protected(messages[run_end][1]):
                run_end += 1

            run_length = run_end - run_start
            if run_length > 1:
                # Consecutive run: keep last, remove the rest
                for j in range(run_start, run_end - 1):
                    rm_idx, _, rm_size = messages[j]
                    actions.append(PruneAction(
                        line_index=rm_idx,
                        action="remove",
                        reason=f"progress tick {j - run_start + 1}/{run_length}",
                        original_bytes=rm_size,
                        pruned_bytes=0,
                    ))
                    total_pruned += rm_size
            else:
                # Isolated tick: remove entirely
                actions.append(PruneAction(
                    line_index=idx,
                    action="remove",
                    reason="isolated progress tick",
                    original_bytes=size,
                    pruned_bytes=0,
                ))
                total_pruned += size
            i = run_end
        else:
            i += 1

    removed = len(actions)
    return StrategyResult(
        strategy_name="progress-collapse",
        actions=actions,
        original_bytes=total_orig,
        pruned_bytes=total_pruned,
        messages_affected=removed,
        messages_removed=removed,
        messages_replaced=0,
        summary=f"Collapsed {removed} progress ticks",
    )


@strategy("file-history-dedup", "Deduplicate file-history-snapshot messages", "gentle", "3-6%")
def strategy_file_history_dedup(messages: list[Message], config: dict) -> StrategyResult:
    """Remove duplicate file-history-snapshot messages, keeping only the latest per messageId."""
    actions: list[PruneAction] = []
    total_orig = sum(b for _, _, b in messages)
    total_pruned = 0

    snapshots: dict[str, list[int]] = {}
    for pos, (idx, msg, size) in enumerate(messages):
        if get_msg_type(msg) == "file-history-snapshot" and not is_protected(msg):
            mid = msg.get("messageId", "")
            if mid:
                snapshots.setdefault(mid, []).append(pos)

    for mid, positions in snapshots.items():
        if len(positions) > 1:
            for pos in positions[:-1]:
                idx, _, size = messages[pos]
                actions.append(PruneAction(
                    line_index=idx,
                    action="remove",
                    reason=f"duplicate file-history-snapshot (messageId={mid[:8]}...)",
                    original_bytes=size,
                    pruned_bytes=0,
                ))
                total_pruned += size

    # Collapse consecutive isSnapshotUpdate=true runs
    current_run: list[int] = []
    update_runs: list[list[int]] = []
    for pos, (idx, msg, size) in enumerate(messages):
        if get_msg_type(msg) == "file-history-snapshot" and msg.get("isSnapshotUpdate") and not is_protected(msg):
            current_run.append(pos)
        else:
            if len(current_run) > 1:
                update_runs.append(current_run)
            current_run = []
    if len(current_run) > 1:
        update_runs.append(current_run)

    already_removed = {a.line_index for a in actions}
    for run in update_runs:
        for pos in run[:-1]:
            idx, _, size = messages[pos]
            if idx not in already_removed:
                actions.append(PruneAction(
                    line_index=idx,
                    action="remove",
                    reason="consecutive snapshot update",
                    original_bytes=size,
                    pruned_bytes=0,
                ))
                total_pruned += size

    removed = len(actions)
    return StrategyResult(
        strategy_name="file-history-dedup",
        actions=actions,
        original_bytes=total_orig,
        pruned_bytes=total_pruned,
        messages_affected=removed,
        messages_removed=removed,
        messages_replaced=0,
        summary=f"Removed {removed} duplicate file-history snapshots",
    )


@strategy("metadata-strip", "Strip token usage stats, signatures, stop_reason", "gentle", "1-3%")
def strategy_metadata_strip(messages: list[Message], config: dict) -> StrategyResult:
    """Remove metadata fields: usage, stop_reason, stop_sequence, costUSD, duration."""
    strip_inner = {"usage", "stop_reason", "stop_sequence"}
    strip_outer = {"costUSD", "duration", "apiDuration"}

    actions: list[PruneAction] = []
    total_orig = sum(b for _, _, b in messages)
    total_pruned = 0
    replaced = 0

    # Capture exact token counts before stripping (usage fields will be deleted).
    # Coerce via _as_int — a present-but-null/string usage value (malformed
    # transcript) would otherwise raise TypeError here and crash `treat --rx gentle`
    # (and silently disable SOFT-tier gentle pruning in the daemon). Sibling of the
    # tokens.py coercion fix (the #123 fix-one-miss-the-sibling lesson).
    from ..tokens import _as_int
    exact_tokens_before = 0
    for _, msg, _ in messages:
        inner = msg.get("message")
        if not isinstance(inner, dict):  # a non-dict "message" (str/None) would crash .get()
            continue
        usage = inner.get("usage")
        if isinstance(usage, dict):
            exact_tokens_before += _as_int(usage.get("input_tokens", 0)) + _as_int(usage.get("output_tokens", 0))

    for pos, (idx, msg, size) in enumerate(messages):
        if is_protected(msg):
            continue
        # The loader guarantees a normal message's inner "message" is a dict (or
        # absent); a non-dict inner is wrapped as a _parse_error upstream. So the
        # absent-key default {} is safe here.
        new_msg = {**msg, "message": {**msg.get("message", {})}}  # Shallow copy outer + inner
        changed = False

        inner = new_msg.get("message", {})
        for f in strip_inner:
            if f in inner:
                del inner[f]
                changed = True

        for f in strip_outer:
            if f in new_msg:
                del new_msg[f]
                changed = True

        if changed:
            new_size = msg_bytes(new_msg)
            saved = size - new_size
            if saved > 0:
                actions.append(PruneAction(
                    line_index=idx,
                    action="replace",
                    reason="metadata-strip",
                    original_bytes=size,
                    pruned_bytes=new_size,
                    replacement=new_msg,
                ))
                total_pruned += saved
                replaced += 1

    summary = f"Stripped metadata from {replaced} messages"
    if exact_tokens_before > 0:
        summary += f" (exact usage before strip: {exact_tokens_before:,} tokens)"

    return StrategyResult(
        strategy_name="metadata-strip",
        actions=actions,
        original_bytes=total_orig,
        pruned_bytes=total_pruned,
        messages_affected=replaced,
        messages_removed=0,
        messages_replaced=replaced,
        summary=summary,
    )
