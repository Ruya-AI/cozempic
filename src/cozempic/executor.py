"""Action executor and prescription runner."""

from __future__ import annotations

from .helpers import get_content_blocks, msg_bytes, set_content_blocks
from .registry import STRATEGIES
from .types import Message, PruneAction, StrategyResult


def execute_actions(
    messages: list[Message],
    actions: list[PruneAction],
) -> list[Message]:
    """Apply PruneActions to messages and return the new message list."""
    removals: set[int] = set()
    replacements: dict[int, dict] = {}

    for action in actions:
        if action.action == "remove":
            removals.add(action.line_index)
        elif action.action == "replace" and action.replacement:
            replacements[action.line_index] = action.replacement

    # T1.5: Protect tool_use messages whose tool_results are kept
    tool_result_refs: set[str] = set()
    for idx, msg, _ in messages:
        if idx in removals:
            continue
        for block in get_content_blocks(msg):
            if block.get("type") == "tool_result":
                use_id = block.get("tool_use_id", "")
                if use_id:
                    tool_result_refs.add(use_id)

    for idx, msg, _ in messages:
        if idx not in removals:
            continue
        for block in get_content_blocks(msg):
            if block.get("type") == "tool_use" and block.get("id", "") in tool_result_refs:
                removals.discard(idx)
                break

    result: list[Message] = []
    for idx, msg, size in messages:
        if idx in removals:
            continue
        if idx in replacements:
            new_msg = replacements[idx]
            new_size = msg_bytes(new_msg)
            result.append((idx, new_msg, new_size))
        else:
            result.append((idx, msg, size))

    # T1.4: Re-link parent chains through removed messages
    result = _relink_parent_chain(messages, result, removals)

    return result


def _relink_parent_chain(
    messages_before: list[Message],
    messages_after: list[Message],
    removals: set[int],
) -> list[Message]:
    """Re-link parentUuid and logicalParentUuid, skipping removed entries."""
    if not removals:
        return messages_after

    # Build maps from the original messages
    uuid_to_parent: dict[str, str] = {}
    uuid_to_logical: dict[str, str] = {}
    removed_uuids: set[str] = set()

    for idx, msg, _ in messages_before:
        u = msg.get("uuid", "")
        if u:
            if "parentUuid" in msg:
                uuid_to_parent[u] = msg.get("parentUuid") or ""
            if "logicalParentUuid" in msg:
                uuid_to_logical[u] = msg.get("logicalParentUuid") or ""
        if idx in removals and u:
            removed_uuids.add(u)

    if not removed_uuids:
        return messages_after

    def resolve(uuid: str, chain: dict[str, str]) -> str | None:
        """Walk up the chain until we find a non-removed UUID."""
        seen: set[str] = set()
        cur = uuid
        while cur and cur not in seen:
            seen.add(cur)
            if cur not in removed_uuids:
                return cur
            cur = chain.get(cur, "")
        return None

    result = []
    for idx, msg, size in messages_after:
        changed = False
        new_msg = msg

        if msg.get("parentUuid") in removed_uuids:
            new_msg = dict(new_msg)
            new_msg["parentUuid"] = resolve(msg["parentUuid"], uuid_to_parent)
            changed = True

        if msg.get("logicalParentUuid") in removed_uuids:
            if new_msg is msg:
                new_msg = dict(msg)
            new_msg["logicalParentUuid"] = resolve(msg["logicalParentUuid"], uuid_to_logical)
            changed = True

        if changed:
            result.append((idx, new_msg, msg_bytes(new_msg)))
        else:
            result.append((idx, msg, size))

    return result


def fix_orphaned_tool_results(messages: list[Message]) -> tuple[list[Message], int]:
    """Remove or fix tool_result blocks whose matching tool_use was removed.

    The Claude API requires every tool_result to have a corresponding tool_use
    in the preceding message. When strategies remove messages containing
    tool_use blocks, the paired tool_result becomes orphaned and causes
    400 errors on compact/resume.

    Returns (fixed_messages, orphans_fixed).
    """
    # Pass 1: collect all tool_use IDs present in the messages
    tool_use_ids: set[str] = set()
    for _, msg, _ in messages:
        for block in get_content_blocks(msg):
            if block.get("type") == "tool_use":
                use_id = block.get("id", "")
                if use_id:
                    tool_use_ids.add(use_id)

    # Pass 2: find and remove orphaned tool_result blocks
    orphans_fixed = 0
    result: list[Message] = []

    for idx, msg, size in messages:
        blocks = get_content_blocks(msg)
        if not blocks:
            result.append((idx, msg, size))
            continue

        has_orphan = False
        for block in blocks:
            if block.get("type") == "tool_result":
                use_id = block.get("tool_use_id", "")
                if use_id and use_id not in tool_use_ids:
                    has_orphan = True
                    break

        if not has_orphan:
            result.append((idx, msg, size))
            continue

        # Filter out orphaned tool_result blocks, keep everything else
        new_blocks = []
        for block in blocks:
            if block.get("type") == "tool_result":
                use_id = block.get("tool_use_id", "")
                if use_id and use_id not in tool_use_ids:
                    orphans_fixed += 1
                    continue
            new_blocks.append(block)

        if new_blocks:
            new_msg = set_content_blocks(msg, new_blocks)
            result.append((idx, new_msg, msg_bytes(new_msg)))
        else:
            # All blocks were orphaned — drop the entire message
            orphans_fixed += 1

    return result, orphans_fixed


def run_prescription(
    messages: list[Message],
    strategy_names: list[str],
    config: dict,
) -> tuple[list[Message], list[StrategyResult]]:
    """Run strategies sequentially, each on the result of the previous.

    This ensures replacements compose correctly when multiple strategies
    modify the same message. After all strategies run, the pipeline is:

      1. Run each strategy in order.
      2. ``fix_orphaned_tool_results`` removes orphaned tool_result blocks.
      3. (P0-C) ``enforce_floor`` re-adds must-preserve messages dropped by
         strategies — root, last-K turns, conversation survival cap.
      4. (P0-B) ``validate_post_prune`` runs C1–C6 structural checks. If
         any fails, propagate ``PruneValidationError`` to the caller; the
         caller (guard / cli) is responsible for skipping the save.

    The internal flag ``_disable_floor_for_test`` skips step 3 for the test
    that verifies the validation gate alone catches a missing root.
    """
    from .safety import enforce_floor, validate_post_prune
    from .config import load_config

    current = messages
    results: list[StrategyResult] = []
    for sname in strategy_names:
        if sname not in STRATEGIES:
            continue
        sr = STRATEGIES[sname].func(current, config)
        results.append(sr)
        if sr.actions:
            old_current = current
            current = execute_actions(current, sr.actions)
            del old_current  # Free previous list immediately

    # Step 2: orphaned tool_result cleanup.
    current, orphans = fix_orphaned_tool_results(current)
    if orphans > 0:
        results.append(StrategyResult(
            strategy_name="orphan-fix",
            actions=[],
            original_bytes=0,
            pruned_bytes=0,
            messages_affected=orphans,
            messages_removed=0,
            messages_replaced=orphans,
            summary=f"Fixed {orphans} orphaned tool_result block(s)",
        ))

    # Step 3: floor preservation — re-add must-preserve messages.
    cfg = load_config()
    if not config.get("_disable_floor_for_test"):
        current = enforce_floor(messages, current, cfg=cfg.floor)

    # Step 4: structural validation. Raises PruneValidationError on failure.
    validate_post_prune(messages, current)

    return current, results
