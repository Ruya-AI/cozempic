"""Action executor and prescription runner."""

from __future__ import annotations

from .helpers import (
    _METADATA_SINGLETON_KEY,
    get_content_blocks,
    hashable_str,
    msg_bytes,
    set_content_blocks,
)
from ._validation import ConfigError
from .registry import STRATEGIES
from .types import Message, PruneAction, StrategyResult


# P0-D — last-of-type metadata singleton protection.
# The LAST occurrence of each of these types is tagged before strategies run
# so is_protected() skips them. The tag is internal-only and stripped before
# run_prescription returns — it MUST NOT persist to disk.
_LAST_OF_TYPE_PROTECTED: frozenset[str] = frozenset({
    "ai-title",
    "last-prompt",
    "permission-mode",
})
# Re-export for callers that want to reference the tag name without importing helpers
_SINGLETON_TAG: str = _METADATA_SINGLETON_KEY


def _tag_last_of_metadata_types(messages: list[Message]) -> None:
    """Mark the LAST occurrence of each protected-singleton type in-place.

    Sets ``msg[_SINGLETON_TAG] = True`` on the last entry per protected type.
    ``is_protected()`` in helpers.py honors this tag so subsequent strategies
    skip the entry. ``_strip_metadata_singleton_tags`` MUST be called before
    returning from ``run_prescription`` to ensure the tag does not leak to disk.
    """
    last_pos: dict[str, int] = {}
    for pos, (_, msg, _) in enumerate(messages):
        t = msg.get("type", "")
        if t in _LAST_OF_TYPE_PROTECTED:
            last_pos[t] = pos
    for pos in last_pos.values():
        _, msg, _ = messages[pos]
        msg[_SINGLETON_TAG] = True


def _strip_metadata_singleton_tags(messages: list[Message]) -> None:
    """Remove the internal singleton tag from every message in-place."""
    for _, msg, _ in messages:
        if _SINGLETON_TAG in msg:
            msg.pop(_SINGLETON_TAG, None)


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
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                use_id = hashable_str(block.get("tool_use_id"))  # unhashable -> "" (R6)
                if use_id:
                    tool_result_refs.add(use_id)

    for idx, msg, _ in messages:
        if idx not in removals:
            continue
        for block in get_content_blocks(msg):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and hashable_str(block.get("id")) in tool_result_refs:
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
        # hashable_str: uuid/parentUuid are top-level fields used as dict keys / set
        # members; an unhashable value (poisoned JSONL) would crash the parent-relink
        # (which runs on EVERY prune, OUTSIDE per-strategy isolation) (R6 crash class).
        u = hashable_str(msg.get("uuid"))
        if u:
            if "parentUuid" in msg:
                uuid_to_parent[u] = hashable_str(msg.get("parentUuid"))
            if "logicalParentUuid" in msg:
                uuid_to_logical[u] = hashable_str(msg.get("logicalParentUuid"))
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

        if hashable_str(msg.get("parentUuid")) in removed_uuids:
            new_msg = dict(new_msg)
            new_msg["parentUuid"] = resolve(hashable_str(msg.get("parentUuid")), uuid_to_parent)
            changed = True

        if hashable_str(msg.get("logicalParentUuid")) in removed_uuids:
            if new_msg is msg:
                new_msg = dict(msg)
            new_msg["logicalParentUuid"] = resolve(hashable_str(msg.get("logicalParentUuid")), uuid_to_logical)
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
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                use_id = hashable_str(block.get("id"))  # unhashable -> "" (R6 crash class)
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
            if isinstance(block, dict) and block.get("type") == "tool_result":
                use_id = hashable_str(block.get("tool_use_id"))
                if use_id and use_id not in tool_use_ids:
                    has_orphan = True
                    break

        if not has_orphan:
            result.append((idx, msg, size))
            continue

        # Filter out orphaned tool_result blocks, keep everything else
        new_blocks = []
        for block in blocks:
            # Non-dict elements are PRESERVED (append unchanged) — never dropped
            # (that would be data loss); only a genuine orphaned tool_result dict
            # is filtered.
            if isinstance(block, dict) and block.get("type") == "tool_result":
                use_id = hashable_str(block.get("tool_use_id"))
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
    *,
    floor_config: "FloorConfig | None" = None,
) -> tuple[list[Message], list[StrategyResult]]:
    """Run strategies sequentially, each on the result of the previous.

    This ensures replacements compose correctly when multiple strategies
    modify the same message. After all strategies run, the pipeline is:

      Step 0. (P0-D) Tag last-of-type metadata singletons so strategies that
              honor is_protected() skip them. Strip runs in finally.
      Step 1. Run each strategy in order.
      Step 2. (P0-C) enforce_floor — re-add must-preserve messages.
              ``floor_config=None`` → resolve via ``load_config().floor``
              (always-on in production). Pass ``FloorConfig.disabled()`` to
              opt out (test code only — not reachable from external JSON config,
              satisfying review finding H-2).
      Step 3. fix_orphaned_tool_results. Runs AFTER floor so floor re-adds
              that resurrect tool_result carriers are cleaned before save.
      Step 4. (P0-B) validate_post_prune — C1-C7 structural checks. Propagates
              PruneValidationError to caller on failure; caller skips the save.
    """
    # Lazy imports — safety.py and config.py are new modules introduced by this
    # PR. Importing at the top level would break all existing callers that import
    # executor.py before safety.py/config.py are available during partial installs.
    from .config import FloorConfig, load_config
    from .safety import enforce_floor, validate_post_prune

    # Step 0 (P0-D): tag last-of-type metadata singletons before strategies run.
    _tag_last_of_metadata_types(messages)

    current = messages
    results: list[StrategyResult] = []
    # Wrap from this point in try/finally so the singleton-tag strip ALWAYS runs
    # even if a downstream step (validation) raises. Without the finally, a
    # PruneValidationError leaves the caller's input list carrying the internal
    # __cozempic_metadata_singleton__ flag, which would leak to disk on the next
    # successful save_messages call (REVIEW-max A.3).
    try:
        # Step 1: run strategies. Each strategy is ISOLATED — a crash on a
        # malformed/poisoned message (an unexpected non-dict/non-string field deep
        # in untrusted JSONL) skips THAT strategy and continues with the rest,
        # rather than aborting the whole prune. This is the systemic defense for the
        # large untrusted-field surface: one bad message can no longer (a) abort
        # `treat`/`reload`, or (b) crash the guard cycle into a respawn storm.
        # KeyboardInterrupt/SystemExit (and the structural PruneValidationError from
        # later steps) still propagate; only a strategy-internal error is contained.
        for sname in strategy_names:
            if sname not in STRATEGIES:
                continue
            try:
                sr = STRATEGIES[sname].func(current, config)
            except (KeyboardInterrupt, SystemExit):
                raise
            except ConfigError:
                # A user CONFIG error (bad coerce_non_negative_int value) must NOT be
                # swallowed and mislabeled as a "malformed message" — that hides the
                # very validation this layer adds. Propagate it (ynaamane review #6).
                raise
            except Exception as _strat_exc:
                import sys as _sys
                print(f"  Cozempic: strategy '{sname}' skipped after an unexpected error "
                      f"(malformed message?): {_strat_exc!r}", file=_sys.stderr)
                continue
            results.append(sr)
            if sr.actions:
                old_current = current
                current = execute_actions(current, sr.actions)
                del old_current  # Free previous list immediately

        # Step 2 (P0-C): floor preservation — re-add must-preserve messages.
        cfg_floor = floor_config if floor_config is not None else load_config().floor
        current = enforce_floor(messages, current, cfg=cfg_floor)

        # Step 3: orphaned tool_result cleanup. Runs AFTER floor so a re-added
        # user carrying a tool_result whose paired tool_use is still missing
        # has its orphan block stripped before save.
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

        # Step 4 (P0-B): structural validation. Raises PruneValidationError on failure.
        validate_post_prune(messages, current)

    finally:
        # Strip the internal singleton tag from every surviving entry so it
        # does not leak to the saved JSONL. Also strip from the input msgs_before
        # list — the tag was applied in place there too (REVIEW-max A.3).
        _strip_metadata_singleton_tags(current)
        _strip_metadata_singleton_tags(messages)

    return current, results
