"""Safety guards for the session pruner.

Implements P0-B and P0-C from the prune-safety defense-in-depth port
onto the v1.8.18 terminate-first flow:

P0-B — Post-prune structural validation:
  - ``PruneValidationError`` exception with reason + evidence dict
  - ``validate_post_prune(msgs_before, msgs_after)`` — raises on C1-C8

P0-C — Floor preservation:
  - ``enforce_floor(msgs_before, msgs_after, cfg)`` — re-adds must-preserve msgs

P0-D helpers (executor.py owns the tagging; this module owns enforcement):
  - ``FloorConfig`` re-exported from config.py for ergonomic imports

P0-A (idle guard) is intentionally out of scope for this PR.
``simulate_replay_readiness`` (single-list replay probe) deferred —
``validate_post_prune`` (two-list, in the prune path) covers the
prune-path guarantee; a standalone diagnostic is a separate PR if wanted.
"""

from __future__ import annotations

import math

from .config import FloorConfig
from .helpers import hashable_str

__all__ = [
    "PruneValidationError",
    "FloorConfig",
    "validate_post_prune",
    "enforce_floor",
]


# ── P0-B — Post-prune structural validation ───────────────────────────────────


class PruneValidationError(Exception):
    """Raised when the pruned message list fails structural validation.

    ``reason`` is a human-readable summary. ``evidence`` is a dict that
    callers (guard daemon, CLI) log; it always contains a ``failed_check``
    key matching one of ``"C1".."C8"`` so log aggregators can group failures.
    """

    def __init__(self, reason: str, evidence: dict):
        self.reason = reason
        self.evidence = evidence
        super().__init__(f"Pruned session would not replay cleanly: {reason}")


def _last_of_type(
    messages: list[tuple[int, dict, int]],
    msg_type: str,
) -> dict | None:
    """Return the last message dict whose ``type`` equals ``msg_type``."""
    last: dict | None = None
    for _, msg, _ in messages:
        if msg.get("type") == msg_type:
            last = msg
    return last


def _last_compact_boundary(
    messages: list[tuple[int, dict, int]],
) -> dict | None:
    """Return the last system/subtype=compact_boundary entry."""
    last: dict | None = None
    for _, msg, _ in messages:
        if (msg.get("type") == "system"
                and msg.get("subtype") == "compact_boundary"):
            last = msg
    return last


def _last_active_compact_boundary_idx(
    messages: list[tuple[int, dict, int]],
) -> int | None:
    """Return the line index of the last ACTIVE compact_boundary in messages.

    An active boundary is one WITHOUT ``hasPreservedSegment=True``.
    Returns None when no active boundary is found (normal non-compacted sessions).

    Used by enforce_floor and validate_post_prune to determine which messages
    are pre-boundary and therefore must not be re-added (P0-A) or required to
    survive (C2 eligibility).
    """
    last_idx: int | None = None
    for idx, msg, _ in messages:
        if (msg.get("type") == "system"
                and msg.get("subtype") == "compact_boundary"
                and not msg.get("hasPreservedSegment")):
            last_idx = idx
    return last_idx


def _build_orphan_shells(
    msgs_before: list[tuple[int, dict, int]],
) -> set[str]:
    """Compute the set of uuids of orphan-shell messages in msgs_before.

    An orphan-shell is a message that meets ALL of:
      1. Has ≥ 1 content block.
      2. ALL its content blocks are ``tool_result`` blocks.
      3. Every such ``tool_result``'s ``tool_use_id`` is NOT present in any
         ``tool_use`` block anywhere in msgs_before (cross-session orphan).

    ``fix_orphaned_tool_results`` legitimately drops orphan-shells because the
    Anthropic API requires every ``tool_result`` to have a matching ``tool_use``
    in the message history. Dropping a cross-session orphan-shell is correct
    API-hygiene, NOT a prune-induced structural failure.

    C1 and C2 must exclude orphan-shell uuids from their "prune-induced break"
    checks so that legit orphan-shell drops do not cause false-positive aborts
    (the C-1 CRITICAL correctness regression on resumed sessions).
    """
    # Pass 1: collect all tool_use ids present in msgs_before.
    before_tool_use_ids: set[str] = set()
    for _, msg, _ in msgs_before:
        content = msg.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tid = hashable_str(block.get("id"))  # unhashable id -> "" (R6 crash class)
                if tid:
                    before_tool_use_ids.add(tid)

    # Pass 2: identify orphan-shell uuids.
    orphan_shells: set[str] = set()
    for _, msg, _ in msgs_before:
        uuid = msg.get("uuid", "")
        if not uuid:
            continue
        content = msg.get("message", {}).get("content", [])
        if not isinstance(content, list) or not content:
            continue
        # All blocks must be tool_result blocks whose tool_use_id ∉ before_tool_use_ids.
        if all(
            isinstance(blk, dict)
            and blk.get("type") == "tool_result"
            and hashable_str(blk.get("tool_use_id")) not in before_tool_use_ids
            for blk in content
        ):
            orphan_shells.add(uuid)

    return orphan_shells


def validate_post_prune(
    msgs_before: list[tuple[int, dict, int]],
    msgs_after: list[tuple[int, dict, int]],
) -> None:
    """Validate the pruned message list. Raise PruneValidationError on failure.

    Checks run fail-fast in order C3 → C2 → C9 → C4 → C5 → C6 → C7 → C1 → C8
    (semantic checks before structural so the failure attribution is actionable):

      C1. parentUuid resolution — baseline-relative + orphan-shell-aware: only
          flag a parent that (a) WAS in msgs_before, (b) is absent from msgs_after,
          AND (c) is NOT a legit_removed_orphan_shell. Cross-session pointers
          (parent ∉ before_uuids) and legitimately-dropped orphan-shell parents
          are both valid and skipped.
      C2. Root preserved — at least one ELIGIBLE original ``parentUuid=null`` uuid
          from msgs_before must survive. Eligible = NOT a legit_removed_orphan_shell
          AND NOT a pre-boundary root legitimately retired by compact-summary-collapse
          (P0-A fix: compact_boundary becomes the new root in that case).
      C3. Conversation survival — ≥1 user AND ≥1 assistant survives.
      C4. compact_boundary — if msgs_before had a system/compact_boundary
          entry, the LAST such entry MUST survive.
      C5. permission-mode — if msgs_before had permission-mode entries, the
          LAST one MUST survive.
      C6. last-prompt — if msgs_before had last-prompt entries, the LAST one
          MUST survive.
      C7. ai-title — if msgs_before had ai-title entries, the LAST one MUST
          survive (REVIEW-max E.4).
      C8. tool_use↔tool_result — a surviving ``tool_use`` whose paired
          ``tool_result`` existed in msgs_before MUST keep that result in
          msgs_after (a dangling tool_use is structurally valid but
          unresumable; mirror of the orphaned-tool_result handling).
      C9. Single-root invariant — baseline-relative: msgs_after must not have
          MORE roots (parentUuid=None, non-orphan-shell) than msgs_before.
          Catches 2-root DAG forks introduced by any strategy or floor bug.

    Each check is CONDITIONAL on its precondition existing in msgs_before —
    a session that never had a permission-mode entry passes C5 trivially.
    """
    surviving_uuids: set[str] = {
        msg.get("uuid", "") for _, msg, _ in msgs_after if msg.get("uuid")
    }

    # Compute orphan-shell set once for use by both C2 and C1.
    legit_removed_orphan_shells = _build_orphan_shells(msgs_before)

    # ── C3: conversation survival ─────────────────────────────────────────────
    # Check before C2/C1 because a wholesale wipe is more actionable to report
    # as "no conversation" than as "root dropped" or "chain break".
    surviving_user = sum(1 for _, m, _ in msgs_after if m.get("type") == "user")
    surviving_asst = sum(1 for _, m, _ in msgs_after if m.get("type") == "assistant")
    before_user = any(m.get("type") == "user" for _, m, _ in msgs_before)
    before_asst = any(m.get("type") == "assistant" for _, m, _ in msgs_before)
    if (before_user and surviving_user == 0) or (before_asst and surviving_asst == 0):
        raise PruneValidationError(
            reason=(
                f"conversation wiped — surviving users={surviving_user}, "
                f"assistants={surviving_asst}"
            ),
            evidence={
                "failed_check": "C3",
                "surviving_user_count": surviving_user,
                "surviving_assistant_count": surviving_asst,
            },
        )

    # ── C2: original root uuid preserved ─────────────────────────────────────
    # Review finding H-1: a structural ``any(parentUuid is None)`` check is
    # bypassed by _relink_parent_chain re-pointing dead-end chains to None
    # when the original root is dropped. Require that AT LEAST ONE ELIGIBLE
    # original parentUuid=null uuid from msgs_before survives.
    # ELIGIBLE = not a legit_removed_orphan_shell (C-1 fix): an orphan-shell
    # root is legitimately dropped by fix_orphaned_tool_results; its absence
    # must not trigger C2. (REVIEW-max B.11 + C-1 correctness fix.)
    # ELIGIBLE also excludes pre-boundary roots when an active compact_boundary
    # exists (P0-A fix): compact-summary-collapse legitimately retires the
    # pre-boundary root; the compact_boundary itself becomes the new session root
    # (with parentUuid=None after _relink_parent_chain). Requiring the original
    # root to survive would contradict the collapse contract.
    compact_boundary_before_line_idx = _last_active_compact_boundary_idx(msgs_before)
    original_root_uuids: set[str] = set()
    for idx, msg, _ in msgs_before:
        if msg.get("parentUuid") is None and msg.get("uuid"):
            # Pre-boundary root legitimately dropped by compact-summary-collapse.
            if (compact_boundary_before_line_idx is not None
                    and idx < compact_boundary_before_line_idx):
                continue
            original_root_uuids.add(msg["uuid"])
    eligible_roots = original_root_uuids - legit_removed_orphan_shells
    if eligible_roots and not (eligible_roots & surviving_uuids):
        raise PruneValidationError(
            reason=(
                f"every original session root uuid was dropped "
                f"(expected one of {sorted(eligible_roots)} to survive)"
            ),
            evidence={
                "failed_check": "C2",
                "expected_root_uuid": sorted(eligible_roots)[0],
                "expected_root_uuids": sorted(eligible_roots),
                "before_count": len(msgs_before),
                "after_count": len(msgs_after),
            },
        )

    # ── C9: single-root invariant (no multi-root fork) ───────────────────────
    # P0-A enforces this in enforce_floor; C9 is the defensive net that catches
    # any future code path creating a second DAG root (e.g. a new strategy, a
    # floor edge case, a hasPreservedSegment interaction).
    # BASELINE-RELATIVE: only raise if msgs_after has MORE roots than msgs_before.
    # This allows legitimate 2-chain team sessions (2 roots before → 2 after = OK;
    # 1 root before → 2 after = C9).
    before_roots_count = sum(
        1 for _, msg, _ in msgs_before
        if not msg.get("parentUuid") and msg.get("uuid")
        and hashable_str(msg.get("uuid")) not in legit_removed_orphan_shells
    )
    after_roots: list[str] = [
        hashable_str(msg.get("uuid"))
        for _, msg, _ in msgs_after
        if not msg.get("parentUuid")
        and msg.get("uuid")
        and hashable_str(msg.get("uuid")) not in legit_removed_orphan_shells
    ]
    if len(after_roots) > before_roots_count:
        raise PruneValidationError(
            reason=(
                f"prune introduced additional DAG roots: "
                f"{len(after_roots)} roots after vs {before_roots_count} before"
            ),
            evidence={
                "failed_check": "C9",
                "root_uuids_after": sorted(after_roots),
                "root_count_after": len(after_roots),
                "root_count_before": before_roots_count,
            },
        )

    # ── C4: last compact_boundary preserved ──────────────────────────────────
    last_before_cb = _last_compact_boundary(msgs_before)
    if last_before_cb is not None:
        last_after_cb = _last_compact_boundary(msgs_after)
        if last_after_cb is None or (
            last_after_cb.get("uuid") != last_before_cb.get("uuid")
        ):
            raise PruneValidationError(
                reason="last compact_boundary entry was dropped",
                evidence={
                    "failed_check": "C4",
                    "expected_uuid": last_before_cb.get("uuid"),
                    "actual_uuid": last_after_cb.get("uuid") if last_after_cb else None,
                },
            )

    # ── C5: last permission-mode preserved ───────────────────────────────────
    last_before_pm = _last_of_type(msgs_before, "permission-mode")
    if last_before_pm is not None:
        last_after_pm = _last_of_type(msgs_after, "permission-mode")
        if last_after_pm is None or (
            last_after_pm.get("uuid") != last_before_pm.get("uuid")
        ):
            raise PruneValidationError(
                reason="last permission-mode entry was dropped",
                evidence={
                    "failed_check": "C5",
                    "expected_uuid": last_before_pm.get("uuid"),
                    "actual_uuid": last_after_pm.get("uuid") if last_after_pm else None,
                },
            )

    # ── C6: last last-prompt preserved ───────────────────────────────────────
    last_before_lp = _last_of_type(msgs_before, "last-prompt")
    if last_before_lp is not None:
        last_after_lp = _last_of_type(msgs_after, "last-prompt")
        if last_after_lp is None or (
            last_after_lp.get("uuid") != last_before_lp.get("uuid")
        ):
            raise PruneValidationError(
                reason="last last-prompt entry was dropped",
                evidence={
                    "failed_check": "C6",
                    "expected_uuid": last_before_lp.get("uuid"),
                    "actual_uuid": last_after_lp.get("uuid") if last_after_lp else None,
                },
            )

    # ── C7: last ai-title preserved ───────────────────────────────────────────
    # REVIEW-max E.4: mirror C5/C6 for the third member of
    # executor._LAST_OF_TYPE_PROTECTED; any path that bypasses the singleton
    # tag would silently drop the last ai-title without this check.
    last_before_at = _last_of_type(msgs_before, "ai-title")
    if last_before_at is not None:
        last_after_at = _last_of_type(msgs_after, "ai-title")
        if last_after_at is None or (
            last_after_at.get("uuid") != last_before_at.get("uuid")
        ):
            raise PruneValidationError(
                reason="last ai-title entry was dropped",
                evidence={
                    "failed_check": "C7",
                    "expected_uuid": last_before_at.get("uuid"),
                    "actual_uuid": last_after_at.get("uuid") if last_after_at else None,
                },
            )

    # ── C1: parent chain resolves (baseline-relative + orphan-shell-aware) ────
    # Defense-in-depth fallback. The executor's _relink_parent_chain step
    # SHOULD ensure every surviving parentUuid resolves; this re-verifies.
    # REVIEW-max B.10: treat falsy parentUuid (None, "", 0, ...) as equivalent
    # to None — empty string isn't a chain reference.
    #
    # PR #102 fix — unconditionally baseline-relative:
    # Skip any parent absent from before_uuids (never existed in this session
    # before the prune) — it is a cross-session pointer.
    # C-1 fix — orphan-shell-aware:
    # Also skip parents in legit_removed_orphan_shells. When a parent was
    # legitimately dropped by fix_orphaned_tool_results (cross-session orphan),
    # the child's dangling parentUuid is NOT a prune-induced break.
    # Only raise when the parent was a REAL (non-orphan-shell) message in
    # msgs_before that the prune removed without relinking.
    before_uuids: set[str] = {
        msg.get("uuid", "") for _, msg, _ in msgs_before if msg.get("uuid")
    }
    for _, msg, _ in msgs_after:
        parent = msg.get("parentUuid")
        if not parent:
            continue
        if parent not in surviving_uuids:
            if parent not in before_uuids:
                # Parent was never in this file — cross-session pointer; skip.
                continue
            if parent in legit_removed_orphan_shells:
                # Parent was an orphan-shell legitimately dropped by orphan-fix;
                # not a prune-induced break — skip.
                continue
            # Parent was a real message in before_uuids (existed pre-prune) but
            # absent after, and it was NOT an orphan-shell — prune introduced
            # this chain break. Raise C1.
            raise PruneValidationError(
                reason=(
                    f"parentUuid {parent!r} on uuid {msg.get('uuid')!r} "
                    f"resolved before prune but is absent after — "
                    f"prune introduced a chain break"
                ),
                evidence={
                    "failed_check": "C1",
                    "dangling_uuid": msg.get("uuid"),
                    "dangling_parent": parent,
                    "surviving_count": len(surviving_uuids),
                },
            )

    # ── C8: surviving tool_use keeps its tool_result (baseline-relative) ──────
    # The Anthropic API rejects an assistant ``tool_use`` block whose paired
    # ``tool_result`` is absent on resume — the mirror of the orphaned-
    # ``tool_result`` case ``fix_orphaned_tool_results`` already handles. A
    # removal strategy can drop the user message carrying a ``tool_result``
    # while leaving the assistant ``tool_use`` in place, producing a dangling
    # ``tool_use`` that is structurally valid (DAG resolves) but unresumable.
    # Baseline-relative, exactly like C1: only raise when the pairing EXISTED
    # in msgs_before and the prune broke it. A ``tool_use`` that was already
    # awaiting its result pre-prune (e.g. the session's final, in-flight turn)
    # is NOT a prune-induced break, so it is skipped.
    before_result_ids: set[str] = set()
    for _, m, _ in msgs_before:
        content = (m.get("message") or {}).get("content")
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "tool_result":
                    rid = hashable_str(blk.get("tool_use_id"))  # unhashable -> "" (R6)
                    if rid:
                        before_result_ids.add(rid)
    after_result_ids: set[str] = set()
    for _, m, _ in msgs_after:
        content = (m.get("message") or {}).get("content")
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "tool_result":
                    rid = hashable_str(blk.get("tool_use_id"))
                    if rid:
                        after_result_ids.add(rid)
    for _, msg, _ in msgs_after:
        content = (msg.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for blk in content:
            if not (isinstance(blk, dict) and blk.get("type") == "tool_use"):
                continue
            tid = hashable_str(blk.get("id"))
            if not tid:
                continue
            if tid not in after_result_ids and tid in before_result_ids:
                # tool_use survived but its tool_result (present pre-prune) was
                # dropped — dangling tool_use, unresumable. Raise C8.
                raise PruneValidationError(
                    reason=(
                        f"tool_use {tid!r} on uuid {msg.get('uuid')!r} survived "
                        f"but its tool_result (present before prune) was dropped "
                        f"— dangling tool_use is unresumable"
                    ),
                    evidence={
                        "failed_check": "C8",
                        "dangling_tool_use_id": tid,
                        "uuid": msg.get("uuid"),
                    },
                )


# ── P0-C — Floor preservation ─────────────────────────────────────────────────


def enforce_floor(
    msgs_before: list[tuple[int, dict, int]],
    msgs_after: list[tuple[int, dict, int]],
    *,
    cfg: FloorConfig,
) -> list[tuple[int, dict, int]]:
    """Re-add must-preserve messages dropped by strategies.

    Algorithm:

      1. Compute ``kept_uuids`` = uuids present in ``msgs_after``.
      2. Identify ``must_preserve_uuids`` from ``msgs_before``:
         (a) First parentUuid=null message (if ``preserve_first_message``).
         (b) Last ``preserve_last_k_turns`` user + assistant by line order.
         (c) Enough additional user/assistant to bring survival ≥
             ``(1 - max_user_assistant_drop_pct)``, most-recent first.
      3. Pair-counterpart closure (REVIEW-max C.5): when re-adding a message
         that carries a tool_use/tool_result, also add the paired counterpart
         to avoid orphaned blocks after orphan-fix.
      4. Re-insert the ORIGINAL msgs_before entries at positions that preserve
         line-index ordering.
      5. Re-run _relink_parent_chain to fix any newly-broken pointers.

    A replaced-in-place message (same uuid, modified payload) is already in
    ``kept_uuids``, so the floor does NOT revert the replacement — the
    truncated/modified version stays (REVIEW-round3 F.N4).
    """
    from .executor import _relink_parent_chain

    # ── Step 1: what survived the strategies ─────────────────────────────────
    kept_uuids: set[str] = {
        m.get("uuid", "") for _, m, _ in msgs_after if m.get("uuid")
    }

    # ── Step 2: must-preserve candidates ─────────────────────────────────────
    must_preserve: set[str] = set()

    # P0-A: If there is an active compact_boundary (hasPreservedSegment not set),
    # all messages at indices BEFORE the boundary are pre-boundary turns that
    # compact-summary-collapse legitimately dropped. Re-adding them would create a
    # 2-root DAG fork (root-0 and cb-1 both have parentUuid=None).
    # We compute the boundary line index here so step 2a/b/c can all skip pre-boundary.
    compact_boundary_line_idx = _last_active_compact_boundary_idx(msgs_before)
    # compact_boundary_line_idx is None when no active boundary exists (normal sessions).

    def _is_pre_boundary(candidate_idx: int) -> bool:
        """True iff the candidate is a pre-boundary turn that must not be re-added."""
        return (compact_boundary_line_idx is not None
                and candidate_idx < compact_boundary_line_idx)

    # preserve_first_message pins the first parentUuid=null root. NOTE: with the
    # default (True) this also satisfies validate_post_prune C2 (which requires an
    # original root to survive). Setting preserve_first_message=False together with
    # an aggressive root-dropping prune can therefore produce a result C2 rejects —
    # an unsupported combination (the guard will defer such prunes). It is left as
    # a gated knob rather than force-preserving here, because unconditionally
    # re-adding the original root would undo compact-summary-collapse, which
    # legitimately drops the pre-boundary root in favor of the compact summary.
    if cfg.preserve_first_message:
        for idx, msg, _ in msgs_before:
            if msg.get("parentUuid") is None and msg.get("uuid"):
                if not _is_pre_boundary(idx):
                    must_preserve.add(msg["uuid"])
                break

    # Pre-boundary turns must not be re-added (P0-A). Exclude them from all
    # must-preserve candidate lists so last_k and survival-cap never pin them.
    users_in_order = [
        (idx, m) for idx, m, _ in msgs_before
        if m.get("type") == "user" and not _is_pre_boundary(idx)
    ]
    # Real conversational user turns exclude tool_result-carrier user messages
    # (whose content is tool_result blocks, not a user utterance) — used for the
    # last-K-turns floor so `preserve_last_k_turns=K` preserves K actual turns,
    # not K tool-result envelopes (the tool pairs are re-added by step-3 closure).
    def _is_real_user_turn(m: dict) -> bool:
        c = (m.get("message") or {}).get("content", "")
        if isinstance(c, str):
            return True
        if isinstance(c, list):
            return any(isinstance(b, dict) and b.get("type") == "text" for b in c)
        return False

    user_turns_in_order = [
        (idx, m) for idx, m, _ in msgs_before
        if m.get("type") == "user" and _is_real_user_turn(m) and not _is_pre_boundary(idx)
    ]
    asst_in_order = [
        (idx, m) for idx, m, _ in msgs_before
        if m.get("type") == "assistant" and not _is_pre_boundary(idx)
    ]

    last_k = max(0, int(cfg.preserve_last_k_turns))
    for _, m in (user_turns_in_order[-last_k:] if last_k > 0 else []):
        if m.get("uuid"):
            must_preserve.add(m["uuid"])
    for _, m in (asst_in_order[-last_k:] if last_k > 0 else []):
        if m.get("uuid"):
            must_preserve.add(m["uuid"])

    # (c) Survival cap: top up to (1 - max_drop_pct) of each kind, most-recent first.
    # REVIEW-max E.6: use math.ceil for the survival-target rounding. Skip the cap
    # entirely on micro-sessions (total < 2) — intended for bulk-prune disasters,
    # not single-message sessions.
    survival_floor_pct = 1.0 - float(cfg.max_user_assistant_drop_pct)
    if survival_floor_pct > 0.0:
        for in_order in (users_in_order, asst_in_order):
            total = len(in_order)
            if total < 2:
                continue
            preserved = sum(
                1 for _, m in in_order
                if (u := m.get("uuid", "")) and (u in kept_uuids or u in must_preserve)
            )
            target = math.ceil(survival_floor_pct * total)
            if preserved >= target:
                continue
            for _, m in reversed(in_order):
                if preserved >= target:
                    break
                u = m.get("uuid", "")
                if not u or u in kept_uuids or u in must_preserve:
                    continue
                must_preserve.add(u)
                preserved += 1

    # ── Step 3: pair-counterpart closure (REVIEW-max C.5) ────────────────────
    # When re-adding a message that carries a tool_use or tool_result, also add
    # its paired counterpart. Repeat until stable (normally 1-hop).
    before_by_uuid: dict[str, tuple[int, dict, int]] = {
        m.get("uuid", ""): (idx, m, size)
        for idx, m, size in msgs_before
        if m.get("uuid")
    }
    # Build set-valued maps (defensive against malformed sessions with duplicate ids).
    tool_use_id_to_owner: dict[str, set[str]] = {}
    tool_use_id_to_results: dict[str, set[str]] = {}
    for _, m, _ in msgs_before:
        u = m.get("uuid", "")
        if not u:
            continue
        content = m.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use":
                tid = hashable_str(block.get("id"))  # unhashable id -> "" (R6 crash class)
                if tid:
                    tool_use_id_to_owner.setdefault(tid, set()).add(u)
            elif btype == "tool_result":
                tid = hashable_str(block.get("tool_use_id"))
                if tid:
                    tool_use_id_to_results.setdefault(tid, set()).add(u)

    while True:
        new_additions: set[str] = set()
        for u in must_preserve:
            entry = before_by_uuid.get(u)
            if entry is None:
                continue
            _, m, _ = entry
            content = m.get("message", {}).get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "tool_use":
                    tid = hashable_str(block.get("id"))  # unhashable id -> "" (R6 crash class)
                    for p in tool_use_id_to_results.get(tid, set()):
                        # M-1 (review): don't re-add a PRE-boundary pair partner past an
                        # active compact_boundary — else step-3 pair-closure re-introduces a
                        # pre-boundary root (the 2-root fork P0-A prevents elsewhere; C9 would
                        # otherwise have to abort the prune).
                        pe = before_by_uuid.get(p)
                        if p not in must_preserve and not (pe and _is_pre_boundary(pe[0])):
                            new_additions.add(p)
                elif btype == "tool_result":
                    tid = hashable_str(block.get("tool_use_id"))
                    for owner in tool_use_id_to_owner.get(tid, set()):
                        oe = before_by_uuid.get(owner)
                        if owner not in must_preserve and not (oe and _is_pre_boundary(oe[0])):
                            new_additions.add(owner)
        if not new_additions:
            break
        must_preserve.update(new_additions)

    # ── Step 4: re-insert dropped must-preserve entries in line-index order ──
    # `to_re_add = must_preserve − kept_uuids` is disjoint from `kept_uuids`
    # by the definition of set difference (L-2: removed the tautological assert
    # that was always True). In-place replacements (same uuid, modified payload)
    # remain in `kept_uuids` → never re-inserted from msgs_before, so strategy
    # replacements are preserved (REVIEW-round3 F.N4).
    to_re_add = must_preserve - kept_uuids
    if not to_re_add:
        return msgs_after

    re_add_entries = [before_by_uuid[u] for u in to_re_add if u in before_by_uuid]

    # Merge sorted by line index to preserve JSONL line-order invariant.
    merged = list(msgs_after) + re_add_entries
    merged.sort(key=lambda t: t[0])

    # ── Step 5: re-link parent chains ────────────────────────────────────────
    # Re-added entries may have their original parentUuid pointing to a
    # still-removed ancestor; relink resolves the chain forward.
    merged_uuids = {m.get("uuid", "") for _, m, _ in merged if m.get("uuid")}
    effective_removals: set[int] = set()
    for idx, msg, _ in msgs_before:
        u = msg.get("uuid", "")
        if u and u not in merged_uuids:
            effective_removals.add(idx)

    return _relink_parent_chain(msgs_before, merged, removals=effective_removals)
