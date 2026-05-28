"""Safety guards for the session pruner.

Implements the four protections introduced by the resume-break fix
(see PR description and PLAN.md):

P0-A — Active-session idle guard (this commit):
  - `is_session_idle(path, min_idle_hours)`
  - `assert_session_idle_or_force(path, min_idle_hours, force)`
  - `ActiveSessionError` exception with path / idle_minutes / threshold_hours
  - `resolve_min_idle_hours()` env-aware threshold resolver

P0-B — Post-prune structural validation (commit 2):
  - `validate_post_prune(msgs_before, msgs_after, strict)`
  - `PruneValidationError` exception with reason + evidence dict
  - `simulate_replay_readiness(messages)` structural replay probe

P0-C — Floor preservation (commit 3):
  - `FloorConfig` dataclass (re-exported from config.py for ergonomics)
  - `enforce_floor(msgs_before, msgs_after, cfg)`

P0-D — Last-of-type metadata singleton protection (commit 4):
  - Helpers used by executor; see `executor._tag_last_of_metadata_types`.
"""

from __future__ import annotations

from pathlib import Path

from .config import FloorConfig, resolve_min_idle_hours

__all__ = [
    "ActiveSessionError",
    "PruneValidationError",
    "FloorConfig",
    "is_session_idle",
    "assert_session_idle_or_force",
    "resolve_min_idle_hours",
    "validate_post_prune",
    "enforce_floor",
    "simulate_replay_readiness",
]


# ── P0-A — Active-session idle guard ───────────────────────────────────────


class ActiveSessionError(Exception):
    """Raised when a prune is blocked by the active-session guard.

    Carries enough context for the operator to act:
      - ``path``: the session JSONL that was rejected
      - ``idle_minutes``: how long since the file was last modified
      - ``threshold_hours``: the active threshold at refusal time

    The string representation is suitable for direct emission to stderr or
    CLI exit-code 4 paths.
    """

    def __init__(self, path: Path, idle_minutes: float, threshold_hours: float):
        self.path = path
        self.idle_minutes = idle_minutes
        self.threshold_hours = threshold_hours
        super().__init__(
            f"Session {path.name} was modified {idle_minutes:.1f} min ago "
            f"(< {threshold_hours}h idle threshold). Refusing prune. "
            f"Use --force to override or wait for the session to idle."
        )


def is_session_idle(path: Path, min_idle_hours: float) -> tuple[bool, float]:
    """Return ``(is_idle, minutes_since_last_mtime)``.

    ``is_idle`` is ``True`` when the file's last modification predates
    ``now - min_idle_hours``. ``minutes_since_last_mtime`` is the elapsed
    minutes since ``stat().st_mtime``.

    A missing file is treated as idle (nothing to protect) and reports
    ``minutes`` as ``inf`` so callers can log a meaningful value.
    """
    import time

    try:
        mtime = path.stat().st_mtime
    except (FileNotFoundError, OSError):
        return True, float("inf")

    elapsed_seconds = max(0.0, time.time() - mtime)
    elapsed_minutes = elapsed_seconds / 60.0
    threshold_minutes = float(min_idle_hours) * 60.0
    return elapsed_minutes >= threshold_minutes, elapsed_minutes


def assert_session_idle_or_force(
    path: Path,
    *,
    min_idle_hours: float,
    force: bool,
) -> None:
    """Raise ``ActiveSessionError`` if the session is not idle and ``force`` is False.

    ``min_idle_hours`` of 0.0 disables the guard entirely (every session
    counts as idle). ``force=True`` is the explicit operator override; the
    overflow recovery path uses it for its emergency prune.
    """
    if force:
        return
    if min_idle_hours <= 0.0:
        return
    idle, minutes = is_session_idle(path, min_idle_hours)
    if not idle:
        raise ActiveSessionError(
            path=path,
            idle_minutes=minutes,
            threshold_hours=float(min_idle_hours),
        )


# ── P0-B — Post-prune validation (implemented in commit 2) ─────────────────


class PruneValidationError(Exception):
    """Raised when the pruned message list fails structural validation.

    ``reason`` is a human-readable summary. ``evidence`` is a dict that
    callers (guard daemon, CLI) log; it always contains a ``failed_check``
    key matching one of ``"C1".."C6"`` so log aggregators can group failures.
    """

    def __init__(self, reason: str, evidence: dict):
        self.reason = reason
        self.evidence = evidence
        super().__init__(f"Pruned session would not replay cleanly: {reason}")


def _last_of_type(messages: list[tuple[int, dict, int]], msg_type: str) -> dict | None:
    """Return the last message dict whose ``type`` equals ``msg_type``."""
    last: dict | None = None
    for _, msg, _ in messages:
        if msg.get("type") == msg_type:
            last = msg
    return last


def _last_compact_boundary(
    messages: list[tuple[int, dict, int]],
) -> dict | None:
    last: dict | None = None
    for _, msg, _ in messages:
        if (msg.get("type") == "system"
                and msg.get("subtype") == "compact_boundary"):
            last = msg
    return last


def validate_post_prune(
    msgs_before: list[tuple[int, dict, int]],
    msgs_after: list[tuple[int, dict, int]],
    *,
    strict: bool = True,
) -> None:
    """Validate the pruned message list. Raise PruneValidationError on failure.

    Checks (fail-fast, in order):
      C1. parentUuid resolution — every non-null parentUuid in msgs_after must
          point to a uuid that exists in msgs_after.
      C2. Root preserved — if msgs_before had at least one parentUuid=null
          entry, msgs_after MUST also have at least one.
      C3. Conversation survival — ≥1 user AND ≥1 assistant survives.
      C4. compact_boundary — if msgs_before had a system/subtype=
          compact_boundary entry, the LAST such entry MUST survive.
      C5. permission-mode — if msgs_before had permission-mode entries, the
          LAST one MUST survive.
      C6. last-prompt — if msgs_before had last-prompt entries, the LAST one
          MUST survive.

    Each check is CONDITIONAL on its precondition existing in msgs_before —
    a session that never had a permission-mode entry passes C5 trivially.
    """
    surviving_uuids: set[str] = {
        msg.get("uuid", "") for _, msg, _ in msgs_after if msg.get("uuid")
    }

    # Check order: semantic checks (C3, C2, C4-C6) run BEFORE the structural
    # C1 parent-chain probe, because a wholesale conversation wipe or missing
    # root necessarily breaks the chain for descendants — reporting C1 in
    # that case would mask the underlying cause. C3 (conversation wipe) is
    # checked before C2 (root) because the conversation IS the user-visible
    # symptom; in practice losing every user also loses the root, but the
    # error is more actionable when reported as "no conversation".
    # C1 is the structural fallback for the case where root + conversation
    # + metadata are intact but a middle link was dropped without re-linking.

    # ── C3: conversation survival ───────────────────────────────────────────
    surviving_user = sum(
        1 for _, m, _ in msgs_after if m.get("type") == "user"
    )
    surviving_asst = sum(
        1 for _, m, _ in msgs_after if m.get("type") == "assistant"
    )
    before_user = any(m.get("type") == "user" for _, m, _ in msgs_before)
    before_asst = any(m.get("type") == "assistant" for _, m, _ in msgs_before)
    if (before_user and surviving_user == 0) or (
        before_asst and surviving_asst == 0
    ):
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

    # ── C2: root preserved ──────────────────────────────────────────────────
    before_has_root = any(
        msg.get("parentUuid") is None for _, msg, _ in msgs_before
    )
    after_has_root = any(
        msg.get("parentUuid") is None for _, msg, _ in msgs_after
    )
    if before_has_root and not after_has_root:
        raise PruneValidationError(
            reason="session root (parentUuid=null) was dropped",
            evidence={
                "failed_check": "C2",
                "before_count": len(msgs_before),
                "after_count": len(msgs_after),
            },
        )

    # ── C4: last compact_boundary preserved ─────────────────────────────────
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
                    "actual_uuid": (
                        last_after_cb.get("uuid") if last_after_cb else None
                    ),
                },
            )

    # ── C5: last permission-mode preserved ──────────────────────────────────
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
                    "actual_uuid": (
                        last_after_pm.get("uuid") if last_after_pm else None
                    ),
                },
            )

    # ── C6: last last-prompt preserved ──────────────────────────────────────
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
                    "actual_uuid": (
                        last_after_lp.get("uuid") if last_after_lp else None
                    ),
                },
            )

    # ── C1: parent chain resolves ───────────────────────────────────────────
    # Defense-in-depth fallback. The executor's _relink_parent_chain step
    # SHOULD ensure every surviving parentUuid resolves; this re-verifies.
    for _, msg, _ in msgs_after:
        parent = msg.get("parentUuid")
        if parent is None:
            continue
        if parent not in surviving_uuids:
            raise PruneValidationError(
                reason=(
                    f"parentUuid {parent!r} on uuid {msg.get('uuid')!r} "
                    f"does not resolve to a surviving message"
                ),
                evidence={
                    "failed_check": "C1",
                    "dangling_uuid": msg.get("uuid"),
                    "dangling_parent": parent,
                    "surviving_count": len(surviving_uuids),
                },
            )


def simulate_replay_readiness(
    messages: list[tuple[int, dict, int]],
) -> tuple[bool, str]:
    """Structural probe: walk the parentUuid graph as Claude Code's resume would.

    Returns ``(ok, reason)``. ``ok=False`` indicates the session would not
    bootstrap cleanly — e.g., missing root, dangling parentUuid chain, or no
    conversational content. ``ok=True`` returns reason="".

    This is a pure-Python simulation. No Claude binary is spawned.
    """
    if not messages:
        return False, "empty message list"

    surviving_uuids: set[str] = {
        m.get("uuid", "") for _, m, _ in messages if m.get("uuid")
    }
    roots = [m for _, m, _ in messages if m.get("parentUuid") is None]
    if not roots:
        return False, "no root (no parentUuid=null entry)"

    # Walk every entry; chain must resolve.
    for _, msg, _ in messages:
        parent = msg.get("parentUuid")
        if parent is None:
            continue
        if parent not in surviving_uuids:
            return False, (
                f"chain break: parentUuid {parent!r} on uuid "
                f"{msg.get('uuid')!r} does not resolve"
            )

    # Conversation must include at least one user AND one assistant.
    has_user = any(m.get("type") == "user" for _, m, _ in messages)
    has_asst = any(m.get("type") == "assistant" for _, m, _ in messages)
    if not (has_user and has_asst):
        return False, "no conversation (zero users or zero assistants)"

    return True, ""


# ── P0-C — Floor preservation (implemented in commit 3) ────────────────────


def enforce_floor(
    msgs_before: list[tuple[int, dict, int]],
    msgs_after: list[tuple[int, dict, int]],
    *,
    cfg: FloorConfig,
) -> list[tuple[int, dict, int]]:
    """Re-add must-preserve messages dropped by strategies.

    Algorithm (PLAN § 2.4):

      1. Compute ``kept_uuids`` = uuids present in ``msgs_after``.
      2. Identify ``must_preserve_uuids`` from ``msgs_before``:
         (a) First parentUuid=null message (if ``preserve_first_message``).
         (b) Last ``preserve_last_k_turns`` user + assistant by line order.
         (c) Enough additional user/assistant to bring survival ≥
             ``(1 - max_user_assistant_drop_pct)``, most-recent first.
      3. For each must-preserve uuid not in kept: re-insert the ORIGINAL
         msgs_before entry at the position that preserves line-index
         ordering (so the JSONL stays roughly chronological).
      4. Re-run executor._relink_parent_chain to fix any newly-broken
         pointers (re-adding a message can resurrect a uuid that other
         messages were dangling against).

    A replaced-in-place message (same uuid, modified payload) is in
    ``kept_uuids`` already, so the floor does NOT undo the replacement —
    the truncated version stays.
    """
    from .executor import _relink_parent_chain

    # ── Step 1 ──────────────────────────────────────────────────────────────
    kept_uuids: set[str] = {
        m.get("uuid", "") for _, m, _ in msgs_after if m.get("uuid")
    }

    # ── Step 2: must-preserve ───────────────────────────────────────────────
    must_preserve: set[str] = set()

    if cfg.preserve_first_message:
        for _, msg, _ in msgs_before:
            if msg.get("parentUuid") is None and msg.get("uuid"):
                must_preserve.add(msg["uuid"])
                break

    # (b) Last K user + assistant by line order.
    users_in_order = [
        (idx, m) for idx, m, _ in msgs_before if m.get("type") == "user"
    ]
    asst_in_order = [
        (idx, m) for idx, m, _ in msgs_before if m.get("type") == "assistant"
    ]
    last_k = max(0, int(cfg.preserve_last_k_turns))
    for _, m in users_in_order[-last_k:] if last_k > 0 else []:
        if m.get("uuid"):
            must_preserve.add(m["uuid"])
    for _, m in asst_in_order[-last_k:] if last_k > 0 else []:
        if m.get("uuid"):
            must_preserve.add(m["uuid"])

    # (c) Top up user/assistant survival to the floor cap (most-recent first).
    # The cap is on the dropped fraction, so survival ≥ (1 - max_drop_pct).
    survival_floor_pct = 1.0 - float(cfg.max_user_assistant_drop_pct)
    if survival_floor_pct > 0.0:
        for kind, in_order in (("user", users_in_order),
                                ("assistant", asst_in_order)):
            total = len(in_order)
            if total == 0:
                continue
            # Count what we already preserve (in kept OR in must_preserve).
            preserved = 0
            for _, m in in_order:
                u = m.get("uuid", "")
                if u and (u in kept_uuids or u in must_preserve):
                    preserved += 1
            target = int(survival_floor_pct * total + 0.999999)
            if preserved >= target:
                continue
            # Walk msgs_before from newest to oldest, adding entries until
            # we hit the target.
            for _, m in reversed(in_order):
                if preserved >= target:
                    break
                u = m.get("uuid", "")
                if not u:
                    continue
                if u in kept_uuids or u in must_preserve:
                    continue
                must_preserve.add(u)
                preserved += 1

    # ── Step 3: re-insert dropped must-preserve entries in line-index order ──
    to_re_add = must_preserve - kept_uuids
    if not to_re_add:
        return msgs_after

    # Look up the ORIGINAL entries from msgs_before by uuid.
    before_by_uuid: dict[str, tuple[int, dict, int]] = {
        m.get("uuid", ""): (idx, m, size)
        for idx, m, size in msgs_before
        if m.get("uuid")
    }
    re_add_entries = [before_by_uuid[u] for u in to_re_add if u in before_by_uuid]

    # Merge sorted by line index. This preserves the JSONL line order
    # invariant the downstream code (save_messages, _relink_parent_chain)
    # relies on.
    merged = list(msgs_after) + re_add_entries
    merged.sort(key=lambda t: t[0])

    # ── Step 4: re-link parent chains ────────────────────────────────────────
    # Some entries in msgs_after may have had their parentUuid re-pointed
    # away from a now-re-added message; re-running the relink with no
    # removals leaves valid pointers untouched.
    return _relink_parent_chain(msgs_before, merged, removals=set())
