"""RED tests for floor preservation (P0-C).

Per PLAN.md § 2.4:
- `enforce_floor(msgs_before, msgs_after, *, cfg: FloorConfig)` re-adds
  must-preserve messages that fell to the prune. The algorithm:
    1. Compute kept_uuids from msgs_after.
    2. Identify must_preserve_uuids from msgs_before:
       (a) The first parentUuid=null message (if preserve_first_message).
       (b) The most recent K user + K assistant messages by line order.
       (c) Enough additional user/assistant messages to bring user/assistant
           survival ≥ (1.0 - max_user_assistant_drop_pct), most-recent first.
    3. For each must_preserve_uuid not in kept_uuids: re-insert the original
       message at the position that preserves line-index ordering.
    4. Re-run _relink_parent_chain after the insert.
    5. Return the floor-enforced msgs_after.

`enforce_floor` is invoked from `executor.run_prescription` AFTER
`fix_orphaned_tool_results` and BEFORE `validate_post_prune`.

These tests SHOULD FAIL until P0-C lands (`cozempic.safety.enforce_floor`
and `FloorConfig` do not exist yet, and `run_prescription` does not call
them).
"""

from __future__ import annotations

import unittest

from cozempic.helpers import msg_bytes


try:
    from cozempic import safety as _safety  # type: ignore  # noqa: F401
    _SAFETY_AVAILABLE = True
except ImportError:
    _SAFETY_AVAILABLE = False


# ─── Shared fixture builders ─────────────────────────────────────────────────


def _msg(idx: int, payload: dict) -> tuple[int, dict, int]:
    return (idx, payload, msg_bytes(payload))


def _build_chain(n_user: int, n_asst: int) -> list[tuple[int, dict, int]]:
    """Build a chain of n_user users and n_asst assistants, interleaved with
    valid parent pointers, rooted at parentUuid=None."""
    msgs: list[tuple[int, dict, int]] = []
    line = 0
    prev = None
    # First message is the root user.
    msgs.append(_msg(line, {
        "type": "user", "uuid": "u0", "parentUuid": None,
        "message": {"role": "user", "content": "u0"},
    }))
    prev = "u0"
    line += 1
    for i in range(min(n_asst, n_user)):
        if i > 0:
            uid = f"u{i}"
            msgs.append(_msg(line, {
                "type": "user", "uuid": uid, "parentUuid": prev,
                "message": {"role": "user", "content": uid},
            }))
            prev = uid
            line += 1
        aid = f"a{i}"
        msgs.append(_msg(line, {
            "type": "assistant", "uuid": aid, "parentUuid": prev,
            "message": {"role": "assistant",
                        "content": [{"type": "text", "text": aid}]},
        }))
        prev = aid
        line += 1
    return msgs


@unittest.skipUnless(_SAFETY_AVAILABLE, "P0-C safety module not yet implemented (expected RED)")
class TestFloorReAddsDropped(unittest.TestCase):
    """Floor brings survival up to the cap, preferring most recent messages."""

    def test_floor_re_adds_dropped_user_assistant_when_below_threshold(self):
        """100 users + 100 assistants before, after has 10+10 → floor brings to ≥50%."""
        from cozempic.safety import FloorConfig, enforce_floor  # type: ignore

        before = _build_chain(n_user=100, n_asst=100)
        users = [m for m in before if m[1].get("type") == "user"]
        asst = [m for m in before if m[1].get("type") == "assistant"]
        # Keep only the LAST 10 of each (very aggressive prune).
        keep = ({m[1]["uuid"] for m in users[-10:]}
                | {m[1]["uuid"] for m in asst[-10:]})
        after = [m for m in before if m[1].get("uuid") in keep]

        cfg = FloorConfig(
            max_user_assistant_drop_pct=0.50,
            preserve_last_k_turns=10,
            preserve_first_message=True,
        )
        re_added = enforce_floor(before, after, cfg=cfg)

        before_ua = len(users) + len(asst)
        re_added_ua = sum(
            1 for _, m, _ in re_added if m.get("type") in ("user", "assistant")
        )
        self.assertGreaterEqual(re_added_ua / before_ua, 0.50)


@unittest.skipUnless(_SAFETY_AVAILABLE, "P0-C safety module not yet implemented (expected RED)")
class TestFloorPreservesLastK(unittest.TestCase):
    """The last K turns of each role survive any prune, regardless of position."""

    def test_floor_preserves_last_k_turns(self):
        from cozempic.safety import FloorConfig, enforce_floor  # type: ignore

        before = _build_chain(n_user=100, n_asst=100)
        users = [m for m in before if m[1].get("type") == "user"]
        asst = [m for m in before if m[1].get("type") == "assistant"]
        last_k_user = {m[1]["uuid"] for m in users[-10:]}
        last_k_asst = {m[1]["uuid"] for m in asst[-10:]}
        # Strategy drops THE ENTIRE LAST K BLOCK + every other thing.
        dropped = last_k_user | last_k_asst
        after = [m for m in before if m[1].get("uuid") not in dropped]

        cfg = FloorConfig(
            max_user_assistant_drop_pct=0.50,
            preserve_last_k_turns=10,
            preserve_first_message=True,
        )
        re_added = enforce_floor(before, after, cfg=cfg)
        re_added_uuids = {m[1].get("uuid") for m in re_added}

        for uuid in last_k_user | last_k_asst:
            self.assertIn(uuid, re_added_uuids)


@unittest.skipUnless(_SAFETY_AVAILABLE, "P0-C safety module not yet implemented (expected RED)")
class TestFloorPreservesFirstMessage(unittest.TestCase):
    """The first message (parentUuid=None root) survives any prune."""

    def test_floor_preserves_first_message(self):
        from cozempic.safety import FloorConfig, enforce_floor  # type: ignore

        before = _build_chain(n_user=100, n_asst=100)
        # Strategy drops the root.
        after = [m for m in before if m[1].get("uuid") != "u0"]

        cfg = FloorConfig(preserve_first_message=True)
        re_added = enforce_floor(before, after, cfg=cfg)
        re_added_uuids = {m[1].get("uuid") for m in re_added}

        self.assertIn("u0", re_added_uuids)

    def test_floor_disabled_for_first_message_when_flag_off(self):
        """preserve_first_message=False permits the root to be dropped (defensive)."""
        from cozempic.safety import FloorConfig, enforce_floor  # type: ignore

        before = _build_chain(n_user=100, n_asst=100)
        after = [m for m in before if m[1].get("uuid") != "u0"]

        cfg = FloorConfig(
            preserve_first_message=False,
            preserve_last_k_turns=0,
            max_user_assistant_drop_pct=1.0,  # allow any drop
        )
        re_added = enforce_floor(before, after, cfg=cfg)
        re_added_uuids = {m[1].get("uuid") for m in re_added}

        self.assertNotIn("u0", re_added_uuids)


@unittest.skipUnless(_SAFETY_AVAILABLE, "P0-C safety module not yet implemented (expected RED)")
class TestFloorIsNoOpWhenStrategiesRespectFloor(unittest.TestCase):
    """A gentle prune that already respects the floor → no re-adds."""

    def test_floor_no_op_when_strategies_respect_floor(self):
        from cozempic.safety import FloorConfig, enforce_floor  # type: ignore

        before = _build_chain(n_user=100, n_asst=100)
        users = [m for m in before if m[1].get("type") == "user"]
        asst = [m for m in before if m[1].get("type") == "assistant"]
        # Drop only 10% of each (well within 50% cap, includes last K).
        drop_u = {u[1]["uuid"] for u in users[10:20]}
        drop_a = {a[1]["uuid"] for a in asst[10:20]}
        dropped = drop_u | drop_a
        after = [m for m in before if m[1].get("uuid") not in dropped]

        cfg = FloorConfig(
            max_user_assistant_drop_pct=0.50,
            preserve_last_k_turns=10,
            preserve_first_message=True,
        )
        re_added = enforce_floor(before, after, cfg=cfg)
        re_added_uuids = {m[1].get("uuid") for m in re_added}
        after_uuids = {m[1].get("uuid") for m in after}

        # The result equals msgs_after exactly — no uuids re-added.
        self.assertEqual(re_added_uuids, after_uuids)


@unittest.skipUnless(_SAFETY_AVAILABLE, "P0-C safety module not yet implemented (expected RED)")
class TestFloorRunsBeforeValidation(unittest.TestCase):
    """Order matters: floor must run BEFORE validate_post_prune in run_prescription.

    Without this ordering, a strategy that drops every user would trigger C3,
    and validation would abort — even though the floor would have re-added users
    to bring survival above 0. The test checks the integration: an aggressive
    strategy that drops the root + most users on a session is rescued by the floor.
    """

    def test_floor_rescues_aggressive_prune_so_validation_passes(self):
        from cozempic.executor import run_prescription
        from cozempic.registry import strategy, STRATEGIES
        from cozempic.types import PruneAction, StrategyResult

        @strategy("test-drop-most-users", "Test only", "gentle", "90%")
        def _drop_most(messages, config):  # noqa: ANN001
            actions = []
            users = [(idx, msg, size)
                     for idx, msg, size in messages if msg.get("type") == "user"]
            # Drop all but the last 2 users.
            for idx, msg, size in users[:-2]:
                actions.append(PruneAction(
                    line_index=idx, action="remove",
                    reason="test", original_bytes=size, pruned_bytes=0,
                ))
            return StrategyResult(
                strategy_name="test-drop-most-users", actions=actions,
                original_bytes=sum(b for _, _, b in messages),
                pruned_bytes=sum(a.original_bytes for a in actions),
                messages_affected=len(actions),
                messages_removed=len(actions),
                messages_replaced=0,
                summary="dropped most users",
            )

        try:
            before = _build_chain(n_user=100, n_asst=100)
            # Should complete: floor re-adds users to ≥50%, validation then passes.
            after, _ = run_prescription(before, ["test-drop-most-users"], {})
            # Sanity: at least one user, at least one assistant.
            self.assertTrue(any(m[1].get("type") == "user" for m in after))
            self.assertTrue(any(m[1].get("type") == "assistant" for m in after))
        finally:
            STRATEGIES.pop("test-drop-most-users", None)


@unittest.skipUnless(_SAFETY_AVAILABLE, "P0-C safety module not yet implemented (expected RED)")
class TestFloorDoesNotUndoReplacements(unittest.TestCase):
    """When a strategy replaces a message in-place (uuid preserved), the floor
    must not re-add the ORIGINAL — the replacement is the kept one.

    enforce_floor's must_preserve check is based on `kept_uuids` from msgs_after.
    A replaced message keeps its uuid → is in kept_uuids → floor sees no gap.
    """

    def test_floor_keeps_replacement(self):
        from cozempic.safety import FloorConfig, enforce_floor  # type: ignore

        before = _build_chain(n_user=10, n_asst=10)
        # In-place replace: same uuid, smaller payload.
        after: list[tuple[int, dict, int]] = []
        for idx, msg, size in before:
            if msg.get("uuid") == "u0":
                truncated = {
                    **msg,
                    "message": {"role": "user", "content": "[truncated]"},
                }
                after.append((idx, truncated, msg_bytes(truncated)))
            else:
                after.append((idx, msg, size))

        cfg = FloorConfig(preserve_first_message=True)
        re_added = enforce_floor(before, after, cfg=cfg)

        # u0 must be present exactly ONCE, and it must be the TRUNCATED version
        # (smaller payload), not the original.
        u0_entries = [m for m in re_added if m[1].get("uuid") == "u0"]
        self.assertEqual(len(u0_entries), 1)
        content = u0_entries[0][1].get("message", {}).get("content", "")
        self.assertEqual(content, "[truncated]")


if __name__ == "__main__":
    unittest.main()
