"""RED test for review-max Group C — floor ⇔ orphan-fix pair preservation.

Finding C.5: after the C-1 swap (floor BEFORE orphan-fix), orphan-fix can
drop a floor-resurrected user whose only blocks are tool_results pointing
at dropped tool_uses. Concrete scenario: floor's preserve_first_message
rescues the root user carrying only a tool_result; orphan-fix strips the
block; the message has 0 blocks left and orphan-fix DROPS it entirely.

Fix: extend ``enforce_floor`` so that when re-adding a message containing
``tool_use``/``tool_result`` blocks, it ALSO re-adds the paired counterpart
(the message holding the other end of the pair) from ``msgs_before``.

RED on pre-fix code: pair-counterpart is NOT pulled in; orphan-fix strips
the lone tool_result → drops the message → the original root vanishes.
GREEN after fix: both halves of the pair survive together, the root with
its tool_result remains intact, and orphan-fix is a no-op.
"""

from __future__ import annotations

import unittest

from cozempic.helpers import msg_bytes


def _msg(idx: int, payload: dict) -> tuple[int, dict, int]:
    return (idx, payload, msg_bytes(payload))


class TestFloorPullsInPairedCounterpart(unittest.TestCase):
    """C.5 — when the floor re-adds a message holding a tool_result block,
    it must also re-add the message holding the matching tool_use, so the
    pair survives orphan-fix as a whole."""

    def test_floor_re_adds_paired_tool_use_when_re_adding_tool_result(self):
        from cozempic.executor import run_prescription
        from cozempic.registry import strategy, STRATEGIES
        from cozempic.types import PruneAction, StrategyResult

        TOOL_USE_ID = "tool-use-paired"
        u_tool_uuid = "u_tool"
        a_tool_uuid = "a_tool"

        # Root user carries ONLY a tool_result (no text block — so if
        # orphan-fix strips the tool_result the whole message is dropped).
        # Following assistant carries the matching tool_use. Tail filler
        # ensures last-K does not rescue a_tool.
        messages: list[tuple[int, dict, int]] = []
        line = 0
        messages.append(_msg(line, {
            "type": "user", "uuid": u_tool_uuid, "parentUuid": None,
            "message": {
                "role": "user",
                "content": [{"type": "tool_result",
                             "tool_use_id": TOOL_USE_ID,
                             "content": "ok"}],
            },
        }))
        line += 1
        messages.append(_msg(line, {
            "type": "assistant", "uuid": a_tool_uuid,
            "parentUuid": u_tool_uuid,
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": TOOL_USE_ID,
                             "name": "Bash", "input": {"cmd": "ls"}}],
            },
        }))
        line += 1
        prev = a_tool_uuid

        # Tail filler — 200 user/assistant turns saturating the last-K=50
        # window. The pair sits at positions 0-1; outside last-K.
        for i in range(1, 201):
            uid = f"u{i}"
            messages.append(_msg(line, {
                "type": "user", "uuid": uid, "parentUuid": prev,
                "message": {"role": "user", "content": f"u {i}"},
            }))
            line += 1
            aid = f"a{i}"
            messages.append(_msg(line, {
                "type": "assistant", "uuid": aid, "parentUuid": uid,
                "message": {"role": "assistant",
                            "content": [{"type": "text", "text": f"a {i}"}]},
            }))
            line += 1
            prev = aid

        # Strategy drops the entire pair.
        @strategy("test-c5-drop", "Test only", "gentle", "n/a")
        def _drop(msgs, config):  # noqa: ANN001
            actions = []
            for idx, m, size in msgs:
                if m.get("uuid") in (u_tool_uuid, a_tool_uuid):
                    actions.append(PruneAction(
                        line_index=idx, action="remove",
                        reason="t", original_bytes=size, pruned_bytes=0,
                    ))
            return StrategyResult(
                strategy_name="test-c5-drop", actions=actions,
                original_bytes=sum(b for _, _, b in msgs),
                pruned_bytes=sum(a.original_bytes for a in actions),
                messages_affected=len(actions),
                messages_removed=len(actions), messages_replaced=0,
                summary="",
            )

        try:
            after, _ = run_prescription(messages, ["test-c5-drop"], {})
        finally:
            STRATEGIES.pop("test-c5-drop", None)

        survivor_uuids = {m[1].get("uuid") for m in after}
        # The pair must survive together — both halves present.
        self.assertIn(
            u_tool_uuid, survivor_uuids,
            msg="floor must pull in u_tool (carries tool_result) — root",
        )
        self.assertIn(
            a_tool_uuid, survivor_uuids,
            msg="floor must pull in a_tool (carries tool_use) as the paired "
                "counterpart of the re-added u_tool, otherwise orphan-fix "
                "strips the lone tool_result and the floor's intent is lost",
        )

        # And the tool_result block is intact in u_tool.
        for _, m, _ in after:
            if m.get("uuid") == u_tool_uuid:
                content = m.get("message", {}).get("content", [])
                tool_results = [b for b in content
                                if isinstance(b, dict)
                                and b.get("type") == "tool_result"]
                self.assertEqual(
                    len(tool_results), 1,
                    msg="u_tool's tool_result block should survive — its "
                        "tool_use was floor-rescued so it is no longer orphan",
                )


if __name__ == "__main__":
    unittest.main()
