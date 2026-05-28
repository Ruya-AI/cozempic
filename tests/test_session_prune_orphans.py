"""RED test for review finding C-1 — floor must not resurrect orphans.

If ``enforce_floor`` runs AFTER ``fix_orphaned_tool_results``, a re-added
``user`` carrying a ``tool_result`` block can leave the saved JSONL with an
orphan whose paired ``tool_use`` was dropped by an earlier strategy. The
docstring of ``fix_orphaned_tool_results`` itself states the consequence:
the Anthropic API returns 400 on resume.

Per the review fix: swap the pipeline order to
``strategies → enforce_floor → fix_orphaned_tool_results → validate``
so orphan-fix runs over the FINAL post-floor message list.

This test SHOULD fail RED on the pre-fix pipeline order and GREEN after
the swap.

Attack vector — getting only one half of the pair rescued by the floor:
  * The ``user`` carrying the ``tool_result`` is the ROOT of the session
    (``parentUuid=None``). The floor's ``preserve_first_message`` rule
    rescues it unconditionally.
  * The ``assistant`` carrying the matching ``tool_use`` is NOT root and
    is OUTSIDE the last-K window; the strategy drops it and no floor rule
    rescues it. The matching tool_use is gone.

This layout puts ``tool_result`` ahead of its ``tool_use`` in line order
which inverts the normal API sequence — but the orphan check is purely
structural and does not care about ordering, only paired presence. The
attack mirrors a real scenario where a strategy collapses ALL pre-boundary
assistants for byte savings, leaving a re-anchored user with an orphan.
"""

from __future__ import annotations

import unittest

from cozempic.helpers import msg_bytes


def _msg(idx: int, payload: dict) -> tuple[int, dict, int]:
    return (idx, payload, msg_bytes(payload))


class TestFloorDoesNotResurrectOrphans(unittest.TestCase):
    """C-1 — floor's re-add must not bypass orphan cleanup."""

    def test_floor_re_add_does_not_resurrect_orphaned_tool_result(self):
        from cozempic.executor import run_prescription
        from cozempic.registry import strategy, STRATEGIES
        from cozempic.types import PruneAction, StrategyResult, StrategyInfo

        TOOL_USE_ID = "tool-use-xyz"
        asst_tool_uuid = "a_tool"
        u_tool_uuid = "u_tool"

        # Layout:
        #   u_tool   (tool_result, parentUuid=None — ROOT)
        #   asst_tool (tool_use, parentUuid=u_tool)
        #   200 user/assistant filler turns
        messages: list[tuple[int, dict, int]] = []
        line = 0
        # u_tool also carries a text block so that orphan-fix stripping the
        # tool_result leaves a non-empty content list — otherwise orphan-fix
        # drops the entire message and we lose the root, masking the bug we
        # want to demonstrate.
        messages.append(_msg(line, {
            "type": "user", "uuid": u_tool_uuid, "parentUuid": None,
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "root prompt"},
                    {"type": "tool_result", "tool_use_id": TOOL_USE_ID,
                     "content": "ok"},
                ],
            },
        }))
        line += 1
        messages.append(_msg(line, {
            "type": "assistant", "uuid": asst_tool_uuid, "parentUuid": u_tool_uuid,
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": TOOL_USE_ID,
                             "name": "Bash", "input": {"cmd": "ls"}}],
            },
        }))
        line += 1
        prev = asst_tool_uuid

        # Fill the tail with 200 plain user/assistant turns so the last-K=50
        # window is fully saturated by recent entries (the tool pair is
        # outside last-K because it sits at positions 0-1 of the file).
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

        # Pathological strategy: drop both u_tool and asst_tool.
        #
        # After execute_actions:
        #   - execute_actions's T1.5 sibling protection scans only SURVIVORS.
        #     u_tool is being removed, so its tool_result is not scanned and
        #     asst_tool's tool_use ID does not end up in tool_result_refs →
        #     asst_tool's removal is honored (NOT rescued).
        # After fix_orphaned_tool_results:
        #   - u_tool already gone, no orphans to fix.
        # After enforce_floor:
        #   - preserve_first_message rescues u_tool (it is the only
        #     parentUuid=None entry in msgs_before).
        #   - asst_tool: NOT in last-K (sits at line 1), the surviving
        #     200 fillers easily satisfy the 50% assistant survival cap.
        #     asst_tool stays dropped.
        # Pre-fix pipeline (orphan-fix BEFORE floor):
        #   u_tool returns carrying an orphan tool_result → 400 on resume.
        # Post-fix pipeline (orphan-fix AFTER floor):
        #   orphan-fix sees u_tool's tool_result references a tool_use ID
        #   missing from the surviving tool_use set → strips the block.

        @strategy("test-c1-drop", "Test only", "gentle", "n/a")
        def _drop(msgs, config):  # noqa: ANN001
            actions = []
            for idx, m, size in msgs:
                if m.get("uuid") in (asst_tool_uuid, u_tool_uuid):
                    actions.append(PruneAction(
                        line_index=idx, action="remove",
                        reason="test-c1-drop", original_bytes=size,
                        pruned_bytes=0,
                    ))
            return StrategyResult(
                strategy_name="test-c1-drop", actions=actions,
                original_bytes=sum(b for _, _, b in msgs),
                pruned_bytes=sum(a.original_bytes for a in actions),
                messages_affected=len(actions),
                messages_removed=len(actions),
                messages_replaced=0,
                summary="c1 drop",
            )

        try:
            after, _ = run_prescription(messages, ["test-c1-drop"], {})
        finally:
            STRATEGIES.pop("test-c1-drop", None)

        # Collect every tool_use id present in the survivors.
        surviving_tool_use_ids: set[str] = set()
        for _, m, _ in after:
            inner = m.get("message", {})
            content = inner.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tid = block.get("id", "")
                        if tid:
                            surviving_tool_use_ids.add(tid)

        # Assert: no surviving message has a tool_result whose tool_use_id is
        # not in the surviving tool_use set.
        orphans: list[tuple[str, str]] = []
        for _, m, _ in after:
            inner = m.get("message", {})
            content = inner.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tid = block.get("tool_use_id", "")
                        if tid and tid not in surviving_tool_use_ids:
                            orphans.append((m.get("uuid", "?"), tid))

        self.assertEqual(
            orphans, [],
            msg=(
                "Floor re-added a tool_result whose paired tool_use is missing "
                f"in the survivors. Orphans (uuid, tool_use_id): {orphans}. "
                "The saved JSONL would 400 on resume — the exact class of bug "
                "this PR is supposed to prevent."
            ),
        )


if __name__ == "__main__":
    unittest.main()
