"""Tests for Track 2: compact-summary-collapse, attribution-snapshot-strip, microcompact-aware trim, registry."""

from __future__ import annotations

import unittest

from cozempic.helpers import msg_bytes
from cozempic.registry import STRATEGIES, PRESCRIPTIONS
from cozempic.executor import run_prescription

import cozempic.strategies  # noqa: F401


def make_message(line_idx: int, msg: dict) -> tuple[int, dict, int]:
    return (line_idx, msg, msg_bytes(msg))


def make_compact_boundary(line_idx: int, has_preserved: bool = False) -> tuple[int, dict, int]:
    msg = {"type": "system", "subtype": "compact_boundary"}
    if has_preserved:
        msg["hasPreservedSegment"] = True
    return make_message(line_idx, msg)


def make_compact_summary(line_idx: int) -> tuple[int, dict, int]:
    msg = {
        "type": "user",
        "isCompactSummary": True,
        "message": {"role": "user", "content": "Summary of prior conversation..."},
    }
    return make_message(line_idx, msg)


def make_user(line_idx: int, text: str = "hi") -> tuple[int, dict, int]:
    msg = {
        "type": "user",
        "message": {"role": "user", "content": text},
        "uuid": f"uuid-{line_idx}",
    }
    return make_message(line_idx, msg)


def make_assistant(line_idx: int, text: str = "ok") -> tuple[int, dict, int]:
    msg = {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        "uuid": f"uuid-{line_idx}",
    }
    return make_message(line_idx, msg)


def make_attribution_snapshot(line_idx: int) -> tuple[int, dict, int]:
    msg = {
        "type": "attribution-snapshot",
        "data": {"commits": ["abc123"], "pr": ""},
    }
    return make_message(line_idx, msg)


def make_metadata_singleton(line_idx: int, mtype: str = "last-prompt") -> tuple[int, dict, int]:
    msg = {"type": mtype, "data": "some-value"}
    return make_message(line_idx, msg)


# ---------------------------------------------------------------------------
# T2.1: compact-summary-collapse
# ---------------------------------------------------------------------------

class TestCompactSummaryCollapse(unittest.TestCase):

    def test_removes_pre_boundary_messages(self):
        """Everything before the boundary should be removed."""
        messages = [
            make_user(0, "old message 1"),
            make_assistant(1, "old response 1"),
            make_user(2, "old message 2"),
            make_compact_boundary(3),
            make_compact_summary(4),
            make_user(5, "new message"),
            make_assistant(6, "new response"),
        ]
        sr = STRATEGIES["compact-summary-collapse"].func(messages, {})
        removed_lines = {a.line_index for a in sr.actions}
        # Lines 0, 1, 2 should be removed (pre-boundary)
        self.assertEqual(removed_lines, {0, 1, 2})
        self.assertEqual(sr.messages_removed, 3)

    def test_no_op_without_boundary(self):
        """No boundary means nothing to collapse."""
        messages = [make_user(0), make_assistant(1)]
        sr = STRATEGIES["compact-summary-collapse"].func(messages, {})
        self.assertEqual(len(sr.actions), 0)
        self.assertIn("No compact_boundary", sr.summary)

    def test_skips_has_preserved_segment(self):
        """hasPreservedSegment=True means pre-boundary content is still needed."""
        messages = [
            make_user(0, "old"),
            make_compact_boundary(1, has_preserved=True),
            make_compact_summary(2),
        ]
        sr = STRATEGIES["compact-summary-collapse"].func(messages, {})
        self.assertEqual(len(sr.actions), 0)
        self.assertIn("hasPreservedSegment", sr.summary)

    def test_skips_protected_messages(self):
        """Protected messages before boundary should NOT be removed."""
        messages = [
            make_user(0, "old"),
            make_message(1, {"type": "content-replacement", "data": "frozen"}),
            make_assistant(2, "old resp"),
            make_compact_boundary(3),
            make_compact_summary(4),
        ]
        sr = STRATEGIES["compact-summary-collapse"].func(messages, {})
        removed_lines = {a.line_index for a in sr.actions}
        # content-replacement (line 1) is protected, should NOT be in removals
        self.assertNotIn(1, removed_lines)
        self.assertIn(0, removed_lines)
        self.assertIn(2, removed_lines)

    def test_keeps_metadata_singleton_only_before_boundary(self):
        """Metadata singletons that only appear pre-boundary should be kept (re-anchor)."""
        messages = [
            make_user(0, "old"),
            make_metadata_singleton(1, "last-prompt"),
            make_compact_boundary(2),
            make_compact_summary(3),
        ]
        sr = STRATEGIES["compact-summary-collapse"].func(messages, {})
        removed_lines = {a.line_index for a in sr.actions}
        # last-prompt (line 1) only exists pre-boundary — should be kept
        self.assertNotIn(1, removed_lines)
        self.assertIn(0, removed_lines)

    def test_removes_metadata_singleton_if_also_post_boundary(self):
        """Metadata singleton that also exists post-boundary can be removed pre-boundary."""
        messages = [
            make_user(0, "old"),
            make_metadata_singleton(1, "last-prompt"),
            make_compact_boundary(2),
            make_compact_summary(3),
            make_metadata_singleton(4, "last-prompt"),
        ]
        sr = STRATEGIES["compact-summary-collapse"].func(messages, {})
        removed_lines = {a.line_index for a in sr.actions}
        # last-prompt exists post-boundary too, so pre-boundary copy can go
        self.assertIn(1, removed_lines)

    def test_uses_last_boundary(self):
        """With multiple boundaries, use the last one."""
        messages = [
            make_user(0, "very old"),
            make_compact_boundary(1),
            make_compact_summary(2),
            make_user(3, "mid-old"),
            make_compact_boundary(4),
            make_compact_summary(5),
            make_user(6, "new"),
        ]
        sr = STRATEGIES["compact-summary-collapse"].func(messages, {})
        removed_lines = {a.line_index for a in sr.actions}
        # Everything before pos 4 (last boundary) should be removable
        # Lines 0, 3 are normal messages — removed
        # Lines 1 is compact_boundary (protected), line 2 is compact_summary (protected)
        self.assertIn(0, removed_lines)
        self.assertIn(3, removed_lines)
        self.assertNotIn(1, removed_lines)  # compact_boundary is protected
        self.assertNotIn(2, removed_lines)  # isCompactSummary is protected

    def test_large_savings(self):
        """Verify significant byte savings."""
        pre_messages = [make_user(i, "x" * 1000) for i in range(100)]
        boundary = make_compact_boundary(100)
        summary = make_compact_summary(101)
        post = [make_user(102, "new")]
        messages = pre_messages + [boundary, summary] + post
        sr = STRATEGIES["compact-summary-collapse"].func(messages, {})
        self.assertEqual(sr.messages_removed, 100)
        self.assertGreater(sr.pruned_bytes, 50000)


# ---------------------------------------------------------------------------
# T2.2: attribution-snapshot-strip
# ---------------------------------------------------------------------------

class TestAttributionSnapshotStrip(unittest.TestCase):

    def test_removes_attribution_snapshots(self):
        messages = [
            make_user(0),
            make_attribution_snapshot(1),
            make_assistant(2),
            make_attribution_snapshot(3),
        ]
        sr = STRATEGIES["attribution-snapshot-strip"].func(messages, {})
        removed_lines = {a.line_index for a in sr.actions}
        self.assertEqual(removed_lines, {1, 3})

    def test_no_op_without_snapshots(self):
        messages = [make_user(0), make_assistant(1)]
        sr = STRATEGIES["attribution-snapshot-strip"].func(messages, {})
        self.assertEqual(len(sr.actions), 0)

    def test_does_not_remove_non_attribution(self):
        messages = [
            make_user(0),
            make_message(1, {"type": "file-history-snapshot", "messageId": "m1", "snapshot": {}}),
        ]
        sr = STRATEGIES["attribution-snapshot-strip"].func(messages, {})
        self.assertEqual(len(sr.actions), 0)


# ---------------------------------------------------------------------------
# T2.3: microcompact-aware tool-output-trim
# ---------------------------------------------------------------------------

class TestMicrocompactAwareTrim(unittest.TestCase):

    def test_skips_compacted_tool_results(self):
        """Tool results already summarized by microcompact should not be trimmed."""
        messages = [
            # microcompact boundary says t1 is already compacted
            make_message(0, {
                "type": "system",
                "subtype": "microcompact_boundary",
                "compactedToolIds": ["t1"],
            }),
            # Large tool result for t1 — should NOT be trimmed
            make_message(1, {
                "type": "user",
                "message": {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "line\n" * 200},
                ]},
            }),
        ]
        sr = STRATEGIES["tool-output-trim"].func(messages, {"tool_output_max_lines": 100})
        self.assertEqual(len(sr.actions), 0)

    def test_trims_non_compacted_tool_results(self):
        """Tool results NOT in compactedToolIds should still be trimmed."""
        messages = [
            make_message(0, {
                "type": "system",
                "subtype": "microcompact_boundary",
                "compactedToolIds": ["t1"],
            }),
            # t2 is NOT compacted — should be trimmed
            make_message(1, {
                "type": "user",
                "message": {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t2", "content": "line\n" * 200},
                ]},
            }),
        ]
        sr = STRATEGIES["tool-output-trim"].func(messages, {"tool_output_max_lines": 100})
        self.assertEqual(sr.messages_replaced, 1)

    def test_no_microcompact_boundary_trims_all(self):
        """Without microcompact boundaries, all large results are trimmed as before."""
        messages = [
            make_message(0, {
                "type": "user",
                "message": {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "line\n" * 200},
                ]},
            }),
        ]
        sr = STRATEGIES["tool-output-trim"].func(messages, {"tool_output_max_lines": 100})
        self.assertEqual(sr.messages_replaced, 1)


# ---------------------------------------------------------------------------
# T2.4: Registry prescriptions
# ---------------------------------------------------------------------------

class TestRegistryPrescriptions(unittest.TestCase):

    def test_compact_summary_collapse_is_first(self):
        for name, strategies in PRESCRIPTIONS.items():
            self.assertEqual(strategies[0], "compact-summary-collapse",
                             f"compact-summary-collapse must be first in {name}")

    def test_attribution_snapshot_strip_in_all(self):
        for name, strategies in PRESCRIPTIONS.items():
            self.assertIn("attribution-snapshot-strip", strategies,
                          f"attribution-snapshot-strip missing from {name}")

    def test_all_strategies_registered(self):
        """Every strategy listed in prescriptions must be registered."""
        for name, strategies in PRESCRIPTIONS.items():
            for s in strategies:
                self.assertIn(s, STRATEGIES, f"Strategy '{s}' in {name} not registered")

    def test_strategy_count(self):
        """Verify expected strategy counts per tier."""
        self.assertEqual(len(PRESCRIPTIONS["gentle"]), 5)
        self.assertEqual(len(PRESCRIPTIONS["standard"]), 11)
        self.assertEqual(len(PRESCRIPTIONS["aggressive"]), 18)


# ---------------------------------------------------------------------------
# Integration: run_prescription with new strategies
# ---------------------------------------------------------------------------

class TestPrescriptionIntegration(unittest.TestCase):

    def test_gentle_with_boundary(self):
        """Full gentle prescription on a session with compact boundary.

        Post-2026-05-28 fix (PR — session pruner resumable state): the floor
        preserves the first message, last K conversational turns, and at
        least 50% of users + assistants. On this tiny 2-user / 2-assistant
        fixture the floor re-adds the pre-boundary user/assistant; the
        attribution-snapshot still drops because it isn't conversational.
        """
        messages = [
            make_user(0, "old"),
            make_assistant(1, "old resp"),
            make_attribution_snapshot(2),
            make_compact_boundary(3),
            make_compact_summary(4),
            make_user(5, "new"),
            make_assistant(6, "new resp"),
        ]
        result_msgs, results = run_prescription(messages, PRESCRIPTIONS["gentle"], {})
        result_indices = {idx for idx, _, _ in result_msgs}
        # Attribution-snapshot (not floor-protected) is still dropped.
        self.assertNotIn(2, result_indices)
        # Post-boundary content remains.
        self.assertIn(5, result_indices)
        self.assertIn(6, result_indices)
        # Pre-boundary conversation is now floor-protected (root + last-K).
        self.assertIn(0, result_indices)
        self.assertIn(1, result_indices)

    def test_gentle_without_boundary(self):
        """Gentle prescription without boundary should still run other strategies."""
        messages = [
            make_user(0),
            make_attribution_snapshot(1),
            make_assistant(2),
        ]
        result_msgs, results = run_prescription(messages, PRESCRIPTIONS["gentle"], {})
        result_indices = {idx for idx, _, _ in result_msgs}
        # Attribution snapshot should be removed
        self.assertNotIn(1, result_indices)
        # User and assistant should remain
        self.assertIn(0, result_indices)
        self.assertIn(2, result_indices)


if __name__ == "__main__":
    unittest.main()
