"""Regression guards for build_team_recovery_receipt terminal-status predicate.

P0-C: team.py:271 used `not in {"done", "completed"}` to classify active teammates,
missing all _STATUS_TERMINAL members except those two. A "failed" or "cancelled"
teammate counted as active → verdict='partial' when the session was actually quiescent.

REGRESSION GUARD tests are proven RED at base (HEAD e65636e before P0-C fix):
  with only a terminal teammate and no other active work, verdict='partial'
  (bug: has_active_work=True adds "per_teammate_event_cursors_not_recorded" gap).
  After fix: terminal teammate → has_active_work=False → verdict='complete'.

Also includes the _TEAMMATE_QUIESCENT == guard._STATUS_TERMINAL equality test per
lead Q-A: the local frozen set in team.py must mirror guard._STATUS_TERMINAL members.
"""
import unittest


class TestBuildTeamRecoveryReceiptTerminalStatuses(unittest.TestCase):
    """build_team_recovery_receipt must treat all _STATUS_TERMINAL statuses as
    non-active, not just 'done'/'completed'.

    Test state: single terminal teammate + one completed task + full cursors.
    No running subagents, no active tasks → has_active_work must be False → no
    "per_teammate_event_cursors_not_recorded" gap → verdict='complete'.

    At base (L271 uses {"done","completed"}), 'failed'/'cancelled'/'aborted' count
    as active teammates → has_active_work=True → verdict='partial'.
    """

    def _make_quiescent_state(self, tm_status: str):
        from cozempic.team import TeamState, TeammateInfo, TaskInfo
        return TeamState(
            team_name="myteam",
            lead_agent_id="lead@myteam",
            lead_session_id="sess-1",
            config_source="jsonl",
            last_coordination_index=10,  # cursor present
            tasks=[TaskInfo(task_id="1", subject="clean up", status="completed")],
            teammates=[TeammateInfo(agent_id="w@myteam", name="w", status=tm_status)],
            subagents=[],
        )

    def test_failed_teammate_not_active(self):
        """REGRESSION GUARD: a 'failed' teammate with all other work quiescent must
        yield verdict='complete', not 'partial'.

        RED at base (e65636e): 'failed' not in {"done","completed"} → has_active_work=True
        → gaps=['per_teammate_event_cursors_not_recorded'] → verdict='partial'.
        GREEN after fix: 'failed' in _TEAMMATE_QUIESCENT → has_active_work=False
        → gaps=[] → verdict='complete'.
        """
        from cozempic.team import build_team_recovery_receipt
        receipt = build_team_recovery_receipt(self._make_quiescent_state("failed"))
        self.assertEqual(
            "complete",
            receipt["recovery_verdict"],
            f"A 'failed' teammate with no other active work must yield verdict='complete'; "
            f"got {receipt['recovery_verdict']!r}. Audit gaps: {receipt['audit_gaps']}. "
            f"team.py:271 predicate uses {{\"done\",\"completed\"}} which misses 'failed'.",
        )

    def test_cancelled_teammate_not_active(self):
        """REGRESSION GUARD: a 'cancelled' teammate with all other work quiescent must
        yield verdict='complete'.

        RED at base: 'cancelled' not in {"done","completed"} → partial.
        GREEN after fix: 'cancelled' in _TEAMMATE_QUIESCENT → complete.
        """
        from cozempic.team import build_team_recovery_receipt
        receipt = build_team_recovery_receipt(self._make_quiescent_state("cancelled"))
        self.assertEqual(
            "complete",
            receipt["recovery_verdict"],
            f"A 'cancelled' teammate with no other active work must yield verdict='complete'; "
            f"got {receipt['recovery_verdict']!r}. Audit gaps: {receipt['audit_gaps']}.",
        )

    def test_aborted_teammate_not_active(self):
        """REGRESSION GUARD: 'aborted' is in _STATUS_TERMINAL but not {"done","completed"}.

        RED at base: 'aborted' not in {"done","completed"} → partial.
        GREEN after fix: 'aborted' in _TEAMMATE_QUIESCENT → complete.
        """
        from cozempic.team import build_team_recovery_receipt
        receipt = build_team_recovery_receipt(self._make_quiescent_state("aborted"))
        self.assertEqual(
            "complete",
            receipt["recovery_verdict"],
            f"'aborted' teammate must yield verdict='complete'; got {receipt!r}",
        )

    def test_done_teammate_was_already_not_active(self):
        """Positive control (GREEN at base): 'done' was already in the old set.

        Verifies the fix doesn't regress the existing behavior for 'done'.
        """
        from cozempic.team import build_team_recovery_receipt
        receipt = build_team_recovery_receipt(self._make_quiescent_state("done"))
        self.assertEqual(
            "complete",
            receipt["recovery_verdict"],
            f"'done' teammate (already covered at base) must yield verdict='complete'; "
            f"got {receipt['recovery_verdict']!r}. Audit gaps: {receipt['audit_gaps']}.",
        )

    def test_completed_teammate_not_active(self):
        """Positive control (GREEN at base): 'completed' was already in the old set."""
        from cozempic.team import build_team_recovery_receipt
        receipt = build_team_recovery_receipt(self._make_quiescent_state("completed"))
        self.assertEqual(
            "complete",
            receipt["recovery_verdict"],
            f"'completed' teammate must yield verdict='complete'; "
            f"got {receipt['recovery_verdict']!r}.",
        )


class TestTeammateQuiescentSetParity(unittest.TestCase):
    """_TEAMMATE_QUIESCENT in team.py must mirror guard._STATUS_TERMINAL members.

    Lead Q-A answer: use a local frozenset named _TEAMMATE_QUIESCENT (duplicated
    until a future PR consolidates into _constants.py). Guard parity with a test
    that imports both modules — no runtime cycle (test can import both).
    """

    def test_teammate_quiescent_equals_status_terminal(self):
        """_TEAMMATE_QUIESCENT (team.py) must equal guard._STATUS_TERMINAL in membership.

        If this fails after adding _TEAMMATE_QUIESCENT, the two sets have diverged
        and the P0-C fix is incomplete.  ERROR at base (_TEAMMATE_QUIESCENT doesn't
        exist yet → ImportError). GREEN after fix.
        """
        from cozempic.team import _TEAMMATE_QUIESCENT
        from cozempic.guard import _STATUS_TERMINAL
        self.assertEqual(
            _TEAMMATE_QUIESCENT,
            _STATUS_TERMINAL,
            f"team._TEAMMATE_QUIESCENT must equal guard._STATUS_TERMINAL.\n"
            f"  _TEAMMATE_QUIESCENT: {sorted(_TEAMMATE_QUIESCENT)}\n"
            f"  _STATUS_TERMINAL:    {sorted(_STATUS_TERMINAL)}\n"
            f"  In _QUIESCENT not _TERMINAL: {_TEAMMATE_QUIESCENT - _STATUS_TERMINAL}\n"
            f"  In _TERMINAL not _QUIESCENT: {_STATUS_TERMINAL - _TEAMMATE_QUIESCENT}",
        )


if __name__ == "__main__":
    unittest.main()
