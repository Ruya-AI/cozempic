"""Regression tests for @ynaamane's independent peer review of PR #138.

Each test pins one of his findings (the daemon-asymmetry framing: over-defer is
recoverable; wrongly killing live Claude work is catastrophic).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cozempic.overflow import CircuitBreaker, OverflowRecovery


class TestOverflowGateFailClosed(unittest.TestCase):
    """#1 (HIGH) + #3: the reactive in-flight safety gate must FAIL CLOSED (defer, not
    kill) if it throws, and a benign defer must NOT consume a circuit-breaker slot."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.session_path = Path(self.tmpdir) / "session.jsonl"
        self.session_path.write_text(json.dumps({"type": "user", "message": "hi"}) + "\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _recovery(self, breaker):
        return OverflowRecovery(self.session_path, "tn-gate", self.tmpdir, breaker,
                                danger_threshold_mb=100.0, claude_pid=7777)

    def test_gate_error_defers_does_not_kill(self):
        breaker = CircuitBreaker(session_id="tn-gate", max_recoveries=3)
        breaker.reset()
        try:
            rec = self._recovery(breaker)
            with (
                patch.object(rec, "detect_overflow", return_value=True),
                patch("cozempic.guard.guard_prune_cycle", return_value={"saved_mb": 1.0,
                      "original_tokens": 1000, "final_tokens": 500}),
                patch("cozempic.guard._terminate_and_resume") as mock_kill,
                patch("cozempic.guard.checkpoint_team"),
                patch("cozempic.session.find_claude_pid", return_value=None),
                # make the safety gate itself throw
                patch("cozempic.guard.safe_to_reload", side_effect=RuntimeError("boom")),
            ):
                rec.recover()
            # FAIL-CLOSED: on a gate error we must DEFER — never kill a session that
            # may hold in-flight subagents.
            mock_kill.assert_not_called()
            # And the benign defer must NOT have consumed a breaker slot (#3).
            self.assertEqual(breaker.recovery_count(), 0)
        finally:
            breaker.reset()

    def test_unsafe_point_defers_without_burning_breaker(self):
        breaker = CircuitBreaker(session_id="tn-gate", max_recoveries=3)
        breaker.reset()
        try:
            rec = self._recovery(breaker)
            with (
                patch.object(rec, "detect_overflow", return_value=True),
                patch("cozempic.guard.guard_prune_cycle", return_value={"saved_mb": 1.0,
                      "original_tokens": 1000, "final_tokens": 500}),
                patch("cozempic.guard._terminate_and_resume") as mock_kill,
                patch("cozempic.guard.checkpoint_team"),
                patch("cozempic.session.find_claude_pid", return_value=None),
                patch("cozempic.guard.safe_to_reload", return_value=(False, "agents active")),
            ):
                rec.recover()
            mock_kill.assert_not_called()
            self.assertEqual(breaker.recovery_count(), 0)
        finally:
            breaker.reset()


class TestNonStrTextNoCrash(unittest.TestCase):
    """#2: a non-str `text` must not crash extract_team_state / _extract_block_text."""

    def test_extract_block_text_non_str(self):
        from cozempic.team import _extract_block_text
        blk = {"type": "tool_result", "tool_use_id": "x",
               "content": [{"type": "text", "text": 99999}]}
        self.assertEqual(_extract_block_text(blk), "")  # no TypeError

    def test_extract_team_state_non_str_text(self):
        from cozempic.session import load_messages
        from cozempic.team import extract_team_state
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.jsonl"
            p.write_text(json.dumps({"type": "assistant", "uuid": "a1", "message": {"role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Task", "input": {"prompt": "go"}}]}}) + "\n"
                + json.dumps({"type": "user", "uuid": "u1", "parentUuid": "a1", "message": {"role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": 12345}]}]}}) + "\n")
            extract_team_state(load_messages(p))  # must not raise


class TestConfigErrorNotSwallowed(unittest.TestCase):
    """#6: a user ConfigError must propagate, not be mislabeled as a malformed message."""

    def test_config_error_propagates(self):
        import cozempic.strategies.aggressive  # noqa: register strategies
        from cozempic.registry import STRATEGIES
        from cozempic.session import load_messages
        from cozempic.executor import run_prescription
        from cozempic._validation import ConfigError
        mega = [n for n in STRATEGIES if "mega" in n]
        self.assertTrue(mega)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.jsonl"
            p.write_text(json.dumps({"type": "assistant", "uuid": "a1",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "x" * 60000}]}}) + "\n")
            with self.assertRaises(ConfigError):
                run_prescription(load_messages(p), [mega[0]], {"mega_block_max_bytes": "not-an-int"})


class TestWatchdogErrorStormDiscriminator(unittest.TestCase):
    """#7 (+ R14): the error/escalation branch is gated on ZERO prune cycles. A guard
    that is PRODUCTIVELY pruning is never flagged — even one whose tail carries stale
    escalations from dead generations (the #7 false-positive). But a deterministic-
    error RESPAWN STORM (many restarts, 0 prunes) is still caught — generation-scoping
    (the first #7 attempt) let it escape because each generation exits below the
    thresholds (R14)."""

    def test_healthy_pruning_guard_with_stale_escalations_not_flagged(self):
        from cozempic.watchdog import scan_log_text
        text = ("Guard daemon started\n"
                + "  Guard cycle-error escalation: 5 consecutive cycle errors\n" * 3  # dead gen
                + "Guard daemon started\n"
                + "  Pruned: 12,345 tokens freed (45.0%)\n" * 5)  # healthy: real prune lines
        self.assertFalse(scan_log_text(text).looping)

    def test_multi_generation_error_storm_flagged(self):
        from cozempic.watchdog import scan_log_text
        # 26 generations, each 5 errors + 1 escalation then exit, ZERO prunes — the
        # exact storm generation-scoping let escape (each gen < the thresholds).
        gen = ("--- Guard daemon started ---\n"
               + "  Guard: skipping a cycle after an unexpected error (1/5): E\n" * 5
               + "  Guard cycle-error escalation: 5 consecutive cycle errors\n")
        self.assertTrue(scan_log_text(gen * 26).looping)

    def test_single_inert_generation_flagged(self):
        from cozempic.watchdog import scan_log_text
        text = "Guard daemon started\n" + "  Guard: skipping a cycle after an unexpected error (1/5): E\n" * 22
        self.assertTrue(scan_log_text(text).looping)

    def test_R15_storm_with_stray_productive_prune_still_flagged(self):
        # R15 FN: the total_prune_cycles==0 gate let an error storm escape if ANY stray
        # productive prune line survived in the tail. The errors-outnumber-productive-
        # prunes rule flags it (130 errors >> 1 prune).
        from cozempic.watchdog import scan_log_text
        gen = ("--- Guard daemon started ---\n"
               + "  Guard: skipping a cycle after an unexpected error (1/5): E\n" * 5
               + "  Guard cycle-error escalation: 5 consecutive cycle errors\n")
        stray = "  Pruned: 210,000 tokens freed (38.0%)\n"
        self.assertTrue(scan_log_text(stray + gen * 26).looping)

    def test_R15_healthy_idle_multi_restart_not_flagged(self):
        # R15 FP: a healthy idle guard (short sessions, 0 prunes, 0 errors) with several
        # restarts must NOT be flagged (the daemon_starts-only trigger false-flagged it).
        from cozempic.watchdog import scan_log_text
        text = "--- Guard daemon started ---\nCWD: /x\n  Read-only — live session not rewritten\n" * 6
        self.assertFalse(scan_log_text(text).looping)

    def test_R15_healthy_busy_with_transient_errors_not_flagged(self):
        # A long healthy daemon: many productive prunes, a few transient errors that do
        # NOT outnumber the prunes -> not flagged.
        from cozempic.watchdog import scan_log_text
        text = ("--- Guard daemon started ---\n"
                + "  Pruned: 12,345 tokens freed (48.0%)\n" * 50
                + "  Guard: skipping a cycle after an unexpected error (1/5): E\n" * 20)
        self.assertFalse(scan_log_text(text).looping)


class TestAtomicCheckpoint(unittest.TestCase):
    """#5: write_team_checkpoint writes atomically (no partial file)."""

    def test_checkpoint_written(self):
        from cozempic.team import TeamState, write_team_checkpoint
        with tempfile.TemporaryDirectory() as d:
            path = write_team_checkpoint(TeamState(team_name="t"), project_dir=Path(d))
            self.assertTrue(path.exists() and path.read_text())


class TestRedosNonCapturingGroupNotOverRejected(unittest.TestCase):
    """LOW: rule-2 must not over-reject (?:abc)+ / (?i:abc)+ / (?P<n>abc)+."""

    def test_non_capturing_groups_allowed(self):
        from cozempic.helpers import _pattern_is_redos_risky as risky
        for p in [r"(?:abc)+", r"(?i:abc)+", r"(?P<n>abc)+"]:
            self.assertFalse(risky(p), f"non-capturing/flag/named group over-rejected: {p}")
        for p in [r"(?:a+)+", r"(?:a|b)+"]:  # genuinely dangerous still flagged
            self.assertTrue(risky(p), f"dangerous group not flagged: {p}")


if __name__ == "__main__":
    unittest.main()
