"""Longitudinal loop test — drive the REAL daemon loop over many cycles.

Every other guard test checks a DIFF: one prune, one breaker decision, one helper.
But the f641174c reload-loop was EMERGENT — it only existed as a behaviour of the
daemon running over time, which no diff-scoped test could see. That is exactly the
process gap this file closes.

It runs the REAL ``start_guard`` loop (real ``_account_hard_prune`` closure, real
``HARD_LOOP_EXIT_THRESHOLD`` / back-off constants) against a real over-threshold
session whose every prune is futile, and asserts the system as a whole:
  (1) K-exits within the bounded number of cycles (does NOT loop forever), and
  (2) escalates its back-off before exiting (does NOT spin at full speed).

Only the I/O leaves (token estimate, prune compute, sleep, network) are stubbed —
the loop body and breaker are the real shipped code.
"""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import cozempic.guard as G
from cozempic.guard import HARD_LOOP_EXIT_THRESHOLD


class _EmptyState:
    """Stand-in for a checkpoint_team() result with no agents (agentless path)."""
    subagents: list = []
    tasks: list = []
    message_count = 0

    def is_empty(self):
        return True


class TestLongitudinalUnprunableLoop(unittest.TestCase):
    def setUp(self):
        self._td = TemporaryDirectory()
        self.session = Path(self._td.name) / "test1234.jsonl"
        # A real, over-threshold-but-unprunable session: dense lines we "can't" prune.
        self.session.write_text(
            "".join('{"type":"user","message":{"role":"user","content":"x"}}\n'
                    for _ in range(50)),
            encoding="utf-8",
        )
        self.sess = {"path": self.session, "session_id": "test1234", "project": "p",
                     "size": self.session.stat().st_size, "mtime": 0, "lines": 50}
        self.sleeps: list = []
        self.prune_calls = 0

    def tearDown(self):
        self._td.cleanup()

    def _futile_prune(self, *a, **k):
        # Every prune frees nothing — the unprunable-session case. No "reloading"
        # key, so the loop falls through to _account_hard_prune (the breaker).
        self.prune_calls += 1
        return {"saved_mb": 0.0, "original_tokens": 600_000, "final_tokens": 600_000,
                "live_write_skipped": False, "would_free_mb": 0.0,
                "original_bytes": self.session.stat().st_size}

    def _run_guard(self, token_estimator=None):
        patches = {
            "find_current_session": lambda *a, **k: self.sess,
            "load_messages": lambda *a, **k: [(0, {"type": "user"}, 10)],
            "checkpoint_team": lambda *a, **k: _EmptyState(),
            "quick_token_estimate": token_estimator or (lambda *a, **k: 600_000),   # >= 55% of 1M, < 80%
            "guard_prune_cycle": self._futile_prune,
            "cleanup_old_backups": lambda *a, **k: None,
            "ping_install_if_new": lambda *a, **k: None,
            "maybe_auto_update": lambda *a, **k: None,
            "_cleanup_stale_watchers": lambda *a, **k: None,
            "_detect_interactive": lambda *a, **k: False,      # headless → no defer
            "find_claude_pid": lambda *a, **k: None,
            "_safe_unlink_session_pidfile": lambda *a, **k: None,
        }
        cms = [mock.patch.object(G, name, fn) for name, fn in patches.items()]
        cms.append(mock.patch.object(G.time, "sleep", lambda s: self.sleeps.append(s)))
        cms.append(mock.patch("cozempic.tokens.detect_context_window", lambda *a, **k: 1_000_000))
        cms.append(mock.patch("cozempic.tokens.default_token_thresholds_4tier",
                              lambda cw: (250_000, 550_000, 800_000)))
        cms.append(mock.patch("cozempic.session.record_session", lambda *a, **k: None))
        for c in cms:
            c.start()
        try:
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    G.start_guard(cwd=self._td.name, threshold_mb=50.0,
                                  interval=2, reactive=False, auto_reload=True)
            return cm.exception
        finally:
            for c in reversed(cms):
                c.stop()

    def test_unprunable_session_kexits_and_does_not_loop_forever(self):
        exc = self._run_guard()
        self.assertEqual(exc.code, 0, "K-exit is a clean (0) exit")
        # (1) It stopped — bounded, not 202x. The breaker exits at exactly K=10
        # futile HARD cycles when no agents are active.
        self.assertEqual(
            self.prune_calls, HARD_LOOP_EXIT_THRESHOLD,
            f"must K-exit after exactly {HARD_LOOP_EXIT_THRESHOLD} futile cycles, "
            f"not loop forever (the f641174c bug ran 202x)")

    def test_backoff_escalates_before_exit(self):
        self._run_guard()
        # (2) Among the sleeps, at least one EXCEEDS the base interval (2s) — proof
        # the exponential back-off engaged instead of spinning at full cadence.
        self.assertTrue(
            any(s > 2 for s in self.sleeps),
            f"back-off must escalate beyond the base interval; saw sleeps={self.sleeps}")
        # And the back-off is monotonic non-decreasing across the futile run.
        backoff_sleeps = [s for s in self.sleeps if s > 2]
        self.assertEqual(backoff_sleeps, sorted(backoff_sleeps),
                         "back-off must grow, not oscillate")

    def test_cycle_error_does_not_kill_daemon(self):
        # A per-cycle exception (e.g. a malformed-usage TypeError that escaped the
        # token estimator before the _as_int fix) must NOT kill the daemon — the
        # loop-body guard logs and continues. Raise on the first 3 cycles, then
        # behave normally so the run still reaches the K-exit (proving survival).
        calls = {"n": 0}
        def flaky_estimator(*a, **k):
            calls["n"] += 1
            if calls["n"] <= 3:
                raise TypeError("unsupported operand type(s) for +: 'NoneType' and 'int'")
            return 600_000
        exc = self._run_guard(token_estimator=flaky_estimator)
        # Survived the 3 raising cycles and still reached the clean K-exit.
        self.assertEqual(exc.code, 0, "daemon must survive bad cycles and K-exit cleanly")
        self.assertGreater(calls["n"], 3, "must have continued past the raising cycles")
        self.assertEqual(self.prune_calls, HARD_LOOP_EXIT_THRESHOLD,
                         "post-error cycles still reach the futile-loop K-exit")

    def test_permanent_cycle_error_escalates_and_exits_not_inert(self):
        # A DETERMINISTIC per-cycle error must NOT spin forever as an inert-but-alive
        # daemon (watchdog-invisible) — it must escalate after GUARD_CYCLE_ERROR_EXIT
        # and exit(1) so SessionStart respawns (C2). Estimator raises EVERY cycle.
        from cozempic.guard import GUARD_CYCLE_ERROR_EXIT
        calls = {"n": 0}
        def always_raises(*a, **k):
            calls["n"] += 1
            raise TypeError("deterministic per-cycle failure")
        exc = self._run_guard(token_estimator=always_raises)  # harness asserts SystemExit
        self.assertEqual(exc.code, 1, "must exit(1) for respawn, not spin inert forever")
        self.assertEqual(calls["n"], GUARD_CYCLE_ERROR_EXIT,
                         f"must escalate after exactly {GUARD_CYCLE_ERROR_EXIT} consecutive errors")

    def test_unicode_decode_error_is_benign_skip_not_respawn_storm(self):
        # C1/C2 crossfix: a non-UTF-8 session raises UnicodeDecodeError every cycle.
        # It must NOT count toward the escalation (which would respawn-storm); the
        # guard skips that session and keeps running. Raise it 8x (> the exit
        # threshold of 5) then behave; the run must reach the normal K-exit (code 0),
        # proving the decode error never escalated to the respawn exit(1).
        from cozempic.guard import GUARD_CYCLE_ERROR_EXIT
        calls = {"n": 0}
        def decode_then_ok(*a, **k):
            calls["n"] += 1
            if calls["n"] <= GUARD_CYCLE_ERROR_EXIT + 3:
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
            return 600_000
        exc = self._run_guard(token_estimator=decode_then_ok)
        self.assertEqual(exc.code, 0, "UnicodeDecodeError must be a benign skip, not escalate to exit(1)")
        self.assertGreater(calls["n"], GUARD_CYCLE_ERROR_EXIT + 3,
                           "must keep running past the error threshold (decode errors don't escalate)")


if __name__ == "__main__":
    unittest.main()
