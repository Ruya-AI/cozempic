"""Guard-loop watchdog: the OUTSIDE-the-daemon process safeguard.

The in-process circuit breaker (1.8.29 / 1.8.19) protects daemons running the
fixed code. The watchdog protects against the ones that DON'T self-arrest — an old
brew install still resident, or a future regression — by scanning the guard logs
the daemons already write for the futile-churn signature.

GROUND TRUTH: these tests run against a slice of the ACTUAL f641174c guard log
(``tests/fixtures/guard_logs/f641174c_reload_loop.log``). That captured log taught
us the real pathology was a RESPAWN STORM — 23 daemon starts, 216 futile prunes,
21 K-exits — i.e. each daemon DID self-arrest, yet the SessionStart hook kept
respawning a fresh one onto a permanently-unprunable session. So "saw an exit
line" is NOT proof of health (my first synthetic version of these tests wrongly
assumed it was — exactly the synthetic-vs-real trap). The true signal is
futile-churn DOMINANCE, exit or no exit.
"""

import signal
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from cozempic.watchdog import (
    scan_log_text, scan_guard_logs,
    FUTILE_PCT_FLOOR, LOOP_TRIP_DEFAULT, BACKOFF_CAP_S, STORM_TRIP,
)

FIXTURES = Path(__file__).parent / "fixtures" / "guard_logs"


def _futile_cycle(ts="12:00:00", n=1):
    return (f"  [{ts}] HARD THRESHOLD (55%): 776,558 tokens >= 550,000 (55%)\n"
            f"  Standard prune + reload (cycle #{n})...\n"
            f"  Pruned: 0 tokens freed (0.0%), 0.0MB saved\n")


def _good_cycle(n=1, pct=35.0):
    return (f"  [12:00:00] HARD THRESHOLD (55%): 600,000 tokens >= 550,000 (55%)\n"
            f"  Standard prune + reload (cycle #{n})...\n"
            f"  Pruned: 210.0K tokens freed ({pct}%), 1.5MB saved\n")


def _daemon_start(ts="2026-06-10T15:55:00"):
    return f"--- Guard daemon started at {ts} ---\nCWD: /x\n\n"


# A respawn storm reconstructed from the real shape: many short runs, each
# K-exiting at 10 futile cycles, re-spawned over and over.
def _respawn_storm(runs=23, per_run=10):
    out = []
    for r in range(runs):
        out.append(_daemon_start(ts=f"2026-06-10T1{r%6}:00:00"))
        out += [_futile_cycle(n=i) for i in range(1, per_run + 1)]
        out.append("  [15:43:58] Guard powerless against live-context dominance "
                    f"({per_run} consecutive 0-byte HARD prunes). Exiting.\n")
    return "".join(out)


class TestRealFixture(unittest.TestCase):
    """The captured real f641174c log slice is the anchor for everything."""

    def setUp(self):
        self.path = FIXTURES / "f641174c_reload_loop.log"
        self.assertTrue(self.path.exists(), "real-log corpus fixture must be committed")
        self.text = self.path.read_text(encoding="utf-8")

    def test_real_first_run_kexits_cleanly_not_flagged(self):
        # The committed slice is the FIRST daemon run: 10 futile cycles then a
        # clean "powerless… Exiting". That is the HEALTHY self-arrest — 10 < trip,
        # so it must NOT be flagged (even though it contains an exit line).
        rep = scan_log_text(self.text)
        self.assertGreaterEqual(rep.futile_cycles, 10)
        self.assertTrue(rep.has_exit, "fixture contains the real K-exit line")
        self.assertEqual(rep.daemon_starts, 1)
        self.assertFalse(rep.looping,
                         "a single run that K-exited at 10 is healthy, not a loop")


class TestRespawnStorm(unittest.TestCase):
    def test_storm_flagged_despite_kexits(self):
        # THE lesson: 23 runs that each K-exit is still a stuck loop.
        rep = scan_log_text(_respawn_storm())
        self.assertTrue(rep.looping, "a respawn storm must be flagged despite K-exits")
        self.assertTrue(rep.has_exit)
        self.assertGreaterEqual(rep.daemon_starts, STORM_TRIP)
        self.assertIn("respawn storm", rep.reason)

    def test_single_run_infinite_loop_flagged(self):
        # The other shape: one daemon, no exit, hundreds of futile cycles.
        text = _daemon_start() + "".join(_futile_cycle(n=i) for i in range(1, 203))
        rep = scan_log_text(text)
        self.assertTrue(rep.looping)
        self.assertFalse(rep.has_exit)
        self.assertEqual(rep.daemon_starts, 1)
        self.assertIn("reload-looping", rep.reason)


class TestHealthy(unittest.TestCase):
    def test_single_kexit_run_not_flagged(self):
        text = _daemon_start() + "".join(_futile_cycle(n=i) for i in range(1, 11))
        text += "  Guard powerless… Exiting.\n"
        self.assertFalse(scan_log_text(text).looping,
                         "10 futile cycles + K-exit (the agentless cap) is healthy")

    def test_real_prunes_not_flagged(self):
        rep = scan_log_text(_daemon_start() + "".join(_good_cycle(n=i) for i in range(50)))
        # The K-suffixed productive lines MUST be counted (the old regex silently
        # dropped them — assert the absolute count so a parse-miss can never again
        # masquerade as "no futile cycles").
        self.assertEqual(rep.total_prune_cycles, 50,
                         "productive K-format prunes must be parsed, not dropped")
        self.assertEqual(rep.futile_cycles, 0)
        self.assertFalse(rep.looping)

    def test_K_and_M_suffix_prunes_counted_non_futile(self):
        # Direct regression for the watchdog K/M-regex P1: both K- and M-format
        # productive prune lines parse and classify non-futile.
        text = _daemon_start()
        text += "  Pruned: 5.0K tokens freed (12.5%), 1.5MB saved\n"
        text += "  Pruned: 1.2M tokens freed (60.0%), 9.0MB saved\n"
        text += "  Pruned: 1,234 tokens freed (8.0%), 0.1MB saved\n"
        rep = scan_log_text(text)
        self.assertEqual(rep.total_prune_cycles, 3, "K/M/comma forms must all parse")
        self.assertEqual(rep.futile_cycles, 0)
        self.assertFalse(rep.looping)

    def test_busy_readonly_checkpoints_not_flagged(self):
        # Agents-active deferral emits read-only-checkpoint lines, NOT "Pruned: …"
        # lines — so a long busy session accrues ZERO futile prune cycles.
        def _ro(i):
            return (
                "  [12:00:00] HARD THRESHOLD (55%): 600,000 tokens >= 550,000 (55%)\n"
                f"  Agents active — read-only checkpoint, deferring prune+reload (cycle #{i})...\n"
                "  Read-only — live session not rewritten (#106).\n")
        text = _daemon_start() + "".join(_ro(i) for i in range(60))
        rep = scan_log_text(text)
        self.assertEqual(rep.total_prune_cycles, 0)
        self.assertFalse(rep.looping)

    def test_mostly_real_some_futile_not_flagged(self):
        # 40 real (K-format) prunes + 25 marginal futile → futile is 25/65 = 38%,
        # NOT dominant. This only exercises the dominance math if the real prunes
        # are actually counted — assert the full denominator so the K-format
        # parse-miss (the watchdog P1) can't make this pass vacuously.
        text = _daemon_start() + "".join(_good_cycle(n=i) for i in range(40))
        text += "".join(_futile_cycle(n=i) for i in range(25))
        rep = scan_log_text(text)
        self.assertEqual(rep.total_prune_cycles, 65,
                         "all 40 real + 25 futile prunes must be in the denominator")
        self.assertEqual(rep.futile_cycles, 25)
        self.assertFalse(rep.looping, "futile (38%) must DOMINATE (>=80%) to flag")

    def test_empty_log(self):
        self.assertFalse(scan_log_text("").looping)

    def test_just_under_trip_not_flagged(self):
        text = _daemon_start() + "".join(_futile_cycle(n=i) for i in range(LOOP_TRIP_DEFAULT - 1))
        self.assertFalse(scan_log_text(text).looping)


class TestScanGuardLogs(unittest.TestCase):
    def setUp(self):
        self._td = TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def _write(self, sid, text, pid=None):
        (self.dir / f"cozempic_guard_{sid}.log").write_text(text, encoding="utf-8")
        if pid is not None:
            (self.dir / f"cozempic_guard_{sid}.pid").write_text(str(pid), encoding="utf-8")

    def test_flags_live_storm(self):
        self._write("aaa", _respawn_storm(), pid=4242)
        with mock.patch("cozempic.watchdog._pid_alive", lambda pid: True):
            hits = scan_guard_logs(self.dir)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].pid, 4242)
        self.assertTrue(hits[0].pid_alive)

    def test_dead_pid_reported_not_alive(self):
        self._write("bbb", _respawn_storm(), pid=999999)
        with mock.patch("cozempic.watchdog._pid_alive", lambda pid: False):
            hits = scan_guard_logs(self.dir)
        self.assertEqual(len(hits), 1)
        self.assertFalse(hits[0].pid_alive)

    def test_healthy_not_in_hits(self):
        self._write("ccc", _daemon_start() + "".join(_good_cycle(n=i) for i in range(40)), pid=123)
        self.assertEqual(scan_guard_logs(self.dir), [])

    def test_real_fixture_single_run_not_flagged(self):
        self._write("ddd", (FIXTURES / "f641174c_reload_loop.log").read_text(), pid=1)
        self.assertEqual(scan_guard_logs(self.dir), [],
                         "the healthy single-run real fixture must not be a hit")

    def test_missing_dir(self):
        self.assertEqual(scan_guard_logs(self.dir / "nope"), [])

    def test_tail_read_on_huge_log(self):
        big = _daemon_start() + "".join(_futile_cycle(n=i) for i in range(3000))
        self._write("eee", big, pid=7)
        with mock.patch("cozempic.watchdog._pid_alive", lambda pid: True):
            hits = scan_guard_logs(self.dir, max_tail_bytes=64 * 1024)
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0].report.looping)


class TestCliCommand(unittest.TestCase):
    def setUp(self):
        self._td = TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def _run(self, fix=False, pid=4242):
        from types import SimpleNamespace
        from cozempic.cli import cmd_guard_watchdog
        (self.dir / "cozempic_guard_zzz.log").write_text(_respawn_storm(), encoding="utf-8")
        (self.dir / "cozempic_guard_zzz.pid").write_text(str(pid), encoding="utf-8")
        args = SimpleNamespace(fix=fix, log_dir=str(self.dir), loop_trip=20)
        return cmd_guard_watchdog(args)

    def test_report_only_exits_nonzero_on_live_loop(self):
        with mock.patch("cozempic.watchdog._pid_alive", lambda pid: True):
            with self.assertRaises(SystemExit) as cm:
                self._run(fix=False)
        self.assertEqual(cm.exception.code, 3)

    def test_fix_sends_sigterm(self):
        """--fix SIGTERMs a pid confirmed as a cozempic guard (guard_confirmed=True)."""
        killed = {}
        def fake_kill(pid, sig):
            killed["pid"], killed["sig"] = pid, sig
        with mock.patch("cozempic.watchdog._pid_alive", lambda pid: True), \
             mock.patch("cozempic.guard._is_cozempic_guard_process", return_value=True), \
             mock.patch("os.kill", fake_kill):
            self._run(fix=True)
        self.assertEqual(killed.get("pid"), 4242)
        self.assertEqual(killed.get("sig"), signal.SIGTERM)


class TestFixIdentityGate(unittest.TestCase):
    """L2 HIGH: --fix must verify guard identity before sending SIGTERM.

    Confused-deputy scenario: guard exits hard (SIGKILL / OOM), pidfile is
    never unlinked, OS recycles the PID to an unrelated same-user process,
    operator runs `cozempic guard-watchdog --fix`.  Without an identity gate,
    the innocent process is SIGTERMed.

    RED-at-base proof: before the fix, ``test_fix_refuses_to_kill_non_guard``
    fails because os.kill IS called on the recycled pid (SIGTERM to innocent).
    After the fix, the gate blocks the kill and the test passes.
    """

    def setUp(self):
        self._td = TemporaryDirectory()
        self.dir = Path(self._td.name)
        (self.dir / "cozempic_guard_zzz.log").write_text(_respawn_storm(), encoding="utf-8")
        (self.dir / "cozempic_guard_zzz.pid").write_text("7777", encoding="utf-8")

    def tearDown(self):
        self._td.cleanup()

    def _run_fix(self, is_guard: bool) -> dict:
        """Drive cmd_guard_watchdog(fix=True) with identity mock; return kill calls."""
        from types import SimpleNamespace
        from cozempic.cli import cmd_guard_watchdog
        killed: dict = {}

        def fake_kill(pid, sig):
            killed["pid"], killed["sig"] = pid, sig

        with mock.patch("cozempic.watchdog._pid_alive", return_value=True), \
             mock.patch("cozempic.guard._is_cozempic_guard_process", return_value=is_guard), \
             mock.patch("os.kill", fake_kill):
            args = SimpleNamespace(fix=True, log_dir=str(self.dir), loop_trip=20)
            try:
                cmd_guard_watchdog(args)
            except SystemExit:
                # sys.exit(3) when live loop not arrested (non-guard case) — expected
                pass
        return killed

    def test_fix_refuses_to_kill_non_guard(self):
        """RED-at-base: a recycled (non-guard) pid must NOT receive SIGTERM.

        Without guard_confirmed gate, os.kill IS called — this test FAILS at
        base and PASSES after the fix adds the identity check.
        """
        killed = self._run_fix(is_guard=False)
        self.assertNotIn("pid", killed,
                         "os.kill was called on a recycled non-guard pid — "
                         "confused-deputy bug: the identity gate is missing")

    def test_fix_kills_confirmed_guard(self):
        """A pid confirmed as a cozempic guard MUST receive SIGTERM under --fix."""
        killed = self._run_fix(is_guard=True)
        self.assertEqual(killed.get("pid"), 7777)
        self.assertEqual(killed.get("sig"), signal.SIGTERM)


def _error_cycle(n=1):
    """One guard error-skip line (no timestamp, matches _CYCLE_ERR_RE)."""
    return f"  Guard: skipping a cycle after an unexpected error ({n}/5): RuntimeError(boom)\n"


def _escalation():
    """One guard cycle-error escalation line (matches _CYCLE_ESCALATION_RE)."""
    return "  Guard cycle-error escalation: 5 consecutive cycle errors (0 deferred) — exiting for respawn (last: RuntimeError(boom)).\n"


class TestC7RateWindow(unittest.TestCase):
    """Rate-based storm detection: the rate-window path must fire before the flat-count.

    T-1 (AC-1): multi-generation error storm masked by stale prunes must FLAG.
    T-2 (AC-2a): single healthy daemon with scattered recovered errors must NOT flag.
    T-3 (AC-2b): spread-out restarts across a day must NOT flag.
    T-4 (AC-3): real f641174c fixture stays NOT flagged.
    T-5 (edge): no-timestamp logs fall back to flat-count (no regression).
    T-6 (edge): mixed parseable and None timestamps — anchor uses latest parseable.
    """

    def test_storm_flagged_stale_prunes_T1(self):
        """T-1 (AC-1): 5 rapid respawns with error blocks inside 1h are a storm.

        At base (flat-count only), the stale productive prunes from Gen 1 mask
        the current error storm → NOT flagged (FN). After the rate-window fix,
        recent_starts >= RATE_STORM_TRIP → flagged.

        The flat-count FN arises because productive_prunes (35) > cycle_errors (25),
        so the C7 branch doesn't fire. The rate path must fire first.
        """
        from datetime import datetime, timedelta
        now = datetime(2026, 6, 10, 20, 0, 0)
        # Gen 1: healthy, dead — 35 productive prune cycles, started 11h ago
        gen1_ts = (now - timedelta(hours=11)).isoformat()
        gen1 = _daemon_start(ts=gen1_ts)
        gen1 += "".join(_good_cycle(n=i) for i in range(1, 36))  # 35 productive prunes

        # Gen 2–6: current error storm — 5 rapid restarts, each with 5 error lines,
        # all within 30 minutes of each other
        storm = ""
        for i in range(5):
            storm_ts = (now - timedelta(minutes=30 - i * 5)).isoformat()
            storm += _daemon_start(ts=storm_ts)
            storm += "".join(_error_cycle(n=j) for j in range(1, 6))
            storm += _escalation()

        text = gen1 + storm
        rep = scan_log_text(text)
        # At base: productive_prunes=35 > cycle_errors=25 → NOT looping (FN)
        # After fix: recent_starts=5 >= RATE_STORM_TRIP=5 → looping=True
        self.assertTrue(rep.looping,
                        "rate-based path must flag a storm of 5 rapid respawns "
                        "even when stale prune lines outnumber errors in the window")
        self.assertIn("storm", rep.reason)

    def test_scattered_errors_not_flagged_T2(self):
        """T-2 (AC-2a): single daemon, 21 scattered recovered errors, must NOT flag.

        At base (flat-count): 21 >= 20 (loop_trip) AND 21 > 3 (productive_prunes)
        → looping=True (FP). After the rate-window fix, recent_starts=1 <
        RATE_STORM_TRIP=5 → rate path skips, flat-count doesn't fire because
        we never reach the flat-count unless timestamps are absent or < 2.

        Wait — the design is rate-FIRST: if recent_starts < trip, rate path skips
        and falls through to flat-count. This means T-2 actually STILL hits the
        flat-count. So the fix only resolves the FP if the rate path changes
        behavior here. The rate-first approach alone doesn't fix T-2 unless the
        rate path overrides or the flat-count is only used as fallback for
        insufficient timestamps (< 2 parseable starts). RE-READ the design:

        The fallback condition is: "if FEWER THAN 2 daemon-start timestamps parse,
        fall through to the existing flat-count logic". With 1 daemon-start that
        HAS a parseable timestamp, we have exactly 1 parseable timestamp.
        1 < 2 → we ARE in the fallback zone → flat-count runs.

        But the handoff says: "recent_starts=1 < RATE_STORM_TRIP=5 → not flagged".
        The design intent is that 1 recent start means no storm.

        The correct design: rate path runs when recent_starts is computed (any number
        of parseable timestamps). If recent_starts < RATE_STORM_TRIP, rate path
        says "not a storm" (does NOT flag). Then flat-count is NOT used at all when
        timestamps are available. Flat-count is ONLY the fallback when < 2 parseable
        timestamps exist (i.e., we can't compute a meaningful rate).

        This means T-2 must NOT hit flat-count (because we have 1 parseable timestamp
        and can compute recent_starts=1 < 5). The flat-count fallback threshold is
        "< 2 parseable timestamps" not "recent_starts < trip".
        """
        from datetime import datetime, timedelta
        now = datetime(2026, 6, 10, 10, 0, 0)
        # Single daemon started 2h ago with a parseable ISO timestamp
        text = _daemon_start(ts=(now - timedelta(hours=2)).isoformat())
        text += "".join(_good_cycle(n=i) for i in range(1, 4))  # 3 productive prunes
        # 21 error lines across 3 recovered streaks (7 per streak)
        for streak in range(3):
            text += "".join(_error_cycle(n=j) for j in range(1, 8))
        rep = scan_log_text(text)
        # At base: 21 >= 20 (loop_trip) AND 21 > 3 (productive_prunes) → looping=True (FP)
        # After fix: recent_starts=1 < RATE_STORM_TRIP=5 → rate says "not a storm";
        #            since we have >=1 parseable timestamp, flat-count fallback does NOT run
        self.assertFalse(rep.looping,
                         "a single daemon with scattered recovered errors must NOT be "
                         "flagged — 1 daemon-start within the rate window is not a storm")

    def test_spread_out_restarts_not_flagged_T3(self):
        """T-3 (AC-2b): 6 restarts 4h apart across a day must NOT flag as a storm.

        All 6 have parseable timestamps; total daemon_starts=6 > STORM_TRIP=5.
        But recent_starts within the 1h window = 1 (only the last one).
        Must NOT be flagged by rate path (not a storm in the temporal sense).
        """
        from datetime import datetime, timedelta
        now = datetime(2026, 6, 10, 20, 0, 0)
        parts = []
        for i in range(6):
            ts = (now - timedelta(hours=(5 - i) * 4)).isoformat()  # T-20h, T-16h, ..., T-0h
            parts.append(_daemon_start(ts=ts))
            parts.append(_error_cycle(n=1))  # one error per gen
        text = "".join(parts)
        rep = scan_log_text(text)
        self.assertFalse(rep.looping,
                         "spread-out restarts (4h apart) are not a storm — "
                         "recent_starts within 1h window must be 1, not 6")

    def test_real_fixture_unchanged_T4(self):
        """T-4 (AC-3): real f641174c fixture must still NOT be flagged after the fix.

        The fixture has 1 daemon-start with a parseable ISO timestamp.
        recent_starts=1 < RATE_STORM_TRIP=5 → rate path does not flag.
        Flat-count fallback does not run (>=1 parseable timestamp).
        The futile-prune path owns this fixture (10 futile < 20 trip → NOT flagged).
        """
        text = (FIXTURES / "f641174c_reload_loop.log").read_text(encoding="utf-8")
        rep = scan_log_text(text)
        self.assertFalse(rep.looping,
                         "f641174c real fixture: single K-exit run must remain NOT flagged")

    def test_no_timestamp_fallback_to_flat_count_T5(self):
        """T-5 (edge): log with no ISO timestamp in daemon-start header falls back to flat-count.

        "--- Guard daemon started ---" (no "at <ISO>" part) → 0 parseable timestamps
        → < 2 parseable timestamps → flat-count fallback runs.
        25 error lines, 0 productive prunes → flat-count fires → looping=True.
        """
        # Daemon-start line WITHOUT the "at <ISO>" part
        text = "--- Guard daemon started ---\nCWD: /x\n\n"
        text += "".join(_error_cycle(n=j) for j in range(1, 26))  # 25 error lines
        rep = scan_log_text(text)
        self.assertTrue(rep.looping,
                        "no-timestamp log must fall back to flat-count and flag 25 errors")

    def test_mixed_timestamps_anchor_uses_latest_parseable_T6(self):
        """T-6 (edge): if latest daemon-start has no ISO, anchor uses prior parseable ts.

        Gen 1 at T-2h (parseable), Gen 2 at T-30min (parseable),
        Gen 3 with no 'at <ISO>' suffix (None timestamp) → anchor=Gen 2 ts.
        We need 5 daemon-starts within 1h. Use 5 starts close together + 1 no-ISO.
        """
        from datetime import datetime, timedelta
        now = datetime(2026, 6, 10, 20, 0, 0)
        parts = []
        # 5 parseable starts within 30 min
        for i in range(5):
            ts = (now - timedelta(minutes=30 - i * 5)).isoformat()
            parts.append(_daemon_start(ts=ts))
            parts.append(_error_cycle(n=1))
        # 1 no-ISO start at the very end (will be the latest in text position)
        parts.append("--- Guard daemon started ---\nCWD: /x\n\n")
        parts.append(_error_cycle(n=1))
        text = "".join(parts)
        rep = scan_log_text(text)
        # Anchor = latest parseable = T-0min (5th parseable start)
        # recent_starts = all 5 parseable starts within 30min <= 3600s → 5 >= 5 → storm
        self.assertTrue(rep.looping,
                        "with mixed timestamps, anchor uses latest parseable; "
                        "5 starts within 1h must flag as a storm")
        self.assertIn("storm", rep.reason)


if __name__ == "__main__":
    unittest.main()
