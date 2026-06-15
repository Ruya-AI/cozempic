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


if __name__ == "__main__":
    unittest.main()
