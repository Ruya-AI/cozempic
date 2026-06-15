"""RED tests for guard-cleanup PR (GC-1, GC-2, GC-3).

GC-1 — SIGTERM handler leaks pidfile + armed sentinel if the signal fires
        before the try: block in start_guard (guard.py ~896). The handler must
        call _safe_unlink_session_pidfile and clear_armed.

GC-2 — test_guard_robustness.py:56 + test_guard_reload_watcher_poll.py:159
        use hardcoded Path("/tmp/cozempic_guard_*.log") — real files that
        escape test teardown on macOS (where gettempdir() is /var/folders/…,
        not /tmp). Both tests should patch _guard_tmp_root so the log paths
        live in a TemporaryDirectory.

GC-3 — guard.py:_pid_is_alive (canonical), session.py:_pid_alive, and
        watchdog.py:_pid_alive are three separate implementations of the same
        function. The canonical is moved to helpers.py; session.py and
        watchdog.py import it. Behavioral alignment: the canonical returns
        True on POSIX-unknown-OSError (fail-open, never skip a live process);
        session.py previously returned False there (premature dead-call).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cozempic.helpers import _pid_is_alive as _canonical_pid_is_alive


# ─────────────────────────── GC-1: SIGTERM handler ───────────────────────────

class TestSigtermHandlerCleansUp(unittest.TestCase):
    """_graceful_shutdown must unlink the session pidfile and clear armed sentinel."""

    def test_sigterm_handler_calls_safe_unlink(self):
        """SIGTERM fires after pidfile is written — _safe_unlink_session_pidfile must be called."""
        from cozempic import guard

        unlinked_ids = []
        cleared_ids = []

        def fake_unlink(session_id):
            unlinked_ids.append(session_id)

        def fake_clear(session_id, session_path):
            cleared_ids.append(session_id)

        sid = "aabbccdd-eeff-0011-2233-445566778899"

        with (
            patch.object(guard, "_safe_unlink_session_pidfile", side_effect=fake_unlink),
            patch.object(guard, "clear_armed", side_effect=fake_clear),
            patch.object(guard, "checkpoint_team"),
        ):
            # Simulate what the handler does: look up the registered handler and call it.
            # We test via the handler's side-effects, not by sending a real signal.
            # The handler must accept (session_id, session_path) in its closure.
            #
            # If the fix is in place, guard._make_sigterm_handler (or similar) exposes
            # the cleanup logic. For now verify through the module's public contract:
            # the handler registered by start_guard for a given session_id must call
            # _safe_unlink_session_pidfile(session_id) when invoked.
            #
            # RED condition: before the fix, _graceful_shutdown calls only
            # checkpoint_team and overflow_watcher.stop — no pidfile cleanup.
            handler = guard._make_sigterm_handler(
                session_id=sid,
                session_path=Path("/tmp"),
                overflow_watcher=None,
            )
            try:
                handler(signal.SIGTERM, None)
            except SystemExit:
                pass

        self.assertIn(sid, unlinked_ids,
                      "_graceful_shutdown must call _safe_unlink_session_pidfile(session_id)")
        self.assertIn(sid, cleared_ids,
                      "_graceful_shutdown must call clear_armed(session_id, session_path)")


# ─────────────────────────── GC-2: /tmp log leak ─────────────────────────────

class TestWatcherLogUsesGuardTmpRoot(unittest.TestCase):
    """guard_log in test_guard_reload_watcher_poll.py:159 must use _guard_tmp_root,
    not the hardcoded Path('/tmp/cozempic_guard.log').

    RED: the test file still has the hardcoded string, so reading it shows '/tmp'.
    GREEN: the file is patched to use _guard_tmp_root() / '...' instead.
    """

    def test_watcher_log_not_hardcoded_slash_tmp(self):
        """test_guard_reload_watcher_poll.py must not have a hardcoded /tmp guard log path."""
        import inspect
        import tests.test_guard_reload_watcher_poll as mod
        src = inspect.getsource(mod)
        # The GC-2 fix removes: Path("/tmp/cozempic_guard.log")
        # After fix: _guard_tmp_root() or tmp_path is used instead.
        self.assertNotIn(
            'Path("/tmp/cozempic_guard.log")',
            src,
            "test_guard_reload_watcher_poll.py still has hardcoded Path('/tmp/cozempic_guard.log') "
            "at line 159 — this leaks a real file on macOS. GC-2 not applied (RED).",
        )


class TestRobustnessTestNoLeak(unittest.TestCase):
    """test_guard_robustness.py:56 must use _guard_tmp_root, not Path('/tmp')."""

    def test_robustness_log_not_hardcoded_slash_tmp(self):
        """test_guard_robustness.py must not have hardcoded Path('/tmp') for session_log."""
        import inspect
        import tests.test_guard_robustness as mod
        src = inspect.getsource(mod)
        # The GC-2 fix replaces:
        #   session_log = Path("/tmp") / f"cozempic_guard_{uuid[:12]}.log"
        #   session_pid = Path("/tmp") / f"cozempic_guard_{uuid[:12]}.pid"
        # with _guard_tmp_root()-based paths inside the patch context.
        self.assertNotIn(
            'Path("/tmp") / f"cozempic_guard_{uuid[:12]}.log"',
            src,
            "test_guard_robustness.py still has hardcoded Path('/tmp') session_log at line 56 "
            "— real file leaks on macOS. GC-2 not applied (RED).",
        )


# ─────────────────────────── GC-3: _pid_is_alive consolidation ───────────────

class TestPidIsAliveMigratedToHelpers(unittest.TestCase):
    """helpers.py must export _pid_is_alive after GC-3."""

    def test_pid_is_alive_importable_from_helpers(self):
        """_pid_is_alive must be importable from cozempic.helpers."""
        self.assertTrue(callable(_canonical_pid_is_alive))

    def test_session_imports_pid_is_alive_from_helpers(self):
        """cozempic.session._pid_alive must be the canonical helpers._pid_is_alive."""
        from cozempic import session
        self.assertIs(getattr(session, "_pid_alive", None), _canonical_pid_is_alive,
                      "session._pid_alive must be the canonical helpers._pid_is_alive after GC-3")

    def test_watchdog_imports_pid_is_alive_from_helpers(self):
        """cozempic.watchdog._pid_alive must be the canonical helpers._pid_is_alive."""
        from cozempic import watchdog
        self.assertIs(getattr(watchdog, "_pid_alive", None), _canonical_pid_is_alive,
                      "watchdog._pid_alive must be the canonical helpers._pid_is_alive after GC-3")


class TestPidIsAliveCanonicalBehavior(unittest.TestCase):
    """The canonical _pid_is_alive must match guard.py's behavior on all error paths."""

    def test_dead_pid_returns_false(self):
        with patch("os.kill", side_effect=ProcessLookupError):
            self.assertFalse(_canonical_pid_is_alive(99999))

    def test_permission_error_returns_true(self):
        """PermissionError: process exists but owned by another user — alive."""
        with patch("os.kill", side_effect=PermissionError):
            self.assertTrue(_canonical_pid_is_alive(99999))

    def test_overflow_error_returns_false(self):
        """Malformed huge PID → dead."""
        with patch("os.kill", side_effect=OverflowError):
            self.assertFalse(_canonical_pid_is_alive(99999))

    def test_posix_unknown_oserror_returns_true(self):
        """POSIX unknown OSError → fail-open (assume alive). This is the behavioral
        fix vs session.py's old _pid_alive which returned False here."""
        with (
            patch("os.kill", side_effect=OSError("unexpected")),
            patch("os.name", "posix"),
        ):
            # canonical: return os.name != "nt" → True on POSIX
            self.assertTrue(_canonical_pid_is_alive(99999))

    def test_windows_oserror_returns_false(self):
        """Windows OSError on os.kill(pid, 0) → dead."""
        with (
            patch("os.kill", side_effect=OSError("WinError 87")),
            patch("os.name", "nt"),
        ):
            self.assertFalse(_canonical_pid_is_alive(99999))

    def test_zero_pid_returns_false(self):
        self.assertFalse(_canonical_pid_is_alive(0))

    def test_negative_pid_returns_false(self):
        self.assertFalse(_canonical_pid_is_alive(-1))

    def test_non_int_pid_returns_false(self):
        """Garbage (non-parseable) string → False.  Only numeric strings must be coerced."""
        self.assertFalse(_canonical_pid_is_alive("notanint"))  # type: ignore[arg-type]

    def test_numeric_string_pid_coerced_and_probed(self):
        """session.py:581 passes string dict-keys to _pid_is_alive.

        Pre-fix: isinstance("1234", int) is False → instant False → live entries
        pruned as dead (bug: record_active_transcript wipes existing live PIDs).
        Post-fix: numeric strings are coerced via int() before the liveness probe.

        Use os.getpid() (this process, guaranteed alive) as the numeric string.

        RED at base: _pid_is_alive(str(os.getpid())) returns False (isinstance guard
        rejects strings before os.kill is reached).
        GREEN after fix: coercion to int → os.kill succeeds → True.
        """
        result = _canonical_pid_is_alive(str(os.getpid()))  # type: ignore[arg-type]
        self.assertTrue(result,
                        f"_pid_is_alive('{os.getpid()}') returned False — "
                        f"numeric string pid must be coerced to int and probed, "
                        f"not rejected as non-int")


# ─────────── C-1: record_active_transcript retains live pids ──────────────────

class TestRecordActiveTranscriptRetainsLivePids(unittest.TestCase):
    """record_active_transcript must NOT prune entries whose pid is alive.

    Root cause of bug: _pid_is_alive now receives string keys (JSON dict keys
    are always strings) but the isinstance(pid, int) guard in the old helpers.py
    rejects ALL strings → live entries vanish on every write (wipes #124).

    This test is UNMOCKED with respect to _pid_is_alive — it calls real process
    liveness probes so the string-vs-int mismatch is visible end-to-end.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="cozempic_rat_")
        self._claude_dir = Path(self._tmpdir) / ".claude"
        self._claude_dir.mkdir()
        self._sessions_file = self._claude_dir / "cozempic-active-sessions.json"

        # Pre-seed with two live PIDs (current process + parent).  Both are
        # guaranteed alive throughout the test.
        self._live_pid1 = os.getpid()
        self._live_pid2 = os.getppid()
        data = {
            str(self._live_pid1): {
                "transcript_path": str(self._claude_dir / "session1.jsonl"),
                "session_id": "session1",
                "recorded_at": "2026-06-15T00:00:00",
            },
            str(self._live_pid2): {
                "transcript_path": str(self._claude_dir / "session2.jsonl"),
                "session_id": "session2",
                "recorded_at": "2026-06-15T00:00:01",
            },
        }
        self._sessions_file.write_text(json.dumps(data), encoding="utf-8")

        # Also create a real transcript file for the call below to accept.
        self._new_transcript = self._claude_dir / "session3.jsonl"
        self._new_transcript.write_text('{"type":"user"}\n', encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_live_pid_entries_retained_after_write(self):
        """record_active_transcript with a new PID must NOT prune live pre-existing entries.

        RED at #136 HEAD: both live-pid entries are wiped because
        _pid_is_alive(str(pid)) returns False (isinstance guard rejects strings).
        GREEN after fix: string keys are coerced to int → live probes pass →
        both entries survive.
        """
        from cozempic.session import record_active_transcript

        # Re-use live_pid1 as the caller slot — it is already in the file so
        # it gets overwritten (that's fine).  We care that live_pid2 survives.
        new_pid = self._live_pid1

        with (
            patch("cozempic.session.get_claude_dir", return_value=self._claude_dir),
        ):
            record_active_transcript(
                transcript_path=str(self._new_transcript),
                claude_pid=new_pid,
            )

        data = json.loads(self._sessions_file.read_text(encoding="utf-8"))
        # live_pid2 must survive (it was in the file, its pid is alive, and it
        # is NOT the caller pid, so it should only be pruned if _pid_is_alive
        # says it's dead — which is the bug).
        self.assertIn(
            str(self._live_pid2),
            data,
            f"live pid {self._live_pid2} was pruned by record_active_transcript "
            f"even though it is alive — numeric-string coercion missing in "
            f"_pid_is_alive (C-1 regression)",
        )


# ─────────── GC-1 MED: checkpoint_team raising must not block cleanup ─────────

class TestSigtermHandlerCleanupSurvivesCheckpointRaise(unittest.TestCase):
    """If checkpoint_team raises, _safe_unlink and clear_armed must still run.

    GC-1 MED: the handler calls checkpoint_team THEN cleanup. If checkpoint_team
    raises (e.g. serialization error on a corrupt TeamState), the cleanup is
    skipped — pidfile and armed-sentinel leak.

    Fix: wrap checkpoint_team in try/finally so cleanup always runs.
    """

    def test_cleanup_runs_even_if_checkpoint_raises(self):
        """_safe_unlink_session_pidfile and clear_armed must be called even when
        checkpoint_team raises.

        RED at base: the handler is a straight sequence — checkpoint_team()
        raises → execution stops → unlink/clear never reached.
        GREEN after fix: checkpoint_team in try/finally block.
        """
        from cozempic import guard

        unlinked = []
        cleared = []
        sid = "deadbeef-0000-0000-0000-000000000000"

        with (
            patch.object(guard, "checkpoint_team", side_effect=RuntimeError("boom")),
            patch.object(guard, "_safe_unlink_session_pidfile",
                         side_effect=lambda s: unlinked.append(s)),
            patch.object(guard, "clear_armed",
                         side_effect=lambda s, p: cleared.append(s)),
        ):
            handler = guard._make_sigterm_handler(
                session_id=sid,
                session_path=Path("/tmp"),
                overflow_watcher=None,
            )
            try:
                handler(signal.SIGTERM, None)
            except SystemExit:
                pass
            except RuntimeError:
                # If the fix isn't in place, RuntimeError propagates out of the handler.
                pass

        self.assertIn(sid, unlinked,
                      "_safe_unlink_session_pidfile not called when checkpoint_team raised")
        self.assertIn(sid, cleared,
                      "clear_armed not called when checkpoint_team raised")


if __name__ == "__main__":
    unittest.main()
