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

import signal
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


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
        try:
            from cozempic.helpers import _pid_is_alive
        except ImportError:
            self.fail("_pid_is_alive not found in cozempic.helpers — GC-3 not applied (RED)")
        self.assertTrue(callable(_pid_is_alive))

    def test_session_imports_pid_is_alive_from_helpers(self):
        """cozempic.session._pid_alive must delegate to helpers._pid_is_alive (or be the same function)."""
        from cozempic.helpers import _pid_is_alive as canonical
        from cozempic import session

        # After GC-3: session._pid_alive IS helpers._pid_is_alive
        # (either via `from .helpers import _pid_is_alive as _pid_alive` or the function object matches)
        self.assertIs(getattr(session, "_pid_alive", None), canonical,
                      "session._pid_alive must be the canonical helpers._pid_is_alive after GC-3")

    def test_watchdog_imports_pid_is_alive_from_helpers(self):
        """cozempic.watchdog._pid_alive must delegate to helpers._pid_is_alive."""
        from cozempic.helpers import _pid_is_alive as canonical
        from cozempic import watchdog

        self.assertIs(getattr(watchdog, "_pid_alive", None), canonical,
                      "watchdog._pid_alive must be the canonical helpers._pid_is_alive after GC-3")


class TestPidIsAliveCanonicalBehavior(unittest.TestCase):
    """The canonical _pid_is_alive must match guard.py's behavior on all error paths."""

    def _import_canonical(self):
        from cozempic.helpers import _pid_is_alive
        return _pid_is_alive

    def test_dead_pid_returns_false(self):
        fn = self._import_canonical()
        with patch("os.kill", side_effect=ProcessLookupError):
            self.assertFalse(fn(99999))

    def test_permission_error_returns_true(self):
        """PermissionError: process exists but owned by another user — alive."""
        fn = self._import_canonical()
        with patch("os.kill", side_effect=PermissionError):
            self.assertTrue(fn(99999))

    def test_overflow_error_returns_false(self):
        """Malformed huge PID → dead."""
        fn = self._import_canonical()
        with patch("os.kill", side_effect=OverflowError):
            self.assertFalse(fn(99999))

    def test_posix_unknown_oserror_returns_true(self):
        """POSIX unknown OSError → fail-open (assume alive). This is the behavioral
        fix vs session.py's old _pid_alive which returned False here."""
        import os as _os
        fn = self._import_canonical()
        with (
            patch("os.kill", side_effect=OSError("unexpected")),
            patch("os.name", "posix"),
        ):
            # canonical: return os.name != "nt" → True on POSIX
            self.assertTrue(fn(99999))

    def test_windows_oserror_returns_false(self):
        """Windows OSError on os.kill(pid, 0) → dead."""
        fn = self._import_canonical()
        with (
            patch("os.kill", side_effect=OSError("WinError 87")),
            patch("os.name", "nt"),
        ):
            self.assertFalse(fn(99999))

    def test_zero_pid_returns_false(self):
        fn = self._import_canonical()
        self.assertFalse(fn(0))

    def test_negative_pid_returns_false(self):
        fn = self._import_canonical()
        self.assertFalse(fn(-1))

    def test_non_int_pid_returns_false(self):
        fn = self._import_canonical()
        self.assertFalse(fn("notanint"))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
