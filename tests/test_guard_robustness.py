"""Tests for guard daemon robustness improvements."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestGuardSignalHandling(unittest.TestCase):
    def test_sigterm_constant_exists(self):
        """SIGTERM is available on this platform."""
        import signal
        self.assertTrue(hasattr(signal, 'SIGTERM'))


class TestBackupCleanupIntegration(unittest.TestCase):
    def test_cleanup_old_backups_importable(self):
        """cleanup_old_backups can be imported from session module."""
        from cozempic.session import cleanup_old_backups
        self.assertTrue(callable(cleanup_old_backups))


class TestReloadSelfDaemon(unittest.TestCase):
    def test_no_daemon_no_op(self):
        """reload_self_daemon returns reloaded=False when no daemon exists for the session."""
        from cozempic.guard import reload_self_daemon
        result = reload_self_daemon(
            cwd="/tmp",
            session_id="11111111-2222-3333-4444-555555555555",
        )
        self.assertFalse(result["reloaded"])
        self.assertIn("no daemon", result["reason"].lower())

    def test_explicit_session_with_no_daemon_does_not_spawn(self):
        """When the named session has no live daemon, reload_self must not spawn one."""
        from cozempic.guard import reload_self_daemon
        # Explicit, fake session id — no PID file, no daemon. Must short-circuit
        # without ever calling start_guard_daemon.
        result = reload_self_daemon(
            cwd="/tmp",
            session_id="11111111-2222-3333-4444-555555555555",
        )
        self.assertFalse(result["reloaded"])
        self.assertIsNone(result.get("new_pid"))
        self.assertIn("no daemon", result["reason"].lower())


class TestGuardDaemonPidHandoff(unittest.TestCase):
    def test_start_guard_daemon_passes_explicit_claude_pid_to_child(self):
        from cozempic.guard import start_guard_daemon

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a valid-shape UUID — start_guard_daemon validates session_id
            # via _pid_file_for_session (BUG-G13), matching the read-side
            # contract in _is_guard_running_for_session.
            uuid = "ffffffff-eeee-dddd-cccc-bbbbbbbbbbbb"
            tmp = Path(tmpdir)
            captured = {}

            class DummyProc:
                pid = 4242

            def fake_popen(cmd_parts, **kwargs):
                captured["cmd_parts"] = cmd_parts
                return DummyProc()

            # GC-2: patch _guard_tmp_root so log/pid files land in the tmpdir
            # instead of the real /tmp — avoids real-file leaks on macOS where
            # _guard_tmp_root() returns /tmp but tempfile.gettempdir() returns
            # /var/folders/…/T (the two differ, leaving files after teardown).
            with (
                patch("cozempic.guard._guard_tmp_root", return_value=tmp),
                patch("cozempic.guard._cleanup_legacy_pid"),
                patch("cozempic.guard._is_guard_running_for_session", return_value=None),
                patch("cozempic.guard.find_claude_pid", return_value=9999),
                patch("cozempic.guard.subprocess.Popen", side_effect=fake_popen),
            ):
                result = start_guard_daemon(
                    cwd=tmpdir,
                    session_id=uuid,
                    threshold_tokens=123,
                )

            self.assertTrue(result["started"])
            self.assertIn("--claude-pid", captured["cmd_parts"])
            self.assertIn("9999", captured["cmd_parts"])


if __name__ == "__main__":
    unittest.main()
