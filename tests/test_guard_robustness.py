"""Tests for guard daemon robustness improvements."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestGuardSignalHandling(unittest.TestCase):
    def test_sigterm_constant_exists(self):
        """SIGTERM is available on this platform."""
        import signal
        self.assertTrue(hasattr(signal, 'SIGTERM'))


class TestGuardLogEncoding(unittest.TestCase):
    def test_guard_log_preserves_surrogateescaped_path_bytes(self):
        from cozempic.guard import _open_guard_log

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "guard.log"
            with _open_guard_log(log_path) as log:
                log.write("CWD: bad" + chr(0xDCFF) + "\n")

            self.assertEqual(log_path.read_bytes(), b"CWD: bad\xff\n")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires mkfifo")
    def test_guard_log_fifo_is_rejected_without_blocking(self):
        from cozempic.guard import _open_guard_log

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "guard.log"
            os.mkfifo(log_path)
            with self.assertRaises(OSError):
                _open_guard_log(log_path)

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "requires O_NOFOLLOW")
    def test_guard_log_symlink_is_rejected(self):
        from cozempic.guard import _open_guard_log

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "target.log"
            target.write_text("keep")
            log_path = Path(tmpdir) / "guard.log"
            log_path.symlink_to(target)
            with self.assertRaises(OSError):
                _open_guard_log(log_path)
            self.assertEqual(target.read_text(), "keep")


class TestReloadStateRegularFiles(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires mkfifo")
    def test_read_armed_ignores_fifo_without_blocking(self):
        from cozempic.guard import read_armed

        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = Path(tmpdir) / "armed.json"
            os.mkfifo(sentinel)
            with patch("cozempic.guard._reload_armed_path", return_value=sentinel):
                self.assertIsNone(read_armed("test-session"))

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "requires O_NOFOLLOW")
    def test_read_armed_ignores_symlink(self):
        from cozempic.guard import read_armed

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "target.json"
            target.write_text('{"warned": true}')
            sentinel = Path(tmpdir) / "armed.json"
            sentinel.symlink_to(target)
            with patch("cozempic.guard._reload_armed_path", return_value=sentinel):
                self.assertIsNone(read_armed("test-session"))

    def test_read_armed_ignores_non_object_json(self):
        from cozempic.guard import read_armed

        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = Path(tmpdir) / "armed.json"
            sentinel.write_text("[]")
            with patch("cozempic.guard._reload_armed_path", return_value=sentinel):
                self.assertIsNone(read_armed("test-session"))

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires mkfifo")
    def test_reload_ledger_ignores_fifo_without_blocking(self):
        from cozempic.guard import _reload_rate_exceeded

        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = Path(tmpdir) / "reload.history"
            os.mkfifo(ledger)
            self.assertEqual(_reload_rate_exceeded(ledger, now=1000.0), (False, 1))

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "requires O_NOFOLLOW")
    def test_reload_ledger_ignores_symlink(self):
        from cozempic.guard import _reload_rate_exceeded

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "target.history"
            target.write_text("[]")
            ledger = Path(tmpdir) / "reload.history"
            ledger.symlink_to(target)
            self.assertEqual(_reload_rate_exceeded(ledger, now=1000.0), (False, 1))
            self.assertEqual(target.read_text(), "[]")


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

    def test_retry_preserves_orphaned_pid_from_first_spawn(self):
        from cozempic.guard import reload_self_daemon

        session_id = "11111111-2222-3333-4444-555555555555"
        with tempfile.TemporaryDirectory() as tmpdir:
            pid_path = Path(tmpdir) / "guard.pid"
            first = {"started": False, "already_running": False, "orphaned_pid": 111}
            second = {"started": True, "already_running": False, "pid": 222, "log_file": "guard.log"}
            with (
                patch("cozempic.guard._is_guard_running_for_session", return_value=123),
                patch("cozempic.guard._is_cozempic_guard_process", return_value=True),
                patch("cozempic.guard._pid_file_for_session", return_value=pid_path),
                patch("cozempic.guard._wait_for_exit", return_value=True),
                patch("cozempic.guard.os.kill"),
                patch("cozempic.guard.time.sleep"),
                patch("cozempic.guard.start_guard_daemon", side_effect=[first, second]),
            ):
                result = reload_self_daemon(cwd=tmpdir, session_id=session_id)

        self.assertTrue(result["reloaded"])
        self.assertEqual(result["new_pid"], 222)
        self.assertEqual(result["orphaned_pid"], 111)


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

    def test_start_guard_daemon_reclaims_fresh_leftover_publication_temp(self):
        from cozempic import guard

        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = "ffffffff-eeee-dddd-cccc-bbbbbbbbbbbb"
            tmp = Path(tmpdir)

            class DummyProc:
                pid = 4242

            with (
                patch("cozempic.guard._guard_tmp_root", return_value=tmp),
                patch("cozempic.guard._cleanup_legacy_pid"),
                patch("cozempic.guard._is_guard_running_for_session", return_value=None),
                patch("cozempic.guard.find_claude_pid", return_value=9999),
                patch("cozempic.guard.subprocess.Popen", return_value=DummyProc()),
            ):
                pid_path = guard._pid_file_for_session(session_id)
                tmp_path = pid_path.with_suffix(".pid.tmp")
                tmp_path.write_text("leftover reservation")
                result = guard.start_guard_daemon(cwd=tmpdir, session_id=session_id)

            self.assertTrue(result["started"])
            self.assertFalse(tmp_path.exists())

    def test_start_guard_daemon_labels_subprocess_errors(self):
        from cozempic.guard import start_guard_daemon

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("cozempic.guard._guard_tmp_root", return_value=Path(tmpdir)),
                patch("cozempic.guard._cleanup_legacy_pid"),
                patch("cozempic.guard._is_guard_running_for_session", return_value=None),
                patch("cozempic.guard.find_claude_pid", return_value=9999),
                patch("cozempic.guard.subprocess.Popen", side_effect=OSError("no slots")),
            ):
                result = start_guard_daemon(
                    cwd=tmpdir,
                    session_id="ffffffff-eeee-dddd-cccc-bbbbbbbbbbbb",
                    threshold_tokens=123,
                )

        self.assertFalse(result["started"])
        self.assertTrue(result["reason"].startswith("guard subprocess:"))


if __name__ == "__main__":
    unittest.main()
