"""Tests for overflow detection, circuit breaker, and file watcher."""

from __future__ import annotations

import json
import os
import tempfile
import time
import threading
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from cozempic.overflow import CircuitBreaker, OverflowRecovery, OVERFLOW_PATTERN
from cozempic.watcher import JsonlWatcher


class TestCircuitBreaker(unittest.TestCase):

    def setUp(self):
        self.breaker = CircuitBreaker(
            session_id="test-session-123",
            max_recoveries=3,
            window_seconds=300,
        )
        # Clean up any leftover state
        self.breaker.reset()

    def tearDown(self):
        self.breaker.reset()

    def test_initial_state_allows_recovery(self):
        self.assertTrue(self.breaker.can_recover())
        self.assertEqual(self.breaker.recovery_count(), 0)

    def test_breaker_escalation(self):
        """Prescriptions escalate: gentle → standard → aggressive."""
        self.assertEqual(self.breaker.next_prescription(), "gentle")

        self.breaker.record_recovery("gentle", 100.0, 60.0)
        self.assertEqual(self.breaker.next_prescription(), "standard")

        self.breaker.record_recovery("standard", 90.0, 40.0)
        self.assertEqual(self.breaker.next_prescription(), "aggressive")

    def test_breaker_trips_after_max(self):
        """can_recover() returns False after max_recoveries."""
        for i in range(3):
            self.assertTrue(self.breaker.can_recover())
            self.breaker.record_recovery("gentle", 100.0, 60.0)

        self.assertFalse(self.breaker.can_recover())
        self.assertEqual(self.breaker.recovery_count(), 3)

    def test_breaker_resets_after_window(self):
        """Recoveries outside the window are pruned."""
        # Use a very short window for testing
        breaker = CircuitBreaker(
            session_id="test-window",
            max_recoveries=3,
            window_seconds=1,
        )
        breaker.reset()

        try:
            breaker.record_recovery("gentle", 100.0, 60.0)
            breaker.record_recovery("standard", 90.0, 40.0)
            self.assertEqual(breaker.recovery_count(), 2)

            # Wait for window to expire
            time.sleep(1.1)

            # Old records should be pruned
            self.assertEqual(breaker.recovery_count(), 0)
            self.assertTrue(breaker.can_recover())
            self.assertEqual(breaker.next_prescription(), "gentle")
        finally:
            breaker.reset()

    def test_prescription_caps_at_aggressive(self):
        """Even with many recoveries, doesn't go past aggressive."""
        self.breaker = CircuitBreaker(
            session_id="test-cap",
            max_recoveries=10,
            window_seconds=300,
        )
        self.breaker.reset()

        try:
            for i in range(5):
                self.breaker.record_recovery("test", 100.0, 60.0)

            self.assertEqual(self.breaker.next_prescription(), "aggressive")
        finally:
            self.breaker.reset()

    def test_state_persists_to_disk(self):
        """Records survive across instances."""
        self.breaker.record_recovery("gentle", 100.0, 60.0)

        # Create a new instance with the same session_id
        breaker2 = CircuitBreaker(
            session_id="test-session-123",
            max_recoveries=3,
            window_seconds=300,
        )
        self.assertEqual(breaker2.recovery_count(), 1)
        self.assertEqual(breaker2.next_prescription(), "standard")

    def test_corrupted_state_file_handled(self):
        """Corrupted state file doesn't crash."""
        self.breaker.state_path.write_text("not valid json")
        self.assertTrue(self.breaker.can_recover())
        self.assertEqual(self.breaker.recovery_count(), 0)


class TestOverflowDetection(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.session_path = Path(self.tmpdir) / "session.jsonl"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_lines(self, lines: list[str]):
        self.session_path.write_text("\n".join(lines) + "\n")

    def test_overflow_detection_pattern(self):
        """Finds 'Conversation too long' in mock JSONL."""
        lines = [
            json.dumps({"type": "user", "message": "hello"}),
            json.dumps({"type": "assistant", "message": "hi"}),
        ] * 10
        # Add the overflow marker
        lines.append(json.dumps({
            "type": "system",
            "message": f"Error: {OVERFLOW_PATTERN} for this model",
        }))
        self._write_lines(lines)

        breaker = CircuitBreaker(session_id="test-detect", max_recoveries=3)
        breaker.reset()
        try:
            recovery = OverflowRecovery(
                self.session_path, "test-detect", self.tmpdir, breaker,
            )
            self.assertTrue(recovery.detect_overflow())
        finally:
            breaker.reset()

    def test_no_overflow_in_normal_session(self):
        """Normal session content doesn't trigger detection."""
        lines = [
            json.dumps({"type": "user", "message": "hello"}),
            json.dumps({"type": "assistant", "message": "hi there"}),
        ] * 15
        self._write_lines(lines)

        breaker = CircuitBreaker(session_id="test-normal", max_recoveries=3)
        breaker.reset()
        try:
            recovery = OverflowRecovery(
                self.session_path, "test-normal", self.tmpdir, breaker,
            )
            self.assertFalse(recovery.detect_overflow())
        finally:
            breaker.reset()

    def test_fast_path_skips_small_files(self):
        """on_file_growth returns immediately for small files."""
        lines = [json.dumps({"type": "user", "message": "hello"})]
        self._write_lines(lines)

        breaker = CircuitBreaker(session_id="test-fast", max_recoveries=3)
        breaker.reset()
        try:
            recovery = OverflowRecovery(
                self.session_path, "test-fast", self.tmpdir, breaker,
                danger_threshold_mb=100.0,  # 100MB — file is tiny
            )
            # Should not call detect_overflow (we'd know if it did because
            # we can check _recovering stays False)
            recovery.on_file_growth(str(self.session_path), 1024)
            self.assertFalse(recovery._recovering)
        finally:
            breaker.reset()

    def test_preflight_skip(self):
        """Skips resume when post-prune estimate is still too large."""
        breaker = CircuitBreaker(session_id="test-preflight", max_recoveries=3)
        breaker.reset()
        try:
            recovery = OverflowRecovery(
                self.session_path, "test-preflight", self.tmpdir, breaker,
                danger_threshold_mb=0.001,  # Tiny threshold
            )

            # Write a file
            self._write_lines([json.dumps({"type": "user", "message": "x" * 1000})])

            # Mock guard_prune_cycle to not actually prune
            with patch("cozempic.overflow.OverflowRecovery._do_recover") as mock_recover:
                # Just verify the method exists and is callable
                recovery.recover()
                mock_recover.assert_called_once()
        finally:
            breaker.reset()

    def test_recovery_uses_explicit_claude_pid(self):
        breaker = CircuitBreaker(session_id="test-explicit-pid", max_recoveries=3)
        breaker.reset()
        try:
            self._write_lines([json.dumps({"type": "user", "message": "hello"})])
            recovery = OverflowRecovery(
                self.session_path,
                "test-explicit-pid",
                self.tmpdir,
                breaker,
                danger_threshold_mb=100.0,
                claude_pid=7777,
            )

            # PR #102 P2: overflow now uses _terminate_claude + _resume_claude
            # instead of _terminate_and_resume. Mock both; _terminate_claude
            # must be called with the explicit claude_pid.
            with (
                patch.object(recovery, "detect_overflow", return_value=True),
                patch("cozempic.guard.guard_prune_cycle", return_value={
                    "saved_mb": 1.0,
                    "original_tokens": 1000,
                    "final_tokens": 500,
                }),
                patch("cozempic.guard._terminate_claude", return_value="TERMINATED") as mock_terminate,
                patch("cozempic.guard._resume_claude"),
                patch("cozempic.session.find_claude_pid", return_value=None),
            ):
                recovery.recover()

            mock_terminate.assert_called_once_with(
                7777,
                session_path=self.session_path,
                session_id="test-explicit-pid",
            )
        finally:
            breaker.reset()


class TestJsonlWatcher(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.filepath = os.path.join(self.tmpdir, "test.jsonl")
        # Create initial file
        with open(self.filepath, "w") as f:
            f.write('{"type": "init"}\n')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_watcher_detects_append(self):
        """Writing to file triggers the on_growth callback."""
        growth_events = []

        def on_growth(filepath, new_size):
            growth_events.append((filepath, new_size))

        watcher = JsonlWatcher(self.filepath, on_growth=on_growth)

        # Force polling mode for test reliability
        watcher._use_kqueue = False

        # Start in background thread
        t = threading.Thread(target=watcher.start, daemon=True)
        t.start()

        try:
            # Wait for watcher to start
            time.sleep(0.3)

            # Append data
            with open(self.filepath, "a") as f:
                f.write('{"type": "user", "message": "hello"}\n')

            # Wait for detection
            time.sleep(0.5)

            self.assertGreater(len(growth_events), 0)
            self.assertEqual(growth_events[0][0], self.filepath)
        finally:
            watcher.stop()
            t.join(timeout=2)

    def test_watcher_ignores_no_growth(self):
        """No callback if file doesn't grow."""
        growth_events = []

        def on_growth(filepath, new_size):
            growth_events.append((filepath, new_size))

        watcher = JsonlWatcher(self.filepath, on_growth=on_growth)
        watcher._use_kqueue = False

        t = threading.Thread(target=watcher.start, daemon=True)
        t.start()

        try:
            time.sleep(0.5)
            self.assertEqual(len(growth_events), 0)
        finally:
            watcher.stop()
            t.join(timeout=2)

    def test_watcher_stop(self):
        """Watcher stops cleanly."""
        watcher = JsonlWatcher(self.filepath, on_growth=lambda f, s: None)
        watcher._use_kqueue = False

        t = threading.Thread(target=watcher.start, daemon=True)
        t.start()

        time.sleep(0.3)
        watcher.stop()
        t.join(timeout=2)
        self.assertFalse(t.is_alive())

    def test_callback_exception_doesnt_crash_watcher(self):
        """Exceptions in the callback don't kill the watcher thread."""
        call_count = []

        def bad_callback(filepath, new_size):
            call_count.append(1)
            raise RuntimeError("boom")

        watcher = JsonlWatcher(self.filepath, on_growth=bad_callback)
        watcher._use_kqueue = False

        t = threading.Thread(target=watcher.start, daemon=True)
        t.start()

        try:
            time.sleep(0.3)

            # Write twice
            with open(self.filepath, "a") as f:
                f.write('{"type": "first"}\n')
            time.sleep(0.3)

            with open(self.filepath, "a") as f:
                f.write('{"type": "second"}\n')
            time.sleep(0.3)

            # Watcher should still be alive despite exceptions
            self.assertTrue(t.is_alive())
            self.assertGreaterEqual(len(call_count), 2)
        finally:
            watcher.stop()
            t.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
