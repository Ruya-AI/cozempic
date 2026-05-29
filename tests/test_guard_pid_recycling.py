"""RED tests for PID start-time tracking (recycled-PID resurrection vector).

Architect spec: AUDIT_REPORT_pid_recycling.md (commit 5af7c0d)
Vector: os.kill(pid, 0) answers liveness, not identity. If Claude dies and its
PID is recycled to a live unrelated process AND the JSONL is freshly written,
_is_claude_process's mtime fallback still returns True → resurrection.
Fix: record (pid, start_time) at guard startup via _record_claude_identity;
gate _terminate_and_resume with _pid_identity_match before _is_claude_process.

These 4 tests are expected to FAIL (AttributeError / AssertionError) until
the implementation in fix commit 2 + 3 lands.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Test 1 — _pid_identity_match: real psutil, recycled PID detected
# ---------------------------------------------------------------------------
class TestPidIdentityMatchRecycledPid(unittest.TestCase):
    """Unit test: record real process start_time, then mock _get_pid_start_time
    to return a different start_time (simulating PID recycling after the original
    process died). _pid_identity_match must return False.
    """

    def setUp(self):
        import cozempic.guard as g
        g._CLAUDE_IDENTITY.clear()

    def tearDown(self):
        import cozempic.guard as g
        g._CLAUDE_IDENTITY.clear()

    def test_recycled_pid_start_time_mismatch_returns_false(self):
        import cozempic.guard as g

        # Use a real live PID (ourselves) and record its real start_time.
        # _record_claude_identity now validates the PID is Claude (MED-1 hardening);
        # mock that gate True for the test since pytest is not Claude.
        pid = os.getpid()
        session_id = "aaaa1111bbbb2222cccc3333dddd4444"
        with patch.object(g, "_is_claude_process", return_value=True):
            g._record_claude_identity(session_id, pid)

        # Confirm identity was recorded.
        self.assertIn(session_id, g._CLAUDE_IDENTITY)
        recorded_pid, recorded_start_time = g._CLAUDE_IDENTITY[session_id]
        self.assertEqual(recorded_pid, pid)

        # Simulate PID recycled: same PID but different start_time (10000s later).
        recycled_start_time = recorded_start_time + 10000.0
        with patch("cozempic.guard._get_pid_start_time", return_value=recycled_start_time):
            result = g._pid_identity_match(pid, session_id)

        self.assertFalse(
            result,
            "recycled PID with different start_time must return False",
        )

    def test_matching_pid_start_time_returns_true(self):
        import cozempic.guard as g

        pid = os.getpid()
        session_id = "aaaa1111bbbb2222cccc3333dddd5555"
        with patch.object(g, "_is_claude_process", return_value=True):
            g._record_claude_identity(session_id, pid)

        recorded_pid, recorded_start_time = g._CLAUDE_IDENTITY[session_id]

        # Same start_time: identity matches.
        with patch("cozempic.guard._get_pid_start_time", return_value=recorded_start_time):
            result = g._pid_identity_match(pid, session_id)

        self.assertTrue(result, "same PID + same start_time must return True")

    def test_different_pid_than_recorded_returns_false(self):
        import cozempic.guard as g

        pid = os.getpid()
        session_id = "aaaa1111bbbb2222cccc3333dddd6666"
        with patch.object(g, "_is_claude_process", return_value=True):
            g._record_claude_identity(session_id, pid)

        # Pass a completely different PID.
        with patch("cozempic.guard._get_pid_start_time", return_value=9999999.0):
            result = g._pid_identity_match(pid + 9999, session_id)

        self.assertFalse(result, "different PID than recorded must return False")


import os  # noqa: E402 — placed after class for import order clarity


# ---------------------------------------------------------------------------
# Test 2 — No resurrection when PID recycled (integration)
# ---------------------------------------------------------------------------
class TestNoResurrectionOnRecycledPid(unittest.TestCase):
    """Integration test: full _terminate_and_resume with a recycled PID.

    Scenario (architect's reproducer § "POST-fix flow"):
    - session_id recorded with (pid=89113, start_time=T0)
    - Claude dies, JSONL mtime refreshed by save_messages → fresh
    - OS recycles pid=89113 to an unrelated process with start_time=T1
    - _terminate_and_resume is called
    Expected: _pid_identity_match returns False → early return → NO SIGTERM,
    NO SIGKILL, NO sentinel write, NO _spawn_reload_watcher call.
    """

    def setUp(self):
        import cozempic.guard as g
        g._CLAUDE_IDENTITY.clear()

    def tearDown(self):
        import cozempic.guard as g
        g._CLAUDE_IDENTITY.clear()

    def test_recycled_pid_no_resurrection(self):
        import cozempic.guard as g

        session_id = "ddddddddeeee1111222233334444aaaa"
        fake_pid = 89113
        T0 = 1716220000.0
        T1 = 1716230000.0  # 10000s later — clearly different process

        # Record original identity.
        g._CLAUDE_IDENTITY[session_id] = (fake_pid, T0)

        with tempfile.TemporaryDirectory() as td:
            jsonl = Path(td) / "session.jsonl"
            jsonl.write_text("{}\n")  # fresh mtime — would fool mtime fallback

            # _pid_is_alive returns True (recycled process IS alive).
            # _get_pid_start_time returns T1 (different process).
            with patch("cozempic.guard._pid_is_alive", return_value=True), \
                 patch("cozempic.guard._get_pid_start_time", return_value=T1), \
                 patch("cozempic.guard._spawn_reload_watcher") as mock_watcher, \
                 patch("cozempic.guard.write_reload_sentinel") as mock_sentinel, \
                 patch("cozempic.guard.os.kill") as mock_kill, \
                 patch("cozempic.guard._detect_terminal_env", return_value="plain"):
                g._terminate_and_resume(
                    claude_pid=fake_pid,
                    project_dir=td,
                    session_id=session_id,
                    session_path=jsonl,
                )

        mock_watcher.assert_not_called()
        mock_sentinel.assert_not_called()
        mock_kill.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3 — _pid_identity_match degrades gracefully when psutil unavailable
# ---------------------------------------------------------------------------
class TestPidIdentityMatchDegrades(unittest.TestCase):
    """Unit test: psutil unavailable → _pid_identity_match returns True (fail-OPEN).

    Fail-OPEN rationale: without psutil we can't check start_time; we fall through
    to the existing _pid_is_alive + _is_claude_process layers (same risk as v1.8.16).
    Fail-CLOSED would break non-Claude-Code installs and CI without psutil.
    """

    def setUp(self):
        import cozempic.guard as g
        g._CLAUDE_IDENTITY.clear()

    def tearDown(self):
        import cozempic.guard as g
        g._CLAUDE_IDENTITY.clear()

    def test_psutil_unavailable_returns_true(self):
        import cozempic.guard as g

        pid = os.getpid()
        session_id = "aaaa1111bbbb2222cccc3333eeee7777"
        # Record an identity so the session_id IS in _CLAUDE_IDENTITY.
        g._CLAUDE_IDENTITY[session_id] = (pid, 1716220000.0)

        # _get_pid_start_time returns None when psutil raises ImportError.
        with patch("cozempic.guard._get_pid_start_time", return_value=None):
            result = g._pid_identity_match(pid, session_id)

        self.assertTrue(
            result,
            "psutil unavailable (None start_time) must return True (fail-OPEN)",
        )

    def test_no_session_id_returns_true(self):
        import cozempic.guard as g

        result = g._pid_identity_match(os.getpid(), session_id=None)
        self.assertTrue(result, "None session_id must return True (conservative allow)")

    def test_session_id_not_recorded_returns_true(self):
        import cozempic.guard as g

        # Session not in _CLAUDE_IDENTITY at all.
        result = g._pid_identity_match(os.getpid(), session_id="not-recorded-xxxx")
        self.assertTrue(result, "unrecorded session_id must return True (conservative allow)")


# ---------------------------------------------------------------------------
# Test 4 — _record_claude_identity called during start_guard
# ---------------------------------------------------------------------------
class TestRecordClaudeIdentityOnStartGuard(unittest.TestCase):
    """Verify _record_claude_identity(session_id, pid) is called once during
    start_guard after find_claude_pid() resolves, before the watchdog loop runs.
    """

    def setUp(self):
        import cozempic.guard as g
        g._CLAUDE_IDENTITY.clear()

    def tearDown(self):
        import cozempic.guard as g
        g._CLAUDE_IDENTITY.clear()

    def test_record_called_with_correct_args(self):
        import cozempic.guard as g

        fake_pid = 12345
        fake_session_id = "sssssssstttt1111222233334444ssss"

        with tempfile.TemporaryDirectory() as td:
            jsonl = Path(td) / "session.jsonl"
            jsonl.write_text("{}\n")

            fake_session = {
                "session_id": fake_session_id,
                "session_path": str(jsonl),
                "path": jsonl,
                "project_dir": td,
            }

            calls = []

            def fake_record(sid, pid):
                calls.append((sid, pid))

            # start_guard calls _resolve_session_by_id (not find_current_session)
            # when session_id is passed, then load_messages, detect_context_window,
            # record_session, etc. Patch everything between session resolution and
            # the watchdog loop so we reach line 503 (_record_claude_identity call).
            with patch("cozempic.guard._resolve_session_by_id", return_value=fake_session), \
                 patch("cozempic.guard.find_claude_pid", return_value=fake_pid), \
                 patch("cozempic.guard._record_claude_identity", side_effect=fake_record), \
                 patch("cozempic.guard.load_messages", return_value=[]), \
                 patch("cozempic.tokens.detect_context_window", return_value=200000), \
                 patch("cozempic.tokens.default_token_thresholds_4tier", return_value=(50000, 110000, 160000)), \
                 patch("cozempic.session.record_session"), \
                 patch("cozempic.guard._cleanup_stale_watchers"), \
                 patch("cozempic.guard.ping_install_if_new"), \
                 patch("cozempic.guard.maybe_auto_update"), \
                 patch("cozempic.guard.checkpoint_team"), \
                 patch("cozempic.guard.time.sleep", side_effect=RuntimeError("stop loop")):
                try:
                    g.start_guard(session_id=fake_session_id, claude_pid=fake_pid)
                except (RuntimeError, SystemExit):
                    pass

        self.assertEqual(len(calls), 1, "_record_claude_identity must be called exactly once")
        self.assertEqual(calls[0], (fake_session_id, fake_pid))


# ---------------------------------------------------------------------------
# Test 5 — Linux /proc backend (skipUnless Linux)
# ---------------------------------------------------------------------------
@unittest.skipUnless(sys.platform.startswith("linux"), "Linux /proc only")
class TestLinuxStdlibStartTime(unittest.TestCase):
    """Verify _get_pid_start_time_linux returns a plausible float for self PID."""

    def test_returns_float_for_self(self):
        import cozempic.guard as g

        result = g._get_pid_start_time_linux(os.getpid())
        self.assertIsInstance(result, float, "should return float on Linux")
        self.assertGreater(result, 0.0, "epoch timestamp must be positive")

    def test_result_is_before_current_time(self):
        import cozempic.guard as g

        now = time.time()
        result = g._get_pid_start_time_linux(os.getpid())
        self.assertIsNotNone(result)
        # Process can't have started in the future.
        self.assertLess(result, now + 1.0)
        # Process started within the last 24h (generous bound for CI).
        self.assertGreater(result, now - 86400)

    def test_nonexistent_pid_returns_none(self):
        import cozempic.guard as g

        # PID 0 is the idle/swapper process and its stat is not accessible.
        result = g._get_pid_start_time_linux(0)
        self.assertIsNone(result, "invalid/unreachable PID should return None")

    def test_same_pid_repeated_call_stable(self):
        import cozempic.guard as g

        pid = os.getpid()
        r1 = g._get_pid_start_time_linux(pid)
        r2 = g._get_pid_start_time_linux(pid)
        self.assertIsNotNone(r1)
        self.assertIsNotNone(r2)
        # deterministic int arithmetic → exact equality
        self.assertEqual(r1, r2, "/proc start_time must be stable across repeated reads")


# ---------------------------------------------------------------------------
# Test 6 — macOS ps lstart= backend (skipUnless Darwin)
# ---------------------------------------------------------------------------
@unittest.skipUnless(sys.platform == "darwin", "macOS ps lstart= only")
class TestMacosStdlibStartTime(unittest.TestCase):
    """Verify _get_pid_start_time_macos returns a plausible float for self PID."""

    def test_returns_float_for_self(self):
        import cozempic.guard as g

        result = g._get_pid_start_time_macos(os.getpid())
        self.assertIsInstance(result, float, "should return float on macOS")
        self.assertGreater(result, 0.0, "epoch timestamp must be positive")

    def test_result_is_before_current_time(self):
        import cozempic.guard as g

        now = time.time()
        result = g._get_pid_start_time_macos(os.getpid())
        self.assertIsNotNone(result)
        self.assertLess(result, now + 1.0)
        # Allow up to 24h of process age for CI runners.
        self.assertGreater(result, now - 86400)

    def test_nonexistent_pid_returns_none(self):
        import cozempic.guard as g

        # Use a definitively-dead PID (spawn + reap).
        proc = subprocess.Popen([sys.executable, "-c", ""])
        proc.wait()
        dead_pid = proc.pid
        result = g._get_pid_start_time_macos(dead_pid)
        self.assertIsNone(result, "dead PID should return None")

    def test_same_pid_repeated_call_stable(self):
        import cozempic.guard as g

        pid = os.getpid()
        r1 = g._get_pid_start_time_macos(pid)
        r2 = g._get_pid_start_time_macos(pid)
        self.assertIsNotNone(r1)
        self.assertIsNotNone(r2)
        # lstart is a fixed epoch second → exact equality on repeated calls.
        self.assertEqual(r1, r2, "ps lstart must be stable across repeated reads")

    def test_identity_match_uses_macos_backend(self):
        """End-to-end: _pid_identity_match uses the macOS backend when no psutil."""
        import cozempic.guard as g

        pid = os.getpid()
        session_id = "macostest1111222233334444aaaabbbb"
        g._CLAUDE_IDENTITY.clear()

        # Record using the real macOS backend (no psutil mock needed).
        with patch("cozempic.guard._is_claude_process", return_value=True):
            g._record_claude_identity(session_id, pid)

        self.assertIn(session_id, g._CLAUDE_IDENTITY,
                      "identity should be recorded via macOS ps backend")

        # Verify the match works without touching psutil at all.
        with patch("cozempic.guard._get_pid_start_time_psutil",
                   side_effect=AssertionError("psutil must not be called")):
            result = g._pid_identity_match(pid, session_id)

        self.assertTrue(result, "macOS backend identity match must return True for same PID")
        g._CLAUDE_IDENTITY.clear()


# ---------------------------------------------------------------------------
# Test 7 — Full stdlib fallthrough (all backends fail → None → fail-OPEN)
# ---------------------------------------------------------------------------
class TestStdlibFallthrough(unittest.TestCase):
    """Cross-platform: when all three backends return None, _get_pid_start_time
    returns None and _pid_identity_match fails-OPEN (returns True).
    """

    def setUp(self):
        import cozempic.guard as g
        g._CLAUDE_IDENTITY.clear()

    def tearDown(self):
        import cozempic.guard as g
        g._CLAUDE_IDENTITY.clear()

    def test_all_backends_fail_returns_none(self):
        import cozempic.guard as g

        with patch("cozempic.guard._get_pid_start_time_linux", return_value=None), \
             patch("cozempic.guard._get_pid_start_time_macos", return_value=None), \
             patch("cozempic.guard._get_pid_start_time_psutil", return_value=None):
            result = g._get_pid_start_time(os.getpid())

        self.assertIsNone(result, "all backends failing must return None")

    def test_all_backends_fail_pid_match_fails_open(self):
        import cozempic.guard as g

        pid = os.getpid()
        session_id = "fallthrough111122223333444455556666"
        # Pre-load identity with a known start_time.
        g._CLAUDE_IDENTITY[session_id] = (pid, 1716220000.0)

        with patch("cozempic.guard._get_pid_start_time_linux", return_value=None), \
             patch("cozempic.guard._get_pid_start_time_macos", return_value=None), \
             patch("cozempic.guard._get_pid_start_time_psutil", return_value=None):
            result = g._pid_identity_match(pid, session_id)

        self.assertTrue(result, "all backends failing must return True (fail-OPEN)")

    def test_linux_backend_tried_first_on_linux(self):
        """On Linux, the Linux backend is tried before macOS and psutil."""
        import cozempic.guard as g
        import platform

        if platform.system() != "Linux":
            self.skipTest("Linux dispatch only")

        called = []
        def fake_linux(pid):
            called.append("linux")
            return 1234567890.0

        with patch("cozempic.guard._get_pid_start_time_linux", side_effect=fake_linux), \
             patch("cozempic.guard._get_pid_start_time_macos",
                   side_effect=AssertionError("macOS must not be called on Linux")), \
             patch("cozempic.guard._get_pid_start_time_psutil",
                   side_effect=AssertionError("psutil must not be called when Linux works")):
            result = g._get_pid_start_time(os.getpid())

        self.assertEqual(result, 1234567890.0)
        self.assertIn("linux", called)

    def test_macos_backend_tried_first_on_darwin(self):
        """On Darwin, the macOS backend is tried before psutil."""
        import cozempic.guard as g
        import platform

        if platform.system() != "Darwin":
            self.skipTest("Darwin dispatch only")

        called = []
        def fake_macos(pid):
            called.append("macos")
            return 1234567890.0

        with patch("cozempic.guard._get_pid_start_time_macos", side_effect=fake_macos), \
             patch("cozempic.guard._get_pid_start_time_psutil",
                   side_effect=AssertionError("psutil must not be called when macOS works")):
            result = g._get_pid_start_time(os.getpid())

        self.assertEqual(result, 1234567890.0)
        self.assertIn("macos", called)

    def test_psutil_fallback_when_platform_backend_fails(self):
        """When the platform-native backend returns None, psutil is tried."""
        import cozempic.guard as g
        import platform

        _sys = platform.system()
        psutil_called = []

        def fake_psutil(pid):
            psutil_called.append(pid)
            return 9999999999.0

        if _sys == "Linux":
            ctx = patch("cozempic.guard._get_pid_start_time_linux", return_value=None)
        elif _sys == "Darwin":
            ctx = patch("cozempic.guard._get_pid_start_time_macos", return_value=None)
        else:
            self.skipTest("Only Linux/Darwin dispatch tested here")

        with ctx, patch("cozempic.guard._get_pid_start_time_psutil", side_effect=fake_psutil):
            result = g._get_pid_start_time(os.getpid())

        self.assertEqual(result, 9999999999.0)
        self.assertEqual(len(psutil_called), 1, "psutil must be called as fallback")
