"""Tests for the Windows guard-daemon lifecycle fixes.

Context (2026-08-03, Windows 11): guard daemons accumulated 10-30 deep in the
task manager because the entire daemon lifecycle layer was POSIX-only:

1. ``session.find_claude_pid`` walked ancestry with ``ps -o ppid=,comm=`` —
   Git Bash's ``ps`` has no ``-o``, so the walk ALWAYS returned None on
   Windows, disarming the Claude-exit watchdog (the only exit path for a
   normal-size session). Daemons outlived Claude Code indefinitely.
2. ``os.kill(pid, 0)`` is not a liveness probe on Windows (signal 0 is
   CTRL_C_EVENT → GenerateConsoleCtrlEvent). It raised OSError for LIVE
   detached pythonw daemons, so ``spawn_lock._is_process_alive`` classified
   live peers' pidfiles as stale and any two SessionStart hooks firing more
   than the 5s fresh window apart spawned DUPLICATE daemons per session
   (reproduced live before the fix).

The fixes under test: a pure ancestry walk over a Toolhelp32 snapshot
(``_walk_up_to_claude`` — testable on any platform), the native OpenProcess
liveness probe (Windows-only integration tests), and the orphan backstop
env knob (``_read_orphan_exit_seconds``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import pytest

from cozempic.guard import _split_windows_cmdline
from cozempic.helpers import _pid_is_alive
from cozempic.session import _walk_up_to_claude, find_claude_pid
from cozempic.spawn_lock import _is_process_alive


class TestSplitWindowsCmdline(unittest.TestCase):
    """Pure-function tests for the quote-aware cmdline split (any platform)."""

    def test_unquoted_argv0(self):
        argv0, tokens = _split_windows_cmdline(
            r"C:\Python\pythonw.exe -m cozempic.cli guard --cwd C:\proj"
        )
        self.assertEqual(argv0, r"C:\Python\pythonw.exe")
        self.assertEqual(tokens[:3], ["-m", "cozempic.cli", "guard"])

    def test_quoted_argv0_with_spaces(self):
        argv0, tokens = _split_windows_cmdline(
            r'"C:\Program Files\Python\python.exe" -m cozempic.cli guard'
        )
        self.assertEqual(argv0, r"C:\Program Files\Python\python.exe")
        self.assertEqual(tokens, ["-m", "cozempic.cli", "guard"])

    def test_bare_quoted_binary(self):
        argv0, tokens = _split_windows_cmdline('"cozempic.exe" guard --daemon')
        self.assertEqual(argv0, "cozempic.exe")
        self.assertEqual(tokens, ["guard", "--daemon"])

    def test_unterminated_quote_degrades(self):
        argv0, tokens = _split_windows_cmdline('"C:\\broken')
        self.assertEqual(argv0, "C:\\broken")
        self.assertEqual(tokens, [])


class TestWalkUpToClaude(unittest.TestCase):
    """Pure-function tests for the snapshot ancestry walk (any platform)."""

    def test_finds_claude_ancestor(self):
        pmap = {
            100: (1, "wininit.exe"),
            200: (100, "Claude.exe"),
            300: (200, "bash.exe"),
            400: (300, "python.exe"),
        }
        self.assertEqual(_walk_up_to_claude(pmap, 400), 200)

    def test_finds_node_ancestor(self):
        pmap = {
            200: (1, "node.exe"),
            300: (200, "bash.exe"),
            400: (300, "python.exe"),
        }
        self.assertEqual(_walk_up_to_claude(pmap, 400), 200)

    def test_dead_ancestry_returns_none(self):
        # A detached daemon's parent has exited: ppid points outside the
        # snapshot. Must return None, never guess.
        pmap = {400: (399, "pythonw.exe")}
        self.assertIsNone(_walk_up_to_claude(pmap, 400))

    def test_no_claude_in_chain_returns_none(self):
        pmap = {
            100: (0, "wininit.exe"),
            300: (100, "bash.exe"),
            400: (300, "python.exe"),
        }
        self.assertIsNone(_walk_up_to_claude(pmap, 400))

    def test_self_parented_root_terminates(self):
        # System idle/root pseudo-processes are self-parented on Windows.
        pmap = {4: (4, "System"), 400: (4, "python.exe")}
        self.assertIsNone(_walk_up_to_claude(pmap, 400))

    def test_ppid_cycle_terminates(self):
        # Toolhelp parent PIDs can be stale after PID recycling and form a
        # cycle A->B->A. The walk must terminate, not spin.
        pmap = {400: (300, "python.exe"), 300: (400, "bash.exe")}
        self.assertIsNone(_walk_up_to_claude(pmap, 400))

    def test_max_hops_bounds_walk(self):
        # 15-deep chain with claude at the root — more hops than the cap.
        pmap = {i: (i + 1, "proc.exe") for i in range(100, 115)}
        pmap[115] = (1, "Claude.exe")
        self.assertIsNone(_walk_up_to_claude(pmap, 100, max_hops=10))
        self.assertEqual(_walk_up_to_claude(pmap, 100, max_hops=16), 115)

    def test_missing_start_pid_returns_none(self):
        self.assertIsNone(_walk_up_to_claude({}, 400))

    def test_matches_current_pid_name(self):
        # Parity with the POSIX walk: the START pid's own name is checked
        # too (a `claude` ancestor chain may begin at the probe process
        # when called from inside a node subprocess).
        pmap = {400: (1, "node.exe")}
        self.assertEqual(_walk_up_to_claude(pmap, 400), 400)


class TestOrphanExitKnob(unittest.TestCase):
    """Env parsing for COZEMPIC_GUARD_ORPHAN_EXIT_SECONDS (any platform)."""

    def _read(self, value):
        from cozempic.guard import _read_orphan_exit_seconds
        env = {} if value is None else {"COZEMPIC_GUARD_ORPHAN_EXIT_SECONDS": value}
        with patch.dict(os.environ, env, clear=False):
            if value is None:
                os.environ.pop("COZEMPIC_GUARD_ORPHAN_EXIT_SECONDS", None)
            return _read_orphan_exit_seconds()

    def test_default_when_unset(self):
        from cozempic.guard import _DEFAULT_ORPHAN_EXIT_SECONDS
        self.assertEqual(self._read(None), _DEFAULT_ORPHAN_EXIT_SECONDS)

    def test_zero_disables(self):
        self.assertEqual(self._read("0"), 0.0)

    def test_valid_override(self):
        self.assertEqual(self._read("600"), 600.0)

    def test_garbage_falls_back(self):
        from cozempic.guard import _DEFAULT_ORPHAN_EXIT_SECONDS
        for bad in ("banana", "nan", "inf", "-5", str(365 * 24 * 3600)):
            self.assertEqual(self._read(bad), _DEFAULT_ORPHAN_EXIT_SECONDS, bad)


@pytest.mark.skipif(os.name != "nt", reason="Windows-only liveness probe")
class TestWindowsLivenessProbe(unittest.TestCase):
    """Integration tests for the OpenProcess probe against real processes."""

    def _spawn_detached_windowless(self):
        """Reproduce the real topology that broke os.kill(pid, 0): a
        detached, console-less child in its own process group — exactly how
        start_guard_daemon spawns the guard."""
        return subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            ),
        )

    def test_live_detached_process_reads_alive(self):
        # THE regression: the old os.kill(pid, 0) probe raised OSError for
        # this exact topology, classifying a live daemon as dead — which
        # made DaemonSpawnClaim unlink its pidfile and spawn a duplicate.
        proc = self._spawn_detached_windowless()
        try:
            self.assertTrue(_pid_is_alive(proc.pid))
            self.assertTrue(_is_process_alive(proc.pid))
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def test_exited_process_reads_dead(self):
        proc = self._spawn_detached_windowless()
        proc.kill()
        proc.wait(timeout=10)
        # Popen keeps a handle open on the exited child — the probe must
        # see through that via GetExitCodeProcess, not just OpenProcess.
        self.assertFalse(_pid_is_alive(proc.pid))
        self.assertFalse(_is_process_alive(proc.pid))

    def test_never_existing_pid_reads_dead(self):
        # PIDs are multiples of 4 on Windows; 4_000_000+2 cannot exist.
        self.assertFalse(_pid_is_alive(4_000_002))
        self.assertFalse(_is_process_alive(4_000_002))

    def test_reload_lock_probe_agrees(self):
        from cozempic.reload_lock import _is_process_alive as _rl_alive
        proc = self._spawn_detached_windowless()
        try:
            self.assertTrue(_rl_alive(proc.pid))
        finally:
            proc.kill()
            proc.wait(timeout=10)
        self.assertFalse(_rl_alive(proc.pid))

    def test_kill0_probe_semantics(self):
        from cozempic.guard import _kill0_probe
        proc = self._spawn_detached_windowless()
        try:
            _kill0_probe(proc.pid)  # alive — must not raise
        finally:
            proc.kill()
            proc.wait(timeout=10)
        with self.assertRaises(ProcessLookupError):
            _kill0_probe(proc.pid)


@pytest.mark.skipif(os.name != "nt", reason="Windows-only cmdline identity check")
class TestWindowsGuardIdentity(unittest.TestCase):
    def test_own_cmdline_readable(self):
        from cozempic.helpers import _windows_process_cmdline
        cmdline = _windows_process_cmdline(os.getpid())
        self.assertIsNotNone(cmdline)
        self.assertIn("python", cmdline.lower())

    def test_non_guard_python_rejected(self):
        from cozempic.guard import _is_cozempic_guard_process
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            self.assertFalse(_is_cozempic_guard_process(proc.pid))
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def test_guard_shaped_cmdline_accepted(self):
        # A python process whose argv carries the discrete "cozempic.cli" and
        # "guard" tokens — the exact spawn shape of start_guard_daemon. The
        # extra argv words are inert for `-c` but visible to the checker,
        # mirroring what the POSIX ps-token check would also accept.
        from cozempic.guard import _is_cozempic_guard_process
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)",
             "cozempic.cli", "guard"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            self.assertTrue(_is_cozempic_guard_process(proc.pid))
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def test_dead_pid_rejected(self):
        from cozempic.guard import _is_cozempic_guard_process
        self.assertFalse(_is_cozempic_guard_process(4_000_002))


@pytest.mark.skipif(os.name != "nt", reason="Windows-only ancestry snapshot")
class TestWindowsProcessMap(unittest.TestCase):
    def test_snapshot_contains_self_and_parent(self):
        from cozempic.session import _windows_process_map
        pmap = _windows_process_map()
        me = pmap.get(os.getpid())
        self.assertIsNotNone(me)
        ppid, exe = me
        self.assertIn("python", exe.lower())
        # Our parent is live (it is running this test session).
        self.assertIn(ppid, pmap)

    def test_find_claude_pid_returns_int_or_none(self):
        # Smoke: must never raise on Windows (the old code silently broke
        # inside ps; the new code must degrade to None at worst).
        pid = find_claude_pid()
        self.assertTrue(pid is None or (isinstance(pid, int) and pid > 0))


if __name__ == "__main__":
    unittest.main()
