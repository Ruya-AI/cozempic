"""Tests for find_linux_terminal_launch_command (#183).

The Linux auto-resume path in _spawn_watcher (cli.py) and
_spawn_reload_watcher (guard.py) only ever checked gnome-terminal and xterm,
so anyone on Terminator, KDE, XFCE, or most tiling-WM terminal emulators got
a silent no-op instead of a resumed session. This also covers the follow-on
fix in _spawn_watcher: when no terminal is found, it must print manual
resume instructions and return without spawning a background watcher that
can never report back once the original Claude process has exited.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _which_only(*allowed: str):
    """Return a shutil.which stand-in that reports only `allowed` binaries as installed."""
    def _which(binary, *a, **k):
        return f"/usr/bin/{binary}" if binary in allowed else None
    return _which


class TestFindLinuxTerminalLaunchCommand(unittest.TestCase):
    def test_returns_none_when_no_terminal_is_installed(self):
        from cozempic.helpers import find_linux_terminal_launch_command
        with patch("shutil.which", side_effect=_which_only()):
            self.assertIsNone(find_linux_terminal_launch_command("echo hi"))

    def test_prefers_gnome_terminal_when_multiple_are_installed(self):
        from cozempic.helpers import find_linux_terminal_launch_command
        with patch("shutil.which", side_effect=_which_only("xterm", "gnome-terminal", "konsole")):
            cmd = find_linux_terminal_launch_command("echo hi")
        self.assertIn("gnome-terminal", cmd)
        self.assertNotIn("konsole", cmd)

    def test_falls_through_to_terminator_when_gnome_terminal_is_missing(self):
        from cozempic.helpers import find_linux_terminal_launch_command
        with patch("shutil.which", side_effect=_which_only("terminator")):
            cmd = find_linux_terminal_launch_command("echo hi")
        self.assertIn("terminator", cmd)
        self.assertIn("-x bash -c", cmd)

    def test_falls_through_to_konsole_xfce4_terminal_tilix_alacritty_kitty(self):
        from cozempic.helpers import find_linux_terminal_launch_command
        for binary in ("konsole", "xfce4-terminal", "tilix", "alacritty", "kitty"):
            with patch("shutil.which", side_effect=_which_only(binary)):
                cmd = find_linux_terminal_launch_command("echo hi")
            self.assertIn(binary, cmd, f"expected {binary} in launch command: {cmd!r}")

    def test_xterm_is_the_last_resort(self):
        from cozempic.helpers import find_linux_terminal_launch_command
        with patch("shutil.which", side_effect=_which_only("xterm")):
            cmd = find_linux_terminal_launch_command("echo hi")
        self.assertIn("xterm -e", cmd)

    def test_inner_command_is_shell_quoted_exactly_once(self):
        from cozempic.helpers import find_linux_terminal_launch_command
        with patch("shutil.which", side_effect=_which_only("gnome-terminal")):
            cmd = find_linux_terminal_launch_command("echo 'it'\"'\"'s a test'")
        # Should produce one shell-quoted argument, not a syntax-breaking mix.
        self.assertIn("gnome-terminal -- bash -c ", cmd)
        self.assertEqual(cmd.count("bash -c "), 1)


class TestSpawnWatcherLinuxTerminalFallback(unittest.TestCase):
    """cli.py's _spawn_watcher: no terminal found → print instructions, no Popen."""

    def _run_spawn_watcher(self, which_side_effect):
        from cozempic import cli
        with patch("cozempic.cli.platform.system", return_value="Linux"), \
             patch("cozempic.cli.is_ssh_session", return_value=False), \
             patch("cozempic.guard._detect_claude_flags", return_value=""), \
             patch("shutil.which", side_effect=which_side_effect), \
             patch("cozempic.cli.subprocess.Popen") as mock_popen, \
             patch("builtins.print") as mock_print:
            cli._spawn_watcher(claude_pid=4321, project_dir="/tmp/proj", session_id="abc123")
        return mock_popen, mock_print

    def test_no_terminal_found_prints_manual_instructions_and_does_not_spawn(self):
        mock_popen, mock_print = self._run_spawn_watcher(_which_only())
        mock_popen.assert_not_called()
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("claude", printed)
        self.assertIn("--resume", printed)

    def test_terminal_found_still_spawns_the_watcher(self):
        mock_popen, _ = self._run_spawn_watcher(_which_only("gnome-terminal"))
        mock_popen.assert_called_once()
        argv = mock_popen.call_args[0][0]
        script = argv[-1]
        self.assertIn("gnome-terminal", script)


if __name__ == "__main__":
    unittest.main()
