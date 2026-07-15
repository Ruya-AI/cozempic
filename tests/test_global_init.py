"""Tests for global auto-init + uninstall + opt-out paths."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TestGlobalAutoInit(unittest.TestCase):
    def _stub_marker(self, tmpdir):
        return Path(tmpdir) / ".cozempic_global_initialized"

    def _stub_home_claude(self, tmpdir):
        d = Path(tmpdir) / ".claude"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_stale_global_hook(self, claude_dir):
        settings = claude_dir / "settings.json"
        settings.write_text(json.dumps({
            "hooks": {
                "SessionStart": [{
                    "matcher": "",
                    "hooks": [{
                        "type": "command",
                        "command": "cozempic guard --daemon # cozempic-hook-schema=v10",
                    }],
                }],
            },
        }))

    def test_skipped_when_env_set(self):
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            self._stub_home_claude(tmp)
            with mock.patch.dict(os.environ, {"COZEMPIC_NO_GLOBAL_INIT": "1"}):
                with mock.patch.object(cli, "_GLOBAL_INIT_MARKER", self._stub_marker(tmp)):
                    with mock.patch.object(cli.Path, "home", return_value=Path(tmp)):
                        cli._maybe_global_init(["list"])
                        # Marker must NOT have been touched
                        self.assertFalse(self._stub_marker(tmp).exists())

    def test_skipped_when_marker_exists(self):
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            self._stub_home_claude(tmp)
            marker = self._stub_marker(tmp)
            marker.touch()
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("COZEMPIC_NO_GLOBAL_INIT", None)
                with mock.patch.object(cli, "_GLOBAL_INIT_MARKER", marker):
                    with mock.patch.object(cli.Path, "home", return_value=Path(tmp)):
                        # Should bail out before calling run_init
                        with mock.patch.object(cli, "run_init") as ri:
                            cli._maybe_global_init(["list"])
                            ri.assert_not_called()

    def test_refreshes_stale_hooks_when_marker_exists(self):
        """A prior global install must refresh after the package schema changes."""
        from cozempic import cli
        from cozempic.init import HOOK_SCHEMA_MARKER

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            home_claude = self._stub_home_claude(tmp)
            self._write_stale_global_hook(home_claude)
            marker = self._stub_marker(tmp)
            marker.touch()

            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("COZEMPIC_NO_GLOBAL_INIT", None)
                with mock.patch.object(cli, "_GLOBAL_INIT_MARKER", marker):
                    with mock.patch.object(cli.Path, "home", return_value=home):
                        with mock.patch.object(cli.sys.stdin, "isatty", return_value=True):
                            with mock.patch.object(cli.sys.stderr, "isatty", return_value=True):
                                with mock.patch("builtins.input") as prompt:
                                    cli._maybe_global_init(["list"])

            prompt.assert_not_called()
            settings = json.loads((home_claude / "settings.json").read_text())
            command = settings["hooks"]["SessionStart"][0]["hooks"][0]["command"]
            self.assertIn(HOOK_SCHEMA_MARKER, command)

    def test_skipped_when_no_claude_dir(self):
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            # No ~/.claude/ created
            os.environ.pop("COZEMPIC_NO_GLOBAL_INIT", None)
            marker = self._stub_marker(tmp)
            with mock.patch.object(cli, "_GLOBAL_INIT_MARKER", marker):
                with mock.patch.object(cli.Path, "home", return_value=Path(tmp)):
                    with mock.patch.object(cli, "run_init") as ri:
                        cli._maybe_global_init(["list"])
                        ri.assert_not_called()
                        self.assertFalse(marker.exists())

    def test_runs_when_unconfigured_non_interactive(self):
        """Non-TTY (CI / Claude subprocess): silent auto-install."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            self._stub_home_claude(tmp)
            os.environ.pop("COZEMPIC_NO_GLOBAL_INIT", None)
            marker = self._stub_marker(tmp)
            with mock.patch.object(cli, "_GLOBAL_INIT_MARKER", marker):
                with mock.patch.object(cli.Path, "home", return_value=Path(tmp)):
                    # Force non-interactive mode (default for tests anyway)
                    with mock.patch.object(cli.sys.stdin, "isatty", return_value=False):
                        cli._maybe_global_init(["list"])
                    settings = Path(tmp) / ".claude" / "settings.json"
                    self.assertTrue(settings.exists())
                    data = json.loads(settings.read_text())
                    self.assertIn("hooks", data)
                    self.assertIn("SessionStart", data["hooks"])
                    self.assertTrue(marker.exists())

    def test_interactive_yes_installs(self):
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            self._stub_home_claude(tmp)
            os.environ.pop("COZEMPIC_NO_GLOBAL_INIT", None)
            marker = self._stub_marker(tmp)
            with mock.patch.object(cli, "_GLOBAL_INIT_MARKER", marker):
                with mock.patch.object(cli.Path, "home", return_value=Path(tmp)):
                    with mock.patch.object(cli.sys.stdin, "isatty", return_value=True):
                        with mock.patch.object(cli.sys.stderr, "isatty", return_value=True):
                            with mock.patch("builtins.input", return_value="y"):
                                cli._maybe_global_init(["list"])
                    self.assertTrue((Path(tmp) / ".claude" / "settings.json").exists())
                    self.assertTrue(marker.exists())

    def test_interactive_no_skips_install_but_marks(self):
        """User declined — don't install, but DO set marker so we never ask again."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            self._stub_home_claude(tmp)
            os.environ.pop("COZEMPIC_NO_GLOBAL_INIT", None)
            marker = self._stub_marker(tmp)
            with mock.patch.object(cli, "_GLOBAL_INIT_MARKER", marker):
                with mock.patch.object(cli.Path, "home", return_value=Path(tmp)):
                    with mock.patch.object(cli.sys.stdin, "isatty", return_value=True):
                        with mock.patch.object(cli.sys.stderr, "isatty", return_value=True):
                            with mock.patch("builtins.input", return_value="n"):
                                cli._maybe_global_init(["list"])
                    self.assertFalse((Path(tmp) / ".claude" / "settings.json").exists())
                    self.assertTrue(marker.exists())  # marker set so we don't re-prompt

    def test_interactive_ctrl_c_treated_as_no(self):
        """KeyboardInterrupt at the prompt is treated as decline (no install, marker set)."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            self._stub_home_claude(tmp)
            os.environ.pop("COZEMPIC_NO_GLOBAL_INIT", None)
            marker = self._stub_marker(tmp)
            with mock.patch.object(cli, "_GLOBAL_INIT_MARKER", marker):
                with mock.patch.object(cli.Path, "home", return_value=Path(tmp)):
                    with mock.patch.object(cli.sys.stdin, "isatty", return_value=True):
                        with mock.patch.object(cli.sys.stderr, "isatty", return_value=True):
                            with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
                                cli._maybe_global_init(["list"])
                    self.assertFalse((Path(tmp) / ".claude" / "settings.json").exists())
                    self.assertTrue(marker.exists())

    def test_version_check_triggers_init(self):
        """`cozempic --version` (no subcommand) should trigger global init."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            self._stub_home_claude(tmp)
            os.environ.pop("COZEMPIC_NO_GLOBAL_INIT", None)
            marker = self._stub_marker(tmp)
            with mock.patch.object(cli, "_GLOBAL_INIT_MARKER", marker):
                with mock.patch.object(cli.Path, "home", return_value=Path(tmp)):
                    with mock.patch.object(cli.sys.stdin, "isatty", return_value=False):
                        cli._maybe_global_init(["--version"])
                    self.assertTrue((Path(tmp) / ".claude" / "settings.json").exists())

    def test_help_does_not_trigger_init(self):
        """`cozempic --help` / `-h` must NOT trigger init (purely informational)."""
        from cozempic import cli
        for help_flag in ("--help", "-h"):
            with self.subTest(flag=help_flag):
                with tempfile.TemporaryDirectory() as tmp:
                    self._stub_home_claude(tmp)
                    os.environ.pop("COZEMPIC_NO_GLOBAL_INIT", None)
                    marker = self._stub_marker(tmp)
                    with mock.patch.object(cli, "_GLOBAL_INIT_MARKER", marker):
                        with mock.patch.object(cli.Path, "home", return_value=Path(tmp)):
                            cli._maybe_global_init([help_flag])
                    self.assertFalse((Path(tmp) / ".claude" / "settings.json").exists())
                    self.assertFalse(marker.exists())


class TestUninstallHooks(unittest.TestCase):
    def test_removes_cozempic_hooks_only(self):
        from cozempic.init import wire_hooks, uninstall_hooks
        with tempfile.TemporaryDirectory() as tmp:
            # Set up a settings.json with a non-cozempic hook + cozempic hooks
            (Path(tmp) / ".claude").mkdir()
            settings_path = Path(tmp) / ".claude" / "settings.json"
            settings_path.write_text(json.dumps({
                "hooks": {
                    "SessionStart": [{
                        "matcher": "",
                        "hooks": [{"type": "command", "command": "echo 'user-hook'"}],
                    }],
                }
            }))
            # Wire cozempic on top
            wire_hooks(tmp)
            after_wire = json.loads(settings_path.read_text())
            self.assertGreater(len(after_wire["hooks"]["SessionStart"]), 1)

            # Uninstall — user hook stays, cozempic hooks go
            result = uninstall_hooks(tmp)
            self.assertGreater(len(result["removed"]), 0)
            after = json.loads(settings_path.read_text())
            ss = after["hooks"]["SessionStart"]
            self.assertEqual(len(ss), 1)
            self.assertIn("user-hook", ss[0]["hooks"][0]["command"])

    def test_idempotent_on_missing_settings(self):
        from cozempic.init import uninstall_hooks
        with tempfile.TemporaryDirectory() as tmp:
            result = uninstall_hooks(tmp)
            self.assertEqual(result["removed"], [])

    def test_mixed_entry_preserves_user_commands(self):
        """An entry containing BOTH cozempic and user commands in its `hooks`
        list must only lose the cozempic commands. Regression for 'uninstall
        nukes user commands in shared entries' (bug 4.1)."""
        from cozempic.init import uninstall_hooks, HOOK_SCHEMA_MARKER
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".claude").mkdir()
            settings_path = Path(tmp) / ".claude" / "settings.json"
            settings_path.write_text(json.dumps({
                "hooks": {
                    "SessionStart": [{
                        "matcher": "",
                        "hooks": [
                            # User command (no cozempic, no schema marker)
                            {"type": "command", "command": "echo 'my-hook'"},
                            # Cozempic command with current schema marker
                            {"type": "command", "command": f"cozempic guard --daemon # {HOOK_SCHEMA_MARKER}"},
                        ],
                    }],
                }
            }))
            result = uninstall_hooks(tmp)
            self.assertTrue(result["removed"], "expected at least one removal")
            after = json.loads(settings_path.read_text())
            remaining = after["hooks"]["SessionStart"][0]["hooks"]
            self.assertEqual(len(remaining), 1)
            self.assertEqual(remaining[0]["command"], "echo 'my-hook'")

    def test_malformed_json_does_not_crash(self):
        from cozempic.init import uninstall_hooks
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".claude").mkdir()
            settings_path = Path(tmp) / ".claude" / "settings.json"
            settings_path.write_text("{not valid json")
            result = uninstall_hooks(tmp)
            self.assertEqual(result["removed"], [])
            self.assertIn("error", result)


class TestStaleHookRefresh(unittest.TestCase):
    def test_stale_cozempic_hook_gets_refreshed(self):
        """A settings.json with a pre-schema cozempic hook (old wrapper command,
        no schema marker) must be upgraded to the current schema on wire_hooks.
        Uses the realistic pre-v2 command pattern which had `python3 -m cozempic`
        as the fallback wrapper — that's how we detect 'ours' pre-schema.
        """
        from cozempic.init import wire_hooks, HOOK_SCHEMA_MARKER
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".claude").mkdir()
            settings_path = Path(tmp) / ".claude" / "settings.json"
            settings_path.write_text(json.dumps({
                "hooks": {
                    "SessionStart": [{
                        "matcher": "",
                        "hooks": [
                            # Stale cozempic hook — the pre-v2 canonical pattern
                            # (python3 -m cozempic fallback, no schema marker).
                            {"type": "command", "command": "{ cozempic guard --daemon 2>/dev/null || python3 -m cozempic guard --daemon 2>/dev/null; } || true"},
                        ],
                    }],
                }
            }))
            result = wire_hooks(tmp)
            # The event should be marked "updated" (not skipped, not added)
            self.assertIn("SessionStart[]", result["updated"])
            after = json.loads(settings_path.read_text())
            cmd = after["hooks"]["SessionStart"][0]["hooks"][0]["command"]
            self.assertIn(HOOK_SCHEMA_MARKER, cmd, "schema marker must be present after refresh")

    def test_user_command_with_cozempic_substring_is_not_treated_as_ours(self):
        """A user-authored chain command like `cozempic checkpoint && backup.sh`
        must NOT be classified as cozempic-installed (bug 1.4). Previously the
        substring-match fallback would false-match and `uninstall_hooks` would
        delete the user's backup script.

        Also covers the secondary regression (bug 7.1 from round-3 audit): a
        user chain `pre; python3 -m cozempic X; post` must NOT match either —
        requires the FULL canonical wrapper shape (brace-open + python
        fallback), not just one or the other.
        """
        from cozempic.init import _is_cozempic_command
        # Chain with cozempic + user step — no wrapper, no match
        self.assertFalse(_is_cozempic_command("cozempic checkpoint && my-backup.sh"))
        # User hook referencing cozempic in a string — no match
        self.assertFalse(_is_cozempic_command('echo "cozempic notes" > /tmp/out'))
        # User chain that includes `python3 -m cozempic` — still no match
        # (was a false positive in the prior fix). Lacks `{ cozempic ` opener.
        self.assertFalse(_is_cozempic_command(
            "pre-step && python3 -m cozempic checkpoint && post-step"
        ))
        self.assertFalse(_is_cozempic_command(
            "my-script.sh; python3 -m cozempic digest flush"
        ))
        # A real canonical hook with the python fallback wrapper — SHOULD match
        self.assertTrue(_is_cozempic_command(
            "{ cozempic checkpoint 2>/dev/null || python3 -m cozempic checkpoint 2>/dev/null; } || true"
        ))
        # A v4+ hook with the schema marker — SHOULD match
        self.assertTrue(_is_cozempic_command(
            "echo 'hi' # cozempic-hook-schema=v4"
        ))

    def test_refresh_preserves_user_command_in_mixed_entry(self):
        """Regression for bug 1.3 + 1.4: wire_hooks refresh must preserve
        user-authored commands in a mixed entry AND keep them at their
        original position in the hooks list."""
        from cozempic.init import wire_hooks
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".claude").mkdir()
            settings_path = Path(tmp) / ".claude" / "settings.json"
            settings_path.write_text(json.dumps({
                "hooks": {
                    "SessionStart": [{
                        "matcher": "",
                        "hooks": [
                            # User hook FIRST (order matters for some setups)
                            {"type": "command", "command": "export MY_SETUP=1"},
                            # Stale cozempic hook
                            {"type": "command", "command": "{ cozempic guard --daemon 2>/dev/null || python3 -m cozempic guard --daemon 2>/dev/null; } || true"},
                            # Another user hook AFTER
                            {"type": "command", "command": "echo 'session started' >> /tmp/user.log"},
                        ],
                    }],
                }
            }))
            wire_hooks(tmp)
            after = json.loads(settings_path.read_text())
            cmds = [h["command"] for h in after["hooks"]["SessionStart"][0]["hooks"]]
            # User commands both present
            self.assertIn("export MY_SETUP=1", cmds)
            self.assertIn("echo 'session started' >> /tmp/user.log", cmds)
            # First cmd is the first user cmd (order preserved)
            self.assertEqual(cmds[0], "export MY_SETUP=1")
            # Last cmd is the second user cmd (still at end)
            self.assertEqual(cmds[-1], "echo 'session started' >> /tmp/user.log")

    def test_current_schema_hook_is_skipped(self):
        """wire_hooks on an already-current settings.json must not touch it."""
        from cozempic.init import wire_hooks, run_init
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".claude").mkdir()
            # Initial init
            run_init(tmp, skip_slash=True)
            settings_path = Path(tmp) / ".claude" / "settings.json"
            first_content = settings_path.read_text()
            # Second init — everything should be skipped
            result = wire_hooks(tmp)
            self.assertEqual(result["added"], [])
            self.assertEqual(result["updated"], [])
            self.assertGreater(len(result["skipped"]), 0)
            # File unchanged
            self.assertEqual(first_content, settings_path.read_text())


class TestPIDReuseCheck(unittest.TestCase):
    def test_is_cozempic_guard_process_false_for_other_pid(self):
        """Random PID (e.g. init, 1) should not be identified as a cozempic guard."""
        from cozempic.guard import _is_cozempic_guard_process
        # PID 1 exists on every Unix but isn't cozempic
        self.assertFalse(_is_cozempic_guard_process(1))

    def test_is_cozempic_guard_process_false_for_nonexistent(self):
        from cozempic.guard import _is_cozempic_guard_process
        # Almost certainly not a real PID
        self.assertFalse(_is_cozempic_guard_process(999999))


class TestAtomicWrite(unittest.TestCase):
    def test_save_settings_is_atomic_on_json_error(self):
        """If json serialization raises mid-write, the existing settings.json
        must remain intact (atomic write via tempfile + os.replace)."""
        from cozempic.init import _save_settings
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".claude").mkdir()
            path = Path(tmp) / ".claude" / "settings.json"
            path.write_text('{"existing": "config"}')
            original = path.read_text()

            # Try to write something non-serializable — should fail without
            # touching the original file.
            class NotJsonable:
                pass
            try:
                _save_settings(path, {"bad": NotJsonable()})
            except TypeError:
                pass
            else:
                self.fail("expected TypeError from non-serializable value")

            # Original file must still be intact
            self.assertEqual(path.read_text(), original)
            # No tempfile left behind
            temps = list(Path(tmp, ".claude").glob(".cozempic-settings-*"))
            self.assertEqual(temps, [], f"tempfile not cleaned up: {temps}")

    def test_save_settings_writes_successfully(self):
        """Normal save path must produce the expected content at the target path."""
        from cozempic.init import _save_settings
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".claude" / "settings.json"
            _save_settings(path, {"hooks": {"SessionStart": []}})
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded, {"hooks": {"SessionStart": []}})

    def test_save_settings_preserves_permissions(self):
        """mkstemp creates 0o600; we must restore the original file's mode."""
        from cozempic.init import _save_settings
        import stat
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".claude" / "settings.json"
            (Path(tmp) / ".claude").mkdir()
            path.write_text("{}")
            os.chmod(path, 0o644)
            _save_settings(path, {"updated": True})
            actual = stat.S_IMODE(os.stat(path).st_mode)
            self.assertEqual(actual, 0o644, f"permissions changed to {oct(actual)}")


class TestWireHooksGracefulParseFailure(unittest.TestCase):
    def test_malformed_settings_returns_error_not_crash(self):
        """wire_hooks on malformed JSON returns error field instead of raising."""
        from cozempic.init import wire_hooks
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".claude").mkdir()
            (Path(tmp) / ".claude" / "settings.json").write_text("{not valid")
            result = wire_hooks(tmp)
            self.assertEqual(result["added"], [])
            self.assertEqual(result["updated"], [])
            self.assertIn("error", result)
            self.assertIn("could not parse", result["error"])


class TestDoSMarkerOnFailure(unittest.TestCase):
    def test_marker_touched_on_run_init_failure(self):
        """If run_init raises, the marker must still be touched to prevent
        a DoS loop of re-attempts. (Bug 6.1)"""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            # Fake home with .claude dir
            (Path(tmp) / ".claude").mkdir()
            os.environ.pop("COZEMPIC_NO_GLOBAL_INIT", None)
            marker = Path(tmp) / ".cozempic_global_initialized"
            with mock.patch.object(cli, "_GLOBAL_INIT_MARKER", marker):
                with mock.patch.object(cli.Path, "home", return_value=Path(tmp)):
                    with mock.patch.object(cli.sys.stdin, "isatty", return_value=False):
                        with mock.patch.object(cli, "run_init", side_effect=OSError("boom")):
                            cli._maybe_global_init(["list"])
            self.assertTrue(marker.exists(), "marker must be touched even on failure")


if __name__ == "__main__":
    unittest.main()
