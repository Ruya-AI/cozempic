"""Tests for global auto-init + uninstall + opt-out paths."""
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TestGlobalAutoInit(unittest.TestCase):
    def setUp(self):
        # These legacy tests model the default profile; dedicated tests below
        # cover CLAUDE_CONFIG_DIR explicitly.
        self._config_patch = mock.patch(
            "cozempic.session.get_claude_dir",
            side_effect=lambda: Path.home() / ".claude",
        )
        self._config_patch.start()

    def tearDown(self):
        self._config_patch.stop()

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

    def _write_current_global_hook(self, claude_dir):
        from cozempic.init import HOOK_SCHEMA_MARKER

        settings = claude_dir / "settings.json"
        settings.write_text(json.dumps({
            "hooks": {
                "SessionStart": [{
                    "matcher": "",
                    "hooks": [{
                        "type": "command",
                        "command": f"cozempic guard --daemon # {HOOK_SCHEMA_MARKER}",
                    }],
                }],
            },
        }))

    def _write_stale_local_hook(self, claude_dir):
        settings = claude_dir / "settings.local.json"
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

    def test_legacy_default_marker_migrates_without_reinstalling(self):
        from cozempic import cli

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            claude_dir = home / ".claude"
            claude_dir.mkdir(parents=True)
            (home / ".cozempic_global_initialized").touch()
            with mock.patch.dict(os.environ, {}, clear=True), \
                    mock.patch("cozempic.init.Path.home", return_value=home), \
                    mock.patch("cozempic.session.get_claude_dir", return_value=claude_dir), \
                    mock.patch.object(cli, "run_init") as run_init:
                cli._maybe_global_init(["list"])
            run_init.assert_not_called()
            self.assertTrue((claude_dir / ".cozempic_global_initialized").exists())

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

    def test_marker_does_not_rewire_for_stale_local_hook(self):
        """Global init must ignore settings.local.json it does not manage."""
        from cozempic import cli

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            home_claude = self._stub_home_claude(tmp)
            self._write_stale_local_hook(home_claude)
            marker = self._stub_marker(tmp)
            marker.touch()

            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("COZEMPIC_NO_GLOBAL_INIT", None)
                with mock.patch.object(cli, "_GLOBAL_INIT_MARKER", marker):
                    with mock.patch.object(cli.Path, "home", return_value=home):
                        with mock.patch.object(cli, "run_init") as run_init:
                            cli._maybe_global_init(["list"])

            run_init.assert_not_called()
            self.assertFalse((home_claude / "settings.json").exists())

    def test_marker_with_malformed_global_settings_surfaces_error(self):
        """A marker must not turn unreadable managed settings into a decline."""
        from cozempic import cli

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            home_claude = self._stub_home_claude(tmp)
            (home_claude / "settings.json").write_text("{broken json")
            marker = self._stub_marker(tmp)
            marker.touch()

            with mock.patch.object(cli, "_GLOBAL_INIT_MARKER", marker):
                with mock.patch.object(cli.Path, "home", return_value=home):
                    with mock.patch("sys.stderr") as mock_stderr:
                        with mock.patch.object(cli, "run_init") as run_init:
                            cli._maybe_global_init(["list"])

            run_init.assert_not_called()
            output = "".join(
                call.args[0] for call in mock_stderr.write.call_args_list if call.args
            )
            self.assertIn("global init FAILED", output)
            self.assertIn("could not parse", output)

    def test_stale_global_hook_establishes_consent_without_prompt(self):
        """Pre-marker hooks must refresh without a second consent prompt."""
        from cozempic import cli
        from cozempic.init import HOOK_SCHEMA_MARKER

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            home_claude = self._stub_home_claude(tmp)
            self._write_stale_global_hook(home_claude)
            marker = self._stub_marker(tmp)

            with mock.patch.object(cli, "_GLOBAL_INIT_MARKER", marker):
                with mock.patch.object(cli.Path, "home", return_value=home):
                    with mock.patch.object(cli.sys.stdin, "isatty", return_value=True):
                        with mock.patch.object(cli.sys.stderr, "isatty", return_value=True):
                            with mock.patch("builtins.input") as prompt:
                                cli._maybe_global_init(["list"])

            prompt.assert_not_called()
            command = json.loads(
                (home_claude / "settings.json").read_text()
            )["hooks"]["SessionStart"][0]["hooks"][0]["command"]
            self.assertIn(HOOK_SCHEMA_MARKER, command)

    def test_current_global_hook_ignores_stale_local_hook(self):
        """An unmanaged local stale hook cannot make global wiring stale."""
        from cozempic import cli

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            home_claude = self._stub_home_claude(tmp)
            self._write_current_global_hook(home_claude)
            self._write_stale_local_hook(home_claude)

            with mock.patch.object(cli.Path, "home", return_value=home):
                with mock.patch.object(cli, "run_init") as run_init:
                    cli._maybe_global_init(["list"])

            run_init.assert_not_called()

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

    def test_global_init_honors_config_dir(self):
        """Global hooks belong to Claude's configured profile, not ~/.claude."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            profile = Path(tmp) / "profile"
            profile.mkdir()
            marker = self._stub_marker(tmp)
            with mock.patch.object(cli, "_GLOBAL_INIT_MARKER", marker):
                with mock.patch.object(cli.Path, "home", return_value=home):
                    with mock.patch(
                        "cozempic.session.get_claude_dir", return_value=profile
                    ):
                        with mock.patch.object(cli.sys.stdin, "isatty", return_value=False):
                            cli._maybe_global_init(["list"])
            self.assertTrue((profile / "settings.json").exists())
            self.assertFalse((home / ".claude" / "settings.json").exists())

    def test_decline_in_one_profile_does_not_suppress_another(self):
        """Each CLAUDE_CONFIG_DIR profile owns its global-init decision."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            first.mkdir()
            second.mkdir()
            with mock.patch.object(cli.Path, "home", return_value=home):
                with mock.patch("cozempic.session.get_claude_dir", return_value=first):
                    with mock.patch.object(cli.sys.stdin, "isatty", return_value=True):
                        with mock.patch.object(cli.sys.stderr, "isatty", return_value=True):
                            with mock.patch("builtins.input", return_value="n"):
                                cli._maybe_global_init(["list"])
                with mock.patch("cozempic.session.get_claude_dir", return_value=second):
                    with mock.patch.object(cli.sys.stdin, "isatty", return_value=False):
                        cli._maybe_global_init(["list"])
            self.assertTrue((first / ".cozempic_global_initialized").exists())
            self.assertTrue((second / "settings.json").exists())

    def test_valid_non_object_global_settings_reports_error(self):
        """A valid JSON scalar must not crash global auto-init."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile"
            profile.mkdir()
            (profile / "settings.json").write_text("null")
            with mock.patch("cozempic.session.get_claude_dir", return_value=profile):
                with mock.patch("sys.stderr") as stderr:
                    cli._maybe_global_init(["list"])
            output = "".join(call.args[0] for call in stderr.write.call_args_list if call.args)
            self.assertIn("settings.json must be a JSON object", output)


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

    def test_duplicate_matcher_entries_are_consolidated(self):
        """A stale duplicate cannot survive beside a current canonical hook."""
        from cozempic.init import HOOK_SCHEMA_MARKER, _is_cozempic_command, wire_hooks

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".claude").mkdir()
            settings_path = Path(tmp) / ".claude" / "settings.json"
            settings_path.write_text(json.dumps({
                "hooks": {
                    "SessionStart": [
                        {
                            "matcher": "",
                            "hooks": [{
                                "type": "command",
                                "command": f"cozempic guard --daemon # {HOOK_SCHEMA_MARKER}",
                            }],
                        },
                        {
                            "matcher": "",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "cozempic guard --daemon # cozempic-hook-schema=v10",
                                },
                                {"type": "command", "command": "echo keep-me"},
                            ],
                        },
                    ],
                },
            }))

            result = wire_hooks(tmp)
            after = json.loads(settings_path.read_text())
            commands = [
                hook["command"]
                for entry in after["hooks"]["SessionStart"]
                for hook in entry["hooks"]
            ]
            cozempic_commands = [
                command for command in commands if _is_cozempic_command(command)
            ]

            self.assertIn("SessionStart[]", result["updated"])
            self.assertEqual(len(cozempic_commands), 1)
            self.assertIn(HOOK_SCHEMA_MARKER, cozempic_commands[0])
            self.assertIn("echo keep-me", commands)

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


class TestGlobalInitFailure(unittest.TestCase):
    def test_run_init_failure_does_not_become_a_decline(self):
        """A failed install must retry instead of creating an opt-out marker."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            # Fake home with .claude dir
            (Path(tmp) / ".claude").mkdir()
            os.environ.pop("COZEMPIC_NO_GLOBAL_INIT", None)
            marker = Path(tmp) / ".cozempic_global_initialized"
            with mock.patch.object(cli, "_GLOBAL_INIT_MARKER", marker):
                with mock.patch.object(cli.Path, "home", return_value=Path(tmp)):
                    with mock.patch.object(cli.sys.stdin, "isatty", return_value=False):
                        with mock.patch.object(cli.sys.stderr, "isatty", return_value=False):
                            with mock.patch.object(cli, "run_init", side_effect=OSError("boom")) as run_init:
                                cli._maybe_global_init(["list"])
                                cli._maybe_global_init(["list"])
            self.assertFalse(marker.exists(), "failure must not become an opt-out")
            self.assertEqual(run_init.call_count, 2)

    def test_explicit_global_init_error_does_not_mark_declined(self):
        from cozempic import cli
        import argparse

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            profile = Path(tmp) / "profile"
            profile.mkdir()
            marker = Path(tmp) / ".cozempic_global_initialized"
            (profile / "settings.json").write_text("{broken json")
            args = argparse.Namespace(
                uninstall_global=False,
                global_install=True,
                cwd=None,
                no_slash_command=True,
            )
            with mock.patch.object(cli, "_GLOBAL_INIT_MARKER", marker):
                with mock.patch.object(cli.Path, "home", return_value=home):
                    with mock.patch(
                        "cozempic.session.get_claude_dir", return_value=profile
                    ):
                        output = io.StringIO()
                        with mock.patch.object(cli.sys, "stdout", output):
                            cli.cmd_init(args)
            self.assertFalse(marker.exists())
            self.assertIn("Setup incomplete", output.getvalue())


if __name__ == "__main__":
    unittest.main()
