"""Tests for _maybe_auto_init() skipping local init when global hooks are current."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TestAutoInitGlobalSkip(unittest.TestCase):
    """_maybe_auto_init() must skip local init when global hooks are current."""

    def setUp(self):
        self._config_patch = mock.patch(
            "cozempic.session.get_claude_dir",
            side_effect=lambda: Path.home() / ".claude",
        )
        self._config_patch.start()

    def tearDown(self):
        self._config_patch.stop()

    def _make_project(self, tmpdir):
        """Create a fake project with .claude/ dir under tmpdir."""
        project = Path(tmpdir) / "myproject"
        (project / ".claude").mkdir(parents=True)
        return project

    def _write_current_hooks(self, claude_dir, filename="settings.json"):
        """Write a settings.json with current-schema cozempic hooks."""
        from cozempic.init import HOOK_SCHEMA_MARKER
        settings = claude_dir / filename
        settings.write_text(json.dumps({
            "hooks": {
                "SessionStart": [{
                    "matcher": "",
                    "hooks": [{
                        "type": "command",
                        "command": f"cozempic guard --daemon # {HOOK_SCHEMA_MARKER}",
                    }],
                }],
            }
        }))

    def _write_stale_hooks(self, claude_dir, filename="settings.json"):
        """Write a settings.json with stale (pre-schema) cozempic hooks."""
        settings = claude_dir / filename
        settings.write_text(json.dumps({
            "hooks": {
                "SessionStart": [{
                    "matcher": "",
                    "hooks": [{
                        "type": "command",
                        "command": "{ cozempic guard --daemon 2>/dev/null || python3 -m cozempic guard --daemon 2>/dev/null; } || true",
                    }],
                }],
            }
        }))

    def test_skips_when_global_hooks_current(self):
        """Global hooks current -> no local init."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home_claude = home / ".claude"
            home_claude.mkdir(parents=True)
            self._write_current_hooks(home_claude)

            project = self._make_project(tmp)

            with mock.patch.object(cli.Path, "home", return_value=home):
                with mock.patch.object(cli.Path, "cwd", return_value=project):
                    with mock.patch.object(cli, "run_init") as ri:
                        cli._maybe_auto_init(["list"])
                        ri.assert_not_called()

    def test_skips_when_managed_global_hooks_current_despite_stale_local_file(self):
        """Global settings.local.json must not override managed global hooks."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home_claude = home / ".claude"
            home_claude.mkdir(parents=True)
            self._write_current_hooks(home_claude)
            self._write_stale_hooks(home_claude, "settings.local.json")
            project = self._make_project(tmp)

            with mock.patch.object(cli.Path, "home", return_value=home):
                with mock.patch.object(cli.Path, "cwd", return_value=project):
                    with mock.patch.object(cli, "run_init") as ri:
                        cli._maybe_auto_init(["list"])
                        ri.assert_not_called()

    def test_skips_when_managed_global_hooks_current_despite_malformed_local_file(self):
        """Unreadable unmanaged global-local state must not trigger local hooks."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home_claude = home / ".claude"
            home_claude.mkdir(parents=True)
            self._write_current_hooks(home_claude)
            (home_claude / "settings.local.json").write_text("{broken json")
            project = self._make_project(tmp)

            with mock.patch.object(cli.Path, "home", return_value=home):
                with mock.patch.object(cli.Path, "cwd", return_value=project):
                    with mock.patch.object(cli, "run_init") as ri:
                        cli._maybe_auto_init(["list"])
                        ri.assert_not_called()

    def test_skips_local_init_when_managed_global_settings_are_malformed(self):
        """Do not install local hooks while managed global settings need repair."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home_claude = home / ".claude"
            home_claude.mkdir(parents=True)
            (home_claude / "settings.json").write_text("{broken json")
            project = self._make_project(tmp)

            with mock.patch.object(cli.Path, "home", return_value=home):
                with mock.patch.object(cli.Path, "cwd", return_value=project):
                    with mock.patch("sys.stderr") as mock_stderr:
                        with mock.patch.object(cli, "run_init") as ri:
                            cli._maybe_auto_init(["list"])
                            ri.assert_not_called()
            output = "".join(
                call.args[0] for call in mock_stderr.write.call_args_list if call.args
            )
            self.assertIn("auto-init skipped", output)

    def test_fires_when_global_hooks_absent(self):
        """No global hooks -> local auto-init should fire."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home_claude = home / ".claude"
            home_claude.mkdir(parents=True)
            # No hooks written to global settings

            project = self._make_project(tmp)

            with mock.patch.object(cli.Path, "home", return_value=home):
                with mock.patch.object(cli.Path, "cwd", return_value=project):
                    with mock.patch.object(cli, "run_init", return_value={"hooks": {"added": ["SessionStart[]"], "updated": []}}) as ri:
                        cli._maybe_auto_init(["list"])
                        ri.assert_called_once()

    def test_skips_when_project_local_hooks_current(self):
        """A current local hook must not be duplicated into settings.json."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            (home / ".claude").mkdir(parents=True)
            project = self._make_project(tmp)
            self._write_current_hooks(project / ".claude", "settings.local.json")

            with mock.patch.object(cli.Path, "home", return_value=home):
                with mock.patch.object(cli.Path, "cwd", return_value=project):
                    with mock.patch.object(cli, "run_init") as ri:
                        cli._maybe_auto_init(["list"])
                        ri.assert_not_called()

    def test_fires_when_global_hooks_stale(self):
        """Global hooks stale (old schema) -> local auto-init should fire."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home_claude = home / ".claude"
            home_claude.mkdir(parents=True)
            self._write_stale_hooks(home_claude)

            project = self._make_project(tmp)

            with mock.patch.object(cli.Path, "home", return_value=home):
                with mock.patch.object(cli.Path, "cwd", return_value=project):
                    with mock.patch.object(cli, "run_init", return_value={"hooks": {"added": ["SessionStart[]"], "updated": []}}) as ri:
                        cli._maybe_auto_init(["list"])
                        ri.assert_called_once()

    def test_warns_when_global_current_and_local_hooks_present(self):
        """Global hooks current + local hooks exist -> warn to stderr."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home_claude = home / ".claude"
            home_claude.mkdir(parents=True)
            self._write_current_hooks(home_claude)

            project = self._make_project(tmp)
            # Write local hooks too
            self._write_current_hooks(project / ".claude")

            with mock.patch.object(cli.Path, "home", return_value=home):
                with mock.patch.object(cli.Path, "cwd", return_value=project):
                    with mock.patch("sys.stderr") as mock_stderr:
                        with mock.patch.object(cli, "run_init") as ri:
                            cli._maybe_auto_init(["list"])
                            ri.assert_not_called()
                        # Check warning was printed
                        mock_stderr.write.assert_called()
                        output = "".join(
                            call.args[0] for call in mock_stderr.write.call_args_list
                            if call.args
                        )
                        self.assertIn("redundant", output.lower())

    def test_no_warn_when_global_current_and_no_local_hooks(self):
        """Global hooks current + no local hooks -> skip silently, no warning."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home_claude = home / ".claude"
            home_claude.mkdir(parents=True)
            self._write_current_hooks(home_claude)

            project = self._make_project(tmp)
            # No local hooks written

            with mock.patch.object(cli.Path, "home", return_value=home):
                with mock.patch.object(cli.Path, "cwd", return_value=project):
                    with mock.patch("sys.stderr") as mock_stderr:
                        with mock.patch.object(cli, "run_init") as ri:
                            cli._maybe_auto_init(["list"])
                            ri.assert_not_called()
                        # No warning should be printed
                        output = "".join(
                            call.args[0] for call in mock_stderr.write.call_args_list
                            if call.args
                        )
                        self.assertNotIn("redundant", output.lower())

    def test_no_skip_when_cwd_is_home(self):
        """When cwd is home dir, global == local -- guard must prevent
        the global-skip branch from firing. We verify by making
        _project_is_cozempic_current return False so run_init is called,
        proving the global-skip branch (which would return early) was NOT taken."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home_claude = home / ".claude"
            home_claude.mkdir(parents=True)
            self._write_current_hooks(home_claude)

            # cwd IS the home dir -- home_claude == claude_dir
            with mock.patch.object(cli.Path, "home", return_value=home):
                with mock.patch.object(cli.Path, "cwd", return_value=home):
                    # Return an empty local state so the local check falls through to run_init.
                    # If the guard were missing, the global-skip branch would fire
                    # (real hooks ARE current) and return early -- run_init would
                    # never be called. So run_init being called proves the guard works.
                    with mock.patch.object(
                        cli,
                        "_project_cozempic_hook_state",
                        return_value=mock.Mock(found=False, current=False, error=None),
                    ):
                        with mock.patch.object(cli, "run_init", return_value={"hooks": {"added": ["SessionStart[]"], "updated": []}}) as ri:
                            cli._maybe_auto_init(["list"])
                            ri.assert_called_once()

    def test_skips_when_project_settings_are_malformed(self):
        """A broken local settings file must not be overwritten by auto-init."""
        from cozempic import cli
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            (home / ".claude").mkdir(parents=True)
            project = self._make_project(tmp)
            (project / ".claude" / "settings.local.json").write_text("{broken json")

            with mock.patch.object(cli.Path, "home", return_value=home):
                with mock.patch.object(cli.Path, "cwd", return_value=project):
                    with mock.patch.object(cli, "run_init") as run_init:
                        cli._maybe_auto_init(["list"])
                        run_init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
