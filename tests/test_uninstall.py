"""cozempic uninstall — reverse of init (issue #147 FR)."""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cozempic import init as cz_init

# A realistic cozempic hook command (carries the schema marker + canonical wrapper
# shape that _is_cozempic_command recognizes — a bare "cozempic ..." is NOT matched
# by design, so user inline calls are never eaten).
COZ_CMD = ("export COZEMPIC_NO_AUTO_INIT=1; { cozempic checkpoint 2>/dev/null || "
           "python3 -m cozempic checkpoint; }  # cozempic-hook-schema=2")


def _settings_with(hooks):
    return {"hooks": hooks}


class _Base(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="cz_uninstall_"))
        # redirect HOME and the module-level markers into the temp home
        self._patches = [
            patch.dict(os.environ, {"HOME": str(self.home)}),
            patch.object(cz_init, "_GLOBAL_INIT_MARKER", self.home / ".cozempic_global_initialized"),
            patch.object(cz_init, "_REMIND_COUNTER", self.home / ".cozempic_remind_counter"),
            patch("cozempic.session.get_claude_dir", return_value=self.home / ".claude"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        import shutil
        shutil.rmtree(self.home, ignore_errors=True)

    def _write_global_settings(self, settings):
        p = self.home / ".claude" / "settings.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(settings))
        return p

    def _write_global_local_settings(self, settings):
        p = self.home / ".claude" / "settings.local.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(settings))
        return p

    def _write_slash(self, content):
        p = self.home / ".claude" / "commands" / "cozempic.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p


class TestRunUninstall(_Base):
    def test_project_uninstall_blocks_auto_init_until_explicit_init(self):
        from cozempic import cli

        project = self.home / "project"
        (project / ".claude").mkdir(parents=True)
        previous_cwd = Path.cwd()
        os.chdir(project)
        try:
            result = cz_init.run_uninstall("project")
        finally:
            os.chdir(previous_cwd)
        marker = project / ".claude" / ".cozempic_uninstalled"
        self.assertTrue(marker.exists())
        self.assertTrue(result["project_opt_out_set"])
        self.assertFalse((self.home / ".cozempic_global_initialized").exists())

        with patch.object(cli.Path, "cwd", return_value=project):
            with patch.object(cli, "run_init") as auto_init:
                cli._maybe_auto_init(["list"])
                auto_init.assert_not_called()

        cz_init.run_init(str(project), skip_slash=True)
        self.assertFalse(marker.exists())

    def test_project_uninstall_error_does_not_set_opt_out_marker(self):
        project = self.home / "project-error"
        claude = project / ".claude"
        claude.mkdir(parents=True)
        (claude / "settings.json").write_text('{"hooks": {}}')
        (claude / "settings.local.json").write_text("{broken json")
        previous_cwd = Path.cwd()
        os.chdir(project)
        try:
            result = cz_init.run_uninstall("project")
        finally:
            os.chdir(previous_cwd)
        self.assertTrue(result["errors"])
        self.assertFalse((claude / ".cozempic_uninstalled").exists())

    def test_removes_global_hooks_and_slash(self):
        self._write_global_settings(_settings_with({
            "SessionStart": [{"hooks": [{"type": "command", "command": COZ_CMD}]}]
        }))
        slash = self._write_slash("# cozempic\nDiagnose and prune bloated Claude Code context\ncozempic treat")
        res = cz_init.run_uninstall("global")
        self.assertTrue(any(h.get("removed") for h in res["hooks"]))
        self.assertTrue(res["slash_command"]["removed"])
        self.assertFalse(slash.exists())
        self.assertTrue((self.home / ".claude" / "commands" / "cozempic.md.bak").exists())
        self.assertTrue(res["opt_out_set"])
        self.assertTrue((self.home / ".cozempic_global_initialized").exists())  # opt-out marker

    def test_preserves_user_hooks_in_mixed_entry(self):
        self._write_global_settings(_settings_with({
            "SessionStart": [{"hooks": [
                {"type": "command", "command": COZ_CMD},
                {"type": "command", "command": "my-own-tool --do-thing"},
            ]}]
        }))
        cz_init.run_uninstall("global")
        s = json.loads((self.home / ".claude" / "settings.json").read_text())
        cmds = [h["command"] for e in s["hooks"]["SessionStart"] for h in e["hooks"]]
        self.assertIn("my-own-tool --do-thing", cmds)  # user hook kept
        self.assertNotIn(COZ_CMD, cmds)  # cozempic hook gone

    def test_leaves_foreign_slash_untouched(self):
        slash = self._write_slash("# My own command named cozempic\nnothing to do with the tool")
        res = cz_init.run_uninstall("global")
        self.assertTrue(slash.exists())  # not ours -> not removed
        self.assertTrue(res["slash_command"]["skipped_foreign"])

    def test_purge_removes_data_with_marker_kept(self):
        (self.home / ".cozempic").mkdir()
        (self.home / ".cozempic" / "receipts").mkdir()
        (self.home / ".cozempic_savings.json").write_text("{}")
        res = cz_init.run_uninstall("global", purge=True)
        self.assertFalse((self.home / ".cozempic").exists())
        self.assertFalse((self.home / ".cozempic_savings.json").exists())
        self.assertIn(str(self.home / ".cozempic"), res["purged"])
        # opt-out marker still set even on purge (so auto-init doesn't re-fire)
        self.assertTrue((self.home / ".cozempic_global_initialized").exists())

    def test_purge_failure_is_reported(self):
        data_dir = self.home / ".cozempic"
        data_dir.mkdir()
        with patch("shutil.rmtree", side_effect=OSError("permission denied")):
            result = cz_init.run_uninstall("global", purge=True)
        self.assertIn("Purge failed", result["errors"][0])

    def test_purge_is_skipped_when_global_hook_cleanup_fails(self):
        path = self.home / ".claude" / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{broken json")
        data_dir = self.home / ".cozempic"
        data_dir.mkdir()

        result = cz_init.run_uninstall("global", purge=True)

        self.assertTrue(result["errors"])
        self.assertEqual(result["purged"], [])
        self.assertTrue(data_dir.exists())

    def test_uninstall_write_failure_is_reported(self):
        self._write_global_settings(_settings_with({
            "SessionStart": [{"hooks": [{"type": "command", "command": COZ_CMD}]}]
        }))
        with patch("cozempic.init._save_settings", side_effect=OSError("disk full")):
            result = cz_init.run_uninstall("global")
        self.assertEqual(result["errors"], [
            f"could not write {self.home / '.claude' / 'settings.json'}: disk full"
        ])

    def test_project_scope_purge_keeps_global_data(self):
        (self.home / ".cozempic").mkdir()
        (self.home / ".cozempic_savings.json").write_text("{}")
        cz_init.run_uninstall("project", purge=True)
        self.assertTrue((self.home / ".cozempic").exists())
        self.assertTrue((self.home / ".cozempic_savings.json").exists())

    def test_cmd_uninstall_prints_non_hook_errors(self):
        from cozempic import cli

        result = {
            "hooks": [],
            "slash_command": None,
            "purged": [],
            "opt_out_set": False,
            "errors": ["Purge failed for /home/example/.cozempic: permission denied"],
        }
        output = io.StringIO()
        with patch("cozempic.init.run_uninstall", return_value=result), patch("sys.stdout", output):
            cli.cmd_uninstall(
                argparse.Namespace(project=False, all=False, purge=False, dry_run=False)
            )
        self.assertIn("ERROR: Purge failed", output.getvalue())

    def test_slash_backup_failure_is_reported_without_removal(self):
        slash = self._write_slash("# cozempic\nDiagnose and prune bloated Claude Code context\n")
        with patch("cozempic.init.shutil.copy2", side_effect=OSError("disk full")):
            result = cz_init.uninstall_slash_command()
        self.assertIn("Could not back up", result["error"])
        self.assertTrue(slash.exists())

    def test_init_marker_cleanup_failure_is_a_warning(self):
        project = self.home / "project-marker-error"
        project.mkdir()
        marker = unittest.mock.Mock()
        marker.unlink.side_effect = OSError("permission denied")
        result = None
        previous_cwd = Path.cwd()
        os.chdir(project)
        try:
            with patch.object(cz_init, "project_uninstall_marker", return_value=marker):
                result = cz_init.run_init(".", skip_slash=True)
        finally:
            os.chdir(previous_cwd)
        self.assertIn("Could not clear project uninstall marker", result["hooks"]["warnings"][0])

    def test_no_purge_keeps_data(self):
        (self.home / ".cozempic").mkdir()
        (self.home / ".cozempic_savings.json").write_text("{}")
        cz_init.run_uninstall("global", purge=False)
        self.assertTrue((self.home / ".cozempic").exists())
        self.assertTrue((self.home / ".cozempic_savings.json").exists())

    def test_idempotent_second_run(self):
        self._write_global_settings(_settings_with({
            "SessionStart": [{"hooks": [{"type": "command", "command": COZ_CMD}]}]
        }))
        cz_init.run_uninstall("global")
        res2 = cz_init.run_uninstall("global")  # nothing left
        self.assertFalse(any(h.get("removed") for h in res2["hooks"]))

    def test_removes_remind_counter(self):
        (self.home / ".cozempic_remind_counter").write_text("3")
        res = cz_init.run_uninstall("global")
        self.assertTrue(res["remind_counter_removed"])
        self.assertFalse((self.home / ".cozempic_remind_counter").exists())

    def test_surfaces_malformed_global_settings(self):
        path = self.home / ".claude" / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{broken json")
        result = cz_init.run_uninstall("global")
        self.assertTrue(result["errors"])
        self.assertFalse(result["opt_out_set"])

        from cozempic import cli
        output = io.StringIO()
        with patch("sys.stdout", output):
            cli.cmd_uninstall(
                argparse.Namespace(project=False, all=False, purge=False, dry_run=False)
            )
        self.assertIn("ERROR", output.getvalue())

    def test_surfaces_valid_non_object_global_settings(self):
        path = self.home / ".claude" / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("null")
        result = cz_init.run_uninstall("global")
        self.assertTrue(result["errors"])
        self.assertIn("JSON object", result["errors"][0])


class TestPreviewAndDryRun(_Base):
    def test_purge_uses_bounded_confirmation(self):
        from cozempic import cli

        with patch.object(cli.sys.stdin, "isatty", return_value=True), \
                patch.object(cli.sys.stderr, "isatty", return_value=True), \
                patch.object(cli, "_prompt_with_timeout", return_value="n") as prompt, \
                patch.object(cz_init, "run_uninstall") as run_uninstall:
            cli.cmd_uninstall(
                argparse.Namespace(project=False, all=False, purge=True, dry_run=False)
            )
        prompt.assert_called_once_with("  Continue? [y/N] ", timeout=30, default="n")
        run_uninstall.assert_not_called()

    def test_purge_requires_interactive_confirmation(self):
        from cozempic import cli

        with patch.object(cli.sys.stdin, "isatty", return_value=False), \
                patch.object(cli, "_prompt_with_timeout") as prompt, \
                patch.object(cz_init, "run_uninstall") as run_uninstall:
            cli.cmd_uninstall(argparse.Namespace(project=False, all=False, purge=True, dry_run=False))
        prompt.assert_not_called()
        run_uninstall.assert_called_once_with("global", False)

    def test_project_purge_is_rejected(self):
        from cozempic import cli

        output = io.StringIO()
        with patch("sys.stdout", output), patch.object(cz_init, "run_uninstall") as run_uninstall:
            cli.cmd_uninstall(argparse.Namespace(project=True, all=False, purge=True, dry_run=False))
        self.assertIn("only applies to global data", output.getvalue())
        run_uninstall.assert_not_called()

    def test_project_preview_does_not_offer_global_remind_counter_removal(self):
        (self.home / ".cozempic_remind_counter").write_text("3")
        project = self.home / "project"
        project.mkdir()
        previous_cwd = Path.cwd()
        os.chdir(project)
        try:
            preview = cz_init.preview_uninstall("project")
        finally:
            os.chdir(previous_cwd)
        self.assertFalse(preview["remind_counter"])

    def test_project_preview_does_not_offer_global_purge(self):
        (self.home / ".cozempic").mkdir()
        preview = cz_init.preview_uninstall("project", purge=True)
        self.assertEqual(preview["purge_data"], [])

    def test_preview_reports_without_mutating(self):
        sp = self._write_global_settings(_settings_with({
            "SessionStart": [{"hooks": [{"type": "command", "command": COZ_CMD}]}]
        }))
        before = sp.read_text()
        prev = cz_init.preview_uninstall("global")
        self.assertIn(str(sp), prev["hooks_in"])
        self.assertEqual(sp.read_text(), before)  # untouched

    def test_global_uninstall_and_preview_include_local_settings(self):
        settings = _settings_with({
            "SessionStart": [{"hooks": [{"type": "command", "command": COZ_CMD}]}]
        })
        local = self._write_global_local_settings(settings)
        preview = cz_init.preview_uninstall("global")
        self.assertIn(str(local), preview["hooks_in"])
        cz_init.run_uninstall("global")
        self.assertNotIn("cozempic", local.read_text())

    def test_cmd_dry_run_changes_nothing(self):
        from cozempic import cli

        sp = self._write_global_settings(_settings_with({
            "SessionStart": [{"hooks": [{"type": "command", "command": COZ_CMD}]}]
        }))
        before = sp.read_text()
        cli.cmd_uninstall(argparse.Namespace(project=False, all=False, purge=False, dry_run=True))
        self.assertEqual(sp.read_text(), before)
        self.assertFalse((self.home / ".cozempic_global_initialized").exists())  # no opt-out write either

    def test_preview_reports_malformed_settings(self):
        path = self.home / ".claude" / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{broken json")
        preview = cz_init.preview_uninstall("global")
        self.assertTrue(preview["errors"])

    def test_uninstall_creates_missing_opt_out_marker_parents(self):
        self.assertTrue(cz_init.run_uninstall("global")["opt_out_set"])
        self.assertTrue((self.home / ".cozempic_global_initialized").exists())

        project = self.home / "project"
        project.mkdir()
        previous_cwd = Path.cwd()
        os.chdir(project)
        try:
            self.assertTrue(cz_init.run_uninstall("project")["project_opt_out_set"])
        finally:
            os.chdir(previous_cwd)
        self.assertTrue((project / ".claude" / ".cozempic_uninstalled").exists())

    def test_install_slash_command_returns_error_instead_of_raising(self):
        with patch("cozempic.init.shutil.copy2", side_effect=OSError("disk full")):
            result = cz_init.install_slash_command(".")
        self.assertIn("Could not install slash command", result["error"])

    def test_uninstall_rejects_unknown_scope(self):
        with self.assertRaisesRegex(ValueError, "Unknown uninstall scope"):
            cz_init.preview_uninstall("typo")
        with self.assertRaisesRegex(ValueError, "Unknown uninstall scope"):
            cz_init.run_uninstall("typo")

    def test_dry_run_labels_slash_errors_and_formats_paths(self):
        from cozempic import cli

        preview = {
            "hooks_in": ["/one/settings.json", "/two/settings.local.json"],
            "slash_command": False,
            "remind_counter": False,
            "purge_data": ["/home/example/.cozempic"],
            "hook_errors": ["invalid settings"],
            "slash_error": "Could not read slash command: permission denied",
        }
        output = io.StringIO()
        with patch("cozempic.init.preview_uninstall", return_value=preview), patch("sys.stdout", output):
            cli.cmd_uninstall(argparse.Namespace(project=False, all=False, purge=True, dry_run=True))
        text = output.getvalue()
        self.assertIn("Hooks: ERROR — invalid settings", text)
        self.assertIn("Slash command: ERROR — Could not read slash command", text)
        self.assertIn("Slash command (~/.claude/commands/cozempic.md): (could not determine)", text)
        self.assertNotIn("(not present / not ours)", text)
        self.assertIn("/one/settings.json, /two/settings.local.json", text)
        self.assertNotIn("['/one/settings.json'", text)


class TestOptOutHolds(_Base):
    def test_opt_out_marker_blocks_refire(self):
        # after uninstall, the global-init marker exists -> auto-init must skip
        cz_init.run_uninstall("global")
        self.assertTrue(cz_init._GLOBAL_INIT_MARKER.exists())


if __name__ == "__main__":
    unittest.main()
