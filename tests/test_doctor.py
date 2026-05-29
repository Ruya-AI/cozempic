"""Tests for doctor health checks."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cozempic.doctor import (
    check_agent_model_mismatch,
    check_claude_json_corruption,
    check_corrupted_tool_use,
    check_hooks_trust_flag,
    check_orphaned_tool_results,
    check_oversized_sessions,
    check_stale_tmp_artifacts,
    check_zombie_teams,
    fix_claude_json_corruption,
    fix_hooks_trust_flag,
    fix_stale_tmp_artifacts,
)


class TestClaudeJsonCorruption(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.claude_json = Path(self.tmpdir) / ".claude.json"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_valid_json_ok(self):
        self.claude_json.write_text(json.dumps({"numStartups": 50, "auth": "token123"}))
        with patch("cozempic.doctor.get_claude_json_path", return_value=self.claude_json):
            result = check_claude_json_corruption()
        self.assertEqual(result.status, "ok")

    def test_empty_file_is_issue(self):
        self.claude_json.write_text("")
        with patch("cozempic.doctor.get_claude_json_path", return_value=self.claude_json):
            result = check_claude_json_corruption()
        self.assertEqual(result.status, "issue")
        self.assertIn("empty", result.message)

    def test_truncated_json_is_issue(self):
        self.claude_json.write_text('{"numStartups": 50, "auth": "tok')
        with patch("cozempic.doctor.get_claude_json_path", return_value=self.claude_json):
            result = check_claude_json_corruption()
        self.assertEqual(result.status, "issue")
        self.assertIn("invalid JSON", result.message)

    def test_missing_file_is_ok(self):
        missing = Path(self.tmpdir) / "nonexistent.json"
        with patch("cozempic.doctor.get_claude_json_path", return_value=missing):
            result = check_claude_json_corruption()
        self.assertEqual(result.status, "ok")

    def test_fix_restores_from_backup(self):
        # Create corrupted file
        self.claude_json.write_text("corrupted{{{")
        # Create valid backup
        backup = self.claude_json.parent / ".claude.json.bak"
        backup.write_text(json.dumps({"numStartups": 100, "auth": "valid"}))

        with patch("cozempic.doctor.get_claude_json_path", return_value=self.claude_json):
            msg = fix_claude_json_corruption()

        self.assertIn("Restored", msg)
        # Verify restored content is valid
        data = json.loads(self.claude_json.read_text())
        self.assertEqual(data["numStartups"], 100)


class TestCorruptedToolUse(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.session_path = Path(self.tmpdir) / "projects" / "test" / "session.jsonl"
        self.session_path.parent.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_session(self, messages):
        with open(self.session_path, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

    def test_detects_long_tool_name(self):
        self._write_session([
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "Task" + "x" * 300, "input": {}}
                    ],
                },
            }
        ])
        sessions = [{"path": self.session_path, "session_id": "test", "size": 1000}]
        with patch("cozempic.doctor.find_sessions", return_value=sessions):
            result = check_corrupted_tool_use()
        self.assertEqual(result.status, "issue")

    def test_normal_tool_name_ok(self):
        self._write_session([
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/foo"}}
                    ],
                },
            }
        ])
        sessions = [{"path": self.session_path, "session_id": "test", "size": 1000}]
        with patch("cozempic.doctor.find_sessions", return_value=sessions):
            result = check_corrupted_tool_use()
        self.assertEqual(result.status, "ok")


class TestOrphanedToolResults(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.session_path = Path(self.tmpdir) / "projects" / "test" / "session.jsonl"
        self.session_path.parent.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_session(self, messages):
        with open(self.session_path, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

    def test_detects_orphaned_tool_result(self):
        self._write_session([
            # tool_result with no matching tool_use
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "missing_id", "content": "result"}
                    ],
                },
            }
        ])
        sessions = [{"path": self.session_path, "session_id": "test", "size": 1000}]
        with patch("cozempic.doctor.find_sessions", return_value=sessions):
            result = check_orphaned_tool_results()
        self.assertEqual(result.status, "issue")

    def test_paired_tool_use_result_ok(self):
        self._write_session([
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "file content"}
                    ],
                },
            }
        ])
        sessions = [{"path": self.session_path, "session_id": "test", "size": 1000}]
        with patch("cozempic.doctor.find_sessions", return_value=sessions):
            result = check_orphaned_tool_results()
        self.assertEqual(result.status, "ok")


class TestZombieTeams(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.teams_dir = Path(self.tmpdir) / ".claude" / "teams"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_teams_dir_ok(self):
        claude_dir = Path(self.tmpdir) / ".claude"
        claude_dir.mkdir(parents=True)
        with patch("cozempic.doctor.get_claude_dir", return_value=claude_dir):
            result = check_zombie_teams()
        self.assertEqual(result.status, "ok")

    def test_team_without_config_is_stale(self):
        self.teams_dir.mkdir(parents=True)
        stale_team = self.teams_dir / "dead-team"
        stale_team.mkdir()
        # No config.json inside

        claude_dir = Path(self.tmpdir) / ".claude"
        with patch("cozempic.doctor.get_claude_dir", return_value=claude_dir):
            result = check_zombie_teams()
        self.assertIn(result.status, ("warning", "issue"))
        self.assertIn("stale", result.message)

    def test_fresh_team_with_config_ok(self):
        self.teams_dir.mkdir(parents=True)
        active_team = self.teams_dir / "active-team"
        active_team.mkdir()
        config = active_team / "config.json"
        config.write_text(json.dumps({"name": "active", "members": []}))

        claude_dir = Path(self.tmpdir) / ".claude"
        with patch("cozempic.doctor.get_claude_dir", return_value=claude_dir):
            result = check_zombie_teams()
        self.assertEqual(result.status, "ok")


class TestHooksTrustFlag(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.claude_json = Path(self.tmpdir) / ".claude.json"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_trusted_workspace_missing_hooks_flag_is_issue(self):
        self.claude_json.write_text(json.dumps({
            "/path/to/project": {"hasTrustDialogAccepted": True},
        }))
        with patch("cozempic.doctor.get_claude_json_path", return_value=self.claude_json):
            result = check_hooks_trust_flag()
        self.assertEqual(result.status, "issue")
        self.assertIn("hasTrustDialogHooksAccepted", result.message)

    def test_trusted_workspace_with_hooks_flag_is_ok(self):
        self.claude_json.write_text(json.dumps({
            "/path/to/project": {
                "hasTrustDialogAccepted": True,
                "hasTrustDialogHooksAccepted": True,
            },
        }))
        with patch("cozempic.doctor.get_claude_json_path", return_value=self.claude_json):
            result = check_hooks_trust_flag()
        self.assertEqual(result.status, "ok")

    def test_untrusted_workspace_is_ok(self):
        self.claude_json.write_text(json.dumps({
            "/path/to/project": {"hasTrustDialogAccepted": False},
        }))
        with patch("cozempic.doctor.get_claude_json_path", return_value=self.claude_json):
            result = check_hooks_trust_flag()
        self.assertEqual(result.status, "ok")

    def test_missing_file_is_ok(self):
        missing = Path(self.tmpdir) / "nonexistent.json"
        with patch("cozempic.doctor.get_claude_json_path", return_value=missing):
            result = check_hooks_trust_flag()
        self.assertEqual(result.status, "ok")

    def test_fix_sets_hooks_flag(self):
        self.claude_json.write_text(json.dumps({
            "/path/to/project": {"hasTrustDialogAccepted": True},
        }))
        with patch("cozempic.doctor.get_claude_json_path", return_value=self.claude_json):
            msg = fix_hooks_trust_flag()
        self.assertIn("1", msg)
        data = json.loads(self.claude_json.read_text())
        self.assertTrue(data["/path/to/project"]["hasTrustDialogHooksAccepted"])

    def test_fix_multiple_projects(self):
        self.claude_json.write_text(json.dumps({
            "/project/a": {"hasTrustDialogAccepted": True},
            "/project/b": {"hasTrustDialogAccepted": True},
            "/project/c": {"hasTrustDialogAccepted": False},
        }))
        with patch("cozempic.doctor.get_claude_json_path", return_value=self.claude_json):
            msg = fix_hooks_trust_flag()
        self.assertIn("2", msg)
        data = json.loads(self.claude_json.read_text())
        self.assertTrue(data["/project/a"]["hasTrustDialogHooksAccepted"])
        self.assertTrue(data["/project/b"]["hasTrustDialogHooksAccepted"])
        self.assertNotIn("hasTrustDialogHooksAccepted", data["/project/c"])


class TestAgentModelMismatch(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.claude_dir = Path(self.tmpdir) / ".claude"
        self.claude_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_teams_dir_is_ok(self):
        with patch("cozempic.doctor.get_claude_dir", return_value=self.claude_dir):
            result = check_agent_model_mismatch()
        self.assertEqual(result.status, "ok")

    def test_empty_teams_dir_is_ok(self):
        (self.claude_dir / "teams").mkdir()
        with patch("cozempic.doctor.get_claude_dir", return_value=self.claude_dir):
            result = check_agent_model_mismatch()
        self.assertEqual(result.status, "ok")

    def test_teams_with_model_in_settings_is_ok(self):
        (self.claude_dir / "teams" / "my-team").mkdir(parents=True)
        (self.claude_dir / "settings.json").write_text(
            json.dumps({"model": "claude-opus-4-7"})
        )
        with patch("cozempic.doctor.get_claude_dir", return_value=self.claude_dir):
            result = check_agent_model_mismatch()
        self.assertEqual(result.status, "ok")
        self.assertIn("claude-opus-4-7", result.message)

    def test_teams_without_model_in_settings_is_warning(self):
        (self.claude_dir / "teams" / "my-team").mkdir(parents=True)
        (self.claude_dir / "settings.json").write_text(json.dumps({"theme": "dark"}))
        with patch("cozempic.doctor.get_claude_dir", return_value=self.claude_dir):
            result = check_agent_model_mismatch()
        self.assertEqual(result.status, "warning")

    def test_teams_without_settings_file_is_warning(self):
        (self.claude_dir / "teams" / "my-team").mkdir(parents=True)
        with patch("cozempic.doctor.get_claude_dir", return_value=self.claude_dir):
            result = check_agent_model_mismatch()
        self.assertEqual(result.status, "warning")


class TestStaleTmpArtifacts(unittest.TestCase):
    """stale-tmp-artifacts check must detect dead-PID .pid files, their paired
    .log files, and orphaned hook .lock files — without ever deleting anything
    still in use (live guard PID or held flock)."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        # patch the /tmp directory used by the check so we stay isolated
        self._patcher = patch("cozempic.doctor._TMP_DIR", self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_pid_file(self, slug: str, pid: int) -> Path:
        p = self.tmpdir / f"cozempic_guard_{slug}.pid"
        p.write_text(str(pid))
        return p

    def _make_log_file(self, slug: str, content: str = "log\n") -> Path:
        p = self.tmpdir / f"cozempic_guard_{slug}.log"
        p.write_text(content)
        return p

    def _make_lock_file(self, slug: str) -> Path:
        p = self.tmpdir / f"cozempic_hook_{slug}.lock"
        p.write_text("")
        return p

    # ── check: detection ──────────────────────────────────────────────────────
    def test_no_artifacts_is_ok(self):
        result = check_stale_tmp_artifacts()
        self.assertEqual(result.status, "ok")

    def test_live_guard_is_not_stale(self):
        """A .pid file pointing to a live cozempic guard + its paired .log
        must not be flagged."""
        live_pid = 42
        self._make_pid_file("livesess-12", live_pid)
        self._make_log_file("livesess-12")
        with patch("cozempic.doctor._is_live_guard_pid", return_value=True):
            result = check_stale_tmp_artifacts()
        self.assertEqual(result.status, "ok")

    def test_dead_pid_file_is_stale(self):
        """A .pid file whose PID is no longer a running cozempic guard is stale."""
        self._make_pid_file("deadsess-01", 999999)
        with patch("cozempic.doctor._is_live_guard_pid", return_value=False):
            result = check_stale_tmp_artifacts()
        self.assertIn(result.status, ("warning", "issue"))
        self.assertIn("pid", result.message.lower())

    def test_orphan_log_without_pid_is_stale(self):
        """A .log file with no matching .pid file is an orphan."""
        self._make_log_file("orphanlog-03")
        result = check_stale_tmp_artifacts()
        self.assertIn(result.status, ("warning", "issue"))
        self.assertIn("log", result.message.lower())

    def test_orphan_lock_is_stale_when_not_held(self):
        """A .lock file that can be acquired with LOCK_EX|LOCK_NB is orphaned."""
        self._make_lock_file("oldhook-05")
        with patch("cozempic.doctor._is_lock_held", return_value=False):
            result = check_stale_tmp_artifacts()
        self.assertIn(result.status, ("warning", "issue"))
        self.assertIn("lock", result.message.lower())

    def test_held_lock_is_not_stale(self):
        """A .lock file currently held by another process must be preserved."""
        self._make_lock_file("activehook-06")
        with patch("cozempic.doctor._is_lock_held", return_value=True):
            result = check_stale_tmp_artifacts()
        self.assertEqual(result.status, "ok")

    def test_garbage_pid_content_is_treated_as_stale(self):
        """Non-integer .pid file content must not crash — treat as stale."""
        pid_path = self.tmpdir / "cozempic_guard_badcontent-07.pid"
        pid_path.write_text("not-an-integer")
        result = check_stale_tmp_artifacts()
        self.assertIn(result.status, ("warning", "issue"))

    def test_threshold_escalation_to_issue(self):
        """Many stale artifacts escalate status from warning to issue."""
        for i in range(15):
            self._make_pid_file(f"stale{i:02d}-x", 999900 + i)
        with patch("cozempic.doctor._is_live_guard_pid", return_value=False):
            result = check_stale_tmp_artifacts()
        self.assertEqual(result.status, "issue")

    def test_protected_globals_are_never_reported(self):
        """Global append-only files (breaker state, cozempic_guard.log,
        cozempic_reload.log) must NEVER be flagged — they are intentional."""
        (self.tmpdir / "cozempic_guard.log").write_text("global log")
        (self.tmpdir / "cozempic_reload.log").write_text("reload log")
        (self.tmpdir / "cozempic_breaker_abc123.json").write_text("{}")
        result = check_stale_tmp_artifacts()
        self.assertEqual(result.status, "ok")

    # ── fix: deletion safety ──────────────────────────────────────────────────
    def test_fix_deletes_stale_pid_and_paired_log(self):
        self._make_pid_file("deadsess-10", 999999)
        log_path = self._make_log_file("deadsess-10")
        with patch("cozempic.doctor._is_live_guard_pid", return_value=False):
            msg = fix_stale_tmp_artifacts()
        self.assertFalse((self.tmpdir / "cozempic_guard_deadsess-10.pid").exists())
        self.assertFalse(log_path.exists())
        self.assertIn("2", msg)  # reports 2 files deleted

    def test_fix_preserves_live_guard_files(self):
        pid_path = self._make_pid_file("livesess-20", 42)
        log_path = self._make_log_file("livesess-20")
        with patch("cozempic.doctor._is_live_guard_pid", return_value=True):
            fix_stale_tmp_artifacts()
        self.assertTrue(pid_path.exists())
        self.assertTrue(log_path.exists())

    def test_fix_preserves_held_locks(self):
        lock_path = self._make_lock_file("activehook-21")
        with patch("cozempic.doctor._is_lock_held", return_value=True):
            fix_stale_tmp_artifacts()
        self.assertTrue(lock_path.exists())

    def test_fix_preserves_global_files(self):
        guard_log = self.tmpdir / "cozempic_guard.log"
        guard_log.write_text("global")
        breaker = self.tmpdir / "cozempic_breaker_xyz.json"
        breaker.write_text("{}")
        fix_stale_tmp_artifacts()
        self.assertTrue(guard_log.exists())
        self.assertTrue(breaker.exists())

    def test_fix_is_idempotent(self):
        """Running fix twice on empty /tmp returns a sensible message."""
        msg1 = fix_stale_tmp_artifacts()
        msg2 = fix_stale_tmp_artifacts()
        self.assertIn("0", msg1)
        self.assertIn("0", msg2)


class TestOversizedSessions(unittest.TestCase):

    def _make_session(self, session_id: str, size_mb: int) -> dict:
        return {"session_id": session_id, "path": f"/tmp/fake/{session_id}.jsonl", "size": size_mb * 1024 * 1024}

    def test_no_large_sessions_ok(self):
        sessions = [self._make_session("aabbccdd1122", 10)]
        with patch("cozempic.doctor.find_sessions", return_value=sessions):
            result = check_oversized_sessions()
        self.assertEqual(result.status, "ok")
        self.assertIsNone(result.fix_description)

    def test_fix_description_contains_real_session_ids(self):
        sessions = [
            self._make_session("aabbccdd1122eeff", 80),
            self._make_session("11223344aabbccdd", 60),
        ]
        with patch("cozempic.doctor.find_sessions", return_value=sessions):
            result = check_oversized_sessions()
        self.assertEqual(result.status, "issue")
        self.assertIn("cozempic treat aabbccdd", result.fix_description)
        self.assertIn("cozempic treat 11223344", result.fix_description)
        self.assertNotIn("<session>", result.fix_description)

    def test_fix_description_one_line_per_session(self):
        sessions = [
            self._make_session("aaaa1111bbbb2222", 100),
            self._make_session("cccc3333dddd4444", 75),
            self._make_session("eeee5555ffff6666", 55),
        ]
        with patch("cozempic.doctor.find_sessions", return_value=sessions):
            result = check_oversized_sessions()
        lines = [l for l in result.fix_description.splitlines() if "cozempic treat" in l]
        self.assertEqual(len(lines), 3)

    def test_fix_description_sorted_largest_first(self):
        sessions = [
            self._make_session("small11122233344", 55),
            self._make_session("large99988877766", 200),
        ]
        with patch("cozempic.doctor.find_sessions", return_value=sessions):
            result = check_oversized_sessions()
        idx_large = result.fix_description.index("large999")
        idx_small = result.fix_description.index("small111")
        self.assertLess(idx_large, idx_small)


if __name__ == "__main__":
    unittest.main()
