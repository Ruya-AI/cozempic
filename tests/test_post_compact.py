"""Tests for PostCompact recovery — read_team_checkpoint, cmd_post_compact, and hook config."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from cozempic.team import read_team_checkpoint
from cozempic.init import COZEMPIC_HOOKS


def _write_session_file(proj_dir: Path, session_id: str, content: str = "") -> Path:
    """Helper: write a minimal JSONL session file into proj_dir."""
    proj_dir.mkdir(parents=True, exist_ok=True)
    p = proj_dir / f"{session_id}.jsonl"
    p.write_text(
        content or json.dumps({"role": "user", "content": "hi"}) + "\n",
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# TestCmdPostCompactCrossProjectIsolation — Bug C regression
# ---------------------------------------------------------------------------

class TestCmdPostCompactCrossProjectIsolation(unittest.TestCase):
    """cmd_post_compact must NEVER inject another project's checkpoint."""

    def _run_post_compact(self, cwd: str) -> str:
        from cozempic.cli import cmd_post_compact
        args = argparse.Namespace(cwd=cwd)
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            cmd_post_compact(args)
        finally:
            sys.stdout = old_stdout
        return captured.getvalue()

    def test_does_not_return_other_projects_checkpoint_when_other_is_newer(self):
        """Core bug: Strategy 5 picks a newer OTHER project's session → wrong checkpoint.

        Fixture uses the CORRECT dir names (as Claude Code actually creates them, with dashes
        for underscores). Old code computes broken slug with '_', so Strategy 4 misses project A
        and Strategy 5 returns project B's (newer) session → contamination.
        """
        import re as _re
        tmp_path = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp_path, ignore_errors=True)

        # The CORRECT (fixed) slug formula — what Claude Code actually stores on disk
        def _correct_slug(cwd: str) -> str:
            return _re.sub(r"[^a-zA-Z0-9]", "-", cwd)

        # Project A: topstep_automation — dir name uses dashes (Claude's real format)
        cwd_a = "/Users/x/topstep_automation"
        slug_a_correct = _correct_slug(cwd_a)   # "-Users-x-topstep-automation"
        proj_a = tmp_path / "projects" / slug_a_correct
        _write_session_file(proj_a, "aaaa1111-0000-0000-0000-000000000001")
        # Write a checkpoint for project A
        (proj_a / "team-checkpoint.md").write_text("TOPSTEP", encoding="utf-8")

        # Small sleep ensures project B mtime is strictly newer
        time.sleep(0.01)

        # Project B: fanugugc (no underscore → still returned by Strategy 5 when A is missed)
        cwd_b = "/Users/x/fanugugc"
        slug_b_correct = _correct_slug(cwd_b)   # "-Users-x-fanugugc"
        proj_b = tmp_path / "projects" / slug_b_correct
        _write_session_file(proj_b, "bbbb2222-0000-0000-0000-000000000002")
        # Give project B a team-checkpoint too (the one that must NOT appear)
        (proj_b / "team-checkpoint.md").write_text("FANNU", encoding="utf-8")

        with (
            patch("cozempic.session.get_projects_dir", return_value=tmp_path / "projects"),
            patch("cozempic.session._session_id_from_process", return_value=None),
            # Block Strategy 1 (active-transcript keyed by live Claude PID)
            # so a real running session in the developer's home cannot bypass strict.
            patch("cozempic.session.find_claude_pid", return_value=None),
        ):
            output = self._run_post_compact(cwd=cwd_a)

        self.assertNotIn(
            "FANNU", output,
            "cmd_post_compact must NOT output fanugugc's checkpoint when cwd=topstep_automation. "
            "Cross-project contamination detected."
        )
        # Output must be either the correct checkpoint or empty (strict→None→Path(cwd) fallback)
        if output.strip():
            self.assertIn(
                "TOPSTEP", output,
                "If cmd_post_compact outputs anything, it must be the current project's checkpoint."
            )

    def test_falls_back_safely_when_no_session_found(self):
        """strict→None→Path(cwd) fallback must not crash and produce no output."""
        tmp_path = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp_path, ignore_errors=True)
        # Empty projects dir — no sessions at all
        (tmp_path / "projects").mkdir(parents=True, exist_ok=True)

        cwd = str(tmp_path / "my_project")
        Path(cwd).mkdir(exist_ok=True)
        # No team-checkpoint.md in cwd

        with (
            patch("cozempic.session.get_projects_dir", return_value=tmp_path / "projects"),
            patch("cozempic.session._session_id_from_process", return_value=None),
        ):
            output = self._run_post_compact(cwd=cwd)

        self.assertEqual(output, "", "cmd_post_compact must be silent when no checkpoint exists.")

    def test_global_checkpoint_not_read_when_local_absent(self):
        """Global ~/.claude/team-checkpoint.md must NOT be returned by cmd_post_compact.

        The global file is a cross-project read vector: it holds the most-recently
        written checkpoint regardless of project. When the resolved project_dir has no
        local checkpoint, cmd_post_compact must be silent (not inject the global file).

        This tests the include_global=False guard added to the read_team_checkpoint call.
        """
        from cozempic.session import get_claude_dir

        tmp_path = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp_path, ignore_errors=True)
        (tmp_path / "projects").mkdir(parents=True, exist_ok=True)

        cwd = str(tmp_path / "my_project")
        Path(cwd).mkdir(exist_ok=True)
        # No local team-checkpoint.md in cwd

        # Place a checkpoint in the global ~/.claude/ location (simulated)
        global_cp = tmp_path / "claude_dir" / "team-checkpoint.md"
        global_cp.parent.mkdir(parents=True, exist_ok=True)
        global_cp.write_text("GLOBAL_CHECKPOINT", encoding="utf-8")

        with (
            patch("cozempic.session.get_projects_dir", return_value=tmp_path / "projects"),
            patch("cozempic.session._session_id_from_process", return_value=None),
            # get_claude_dir is imported inside read_team_checkpoint via `from .session import`
            # so we patch it at the source module level.
            patch("cozempic.session.get_claude_dir", return_value=tmp_path / "claude_dir"),
        ):
            output = self._run_post_compact(cwd=cwd)

        self.assertNotIn(
            "GLOBAL_CHECKPOINT", output,
            "cmd_post_compact must not inject the global team-checkpoint.md. "
            "include_global=False is not being passed to read_team_checkpoint."
        )
        self.assertEqual(
            output, "",
            "cmd_post_compact must be silent when only global checkpoint present."
        )


class TestPostCompactStrategy1Isolation(unittest.TestCase):
    """R-1: Strategy 1 (lookup_active_transcript) must be explicitly blocked.

    test_falls_back_safely and test_global_checkpoint_not_read both use an empty
    projects dir, which incidentally prevents Strategy 1 from running (find_sessions()
    returns [] → early return before Strategy 1 is reached). The isolation is
    implicit, not explicit — a future move of Strategy 1 before the `if not sessions`
    guard, or a test variant that adds sessions, would make both tests non-hermetic.

    This test proves the gap: when a non-empty projects dir allows Strategy 1 to
    run AND lookup_active_transcript returns a fake cross-project session, the
    cmd_post_compact output is wrong (wrong-project checkpoint bleeds in).
    Adding find_claude_pid → None to both tests closes this gap explicitly.
    """

    def _run_post_compact(self, cwd: str) -> str:
        from cozempic.cli import cmd_post_compact
        args = argparse.Namespace(cwd=cwd)
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            cmd_post_compact(args)
        finally:
            sys.stdout = old_stdout
        return captured.getvalue()

    def test_strategy1_injects_wrong_project_when_find_claude_pid_not_patched(self):
        """R-1 (RED at HEAD `ae7fe54`): without find_claude_pid → None, Strategy 1
        fires and injects a wrong-project checkpoint into cmd_post_compact output.

        Setup:
        - projects/ has project B with a session file + checkpoint "PROJ_B_STATE"
        - cmd_post_compact is called with cwd=project_a (no session, no checkpoint)
        - find_claude_pid returns fake PID 99999 (not patched → None as in the fix)
        - lookup_active_transcript returns a fake record pointing at B's session

        Without the find_claude_pid → None patch in test_falls_back_safely and
        test_global_checkpoint_not_read, Strategy 1 would fire from a real live session
        and inject the wrong-project checkpoint.  This test reproduces that failure
        deterministically by injecting both ends of the Strategy 1 chain.

        After fix: test_falls_back_safely + test_global_checkpoint_not_read each
        add `patch("cozempic.session.find_claude_pid", return_value=None)` so that
        lookup_active_transcript → find_claude_pid() → None → returns None,
        Strategy 1 is blocked, and the tests are hermetic regardless of session order.

        This test stays GREEN after the fix (Strategy 1 blocked → no B checkpoint)
        and RED at HEAD (no blocking → B's checkpoint bleeds in).
        """
        tmp_path = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp_path, ignore_errors=True)

        # Project B: non-empty projects dir so find_sessions() returns sessions,
        # allowing Strategy 1 to execute (bypasses the early-exit guard).
        proj_b = tmp_path / "projects" / "-proj-b"
        proj_b.mkdir(parents=True)
        sess_b = "bbbb2222-0000-0000-0000-000000000002"
        sess_b_path = proj_b / f"{sess_b}.jsonl"
        sess_b_path.write_text('{"role":"user","content":"hi"}\n', encoding="utf-8")
        (proj_b / "team-checkpoint.md").write_text("PROJ_B_STATE", encoding="utf-8")

        # Strategy 1 returns B's session: fake active-transcript record for B
        fake_active_record = {
            "transcript_path": str(sess_b_path),
            "pid": 99999,
        }

        # cwd=project_a has no checkpoint — output must be "" for the correct project
        cwd_a = str(tmp_path / "project_a")
        Path(cwd_a).mkdir(exist_ok=True)

        with (
            patch("cozempic.session.get_projects_dir", return_value=tmp_path / "projects"),
            patch("cozempic.session._session_id_from_process", return_value=None),
            # Omit find_claude_pid → None to simulate the gap in the existing tests:
            # find_claude_pid returns a non-None value so lookup_active_transcript
            # proceeds to return fake_active_record.
            patch("cozempic.session.find_claude_pid", return_value=99999),
            patch("cozempic.session.lookup_active_transcript", return_value=fake_active_record),
        ):
            output = self._run_post_compact(cwd=cwd_a)

        # RED at HEAD: Strategy 1 fires, resolves to B's project dir, finds B's
        # checkpoint, and injects it — cmd_post_compact must NOT output B's state
        # when called with cwd=project_a.
        self.assertNotIn(
            "PROJ_B_STATE", output,
            "Strategy 1 injected project B's checkpoint into project A's cmd_post_compact "
            "output. Add find_claude_pid → None to test_falls_back_safely and "
            "test_global_checkpoint_not_read to block Strategy 1 explicitly."
        )


class TestReadTeamCheckpoint(unittest.TestCase):

    def test_returns_content_when_file_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "team-checkpoint.md"
            checkpoint.write_text("# Team State\nTeam: test-team\n", encoding="utf-8")
            result = read_team_checkpoint(Path(tmpdir))
            self.assertIsNotNone(result)
            self.assertIn("Team: test-team", result)

    def test_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = read_team_checkpoint(Path(tmpdir))
            self.assertIsNone(result)

    def test_returns_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "team-checkpoint.md"
            checkpoint.write_text("", encoding="utf-8")
            result = read_team_checkpoint(Path(tmpdir))
            self.assertIsNone(result)

    def test_returns_none_when_whitespace_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "team-checkpoint.md"
            checkpoint.write_text("   \n\n  ", encoding="utf-8")
            result = read_team_checkpoint(Path(tmpdir))
            self.assertIsNone(result)

    def test_prefers_project_dir_over_global(self):
        with tempfile.TemporaryDirectory() as project_dir:
            checkpoint = Path(project_dir) / "team-checkpoint.md"
            checkpoint.write_text("# Project Team", encoding="utf-8")

            # Even if global exists, project dir should win
            result = read_team_checkpoint(Path(project_dir))
            self.assertEqual(result, "# Project Team")

    def test_falls_back_to_none_when_dir_missing(self):
        # include_global=False: prevents this test from accidentally passing by
        # reading the developer's real ~/.claude/team-checkpoint.md when present.
        result = read_team_checkpoint(Path("/nonexistent/dir"), include_global=False)
        self.assertIsNone(
            result,
            "read_team_checkpoint must return None when project_dir doesn't exist "
            "and include_global=False."
        )


class TestCmdPostCompact(unittest.TestCase):

    @patch("cozempic.team.read_team_checkpoint")
    @patch("cozempic.session.find_current_session")
    def test_outputs_recovery_when_checkpoint_exists(self, mock_session, mock_read):
        from cozempic.cli import cmd_post_compact
        import argparse

        mock_session.return_value = {
            "path": Path("/fake/project/session.jsonl"),
            "session_id": "test-123",
        }
        mock_read.return_value = "# Team State\nTeam: recovery-test"

        args = argparse.Namespace(cwd=None)
        captured = io.StringIO()
        sys.stdout = captured
        try:
            cmd_post_compact(args)
        finally:
            sys.stdout = sys.__stdout__

        self.assertIn("Team: recovery-test", captured.getvalue())

    @patch("cozempic.team.read_team_checkpoint")
    @patch("cozempic.session.find_current_session")
    def test_silent_when_no_checkpoint(self, mock_session, mock_read):
        from cozempic.cli import cmd_post_compact
        import argparse

        mock_session.return_value = {
            "path": Path("/fake/project/session.jsonl"),
            "session_id": "test-123",
        }
        mock_read.return_value = None

        args = argparse.Namespace(cwd=None)
        captured = io.StringIO()
        sys.stdout = captured
        try:
            cmd_post_compact(args)
        finally:
            sys.stdout = sys.__stdout__

        self.assertEqual(captured.getvalue(), "")


class TestInitHooksIncludePostCompact(unittest.TestCase):

    def test_post_compact_in_cozempic_hooks(self):
        self.assertIn("PostCompact", COZEMPIC_HOOKS)

    def test_post_compact_hook_command_correct(self):
        entries = COZEMPIC_HOOKS["PostCompact"]
        self.assertEqual(len(entries), 1)

        hooks = entries[0]["hooks"]
        self.assertEqual(len(hooks), 1)

        command = hooks[0]["command"]
        self.assertIn("cozempic post-compact", command)

    def test_pre_compact_still_exists(self):
        """Ensure PreCompact wasn't accidentally removed."""
        self.assertIn("PreCompact", COZEMPIC_HOOKS)

    def test_all_expected_hooks_present(self):
        """Verify all expected hook events are defined."""
        expected = {"SessionStart", "PostToolUse", "PreCompact", "PostCompact", "Stop"}
        self.assertEqual(expected, set(COZEMPIC_HOOKS.keys()))


if __name__ == "__main__":
    unittest.main()
