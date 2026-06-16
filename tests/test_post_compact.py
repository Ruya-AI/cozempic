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

from cozempic.session import cwd_to_project_slug, get_claude_dir
from cozempic.team import read_team_checkpoint
from cozempic.init import COZEMPIC_HOOKS


def _run_post_compact(cwd: str) -> str:
    """Capture cmd_post_compact stdout for the given cwd."""
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

    def test_does_not_return_other_projects_checkpoint_when_other_is_newer(self):
        """Core bug: Strategy 5 picks a newer OTHER project's session → wrong checkpoint.

        Fixture uses the CORRECT dir names (as Claude Code actually creates them, with dashes
        for underscores). Old code computes broken slug with '_', so Strategy 4 misses project A
        and Strategy 5 returns project B's (newer) session → contamination.
        """
        tmp_path = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp_path, ignore_errors=True)

        # Project A: topstep_automation — dir name uses dashes (Claude's real format)
        cwd_a = "/Users/x/topstep_automation"
        proj_a = tmp_path / "projects" / cwd_to_project_slug(cwd_a)   # "-Users-x-topstep-automation"
        _write_session_file(proj_a, "aaaa1111-0000-0000-0000-000000000001")
        # Write a checkpoint for project A
        (proj_a / "team-checkpoint.md").write_text("TOPSTEP", encoding="utf-8")

        # Small sleep ensures project B mtime is strictly newer
        time.sleep(0.01)

        # Project B: fanugugc (no underscore → still returned by Strategy 5 when A is missed)
        cwd_b = "/Users/x/fanugugc"
        slug_b_correct = cwd_to_project_slug(cwd_b)   # "-Users-x-fanugugc"
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
            output = _run_post_compact(cwd=cwd_a)

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
            # Explicit isolation: block Strategy 1 so a real live Claude session
            # on the host cannot inject an active-transcript record.
            patch("cozempic.session.find_claude_pid", return_value=None),
        ):
            output = _run_post_compact(cwd=cwd)

        self.assertEqual(output, "", "cmd_post_compact must be silent when no checkpoint exists.")

    def test_global_checkpoint_not_read_when_local_absent(self):
        """Global ~/.claude/team-checkpoint.md must NOT be returned by cmd_post_compact.

        The global file is a cross-project read vector: it holds the most-recently
        written checkpoint regardless of project. When the resolved project_dir has no
        local checkpoint, cmd_post_compact must be silent (not inject the global file).

        This tests the include_global=False guard added to the read_team_checkpoint call.
        """
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
            # Explicit isolation: block Strategy 1 so a real live Claude session
            # on the host cannot inject an active-transcript record.
            patch("cozempic.session.find_claude_pid", return_value=None),
            # get_claude_dir is imported inside read_team_checkpoint via `from .session import`
            # so we patch it at the source module level.
            patch("cozempic.session.get_claude_dir", return_value=tmp_path / "claude_dir"),
        ):
            output = _run_post_compact(cwd=cwd)

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
    """R-1: cmd_post_compact must be hermetic when find_claude_pid is blocked.

    Without an explicit find_claude_pid → None patch, a live Claude session on the
    host can make Strategy 1 fire and return a wrong-project checkpoint.

    RED at HEAD `ae7fe54`: find_claude_pid → None was absent from test_falls_back_safely
    and test_global_checkpoint_not_read → those tests were incidentally safe only because
    their empty projects dir triggered an early return before Strategy 1 ran.
    GREEN after fix: both tests gain find_claude_pid → None; this guard also patches it.
    """

    def test_strategy1_blocked_when_find_claude_pid_is_none(self):
        """R-1 guard: with find_claude_pid → None, Strategy 1 is explicitly blocked
        and a cross-project session cannot contaminate cmd_post_compact output.

        RED at HEAD `ae7fe54`: the existing test_falls_back_safely and
        test_global_checkpoint_not_read do NOT patch find_claude_pid → None.
        A real live session on the host could make lookup_active_transcript return a
        record pointing at another project's session, injecting its checkpoint.

        This test explicitly patches find_claude_pid → None and asserts that even
        when a cross-project session file + checkpoint exist in the projects dir,
        cmd_post_compact output is "" for a different cwd.

        GREEN after fix: both production tests gain find_claude_pid → None.
        """
        tmp_path = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp_path, ignore_errors=True)

        # Non-empty projects dir: project B has a session + checkpoint.
        # Strategy 1 CAN fire here (find_sessions returns non-empty).
        proj_b = tmp_path / "projects" / "-proj-b"
        sess_b = "bbbb2222-0000-0000-0000-000000000002"
        _write_session_file(proj_b, sess_b)
        (proj_b / "team-checkpoint.md").write_text("PROJ_B_STATE", encoding="utf-8")

        # cwd=project_a — no local session, no checkpoint
        cwd_a_path = tmp_path / "project_a"
        cwd_a_path.mkdir(exist_ok=True)
        cwd_a = str(cwd_a_path)

        with (
            patch("cozempic.session.get_projects_dir", return_value=tmp_path / "projects"),
            patch("cozempic.session._session_id_from_process", return_value=None),
            # Explicit isolation: find_claude_pid → None blocks Strategy 1
            # so lookup_active_transcript returns None → no cross-project leak.
            patch("cozempic.session.find_claude_pid", return_value=None),
        ):
            output = _run_post_compact(cwd=cwd_a)

        self.assertEqual(
            output, "",
            "cmd_post_compact must be silent for project_a even when project_b has "
            "a session and checkpoint in the projects dir. Strategy 1 must be blocked "
            "via find_claude_pid → None."
        )

class TestCorrectSlugUsesCwdToProjectSlug(unittest.TestCase):
    """Characterization of cwd_to_project_slug normpath behavior (post-P0-C).

    P0-C replaced 3 inline `re.sub(r"[^a-zA-Z0-9]", "-", cwd)` helpers with
    direct cwd_to_project_slug calls.  The canonical function applies normpath
    before slug-ifying, so trailing-slash inputs are normalized correctly.

    This class holds a characterization test (not a behavioral RED/GREEN guard)
    documenting the normpath contract the helpers now rely on.
    """

    def test_cwd_to_project_slug_normalizes_trailing_slash(self):
        """Characterization: cwd_to_project_slug strips trailing slashes via normpath.

        P0-C replaced the 3 inline `re.sub(r"[^a-zA-Z0-9]", "-", cwd)` helpers with
        direct calls to cwd_to_project_slug.  The value of that swap is DRY + normpath-
        correctness: the old inline formula produced "-Users-x-proj-" for trailing-slash
        inputs (normpath not applied), while cwd_to_project_slug applies normpath first
        and returns "-Users-x-proj".

        This test characterizes the canonical behavior the helpers now rely on.  It is
        NOT a behavioral RED/GREEN guard for the swap itself — cwd_to_project_slug is
        production code that was already correct before P0-C; the swap's correctness is
        verified by the diff (3 `re.sub` sites → `cwd_to_project_slug`), not by this test.

        Canonical behavior:
        - re.sub(r"[^a-zA-Z0-9]", "-", "/Users/x/proj/") → "-Users-x-proj-" (old inline, wrong)
        - cwd_to_project_slug("/Users/x/proj/")           → "-Users-x-proj"  (canonical)
        """
        slug = cwd_to_project_slug("/Users/x/proj/")
        self.assertEqual(
            slug, "-Users-x-proj",
            f"cwd_to_project_slug must strip trailing slash via normpath. Got {slug!r}."
        )
        self.assertFalse(
            slug.endswith("-"),
            f"cwd_to_project_slug must not produce a trailing '-' for trailing-slash inputs. "
            f"Got {slug!r}."
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
