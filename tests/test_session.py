"""Tests for session module path helpers."""

from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace

import pytest

from pathlib import Path
from unittest.mock import patch

from cozempic.session import (
    MAX_LINE_BYTES,
    cwd_to_project_slug,
    find_claude_pid,
    get_claude_dir,
    get_claude_json_path,
    load_messages,
)


# ---------------------------------------------------------------------------
# TestCwdToProjectSlug — slug parity with real ~/.claude/projects/ dir names
# ---------------------------------------------------------------------------

class TestCwdToProjectSlug:
    """Table-driven slug parity. Every non-alphanumeric char → '-', 1:1."""

    def test_underscore_maps_to_dash(self):
        """Bug A regression: '_' must become '-', not stay '_'."""
        result = cwd_to_project_slug("/Users/x/topstep_automation")
        assert result == "-Users-x-topstep-automation", (
            f"Expected '-Users-x-topstep-automation', got {result!r}. "
            "Underscore is not being replaced."
        )

    def test_dot_maps_to_double_dash(self):
        """'.claude' produces double-dash because '.' → '-' and leading '/' → '-'."""
        result = cwd_to_project_slug("/Users/x/.claude")
        assert result == "-Users-x--claude", (
            f"Expected '-Users-x--claude', got {result!r}. "
            "Dot is not being replaced."
        )

    def test_worktree_dotclaude_in_path(self):
        """Worktrees whose path contains '.claude' must produce correct slug."""
        result = cwd_to_project_slug("/Users/x/cozempic/.claude/worktrees/fix-slug")
        assert result == "-Users-x-cozempic--claude-worktrees-fix-slug", (
            f"Got {result!r}. Dot in path component not replaced."
        )

    def test_plain_path_identity(self):
        """Plain paths (no underscores or dots) must be unchanged by fix."""
        result = cwd_to_project_slug("/Users/x/myproject")
        assert result == "-Users-x-myproject"

    def test_space_maps_to_dash(self):
        """Spaces must become dashes (Claude normalises all non-alnum chars)."""
        result = cwd_to_project_slug("/Users/x/my project")
        assert result == "-Users-x-my-project", (
            f"Got {result!r}. Space not replaced."
        )

    def test_digits_preserved(self):
        """Digits are alphanumeric and must not be replaced."""
        result = cwd_to_project_slug("/Users/x/project123")
        assert result == "-Users-x-project123"

    def test_existing_hyphens_preserved(self):
        """Already-hyphenated paths must remain hyphenated (idempotent on '-')."""
        result = cwd_to_project_slug("/Users/x/my-project")
        assert result == "-Users-x-my-project"

    def test_trailing_slash_normalized(self):
        """normpath strips trailing slash before regex — no trailing dash in output."""
        result = cwd_to_project_slug("/Users/x/myproject/")
        assert result == "-Users-x-myproject", (
            f"Got {result!r}. normpath should have stripped the trailing slash."
        )

    def test_dot_segment_normalized(self):
        """`./` and embedded `.` segments are collapsed by normpath."""
        result = cwd_to_project_slug("/Users/x/./p")
        assert result == "-Users-x-p", (
            f"Got {result!r}. normpath should collapse the dot segment."
        )

    def test_double_slash_normalized(self):
        """Double slashes are collapsed by normpath."""
        result = cwd_to_project_slug("/Users/x//p")
        assert result == "-Users-x-p", (
            f"Got {result!r}. normpath should collapse double slash."
        )

    def test_none_cwd_uses_os_getcwd(self):
        """None cwd delegates to os.getcwd()."""
        with patch("cozempic.session.os.getcwd", return_value="/Users/x/foo"):
            result = cwd_to_project_slug(None)
        assert result == "-Users-x-foo"

    def test_unicode_char_maps_to_dash(self):
        """Non-ASCII characters (e.g. accented letters) must be replaced."""
        # re.sub(r'[^a-zA-Z0-9]', '-', ...) replaces 'é' with '-'
        result = cwd_to_project_slug("/Users/x/café")
        expected = re.sub(r"[^a-zA-Z0-9]", "-", "/Users/x/café")
        assert result == expected, (
            f"Got {result!r}, expected {expected!r}. Unicode char not replaced."
        )


class TestGetClaudeDir:
    def test_default(self):
        with patch.dict("os.environ", {}, clear=True):
            assert get_claude_dir() == Path.home() / ".claude"

    def test_with_config_dir(self, tmp_path):
        with patch.dict("os.environ", {"CLAUDE_CONFIG_DIR": str(tmp_path)}):
            assert get_claude_dir() == tmp_path


class TestGetClaudeJsonPath:
    def test_default(self):
        with patch.dict("os.environ", {}, clear=True):
            assert get_claude_json_path() == Path.home() / ".claude.json"

    def test_with_config_dir(self, tmp_path):
        with patch.dict("os.environ", {"CLAUDE_CONFIG_DIR": str(tmp_path)}):
            assert get_claude_json_path() == tmp_path / ".claude.json"

    def test_not_inside_claude_dir(self):
        """Default .claude.json is at ~/.claude.json, not ~/.claude/.claude.json."""
        with patch.dict("os.environ", {}, clear=True):
            assert get_claude_json_path() != get_claude_dir() / ".claude.json"


class TestLoadMessagesLimits:
    def test_skips_oversized_lines(self, tmp_path):
        """Lines exceeding MAX_LINE_BYTES are silently skipped."""
        jsonl = tmp_path / "test.jsonl"
        normal = json.dumps({"role": "user", "content": "hello"})
        oversized = json.dumps({"role": "user", "content": "x" * (MAX_LINE_BYTES + 1)})
        jsonl.write_text(normal + "\n" + oversized + "\n")
        messages = load_messages(jsonl)
        assert len(messages) == 1
        assert messages[0][1]["content"] == "hello"

    def test_normal_lines_unaffected(self, tmp_path):
        """Normal-sized lines parse correctly."""
        jsonl = tmp_path / "test.jsonl"
        lines = [
            json.dumps({"role": "user", "content": "first"}),
            json.dumps({"role": "assistant", "content": "second"}),
        ]
        jsonl.write_text("\n".join(lines) + "\n")
        messages = load_messages(jsonl)
        assert len(messages) == 2
        assert messages[0][1]["content"] == "first"
        assert messages[1][1]["content"] == "second"


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX ps ancestry walk — Windows uses a Toolhelp32 snapshot "
           "(see tests/test_windows_lifecycle.py::TestWalkUpToClaude)",
)
class TestFindClaudePid:
    def test_finds_claude_process_in_ancestor_chain(self):
        with (
            patch("cozempic.session.os.getpid", return_value=400),
            patch(
                "cozempic.session.subprocess.run",
                side_effect=[
                    SimpleNamespace(stdout="300 python\n"),
                    SimpleNamespace(stdout="200 node\n"),
                ],
            ),
        ):
            assert find_claude_pid() == 300

    def test_returns_none_when_detached_guard_parent_is_systemd(self):
        with (
            patch("cozempic.session.os.getpid", return_value=400),
            patch(
                "cozempic.session.subprocess.run",
                side_effect=[
                    SimpleNamespace(stdout="300 python\n"),
                    SimpleNamespace(stdout="1 systemd\n"),
                ],
            ),
        ):
            assert find_claude_pid() is None
