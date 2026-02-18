"""Tests for session module path helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from cozempic.session import get_claude_dir, get_claude_json_path


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
