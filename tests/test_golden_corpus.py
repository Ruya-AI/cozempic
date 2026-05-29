"""Golden corpus regression tests.

Each fixture in tests/fixtures/sessions/ covers a real-world session shape.
These tests verify that strategies produce correct outcomes and never regress.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "sessions"


def load_fixture(name: str):
    from cozempic.session import load_messages
    return load_messages(FIXTURES / name)


# ─── solo_bloated.jsonl ───────────────────────────────────────────────────────

def test_standard_rx_reduces_bloated_session():
    """Standard prescription must cut a bloated session to under 60% of original size."""
    from cozempic.registry import PRESCRIPTIONS
    from cozempic.executor import run_prescription

    messages = load_fixture("solo_bloated.jsonl")
    original_bytes = sum(b for _, _, b in messages)

    new_messages, _ = run_prescription(messages, PRESCRIPTIONS["standard"], {})
    final_bytes = sum(b for _, _, b in new_messages)

    ratio = final_bytes / original_bytes
    assert ratio < 0.85, f"Standard rx only reduced to {ratio:.1%} — expected < 85%"


def test_aggressive_rx_reduces_more_than_standard():
    from cozempic.registry import PRESCRIPTIONS
    from cozempic.executor import run_prescription

    messages = load_fixture("solo_bloated.jsonl")
    std_msgs, _ = run_prescription(messages, PRESCRIPTIONS["standard"], {})
    agg_msgs, _ = run_prescription(messages, PRESCRIPTIONS["aggressive"], {})

    assert sum(b for _, _, b in agg_msgs) <= sum(b for _, _, b in std_msgs), \
        "Aggressive must produce equal or smaller output than standard"


# ─── team_two_subagents.jsonl ─────────────────────────────────────────────────

def test_team_protect_preserves_both_task_results():
    """prune_with_team_protect must keep both Task tool_result messages intact."""
    from cozempic.guard import prune_with_team_protect

    messages = load_fixture("team_two_subagents.jsonl")
    pruned, _, team_state = prune_with_team_protect(messages, rx_name="standard")

    # Both critical subagent results must survive
    result_texts = []
    for _, msg, _ in pruned:
        content = msg.get("message", {}).get("content", [])
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result":
                    c = block.get("content", "")
                    if isinstance(c, str):
                        result_texts.append(c)

    assert any("CRITICAL RESULT FROM SUBAGENT 1" in t for t in result_texts), \
        "Subagent 1 result was pruned"
    assert any("CRITICAL RESULT FROM SUBAGENT 2" in t for t in result_texts), \
        "Subagent 2 result was pruned"


# ─── corrupted_tool_use.jsonl ─────────────────────────────────────────────────

def test_fix_corrupted_tool_use_repairs_and_produces_valid_jsonl(tmp_path):
    """fix_corrupted_tool_use must repair the corrupted block and write valid JSONL."""
    import shutil
    from cozempic.doctor import fix_corrupted_tool_use, check_corrupted_tool_use
    from cozempic.session import get_projects_dir

    # Copy fixture into a temp projects dir
    proj_dir = tmp_path / "projects" / "test-proj"
    proj_dir.mkdir(parents=True)
    dest = proj_dir / "sess.jsonl"
    shutil.copy(FIXTURES / "corrupted_tool_use.jsonl", dest)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cozempic.session.get_projects_dir", lambda: tmp_path / "projects")

        check_result = check_corrupted_tool_use()
        assert check_result.status == "issue", "Should detect corrupted block"

        fix_corrupted_tool_use()

        # Output must be valid JSONL
        for line in dest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                json.loads(line)

        # Repaired block must have name <= 200 chars
        from cozempic.session import load_messages
        for _, msg, _ in load_messages(dest):
            content = msg.get("message", {}).get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    assert len(block.get("name", "")) <= 200, "Block still corrupted after fix"


# ─── orphaned_tool_results.jsonl ─────────────────────────────────────────────

def test_fix_orphaned_tool_results_removes_orphan(tmp_path):
    import shutil
    from cozempic.doctor import check_orphaned_tool_results, fix_orphaned_tool_results

    proj_dir = tmp_path / "projects" / "test-proj"
    proj_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "orphaned_tool_results.jsonl", proj_dir / "sess.jsonl")

    # Set mtime to 25 hours ago so the idle gate (default 24h) passes.
    # PR #102 P4 added an idle gate to fix_orphaned_tool_results; a freshly
    # copied fixture would be refused without this.
    import os as _os, time as _time
    _sess_path = proj_dir / "sess.jsonl"
    _old_mtime = _time.time() - 25 * 3600
    _os.utime(_sess_path, (_old_mtime, _old_mtime))

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cozempic.session.get_projects_dir", lambda: tmp_path / "projects")

        check_result = check_orphaned_tool_results()
        assert check_result.status == "issue"

        fix_orphaned_tool_results()

        from cozempic.session import load_messages
        messages = load_messages(proj_dir / "sess.jsonl")
        for _, msg, _ in messages:
            content = msg.get("message", {}).get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict):
                    assert block.get("type") != "tool_result" or block.get("tool_use_id") != "missing-tool-use-id"
