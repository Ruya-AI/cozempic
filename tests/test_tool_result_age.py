"""Tests for tool-result-age strategy — age-based tool result compaction."""

from __future__ import annotations

import json
import unittest

from cozempic.helpers import msg_bytes
from cozempic.registry import STRATEGIES

import cozempic.strategies  # noqa: F401


def make_message(line_idx: int, msg: dict) -> tuple[int, dict, int]:
    return (line_idx, msg, msg_bytes(msg))


def make_user(line_idx: int, text: str = "hi") -> tuple[int, dict, int]:
    return make_message(line_idx, {
        "type": "user",
        "message": {"role": "user", "content": text},
    })


def make_tool_use(line_idx: int, tool_id: str, name: str = "Read",
                  input_data: dict | None = None) -> tuple[int, dict, int]:
    return make_message(line_idx, {
        "type": "assistant",
        "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": tool_id, "name": name,
             "input": input_data or {"file_path": "/src/app.py"}},
        ]},
    })


def make_tool_result(line_idx: int, tool_id: str, content: str = "x" * 500) -> tuple[int, dict, int]:
    return make_message(line_idx, {
        "type": "user",
        "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tool_id, "content": content},
        ]},
    })


def _build_session(num_turns: int = 60, result_size: int = 1000) -> list:
    """Build a session with num_turns user turns and tool calls throughout."""
    messages = []
    idx = 0
    for turn in range(num_turns):
        messages.append(make_user(idx, f"turn {turn}"))
        idx += 1
        tid = f"t{turn}"
        messages.append(make_tool_use(idx, tid, "Read", {"file_path": f"/src/file{turn}.py"}))
        idx += 1
        messages.append(make_tool_result(idx, tid, "x" * result_size))
        idx += 1
    return messages


class TestMinifyDiffGate(unittest.TestCase):
    """The diff-collapse must only fire on a GENUINE unified diff. The old gate
    over-triggered on any content containing '\\n@@', then collapsed every
    space-indented line into '[...unchanged...]' = silent data loss (audit P1)."""

    def test_real_diff_still_collapses(self):
        from cozempic.strategies.standard import _minify_tool_content
        ctx = "".join(f" unchanged context line number {i}\n" for i in range(12))
        diff = "--- a/x.py\n+++ b/x.py\n@@ -1,14 +1,14 @@\n" + ctx + "-old\n+new\n"
        out = _minify_tool_content(diff)
        self.assertIn("unchanged lines", out, "a real unified diff must still collapse context")
        self.assertIn("+new", out)

    def test_single_fake_hunk_line_with_indented_logs_preserved(self):
        # Fleet repro: ONE hunk-shaped line in non-diff output (no ---/+++ envelope)
        # must NOT open the gate; indented log lines must survive verbatim.
        from cozempic.strategies.standard import _minify_tool_content
        content = (
            "Replaying journal @@ -1 +1 @@ marker found:\n"
            "   ERROR connection refused to db-primary\n"
            "   ERROR connection refused to db-replica\n"
            "   WARN retry budget exhausted\n"
            "   INFO falling back to cache\n"
            "   INFO request completed in 4.2s\n"
            "done\n"
        )
        self.assertEqual(_minify_tool_content(content), content,
                         "a lone hunk-shaped line must not trigger collapse of indented logs")

    def test_git_log_p_second_commit_body_preserved(self):
        # Fleet P1: content after a hunk (a git-log-p second commit's indented
        # message body) must survive — in_hunk must reset after the hunk ends.
        from cozempic.strategies.standard import _minify_tool_content
        ctx = "".join(f" ctx line {i}\n" for i in range(12))
        content = (
            "commit abc123\n"
            "--- a/x.py\n+++ b/x.py\n@@ -1,14 +1,14 @@\n" + ctx + "-old\n+new\n"
            "\n"
            "commit def456\n"
            "Author: someone\n"
            "\n"
            "    SECOND_COMMIT_BODY_LINE_DO_NOT_LOSE\n"
            "    more indented body text that must survive\n"
        )
        out = _minify_tool_content(content)
        self.assertIn("unchanged lines", out, "the real hunk must still collapse")
        self.assertIn("SECOND_COMMIT_BODY_LINE_DO_NOT_LOSE", out,
                      "content after the hunk must NOT be collapsed away")

    def test_indented_config_after_fake_hunk_preserved(self):
        from cozempic.strategies.standard import _minify_tool_content
        content = (
            "@@ -1 +1 @@\n"
            "   api_key = SECRET_DO_NOT_LOSE\n"
            "   host = db-primary\n"
            "   port = 5432\n"
            "   timeout = 30\n"
            "   retries = 5\n"
        )
        out = _minify_tool_content(content)
        self.assertIn("SECRET_DO_NOT_LOSE", out, "indented config must never be collapsed away")
        self.assertEqual(out, content)

    def test_non_diff_with_at_at_substring_preserved(self):
        # Prose/log that merely contains '@@' and space-indented lines must be
        # returned VERBATIM — not run through the diff collapser.
        from cozempic.strategies.standard import _minify_tool_content
        content = (
            "Decorator usage:\n"
            "  @@app.route\n"           # '@@' but NOT a hunk header
            "   indented code line one\n"  # leading spaces — would be collapsed by the old gate
            "   indented code line two\n"
            "   indented code line three\n"
            "Done.\n"
        )
        self.assertEqual(_minify_tool_content(content), content,
                         "non-diff content must be preserved verbatim, never collapsed")

    def test_indented_prose_block_not_destroyed(self):
        from cozempic.strategies.standard import _minify_tool_content
        content = "Log output:\n" + "".join(f"   line {i}\n" for i in range(40)) + "@@ note @@\n"
        self.assertEqual(_minify_tool_content(content), content)


class TestToolResultAge(unittest.TestCase):

    def test_recent_results_untouched(self):
        """Tool results within mid_age turns should not be modified."""
        messages = _build_session(num_turns=10, result_size=500)
        sr = STRATEGIES["tool-result-age"].func(messages, {"tool_result_mid_age": 15})
        self.assertEqual(sr.messages_affected, 0)

    def test_old_results_stubbed(self):
        """Tool results older than old_age turns should be replaced with stubs."""
        messages = _build_session(num_turns=60, result_size=2000)
        sr = STRATEGIES["tool-result-age"].func(messages, {
            "tool_result_mid_age": 10,
            "tool_result_old_age": 30,
        })
        self.assertGreater(sr.messages_affected, 0)
        # Check that stubs contain the expected format
        for action in sr.actions:
            if action.replacement and "tool-result-age" in action.reason:
                content_blocks = action.replacement.get("message", {}).get("content", [])
                for block in content_blocks:
                    if block.get("type") == "tool_result" and "[cozempic" in block.get("content", ""):
                        self.assertIn("lines", block["content"])
                        self.assertIn("KB]", block["content"])

    def test_mid_age_json_minified(self):
        """Mid-age tool results with JSON content should be minified."""
        pretty_json = json.dumps({"key": "value", "nested": {"a": 1, "b": 2}}, indent=2)
        messages = _build_session(num_turns=30, result_size=100)
        # Replace one result with pretty JSON at a mid-age position
        for i, (idx, msg, size) in enumerate(messages):
            content = msg.get("message", {}).get("content", [])
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        block["content"] = pretty_json
                        messages[i] = make_message(idx, msg)
                        break

        sr = STRATEGIES["tool-result-age"].func(messages, {
            "tool_result_mid_age": 10,
            "tool_result_old_age": 50,
        })
        # Some mid-age results should be minified
        for action in sr.actions:
            if action.replacement:
                content_blocks = action.replacement.get("message", {}).get("content", [])
                for block in content_blocks:
                    if block.get("type") == "tool_result":
                        c = block.get("content", "")
                        if c.startswith("{"):
                            self.assertNotIn("\n", c, "JSON should be minified")

    def test_mid_age_diff_collapsed(self):
        """Mid-age diff results should have context lines collapsed."""
        diff_content = """diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -10,7 +10,7 @@
 unchanged line 1
 unchanged line 2
 unchanged line 3
-old line
+new line
 unchanged line 4
 unchanged line 5
 unchanged line 6"""

        messages = []
        # 30 turns of filler
        idx = 0
        for t in range(30):
            messages.append(make_user(idx, f"turn {t}"))
            idx += 1
        # Early tool call with diff result
        messages.insert(3, make_tool_use(900, "tdiff", "Bash", {"command": "git diff"}))
        messages.insert(4, make_tool_result(901, "tdiff", diff_content))

        sr = STRATEGIES["tool-result-age"].func(messages, {
            "tool_result_mid_age": 10,
            "tool_result_old_age": 50,
        })
        for action in sr.actions:
            if action.replacement:
                content_blocks = action.replacement.get("message", {}).get("content", [])
                for block in content_blocks:
                    if block.get("type") == "tool_result" and "unchanged" not in block.get("content", ""):
                        self.assertIn("+new line", block["content"])
                        self.assertIn("-old line", block["content"])

    def test_small_results_skipped(self):
        """Tool results under 100 chars should not be touched regardless of age."""
        messages = _build_session(num_turns=60, result_size=50)  # Small results
        sr = STRATEGIES["tool-result-age"].func(messages, {
            "tool_result_mid_age": 5,
            "tool_result_old_age": 10,
        })
        self.assertEqual(sr.messages_affected, 0)

    def test_protected_messages_skipped(self):
        """Protected messages should never be modified."""
        messages = _build_session(num_turns=60, result_size=2000)
        # Mark one old message as protected
        old_idx = 6  # Turn 2, should be old
        _, old_msg, _ = messages[old_idx]
        old_msg["isCompactSummary"] = True
        messages[old_idx] = make_message(messages[old_idx][0], old_msg)

        sr = STRATEGIES["tool-result-age"].func(messages, {})
        for action in sr.actions:
            self.assertNotEqual(action.line_index, messages[old_idx][0])

    def test_stub_includes_tool_name(self):
        """Stubs should include the tool name from the matching tool_use."""
        messages = []
        idx = 0
        for t in range(50):
            messages.append(make_user(idx, f"turn {t}"))
            idx += 1
        # Insert a Read tool call early
        messages.insert(2, make_tool_use(500, "tread", "Read", {"file_path": "/src/main.py"}))
        messages.insert(3, make_tool_result(501, "tread", "x" * 2000))

        sr = STRATEGIES["tool-result-age"].func(messages, {
            "tool_result_mid_age": 5,
            "tool_result_old_age": 10,
        })
        found_stub = False
        for action in sr.actions:
            if action.replacement:
                for block in action.replacement.get("message", {}).get("content", []):
                    if block.get("type") == "tool_result" and "[cozempic" in block.get("content", ""):
                        self.assertIn("Read", block["content"])
                        self.assertIn("/src/main.py", block["content"])
                        found_stub = True
        self.assertTrue(found_stub, "Should have found a stub with tool name")

    def test_savings_significant(self):
        """On a realistic session, savings should be meaningful."""
        messages = _build_session(num_turns=60, result_size=3000)
        sr = STRATEGIES["tool-result-age"].func(messages, {})
        self.assertGreater(sr.pruned_bytes, 50000, "Should save >50KB on 60-turn session")

    def test_in_registry(self):
        """Strategy should be registered."""
        self.assertIn("tool-result-age", STRATEGIES)

    def test_in_standard_prescription(self):
        """Strategy should be in standard and aggressive prescriptions."""
        from cozempic.registry import PRESCRIPTIONS
        self.assertIn("tool-result-age", PRESCRIPTIONS["standard"])
        self.assertIn("tool-result-age", PRESCRIPTIONS["aggressive"])
        self.assertNotIn("tool-result-age", PRESCRIPTIONS["gentle"])


if __name__ == "__main__":
    unittest.main()
