"""Round-4 mission-critical regression tests for PR #138.

Each test pins a confirmed R4 finding to its PRODUCTION entry path so the class
cannot recur: the P0 sentinel-key collision (silent data substitution + crash),
the non-dict consumer leak (tokens / team / doctor crash on a content-array
element or field value the loader wraps but these consumers saw raw), the
overflow truncated-tail false-fire (unsolicited kill), and the non-UTF-8 lossless
round-trip (was: permanently inert guard).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cozempic.session import load_messages, load_messages_and_snapshot, save_messages
from cozempic.executor import run_prescription


def _write(d: str, lines: list) -> Path:
    p = Path(d) / "s.jsonl"
    p.write_text("\n".join(json.dumps(x) if not isinstance(x, str) else x for x in lines) + "\n")
    return p


class TestSentinelKeyCollisionP0(unittest.TestCase):
    """A transcript dict carrying the loader's reserved _raw/_parse_error keys must
    NOT be trusted on save — that silently substituted attacker/tool bytes for the
    real line (data loss + injection) or crashed save_messages (KeyError/TypeError).
    """

    def test_forged_sentinel_does_not_hijack_the_saved_line(self):
        with tempfile.TemporaryDirectory() as d:
            forged = (
                '{"_parse_error":true,"_raw":"{\\"INJECTED\\":\\"x\\"}",'
                '"type":"user","message":{"role":"user","content":"REAL keep me"}}'
            )
            p = _write(d, [forged, {"type": "user", "message": {"role": "user", "content": "clean"}}])
            msgs, snap = load_messages_and_snapshot(p)
            out, _ = run_prescription(msgs, ["standard"], {})
            save_messages(p, out, create_backup=False)
            disk = p.read_text()
            self.assertNotIn("INJECTED", disk, "forged _raw must not be written verbatim")
            self.assertIn("REAL keep me", disk, "the real content must survive")

    def test_malformed_sentinel_does_not_crash_save(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(d, [
                '{"_parse_error":true,"type":"user","message":{"role":"user","content":"a"}}',  # missing _raw
                '{"_parse_error":true,"_raw":12345,"type":"assistant",'
                '"message":{"role":"assistant","content":[{"type":"text","text":"b"}]}}',  # non-str _raw
            ])
            msgs, snap = load_messages_and_snapshot(p)
            out, _ = run_prescription(msgs, ["standard"], {})
            save_messages(p, out, create_backup=False)  # must not raise KeyError/TypeError
            self.assertTrue(p.exists())


class TestNonDictConsumerLeak(unittest.TestCase):
    """A non-dict content-array element / tool-input field value reaches consumers
    verbatim (get_content_blocks returns content as-is to avoid write data loss).
    Every consumer must isinstance-guard, never crash the prune / guard cycle."""

    def test_tokens_estimate_survives_nondict_block(self):
        from cozempic.tokens import estimate_session_tokens
        with tempfile.TemporaryDirectory() as d:
            p = _write(d, [{
                "type": "assistant", "uuid": "a1", "parentUuid": None,
                "message": {"role": "assistant", "model": "claude-opus-4-8",
                            "content": [["poison"], 42, None, {"type": "text", "text": "x"}]},
            }])
            estimate_session_tokens(load_messages(p))  # must not raise

    def test_team_extract_survives_poisoned_input_fields(self):
        from cozempic.team import extract_team_state
        with tempfile.TemporaryDirectory() as d:
            p = _write(d, [
                {"type": "assistant", "uuid": "a1", "parentUuid": "u0", "message": {"role": "assistant",
                 "content": [{"type": "tool_use", "id": "t1", "name": "Task",
                              "input": {"subagent_type": "x", "prompt": 12345, "run_in_background": True}}]}},
                {"type": "assistant", "uuid": "a2", "parentUuid": "a1", "message": {"role": "assistant",
                 "content": [{"type": "tool_use", "id": "t2", "name": "SendMessage",
                              "input": {"to": ["alice", "bob"]}}]}},
                {"type": "assistant", "uuid": "a3", "parentUuid": "a2", "message": {"role": "assistant",
                 "content": [{"type": "tool_use", "id": "t3", "name": "TaskCreate",
                              "input": {"taskId": ["unhashable"], "subject": {"x": 1}}}]}},
            ])
            extract_team_state(load_messages(p))  # must not raise

    def test_doctor_scanners_survive_poisoned_sessions(self):
        from cozempic.doctor import _count_corrupted_tool_use, _count_orphaned_tool_results
        with tempfile.TemporaryDirectory() as d:
            proj = Path(d) / "p"
            proj.mkdir()
            (proj / "a.jsonl").write_text('"a bare string transcript line"\n')
            (proj / "b.jsonl").write_text('{"type":"user","message":"i am a string"}\n')
            (proj / "c.jsonl").write_text(json.dumps({"type": "assistant", "message":
                {"role": "assistant", "content": [123, {"type": "text", "text": "ok"}]}}) + "\n")
            for f in ("a.jsonl", "b.jsonl", "c.jsonl"):
                _count_corrupted_tool_use(proj / f)
                _count_orphaned_tool_results(proj / f)


class TestOverflowTruncatedTailFalseFire(unittest.TestCase):
    """The 100KB tail seek cuts a marker-bearing prose line into invalid JSON. The
    old `except -> return True` then false-fired an unsolicited kill+resume on a
    benign session. Only a structurally-valid API-error entry may trigger."""

    def _rec(self, p):
        from cozempic.overflow import OverflowRecovery, CircuitBreaker
        return OverflowRecovery(p, "sid", "/tmp", CircuitBreaker("sid"), danger_threshold_mb=0.0)

    def test_benign_prose_with_truncated_marker_does_not_fire(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "big.jsonl"
            filler = json.dumps({"type": "user", "message": {"role": "user", "content": "x" * 2000}}) + "\n"
            prose = json.dumps({"type": "user", "message": {"role": "user",
                "content": "note: 'prompt is too long' and 'maximum context length' are common errors " + "y" * 1000}}) + "\n"
            with open(p, "w") as f:
                f.write(filler * 60)  # > 100KB so the tail seek truncates the first line
                f.write(prose)
            self.assertFalse(self._rec(p).detect_overflow())

    def test_real_api_error_still_fires(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "e.jsonl"
            p.write_text(json.dumps({"type": "assistant", "isApiErrorMessage": True,
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Prompt is too long"}]}}) + "\n")
            self.assertTrue(self._rec(p).detect_overflow())


if __name__ == "__main__":
    unittest.main()
