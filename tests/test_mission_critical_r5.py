"""Round-5 mission-critical regression tests for PR #138.

Pins the R5 findings — including a P0 that R4 itself introduced — to their
production entry paths: the surrogateescape in-string byte must round-trip
BYTE-EXACT (not be rewritten to a literal \\udcXX escape); the team pre-pass /
membership pre-filters must not crash on an unhashable name/id; session
enumeration must tolerate a stray byte; and the read-only block consumers (recap,
diagnosis) must not crash on a non-dict content element.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from cozempic.session import (
    load_messages, load_messages_and_snapshot, save_messages,
    find_sessions, find_current_session,
)
from cozempic.executor import run_prescription


class TestSurrogateInStringByteExactP0(unittest.TestCase):
    """R4 switched to surrogateescape but save used ensure_ascii=True (default), which
    escaped an in-string non-UTF-8 byte to the literal 6-char text \\udcXX — silent
    corruption of the live transcript on every prune. ensure_ascii=False round-trips
    the exact byte."""

    def test_in_string_byte_round_trips_byte_exact_through_prune(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.jsonl"
            p.write_bytes(
                b'{"type":"user","uuid":"u0","message":{"role":"user","content":"keep ' + b"z" * 200 + b'"}}\n'
                b'{"type":"assistant","uuid":"a1","message":{"role":"assistant","content":"caf\xe9.txt"}}\n'
            )
            msgs, snap = load_messages_and_snapshot(p)
            out, _ = run_prescription(msgs, ["standard"], {})
            save_messages(p, out, create_backup=False)
            after = p.read_bytes()
            self.assertIn(b'caf\xe9.txt', after, "in-string byte must survive byte-exact")
            self.assertNotIn(b'\\udce9', after, "must NOT corrupt to a literal escape")


class TestTeamPrePassUnhashable(unittest.TestCase):
    """The R4 _sfield coercion hardened the main extraction loop but missed the
    pre-pass and the `name in <set>` / `id in <set>` membership pre-filters. An
    unhashable list/dict name or id raised TypeError -> guard respawn storm."""

    def test_unhashable_name_and_id_do_not_crash_extract(self):
        from cozempic.team import extract_team_state
        for blk in (
            {"type": "tool_use", "id": ["evil"], "name": "Task", "input": {"prompt": "x"}},
            {"type": "tool_use", "id": "ok", "name": ["Task"], "input": {}},
            {"type": "tool_use", "id": {"k": "v"}, "name": "TeamCreate", "input": {}},
        ):
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "s.jsonl"
                p.write_text(json.dumps({"type": "assistant", "uuid": "a1",
                    "message": {"role": "assistant", "content": [blk]}}) + "\n")
                extract_team_state(load_messages(p))  # must not raise


class TestSessionEnumerationStrayByte(unittest.TestCase):
    """find_sessions / find_current_session opened with strict utf-8 and crashed on a
    stray byte in ANY project file — re-instating guard inertness upstream of the
    R4 load/save fix."""

    def test_enumeration_tolerates_stray_byte(self):
        with tempfile.TemporaryDirectory() as d:
            old = os.environ.get("CLAUDE_CONFIG_DIR")
            os.environ["CLAUDE_CONFIG_DIR"] = d
            try:
                proj = Path(d) / "projects" / "p"
                proj.mkdir(parents=True)
                (proj / "bad.jsonl").write_bytes(
                    b'{"type":"user","message":{"role":"user","content":"x \xff y"}}\n')
                find_sessions()  # must not raise
                find_current_session()  # must not raise
            finally:
                if old is None:
                    os.environ.pop("CLAUDE_CONFIG_DIR", None)
                else:
                    os.environ["CLAUDE_CONFIG_DIR"] = old


class TestReadOnlyConsumersNonDictBlock(unittest.TestCase):
    """recap / diagnosis iterate content blocks and crashed on a non-dict element
    (get_content_blocks returns the array verbatim). They now use get_dict_blocks."""

    def _session(self, d):
        p = Path(d) / "s.jsonl"
        p.write_text(json.dumps({"type": "assistant", "uuid": "a1",
            "message": {"role": "assistant", "content": [42, ["x"], None, {"type": "text", "text": "real"}]}}) + "\n")
        return load_messages(p)

    def test_recap_survives_non_dict_block(self):
        from cozempic.recap import generate_recap
        with tempfile.TemporaryDirectory() as d:
            generate_recap(self._session(d))  # must not raise

    def test_diagnose_survives_non_dict_block(self):
        from cozempic.diagnosis import diagnose_session
        with tempfile.TemporaryDirectory() as d:
            diagnose_session(self._session(d))  # must not raise


if __name__ == "__main__":
    unittest.main()
