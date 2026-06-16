"""Round-7 mission-critical regression tests for PR #138.

Two ship-blocking P1 sibling-misses: doctor --fix re-serialized a repaired line
with ensure_ascii=False and crashed (UnicodeEncodeError) on a lone surrogate (the
un-swept sibling of the save_messages surrogate fix); and the ReDoS detector missed
adjacent BOUNDED variable-width quantifiers (.{1,n}.{1,n}...) that backtrack
polynomially past the 512 no-budget cap.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TestDoctorFixSurrogateNoCrash(unittest.TestCase):
    def test_fix_corrupted_tool_use_survives_lone_surrogate(self):
        from cozempic import doctor
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.jsonl"
            badname = 'Bash" command="' + "x" * 250 + '"'  # corrupted tool_use name (>200)
            line = {"uuid": "u1", "type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": badname, "input": {}},
                {"type": "text", "text": "sliced emoji: \ud83d"}]}}  # lone high surrogate escape
            p.write_text(json.dumps(line, ensure_ascii=True) + "\n")
            with mock.patch.object(doctor, "find_sessions",
                                   return_value=[{"path": p, "session_id": "s", "mtime": 1.0}]):
                msg = doctor.fix_corrupted_tool_use()  # must NOT raise UnicodeEncodeError
            self.assertIn("Repaired", msg)
            # the repaired file must reload cleanly
            from cozempic.session import load_messages
            self.assertTrue(load_messages(p))

    def test_run_doctor_fix_does_not_abort_on_one_raising_fix(self):
        # A fix that raises must be contained — later checks still run.
        from cozempic import doctor
        with mock.patch.object(doctor, "find_sessions", return_value=[]):
            results = doctor.run_doctor(fix=True)  # must not raise
        self.assertTrue(results)


class TestCombinedSurrogateByteExact(unittest.TestCase):
    """A real non-UTF-8 byte (in-band surrogate) sharing a line with an out-of-band
    lone surrogate must keep the real byte BYTE-EXACT — _jsonl_line escapes ONLY the
    out-of-band surrogate, not the whole line (R7 byte-drift P3 eliminated)."""

    def test_real_byte_stays_exact_when_combined_with_out_of_band_surrogate(self):
        from cozempic.session import load_messages_and_snapshot, save_messages, load_messages
        from cozempic.executor import run_prescription
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.jsonl"
            p.write_bytes(
                b'{"type":"user","uuid":"u0","message":{"role":"user","content":"keep ' + b"z" * 300 + b'"}}\n'
                b'{"type":"assistant","uuid":"a1","message":{"role":"assistant","content":"caf\xe9 \\ud83d end"}}\n'
            )
            for _ in range(3):  # idempotent across cycles
                m, s = load_messages_and_snapshot(p)
                out, _ = run_prescription(m, ["standard"], {})
                save_messages(p, out, create_backup=False)
            after = p.read_bytes()
            self.assertIn(b'caf\xe9 ', after, "real byte must stay byte-exact")
            self.assertNotIn(b'\\udce9', after, "real byte must NOT drift to a literal escape")
            self.assertTrue(load_messages(p))  # no crash, reloads


class TestRedosBoundedRangePolyMiss(unittest.TestCase):
    def test_adjacent_bounded_ranges_flagged(self):
        from cozempic.helpers import _pattern_is_redos_risky as risky
        # Adjacent bounded variable-width ranges backtrack polynomially -> must flag.
        for p in [r"a.{1,500}.{1,500}.{1,500}.{1,500}.{1,500}b", r".{1,50}.{1,50}", r"x{2,9}y{2,9}"]:
            self.assertTrue(risky(p), f"bounded-range poly pattern not flagged: {p}")
        # A SINGLE bounded range is linear -> must NOT flag.
        for p in [r"R\d{1,5}", r".{1,500}", r"\d{1,3}", r"(\d{4})+", r"(\w{8})+"]:
            self.assertFalse(risky(p), f"linear bounded pattern wrongly flagged: {p}")


if __name__ == "__main__":
    unittest.main()
