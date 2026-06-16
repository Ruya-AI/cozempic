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

    def test_count_rule_never_under_rejects_freeze_vectors(self):
        # ROUNDS 7-10 LESSON: precise separator heuristics (R8 adjacency, R9
        # top-level-anchor) each reopened the daemon FREEZE because real ReDoS
        # detection needs class-overlap + branch-emptiness analysis. The detector
        # reverted to the provably-safe COUNT rule (>=2 variable-width quantifiers),
        # which can NEVER under-reject. Every freeze vector the precise rules missed
        # (empty-alternation separators, class-overlapping literals, group-wrapped /
        # optional-separated variables) MUST be flagged.
        from cozempic.helpers import _pattern_is_redos_risky as risky
        for p in [r".*(?:Z|).*(?:Z|).*(?:Z|).*c", r"a*(b|)a*(b|)a*(b|)a*c",
                  r".*a.*a.*a.*a.*X", r".*log.*log.*log.*END", r".*/.*/.*/.*/x",
                  r".*X.*X.*X.*X.*Y", r".*X.*X.*Y",
                  r"(.{1,500})(.{1,500})(.{1,500})c", r"(a*)(a*)(a*)c",
                  r".{1,500}x?.{1,500}x?.{1,500}c", r".*x?.*y?.*z?.*c",
                  r"\w*\W?\w*\W?\w*Q", r".{1,50}.{1,50}",
                  r"a.{1,500}.{1,500}.{1,500}.{1,500}.{1,500}b",
                  # R11: a flat optional-quantifier chain backtracks 2^N — `?` MUST be
                  # counted as a branch quantifier or the count rule under-rejects it.
                  ("a?" * 30) + ("a" * 30), ("x?" * 12) + ("x" * 12), r"a?a?a?aaa"]:
            self.assertTrue(risky(p), f"freeze vector MUST be flagged (count rule): {p}")
        # A SINGLE optional `?` is linear and must stay allowed (https?, colou?r).
        for p in [r"https?", r"colou?r", r"(www\.)?x", r"a?", r".*?", r"a+?", r"(?:abc)"]:
            self.assertFalse(risky(p), f"single/lazy/marker `?` wrongly flagged: {p}")
        # Single-quantifier / fixed-brace patterns stay allowed (no over-block of the
        # common safe shapes). NOTE: literal-separated multi-range patterns
        # (\d{1,3}\.\d{1,3}) ARE conservatively flagged by the count rule — that is the
        # accepted fail-open over-rejection on the no-SIGALRM path, not tested as "safe".
        # NOTE (R12): alternation patterns ((TODO|FIXME)+) are NOT here — they are now
        # categorically flagged (fail-open) since ambiguous alternation can't be told
        # from benign without full regex analysis.
        for p in [r"R\d{1,5}", r".{1,500}", r"\d{1,3}", r"(\d{4})+", r"(\w{8})+",
                  r"(KEEP)+", r".*", r"\bword\b"]:
            self.assertFalse(risky(p), f"benign single-quantifier pattern wrongly flagged: {p}")

    def test_r12_alternation_categorically_flagged(self):
        # R12: the alternation backtracking class is closed categorically — nested,
        # unquantified-chain, and overlapping ambiguous alternations all freeze and
        # can't be distinguished from benign ones, so ANY `|` is refused.
        from cozempic.helpers import _pattern_is_redos_risky as risky
        for p in [r"((a|a))+c", "(a|a)" * 25 + "b", "(a|a|a)" * 16 + "x",
                  "(aa|a)" * 20 + "X", r"foo|bar|baz", r"(TODO|FIXME)+"]:
            self.assertTrue(risky(p), f"alternation must be categorically flagged: {p}")
        # `|` inside a character class is NOT an alternation — must stay allowed.
        for p in [r"[a|b]+", r"[|]"]:
            self.assertFalse(risky(p), f"`|` in a char class is literal, not alternation: {p}")


if __name__ == "__main__":
    unittest.main()
