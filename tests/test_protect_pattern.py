"""--protect-pattern (#122, originally by @eggrollofchaos; extracted + hardened):
user-defined regex prune-immunity. Messages whose text matches a pattern get the
same is_protected() immunity as team messages, so every strategy spares them.

Mechanism proof: the prescription strategies (gentle/standard/aggressive) all check
is_protected() before pruning, and is_protected() honors __cozempic_pattern_protected__.
These tests verify the matcher surfaces, the safeguards (ReDoS-length cap, input cap,
over-protection warn), the crash-safe strip, and that a tagged message is excluded
from a strategy's removal actions.
"""

import io
import unittest
from contextlib import redirect_stderr
from unittest import mock

from cozempic.helpers import (
    compile_protect_patterns, tag_pattern_matches, strip_pattern_tags, is_protected,
    _msg_text_matches_any, _PATTERN_PROTECTED_KEY,
    _MAX_PROTECT_PATTERN_LEN,
)
from cozempic.registry import STRATEGIES
import cozempic.strategies  # noqa: F401  (register strategies)


def _txt(t):
    return {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": t}]}}


def _tool_result(t):
    return {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": t}]}}


def _user_str(t):
    return {"type": "user", "message": {"role": "user", "content": t}}


class TestCompileGuards(unittest.TestCase):
    def test_valid(self):
        pats = compile_protect_patterns(["foo", r"R\d+"])
        self.assertEqual(len(pats), 2)

    def test_invalid_regex_rejected(self):
        with self.assertRaises(ValueError):
            compile_protect_patterns(["(unclosed"])

    def test_overlong_pattern_rejected(self):
        with self.assertRaises(ValueError):
            compile_protect_patterns(["a" * (_MAX_PROTECT_PATTERN_LEN + 1)])

    def test_nonstring_rejected(self):
        with self.assertRaises(ValueError):
            compile_protect_patterns([123])

    def test_none_and_empty(self):
        self.assertEqual(compile_protect_patterns(None), [])
        self.assertEqual(compile_protect_patterns([]), [])


class TestMatcherSurfaces(unittest.TestCase):
    def setUp(self):
        self.pats = compile_protect_patterns([r"GATE CONTRACT R\d+"])

    def test_matches_text_block(self):
        self.assertTrue(_msg_text_matches_any(_txt("see GATE CONTRACT R070 here"), self.pats))

    def test_matches_tool_result(self):
        # The key fix vs the original PR: tool outputs (which ARE pruned) are scanned.
        self.assertTrue(_msg_text_matches_any(_tool_result("output: GATE CONTRACT R12"), self.pats))

    def test_matches_user_string(self):
        self.assertTrue(_msg_text_matches_any(_user_str("GATE CONTRACT R5 standing rule"), self.pats))

    def test_no_false_match(self):
        self.assertFalse(_msg_text_matches_any(_txt("ordinary chatter"), self.pats))

    def test_input_capped_but_still_matches_early(self):
        # A huge message still matches a pattern near the start without scanning all of it.
        self.assertTrue(_msg_text_matches_any(_txt("GATE CONTRACT R1 " + "x" * 500_000), self.pats))


class TestTagAndStrip(unittest.TestCase):
    def test_tags_and_counts(self):
        msgs = [(0, _txt("GATE CONTRACT R1"), 10), (1, _txt("nope"), 10), (2, _tool_result("GATE CONTRACT R2"), 10)]
        n = tag_pattern_matches(msgs, compile_protect_patterns([r"GATE CONTRACT R\d+"]))
        self.assertEqual(n, 2)
        self.assertEqual([is_protected(d) for _, d, _ in msgs], [True, False, True])

    def test_strip_clears(self):
        msgs = [(0, _txt("GATE CONTRACT R1"), 10)]
        tag_pattern_matches(msgs, compile_protect_patterns([r"GATE CONTRACT R\d+"]))
        strip_pattern_tags(msgs)
        self.assertFalse(is_protected(msgs[0][1]))
        self.assertNotIn(_PATTERN_PROTECTED_KEY, msgs[0][1])

    def test_empty_patterns_noop(self):
        msgs = [(0, _txt("x"), 10)]
        self.assertEqual(tag_pattern_matches(msgs, []), 0)

    def test_overprotection_warns(self):
        msgs = [(i, _txt("GATE CONTRACT R1"), 10) for i in range(5)]
        buf = io.StringIO()
        with redirect_stderr(buf):
            tag_pattern_matches(msgs, compile_protect_patterns([r"GATE CONTRACT R\d+"]))
        self.assertIn("matched 5/5 matchable", buf.getvalue())


class TestHardening1828(unittest.TestCase):
    """QA-fleet-driven hardening of the extracted #122 feature."""

    def _tu(self, i):
        return {"type": "assistant", "message": {"role": "assistant",
                "content": [{"type": "tool_use", "id": i, "name": "Bash", "input": {"command": "x"}}]}}

    def test_redos_fails_closed_when_no_sigalrm(self):
        # Windows / non-main-thread: no SIGALRM budget can be armed, and a pure
        # thread can't interrupt a CPU-bound re match — so a redos-shaped pattern
        # must be REFUSED up front (fail closed), not run unbounded. Emulate "no
        # SIGALRM" by patching _have_sigalrm (real Windows e2e needs a Windows box).
        import os, time
        from cozempic import helpers
        with mock.patch.object(helpers, "_have_sigalrm", return_value=False), \
             mock.patch.dict(os.environ, {"COZEMPIC_PROTECT_MATCH_SECONDS": "2.0"}):
            evil = compile_protect_patterns([r"(a+)+$"])
            msgs = [(0, _txt("a" * 5000 + "!"), 50)]  # would hang for minutes if matched
            buf = io.StringIO()
            t0 = time.perf_counter()
            with redirect_stderr(buf):
                n = tag_pattern_matches(msgs, evil)
            dt = time.perf_counter() - t0
            self.assertLess(dt, 1.0, "must refuse the risky pattern before matching, not hang")
            self.assertEqual(n, 0, "fail closed: no protection applied")
            self.assertNotIn(_PATTERN_PROTECTED_KEY, msgs[0][1])
            self.assertIn("no time budget on this platform", buf.getvalue())

    def test_safe_pattern_still_works_when_no_sigalrm(self):
        # A non-redos pattern must still match normally on the no-SIGALRM path.
        from cozempic import helpers
        with mock.patch.object(helpers, "_have_sigalrm", return_value=False):
            pats = compile_protect_patterns([r"GATE CONTRACT R\d+"])
            msgs = [(0, _txt("GATE CONTRACT R1 standing rule"), 10)]
            self.assertEqual(tag_pattern_matches(msgs, pats), 1)
            self.assertIn(_PATTERN_PROTECTED_KEY, msgs[0][1])

    def test_redos_shape_detector(self):
        from cozempic.helpers import _pattern_is_redos_risky as risky
        # Every catastrophic FORM must be flagged — including the R4-added
        # UNGROUPED adjacent quantifiers (.*.*.*.*c / a*a*) the prior group-only
        # detector MISSED and which froze the Windows daemon under the 512 cap.
        for p in [r"(a+)+$", r"(a*)*", r"(ab+c)*", r"(a|a)+", r"(x+){10}",
                  r"(x+){2,}", r"(a?)+", r"((a)|(a))+", r"(\S+\s+){5,}", r"(.*X){8}",
                  r".*.*.*.*c", r"a*a*", r"(secret-\d+)+",
                  # R12: ANY alternation is categorically refused (nested / unquantified
                  # ambiguous-alternation chains freeze and can't be told from benign ones).
                  r"foo|bar|baz", r"(TODO|FIXME)+"]:
            self.assertTrue(risky(p), f"catastrophic / alternation form not flagged: {p}")
        # Safe patterns stay usable (no false-positive freeze / silent protect drop).
        # R4: benign SINGLE-quantifier groups must NOT be flagged. R5 P3: a FIXED-count
        # inner brace ((\d{4})+, (\w{8})+) is unambiguous/linear and must NOT be flagged.
        # NOTE (R12): patterns with an alternation `|` are now conservatively flagged
        # (categorical) — they are NOT in this safe list.
        for p in [r"GATE CONTRACT R\d+", r"R\d{1,5}", r"\bword\b",
                  r"[A-Za-z0-9_]+", r"(KEEP)+", r"(DO-NOT-PRUNE)+",
                  r"(important){1,3}", r".*", r"(a+)", r"(\d{4})+", r"(\w{8})+"]:
            self.assertFalse(risky(p), f"safe pattern wrongly flagged: {p}")

    def test_redos_pattern_fails_open_within_budget(self):
        # A catastrophic-backtracking pattern must NOT hang the prune/daemon — it
        # times out and fails open (no protection this cycle), not seconds/minutes.
        import os, time
        # patch.dict save/restores any pre-existing value instead of unconditionally
        # popping it on teardown (which would clobber a real, externally-set
        # COZEMPIC_PROTECT_MATCH_SECONDS the developer had in their environment).
        with mock.patch.dict(os.environ, {"COZEMPIC_PROTECT_MATCH_SECONDS": "1.0"}):
            evil = compile_protect_patterns([r"(a+)+$"])
            msgs = [(0, _txt("a" * 40 + "!"), 50)]
            t0 = time.perf_counter()
            buf = io.StringIO()
            with redirect_stderr(buf):
                n = tag_pattern_matches(msgs, evil)
            dt = time.perf_counter() - t0
            self.assertLess(dt, 5.0, "ReDoS pattern must be time-bounded")
            self.assertEqual(n, 0, "must fail open (skip protection) on timeout")
            self.assertNotIn(_PATTERN_PROTECTED_KEY, msgs[0][1], "no half-protection left behind")
            self.assertIn("exceeded its time budget", buf.getvalue())

    def test_thinking_block_is_scanned(self):
        think = {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "SECRET reasoning GATE CONTRACT R9", "signature": "s"},
            {"type": "text", "text": "visible"}]}}
        self.assertTrue(_msg_text_matches_any(think, compile_protect_patterns([r"GATE CONTRACT R\d+"])))

    def test_overprotection_warn_uses_matchable_denominator(self):
        # Mixed session: 2 matchable text msgs + 2 unmatchable tool_use carriers.
        # A broad pattern protects all matchable content → must warn (it didn't when
        # the denominator was ALL messages: 2/4 = 50% < 80%).
        mixed = [(0, _txt("keep a"), 10), (1, self._tu("a"), 10),
                 (2, _txt("keep b"), 10), (3, self._tu("b"), 10)]
        buf = io.StringIO()
        with redirect_stderr(buf):
            tag_pattern_matches(mixed, compile_protect_patterns(["keep"]))
        self.assertIn("matched 2/2 matchable", buf.getvalue())

    def test_already_tagged_not_recounted(self):
        msgs = [(0, _txt("GATE CONTRACT R1"), 10)]
        pats = compile_protect_patterns([r"GATE CONTRACT R\d+"])
        self.assertEqual(tag_pattern_matches(msgs, pats), 1)
        self.assertEqual(tag_pattern_matches(msgs, pats), 0)  # idempotent


class TestStrategyHonorsPatternTag(unittest.TestCase):
    """The linchpin: a pattern-protected message must be excluded from a strategy's
    removal actions (every prescription strategy checks is_protected())."""

    def _prog(self, i):
        return (i, {"type": "progress", "data": {"type": "hook_progress"}, "uuid": f"u{i}"}, 30)

    def test_progress_collapse_skips_protected(self):
        msgs = [self._prog(0), self._prog(1), self._prog(2),
                (3, {"type": "user", "message": {"role": "user", "content": "x"}}, 10)]
        # Sanity: without protection, progress-collapse removes some of 0/1/2.
        base = STRATEGIES["progress-collapse"].func(
            [(i, dict(d), b) for i, d, b in msgs], {})
        self.assertTrue(base.actions, "progress-collapse should remove something here")
        # Protect the first progress message via the pattern key.
        msgs[0][1][_PATTERN_PROTECTED_KEY] = True
        sr = STRATEGIES["progress-collapse"].func(msgs, {})
        removed = {a.line_index for a in sr.actions}
        self.assertNotIn(0, removed, "a pattern-protected message must never be removed")


if __name__ == "__main__":
    unittest.main()
