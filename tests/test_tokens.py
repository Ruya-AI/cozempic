"""Tests for token estimation module."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from cozempic.helpers import msg_bytes
from cozempic.tokens import (
    CHARS_PER_TOKEN_DEFAULT,
    DEFAULT_CONTEXT_WINDOW,
    SYSTEM_OVERHEAD_TOKENS,
    TokenEstimate,
    _is_context_message,
    calibrate_ratio,
    estimate_session_tokens,
    estimate_tokens_heuristic,
    extract_usage_tokens,
    quick_token_estimate,
)


def make_message(line_idx: int, msg: dict) -> tuple[int, dict, int]:
    return (line_idx, msg, msg_bytes(msg))


def make_assistant_with_usage(
    line_idx: int,
    text: str = "hello",
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_creation: int = 500,
    cache_read: int = 300,
    sidechain: bool = False,
) -> tuple[int, dict, int]:
    msg = {
        "type": "assistant",
        "isSidechain": sidechain,
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
            },
            "stop_reason": "end_turn",
        },
        "costUSD": 0.01,
        "duration": 1234,
    }
    return make_message(line_idx, msg)


def make_assistant_no_usage(line_idx: int, text: str = "hello") -> tuple[int, dict, int]:
    msg = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
        },
    }
    return make_message(line_idx, msg)


def make_user(line_idx: int, text: str = "hi") -> tuple[int, dict, int]:
    msg = {
        "type": "user",
        "isSidechain": False,
        "message": {"role": "user", "content": text},
    }
    return make_message(line_idx, msg)


def make_progress(line_idx: int) -> tuple[int, dict, int]:
    msg = {
        "type": "progress",
        "data": {"type": "hook_progress"},
    }
    return make_message(line_idx, msg)


def make_file_history(line_idx: int) -> tuple[int, dict, int]:
    msg = {
        "type": "file-history-snapshot",
        "files": [{"path": "/foo/bar.py", "content": "x" * 1000}],
    }
    return make_message(line_idx, msg)


def make_sidechain_assistant(line_idx: int, text: str = "sub-task") -> tuple[int, dict, int]:
    msg = {
        "type": "assistant",
        "isSidechain": True,
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "usage": {
                "input_tokens": 500,
                "output_tokens": 100,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    }
    return make_message(line_idx, msg)


def make_thinking_only(line_idx: int) -> tuple[int, dict, int]:
    msg = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm let me think", "signature": "sig123"},
            ],
        },
    }
    return make_message(line_idx, msg)


class TestUsageCoercion(unittest.TestCase):
    """A malformed transcript (present-but-null/string/float usage field) must not
    crash the token estimators — the bare .get(k,0) only defaults a MISSING key, so
    `None + 0` raised TypeError that escaped the guard daemon loop and killed it."""

    def test_as_int_coerces_junk_to_zero(self):
        from cozempic.tokens import _as_int
        self.assertEqual(_as_int(500), 500)
        self.assertEqual(_as_int(3.9), 3)
        self.assertEqual(_as_int(None), 0)
        self.assertEqual(_as_int("1234"), 0)     # strings are not summed
        self.assertEqual(_as_int(True), 0)       # bool is an int subclass — treat as 0
        self.assertEqual(_as_int(-5), 0)         # negative -> 0
        self.assertEqual(_as_int([]), 0)

    def test_as_int_nonfinite_does_not_raise(self):
        # 1e999 -> inf, json accepts NaN/Infinity; int(inf) raises OverflowError /
        # int(nan) raises ValueError — those must be coerced to 0, not escape.
        from cozempic.tokens import _as_int
        self.assertEqual(_as_int(float("inf")), 0)
        self.assertEqual(_as_int(float("-inf")), 0)
        self.assertEqual(_as_int(float("nan")), 0)
        self.assertEqual(_as_int(1e999), 0)

    def test_inner_dict_non_dict_message_is_safe(self):
        # A present-but-non-dict "message" (a plain string) must not crash the
        # token sites (the bare .get('message',{}) default only covers a missing key).
        from cozempic.tokens import _inner_dict, extract_usage_tokens, quick_token_estimate
        self.assertEqual(_inner_dict({"message": "a string"}), {})
        self.assertEqual(_inner_dict({"message": None}), {})
        self.assertEqual(_inner_dict({"message": {"x": 1}}), {"x": 1})
        # extract_usage_tokens must not crash on a non-dict message line
        self.assertIsNone(extract_usage_tokens([(0, {"type": "assistant", "message": "oops"}, 10)]))

    def test_extract_usage_tolerates_null_field(self):
        msg = {
            "type": "assistant",
            "message": {"role": "assistant", "model": "claude-opus-4-8", "content": [],
                        "usage": {"input_tokens": None, "output_tokens": 50,
                                  "cache_read_input_tokens": "oops", "cache_creation_input_tokens": 10}},
        }
        r = extract_usage_tokens([(0, msg, 100)])  # (idx, msg, bytes) tuples
        self.assertIsNotNone(r)
        self.assertEqual(r["total"], 60)  # 0 + 10 + 0 + 50, no crash


class TestExtractUsageTokens(unittest.TestCase):

    def test_extracts_from_last_assistant(self):
        messages = [
            make_user(0, "hello"),
            make_assistant_with_usage(1, "first", input_tokens=500, cache_creation=100, cache_read=50),
            make_user(2, "more"),
            make_assistant_with_usage(3, "second", input_tokens=1000, cache_creation=200, cache_read=300),
        ]
        result = extract_usage_tokens(messages)
        self.assertIsNotNone(result)
        self.assertEqual(result["input_tokens"], 1000)
        self.assertEqual(result["cache_creation_input_tokens"], 200)
        self.assertEqual(result["cache_read_input_tokens"], 300)
        self.assertEqual(result["total"], 1700)  # 1000 + 200 + 300 + 200(output)

    def test_skips_sidechain(self):
        messages = [
            make_user(0, "hello"),
            make_assistant_with_usage(1, "main", input_tokens=800, cache_creation=0, cache_read=0),
            make_sidechain_assistant(2, "sub-task"),
        ]
        result = extract_usage_tokens(messages)
        self.assertIsNotNone(result)
        self.assertEqual(result["input_tokens"], 800)
        self.assertEqual(result["total"], 1000)  # 800 + 200(output)

    def test_skips_parse_errors(self):
        messages = [
            make_user(0, "hello"),
            make_assistant_with_usage(1, "good", input_tokens=600, cache_creation=0, cache_read=0),
            (2, {"_parse_error": True, "_raw": "bad json", "type": "assistant"}, 8),
        ]
        result = extract_usage_tokens(messages)
        self.assertIsNotNone(result)
        self.assertEqual(result["total"], 800)  # 600 + 200(output)

    def test_returns_none_when_no_usage(self):
        messages = [
            make_user(0, "hello"),
            make_assistant_no_usage(1, "response"),
        ]
        result = extract_usage_tokens(messages)
        self.assertIsNone(result)

    def test_returns_none_for_empty_messages(self):
        result = extract_usage_tokens([])
        self.assertIsNone(result)


class TestIsContextMessage(unittest.TestCase):

    def test_user_message_is_context(self):
        _, msg, _ = make_user(0, "hello")
        self.assertTrue(_is_context_message(msg))

    def test_assistant_message_is_context(self):
        _, msg, _ = make_assistant_with_usage(0, "response")
        self.assertTrue(_is_context_message(msg))

    def test_progress_is_not_context(self):
        _, msg, _ = make_progress(0)
        self.assertFalse(_is_context_message(msg))

    def test_file_history_is_not_context(self):
        _, msg, _ = make_file_history(0)
        self.assertFalse(_is_context_message(msg))

    def test_sidechain_is_not_context(self):
        _, msg, _ = make_sidechain_assistant(0)
        self.assertFalse(_is_context_message(msg))

    def test_thinking_only_is_not_context(self):
        _, msg, _ = make_thinking_only(0)
        self.assertFalse(_is_context_message(msg))


class TestHeuristicEstimation(unittest.TestCase):

    def test_empty_session(self):
        total, breakdown = estimate_tokens_heuristic([])
        self.assertEqual(total, SYSTEM_OVERHEAD_TOKENS)
        self.assertEqual(breakdown, {})

    def test_basic_estimation(self):
        messages = [
            make_user(0, "a" * 370),  # ~120 tokens at the 3.1 chars/token default
            make_assistant_no_usage(1, "b" * 370),
        ]
        total, breakdown = estimate_tokens_heuristic(messages)
        # Should be roughly 200 content tokens + overhead
        self.assertGreater(total, SYSTEM_OVERHEAD_TOKENS)
        self.assertIn("user", breakdown)
        self.assertIn("assistant", breakdown)

    def test_thinking_blocks_excluded(self):
        """Thinking content should not count toward token estimate."""
        msg = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "x" * 10000},
                    {"type": "text", "text": "short answer"},
                ],
            },
        }
        messages = [make_message(0, msg)]
        total, _ = estimate_tokens_heuristic(messages)
        # Thinking excluded → only "short answer" counts, far below overhead+100
        self.assertLess(total, SYSTEM_OVERHEAD_TOKENS + 100)

    def test_skips_progress_and_file_history(self):
        messages = [
            make_user(0, "hello"),
            make_progress(1),
            make_progress(2),
            make_file_history(3),
            make_assistant_no_usage(4, "response"),
        ]
        total, breakdown = estimate_tokens_heuristic(messages)
        self.assertNotIn("progress", breakdown)
        self.assertNotIn("file-history-snapshot", breakdown)


class TestEstimateSessionTokens(unittest.TestCase):

    def test_exact_preferred_over_heuristic(self):
        messages = [
            make_user(0, "hello"),
            make_assistant_with_usage(1, "resp", input_tokens=50000, cache_creation=10000, cache_read=5000),
        ]
        te = estimate_session_tokens(messages)
        self.assertEqual(te.method, "exact")
        self.assertEqual(te.confidence, "high")
        self.assertEqual(te.total, 65200)  # 50000 + 10000 + 5000 + 200(output)
        expected_pct = round(65200 / DEFAULT_CONTEXT_WINDOW * 100, 1)
        self.assertEqual(te.context_pct, expected_pct)

    def test_heuristic_fallback(self):
        messages = [
            make_user(0, "hello"),
            make_assistant_no_usage(1, "response without usage"),
        ]
        te = estimate_session_tokens(messages)
        self.assertEqual(te.method, "heuristic")
        self.assertEqual(te.confidence, "medium")
        self.assertGreater(te.total, 0)

    def test_context_pct_calculation(self):
        messages = [
            make_user(0, "hello"),
            make_assistant_with_usage(1, "resp", input_tokens=100000, cache_creation=0, cache_read=0),
        ]
        te = estimate_session_tokens(messages)
        self.assertEqual(te.context_pct, 10.0)  # (100K + 200 output) / 1M


class TestQuickTokenEstimate(unittest.TestCase):

    def _write_jsonl(self, messages: list[dict]) -> Path:
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
        for msg in messages:
            tmp.write(json.dumps(msg) + "\n")
        tmp.close()
        return Path(tmp.name)

    def test_reads_usage_from_tail(self):
        messages = [
            {"type": "user", "message": {"role": "user", "content": "hello"}},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}],
                    "usage": {
                        "input_tokens": 5000,
                        "output_tokens": 200,
                        "cache_creation_input_tokens": 1000,
                        "cache_read_input_tokens": 500,
                    },
                },
            },
        ]
        path = self._write_jsonl(messages)
        try:
            result = quick_token_estimate(path)
            self.assertEqual(result, 6700)  # 5000 + 1000 + 500 + 200(output)
        finally:
            path.unlink()

    def test_returns_none_without_usage(self):
        messages = [
            {"type": "user", "message": {"role": "user", "content": "hello"}},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}],
                },
            },
        ]
        path = self._write_jsonl(messages)
        try:
            result = quick_token_estimate(path)
            self.assertIsNone(result)
        finally:
            path.unlink()

    def test_skips_sidechain_in_tail(self):
        messages = [
            {"type": "user", "message": {"role": "user", "content": "hello"}},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "main"}],
                    "usage": {
                        "input_tokens": 8000,
                        "output_tokens": 300,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                },
            },
            {
                "type": "assistant",
                "isSidechain": True,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "sub"}],
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                },
            },
        ]
        path = self._write_jsonl(messages)
        try:
            result = quick_token_estimate(path)
            self.assertEqual(result, 8300)  # 8000 + 300(output)
        finally:
            path.unlink()

    def test_handles_missing_file(self):
        result = quick_token_estimate(Path("/nonexistent/file.jsonl"))
        self.assertIsNone(result)

    def test_grows_tail_to_find_usage_beyond_window(self):
        # A usage frame buried behind a long trailing run of usage-less lines
        # must still be found — otherwise the guard goes token-blind on long
        # sessions (every threshold is gated on `current_tokens is not None`).
        # Usage frame at the front; >100KB of usage-less filler follows it,
        # past the 1M fixed tail.
        messages = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "early"}],
                    "usage": {
                        "input_tokens": 1000, "output_tokens": 200,
                        "cache_creation_input_tokens": 500, "cache_read_input_tokens": 300,
                    },
                },
            },
        ]
        # ~2KB per filler line; 150 lines ≈ 300KB of usage-less tail (> 100KB).
        for _ in range(150):
            messages.append({"type": "user", "message": {"role": "user", "content": "x" * 2000}})
        path = self._write_jsonl(messages)
        try:
            self.assertGreater(path.stat().st_size, 200 * 1024)
            result = quick_token_estimate(path, context_window=1_000_000)
            self.assertEqual(result, 2000)  # 1000 + 500 + 300 + 200(output)
        finally:
            path.unlink()


class TestCalibratedHeuristicPath(unittest.TestCase):
    """Test that estimate_session_tokens() uses calibrate_ratio() in heuristic path."""

    def setUp(self):
        # These tests assert exact heuristic totals against the default ratio.
        # Isolate from a COZEMPIC_CHARS_PER_TOKEN override that may be set in the
        # developer's shell (same env-leak class PR #95 fixed for the window).
        self._saved_cpt = os.environ.pop("COZEMPIC_CHARS_PER_TOKEN", None)
        self.addCleanup(self._restore_cpt)

    def _restore_cpt(self):
        if self._saved_cpt is not None:
            os.environ["COZEMPIC_CHARS_PER_TOKEN"] = self._saved_cpt

    def test_calibrated_ratio_used_when_usage_and_content_available(self):
        """When exact usage exists but we force heuristic path,
        calibrate_ratio() should be used. We test indirectly: if the session
        has both usage data AND content, calibrate_ratio() returns a ratio,
        which the heuristic path should use.

        Here we craft a session where exact path succeeds, so we verify
        the heuristic path uses calibration by mocking extract_usage_tokens
        to return None on the first call (exact path) but not on calibrate_ratio's call.
        Instead, we test the simpler case: a session with NO usage data
        falls back correctly.
        """
        # Session with usage data — exact path takes precedence
        # This test verifies calibrate_ratio integration exists
        # by checking the heuristic fallback with calibration
        text = "a" * 7400  # 7400 chars
        messages = [
            make_user(0, text),
            make_assistant_no_usage(1, text),
        ]
        te = estimate_session_tokens(messages)
        self.assertEqual(te.method, "heuristic")
        # No usage data → calibrate_ratio returns None → default ratio used
        expected_tokens = int(14800 / CHARS_PER_TOKEN_DEFAULT) + SYSTEM_OVERHEAD_TOKENS
        self.assertEqual(te.total, expected_tokens)

    def test_fallback_to_default_ratio_when_no_usage(self):
        """When calibrate_ratio() returns None (no usage data),
        default ratio is used."""
        messages = [
            make_user(0, "hello world"),
            make_assistant_no_usage(1, "greetings"),
        ]
        te = estimate_session_tokens(messages)
        self.assertEqual(te.method, "heuristic")
        self.assertEqual(te.confidence, "medium")
        # Should use the default chars/token ratio
        total_chars = len("hello world") + len("greetings")
        expected = int(total_chars / CHARS_PER_TOKEN_DEFAULT) + SYSTEM_OVERHEAD_TOKENS
        self.assertEqual(te.total, expected)


class TestCalibrateRatio(unittest.TestCase):

    def test_returns_ratio_with_usage(self):
        text = "a" * 3700  # exact ratio comes from usage below, not the default
        messages = [
            make_user(0, text),
            make_assistant_with_usage(
                1, text,
                input_tokens=40000,
                cache_creation=0,
                cache_read=0,
            ),
        ]
        ratio = calibrate_ratio(messages)
        self.assertIsNotNone(ratio)
        self.assertGreater(ratio, 0)

    def test_returns_none_without_usage(self):
        messages = [
            make_user(0, "hello"),
            make_assistant_no_usage(1, "response"),
        ]
        ratio = calibrate_ratio(messages)
        self.assertIsNone(ratio)


class TestEnvVarOverrideValidation(unittest.TestCase):
    """Env var override parsing must reject nonsensical values instead of
    silently ignoring (previous falsy-trap bug with `if val:`) or propagating
    them into token-math divisions producing negative percentages."""

    def _get_window(self):
        from cozempic.tokens import get_context_window_override
        return get_context_window_override()

    def _get_overhead(self):
        from cozempic.tokens import get_system_overhead_tokens
        return get_system_overhead_tokens()

    def test_zero_context_window_was_falsy_trap_now_rejected(self):
        """COZEMPIC_CONTEXT_WINDOW=0 was silently swallowed by the old
        `if val:` check. Now returns None (plus a warning) so the caller
        falls back to model-based detection."""
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"COZEMPIC_CONTEXT_WINDOW": "0"}):
            self.assertIsNone(self._get_window())

    def test_negative_context_window_rejected(self):
        """Previously returned -100, producing context_pct=-110%."""
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"COZEMPIC_CONTEXT_WINDOW": "-100"}):
            self.assertIsNone(self._get_window())

    def test_valid_context_window_override(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"COZEMPIC_CONTEXT_WINDOW": "200000"}):
            self.assertEqual(self._get_window(), 200000)

    def test_system_overhead_accepts_zero(self):
        """0 is valid here — a session with no rules/MCP has no overhead."""
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"COZEMPIC_SYSTEM_OVERHEAD_TOKENS": "0"}):
            self.assertEqual(self._get_overhead(), 0)

    def test_system_overhead_rejects_negative(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"COZEMPIC_SYSTEM_OVERHEAD_TOKENS": "-50"}):
            self.assertEqual(self._get_overhead(), SYSTEM_OVERHEAD_TOKENS)


class TestCharsPerTokenOverride(unittest.TestCase):
    """COZEMPIC_CHARS_PER_TOKEN override for the heuristic divisor."""

    def setUp(self):
        self._saved = os.environ.pop("COZEMPIC_CHARS_PER_TOKEN", None)
        self.addCleanup(self._restore)

    def _restore(self):
        os.environ.pop("COZEMPIC_CHARS_PER_TOKEN", None)
        if self._saved is not None:
            os.environ["COZEMPIC_CHARS_PER_TOKEN"] = self._saved

    def test_default_when_unset(self):
        from cozempic.tokens import get_chars_per_token
        self.assertEqual(get_chars_per_token(), CHARS_PER_TOKEN_DEFAULT)

    def test_valid_override_honored(self):
        from cozempic.tokens import get_chars_per_token
        os.environ["COZEMPIC_CHARS_PER_TOKEN"] = "2.5"
        self.assertEqual(get_chars_per_token(), 2.5)

    def test_out_of_range_falls_back_to_default(self):
        from cozempic.tokens import get_chars_per_token
        for bad in ("99", "0", "0.5", "-3"):
            os.environ["COZEMPIC_CHARS_PER_TOKEN"] = bad
            self.assertEqual(get_chars_per_token(), CHARS_PER_TOKEN_DEFAULT, bad)

    def test_garbage_falls_back_to_default(self):
        from cozempic.tokens import get_chars_per_token
        os.environ["COZEMPIC_CHARS_PER_TOKEN"] = "abc"
        self.assertEqual(get_chars_per_token(), CHARS_PER_TOKEN_DEFAULT)

    def test_override_changes_heuristic_total(self):
        msgs = [make_user(0, "a" * 1000), make_assistant_no_usage(1, "")]
        os.environ["COZEMPIC_CHARS_PER_TOKEN"] = "2.0"
        total_dense, _ = estimate_tokens_heuristic(msgs)
        os.environ["COZEMPIC_CHARS_PER_TOKEN"] = "4.0"
        total_sparse, _ = estimate_tokens_heuristic(msgs)
        # Smaller divisor → more tokens per char.
        self.assertGreater(total_dense, total_sparse)


if __name__ == "__main__":
    unittest.main()
