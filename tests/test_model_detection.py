"""Tests for model detection and context window logic."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from cozempic.helpers import msg_bytes
from cozempic.tokens import (
    DEFAULT_CONTEXT_WINDOW,
    MODEL_CONTEXT_WINDOWS,
    detect_context_window,
    detect_model,
    estimate_session_tokens,
    get_context_window_override,
)


def make_message(line_idx: int, msg: dict) -> tuple[int, dict, int]:
    return (line_idx, msg, msg_bytes(msg))


def make_assistant_with_model(line_idx: int, model: str, input_tokens: int = 1000) -> tuple[int, dict, int]:
    msg = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": "response"}],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": 100,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    }
    return make_message(line_idx, msg)


class TestDetectModel(unittest.TestCase):

    def test_detects_model_from_assistant(self):
        messages = [make_assistant_with_model(0, "claude-opus-4-6")]
        self.assertEqual(detect_model(messages), "claude-opus-4-6")

    def test_uses_last_assistant(self):
        messages = [
            make_assistant_with_model(0, "claude-sonnet-4-5"),
            make_assistant_with_model(1, "claude-opus-4-6"),
        ]
        self.assertEqual(detect_model(messages), "claude-opus-4-6")

    def test_skips_sidechain(self):
        sidechain = make_assistant_with_model(1, "claude-haiku-4-5")
        sidechain_msg = sidechain[1]
        sidechain_msg["isSidechain"] = True
        messages = [
            make_assistant_with_model(0, "claude-opus-4-6"),
            (1, sidechain_msg, sidechain[2]),
        ]
        self.assertEqual(detect_model(messages), "claude-opus-4-6")

    def test_returns_none_for_empty(self):
        self.assertIsNone(detect_model([]))

    def test_returns_none_for_no_model(self):
        msg = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
            },
        }
        messages = [make_message(0, msg)]
        self.assertIsNone(detect_model(messages))


class TestDetectContextWindow(unittest.TestCase):

    def test_opus_46_is_200k(self):
        messages = [make_assistant_with_model(0, "claude-opus-4-6")]
        self.assertEqual(detect_context_window(messages), 200_000)

    def test_sonnet_46_is_200k(self):
        messages = [make_assistant_with_model(0, "claude-sonnet-4-6")]
        self.assertEqual(detect_context_window(messages), 200_000)

    def test_unknown_model_falls_back(self):
        messages = [make_assistant_with_model(0, "claude-future-99")]
        self.assertEqual(detect_context_window(messages), DEFAULT_CONTEXT_WINDOW)

    def test_prefix_match(self):
        """Versioned model IDs like claude-opus-4-6-20260301 should match."""
        messages = [make_assistant_with_model(0, "claude-opus-4-6-20260301")]
        self.assertEqual(detect_context_window(messages), 200_000)

    def test_env_override(self):
        messages = [make_assistant_with_model(0, "claude-opus-4-6")]
        with patch.dict(os.environ, {"COZEMPIC_CONTEXT_WINDOW": "1000000"}):
            self.assertEqual(detect_context_window(messages), 1_000_000)

    def test_env_override_beats_model(self):
        messages = [make_assistant_with_model(0, "claude-sonnet-4-6")]
        with patch.dict(os.environ, {"COZEMPIC_CONTEXT_WINDOW": "500000"}):
            self.assertEqual(detect_context_window(messages), 500_000)

    def test_invalid_env_override_ignored(self):
        messages = [make_assistant_with_model(0, "claude-opus-4-6")]
        with patch.dict(os.environ, {"COZEMPIC_CONTEXT_WINDOW": "not_a_number"}):
            self.assertEqual(detect_context_window(messages), 200_000)

    # ─── 1M context window tests ────────────────────────────────────────────

    def test_opus_46_1m_exact_match(self):
        """claude-opus-4-6[1m] should return 1M context."""
        messages = [make_assistant_with_model(0, "claude-opus-4-6[1m]")]
        self.assertEqual(detect_context_window(messages), 1_000_000)

    def test_sonnet_46_1m_exact_match(self):
        """claude-sonnet-4-6[1m] should return 1M context."""
        messages = [make_assistant_with_model(0, "claude-sonnet-4-6[1m]")]
        self.assertEqual(detect_context_window(messages), 1_000_000)

    def test_opus_45_1m_exact_match(self):
        """claude-opus-4-5[1m] should return 1M context."""
        messages = [make_assistant_with_model(0, "claude-opus-4-5[1m]")]
        self.assertEqual(detect_context_window(messages), 1_000_000)

    def test_haiku_45_1m_exact_match(self):
        """claude-haiku-4-5[1m] should return 1M context."""
        messages = [make_assistant_with_model(0, "claude-haiku-4-5[1m]")]
        self.assertEqual(detect_context_window(messages), 1_000_000)

    def test_1m_versioned_prefix_match(self):
        """claude-opus-4-6-20260301[1m] should match via prefix logic."""
        messages = [make_assistant_with_model(0, "claude-opus-4-6-20260301[1m]")]
        self.assertEqual(detect_context_window(messages), 1_000_000)

    def test_1m_sonnet_versioned_prefix_match(self):
        """claude-sonnet-4-6-20260301[1m] should match via prefix logic."""
        messages = [make_assistant_with_model(0, "claude-sonnet-4-6-20260301[1m]")]
        self.assertEqual(detect_context_window(messages), 1_000_000)

    def test_versioned_200k_not_confused_with_1m(self):
        """claude-opus-4-6-20260301 (no [1m]) should stay 200K."""
        messages = [make_assistant_with_model(0, "claude-opus-4-6-20260301")]
        self.assertEqual(detect_context_window(messages), 200_000)

    def test_env_override_beats_1m_model(self):
        """Env var override should take priority over [1m] model detection."""
        messages = [make_assistant_with_model(0, "claude-opus-4-6[1m]")]
        with patch.dict(os.environ, {"COZEMPIC_CONTEXT_WINDOW": "500000"}):
            self.assertEqual(detect_context_window(messages), 500_000)


class TestGetContextWindowOverride(unittest.TestCase):

    def test_returns_none_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(get_context_window_override())

    def test_returns_int_when_set(self):
        with patch.dict(os.environ, {"COZEMPIC_CONTEXT_WINDOW": "1000000"}):
            self.assertEqual(get_context_window_override(), 1_000_000)

    def test_returns_none_for_invalid(self):
        with patch.dict(os.environ, {"COZEMPIC_CONTEXT_WINDOW": "abc"}):
            self.assertIsNone(get_context_window_override())


class TestEstimateSessionTokensWithModel(unittest.TestCase):

    def test_includes_model_in_result(self):
        messages = [make_assistant_with_model(0, "claude-opus-4-6", input_tokens=50000)]
        te = estimate_session_tokens(messages)
        self.assertEqual(te.model, "claude-opus-4-6")
        self.assertEqual(te.context_window, 200_000)
        self.assertEqual(te.total, 50000)
        self.assertEqual(te.context_pct, 25.0)

    def test_context_pct_uses_detected_window(self):
        """100K tokens on a 200K window should be 50%."""
        messages = [make_assistant_with_model(0, "claude-sonnet-4-6", input_tokens=100000)]
        te = estimate_session_tokens(messages)
        self.assertEqual(te.context_pct, 50.0)

    def test_1m_model_context_pct(self):
        """100K tokens on a 1M window should be 10%."""
        messages = [make_assistant_with_model(0, "claude-opus-4-6[1m]", input_tokens=100000)]
        te = estimate_session_tokens(messages)
        self.assertEqual(te.model, "claude-opus-4-6[1m]")
        self.assertEqual(te.context_window, 1_000_000)
        self.assertEqual(te.context_pct, 10.0)

    def test_1m_model_500k_tokens(self):
        """500K tokens on a 1M window should be 50%."""
        messages = [make_assistant_with_model(0, "claude-opus-4-6[1m]", input_tokens=500000)]
        te = estimate_session_tokens(messages)
        self.assertEqual(te.context_pct, 50.0)


class TestDetectModel1M(unittest.TestCase):
    """Test that detect_model() correctly returns [1m]-suffixed model IDs."""

    def test_detects_1m_model(self):
        messages = [make_assistant_with_model(0, "claude-opus-4-6[1m]")]
        self.assertEqual(detect_model(messages), "claude-opus-4-6[1m]")

    def test_detects_1m_versioned_model(self):
        messages = [make_assistant_with_model(0, "claude-sonnet-4-6-20260301[1m]")]
        self.assertEqual(detect_model(messages), "claude-sonnet-4-6-20260301[1m]")


class TestDefaultTokenThresholds1M(unittest.TestCase):
    """Test that token thresholds scale correctly with 1M context."""

    def test_200k_thresholds(self):
        from cozempic.tokens import default_token_thresholds
        hard, soft = default_token_thresholds(200_000)
        self.assertEqual(hard, 150_000)   # 75% of 200K
        self.assertEqual(soft, 90_000)    # 45% of 200K

    def test_1m_thresholds(self):
        from cozempic.tokens import default_token_thresholds
        hard, soft = default_token_thresholds(1_000_000)
        self.assertEqual(hard, 750_000)   # 75% of 1M
        self.assertEqual(soft, 450_000)   # 45% of 1M


if __name__ == "__main__":
    unittest.main()
