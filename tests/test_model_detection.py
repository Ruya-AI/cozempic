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

    def setUp(self):
        # Isolate from inherited COZEMPIC_CONTEXT_WINDOW env var. Some dev
        # boxes set this globally (e.g., to force a specific window for the
        # guard daemon). When present, get_context_window_override() returns
        # the env value and short-circuits the MODEL_CONTEXT_WINDOWS dict
        # lookup that these tests target — producing "1M != 200K" failures
        # on model-mapping tests that ARE actually correct in production
        # code. Tests that exercise the override path explicitly use
        # patch.dict to set the env var inside their own context.
        self._prev_env = os.environ.pop("COZEMPIC_CONTEXT_WINDOW", None)
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        if self._prev_env is not None:
            os.environ["COZEMPIC_CONTEXT_WINDOW"] = self._prev_env

    def test_opus_47_is_1m(self):
        """Current Opus 4.7 defaults to 1M (standard for Claude Code Max)."""
        messages = [make_assistant_with_model(0, "claude-opus-4-7")]
        self.assertEqual(detect_context_window(messages), 1_000_000)

    def test_opus_46_is_1m(self):
        """Current Opus 4.6 defaults to 1M (standard for Claude Code Max)."""
        messages = [make_assistant_with_model(0, "claude-opus-4-6")]
        self.assertEqual(detect_context_window(messages), 1_000_000)

    def test_sonnet_46_is_1m(self):
        """Current Sonnet 4.6 defaults to 1M."""
        messages = [make_assistant_with_model(0, "claude-sonnet-4-6")]
        self.assertEqual(detect_context_window(messages), 1_000_000)

    def test_opus_45_is_1m(self):
        messages = [make_assistant_with_model(0, "claude-opus-4-5")]
        self.assertEqual(detect_context_window(messages), 1_000_000)

    def test_haiku_45_is_200k(self):
        """Haiku stays at 200K."""
        messages = [make_assistant_with_model(0, "claude-haiku-4-5")]
        self.assertEqual(detect_context_window(messages), 200_000)

    def test_older_models_200k(self):
        """Older claude-3 models are 200K."""
        for model in ["claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku"]:
            messages = [make_assistant_with_model(0, model)]
            self.assertEqual(detect_context_window(messages), 200_000, f"{model} should be 200K")

    def test_unknown_model_falls_back(self):
        messages = [make_assistant_with_model(0, "claude-future-99")]
        self.assertEqual(detect_context_window(messages), DEFAULT_CONTEXT_WINDOW)

    def test_prefix_match_versioned(self):
        """Versioned model IDs like claude-opus-4-7-20260401 should match."""
        messages = [make_assistant_with_model(0, "claude-opus-4-7-20260401")]
        self.assertEqual(detect_context_window(messages), 1_000_000)

    def test_prefix_match_versioned_46(self):
        """Versioned model IDs like claude-opus-4-6-20260301 should match."""
        messages = [make_assistant_with_model(0, "claude-opus-4-6-20260301")]
        self.assertEqual(detect_context_window(messages), 1_000_000)

    def test_env_override(self):
        messages = [make_assistant_with_model(0, "claude-opus-4-6")]
        with patch.dict(os.environ, {"COZEMPIC_CONTEXT_WINDOW": "200000"}):
            self.assertEqual(detect_context_window(messages), 200_000)

    def test_env_override_beats_model(self):
        messages = [make_assistant_with_model(0, "claude-sonnet-4-6")]
        with patch.dict(os.environ, {"COZEMPIC_CONTEXT_WINDOW": "500000"}):
            self.assertEqual(detect_context_window(messages), 500_000)

    def test_invalid_env_override_ignored(self):
        messages = [make_assistant_with_model(0, "claude-opus-4-6")]
        with patch.dict(os.environ, {"COZEMPIC_CONTEXT_WINDOW": "not_a_number"}):
            self.assertEqual(detect_context_window(messages), 1_000_000)


class TestContextWindowFamilyFallback(unittest.TestCase):
    """PR-3 (L10): an unknown future Haiku must size to 200K, not the 1M default."""

    def setUp(self):
        # Same isolation as TestDetectContextWindow — an inherited
        # COZEMPIC_CONTEXT_WINDOW would short-circuit the model-mapping path.
        self._prev_env = os.environ.pop("COZEMPIC_CONTEXT_WINDOW", None)

    def tearDown(self):
        if self._prev_env is not None:
            os.environ["COZEMPIC_CONTEXT_WINDOW"] = self._prev_env

    def test_future_haiku_version_is_200k(self):
        # RED-at-base: a Haiku generation not in MODEL_CONTEXT_WINDOWS currently
        # falls through to the 1M DEFAULT — a 5x over-estimate that mis-times every
        # guard tier on a real 200K Haiku session.
        for model in ["claude-haiku-5", "claude-haiku-4-6", "claude-haiku-4-6-20260601"]:
            messages = [make_assistant_with_model(0, model)]
            self.assertEqual(detect_context_window(messages), 200_000, f"{model} should be 200K")
        # claude-haiku-4-5[1m] — deliberate behaviour CHANGE (1M → 200K), not
        # preservation. The old bracket-aware loop required bracket_suffix == ""
        # to enter the second prefix pass, so a bracket-suffixed Haiku fell through
        # to DEFAULT (1M). The new simple prefix match correctly routes it to 200K
        # via the prefix loop (startswith "claude-haiku-4-5"). This is an
        # improvement, not a regression.
        self.assertEqual(
            detect_context_window([make_assistant_with_model(0, "claude-haiku-4-5[1m]")]),
            200_000,
        )

    def test_known_haiku_still_200k(self):
        messages = [make_assistant_with_model(0, "claude-haiku-4-5")]
        self.assertEqual(detect_context_window(messages), 200_000)

    def test_opus_48_is_1m(self):
        messages = [make_assistant_with_model(0, "claude-opus-4-8")]
        self.assertEqual(detect_context_window(messages), 1_000_000)

    def test_unknown_non_haiku_still_defaults_1m(self):
        messages = [make_assistant_with_model(0, "claude-future-99")]
        self.assertEqual(detect_context_window(messages), DEFAULT_CONTEXT_WINDOW)

    def test_bracket_suffix_opus_resolves_by_prefix(self):
        # CC does not emit "[1m]", but if it ever did for a 1M model, the simple
        # prefix match handles it with no behaviour change vs the old bracket-aware
        # code. This is the no-change direction that justifies removing the dead
        # bracket-handling branch: claude-opus-4-6[1m] → 1M in both old and new code.
        self.assertEqual(
            detect_context_window([make_assistant_with_model(0, "claude-opus-4-6[1m]")]),
            1_000_000,
        )


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

    def setUp(self):
        # Same env-isolation pattern as TestDetectContextWindow — see comment
        # there. estimate_session_tokens() calls detect_context_window()
        # internally, which honours COZEMPIC_CONTEXT_WINDOW. Without this
        # cleanup, test_context_pct_200k_model fails on dev boxes that set
        # the env var globally.
        self._prev_env = os.environ.pop("COZEMPIC_CONTEXT_WINDOW", None)
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        if self._prev_env is not None:
            os.environ["COZEMPIC_CONTEXT_WINDOW"] = self._prev_env

    def test_includes_model_in_result(self):
        messages = [make_assistant_with_model(0, "claude-opus-4-6", input_tokens=50000)]
        te = estimate_session_tokens(messages)
        self.assertEqual(te.model, "claude-opus-4-6")
        self.assertEqual(te.context_window, 1_000_000)  # Opus 4.6 defaults to 1M
        self.assertEqual(te.total, 50100)  # 50000 + 100(output)
        self.assertEqual(te.context_pct, 5.0)  # 50100 / 1M ≈ 5.0%

    def test_context_pct_1m_model(self):
        """100K tokens on a 1M window should be ~10%."""
        messages = [make_assistant_with_model(0, "claude-sonnet-4-6", input_tokens=100000)]
        te = estimate_session_tokens(messages)
        self.assertEqual(te.context_window, 1_000_000)
        self.assertEqual(te.context_pct, 10.0)

    def test_context_pct_200k_model(self):
        """100K tokens on a 200K window (Haiku) should be ~50%."""
        messages = [make_assistant_with_model(0, "claude-haiku-4-5", input_tokens=100000)]
        te = estimate_session_tokens(messages)
        self.assertEqual(te.context_window, 200_000)
        self.assertEqual(te.context_pct, 50.0)

    def test_500k_tokens_on_1m(self):
        """500K tokens on a 1M window should be 50%."""
        messages = [make_assistant_with_model(0, "claude-opus-4-6", input_tokens=500000)]
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
    """Test that token thresholds scale correctly with context window size."""

    def test_200k_thresholds(self):
        from cozempic.tokens import default_token_thresholds
        hard, soft = default_token_thresholds(200_000)
        self.assertEqual(hard, 110_000)   # 55% of 200K (hard1)
        self.assertEqual(soft, 50_000)    # 25% of 200K

    def test_1m_thresholds(self):
        from cozempic.tokens import default_token_thresholds
        hard, soft = default_token_thresholds(1_000_000)
        self.assertEqual(hard, 550_000)   # 55% of 1M (hard1)
        self.assertEqual(soft, 250_000)   # 25% of 1M

    def test_4tier_thresholds(self):
        from cozempic.tokens import default_token_thresholds_4tier
        soft, hard1, hard2 = default_token_thresholds_4tier(1_000_000)
        self.assertEqual(soft, 250_000)    # 25% of 1M
        self.assertEqual(hard1, 550_000)   # 55% of 1M
        self.assertEqual(hard2, 800_000)   # 80% of 1M


if __name__ == "__main__":
    unittest.main()
