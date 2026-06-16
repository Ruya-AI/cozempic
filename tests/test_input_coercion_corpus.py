"""Standing safeguard (1.8.23): a table-driven ADVERSARIAL CORPUS thrown at every
numeric input-validator, so the NaN/inf/huge-int "gate-disable" class (PR #116 +
the COZEMPIC_RELOAD_WARN_GRACE follow-up) cannot regress.

Why this exists: IEEE-754 makes every comparison with NaN False, and `inf <= 0` is
False, so a NaN/inf that slips past a `value <= 0` check silently DISABLES the
threshold/gate it controls (it compares False to everything). A huge int (10**400)
does the same by magnitude. These bugs are silent — no crash — so logic tests miss
them. This corpus catches the whole class with an OUTPUT-shaped invariant:

    For every adversarial input, a validator must EITHER reject it (raise / clean
    argparse error) OR return a finite, in-range, correctly-typed value.

Adding a new COZEMPIC_* numeric knob? Add one row to _ENV_VALIDATORS and the corpus
is applied automatically. Zero dependencies (cozempic is stdlib-only — no hypothesis).

Extended in PR-2 (input-validation hardening) to cover:
  P-A: parse_env_positive_int / parse_env_non_negative_int upper-bound (huge-int path)
  P-B: _clamp_float / _clamp_int bool-rejection
  P-C: cli._apply_token_env_overrides truthiness fix (0 is a valid system-overhead value)
"""

import argparse
import math
import os
import types
import unittest
from contextlib import contextmanager
from unittest import mock

import cozempic.cli as cli
import cozempic.config as config
import cozempic.guard as g
import cozempic.tokens as t
from cozempic._validation import (
    ConfigError,
    coerce_positive_float,
    coerce_positive_int,
    parse_env_non_negative_int,
    parse_env_positive_int,
)

# env / CLI inputs are ALWAYS strings
STR_CORPUS = ["nan", "NaN", "inf", "+inf", "-inf", "infinity", "1e999", "-1e999",
              "-0", "", "   ", "0", "-1", "-0.5", "abc", "١٢٣",
              "1" + "0" * 400, "1.5", "50"]
# Shared huge-int sentinel — used in both the native corpus and the P-A upper-bound tests.
# A single definition avoids the 10**400 expression being evaluated twice at module load.
_HUGE_INT = 10 ** 400

# native-typed corpus for the config-DICT helpers (built in-process, not from a string)
NATIVE_CORPUS = [float("nan"), float("inf"), float("-inf"), -0.0, _HUGE_INT,
                 -1, 0, True, False, None, "x", [], {}]

_UPPER = 10 ** 12  # any CLI/env validator output must be well under this

# Pre-existing constants safe to reference at module scope (present on origin/main).
_DEFAULT_CONTEXT_WINDOW = t.DEFAULT_CONTEXT_WINDOW  # 1_000_000
_SYSTEM_OVERHEAD_DEFAULT = t.SYSTEM_OVERHEAD_TOKENS  # 21_000
# NOTE: t.MAX_CONTEXT_WINDOW is NEW (not on origin/main), so it must be
# accessed inside test methods to avoid an AttributeError at collection against base.


@contextmanager
def _env(name, value):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


_TOKEN_ENV_KEYS = ("COZEMPIC_CONTEXT_WINDOW", "COZEMPIC_SYSTEM_OVERHEAD_TOKENS")


def _token_env_clean() -> dict:
    """Return current environ with both token override keys removed.

    Used with ``mock.patch.dict(..., clear=True)`` to give each test a known-clean
    token env without affecting unrelated env vars.  Shared by tests that exercise
    either _apply_token_env_overrides (both keys) or _prescan_argv (one key), so
    both classes exclude the full pair and test isolation is consistent.
    """
    return {k: v for k, v in os.environ.items() if k not in _TOKEN_ENV_KEYS}


def _assert_finite(tc, r):
    tc.assertIsInstance(r, (int, float))
    if isinstance(r, float):
        tc.assertTrue(math.isfinite(r), f"validator leaked a non-finite value: {r!r}")


def _assert_finite_inrange(tc, r):
    _assert_finite(tc, r)
    tc.assertLessEqual(abs(r), _UPPER, f"validator leaked an out-of-bound value: {r!r}")


# (validator, ENV var) — these return a default on bad input (never raise), so the
# RETURN value must always be sane.
_ENV_VALIDATORS = [
    (g._reload_warn_grace,        "COZEMPIC_RELOAD_WARN_GRACE"),
    (g._force_reload_pct,         "COZEMPIC_FORCE_RELOAD_PCT"),
    (g._idle_reload_cycles,       "COZEMPIC_IDLE_RELOAD_CYCLES"),
    (g._idle_backoff_cycles,      "COZEMPIC_IDLE_BACKOFF_CYCLES"),
    (g._read_min_prune_ratio,     "COZEMPIC_MIN_PRUNE_RATIO"),
    (g._read_hard_exit_threshold, "COZEMPIC_GUARD_HARD_EXIT_K"),
    (t.get_chars_per_token,       "COZEMPIC_CHARS_PER_TOKEN"),
    # ── PR #137 additions — maintenance contract: one row per new knob ────────
    (g._reload_ledger_window_s,   "COZEMPIC_RELOAD_WINDOW_S"),
    (g._reload_ledger_max,        "COZEMPIC_RELOAD_MAX"),
]


class TestEnvNumericCorpus(unittest.TestCase):
    def test_env_validators_never_leak_nonfinite_or_huge(self):
        for fn, var in _ENV_VALIDATORS:
            for raw in STR_CORPUS:
                with self.subTest(fn=fn.__name__, value=raw), _env(var, raw):
                    _assert_finite_inrange(self, fn())


class TestCliValidatorCorpus(unittest.TestCase):
    def test_positive_float_int_reject_or_return_sane(self):
        for fn in (cli._positive_float, cli._positive_int):
            for raw in STR_CORPUS:
                with self.subTest(fn=fn.__name__, value=raw):
                    try:
                        r = fn(raw)
                    except argparse.ArgumentTypeError:
                        continue  # a clean reject is acceptable
                    _assert_finite_inrange(self, r)


class TestConfigDictCorpus(unittest.TestCase):
    def test_coerce_helpers_reject_or_return_finite(self):
        for fn in (coerce_positive_float, coerce_positive_int):
            for val in NATIVE_CORPUS:
                with self.subTest(fn=fn.__name__, value=repr(val)):
                    try:
                        r = fn({"k": val}, "k", 1)
                    except (ConfigError, ValueError, TypeError):
                        continue  # a clean reject is acceptable
                    _assert_finite(self, r)  # a silent-accept of nan/inf fails here


# ── P-A: env-parser upper-bound (huge-int) ────────────────────────────────────


class TestParseEnvPositiveIntMaximum(unittest.TestCase):
    """P-A: parse_env_positive_int must reject values above the maximum kwarg."""

    def test_huge_int_rejected_with_maximum(self):
        """10**400 > maximum=4_000_000 → must return None (not the huge int)."""
        with mock.patch.dict(os.environ, {"COZEMPIC_TEST_CW": str(_HUGE_INT)}):
            result = parse_env_positive_int("COZEMPIC_TEST_CW", maximum=t.MAX_CONTEXT_WINDOW)
        self.assertIsNone(result, f"huge int leaked through: {result!r}")

    def test_above_maximum_rejected(self):
        """5_000_000 > 4_000_000 → None."""
        with mock.patch.dict(os.environ, {"COZEMPIC_TEST_CW": "5000000"}):
            result = parse_env_positive_int("COZEMPIC_TEST_CW", maximum=t.MAX_CONTEXT_WINDOW)
        self.assertIsNone(result, f"above-max value leaked through: {result!r}")

    def test_below_maximum_accepted(self):
        """200_000 <= 4_000_000 → 200_000 (valid override)."""
        with mock.patch.dict(os.environ, {"COZEMPIC_TEST_CW": "200000"}):
            result = parse_env_positive_int("COZEMPIC_TEST_CW", maximum=t.MAX_CONTEXT_WINDOW)
        self.assertEqual(result, 200_000)

    def test_no_maximum_still_works(self):
        """Without maximum= kwarg the old behavior (no upper bound) is unchanged."""
        with mock.patch.dict(os.environ, {"COZEMPIC_TEST_CW": "200000"}):
            result = parse_env_positive_int("COZEMPIC_TEST_CW")
        self.assertEqual(result, 200_000)


class TestParseEnvNonNegativeIntMaximum(unittest.TestCase):
    """P-A: parse_env_non_negative_int must reject values above the maximum kwarg."""

    def test_huge_int_rejected_with_maximum(self):
        with mock.patch.dict(os.environ, {"COZEMPIC_TEST_SOH": str(_HUGE_INT)}):
            result = parse_env_non_negative_int(
                "COZEMPIC_TEST_SOH", maximum=_DEFAULT_CONTEXT_WINDOW
            )
        self.assertIsNone(result, f"huge int leaked through: {result!r}")

    def test_zero_still_valid(self):
        """0 is a legitimate 'no overhead' value and must not be rejected."""
        with mock.patch.dict(os.environ, {"COZEMPIC_TEST_SOH": "0"}):
            result = parse_env_non_negative_int(
                "COZEMPIC_TEST_SOH", maximum=_DEFAULT_CONTEXT_WINDOW
            )
        self.assertEqual(result, 0)


# ── P-A: tokens-layer integration (uses the constants once they exist) ────────


class TestTokensUpperBound(unittest.TestCase):
    """P-A: get_context_window_override and get_system_overhead_tokens must
    apply the maximum= bound so a huge env var can never silently disable the guard."""

    def test_context_window_huge_int_returns_none(self):
        with mock.patch.dict(os.environ,
                             {"COZEMPIC_CONTEXT_WINDOW": str(_HUGE_INT)}, clear=False):
            result = t.get_context_window_override()
        self.assertIsNone(result, f"huge int leaked from get_context_window_override: {result!r}")

    def test_context_window_5m_returns_none(self):
        """5_000_000 > MAX_CONTEXT_WINDOW (4_000_000) → None."""
        with mock.patch.dict(os.environ,
                             {"COZEMPIC_CONTEXT_WINDOW": "5000000"}, clear=False):
            result = t.get_context_window_override()
        self.assertIsNone(result, f"above-max value leaked: {result!r}")

    def test_context_window_200k_accepted(self):
        with mock.patch.dict(os.environ,
                             {"COZEMPIC_CONTEXT_WINDOW": "200000"}, clear=False):
            result = t.get_context_window_override()
        self.assertEqual(result, 200_000)

    def test_system_overhead_huge_int_falls_back_to_default(self):
        """Huge COZEMPIC_SYSTEM_OVERHEAD_TOKENS → falls back to SYSTEM_OVERHEAD_TOKENS (21000)."""
        with mock.patch.dict(os.environ,
                             {"COZEMPIC_SYSTEM_OVERHEAD_TOKENS": str(_HUGE_INT)}, clear=False):
            result = t.get_system_overhead_tokens()
        self.assertEqual(result, _SYSTEM_OVERHEAD_DEFAULT,
                         f"huge int leaked from get_system_overhead_tokens: {result!r}")

    def test_system_overhead_zero_accepted(self):
        """0 is a legitimate 'no overhead' value → must return 0, not the default."""
        with mock.patch.dict(os.environ,
                             {"COZEMPIC_SYSTEM_OVERHEAD_TOKENS": "0"}, clear=False):
            result = t.get_system_overhead_tokens()
        self.assertEqual(result, 0,
                         "0 system-overhead was silently dropped (truthiness / upper-bound bug)")


# ── P-B: config clamp helpers must reject bool ────────────────────────────────


class TestClampBoolRejection(unittest.TestCase):
    """P-B: _clamp_float and _clamp_int must treat bool as invalid (return default)."""

    def test_clamp_float_true_returns_default(self):
        """True coerces to 1.0 in Python, which is in-range — must return default instead."""
        result = config._clamp_float(True, 0.0, 1.0, 0.5)
        self.assertEqual(result, 0.5,
                         f"bool True leaked through _clamp_float as {result!r} instead of default 0.5")

    def test_clamp_float_false_returns_default(self):
        result = config._clamp_float(False, 0.0, 1.0, 0.5)
        self.assertEqual(result, 0.5)

    def test_clamp_int_true_returns_default(self):
        """True == 1 in Python, which passes range check — must return default instead."""
        result = config._clamp_int(True, 0, 10, 5)
        self.assertEqual(result, 5,
                         f"bool True leaked through _clamp_int as {result!r} instead of default 5")

    def test_clamp_int_false_returns_default(self):
        result = config._clamp_int(False, 0, 10, 5)
        self.assertEqual(result, 5)

    def test_clamp_float_valid_value_still_works(self):
        """Sanity: a valid float must not be broken by the bool guard."""
        result = config._clamp_float(0.9, 0.0, 1.0, 0.5)
        self.assertAlmostEqual(result, 0.9)

    def test_clamp_int_valid_value_still_works(self):
        result = config._clamp_int(7, 0, 10, 5)
        self.assertEqual(result, 7)


# ── P-C: cli._apply_token_env_overrides (DRY helper + truthiness fix) ─────────


class TestApplyTokenEnvOverrides(unittest.TestCase):
    """P-C: _apply_token_env_overrides must set env vars using `is not None`,
    so --system-overhead-tokens 0 (legitimate 'no overhead') is honored."""

    def test_zero_system_overhead_sets_env(self):
        """0 is a valid 'no overhead' value — must NOT be silently dropped."""
        args = types.SimpleNamespace(system_overhead_tokens=0, context_window=None)
        with mock.patch.dict(os.environ, _token_env_clean(), clear=True):
            cli._apply_token_env_overrides(args)
            self.assertEqual(os.environ.get("COZEMPIC_SYSTEM_OVERHEAD_TOKENS"), "0",
                             "system_overhead_tokens=0 was silently dropped (truthiness bug)")
            self.assertNotIn("COZEMPIC_CONTEXT_WINDOW", os.environ)

    def test_zero_context_window_sets_env(self):
        """context_window=0: also set (downstream parse_env_positive_int will reject it,
        but the CLI layer must not silently drop it first)."""
        args = types.SimpleNamespace(system_overhead_tokens=None, context_window=0)
        with mock.patch.dict(os.environ, _token_env_clean(), clear=True):
            cli._apply_token_env_overrides(args)
            self.assertEqual(os.environ.get("COZEMPIC_CONTEXT_WINDOW"), "0")
            self.assertNotIn("COZEMPIC_SYSTEM_OVERHEAD_TOKENS", os.environ)

    def test_both_none_sets_neither(self):
        """When both attrs are None, neither env var is touched."""
        args = types.SimpleNamespace(system_overhead_tokens=None, context_window=None)
        with mock.patch.dict(os.environ, _token_env_clean(), clear=True):
            cli._apply_token_env_overrides(args)
            self.assertNotIn("COZEMPIC_CONTEXT_WINDOW", os.environ)
            self.assertNotIn("COZEMPIC_SYSTEM_OVERHEAD_TOKENS", os.environ)

    def test_positive_values_set_env(self):
        """Normal positive values are set as expected."""
        args = types.SimpleNamespace(system_overhead_tokens=30000, context_window=200000)
        with mock.patch.dict(os.environ, _token_env_clean(), clear=True):
            cli._apply_token_env_overrides(args)
            self.assertEqual(os.environ["COZEMPIC_CONTEXT_WINDOW"], "200000")
            self.assertEqual(os.environ["COZEMPIC_SYSTEM_OVERHEAD_TOKENS"], "30000")


# ── H-1: _prescan_argv must allow --system-overhead-tokens 0 through to the env ──


class TestPrescanArgvZeroOverhead(unittest.TestCase):
    """H-1: _prescan_argv uses `< 0` (not `<= 0`) for --system-overhead-tokens,
    so the legitimate 'no overhead' value 0 reaches the env var and is not silently
    dropped with a spurious warning."""

    def test_zero_overhead_reaches_env_via_prescan(self):
        """E2E: drive argv through _prescan_argv; assert env var is set to '0'.

        This test is RED at base (prescan rejects 0 with `<= 0` gate) and GREEN
        after the fix (gate changed to `< 0`).
        """
        with mock.patch.dict(os.environ, _token_env_clean(), clear=True):
            cleaned = cli._prescan_argv(["guard", "--system-overhead-tokens", "0"])
            self.assertEqual(
                os.environ.get("COZEMPIC_SYSTEM_OVERHEAD_TOKENS"), "0",
                "--system-overhead-tokens 0 was silently dropped by _prescan_argv "
                "(gate was `<= 0`; must be `< 0` so 0 is passed through)"
            )
        # The flag must be consumed by prescan (not left for argparse to see)
        self.assertNotIn("--system-overhead-tokens", cleaned)
        self.assertNotIn("0", cleaned)

    def test_zero_overhead_eq_form_reaches_env(self):
        """E2E: --system-overhead-tokens=0 (= form) also goes through prescan correctly."""
        with mock.patch.dict(os.environ, _token_env_clean(), clear=True):
            cli._prescan_argv(["guard", "--system-overhead-tokens=0"])
            self.assertEqual(
                os.environ.get("COZEMPIC_SYSTEM_OVERHEAD_TOKENS"), "0",
                "--system-overhead-tokens=0 (= form) was silently dropped by _prescan_argv"
            )

    def test_negative_overhead_still_rejected(self):
        """Negatives must still be rejected; only 0 was wrongly excluded before."""
        with mock.patch.dict(os.environ, _token_env_clean(), clear=True):
            cli._prescan_argv(["guard", "--system-overhead-tokens", "-1"])
            self.assertIsNone(
                os.environ.get("COZEMPIC_SYSTEM_OVERHEAD_TOKENS"),
                "negative --system-overhead-tokens must still be rejected by prescan"
            )

    def test_context_window_zero_still_rejected(self):
        """--context-window 0 must still be rejected (0 IS invalid for context window)."""
        with mock.patch.dict(os.environ, _token_env_clean(), clear=True):
            cli._prescan_argv(["guard", "--context-window", "0"])
            self.assertIsNone(
                os.environ.get("COZEMPIC_CONTEXT_WINDOW"),
                "--context-window 0 must be rejected (0 is not a valid context window)"
            )


if __name__ == "__main__":
    unittest.main()
