"""Tests for cozempic._validation — generic helpers used by strategies, CLI,
and env-var parsing."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from cozempic._validation import (
    ConfigError,
    coerce_choice,
    coerce_non_negative_int,
    coerce_positive_float,
    coerce_positive_int,
    parse_env_non_negative_int,
    parse_env_positive_int,
)


class TestCoercePositiveInt(unittest.TestCase):
    """Strict > 0. Distinct from coerce_non_negative_int (which allows 0)."""

    def test_returns_default_when_absent(self):
        self.assertEqual(coerce_positive_int({}, "k", default=30), 30)

    def test_returns_value_when_positive(self):
        self.assertEqual(coerce_positive_int({"k": 5}, "k", default=30), 5)

    def test_rejects_zero(self):
        with self.assertRaises(ConfigError) as ctx:
            coerce_positive_int({"k": 0}, "k", default=30)
        self.assertIn("positive", str(ctx.exception))

    def test_rejects_negative(self):
        with self.assertRaises(ConfigError):
            coerce_positive_int({"k": -1}, "k", default=30)

    def test_rejects_float(self):
        with self.assertRaises(ConfigError):
            coerce_positive_int({"k": 5.5}, "k", default=30)

    def test_rejects_string(self):
        with self.assertRaises(ConfigError):
            coerce_positive_int({"k": "5"}, "k", default=30)

    def test_rejects_bool(self):
        """True is an int in Python but almost never intended here."""
        with self.assertRaises(ConfigError):
            coerce_positive_int({"k": True}, "k", default=30)


class TestCoercePositiveFloat(unittest.TestCase):
    """Strict > 0 for MB thresholds. Accepts int in addition to float."""

    def test_returns_default_when_absent(self):
        self.assertEqual(coerce_positive_float({}, "mb", default=50.0), 50.0)

    def test_accepts_int(self):
        """User writes threshold=50 (int) expecting 50.0 MB — must not reject."""
        result = coerce_positive_float({"mb": 50}, "mb", default=10.0)
        self.assertEqual(result, 50.0)
        self.assertIsInstance(result, float)

    def test_accepts_float(self):
        self.assertEqual(coerce_positive_float({"mb": 50.5}, "mb", default=10.0), 50.5)

    def test_rejects_zero(self):
        with self.assertRaises(ConfigError):
            coerce_positive_float({"mb": 0}, "mb", default=10.0)

    def test_rejects_zero_float(self):
        with self.assertRaises(ConfigError):
            coerce_positive_float({"mb": 0.0}, "mb", default=10.0)

    def test_rejects_negative(self):
        with self.assertRaises(ConfigError):
            coerce_positive_float({"mb": -1.0}, "mb", default=10.0)

    def test_rejects_string(self):
        with self.assertRaises(ConfigError):
            coerce_positive_float({"mb": "50"}, "mb", default=10.0)

    def test_rejects_bool(self):
        with self.assertRaises(ConfigError):
            coerce_positive_float({"mb": True}, "mb", default=10.0)


class TestParseEnvPositiveInt(unittest.TestCase):
    """Env var helper: warn+fallback (does NOT raise). Used for
    COZEMPIC_CONTEXT_WINDOW — zero would cause divide-by-zero downstream."""

    def test_returns_none_when_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_ENV_POSINT", None)
            self.assertIsNone(parse_env_positive_int("TEST_ENV_POSINT"))

    def test_returns_none_when_empty(self):
        with patch.dict(os.environ, {"TEST_ENV_POSINT": ""}):
            self.assertIsNone(parse_env_positive_int("TEST_ENV_POSINT"))

    def test_returns_value_when_valid(self):
        with patch.dict(os.environ, {"TEST_ENV_POSINT": "1000000"}):
            self.assertEqual(parse_env_positive_int("TEST_ENV_POSINT"), 1000000)

    def test_returns_none_on_zero(self):
        """The falsy-trap bug: `0` currently passes `if val:` test in
        tokens.py and silently ignores the override. We reject it loudly."""
        with patch.dict(os.environ, {"TEST_ENV_POSINT": "0"}):
            self.assertIsNone(parse_env_positive_int("TEST_ENV_POSINT"))

    def test_returns_none_on_negative(self):
        with patch.dict(os.environ, {"TEST_ENV_POSINT": "-100"}):
            self.assertIsNone(parse_env_positive_int("TEST_ENV_POSINT"))

    def test_returns_none_on_non_numeric(self):
        with patch.dict(os.environ, {"TEST_ENV_POSINT": "abc"}):
            self.assertIsNone(parse_env_positive_int("TEST_ENV_POSINT"))

    def test_warns_on_invalid(self):
        """User should see a message on stderr — silent swallow is a UX bug."""
        import io
        import contextlib
        buf = io.StringIO()
        with patch.dict(os.environ, {"TEST_ENV_POSINT": "-100"}):
            with contextlib.redirect_stderr(buf):
                parse_env_positive_int("TEST_ENV_POSINT")
        self.assertIn("TEST_ENV_POSINT", buf.getvalue())
        self.assertIn("-100", buf.getvalue())

    def test_silent_when_unset(self):
        """No warning when the var is simply not set — that's the normal path."""
        import io
        import contextlib
        buf = io.StringIO()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_ENV_POSINT", None)
            with contextlib.redirect_stderr(buf):
                parse_env_positive_int("TEST_ENV_POSINT")
        self.assertEqual(buf.getvalue(), "")


class TestParseEnvNonNegativeInt(unittest.TestCase):
    """Like positive-int but accepts 0 (valid for system_overhead_tokens —
    a session with no rules file legitimately has zero overhead)."""

    def test_accepts_zero(self):
        with patch.dict(os.environ, {"TEST_ENV_NNINT": "0"}):
            self.assertEqual(parse_env_non_negative_int("TEST_ENV_NNINT"), 0)

    def test_returns_value_when_positive(self):
        with patch.dict(os.environ, {"TEST_ENV_NNINT": "25000"}):
            self.assertEqual(parse_env_non_negative_int("TEST_ENV_NNINT"), 25000)

    def test_rejects_negative(self):
        with patch.dict(os.environ, {"TEST_ENV_NNINT": "-1"}):
            self.assertIsNone(parse_env_non_negative_int("TEST_ENV_NNINT"))

    def test_rejects_non_numeric(self):
        with patch.dict(os.environ, {"TEST_ENV_NNINT": "xyz"}):
            self.assertIsNone(parse_env_non_negative_int("TEST_ENV_NNINT"))


# ── Backwards compat: re-exports from strategies/_config still work ────────

class TestBackwardsCompatReExport(unittest.TestCase):
    """strategies/_config.py re-exports these — existing strategy imports
    must continue to resolve after the refactor."""

    def test_reexport_coerce_non_negative_int(self):
        from cozempic.strategies._config import coerce_non_negative_int as reexported
        self.assertIs(reexported, coerce_non_negative_int)

    def test_reexport_coerce_choice(self):
        from cozempic.strategies._config import coerce_choice as reexported
        self.assertIs(reexported, coerce_choice)

    def test_reexport_ConfigError(self):
        from cozempic.strategies._config import ConfigError as reexported
        self.assertIs(reexported, ConfigError)


class TestParseEnvBool(unittest.TestCase):
    """Env var helper for boolean flags.

    Truthy tokens:  1 / true / yes / on  (case-insensitive, whitespace stripped)
    Falsy tokens:   0 / false / no / off (case-insensitive, whitespace stripped)
    Absent / empty: return default silently.
    Unrecognized:   warn to stderr, return default.
    """

    _VAR = "TEST_ENV_BOOL"

    def setUp(self):
        # Import here so the test fails with ImportError (not AttributeError)
        # until the helper is implemented — correct RED failure mode.
        from cozempic._validation import parse_env_bool
        self.parse_env_bool = parse_env_bool

    def _call(self, raw=None, default=False, warn=True):
        env = {self._VAR: raw} if raw is not None else {}
        with patch.dict(os.environ, env, clear=False):
            if raw is None:
                os.environ.pop(self._VAR, None)
            return self.parse_env_bool(self._VAR, default=default, warn=warn)

    # ── absent / empty ──────────────────────────────────────────────────────

    def test_absent_returns_default_false(self):
        self.assertFalse(self._call())

    def test_empty_returns_default_false(self):
        self.assertFalse(self._call(raw=""))

    def test_absent_with_default_true(self):
        self.assertTrue(self._call(default=True))

    # ── truthy tokens ────────────────────────────────────────────────────────

    def test_true_token_1(self):
        self.assertTrue(self._call(raw="1"))

    def test_true_token_true(self):
        self.assertTrue(self._call(raw="true"))

    def test_true_token_True_mixed_case(self):
        self.assertTrue(self._call(raw="True"))

    def test_true_token_yes(self):
        self.assertTrue(self._call(raw="yes"))

    def test_true_token_YES_uppercase(self):
        self.assertTrue(self._call(raw="YES"))

    def test_true_token_on(self):
        self.assertTrue(self._call(raw="on"))

    # ── falsy tokens ─────────────────────────────────────────────────────────

    def test_false_token_0(self):
        self.assertFalse(self._call(raw="0"))

    def test_false_token_false(self):
        self.assertFalse(self._call(raw="false"))

    def test_false_token_no(self):
        self.assertFalse(self._call(raw="no"))

    def test_false_token_off(self):
        self.assertFalse(self._call(raw="off"))

    # ── unrecognized: warn + return default ──────────────────────────────────

    def test_unrecognized_warns_and_returns_default(self):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            result = self._call(raw="foo")
        self.assertFalse(result)
        self.assertGreater(len(buf.getvalue()), 0, "expected a warning on stderr")

    def test_unrecognized_includes_var_name_in_warning(self):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            self._call(raw="maybe")
        self.assertIn(self._VAR, buf.getvalue())

    # ── whitespace stripping ─────────────────────────────────────────────────

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace must be stripped before token lookup."""
        self.assertTrue(self._call(raw="  true  "))

    def test_whitespace_stripped_numeric(self):
        """Whitespace around a numeric truthy token must also be stripped."""
        self.assertTrue(self._call(raw="  1  "))

    def test_whitespace_stripped_uppercase_on(self):
        """Whitespace + uppercase truthy token 'ON' → True (strip + lower)."""
        self.assertTrue(self._call(raw="  ON  "))

    # ── warn=False suppresses stderr on unrecognized ─────────────────────────

    def test_warn_false_suppresses_warning(self):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            result = self._call(raw="garbage", warn=False)
        self.assertFalse(result)
        self.assertEqual(buf.getvalue(), "", "expected NO warning when warn=False")

    def test_warn_false_recognized_token_still_works(self):
        """warn=False must not suppress recognized-token parsing."""
        self.assertTrue(self._call(raw="yes", warn=False))

    # ── whitespace-only input ────────────────────────────────────────────────

    def test_whitespace_only_returns_default_silently(self):
        """COZEMPIC_DEBUG='   ' must return default WITHOUT a warning.

        The bug: raw == "" guard fires BEFORE strip(), so '   ' slips
        through as non-empty → normalized="" → not in token sets → spurious
        warning on stderr. Fix: check raw.strip() == "" instead of raw == "".
        """
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            result = self._call(raw="   ")
        self.assertFalse(result, "whitespace-only must return default (False)")
        self.assertEqual(buf.getvalue(), "",
                         "whitespace-only must produce NO warning on stderr")


if __name__ == "__main__":
    unittest.main()
