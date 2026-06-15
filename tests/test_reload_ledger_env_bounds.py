"""Tests for reload-storm guard env-var upper clamps.

The reload-rate ledger uses two env-overridable knobs:

  COZEMPIC_RELOAD_WINDOW_S  — look-back window in seconds (default 600)
  COZEMPIC_RELOAD_MAX       — max reloads allowed per window (default 3)

Without upper bounds, a huge env value (e.g. from a typo like
COZEMPIC_RELOAD_WINDOW_S=86400000 meaning "86400 with extra zeros") makes
the ledger window effectively infinite, so the storm guard NEVER fires.
This silently disables the protection for the session's lifetime.

RED-at-base proof strategy:
  - Set env var to a value far above the expected ceiling.
  - Call the ledger accessor function.
  - Assert the returned value is <= the ceiling.
  - These assertions FAIL at base (raw huge int passes through), PASS after fix.

Regression guard:
  - Normal in-range values must not be altered by the fix.
  - Default (env unset) must return the documented default.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch


class TestReloadLedgerWindowSEnvBounds(unittest.TestCase):
    """_reload_ledger_window_s() must clamp to <= 86400 s (1 day)."""

    def _call(self, env_val=None):
        # Import inside test to get the live (post-patch) definition.
        from cozempic.guard import _reload_ledger_window_s

        env = {}
        if env_val is not None:
            env["COZEMPIC_RELOAD_WINDOW_S"] = env_val
        with patch.dict(os.environ, env, clear=False):
            if env_val is None:
                os.environ.pop("COZEMPIC_RELOAD_WINDOW_S", None)
            return _reload_ledger_window_s()

    # ── RED-at-base: huge value must not pass through ─────────────────────────

    def test_huge_window_env_is_clamped(self):
        """A 23-digit env value silently disables the storm guard at base.

        RED at base: max(60, int('10000000000000000000000')) == 10^22
        (>> 86400 s / 1 day), so the ledger never expires and the guard
        never fires.
        GREEN after fix: parse_env_positive_int with maximum=86400 warns
        and returns None → fallback 600.
        """
        result = self._call("10000000000000000000000")
        self.assertLessEqual(
            result,
            86400,
            f"_reload_ledger_window_s returned {result} for a huge env value "
            f"— storm guard is silently disabled",
        )

    def test_value_above_ceiling_is_clamped(self):
        """86401 is just one second above the documented 86400 ceiling.

        RED at base: max(60, 86401) == 86401 (passes through unclamped).
        GREEN after fix: warns and returns fallback 600.
        """
        result = self._call("86401")
        self.assertLessEqual(
            result,
            86400,
            f"_reload_ledger_window_s returned {result} for env=86401 "
            f"— must reject values above the 86400 ceiling",
        )

    # ── Regression guards (must pass at base AND after fix) ──────────────────

    def test_default_when_unset(self):
        """Env absent → default 600 seconds."""
        self.assertEqual(self._call(), 600)

    def test_valid_value_within_bounds(self):
        """A value well within [60, 86400] is returned as-is (clamped to floor)."""
        self.assertEqual(self._call("1800"), 1800)

    def test_floor_clamp_at_60(self):
        """Values below the floor 60 are raised to 60."""
        result = self._call("30")
        self.assertGreaterEqual(result, 60)

    def test_ceiling_value_exact(self):
        """86400 exactly is at the boundary — must be accepted (not rejected)."""
        result = self._call("86400")
        self.assertEqual(result, 86400)

    def test_invalid_string_returns_fallback(self):
        """Non-numeric env var falls back to 600 (pre-existing behaviour)."""
        self.assertEqual(self._call("not-a-number"), 600)


class TestReloadLedgerMaxEnvBounds(unittest.TestCase):
    """_reload_ledger_max() must clamp to <= 100 reloads per window."""

    def _call(self, env_val=None):
        from cozempic.guard import _reload_ledger_max

        env = {}
        if env_val is not None:
            env["COZEMPIC_RELOAD_MAX"] = env_val
        with patch.dict(os.environ, env, clear=False):
            if env_val is None:
                os.environ.pop("COZEMPIC_RELOAD_MAX", None)
            return _reload_ledger_max()

    # ── RED-at-base: huge value must not pass through ─────────────────────────

    def test_huge_max_env_is_clamped(self):
        """999999999 reloads allowed → storm guard never trips.

        RED at base: max(1, int('999999999')) == 999999999 (guard never fires).
        GREEN after fix: parse_env_positive_int with maximum=100 warns and
        returns None → fallback 3.
        """
        result = self._call("999999999")
        self.assertLessEqual(
            result,
            100,
            f"_reload_ledger_max returned {result} for a huge env value "
            f"— storm guard is silently disabled",
        )

    def test_value_above_ceiling_is_clamped(self):
        """101 is just one above the documented 100 ceiling.

        RED at base: max(1, 101) == 101 (passes through unclamped).
        GREEN after fix: warns and returns fallback 3.
        """
        result = self._call("101")
        self.assertLessEqual(
            result,
            100,
            f"_reload_ledger_max returned {result} for env=101 "
            f"— must reject values above the 100 ceiling",
        )

    # ── Regression guards (must pass at base AND after fix) ──────────────────

    def test_default_when_unset(self):
        """Env absent → default 3."""
        self.assertEqual(self._call(), 3)

    def test_valid_value_within_bounds(self):
        """A value well within [1, 100] is returned as-is."""
        self.assertEqual(self._call("10"), 10)

    def test_floor_clamp_at_1(self):
        """Values below the floor 1 are raised to 1."""
        result = self._call("0")
        self.assertGreaterEqual(result, 1)

    def test_ceiling_value_exact(self):
        """100 exactly is at the boundary — must be accepted (not rejected)."""
        result = self._call("100")
        self.assertEqual(result, 100)

    def test_invalid_string_returns_fallback(self):
        """Non-numeric env var falls back to 3 (pre-existing behaviour)."""
        self.assertEqual(self._call("not-a-number"), 3)
