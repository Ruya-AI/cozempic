"""Tests for reload-storm guard env-var upper bounds.

The reload-rate ledger uses two env-overridable knobs:

  COZEMPIC_RELOAD_WINDOW_S  — look-back window in seconds (default 600)
  COZEMPIC_RELOAD_MAX       — max reloads allowed per window (default 3)

Without upper bounds, a huge env value (e.g. from a typo like
COZEMPIC_RELOAD_WINDOW_S=86400000 meaning "86400 with extra zeros") makes
the ledger window effectively infinite, so the storm guard NEVER fires.
This silently disables the protection for the session's lifetime.

RED-at-base proof strategy:
  - Set env var to a value above the ceiling.
  - Call the ledger accessor function.
  - Assert the returned value equals the safe default (not a huge int).
  - These assertions FAIL at base (raw huge int passes through), PASS after fix.

Rejection semantics (not clamp semantics):
  Out-of-range values are REJECTED → the safe default is returned (not
  clamped to the ceiling). This is the conservative choice for a
  storm-guard knob: an absurd value falls to the strict default, not the
  lenient ceiling.

Regression guard:
  - Normal in-range values must not be altered by the fix.
  - Default (env unset) must return the documented default.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch


class TestReloadLedgerWindowSEnvBounds(unittest.TestCase):
    """_reload_ledger_window_s() rejects out-of-range values → safe default 600.

    Out-of-range env values are REJECTED → the safe default (not clamped to
    the bound) — the conservative choice for a storm-guard knob.
    """

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

    # ── RED-at-base: out-of-range values must be rejected to default ──────────

    def test_huge_window_env_rejected_falls_back_to_default(self):
        """A 23-digit env value silently disables the storm guard at base.

        RED at base: max(60, int('10000000000000000000000')) == 10^22
        (>> 86400 s / 1 day), so the ledger never expires and the guard
        never fires.
        GREEN after fix: parse_env_positive_int with maximum=86400 rejects
        the value, returns None → fallback default 600.
        """
        result = self._call("10000000000000000000000")
        self.assertEqual(
            result,
            600,
            f"_reload_ledger_window_s returned {result} for a huge env value "
            f"— expected rejection to safe default 600",
        )

    def test_window_above_ceiling_rejected_to_default(self):
        """86401 is just one second above the 86400 ceiling.

        RED at base: max(60, 86401) == 86401 (passes through unchecked).
        GREEN after fix: rejected → fallback default 600.
        """
        result = self._call("86401")
        self.assertEqual(
            result,
            600,
            f"_reload_ledger_window_s returned {result} for env=86401 "
            f"— expected rejection to safe default 600",
        )

    # ── Regression guards (must pass at base AND after fix) ──────────────────

    def test_default_when_unset(self):
        """Env absent → default 600 seconds."""
        self.assertEqual(self._call(), 600)

    def test_valid_value_within_bounds(self):
        """A value well within [60, 86400] is returned as-is."""
        self.assertEqual(self._call("1800"), 1800)

    def test_floor_clamp_at_60(self):
        """Values below the floor 60 are clamped exactly to 60 (not reject).

        Input "30" → max(60, 30) = 60 exactly. assertGreaterEqual would not
        catch a future regression to max(90, v) = 90; assertEqual is precise.
        """
        result = self._call("30")
        self.assertEqual(result, 60)

    def test_ceiling_value_exact(self):
        """86400 exactly is at the boundary — must be accepted (not rejected)."""
        result = self._call("86400")
        self.assertEqual(result, 86400)

    def test_invalid_string_returns_fallback(self):
        """Non-numeric env var falls back to 600 (pre-existing behaviour)."""
        self.assertEqual(self._call("not-a-number"), 600)


class TestReloadLedgerMaxEnvBounds(unittest.TestCase):
    """_reload_ledger_max() rejects out-of-range values → safe default 3.

    Out-of-range env values are REJECTED → the safe default (not clamped to
    the bound) — the conservative choice for a storm-guard knob.
    """

    def _call(self, env_val=None):
        from cozempic.guard import _reload_ledger_max

        env = {}
        if env_val is not None:
            env["COZEMPIC_RELOAD_MAX"] = env_val
        with patch.dict(os.environ, env, clear=False):
            if env_val is None:
                os.environ.pop("COZEMPIC_RELOAD_MAX", None)
            return _reload_ledger_max()

    # ── RED-at-base: out-of-range values must be rejected to default ──────────

    def test_huge_max_env_rejected_falls_back_to_default(self):
        """999999999 reloads allowed → storm guard never trips.

        RED at base: max(1, int('999999999')) == 999999999 (guard never fires).
        GREEN after fix: parse_env_positive_int with maximum=100 rejects
        the value, returns None → fallback default 3.
        """
        result = self._call("999999999")
        self.assertEqual(
            result,
            3,
            f"_reload_ledger_max returned {result} for a huge env value "
            f"— expected rejection to safe default 3",
        )

    def test_max_above_ceiling_rejected_to_default(self):
        """51 is just one above the new 50 ceiling (was 100 at ae85bcc).

        RED at base: max(1, 51) == 51 (passes through unchecked).
        GREEN after fix: parse_env_positive_int with maximum=50 rejects → 3.
        """
        result = self._call("51")
        self.assertEqual(
            result,
            3,
            f"_reload_ledger_max returned {result} for env=51 "
            f"— expected rejection to safe default 3",
        )

    def test_max_above_write_cap_rejected_to_default(self):
        """RELOAD_MAX=80 passes maximum=100 at base and returns 80.

        The storm-guard fires when len(hist) >= max; hist is capped at 50 on
        write, so len(hist) <= 50, making 50 >= 80 always False — guard never
        trips for any value in [51, 100].

        RED at base: max(1, 80) == 80 (accepted; storm-guard silently disabled).
        GREEN after fix: 80 > 50 (new ceiling) → rejected → returns default 3.
        """
        result = self._call("80")
        self.assertEqual(
            result,
            3,
            f"_reload_ledger_max returned {result} for env=80 "
            f"— expected rejection to safe default 3 (incoherent with hist[-50:] write-cap)",
        )

    def test_ceiling_100_rejected_after_fix(self):
        """100 was the former ceiling; after fix it must be rejected.

        RED at base: max(1, 100) == 100 (returned as ceiling).
        GREEN after fix: 100 > 50 → rejected → 3.
        """
        result = self._call("100")
        self.assertEqual(
            result,
            3,
            f"_reload_ledger_max returned {result} for env=100 "
            f"— expected rejection to 3 (old ceiling, no longer valid)",
        )

    # ── Regression guards (must pass at base AND after fix) ──────────────────

    def test_default_when_unset(self):
        """Env absent → default 3."""
        self.assertEqual(self._call(), 3)

    def test_valid_value_within_bounds(self):
        """A value well within [1, 50] is returned as-is."""
        self.assertEqual(self._call("10"), 10)

    def test_zero_rejected_to_default(self):
        """COZEMPIC_RELOAD_MAX=0 is rejected (not positive) → default 3.

        parse_env_positive_int rejects 0 (not positive) → None → default 3.
        This is rejection, not a clamp: max(1, v) was dead code, now removed.
        The old test (test_floor_clamp_at_1) had a misleading docstring
        asserting a "genuine clamp" that never happened.
        """
        result = self._call("0")
        self.assertEqual(result, 3, "0 must be rejected to default 3, not clamped to 1")

    def test_ceiling_value_exact(self):
        """50 exactly is at the new boundary — must be accepted (not rejected)."""
        result = self._call("50")
        self.assertEqual(result, 50)

    def test_invalid_string_returns_fallback(self):
        """Non-numeric env var falls back to 3 (pre-existing behaviour)."""
        self.assertEqual(self._call("not-a-number"), 3)


class TestReloadWarnGraceBounds(unittest.TestCase):
    """_reload_warn_grace() rejects huge finite values → safe default 120.0.

    Cap = 3600 (1h): a grace period above 1h is functionally infinite for any
    interactive session. Values above 3600 must be rejected to default 120.0;
    the <=0 disable-semantic must be preserved.
    """

    def _call(self, env_val=None):
        from cozempic.guard import _reload_warn_grace

        env = {}
        if env_val is not None:
            env["COZEMPIC_RELOAD_WARN_GRACE"] = env_val
        with patch.dict(os.environ, env, clear=False):
            if env_val is None:
                os.environ.pop("COZEMPIC_RELOAD_WARN_GRACE", None)
            return _reload_warn_grace()

    # ── RED-at-base: huge finite values must be rejected to default ──────────

    def test_huge_grace_rejected_to_default(self):
        """99999999999s (~3170 years) makes elapsed >= grace permanently False.

        RED at base: math.isfinite(99999999999.0) is True → returns 99999999999.0,
        silently disabling the idle-reload fallback forever.
        GREEN after fix: v > 3600 → returns 120.0.
        """
        result = self._call("99999999999")
        self.assertEqual(
            result,
            120.0,
            f"_reload_warn_grace returned {result} for '99999999999' "
            f"— expected rejection to safe default 120.0 (large-finite gate-disable class)",
        )

    def test_above_ceiling_rejected(self):
        """3601 is one second above the 3600 (1h) ceiling — must be rejected.

        RED at base: isfinite(3601.0) True → 3601.0 returned.
        GREEN after fix: > 3600 → 120.0.
        """
        result = self._call("3601")
        self.assertEqual(
            result,
            120.0,
            f"_reload_warn_grace returned {result} for '3601' "
            f"— expected rejection to 120.0",
        )

    # ── Regression guards (must pass at base AND after fix) ──────────────────

    def test_disable_semantic_preserved_negative(self):
        """<= 0 DISABLES the grace wait (per docstring); negative still works."""
        result = self._call("-1")
        self.assertLessEqual(result, 0)

    def test_disable_semantic_zero(self):
        """Zero disables the grace wait."""
        result = self._call("0")
        self.assertLessEqual(result, 0)

    def test_default_when_unset(self):
        """Env absent → default 120.0 seconds."""
        self.assertEqual(self._call(), 120.0)

    def test_valid_value_passthrough(self):
        """300s is well within the ceiling — returned as-is."""
        self.assertAlmostEqual(self._call("300"), 300.0)

    def test_nan_rejected(self):
        """NaN already handled by isfinite; ensure it survives the refactor."""
        self.assertEqual(self._call("nan"), 120.0)

    def test_inf_rejected(self):
        """Inf already handled by isfinite; ensure it survives the refactor."""
        self.assertEqual(self._call("inf"), 120.0)

    def test_ceiling_value_exact(self):
        """3600 exactly is at the boundary — must be accepted (not rejected)."""
        self.assertAlmostEqual(self._call("3600"), 3600.0)
