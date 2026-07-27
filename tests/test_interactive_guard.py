"""1.8.22 — interactive guard daemon helpers (E/F/H) + nudge arming.

Covers component H (_detect_interactive), F (_idle_backoff_cycles), E
(_force_reload_pct + _arm_nudge_from_result). The loop-level gating
(defer-mid-turn vs reload-at-idle) is exercised end-to-end by the safe-point
suite (guard_prune_cycle with auto_reload toggled) plus these helpers, which are
the only new branch inputs the loop reads.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestDetectInteractive(unittest.TestCase):
    def test_env_on_forces_true(self):
        from cozempic.guard import _detect_interactive
        with patch.dict("os.environ", {"COZEMPIC_INTERACTIVE": "on"}):
            self.assertTrue(_detect_interactive(None))
            self.assertTrue(_detect_interactive(12345))

    def test_env_off_forces_false(self):
        from cozempic.guard import _detect_interactive
        with patch.dict("os.environ", {"COZEMPIC_INTERACTIVE": "off"}):
            self.assertFalse(_detect_interactive(None))
            self.assertFalse(_detect_interactive(12345))

    def test_auto_no_pid_defaults_interactive(self):
        from cozempic.guard import _detect_interactive
        with patch.dict("os.environ", {"COZEMPIC_INTERACTIVE": "auto"}):
            self.assertTrue(_detect_interactive(None))

    def test_auto_with_tty_is_interactive(self):
        from cozempic import guard
        with patch.dict("os.environ", {"COZEMPIC_INTERACTIVE": "auto"}), \
             patch("cozempic.guard.subprocess.run",
                   return_value=MagicMock(stdout="ttys001\n")):
            self.assertTrue(guard._detect_interactive(4242))

    def test_auto_no_tty_is_headless(self):
        from cozempic import guard
        for ttyval in ("??", "?", "-", ""):
            with patch.dict("os.environ", {"COZEMPIC_INTERACTIVE": "auto"}), \
                 patch("cozempic.guard.subprocess.run",
                       return_value=MagicMock(stdout=ttyval + "\n")):
                self.assertFalse(guard._detect_interactive(4242), ttyval)

    def test_auto_ps_failure_defaults_interactive(self):
        from cozempic import guard
        with patch.dict("os.environ", {"COZEMPIC_INTERACTIVE": "auto"}), \
             patch("cozempic.guard.subprocess.run", side_effect=OSError):
            self.assertTrue(guard._detect_interactive(4242))


class TestIdleBackoffCycles(unittest.TestCase):
    def test_default_is_four(self):
        from cozempic.guard import _idle_backoff_cycles
        with patch.dict("os.environ", {}, clear=False) as _:
            import os
            os.environ.pop("COZEMPIC_IDLE_BACKOFF_CYCLES", None)
            self.assertEqual(_idle_backoff_cycles(), 4)

    def test_env_override(self):
        from cozempic.guard import _idle_backoff_cycles
        with patch.dict("os.environ", {"COZEMPIC_IDLE_BACKOFF_CYCLES": "10"}):
            self.assertEqual(_idle_backoff_cycles(), 10)

    def test_zero_disables(self):
        from cozempic.guard import _idle_backoff_cycles
        with patch.dict("os.environ", {"COZEMPIC_IDLE_BACKOFF_CYCLES": "0"}):
            self.assertEqual(_idle_backoff_cycles(), 0)

    def test_garbage_falls_back(self):
        from cozempic.guard import _idle_backoff_cycles
        with patch.dict("os.environ", {"COZEMPIC_IDLE_BACKOFF_CYCLES": "nope"}):
            self.assertEqual(_idle_backoff_cycles(), 4)


class TestForceReloadPct(unittest.TestCase):
    def test_default(self):
        from cozempic.guard import _force_reload_pct
        import os
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("COZEMPIC_FORCE_RELOAD_PCT", None)
            self.assertAlmostEqual(_force_reload_pct(), 0.88)

    def test_env_override(self):
        from cozempic.guard import _force_reload_pct
        with patch.dict("os.environ", {"COZEMPIC_FORCE_RELOAD_PCT": "0.95"}):
            self.assertAlmostEqual(_force_reload_pct(), 0.95)

    def test_out_of_range_disables(self):
        from cozempic.guard import _force_reload_pct
        for bad in ("0", "-1", "1.5", "2"):
            with patch.dict("os.environ", {"COZEMPIC_FORCE_RELOAD_PCT": bad}):
                self.assertEqual(_force_reload_pct(), 0.0, bad)

    def test_garbage_falls_back(self):
        from cozempic.guard import _force_reload_pct
        with patch.dict("os.environ", {"COZEMPIC_FORCE_RELOAD_PCT": "x"}):
            self.assertAlmostEqual(_force_reload_pct(), 0.88)


class TestArmNudgeFromResult(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(tempfile.mkdtemp(prefix="cozempic_arm_"))

    def tearDown(self):
        shutil.rmtree(self.scratch, ignore_errors=True)

    def test_arms_with_real_projected_pct(self):
        from cozempic import guard
        with patch("cozempic.guard._guard_tmp_root", return_value=self.scratch):
            guard._arm_nudge_from_result(
                "sess-arm-1", None, 55,
                {"original_tokens": 100_000, "final_tokens": 60_000},
            )
            armed = guard.read_armed("sess-arm-1", None)
        self.assertIsNotNone(armed)
        self.assertEqual(armed["tier"], 55)
        self.assertAlmostEqual(armed["projected_pct"], 40.0, places=1)

    def test_zero_reduction_arms_with_zero(self):
        from cozempic import guard
        with patch("cozempic.guard._guard_tmp_root", return_value=self.scratch):
            guard._arm_nudge_from_result(
                "sess-arm-2", None, 80,
                {"original_tokens": 100_000, "final_tokens": 100_000},
            )
            armed = guard.read_armed("sess-arm-2", None)
        self.assertEqual(armed["projected_pct"], 0.0)

    def test_missing_keys_does_not_raise(self):
        from cozempic import guard
        with patch("cozempic.guard._guard_tmp_root", return_value=self.scratch):
            guard._arm_nudge_from_result("sess-arm-3", None, 55, {})  # must not raise
            armed = guard.read_armed("sess-arm-3", None)
        self.assertEqual(armed["projected_pct"], 0.0)

    def test_arm_prefers_projected_final_tokens(self):
        # The real projected reduction comes from the read-only project=True path.
        from cozempic import guard
        with patch("cozempic.guard._guard_tmp_root", return_value=self.scratch):
            guard._arm_nudge_from_result(
                "sess-arm-4", None, 55,
                {"original_tokens": 100_000, "final_tokens": 100_000,  # read-only: equal
                 "projected_final_tokens": 60_000},                    # the real estimate
            )
            armed = guard.read_armed("sess-arm-4", None)
        self.assertAlmostEqual(armed["projected_pct"], 40.0, places=1)

    def test_write_armed_sets_and_preserves_armed_at(self):
        from cozempic import guard
        with patch("cozempic.guard._guard_tmp_root", return_value=self.scratch):
            guard.write_armed("sess-at", None, 55, 30.0)
            a1 = guard.read_armed("sess-at", None)
            self.assertIn("armed_at", a1)
            guard.mark_armed_warned("sess-at", None)
            guard.write_armed("sess-at", None, 55, None)  # re-arm same tier, no new projection
            a2 = guard.read_armed("sess-at", None)
        self.assertEqual(a1["armed_at"], a2["armed_at"], "grace clock preserved on re-arm")
        self.assertTrue(a2["warned"], "warned preserved on same-tier re-arm")
        self.assertEqual(a2["projected_pct"], 30.0, "projection preserved when re-arm has none")

    def test_write_armed_overwrites_projection_with_zero(self):
        from cozempic import guard

        with patch("cozempic.guard._guard_tmp_root", return_value=self.scratch):
            guard.write_armed("sess-zero", None, 55, 30.0)
            guard.write_armed("sess-zero", None, 55, 0.0)
            armed = guard.read_armed("sess-zero", None)

        self.assertEqual(armed["projected_pct"], 0.0)

    def test_read_armed_drops_nonnumeric_grace_fields(self):
        from cozempic import guard

        with patch("cozempic.guard._guard_tmp_root", return_value=self.scratch):
            path = guard._reload_armed_path("sess-malformed", None)
            path.write_text('{"armed_at": "later", "tier": "hard"}')
            armed = guard.read_armed("sess-malformed", None)

        self.assertEqual(armed, {})

    def test_mark_armed_warned_upserts(self):
        # The nudge can warn before the daemon arms — mark must CREATE the sentinel.
        from cozempic import guard
        with patch("cozempic.guard._guard_tmp_root", return_value=self.scratch):
            self.assertIsNone(guard.read_armed("sess-up", None))
            guard.mark_armed_warned("sess-up", None)
            armed = guard.read_armed("sess-up", None)
        self.assertIsNotNone(armed)
        self.assertTrue(armed["warned"])
        self.assertIn("armed_at", armed)

    def test_warned_is_sticky_across_rearm(self):
        # P0 regression: the nudge upserts warned (tier 0), then the daemon re-arms
        # at tier 80 — warned MUST stay True (escalating the tier or re-arming must
        # not un-warn the user, or the reload waits on the blind grace timer / wedges).
        from cozempic import guard
        with patch("cozempic.guard._guard_tmp_root", return_value=self.scratch):
            guard.mark_armed_warned("sticky1", None)         # nudge warns first (tier 0)
            self.assertTrue(guard.read_armed("sticky1", None)["warned"])
            guard.write_armed("sticky1", None, 80, 30.0)      # daemon re-arms at 80
            a = guard.read_armed("sticky1", None)
        self.assertTrue(a["warned"], "warned must survive a re-arm at a different tier")
        self.assertEqual(a["tier"], 80)

    def test_terminate_and_resume_clears_armed(self):
        # P1: every reload path funnels through _terminate_and_resume; it MUST clear
        # the sentinel so a sticky warned=True can't survive into the resumed
        # session (same session_id) and cause an unwarned reload there.
        from cozempic import guard
        called = []
        with patch("cozempic.guard.clear_armed", side_effect=lambda *a, **k: called.append(a)), \
             patch("cozempic.guard._detect_claude_flags", return_value=""), \
             patch("cozempic.guard._detect_terminal_env", return_value={}), \
             patch("cozempic.guard._is_claude_process", return_value=False), \
             patch("cozempic.guard.platform.system", return_value="Unknown"):
            try:
                guard._terminate_and_resume(99999, "/tmp", session_id="term1", session_path=None)
            except Exception:
                pass
        self.assertTrue(called, "_terminate_and_resume must clear the armed sentinel")
        self.assertEqual(called[0][0], "term1")

    def test_clear_armed_neutralizes_on_unlink_failure(self):
        # P1 defense-in-depth: if unlink fails, clear_armed must NEUTRALIZE the
        # sentinel (warned=False) so a survivor can't carry a stale warning.
        from cozempic import guard
        with patch("cozempic.guard._guard_tmp_root", return_value=self.scratch):
            guard.mark_armed_warned("cn1", None)
            self.assertTrue(guard.read_armed("cn1", None)["warned"])
            with patch("pathlib.Path.unlink", side_effect=OSError("locked")):
                guard.clear_armed("cn1", None)
            a = guard.read_armed("cn1", None)
        self.assertFalse((a or {}).get("warned"), "must neutralize warned on unlink failure")


if __name__ == "__main__":
    unittest.main()
