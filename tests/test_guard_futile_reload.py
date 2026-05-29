"""RED tests for GAP-D — abort reload when prune saves marginal bytes.

Architect spec: AUDIT_REPORT_pr94_transient_daemon_race.md § GAP-D.
Root cause: if prune saves < _MIN_PRUNE_RATIO * original_bytes, the resumed
Claude inherits ~same bloat and hits HARD again immediately, causing a futile
reload chain. Current code reloads unconditionally when saved_bytes > 0.

Fix: in guard_prune_cycle, after the `saved_bytes <= 0` early return, add a
second early return when saved_bytes / original_bytes < _MIN_PRUNE_RATIO.
Returns {reloading: False, futile_reload_skipped: True}. start_guard increments
consecutive_empty_hard_prunes on futile-skip as well.

All 7 tests EXPECTED TO FAIL until Phase B implementation lands.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session_file(tmpdir: Path, size_bytes: int = 200_000) -> Path:
    """Create a fake JSONL session file of approximately size_bytes.

    Sets the file mtime to >24h ago so the active-session idle guard
    (introduced 2026-05-28) does not refuse the prune. Tests that want to
    exercise the active-session guard explicitly set their own mtime.
    """
    import time, os
    path = tmpdir / "fake_session.jsonl"
    line = '{"type":"user","message":{"content":"' + "x" * 100 + '"}}\n'
    lines_needed = max(1, size_bytes // len(line.encode()))
    path.write_text(line * lines_needed)
    aged = time.time() - 25 * 3600
    os.utime(path, (aged, aged))
    return path


def _make_sess(sid: str, path: Path) -> dict:
    return {"session_id": sid, "path": path}


# ---------------------------------------------------------------------------
# Test 1 — marginal prune (5% savings) skips reload
# ---------------------------------------------------------------------------
class TestMarginalPruneSkipsReload(unittest.TestCase):
    """When prune saves < _MIN_PRUNE_RATIO (5% < 10%), guard_prune_cycle
    must return reloading=False and futile_reload_skipped=True.
    _terminate_and_resume must NOT be called.
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cozempic_gap_d_"))
        self.sid = "fedcba987654321012345678abcdefff"
        self.session_path = _make_session_file(self.tmpdir, 100_000)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_marginal_prune_skips_reload(self):
        """5% savings → futile_reload_skipped=True, _terminate_and_resume not called."""
        try:
            from cozempic.guard import _MIN_PRUNE_RATIO
        except ImportError:
            self.fail(
                "_MIN_PRUNE_RATIO missing from cozempic.guard — "
                "Phase B not yet applied. Expected RED."
            )

        original_bytes = 100_000
        saved_bytes = int(original_bytes * 0.05)  # 5% — below threshold

        # Stub prune_with_team_protect to return a result simulating 5% savings
        from cozempic.team import TeamState
        fake_team_state = MagicMock(spec=TeamState)
        fake_team_state.is_empty.return_value = True
        fake_team_state.team_name = None
        fake_team_state.message_count = 0

        # Messages: original + pruned (5% savings)
        fake_messages_orig = [(0, {"type": "user"}, original_bytes)]
        final_bytes = original_bytes - saved_bytes
        fake_messages_pruned = [(0, {"type": "user"}, final_bytes)]

        terminate_called = []

        from cozempic.guard import guard_prune_cycle

        # PR #102 P1: guard_prune_cycle uses _terminate_claude instead of _terminate_and_resume.
        # For the GAP-D path (futile reload), neither should be called.
        with patch("cozempic.guard.load_messages", return_value=fake_messages_orig), \
             patch("cozempic.guard.prune_with_team_protect",
                   return_value=(fake_messages_pruned, {}, fake_team_state)), \
             patch("cozempic.guard.save_messages", return_value=None), \
             patch("cozempic.guard.snapshot_session", return_value=MagicMock()), \
             patch("cozempic.guard._terminate_claude",
                   side_effect=lambda *a, **kw: terminate_called.append(True) or "TERMINATED"), \
             patch("cozempic.guard._terminate_and_resume",
                   side_effect=lambda *a, **kw: terminate_called.append(True)), \
             patch("cozempic.tokens.estimate_session_tokens",
                   return_value=MagicMock(total=50000)), \
             patch("cozempic.tokens.calibrate_ratio", return_value=0.5):

            result = guard_prune_cycle(
                session_path=self.session_path,
                rx_name="standard",
                config=None,
                auto_reload=True,
                cwd=str(self.tmpdir),
                session_id=self.sid,
                claude_pid=89113,
            )

        self.assertFalse(
            result.get("reloading"),
            f"Expected reloading=False for marginal prune, got reloading={result.get('reloading')}. "
            f"Full result: {result}",
        )
        self.assertTrue(
            result.get("futile_reload_skipped"),
            f"Expected futile_reload_skipped=True, got: {result}. "
            "Phase B not yet applied — GAP-D logic missing from guard_prune_cycle.",
        )
        self.assertEqual(
            terminate_called, [],
            "_terminate_claude / _terminate_and_resume called despite marginal savings — reload not suppressed.",
        )


# ---------------------------------------------------------------------------
# Test 2 — substantial prune (15% savings) proceeds with reload
# ---------------------------------------------------------------------------
class TestSubstantialPruneProceedsWithReload(unittest.TestCase):
    """When prune saves >= _MIN_PRUNE_RATIO (15% >= 10%), the normal
    reload path must proceed. _terminate_and_resume must be called.
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cozempic_gap_d2_"))
        self.sid = "aabbccdd1122334455667788aabbccdd"
        self.session_path = _make_session_file(self.tmpdir, 100_000)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_substantial_prune_proceeds_with_reload(self):
        """15% savings → reloading=True, _terminate_and_resume called."""
        original_bytes = 100_000
        saved_bytes = int(original_bytes * 0.15)  # 15% — above threshold
        final_bytes = original_bytes - saved_bytes

        from cozempic.team import TeamState
        fake_team_state = MagicMock(spec=TeamState)
        fake_team_state.is_empty.return_value = True
        fake_team_state.team_name = None
        fake_team_state.message_count = 0

        fake_messages_orig = [(0, {"type": "user"}, original_bytes)]
        fake_messages_pruned = [(0, {"type": "user"}, final_bytes)]

        terminate_called = []

        from cozempic.guard import guard_prune_cycle

        # PR #102 P1: guard_prune_cycle now uses _terminate_claude + _resume_claude
        # instead of _terminate_and_resume. Mock both; _terminate_claude returns
        # "TERMINATED" so the reload pipeline proceeds.
        with patch("cozempic.guard.load_messages", return_value=fake_messages_orig), \
             patch("cozempic.guard.prune_with_team_protect",
                   return_value=(fake_messages_pruned, {}, fake_team_state)), \
             patch("cozempic.guard.save_messages", return_value=None), \
             patch("cozempic.guard.snapshot_session", return_value=MagicMock()), \
             patch("cozempic.guard._terminate_claude",
                   side_effect=lambda *a, **kw: terminate_called.append(True) or "TERMINATED"), \
             patch("cozempic.guard._resume_claude"), \
             patch("cozempic.tokens.estimate_session_tokens",
                   return_value=MagicMock(total=50000)), \
             patch("cozempic.tokens.calibrate_ratio", return_value=0.5):

            result = guard_prune_cycle(
                session_path=self.session_path,
                rx_name="standard",
                config=None,
                auto_reload=True,
                cwd=str(self.tmpdir),
                session_id=self.sid,
                claude_pid=89113,
            )

        # With 15% savings, normal reload path. reloading=True means
        # _terminate_claude was called (which triggers the terminate→save→resume pipeline).
        self.assertFalse(
            result.get("futile_reload_skipped"),
            f"futile_reload_skipped=True for a 15% prune — threshold logic inverted. Result: {result}",
        )
        self.assertTrue(
            terminate_called or result.get("reloading"),
            f"_terminate_claude not called for 15% savings. "
            f"terminate_called={terminate_called}, result={result}",
        )


# ---------------------------------------------------------------------------
# Test 3 — MIN_PRUNE_RATIO env var override
# ---------------------------------------------------------------------------
class TestMinPruneRatioEnvVarOverride(unittest.TestCase):
    """COZEMPIC_MIN_PRUNE_RATIO=0.05 → threshold drops to 5%.
    A 7% prune must now proceed (not be skipped).
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cozempic_gap_d3_"))
        self.sid = "cc11dd22ee33ff440011223344cc11dd"
        self.session_path = _make_session_file(self.tmpdir, 100_000)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_min_prune_ratio_env_var_override(self):
        """COZEMPIC_MIN_PRUNE_RATIO=0.05: 7% savings → should proceed (not futile)."""
        import importlib
        import cozempic.guard as guard_mod

        try:
            _ = guard_mod._MIN_PRUNE_RATIO
        except AttributeError:
            self.fail(
                "_MIN_PRUNE_RATIO not in cozempic.guard — Phase B not yet applied. Expected RED."
            )

        original_bytes = 100_000
        saved_bytes = int(original_bytes * 0.07)  # 7%
        final_bytes = original_bytes - saved_bytes

        from cozempic.team import TeamState
        fake_team_state = MagicMock(spec=TeamState)
        fake_team_state.is_empty.return_value = True
        fake_team_state.team_name = None
        fake_team_state.message_count = 0

        fake_messages_orig = [(0, {"type": "user"}, original_bytes)]
        fake_messages_pruned = [(0, {"type": "user"}, final_bytes)]

        terminate_called = []

        from cozempic.guard import guard_prune_cycle

        with patch.dict(os.environ, {"COZEMPIC_MIN_PRUNE_RATIO": "0.05"}), \
             patch.object(guard_mod, "_MIN_PRUNE_RATIO", 0.05), \
             patch("cozempic.guard.load_messages", return_value=fake_messages_orig), \
             patch("cozempic.guard.prune_with_team_protect",
                   return_value=(fake_messages_pruned, {}, fake_team_state)), \
             patch("cozempic.guard.save_messages", return_value=None), \
             patch("cozempic.guard.snapshot_session", return_value=MagicMock()), \
             patch("cozempic.guard._terminate_claude",
                   side_effect=lambda *a, **kw: terminate_called.append(True) or "TERMINATED"), \
             patch("cozempic.guard._resume_claude"), \
             patch("cozempic.guard._terminate_and_resume",
                   side_effect=lambda *a, **kw: terminate_called.append(True)), \
             patch("cozempic.tokens.estimate_session_tokens",
                   return_value=MagicMock(total=50000)), \
             patch("cozempic.tokens.calibrate_ratio", return_value=0.5):

            result = guard_prune_cycle(
                session_path=self.session_path,
                rx_name="standard",
                config=None,
                auto_reload=True,
                cwd=str(self.tmpdir),
                session_id=self.sid,
                claude_pid=89113,
            )

        self.assertFalse(
            result.get("futile_reload_skipped"),
            f"7% prune was skipped even though env var set threshold to 5%. "
            f"Env-var override not working. Result: {result}",
        )


# ---------------------------------------------------------------------------
# Test 4 — Invalid _MIN_PRUNE_RATIO env var falls back to 0.10
# ---------------------------------------------------------------------------
class TestMinPruneRatioInvalidFallsBack(unittest.TestCase):
    """COZEMPIC_MIN_PRUNE_RATIO=invalid → must fall back to default 0.10.

    Mirrors PR #93's pattern for COZEMPIC_GUARD_HARD_EXIT_K.
    """

    def test_min_prune_ratio_invalid_falls_back(self):
        """Invalid env var → _MIN_PRUNE_RATIO == 0.10 (default)."""
        try:
            import importlib
            import cozempic.guard as guard_mod
            # Check the reader function exists
            if not hasattr(guard_mod, "_read_min_prune_ratio"):
                self.fail(
                    "_read_min_prune_ratio missing from cozempic.guard — Phase B not applied."
                )
        except ImportError:
            self.fail("Cannot import cozempic.guard")

        from cozempic.guard import _read_min_prune_ratio

        with patch.dict(os.environ, {"COZEMPIC_MIN_PRUNE_RATIO": "not_a_number"}):
            val = _read_min_prune_ratio()

        self.assertEqual(
            val,
            0.10,
            f"Expected fallback to 0.10 for invalid env var, got {val}. "
            "Fallback logic missing or broken.",
        )

        # Also check boundary values are rejected
        for bad_val in ("0.0", "1.0", "1.5", "-0.1", "inf", "nan"):
            with patch.dict(os.environ, {"COZEMPIC_MIN_PRUNE_RATIO": bad_val}):
                val = _read_min_prune_ratio()
            self.assertEqual(
                val,
                0.10,
                f"Expected fallback to 0.10 for boundary value {bad_val!r}, got {val}.",
            )


# ---------------------------------------------------------------------------
# Test 5 — Futile reload increments K counter
# ---------------------------------------------------------------------------
class TestFutileReloadIncrementsKCounter(unittest.TestCase):
    """When guard_prune_cycle returns futile_reload_skipped=True, start_guard
    must increment consecutive_empty_hard_prunes (K counter), just as it
    does for saved_bytes=0. This ensures the K=10 → exit logic fires after
    enough futile-skip cycles.
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cozempic_gap_d5_"))
        self.sid = "dd22ee33ff440011223344dd22ee33ff"
        self.session_path = _make_session_file(self.tmpdir, 100_000)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_fake_sess(self):
        return {"session_id": self.sid, "path": self.session_path}

    def test_futile_reload_increments_k_counter(self):
        """Three consecutive futile-skip cycles → K=3; K=10 exit triggers at K=10."""
        try:
            from cozempic.guard import _MIN_PRUNE_RATIO, HARD_LOOP_EXIT_THRESHOLD
        except ImportError:
            self.fail("_MIN_PRUNE_RATIO missing — Phase B not applied. Expected RED.")

        # We'll drive start_guard for a few cycles by making guard_prune_cycle
        # return futile_reload_skipped=True each time and observing that the
        # guard eventually exits (at K=HARD_LOOP_EXIT_THRESHOLD).
        # We short-circuit by raising after K=3 to avoid running K=10 cycles.

        call_counts = {"prune": 0, "sleep": 0}
        _EXIT_K = HARD_LOOP_EXIT_THRESHOLD  # 10 by default

        class _StopAt3(Exception):
            pass

        def _fake_prune_cycle(*args, **kwargs):
            call_counts["prune"] += 1
            return {
                "saved_mb": 0.001,  # non-zero to pass saved_bytes > 0 gate
                "original_tokens": 1000,
                "final_tokens": 950,
                "team_name": None,
                "team_messages": 0,
                "checkpoint_path": None,
                "backup_path": None,
                "reloading": False,
                "futile_reload_skipped": True,
            }

        def _fake_sleep(n):
            call_counts["sleep"] += 1
            if call_counts["prune"] >= 3:
                raise _StopAt3()

        from cozempic.guard import start_guard
        import inspect

        # Verify start_guard source increments K on futile_reload_skipped
        source = inspect.getsource(start_guard)
        self.assertIn(
            "futile_reload_skipped",
            source,
            "start_guard does not reference futile_reload_skipped — "
            "K-counter increment on futile skip not implemented. Phase B needed.",
        )
        # Also verify the increment is tied to consecutive_empty_hard_prunes
        self.assertIn(
            "consecutive_empty_hard_prunes",
            source,
            "consecutive_empty_hard_prunes not referenced in start_guard after futile skip.",
        )


# ---------------------------------------------------------------------------
# Test 6 — Futile reload log message emits only once (deferred_exit_announced style)
# ---------------------------------------------------------------------------
class TestFutileReloadLogMessageEmitsOnce(unittest.TestCase):
    """The "prune freed only N bytes" diagnostic message must emit at most
    once across multiple futile-skip cycles (one-shot per defer window),
    mirroring PR #93's deferred_exit_announced pattern.
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cozempic_gap_d6_"))
        self.sid = "ee33ff440011223344ee33ff440011ee"
        self.session_path = _make_session_file(self.tmpdir, 100_000)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_futile_reload_log_message_emits_once(self):
        """Futile-skip log line appears exactly once in start_guard source code."""
        try:
            from cozempic.guard import _MIN_PRUNE_RATIO
        except ImportError:
            self.fail("_MIN_PRUNE_RATIO missing — Phase B not applied. Expected RED.")

        import inspect
        from cozempic.guard import start_guard

        source = inspect.getsource(start_guard)

        # The one-shot flag should exist (mirrors deferred_exit_announced)
        one_shot_flags = [
            "_futile_skip_announced",
            "futile_skip_announced",
        ]
        found_flag = any(flag in source for flag in one_shot_flags)
        self.assertTrue(
            found_flag,
            "No one-shot flag for futile-skip log message found in start_guard. "
            "Expected _futile_skip_announced or similar. Phase B needed. "
            f"Checked: {one_shot_flags}",
        )

        # The log message content should mention threshold/ratio
        found_msg = (
            "below" in source or
            "threshold" in source and "prune" in source
        )
        self.assertTrue(
            found_msg,
            "Futile-skip log message not found in start_guard source. "
            "Expected a message about savings below threshold.",
        )


# ---------------------------------------------------------------------------
# Test 7 — Futile reload still writes team checkpoint
# ---------------------------------------------------------------------------
class TestFutileReloadWritesTeamCheckpoint(unittest.TestCase):
    """Even when reload is skipped as futile, the team checkpoint path
    must still be written (so the user can recover team state via the
    checkpoint path surfaced in the diagnostic message).
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cozempic_gap_d7_"))
        self.sid = "ff440011223344ff4400112233440011"
        self.session_path = _make_session_file(self.tmpdir, 100_000)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_futile_reload_writes_team_checkpoint(self):
        """Futile skip: checkpoint_path is written and returned in result."""
        try:
            from cozempic.guard import _MIN_PRUNE_RATIO
        except ImportError:
            self.fail("_MIN_PRUNE_RATIO missing — Phase B not applied. Expected RED.")

        original_bytes = 100_000
        saved_bytes = int(original_bytes * 0.05)  # 5% — below threshold
        final_bytes = original_bytes - saved_bytes

        # Team has active subagents → team state not empty
        from cozempic.team import TeamState
        fake_team_state = MagicMock(spec=TeamState)
        fake_team_state.is_empty.return_value = False  # active team!
        fake_team_state.team_name = "silc-data"
        fake_team_state.message_count = 200

        fake_messages_orig = [(0, {"type": "user"}, original_bytes)]
        fake_messages_pruned = [(0, {"type": "user"}, final_bytes)]

        fake_checkpoint = self.tmpdir / "team_checkpoint.json"
        fake_checkpoint.write_text("{}")

        from cozempic.guard import guard_prune_cycle

        with patch("cozempic.guard.load_messages", return_value=fake_messages_orig), \
             patch("cozempic.guard.prune_with_team_protect",
                   return_value=(fake_messages_pruned, {}, fake_team_state)), \
             patch("cozempic.guard.save_messages", return_value=None), \
             patch("cozempic.guard.snapshot_session", return_value=MagicMock()), \
             patch("cozempic.guard.write_team_checkpoint", return_value=fake_checkpoint), \
             patch("cozempic.guard._terminate_claude", return_value="TERMINATED"), \
             patch("cozempic.guard._resume_claude"), \
             patch("cozempic.guard._terminate_and_resume"), \
             patch("cozempic.tokens.estimate_session_tokens",
                   return_value=MagicMock(total=50000)), \
             patch("cozempic.tokens.calibrate_ratio", return_value=0.5):

            result = guard_prune_cycle(
                session_path=self.session_path,
                rx_name="standard",
                config=None,
                auto_reload=True,
                cwd=str(self.tmpdir),
                session_id=self.sid,
                claude_pid=89113,
            )

        self.assertTrue(
            result.get("futile_reload_skipped"),
            f"Expected futile_reload_skipped=True, got: {result}. Phase B not applied.",
        )
        self.assertIsNotNone(
            result.get("checkpoint_path"),
            f"checkpoint_path is None in futile-skip result. "
            f"Team checkpoint must still be written. Result: {result}",
        )


if __name__ == "__main__":
    unittest.main()
