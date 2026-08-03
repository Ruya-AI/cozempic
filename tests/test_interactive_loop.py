"""1.8.22 — loop-level integration for component E.

Drives start_guard for two controlled cycles and captures the auto_reload arg it
hands guard_prune_cycle each cycle. This is the load-bearing wiring proof:
  • interactive + mid-turn (transcript growing)  → defer (auto_reload=False)
  • interactive + idle (transcript stable)        → reload (auto_reload=True)
  • headless                                       → reload immediately (unchanged)
  • interactive + past the force line (~88%)       → reload even mid-turn

Cycle control: prev_size starts -1, so cycle 1 is never "idle"; the session file
size is held constant, so cycle 2 IS idle. time.sleep raises KeyboardInterrupt on
the 3rd call to exit the loop cleanly (start_guard catches it).
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


class _StopLoop(KeyboardInterrupt):
    pass


def _run_guard(token_estimate: int, env: dict, context_window: int = 200_000):
    """Run start_guard for 2 cycles; return the list of auto_reload values
    passed to guard_prune_cycle."""
    from cozempic import guard
    from cozempic.team import TeamState

    tmp = Path(tempfile.mkdtemp(prefix="cozempic_loop_"))
    sess_path = tmp / "s.jsonl"
    sess_path.write_text("x" * 5000)  # constant size across cycles
    sess = {"session_id": "loopsess0001", "path": sess_path}

    calls = []
    sleeps = {"n": 0}

    def _sleep(_):
        sleeps["n"] += 1
        if sleeps["n"] >= 4:  # let 3 full cycles run (sustained-idle needs 2 idle cycles)
            raise _StopLoop()

    def _cycle(*a, **k):
        calls.append(k.get("auto_reload"))
        # read-only-shaped result: no "reloading" key → loop proceeds to next
        # cycle, and "live_write_skipped" keeps _fmt_prune_result off the path.
        return {"saved_mb": 0.0, "live_write_skipped": True,
                "original_tokens": 165_000, "final_tokens": 110_000}

    import os
    e = {kk: vv for kk, vv in os.environ.items() if not kk.startswith("COZEMPIC")}
    # Disable the warned-before-reload grace wait so these tests exercise the
    # idle-reload wiring directly (the warned-gate has its own dedicated tests).
    e.setdefault("COZEMPIC_RELOAD_WARN_GRACE", "0")
    e.update(env)

    with contextlib.ExitStack() as s:
        p = lambda *a, **k: s.enter_context(patch(*a, **k))
        p("cozempic.guard.find_current_session", return_value=sess)
        p("cozempic.guard._resolve_session_by_id", return_value=sess)
        p("cozempic.tokens.detect_context_window", return_value=context_window)
        p("cozempic.guard.load_messages", return_value=[])
        p("cozempic.session.record_session", return_value=None)
        p("cozempic.guard._cleanup_stale_watchers", return_value=None)
        p("cozempic.guard.ping_install_if_new", return_value=None)
        p("cozempic.guard.maybe_auto_update", return_value=None)
        p("cozempic.guard.signal.signal", return_value=None)
        p("cozempic.guard.find_claude_pid", return_value=4242)
        p("cozempic.guard._record_claude_identity", return_value=None)
        p("cozempic.guard.os.kill", return_value=None)
        # Windows watchdog probes liveness via guard._pid_is_alive (native
        # OpenProcess — os.kill(pid, 0) is CTRL_C_EVENT delivery there, not a
        # probe), so the fake claude_pid=4242 must read alive through that
        # seam too. Inert on POSIX, where the loop keeps the raw os.kill probe.
        p("cozempic.guard._pid_is_alive", return_value=True)
        p("cozempic.guard._pid_identity_match", return_value=True)
        p("cozempic.guard._is_claude_process", return_value=True)
        p("cozempic.guard.checkpoint_team", return_value=TeamState())
        p("cozempic.guard.quick_token_estimate", return_value=token_estimate)
        p("cozempic.guard.cleanup_old_backups", return_value=None)
        p("cozempic.guard._safe_unlink_session_pidfile", return_value=None)
        p("cozempic.guard._guard_tmp_root", return_value=tmp)
        p("cozempic.guard.guard_prune_cycle", side_effect=_cycle)
        p("cozempic.guard.time.sleep", side_effect=_sleep)
        with patch.dict("os.environ", e, clear=True):
            try:
                guard.start_guard(
                    cwd=str(tmp), interval=1, reactive=False,
                    session_id="loopsess0001", claude_pid=4242,
                )
            except _StopLoop:
                pass
    shutil.rmtree(tmp, ignore_errors=True)
    return calls


class TestInteractiveLoopWiring(unittest.TestCase):
    # 165K/200K = 82.5% → over HARD2 (80%), under force (88%)
    OVER_HARD2 = 165_000
    OVER_FORCE = 185_000  # 92.5% → past force line

    def test_interactive_defers_then_reloads_at_sustained_idle(self):
        # cycle1 not-idle (prev=-1) → defer; cycle2 idle but idle_cycles=1<2 →
        # still defer (one stable cycle could be a mid-turn stall); cycle3
        # idle_cycles=2 → sustained idle → reload.
        calls = _run_guard(self.OVER_HARD2, {"COZEMPIC_INTERACTIVE": "on"})
        self.assertEqual(calls, [False, False, True],
                         "interactive reloads only after SUSTAINED idle (2 cycles)")

    def test_headless_reloads_immediately(self):
        calls = _run_guard(self.OVER_HARD2, {"COZEMPIC_INTERACTIVE": "off"})
        self.assertEqual(calls, [True, True, True],
                         "headless: reload every cycle (today's behavior, unchanged)")

    def test_force_line_reloads_even_mid_turn(self):
        calls = _run_guard(self.OVER_FORCE, {"COZEMPIC_INTERACTIVE": "on"})
        self.assertEqual(calls, [True, True, True],
                         "past ~88%: reload even mid-turn (beats the autocompact wall)")

    def test_force_disabled_keeps_deferring_until_sustained_idle(self):
        calls = _run_guard(self.OVER_FORCE, {"COZEMPIC_INTERACTIVE": "on",
                                             "COZEMPIC_FORCE_RELOAD_PCT": "0"})
        self.assertEqual(calls, [False, False, True],
                         "force disabled: defer mid-turn, reload only at sustained idle")

    def test_warned_gate_holds_idle_reload_until_warned(self):
        # grace high + never warned (no nudge fires in this harness) → even at
        # sustained idle the reload is HELD (armed, waiting for the warning), so
        # the user is never reloaded without first being warned.
        calls = _run_guard(self.OVER_HARD2, {"COZEMPIC_INTERACTIVE": "on",
                                             "COZEMPIC_RELOAD_WARN_GRACE": "9999"})
        self.assertEqual(calls, [False, False, False],
                         "interactive idle reload must wait for the warning")


if __name__ == "__main__":
    unittest.main()
