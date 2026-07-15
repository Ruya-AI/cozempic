"""Mechanical reproducers for the 2026-05-18 cozempic guard crash.

Source handoff: /Users/yanisnaamane/sanofi/silc-data/.claude/handoffs/cozempic-guard-crash-2026-05-18.md

Two bugs to reproduce on the current v1.8.11 / main-branch code:

R1 — Process-vs-process daemon race
    The SessionStart hook fires on both ``startup`` and ``resume`` events. When
    a HARD-threshold reload kicks in and the resume hook fires from two
    different angles within ms of each other, two completely separate Python
    processes call ``start_guard_daemon(session_id=<same uuid>)``. The
    in-process ``threading.Lock`` in ``_spawn_locks`` does NOT span processes;
    only the ``O_CREAT|O_EXCL`` race on the pidfile + the on-disk pid-read
    fallback can prevent a double spawn.

    This test launches TWO ``multiprocessing.Process`` instances synchronized
    at an ``os.Barrier(2)`` instant and asserts that EXACTLY ONE comes back
    with ``started=True`` and the other with ``already_running=True``. Loops
    50× to catch the race window.

R2 — HARD-threshold zero-byte loop
    When ``guard_prune_cycle`` repeatedly returns ``saved_mb == 0`` (because
    the live conversation is dominated by immutable tool-result blocks the
    soft prune cannot touch), the current loop:
      - prints a warning after 3 consecutive 0-byte hards,
      - sleeps ``interval * 4`` once,
      - resets the consecutive counter to 0,
      - keeps looping at the original 30s interval forever.

    A 20-cycle bounded run with mocked dependencies and a counted
    ``time.sleep`` proves the loop never exits, never backs off long-term,
    and never raises a diagnostic exception.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure src/ is importable when run with PYTHONPATH=src
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# R1: daemon-vs-daemon process-vs-process race
# ---------------------------------------------------------------------------


def _race_worker(
    barrier_handle,
    result_queue,
    session_id: str,
    cwd: str,
    worker_index: int,
) -> None:
    """Run in a separate process. Wait at the barrier so both processes call
    ``start_guard_daemon`` at essentially the same instant, then report the
    result back via the shared queue.

    Heavy-handed mocking is used so the child does NOT actually spawn a guard
    subprocess — we want to test the pidfile claim contention, not the full
    daemon lifecycle.
    """
    # Re-import in the child (multiprocessing spawn does not inherit imports
    # in a useful way on macOS).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    from unittest.mock import patch as _patch

    from cozempic.guard import start_guard_daemon

    class _DummyProc:
        # PID range avoids collision with the host process's real PIDs.
        def __init__(self, pid: int):
            self.pid = pid

    def _fake_popen(cmd_parts, **kwargs):
        # The PID we hand back must look unique per worker so the test can
        # tell which child "won" if both raced through. Use a value that is
        # extremely unlikely to be a real running PID.
        fake_pid = 900_000 + worker_index
        return _DummyProc(fake_pid)

    def _fake_is_process_alive(pid: int) -> bool:
        """Model the spawned fake guard as alive for the contention check."""
        return pid > 0

    # Mock subprocess.Popen so no real daemon is spawned, and mock
    # find_claude_pid so we don't fail when no Claude is running.
    with (
        _patch("cozempic.guard.subprocess.Popen", side_effect=_fake_popen),
        _patch("cozempic.guard.find_claude_pid", return_value=12345),
        _patch("cozempic.guard._cleanup_legacy_pid"),
        _patch("cozempic.guard._is_guard_running_for_session", return_value=None),
        _patch(
            "cozempic.spawn_lock._is_process_alive", side_effect=_fake_is_process_alive
        ),
    ):
        # Barrier sync: both children pile up here and release at the same instant
        try:
            barrier_handle.wait(timeout=5.0)
        except Exception as e:  # BrokenBarrierError or timeout
            result_queue.put(
                {"error": f"barrier failed: {e!r}", "worker": worker_index}
            )
            return

        try:
            r = start_guard_daemon(
                cwd=cwd,
                session_id=session_id,
                threshold_tokens=1000,
            )
        except Exception as e:
            r = {"error": repr(e)}

        r["worker"] = worker_index
        result_queue.put(r)


class TestR1_DaemonProcessRace(unittest.TestCase):
    """Process-vs-process race on start_guard_daemon for the same session UUID.

    The in-process ``_spawn_locks`` ``threading.Lock`` does NOT span Python
    processes — only the on-disk pidfile claim does. This test verifies the
    O_CREAT|O_EXCL claim path holds across process boundaries.
    """

    # Must match _SESSION_ID_RE = ^[0-9a-f][0-9a-f-]{11,}$  (hex chars + dashes only)
    SESSION_ID = "abcd1234-5678-9abc-def0-2026051811aa"
    ITERATIONS = 50

    def setUp(self):
        # Compute and pre-clean the pidfile + log file for our session.
        from cozempic.guard import _pid_file_for_session

        self.pid_path = _pid_file_for_session(self.SESSION_ID)
        self.pid_path.unlink(missing_ok=True)
        self.log_path = self.pid_path.with_suffix(".log")
        self.log_path.unlink(missing_ok=True)

    def tearDown(self):
        self.pid_path.unlink(missing_ok=True)
        self.log_path.unlink(missing_ok=True)

    def _race_once(self) -> list[dict]:
        """Spawn two processes, sync at a barrier, collect results."""
        # On macOS the default start method is spawn; force it explicitly so
        # the test is platform-deterministic.
        ctx = mp.get_context("spawn")

        barrier = ctx.Barrier(2)
        result_queue = ctx.Queue()

        # Fresh cwd per iteration so the legacy-pid cleanup path doesn't
        # interfere across iterations.
        cwd = os.getcwd()

        procs = [
            ctx.Process(
                target=_race_worker,
                args=(barrier, result_queue, self.SESSION_ID, cwd, i),
                name=f"race-child-{i}",
            )
            for i in range(2)
        ]
        for p in procs:
            p.start()

        # Collect both results
        results = []
        for _ in range(2):
            try:
                results.append(result_queue.get(timeout=10.0))
            except Exception as e:
                results.append({"error": f"queue.get failed: {e!r}"})

        for p in procs:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)

        return results

    def test_two_processes_one_winner(self):
        """Across ITERATIONS races, every race must produce exactly one
        ``started=True`` and exactly one ``already_running=True``.

        If either:
          - both report started=True (double spawn — orphan daemon), or
          - both report already_running=True (deadlock, no daemon spawned), or
          - any race throws an unexpected exception,
        the test fails with the full record so the team-lead can diagnose.
        """
        failures = []

        for it in range(self.ITERATIONS):
            # Clean state before each race
            self.pid_path.unlink(missing_ok=True)
            self.log_path.unlink(missing_ok=True)

            results = self._race_once()

            started = [r for r in results if r.get("started") is True]
            already = [r for r in results if r.get("already_running") is True]
            errors = [r for r in results if "error" in r]

            if errors or len(started) != 1 or len(already) != 1:
                failures.append({"iteration": it, "results": results})

        if failures:
            self.fail(
                f"Process-vs-process race produced bad outcomes in "
                f"{len(failures)}/{self.ITERATIONS} iterations.\n"
                f"First 3 failure records:\n" + "\n".join(repr(f) for f in failures[:3])
            )


# ---------------------------------------------------------------------------
# R2: HARD-threshold zero-byte loop
# ---------------------------------------------------------------------------


class _StopAfterNSleeps(Exception):
    """Sentinel raised by the patched time.sleep after N cycles to break the
    otherwise-infinite guard loop deterministically."""


class TestR2_HardThresholdZeroByteLoop(unittest.TestCase):
    """When guard_prune_cycle keeps returning ``saved_mb == 0`` at the HARD
    threshold, the loop must NOT silently continue at the original interval
    forever.

    Acceptable behavior (any one of):
      a) raise a backoff exception that takes the daemon down cleanly,
      b) progressively double the sleep interval (exponential backoff),
      c) exit the loop with a diagnostic message.

    Current v1.8.11 behavior (the bug): logs a warning after 3 consecutive
    0-byte hards, sleeps ``interval * 4`` ONCE, resets the counter to 0, then
    keeps looping at the original 30s interval. This test demonstrates the
    bug by running 20 cycles and showing the loop never exits and the average
    sleep stays near the original interval.
    """

    def test_loop_does_not_back_off_under_sustained_zero_byte_hards(self):
        from cozempic import guard as guard_mod

        # ---- Per-cycle bookkeeping ----------------------------------------
        sleep_calls: list[float] = []
        max_cycles = 20
        interval = 30  # The default in production

        # ---- Patched time.sleep -------------------------------------------
        # Records every sleep; raises _StopAfterNSleeps after max_cycles so
        # the test terminates. Returns immediately (no real sleep) so the
        # test runs in <1s.
        def fake_sleep(duration: float) -> None:
            sleep_calls.append(float(duration))
            if len(sleep_calls) >= max_cycles:
                raise _StopAfterNSleeps(
                    f"Guard loop did not back off / exit after {max_cycles} "
                    f"cycles of 0-byte HARD prunes. Recorded sleeps: {sleep_calls}"
                )

        # ---- Fake session path + minimal session sidecar -------------------
        import tempfile
        from pathlib import Path as _P

        tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmpdir, ignore_errors=True))
        session_path = _P(tmpdir) / "fake_session.jsonl"
        session_path.write_text('{"type":"user","message":{"content":"hi"}}\n')

        fake_session = {"session_id": "fake-r2-uuid-2026-05-18", "path": session_path}

        # ---- Fake state object (looks like TeamState) ----------------------
        class _FakeState:
            subagents = []
            tasks = []
            message_count = 0

            def is_empty(self):
                return True

        # ---- Mock guard_prune_cycle: ALWAYS returns 0 bytes freed ---------
        prune_call_count = {"n": 0}

        def fake_prune_cycle(**kwargs):
            prune_call_count["n"] += 1
            return {
                "saved_mb": 0.0,  # critical: the bug condition
                "original_tokens": 600_000,
                "final_tokens": 600_000,
                "team_name": None,
                "team_messages": 0,
                "checkpoint_path": None,
                "backup_path": None,
                "reloading": False,
            }

        # ---- Mock quick_token_estimate: ALWAYS above HARD threshold ------
        # threshold_tokens=500_000, return 600_000 so HARD branch fires every cycle.
        def fake_quick_token_estimate(_p):
            return 600_000

        # ---- Mock the assorted infrastructure called by start_guard -------
        # checkpoint_team returns the fake state; load_messages returns []
        # so detect_context_window can run; record_session is a no-op.
        # Per H4 hygiene fix: patch ``guard_mod.time.sleep`` explicitly
        # rather than replacing the whole ``time`` module with a Mock.
        # Symmetric with test_guard_hard_loop_backoff.py — leaves the rest
        # of the time module intact so future code paths under test that
        # add new time.* calls don't break with confusing TypeErrors.
        with (
            patch.object(guard_mod.time, "sleep", side_effect=fake_sleep),
            patch.object(
                guard_mod, "_resolve_session_by_id", return_value=fake_session
            ),
            patch.object(guard_mod, "find_current_session", return_value=fake_session),
            patch.object(guard_mod, "find_claude_pid", return_value=None),
            patch.object(guard_mod, "checkpoint_team", return_value=_FakeState()),
            patch.object(guard_mod, "guard_prune_cycle", side_effect=fake_prune_cycle),
            patch.object(
                guard_mod, "quick_token_estimate", side_effect=fake_quick_token_estimate
            ),
            patch.object(guard_mod, "load_messages", return_value=[]),
            patch("cozempic.session.record_session"),
            patch.object(guard_mod, "_cleanup_stale_watchers"),
            patch.object(guard_mod, "ping_install_if_new"),
            patch.object(guard_mod, "maybe_auto_update"),
            patch.object(guard_mod, "cleanup_old_backups"),
            patch("cozempic.tokens.detect_context_window", return_value=1_000_000),
        ):

            # Run the loop. Expect either:
            #   - the loop exits cleanly under a back-off path (PASS), or
            #   - it raises _StopAfterNSleeps because it ran max_cycles
            #     without exiting (FAIL — the bug).
            raised = None
            try:
                guard_mod.start_guard(
                    cwd=tmpdir,
                    threshold_mb=100.0,
                    soft_threshold_mb=50.0,
                    rx_name="standard",
                    interval=interval,
                    auto_reload=False,  # don't try to kill Claude
                    reactive=False,  # don't spin up the watcher thread
                    threshold_tokens=500_000,
                    soft_threshold_tokens=250_000,
                    session_id="fake-r2-uuid-2026-05-18",
                )
            except _StopAfterNSleeps as e:
                raised = e
            except SystemExit as e:
                # Acceptable: guard.start_guard exits cleanly after detecting
                # sustained 0-byte HARDs and emitting a diagnostic. The handoff
                # specifies sys.exit(0) — what matters is that the loop did NOT
                # silently iterate forever. Any number of sleeps strictly fewer
                # than max_cycles proves the loop bounded itself.
                self.assertIn(e.code, (0, None, 1, 2), f"unexpected exit code {e.code}")
                self.assertLess(
                    len(sleep_calls),
                    max_cycles,
                    f"Loop exited but recorded {len(sleep_calls)} sleeps "
                    f"(>= max_cycles={max_cycles}); back-off was ineffective.",
                )
                return  # PASS — exit-with-diagnostic counts as a back-off behavior
            except Exception as e:
                raised = e

        # ---- RED assertions (desired behavior; current v1.8.11 FAILS) ----
        # Acceptable post-fix behavior (any of):
        #   (a) the loop raises a backoff exception → captured in `raised`
        #       as something OTHER than _StopAfterNSleeps,
        #   (b) the loop exponentially backs off → recorded sleeps grow
        #       monotonically over time, so the average late-cycle sleep
        #       is much larger than the original interval,
        #   (c) the loop exits cleanly with a diagnostic → SystemExit was
        #       raised above and we returned early.
        #
        # The current v1.8.11 bug: loop keeps firing at 30s with only a
        # one-shot 4×interval pause every 3 cycles, then resets the counter.
        # This test FAILS RED under that behavior.

        if isinstance(raised, _StopAfterNSleeps):
            # Loop never exited. Check whether back-off happened.
            # True back-off: late-cycle sleeps should be MUCH larger than
            # early-cycle sleeps. Compare last quarter vs first quarter average.
            n = len(sleep_calls)
            if n < 8:
                self.fail(
                    f"Loop did not back off (only {n} cycles recorded — "
                    f"too few to be a real back-off)."
                )
            first_q = sleep_calls[: n // 4]
            last_q = sleep_calls[-(n // 4) :]
            avg_first = sum(first_q) / len(first_q)
            avg_last = sum(last_q) / len(last_q)

            self.fail(
                "Guard loop kept firing HARD prunes at the original interval "
                "with no real back-off. Bug present.\n"
                f"  prune_calls={prune_call_count['n']}\n"
                f"  total_cycles={n}\n"
                f"  first-quarter avg sleep={avg_first:.1f}s  "
                f"last-quarter avg sleep={avg_last:.1f}s\n"
                f"  (true exponential back-off would push last-quarter avg "
                f"to >> {interval}s)\n"
                f"  sleeps={sleep_calls}"
            )

        if raised is not None and not isinstance(raised, _StopAfterNSleeps):
            # A different exception bubbled out — could be a real backoff
            # signal, or a setup bug. Accept it as PASS only if it carries
            # a message suggesting back-off was the cause.
            msg = str(raised).lower()
            if not any(
                kw in msg
                for kw in ("backoff", "back-off", "stalled", "ineffective", "0 bytes")
            ):
                self.fail(f"Unexpected exception (not back-off-related): {raised!r}")

        # If we got here, raised is None — loop completed cleanly.
        # That's an acceptable post-fix behavior IF the prune count is small.
        self.assertLess(
            prune_call_count["n"],
            max_cycles - 5,
            f"Loop returned cleanly but ran {prune_call_count['n']} prunes "
            f"— that's the bug (no back-off, no exit, just exhausted sleeps).",
        )


if __name__ == "__main__":
    unittest.main()
