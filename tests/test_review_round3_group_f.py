"""RED tests for review-round3 Group F — correctness.

Findings:
  F.N2  _PruneLock.__exit__ unlinks AFTER releasing flock (cross-process race)
  F.N4  enforce_floor re-add silently undoes strategy REPLACEMENTS
  F.N7  _resolve_floor_with hardcodes preserve_first_message=True (file_data ignored)
  F.M1  _clamp_int doesn't handle inf (class-of-bug fold from B.2)
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from cozempic.helpers import msg_bytes


def _msg(idx: int, payload: dict) -> tuple[int, dict, int]:
    return (idx, payload, msg_bytes(payload))


# ─── F.N2 — _PruneLock cross-process race ────────────────────────────────────


def _lock_worker(session_path_str: str, holdtime: float, started_path: str,
                 done_path: str) -> int:
    """Subprocess worker: acquire _PruneLock, hold for `holdtime`, release."""
    # Re-import inside the subprocess (multiprocessing spawn-safe).
    from pathlib import Path as P
    from cozempic.session import _PruneLock, PruneLockError
    path = P(session_path_str)
    try:
        with _PruneLock(path):
            P(started_path).touch()
            time.sleep(holdtime)
        P(done_path).touch()
        return 0
    except PruneLockError:
        # Could not acquire — record as 'contention'.
        P(done_path).write_text("contention")
        return 1


class TestPruneLockNoCrossProcessRace(unittest.TestCase):
    """F.N2 — concurrent _PruneLock acquisitions on the SAME path must be
    serialized; the unlink-after-unlock race must not allow simultaneous
    critical-section entry.
    """

    def test_two_processes_cannot_both_hold_lock(self):
        with TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "session.jsonl"
            session_path.write_text("{}\n")

            started_a = Path(tmpdir) / "a_started"
            done_a = Path(tmpdir) / "a_done"
            started_b = Path(tmpdir) / "b_started"
            done_b = Path(tmpdir) / "b_done"

            ctx = multiprocessing.get_context("spawn")
            p_a = ctx.Process(
                target=_lock_worker,
                args=(str(session_path), 1.0,
                      str(started_a), str(done_a)),
            )
            p_b = ctx.Process(
                target=_lock_worker,
                args=(str(session_path), 0.2,
                      str(started_b), str(done_b)),
            )
            p_a.start()

            # Wait for A to actually be inside the lock before launching B.
            deadline = time.time() + 5.0
            while not started_a.exists() and time.time() < deadline:
                time.sleep(0.02)
            self.assertTrue(started_a.exists(), "A failed to start within 5s")

            p_b.start()
            p_a.join(timeout=10.0)
            p_b.join(timeout=10.0)

            # A must have finished cleanly. B must have either: (a) been
            # serialized after A (then started AFTER A's done marker), OR
            # (b) hit PruneLockError contention.
            self.assertTrue(done_a.exists(), "A never marked done")
            self.assertTrue(done_b.exists(), "B never marked done")

            # The critical-section race we are guarding against: A and B
            # both holding the lock at the same time. Equivalent to: B
            # started its critical section BEFORE A finished. If B was
            # serialized, started_b's mtime > done_a's mtime. If B
            # contended, done_b carries the marker.
            if done_b.read_text(errors="ignore") == "contention":
                return  # contention path — correct behaviour
            self.assertGreaterEqual(
                started_b.stat().st_mtime, done_a.stat().st_mtime,
                msg="F.N2: B entered critical section before A exited — "
                    "the unlink-after-unlock race fired",
            )


# ─── F.N4 — floor must preserve strategy replacements ───────────────────────


class TestEnforceFloorPreservesReplacement(unittest.TestCase):
    """F.N4: if a strategy REPLACED a message in-place (same uuid, modified
    payload), enforce_floor must NOT re-insert the original from msgs_before
    — that would silently undo the strategy's modification. The replaced
    version (already in msgs_after) is the one that ships."""

    def test_floor_does_not_revert_replacement(self):
        from cozempic.safety import FloorConfig, enforce_floor

        # Before: a single root user with a long original payload.
        before = [
            _msg(0, {"type": "user", "uuid": "u0", "parentUuid": None,
                     "message": {"role": "user",
                                 "content": "a" * 500}}),
            _msg(1, {"type": "assistant", "uuid": "a1", "parentUuid": "u0",
                     "message": {"role": "assistant",
                                 "content": [{"type": "text",
                                              "text": "reply"}]}}),
        ]
        # After: the strategy replaced u0 in place (same uuid, truncated).
        truncated = {
            "type": "user", "uuid": "u0", "parentUuid": None,
            "message": {"role": "user", "content": "[truncated]"},
        }
        after = [
            (0, truncated, msg_bytes(truncated)),
            before[1],
        ]

        cfg = FloorConfig(
            max_user_assistant_drop_pct=0.50,
            preserve_last_k_turns=50,
            preserve_first_message=True,
        )
        result = enforce_floor(before, after, cfg=cfg)

        # u0 must be present exactly ONCE and must be the TRUNCATED version.
        u0_entries = [m for m in result if m[1].get("uuid") == "u0"]
        self.assertEqual(len(u0_entries), 1,
                         msg="floor introduced a duplicate u0")
        content = u0_entries[0][1].get("message", {}).get("content", "")
        self.assertEqual(
            content, "[truncated]",
            msg="F.N4: floor re-inserted the ORIGINAL u0 payload, "
                "silently undoing the strategy's replacement",
        )


# ─── F.N7 — config preserve_first_message field reads file_data ──────────────


class TestPreserveFirstMessageReadsFromFile(unittest.TestCase):

    def test_file_data_can_disable_preserve_first_message(self):
        from cozempic.config import _resolve_floor_with

        cfg = _resolve_floor_with({"floor": {"preserve_first_message": False}})
        self.assertFalse(
            cfg.preserve_first_message,
            msg="F.N7: file_data['floor']['preserve_first_message']=False "
                "was silently ignored; field hardcoded to True",
        )

    def test_file_data_default_keeps_preserve_first_message_true(self):
        from cozempic.config import _resolve_floor_with
        cfg = _resolve_floor_with({})
        self.assertTrue(cfg.preserve_first_message)

    def test_env_var_can_disable_preserve_first_message(self):
        from cozempic.config import _resolve_floor_with

        with mock.patch.dict(os.environ, {"COZEMPIC_FLOOR_PRESERVE_FIRST": "0"}):
            cfg = _resolve_floor_with({})
        self.assertFalse(cfg.preserve_first_message)

        with mock.patch.dict(os.environ, {"COZEMPIC_FLOOR_PRESERVE_FIRST": "false"}):
            cfg = _resolve_floor_with({})
        self.assertFalse(cfg.preserve_first_message)


# ─── F.M1 — _clamp_int handles inf / nan ─────────────────────────────────────


class TestClampIntRejectsInfAndNan(unittest.TestCase):
    """F.M1: int(float('inf')) raises OverflowError, not in _clamp_int's
    except tuple. int(float('nan')) raises ValueError (covered) but string
    'inf' / 'nan' must also short-circuit before conversion."""

    def test_float_inf_falls_back_to_default(self):
        from cozempic.config import _clamp_int
        self.assertEqual(_clamp_int(float("inf"), 1, 1000, 50), 50)
        self.assertEqual(_clamp_int(float("-inf"), 1, 1000, 50), 50)

    def test_float_nan_falls_back_to_default(self):
        from cozempic.config import _clamp_int
        self.assertEqual(_clamp_int(float("nan"), 1, 1000, 50), 50)

    def test_string_inf_nan_falls_back_to_default(self):
        from cozempic.config import _clamp_int
        for tok in ("inf", "Inf", "INF", "-inf", "nan", "NaN"):
            self.assertEqual(
                _clamp_int(tok, 1, 1000, 50), 50,
                msg=f"_clamp_int({tok!r}) must return default 50",
            )

    def test_env_var_inf_falls_back_to_default(self):
        from cozempic.safety import resolve_min_idle_hours  # B.2 sibling
        # _clamp_int isn't on the env path for hours but the helper is shared
        # for the floor's preserve_last_k env var. Smoke check directly.
        from cozempic.config import _clamp_int
        self.assertEqual(_clamp_int("inf", 1, 1000, 50), 50)


if __name__ == "__main__":
    unittest.main()
