"""RED tests for review-max Group B — input validation hardening.

Findings:
  B.2  _clamp_float lets NaN through
  B.10 C1 treats parentUuid='' as non-null
  B.11 C2 picks FIRST parentUuid=null as canonical root (multi-root sessions)
  B.13 Clock skew on is_session_idle
"""

from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from cozempic.helpers import msg_bytes


def _msg(idx: int, payload: dict) -> tuple[int, dict, int]:
    return (idx, payload, msg_bytes(payload))


# ─── B.2 — NaN / Inf env values fall back to default ─────────────────────────


class TestClampFloatRejectsNaNAndInf(unittest.TestCase):

    def test_nan_env_value_falls_back_to_default(self):
        from cozempic.safety import resolve_min_idle_hours

        with mock.patch.dict(os.environ, {"COZEMPIC_MIN_IDLE_HOURS": "nan"}):
            self.assertEqual(resolve_min_idle_hours(), 24.0)

    def test_inf_env_value_falls_back_to_default(self):
        from cozempic.safety import resolve_min_idle_hours

        with mock.patch.dict(os.environ, {"COZEMPIC_MIN_IDLE_HOURS": "inf"}):
            self.assertEqual(resolve_min_idle_hours(), 24.0)

        with mock.patch.dict(os.environ, {"COZEMPIC_MIN_IDLE_HOURS": "-inf"}):
            self.assertEqual(resolve_min_idle_hours(), 24.0)


# ─── B.10 — parentUuid='' (empty) is treated as null by C1 ───────────────────


class TestC1AcceptsEmptyParentUuid(unittest.TestCase):
    """B.10: C1's `parent is None: continue` misses parentUuid='' (empty
    string), which downstream tools may emit. Empty parents should be
    treated equivalently to None — they're not chain references."""

    def test_validate_does_not_raise_on_empty_parent_uuid(self):
        from cozempic.safety import validate_post_prune

        before = [
            _msg(0, {"type": "user", "uuid": "u0", "parentUuid": None,
                     "message": {"role": "user", "content": "hi"}}),
            _msg(1, {"type": "assistant", "uuid": "a1", "parentUuid": "u0",
                     "message": {"role": "assistant",
                                 "content": [{"type": "text", "text": "x"}]}}),
        ]
        # msgs_after — same shape but a1's parentUuid is '' (empty string).
        # Pre-fix C1 throws because '' is not in surviving_uuids and is not
        # None. Post-fix C1 treats empty as null and accepts.
        after = [
            _msg(0, {"type": "user", "uuid": "u0", "parentUuid": None,
                     "message": {"role": "user", "content": "hi"}}),
            _msg(1, {"type": "assistant", "uuid": "a1", "parentUuid": "",
                     "message": {"role": "assistant",
                                 "content": [{"type": "text", "text": "x"}]}}),
        ]
        # Must not raise.
        validate_post_prune(before, after)


# ─── B.11 — multi-root sessions: at least one original root must survive ────


class TestC2HandlesMultiRoot(unittest.TestCase):
    """B.11: pre-fix C2 captures only the FIRST parentUuid=null in
    msgs_before and requires that exact uuid in msgs_after. Sessions that
    legitimately have multiple roots (resume-of-resume, fork-merge) raise
    a spurious C2 even though a different valid root survives.

    Post-fix: collect ALL original roots; require `len(original_roots &
    surviving) >= 1` — at least one root must survive."""

    def test_validation_passes_when_only_second_root_survives(self):
        from cozempic.safety import validate_post_prune

        # Two roots: u0 (older) and u_new (newer resume entry).
        before = [
            _msg(0, {"type": "user", "uuid": "u0", "parentUuid": None,
                     "message": {"role": "user", "content": "old root"}}),
            _msg(1, {"type": "assistant", "uuid": "a0", "parentUuid": "u0",
                     "message": {"role": "assistant",
                                 "content": [{"type": "text", "text": "x"}]}}),
            _msg(2, {"type": "user", "uuid": "u_new", "parentUuid": None,
                     "message": {"role": "user", "content": "new root"}}),
            _msg(3, {"type": "assistant", "uuid": "a_new",
                     "parentUuid": "u_new",
                     "message": {"role": "assistant",
                                 "content": [{"type": "text", "text": "y"}]}}),
        ]
        # Drop the older root + its descendant; keep the newer root.
        after = [
            _msg(2, {"type": "user", "uuid": "u_new", "parentUuid": None,
                     "message": {"role": "user", "content": "new root"}}),
            _msg(3, {"type": "assistant", "uuid": "a_new",
                     "parentUuid": "u_new",
                     "message": {"role": "assistant",
                                 "content": [{"type": "text", "text": "y"}]}}),
        ]
        # Must not raise — u_new is one of the original roots and survives.
        validate_post_prune(before, after)


# ─── B.13 — clock skew (mtime in the future) treated as idle ─────────────────


class TestIsSessionIdleHandlesClockSkew(unittest.TestCase):
    """B.13: NTP correction or DST can leave a file mtime AHEAD of now.
    The pre-fix `max(0.0, now - mtime)` then yields elapsed=0 → not idle
    → refuses prune for up to 25h. Post-fix: a mtime more than 5 minutes
    in the future is treated as a clock-skew anomaly and reported as
    idle so the daemon doesn't get stuck."""

    def test_future_mtime_treated_as_idle(self):
        from cozempic.safety import is_session_idle

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.jsonl"
            path.write_text("{}\n")
            # Set mtime 1 hour into the future (clock-skew anomaly).
            future = time.time() + 3600
            os.utime(path, (future, future))

            idle, _ = is_session_idle(path, min_idle_hours=24.0)
            self.assertTrue(
                idle,
                msg="future-mtime clock-skew anomaly should be treated as "
                    "idle, not refused for up to 25h",
            )


if __name__ == "__main__":
    unittest.main()
