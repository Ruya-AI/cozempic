"""RED tests for review-max Group E — completeness.

Findings:
  E.4  ai-title missing C7 check (mirrors C5/C6 for last-of-type metadata)
  E.6  Floor target rounding edge cases (replace +0.999999 magic with math.ceil)
  E.9  load_config double config.json read (single-read refactor)
  E.14 TOCTOU on is_session_idle vs _PruneLock (re-check mtime INSIDE the lock)
"""

from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from cozempic.helpers import msg_bytes


def _msg(idx: int, payload: dict) -> tuple[int, dict, int]:
    return (idx, payload, msg_bytes(payload))


# ─── E.4 — C7 for ai-title ───────────────────────────────────────────────────


class TestC7AiTitlePreserved(unittest.TestCase):
    """E.4: executor._LAST_OF_TYPE_PROTECTED tags ai-title as a singleton,
    but safety.validate_post_prune has no corresponding C5-style check.
    A future code path that bypasses the tag (e.g., a tag-aware caller
    that disables it) would silently let the last ai-title drop.
    Add C7 mirroring C5/C6."""

    def test_validation_fails_when_last_ai_title_dropped(self):
        from cozempic.safety import PruneValidationError, validate_post_prune

        before = [
            _msg(0, {"type": "user", "uuid": "u0", "parentUuid": None,
                     "message": {"role": "user", "content": "hi"}}),
            _msg(1, {"type": "assistant", "uuid": "a0", "parentUuid": "u0",
                     "message": {"role": "assistant",
                                 "content": [{"type": "text", "text": "x"}]}}),
            _msg(2, {"type": "ai-title", "uuid": "t0", "parentUuid": "a0",
                     "title": "Old"}),
            _msg(3, {"type": "ai-title", "uuid": "t1", "parentUuid": "a0",
                     "title": "Latest"}),
        ]
        # Drop the last ai-title only (t1); structural C2/C3 still pass.
        after = [m for m in before if m[1].get("uuid") != "t1"]

        with self.assertRaises(PruneValidationError) as ctx:
            validate_post_prune(before, after)
        self.assertEqual(ctx.exception.evidence.get("failed_check"), "C7")


# ─── E.6 — Floor target rounding uses math.ceil ──────────────────────────────


class TestFloorRoundingUsesCeil(unittest.TestCase):
    """E.6: the prior `int(survival_floor_pct * total + 0.999999)` magic
    number is fragile for small totals. Switch to math.ceil and don't
    apply the floor cap on micro-sessions (single message)."""

    def test_floor_does_not_override_single_message_session(self):
        from cozempic.safety import FloorConfig, enforce_floor

        # Single message session (just a root user, no assistants at all).
        before = [
            _msg(0, {"type": "user", "uuid": "u0", "parentUuid": None,
                     "message": {"role": "user", "content": "single"}}),
        ]
        # Strategy keeps the root (no assistants to drop).
        after = list(before)
        cfg = FloorConfig(
            max_user_assistant_drop_pct=0.50,
            preserve_last_k_turns=50,
            preserve_first_message=True,
        )
        re_added = enforce_floor(before, after, cfg=cfg)
        # Must not crash and must not introduce phantom entries.
        self.assertEqual(len(re_added), 1)
        self.assertEqual(re_added[0][1].get("uuid"), "u0")

    def test_floor_target_uses_math_ceil_not_magic_offset(self):
        # Source-level check: enforce_floor uses math.ceil, not the
        # +0.999999 magic offset.
        import inspect
        from cozempic import safety as safety_mod

        src = inspect.getsource(safety_mod.enforce_floor)
        self.assertNotIn(
            "0.999999", src,
            msg="enforce_floor must use math.ceil instead of +0.999999 "
                "magic-number rounding (E.6)",
        )
        self.assertIn(
            "math.ceil", src,
            msg="enforce_floor must use math.ceil for survival cap target",
        )


# ─── E.9 — load_config reads ~/.cozempic/config.json once ────────────────────


class TestLoadConfigSingleFileRead(unittest.TestCase):
    """E.9: load_config previously called _read_config_file twice — once
    inside resolve_min_idle_hours, once inside _resolve_floor. Fix: read
    once at the top of load_config, pass the parsed dict to both
    resolvers."""

    def test_load_config_reads_config_file_at_most_once(self):
        from cozempic import config as config_mod

        with mock.patch.object(
            config_mod, "_read_config_file",
            wraps=config_mod._read_config_file,
        ) as spy:
            config_mod.load_config()
        self.assertLessEqual(
            spy.call_count, 1,
            msg=f"load_config called _read_config_file {spy.call_count} "
                "times — must be at most 1 (E.9)",
        )


# ─── E.14 — TOCTOU: re-check mtime inside _PruneLock ─────────────────────────


class TestActiveSessionRecheckedInsideLock(unittest.TestCase):
    """E.14: guard.guard_prune_cycle's assert_session_idle_or_force runs
    BEFORE _PruneLock acquires. Claude can write between the gate and the
    lock — the first check passed but the file is now active. Fix: re-stat
    AFTER lock acquisition and refuse if the freshly-stat'd mtime falls
    within the threshold."""

    def test_guard_re_stats_mtime_inside_prune_lock(self):
        # Source-level check: guard_prune_cycle's body (post-lock acquire)
        # must contain a second is_session_idle / assert_session_idle_or_force
        # call.
        from cozempic import guard as guard_mod
        import inspect
        src = inspect.getsource(guard_mod.guard_prune_cycle)
        # The first call sits before `with _PruneLock`; the fix adds one
        # immediately after.
        prelock, sep, postlock = src.partition("with _PruneLock")
        self.assertTrue(sep, "guard_prune_cycle source must contain `with _PruneLock`")
        # Post-lock body must invoke the idle gate again.
        gate_terms = ("assert_session_idle_or_force", "is_session_idle")
        self.assertTrue(
            any(term in postlock for term in gate_terms),
            msg="E.14: guard_prune_cycle must re-check mtime INSIDE the "
                "_PruneLock to close the TOCTOU window between the upfront "
                "gate and the actual prune",
        )


if __name__ == "__main__":
    unittest.main()
