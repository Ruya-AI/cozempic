"""RED tests for the active-session safety guard (P0-A).

Per PLAN.md § 2.2:
- guard.guard_prune_cycle() MUST refuse to prune a session whose JSONL mtime is
  within `min_idle_hours` (default 24h) unless force=True.
- cli.cmd_treat exposes --force (default False); without --force on a fresh-mtime
  session, refuses with exit code 4.
- Refusals from the daemon are silent no-ops (return _no_change dict) and do NOT
  advance the K=10 exit counter.
- Env var COZEMPIC_MIN_IDLE_HOURS overrides default; invalid/out-of-range falls
  back to 24.

All tests in this module SHOULD FAIL until P0-A is implemented. The new symbols
they exercise (`cozempic.safety.ActiveSessionError`, `is_session_idle`,
`assert_session_idle_or_force`, the `force` kwarg on `guard_prune_cycle`, and the
`--force` CLI flag) do not yet exist in the codebase.
"""

from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


# RED-state contract: every test must FAIL (not skip) until the safety module
# exists. Per the project's testing iron law, a skipped test proves nothing —
# it cannot catch a regression. We import the module inside each test so the
# failure mode is an ImportError raised AT test execution (genuine red), not
# at collection (which pytest reports as a collection error and would mask
# individual scenarios).


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    """Helper: write a list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d, separators=(",", ":")) + "\n")


def _set_mtime(path: Path, seconds_ago: float) -> None:
    """Set the file's mtime to `now - seconds_ago`."""
    now = time.time()
    target = now - seconds_ago
    os.utime(path, (target, target))


def _minimal_session_lines() -> list[dict]:
    """Return a tiny valid session that satisfies all post-prune validation checks.

    Two users + two assistants with a valid parent chain rooted at parentUuid=None,
    plus the metadata singletons that P0-D / P0-B require.
    """
    return [
        {"type": "user", "uuid": "u1", "parentUuid": None,
         "message": {"role": "user", "content": "hi"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}},
        {"type": "user", "uuid": "u2", "parentUuid": "a1",
         "message": {"role": "user", "content": "second prompt"}},
        {"type": "assistant", "uuid": "a2", "parentUuid": "u2",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "world"}]}},
        {"type": "ai-title", "uuid": "t1", "parentUuid": "a2", "title": "Hello session"},
        {"type": "last-prompt", "uuid": "lp1", "parentUuid": "a2", "text": "second prompt"},
        {"type": "permission-mode", "uuid": "pm1", "parentUuid": "a2", "mode": "default"},
    ]


class TestActiveSessionGuard(unittest.TestCase):
    """Core guard behaviour: refuse on fresh mtime, accept on idle / force."""

    def test_treat_refuses_active_session_within_threshold(self):
        """A session modified 5 min ago must NOT be pruned without --force.

        Expected behaviour after fix:
          - assert_session_idle_or_force raises ActiveSessionError, OR
          - cmd_treat returns exit code 4 with a clear message
          - the JSONL file is unchanged byte-for-byte.
        """
        from cozempic.safety import (  # type: ignore[import-not-found]
            ActiveSessionError,
            assert_session_idle_or_force,
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.jsonl"
            _write_jsonl(path, _minimal_session_lines())
            _set_mtime(path, seconds_ago=5 * 60)  # 5 minutes ago

            with mock.patch.dict(os.environ, {"COZEMPIC_MIN_IDLE_HOURS": "24"}):
                with self.assertRaises(ActiveSessionError):
                    assert_session_idle_or_force(
                        path, min_idle_hours=24.0, force=False,
                    )

    def test_treat_accepts_active_session_with_force(self):
        """`force=True` overrides the guard regardless of mtime."""
        from cozempic.safety import assert_session_idle_or_force  # type: ignore

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.jsonl"
            _write_jsonl(path, _minimal_session_lines())
            _set_mtime(path, seconds_ago=5 * 60)

            # Must not raise — force bypasses the threshold.
            assert_session_idle_or_force(
                path, min_idle_hours=24.0, force=True,
            )

    def test_treat_accepts_idle_session_without_force(self):
        """A session whose mtime is older than the threshold prunes without --force."""
        from cozempic.safety import assert_session_idle_or_force  # type: ignore

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.jsonl"
            _write_jsonl(path, _minimal_session_lines())
            # 25 hours ago, threshold is 24h → idle.
            _set_mtime(path, seconds_ago=25 * 3600)

            assert_session_idle_or_force(
                path, min_idle_hours=24.0, force=False,
            )

    def test_is_session_idle_returns_minutes_since_mtime(self):
        """is_session_idle returns (bool, minutes) tuple per PLAN § 2.2."""
        from cozempic.safety import is_session_idle  # type: ignore

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.jsonl"
            _write_jsonl(path, _minimal_session_lines())
            _set_mtime(path, seconds_ago=10 * 60)  # 10 minutes ago

            idle, minutes = is_session_idle(path, min_idle_hours=24.0)
            self.assertFalse(idle)
            self.assertAlmostEqual(minutes, 10.0, delta=0.5)

            _set_mtime(path, seconds_ago=25 * 3600)  # 25 hours ago
            idle, minutes = is_session_idle(path, min_idle_hours=24.0)
            self.assertTrue(idle)
            self.assertAlmostEqual(minutes, 25 * 60.0, delta=1.0)


class TestActiveSessionErrorShape(unittest.TestCase):
    """The ActiveSessionError exception carries actionable context for the operator."""

    def test_exception_carries_path_minutes_and_threshold(self):
        from cozempic.safety import (  # type: ignore
            ActiveSessionError,
            assert_session_idle_or_force,
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.jsonl"
            _write_jsonl(path, _minimal_session_lines())
            _set_mtime(path, seconds_ago=120)  # 2 minutes ago

            with self.assertRaises(ActiveSessionError) as ctx:
                assert_session_idle_or_force(
                    path, min_idle_hours=24.0, force=False,
                )

            exc = ctx.exception
            self.assertEqual(exc.path, path)
            self.assertAlmostEqual(exc.idle_minutes, 2.0, delta=0.5)
            self.assertEqual(exc.threshold_hours, 24.0)
            # Message should mention both the elapsed time and the override.
            msg = str(exc).lower()
            self.assertIn("force", msg)


class TestGuardPruneCycleRespectsActiveSession(unittest.TestCase):
    """guard.guard_prune_cycle must short-circuit on an active session."""

    def test_guard_prune_cycle_skips_active_session_and_returns_no_change(self):
        """Daemon calls guard_prune_cycle WITHOUT force; refusal returns _no_change shape.

        The fix adds an explicit `active_session_refused` flag to the result
        dict (per PLAN § 2.2 plumbing) so the daemon's K-counter can distinguish
        an upfront refusal from a generic zero-byte prune. Without the fix the
        result dict comes back without this key → assertion fails RED.
        """
        from cozempic.guard import guard_prune_cycle  # noqa: F401

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.jsonl"
            _write_jsonl(path, _minimal_session_lines())
            _set_mtime(path, seconds_ago=60)  # 1 minute ago — clearly active

            with mock.patch.dict(os.environ, {"COZEMPIC_MIN_IDLE_HOURS": "24"}):
                result = guard_prune_cycle(
                    session_path=path,
                    rx_name="gentle",
                    config={},
                    auto_reload=False,
                    cwd=str(path.parent),
                    session_id="test-session",
                )

            # Distinguishes the refusal from generic _no_change (which would
            # also have saved_mb==0). active_session_refused is new in P0-A.
            self.assertTrue(
                result.get("active_session_refused", False),
                msg=f"Expected active_session_refused=True, got result={result}",
            )
            self.assertEqual(result.get("saved_mb", -1.0), 0.0)
            self.assertFalse(result.get("reloading", True))
            bak_files = list(path.parent.glob("*.jsonl.bak"))
            self.assertEqual(bak_files, [])

    def test_active_session_refusal_does_not_advance_k_counter(self):
        """A refusal-due-to-active-session must not count toward the K=10 exit ladder.

        The K counter only advances on prunes that ran but freed 0 bytes; an
        upfront refusal is semantically different and should be invisible to it.

        This test inspects the result dict for a new key indicating refusal —
        guard's main loop uses that key to decide whether to advance K.
        """
        from cozempic.guard import guard_prune_cycle  # noqa: F401

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.jsonl"
            _write_jsonl(path, _minimal_session_lines())
            _set_mtime(path, seconds_ago=60)

            with mock.patch.dict(os.environ, {"COZEMPIC_MIN_IDLE_HOURS": "24"}):
                result = guard_prune_cycle(
                    session_path=path,
                    rx_name="gentle",
                    config={},
                    auto_reload=False,
                    cwd=str(path.parent),
                    session_id="test-session",
                )

            # New flag introduced by P0-A: separates "refused upfront" from
            # "ran and saved 0 bytes". Daemon's K-counter ignores the former.
            self.assertTrue(result.get("active_session_refused", False))


class TestEnvVarOverride(unittest.TestCase):
    """COZEMPIC_MIN_IDLE_HOURS env var controls the default threshold."""

    def test_env_var_zero_always_prunes(self):
        """COZEMPIC_MIN_IDLE_HOURS=0 disables the active-session guard entirely."""
        from cozempic.safety import assert_session_idle_or_force  # type: ignore

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.jsonl"
            _write_jsonl(path, _minimal_session_lines())
            _set_mtime(path, seconds_ago=30)  # 30 seconds ago — VERY active

            with mock.patch.dict(os.environ, {"COZEMPIC_MIN_IDLE_HOURS": "0"}):
                # min_idle_hours=0.0 means "any age is idle enough"
                assert_session_idle_or_force(
                    path, min_idle_hours=0.0, force=False,
                )

    def test_env_var_invalid_falls_back_to_default(self):
        """Garbage env values fall back to the documented 24h default."""
        # Helper for resolving the active threshold from env. Per PLAN § 2.2
        # this lives in cozempic.safety (module-level loader).
        from cozempic.safety import resolve_min_idle_hours  # type: ignore

        # Invalid string
        with mock.patch.dict(os.environ, {"COZEMPIC_MIN_IDLE_HOURS": "garbage"}):
            self.assertEqual(resolve_min_idle_hours(), 24.0)

        # Negative — out of allowed range [0.0, 168.0]
        with mock.patch.dict(os.environ, {"COZEMPIC_MIN_IDLE_HOURS": "-5"}):
            self.assertEqual(resolve_min_idle_hours(), 24.0)

        # Way too large — out of range
        with mock.patch.dict(os.environ, {"COZEMPIC_MIN_IDLE_HOURS": "9999"}):
            self.assertEqual(resolve_min_idle_hours(), 24.0)

        # Unset → default
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_min_idle_hours(), 24.0)

    def test_env_var_valid_float_used(self):
        from cozempic.safety import resolve_min_idle_hours  # type: ignore

        with mock.patch.dict(os.environ, {"COZEMPIC_MIN_IDLE_HOURS": "48"}):
            self.assertEqual(resolve_min_idle_hours(), 48.0)

        with mock.patch.dict(os.environ, {"COZEMPIC_MIN_IDLE_HOURS": "0.5"}):
            self.assertEqual(resolve_min_idle_hours(), 0.5)


class TestTreatCLIForceFlag(unittest.TestCase):
    """The `cozempic treat` CLI accepts --force; without it, exit code 4 on active."""

    def test_cli_parser_accepts_force_flag(self):
        """The argparse `treat` subparser must expose --force."""
        from cozempic.cli import build_parser

        parser = build_parser()
        # Should not raise SystemExit on --force being present.
        ns = parser.parse_args(["treat", "some-session", "--force"])
        self.assertTrue(getattr(ns, "force", False))


if __name__ == "__main__":
    unittest.main()
