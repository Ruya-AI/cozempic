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
    """guard.guard_prune_cycle must short-circuit on an active session.

    Two cases:
      - user-initiated path (force=False): MUST refuse on active session (P0-A guard stays).
      - daemon-managed path (force=True): MUST prune even when session is active.
    """

    def test_guard_prune_cycle_user_treat_force_false_refuses_active(self):
        """User-initiated path: force=False on fresh mtime → active_session_refused=True.

        P0-A protection for user-initiated paths. This test STAYS GREEN through
        P1 because the daemon's force=True fix does not affect the force=False path.
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
                    force=False,
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

    def test_guard_prune_cycle_daemon_force_true_prunes_active(self):
        """Daemon HARD2 tier must call guard_prune_cycle with force=True.

        RED at current HEAD: the HARD2 call at guard.py:614 omits force= (defaults
        to False). After fix: the daemon call site explicitly passes force=True so
        over-threshold sessions are pruned even when mtime is recent.

        Implementation: inspect the source code of start_guard for the 4 daemon
        call sites and assert each one passes force=True (after the fix).
        This is a structural test — it is RED because the current source omits
        force= at guard.py:614, 649, 661, and 858.
        """
        import inspect
        import cozempic.guard as guard_mod

        src = inspect.getsource(guard_mod.start_guard)

        # Find all guard_prune_cycle( call blocks in start_guard.
        # After fix, every call site that does auto-prune work must pass force=True.
        # We check that the source contains NO call to guard_prune_cycle without force=True
        # in the daemon's managed (HARD1/HARD2/SOFT) paths.
        # The HARD1/HARD2 paths must use force=True; SOFT also should.
        # RED signal: "guard_prune_cycle(" appears without "force=True" nearby.

        import re
        # Extract each guard_prune_cycle call block (everything from the call
        # to the closing paren) from start_guard's source.
        call_blocks = []
        for m in re.finditer(r"guard_prune_cycle\(", src):
            start = m.start()
            # Collect up to 600 chars after the opening paren for the kwarg list
            snippet = src[start:start + 600]
            call_blocks.append(snippet)

        self.assertTrue(
            call_blocks,
            "No guard_prune_cycle calls found in start_guard — source changed unexpectedly.",
        )

        missing_force = [b for b in call_blocks if "force=True" not in b]
        self.assertEqual(
            missing_force, [],
            msg=(
                f"Found {len(missing_force)} guard_prune_cycle call(s) in start_guard "
                f"that do NOT pass force=True. Daemon tiers must use force=True so active "
                f"sessions are pruned. Offending snippets (first 150 chars each):\n"
                + "\n---\n".join(b[:150] for b in missing_force)
            ),
        )

    def test_guard_prune_cycle_skips_active_session_and_returns_no_change(self):
        """Legacy name preserved for backward compat. Delegates to user-treat variant."""
        self.test_guard_prune_cycle_user_treat_force_false_refuses_active()

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


# ── P3 RED tests — C1 baseline-relative check (PLAN.md §2.5) ──────────────

class TestC1BaselineRelative(unittest.TestCase):
    """validate_post_prune C1 must compare against msgs_before, not fire
    unconditionally on any unresolved parentUuid.

    A resumed/forked session has a first message whose parentUuid points to the
    parent session — that parent was absent BEFORE the prune AND AFTER it. C1
    must NOT fire (the prune introduced no break). The fix mirrors the existing
    C2 baseline pattern (safety.py:250–269).
    """

    def _pack(self, raw: list[dict]) -> list[tuple[int, dict, int]]:
        """Pack raw dicts into (line, msg, bytes) triples."""
        return [(i, m, len(json.dumps(m))) for i, m in enumerate(raw)]

    def test_c1_does_not_fire_on_preexisting_external_parent(self):
        """Cross-session pointer that was absent before the prune must not trigger C1.

        Setup: session with first message whose parentUuid == "external-uuid"
        (never defined in this file, neither before nor after). The prune removes
        one message (so before has more UUIDs than after), triggering baseline mode.
        The cross-session pointer was absent before AND after — no chain break introduced.

        Current code raises PruneValidationError(C1) for any unresolved parent.
        Fixed code: skip cross-session pointers in baseline mode.
        """
        from cozempic.safety import validate_post_prune  # type: ignore

        # msgs_before: session with extra messages (to be pruned) + cross-session pointer.
        # r0 is a true root that survives, allowing C2 to pass.
        # m_old / a_old are prunable messages (not roots, not singletons).
        r0 = {"type": "user", "uuid": "r0", "parentUuid": None,
              "message": {"role": "user", "content": "root"}}
        a0 = {"type": "assistant", "uuid": "a0", "parentUuid": "r0",
              "message": {"role": "assistant",
                          "content": [{"type": "text", "text": "reply"}]}}
        m_old = {"type": "user", "uuid": "m_old", "parentUuid": "a0",
                 "message": {"role": "user", "content": "old message"}}
        a_old = {"type": "assistant", "uuid": "a_old", "parentUuid": "m_old",
                 "message": {"role": "assistant",
                             "content": [{"type": "text", "text": "old reply"}]}}
        msgs = [
            r0, a0, m_old, a_old,
            # This message has a cross-session parentUuid (from a resumed session)
            {"type": "user", "uuid": "u1", "parentUuid": "external-uuid",
             "message": {"role": "user", "content": "hi"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "message": {"role": "assistant",
                         "content": [{"type": "text", "text": "hello"}]}},
            {"type": "ai-title", "uuid": "t1", "parentUuid": "a1",
             "title": "Test session"},
            {"type": "last-prompt", "uuid": "lp1", "parentUuid": "a1",
             "text": "hi"},
            {"type": "permission-mode", "uuid": "pm1", "parentUuid": "a1",
             "mode": "default"},
        ]
        msgs_before = self._pack(msgs)
        # Prune removes m_old and a_old — msgs_before has more UUIDs than msgs_after.
        # r0 survives (C2 passes). cross-session pointer (external-uuid) was absent
        # before AND after — no chain break introduced by the prune.
        msgs_after = self._pack([m for m in msgs if m.get("uuid") not in ("m_old", "a_old")])
        validate_post_prune(msgs_before, msgs_after)  # must NOT raise

    def test_c1_fires_on_prune_introduced_break(self):
        """Intra-file parent that existed before but was pruned away must fire C1.

        Setup: msgs_before has an intact chain root→m1→m2→m3. msgs_after drops
        m1 but keeps m2 (parentUuid="m1") — prune broke the chain. C1 must fire.

        The session is constructed so C2 (root preservation) passes: the original
        root (parentUuid=None) is "root" which survives in msgs_after, so C2 does
        not fire first. Only C1 fires because m2.parentUuid="m1" was present in
        before_uuids but is absent from surviving_uuids.
        """
        from cozempic.safety import PruneValidationError, validate_post_prune  # type: ignore

        root = {"type": "user", "uuid": "root", "parentUuid": None,
                "message": {"role": "user", "content": "root"}}
        m1 = {"type": "assistant", "uuid": "m1", "parentUuid": "root",
              "message": {"role": "assistant",
                          "content": [{"type": "text", "text": "r1"}]}}
        m2 = {"type": "user", "uuid": "m2", "parentUuid": "m1",
              "message": {"role": "user", "content": "q2"}}
        m3 = {"type": "assistant", "uuid": "m3", "parentUuid": "m2",
              "message": {"role": "assistant",
                          "content": [{"type": "text", "text": "r3"}]}}
        ai = {"type": "ai-title", "uuid": "t1", "parentUuid": "m3",
              "title": "Test"}
        lp = {"type": "last-prompt", "uuid": "lp1", "parentUuid": "m3",
              "text": "q2"}
        pm = {"type": "permission-mode", "uuid": "pm1", "parentUuid": "m3",
              "mode": "default"}

        msgs_before = self._pack([root, m1, m2, m3, ai, lp, pm])
        # Drop m1 — m2's parentUuid="m1" is now dangling; m1 WAS in before_uuids.
        # "root" survives → C2 passes. C1 must catch the m1 break.
        msgs_after = self._pack([root, m2, m3, ai, lp, pm])

        with self.assertRaises(PruneValidationError) as ctx:
            validate_post_prune(msgs_before, msgs_after)

        self.assertEqual(ctx.exception.evidence.get("failed_check"), "C1")

    def test_c1_does_not_fire_on_zero_removal_resumed_session(self):
        """L-2: zero-removal prune of a resumed session must NOT raise C1.

        This is Junaid's EXACT named bug: `cozempic treat -rx gentle` on a
        resumed/forked session where the strategy removes nothing → exit 5.

        Setup: msgs_before == msgs_after (zero-removal prune). The session's
        first message has parentUuid pointing to the parent session (external UUID,
        never defined in this file). With before_uuids == surviving_uuids, the
        old `_has_baseline=False` guard incorrectly fired C1. The correct fix
        (unconditional baseline-relative: skip if parent not in before_uuids)
        handles this: external-uuid is not in before_uuids → skip.
        """
        from cozempic.safety import validate_post_prune  # type: ignore

        msgs = [
            # Cross-session anchor: first message references the parent session's
            # last message UUID (not present in this file).
            {"type": "user", "uuid": "u1",
             "parentUuid": "parent-session-uuid-00000000000000000",
             "message": {"role": "user", "content": "hi after resume"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "message": {"role": "assistant",
                         "content": [{"type": "text", "text": "hello"}]}},
            {"type": "ai-title", "uuid": "t1", "parentUuid": "a1",
             "title": "Resumed session"},
            {"type": "last-prompt", "uuid": "lp1", "parentUuid": "a1",
             "text": "hi after resume"},
            {"type": "permission-mode", "uuid": "pm1", "parentUuid": "a1",
             "mode": "default"},
        ]
        packed = self._pack(msgs)
        # Zero-removal prune: msgs_before == msgs_after (gentle strategy found nothing
        # to remove). This is Junaid's exact reported case.
        validate_post_prune(packed, packed)  # must NOT raise — prune introduced no break


class TestSimulateReplayReadiness(unittest.TestCase):
    """simulate_replay_readiness — PR #102 fix: cross-session anchors tolerated.

    After the PR #102 fix, cross-session pointers (parentUuid not defined as any
    uuid in the passed list) are treated as external anchors, NOT chain breaks.
    A resumed/forked session whose first message references the parent session's
    last UUID now returns ok=True.

    The "root" check also updated: any message whose parentUuid is absent from
    the file (None or external UUID) is a valid chain anchor. A resumed session
    with NO parentUuid=None message is valid as long as it has an external anchor.

    Genuinely single-list-detectable failures: empty list, no anchor at all
    (circular graph), no conversational content.
    """

    def _pack(self, raw: list[dict]) -> list[tuple[int, dict, int]]:
        return [(i, m, len(json.dumps(m))) for i, m in enumerate(raw)]

    def test_simulate_replay_ok_on_cross_session_pointer(self):
        """Resumed session with external anchor returns ok=True.

        The first message's parentUuid points outside the file — this is a
        cross-session anchor for a resumed session and must NOT be a chain break.
        """
        from cozempic.safety import simulate_replay_readiness  # type: ignore

        # Pure resumed session: no parentUuid=None (the external-anchor message
        # serves as the chain head). This is Junaid's false-positive scenario.
        messages = self._pack([
            {"type": "user", "uuid": "u1",
             "parentUuid": "parent-session-last-uuid-0000000000",
             "message": {"role": "user", "content": "first resumed turn"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "message": {"role": "assistant",
                         "content": [{"type": "text", "text": "reply"}]}},
            {"type": "user", "uuid": "u2", "parentUuid": "a1",
             "message": {"role": "user", "content": "second"}},
            {"type": "assistant", "uuid": "a2", "parentUuid": "u2",
             "message": {"role": "assistant",
                         "content": [{"type": "text", "text": "r2"}]}},
        ])
        ok, reason = simulate_replay_readiness(messages)
        self.assertTrue(
            ok,
            msg=(
                f"Resumed session with external anchor must return ok=True. "
                f"Got ok={ok!r}, reason={reason!r}. "
                f"Cross-session pointers are NOT chain breaks (Junaid's named false positive)."
            ),
        )
        self.assertEqual(reason, "")

    def test_simulate_replay_ok_on_clean_session_with_null_root(self):
        """Standard session with parentUuid=None root returns ok=True."""
        from cozempic.safety import simulate_replay_readiness  # type: ignore

        messages = self._pack([
            {"type": "user", "uuid": "u1", "parentUuid": None,
             "message": {"role": "user", "content": "first"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "message": {"role": "assistant",
                         "content": [{"type": "text", "text": "r1"}]}},
        ])
        ok, reason = simulate_replay_readiness(messages)
        self.assertTrue(ok, f"Clean session should return ok=True. Got: {reason!r}")
        self.assertEqual(reason, "")

    def test_simulate_replay_fails_on_no_conversation(self):
        """A list with no user/assistant content returns ok=False.
        This is genuinely detectable without a before/after split.
        """
        from cozempic.safety import simulate_replay_readiness  # type: ignore

        messages = self._pack([
            {"type": "ai-title", "uuid": "t1", "parentUuid": None, "title": "x"},
        ])
        ok, reason = simulate_replay_readiness(messages)
        self.assertFalse(ok, "Metadata-only list must return ok=False.")
        self.assertIn("conversation", reason.lower())

    def test_simulate_replay_fails_on_empty_list(self):
        """Empty message list returns ok=False."""
        from cozempic.safety import simulate_replay_readiness  # type: ignore

        ok, reason = simulate_replay_readiness([])
        self.assertFalse(ok)
        self.assertIn("empty", reason.lower())


if __name__ == "__main__":
    unittest.main()
