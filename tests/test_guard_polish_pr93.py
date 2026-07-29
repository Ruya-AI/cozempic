"""RED tests for PR #93 (Junaid PR #92 followups items 2-5).

Items covered:
  #2 — Dead `_spawn_locks` dict + dead consumer branch in
       `_is_guard_running_for_session` must be removed.
  #3 — Session pidfile must be unlinked on ALL daemon-exit paths
       (SIGTERM, K=10, KeyboardInterrupt, every `break` in main loop).
       Class-of-bug fold: also covers pre-existing TODO:55 leak in
       `_graceful_shutdown` (PR #92 added the K=10 leak alongside it).
  #4 — At K=10, when `agents_active=True`, the daemon must DEFER exit
       (keep cycling at backoff cap) rather than `sys.exit(0)`. A hard
       cap (default K=50, configurable via COZEMPIC_GUARD_HARD_EXIT_K)
       ensures eventual exit even if agents perma-run.
  #5 — `DaemonSpawnClaim` + daemon hand-off must write the same 3-line
       payload as `_ReloadLock` (pid + iso-timestamp + initiator) for
       operator-triage parity. Helper `_parse_pidfile_pid` tolerates
       both legacy 1-line and new 3-line formats. Bash hook reads with
       `head -1`. Hook schema bumps v8 → v9.

These tests are RED on HEAD `4195228` (architect doc commit, no impl
changes). They flip GREEN incrementally across commits 2, 3, 4.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch


# ─── Item #2 — Dead _spawn_locks dict removed ────────────────────────────────


class TestPolishPR93_SpawnLocksDictRemoved(unittest.TestCase):
    """The `_spawn_locks` dict + `_spawn_locks_mu` lock in guard.py are
    consumer-only (no producer exists in the tree). PR #92 replaced the
    in-process fast-path with `DaemonSpawnClaim` (cross-process O_CREAT|
    O_EXCL). The dict must be removed to eliminate dead code that the
    consumer branch in `_is_guard_running_for_session` keys on.
    """

    def test_spawn_locks_dict_attribute_gone(self):
        from cozempic import guard
        self.assertFalse(
            hasattr(guard, "_spawn_locks"),
            "`_spawn_locks` dict must be removed (dead code — no producer "
            "exists). The cross-process kernel claim in spawn_lock.py "
            "subsumes the in-process fast-path entirely.",
        )

    def test_spawn_locks_mu_attribute_gone(self):
        from cozempic import guard
        self.assertFalse(
            hasattr(guard, "_spawn_locks_mu"),
            "`_spawn_locks_mu` must be removed alongside `_spawn_locks` — "
            "the lock guards an empty dict.",
        )


class TestPolishPR93_PlaceholderPidProbe(unittest.TestCase):
    """Regression test: a placeholder pidfile resolves to None without
    pre-claim cleanup. DaemonSpawnClaim owns serialized stale recovery."""

    SESSION_ID = "pr93pl00-0000-0000-0000-000000000093"

    def setUp(self):
        from cozempic.guard import _pid_file_for_session
        self.pid_path = _pid_file_for_session(self.SESSION_ID)
        self.pid_path.unlink(missing_ok=True)

    def tearDown(self):
        self.pid_path.unlink(missing_ok=True)

    def test_zero_pid_resolves_to_none_without_unlinking(self):
        from cozempic.guard import _is_guard_running_for_session
        self.pid_path.write_text("0\n")
        result = _is_guard_running_for_session(self.SESSION_ID)
        self.assertIsNone(result, "placeholder pid 0 must resolve to None")
        self.assertTrue(self.pid_path.exists(), "probe must not unlink pidfile")

    def test_negative_pid_resolves_to_none_without_unlinking(self):
        from cozempic.guard import _is_guard_running_for_session
        self.pid_path.write_text("-1\n")
        result = _is_guard_running_for_session(self.SESSION_ID)
        self.assertIsNone(result, "negative pid must resolve to None")
        self.assertTrue(self.pid_path.exists(), "probe must not unlink pidfile")


# ─── Item #3 — Pidfile unlinked on ALL daemon-exit paths ────────────────────


class TestPolishPR93_SafeUnlinkHelper(unittest.TestCase):
    """`_safe_unlink_session_pidfile(session_id)` is the shared helper
    used by all daemon-exit paths. It MUST:
      - Skip on falsy/empty session_id (no-op).
      - Swallow ValueError (malformed session_id) and OSError.
      - Use CAS via `_pid_file_points_to(session_id, os.getpid())` so it
        doesn't destroy a peer's just-completed claim during reload.
    """

    SESSION_ID = "pr93un00-0000-0000-0000-000000000093"

    def setUp(self):
        from cozempic.guard import _pid_file_for_session
        self.pid_path = _pid_file_for_session(self.SESSION_ID)
        self.pid_path.unlink(missing_ok=True)

    def tearDown(self):
        self.pid_path.unlink(missing_ok=True)

    def test_helper_exists(self):
        from cozempic import guard
        self.assertTrue(
            hasattr(guard, "_safe_unlink_session_pidfile"),
            "guard._safe_unlink_session_pidfile helper must exist",
        )

    def test_helper_unlinks_when_pid_matches(self):
        from cozempic.guard import _safe_unlink_session_pidfile
        # Write our own PID — CAS gate should pass.
        self.pid_path.write_text(f"{os.getpid()}\n")
        _safe_unlink_session_pidfile(self.SESSION_ID)
        self.assertFalse(self.pid_path.exists())

    def test_helper_skips_when_pid_mismatch(self):
        """CAS gate: if pidfile contains another PID (peer's fresh claim),
        we must NOT unlink — that would destroy the peer's claim."""
        from cozempic.guard import _safe_unlink_session_pidfile
        # Write a foreign PID (not ours) — CAS gate should skip the unlink.
        foreign_pid = os.getpid() + 12345
        self.pid_path.write_text(f"{foreign_pid}\n")
        _safe_unlink_session_pidfile(self.SESSION_ID)
        self.assertTrue(
            self.pid_path.exists(),
            "helper must NOT unlink when pidfile contains a foreign PID "
            "(would destroy a peer's fresh claim)",
        )

    def test_helper_swallows_invalid_session_id(self):
        from cozempic.guard import _safe_unlink_session_pidfile
        # Must not raise on malformed session_id (shutdown path — worst
        # possible time for an exception).
        try:
            _safe_unlink_session_pidfile("not-a-uuid")
        except Exception as exc:
            self.fail(f"helper must swallow invalid session_id, raised {exc!r}")

    def test_helper_swallows_none_and_empty(self):
        from cozempic.guard import _safe_unlink_session_pidfile
        # Falsy inputs must no-op.
        _safe_unlink_session_pidfile(None)
        _safe_unlink_session_pidfile("")

    def test_helper_handles_missing_pidfile(self):
        from cozempic.guard import _safe_unlink_session_pidfile
        # Pidfile already gone → must not raise.
        self.pid_path.unlink(missing_ok=True)
        _safe_unlink_session_pidfile(self.SESSION_ID)


class TestPolishPR93_PidfileUnlinkedOnExit(unittest.TestCase):
    """Drive start_guard through each exit path and assert the session
    pidfile is unlinked via the finally-block helper."""

    SESSION_ID = "pr93ex00-0000-0000-0000-000000000093"

    def setUp(self):
        from cozempic.guard import _pid_file_for_session
        self.pid_path = _pid_file_for_session(self.SESSION_ID)
        self.pid_path.unlink(missing_ok=True)
        # Write OUR pid so the CAS gate in the helper accepts it.
        self.pid_path.write_text(f"{os.getpid()}\n")

        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmpdir, ignore_errors=True))
        self.session_path = Path(self.tmpdir) / "fake_session.jsonl"
        self.session_path.write_text('{"type":"user","message":{"content":"hi"}}\n')
        self.fake_session = {"session_id": self.SESSION_ID, "path": self.session_path}

    def tearDown(self):
        self.pid_path.unlink(missing_ok=True)

    def _drive_start_guard_until_exit(self, prune_returns_zero=True, agents_active=False):
        """Helper: drive start_guard and trigger K=10 exit. Returns whether
        SystemExit was raised."""
        from cozempic import guard as guard_mod

        class _FakeState:
            def __init__(self, has_agents):
                self.has_agents = has_agents
                self.subagents = []
                if has_agents:
                    sub = type("S", (), {"status": "running"})()
                    self.subagents = [sub]
                self.tasks = []
                self.message_count = 0

            def is_empty(self):
                return not self.has_agents

        fake_state = _FakeState(agents_active)

        def fake_prune_cycle(**kwargs):
            return {
                "saved_mb": 0.0 if prune_returns_zero else 10.0,
                "original_tokens": 600_000,
                "final_tokens": 600_000,
                "team_name": None,
                "team_messages": 0,
                "checkpoint_path": None,
                "backup_path": None,
                "reloading": False,
            }

        sleep_count = {"n": 0}

        def fake_sleep(_):
            sleep_count["n"] += 1
            if sleep_count["n"] > 200:
                raise RuntimeError("test sleep limit exceeded — loop did not exit")

        with (
            patch.object(guard_mod.time, "sleep", side_effect=fake_sleep),
            patch.object(guard_mod, "_resolve_session_by_id", return_value=self.fake_session),
            patch.object(guard_mod, "find_current_session", return_value=self.fake_session),
            patch.object(guard_mod, "find_claude_pid", return_value=None),
            patch.object(guard_mod, "checkpoint_team", return_value=fake_state),
            patch.object(guard_mod, "guard_prune_cycle", side_effect=fake_prune_cycle),
            patch.object(guard_mod, "quick_token_estimate", return_value=600_000),
            patch.object(guard_mod, "load_messages", return_value=[]),
            patch("cozempic.session.record_session"),
            patch.object(guard_mod, "_cleanup_stale_watchers"),
            patch.object(guard_mod, "ping_install_if_new"),
            patch.object(guard_mod, "maybe_auto_update"),
            patch.object(guard_mod, "cleanup_old_backups"),
            patch("cozempic.tokens.detect_context_window", return_value=1_000_000),
        ):
            try:
                guard_mod.start_guard(
                    cwd=self.tmpdir,
                    threshold_mb=100.0,
                    soft_threshold_mb=50.0,
                    rx_name="standard",
                    interval=30,
                    auto_reload=False,
                    reactive=False,
                    threshold_tokens=500_000,
                    soft_threshold_tokens=250_000,
                    session_id=self.SESSION_ID,
                )
                return False  # No exit raised
            except SystemExit:
                return True

    def test_k10_exit_unlinks_pidfile(self):
        """K=10 voluntary exit path must unlink the session pidfile via the
        finally-block helper."""
        exited = self._drive_start_guard_until_exit(prune_returns_zero=True, agents_active=False)
        self.assertTrue(exited, "K=10 should have triggered SystemExit")
        self.assertFalse(
            self.pid_path.exists(),
            "pidfile must be unlinked after K=10 SystemExit "
            "(architect spec: finally block in start_guard wraps "
            "_safe_unlink_session_pidfile)",
        )


# ─── Item #4 — agents_active K=10 deferral + hard cap ───────────────────────


class TestPolishPR93_K10DeferWhenAgentsActive(unittest.TestCase):
    """When K reaches HARD_LOOP_EXIT_THRESHOLD (=10) but agents are still
    running, the daemon MUST defer exit. A hard cap at
    HARD_LOOP_HARD_EXIT_THRESHOLD (default 50, configurable via
    COZEMPIC_GUARD_HARD_EXIT_K) ensures eventual exit.
    """

    SESSION_ID = "pr93k10a-0000-0000-0000-000000000093"

    def setUp(self):
        from cozempic.guard import _pid_file_for_session
        self.pid_path = _pid_file_for_session(self.SESSION_ID)
        self.pid_path.unlink(missing_ok=True)
        self.pid_path.write_text(f"{os.getpid()}\n")

        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmpdir, ignore_errors=True))
        self.session_path = Path(self.tmpdir) / "fake_session.jsonl"
        self.session_path.write_text('{"type":"user","message":{"content":"hi"}}\n')
        self.fake_session = {"session_id": self.SESSION_ID, "path": self.session_path}

    def tearDown(self):
        self.pid_path.unlink(missing_ok=True)

    def test_hard_exit_threshold_constant_exists(self):
        from cozempic import guard
        self.assertTrue(
            hasattr(guard, "HARD_LOOP_HARD_EXIT_THRESHOLD"),
            "guard.HARD_LOOP_HARD_EXIT_THRESHOLD constant must exist "
            "(architectural defer + hard cap)",
        )
        # Default is 50 per architect spec.
        self.assertEqual(guard.HARD_LOOP_HARD_EXIT_THRESHOLD, 50)

    def _run_loop(self, agents_active=True, max_cycles=80, teammate_status=None):
        """Run start_guard with prune=0 forever and the given agents_active
        state. Returns (exited, cycle_count).

        teammate_status: if not None, adds a single teammate with that status
        string and EMPTY subagents list, so the test exercises the teammate
        path exclusively.
        """
        from cozempic import guard as guard_mod

        class _FakeState:
            def __init__(self, has_agents, tm_status):
                self.has_agents = has_agents
                sub = type("S", (), {"status": "running"})()
                self.subagents = [sub] if has_agents else []
                # Inline teammate stub — matches the subset of TeammateInfo that
                # _compute_agents_active reads (only .status).
                tm = type("T", (), {"status": tm_status})() if tm_status is not None else None
                self.teammates = [tm] if tm else []
                self.tasks = []
                self.message_count = 0

            def is_empty(self):
                return not self.has_agents and not self.teammates

        fake_state = _FakeState(agents_active, teammate_status)

        def fake_prune_cycle(**kwargs):
            return {
                "saved_mb": 0.0,
                "original_tokens": 600_000,
                "final_tokens": 600_000,
                "team_name": None,
                "team_messages": 0,
                "checkpoint_path": None,
                "backup_path": None,
                "reloading": False,
            }

        sleep_calls = {"n": 0}

        def fake_sleep(_):
            sleep_calls["n"] += 1
            if sleep_calls["n"] >= max_cycles:
                raise RuntimeError(f"max_cycles={max_cycles} reached")

        exited = False
        with (
            patch.object(guard_mod.time, "sleep", side_effect=fake_sleep),
            patch.object(guard_mod, "_resolve_session_by_id", return_value=self.fake_session),
            patch.object(guard_mod, "find_current_session", return_value=self.fake_session),
            patch.object(guard_mod, "find_claude_pid", return_value=None),
            patch.object(guard_mod, "checkpoint_team", return_value=fake_state),
            patch.object(guard_mod, "guard_prune_cycle", side_effect=fake_prune_cycle),
            patch.object(guard_mod, "quick_token_estimate", return_value=600_000),
            patch.object(guard_mod, "load_messages", return_value=[]),
            patch("cozempic.session.record_session"),
            patch.object(guard_mod, "_cleanup_stale_watchers"),
            patch.object(guard_mod, "ping_install_if_new"),
            patch.object(guard_mod, "maybe_auto_update"),
            patch.object(guard_mod, "cleanup_old_backups"),
            patch("cozempic.tokens.detect_context_window", return_value=1_000_000),
        ):
            try:
                guard_mod.start_guard(
                    cwd=self.tmpdir,
                    threshold_mb=100.0,
                    soft_threshold_mb=50.0,
                    rx_name="standard",
                    interval=30,
                    auto_reload=False,
                    reactive=False,
                    threshold_tokens=500_000,
                    soft_threshold_tokens=250_000,
                    session_id=self.SESSION_ID,
                )
            except SystemExit:
                exited = True
            except RuntimeError:
                exited = False
        return exited, sleep_calls["n"]

    def test_k10_exits_when_no_agents(self):
        """Regression: with no agents active, K=10 exit still fires (current
        v1.8.14 behaviour preserved)."""
        exited, _ = self._run_loop(agents_active=False, max_cycles=80)
        self.assertTrue(
            exited,
            "K=10 should trigger SystemExit when no agents are active "
            "(current v1.8.14 behaviour)",
        )

    def test_k10_defers_when_subagents_running(self):
        """With agents_active=True at K=10, the daemon must NOT exit. The
        loop should keep cycling at backoff cap until either agents quiesce
        or the hard cap is reached."""
        # 40 cycles is well past K=10 but well below the hard cap of 50.
        exited, n = self._run_loop(agents_active=True, max_cycles=40)
        self.assertFalse(
            exited,
            f"K=10 must NOT exit when agents_active=True (deferred). "
            f"Got exited=True after {n} cycles.",
        )

    def test_k10_defers_when_teammate_running(self):
        """REGRESSION GUARD (PR #131 / L8) — K-exit wiring: a running teammate
        must cause the daemon to DEFER at K=10, exactly like a running subagent.

        Exercises the full wiring path: state.teammates[0].status="running"
        → _compute_agents_active(state) → True → daemon defers sys.exit at K=10.
        Behavioural RED at base: inline `any(s.status ... for s in state.subagents)`
        exits at K=10 (subagents=[], teammate ignored).
        """
        exited, n = self._run_loop(
            agents_active=False, teammate_status="running", max_cycles=40
        )
        self.assertFalse(
            exited,
            f"K=10 must NOT exit when a teammate is running (status='running'). "
            f"Got exited=True after {n} cycles — daemon ignored the teammate. "
            f"Check _compute_agents_active covers state.teammates.",
        )

    def test_hard_cap_exits_with_agents_active(self):
        """Even with agents perma-running, the hard cap must fire eventually.
        Patch the constant directly rather than env-var-reloading the module
        so we don't pollute other tests with a leftover overridden value."""
        import cozempic.guard as guard_mod
        with patch.object(guard_mod, "HARD_LOOP_HARD_EXIT_THRESHOLD", 15):
            self.assertEqual(guard_mod.HARD_LOOP_HARD_EXIT_THRESHOLD, 15)
            exited, n = self._run_loop(agents_active=True, max_cycles=40)
            self.assertTrue(
                exited,
                f"Hard cap K=15 should fire even with agents_active=True. "
                f"Got exited=False after {n} cycles.",
            )
        # After patch exits, constant is restored to module default (50).
        self.assertEqual(guard_mod.HARD_LOOP_HARD_EXIT_THRESHOLD, 50)

    def test_env_var_overrides_hard_cap(self):
        """Module-import env var COZEMPIC_GUARD_HARD_EXIT_K is read by
        _read_hard_exit_threshold and clamped. Test the helper directly
        (not via importlib.reload which is hard to clean up reliably)."""
        import cozempic.guard as guard_mod
        with patch.dict(os.environ, {"COZEMPIC_GUARD_HARD_EXIT_K": "25"}):
            self.assertEqual(guard_mod._read_hard_exit_threshold(), 25)

    def test_env_var_invalid_falls_back_to_default(self):
        import cozempic.guard as guard_mod
        for bad in ("not-a-number", "0", "-5", "10", "9", "10000"):
            with self.subTest(value=bad):
                with patch.dict(os.environ, {"COZEMPIC_GUARD_HARD_EXIT_K": bad}):
                    self.assertEqual(
                        guard_mod._read_hard_exit_threshold(), 50,
                        f"invalid value {bad!r} must fall back to default 50",
                    )


# ─── Item #5 — DaemonSpawnClaim metadata parity with _ReloadLock ────────────


class TestPolishPR93_SpawnClaimMetadataParity(unittest.TestCase):
    """`DaemonSpawnClaim._claim` and the daemon hand-off in
    `start_guard_daemon` must write a 3-line payload (pid + iso-timestamp
    + initiator) matching `reload_lock._ReloadLock`'s convention."""

    SESSION_ID = "pr93md00-0000-0000-0000-000000000093"

    def setUp(self):
        from cozempic.guard import _pid_file_for_session
        self.pid_path = _pid_file_for_session(self.SESSION_ID)
        self.pid_path.unlink(missing_ok=True)

    def tearDown(self):
        self.pid_path.unlink(missing_ok=True)
        self.pid_path.with_suffix(".pid.tmp").unlink(missing_ok=True)

    def test_init_constants_exist(self):
        from cozempic import spawn_lock
        self.assertTrue(
            hasattr(spawn_lock, "INIT_SPAWN_PARENT"),
            "spawn_lock.INIT_SPAWN_PARENT constant must exist",
        )
        self.assertTrue(
            hasattr(spawn_lock, "INIT_SPAWN_DAEMON"),
            "spawn_lock.INIT_SPAWN_DAEMON constant must exist",
        )
        self.assertEqual(spawn_lock.INIT_SPAWN_PARENT, "spawn-claim-parent")
        self.assertEqual(spawn_lock.INIT_SPAWN_DAEMON, "spawn-claim-daemon")

    def test_spawn_claim_writes_3_line_payload(self):
        """DaemonSpawnClaim writes pid + timestamp + 'spawn-claim-parent'."""
        from cozempic.spawn_lock import DaemonSpawnClaim, INIT_SPAWN_PARENT

        claim = DaemonSpawnClaim(self.SESSION_ID, self.pid_path)
        try:
            claim.__enter__()
            content = self.pid_path.read_text()
            lines = content.strip().split("\n")
            self.assertEqual(
                len(lines), 3,
                f"pidfile must be 3 lines (pid, timestamp, initiator); got {lines!r}",
            )
            self.assertEqual(int(lines[0]), os.getpid())
            # Line 2 must parse as ISO timestamp.
            try:
                datetime.fromisoformat(lines[1])
            except ValueError:
                self.fail(f"line 2 must be ISO timestamp; got {lines[1]!r}")
            self.assertEqual(lines[2], INIT_SPAWN_PARENT)
        finally:
            claim.__exit__(None, None, None)

    def test_parse_pidfile_handles_legacy_single_line(self):
        from cozempic.spawn_lock import _parse_pidfile_pid
        self.pid_path.write_text("12345\n")
        self.assertEqual(_parse_pidfile_pid(self.pid_path), 12345)

    def test_parse_pidfile_handles_new_3_line(self):
        from cozempic.spawn_lock import _parse_pidfile_pid
        self.pid_path.write_text("12345\n2026-05-19T10:00:00\nspawn-claim-parent\n")
        self.assertEqual(_parse_pidfile_pid(self.pid_path), 12345)

    def test_parse_pidfile_handles_garbage(self):
        from cozempic.spawn_lock import _parse_pidfile_pid
        self.pid_path.write_text("not-a-number\nfoo\n")
        self.assertEqual(_parse_pidfile_pid(self.pid_path), 0)

    def test_parse_pidfile_handles_empty(self):
        from cozempic.spawn_lock import _parse_pidfile_pid
        self.pid_path.write_text("")
        self.assertEqual(_parse_pidfile_pid(self.pid_path), 0)

    def test_parse_pidfile_handles_missing(self):
        from cozempic.spawn_lock import _parse_pidfile_pid
        self.pid_path.unlink(missing_ok=True)
        self.assertEqual(_parse_pidfile_pid(self.pid_path), 0)

    def test_is_guard_running_uses_tolerant_parser(self):
        """Write a 3-line pidfile with the test runner's PID and force the
        guard-identity gate to True. Confirm `_is_guard_running_for_session`
        returns the PID — proving the parser extracted line 1 correctly.

        Under the OLD `int(read_text().strip())` parser, the multi-line
        content stripped is "PID\n2026...\nspawn-claim-parent" which
        raises ValueError → result is None → assertion fails. Under the
        NEW `_parse_pidfile_pid`, line 1 is extracted as PID → result is
        the live PID."""
        from cozempic.guard import _is_guard_running_for_session
        my_pid = os.getpid()
        self.pid_path.write_text(
            f"{my_pid}\n2026-05-19T10:00:00\nspawn-claim-parent\n"
        )
        # Force the guard-identity gate to True so we exercise the parse
        # path through to a positive return rather than the fresh-window
        # fallback at line 1314.
        with patch("cozempic.guard._is_cozempic_guard_process", return_value=True):
            result = _is_guard_running_for_session(self.SESSION_ID)
        self.assertEqual(
            result, my_pid,
            f"_is_guard_running_for_session must use a tolerant parser "
            f"(first-line only) on 3-line pidfile. Got {result!r}, "
            f"expected {my_pid} (live PID).",
        )


class TestPolishPR93_HookSchemaV9(unittest.TestCase):
    """Hook schema bumps v8 → v9 because the bash hook's `cat | kill -0`
    pattern would break on the new 3-line pidfile format. Migrated to
    `head -1` which is POSIX and extracts only line 1 (the PID)."""

    def test_hook_schema_version_v9(self):
        from cozempic.init import HOOK_SCHEMA_VERSION
        # Durable floor check (was a hardcoded allow-list that broke on every bump).
        self.assertEqual(HOOK_SCHEMA_VERSION[0], "v"[0])
        self.assertGreaterEqual(
            int(HOOK_SCHEMA_VERSION.lstrip("v")), 9,
            "HOOK_SCHEMA_VERSION must be v9 or higher (PR #93 head -1 change, PR #94 Phase B bump, #109 NO_AUTO_INIT)",
        )

    def test_hooks_json_uses_head_minus_1(self):
        """The bash hook MUST use `head -n 1` (or `head -1`) to read the PID,
        not `cat`. With the new 3-line pidfile, `cat` returns multiple
        whitespace-separated tokens to `kill -0`, which is undefined."""
        for hooks_rel in (
            "plugin/hooks/hooks.json",
            "src/cozempic/data/hooks.json",
        ):
            hooks_path = Path(__file__).parent.parent / hooks_rel
            with self.subTest(path=hooks_rel):
                hooks = json.loads(hooks_path.read_text())
                ss_cmd = hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]
                # Must use head, not cat, for the GUARD_PID_FILE liveness probe.
                self.assertIn(
                    "head -n 1 \"$GUARD_PID_FILE\"",
                    ss_cmd,
                    f"{hooks_rel}: bash hook must use `head -n 1` to read "
                    f"the PID from GUARD_PID_FILE (3-line format compat)",
                )
                # Must NOT still use the old cat pattern for GUARD_PID_FILE.
                self.assertNotIn(
                    "cat \"$GUARD_PID_FILE\"",
                    ss_cmd,
                    f"{hooks_rel}: bash hook must not use `cat` "
                    f"on GUARD_PID_FILE (breaks on 3-line pidfile)",
                )

    def test_hooks_json_marker_v9(self):
        for hooks_rel in (
            "plugin/hooks/hooks.json",
            "src/cozempic/data/hooks.json",
        ):
            hooks_path = Path(__file__).parent.parent / hooks_rel
            with self.subTest(path=hooks_rel):
                body = hooks_path.read_text()
                from cozempic.init import HOOK_SCHEMA_MARKER
                # Track the current marker (was a hardcoded version list).
                self.assertIn(
                    HOOK_SCHEMA_MARKER, body,
                    f"{hooks_rel}: must carry the current schema marker {HOOK_SCHEMA_MARKER}",
                )
                self.assertNotIn(
                    "cozempic-hook-schema=v8",
                    body,
                    f"{hooks_rel}: v8 marker must be migrated to v9+",
                )


# ─── DA round 1 folded items (N3, M1, M2 — folded into commit 3) ────────────


class TestPolishPR93_FreshWindowDocstring(unittest.TestCase):
    """N3: `_read_fresh_window_seconds` docstring must clarify that the
    env var COZEMPIC_PIDFILE_FRESH_SECONDS is read at module import time
    and requires a daemon restart (not just env-var update) to take effect.
    """

    def test_docstring_mentions_import_time(self):
        from cozempic.spawn_lock import _read_fresh_window_seconds
        doc = _read_fresh_window_seconds.__doc__ or ""
        self.assertTrue(
            "import" in doc.lower() or "restart" in doc.lower(),
            "docstring must clarify env var is read at import / requires restart",
        )


class TestPolishPR93_PidfileFsync(unittest.TestCase):
    """M1: After `os.rename(.pid.tmp, .pid)`, the parent directory must
    be fsynced for durability. Best verified by inspecting the source for
    the explicit fsync call near the rename."""

    def test_start_guard_daemon_fsyncs_parent_after_rename(self):
        import inspect
        from cozempic.guard import start_guard_daemon
        src = inspect.getsource(start_guard_daemon)
        # The hand-off block uses os.rename(tmp_path, pid_path). After
        # that rename we must fsync the parent dir (or the .pid fd) to
        # make the rename durable across an abrupt power loss.
        self.assertTrue(
            "fsync" in src,
            "start_guard_daemon must call os.fsync on the parent dir "
            "after the .pid.tmp → .pid rename (durability)",
        )


class TestPolishPR93_PidfileEACCES(unittest.TestCase):
    """M2: `DaemonSpawnClaim._is_pidfile_fresh` must return True on EACCES
    (PermissionError from stat) — a conservative-fresh classification.
    A pidfile we can't stat could still be a live peer claim; returning
    False would let us treat it as stale and re-claim, racing the peer."""

    def test_is_pidfile_fresh_returns_true_on_permission_error(self):
        from cozempic.spawn_lock import DaemonSpawnClaim
        from unittest.mock import MagicMock, patch as _patch

        claim = DaemonSpawnClaim("any-session-12345", Path("/tmp/x.pid"))

        with _patch("cozempic.spawn_lock.os.stat", side_effect=PermissionError("EACCES")):
            result = claim._is_pidfile_fresh()
        self.assertTrue(
            result,
            "EACCES on stat must classify pidfile as fresh (conservative — "
            "DA round 1 M2 recommendation)",
        )


if __name__ == "__main__":
    unittest.main()
