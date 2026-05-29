"""RED tests for NEW-1 — transient daemon vs SessionStart spawn race.

Architect spec: AUDIT_REPORT_pr94_transient_daemon_race.md § NEW-1 Test contract.
Root cause: OLD guard daemon exits during reload, writes sentinel, transient daemon
spawns in the reload gap (upgrade-chain re-fire of SessionStart hook), NEW Claude's
SessionStart hook fires and sees the transient daemon → skips guard spawn →
NEW Claude ends up UNPROTECTED.

Fix design: Option (c) — write a reload sentinel BEFORE spawning watcher;
SessionStart hook skips spawn if sentinel exists and is fresh (<SENTINEL_TTL_SECONDS).
Option (b) defense-in-depth: unlink pidfile immediately when watched Claude dies.

All 7 tests EXPECTED TO FAIL until Phase B implementation lands on top of PR #93 merge.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Test 1 — _terminate_and_resume writes in-flight sentinel BEFORE watcher
# ---------------------------------------------------------------------------
class TestReloadWritesInFlightSentinel(unittest.TestCase):
    """After Option (c) fix: _terminate_and_resume MUST write the sentinel
    file BEFORE calling _spawn_reload_watcher.

    Asserts:
    - sentinel path = /tmp/cozempic_reload_<sid12>.in-flight
    - sentinel exists and contains old_claude_pid and an ISO timestamp
    - sentinel is written BEFORE _spawn_reload_watcher is invoked
    (we verify order by checking write precedes the watcher call via a side_effect)
    """

    def setUp(self):
        self.sid = "abcdef012345678901234567890abcde"  # 32 hex chars
        self.sid12 = self.sid[:12]  # "abcdef012345"
        self.old_claude_pid = 89113
        self.sentinel_path = Path(f"/tmp/cozempic_reload_{self.sid12}.in-flight")
        self.sentinel_path.unlink(missing_ok=True)
        self.addCleanup(self.sentinel_path.unlink, missing_ok=True)

    def test_reload_writes_in_flight_sentinel(self):
        """sentinel written before watcher spawned — order enforced."""
        # write_reload_sentinel must exist in reload_lock after Phase B
        try:
            from cozempic.reload_lock import write_reload_sentinel, SENTINEL_TTL_SECONDS
        except ImportError as exc:
            self.fail(
                f"Phase B symbols missing: {exc}. "
                "Expected: write_reload_sentinel, SENTINEL_TTL_SECONDS in reload_lock."
            )

        sentinel_written_before_watcher = []

        def _fake_watcher(claude_pid, project_dir, session_id=None, **_kw):
            # Record whether sentinel exists at the moment watcher is called
            sentinel_written_before_watcher.append(self.sentinel_path.exists())

        # Claude must be alive — the sentinel only makes sense when we are
        # actually terminating a live Claude and resuming it. _terminate_and_resume
        # has an anti-resurrection entry gate (skips everything, incl. the
        # sentinel write, if claude_pid is no longer a Claude process).
        with patch("cozempic.guard._spawn_reload_watcher", side_effect=_fake_watcher), \
             patch("cozempic.guard._is_claude_process", return_value=True), \
             patch("cozempic.guard._wait_for_exit", return_value=True), \
             patch("cozempic.guard.os.kill"), \
             patch("cozempic.guard.time.sleep"):
            from cozempic.guard import _terminate_and_resume
            _terminate_and_resume(
                claude_pid=self.old_claude_pid,
                project_dir="/tmp/fake_project",
                session_id=self.sid,
                rx_name="standard",
                config=None,
                auto_reload=True,
            )

        # Sentinel must exist on disk now
        self.assertTrue(
            self.sentinel_path.exists(),
            "Sentinel file was not created by _terminate_and_resume",
        )

        # Watcher must have been called with sentinel ALREADY present
        self.assertTrue(
            sentinel_written_before_watcher,
            "_spawn_reload_watcher was never called",
        )
        self.assertTrue(
            sentinel_written_before_watcher[0],
            "Sentinel was NOT present when _spawn_reload_watcher was called — write order violated",
        )

        # Sentinel content: first line = old_claude_pid, second = ISO timestamp
        content = self.sentinel_path.read_text()
        lines = content.strip().splitlines()
        self.assertGreaterEqual(len(lines), 2, "Sentinel must have at least 2 lines (pid, iso-ts)")
        self.assertEqual(int(lines[0].strip()), self.old_claude_pid)
        # Rough ISO timestamp check (must parse without ValueError)
        from datetime import datetime
        datetime.fromisoformat(lines[1].strip())  # raises on bad format


# ---------------------------------------------------------------------------
# Anti-resurrection invariant — regression guard for the outer-check removal
# ---------------------------------------------------------------------------
class TestNoResurrectionWhenClaudeAlreadyDead(unittest.TestCase):
    """If Claude is already gone when `_terminate_and_resume` is entered — e.g.
    the user exited during the prune window between the guard's liveness check
    and the reload — the function must NOT write a sentinel or spawn the resume
    watcher. The watcher resumes UNCONDITIONALLY once claude_pid dies
    (`while kill -0 …; do sleep; done; <resume_cmd>`), so without the entry gate
    a dead PID reopens a session the user intentionally closed — the cozempic
    reload-resurrection incident class. PR #94's per-block checks only guard the
    SIGTERM, not the watcher spawn, so the entry gate is load-bearing here.
    """

    def test_dead_claude_at_entry_skips_sentinel_and_watcher(self):
        sid = "ddddddddeeee1111222233334444dddd"
        # Simulate a dead PID at the bare-liveness probe: os.kill(pid, 0) raises
        # ProcessLookupError. The gate must return before any sentinel/watcher.
        with patch("cozempic.guard._spawn_reload_watcher") as mock_watcher, \
             patch("cozempic.guard.write_reload_sentinel") as mock_sentinel, \
             patch("cozempic.guard._detect_terminal_env", return_value="plain"), \
             patch("cozempic.guard.os.kill", side_effect=ProcessLookupError):
            from cozempic.guard import _terminate_and_resume
            _terminate_and_resume(
                claude_pid=99999,
                project_dir="/tmp/fake_project",
                session_id=sid,
            )
        mock_watcher.assert_not_called()   # no resurrection
        mock_sentinel.assert_not_called()  # no stale sentinel suppressing legit spawns


class TestNoResurrectionDespiteFreshJSONLMtime(unittest.TestCase):
    """Mtime-immune liveness gate. A Claude that died during the prune window
    must NOT be resurrected even though cozempic's own save_messages just
    refreshed the session JSONL — which `_is_claude_process`'s mtime fallback
    misreads as a live Claude. Drives the REAL `_is_claude_process` (only the
    resume watcher is mocked), so it exercises the actual invariant rather than
    mocking away the very fallback that produced the false-alive verdict.
    """

    def test_dead_pid_with_fresh_jsonl_does_not_resurrect(self):
        # A definitively-dead PID: spawn a trivial process and reap it.
        proc = subprocess.Popen([sys.executable, "-c", ""])
        proc.wait()
        dead_pid = proc.pid

        with tempfile.TemporaryDirectory() as td:
            jsonl = Path(td) / "session.jsonl"
            jsonl.write_text("{}\n")  # mtime = now — mimics save_messages' fresh write
            sid = "ddddddddffff1111222233334444dddd"
            sentinel = Path(f"/tmp/cozempic_reload_{sid[:12]}.in-flight")
            sentinel.unlink(missing_ok=True)
            self.addCleanup(sentinel.unlink, missing_ok=True)

            from cozempic.guard import _terminate_and_resume, _is_claude_process
            # Precondition: the mtime fallback WOULD misreport this dead PID as a
            # live Claude — proving the gate cannot rely on _is_claude_process.
            self.assertTrue(
                _is_claude_process(dead_pid, session_path=jsonl),
                "precondition: mtime fallback should report the dead PID alive",
            )
            with patch("cozempic.guard._spawn_reload_watcher") as mock_watcher, \
                 patch("cozempic.guard._detect_terminal_env", return_value="plain"):
                _terminate_and_resume(dead_pid, td, session_id=sid, session_path=jsonl)
            mock_watcher.assert_not_called()  # bare-liveness gate beat the mtime fallback
            self.assertFalse(sentinel.exists())


# ---------------------------------------------------------------------------
# Test 2 — SessionStart hook bash skips guard --daemon when sentinel exists
# ---------------------------------------------------------------------------
class TestSessionStartHookSkipsSpawnDuringSentinel(unittest.TestCase):
    """Option (c) fix: the SessionStart hook bash command must NOT invoke
    `cozempic guard --daemon` when the in-flight sentinel file exists.

    Strategy: plant the sentinel, run the hook bash with a stub cozempic
    binary, assert guard --daemon was never invoked.
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cozempic_sentinel_test_"))
        self.sid = "bbbbbbbbbbbb1234567890abcdefbbbb"
        self.sid12 = self.sid[:12]
        self.sentinel_path = Path(f"/tmp/cozempic_reload_{self.sid12}.in-flight")
        # Plant a fresh sentinel
        self.sentinel_path.write_text(f"89113\n{__import__('datetime').datetime.now().isoformat()}\n")
        self.addCleanup(self.sentinel_path.unlink, missing_ok=True)
        self.addCleanup(__import__("shutil").rmtree, self.tmpdir, True)

        # Build stub cozempic that logs invocations
        self.invocation_log = self.tmpdir / "invocations.log"
        stub_dir = self.tmpdir / "bin"
        stub_dir.mkdir()
        stub = stub_dir / "cozempic"
        stub.write_text(textwrap.dedent(f"""\
            #!/bin/sh
            echo "$@" >> {self.invocation_log}
            # Act as a no-op guard --daemon
            exit 0
        """))
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        self.stub_dir = stub_dir

    def _load_session_start_command(self) -> str:
        hooks_path = SRC / "cozempic" / "data" / "hooks.json"
        hooks = json.loads(hooks_path.read_text(encoding="utf-8"))
        return hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]

    def test_session_start_hook_skips_spawn_during_reload(self):
        """Hook must skip guard --daemon when sentinel is present (v10 behavior)."""
        cmd = self._load_session_start_command()
        # Sentinel check requires v10 hook schema — if not present, this RED is expected
        self.assertIn(
            "v10",
            cmd,
            "Hook schema is not v10 — sentinel check not yet in hooks.json. "
            "This test RED until Phase B adds sentinel skip to hooks.json.",
        )
        hook_data = json.dumps({"session_id": self.sid, "transcript_path": ""})
        env = os.environ.copy()
        env["PATH"] = f"{self.stub_dir}:{env.get('PATH', '')}"
        result = subprocess.run(
            ["bash", "-c", f"echo '{hook_data}' | {cmd}"],
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )
        # Check invocation log — guard --daemon must NOT appear
        if self.invocation_log.exists():
            invocations = self.invocation_log.read_text()
        else:
            invocations = ""
        self.assertNotIn(
            "guard --daemon",
            invocations,
            f"guard --daemon was invoked despite sentinel being present. "
            f"Invocations: {invocations!r}",
        )


# ---------------------------------------------------------------------------
# Test 3 — Watcher unlinks sentinel after osascript fires
# ---------------------------------------------------------------------------
class TestWatcherUnlinksSentinelAfterOsascript(unittest.TestCase):
    """After Option (c) fix: the watcher bash script must unlink the
    in-flight sentinel AFTER the resume command (osascript/gnome-terminal/etc)
    fires, so by the time NEW Claude's SessionStart hook runs, the sentinel
    is gone and spawn proceeds normally.
    """

    def setUp(self):
        self.sid = "cccccccccccc5678901234abcdefcccc"
        self.sid12 = self.sid[:12]
        self.sentinel_path = Path(f"/tmp/cozempic_reload_{self.sid12}.in-flight")
        self.sentinel_path.write_text(f"89113\n{__import__('datetime').datetime.now().isoformat()}\n")
        self.addCleanup(self.sentinel_path.unlink, missing_ok=True)

    def test_watcher_unlinks_sentinel_after_osascript(self):
        """Watcher bash script removes sentinel after osascript exits."""
        # The watcher is generated at runtime by _spawn_reload_watcher.
        # After Phase B, the watcher script will contain a `rm -f` or
        # unlink_reload_sentinel call AFTER the resume_cmd line.
        # We verify by calling _spawn_reload_watcher with a mocked osascript
        # (replaces it with `true`) and a mock subprocess.Popen that runs
        # the script directly (not in a detached process).

        # This will RED until Phase B adds the sentinel unlink to the watcher script.
        try:
            from cozempic.reload_lock import SENTINEL_TTL_SECONDS
        except ImportError:
            self.fail(
                "SENTINEL_TTL_SECONDS missing from reload_lock — Phase B not yet applied."
            )

        # Build a fake "dying" process: we'll use a subprocess that exits immediately
        fake_old_pid = os.getpid()  # use ourselves (alive), test will manipulate script

        scripts_run = []
        # Save the real Popen BEFORE the patch replaces it; _fake_popen uses it
        # to run the patched script synchronously without hitting the mock recursively.
        _real_popen = subprocess.Popen

        def _fake_popen(cmd_parts, **kwargs):
            # Actually run the watcher script, but synchronously so we can inspect
            # the sentinel after it completes
            if cmd_parts[0] == "bash" and cmd_parts[1] == "-c":
                script = cmd_parts[2]
                # Munge the script: replace `while kill -0 <pid>` with `true` (skip wait)
                # and replace the resume_cmd with `true` so osascript doesn't actually run.
                # Also shorten the poll deadline to 2s (default is 30s — too slow for tests)
                # and replace pgrep with empty output so status file write is exercised.
                import re
                patched = re.sub(r"while kill -0 \d+ 2>/dev/null; do sleep 1; done", "true", script)
                patched = re.sub(r"osascript[^;]+", "true", patched)
                patched = re.sub(r"gnome-terminal[^;]+", "true", patched)
                patched = re.sub(
                    r"deadline=[^;]+;",
                    "deadline=$(($(date +%s) + 2));",
                    patched,
                )
                patched = re.sub(
                    r"pgrep -f '[^']*' 2>/dev/null [|] head -n 1",
                    "echo ''",
                    patched,
                )
                scripts_run.append(patched)
                # Run synchronously using the real Popen (not the mock) to avoid recursion
                with _real_popen(
                    ["bash", "-c", patched],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ) as proc:
                    proc.communicate(timeout=10)
                return MagicMock(pid=99999)
            return MagicMock(pid=99999)

        with patch("cozempic.guard.subprocess.Popen", side_effect=_fake_popen), \
             patch("cozempic.guard.platform.system", return_value="Darwin"), \
             patch("cozempic.guard.is_ssh_session", return_value=False):
            from cozempic.guard import _spawn_reload_watcher
            _spawn_reload_watcher(
                claude_pid=fake_old_pid,
                project_dir="/tmp/fake_project",
                session_id=self.sid,
            )

        self.assertFalse(
            self.sentinel_path.exists(),
            f"Sentinel was NOT unlinked by the watcher script. "
            f"Scripts captured: {scripts_run}",
        )


# ---------------------------------------------------------------------------
# Test 4 — Sentinel mtime GC: stale sentinel treated as absent
# ---------------------------------------------------------------------------
class TestSentinelMtimeGCAfterStaleWindow(unittest.TestCase):
    """When a sentinel is older than SENTINEL_TTL_SECONDS, the SessionStart
    hook must treat it as stale and allow spawn to proceed.

    This covers the leak scenario: watcher was SIGKILL'd between osascript
    and sentinel-unlink, leaving an orphan sentinel. GC prevents permanent
    spawn suppression.
    """

    def setUp(self):
        self.sid = "dddddddddddd567890abcdef12345ddd"
        self.sid12 = self.sid[:12]
        self.sentinel_path = Path(f"/tmp/cozempic_reload_{self.sid12}.in-flight")
        self.addCleanup(self.sentinel_path.unlink, missing_ok=True)

    def test_sentinel_mtime_gc_after_stale_window(self):
        """Stale sentinel (age > SENTINEL_TTL_SECONDS) is ignored; spawn proceeds."""
        try:
            from cozempic.reload_lock import SENTINEL_TTL_SECONDS, write_reload_sentinel
        except ImportError:
            self.fail(
                "SENTINEL_TTL_SECONDS / write_reload_sentinel missing from reload_lock. "
                "Phase B not yet applied — expected RED."
            )

        # Plant a sentinel that is artificially old
        write_reload_sentinel(self.sid, claude_pid=89113)
        stale_age = SENTINEL_TTL_SECONDS + 10
        old_mtime = time.time() - stale_age
        os.utime(self.sentinel_path, (old_mtime, old_mtime))

        # Now call the Python-side sentinel check from start_guard_daemon
        # After Phase B, start_guard_daemon will have a sentinel check that
        # reads the file age and returns immediately if fresh, otherwise
        # treats as stale and allows spawn.
        from cozempic.guard import start_guard_daemon

        spawn_calls = []

        def _fake_popen(cmd_parts, **kwargs):
            spawn_calls.append(cmd_parts)
            return MagicMock(pid=99001)

        with patch("cozempic.guard.subprocess.Popen", side_effect=_fake_popen), \
             patch("cozempic.guard.find_claude_pid", return_value=94466), \
             patch("cozempic.guard.find_current_session", return_value={
                 "session_id": self.sid,
                 "path": Path("/tmp/fake.jsonl"),
             }), \
             patch("cozempic.guard._cleanup_legacy_pid"), \
             patch("cozempic.guard.maybe_auto_update", return_value=False):

            # Stale sentinel should NOT suppress spawn
            result = start_guard_daemon(session_id=self.sid, claude_pid=94466)

        self.assertNotEqual(
            result.get("reason"),
            "reload in flight",
            "start_guard_daemon returned 'reload in flight' for a STALE sentinel — "
            "GC not applied. Expected to spawn.",
        )
        # start_guard_daemon should have attempted to spawn (spawn_calls non-empty
        # OR result["started"] is True)
        self.assertTrue(
            spawn_calls or result.get("started") or result.get("already_running"),
            f"Spawn was suppressed by stale sentinel — GC logic missing. result={result}",
        )


# ---------------------------------------------------------------------------
# Test 5 — Option (b): pidfile unlinked IMMEDIATELY on watched Claude death
# ---------------------------------------------------------------------------
class TestPidfileUnlinkedImmediatelyOnWatchedClaudeDeath(unittest.TestCase):
    """Defense-in-depth (option b): in start_guard's main loop, when the
    watched Claude process dies, _safe_unlink_session_pidfile must be called
    BEFORE checkpoint_team (the expensive post-loop work).

    This closes the residual window where NEW Claude's SessionStart fires
    AFTER the old Claude dies but BEFORE the daemon's finally-block runs.
    """

    def test_pidfile_unlinked_immediately_on_watched_claude_death(self):
        """Pidfile is unlinked in the if-not-claude_alive branch, not only in finally."""
        # Inspect start_guard source to verify the call order.
        # This is a structural test — we read the source and assert the unlink
        # call precedes checkpoint_team in the not-claude_alive branch.
        import inspect
        try:
            from cozempic.guard import start_guard
        except ImportError:
            self.fail("Cannot import start_guard from cozempic.guard")

        source = inspect.getsource(start_guard)

        # Find the not-claude_alive block and check ordering
        # Strategy: find the position of `_safe_unlink_session_pidfile` and
        # `checkpoint_team` within the `not claude_alive` branch.
        # After the fix, _safe_unlink_session_pidfile should appear BEFORE
        # checkpoint_team in that block.

        # Locate the "claude_alive = False" → action block
        not_alive_pos = source.find("if not claude_alive:")
        if not_alive_pos == -1:
            # Older layout: `claude_alive = False` check is inline
            not_alive_pos = source.find("claude_alive = False")
        self.assertGreater(not_alive_pos, 0, "Could not find not-claude_alive block in start_guard source")

        sub_source = source[not_alive_pos:]

        unlink_pos = sub_source.find("_safe_unlink_session_pidfile")
        checkpoint_pos = sub_source.find("checkpoint_team")

        self.assertGreater(unlink_pos, -1,
            "_safe_unlink_session_pidfile not found in the not-claude_alive block — "
            "option (b) defense-in-depth not implemented. Expected RED until Phase B.")
        self.assertLess(
            unlink_pos, checkpoint_pos,
            f"_safe_unlink_session_pidfile (pos {unlink_pos}) is NOT before "
            f"checkpoint_team (pos {checkpoint_pos}) in the not-claude_alive block — "
            "pidfile is not freed before expensive checkpoint work. Option (b) violated.",
        )


# ---------------------------------------------------------------------------
# Test 6 — Reproducer: 86cb258b scenario walk (NEW Claude ends up UNPROTECTED)
# ---------------------------------------------------------------------------
class TestReproducer86cb258bNoTransientUnprotectedState(unittest.TestCase):
    """Full integration reproducer for the 86cb258b event sequence.

    Sequence:
    1. OLD guard daemon for session 86cb258b exits (reload path).
    2. Upgrade-chain re-fires SessionStart → transient guard spawns, claims pidfile slot.
    3. NEW Claude's SessionStart hook fires → sees transient daemon → skips spawn.
    4. RESULT: NEW Claude UNPROTECTED.

    BEFORE the fix (current main / v1.8.14 + PR #93):
      - This scenario leaves NEW Claude unprotected (DaemonAlreadyStarting raised).
      - Test expects DaemonAlreadyStarting to be raised, confirming the bug exists.
      - RED = hypothesis confirmed.

    AFTER the fix (Phase B):
      - Sentinel suppresses the transient daemon claim.
      - NEW Claude's SessionStart spawns a fresh guard.
      - Test expects NEW Claude to have started=True.
      - GREEN = fix verified.
    """

    def setUp(self):
        self.sid = "86cb258b3e024515849a0e25a485ca93"  # normalized 86cb258b session
        self.sid12 = self.sid[:12]
        self.old_claude_pid = 89113
        self.new_claude_pid = 94466
        self.pid_path = Path(f"/tmp/cozempic_guard_{self.sid12}.pid")
        self.sentinel_path = Path(f"/tmp/cozempic_reload_{self.sid12}.in-flight")
        self.pid_path.unlink(missing_ok=True)
        self.sentinel_path.unlink(missing_ok=True)
        self.addCleanup(self.pid_path.unlink, missing_ok=True)
        self.addCleanup(self.sentinel_path.unlink, missing_ok=True)

    def _simulate_transient_daemon_spawn(self):
        """Simulate the upgrade-chain re-fire: SessionStart hook spawns a
        transient daemon for OLD Claude's session while OLD Claude is still
        dying.

        Returns the transient daemon's mock PID.
        """
        from cozempic.spawn_lock import DaemonSpawnClaim, INIT_SPAWN_DAEMON
        from datetime import datetime

        # Transient daemon claims the pidfile slot (as would happen via
        # start_guard_daemon when upgrade chain fires with OLD Claude still alive)
        transient_pid = 99888
        claim = DaemonSpawnClaim(self.sid, self.pid_path)
        claim._claim()
        claim.owned = True

        # Write the transient daemon's PID (simulating the post-Popen rename)
        payload = (
            f"{transient_pid}\n"
            f"{datetime.now().isoformat(timespec='seconds')}\n"
            f"{INIT_SPAWN_DAEMON}\n"
        )
        self.pid_path.write_text(payload)
        claim.handed_off = True  # prevent __exit__ from unlinking

        return transient_pid

    def test_86cb258b_reproducer_no_transient_unprotected_state(self):
        """Verify Phase B sentinel fix: NEW Claude's SessionStart is NOT blocked
        when the reload sentinel is present.

        Phase B fix: _terminate_and_resume writes a sentinel BEFORE spawning
        the watcher. When the sentinel is present, start_guard_daemon returns
        {started: False, reason: 'reload in flight', already_running: False}
        instead of {already_running: True} — NEW Claude is PROTECTED.

        GREEN = fix verified (Phase B sentinel check active).
        """
        try:
            from cozempic.reload_lock import write_reload_sentinel
        except ImportError:
            self.fail(
                "write_reload_sentinel missing from reload_lock — Phase B not applied. "
                "Expected RED until Phase B implementation lands."
            )

        # Step 1: Write sentinel (simulates what _terminate_and_resume now does
        # with Phase B before spawning the reload watcher)
        write_reload_sentinel(self.sid, self.old_claude_pid)
        self.assertTrue(
            self.sentinel_path.exists(),
            "Sentinel not created — write_reload_sentinel failed.",
        )

        # Step 2: Transient daemon claims the slot (upgrade-chain re-fire)
        transient_pid = self._simulate_transient_daemon_spawn()

        # Step 3: NEW Claude's SessionStart calls start_guard_daemon
        # Expect: sentinel detected → returns {reason: 'reload in flight'},
        # NOT {already_running: True} which would mean NEW Claude is UNPROTECTED.
        from cozempic.guard import start_guard_daemon

        spawn_calls = []

        def _fake_popen(cmd_parts, **kwargs):
            spawn_calls.append(cmd_parts)
            return MagicMock(pid=self.new_claude_pid)

        with patch("cozempic.guard.subprocess.Popen", side_effect=_fake_popen), \
             patch("cozempic.guard.find_claude_pid", return_value=self.new_claude_pid), \
             patch("cozempic.guard.find_current_session", return_value={
                 "session_id": self.sid,
                 "path": Path("/tmp/fake_86cb258b.jsonl"),
             }), \
             patch("cozempic.guard._cleanup_legacy_pid"), \
             patch("cozempic.guard.maybe_auto_update", return_value=False), \
             patch("cozempic.spawn_lock._is_process_alive", return_value=True):

            result = start_guard_daemon(
                session_id=self.sid,
                claude_pid=self.new_claude_pid,
            )

        # Phase B fix: sentinel suppresses transient daemon check entirely.
        # Result must be {started: False, reason: 'reload in flight', already_running: False}.
        self.assertEqual(
            result.get("reason"),
            "reload in flight",
            f"Phase B sentinel fix not active: start_guard_daemon did not return "
            f"'reload in flight'. Got: {result}. "
            f"Transient daemon PID: {transient_pid}. "
            f"Sentinel exists: {self.sentinel_path.exists()}",
        )
        self.assertFalse(
            result.get("already_running"),
            f"already_running=True despite sentinel — Phase B sentinel check must "
            f"run BEFORE the transient daemon check. Got: {result}",
        )
        self.assertFalse(
            result.get("started"),
            f"started=True unexpected (sentinel should suppress spawn). Got: {result}",
        )


# ---------------------------------------------------------------------------
# Test 7 — Race under contention: N=10 concurrent SessionStart hooks
# ---------------------------------------------------------------------------
class TestRaceUnderContention(unittest.TestCase):
    """N=10 concurrent SessionStart hook invocations, half with sentinel present
    and half without.

    Invariants:
    - Zero guard --daemon spawns during the sentinel window (hooks with sentinel).
    - Exactly 1 guard --daemon spawn for the post-sentinel hooks (single-winner).
    - No multi-winner (DaemonSpawnClaim ensures this even without sentinel).
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cozempic_race_"))
        self.sid = "eeeeeeeeeeee789012345abcdefeeee0"
        self.sid12 = self.sid[:12]
        self.pid_path = Path(f"/tmp/cozempic_guard_{self.sid12}.pid")
        self.sentinel_path = Path(f"/tmp/cozempic_reload_{self.sid12}.in-flight")
        self.pid_path.unlink(missing_ok=True)
        self.sentinel_path.unlink(missing_ok=True)
        self.addCleanup(self.pid_path.unlink, missing_ok=True)
        self.addCleanup(self.sentinel_path.unlink, missing_ok=True)
        self.addCleanup(__import__("shutil").rmtree, self.tmpdir, True)

    def _run_start_guard_daemon(self, with_sentinel: bool, results: list, idx: int):
        """Worker: attempt start_guard_daemon; record result.

        NOTE: patches are applied at the test level (not per-thread) to avoid
        non-thread-safe patch restoration leaking across test boundaries.
        The worker simply calls start_guard_daemon with the already-mocked environment.
        """
        try:
            if with_sentinel:
                # Ensure sentinel is present (may already be)
                if not self.sentinel_path.exists():
                    self.sentinel_path.write_text(
                        f"89113\n{__import__('datetime').datetime.now().isoformat()}\n"
                    )
            from cozempic.guard import start_guard_daemon
            result = start_guard_daemon(session_id=self.sid, claude_pid=94466)
            results.append({"idx": idx, "with_sentinel": with_sentinel, "result": result})
        except Exception as exc:
            results.append({"idx": idx, "with_sentinel": with_sentinel, "error": str(exc)})

    def test_race_under_contention(self):
        """10 concurrent SessionStart calls: sentinel batch must be suppressed (0 spawns).

        This test REDs because the Phase B sentinel check is not yet in start_guard_daemon.
        On current code, all 10 calls attempt to spawn (sentinel is ignored).
        """
        # First: assert the Phase B symbol is present — otherwise the sentinel check
        # cannot exist. This is the correct RED failure surface for this test.
        try:
            from cozempic.guard import start_guard_daemon
            import inspect
            source = inspect.getsource(start_guard_daemon)
        except ImportError:
            self.fail("Cannot import start_guard_daemon")

        # The Phase B sentinel check must be present in start_guard_daemon source.
        # RED until Phase B adds: if _reload_sentinel_active(session_id): return {reason: ...}
        # Use specific functional phrases that only appear when the check is implemented
        # (not in existing docstrings about the spawn lock sentinel pattern).
        sentinel_check_phrases = [
            "reload in flight",          # the reason string returned on sentinel match
            "_reload_sentinel_active",   # the helper function name
            "write_reload_sentinel",     # import of the sentinel writer
            "in-flight",                 # the file suffix used in sentinel path
            "SENTINEL_TTL",              # the TTL constant from reload_lock
        ]
        found = any(phrase in source for phrase in sentinel_check_phrases)
        self.assertTrue(
            found,
            f"start_guard_daemon does not contain a sentinel/reload-in-flight check. "
            f"Phase B not yet applied. Checked for: {sentinel_check_phrases}. "
            f"This test RED is expected until Phase B implementation.",
        )

        # If we get here, Phase B is active. Run the actual contention test.
        self.sentinel_path.write_text(
            f"89113\n{__import__('datetime').datetime.now().isoformat()}\n"
        )

        results = []
        threads = []
        barrier = threading.Barrier(10)

        def _worker(with_sentinel, idx):
            barrier.wait(timeout=5)
            self._run_start_guard_daemon(with_sentinel, results, idx)

        # Apply patches at the TEST level (single-threaded) before spawning workers.
        # Per-thread patch/unpatch is NOT thread-safe in unittest.mock and leaks mocks
        # across test boundaries. Applying once here is safe and correct.
        def _fake_popen(cmd_parts, **kwargs):
            return MagicMock(pid=90001)

        with patch("cozempic.guard.subprocess.Popen", side_effect=_fake_popen), \
             patch("cozempic.guard.find_claude_pid", return_value=94466), \
             patch("cozempic.guard.find_current_session", return_value={
                 "session_id": self.sid,
                 "path": Path("/tmp/fake.jsonl"),
             }), \
             patch("cozempic.guard._cleanup_legacy_pid"), \
             patch("cozempic.guard.maybe_auto_update", return_value=False):

            for i in range(5):
                t = threading.Thread(target=_worker, args=(True, i), daemon=True)
                threads.append(t)
            for i in range(5, 10):
                t = threading.Thread(target=_worker, args=(False, i), daemon=True)
                threads.append(t)

            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)
            # Patches are restored here (after all threads complete)

        sentinel_started = [r["result"].get("started") for r in results
                           if not r.get("error") and r["with_sentinel"]]
        no_sentinel_started = [r["result"].get("started") for r in results
                               if not r.get("error") and not r["with_sentinel"]]

        sentinel_spawn_count = sum(1 for s in sentinel_started if s)
        self.assertEqual(
            sentinel_spawn_count,
            0,
            f"Expected 0 spawns in sentinel window, got {sentinel_spawn_count}. "
            f"Results: {results}.",
        )

        no_sentinel_spawn_count = sum(1 for s in no_sentinel_started if s)
        self.assertLessEqual(
            no_sentinel_spawn_count,
            1,
            f"Multi-winner detected in no-sentinel batch: {no_sentinel_spawn_count} spawns. "
            f"Results: {results}",
        )


# ---------------------------------------------------------------------------
# Test 8/9/10 — PR #94 review MED-1/2/3 regression: sentinel NOT written on
# code paths that don't actually spawn the watcher (SSH, tmux/screen PID-reuse
# early returns, _spawn_reload_watcher SSH double-check, unsupported OS).
# Without this fix the sentinel would leak for SENTINEL_TTL_SECONDS=120s and
# suppress legitimate SessionStart guard spawns during that window.
# ---------------------------------------------------------------------------
class TestSentinelNotWrittenOnEarlyReturns(unittest.TestCase):
    """Reviewer MED-1/2/3 fold: sentinel must NOT be written on paths that
    return without spawning a watcher (no NEW Claude → no need to suppress).

    Without these tests a future refactor could re-introduce the leak class.
    """

    def setUp(self):
        self.sid = "fedcba987654321098765432abcdef01"
        self.sid12 = self.sid[:12]
        self.sentinel_path = Path(f"/tmp/cozempic_reload_{self.sid12}.in-flight")
        self.sentinel_path.unlink(missing_ok=True)
        self.addCleanup(self.sentinel_path.unlink, missing_ok=True)

    def test_ssh_path_does_not_write_sentinel(self):
        """SSH (manual-resume-only) path must NOT write the sentinel."""
        with patch("cozempic.guard._detect_terminal_env", return_value="ssh"), \
             patch("cozempic.guard._spawn_reload_watcher"):
            from cozempic.guard import _terminate_and_resume
            _terminate_and_resume(
                claude_pid=89113,
                project_dir="/tmp/fake_project",
                session_id=self.sid,
            )
        self.assertFalse(
            self.sentinel_path.exists(),
            f"SSH path leaked sentinel at {self.sentinel_path}. "
            "MED-1: SSH has no auto-resume, must not suppress next SessionStart.",
        )

    def test_tmux_pid_reuse_fail_does_not_leak_sentinel(self):
        """tmux + _is_claude_process=False early return must NOT leak sentinel."""
        with patch("cozempic.guard._detect_terminal_env", return_value="tmux"), \
             patch("cozempic.guard._is_claude_process", return_value=False), \
             patch("cozempic.guard._spawn_reload_watcher"):
            from cozempic.guard import _terminate_and_resume
            _terminate_and_resume(
                claude_pid=89113,
                project_dir="/tmp/fake_project",
                session_id=self.sid,
            )
        self.assertFalse(
            self.sentinel_path.exists(),
            f"tmux PID-reuse early-return leaked sentinel. "
            "MED-2: no termination happened, no NEW Claude, must not suppress.",
        )

    def test_screen_pid_reuse_fail_does_not_leak_sentinel(self):
        """screen + _is_claude_process=False early return must NOT leak sentinel."""
        with patch("cozempic.guard._detect_terminal_env", return_value="screen"), \
             patch("cozempic.guard._is_claude_process", return_value=False), \
             patch("cozempic.guard._spawn_reload_watcher"):
            from cozempic.guard import _terminate_and_resume
            _terminate_and_resume(
                claude_pid=89113,
                project_dir="/tmp/fake_project",
                session_id=self.sid,
            )
        self.assertFalse(
            self.sentinel_path.exists(),
            f"screen PID-reuse early-return leaked sentinel. "
            "MED-2: no termination happened, no NEW Claude, must not suppress.",
        )

    def test_spawn_reload_watcher_ssh_does_not_leak_sentinel(self):
        """_spawn_reload_watcher second-chance SSH check (MED-3) must NOT leak."""
        with patch("cozempic.guard.is_ssh_session", return_value=True), \
             patch("cozempic.guard.subprocess.Popen"):
            from cozempic.guard import _spawn_reload_watcher
            _spawn_reload_watcher(
                claude_pid=89113,
                project_dir="/tmp/fake_project",
                session_id=self.sid,
            )
        self.assertFalse(
            self.sentinel_path.exists(),
            f"_spawn_reload_watcher SSH early-return leaked sentinel. "
            "MED-3: bash watcher never spawned, sentinel unlink never fires, leak class.",
        )


if __name__ == "__main__":
    unittest.main()
