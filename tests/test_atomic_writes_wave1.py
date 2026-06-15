"""Wave 1 — atomic write hardening + concurrency tests.

Covers fixes for the production incidents where:
- _save_sidecar collided on a fixed `.tmp` filename when two guard daemons
  started simultaneously (FileNotFoundError on the loser's os.replace)
- save_messages had the same fixed-tmp bug
- record_savings was not atomic at all (direct write_text)
- cmd_reload/cmd_treat/cmd_strategy bypassed _PruneLock
- SessionStart hook fired guard --daemon twice (unflocked foreground + flocked)
- overflow_watcher thread leaked past normal-exit breaks
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


# ─── atomic_write_text (new shared primitive) ────────────────────────────────

class TestAtomicWriteText(unittest.TestCase):
    def test_basic_write(self):
        from cozempic.helpers import atomic_write_text
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out.json"
            atomic_write_text(target, '{"key": "value"}')
            self.assertEqual(target.read_text(), '{"key": "value"}')

    def test_overwrite_existing(self):
        from cozempic.helpers import atomic_write_text
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out.json"
            target.write_text("old")
            atomic_write_text(target, "new")
            self.assertEqual(target.read_text(), "new")

    def test_no_tmp_file_left_on_success(self):
        from cozempic.helpers import atomic_write_text
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out.json"
            atomic_write_text(target, "data")
            # No .tmp* files should remain
            leftovers = list(Path(tmp).glob(".tmp.*"))
            self.assertEqual(leftovers, [], f"tmp files leaked: {leftovers}")

    def test_concurrent_writes_no_collision(self):
        """20 parallel writes to the SAME target succeed without
        FileNotFoundError (the production bug). Last writer wins; no crashes."""
        from cozempic.helpers import atomic_write_text
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out.json"
            errors = []

            def writer(i):
                try:
                    atomic_write_text(target, f'{{"writer": {i}}}')
                except Exception as e:
                    errors.append((i, type(e).__name__, str(e)))

            with ThreadPoolExecutor(max_workers=20) as ex:
                list(ex.map(writer, range(20)))

            self.assertEqual(errors, [], f"Concurrent writes raised: {errors}")
            # Target is valid JSON (last writer wins)
            self.assertIsInstance(json.loads(target.read_text()), dict)
            # No tmp leftovers
            self.assertEqual(list(Path(tmp).glob(".tmp.*")), [])

    def test_failure_cleans_up_tmp(self):
        """If write fails (e.g. disk full), tmp file is unlinked."""
        from cozempic.helpers import atomic_write_text
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out.json"
            # Simulate fdopen failure
            with patch("cozempic.helpers.os.fdopen", side_effect=OSError("simulated")):
                with self.assertRaises(OSError):
                    atomic_write_text(target, "data")
            self.assertEqual(list(Path(tmp).glob(".tmp.*")), [])


# ─── _save_sidecar — the production crash ────────────────────────────────────

class TestSaveSidecarConcurrent(unittest.TestCase):
    def test_concurrent_record_session_no_filenotfound(self):
        """The exact production race: 10 threads calling record_session
        simultaneously. Pre-fix produced FileNotFoundError on N-1 of them."""
        from cozempic.session import record_session
        with tempfile.TemporaryDirectory() as tmp:
            # Redirect sidecar to tmp
            with patch("cozempic.session.get_sidecar_path",
                       return_value=Path(tmp) / "cozempic-sessions.json"):
                errors = []

                def writer(i):
                    try:
                        record_session(f"session-{i:04d}", "/tmp/cwd", 1_000_000)
                    except Exception as e:
                        errors.append((i, type(e).__name__, str(e)))

                with ThreadPoolExecutor(max_workers=10) as ex:
                    list(ex.map(writer, range(10)))

                self.assertEqual(errors, [], f"record_session raised: {errors}")
                # All 10 sessions present (the _HostFileLock prevents
                # lost updates even though writes themselves are atomic)
                data = json.loads((Path(tmp) / "cozempic-sessions.json").read_text())
                self.assertEqual(len(data), 10,
                    f"Expected 10 sessions, got {len(data)} — lost updates!")


# ─── record_savings — lost-update protection ─────────────────────────────────

class TestRecordSavingsAtomic(unittest.TestCase):
    def test_concurrent_increments_no_lost_updates(self):
        """50 concurrent record_savings(1000) calls — final total must be 50000.
        Pre-fix used direct write_text with no lock → races lost updates."""
        from cozempic.helpers import record_savings
        with tempfile.TemporaryDirectory() as tmp:
            savings_file = Path(tmp) / ".cozempic_savings.json"
            with patch("cozempic.helpers._SAVINGS_FILE", savings_file):
                # Disable telemetry pings
                with patch.dict(os.environ, {"COZEMPIC_NO_TELEMETRY": "1"}):
                    def writer(_i):
                        record_savings(1000)

                    with ThreadPoolExecutor(max_workers=10) as ex:
                        list(ex.map(writer, range(50)))

                data = json.loads(savings_file.read_text())
                self.assertEqual(data["tokens_saved"], 50000,
                    f"Lost-update race: expected 50000, got {data['tokens_saved']}")
                self.assertEqual(data["prune_count"], 50)


# ─── save_messages — mkstemp protection ──────────────────────────────────────

class TestSaveMessagesMkstemp(unittest.TestCase):
    def test_save_uses_unique_tmp_filename(self):
        """tmp file is NOT path.with_suffix('.tmp') anymore — uses mkstemp."""
        from cozempic.session import save_messages
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            path.write_text('{"a": 1}\n')
            messages = [(0, {"type": "user", "message": {"content": "hi"}}, 30)]

            # Capture which tmp filenames mkstemp generates
            tmp_names_seen = []
            real_mkstemp = tempfile.mkstemp

            def spy_mkstemp(*args, **kwargs):
                fd, name = real_mkstemp(*args, **kwargs)
                tmp_names_seen.append(name)
                return fd, name

            with patch("tempfile.mkstemp", side_effect=spy_mkstemp):
                save_messages(path, messages, create_backup=False)

            self.assertEqual(len(tmp_names_seen), 1)
            tmp_name = Path(tmp_names_seen[0]).name
            # Must NOT be the old fixed "session.tmp" form
            self.assertNotEqual(tmp_name, "session.tmp",
                "save_messages still uses fixed tmp filename!")
            # Must start with .tmp. prefix
            self.assertTrue(tmp_name.startswith(".tmp."), f"Unexpected tmp name: {tmp_name}")


# ─── _PruneLock + snapshot in cmd_treat / cmd_strategy / cmd_reload ──────────

class TestCmdReloadAcquiresPruneLock(unittest.TestCase):
    """cmd_reload now passes snapshot= to save_messages AND wraps the save
    in _PruneLock. The forensic report flagged this as the data-loss path
    when guard auto-fires concurrently with manual reload."""

    def test_imports_have_prunelock_and_snapshot(self):
        """Smoke test — cli.py imports _PruneLock, PruneConflictError,
        PruneLockError, snapshot_session from session."""
        from cozempic.cli import _PruneLock, PruneLockError, PruneConflictError, snapshot_session
        self.assertTrue(callable(_PruneLock))
        self.assertTrue(callable(snapshot_session))

    def test_cmd_treat_exits_2_when_prune_lock_held(self):
        """The exact race that hit production: cmd_treat --execute while
        guard holds _PruneLock on the same session. Must exit cleanly with
        code 2 and a clear error message — NOT silently overwrite the
        guard's pruned output."""
        import argparse
        from cozempic.cli import cmd_treat, _PruneLock
        from cozempic.session import save_messages as _real_save_messages

        with tempfile.TemporaryDirectory() as tmp:
            session_path = Path(tmp) / "session.jsonl"
            # Real session content — a single valid JSONL line
            session_path.write_text(
                '{"type":"user","message":{"content":"hi"},"uuid":"a","sessionId":"s"}\n'
            )

            # Acquire the prune lock in a background thread and HOLD it
            lock_acquired = threading.Event()
            release_lock = threading.Event()

            def hold_lock():
                with _PruneLock(session_path):
                    lock_acquired.set()
                    release_lock.wait(timeout=10)

            holder = threading.Thread(target=hold_lock, daemon=True)
            holder.start()
            try:
                self.assertTrue(lock_acquired.wait(timeout=5),
                    "Background thread failed to acquire _PruneLock")

                # Build the args object cmd_treat expects
                args = argparse.Namespace(
                    session=str(session_path),
                    project=None,
                    rx="standard",
                    execute=True,
                    thinking_mode=None,
                    force=True,  # skip the active-tasks prompt
                )

                # Patch resolve_session to return our test path directly
                with patch("cozempic.cli.resolve_session", return_value=session_path):
                    # cmd_treat must exit(2) when the lock is held
                    with self.assertRaises(SystemExit) as cm:
                        cmd_treat(args)
                    self.assertEqual(cm.exception.code, 2,
                        f"Expected exit code 2 (lock held), got {cm.exception.code}")
            finally:
                release_lock.set()
                holder.join(timeout=5)


class TestHostFileLockWindows(unittest.TestCase):
    """Coverage for the msvcrt branch of _HostFileLock — runs on POSIX too
    by monkey-patching os.name and injecting a fake msvcrt module."""

    def test_windows_branch_uses_msvcrt_locking(self):
        from cozempic.helpers import _HostFileLock
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out.json"

            # Build a fake msvcrt module that records calls
            class FakeMsvcrt:
                LK_LOCK = 1
                LK_UNLCK = 0
                calls = []

                @classmethod
                def locking(cls, fd, mode, nbytes):
                    cls.calls.append((mode, nbytes))

            import sys as _sys
            with patch.object(os, "name", "nt"):
                # Inject fake msvcrt into sys.modules so the lazy `import msvcrt`
                # inside _HostFileLock picks it up.
                with patch.dict(_sys.modules, {"msvcrt": FakeMsvcrt}):
                    with _HostFileLock(target):
                        pass

            # Verify msvcrt.locking was called: once LK_LOCK on enter, once LK_UNLCK on exit
            self.assertEqual(len(FakeMsvcrt.calls), 2,
                f"Expected 2 msvcrt.locking calls (lock + unlock), got {FakeMsvcrt.calls}")
            self.assertEqual(FakeMsvcrt.calls[0], (FakeMsvcrt.LK_LOCK, 1),
                "First call should be LK_LOCK")
            self.assertEqual(FakeMsvcrt.calls[1], (FakeMsvcrt.LK_UNLCK, 1),
                "Second call should be LK_UNLCK")

    def test_windows_branch_no_msvcrt_degrades_gracefully(self):
        """If msvcrt import fails on Windows (shouldn't happen, but be safe),
        the lock degrades to no-op rather than crashing."""
        from cozempic.helpers import _HostFileLock
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out.json"

            import sys as _sys
            with patch.object(os, "name", "nt"):
                # Remove msvcrt from sys.modules and block import
                # by setting it to a module that raises ImportError on access
                with patch.dict(_sys.modules, {"msvcrt": None}):
                    # Should NOT raise — degrades to _fh = None
                    with _HostFileLock(target) as lock:
                        self.assertIsNone(lock._fh,
                            "Expected lock to degrade to no-op when msvcrt unavailable")


# ─── Hook schema bump v8 → v9 (PR #93 — pidfile 3-line format + head -1) ────

class TestHookSchemaV9(unittest.TestCase):
    """v9 migrates the bash hook's PID liveness probe from
    ``cat "$GUARD_PID_FILE"`` to ``head -n 1 "$GUARD_PID_FILE"`` to
    handle the new 3-line pidfile format introduced in PR #93 (item #5)
    for operator-triage metadata parity with ``_ReloadLock``.

    Without ``head -n 1``, the multi-line pidfile content would be
    passed to ``kill -0`` as multiple whitespace-separated arguments,
    producing shell-implementation-defined behaviour (the multi-arg
    ``kill -0`` syntax exists but tries each PID — likely succeeds on
    the first valid line, but the timestamp/initiator lines would emit
    confusing errors). Using ``head -n 1`` extracts only line 1 (the
    PID) cleanly across all POSIX shells.

    v8's C2 slug convergence is preserved verbatim (kept as the
    codebase convention); v9 only changes the PID-reading mechanism."""

    def test_schema_marker_bumped(self):
        from cozempic.init import HOOK_SCHEMA_VERSION
        # Durable floor (was pinned to an exact version → broke on every bump).
        self.assertGreaterEqual(int(HOOK_SCHEMA_VERSION.lstrip("v")), 12)

    def test_no_unflocked_foreground_guard_daemon_call(self):
        """The unflocked foreground `cozempic guard --daemon` call stays removed.
        v7's three-branch idempotency (kept verbatim in v8) wraps the spawn in
        `if PID alive then : ; elif has flock then (flock -n 8 || exit 0;
        spawn) 8>$STARTUP_LOCK ; else spawn ; fi`. Two branches (elif + else)
        each carry the `cozempic guard --daemon || python3 -m cozempic guard
        --daemon` pair, so the total occurrence count is 4 (2 branches × 2
        fallback variants). Anything else means an unguarded branch or a
        missing fallback."""
        hooks_path = Path(__file__).parent.parent / "src" / "cozempic" / "data" / "hooks.json"
        hooks = json.loads(hooks_path.read_text())
        ss_cmd = hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]
        count = ss_cmd.count("cozempic guard --daemon")
        self.assertEqual(count, 4,
            f"Expected 4 'cozempic guard --daemon' occurrences (2 branches × "
            f"[primary + python3 fallback]), got {count}")

    def test_startup_lock_present(self):
        """v8 keeps v7's second-layer guard: when flock is available, the
        daemon spawn runs inside a flock on a dedicated `*.startup-lock`
        file (distinct from the existing hook-level `*.lock`). The two locks
        MUST be distinct files so the daemon-spawn lock doesn't serialize
        unrelated hook work. Slug source is ``${SESSION_ID:0:12}`` — the
        bash side uses the permissive ``re.sub`` sanitiser, the same
        12-char prefix Python now produces via the relaxed regex."""
        hooks_path = Path(__file__).parent.parent / "src" / "cozempic" / "data" / "hooks.json"
        hooks = json.loads(hooks_path.read_text())
        ss_cmd = hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]
        self.assertIn("GUARD_STARTUP_LOCK=", ss_cmd,
            "Expected a dedicated GUARD_STARTUP_LOCK variable for the "
            "second-layer flock guarding the daemon spawn")
        self.assertIn("/tmp/cozempic_guard_${SESSION_ID:0:12}.startup-lock", ss_cmd,
            "GUARD_STARTUP_LOCK path must use ${SESSION_ID:0:12} — first 12 chars "
            "of the bash-sanitised lowercased session_id. Convergence (C2/Option B) "
            "is achieved by Python relaxing its regex to match this character set, "
            "not by bash tightening to match a stricter Python regex.")
        # The startup-lock must use a DIFFERENT fd than the outer hook lock
        # (fd 9) — otherwise the inner flock acquire would race the outer.
        self.assertIn("flock -n 8", ss_cmd,
            "Inner startup-lock flock must use fd 8 (outer hook lock owns fd 9)")
        self.assertIn("8>\"$GUARD_STARTUP_LOCK\"", ss_cmd,
            "fd 8 must be redirected to the startup-lock file via 8>\"$GUARD_STARTUP_LOCK\"")

    def test_pid_fast_path_present(self):
        """v8 keeps v7's fast-path invariant: the SessionStart hook must wrap
        the guard --daemon spawn in a `kill -0 $(cat GUARD_PID_FILE)` fast-path
        so a healthy daemon for the current session is NOT respawned. The
        GUARD_PID_FILE path uses ``${SESSION_ID:0:12}`` (bash-sanitised
        lowercased session_id, first 12 chars) — Python's relaxed regex
        produces the same slug for the same input."""
        hooks_path = Path(__file__).parent.parent / "src" / "cozempic" / "data" / "hooks.json"
        hooks = json.loads(hooks_path.read_text())
        ss_cmd = hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]
        self.assertIn("GUARD_PID_FILE=", ss_cmd,
            "Expected the SessionStart hook to declare GUARD_PID_FILE for the fast-path")
        self.assertIn("kill -0", ss_cmd,
            "Expected `kill -0` liveness probe to short-circuit redundant daemon spawns")
        # Path must match Python's `_pid_file_for_session` convention
        # (/tmp/cozempic_guard_<slug>.pid where <slug> = first 12 chars of
        # the lowercased+sanitised session_id; Python validates same charset).
        self.assertIn("/tmp/cozempic_guard_${SESSION_ID:0:12}.pid", ss_cmd,
            "Fast-path PID file path must use ${SESSION_ID:0:12} — first 12 chars "
            "of the bash-sanitised lowercased session_id, which (post-C2 Option B) "
            "matches the slug Python computes via the relaxed _SESSION_ID_RE.")

    def test_session_id_lowercased(self):
        """The hook MUST lowercase the session_id before sanitising — Python's
        ``_pid_file_for_session`` calls ``.lower()`` first, so any UPPER-case
        UUID from Claude Code would compute a different slug in bash than in
        Python if bash didn't also lowercase. (Pre-C2 finding was about
        character-set divergence; this pins the case-handling parity that
        prevents a second, narrower divergence.)"""
        hooks_path = Path(__file__).parent.parent / "src" / "cozempic" / "data" / "hooks.json"
        hooks = json.loads(hooks_path.read_text())
        ss_cmd = hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]
        self.assertIn(".lower()", ss_cmd,
            "session_id extraction must call .lower() to match the Python "
            "PID file naming convention; otherwise an upper-case UUID from "
            "Claude Code stdin would cause the fast-path to look at the "
            "wrong file and the daemon spawn would not be deduplicated.")
        # The bash sanitiser must use the permissive character class that
        # mirrors Python's relaxed regex (C2 Option B convergence).
        self.assertIn("[^a-z0-9_-]", ss_cmd,
            "bash sanitiser must use the permissive [^a-z0-9_-] character "
            "class that mirrors Python's relaxed _SESSION_ID_RE — same "
            "character set as reload_lock._slug_for and spawn_lock._slug_for.")

    def test_plugin_and_data_hooks_synced(self):
        plugin_path = Path(__file__).parent.parent / "plugin" / "hooks" / "hooks.json"
        data_path = Path(__file__).parent.parent / "src" / "cozempic" / "data" / "hooks.json"
        self.assertEqual(plugin_path.read_text(), data_path.read_text())


# ─── doctor.fix_corrupted_tool_use acquires _PruneLock ───────────────────────

class TestDoctorRespectsPruneLock(unittest.TestCase):
    def test_fix_corrupted_uses_prune_lock_and_reports_skipped(self):
        """fix_corrupted_tool_use must acquire _PruneLock and surface skipped
        sessions to the user. Source-inspection smoke test — behavioral test
        would require a corrupted session fixture which is expensive."""
        from cozempic.doctor import fix_corrupted_tool_use
        import inspect
        src = inspect.getsource(fix_corrupted_tool_use)
        self.assertIn("_PruneLock", src,
            "fix_corrupted_tool_use must use _PruneLock")
        self.assertIn("Skipped", src,
            "fix_corrupted_tool_use must report skipped sessions")

    def test_fix_corrupted_repairs_line_with_unicode_separator(self):
        """A corrupt tool_use on a line that ALSO carries a raw U+2028 must still be
        repaired — str.splitlines() would tear it into unparseable fragments and
        silently skip the repair (4th sibling of the splitlines class)."""
        import json
        from cozempic import session as S
        from cozempic import doctor as D
        proj = S.get_projects_dir() / "-unic"
        proj.mkdir(parents=True, exist_ok=True)
        sess = proj / "unicrep1.jsonl"
        corrupt_name = 'Bash" command="' + ("y" * 250) + '"'
        # One assistant line: a corrupt tool_use AND a text block with a raw U+2028.
        line = json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "before after"},
            {"type": "tool_use", "id": "t1", "name": corrupt_name, "input": {}},
        ]}}, ensure_ascii=False)
        sess.write_text(line + "\n", encoding="utf-8")
        D.fix_corrupted_tool_use()
        # Read back splitting on "\n" only (NOT splitlines — the line carries U+2028).
        objs = [json.loads(ln) for ln in sess.read_text(encoding="utf-8").split("\n") if ln.strip()]
        tu = next(b for o in objs for b in (o.get("message", {}).get("content") or [])
                  if isinstance(o.get("message", {}).get("content"), list) and b.get("type") == "tool_use")
        self.assertEqual(tu["name"], "Bash", "corrupt block on a U+2028 line must still be repaired")

    def test_fix_corrupted_preserves_concurrent_append_behavioral(self):
        """BEHAVIORAL (not static): fix_corrupted_tool_use must NOT drop a line
        Claude appends between our read and our write. Regression for the audit
        P1 (read_text→atomic_write with no snapshot). We inject the concurrent
        append exactly between read and classify via a wrapper snapshot."""
        import json
        from pathlib import Path
        from unittest import mock
        from cozempic import session as S
        from cozempic import doctor as D

        proj = S.get_projects_dir() / "-proj"
        proj.mkdir(parents=True, exist_ok=True)
        sess = proj / "behav1234.jsonl"
        # A corrupted tool_use block: name > 200 chars in the flattened-XML form.
        corrupt_name = 'Bash" command="' + ("x" * 250) + '"'
        line0 = json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}})
        line1 = json.dumps({"type": "assistant", "message": {"role": "assistant",
                  "content": [{"type": "tool_use", "id": "t1", "name": corrupt_name, "input": {}}]}})
        sess.write_text(line0 + "\n" + line1 + "\n", encoding="utf-8")

        # The line Claude "appends" mid-repair — must survive.
        appended_obj = {"type": "user", "message": {"role": "user", "content": "APPENDED_LIVE_TURN"}}
        appended_line = json.dumps(appended_obj) + "\n"

        class _InjectingSnapshot:
            def __init__(self, real, path):
                self._real, self._path, self._done = real, path, False
            @property
            def size(self):
                return self._real.size
            def read_delta(self, p):
                return self._real.read_delta(p)
            def _inject(self):
                if not self._done:  # inject append between the function's read and classify
                    with open(self._path, "a", encoding="utf-8") as f:
                        f.write(appended_line)
                    self._done = True
            def classify(self, p):
                self._inject()
                return self._real.classify(p)
            def classify_and_delta(self, p):
                self._inject()
                return self._real.classify_and_delta(p)

        # doctor now reads once via _FileSnapshot.from_bytes(path, raw); wrap that
        # so the append is injected between the read and classify.
        real_from_bytes = S._FileSnapshot.from_bytes.__func__
        def injecting_from_bytes(cls, path, raw):
            return _InjectingSnapshot(real_from_bytes(cls, path, raw), path)

        with mock.patch.object(S._FileSnapshot, "from_bytes", classmethod(injecting_from_bytes)):
            D.fix_corrupted_tool_use()

        final = sess.read_text(encoding="utf-8")
        # 1) The concurrently-appended live turn survived (no data loss).
        self.assertIn("APPENDED_LIVE_TURN", final,
                      "fix_corrupted_tool_use dropped a concurrently-appended line (data loss)")
        # 2) The corruption was actually repaired (name parsed back to 'Bash').
        objs = [json.loads(ln) for ln in final.splitlines() if ln.strip()]
        tu = next(b for o in objs for b in o.get("message", {}).get("content", [])
                  if isinstance(o.get("message", {}).get("content"), list) and b.get("type") == "tool_use")
        self.assertEqual(tu["name"], "Bash", "corruption must still be repaired")

    def test_fix_orphaned_uses_prune_lock_snapshot_and_reports_skipped(self):
        """fix_orphaned_tool_results was the second unprotected save_messages
        site flagged by the architecture review. Must mirror fix_corrupted's
        protection: _PruneLock + snapshot + skip-on-conflict."""
        from cozempic.doctor import fix_orphaned_tool_results
        import inspect
        src = inspect.getsource(fix_orphaned_tool_results)
        self.assertIn("_PruneLock", src,
            "fix_orphaned_tool_results must use _PruneLock")
        self.assertIn("load_messages_and_snapshot", src,
            "fix_orphaned_tool_results must read once (snapshot+messages from same bytes)")
        self.assertIn("Skipped", src,
            "fix_orphaned_tool_results must report skipped sessions")


class TestMcpTreatSessionPruneLock(unittest.TestCase):
    """The MCP plugin's treat_session tool was the third unprotected
    save_messages site flagged by the architecture review. It exposes the
    same data-loss race to users invoking /cozempic:treat via Claude Code."""

    def test_mcp_treat_session_aborts_on_conflict_behavioral(self):
        """BEHAVIORAL (not a static scan): the MCP treat_session(execute=True) guard
        must ABORT (PruneConflictError) and NOT clobber when the session is rewritten
        mid-prune. Loads the plugin with a fastmcp stub and drives the real function."""
        import importlib.util
        import json
        import sys
        import types
        from pathlib import Path
        from unittest import mock
        from cozempic import session as S

        # Stub fastmcp so @mcp.tool() returns the function unchanged + import is cheap.
        fake = types.ModuleType("fastmcp")
        class _FakeMCP:
            def __init__(self, *a, **k): pass
            def tool(self, *a, **k):
                def deco(fn): return fn
                return deco
        fake.FastMCP = _FakeMCP
        plugin_path = Path(__file__).parent.parent / "plugin" / "servers" / "cozempic_mcp.py"
        with mock.patch.dict(sys.modules, {"fastmcp": fake}):
            spec = importlib.util.spec_from_file_location("cozempic_mcp_behav", plugin_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

        # A real session under the isolated projects dir (conftest sets CLAUDE_CONFIG_DIR).
        proj = S.get_projects_dir() / "-mcp"
        proj.mkdir(parents=True, exist_ok=True)
        sess_path = proj / "mcpbehav1.jsonl"
        body = "".join(json.dumps({"type": "user", "message": {"role": "user",
                "content": "ORIGINAL_LINE_%d" % i}}) + "\n" for i in range(20))
        sess_path.write_text(body, encoding="utf-8")
        sess = {"path": sess_path, "session_id": "mcpbehav1", "project": "-mcp",
                "size": sess_path.stat().st_size, "mtime": 0, "lines": 20}

        # treat_session reads once via load_messages_and_snapshot; wrap _FileSnapshot
        # .from_bytes so the returned snapshot injects a same-prefix rewrite at
        # classify time (→ "conflict"), simulating a concurrent mutation mid-prune.
        class _Conflicting:
            def __init__(self, real, path): self._r, self._p, self._done = real, path, False
            @property
            def size(self): return self._r.size
            def read_delta(self, p): return self._r.read_delta(p)
            def _inject(self):
                if not self._done:
                    with open(self._p, "r+b") as f:  # mutate the prefix in place
                        f.seek(0); f.write(b'{"type":"user","message":{"role":"user","content":"REWRITTEN"}}')
                    self._done = True
            def classify(self, p):
                self._inject()
                return self._r.classify(p)
            def classify_and_delta(self, p):
                self._inject()
                return self._r.classify_and_delta(p)

        real_from_bytes = S._FileSnapshot.from_bytes.__func__
        def injecting_from_bytes(cls, path, raw):
            return _Conflicting(real_from_bytes(cls, path, raw), path)

        with mock.patch.object(S, "find_current_session", lambda *a, **k: sess), \
             mock.patch.object(S._FileSnapshot, "from_bytes", classmethod(injecting_from_bytes)):
            out = mod.treat_session(prescription="standard", execute=True)

        self.assertIn("Aborted: session changed mid-prune", out,
                      "treat_session must abort on a concurrent rewrite, not clobber")
        # The original content must NOT have been replaced by the pruned buffer.
        self.assertIn("ORIGINAL_LINE_19", sess_path.read_text(encoding="utf-8"),
                      "conflict abort must leave the live session intact (no data loss)")

    def test_mcp_treat_session_uses_prune_lock_and_snapshot(self):
        import inspect
        # The plugin lives outside the src/ tree; load it via path
        plugin_path = Path(__file__).parent.parent / "plugin" / "servers" / "cozempic_mcp.py"
        src = plugin_path.read_text()
        # The treat_session function must contain _PruneLock + snapshot_session
        # + the abort messages for both error paths
        self.assertIn("_PruneLock", src,
            "MCP treat_session must use _PruneLock")
        self.assertIn("load_messages_and_snapshot", src,
            "MCP treat_session must read once (snapshot+messages from same bytes)")
        self.assertIn("PruneLockError", src,
            "MCP treat_session must handle PruneLockError")
        self.assertIn("PruneConflictError", src,
            "MCP treat_session must handle PruneConflictError")


# ─── Overflow watcher cleanup in finally ─────────────────────────────────────

class TestOverflowWatcherCleanup(unittest.TestCase):
    def test_start_guard_has_finally_block_for_watcher(self):
        """The watcher.stop() must be in a finally so it fires on ALL exit
        paths (KeyboardInterrupt + 4 break paths)."""
        import inspect
        from cozempic.guard import start_guard
        src = inspect.getsource(start_guard)
        # Find the finally block near end
        self.assertIn("finally:", src)
        # Inside it, must reference overflow_watcher.stop()
        # Crude check: 'finally' followed by 'overflow_watcher' somewhere after
        finally_idx = src.rfind("finally:")
        tail = src[finally_idx:]
        self.assertIn("overflow_watcher", tail,
            "finally block must reference overflow_watcher")
        self.assertIn(".stop()", tail,
            "finally block must call .stop()")


if __name__ == "__main__":
    unittest.main()
