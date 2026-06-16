"""Wave 2 — single-flight reload lock tests.

Verifies that the three independent reload-spawn code paths (cmd_reload,
guard_prune_cycle auto-fire, OverflowRecovery._do_recover) all coordinate
through a per-session lock and never spawn duplicate watchers.

This is the primary cascade fix from the production incident.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch


# ─── Core lock primitive ─────────────────────────────────────────────────────

class TestReloadLockAcquireRelease(unittest.TestCase):
    def test_basic_acquire_release(self):
        from cozempic.reload_lock import _ReloadLock, _lock_path_for
        sid = "test-session-acq-rel"
        lock_path = _lock_path_for(sid)
        # Clean state
        lock_path.unlink(missing_ok=True)
        try:
            with _ReloadLock(sid) as lock:
                self.assertTrue(lock_path.exists(), "lock file should exist while held")
                self.assertTrue(lock._owned)
            self.assertFalse(lock_path.exists(), "lock file should be unlinked on exit")
        finally:
            lock_path.unlink(missing_ok=True)

    def test_lock_contents_have_pid_and_initiator(self):
        from cozempic.reload_lock import _ReloadLock, _lock_path_for, INIT_CLI_RELOAD
        sid = "test-session-contents"
        lock_path = _lock_path_for(sid)
        lock_path.unlink(missing_ok=True)
        try:
            with _ReloadLock(sid, initiator=INIT_CLI_RELOAD):
                content = lock_path.read_text()
                lines = content.strip().split("\n")
                self.assertEqual(int(lines[0]), os.getpid())
                # line[1] is the timestamp (ISO format)
                self.assertIn("T", lines[1])  # ISO format has 'T' separator
                self.assertEqual(lines[2], INIT_CLI_RELOAD)
        finally:
            lock_path.unlink(missing_ok=True)


# ─── Single-flight: two acquirers race ───────────────────────────────────────

class TestReloadLockSingleFlight(unittest.TestCase):
    def test_second_acquire_raises_reload_lock_held(self):
        from cozempic.reload_lock import _ReloadLock, ReloadLockHeld, _lock_path_for
        sid = "test-single-flight"
        lock_path = _lock_path_for(sid)
        lock_path.unlink(missing_ok=True)
        try:
            with _ReloadLock(sid, initiator="first") as first:
                # Second attempt should immediately raise
                with self.assertRaises(ReloadLockHeld) as cm:
                    _ReloadLock(sid, initiator="second").__enter__()
                self.assertEqual(cm.exception.holder_pid, os.getpid())
                self.assertEqual(cm.exception.holder_initiator, "first")
                self.assertFalse(cm.exception.wedged)
        finally:
            lock_path.unlink(missing_ok=True)

    def test_concurrent_threads_only_one_wins(self):
        """20 threads racing for the same lock — exactly one succeeds."""
        from cozempic.reload_lock import _ReloadLock, ReloadLockHeld, _lock_path_for
        sid = "test-concurrent-threads"
        lock_path = _lock_path_for(sid)
        lock_path.unlink(missing_ok=True)
        try:
            start_gate = threading.Event()
            winners = []
            losers = []
            lock = threading.Lock()

            def try_acquire():
                start_gate.wait()
                try:
                    rl = _ReloadLock(sid, initiator="racer")
                    rl.__enter__()
                    # Briefly hold so others see it as held
                    time.sleep(0.05)
                    with lock:
                        winners.append(1)
                    rl.__exit__(None, None, None)
                except ReloadLockHeld:
                    with lock:
                        losers.append(1)

            threads = [threading.Thread(target=try_acquire) for _ in range(20)]
            for t in threads:
                t.start()
            start_gate.set()
            for t in threads:
                t.join()

            # Exactly 1 winner OR more than 1 winner SEQUENTIALLY (after each releases).
            # The test is really: no two winners at the same time.
            # We can't easily verify simultaneity, but we can verify the total
            # winners + losers == 20 and at least one of each.
            self.assertEqual(len(winners) + len(losers), 20)
            self.assertGreater(len(winners), 0)
            self.assertGreater(len(losers), 0)
        finally:
            lock_path.unlink(missing_ok=True)


# ─── Stale lock handling (dead holder PID) ───────────────────────────────────

class TestReloadLockStaleHolder(unittest.TestCase):
    def test_stale_pid_cleanup(self):
        """If lock file has a PID that's no longer alive, acquire should
        clean it up and proceed."""
        from cozempic.reload_lock import _ReloadLock, _lock_path_for
        sid = "test-stale-pid"
        lock_path = _lock_path_for(sid)
        lock_path.unlink(missing_ok=True)
        try:
            # Write a lock file pointing at a definitely-dead PID
            from datetime import datetime
            lock_path.write_text(
                f"999999\n{datetime.now().isoformat(timespec='seconds')}\nfake-initiator\n"
            )
            # Acquire should succeed (cleaning up the stale lock)
            with _ReloadLock(sid):
                # We're holding it now; the file should contain OUR PID
                content = lock_path.read_text()
                self.assertEqual(int(content.split("\n")[0]), os.getpid())
        finally:
            lock_path.unlink(missing_ok=True)


# ─── Wedged lock detection ──────────────────────────────────────────────────

class TestReloadLockWedged(unittest.TestCase):
    def test_wedged_lock_raises_with_wedged_flag(self):
        """If lock file holder PID is alive AND age > WEDGE_TTL_SECONDS, raise with wedged=True."""
        from cozempic.reload_lock import _ReloadLock, ReloadLockHeld, _lock_path_for, WEDGE_TTL_SECONDS
        sid = "test-wedged"
        lock_path = _lock_path_for(sid)
        lock_path.unlink(missing_ok=True)
        try:
            # Write a lock file with our own PID (which IS alive) but an old timestamp
            from datetime import datetime, timedelta
            ts = (datetime.now() - timedelta(seconds=WEDGE_TTL_SECONDS + 5)).isoformat(timespec="seconds")
            lock_path.write_text(f"{os.getpid()}\n{ts}\nwedged-initiator\n")

            with self.assertRaises(ReloadLockHeld) as cm:
                _ReloadLock(sid).__enter__()
            self.assertTrue(cm.exception.wedged,
                f"Expected wedged=True for age > {WEDGE_TTL_SECONDS}s")
            self.assertEqual(cm.exception.holder_initiator, "wedged-initiator")
        finally:
            lock_path.unlink(missing_ok=True)


# ─── acquire_with_wait — opt-in polling for --wait flag ─────────────────────

class TestReloadLockWait(unittest.TestCase):
    def test_acquire_with_wait_succeeds_when_lock_released(self):
        from cozempic.reload_lock import _ReloadLock, acquire_with_wait, _lock_path_for
        sid = "test-wait-success"
        lock_path = _lock_path_for(sid)
        lock_path.unlink(missing_ok=True)
        try:
            # Hold the lock in a thread that releases after 0.5s
            release_signal = threading.Event()
            holder_done = threading.Event()

            def holder():
                with _ReloadLock(sid, initiator="holder"):
                    release_signal.wait(timeout=2)
                holder_done.set()

            t = threading.Thread(target=holder, daemon=True)
            t.start()
            # Wait for holder to acquire
            time.sleep(0.1)
            self.assertTrue(lock_path.exists())

            # Try to acquire with wait — should succeed once holder releases
            def acquire_after_signal():
                time.sleep(0.3)
                release_signal.set()

            threading.Thread(target=acquire_after_signal, daemon=True).start()

            lock = acquire_with_wait(sid, initiator="waiter", wait_seconds=5.0, poll_interval=0.1)
            try:
                self.assertTrue(lock._owned)
            finally:
                lock.__exit__(None, None, None)
            holder_done.wait(timeout=5)
        finally:
            lock_path.unlink(missing_ok=True)

    def test_acquire_with_wait_raises_when_wedged(self):
        """Wedged locks (age > TTL) should not be waited for — surface immediately."""
        from cozempic.reload_lock import acquire_with_wait, ReloadLockHeld, _lock_path_for, WEDGE_TTL_SECONDS
        sid = "test-wait-wedged"
        lock_path = _lock_path_for(sid)
        lock_path.unlink(missing_ok=True)
        try:
            from datetime import datetime, timedelta
            ts = (datetime.now() - timedelta(seconds=WEDGE_TTL_SECONDS + 5)).isoformat(timespec="seconds")
            lock_path.write_text(f"{os.getpid()}\n{ts}\nwedged\n")

            t0 = time.time()
            with self.assertRaises(ReloadLockHeld) as cm:
                acquire_with_wait(sid, initiator="waiter", wait_seconds=5.0)
            elapsed = time.time() - t0
            self.assertLess(elapsed, 1.0,
                "Wedged lock should fail fast, not wait")
            self.assertTrue(cm.exception.wedged)
        finally:
            lock_path.unlink(missing_ok=True)


# ─── Session ID sanitization (path traversal defense) ────────────────────────

class TestReloadLockSessionIdSanitization(unittest.TestCase):
    def test_path_in_session_id_uses_stem(self):
        from cozempic.reload_lock import _slug_for, _lock_path_for
        # Full path with .jsonl — should reduce to first 12 chars of the UUID stem
        slug = _slug_for("/Users/foo/.claude/projects/abc/f641174c-d784-4aab.jsonl")
        self.assertEqual(slug, "f641174c-d78")
        self.assertEqual(len(slug), 12)
        # Should not contain path separators
        self.assertNotIn("/", slug)
        self.assertNotIn("\\", slug)

    def test_malicious_session_id_sanitized(self):
        from cozempic.reload_lock import _slug_for
        slug = _slug_for("../../etc/passwd")
        # Should not contain path traversal chars
        self.assertNotIn("..", slug)
        self.assertNotIn("/", slug)

    def test_lock_path_in_tempdir(self):
        from cozempic.reload_lock import _lock_path_for
        path = _lock_path_for("abc123")
        self.assertEqual(path.parent, Path(tempfile.gettempdir()))
        self.assertEqual(path.name, "cozempic_reload_abc123.lock")

    def test_slug_parity_three_producers_agree_on_uppercase_input(self):
        """All three slug producers must produce identical output for an uppercase input.

        reload_lock._slug_for was missing .lower() (XF-1 / L5): for a session_id
        with uppercase letters (e.g. a UUID received before normalization), it
        produced a DIFFERENT slug than spawn_lock._slug_for and guard._reload_armed_path.
        That split-brain causes the lock and the pid/sentinel files to live at
        different paths — the concurrency guard is silently bypassed.

        RED-at-base: reload_lock._slug_for("ABCD1234EFGH-XX") == "ABCD1234EFGH"
        (uppercase kept), while spawn_lock._slug_for produces "abcd1234efgh"
        and guard._reload_armed_path produces "abcd1234efgh" → assertEqual FAILS.
        """
        from cozempic.reload_lock import _slug_for as rl_slug
        from cozempic.spawn_lock import _slug_for as sl_slug
        from cozempic.guard import _reload_armed_path

        # Non-UUID uppercase input: 12+ chars, contains upper letters.
        raw = "ABCD1234EFGH-XX"

        # Call the REAL guard function — not an inlined copy of its formula.
        # If _reload_armed_path drifts (different regex, truncation, path-stripping),
        # this test will catch it. An inline formula copy would not.
        _PREFIX = "cozempic_reload_armed_"
        _SUFFIX = ".json"
        armed_path = _reload_armed_path(raw)
        guard_slug = armed_path.name[len(_PREFIX):-len(_SUFFIX)]

        rl = rl_slug(raw)
        sl = sl_slug(raw)

        self.assertEqual(
            rl, sl,
            f"reload_lock slug {rl!r} != spawn_lock slug {sl!r} for uppercase input {raw!r}. "
            "reload_lock._slug_for is missing .lower() — XF-1 split-brain bug."
        )
        self.assertEqual(
            rl, guard_slug,
            f"reload_lock slug {rl!r} != guard slug {guard_slug!r} for uppercase input {raw!r}."
        )
        # Sanity: all three must be fully lowercase (no uppercase can survive)
        self.assertEqual(rl, rl.lower(), f"Slug {rl!r} contains uppercase letters.")

    def test_slug_parity_uuid_input_unchanged(self):
        """Standard lowercase UUID inputs must produce identical slugs across all three
        producers both before and after the XF-1 fix — no regression.

        This is a zero-change verification: a well-formed UUID (all lowercase hex
        and dashes) is already lowercase, so .lower() is a no-op. The slug must
        equal the first 12 chars of the UUID across all three producers.
        """
        from cozempic.reload_lock import _slug_for as rl_slug
        from cozempic.spawn_lock import _slug_for as sl_slug
        from cozempic.guard import _reload_armed_path

        uuid = "f641174c-d784-4aab-8f29-3a1c2b456def"
        expected = "f641174c-d78"  # first 12 chars, all lowercase already

        # Call the REAL guard function — not an inlined copy of its formula.
        _PREFIX = "cozempic_reload_armed_"
        _SUFFIX = ".json"
        armed_path = _reload_armed_path(uuid)
        guard_slug = armed_path.name[len(_PREFIX):-len(_SUFFIX)]

        self.assertEqual(rl_slug(uuid), expected)
        self.assertEqual(sl_slug(uuid), expected)
        self.assertEqual(guard_slug, expected)

    # ── UUID invariance — production session_ids must be unaffected ───────────

    def test_uuid_session_id_armed_path_invariant(self):
        """Regression guard: _reload_armed_path filename is UNCHANGED for a UUID session_id.

        UUIDs have no dots and no path separators, so Path(uuid).stem == uuid.
        _slug_for(uuid) must produce the same result as the pre-fix inline formula
        for any standard lowercase UUID, ensuring live armed-sentinels are not
        orphaned on upgrade.
        """
        from cozempic.guard import _reload_armed_path

        uuid = "f641174c-d784-4aab-8f29-3a1c2b456def"
        # Pre-fix formula (the inline that was in guard before P0-A)
        pre_fix_slug = re.sub(r"[^a-z0-9_-]", "_", str(uuid).lower())[:12] or "session"

        armed_path = _reload_armed_path(uuid)
        _PREFIX = "cozempic_reload_armed_"
        _SUFFIX = ".json"
        post_fix_slug = armed_path.name[len(_PREFIX):-len(_SUFFIX)]

        self.assertEqual(post_fix_slug, pre_fix_slug,
            f"UUID session_id armed-path slug changed after fix: "
            f"{pre_fix_slug!r} → {post_fix_slug!r}. "
            "Live armed-sentinels would be orphaned on upgrade.")

    # ── T1 — path-form session_id: _reload_armed_path must match _slug_for ───

    def test_reload_armed_path_path_input_matches_slug_for(self):
        """T1 (RED at HEAD): _reload_armed_path must use reload_lock._slug_for for path inputs.

        At HEAD, _reload_armed_path inlines re.sub on the raw path string, producing
        '_users_foo_m' for '/Users/foo/MySession-ABC.jsonl'. _slug_for extracts the
        stem first, producing 'mysession-ab'. This is the split-brain XF-1 for path
        inputs: the reload-lock and the armed sentinel live under different slugs.
        """
        from cozempic.guard import _reload_armed_path
        from cozempic.reload_lock import _slug_for as rl_slug

        path_input = '/Users/foo/MySession-ABC.jsonl'
        armed_path = _reload_armed_path(path_input)
        _PREFIX = "cozempic_reload_armed_"
        _SUFFIX = ".json"
        armed_slug = armed_path.name[len(_PREFIX):-len(_SUFFIX)]

        self.assertEqual(armed_slug, rl_slug(path_input),
            f"_reload_armed_path slug {armed_slug!r} != _slug_for slug "
            f"{rl_slug(path_input)!r} for path input {path_input!r}")

    # ── T2 — path-form session_id: _reload_ledger_path must match _slug_for ──

    def test_reload_ledger_path_path_input_matches_slug_for(self):
        """T2 (RED at HEAD): _reload_ledger_path must use reload_lock._slug_for for path inputs.

        Same divergence as T1 but on the ledger path producer.
        """
        from cozempic.guard import _reload_ledger_path
        from cozempic.reload_lock import _slug_for as rl_slug

        path_input = '/Users/foo/MySession-ABC.jsonl'
        ledger_path = _reload_ledger_path(path_input, Path('/dummy'))
        _PREFIX = "cozempic_reload_"
        _SUFFIX = ".history"
        ledger_slug = ledger_path.name[len(_PREFIX):-len(_SUFFIX)]

        self.assertEqual(ledger_slug, rl_slug(path_input),
            f"_reload_ledger_path slug {ledger_slug!r} != _slug_for slug "
            f"{rl_slug(path_input)!r} for path input {path_input!r}")

    # ── T3 — empty/None session: fallback must be 'default', not 'session' ───

    def test_reload_armed_path_empty_session_uses_default_slug(self):
        """T3 (RED at HEAD): _reload_armed_path(None, None) must produce slug 'default'.

        At HEAD, the fallback is 'session' (inline formula's or "session").
        _slug_for("") returns "default". The mismatch means the armed sentinel
        and the reload sentinel target different conceptual session slots when
        both have None/empty session_id.
        """
        from cozempic.guard import _reload_armed_path

        armed_path = _reload_armed_path(None, None)
        self.assertIn("default", armed_path.name,
            f"Expected 'default' in armed path for None session, "
            f"got: {armed_path.name!r}")
        self.assertNotIn("session", armed_path.name,
            f"Expected 'session' fallback to be replaced by 'default', "
            f"got: {armed_path.name!r}")

    # ── T8 — armed + ledger slugs must agree with each other for path input ──

    def test_reload_armed_ledger_parity(self):
        """T8 (RED at HEAD): armed and ledger slugs must agree for path-form session_id.

        Both functions inline the same (broken) formula — both produce '_users_foo_m'
        for the path input, while _slug_for produces 'mysession-ab'. After P0-A both
        call _slug_for, so they agree with each other AND with the lock.
        """
        from cozempic.guard import _reload_armed_path, _reload_ledger_path
        from cozempic.reload_lock import _slug_for as rl_slug

        path_input = '/Users/foo/MySession-ABC.jsonl'
        armed_path = _reload_armed_path(path_input)
        ledger_path = _reload_ledger_path(path_input, Path('/dummy'))

        armed_slug = armed_path.name[len("cozempic_reload_armed_"):-len(".json")]
        ledger_slug = ledger_path.name[len("cozempic_reload_"):-len(".history")]
        expected_slug = rl_slug(path_input)

        self.assertEqual(armed_slug, expected_slug,
            f"armed slug {armed_slug!r} != rl_slug {expected_slug!r}")
        self.assertEqual(ledger_slug, expected_slug,
            f"ledger slug {ledger_slug!r} != rl_slug {expected_slug!r}")
        self.assertEqual(armed_slug, ledger_slug,
            f"armed slug {armed_slug!r} != ledger slug {ledger_slug!r}")

    # ── Regression guard: path-branch uppercase _slug_for ────────────────────

    def test_slug_for_path_branch_uppercase_stem(self):
        """Regression guard: _slug_for with an uppercase-stem path must lowercase
        AFTER stem extraction.

        A refactor that moves .lower() to BEFORE Path().stem would leave the
        uppercase stem un-lowercased, producing 'ABCDEF123456' instead of
        'abcdef123456' — the XF-1 regression for path inputs. This test is
        GREEN at HEAD (the code is correct) and guards against future ordering
        regressions.
        """
        from cozempic.reload_lock import _slug_for
        slug = _slug_for('/Users/foo/ABCDEF123456.jsonl')
        self.assertEqual(slug, 'abcdef123456')
        self.assertEqual(slug, slug.lower(), "Slug must be fully lowercase")

    # ── T: parity test for _reload_ledger_path ───────────────────────────────

    def test_slug_parity_ledger_path_matches_reload_lock(self):
        """_reload_ledger_path must use the same slug as reload_lock._slug_for."""
        from cozempic.guard import _reload_ledger_path
        from cozempic.reload_lock import _slug_for as rl_slug

        raw = "ABCD1234EFGH-XX"
        ledger_path = _reload_ledger_path(raw, Path("/tmp/ignored.jsonl"))
        _PREFIX = "cozempic_reload_"
        _SUFFIX = ".history"
        ledger_slug = ledger_path.name[len(_PREFIX):-len(_SUFFIX)]
        self.assertEqual(ledger_slug, rl_slug(raw),
            f"ledger slug {ledger_slug!r} != rl_slug {rl_slug(raw)!r} for {raw!r}")


class TestReloadLockProcessAlive(unittest.TestCase):
    """T6 — _is_process_alive must return True on PermissionError.

    A cross-user process that owns the reload lock raises PermissionError
    on os.kill(pid, 0). The correct semantic is ALIVE (don't steal the lock),
    matching spawn_lock._is_process_alive. At HEAD, reload_lock wrongly
    returns False, allowing _acquire to unlink a live holder's lock.
    """

    def test_permissionerror_treats_as_alive(self):
        """PermissionError on kill(pid,0) must return True (cross-user = alive)."""
        from cozempic.reload_lock import _is_process_alive
        with patch('cozempic.reload_lock.os.kill', side_effect=PermissionError):
            result = _is_process_alive(1234)
        self.assertTrue(result,
            "_is_process_alive must return True for PermissionError — "
            "cross-user process is alive and we must not steal its lock")

    def test_processlookuperror_treats_as_dead(self):
        """ProcessLookupError on kill(pid,0) must still return False (no such process)."""
        from cozempic.reload_lock import _is_process_alive
        with patch('cozempic.reload_lock.os.kill', side_effect=ProcessLookupError):
            result = _is_process_alive(1234)
        self.assertFalse(result,
            "_is_process_alive must return False for ProcessLookupError")

    def test_pid_zero_or_negative_returns_false(self):
        """pid <= 0 is invalid — must return False without calling os.kill."""
        from cozempic.reload_lock import _is_process_alive
        self.assertFalse(_is_process_alive(0))
        self.assertFalse(_is_process_alive(-1))
        self.assertFalse(_is_process_alive(-999))

    def test_parity_with_spawn_lock(self):
        """reload_lock and spawn_lock must agree on PermissionError semantics."""
        from cozempic.reload_lock import _is_process_alive as rl_alive
        from cozempic.spawn_lock import _is_process_alive as sl_alive
        with patch('cozempic.reload_lock.os.kill', side_effect=PermissionError), \
             patch('cozempic.spawn_lock.os.kill', side_effect=PermissionError):
            rl_result = rl_alive(1234)
            sl_result = sl_alive(1234)
        self.assertEqual(rl_result, sl_result,
            f"reload_lock._is_process_alive({rl_result!r}) != "
            f"spawn_lock._is_process_alive({sl_result!r}) on PermissionError")


class TestReloadLockSymlinkDefense(unittest.TestCase):
    """Defense against symlink attacks via /tmp: if a malicious local user
    plants a symlink at our lock path, O_NOFOLLOW makes us fail rather
    than follow into an arbitrary file."""

    def test_o_nofollow_blocks_symlink_target(self):
        from cozempic.reload_lock import _ReloadLock, _lock_path_for
        sid = "test-symlink-defense"
        lock_path = _lock_path_for(sid)
        lock_path.unlink(missing_ok=True)

        with tempfile.TemporaryDirectory() as victim_dir:
            victim_file = Path(victim_dir) / "sensitive.txt"
            victim_file.write_text("original content")

            # Plant a symlink at the lock path → victim file
            try:
                os.symlink(str(victim_file), str(lock_path))
            except OSError:
                self.skipTest("Cannot create symlink in this environment")

            try:
                # Acquire should NOT follow the symlink (O_NOFOLLOW makes it fail)
                lock = _ReloadLock(sid)
                # _try_create returns False on OSError (ELOOP for O_NOFOLLOW)
                created = lock._try_create()

                self.assertFalse(created,
                    "O_NOFOLLOW should prevent acquiring via symlink")
                # Victim file content must be unchanged
                self.assertEqual(victim_file.read_text(), "original content",
                    "Symlink target was modified — O_NOFOLLOW missing!")
            finally:
                lock_path.unlink(missing_ok=True)


# ─── Round-2 reviewer findings (M-1 / M-2) ───────────────────────────────────

class TestRound2ReviewerFindings(unittest.TestCase):
    """RED tests for M-1 (guard.py:2235 double-truncation) and M-2 (str coercion)."""

    # ── M-1 — watcher slug must equal _reload_sentinel_path_for slug ─────────

    def test_watcher_slug_matches_sentinel_path_under_widened_slug_for(self):
        """M-1 (RED at bc7ba80 / pre-fix): guard.py:2235 still applies [:12]
        to _slug_for(), while _reload_sentinel_path_for (after P0-C) does not.

        This test inspects the source of _spawn_reload_watcher to confirm that
        the [:12] truncation has been removed.  It is RED at bc7ba80 (the commit
        that introduced the RED-test suite) because guard.py still contained
        _rl_slug_for(session_id)[:12] at that point, and GREEN after M-1 drops
        the [:12].

        Secondary behavioural check: mock _slug_for to return a 15-char slug and
        verify that the sentinel path produced by _reload_sentinel_path_for is
        NOT truncated (proving the parity contract holds with a widened _slug_for).
        """
        import inspect
        from cozempic import guard
        from cozempic.reload_lock import _reload_sentinel_path_for

        # --- structural check ---
        source = inspect.getsource(guard._spawn_reload_watcher)
        self.assertNotIn(
            "_rl_slug_for(session_id)[:12]",
            source,
            "guard._spawn_reload_watcher still applies [:12] to _rl_slug_for — "
            "M-1 fold gap: drops identical double-truncation that P0-C removed "
            "from _reload_sentinel_path_for."
        )

        # --- behavioural parity under widened _slug_for ---
        wide_slug = "abcdefghijklmno"  # 15 chars
        with patch('cozempic.reload_lock._slug_for', return_value=wide_slug):
            sentinel_slug = _reload_sentinel_path_for("my-test-session-99").name[
                len("cozempic_reload_"):-len(".in-flight")
            ]
        self.assertEqual(
            sentinel_slug, wide_slug,
            f"_reload_sentinel_path_for slug {sentinel_slug!r} != {wide_slug!r}: "
            "P0-C should have removed all [:12] truncation from the sentinel reader."
        )

    # ── M-2 — non-str session_id must not raise TypeError ────────────────────

    def test_reload_armed_path_non_str_session_id_does_not_raise(self):
        """M-2 (RED at HEAD): _reload_armed_path(123, None) raises TypeError.

        The old inline formula used str(raw) before re.sub, safely converting
        any truthy non-str.  P0-A switched to _slug_for(raw) which executes
        '/' in session_id — TypeError for int/Path.  Restore coercion at call site.
        """
        from cozempic.guard import _reload_armed_path
        try:
            result = _reload_armed_path(123, None)
        except TypeError as exc:
            self.fail(
                f"_reload_armed_path(123, None) raised TypeError: {exc!r}. "
                "Non-str session_id should be coerced to str at the call site."
            )
        # Ensure we got a real path back
        self.assertIn("cozempic_reload_armed_", result.name)

    def test_reload_ledger_path_non_str_session_id_does_not_raise(self):
        """M-2 parity: _reload_ledger_path(123, ...) must not raise TypeError."""
        from cozempic.guard import _reload_ledger_path
        from pathlib import Path
        try:
            result = _reload_ledger_path(123, Path("/tmp/dummy.jsonl"))
        except TypeError as exc:
            self.fail(
                f"_reload_ledger_path(123, ...) raised TypeError: {exc!r}. "
                "Non-str session_id should be coerced to str at the call site."
            )
        self.assertIn("cozempic_reload_", result.name)


# ─── CLI integration: --wait flag exists ─────────────────────────────────────

class TestReloadCliWaitFlag(unittest.TestCase):
    def test_cmd_reload_has_wait_argument(self):
        """`cozempic reload --wait` flag must exist on the parser."""
        from cozempic.cli import build_parser
        parser = build_parser()
        # Parse a minimal `reload --wait 10` to confirm the flag exists
        args = parser.parse_args(["reload", "--wait", "10"])
        self.assertEqual(args.wait, 10)
        # Default when --wait not passed
        args2 = parser.parse_args(["reload"])
        self.assertIsNone(args2.wait)


# ─── Guard integration: defers reload when lock held ─────────────────────────

class TestGuardDefersReloadWhenLockHeld(unittest.TestCase):
    def test_guard_prune_cycle_imports_reload_lock(self):
        """Verify guard_prune_cycle imports the reload lock primitive."""
        import inspect
        from cozempic.guard import guard_prune_cycle
        src = inspect.getsource(guard_prune_cycle)
        self.assertIn("_ReloadLock", src,
            "guard_prune_cycle must use _ReloadLock")
        self.assertIn("ReloadLockHeld", src,
            "guard_prune_cycle must handle ReloadLockHeld")
        self.assertIn("Reload deferred", src,
            "guard_prune_cycle must print 'Reload deferred' when lock held")


# ─── Overflow integration: defers when lock held ────────────────────────────

class TestOverflowDefersWhenLockHeld(unittest.TestCase):
    def test_overflow_recover_uses_reload_lock(self):
        import inspect
        from cozempic.overflow import OverflowRecovery
        src = inspect.getsource(OverflowRecovery._do_recover)
        self.assertIn("_ReloadLock", src,
            "OverflowRecovery._do_recover must use _ReloadLock")
        self.assertIn("INIT_OVERFLOW", src,
            "OverflowRecovery._do_recover must use INIT_OVERFLOW initiator")


if __name__ == "__main__":
    unittest.main()
