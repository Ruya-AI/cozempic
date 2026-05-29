"""RED tests for P1 — _terminate_claude / _resume_claude split + guard_prune_cycle
terminate-first ordering (PLAN.md §2.3, §3.2).

All tests in this module MUST FAIL at current HEAD because:
- _terminate_claude and _resume_claude do not yet exist.
- guard_prune_cycle saves BEFORE calling _terminate_and_resume (wrong order).
- FAILED_TO_DIE path does not exist.
- ALREADY_GONE path does not exist.
"""

from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch, call


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d, separators=(",", ":")) + "\n")


def _set_mtime(path: Path, seconds_ago: float) -> None:
    now = time.time()
    target = now - seconds_ago
    os.utime(path, (target, target))


def _minimal_session() -> list[dict]:
    """Minimal valid session for guard_prune_cycle."""
    return [
        {"type": "user", "uuid": "u1", "parentUuid": None,
         "message": {"role": "user", "content": "hello " * 2000}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "message": {"role": "assistant",
                     "content": [{"type": "text", "text": "world " * 2000}]}},
        {"type": "user", "uuid": "u2", "parentUuid": "a1",
         "message": {"role": "user", "content": "again " * 2000}},
        {"type": "assistant", "uuid": "a2", "parentUuid": "u2",
         "message": {"role": "assistant",
                     "content": [{"type": "text", "text": "yes " * 2000}]}},
        {"type": "ai-title", "uuid": "t1", "parentUuid": "a2",
         "title": "Test session"},
        {"type": "last-prompt", "uuid": "lp1", "parentUuid": "a2",
         "text": "again"},
        {"type": "permission-mode", "uuid": "pm1", "parentUuid": "a2",
         "mode": "default"},
    ]


FAKE_PID = 99991


# ─── TestTerminateClaudeSplit — the new _terminate_claude function ─────────────

class TestTerminateClaudeSplit(unittest.TestCase):
    """_terminate_claude must exist and return the 4-state string.

    RED at current HEAD: _terminate_claude is not defined in cozempic.guard.
    """

    def test_terminate_claude_returns_terminated_on_live_process(self):
        """Mock live Claude (pid_is_alive=True, identity=True, is_claude=True).
        _terminate_clause must return 'TERMINATED' and NOT call any resume
        send-keys or _spawn_reload_watcher.
        """
        from cozempic.guard import _terminate_claude  # type: ignore  # RED: ImportError

        spawn_called = []
        with patch("cozempic.guard._pid_is_alive", return_value=True), \
             patch("cozempic.guard._pid_identity_match", return_value=True), \
             patch("cozempic.guard._is_claude_process", return_value=True), \
             patch("cozempic.guard._detect_terminal_env", return_value="plain"), \
             patch("cozempic.guard._wait_for_exit", return_value=True), \
             patch("cozempic.guard._spawn_reload_watcher",
                   side_effect=lambda *a, **kw: spawn_called.append(True)), \
             patch("os.kill"):
            status = _terminate_claude(FAKE_PID, session_id="s1")

        self.assertEqual(
            status, "TERMINATED",
            f"Expected 'TERMINATED' for live process that exits; got {status!r}.",
        )
        self.assertEqual(
            spawn_called, [],
            "_terminate_claude must NOT call _spawn_reload_watcher — resume is for _resume_claude.",
        )

    def test_terminate_claude_returns_already_gone_on_dead_pid(self):
        """_pid_is_alive returns False → _terminate_claude returns 'ALREADY_GONE'.
        No kill attempted, no sentinel written.
        """
        from cozempic.guard import _terminate_claude  # type: ignore

        with patch("cozempic.guard._pid_is_alive", return_value=False), \
             patch("os.kill") as mock_kill:
            status = _terminate_claude(FAKE_PID, session_id="s1")

        self.assertEqual(status, "ALREADY_GONE",
                         f"Expected 'ALREADY_GONE' for dead pid; got {status!r}.")
        mock_kill.assert_not_called()

    def test_terminate_claude_returns_failed_to_die_when_kill_ineffective(self):
        """_wait_for_exit always returns False → process never exits → 'FAILED_TO_DIE'.
        No sentinel written (no save on live file).
        """
        from cozempic.guard import _terminate_claude  # type: ignore

        with patch("cozempic.guard._pid_is_alive", return_value=True), \
             patch("cozempic.guard._pid_identity_match", return_value=True), \
             patch("cozempic.guard._is_claude_process", return_value=True), \
             patch("cozempic.guard._detect_terminal_env", return_value="plain"), \
             patch("cozempic.guard._wait_for_exit", return_value=False), \
             patch("cozempic.guard.write_reload_sentinel") as mock_sentinel, \
             patch("os.kill"):
            status = _terminate_claude(FAKE_PID, session_id="s1")

        self.assertEqual(status, "FAILED_TO_DIE",
                         f"Expected 'FAILED_TO_DIE' when process won't exit; got {status!r}.")
        mock_sentinel.assert_not_called()

    def test_terminate_claude_returns_skipped_ssh_in_ssh_env(self):
        """SSH environment → _terminate_claude returns 'SKIPPED_SSH'. No kill."""
        from cozempic.guard import _terminate_claude  # type: ignore

        with patch("cozempic.guard._detect_terminal_env", return_value="ssh"), \
             patch("os.kill") as mock_kill:
            status = _terminate_claude(FAKE_PID, session_id="s1")

        self.assertEqual(status, "SKIPPED_SSH",
                         f"Expected 'SKIPPED_SSH' in ssh env; got {status!r}.")
        mock_kill.assert_not_called()


# ─── TestResumeClaude — the new _resume_claude function ───────────────────────

class TestResumeClaude(unittest.TestCase):
    """_resume_claude must exist and handle the resume-only path.

    RED at current HEAD: _resume_claude is not defined in cozempic.guard.
    """

    def test_resume_claude_plain_calls_spawn_reload_watcher(self):
        """Plain terminal: _resume_claude must call _spawn_reload_watcher."""
        from cozempic.guard import _resume_claude  # type: ignore  # RED

        spawn_called = []
        with patch("cozempic.guard._detect_terminal_env", return_value="plain"), \
             patch("cozempic.guard._spawn_reload_watcher",
                   side_effect=lambda *a, **kw: spawn_called.append(True)), \
             patch("cozempic.guard.write_reload_sentinel"):
            _resume_claude(FAKE_PID, "/tmp/proj", session_id="s1")

        self.assertTrue(spawn_called, "_resume_claude must call _spawn_reload_watcher for plain terminal.")

    def test_resume_claude_flags_captured_before_kill(self):
        """R-2: original_flags must survive even when PID is dead at resume time.

        Simulates the scenario: _detect_claude_flags(pid) returns empty after
        death, but the flags captured BEFORE the kill are threaded through.
        _resume_claude must produce the correct resume_cmd with the original flags.
        """
        from cozempic.guard import _terminate_claude, _resume_claude  # type: ignore

        captured_cmds: list[str] = []

        def fake_spawn(pid, project_dir, session_id=None, original_flags=None, **kw):
            captured_cmds.append(str(original_flags or ""))

        with patch("cozempic.guard._pid_is_alive", return_value=True), \
             patch("cozempic.guard._pid_identity_match", return_value=True), \
             patch("cozempic.guard._is_claude_process", return_value=True), \
             patch("cozempic.guard._detect_terminal_env", return_value="plain"), \
             patch("cozempic.guard._wait_for_exit", return_value=True), \
             patch("cozempic.guard._detect_claude_flags",
                   side_effect=lambda pid: "--dangerously-skip-permissions" if pid == FAKE_PID else ""), \
             patch("cozempic.guard._spawn_reload_watcher", side_effect=fake_spawn), \
             patch("cozempic.guard.write_reload_sentinel"), \
             patch("os.kill"):
            status = _terminate_claude(FAKE_PID, session_id="s1")
            # After _terminate_claude, PID is "dead" — _detect_claude_flags now returns ""
            with patch("cozempic.guard._detect_claude_flags", return_value=""):
                _resume_claude(FAKE_PID, "/tmp/proj", session_id="s1",
                               original_flags="--dangerously-skip-permissions")

        self.assertTrue(
            any("--dangerously-skip-permissions" in c for c in captured_cmds),
            msg=(
                f"R-2: resume_cmd must preserve original_flags even after process death. "
                f"Captured spawn args: {captured_cmds}. "
                f"_detect_claude_flags returns '' for dead PIDs — flags must be captured "
                f"inside _terminate_claude BEFORE the kill and threaded to _resume_claude."
            ),
        )


# ─── TestTerminateFirstOrdering — guard_prune_cycle must terminate before save ─

class TestTerminateFirstOrdering(unittest.TestCase):
    """guard_prune_cycle(auto_reload=True) must: terminate → save → resume.

    RED at current HEAD: prune/save happens BEFORE _terminate_and_resume.
    """

    def setUp(self):
        self._tmpdir_obj = TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir_obj.name)
        self.session_path = self.tmpdir / "session.jsonl"
        _write_jsonl(self.session_path, _minimal_session())
        _set_mtime(self.session_path, seconds_ago=25 * 3600)  # idle

    def tearDown(self):
        self._tmpdir_obj.cleanup()

    def _base_patches(self):
        """Common patches for guard_prune_cycle tests."""
        from cozempic.team import TeamState
        fake_state = MagicMock(spec=TeamState)
        fake_state.is_empty.return_value = True
        fake_state.team_name = None
        fake_state.message_count = 0

        orig_bytes = 100_000
        pruned_bytes = 70_000  # 30% savings — above _MIN_PRUNE_RATIO
        fake_orig = [(0, {"type": "user"}, orig_bytes)]
        fake_pruned = [(0, {"type": "user"}, pruned_bytes)]

        return fake_orig, fake_pruned, fake_state

    def test_terminate_called_before_save_in_guard_prune_cycle(self):
        """_terminate_claude must be called BEFORE save_messages.

        RED: today's code calls save_messages inside _PruneLock, then calls
        _terminate_and_resume after the lock is released. The new design must
        terminate first (kill + wait), THEN save on the quiesced file.
        """
        from cozempic.guard import guard_prune_cycle  # type: ignore

        call_order: list[str] = []
        fake_orig, fake_pruned, fake_state = self._base_patches()

        def fake_terminate(pid, **kw):
            call_order.append("terminate")
            return "TERMINATED"

        def fake_save(*a, **kw):
            call_order.append("save")
            return None

        with patch("cozempic.guard.load_messages", return_value=fake_orig), \
             patch("cozempic.guard.prune_with_team_protect",
                   return_value=(fake_pruned, {}, fake_state)), \
             patch("cozempic.guard.save_messages", side_effect=fake_save), \
             patch("cozempic.guard.snapshot_session", return_value=MagicMock()), \
             patch("cozempic.guard._terminate_claude", side_effect=fake_terminate), \
             patch("cozempic.guard._resume_claude"), \
             patch("cozempic.tokens.estimate_session_tokens",
                   return_value=MagicMock(total=500_000)), \
             patch("cozempic.tokens.calibrate_ratio", return_value=0.5):

            guard_prune_cycle(
                session_path=self.session_path,
                rx_name="standard",
                config=None,
                auto_reload=True,
                force=True,
                cwd=str(self.tmpdir),
                session_id="s1",
                claude_pid=FAKE_PID,
            )

        terminate_idx = next((i for i, e in enumerate(call_order) if e == "terminate"), None)
        save_idx = next((i for i, e in enumerate(call_order) if e == "save"), None)

        self.assertIsNotNone(
            terminate_idx,
            f"_terminate_claude was not called. Order: {call_order}. "
            f"P1 not implemented: guard_prune_cycle still uses _terminate_and_resume.",
        )
        self.assertIsNotNone(
            save_idx,
            f"save_messages was not called. Order: {call_order}.",
        )
        self.assertLess(
            terminate_idx, save_idx,
            msg=(
                f"_terminate_claude must be called BEFORE save_messages. "
                f"Got order: {call_order}. "
                f"Current code saves first, then terminates — race condition #106."
            ),
        )

    def test_resume_called_after_save_in_guard_prune_cycle(self):
        """_resume_claude must be called AFTER save_messages.

        RED: current code doesn't call _resume_claude at all.
        """
        from cozempic.guard import guard_prune_cycle  # type: ignore

        call_order: list[str] = []
        fake_orig, fake_pruned, fake_state = self._base_patches()

        def fake_save(*a, **kw):
            call_order.append("save")
            return None

        def fake_resume(*a, **kw):
            call_order.append("resume")

        with patch("cozempic.guard.load_messages", return_value=fake_orig), \
             patch("cozempic.guard.prune_with_team_protect",
                   return_value=(fake_pruned, {}, fake_state)), \
             patch("cozempic.guard.save_messages", side_effect=fake_save), \
             patch("cozempic.guard.snapshot_session", return_value=MagicMock()), \
             patch("cozempic.guard._terminate_claude", return_value="TERMINATED"), \
             patch("cozempic.guard._resume_claude", side_effect=fake_resume), \
             patch("cozempic.tokens.estimate_session_tokens",
                   return_value=MagicMock(total=500_000)), \
             patch("cozempic.tokens.calibrate_ratio", return_value=0.5):

            guard_prune_cycle(
                session_path=self.session_path,
                rx_name="standard",
                config=None,
                auto_reload=True,
                force=True,
                cwd=str(self.tmpdir),
                session_id="s1",
                claude_pid=FAKE_PID,
            )

        save_idx = next((i for i, e in enumerate(call_order) if e == "save"), None)
        resume_idx = next((i for i, e in enumerate(call_order) if e == "resume"), None)

        self.assertIsNotNone(
            resume_idx,
            f"_resume_claude was not called. Order: {call_order}. "
            f"P1 not implemented: guard_prune_cycle still uses _terminate_and_resume.",
        )
        self.assertIsNotNone(save_idx, f"save_messages not called. Order: {call_order}.")
        self.assertLess(
            save_idx, resume_idx,
            msg=(
                f"save_messages must be called BEFORE _resume_claude. "
                f"Got order: {call_order}."
            ),
        )

    def test_failed_to_die_aborts_save(self):
        """FAILED_TO_DIE → save_messages NOT called, result has terminate_failed=True.

        RED: current code doesn't have FAILED_TO_DIE path.
        """
        from cozempic.guard import guard_prune_cycle  # type: ignore

        save_called = []
        fake_orig, fake_pruned, fake_state = self._base_patches()

        with patch("cozempic.guard.load_messages", return_value=fake_orig), \
             patch("cozempic.guard.prune_with_team_protect",
                   return_value=(fake_pruned, {}, fake_state)), \
             patch("cozempic.guard.save_messages",
                   side_effect=lambda *a, **kw: save_called.append(True)), \
             patch("cozempic.guard.snapshot_session", return_value=MagicMock()), \
             patch("cozempic.guard._terminate_claude", return_value="FAILED_TO_DIE"), \
             patch("cozempic.tokens.estimate_session_tokens",
                   return_value=MagicMock(total=500_000)), \
             patch("cozempic.tokens.calibrate_ratio", return_value=0.5):

            result = guard_prune_cycle(
                session_path=self.session_path,
                rx_name="standard",
                config=None,
                auto_reload=True,
                force=True,
                cwd=str(self.tmpdir),
                session_id="s1",
                claude_pid=FAKE_PID,
            )

        self.assertEqual(
            save_called, [],
            msg=(
                f"save_messages MUST NOT be called when _terminate_claude returns "
                f"'FAILED_TO_DIE' — do NOT os.replace a live file (race #106). "
                f"save_called={save_called}."
            ),
        )
        self.assertTrue(
            result.get("terminate_failed"),
            msg=(
                f"Result must include terminate_failed=True on FAILED_TO_DIE. "
                f"Got result={result}."
            ),
        )

    def test_already_gone_saves_without_resume(self):
        """ALREADY_GONE → save_messages called, _resume_claude NOT called.

        Anti-resurrection: file is quiesced (dead Claude), safe to save.
        Resuming a session the user intentionally closed is forbidden.

        RED: current code doesn't have ALREADY_GONE path.
        """
        from cozempic.guard import guard_prune_cycle  # type: ignore

        save_called = []
        resume_called = []
        fake_orig, fake_pruned, fake_state = self._base_patches()

        with patch("cozempic.guard.load_messages", return_value=fake_orig), \
             patch("cozempic.guard.prune_with_team_protect",
                   return_value=(fake_pruned, {}, fake_state)), \
             patch("cozempic.guard.save_messages",
                   side_effect=lambda *a, **kw: save_called.append(True) or None), \
             patch("cozempic.guard.snapshot_session", return_value=MagicMock()), \
             patch("cozempic.guard._terminate_claude", return_value="ALREADY_GONE"), \
             patch("cozempic.guard._resume_claude",
                   side_effect=lambda *a, **kw: resume_called.append(True)), \
             patch("cozempic.tokens.estimate_session_tokens",
                   return_value=MagicMock(total=500_000)), \
             patch("cozempic.tokens.calibrate_ratio", return_value=0.5):

            result = guard_prune_cycle(
                session_path=self.session_path,
                rx_name="standard",
                config=None,
                auto_reload=True,
                force=True,
                cwd=str(self.tmpdir),
                session_id="s1",
                claude_pid=FAKE_PID,
            )

        self.assertTrue(
            save_called,
            msg=(
                f"save_messages must be called on ALREADY_GONE path "
                f"(file is quiesced — safe to replace). save_called={save_called}."
            ),
        )
        self.assertEqual(
            resume_called, [],
            msg=(
                f"_resume_claude must NOT be called on ALREADY_GONE "
                f"(anti-resurrection). resume_called={resume_called}."
            ),
        )
        self.assertFalse(
            result.get("reloading", True),
            msg=f"reloading must be False on ALREADY_GONE. Result={result}.",
        )


# ─── TestTerminateAndResumePreserved — backward compat ────────────────────────

class TestTerminateAndResumePreserved(unittest.TestCase):
    """_terminate_and_resume must remain importable and compose the split functions.

    After P1, it is a thin wrapper: _terminate_claude → (if TERMINATED) _resume_claude.
    Existing callers (cmd_reload, tests) must be unaffected.
    """

    def test_terminate_and_resume_still_importable(self):
        """_terminate_and_resume must be importable from cozempic.guard."""
        from cozempic.guard import _terminate_and_resume  # type: ignore  # noqa: F401
        self.assertTrue(callable(_terminate_and_resume))

    def test_terminate_and_resume_calls_resume_on_terminated(self):
        """Composition check: TERMINATED → _resume_claude called.

        After P1 refactor _terminate_and_resume is: _terminate_claude + _resume_claude.
        If _terminate_claude returns TERMINATED, _resume_claude must be called.
        """
        from cozempic.guard import _terminate_and_resume  # type: ignore

        resume_called = []
        with patch("cozempic.guard._terminate_claude", return_value="TERMINATED"), \
             patch("cozempic.guard._resume_claude",
                   side_effect=lambda *a, **kw: resume_called.append(True)):
            _terminate_and_resume(FAKE_PID, "/tmp/proj", session_id="s1")

        self.assertTrue(
            resume_called,
            "_terminate_and_resume must call _resume_claude when _terminate_claude returns TERMINATED.",
        )


# ─── TestTerminateKCounterSkip — R-1 FAILED_TO_DIE does not advance K ─────────

class TestTerminateKCounterSkip(unittest.TestCase):
    """R-1: terminate_failed=True must be excluded from K-counter advance.

    guard.py:683 checks `active_session_refused` and `validation_failed` to
    skip K advance. `terminate_failed` must be added to that skip-list.
    """

    def test_terminate_failed_key_in_result_when_failed_to_die(self):
        """guard_prune_cycle returns terminate_failed=True on FAILED_TO_DIE.

        This key is read by start_guard's K-counter logic (guard.py:683).
        The key's presence is what the carve-out checks.
        """
        from cozempic.guard import guard_prune_cycle  # type: ignore

        from cozempic.team import TeamState
        fake_state = MagicMock(spec=TeamState)
        fake_state.is_empty.return_value = True
        fake_state.team_name = None
        fake_state.message_count = 0

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.jsonl"
            _write_jsonl(path, _minimal_session())
            _set_mtime(path, seconds_ago=25 * 3600)

            with patch("cozempic.guard.load_messages",
                       return_value=[(0, {"type": "user"}, 100_000)]), \
                 patch("cozempic.guard.prune_with_team_protect",
                       return_value=([(0, {"type": "user"}, 70_000)], {}, fake_state)), \
                 patch("cozempic.guard.save_messages", return_value=None), \
                 patch("cozempic.guard.snapshot_session", return_value=MagicMock()), \
                 patch("cozempic.guard._terminate_claude", return_value="FAILED_TO_DIE"), \
                 patch("cozempic.tokens.estimate_session_tokens",
                       return_value=MagicMock(total=500_000)), \
                 patch("cozempic.tokens.calibrate_ratio", return_value=0.5):
                result = guard_prune_cycle(
                    session_path=path,
                    rx_name="standard",
                    auto_reload=True,
                    force=True,
                    cwd=tmpdir,
                    session_id="s1",
                    claude_pid=FAKE_PID,
                )

        self.assertTrue(
            result.get("terminate_failed"),
            msg=(
                f"R-1: result must contain terminate_failed=True on FAILED_TO_DIE "
                f"so start_guard can skip K-counter advance. Got result={result}."
            ),
        )


if __name__ == "__main__":
    unittest.main()
