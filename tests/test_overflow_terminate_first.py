"""RED tests for P2 — overflow._do_recover terminate-first restructure (PLAN.md §2.4, §3.3)
and P5 — #106 race regression test (PLAN.md §2.7, §3.3).

P2 RED: _do_recover currently prunes BEFORE terminate (overflow.py:217–224 before 263–288).
P5 GREEN/REGRESSION: the append-aware save_messages path preserves or refuses sentinel lines.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d, separators=(",", ":")) + "\n")


def _minimal_session() -> list[dict]:
    return [
        {"type": "user", "uuid": "u1", "parentUuid": None,
         "message": {"role": "user", "content": "hello " * 500}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "message": {"role": "assistant",
                     "content": [{"type": "text", "text": "world " * 500}]}},
        {"type": "user", "uuid": "u2", "parentUuid": "a1",
         "message": {"role": "user", "content": "again " * 500}},
        {"type": "assistant", "uuid": "a2", "parentUuid": "u2",
         "message": {"role": "assistant",
                     "content": [{"type": "text", "text": "yes " * 500}]}},
        {"type": "ai-title", "uuid": "t1", "parentUuid": "a2",
         "title": "Test"},
        {"type": "last-prompt", "uuid": "lp1", "parentUuid": "a2",
         "text": "again"},
        {"type": "permission-mode", "uuid": "pm1", "parentUuid": "a2",
         "mode": "default"},
    ]


FAKE_PID = 77771


class TestOverflowTerminateFirst(unittest.TestCase):
    """overflow._do_recover must terminate Claude BEFORE calling guard_prune_cycle.

    RED at current HEAD: _do_recover calls guard_prune_cycle (which does save_messages)
    BEFORE calling _terminate_and_resume — the classic #106 race.
    """

    def setUp(self):
        self._tmpdir_obj = TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir_obj.name)
        self.session_path = self.tmpdir / "session.jsonl"
        _write_jsonl(self.session_path, _minimal_session())
        self.session_id = hashlib.md5(b"test").hexdigest()

        from cozempic.overflow import CircuitBreaker
        self.breaker = CircuitBreaker(
            session_id=self.session_id,
            max_recoveries=3,
            window_seconds=60,
        )
        self.breaker.reset()

    def tearDown(self):
        self.breaker.reset()
        self._tmpdir_obj.cleanup()

    def _make_recovery(self, claude_pid=FAKE_PID):
        from cozempic.overflow import OverflowRecovery
        return OverflowRecovery(
            session_path=self.session_path,
            session_id=self.session_id,
            cwd=str(self.tmpdir),
            breaker=self.breaker,
            danger_threshold_mb=0.001,  # tiny threshold so session is always "large"
            claude_pid=claude_pid,
        )

    def test_overflow_terminates_before_prune(self):
        """_terminate_claude must be called BEFORE guard_prune_cycle.

        RED: today's code calls guard_prune_cycle at overflow.py:217, then
        _terminate_and_resume at overflow.py:273. The fix reverses this.
        """
        call_order: list[str] = []

        def fake_terminate(pid, **kw):
            call_order.append("terminate")
            return "TERMINATED"

        def fake_gpc(**kw):
            call_order.append("prune")
            return {
                "saved_mb": 1.0, "original_tokens": 50_000, "final_tokens": 30_000,
                "team_name": None, "team_messages": 0,
                "checkpoint_path": None, "backup_path": None, "reloading": False,
            }

        recovery = self._make_recovery()

        with patch("cozempic.guard.guard_prune_cycle", side_effect=fake_gpc), \
             patch("cozempic.guard._terminate_claude", side_effect=fake_terminate), \
             patch("cozempic.guard._resume_claude"), \
             patch("cozempic.session.find_claude_pid", return_value=FAKE_PID), \
             patch("cozempic.guard.checkpoint_team"):
            recovery._do_recover()

        terminate_idx = next((i for i, e in enumerate(call_order) if e == "terminate"), None)
        prune_idx = next((i for i, e in enumerate(call_order) if e == "prune"), None)

        self.assertIsNotNone(
            terminate_idx,
            f"_terminate_claude was not called. Order: {call_order}. "
            f"P2 not implemented: _do_recover still uses old guard_prune_cycle-then-terminate order.",
        )
        self.assertIsNotNone(prune_idx, f"guard_prune_cycle was not called. Order: {call_order}.")
        self.assertLess(
            terminate_idx, prune_idx,
            msg=(
                f"_terminate_claude must be called BEFORE guard_prune_cycle. "
                f"Got order: {call_order}. "
                f"Current code prunes (os.replace on live file) before terminate — race #106."
            ),
        )

    def test_overflow_failed_to_die_aborts(self):
        """_terminate_claude returns FAILED_TO_DIE → guard_prune_cycle NOT called.

        Do NOT os.replace a still-live session file. Abort recovery.
        """
        prune_called = []

        def fake_terminate(pid, **kw):
            return "FAILED_TO_DIE"

        recovery = self._make_recovery()

        with patch("cozempic.guard.guard_prune_cycle",
                   side_effect=lambda **kw: prune_called.append(True) or {}), \
             patch("cozempic.guard._terminate_claude", side_effect=fake_terminate), \
             patch("cozempic.session.find_claude_pid", return_value=FAKE_PID), \
             patch("cozempic.guard.checkpoint_team"):
            recovery._do_recover()

        self.assertEqual(
            prune_called, [],
            msg=(
                f"guard_prune_cycle must NOT be called when _terminate_claude returns "
                f"FAILED_TO_DIE. Cannot os.replace a live JSONL. prune_called={prune_called}."
            ),
        )

    def test_overflow_already_gone_prune_no_resume(self):
        """_terminate_claude returns ALREADY_GONE → prune runs, _resume_claude NOT called.

        Anti-resurrection: file is quiesced (dead Claude), safe to prune.
        But do NOT resume a session the user intentionally closed.

        RED: current code doesn't use split functions.
        """
        resume_called = []

        def fake_gpc(**kw):
            return {
                "saved_mb": 1.0, "original_tokens": 50_000, "final_tokens": 30_000,
                "team_name": None, "team_messages": 0,
                "checkpoint_path": None, "backup_path": None, "reloading": False,
            }

        recovery = self._make_recovery()

        with patch("cozempic.guard.guard_prune_cycle", side_effect=fake_gpc), \
             patch("cozempic.guard._terminate_claude", return_value="ALREADY_GONE"), \
             patch("cozempic.guard._resume_claude",
                   side_effect=lambda *a, **kw: resume_called.append(True)), \
             patch("cozempic.session.find_claude_pid", return_value=FAKE_PID), \
             patch("cozempic.guard.checkpoint_team"):
            recovery._do_recover()

        self.assertEqual(
            resume_called, [],
            msg=(
                f"_resume_claude must NOT be called when _terminate_claude returns "
                f"ALREADY_GONE (anti-resurrection). resume_called={resume_called}."
            ),
        )


class TestRace106Regression(unittest.TestCase):
    """P5 — #106 race: append after classify must be preserved or refused.

    save_messages has append-aware logic: if Claude appended lines after our
    snapshot but before our save, the output must either:
    (a) Merge the new lines into the output (preserve), OR
    (b) Raise PruneConflictError (refuse the swap).

    It must NEVER silently drop the appended lines.

    This test acts as a regression guard that P1/P2 changes don't break the
    append-aware path in save_messages.
    """

    def _write_jsonl(self, path: Path, lines: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for d in lines:
                f.write(json.dumps(d, separators=(",", ":")) + "\n")

    def test_106_append_after_classify_preserved_or_refused(self):
        """Sentinel line appended after snapshot must survive in output or cause a refusal.

        Setup:
        1. Write initial JSONL.
        2. Take snapshot.
        3. Append sentinel line (simulating Claude writing mid-prune).
        4. Call save_messages(snapshot=snapshot, ...).

        Assertion: Either PruneConflictError raised OR output file contains sentinel.
        The sentinel is NEVER silently dropped.
        """
        from cozempic.session import save_messages, snapshot_session  # type: ignore
        from cozempic.session import PruneConflictError  # type: ignore

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.jsonl"

            original_lines = _minimal_session()
            self._write_jsonl(path, original_lines)

            # Take snapshot BEFORE append
            snap = snapshot_session(path)

            # Simulate Claude appending a new line mid-prune
            sentinel = {"type": "user", "uuid": "sentinel-uuid",
                        "parentUuid": "a2",
                        "message": {"role": "user", "content": "SENTINEL-LINE"}}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(sentinel, separators=(",", ":")) + "\n")

            # Pruned messages (same as original — zero-removal prune)
            from cozempic.session import load_messages  # type: ignore
            # Load a version of messages that does NOT include the sentinel
            # (as if we loaded before the append)
            msgs_without_sentinel = [
                (i, m, len(json.dumps(m)))
                for i, m in enumerate(original_lines)
            ]

            # Attempt to save the pruned version that excludes the sentinel
            raised_conflict = False
            try:
                save_messages(
                    path,
                    msgs_without_sentinel,
                    create_backup=True,
                    snapshot=snap,
                    messages_before_prune=msgs_without_sentinel,
                )
            except PruneConflictError:
                raised_conflict = True

            if not raised_conflict:
                # If no conflict raised, verify the sentinel was preserved
                content = path.read_text(encoding="utf-8")
                self.assertIn(
                    "SENTINEL-LINE", content,
                    msg=(
                        "save_messages neither raised PruneConflictError nor preserved "
                        "the sentinel line appended after the snapshot. "
                        "The sentinel was silently dropped — #106 race regression."
                    ),
                )
            # If raised_conflict=True, the test passes — conflict detection worked.


if __name__ == "__main__":
    unittest.main()
