"""Wrong-session detection RCA + regression (2026-06-10).

`/cozempic reload` resolved the WRONG session: the active session (f464a40c) lived
under project dir ``-Users-ruya`` while the invocation's cwd mapped to
``-Users-ruya-Documents-Advisor-Cozempic``, whose most-recent session was a STALE
one (f641174c). ``find_current_session`` fell through Strategy 1 (lsof on open
``.claude/tasks/`` dirs — empty when no agent team is running) to Strategy 3 (cwd
slug → most-recent in that project dir) and picked the stale session.

Root cause: the manual CLI path inferred the session from cwd, while Claude Code
itself knows the active session as ``transcript_path`` (handed to the SessionStart
hook). The durable fix records that transcript keyed by the LIVE Claude PID — one
claude process == one session, which disambiguates precisely where cwd cannot —
and ``find_current_session`` consults it FIRST.

These tests pin: (1) the bug shape (cwd≠project picks the wrong session without the
active record), (2) the fix (with an active record for the live pid, the RIGHT
session wins regardless of cwd), (3) staleness guards (dead pid / missing transcript
fall through), and (4) the record round-trips through the store.
"""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from cozempic import session as S


def _write_session(proj_dir: Path, sid: str, body: str = '{"type":"user"}\n') -> Path:
    proj_dir.mkdir(parents=True, exist_ok=True)
    p = proj_dir / f"{sid}.jsonl"
    p.write_text(body, encoding="utf-8")
    return p


class _Base(unittest.TestCase):
    def setUp(self):
        self._td = TemporaryDirectory()
        self.root = Path(self._td.name)
        self.projects = self.root / "projects"
        self.projects.mkdir(parents=True)
        # Patch the claude dir so both session discovery and the active-store
        # write under our temp root.
        self._patches = [
            mock.patch.object(S, "get_claude_dir", lambda: self.root),
            mock.patch.object(S, "find_project_dirs",
                              lambda project_filter=None: sorted(self.projects.iterdir())),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._td.cleanup()


class TestWrongSessionBugShape(_Base):
    def test_cwd_mismatch_picks_wrong_session_without_active_record(self):
        # Active session lives under -Users-ruya; cwd maps to the Cozempic dir whose
        # most-recent session is the STALE one. No active record, no live pid.
        active = _write_session(self.projects / "-Users-ruya", "f464a40c")
        os.utime(active, (1000, 1000))
        stale = _write_session(self.projects / "-Users-ruya-Documents-Advisor-Cozempic", "f641174c")
        os.utime(stale, (2000, 2000))  # newer mtime in the cwd's project dir
        with mock.patch.object(S, "_session_id_from_process", lambda: None), \
             mock.patch.object(S, "find_claude_pid", lambda: None):
            got = S.find_current_session("/Users/ruya/Documents/Advisor/Cozempic", strict=True)
        self.assertEqual(got["session_id"], "f641174c",
                         "documents the bug: cwd-inference picks the stale session")
        self.assertNotEqual(str(got["path"]), str(active))


class TestActiveTranscriptFix(_Base):
    def test_active_record_overrides_cwd_inference(self):
        active = _write_session(self.projects / "-Users-ruya", "f464a40c")
        os.utime(active, (1000, 1000))
        stale = _write_session(self.projects / "-Users-ruya-Documents-Advisor-Cozempic", "f641174c")
        os.utime(stale, (2000, 2000))
        # The live claude pid (54513) recorded its active transcript via the hook.
        S.record_active_transcript(str(active), claude_pid=54513)
        with mock.patch.object(S, "_session_id_from_process", lambda: None), \
             mock.patch.object(S, "find_claude_pid", lambda: 54513), \
             mock.patch.object(S, "_pid_alive", lambda pid: True):
            got = S.find_current_session("/Users/ruya/Documents/Advisor/Cozempic", strict=True)
        self.assertEqual(got["session_id"], "f464a40c",
                         "the active transcript must win over cwd-inference")
        self.assertEqual(str(got["path"]), str(active))

    def test_two_terminals_two_pids_resolve_independently(self):
        a = _write_session(self.projects / "-Users-ruya", "aaaaaaaa")
        b = _write_session(self.projects / "-Users-ruya-Documents-Advisor-Cozempic", "bbbbbbbb")
        # `_pid_alive` must be patched around the record calls too — not only the
        # lookups below. `record_active_transcript` prunes dead pids on write, so
        # with the real `os.kill` the first entry (pid 111) is dropped during the
        # second record on any host where 111 isn't a live process, leaving the
        # lookup to return None (host-dependent flake). Patch it for the whole body.
        with mock.patch.object(S, "_session_id_from_process", lambda: None), \
             mock.patch.object(S, "_pid_alive", lambda pid: True):
            S.record_active_transcript(str(a), claude_pid=111)
            S.record_active_transcript(str(b), claude_pid=222)
            with mock.patch.object(S, "find_claude_pid", lambda: 111):
                self.assertEqual(S.find_current_session("/tmp", strict=True)["session_id"], "aaaaaaaa")
            with mock.patch.object(S, "find_claude_pid", lambda: 222):
                self.assertEqual(S.find_current_session("/tmp", strict=True)["session_id"], "bbbbbbbb")


class TestStalenessGuards(_Base):
    def test_dead_pid_record_is_ignored(self):
        active = _write_session(self.projects / "-Users-ruya", "f464a40c")
        os.utime(active, (1000, 1000))
        stale = _write_session(self.projects / "-Users-ruya-Documents-Advisor-Cozempic", "f641174c")
        os.utime(stale, (2000, 2000))
        S.record_active_transcript(str(active), claude_pid=54513)
        with mock.patch.object(S, "_session_id_from_process", lambda: None), \
             mock.patch.object(S, "find_claude_pid", lambda: 54513), \
             mock.patch.object(S, "_pid_alive", lambda pid: False):  # pid died
            got = S.find_current_session("/Users/ruya/Documents/Advisor/Cozempic", strict=True)
        self.assertEqual(got["session_id"], "f641174c",
                         "a dead-pid record must be ignored (fall through to cwd)")

    def test_missing_transcript_file_is_ignored(self):
        active = _write_session(self.projects / "-Users-ruya", "f464a40c")
        S.record_active_transcript(str(active), claude_pid=54513)
        active.unlink()  # transcript vanished
        stale = _write_session(self.projects / "-Users-ruya-Documents-Advisor-Cozempic", "f641174c")
        with mock.patch.object(S, "_session_id_from_process", lambda: None), \
             mock.patch.object(S, "find_claude_pid", lambda: 54513), \
             mock.patch.object(S, "_pid_alive", lambda pid: True):
            got = S.find_current_session("/Users/ruya/Documents/Advisor/Cozempic", strict=True)
        self.assertEqual(got["session_id"], "f641174c")


class TestRecordRoundTrip(_Base):
    def test_record_and_lookup(self):
        active = _write_session(self.projects / "-Users-ruya", "f464a40c")
        S.record_active_transcript(str(active), claude_pid=999)
        with mock.patch.object(S, "_pid_alive", lambda pid: True):
            rec = S.lookup_active_transcript(999)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["session_id"], "f464a40c")
        self.assertEqual(rec["transcript_path"], str(active))

    def test_record_no_pid_is_noop(self):
        active = _write_session(self.projects / "-Users-ruya", "f464a40c")
        # No discoverable claude pid → nothing recorded, no crash.
        with mock.patch.object(S, "find_claude_pid", lambda: None):
            S.record_active_transcript(str(active), claude_pid=None)
        self.assertFalse(S._active_sessions_path().exists()
                         and S.lookup_active_transcript(0))

    def test_dead_pids_pruned_on_write(self):
        a = _write_session(self.projects / "-Users-ruya", "aaaaaaaa")
        b = _write_session(self.projects / "-Users-ruya", "bbbbbbbb")
        with mock.patch.object(S, "_pid_alive", lambda pid: pid == 222):
            S.record_active_transcript(str(a), claude_pid=111)  # 111 will be dead
            S.record_active_transcript(str(b), claude_pid=222)  # 222 alive
            import json as _j
            data = _j.loads(S._active_sessions_path().read_text())
            self.assertNotIn("111", data, "dead pid 111 must be pruned on write")
            self.assertIn("222", data)


if __name__ == "__main__":
    unittest.main()
