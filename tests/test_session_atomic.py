"""Tests for atomic write behaviour in save_messages."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from cozempic.session import (
    PruneConflictError,
    PruneLockError,
    _PruneLock,
    load_messages,
    save_messages,
    snapshot_session,
)


def _make_messages(path: Path, n: int = 5) -> list:
    lines = [json.dumps({"message": {"role": "user", "content": f"msg {i}"}}) for i in range(n)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_messages(path)


class TestAtomicWrite:
    def test_no_tmp_left_on_success(self, tmp_path):
        """No .tmp file should remain after a successful save."""
        jsonl = tmp_path / "sess.jsonl"
        messages = _make_messages(jsonl)
        save_messages(jsonl, messages, create_backup=False)
        # No atomic-write temp orphan of ANY naming remains (the mkstemp temp is
        # ".tmp.<name>.<rand>.partial", never "sess.tmp" — the old assertion was vacuous).
        assert list(tmp_path.glob(".tmp.*")) == []

    def test_content_correct_after_save(self, tmp_path):
        jsonl = tmp_path / "sess.jsonl"
        messages = _make_messages(jsonl)
        save_messages(jsonl, messages, create_backup=False)
        reloaded = load_messages(jsonl)
        assert len(reloaded) == len(messages)
        for (_, orig, _), (_, reloaded_msg, _) in zip(messages, reloaded):
            assert orig == reloaded_msg

    def test_tmp_cleaned_on_fsync_error(self, tmp_path, monkeypatch):
        """If os.fsync raises, the .tmp file is deleted and the original untouched."""
        jsonl = tmp_path / "sess.jsonl"
        original_text = "\n".join(
            json.dumps({"message": {"role": "user", "content": f"original {i}"}}) for i in range(3)
        ) + "\n"
        jsonl.write_text(original_text, encoding="utf-8")
        messages = load_messages(jsonl)

        import os as _os
        real_fsync = _os.fsync

        def boom(fd):
            raise OSError("disk full")

        monkeypatch.setattr(_os, "fsync", boom)

        with pytest.raises(OSError):
            save_messages(jsonl, messages, create_backup=False)

        # Original file should be intact
        assert jsonl.read_text(encoding="utf-8") == original_text
        # the atomic-write temp must be cleaned up — no orphan of ANY naming
        assert list(jsonl.parent.glob(".tmp.*")) == []

    def test_concurrent_writer_produces_valid_jsonl(self, tmp_path):
        """A background thread appending lines while save_messages runs must not
        corrupt the file (atomic rename guarantees the reader sees either the
        old or new version, never a partial write)."""
        jsonl = tmp_path / "sess.jsonl"
        messages = _make_messages(jsonl, n=20)

        errors: list[str] = []
        stop = threading.Event()

        def _appender():
            """Simulates Claude appending new lines to the session file."""
            while not stop.is_set():
                try:
                    with open(jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"message": {"role": "user", "content": "appended"}}) + "\n")
                except OSError:
                    pass
                time.sleep(0.005)

        t = threading.Thread(target=_appender, daemon=True)
        t.start()

        # Run several save cycles while the appender is active
        for _ in range(10):
            try:
                save_messages(jsonl, messages, create_backup=False)
            except Exception as e:
                errors.append(str(e))
            time.sleep(0.01)

        stop.set()
        t.join(timeout=2)

        assert not errors, f"save_messages raised: {errors}"

        # Final file must be valid JSONL (no partial lines)
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                json.loads(line)  # raises if corrupt

    def test_backup_created_with_timestamp(self, tmp_path):
        jsonl = tmp_path / "sess.jsonl"
        messages = _make_messages(jsonl)
        backup = save_messages(jsonl, messages, create_backup=True)
        assert backup is not None
        assert backup.exists()
        assert backup.suffix == ".bak"
        assert "jsonl" in backup.name

    def test_no_backup_when_disabled(self, tmp_path):
        jsonl = tmp_path / "sess.jsonl"
        messages = _make_messages(jsonl)
        backup = save_messages(jsonl, messages, create_backup=False)
        assert backup is None


class TestSnapshotAndAppend:
    def test_unchanged_snapshot_saves_normally(self, tmp_path):
        """Snapshot with no changes in between → unchanged → normal save."""
        jsonl = tmp_path / "sess.jsonl"
        messages = _make_messages(jsonl)
        snap = snapshot_session(jsonl)
        backup = save_messages(jsonl, messages, create_backup=False, snapshot=snap)
        assert backup is None
        reloaded = load_messages(jsonl)
        assert len(reloaded) == len(messages)

    def test_appended_lines_preserved(self, tmp_path):
        """Lines Claude appends mid-prune survive in the output."""
        jsonl = tmp_path / "sess.jsonl"
        messages = _make_messages(jsonl, n=5)
        snap = snapshot_session(jsonl)

        # Simulate Claude appending a new line after snapshot
        extra = json.dumps({"message": {"role": "assistant", "content": "new reply"}}) + "\n"
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(extra)

        save_messages(jsonl, messages, create_backup=False, snapshot=snap)

        reloaded = load_messages(jsonl)
        # Pruned 5 lines + 1 appended delta = 6 total
        assert len(reloaded) == 6
        contents = [m["message"]["content"] for _, m, _ in reloaded]
        assert "new reply" in contents

    def test_conflict_raises_and_leaves_file_intact(self, tmp_path):
        """If the prefix was rewritten mid-prune, PruneConflictError is raised."""
        jsonl = tmp_path / "sess.jsonl"
        messages = _make_messages(jsonl, n=5)
        snap = snapshot_session(jsonl)

        # Simulate a full rewrite (inode change via os.replace)
        import os
        new_content = json.dumps({"message": {"role": "user", "content": "rewritten"}}) + "\n"
        tmp = jsonl.with_suffix(".conflict_tmp")
        tmp.write_text(new_content, encoding="utf-8")
        os.replace(tmp, jsonl)

        original_text = jsonl.read_text(encoding="utf-8")
        with pytest.raises(PruneConflictError):
            save_messages(jsonl, messages, create_backup=False, snapshot=snap)

        # File must be unchanged from the rewrite
        assert jsonl.read_text(encoding="utf-8") == original_text
        # No orphaned temp left behind (any naming, incl. the .partial temp)
        assert list(jsonl.parent.glob(".tmp.*")) == []

    def test_no_orphan_backup_on_conflict(self, tmp_path):
        """Backup is NOT created when a conflict aborts the prune."""
        jsonl = tmp_path / "sess.jsonl"
        messages = _make_messages(jsonl, n=3)
        snap = snapshot_session(jsonl)

        import os
        tmp = jsonl.with_suffix(".ct")
        tmp.write_text(json.dumps({"rewritten": True}) + "\n", encoding="utf-8")
        os.replace(tmp, jsonl)

        with pytest.raises(PruneConflictError):
            save_messages(jsonl, messages, create_backup=True, snapshot=snap)

        bak_files = list(jsonl.parent.glob("*.bak"))
        assert bak_files == [], "no backup should be created on conflict"

    def test_incomplete_append_raises_conflict(self, tmp_path):
        """A delta that doesn't end with newline (mid-write) raises PruneConflictError."""
        jsonl = tmp_path / "sess.jsonl"
        messages = _make_messages(jsonl, n=3)
        snap = snapshot_session(jsonl)

        # Append bytes that don't end with newline — Claude mid-write
        with open(jsonl, "ab") as f:
            f.write(b'{"message":{"role":"user","content":"partial"')  # no closing brace or newline

        with pytest.raises(PruneConflictError):
            save_messages(jsonl, messages, create_backup=False, snapshot=snap)

    def test_held_open_replace_defers_as_conflict(self, tmp_path, monkeypatch):
        """#112: on Windows os.replace onto a held-open transcript raises
        PermissionError [WinError 5]. save_messages must treat this as a
        deferred prune (PruneConflictError), not a hard crash, and leave the
        original file intact with no orphaned .tmp."""
        jsonl = tmp_path / "sess.jsonl"
        original_text = "\n".join(
            json.dumps({"message": {"role": "user", "content": f"original {i}"}}) for i in range(3)
        ) + "\n"
        jsonl.write_text(original_text, encoding="utf-8")
        messages = load_messages(jsonl)

        import os as _os
        real_replace = _os.replace

        def denied(src, dst, *a, **k):
            # only the final transcript replace is denied; tmp scaffolding is allowed
            if str(dst) == str(jsonl):
                raise PermissionError("[WinError 5] Access is denied")
            return real_replace(src, dst, *a, **k)

        monkeypatch.setattr(_os, "replace", denied)

        with pytest.raises(PruneConflictError):
            save_messages(jsonl, messages, create_backup=False)

        # original intact, no orphan temp (any naming, incl. the .partial temp)
        assert jsonl.read_text(encoding="utf-8") == original_text
        assert list(jsonl.parent.glob(".tmp.*")) == []

    def test_held_open_replace_cleans_backup(self, tmp_path, monkeypatch):
        """#112: when the deferred-prune path fires with create_backup=True,
        the just-created .bak must be cleaned up so deferred cycles don't
        accumulate orphan backups."""
        jsonl = tmp_path / "sess.jsonl"
        jsonl.write_text(
            json.dumps({"message": {"role": "user", "content": "x"}}) + "\n", encoding="utf-8"
        )
        messages = load_messages(jsonl)

        import os as _os
        real_replace = _os.replace

        def denied(src, dst, *a, **k):
            if str(dst) == str(jsonl):
                raise PermissionError("[WinError 5] Access is denied")
            return real_replace(src, dst, *a, **k)

        monkeypatch.setattr(_os, "replace", denied)

        with pytest.raises(PruneConflictError):
            save_messages(jsonl, messages, create_backup=True)

        assert list(jsonl.parent.glob("*.bak")) == [], "deferred prune must not leave an orphan backup"


class TestPruneLock:
    def test_lock_acquired_and_released(self, tmp_path):
        """Lock file is created on enter and removed on exit."""
        jsonl = tmp_path / "sess.jsonl"
        jsonl.write_text("{}\n", encoding="utf-8")
        lock_path = jsonl.with_suffix(".prune-lock")

        with _PruneLock(jsonl):
            assert lock_path.exists()

        assert not lock_path.exists()

    def test_second_lock_raises(self, tmp_path):
        """A second lock on the same file raises PruneLockError."""
        jsonl = tmp_path / "sess.jsonl"
        jsonl.write_text("{}\n", encoding="utf-8")

        with _PruneLock(jsonl):
            with pytest.raises(PruneLockError):
                with _PruneLock(jsonl):
                    pass  # should not reach here

    def test_lock_released_after_exception(self, tmp_path):
        """Lock file is cleaned up even when body raises."""
        jsonl = tmp_path / "sess.jsonl"
        jsonl.write_text("{}\n", encoding="utf-8")
        lock_path = jsonl.with_suffix(".prune-lock")

        with pytest.raises(RuntimeError):
            with _PruneLock(jsonl):
                raise RuntimeError("body error")

        assert not lock_path.exists()

    def test_second_lock_succeeds_after_first_released(self, tmp_path):
        """After the first lock is released, a second acquisition succeeds."""
        jsonl = tmp_path / "sess.jsonl"
        jsonl.write_text("{}\n", encoding="utf-8")

        with _PruneLock(jsonl):
            pass
        # Should not raise
        with _PruneLock(jsonl):
            pass
