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
    load_messages_and_snapshot,
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

    def test_unicode_line_separators_do_not_split_a_json_line(self, tmp_path):
        """A raw U+2028/U+2029/U+0085 inside a JSON string (legal; JS JSON.stringify
        emits U+2028/U+2029 unescaped) must NOT split the line. str.splitlines()
        would tear it into invalid fragments and corrupt it on save — the loaders
        must match text-mode open() (split on \\n/\\r only). Parity across all three
        readers + a save round-trip that preserves the line."""
        from cozempic.session import load_messages_and_snapshot, load_messages_incremental
        jsonl = tmp_path / "u.jsonl"
        for sep in (" ", " ", ""):
            msg = {"type": "assistant", "message": {"role": "assistant", "content": f"a{sep}b"}}
            keep = {"type": "user", "message": {"role": "user", "content": "ok"}}
            jsonl.write_bytes((json.dumps(msg, ensure_ascii=False) + "\n"
                               + json.dumps(keep) + "\n").encode("utf-8"))
            a = load_messages(jsonl)
            b, snap = load_messages_and_snapshot(jsonl)
            c = load_messages_incremental(jsonl)
            assert len(a) == len(b) == len(c) == 2, f"{sep!r}: a valid JSON line must stay one line"
            assert [m for _, m, _ in a] == [m for _, m, _ in b], f"{sep!r}: parsed dicts must match"
            assert not any(m.get("_parse_error") for _, m, _ in b), f"{sep!r}: no fragment parse-errors"
            # Save round-trip keeps the separator-bearing message intact (no corruption).
            save_messages(jsonl, b, create_backup=False, snapshot=snap)
            reloaded = load_messages(jsonl)
            assert reloaded[0][1]["message"]["content"] == f"a{sep}b", f"{sep!r}: content corrupted on save"

    def test_appended_unicode_separator_line_merges_intact(self, tmp_path):
        """The append-merge delta path (_parse_delta_lines) must not tear an
        appended JSONL line containing a raw U+2028/U+2029 into fragments."""
        jsonl = tmp_path / "d.jsonl"
        _make_messages(jsonl, n=3)
        messages, snap = load_messages_and_snapshot(jsonl)
        appended = {"type": "assistant", "message": {"role": "assistant", "content": "x y"}}
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(appended, ensure_ascii=False) + "\n")
        assert snap.classify(jsonl) == "appended"
        save_messages(jsonl, messages, create_backup=False, snapshot=snap)
        reloaded = load_messages(jsonl)
        assert not any(m.get("_parse_error") for _, m, _ in reloaded), "appended U+2028 line torn into fragments"
        assert reloaded[-1][1]["message"]["content"] == "x y", "appended separator line corrupted on merge"

    def test_non_dict_lines_wrapped_not_crash(self, tmp_path):
        """Mission-critical C4 root: a valid-JSON-but-non-dict line (bare string /
        number / array / null) OR a dict line with a non-dict inner 'message' must
        be wrapped as a _parse_error (preserved via _raw on save), so no downstream
        consumer ever receives a non-dict message and crashes."""
        jsonl = tmp_path / "poison.jsonl"
        good = '{"type":"user","message":{"role":"user","content":"keep"}}'
        poison = ['"bare string"', "null", "42", "[1,2,3]",
                  '{"type":"assistant","message":"inner is a string"}']
        jsonl.write_text(good + "\n" + "\n".join(poison) + "\n" + good + "\n", encoding="utf-8")
        msgs = load_messages(jsonl)
        # every message element is a dict
        assert all(isinstance(m, dict) for _, m, _ in msgs)
        # the poison lines are flagged _parse_error and preserved verbatim via _raw
        errs = [m for _, m, _ in msgs if m.get("_parse_error")]
        assert len(errs) == len(poison), (len(errs), len(poison))
        # round-trips losslessly (save writes _raw)
        m2, snap = load_messages_and_snapshot(jsonl)
        save_messages(jsonl, m2, create_backup=False, snapshot=snap)
        raw = jsonl.read_text(encoding="utf-8")
        for p in poison:
            assert p in raw, f"poison line not preserved on save: {p}"

    def test_invalid_utf8_roundtrips_losslessly_not_corrupts_not_aborts(self, tmp_path):
        """Mission-critical (R4): a non-UTF-8 byte must neither ABORT the prune
        (the round-3 strict-decode behavior, which left the guard permanently inert
        on any session with one stray byte) NOR be silently rewritten to U+FFFD
        (the errors='replace' corruption). surrogateescape round-trips the exact
        bytes losslessly while letting the prune proceed."""
        jsonl = tmp_path / "bad.jsonl"
        good = b'{"type":"user","message":{"role":"user","content":"keep"}}\n'
        # (a) bad byte INSIDE a JSON string value — loads as a surrogate, content
        # survives a load->save->load round-trip identically.
        in_string = b'{"type":"user","message":{"role":"user","content":"raw \xff byte"}}\n'
        # (b) bad byte OUTSIDE a string (structural) — json.loads fails, wrapped as
        # _raw and written back BYTE-FOR-BYTE through the surrogateescape save.
        structural = b'{"type":"user"\xfe,"message":{"role":"user","content":"x"}}\n'
        jsonl.write_bytes(good + in_string + structural)

        # No abort — load proceeds.
        messages, snap = load_messages_and_snapshot(jsonl)
        assert len(messages) == 3
        # The structural line round-trips as an opaque _raw wrapper (no consumer
        # sees a non-dict; preserved verbatim on save).
        assert messages[2][1].get("_parse_error") is True

        # Save and reload: BOTH the structural _raw line AND the in-string byte must
        # survive BYTE-FOR-BYTE (R5 P0: the in-string case must NOT be rewritten to a
        # literal \udcXX escape — that requires ensure_ascii=False on save). Asserting
        # only surrogate-codepoint equality here gave false confidence and masked the
        # P0 corruption; assert the raw bytes are present on disk.
        save_messages(jsonl, messages, create_backup=False, snapshot=snap)
        after = jsonl.read_bytes()
        assert b'\xfe' in after, "structural bad byte must survive byte-for-byte via _raw"
        assert b'raw \xff byte' in after, "in-string bad byte must survive byte-for-byte (no \\udcXX escape)"
        assert b'\\udcff' not in after, "in-string byte must NOT be corrupted to a literal escape"
        reloaded = load_messages(jsonl)
        assert len(reloaded) == 3
        assert reloaded[0][1]["message"]["content"] == "keep"
        assert reloaded[1][1]["message"]["content"].encode("utf-8", "surrogateescape") == b"raw \xff byte"

    def test_load_and_snapshot_no_toctou_duplication(self, tmp_path):
        """Read-once: a line appended AFTER load_messages_and_snapshot must be
        recovered exactly ONCE on save (not duplicated). Regression for the TOCTOU
        where the old snapshot-then-load pattern counted a window-append in both
        the loaded messages and the delta."""
        from cozempic.session import load_messages_and_snapshot
        jsonl = tmp_path / "sess.jsonl"
        _make_messages(jsonl, n=5)
        messages, snap = load_messages_and_snapshot(jsonl)
        assert len(messages) == 5
        # Claude appends a NEW line after our read.
        extra = json.dumps({"message": {"role": "assistant", "content": "appended-once"}}) + "\n"
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(extra)
        # classify must see exactly that one appended line as the delta.
        assert snap.classify(jsonl) == "appended"
        save_messages(jsonl, messages, create_backup=False, snapshot=snap)
        reloaded = load_messages(jsonl)
        contents = [m["message"]["content"] for _, m, _ in reloaded]
        assert contents.count("appended-once") == 1, "window-appended line must appear once (no TOCTOU dup)"
        assert len(reloaded) == 6, "5 pruned + 1 appended delta = 6"

    def test_equal_size_inplace_rewrite_is_conflict_not_clobbered(self, tmp_path):
        """Same-SIZE different-CONTENT rewrite must classify as conflict, not
        'unchanged' — otherwise save_messages silently os.replace()s over Claude's
        live equal-length rewrite (data loss). Regression for the audit P1."""
        from cozempic.session import snapshot_session
        jsonl = tmp_path / "sess.jsonl"
        messages = _make_messages(jsonl, n=5)
        snap = snapshot_session(jsonl)
        # Rewrite the file in place to the SAME byte length but different content.
        original = jsonl.read_bytes()
        mutated = bytearray(original)
        # Flip the last content char (keeps length identical, same inode via in-place write).
        for i in range(len(mutated) - 2, -1, -1):
            if chr(mutated[i]).isalnum():
                mutated[i] = ord("Z") if chr(mutated[i]) != "Z" else ord("Y")
                break
        with open(jsonl, "r+b") as f:
            f.seek(0); f.write(bytes(mutated))
        assert jsonl.stat().st_size == snap.size, "test must keep size equal"
        assert snap.classify(jsonl) == "conflict", "equal-size content change must be a conflict"
        # And the high-level save must refuse rather than clobber the live rewrite.
        with pytest.raises(PruneConflictError):
            save_messages(jsonl, messages, create_backup=False, snapshot=snap)

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
