"""RED tests for L4 — .tmp atomic-write orphan masquerades as a session.

Bug: mkstemp(prefix=".tmp.", suffix=path.name) where path.name ends in ".jsonl"
produces ".tmp.<rand><uuid>.jsonl". pathlib's glob("*.jsonl") MATCHES leading-dot
files, so find_sessions() enumerates crash-orphans as phantom sessions.

Fixes tested here:
  P-B: find_sessions() must skip dotfiles (defense-in-depth; catches old-style
       orphans already on disk and any future source).
  P-A: atomic_write_text / save_messages must produce a temp name that does NOT
       end in ".jsonl" so it is not enumerable by glob("*.jsonl").

These tests must be RED at origin/main and GREEN after the fixes.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from cozempic.session import find_sessions, save_messages, load_messages
from cozempic.helpers import atomic_write_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _real_session_content(n: int = 3) -> str:
    lines = [json.dumps({"message": {"role": "user", "content": f"msg {i}"}}) for i in range(n)]
    return "\n".join(lines) + "\n"


def _write_real_session(proj_dir: Path, session_id: str | None = None) -> Path:
    proj_dir.mkdir(parents=True, exist_ok=True)
    sid = session_id or str(uuid.uuid4())
    p = proj_dir / f"{sid}.jsonl"
    p.write_text(_real_session_content(), encoding="utf-8")
    return p


def _drop_old_style_orphan(proj_dir: Path) -> Path:
    """Create a crash-orphan using the OLD mkstemp call that produced the bug.

    mkstemp(prefix=".tmp.", suffix="<uuid>.jsonl") → ".tmp.<rand><uuid>.jsonl"
    This is what a SIGKILL between mkstemp and os.replace would leave on disk.
    """
    session_id = str(uuid.uuid4())
    fd, tmp_name = tempfile.mkstemp(
        prefix=".tmp.", suffix=session_id + ".jsonl", dir=str(proj_dir)
    )
    os.close(fd)
    orphan = Path(tmp_name)
    # Write valid JSONL content so it would open fine if enumerated
    orphan.write_text(_real_session_content(1), encoding="utf-8")
    return orphan


# ---------------------------------------------------------------------------
# P-B: find_sessions() must not enumerate dotfile orphans
# ---------------------------------------------------------------------------

class TestFindSessionsSkipsDotfileOrphans:
    """find_sessions() must exclude .tmp.* orphans from the enumeration."""

    def test_old_style_orphan_not_returned(self, tmp_path):
        """A .tmp.<rand><uuid>.jsonl orphan must NOT appear in find_sessions results.

        This is the exact name shape a SIGKILL between mkstemp(prefix=".tmp.",
        suffix="<uuid>.jsonl") and os.replace would leave on disk.

        RED at base: glob("*.jsonl") matches dotfiles → orphan IS enumerated.
        GREEN after P-B: dotfile skip added → orphan excluded.
        """
        proj_dir = tmp_path / "projects" / "-some-project"
        proj_dir.mkdir(parents=True)

        real_path = _write_real_session(proj_dir)
        orphan_path = _drop_old_style_orphan(proj_dir)

        # Confirm both match the glob pattern (the raw bug)
        globbed = list(proj_dir.glob("*.jsonl"))
        assert any(p.name == real_path.name for p in globbed), "real session must match glob"
        assert any(p.name == orphan_path.name for p in globbed), (
            "orphan must match glob (confirms the raw bug exists)"
        )

        with patch("cozempic.session.get_projects_dir", return_value=tmp_path / "projects"):
            sessions = find_sessions()

        session_ids = {s["session_id"] for s in sessions}
        real_stem = real_path.stem
        orphan_stem = orphan_path.stem

        assert real_stem in session_ids, "real session must be found"
        assert orphan_stem not in session_ids, (
            f"orphan {orphan_path.name!r} must NOT be enumerated as a session "
            f"(stem={orphan_stem!r})"
        )

    def test_only_real_sessions_returned_when_multiple_orphans(self, tmp_path):
        """Multiple orphans present; only real sessions surfaced."""
        proj_dir = tmp_path / "projects" / "-proj"
        proj_dir.mkdir(parents=True)

        real_paths = [_write_real_session(proj_dir) for _ in range(3)]
        orphans = [_drop_old_style_orphan(proj_dir) for _ in range(2)]

        with patch("cozempic.session.get_projects_dir", return_value=tmp_path / "projects"):
            sessions = find_sessions()

        assert len(sessions) == 3, (
            f"Expected 3 real sessions, got {len(sessions)}: "
            f"{[s['session_id'] for s in sessions]}"
        )
        session_ids = {s["session_id"] for s in sessions}
        for orphan in orphans:
            assert orphan.stem not in session_ids

    def test_leading_dot_any_variant_skipped(self, tmp_path):
        """Any dotfile variant (.something.jsonl) is excluded — not just .tmp.*."""
        proj_dir = tmp_path / "projects" / "-proj"
        proj_dir.mkdir(parents=True)

        real_path = _write_real_session(proj_dir)
        # Generic hidden file that ends in .jsonl (e.g. editor swap or other tmp)
        hidden = proj_dir / ".hidden_something.jsonl"
        hidden.write_text(_real_session_content(1), encoding="utf-8")

        with patch("cozempic.session.get_projects_dir", return_value=tmp_path / "projects"):
            sessions = find_sessions()

        session_ids = {s["session_id"] for s in sessions}
        assert real_path.stem in session_ids
        assert hidden.stem not in session_ids


# ---------------------------------------------------------------------------
# P-A: new temp name must NOT end in .jsonl
# ---------------------------------------------------------------------------

class TestAtomicWriteTmpNotJsonl:
    """atomic_write_text and save_messages must produce a .partial temp,
    not a .jsonl temp that glob("*.jsonl") would enumerate.

    Assertion strategy: monkeypatch tempfile.mkstemp to capture what name is
    constructed, then verify it does NOT match the *.jsonl glob.
    """

    def test_atomic_write_text_tmp_does_not_end_in_jsonl(self, tmp_path):
        """atomic_write_text temp name must NOT end in .jsonl.

        RED at base: suffix=target.name → ".tmp.<rand>session.jsonl" ends in ".jsonl".
        GREEN after P-A: suffix=".partial" → ends in ".partial".
        """
        target = tmp_path / "session.jsonl"
        captured_tmp: list[str] = []

        import cozempic.helpers as helpers_mod
        real_mkstemp = helpers_mod._tempfile.mkstemp

        def capturing_mkstemp(**kwargs):
            fd, name = real_mkstemp(**kwargs)
            captured_tmp.append(name)
            return fd, name

        with patch.object(helpers_mod._tempfile, "mkstemp", side_effect=capturing_mkstemp):
            atomic_write_text(target, '{"test": 1}\n')

        assert captured_tmp, "mkstemp must have been called"
        tmp_name = captured_tmp[0]
        assert not tmp_name.endswith(".jsonl"), (
            f"atomic_write_text produced a .jsonl temp: {tmp_name!r}. "
            "This temp would be enumerated by glob('*.jsonl') if left orphaned."
        )
        # Must not match the session glob
        assert len(list(tmp_path.glob("*.jsonl"))) == 1, "only the real target should exist"
        assert Path(tmp_name).name not in {p.name for p in tmp_path.glob("*.jsonl")}

    def test_save_messages_tmp_does_not_end_in_jsonl(self, tmp_path):
        """save_messages temp name must NOT end in .jsonl.

        RED at base: suffix=path.name → ".tmp.<rand>session.jsonl" ends in ".jsonl".
        GREEN after P-A: suffix=".partial" → ends in ".partial".
        """
        jsonl = tmp_path / "session.jsonl"
        content = _real_session_content()
        jsonl.write_text(content, encoding="utf-8")
        messages = load_messages(jsonl)

        captured_tmp: list[str] = []

        # save_messages imports tempfile as _tempfile locally in the function
        import tempfile as tempfile_mod
        real_mkstemp = tempfile_mod.mkstemp

        def capturing_mkstemp(**kwargs):
            fd, name = real_mkstemp(**kwargs)
            captured_tmp.append(name)
            return fd, name

        with patch.object(tempfile_mod, "mkstemp", side_effect=capturing_mkstemp):
            save_messages(jsonl, messages, create_backup=False)

        assert captured_tmp, "mkstemp must have been called by save_messages"
        tmp_name = captured_tmp[0]
        assert not tmp_name.endswith(".jsonl"), (
            f"save_messages produced a .jsonl temp: {tmp_name!r}. "
            "This temp would be enumerated by glob('*.jsonl') if left orphaned."
        )

    def test_no_jsonl_glob_match_if_orphaned(self, tmp_path):
        """If a temp file is left orphaned (cleanup bypassed), it must NOT match *.jsonl.

        Simulate an orphaned temp: capture the tmp name from mkstemp, then leave it
        on disk without renaming it (simulating SIGKILL — the except-cleanup never runs).
        Then check whether glob('*.jsonl') in the dir picks it up.

        RED at base: temp ends in .jsonl → orphan IS found by glob.
        GREEN after P-A: temp ends in .partial → NOT found by glob.
        """
        import cozempic.helpers as helpers_mod

        target = tmp_path / "session.jsonl"
        target.write_text("{}\n", encoding="utf-8")

        orphaned_name: list[str] = []
        real_mkstemp = helpers_mod._tempfile.mkstemp

        def capturing_mkstemp(**kwargs):
            fd, name = real_mkstemp(**kwargs)
            orphaned_name.append(name)
            return fd, name

        # Patch unlink to no-op so the cleanup branch doesn't remove the temp
        # (simulating SIGKILL where cleanup never runs at all)
        with (
            patch.object(helpers_mod._tempfile, "mkstemp", side_effect=capturing_mkstemp),
            patch("os.replace", side_effect=OSError("simulated crash")),
            patch.object(helpers_mod._Path, "unlink", return_value=None),
        ):
            with pytest.raises(OSError):
                atomic_write_text(target, '{"after": 1}\n')

        # The temp file name was recorded; check if it would be matched by *.jsonl
        assert orphaned_name, "mkstemp must have been called"
        orphan = Path(orphaned_name[0])
        orphan_matches_jsonl_glob = orphan.suffix == ".jsonl" or orphan.name.endswith(".jsonl")
        assert not orphan_matches_jsonl_glob, (
            f"Orphaned temp {orphan.name!r} would be matched by glob('*.jsonl'). "
            "A SIGKILL between mkstemp and os.replace would leave this as a phantom session."
        )


# ---------------------------------------------------------------------------
# Vacuous-assertion replacements — previously checked wrong names
# ---------------------------------------------------------------------------

class TestAtomicNoOrphanJsonl:
    """Replace the vacuous `.tmp` / `.ct` / `.conflict_tmp` checks.

    The OLD assertions in TestAtomicWrite / TestSnapshotAndAppend checked for
    `sess.tmp` or `sess.conflict_tmp` — names mkstemp never produces.  These
    check the REAL invariant: after a successful or failed write, NO *.jsonl
    file OTHER than the target exists in the directory.

    Mutation check: if cleanup code is removed from save_messages, the
    temp file survives and the glob count rises above 1, failing the test.
    """

    def test_no_orphan_jsonl_after_successful_save(self, tmp_path):
        """After a successful save, no extra *.jsonl file remains in the dir."""
        jsonl = tmp_path / "sess.jsonl"
        lines = [json.dumps({"message": {"role": "user", "content": f"m{i}"}}) for i in range(3)]
        jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        messages = load_messages(jsonl)
        save_messages(jsonl, messages, create_backup=False)
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert jsonl_files == [jsonl], (
            f"Expected only target {jsonl.name}, found: {[f.name for f in jsonl_files]}"
        )

    def test_no_orphan_jsonl_after_fsync_failure(self, tmp_path, monkeypatch):
        """If fsync raises, no extra *.jsonl orphan is left in the dir."""
        jsonl = tmp_path / "sess.jsonl"
        original = (
            "\n".join(json.dumps({"message": {"role": "user", "content": f"orig {i}"}}) for i in range(3))
            + "\n"
        )
        jsonl.write_text(original, encoding="utf-8")
        messages = load_messages(jsonl)

        import os as _os
        monkeypatch.setattr(_os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk full")))

        with pytest.raises(OSError):
            save_messages(jsonl, messages, create_backup=False)

        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert jsonl_files == [jsonl], (
            f"After fsync failure, unexpected *.jsonl files: {[f.name for f in jsonl_files]}"
        )

    def test_no_orphan_jsonl_after_conflict(self, tmp_path):
        """After a PruneConflictError, no extra *.jsonl orphan is left."""
        from cozempic.session import PruneConflictError, snapshot_session

        jsonl = tmp_path / "sess.jsonl"
        lines = [json.dumps({"message": {"role": "user", "content": f"m{i}"}}) for i in range(3)]
        jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        messages = load_messages(jsonl)
        snap = snapshot_session(jsonl)

        # Trigger conflict: full rewrite via os.replace
        new_content = json.dumps({"message": {"role": "user", "content": "rewritten"}}) + "\n"
        replacement = jsonl.with_suffix(".ct")
        replacement.write_text(new_content, encoding="utf-8")
        os.replace(replacement, jsonl)

        with pytest.raises(PruneConflictError):
            save_messages(jsonl, messages, create_backup=False, snapshot=snap)

        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert jsonl_files == [jsonl], (
            f"After conflict, unexpected *.jsonl files: {[f.name for f in jsonl_files]}"
        )

    def test_no_orphan_jsonl_after_held_open_replace(self, tmp_path, monkeypatch):
        """After a PermissionError on os.replace, no extra *.jsonl file remains."""
        from cozempic.session import PruneConflictError

        jsonl = tmp_path / "sess.jsonl"
        original = (
            json.dumps({"message": {"role": "user", "content": "x"}}) + "\n"
        )
        jsonl.write_text(original, encoding="utf-8")
        messages = load_messages(jsonl)

        import os as _os
        real_replace = _os.replace

        def denied(src, dst, *a, **k):
            if str(dst) == str(jsonl):
                raise PermissionError("[WinError 5] Access is denied")
            return real_replace(src, dst, *a, **k)

        monkeypatch.setattr(_os, "replace", denied)

        with pytest.raises(PruneConflictError):
            save_messages(jsonl, messages, create_backup=False)

        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert jsonl_files == [jsonl], (
            f"After held-open PermissionError, unexpected *.jsonl files: {[f.name for f in jsonl_files]}"
        )

    def test_orphaned_jsonl_temp_enumerated_by_find_sessions(self, tmp_path):
        """Confirm the bug: an old-style orphan (.tmp.*.<uuid>.jsonl) IS found by
        find_sessions() before the fix (P-B). This test must be RED at base.

        Strategy: drop an orphan, run find_sessions(), assert the orphan is NOT
        returned. Before P-B the orphan IS returned → assertion fails → RED.
        After P-B → assertion passes → GREEN.
        """
        proj_dir = tmp_path / "projects" / "-proj"
        proj_dir.mkdir(parents=True)

        real_path = _write_real_session(proj_dir)
        orphan_path = _drop_old_style_orphan(proj_dir)

        with patch("cozempic.session.get_projects_dir", return_value=tmp_path / "projects"):
            sessions = find_sessions()

        orphan_stems = {orphan_path.stem}
        found_stems = {s["session_id"] for s in sessions}
        assert not orphan_stems.intersection(found_stems), (
            f"Orphan {orphan_path.name!r} was enumerated as a session. "
            "find_sessions() must skip dotfiles."
        )
