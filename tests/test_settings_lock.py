"""Tests for _SettingsLock cross-platform behavior.

The lock is exercised by `wire_hooks` / `uninstall_hooks` during `cozempic
init`. Until the v1.8.x cross-platform fix it degraded to a silent no-op
on Windows because `import fcntl` raised ImportError. This file pins the
new behavior:

    POSIX → fcntl.lockf (record locks, NFS-reliable, unchanged)
    Windows → msvcrt.locking (per-byte LK_LOCK/LK_UNLCK)
    Both missing → no-op fallback (no crash)
    OSError on real platform → warn + no-op (settings lock unavailable)

The four `test_windows_*` tests monkeypatch `os.name = "nt"` and inject
a fake `msvcrt` into `sys.modules` so the Windows branch runs
unconditionally regardless of host OS — on Windows the fake REPLACES
the real `msvcrt` (we deliberately want to assert on the call shape,
not exercise the real kernel lock). The POSIX branch is exercised by
the two `test_posix_*` tests when host OS is POSIX, and by the existing
`test_global_init.py` suite that hits `wire_hooks` end-to-end.
"""
from __future__ import annotations

import io
import os
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

from cozempic.init import _SettingsLock


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def settings_path(tmp_path: Path) -> Path:
    """A settings.json path; the lock sibling lives at .cozempic-init.lock."""
    return tmp_path / "settings.json"


# ─── POSIX path ──────────────────────────────────────────────────────────────


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only — fcntl unavailable on Windows")
def test_posix_acquires_fcntl_lockf(settings_path: Path) -> None:
    """On POSIX the lock issues fcntl.lockf LOCK_EX on enter, LOCK_UN on exit."""
    import fcntl
    with mock.patch("fcntl.lockf") as lockf:
        with _SettingsLock(settings_path):
            pass
    assert lockf.call_count == 2
    enter_args = lockf.call_args_list[0].args
    exit_args = lockf.call_args_list[1].args
    assert enter_args[1] == fcntl.LOCK_EX
    assert exit_args[1] == fcntl.LOCK_UN


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only — needs real fcntl + tmpfs")
def test_posix_real_lockfile_created(settings_path: Path) -> None:
    """The actual lock file lands alongside settings.json."""
    with _SettingsLock(settings_path):
        assert (settings_path.parent / ".cozempic-init.lock").exists()


# ─── Windows path (exercised via fake msvcrt on POSIX) ───────────────────────


class _FakeMsvcrt:
    """In-memory model of msvcrt.locking for cross-platform testing.

    We don't simulate cross-process contention — that's what the real
    msvcrt.locking would do. We just record the calls so we can assert
    the right operations happen in the right order with the right args.
    """
    LK_LOCK = 1
    LK_UNLCK = 0

    def __init__(self):
        self.calls: list[tuple[int, int, int]] = []  # (fd, mode, nbytes)

    def locking(self, fd: int, mode: int, nbytes: int) -> None:
        self.calls.append((fd, mode, nbytes))


def _force_windows_mode(monkeypatch, fake: _FakeMsvcrt) -> None:
    """Monkeypatch os.name == 'nt' and inject fake msvcrt module."""
    monkeypatch.setattr(os, "name", "nt")
    fake_module = types.ModuleType("msvcrt")
    fake_module.LK_LOCK = fake.LK_LOCK
    fake_module.LK_UNLCK = fake.LK_UNLCK
    fake_module.locking = fake.locking
    monkeypatch.setitem(sys.modules, "msvcrt", fake_module)


def test_windows_acquires_msvcrt_locking(monkeypatch, settings_path: Path) -> None:
    """On Windows-mode the lock issues msvcrt.locking LK_LOCK at byte 0 on enter, LK_UNLCK on exit."""
    fake = _FakeMsvcrt()
    _force_windows_mode(monkeypatch, fake)

    with _SettingsLock(settings_path):
        pass

    # Exactly two locking calls — lock then unlock — both on 1 byte.
    assert len(fake.calls) == 2, f"expected 2 calls, got {fake.calls}"
    enter_fd, enter_mode, enter_n = fake.calls[0]
    exit_fd, exit_mode, exit_n = fake.calls[1]
    assert enter_mode == _FakeMsvcrt.LK_LOCK
    assert exit_mode == _FakeMsvcrt.LK_UNLCK
    assert enter_n == 1 and exit_n == 1
    # Same fd used for lock and unlock (otherwise we'd be unlocking a
    # different file).
    assert enter_fd == exit_fd


def test_windows_seek_zero_before_unlock(monkeypatch, settings_path: Path) -> None:
    """msvcrt.locking is position-relative — both __enter__ AND __exit__ must seek(0).

    'a' (append) mode leaves the file pointer at end-of-file. For a fresh
    empty lock file EOF==0, but for a stale non-empty lock file from a
    prior crashed run, EOF>0. Without seek(0) before BOTH LK_LOCK (acquire)
    AND LK_UNLCK (release), the two operations target different byte
    ranges and silently fail to serialize. The test asserts seek(0) is
    called at least twice — once around acquire, once around release.
    """
    fake = _FakeMsvcrt()
    _force_windows_mode(monkeypatch, fake)

    # Capture seek calls on the file handle.
    seeks: list[int] = []
    original_open = open

    def tracking_open(path, *args, **kwargs):
        fh = original_open(path, *args, **kwargs)
        original_seek = fh.seek

        def recording_seek(offset, *seek_args, **seek_kwargs):
            seeks.append(offset)
            return original_seek(offset, *seek_args, **seek_kwargs)

        fh.seek = recording_seek
        return fh

    monkeypatch.setattr("builtins.open", tracking_open)

    with _SettingsLock(settings_path):
        pass

    # seek(0) must have been called at least twice: once before LK_LOCK
    # (defense-in-depth for stale non-empty lock files), once before
    # LK_UNLCK. If either is missing, the lock/release pair targets
    # different byte positions on non-empty lock files.
    assert seeks.count(0) >= 2, (
        f"expected at least 2 seek(0) calls (before LK_LOCK and before LK_UNLCK), "
        f"got seeks={seeks}"
    )


def test_windows_acquire_position_normalized_on_stale_lock_file(monkeypatch, settings_path: Path) -> None:
    """Regression test for the stale-non-empty-lock-file case: the byte locked
    by LK_LOCK MUST equal the byte unlocked by LK_UNLCK (both byte 0).

    Reproducer: pre-populate the lock file with content so EOF > 0. Without
    seek(0) before LK_LOCK, the lock would land at byte EOF while the
    unlock would target byte 0 — different ranges, no mutual exclusion.
    Asserted by recording the file position at each msvcrt.locking call.
    """
    # Pre-populate the lock file so EOF > 0.
    lock_path = settings_path.parent / ".cozempic-init.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("stale content from prior crashed run\n")
    assert lock_path.stat().st_size > 0

    # Capture the file position at the moment each msvcrt.locking call fires.
    position_at_call: list = []
    captured_fh: list = []

    fake = _FakeMsvcrt()
    original_locking = fake.locking

    def position_aware_locking(fd, mode, nbytes):
        # captured_fh[0] is the file handle opened by _SettingsLock
        if captured_fh:
            position_at_call.append((mode, captured_fh[0].tell()))
        return original_locking(fd, mode, nbytes)

    fake.locking = position_aware_locking
    _force_windows_mode(monkeypatch, fake)

    original_open = open

    def capturing_open(path, *args, **kwargs):
        fh = original_open(path, *args, **kwargs)
        captured_fh.append(fh)
        return fh

    monkeypatch.setattr("builtins.open", capturing_open)

    with _SettingsLock(settings_path):
        pass

    # Both LK_LOCK and LK_UNLCK must have fired at position 0 — anything
    # else means the seek(0) before-LK_LOCK or before-LK_UNLCK is missing.
    assert len(position_at_call) == 2, f"expected 2 locking calls, got {position_at_call}"
    lock_mode, lock_pos = position_at_call[0]
    unlock_mode, unlock_pos = position_at_call[1]
    assert lock_mode == _FakeMsvcrt.LK_LOCK
    assert unlock_mode == _FakeMsvcrt.LK_UNLCK
    assert lock_pos == 0, f"LK_LOCK fired at byte {lock_pos}, expected 0 (stale file case)"
    assert unlock_pos == 0, f"LK_UNLCK fired at byte {unlock_pos}, expected 0"


def test_windows_oserror_during_lock_warns_and_degrades(monkeypatch, capsys, settings_path: Path) -> None:
    """OSError from msvcrt.locking degrades to no-op + stderr warning + fh close."""
    fake = _FakeMsvcrt()

    def raising_locking(fd, mode, nbytes):
        raise OSError("simulated Windows lock failure (e.g. read-only filesystem)")

    fake.locking = raising_locking
    _force_windows_mode(monkeypatch, fake)

    # Capture stderr
    stderr_buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr_buf)

    # Capture the file handle that was opened-then-abandoned by the OSError
    # cleanup path, so we can assert it was closed (no fd leak).
    opened_handles: list = []
    original_open = open

    def tracking_open(path, *args, **kwargs):
        fh = original_open(path, *args, **kwargs)
        opened_handles.append(fh)
        return fh

    monkeypatch.setattr("builtins.open", tracking_open)

    with _SettingsLock(settings_path) as lock:
        # _fh should be None — degraded path
        assert lock._fh is None

    assert "settings lock unavailable" in stderr_buf.getvalue()
    # The file handle that was opened before msvcrt.locking failed MUST have
    # been closed by the OSError cleanup branch — otherwise the fd leaks
    # every time a settings lock acquisition fails (read-only mount, etc.).
    assert opened_handles, "expected open() to have been called"
    assert opened_handles[0].closed, "fd leak: opened lock file was not closed on OSError"


# ─── Missing-both fallback (neither fcntl nor msvcrt) ────────────────────────


def test_missing_both_locking_modules_degrades(monkeypatch, settings_path: Path) -> None:
    """Platform missing both fcntl and msvcrt → degrade to no-op without crash."""
    # Force a "windows-shaped" environment but with msvcrt absent (extreme
    # edge case — captures the defense-in-depth ImportError catch).
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setitem(sys.modules, "msvcrt", None)

    with _SettingsLock(settings_path) as lock:
        assert lock._fh is None  # Degraded — but no crash.


# ─── End-to-end exit safety (re-entry, exception during body) ────────────────


def test_lock_releases_on_exception(settings_path: Path) -> None:
    """If the body raises, the lock is still released (file handle closed)."""
    captured_fh = []

    class _Spy(_SettingsLock):
        def __enter__(self):
            super().__enter__()
            captured_fh.append(self._fh)
            return self

    with pytest.raises(ValueError):
        with _Spy(settings_path):
            raise ValueError("body failure")

    # File handle should be closed after __exit__ even on exception.
    if captured_fh and captured_fh[0] is not None:
        assert captured_fh[0].closed
