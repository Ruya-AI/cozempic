"""A running guard must keep its pidfile alive against /tmp cleaners.

macOS 14+ tmp_cleaner deletes /tmp files whose atime, mtime and ctime are all
older than 3 days. The pidfile was written once at spawn, so a guard running
longer than that lost it at the next midnight, and the next SessionStart
spawned a duplicate guard for the same session.
"""
import os
import time
from unittest import mock

import pytest

from cozempic import guard as G

SID = "0123456789ab-cdef-0123-4567-89abcdef0123"


@pytest.fixture
def tmp_root(tmp_path):
    with mock.patch.object(G, "_guard_tmp_root", lambda: tmp_path):
        yield tmp_path


def _write_pidfile(path, pid):
    path.write_text(f"{pid}\n2026-01-01T00:00:00\nspawn-claim-daemon\n")


def test_own_pidfile_and_log_are_touched(tmp_root):
    pid_path = G._pid_file_for_session(SID)
    log_path = pid_path.with_suffix(".log")
    _write_pidfile(pid_path, os.getpid())
    log_path.write_text("log\n")
    old = time.time() - 5 * 86400
    os.utime(pid_path, (old, old))
    os.utime(log_path, (old, old))

    state = {}
    G._keep_own_pidfile_alive(SID, state)

    assert state["owned"] is True
    for p in (pid_path, log_path):
        st = p.stat()
        assert time.time() - st.st_mtime < 60
        assert time.time() - st.st_atime < 60


def test_deleted_pidfile_is_restored_by_its_owner(tmp_root):
    pid_path = G._pid_file_for_session(SID)
    _write_pidfile(pid_path, os.getpid())
    state = {}
    G._keep_own_pidfile_alive(SID, state)
    pid_path.unlink()  # what tmp_cleaner does at 00:00

    G._keep_own_pidfile_alive(SID, state)

    assert G._pid_file_points_to(SID, os.getpid())
    assert pid_path.read_text().splitlines()[2] == "guard-restored"


def test_never_creates_a_pidfile_it_did_not_own(tmp_root):
    # A foreground guard, or one whose claim was handed off, must not invent one.
    G._keep_own_pidfile_alive(SID, {})
    assert not G._pid_file_for_session(SID).exists()


def test_never_overwrites_a_peer_claim(tmp_root):
    pid_path = G._pid_file_for_session(SID)
    _write_pidfile(pid_path, os.getpid())
    state = {}
    G._keep_own_pidfile_alive(SID, state)
    _write_pidfile(pid_path, 999999)  # a reload hand-off or concurrent spawn
    before = pid_path.read_text()
    old = time.time() - 5 * 86400
    os.utime(pid_path, (old, old))

    G._keep_own_pidfile_alive(SID, state)

    assert pid_path.read_text() == before
    assert time.time() - pid_path.stat().st_mtime > 86400  # not ours, not touched


def test_invalid_session_id_is_a_no_op(tmp_root):
    G._keep_own_pidfile_alive(None, {})
    G._keep_own_pidfile_alive("bad id!", {"owned": True})
    assert list(tmp_root.iterdir()) == []
