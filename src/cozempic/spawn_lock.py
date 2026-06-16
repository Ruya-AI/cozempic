"""Cross-process daemon spawn claim — single-flight via O_CREAT|O_EXCL.

History note (BMAD V4 finding, 2026-05-18): the previous implementation of
this module used ``fcntl.flock`` on a separate sentinel file with unlink on
release. That pattern has a textbook flock-unlink race exposed only at
N >= 3 contending processes:

    1. Process A acquires flock on inode I1, completes spawn, unlinks the
       sentinel path → frees the path entry, retains I1 only until A closes
       the fd.
    2. Between A's unlink and A's fd close, B and C each O_CREAT a NEW file
       at the (now-absent) path, getting NEW inodes I2 and I3.
    3. B's ``flock(I2)`` and C's ``flock(I3)`` are on different kernel
       objects → both succeed.
    4. Both B and C proceed past the lock → BOTH ``subprocess.Popen`` →
       both report ``started=True`` for the same session UUID.

Race-reproducer V4 stress (10 processes × 30 iterations) observed this 3
times out of 30, with both 2-winner and 3-winner outcomes, plus one worker
returning the undefined ``started=False, already_running=False, pid=None``
state. Raw evidence: ``/tmp/v4-stress-output.txt``.

The fix (race-reproducer's option 2, approved by team-lead): drop ``flock``
entirely. The PID file IS the lock. ``O_CREAT|O_EXCL|O_WRONLY`` on the PID
file is a POSIX-guaranteed atomic operation; if two processes race, exactly
one wins with the file create, the other sees ``FileExistsError`` (EEXIST).
This is the same pattern ``reload_lock.py:200-262`` already uses for the
reload-side single-flight.

Lifecycle:
  - On claim (enter): ``O_CREAT|O_EXCL|O_NOFOLLOW`` on the PID file. Loser
    reads the file, returns ``DaemonAlreadyStarting(holder_pid=<read pid>)``.
    Stale-holder detection: if the read PID is not alive (``kill(pid, 0)``
    fails), unlink + retry once.
  - Inside the claim: caller writes the real daemon PID atomically via
    temp-file + rename (``os.rename`` on same FS is POSIX-atomic). Readers
    transitioning between our parent-PID and the daemon PID observe the OLD
    real PID then the NEW real PID — never a placeholder, never an empty
    file mid-write.
  - On normal exit: the PID file is LEFT in place — the daemon owns it for
    its lifetime; the daemon's shutdown path is responsible for unlinking.
  - On exceptional exit (Popen raised, log open failed, etc.): the PID
    file is unlinked so a retry by the same or a sibling process can
    re-claim cleanly.

The owner-write pattern (parent PID first, then daemon PID via rename):
  Readers that probe between our claim and our rename will see OUR parent
  PID. Since this is a real, alive process, ``_is_guard_running_for_session``
  will treat it as an in-flight daemon and return ``already_running``. By
  the time the rename completes the daemon PID is visible. No "0"
  placeholder ever appears — closes the original Bug 3 too.

Symlink defense: ``O_NOFOLLOW`` mirrors ``reload_lock.py:213-214``. A local
attacker who can write to ``/tmp`` could plant a symlink at our PID path
pointing at an arbitrary file; without O_NOFOLLOW our O_CREAT would follow
the link.

Backward compatibility shim: the ``daemon_spawn_lock`` context-manager API
is kept for the existing test surface (``test_spawn_lock.py`` Unit tests)
and any direct callers. Internally it delegates to ``DaemonSpawnClaim``.
``DaemonSpawnClaim`` is the preferred new API for callers that need access
to the holder PID on contention (the contextmanager loses that information
because the loser shape is just the exception).
"""

from __future__ import annotations

import math
import os
import re
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

# A pidfile younger than this is considered "fresh — a peer is mid-spawn"
# even if the PID it carries is not alive. Rationale: stale pidfiles from
# crashed prior spawns are typically minutes/hours old; a freshly-written
# file with a dead PID is overwhelmingly likely to be a test mock or a peer
# whose daemon hasn't fully started yet. Treating it as stale would let
# multiple peers re-spawn against the same session.
#
# 5 seconds is the default — generous for a clean macOS laptop where a
# Python cold-start + cozempic import is well under 1s. On slower setups
# (CI hosted runners, macOS with Crowdstrike/SentinelOne EDR scanning the
# Python binary, cold filesystem cache after pip install) Popen can take
# several seconds. Operators can override via the
# ``COZEMPIC_PIDFILE_FRESH_SECONDS`` env var without code changes.
#
# Override is parsed at import time and CLAMPED to ``(0, _FRESH_MAX]``.
# Invalid values (non-numeric, NaN, inf, ≤0, > _FRESH_MAX) silently fall
# back to the default — keeps the daemon working rather than failing at
# startup over a misconfigured env var, and prevents operator typos like
# "inf" or "-1" from silently disabling staleness recovery entirely. The
# upper bound matches ``HARD_LOOP_BACKOFF_CAP_SECONDS`` in guard.py — any
# legitimate slow-Popen scenario lives well under 5 minutes; values higher
# than that are almost certainly typos that would let stale pidfiles from
# crashed prior spawns block new ones for hours. Operators who care about
# the active value can read ``spawn_lock._FRESH_PIDFILE_SECONDS``.
_DEFAULT_FRESH = 5.0
_FRESH_MAX = 300.0


def _read_fresh_window_seconds() -> float:
    """Read the COZEMPIC_PIDFILE_FRESH_SECONDS env var and clamp.

    IMPORTANT: this value is read EXACTLY ONCE at module import time
    and cached in module-level ``_FRESH_PIDFILE_SECONDS``. Changing
    the env var at runtime has NO effect on an already-running daemon
    — the daemon must be restarted (or the module re-imported via
    ``importlib.reload``) to pick up the new value. This mirrors the
    "config frozen at process start" convention used for other
    cozempic env vars (e.g. ``COZEMPIC_GUARD_HARD_EXIT_K`` in guard.py).
    Operators tuning the fresh window for a fleet should set the env
    var in their shell rc BEFORE launching Claude Code, not after.

    Clamps to ``(0, _FRESH_MAX]``. Invalid values (non-numeric, NaN,
    inf, ≤0, > _FRESH_MAX) silently fall back to the default — keeps
    the daemon working rather than failing at startup over a
    misconfigured env var. (DA round 1 N3 docstring fold.)
    """
    raw = os.environ.get("COZEMPIC_PIDFILE_FRESH_SECONDS")
    if raw is None:
        return _DEFAULT_FRESH
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_FRESH
    if not math.isfinite(val) or val <= 0 or val > _FRESH_MAX:
        return _DEFAULT_FRESH
    return val


_FRESH_PIDFILE_SECONDS = _read_fresh_window_seconds()


# Initiator strings — mirrored from ``reload_lock.INIT_*`` conventions
# for operator-triage parity. The parent-vs-daemon split is the only
# meaningful distinction for spawn-claim payloads:
#
#   spawn-claim-parent  — written by ``DaemonSpawnClaim._claim`` when
#                         the SessionStart hook (parent process) wins
#                         the O_CREAT|O_EXCL race
#   spawn-claim-daemon  — written by the atomic ``os.rename`` hand-off
#                         in ``start_guard_daemon`` after Popen
#
# Operators inspecting a stale pidfile can ``cat /tmp/cozempic_guard_*.pid``
# and see immediately who wrote it and when (PR #93 commit 3, item #5).
INIT_SPAWN_PARENT = "spawn-claim-parent"
INIT_SPAWN_DAEMON = "spawn-claim-daemon"


def _parse_pidfile_pid(pid_path: Path) -> int:
    """Read PID from a cozempic guard pidfile (1-line or 3-line format).

    Tolerates both:
      - legacy single-line ``<pid>\\n`` (v1.8.14 and earlier)
      - new 3-line ``<pid>\\n<iso-timestamp>\\n<initiator>\\n``
        (v1.8.15+, parity with ``reload_lock._ReloadLock``)

    Returns 0 on any parse failure (missing file, empty file, garbled
    first line, OSError on read). Callers treat 0 as "no live pid here".
    Mirrors ``DaemonSpawnClaim._read_existing_pid`` semantics and
    ``reload_lock._read_lock_metadata``'s tolerant parsing.

    Bash callers (``hooks.json``) should use ``head -n 1 $path`` — they
    only need line 1 and ``cat`` would feed multi-token output to
    ``kill -0``, which is undefined-behaviour across shells.
    """
    try:
        content = pid_path.read_text()
    except OSError:
        return 0
    if not content:
        return 0
    lines = content.splitlines()
    if not lines:
        return 0
    first = lines[0].strip()
    if not first:
        return 0
    try:
        pid = int(first)
    except ValueError:
        return 0
    return pid if pid > 0 else 0


# XF-1 parity: lowercase-only char class, matching reload_lock._SAFE_CHARS_RE.
# Lowercasing BEFORE substitution ensures uppercase session_id inputs map to
# the same slug as their lowercase equivalents (XF-1 split-brain fix).
_SAFE_CHARS_RE = re.compile(r"[^a-z0-9_-]")


def _slug_for(session_id: str) -> str:
    """Reduce session_id to a 12-char safe slug for the PID file path.

    Mirrors ``reload_lock._slug_for`` and ``guard._reload_armed_path`` —
    all three lowercase BEFORE substitution so the same session_id always
    maps to the same slug regardless of case (XF-1 parity fix).
    """
    if not session_id:
        return "default"
    if "/" in session_id or "\\" in session_id or session_id.endswith(".jsonl"):
        session_id = Path(session_id).stem
    # Lowercase BEFORE substitution so uppercase letters survive as their
    # lowercase equivalents (a real character), not as the underscore
    # placeholder. Matches guard._pid_file_for_session's flow.
    sanitized = _SAFE_CHARS_RE.sub("_", session_id.lower())
    return sanitized[:12] or "default"


def _spawn_lock_path(session_id: str) -> Path:
    """Compose the spawn-claim PID file path for a session.

    Kept as a separate helper for diagnostics + the legacy
    ``daemon_spawn_lock`` ctx-manager test surface. In production the
    actual PID file used is ``guard._pid_file_for_session`` — see
    ``DaemonSpawnClaim`` which accepts an explicit ``pid_file`` so the
    claim file is the same inode the rest of guard.py reads/writes.
    """
    return Path(tempfile.gettempdir()) / f"cozempic_guard_{_slug_for(session_id)}.pid"


def _is_process_alive(pid: int) -> bool:
    """Return True if ``pid`` is a live process (or owned by another user).

    ``kill(pid, 0)`` returns 0 if the signal could be delivered, raises
    ``ProcessLookupError`` if no such process exists, and
    ``PermissionError`` if the process exists but we lack permission.
    Conservative interpretation: PermissionError → alive (not ours, but
    real), since a guard daemon under another user counts as in-flight.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


class DaemonAlreadyStarting(Exception):
    """Raised when a peer process has already claimed the PID file.

    The caller should treat this as ``already_running=True`` and propagate
    ``holder_pid`` to the result dict so SessionStart hooks can log /
    introspect which PID owns the slot. ``holder_pid == 0`` means we
    couldn't parse the file (stale, empty, garbled) — surface as 0 rather
    than guessing.
    """

    def __init__(self, session_id: str, holder_pid: int = 0):
        self.session_id = session_id
        self.holder_pid = holder_pid
        super().__init__(
            f"Daemon already starting for {session_id} "
            f"(holder PID {holder_pid or 'unknown'})"
        )


class DaemonSpawnClaim:
    """Atomic PID-file claim via O_CREAT|O_EXCL.

    Usage:
        try:
            with DaemonSpawnClaim(session_id, pid_file) as claim:
                proc = subprocess.Popen([...])
                # Atomically replace our parent PID with the daemon PID
                tmp = pid_file.with_suffix(".pid.tmp")
                tmp.write_text(str(proc.pid))
                os.rename(tmp, pid_file)
                # Tell the claim "we wrote the real PID — don't unlink on exit"
                claim.handed_off = True
        except DaemonAlreadyStarting as exc:
            return {"started": False, "already_running": True,
                    "pid": exc.holder_pid, ...}

    On normal exit:
      - If ``handed_off`` is True: leave the PID file in place (daemon
        owns it for its lifetime).
      - If ``handed_off`` is False (caller forgot to set it, or never
        reached the hand-off): unlink — we created it; we own cleanup.

    On exceptional exit:
      - Always unlink so a retry can re-claim. The daemon never started,
        so there's no surviving owner for the PID file.
    """

    def __init__(self, session_id: str, pid_file: Path):
        self.session_id = session_id
        self.pid_file = pid_file
        self.owned = False
        self.handed_off = False  # caller sets True after atomic rename

    def __enter__(self) -> "DaemonSpawnClaim":
        self._claim()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self.owned:
            return
        # Unlink on exception path OR on clean exit when caller didn't
        # complete the hand-off (defensive: caller bug = retry-able state).
        if exc_type is not None or not self.handed_off:
            try:
                self.pid_file.unlink(missing_ok=True)
            except OSError:
                pass

    def _claim(self) -> None:
        """Acquire the PID file via O_CREAT|O_EXCL.

        On EEXIST: read the existing PID. The file is treated as a live
        peer claim (DaemonAlreadyStarting) when ANY of:
          - the holder PID is alive (``kill(pid, 0)`` succeeds), OR
          - the file is younger than ``_FRESH_PIDFILE_SECONDS`` (a peer
            wrote it moments ago, even if the spawned process hasn't fully
            started — or if we're in a test where the spawn is mocked and
            the recorded PID is fictitious).

        Only when BOTH the PID is dead AND the file is older than the
        fresh window do we classify the claim as stale and unlink + retry.
        This prevents the N>=3 contention race where peer A renames the
        pidfile to a fresh PID, peer B reads it and (with a dead test-mock
        PID) incorrectly classifies it as stale and re-claims.
        """
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW

        try:
            fd = os.open(str(self.pid_file), flags, 0o600)
        except FileExistsError:
            holder_pid = self._read_existing_pid()
            holder_alive = holder_pid > 0 and _is_process_alive(holder_pid)
            fresh = self._is_pidfile_fresh()
            if holder_alive or fresh:
                raise DaemonAlreadyStarting(self.session_id, holder_pid=holder_pid)
            # Stale (dead PID AND old file): unlink and retry exactly once.
            try:
                self.pid_file.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                fd = os.open(str(self.pid_file), flags, 0o600)
            except FileExistsError:
                # A peer reclaimed between our unlink and our retry.
                holder_pid = self._read_existing_pid()
                raise DaemonAlreadyStarting(self.session_id, holder_pid=holder_pid)

        # Won the claim. Write our parent PID + timestamp + initiator
        # so concurrent readers see a real, alive PID (not a placeholder)
        # AND operators inspecting the pidfile have triage metadata.
        # Three-line payload mirrors ``reload_lock._ReloadLock._try_create``
        # (PR #93 commit 3, item #5 — operator-triage parity). Readers
        # use ``_parse_pidfile_pid`` which tolerates both 1-line legacy
        # and 3-line new formats.
        try:
            payload = (
                f"{os.getpid()}\n"
                f"{datetime.now().isoformat(timespec='seconds')}\n"
                f"{INIT_SPAWN_PARENT}\n"
            )
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)
        self.owned = True

    def _is_pidfile_fresh(self) -> bool:
        """Return True if the PID file's mtime is within the fresh window.

        Used to defeat the dead-pid + stale-classification race: a peer
        that just wrote a real PID may have that PID die between the
        write and our read (test mock, fast crash, PID reuse). A fresh
        file with a dead PID is still a peer claim, not a stale file.

        EACCES handling (DA round 1 M2): if we can't stat() the file
        (PermissionError — e.g. multi-user /tmp with restrictive
        permissions, SELinux), return True. Conservative: treat an
        unreadable pidfile as a live peer claim rather than re-claiming
        over what might be a real peer's slot. Returning False would
        race the peer and let two daemons spawn for the same session.
        Other OSErrors (ENOENT — file vanished between exists() and
        stat() — most likely) return False so we can re-claim cleanly.
        """
        try:
            mtime = self.pid_file.stat().st_mtime
        except PermissionError:
            # EACCES: can't read mtime → assume fresh (don't race a
            # potentially-live peer claim).
            return True
        except OSError:
            return False
        return (time.time() - mtime) < _FRESH_PIDFILE_SECONDS

    def _read_existing_pid(self) -> int:
        """Best-effort read of the PID currently in the file.

        Delegates to module-level ``_parse_pidfile_pid`` for DRY parity
        with all other pidfile readers (guard.py, doctor.py). Tolerates
        both legacy 1-line and new 3-line payload formats.
        """
        return _parse_pidfile_pid(self.pid_file)


@contextmanager
def daemon_spawn_lock(session_id: str) -> Iterator[Path]:
    """Back-compat wrapper: claim via DaemonSpawnClaim against the canonical
    spawn-lock path. Yields the PID file path on success; raises
    ``DaemonAlreadyStarting`` on contention.

    Kept for ``tests/test_spawn_lock.py`` unit tests + any direct callers.
    Production code in ``guard.py`` uses ``DaemonSpawnClaim`` directly so
    it can pass the canonical pidfile path and set ``handed_off`` after
    the atomic rename.
    """
    pid_file = _spawn_lock_path(session_id)
    claim = DaemonSpawnClaim(session_id, pid_file)
    try:
        claim.__enter__()
    except DaemonAlreadyStarting:
        raise
    try:
        yield pid_file
    finally:
        # ctx-manager surface has no hand-off concept — always unlink on
        # exit so this matches the prior ``daemon_spawn_lock`` lifecycle
        # (sentinel removed on release). Test surface relies on this.
        claim.handed_off = False
        claim.__exit__(None, None, None)


_HAVE_FCNTL = False  # exported for back-compat; we no longer use fcntl
