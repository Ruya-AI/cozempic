"""Single-flight reload coordinator.

Cozempic has THREE independent code paths that can spawn a reload watcher
(treat session → kill Claude → spawn watcher → osascript new terminal):

  1. `cozempic reload` (cli.py:cmd_reload) — user-initiated
  2. `guard_prune_cycle` (guard.py) — auto-fired at HARD1/HARD2 thresholds
  3. `OverflowRecovery._do_recover` (overflow.py) — reactive overflow detection

Without coordination, all three can fire simultaneously. The production
cascade we observed: user typed `/cozempic treat reload` while the guard
daemon was already crossing its 55% threshold. Both spawned watchers. When
the user typed `/exit`, both watchers detected Claude's death and both opened
new terminals via osascript. Two Claudes attached to the same JSONL → session
conflict → one exited mid-startup → confusing UX.

This module provides a per-session lock that all three paths consult before
spawning a watcher. The first to acquire wins; others see `ReloadLockHeld`
and either fail-fast (CLI), defer to next cycle (guard), or skip (overflow).

The lock file lives at `{tempfile.gettempdir()}/cozempic_reload_<sid:12>.lock`
and contains `<pid>\\n<iso-timestamp>\\n<initiator>\\n`. Stale locks (holder
PID is dead) are auto-cleared. Wedged locks (holder alive but lock age > 30s)
surface as `ReloadLockHeld(wedged=True)` — operator action required.

The session_id is normalized + sanitized to match the existing PID-file naming
convention. Validation prevents path traversal via crafted session IDs.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional


# Wedge threshold: a normal kill+resume cycle is <10s. 60s gives generous
# headroom for slow osascript / Terminal startup / SSH-tunneled X-forwarding
# / cold-cache macOS Terminal startup without letting a truly wedged process
# block reloads for too long. Empirical: tmux+osascript on a busy macOS box
# can plausibly take 30+ seconds; 60s avoids false-wedged classification.
WEDGE_TTL_SECONDS = 60

# Initiator strings — used in lock file contents + error messages for triage.
INIT_CLI_RELOAD = "cli-reload"
INIT_GUARD_HARD1 = "guard-hard1"
INIT_GUARD_HARD2 = "guard-hard2"
INIT_OVERFLOW = "overflow"

# Sentinel constants — guard the reload window to prevent transient-daemon races.
#
# The sentinel file lives at {tempfile.gettempdir()}/cozempic_reload_<slug>.in-flight
# and is written by _terminate_and_resume BEFORE spawning the watcher, so any
# concurrent SessionStart hook (upgrade-chain re-fire, parallel Tab) that calls
# start_guard_daemon sees the sentinel and skips the spawn.
#
# SENTINEL_TTL_SECONDS is longer than WEDGE_TTL_SECONDS (60s) because it guards
# the full reload chain: SIGTERM → watcher detach → osascript → Terminal startup
# → claude -r auth. The watcher unlinks it after osascript fires (async unlink),
# so the TTL is a safety net for the watcher-SIGKILL scenario only.
SENTINEL_TTL_SECONDS = 120
INIT_RELOAD_SENTINEL = "reload-sentinel"

# Session ID sanitization — matches _pid_file_for_session in guard.py.
# UUIDs are hex+dashes (32 chars + 4 dashes), but session_id can be passed
# as a path (e.g. from $TRANSCRIPT) which we normalize. Strip to first 12
# chars for the lock filename so paths and UUIDs both produce a short slug.
# Lowercase-only char class: we .lower() before substitution (XF-1 fix) so
# uppercase letters survive as their lowercase equivalent rather than as '_'.
_SAFE_CHARS_RE = re.compile(r"[^a-z0-9_-]")


def _slug_for(session_id: str) -> str:
    """Reduce session_id to a 12-char safe slug for the lock filename.

    Mirrors ``spawn_lock._slug_for`` and ``guard._reload_armed_path`` —
    same session_id always produces the same slug across all three producers,
    so the lock, pid file, and reload-sentinel co-locate correctly.

    Lowercases BEFORE substitution (XF-1 fix): uppercase inputs such as a
    UUID received before normalization produce the same slug as their
    lowercase equivalents, preventing split-brain where the lock lives at
    a different path than the pid/sentinel files.
    """
    if not session_id:
        return "default"
    # If it looks like a path, take the basename and drop suffix
    if "/" in session_id or "\\" in session_id or session_id.endswith(".jsonl"):
        session_id = Path(session_id).stem
    # Lowercase BEFORE substitution so uppercase letters survive as their
    # lowercase equivalents (a real character), not as the underscore
    # placeholder. Matches spawn_lock._slug_for and guard._reload_armed_path.
    sanitized = _SAFE_CHARS_RE.sub("_", session_id.lower())
    return sanitized[:12] or "default"


def _lock_path_for(session_id: str) -> Path:
    """Compose the lock file path for a session."""
    slug = _slug_for(session_id)
    return Path(tempfile.gettempdir()) / f"cozempic_reload_{slug}.lock"


def _is_process_alive(pid: int) -> bool:
    """Return True if ``pid`` is a live process (or owned by another user).

    ``kill(pid, 0)`` returns 0 if the signal could be delivered, raises
    ``ProcessLookupError`` if no such process exists, and
    ``PermissionError`` if the process exists but we lack permission.
    Conservative interpretation: PermissionError → alive (not ours, but
    real), since a reload lock held by another user's process must not be
    stolen. Matches spawn_lock._is_process_alive semantics exactly.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # cross-user process is alive; don't steal its lock
    except OSError:
        return False


class ReloadLockHeld(Exception):
    """Raised when the reload lock for a session is held by another process.

    Attributes:
        session_id: the session the lock guards
        holder_pid: PID of the process holding the lock (0 if unknown)
        holder_initiator: which code path acquired the lock
                          (cli-reload, guard-hard1, guard-hard2, overflow)
        age_sec: how long the lock has been held (None if unparseable)
        wedged: True if age > WEDGE_TTL_SECONDS — operator action recommended
    """

    def __init__(
        self,
        session_id: str,
        holder_pid: int = 0,
        holder_initiator: str = "unknown",
        age_sec: Optional[float] = None,
        wedged: bool = False,
    ):
        self.session_id = session_id
        self.holder_pid = holder_pid
        self.holder_initiator = holder_initiator
        self.age_sec = age_sec
        self.wedged = wedged
        age_str = f"{age_sec:.0f}s" if age_sec is not None else "unknown"
        wedge_note = " [WEDGED]" if wedged else ""
        super().__init__(
            f"Reload lock held by {holder_initiator} (PID {holder_pid}, "
            f"{age_str} ago){wedge_note}"
        )


def _read_lock_metadata(lock_path: Path) -> tuple[int, str, Optional[float]]:
    """Parse a lock file. Returns (pid, initiator, age_sec) — best effort.

    On any read/parse failure, returns (0, "unknown", None) so the caller
    can decide whether to treat as stale.
    """
    try:
        content = lock_path.read_text(encoding="utf-8").strip().split("\n")
    except OSError:
        return 0, "unknown", None
    pid = 0
    initiator = "unknown"
    age = None
    if len(content) >= 1:
        try:
            pid = int(content[0].strip())
        except ValueError:
            pass
    if len(content) >= 2:
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(content[1].strip())
            age = max(0.0, time.time() - ts.timestamp())
        except (ValueError, IndexError):
            pass
    if len(content) >= 3:
        initiator = content[2].strip() or "unknown"
    return pid, initiator, age


class _ReloadLock:
    """Per-session single-flight reload lock.

    Context-manager API:
        try:
            with _ReloadLock(session_id, initiator=INIT_CLI_RELOAD):
                # spawn watcher / kill claude / etc
                ...
        except ReloadLockHeld as exc:
            # another process is mid-reload; decide whether to retry / abort
            ...

    Implementation notes:
      - Uses O_CREAT|O_EXCL for atomic acquisition. Two processes racing the
        same lock will have exactly one win (POSIX guarantee).
      - On acquisition failure, reads the existing lock to disambiguate:
        stale (dead PID) → unlink + retry once; wedged (alive PID, age > TTL)
        → raise with wedged=True; fresh (alive PID, age <= TTL) → raise normally.
      - The lock is unlinked on __exit__ regardless of success/failure inside.
      - If the holder process crashes without releasing, the lock becomes
        stale and the next acquirer cleans it up.
    """

    def __init__(self, session_id: str, initiator: str = INIT_CLI_RELOAD):
        self.session_id = session_id
        self.initiator = initiator
        self._lock_path = _lock_path_for(session_id)
        self._owned = False

    def __enter__(self) -> "_ReloadLock":
        self._acquire()
        return self

    def __exit__(self, *_):
        if self._owned:
            try:
                self._lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._owned = False

    def _try_create(self) -> bool:
        """Attempt atomic O_CREAT|O_EXCL. Returns True if we created it.

        Uses O_NOFOLLOW to defeat symlink attacks: a local attacker who
        can write to /tmp could plant a symlink at our lock path pointing
        at an arbitrary file; without O_NOFOLLOW, our O_CREAT would
        follow the link. O_NOFOLLOW makes us fail (with ELOOP) on
        symlink targets, which is the correct conservative behavior.
        """
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        # O_NOFOLLOW exists on POSIX (Linux + macOS); not on Windows
        # where tempfile.gettempdir() is per-user anyway.
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(str(self._lock_path), flags, 0o600)
        except FileExistsError:
            return False
        except OSError:
            return False
        try:
            from datetime import datetime
            payload = (
                f"{os.getpid()}\n"
                f"{datetime.now().isoformat(timespec='seconds')}\n"
                f"{self.initiator}\n"
            )
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)
        self._owned = True
        return True

    def _acquire(self) -> None:
        # First attempt
        if self._try_create():
            return

        # Lock exists. Inspect.
        pid, initiator, age = _read_lock_metadata(self._lock_path)

        # Holder is dead → stale → reclaim
        if pid > 0 and not _is_process_alive(pid):
            try:
                self._lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            # Retry once after stale cleanup
            if self._try_create():
                return
            # Still failed — surface as held
            pid, initiator, age = _read_lock_metadata(self._lock_path)

        # Holder is alive and lock is fresh
        wedged = age is not None and age > WEDGE_TTL_SECONDS
        raise ReloadLockHeld(
            session_id=self.session_id,
            holder_pid=pid,
            holder_initiator=initiator,
            age_sec=age,
            wedged=wedged,
        )


def acquire_with_wait(
    session_id: str,
    initiator: str,
    wait_seconds: float = 30.0,
    poll_interval: float = 0.5,
) -> _ReloadLock:
    """Try to acquire a reload lock, polling for up to `wait_seconds`.

    Used by `cozempic reload --wait` to give the user an opt-in queueing
    behavior instead of fail-fast. The poll interval is short so we don't
    hold up the user when the lock is released quickly.

    Wedged locks (age > WEDGE_TTL_SECONDS) surface immediately — waiting
    on a wedged process is futile; the operator needs to intervene.

    Prints a one-time "queued, waiting" notice on the first failed attempt
    so the user knows we're polling rather than hung.

    Returns an ENTERED lock — caller must use `try/finally` to release.
    Raises `ReloadLockHeld` if the wait expires (or immediately if wedged).
    """
    import sys
    deadline = time.monotonic() + wait_seconds
    notified = False
    while True:
        lock = _ReloadLock(session_id, initiator=initiator)
        try:
            lock._acquire()
            return lock
        except ReloadLockHeld as exc:
            if exc.wedged:
                raise
            if time.monotonic() >= deadline:
                raise
            if not notified:
                print(
                    f"  Queued; waiting up to {wait_seconds:.0f}s for in-flight "
                    f"reload ({exc.holder_initiator}, PID {exc.holder_pid}) to finish...",
                    file=sys.stderr,
                )
                notified = True
            time.sleep(poll_interval)


# ── Reload sentinel (NEW-1 option c) ─────────────────────────────────────────
#
# Complements the reload lock: the lock prevents CONCURRENT reloads (two callers
# racing to spawn a watcher). The sentinel prevents a TRANSIENT guard daemon from
# spawning in the gap between:
#   1. OLD guard's finally-block unlink (slot is FREE)
#   2. NEW Claude's SessionStart spawning the real replacement guard
#
# The sentinel is written by _terminate_and_resume BEFORE spawning the watcher,
# and unlinked by the watcher bash script AFTER osascript fires. Both the bash
# fast-path in hooks.json AND the Python start_guard_daemon path check it.
# _________________________________________________________________________


def _reload_sentinel_path_for(session_id: str) -> Path:
    """Return the sentinel file path for a session.

    Uses /tmp directly (same as _pid_file_for_session in guard.py and as the
    bash hook scripts) for cross-process consistency. tempfile.gettempdir()
    resolves to /var/folders/... on macOS which differs from the /tmp symlink
    that bash scripts use — both point to the same inode on macOS, but Path
    equality checks fail. Using /tmp directly is also more readable.

    Validates that the slug contains no path separators to prevent traversal.
    """
    slug = _slug_for(session_id)
    # _slug_for already returns ≤12 chars. A second [:12] would be a no-op today
    # but would diverge from _lock_path_for (line 102) if _slug_for is ever widened.
    # Belt-and-suspenders: confirm no path separators survived sanitization.
    if "/" in slug or "\\" in slug:
        raise ValueError(
            f"sentinel slug contains path separator (session_id type "
            f"{type(session_id).__name__}, length {len(session_id)})"
        )
    return Path("/tmp") / f"cozempic_reload_{slug}.in-flight"


def _read_sentinel_metadata(sentinel_path: Path) -> tuple[int, Optional[float]]:
    """Parse a sentinel file. Returns (claude_pid, age_sec) — best effort.

    On any read/parse failure, returns (0, None) so the caller can decide
    whether to treat as stale. Does NOT return the initiator field — callers
    only need pid and age for GC decisions.
    """
    try:
        content = sentinel_path.read_text(encoding="utf-8").strip().split("\n")
    except OSError:
        return 0, None
    pid = 0
    age = None
    if len(content) >= 1:
        try:
            pid = int(content[0].strip())
        except ValueError:
            pass
    if len(content) >= 2:
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(content[1].strip())
            age = max(0.0, time.time() - ts.timestamp())
        except (ValueError, IndexError):
            pass
    return pid, age


def write_reload_sentinel(session_id: str, claude_pid: int) -> Path:
    """Write a reload sentinel for session_id, recording claude_pid.

    Uses O_CREAT|O_EXCL|O_WRONLY|O_NOFOLLOW for atomic creation (same
    pattern as _ReloadLock._try_create). If the sentinel already exists
    from a prior leaked reload cycle, unlinks it and retries ONCE.

    Returns the sentinel path.
    Raises OSError on persistent failure (after the retry). Callers should
    wrap in try/except OSError and treat failure as "no sentinel" (degrades
    gracefully: the race window remains but the system doesn't crash).
    """
    from datetime import datetime

    sentinel_path = _reload_sentinel_path_for(session_id)
    payload = (
        f"{claude_pid}\n"
        f"{datetime.now().isoformat(timespec='seconds')}\n"
        f"{INIT_RELOAD_SENTINEL}\n"
    ).encode("utf-8")

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    for attempt in range(2):
        try:
            fd = os.open(str(sentinel_path), flags, 0o600)
            try:
                os.write(fd, payload)
            finally:
                os.close(fd)
            return sentinel_path
        except FileExistsError:
            if attempt == 0:
                # Stale sentinel from a prior reload cycle that leaked.
                # Unlink and retry once — same pattern as _ReloadLock._acquire.
                try:
                    sentinel_path.unlink(missing_ok=True)
                except OSError:
                    pass
                # loop continues for attempt == 1
            else:
                # Retry also failed (another process won the race — their
                # sentinel is valid). Re-raise so the outer try/except decides.
                raise
        except OSError:
            raise

    # Unreachable (loop always returns or raises), but satisfies type checker
    return sentinel_path  # pragma: no cover


def unlink_reload_sentinel(session_id: str) -> None:
    """Best-effort unlink of the reload sentinel.

    Called by:
    - The watcher bash script after osascript fires (async unlink)
    - _terminate_and_resume for the tmux/screen paths (sync unlink at end of block)

    Swallows all OSError (including ENOENT). No CAS needed: the sentinel is
    single-writer (only _terminate_and_resume creates it) and the watcher is
    the only async unlinker. The CAS invariant is maintained by the mtime GC
    in _reload_sentinel_active.
    """
    try:
        _reload_sentinel_path_for(session_id).unlink(missing_ok=True)
    except OSError:
        pass


def _reload_sentinel_active(session_id: str) -> bool:
    """Return True if a FRESH reload sentinel exists for session_id.

    A sentinel is "fresh" if:
      - The file exists AND
      - Its filesystem mtime age is < SENTINEL_TTL_SECONDS

    Uses filesystem mtime (not the ISO timestamp in the file content) for the
    freshness check. mtime is set by the OS on write and can be overridden by
    tests via os.utime, making it the canonical freshness signal. The content
    timestamp is used for diagnostics only.

    Side-effect on stale detection: unlinks the stale sentinel and returns
    False. This GC behavior prevents permanently suppressed spawns when the
    watcher was SIGKILL'd between sentinel write and sentinel unlink.

    Note: this function has a write side-effect despite the interrogative name.
    The full name would be _is_reload_sentinel_active_or_gc, but that is too
    verbose. The docstring makes the side-effect explicit.
    """
    sentinel_path = _reload_sentinel_path_for(session_id)
    if not sentinel_path.exists():
        return False

    # Use mtime for freshness — tests can manipulate it via os.utime, and it's
    # set atomically by the OS on file write (no parse errors possible).
    try:
        age = time.time() - sentinel_path.stat().st_mtime
    except OSError:
        # File disappeared between exists() and stat() — treat as absent
        return False

    if age >= SENTINEL_TTL_SECONDS:
        # Stale — GC it
        try:
            sentinel_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False

    return True
