"""Session discovery and I/O for Claude Code JSONL files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from .types import Message


# ─── Concurrent-write safety primitives ──────────────────────────────────────

class PruneConflictError(Exception):
    """The session file's original bytes changed during pruning.

    Raised by save_messages() when a snapshot is provided and the file's
    prefix was mutated (re-written or truncated) between snapshot time and
    replace time.  The caller should discard the pruned output and retry
    from a fresh load on the next cycle.
    """


class PruneLockError(Exception):
    """The prune lock is held by another process or guard cycle."""


class _FileSnapshot:
    """Immutable point-in-time identity of a JSONL file.

    Captures inode, size, and an MD5 of the full content so that save_messages()
    can classify what happened to the file while pruning was in progress:

    - "unchanged"  — byte-for-byte identical; safe to replace normally.
    - "appended"   — file grew; all original bytes are intact as prefix.
                     Delta lines can be appended to the pruned output.
    - "conflict"   — inode changed, file shrank, or prefix was mutated.
                     Caller must abort and retry.
    """
    __slots__ = ("inode", "size", "content_hash")

    def __init__(self, path: Path) -> None:
        st = path.stat()
        self.inode: int = st.st_ino
        self.size: int = st.st_size
        self.content_hash: str = hashlib.md5(path.read_bytes()).hexdigest()

    def classify(self, path: Path) -> Literal["unchanged", "appended", "conflict"]:
        """Classify what happened to the file since this snapshot was taken."""
        try:
            st = path.stat()
        except OSError:
            return "conflict"
        if st.st_ino != self.inode:
            return "conflict"
        if st.st_size == self.size:
            return "unchanged"
        if st.st_size > self.size:
            data = path.read_bytes()
            if hashlib.md5(data[: self.size]).hexdigest() == self.content_hash:
                return "appended"
        return "conflict"

    def read_delta(self, path: Path) -> bytes:
        """Return bytes appended since snapshot. Caller must verify 'appended' first."""
        return path.read_bytes()[self.size :]


def snapshot_session(path: Path) -> _FileSnapshot:
    """Snapshot a session file's identity before loading, for append-safe writes."""
    return _FileSnapshot(path)


def _parse_delta_lines(delta: bytes) -> list[str]:
    """Parse appended bytes into validated JSONL lines.

    Raises ValueError if the delta does not end on a newline boundary (Claude
    mid-write) or json.JSONDecodeError if any line is not valid JSON.
    Returns a list of raw JSON line strings (no trailing newline per element).
    """
    text = delta.decode("utf-8", errors="replace")
    if not text.endswith("\n"):
        raise ValueError("delta does not end on newline boundary — Claude may be mid-write")
    lines = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        json.loads(raw)  # validates; raises json.JSONDecodeError if corrupt
        lines.append(raw)
    return lines


class _PruneLock:
    """Advisory lock preventing concurrent prune cycles on the same session file.

    Uses fcntl.LOCK_EX|LOCK_NB on a companion .prune-lock file so two guard
    instances (or a guard + a manual `cozempic treat --execute`) cannot race
    each other.  Falls back silently to a no-op on platforms without fcntl
    (Windows).
    """

    def __init__(self, session_path: Path) -> None:
        self._lock_path = session_path.with_suffix(".prune-lock")
        self._fh = None

    def __enter__(self) -> "_PruneLock":
        try:
            import fcntl
            self._fh = open(self._lock_path, "w", encoding="utf-8")
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            self._fh = None  # Windows — skip locking
        except OSError as exc:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            raise PruneLockError(
                f"Another prune cycle is active for {self._lock_path.name}"
            ) from exc
        return self

    def __exit__(self, *_) -> None:
        if self._fh is not None:
            try:
                import fcntl
                fcntl.flock(self._fh, fcntl.LOCK_UN)
            except Exception:
                pass
            self._fh.close()
            self._fh = None
        self._lock_path.unlink(missing_ok=True)


def get_claude_dir() -> Path:
    import os
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        return Path(config_dir)
    return Path.home() / ".claude"


def get_claude_json_path() -> Path:
    import os
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        return Path(config_dir) / ".claude.json"
    return Path.home() / ".claude.json"


def get_projects_dir() -> Path:
    """Return the Claude projects directory."""
    return get_claude_dir() / "projects"


def find_project_dirs(project_filter: str | None = None) -> list[Path]:
    """Find project directories, optionally filtered by name."""
    projects = get_projects_dir()
    if not projects.exists():
        return []
    dirs = sorted(projects.iterdir())
    if project_filter:
        dirs = [d for d in dirs if project_filter.lower() in d.name.lower()]
    return [d for d in dirs if d.is_dir()]


def find_sessions(project_filter: str | None = None) -> list[dict]:
    """Find all JSONL session files with metadata."""
    sessions = []
    for proj_dir in find_project_dirs(project_filter):
        for f in sorted(proj_dir.glob("*.jsonl")):
            if ".jsonl.bak" in f.name or f.name.endswith(".bak"):
                continue
            if f.name.startswith("."):
                # atomic-write temp orphan (.tmp.*) or any dotfile — never a real
                # session, and a crash between mkstemp and os.replace can leave one
                # that the *.jsonl glob would otherwise enumerate as a phantom session.
                continue
            size = f.stat().st_size
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            session_id = f.stem
            line_count = 0
            with open(f, "r", encoding="utf-8") as fh:
                for _ in fh:
                    line_count += 1
            sessions.append({
                "path": f,
                "project": proj_dir.name,
                "session_id": session_id,
                "size": size,
                "mtime": mtime,
                "lines": line_count,
            })
    return sessions


def cwd_to_project_slug(cwd: str | None = None) -> str:
    """Convert a working directory path to the Claude project slug format.

    Claude stores projects under ~/.claude/projects/ replacing every
    non-alphanumeric character with a single '-' (1:1, no run collapsing).

    Examples:
      /Users/foo/topstep_automation -> -Users-foo-topstep-automation
      /Users/foo/.claude            -> -Users-foo--claude  (dot → dash, double-dash)
    """
    if cwd is None:
        cwd = os.getcwd()
    cwd = os.path.normpath(cwd)
    return re.sub(r"[^a-zA-Z0-9]", "-", cwd)


def project_slug_to_path(slug: str) -> str:
    """Convert a Claude project slug back to a directory path.

    e.g. -Users-foo-myproject -> /Users/foo/myproject
    """
    # Slug starts with '-' because paths start with '/'
    return slug.replace("-", "/")


def find_claude_pid() -> int | None:
    """Walk up the process tree to find the Claude Code node process."""
    try:
        pid = os.getpid()
        for _ in range(10):
            result = subprocess.run(
                ["ps", "-o", "ppid=,comm=", "-p", str(pid)],
                capture_output=True, text=True,
            )
            parts = result.stdout.strip().split(None, 1)
            if len(parts) < 2:
                break
            ppid, comm = int(parts[0]), parts[1]
            if "node" in comm.lower() or "claude" in comm.lower():
                return pid
            pid = ppid
            if pid <= 1:
                break
    except (ValueError, OSError):
        pass

    # No Claude ancestor found. Do not fall back to the immediate parent PID:
    # detached guards can be reparented under systemd --user, and treating that
    # parent as Claude can terminate the whole desktop session on reload.
    return None


def _session_id_from_process() -> str | None:
    """Detect the current session ID from Claude's open file descriptors.

    Claude keeps .claude/tasks/<session-id>/ directories open. We can use
    lsof to find the session UUID from the parent Claude process.
    """
    claude_pid = find_claude_pid()
    if not claude_pid:
        return None

    try:
        result = subprocess.run(
            ["lsof", "-p", str(claude_pid)],
            capture_output=True, text=True, timeout=5,
        )
        import re
        # Match UUID pattern in .claude/tasks/ paths
        uuids = re.findall(
            r'\.claude/tasks/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})',
            result.stdout,
        )
        if uuids:
            # Return the most common one (in case of duplicates)
            from collections import Counter
            return Counter(uuids).most_common(1)[0][0]
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def _match_session_by_text(sessions: list[dict], match_text: str) -> dict | None:
    """Find a session by matching text in its last N lines.

    Searches the tail of each session file for the given text snippet.
    Useful when multiple sessions are active and CWD/process detection fails.
    """
    for sess in sorted(sessions, key=lambda s: s["mtime"], reverse=True):
        try:
            with open(sess["path"], "r", encoding="utf-8") as f:
                # Read last 50 lines efficiently
                lines = f.readlines()
                tail = lines[-50:] if len(lines) > 50 else lines
                tail_text = "".join(tail)
                if match_text in tail_text:
                    return sess
        except (OSError, UnicodeDecodeError):
            continue
    return None


def find_current_session(
    cwd: str | None = None,
    match_text: str | None = None,
    strict: bool = False,
) -> dict | None:
    """Find the current Claude Code session using multiple strategies.

    Detection priority:
    1. Active transcript: the session Claude Code itself reported to the
       SessionStart hook, keyed by the live Claude PID (authoritative — follows
       CC's own active session, immune to cwd/project mismatch).
    2. Process-based: lsof on parent Claude process to find session UUID
    3. Text matching: search session files for a unique text snippet
    4. CWD slug: match working directory against project directory names
    5. Fallback: most recently modified session (only when strict=False)

    When strict=True, Strategy 5 is disabled — callers that perform
    destructive writes must not proceed on an ambiguous match.
    """
    sessions = find_sessions()
    if not sessions:
        return None

    # Strategy 1: Active transcript reported by Claude Code to the SessionStart
    # hook, keyed by the live Claude PID. This is the ONLY signal that follows
    # CC's own notion of the active session, so it must win over cwd-inference
    # (which mis-resolved when the active session lived in a different project
    # dir than cwd — the f464a40c wrong-session incident).
    rec = lookup_active_transcript()
    if rec:
        tp = rec.get("transcript_path", "")
        for s in sessions:
            if str(s["path"]) == tp:
                return s
        # The transcript exists (lookup verified it) but wasn't enumerated by
        # find_sessions() — e.g. a project-dir filter. Build a record from it so
        # we still follow CC's active session rather than guessing from cwd.
        p = Path(tp)
        if p.exists():
            try:
                st = p.stat()
                return {
                    "path": p,
                    "project": p.parent.name,
                    "session_id": p.stem,
                    "size": st.st_size,
                    "mtime": datetime.fromtimestamp(st.st_mtime),
                    "lines": sum(1 for _ in open(p, "r", encoding="utf-8")),
                }
            except OSError:
                pass

    # Strategy 2: Process-based detection (open .claude/tasks/ dirs)
    proc_session_id = _session_id_from_process()
    if proc_session_id:
        for s in sessions:
            if s["session_id"] == proc_session_id:
                return s

    # Strategy 3: Text matching (for multi-session disambiguation)
    if match_text:
        matched = _match_session_by_text(sessions, match_text)
        if matched:
            return matched

    # Strategy 4: CWD slug match — exact, not substring.
    # Substring caused prefix collisions: '-Users-x-foo' IN '-Users-x-foobar'.
    # Worktrees get their own project dir so exact-match is always correct.
    slug = cwd_to_project_slug(cwd)
    matching = [s for s in sessions if s["project"] == slug]
    if matching:
        return max(matching, key=lambda s: s["mtime"])

    # Strategy 5: Fallback to most recently modified
    # Disabled in strict mode — refuse to guess on destructive paths.
    if strict:
        return None
    return max(sessions, key=lambda s: s["mtime"])


def resolve_session(
    session_arg: str,
    project_filter: str | None = None,
    strict: bool = False,
) -> Path:
    """Resolve a session argument to a JSONL file path.

    Accepts: full path, UUID, UUID prefix, or "current" for auto-detection.
    When strict=True, auto-detection refuses to fall back to "most recent session".
    """
    if session_arg == "current":
        sess = find_current_session(strict=strict)
        if sess:
            return sess["path"]
        print("Error: Could not auto-detect current session.", file=sys.stderr)
        if strict:
            print("Cannot determine session unambiguously — use an explicit session ID.", file=sys.stderr)
        print("Use 'cozempic list' to find the session ID.", file=sys.stderr)
        sys.exit(1)

    p = Path(session_arg)
    if p.exists() and p.suffix == ".jsonl":
        return p

    for sess in find_sessions(project_filter):
        if sess["session_id"] == session_arg:
            return sess["path"]
        if sess["session_id"].startswith(session_arg):
            return sess["path"]

    print(f"Error: Cannot find session '{session_arg}'", file=sys.stderr)
    print("Use 'cozempic list' to see available sessions.", file=sys.stderr)
    sys.exit(1)


# ─── Session sidecar store ────────────────────────────────────────────────────
#
# Maps session_id → {cwd, context_window, created_at, last_seen_at}.
# Populated by the guard daemon at startup and refreshed on each checkpoint.
# Consumers (reload, guard resume) prefer this over slug reversal, which is
# ambiguous for paths containing hyphens.

_SIDECAR_FILENAME = "cozempic-sessions.json"
_SIDECAR_MAX_ENTRIES = 200


def get_sidecar_path() -> Path:
    """Return the path to the session sidecar store."""
    return get_claude_dir() / _SIDECAR_FILENAME


def _load_sidecar() -> dict:
    """Load the sidecar store. Returns {} on missing, corrupt, or non-dict file
    (a top-level list/string would otherwise crash the `.get()` callers)."""
    p = get_sidecar_path()
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_sidecar(data: dict) -> None:
    """Atomically write the sidecar store via mkstemp.

    Two concurrent guard daemons calling record_session can no longer
    collide on a shared `.tmp` filename (the bug that caused
    `FileNotFoundError: cozempic-sessions.tmp -> cozempic-sessions.json`
    in production when SessionStart fires twice within the same ms).
    Use record_session() if you need read-modify-write atomicity — it
    wraps load+modify+save in a host-wide flock to prevent lost updates.
    """
    from .helpers import atomic_write_text
    atomic_write_text(get_sidecar_path(), json.dumps(data, indent=2))


def record_session(
    session_id: str,
    cwd: str,
    context_window: int | None = None,
    nudge_tiers: "list | tuple | None" = None,
) -> None:
    """Record or refresh a session's cwd and context window in the sidecar store.

    Called from the guard daemon at startup and on each checkpoint so the map
    stays current across long-running sessions. Capped at _SIDECAR_MAX_ENTRIES
    (oldest last_seen_at evicted first) to prevent unbounded growth.

    ``nudge_tiers`` records the guard's RESOLVED reload-tier fractions (soft/hard1/
    hard2 as fractions of the window) so the Stop-hook nudge fires at the points
    the guard actually reloads — even when the user raised the reload threshold.
    """
    if not session_id or not cwd:
        return
    # Wrap read+modify+write in a host-wide flock so two concurrent guard
    # daemons don't lose each other's updates. atomic_write_text inside
    # _save_sidecar handles the tmp-file collision; this lock handles
    # the lost-update race that atomic-write alone can't fix.
    from .helpers import _HostFileLock
    with _HostFileLock(get_sidecar_path()):
        data = _load_sidecar()
        existing = data.get(session_id, {})
        now = datetime.now().isoformat(timespec="seconds")
        data[session_id] = {
            "cwd": cwd,
            "context_window": (
                context_window if context_window is not None
                else existing.get("context_window")
            ),
            "nudge_tiers": (
                [round(float(t), 4) for t in nudge_tiers] if nudge_tiers
                else existing.get("nudge_tiers")
            ),
            "created_at": existing.get("created_at", now),
            "last_seen_at": now,
        }
        if len(data) > _SIDECAR_MAX_ENTRIES:
            by_age = sorted(data, key=lambda k: data[k].get("last_seen_at", ""), reverse=True)
            data = {k: data[k] for k in by_age[:_SIDECAR_MAX_ENTRIES]}
        _save_sidecar(data)


def get_session_nudge_tiers(session_id: str) -> "list | None":
    """Return the guard's recorded reload-tier fractions for a session, or None."""
    if not session_id:
        return None
    rec = _load_sidecar().get(session_id)
    tiers = rec.get("nudge_tiers") if isinstance(rec, dict) else None
    if isinstance(tiers, (list, tuple)) and tiers:
        try:
            vals = sorted(float(t) for t in tiers if 0.0 < float(t) <= 1.0)
            return vals or None
        except (TypeError, ValueError):
            return None
    return None


# ─── Active-transcript store (follows Claude Code's own session) ──────────────
#
# The MANUAL CLI path (`cozempic current`, `/cozempic reload`) cannot ask Claude
# Code which session is live, so it historically inferred it from cwd → project
# slug → most-recent file. That picks the WRONG session whenever the active
# session lives in a different project dir than cwd (the f464a40c incident:
# active under -Users-ruya, cwd mapped to the Cozempic dir whose most-recent
# session was a stale one).
#
# Claude Code DOES know the active session: it hands the SessionStart hook a
# `transcript_path`. We record that path keyed by the LIVE Claude PID — one
# claude process == exactly one session, which disambiguates precisely where cwd
# cannot. `find_current_session` consults this FIRST. The record self-expires:
# dead PIDs are pruned on every write, and a vanished transcript falls through.

_ACTIVE_SESSIONS_FILENAME = "cozempic-active-sessions.json"
_ACTIVE_SESSIONS_MAX = 64


def _active_sessions_path() -> Path:
    return get_claude_dir() / _ACTIVE_SESSIONS_FILENAME


def _pid_alive(pid: int) -> bool:
    """True if `pid` is a live process. Signal 0 probes without delivering."""
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, OverflowError, ValueError):
        return False
    except PermissionError:
        # Exists but owned by another user — still alive.
        return True
    except OSError:
        return False


def record_active_transcript(transcript_path: str, claude_pid: int | None = None) -> None:
    """Record the active session's transcript keyed by the live Claude PID.

    Called from the SessionStart hook path (which receives `transcript_path` and
    runs as a child of the live claude process). Best-effort: never raises into
    the hook. Dead PIDs are pruned on write so the store stays small and honest.
    """
    try:
        if not transcript_path:
            return
        tp = Path(transcript_path)
        if not tp.exists() or tp.suffix != ".jsonl":
            return
        if claude_pid is None:
            claude_pid = find_claude_pid()
        if not claude_pid:
            return
        from .helpers import _HostFileLock
        with _HostFileLock(_active_sessions_path()):
            try:
                raw = _active_sessions_path().read_text(encoding="utf-8")
                data = json.loads(raw)
                if not isinstance(data, dict):
                    data = {}
            except (OSError, ValueError):
                data = {}
            # Prune dead pids and the entry we're about to overwrite.
            data = {k: v for k, v in data.items()
                    if k != str(claude_pid) and _pid_alive(k)}
            data[str(claude_pid)] = {
                "transcript_path": str(tp),
                "session_id": tp.stem,
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
            }
            if len(data) > _ACTIVE_SESSIONS_MAX:
                by_age = sorted(data, key=lambda k: data[k].get("recorded_at", ""), reverse=True)
                data = {k: data[k] for k in by_age[:_ACTIVE_SESSIONS_MAX]}
            from .helpers import atomic_write_text
            atomic_write_text(_active_sessions_path(), json.dumps(data, indent=2))
    except Exception:
        # The hook must never fail because we couldn't record the active session.
        pass


def lookup_active_transcript(claude_pid: int | None = None) -> dict | None:
    """Return the active-transcript record for a live Claude PID, or None.

    Honours staleness: a record whose PID is dead, or whose transcript file no
    longer exists, returns None so the caller falls through to other strategies.
    """
    try:
        if claude_pid is None:
            claude_pid = find_claude_pid()
        if not claude_pid:
            return None
        try:
            data = json.loads(_active_sessions_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        rec = data.get(str(claude_pid))
        if not isinstance(rec, dict):
            return None
        if not _pid_alive(claude_pid):
            return None
        tp = rec.get("transcript_path", "")
        if not tp or not Path(tp).exists():
            return None
        return rec
    except Exception:
        return None


def get_session_cwd(session_id: str) -> str | None:
    """Return the recorded cwd for a session from the sidecar store, or None."""
    if not session_id:
        return None
    rec = _load_sidecar().get(session_id)
    return rec.get("cwd") if isinstance(rec, dict) else None


def get_session_context_window(session_id: str) -> int | None:
    """Return the recorded context window for a session from the sidecar, or None."""
    if not session_id:
        return None
    rec = _load_sidecar().get(session_id)
    return rec.get("context_window") if isinstance(rec, dict) else None


# ─── JSONL I/O ────────────────────────────────────────────────────────────────

MAX_LINE_BYTES = 10 * 1024 * 1024  # 10MB per-line safety limit


def _parse_one_line(raw: str, idx: int) -> Message | None:
    """Parse a single stripped JSONL line into a Message tuple.

    Returns None if the line is empty or oversized (skip). Matches the
    behaviour shared by load_messages() and _parse_jsonl_chunk() — both
    full-read and incremental-read paths route through this helper to
    guarantee identical byte-length accounting and _parse_error shape.
    """
    stripped = raw.strip()
    if not stripped:
        return None
    if len(stripped) > MAX_LINE_BYTES:
        print(
            f"  Warning: skipping oversized line {idx} ({len(stripped)} bytes)",
            file=sys.stderr,
        )
        return None
    byte_len = len(stripped.encode("utf-8"))
    try:
        return (idx, json.loads(stripped), byte_len)
    except json.JSONDecodeError:
        return (idx, {"_raw": stripped, "_parse_error": True}, byte_len)


def load_messages(path: Path) -> list[Message]:
    """Load JSONL file. Returns list of (line_index, message_dict, byte_size)."""
    messages: list[Message] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parsed = _parse_one_line(line, i)
            if parsed is not None:
                messages.append(parsed)
    return messages


# ─── Incremental JSONL read (read-only scan path) ───────────────────────────
#
# The guard daemon's main loop checkpoints the session every ~30s by calling
# load_messages() and scanning the result to extract team state. On a long-
# running session the full-read pattern produces a large allocation each
# cycle; even though Python frees the list, libmalloc's LARGE_REUSABLE zone
# retains the chunks. Over hours this manifests as unbounded RSS growth.
#
# load_messages_incremental() keeps a per-path cache of parsed messages and
# advances by byte offset on subsequent calls. Appends pay only the cost of
# the newly-written bytes. Rewrites (prune via os.replace, truncation) are
# detected via (inode, size, mtime_ns) and trigger a full re-read.
#
# The function is READ-ONLY — do NOT use it on mutation paths (prune cycles,
# save roundtrips). Those still need full-read semantics paired with
# _FileSnapshot for append-aware conflict detection.

MAX_CACHED_MESSAGES = 5000  # per-session cache cap; evicts oldest on overflow
MAX_CACHE_SESSIONS = 8      # LRU cap on distinct session paths held at once


@dataclass
class _CacheEntry:
    messages: list[Message] = field(default_factory=list)
    offset: int = 0       # byte position after the last fully-parsed newline
    mtime_ns: int = 0
    size: int = 0
    inode: int = 0
    next_line_index: int = 0  # running file-line counter for Message tuples


# OrderedDict supports move_to_end / popitem(last=False) for LRU bookkeeping.
# The per-path cache covers the guard daemon (one session) but also any
# library-API consumer that iterates many sessions in a long-lived process.
_INCR_CACHE: "OrderedDict[Path, _CacheEntry]" = OrderedDict()
_INCR_LOCK = threading.Lock()


def _parse_jsonl_chunk(
    chunk: str, start_line_index: int
) -> tuple[list[Message], int]:
    """Parse a newline-delimited JSONL chunk. Returns (messages, lines_consumed).

    Empty lines advance the line counter but are not emitted (matches
    load_messages behaviour). Oversized lines are warned and skipped but
    still consume a line index.
    """
    out: list[Message] = []
    lines_consumed = 0
    for offset, raw in enumerate(chunk.splitlines()):
        lines_consumed += 1
        parsed = _parse_one_line(raw, start_line_index + offset)
        if parsed is not None:
            out.append(parsed)
    return out, lines_consumed


def load_messages_incremental(path: Path) -> list[Message]:
    """Return parsed JSONL messages using a byte-offset cache.

    Equivalent to load_messages() on the happy path: same tuple shape, same
    ordering, same error handling. Diverges only for files larger than
    MAX_CACHED_MESSAGES — the cache retains the newest N entries, so the
    returned list is likewise truncated. Callers that need full historical
    state (prune, save roundtrip) must use load_messages() instead.

    Invalidation: inode change (os.replace), size shrink (truncation), or
    mtime regression trigger a full re-read. Partial trailing lines (no
    terminating newline) are deferred until the write completes.

    Thread-safe via a module-global lock.
    """
    path = Path(path)
    key = path.resolve()
    with _INCR_LOCK:
        try:
            st = path.stat()
        except OSError:
            _INCR_CACHE.pop(key, None)
            return []

        entry = _INCR_CACHE.get(key)
        # Same-size in-place rewrite (open('r+')): inode holds, size holds,
        # but mtime advances. Treat that as a cache-miss — otherwise the
        # early-exit would return the pre-rewrite content.
        needs_full_read = (
            entry is None
            or st.st_ino != entry.inode
            or st.st_size < entry.size
            or st.st_mtime_ns < entry.mtime_ns
            or (st.st_mtime_ns > entry.mtime_ns and st.st_size == entry.size)
        )

        if needs_full_read:
            entry = _CacheEntry(inode=st.st_ino)
            _INCR_CACHE[key] = entry
            start_offset = 0
        elif st.st_size == entry.size and st.st_mtime_ns == entry.mtime_ns:
            _INCR_CACHE.move_to_end(key)
            return list(entry.messages)
        else:
            start_offset = entry.offset

        with open(path, "rb") as f:
            f.seek(start_offset)
            raw_bytes = f.read(st.st_size - start_offset)

        # Stop at the last complete line — a trailing partial line means the
        # writer is mid-append. We'll pick up the remainder on the next call.
        last_newline = raw_bytes.rfind(b"\n")
        if last_newline == -1:
            # No complete lines in the new region yet; leave cache untouched.
            _INCR_CACHE.move_to_end(key)
            return list(entry.messages)

        complete = raw_bytes[: last_newline + 1]
        try:
            chunk = complete.decode("utf-8")
        except UnicodeDecodeError:
            chunk = complete.decode("utf-8", errors="replace")

        new_messages, lines_consumed = _parse_jsonl_chunk(
            chunk, entry.next_line_index
        )
        entry.messages.extend(new_messages)
        entry.next_line_index += lines_consumed
        entry.offset = start_offset + (last_newline + 1)
        entry.size = st.st_size
        entry.mtime_ns = st.st_mtime_ns

        if len(entry.messages) > MAX_CACHED_MESSAGES:
            # Retain the newest MAX_CACHED_MESSAGES; byte-offset tracking is
            # independent of what we hold in memory.
            del entry.messages[:-MAX_CACHED_MESSAGES]

        _INCR_CACHE.move_to_end(key)
        while len(_INCR_CACHE) > MAX_CACHE_SESSIONS:
            _INCR_CACHE.popitem(last=False)

        return list(entry.messages)


def save_messages(
    path: Path,
    messages: list[Message],
    create_backup: bool = True,
    snapshot: _FileSnapshot | None = None,
) -> Path | None:
    """Save messages back to JSONL, optionally creating a timestamped backup.

    When *snapshot* is provided (taken via snapshot_session() before load_messages()),
    the file is classified before replacing:

    - "unchanged"  — safe to replace; proceeds normally.
    - "appended"   — Claude wrote new lines while pruning was in progress; the
                     delta is validated and appended to the pruned output so no
                     messages are lost.
    - "conflict"   — prefix was mutated (rewrite or truncation); raises
                     PruneConflictError.  The backup is NOT created and the
                     original file is left untouched.

    Returns the backup path if created, else None.
    """
    # Use mkstemp for collision-safe tmp filename. Previously this was
    # path.with_suffix(".tmp") — two concurrent prune cycles on the same
    # session (which the _PruneLock should prevent, but doctor.py and
    # cmd_reload bypassed it) collided on a single tmp path, causing
    # FileNotFoundError on the loser's os.replace.
    import tempfile as _tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = _tempfile.mkstemp(
        prefix=".tmp." + path.name + ".", suffix=".partial", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for _, msg, _ in messages:
                if msg.get("_parse_error"):
                    f.write(msg["_raw"] + "\n")
                else:
                    f.write(json.dumps(msg, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())

        # ── Append-aware conflict detection ──────────────────────────────────
        if snapshot is not None:
            state = snapshot.classify(path)
            if state == "conflict":
                tmp_path.unlink(missing_ok=True)
                raise PruneConflictError(
                    f"Session file was modified (prefix changed) while pruning: {path}"
                )
            if state == "appended":
                delta = snapshot.read_delta(path)
                try:
                    extra_lines = _parse_delta_lines(delta)
                except (ValueError, json.JSONDecodeError) as exc:
                    # Claude is mid-write — treat as conflict; retry next cycle.
                    tmp_path.unlink(missing_ok=True)
                    raise PruneConflictError(
                        f"Session file has an incomplete append — deferring prune: {path}"
                    ) from exc
                if extra_lines:
                    with open(tmp_path, "a", encoding="utf-8") as fa:
                        for line in extra_lines:
                            fa.write(line + "\n")
                        fa.flush()
                        os.fsync(fa.fileno())
        # ─────────────────────────────────────────────────────────────────────

        # Backup is created after conflict check so orphaned backups are not
        # left behind when a conflict causes an early return.
        backup_path: Path | None = None
        if create_backup:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = path.with_suffix(f".{ts}.jsonl.bak")
            shutil.copy2(path, backup_path)

        try:
            os.replace(tmp_path, path)
        except PermissionError as exc:
            # Windows (#112): Claude Code holds the live transcript open
            # without FILE_SHARE_DELETE, so os.replace onto it raises
            # PermissionError [WinError 5]. This is not a hard failure —
            # defer the prune to the next cycle, exactly like an
            # incomplete-append conflict. Clean up the tmp file and the
            # just-created backup so a deferred cycle leaves no orphans.
            tmp_path.unlink(missing_ok=True)
            if backup_path is not None:
                backup_path.unlink(missing_ok=True)
            raise PruneConflictError(
                f"Session file is held open by another process — deferring prune: {path}"
            ) from exc
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return backup_path


def cleanup_old_backups(session_path: Path, keep: int = 3) -> int:
    """Delete old timestamped .jsonl.bak files for this session, keeping the newest `keep`.

    Prevents disk fill when the guard fires many prune cycles (#19).
    Returns the number of files deleted.
    """
    pattern = f"{session_path.stem}.*.jsonl.bak"
    bak_files = sorted(
        session_path.parent.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    deleted = 0
    for old in bak_files[keep:]:
        try:
            old.unlink()
            deleted += 1
        except OSError:
            pass
    return deleted
