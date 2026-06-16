"""Health checks for Claude Code configuration and environment.

The 'doctor' command diagnoses known issues beyond session bloat —
config bugs, oversized sessions, stale backups, and disk usage.
"""

from __future__ import annotations

import json
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .session import find_sessions, get_claude_dir, get_claude_json_path


# Directory where cozempic writes runtime artifacts (.pid / .log / .lock files).
# Exposed as a module-level constant so tests can redirect it to a tmpdir
# without monkeypatching every glob call. Always points at POSIX /tmp today;
# a future Windows port would swap this for tempfile.gettempdir().
_TMP_DIR = Path("/tmp")


@dataclass
class CheckResult:
    """Result of a single health check."""

    name: str
    status: str  # "ok" | "warning" | "issue" | "fixed"
    message: str
    fix_description: str | None = None


# ─── Checks ──────────────────────────────────────────────────────────────────


def check_trust_dialog_hang() -> CheckResult:
    """Check for hasTrustDialogAccepted causing resume hangs.

    On Windows, setting hasTrustDialogAccepted=true in ~/.claude.json
    causes `claude --resume` to hang. The trust dialog initialization
    path is skipped, but resume depends on something it sets up.

    Ref: anthropics/claude-code#18532
    """
    claude_json = get_claude_json_path()

    if not claude_json.exists():
        return CheckResult(
            name="trust-dialog-hang",
            status="ok",
            message=f"No {claude_json} found (fresh install)",
        )

    try:
        data = json.loads(claude_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return CheckResult(
            name="trust-dialog-hang",
            status="warning",
            message=f"Could not read {claude_json}: {e}",
        )

    # Check top-level and project-specific entries
    locations = []
    if data.get("hasTrustDialogAccepted") is True:
        locations.append("top-level")

    for key, value in data.items():
        if isinstance(value, dict) and value.get("hasTrustDialogAccepted") is True:
            locations.append(key[:60])

    if not locations:
        return CheckResult(
            name="trust-dialog-hang",
            status="ok",
            message="Trust dialog flag not set — no issue",
        )

    is_windows = platform.system() == "Windows"
    severity = "issue" if is_windows else "warning"

    return CheckResult(
        name="trust-dialog-hang",
        status=severity,
        message=(
            f"hasTrustDialogAccepted=true in {len(locations)} location(s). "
            f"{'This causes resume hangs on Windows.' if is_windows else 'Known to cause resume hangs on Windows — safe on macOS/Linux.'}"
        ),
        fix_description="Reset hasTrustDialogAccepted to false (trust prompt will reappear once)",
    )


def fix_trust_dialog_hang() -> str:
    """Fix the trust dialog hang by resetting hasTrustDialogAccepted."""
    claude_json = get_claude_json_path()

    if not claude_json.exists():
        return f"No {claude_json} found — nothing to fix."

    try:
        data = json.loads(claude_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return f"Could not read {claude_json}: {e}"

    changed = False

    if data.get("hasTrustDialogAccepted") is True:
        data["hasTrustDialogAccepted"] = False
        changed = True

    for key, value in data.items():
        if isinstance(value, dict) and value.get("hasTrustDialogAccepted") is True:
            value["hasTrustDialogAccepted"] = False
            changed = True

    if not changed:
        return "No hasTrustDialogAccepted=true found — nothing to fix."

    # Backup before modifying
    backup = claude_json.parent / ".claude.json.bak"
    shutil.copy2(claude_json, backup)

    claude_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return f"Reset hasTrustDialogAccepted to false. Backup: {backup}"


def check_oversized_sessions() -> CheckResult:
    """Check for session files large enough to cause resume hangs (>50MB)."""
    sessions = find_sessions()
    large = [s for s in sessions if s["size"] > 50 * 1024 * 1024]

    if not large:
        return CheckResult(
            name="oversized-sessions",
            status="ok",
            message=f"No oversized sessions found ({len(sessions)} sessions checked)",
        )

    sorted_large = sorted(large, key=lambda s: s["size"], reverse=True)
    sizes = ", ".join(
        f"{s['session_id'][:8]}…({s['size'] / 1024 / 1024:.0f}MB)"
        for s in sorted_large[:5]
    )
    cmds = "\n".join(
        f"  cozempic treat {s['session_id'][:8]} -rx aggressive --execute"
        for s in sorted_large
    )

    return CheckResult(
        name="oversized-sessions",
        status="issue",
        message=f"{len(large)} session(s) over 50MB: {sizes}. These will likely hang on resume.",
        fix_description=f"Treat each oversized session:\n{cmds}",
    )


def check_stale_backups() -> CheckResult:
    """Check for old .bak files from previous treatments wasting disk space."""
    claude_dir = get_claude_dir()
    projects_dir = claude_dir / "projects"

    if not projects_dir.exists():
        return CheckResult(
            name="stale-backups",
            status="ok",
            message="No projects directory found",
        )

    bak_files = list(projects_dir.rglob("*.jsonl.bak"))
    if not bak_files:
        return CheckResult(
            name="stale-backups",
            status="ok",
            message="No stale backup files found",
        )

    total_bytes = sum(f.stat().st_size for f in bak_files)
    return CheckResult(
        name="stale-backups",
        status="warning" if total_bytes > 100 * 1024 * 1024 else "ok",
        message=f"{len(bak_files)} backup file(s) using {total_bytes / 1024 / 1024:.1f}MB",
        fix_description="Delete old backups to reclaim disk space" if total_bytes > 100 * 1024 * 1024 else None,
    )


def fix_stale_backups() -> str:
    """Delete stale backup files."""
    claude_dir = get_claude_dir()
    projects_dir = claude_dir / "projects"

    if not projects_dir.exists():
        return "No projects directory found."

    bak_files = list(projects_dir.rglob("*.jsonl.bak"))
    if not bak_files:
        return "No backup files to clean."

    total = 0
    for f in bak_files:
        total += f.stat().st_size
        f.unlink()

    return f"Deleted {len(bak_files)} backup file(s), freed {total / 1024 / 1024:.1f}MB"


def check_disk_usage() -> CheckResult:
    """Check total Claude session disk usage."""
    sessions = find_sessions()
    total = sum(s["size"] for s in sessions)

    if total < 500 * 1024 * 1024:
        status = "ok"
    elif total < 2 * 1024 * 1024 * 1024:
        status = "warning"
    else:
        status = "issue"

    return CheckResult(
        name="disk-usage",
        status=status,
        message=f"{len(sessions)} sessions using {total / 1024 / 1024:.1f}MB total",
        fix_description="Run: cozempic treat <session> -rx standard --execute" if status != "ok" else None,
    )


# ─── stale-tmp-artifacts ─────────────────────────────────────────────────────
# When a guard daemon exits (crash, `kill -9`, reboot) without going through
# its normal shutdown path, it leaves behind a /tmp/cozempic_guard_*.pid file,
# a paired /tmp/cozempic_guard_*.log, and — if the crash happened while a
# SessionStart hook was running — a /tmp/cozempic_hook_*.lock.
#
# Existing cleanup is lazy and per-session: `_is_guard_running_for_session`
# (guard.py:947) only unlinks a stale .pid when a *new* guard for the SAME
# session starts. Crashed sessions that never resume leave the artifact
# forever. Across weeks this accumulates: a real user system was observed
# with 20 .pid files (19 stale), 96 .log files, and 89 orphan .lock files.
# This check surfaces and (with --fix) cleans them.

# Files we must NEVER touch: intentionally-persistent global append logs
# (guard.py:890,904 and cli.py:541,552) and the circuit-breaker state file
# (overflow.py:41) which is designed to survive daemon restarts.
_PROTECTED_TMP_NAMES = frozenset({
    "cozempic_guard.log",
    "cozempic_reload.log",
})
_PROTECTED_TMP_PREFIXES = ("cozempic_breaker_",)


def _is_protected_tmp_artifact(name: str) -> bool:
    """True if the file is an intentionally-persistent global artifact."""
    if name in _PROTECTED_TMP_NAMES:
        return True
    return any(name.startswith(p) for p in _PROTECTED_TMP_PREFIXES)


def _is_live_guard_pid(pid: int) -> bool:
    """Return True when `pid` is both alive AND a cozempic guard process.

    Delegates process-argv verification to the canonical helper in guard.py
    (`_is_cozempic_guard_process`) so PID-reuse defense stays in one place.
    """
    import os as _os

    try:
        _os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    from .guard import _is_cozempic_guard_process
    return _is_cozempic_guard_process(pid)


def _is_lock_held(lock_path: Path) -> bool:
    """Return True if a POSIX flock is currently held on `lock_path`.

    Uses a non-blocking LOCK_EX trylock: if acquisition succeeds the lock
    was orphaned (holder crashed or exited), so we release immediately and
    report it as not-held. If the trylock fails with EAGAIN/EWOULDBLOCK
    another process is still holding it and the file must be preserved.
    On platforms without fcntl (e.g., Windows) conservatively report True
    so we never delete a file whose lock status we cannot verify.
    """
    try:
        import fcntl
    except ImportError:
        return True
    try:
        fd = open(lock_path, "a+")
    except OSError:
        # Can't open — leave the file alone.
        return True
    try:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            return True
        # Acquired — lock was orphaned. Release before returning.
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        return False
    finally:
        fd.close()


def _classify_tmp_artifacts() -> tuple[list[Path], list[Path], list[Path]]:
    """Scan `_TMP_DIR` and partition cozempic artifacts into three buckets:
    stale .pid/.log files, orphan .lock files, and kept files (unused here
    but isolates the pure-IO from the status/message logic).

    Never returns a file listed in `_is_protected_tmp_artifact`.
    """
    stale_pids: list[Path] = []
    orphan_logs: list[Path] = []
    orphan_locks: list[Path] = []

    live_slugs: set[str] = set()
    pid_files: list[Path] = []

    try:
        entries = list(_TMP_DIR.glob("cozempic_guard_*.pid"))
    except OSError:
        entries = []

    from .spawn_lock import _parse_pidfile_pid
    for pid_path in entries:
        if _is_protected_tmp_artifact(pid_path.name):
            continue
        slug = pid_path.stem[len("cozempic_guard_"):]  # strip prefix, keep before .pid
        # Tolerant parse: handles both legacy 1-line and new 3-line
        # pidfile formats (PR #93 item #5). Returns 0 on garble → we
        # classify as stale.
        pid = _parse_pidfile_pid(pid_path)
        if pid <= 0:
            stale_pids.append(pid_path)
            continue
        if _is_live_guard_pid(pid):
            live_slugs.add(slug)
        else:
            stale_pids.append(pid_path)
        pid_files.append(pid_path)

    pid_slugs = {p.stem[len("cozempic_guard_"):] for p in pid_files}

    try:
        log_entries = list(_TMP_DIR.glob("cozempic_guard_*.log"))
    except OSError:
        log_entries = []
    for log_path in log_entries:
        if _is_protected_tmp_artifact(log_path.name):
            continue
        slug = log_path.stem[len("cozempic_guard_"):]
        if slug in live_slugs:
            continue  # paired with a live guard
        if slug in pid_slugs:
            orphan_logs.append(log_path)  # paired with a stale .pid — delete together
        else:
            orphan_logs.append(log_path)  # no pid file at all — orphan

    try:
        lock_entries = list(_TMP_DIR.glob("cozempic_hook_*.lock"))
    except OSError:
        lock_entries = []
    for lock_path in lock_entries:
        if _is_protected_tmp_artifact(lock_path.name):
            continue
        if not _is_lock_held(lock_path):
            orphan_locks.append(lock_path)

    return stale_pids, orphan_logs, orphan_locks


def check_stale_tmp_artifacts() -> CheckResult:
    """Detect accumulated cozempic runtime artifacts left behind by crashed
    or abnormally-terminated guard daemons.

    Surfaces three classes of artifact:
      - stale /tmp/cozempic_guard_*.pid files whose PID is dead or not a
        cozempic guard (PID-reuse safe via `_is_cozempic_guard_process`)
      - /tmp/cozempic_guard_*.log files with no matching live guard
      - /tmp/cozempic_hook_*.lock files not currently held by any flock

    Global append-only files (cozempic_guard.log, cozempic_reload.log) and
    the circuit-breaker state file (cozempic_breaker_*.json) are ignored —
    those are intentionally persistent.
    """
    stale_pids, orphan_logs, orphan_locks = _classify_tmp_artifacts()
    total = len(stale_pids) + len(orphan_logs) + len(orphan_locks)

    if total == 0:
        return CheckResult(
            name="stale-tmp-artifacts",
            status="ok",
            message="No stale cozempic artifacts in /tmp/",
        )

    # Size aggregate — helps surface the "96 log files" case.
    size_bytes = 0
    for path in (*stale_pids, *orphan_logs, *orphan_locks):
        try:
            size_bytes += path.stat().st_size
        except OSError:
            pass

    # Graduated severity: a handful is usually a crashed session or two,
    # ten or more points at a recurring failure mode worth surfacing.
    status = "issue" if total >= 10 else "warning"
    parts = []
    if stale_pids:
        parts.append(f"{len(stale_pids)} stale .pid file(s)")
    if orphan_logs:
        parts.append(f"{len(orphan_logs)} orphan .log file(s)")
    if orphan_locks:
        parts.append(f"{len(orphan_locks)} orphan .lock file(s)")
    detail = ", ".join(parts)

    return CheckResult(
        name="stale-tmp-artifacts",
        status=status,
        message=f"{detail} ({size_bytes / 1024:.0f}KB)",
        fix_description=(
            "Delete artifacts whose owning guard/hook is no longer running "
            "(preserves live PIDs and held locks)"
        ),
    )


def fix_stale_tmp_artifacts() -> str:
    """Delete stale .pid, orphan .log, and orphan .lock files.

    Re-classifies artifacts at fix time (the set may have changed since the
    check ran) so a guard that went live between check and fix is protected.
    Every unlink uses `missing_ok=True` to tolerate races with concurrent
    cleanup from `_is_guard_running_for_session`.
    """
    stale_pids, orphan_logs, orphan_locks = _classify_tmp_artifacts()
    deleted = 0
    for path in (*stale_pids, *orphan_logs, *orphan_locks):
        try:
            path.unlink(missing_ok=True)
            deleted += 1
        except OSError:
            pass
    return f"Deleted {deleted} stale artifact(s)"


def check_corrupted_tool_use() -> CheckResult:
    """Check for corrupted tool_use blocks where parameters are merged into the name field.

    Claude Code can corrupt tool_use blocks during serialization (especially
    with parallel Task calls or after compaction), flattening input parameters
    into the name field. This produces names >200 chars, causing unrecoverable
    400 API errors on resume.

    Ref: anthropics/claude-code#25812
    """
    sessions = find_sessions()
    corrupted_sessions = []

    for sess in sessions:
        try:
            count = _count_corrupted_tool_use(sess["path"])
            if count > 0:
                corrupted_sessions.append((sess, count))
        except (OSError, UnicodeDecodeError):
            continue

    if not corrupted_sessions:
        return CheckResult(
            name="corrupted-tool-use",
            status="ok",
            message=f"No corrupted tool_use blocks found ({len(sessions)} sessions checked)",
        )

    details = ", ".join(
        f"{s['session_id'][:8]}…({count} blocks)"
        for s, count in sorted(corrupted_sessions, key=lambda x: x[1], reverse=True)[:5]
    )
    total = sum(c for _, c in corrupted_sessions)

    return CheckResult(
        name="corrupted-tool-use",
        status="issue",
        message=(
            f"{total} corrupted tool_use block(s) in {len(corrupted_sessions)} session(s): {details}. "
            f"These cause 400 API errors on resume (name >200 chars)."
        ),
        fix_description="Repair corrupted tool_use blocks (restore name + reconstruct input params)",
    )


def _raw_content_blocks(obj) -> list:
    """Content blocks of a RAW-parsed (json.loads, not session._parse_one_line)
    JSONL line, defended against the non-dict shapes the session loader wraps but
    doctor's own scanners see raw: a non-dict top-level line, a non-dict inner
    "message", or a non-list content. Returns [] for any non-conforming shape so
    one poisoned session can't abort the whole doctor run with an AttributeError
    (R4 findings doctor-nondict-*). Callers still isinstance-guard each element.
    """
    if not isinstance(obj, dict):
        return []
    msg = obj.get("message")
    if not isinstance(msg, dict):
        return []
    content = msg.get("content", [])
    return content if isinstance(content, list) else []


def _count_corrupted_tool_use(path: Path) -> int:
    """Count corrupted tool_use blocks in a session file."""
    import json as _json
    count = 0
    with open(path, "r", encoding="utf-8", errors="surrogateescape") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            for block in _raw_content_blocks(obj):
                if not isinstance(block, dict):
                    continue
                name = block.get("name", "")
                if block.get("type") == "tool_use" and isinstance(name, str) and len(name) > 200:
                    count += 1
    return count


def fix_corrupted_tool_use() -> str:
    """Repair corrupted tool_use blocks across all sessions.

    Parses the corrupted name field (which contains flattened XML-style
    parameters) back into proper name + input fields.
    """
    import html
    import re
    import shutil

    from .session import _PruneLock, PruneLockError, _FileSnapshot, _parse_delta_lines, _split_physical_lines

    sessions = find_sessions()
    total_fixed = 0
    sessions_fixed = 0
    skipped_sessions = []

    for sess in sessions:
        path = sess["path"]
        try:
            count = _count_corrupted_tool_use(path)
            if count == 0:
                continue
        except (OSError, UnicodeDecodeError):
            continue

        # Acquire the per-session prune lock so we don't race the guard
        # daemon's prune cycle (which would overwrite our fix mid-write).
        # If guard is actively pruning, skip — user can re-run doctor later.
        try:
            _prune_lock_ctx = _PruneLock(path)
            _prune_lock_ctx.__enter__()
        except PruneLockError:
            skipped_sessions.append(sess["session_id"])
            continue

        # Backup before modifying
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_suffix(f".{ts}.jsonl.bak")
        shutil.copy2(path, backup)

        # Read the file ONCE and derive both the working buffer AND the snapshot
        # from the SAME bytes (snapshot.from_bytes) — so a line Claude appends in
        # the window cannot land in both `lines` and the append-delta (the TOCTOU
        # duplication audit P1). Recovers concurrent appends without clobbering;
        # this function previously read_text→atomic_write with no snapshot at all.
        _raw = path.read_bytes()
        snapshot = _FileSnapshot.from_bytes(path, _raw)
        # STRICT decode — doctor WRITES the buffer back. A non-UTF-8 byte must skip
        # the session (can't safely repair), never be rewritten to U+FFFD.
        try:
            _text = _raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped_sessions.append(sess["session_id"])
            try:
                _prune_lock_ctx.__exit__(None, None, None)
            except Exception:
                pass
            continue
        # _split_physical_lines (not str.splitlines) so a corrupt tool_use on a
        # line that ALSO carries a raw U+2028/U+2029/U+0085 isn't torn into
        # unparseable fragments and silently skipped (4th sibling of the class).
        # Re-append "\n" to keep the keepends "".join(lines) reconstruction, BUT
        # only re-terminate the FINAL line if the original had a trailing newline —
        # otherwise a mid-write partial last line gets a spurious "\n" that splits
        # it into two broken fragments when Claude completes it (confirmed P2).
        _parts = _split_physical_lines(_text)
        lines = [p + "\n" for p in _parts]
        if lines and not _text.endswith(("\n", "\r")):
            lines[-1] = _parts[-1]  # preserve the un-terminated partial last line
        fixed_in_session = 0

        for idx, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            try:
                obj = json.loads(line_stripped)
            except json.JSONDecodeError:
                continue

            changed = False
            for block in _raw_content_blocks(obj):
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use":
                    continue
                name = block.get("name", "")
                if not isinstance(name, str) or len(name) <= 200:
                    continue

                # Parse corrupted name: 'ToolName" key1="val1" key2="val2"...'
                tool_name = name.split('"')[0].strip()
                params = {}
                keys = re.findall(r'(\w+)="', name)
                for i, key in enumerate(keys):
                    marker = key + '="'
                    start = name.index(marker) + len(marker)
                    if i + 1 < len(keys):
                        next_marker = keys[i + 1] + '="'
                        end = name.index(next_marker, start)
                        value = name[start:end].rstrip().rstrip('"')
                    else:
                        value = name[start:].rstrip('"')
                    params[key] = html.unescape(value.strip())

                block["name"] = tool_name
                block["input"] = params
                fixed_in_session += 1
                changed = True

            if changed:
                # ensure_ascii=True (NOT False): a repaired line may carry a lone
                # surrogate (a \udXXX escape CC emits for a sliced astral char) which
                # json.loads turned into a real surrogate char. ensure_ascii=False would
                # re-emit it raw -> atomic_write_text's strict utf-8 write raises
                # UnicodeEncodeError, aborting the whole `doctor --fix` run (R7: the
                # un-swept sibling of the save_messages surrogate fix). ensure_ascii=True
                # re-escapes every surrogate to ASCII \uXXXX (always encodable; the file
                # was strict-decoded above so no real-byte/in-band surrogate is present).
                lines[idx] = json.dumps(obj) + "\n"

        try:
            if fixed_in_session > 0:
                # Single read for both the prefix-check and the delta (no TOCTOU
                # window between classify() and read_delta()).
                kind, _delta_bytes = snapshot.classify_and_delta(path)
                if kind == "conflict":
                    # Claude rewrote/truncated the prefix — our repaired buffer is
                    # stale. Refuse to write (the .bak preserves the pre-fix state);
                    # the user can re-run doctor once the session is idle.
                    skipped_sessions.append(sess["session_id"])
                elif kind == "appended":
                    # Claude appended new lines after our read — keep them so the
                    # repair never drops live turns. Validate the delta (raises if
                    # Claude is mid-write / corrupt → treat as conflict, skip).
                    try:
                        delta = [ln + "\n" for ln in _parse_delta_lines(_delta_bytes)]
                    except Exception:
                        # mid-write (no newline boundary) or corrupt delta — don't risk it
                        skipped_sessions.append(sess["session_id"])
                        delta = None
                    if delta is not None:
                        from .helpers import atomic_write_text
                        atomic_write_text(path, "".join(lines) + "".join(delta))
                        total_fixed += fixed_in_session
                        sessions_fixed += 1
                else:  # "unchanged" — safe to write our repaired buffer
                    from .helpers import atomic_write_text
                    atomic_write_text(path, "".join(lines))
                    total_fixed += fixed_in_session
                    sessions_fixed += 1
        finally:
            try:
                _prune_lock_ctx.__exit__(None, None, None)
            except Exception:
                pass

    skipped_note = ""
    if skipped_sessions:
        skipped_note = (
            f" Skipped {len(skipped_sessions)} session(s) with active guard cycles "
            f"(re-run after guard is idle)."
        )
    if total_fixed == 0:
        return f"No corrupted tool_use blocks found.{skipped_note}"
    return (
        f"Repaired {total_fixed} tool_use block(s) in {sessions_fixed} session(s). "
        f"Backups created.{skipped_note}"
    )


def check_orphaned_tool_results() -> CheckResult:
    """Check for orphaned tool_result blocks missing their matching tool_use.

    The Claude API requires every tool_result to have a corresponding tool_use
    in the preceding message. Orphans cause 400 errors on compact/resume.

    This can happen when pruning strategies remove messages with tool_use blocks
    but leave the paired tool_result in a later message.
    """
    sessions = find_sessions()
    orphaned_sessions = []

    for sess in sessions:
        try:
            count = _count_orphaned_tool_results(sess["path"])
            if count > 0:
                orphaned_sessions.append((sess, count))
        except (OSError, UnicodeDecodeError):
            continue

    if not orphaned_sessions:
        return CheckResult(
            name="orphaned-tool-results",
            status="ok",
            message=f"No orphaned tool_result blocks found ({len(sessions)} sessions checked)",
        )

    details = ", ".join(
        f"{s['session_id'][:8]}…({count} blocks)"
        for s, count in sorted(orphaned_sessions, key=lambda x: x[1], reverse=True)[:5]
    )
    total = sum(c for _, c in orphaned_sessions)

    return CheckResult(
        name="orphaned-tool-results",
        status="issue",
        message=(
            f"{total} orphaned tool_result block(s) in {len(orphaned_sessions)} session(s): {details}. "
            f"These cause 400 API errors on compact/resume."
        ),
        fix_description="Remove orphaned tool_result blocks (matching tool_use was removed by pruning or compaction)",
    )


def _count_orphaned_tool_results(path: Path) -> int:
    """Count orphaned tool_result blocks in a session file."""
    import json as _json

    tool_use_ids: set[str] = set()
    all_results: list[str] = []

    with open(path, "r", encoding="utf-8", errors="surrogateescape") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            for block in _raw_content_blocks(obj):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    use_id = block.get("id", "")
                    if isinstance(use_id, str) and use_id:  # unhashable id -> skip (R6 crash class)
                        tool_use_ids.add(use_id)
                elif block.get("type") == "tool_result":
                    use_id = block.get("tool_use_id", "")
                    if isinstance(use_id, str) and use_id:  # kept str-only: `r not in <set>` below
                        all_results.append(use_id)

    return sum(1 for r in all_results if r not in tool_use_ids)


def fix_orphaned_tool_results() -> str:
    """Remove orphaned tool_result blocks from all sessions.

    Acquires per-session _PruneLock + passes snapshot to save_messages so we
    can't race the guard daemon's prune cycle. Sessions with an active guard
    cycle are skipped (user can re-run after guard is idle) — same protection
    as fix_corrupted_tool_use.
    """
    from .session import (
        _PruneLock, PruneConflictError, PruneLockError,
        load_messages_and_snapshot, save_messages,
    )

    sessions = find_sessions()
    total_fixed = 0
    sessions_fixed = 0
    skipped_sessions = []

    for sess in sessions:
        try:
            count = _count_orphaned_tool_results(sess["path"])
            if count == 0:
                continue
        except (OSError, UnicodeDecodeError):
            continue

        from .executor import fix_orphaned_tool_results as _fix
        path = sess["path"]
        # Read once: messages + snapshot from the SAME bytes so a concurrent
        # append can't be both loaded AND counted in the delta (TOCTOU dup).
        messages, snapshot = load_messages_and_snapshot(path)
        fixed_messages, orphans = _fix(messages)

        if orphans > 0:
            try:
                with _PruneLock(path):
                    save_messages(path, fixed_messages, create_backup=True, snapshot=snapshot)
                total_fixed += orphans
                sessions_fixed += 1
            except PruneLockError:
                skipped_sessions.append(sess["session_id"])
                continue
            except PruneConflictError:
                # Session changed mid-fix — skip rather than corrupt
                skipped_sessions.append(sess["session_id"])
                continue

    skipped_note = ""
    if skipped_sessions:
        skipped_note = (
            f" Skipped {len(skipped_sessions)} session(s) with active guard cycles "
            f"or concurrent appends (re-run after guard is idle)."
        )
    if total_fixed == 0:
        return f"No orphaned tool_result blocks found.{skipped_note}"
    return (
        f"Removed {total_fixed} orphaned tool_result block(s) in {sessions_fixed} session(s). "
        f"Backups created.{skipped_note}"
    )


def check_claude_json_corruption() -> CheckResult:
    """Check for corrupted .claude.json from concurrent session race conditions.

    Multiple Claude Code instances writing to .claude.json simultaneously can
    cause truncated JSON, missing auth keys, or numStartups anomalies. This
    affects all multi-instance users and has been reported 15+ times.

    Ref: anthropics/claude-code#28847, #28923, #28813, #28806
    """
    claude_json = get_claude_json_path()

    if not claude_json.exists():
        return CheckResult(
            name="claude-json-corruption",
            status="ok",
            message=f"No {claude_json} found (fresh install)",
        )

    issues = []

    # Check 1: Is the JSON parseable?
    try:
        raw = claude_json.read_text(encoding="utf-8")
    except OSError as e:
        return CheckResult(
            name="claude-json-corruption",
            status="issue",
            message=f"Cannot read {claude_json}: {e}",
            fix_description="Restore from backup",
        )

    if not raw.strip():
        issues.append("file is empty")
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            issues.append(f"invalid JSON: {e}")
            # Check for truncation
            if raw.rstrip()[-1] not in ("}", "]"):
                issues.append("appears truncated (doesn't end with } or ])")

    if issues:
        return CheckResult(
            name="claude-json-corruption",
            status="issue",
            message=f".claude.json is corrupted: {'; '.join(issues)}",
            fix_description="Restore from most recent valid backup",
        )

    # Check 2: Missing critical keys (auth wiped by race condition)
    data = json.loads(raw)
    if isinstance(data, dict):
        # numStartups reset to very low number is a sign of corruption cascade
        num_startups = data.get("numStartups", 0)
        if isinstance(num_startups, (int, float)) and 0 < num_startups < 5:
            # Check if there are backup files suggesting recent corruption
            backups = list(claude_json.parent.glob(".claude.json.bak*"))
            if backups:
                issues.append(f"numStartups={num_startups} with {len(backups)} backup(s) — possible corruption cascade")

    # Check 3: Look for rapid backup file creation (corruption detection cascade)
    backups = sorted(claude_json.parent.glob(".claude.json.bak*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(backups) > 5:
        import time
        recent = [b for b in backups if time.time() - b.stat().st_mtime < 86400]
        if len(recent) > 3:
            issues.append(f"{len(recent)} backup files created in last 24h — corruption detection cascade")

    if not issues:
        return CheckResult(
            name="claude-json-corruption",
            status="ok",
            message=f".claude.json is valid ({len(raw)} bytes)",
        )

    return CheckResult(
        name="claude-json-corruption",
        status="issue" if any("corrupted" in i or "invalid" in i or "truncated" in i for i in issues) else "warning",
        message=f".claude.json issues: {'; '.join(issues)}",
        fix_description="Restore from most recent valid backup",
    )


def fix_claude_json_corruption() -> str:
    """Restore .claude.json from the most recent valid backup."""
    claude_json = get_claude_json_path()

    # Find backups
    backups = sorted(
        claude_json.parent.glob(".claude.json.bak*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not backups:
        return "No backup files found. Cannot auto-repair."

    # Find the most recent valid backup
    for backup in backups:
        try:
            raw = backup.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict) and len(raw) > 10:
                # Valid backup found — restore it
                # Backup current corrupted file first
                corrupted_backup = claude_json.with_suffix(".json.corrupted")
                if claude_json.exists():
                    shutil.copy2(claude_json, corrupted_backup)
                shutil.copy2(backup, claude_json)
                return f"Restored from {backup.name} ({len(raw)} bytes). Corrupted version saved as {corrupted_backup.name}."
        except (json.JSONDecodeError, OSError):
            continue

    return "No valid backup found among existing backup files."


def check_hooks_trust_flag() -> CheckResult:
    """Check for hasTrustDialogHooksAccepted missing in .claude.json.

    Since v2.1.51, Claude Code introduced a separate hooks trust gate.
    hasTrustDialogHooksAccepted must be true for hooks (SessionStart,
    PreToolUse, etc.) to load. This flag is never written automatically
    even when the user accepts the workspace trust dialog, causing hooks
    to silently fail with no error message.

    Ref: anthropics/claude-code#32424
    """
    claude_json = get_claude_json_path()

    if not claude_json.exists():
        return CheckResult(
            name="hooks-trust-flag",
            status="ok",
            message=f"No {claude_json} found (fresh install)",
        )

    try:
        data = json.loads(claude_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return CheckResult(
            name="hooks-trust-flag",
            status="warning",
            message=f"Could not read {claude_json}: {e}",
        )

    if not isinstance(data, dict):
        return CheckResult(
            name="hooks-trust-flag",
            status="ok",
            message="No project entries found",
        )

    # Find entries where workspace is trusted but hooks trust flag is missing
    missing = []

    if data.get("hasTrustDialogAccepted") is True and not data.get("hasTrustDialogHooksAccepted"):
        missing.append("top-level")

    for key, value in data.items():
        if isinstance(value, dict):
            if value.get("hasTrustDialogAccepted") is True and not value.get("hasTrustDialogHooksAccepted"):
                missing.append(key[:60])

    if not missing:
        return CheckResult(
            name="hooks-trust-flag",
            status="ok",
            message="Hooks trust flag set correctly — hooks will load normally",
        )

    return CheckResult(
        name="hooks-trust-flag",
        status="issue",
        message=(
            f"hasTrustDialogHooksAccepted missing in {len(missing)} location(s). "
            f"Hooks (SessionStart, PreToolUse, etc.) silently blocked even after trust accepted (v2.1.51+ bug). "
            f"Affected: {', '.join(missing[:3])}"
        ),
        fix_description="Set hasTrustDialogHooksAccepted=true for all trusted workspaces",
    )


def fix_hooks_trust_flag() -> str:
    """Fix missing hasTrustDialogHooksAccepted for all trusted workspaces."""
    claude_json = get_claude_json_path()

    if not claude_json.exists():
        return f"No {claude_json} found — nothing to fix."

    try:
        data = json.loads(claude_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return f"Could not read {claude_json}: {e}"

    if not isinstance(data, dict):
        return "No project entries found — nothing to fix."

    changed = 0

    if data.get("hasTrustDialogAccepted") is True and not data.get("hasTrustDialogHooksAccepted"):
        data["hasTrustDialogHooksAccepted"] = True
        changed += 1

    for key, value in data.items():
        if isinstance(value, dict):
            if value.get("hasTrustDialogAccepted") is True and not value.get("hasTrustDialogHooksAccepted"):
                value["hasTrustDialogHooksAccepted"] = True
                changed += 1

    if changed == 0:
        return "No missing hooks trust flags — nothing to fix."

    backup = claude_json.parent / ".claude.json.bak"
    shutil.copy2(claude_json, backup)
    claude_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return f"Set hasTrustDialogHooksAccepted=true in {changed} location(s). Hooks will load on next Claude start."


def check_agent_model_mismatch() -> CheckResult:
    """Check for missing model config that breaks agent team inheritance.

    When Claude Code spawns subagents in agent teams, they may not inherit
    the team lead's model, defaulting to claude-opus-4-7 and causing 403
    errors on custom or restricted endpoints.

    Ref: anthropics/claude-code#32368
    """
    claude_dir = get_claude_dir()
    settings_path = claude_dir / "settings.json"
    teams_dir = claude_dir / "teams"

    # Only relevant if teams are in use
    if not teams_dir.is_dir():
        return CheckResult(
            name="agent-model-mismatch",
            status="ok",
            message="No agent teams directory — model inheritance not applicable",
        )

    try:
        team_dirs = [d for d in teams_dir.iterdir() if d.is_dir()]
    except OSError:
        team_dirs = []

    if not team_dirs:
        return CheckResult(
            name="agent-model-mismatch",
            status="ok",
            message="No agent teams in use — model inheritance not applicable",
        )

    if not settings_path.exists():
        return CheckResult(
            name="agent-model-mismatch",
            status="warning",
            message=(
                f"Agent teams active ({len(team_dirs)} team(s)) but no ~/.claude/settings.json found. "
                "Spawned subagents will default to claude-opus-4-7 regardless of ANTHROPIC_MODEL env var."
            ),
            fix_description='Create ~/.claude/settings.json with {"model": "your-model-id"} to ensure subagents inherit the correct model',
        )

    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return CheckResult(
            name="agent-model-mismatch",
            status="warning",
            message=f"Could not read settings.json: {e}",
        )

    model = settings.get("model") if isinstance(settings, dict) else None
    if not model:
        return CheckResult(
            name="agent-model-mismatch",
            status="warning",
            message=(
                f"Agent teams active ({len(team_dirs)} team(s)) but no model set in ~/.claude/settings.json. "
                "Spawned subagents may default to claude-opus-4-7 causing 403 errors on restricted endpoints."
            ),
            fix_description='Add "model": "claude-opus-4-7" to ~/.claude/settings.json to ensure subagents inherit the correct model',
        )

    return CheckResult(
        name="agent-model-mismatch",
        status="ok",
        message=f"Model '{model}' set in settings.json — subagents should inherit correctly",
    )


def check_cozempic_daemon_running() -> CheckResult:
    """Report whether a cozempic guard daemon is running for the current session.

    Verifies via `_is_cozempic_guard_process` (ps argv match) so a PID-reused
    stranger process doesn't produce a false-positive "daemon running".
    """
    from pathlib import Path as _Path
    import os as _os, glob as _glob
    from .guard import _is_cozempic_guard_process
    from .spawn_lock import _parse_pidfile_pid
    pids_alive: list[int] = []
    for pidf in _glob.glob("/tmp/cozempic_guard_*.pid"):
        # Tolerant parse: handles both legacy 1-line and new 3-line
        # pidfile formats (PR #93 item #5). Returns 0 on garble.
        pid = _parse_pidfile_pid(_Path(pidf))
        if pid <= 0:
            continue
        try:
            _os.kill(pid, 0)
        except OSError:
            continue
        # Verify this PID is actually our guard (not a PID-reused stranger)
        if _is_cozempic_guard_process(pid):
            pids_alive.append(pid)
    if pids_alive:
        return CheckResult(
            name="cozempic-daemon-running",
            status="ok",
            message=f"{len(pids_alive)} guard daemon(s) running (PIDs: {', '.join(str(p) for p in pids_alive[:5])})",
        )
    return CheckResult(
        name="cozempic-daemon-running",
        status="warning",
        message="No guard daemon is running. Background protection is inactive for this session.",
        fix_description=(
            "The daemon starts automatically on next Claude Code session via the "
            "SessionStart hook. Start one now: `cozempic guard --daemon`."
        ),
    )


def check_cozempic_project_init() -> CheckResult:
    """Check that the current working directory's .claude/ has cozempic hooks wired.

    Distinct from check_cozempic_hooks (which inspects ~/.claude/settings.json,
    user-global). This one verifies that the *current project* will start the
    guard daemon on its next session — the most common silent-failure mode
    (user pip-installed cozempic but never ran `cozempic init` in this project).
    """
    from pathlib import Path
    claude_dir = Path.cwd() / ".claude"
    if not claude_dir.exists():
        return CheckResult(
            name="cozempic-project-init",
            status="ok",
            message="Not a Claude project (no .claude/ in cwd) — skipping",
        )

    found = False
    for name in ("settings.json", "settings.local.json"):
        p = claude_dir / name
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        hooks = data.get("hooks", {}) or {}
        for entries in hooks.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                for h in entry.get("hooks", []) or []:
                    from .init import _is_cozempic_command
                    if _is_cozempic_command(str(h.get("command", ""))):
                        found = True
                        break

    if found:
        return CheckResult(
            name="cozempic-project-init",
            status="ok",
            message="Project is initialized — cozempic hooks present in .claude/",
        )
    return CheckResult(
        name="cozempic-project-init",
        status="warning",
        message="Current project is NOT initialized. Guard daemon will not start on session.",
        fix_description="Run: cozempic init (or run any cozempic command — auto-init wires it on first use)",
    )


def check_cozempic_hooks() -> CheckResult:
    """Check that all expected cozempic hooks are wired in settings.json.

    Verifies that SessionStart, PostToolUse, PreCompact, PostCompact, and Stop
    hooks are present. Missing hooks mean protection gaps — e.g., no PostCompact
    means team state isn't re-injected after native compaction.
    """
    claude_dir = get_claude_dir()
    settings_path = claude_dir / "settings.json"

    if not settings_path.exists():
        return CheckResult(
            name="cozempic-hooks",
            status="warning",
            message="No ~/.claude/settings.json found — no hooks configured",
            fix_description="Run: cozempic init",
        )

    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return CheckResult(
            name="cozempic-hooks",
            status="warning",
            message=f"Could not read settings.json: {e}",
        )

    hooks = settings.get("hooks", {})
    expected = {"SessionStart", "PreCompact", "PostCompact", "Stop"}
    missing = []

    for event in expected:
        entries = hooks.get(event, [])
        from .init import _is_cozempic_command
        has_cozempic = any(
            _is_cozempic_command(h.get("command", ""))
            for entry in entries
            for h in entry.get("hooks", [])
        )
        if not has_cozempic:
            missing.append(event)

    if not missing:
        return CheckResult(
            name="cozempic-hooks",
            status="ok",
            message="All cozempic hooks are wired (SessionStart, PreCompact, PostCompact, Stop)",
        )

    return CheckResult(
        name="cozempic-hooks",
        status="warning",
        message=f"Missing cozempic hooks: {', '.join(missing)}. Protection gaps exist.",
        fix_description="Run: cozempic init",
    )


def check_zombie_teams() -> CheckResult:
    """Check for stale/zombie team directories in ~/.claude/teams/.

    Team agents that go idle without completing work become zombies — they
    don't respond to shutdown requests and their team directories accumulate.
    These waste disk space and can confuse team detection.

    Ref: anthropics/claude-code#29908
    """
    teams_dir = get_claude_dir() / "teams"
    if not teams_dir.is_dir():
        return CheckResult(
            name="zombie-teams",
            status="ok",
            message="No teams directory found",
        )

    import time
    team_dirs = [d for d in teams_dir.iterdir() if d.is_dir()]
    if not team_dirs:
        return CheckResult(
            name="zombie-teams",
            status="ok",
            message="No team directories found",
        )

    stale_teams = []
    active_teams = []

    for team_dir in team_dirs:
        config_path = team_dir / "config.json"
        if not config_path.exists():
            # No config = orphaned directory
            stale_teams.append((team_dir.name, "no config.json"))
            continue

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            stale_teams.append((team_dir.name, "corrupted config.json"))
            continue

        # Check age — teams older than 24h with no recent activity are likely stale
        mtime = config_path.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600

        if age_hours > 24:
            member_count = len(config.get("members", []))
            stale_teams.append((team_dir.name, f"{member_count} members, {age_hours:.0f}h old"))
        else:
            active_teams.append(team_dir.name)

    if not stale_teams:
        return CheckResult(
            name="zombie-teams",
            status="ok",
            message=f"{len(active_teams)} active team(s), no stale teams found",
        )

    details = "; ".join(f"{name} ({reason})" for name, reason in stale_teams[:5])
    total_size = 0
    for name, _ in stale_teams:
        team_path = teams_dir / name
        for f in team_path.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size

    return CheckResult(
        name="zombie-teams",
        status="warning" if len(stale_teams) <= 3 else "issue",
        message=f"{len(stale_teams)} stale team(s) ({total_size / 1024:.0f}KB): {details}",
        fix_description="Remove stale team directories (>24h old with no activity)",
    )


def fix_zombie_teams() -> str:
    """Remove stale team directories older than 24 hours."""
    import time

    teams_dir = get_claude_dir() / "teams"
    if not teams_dir.is_dir():
        return "No teams directory found."

    removed = 0
    freed = 0

    for team_dir in list(teams_dir.iterdir()):
        if not team_dir.is_dir():
            continue

        config_path = team_dir / "config.json"

        # Remove if no config or older than 24h
        should_remove = False
        if not config_path.exists():
            should_remove = True
        else:
            mtime = config_path.stat().st_mtime
            if (time.time() - mtime) / 3600 > 24:
                should_remove = True

        if should_remove:
            for f in team_dir.rglob("*"):
                if f.is_file():
                    freed += f.stat().st_size
            shutil.rmtree(team_dir)
            removed += 1

    if removed == 0:
        return "No stale teams to remove."
    return f"Removed {removed} stale team(s), freed {freed / 1024:.0f}KB."


# ─── Registry ────────────────────────────────────────────────────────────────

# (name, check_fn, fix_fn_or_None)
ALL_CHECKS: list[tuple[str, callable, callable | None]] = [
    ("trust-dialog-hang", check_trust_dialog_hang, fix_trust_dialog_hang),
    ("hooks-trust-flag", check_hooks_trust_flag, fix_hooks_trust_flag),
    ("cozempic-hooks", check_cozempic_hooks, None),
    ("cozempic-project-init", check_cozempic_project_init, None),
    ("cozempic-daemon-running", check_cozempic_daemon_running, None),
    ("claude-json-corruption", check_claude_json_corruption, fix_claude_json_corruption),
    ("corrupted-tool-use", check_corrupted_tool_use, fix_corrupted_tool_use),
    ("orphaned-tool-results", check_orphaned_tool_results, fix_orphaned_tool_results),
    ("zombie-teams", check_zombie_teams, fix_zombie_teams),
    ("agent-model-mismatch", check_agent_model_mismatch, None),
    ("oversized-sessions", check_oversized_sessions, None),
    ("stale-backups", check_stale_backups, fix_stale_backups),
    ("stale-tmp-artifacts", check_stale_tmp_artifacts, fix_stale_tmp_artifacts),
    ("disk-usage", check_disk_usage, None),
]


def run_doctor(fix: bool = False) -> list[CheckResult]:
    """Run all health checks. If fix=True, apply available fixes for issues."""
    results = []
    for name, check_fn, fix_fn in ALL_CHECKS:
        # Defense in depth (R7): one poisoned session must NOT abort the whole doctor
        # run. A check/fix that raises is reported as an error for THAT check and the
        # remaining checks/fixes still run.
        try:
            result = check_fn()
        except Exception as _check_exc:
            results.append(CheckResult(name=name, status="error",
                message=f"check raised an unexpected error: {_check_exc!r}"))
            continue
        results.append(result)
        if fix and result.status in ("issue", "warning") and result.fix_description and fix_fn:
            try:
                fix_msg = fix_fn()
            except Exception as _fix_exc:
                result.message += f"\n      Fix raised an unexpected error (skipped): {_fix_exc!r}"
                continue
            # Only claim "fixed" if the issue is actually gone — re-run the check
            # and verify. A no-op fix ("nothing found", "skipped N sessions") or a
            # partial/failed fix previously reported false success (audit P1).
            try:
                recheck = check_fn()
            except Exception:
                recheck = None
            if recheck is not None and recheck.status == "ok":
                result.status = "fixed"
                result.message += f"\n      Fixed: {fix_msg}"
            else:
                # Still not clean — surface the real post-fix state, don't lie.
                if recheck is not None:
                    result.status = recheck.status
                    result.message = recheck.message
                result.message += f"\n      Fix attempted (not fully resolved): {fix_msg}"
    return results
