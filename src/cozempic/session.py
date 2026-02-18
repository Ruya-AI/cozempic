"""Session discovery and I/O for Claude Code JSONL files."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .types import Message


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

    Claude stores projects under ~/.claude/projects/ using the path with
    slashes replaced by dashes, e.g. /Users/foo/myproject -> -Users-foo-myproject
    """
    import os
    if cwd is None:
        cwd = os.getcwd()
    return cwd.replace("/", "-")


def project_slug_to_path(slug: str) -> str:
    """Convert a Claude project slug back to a directory path.

    e.g. -Users-foo-myproject -> /Users/foo/myproject
    """
    # Slug starts with '-' because paths start with '/'
    return slug.replace("-", "/")


def find_claude_pid() -> int | None:
    """Walk up the process tree to find the Claude Code node process."""
    from .helpers import is_ssh_session

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
    # Fallback: PPID is often Claude when invoked from within a session.
    # Skip over SSH — PPID would be the SSH shell, not Claude.
    if is_ssh_session():
        return None
    ppid = os.getppid()
    if ppid > 1:
        return ppid
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


def find_current_session(cwd: str | None = None, match_text: str | None = None) -> dict | None:
    """Find the current Claude Code session using multiple strategies.

    Detection priority:
    1. Process-based: lsof on parent Claude process to find session UUID
    2. Text matching: search session files for a unique text snippet
    3. CWD slug: match working directory against project directory names
    4. Fallback: most recently modified session across all projects
    """
    sessions = find_sessions()
    if not sessions:
        return None

    # Strategy 1: Process-based detection (most reliable for active sessions)
    proc_session_id = _session_id_from_process()
    if proc_session_id:
        for s in sessions:
            if s["session_id"] == proc_session_id:
                return s

    # Strategy 2: Text matching (for multi-session disambiguation)
    if match_text:
        matched = _match_session_by_text(sessions, match_text)
        if matched:
            return matched

    # Strategy 3: CWD slug match
    slug = cwd_to_project_slug(cwd)
    matching = [s for s in sessions if slug in s["project"]]
    if matching:
        return max(matching, key=lambda s: s["mtime"])

    # Strategy 4: Fallback to most recently modified
    return max(sessions, key=lambda s: s["mtime"])


def resolve_session(session_arg: str, project_filter: str | None = None) -> Path:
    """Resolve a session argument to a JSONL file path.

    Accepts: full path, UUID, UUID prefix, or "current" for auto-detection.
    """
    if session_arg == "current":
        sess = find_current_session()
        if sess:
            return sess["path"]
        print("Error: Could not auto-detect current session.", file=sys.stderr)
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


def load_messages(path: Path) -> list[Message]:
    """Load JSONL file. Returns list of (line_index, message_dict, byte_size)."""
    messages: list[Message] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                messages.append((i, msg, len(line.encode("utf-8"))))
            except json.JSONDecodeError:
                messages.append((i, {"_raw": line, "_parse_error": True}, len(line.encode("utf-8"))))
    return messages


def save_messages(
    path: Path,
    messages: list[Message],
    create_backup: bool = True,
) -> Path | None:
    """Save messages back to JSONL, optionally creating a timestamped backup.

    Returns the backup path if created, else None.
    """
    backup_path = None
    if create_backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_suffix(f".{ts}.jsonl.bak")
        shutil.copy2(path, backup_path)

    with open(path, "w", encoding="utf-8") as f:
        for _, msg, _ in messages:
            if msg.get("_parse_error"):
                f.write(msg["_raw"] + "\n")
            else:
                f.write(json.dumps(msg, separators=(",", ":")) + "\n")

    return backup_path
