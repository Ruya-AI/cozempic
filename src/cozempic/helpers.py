"""Shared helper functions for message inspection and manipulation."""

from __future__ import annotations

import copy
import json as _json
import os
import tempfile as _tempfile
from pathlib import Path as _Path

_SAVINGS_FILE = _Path.home() / ".cozempic_savings.json"


# ── Atomic write primitive ──────────────────────────────────────────────────
#
# Used by all single-writer-per-host paths (_save_sidecar, record_savings,
# save_messages, doctor.fix_corrupted_tool_use). Each call uses a unique
# tempfile name via mkstemp so two concurrent writers don't clobber each
# other's tmp file mid-rename. fsync before replace guarantees the new bytes
# are durable before the rename, so power-loss or OOM-kill leaves the target
# either fully-old or fully-new — never zeroed.

def atomic_write_text(target: _Path, data: str, encoding: str = "utf-8") -> None:
    """Atomic, collision-safe text write.

    Two concurrent calls on the same `target` BOTH succeed without losing
    each other's tmp file (each gets a unique mkstemp name). The final
    `os.replace` is atomic; last writer wins for the target content, but
    neither raises FileNotFoundError from a stolen tmp file.

    For read-modify-write workflows (e.g. record_savings), callers must
    additionally wrap the read+modify+write cycle in a file lock to prevent
    lost-update races — atomic-write alone doesn't protect against that.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = _tempfile.mkstemp(
        prefix=".tmp.", suffix=target.name, dir=str(target.parent)
    )
    tmp_path = _Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync unsupported on some filesystems — atomicity still
                # provided by os.replace, just without durability guarantee.
                pass
        os.replace(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


class _HostFileLock:
    """Per-host advisory lock around a file path.

    Used to serialize read-modify-write cycles on shared state files
    (cozempic-sessions.json, .cozempic_savings.json). The lock is keyed
    on a companion `.lock` file alongside the target; the target itself
    is never opened by the lock.

    POSIX: fcntl.flock — blocks other processes that take the same lock.
    Windows: msvcrt.locking — same semantics on a per-byte basis.
    Unknown platform: degrades to no-op (best-effort, no crash).
    """
    def __init__(self, target: _Path):
        self._lock_path = target.parent / f"{target.name}.lock"
        self._fh = None

    def __enter__(self):
        try:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._lock_path, "a")
            if os.name == "nt":
                import msvcrt
                # Lock first byte; blocking
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            # Lock unavailable — degrade to no-op. Race window remains
            # but writes are still atomic per atomic_write_text.
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
            self._fh = None
        return self

    def __exit__(self, *_):
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                # Rewind to byte 0 to unlock the same byte we locked
                try:
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        try:
            self._fh.close()
        except OSError:
            pass


def record_savings(tokens_saved: int, total_tokens: int = 0, turn_count: int = 0) -> None:
    """Add tokens saved to the lifetime tracker. Called after successful prune+reload.

    If total_tokens and turn_count are provided, estimates extra turns gained
    from the freed headroom.

    Atomic-safe: read-modify-write is wrapped in a host-wide flock so two
    concurrent prune cycles don't lose increments. Write itself uses mkstemp
    for collision safety. Both layers degrade to best-effort on platforms
    without fcntl/msvcrt.
    """
    if tokens_saved <= 0:
        return
    try:
        with _HostFileLock(_SAVINGS_FILE):
            try:
                data = _json.loads(_SAVINGS_FILE.read_text()) if _SAVINGS_FILE.exists() else {}
            except Exception:
                data = {}
            data["tokens_saved"] = data.get("tokens_saved", 0) + tokens_saved
            data["tokens_processed"] = data.get("tokens_processed", 0) + total_tokens
            data["prune_count"] = data.get("prune_count", 0) + 1
            if "since" not in data:
                from datetime import date
                data["since"] = date.today().isoformat()

            # Estimate extra turns gained from freed headroom
            if turn_count > 0 and total_tokens > 0:
                avg_per_turn = total_tokens / turn_count
                if avg_per_turn > 0:
                    extra_turns = int(tokens_saved / avg_per_turn)
                    data["turns_gained"] = data.get("turns_gained", 0) + extra_turns

            try:
                atomic_write_text(_SAVINGS_FILE, _json.dumps(data))
            except Exception:
                pass
    except Exception:
        # Never let savings tracking crash the prune cycle
        pass

    # Ping global counters (anonymous, no user data, quick with short timeout)
    if os.environ.get("COZEMPIC_NO_TELEMETRY"):
        return
    try:
        from urllib.request import Request, urlopen
        urlopen(Request("https://cozempic-counters.counterapi-ruya.workers.dev/counter/prunes/up",
                       headers={"User-Agent": "cozempic"}), timeout=2)
        if tokens_saved < 100_000:
            bucket = "saved_under_100k"
        elif tokens_saved < 500_000:
            bucket = "saved_100k_500k"
        elif tokens_saved < 1_000_000:
            bucket = "saved_500k_1m"
        else:
            bucket = "saved_over_1m"
        urlopen(Request(f"https://cozempic-counters.counterapi-ruya.workers.dev/counter/{bucket}/up",
                       headers={"User-Agent": "cozempic"}), timeout=2)
    except Exception:
        pass


def get_savings_line() -> str | None:
    """Return a single-line lifetime savings summary, or None if no savings recorded."""
    try:
        if not _SAVINGS_FILE.exists():
            return None
        data = _json.loads(_SAVINGS_FILE.read_text())
        total = data.get("tokens_saved", 0)
        processed = data.get("tokens_processed", 0)
        count = data.get("prune_count", 0)
        turns = data.get("turns_gained", 0)
        since = data.get("since", "")
        if total <= 0:
            return None
        if total >= 1_000_000:
            tok_str = f"{total / 1_000_000:.1f}M"
        elif total >= 1_000:
            tok_str = f"{total / 1_000:.0f}K"
        else:
            tok_str = str(total)

        # Session extension multiplier: processed / (processed - saved)
        remaining = processed - total
        multiplier = f"{processed / remaining:.1f}x" if remaining > 0 else ""

        parts = [f"Cozempic: {tok_str} tokens saved"]
        if multiplier:
            parts.append(f"{multiplier} longer sessions")
        if turns > 0:
            parts.append(f"~{turns} extra turns")
        return " | ".join(parts)
    except Exception:
        return None
import json


def msg_bytes(msg: dict) -> int:
    """Calculate the serialized byte size of a message."""
    return len(json.dumps(msg, separators=(",", ":")).encode("utf-8"))


def get_msg_type(msg: dict) -> str:
    """Get the type field from a message."""
    return msg.get("type", "unknown")


def get_content_blocks(msg: dict) -> list[dict]:
    """Extract content blocks from a message's inner message object."""
    m = msg.get("message", {})
    content = m.get("content", [])
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return []


def content_block_bytes(block: dict) -> int:
    """Calculate the serialized byte size of a content block."""
    return len(json.dumps(block, separators=(",", ":")).encode("utf-8"))


def set_content_blocks(msg: dict, blocks: list[dict]) -> dict:
    """Return a deep copy of msg with content blocks replaced."""
    msg = copy.deepcopy(msg)
    if "message" in msg:
        msg["message"]["content"] = blocks
    return msg


def shell_quote(s: str) -> str:
    """Single-quote a string for shell use."""
    return "'" + s.replace("'", "'\\''") + "'"


def is_ssh_session() -> bool:
    """Detect if we're running inside an SSH session."""
    import os
    return bool(
        os.environ.get("SSH_TTY")
        or os.environ.get("SSH_CONNECTION")
        or os.environ.get("SSH_CLIENT")
    )


_PROTECTED_TYPES = frozenset({
    "content-replacement",
    "marble-origami-commit",
    "marble-origami-snapshot",
    "worktree-state",
    "task-summary",
})


def is_protected(msg: dict) -> bool:
    """Return True if this entry must NEVER be removed or structurally modified."""
    t = msg.get("type", "")
    if t in _PROTECTED_TYPES:
        return True
    if t == "user" and msg.get("isCompactSummary"):
        return True
    if t == "system" and msg.get("subtype") in ("compact_boundary", "microcompact_boundary"):
        return True
    if msg.get("isVisibleInTranscriptOnly"):
        return True
    if msg.get("__cozempic_behavioral_digest__"):
        return True
    if msg.get("__cozempic_team_protected__"):
        return True
    # P0-D — last-of-type metadata singleton (ai-title / last-prompt /
    # permission-mode). Tagged in place by executor._tag_last_of_metadata_types
    # before strategies run; stripped after run_prescription completes.
    if msg.get("__cozempic_metadata_singleton__"):
        return True
    return False


def find_active_background_tasks(messages: list) -> list[dict]:
    """Find background tasks that were spawned but have no completion result.

    Returns list of {tool_use_id, description} for each active task.
    """
    import re
    spawns: dict[str, str] = {}  # tool_use_id -> description
    completions: set[str] = set()

    for _, msg, _ in messages:
        inner = msg.get("message", {})
        content = inner.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "tool_use" and block.get("name") == "Task":
                        inp = block.get("input", {})
                        if inp.get("run_in_background"):
                            spawns[block.get("id", "")] = inp.get("description", "")
                    if block.get("type") == "tool_result":
                        completions.add(block.get("tool_use_id", ""))

        # Check queue-operation for completed tasks
        if msg.get("type") == "queue-operation":
            body = str(msg.get("content", "") or msg.get("body", ""))
            if "<status>completed</status>" in body or "<status>failed</status>" in body:
                m = re.search(r"<tool-use-id>(.*?)</tool-use-id>", body)
                if m:
                    completions.add(m.group(1))

    return [
        {"tool_use_id": tid, "description": desc}
        for tid, desc in spawns.items()
        if tid not in completions
    ]


def text_of(block: dict) -> str:
    """Get the text content of a content block, handling all block types."""
    result = block.get("text", "") or block.get("thinking", "") or block.get("content", "")
    if isinstance(result, list):
        return " ".join(
            sub.get("text", "") for sub in result if isinstance(sub, dict)
        )
    if not isinstance(result, str):
        return ""
    return result
