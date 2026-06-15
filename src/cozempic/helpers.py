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
        prefix=".tmp." + target.name + ".", suffix=".partial", dir=str(target.parent)
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
                # Windows — msvcrt.locking locks bytes from the CURRENT file
                # position. "a" (append) mode leaves the pointer at EOF:
                # byte 0 on a fresh empty lock file, but >0 if a stale
                # non-empty lock file was left from a prior crashed run.
                # __exit__ already rewinds to byte 0 before LK_UNLCK
                # (helpers.py:105), so without this matching seek(0) before
                # LK_LOCK the two operations would target different byte
                # ranges and silently fail to serialize. Mirrors the
                # _SettingsLock fix in init.py (PR #96).
                self._fh.seek(0)
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
        # Version-tagged prune counter: lets us attribute prunes to a release so a
        # future prune-rate spike can be pinned to a specific version (the plain
        # `prunes` counter is version-blind — its UA is just "cozempic"). Cardinality
        # grows ~1 per release; version is already public, no install-id, no PII.
        try:
            from . import __version__ as _cz_ver
            _vtag = "".join(c if (c.isalnum() or c == "_") else "_" for c in _cz_ver.replace(".", "_"))
            urlopen(Request(f"https://cozempic-counters.counterapi-ruya.workers.dev/counter/prunes_v{_vtag}/up",
                           headers={"User-Agent": f"cozempic/{_cz_ver}"}), timeout=2)
        except Exception:
            pass
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

# P0-D — last-of-type metadata singleton tag.
# Defined here (not in executor) so is_protected() can check it without
# importing from executor.py, which would create a circular dependency.
# executor.py imports this constant to apply the tag; strategies call
# is_protected() which reads it. The string value is the sole source of truth.
_METADATA_SINGLETON_KEY: str = "__cozempic_metadata_singleton__"


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
    # P0-D: last-of-type metadata singleton — executor tags the last occurrence
    # of each protected type before strategies run; strip happens after.
    if msg.get(_METADATA_SINGLETON_KEY):
        return True
    # --protect-pattern: user-defined regex protection (#122, @eggrollofchaos).
    # Tagged before prune by tag_pattern_matches(), stripped after in a finally.
    if msg.get(_PATTERN_PROTECTED_KEY):
        return True
    return False


# ── --protect-pattern: user-defined regex prune-immunity (#122, @eggrollofchaos) ──
# Patterns are matched against EVERY message's text on each prune, so two safeguards
# bound the footgun: a length cap per pattern (a crude complexity bound — stdlib `re`
# has no step limit, so a catastrophic-backtracking pattern can't be fully prevented,
# only discouraged + documented), and a cap on the text matched per block. A pattern
# that protects most of the session is WARNED (a too-broad pattern makes the prune a
# no-op — the inert-guard failure cozempic exists to prevent).
_PATTERN_PROTECTED_KEY: str = "__cozempic_pattern_protected__"
_MAX_PROTECT_PATTERN_LEN: int = 1000
_MAX_PROTECT_MATCH_BYTES: int = 256 * 1024
_PROTECT_OVERMATCH_WARN_FRACTION: float = 0.8


def compile_protect_patterns(raw_patterns: list) -> list:
    """Compile --protect-pattern regex strings to compiled patterns. Raises
    ValueError (with context) on an invalid, non-string, or over-long pattern."""
    import re
    compiled = []
    for pat in raw_patterns or []:
        if not isinstance(pat, str):
            raise ValueError(f"protect-pattern must be a string, got {type(pat).__name__}")
        if len(pat) > _MAX_PROTECT_PATTERN_LEN:
            raise ValueError(
                f"protect-pattern too long ({len(pat)} > {_MAX_PROTECT_PATTERN_LEN} chars)")
        try:
            compiled.append(re.compile(pat))
        except re.error as e:
            raise ValueError(f"Invalid protect-pattern regex {pat!r}: {e}") from e
    return compiled


def _iter_msg_texts(msg: dict):
    """Yield every text surface of a message that a prune strategy might remove:
    the raw string content (typed user / queue-operation messages), assistant/user
    `text` blocks, and `tool_result` content. Scanning tool_result text is what makes
    --protect-pattern actually protect prunable content (tool outputs are pruned;
    assistant prose mostly isn't). Each surface is capped at _MAX_PROTECT_MATCH_BYTES."""
    def _cap(t):
        return t[:_MAX_PROTECT_MATCH_BYTES] if isinstance(t, str) and len(t) > _MAX_PROTECT_MATCH_BYTES else t

    root = msg.get("content")
    if isinstance(root, str):
        yield _cap(root)
    inner = msg.get("message") if isinstance(msg.get("message"), dict) else {}
    content = inner.get("content")
    if isinstance(content, str):
        yield _cap(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if t == "text" and isinstance(block.get("text"), str):
                yield _cap(block["text"])
            elif t == "thinking" and isinstance(block.get("thinking"), str):
                yield _cap(block["thinking"])  # thinking-blocks strategy prunes these
            elif t == "tool_result":
                rc = block.get("content")
                if isinstance(rc, str):
                    yield _cap(rc)
                elif isinstance(rc, list):
                    for sub in rc:
                        if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                            yield _cap(sub["text"])


def _msg_text_matches_any(msg: dict, patterns: list) -> bool:
    """True if any compiled pattern matches any text surface of the message
    (string content, text blocks, or tool_result content — see _iter_msg_texts)."""
    for text in _iter_msg_texts(msg):
        if not isinstance(text, str):
            continue
        for p in patterns:
            if p.search(text):
                return True
    return False


class _ProtectMatchTimeout(Exception):
    """Raised when --protect-pattern matching exceeds its wall-clock budget."""


def _protect_match_budget() -> float:
    """Seconds budget for a full --protect-pattern matching pass (#122 hardening).
    stdlib `re` has no step limit, so a catastrophic-backtracking user pattern would
    otherwise hang the prune — worst, the guard daemon which re-scans every cycle.
    COZEMPIC_PROTECT_MATCH_SECONDS overrides; finite, clamped to [0, 60]; 0 disables."""
    import math
    try:
        v = float(os.environ.get("COZEMPIC_PROTECT_MATCH_SECONDS", "2.0"))
    except (TypeError, ValueError):
        return 2.0
    if not math.isfinite(v) or v < 0:
        return 2.0
    return min(v, 60.0)


def _match_time_budget(seconds: float):
    """Best-effort wall-clock budget for regex matching via SIGALRM (POSIX main
    thread only). On Windows or a non-main thread it yields WITHOUT a timeout — the
    pattern-length + per-surface input caps remain the only bound there (documented).
    Returns a context manager."""
    import signal
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        if seconds <= 0 or not hasattr(signal, "SIGALRM"):
            yield
            return

        def _on_alarm(signum, frame):
            raise _ProtectMatchTimeout()

        try:
            old = signal.signal(signal.SIGALRM, _on_alarm)
        except (ValueError, OSError):
            yield  # not the main thread — SIGALRM unavailable
            return
        try:
            signal.setitimer(signal.ITIMER_REAL, seconds)
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)

    return _cm()


def tag_pattern_matches(messages: list, patterns: list) -> int:
    """Tag messages whose text matches any compiled pattern with the
    pattern-protected key so is_protected() spares them. Returns the count newly
    tagged.

    Two safeguards: a wall-clock budget around the whole scan (a runaway regex
    fails OPEN — skip protection this cycle, never hang the daemon); and an
    over-protection warn measured over MATCHABLE messages (those with text the
    pattern could match), so a broad pattern that immunizes all the prunable content
    is flagged even when unmatchable carriers/singletons dilute the total."""
    if not patterns:
        return 0
    count = 0
    matchable = 0
    try:
        with _match_time_budget(_protect_match_budget()):
            for _, msg_dict, _ in messages:
                if not isinstance(msg_dict, dict):
                    continue
                texts = [t for t in _iter_msg_texts(msg_dict) if isinstance(t, str)]
                if texts:
                    matchable += 1
                if msg_dict.get(_PATTERN_PROTECTED_KEY):
                    continue
                if any(p.search(t) for t in texts for p in patterns):
                    msg_dict[_PATTERN_PROTECTED_KEY] = True
                    count += 1
    except _ProtectMatchTimeout:
        import sys
        strip_pattern_tags(messages)  # fail-open: don't leave a half-protected session
        print("  Cozempic: --protect-pattern matching exceeded its time budget — the "
              "pattern is too expensive; skipping pattern protection this cycle "
              "(COZEMPIC_PROTECT_MATCH_SECONDS to tune).", file=sys.stderr)
        return 0
    if matchable and count >= _PROTECT_OVERMATCH_WARN_FRACTION * matchable:
        import sys
        print(f"  Cozempic: --protect-pattern matched {count}/{matchable} matchable "
              f"messages ({100 * count // matchable}%) — pruning may free little; "
              f"consider a narrower pattern.", file=sys.stderr)
    return count


def strip_pattern_tags(messages: list) -> None:
    """Remove the pattern-protected tag from all messages (call in a finally after a
    prune, so the transient tag never persists into the saved session)."""
    for item in messages:
        msg_dict = item[1] if isinstance(item, tuple) and len(item) > 1 else item
        if isinstance(msg_dict, dict):
            msg_dict.pop(_PATTERN_PROTECTED_KEY, None)


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
