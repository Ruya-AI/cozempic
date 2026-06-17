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

def atomic_write_text(target: _Path, data: str, encoding: str = "utf-8",
                      errors: str = "strict") -> None:
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
        with os.fdopen(fd, "w", encoding=encoding, errors=errors) as f:
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
    """Get the type field from a message. Non-dict-safe (defense-in-depth: the
    loader now wraps non-dict lines, but this is called widely)."""
    if not isinstance(msg, dict):
        return "unknown"
    return msg.get("type", "unknown")


def get_content_blocks(msg: dict) -> list[dict]:
    """Extract content blocks from a message's inner message object.

    Non-dict-safe at the message / inner-message level (returns [] for a non-dict
    msg or non-dict inner "message"). Returns the content list VERBATIM — it does
    NOT coerce or drop non-dict ELEMENTS: an earlier coercion lost those elements on
    the prune-WRITE path (a strategy that read coerced blocks then wrote them back
    dropped the originals = silent data loss). Consumers that iterate blocks must
    isinstance-guard each element themselves; strategy crashes are contained by the
    executor's per-strategy isolation, and read-only helpers (text_of, the token
    estimator) skip non-dict/non-string elements without writing."""
    if not isinstance(msg, dict):
        return []
    m = msg.get("message")
    if not isinstance(m, dict):
        return []
    content = m.get("content", [])
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return []


def hashable_str(v) -> str:
    """An untrusted block field (tool_use `id` / `name` / `tool_use_id`) coerced to a
    hashable str for SAFE use as a set member, dict key, or `in <set>` test. A non-str
    value (an unhashable list/dict, or an int) becomes "" — the empty string is falsy,
    so the ubiquitous `if tid: set.add(tid)` guard then skips it. This closes the
    recurring TypeError-on-unhashable-key crash class (`cannot use 'list' as a set
    element`) that appears at EVERY prune/guard/safety/executor/doctor site reading a
    block field as a hashable on poisoned JSONL (R6 sibling-miss sweep). Coerce at the
    `tid = ...` assignment so every downstream .add/[key]/.get/`in` is safe at once."""
    return v if isinstance(v, str) else ""


def get_dict_blocks(msg: dict) -> list[dict]:
    """Like get_content_blocks but yields ONLY dict elements — for READ-ONLY
    consumers (diagnosis, recap, token estimation) that never write the blocks
    back, so dropping a non-dict element is safe. This is the shared guarded
    iterator: read-only sites should use it instead of re-implementing a per-site
    isinstance guard (the recurring sibling-miss this PR kept hitting — R5). WRITE
    paths (strategies, executor) MUST keep get_content_blocks (verbatim) + their own
    per-element guard, because they round-trip the list and dropping a non-dict
    element there would be silent data loss."""
    return [b for b in get_content_blocks(msg) if isinstance(b, dict)]


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
# Hard per-surface cap on the no-SIGALRM (Windows / non-main-thread) match path,
# where no wall-clock timer can interrupt a runaway regex. 512 (not 4096): a
# super-linear pattern the quantifier-count detector MISSED is bounded to ~512 chars
# of backtracking — 4096 was ~6.5s/message, 512 is ~64x faster (~0.1s), a blink.
# _pattern_is_redos_risky refuses any pattern with >=2 unbounded quantifiers (the
# necessary condition for exponential ReDoS), so this cap only has to bound the
# residual single-quantifier-ambiguous-alternation case. Trade-off (no-budget path
# only): a legitimate marker past char 512 in one surface may not match — acceptable
# vs a frozen daemon.
_NO_BUDGET_MATCH_CAP: int = 512
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


def _have_sigalrm() -> bool:
    """True iff a real SIGALRM wall-clock budget can be armed here (POSIX main
    thread). On Windows SIGALRM is absent; off the main thread signal.signal
    raises. Factored out so both the budget CM and the Windows fail-closed
    pre-check agree, and so tests can emulate Windows by patching this."""
    import signal
    import threading
    if not hasattr(signal, "SIGALRM"):
        return False
    return threading.current_thread() is threading.main_thread()


# Classic catastrophic-backtracking shapes: a quantified group that is itself
# quantified — (x+)+ , (x*)* , (x+)* — or two unbounded quantifiers adjacent
# across a group close. Conservative (may false-positive); used ONLY to fail
# CLOSED where no wall-clock budget exists (Windows / non-main thread), never to
# relax the POSIX SIGALRM path.
def _count_variable_quantifiers(pattern: str) -> int:
    """Count VARIABLE-WIDTH repetition operators (`*`, `+`, open-ended `{n,}`, and a
    bounded range `{n,m}` with m != n) in PATTERN, ignoring escaped operators (`\\*`)
    and operators inside a character class `[...]` (literal there). A fixed `{n}` /
    `{n,n}` and `?` are single/bounded width and are NOT counted.

    This is the PROVABLY-SAFE necessary-condition for super-linear backtracking:
    catastrophic/polynomial ReDoS REQUIRES at least two overlapping variable-width
    repetitions. So `>= 2` NEVER under-rejects (it cannot miss a freeze). The cost is
    OVER-rejection: it also flags benign literal-separated ranges (`\\d{1,3}\\.\\d{1,3}`
    IPv4, `.*foo.*`) whose required separator actually makes them linear.

    WHY THE COUNT AND NOT A MORE PRECISE RULE (the hard-won lesson, rounds 7-10):
    distinguishing a real separator from a fake one needs full regex semantics —
    character-class OVERLAP (`.*X.*X` freezes because `.` matches `X`) and branch
    EMPTINESS (`.*(?:Z|).*` freezes because the branch can be empty). Every heuristic
    that tried to be precise (R8 adjacency, R9 top-level-anchor) re-opened the daemon
    FREEZE in the dangerous direction. The over-rejection is the SAFE direction: it is
    FAIL-OPEN (skip pattern protection this cycle + a stderr warning, content stays
    prunable — never destroyed, never a crash, never a freeze), it only affects the
    no-SIGALRM path (Windows / non-main-thread), and the 512-char no-budget cap is an
    independent backstop. A durable fix for the over-rejection would require running
    the match under a real hard timeout (a killable subprocess), not a shape heuristic."""
    count = 0
    i, n, in_class = 0, len(pattern), False
    prev = ""  # previous structural char, to classify a bare '?'
    while i < n:
        c = pattern[i]
        if c == "\\":
            i += 2
            prev = "x"  # an escaped atom
            continue
        if in_class:
            if c == "]":
                in_class = False
                prev = "]"
            i += 1
            continue
        if c == "[":
            in_class = True
            i += 1
            prev = "["
            continue
        if c in "*+":
            count += 1
            i += 1
            prev = c
            continue
        if c == "?":
            # A '?' is a BRANCH-introducing quantifier (count it) EXCEPT when it is a
            # group-type marker `(?...` or a lazy/possessive modifier on a preceding
            # quantifier (`*?`, `+?`, `??`, `}?`). R11: a flat chain of optional atoms
            # `a?a?...aaa` backtracks EXPONENTIALLY in the number of `?`, so excluding
            # `?` made the count rule UNDER-reject (freeze). Counting it restores the
            # true necessary condition: >=2 branch quantifiers of ANY kind => risky.
            if prev not in ("(", "*", "+", "?", "}"):
                count += 1
            i += 1
            prev = "?"
            continue
        if c == "{":
            j = pattern.find("}", i)
            if j == -1:
                i += 1
                prev = "{"
                continue
            sp = pattern[i + 1:j]
            if "," in sp:
                lo, _, hi = sp.partition(",")
                hi = hi.strip()
                if hi == "" or hi != lo.strip():
                    count += 1
            i = j + 1
            prev = "}"
            continue
        i += 1
        prev = c
    return count


def _has_alternation(pattern: str) -> bool:
    """True if PATTERN contains an alternation `|` outside a character class / escape.

    On the no-budget (no-SIGALRM) path we REFUSE all alternation categorically (R12).
    Alternation is a super-linear backtracking source, and an AMBIGUOUS alternation
    (nested `((a|a))+`, an unquantified chain `(a|a)(a|a)...`, an overlapping
    `(aa|a)...`) cannot be reliably distinguished from a benign DISJOINT one
    (`foo|bar`) without full regex analysis — rounds 8-12 proved every precise
    heuristic (adjacency, top-level-anchor, recursive ambiguity, quantifier counting)
    leaks and reopens the daemon freeze. Refusing ALL `|` closes the entire class
    SOUNDLY. Verified empirically complete: with no `|` and < 2 quantifiers, no pattern
    (incl backreferences) backtracks. The cost is over-rejecting benign alternations —
    fail-OPEN (skip pattern protection this cycle + a stderr warning, content stays
    prunable, no crash/freeze) and ONLY on the no-SIGALRM path (Windows / non-main
    thread); the ~70k POSIX main-thread users use the SIGALRM timer and never consult
    this detector. The zero-over-rejection alternative is a killable-subprocess hard
    timeout (a larger change, out of scope for this PR)."""
    i, n, in_class = 0, len(pattern), False
    while i < n:
        c = pattern[i]
        if c == "\\":
            i += 2
            continue
        if in_class:
            if c == "]":
                in_class = False
            i += 1
            continue
        if c == "[":
            in_class = True
            i += 1
            continue
        if c == "|":
            return True
        i += 1
    return False


def _strip_group_prefix(body: str) -> str:
    """Strip a leading group-type marker from a group body so rule-2 doesn't see the
    marker's `?` as an inner quantifier (ynaamane review, LOW): `(?:abc)+` was
    over-rejected because the body `?:abc` tripped _body_has_inner_quantifier on the
    leading `?`. Handles `(?:` non-capturing, `(?flags:` inline-flags, `(?P<name>`
    named; lookaround / comment groups have no real repeatable body (and a genuinely
    nested quantifier inside them is still caught by the >=2-quantifier count rule)."""
    if not body.startswith("?"):
        return body
    if body.startswith("?:"):
        return body[2:]
    if body.startswith("?P<") or body.startswith("?<"):  # named (?P<n>) — (?<= / (?<! handled below
        if body.startswith(("?<=", "?<!")):
            return ""  # lookbehind: no repeatable body
        gt = body.find(">")
        return body[gt + 1:] if gt != -1 else ""
    colon = body.find(":")
    if colon != -1 and all(c in "aiLmsux" for c in body[1:colon]):  # (?ims: inline flags
        return body[colon + 1:]
    return ""  # lookahead (?= (?!, comment (?#, or unknown — no repeatable body


def _scan_quantified_group_bodies(pattern: str) -> list[str]:
    """Inner body of each group `(...)` immediately followed by an unbounded or
    braced quantifier (`+`, `*`, `{...}`). Honors escapes, char classes, and
    nesting. A group followed by `?` (or nothing) is NOT returned — `(...)?` alone
    cannot drive catastrophic backtracking."""
    bodies: list[str] = []
    stack: list[int] = []
    i, n, in_class = 0, len(pattern), False
    while i < n:
        c = pattern[i]
        if c == "\\":
            i += 2
            continue
        if in_class:
            if c == "]":
                in_class = False
            i += 1
            continue
        if c == "[":
            in_class = True
            i += 1
            continue
        if c == "(":
            stack.append(i)
            i += 1
            continue
        if c == ")":
            if stack:
                start = stack.pop()
                nxt = pattern[i + 1] if i + 1 < n else ""
                if nxt in ("+", "*", "{"):  # NOT `in "+*{"` — "" is a substring of every str
                    bodies.append(_strip_group_prefix(pattern[start + 1:i]))
            i += 1
            continue
        i += 1
    return bodies


def _body_has_inner_quantifier(body: str) -> bool:
    """True if BODY contains a VARIABLE-WIDTH inner quantifier outside a char class /
    escape. A quantified group whose body is itself variable-width — `(a+)+`, `(a?)+`,
    `(.*X){8}`, `(x+){10}`, `(\\d{1,3}){2,}` — is a classic exponential/polynomial
    ReDoS (the inner can match different widths, so the outer quantifier has many
    ways to split the input). A FIXED-count inner `{n}` (e.g. `(\\d{4})+`) is NOT
    variable-width — the partition is unique, so it is linear and must NOT be flagged
    (R5 P3 over-rejection of bounded protect patterns)."""
    i, n, in_class = 0, len(body), False
    while i < n:
        c = body[i]
        if c == "\\":
            i += 2
            continue
        if in_class:
            if c == "]":
                in_class = False
            i += 1
            continue
        if c == "[":
            in_class = True
            i += 1
            continue
        if c in "*+?":
            return True
        if c == "{":
            j = body.find("}", i)
            if j == -1:
                i += 1
                continue
            # variable-width only if the brace spec carries a comma ({n,} / {n,m});
            # a bare {n} is fixed-width and unambiguous.
            if "," in body[i + 1:j]:
                return True
            i = j + 1
            continue
        i += 1
    return False


def _split_top_level_alts(body: str) -> list[str]:
    """Split BODY on top-level `|` only (not inside nested groups / char classes)."""
    alts: list[str] = []
    cur: list[str] = []
    depth, in_class = 0, False
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c == "\\":
            cur.append(body[i:i + 2])
            i += 2
            continue
        if in_class:
            cur.append(c)
            if c == "]":
                in_class = False
            i += 1
            continue
        if c == "[":
            in_class = True
            cur.append(c)
            i += 1
            continue
        if c == "(":
            depth += 1
            cur.append(c)
            i += 1
            continue
        if c == ")":
            depth -= 1
            cur.append(c)
            i += 1
            continue
        if c == "|" and depth == 0:
            alts.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    alts.append("".join(cur))
    return alts


def _ambiguous_alternation(body: str) -> bool:
    """True if BODY is an alternation whose branches can match a common first
    character — `(a|a)+`, `((a)|(a))+` backtrack catastrophically. A DISJOINT
    alternation like `(TODO|FIXME)+` (T vs F) is unambiguous and SAFE, so it is
    NOT flagged (the R4 over-rejection complaint)."""
    if "|" not in body:
        return False
    alts = _split_top_level_alts(body)
    if len(alts) < 2:
        return False

    def _first(alt: str) -> str | None:
        j = 0
        while j < len(alt) and alt[j] == "(":
            j += 1
        if j >= len(alt):
            return None  # empty branch → group is optional+ambiguous
        ch = alt[j]
        return "*WILD*" if ch in r".[\\" else ch  # dot/class/escape overlaps anything

    firsts = [_first(a) for a in alts]
    if any(f is None or f == "*WILD*" for f in firsts):
        return True
    return len(set(firsts)) < len(firsts)  # two branches share a first char


def _pattern_is_redos_risky(pattern: str) -> bool:
    """Heuristic: True if PATTERN can backtrack catastrophically. Used ONLY to fail
    CLOSED where no wall-clock budget exists (Windows / non-main thread), where a
    pure-Python thread cannot interrupt a CPU-bound `re` match.

    Four CATEGORICAL fail-CLOSED rules (never under-reject → never let a freeze
    through), at the cost of safe-direction over-rejection on this niche path. The
    three backtracking sources — repetition interaction, quantifier-over-ambiguity,
    and alternation — are each closed CATEGORICALLY rather than by precise detection
    (rounds 8-12 proved precise heuristics leak); backreferences with < 2 quantifiers
    and no alternation are empirically freeze-free:
      1. >= 2 variable-width quantifiers anywhere (`.*.*`, `a*a*`, `.{1,500}.{1,500}`,
         the optional chain `a?a?...`, `.*X.*X`, ...). Conservatively also flags linear
         literal-separated ranges — accepted, fail-open (see _count_variable_quantifiers).
      2. a quantified group whose body is itself variable-width (`(a?)+`, `(.*X){8}`).
      3. ANY alternation `|` (R12): closes nested `((a|a))+`, unquantified chains
         `(a|a)(a|a)...`, overlapping `(aa|a)...` — and conservatively benign `foo|bar`
         too (fail-open; see _has_alternation for why categorical, not precise).
      4. (subsumed by 3, kept defensively) a quantified group with an ambiguous
         alternation."""
    if _count_variable_quantifiers(pattern) >= 2:
        return True
    if _has_alternation(pattern):
        return True
    for body in _scan_quantified_group_bodies(pattern):
        if _body_has_inner_quantifier(body) or _ambiguous_alternation(body):
            return True
    return False


def _match_time_budget(seconds: float):
    """Best-effort wall-clock budget for regex matching via SIGALRM (POSIX main
    thread only). On Windows or a non-main thread it yields WITHOUT a timeout —
    tag_pattern_matches compensates there by REFUSING redos-shaped patterns up
    front (fail closed), so the daemon can't be frozen by a poisoned pattern.
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
    # Windows / non-main-thread fail-closed: when the budget is enabled but no
    # real SIGALRM timer can be armed, a redos-shaped pattern could freeze the
    # daemon with no interrupt. Refuse such a pattern up front (skip protection
    # this cycle, warn) — the same OUTCOME as the POSIX fail-open, delivered
    # before the match instead of via a timer. Safe-shaped patterns proceed.
    _budget = _protect_match_budget()
    _no_budget = _budget > 0 and not _have_sigalrm()
    if _no_budget:
        risky = [p for p in patterns if _pattern_is_redos_risky(getattr(p, "pattern", str(p)))]
        if risky:
            import sys
            print("  Cozempic: --protect-pattern matching has no time budget on this "
                  "platform (no SIGALRM) and a supplied pattern can backtrack "
                  "catastrophically; skipping pattern protection this cycle. Simplify "
                  "the pattern or set COZEMPIC_PROTECT_MATCH_SECONDS=0 to opt out.",
                  file=sys.stderr)
            return 0
    count = 0
    matchable = 0
    try:
        with _match_time_budget(_budget):
            for _, msg_dict, _ in messages:
                if not isinstance(msg_dict, dict):
                    continue
                texts = [t for t in _iter_msg_texts(msg_dict) if isinstance(t, str)]
                if texts:
                    matchable += 1
                if msg_dict.get(_PATTERN_PROTECTED_KEY):
                    continue
                # DEFINITIVE no-budget bound (independent of the shape detector's
                # completeness): with no real timer, cap each surface to
                # _NO_BUDGET_MATCH_CAP chars so even a catastrophic pattern the
                # detector MISSED backtracks over a few KB (bounded ms), never the
                # full 256KB surface (effectively infinite). On POSIX the SIGALRM
                # timer is the bound, so no cap there.
                if _no_budget:
                    texts = [t[:_NO_BUDGET_MATCH_CAP] for t in texts]
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
        content = inner.get("content", []) if isinstance(inner, dict) else []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "tool_use" and block.get("name") == "Task":
                        inp = block.get("input", {})
                        if isinstance(inp, dict) and inp.get("run_in_background"):
                            sid = hashable_str(block.get("id"))  # unhashable -> "" (R6)
                            if sid:
                                spawns[sid] = inp.get("description", "")
                    if block.get("type") == "tool_result":
                        cid = hashable_str(block.get("tool_use_id"))
                        if cid:
                            completions.add(cid)

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
    """Get the text content of a content block, handling all block types.

    Non-string-safe: a block's text/thinking/content field — or a nested sub-block's
    text — can legally be a non-string in untrusted JSONL. Used by the token
    estimator (doctor / `current` / nudge), which is OUTSIDE the executor's
    per-strategy isolation, so it must never raise."""
    if not isinstance(block, dict):
        return ""
    result = block.get("text", "") or block.get("thinking", "") or block.get("content", "")
    if isinstance(result, list):
        return " ".join(
            sub["text"] for sub in result
            if isinstance(sub, dict) and isinstance(sub.get("text"), str)
        )
    if not isinstance(result, str):
        return ""
    return result
