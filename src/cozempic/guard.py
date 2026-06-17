"""Guard daemon — continuous team checkpointing + emergency prune.

Architecture:
  EVERY interval:  Extract team state → write checkpoint (lightweight, no prune)
  AT threshold:    Prune non-team messages → inject recovery → optionally reload

The checkpoint runs continuously so team state is ALWAYS on disk, regardless
of whether the threshold is ever hit. The threshold prune is the emergency
fallback — not the primary protection mechanism.

Checkpoint triggers:
  1. Every N seconds (guard daemon)
  2. On demand via `cozempic checkpoint` (hook-driven)
  3. At file size threshold (emergency prune)
"""

from __future__ import annotations

import math
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# P0-B: imported at module level so guard_prune_cycle can catch the exception
# without an inner try/except that masks import errors during testing.
# safety.py is a new module (this PR); the import is always available once
# the package is installed from this version onwards.
from .safety import PruneValidationError

# ── HARD-threshold back-off + exit constants ────────────────────────────────
# When ``guard_prune_cycle`` keeps returning saved_bytes == 0 at the HARD
# threshold (because the live conversation is dominated by immutable tool-
# result blocks the soft prune cannot touch), the daemon used to loop at the
# original 30s interval indefinitely — production log showed 265 cycles over
# 5h21m. The current contract:
#
#   K < HARD_LOOP_BACKOFF_START   → sleep ``interval`` (original cadence)
#   K >= HARD_LOOP_BACKOFF_START  → sleep min(interval * 2 ** (K - 2),
#                                              HARD_LOOP_BACKOFF_CAP_SECONDS)
#   K >= HARD_LOOP_EXIT_THRESHOLD → log diagnostic, write final checkpoint,
#                                   sys.exit(0). SessionStart hook will respawn.
#
# Any prune that returns saved_bytes > 0 resets K to 0 (counter never decays
# on its own — only a genuine prune signals "we can still make progress").
# The cap is 5 minutes: longer is operator-hostile (HARD threshold context
# may genuinely need attention), shorter wastes cycles on doomed prunes.
HARD_LOOP_BACKOFF_START = 3
HARD_LOOP_BACKOFF_CAP_SECONDS = 300
HARD_LOOP_EXIT_THRESHOLD = 10
# C2: consecutive per-cycle exceptions after which the daemon stops silently
# spinning (inert-but-alive, watchdog-invisible) and exits for SessionStart to
# respawn — turning a deterministic repeating failure into a visible respawn.
GUARD_CYCLE_ERROR_EXIT = 5
# C2: max consecutive cycles the escalation-exit will defer while agents are
# active before escalating anyway (a stuck guard must not spin inert forever).
GUARD_CYCLE_ERROR_DEFER_MAX = 20


# ── Hard cap: K=10 exit deferral when agents_active (PR #93 item #4) ────────
# When K reaches HARD_LOOP_EXIT_THRESHOLD (=10) AND `agents_active=True`,
# the daemon used to `sys.exit(0)` mid-task, killing the subagents'
# protection AND telling the operator to `/clear` (which destroys
# subagent state). PR #93 defers the exit while agents are running and
# only exits at the HARD cap below — giving subagents a chance to
# finish before context dies.
#
# Default cap K=50 ≈ 4 hours wall time at the backoff cap (300s/cycle),
# well past any normal subagent batch but short enough that a stuck
# session doesn't outlive an operator's workday.
#
# Override via env var COZEMPIC_GUARD_HARD_EXIT_K (sister-module
# precedent: spawn_lock._read_fresh_window_seconds clamps + falls back
# on garbage). Read EXACTLY ONCE at module import time — requires a
# daemon restart to take effect (same convention as
# COZEMPIC_PIDFILE_FRESH_SECONDS).
def _read_hard_exit_threshold() -> int:
    """Read COZEMPIC_GUARD_HARD_EXIT_K env var. Clamps to (10, 1000].

    Read at module import time only — restart the daemon to apply
    a new value. Invalid values (non-numeric, <=K=10, > 1000) fall
    back to the default 50.
    """
    raw = os.environ.get("COZEMPIC_GUARD_HARD_EXIT_K")
    if raw is None:
        return 50
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return 50
    # Must be strictly > HARD_LOOP_EXIT_THRESHOLD (otherwise no defer
    # window). Cap at 1000 to prevent absurd values (~3.5 days at 5min
    # cap) from silently disabling the circuit breaker.
    if val <= HARD_LOOP_EXIT_THRESHOLD or val > 1000:
        return 50
    return val


HARD_LOOP_HARD_EXIT_THRESHOLD = _read_hard_exit_threshold()

# ── Watcher poll constants (GAP-B) ───────────────────────────────────────────
# After osascript fires, the watcher polls for a new claude process for up to
# RELOAD_WATCHER_POLL_TIMEOUT_SECONDS. 30s matches acquire_with_wait default.
# On timeout, writes a structured status file read by the next SessionStart hook.
RELOAD_WATCHER_POLL_TIMEOUT_SECONDS = 30
RELOAD_WATCHER_POLL_INTERVAL_SECONDS = 1

# ── Futile-reload threshold (GAP-D) ─────────────────────────────────────────
# Minimum fraction of session bytes that prune must save to justify a reload.
# If saved_bytes / original_bytes < _MIN_PRUNE_RATIO, the resumed Claude would
# re-trigger HARD immediately (context dominated by immutable tool-result blocks).
# Override via env var COZEMPIC_MIN_PRUNE_RATIO. Read at module import time only
# — restart the daemon to apply a new value.
_DEFAULT_MIN_PRUNE_RATIO = 0.10


def _read_min_prune_ratio() -> float:
    """Read COZEMPIC_MIN_PRUNE_RATIO env var. Clamps to (0.0, 1.0) exclusive.

    Read at module import time only — restart the daemon to apply a new
    value. Invalid values (non-numeric, NaN, inf, <= 0.0, >= 1.0) fall
    back to the default 0.10.
    """
    raw = os.environ.get("COZEMPIC_MIN_PRUNE_RATIO")
    if raw is None:
        return _DEFAULT_MIN_PRUNE_RATIO
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MIN_PRUNE_RATIO
    if not math.isfinite(val) or val <= 0.0 or val >= 1.0:
        return _DEFAULT_MIN_PRUNE_RATIO
    return val


_MIN_PRUNE_RATIO = _read_min_prune_ratio()


def _hard_prune_counts_as_futile(result: dict) -> bool:
    """Whether a HARD-tier prune cycle counts toward the futile-loop K-exit counter.

    Counts when the cycle freed nothing usable: a deferred-conflict, an explicit
    futile-reload-skip, or a non-live prune that saved <= 0.

    The subtle case (the f641174c 202-cycle loop, 2026-06-10): a READ-ONLY live
    skip is normally BENIGN — the session is busy and a real prune will help once
    Claude pauses — so it must NOT count (a long agent run can't be allowed to trip
    the K-exit). BUT when the COMPUTED prune barely helps (projected token reduction
    below _MIN_PRUNE_RATIO), the session is fundamentally unprunable and the guard is
    powerless against it busy or not; that DOES count, so the daemon K-exits instead
    of SIGKILL-reloading a session it can never get below threshold."""
    if result.get("prune_deferred_conflict") or result.get("futile_reload_skipped"):
        return True
    if result.get("live_write_skipped"):
        # Primary signal: the COMPUTED prune's BYTE reduction (always present on the
        # read-only return, matching the GAP-D byte-ratio definition of "futile").
        orig = result.get("original_bytes", 0)
        if isinstance(orig, (int, float)) and orig > 0:
            # Coerce defensively: a malformed/None would_free_mb must never raise
            # into the daemon's main loop (which has no broad try/except).
            wf = result.get("would_free_mb", 0.0)
            would_free = (wf if isinstance(wf, (int, float)) else 0.0) * 1024 * 1024
            return (would_free / orig) < _MIN_PRUNE_RATIO
        # Fallback: token projection (only present when project=True).
        proj = result.get("projected_final_tokens")
        pre = result.get("final_tokens", 0)
        return (proj is not None and isinstance(pre, (int, float)) and pre > 0
                and (pre - proj) / pre < _MIN_PRUNE_RATIO)
    return result.get("saved_mb", 0) <= 0


from ._validation import ConfigError
from .executor import run_prescription
from .helpers import hashable_str, is_ssh_session, shell_quote, tag_pattern_matches, strip_pattern_tags
from .registry import PRESCRIPTIONS
import cozempic.strategies  # noqa: F401 — register strategies so guard_prune_cycle can actually prune (#15)
from .session import (
    PruneConflictError,
    PruneLockError,
    _PruneLock,
    cleanup_old_backups,
    find_claude_pid,
    find_current_session,
    find_sessions,
    load_messages,
    load_messages_and_snapshot,
    load_messages_incremental,
    save_messages,
    snapshot_session,
)
from .team import (
    TeamState, extract_team_state, inject_team_recovery, write_team_checkpoint,
    _AGENT_DONE_TRAILER_RE,
)
from .tokens import default_token_thresholds, quick_token_estimate
# Eager import: ensures the daemon's upgrade check uses code from the daemon's
# OWN install state (frozen at import time), not whatever happens to be on
# disk when this function runs post-upgrade. Prevents old-daemon/new-updater
# version skew.
from .updater import maybe_auto_update, ping_install_if_new
# NEW-1 sentinel: imported at module level so start_guard_daemon can call
# _reload_sentinel_active without a nested import, and _terminate_and_resume
# can call write_reload_sentinel from all code paths (tmux, screen, plain terminal).
from .reload_lock import write_reload_sentinel, unlink_reload_sentinel, _reload_sentinel_active  # noqa: E402


def _normalize_session_id(session_id: str) -> str:
    """Extract UUID from a session_id that might be a full path."""
    if session_id.endswith(".jsonl"):
        return Path(session_id).stem
    return session_id


def _resolve_session_by_id(session_id: str, max_retries: int = 10, retry_delay: float = 1.5) -> dict | None:
    """Find a session by explicit ID, UUID prefix, or path.

    Handles full JSONL paths (from SessionStart hook), UUIDs, and prefixes.
    Retries up to max_retries times (15s total) to handle the race condition
    where the hook fires before Claude Code creates the JSONL file (#73).
    """
    p = Path(session_id)

    # Fast path: full path exists on disk. Guard p.exists()/p.stat() with try/except —
    # a pathological session_id (e.g. an over-long path → ENAMETOOLONG) raises OSError,
    # which, now that the patient wait (#121) re-checks in a loop, would crash the
    # daemon repeatedly. Treat any such error as "not a usable path yet".
    def _from_path():
        try:
            if p.suffix == ".jsonl" and p.exists():
                return {"path": p, "session_id": p.stem,
                        "size": p.stat().st_size, "project": p.parent.name}
        except OSError:
            pass
        return None

    hit = _from_path()
    if hit:
        return hit

    # Extract UUID from path-like input (file may not exist yet)
    search_id = _normalize_session_id(session_id)

    for attempt in range(max_retries):
        hit = _from_path()  # re-check the path on each retry (file may appear)
        if hit:
            return hit
        sessions = find_sessions()
        # Prefer an EXACT id match over a prefix match. The guard is DESTRUCTIVE, so a
        # prefix collision (two sessions sharing a leading segment, e.g. across
        # projects) must never attach it to the WRONG session — exact wins (QA P2).
        for sess in sessions:
            if sess["session_id"] == search_id:
                return sess
        for sess in sessions:
            if sess["session_id"].startswith(search_id):
                return sess
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    return None


# Seconds to patiently wait for an EXPLICIT session's JSONL to appear. Claude Code
# writes the session JSONL LAZILY, on the FIRST USER TURN — user-paced, routinely
# far beyond _resolve_session_by_id's 15s budget (#121 measured 109s). Generous
# default so the guard survives a slow first message; bounded so a truly-abandoned
# session (user opened Claude, never typed, walked away) sleeps AT MOST the budget
# then exits cleanly — a known claude_pid lets it exit sooner, but the SessionStart
# hook doesn't pass one today, so the budget is the operative bound.
_DEFAULT_SESSION_WAIT_SECONDS = 900.0


def _session_wait_budget() -> float:
    """Total seconds to wait for an explicit session's lazily-created JSONL (#121).
    `COZEMPIC_SESSION_WAIT_SECONDS` overrides; finite, clamped to [0, 3600]. 0
    disables the patient wait (the daemon reverts to the bare 15s resolve)."""
    try:
        v = float(os.environ.get("COZEMPIC_SESSION_WAIT_SECONDS", _DEFAULT_SESSION_WAIT_SECONDS))
    except (TypeError, ValueError):
        return _DEFAULT_SESSION_WAIT_SECONDS
    # Reject NaN/inf (every comparison would be False → the loop never bounds).
    if not math.isfinite(v) or v < 0:
        return _DEFAULT_SESSION_WAIT_SECONDS
    return min(v, 3600.0)


def _resolve_session_patiently(session_id: str, claude_pid: int | None = None) -> dict | None:
    """Resolve an EXPLICIT (harness-vouched) session id, waiting for Claude Code to
    create its JSONL lazily on the first user turn (#121, regression of #73).

    After the initial 15s `_resolve_session_by_id` attempt, keep polling with
    backoff until (a) the file appears, (b) the session's Claude process exits when
    a `claude_pid` is known — nothing left to guard, or (c) the
    `COZEMPIC_SESSION_WAIT_SECONDS` budget elapses. The daemon keeps its
    already-acquired spawn claim while waiting, so a slow first message no longer
    kills the guard for the whole session, and `doctor` still sees it alive.

    ONLY for an explicit id the harness vouched for — auto-detect (no `--session`)
    keeps the bare 15s resolve, since there is no authoritative id to wait on."""
    sess = _resolve_session_by_id(session_id)
    if sess:
        return sess
    budget = _session_wait_budget()
    if budget <= 0:
        return None
    print(
        f"  Session JSONL not written yet — Claude Code creates it on the first user "
        f"turn; waiting up to {int(budget)}s (COZEMPIC_SESSION_WAIT_SECONDS to tune).",
        file=sys.stderr,
    )
    waited = 0.0
    delay = 3.0
    while waited < budget:
        if claude_pid is not None and not _pid_is_alive(claude_pid):
            return None  # the session's Claude is gone — don't guard a dead session
        time.sleep(delay)
        waited += delay
        sess = _resolve_session_by_id(session_id, max_retries=1, retry_delay=0)
        if sess:
            return sess
        delay = min(delay * 1.5, 30.0)  # back off, cap at 30s
    return None


# ─── Lightweight checkpoint (no prune) ───────────────────────────────────────

def checkpoint_team(
    cwd: str | None = None,
    session_path: Path | None = None,
    quiet: bool = False,
) -> TeamState | None:
    """Extract and save team state from the current session. No pruning.

    This is fast and safe — it only reads the JSONL and writes a checkpoint.
    Designed to be called from hooks, guard daemon, or CLI.

    Returns the extracted TeamState, or None if no session found.
    """
    if session_path is None:
        # strict=True: refuse Strategy 4 (global most-recent fallback).
        # Writing a checkpoint from the wrong project's session is worse than
        # writing no checkpoint at all — the latter leaves PostCompact with
        # nothing to inject, while the former injects another project's state.
        sess = find_current_session(cwd, strict=True)
        if not sess:
            if not quiet:
                print("  No active session found.", file=sys.stderr)
            return None
        session_path = sess["path"]

    # Scan-only hot path — use incremental loader to avoid unbounded RSS growth
    # from repeated full-file reads in the guard's 30s main loop.
    messages = load_messages_incremental(session_path)
    state = extract_team_state(messages)

    if state.is_empty():
        if not quiet:
            print("  No team state detected.")
        return state

    project_dir = session_path.parent
    cp_path = write_team_checkpoint(state, project_dir)

    if not quiet:
        agents = len(state.subagents)
        teammates = len(state.teammates)
        tasks = len(state.tasks)
        parts = []
        if agents:
            parts.append(f"{agents} subagents")
        if teammates:
            parts.append(f"{teammates} teammates")
        if tasks:
            parts.append(f"{tasks} tasks")
        summary = ", ".join(parts) if parts else "empty"
        print(f"  Checkpoint: {summary} → {cp_path.name}")

    return state


# ─── Team-aware pruning ──────────────────────────────────────────────────────

def prune_with_team_protect(
    messages: list,
    rx_name: str = "standard",
    config: dict | None = None,
) -> tuple[list, list, TeamState]:
    """Run a prescription but protect team-related messages from pruning.

    Returns (pruned_messages, strategy_results, team_state).

    Strategy:
    1. Extract team state
    2. Tag team messages with __cozempic_team_protected__ (is_protected() skips them)
    3. Run prescription on the FULL list (no splitting, no memory doubling)
    4. Remove tags, inject team recovery messages
    """
    from .team import _is_team_message

    config = config or {}
    strategy_names = PRESCRIPTIONS.get(rx_name, PRESCRIPTIONS["standard"])

    # 1. Extract team state
    team_state = extract_team_state(messages)

    if team_state.is_empty():
        # No team — standard pruning
        new_messages, results = run_prescription(messages, strategy_names, config)
        return new_messages, results, team_state

    # 2. Build pending_task_ids
    from .team import TEAM_TOOL_NAMES
    pending_task_ids: set[str] = set()
    for _, msg_dict, _ in messages:
        inner = msg_dict.get("message", {})
        if not isinstance(inner, dict):
            continue
        for block in (inner.get("content", []) if isinstance(inner.get("content"), list) else []):
            # isinstance/hashable_str guards: this is the prune_with_team_protect SIBLING
            # of the team.py extraction loop — an unhashable name/id (poisoned JSONL) or a
            # non-dict block crashed the prune EVERY cycle -> guard respawn storm (R6).
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and hashable_str(block.get("name")) in TEAM_TOOL_NAMES:
                tool_use_id = hashable_str(block.get("id"))
                if tool_use_id:
                    pending_task_ids.add(tool_use_id)

    # 3+4. Tag team messages as protected, then prune. The apply loop is INSIDE
    # the try so the finally strip covers it unconditionally — including a signal
    # delivered between the apply and run_prescription (hardening, no disk leak
    # either way since the list is discarded on exit). Tags are still applied
    # before run_prescription so strategies see them via is_protected().
    try:
        for _, msg_dict, _ in messages:
            if _is_team_message(msg_dict, pending_task_ids):
                msg_dict["__cozempic_team_protected__"] = True

        pruned_messages, results = run_prescription(messages, strategy_names, config)
    finally:
        # 5. Remove tags from the source list (messages) — covers the abort path
        # where pruned_messages may be partially built or identical to messages.
        for _, msg_dict, _ in messages:
            msg_dict.pop("__cozempic_team_protected__", None)

    # 5b. Also strip from pruned_messages (they may be a different list).
    for _, msg_dict, _ in pruned_messages:
        msg_dict.pop("__cozempic_team_protected__", None)

    # 6. Inject team recovery messages at the end
    pruned_messages = inject_team_recovery(pruned_messages, team_state)

    return pruned_messages, results, team_state


# ─── Guard daemon ─────────────────────────────────────────────────────────────


def _validate_finite_thresholds(
    threshold_mb=None,
    soft_threshold_mb=None,
    interval=None,
    threshold_tokens=None,
    soft_threshold_tokens=None,
) -> None:
    """Reject NaN, inf, and huge ints in numeric threshold parameters.

    Belt-and-braces guard for direct Python callers that bypass argparse.
    Mirrors coerce_positive_float's finite contract (P0-A + P0-F). Raises
    ConfigError with 'must be a finite number' when any numeric param is
    nan, inf, or a huge int (e.g. 10**400) that would overflow on conversion.

    Bools are excluded (they are int subclasses in Python; True/False are
    validated separately by type checks downstream). None is also skipped
    (optional params default to None when not supplied by the caller).
    """
    from ._validation import ConfigError

    for _name, _v in (
        ("threshold_mb", threshold_mb),
        ("soft_threshold_mb", soft_threshold_mb),
        ("interval", interval),
        ("threshold_tokens", threshold_tokens),
        ("soft_threshold_tokens", soft_threshold_tokens),
    ):
        if _v is None or isinstance(_v, bool) or not isinstance(_v, (int, float)):
            continue
        try:
            _finite = math.isfinite(_v)
        except OverflowError:
            _finite = False  # int too large to convert to float (e.g. 10**400)
        if not _finite:
            raise ConfigError(f"{_name} must be a finite number, got {_v!r}")


def start_guard(
    cwd: str | None = None,
    threshold_mb: float = 50.0,
    soft_threshold_mb: float | None = None,
    rx_name: str = "standard",
    interval: int = 30,
    auto_reload: bool = True,
    config: dict | None = None,
    reactive: bool = True,
    threshold_tokens: int | None = None,
    soft_threshold_tokens: int | None = None,
    session_id: str | None = None,
    claude_pid: int | None = None,
    protect_patterns: list | None = None,
) -> None:
    """Start the guard daemon with tiered pruning.

    Three-phase protection:
      1. CHECKPOINT every interval — extract team state, write to disk
      2. SOFT at soft threshold — read-only checkpoint (no live-file write, #106)
      3. HARD PRUNE at hard threshold — terminate-first prune + resume (team-protect)

    Thresholds can be bytes-based, token-based, or both. When both are set,
    whichever is hit first triggers the action.

    Default soft threshold is 60% of hard threshold if not specified.

    Args:
        cwd: Working directory for session detection.
        threshold_mb: Hard threshold in MB — emergency prune + optional reload.
        soft_threshold_mb: Soft threshold in MB — gentle prune, no reload.
            Defaults to 60% of threshold_mb.
        rx_name: Prescription to apply at hard threshold.
        interval: Check interval in seconds.
        auto_reload: If True, kill Claude and auto-resume after hard prune.
        config: Extra config for pruning strategies.
        threshold_tokens: Hard threshold in tokens (optional, checked alongside bytes).
        soft_threshold_tokens: Soft threshold in tokens (optional, checked alongside bytes).
        session_id: Explicit session ID to monitor (bypasses auto-detection).
    """
    # Validate ordering invariants FIRST — a reload storm caused by a
    # swapped soft/hard threshold is much worse than a clean upfront error.
    # Argparse already rejects non-positive values, but direct Python callers
    # (guard.start_guard(...)) bypass argparse, so belt-and-braces check.
    _validate_finite_thresholds(
        threshold_mb=threshold_mb,
        soft_threshold_mb=soft_threshold_mb,
        interval=interval,
        threshold_tokens=threshold_tokens,
        soft_threshold_tokens=soft_threshold_tokens,
    )
    if threshold_mb <= 0:
        raise ConfigError(f"threshold_mb must be positive, got {threshold_mb}")
    if soft_threshold_mb is not None and soft_threshold_mb <= 0:
        raise ConfigError(f"soft_threshold_mb must be positive, got {soft_threshold_mb}")
    if (
        soft_threshold_mb is not None
        and soft_threshold_mb >= threshold_mb
    ):
        raise ConfigError(
            f"soft_threshold_mb={soft_threshold_mb} must be strictly less than "
            f"threshold_mb={threshold_mb}"
        )
    if interval <= 0:
        raise ConfigError(f"interval must be positive, got {interval}")
    if threshold_tokens is not None and threshold_tokens <= 0:
        raise ConfigError(f"threshold_tokens must be positive, got {threshold_tokens}")
    if soft_threshold_tokens is not None and soft_threshold_tokens <= 0:
        raise ConfigError(f"soft_threshold_tokens must be positive, got {soft_threshold_tokens}")
    if (
        threshold_tokens is not None
        and soft_threshold_tokens is not None
        and soft_threshold_tokens >= threshold_tokens
    ):
        raise ConfigError(
            f"soft_threshold_tokens={soft_threshold_tokens} must be strictly less than "
            f"threshold_tokens={threshold_tokens}"
        )

    hard_threshold_bytes = int(threshold_mb * 1024 * 1024)

    if soft_threshold_mb is None:
        soft_threshold_mb = round(threshold_mb * 0.6, 1)
    soft_threshold_bytes = int(soft_threshold_mb * 1024 * 1024)

    # Find the session — explicit ID or auto-detect
    # strict=True: guard is destructive, refuse to fall back to "most recently modified"
    if session_id:
        # Patient wait: Claude Code writes the session JSONL lazily on the first
        # user turn (user-paced), far beyond the bare 15s resolve budget — giving up
        # there killed the guard for the whole session with no respawn (#121).
        sess = _resolve_session_patiently(session_id, claude_pid=claude_pid)
    else:
        sess = find_current_session(cwd, strict=True)
    if not sess:
        # Release our claim — CAS unlink (only if the pidfile still holds OUR pid),
        # mirroring every other start_guard exit path. A bare unlink could clobber a
        # peer daemon's just-written claim (QA P3).
        if session_id:
            _safe_unlink_session_pidfile(session_id)
        print("  ERROR: Could not detect current session.", file=sys.stderr)
        if session_id:
            # Make the give-up visible in the guard log instead of a silent death.
            print("  (waited for the session JSONL but it never appeared within the "
                  "budget — raise COZEMPIC_SESSION_WAIT_SECONDS to wait longer)",
                  file=sys.stderr)
        else:
            print("  Tip: Use --session <session_id> for explicit targeting.", file=sys.stderr)
        sys.exit(1)

    session_path = sess["path"]

    # Detect context window from session data (used for display + overflow scaling)
    from .tokens import detect_context_window, default_token_thresholds_4tier, DEFAULT_HARD2_TOKEN_PCT
    messages_for_model = load_messages(session_path)
    context_window = detect_context_window(messages_for_model)

    # Default to 4-tier token thresholds when none specified
    if threshold_tokens is None:
        soft_threshold_tokens, threshold_tokens, hard2_threshold_tokens = default_token_thresholds_4tier(context_window)
    else:
        hard2_threshold_tokens = int(context_window * DEFAULT_HARD2_TOKEN_PCT)
        if soft_threshold_tokens is None:
            soft_threshold_tokens = int(threshold_tokens * 0.45)

    # Persist cwd + context_window to the sidecar so reload and guard resume
    # can resolve the project directory without relying on slug reversal. Also
    # record the RESOLVED reload-tier fractions so the Stop-hook nudge fires at
    # the points this guard actually reloads (tracks a raised --threshold).
    from .session import record_session
    _nudge_tiers = None
    if context_window:
        _nudge_tiers = [
            round(t / context_window, 4) for t in
            (soft_threshold_tokens, threshold_tokens, hard2_threshold_tokens)
            if t
        ] or None
    record_session(sess["session_id"], cwd or os.getcwd(), context_window, nudge_tiers=_nudge_tiers)

    # Clean up stale reload watchers from previous versions
    _cleanup_stale_watchers()

    # Auto-update check — force=True so it works even when guard runs via hook (no TTY)
    ping_install_if_new()
    maybe_auto_update(force=True)

    # Format context window for display
    if context_window >= 1_000_000:
        ctx_str = f"{context_window / 1_000_000:.1f}M"
    else:
        ctx_str = f"{context_window / 1_000:.0f}K"

    # Compute threshold %s for display
    soft_pct = int(soft_threshold_tokens / context_window * 100) if soft_threshold_tokens and context_window else 25
    hard1_pct = int(threshold_tokens / context_window * 100) if threshold_tokens and context_window else 55
    hard2_pct = int(hard2_threshold_tokens / context_window * 100) if hard2_threshold_tokens and context_window else 80

    print(
        f"\n  4-tier guard protecting context ({ctx_str} window):\n"
        f"    Soft  ({soft_pct}%): read-only checkpoint, no live-file write (#106)\n"
        f"    Hard1 ({hard1_pct}%): {rx_name} prune + reload (terminate-first)\n"
        f"    Hard2 ({hard2_pct}%): aggressive prune + reload (emergency)\n"
        f"    User  (90%): manual aggressive (cozempic treat -rx aggressive --execute)\n"
    )

    # Reactive overflow recovery via file watcher
    overflow_watcher = None
    if reactive:
        import threading
        from .overflow import CircuitBreaker, OverflowRecovery
        from .watcher import JsonlWatcher

        # Scale danger thresholds based on context window size
        danger_mb = round(threshold_mb * 1.8, 1)
        danger_tokens = int(context_window * 0.90) if context_window else None

        breaker = CircuitBreaker(session_id=sess["session_id"])
        recovery = OverflowRecovery(
            session_path, sess["session_id"], cwd or os.getcwd(), breaker,
            danger_threshold_mb=danger_mb,
            danger_threshold_tokens=danger_tokens,
            claude_pid=claude_pid,
        )
        overflow_watcher = JsonlWatcher(
            str(session_path), on_growth=recovery.on_file_growth,
        )
        watcher_thread = threading.Thread(
            target=overflow_watcher.start, daemon=True, name="cozempic-watcher",
        )
        watcher_thread.start()

    # Graceful shutdown on SIGTERM
    def _graceful_shutdown(signum, frame):
        print(f"\n  [{_now()}] Signal {signum} received — final checkpoint...")
        checkpoint_team(session_path=session_path, quiet=False)
        if overflow_watcher:
            overflow_watcher.stop()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    # Resolve Claude before daemonization or other reparenting can obscure it.
    if claude_pid is None:
        claude_pid = find_claude_pid()
    # Record PID + start_time NOW — earliest point where both claude_pid and
    # session_id are known and Claude's identity is confirmed by find_claude_pid.
    if claude_pid and session_id:
        _record_claude_identity(session_id, claude_pid)
    claude_alive = True

    prune_count = 0
    soft_prune_count = 0
    checkpoint_count = 0
    cycle_count = 0
    last_team_hash = ""
    consecutive_empty_hard_prunes = 0
    # PR #93 item #4: one-shot flag so the "deferring K=10 exit" log
    # line only emits once per defer-window, not every cycle.
    deferred_exit_announced = False
    # GAP-D: one-shot flag so the futile-skip diagnostic emits once per
    # defer-window (mirrors deferred_exit_announced pattern).
    _futile_skip_announced = False

    def _account_hard_prune(result, agents_active, state):
        # Post-prune circuit-breaker accounting shared by the HARD1 (55%) and
        # HARD2 (80%) tiers. Increments the empty-prune counter, emits the
        # GAP-D futile diagnostic, decides the K-exit-vs-defer, and applies the
        # exponential back-off sleep. Extracted verbatim from the HARD1 block so
        # HARD2 gets the same breaker (previously it had none → infinite
        # kill→no-write→resume loop on a sustained deferred-conflict).
        nonlocal consecutive_empty_hard_prunes, deferred_exit_announced, _futile_skip_announced

        # A benign read-only deferral (a LIVE session that is busy but COULD be
        # pruned once Claude pauses) leaves the breaker untouched — a long agent run
        # must not trip the K-exit. BUT a live session that is ALSO unprunable (the
        # COMPUTED prune barely helps) must count toward K-exit, or the guard
        # reload-loops forever on a session it is powerless against — busy or not
        # (f641174c: 202 cycles freeing 0 tokens, 2026-06-10). _hard_prune_counts_as
        # _futile() draws that line. HARD2 (80%) still force-reloads if needed.
        if _hard_prune_counts_as_futile(result):
            consecutive_empty_hard_prunes += 1

            # GAP-D: emit one-shot diagnostic when reload was skipped
            # as futile (prune saved too few bytes to justify a reload
            # that would immediately re-trigger HARD).
            if result.get("futile_reload_skipped") and not _futile_skip_announced:
                would_free_mb = result.get("would_free_mb", result.get("saved_mb", 0))
                orig_bytes = result.get("original_bytes", 0)
                saved_pct = (would_free_mb * 1024 * 1024 / orig_bytes * 100
                             if orig_bytes > 0 else 0)
                checkpoint_ref = (
                    f" Checkpoint: {result['checkpoint_path']}"
                    if result.get("checkpoint_path") else ""
                )
                print(
                    f"  [{_now()}] Hard prune would free only {would_free_mb:.3f}MB "
                    f"(~{saved_pct:.0f}%) — below {int(_MIN_PRUNE_RATIO * 100)}% "
                    f"threshold. Reload skipped (live file left intact): resumed Claude would re-trigger "
                    f"HARD immediately. Likely cause: subagent transcripts or large "
                    f"tool-results dominate context. Recommend: /clear (loses subagent "
                    f"state) or fresh session with restored team "
                    f"checkpoint.{checkpoint_ref}",
                    flush=True,
                )
                _futile_skip_announced = True

            # Exit path: the daemon is powerless against this context
            # (live tool-result blocks dominate; HARD prune cannot free
            # bytes; reload+0-byte = the cascade that crashed sessions
            # in production). Exit gracefully and let the SessionStart
            # hook respawn on next activity. Do NOT change reload-trigger
            # gating in guard_prune_cycle — that's not the right escape.
            #
            # PR #93 item #4: defer the exit when `agents_active=True`.
            # Killing the daemon mid-task destroys subagent protection
            # AND the diagnostic recommends `/clear` (which also
            # destroys subagent state). Hard cap at
            # HARD_LOOP_HARD_EXIT_THRESHOLD (default 50, override via
            # COZEMPIC_GUARD_HARD_EXIT_K) ensures eventual exit so a
            # stuck `extract_team_state` (BUG-G15 family) can't wedge
            # the daemon forever.
            if consecutive_empty_hard_prunes >= HARD_LOOP_EXIT_THRESHOLD:
                if (
                    agents_active
                    and consecutive_empty_hard_prunes < HARD_LOOP_HARD_EXIT_THRESHOLD
                ):
                    # Defer: stay alive, keep cycling at backoff cap.
                    if not deferred_exit_announced:
                        running_count = sum(
                            1 for s in state.subagents
                            if s.status in ("running", "unknown")
                        )
                        worst_case_min = (
                            HARD_LOOP_HARD_EXIT_THRESHOLD
                            * HARD_LOOP_BACKOFF_CAP_SECONDS
                            // 60
                        )
                        print(
                            f"  [{_now()}] K={consecutive_empty_hard_prunes} "
                            f"reached normal exit threshold "
                            f"({HARD_LOOP_EXIT_THRESHOLD}) but "
                            f"{running_count} subagent(s) still active. "
                            f"Deferring daemon exit until agents quiesce "
                            f"or K reaches hard cap "
                            f"({HARD_LOOP_HARD_EXIT_THRESHOLD}, "
                            f"~{worst_case_min} min worst case).",
                            flush=True,
                        )
                        deferred_exit_announced = True
                    # Fall through to the back-off sleep below.
                    # We do NOT sys.exit while agents are working.
                else:
                    # Either no agents (original K=10 exit) OR hard
                    # cap reached even with agents (circuit breaker).
                    try:
                        checkpoint_team(session_path=session_path, quiet=True)
                    except Exception:
                        # Checkpoint failure must not prevent exit —
                        # final checkpoint is best-effort here; the
                        # SOFT loop above has been writing checkpoints
                        # every cycle for the entire run, so on-disk
                        # state is already current.
                        pass
                    if (
                        agents_active
                        and consecutive_empty_hard_prunes >= HARD_LOOP_HARD_EXIT_THRESHOLD
                    ):
                        # Hard cap fired with agents still active —
                        # different diagnostic. Do NOT tell the
                        # operator to `/clear` (that destroys
                        # subagent state too).
                        print(
                            f"  [{_now()}] Guard hard-cap exit "
                            f"(K={consecutive_empty_hard_prunes} >= "
                            f"{HARD_LOOP_HARD_EXIT_THRESHOLD}). "
                            f"Subagents are still active; their state "
                            f"may be lost on the next compaction. "
                            f"Consider letting current subagents "
                            f"finish then starting a fresh session.",
                            flush=True,
                        )
                    else:
                        # Original K=10 exit (no agents — operator
                        # can safely `/clear`).
                        print(
                            f"  [{_now()}] Guard powerless against live-context "
                            f"dominance ({HARD_LOOP_EXIT_THRESHOLD} consecutive "
                            f"0-byte HARD prunes). Exiting — NO further guard "
                            f"protection in this session. SessionStart fires only "
                            f"on startup/resume/clear, NOT on tool calls or "
                            f"message turns, so the daemon will NOT auto-respawn "
                            f"while the session continues. To re-enable cozempic: "
                            f"type /clear or restart the session. Recommended: "
                            f"split work across fresh sessions to avoid >55% "
                            f"context dominance by immutable tool-result blocks.",
                            flush=True,
                        )
                    # _safe_unlink_session_pidfile is called via the
                    # finally block (PR #93 commit 2) — covers this
                    # sys.exit path automatically.
                    sys.exit(0)

            # Back-off path: replace the original fixed-cadence sleep at
            # the bottom of the loop with an exponentially growing one.
            # The loop's primary ``time.sleep(interval)`` at the top of
            # the next iteration is the normal cadence — we ADD an extra
            # back-off sleep here so the next prune is genuinely delayed.
            backoff = _hard_loop_backoff_sleep(
                consecutive_empty_hard_prunes, interval
            )
            # Only emit a back-off sleep beyond the normal interval to
            # avoid double-sleeping at K=1 / K=2 where backoff == interval.
            if backoff > interval:
                if consecutive_empty_hard_prunes == HARD_LOOP_BACKOFF_START:
                    print(
                        f"  [{_now()}] Hard prune freed 0 bytes "
                        f"{HARD_LOOP_BACKOFF_START}x — entering exponential "
                        f"back-off (next sleep: {backoff}s, cap "
                        f"{HARD_LOOP_BACKOFF_CAP_SECONDS}s, exit after "
                        f"{HARD_LOOP_EXIT_THRESHOLD} cycles)."
                    )
                time.sleep(backoff)
        else:
            consecutive_empty_hard_prunes = 0
            # Reset the defer announcement so a fresh K-cycle that
            # reaches K=10-with-agents will emit the notice again
            # (PR #93 item #4 — operator-friendly).
            deferred_exit_announced = False
            # Reset futile-skip announcement so a fresh K-cycle
            # emits the diagnostic again (GAP-D — mirrors above).
            _futile_skip_announced = False

    # ── E/F/H pre-loop state (1.8.22 interactive guard) ───────────────────
    # F adjusts a SEPARATE poll_interval (top-of-loop idle sleep) and leaves the
    # configured `interval` immutable — the HARD-loop circuit-breaker back-off
    # (_hard_loop_backoff_sleep) uses `interval` as its base, so inflating it here
    # would suppress that back-off's escalation. Keep the two cadences decoupled.
    poll_interval = interval          # F: current idle-adjusted top-of-loop sleep
    prev_size = -1                    # last cycle's transcript size (idle detection)
    idle_cycles = 0                   # F: consecutive stable-size cycles
    noop_cycles = 0                   # G: cycles where a fire was skipped as a no-op
    consecutive_cycle_errors = 0      # C2: deterministic per-cycle error -> escalate, not silent-inert
    last_agents_active = False        # C2: don't escalate-exit while a subagent team is live
    logged_decode_skip = False        # C1/C2: log a non-UTF-8 session once, don't respawn-storm
    deferred_error_cycles = 0         # C2: bound the agents-active escalation deferral
    interactive_mode = _detect_interactive(claude_pid)   # H
    force_pct = _force_reload_pct()                       # E
    force_threshold_tokens = (
        int(context_window * force_pct) if (context_window and force_pct) else None
    )
    if interactive_mode:
        _fp = f", force at {int(force_pct * 100)}%" if force_threshold_tokens else ""
        print(f"  Interactive session: hard reloads wait for an idle breakpoint "
              f"(never mid-turn{_fp}).")

    try:
        while True:
            time.sleep(poll_interval)
            try:
                cycle_count += 1

                # Periodic backup cleanup every 10 cycles (~5min)
                if cycle_count % 10 == 0:
                    cleanup_old_backups(session_path, keep=3)

                # Re-check file exists
                if not session_path.exists():
                    print("  WARNING: Session file disappeared. Stopping guard.")
                    break

                # Watchdog: detect Claude exit (workaround for Stop hook not firing)
                if claude_pid and claude_alive:
                    try:
                        os.kill(claude_pid, 0)
                    except (ProcessLookupError, PermissionError):
                        claude_alive = False
                    else:
                        # Liveness confirmed — also verify PID identity to guard against
                        # PID reuse (daemon started hours ago; original Claude exited and
                        # kernel recycled its PID to an unrelated process).
                        try:
                            if not _pid_identity_match(claude_pid, session_id) \
                                    or not _is_claude_process(claude_pid, session_path=session_path):
                                claude_alive = False
                        except ProcessLookupError:
                            claude_alive = False
                    if not claude_alive:
                        print(f"  [{_now()}] Claude process exited (PID {claude_pid}). Final checkpoint...")
                        # Clear start-time record: this session's Claude is gone.
                        if session_id:
                            _CLAUDE_IDENTITY.pop(session_id, None)
                        # Option (b) defense-in-depth: unlink pidfile IMMEDIATELY so a
                        # concurrent SessionStart for the new Claude doesn't see a stale
                        # transient-daemon slot. The finally-block call is a no-op after
                        # this (CAS fails cleanly — we no longer own the file).
                        _safe_unlink_session_pidfile(sess.get("session_id"))
                        checkpoint_team(session_path=session_path, quiet=False)
                        print(f"  Guard stopping (Claude exited).")
                        break

                current_size = session_path.stat().st_size

                # ── F/H: idle detection + adaptive poll back-off ──────────
                # "idle" = the transcript hasn't grown since last cycle (we're between
                # turns). Drives the interactive reload gate (E), the no-op skip (G),
                # and exponential poll back-off (F).
                idle = (prev_size >= 0 and current_size == prev_size)
                if idle:
                    idle_cycles += 1
                else:
                    idle_cycles = 0
                # NB: poll_interval (F back-off) is decided at the END of the cycle,
                # once we know whether a HARD tier fired — over a hard tier the reload
                # (E) or the HARD circuit-breaker owns the cadence, so F stands down.

                # ── Phase 1: Continuous checkpoint ────────────────────────
                state = checkpoint_team(
                    session_path=session_path,
                    quiet=True,
                )

                # Track team state changes silently — only note when prune/threshold fires
                if state and not state.is_empty():
                    team_hash = f"{len(state.subagents)}:{len(state.tasks)}:{state.message_count}"
                    if team_hash != last_team_hash:
                        checkpoint_count += 1
                        last_team_hash = team_hash

                # ── Token check (fast, from tail of file) ────────────────
                current_tokens = None
                if threshold_tokens is not None or soft_threshold_tokens is not None:
                    current_tokens = quick_token_estimate(session_path)

                # Detect if agents are actively running (reload would kill them)
                agents_active = False
                if state and not state.is_empty():
                    agents_active = any(
                        s.status in ("running", "unknown")
                        for s in state.subagents
                    )
                last_agents_active = agents_active  # C2: remembered for the escalation gate

                # ── E: interactive reload gating ──────────────────────────
                # Interactive sessions never reload mid-turn — they wait for an idle
                # breakpoint (the Stop-hook nudge has already warned the user at the
                # turn that crossed the tier). Once past the force line (~88%) a
                # higher-fidelity reload still beats hitting the autocompact wall, so we
                # allow it even mid-turn; the safe_to_reload gate inside
                # guard_prune_cycle keeps protecting any in-flight Workflow/subagent
                # even then. Headless sessions are unchanged (reload immediately).
                force_now = (
                    force_threshold_tokens is not None
                    and current_tokens is not None
                    and current_tokens >= force_threshold_tokens
                )
                # Require SUSTAINED idle (N consecutive stable cycles), not a single
                # one — a momentary mid-turn stall must not be read as a breakpoint.
                sustained_idle = idle and idle_cycles >= _idle_reload_cycles()
                # E (warned-before-reload): an interactive reload needs BOTH a sustained
                # idle breakpoint AND the user warned (the nudge upserts sentinel.warned
                # at the turn that crossed the tier), with a grace fallback so a
                # missing/disabled nudge can't wedge it. Force (88%) + headless bypass.
                defer_for_turn = False
                if interactive_mode and not force_now:
                    if not sustained_idle:
                        defer_for_turn = True                      # mid-turn: never reload
                    else:
                        _grace = _reload_warn_grace()
                        _armed = read_armed(sess["session_id"], session_path)
                        _warned = bool(_armed and _armed.get("warned"))
                        _at = (_armed or {}).get("armed_at")
                        _grace_ok = _grace <= 0 or (bool(_armed) and _at is not None
                                                    and (time.time() - _at) >= _grace)
                        if _warned or _grace_ok:
                            defer_for_turn = False                 # warned/waited → reload
                        else:
                            # Arm (so the nudge can warn) when unarmed OR when an
                            # existing sentinel lacks the grace clock — backfilling
                            # armed_at so a corrupt/old sentinel can't wedge forever.
                            if not _armed or _at is None:
                                _arm_tier = 80 if (hard2_threshold_tokens and current_tokens
                                                   and current_tokens >= hard2_threshold_tokens) else 55
                                write_armed(sess["session_id"], session_path, _arm_tier, 0.0)
                            defer_for_turn = True                   # hold until warned/grace
                eff_auto_reload = auto_reload and not defer_for_turn
                # Whether a HARD tier is active this cycle (gates F's idle back-off).
                hard_active = (
                    (hard2_threshold_tokens is not None and current_tokens is not None
                     and current_tokens >= hard2_threshold_tokens)
                    or (threshold_tokens is not None and current_tokens is not None
                        and current_tokens >= threshold_tokens)
                )

                # ── Phase 4: HARD2 (80%) — aggressive + reload, GATED by the
                #    safe-point check. NEVER force-terminates through in-flight work
                #    (running Workflow / subagent / open call): the safe_to_reload gate
                #    inside guard_prune_cycle defers those to a read-only checkpoint and
                #    lets the autocompact wall be the lesser evil. (Was: "reload ALWAYS,
                #    even with agents" — that was the catastrophic data-loss bug.) ──
                hard2_tokens_hit = (
                    hard2_threshold_tokens is not None
                    and current_tokens is not None
                    and current_tokens >= hard2_threshold_tokens
                )
                if hard2_tokens_hit:
                    prune_count += 1
                    reason = f"{current_tokens:,} tokens >= {hard2_threshold_tokens:,} (80%)"
                    print(f"  [{_now()}] HARD2 THRESHOLD (80%): {reason}")
                    print(f"  Aggressive prune + reload (cycle #{prune_count}) — gated by safe-point check...")

                    if defer_for_turn:
                        _force_note = (f" (or force at {int(force_pct * 100)}%)"
                                       if force_threshold_tokens else "")
                        _why = ("waiting for the warning to reach you" if sustained_idle
                                else "interactive turn in progress")
                        print(f"  Armed — {_why}; will reload at the next safe breakpoint"
                              f"{_force_note}. Read-only checkpoint now.")
                    result = guard_prune_cycle(
                        session_path=session_path,
                        rx_name="aggressive",
                        config=config,
                        auto_reload=eff_auto_reload,
                        cwd=cwd or os.getcwd(),
                        session_id=sess["session_id"],
                        claude_pid=claude_pid,
                        protect_patterns=protect_patterns,
                        # --no-reload OR an interactive mid-turn defer: we won't
                        # terminate Claude, so we can't safely write the live file
                        # (#106) — go read-only instead of falsely reporting a prune
                        # that never persisted.
                        read_only_live=not eff_auto_reload,
                        project=defer_for_turn,  # compute the real reclaim % for the nudge
                    )
                    if defer_for_turn:
                        _arm_nudge_from_result(sess["session_id"], session_path, 80, result)

                    if result.get("reloading"):
                        clear_armed(sess["session_id"], session_path)  # consumed
                        from .helpers import get_savings_line
                        savings = get_savings_line()
                        if savings:
                            print(f"  {savings}")
                        print(f"  Reload triggered. Guard exiting.")
                        break

                    if result.get("reload_unsafe"):
                        pass  # safe-point gate already printed "Reload DEFERRED — <reason>"
                    elif result.get("live_write_skipped"):
                        print(f"  Read-only — live session not rewritten (#106).")
                    elif result.get("futile_reload_skipped"):
                        pass  # futile prune — nothing persisted (live file untouched)
                    else:
                        print(f"  Pruned: {_fmt_prune_result(result)}")
                    if result.get("team_name"):
                        print(f"  Team '{result['team_name']}' state preserved ({result['team_messages']} messages)")
                    # Reaching here means HARD2 did NOT reload (the reloading branch
                    # above breaks/exits first). Apply the same circuit-breaker
                    # accounting as HARD1 so a sustained deferred-conflict / futile
                    # prune at 80% backs off and eventually exits instead of
                    # spinning kill→no-write→resume forever.
                    _account_hard_prune(result, agents_active, state)
                    print()

                # ── Phase 3: HARD1 (55%) — standard + reload (SKIP reload if agents active) ──
                elif (threshold_tokens is not None
                      and current_tokens is not None
                      and current_tokens >= threshold_tokens):
                    prune_count += 1
                    reason = f"{current_tokens:,} tokens >= {threshold_tokens:,} (55%)"

                    if agents_active:
                        # Agents running — read-only checkpoint, no reload (don't kill
                        # active work) and no live write (#106: rewriting the file
                        # Claude holds open races the harness). HARD2 (80%) force-
                        # reloads later if context keeps growing, terminating first.
                        print(f"  [{_now()}] HARD THRESHOLD (55%): {reason}")
                        print(f"  Agents active — read-only checkpoint, deferring prune+reload (cycle #{prune_count})...")

                        result = guard_prune_cycle(
                            session_path=session_path,
                            rx_name=rx_name,
                            config=config,
                            auto_reload=False,  # Don't reload — agents are working
                            cwd=cwd or os.getcwd(),
                            session_id=sess["session_id"],
                            read_only_live=True,
                        )
                    else:
                        print(f"  [{_now()}] HARD THRESHOLD (55%): {reason}")
                        if defer_for_turn:
                            _why = ("waiting for the warning to reach you" if sustained_idle
                                    else "interactive turn in progress")
                            print(f"  Armed — {_why}; will reload at the next safe "
                                  f"breakpoint. Read-only checkpoint now.")
                        else:
                            print(f"  Standard prune + reload (cycle #{prune_count})...")

                        result = guard_prune_cycle(
                            session_path=session_path,
                            rx_name=rx_name,
                            config=config,
                            auto_reload=eff_auto_reload,
                            cwd=cwd or os.getcwd(),
                            session_id=sess["session_id"],
                            claude_pid=claude_pid,
                            protect_patterns=protect_patterns,
                            # --no-reload OR an interactive mid-turn defer: read-only
                            # (can't safely write a live file without terminating
                            # Claude — #106).
                            read_only_live=not eff_auto_reload,
                            project=defer_for_turn,  # compute the real reclaim % for the nudge
                        )
                        if defer_for_turn:
                            _arm_nudge_from_result(sess["session_id"], session_path, 55, result)

                    if result.get("reloading"):
                        clear_armed(sess["session_id"], session_path)  # consumed
                        from .helpers import get_savings_line
                        savings = get_savings_line()
                        if savings:
                            print(f"  {savings}")
                        print(f"  Reload triggered. Guard exiting.")
                        break

                    if result.get("reload_unsafe"):
                        pass  # safe-point gate already printed "Reload DEFERRED — <reason>"
                    elif result.get("live_write_skipped"):
                        print(f"  Read-only — live session not rewritten (#106).")
                    elif result.get("futile_reload_skipped"):
                        pass  # futile prune — nothing persisted (live file untouched)
                    else:
                        print(f"  Pruned: {_fmt_prune_result(result)}")
                    if result.get("team_name"):
                        print(f"  Team '{result['team_name']}' state preserved ({result['team_messages']} messages)")

                    _account_hard_prune(result, agents_active, state)
                    print()

                # ── Phase 2: SOFT (25%) — gentle, no reload (file maintenance only) ──
                else:
                    soft_bytes_hit = current_size >= soft_threshold_bytes
                    soft_tokens_hit = (
                        soft_threshold_tokens is not None
                        and current_tokens is not None
                        and current_tokens >= soft_threshold_tokens
                    )
                    if (soft_bytes_hit or soft_tokens_hit) and idle:
                        # G (no-op accounting): the transcript hasn't grown since last
                        # cycle, so a gentle recompute would reproduce the identical
                        # read-only result. Skip it — the Phase-1 checkpoint above
                        # already refreshed team state — and do NOT count it as a SOFT
                        # "fire" (the 1056-no-op problem from issue #115).
                        noop_cycles += 1
                    elif soft_bytes_hit or soft_tokens_hit:
                        soft_prune_count += 1
                        reason = f"{current_tokens:,} tokens >= {soft_threshold_tokens:,} (25%)" if soft_tokens_hit else f"{current_size / 1024 / 1024:.1f}MB"
                        print(f"  [{_now()}] SOFT THRESHOLD (25%): {reason}")
                        print(f"  Read-only checkpoint — live prune deferred to reload tier (#106) (cycle #{soft_prune_count})...")

                        result = guard_prune_cycle(
                            session_path=session_path,
                            rx_name="gentle",
                            config=config,
                            auto_reload=False,
                            cwd=cwd or os.getcwd(),
                            session_id=sess["session_id"],
                            read_only_live=True,
                        )

                        if result.get("team_name"):
                            print(f"  Team '{result['team_name']}' checkpointed ({result['team_messages']} messages)")
                        print()

                # ── F: idle poll back-off (decided here, after the tier check) ──
                # Back off the top-of-loop poll ONLY when nothing is actionable: the
                # session is idle (no growth) AND below the HARD tiers. Over a hard
                # tier, E's idle reload or the HARD circuit-breaker owns the cadence,
                # so we keep polling at the base interval. `interval` itself is never
                # mutated — only this separate poll_interval.
                _bo = _idle_backoff_cycles()
                if not idle or hard_active:
                    poll_interval = interval
                elif _bo and idle_cycles >= _bo:
                    poll_interval = min(interval * (2 ** (idle_cycles - _bo + 1)), 300)

                # End-of-cycle: remember this size so the next cycle can detect idle.
                prev_size = current_size
                consecutive_cycle_errors = 0  # a clean cycle clears the error streak
                deferred_error_cycles = 0

            except (KeyboardInterrupt, SystemExit):
                raise  # voluntary exits (Ctrl-C, K-exit) reach the outer handler/finally
            except UnicodeDecodeError as _dec_exc:
                # C1/C2 crossfix: a non-UTF-8 session makes the strict loader raise
                # EVERY cycle. Escalating+respawning on it would be a RESPAWN STORM
                # (the error is deterministic — a fresh daemon hits it too). Treat it
                # as a BENIGN skip: this session just isn't prunable until its bytes
                # become valid UTF-8. Do NOT count toward the escalation; log once.
                if not logged_decode_skip:
                    print(f"  Guard: session is not valid UTF-8 — skipping prune for this "
                          f"session (not an error, no respawn): {_dec_exc!r}", flush=True)
                    logged_decode_skip = True
                continue
            except Exception as _cycle_exc:
                # Defense-in-depth: one malformed cycle must never kill the daemon
                # (the only outer handler is KeyboardInterrupt). BUT a DETERMINISTIC
                # per-cycle error must not silently spin forever as an inert-but-alive
                # daemon (invisible to the watchdog). Count consecutive errors; after
                # K, escalate with a watchdog-detectable marker and EXIT so the
                # SessionStart hook respawns a fresh daemon (which also makes a
                # genuine repeating failure visible as a respawn pattern).
                consecutive_cycle_errors += 1
                print(f"  Guard: skipping a cycle after an unexpected error "
                      f"({consecutive_cycle_errors}/{GUARD_CYCLE_ERROR_EXIT}): {_cycle_exc!r}", flush=True)
                if consecutive_cycle_errors >= GUARD_CYCLE_ERROR_EXIT:
                    # C2: defer the escalation-exit while a subagent team is live —
                    # don't drop guard coverage mid-team. BUT BOUND the deferral: a
                    # guard erroring this long while "agents active" may be reading a
                    # stale/zombie team state, and spinning inert forever (the
                    # round-3 regression) is worse than a brief gap. After
                    # GUARD_CYCLE_ERROR_DEFER_MAX deferred cycles, escalate anyway.
                    if last_agents_active and deferred_error_cycles < GUARD_CYCLE_ERROR_DEFER_MAX:
                        deferred_error_cycles += 1
                        print(f"  Guard: {consecutive_cycle_errors} cycle errors but agents are "
                              f"active — deferring respawn ({deferred_error_cycles}/"
                              f"{GUARD_CYCLE_ERROR_DEFER_MAX}).", flush=True)
                        consecutive_cycle_errors = GUARD_CYCLE_ERROR_EXIT  # hold at threshold
                        continue
                    print(f"  Guard cycle-error escalation: {consecutive_cycle_errors} consecutive "
                          f"cycle errors ({deferred_error_cycles} deferred) — exiting for respawn "
                          f"(last: {_cycle_exc!r}).", flush=True)
                    sys.exit(1)
                continue
    except KeyboardInterrupt:
        # Final checkpoint before exit
        checkpoint_team(session_path=session_path, quiet=True)
        total_prunes = prune_count + soft_prune_count
        _noop_note = f" ({noop_cycles} idle no-op cycles skipped)" if noop_cycles else ""
        if total_prunes:
            print(f"\n  Guard stopped. Pruned {total_prunes}x during this session.{_noop_note}")
        else:
            print(f"\n  Guard stopped.{_noop_note}")
    finally:
        # Stop reactive watcher on ALL exit paths (KeyboardInterrupt and the
        # four `break` paths inside the main loop: file disappeared,
        # Claude exited, Hard2 reload, Hard1 reload). Previously the watcher
        # thread would leak past normal-exit breaks and fire one more
        # recovery on a dead session.
        if overflow_watcher:
            try:
                overflow_watcher.stop()
            except Exception:
                pass
        # Unlink session pidfile on EVERY daemon-exit path (PR #93 commit 2,
        # class-of-bug fold). Covers SIGTERM, K=10 voluntary exit,
        # KeyboardInterrupt, and the four `break` paths above. The helper
        # CAS-checks ``_pid_file_points_to(session_id, os.getpid())`` so we
        # never destroy a peer's just-completed claim during a hot reload.
        # ``sys.exit(0)`` raises ``SystemExit`` which DOES run try/finally,
        # so this single call site is sufficient for all 6 surfaces.
        try:
            _safe_unlink_session_pidfile(sess["session_id"])
        except Exception:
            pass


def _detect_interactive(claude_pid: int | None) -> bool:
    """Component H — is this an interactive (TTY) Claude session?

    COZEMPIC_INTERACTIVE=on|off forces the answer; the default ``auto`` checks
    whether the Claude process owns a controlling terminal (``claude -p`` /
    headless / CI runs have none). When uncertain we default to *interactive*
    (True) — the safer bias, because interactive mode only ever makes the guard
    MORE conservative about reloading (warn + reload at idle, never mid-turn).
    """
    mode = os.environ.get("COZEMPIC_INTERACTIVE", "auto").strip().lower()
    if mode in ("on", "1", "true", "yes"):
        return True
    if mode in ("off", "0", "false", "no"):
        return False
    if not claude_pid:
        return True
    try:
        out = subprocess.run(
            ["ps", "-o", "tty=", "-p", str(claude_pid)],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
    except Exception:
        return True
    # A real terminal looks like "s001"/"ttys001"/"pts/3"; headless is "??"/"?"/"-"/"".
    return bool(out) and out not in ("??", "?", "-")


def _idle_backoff_cycles() -> int:
    """Component F — cycles of a stable transcript before poll back-off kicks in."""
    try:
        n = int(os.environ.get("COZEMPIC_IDLE_BACKOFF_CYCLES", "4"))
        return min(n, 10_000) if n > 0 else 0  # cap absurd values
    except (TypeError, ValueError):
        return 4


def _idle_reload_cycles() -> int:
    """Component E — consecutive stable-transcript cycles required before an
    interactive reload treats the session as a genuine breakpoint. A single
    stable cycle is NOT enough: a live turn can stall one poll interval (deep
    model thinking, rate-limit back-off, a slow buffered tool) without the
    JSONL growing, and reloading then would interrupt an in-progress turn. Two
    consecutive idle cycles (~2 intervals) distinguishes a real lull from a
    momentary stall. Minimum 1."""
    try:
        n = int(os.environ.get("COZEMPIC_IDLE_RELOAD_CYCLES", "2"))
        # Cap an absurd value (e.g. 10**400) so `idle_cycles >= n` can't be made
        # permanently unreachable (which would silently disable idle reloads).
        return min(n, 10_000) if n >= 1 else 1
    except (TypeError, ValueError):
        return 2


def _reload_warn_grace() -> float:
    """Component E — seconds the daemon waits for the nudge to WARN the user before
    an interactive idle reload proceeds anyway (fallback so a missing/disabled
    Stop-hook nudge can't wedge reloads forever). Default 120s; <=0 disables the
    wait (reload as soon as idle)."""
    try:
        v = float(os.environ.get("COZEMPIC_RELOAD_WARN_GRACE", "120"))
        # Reject NaN/inf: a non-finite grace makes `elapsed >= grace` always False,
        # which permanently DISABLES this fallback (the exact gate-disable bug class
        # as the CLI/config thresholds — IEEE-754: every NaN/inf comparison fails),
        # silently wedging the interactive idle reload. Mirror _read_min_prune_ratio.
        return v if math.isfinite(v) else 120.0
    except (TypeError, ValueError):
        return 120.0


def _force_reload_pct() -> float:
    """Component E — context fraction past which an interactive reload fires even
    mid-turn (a higher-fidelity reload still beats the autocompact wall; the
    safe_to_reload gate inside the cycle keeps protecting in-flight work even
    here). Default 0.88; set <=0 or >1 to disable the mid-turn force entirely."""
    try:
        v = float(os.environ.get("COZEMPIC_FORCE_RELOAD_PCT", "0.88"))
        return v if 0.0 < v <= 1.0 else 0.0
    except (TypeError, ValueError):
        return 0.88


def _arm_nudge_from_result(session_id, session_path, tier, result):
    """Component E — after an interactive mid-turn defer (a read-only HARD cycle),
    arm the nudge sentinel so the Stop-hook nudge can show the REAL projected
    reduction the deferred reload would reclaim. Best-effort; never raises into
    the daemon loop."""
    try:
        orig = result.get("original_tokens") or 0
        # Prefer the projected post-prune estimate (computed on the read-only defer
        # path when project=True); fall back to final_tokens.
        fin = result.get("projected_final_tokens")
        if fin is None:
            fin = result.get("final_tokens")
        proj = 0.0
        if orig and fin is not None and 0 <= fin < orig:
            proj = (orig - fin) / orig * 100.0
        write_armed(session_id, session_path, tier, proj)
    except Exception:
        pass


def guard_prune_cycle(
    session_path: Path,
    rx_name: str = "standard",
    config: dict | None = None,
    auto_reload: bool = True,
    cwd: str = "",
    session_id: str | None = None,
    claude_pid: int | None = None,
    read_only_live: bool = False,
    project: bool = False,
    protect_patterns: list | None = None,
) -> dict:
    """Execute a single guard prune cycle.

    Holds a _PruneLock for the duration so concurrent guard instances cannot
    race each other.  Takes a _FileSnapshot before loading so that any lines
    Claude appends while pruning is in progress are preserved in the output
    (or the cycle is deferred on conflict).

    Returns dict with: saved_mb, team_name, team_messages, reloading, checkpoint_path
    """
    from .tokens import estimate_session_tokens, calibrate_ratio

    _no_change = {
        "saved_mb": 0.0,
        "original_tokens": 0,
        "final_tokens": 0,
        "team_name": None,
        "team_messages": 0,
        "checkpoint_path": None,
        "backup_path": None,
        "reloading": False,
    }

    try:
        with _PruneLock(session_path):
            # Size guard: skip prune for very large sessions (OOM risk #74).
            # Cheap stat — do it before the full read.
            file_size_mb = session_path.stat().st_size / 1024 / 1024
            if file_size_mb > 200:
                print(f"  [{_now()}] Session {file_size_mb:.0f}MB exceeds 200MB — skipping prune (OOM risk).", file=sys.stderr)
                return _no_change

            # Read once: messages + snapshot from identical bytes so a line Claude
            # appends mid-prune can't be both loaded AND counted in the delta
            # (TOCTOU append-duplication). Replaces snapshot_session()+load_messages().
            messages, snap = load_messages_and_snapshot(session_path)
            original_bytes = sum(b for _, _, b in messages)

            # --protect-pattern (#122): tag matching messages so the strategies spare
            # them (via is_protected) during this prune cycle. Stripped after the prune.
            if protect_patterns:
                tag_pattern_matches(messages, protect_patterns)

            # Token estimate before pruning — capture calibrated ratio before metadata-strip
            pre_te = estimate_session_tokens(messages)
            pre_ratio = calibrate_ratio(messages)

            # Prune with team protection.
            # P0-B: catch PruneValidationError from run_prescription inside
            # prune_with_team_protect. On validation failure: log the failure,
            # return _no_change immediately. The deferred writer (_write_pruned_after_exit)
            # is NOT set (it is defined later in this function), so the file stays
            # untouched and Claude is NOT terminated.
            try:
                pruned_messages, results, team_state = prune_with_team_protect(
                    messages, rx_name=rx_name, config=config,
                )
            except PruneValidationError as ve:
                check = ve.evidence.get("failed_check", "?")
                print(
                    f"  [{_now()}] Prune validation failed ({check}): {ve.reason} "
                    f"— aborting prune, session file unchanged.",
                    file=sys.stderr,
                )
                return {
                    **_no_change,
                    "validation_error": ve.reason,
                    "evidence": ve.evidence,
                }

            # --protect-pattern: strip the transient tag now (the strategies already
            # honored it via is_protected) so it can never persist into the saved
            # session. On the validation-error path above we return before saving, so
            # the in-memory tag is discarded with no disk leak.
            if protect_patterns:
                strip_pattern_tags(messages)
                strip_pattern_tags(pruned_messages)

            # #106 — never rewrite a live session that Claude holds open.
            # The no-reload tiers (SOFT 25%, agents-active HARD) reach here with
            # read_only_live=True. os.replace-ing the file Claude is actively
            # appending to races the harness (TOCTOU + inode-swap → lost/garbled
            # transcript), and because Claude reads the JSONL only at
            # startup/resume the on-disk rewrite cannot shrink the LIVE context
            # anyway — all risk, no upside. Preserve team state via a read-only
            # checkpoint and skip the destructive write. The HARD/reload tiers
            # (which terminate Claude first) still do the real prune.
            if read_only_live:
                checkpoint_path = None
                if not team_state.is_empty():
                    checkpoint_path = write_team_checkpoint(team_state, session_path.parent)
                # When arming the nudge (project=True), compute the post-prune token
                # estimate so the warning can show the REAL reclaim %. The prune was
                # already computed (pruned_messages) — we just don't WRITE it. Kept
                # behind the flag so the every-cycle SOFT checkpoint stays cheap.
                projected_final = None
                if project:
                    try:
                        projected_final = estimate_session_tokens(
                            pruned_messages, pre_calibrated_ratio=pre_ratio).total
                    except Exception:
                        projected_final = None
                # ALWAYS expose the COMPUTED prune's byte reduction (the prune was
                # already run; this is cheap) so the futile-loop breaker can tell an
                # UNPRUNABLE live session (must K-exit) from a busy-but-prunable one
                # (benign). Without this, a live session that frees ~0 looped forever
                # (f641174c, 2026-06-10).
                _ro_would_free = original_bytes - sum(b for _, _, b in pruned_messages)
                return {
                    "saved_mb": 0.0,
                    "original_tokens": pre_te.total,
                    "final_tokens": pre_te.total,
                    "projected_final_tokens": projected_final,
                    "would_free_mb": _ro_would_free / 1024 / 1024,
                    "original_bytes": original_bytes,
                    "team_name": _log_safe(team_state.team_name) or None,
                    "team_messages": team_state.message_count,
                    "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                    "backup_path": None,
                    "reloading": False,
                    "live_write_skipped": True,
                }

            final_bytes = sum(b for _, _, b in pruned_messages)
            saved_bytes = original_bytes - final_bytes

            # If pruning freed nothing (or grew the file via team recovery injection), don't
            # save — avoids backup accumulation and file growth on ineffective prescriptions (#16, #19).
            if saved_bytes <= 0:
                return {
                    "saved_mb": 0.0,
                    "original_tokens": pre_te.total,
                    "final_tokens": pre_te.total,
                    "team_name": _log_safe(team_state.team_name),
                    "team_messages": team_state.message_count,
                    "checkpoint_path": None,
                    "backup_path": None,
                    "reloading": False,
                }

            # GAP-D: futile-reload abort. If prune saved less than _MIN_PRUNE_RATIO
            # of original bytes, the resumed Claude would re-trigger HARD immediately
            # (context is dominated by immutable tool-result blocks that prune cannot
            # touch). Skip the reload; persist the prune output; let K-counter advance
            # so the circuit breaker eventually exits the daemon.
            if 0 < saved_bytes < original_bytes * _MIN_PRUNE_RATIO:
                # Futile: the prune saved too little to justify a reload that
                # would immediately re-trigger HARD. We are NOT terminating Claude
                # this cycle, so per #106 we must NOT os.replace the live file the
                # harness holds open — just checkpoint team state. The K-counter
                # still advances so the circuit breaker eventually exits.
                checkpoint_path = None
                if not team_state.is_empty():
                    project_dir = session_path.parent
                    checkpoint_path = write_team_checkpoint(team_state, project_dir)
                return {
                    "saved_mb": 0.0,  # nothing persisted — live write skipped (#106)
                    "would_free_mb": saved_bytes / 1024 / 1024,
                    "original_bytes": original_bytes,
                    "original_tokens": pre_te.total,
                    "final_tokens": pre_te.total,  # post_te not computed (early return)
                    "team_name": _log_safe(team_state.team_name) or None,
                    "team_messages": team_state.message_count,
                    "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                    "backup_path": None,
                    "reloading": False,
                    "futile_reload_skipped": True,
                }

            # Token estimate after pruning — pass pre-calibrated ratio
            post_te = estimate_session_tokens(pruned_messages, pre_calibrated_ratio=pre_ratio)

            # Post-prune TOKEN-PROGRESS gate (the confirmed-write reload-loop fix).
            # The byte gate above (saved_bytes < 10% of original) catches a prune
            # that freed little DATA, but a prune can free >10% of *bytes* while
            # barely reducing *tokens* (low-token-density content: progress ticks,
            # whitespace, repeated boilerplate). Reloading then resumes a session
            # whose token count is essentially unchanged, which immediately
            # re-triggers HARD — and because a successful reload makes the daemon
            # exit (a fresh guard respawns with the in-process breaker reborn at 0),
            # this loops one CONFIRMED prune+counter-ping per cycle, invisible to
            # the per-process circuit breaker. If the prune did not reduce TOKENS
            # by at least _MIN_PRUNE_RATIO, treat the reload as futile: skip it
            # (read-only, #106) and advance the breaker so the daemon backs off and
            # exits. Gating on PROGRESS (not an absolute token floor) deliberately
            # ALLOWS a prune that frees real headroom to reload even if it lands
            # just above the trigger band — that converges over a cycle or two, and
            # the disk reload-rate ledger below bounds any residual regrow loop.
            # NB: gate on pre_te.total only — a maximal prune to post_te.total==0
            # is FULL progress (pre - 0), not zero; `and post_te.total` would
            # wrongly read it as 0 progress and skip the reload.
            _tokens_saved_now = (
                pre_te.total - post_te.total if pre_te.total else 0
            )
            if (
                auto_reload
                and pre_te.total
                and _tokens_saved_now < pre_te.total * _MIN_PRUNE_RATIO
            ):
                checkpoint_path = None
                if not team_state.is_empty():
                    project_dir = session_path.parent
                    checkpoint_path = write_team_checkpoint(team_state, project_dir)
                print(
                    f"  [{_now()}] Reload skipped — prune reduced tokens by only "
                    f"{_tokens_saved_now:,} (<{int(_MIN_PRUNE_RATIO * 100)}% of "
                    f"{pre_te.total:,}); a reload would re-trigger HARD immediately "
                    f"(futile).",
                    file=sys.stderr,
                )
                return {
                    "saved_mb": 0.0,  # nothing persisted — live write skipped (#106)
                    "would_free_mb": saved_bytes / 1024 / 1024,
                    "original_bytes": original_bytes,
                    "original_tokens": pre_te.total,
                    "final_tokens": post_te.total,
                    "team_name": _log_safe(team_state.team_name) or None,
                    "team_messages": team_state.message_count,
                    "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                    "backup_path": None,
                    "reloading": False,
                    "futile_reload_skipped": True,
                    "token_progress_insufficient": True,
                }

            # Write checkpoint if team exists
            checkpoint_path = None
            if not team_state.is_empty():
                project_dir = session_path.parent
                checkpoint_path = write_team_checkpoint(team_state, project_dir)

            # #106: the pruned session is NOT written here. The live file the
            # harness holds open must only be os.replace'd AFTER Claude is
            # terminated — see the deferred writer below (_write_pruned_after_exit).
            # Team state is checkpointed above (read-only) regardless.

    except PruneLockError as exc:
        print(f"  [{_now()}] Prune deferred — lock held: {exc}", file=sys.stderr)
        return _no_change
    except PruneConflictError as exc:
        print(f"  [{_now()}] Prune deferred — conflict detected: {exc}", file=sys.stderr)
        return _no_change

    # Projected savings — recorded to the lifetime tracker + global prune counter
    # ONLY after the prune is CONFIRMED persisted (see the written-gate in the
    # reload block below). Recording here (pre-write) inflated the prune/tokens
    # counters on every deferred or looping cycle that never actually wrote — the
    # in-the-wild prune-spike signature: a stuck guard re-pruning every interval
    # bumps the counter each time despite persisting nothing.
    # Gate on pre_te.total ONLY (ynaamane review #4): `and post_te.total` is the same
    # trap fixed at _tokens_saved_now above — a maximal prune to post_te.total == 0 is
    # FULL progress (pre - 0), not zero, so requiring post_te.total to be truthy makes
    # the largest-possible savings event record as 0.
    tokens_saved = (pre_te.total - post_te.total) if pre_te.total else 0

    def _record_persisted_savings():
        """Record savings to the lifetime tracker / global counter. Call ONLY
        once the prune has actually been written to disk."""
        if tokens_saved <= 0:
            return
        from .helpers import record_savings, get_msg_type
        turn_count = sum(1 for _, m, _ in messages
                       if get_msg_type(m) == "user"
                       and isinstance(m.get("message", {}).get("content", ""), str))
        record_savings(tokens_saved, total_tokens=pre_te.total, turn_count=turn_count)

    # #106 deferred writer — persists the pruned session ONLY after the process
    # holding it is dead. Re-acquires the prune lock; the snapshot makes the
    # write append-aware (any lines Claude wrote before dying are preserved); on
    # conflict it aborts, leaving the original intact (Claude resumes from the
    # full file — safe). Invoked by _terminate_and_resume after _wait_for_exit.
    _write_holder = {"backup": None, "written": False, "error": None}

    def _write_pruned_after_exit():
        try:
            with _PruneLock(session_path):
                bk = save_messages(
                    session_path, pruned_messages, create_backup=True, snapshot=snap
                )
            if bk:
                cleanup_old_backups(session_path, keep=3)
            _write_holder["backup"] = bk
            _write_holder["written"] = True
            # Record savings ONLY now that the prune is confirmed persisted. This
            # is the single recording point for BOTH the auto_reload terminate-
            # first path and the auto_reload=False overflow path (both invoke this
            # writer post-death), so a deferred/futile/looping cycle that never
            # writes can never inflate the prune/tokens counters.
            _record_persisted_savings()
        except (PruneConflictError, PruneLockError) as exc:
            _write_holder["error"] = "conflict"
            print(f"  [{_now()}] Deferred prune write skipped — {exc}", file=sys.stderr)
        except OSError as exc:
            _write_holder["error"] = "oserror"
            # Disk-full / EIO / permission at the post-kill write instant. The
            # write is atomic (save_messages leaves the original intact on any
            # failure), so there's no corruption — but this runs AFTER Claude was
            # terminated and BEFORE the resume watcher spawns, so an uncaught
            # error would propagate out of _terminate_and_resume and crash the
            # daemon, leaving Claude killed-but-not-resumed. Contain it: leave the
            # full file for resume (written stays False) and let the reload proceed.
            print(f"  [{_now()}] Deferred prune write failed ({exc}) — resuming from full file.", file=sys.stderr)

    result = {
        "saved_mb": saved_bytes / 1024 / 1024,
        "original_tokens": pre_te.total,
        "final_tokens": post_te.total,
        "team_name": _log_safe(team_state.team_name) or None,
        "team_messages": team_state.message_count,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "backup_path": None,
        "reloading": False,
    }

    # Trigger reload — terminate Claude FIRST, then the deferred writer persists
    # the prune (post-death), then resume. This closes the #106 race: the live
    # inode is never swapped while Claude holds the file open.
    if auto_reload:
        reload_pid = claude_pid if claude_pid is not None else find_claude_pid()
        if reload_pid:
            # ── SAFE-POINT GATE (1.8.22) — validate BEFORE terminate ──────────
            # NEVER SIGKILL the process while in-flight work would be destroyed:
            # a running Workflow-tool orchestration, a background subagent, an
            # open tool call, or an active agent team. That state is harness-side
            # / not-yet-flushed and a transcript resume cannot recover it. Unsafe
            # ⇒ read-only checkpoint + defer (the autocompact wall is the lesser
            # evil — it keeps the live process + workflow alive; a terminate kills
            # them). Applies to ALL tiers incl. HARD2 (replaces the old "reload
            # ALWAYS, even with agents" behavior — the catastrophic bug).
            _safe, _reason = safe_to_reload(team_state, messages, session_path)
            if not _safe:
                print(
                    f"  [{_now()}] Reload DEFERRED — {_reason}. NOT terminating "
                    f"Claude (would destroy in-flight work); read-only checkpoint "
                    f"kept. Will reload once the session is at a safe point.",
                    file=sys.stderr,
                )
                result["saved_mb"] = 0.0
                result["live_write_skipped"] = True
                result["reload_unsafe"] = True
                result["unsafe_reason"] = _reason
                # Deliberately do NOT set futile_reload_skipped: deferring around
                # genuine in-flight work is NOT a futile prune. _account_hard_prune's
                # benign-read-only branch (live_write_skipped & not conflict) leaves
                # the circuit breaker untouched, so a long-running Workflow doesn't
                # trip the K-exit and quit the guard. The token-% nudge is the
                # user-facing signal here; the guard keeps checkpointing meanwhile.
                return result
            # Cross-respawn reload-rate cap: if this session has already reloaded
            # too many times within the window, stop auto-reloading it. A session
            # that re-bloats to threshold immediately after each prune would
            # otherwise churn kill→resume→re-bloat forever (one confirmed prune+
            # ping per cycle), invisible to the per-process breaker which is reborn
            # at 0 on every respawn. The ledger is on disk so it survives respawn.
            _ledger = _reload_ledger_path(session_id, session_path)
            _capped, _n = _reload_rate_exceeded(_ledger)
            if _capped:
                print(
                    f"  [{_now()}] Reload rate cap hit ({_n} reloads in "
                    f"{_reload_ledger_window_s() // 60}min) — this session is "
                    f"regrowing to the threshold immediately after each prune. "
                    f"Stopping auto-reload to avoid a kill→resume→re-bloat loop. "
                    f"Consider /clear or splitting the work into a fresh session.",
                    file=sys.stderr,
                )
                result["saved_mb"] = 0.0
                result["live_write_skipped"] = True
                result["reload_rate_capped"] = True
                result["futile_reload_skipped"] = True  # account to breaker → exit
                return result
            from .reload_lock import (
                _ReloadLock, ReloadLockHeld,
                INIT_GUARD_HARD1, INIT_GUARD_HARD2,
            )
            # Pick initiator based on prescription tier — aggressive ==
            # Hard2 (80% emergency), everything else == Hard1 (55% standard).
            initiator = INIT_GUARD_HARD2 if rx_name == "aggressive" else INIT_GUARD_HARD1
            try:
                with _ReloadLock(session_id or session_path.stem, initiator=initiator):
                    _terminate_and_resume(
                        reload_pid, cwd,
                        session_id=session_id,
                        session_path=session_path,
                        write_pruned=_write_pruned_after_exit,
                    )
                # The deferred writer fires only after a confirmed kill, so a
                # successful write == Claude was terminated == a real reload is
                # under way. If it did NOT write (anti-resurrection entry gate
                # because Claude already exited, a failed kill, or an append
                # conflict), nothing was persisted and no real reload happened —
                # keep the daemon alive (reloading=False) and leave the full file
                # for resume. This avoids a misleading "Reload triggered" + exit.
                if _write_holder["written"]:
                    result["reloading"] = True
                    result["backup_path"] = (
                        str(_write_holder["backup"]) if _write_holder["backup"] else None
                    )
                else:
                    result["saved_mb"] = 0.0
                    result["live_write_skipped"] = True
                    if _write_holder.get("error"):
                        result["prune_deferred_conflict"] = True
            except ReloadLockHeld as exc:
                # Another reload pipeline is in flight — it terminates + writes
                # its own prune. We did NOT write the live file (#106-safe).
                print(
                    f"  Reload deferred — another pipeline in flight "
                    f"({exc.holder_initiator}, PID {exc.holder_pid})."
                )
                result["reloading"] = False
                result["saved_mb"] = 0.0
                result["live_write_skipped"] = True
        else:
            # No live Claude PID found. We cannot prove the file is unheld, so
            # per #106 we do NOT rewrite it; resume manually from the full file.
            resume_flag = f"--resume {session_id}" if session_id else "--resume"
            print("  WARNING: Could not find Claude PID — not reloading, live file left intact.")
            print(f"  Restart manually: claude {resume_flag}")
            result["saved_mb"] = 0.0
            result["live_write_skipped"] = True
    else:
        # auto_reload=False reaching here = overflow recovery (a substantial prune;
        # SOFT / agents-active returned read-only earlier). Hand the deferred
        # writer + projected final size to the caller, which terminates Claude
        # itself and then invokes the writer post-death.
        result["_deferred_writer"] = _write_pruned_after_exit
        result["_write_holder"] = _write_holder
        result["_final_bytes"] = final_bytes

    return result


def _is_cozempic_watcher_process(pid: int) -> bool:
    """Verify that `pid` is a cozempic reload watcher (bash + cozempic watcher script).

    Guards against false positives from pgrep substring matching.
    """
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if result.returncode != 0:
            return False
        args = (result.stdout or "").strip()
        # Real watcher script contains both "bash" and "Cozempic guard resumed Claude"
        return "bash" in args and "Cozempic guard resumed Claude" in args
    except (subprocess.SubprocessError, OSError):
        return False


def _cleanup_stale_watchers() -> None:
    """Kill stale reload watchers from previous Cozempic versions.

    Old watchers (pre-1.6.10) had hardcoded resume commands without flag
    detection. They linger as zombie processes waiting for Claude to exit.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", "cozempic.*resumed Claude"],
            capture_output=True, text=True, timeout=5,
        )
        for pid_str in result.stdout.strip().split("\n"):
            if pid_str:
                try:
                    pid = int(pid_str)
                    if _is_cozempic_watcher_process(pid):
                        os.kill(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError, ValueError):
                    pass
    except Exception:
        pass


def _detect_skip_permissions(pid: int) -> bool:
    """Check if the Claude process was launched with --dangerously-skip-permissions."""
    flags = _detect_claude_flags(pid)
    return "--dangerously-skip-permissions" in flags


def _detect_claude_flags(pid: int) -> str:
    """Extract CLI flags from the running Claude process.

    Returns the flags portion of the command line (everything after 'claude'
    but excluding --resume/--continue and the session ID).

    Uses psutil for accurate argv preservation (preserves spaces in values).
    Falls back to ps -o args= with shlex.split when psutil is unavailable.
    """
    import shlex

    parts: list[str] = []

    # Preferred path: psutil preserves original argv boundaries exactly.
    try:
        import psutil
        parts = psutil.Process(pid).cmdline()
    except (ImportError, Exception):
        pass

    # Fallback: ps -o args= + shlex.split (loses space-boundary info on macOS).
    if not parts:
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "args="],
                capture_output=True, text=True, timeout=5,
            )
            raw = result.stdout.strip()
            if not raw or "claude" not in raw:
                return ""
            parts = shlex.split(raw)
        except Exception:
            return ""

    if not parts:
        return ""

    # Find 'claude' binary in the argv list.
    claude_idx = next((i for i, p in enumerate(parts) if p.endswith("claude")), -1)
    if claude_idx < 0:
        return ""

    tokens = parts[claude_idx + 1:]

    # Walk tokens pairing --flags with their values.
    # Consecutive non-flag tokens are joined as a single value (preserves paths
    # with spaces when the argv source can provide them).
    # Flags/values containing shell metacharacters are dropped to prevent injection.
    _shell_metachars = set(';`$|()')
    cleaned: list[str] = []
    skip_count = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if skip_count > 0:
            skip_count -= 1
            i += 1
            continue

        # Skip resume/continue flags and their session ID argument
        if tok in ("--resume", "--continue", "-c"):
            skip_count = 1
            i += 1
            continue

        # Skip bare UUID-like session ID args
        if len(tok) >= 32 and "-" in tok and not tok.startswith("-"):
            i += 1
            continue

        if tok.startswith("-"):
            # Collect all following non-flag tokens as this flag's value
            j = i + 1
            while j < len(tokens) and not tokens[j].startswith("-"):
                j += 1
            value_tokens = tokens[i + 1:j]
            value = " ".join(value_tokens) if value_tokens else ""

            # Drop flag+value if value contains shell injection metacharacters
            if any(c in _shell_metachars for c in value):
                i = j
                continue

            if value:
                cleaned.append(tok)
                cleaned.append(shlex.quote(value))
            else:
                cleaned.append(tok)
            i = j
        else:
            # Bare non-flag token (shouldn't be common after flag extraction)
            if not any(c in _shell_metachars for c in tok):
                cleaned.append(shlex.quote(tok))
            i += 1

    return " ".join(cleaned)


def _detect_terminal_env() -> str:
    """Detect the terminal environment: 'tmux', 'screen', 'ssh', or 'plain'."""
    if os.environ.get("TMUX"):
        return "tmux"
    if os.environ.get("STY"):
        return "screen"
    if is_ssh_session():
        return "ssh"
    return "plain"


def _wait_for_exit(pid: int, timeout: float = 5.0) -> bool:
    """Wait for a process to exit. Returns True if exited, False if still alive."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
            time.sleep(0.2)
        except (ProcessLookupError, PermissionError, OSError):
            return True
    return False


def _terminate_and_resume(
    claude_pid: int,
    project_dir: str,
    session_id: str | None = None,
    session_path: Path | None = None,
    write_pruned=None,
    **_ignored_kwargs: object,
) -> None:
    """Gracefully exit Claude and resume in the same terminal where possible.

    Priority:
      1. tmux/screen: send-keys "/exit" → wait → send-keys "claude --resume" (same pane)
      2. Plain terminal: SIGTERM → open new terminal with resume
      3. SSH: skip terminate, print manual instructions

    When session_path is supplied, the ps-based identity check in
    _is_claude_process falls back to JSONL mtime recency — matching the
    watchdog's behaviour. Without this, a forked subshell whose argv drops
    the claude-code marker is recognised as alive by the watchdog but
    rejected by this function, silently skipping the reload.

    ``**_ignored_kwargs`` is accepted for forward-compatibility: test harnesses
    and future callers may pass rx_name/config/auto_reload without causing a
    TypeError (blueprint § NEW-1 test compat).
    """
    # Consume the armed-reload sentinel at THE single choke point every reload
    # path funnels through (daemon HARD tiers + OverflowRecovery), so a stale
    # warned=True can't survive into the resumed session (same session_id → same
    # slug) and cause an UNWARNED reload there. clear_armed is best-effort; this
    # is the authoritative clear regardless of which caller initiated the reload.
    clear_armed(session_id, session_path)

    resume_flag = f"--resume {session_id}" if session_id else "--resume"

    # Preserve all CLI flags from the original Claude process
    original_flags = _detect_claude_flags(claude_pid)
    resume_cmd = f"claude {original_flags} {resume_flag}".replace("  ", " ").strip()
    term_env = _detect_terminal_env()
    system = platform.system()

    # PR #94 review MED-1/2/3 fold: sentinel is written ONLY in paths that
    # actually terminate OLD Claude + spawn NEW Claude (tmux, screen, plain
    # terminal post-SIGTERM via _spawn_reload_watcher). SSH paths + PID-reuse
    # early returns do NOT write the sentinel, eliminating the 120s
    # suppression-window UX bug surfaced by reviewer-e2e-pr94 review.

    if term_env == "ssh":
        print(f"  SSH session — skipping terminate+resume. Resume manually: {resume_cmd}")
        return

    # Anti-resurrection entry gate. The reload watcher resumes UNCONDITIONALLY
    # once claude_pid dies (`while kill -0 …; do sleep; done; <resume_cmd>`), so
    # entering here with an already-dead Claude — e.g. the user exited during
    # the prune window — would reopen a session the user closed. The per-block
    # checks below only guard each SIGTERM/SIGKILL, NOT the watcher spawn, so
    # this gate is load-bearing. It returns before any sentinel write too,
    # consistent with "sentinel only on paths that actually terminate+resume."
    #
    # Liveness FIRST, and mtime-IMMUNE: guard_prune_cycle's own save_messages
    # refreshes the JSONL mtime moments before this call, so _is_claude_process's
    # mtime fallback can misreport a dead Claude as alive. os.kill is not fooled.
    if not _pid_is_alive(claude_pid):
        print(f"  PID {claude_pid} is gone — skipping terminate+resume (no resurrection).")
        return
    # Start-time identity gate: if the PID was recycled to a different process
    # after Claude died, the start_time recorded at startup will differ. This
    # closes the residual resurrection vector left by the mtime fallback even
    # after Junaid's mtime-immune liveness gate (06f91c3) — a recycled PID IS
    # alive but is NOT the same Claude. Fails-OPEN when psutil is absent.
    if not _pid_identity_match(claude_pid, session_id):
        print(f"  PID {claude_pid} start-time mismatch — PID was recycled, skipping terminate+resume.")
        return
    # Identity (anti-PID-reuse): is this still actually Claude, not a recycled
    # PID? Per-block checks re-verify before each kill; this is the fail-fast.
    if not _is_claude_process(claude_pid, session_path=session_path):
        print(f"  PID {claude_pid} is no longer a Claude process — skipping terminate+resume.")
        return

    if term_env == "tmux":
        # tmux: graceful /exit via send-keys, then resume in same pane.
        # Verify PID identity before sending keyboard events (PID reuse guard).
        if not _is_claude_process(claude_pid, session_path=session_path):
            print(f"  WARNING: PID {claude_pid} is no longer a Claude process — skipping tmux terminate+resume.")
            return
        # PID check passed — we ARE going to terminate + auto-resume. Write the
        # sentinel BEFORE send-keys so the resumed Claude's SessionStart hook
        # sees it and skips the daemon spawn during the resume window.
        if session_id:
            try:
                write_reload_sentinel(session_id, claude_pid)
            except OSError:
                pass  # best-effort; stale-GC clears any leaked sentinel
        pane = os.environ.get("TMUX_PANE", "")
        target = f"-t {pane}" if pane else ""
        print(f"  tmux detected — sending /exit and auto-resuming in same pane...")

        # Send /exit to Claude
        subprocess.run(
            ["tmux", "send-keys", *(["-t", pane] if pane else []), "/exit", "Enter"],
            capture_output=True, timeout=5,
        )

        # Wait for Claude to exit
        if not _wait_for_exit(claude_pid, timeout=10.0):
            if _is_claude_process(claude_pid, session_path=session_path):
                os.kill(claude_pid, signal.SIGTERM)
            _wait_for_exit(claude_pid, timeout=5.0)

        time.sleep(1)

        # #106: write the pruned session NOW — Claude has exited, so the
        # os.replace can no longer swap an inode out from under a live fd. Gated
        # on confirmed death; if Claude somehow survived, skip the write and let
        # it resume from the untouched (full) file rather than risk corruption.
        if write_pruned is not None and not _pid_is_alive(claude_pid):
            write_pruned()

        # Resume in same pane
        subprocess.run(
            ["tmux", "send-keys", *(["-t", pane] if pane else []),
             f"cd {shell_quote(project_dir)} && {resume_cmd}", "Enter"],
            capture_output=True, timeout=5,
        )
        # tmux resume is synchronous (send-keys returns after command starts).
        # Unlink the sentinel here so the resumed Claude's SessionStart hook
        # can spawn its own guard without suppression.
        if session_id:
            try:
                unlink_reload_sentinel(session_id)
            except OSError:
                pass
        return

    if term_env == "screen":
        # GNU screen: similar to tmux. Verify PID identity before sending keyboard events.
        if not _is_claude_process(claude_pid, session_path=session_path):
            print(f"  WARNING: PID {claude_pid} is no longer a Claude process — skipping screen terminate+resume.")
            return
        # PID check passed — write the sentinel before send-keys (see tmux block).
        if session_id:
            try:
                write_reload_sentinel(session_id, claude_pid)
            except OSError:
                pass
        screen_session = os.environ.get("STY", "")
        print(f"  screen detected — sending /exit and auto-resuming...")

        subprocess.run(
            ["screen", "-S", screen_session, "-X", "stuff", "/exit\n"],
            capture_output=True, timeout=5,
        )

        if not _wait_for_exit(claude_pid, timeout=10.0):
            if _is_claude_process(claude_pid, session_path=session_path):
                os.kill(claude_pid, signal.SIGTERM)
            _wait_for_exit(claude_pid, timeout=5.0)

        time.sleep(1)

        # #106: write the pruned session now that Claude has exited (see tmux note).
        if write_pruned is not None and not _pid_is_alive(claude_pid):
            write_pruned()

        subprocess.run(
            ["screen", "-S", screen_session, "-X", "stuff",
             f"cd {shell_quote(project_dir)} && {resume_cmd}\n"],
            capture_output=True, timeout=5,
        )
        # screen resume is synchronous. Unlink sentinel so the resumed Claude's
        # SessionStart hook can spawn its guard.
        if session_id:
            try:
                unlink_reload_sentinel(session_id)
            except OSError:
                pass
        return

    # Plain terminal — SIGTERM + spawn resume watcher
    try:
        if system == "Windows":
            if _is_claude_process(claude_pid, session_path=session_path):
                subprocess.call(["taskkill", "/PID", str(claude_pid)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            if _is_claude_process(claude_pid, session_path=session_path):
                os.kill(claude_pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass

    if not _wait_for_exit(claude_pid, timeout=5.0):
        try:
            if system == "Windows":
                if _is_claude_process(claude_pid, session_path=session_path):
                    subprocess.call(["taskkill", "/F", "/PID", str(claude_pid)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                if _is_claude_process(claude_pid, session_path=session_path):
                    os.kill(claude_pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    # #106: Claude has been terminated above. Wait briefly for the fd to be
    # released, then write the pruned session BEFORE spawning the resume
    # watcher, so the os.replace never swaps an inode out from under a live
    # Claude. Gated on confirmed death — if Claude somehow survived the kill,
    # skip the write and let it resume from the untouched (full) file.
    if write_pruned is not None:
        _wait_for_exit(claude_pid, timeout=2.0)
        if not _pid_is_alive(claude_pid):
            write_pruned()

    # Plain-terminal path: write sentinel here, JUST BEFORE the watcher Popen.
    # SSH and PID-reuse-fail blocks above return without reaching this point,
    # so they leave no sentinel. The watcher script will unlink the sentinel
    # after osascript fires (NEW Claude SessionStart can spawn freely).
    if session_id:
        try:
            write_reload_sentinel(session_id, claude_pid)
        except OSError:
            pass  # best-effort; stale-GC clears any leaked sentinel

    _spawn_reload_watcher(claude_pid, project_dir, session_id=session_id)


def _spawn_reload_watcher(claude_pid: int, project_dir: str, session_id: str | None = None):
    """Spawn a detached watcher that resumes Claude after exit.

    Extended (Phase B):
    - Unlinks the reload sentinel AFTER osascript fires (NEW-1 option c) so
      the new Claude's SessionStart hook can spawn its guard without suppression.
    - Polls for the new Claude process for RELOAD_WATCHER_POLL_TIMEOUT_SECONDS
      (GAP-B); writes a structured status file to /tmp/cozempic_reload_<sid12>.status
      on timeout, which the next SessionStart hook surfaces to the operator.
    """
    resume_flag = f"--resume {session_id}" if session_id else "--resume"
    original_flags = _detect_claude_flags(claude_pid)
    if original_flags:
        resume_flag = f"{original_flags} {resume_flag}"

    # SSH sessions can't open GUI terminals — skip auto-resume.
    # PR #94 review MED-3: the upstream _terminate_and_resume already wrote
    # the sentinel for the plain-terminal path before calling us. If we early
    # return here (double-SSH-disagree edge: _detect_terminal_env said NOT ssh
    # but is_ssh_session() says yes), the watcher will never fire its unlink.
    # Clean up the sentinel here so the user's manual re-resume isn't suppressed.
    if is_ssh_session():
        print(f"  SSH session detected — skipping auto-resume.")
        print(f"  Resume manually: cd {project_dir} && claude {resume_flag}")
        if session_id:
            try:
                unlink_reload_sentinel(session_id)
            except OSError:
                pass
        return

    system = platform.system()

    # log_dir is a bash-safe representation of project_dir for the echo log line.
    # shell_quote wraps in single quotes (POSIX safe); metachars are not executable.
    log_dir = shell_quote(project_dir)

    # Compute sentinel + status paths at generation time so bash script is
    # self-contained (no Python dependency inside the watcher).
    # The slug uses reload_lock._slug_for so it matches _reload_sentinel_path_for.
    from .reload_lock import _slug_for as _rl_slug_for
    if session_id:
        sid12 = _rl_slug_for(session_id)[:12]
        sentinel_path = f"/tmp/cozempic_reload_{sid12}.in-flight"
        status_path = f"/tmp/cozempic_reload_{sid12}.status"
        pgrep_pattern = f"claude.*{sid12}"
    else:
        sid12 = ""
        sentinel_path = ""
        status_path = "/dev/null"
        pgrep_pattern = "claude"

    if system == "Darwin":
        resume_cmd = (
            f"osascript -e 'tell application \"Terminal\" to do script "
            f"\"cd {shell_quote(project_dir)} && claude {resume_flag}\"'"
        )
    elif system == "Linux":
        resume_cmd = (
            f"if command -v gnome-terminal >/dev/null 2>&1; then "
            f"gnome-terminal -- bash -c 'cd {shell_quote(project_dir)} && claude {resume_flag}; exec bash'; "
            f"elif command -v xterm >/dev/null 2>&1; then "
            f"xterm -e 'cd {shell_quote(project_dir)} && claude {resume_flag}' & "
            f"else echo 'No terminal emulator found' >> /tmp/cozempic_guard.log; fi"
        )
    elif system == "Windows":
        # Windows has no bash, and the POSIX watcher below uses kill -0 / pgrep /
        # date +%s / /tmp — so the old code (which built a cmd.exe resume_cmd and
        # then ran it via Popen(["bash", ...])) was DEAD on Windows: auto-resume
        # never fired. Build a PowerShell-native watcher instead and spawn it via
        # powershell (the dedicated Windows branch at the spawn site below).
        # Escape cmd.exe metacharacters in project_dir so they cannot execute when
        # cmd.exe runs the `cd /d <dir> && claude` line.
        _cmd_metachars = set('&|<>^"')
        escaped_dir = "".join(f"^{c}" if c in _cmd_metachars else c for c in project_dir)
        _win_resume_inner = f"cd /d {escaped_dir} && claude {resume_flag}"
        _win_status = status_path
        if status_path == "/dev/null" or status_path.startswith("/tmp"):
            # /tmp / /dev/null don't exist on Windows — use the Windows temp dir.
            _win_status = str(Path(tempfile.gettempdir()) / (f"cozempic_reload_{sid12}.status" if sid12 else "cozempic_reload.status"))
        _win_sentinel = sentinel_path
        if sentinel_path.startswith("/tmp"):
            _win_sentinel = str(Path(tempfile.gettempdir()) / f"cozempic_reload_{sid12}.in-flight")
        _win_log = str(Path(tempfile.gettempdir()) / "cozempic_guard.log")

        def _ps_q(s: str) -> str:  # PowerShell single-quote literal (double internal ')
            return "'" + s.replace("'", "''") + "'"

        # SESSION-SCOPED success poll (ynaamane review, LOW): mirror the POSIX
        # `pgrep -f 'claude.*<sid12>'` so a DIFFERENT session's Claude on a
        # multi-session Windows host is not misread as OUR successful resume.
        # Get-Process has no CommandLine, so use Get-CimInstance Win32_Process and
        # match the session id on the command line. sid12 is sanitized to
        # [a-z0-9_-] so it is safe inside a -like glob. Fall back to any-claude only
        # when the session id is unknown.
        if sid12:
            _ps_find_new = (
                "$new=Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | "
                f"Where-Object {{ $_.CommandLine -like '*claude*' -and $_.CommandLine -like '*{sid12}*' }} | "
                "Select-Object -First 1"
            )
        else:
            _ps_find_new = "$new=Get-Process -Name claude -ErrorAction SilentlyContinue | Select-Object -First 1"

        windows_ps_script = (
            "$ErrorActionPreference='SilentlyContinue'; "
            # Phase 1: wait for the old Claude to exit.
            f"while (Get-Process -Id {int(claude_pid)} -ErrorAction SilentlyContinue) {{ Start-Sleep -Seconds 1 }}; "
            "Start-Sleep -Seconds 1; "
            # Phase 2: open a new window and resume (cmd.exe runs the cd && claude).
            f"Start-Process -FilePath 'cmd.exe' -ArgumentList '/c',{_ps_q(_win_resume_inner)}; "
            # Phase 3: drop the in-flight sentinel so the new session's guard can spawn.
            + (f"Remove-Item -Force -ErrorAction SilentlyContinue {_ps_q(_win_sentinel)}; " if _win_sentinel else "")
            # Phase 4: poll for the new claude; log on success, write status on timeout.
            + f"$deadline=(Get-Date).AddSeconds({RELOAD_WATCHER_POLL_TIMEOUT_SECONDS}); $new=$null; "
            f"while ((Get-Date) -lt $deadline) {{ {_ps_find_new}; if ($new) {{ break }}; Start-Sleep -Seconds {RELOAD_WATCHER_POLL_INTERVAL_SECONDS} }}; "
            f"if ($new) {{ Add-Content -Path {_ps_q(_win_log)} -Value \"$(Get-Date): Cozempic guard resumed Claude (new PID $($new.Id))\" }} "
            f"else {{ Set-Content -Path {_ps_q(_win_status)} -Value \"failed`n$(Get-Date -Format o)`nnew Claude did not start within {RELOAD_WATCHER_POLL_TIMEOUT_SECONDS}s`ninvestigate: claude on PATH / auth / JSONL path\" }}"
        )
    else:
        print(f"  WARNING: Auto-resume not supported on {system}.")
        # MED-3 fold: upstream wrote sentinel for plain path before calling us.
        # Unsupported OS = no watcher spawn = no unlink fire. Clean up here.
        if session_id:
            try:
                unlink_reload_sentinel(session_id)
            except OSError:
                pass
        return

    # Compose the sentinel unlink fragment (empty string when no session_id).
    # The bash watcher_script below is POSIX-only; Windows uses windows_ps_script
    # (built in the Windows branch above) — guard the assembly so we don't
    # reference the now-Windows-undefined `resume_cmd`/`log_dir` on Windows.
    watcher_script = None
    if system != "Windows":
      _sentinel_unlink = f"rm -f '{sentinel_path}'; " if sentinel_path else ""
      watcher_script = (
        # Phase 1: wait for old Claude to exit
        f"while kill -0 {int(claude_pid)} 2>/dev/null; do sleep 1; done; "
        f"sleep 1; "
        # Phase 2: fire the resume command (osascript / gnome-terminal / etc)
        f"{resume_cmd}; "
        f"RESUME_EXIT=$?; "
        # Phase 3: unlink sentinel AFTER osascript so the new Claude's SessionStart
        # can spawn its own guard (sentinel no longer suppresses spawn).
        f"{_sentinel_unlink}"
        # Phase 4 (GAP-B): poll for the new claude process for up to
        # RELOAD_WATCHER_POLL_TIMEOUT_SECONDS. On success: log the new PID.
        # On timeout: write a structured status file for the next SessionStart.
        f"deadline=$(( $(date +%s) + {RELOAD_WATCHER_POLL_TIMEOUT_SECONDS} )); "
        f"new_pid=''; "
        f"while [ $(date +%s) -lt $deadline ]; do "
        f"  new_pid=$(pgrep -f '{pgrep_pattern}' 2>/dev/null | head -n 1); "
        f"  [ -n \"$new_pid\" ] && break; "
        f"  sleep {RELOAD_WATCHER_POLL_INTERVAL_SECONDS}; "
        f"done; "
        f"if [ -n \"$new_pid\" ]; then "
        f"  echo \"$(date): Cozempic guard resumed Claude in {log_dir} (new PID $new_pid)\" >> /tmp/cozempic_guard.log; "
        f"else "
        f"  printf '%s\\n%s\\n%s\\n%s\\n' 'failed' "
        f"    \"$(date -Iseconds 2>/dev/null || date)\" "
        f"    \"new Claude did not start within {RELOAD_WATCHER_POLL_TIMEOUT_SECONDS}s after resume_cmd (exit=$RESUME_EXIT)\" "
        f"    'investigate: Terminal automation permission / claude -r auth / JSONL path / network' "
        f"    > '{status_path}'; "
        f"  echo \"$(date): Cozempic guard reload FAILED — no new Claude after {RELOAD_WATCHER_POLL_TIMEOUT_SECONDS}s\" >> /tmp/cozempic_guard.log; "
        f"fi"
    )

    if system == "Windows":
        # PowerShell-native watcher (the bash watcher_script above is POSIX-only
        # and would never run on Windows). DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP
        # so it outlives this process; no start_new_session (POSIX-only kwarg).
        _flags = 0
        _flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
        _flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", windows_ps_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=_flags,
        )
    else:
        subprocess.Popen(
            ["bash", "-c", watcher_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )


# Session-id validation for pidfile path composition.
# Round-3 / DA C2 fix (Option B per team-lead + code-auditor): accepts
# lowercase alphanumeric + underscore + dash, matching the SessionStart
# hook bash sanitiser (`re.sub(r'[^a-z0-9_-]', '_', s.lower())`) and
# ``reload_lock._slug_for`` / ``spawn_lock._slug_for`` (both use
# ``[^a-zA-Z0-9_-]`` as their substitution character class). UUIDs are a
# strict subset, so no regression for existing inputs. The first char must
# be alphanumeric (not ``_`` or ``-``) — preserves the dash-collision
# security property pinned by ``TestPolishV2_SessionIdRegexRequiresHexFirstChar``
# in test_guard_hardening.py (pure-dash and leading-dash inputs would
# otherwise collide after [:12] truncation onto the same pidfile path).
# 12+ chars keeps the ``[:12]`` truncation meaningful and prevents
# zero-byte slug paths.
# Note: ``_pid_file_for_session`` lowercases session_id BEFORE matching,
# so the regex intentionally accepts lowercase only (not an RFC-4122
# uppercase bug — uppercase UUIDs are normalized first).
_SESSION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{11,}$")


def _guard_tmp_root() -> Path:
    """Directory for guard PID/log files.

    POSIX keeps the historical ``/tmp`` so the path stays byte-identical to the
    SessionStart shell hook (which hardcodes ``/tmp/cozempic_guard_*.pid`` and
    cannot call ``tempfile.gettempdir()``); diverging would make the hook's
    "guard already running" fast-path always miss on macOS, where gettempdir()
    is ``/var/folders/.../T``. Windows has no ``/tmp`` — a literal
    ``Path("/tmp")`` resolves to ``C:\\tmp`` which is not guaranteed to exist,
    raising FileNotFoundError during daemon spawn — so use the platform tempdir
    there.
    """
    if os.name == "nt":
        return Path(tempfile.gettempdir())
    return Path("/tmp")


# ── Cross-respawn reload-rate ledger (regrow-loop / reload-storm backstop) ────
# A confirmed prune that frees real bytes but leaves the session destined to
# re-hit the HARD threshold reloads, exits the daemon, and a fresh guard
# respawns with the in-process circuit breaker reborn at 0 — so a session that
# keeps re-bloating reloads indefinitely, one confirmed prune+ping per cycle,
# invisible to the per-process breaker. This DISK-backed ledger survives the
# respawn: if a session reloads too many times within a window, the guard stops
# auto-reloading it (and accounts it to the breaker so the daemon exits) rather
# than churning kill→resume→re-bloat forever. Window/cap are env-overridable.
def _reload_ledger_window_s() -> int:
    try:
        return max(60, int(os.environ.get("COZEMPIC_RELOAD_WINDOW_S", "600")))
    except Exception:
        return 600


def _reload_ledger_max() -> int:
    try:
        return max(1, int(os.environ.get("COZEMPIC_RELOAD_MAX", "3")))
    except Exception:
        return 3


# ── In-flight work detector (1.8.22 safe-point gate, component A) ─────────────
# A guard reload SIGKILLs the Claude process and resumes from the pruned
# transcript. Harness-side state that is NOT in the transcript — a running
# Workflow-tool orchestration, a background subagent — is destroyed with no
# recovery. cozempic could not see these before. This detects them from the
# transcript via launch/completion markers (both DO appear in the JSONL):
#   - Workflow launch:  "Workflow launched in background. Task ID: <id>"
#   - background launch: "running in background with ID: <id>"
#   - completion:        <task-notification>...<task-id><id></task-id>...<status>completed|failed</status>
# plus any open synchronous tool_use with no matching tool_result (a tool/subagent
# mid-call). Conservative by construction: it flags only on positive
# launch-without-completion or a genuinely open call, so it does not over-defer.
# Launch markers — VERIFIED against real Claude Code transcripts (2026-06-07,
# 195 Agent / 12 Workflow / 0 Task tool_uses across 13 sessions). The dominant
# real path is the `Agent` tool's "Async agent launched successfully. agentId: X"
# (background subagent) — the original detector matched only the Workflow/Task-ID
# phrasing, which is NOT what the harness emits, so a running background subagent
# was invisible and a reload would SIGKILL it. All three launch ids complete via
# the SAME-namespace <task-notification><task-id>X</task-id> block, so
# launched−completed accounting works once each launch phrasing is recognized.
# Patterns are drift-tolerant (optional "the"/punctuation, Task|Run ID, ID:/=/( ).
_AGENT_LAUNCH_RE = re.compile(r"Async agent launched successfully\.?\s*agentId:\s*([A-Za-z0-9_-]+)", re.IGNORECASE)
_WF_LAUNCH_RE = re.compile(r"[Ww]orkflow launched in (?:the )?background[.,]?\s*(?:Task|Run) ID:\s*([A-Za-z0-9_-]+)", re.IGNORECASE)
_BG_LAUNCH_RE = re.compile(r"running in (?:the )?background(?: with| \()?\s*ID[:=]?\s*([A-Za-z0-9_-]+)", re.IGNORECASE)
_TN_BLOCK_RE = re.compile(r"<task-notification(?:\s[^>]*)?>(.*?)</task-notification>", re.DOTALL | re.IGNORECASE)
_TN_ID_RE = re.compile(r"<task-id>([^<]+)</task-id>", re.IGNORECASE)
_TN_STATUS_RE = re.compile(r"<status>([^<]+)</status>", re.IGNORECASE)
# Terminal completion vocabulary — broadened so a harness phrasing skew (success/
# done/finished vs completed) can't pin a finished task "in-flight" forever.
_INFLIGHT_DONE = {"completed", "complete", "failed", "cancelled", "canceled",
                  "stopped", "killed", "error", "success", "succeeded", "done",
                  "ok", "finished", "aborted", "timeout", "timed_out"}
# Terminal (finished) statuses. The safe-point gate is a DENYLIST: anything NOT
# terminal is treated as still-executing → block the reload. This fails SAFE on
# unrecognized/off-vocabulary working statuses (e.g. the hyphen variant
# "in-progress", or "busy"/"waiting"/"executing") which an active-allowlist would
# have let through and destroyed.
_STATUS_TERMINAL = {"completed", "complete", "done", "failed", "cancelled",
                    "canceled", "stopped", "killed", "aborted", "error",
                    "success", "succeeded", "finished", "timeout", "timed_out",
                    "ok"}  # kept in sync with _INFLIGHT_DONE (same "finished" concept)
# Benign teammate membership markers that are NOT "actively working" and must not
# wedge the gate (a teammate legitimately sits in these between tasks). Kept
# minimal/conservative — anything not here AND not terminal blocks.
_TEAMMATE_BENIGN = {"", "config", "idle", "unknown"}


def _msg_dict(item) -> dict:
    """Accept either a (idx, dict, size) tuple or a raw message dict."""
    if isinstance(item, tuple):
        return item[1] if len(item) > 1 and isinstance(item[1], dict) else {}
    return item if isinstance(item, dict) else {}


def _block_text(b: dict) -> str:
    """Text of a single content block (tool_result content as str or list-of-{text})."""
    parts = []
    if isinstance(b.get("text"), str):
        parts.append(b["text"])
    rc = b.get("content")
    if isinstance(rc, str):
        parts.append(rc)
    elif isinstance(rc, list):
        for sub in rc:
            if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                parts.append(sub["text"])
    return "\n".join(parts)


def _completion_text(msg: dict) -> str:
    """Text from a message's GENUINE harness-delivery surfaces only — the root
    `content` string (queue-operation notifications, verified to be where real
    task-notifications land) and a user message's top-level string content. It
    deliberately EXCLUDES tool_result blocks and assistant text blocks, so a
    <task-notification> merely quoted/echoed inside some tool's output or the
    model's prose cannot CLEAR a genuinely in-flight launch (a false-negative →
    SIGKILL). Mirror of the launch side's tool-type correlation."""
    parts = []
    root = msg.get("content")
    if isinstance(root, str):
        parts.append(root)
    c = (msg.get("message") or {}).get("content")
    if isinstance(c, str):
        parts.append(c)
    return "\n".join(parts)


# Which tool a launch marker must be paired with to count as a REAL launch (vs a
# marker-shaped string merely quoted/echoed inside some other tool's output, or
# written as prose). Keyed by marker kind.
_LAUNCH_TOOLS = {"wf": ("Workflow",), "agent": ("Agent", "Task"), "bg": ("Bash",)}


def detect_in_flight(messages) -> dict:
    """Detect harness-side in-flight work the transcript reload would destroy.

    Returns {workflow, background, agent, open_call: bool, ids: [...]}.
    workflow/background/agent = launched-but-not-completed background tasks
    (Workflow tool, run_in_background Bash, and the `Agent` tool respectively);
    open_call = a tool_use with no matching tool_result (synchronous mid-call).
    Ids are compared case-insensitively so a casing skew can't strand a launch.

    A launch marker only counts inside a tool_result whose paired tool_use is of
    the matching type — so a marker-shaped string quoted in another tool's output
    or in the model's prose can't fabricate a PHANTOM in-flight task that wedges
    the gate (verified against a live workflow run). A result whose tool_use was
    pruned away is credited conservatively, so a REAL launch is never missed (the
    catastrophic direction). Completions are matched broadly — they only ever
    CLEAR an id, so a quoted one is harmless.
    """
    launched_wf: set[str] = set()
    launched_bg: set[str] = set()
    launched_agent: set[str] = set()
    completed: set[str] = set()
    use_ids: set[str] = set()
    res_ids: set[str] = set()
    use_name: dict = {}            # tool_use_id -> tool name
    bg_bash_ids: set[str] = set()  # ids of Bash tool_uses with run_in_background=true
    results: list = []            # (tool_use_id|None, text) per tool_result
    open_unkeyed = False
    for item in messages or []:
        msg = _msg_dict(item)
        if not msg:
            continue
        text = _completion_text(msg)   # genuine deliveries only (not quoted/echoed)
        if text:
            for blk in _TN_BLOCK_RE.findall(text):
                ids = _TN_ID_RE.findall(blk)
                sts = _TN_STATUS_RE.findall(blk)
                if ids and sts and sts[-1].strip().lower() in _INFLIGHT_DONE:
                    completed.add(ids[0].strip().lower())
        inner = (msg.get("message") or {})
        c = inner.get("content")
        if isinstance(c, list):
            for b in c:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "tool_use":
                    bid = hashable_str(b.get("id"))  # unhashable id -> "" (R6 crash class)
                    if bid:
                        use_ids.add(bid)
                        use_name[bid] = hashable_str(b.get("name"))
                        # Only a Bash with run_in_background=true actually launches a
                        # bg task; a normal Bash whose OUTPUT merely contains the
                        # ack-marker text (a test/grep printing "running in background
                        # with ID: X") must NOT be credited as a launch (real-transcript
                        # pollution, 2026-06-09).
                        if (hashable_str(b.get("name")) == "Bash"
                                and isinstance(b.get("input"), dict)
                                and b["input"].get("run_in_background")):
                            bg_bash_ids.add(bid)
                    else:
                        # A tool_use with no id can never be paired to a result —
                        # fail toward "open" rather than silently treat it closed.
                        open_unkeyed = True
                elif t == "tool_result":
                    tid = hashable_str(b.get("tool_use_id"))
                    if tid:
                        res_ids.add(tid)
                    results.append((tid, _block_text(b)))

    def _ok(tid, names):
        nm = use_name.get(tid)
        # unknown (pruned tool_use) → conservative credit; match the tool name
        # case-insensitively so a harness casing drift ("workflow" vs "Workflow")
        # can't strand a real launch into a missed-launch SIGKILL.
        return nm is None or nm.strip().lower() in {x.lower() for x in names}

    for tid, rtext in results:
        if not rtext:
            continue
        # A FOREGROUND-completed Agent result is the agent's OUTPUT (prose), not a
        # launch ack — skip it, so a launch marker QUOTED in an agent's text (agents
        # that discuss/echo "Async agent launched ... agentId: X", e.g. when working
        # ON cozempic) can't fabricate a PHANTOM in-flight task that wedges the gate
        # (real-transcript pollution, 2026-06-09).
        if _AGENT_DONE_TRAILER_RE.search(rtext):
            continue
        if _ok(tid, _LAUNCH_TOOLS["wf"]):
            for m in _WF_LAUNCH_RE.findall(rtext):
                launched_wf.add(m.strip().lower())
        if _ok(tid, _LAUNCH_TOOLS["agent"]):
            for m in _AGENT_LAUNCH_RE.findall(rtext):
                launched_agent.add(m.strip().lower())
        # bg launch credited only from a result whose paired tool_use is a
        # run_in_background Bash (or a pruned/unknown tool_use → conservative).
        if _ok(tid, _LAUNCH_TOOLS["bg"]) and (use_name.get(tid) is None or tid in bg_bash_ids):
            for m in _BG_LAUNCH_RE.findall(rtext):
                launched_bg.add(m.strip().lower())
    wf = launched_wf - completed
    bg = launched_bg - completed
    agent = launched_agent - completed
    open_call = bool(use_ids - res_ids) or open_unkeyed
    return {
        "workflow": bool(wf),
        "background": bool(bg),
        "agent": bool(agent),
        "open_call": open_call,
        "ids": sorted(wf | bg | agent),
    }


# Team-MEMBER coordination tool names (exclude the shared-list Task* + read-only
# TeamStatus/Get/List). A tool_use with one of THESE names is unambiguous team
# coordination (it is a tool NAME, not free text), so its presence with an empty
# roster = a parse gap → block.
_TEAM_COORD_TOOLS = {"TeamCreate", "SpawnTeammate", "SendMessage", "TeamMessage"}
# Tools a spawn marker must be PAIRED with (via tool_use_id) to count — exactly the
# correlation detect_in_flight uses. A marker-shaped string in a Read/Grep/Bash/cat
# result (source code, YAML, .env, structured logs that contain `agent_id:` etc.)
# must NEVER count, or the gate over-blocks teamless sessions and the guard goes
# inert (fleet P0, 2026-06-09).
_TEAM_SPAWN_TOOLS = {"Agent", "Task", "TeamCreate", "SpawnTeammate"}
# Agent/team spawn markers — snake AND camelCase id, both spawn phrasings, and the
# 1.8.22 background launch. Only ever consulted inside a paired-spawn-tool result.
_TEAM_SPAWN_MARKER_RE = re.compile(
    r"agent_?id\s*[:=]|Spawned successfully|Async agent launched", re.IGNORECASE)
# The STRUCTURAL teammate-message carrier the harness emits (and extract_team_state
# parses) — `<teammate-message ... teammate_id="...">`. Matched as a tag, NOT as a
# bare "teammate-message"/"idle_notification" substring, so prose merely *discussing*
# the protocol (or cozempic's own source/docs) does not trip it (fleet P0).
_TEAMMATE_MSG_MARKER_RE = re.compile(
    r'<teammate-message\s[^>]*teammate_id\s*=\s*"', re.IGNORECASE)


def _unresolved_team_coordination(messages, team_state) -> bool:
    """Deny-by-default net (1.8.24, lens L0). True when the transcript shows team
    coordination we could NOT resolve to a roster, so the gate blocks instead of
    SIGKILLing a team whose markers drifted. Fires ONLY on an empty roster (a
    normally-parsed live OR finished team is unaffected) and ONLY on signals that
    cannot be fabricated by arbitrary message text:
      * a `_TEAM_COORD_TOOLS` tool_use (an unambiguous tool NAME), or
      * a spawn marker inside a tool_result whose PAIRED tool_use is a spawn tool
        (correlated like detect_in_flight — a marker in a Read/Grep/log result does
        NOT count), or
      * a STRUCTURAL `<teammate-message teammate_id="...">` carrier (not a bare
        substring — prose discussing the protocol does not count).
    """
    # getattr-with-default (not direct attribute access): teammates/subagents are
    # default_factory dataclass fields — present on instances but NOT on the class,
    # so a MagicMock(spec=TeamState) raises AttributeError on direct access. getattr
    # swallows that and reads the real lists on a real TeamState.
    if team_state is not None and (
            getattr(team_state, "teammates", None) or getattr(team_state, "subagents", None)):
        return False  # roster parsed → the explicit teammate/subagent checks govern
    use_name: dict = {}        # tool_use_id -> tool name
    results: list = []         # (tool_use_id|None, text) per tool_result
    for item in messages or []:
        msg = _msg_dict(item)
        # Structural teammate-message carrier — checked ONLY on the harness's
        # synthetic delivery surface (root / queue-operation `content`), NOT a
        # user's typed `message.content`. A user PASTING a teammate-message line
        # into a teamless session must not wedge the guard (fleet P1, 2026-06-09);
        # and in cozempic's real pre-prune gating a genuine team's spawn tool_use
        # is still present → non-empty roster → this net is skipped anyway, so
        # narrowing the surface loses no real-team coverage.
        _root = msg.get("content")
        if isinstance(_root, str) and _TEAMMATE_MSG_MARKER_RE.search(_root):
            return True
        c = (msg.get("message") or {}).get("content")
        if isinstance(c, list):
            for b in c:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "tool_use":
                    if hashable_str(b.get("name")) in _TEAM_COORD_TOOLS:
                        return True
                    bid = hashable_str(b.get("id"))  # unhashable id -> "" (R6 crash class)
                    if bid:
                        use_name[bid] = hashable_str(b.get("name"))
                elif t == "tool_result":
                    results.append((hashable_str(b.get("tool_use_id")), _block_text(b)))
    # Spawn marker — credited ONLY when its paired tool_use is a spawn tool we can
    # SEE. A pruned/unknown tool_use is NOT credited here (that would re-open the
    # teamless-session over-block); the roster + detect_in_flight remain the primary
    # signals, so this net staying conservative-but-not-paranoid is the right trade.
    for tid, rtext in results:
        nm = use_name.get(tid)
        if (nm is not None and nm.strip() in _TEAM_SPAWN_TOOLS
                and rtext and _TEAM_SPAWN_MARKER_RE.search(rtext)):
            return True
    return False


def safe_to_reload(team_state, messages, session_path) -> tuple[bool, str]:
    """Validate-BEFORE-terminate safe-point gate (1.8.22 component B).

    A reload terminates Claude and resumes from the pruned transcript. It is only
    safe when nothing in-flight would be destroyed. Returns (safe, reason).
    Conservative: any positive in-flight signal ⇒ unsafe ⇒ caller defers
    (read-only checkpoint + warn), never force-terminates. Never blocks a
    genuinely-quiescent reload (avoids re-creating the inert guard).
    """
    inflight = detect_in_flight(messages)
    # Harness-side in-flight work — the gap cozempic was blind to. These are the
    # MOST reliable signals (launched−completed via task-notifications, verified
    # against real transcripts). NEVER reload through them. Checked first so an
    # active background subagent is caught even if TeamState is empty/stale.
    if inflight["workflow"]:
        return (False, "Workflow orchestrating")
    if inflight["background"]:
        return (False, "background task in flight")
    if inflight["agent"]:
        return (False, "background subagent running")
    if inflight["open_call"]:
        return (False, "open tool call (result not yet flushed)")
    # Tracked team quiescence (TeamState). We DO hard-block on a non-benign
    # teammate status: an Agent-spawned agent team (the guard's primary use case)
    # is NOT reliably visible via subagent entries / open calls / un-completed
    # launches — those markers fire for background subagents, not for `Agent`-tool
    # teammates — so the teammate roster is the only signal that catches it, and
    # this block is load-bearing (closing the F1 destroy-active-team gap). The
    # extractor transitions a teammate to a terminal/idle (benign) status via
    # task-notification + idle_notification (chronology-aware), and the
    # idle-notification carrier is prune-protected (_is_team_message), so the
    # clear signal survives a prune and the block no longer wedges. Subagent
    # entries, active todo tasks, and the in-flight detector above remain
    # complementary signals checked alongside it.
    if team_state is not None and not team_state.is_empty():
        try:
            # Subagent block — DENYLIST: any non-terminal status is treated as
            # still-executing (subagent entries are reliably updated to a terminal
            # status by task-notification completions, so a finished one clears).
            if any((s.status or "").strip().lower() not in _STATUS_TERMINAL
                   for s in (team_state.subagents or [])):
                return (False, "subagent mid-execution")
            # Teammate block — DENYLIST with a benign-marker exempt set so an
            # idle/config membership row never wedges the gate.
            #
            # Safety invariant (replaces the removed P0-E _team_is_current_session
            # gate, 2026-06-08):
            #   A non-benign teammate status ("running") can ONLY originate from
            #   THIS session's JSONL (Agent spawn / TeamCreate-inline / SendMessage).
            #   merge_config_into_state hardcodes status="config" (∈ _TEAMMATE_BENIGN)
            #   for members added from config.json — config-only members are always
            #   benign and never block. Therefore the block can fire unconditionally
            #   on any non-benign status without risk of a stale-config false-block.
            #
            #   The P0-E session-ID gate was removed because it was redundant (the
            #   status mechanism is sufficient) AND could MISFIRE: a stale same-name
            #   config.json could inject a different leadSessionId, causing the gate
            #   to skip the block for a LIVE "running" teammate → false-safe SIGKILL.
            if any((t.status or "").strip().lower()
                   not in (_STATUS_TERMINAL | _TEAMMATE_BENIGN)
                   for t in (team_state.teammates or [])):
                return (False, "teammate mid-execution")
            active_tasks, _, _ = team_state._task_groups()
            if active_tasks:
                return (False, "active task in flight")
        except Exception:
            # Malformed state → be conservative.
            return (False, "team state unreadable")
    # Deny-by-default net (1.8.24): if the transcript shows team/agent coordination
    # we could NOT resolve to a roster (an unrecognized or drifted marker shape —
    # the class behind the 1.8.22 Agent-marker + #117 team blindness), BLOCK rather
    # than SIGKILL. Fires only when the roster is empty yet coordination is present,
    # so a normally-parsed live/finished team is unaffected; only the can't-parse-it
    # case errs to safe. A reload gate must fail toward "block", never "SIGKILL".
    if _unresolved_team_coordination(messages, team_state):
        return (False, "unresolved team coordination (unparsed — failing safe)")
    return (True, "quiescent")


# ── Armed-reload sentinel (1.8.22 components D+E) ────────────────────────────
# When the daemon detects a HARD threshold crossed mid-turn it ARMS a reload by
# writing this sentinel (carrying the projected reduction % when known); the
# Stop-hook nudge READS it to enrich the warning and marks it `warned`. The
# warning-before-reload guarantee comes from the NUDGE itself, which fires at the
# turn-end (Stop hook) that crosses the tier — BEFORE the daemon's next idle poll
# can execute the reload. The execute decision gates on safe_to_reload AND
# SUSTAINED idle (it does not hard-read `warned`, so a missing/disabled Stop hook
# can't wedge it; the trade-off is the warning is best-effort, not a hard
# precondition). `cozempic reload` clears the sentinel (user took control).
# Shared by guard (write) and cli `nudge` (read/warn).
def _reload_armed_path(session_id: str | None, session_path: Path | None = None) -> Path:
    raw = (session_id or (session_path.stem if session_path else None) or "session")
    slug = re.sub(r"[^a-z0-9_-]", "_", str(raw).lower())[:12] or "session"
    return _guard_tmp_root() / f"cozempic_reload_armed_{slug}.json"


def read_armed(session_id: str | None, session_path: Path | None = None) -> dict | None:
    import json as _json
    try:
        p = _reload_armed_path(session_id, session_path)
        return _json.loads(p.read_text()) if p.exists() else None
    except Exception:
        return None


def _write_armed_atomic(path: Path, data: dict) -> None:
    """Atomically write the armed sentinel (temp + os.replace) so the daemon and
    the nudge process — which both write it — can't tear each other's writes. The
    temp is always cleaned up (no .tmp orphan on a write/replace failure)."""
    import json as _json
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    try:
        tmp.write_text(_json.dumps(data))
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)  # no-op after a successful replace
        except Exception:
            pass


def write_armed(session_id, session_path, tier: int, projected_pct: float) -> None:
    import time as _time
    try:
        p = _reload_armed_path(session_id, session_path)
        existing = read_armed(session_id, session_path) or {}
        # `warned` is STICKY: once the user has been warned of a queued reload it
        # stays warned until the reload consumes the sentinel (clear_armed) —
        # escalating the tier (55→80) or the daemon re-arming must NOT un-warn
        # them (the bug: a tier-0 upsert from the nudge + a tier-80 re-arm dropped
        # warned). Preserve the grace clock (armed_at) so re-arming never resets
        # the warned-before-reload timeout; keep a known projection if none given.
        warned = bool(existing.get("warned"))
        armed_at = existing.get("armed_at") or _time.time()
        proj = round(projected_pct, 1) if projected_pct else existing.get("projected_pct", 0.0)
        _write_armed_atomic(p, {"tier": tier, "projected_pct": proj,
                                "warned": warned, "armed_at": armed_at})
    except Exception:
        pass


def mark_armed_warned(session_id, session_path: Path | None = None) -> None:
    """Mark the armed reload as warned. UPSERTS — if no sentinel exists yet (the
    nudge fired before the daemon's poll armed it), create one with warned=True so
    the warning can't be lost to that race."""
    import time as _time
    try:
        p = _reload_armed_path(session_id, session_path)
        d = read_armed(session_id, session_path) or {}
        d["warned"] = True
        d.setdefault("armed_at", _time.time())
        d.setdefault("tier", 0)
        d.setdefault("projected_pct", 0.0)
        _write_armed_atomic(p, d)
    except Exception:
        pass


def clear_armed(session_id, session_path: Path | None = None) -> None:
    p = _reload_armed_path(session_id, session_path)
    try:
        p.unlink(missing_ok=True)
    except Exception:
        # Couldn't remove it — NEUTRALIZE it instead, so a survivor can't carry a
        # stale warned=True / armed_at into the resumed session and trigger an
        # unwarned reload (warned=False + no armed_at → the gate re-arms fresh).
        try:
            _write_armed_atomic(p, {"warned": False})
        except Exception:
            pass


def _reload_ledger_path(session_id: str | None, session_path: Path) -> Path:
    raw = (session_id or session_path.stem or "session")
    slug = re.sub(r"[^a-z0-9_-]", "_", raw.lower())[:12] or "session"
    return _guard_tmp_root() / f"cozempic_reload_{slug}.history"


def _reload_rate_exceeded(ledger_path: Path, now: float | None = None) -> tuple[bool, int]:
    """Return (exceeded, count_in_window) for the per-session reload ledger.

    Prunes entries older than the window. If the in-window count has reached the
    cap, returns (True, count) WITHOUT recording — the caller must skip the
    reload. Otherwise records ``now`` and returns (False, count_after_record).
    Best-effort: any IO/JSON error degrades to (False, 0) so the ledger can
    never block a legitimate reload on a transient error.
    """
    import time as _time
    import json as _json
    if now is None:
        now = _time.time()
    window = _reload_ledger_window_s()
    try:
        hist = _json.loads(ledger_path.read_text()) if ledger_path.exists() else []
        if not isinstance(hist, list):
            hist = []
    except Exception:
        hist = []
    hist = [t for t in hist if isinstance(t, (int, float)) and 0 <= now - t < window]
    if len(hist) >= _reload_ledger_max():
        return True, len(hist)
    hist.append(now)
    try:
        ledger_path.write_text(_json.dumps(hist[-50:]))
    except Exception:
        pass
    return False, len(hist)


def _pid_file_for_session(session_id: str) -> Path:
    """Return the PID file path for a guard daemon watching a specific session.

    Validates ``session_id`` against a relaxed alphanumeric+_- regex (matches
    the bash hook sanitiser and reload_lock/spawn_lock slug rules — codebase
    consistency, fix for DA round-1 C2 finding). Leading char must be
    alphanumeric to prevent dash-collision after ``[:12]`` truncation
    (security property — see ``TestPolishV2_SessionIdRegexRequiresHexFirstChar``).
    Normalizes to lowercase BEFORE matching so different-case variants of
    the same UUID map to the same pidfile (prevents split-brain spawning).
    Raises ValueError on malformed input so callers fail fast; library-API
    callers like ``_is_guard_running_for_session`` catch and return None
    (treat invalid session as "no daemon"). Error message logs only type
    and length — never raw content — to avoid PII leaks.
    """
    session_id = _normalize_session_id(session_id).lower()
    if not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError(
            f"session_id must be alphanumeric+_- (leading-alphanumeric, >=12 chars), "
            f"got {type(session_id).__name__} of length {len(session_id)}"
        )
    return _guard_tmp_root() / f"cozempic_guard_{session_id[:12]}.pid"


def _pid_file_for_cwd(cwd: str) -> Path:
    """Legacy: PID file keyed by CWD hash. Used for migration cleanup only."""
    import hashlib
    slug = hashlib.md5(cwd.encode()).hexdigest()[:12]
    return _guard_tmp_root() / f"cozempic_guard_{slug}.pid"


def _cleanup_legacy_pid(cwd: str) -> None:
    """Remove old CWD-keyed PID files from pre-1.6.13 installations."""
    legacy = _pid_file_for_cwd(cwd)
    if legacy.exists():
        try:
            pid = int(legacy.read_text().strip())
            os.kill(pid, 0)
            # Only SIGTERM if we can confirm this is actually our daemon.
            if _is_cozempic_guard_process(pid):
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            pass
        legacy.unlink(missing_ok=True)
    # Also clean session file
    legacy_sess = Path(str(legacy).replace(".pid", "_session.txt"))
    legacy_sess.unlink(missing_ok=True)


def _safe_unlink_session_pidfile(session_id: str | None) -> None:
    """Best-effort pidfile unlink on daemon exit paths.

    Used by every daemon shutdown surface (SIGTERM handler, K=10
    voluntary exit, KeyboardInterrupt, the four `break` paths in
    ``start_guard``'s main loop). Wired through the ``finally`` block
    of ``start_guard`` so a single call site covers all 6 exit paths.

    CAS gate: only unlinks if the pidfile currently contains OUR PID
    (``_pid_file_points_to(session_id, os.getpid())``). This prevents
    destroying a peer's just-completed claim during the brief window
    where a concurrent SessionStart hook may have already spawned a
    replacement daemon and rewritten the pidfile with its PID. Mirrors
    the CAS pattern in ``reload_self_daemon`` (sister-module precedent
    at lines 1802, 1809, 1823, 1829).

    Swallows ValueError (malformed session_id passed in via stale
    closure capture) and OSError (pidfile already gone, /tmp unwritable,
    EACCES). Never raises — the daemon is mid-shutdown; nothing useful
    to do on failure.

    Class-of-bug fold (PR #93 commit 2): consolidates the unlink so
    adding a new ``sys.exit`` path requires touching ONE callsite, not
    N. Covers the pre-existing ``_graceful_shutdown`` leak (TODO:55,
    pre-PR-#88) AND the K=10 leak PR #92 introduced — both daemon-exit
    surfaces now reach the same ``finally`` block in ``start_guard``.
    """
    if not session_id:
        return
    try:
        if _pid_file_points_to(session_id, os.getpid()):
            _pid_file_for_session(session_id).unlink(missing_ok=True)
    except (ValueError, OSError):
        pass


def _is_guard_running_for_session(session_id: str) -> int | None:
    """Check if a guard daemon is already running for this specific session.

    Returns the PID if running, None otherwise.

    An invalid `session_id` (non-UUID) is treated as "no daemon" (None)
    rather than raising — library-API safety. Callers outside the CLI
    (hooks, pytest, third-party integrations) should get a safe default
    instead of a ValueError propagating up from `_pid_file_for_session`.
    """
    try:
        pid_path = _pid_file_for_session(session_id)
    except ValueError:
        # Invalid session_id shape — no daemon can exist for it.
        return None
    if not pid_path.exists():
        return None

    try:
        # Tolerant parse: handles both legacy 1-line and new 3-line
        # pidfile formats (PR #93 item #5). Returns 0 on garbled/empty
        # content — caller's `if pid <= 0` branch then unlinks the stale
        # file. Replaces `int(read_text().strip())` which would raise
        # ValueError on 3-line content and (via the except below) skip
        # the unlink, leaking the stale file.
        from .spawn_lock import _parse_pidfile_pid
        pid = _parse_pidfile_pid(pid_path)
        if pid <= 0:
            # Pidfile contains a sentinel/placeholder — treat as stale.
            # Guards against the PID-reuse footgun where os.kill(0, sig)
            # broadcasts to the caller's process group rather than
            # targeting a sentinel. Cross-process freshness for in-flight
            # claims is enforced by DaemonSpawnClaim's O_CREAT|O_EXCL +
            # _FRESH_PIDFILE_SECONDS gate, not by an in-process dict.
            pid_path.unlink(missing_ok=True)
            return None
        os.kill(pid, 0)
        # Verify the PID is actually our guard — defend against PID reuse.
        if not _is_cozempic_guard_process(pid):
            # Don't eagerly unlink a fresh-looking pidfile here. A peer
            # process that just did O_CREAT|O_EXCL in DaemonSpawnClaim has
            # written its own parent PID into the file BEFORE renaming to
            # the daemon PID; in that brief window the holding PID is a
            # legitimate Python process that isn't yet a cozempic guard.
            # Treating it as PID-reuse and unlinking would destroy the
            # peer's claim and let multiple workers spawn. Only unlink
            # truly old pidfiles — those are real PID-reuse or genuine
            # stale state from a crashed prior spawn. The threshold is
            # shared with ``DaemonSpawnClaim._is_pidfile_fresh`` so both
            # sides of the claim/probe dichotomy agree on what "fresh"
            # means (H1 fix — single source of truth).
            from .spawn_lock import _FRESH_PIDFILE_SECONDS
            try:
                age = time.time() - pid_path.stat().st_mtime
            except OSError:
                age = 0.0
            if age >= _FRESH_PIDFILE_SECONDS:
                pid_path.unlink(missing_ok=True)
            return None
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        # Apparently dead PID — but a freshly-written pidfile that holds
        # the soon-to-exist daemon PID can momentarily look "dead" while
        # the daemon is still starting (a real Popen returns the child
        # PID before the OS finishes wiring up the process; test mocks
        # use fake PIDs that are never alive). Only unlink truly old
        # pidfiles to avoid destroying a peer's just-completed claim
        # and letting another worker spawn a duplicate daemon. Same
        # threshold as the holder-alive-but-not-guard branch above.
        from .spawn_lock import _FRESH_PIDFILE_SECONDS
        try:
            age = time.time() - pid_path.stat().st_mtime
        except OSError:
            age = 0.0
        if age >= _FRESH_PIDFILE_SECONDS:
            pid_path.unlink(missing_ok=True)
        return None
    except OSError:
        # Windows: os.kill(pid, 0) raises a bare OSError [WinError 87]
        # (invalid parameter) for a non-existent PID instead of the POSIX
        # ProcessLookupError caught above. Treat it as a dead PID, reusing the
        # same freshness-aware unlink so we don't destroy a peer's just-written
        # claim. Re-raise on POSIX, where a bare OSError here is unexpected and
        # must not be silently masked.
        if os.name != "nt":
            raise
        from .spawn_lock import _FRESH_PIDFILE_SECONDS
        try:
            age = time.time() - pid_path.stat().st_mtime
        except OSError:
            age = 0.0
        if age >= _FRESH_PIDFILE_SECONDS:
            pid_path.unlink(missing_ok=True)
        return None


# Backward compat aliases
def _pid_file(cwd: str) -> Path:
    return _pid_file_for_cwd(cwd)


def start_guard_daemon(
    cwd: str | None = None,
    threshold_mb: float = 50.0,
    soft_threshold_mb: float | None = None,
    rx_name: str = "standard",
    interval: int = 30,
    auto_reload: bool = True,
    reactive: bool = True,
    threshold_tokens: int | None = None,
    soft_threshold_tokens: int | None = None,
    session_id: str | None = None,
    claude_pid: int | None = None,
    protect_patterns: list | None = None,
) -> dict:
    """Start the guard as a background daemon.

    Spawns a detached subprocess running `cozempic guard` with output
    redirected to a log file. Uses a PID file to prevent double-starts.

    Pre-validates numeric parameters before spawning the child process.
    Without this, bad values (negative thresholds, zero intervals) would
    pass to the child via CLI args, be accepted by argparse (which only
    runs in the child), and cause the child to die immediately — while
    the caller sees started=True.

    Returns dict with: started (bool), pid (int|None), pid_file, log_file,
    already_running (bool).
    """
    _validate_finite_thresholds(
        threshold_mb=threshold_mb,
        soft_threshold_mb=soft_threshold_mb,
        interval=interval,
        threshold_tokens=threshold_tokens,
        soft_threshold_tokens=soft_threshold_tokens,
    )
    if threshold_mb is not None and threshold_mb <= 0:
        raise ConfigError(f"threshold_mb must be positive, got {threshold_mb}")
    if soft_threshold_mb is not None and soft_threshold_mb <= 0:
        raise ConfigError(f"soft_threshold_mb must be positive, got {soft_threshold_mb}")
    if soft_threshold_mb is not None and threshold_mb is not None and soft_threshold_mb >= threshold_mb:
        raise ConfigError(
            f"soft_threshold_mb ({soft_threshold_mb}) must be strictly less than "
            f"threshold_mb ({threshold_mb})"
        )
    if interval is not None and interval <= 0:
        raise ConfigError(f"interval must be positive, got {interval}")
    if threshold_tokens is not None and threshold_tokens <= 0:
        raise ConfigError(f"threshold_tokens must be positive, got {threshold_tokens}")
    if soft_threshold_tokens is not None and soft_threshold_tokens <= 0:
        raise ConfigError(f"soft_threshold_tokens must be positive, got {soft_threshold_tokens}")

    cwd = cwd or os.getcwd()

    # Migrate: clean up legacy CWD-keyed PID files from pre-1.6.13
    _cleanup_legacy_pid(cwd)

    # NEW-1 sentinel check: if a reload is in flight for this session, skip spawn.
    # The reload watcher will unlink the sentinel after osascript fires; the new
    # Claude's own SessionStart then spawns the real guard. This prevents the
    # transient-daemon race where a concurrent upgrade-chain SessionStart re-fire
    # claims the pidfile slot for the OLD Claude's dying session.
    if session_id and _reload_sentinel_active(session_id):
        return {
            "started": False,
            "reason": "reload in flight",
            "pid": None,
            "pid_file": None,
            "log_file": None,
            "already_running": False,
        }

    # If we have a session_id, check if a guard already exists for THIS session
    if session_id:
        existing_pid = _is_guard_running_for_session(session_id)
        if existing_pid:
            return {
                "started": False,
                "pid": existing_pid,
                "pid_file": str(_pid_file_for_session(session_id)),
                "log_file": None,
                "already_running": True,
            }
    else:
        # No session_id — detect from CWD (backward compat with old hooks).
        # strict=True: if ambiguous, skip dedup rather than dedup against the
        # wrong session's PID file (which would pass spuriously and spawn a
        # second daemon). Behavior with strict→None matches old hook invocations
        # that provided no session_id (dedup was simply skipped then too).
        sess = find_current_session(cwd, strict=True)
        if sess:
            session_id = sess.get("session_id", "")

        if session_id:
            existing_pid = _is_guard_running_for_session(session_id)
            if existing_pid:
                return {
                    "started": False,
                    "pid": existing_pid,
                    "pid_file": str(_pid_file_for_session(session_id)),
                    "log_file": None,
                    "already_running": True,
                }

    # Normalize early — session_id may be a full .jsonl path from the hook's
    # $TRANSCRIPT variable. Must extract the UUID before using it as a filename
    # component (otherwise "/Users/foo/..." ends up in the log/pid path).
    if session_id:
        session_id = _normalize_session_id(session_id)

    # Use session_id for PID file if available, fall back to CWD hash.
    # Route through `_pid_file_for_session` so the UUID-shape / lowercase /
    # hex-first-char validation applies at the spawn path too. Without this
    # the write-side builds a different path than the read-side helper
    # (`_is_guard_running_for_session`), and the caller's own daemon becomes
    # an unreachable orphan for non-UUID session ids.
    if session_id:
        try:
            pid_path = _pid_file_for_session(session_id)
        except ValueError as e:
            return {
                "started": False,
                "reason": f"invalid session_id: {e}",
                "pid": None,
                "pid_file": None,
                "log_file": None,
                "already_running": False,
            }
        log_file = pid_path.with_suffix(".log")
    else:
        import hashlib
        pid_key = hashlib.md5(cwd.encode()).hexdigest()[:12]
        log_file = _guard_tmp_root() / f"cozempic_guard_{pid_key}.log"
        pid_path = _guard_tmp_root() / f"cozempic_guard_{pid_key}.pid"

    if claude_pid is None:
        claude_pid = find_claude_pid()

    # ── Cross-process spawn claim (Bug 2 + Bug 3 fix, V4 rework) ────────────
    # The PID file IS the lock. O_CREAT|O_EXCL on the PID file is the only
    # primitive used: POSIX guarantees exactly one process wins the create,
    # all others see EEXIST and become losers via DaemonAlreadyStarting.
    # This mirrors reload_lock.py:200-262 (same pattern, different file).
    #
    # Why not fcntl.flock on a separate sentinel? Race-reproducer's V4 stress
    # (10 processes × 30 iterations) found a textbook flock-unlink race: when
    # the holder unlinks the sentinel on release, peers immediately O_CREAT
    # NEW inodes and their flocks attach to those new inodes — different
    # kernel objects, so multiple "winners" each acquire flock simultaneously.
    # See spawn_lock.py module docstring for the full failure mode + evidence.
    from .spawn_lock import DaemonAlreadyStarting, DaemonSpawnClaim

    try:
        claim = DaemonSpawnClaim(session_id or cwd, pid_path)
        claim.__enter__()
    except DaemonAlreadyStarting as exc:
        # Peer process holds the PID-file claim. Surface their PID so the
        # SessionStart hook can introspect / log it. holder_pid may be 0 if
        # the file was unreadable (rare; race-reproducer's "undefined state"
        # was an artifact of the OSError path that no longer exists).
        return {
            "started": False,
            "pid": exc.holder_pid,
            "pid_file": str(pid_path),
            "log_file": None,
            "already_running": True,
        }

    try:
        # Build the guard command
        cmd_parts = [
            sys.executable, "-m", "cozempic.cli", "guard",
            "--cwd", cwd,
            "--threshold", str(threshold_mb),
            "--interval", str(interval),
            "-rx", rx_name,
        ]
        if soft_threshold_mb is not None:
            cmd_parts.extend(["--soft-threshold", str(soft_threshold_mb)])
        if not auto_reload:
            cmd_parts.append("--no-reload")
        if not reactive:
            cmd_parts.append("--no-reactive")
        if threshold_tokens is not None:
            cmd_parts.extend(["--threshold-tokens", str(threshold_tokens)])
        if soft_threshold_tokens is not None:
            cmd_parts.extend(["--soft-threshold-tokens", str(soft_threshold_tokens)])
        if session_id is not None:
            cmd_parts.extend(["--session", _normalize_session_id(session_id)])
        if claude_pid is not None:
            cmd_parts.extend(["--claude-pid", str(claude_pid)])
        # --protect-pattern (#122): serialize each user regex back to a child CLI arg.
        # Use the `--opt=VALUE` form (not two argv elements): a pattern starting with
        # `-` (e.g. `-rf`) is otherwise mis-parsed by the child's argparse as a flag,
        # which exits(2) and silently kills the spawned daemon. list-argv → no shell
        # injection; the child re-compiles the verbatim pattern.
        for _pp in (protect_patterns or []):
            cmd_parts.append(f"--protect-pattern={getattr(_pp, 'pattern', str(_pp))}")

        # Wrap the spawn body in a graceful OSError handler so a
        # non-interactive SessionStart hook never crashes with a stack
        # trace. ENOSPC / EROFS / EACCES / EMFILE on /tmp surface as
        # structured `{started: False, reason: ...}`. The claim's
        # __exit__ will unlink the PID file on exception, so a retry is
        # possible.
        try:
            # Defense-in-depth: if the log file's parent dir was removed
            # mid-spawn (race with operator cleanup, /tmp eviction, etc.)
            # recreate it once and retry the open.
            try:
                lf = open(log_file, "a", encoding="utf-8")
            except FileNotFoundError:
                log_dir = os.path.dirname(str(log_file))
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                lf = open(log_file, "a", encoding="utf-8")

            try:
                from datetime import datetime
                lf.write(f"\n--- Guard daemon started at {datetime.now().isoformat()} ---\n")
                lf.write(f"CWD: {cwd}\n")
                lf.write(f"CMD: {' '.join(cmd_parts)}\n\n")
                lf.flush()

                # PYTHONUNBUFFERED=1 ensures guard log output is written immediately (#14)
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                # Detach the child so it outlives the parent. start_new_session
                # is POSIX-only — on Windows it raises OSError [WinError 87]
                # (invalid parameter), especially when the parent's stdio
                # handles aren't inheritable (spawned under wscript -> hidden
                # powershell -> Start-Process). Use the Windows creationflags
                # equivalents there.
                popen_kwargs = {
                    "stdout": lf,
                    "stderr": lf,
                    "stdin": subprocess.DEVNULL,
                    "cwd": cwd,
                    "env": env,
                }
                if os.name == "nt":
                    popen_kwargs["creationflags"] = (
                        subprocess.DETACHED_PROCESS
                        | subprocess.CREATE_NEW_PROCESS_GROUP
                        | subprocess.CREATE_NO_WINDOW
                    )
                else:
                    popen_kwargs["start_new_session"] = True
                proc = subprocess.Popen(cmd_parts, **popen_kwargs)
            finally:
                lf.close()

            # Atomically replace our parent PID (written by DaemonSpawnClaim
            # on _claim) with the daemon's real PID. tmp+rename is atomic
            # on the same filesystem — readers transitioning across the
            # rename see either the parent PID (alive) or the daemon PID
            # (alive). Never empty, never "0", never partial.
            #
            # CRIT C1 fix: open the .pid.tmp via os.open(O_CREAT|O_EXCL|
            # O_NOFOLLOW) instead of Path.write_text. The default write_text
            # follows symlinks — an attacker who pre-plants the .pid.tmp
            # path as a symlink to ~/.zshrc or ~/.ssh/authorized_keys would
            # have the file overwritten with the PID number. O_EXCL also
            # surfaces orphan .pid.tmp files (from a prior SIGKILLed spawn)
            # as a FileExistsError instead of silently truncating them,
            # which closes a re-attack window in CRIT C3.
            tmp_path = pid_path.with_suffix(".pid.tmp")
            # CRIT C3 fix: catch ANY exception (not just OSError) around
            # the write+rename block. A SIGINT/InterruptedError or other
            # non-OSError between write_text and rename used to leak the
            # .pid.tmp orphan; we now unlink it on every failure path.
            try:
                from .spawn_lock import INIT_SPAWN_DAEMON
                from datetime import datetime as _dt
                _tmp_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                if hasattr(os, "O_NOFOLLOW"):
                    _tmp_flags |= os.O_NOFOLLOW
                _tmp_fd = os.open(str(tmp_path), _tmp_flags, 0o600)
                try:
                    # 3-line payload: pid + iso-timestamp + initiator.
                    # Mirrors DaemonSpawnClaim._claim and
                    # reload_lock._ReloadLock._try_create. Operators
                    # cat-ing the pidfile see immediately who wrote it
                    # (parent vs daemon) and when (PR #93 item #5).
                    payload = (
                        f"{proc.pid}\n"
                        f"{_dt.now().isoformat(timespec='seconds')}\n"
                        f"{INIT_SPAWN_DAEMON}\n"
                    )
                    os.write(_tmp_fd, payload.encode("utf-8"))
                    # Fsync the payload to disk BEFORE rename so a power
                    # loss between rename and parent-dir-fsync can't
                    # produce a renamed-but-empty pidfile that readers
                    # then misclassify as garbled (DA round 1 M1).
                    try:
                        os.fsync(_tmp_fd)
                    except OSError:
                        pass
                finally:
                    os.close(_tmp_fd)
                # os.replace (not os.rename): on Windows os.rename raises
                # FileExistsError [WinError 183] when pid_path already exists
                # (a stale .pid from an abruptly-terminated guard is the
                # documented leftover state — see doctor.py). os.replace
                # overwrites atomically on both POSIX and Windows. (#113)
                os.replace(str(tmp_path), str(pid_path))
                # Fsync the parent directory so the rename itself is
                # durable across abrupt power loss (DA round 1 M1).
                # Without this, the rename is in the kernel's metadata
                # journal but not yet on stable storage — a crash
                # between rename and the next fs commit could roll the
                # filesystem back to pre-rename state, leaving an
                # orphan .pid.tmp and no .pid (next spawn would see no
                # pidfile and start a duplicate daemon).
                try:
                    parent_dir = os.path.dirname(str(pid_path)) or "."
                    _dir_fd = os.open(parent_dir, os.O_RDONLY)
                    try:
                        os.fsync(_dir_fd)
                    finally:
                        os.close(_dir_fd)
                except OSError:
                    # Some filesystems (network FS, tmpfs on certain
                    # kernels) reject directory fsync — best-effort.
                    pass
            except Exception:
                # Unlink any partial .pid.tmp we may have created so a
                # retry can succeed. unlink is symlink-safe (operates on
                # the directory entry, not the symlink target).
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            # Tell the claim "we wrote the real PID — leave the file in
            # place on clean exit; the daemon now owns its lifecycle."
            claim.handed_off = True
        except OSError as exc:
            # The .pid.tmp orphan was already cleaned by the inner
            # try/except above; here we only need to surface the failure.
            # The claim's __exit__ will unlink the .pid file because
            # handed_off is still False, so a retry can re-claim.
            return {
                "started": False,
                "reason": f"pidfile: {exc}",
                "pid": None,
                "pid_file": str(pid_path),
                "log_file": None,
                "already_running": False,
            }

        return {
            "started": True,
            "pid": proc.pid,
            "pid_file": str(pid_path),
            "log_file": str(log_file),
            "already_running": False,
        }
    finally:
        # If we reach here without an exception, claim.handed_off == True
        # and __exit__ is a no-op (daemon owns the PID file). If we raised
        # inside the spawn body, __exit__ unlinks for retry.
        claim.__exit__(None, None, None)


def _is_cozempic_guard_process(pid: int) -> bool:
    """Verify that `pid` is actually a cozempic guard daemon before we signal it.

    Guards against PID reuse: when our daemon exits and the kernel recycles
    its PID to an unrelated user process, a blind `os.kill(pid, SIGTERM)` on
    the recycled PID is a confused-deputy bug (we'd kill something arbitrary).
    Inspects the process's argv; requires BOTH "cozempic.cli guard" (matches
    our spawn pattern in start_guard_daemon) OR the explicit entry-point
    "cozempic guard" — not just substring "cozempic" + "guard" which could
    match unrelated things like `vim /tmp/cozempic_guard_notes.md`.
    """
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if result.returncode != 0:
            return False
        args = (result.stdout or "").strip()
        tokens = args.split()
        if not tokens:
            return False
        binary = Path(tokens[0]).name.lower()
        # tokens[0] must be a python interpreter (any minor/patch version) or
        # the cozempic entry-point. Rejects `run-cozempic`, `fake-cozempic`,
        # `python-attacker`. Accepts `python3.11`, `python3.13.12`, etc. used
        # by pyenv / Homebrew / distro packaging.
        if not (binary == "cozempic" or re.fullmatch(r"^python(\d+(\.\d+)*)?$", binary)):
            return False
        # "cozempic.cli" and "guard" must appear as discrete arg tokens, not as
        # substrings in filenames/paths (grep, less, vim on our source tree).
        if "cozempic.cli" in tokens and "guard" in tokens:
            return True
        if len(tokens) >= 2 and binary == "cozempic" and tokens[1] == "guard":
            return True
        return False
    except (subprocess.SubprocessError, OSError, TypeError):
        # If we can't verify, err on the side of NOT signaling a potentially
        # unrelated process. The session stays with the existing daemon (or
        # no daemon), which is strictly safer than signaling the wrong one.
        # TypeError covers the test-only case where a Popen mock returns a
        # bare object that doesn't support the ctx-manager protocol
        # subprocess.run uses internally; production callers never hit it,
        # but any unhandled exception here would propagate to the
        # non-interactive SessionStart hook surface — fail closed.
        return False


_MTIME_LIVENESS_WINDOW_SEC = 60

# ── PID start-time identity store (anti-recycling gate) ─────────────────────
# Keyed by session_id → (expected_pid, expected_start_time_float).
# Populated once at start_guard startup after Claude PID is confirmed.
# Cleared when Claude exits (watchdog break path).
# In-memory is sufficient: the recycled-PID race occurs within one daemon
# lifecycle. Daemon restart → fresh find_claude_pid() → fresh recording.
_CLAUDE_IDENTITY: dict[str, tuple[int, float]] = {}


def _get_pid_start_time_linux(pid: int) -> float | None:
    """Linux: read start_time from /proc/<pid>/stat + /proc/stat btime. No subprocess."""
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text()
        # comm field may contain spaces and ')'; rfind(")") finds end of comm safely.
        # After "pid (comm) ", fields are 0-indexed: index 19 = starttime (field 22).
        # Guard malformed /proc (fuse / WSL1 / BSD emulation); kernel /proc always
        # has parens but the slice would silently misalign on no-parens input.
        close_paren = stat_text.rfind(")")
        if close_paren < 0:
            return None
        after_comm = stat_text[close_paren + 2:]
        starttime_ticks = int(after_comm.split()[19])
        btime_line = next(
            line for line in Path("/proc/stat").read_text().splitlines()
            if line.startswith("btime ")
        )
        btime = int(btime_line.split()[1])
        return float(btime + starttime_ticks / os.sysconf("SC_CLK_TCK"))
    except Exception:
        return None


def _get_pid_start_time_macos(pid: int) -> float | None:
    """macOS: parse ps -o lstart= output. 1-second resolution; LC_ALL=C for locale safety."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True, text=True, timeout=2.0, check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        # Normalize whitespace: single-digit days produce double-space ("May  1").
        normalized = re.sub(r"\s+", " ", result.stdout.strip()).strip()
        return float(time.mktime(time.strptime(normalized, "%a %b %d %H:%M:%S %Y")))
    except Exception:
        return None


def _get_pid_start_time_psutil(pid: int) -> float | None:
    """psutil fallback: microsecond precision; lazy-import (no required dep)."""
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except ImportError:
        return None
    except Exception:
        return None


def _get_pid_start_time(pid: int) -> float | None:
    """Return process creation time in seconds since epoch, or None.

    Platform-ordered chain (zero required deps):
      Linux  → /proc/<pid>/stat  (10ms resolution, no subprocess)
      macOS  → ps -p <pid> -o lstart=  (1s resolution, subprocess)
      psutil → lazy-import fallback    (microsecond precision, all platforms)

    Falls through to psutil if the platform-native backend fails (e.g.,
    restricted /proc on containerised Linux, ps absent, permission error).
    Returns None only when all backends fail → _pid_identity_match fails-OPEN.
    """
    _sys = platform.system()
    if _sys == "Linux":
        result = _get_pid_start_time_linux(pid)
        if result is not None:
            return result
    elif _sys == "Darwin":
        result = _get_pid_start_time_macos(pid)
        if result is not None:
            return result
    return _get_pid_start_time_psutil(pid)


def _record_claude_identity(session_id: str, pid: int) -> None:
    """Record (pid, start_time) for the anti-recycling gate. Call once at startup.

    Validates pid is actually Claude (argv check) before recording — defense
    in depth in case a future caller bypasses find_claude_pid's identity gate.
    """
    if not _is_claude_process(pid):
        return
    start_time = _get_pid_start_time(pid)
    if start_time is not None and session_id:
        _CLAUDE_IDENTITY[session_id] = (pid, start_time)


def _pid_identity_match(pid: int, session_id: str | None) -> bool:
    """True if pid matches the recorded identity (same PID + same start_time).

    Returns True conservatively when:
    - session_id is None (no session context — backward compat)
    - no identity has been recorded for this session_id (daemon restarted)
    - all start-time backends fail (can't get start_time — degrade gracefully)

    Fail-OPEN rationale: in all these cases we fall through to the existing
    _pid_is_alive + _is_claude_process layers. No regression vs v1.8.16.
    """
    if not session_id:
        return True
    identity = _CLAUDE_IDENTITY.get(session_id)
    if identity is None:
        return True
    recorded_pid, recorded_start_time = identity
    if pid != recorded_pid:
        return False
    current_start_time = _get_pid_start_time(pid)
    if current_start_time is None:
        return True  # all backends failed — degrade gracefully (fail-OPEN)
    # 0.1s tolerance absorbs float-precision noise across psutil's kernel-clock
    # conversion; real PID-recycle gaps are seconds-to-hours, never sub-second.
    return abs(current_start_time - recorded_start_time) < 0.1


def _pid_is_alive(pid: int) -> bool:
    """Bare process-liveness probe — does NOT consult the JSONL mtime.

    Anti-resurrection: a dead PID must read as dead even when cozempic's own
    ``save_messages`` just refreshed the session JSONL moments earlier.
    ``_is_claude_process``'s mtime fallback would misread that fresh write as a
    live Claude and let the reload watcher resurrect a session the user closed.
    ``os.kill(pid, 0)`` answers liveness directly and is not fooled by it.
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists, owned by another user
    except OverflowError:
        return False  # pid too large to be a real process id (malformed --claude-pid)
    except OSError:
        # Windows raises OSError [WinError 87] for a non-existent PID; treat any
        # Windows os.kill failure as "gone". On POSIX an unexpected OSError here
        # is rare — assume alive so we never skip a legitimate reload.
        return os.name != "nt"


def _is_claude_process(pid: int, session_path: Path | None = None) -> bool:
    """Verify that `pid` is a Claude Code process (node/claude binary).

    Mirrors _is_cozempic_guard_process but for the Claude client side.
    Guards against PID reuse: if Claude exits and its PID is recycled, a blind
    SIGTERM on the recycled PID is a confused-deputy bug.

    When `session_path` is provided and the ps-based check is inconclusive,
    falls back to JSONL-mtime corroboration: a file written within the last
    minute means Claude is almost certainly still alive, even if ps misses
    the match (observed on macOS when Claude forks a subshell whose args
    don't carry the claude-code marker).

    On Windows, `ps` is unavailable — uses `tasklist /FI "PID eq <pid>" /FO CSV`
    instead. If tasklist also fails, falls back to liveness-only (returns True
    for a live PID) so callers can still proceed with taskkill.
    """
    if platform.system() == "Windows":
        return _is_claude_process_windows(pid)
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if result.returncode == 0:
            args = (result.stdout or "").strip()
            tokens = args.split()
            if tokens:
                binary = Path(tokens[0]).name.lower()
                # Match native claude binary (whole name, not substring)
                if binary == "claude":
                    return True
                # Match node-based Claude Code: binary must be exactly "node"
                # or "node.js" AND args must contain a Claude Code marker.
                if binary in ("node", "node.js"):
                    if "@anthropic-ai/claude-code" in args:
                        return True
                    if "claude-code/cli.js" in args or "claude-code\\cli.js" in args:
                        return True
    except (subprocess.SubprocessError, OSError):
        pass

    # ps was inconclusive (no match, or subprocess error). If we have a
    # session path and its JSONL was touched very recently, take that as
    # corroboration: the Claude daemon is the only writer on that file.
    if session_path is not None:
        try:
            age = time.time() - session_path.stat().st_mtime
            if age < _MTIME_LIVENESS_WINDOW_SEC:
                return True
        except OSError:
            pass
    return False


def _is_claude_process_windows(pid: int) -> bool:
    """Windows-specific helper: probe via tasklist /FO CSV."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode != 0:
            return True  # liveness fallback — let caller proceed with taskkill
        output = (result.stdout or "").strip().lower()
        if not output or "no tasks are running" in output:
            return False
        # CSV row: "image_name","pid","session_name","session#","mem_usage"
        # Image name is the first quoted field.
        image_name = output.split(",")[0].strip('"')
        return any(marker in image_name for marker in ("claude", "node"))
    except (subprocess.SubprocessError, OSError):
        return True  # liveness fallback — let caller proceed with taskkill


def _pid_file_points_to(session_id: str, expected_pid: int) -> bool:
    """CAS helper: return True if the session pid file currently contains
    `expected_pid`. Used before unlink() to avoid clobbering a fresh pid
    file written by a concurrent SessionStart hook.

    Uses ``_parse_pidfile_pid`` so both legacy 1-line and new 3-line
    pidfile formats parse correctly (PR #93 item #5). A garbled file
    returns 0 from the parser, which won't match any expected_pid (>0),
    so the CAS skips the unlink — the conservative behaviour.
    """
    try:
        from .spawn_lock import _parse_pidfile_pid
        path = _pid_file_for_session(session_id)
        if not path.exists():
            return False
        return _parse_pidfile_pid(path) == expected_pid
    except (ValueError, OSError):
        return False


def reload_self_daemon(
    cwd: str | None = None,
    session_id: str | None = None,
    threshold_mb: float = 50.0,
    soft_threshold_mb: float | None = None,
    rx_name: str = "standard",
    interval: int = 30,
    auto_reload: bool = True,
    reactive: bool = True,
    threshold_tokens: int | None = None,
    soft_threshold_tokens: int | None = None,
    protect_patterns: list | None = None,
) -> dict:
    """Gracefully restart the running guard daemon for this session.

    Used after an in-place cozempic upgrade so the daemon picks up the new code
    on disk. SIGTERMs the existing daemon (it writes a final checkpoint via the
    SIGTERM handler), waits for it to exit, then spawns a fresh daemon with the
    same args. The new daemon imports from the freshly-installed package files.

    Returns dict: {reloaded: bool, old_pid, new_pid, log_file, reason}.
    """
    # Validate BEFORE any destructive action — a NaN/inf threshold must not
    # kill the live daemon and then fail to spawn a replacement, leaving the
    # session unprotected.  Mirrors the belt-and-braces checks in start_guard
    # and start_guard_daemon.
    _validate_finite_thresholds(
        threshold_mb=threshold_mb,
        soft_threshold_mb=soft_threshold_mb,
        interval=interval,
        threshold_tokens=threshold_tokens,
        soft_threshold_tokens=soft_threshold_tokens,
    )
    cwd = cwd or os.getcwd()

    if not session_id:
        # strict=True: if Strategy 3 fails, return "could not detect session" rather
        # than looking for the reload target under a wrong (Strategy 4) session UUID,
        # which would fail anyway (no daemon under that UUID) and give a misleading
        # "no daemon running" error instead of the actual "ambiguous session" cause.
        sess = find_current_session(cwd, strict=True)
        if sess:
            session_id = sess.get("session_id", "")

    if not session_id:
        return {"reloaded": False, "reason": "could not detect session"}

    session_id = _normalize_session_id(session_id)

    # `_is_guard_running_for_session` catches ValueError from the regex gate
    # and returns None for invalid session_ids, so subsequent direct calls
    # to `_pid_file_for_session` below are safe when old_pid is truthy.
    old_pid = _is_guard_running_for_session(session_id)
    if not old_pid:
        return {"reloaded": False, "reason": "no daemon running for session"}

    # Verify the PID is actually our daemon — defend against PID reuse.
    if not _is_cozempic_guard_process(old_pid):
        # Stale pid file pointing at a recycled (non-cozempic) PID. Clear it
        # (only if it still points at the stale pid — CAS) and spawn fresh;
        # do NOT signal the unrelated process.
        if _pid_file_points_to(session_id, old_pid):
            _pid_file_for_session(session_id).unlink(missing_ok=True)
        old_pid = None
    else:
        try:
            os.kill(old_pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            if _pid_file_points_to(session_id, old_pid):
                _pid_file_for_session(session_id).unlink(missing_ok=True)
            old_pid = None

        if old_pid is not None and not _wait_for_exit(old_pid, timeout=10.0):
            # Didn't exit on SIGTERM — escalate, but only if we still see our
            # daemon (guard against the unlikely race where another process
            # grabbed the PID right as the old daemon finally died).
            if _is_cozempic_guard_process(old_pid):
                try:
                    os.kill(old_pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            # CAS unlink — don't wipe a fresh pid file from a concurrent spawn
            if _pid_file_points_to(session_id, old_pid):
                _pid_file_for_session(session_id).unlink(missing_ok=True)
        elif old_pid is not None:
            # Clean exit. CAS unlink — if a concurrent SessionStart hook
            # already spawned a new daemon and rewrote the pid file with its
            # PID, we leave that fresh file alone.
            if _pid_file_points_to(session_id, old_pid):
                _pid_file_for_session(session_id).unlink(missing_ok=True)

    # Always re-activate what we just disabled. Retry once on transient failures,
    # but NOT on `already_running` (that means a concurrent SessionStart hook
    # already spawned a new daemon — accept that one, don't start a second).
    daemon_args = dict(
        cwd=cwd,
        threshold_mb=threshold_mb,
        soft_threshold_mb=soft_threshold_mb,
        rx_name=rx_name,
        interval=interval,
        auto_reload=auto_reload,
        reactive=reactive,
        threshold_tokens=threshold_tokens,
        soft_threshold_tokens=soft_threshold_tokens,
        session_id=session_id,
        protect_patterns=protect_patterns,
    )
    result = start_guard_daemon(**daemon_args)
    if not result.get("started") and not result.get("already_running"):
        time.sleep(1)
        # Only clear a pid file we know is stale (pointing at a dead pid).
        # Do NOT blindly unlink — a live concurrent daemon may have written it.
        pid_path = _pid_file_for_session(session_id)
        try:
            if pid_path.exists():
                from .spawn_lock import _parse_pidfile_pid
                stale_pid = _parse_pidfile_pid(pid_path)
                if stale_pid <= 0:
                    # Garbled or empty — treat as stale and unlink.
                    pid_path.unlink(missing_ok=True)
                    stale_pid = 0
                try:
                    if stale_pid > 0:
                        os.kill(stale_pid, 0)
                    # Still alive — leave the pid file alone and let
                    # start_guard_daemon below return already_running.
                except (ProcessLookupError, PermissionError):
                    pid_path.unlink(missing_ok=True)
        except (ValueError, OSError):
            pid_path.unlink(missing_ok=True)
        result = start_guard_daemon(**daemon_args)

    reloaded = bool(result.get("started") or result.get("already_running"))
    if reloaded:
        reason = "ok"
    else:
        reason = "could not start fresh daemon after retry — session is unprotected"

    return {
        "reloaded": reloaded,
        "old_pid": old_pid,
        "new_pid": result.get("pid"),
        "log_file": result.get("log_file"),
        "reason": reason,
    }


def _hard_loop_backoff_sleep(consecutive_empty: int, interval: int) -> int:
    """Compute the sleep duration for the next HARD-loop cycle.

    Doubles the wait starting at ``HARD_LOOP_BACKOFF_START`` consecutive
    zero-byte HARD prunes, capped at ``HARD_LOOP_BACKOFF_CAP_SECONDS``.
    Returns ``interval`` unchanged for K < HARD_LOOP_BACKOFF_START.

    With defaults (interval=30, start=3, cap=300):
        K=1 → 30s   (normal)
        K=2 → 30s   (normal)
        K=3 → 60s   (interval * 2 ** 1)
        K=4 → 120s  (interval * 2 ** 2)
        K=5 → 240s  (interval * 2 ** 3)
        K=6 → 300s  (capped from 480s)
        K=7+ → 300s (cap)
    """
    if consecutive_empty < HARD_LOOP_BACKOFF_START:
        return interval
    # Exponent grows from 1 at K=3 onwards: K - (start - 1).
    exp = consecutive_empty - (HARD_LOOP_BACKOFF_START - 1)
    return min(interval * (2 ** exp), HARD_LOOP_BACKOFF_CAP_SECONDS)


def _fmt_prune_result(result: dict) -> str:
    """Format a prune cycle result, leading with tokens if available."""
    orig_tok = result.get("original_tokens")
    final_tok = result.get("final_tokens")
    if orig_tok and final_tok:
        saved_tok = orig_tok - final_tok
        # Negative => exact count re-anchored after metadata-strip (#105); the
        # token delta is not meaningful, so report the reliable byte savings.
        if saved_tok >= 0 and orig_tok > 0:
            tok_str = f"{saved_tok / 1000:.1f}K" if saved_tok >= 1000 else str(saved_tok)
            pct = f"{saved_tok / orig_tok * 100:.1f}%"
            return f"{tok_str} tokens freed ({pct}), {result['saved_mb']:.1f}MB saved"
    return f"{result['saved_mb']:.1f}MB saved"


def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


def _log_safe(text, limit: int = 80) -> str:
    """Make untrusted text safe to print into the guard log: collapse newlines /
    control chars to spaces and cap length. Without this, an attacker-controlled
    team_name could inject fake 'Pruned: 0 tokens freed (0.0%)' lines into the log
    and trip cozempic guard-watchdog against a healthy daemon (C7 log-injection)."""
    import re as _re
    s = _re.sub(r"[\x00-\x1f\x7f]+", " ", str(text if text is not None else ""))
    s = _re.sub(r"\s+", " ", s).strip()
    return s[:limit]
