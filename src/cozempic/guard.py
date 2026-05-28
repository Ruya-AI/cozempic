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

from ._validation import ConfigError
from .executor import run_prescription
from .helpers import is_ssh_session, shell_quote
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
    load_messages_incremental,
    save_messages,
    snapshot_session,
)
from .team import TeamState, extract_team_state, inject_team_recovery, write_team_checkpoint
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

    # Fast path: full path exists on disk
    if p.exists() and p.suffix == ".jsonl":
        return {
            "path": p,
            "session_id": p.stem,
            "size": p.stat().st_size,
            "project": p.parent.name,
        }

    # Extract UUID from path-like input (file may not exist yet)
    search_id = _normalize_session_id(session_id)

    for attempt in range(max_retries):
        # Re-check path on each retry (file may appear)
        if p.suffix == ".jsonl" and p.exists():
            return {
                "path": p,
                "session_id": p.stem,
                "size": p.stat().st_size,
                "project": p.parent.name,
            }
        for sess in find_sessions():
            if sess["session_id"] == search_id or sess["session_id"].startswith(search_id):
                return sess
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
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
        sess = find_current_session(cwd)
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
        for block in (inner.get("content", []) if isinstance(inner.get("content"), list) else []):
            if block.get("type") == "tool_use" and block.get("name") in TEAM_TOOL_NAMES:
                tool_use_id = block.get("id", "")
                if tool_use_id:
                    pending_task_ids.add(tool_use_id)

    # 3. Tag team messages as protected (strategies skip via is_protected())
    tagged_indices: list[int] = []
    for _, msg_dict, _ in messages:
        if _is_team_message(msg_dict, pending_task_ids):
            msg_dict["__cozempic_team_protected__"] = True
            tagged_indices.append(id(msg_dict))

    # 4. Prune full list — team messages are protected, no list splitting needed
    pruned_messages, results = run_prescription(messages, strategy_names, config)

    # 5. Remove tags from surviving messages
    for _, msg_dict, _ in pruned_messages:
        msg_dict.pop("__cozempic_team_protected__", None)

    # 6. Inject team recovery messages at the end
    pruned_messages = inject_team_recovery(pruned_messages, team_state)

    return pruned_messages, results, team_state


# ─── Guard daemon ─────────────────────────────────────────────────────────────

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
) -> None:
    """Start the guard daemon with tiered pruning.

    Three-phase protection:
      1. CHECKPOINT every interval — extract team state, write to disk
      2. SOFT PRUNE at soft threshold — gentle prune, no reload, no disruption
      3. HARD PRUNE at hard threshold — full prune with team-protect + optional reload

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
        sess = _resolve_session_by_id(session_id)
    else:
        sess = find_current_session(cwd, strict=True)
    if not sess:
        # Clean up any stale PID file from this failed startup
        if session_id:
            try:
                _pid_file_for_session(session_id).unlink(missing_ok=True)
            except Exception:
                pass
        print("  ERROR: Could not detect current session.", file=sys.stderr)
        if not session_id:
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
    # can resolve the project directory without relying on slug reversal.
    from .session import record_session
    record_session(sess["session_id"], cwd or os.getcwd(), context_window)

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
        f"    Soft  ({soft_pct}%): gentle prune, no reload (file maintenance)\n"
        f"    Hard1 ({hard1_pct}%): {rx_name} prune + reload\n"
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

    try:
        while True:
            time.sleep(interval)
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
                        if not _is_claude_process(claude_pid, session_path=session_path):
                            claude_alive = False
                    except ProcessLookupError:
                        claude_alive = False
                if not claude_alive:
                    print(f"  [{_now()}] Claude process exited (PID {claude_pid}). Final checkpoint...")
                    # Option (b) defense-in-depth: unlink pidfile IMMEDIATELY so a
                    # concurrent SessionStart for the new Claude doesn't see a stale
                    # transient-daemon slot. The finally-block call is a no-op after
                    # this (CAS fails cleanly — we no longer own the file).
                    _safe_unlink_session_pidfile(sess.get("session_id"))
                    checkpoint_team(session_path=session_path, quiet=False)
                    print(f"  Guard stopping (Claude exited).")
                    break

            current_size = session_path.stat().st_size

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

            # ── Phase 4: HARD2 (80%) — aggressive + reload (ALWAYS, even with agents) ──
            hard2_tokens_hit = (
                hard2_threshold_tokens is not None
                and current_tokens is not None
                and current_tokens >= hard2_threshold_tokens
            )
            if hard2_tokens_hit:
                prune_count += 1
                reason = f"{current_tokens:,} tokens >= {hard2_threshold_tokens:,} (80%)"
                print(f"  [{_now()}] EMERGENCY THRESHOLD (80%): {reason}")
                if agents_active:
                    print(f"  WARNING: Agents are active but compaction is imminent — reload required.")
                print(f"  Aggressive prune + reload (cycle #{prune_count})...")

                result = guard_prune_cycle(
                    session_path=session_path,
                    rx_name="aggressive",
                    config=config,
                    auto_reload=auto_reload,
                    cwd=cwd or os.getcwd(),
                    session_id=sess["session_id"],
                    claude_pid=claude_pid,
                )

                if result.get("reloading"):
                    from .helpers import get_savings_line
                    savings = get_savings_line()
                    if savings:
                        print(f"  {savings}")
                    print(f"  Reload triggered. Guard exiting.")
                    break

                print(f"  Pruned: {_fmt_prune_result(result)}")
                if result.get("team_name"):
                    print(f"  Team '{result['team_name']}' state preserved ({result['team_messages']} messages)")
                print()

            # ── Phase 3: HARD1 (55%) — standard + reload (SKIP reload if agents active) ──
            elif (threshold_tokens is not None
                  and current_tokens is not None
                  and current_tokens >= threshold_tokens):
                prune_count += 1
                reason = f"{current_tokens:,} tokens >= {threshold_tokens:,} (55%)"

                if agents_active:
                    # Agents running — prune file only, no reload (don't kill active work)
                    print(f"  [{_now()}] HARD THRESHOLD (55%): {reason}")
                    print(f"  Agents active — prune file only, deferring reload (cycle #{prune_count})...")

                    result = guard_prune_cycle(
                        session_path=session_path,
                        rx_name=rx_name,
                        config=config,
                        auto_reload=False,  # Don't reload — agents are working
                        cwd=cwd or os.getcwd(),
                        session_id=sess["session_id"],
                    )
                else:
                    print(f"  [{_now()}] HARD THRESHOLD (55%): {reason}")
                    print(f"  Standard prune + reload (cycle #{prune_count})...")

                    result = guard_prune_cycle(
                        session_path=session_path,
                        rx_name=rx_name,
                        config=config,
                        auto_reload=auto_reload,
                        cwd=cwd or os.getcwd(),
                        session_id=sess["session_id"],
                        claude_pid=claude_pid,
                    )

                if result.get("reloading"):
                    from .helpers import get_savings_line
                    savings = get_savings_line()
                    if savings:
                        print(f"  {savings}")
                    print(f"  Reload triggered. Guard exiting.")
                    break

                print(f"  Pruned: {_fmt_prune_result(result)}")
                if result.get("team_name"):
                    print(f"  Team '{result['team_name']}' state preserved ({result['team_messages']} messages)")

                if result.get("active_session_refused"):
                    # P0-A: refusal upfront is not "an empty hard prune" — the
                    # cycle never ran. Do not advance the K-counter; the daemon
                    # stays alive and retries on the next interval.
                    pass
                elif result.get("saved_mb", 0) <= 0 or result.get("futile_reload_skipped"):
                    consecutive_empty_hard_prunes += 1

                    # GAP-D: emit one-shot diagnostic when reload was skipped
                    # as futile (prune saved too few bytes to justify a reload
                    # that would immediately re-trigger HARD).
                    if result.get("futile_reload_skipped") and not _futile_skip_announced:
                        saved_mb = result.get("saved_mb", 0)
                        orig_bytes = result.get("original_bytes", 0)
                        saved_pct = (saved_mb * 1024 * 1024 / orig_bytes * 100
                                     if orig_bytes > 0 else 0)
                        checkpoint_ref = (
                            f" Checkpoint: {result['checkpoint_path']}"
                            if result.get("checkpoint_path") else ""
                        )
                        print(
                            f"  [{_now()}] Hard prune freed {saved_mb:.3f}MB "
                            f"(~{saved_pct:.0f}%) — below {int(_MIN_PRUNE_RATIO * 100)}% "
                            f"threshold. Reload skipped: resumed Claude would re-trigger "
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
                print()

            # ── Phase 2: SOFT (25%) — gentle, no reload (file maintenance only) ──
            else:
                soft_bytes_hit = current_size >= soft_threshold_bytes
                soft_tokens_hit = (
                    soft_threshold_tokens is not None
                    and current_tokens is not None
                    and current_tokens >= soft_threshold_tokens
                )
                if soft_bytes_hit or soft_tokens_hit:
                    soft_prune_count += 1
                    reason = f"{current_tokens:,} tokens >= {soft_threshold_tokens:,} (25%)" if soft_tokens_hit else f"{current_size / 1024 / 1024:.1f}MB"
                    print(f"  [{_now()}] SOFT THRESHOLD (25%): {reason}")
                    print(f"  Gentle file cleanup, no reload (cycle #{soft_prune_count})...")

                    result = guard_prune_cycle(
                        session_path=session_path,
                        rx_name="gentle",
                        config=config,
                        auto_reload=False,
                        cwd=cwd or os.getcwd(),
                        session_id=sess["session_id"],
                    )

                    print(f"  Trimmed: {_fmt_prune_result(result)}")
                    print()

    except KeyboardInterrupt:
        # Final checkpoint before exit
        checkpoint_team(session_path=session_path, quiet=True)
        total_prunes = prune_count + soft_prune_count
        if total_prunes:
            print(f"\n  Guard stopped. Pruned {total_prunes}x during this session.")
        else:
            print(f"\n  Guard stopped.")
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


def guard_prune_cycle(
    session_path: Path,
    rx_name: str = "standard",
    config: dict | None = None,
    auto_reload: bool = True,
    cwd: str = "",
    session_id: str | None = None,
    claude_pid: int | None = None,
    force: bool = False,
) -> dict:
    """Execute a single guard prune cycle.

    Holds a _PruneLock for the duration so concurrent guard instances cannot
    race each other.  Takes a _FileSnapshot before loading so that any lines
    Claude appends while pruning is in progress are preserved in the output
    (or the cycle is deferred on conflict).

    ``force=True`` overrides the active-session idle guard — used by the
    reactive overflow recovery path (the 90% emergency case where pruning an
    active session IS the intent).

    Returns dict with: saved_mb, team_name, team_messages, reloading, checkpoint_path
    """
    from .tokens import estimate_session_tokens, calibrate_ratio
    from .safety import (
        ActiveSessionError,
        assert_session_idle_or_force,
        resolve_min_idle_hours,
    )

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

    # P0-A: active-session idle guard. Runs BEFORE _PruneLock acquisition so a
    # refused cycle does not block other consumers of the lock. K-counter
    # does NOT advance on refusal (the active_session_refused key tells the
    # daemon's outer loop to skip the K increment).
    try:
        assert_session_idle_or_force(
            session_path,
            min_idle_hours=resolve_min_idle_hours(),
            force=force,
        )
    except ActiveSessionError as exc:
        print(
            f"  [{_now()}] Prune refused — session active "
            f"(mtime {exc.idle_minutes:.1f} min ago, threshold "
            f"{exc.threshold_hours}h). Use --force to override.",
            file=sys.stderr,
        )
        return {**_no_change, "active_session_refused": True}

    try:
        with _PruneLock(session_path):
            # Snapshot before load so we can detect Claude appending mid-prune
            snap = snapshot_session(session_path)

            # Size guard: skip prune for very large sessions (OOM risk #74)
            file_size_mb = session_path.stat().st_size / 1024 / 1024
            if file_size_mb > 200:
                print(f"  [{_now()}] Session {file_size_mb:.0f}MB exceeds 200MB — skipping prune (OOM risk).", file=sys.stderr)
                return _no_change

            messages = load_messages(session_path)
            original_bytes = sum(b for _, _, b in messages)

            # Token estimate before pruning — capture calibrated ratio before metadata-strip
            pre_te = estimate_session_tokens(messages)
            pre_ratio = calibrate_ratio(messages)

            # Prune with team protection
            pruned_messages, results, team_state = prune_with_team_protect(
                messages, rx_name=rx_name, config=config,
            )

            final_bytes = sum(b for _, _, b in pruned_messages)
            saved_bytes = original_bytes - final_bytes

            # If pruning freed nothing (or grew the file via team recovery injection), don't
            # save — avoids backup accumulation and file growth on ineffective prescriptions (#16, #19).
            if saved_bytes <= 0:
                return {
                    "saved_mb": 0.0,
                    "original_tokens": pre_te.total,
                    "final_tokens": pre_te.total,
                    "team_name": team_state.team_name,
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
                # Still write checkpoint (so user can recover team state via the
                # checkpoint path surfaced in the diagnostic) and save messages
                # (prune output is valid — it just won't save enough to avoid
                # immediate re-trigger).
                checkpoint_path = None
                if not team_state.is_empty():
                    project_dir = session_path.parent
                    checkpoint_path = write_team_checkpoint(team_state, project_dir)
                backup = save_messages(session_path, pruned_messages, create_backup=True, snapshot=snap)
                if backup:
                    cleanup_old_backups(session_path, keep=3)
                return {
                    "saved_mb": saved_bytes / 1024 / 1024,
                    "original_bytes": original_bytes,
                    "original_tokens": pre_te.total,
                    "final_tokens": pre_te.total,  # post_te not computed (early return)
                    "team_name": team_state.team_name,
                    "team_messages": team_state.message_count,
                    "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                    "backup_path": str(backup) if backup else None,
                    "reloading": False,
                    "futile_reload_skipped": True,
                }

            # Token estimate after pruning — pass pre-calibrated ratio
            post_te = estimate_session_tokens(pruned_messages, pre_calibrated_ratio=pre_ratio)

            # Write checkpoint if team exists
            checkpoint_path = None
            if not team_state.is_empty():
                project_dir = session_path.parent
                checkpoint_path = write_team_checkpoint(team_state, project_dir)

            # Save pruned session — snapshot enables append-aware atomic write
            backup = save_messages(session_path, pruned_messages, create_backup=True, snapshot=snap)

            # Cap backup retention at 3 files to prevent disk fill (#19)
            if backup:
                cleanup_old_backups(session_path, keep=3)

    except PruneLockError as exc:
        print(f"  [{_now()}] Prune deferred — lock held: {exc}", file=sys.stderr)
        return _no_change
    except PruneConflictError as exc:
        print(f"  [{_now()}] Prune deferred — conflict detected: {exc}", file=sys.stderr)
        return _no_change

    # Track lifetime savings
    tokens_saved = pre_te.total - post_te.total if pre_te.total and post_te.total else 0
    if tokens_saved > 0:
        from .helpers import record_savings, get_msg_type
        turn_count = sum(1 for _, m, _ in messages
                       if get_msg_type(m) == "user"
                       and isinstance(m.get("message", {}).get("content", ""), str))
        record_savings(tokens_saved, total_tokens=pre_te.total, turn_count=turn_count)

    result = {
        "saved_mb": saved_bytes / 1024 / 1024,
        "original_tokens": pre_te.total,
        "final_tokens": post_te.total,
        "team_name": team_state.team_name,
        "team_messages": team_state.message_count,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "backup_path": str(backup) if backup else None,
        "reloading": False,
    }

    # Trigger reload if configured — terminate Claude then auto-resume.
    # Wave 2: acquire the single-flight reload lock before spawning the
    # watcher. If another process (manual `cozempic reload`, overflow
    # recovery, or another guard daemon instance) is already in the middle
    # of a reload pipeline, defer ours — the prune itself completed and
    # the user already has the saved state. Next cycle (or next user-
    # initiated event) will re-trigger if conditions still warrant.
    if auto_reload:
        reload_pid = claude_pid if claude_pid is not None else find_claude_pid()
        if reload_pid:
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
                    )
                    result["reloading"] = True
            except ReloadLockHeld as exc:
                # Another reload pipeline is already in flight — defer.
                # Prune output is already saved; the in-flight pipeline
                # will do the kill+resume.
                print(
                    f"  Reload deferred — another pipeline in flight "
                    f"({exc.holder_initiator}, PID {exc.holder_pid})."
                )
                result["reloading"] = False
        else:
            resume_flag = f"--resume {session_id}" if session_id else "--resume"
            print("  WARNING: Could not find Claude PID. Pruned but not reloading.")
            print(f"  Restart manually: claude {resume_flag}")

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
        # Escape cmd.exe metacharacters in project_dir so they cannot execute.
        # ^ is the cmd.exe escape character; prefix each metachar with ^ to
        # prevent them from being interpreted as shell operators.
        _cmd_metachars = set('&|<>^"')
        escaped_dir = "".join(f"^{c}" if c in _cmd_metachars else c for c in project_dir)
        resume_cmd = (
            f"start cmd /c \"cd /d {escaped_dir} && claude {resume_flag}\""
        )
        # Use escaped form in log line too so the watcher_script has no raw metachars
        log_dir = escaped_dir
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

    # Compose the sentinel unlink fragment (empty string when no session_id)
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
    from ._validation import ConfigError

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
        # No session_id — detect from CWD (backward compat with old hooks)
        sess = find_current_session(cwd)
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
                os.rename(str(tmp_path), str(pid_path))
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
) -> dict:
    """Gracefully restart the running guard daemon for this session.

    Used after an in-place cozempic upgrade so the daemon picks up the new code
    on disk. SIGTERMs the existing daemon (it writes a final checkpoint via the
    SIGTERM handler), waits for it to exit, then spawns a fresh daemon with the
    same args. The new daemon imports from the freshly-installed package files.

    Returns dict: {reloaded: bool, old_pid, new_pid, log_file, reason}.
    """
    cwd = cwd or os.getcwd()

    if not session_id:
        sess = find_current_session(cwd)
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
        tok_str = f"{saved_tok / 1000:.1f}K" if saved_tok >= 1000 else str(saved_tok)
        pct = f"{saved_tok / orig_tok * 100:.1f}%" if orig_tok > 0 else "0%"
        return f"{tok_str} tokens freed ({pct}), {result['saved_mb']:.1f}MB saved"
    return f"{result['saved_mb']:.1f}MB saved"


def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")
