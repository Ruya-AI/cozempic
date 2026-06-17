"""Guard-loop watchdog — a process safeguard against futile reload-loops.

History: a guard daemon reload-looped 202x on an UNPRUNABLE over-threshold session
(f641174c, 2026-06-10) and again in the PilotCC incident (deferred-writer failure).
Each was a CODE bug fixed in-process (1.8.29, 1.8.19). But code fixes only protect
daemons running the new code — an OLD daemon (e.g. a brew-1.8.22 install still
resident) keeps looping until killed by hand. And the QA process reviews a DIFF,
not the SYSTEM running over time, so an emergent loop is invisible to it.

This watchdog closes that gap from the OUTSIDE: it reads the guard log files the
daemons already write and flags any that show the full-speed futile-loop signature
— many near-0% prunes with NO escalating back-off and NO circuit-breaker exit. The
trip threshold sits well ABOVE the in-process K-exit (10), so a CORRECTLY behaving
1.8.29+ guard (which K-exits or backs off to the 300s cap then exits) never trips
it; only a daemon that fails to self-arrest does.

Pure detection lives in ``scan_log_text`` (fully unit-testable on captured/synthetic
log text). ``scan_guard_logs`` adds filesystem + pid-liveness. The CLI reports by
default and only terminates a confirmed-looping daemon under an explicit ``--fix``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .helpers import _pid_is_alive as _pid_alive

# A prune freeing less than this percentage of tokens is "futile" (mirrors the
# guard's own _MIN_PRUNE_RATIO=0.10 intent; we use 1% as the unambiguous
# "barely moved" floor so a marginal 2-3% prune isn't counted as a stuck loop).
FUTILE_PCT_FLOOR = 1.0

# Number of futile prune cycles that constitutes a stuck loop. Set ABOVE the
# in-process K-exit threshold (HARD_LOOP_EXIT_THRESHOLD=10) so a single HEALTHY
# 1.8.29+ daemon never trips it: the agentless reload path K-exits at exactly 10
# futile cycles, and the agents-active path emits "Read-only — live session not
# rewritten" lines (NOT "Pruned: …" lines), so it never accrues futile prune
# cycles at all. Reaching >= 20 means EITHER a single daemon that failed to
# self-arrest OR — the REAL f641174c shape — a RESPAWN STORM: the SessionStart
# hook respawning a fresh guard onto a permanently-unprunable session over and
# over, each run dutifully K-exiting yet churning reloads. (Validated against the
# captured real log: 23 daemon starts, 216 futile prunes, 21 K-exits.)
LOOP_TRIP_DEFAULT = 20

# Fraction of prune cycles that must be futile for the log to count as churn.
# A daemon doing mostly REAL prunes (freeing tokens) is healthy even if a few
# cycles are marginal.
FUTILE_DOMINANCE = 0.8

# >= this many daemon-start markers in one log == a respawn storm (vs a single
# stuck daemon). Affects only the operator-facing reason wording.
STORM_TRIP = 5

# The guard's back-off cap (HARD_LOOP_BACKOFF_CAP), recorded for diagnostics.
BACKOFF_CAP_S = 300

# Rate-based storm detection: time window (seconds) within which daemon restarts
# are counted. A storm = many restarts close together. 3600s (1h) separates
# "storming right now" from "user restarting Claude across a day". The daemon's
# own back-off cap is BACKOFF_CAP_S=300; a storm spanning >1h is either arrested
# by the daemon's own circuit-breaker or has been running long enough that
# flagging it is correct.
RATE_WINDOW_S: int = 3600

# Minimum daemon restarts within RATE_WINDOW_S to declare an error respawn storm.
# Set to 5, consistent with the existing STORM_TRIP constant for futile-prune storms.
# Five escalation-respawn cycles happen in < 5 min during a real error loop; user
# restarts across a day don't cluster within a 1h window.
RATE_STORM_TRIP: int = 5

# The token-count group must tolerate the K/M/G suffix and comma grouping that
# guard._fmt_prune_result emits ("210.0K tokens freed", "1.2M tokens freed") for
# any prune >= 1000 tokens — WITHOUT this, every productive prune line is
# unparsed, so the futile-dominance ratio skews to ~1.0 and the watchdog
# FALSE-FLAGS a healthy daemon as looping (and --fix would SIGTERM it). The count
# is matched but NOT captured (the percent is the only capture group, group 1, and
# the only value consumed downstream), so the K/M/comma suffix only needs tolerating.
# Anchored to line-start (re.M, optional leading whitespace) — the guard emits its
# real prune line as "  Pruned: …" at the START of a log line. Anchoring stops a
# "Pruned: 0 tokens freed (0.0%)" substring embedded MID-line (e.g. inside a
# Team '<attacker-name>' state preserved log line) from forging a futile-prune
# match and false-tripping the watchdog (C7 log-injection — the residual the
# newline-only _log_safe scrub didn't cover).
_PRUNED_RE = re.compile(
    r"^\s*Pruned:\s+[0-9][0-9,]*(?:\.[0-9]+)?[KMG]?\s+tokens freed\s+\(([0-9.]+)%\)",
    re.IGNORECASE | re.MULTILINE,
)
_BACKOFF_RE = re.compile(r"back-off \(next sleep:\s*(\d+)s", re.IGNORECASE)
# Line-anchored to prevent mid-line forged "Guard daemon started at <ISO>" substrings
# (e.g. inside a "Team '<attacker-name>' state preserved" log line) from registering
# as daemon-start markers — same defence class as the _PRUNED_RE anchor (C7 log-injection
# that the newline-only _log_safe scrub doesn't cover). The real guard format is
# "--- Guard daemon started at <ISO> ---" which starts a new line; the leading "--- "
# (dashes + space) must be matched explicitly. Optional leading whitespace covers
# any compact variant. The pattern covers both forms:
#   ^---\s*Guard daemon started …   (real format with dashes)
#   ^\s*Guard daemon started …       (whitespace-only prefix variant)
# A mid-line occurrence like "Team 'Guard daemon started at T' state preserved"
# cannot match because "Team '" is neither dashes nor whitespace at line-start.
_DAEMON_START_RE = re.compile(
    r"^(?:---\s*|\s*)Guard daemon started(?:\s+at\s+([\d\-T:.Z+-]+))?",
    re.IGNORECASE | re.MULTILINE,
)
# Circuit-breaker / daemon-exit markers (recorded for diagnostics — NOT treated
# as proof of health: the real f641174c storm K-exited 21x and still looped).
_EXIT_RE = re.compile(
    r"(circuit breaker|K=\d+\s*>=|exiting guard|guard powerless|hard-cap exit|"
    r"giving up|consecutive empty|reload-loop)",
    re.IGNORECASE,
)
# C7: the inert/erroring-guard signature emits NO "Pruned:" line — a per-cycle
# exception spinner logs "skipping a cycle after an unexpected error", and the
# C2 escalation logs "cycle-error escalation" before exiting for respawn. The
# watchdog must SEE these (it was previously blind to any non-futile-prune loop).
_CYCLE_ERR_RE = re.compile(r"skipping a cycle after an unexpected error", re.IGNORECASE)
_CYCLE_ESCALATION_RE = re.compile(r"cycle-error escalation", re.IGNORECASE)


@dataclass
class LoopReport:
    """Result of scanning one guard log's text."""
    total_prune_cycles: int = 0
    futile_cycles: int = 0
    daemon_starts: int = 0
    cycle_errors: int = 0
    cycle_escalations: int = 0
    max_backoff_s: int = 0
    has_backoff: bool = False
    has_exit: bool = False
    looping: bool = False
    reason: str = ""
    recent_pcts: list = field(default_factory=list)
    daemon_start_times: list = field(default_factory=list)  # datetime | None per start
    recent_starts: int = 0  # starts within RATE_WINDOW_S of the anchor timestamp


def scan_log_text(text: str, loop_trip: int = LOOP_TRIP_DEFAULT) -> LoopReport:
    """Detect guard reload-churn or error-storm in one guard log's text (pure).

    Two detection paths:

    **Futile-prune path**: flags ``looping`` when >= ``loop_trip`` futile prune
    cycles (<1% freed) dominate all prune cycles (>= ``FUTILE_DOMINANCE``). This
    is the f641174c respawn-storm shape. Crucially does NOT treat a circuit-breaker
    exit as proof of health — each daemon in a respawn storm DID K-exit, yet the
    SessionStart hook kept respawning onto an unprunable session.

    **C7 error-storm path (rate-based)**: when ≥1 parseable ISO timestamp exists in
    daemon-start header lines, PATH A is authoritative. Two sub-verdicts:

    * **Storm** (``recent_starts >= RATE_STORM_TRIP``): the maximum number of daemon
      restarts that fall within ANY ``RATE_WINDOW_S``-second sliding window reaches the
      trip threshold. Uses a sliding-window algorithm (sort + per-point count) so a
      single outlier timestamp (forged future or far-past) forms its own window of size 1
      and cannot mask a genuine restart cluster.
    * **Inert/escalating** (``recent_starts < RATE_STORM_TRIP`` but
      ``cycle_escalations >= 1`` and ``cycle_errors >= loop_trip`` and
      ``cycle_errors > productive_prunes``): a single daemon hit a deterministic failure,
      accumulated many per-cycle errors, and exited via the C2 escalation path. Not a
      storm but genuinely stuck.

    When no parseable timestamps exist, the older flat-count fallback (PATH B) runs
    unchanged — zero regression for old-format logs.
    """
    rep = LoopReport()
    futile = 0
    for m in _PRUNED_RE.finditer(text):
        rep.total_prune_cycles += 1
        try:
            pct = float(m.group(1))
        except (TypeError, ValueError):
            continue
        rep.recent_pcts.append(pct)
        if pct < FUTILE_PCT_FLOOR:
            futile += 1
    rep.futile_cycles = futile
    rep.recent_pcts = rep.recent_pcts[-loop_trip:]

    # Parse daemon-start markers and attempt to extract ISO timestamps from each.
    # The capture group is optional — lines without "at <ISO>" yield group(1)=None.
    # Tz-aware timestamps (guard emits naive local time via datetime.now().isoformat(),
    # so a tz-aware result means a forged/unexpected format) are treated as None: mixing
    # tz-aware + naive datetimes in comparisons raises TypeError.
    for m in _DAEMON_START_RE.finditer(text):
        rep.daemon_starts += 1
        raw_ts = m.group(1)
        dt: datetime | None = None
        if raw_ts is not None:
            try:
                parsed = datetime.fromisoformat(raw_ts)
                dt = parsed if parsed.tzinfo is None else None
            except ValueError:
                dt = None
        rep.daemon_start_times.append(dt)

    # Compute recent_starts: maximum number of daemon restarts that fall within any
    # sliding RATE_WINDOW_S-second window of each other. Uses a sort + O(n²) inner
    # count (n = daemon_starts, typically < 50) rather than max()-anchoring, which
    # is defeated by a single forged far-future or far-past timestamp inflating the
    # anchor: a lone outlier forms its own window of size 1 and can't suppress
    # legitimate clusters.
    # If no parseable timestamps exist, recent_starts stays 0 → rate path skipped,
    # flat-count fallback (PATH B) handles old-format logs with no ISO timestamps.
    parseable = [dt for dt in rep.daemon_start_times if dt is not None]
    if parseable:
        sorted_ts = sorted(parseable)
        rep.recent_starts = max(
            sum(
                1 for t2 in sorted_ts
                if 0 <= (t2 - t1).total_seconds() <= RATE_WINDOW_S
            )
            for t1 in sorted_ts
        )

    backoffs = [int(s) for s in _BACKOFF_RE.findall(text)]
    if backoffs:
        rep.has_backoff = True
        rep.max_backoff_s = max(backoffs)
    rep.has_exit = bool(_EXIT_RE.search(text))

    # C7: erroring/inert-guard signature. Two-path verdict (rate-first, flat-fallback):
    #
    # PATH A — rate-based (implemented here, fixes the known FN/FP residuals):
    #   When ≥1 parseable ISO timestamp exists in daemon-start markers, count how many
    #   restarts fall within RATE_WINDOW_S of the most recent one. If that count
    #   reaches RATE_STORM_TRIP the log shows an error respawn storm regardless of
    #   what older productive-prune lines say. This is the "durable fix" referenced in
    #   the KNOWN RESIDUAL comment below (now implemented).
    #     FN fix: a current error storm is flagged even when stale productive-prune
    #       lines from an earlier dead generation dominate the window.
    #     FP fix: a healthy single-daemon with scattered recovered errors has
    #       recent_starts=1 < RATE_STORM_TRIP → NOT flagged.
    #   When recent_starts < RATE_STORM_TRIP (and ≥1 parseable timestamp exists), the
    #   rate path is authoritative — the flat-count fallback is suppressed.
    #
    # PATH B — flat-count fallback (original R15 discriminator):
    #   Used only when 0 parseable timestamps exist (old daemon versions, truncated
    #   lines). Behaviour for those logs is IDENTICAL to before — zero regression.
    #   Discriminator:  cycle_errors >= loop_trip  AND  cycle_errors > productive_prunes
    #
    rep.cycle_errors = len(_CYCLE_ERR_RE.findall(text))
    rep.cycle_escalations = len(_CYCLE_ESCALATION_RE.findall(text))
    _productive_prunes = rep.total_prune_cycles - rep.futile_cycles

    if parseable:
        # PATH A: rate-based verdict. Authoritative when any ISO timestamp was parsed.
        if rep.recent_starts >= RATE_STORM_TRIP:
            rep.looping = True
            rep.reason = (
                f"{rep.recent_starts} guard restarts in the last "
                f"{RATE_WINDOW_S // 60}min — error respawn storm (rate-based); "
                f"{rep.cycle_errors} per-cycle errors / "
                f"{rep.cycle_escalations} escalations"
            )
            return rep
        # recent_starts < RATE_STORM_TRIP with parseable timestamps: not a storm.
        # Check for the inert/escalating-single-daemon case: a daemon that hit a
        # deterministic failure, emitted ≥loop_trip per-cycle errors, and exited
        # via the C2 escalation path (cycle_escalations ≥ 1) for operator respawn.
        # This is NOT a respawn storm (recent_starts is small) but IS genuinely
        # stuck — the escalation proves it tried and failed repeatedly, not just
        # a transient blip that recovered. Without this gate, PATH A's flat-count
        # suppression would mask single-gen inert daemons.
        #
        # DEFERRED RESIDUAL: a daemon that errors ≥loop_trip times but NEVER
        # escalates (C2 escalation path was disabled or the loop exited via a
        # different path) is not caught by this discriminator. That shape is rare
        # (C2 escalation fires before K=loop_trip in normal operation) and is
        # already handled by PATH B for no-timestamp logs. Tracking in TODO.md.
        if (
            rep.cycle_escalations >= 1
            and rep.cycle_errors >= loop_trip
            and rep.cycle_errors > _productive_prunes
        ):
            rep.looping = True
            rep.reason = (
                f"inert/erroring guard (escalated): {rep.cycle_errors} per-cycle errors / "
                f"{rep.cycle_escalations} escalations vs {_productive_prunes} productive prunes "
                f"— genuinely stuck (not a recovered transient)"
            )
            return rep
    else:
        # PATH B: flat-count fallback — no parseable timestamps (old log format).
        # Original R15 discriminator: erroring more than productively pruning.
        if rep.cycle_errors >= loop_trip and rep.cycle_errors > _productive_prunes:
            rep.looping = True
            rep.reason = (
                f"{rep.cycle_errors} per-cycle errors / {rep.cycle_escalations} escalations / "
                f"{rep.daemon_starts} restarts vs {_productive_prunes} space-freeing prunes "
                f"— guard is erroring more than it is productively pruning (inert or "
                f"respawn-cycling on a deterministic failure); investigate the logged exception"
            )
            return rep

    futile_ratio = futile / rep.total_prune_cycles if rep.total_prune_cycles else 0.0
    if futile >= loop_trip and futile_ratio >= FUTILE_DOMINANCE:
        rep.looping = True
        if rep.daemon_starts >= STORM_TRIP:
            rep.reason = (
                f"respawn storm: {rep.daemon_starts} guard restarts churning "
                f"{futile} futile prune cycles (<{FUTILE_PCT_FLOOR:.0f}% freed, "
                f"{futile_ratio*100:.0f}% of all prunes) — SessionStart keeps "
                f"respawning a guard onto an unprunable session"
                + ("; each run K-exits yet the storm continues" if rep.has_exit else "")
            )
        else:
            rep.reason = (
                f"{futile} futile prune cycles (<{FUTILE_PCT_FLOOR:.0f}% freed, "
                f"{futile_ratio*100:.0f}% of all prunes) "
                + ("with no circuit-breaker exit " if not rep.has_exit else "")
                + "— daemon is reload-looping on an unprunable session"
            )
    return rep


@dataclass
class GuardLoopHit:
    log_file: Path
    pid_file: Path | None
    pid: int | None
    pid_alive: bool
    report: LoopReport
    guard_confirmed: bool = False  # True iff pid_alive AND process is a cozempic guard


def _read_pid(pid_file: Path) -> int | None:
    try:
        first = pid_file.read_text(encoding="utf-8").strip().splitlines()[0]
        return int(first.strip())
    except (OSError, ValueError, IndexError):
        return None


def scan_guard_logs(
    log_dir: str | Path,
    loop_trip: int = LOOP_TRIP_DEFAULT,
    max_tail_bytes: int = 256 * 1024,
) -> list[GuardLoopHit]:
    """Scan every ``cozempic_guard_*.log`` under ``log_dir`` for stuck loops.

    Returns one ``GuardLoopHit`` per log whose tail shows the loop signature.
    The paired ``cozempic_guard_*.pid`` (if present) is read so a caller can tell
    a LIVE stuck daemon (actionable) from a dead one's stale log (already gone).
    """
    log_dir = Path(log_dir)
    hits: list[GuardLoopHit] = []
    if not log_dir.is_dir():
        return hits
    for log_file in sorted(log_dir.glob("cozempic_guard_*.log")):
        try:
            size = log_file.stat().st_size
            with open(log_file, "r", encoding="utf-8", errors="replace") as fh:
                if size > max_tail_bytes:
                    fh.seek(size - max_tail_bytes)
                    fh.readline()  # discard partial line
                text = fh.read()
        except OSError:
            continue
        rep = scan_log_text(text, loop_trip=loop_trip)
        if not rep.looping:
            continue
        pid_file = log_file.with_suffix(".pid")
        pid = _read_pid(pid_file) if pid_file.exists() else None
        alive = _pid_alive(pid)
        if alive:
            # Lazy import: avoids the heavy module-level guard.py load (large,
            # side-effecty) and prevents a circular import (guard imports watchdog
            # indirectly through its own helpers). alive=True implies pid is not
            # None (since _pid_alive(None) returns False).
            from .guard import _is_cozempic_guard_process
            confirmed = _is_cozempic_guard_process(pid)
        else:
            confirmed = False
        hits.append(GuardLoopHit(
            log_file=log_file,
            pid_file=pid_file if pid_file.exists() else None,
            pid=pid,
            pid_alive=alive,
            report=rep,
            guard_confirmed=confirmed,
        ))
    return hits
