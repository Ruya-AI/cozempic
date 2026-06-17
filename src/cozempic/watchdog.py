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

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

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
_DAEMON_START_RE = re.compile(r"Guard daemon started", re.IGNORECASE)
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


def scan_log_text(text: str, loop_trip: int = LOOP_TRIP_DEFAULT) -> LoopReport:
    """Detect futile reload-churn in one guard log's text (pure).

    Flags ``looping`` when the log shows >= ``loop_trip`` futile prune cycles
    (<1% freed) AND those dominate the prune cycles (>= ``FUTILE_DOMINANCE``).
    Crucially this does NOT treat a circuit-breaker exit as proof of health —
    the real f641174c failure was a RESPAWN STORM in which each daemon DID
    K-exit, so "saw an exit line" is recorded for diagnostics but never clears
    the verdict. The agents-active deferral path emits read-only-checkpoint lines
    (not "Pruned: …"), so a busy-but-healthy session accrues no futile prune
    cycles and is never flagged.
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
    rep.daemon_starts = len(_DAEMON_START_RE.findall(text))

    backoffs = [int(s) for s in _BACKOFF_RE.findall(text)]
    if backoffs:
        rep.has_backoff = True
        rep.max_backoff_s = max(backoffs)
    rep.has_exit = bool(_EXIT_RE.search(text))

    # C7: erroring/inert-guard signature — the guard is erroring MORE than it is
    # productively pruning. Discriminator (R15, after three weaker attempts each leaked):
    #   cycle_errors >= loop_trip  AND  cycle_errors > productive_prunes
    # where productive_prunes = total_prune_cycles - futile_cycles (prunes that actually
    # freed space). This is robust in BOTH directions on a flat append-mode log:
    #  - HEALTHY busy guard (many productive prunes, occasional transient error):
    #    cycle_errors <= productive_prunes -> NOT flagged.
    #  - HEALTHY IDLE guard (short sessions, 0 prunes, 0 errors, several restarts):
    #    cycle_errors (0) < loop_trip -> NOT flagged (the daemon_starts-only trigger of
    #    the prior attempt false-flagged this — R15 FP).
    #  - HEALTHY fresh gen + a couple of STALE escalations from dead gens (~10 errors):
    #    < loop_trip -> NOT flagged (R15 FP).
    #  - ERROR respawn storm, even WITH a stray early productive prune line in the tail
    #    (the R15 FN that defeated the total_prune_cycles==0 gate): many errors greatly
    #    outnumber the lone productive prune -> flagged.
    #  - single INERT generation (>= loop_trip errors, 0 prunes) -> flagged.
    # The futile-PRUNE storm (f641174c: many <1%-freed prunes, ~0 errors) has
    # productive_prunes near 0 but cycle_errors near 0 too, so it is owned by the
    # separate futile-dominance branch below (which keeps the "respawn storm" reason).
    #
    # KNOWN RESIDUAL (accepted, REPORT-ONLY; tracked follow-up = a rate-based redesign):
    # because this counts over a flat 256KB append-mode tail with no recency/rate
    # awareness, two edge cases survive — (FN) a CURRENT error-storm can be masked if
    # stale productive-prune lines from an earlier dead generation still in the window
    # outnumber the errors; (FP) a long healthy daemon that self-recovered >= loop_trip
    # SCATTERED transient errors (counter reset each time, never escalated) with its
    # productive prunes scrolled out of the window can be flagged. Both are bounded:
    # this is a REPORT-ONLY monitor (--fix is manual + guard-identity-gated), the
    # daemon's OWN circuit-breaker self-arrests real error-storms, and the well-tested
    # futile-PRUNE loop detection (the real f641174c incident shape) is unaffected. The
    # durable fix is to parse the per-line timestamps and key the verdict on RESTART
    # RATE within a recent window — deferred to a follow-up (not blocking PR #138).
    rep.cycle_errors = len(_CYCLE_ERR_RE.findall(text))
    rep.cycle_escalations = len(_CYCLE_ESCALATION_RE.findall(text))
    _productive_prunes = rep.total_prune_cycles - rep.futile_cycles
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


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except (ProcessLookupError, OverflowError, ValueError, OSError):
        return False


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
