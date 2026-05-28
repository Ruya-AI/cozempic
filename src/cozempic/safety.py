"""Safety guards for the session pruner.

Implements the four protections introduced by the resume-break fix
(see PR description and PLAN.md):

P0-A — Active-session idle guard (this commit):
  - `is_session_idle(path, min_idle_hours)`
  - `assert_session_idle_or_force(path, min_idle_hours, force)`
  - `ActiveSessionError` exception with path / idle_minutes / threshold_hours
  - `resolve_min_idle_hours()` env-aware threshold resolver

P0-B — Post-prune structural validation (commit 2):
  - `validate_post_prune(msgs_before, msgs_after, strict)`
  - `PruneValidationError` exception with reason + evidence dict
  - `simulate_replay_readiness(messages)` structural replay probe

P0-C — Floor preservation (commit 3):
  - `FloorConfig` dataclass (re-exported from config.py for ergonomics)
  - `enforce_floor(msgs_before, msgs_after, cfg)`

P0-D — Last-of-type metadata singleton protection (commit 4):
  - Helpers used by executor; see `executor._tag_last_of_metadata_types`.
"""

from __future__ import annotations

from pathlib import Path

from .config import FloorConfig, resolve_min_idle_hours

__all__ = [
    "ActiveSessionError",
    "PruneValidationError",
    "FloorConfig",
    "is_session_idle",
    "assert_session_idle_or_force",
    "resolve_min_idle_hours",
    "validate_post_prune",
    "enforce_floor",
    "simulate_replay_readiness",
]


# ── P0-A — Active-session idle guard ───────────────────────────────────────


class ActiveSessionError(Exception):
    """Raised when a prune is blocked by the active-session guard.

    Carries enough context for the operator to act:
      - ``path``: the session JSONL that was rejected
      - ``idle_minutes``: how long since the file was last modified
      - ``threshold_hours``: the active threshold at refusal time

    The string representation is suitable for direct emission to stderr or
    CLI exit-code 4 paths.
    """

    def __init__(self, path: Path, idle_minutes: float, threshold_hours: float):
        self.path = path
        self.idle_minutes = idle_minutes
        self.threshold_hours = threshold_hours
        super().__init__(
            f"Session {path.name} was modified {idle_minutes:.1f} min ago "
            f"(< {threshold_hours}h idle threshold). Refusing prune. "
            f"Use --force to override or wait for the session to idle."
        )


def is_session_idle(path: Path, min_idle_hours: float) -> tuple[bool, float]:
    """Return ``(is_idle, minutes_since_last_mtime)``.

    ``is_idle`` is ``True`` when the file's last modification predates
    ``now - min_idle_hours``. ``minutes_since_last_mtime`` is the elapsed
    minutes since ``stat().st_mtime``.

    A missing file is treated as idle (nothing to protect) and reports
    ``minutes`` as ``inf`` so callers can log a meaningful value.
    """
    import time

    try:
        mtime = path.stat().st_mtime
    except (FileNotFoundError, OSError):
        return True, float("inf")

    elapsed_seconds = max(0.0, time.time() - mtime)
    elapsed_minutes = elapsed_seconds / 60.0
    threshold_minutes = float(min_idle_hours) * 60.0
    return elapsed_minutes >= threshold_minutes, elapsed_minutes


def assert_session_idle_or_force(
    path: Path,
    *,
    min_idle_hours: float,
    force: bool,
) -> None:
    """Raise ``ActiveSessionError`` if the session is not idle and ``force`` is False.

    ``min_idle_hours`` of 0.0 disables the guard entirely (every session
    counts as idle). ``force=True`` is the explicit operator override; the
    overflow recovery path uses it for its emergency prune.
    """
    if force:
        return
    if min_idle_hours <= 0.0:
        return
    idle, minutes = is_session_idle(path, min_idle_hours)
    if not idle:
        raise ActiveSessionError(
            path=path,
            idle_minutes=minutes,
            threshold_hours=float(min_idle_hours),
        )


# ── P0-B — Post-prune validation (implemented in commit 2) ─────────────────


class PruneValidationError(Exception):
    """Raised when the pruned message list fails structural validation.

    ``reason`` is a human-readable summary. ``evidence`` is a dict that
    callers (guard daemon, CLI) log; it always contains a ``failed_check``
    key matching one of ``"C1".."C6"`` so log aggregators can group failures.
    """

    def __init__(self, reason: str, evidence: dict):
        self.reason = reason
        self.evidence = evidence
        super().__init__(f"Pruned session would not replay cleanly: {reason}")


def validate_post_prune(
    msgs_before: list[tuple[int, dict, int]],
    msgs_after: list[tuple[int, dict, int]],
    *,
    strict: bool = True,
) -> None:
    """Validate the pruned message list. Implementation lands in commit 2."""
    raise NotImplementedError(
        "validate_post_prune is implemented in commit 2 (P0-B)"
    )


def simulate_replay_readiness(
    messages: list[tuple[int, dict, int]],
) -> tuple[bool, str]:
    """Structural replay probe. Implementation lands in commit 2."""
    raise NotImplementedError(
        "simulate_replay_readiness is implemented in commit 2 (P0-B)"
    )


# ── P0-C — Floor preservation (implemented in commit 3) ────────────────────


def enforce_floor(
    msgs_before: list[tuple[int, dict, int]],
    msgs_after: list[tuple[int, dict, int]],
    *,
    cfg: FloorConfig,
) -> list[tuple[int, dict, int]]:
    """Re-add must-preserve messages dropped by strategies. Implemented in commit 3."""
    raise NotImplementedError(
        "enforce_floor is implemented in commit 3 (P0-C)"
    )
