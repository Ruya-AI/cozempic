"""Runtime configuration for cozempic safety guards.

Single source of truth for the tunables introduced by the session-pruner
resume-break fix (PR — session pruner resumable state):

  - `min_idle_hours`: how long a session must be untouched before the guard
    daemon and `cozempic treat`/`reload` will prune it without `--force`.
  - `floor`: per-prune protections — max % of user/assistant messages that
    may drop, last-K turns guaranteed to survive, first-message guarantee.

Precedence: environment variable > `~/.cozempic/config.json` > built-in default.

Invalid values (out-of-range, garbage strings, wrong type) silently fall
back to the default. Reading config never raises — a daemon mid-flight must
not crash because the operator stashed a stale env var in their shell rc.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# ── Defaults + clamps ──────────────────────────────────────────────────────

_MIN_IDLE_HOURS_DEFAULT: float = 24.0
_MIN_IDLE_HOURS_RANGE: tuple[float, float] = (0.0, 168.0)

_FLOOR_MAX_DROP_PCT_DEFAULT: float = 0.50
_FLOOR_MAX_DROP_PCT_RANGE: tuple[float, float] = (0.0, 1.0)

_FLOOR_PRESERVE_LAST_K_DEFAULT: int = 50
_FLOOR_PRESERVE_LAST_K_RANGE: tuple[int, int] = (1, 1000)

_CONFIG_FILE_PATH = Path.home() / ".cozempic" / "config.json"


@dataclass(frozen=True)
class FloorConfig:
    """Per-prune floor preservation parameters."""

    max_user_assistant_drop_pct: float = _FLOOR_MAX_DROP_PCT_DEFAULT
    preserve_last_k_turns: int = _FLOOR_PRESERVE_LAST_K_DEFAULT
    preserve_first_message: bool = True


@dataclass(frozen=True)
class Config:
    """Top-level cozempic runtime config."""

    min_idle_hours: float = _MIN_IDLE_HOURS_DEFAULT
    floor: FloorConfig = field(default_factory=FloorConfig)


# ── Loaders ────────────────────────────────────────────────────────────────


def _clamp_float(value: float, lo: float, hi: float, default: float) -> float:
    """Return value if it lies in [lo, hi] inclusive, else default.

    REVIEW-max B.2: explicitly reject NaN and infinities BEFORE the range
    check — NaN compares False to every threshold so a naive `v < lo or
    v > hi` lets it through and downstream arithmetic silently propagates.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    if v < lo or v > hi:
        return default
    return v


def _clamp_int(value: int, lo: int, hi: int, default: int) -> int:
    """Return value coerced to int and clamped to [lo, hi] inclusive.

    REVIEW-round3 F.M1: class-of-bug fold from B.2. ``int(float('inf'))``
    raises ``OverflowError`` (not in the prior except tuple) so an inf env
    var would crash the daemon at config-load time. NaN / inf string tokens
    short-circuit before conversion so the fall-back path is uniform with
    ``_clamp_float``.
    """
    # Short-circuit on string tokens that float() accepts but produce
    # non-finite values (inf, -inf, nan in any case).
    if isinstance(value, str):
        tok = value.strip().lower()
        if tok in ("inf", "+inf", "-inf", "infinity", "+infinity", "-infinity",
                   "nan", "+nan", "-nan"):
            return default
    if isinstance(value, float):
        import math
        if math.isnan(value) or math.isinf(value):
            return default
    try:
        v = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if v < lo or v > hi:
        return default
    return v


def _read_config_file() -> dict[str, Any]:
    """Read ~/.cozempic/config.json. Returns {} on any failure."""
    try:
        if not _CONFIG_FILE_PATH.exists():
            return {}
        with open(_CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_min_idle_hours_with(file_data: dict[str, Any]) -> float:
    """Resolve min_idle_hours given pre-read config file data."""
    lo, hi = _MIN_IDLE_HOURS_RANGE
    raw_env = os.environ.get("COZEMPIC_MIN_IDLE_HOURS")
    if raw_env is not None and raw_env != "":
        return _clamp_float(raw_env, lo, hi, _MIN_IDLE_HOURS_DEFAULT)
    if "min_idle_hours" in file_data:
        return _clamp_float(file_data["min_idle_hours"], lo, hi, _MIN_IDLE_HOURS_DEFAULT)
    return _MIN_IDLE_HOURS_DEFAULT


def resolve_min_idle_hours() -> float:
    """Resolve the active min_idle_hours from env → file → default.

    Precedence:
      1. ``COZEMPIC_MIN_IDLE_HOURS`` env var
      2. ``min_idle_hours`` key in ``~/.cozempic/config.json``
      3. Default (24.0)

    Invalid values (garbage strings, out of [0.0, 168.0], NaN, ±inf) at
    every layer fall back to the default — never to the lower layer. This
    avoids the surprise where a misconfigured env var resurrects a stale
    config file value.
    """
    return _resolve_min_idle_hours_with(_read_config_file())


def _resolve_floor_with(file_data: dict[str, Any]) -> FloorConfig:
    """Resolve FloorConfig given pre-read config file data."""
    floor_data = file_data.get("floor", {}) or {}
    if not isinstance(floor_data, dict):
        floor_data = {}

    # max_user_assistant_drop_pct
    raw_env = os.environ.get("COZEMPIC_FLOOR_MAX_DROP_PCT")
    if raw_env is not None and raw_env != "":
        drop_pct = _clamp_float(
            raw_env, *_FLOOR_MAX_DROP_PCT_RANGE, _FLOOR_MAX_DROP_PCT_DEFAULT,
        )
    elif "max_user_assistant_drop_pct" in floor_data:
        drop_pct = _clamp_float(
            floor_data["max_user_assistant_drop_pct"],
            *_FLOOR_MAX_DROP_PCT_RANGE,
            _FLOOR_MAX_DROP_PCT_DEFAULT,
        )
    else:
        drop_pct = _FLOOR_MAX_DROP_PCT_DEFAULT

    # preserve_last_k_turns
    raw_env = os.environ.get("COZEMPIC_FLOOR_PRESERVE_LAST_K")
    if raw_env is not None and raw_env != "":
        last_k = _clamp_int(
            raw_env, *_FLOOR_PRESERVE_LAST_K_RANGE, _FLOOR_PRESERVE_LAST_K_DEFAULT,
        )
    elif "preserve_last_k_turns" in floor_data:
        last_k = _clamp_int(
            floor_data["preserve_last_k_turns"],
            *_FLOOR_PRESERVE_LAST_K_RANGE,
            _FLOOR_PRESERVE_LAST_K_DEFAULT,
        )
    else:
        last_k = _FLOOR_PRESERVE_LAST_K_DEFAULT

    # REVIEW-round3 F.N7: preserve_first_message must read from file_data
    # and the env var override. The prior hardcoded True silently ignored
    # the user's config setting. Per E.6 prior decision (micro-session
    # carve-out), False is now a legitimate value the operator may pick.
    raw_env = os.environ.get("COZEMPIC_FLOOR_PRESERVE_FIRST")
    if raw_env is not None and raw_env != "":
        preserve_first = _parse_bool(raw_env, default=True)
    elif "preserve_first_message" in floor_data:
        preserve_first = bool(floor_data["preserve_first_message"])
    else:
        preserve_first = True

    return FloorConfig(
        max_user_assistant_drop_pct=drop_pct,
        preserve_last_k_turns=last_k,
        preserve_first_message=preserve_first,
    )


def _parse_bool(raw: str, *, default: bool) -> bool:
    """Permissive bool parse for env strings. Accepts 0/1, true/false,
    yes/no, on/off (case-insensitive). Unparseable values return ``default``."""
    tok = raw.strip().lower()
    if tok in ("1", "true", "yes", "on", "y", "t"):
        return True
    if tok in ("0", "false", "no", "off", "n", "f"):
        return False
    return default


def _resolve_floor() -> FloorConfig:
    """Backwards-compat wrapper kept for legacy callers (none in-tree)."""
    return _resolve_floor_with(_read_config_file())


def load_config() -> Config:
    """Load the active runtime config (env → file → default).

    REVIEW-max E.9: reads ``~/.cozempic/config.json`` exactly ONCE and
    passes the parsed dict to both resolvers. The prior implementation
    re-read the file inside each resolver — wasteful and a TOCTOU
    window where mid-cycle config edits flipped floor behavior between
    the two reads.
    """
    file_data = _read_config_file()
    return Config(
        min_idle_hours=_resolve_min_idle_hours_with(file_data),
        floor=_resolve_floor_with(file_data),
    )
