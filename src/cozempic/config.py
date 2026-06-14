"""Runtime configuration for cozempic safety guards.

Single source of truth for the floor preservation tunables introduced by the
prune-safety defense-in-depth fix (P0-B/C/D port onto v1.8.18 terminate-first):

  - ``floor``: per-prune protections — max % of user/assistant messages that
    may drop, last-K turns guaranteed to survive, first-message guarantee.

Precedence: environment variable > ``~/.cozempic/config.json`` > built-in default.

Invalid values (out-of-range, garbage strings, wrong type) silently fall back
to the default. Reading config never raises — a daemon mid-flight must not
crash because the operator stashed a stale env var in their shell rc.

P0-A symbols (``min_idle_hours``, ``resolve_min_idle_hours``) are intentionally
absent from this module: they belong to the idle-guard feature which is out of
scope for this PR (see PLAN.md §1 scope boundary).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Defaults + clamps ─────────────────────────────────────────────────────────

_FLOOR_MAX_DROP_PCT_DEFAULT: float = 0.50
_FLOOR_MAX_DROP_PCT_RANGE: tuple[float, float] = (0.0, 1.0)

_FLOOR_PRESERVE_LAST_K_DEFAULT: int = 10
_FLOOR_PRESERVE_LAST_K_RANGE: tuple[int, int] = (1, 1000)

_CONFIG_FILE_PATH = Path.home() / ".cozempic" / "config.json"


@dataclass(frozen=True)
class FloorConfig:
    """Per-prune floor preservation parameters.

    All three constraints are applied together every prune cycle. The floor
    is always-on in production — callers that genuinely want no floor (tests,
    dry-run diagnostics) pass ``FloorConfig.disabled()``.
    """

    max_user_assistant_drop_pct: float = _FLOOR_MAX_DROP_PCT_DEFAULT
    preserve_last_k_turns: int = _FLOOR_PRESERVE_LAST_K_DEFAULT
    preserve_first_message: bool = True

    @classmethod
    def disabled(cls) -> "FloorConfig":
        """Return a no-op FloorConfig (all constraints off).

        Use for test fixtures or diagnostic dry-runs that must not re-add
        messages. Not reachable from external JSON config (satisfies review
        finding H-2: floor control must not be exposed to arbitrary callers).
        """
        return cls(
            max_user_assistant_drop_pct=1.0,
            preserve_last_k_turns=0,
            preserve_first_message=False,
        )


@dataclass(frozen=True)
class Config:
    """Top-level cozempic runtime config."""

    floor: FloorConfig = field(default_factory=FloorConfig)


# ── Clamping helpers ──────────────────────────────────────────────────────────


def _clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    """Return value if it lies in [lo, hi] inclusive, else default.

    REVIEW-max B.2: explicitly reject NaN and infinities BEFORE the range
    check — NaN compares False to every threshold so a naive ``v < lo or
    v > hi`` lets it through and downstream arithmetic silently propagates.

    PR-2 P-B: reject bool before float() coercion — bool is a subclass of int,
    so float(True)==1.0 and float(False)==0.0 both pass the range check for
    [0.0, 1.0], silently coercing a JSON `true` into 1.0 and disabling the
    max-drop-pct floor. Match the _is_strict_number guard from _validation.py.
    """
    if isinstance(value, bool):
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    if v < lo or v > hi:
        return default
    return v


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    """Return value coerced to int and clamped to [lo, hi] inclusive.

    REVIEW-round3 F.M1: class-of-bug fold from B.2. ``int(float('inf'))``
    raises ``OverflowError`` (not in the prior except tuple) so an inf env
    var would crash the daemon at config-load time. NaN / inf string tokens
    short-circuit before conversion so the fall-back path is uniform with
    ``_clamp_float``.

    PR-2 P-B: class-of-bug fold from _clamp_float — bool is a subclass of int,
    so int(True)==1 and int(False)==0 pass the range check. Reject before
    conversion, consistent with _clamp_float and _is_strict_number.
    """
    # Short-circuit on string tokens that float() accepts but produce
    # non-finite values (inf, -inf, nan in any case).
    if isinstance(value, str):
        tok = value.strip().lower()
        if tok in ("inf", "+inf", "-inf", "infinity", "+infinity", "-infinity",
                   "nan", "+nan", "-nan"):
            return default
    if isinstance(value, bool):
        return default
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return default
    try:
        v = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if v < lo or v > hi:
        return default
    return v


def _parse_bool(raw: str, *, default: bool) -> bool:
    """Permissive bool parse for env strings.

    Accepts 0/1, true/false, yes/no, on/off (case-insensitive).
    Unparseable values return ``default``.
    """
    tok = raw.strip().lower()
    if tok in ("1", "true", "yes", "on", "y", "t"):
        return True
    if tok in ("0", "false", "no", "off", "n", "f"):
        return False
    return default


# ── Config file reader ────────────────────────────────────────────────────────


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


# ── Floor resolver ────────────────────────────────────────────────────────────


def _resolve_floor_with(file_data: dict[str, Any]) -> FloorConfig:
    """Resolve FloorConfig given pre-read config file data.

    Precedence per field: env var → config file → default.
    """
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

    # preserve_first_message — REVIEW-round3 F.N7: must read from file_data
    # and env var. Prior hardcoded True silently ignored operator config.
    # R3-1 fix: the file path previously used bare bool(), so bool("false") and
    # bool("0") both returned True — opposite of intent and inconsistent with the
    # env path which uses _parse_bool. Now: pass native JSON bools through as-is;
    # coerce non-bool values via _parse_bool (handles "false"/"0"/"no" correctly).
    raw_env = os.environ.get("COZEMPIC_FLOOR_PRESERVE_FIRST")
    if raw_env is not None and raw_env != "":
        preserve_first = _parse_bool(raw_env, default=True)
    elif "preserve_first_message" in floor_data:
        val = floor_data["preserve_first_message"]
        preserve_first = val if isinstance(val, bool) else _parse_bool(str(val), default=True)
    else:
        preserve_first = True

    return FloorConfig(
        max_user_assistant_drop_pct=drop_pct,
        preserve_last_k_turns=last_k,
        preserve_first_message=preserve_first,
    )


def load_config() -> Config:
    """Load the active runtime config (env → file → default).

    REVIEW-max E.9: reads ``~/.cozempic/config.json`` exactly ONCE and
    passes the parsed dict to all resolvers. The prior per-resolver file read
    was wasteful and had a TOCTOU window where mid-cycle config edits flipped
    floor behavior between reads.
    """
    file_data = _read_config_file()
    return Config(floor=_resolve_floor_with(file_data))
