"""Shared validation helpers for user-supplied numeric input.

Collects the cross-cutting helpers used by:
  - strategy `config` dicts (originally introduced in strategies/_config.py,
    PR #79) — now re-exported from here to avoid duplicating error strings,
  - argparse CLI argument validators (via small wrappers in cli.py),
  - environment-variable parsing in tokens.py.

Design goals:

  1. A single source of truth for the error-message shape, so the user sees
     the same phrasing whether the bad value came from a config dict, a CLI
     flag, or an env var.

  2. ConfigError subclasses ValueError — callers that already catch
     ValueError keep working; new code can catch the narrower exception.

  3. Env-var helpers WARN rather than RAISE: an env var is ambient config
     (often set once in a shell profile), and we don't want a daemon that
     was running fine to crash on the next reload just because `$HOME`
     carries a stale override. CLI args and in-process config dicts, in
     contrast, raise — the call site is intentional and the user is there.
"""

from __future__ import annotations

import os
import sys
from typing import Any


class ConfigError(ValueError):
    """Raised when a config value violates a required invariant.

    Subclasses ValueError so callers that already catch ValueError (e.g.
    the executor's outer wrapper) keep working, but type-checking
    machinery and humans reading tracebacks see the specific name.
    """


# ── Type guards ────────────────────────────────────────────────────────────


def _is_strict_int(value: Any) -> bool:
    """Strict int check: rejects bool (True/False are ints in Python) and float.

    YAML parsers in particular will turn `yes`/`no` into booleans, and users
    occasionally write `8192.0` in a JSON config. Both are almost never what
    was meant when the field expects a byte/line count.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _is_strict_number(value: Any) -> bool:
    """Accept int or float for float-typed fields (e.g. MB thresholds).

    Reject bool — same YAML concern as `_is_strict_int` — and reject anything
    non-numeric. Users legitimately write `threshold=50` (int) expecting 50.0 MB,
    so int is fine; but `True` or `"50"` are not.
    """
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


# ── Config-dict helpers (shared across strategies + guard) ─────────────────


def coerce_non_negative_int(config: dict, key: str, default: int) -> int:
    """Return `config[key]` as a non-negative int, or `default` if key absent.

    Raises `ConfigError` if the value is present but the wrong type or sign.
    """
    if key not in config:
        return default
    value = config[key]
    if not _is_strict_int(value):
        raise ConfigError(
            f"config[{key!r}] must be an int, got {type(value).__name__} {value!r}"
        )
    if value < 0:
        raise ConfigError(
            f"config[{key!r}] must be non-negative, got {value}"
        )
    return value


def coerce_positive_int(config: dict, key: str, default: int) -> int:
    """Like `coerce_non_negative_int` but strict `> 0`.

    Separate helper because zero is nonsensical for values like polling
    intervals (would produce a spin loop) and context-window sizes
    (would divide by zero downstream), whereas zero IS valid for values
    like `system_overhead_tokens` (a session with no rules file).
    """
    if key not in config:
        return default
    value = config[key]
    if not _is_strict_int(value):
        raise ConfigError(
            f"config[{key!r}] must be an int, got {type(value).__name__} {value!r}"
        )
    if value <= 0:
        raise ConfigError(
            f"config[{key!r}] must be positive, got {value}"
        )
    return value


def coerce_positive_float(config: dict, key: str, default: float) -> float:
    """Return `config[key]` as a strictly-positive float, or `default` if absent.

    Accepts int in addition to float (users write `threshold=50`). Rejects
    bool, str, None, negative, and zero.
    """
    if key not in config:
        return default
    value = config[key]
    if not _is_strict_number(value):
        raise ConfigError(
            f"config[{key!r}] must be a number, got {type(value).__name__} {value!r}"
        )
    if value <= 0:
        raise ConfigError(
            f"config[{key!r}] must be positive, got {value}"
        )
    return float(value)


def coerce_choice(config: dict, key: str, choices: tuple[str, ...], default: str) -> str:
    """Return `config[key]` if it matches one of `choices`, else `default`.

    Raises `ConfigError` when the value is present but not a recognized
    choice. Error message lists the accepted values so the user can
    self-correct without reading the source.
    """
    if key not in config:
        return default
    value = config[key]
    if not isinstance(value, str):
        raise ConfigError(
            f"config[{key!r}] must be a string, got {type(value).__name__} {value!r}"
        )
    if value not in choices:
        raise ConfigError(
            f"config[{key!r}]={value!r} is not one of {list(choices)}"
        )
    return value


# ── Environment-variable helpers (warn + fall back, never raise) ───────────


def _env_warn(name: str, value: str, reason: str) -> None:
    """Emit a single consistent warning line to stderr and continue.

    Using bare stderr (not `logging`) matches the convention already used by
    `cli._prescan_argv` for the same class of warning — users running
    `cozempic ...` see the warning immediately above the command output
    without having to configure a logging handler."""
    print(
        f"Warning: ignoring {name}={value!r} — {reason}",
        file=sys.stderr,
    )


def parse_env_positive_int(name: str) -> int | None:
    """Read env var `name`, parse as strictly-positive int, return None if
    absent or invalid (after emitting a warning on stderr).

    Used by `tokens.get_context_window_override` — a context window of 0 or
    negative is nonsensical and would propagate into `pct = total / cw`
    producing negative or divide-by-zero errors deep in diagnostics output.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        _env_warn(name, raw, "must be an integer")
        return None
    if value <= 0:
        _env_warn(name, raw, "must be a positive integer")
        return None
    return value


def parse_env_non_negative_int(name: str) -> int | None:
    """Read env var, parse as non-negative int (0 OK), warn-and-fallback on
    failure.

    Used by `tokens.get_system_overhead_tokens` — `0` is a legitimate value
    meaning "no system overhead in this session" (e.g. a user with no CLAUDE.md
    and no MCP servers). Negatives and non-numerics are still rejected.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        _env_warn(name, raw, "must be an integer")
        return None
    if value < 0:
        _env_warn(name, raw, "must be non-negative")
        return None
    return value


# Token sets are module-level constants so tests can import and inspect them.
_BOOL_TRUE_TOKENS: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_BOOL_FALSE_TOKENS: frozenset[str] = frozenset({"0", "false", "no", "off"})
# Pre-computed help string — avoids recomputing sorted() on every warn call
# and keeps truthy/falsy groups distinct so users can self-correct.
_BOOL_TOKENS_HELP = (
    f"truthy: {sorted(_BOOL_TRUE_TOKENS)}; falsy: {sorted(_BOOL_FALSE_TOKENS)}"
)


def parse_env_bool(name: str, default: bool = False, warn: bool = True) -> bool:
    """Read env var `name`, parse as boolean, return `default` if absent.

    Truthy tokens (case-insensitive, whitespace stripped): 1, true, yes, on
    Falsy tokens  (case-insensitive, whitespace stripped): 0, false, no, off
    Unrecognized non-empty value: if `warn`, emits a warning to stderr and
    returns `default`.  Empty / absent / whitespace-only: returns `default`
    silently.

    Follows the warn+fallback contract of parse_env_positive_int:
    env vars are ambient config; an unrecognized value must not crash.

    The `warn` parameter suppresses the warning when set to False — useful
    for in-body re-reads that run on every call (e.g. _debug()) where a
    single module-load warning is enough and per-call spam would drown output.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in _BOOL_TRUE_TOKENS:
        return True
    if normalized in _BOOL_FALSE_TOKENS:
        return False
    if warn:
        _env_warn(name, raw, f"must be a boolean ({_BOOL_TOKENS_HELP})")
    return default
