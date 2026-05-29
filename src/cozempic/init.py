"""cozempic init — auto-wire hooks and slash command into a Claude Code project.

After `pip install cozempic`, users still need to:
  1. Wire hooks into .claude/settings.json for checkpoint triggers
  2. Optionally install the /cozempic slash command

This module automates both so `cozempic init` is the only setup step.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


# ─── Command resolver ────────────────────────────────────────────────────────
# Use a shell snippet that finds cozempic regardless of PATH setup.
# Tries: bare command → python3 -m → python -m. Defined once, used in all hooks.
_CMD = '$(command -v cozempic >/dev/null 2>&1 && echo cozempic || echo "python3 -m cozempic")'

# For inline use in hook commands:
def _c(args: str) -> str:
    """Build a cozempic command that works regardless of PATH."""
    return f'{{ cozempic {args} 2>/dev/null || python3 -m cozempic {args} 2>/dev/null; }}'


# ─── Hook definitions ────────────────────────────────────────────────────────
# The canonical source is src/cozempic/data/hooks.json (also shipped to
# users via the plugin marketplace as plugin/hooks/hooks.json — kept in sync
# by tests/test_hooks_sync.py). Editing the dict literal that used to live
# here is gone; edit the JSON file instead.


# Bump this whenever a canonical hook's semantics change in a way that requires
# existing installations to pick up a new command string. The marker embedded
# at the end of every canonical hook command (e.g. "# cozempic-hook-schema=v2")
# is what `_is_current_cozempic_hook` looks for — old hooks without the current
# marker are treated as stale and get refreshed on next init.
HOOK_SCHEMA_VERSION = "v10"
HOOK_SCHEMA_MARKER = f"cozempic-hook-schema={HOOK_SCHEMA_VERSION}"


_LOAD_ERROR: str | None = None  # populated if _load_canonical_hooks failed


def _load_canonical_hooks() -> dict:
    """Load the canonical hook definitions bundled with the package.

    Returns an empty dict on any failure (missing file, corrupt JSON, old
    wheel without the packaged data). Records the error in `_LOAD_ERROR` so
    wire_hooks / uninstall_hooks can surface it to the user on the first
    operation (NOT at import time — otherwise every cozempic invocation
    spams stderr on a broken install).
    """
    global _LOAD_ERROR
    try:
        try:
            from importlib.resources import files
            data = files("cozempic").joinpath("data/hooks.json").read_text(encoding="utf-8")
        except Exception:
            data = (Path(__file__).parent / "data" / "hooks.json").read_text(encoding="utf-8")
        return json.loads(data).get("hooks", {})
    except Exception as exc:
        _LOAD_ERROR = (
            f"could not load bundled hook definitions ({exc}). "
            "Run `cozempic self-update` to repair the install."
        )
        return {}


COZEMPIC_HOOKS = _load_canonical_hooks()


# ─── Core logic ──────────────────────────────────────────────────────────────

def _is_cozempic_hook(hook_entry: dict) -> bool:
    """Return True if this entry contains AT LEAST ONE cozempic-installed
    command. See `_is_cozempic_command` for per-command granularity used by
    uninstall (which must preserve user-authored commands in mixed entries).
    """
    for h in hook_entry.get("hooks", []):
        if _is_cozempic_command(h.get("command", "")):
            return True
    return False


def _is_cozempic_command(command: str) -> bool:
    """Return True if this single hook command was installed by cozempic.

    Detection order:
      1. `cozempic-hook-schema=<ver>` marker (v2+ installs)
      2. Our FULL canonical wrapper shape — requires both a `{ cozempic <word> `
         opener AND a `python3 -m cozempic` (or `python -m cozempic`) fallback
         in the same command. That pair is distinctive to our template.

    Explicitly does NOT match:
      - Bare `"cozempic" in command` — false-matches user chains like
        `cozempic checkpoint && my-backup.sh`.
      - Bare `"python3 -m cozempic" in command` — false-matches user chains
        like `pre-step; python3 -m cozempic checkpoint; post-step` which would
        silently lose pre-step and post-step on uninstall.

    Net effect: user-authored commands that happen to invoke cozempic inline
    are left alone; only commands produced by our `_c()` template / canonical
    hooks.json entries are recognized as ours.
    """
    if "cozempic-hook-schema=" in command:
        return True
    has_wrapper_open = "{ cozempic " in command
    has_python_fallback = (
        "python3 -m cozempic" in command or "python -m cozempic" in command
    )
    return has_wrapper_open and has_python_fallback


def _is_current_cozempic_command(command: str) -> bool:
    """Return True if this command is at the CURRENT schema version (fresh).
    Used by auto-init to decide whether to refresh stale hooks."""
    return HOOK_SCHEMA_MARKER in command


def _entry_has_current_cozempic_hook(hook_entry: dict) -> bool:
    """Return True if at least one command in this entry matches the current
    schema. Used by `_maybe_auto_init` and `_maybe_global_init` to decide
    whether the project/user needs a refresh."""
    for h in hook_entry.get("hooks", []):
        if _is_current_cozempic_command(h.get("command", "")):
            return True
    return False


def has_current_schema(settings: dict) -> bool:
    """Return True if the settings dict has at least one current-schema
    cozempic hook across any event. Used by cli auto-init to avoid refresh
    when nothing's stale."""
    hooks = settings.get("hooks", {}) or {}
    if not isinstance(hooks, dict):
        return False
    for entries in hooks.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and _entry_has_current_cozempic_hook(entry):
                return True
    return False


def _settings_path(project_dir: str) -> Path:
    """Return the .claude/settings.json path for a project."""
    return Path(project_dir) / ".claude" / "settings.json"


def _load_settings(path: Path) -> dict:
    """Load settings.json, returning empty dict if missing."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _backup_settings(path: Path) -> Path | None:
    """Create timestamped backup of settings.json."""
    if not path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(f".{ts}.bak")
    shutil.copy2(path, backup)
    return backup


def _save_settings(path: Path, settings: dict) -> None:
    """Write settings.json atomically.

    Writes to a tempfile alongside the target, fsyncs, then os.replace onto the
    real path. Crash-safe: a Ctrl-C / OOM-kill / power-loss mid-write leaves
    settings.json in its PREVIOUS state, not zeroed or half-written. Previously
    we opened path for "w" (truncate-in-place) which could wipe the user's
    entire Claude Code config on a bad interrupt.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    import os as _os, tempfile as _tempfile, stat as _stat

    # Capture original permissions before we replace (Claude Code creates
    # settings.json as 0o644; mkstemp creates 0o600 — we must restore).
    orig_mode = None
    try:
        orig_mode = _os.stat(path).st_mode & 0o7777
    except OSError:
        pass  # file doesn't exist yet; default perms are fine

    fd, tmp_name = _tempfile.mkstemp(
        prefix=".cozempic-settings-", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with _os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")
            f.flush()
            try:
                _os.fsync(f.fileno())
            except OSError:
                pass
            # Restore original file mode BEFORE replace so the target
            # inherits the right permissions atomically.
            if orig_mode is not None and hasattr(_os, "fchmod"):
                try:
                    _os.fchmod(f.fileno(), orig_mode)
                except OSError:
                    pass
        _os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


class _SettingsLock:
    """File-lock around settings.json read-modify-write cycles.

    Prevents two parallel `cozempic init` invocations (common when two shells
    run a first-init concurrently, or when SessionStart hooks fire for two
    sessions simultaneously) from clobbering each other's additions.

    Cross-platform: POSIX uses fcntl.lockf (chosen over flock for NFS
    reliability — flock silent-no-ops on NFSv3), Windows uses
    msvcrt.locking on the first byte of the lock file (same per-byte
    semantics, sibling pattern to _HostFileLock in helpers.py). Falls
    back to no-op on platforms missing both fcntl AND msvcrt.
    """
    def __init__(self, settings_path: Path):
        self.lock_path = settings_path.parent / ".cozempic-init.lock"
        self._fh = None

    def __enter__(self):
        import os as _os
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            # Append mode so two racing opens don't truncate each other; the
            # lock file content is irrelevant — we only use the fd for locking.
            self._fh = open(self.lock_path, "a")
            if _os.name == "nt":
                # Windows — msvcrt.locking locks bytes from the CURRENT file
                # position. "a" (append) mode on Windows leaves the pointer at
                # EOF: byte 0 on a fresh empty lock file, but >0 if a stale
                # non-empty lock file was left from a prior crashed run.
                # __exit__ rewinds to byte 0 before LK_UNLCK, so without
                # this matching seek(0) before LK_LOCK the two operations
                # would target different byte ranges and silently fail to
                # serialize. Mirrors the _HostFileLock pattern in helpers.py
                # (which has the same defense-in-depth gap — separate fix).
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
            else:
                # POSIX — use lockf (record locks) not flock: lockf works
                # reliably over NFS, whereas flock on NFSv3 historically
                # silent-no-ops (returns success without actually locking),
                # which would let two init runs clobber each other on
                # network homes.
                import fcntl
                fcntl.lockf(self._fh.fileno(), fcntl.LOCK_EX)
        except ImportError:
            # Platform missing both fcntl AND msvcrt (extremely unusual —
            # e.g. some embedded cpython builds). Degrade to unlocked; we
            # lose the concurrency guarantee but don't crash.
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
            self._fh = None
        except OSError as exc:
            # Permission error (read-only .claude/), disk full, etc. Degrade to
            # unlocked but warn — silent skipping would hide real setup problems.
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
            self._fh = None
            sys.stderr.write(
                f"  Cozempic: settings lock unavailable ({exc}); proceeding without concurrency guard.\n"
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fh is None:
            return
        import os as _os
        try:
            if _os.name == "nt":
                import msvcrt
                # msvcrt.locking unlocks bytes from the current file
                # position — seek back to byte 0 to match the lock site.
                try:
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl
                fcntl.lockf(self._fh.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        try:
            self._fh.close()
        except OSError:
            pass


def wire_hooks(project_dir: str) -> dict:
    """Add cozempic checkpoint hooks to .claude/settings.json.

    Idempotent within a schema version. Detects STALE cozempic hooks
    (installed by an older cozempic whose schema marker is absent/old) and
    REPLACES them with the current canonical definition, so bug fixes in
    the hook command propagate to already-initialized projects.

    Returns dict with:
      added    — list of hook events newly installed
      updated  — list of hook events refreshed (stale → current)
      skipped  — list of hook events already at current schema
      settings_path, backup_path
    """
    path = _settings_path(project_dir)

    if not COZEMPIC_HOOKS:
        return {
            "added": [], "updated": [], "skipped": [],
            "settings_path": str(path), "backup_path": None,
            "error": _LOAD_ERROR or "bundled hook definitions unavailable — run `cozempic self-update`",
        }

    with _SettingsLock(path):
        try:
            settings = _load_settings(path)
        except (OSError, json.JSONDecodeError) as exc:
            # Malformed or unreadable settings.json — don't crash cmd_init with
            # a raw traceback. Surface via the `error` field like uninstall does.
            return {
                "added": [], "updated": [], "skipped": [],
                "settings_path": str(path),
                "backup_path": None,
                "error": f"could not parse {path}: {exc}. Back up + fix the file, then rerun.",
            }
        hooks = settings.setdefault("hooks", {})

        added: list[str] = []
        updated: list[str] = []
        skipped: list[str] = []

        for event_name, hook_entries in COZEMPIC_HOOKS.items():
            existing = hooks.get(event_name, [])

            for new_entry in hook_entries:
                matcher = new_entry.get("matcher", "")
                display = new_entry.get("matcher", "(all)")

                # Find the cozempic-installed entry with the same matcher (if any).
                # Mixed entries (cozempic + user command in the same hooks-list)
                # are preserved — we only replace the cozempic commands within.
                our_entry_idx = None
                for idx, existing_entry in enumerate(existing):
                    if existing_entry.get("matcher", "") != matcher:
                        continue
                    if _is_cozempic_hook(existing_entry):
                        our_entry_idx = idx
                        break

                if our_entry_idx is None:
                    existing.append(new_entry)
                    added.append(f"{event_name}[{display}]")
                    continue

                # Check if existing is at current schema
                our_entry = existing[our_entry_idx]
                if _entry_has_current_cozempic_hook(our_entry):
                    skipped.append(f"{event_name}[{display}]")
                    continue

                # STALE — refresh. Replace cozempic commands IN PLACE (preserve
                # original position of user-authored commands in the list).
                old_hooks = our_entry.get("hooks", []) or []
                canonical_new = list(new_entry.get("hooks", []))
                new_hooks: list = []
                canonical_inserted = False
                for h in old_hooks:
                    if _is_cozempic_command(h.get("command", "")):
                        # Splice in the new canonical commands at the position
                        # of the FIRST stale cozempic command; drop the rest.
                        if not canonical_inserted:
                            new_hooks.extend(canonical_new)
                            canonical_inserted = True
                        # else: this stale cozempic command is skipped
                    else:
                        new_hooks.append(h)
                if not canonical_inserted:
                    # Defensive: shouldn't happen (we only entered refresh path
                    # because a stale cozempic command exists), but fall back
                    # to appending.
                    new_hooks.extend(canonical_new)
                our_entry["hooks"] = new_hooks
                updated.append(f"{event_name}[{display}]")

            hooks[event_name] = existing

        # Only write if something changed
        backup = None
        if added or updated:
            backup = _backup_settings(path)
            _save_settings(path, settings)

    return {
        "added": added,
        "updated": updated,
        "skipped": skipped,
        "settings_path": str(path),
        "backup_path": str(backup) if backup else None,
    }


def install_slash_command(project_dir: str) -> dict:
    """Copy the /cozempic slash command to ~/.claude/commands/.

    Always overwrites to keep the slash command up-to-date with the
    installed cozempic version.

    Returns dict with: installed (bool), path, already_existed (bool), updated (bool).
    """
    # Find the slash command source — bundled as package data
    source = Path(__file__).parent / "data" / "cozempic_slash_command.md"

    # Fallback: dev/editable install — check repo root
    if not source.exists():
        source = Path(__file__).parent.parent.parent / ".claude" / "commands" / "cozempic.md"

    from .session import get_claude_dir
    target_dir = get_claude_dir() / "commands"
    target = target_dir / "cozempic.md"

    if not source.exists():
        return {"installed": False, "path": None, "already_existed": False, "updated": False}

    already_existed = target.exists()

    # Check if content differs
    if already_existed:
        if source.read_text(encoding="utf-8") == target.read_text(encoding="utf-8"):
            return {"installed": False, "path": str(target), "already_existed": True, "updated": False}

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

    return {"installed": True, "path": str(target), "already_existed": already_existed, "updated": already_existed}


def run_init(project_dir: str, skip_slash: bool = False) -> dict:
    """Full init: wire hooks + install slash command.

    Returns combined result dict.
    """
    hook_result = wire_hooks(project_dir)
    slash_result = {"installed": False, "path": None, "already_existed": False}

    if not skip_slash:
        slash_result = install_slash_command(project_dir)

    return {
        "hooks": hook_result,
        "slash_command": slash_result,
    }


def uninstall_hooks(project_dir: str) -> dict:
    """Remove cozempic-installed hooks from a settings.json. Idempotent.

    Surgical — removes only cozempic commands, preserving any user-authored
    commands that shared the same entry. An entry is deleted only when its
    `hooks` list becomes empty after cozempic commands are filtered out.

    Returns: {removed: list[str], settings_path: str | None, backup_path: str | None}
    """
    path = _settings_path(project_dir)
    if not path.exists():
        return {"removed": [], "settings_path": None, "backup_path": None}

    with _SettingsLock(path):
        try:
            settings = _load_settings(path)
        except (OSError, json.JSONDecodeError) as exc:
            # Malformed settings.json — don't crash, just report and bail.
            return {
                "removed": [],
                "settings_path": str(path),
                "backup_path": None,
                "error": f"could not parse settings.json: {exc}",
            }

        hooks = settings.get("hooks", {})
        if not isinstance(hooks, dict) or not hooks:
            return {"removed": [], "settings_path": str(path), "backup_path": None}

        removed: list[str] = []
        changed = False

        for event in list(hooks.keys()):
            entries = hooks.get(event, [])
            if not isinstance(entries, list):
                continue

            new_entries: list = []
            for entry in entries:
                if not isinstance(entry, dict):
                    new_entries.append(entry)
                    continue

                old_inner = entry.get("hooks", []) or []
                kept_inner = [
                    h for h in old_inner
                    if not (isinstance(h, dict) and _is_cozempic_command(h.get("command", "")))
                ]

                if len(kept_inner) == len(old_inner):
                    # Nothing to remove in this entry
                    new_entries.append(entry)
                    continue

                # We removed at least one cozempic command from this entry
                changed = True
                matcher = entry.get("matcher", "(all)")
                removed.append(f"{event}[{matcher}]")
                if kept_inner:
                    # Preserve entry but with user hooks only
                    entry = {**entry, "hooks": kept_inner}
                    new_entries.append(entry)
                # else: drop entry entirely

            if new_entries:
                hooks[event] = new_entries
            else:
                del hooks[event]

        if not changed:
            return {"removed": [], "settings_path": str(path), "backup_path": None}

        backup = _backup_settings(path)
        if hooks:
            settings["hooks"] = hooks
        else:
            settings.pop("hooks", None)
        _save_settings(path, settings)

    return {
        "removed": removed,
        "settings_path": str(path),
        "backup_path": str(backup) if backup else None,
    }
