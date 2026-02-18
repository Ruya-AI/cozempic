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


# ─── Hook definitions ────────────────────────────────────────────────────────

COZEMPIC_HOOKS = {
    "SessionStart": [
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": "cozempic guard --daemon 2>/dev/null || true",
                }
            ],
        },
    ],
    "PostToolUse": [
        {
            "matcher": "Task",
            "hooks": [
                {
                    "type": "command",
                    "command": "cozempic checkpoint 2>/dev/null || true",
                }
            ],
        },
        {
            "matcher": "TaskCreate|TaskUpdate",
            "hooks": [
                {
                    "type": "command",
                    "command": "cozempic checkpoint 2>/dev/null || true",
                }
            ],
        },
    ],
    "PreCompact": [
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": "cozempic checkpoint 2>/dev/null || true",
                }
            ],
        },
    ],
    "Stop": [
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": "cozempic checkpoint 2>/dev/null || true",
                }
            ],
        },
    ],
}


# ─── Core logic ──────────────────────────────────────────────────────────────

def _is_cozempic_hook(hook_entry: dict) -> bool:
    """Check if a hook entry was installed by cozempic."""
    for h in hook_entry.get("hooks", []):
        cmd = h.get("command", "")
        if "cozempic" in cmd:
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
    """Write settings.json with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")


def wire_hooks(project_dir: str) -> dict:
    """Add cozempic checkpoint hooks to .claude/settings.json.

    Idempotent — skips hooks that already exist.

    Returns dict with: added (list of hook events added), skipped (list already present),
    settings_path, backup_path.
    """
    path = _settings_path(project_dir)
    settings = _load_settings(path)

    hooks = settings.setdefault("hooks", {})

    added = []
    skipped = []

    for event_name, hook_entries in COZEMPIC_HOOKS.items():
        existing = hooks.get(event_name, [])

        # Check which hook entries are already installed
        for new_entry in hook_entries:
            already_exists = False
            for existing_entry in existing:
                if _is_cozempic_hook(existing_entry):
                    # Check if same matcher
                    if existing_entry.get("matcher", "") == new_entry.get("matcher", ""):
                        already_exists = True
                        break

            if already_exists:
                matcher = new_entry.get("matcher", "(all)")
                skipped.append(f"{event_name}[{matcher}]")
            else:
                existing.append(new_entry)
                matcher = new_entry.get("matcher", "(all)")
                added.append(f"{event_name}[{matcher}]")

        hooks[event_name] = existing

    # Only write if we added something
    backup = None
    if added:
        backup = _backup_settings(path)
        _save_settings(path, settings)

    return {
        "added": added,
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
