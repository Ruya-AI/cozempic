"""Health checks for Claude Code configuration and environment.

The 'doctor' command diagnoses known issues beyond session bloat —
config bugs, oversized sessions, stale backups, and disk usage.
"""

from __future__ import annotations

import json
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .session import find_sessions, get_claude_dir


@dataclass
class CheckResult:
    """Result of a single health check."""

    name: str
    status: str  # "ok" | "warning" | "issue" | "fixed"
    message: str
    fix_description: str | None = None


# ─── Checks ──────────────────────────────────────────────────────────────────


def check_trust_dialog_hang() -> CheckResult:
    """Check for hasTrustDialogAccepted causing resume hangs.

    On Windows, setting hasTrustDialogAccepted=true in ~/.claude.json
    causes `claude --resume` to hang. The trust dialog initialization
    path is skipped, but resume depends on something it sets up.

    Ref: anthropics/claude-code#18532
    """
    claude_json = get_claude_dir() / ".claude.json"

    if not claude_json.exists():
        return CheckResult(
            name="trust-dialog-hang",
            status="ok",
            message=f"No {claude_json} found (fresh install)",
        )

    try:
        data = json.loads(claude_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return CheckResult(
            name="trust-dialog-hang",
            status="warning",
            message=f"Could not read {claude_json}: {e}",
        )

    # Check top-level and project-specific entries
    locations = []
    if data.get("hasTrustDialogAccepted") is True:
        locations.append("top-level")

    for key, value in data.items():
        if isinstance(value, dict) and value.get("hasTrustDialogAccepted") is True:
            locations.append(key[:60])

    if not locations:
        return CheckResult(
            name="trust-dialog-hang",
            status="ok",
            message="Trust dialog flag not set — no issue",
        )

    is_windows = platform.system() == "Windows"
    severity = "issue" if is_windows else "warning"

    return CheckResult(
        name="trust-dialog-hang",
        status=severity,
        message=(
            f"hasTrustDialogAccepted=true in {len(locations)} location(s). "
            f"{'This causes resume hangs on Windows.' if is_windows else 'Known to cause resume hangs on Windows — safe on macOS/Linux.'}"
        ),
        fix_description="Reset hasTrustDialogAccepted to false (trust prompt will reappear once)",
    )


def fix_trust_dialog_hang() -> str:
    """Fix the trust dialog hang by resetting hasTrustDialogAccepted."""
    claude_json = get_claude_dir() / ".claude.json"

    if not claude_json.exists():
        return f"No {claude_json} found — nothing to fix."

    try:
        data = json.loads(claude_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return f"Could not read {claude_json}: {e}"

    changed = False

    if data.get("hasTrustDialogAccepted") is True:
        data["hasTrustDialogAccepted"] = False
        changed = True

    for key, value in data.items():
        if isinstance(value, dict) and value.get("hasTrustDialogAccepted") is True:
            value["hasTrustDialogAccepted"] = False
            changed = True

    if not changed:
        return "No hasTrustDialogAccepted=true found — nothing to fix."

    # Backup before modifying
    backup = claude_json.parent / ".claude.json.bak"
    shutil.copy2(claude_json, backup)

    claude_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return f"Reset hasTrustDialogAccepted to false. Backup: {backup}"


def check_oversized_sessions() -> CheckResult:
    """Check for session files large enough to cause resume hangs (>50MB)."""
    sessions = find_sessions()
    large = [s for s in sessions if s["size"] > 50 * 1024 * 1024]

    if not large:
        return CheckResult(
            name="oversized-sessions",
            status="ok",
            message=f"No oversized sessions found ({len(sessions)} sessions checked)",
        )

    sizes = ", ".join(
        f"{s['session_id'][:8]}…({s['size'] / 1024 / 1024:.0f}MB)"
        for s in sorted(large, key=lambda s: s["size"], reverse=True)[:5]
    )

    return CheckResult(
        name="oversized-sessions",
        status="issue",
        message=f"{len(large)} session(s) over 50MB: {sizes}. These will likely hang on resume.",
        fix_description="Run: cozempic treat <session> -rx aggressive --execute",
    )


def check_stale_backups() -> CheckResult:
    """Check for old .bak files from previous treatments wasting disk space."""
    claude_dir = get_claude_dir()
    projects_dir = claude_dir / "projects"

    if not projects_dir.exists():
        return CheckResult(
            name="stale-backups",
            status="ok",
            message="No projects directory found",
        )

    bak_files = list(projects_dir.rglob("*.bak"))
    if not bak_files:
        return CheckResult(
            name="stale-backups",
            status="ok",
            message="No stale backup files found",
        )

    total_bytes = sum(f.stat().st_size for f in bak_files)
    return CheckResult(
        name="stale-backups",
        status="warning" if total_bytes > 100 * 1024 * 1024 else "ok",
        message=f"{len(bak_files)} backup file(s) using {total_bytes / 1024 / 1024:.1f}MB",
        fix_description="Delete old backups to reclaim disk space" if total_bytes > 100 * 1024 * 1024 else None,
    )


def fix_stale_backups() -> str:
    """Delete stale backup files."""
    claude_dir = get_claude_dir()
    projects_dir = claude_dir / "projects"

    if not projects_dir.exists():
        return "No projects directory found."

    bak_files = list(projects_dir.rglob("*.bak"))
    if not bak_files:
        return "No backup files to clean."

    total = 0
    for f in bak_files:
        total += f.stat().st_size
        f.unlink()

    return f"Deleted {len(bak_files)} backup file(s), freed {total / 1024 / 1024:.1f}MB"


def check_disk_usage() -> CheckResult:
    """Check total Claude session disk usage."""
    sessions = find_sessions()
    total = sum(s["size"] for s in sessions)

    if total < 500 * 1024 * 1024:
        status = "ok"
    elif total < 2 * 1024 * 1024 * 1024:
        status = "warning"
    else:
        status = "issue"

    return CheckResult(
        name="disk-usage",
        status=status,
        message=f"{len(sessions)} sessions using {total / 1024 / 1024:.1f}MB total",
        fix_description="Run: cozempic treat <session> -rx standard --execute" if status != "ok" else None,
    )


# ─── Registry ────────────────────────────────────────────────────────────────

# (name, check_fn, fix_fn_or_None)
ALL_CHECKS: list[tuple[str, callable, callable | None]] = [
    ("trust-dialog-hang", check_trust_dialog_hang, fix_trust_dialog_hang),
    ("oversized-sessions", check_oversized_sessions, None),
    ("stale-backups", check_stale_backups, fix_stale_backups),
    ("disk-usage", check_disk_usage, None),
]


def run_doctor(fix: bool = False) -> list[CheckResult]:
    """Run all health checks. If fix=True, apply available fixes for issues."""
    results = []
    for name, check_fn, fix_fn in ALL_CHECKS:
        result = check_fn()
        results.append(result)
        if fix and result.status in ("issue", "warning") and result.fix_description and fix_fn:
            fix_msg = fix_fn()
            result.message += f"\n      Fixed: {fix_msg}"
            result.status = "fixed"
    return results
