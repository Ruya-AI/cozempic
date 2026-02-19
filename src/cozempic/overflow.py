"""Overflow detection, circuit breaker, and recovery orchestration.

Detects when Claude's inbox delivery spikes the JSONL past the context
limit, and orchestrates recovery: escalating prune → kill → resume.

A circuit breaker prevents infinite recovery loops.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path


# ─── Circuit Breaker ─────────────────────────────────────────────────────────

BREAKER_MAX_RECOVERIES = 3
BREAKER_WINDOW_SECONDS = 300  # 5 minutes
PRESCRIPTION_LADDER = ["gentle", "standard", "aggressive"]


class CircuitBreaker:
    """Prevents infinite prune → resume → crash loops.

    Tracks recoveries within a rolling window. Escalates the prescription
    on each consecutive recovery. Trips (halts) after max recoveries.
    Auto-resets after the window expires with no new recoveries.
    """

    def __init__(
        self,
        session_id: str,
        max_recoveries: int = BREAKER_MAX_RECOVERIES,
        window_seconds: int = BREAKER_WINDOW_SECONDS,
    ):
        slug = hashlib.md5(session_id.encode()).hexdigest()[:12]
        self.state_path = Path(f"/tmp/cozempic_breaker_{slug}.json")
        self.max_recoveries = max_recoveries
        self.window_seconds = window_seconds

    def _load(self) -> list[dict]:
        """Load recovery records, pruning expired entries."""
        if not self.state_path.exists():
            return []
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        cutoff = time.time() - self.window_seconds
        return [r for r in data if r.get("ts", 0) > cutoff]

    def _save(self, records: list[dict]) -> None:
        try:
            self.state_path.write_text(json.dumps(records), encoding="utf-8")
        except OSError:
            pass

    def can_recover(self) -> bool:
        """True if we haven't exhausted recovery attempts in the window."""
        return len(self._load()) < self.max_recoveries

    def recovery_count(self) -> int:
        """Number of recoveries in the current window."""
        return len(self._load())

    def next_prescription(self) -> str:
        """Escalating prescription: gentle → standard → aggressive."""
        count = len(self._load())
        idx = min(count, len(PRESCRIPTION_LADDER) - 1)
        return PRESCRIPTION_LADDER[idx]

    def record_recovery(
        self,
        rx: str,
        before_mb: float,
        after_mb: float,
    ) -> None:
        """Record a recovery event."""
        records = self._load()
        records.append({
            "ts": time.time(),
            "rx": rx,
            "before_mb": round(before_mb, 2),
            "after_mb": round(after_mb, 2),
        })
        self._save(records)

    def reset(self) -> None:
        """Clear all recovery records."""
        self.state_path.unlink(missing_ok=True)


# ─── Overflow Recovery ────────────────────────────────────────────────────────

OVERFLOW_PATTERN = "Conversation too long"


class OverflowRecovery:
    """Detects context overflow and orchestrates recovery.

    Wired to JsonlWatcher.on_growth — fires on every file size increase.
    Fast-path exits immediately for normal growth. Only does work when
    size is concerning or overflow is detected.
    """

    def __init__(
        self,
        session_path: Path,
        session_id: str,
        cwd: str,
        breaker: CircuitBreaker,
        danger_threshold_mb: float = 90.0,
        danger_threshold_tokens: int | None = None,
    ):
        self.session_path = session_path
        self.session_id = session_id
        self.cwd = cwd
        self.breaker = breaker
        self.danger_threshold_bytes = int(danger_threshold_mb * 1024 * 1024)
        self.danger_threshold_tokens = danger_threshold_tokens
        self._recovering = False  # Prevent re-entrant recovery

    def detect_overflow(self) -> bool:
        """Check last 20 lines of the JSONL for overflow markers."""
        try:
            with open(self.session_path, "rb") as f:
                # Seek to last ~100KB to read tail efficiently
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 102400))
                tail = f.read().decode("utf-8", errors="replace")
        except OSError:
            return False

        # Check last 20 lines
        lines = tail.strip().split("\n")
        for line in lines[-20:]:
            if OVERFLOW_PATTERN in line:
                return True
        return False

    def on_file_growth(self, filepath: str, new_size: int) -> None:
        """Callback wired to JsonlWatcher. Fast-path for normal growth."""
        # Fast path: check bytes threshold
        bytes_danger = new_size >= self.danger_threshold_bytes

        # Check token threshold if configured
        tokens_danger = False
        if self.danger_threshold_tokens is not None and not bytes_danger:
            from .tokens import quick_token_estimate
            tok = quick_token_estimate(self.session_path)
            if tok is not None and tok >= self.danger_threshold_tokens:
                tokens_danger = True

        if not bytes_danger and not tokens_danger:
            return

        # Prevent re-entrant recovery
        if self._recovering:
            return

        # Slow path: check for actual overflow
        if not self.detect_overflow():
            return

        self.recover()

    def recover(self) -> None:
        """Execute recovery: breaker check → prune → kill → resume."""
        self._recovering = True
        try:
            self._do_recover()
        finally:
            self._recovering = False

    def _do_recover(self) -> None:
        from .guard import checkpoint_team, guard_prune_cycle, _spawn_reload_watcher
        from .session import find_claude_pid

        now = _now()
        print(f"\n  [{now}] OVERFLOW DETECTED — reactive recovery triggered", file=sys.stderr)

        # 1. Check breaker
        if not self.breaker.can_recover():
            count = self.breaker.recovery_count()
            print(
                f"  [{now}] CIRCUIT BREAKER TRIPPED — {count} recoveries in "
                f"{self.breaker.window_seconds}s window. Halting.",
                file=sys.stderr,
            )
            print(
                f"  [{now}] Saving final checkpoint. Manual intervention required.",
                file=sys.stderr,
            )
            checkpoint_team(session_path=self.session_path, quiet=False)
            return

        # 2. Get escalating prescription
        rx = self.breaker.next_prescription()
        before_mb = self.session_path.stat().st_size / 1024 / 1024
        print(
            f"  [{now}] Recovery #{self.breaker.recovery_count() + 1}: "
            f"rx={rx}, size={before_mb:.1f}MB",
            file=sys.stderr,
        )

        # 3. Run the prune cycle (team-protect, backup, checkpoint)
        result = guard_prune_cycle(
            session_path=self.session_path,
            rx_name=rx,
            auto_reload=False,  # We handle reload ourselves
            cwd=self.cwd,
            session_id=self.session_id,
        )

        after_mb = self.session_path.stat().st_size / 1024 / 1024

        # 4. Pre-flight: if still dangerously large, don't resume
        if after_mb * 1024 * 1024 > self.danger_threshold_bytes * 0.95:
            print(
                f"  [{now}] Post-prune size {after_mb:.1f}MB still too large. "
                f"Skipping resume.",
                file=sys.stderr,
            )
            self.breaker.record_recovery(rx, before_mb, after_mb)
            checkpoint_team(session_path=self.session_path, quiet=False)
            return

        # 5. Record in breaker
        self.breaker.record_recovery(rx, before_mb, after_mb)
        orig_tok = result.get("original_tokens")
        final_tok = result.get("final_tokens")
        if orig_tok and final_tok:
            saved_tok = orig_tok - final_tok
            tok_str = f"{saved_tok / 1000:.1f}K" if saved_tok >= 1000 else str(saved_tok)
            print(
                f"  [{now}] Pruned {tok_str} tokens freed "
                f"({before_mb:.1f}MB → {after_mb:.1f}MB)",
                file=sys.stderr,
            )
        else:
            print(
                f"  [{now}] Pruned {before_mb:.1f}MB → {after_mb:.1f}MB "
                f"(saved {result['saved_mb']:.1f}MB)",
                file=sys.stderr,
            )

        # 6. Kill Claude + auto-resume
        claude_pid = find_claude_pid()
        if claude_pid:
            _spawn_reload_watcher(claude_pid, self.cwd, session_id=self.session_id)
            print(
                f"  [{now}] Kill + resume triggered (PID {claude_pid}). "
                f"~10s downtime.",
                file=sys.stderr,
            )
        else:
            resume_flag = f"--resume {self.session_id}" if self.session_id else "--resume"
            print(
                f"  [{now}] Could not find Claude PID. Pruned but not reloading.",
                file=sys.stderr,
            )
            print(f"  Restart manually: claude {resume_flag}", file=sys.stderr)


def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")
