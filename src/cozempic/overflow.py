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

# Substrings that signal Claude Code hit a context-overflow / prompt-too-long
# error in the transcript tail. Kept as a LIST (was a single hardcoded
# "Conversation too long") because that exact string may be TUI-only and not the
# form persisted to the JSONL — if so, single-string detection left the reactive
# overflow path silently INERT. Widening is strictly broader (more markers caught,
# never fewer), so it cannot regress. ⚠ ARTIFACT-BLOCKED for full confidence: we
# still lack a captured REAL overflowed-CC JSONL tail to pin the exact persisted
# field/text — capture one and tighten/confirm against it (the recurring "capture
# the real artifact before trusting the detector" lesson).
OVERFLOW_MARKERS = (
    "Conversation too long",
    "Prompt is too long",
    "prompt is too long",
    "exceed the context",            # "...exceeds/exceed the context window/limit"
    "context_length_exceeded",       # API error code form
    "maximum context length",
)
# Back-compat alias (older imports / tests referenced the singular constant).
OVERFLOW_PATTERN = OVERFLOW_MARKERS[0]


class OverflowRecovery:
    """Detects context overflow and orchestrates recovery.

    Wired to JsonlWatcher.on_growth — fires on every file size increase.
    Fast-path exits immediately for normal growth. Only does work when
    size is concerning or overflow is detected.
    """

    # After a safe-point defer (in-flight work / gate error), suppress re-running a
    # full prune cycle on every subsequent growth event for this long (ynaamane #3c).
    _DEFER_COOLDOWN_S: float = 60.0

    def __init__(
        self,
        session_path: Path,
        session_id: str,
        cwd: str,
        breaker: CircuitBreaker,
        danger_threshold_mb: float = 90.0,
        danger_threshold_tokens: int | None = None,
        claude_pid: int | None = None,
    ):
        self.session_path = session_path
        self.session_id = session_id
        self.cwd = cwd
        self.breaker = breaker
        self.danger_threshold_bytes = int(danger_threshold_mb * 1024 * 1024)
        self.danger_threshold_tokens = danger_threshold_tokens
        self.claude_pid = claude_pid
        self._recovering = False  # Prevent re-entrant recovery
        self._defer_until = 0.0   # monotonic deadline; suppress re-run after a safe-point defer

    def detect_overflow(self) -> bool:
        """Check last 20 lines of the JSONL for overflow markers."""
        try:
            with open(self.session_path, "rb") as f:
                # Seek to last ~100KB to read tail efficiently
                f.seek(0, 2)
                size = f.tell()
                seek_to = max(0, size - 102400)
                f.seek(seek_to)
                tail = f.read().decode("utf-8", errors="replace")
        except OSError:
            return False

        truncated_head = seek_to > 0  # the first tail line is a partial fragment

        # A marker counts ONLY in an actual API-ERROR line — NOT a user turn that
        # merely discusses context limits ("my prompt is too long..."). The bare
        # substring scan false-fired on normal user text and triggered an
        # unsolicited kill+resume (mission-critical C6). Structural rule: parse the
        # line and require either isApiErrorMessage:true, or a non-user message
        # whose serialized content carries the marker.
        import json as _json
        lines = tail.split("\n")
        # The 100KB seek lands MID-LINE, so the first element is a truncated JSON
        # fragment. If it happens to contain a marker substring it fails json.loads —
        # and the old `except → return True` then false-fired an unsolicited
        # kill+resume on a perfectly benign large session whose tail merely DISCUSSED
        # overflow phrasing (R4 finding overflow-falsefire-unparseable-tail). Drop
        # that partial fragment, and NEVER infer overflow from an unparseable line:
        # a real overflow is a structurally-valid API-error entry (below).
        if truncated_head and lines:
            lines = lines[1:]
        for line in [ln.strip() for ln in lines[-20:]]:
            if not line or not any(m in line for m in OVERFLOW_MARKERS):
                continue
            try:
                obj = _json.loads(line)
            except (ValueError, _json.JSONDecodeError):
                # Unparseable marker-bearing line — cannot confirm it is a genuine
                # API error, so do NOT treat it as overflow (the structural gate
                # below is the only trigger). Prevents the truncated-tail false-kill.
                continue
            if not isinstance(obj, dict):
                continue
            # ONLY a genuine API-ERROR entry counts. Requiring isApiErrorMessage (or
            # an explicit error type/level) — NOT merely "any non-user line" — is
            # what stops the false-kill: the assistant, a `type:summary`
            # auto-compaction line, or a system meta line routinely DISCUSS overflow
            # phrasing ("maximum context length", "prompt is too long") without it
            # being a real overflow. Those must never trigger a kill+resume (C6).
            if obj.get("isApiErrorMessage") is True:
                return True
            if obj.get("type") == "error" or obj.get("level") == "error" or obj.get("isError") is True:
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

        # Re-entry throttle (ynaamane #3c): after a safe-point defer we deliberately
        # did NOT count a breaker slot, so the breaker won't auto-halt a busy in-flight
        # session — instead suppress re-running a full prune cycle until the cooldown
        # elapses, so frequent growth events don't hammer guard_prune_cycle.
        if time.monotonic() < self._defer_until:
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
        from .guard import checkpoint_team, guard_prune_cycle, _terminate_and_resume
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

        # #106: guard_prune_cycle no longer writes the live file inline — it
        # returns a deferred writer that is invoked only AFTER Claude is
        # terminated below (so the os.replace never swaps an inode under a live
        # fd). Use the PROJECTED post-prune size for the pre-flight; the on-disk
        # file is still the full pre-prune file until the deferred write fires.
        deferred_writer = result.get("_deferred_writer")
        final_bytes = result.get("_final_bytes")
        if final_bytes is not None:
            after_mb = final_bytes / 1024 / 1024
        else:
            # Futile / no-change prune (nothing to write) — file is unchanged.
            after_mb = self.session_path.stat().st_size / 1024 / 1024

        # 4. Pre-flight: if still dangerously large, don't resume.
        # Byte axis:
        still_dangerous = after_mb * 1024 * 1024 > self.danger_threshold_bytes * 0.95
        # Token axis (symmetry fix): a TOKEN-triggered overflow must also re-check
        # tokens post-prune — the byte-only preflight left the "don't resume if
        # still dangerous" guard INERT for the token path (audit P1). Use the
        # projected post-prune token count when present.
        if not still_dangerous and self.danger_threshold_tokens is not None:
            _proj = result.get("projected_final_tokens")
            if _proj is None:
                _proj = result.get("final_tokens")
            if isinstance(_proj, (int, float)) and _proj > self.danger_threshold_tokens * 0.95:
                still_dangerous = True
        if still_dangerous:
            print(
                f"  [{now}] Post-prune size {after_mb:.1f}MB still too large. "
                f"Skipping resume.",
                file=sys.stderr,
            )
            self.breaker.record_recovery(rx, before_mb, after_mb)
            checkpoint_team(session_path=self.session_path, quiet=False)
            return

        # 5. (breaker recording moved BELOW the 5b safe-point gate — a benign
        # in-flight defer must NOT consume a circuit-breaker slot, else 3 benign
        # defers silently disable the reactive net; mirrors the proactive path
        # which does not count a safe-point defer. ynaamane review #3.)
        orig_tok = result.get("original_tokens")
        final_tok = result.get("final_tokens")
        saved_tok = (orig_tok - final_tok) if (orig_tok and final_tok) else -1
        # saved_tok < 0 means the exact count re-anchored after metadata-strip
        # (#105) — the token delta is not meaningful, so fall through to the
        # MB-only line below rather than print "Pruned -648.7K tokens freed".
        if orig_tok and final_tok and saved_tok >= 0:
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

        # 5b. SAFE-POINT GATE before any kill (mission-critical C6): reactive
        # recovery must NOT terminate a live Claude that has in-flight work
        # (running subagents / agent team / open tool call) — the prune output is
        # already saved, so on an unsafe point we defer the kill rather than
        # destroy in-flight state. Mirrors guard_prune_cycle's safe_to_reload gate,
        # which the reactive path was missing.
        try:
            from .guard import safe_to_reload as _safe_to_reload
            from .session import load_messages as _load_messages
            from .team import extract_team_state as _extract_team_state
            _msgs = _load_messages(self.session_path)
            _safe, _reason = _safe_to_reload(_extract_team_state(_msgs), _msgs, self.session_path)
        except Exception as _gate_exc:
            # FAIL-CLOSED (ynaamane review #1): the daemon's asymmetry is "over-defer
            # is recoverable; wrongly SIGKILLing live Claude work is catastrophic." So
            # if the safety gate ITSELF throws (e.g. a malformed tool_result crashes
            # extract_team_state), DEFER the kill — never proceed to terminate a session
            # that may hold running subagents. The prune is recomputed next cycle.
            _safe, _reason = False, f"safety-gate error: {_gate_exc!r}"
        if not _safe:
            # NOTE: the pruned output is NOT yet on disk here — guard_prune_cycle
            # returns a DEFERRED writer that only fires inside _terminate_and_resume
            # (post-kill). So on a defer NOTHING is persisted this cycle; we re-run on
            # the next growth event (throttled below). (ynaamane review #3b.)
            print(f"  [{now}] In-flight work / gate defer ({_reason}) — deferring kill; "
                  f"prune NOT persisted this cycle, will retry.", file=sys.stderr)
            checkpoint_team(session_path=self.session_path, quiet=True)
            # Throttle re-entry: a busy in-flight session can fire many growth events;
            # don't re-run a full prune cycle on each. (ynaamane review #3c.)
            self._defer_until = time.monotonic() + self._DEFER_COOLDOWN_S
            return

        # Record the recovery only now that we are actually proceeding to kill+resume
        # (a real recovery), so a benign defer above never burns a breaker slot.
        self.breaker.record_recovery(rx, before_mb, after_mb)

        # 6. Terminate Claude + auto-resume
        # Wave 2: acquire single-flight reload lock. If another reload
        # pipeline is already in flight (manual `cozempic reload`, guard
        # threshold-fire, or another overflow recovery instance), defer
        # ours. The prune output is already saved; the in-flight pipeline
        # will do the kill+resume.
        claude_pid = self.claude_pid if self.claude_pid is not None else find_claude_pid()
        if claude_pid:
            from .reload_lock import _ReloadLock, ReloadLockHeld, INIT_OVERFLOW
            try:
                with _ReloadLock(self.session_id, initiator=INIT_OVERFLOW):
                    # Pass session_path so the identity check's forked-Claude
                    # mtime fallback works (happy-path symmetry with
                    # guard_prune_cycle). The bare-liveness gate in
                    # _terminate_and_resume still prevents resurrection.
                    # #106: write_pruned persists the prune AFTER Claude is dead.
                    _terminate_and_resume(
                        claude_pid, self.cwd,
                        session_id=self.session_id,
                        session_path=self.session_path,
                        write_pruned=deferred_writer,
                    )
                    write_holder = result.get("_write_holder") or {}
                    if write_holder.get("written") or deferred_writer is None:
                        print(
                            f"  [{now}] Kill + resume triggered (PID {claude_pid}). "
                            f"~10s downtime.",
                            file=sys.stderr,
                        )
                    else:
                        # #106: the deferred write was skipped (Claude survived the
                        # kill, or an append-conflict) — the live file was left
                        # intact, so the resumed Claude reloads the FULL file. The
                        # circuit breaker bounds repeated no-op recoveries.
                        print(
                            f"  [{now}] Kill triggered (PID {claude_pid}) but prune was "
                            f"not persisted — resuming from the full file.",
                            file=sys.stderr,
                        )
            except ReloadLockHeld as exc:
                print(
                    f"  [{now}] Reload deferred — another pipeline in flight "
                    f"({exc.holder_initiator}, PID {exc.holder_pid}).",
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
