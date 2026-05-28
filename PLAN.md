# PLAN — Session Pruner Resumable State Fix

Worktree: `.claude/worktrees/fix-session-pruner-resumable-state`
Branch: `worktree-fix-session-pruner-resumable-state`
Target prefix: cozempic `src/cozempic/`
Out-of-scope: ECC plugin Rust code, auto-restore UX (P2), `cozempic log` (P2)

---

## Evidence — files read end-to-end before planning

1. `src/cozempic/executor.py` (221 lines, read in full) — `execute_actions`, `_relink_parent_chain`, `fix_orphaned_tool_results`, `run_prescription`
2. `src/cozempic/guard.py` (lines 1–1270 of 2457 read; remainder identified by grep — covers `start_guard`, `guard_prune_cycle`, futile-reload + K=10 exit) — the auto-prune cadence + reload paths
3. `src/cozempic/helpers.py` (340 lines, read in full) — `is_protected`, `_PROTECTED_TYPES`, `find_active_background_tasks`, atomic write primitives
4. `src/cozempic/session.py` (760 lines, read in full) — `_PruneLock`, `_FileSnapshot`, `save_messages`, `load_messages`, `cleanup_old_backups`
5. `src/cozempic/strategies/gentle.py` (296 lines, read in full) — including `strategy_compact_summary_collapse` (gentle) which is THE culprit for the 98% loss
6. `src/cozempic/strategies/standard.py` (563 lines, read in full) — `tool-result-age`, `thinking-blocks`, `tool-output-trim`, `system-reminder-dedup`, `tool-use-result-strip`
7. `src/cozempic/strategies/aggressive.py` (594 lines, read in full) — `mega-block-trim`, `envelope-strip`, `image-strip`, `document-dedup`, `error-retry-collapse`
8. `src/cozempic/strategies/_config.py` (57 lines, full) — coercion helpers, `coerce_ordered_pair`
9. `src/cozempic/registry.py` (67 lines, full) — `PRESCRIPTIONS` definitions
10. `src/cozempic/cli.py` (partial — `cmd_treat` lines 268–380, `build_parser` lines 1156–1198) — CLI surface
11. Bug handoff `~/cozempic-bug-handoff.md` (218 lines, full)

---

## 1. Root cause analysis

### 1.1 What actually pruned the 63MB → 730KB

**The trigger was `compact-summary-collapse` in the `gentle` prescription,** not the aggressive prescription as one might assume from the damage.

Evidence:

- `strategies/gentle.py:17` declares `compact-summary-collapse` with expected savings `"85-95%"`. The 98% loss matches that range almost exactly.
- The strategy logic at `strategies/gentle.py:33-64`: finds the LAST `system`-typed entry with `subtype == "compact_boundary"`, then removes EVERY non-protected message at positions `[0, last_boundary_pos)`. It calls this "pre-boundary, already summarized."
- The `gentle` prescription (`registry.py:12-18`) is what `start_guard` runs at the **SOFT (25% tokens) tier** every 30s (`guard.py:822-845`). On a 1M-context session that hits 250K tokens, gentle fires on every cycle.
- For the `fannyugc` session: somewhere in its history, Claude Code's native `/compact` was run (or auto-compaction triggered), inserting a `compact_boundary` entry near the end. Once that boundary exists, `compact-summary-collapse` mass-deletes everything BEFORE it — which is essentially the entire conversation, since the boundary lands near the tail.

This explains every observed loss:

| Type            | Damage in handoff | Why                                                                                                            |
| --------------- | ----------------- | -------------------------------------------------------------------------------------------------------------- |
| `user`          | -85%              | All pre-boundary user messages dropped                                                                         |
| `assistant`     | -91%              | Same                                                                                                           |
| `attachment`    | -99%              | Same                                                                                                           |
| `system`        | -99%              | All non-`compact_boundary` system entries dropped (only the boundary itself + a few post-boundary `system` rows survive) |
| `ai-title`      | -98%              | `_META_TYPES` exemption at `gentle.py:45` only keeps a metadata singleton if its type does NOT appear post-boundary; if a fresher `ai-title` exists post-boundary, all earlier ones are dropped. |
| `last-prompt`   | -99%              | Same exemption logic; only the post-boundary `last-prompt` survives.                                            |
| `permission-mode` | -98%            | NOT in `_META_TYPES` (`gentle.py:45` lists only `last-prompt`, `pr-link`, `custom-title`, `ai-title`, `attribution-snapshot`) → no protection at all. |
| `worktree-state`  | +1%             | Already in `_PROTECTED_TYPES` (`helpers.py:266-272`) — `is_protected()` skips it. |

The strategy IS internally consistent with its design assumption: "post-compact, the pre-boundary content is summarized in the `isCompactSummary` message — Claude Code itself discards it at load time for files >5MB" (docstring at `gentle.py:21-23`).

But the **assumption that the survivor file is replayable** is wrong in this case:

- `claude --resume` does NOT do the same "summary-first replay" that `/compact` produces. The file must still be a coherent JSONL message stream with a valid `parentUuid` chain rooted at a `parentUuid == null` entry.
- `_relink_parent_chain` (`executor.py:60-119`) DOES preserve chain pointers (the team-lead's forensic notes confirm 0 broken pointers). But the resulting head of the file is no longer rooted at a `parentUuid == null` entry — the original root user message was dropped, and the post-boundary entries' chain still points back through `parentUuid` to UUIDs that survived only as the `compact_boundary` system message.
- More importantly: removing 99% of `system`, `last-prompt`, `permission-mode`, `ai-title` entries strips the **session bootstrap metadata** that Claude Code's resume engine reads to reconstruct the session (model selection, slash-command state, attached files, allowed-tools list, project context). Without these, the resume path either hangs (waiting for state that never arrives) or silently fails.

### 1.2 Was it the guard daemon or a one-shot `treat`?

The forensic evidence (three `.bak` snapshots at 11:10, 11:20, 11:26 — exactly 10 cycles apart at `interval=30s` × 20 saved-prune cycles, or one per minute on the soft-prune cadence — see `guard.py:522` `if cycle_count % 10 == 0: cleanup_old_backups(... keep=3)`) **strongly indicates the guard daemon ran the `gentle` prescription multiple times** before producing the final 730KB state.

But almost certainly the FINAL kill blow came on a single cycle where `compact_boundary` was newly inserted into the file (e.g., the user ran `/compact` or auto-compaction fired between cycles, then 30s later the guard's `gentle` prescription wiped everything pre-boundary).

### 1.3 Which strategy did the bulk of the damage

`compact-summary-collapse` (gentle tier). All other strategies in `gentle`, `standard`, and `aggressive` are non-destructive (replace/truncate operations, or removal of safe-by-definition entries like `attribution-snapshot`, `progress`, duplicate `file-history-snapshot`, etc.). None of them remove `user`/`assistant` content blocks wholesale.

### 1.4 Are `ai-title`, `last-prompt`, `permission-mode`, `system` entries semantically critical?

Per the JSONL forensic evidence and Claude Code's documented session layout:

- `last-prompt`: stores the most recent user-typed prompt for the "Resume previous prompt" UI. Losing all but the very last is acceptable BUT only if the very last is preserved.
- `ai-title`: session display name. Losing all but the most recent is acceptable.
- `permission-mode`: encodes which tools are allowed (`allowedTools`, `disallowedTools`, `dangerously-skip-permissions` flag). If the LAST permission-mode entry is missing, Claude Code does NOT know what permission posture to bootstrap. **Almost certainly load-bearing for resume.**
- `system`: this is the bucket Claude Code uses for `compact_boundary`, `microcompact_boundary`, but also for "session opened", "model switched", "context window expanded" metadata. The compact_boundary entry MUST survive. Other system entries are likely metadata-only.

**Verdict:** dropping these wholesale is the proximate cause of `claude --resume` hanging.

---

## 2. Fix design — 4 mandatory pillars

### 2.1 Files we will touch

| File | Why |
|---|---|
| `src/cozempic/_validation.py` | Add `coerce_positive_float` already exists; add config-file loader. |
| `src/cozempic/config.py` *(NEW)* | Single source of truth for runtime tunables (mtime idle hours, floor percentages, last-K-turns count). Read once at module import, with env-var override. |
| `src/cozempic/safety.py` *(NEW)* | New module: `assert_session_idle_or_force(path, min_idle_hours)`, `validate_post_prune(path_before, msgs_after)`, `enforce_floor(msgs_before, msgs_after, config)`. Pure functions; testable. |
| `src/cozempic/helpers.py` | Extend `_PROTECTED_TYPES` to include `permission-mode`, `last-prompt`. Extend `is_protected()` to keep recent-K of certain metadata types. |
| `src/cozempic/executor.py` | After `run_prescription`, apply floor enforcement + post-prune validation. On validation failure, raise `PruneValidationError`. |
| `src/cozempic/guard.py` | At entry to `guard_prune_cycle` (both `start_guard` callers AND `cmd_reload` callers), check active-session guard BEFORE acquiring `_PruneLock`. On refusal, log structured reason, no-op, no exit. |
| `src/cozempic/session.py` | New: `PruneValidationError`, exposed for callers. New `is_session_idle(path, min_idle_hours)` helper using `stat().st_mtime`. |
| `src/cozempic/cli.py` | Add `--force` flag to `treat`, `reload`. Add `--force` plumbing through `cmd_treat`, `cmd_reload`. New `cozempic safety-check` subcommand for dry-run diagnostic of why a prune would refuse. |
| `src/cozempic/strategies/gentle.py` | **Modify `compact-summary-collapse`** to honor the new floor + extended `_META_TYPES`. Add `permission-mode` to `_META_TYPES`. Keep last-K conversational turns (`user`+`assistant`) regardless of pre-boundary position. |
| `src/cozempic/registry.py` | Optionally lower `compact-summary-collapse` from the default `gentle` prescription, OR keep but rely on floor enforcement. (See open question Q-A.) |
| `tests/test_session_prune_safety.py` *(NEW)* | RED tests for P0-A |
| `tests/test_session_prune_validation.py` *(NEW)* | RED tests for P0-B |
| `tests/test_session_prune_floor.py` *(NEW)* | RED tests for P0-C |
| `tests/test_session_prune_metadata.py` *(NEW)* | RED tests for P0-D |
| `tests/test_session_prune_e2e.py` *(NEW)* | Integration RED test reproducing the 63MB → 730KB bug |
| `tests/fixtures/realistic_session_with_compact_boundary.jsonl` *(NEW)* | Synthetic ~1MB fixture |

### 2.2 P0-A — Active-session safety guard

**Goal:** the guard MUST refuse to prune a session whose JSONL `mtime` is within `N` hours unless the caller passes `force=True`.

**Public API (new):**

```python
# src/cozempic/safety.py
class ActiveSessionError(Exception):
    """Raised when a prune is blocked by the active-session guard."""
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
    """Return (is_idle, minutes_since_last_mtime)."""
    ...

def assert_session_idle_or_force(
    path: Path, *, min_idle_hours: float, force: bool
) -> None:
    """Raise ActiveSessionError if the session is not idle and force=False."""
    ...
```

**Plumbing:**

- `guard.guard_prune_cycle()` calls `assert_session_idle_or_force(session_path, min_idle_hours=cfg.min_idle_hours, force=False)` at the top, BEFORE `_PruneLock`. On `ActiveSessionError`, logs a structured warning (with idle minutes + threshold) and returns the `_no_change` dict — same shape as PruneLockError / PruneConflictError paths. Daemon stays alive; the K-counter does NOT advance (it advances only on saved_bytes == 0 prunes that actually ran).
- `guard.start_guard()` keeps the daemon alive even on repeated refusals — the operator may be actively typing.
- `cli.cmd_treat` exposes `--force` (default `False`). Without `--force`, calls `assert_session_idle_or_force(path, min_idle_hours=cfg.min_idle_hours, force=False)`. On `ActiveSessionError`, prints the exception message + exits with code 4 (new exit code, distinct from existing 2=lock, 3=conflict).
- `cli.cmd_reload` similarly.
- `overflow.OverflowRecovery.on_file_growth` — the reactive 90% emergency path — needs an exception. The 90% emergency IS the active-session case; that's the explicit purpose. Either it auto-applies `force=True` (recommended) OR has its own bypass. Open question Q-B.

**Default config:**

- `min_idle_hours = 24.0` (recommend over the 48h surfaced in the lead's brief — see § 6 Q-C).
- Override via env var `COZEMPIC_MIN_IDLE_HOURS` (float, clamped to `[0.0, 168.0]` — 0 = always-prune, 168h = never-prune; invalid values fall back to 24).
- Override via config file `~/.cozempic/config.json` (key `"min_idle_hours"`).
- Env var wins over config file wins over default.
- Per-session override: future work; not in scope.

**Exit code on refusal:** `4` for CLI (treat/reload). No exit for daemon (silent no-op + log line).

**Log message format:**

```
[2026-05-28 12:34:56] Prune refused — session active (mtime 4.2 min ago, threshold 24.0h). Use --force to override.
```

### 2.3 P0-B — Post-prune structural validation

**Goal:** after `run_prescription` produces the candidate `pruned_messages` list, verify the result is replayable BEFORE `save_messages` writes it. If validation fails, do NOT save; raise; the caller logs and skips this cycle.

**No `claude --resume` dry-run.** Spawning Claude is too expensive and depends on the binary being on PATH. Use pure-Python structural checks.

**Public API (new):**

```python
# src/cozempic/safety.py
class PruneValidationError(Exception):
    """Raised when the pruned message list fails structural validation."""
    def __init__(self, reason: str, evidence: dict):
        self.reason = reason
        self.evidence = evidence  # {failed_check, surviving_count, ...}
        super().__init__(f"Pruned session would not replay cleanly: {reason}")

def validate_post_prune(
    msgs_before: list[Message],
    msgs_after: list[Message],
    *,
    strict: bool = True,
) -> None:
    """Validate the pruned message list. Raise PruneValidationError on failure.

    Checks (in order, fail-fast):
      C1. parentUuid resolution: every msgs_after entry with a non-null parentUuid
          MUST point to a uuid that exists in msgs_after, OR (post-relink) to None.
          The _relink_parent_chain step in executor.py:60 SHOULD guarantee this,
          but we re-verify here as defense-in-depth.
      C2. Root preserved: the FIRST msgs_after entry whose parentUuid is null
          MUST exist. If msgs_before had a parentUuid=null root, msgs_after MUST
          ALSO have a parentUuid=null root (it can be a different one — the
          original root may have been dropped — but at least one must exist
          and it must precede every non-null-parented entry by line order).
      C3. Conversation survival: at least ONE 'user' AND ONE 'assistant'
          message survives. A pruned file with zero conversation messages is
          unreplayable by definition.
      C4. Bootstrap metadata: if msgs_before contained at least one 'system'
          entry with subtype in {'compact_boundary','microcompact_boundary'},
          msgs_after MUST contain the LAST such entry.
      C5. Bootstrap metadata: if msgs_before contained at least one
          'permission-mode' entry, msgs_after MUST contain the LAST one.
      C6. Bootstrap metadata: if msgs_before contained at least one
          'last-prompt' entry, msgs_after MUST contain the LAST one.
    """
    ...
```

**Plumbing:**

- Called from `executor.run_prescription()` immediately after `fix_orphaned_tool_results`. If validation fails: re-raise `PruneValidationError` — DO NOT mutate.
- `guard.guard_prune_cycle()` catches `PruneValidationError`, logs the reason + evidence dict, returns `_no_change` (same shape as PruneLockError) — does NOT call `save_messages`. K-counter does NOT advance (the prune was aborted before save, not because it failed to save bytes).
- `cli.cmd_treat` catches it, prints the failure reason, exits with code 5.
- No automatic rollback needed because we never wrote anything. The `.bak` is unaffected (it's only created INSIDE `save_messages`).

**Rollback for the edge case where `save_messages` succeeds but the post-write file fails to validate:** out of scope. The `_FileSnapshot` mechanism already prevents corruption-in-flight; combined with C1–C6 running pre-write, the gap is theoretical only.

### 2.4 P0-C — Floor preservation

**Goal:** strategies MUST never drop more than `X%` of `user`/`assistant` messages and MUST never drop the last K conversation turns or the first message.

**Architecture decision:** **enforce at executor level (post-action clamp), not at strategy level.**

Reason: each strategy independently computing the floor would (a) require touching every strategy in `strategies/*.py` and (b) admit a "death by 10000 cuts" bug where each strategy stays under its individual cap but the cumulative loss exceeds the floor. Centralizing at the executor catches all paths uniformly and is testable as a single unit.

**Public API (new):**

```python
# src/cozempic/safety.py
@dataclass(frozen=True)
class FloorConfig:
    max_user_assistant_drop_pct: float = 0.50   # never drop more than 50%
    preserve_last_k_turns: int = 50              # never drop the most recent 50 user+50 assistant
    preserve_first_message: bool = True

def enforce_floor(
    msgs_before: list[Message],
    msgs_after: list[Message],
    *,
    cfg: FloorConfig,
) -> list[Message]:
    """Re-add must-preserve messages that fell to the prune.

    Algorithm:
      1. Compute kept_uuids = {m['uuid'] for m in msgs_after if m has uuid}
      2. Identify must_preserve_uuids from msgs_before:
         (a) The first msg whose parentUuid is null.
         (b) The most recent K user msgs + K assistant msgs (by line order).
         (c) Enough additional user/assistant msgs to bring the survival
             percentage above (1.0 - max_user_assistant_drop_pct), picking
             from most-recent first.
      3. For each must_preserve_uuid not in kept_uuids: insert the original
         message back into msgs_after at the position that preserves
         line-index ordering. Re-run _relink_parent_chain after the insert.
      4. Return the floor-enforced msgs_after.
    """
    ...
```

**Plumbing:**

- `executor.run_prescription()` calls `enforce_floor(messages, current, cfg=cfg.floor)` AFTER `fix_orphaned_tool_results` and BEFORE `validate_post_prune`. Order matters: floor enforcement must run first because adding messages back may change which orphans exist, and validation runs last as the final gate.

**Defaults:**

- `max_user_assistant_drop_pct = 0.50` (override `COZEMPIC_FLOOR_MAX_DROP_PCT`, range `(0.0, 1.0)`)
- `preserve_last_k_turns = 50` (override `COZEMPIC_FLOOR_PRESERVE_LAST_K`, range `[1, 1000]`)
- `preserve_first_message = True` (no override — always-on)

### 2.5 P0-D — Special-type semantic protection

**Goal:** `ai-title`, `last-prompt`, `permission-mode` entries are session metadata. Preserve at least the LAST N of each. Hook into `is_protected()` in `helpers.py:275`.

**Decision:** rather than make `is_protected()` content-aware (which would break its pure-function contract), introduce a pre-pass in the executor that tags the last-of-each metadata type before running strategies. Strategies that respect `is_protected()` (which is ALL of them in `gentle.py`, `standard.py`, `aggressive.py`) will then skip them.

**Public API (extends executor):**

```python
# src/cozempic/executor.py (new helper)
_LAST_OF_TYPE_PROTECTED: frozenset[str] = frozenset({
    "ai-title",
    "last-prompt",
    "permission-mode",
})

def _tag_last_of_metadata_types(messages: list[Message]) -> None:
    """Mark the LAST occurrence of each protected-singleton type as protected.

    Sets msg['__cozempic_metadata_singleton__'] = True on the latest entry
    of each type. The tag is removed by run_prescription before returning.
    """
    ...
```

`is_protected()` extension:

```python
# src/cozempic/helpers.py — add to is_protected()
if msg.get("__cozempic_metadata_singleton__"):
    return True
```

**Why singleton-only (last-of-type) and not all-of-type:**

- `ai-title` and `last-prompt` are intentionally rewritten over the course of a session — the LATEST one is what matters for resume bootstrap. Pre-boundary `ai-title` entries are stale by design.
- `permission-mode` is the same — only the current mode bootstraps resume; older ones encode obsolete state.
- This matches the existing `_META_TYPES` exemption logic in `compact-summary-collapse` (`gentle.py:45-56`) which keeps a metadata singleton only when its type does not appear post-boundary. We extend the protection to apply REGARDLESS of post-boundary presence — i.e., always keep the LAST one.

**Also update `_META_TYPES` in `gentle.py`:** add `permission-mode` to the set at line 45 so even pre-validation, `compact-summary-collapse` doesn't drop the last permission-mode singleton.

---

## 3. New test surface (Phase 3 RED — for builder)

Each file gets a `Test...` class capturing the failure shape we expect from CURRENT code (RED), which fix code will turn GREEN.

### 3.1 `tests/test_session_prune_safety.py`

Scenarios (one method per scenario):

- `test_treat_refuses_active_session_within_threshold`
  - Setup: temp JSONL with `mtime = now - 5 min`. Set `COZEMPIC_MIN_IDLE_HOURS=24`.
  - Assertion: `cmd_treat(execute=True, force=False)` raises `ActiveSessionError` OR exits with code 4. JSONL is unchanged.
- `test_treat_accepts_active_session_with_force`
  - Setup: same fixture; `force=True`.
  - Assertion: prune runs, file is rewritten (assert mtime advanced + content changed).
- `test_treat_accepts_idle_session_without_force`
  - Setup: `mtime = now - 25 hours`. `COZEMPIC_MIN_IDLE_HOURS=24`.
  - Assertion: prune runs without force.
- `test_guard_prune_cycle_skips_active_session`
  - Setup: live session; `mtime = now - 1 min`. Invoke `guard_prune_cycle` directly.
  - Assertion: returns `_no_change` shape with `saved_mb == 0.0`, JSONL unmodified, log line emitted.
- `test_guard_prune_cycle_advances_k_counter_only_on_genuine_zero_byte_prune`
  - Assertion: a refusal does NOT count toward the K=10 exit ladder.
- `test_env_var_override`
  - Setup: `COZEMPIC_MIN_IDLE_HOURS=0` → always-prune even on live.
  - Assertion: prune of fresh-mtime session proceeds without force.
- `test_env_var_invalid_value_falls_back_to_default`
  - Setup: `COZEMPIC_MIN_IDLE_HOURS=garbage` or `=-1` or `=200`.
  - Assertion: default `24` applies; `mtime = now - 25h` allowed, `mtime = now - 1h` blocked.

### 3.2 `tests/test_session_prune_validation.py`

Scenarios:

- `test_validation_passes_on_clean_prune`
  - Setup: fixture with intact parent chain, both before & after.
  - Assertion: `validate_post_prune` does not raise.
- `test_validation_fails_when_root_missing`
  - Setup: msgs_before has a root (`parentUuid=null`), msgs_after has zero `parentUuid=null` entries.
  - Assertion: raises `PruneValidationError`; `evidence["failed_check"] == "C2"`.
- `test_validation_fails_when_zero_user_messages_survive`
  - Setup: msgs_before had 100 users; msgs_after has 0.
  - Assertion: raises with `evidence["failed_check"] == "C3"`.
- `test_validation_fails_when_compact_boundary_dropped`
  - Setup: msgs_before had a `system/subtype=compact_boundary`; msgs_after lacks it.
  - Assertion: raises with `evidence["failed_check"] == "C4"`.
- `test_validation_fails_when_last_permission_mode_dropped`
  - Assertion: raises with `evidence["failed_check"] == "C5"`.
- `test_validation_fails_when_last_prompt_dropped`
  - Assertion: raises with `evidence["failed_check"] == "C6"`.
- `test_validation_fails_when_parent_chain_broken`
  - Setup: synthetic msgs_after with parentUuid pointing to a non-existent uuid.
  - Assertion: raises with `evidence["failed_check"] == "C1"`.
- `test_validation_skipped_when_msgs_before_had_no_root`
  - Setup: msgs_before has no parentUuid=null entry (e.g., pure metadata file).
  - Assertion: does NOT raise — C2 is conditional on msgs_before having a root.
- `test_executor_aborts_save_on_validation_failure`
  - Setup: arrange a strategy combo that produces an invalid post-prune list.
  - Assertion: `save_messages` was never called; original JSONL untouched.

### 3.3 `tests/test_session_prune_floor.py`

Scenarios:

- `test_floor_re_adds_dropped_user_assistant_when_below_threshold`
  - Setup: 100 users + 100 assistants before. Strategy drops 90% of each (10 + 10 after).
  - Assertion: `enforce_floor` brings survival up to >= 50%, prefers most recent.
- `test_floor_preserves_last_k_turns`
  - Setup: 100 users + 100 assistants, K=10. Strategy drops the entire last K block.
  - Assertion: the last 10 users AND last 10 assistants are re-added regardless of position.
- `test_floor_preserves_first_message`
  - Setup: 100 users, strategy drops the first (parentUuid=null) message.
  - Assertion: first message is re-added.
- `test_floor_no_op_when_strategies_respect_floor`
  - Setup: a strategy that drops only 10% of users/assistants.
  - Assertion: `enforce_floor` returns msgs_after unchanged.
- `test_floor_runs_before_validation`
  - Setup: arrange a case where without floor enforcement, validation would fail (zero users left), but with floor the validation passes.
  - Assertion: `run_prescription` succeeds end-to-end.
- `test_floor_does_not_undo_replacements`
  - Setup: a strategy that REPLACES a user message (in-place edit, e.g., `thinking-blocks` truncate).
  - Assertion: the replacement survives floor enforcement; the original is NOT re-added.

### 3.4 `tests/test_session_prune_metadata.py`

Scenarios:

- `test_last_ai_title_protected`
  - Setup: 100 ai-title entries scattered through the session.
  - Assertion: after any prescription, the LAST one survives.
- `test_last_last_prompt_protected`
  - Same.
- `test_last_permission_mode_protected`
  - Same. **Failing on current code = RED test for the bug.**
- `test_pre_boundary_metadata_dropped_when_replaced_post_boundary`
  - Setup: 50 ai-title pre-boundary, 5 ai-title post-boundary, then a `compact_boundary`.
  - Assertion: only the LAST (post-boundary) ai-title survives. The other 54 are dropped. (Matches current `gentle.py:45-56` exemption logic.)
- `test_singleton_protection_survives_compact_summary_collapse`
  - Setup: 50 permission-mode pre-boundary, 0 post-boundary.
  - Assertion: the LAST pre-boundary permission-mode survives. (Currently fails because permission-mode is not in `_META_TYPES`.)

### 3.5 `tests/test_session_prune_e2e.py`

Scenarios:

- `test_realistic_session_with_compact_boundary_does_not_lose_resumable_state`
  - Setup: load `tests/fixtures/realistic_session_with_compact_boundary.jsonl` — synthetic ~1MB file mimicking the corrupted fannyugc shape (10 000 entries, mixture of types matching the handoff damage table, with a `compact_boundary` 95% through the file).
  - Action: run `run_prescription(messages, ["gentle", "compact-summary-collapse", ...], {})`.
  - Assertions:
    - Final user count >= 50% of before.
    - Final assistant count >= 50% of before.
    - At least one `permission-mode` entry survives (the LAST one).
    - At least one `last-prompt` entry survives.
    - At least one `ai-title` entry survives.
    - First message (`parentUuid=null` root) survives.
    - `compact_boundary` entry survives.
    - Every surviving `parentUuid` resolves to either `null` or another surviving uuid.
    - File size reduction is between 30% and 80% (gentle should still produce meaningful reduction without destroying resume).
  - **This is the regression test for the bug.** Currently RED, will be GREEN after the four fix pillars are in place.

### 3.6 Fixture: `tests/fixtures/realistic_session_with_compact_boundary.jsonl`

Built by a fixture generator in `tests/conftest.py`. Specs:

- 10 000 entries total
- 1 root user message (`parentUuid: null`)
- 600 `user` entries (chained)
- 1 000 `assistant` entries
- 8 000 `attachment`-type entries (small, to inflate the count)
- 300 `ai-title` entries
- 300 `last-prompt` entries
- 300 `permission-mode` entries
- 200 `system` entries (no subtype)
- 1 `system` entry with `subtype=compact_boundary` at position 9 800
- 50 `file-history-snapshot` entries
- All `parentUuid` chains valid before pruning

Target on-disk size after JSON serialization: ~1MB. Generator function: `make_realistic_session(target_size_mb: float = 1.0, with_compact_boundary: bool = True)`.

---

## 4. Migration / backward compat

### 4.1 Currently-running cozempic guards

Per CLAUDE.md: yes, guards run in production for Yanis on multiple Python installs (system py3.12 + pyenv py3.11). The auto-upgrade hooks pull new PyPI releases on SessionStart.

### 4.2 Backward-compat strategy

- **Default-on, env-override-off** rather than the lead's suggested `COZEMPIC_FORCE_PRUNE_ACTIVE=1` knob:
  - `min_idle_hours` defaults to `24.0` — covers most active sessions safely.
  - Users who explicitly want the old behavior (always-prune) set `COZEMPIC_MIN_IDLE_HOURS=0`.
  - This is a SAFER default than "warn but proceed" because the bug we're fixing already shipped to prod and ate data.
- **No CLI signature break:** `cozempic treat <session>` continues to work; `--force` is a NEW optional flag, default `False`. Behavior change: `treat` now refuses with exit code 4 instead of proceeding on an active session. This IS a behavior change but it preserves data; users hitting the new exit code see a clear message and can re-run with `--force`.
- **CHANGELOG entry required** (in the PR body):
  - "BREAKING: `cozempic treat` and the guard daemon now refuse to prune sessions modified in the last 24h. Use `--force` (CLI) or `COZEMPIC_MIN_IDLE_HOURS=0` (env) to restore the old behavior."

### 4.3 Telemetry

The `record_savings` ping (helpers.py:120-179) is unchanged. New refusal events emit no telemetry (we deliberately don't want to ping Counters API for safety guards — it's noise).

---

## 5. Files NOT to touch

- `ecc2/src/session/manager.rs` (different repo)
- `src/cozempic/overflow.py` reactive recovery logic except for the explicit `force=True` pass-through described in P0-A (Q-B will resolve scope).
- `src/cozempic/team.py`, `team_state` extraction — orthogonal to this fix.
- `src/cozempic/reload_lock.py`, `spawn_lock.py` — orthogonal.
- The `cozempic log` command (P2 from the handoff) — defer to a follow-up PR. Add a TODO.md entry for it.
- Auto-restore UX (P2 — prompt user when resume fails) — defer.

---

## 6. Risks + open questions for lead

### Q-A — Should `compact-summary-collapse` stay in the default `gentle` prescription?

**Tradeoff:**

- **Keep it (relying on floor + validation):** preserves the documented gentle savings of 85-95% for sessions that ARE post-compact and have moved on. Floor enforcement prevents the 98% data loss. Validation catches edge cases.
- **Remove it from gentle (move to standard or aggressive):** safer default, but loses meaningful savings for legitimately-compacted sessions. The user would need to explicitly opt into a more aggressive prescription to get the gentle's old behavior.

**My recommendation:** keep it, rely on floor + validation. The original strategy intent is correct (post-compact pre-boundary content IS summarized), and the bug is structural (no floor, no validation). Fix the structural issue rather than removing the strategy.

**Lead's call needed:** confirm.

### Q-B — Overflow recovery bypass

`overflow.OverflowRecovery.on_file_growth` triggers at the 90% reactive threshold — the exact case where active-session pruning is intentional (it's emergency context recovery). Three options:

1. **OverflowRecovery silently sets `force=True`** for its prune call.
2. **OverflowRecovery is exempt from the active-session check entirely** (different code path).
3. **OverflowRecovery still refuses unless the session has been idle for some shorter threshold** (e.g., 1 min) — protects against the watchdog firing on a session that's mid-burst-write.

**My recommendation:** option 1 — `force=True` is the most explicit, makes it auditable in logs, and aligns with the "emergency" semantic.

**Lead's call needed:** confirm option choice.

### Q-C — Default mtime threshold (24h vs 48h)

**Tradeoff:**

- **24h:** matches typical workday cycles. Idle sessions get pruned overnight while the user is away. Reasonable safety floor.
- **48h:** more conservative; covers users who pause work over a weekend or take a day off mid-task.

**My recommendation:** 24h. Reasons:
- Pairs with the existing soft-prune cadence (every 30s) and `cleanup_old_backups(keep=3)` retention — a 48h session would accumulate ~5760 prune attempts at the soft tier; refusing all of them is fine, but 24h covers the actual high-risk window (active editing).
- Easy to override per-user via env var.
- Aligns with the typical Claude Code daily workflow.

**Lead's call needed:** confirm 24h or override to 48h.

### Q-D — Floor percentage (50% vs 30%)

**Tradeoff:**

- **50% floor:** generous — guarantees at least half the conversation survives. Limits the savings ceiling of `compact-summary-collapse` to ~50%.
- **30% floor:** tighter — allows more aggressive pruning. The 730KB / 63MB = 1.2% survival in the bug was catastrophic; even 30% would have prevented the resume failure.

**My recommendation:** 50%. The intent of cozempic is to prune session FILES (mostly oversize tool results, attachments, image blobs), not conversation messages. A 50% floor on `user`/`assistant` messages does not cap byte savings significantly — they're the smallest entries by volume. The strategies that actually save bytes (`tool-result-age`, `image-strip`, `mega-block-trim`, `tool-use-result-strip`) target non-conversation content.

**Lead's call needed:** confirm 50%.

### Q-E — Should we ship the fix in one PR or split?

Per CLAUDE.md hard rule (multi-PR sequencing):

- The 4 pillars are: mechanical (P0-D singleton extension) + defensive (P0-A active-session guard, P0-B validation) + architectural (P0-C floor enforcement).
- That's 3+ atomic commits but ONE risk profile (all defensive safety) and ONE reviewable concern (session pruner resume integrity).

**My recommendation:** single PR, sequenced commits in trust-building order:
1. `feat(safety): add active-session idle guard with force override` (P0-A)
2. `feat(safety): add post-prune structural validation` (P0-B)
3. `feat(safety): enforce floor preservation in executor` (P0-C)
4. `fix(strategies): protect last-of-type metadata singletons` (P0-D)
5. `test(e2e): regression test for 63MB to 730KB resume-break bug`
6. `docs(changelog): document new safety defaults and --force flag`

**Lead's call needed:** confirm single PR vs split.

### R-1 — Risk: validation false-positives

`validate_post_prune` might reject legitimate prunes if our heuristics are too strict (e.g., a session that legitimately has no `permission-mode` entries before the prune triggers C5 fallback). Mitigation: each check `Ci` is conditional on the corresponding type existing in msgs_before. Tests `test_validation_skipped_when_msgs_before_had_no_root` enforce this. Risk residual: medium → low.

### R-2 — Risk: floor enforcement vs replacements

`enforce_floor` reads `kept_uuids` from msgs_after. Strategies like `thinking-blocks` REPLACE entries in-place — the uuid is preserved, so the kept_uuids set is correct. But strategies like `compact-summary-collapse` REMOVE entries. If a single message is both replaced AND has its uuid count as "kept", that's correct. Risk: low.

### R-3 — Risk: idle threshold causes guards to never prune in active sessions

By design. A user actively typing wants guard protection at the 80% emergency tier (overflow recovery, force=True). The 24h idle threshold only blocks the gentle (25% soft) and standard (55% hard1) tiers. The 80% reactive tier bypasses (Q-B option 1). Risk: low.

### R-4 — Risk: backward compat — users with automation

Some users (incl. Yanis) auto-upgrade cozempic via SessionStart hooks. After the upgrade, their guard daemons silently start refusing to prune live sessions. They will see new log lines, may panic, may file bugs. Mitigation: clear CHANGELOG entry + a one-line stdout banner on guard startup explaining the new default and the env-var override. Risk: medium.

### R-5 — Risk: the `compact-summary-collapse` strategy's documented "85-95%" savings claim becomes a lie

Floor enforcement will cap real-world savings of this strategy at ~50% on the user/assistant fraction. The byte savings can still be high if pre-boundary attachments/tool-results dominate. Update the strategy's `@strategy` decorator metadata from `"85-95%"` to `"50-90% (floor-clamped)"`. Risk: cosmetic; low.

---

## 7. Verification envelope

## Verification
- Confidence: 88% (reasoning: read 10 source files end-to-end including all four strategy files and the entire helpers/session/executor modules; cross-referenced the handoff's damage table against the gentle.py compact-summary-collapse logic; the per-type loss pattern is fully explained by the strategy's removal of pre-boundary content combined with the META_TYPES exemption logic. Residual 12% uncertainty: (a) I did not read all 2457 lines of guard.py — the lower half could contain code paths that change my plumbing assumptions for cmd_reload, watcher.py interactions, or overflow.py; (b) The synthetic ~1MB fixture spec is plausible but unvalidated against a real Claude Code JSONL — the builder may need to adjust entry shapes during Phase 3 RED.)
- Signals:
  1. `strategies/gentle.py:17-75` — `compact-summary-collapse` strategy with `_META_TYPES = {"last-prompt", "pr-link", "custom-title", "ai-title", "attribution-snapshot"}` (note: NO `permission-mode`) — explains the -98% permission-mode loss
  2. `registry.py:12-18` — `gentle` prescription is what `start_guard` runs at the soft 25% tier, called every 30s
  3. `guard.py:822-845` — soft-prune calls `gentle` every cycle if size exceeds soft threshold
  4. `executor.py:60-119` — `_relink_parent_chain` works correctly (handoff confirmed 0 broken pointers); the bug is semantic loss not structural breakage
  5. `helpers.py:266-290` — `_PROTECTED_TYPES` is type-name-only; doesn't recognize `permission-mode`, `last-prompt`, or "last of type" semantics
  6. `session.py:656-737` — `save_messages` flow including `_FileSnapshot` conflict detection (already battle-tested)
  7. Handoff per-type damage table matches strategy logic 1:1 — the bug is fully explained, no hidden mechanism
- Cross-checked:
  - Read all 4 strategy files; confirmed only `compact-summary-collapse` removes user/assistant content wholesale; all other strategies replace/truncate or remove safe-only types
  - Verified `cli.cmd_treat` already has `--force`-adjacent plumbing (lines 320-334 handle `find_active_background_tasks`); the new `--force` flag matches the existing convention
  - Verified `_PruneLock` and `_FileSnapshot` mechanisms (session.py:103-141, 37-100) — already prevent corruption in flight; our new validation runs PRE-write so it doesn't compete with them
- Not verified:
  - Lines 1271–2457 of guard.py (specifically: post-K=10 exit handling, sentinel write paths, watcher PID/identity logic) — could affect cmd_reload plumbing
  - Whether `overflow.OverflowRecovery.on_file_growth` already handles a "session too active" case in its own logic (Q-B)
  - That the 1MB fixture shape will accurately reproduce the bug on first try; builder may need to iterate
  - Whether Claude Code's resume engine cares about `permission-mode` ordering vs presence (we assume presence-of-last-one is sufficient)
