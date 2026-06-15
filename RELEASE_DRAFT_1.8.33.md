# Cozempic v1.8.33 — DRAFT (unreleased)

> Status: on branch `remediation/audit-p1`, PR #138 (open, MERGEABLE). **Landing only — not yet released to any channel.** Verified across 6 adversarial QA-fleet gates run to convergence; full test suite green (1439 passed; 3 known env-flaky deselected).

Whole-codebase audit remediation: closes every P1 from the 2026-06-15 adversarial QA-fleet audit and folds in three contributor PRs.

## 🔒 Data integrity (silent-data-loss fixes)
- **Equal-size overwrite** — `_FileSnapshot.classify()` re-hashes on an equal-*size* file instead of assuming "unchanged", so an in-place rewrite by Claude of the same byte length is no longer silently clobbered on prune-save.
- **Read-once snapshots (TOCTOU ×2)** — a line appended in the window between snapshot and load could be duplicated, and a concurrent rewrite between `classify` and `read_delta` could merge an unvalidated tail. New `_FileSnapshot.from_bytes` / `load_messages_and_snapshot` (single read) and `classify_and_delta` (single read) close both windows; routed through all 6 mutate-then-save call sites.
- **Unicode-separator corruption** — JSONL parsing used `str.splitlines()`, which splits on U+2028/U+2029/U+0085 (legal inside JSON strings, emitted unescaped by JS `JSON.stringify`), tearing a valid line into broken fragments. New `_split_physical_lines` (splits on `\n`/`\r` only) applied to all 4 raw-JSONL parse paths.
- **`tool-result-age`** — diff-collapse only collapses context *inside* a real unified-diff hunk, so indented non-diff tool output (git-log, CI, config) is preserved verbatim.
- **doctor `fix_corrupted_tool_use`** — snapshots + append-merges instead of clobbering, so a turn appended mid-repair is preserved.

## 🛡️ Daemon resilience
- **Crash on malformed `usage`** — a null/string token value no longer crashes the guard; `_as_int` coercion + a per-cycle loop-body guard keep one bad cycle from killing the daemon.
- **Guard-loop watchdog** — fixed the K/M/comma-suffixed prune-line regex so a healthy daemon is no longer false-flagged as looping; with the merged #128 identity gate, `guard-watchdog --fix` can no longer SIGTERM a productive guard.

## 🪟 Windows
- **Auto-resume** — replaced the dead cmd-in-bash reload-watcher with a PowerShell-native watcher.
- **ReDoS budget** — the `--protect-pattern` match budget (POSIX-only SIGALRM) now fails *closed* on Windows: a catastrophic-backtracking pattern is refused up front instead of freezing the daemon.

## 🧹 Security & correctness
- **Digest prompt-injection** — untrusted rule/evidence text is sanitized (single-line, markdown-defanged) at every sink that writes into Claude's memory, including the `cmd_remind` hook.
- **Overflow recovery** — added a post-prune *token*-axis preflight (was byte-only) and widened the overflow-marker set.
- **doctor `--fix`** — reports "fixed" only when a re-check confirms the issue is gone (no more false success on a no-op/failed fix).

## 📦 Merged contributor PRs (@ynaamane)
- **#130** — `enforce_floor` no longer forks the conversation DAG on a compacted session
- **#129** — sub-agent (sidechain) turns excluded from learned corrections
- **#128** — guard-identity verification before a `--fix` SIGTERM

## 🧰 Internal
- Consolidated three atomic-write implementations into one hardened helper.
- Replaced false-confidence static tests (MCP `treat_session`, watchdog) with behavioral ones.
- +60-odd regression tests across the touched subsystems.

---

## Confidence ledger (for reviewers)
- **Full confidence** (offline-verified + regression-tested): everything except the two below.
- **Emulation-only**: the two Windows fixes — no Windows CI runner; verified by emulating the no-SIGALRM / `os.name == 'nt'` paths.
- **Artifact-blocked**: the overflow-marker widening is strictly broader (cannot regress detection) but full confidence needs a captured **real** overflowed-CC JSONL tail to pin the persisted marker text (flagged in-code).

## Still pending
**To land:** merge PR #138; cut the release (PyPI / npm / GitHub / Homebrew tap / packaging mirrors) as a separate deliberate step.
**To lift the caveats:** add a `windows-latest` CI job; capture a real overflowed-CC JSONL fixture.
**Fast-follow:** a few P2/P3 nits ranked below the ship line; move `packaging/ci/publish.yml` (Trusted Publishing) into `.github/workflows/` once a `workflow`-scoped token is available.
**Out of scope this run:** PRs #131–137; the de-scoped npm `COZEMPIC_NO_GLOBAL_INIT` opt-out; the team-checkpoint→post-compact untrusted-text surface (distinct pre-existing class).
