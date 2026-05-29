# Cozempic

![Downloads](https://img.shields.io/badge/downloads-50k%2B-brightgreen) ![Version](https://img.shields.io/badge/version-1.8.17-blue) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

**50,000+ power users** trust Cozempic to keep their Claude Code sessions lean.

Context cleaning for [Claude Code](https://claude.ai/code) — **remove the bloat, keep everything that matters, protect Agent Teams from context loss**.

## What It Does

Claude Code sessions fill up with dead weight: progress ticks, thinking blocks, stale file reads, duplicate CLAUDE.md injections, base64 screenshots, oversized tool outputs, and metadata bloat. A typical session carries 8-46MB — most of it noise that inflates every API call.

Cozempic removes it with **18 composable strategies** across 3 prescription tiers, while your actual conversation, decisions, and working context stay untouched. The guard daemon runs automatically — install once, forget about it.

### Key Features

- **18 pruning strategies** — gentle (5), standard (11), aggressive (18)
- **Guard daemon** — auto-starts via SessionStart hook, monitors and prunes continuously
- **compact-summary-collapse** — 85-95% savings by removing pre-compaction messages already in the summary
- **Agent Teams protection** — checkpoints team state through compaction, reactive overflow recovery
- **Behavioral digest** — extracts your corrections ("don't do X"), persists them to Claude Code's memory system so they survive compaction
- **13 doctor checks** — diagnose and auto-fix session corruption, orphaned tool results, zombie teams
- **Token-aware diagnostics** — exact token counts from `usage` fields, cache hit rate, context % bar
- **Auto-detects 1M context** — correct thresholds for both 200K and 1M models
- **Auto-updates** — checks PyPI daily, upgrades in-place

**Zero external dependencies.** Python 3.10+ stdlib only.

## Install

Pick your package manager:

```bash
# pip (Python ≥ 3.10)
pip install cozempic

# pipx — isolated user install, always available on PATH
pipx install cozempic

# uv / uvx — no install needed, run on demand
uvx cozempic --help

# Homebrew (macOS / Linux)
brew install Ruya-AI/cozempic/cozempic

# Nix flake
nix profile install github:Ruya-AI/cozempic?dir=packaging/nix
```

AUR (`yay -S cozempic`) and MacPorts (`port install py-cozempic`) submissions are in progress — see [`packaging/README.md`](packaging/README.md) for status and PKGBUILD/Portfile sources.

That's it. Cozempic auto-initializes on first use — hooks are wired globally, guard daemon auto-starts on every Claude Code session. No manual setup needed. Opt out with `COZEMPIC_NO_GLOBAL_INIT=1`.

### As a Claude Code Plugin

Install cozempic (any method above), then inside Claude Code:

```
/plugin marketplace add Ruya-AI/cozempic
/plugin install cozempic
```

This gives you MCP tools, skills (`/cozempic:diagnose`, `/cozempic:treat`, etc.), and auto-wired hooks.

## Quick Start

```bash
# Auto-detect and diagnose the current session
cozempic current --diagnose

# Dry-run the standard prescription
cozempic treat current

# Apply with backup
cozempic treat current --execute

# Go aggressive on a specific session
cozempic treat <session_id> -rx aggressive --execute

# Check for session corruption
cozempic doctor

# View behavioral digest rules
cozempic digest show

# Show all strategies & prescriptions
cozempic formulary
```

## Strategies

| # | Strategy | Tier | What It Does | Expected |
|---|----------|------|-------------|----------|
| 1 | `compact-summary-collapse` | gentle | Remove all pre-compaction messages (already in the summary) | 85-95% |
| 2 | `attribution-snapshot-strip` | gentle | Strip attribution-snapshot metadata entries | 0-2% |
| 3 | `progress-collapse` | gentle | Collapse consecutive and isolated progress tick messages | 40-48% |
| 4 | `file-history-dedup` | gentle | Deduplicate file-history-snapshot messages | 3-6% |
| 5 | `metadata-strip` | gentle | Strip token usage stats, stop_reason, costs | 1-3% |
| 6 | `thinking-blocks` | standard | Remove/truncate thinking content + signatures | 2-5% |
| 7 | `tool-output-trim` | standard | Trim large tool results (>8KB or >100 lines), microcompact-aware | 1-8% |
| 8 | `tool-result-age` | standard | Compact old tool results by age — minify mid-age, stub old | 10-40% |
| 9 | `stale-reads` | standard | Remove file reads superseded by later edits | 0.5-2% |
| 10 | `system-reminder-dedup` | standard | Deduplicate repeated system-reminder tags | 0.1-3% |
| 11 | `tool-use-result-strip` | standard | Strip toolUseResult envelope field (Edit diffs, never sent to API) | 5-50% |
| 12 | `image-strip` | aggressive | Strip old base64 image blocks, keep most recent 20% | 1-40% |
| 13 | `http-spam` | aggressive | Collapse consecutive HTTP request runs | 0-2% |
| 14 | `error-retry-collapse` | aggressive | Collapse repeated error-retry sequences | 0-5% |
| 15 | `background-poll-collapse` | aggressive | Collapse repeated polling messages | 0-1% |
| 16 | `document-dedup` | aggressive | Deduplicate large document blocks (CLAUDE.md injection) | 0-44% |
| 17 | `mega-block-trim` | aggressive | Trim any content block over 32KB | safety net |
| 18 | `envelope-strip` | aggressive | Strip constant envelope fields (cwd, version, slug) | 2-4% |

### Prescriptions

| Prescription | Strategies | Risk | Typical Savings |
|---|---|---|---|
| `gentle` | 5 | Minimal | 85-95% (with compact boundary) |
| `standard` | 11 | Low | 25-45% |
| `aggressive` | 18 | Moderate | 35-60% |

**Dry-run is the default.** Nothing is modified until you pass `--execute`. Backups are always created.

## Guard — Continuous Protection

The guard daemon monitors your session and prunes automatically:

```bash
# Auto-starts via SessionStart hook after cozempic init
# Or run manually:
cozempic guard --daemon
```

**4-tier proactive pruning** (every 30s):

| Tier | Threshold | Action | Reload? |
|------|-----------|--------|---------|
| Soft | 25% | gentle file cleanup | No |
| Hard | 55% | standard prune | Yes (deferred if agents active) |
| Emergency | 80% | aggressive prune | Yes (forced) |
| User | 90% | manual aggressive | Yes |

**Reactive overflow recovery** — kqueue/polling file watcher detects inbox-flood overflow within milliseconds, auto-prunes with escalating prescriptions, circuit breaker prevents loops.

**tmux/screen** — reload resumes in the same pane via `send-keys`. Plain terminals open a new window.

**Token thresholds auto-detect** — 200K and 1M models detected automatically. Override with `COZEMPIC_CONTEXT_WINDOW=200000` for Pro plan.

## Behavioral Digest

Cozempic extracts your corrections and persists them across compactions:

```bash
# View extracted rules
cozempic digest show

# Manually extract from current session
cozempic digest update

# Sync rules to Claude Code's memory system
cozempic digest inject
```

**How it works:**
- Detects correction signals in your messages ("don't do X", "stop adding Y", "always use Z")
- All corrections start as "pending" and activate after 2 occurrences (prevents one-shot noise from polluting the digest)
- Rules synced to Claude Code's native memory system (`~/.claude/projects/<cwd>/memory/`)
- Claude reads these as feedback memories on every turn — they survive compaction natively
- PreCompact and Stop hooks auto-extract before context is lost

## Agent Teams Protection

When Claude's auto-compaction fires, Agent Teams lose coordination state. Cozempic prevents this with five layers:

1. **Continuous checkpoint** — saves team state every N seconds
2. **Hook-driven checkpoint** — fires after every Task spawn, TaskCreate/Update, before compaction, at session end
3. **Tiered pruning** — soft threshold trims without disruption; hard threshold does full prune + reload
4. **Reactive overflow recovery** — detects inbox-flood within milliseconds, auto-recovers (~10s downtime)
5. **is_protected()** — compact summaries, compact boundaries, content-replacement entries, and behavioral digest messages are never stripped

## Doctor

```bash
cozempic doctor        # Diagnose issues
cozempic doctor --fix  # Auto-fix where possible
```

| Check | What It Detects | Auto-Fix |
|-------|----------------|----------|
| `trust-dialog-hang` | Resume hangs on Windows | Reset flag |
| `claude-json-corruption` | Truncated/corrupted JSON | Restore from backup |
| `corrupted-tool-use` | `tool_use.name` >200 chars | Parse and repair |
| `orphaned-tool-results` | `tool_result` missing matching `tool_use` — causes 400 errors | Strip orphans |
| `zombie-teams` | Stale team directories with dead agents | Remove stale dirs |
| `oversized-sessions` | Session files >50MB | — |
| `stale-backups` | Old `.jsonl.bak` files wasting disk | Delete old backups |
| `disk-usage` | Session storage exceeding healthy thresholds | — |

## Commands

```
cozempic init                               Wire hooks + slash command into project
cozempic list                               List sessions with sizes and token estimates
cozempic current [-d]                       Show/diagnose current session
cozempic diagnose <session>                 Analyze bloat sources
cozempic treat <session> [-rx PRESET]       Run prescription (dry-run default)
cozempic treat <session> --execute          Apply changes with backup
cozempic strategy <name> <session>          Run single strategy
cozempic reload [-rx PRESET]                Treat + auto-resume in new terminal
cozempic checkpoint [--show]                Save team state to disk
cozempic guard [--daemon]                   Start guard (auto-starts via hook)
cozempic doctor [--fix]                     Check for known issues
cozempic digest [show|update|clear|flush|recover|inject]
cozempic self-update                        Upgrade to latest version from PyPI
cozempic formulary                          Show all strategies & prescriptions
```

## Hook Integration

After `cozempic init`, these hooks are wired automatically:

| Hook | When | What |
|------|------|------|
| `SessionStart` | Session opens | Guard daemon + digest inject |
| `PostToolUse[Task]` | Agent spawn | Team checkpoint |
| `PostToolUse[TaskCreate\|TaskUpdate]` | Todo changes | Team checkpoint |
| `PreCompact` | Before compaction | Checkpoint + digest flush |
| `Stop` | Session end | Checkpoint + digest flush |

## Safety

- **Dry-run by default** — `--execute` required to modify files
- **Atomic writes** — `write → fsync → os.replace()` — no partial writes
- **Strict session resolution** — refuses to act on ambiguous matches
- **Timestamped backups** — automatic `.jsonl.bak` before any modification
- **is_protected()** — compact summaries, boundaries, marble-origami state, content-replacement, behavioral digest entries are never removed
- **parentUuid re-linking** — conversation chain integrity maintained after removals
- **Sibling tool_use protection** — tool_use blocks are kept when their tool_result is kept
- **Team messages protected** — Task, TaskCreate, SendMessage never pruned
- **Strategies compose sequentially** — each runs on the output of the previous
- **Active-session idle guard** — `cozempic treat`/`reload` and the guard daemon refuse to prune sessions modified within the last 24h. Override with `--force` (CLI) or set `COZEMPIC_MIN_IDLE_HOURS=0` (env). Overflow recovery (90% emergency) bypasses the guard automatically.
- **Floor preservation** — each prune is post-processed to guarantee the root message, the last 50 user + 50 assistant turns, and at least 50% of conversation messages survive. The last `ai-title` / `last-prompt` / `permission-mode` singletons are tagged and protected before strategies run.
- **Post-prune structural validation** — six checks (C1–C6) run before `save_messages` writes: parent chain resolves, root preserved, ≥1 user + ≥1 assistant survive, last compact_boundary preserved, last permission-mode preserved, last last-prompt preserved. A failure aborts the save and exits with code 5.

### Tunables

Configuration sources (precedence: env > `~/.cozempic/config.json` > default):

| Knob | Default | Range | Override |
|------|---------|-------|----------|
| `min_idle_hours` | `24.0` | `[0.0, 168.0]` | `COZEMPIC_MIN_IDLE_HOURS` |
| `floor.max_user_assistant_drop_pct` | `0.50` | `[0.0, 1.0]` | `COZEMPIC_FLOOR_MAX_DROP_PCT` |
| `floor.preserve_last_k_turns` | `50` | `[1, 1000]` | `COZEMPIC_FLOOR_PRESERVE_LAST_K` |
| `floor.preserve_first_message` | `true` | `true` / `false` | `COZEMPIC_FLOOR_PRESERVE_FIRST` |

Range endpoints are inclusive. `max_user_assistant_drop_pct=0.0` disables top-up (the floor still preserves the root and last-K turns); `=1.0` disables the floor's survival cap entirely (strategies' user/assistant drops are not clamped). `preserve_first_message=false` lets the root user be dropped on micro-sessions where the operator wants the strategy to win.

Example `~/.cozempic/config.json`:

```json
{
  "min_idle_hours": 12.0,
  "floor": {
    "max_user_assistant_drop_pct": 0.40,
    "preserve_last_k_turns": 100
  }
}
```

### Exit codes (`treat`, `reload`)

| Code | Reason |
|------|--------|
| 0 | Success |
| 1 | Generic error (unknown prescription, missing session) |
| 2 | Another prune cycle holds the lock — retry shortly |
| 3 | Session changed mid-prune (Claude appended new lines) |
| 4 | Active-session guard refused — session modified <24h ago; pass `--force` to override |
| 5 | Post-prune validation failed — pruned file would not replay cleanly |

## Example Output

```
  Prescription: aggressive
  Before: 158.2K tokens (29.56MB, 6602 messages)
  After:  121.5K tokens (23.09MB, 5073 messages)
  Freed:  36.7K tokens (23.2%) — 6.47MB, 1529 removed, 4038 modified
  Context: [============--------] 61%

  Strategy Results:
    compact-summary-collapse       8.17MB saved (85.2%)  (4201 removed)
    progress-collapse              1.63MB saved  (5.5%)  (1525 removed)
    metadata-strip                693.9KB saved  (2.3%)  (2735 modified)
    tool-use-result-strip          1.44MB saved  (4.9%)  (891 modified)
    thinking-blocks                1.11MB saved  (3.8%)  (1127 modified)
    tool-output-trim               1.72MB saved  (5.8%)  (167 modified)
    ...
```

## Changelog

### v1.7.1

- **`cozempic reload --session <id|path>`** escape hatch when auto-detect fails in multi-agent sessions. Previously reload had no way to recover from ambiguous session detection, leaving users stuck. Matches `guard --session`.
- Error message now names the flag to use (was "use an explicit session ID" with no instruction on how)
- **Auto-update message clarified**: after upgrade, says "active on next run (this process still vX.Y.Z)" — users no longer think the upgrade failed when `--version` still prints the old number (the running Python process can't hot-swap its own code)

### v1.7.0

- **Telemetry opt-out**: `COZEMPIC_NO_TELEMETRY=1` disables anonymous usage counters
- **Documented configuration**: `COZEMPIC_NO_AUTO_UPDATE`, `COZEMPIC_NO_TELEMETRY`, `COZEMPIC_CONTEXT_WINDOW` env vars

### v1.6.x

- **4-tier pruning**: soft (25%, no reload) → hard (55%, reload) → emergency (80%, aggressive reload) → user (90%, manual)
- **Agent-aware reload**: defers reload at 55% when agents are running, forces at 80%
- **Same-terminal resume**: tmux/screen users get `/exit` + `claude --resume` in the same pane
- **Clean messaging**: only shows strategies that did something, 1-line hook status output
- **1M default**: Opus/Sonnet 4.5/4.6 default to 1M context (CC doesn't use `[1m]` suffix)
- **Auto-upgrade everywhere**: SessionStart hook backgrounds `pip install --upgrade cozempic` on every session. MCP/plugin use `uv run --upgrade`. npm install.js always upgrades.
- **`cozempic self-update`**: force-upgrade from PyPI regardless of install method (pip, uv, editable, clone)
- **Auto-updater fixed**: removed TTY check (was blocking hook-triggered updates), tries uv → pip → pipx

### v1.5.0

- **`tool-result-age` strategy** — age-based tool result compaction. Recent results stay verbatim, mid-age get JSON minified and diff context collapsed, old replaced with compact stubs. Claude can re-read any file. 10-40% additional savings targeting the 45% of session size that tool results occupy.
- 18 strategies total, standard prescription 11, aggressive 18
- Tests: 273 → 283

### v1.4.0 / v1.4.1

- **Track 1 — Bug fixes**: `is_protected()` guard on all strategies, `isSidechain` preserved in envelope-strip, `output_tokens` in token formula, `parentUuid` re-linking, sibling tool_use protection
- **Track 2 — New strategies**: `compact-summary-collapse` (85-95%), `attribution-snapshot-strip`, microcompact-aware `tool-output-trim`
- **Behavioral digest**: extract corrections, sync to Claude Code memory, CLI commands, hook wiring
- **Context window detection**: MCP server and plugin now auto-detect 200K/1M (was hardcoded 200K)
- **Cache efficiency metrics**: `cozempic diagnose` shows cache hit rate
- **transcript_path**: hooks parse session path from payload for faster resolution
- Tests: 165 → 273

### v1.3.0 / v1.3.1

- Writer-safe live prune + sidecar session store
- Guard startup cleanup, updater fixes, MCP maintenance

### v1.2.0 — v1.2.8

- Atomic file writes, strict session resolution, schema-first team detection
- tool-use-result-strip strategy (5-50% on edit-heavy sessions)
- image-strip strategy (keep last 20%)
- Auto-update, install tracking, npm package
- Safety improvements: SIGTERM handler, backup cleanup, permission error handling

## Configuration

| Variable | Default | Effect |
|----------|---------|--------|
| `COZEMPIC_CONTEXT_WINDOW` | auto-detect | Override context window size (e.g. `200000` for Pro plan) |
| `COZEMPIC_NO_AUTO_UPDATE` | off | Skip automatic version checks. Not recommended — Claude Code ships frequent changes and cozempic updates keep strategies compatible with the latest session format. |
| `COZEMPIC_NO_TELEMETRY` | off | Skip anonymous usage counters. Cozempic pings a simple counter on each prune — no personal data, session content, or identifiable information is sent. Helps us prioritize development. |

## Contributing

Contributions welcome. To add a strategy:

1. Create a function in the appropriate tier file under `src/cozempic/strategies/`
2. Decorate with `@strategy(name, description, tier, expected_savings)`
3. Return a `StrategyResult` with a list of `PruneAction`s
4. Add to the appropriate prescription in `src/cozempic/registry.py`

## License

MIT — see [LICENSE](LICENSE).

Built by [Ruya AI](https://ruya.ai).
