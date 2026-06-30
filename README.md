# Cozempic

![Downloads](https://img.shields.io/badge/downloads-100k%2B-brightgreen) ![Version](https://img.shields.io/badge/version-1.8.38-blue) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

**100,000+ power users** trust Cozempic to keep their Claude Code sessions lean.

Context cleaning for [Claude Code](https://claude.ai/code) — **remove the bloat, keep everything that matters, protect Agent Teams from context loss**.

## What It Does

Claude Code sessions fill up with dead weight: progress ticks, thinking blocks, stale file reads, duplicate CLAUDE.md injections, base64 screenshots, oversized tool outputs, and metadata bloat. A typical session carries 8-46MB — most of it noise that inflates every API call.

Cozempic removes it with **18 composable strategies** across 3 prescription tiers, while your actual conversation, decisions, and working context stay untouched. The guard daemon runs automatically — install once, forget about it.

### Key Features

- **18 pruning strategies** — gentle (5), standard (11), aggressive (18)
- **Guard daemon** — auto-starts via SessionStart hook, monitors and prunes continuously
- **Interactive "prune now?" nudge** — a non-blocking heads-up at 25% / 55% / 80% context (once per tier) recommending `cozempic reload`, so interactive users get cozempic's higher-fidelity prune+resume on their own terms instead of falling back to lossy autocompact. Takes no action on its own; silence with `COZEMPIC_NUDGE_OFF=1`
- **Interactive-safe reload** — in interactive sessions the guard warns first and reloads only at an idle breakpoint (never mid-turn); headless sessions reload as before
- **Safe-point protection** — the guard never terminates-and-resumes through in-flight work: a running Workflow, a background subagent, an agent team, or an open tool call defers the reload so nothing is lost
- **compact-summary-collapse** — 85-95% savings by removing pre-compaction messages already in the summary
- **Agent Teams protection** — checkpoints team state through compaction, reactive overflow recovery
- **Behavioral digest** — extracts your corrections ("don't do X"), persists them to Claude Code's memory system so they survive compaction
- **15 doctor checks** — diagnose and auto-fix session corruption, orphaned tool results, zombie teams, unresumable sessions
- **Token-aware diagnostics** — exact token counts from `usage` fields, cache hit rate, context % bar
- **Auto-detects 1M context** — correct thresholds for both 200K and 1M models
- **Efficient idle polling** — backs off the poll cadence when the session is quiet and skips redundant no-op checkpoints
- **Auto-updates** — checks PyPI daily, upgrades in-place

**Zero external dependencies.** Python 3.10+ stdlib only.

## Install

Pick your package manager. `uvx`/`pipx`/`npm`/`pip` are the lowest-friction (no Homebrew trust prompt — see the note below):

```bash
# uv — persistent install, always on PATH (recommended)
uv tool install cozempic

# uvx — run once without installing (handy to try it, but the guard daemon
# can't auto-start from an ephemeral env — use `uv tool install` for that)
uvx cozempic --help

# pipx — isolated user install, always on PATH
pipx install cozempic

# npm — global install
npm install -g cozempic

# pip (Python ≥ 3.10)
pip install cozempic

# Homebrew (macOS / Linux) — use the fully-qualified name (see note below)
brew install Ruya-AI/cozempic/cozempic

# Nix flake
nix profile install github:Ruya-AI/cozempic?dir=packaging/nix
```

> **Homebrew tap trust:** recent Homebrew versions gate non-official taps behind an explicit trust step. The **fully-qualified** `brew install Ruya-AI/cozempic/cozempic` above trusts just this formula inline — no extra command. A bare `brew install cozempic` / `brew upgrade cozempic` will instead error with *"Refusing to load formula … from untrusted tap"*; if you hit that, run `brew trust Ruya-AI/cozempic` once (or just prefer one of the install methods above, none of which have this gate).

AUR (`yay -S cozempic`) and MacPorts (`port install py-cozempic`) submissions are in progress — see [`packaging/README.md`](packaging/README.md) for status and PKGBUILD/Portfile sources.

That's it. Cozempic auto-initializes on first use — hooks are wired globally, guard daemon auto-starts on every Claude Code session. No manual setup needed. Opt out with `COZEMPIC_NO_GLOBAL_INIT=1`.

### Auto-update & how to control it

Cozempic auto-updates from PyPI by default, **on purpose**: Claude Code ships frequent changes to its session/context format, and auto-update is how a compatibility fix reaches you *before* a stale release can mishandle and lose your context. For most users that safety outweighs the convenience cost, so we recommend leaving it on. If you'd rather control it, set one of these — ideally in your shell profile (`~/.zshrc` / `~/.bashrc`) so it applies *before* the first session wires anything:

| Goal | Set |
|---|---|
| Hold a specific reviewed version, keep hooks + guard running | `export COZEMPIC_PIN=1.8.30` |
| Stop all auto-updates, keep hooks + guard running | `export COZEMPIC_NO_AUTO_UPDATE=1` |
| No global hooks / no daemon at all (manual CLI only) | `export COZEMPIC_NO_GLOBAL_INIT=1` |

Both `COZEMPIC_NO_AUTO_UPDATE` and `COZEMPIC_PIN` are honored by the Python updater, the SessionStart hook's shell upgrade, and the npm installer. `COZEMPIC_PIN` disables auto-update and warns (not auto-installs) if your running version drifts from the pin, so you stay on a version you've reviewed with a human in the loop. (Releases are published from CI via PyPI [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) — no long-lived upload token — to reduce the chance of a compromised publish in the first place.)

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
| Hard | 55% | standard prune | Yes (interactive: at a breakpoint; deferred if agents active) |
| Hard2 | 80% | aggressive prune | Yes (gated by the safe-point check) |
| User | 90% | manual aggressive | Yes |

**Interactive sessions** — instead of a surprise reload mid-work, the guard surfaces a [nudge](#key-features) and reloads only once you pause between turns, after warning you. Near the wall (≈88%) it reloads even mid-turn — a higher-fidelity prune beats hitting autocompact. Detection is automatic (`COZEMPIC_INTERACTIVE=auto`); `on`/`off` force it. Headless/CI sessions reload immediately as before.

**Safe-point reload** — a reload terminates and resumes the Claude process, so the guard validates first: if a Workflow, a background subagent, an agent team, or an open tool call is in flight, the reload defers (read-only checkpoint) rather than destroying that work. Tune the near-wall force point with `COZEMPIC_FORCE_RELOAD_PCT` (default `0.88`).

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
cozempic dashboard                          Build a static HTML report of your prune savings
cozempic self-update                        Upgrade to latest version from PyPI
cozempic formulary                          Show all strategies & prescriptions
cozempic uninstall [--project|--all|--purge]  Reverse init: unwire hooks + slash command
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

### v1.8.23

- **Hardened numeric input validation** — `NaN`, `infinity`, and non-representable huge integers are now rejected with a clear error at every CLI flag, `COZEMPIC_*` env var, and config field (a `NaN` threshold would otherwise silently disable the gate it controls). Thanks to **[@ynaamane](https://github.com/ynaamane)** (#116), and folded in the matching fix for the interactive-guard reload-grace knob
- **Standing adversarial QA fleet** — `docs/qa-fleet.md` + a corpus-driven regression test so this whole class can't regress

### v1.8.22

- **Interactive "prune now?" nudge** — non-blocking heads-up at 25% / 55% / 80% context (once per tier, with hysteresis so it never nags), recommending `cozempic reload`. Brings cozempic's higher-fidelity prune+resume to interactive sessions without surprise reloads. Tunable via `COZEMPIC_NUDGE_PCTS`; silence with `COZEMPIC_NUDGE_OFF=1`
- **Interactive-safe reload** — warns first, then reloads only at an idle breakpoint (never mid-turn); near the wall (`COZEMPIC_FORCE_RELOAD_PCT`, default 88%) it reloads even mid-turn. Headless/CI behaviour unchanged
- **Safe-point protection** — the guard never terminates-and-resumes through a running Workflow, background subagent, agent team, or open tool call; the reload defers instead so in-flight work is preserved
- **Interactivity detection** — `COZEMPIC_INTERACTIVE=auto|on|off`
- **Efficient idle polling** — exponential poll back-off when the session is quiet (`COZEMPIC_IDLE_BACKOFF_CYCLES`) and skips redundant no-op SOFT checkpoints

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
| `COZEMPIC_NO_AUTO_UPDATE` | off | Disable all automatic upgrades (honored by the Python updater, SessionStart hook, and npm installer). Not generally recommended — Claude Code ships frequent changes and cozempic updates keep strategies compatible with the latest session format. See [Auto-update & how to control it](#auto-update--how-to-control-it). |
| `COZEMPIC_PIN` | unset | Hold a reviewed version (e.g. `1.8.30`): disables auto-update on both paths and warns on drift instead of auto-installing. For users who want the protection on a version they've vetted. |
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
