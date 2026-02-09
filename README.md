# Cozempic

Context cleaning for [Claude Code](https://claude.ai/code) — **remove the bloat, keep everything that matters**.

### What gets removed

Claude Code context fills up with dead weight that wastes your token budget: hundreds of progress tick messages, repeated thinking blocks and signatures, stale file reads that were superseded by edits, duplicate document injections, oversized tool outputs, and metadata bloat (token counts, stop reasons, cost fields). A typical session carries 8-46MB — most of it noise. Cozempic identifies and removes all of it using 13 composable strategies, while your actual conversation, decisions, tool results, and working context stay untouched.

### Agent Teams context loss protection

When context gets too large, Claude's auto-compaction summarizes away critical state. For **Agent Teams**, this is catastrophic: the lead agent's context is compacted, team coordination messages (TeamCreate, SendMessage, TaskCreate/Update) are discarded, the lead forgets its teammates exist, and subagents are orphaned with no recovery path. ([#23620](https://github.com/anthropics/claude-code/issues/23620), [#23821](https://github.com/anthropics/claude-code/issues/23821), [#24052](https://github.com/anthropics/claude-code/issues/24052), [#21925](https://github.com/anthropics/claude-code/issues/21925))

Cozempic's **guard mode** prevents this entirely — a background daemon that continuously cleans dead weight so auto-compaction never triggers, while protecting every team message and automatically repairing team state if context is reset.

**Zero external dependencies.** Python 3.10+ stdlib only.

## Install

```bash
pip install cozempic
```

Or run directly:

```bash
git clone https://github.com/Ruya-AI/cozempic.git
cd cozempic
pip install -e .
```

## Quick Start

```bash
# List all sessions with sizes
cozempic list

# Auto-detect and diagnose the current session
cozempic current --diagnose

# Dry-run the standard prescription on current session
cozempic treat current

# Apply with backup
cozempic treat current --execute

# Go aggressive on a specific session
cozempic treat <session_id> -rx aggressive --execute

# Keep context clean automatically — protect Agent Teams (run in a separate terminal)
cozempic guard --threshold 50 -rx standard
```

Session IDs accept full UUIDs, UUID prefixes, file paths, or `current` for auto-detection based on your working directory.

## How It Works

Each type of bloat has a dedicated **strategy** that knows exactly what to remove and what to keep. Strategies are grouped into **prescriptions** — presets that balance cleaning depth against risk:

| Prescription | Strategies | Risk | Typical Savings |
|---|---|---|---|
| `gentle` | 3 | Minimal | 5-8% |
| `standard` | 7 | Low | 15-20% |
| `aggressive` | 13 | Moderate | 20-25% |

**Dry-run is the default.** Nothing is modified until you pass `--execute`. Backups are always created automatically.

## Strategies

| # | Strategy | What It Does | Expected |
|---|----------|-------------|----------|
| 1 | `progress-collapse` | Collapse consecutive progress tick messages | 40-48% |
| 2 | `file-history-dedup` | Deduplicate file-history-snapshot messages | 3-6% |
| 3 | `metadata-strip` | Strip token usage stats, stop_reason, costs | 1-3% |
| 4 | `thinking-blocks` | Remove/truncate thinking content + signatures | 2-5% |
| 5 | `tool-output-trim` | Trim large tool results (>8KB or >100 lines) | 1-8% |
| 6 | `stale-reads` | Remove file reads superseded by later edits | 0.5-2% |
| 7 | `system-reminder-dedup` | Deduplicate repeated system-reminder tags | 0.1-3% |
| 8 | `http-spam` | Collapse consecutive HTTP request runs | 0-2% |
| 9 | `error-retry-collapse` | Collapse repeated error-retry sequences | 0-5% |
| 10 | `background-poll-collapse` | Collapse repeated polling messages | 0-1% |
| 11 | `document-dedup` | Deduplicate large document blocks | 0-44% |
| 12 | `mega-block-trim` | Trim any content block over 32KB | safety net |
| 13 | `envelope-strip` | Strip constant envelope fields | 2-4% |

Run a single strategy:

```bash
cozempic strategy progress-collapse <session_id> -v
cozempic strategy thinking-blocks <session_id> --thinking-mode truncate
```

## Commands

```
cozempic list [--project NAME]          List sessions with sizes
cozempic current [-d]                   Show/diagnose current session (auto-detect)
cozempic diagnose <session>             Analyze bloat sources (read-only)
cozempic treat <session> [-rx PRESET]   Run prescription (dry-run default)
cozempic treat <session> --execute      Apply changes with backup
cozempic strategy <name> <session>      Run single strategy
cozempic reload [-rx PRESET]            Treat + auto-resume in new terminal
cozempic guard [--threshold MB]         Protect & repair Agent Teams context (background)
cozempic doctor [--fix]                 Check for known Claude Code issues
cozempic formulary                      Show all strategies & prescriptions
```

Use `current` as the session argument in any command to auto-detect the active session for your working directory.

## Guard — Agent Teams Context Loss Protection & Repair

Guard is a background daemon that **protects** and **repairs** Agent Teams context. Run it in a separate terminal and forget about it.

```bash
# Protect Agent Teams — run in a separate terminal
cozempic guard --threshold 50 -rx standard

# Without auto-reload (just clean, no restart)
cozempic guard --threshold 50 --no-reload

# Lower threshold, faster checks
cozempic guard --threshold 30 --interval 15
```

**Protection** — prevents context loss before it happens:

1. Monitors context size every 30 seconds
2. When the threshold is approached, cleans dead weight using the same strategies as `treat`
3. Team coordination messages (TeamCreate, SendMessage, TaskCreate/Update) are **never removed**
4. Context stays under threshold — auto-compaction never fires

**Repair** — recovers team state if context is reset:

5. Before cleaning, **extracts full team state** — teammates, tasks, roles, coordination history
6. Writes a crash-safe checkpoint to `.claude/team-checkpoint.md`
7. **Injects team state directly into the context** as a synthetic message pair — Claude *sees* the team as conversation history when it resumes (force-read, not a suggestion)
8. Triggers auto-reload (kill + resume in new terminal) so Claude picks up the clean context with team state intact

**The result:** Your context stays clean. Everything valuable is preserved — conversation history, decisions, tool results, and full Agent Teams coordination. Auto-compaction never fires. No orphaned subagents, no lost context.

## Doctor

Beyond context cleaning, Cozempic can check for known Claude Code configuration issues:

```bash
cozempic doctor        # Diagnose issues
cozempic doctor --fix  # Auto-fix where possible
```

Current checks:

| Check | What It Detects |
|-------|----------------|
| `trust-dialog-hang` | `hasTrustDialogAccepted=true` in `~/.claude.json` causing resume hangs on Windows |
| `oversized-sessions` | Session files >50MB likely to hang on resume |
| `stale-backups` | Old `.bak` files from previous treatments wasting disk |
| `disk-usage` | Total session storage exceeding healthy thresholds |

The `--fix` flag auto-applies fixes where safe (e.g., resetting the trust dialog flag, cleaning stale backups). Backups are created before any config modification.

## Claude Code Integration

### Slash Command

Cozempic ships with a `/cozempic` slash command for Claude Code. Install it by copying the command file to your user-level commands directory:

```bash
cp .claude/commands/cozempic.md ~/.claude/commands/cozempic.md
```

Then from any Claude Code session, type `/cozempic` to diagnose and treat the current session interactively. You can also pass a prescription directly: `/cozempic aggressive`.

After treatment, exit and resume the session to load the pruned context:

```bash
claude --resume
```

### SessionStart Hook (Optional)

To persist the session ID as an environment variable for use in scripts and other hooks:

```bash
cp .claude/hooks/persist-session-id.sh ~/.claude/hooks/
chmod +x ~/.claude/hooks/persist-session-id.sh
```

Add to your `.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "~/.claude/hooks/persist-session-id.sh"
      }]
    }]
  }
}
```

This makes `$CLAUDE_SESSION_ID` available in all Bash commands during the session.

## Safety

- **Always dry-run by default** — `--execute` flag required to modify files
- **Timestamped backups** — automatic `.bak` files before any modification
- **Never touches uuid/parentUuid** — conversation DAG stays intact
- **Never removes summary/queue-operation messages** — structurally important
- **Team messages are protected** — guard mode never prunes TeamCreate, SendMessage, TaskCreate/Update
- **Strategies compose sequentially** — each runs on the output of the previous, so savings are accurate and don't overlap

## Example Output

```
  Prescription: aggressive
  Before: 29.56MB (6602 messages)
  After:  23.09MB (5073 messages)
  Saved:  6.47MB (21.9%) — 1529 removed, 4038 modified

  Strategy Results:
    progress-collapse              1.63MB saved  (5.5%)  (1525 removed)
    file-history-dedup              2.0KB saved  (0.0%)  (4 removed)
    metadata-strip                693.9KB saved  (2.3%)  (2735 modified)
    thinking-blocks                 1.11MB saved  (3.8%)  (1127 modified)
    tool-output-trim               1.72MB saved  (5.8%)  (167 modified)
    stale-reads                   710.0KB saved  (2.3%)  (176 modified)
    system-reminder-dedup          27.6KB saved  (0.1%)  (92 modified)
    envelope-strip                509.2KB saved  (1.7%)  (4657 modified)
```

## Contributing

Contributions welcome. To add a strategy:

1. Create a function in the appropriate tier file under `src/cozempic/strategies/`
2. Decorate with `@strategy(name, description, tier, expected_savings)`
3. Return a `StrategyResult` with a list of `PruneAction`s
4. Add to the appropriate prescription in `src/cozempic/registry.py`

```python
from cozempic.registry import strategy
from cozempic.types import Message, PruneAction, StrategyResult

@strategy("my-strategy", "What it does", "standard", "1-5%")
def my_strategy(messages: list[Message], config: dict) -> StrategyResult:
    actions = []
    # ... analyze messages, build PruneAction list ...
    return StrategyResult(
        strategy_name="my-strategy",
        actions=actions,
        # ...
    )
```

## License

MIT - see [LICENSE](LICENSE).

Built by [Ruya AI](https://ruya.ai).
