# Cozempic

Context weight-loss tool for [Claude Code](https://claude.ai/code). Prune bloated JSONL conversation files using composable strategies.

Claude Code sessions grow large — 8-46MB from progress ticks, repeated thinking blocks, stale file reads, duplicate document injections, and metadata bloat. Cozempic diagnoses the bloat and applies targeted pruning strategies to slim them down.

**Zero external dependencies.** Python 3.10+ stdlib only.

## Install

```bash
pip install cozempic
```

Or run directly:

```bash
git clone https://github.com/junaidtitan/cozempic.git
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
```

Session IDs accept full UUIDs, UUID prefixes, file paths, or `current` for auto-detection based on your working directory.

## How It Works

Cozempic uses **strategies** — pure functions that analyze messages and produce declarative prune actions (remove or replace). Strategies are grouped into **prescriptions**:

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
cozempic formulary                      Show all strategies & prescriptions
```

Use `current` as the session argument in any command to auto-detect the active session for your working directory.

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
