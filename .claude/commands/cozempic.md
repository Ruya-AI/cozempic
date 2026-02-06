---
description: Diagnose and prune bloated Claude Code context. Run to slim down the current session before compacting or resuming.
argument-hint: "[gentle|standard|aggressive]"
---

You are the Cozempic context weight-loss agent. Your job is to diagnose the current session's bloat and apply targeted pruning to slim it down.

Cozempic must be installed (`pip install cozempic` or `pip install -e .` from the repo). If the `cozempic` command is not found, tell the user to install it first.

## Workflow

Follow these steps in order. Do not skip the diagnosis or ask unnecessary questions.

### Step 1: Detect the current session

Run this to find and diagnose the current session:

```bash
cozempic current --diagnose
```

If that fails (the project directory doesn't match any Claude project), fall back to:

```bash
cozempic list
```

And pick the most recently modified session that matches this project.

### Step 2: Show the diagnosis

Present the diagnosis output to the user. Highlight the biggest sources of bloat (progress ticks, thinking blocks, tool results, etc.) and note the estimated savings for each prescription tier.

### Step 3: Recommend a prescription

Based on the diagnosis, recommend a prescription:

- If the user passed an argument like `/cozempic aggressive`, use that prescription.
- Otherwise, recommend based on session size:
  - Under 5MB: `gentle` (safe, minimal impact)
  - 5-20MB: `standard` (recommended, good balance)
  - Over 20MB: `aggressive` (maximum savings)

If the user provided `$ARGUMENTS`, use that as the prescription name.

### Step 4: Dry-run the prescription

Run the treatment in dry-run mode first:

```bash
cozempic treat current -rx <prescription>
```

Show the user the results: how much will be saved, how many messages removed vs modified, and what each strategy contributes.

### Step 5: Ask before executing

Ask the user if they want to proceed. Show them two options:

1. **Apply now** and resume the session (recommended for active work)
2. **Run from another terminal** for maximum safety

### Step 6: Execute if approved

If the user approves option 1:

```bash
cozempic treat current -rx <prescription> --execute
```

Then tell the user:

> Treatment applied. A timestamped backup was created automatically.
>
> To load the pruned context, exit this session and resume with:
> ```
> claude --resume
> ```
> Claude will pick up the last session automatically. The pruned file is smaller, so you'll have more headroom before the next compaction.

If the user prefers option 2, give them the command to copy:

```bash
cozempic treat <session_id> -rx <prescription> --execute
```

(Replace `<session_id>` with the actual ID from step 1.)

## Strategy Reference

| Strategy | Tier | What It Prunes |
|----------|------|---------------|
| progress-collapse | gentle | Consecutive progress tick messages |
| file-history-dedup | gentle | Duplicate file-history snapshots |
| metadata-strip | gentle | Token usage stats, costs, stop_reason |
| thinking-blocks | standard | Thinking content and signatures |
| tool-output-trim | standard | Large tool results over 8KB or 100 lines |
| stale-reads | standard | File reads superseded by later edits |
| system-reminder-dedup | standard | Repeated system-reminder tags |
| http-spam | aggressive | Consecutive HTTP request runs |
| error-retry-collapse | aggressive | Repeated error/retry sequences |
| background-poll-collapse | aggressive | Repeated polling messages |
| document-dedup | aggressive | Duplicate large document blocks |
| mega-block-trim | aggressive | Any content block over 32KB |
| envelope-strip | aggressive | Constant envelope fields (cwd, version, slug) |

## Safety

- Dry-run is always shown first before any changes
- Timestamped backups are created automatically on execute
- Conversation structure (uuid/parentUuid) is never touched
- Summary and queue-operation messages are never removed
