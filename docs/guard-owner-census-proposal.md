# Guard Owner-Census / Cleanup Proposal (read-only)

**Status: proposal — read-only. This document and the commands it describes do
NOT kill any process.** It describes how to enumerate and classify existing
`cozempic.cli guard` daemons left over by the pre-fix SessionStart hook
(ISSUE-AGT-637), so a FUTURE cleanup lane can act with confidence.

## Background

Before this fix, the SessionStart hook backgrounded its entire spawn subshell.
`cozempic guard --daemon` therefore usually ran after reparenting under PID 1,
`find_claude_pid()` returned `None`, and the daemon argv had no `--claude-pid`.
Because the guard only checks owner exit when `claude_pid` is truthy, those
guards never noticed Claude exiting and accumulated as PPID-1 orphans.

Observed on macOS with Cozempic 1.8.39: 117-119 concurrent PPID-1
`cozempic.cli guard` processes.

This fix makes NEW guards correct (identity-verified owner PID threaded through
the hook, guard exits when that PID dies). It deliberately does NOT add an
auto-kill/reaper that guesses owners. Existing orphans are handled by a
separate, human-gated cleanup lane driven by the census below.

## Data available per guard

| Source | Key | Holds |
|---|---|---|
| `/tmp/cozempic_guard_<slug>.pid` | `slug` = first 12 chars of session UUID | daemon PID (1-line, or 3-line newer format: pid on line 1) |
| `/tmp/cozempic_guard_<slug>.log` | same `slug` | the spawn `CMD:` line, which carries `--claude-pid <pid>` for post-fix daemons (absent for pre-fix orphans) |
| `/tmp/cozempic_guard_<slug>.startup-lock` | same `slug` | flock file (no payload) |

There is **no persisted owner-PID record** on disk today: `_record_claude_identity`
keeps `(pid, start_time)` only in the daemon's in-memory `_CLAUDE_IDENTITY` dict.
The only durable owner evidence is the `--claude-pid` argument in the daemon's
log `CMD:` line (post-fix only).

## Read-only census command (proposal)

A single command enumerates every guard and classifies it. It reads only; it
never signals, kills, or unlinks anything.

```bash
# Census: list every guard daemon with its owner classification. READ-ONLY.
for pid_file in /tmp/cozempic_guard_*.pid; do
  slug=${pid_file##*_}; slug=${slug%.pid}
  daemon_pid=$(head -n 1 "$pid_file" 2>/dev/null | tr -d ' ')
  [ -z "$daemon_pid" ] && continue
  ppid=$(ps -o ppid= -p "$daemon_pid" 2>/dev/null | tr -d ' ')
  log="/tmp/cozempic_guard_${slug}.log"
  # Owner PID = --claude-pid argument in the spawn CMD (post-fix only).
  owner=$(sed -n 's/.*--claude-pid \([0-9]*\).*/\1/p' "$log" 2>/dev/null | head -n 1)
  owner_alive=unknown
  if [ -n "$owner" ]; then
    if kill -0 "$owner" 2>/dev/null; then owner_alive=yes; else owner_alive=no; fi
  fi
  daemon_alive=no; kill -0 "$daemon_pid" 2>/dev/null && daemon_alive=yes
  printf '%s\tdaemon_pid=%s\tppid=%s\talive=%s\towner_pid=%s\towner_alive=%s\n' \
    "$slug" "$daemon_pid" "$ppid" "$daemon_alive" "${owner:-unknown}" "$owner_alive"
done
```

A Python-equivalent driver can reuse `_is_cozempic_guard_process(pid)` (in
`cozempic.guard`) instead of the `ps` comm probe for a stricter identity check.

## Classification

| Class | Condition | Disposition (future lane) |
|---|---|---|
| **Confirmed orphan** | daemon alive, PPID == 1, owner known + `owner_alive=no` | safe to SIGTERM + unlink pidfile |
| **Live / owned** | daemon alive, owner known + `owner_alive=yes` | leave alone |
| **Unverifiable owner** | daemon alive, owner NOT in log (pre-fix) | manual review — do NOT guess a kill target (the entire point of "never substitute a blind PID") |
| **Stale pidfile** | daemon not alive | unlink the pidfile only (recover the session's guard slot) |

The `Unverifiable owner` class is exactly the population this fix's hook change
prevents going forward. Because pre-fix orphans have no recorded owner, an
auto-reaper cannot safely kill them by owner; they must be reviewed by an
operator or matched against the session's own process tree (`claude` processes
for that session UUID, verified via `_is_claude_process`) before any action.

## Explicit non-goals

- No process is killed by this document or its commands.
- No auto-reaper is introduced. Owner guessing is forbidden
  (ISSUE-AGT-637 requirement).
- This lane does not modify hooks, guards, or pidfiles.
