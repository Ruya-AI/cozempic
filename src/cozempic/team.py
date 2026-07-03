"""Agent team state extraction, checkpointing, and recovery injection.

Scans JSONL session files for agent team coordination patterns:
- Task tool calls (subagent spawns with subagent_type, prompt, description)
- task-notification messages (actual agent results, status, summaries)
- TaskCreate/TaskUpdate/TaskList/TaskGet (shared todo list)
- TaskOutput (background agent results)
- TeamCreate/SendMessage (explicit team coordination)

Injects team state back into a pruned session so that Claude resumes
with full team awareness.
"""

from __future__ import annotations

import json
import re
import uuid as uuid_mod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .types import Message

# Terminal (finished) statuses for teammate classification in build_team_recovery_receipt.
# Duplicated from guard._STATUS_TERMINAL to avoid a circular import (guard imports team).
# ADD a comment here if guard._STATUS_TERMINAL changes so the two stay in sync.
_TEAMMATE_QUIESCENT: frozenset[str] = frozenset({
    "completed", "complete", "done", "failed", "cancelled",
    "canceled", "stopped", "killed", "aborted", "error",
    "success", "succeeded", "finished", "timeout", "timed_out",
    "ok",
})


def _sfield(d: dict, *keys: str, default: str = "") -> str:
    """First present-and-non-empty STRING value among *keys in untrusted tool-input
    dict *d, else *default. Tool-input fields in poisoned/malformed JSONL can be any
    JSON type — a non-str prompt crashes ``prompt[:200]`` (TypeError 'int' not
    subscriptable) and an unhashable list/dict crashes ``task_id in seen`` / dict-key
    use (TypeError). Every string/slice/.strip()/dict-key read in extract_team_state
    routes through this so a single bad field can't crash the extractor and wedge the
    guard into a respawn storm (R4 finding team-input-field-crash). Mirrors the
    ``x or y or default`` semantics the call sites relied on (skips empty strings)."""
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v:
            return v
    return default


@dataclass
class SubagentInfo:
    """Information about a spawned subagent (Task tool call)."""

    agent_id: str
    description: str = ""
    subagent_type: str = ""
    status: str = "running"  # running, completed, failed
    result_summary: str = ""


@dataclass
class TeammateInfo:
    """Information about a named teammate (explicit team or config.json)."""

    agent_id: str
    name: str
    role: str = ""
    status: str = "unknown"  # running, done, idle
    model: str = ""
    cwd: str = ""


@dataclass
class TaskInfo:
    """Information about a task in the shared task list."""

    task_id: str
    subject: str
    status: str = "pending"
    owner: str = ""
    description: str = ""
    # True once this entry is backed by a real TaskCreate (vs. a placeholder
    # synthesized from an out-of-order TaskUpdate). Used to classify id-reuse
    # (real prior generation → overwrite) vs. a subject-less update fragment
    # (→ merge, keep its authoritative status) without inferring from `subject`.
    from_create: bool = False


@dataclass
class TeamState:
    """Extracted state of an agent team from conversation history."""

    team_name: str = ""
    lead_agent_id: str = ""
    lead_session_id: str = ""
    config_source: str = ""  # "config.json", "jsonl", or "both"
    teammates: list[TeammateInfo] = field(default_factory=list)
    subagents: list[SubagentInfo] = field(default_factory=list)
    tasks: list[TaskInfo] = field(default_factory=list)
    lead_summary: str = ""
    message_count: int = 0
    last_coordination_index: int = -1

    def is_empty(self) -> bool:
        return (
            not self.team_name
            and not self.teammates
            and not self.subagents
            and not self.tasks
        )

    def _task_groups(self) -> tuple[list[TaskInfo], int, int]:
        """Split tasks into active work and low-value completed/blank noise."""
        active: list[TaskInfo] = []
        completed = 0
        blank = 0
        # Terminal task states. Kept consistent with the teammate/subagent terminal
        # concept (e.g. "finished" was terminal there but counted ACTIVE here → an
        # "active task in flight" over-block; fleet P2, 2026-06-09). Only unambiguously
        # terminal words — never an active-work state (pending/in_progress/blocked).
        inactive_statuses = {
            "completed", "complete", "done", "finished", "cancelled", "canceled",
            "closed", "resolved", "skipped", "merged", "archived", "obsolete",
            "wont_fix", "wontfix",
        }
        for task in self.tasks:
            subject = (task.subject or "").strip()
            if not subject:
                blank += 1
                continue
            if (task.status or "").strip().lower() in inactive_statuses:
                completed += 1
                continue
            active.append(task)
        return active, completed, blank

    @staticmethod
    def _san(text) -> str:
        """Sanitize untrusted team-derived text before it lands in a Claude-readable
        checkpoint/recovery surface — the sibling of the digest _sanitize_for_injection
        fix (result_summary/lead_summary/subject/description/name come from tool
        results and team messages and were embedded verbatim, so a multi-line /
        markdown-structured value could inject into CC memory). Lazy import avoids
        any import-order coupling with digest."""
        if not text:
            return ""
        try:
            from .digest import _sanitize_for_injection
            s = _sanitize_for_injection(str(text))
        except Exception:
            # Fail safe: at minimum collapse newlines so injection can't add lines.
            import re as _re
            s = _re.sub(r"\s+", " ", str(text)).strip()
        # Replace any LONE SURROGATE (from a surrogateescape-decoded non-UTF-8 transcript
        # byte) with U+FFFD so the rendered checkpoint is clean UTF-8 (R15): a surrogate
        # in the markdown made the STRICT checkpoint write/read raise UnicodeEncode/Decode
        # — the R14 surrogatepass write merely RELOCATED that crash to read_team_checkpoint
        # / the PostCompact hook. Sanitizing at this single render chokepoint keeps the
        # file strict-UTF-8 clean for both cozempic AND Claude Code's own reader, with no
        # WTF-8 round-trip. A lone surrogate has no display value anyway.
        if any(0xD800 <= ord(c) <= 0xDFFF for c in s):
            s = "".join("�" if 0xD800 <= ord(c) <= 0xDFFF else c for c in s)
        return s

    def to_markdown(self) -> str:
        """Render team state as markdown for checkpoint file."""
        lines = []
        lines.append(f"# Agent Team Checkpoint: {self._san(self.team_name) or 'unnamed'}")
        lines.append(f"_Generated: {datetime.now().isoformat()}_")
        if self.config_source:
            lines.append(f"_Source: {self._san(self.config_source)}_")
        lines.append("")

        if self.lead_agent_id or self.lead_session_id:
            lines.append(f"**Lead:** `{self._san(self.lead_agent_id)}` (session: `{self._san(self.lead_session_id)[:12]}...`)")
            lines.append("")

        if self.teammates:
            lines.append("## Teammates")
            for t in self.teammates:
                status = f" ({self._san(t.status)})" if t.status != "unknown" else ""
                role = f" — {self._san(t.role)}" if t.role else ""
                model = f" [{self._san(t.model)}]" if t.model else ""
                cwd = f" cwd: {self._san(t.cwd)}" if t.cwd else ""
                lines.append(f"- **{self._san(t.name)}** (`{self._san(t.agent_id)}`){role}{model}{status}")
                if cwd:
                    lines.append(f"  {cwd}")
            lines.append("")

        if self.subagents:
            lines.append("## Subagents")
            for s in self.subagents:
                agent_type = f" [{self._san(s.subagent_type)}]" if s.subagent_type else ""
                desc = f" — {self._san(s.description)}" if s.description else ""
                lines.append(f"- `{self._san(s.agent_id)}`{agent_type}{desc} ({self._san(s.status)})")
                if s.result_summary:
                    lines.append(f"  Result: {self._san(s.result_summary)[:200]}")
            lines.append("")

        if self.tasks:
            active_tasks, completed_count, blank_count = self._task_groups()
            lines.append("## Active Task List")
            status_icons = {"completed": "x", "in_progress": "/", "pending": " "}
            if active_tasks:
                for t in active_tasks:
                    icon = status_icons.get(t.status, " ")
                    owner = f" @{self._san(t.owner)}" if t.owner else ""
                    lines.append(f"- [{icon}] {self._san(t.subject)}{owner}")
                    if t.description:
                        lines.append(f"  {self._san(t.description)[:200]}")
            else:
                lines.append("- No active tasks.")
            omitted = completed_count + blank_count
            if omitted:
                detail = []
                if completed_count:
                    detail.append(f"{completed_count} completed")
                if blank_count:
                    detail.append(f"{blank_count} blank")
                lines.append(f"_Omitted {', '.join(detail)} task(s) from recovery context._")
            lines.append("")

        if self.lead_summary:
            lines.append("## Lead Context")
            lines.append(self._san(self.lead_summary))
            lines.append("")

        total = self.message_count
        lines.append(f"_Extracted from {total} team-related messages_")
        return "\n".join(lines)

    def to_recovery_text(self) -> str:
        """Render team state as text for injection into conversation."""
        parts = []
        parts.append(f"Active agent team: {self._san(self.team_name) or 'unnamed'}")
        if self.lead_agent_id:
            parts.append(f"Lead: {self._san(self.lead_agent_id)} (session: {self._san(self.lead_session_id)})")

        if self.teammates:
            parts.append("\nTeammates:")
            for t in self.teammates:
                role = f" — {self._san(t.role)}" if t.role else ""
                model = f" [{self._san(t.model)}]" if t.model else ""
                parts.append(f"  - {self._san(t.name)} (agent_id: {self._san(t.agent_id)}){role}{model} [{self._san(t.status)}]")

        if self.subagents:
            parts.append(f"\nSubagents ({len(self.subagents)}):")
            for s in self.subagents:
                agent_type = f" [{self._san(s.subagent_type)}]" if s.subagent_type else ""
                desc = f" — {self._san(s.description)}" if s.description else ""
                parts.append(f"  - {self._san(s.agent_id)}{agent_type}{desc} [{self._san(s.status)}]")
                if s.result_summary:
                    parts.append(f"    Result: {self._san(s.result_summary)[:150]}")

        if self.tasks:
            active_tasks, completed_count, blank_count = self._task_groups()
            if active_tasks:
                parts.append("\nShared active tasks:")
                shown_tasks = active_tasks[:10]
                for t in shown_tasks:
                    owner = f" (owner: {self._san(t.owner)})" if t.owner else ""
                    parts.append(f"  - [{self._san(t.status).upper()}] {self._san(t.subject)}{owner}")
                if len(active_tasks) > len(shown_tasks):
                    parts.append(f"  - ... {len(active_tasks) - len(shown_tasks)} more active task(s) omitted")
            else:
                parts.append("\nShared task list: no active tasks.")
            omitted = completed_count + blank_count
            if omitted:
                detail = []
                if completed_count:
                    detail.append(f"{completed_count} completed")
                if blank_count:
                    detail.append(f"{blank_count} blank")
                parts.append(f"Completed/empty tasks omitted from recovery context: {', '.join(detail)}.")

        if self.lead_summary:
            parts.append(f"\nCoordination context: {self._san(self.lead_summary)}")

        return "\n".join(parts)


# Status values reach the receipt as dict KEYS, and they originate from raw,
# agent-controlled text (TaskUpdate tool input, <status>…</status> capture), so
# they must be coerced to a fixed vocabulary before they can be serialized —
# otherwise free-text (e.g. a secret accidentally written as a status) would
# leak into a "privacy-safe" artifact via the key. Anything off-vocabulary
# buckets to "other" so counts stay correct without copying raw text.
_KNOWN_STATUSES = frozenset({
    "running", "active", "in_progress", "pending", "queued", "blocked",
    "idle", "done", "completed", "failed", "cancelled", "stopped", "unknown",
})


def _count_by_status(items: list[object]) -> dict[str, int]:
    """Return stable status counts for receipt/debug output.

    Status strings are normalized to a fixed vocabulary (_KNOWN_STATUSES) so
    no raw, agent-controlled text reaches the receipt as a dict key.
    """
    counts: dict[str, int] = {}
    for item in items or []:
        raw = (getattr(item, "status", "unknown") or "unknown").strip().lower()
        status = raw if raw in _KNOWN_STATUSES else "other"
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def build_team_recovery_receipt(state: TeamState) -> dict:
    """Build a privacy-safe receipt for post-compact team recovery.

    The receipt is intentionally count/flag based: it proves what recovery
    state was available without copying team names, task subjects, prompts,
    cwd values, result summaries, or raw session text into a shareable artifact.
    """
    # The intended consumers (bug reports, guard logs) are exactly the contexts
    # where team state may be missing, so a None state must yield an
    # unsafe-to-resume receipt rather than crash. An empty TeamState routes
    # through the is_empty() path below to that verdict.
    if state is None:
        state = TeamState()
    active_tasks, completed_tasks, blank_tasks = state._task_groups()
    # `subagents`/`teammates` can be None on a half-built state (the bug-report /
    # guard-log contexts this receipt targets) — coerce to [] so we summarize
    # instead of crashing.
    subagents = state.subagents or []
    teammates = state.teammates or []
    running_subagents = [s for s in subagents if (s.status or "").lower() == "running"]
    active_teammates = [
        t for t in teammates
        if (t.status or "").strip().lower() not in _TEAMMATE_QUIESCENT
    ]

    gaps: list[str] = []
    if state.is_empty():
        gaps.append("no_team_state")
    if not state.config_source:
        gaps.append("missing_config_source")
    if state.last_coordination_index < 0:
        gaps.append("missing_last_coordination_cursor")
    if not state.tasks:
        gaps.append("no_task_assignment_table")

    # Cozempic can currently identify the last coordination line, but it does
    # not yet expose a per-teammate event/message cursor. Marking active teams
    # as partial until that exists prevents a phantom-team recovery from being
    # presented as complete.
    event_cursors_recorded = False
    has_active_work = bool(active_tasks or running_subagents or active_teammates)
    if has_active_work and not event_cursors_recorded:
        gaps.append("per_teammate_event_cursors_not_recorded")

    if state.is_empty():
        verdict = "unsafe-to-resume"
    elif has_active_work and not event_cursors_recorded:
        verdict = "partial"
    elif gaps:
        verdict = "partial"
    else:
        verdict = "complete"

    return {
        "event": "team.recovery.receipt.v1",
        "recovery_verdict": verdict,
        "source": {
            "config_source": state.config_source or "unknown",
            "team_identity_present": bool(state.team_name or state.lead_agent_id or state.lead_session_id),
            "last_coordination_cursor_present": state.last_coordination_index >= 0,
            "per_teammate_event_cursors_recorded": event_cursors_recorded,
        },
        "counts": {
            "team_messages": state.message_count,
            "teammates_total": len(teammates),
            "teammates_by_status": _count_by_status(teammates),
            "subagents_total": len(subagents),
            "subagents_by_status": _count_by_status(subagents),
            "tasks_active": len(active_tasks),
            "tasks_completed": completed_tasks,
            "tasks_blank": blank_tasks,
            "tasks_by_status": _count_by_status(state.tasks),
        },
        "privacy": {
            "raw_team_name_recorded": False,
            "raw_agent_ids_recorded": False,
            "raw_task_subjects_recorded": False,
            "raw_prompts_or_results_recorded": False,
            "raw_paths_recorded": False,
        },
        "audit_gaps": gaps,
    }


# ─── Patterns for team message detection ─────────────────────────────────────

# Tool names that indicate team/agent coordination.
# NOTE: "Agent" is intentionally NOT included here. Adding it would cause
# prune_with_team_protect (and _is_team_message) to protect ALL Agent tool_uses,
# including plain non-team subagent calls — over-protecting non-team sessions.
# extract_team_state uses _TEAM_EXTRACT_TOOL_NAMES (a superset) for its own
# pre-pass; prune_with_team_protect uses this set via _is_team_message.
TEAM_TOOL_NAMES = {
    # Explicit team coordination
    "TeamCreate", "TeamDelete", "TeamMessage", "SendMessage",
    "SpawnTeammate", "TeamStatus",
    # Shared task list (todo tracking)
    "TaskCreate", "TaskUpdate", "TaskList", "TaskGet",
    # Subagent spawning and results (Claude Code's Task tool)
    "Task", "TaskOutput", "TaskStop",
}

# Extended set for extract_team_state's pre-pass only — includes "Agent" so that
# Agent tool_use + tool_result pairs are scanned for teammate creation. Must NOT
# be used in _is_team_message (prune_with_team_protect path).
_TEAM_EXTRACT_TOOL_NAMES = TEAM_TOOL_NAMES | {"Agent"}


# Patterns for parsing task-notification XML in user messages. Fields are parsed
# INDEPENDENTLY (order-, attribute-, and extra-tag tolerant) rather than as one
# strictly-ordered regex: the REAL harness notification carries <tool-use-id> and
# <output-file> tags BETWEEN <task-id> and <status> (ground-truthed 2026-06-09), which
# a strict task-id→status regex misses → a COMPLETED background-Agent teammate is left
# "running" forever → safe_to_reload wedges the guard inert. Mirrors detect_in_flight's
# lenient _TN_*_RE so the two parsers agree on the same bytes.
# Maximum bytes of content fed into the DOTALL lazy-star block-regex scanner in
# extract_team_state's second pass.  Without a cap, many unmatched openers trigger
# O(openers × len) catastrophic backtracking — same class as recap.py's DoS guard.
# 64KB is ~64× a real notification; a missed notification → over-defers reload
# (recoverable, not under-blocks / SIGKILL).  Single source of truth in _constants;
# guard.py imports the same object — tune both scan sites by changing _constants only.
from ._constants import _RELOAD_GATE_SCAN_CAP

_TASK_NOTIF_BLOCK_RE = re.compile(
    r"<task-notification(?:\s[^>]*)?>(.*?)</task-notification>", re.DOTALL | re.IGNORECASE)
_TASK_NOTIF_ID_RE = re.compile(r"<task-id(?:\s[^>]*)?>([^<]+)</task-id>", re.IGNORECASE)
_TASK_NOTIF_STATUS_RE = re.compile(r"<status(?:\s[^>]*)?>([^<]+)</status>", re.IGNORECASE)
_TASK_NOTIF_SUMMARY_RE = re.compile(
    r"<summary(?:\s[^>]*)?>(.*?)</summary>", re.IGNORECASE | re.DOTALL)
_TASK_NOTIF_RESULT_RE = re.compile(
    r"<result(?:\s[^>]*)?>(.*?)</result>", re.IGNORECASE | re.DOTALL)

# Pattern for agent progress notifications in system-reminder tags
_AGENT_PROGRESS_RE = re.compile(
    r"Agent\s+([a-f0-9]+)\s+progress:.*?(\d+)\s+new\s+tool",
    re.IGNORECASE,
)

# Agent-spawn result format: "Spawned successfully.\nagent_id: NAME@TEAM\n..."
# Verified from production transcripts 2026-06-08 (transcript 371f5917, line 196).
# If the harness changes this format the regex will gracefully miss (fallback:
# placeholder key retained; only terminal transitions are affected).
# Accept snake `agent_id:` AND camelCase `agentId:` (the SHIPPED 1.8.22 background
# marker uses camelCase, so a team spawn very plausibly does too) + `=` separator.
_AGENT_SPAWN_ID_RE = re.compile(
    r"agent_?id\s*[:=]\s*([A-Za-z0-9@._-]+)",
    re.IGNORECASE,
)
_AGENT_SPAWN_TEAM_RE = re.compile(
    r"^team_name:\s*([A-Za-z0-9_-]+)",
    re.IGNORECASE | re.MULTILINE,
)
# Foreground-Agent COMPLETION trailer. When an `Agent` subagent RETURNS its result
# inline the harness appends "...to continue this agent\n<usage>subagent_tokens: N
# tool_uses: N\nduration_ms: N</usage>". GROUND-TRUTHED against a real session
# (27/27 foreground completions carried `duration_ms: N`, 2026-06-09). Its presence
# means the agent RAN AND FINISHED → the teammate is TERMINAL. A live team-spawn ack
# ("Spawned successfully\nagent_id: NAME@TEAM") and a background launch ("Async agent
# launched successfully. agentId: X") have NO duration_ms, so they stay non-terminal
# and keep blocking. Without this a completed FOREGROUND Agent reads "running"
# forever → safe_to_reload wedges the guard inert (real-transcript finding 2026-06-09).
_AGENT_DONE_TRAILER_RE = re.compile(r"\bduration_ms\b\s*[:=]\s*\d+", re.IGNORECASE)

# <teammate-message teammate_id="X">…</teammate-message> blocks in user messages.
# Used in the second pass to parse idle_notification transitions (P0-D).
#
# Body group uses a negative-lookahead to stop before any nested <teammate-message
# opening tag (M-1 fix, 2026-06-08). DOTALL alone was greedy enough to eat a nested
# block, mis-attributing the inner idle_notification to the OUTER teammate_id.
# The lookahead `(?!<teammate-message)` makes the body stop at the first inner tag.
_TEAMMATE_MSG_RE = re.compile(
    r'<teammate-message\s[^>]*teammate_id="([^"]+)"[^>]*>'
    r'((?:(?!<teammate-message).)*?)'
    r'</teammate-message>',
    re.DOTALL | re.IGNORECASE,
)
_IDLE_NOTIFICATION_RE = re.compile(
    r'"type"\s*:\s*"idle_notification"',
    re.IGNORECASE,
)


def _extract_block_text(block: dict) -> str:
    """Return the text payload of a tool-result block.

    Handles both string content and the list-of-sub-blocks form
    (``[{"type": "text", "text": "..."}]``).  Callers use this to avoid
    duplicating the same isinstance chain for result_text extraction.
    """
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # str-guard each sub["text"] (ynaamane review #2): a non-str text (e.g.
        # {"type":"text","text":99999} in malformed JSONL) would make "".join raise
        # TypeError, crashing extract_team_state — which is NOT strategy-isolated and
        # feeds the reactive overflow safe-gate (its throw used to fail-OPEN -> SIGKILL).
        return "".join(
            sub["text"]
            for sub in content
            if isinstance(sub, dict) and sub.get("type") == "text" and isinstance(sub.get("text"), str)
        )
    return ""


def _is_team_message(msg_dict: dict, pending_task_ids: set[str] | None = None) -> bool:
    """Check if a message is related to agent team coordination.

    Handles these JSONL message types:
    - type='assistant': Tool use calls (Task, TaskCreate, etc.)
    - type='user': Nested content with task-notification or teammate-message XML
    - type='queue-operation': Root-level content with task-notification XML
    - Tool results matching known Task tool_use IDs (via pending_task_ids)

    Detection is schema-first: tool_use block names and team XML patterns.
    TEAM_KEYWORDS is NOT used here — it is for enrichment (extract_team_state)
    only, to avoid false positives on messages that merely mention team concepts.
    """
    # Handle queue-operation messages (background task results).
    # These have content at the ROOT level, not under 'message'.
    if msg_dict.get("type") == "queue-operation":
        root_content = msg_dict.get("content", "")
        if isinstance(root_content, str) and "<task-notification>" in root_content:
            return True
        return False

    inner = msg_dict.get("message", {})
    content = inner.get("content", [])

    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type", "")

            # Tool use with team-related name — definitive signal.
            # isinstance(str) guards the `in <set>` membership: an unhashable
            # list/dict name (poisoned JSONL) would raise TypeError here (R5 finding).
            name = block.get("name")
            if block_type == "tool_use" and isinstance(name, str) and name in TEAM_TOOL_NAMES:
                return True

            # Tool result — match by tool_use_id if we know the pending Task IDs;
            # fall back to nothing (don't use TEAM_KEYWORDS — too broad).
            if block_type == "tool_result" and pending_task_ids:
                tool_use_id = block.get("tool_use_id", "")
                if isinstance(tool_use_id, str) and tool_use_id in pending_task_ids:
                    return True

    elif isinstance(content, str):
        # task-notification XML in user messages (agent results) — definitive signal.
        # teammate-message XML (idle_notification, etc.) — also team-coordination:
        # this carrier MUST be prune-protected so that idle transitions are not lost
        # (a pruned idle_notification leaves the teammate permanently "running" →
        # permanent safe_to_reload wedge — C-2 fix, 2026-06-08).
        if "<task-notification>" in content or "<teammate-message" in content:
            return True

    return False


def _is_task_tool_result(msg_dict: dict, pending_task_ids: set[str]) -> bool:
    """Check if a message contains a tool_result for a Task tool call.

    Task tool results carry the agent's output — these are critical to preserve.
    """
    inner = msg_dict.get("message", {})
    content = inner.get("content", [])

    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                if tool_use_id in pending_task_ids:
                    return True

    return False


def extract_team_state(messages: list[Message]) -> TeamState:
    """Scan messages for team coordination patterns and extract state.

    Looks for:
    - Task tool calls (subagent spawns with subagent_type, prompt, description)
    - TaskOutput calls (checking on background agents)
    - TeamCreate tool calls (team name, teammate configs)
    - SendMessage / TeamMessage tool calls
    - TaskCreate / TaskUpdate tool calls (shared todo list)
    - Teammate spawn details (agent IDs, roles)
    """
    state = TeamState()
    seen_teammates: dict[str, TeammateInfo] = {}
    seen_subagents: dict[str, SubagentInfo] = {}
    # Latest line index of a SendMessage to each teammate — used so a completion
    # notification in the (later) second pass does NOT clobber a re-activation
    # that chronologically followed it (the two-pass extractor would otherwise
    # discard event ordering and mark a still-working teammate "completed").
    last_send_line: dict[str, int] = {}
    seen_tasks: dict[str, TaskInfo] = {}
    # TaskCreate's tool INPUT carries no id (the system assigns it and returns it
    # in the tool_result: "Task #N created"). Map each TaskCreate tool_use_id to
    # the temp key its task is parked under, so the result handler can re-key it
    # to the REAL id. Without this, creates were keyed positionally (str(len(...)))
    # while TaskUpdate keys by the real id → completed tasks render as pending and
    # phantom blank tasks appear in the recovered checkpoint.
    taskcreate_uid_to_key: dict[str, str] = {}

    # Track tool_use_id -> tool_name for matching results to calls
    tool_use_id_to_name: dict[str, str] = {}
    # Track tool_use_id -> subagent key for Task tool results
    tool_use_id_to_subagent: dict[str, str] = {}
    # Track bare teammate name -> agentId for SendMessage bare-name resolution (P0-C)
    # and idle_notification resolution (P0-D). Populated when a TeammateInfo is
    # inserted into seen_teammates with a known bare name.
    _name_to_agent_id: dict[str, str] = {}

    # Pre-pass: collect all team tool_use IDs so _is_team_message can match
    # their corresponding tool_result messages (task completions, etc.).
    # Uses _TEAM_EXTRACT_TOOL_NAMES (includes "Agent") so Agent spawn results
    # are also scanned; _is_team_message itself still uses TEAM_TOOL_NAMES only.
    pending_task_ids: set[str] = set()
    for _, msg, _ in messages:
        inner = msg.get("message")
        if not isinstance(inner, dict):
            continue
        content = inner.get("content")
        for block in (content if isinstance(content, list) else []):
            if not isinstance(block, dict):  # a content array can hold a bare string/number
                continue
            # isinstance(str) guards both the set membership AND the set .add:
            # an unhashable list/dict name or id (poisoned JSONL) would raise
            # TypeError in this pre-pass, which runs OUTSIDE the main loop's R4
            # coercion and crashed extract_team_state -> guard respawn storm (R5).
            name = block.get("name")
            if block.get("type") == "tool_use" and isinstance(name, str) and name in _TEAM_EXTRACT_TOOL_NAMES:
                uid = block.get("id", "")
                if isinstance(uid, str) and uid:
                    pending_task_ids.add(uid)

    def _is_extract_message(m: dict) -> bool:
        """Like _is_team_message but uses _TEAM_EXTRACT_TOOL_NAMES (includes 'Agent').

        Used only inside extract_team_state so that Agent tool_use blocks are
        processed for teammate creation. prune_with_team_protect's call to
        _is_team_message is unaffected (uses TEAM_TOOL_NAMES only).
        """
        if m.get("type") == "queue-operation":
            rc = m.get("content", "")
            return isinstance(rc, str) and "<task-notification>" in rc
        inner_m = m.get("message", {})
        c = inner_m.get("content", [])
        if isinstance(c, list):
            for blk in c:
                if not isinstance(blk, dict):
                    continue
                bt = blk.get("type", "")
                name = blk.get("name")
                if bt == "tool_use" and isinstance(name, str) and name in _TEAM_EXTRACT_TOOL_NAMES:
                    return True
                if bt == "tool_result" and pending_task_ids:
                    tid = blk.get("tool_use_id", "")
                    if isinstance(tid, str) and tid in pending_task_ids:
                        return True
        elif isinstance(c, str):
            return "<task-notification>" in c
        return False

    for line_idx, msg, byte_size in messages:
        if not _is_extract_message(msg):
            continue

        state.message_count += 1
        state.last_coordination_index = line_idx

        inner = msg.get("message")
        if not isinstance(inner, dict):
            continue
        content = inner.get("content", [])
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type", "")

            # ── Tool use blocks ──────────────────────────────────────
            if block_type == "tool_use":
                # Coerce name/id to str: both are used as dict keys / in comparisons
                # below; an unhashable list/dict value (poisoned JSONL) would crash
                # `tool_use_id_to_name[tool_use_id]` (R4 team-input-field-crash).
                name = block.get("name", "")
                if not isinstance(name, str):
                    name = ""
                inp = block.get("input", {})
                if not isinstance(inp, dict):  # tool 'input' can be a non-dict in malformed JSONL
                    inp = {}
                tool_use_id = block.get("id", "")
                if not isinstance(tool_use_id, str):
                    tool_use_id = ""

                if tool_use_id and name:
                    tool_use_id_to_name[tool_use_id] = name

                # Task tool = subagent spawn
                if name == "Task":
                    description = _sfield(inp, "description")
                    subagent_type = _sfield(inp, "subagent_type")
                    prompt = _sfield(inp, "prompt")[:200]
                    resume_id = _sfield(inp, "resume")
                    bg = inp.get("run_in_background", False)

                    # Use tool_use_id as temporary key until we get agent_id
                    key = resume_id or tool_use_id or f"task-{len(seen_subagents)}"
                    agent = SubagentInfo(
                        agent_id=key,
                        description=description or prompt[:80],
                        subagent_type=subagent_type,
                        status="running" if bg else "running",
                    )
                    seen_subagents[key] = agent
                    if tool_use_id:
                        tool_use_id_to_subagent[tool_use_id] = key

                    # Infer team name from subagent_type if not set
                    if not state.team_name and subagent_type:
                        state.team_name = f"agents"

                # TaskOutput = checking on background agent
                elif name == "TaskOutput":
                    task_id = _sfield(inp, "task_id")
                    if task_id and task_id in seen_subagents:
                        # Still running, waiting for result
                        pass

                # TaskStop = stopping a background agent
                elif name == "TaskStop":
                    task_id = _sfield(inp, "task_id")
                    if task_id and task_id in seen_subagents:
                        seen_subagents[task_id].status = "stopped"

                # TeamCreate (explicit team)
                elif name == "TeamCreate":
                    # Real TeamCreate tool emits "team_name" key (verified from
                    # production transcripts, 2026-06-08); "name" is the legacy key
                    # kept as a fallback for backward compat. Prefer the real key
                    # so a transcript carrying BOTH (rollout overlap) uses the
                    # authoritative "team_name", not the stale legacy "name".
                    state.team_name = _sfield(inp, "team_name", "name") or state.team_name
                    _tms = inp.get("teammates", [])
                    for tm in (_tms if isinstance(_tms, list) else []):
                        if not isinstance(tm, dict):  # teammates array can hold a non-dict
                            continue
                        # Coerce to str: agent_id is a dict key (seen_teammates),
                        # tm_name/role are rendered — non-str/unhashable values crash.
                        agent_id = _sfield(tm, "agentId", "agent_id")
                        tm_name = _sfield(tm, "name") or agent_id
                        role = _sfield(tm, "role", "description")
                        if agent_id:
                            seen_teammates[agent_id] = TeammateInfo(
                                agent_id=agent_id,
                                name=tm_name,
                                role=role,
                                status="running",
                            )
                            # Populate name → agentId index for bare-name lookups
                            if tm_name and tm_name != agent_id:
                                _name_to_agent_id[tm_name] = agent_id

                # Agent tool = teammate spawn via the Agent tool (not Task tool).
                # The real agentId and team_name are in the tool_result text;
                # create a placeholder entry now so the spawn is immediately
                # visible; the result handler below upgrades to the real agentId.
                elif name == "Agent":
                    agent_name = _sfield(inp, "name")
                    agent_role = _sfield(inp, "role", "description")
                    # Placeholder key: use tool_use_id if available, else name.
                    # The result handler re-keys by the real agentId.
                    placeholder = tool_use_id or agent_name or f"agent-{len(seen_teammates)}"
                    if placeholder not in seen_teammates:
                        seen_teammates[placeholder] = TeammateInfo(
                            agent_id=placeholder,
                            name=agent_name,
                            role=agent_role,
                            status="running",  # non-benign, non-terminal → blocks reload
                        )
                    if tool_use_id:
                        tool_use_id_to_subagent[tool_use_id] = placeholder

                # TeamDelete — the lead disbanded a team; its members are done.
                # Without this a torn-down team's teammates stay "running" forever →
                # safe_to_reload wedges (never reloads). (1.8.24, fleet F1.)
                # Clear ONLY members we can POSITIVELY attribute to the deleted team
                # (their agentId carries its `@team` suffix). A member we can't
                # attribute is LEFT running: a wedge (inert guard) is recoverable,
                # but clearing a LIVE other team's members is an unrecoverable
                # SIGKILL — so on ambiguity we fail toward block, NEVER clear-all
                # (fleet P2/P2b, 2026-06-09). The real F1 case (Agent-spawned teams)
                # uses suffixed ids (alice@myteam), so a genuine disband still lifts
                # the wedge; only bare-id inline rosters fall back to the safe wedge.
                elif name == "TeamDelete":
                    _del_team = _sfield(inp, "team_name", "name").strip()
                    _suffix = "@" + _del_team
                    if _del_team:
                        for _tm in seen_teammates.values():
                            if (_tm.agent_id or "").endswith(_suffix):
                                _tm.status = "completed"

                # TaskCreate (shared todo list)
                elif name == "TaskCreate":
                    # The real task id is assigned by the system and appears ONLY
                    # in the tool_result ("Task #N created"), never in the input.
                    # Park the task under a temp key tied to this tool_use_id; the
                    # tool_result handler re-keys it to the real id so a later
                    # TaskUpdate(taskId=...) lands on the right task.
                    real_id = _sfield(inp, "taskId", "id")
                    if real_id:
                        task_key = real_id
                    elif tool_use_id:
                        task_key = f"__pending_create_{tool_use_id}"
                        taskcreate_uid_to_key[tool_use_id] = task_key
                    else:
                        task_key = str(len(seen_tasks))
                    seen_tasks[task_key] = TaskInfo(
                        task_id=real_id or task_key,
                        subject=_sfield(inp, "subject", "title"),
                        status="pending",
                        owner=_sfield(inp, "owner"),
                        description=_sfield(inp, "description"),
                        from_create=True,
                    )

                # TaskUpdate (shared todo list)
                elif name == "TaskUpdate":
                    task_id = _sfield(inp, "taskId", "id")
                    if task_id in seen_tasks:
                        if _sfield(inp, "status"):
                            seen_tasks[task_id].status = _sfield(inp, "status")
                        if _sfield(inp, "owner"):
                            seen_tasks[task_id].owner = _sfield(inp, "owner")
                        if _sfield(inp, "subject"):
                            seen_tasks[task_id].subject = _sfield(inp, "subject")
                    else:
                        seen_tasks[task_id] = TaskInfo(
                            task_id=task_id,
                            subject=_sfield(inp, "subject"),
                            status=_sfield(inp, "status", default="unknown"),
                            owner=_sfield(inp, "owner"),
                        )

                elif name in ("SendMessage", "TeamMessage"):
                    # _sfield → str: `to` can be a list (multi-recipient) or other
                    # non-hashable in poisoned JSONL; used as a dict key below, an
                    # unhashable value crashes (R4 team-input-field-crash).
                    target = _sfield(inp, "to", "agentId")
                    # Resolve bare name to agentId (P0-C): SendMessage carries a
                    # bare name ("alice") but seen_teammates is keyed by full
                    # agentId ("alice@myteam"). The _name_to_agent_id index bridges
                    # the gap; fall back to the literal target if not in index.
                    resolved = _name_to_agent_id.get(target, target)
                    if resolved in seen_teammates:
                        seen_teammates[resolved].status = "running"
                        last_send_line[resolved] = line_idx

            # ── Tool result blocks ───────────────────────────────────
            elif block_type == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                if not isinstance(tool_use_id, str):  # dict-key use below
                    tool_use_id = ""
                tool_name = tool_use_id_to_name.get(tool_use_id, "")

                # TaskCreate result carries the system-assigned id ("Task #N
                # created"); re-key the parked task from its temp key to the real
                # id so TaskUpdate completions (which carry the real taskId) land
                # on it. A reused id (task store reset) overwrites the older
                # generation, matching the live store which holds only the latest.
                if tool_name == "TaskCreate":
                    _tc_temp = taskcreate_uid_to_key.get(tool_use_id, "")
                    if _tc_temp and _tc_temp in seen_tasks:
                        _m_tc = re.search(r"Task\s+#?(\d+)\s+created",
                                          _extract_block_text(block), re.I)
                        if _m_tc:
                            # Consume the mapping ONLY once we've parsed a real id — a
                            # regex miss (reworded/torn result) leaves it intact so a
                            # later duplicate result can still recover the id.
                            taskcreate_uid_to_key.pop(tool_use_id, None)
                            _rid = _m_tc.group(1)
                            _info = seen_tasks.pop(_tc_temp)
                            _existing = seen_tasks.get(_rid)
                            if _existing is None or _existing.from_create:
                                # Free id, OR a real prior-generation task already holds
                                # it (a store reset reused #N) — latest generation wins:
                                # install the fresh create. `from_create` marks a genuine
                                # prior TaskCreate (not inferred from `subject`, which an
                                # out-of-order TaskUpdate can also carry).
                                _info.task_id = _rid
                                seen_tasks[_rid] = _info
                            else:
                                # The id is held by a placeholder synthesized from an
                                # out-of-order TaskUpdate (torn/reordered transcript).
                                # Keep its authoritative status/owner — never clobber
                                # e.g. "completed" with the create's "pending" — and
                                # backfill the create's subject/description so the row
                                # isn't blank. It's now backed by a real create, so mark
                                # it as such for any later same-id reuse.
                                _existing.task_id = _rid
                                _existing.from_create = True
                                _existing.subject = _info.subject
                                if not _existing.description:
                                    _existing.description = _info.description
                                if not _existing.owner:
                                    _existing.owner = _info.owner

                # Task tool result = subagent finished, capture result
                if tool_name == "Task" or tool_use_id in tool_use_id_to_subagent:
                    subagent_key = tool_use_id_to_subagent.get(tool_use_id, "")
                    result_text = _extract_block_text(block)

                    if subagent_key and subagent_key in seen_subagents:
                        seen_subagents[subagent_key].status = "completed"
                        seen_subagents[subagent_key].result_summary = result_text[:300]

                    # Check if result contains an agent_id we should track
                    agent_id_match = re.search(r"agent[_-]?id[:\s]+([a-f0-9-]+)", result_text, re.I)
                    if agent_id_match and subagent_key and subagent_key in seen_subagents:
                        real_id = agent_id_match.group(1)
                        agent = seen_subagents.pop(subagent_key)
                        agent.agent_id = real_id
                        seen_subagents[real_id] = agent

                # Agent tool result: parse real agentId + team_name from spawn text.
                # Format: "Spawned successfully.\nagent_id: NAME@TEAM\n..."
                # (verified from production transcripts 2026-06-08).
                if tool_name == "Agent":
                    result_text = _extract_block_text(block)
                    # A FOREGROUND Agent that RETURNED its result inline is DONE — the
                    # harness appends the duration_ms usage trailer. Such a teammate is
                    # TERMINAL; without this it reads "running" forever and wedges the
                    # gate inert (real-transcript finding 2026-06-09). A live team-spawn
                    # ack / background launch has no trailer → stays non-terminal.
                    _fg_done = bool(_AGENT_DONE_TRAILER_RE.search(result_text))

                    agent_id_m = _AGENT_SPAWN_ID_RE.search(result_text)
                    if agent_id_m:
                        real_agent_id = agent_id_m.group(1).strip()
                        # Re-key placeholder entry by real agentId
                        placeholder = tool_use_id_to_subagent.get(tool_use_id, "")
                        if placeholder and placeholder in seen_teammates:
                            tm = seen_teammates.pop(placeholder)
                            tm.agent_id = real_agent_id
                            if _fg_done:
                                tm.status = "completed"
                            seen_teammates[real_agent_id] = tm
                            # Populate name → agentId index for bare-name resolution.
                            # Index BOTH the teammate name AND the agentId's bare prefix
                            # (alice ← alice@myteam): a completion task-notification often
                            # carries the BARE id, and without the prefix index it never
                            # resolves → the teammate stays "running" → over-block (fleet
                            # P1, 2026-06-09; the direct-insert branch already does this).
                            if tm.name and tm.name != real_agent_id:
                                _name_to_agent_id[tm.name] = real_agent_id
                            _bare = real_agent_id.split("@")[0]
                            if _bare and _bare != real_agent_id:
                                _name_to_agent_id.setdefault(_bare, real_agent_id)
                        elif real_agent_id not in seen_teammates:
                            # No placeholder — insert directly (robustness)
                            bare = real_agent_id.split("@")[0] if "@" in real_agent_id else real_agent_id
                            seen_teammates[real_agent_id] = TeammateInfo(
                                agent_id=real_agent_id,
                                name=bare,
                                status="completed" if _fg_done else "running",
                            )
                            if bare and bare != real_agent_id:
                                _name_to_agent_id[bare] = real_agent_id
                    else:
                        # No agentId parsed. A FOREGROUND completion (duration_ms) is
                        # TERMINAL. Else mark "failed" ONLY if the result AFFIRMATIVELY
                        # signals failure (quota/error/rejected) — otherwise the spawn
                        # likely SUCCEEDED with an off-pattern string, so keep "running"
                        # (block). Fail toward block, never toward SIGKILL on a parse
                        # miss (1.8.24 fail-safe; the earlier H-1 marked every parse miss
                        # "failed" → false-safe).
                        placeholder = tool_use_id_to_subagent.get(tool_use_id, "")
                        if placeholder and placeholder in seen_teammates:
                            _low = result_text.lower()
                            if _fg_done:
                                seen_teammates[placeholder].status = "completed"
                            elif any(k in _low for k in (
                                    "error", "fail", "quota", "rejected", "denied",
                                    "could not", "unable", "cancel", "timeout")):
                                seen_teammates[placeholder].status = "failed"
                            # else: keep "running" — unparseable success defers (safe)
                    # Parse team_name from result if not yet set
                    team_m = _AGENT_SPAWN_TEAM_RE.search(result_text)
                    if team_m and not state.team_name:
                        state.team_name = team_m.group(1).strip()

    # ── Second pass: scan for task-notifications and idle-notifications ──
    # Both XML patterns live in string content (user messages or queue-operations),
    # but with different surface restrictions:
    #
    #   task-notification — QUEUE-OPERATION ROOT CONTENT ONLY.
    #     Reason: a user can type (or paste) any <task-notification> string.  If the
    #     second pass accepted user-typed message.content it would phantom-TERMINATE a
    #     live teammate, making safe_to_reload return True → SIGKILL (PR-8 C-2 fix).
    #     Fail-safe: a missed completion → teammate stays "running" → gate over-defers
    #     (recoverable), never under-blocks (SIGKILL).
    #
    #   idle_notification — STRING message.content OR queue-operation, but ONLY from
    #     genuine harness carriers (top-level teamName field required — H-1 fix).
    #     Reason: a user can type <teammate-message teammate_id="X">{"type":"idle_
    #     notification",...}</teammate-message> in plain content.  Without the teamName
    #     gate, the idle-notif scan would transition the teammate to "idle" → gate
    #     returns True → SIGKILL live work (phantom-IDLE, unrecoverable).
    #     The harness always sets teamName on genuine teammate-message carriers
    #     (confirmed: 220/220 genuine carriers have teamName; 0/220 user-typed do).
    #     Fail-safe: a genuine idle-notif on a message without teamName is MISSED →
    #     teammate stays "running" → gate over-defers (recoverable), never under-blocks.
    for line_idx, msg, byte_size in messages:
        if msg.get("type") == "queue-operation":
            _raw_content = msg.get("content", "")
            # B: guard against JSON null (content=None) — coerce to "" so the
            # finditer call below never sees a NoneType (TypeError).
            _task_notif_content = _raw_content if isinstance(_raw_content, str) else ""
            # D: queue-ops never carry teamName → H-1 gate will always skip the
            # idle-notif scan for them.  Set "" explicitly rather than aliasing
            # _task_notif_content to avoid a latent hazard if the gate is ever relaxed.
            _idle_notif_content = ""
        else:
            inner = msg.get("message", {})
            raw = inner.get("content", "")
            # task-notifications: queue-operation only (see comment above — C-2).
            _task_notif_content = ""
            # idle-notifications: string content, but only when teamName present (H-1).
            _idle_notif_content = raw if isinstance(raw, str) else ""

        # ── task-notifications ────────────────────────────────────────────
        # Parse each notification block, then its fields INDEPENDENTLY (order- and
        # extra-tag tolerant) so the REAL format (<tool-use-id>/<output-file> between
        # <task-id> and <status>) still clears a completed teammate/subagent.
        # Efficiency: _task_notif_content="" for every non-queue-op message (the
        # common case) — skip the regex entirely rather than matching against "".
        for _blk in (_TASK_NOTIF_BLOCK_RE.finditer(_task_notif_content[:_RELOAD_GATE_SCAN_CAP])
                     if _task_notif_content else ()):
            _body = _blk.group(1)
            _id_m = _TASK_NOTIF_ID_RE.search(_body)
            _st_m = _TASK_NOTIF_STATUS_RE.search(_body)
            if not _id_m or not _st_m:
                continue  # not a real task-notification (no id/status) — skip
            task_id = _id_m.group(1).strip()
            status = _st_m.group(1).strip()
            _sm_m = _TASK_NOTIF_SUMMARY_RE.search(_body)
            _rs_m = _TASK_NOTIF_RESULT_RE.search(_body)
            summary = _sm_m.group(1).strip() if _sm_m else ""
            result = _rs_m.group(1).strip() if _rs_m else ""

            # Find the matching subagent by agent_id
            if task_id in seen_subagents:
                seen_subagents[task_id].status = status
                seen_subagents[task_id].result_summary = result[:300]
                if summary and not seen_subagents[task_id].description:
                    seen_subagents[task_id].description = summary
            else:
                # Agent was spawned but we only have the notification
                seen_subagents[task_id] = SubagentInfo(
                    agent_id=task_id,
                    description=summary,
                    status=status,
                    result_summary=result[:300],
                )

            # Propagate terminal status to the matching TEAMMATE.
            # Resolve bare task_id → real agentId when the notification carries
            # only the bare name (e.g. "alice" instead of "alice@myteam").
            resolved_tid = _name_to_agent_id.get(task_id, task_id)
            for candidate in (task_id, resolved_tid):
                if candidate in seen_teammates and line_idx >= last_send_line.get(candidate, -1):
                    # Only mark terminal when this is the teammate's LATEST event.
                    # A SendMessage after this notification means re-activation
                    # (phase 2) → keep "running" so the gate keeps blocking.
                    seen_teammates[candidate].status = status
                    break

            # Count team-coordination activity once per genuine task-notification
            # block — NOT once per message. (Restored after #134 dedented this to the
            # outer per-message loop, which inflated message_count for every message →
            # bogus team_messages telemetry, config_source='jsonl' on teamless
            # sessions, and a team-change hash that churned every guard cycle.)
            state.message_count += 1

        # ── idle-notifications (P0-D) ────────────────────────────────────
        # <teammate-message teammate_id="X">{"type":"idle_notification",...}</teammate-message>
        # Transition status to "idle" UNLESS a later SendMessage re-activated it.
        # Fail-safe: a teammate-message beyond the cap is MISSED → teammate
        # stays "running" → safe_to_reload/agents_active keep it protected →
        # gate OVER-DEFERS (recoverable), never UNDER-BLOCKS (SIGKILL).
        #
        # H-1: require top-level teamName to authenticate the carrier.  Genuine
        # harness idle-notification messages always carry teamName; user-typed
        # messages never do.  A missing teamName → skip (over-defer, recoverable).
        #
        # H1-B RESIDUAL (DEFERRED): this is a presence-check, not a cryptographic
        # authenticator.  A user who knows about teamName could craft a message with
        # any teamName value and bypass this gate (e.g. user types a fake carrier with
        # teamName="cozempic-pipeline").  Closing H1-B would require a harness-stamped
        # sender field (like the C-3 residual which needs a structural nested_agent_id
        # marker) that user-typed text cannot forge.  Until then H-1 raises the bar
        # from zero knowledge (anyone can trigger it) to harness knowledge (only someone
        # who knows and sets teamName) — a meaningful improvement in a trusted-user
        # context.  Track: same follow-up PR as C-3.
        if not msg.get("teamName"):
            continue
        for tm_match in _TEAMMATE_MSG_RE.finditer(_idle_notif_content[:_RELOAD_GATE_SCAN_CAP]):
            tm_id = tm_match.group(1).strip()
            tm_body = tm_match.group(2)
            resolved = _name_to_agent_id.get(tm_id, tm_id)
            if resolved not in seen_teammates:
                continue
            if _IDLE_NOTIFICATION_RE.search(tm_body):
                # Chronology guard: SendMessage after this line re-activates
                # the teammate — don't clobber "running" back to "idle".
                if line_idx >= last_send_line.get(resolved, -1):
                    seen_teammates[resolved].status = "idle"
            else:
                # A NON-idle teammate-message means the teammate just SPOKE — it is
                # alive and working. Re-activate it from a benign (idle/unknown)
                # status back to "running" so a teammate that idled then RESUMED is
                # not SIGKILLed (fleet F: idle→re-engage false-negative, 2026-06-09).
                # Messages are processed in line order, so a later terminal
                # task-notification still wins; and we never clobber an already
                # TERMINAL status here, so a finished teammate cannot be wedged.
                cur = (seen_teammates[resolved].status or "").strip().lower()
                if cur in ("idle", "unknown", ""):
                    seen_teammates[resolved].status = "running"

    state.teammates = list(seen_teammates.values())
    state.subagents = list(seen_subagents.values())
    # A TaskCreate whose result was missing/torn/reworded never got re-keyed, so
    # its task_id still holds the internal "__pending_create_<uid>" sentinel.
    # Rewrite those to a clean positional id (matching the pre-#167 convention) so
    # the sentinel never appears in internal state. Markdown renders by
    # subject+status, so this is display-invisible; the rewrite keeps task_id sane
    # for any future consumer. Skip ids already taken by a real re-keyed task so we
    # never mint a duplicate task_id.
    _used_ids = {_t.task_id for _t in seen_tasks.values()
                 if not _t.task_id.startswith("__pending_create_")}
    _next = 0
    for _t in seen_tasks.values():
        if _t.task_id.startswith("__pending_create_"):
            while str(_next) in _used_ids:
                _next += 1
            _t.task_id = str(_next)
            _used_ids.add(str(_next))
            _next += 1
    state.tasks = list(seen_tasks.values())
    state.config_source = "jsonl" if state.message_count > 0 else ""

    # Build lead summary from last few team-related assistant messages
    team_msgs: list[str] = []
    for line_idx, msg, byte_size in messages:
        if msg.get("type") == "assistant" and _is_team_message(msg):
            inner = msg.get("message", {})
            content = inner.get("content", []) if isinstance(inner, dict) else []
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        _t = block.get("text", "")
                        if isinstance(_t, str):  # non-str text would crash the [:300] slice (#2)
                            team_msgs.append(_t[:300])

    if team_msgs:
        state.lead_summary = " [...] ".join(team_msgs[-3:])

    # Merge with config.json ground truth (if available)
    state = merge_config_into_state(state)

    return state


# ─── Config.json ground truth ─────────────────────────────────────────────

def load_team_configs() -> list[dict]:
    """Scan ~/.claude/teams/*/config.json for authoritative team configs.

    Claude Code stores team configuration in ~/.claude/teams/<team-name>/config.json.
    This is the ground truth for: team name, lead agent, session ID, members,
    models, working directories.

    Returns a list of parsed config dicts, one per team.
    """
    from .session import get_claude_dir
    teams_dir = get_claude_dir() / "teams"
    configs = []
    if not teams_dir.is_dir():
        return configs

    for config_file in teams_dir.glob("*/config.json"):
        try:
            data = json.loads(config_file.read_text(encoding="utf-8"))
            data["_config_path"] = str(config_file)
            configs.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    return configs


def merge_config_into_state(state: TeamState, configs: list[dict] | None = None) -> TeamState:
    """Merge config.json data into JSONL-extracted team state.

    Config.json is authoritative for:
      team name, lead agent ID, lead session ID, member details (model, cwd, agentType)

    JSONL is authoritative for:
      runtime state (subagent status, task progress, results)

    If configs is None, loads from ~/.claude/teams/ automatically.
    """
    if configs is None:
        configs = load_team_configs()

    if not configs:
        if not state.config_source:
            state.config_source = "jsonl"
        return state

    # Match configs in two phases: strong joins first (session-identity-anchored),
    # then name-only as a weaker fallback.
    #
    # Why two phases (C-1 fix, 2026-06-08):
    #   A name-only match is unreliable for session-identity fields: team names are
    #   frequently reused across sessions (same project, same team composition). A
    #   stale config.json with the same name but an OLD leadSessionId would overwrite
    #   state.lead_session_id; because safe_to_reload's teammate block fires on
    #   status alone (not session ID) this would not cause a missed block, but it
    #   pollutes the state with a stale identity field (F1 reborn via config path).
    #
    #   Strong joins (session ID / agent ID / member ID intersection) are anchored
    #   on identity that is guaranteed unique per session. Only a strong match
    #   authorises overwriting lead_session_id; a name-only match may carry member
    #   details (model, cwd) but must NOT overwrite session-identity fields.
    matched_config = None
    _name_only_match = False  # True when matched by team name alone (weak join)

    # Phase 1: strong joins — leadSessionId > leadAgentId > member ID intersection.
    # Empty ids are dropped: a malformed spawn (agent_id="") and a config member
    # with no "agentId" both yield "", and {""} ∩ {""} would be a false strong
    # match against an unrelated team's config.
    known_agent_ids = {
        aid for aid in (
            {s.agent_id for s in state.subagents}
            | {t.agent_id for t in state.teammates}
        ) if aid
    }
    for cfg in configs:
        if state.lead_session_id and cfg.get("leadSessionId") == state.lead_session_id:
            matched_config = cfg
            break
        if state.lead_agent_id and cfg.get("leadAgentId") == state.lead_agent_id:
            matched_config = cfg
            break
        if known_agent_ids:
            cfg_member_ids = {m.get("agentId", "") for m in cfg.get("members", []) if m.get("agentId")}
            if known_agent_ids & cfg_member_ids:
                matched_config = cfg
                break

    if matched_config is None:
        # Phase 2: name-only fallback — weaker; must not overwrite session-identity fields
        for cfg in configs:
            if state.team_name and cfg.get("name") == state.team_name:
                matched_config = cfg
                _name_only_match = True
                break

    if matched_config is None:
        # No match on any join — skip merge
        if not state.config_source:
            state.config_source = "jsonl"
        return state

    # Merge authoritative fields.
    # lead_session_id is session-identity — only overwrite from a strong join.
    # A name-only match may carry a STALE leadSessionId from a prior session that
    # happened to use the same team name; importing it would pollute session-identity
    # fields in TeamState with stale data. C-1 fix (2026-06-08).
    state.team_name = matched_config.get("name", state.team_name)
    state.lead_agent_id = matched_config.get("leadAgentId", state.lead_agent_id)
    if not _name_only_match:
        state.lead_session_id = matched_config.get("leadSessionId", state.lead_session_id)
    state.config_source = "both" if state.message_count > 0 else "config.json"

    # Merge member details
    existing_teammates = {t.agent_id: t for t in state.teammates}
    for member in matched_config.get("members", []):
        agent_id = member.get("agentId", "")
        if not agent_id:
            continue

        if agent_id in existing_teammates:
            # Enrich existing teammate with config data
            t = existing_teammates[agent_id]
            t.model = member.get("model", t.model)
            t.cwd = member.get("cwd", t.cwd)
            if not t.role:
                t.role = member.get("agentType", "")
        else:
            # Add from config (not seen in JSONL)
            state.teammates.append(TeammateInfo(
                agent_id=agent_id,
                name=member.get("name", agent_id),
                role=member.get("agentType", ""),
                model=member.get("model", ""),
                cwd=member.get("cwd", ""),
                status="config",
            ))

    return state


def write_team_checkpoint(state: TeamState, project_dir: Path | None = None) -> Path:
    """Write team state checkpoint to disk.

    Writes to .claude/team-checkpoint.md in the project directory,
    or to ~/.claude/team-checkpoint.md as fallback.
    """
    if project_dir and project_dir.exists():
        path = project_dir / "team-checkpoint.md"
    else:
        from .session import get_claude_dir
        path = get_claude_dir() / "team-checkpoint.md"

    # atomic_write_text (ynaamane review #5): a SIGKILL/OOM mid-write would otherwise
    # leave a PARTIAL checkpoint that PostCompact reads back as recovery state. Every
    # other shared-state writer is atomic (temp + os.replace); this one was the holdout.
    # Strict UTF-8 is safe now (R15): _san replaces lone surrogates with U+FFFD at the
    # render chokepoint, so to_markdown() is clean UTF-8 — no UnicodeEncodeError on
    # write and no WTF-8 for the reader (cozempic OR Claude Code) to choke on. (The R14
    # errors="surrogatepass" write was reverted because it only RELOCATED the crash to
    # the strict read in read_team_checkpoint / the PostCompact hook.)
    from .helpers import atomic_write_text
    atomic_write_text(path, state.to_markdown())
    return path


def read_team_checkpoint(
    project_dir: Path | None = None,
    include_global: bool = True,
) -> str | None:
    """Read saved team checkpoint from disk.

    Returns the checkpoint content, or None if not found or empty.
    Used by PostCompact hook to re-inject team state after compaction.
    The checkpoint is written by PreCompact (before compaction), so reading
    from disk is safer than re-scanning the compacted JSONL.

    Args:
        project_dir: The resolved project directory (contains team-checkpoint.md).
        include_global: When True (default), falls back to the shared
            ~/.claude/team-checkpoint.md if the project-local file is absent.
            Pass False from cmd_post_compact to prevent cross-project reads:
            with a correctly resolved project_dir the global file is redundant,
            and if resolution fails we prefer silence over injecting another
            project's state.
    """
    from .session import get_claude_dir

    candidates = []
    if project_dir and project_dir.exists():
        candidates.append(project_dir / "team-checkpoint.md")
    if include_global:
        candidates.append(get_claude_dir() / "team-checkpoint.md")

    for path in candidates:
        if path.exists():
            try:
                # errors="surrogatepass": tolerate a STALE checkpoint written by the
                # short-lived R14 surrogatepass code (WTF-8 bytes on disk) so the
                # PostCompact hook degrades gracefully instead of crashing on a strict
                # UnicodeDecodeError (R15). New writes are clean UTF-8 (_san strips
                # surrogates), so this only matters for a pre-existing file.
                content = path.read_text(encoding="utf-8", errors="surrogatepass").strip()
            except (OSError, ValueError):
                # Unreadable / undecodable checkpoint — degrade to None rather than
                # crash the automatic PostCompact hook (which has no try/except).
                continue
            if content:
                return content
    return None


def inject_team_recovery(messages: list[Message], state: TeamState) -> list[Message]:
    """Inject team state as a synthetic message pair at the end of the session.

    Appends:
    1. A 'user' message asking about team state
    2. An 'assistant' message confirming the full team state

    This ensures that when Claude resumes from the pruned JSONL,
    it 'remembers' the team — not as a suggestion but as actual
    conversation history.
    """
    if state.is_empty():
        return messages

    # Find the last message to chain UUIDs
    last_uuid = None
    last_session_id = None
    last_cwd = None
    last_git_branch = None

    for _, msg, _ in reversed(messages):
        if msg.get("uuid"):
            last_uuid = msg["uuid"]
            last_session_id = msg.get("sessionId")
            last_cwd = msg.get("cwd")
            last_git_branch = msg.get("gitBranch")
            break

    if not last_uuid:
        return messages  # Can't chain without a UUID

    now = datetime.now().isoformat()
    user_uuid = str(uuid_mod.uuid4())
    assistant_uuid = str(uuid_mod.uuid4())

    active_tasks, completed_tasks, blank_tasks = state._task_groups()
    has_actionable_context = bool(
        state.teammates
        or state.subagents
        or active_tasks
        or state.lead_agent_id
        or (state.team_name and state.team_name != "unnamed")
    )
    if not has_actionable_context:
        return messages

    recovery_text = state.to_recovery_text()
    for _, msg, _ in reversed(messages[-80:]):
        inner = msg.get("message", {})
        content = inner.get("content", "")
        if (
            isinstance(content, str)
            and "[Cozempic Guard: context was pruned." in content
            and recovery_text in content
        ):
            return messages

    checkpoint_note = (
        "A team state checkpoint was also written to .claude/team-checkpoint.md."
    )

    # Terse confirmation summary — avoid echoing the full team state back.
    summary_bits = []
    if state.team_name:
        summary_bits.append(f"team={TeamState._san(state.team_name)}")
    if state.teammates:
        summary_bits.append(f"{len(state.teammates)} teammate(s)")
    if state.subagents:
        summary_bits.append(f"{len(state.subagents)} subagent(s)")
    if state.tasks:
        pending = sum(1 for t in active_tasks if t.status.lower() == "pending")
        in_progress = sum(1 for t in active_tasks if t.status.lower() == "in_progress")
        task_bit = f"{len(active_tasks)} active task(s)"
        omitted = completed_tasks + blank_tasks
        if pending or in_progress:
            task_bit += f" ({pending} pending, {in_progress} in progress)"
        if omitted:
            task_bit += f", {omitted} omitted"
        summary_bits.append(task_bit)
    summary = ", ".join(summary_bits) if summary_bits else "state restored"

    # User message: trigger for team state recovery
    user_msg = {
        "type": "user",
        "uuid": user_uuid,
        "parentUuid": last_uuid,
        "sessionId": last_session_id,
        "timestamp": now,
        "cwd": last_cwd,
        "gitBranch": last_git_branch,
        "isSidechain": False,
        "userType": "external",
        "message": {
            "role": "user",
            "content": (
                "[Cozempic Guard: context was pruned. Team state restored below "
                "for your reference — do not echo it back, just acknowledge briefly "
                "and continue.]\n\n"
                f"{recovery_text}"
            ),
        },
    }

    # Assistant message: confirms team state
    assistant_msg = {
        "type": "assistant",
        "uuid": assistant_uuid,
        "parentUuid": user_uuid,
        "sessionId": last_session_id,
        "timestamp": now,
        "cwd": last_cwd,
        "gitBranch": last_git_branch,
        "isSidechain": False,
        "userType": "external",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Team state restored ({summary}). {checkpoint_note} "
                        "Continuing."
                    ),
                }
            ],
        },
    }

    user_line = json.dumps(user_msg, separators=(",", ":"))
    assistant_line = json.dumps(assistant_msg, separators=(",", ":"))

    # Append as new messages at the end
    next_idx = max(idx for idx, _, _ in messages) + 1 if messages else 0
    messages = list(messages)  # copy
    messages.append((next_idx, user_msg, len(user_line.encode("utf-8"))))
    messages.append((next_idx + 1, assistant_msg, len(assistant_line.encode("utf-8"))))

    return messages
