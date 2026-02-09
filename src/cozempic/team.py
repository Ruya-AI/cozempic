"""Agent team state extraction, checkpointing, and recovery injection.

Scans JSONL session files for agent team coordination patterns
(TeamCreate, SendMessage, TaskCreate, TaskUpdate, teammate references)
and can inject team state back into a pruned session so that
Claude resumes with full team awareness.
"""

from __future__ import annotations

import json
import re
import uuid as uuid_mod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .types import Message


@dataclass
class TeammateInfo:
    """Information about a single teammate."""

    agent_id: str
    name: str
    role: str = ""
    status: str = "unknown"  # running, done, idle


@dataclass
class TaskInfo:
    """Information about a task in the shared task list."""

    task_id: str
    subject: str
    status: str = "pending"
    owner: str = ""
    description: str = ""


@dataclass
class TeamState:
    """Extracted state of an agent team from conversation history."""

    team_name: str = ""
    teammates: list[TeammateInfo] = field(default_factory=list)
    tasks: list[TaskInfo] = field(default_factory=list)
    lead_summary: str = ""
    message_count: int = 0
    last_coordination_index: int = -1

    def is_empty(self) -> bool:
        return not self.team_name and not self.teammates and not self.tasks

    def to_markdown(self) -> str:
        """Render team state as markdown for checkpoint file."""
        lines = []
        lines.append(f"# Agent Team Checkpoint: {self.team_name or 'unnamed'}")
        lines.append(f"_Generated: {datetime.now().isoformat()}_")
        lines.append("")

        if self.teammates:
            lines.append("## Teammates")
            for t in self.teammates:
                status = f" ({t.status})" if t.status != "unknown" else ""
                role = f" — {t.role}" if t.role else ""
                lines.append(f"- **{t.name}** (`{t.agent_id}`){role}{status}")
            lines.append("")

        if self.tasks:
            lines.append("## Task List")
            status_icons = {"completed": "x", "in_progress": "/", "pending": " "}
            for t in self.tasks:
                icon = status_icons.get(t.status, " ")
                owner = f" @{t.owner}" if t.owner else ""
                lines.append(f"- [{icon}] {t.subject}{owner}")
                if t.description:
                    lines.append(f"  {t.description[:200]}")
            lines.append("")

        if self.lead_summary:
            lines.append("## Lead Context")
            lines.append(self.lead_summary)
            lines.append("")

        lines.append(f"_Extracted from {self.message_count} team-related messages_")
        return "\n".join(lines)

    def to_recovery_text(self) -> str:
        """Render team state as text for injection into conversation."""
        parts = []
        parts.append(f"Active agent team: {self.team_name or 'unnamed'}")

        if self.teammates:
            parts.append("\nTeammates:")
            for t in self.teammates:
                role = f" — {t.role}" if t.role else ""
                parts.append(f"  - {t.name} (agent_id: {t.agent_id}){role} [{t.status}]")

        if self.tasks:
            parts.append("\nShared task list:")
            for t in self.tasks:
                owner = f" (owner: {t.owner})" if t.owner else ""
                parts.append(f"  - [{t.status.upper()}] {t.subject}{owner}")

        if self.lead_summary:
            parts.append(f"\nCoordination context: {self.lead_summary}")

        return "\n".join(parts)


# ─── Patterns for team message detection ─────────────────────────────────────

TEAM_TOOL_NAMES = {
    "TeamCreate", "TeamDelete", "TeamMessage", "SendMessage",
    "TaskCreate", "TaskUpdate", "TaskList", "TaskGet",
    "SpawnTeammate", "TeamStatus",
}

TEAM_KEYWORDS = re.compile(
    r"team.?name|agent.?id|teammate|team.?lead|"
    r"SendMessage|TeamCreate|TaskCreate|TaskUpdate|"
    r"agent.?team|spawn.+teammate|team.+config",
    re.IGNORECASE,
)


def _is_team_message(msg_dict: dict) -> bool:
    """Check if a message is related to agent team coordination."""
    # Check tool_use blocks
    inner = msg_dict.get("message", {})
    content = inner.get("content", [])

    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            # Tool use with team-related name
            if block.get("type") == "tool_use" and block.get("name") in TEAM_TOOL_NAMES:
                return True
            # Tool result containing team data
            if block.get("type") == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, str) and TEAM_KEYWORDS.search(result_content):
                    return True
            # Text mentioning team coordination
            text = block.get("text", "")
            if isinstance(text, str) and TEAM_KEYWORDS.search(text):
                return True

    elif isinstance(content, str) and TEAM_KEYWORDS.search(content):
        return True

    return False


def extract_team_state(messages: list[Message]) -> TeamState:
    """Scan messages for team coordination patterns and extract state.

    Looks for:
    - TeamCreate tool calls (team name, teammate configs)
    - SendMessage / TeamMessage tool calls
    - TaskCreate / TaskUpdate tool calls
    - Teammate spawn details (agent IDs, roles)
    - Task list state
    """
    state = TeamState()
    seen_teammates = {}  # agent_id -> TeammateInfo
    seen_tasks = {}  # task_id -> TaskInfo

    for line_idx, msg, byte_size in messages:
        if not _is_team_message(msg):
            continue

        state.message_count += 1
        state.last_coordination_index = line_idx

        inner = msg.get("message", {})
        content = inner.get("content", [])
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue

            if block.get("type") != "tool_use":
                continue

            name = block.get("name", "")
            inp = block.get("input", {})

            if name == "TeamCreate":
                state.team_name = inp.get("name", state.team_name)
                for tm in inp.get("teammates", []):
                    agent_id = tm.get("agentId", tm.get("agent_id", ""))
                    tm_name = tm.get("name", agent_id)
                    role = tm.get("role", tm.get("description", ""))
                    if agent_id:
                        seen_teammates[agent_id] = TeammateInfo(
                            agent_id=agent_id,
                            name=tm_name,
                            role=role,
                            status="running",
                        )

            elif name in ("TaskCreate",):
                task_id = inp.get("taskId", inp.get("id", str(len(seen_tasks))))
                subject = inp.get("subject", inp.get("title", ""))
                seen_tasks[task_id] = TaskInfo(
                    task_id=task_id,
                    subject=subject,
                    status="pending",
                    owner=inp.get("owner", ""),
                    description=inp.get("description", ""),
                )

            elif name in ("TaskUpdate",):
                task_id = inp.get("taskId", inp.get("id", ""))
                if task_id in seen_tasks:
                    if inp.get("status"):
                        seen_tasks[task_id].status = inp["status"]
                    if inp.get("owner"):
                        seen_tasks[task_id].owner = inp["owner"]
                else:
                    # Task created before our scan window
                    seen_tasks[task_id] = TaskInfo(
                        task_id=task_id,
                        subject=inp.get("subject", ""),
                        status=inp.get("status", "unknown"),
                        owner=inp.get("owner", ""),
                    )

            elif name in ("SendMessage", "TeamMessage"):
                # Track which teammates are active
                target = inp.get("to", inp.get("agentId", ""))
                if target and target in seen_teammates:
                    seen_teammates[target].status = "running"

    state.teammates = list(seen_teammates.values())
    state.tasks = list(seen_tasks.values())

    # Build lead summary from last few team-related assistant messages
    team_msgs = []
    for line_idx, msg, byte_size in messages:
        if msg.get("type") == "assistant" and _is_team_message(msg):
            inner = msg.get("message", {})
            content = inner.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        team_msgs.append(block.get("text", "")[:300])

    if team_msgs:
        # Keep last 3 coordination messages as context
        state.lead_summary = " [...] ".join(team_msgs[-3:])

    return state


def write_team_checkpoint(state: TeamState, project_dir: Path | None = None) -> Path:
    """Write team state checkpoint to disk.

    Writes to .claude/team-checkpoint.md in the project directory,
    or to ~/.claude/team-checkpoint.md as fallback.
    """
    if project_dir and project_dir.exists():
        path = project_dir / "team-checkpoint.md"
    else:
        path = Path.home() / ".claude" / "team-checkpoint.md"

    path.write_text(state.to_markdown())
    return path


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

    recovery_text = state.to_recovery_text()
    checkpoint_note = (
        "A team state checkpoint was also written to .claude/team-checkpoint.md."
    )

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
                "[Cozempic Guard: Context was pruned to prevent compaction. "
                "Confirm the current agent team state below.]\n\n"
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
                        f"Confirmed — I have an active agent team. {recovery_text}\n\n"
                        f"{checkpoint_note}\n\n"
                        "Continuing with team coordination."
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
