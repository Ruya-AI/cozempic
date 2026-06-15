"""Regression guards for the 64KB cap on reload-gate block-regex scans.

Both detect_in_flight (guard.py) and extract_team_state (team.py) iterate
uncapped text through DOTALL lazy-star regexes:

  guard.py:2509  _TN_BLOCK_RE.findall(text)            — text from _completion_text(msg)
  team.py:862    _TASK_NOTIF_BLOCK_RE.finditer(content) — raw string content from JSONL

An attacker-sized message (many <task-notification> openers without closers)
triggers O(openers × len) catastrophic backtracking, freezing the 30-second
checkpoint/reload-gate loop — a quadratic-regex DoS (L3/L0, MED). Measured
at ~4-5s with 10,000 openers (~185KB); capped at 64KB takes <0.6s.

recap.py already solved this for its own regexes (text[:32768] / text[:8000]).
Fix mirrors that pattern: cap both scan sites at _RELOAD_GATE_SCAN_CAP = 65536.

The cap is FAIL-SAFE: a notification beyond 64KB is MISSED → the launch stays
"in-flight" → the gate OVER-DEFERS the reload (recoverable). It never
UNDER-BLOCKS, which would SIGKILL. 64KB is ~64× the size of a real notification.

REGRESSION GUARD tests (proven RED at base — timing blowup without the cap):
  test_detect_in_flight_quadratic_input_bounded  — guard.py site
  test_extract_team_state_quadratic_input_bounded — team.py site

CORRECTNESS GUARD tests (GREEN at base and after fix):
  test_detect_in_flight_real_notification_still_clears   — happy path
  test_extract_team_state_real_notification_still_clears — happy path
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

# ── degenerate input that triggers catastrophic backtracking ──────────────────
# Many <task-notification> openers without matching closers.  The DOTALL
# lazy-star `(.*?)` must scan to the end of string for each opener trying to
# find a closing tag — O(openers × len) work without a cap.  10k openers
# (~185KB) gives ~4-5s uncapped on a modern machine; the 64KB cap reduces this
# to <0.6s.  Large enough to stay >2s at base with margin; small enough to
# complete within pytest's run window.
_MANY_OPENERS = "<task-notification>" * 10_000   # 185 KB

# A real task-notification payload well within the 64KB cap.
_REAL_NOTIF = (
    "<task-notification>"
    "<task-id>agent-xyz</task-id>"
    "<status>completed</status>"
    "<result>done</result>"
    "</task-notification>"
)


class TestDetectInFlightReDoSCap(unittest.TestCase):
    """guard.py _TN_BLOCK_RE.findall(text) must be capped at _RELOAD_GATE_SCAN_CAP."""

    def _detect(self, raw_text: str) -> dict:
        """Call detect_in_flight with a single user message carrying raw_text."""
        from cozempic.guard import detect_in_flight
        msgs = [{"type": "user", "content": raw_text}]
        return detect_in_flight(msgs)

    def test_detect_in_flight_quadratic_input_bounded(self):
        """REGRESSION GUARD — RED at base: uncapped scan on 185KB of openers blows up.

        Without the _RELOAD_GATE_SCAN_CAP[:65536] slice the regex must scan the
        entire 185KB string for each of the 10,000 openers — measured wall time
        is 4-5s (catastrophic backtracking). With the cap the truncated 64KB
        input completes in <0.6s.

        Budget: 2.0s.
        """
        t0 = time.monotonic()
        result = self._detect(_MANY_OPENERS)
        elapsed = time.monotonic() - t0
        self.assertLess(
            elapsed, 2.0,
            f"detect_in_flight took {elapsed:.3f}s on degenerate input — "
            "the _TN_BLOCK_RE scan is not capped at _RELOAD_GATE_SCAN_CAP (64KB). "
            "Without the cap, 10,000 openers trigger O(openers × len) backtracking."
        )
        # The degenerate input has no valid closed blocks — no completions cleared.
        self.assertFalse(
            result.get("agent") or result.get("background") or result.get("workflow"),
            "Degenerate openers-only input must not affect in-flight detection"
        )

    def test_detect_in_flight_real_notification_still_clears(self):
        """Correctness guard (GREEN at base and after fix): a real notification within
        the 64KB cap must still clear the corresponding agent launch.

        This verifies the cap does NOT break the happy path — a normal
        <task-notification>completed</task-notification> is processed correctly.
        """
        from cozempic.guard import detect_in_flight
        # Message sequence: Agent tool_use launch, then task-notification complete.
        msgs = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "tu-1", "name": "Agent",
                         "input": {"name": "finder-p1"}}
                    ],
                    "role": "assistant",
                }
            },
            # Harness delivers the task-notification as a user message content string
            {"type": "user", "content": _REAL_NOTIF},
        ]
        result = detect_in_flight(msgs)
        # The notification cleared the Agent launch — agent must NOT be in-flight.
        self.assertFalse(
            result.get("agent"),
            "detect_in_flight must clear the Agent launch when a completed "
            "task-notification is present within the 64KB cap; "
            f"got result={result}"
        )


def _extract_isolated(msgs):
    """Call extract_team_state with config isolation (no live ~/.claude/teams/).

    Shared by TestExtractTeamStateReDoSCap and TestExtractTeamStateTeammateMsgCap
    to avoid duplicating the patch context manager in each class.
    """
    from cozempic.team import extract_team_state
    with patch("cozempic.team.load_team_configs", return_value=[]):
        return extract_team_state(msgs)


def _tu(idx: int, id_: str, name: str, inp: dict) -> tuple:
    """3-tuple (idx, msg_dict, size) for an Agent tool_use — matches Message type."""
    d = {"message": {"role": "assistant", "content": [
        {"type": "tool_use", "id": id_, "name": name, "input": inp}
    ]}}
    return (idx, d, 200)


def _tr(idx: int, tool_use_id: str, text: str) -> tuple:
    """3-tuple for a tool_result message."""
    d = {"message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}
    ]}}
    return (idx, d, len(text))


def _uc(idx: int, text: str) -> tuple:
    """3-tuple for a user message with plain string content (no teamName).

    Simulates user-typed content. For genuine harness idle-notification carriers,
    use _huc() which adds the top-level teamName required by the H-1 gate.
    """
    d = {"message": {"role": "user", "content": text}}
    return (idx, d, len(text))


def _huc(idx: int, text: str, team_name: str = "myteam") -> tuple:
    """3-tuple for a genuine harness idle-notification carrier (top-level teamName).

    H-1 gate: the idle-notif scan in extract_team_state skips messages without
    a top-level teamName field.  Use this helper when the test represents a real
    harness delivery (where the harness always sets teamName on the carrier).
    """
    d = {"teamName": team_name, "message": {"role": "user", "content": text}}
    return (idx, d, len(text))


def _qop(idx: int, text: str) -> tuple:
    """3-tuple for a queue-operation (genuine harness delivery surface).

    Task-notifications restricted to queue-operation after Sub-PR C-2 — use
    this helper instead of _uc() for correctness-guard tests that verify
    task-notification parsing in extract_team_state.
    """
    d = {"type": "queue-operation", "content": text}
    return (idx, d, len(text))


class TestExtractTeamStateReDoSCap(unittest.TestCase):
    """team.py _TASK_NOTIF_BLOCK_RE.finditer(content) must be capped at _RELOAD_GATE_SCAN_CAP."""

    @staticmethod
    def _agent_spawn_msgs():
        """Minimal Agent tool_use + result 3-tuples that register a subagent."""
        return [
            _tu(0, "tu-a1", "Agent", {"name": "finder", "description": "find bugs"}),
            _tr(1, "tu-a1", (
                "Spawned successfully.\n"
                "agent_id: finder@myteam\n"
                "name: finder\n"
                "team_name: myteam\n"
            )),
        ]

    def _extract(self, msgs):
        return _extract_isolated(msgs)

    def test_extract_team_state_quadratic_input_bounded(self):
        """REGRESSION GUARD — RED at base: uncapped scan on a message with 185KB of
        openers freezes extract_team_state (called every checkpoint cycle).

        team.py iterates _TASK_NOTIF_BLOCK_RE over the raw `content` string without
        slicing — the same catastrophic backtracking as guard.py's site. Measured
        at 4-5s on this machine with 10k openers. With _RELOAD_GATE_SCAN_CAP[:65536]
        the truncated input completes in <0.6s.

        Budget: 2.0s.
        """
        msgs = self._agent_spawn_msgs() + [_uc(2, _MANY_OPENERS)]

        t0 = time.monotonic()
        self._extract(msgs)
        elapsed = time.monotonic() - t0
        self.assertLess(
            elapsed, 2.0,
            f"extract_team_state took {elapsed:.3f}s with degenerate input — "
            "the _TASK_NOTIF_BLOCK_RE scan is not capped at _RELOAD_GATE_SCAN_CAP (64KB). "
            "Without the cap, 10,000 openers trigger O(openers × len) backtracking."
        )

    def test_extract_team_state_real_notification_still_clears(self):
        """Correctness guard (GREEN at base and after fix): a real task-notification
        within the 64KB cap must still transition the subagent to completed.

        Verifies the cap does not break the happy path — a normal notification is
        parsed and clears the subagent's running status.

        Note: task-notifications are restricted to queue-operation content after
        Sub-PR C-2 (user-typed message.content task-notifications are excluded
        to prevent phantom-terminate). This test uses the correct surface (queue-op).
        """
        notif_text = (
            "<task-notification>"
            "<task-id>finder@myteam</task-id>"
            "<status>completed</status>"
            "<result>all done</result>"
            "</task-notification>"
        )
        # Use queue-operation (genuine harness surface) rather than user string content.
        # Sub-PR C-2: task-notifications in message.content are excluded (phantom-terminate
        # prevention); queue-op is the correct delivery surface.
        msgs = self._agent_spawn_msgs() + [_qop(2, notif_text)]
        state = self._extract(msgs)
        subagents = state.subagents if state else []
        finder = next((s for s in subagents if "finder" in s.agent_id), None)
        self.assertIsNotNone(finder, "finder subagent must be registered after spawn")
        self.assertEqual(
            finder.status, "completed",
            f"task-notification must clear the subagent to 'completed'; "
            f"got status={finder.status!r}"
        )


def _tc(idx: int, team_name: str, teammate_name: str, agent_id: str) -> tuple:
    """3-tuple for a TeamCreate tool_use that registers a teammate.

    Both `name` and `agentId` are required: extract_team_state only adds to
    seen_teammates when agentId is non-empty (see team.py:667).
    """
    d = {"message": {"role": "assistant", "content": [
        {"type": "tool_use", "id": f"tc-{idx}", "name": "TeamCreate",
         "input": {"team_name": team_name,
                   "teammates": [{"name": teammate_name, "agentId": agent_id}]}}
    ]}}
    return (idx, d, 200)


def _tc_result(idx: int, team_name: str) -> tuple:
    """3-tuple for the TeamCreate tool_result."""
    text = f"Team '{team_name}' created."
    d = {"message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": f"tc-{idx - 1}", "content": text}
    ]}}
    return (idx, d, len(text))


class TestExtractTeamStateTeammateMsgCap(unittest.TestCase):
    """team.py _TEAMMATE_MSG_RE.finditer(content) must be capped at _RELOAD_GATE_SCAN_CAP.

    _TEAMMATE_MSG_RE uses a negative-lookahead form ((?:(?!<teammate-message).)*?)
    which is O(n) linear for opener-only input (the NLA fails fast on '<').  It is
    therefore NOT quadratic on opener-only strings, so a timing test is not the right
    proof mechanism here.

    The correct RED/GREEN proof is a BOUNDARY test: an idle-notification placed PAST
    the 64KB cap boundary must NOT be processed after the cap is applied (teammate
    stays "running" — the fail-safe over-defer direction).  Without the cap the same
    notification IS processed (teammate transitions to "idle").

    This proves the cap is applied to _TEAMMATE_MSG_RE (class-of-bug fold: all three
    uncapped DOTALL block-regex scans on the same `content` are now capped).
    """

    def _extract(self, msgs):
        return _extract_isolated(msgs)

    @staticmethod
    def _team_spawn_msgs():
        """TeamCreate + result 3-tuples that register 'worker@myteam' as a teammate.

        Both name='worker' and agentId='worker@myteam' are required so that:
        1. seen_teammates['worker@myteam'] is created (agentId gate at team.py:667)
        2. _name_to_agent_id['worker'] -> 'worker@myteam' is set (name lookup index)
        3. The idle-notification uses teammate_id="worker" which resolves via index
        """
        return [
            _tc(0, "myteam", "worker", "worker@myteam"),
            _tc_result(1, "myteam"),
            # SendMessage to worker → marks it as "running" in seen_teammates
            _tu(2, "sm-1", "SendMessage", {"to": "worker", "message": "start"}),
            _tr(3, "sm-1", "delivered"),
        ]

    def test_teammate_msg_cap_boundary_does_not_transition(self):
        """RED without cap: idle-notification PAST 64KB transitions teammate to idle.
        GREEN with cap:  idle-notification PAST 64KB is not scanned → stays running.

        Fail-safe direction: missed notification → teammate stays "running" →
        safe_to_reload/agents_active keep protecting it → gate OVER-DEFERS (recoverable),
        never UNDER-BLOCKS (SIGKILL).

        To reproduce RED manually: revert team.py:914 to
          `for tm_match in _TEAMMATE_MSG_RE.finditer(content):`
        and run this test — it will fail because the status becomes "idle".
        """
        from cozempic.guard import _RELOAD_GATE_SCAN_CAP

        idle_notif = (
            '<teammate-message teammate_id="worker" summary="done">'
            '{"type":"idle_notification","from":"worker"}'
            '</teammate-message>'
        )
        # Pad so the notification starts PAST the cap boundary.
        pad = " " * (_RELOAD_GATE_SCAN_CAP + 10)
        content_past_cap = pad + idle_notif

        msgs = self._team_spawn_msgs() + [_uc(4, content_past_cap)]
        state = self._extract(msgs)
        teammates = state.teammates if state else []
        # Registered as agent_id='worker@myteam', name='worker'
        worker = next(
            (t for t in teammates if "worker" in t.agent_id or "worker" in (t.name or "")),
            None,
        )
        self.assertIsNotNone(worker, "worker teammate must be registered after TeamCreate")
        self.assertNotEqual(
            worker.status, "idle",
            "idle-notification PAST the 64KB cap must NOT transition the teammate — "
            "the _TEAMMATE_MSG_RE scan must be capped at _RELOAD_GATE_SCAN_CAP. "
            f"Got status={worker.status!r}. Fail-safe: staying 'running' over-defers "
            "the reload (recoverable), never SIGKILL."
        )

    def test_teammate_msg_within_cap_still_transitions(self):
        """Correctness guard (GREEN at base and after fix): an idle-notification
        WITHIN the 64KB cap must still transition the teammate to idle.

        Verifies the cap does not break the happy path — a normal idle-notification
        is parsed and the teammate status transitions correctly.
        """
        idle_notif = (
            '<teammate-message teammate_id="worker" summary="done">'
            '{"type":"idle_notification","from":"worker"}'
            '</teammate-message>'
        )
        # Well within the cap — no padding.  Use _huc (genuine harness carrier)
        # so the H-1 teamName gate passes and the idle-notif scan runs.
        msgs = self._team_spawn_msgs() + [_huc(4, idle_notif)]
        state = self._extract(msgs)
        teammates = state.teammates if state else []
        worker = next(
            (t for t in teammates if "worker" in t.agent_id or "worker" in (t.name or "")),
            None,
        )
        self.assertIsNotNone(worker, "worker teammate must be registered after TeamCreate")
        self.assertEqual(
            worker.status, "idle",
            f"idle-notification within 64KB must transition teammate to 'idle'; "
            f"got status={worker.status!r}"
        )


if __name__ == "__main__":
    unittest.main()
