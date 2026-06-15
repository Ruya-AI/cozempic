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
            "Without the cap, 200,000 openers trigger O(openers × len) backtracking."
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
    """3-tuple for a user message with plain string content."""
    d = {"message": {"role": "user", "content": text}}
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
        """Call extract_team_state with config isolation (no live ~/.claude/teams/)."""
        from cozempic.team import extract_team_state
        with patch("cozempic.team.load_team_configs", return_value=[]):
            return extract_team_state(msgs)

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
            "Without the cap, 200,000 openers trigger O(openers × len) backtracking."
        )

    def test_extract_team_state_real_notification_still_clears(self):
        """Correctness guard (GREEN at base and after fix): a real task-notification
        within the 64KB cap must still transition the subagent to completed.

        Verifies the cap does not break the happy path — a normal notification is
        parsed and clears the subagent's running status.
        """
        notif_text = (
            "<task-notification>"
            "<task-id>finder@myteam</task-id>"
            "<status>completed</status>"
            "<result>all done</result>"
            "</task-notification>"
        )
        msgs = self._agent_spawn_msgs() + [_uc(2, notif_text)]
        state = self._extract(msgs)
        subagents = state.subagents if state else []
        finder = next((s for s in subagents if "finder" in s.agent_id), None)
        self.assertIsNotNone(finder, "finder subagent must be registered after spawn")
        self.assertEqual(
            finder.status, "completed",
            f"task-notification must clear the subagent to 'completed'; "
            f"got status={finder.status!r}"
        )


if __name__ == "__main__":
    unittest.main()
