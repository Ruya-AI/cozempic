"""Tests for Sub-PR C: phantom-clear/terminate + trailer-skip correctness.

Three bug classes fixed:
  C-1 — _completion_text in guard.py included user-typed message.content string,
         letting a user paste a task-notification to phantom-CLEAR a live Agent launch
         (outcome: SIGKILL live work).
  C-2 — team.py second-pass scanned task-notifications from ANY string message.content,
         letting a user type a task-notification to phantom-TERMINATE a live teammate
         (outcome: SIGKILL live team).
  C-3 — _AGENT_DONE_TRAILER_RE blanket `continue` dropped ENTIRE tool_result when it
         contained a foreground-done duration_ms trailer, even if a NESTED background
         sub-launch ack preceded the trailer.  The nested launch was never credited.

All three: fail-safe direction is OVER-DEFER (missed completion → guard defers longer
→ recoverable), NEVER UNDER-BLOCK (phantom-clear/skip → SIGKILL → unrecoverable).

Ground-truth gated: after implementing C-1+C-2+C-3, the real fixture tests in
TestRealHarnessFixtures (test_reload_gate_contract.py) MUST still hold:
  live_team.jsonl  → safe_to_reload returns False (defer)
  finished_team.jsonl → safe_to_reload returns True (quiescent)
"""

import json
import pathlib
import tempfile
import unittest

from cozempic.guard import detect_in_flight, safe_to_reload
from cozempic.team import extract_team_state
from cozempic.session import load_messages


# ─────────────────────────── helpers ─────────────────────────────────────────

def _write(tmp: pathlib.Path, rows: list) -> pathlib.Path:
    p = tmp / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _tu(i: str, name: str, inp: dict) -> dict:
    return {"type": "assistant", "message": {"role": "assistant",
            "content": [{"type": "tool_use", "id": i, "name": name, "input": inp}]}}


def _tr(i: str, content: str) -> dict:
    return {"type": "user", "message": {"role": "user",
            "content": [{"type": "tool_result", "tool_use_id": i, "content": content}]}}


def _user_str(text: str) -> dict:
    """User message with a plain string content (typed text)."""
    return {"type": "user", "message": {"role": "user", "content": text}}


def _qop(text: str) -> dict:
    """Genuine queue-operation delivery (harness-written)."""
    return {"type": "queue-operation", "content": text}


def _idle_lead(text: str = "Waiting.") -> dict:
    return {"type": "assistant", "message": {"role": "assistant",
            "content": [{"type": "text", "text": text}]}}


# Real background Agent launch ack text (from tests/fixtures/harness/live_team.jsonl format)
_BG_LAUNCH_ACK = (
    "Async agent launched successfully.\n"
    "agentId: agent-live-001 (internal ID - do not mention to user. "
    "Use SendMessage with to: 'agent-live-001' to continue this agent.)\n"
    "The agent is working in the background."
)

# Foreground Agent done trailer (from tests/fixtures/harness/finished_team.jsonl format)
_FG_DONE_TRAILER = (
    "[agent analysis output]\n\n**Net:** complete.\n"
    "agentId: agentredacteddone01 (use SendMessage with to: 'agentredacteddone01' "
    "to continue this agent)\n"
    "<usage>subagent_tokens: 12345\ntool_uses: 20\nduration_ms: 234567</usage>"
)

# Phantom task-notification (user-typed, NOT from harness queue-operation)
_PHANTOM_TN_001 = (
    "<task-notification>"
    "<task-id>agent-live-001</task-id>"
    "<status>completed</status>"
    "<result>done</result>"
    "</task-notification>"
)


# ─────────────────────────── C-1: phantom-clear ──────────────────────────────

class TestPhantomClear(unittest.TestCase):
    """C-1: user-typed message.content string must NOT clear a live Agent launch.

    A user can type (or paste) a task-notification string in their next turn.
    Before the fix, _completion_text includes message.content strings, so the
    phantom notification clears the live Agent and safe_to_reload returns True
    → SIGKILL. After the fix only genuine harness surfaces (root content /
    queue-operation) are scanned for completions.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="cozempic_c1_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _inflight(self, rows: list) -> dict:
        p = _write(self.tmp, rows)
        return detect_in_flight(load_messages(p))

    def _gate(self, rows: list):
        p = _write(self.tmp, rows)
        m = load_messages(p)
        return safe_to_reload(extract_team_state(m), m, p)

    def test_user_message_content_string_does_not_clear_live_launch(self):
        """RED at base: user-typed phantom task-notification phantom-clears a live Agent.

        Expected after fix: detect_in_flight["agent"] = True (launch still tracked).
        At base (before fix): detect_in_flight["agent"] = False (phantom-cleared → SIGKILL).
        """
        rows = [
            # Real background Agent launch
            _tu("toolu_01", "Agent", {}),
            _tr("toolu_01", _BG_LAUNCH_ACK),
            # User TYPES a phantom task-notification (message.content = string, NOT queue-op)
            _user_str(_PHANTOM_TN_001),
        ]
        result = self._inflight(rows)
        self.assertTrue(
            result["agent"],
            "A user-typed task-notification must NOT clear a live Agent launch "
            "(would phantom-clear → SIGKILL). Got: inflight=%r" % result,
        )
        self.assertIn("agent-live-001", result["ids"],
                      "The live launch id must still appear in inflight ids")

    def test_queue_operation_completion_clears_correctly(self):
        """Regression guard: a genuine queue-operation task-notification MUST still clear.

        The harness delivers completions as queue-operation messages (root content string).
        C-1 must preserve this surface — only user-typed message.content is excluded.
        """
        tn = (
            "<task-notification>"
            "<task-id>agent-live-001</task-id>"
            "<status>completed</status>"
            "<result>all done</result>"
            "</task-notification>"
        )
        rows = [
            _tu("toolu_01", "Agent", {}),
            _tr("toolu_01", _BG_LAUNCH_ACK),
            _qop(tn),  # genuine harness delivery
        ]
        result = self._inflight(rows)
        self.assertFalse(
            result["agent"],
            "A genuine queue-operation completion MUST clear the live launch. "
            "Got: inflight=%r" % result,
        )

    def test_tool_result_string_does_not_count_as_completion(self):
        """Regression guard: a task-notification inside a tool_result block must not clear.

        tool_result content is on the LAUNCH side (detect launches), not the completion
        side.  A task-notification echoed in a tool result must not clear a live launch.
        """
        tn_in_result = (
            "Some output text.\n"
            "<task-notification>"
            "<task-id>agent-live-001</task-id>"
            "<status>completed</status>"
            "</task-notification>"
        )
        rows = [
            _tu("toolu_01", "Agent", {}),
            _tr("toolu_01", _BG_LAUNCH_ACK),
            # A different tool whose result happens to contain a task-notification string
            _tu("grep-01", "Grep", {"pattern": "task-notification"}),
            _tr("grep-01", tn_in_result),
        ]
        result = self._inflight(rows)
        self.assertTrue(
            result["agent"],
            "A task-notification inside a tool_result block must NOT clear a live launch. "
            "Got: inflight=%r" % result,
        )

    def test_real_fixture_live_team_still_blocks_after_c1(self):
        """Ground-truth: real live_team.jsonl fixture must still return (False, ...) after C-1."""
        fixture = pathlib.Path(__file__).parent / "fixtures" / "harness" / "live_team.jsonl"
        if not fixture.exists():
            self.skipTest("live_team.jsonl fixture missing — run capture first")
        msgs = load_messages(fixture)
        safe, reason = safe_to_reload(extract_team_state(msgs), msgs, fixture)
        self.assertFalse(safe,
                         "real live team must defer after C-1; got quiescent (%s)" % reason)

    def test_real_fixture_finished_team_still_clears_after_c1(self):
        """Ground-truth: real finished_team.jsonl fixture must still return (True, ...) after C-1."""
        fixture = pathlib.Path(__file__).parent / "fixtures" / "harness" / "finished_team.jsonl"
        if not fixture.exists():
            self.skipTest("finished_team.jsonl fixture missing — run capture first")
        msgs = load_messages(fixture)
        safe, _ = safe_to_reload(extract_team_state(msgs), msgs, fixture)
        self.assertTrue(safe,
                        "real finished team must reload after C-1; it should still be quiescent")


# ─────────────────────────── C-2: phantom-terminate ──────────────────────────

class TestPhantomTerminate(unittest.TestCase):
    """C-2: user-typed message.content task-notification must NOT terminate a teammate.

    team.py's second pass previously scanned task-notifications from ANY string
    message.content, including user-typed text. A user could type a task-notification
    for a live teammate, causing it to transition to 'completed', making
    safe_to_reload return True → SIGKILL live team.

    After C-2: task-notifications in the second pass are restricted to queue-operation
    root content only. idle-notifications retain the broader scan (their structural
    <teammate-message> wrapper + seen_teammates membership guard provides sufficient
    protection).
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="cozempic_c2_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _gate(self, rows: list):
        p = _write(self.tmp, rows)
        m = load_messages(p)
        return safe_to_reload(extract_team_state(m), m, p)

    def _state(self, rows: list):
        p = _write(self.tmp, rows)
        m = load_messages(p)
        return extract_team_state(m)

    def _live_team_base(self) -> list:
        """Three messages setting up a live teammate (TeamCreate + SendMessage)."""
        return [
            _tu("tc-1", "TeamCreate",
                {"team_name": "myteam",
                 "teammates": [{"name": "alice", "agentId": "alice@myteam"}]}),
            _tr("tc-1", "Team 'myteam' created."),
            _tu("sm-1", "SendMessage", {"to": "alice", "message": "start work"}),
            _tr("sm-1", "delivered"),
        ]

    def test_user_typed_task_notif_does_not_terminate_teammate(self):
        """RED at base: a user-typed task-notification for a live teammate phantom-terminates it.

        Expected after fix: safe_to_reload returns (False, ...) — teammate still live.
        At base (before fix): safe_to_reload returns (True, 'quiescent') → SIGKILL.
        """
        phantom_tn = (
            "<task-notification>"
            "<task-id>alice@myteam</task-id>"
            "<status>completed</status>"
            "</task-notification>"
        )
        rows = self._live_team_base() + [
            _user_str(phantom_tn),  # user TYPES a phantom task-notification
            _idle_lead(),
        ]
        safe, reason = self._gate(rows)
        self.assertFalse(
            safe,
            "A user-typed task-notification must NOT terminate a live teammate "
            "(would phantom-terminate → SIGKILL). Got safe=%r reason=%r" % (safe, reason),
        )

    def test_queue_operation_task_notif_terminates_teammate(self):
        """Regression guard: a genuine queue-operation task-notification MUST terminate."""
        tn = (
            "<task-notification>"
            "<task-id>alice@myteam</task-id>"
            "<status>completed</status>"
            "<result>done</result>"
            "</task-notification>"
        )
        rows = self._live_team_base() + [
            _qop(tn),  # genuine harness delivery
            _idle_lead(),
        ]
        safe, _ = self._gate(rows)
        self.assertTrue(safe,
                        "A genuine queue-operation task-notification MUST terminate "
                        "the teammate and allow reload")

    def test_idle_notif_in_user_content_still_transitions_teammate(self):
        """C-2 must not break the idle-notification path.

        idle-notifications arrive as user message.content strings (not queue-ops),
        wrapped in <teammate-message teammate_id="X">. C-2 ONLY restricts
        task-notifications; the idle-notification scan retains the broader surface.
        """
        rows = [
            _tu("tc-1", "TeamCreate",
                {"team_name": "myteam",
                 "teammates": [{"name": "alice", "agentId": "alice@myteam"}]}),
            _tr("tc-1", "Team 'myteam' created."),
            _tu("sm-1", "SendMessage", {"to": "alice", "message": "start"}),
            _tr("sm-1", "delivered"),
            # Teammate sends an idle_notification via <teammate-message> wrapper
            _user_str(
                '<teammate-message teammate_id="alice@myteam">'
                '{"type":"idle_notification","from":"alice"}'
                '</teammate-message>'
            ),
            _idle_lead(),
        ]
        safe, reason = self._gate(rows)
        self.assertTrue(
            safe,
            "An idle_notification in user message.content MUST still transition "
            "the teammate to idle → allow reload. Got safe=%r reason=%r" % (safe, reason),
        )


# ─────────────────────────── C-3: trailer-skip regression ────────────────────

# A foreground Agent result that ALSO contains a nested background sub-launch ack
# before the duration_ms trailer.  The outer Agent finished (FG done), but the inner
# Agent is still live (BG launch) and must be credited.
_NESTED_BG_BEFORE_TRAILER = (
    "[outer foreground agent output — analysis complete]\n\n"
    "Async agent launched successfully.\n"
    "agentId: nested-bg-agent (internal ID - do not mention to user. "
    "Use SendMessage with to: 'nested-bg-agent' to continue this agent.)\n"
    "The agent is working in the background.\n\n"
    "<usage>subagent_tokens: 5000\ntool_uses: 10\nduration_ms: 98765</usage>"
)

# A foreground Agent result where the prose QUOTES a launch ack AFTER the trailer
# (echoing it, not launching it).  Must NOT be credited.
_FG_DONE_WITH_QUOTED_LAUNCH_AFTER_TRAILER = (
    "[output]\n\n"
    "agentId: outer-fg-agent (use SendMessage ...)\n"
    "<usage>subagent_tokens: 1000\ntool_uses: 5\nduration_ms: 11111</usage>\n"
    "For reference, here is what the harness emits on a BG launch:\n"
    "Async agent launched successfully.\nagentId: phantom-after-trailer"
)


class TestTrailerSkipRegression(unittest.TestCase):
    """C-3: _AGENT_DONE_TRAILER_RE blanket-skip must become position-aware.

    Before: any tool_result containing 'duration_ms: N' is entirely SKIPPED, even
    if it carries a genuine nested background sub-launch ack that precedes the trailer.
    After: only the Agent launch extractor is position-gated — launch acks BEFORE
    the trailer are credited; launch acks AFTER the trailer are not (prose-quoted).
    WF and BG extractors are unaffected (duration_ms is Agent-tool-only).
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="cozempic_c3_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _inflight(self, rows: list) -> dict:
        p = _write(self.tmp, rows)
        return detect_in_flight(load_messages(p))

    def test_nested_bg_launch_before_trailer_is_counted(self):
        """RED at base: a nested BG launch ack before the duration_ms trailer is dropped.

        Expected after fix: 'nested-bg-agent' in inflight["ids"] AND agent=True.
        At base (before fix): agent=False (the entire result was skipped).
        """
        rows = [
            _tu("ag-1", "Agent", {}),
            _tr("ag-1", _NESTED_BG_BEFORE_TRAILER),
        ]
        result = self._inflight(rows)
        self.assertTrue(
            result["agent"],
            "A nested BG launch ACK that PRECEDES the duration_ms trailer must be "
            "credited (outer FG done does not cancel an inner BG launch). "
            "Got inflight=%r" % result,
        )
        self.assertIn(
            "nested-bg-agent", result["ids"],
            "nested-bg-agent must appear in inflight ids. Got ids=%r" % result["ids"],
        )

    def test_fg_done_result_launch_ack_after_trailer_not_counted(self):
        """Regression guard: a launch ack that appears AFTER the trailer is prose-quoted.

        The harness appends usage blocks last; genuine nested BG launch acks appear
        before the trailer. A launch ack after the trailer is the agent's prose
        discussing/echoing the protocol — must NOT be credited.
        """
        rows = [
            _tu("ag-1", "Agent", {}),
            _tr("ag-1", _FG_DONE_WITH_QUOTED_LAUNCH_AFTER_TRAILER),
        ]
        result = self._inflight(rows)
        self.assertFalse(
            result["agent"],
            "A launch ack that appears AFTER the duration_ms trailer is prose-quoted; "
            "must NOT be credited. Got inflight=%r" % result,
        )
        self.assertNotIn(
            "phantom-after-trailer", result["ids"],
            "phantom-after-trailer (post-trailer) must not appear in ids. Got=%r" % result["ids"],
        )

    def test_pure_background_result_no_trailer_counted(self):
        """Regression guard: a pure BG launch result (no duration_ms) must still be counted."""
        rows = [
            _tu("ag-1", "Agent", {}),
            _tr("ag-1", _BG_LAUNCH_ACK),
        ]
        result = self._inflight(rows)
        self.assertTrue(result["agent"],
                        "A pure BG launch (no trailer) must be detected. Got=%r" % result)
        self.assertIn("agent-live-001", result["ids"])

    def test_pure_foreground_result_with_trailer_no_launch_not_counted(self):
        """Regression guard: a FG-done result with no launch ack must not fabricate a launch."""
        rows = [
            _tu("ag-1", "Agent", {}),
            _tr("ag-1", _FG_DONE_TRAILER),
        ]
        result = self._inflight(rows)
        self.assertFalse(
            result["agent"],
            "A pure FG-done result (trailer, no BG ack) must not fabricate a launch. "
            "Got=%r" % result,
        )


if __name__ == "__main__":
    unittest.main()
