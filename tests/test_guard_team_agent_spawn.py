"""F1 fix — RED→GREEN regression guards for Agent-spawned team visibility.

Each test that is a genuine regression guard is PROVEN RED at base 4f15d6d before
any fix is applied; characterization / invariant-preservation tests that pass at
base are clearly labelled as such in their docstrings.

Test classes:
  TestTeamNameExtraction           — P0-A: team_name key mismatch
  TestAgentSpawnRecognition        — P0-B: Agent tool recognition in extract_team_state
  TestSendMessageByNameLookup      — P0-C: by-name SendMessage resolution
  TestIdleNotificationTransition   — P0-D: idle_notification terminal transition
  TestStatusBasedSafetyInvariant   — documents status-only invariant (P0-E gate removed)
  TestFullScenario                 — integration: pure-SendMessage team, no shared tasks
  TestStaleConfigAntiWedge         — C-1: stale config can't inject "running" status
  TestIdleNotificationPruneProtection — C-2: idle carrier prune protection
  TestFailedAgentSpawnNoWedge      — H-1: failed spawn placeholder cleanup
  TestGateRemoval                  — C-1 gate-removal regression guards
  TestNestedTeammateMessageRegex   — M-1: nested teammate-message regex tightening

Isolation:
  All tests go through _extract() which patches load_team_configs to return []
  (no live ~/.claude/teams/ config merge contamination — L11 isolation principle).
  No real 'cozempic guard --daemon', no os.kill on real PIDs.
"""

import unittest
from pathlib import Path
from unittest.mock import patch


# ─── helpers ────────────────────────────────────────────────────────────────

def _extract(msgs):
    """Call extract_team_state with config isolation (no live ~/.claude/teams/ merge).

    All tests in this module must go through this wrapper to avoid contamination
    from the test machine's real team config.json files — which can cause tests to
    pass at base for the wrong reason (config-merge finding the live config before
    the JSONL extraction bug is fixed). L11 isolation principle.
    """
    from cozempic.team import extract_team_state
    with patch("cozempic.team.load_team_configs", return_value=[]):
        return extract_team_state(msgs)


def _tool_use(id_, name, inp):
    return {"message": {"role": "assistant", "content": [
        {"type": "tool_use", "id": id_, "name": name, "input": inp}
    ]}}


def _tool_result(tool_use_id, text):
    return {"message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tool_use_id,
         "content": text}
    ]}}


def _user_content(text):
    """A user message with a plain-string content (e.g. teammate-message XML).

    No teamName — simulates user-typed content (not a genuine harness carrier).
    For genuine idle-notification carriers, use _harness_user_content().
    """
    return {"message": {"role": "user", "content": text}}


def _harness_user_content(text, team_name="myteam"):
    """Genuine harness teammate-message carrier — includes top-level teamName (H-1).

    The harness always sets teamName on messages that carry idle-notifications.
    User-typed messages have no such field and are rejected by the H-1 gate.
    """
    return {"teamName": team_name, "message": {"role": "user", "content": text}}


# Canonical Agent-spawn result text as observed in real transcripts (2026-06-08)
_SPAWN_RESULT_P1 = (
    "Spawned successfully.\n"
    "agent_id: finder-p1@myteam\n"
    "name: finder-p1\n"
    "team_name: myteam\n"
    "The agent is now running and will receive instructions via mailbox."
)

_SPAWN_RESULT_P2 = (
    "Spawned successfully.\n"
    "agent_id: finder-p2@myteam\n"
    "name: finder-p2\n"
    "team_name: myteam\n"
    "The agent is now running and will receive instructions via mailbox."
)

_IDLE_NOTIFICATION_P2 = (
    '<teammate-message teammate_id="finder-p2" summary="idle">'
    '{"type":"idle_notification","from":"finder-p2","timestamp":"2026-06-08T10:00:00Z","idleReason":"available"}'
    '</teammate-message>'
)


def _spawn_msgs_p1(idx_offset=0):
    """Two messages: Agent tool_use + Agent tool_result for finder-p1."""
    return [
        (idx_offset, _tool_use("u1", "Agent", {"name": "finder-p1", "description": "find bugs"}), 200),
        (idx_offset + 1, _tool_result("u1", _SPAWN_RESULT_P1), 300),
    ]


# ─── TestTeamNameExtraction (P0-A) ──────────────────────────────────────────

class TestTeamNameExtraction(unittest.TestCase):
    """P0-A: TeamCreate emits 'team_name' key, but extract_team_state reads 'name'.

    Regression guard: test_teamcreate_team_name_key_extracted is RED at base
    (extract_team_state returns team_name="" before the fix).

    Invariant test: test_teamcreate_name_key_still_works is GREEN at base
    (backward compat — sessions that already used 'name' keep working).
    """

    def test_teamcreate_team_name_key_extracted(self):
        """REGRESSION GUARD — RED at base: inp.get('name') misses 'team_name' key.

        Before fix: state.team_name == "" (the key 'name' is absent; only
        'team_name' is present, so the fallback in the old code is the empty
        default).  After fix: state.team_name == "myteam".
        """
        msgs = [
            (0, _tool_use("u1", "TeamCreate", {
                "team_name": "myteam",
                "description": "test team",
            }), 100),
        ]
        state = _extract(msgs)
        self.assertEqual(state.team_name, "myteam",
                         "TeamCreate with 'team_name' key must set state.team_name")

    def test_teamcreate_name_key_still_works(self):
        """INVARIANT (GREEN at base): backward compat — 'name' key still works.

        This test documents that sessions using the 'name' key continue to work
        after the fix. It is NOT a regression guard (passes at base).
        """
        msgs = [
            (0, _tool_use("u1", "TeamCreate", {
                "name": "myteam",
                "description": "test team",
            }), 100),
        ]
        state = _extract(msgs)
        self.assertEqual(state.team_name, "myteam",
                         "'name' key fallback must still work for backward compat")

    def test_teamcreate_both_keys_prefers_team_name(self):
        """REGRESSION GUARD (code-review max): when BOTH keys are present
        (rollout overlap), the authoritative 'team_name' must win over the
        legacy 'name'. Old priority `inp.get('name', inp.get('team_name'))`
        picked the stale legacy value → a wrong name-join in merge_config.
        """
        msgs = [
            (0, _tool_use("u1", "TeamCreate", {
                "name": "stale-legacy-name",
                "team_name": "real-team",
                "description": "rollout overlap",
            }), 100),
        ]
        state = _extract(msgs)
        self.assertEqual(state.team_name, "real-team",
                         "authoritative 'team_name' must win over legacy 'name'")

    def test_teamcreate_team_name_key_with_inline_teammates(self):
        """REGRESSION GUARD — RED at base: team_name="" when key is 'team_name'.

        Asserts team_name is set correctly; the teammate is a secondary check.
        """
        msgs = [
            (0, _tool_use("u1", "TeamCreate", {
                "team_name": "audit-team",
                "description": "audit team",
                "teammates": [{"agentId": "finder-p1@audit-team", "name": "finder-p1"}],
            }), 200),
        ]
        state = _extract(msgs)
        self.assertEqual(state.team_name, "audit-team",
                         "team_name must be extracted from 'team_name' key")
        self.assertTrue(any(t.name == "finder-p1" for t in state.teammates),
                        "inline teammate must still be populated")


# ─── TestAgentSpawnRecognition (P0-B) ────────────────────────────────────────

class TestAgentSpawnRecognition(unittest.TestCase):
    """P0-B: 'Agent' tool-use is unrecognised; teammates stay empty.

    All tests in this class are REGRESSION GUARDs — RED at base because
    'Agent' ∉ TEAM_TOOL_NAMES so the spawn is completely invisible.
    """

    def test_agent_spawn_creates_teammate_with_running_status(self):
        """REGRESSION GUARD — RED at base: Agent spawn not recognised; teammates=[].

        After fix: len(state.teammates) >= 1, with a non-benign, non-terminal
        status ('running') that blocks safe_to_reload.
        """
        from cozempic.guard import _TEAMMATE_BENIGN, _STATUS_TERMINAL
        state = _extract(_spawn_msgs_p1())
        self.assertGreaterEqual(len(state.teammates), 1,
                                "Agent spawn must create a TeammateInfo entry")
        mate = next((t for t in state.teammates if t.name == "finder-p1"), None)
        self.assertIsNotNone(mate, "teammate must have name 'finder-p1'")
        s = (mate.status or "").strip().lower()
        self.assertNotIn(s, _TEAMMATE_BENIGN,
                         f"spawn status {s!r} must not be benign; must block reload")
        self.assertNotIn(s, _STATUS_TERMINAL,
                         f"spawn status {s!r} must not be terminal; work is ongoing")

    def test_agent_spawn_sets_team_name_from_result(self):
        """REGRESSION GUARD — RED at base: Agent spawn result not parsed; team_name=''.

        After fix: state.team_name == 'myteam' (parsed from spawn result).
        Only fires if no prior TeamCreate set team_name.
        """
        state = _extract(_spawn_msgs_p1())
        self.assertEqual(state.team_name, "myteam",
                         "team_name must be extracted from the Agent spawn result")

    def test_agent_spawn_real_agent_id_extracted(self):
        """REGRESSION GUARD — RED at base: spawn result not parsed; agentId stays placeholder.

        After fix: the TeammateInfo's agent_id is 'finder-p1@myteam' (the real
        agentId from the result text), not the tool_use_id placeholder.
        """
        state = _extract(_spawn_msgs_p1())
        self.assertGreaterEqual(len(state.teammates), 1)
        # The real agentId format includes the @team suffix
        agent_ids = {t.agent_id for t in state.teammates}
        self.assertIn("finder-p1@myteam", agent_ids,
                      "real agentId 'finder-p1@myteam' must be parsed from spawn result")

    def test_agent_spawn_triggers_unsafe_reload(self):
        """REGRESSION GUARD — RED at base: spawn invisible; safe_to_reload returns True.

        This is the core bug scenario: an active Agent-spawned teammate with no
        completion signal must block safe_to_reload (return False).
        After fix: safe_to_reload returns (False, reason) with 'teammate' in reason.
        """
        from cozempic.guard import safe_to_reload
        msgs = _spawn_msgs_p1()
        state = _extract(msgs)
        safe, reason = safe_to_reload(state, msgs, Path("/tmp/fake_session.jsonl"))
        self.assertFalse(safe,
                         "Active Agent-spawned teammate must block safe_to_reload")
        self.assertIn("teammate", reason,
                      f"block reason must mention 'teammate', got: {reason!r}")

    def test_agent_spawn_pruning_non_regression(self):
        """INVARIANT (GREEN at base, must stay GREEN): 'Agent' must NOT be in TEAM_TOOL_NAMES.

        Q-A decision: 'Agent' is NOT added to TEAM_TOOL_NAMES. It is only used in
        extract_team_state's own scope via _TEAM_EXTRACT_TOOL_NAMES. This ensures
        prune_with_team_protect does NOT over-protect non-team Agent calls.
        """
        from cozempic.team import TEAM_TOOL_NAMES
        self.assertNotIn("Agent", TEAM_TOOL_NAMES,
                         "'Agent' must NOT be in TEAM_TOOL_NAMES (would over-protect "
                         "non-team Agent calls in prune_with_team_protect)")

    def test_non_team_agent_call_not_protected_by_is_team_message(self):
        """INVARIANT (GREEN at base, must stay GREEN): _is_team_message returns False
        for Agent tool_use (since 'Agent' ∉ TEAM_TOOL_NAMES).

        Proves the decouple invariant: extract_team_state can recognize Agent spawns
        via its own internal predicate, while prune_with_team_protect's
        _is_team_message call does NOT tag them as team-protected.
        """
        from cozempic.team import _is_team_message
        agent_msg = {"message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "u99", "name": "Agent",
             "input": {"name": "solo-agent", "description": "one-off task"}}
        ]}}
        # With no pending_task_ids, _is_team_message must return False for Agent tool_use
        result = _is_team_message(agent_msg, pending_task_ids=set())
        self.assertFalse(result,
                         "_is_team_message must return False for Agent tool_use "
                         "(Agent ∉ TEAM_TOOL_NAMES — decouple invariant)")


# ─── TestSendMessageByNameLookup (P0-C) ─────────────────────────────────────

class TestSendMessageByNameLookup(unittest.TestCase):
    """P0-C: SendMessage to: bare name misses seen_teammates keyed by agentId.

    Regression guards: RED at base because SendMessage to="alice" (bare name)
    misses seen_teammates keyed by "alice@myteam" (full agentId).
    """

    def test_sendmessage_bare_name_reactivates_after_completion(self):
        """REGRESSION GUARD — RED at base: bare-name lookup misses agentId-keyed teammate.

        TeamCreate → SendMessage → completion → SendMessage (re-activate by bare name).
        The second SendMessage must resolve "alice" → "alice@myteam" and set
        status="running" (chronology: last SendMessage is AFTER the completion).

        Before fix: seen_teammates keyed by "alice@myteam"; to="alice" misses
        → the re-activation at line 3 is lost → status stays "completed"
        → safe_to_reload incorrectly says safe (but the teammate was re-activated).
        """
        msgs = [
            (0, _tool_use("u1", "TeamCreate", {
                "team_name": "myteam",
                "teammates": [{"agentId": "alice@myteam", "name": "alice"}],
            }), 100),
            (1, _tool_use("u2", "SendMessage", {"to": "alice", "message": "phase1"}), 100),
            (2, {"type": "queue-operation",
                 "content": ("<task-notification><task-id>alice@myteam</task-id>"
                             "<status>completed</status><summary>done</summary>"
                             "<result>r</result></task-notification>")}, 100),
            (3, _tool_use("u3", "SendMessage", {"to": "alice", "message": "phase2"}), 100),
        ]
        state = _extract(msgs)
        mate = next((t for t in state.teammates
                     if t.name == "alice" or t.agent_id == "alice@myteam"), None)
        self.assertIsNotNone(mate, "alice must be in teammates")
        self.assertEqual(mate.status, "running",
                         "SendMessage to bare 'alice' after completion must re-activate "
                         "teammate back to 'running'")

    def test_sendmessage_to_agent_spawn_by_name(self):
        """REGRESSION GUARD — RED at base: Agent-spawn + SendMessage by bare name.

        After Agent spawns finder-p1 (agentId = "finder-p1@myteam"),
        a SendMessage to="finder-p1" (bare name) must be recognized
        (bare-name → agentId index) and the teammate must remain active.
        """
        msgs = list(_spawn_msgs_p1()) + [
            (2, _tool_use("u2", "SendMessage", {"to": "finder-p1", "message": "start work"}), 100),
        ]
        state = _extract(msgs)
        mate = next((t for t in state.teammates
                     if t.name == "finder-p1" or "finder-p1" in t.agent_id), None)
        self.assertIsNotNone(mate, "finder-p1 must be in teammates after spawn")
        s = (mate.status or "").strip().lower()
        from cozempic.guard import _TEAMMATE_BENIGN, _STATUS_TERMINAL
        self.assertNotIn(s, _TEAMMATE_BENIGN,
                         f"After spawn + SendMessage, status must not be benign, got {s!r}")
        self.assertNotIn(s, _STATUS_TERMINAL,
                         f"After spawn + SendMessage, status must not be terminal, got {s!r}")


# ─── TestIdleNotificationTransition (P0-D) ───────────────────────────────────

class TestIdleNotificationTransition(unittest.TestCase):
    """P0-D: <teammate-message> idle_notification in user messages must transition
    the teammate to 'idle' (benign) status.

    All tests in this class are REGRESSION GUARDs — RED at base because:
    (a) P0-B makes Agent spawns invisible (teammates=[]), and
    (b) the second pass does not scan for idle_notification blocks.
    """

    def _msgs_with_idle(self):
        """Spawn + spawn result + user message with idle_notification for finder-p2."""
        return [
            (0, _tool_use("u1", "Agent", {"name": "finder-p2", "description": "find issues"}), 200),
            (1, _tool_result("u1", _SPAWN_RESULT_P2), 300),
            (2, _harness_user_content(_IDLE_NOTIFICATION_P2), 200),
        ]

    def test_idle_notification_transitions_to_idle(self):
        """REGRESSION GUARD — RED at base: idle_notification invisible; teammate stays 'running'.

        After fix: the teammate's status becomes 'idle' (∈ _TEAMMATE_BENIGN).
        """
        from cozempic.guard import _TEAMMATE_BENIGN
        state = _extract(self._msgs_with_idle())
        mate = next((t for t in state.teammates
                     if t.name == "finder-p2" or "finder-p2" in t.agent_id), None)
        self.assertIsNotNone(mate,
                             "finder-p2 must be in teammates after Agent spawn + result")
        s = (mate.status or "").strip().lower()
        self.assertIn(s, _TEAMMATE_BENIGN,
                      f"idle_notification must transition status to benign, got {s!r}")

    def test_idle_notification_allows_safe_reload(self):
        """REGRESSION GUARD — RED at base: spawn invisible; 'safe' for wrong reason (is_empty).

        After fix: the correct path — teammate exists + is idle → safe.
        The test also verifies state is NOT empty (proving the correct path is taken).
        """
        from cozempic.guard import safe_to_reload
        msgs = self._msgs_with_idle()
        state = _extract(msgs)
        self.assertFalse(state.is_empty(),
                         "state must not be empty after Agent spawn + idle_notification")
        safe, _ = safe_to_reload(state, msgs, Path("/tmp/fake_session.jsonl"))
        self.assertTrue(safe,
                        "After idle_notification, safe_to_reload must return True")

    def test_genuine_idle_then_sendmessage_chronology_blocks(self):
        """CHRONOLOGY GUARD: genuine idle (teamName) + later SendMessage keeps teammate running.

        Sequence: spawn → genuine idle-notif (teamName, transitions to idle) → SendMessage
        (re-activates at a later line).  The chronology guard (last_send_line > line_idx)
        must prevent the idle from sticking → teammate stays 'running' → gate blocks.

        This tests the ACTUAL chronology guard path (H-1 gate passes the genuine carrier
        in, idle transitions, then SendMessage at a later line re-activates).
        """
        from cozempic.guard import safe_to_reload, _TEAMMATE_BENIGN
        msgs = [
            (0, _tool_use("u1", "Agent", {"name": "finder-p2", "description": "find issues"}), 200),
            (1, _tool_result("u1", _SPAWN_RESULT_P2), 300),
            (2, _harness_user_content(_IDLE_NOTIFICATION_P2), 200),  # genuine idle at line 2
            (3, _tool_use("u3", "SendMessage", {"to": "finder-p2", "message": "one more task"}), 100),
        ]
        state = _extract(msgs)
        mate = next((t for t in state.teammates
                     if t.name == "finder-p2" or "finder-p2" in t.agent_id), None)
        self.assertIsNotNone(mate, "finder-p2 must be in teammates")
        s = (mate.status or "").strip().lower()
        self.assertNotIn(s, _TEAMMATE_BENIGN,
                         "SendMessage (line 3) after genuine idle (line 2) must re-activate "
                         f"teammate via chronology guard — status must not be benign, got {s!r}")
        safe, reason = safe_to_reload(state, msgs, Path("/tmp/fake_session.jsonl"))
        self.assertFalse(safe,
                         "Re-activated teammate (genuine idle → SendMessage) must block reload")

    def test_user_typed_idle_blocks_even_without_sendmessage(self):
        """H-1 gate skips user-typed idle — teammate stays running regardless of SendMessage.

        Replaces the pre-H-1 'not_applied_if_later_sendmessage' test which passed for
        the wrong reason: the idle was SKIPPED by the H-1 teamName gate (no teamName →
        continue), not re-activated by SendMessage.  This test pins that behavior
        explicitly so a reader knows why _user_content (no teamName) makes the test pass.
        """
        from cozempic.guard import safe_to_reload, _TEAMMATE_BENIGN
        msgs = [
            (0, _tool_use("u1", "Agent", {"name": "finder-p2", "description": "find issues"}), 200),
            (1, _tool_result("u1", _SPAWN_RESULT_P2), 300),
            # User-typed idle (no teamName) → H-1 gate skips it → no transition
            (2, _user_content(_IDLE_NOTIFICATION_P2), 200),
        ]
        state = _extract(msgs)
        mate = next((t for t in state.teammates
                     if t.name == "finder-p2" or "finder-p2" in t.agent_id), None)
        self.assertIsNotNone(mate, "finder-p2 must be in teammates")
        s = (mate.status or "").strip().lower()
        self.assertNotIn(s, _TEAMMATE_BENIGN,
                         "User-typed idle (no teamName) must NOT transition — H-1 gate "
                         f"skips it. Status must not be benign, got {s!r}")
        safe, _ = safe_to_reload(state, msgs, Path("/tmp/fake_session.jsonl"))
        self.assertFalse(safe, "Teammate with H-1-skipped idle must still block reload")

    def test_prose_teammate_message_does_not_transition(self):
        """INVARIANT: prose-only <teammate-message> must NOT transition to idle.

        Only the exact JSON idle_notification body triggers the transition.
        This prevents false-idle transitions from progress-update prose blocks.
        """
        from cozempic.guard import _TEAMMATE_BENIGN
        prose_msg = (
            '<teammate-message teammate_id="finder-p2" summary="progress update">'
            'I have completed the first 3 files and found no issues.'
            '</teammate-message>'
        )
        msgs = [
            (0, _tool_use("u1", "Agent", {"name": "finder-p2", "description": "find issues"}), 200),
            (1, _tool_result("u1", _SPAWN_RESULT_P2), 300),
            (2, _user_content(prose_msg), 200),
        ]
        state = _extract(msgs)
        mate = next((t for t in state.teammates
                     if t.name == "finder-p2" or "finder-p2" in t.agent_id), None)
        self.assertIsNotNone(mate, "finder-p2 must be in teammates after Agent spawn")
        s = (mate.status or "").strip().lower()
        self.assertNotIn(s, _TEAMMATE_BENIGN,
                         f"Prose teammate-message must NOT transition to benign, got {s!r}")


# ─── TestStatusBasedSafetyInvariant (documents gate-removal safety) ───────────

class TestStatusBasedSafetyInvariant(unittest.TestCase):
    """Documents the safety invariant that makes removing _team_is_current_session safe.

    The invariant: a non-benign ("running") teammate status can ONLY originate from
    THIS session's JSONL (Agent spawn / TeamCreate / SendMessage in this transcript).
    merge_config_into_state always assigns status="config" (∈ _TEAMMATE_BENIGN) to
    config-only members. So "running" = current-session, always. No session-ID
    comparison is needed.

    The removed P0-E gate was both redundant AND a source of false-safes (C-1): a
    stale same-name config could inject a different leadSessionId, causing the gate to
    skip the block for a LIVE "running" teammate → (True,'quiescent') → SIGKILL.
    """

    def test_running_teammate_always_blocks_regardless_of_session_id(self):
        """INVARIANT: any non-benign status blocks, regardless of lead_session_id.

        No session-ID comparison needed — "running" can only come from this session.
        """
        from cozempic.guard import safe_to_reload
        from cozempic.team import TeamState, TeammateInfo
        # Even with a completely arbitrary lead_session_id, "running" must block.
        for session_id in ("", "old-session", "aabbccdd", None):
            state = TeamState(
                team_name="myteam",
                lead_session_id=session_id or "",
                teammates=[TeammateInfo("alice@myteam", "alice", status="running")],
            )
            safe, reason = safe_to_reload(state, [], Path("/tmp/anysession.jsonl"))
            self.assertFalse(
                safe,
                f"'running' teammate must block regardless of lead_session_id={session_id!r}; "
                f"got safe={safe}, reason={reason!r}"
            )

    def test_config_only_teammate_never_blocks(self):
        """INVARIANT: config-only member (status='config', from merge_config_into_state)
        must never block safe_to_reload.

        merge_config_into_state assigns status='config' to members not seen in JSONL.
        'config' ∈ _TEAMMATE_BENIGN. This is the mechanism that prevents stale
        configs from wedging reloads — not a session-ID comparison.
        """
        from cozempic.guard import safe_to_reload
        from cozempic.team import TeamState, TeammateInfo
        state = TeamState(
            team_name="old-team",
            teammates=[TeammateInfo("alice@old-team", "alice", status="config")],
        )
        safe, _ = safe_to_reload(state, [], Path("/tmp/anysession.jsonl"))
        self.assertTrue(safe,
                        "config-only member (status='config') must never block reload")

    def test_idle_teammate_never_blocks(self):
        """INVARIANT: idle teammate (status='idle', from idle_notification) is benign."""
        from cozempic.guard import safe_to_reload
        from cozempic.team import TeamState, TeammateInfo
        state = TeamState(
            team_name="myteam",
            teammates=[TeammateInfo("alice@myteam", "alice", status="idle")],
        )
        safe, _ = safe_to_reload(state, [], Path("/tmp/anysession.jsonl"))
        self.assertTrue(safe, "idle teammate must not block reload")

    def test_terminal_teammate_never_blocks(self):
        """INVARIANT: terminal status (completed/failed/done) never blocks."""
        from cozempic.guard import safe_to_reload, _STATUS_TERMINAL
        from cozempic.team import TeamState, TeammateInfo
        for status in sorted(_STATUS_TERMINAL)[:3]:
            state = TeamState(
                team_name="myteam",
                teammates=[TeammateInfo("alice@myteam", "alice", status=status)],
            )
            safe, _ = safe_to_reload(state, [], Path("/tmp/anysession.jsonl"))
            self.assertTrue(safe,
                            f"terminal status={status!r} must not block reload")


# ─── TestFullScenario (integration) ──────────────────────────────────────────

class TestFullScenario(unittest.TestCase):
    """End-to-end integration: the exact audit scenario.

    The critical regression guard is test_pure_sendmessage_team_no_tasks_unsafe:
    a team coordinated purely via Agent spawns + SendMessage, with no shared tasks,
    must NOT be considered quiescent by safe_to_reload.

    This test class proves the entire fix chain from P0-A through P0-E working
    together on a realistic minimal message list.
    """

    def _pure_sendmessage_msgs(self, n_spawns=3):
        """Build a message list with n Agent spawns + SendMessages, no TaskCreate.

        Every tool_use must have a matching tool_result so that
        detect_in_flight.open_call stays False; otherwise open-call detection
        returns False for the wrong reason, masking the teammate gate under test.
        """
        msgs = []
        idx = 0
        # TeamCreate with 'team_name' key (P0-A trigger)
        msgs.append((idx, _tool_use("u0", "TeamCreate", {
            "team_name": "myteam",
            "description": "test team",
        }), 100))
        idx += 1
        # TeamCreate result (required to close the open_call for u0)
        msgs.append((idx, _tool_result("u0", "Team created successfully."), 50))
        idx += 1
        # n Agent spawns (P0-B trigger)
        for i in range(1, n_spawns + 1):
            spawn_result = (
                f"Spawned successfully.\n"
                f"agent_id: finder-p{i}@myteam\n"
                f"name: finder-p{i}\n"
                f"team_name: myteam\n"
                "The agent is now running and will receive instructions via mailbox."
            )
            msgs.append((idx, _tool_use(f"us{i}", "Agent", {
                "name": f"finder-p{i}",
                "description": f"finder {i}",
            }), 200))
            idx += 1
            msgs.append((idx, _tool_result(f"us{i}", spawn_result), 300))
            idx += 1
        # SendMessages to each (P0-C trigger: bare names).
        # Each SendMessage must have a matching tool_result so that
        # detect_in_flight.open_call stays False.  Without the result the
        # un-paired tool_use id lands in use_ids - res_ids → open_call=True →
        # safe_to_reload returns False for the WRONG reason (open tool call
        # instead of active teammate), masking the teammate-based gate under test.
        for i in range(1, n_spawns + 1):
            msgs.append((idx, _tool_use(f"um{i}", "SendMessage", {
                "to": f"finder-p{i}",
                "message": "start your task",
            }), 100))
            idx += 1
            msgs.append((idx, _tool_result(f"um{i}", "Message delivered."), 50))
            idx += 1
        return msgs

    def test_pure_sendmessage_team_no_tasks_unsafe(self):
        """REGRESSION GUARD — RED at base: pure-SendMessage team → is_empty()=True → safe.

        The exact audit scenario: finders spawned via Agent tool, coordinated
        via SendMessage only (no TaskCreate/TaskUpdate at all).
        Before fix: team_state.is_empty() → True (teammates=[], subagents=[], tasks=[])
        → safe_to_reload returns (True, 'quiescent') → silent SIGKILL of working team.
        After fix: teammates extracted with running status → blocks reload.

        This is THE critical regression guard for F1.
        """
        from cozempic.guard import safe_to_reload
        msgs = self._pure_sendmessage_msgs(n_spawns=3)
        state = _extract(msgs)
        # After fix, state must not be empty (teammates are visible)
        self.assertFalse(state.is_empty(),
                         "pure-SendMessage team state must not be empty after fix; "
                         f"got team_name={state.team_name!r}, "
                         f"teammates={len(state.teammates)}, "
                         f"subagents={len(state.subagents)}, "
                         f"tasks={len(state.tasks)}")
        safe, reason = safe_to_reload(state, msgs, Path("/tmp/fake_session.jsonl"))
        self.assertFalse(safe,
                         "Pure-SendMessage active team must block safe_to_reload; "
                         f"got safe={safe}, reason={reason!r}")

    def test_pure_sendmessage_team_after_all_idle_is_safe(self):
        """REGRESSION GUARD (partial — also GREEN at base for wrong reason):
        after ALL teammates send idle_notification → safe_to_reload returns True.

        Before fix: is_empty()=True → returns True (correct but for wrong reason;
        no protection was active). After fix: teammates exist + all are idle → True.
        This test validates the CORRECT path post-fix (all idle → benign → safe).
        """
        from cozempic.guard import safe_to_reload
        n = 3
        msgs = list(self._pure_sendmessage_msgs(n_spawns=n))
        idx = len(msgs)
        for i in range(1, n + 1):
            idle = (
                f'<teammate-message teammate_id="finder-p{i}" summary="idle">'
                f'{{"type":"idle_notification","from":"finder-p{i}",'
                f'"timestamp":"2026-06-08T10:00:00Z","idleReason":"available"}}'
                f'</teammate-message>'
            )
            msgs.append((idx, _harness_user_content(idle), 200))
            idx += 1
        state = _extract(msgs)
        safe, _ = safe_to_reload(state, msgs, Path("/tmp/fake_session.jsonl"))
        self.assertTrue(safe,
                        "After ALL idle_notifications, safe_to_reload must return True")


# ─── TestStaleConfigAntiWedge (C-1 regression guards) ───────────────────────

class TestStaleConfigAntiWedge(unittest.TestCase):
    """C-1 (CRITICAL): The now-removed P0-E _team_is_current_session gate could be
    fooled by a stale same-name config.json injecting an old leadSessionId, causing
    it to return (True,'quiescent') for a live team → SIGKILL.

    The fix (gate removal + C-1 name-join guard in merge_config_into_state) means:
    - extract_team_state still returns teammates with "running" status (from JSONL)
    - safe_to_reload blocks unconditionally on any non-benign status
    - stale config can inject at most "config" status members (benign) — never "running"

    These tests are REGRESSION GUARDs proving the fix end-to-end.
    """

    _SPAWN_RESULT = (
        "Spawned successfully.\n"
        "agent_id: finder-p1@myteam\n"
        "name: finder-p1\n"
        "team_name: myteam\n"
        "The agent is now running."
    )

    _STALE_CONFIG = {
        "name": "myteam",
        "leadSessionId": "OLD-SESSION-1111",
        "leadAgentId": "team-lead@myteam",
        "members": [],
    }

    def _stale_spawn_msgs(self):
        return [
            (0, _tool_use("u0", "TeamCreate", {"team_name": "myteam"}), 100),
            (1, _tool_result("u0", "Team created."), 50),
            (2, _tool_use("u1", "Agent", {"name": "finder-p1"}), 200),
            (3, _tool_result("u1", self._SPAWN_RESULT), 300),
        ]

    def test_stale_same_name_config_does_not_allow_false_safe_reload(self):
        """REGRESSION GUARD — RED at base: stale same-name config overwrites
        lead_session_id (now guarded by _name_only_match) → safe_to_reload
        returns (True, 'quiescent') → SIGKILL of live team.

        The live session JSONL stem ('LIVE-SESSION-2222') differs from the
        stale config's leadSessionId ('OLD-SESSION-1111'). After the C-1 fix,
        merge_config_into_state must NOT overwrite lead_session_id when the
        config was matched only by team name (not by session identity).
        """
        from cozempic.guard import safe_to_reload
        msgs = self._stale_spawn_msgs()
        live_path = Path("/tmp/LIVE-SESSION-2222.jsonl")

        with unittest.mock.patch(
            "cozempic.team.load_team_configs", return_value=[self._STALE_CONFIG]
        ):
            from cozempic.team import extract_team_state
            state = extract_team_state(msgs)

        # After fix: lead_session_id must NOT be overwritten by stale config's value;
        # safe_to_reload blocks unconditionally on "running" status regardless.
        safe, reason = safe_to_reload(state, msgs, live_path)
        self.assertFalse(
            safe,
            "Stale same-name config must NOT cause safe_to_reload to return True "
            f"for a live session with active teammates; got safe={safe}, reason={reason!r}"
        )

    def test_live_matching_config_still_blocks(self):
        """INVARIANT: a config.json that IS the live session (leadSessionId matches
        session_path.stem) must still cause safe_to_reload to block.

        This test is GREEN at base (wrong reason); after fix it must stay GREEN
        for the right reason (same session → block).
        """
        from cozempic.guard import safe_to_reload
        msgs = self._stale_spawn_msgs()
        live_path = Path("/tmp/LIVE-SESSION-2222.jsonl")
        matching_config = {
            "name": "myteam",
            "leadSessionId": "LIVE-SESSION-2222",  # matches the live path
            "leadAgentId": "team-lead@myteam",
            "members": [],
        }
        with unittest.mock.patch(
            "cozempic.team.load_team_configs", return_value=[matching_config]
        ):
            from cozempic.team import extract_team_state
            state = extract_team_state(msgs)

        safe, reason = safe_to_reload(state, msgs, live_path)
        self.assertFalse(
            safe,
            "Live-matching config must still cause safe_to_reload to block; "
            f"got safe={safe}, reason={reason!r}"
        )


# ─── TestIdleNotificationPruneProtection (C-2 regression guards) ─────────────

class TestIdleNotificationPruneProtection(unittest.TestCase):
    """C-2 (CRITICAL): idle_notification carrier message is not prune-protected
    (_is_team_message returns False for it). If pruned, the teammate stays
    'running' forever → permanent safe_to_reload wedge.

    The fix: add '<teammate-message' string pattern to _is_team_message alongside
    the existing '<task-notification' pattern.
    """

    def test_idle_notification_carrier_is_team_message(self):
        """REGRESSION GUARD — RED at base: _is_team_message returns False for a
        user message whose content is a <teammate-message> XML string.

        After fix: _is_team_message returns True → carrier is prune-protected.
        """
        from cozempic.team import _is_team_message
        idle_carrier = _harness_user_content(
            '<teammate-message teammate_id="finder-p1@myteam" summary="idle">'
            '{"type":"idle_notification","from":"finder-p1","idleReason":"available"}'
            '</teammate-message>'
        )
        self.assertTrue(
            _is_team_message(idle_carrier, set()),
            "_is_team_message must return True for an idle_notification carrier "
            "so it is prune-protected and cannot be lost to compaction"
        )


# ─── TestFailedAgentSpawnNoWedge (H-1 regression guards) ─────────────────────

class TestFailedAgentSpawnNoWedge(unittest.TestCase):
    """H-1 (HIGH): Failed Agent spawn (result has no 'agent_id:' line) leaves a
    status='running' placeholder → safe_to_reload returns False indefinitely.

    The fix: in the Agent tool_result handler, when no agent_id_m matches,
    remove or mark-terminal the placeholder so a failed spawn is not blocking.
    """

    def test_failed_agent_spawn_does_not_block_reload(self):
        """REGRESSION GUARD — RED at base: failed Agent spawn → placeholder
        status='running' → safe_to_reload returns (False, 'teammate mid-execution').

        After fix: a spawn result with no 'agent_id:' line transitions the
        placeholder to a terminal status → safe_to_reload returns (True, ...).
        """
        from cozempic.guard import safe_to_reload
        msgs = [
            (0, _tool_use("u1", "Agent", {"name": "finder-p1", "description": "find"}), 200),
            (1, _tool_result("u1", "Error: quota exceeded. No agent was created."), 300),
        ]
        state = _extract(msgs)
        # After fix: no teammates (or all terminal) → safe to reload
        if state.teammates:
            all_terminal = all(
                (t.status or "").strip().lower()
                in {"completed", "failed", "error", "stopped", "cancelled", "done"}
                for t in state.teammates
            )
            self.assertTrue(
                all_terminal,
                "Failed spawn placeholder must have a terminal status; "
                f"got teammates={[(t.agent_id, t.status) for t in state.teammates]}"
            )
        safe, reason = safe_to_reload(state, msgs, Path("/tmp/fake_session.jsonl"))
        self.assertTrue(
            safe,
            "Failed Agent spawn must not block safe_to_reload; "
            f"got safe={safe}, reason={reason!r}"
        )

    def test_successful_spawn_still_blocks(self):
        """INVARIANT: a successful spawn (has 'agent_id:' line) must still block.

        This is GREEN at base (wrong reason); must stay GREEN after H-1 fix.
        """
        from cozempic.guard import safe_to_reload
        spawn_result = (
            "Spawned successfully.\nagent_id: finder-p1@myteam\n"
            "name: finder-p1\nteam_name: myteam\nRunning."
        )
        msgs = [
            (0, _tool_use("u1", "Agent", {"name": "finder-p1"}), 200),
            (1, _tool_result("u1", spawn_result), 300),
        ]
        state = _extract(msgs)
        safe, reason = safe_to_reload(state, msgs, Path("/tmp/fake_session.jsonl"))
        self.assertFalse(
            safe,
            "Successful Agent spawn must block safe_to_reload; "
            f"got safe={safe}, reason={reason!r}"
        )


# ─── TestGateRemoval (C-1 gate-removal regression guards) ────────────────────

class TestGateRemoval(unittest.TestCase):
    """C-1 (gate removal): _team_is_current_session was a redundant gate that
    could MISFIRE: stale same-name config could inject a different leadSessionId,
    causing the gate to skip the teammate block and return (True,'quiescent') for
    a live team — SIGKILL.

    The real safety invariant is simpler: a non-benign teammate status ("running")
    can only come from THIS session's JSONL (Agent spawn / TeamCreate / SendMessage).
    Config-only members always get status="config" (∈ _TEAMMATE_BENIGN) from
    merge_config_into_state — they never block. So the teammate block must fire
    UNCONDITIONALLY on any non-benign status, regardless of lead_session_id.

    Regression guards below are RED at round-2 HEAD (where P0-E gate exists),
    GREEN after gate removal.
    """

    def test_running_teammate_blocks_unconditionally_regardless_of_session_id(self):
        """REGRESSION GUARD — RED at round-2 HEAD: P0-E gate skips block when
        lead_session_id differs from session_path.stem.

        A TeamState with a "running" teammate and ANY lead_session_id (including
        one that doesn't match the session path) must block safe_to_reload.
        After gate removal: the block is unconditional → returns (False, ...).
        """
        from cozempic.guard import safe_to_reload
        from cozempic.team import TeamState, TeammateInfo
        # Simulate a state where a stale config injected a wrong lead_session_id
        # but the JSONL gave a "running" teammate (the real current-session signal).
        state = TeamState(
            team_name="myteam",
            lead_session_id="STALE-SESSION-ID",   # does NOT match session_path.stem
            teammates=[TeammateInfo("alice@myteam", "alice", status="running")],
        )
        session_path = Path("/tmp/LIVE-SESSION-2222.jsonl")
        safe, reason = safe_to_reload(state, [], session_path)
        self.assertFalse(
            safe,
            "A 'running' teammate must block safe_to_reload unconditionally, "
            "regardless of lead_session_id vs session_path mismatch; "
            f"got safe={safe}, reason={reason!r}"
        )

    def test_config_only_member_does_not_block_reload(self):
        """INVARIANT (GREEN at both bases): config-only member (status='config')
        must NOT block safe_to_reload.

        merge_config_into_state hardcodes status='config' for members added from
        config.json (not seen in JSONL). 'config' ∈ _TEAMMATE_BENIGN → never blocks.
        This test documents the real safety mechanism.
        """
        from cozempic.guard import safe_to_reload
        from cozempic.team import TeamState, TeammateInfo
        state = TeamState(
            team_name="myteam",
            teammates=[TeammateInfo("alice@myteam", "alice", status="config")],
        )
        safe, _ = safe_to_reload(state, [], Path("/tmp/anysession.jsonl"))
        self.assertTrue(safe,
                        "config-only member (status='config') must not block reload")

    def test_stale_same_name_config_with_live_jsonl_team_must_block(self):
        """REGRESSION GUARD — RED at round-2 HEAD when C-1 name-only guard is absent
        (or even with it present, this repro uses a STRONG join to inject stale id):
        LIVE JSONL team + stale config injected leadSessionId → must still block.

        This is the lead's C-1 repro. The state has:
        - a "running" teammate from JSONL (finder-p1@myteam)
        - stale config injected leadSessionId=OLD-SESSION-1111 via a strong join
          on member-id intersection (bypasses the name-only guard if the stale
          config lists the same member IDs).
        After gate removal: block is unconditional on "running" status → safe=False.
        """
        from cozempic.guard import safe_to_reload
        from cozempic.team import TeamState, TeammateInfo
        # Construct the post-merge state directly: JSONL gave finder-p1 "running",
        # but stale config successfully injected OLD-SESSION-1111 as lead_session_id.
        state = TeamState(
            team_name="myteam",
            lead_session_id="OLD-SESSION-1111",   # injected by stale config
            teammates=[TeammateInfo("finder-p1@myteam", "finder-p1", status="running")],
        )
        live_path = Path("/tmp/LIVE-SESSION-2222.jsonl")
        safe, reason = safe_to_reload(state, [], live_path)
        self.assertFalse(
            safe,
            "C-1 repro: JSONL 'running' teammate + stale lead_session_id must still "
            f"block safe_to_reload; got safe={safe}, reason={reason!r}"
        )


# ─── TestNestedTeammateMessageRegex (M-1 regression guards) ──────────────────

class TestNestedTeammateMessageRegex(unittest.TestCase):
    """M-1: _TEAMMATE_MSG_RE with DOTALL can eat a nested <teammate-message> block
    and mis-attribute its idle_notification to the outer teammate.

    The fix: tighten the regex body group to exclude nested opening tags, e.g.
    using a negative-lookahead: ((?:(?!<teammate-message).)*?)
    so the match stops before any nested <teammate-message.
    """

    def test_nested_teammate_message_does_not_mis_attribute_idle(self):
        """REGRESSION GUARD — RED at round-2 HEAD: DOTALL greedy body matches
        nested <teammate-message>, mis-attributing the inner idle to the outer.

        Scenario: outer block for 'lead' contains a nested block for 'alice'
        (an idle_notification). The outer 'lead' must NOT be transitioned to idle.
        After fix: only 'alice' is transitioned.
        """
        from cozempic.team import extract_team_state, TeammateInfo
        # Build a message with a nested block: outer=lead (NOT idle), inner=alice (idle)
        nested_content = (
            '<teammate-message teammate_id="lead" summary="forwarded">'
            'Forwarded for logging: '
            '<teammate-message teammate_id="alice@myteam" summary="idle">'
            '{"type":"idle_notification","from":"alice","idleReason":"available"}'
            '</teammate-message>'
            '</teammate-message>'
        )
        msgs = [
            (0, _tool_use("u0", "TeamCreate", {
                "team_name": "myteam",
                "teammates": [
                    {"agentId": "alice@myteam", "name": "alice"},
                    {"agentId": "lead@myteam", "name": "lead"},
                ],
            }), 100),
            (1, _harness_user_content(nested_content), 200),
        ]
        with patch("cozempic.team.load_team_configs", return_value=[]):
            state = extract_team_state(msgs)

        lead_mate = next((t for t in state.teammates
                          if t.name == "lead" or t.agent_id == "lead@myteam"), None)
        alice_mate = next((t for t in state.teammates
                           if t.name == "alice" or t.agent_id == "alice@myteam"), None)

        from cozempic.guard import _TEAMMATE_BENIGN
        if lead_mate is not None:
            s = (lead_mate.status or "").strip().lower()
            self.assertNotIn(
                s, _TEAMMATE_BENIGN,
                "Outer 'lead' must NOT be mis-attributed idle from nested block; "
                f"got lead.status={s!r}"
            )
        if alice_mate is not None:
            s = (alice_mate.status or "").strip().lower()
            self.assertIn(
                s, _TEAMMATE_BENIGN,
                "Inner 'alice' idle_notification SHOULD transition alice to benign; "
                f"got alice.status={s!r}"
            )


if __name__ == "__main__":
    unittest.main()
