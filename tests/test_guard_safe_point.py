"""1.8.22 — safe-point reload gate (validate-before-terminate).

A guard reload SIGKILLs the Claude process and resumes from the pruned transcript.
Harness-side in-flight work — a running Workflow-tool orchestration, a background
subagent, an open tool call, or an active agent team — is NOT in the transcript and
is destroyed with no recovery. These tests prove the gate REFUSES to reload through
in-flight work at ALL tiers (incl. HARD2), instead of force-terminating.

T2 (mid-subagent), T3 (mid-Workflow), T7 (pre-wall+in-flight) are the load-bearing
blockers. T3 documents the pre-1.8.22 catastrophe: on old code HARD2 would SIGKILL a
running workflow; on 1.8.22 it defers.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


def _make_session_file(tmpdir: Path, size_bytes: int = 100_000) -> Path:
    path = tmpdir / "fake_session.jsonl"
    line = '{"type":"user","message":{"content":"' + "x" * 100 + '"}}\n'
    n = max(1, size_bytes // len(line.encode()))
    path.write_text(line * n)
    return path


def _m(d):
    return (0, d, 100)


class TestDetectInFlight(unittest.TestCase):
    def test_workflow_launched_no_completion(self):
        from cozempic.guard import detect_in_flight
        msgs = [_m({"message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": "Workflow launched in background. Task ID: wul02ab83\nRun ID: wf_x"}]}})]
        self.assertTrue(detect_in_flight(msgs)["workflow"])

    def test_workflow_completed(self):
        from cozempic.guard import detect_in_flight
        msgs = [
            _m({"message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1",
                 "content": "Workflow launched in background. Task ID: wul02ab83"}]}}),
            _m({"type": "queue-operation",
                "content": "<task-notification><task-id>wul02ab83</task-id><status>completed</status>"
                           "<summary>s</summary><result>r</result></task-notification>"}),
        ]
        self.assertFalse(detect_in_flight(msgs)["workflow"])

    def test_background_command_in_flight(self):
        from cozempic.guard import detect_in_flight
        msgs = [_m({"message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t2",
             "content": "Command running in background with ID: bx1t5ptm2"}]}})]
        self.assertTrue(detect_in_flight(msgs)["background"])

    def test_agent_launched_no_completion(self):
        # The REAL dominant background-subagent path (Agent tool). VERIFIED marker
        # from production transcripts; the old detector was blind to this → P0.
        from cozempic.guard import detect_in_flight
        msgs = [_m({"message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t3",
             "content": "Async agent launched successfully.\nagentId: aa3bddd46882df8ea "
                        "(internal ID - do not mention to user. Use SendMessage with to: "
                        "'aa3bddd46882df8ea' to continue this agent.)\nThe agent is working "
                        "in the background. You will be notified automatically when it completes."}]}})]
        d = detect_in_flight(msgs)
        self.assertTrue(d["agent"], "background Agent subagent must be detected in-flight")
        self.assertIn("aa3bddd46882df8ea", d["ids"])

    def test_agent_completed_via_task_notification(self):
        # Same-namespace completion (verified real): agentId == <task-id>.
        from cozempic.guard import detect_in_flight
        msgs = [
            _m({"message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t3",
                 "content": "Async agent launched successfully.\nagentId: aa3bddd46882df8ea (internal ID)"}]}}),
            _m({"type": "queue-operation",
                "content": "<task-notification><task-id>aa3bddd46882df8ea</task-id>"
                           "<status>completed</status><summary>s</summary></task-notification>"}),
        ]
        self.assertFalse(detect_in_flight(msgs)["agent"], "completed Agent must clear")

    def test_completion_vocab_success_clears(self):
        # A "success"/"done" status (not literally "completed") must still clear.
        from cozempic.guard import detect_in_flight
        msgs = [
            _m({"message": {"role": "user", "content": [
                {"type": "tool_result", "content": "Async agent launched successfully. agentId: zz9"}]}}),
            _m({"content": "<task-notification><task-id>zz9</task-id><status>success</status></task-notification>"}),
        ]
        self.assertFalse(detect_in_flight(msgs)["agent"])

    def test_unkeyed_tool_use_is_open(self):
        from cozempic.guard import detect_in_flight
        msgs = [_m({"message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Bash", "input": {}}]}})]  # no id
        self.assertTrue(detect_in_flight(msgs)["open_call"])

    def test_workflow_marker_quoted_in_bash_result_is_not_inflight(self):
        # Type-correlation (validated against a live workflow run): a Workflow
        # launch-marker STRING echoed/grepped inside a Bash tool_result is NOT a
        # real launch — its paired tool_use is Bash, not Workflow — so it must not
        # fabricate a phantom in-flight workflow that wedges the gate.
        from cozempic.guard import detect_in_flight
        msgs = [
            _m({"message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "b1", "name": "Bash", "input": {}}]}}),
            _m({"message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "b1",
                 "content": "Workflow launched in background. Task ID: quoted_xyz"}]}}),
        ]
        self.assertFalse(detect_in_flight(msgs)["workflow"], "quoted marker must not count")

    def test_real_workflow_result_is_inflight(self):
        # The paired tool_use IS a Workflow → the marker is a real launch.
        from cozempic.guard import detect_in_flight
        msgs = [
            _m({"message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "w1", "name": "Workflow", "input": {}}]}}),
            _m({"message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "w1",
                 "content": "Workflow launched in background. Task ID: w03jq5cm6\nRun ID: wf_x"}]}}),
        ]
        d = detect_in_flight(msgs)
        self.assertTrue(d["workflow"])
        self.assertIn("w03jq5cm6", d["ids"])

    def test_pruned_tooluse_launch_credited_conservatively(self):
        # If the launch's tool_use was pruned away (only the result remains), credit
        # it — never MISS a real launch (the catastrophic direction).
        from cozempic.guard import detect_in_flight
        msgs = [_m({"message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "gone",
             "content": "Workflow launched in background. Task ID: orphan1"}]}})]
        self.assertTrue(detect_in_flight(msgs)["workflow"])

    def test_quoted_completion_in_tool_result_does_not_clear_real_launch(self):
        # P2 (QA): a <task-notification>…completed QUOTED inside a Bash result must
        # NOT clear a genuinely in-flight Agent of the same id (false-negative →
        # SIGKILL). Real completions arrive as top-level queue-operation content.
        from cozempic.guard import detect_in_flight
        msgs = [
            _m({"message": {"role": "user", "content": [
                {"type": "tool_result",
                 "content": "Async agent launched successfully. agentId: live9 (internal ID)"}]}}),
            _m({"message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "b1", "name": "Bash", "input": {}}]}}),
            _m({"message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "b1",
                 "content": "grep hit: <task-notification><task-id>live9</task-id>"
                            "<status>completed</status></task-notification>"}]}}),
        ]
        self.assertTrue(detect_in_flight(msgs)["agent"],
                        "a quoted completion must NOT clear a real in-flight launch")

    def test_real_queue_completion_clears(self):
        from cozempic.guard import detect_in_flight
        msgs = [
            _m({"message": {"role": "user", "content": [
                {"type": "tool_result",
                 "content": "Async agent launched successfully. agentId: live9 (internal ID)"}]}}),
            _m({"type": "queue-operation",
                "content": "<task-notification><task-id>live9</task-id>"
                           "<status>completed</status></task-notification>"}),
        ]
        self.assertFalse(detect_in_flight(msgs)["agent"], "genuine queue completion clears")

    def test_lowercase_tool_name_still_credited(self):
        # P2 (QA): case-insensitive tool-name correlation — a casing drift
        # ("workflow" vs "Workflow") must NOT strand a real launch into a SIGKILL.
        from cozempic.guard import detect_in_flight
        msgs = [
            _m({"message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "w1", "name": "workflow", "input": {}}]}}),
            _m({"message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "w1",
                 "content": "Workflow launched in background. Task ID: lc1"}]}}),
        ]
        self.assertTrue(detect_in_flight(msgs)["workflow"])

    def test_open_tool_call(self):
        from cozempic.guard import detect_in_flight
        msgs = [_m({"message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "u1", "name": "Bash", "input": {}}]}})]
        self.assertTrue(detect_in_flight(msgs)["open_call"])

    def test_matched_tool_call_not_open(self):
        from cozempic.guard import detect_in_flight
        msgs = [
            _m({"message": {"role": "assistant", "content": [{"type": "tool_use", "id": "u1", "name": "Bash", "input": {}}]}}),
            _m({"message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "u1", "content": "done"}]}}),
        ]
        self.assertFalse(detect_in_flight(msgs)["open_call"])


class TestSafeToReload(unittest.TestCase):
    def test_empty_quiescent_is_safe(self):
        from cozempic.guard import safe_to_reload
        from cozempic.team import TeamState
        safe, _ = safe_to_reload(TeamState(), [], None)
        self.assertTrue(safe)

    def test_running_workflow_unsafe(self):
        from cozempic.guard import safe_to_reload
        from cozempic.team import TeamState
        msgs = [_m({"message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": "Workflow launched in background. Task ID: wf123"}]}})]
        safe, reason = safe_to_reload(TeamState(), msgs, None)
        self.assertFalse(safe)
        self.assertIn("Workflow", reason)

    def test_running_subagent_unsafe(self):
        from cozempic.guard import safe_to_reload
        from cozempic.team import TeamState, SubagentInfo
        st = TeamState(team_name="t", subagents=[SubagentInfo("a1", "d", status="running")])
        safe, _ = safe_to_reload(st, [], None)
        self.assertFalse(safe)

    def test_active_task_unsafe(self):
        from cozempic.guard import safe_to_reload
        from cozempic.team import TeamState, TaskInfo
        st = TeamState(team_name="t", tasks=[TaskInfo("t1", "subj", "in_progress")])
        safe, _ = safe_to_reload(st, [], None)
        self.assertFalse(safe)

    def test_quiesced_team_safe(self):
        from cozempic.guard import safe_to_reload
        from cozempic.team import TeamState, SubagentInfo, TeammateInfo
        st = TeamState(team_name="t",
                       subagents=[SubagentInfo("a1", "d", status="completed")],
                       teammates=[TeammateInfo("a2", "n", status="done")])
        safe, _ = safe_to_reload(st, [], None)
        self.assertTrue(safe)

    def test_running_agent_unsafe(self):
        from cozempic.guard import safe_to_reload
        from cozempic.team import TeamState
        msgs = [_m({"message": {"role": "user", "content": [
            {"type": "tool_result",
             "content": "Async agent launched successfully. agentId: bg777 (internal ID)"}]}})]
        safe, reason = safe_to_reload(TeamState(), msgs, None)
        self.assertFalse(safe)
        self.assertIn("subagent", reason)

    def test_finished_team_is_safe(self):
        # P0-2 regression: a FINISHED team must be allowed to reload (the guard's
        # whole purpose for teams). After extract_team_state propagates completions,
        # teammates read terminal — so the gate is SAFE.
        from cozempic.guard import safe_to_reload
        from cozempic.team import TeamState, SubagentInfo, TeammateInfo
        st = TeamState(team_name="t",
                       subagents=[SubagentInfo("a1", "d", status="completed"),
                                  SubagentInfo("a2", "d", status="completed")],
                       teammates=[TeammateInfo("a1", "alice", status="completed"),
                                  TeammateInfo("a2", "bob", status="completed")])
        safe, _ = safe_to_reload(st, [], None)
        self.assertTrue(safe, "finished team must be allowed to reload")

    def test_active_teammate_unsafe(self):
        # The destroy-active-teammate window: a teammate working in the
        # SendMessage→completion gap has no subagent entry yet, only a 'running'
        # teammate entry. The gate must block on it (don't SIGKILL active work).
        from cozempic.guard import safe_to_reload
        from cozempic.team import TeamState, TeammateInfo
        st = TeamState(team_name="t", teammates=[TeammateInfo("a1", "alice", status="running")])
        safe, reason = safe_to_reload(st, [], None)
        self.assertFalse(safe)
        self.assertIn("teammate", reason)

    def test_offvocab_subagent_status_blocks(self):
        # DENYLIST: an unrecognized non-terminal status (hyphen variant, "busy",
        # "executing"...) must fail SAFE (block), not fail-open and destroy work.
        from cozempic.guard import safe_to_reload
        from cozempic.team import TeamState, SubagentInfo
        for st_val in ("in-progress", "busy", "executing", "processing", "waiting", "weird"):
            ts = TeamState(team_name="t", subagents=[SubagentInfo("a1", "d", status=st_val)])
            safe, _ = safe_to_reload(ts, [], None)
            self.assertFalse(safe, f"off-vocab subagent status {st_val!r} must block")

    def test_reactivated_teammate_after_completion_unsafe(self):
        # P0 ordering: a completion followed by a NEW SendMessage (re-activation)
        # must leave the teammate active, not "completed" — else a working teammate
        # is destroyed. Drives extract_team_state end-to-end (two-pass).
        from cozempic.team import extract_team_state
        from cozempic.guard import safe_to_reload
        msgs = [
            (0, {"message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "u1", "name": "TeamCreate",
                 "input": {"team_name": "t", "teammates": [{"agent_id": "alice", "name": "Alice"}]}}]}}, 100),
            (1, {"message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "u2", "name": "SendMessage", "input": {"to": "alice", "body": "phase1"}}]}}, 100),
            (2, {"type": "queue-operation",
                 "content": "<task-notification><task-id>alice</task-id><status>completed</status>"
                            "<summary>p1 done</summary><result>r</result></task-notification>"}, 100),
            (3, {"message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "u3", "name": "SendMessage", "input": {"to": "alice", "body": "phase2"}}]}}, 100),
        ]
        st = extract_team_state(msgs)
        mate = next(t for t in st.teammates if t.agent_id == "alice")
        self.assertEqual(mate.status, "running", "re-activated teammate must stay active")
        safe, reason = safe_to_reload(st, [], None)
        self.assertFalse(safe, "must NOT reload through a re-activated working teammate")

    def test_extract_propagates_completion_to_teammate(self):
        # The root-cause fix: a task-notification completion must transition the
        # matching TEAMMATE (not just the subagent) to terminal.
        from cozempic.team import extract_team_state
        msgs = [
            (0, {"message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "u1", "name": "TeamCreate",
                 "input": {"team_name": "t", "teammates": [{"agent_id": "alice", "name": "Alice"}]}}]}}, 100),
            (0, {"type": "queue-operation",
                 "content": "<task-notification><task-id>alice</task-id><status>completed</status>"
                            "<summary>done</summary><result>r</result></task-notification>"}, 100),
        ]
        st = extract_team_state(msgs)
        mate = next((t for t in st.teammates if t.agent_id == "alice"), None)
        self.assertIsNotNone(mate)
        self.assertEqual(mate.status, "completed", "completion must propagate to teammate")


class TestGuardCycleGate(unittest.TestCase):
    """Integration: guard_prune_cycle must DEFER (not terminate) when unsafe."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cozempic_sp_"))
        self.session_path = _make_session_file(self.tmpdir, 100_000)
        self.scratch = Path(tempfile.mkdtemp(prefix="cozempic_tmproot_"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        shutil.rmtree(self.scratch, ignore_errors=True)

    def _run(self, team_state, load_msgs):
        from cozempic.guard import guard_prune_cycle
        pruned = [(0, {"type": "user"}, 40_000)]
        terminate_called = []
        _totals = iter([100_000, 40_000])  # real token progress → reaches reload block

        def _est(*a, **k):
            try:
                return MagicMock(total=next(_totals))
            except StopIteration:
                return MagicMock(total=40_000)

        with patch("cozempic.guard._guard_tmp_root", return_value=self.scratch), \
             patch("cozempic.guard.load_messages_and_snapshot", return_value=(load_msgs, MagicMock())), \
             patch("cozempic.guard.load_messages", return_value=load_msgs), \
             patch("cozempic.guard.prune_with_team_protect", return_value=(pruned, {}, team_state)), \
             patch("cozempic.guard.save_messages", side_effect=lambda *a, **k: None), \
             patch("cozempic.guard.snapshot_session", return_value=MagicMock()), \
             patch("cozempic.guard._terminate_and_resume",
                   side_effect=lambda *a, **k: terminate_called.append(True)), \
             patch("cozempic.tokens.estimate_session_tokens", side_effect=_est), \
             patch("cozempic.tokens.calibrate_ratio", return_value=0.5):
            result = guard_prune_cycle(
                session_path=self.session_path, rx_name="aggressive", config=None,
                auto_reload=True, claude_pid=999999, session_id="abcdef012345",
            )
        return result, terminate_called

    def test_T1_quiescent_team_reloads(self):
        from cozempic.team import TeamState, SubagentInfo
        st = TeamState(team_name="t", subagents=[SubagentInfo("a1", "d", status="completed")])
        result, terminate_called = self._run(st, load_msgs=[(0, {"type": "user"}, 100_000)])
        self.assertFalse(result.get("reload_unsafe"))
        self.assertTrue(terminate_called, "quiescent team must reload (terminate called)")

    def test_T2_mid_subagent_defers_not_kills(self):
        from cozempic.team import TeamState, SubagentInfo
        st = TeamState(team_name="t", subagents=[SubagentInfo("a1", "d", status="running")])
        result, terminate_called = self._run(st, load_msgs=[(0, {"type": "user"}, 100_000)])
        self.assertTrue(result.get("reload_unsafe"))
        self.assertEqual(result.get("unsafe_reason"), "subagent mid-execution")
        self.assertEqual(terminate_called, [], "mid-subagent: must NOT terminate Claude")

    def test_T3_mid_workflow_defers_not_kills(self):
        # The catastrophic baseline: empty team (agents_active False) BUT a workflow
        # is running. Old code HARD2 would SIGKILL it; 1.8.22 must defer.
        from cozempic.team import TeamState
        wf_msgs = [(0, {"message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": "Workflow launched in background. Task ID: wf999"}]}}, 100_000)]
        result, terminate_called = self._run(TeamState(), load_msgs=wf_msgs)
        self.assertTrue(result.get("reload_unsafe"))
        self.assertEqual(result.get("unsafe_reason"), "Workflow orchestrating")
        self.assertEqual(terminate_called, [], "mid-Workflow: must NOT terminate (T3 blocker)")

    def test_T7_open_call_defers(self):
        from cozempic.team import TeamState
        oc = [(0, {"message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "u1", "name": "Bash", "input": {}}]}}, 100_000)]
        result, terminate_called = self._run(TeamState(), load_msgs=oc)
        self.assertTrue(result.get("reload_unsafe"))
        self.assertEqual(terminate_called, [], "open tool call: must NOT terminate")

    def test_quiescent_empty_reloads(self):
        from cozempic.team import TeamState
        result, terminate_called = self._run(TeamState(), load_msgs=[(0, {"type": "user"}, 100_000)])
        self.assertFalse(result.get("reload_unsafe"))
        self.assertTrue(terminate_called)


if __name__ == "__main__":
    unittest.main()
