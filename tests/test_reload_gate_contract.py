"""Standing safeguard — the HARNESS-CONTRACT / deny-by-default invariant for the
safe-point reload gate.

THE BUG CLASS this catches (the one that produced the 1.8.22 Agent-marker blindness
AND PR #117's Agent-team blindness): cozempic's safety gate recognizes Claude Code
harness activity by *hardcoded, assumed* tool names / input keys / result-string
markers (`name` vs `team_name`; `agent_id:` vs `agentId:`; "Async agent launched"
vs "Spawned successfully"). When the assumed shape is wrong or drifts, the matcher
silently misses → the gate sees an empty roster → it returns "quiescent" → it
SIGKILLs live work. It is SILENT (no crash), SAFETY-CRITICAL (false-negative), and
our *synthetic* unit tests can't catch it because they encode the same assumptions.

THE INVARIANT (deny-by-default): if the raw transcript contains ANY sign of team /
agent coordination — a `TEAM_TOOL_NAMES` tool_use, OR an agent-spawn marker in a
tool_result — then `safe_to_reload` must NOT return "quiescent". Either the
extractor resolves it to a live roster (→ block), or `detect_in_flight` catches it,
or — if we couldn't parse it — we DENY rather than SIGKILL. A reload is only safe
when there is *no* unexplained coordination signal.

NOTE ON STATUS: several cases below are RED on a gate that hasn't been hardened for
Agent-spawned teams (e.g. the `team_name` case fails on 1.8.23 pre-#117; the
camelCase `agentId:` case fails even with #117's snake-only matcher). That is the
POINT — this file is the regression bar the gate must clear, and a deliberate guard
against re-shipping the class. Pair it with REAL redacted transcript fixtures
(tests/fixtures/harness/, captured from an actual agent-team session) so the
matchers are verified against reality, not against our assumptions.
"""

import json
import unittest
from pathlib import Path

from cozempic.guard import detect_in_flight, safe_to_reload
from cozempic.team import TEAM_TOOL_NAMES, extract_team_state
from cozempic.session import load_messages


def _tu(i, name, inp):
    return {"type": "assistant", "message": {"role": "assistant",
            "content": [{"type": "tool_use", "id": i, "name": name, "input": inp}]}}


def _tr(i, content):
    return {"type": "user", "message": {"role": "user",
            "content": [{"type": "tool_result", "tool_use_id": i, "content": content}]}}


def _idle_lead(text="Waiting for the team to finish."):
    return {"type": "assistant", "message": {"role": "assistant",
            "content": [{"type": "text", "text": text}]}}


def _write(tmp, rows):
    p = tmp / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


# Each case: a transcript with a LIVE team, the lead idle, ALL tool calls CLOSED,
# and NO active shared-list task — i.e. the gate would otherwise call it quiescent.
# A reload here SIGKILLs the live team, so the gate must NOT return quiescent.
_LIVE_TEAM_CASES = {
    "teamcreate_team_name_key": [
        _tu("u1", "TeamCreate", {"team_name": "squad", "teammates": [{"name": "alice"}]}),
        _tr("u1", "Team 'squad' created with 1 teammate."),
        _tu("u2", "SendMessage", {"to": "alice", "body": "start"}), _tr("u2", "delivered"),
        _idle_lead(),
    ],
    "agent_spawn_snake_id": [
        _tu("u1", "Agent", {"description": "spin alice", "subagent_type": "general-purpose"}),
        _tr("u1", "Spawned successfully. agent_id: alice@squad"),
        _tu("u2", "SendMessage", {"to": "alice", "body": "go"}), _tr("u2", "delivered"),
        _idle_lead(),
    ],
    # The camelCase smell: the SHIPPED 1.8.22 background marker is `agentId:` (see
    # _AGENT_LAUNCH_RE), so a team spawn very plausibly uses camelCase too. A
    # snake-only matcher misses it → the fix silently does nothing.
    "agent_spawn_camelCase_id": [
        _tu("u1", "Agent", {"description": "spin alice", "subagent_type": "general-purpose"}),
        _tr("u1", "Spawned successfully. agentId: alice@squad"),
        _tu("u2", "SendMessage", {"to": "alice", "body": "go"}), _tr("u2", "delivered"),
        _idle_lead(),
    ],
    "spawnteammate_tool": [
        _tu("u1", "SpawnTeammate", {"name": "alice", "role": "finder"}),
        _tr("u1", "Teammate alice spawned."),
        _idle_lead(),
    ],
    "teammate_message_active": [
        _tu("u1", "TeamCreate", {"team_name": "squad", "teammates": [{"name": "alice"}]}),
        _tr("u1", "created"),
        {"type": "user", "message": {"role": "user", "content":
            '<teammate-message teammate_id="alice" summary="progress">working on 3 of 10</teammate-message>'}},
        _idle_lead(),
    ],
}


class TestReloadGateContract(unittest.TestCase):
    """The reload gate must never call a live-team transcript 'quiescent'."""

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="cozempic_contract_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _gate(self, rows):
        p = _write(self.tmp, rows)
        msgs = load_messages(p)
        return safe_to_reload(extract_team_state(msgs), msgs, p)

    def test_live_team_must_not_be_quiescent(self):
        failures = []
        for name, rows in _LIVE_TEAM_CASES.items():
            safe, reason = self._gate(rows)
            if safe:  # quiescent → would SIGKILL the live team
                failures.append(f"{name}: safe_to_reload returned (True, {reason!r})")
        self.assertEqual(
            failures, [],
            "Reload gate called a LIVE-TEAM transcript quiescent (would SIGKILL it):\n  "
            + "\n  ".join(failures),
        )

    def test_deny_by_default_on_unparsed_coordination(self):
        """Cross-check independent of any specific marker: if the raw transcript has
        a TEAM_TOOL_NAMES tool_use but the gate is quiescent, that's an unparsed
        coordination signal → it must deny, not SIGKILL."""
        for name, rows in _LIVE_TEAM_CASES.items():
            msgs = load_messages(_write(self.tmp, rows))
            has_team_tooluse = any(
                isinstance(b, dict) and b.get("type") == "tool_use"
                and b.get("name") in TEAM_TOOL_NAMES
                for m in msgs for b in ((m[1].get("message") or {}).get("content") or [])
                if isinstance((m[1].get("message") or {}).get("content"), list)
            )
            if not has_team_tooluse:
                continue
            safe, reason = safe_to_reload(extract_team_state(msgs), msgs, None)
            self.assertFalse(
                safe,
                f"{name}: a {TEAM_TOOL_NAMES & {b.get('name') for m in msgs for b in (((m[1].get('message') or {}).get('content')) or []) if isinstance(b, dict)}} "
                f"tool_use is present but the gate is quiescent — unparsed coordination "
                f"must DENY (reason was {reason!r}).",
            )


class TestRealHarnessFixtures(unittest.TestCase):
    """Verify the gate's matchers against REAL redacted harness output, not our
    assumptions. Drop redacted `.jsonl` samples (one live team, one finished team)
    under tests/fixtures/harness/ — captured from a genuine agent-team session — and
    this asserts the gate behaves correctly on real shapes. Skips (loudly) until a
    fixture exists, so the gap stays visible instead of silently 'passing'."""

    FIX = Path(__file__).parent / "fixtures" / "harness"

    def test_live_team_fixture_defers(self):
        f = self.FIX / "live_team.jsonl"
        if not f.exists():
            self.skipTest("NO real live-team fixture yet — capture one (tests/fixtures/harness/live_team.jsonl). "
                          "Until then the team-coordination markers are UNVERIFIED.")
        msgs = load_messages(f)
        safe, reason = safe_to_reload(extract_team_state(msgs), msgs, f)
        self.assertFalse(safe, f"real live team must defer; got quiescent ({reason})")

    def test_finished_team_fixture_reloads(self):
        f = self.FIX / "finished_team.jsonl"
        if not f.exists():
            self.skipTest("NO real finished-team fixture yet — capture one (tests/fixtures/harness/finished_team.jsonl).")
        msgs = load_messages(f)
        safe, _ = safe_to_reload(extract_team_state(msgs), msgs, f)
        self.assertTrue(safe, "real finished team must be allowed to reload")

    def test_idle_team_fixture_reloads(self):
        """H1-A: genuine harness idle-notification carrier (top-level teamName) must reload.

        Corroborates the 220/220 teamName corpus claim: the fixture is a minimal
        redacted capture of a real idle-notification carrier from a live session,
        with harness metadata preserved verbatim (type, teamName, isSidechain) and
        agent IDs / report body / paths redacted.

        Expected: teammate transitions to 'idle' via the teamName-gated idle-notif
        scan (H-1 fix) → safe_to_reload returns True.  If this fixture EVER returns
        False, the H-1 gate is broken — it must pass teamName-bearing carriers through.
        """
        f = self.FIX / "idle_team.jsonl"
        if not f.exists():
            self.skipTest(
                "NO real idle-team fixture yet — capture one "
                "(tests/fixtures/harness/idle_team.jsonl). "
                "Until then the H-1 teamName carrier gate is UNVERIFIED against "
                "real harness output."
            )
        msgs = load_messages(f)
        state = extract_team_state(msgs)
        self.assertFalse(state.is_empty(),
                         "idle_team fixture must register at least one teammate "
                         "(verifies TeamCreate was parsed)")
        safe, reason = safe_to_reload(state, msgs, f)
        self.assertTrue(
            safe,
            "A real idle-notification carrier (top-level teamName) must transition "
            "the teammate to idle → allow reload. H-1 gate must pass genuine carriers "
            f"through. Got safe=False reason={reason!r}"
        )


if __name__ == "__main__":
    unittest.main()


def _user(text):
    return {"type": "user", "message": {"role": "user", "content": text}}


def _harness_user(text, team_name="squad"):
    """Genuine harness teammate-message carrier — top-level teamName (H-1 gate).

    The H-1 gate requires teamName on the message dict for idle-notification
    scans.  Use this helper whenever the test represents a real harness delivery.
    """
    return {"type": "user", "teamName": team_name,
            "message": {"role": "user", "content": text}}


class TestReloadGateHardening1824(unittest.TestCase):
    """1.8.24 hardening on top of #117 — the over-block reducers + fail-safes that
    keep FINISHED/teamless sessions reloading while LIVE/ambiguous ones block.

    The two P0s these pin (fleet, 2026-06-09): the deny-by-default net must NOT
    fire on a marker-shaped string in a Read/Grep/log result (`agent_id:` in code)
    nor on prose merely *discussing* the protocol — those over-blocked teamless
    sessions and turned the guard inert (the exact failure cozempic prevents)."""

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="cozempic_h1824_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _gate(self, rows):
        p = _write(self.tmp, rows)
        m = load_messages(p)
        return safe_to_reload(extract_team_state(m), m, p)

    def test_camelcase_agentid_parses_identity_and_blocks(self):
        # Pins the regex: the camelCase id must be PARSED to the real agent_id (a
        # snake-only matcher leaves the placeholder tool_use_id "u1"), AND the live
        # team must block.
        p = _write(self.tmp, [
            _tu("u1", "Agent", {"description": "spin"}),
            _tr("u1", "Spawned successfully. agentId: alice@squad"),
            _idle_lead()])
        m = load_messages(p)
        ts = extract_team_state(m)
        ids = [t.agent_id for t in ts.teammates]
        self.assertIn("alice@squad", ids, f"camelCase agentId must parse; got {ids}")
        self.assertFalse(safe_to_reload(ts, m, p)[0], "live camelCase team must block")

    def test_teamdelete_lets_team_reload(self):
        # A disbanded team (TeamDelete) must NOT wedge — its members go terminal.
        safe, _ = self._gate([
            _tu("u1", "Agent", {"description": "spin"}),
            _tr("u1", "Spawned successfully. agent_id: alice@squad"),
            _tu("u2", "TeamDelete", {"team_name": "squad"}),
            _tr("u2", "All agents terminated."), _idle_lead()])
        self.assertTrue(safe, "disbanded team must be allowed to reload (no wedge)")

    def test_teamdelete_scoped_leaves_other_team_blocking(self):
        # Deleting team A must NOT clear a LIVE team B (TeamDelete is team-scoped).
        safe, reason = self._gate([
            _tu("a1", "Agent", {"description": "spin alice"}),
            _tr("a1", "Spawned successfully. agent_id: alice@teamA"),
            _tu("b1", "Agent", {"description": "spin bob"}),
            _tr("b1", "Spawned successfully. agent_id: bob@teamB"),
            _tu("d1", "TeamDelete", {"team_name": "teamA"}),
            _tr("d1", "teamA disbanded"), _idle_lead()])
        self.assertFalse(safe, f"deleting teamA must leave live teamB blocking; got {reason!r}")

    def test_unparseable_successful_spawn_fails_safe(self):
        # A spawn result with no parseable id AND no failure word → keep running → block.
        safe, _ = self._gate([
            _tu("u1", "Agent", {"description": "spin"}),
            _tr("u1", "Agent ready: finder-p1@team"), _idle_lead()])
        self.assertFalse(safe, "unparseable-but-successful spawn must fail toward block")

    def test_affirmative_failed_spawn_reloads(self):
        # A spawn that affirmatively FAILED (quota/error) is terminal → reloads.
        safe, _ = self._gate([
            _tu("u1", "Agent", {"description": "spin"}),
            _tr("u1", "Error: agent quota exceeded, spawn rejected."), _idle_lead()])
        self.assertTrue(safe, "an affirmatively-failed spawn must not block forever")

    def test_net_does_not_overblock_finished_idle_team(self):
        # A teammate that sent an idle_notification (and nothing after) is finished →
        # reloads (the net must not over-block a parsed, idle roster).
        safe, _ = self._gate([
            _tu("u1", "Agent", {"description": "spin"}),
            _tr("u1", "Spawned successfully. agent_id: alice@squad"),
            _harness_user('<teammate-message teammate_id="alice@squad">{"type":"idle_notification"}</teammate-message>'),
            _idle_lead()])
        self.assertTrue(safe, "a finished (idle) team must reload, not over-block")

    def test_idle_then_reengage_blocks(self):
        # idle → then a NON-idle teammate-message = the teammate resumed → block.
        safe, reason = self._gate([
            _tu("u1", "Agent", {"description": "spin"}),
            _tr("u1", "Spawned successfully. agent_id: alice@squad"),
            _user('<teammate-message teammate_id="alice@squad" summary="idle">{"type":"idle_notification"}</teammate-message>'),
            _user('<teammate-message teammate_id="alice@squad" summary="progress">Resuming — analyzing the remaining 12 files.</teammate-message>'),
            _idle_lead()])
        self.assertFalse(safe, f"idle→re-engage must block (teammate alive); got reload: {reason!r}")

    # ── P0 regressions: the net must NOT fire on arbitrary text ──────────────
    def test_agentid_in_code_result_reloads(self):
        # `agent_id:` / `agent_id =` in a Grep/Read result is CODE, not a spawn —
        # a teamless session that read such a line must still reload (P0).
        safe, reason = self._gate([
            _tu("g1", "Grep", {"pattern": "agent_id"}),
            _tr("g1", "models.py:10:    agent_id = models.UUIDField()\n"
                      "config.yaml:3:agent_id: 5\n.env:1:AGENT_ID=xyz"),
            _idle_lead("Done searching.")])
        self.assertTrue(safe, f"teamless agent_id: in a Grep result must reload; blocked: {reason!r}")

    def test_protocol_discussion_in_prose_reloads(self):
        # A user merely DISCUSSING the protocol (no structural teammate-message tag)
        # must reload — the net keys on structure, not bare substrings (P0).
        safe, reason = self._gate([
            _user('The harness emits {"type":"idle_notification"} and a '
                  '<teammate-message> tag. Does the net over-block on these substrings?'),
            _idle_lead("Good question.")])
        self.assertTrue(safe, f"prose discussing the protocol must reload; blocked: {reason!r}")

    def test_net_correlates_spawn_marker_to_paired_tool(self):
        # Directly pin the correlation: the SAME marker text counts only when its
        # paired tool_use is a spawn tool — a Read result must not, an Agent result must.
        from cozempic.guard import _unresolved_team_coordination
        read_m = load_messages(_write(self.tmp, [
            _tu("r1", "Read", {"file_path": "x.py"}), _tr("r1", "agent_id: alice@squad")]))
        self.assertFalse(_unresolved_team_coordination(read_m, None),
                         "marker in a Read result must NOT count as coordination")
        agent_m = load_messages(_write(self.tmp, [
            _tu("a1", "Agent", {"description": "x"}), _tr("a1", "agent_id: alice@squad")]))
        self.assertTrue(_unresolved_team_coordination(agent_m, None),
                        "marker in an Agent-tool result MUST count as coordination")

    def test_pasted_teammate_message_reloads(self):
        # A user PASTING a structural teammate-message into a teamless session (no
        # spawn anywhere) must reload — the net keys on the harness delivery surface,
        # not a typed message.content, so a paste can't wedge the guard (P1).
        safe, reason = self._gate([
            _user('Saw this in a log — what does it mean? '
                  '<teammate-message teammate_id="alice@squad" summary="progress">working on 3 of 10</teammate-message>'),
            _idle_lead("It's a teammate progress update.")])
        self.assertTrue(safe, f"a pasted teammate-message must reload; blocked: {reason!r}")

    def test_teamdelete_bare_ids_multi_team_blocks_live_team(self):
        # Two teams with BARE inline ids (no @team suffix); deleting team A must NOT
        # clear-all and SIGKILL the live team B — unattributable members stay live (P2b).
        safe, reason = self._gate([
            _tu("c1", "TeamCreate", {"team_name": "teamA", "teammates": [{"agentId": "alice"}]}),
            _tr("c1", "Team teamA created."),
            _tu("c2", "TeamCreate", {"team_name": "teamB", "teammates": [{"agentId": "carol"}]}),
            _tr("c2", "Team teamB created."),
            _tu("d1", "TeamDelete", {"team_name": "teamA"}), _tr("d1", "teamA disbanded."),
            _idle_lead()])
        self.assertFalse(safe, f"deleting teamA must not SIGKILL live teamB; got reload: {reason!r}")


# Real harness marker formats, captured + verified from a live session 2026-06-09.
# A FOREGROUND Agent that returned its result inline carries the duration_ms usage
# trailer; a BACKGROUND Agent launch / live team-spawn does not.
_FG_DONE = ("[analysis output]\n\n**Net:** complete.\n"
            "agentId: aca896c66561b5001 (use SendMessage with to: 'aca896c66561b5001' "
            "to continue this agent)\n<usage>subagent_tokens: 94765\ntool_uses: 34\n"
            "duration_ms: 445383</usage>")
_BG_LAUNCH = ("Async agent launched successfully.\nagentId: a9a0ae189567a510d (internal "
              "ID - do not mention to user. Use SendMessage with to: 'a9a0ae189567a510d' "
              "to continue this agent.)\nThe agent is working in the background.")


class TestForegroundAgentCompletion1825(unittest.TestCase):
    """1.8.25 — a COMPLETED foreground Agent must not be read as a live teammate
    (real-transcript over-block: the synthetic team-spawn format hid it), and a
    launch marker QUOTED in tool output must not fabricate a phantom in-flight task."""

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="cozempic_fg1825_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _state(self, rows):
        p = _write(self.tmp, rows)
        m = load_messages(p)
        return extract_team_state(m), m, p

    def test_completed_foreground_agent_reloads(self):
        ts, m, p = self._state([
            _tu("t1", "Agent", {"description": "qa", "subagent_type": "general-purpose", "prompt": "x"}),
            _tr("t1", _FG_DONE), _idle_lead("done")])
        self.assertEqual([t.status for t in ts.teammates], ["completed"],
                         "a foreground Agent with the duration_ms trailer must be terminal")
        self.assertTrue(safe_to_reload(ts, m, p)[0], "completed foreground Agent must reload")

    def test_background_launch_blocks(self):
        ts, m, p = self._state([
            _tu("t1", "Agent", {"description": "qa", "subagent_type": "general-purpose", "prompt": "x"}),
            _tr("t1", _BG_LAUNCH), _idle_lead("launched")])
        self.assertFalse(safe_to_reload(ts, m, p)[0], "an in-flight background Agent must block")

    def test_team_spawn_without_trailer_still_blocks(self):
        # The live team-spawn ack (no duration_ms) must remain non-terminal.
        ts, m, p = self._state([
            _tu("t1", "Agent", {"name": "alice"}),
            _tr("t1", "Spawned successfully.\nagent_id: alice@squad"), _idle_lead()])
        self.assertFalse(safe_to_reload(ts, m, p)[0], "a live team-spawn must still block")

    def test_agent_output_quoting_launch_not_credited(self):
        # A COMPLETED agent whose OUTPUT quotes a launch marker must not be read as
        # an in-flight launch (detect_in_flight skips foreground-completed results).
        inflight = detect_in_flight(load_messages(_write(self.tmp, [
            _tu("t1", "Agent", {"description": "review", "subagent_type": "general-purpose", "prompt": "x"}),
            _tr("t1", "I checked _AGENT_LAUNCH_RE against 'Async agent launched "
                      "successfully. agentId: phantom123'.\n" + _FG_DONE)])))
        self.assertNotIn("phantom123", inflight["ids"],
                         "a launch marker quoted in a completed agent's output is not a launch")
        self.assertFalse(inflight["agent"])

    def test_normal_bash_echoing_bg_ack_not_credited(self):
        # A normal Bash (no run_in_background) whose OUTPUT echoes the bg-ack text
        # must not be credited as a launch.
        inflight = detect_in_flight(load_messages(_write(self.tmp, [
            _tu("b1", "Bash", {"command": "grep -r 'in background with ID'"}),
            _tr("b1", "test_x.py:5: running in background with ID: notreal99")])))
        self.assertFalse(inflight["background"], "a Bash output echoing the ack text is not a launch")

    def test_real_run_in_background_bash_still_blocks(self):
        # A genuine run_in_background Bash launch must still be detected.
        inflight = detect_in_flight(load_messages(_write(self.tmp, [
            _tu("b1", "Bash", {"command": "sleep 99", "run_in_background": True}),
            _tr("b1", "Command running in background with ID: realbg7")])))
        self.assertTrue(inflight["background"], "a real run_in_background Bash must be detected")
        self.assertIn("realbg7", inflight["ids"])


# Real task-notification format, ground-truthed 2026-06-09: <task-id> is followed by
# <tool-use-id> and <output-file> tags BEFORE <status> — a strictly-ordered parser
# misses it and leaves a COMPLETED background-Agent teammate "running".
def _real_task_notif(tid, status="completed"):
    return (f"<task-notification>\n<task-id>{tid}</task-id>\n"
            f"<tool-use-id>toolu_01abc</tool-use-id>\n<output-file>/tmp/{tid}.output</output-file>\n"
            f"<status>{status}</status>\n<summary>done</summary>\n<result>findings</result>\n</task-notification>")


def _qop(text):
    return {"type": "queue-operation", "content": text}


class TestTaskNotificationRealFormat1825(unittest.TestCase):
    """1.8.25 — the second-pass task-notification parser must tolerate the REAL
    format (extra tags between task-id and status), resolve a BARE-name completion
    to a suffixed agentId, and count terminal task states as inactive."""

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="cozempic_tn1825_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _gate(self, rows):
        p = _write(self.tmp, rows)
        m = load_messages(p)
        return safe_to_reload(extract_team_state(m), m, p)

    def test_real_task_notification_format_clears_teammate(self):
        # Background Agent + REAL completion notification (tool-use-id/output-file tags).
        safe, reason = self._gate([
            _tu("t1", "Agent", {"description": "a", "subagent_type": "general-purpose", "prompt": "x"}),
            _tr("t1", "Async agent launched successfully.\nagentId: bg777 (Use SendMessage with to: 'bg777' to continue this agent.)"),
            _qop(_real_task_notif("bg777")), _idle_lead()])
        self.assertTrue(safe, f"real-format completion must clear the teammate; got {reason!r}")

    def test_bare_name_completion_resolves_to_suffixed_id(self):
        # Spawn worker7@myteam; a BARE-name "worker7" completion must resolve + clear.
        safe, reason = self._gate([
            _tu("t1", "Agent", {"name": "researcher"}),
            _tr("t1", "Spawned successfully.\nagent_id: worker7@myteam"),
            _qop(_real_task_notif("worker7")), _idle_lead()])
        self.assertTrue(safe, f"bare-name completion must resolve to worker7@myteam; got {reason!r}")

    def test_reordered_attributed_notification_clears(self):
        # Order-independent + tag-attribute tolerant.
        tn = ("<task-notification id='n1'>\n<summary>s</summary>\n<task-id>z9</task-id>\n"
              "<status>completed</status>\n</task-notification>")
        safe, _ = self._gate([
            _tu("t1", "Agent", {"name": "z9"}),
            _tr("t1", "Spawned successfully.\nagent_id: z9"), _qop(tn), _idle_lead()])
        self.assertTrue(safe, "a reordered/attributed notification must still clear")

    def test_incomplete_notification_still_blocks(self):
        # A bg Agent with NO completion notification must still block.
        safe, _ = self._gate([
            _tu("t1", "Agent", {"description": "a", "subagent_type": "general-purpose", "prompt": "x"}),
            _tr("t1", "Async agent launched successfully.\nagentId: bg888 (Use SendMessage with to: 'bg888' to continue this agent.)"),
            _idle_lead()])
        self.assertFalse(safe, "an uncompleted background Agent must still block")

    def test_terminal_task_statuses_are_inactive(self):
        from cozempic.team import TeamState, TaskInfo
        for st in ("closed", "resolved", "finished", "merged", "skipped", "completed", "done"):
            ts = TeamState()
            ts.tasks = [TaskInfo(task_id="T1", subject="do x", status=st)]
            self.assertEqual(ts._task_groups()[0], [], f"task status {st!r} must be inactive")
        ts = TeamState()
        ts.tasks = [TaskInfo(task_id="T2", subject="do y", status="in_progress")]
        self.assertEqual(len(ts._task_groups()[0]), 1, "in_progress task must stay active")
