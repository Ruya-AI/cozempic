"""RED tests for post-prune structural validation + replay-readiness (P0-B / P0-C).

Per PLAN.md § 2.3 (validation) and § 2.4 (floor preservation):

Validation checks performed by `cozempic.safety.validate_post_prune`:
  C1. parentUuid resolution: every msgs_after entry with a non-null parentUuid
      MUST point to a uuid in msgs_after (defensive; _relink_parent_chain
      already aims to guarantee this).
  C2. Root preserved: if msgs_before had a parentUuid=null root, msgs_after
      MUST also have at least one parentUuid=null entry.
  C3. Conversation survival: ≥1 user AND ≥1 assistant survives.
  C4. compact_boundary marker preserved: if msgs_before had a system/subtype=
      compact_boundary entry, msgs_after MUST contain the LAST such entry.
  C5. Last permission-mode preserved.
  C6. Last last-prompt preserved.

Floor preservation (P0-C) is exercised via `enforce_floor`:
  - The last K user + K assistant turns must survive any prune.
  - The first (parentUuid=null) message must survive.
  - The user/assistant drop percentage cannot exceed `max_user_assistant_drop_pct`.

A separate replay-readiness probe simulates Claude Code's resume bootstrap by
walking the surviving parentUuid graph forward from the root: every chain link
must resolve to a surviving uuid.

All tests SHOULD FAIL until P0-B / P0-C are implemented.
"""

from __future__ import annotations

import unittest
from typing import Any

from cozempic.helpers import msg_bytes


try:
    from cozempic import safety as _safety  # type: ignore  # noqa: F401
    _SAFETY_AVAILABLE = True
except ImportError:
    _SAFETY_AVAILABLE = False


# ─── Fixture helpers (no fixtures dir needed — keep tests self-contained) ────


def _msg(line_idx: int, payload: dict) -> tuple[int, dict, int]:
    return (line_idx, payload, msg_bytes(payload))


def _root_user(idx: int, uuid: str) -> tuple[int, dict, int]:
    return _msg(idx, {
        "type": "user",
        "uuid": uuid,
        "parentUuid": None,
        "message": {"role": "user", "content": f"root prompt {uuid}"},
    })


def _user(idx: int, uuid: str, parent: str, text: str = "u") -> tuple[int, dict, int]:
    return _msg(idx, {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "message": {"role": "user", "content": text},
    })


def _assistant(idx: int, uuid: str, parent: str, text: str = "a") -> tuple[int, dict, int]:
    return _msg(idx, {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    })


def _ai_title(idx: int, uuid: str, parent: str, title: str) -> tuple[int, dict, int]:
    return _msg(idx, {"type": "ai-title", "uuid": uuid, "parentUuid": parent, "title": title})


def _last_prompt(idx: int, uuid: str, parent: str, text: str) -> tuple[int, dict, int]:
    return _msg(idx, {"type": "last-prompt", "uuid": uuid, "parentUuid": parent, "text": text})


def _permission_mode(idx: int, uuid: str, parent: str, mode: str) -> tuple[int, dict, int]:
    return _msg(idx, {"type": "permission-mode", "uuid": uuid, "parentUuid": parent, "mode": mode})


def _compact_boundary(idx: int, uuid: str, parent: str) -> tuple[int, dict, int]:
    return _msg(idx, {
        "type": "system", "subtype": "compact_boundary",
        "uuid": uuid, "parentUuid": parent,
    })


def _build_clean_session(
    n_turns: int = 10,
    include_compact_boundary: bool = False,
    include_metadata: bool = True,
) -> list[tuple[int, dict, int]]:
    """Build a structurally-valid session with `n_turns` user+assistant pairs."""
    msgs: list[tuple[int, dict, int]] = []
    line = 0

    # Root user
    msgs.append(_root_user(line, "u0"))
    line += 1
    last_uuid = "u0"

    # Pair turns
    for i in range(1, n_turns):
        a_uuid = f"a{i}"
        msgs.append(_assistant(line, a_uuid, last_uuid, f"reply {i}"))
        line += 1
        u_uuid = f"u{i}"
        msgs.append(_user(line, u_uuid, a_uuid, f"prompt {i}"))
        line += 1
        last_uuid = u_uuid
    # Final assistant for closing pair
    msgs.append(_assistant(line, f"a{n_turns}", last_uuid, "final reply"))
    line += 1

    if include_compact_boundary:
        msgs.append(_compact_boundary(line, "cb1", f"a{n_turns}"))
        line += 1

    if include_metadata:
        msgs.append(_ai_title(line, "t1", f"a{n_turns}", "Session title"))
        line += 1
        msgs.append(_last_prompt(line, "lp1", f"a{n_turns}", "last prompt"))
        line += 1
        msgs.append(_permission_mode(line, "pm1", f"a{n_turns}", "default"))
        line += 1

    return msgs


# ─── C1: parent chain resolution ─────────────────────────────────────────────


class TestParentChainResolution(unittest.TestCase):
    """C1 — every parentUuid in msgs_after must resolve to a surviving uuid or null."""

    def test_passes_when_chain_is_intact(self):
        from cozempic.safety import validate_post_prune  # type: ignore

        before = _build_clean_session(n_turns=5)
        # Identity prune (no removals): must pass all checks.
        validate_post_prune(before, before)

    def test_fails_when_parentuuid_dangles(self):
        """Synthetic msgs_after with parentUuid pointing to a removed uuid → C1 fails."""
        from cozempic.safety import PruneValidationError, validate_post_prune  # type: ignore

        before = _build_clean_session(n_turns=3)
        # Drop the intermediate assistant "a1" from after; but leave "u1" whose
        # parentUuid is still "a1" → dangling reference.
        after = [m for m in before if m[1].get("uuid") != "a1"]

        with self.assertRaises(PruneValidationError) as ctx:
            validate_post_prune(before, after)
        self.assertEqual(ctx.exception.evidence.get("failed_check"), "C1")

    def test_post_pruned_file_has_zero_broken_parentuuid_after_relink(self):
        """End-to-end: gentle prescription on a session with compact_boundary
        produces a structurally-resumable file.

        Currently RED because gentle's `compact-summary-collapse` drops the
        ROOT (parentUuid=null) entry — only P0-C floor enforcement re-adds it.
        The replay simulation needs the safety module to exist (P0-B), so the
        assertion raises ImportError on current code.
        """
        from cozempic.executor import run_prescription
        from cozempic.registry import PRESCRIPTIONS
        from cozempic.safety import simulate_replay_readiness  # type: ignore
        import cozempic.strategies  # noqa: F401 — register strategies

        before = _build_clean_session(n_turns=20, include_compact_boundary=True)
        after, _ = run_prescription(before, PRESCRIPTIONS["gentle"], {})

        # Chain integrity (already provided by _relink_parent_chain).
        surviving_uuids = {m[1].get("uuid") for m in after if m[1].get("uuid")}
        for _, msg, _ in after:
            parent = msg.get("parentUuid")
            if parent is None:
                continue
            self.assertIn(
                parent, surviving_uuids,
                msg=f"parentUuid {parent!r} on uuid {msg.get('uuid')!r} "
                    f"does not resolve to a surviving message",
            )

        # Resume-readiness — requires P0-B; the root must survive.
        ok, reason = simulate_replay_readiness(after)
        self.assertTrue(
            ok,
            msg=f"Pruned session is not replayable: {reason}. "
                f"survivors: {sorted(s for s in surviving_uuids if s)}",
        )


# ─── C2: root preservation ───────────────────────────────────────────────────


class TestRootPreservation(unittest.TestCase):
    """C2 — the ORIGINAL parentUuid=null root uuid MUST survive.

    Review finding H-1: a structural check (`any(parentUuid is None)`) is
    bypassed by ``executor._relink_parent_chain`` which re-points dead-end
    chains to ``None`` (executor.py:135 returns None on dead-end). So a
    descendant whose original parent was dropped becomes a new pseudo-root,
    and the structural C2 silently passes even though the semantic root is
    gone. The fixed C2 captures ``original_root_uuid`` from ``msgs_before``
    and requires it to be in the surviving uuids.
    """

    def test_fails_when_root_is_dropped(self):
        from cozempic.safety import PruneValidationError, validate_post_prune  # type: ignore

        before = _build_clean_session(n_turns=5)
        # Drop the root (uuid u0)
        after = [m for m in before if m[1].get("uuid") != "u0"]

        with self.assertRaises(PruneValidationError) as ctx:
            validate_post_prune(before, after)
        self.assertEqual(ctx.exception.evidence.get("failed_check"), "C2")

    def test_fails_when_original_root_dropped_but_pseudo_root_present(self):
        """H-1 reproducer: a descendant gets parentUuid=None after re-link,
        but the original root uuid is gone — C2 must still fire."""
        from cozempic.safety import PruneValidationError, validate_post_prune  # type: ignore

        before = _build_clean_session(n_turns=5)
        # Strategy drops the original root u0; _relink_parent_chain re-points
        # u0's descendants to parentUuid=None. Model that by:
        #   - omitting u0 from msgs_after
        #   - flipping the next entry's parentUuid from "u0" to None to mimic
        #     the relink behaviour (pseudo-root introduced).
        rewired: list = []
        for idx, msg, size in before:
            if msg.get("uuid") == "u0":
                continue
            payload = dict(msg)
            if payload.get("parentUuid") == "u0":
                payload["parentUuid"] = None
            rewired.append((idx, payload, msg_bytes(payload)))

        with self.assertRaises(PruneValidationError) as ctx:
            validate_post_prune(before, rewired)
        self.assertEqual(ctx.exception.evidence.get("failed_check"), "C2")
        self.assertEqual(ctx.exception.evidence.get("expected_root_uuid"), "u0")

    def test_passes_when_root_preserved_even_if_others_dropped(self):
        from cozempic.safety import validate_post_prune  # type: ignore

        before = _build_clean_session(n_turns=5)
        # Keep root + the last user+assistant pair + metadata; drop middle.
        keep_uuids = {"u0", "u4", "a5", "t1", "lp1", "pm1"}
        after = [m for m in before if m[1].get("uuid") in keep_uuids]
        # parentUuid of u4 may now dangle — re-point it to root for the test.
        rewired = []
        for idx, msg, size in after:
            payload = dict(msg)
            if payload.get("uuid") == "u4":
                payload["parentUuid"] = "u0"
            if payload.get("uuid") == "a5":
                payload["parentUuid"] = "u4"
            if payload.get("uuid") in ("t1", "lp1", "pm1"):
                payload["parentUuid"] = "a5"
            rewired.append((idx, payload, msg_bytes(payload)))

        validate_post_prune(before, rewired)


# ─── C3: conversation survival ────────────────────────────────────────────────


class TestConversationSurvival(unittest.TestCase):
    """C3 — at least one user AND one assistant must survive."""

    def test_fails_when_zero_users_after(self):
        from cozempic.safety import PruneValidationError, validate_post_prune  # type: ignore

        before = _build_clean_session(n_turns=5)
        # Drop every user
        after = [m for m in before if m[1].get("type") != "user"]

        with self.assertRaises(PruneValidationError) as ctx:
            validate_post_prune(before, after)
        self.assertEqual(ctx.exception.evidence.get("failed_check"), "C3")

    def test_fails_when_zero_assistants_after(self):
        from cozempic.safety import PruneValidationError, validate_post_prune  # type: ignore

        before = _build_clean_session(n_turns=5)
        after = [m for m in before if m[1].get("type") != "assistant"]

        with self.assertRaises(PruneValidationError) as ctx:
            validate_post_prune(before, after)
        self.assertEqual(ctx.exception.evidence.get("failed_check"), "C3")


# ─── C4: compact_boundary preserved ──────────────────────────────────────────


class TestCompactionMarkerPreserved(unittest.TestCase):
    """C4 — the LAST compact_boundary entry must survive any prune.

    The bug we are fixing: `compact-summary-collapse` drops everything BEFORE the
    boundary, but the boundary marker itself must remain so Claude Code's resume
    engine knows the file is post-compact (otherwise replay tries to walk the
    original chain that doesn't exist any more).
    """

    def test_fails_when_compact_boundary_dropped(self):
        from cozempic.safety import PruneValidationError, validate_post_prune  # type: ignore

        before = _build_clean_session(n_turns=5, include_compact_boundary=True)
        # Drop the compact_boundary (uuid cb1)
        after = [m for m in before if m[1].get("uuid") != "cb1"]

        with self.assertRaises(PruneValidationError) as ctx:
            validate_post_prune(before, after)
        self.assertEqual(ctx.exception.evidence.get("failed_check"), "C4")

    def test_last_compact_boundary_survives_when_multiple(self):
        from cozempic.safety import PruneValidationError, validate_post_prune  # type: ignore

        before = _build_clean_session(n_turns=5, include_compact_boundary=True)
        # Inject a SECOND boundary later in the file
        more = list(before)
        more.append(_compact_boundary(len(more), "cb2", "a5"))

        # After-state: drop cb2 (the LAST one) → must fail C4.
        after_drop_last = [m for m in more if m[1].get("uuid") != "cb2"]
        with self.assertRaises(PruneValidationError) as ctx:
            validate_post_prune(more, after_drop_last)
        self.assertEqual(ctx.exception.evidence.get("failed_check"), "C4")

        # After-state: drop cb1 only (older one) → must pass.
        after_drop_first = [m for m in more if m[1].get("uuid") != "cb1"]
        validate_post_prune(more, after_drop_first)

    def test_passes_when_no_compact_boundary_in_before(self):
        """C4 is conditional — if msgs_before had no boundary, nothing to check."""
        from cozempic.safety import validate_post_prune  # type: ignore

        before = _build_clean_session(n_turns=5, include_compact_boundary=False)
        after = before  # identity prune
        validate_post_prune(before, after)


# ─── C5: last permission-mode preserved ──────────────────────────────────────


class TestLastPermissionModePreserved(unittest.TestCase):

    def test_fails_when_last_permission_mode_dropped(self):
        from cozempic.safety import PruneValidationError, validate_post_prune  # type: ignore

        before = _build_clean_session(n_turns=5)
        after = [m for m in before if m[1].get("type") != "permission-mode"]

        with self.assertRaises(PruneValidationError) as ctx:
            validate_post_prune(before, after)
        self.assertEqual(ctx.exception.evidence.get("failed_check"), "C5")


# ─── C6: last last-prompt preserved ──────────────────────────────────────────


class TestLastPromptPreserved(unittest.TestCase):

    def test_fails_when_last_prompt_dropped(self):
        from cozempic.safety import PruneValidationError, validate_post_prune  # type: ignore

        before = _build_clean_session(n_turns=5)
        after = [m for m in before if m[1].get("type") != "last-prompt"]

        with self.assertRaises(PruneValidationError) as ctx:
            validate_post_prune(before, after)
        self.assertEqual(ctx.exception.evidence.get("failed_check"), "C6")


# ─── Last-K turn preservation (P0-C floor) ────────────────────────────────────


class TestLastKTurnsPreserved(unittest.TestCase):
    """Floor enforcement: the most recent K user+assistant pairs survive any prune."""

    def test_last_k_user_and_assistant_msgs_survive_after_floor(self):
        from cozempic.safety import FloorConfig, enforce_floor  # type: ignore

        before = _build_clean_session(n_turns=100, include_metadata=False)
        # Synthetic "after": drop the entire last K block (worst-case strategy).
        last_k = 10
        # Identify the uuids of the last K user+assistant entries
        users = [m for m in before if m[1].get("type") == "user"]
        assistants = [m for m in before if m[1].get("type") == "assistant"]
        last_k_user_uuids = {m[1]["uuid"] for m in users[-last_k:]}
        last_k_asst_uuids = {m[1]["uuid"] for m in assistants[-last_k:]}
        dropped = last_k_user_uuids | last_k_asst_uuids
        after = [m for m in before if m[1].get("uuid") not in dropped]

        cfg = FloorConfig(
            max_user_assistant_drop_pct=0.50,
            preserve_last_k_turns=last_k,
            preserve_first_message=True,
        )
        re_added = enforce_floor(before, after, cfg=cfg)
        re_added_uuids = {m[1].get("uuid") for m in re_added}

        # Every last-K user + assistant uuid must be back in the result.
        for uuid in last_k_user_uuids | last_k_asst_uuids:
            self.assertIn(uuid, re_added_uuids, f"uuid {uuid!r} not re-added by floor")

    def test_first_message_re_added_when_dropped(self):
        from cozempic.safety import FloorConfig, enforce_floor  # type: ignore

        before = _build_clean_session(n_turns=10, include_metadata=False)
        # Strategy drops the root.
        after = [m for m in before if m[1].get("uuid") != "u0"]

        cfg = FloorConfig(preserve_first_message=True)
        re_added = enforce_floor(before, after, cfg=cfg)
        re_added_uuids = {m[1].get("uuid") for m in re_added}

        self.assertIn("u0", re_added_uuids)

    def test_drop_percentage_cap_enforced(self):
        """Even when last-K is satisfied, total drop pct must not exceed cap."""
        from cozempic.safety import FloorConfig, enforce_floor  # type: ignore

        # 100 user + 100 assistant; strategy drops 95 of each (95% drop).
        before = _build_clean_session(n_turns=100, include_metadata=False)
        users = [m for m in before if m[1].get("type") == "user"]
        asst = [m for m in before if m[1].get("type") == "assistant"]

        # Keep only the LAST 5 of each → 95% drop, exceeds 50% cap.
        keep_user_uuids = {u[1]["uuid"] for u in users[-5:]}
        keep_asst_uuids = {a[1]["uuid"] for a in asst[-5:]}
        keep = keep_user_uuids | keep_asst_uuids
        after = [m for m in before if m[1].get("uuid") in keep
                 or m[1].get("type") not in ("user", "assistant")]

        cfg = FloorConfig(
            max_user_assistant_drop_pct=0.50,
            preserve_last_k_turns=5,
            preserve_first_message=True,
        )
        re_added = enforce_floor(before, after, cfg=cfg)

        # After floor: surviving user+assistant count must be ≥ 50% of before.
        before_ua = len(users) + len(asst)
        re_added_ua = sum(
            1 for _, m, _ in re_added if m.get("type") in ("user", "assistant")
        )
        self.assertGreaterEqual(
            re_added_ua / before_ua, 0.50,
            f"floor cap violated: re-added {re_added_ua}/{before_ua} = "
            f"{re_added_ua / before_ua:.0%}, expected ≥50%",
        )


# ─── Replay-readiness simulation ─────────────────────────────────────────────


class TestReplaySimulation(unittest.TestCase):
    """Walk the surviving parentUuid graph as a Claude Code resume would.

    A successful replay simulation requires:
      (a) Find a root (parentUuid=None) → must exist.
      (b) Build a uuid → entry map from msgs_after.
      (c) For each non-root entry, parentUuid must resolve via the map.
      (d) The conversation chain from root walked forward must include at
          least one user AND one assistant entry.

    This is a strictly structural simulation — no Claude binary spawn. It is
    the same shape Claude Code's `--resume` bootstrap uses to decide whether to
    re-attach to the JSONL or hang.
    """

    def test_replay_simulation_passes_on_clean_session(self):
        from cozempic.safety import simulate_replay_readiness  # type: ignore

        before = _build_clean_session(n_turns=5, include_compact_boundary=True)
        ok, reason = simulate_replay_readiness(before)
        self.assertTrue(ok, msg=f"clean session failed replay simulation: {reason}")

    def test_replay_simulation_fails_when_root_missing(self):
        from cozempic.safety import simulate_replay_readiness  # type: ignore

        before = _build_clean_session(n_turns=5)
        # Strip the root
        broken = [m for m in before if m[1].get("uuid") != "u0"]
        ok, reason = simulate_replay_readiness(broken)
        self.assertFalse(ok)
        self.assertIn("root", reason.lower())

    def test_replay_simulation_fails_on_broken_chain(self):
        from cozempic.safety import simulate_replay_readiness  # type: ignore

        before = _build_clean_session(n_turns=5)
        # Drop intermediate "a1" but keep entries pointing at it
        broken = [m for m in before if m[1].get("uuid") != "a1"]
        ok, reason = simulate_replay_readiness(broken)
        self.assertFalse(ok)
        self.assertIn("chain", reason.lower())

    def test_replay_simulation_fails_on_zero_conversation(self):
        from cozempic.safety import simulate_replay_readiness  # type: ignore

        # Only metadata, no user/assistant
        msgs = [
            _msg(0, {"type": "ai-title", "uuid": "t1",
                    "parentUuid": None, "title": "x"}),
        ]
        ok, reason = simulate_replay_readiness(msgs)
        self.assertFalse(ok)


# ─── Integration: executor wires validation + floor into run_prescription ─────


class TestExecutorWiresValidationAndFloor(unittest.TestCase):
    """run_prescription must invoke floor + validation before returning."""

    def test_run_prescription_aborts_on_validation_failure(self):
        """If a strategy combo produces an invalid post-prune list,
        run_prescription must raise PruneValidationError BEFORE save_messages
        is called.

        Drops every user message and disables the floor — C3 (conversation
        survival) fires inside run_prescription.
        """
        from cozempic.executor import run_prescription
        from cozempic.registry import strategy, STRATEGIES
        from cozempic.safety import PruneValidationError  # type: ignore
        from cozempic.types import PruneAction, StrategyResult

        @strategy("test-drop-all-users", "Test only: drops all users", "gentle", "100%")
        def _drop_all_users(messages, config):  # noqa: ANN001
            actions = []
            for idx, msg, size in messages:
                if msg.get("type") == "user":
                    actions.append(PruneAction(
                        line_index=idx, action="remove",
                        reason="test", original_bytes=size, pruned_bytes=0,
                    ))
            return StrategyResult(
                strategy_name="test-drop-all-users", actions=actions,
                original_bytes=sum(b for _, _, b in messages),
                pruned_bytes=sum(a.original_bytes for a in actions),
                messages_affected=len(actions),
                messages_removed=len(actions),
                messages_replaced=0,
                summary="dropped all users",
            )

        try:
            before = _build_clean_session(n_turns=5)
            with self.assertRaises(PruneValidationError):
                run_prescription(
                    before,
                    ["test-drop-all-users"],
                    {"_disable_floor_for_test": True},
                )
        finally:
            STRATEGIES.pop("test-drop-all-users", None)

    def test_run_prescription_floor_re_adds_root_when_strategy_drops_it(self):
        """With the floor enabled (default), the executor re-adds the root
        before validation runs, so the prescription succeeds end-to-end."""
        from cozempic.executor import run_prescription
        from cozempic.registry import strategy, STRATEGIES
        from cozempic.types import PruneAction, StrategyResult

        @strategy("test-drop-root2", "Test only: drops the root", "gentle", "100%")
        def _drop_root(messages, config):  # noqa: ANN001
            actions = []
            for idx, msg, size in messages:
                if msg.get("parentUuid") is None:
                    actions.append(PruneAction(
                        line_index=idx, action="remove",
                        reason="test", original_bytes=size, pruned_bytes=0,
                    ))
            return StrategyResult(
                strategy_name="test-drop-root2", actions=actions,
                original_bytes=sum(b for _, _, b in messages),
                pruned_bytes=sum(a.original_bytes for a in actions),
                messages_affected=len(actions),
                messages_removed=len(actions),
                messages_replaced=0,
                summary="dropped root",
            )

        try:
            before = _build_clean_session(n_turns=5)
            after, _ = run_prescription(before, ["test-drop-root2"], {})
            # Floor re-added the root → run_prescription completes; root is back.
            survives = {m[1].get("uuid") for m in after}
            self.assertIn("u0", survives)
        finally:
            STRATEGIES.pop("test-drop-root2", None)


if __name__ == "__main__":
    unittest.main()
