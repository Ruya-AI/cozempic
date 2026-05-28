"""RED tests for review-max Group A — exception-path hygiene.

Findings:
  A.1  save_messages H-3 PruneValidationError uncaught by guard/doctor callers
  A.3  _strip_metadata_singleton_tags not in try/finally
  A.7  cmd_diagnose / cmd_current / cmd_formulary don't catch PruneValidationError
  A.8  validation_failed advances K-counter
  A.15 prune_with_team_protect leaks team tag on PruneValidationError
"""

from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from cozempic.helpers import msg_bytes


def _msg(idx: int, payload: dict) -> tuple[int, dict, int]:
    return (idx, payload, msg_bytes(payload))


def _valid_session() -> list[tuple[int, dict, int]]:
    return [
        _msg(0, {"type": "user", "uuid": "u0", "parentUuid": None,
                 "message": {"role": "user", "content": "hi"}}),
        _msg(1, {"type": "assistant", "uuid": "a1", "parentUuid": "u0",
                 "message": {"role": "assistant",
                             "content": [{"type": "text", "text": "hello"}]}}),
        _msg(2, {"type": "user", "uuid": "u1", "parentUuid": "a1",
                 "message": {"role": "user", "content": "follow"}}),
        _msg(3, {"type": "assistant", "uuid": "a2", "parentUuid": "u1",
                 "message": {"role": "assistant",
                             "content": [{"type": "text", "text": "world"}]}}),
        _msg(4, {"type": "permission-mode", "uuid": "pm1", "parentUuid": "a2",
                 "mode": "default"}),
        _msg(5, {"type": "last-prompt", "uuid": "lp1", "parentUuid": "a2",
                 "text": "follow"}),
    ]


def _write_session_jsonl(path: Path, messages: list[tuple[int, dict, int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for _, m, _ in messages:
            f.write(json.dumps(m, separators=(",", ":")) + "\n")


def _age_mtime(path: Path, hours_ago: float) -> None:
    target = time.time() - hours_ago * 3600
    os.utime(path, (target, target))


# ─── A.1 — guard catches PruneValidationError from save_messages ──────────────


class TestGuardCatchesValidationFromSave(unittest.TestCase):
    """A.1: save_messages raises PruneValidationError on H-3 post-append fail;
    guard.guard_prune_cycle MUST catch it and return _no_change shape."""

    def test_guard_prune_cycle_catches_save_messages_validation_error(self):
        from cozempic.guard import guard_prune_cycle
        from cozempic.safety import PruneValidationError
        from cozempic.team import TeamState

        with TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "session.jsonl"
            _write_session_jsonl(session_path, _valid_session())
            _age_mtime(session_path, 25)  # idle, not refused

            # Patch save_messages to raise PruneValidationError as the H-3
            # post-append validation would. Also short-circuit
            # prune_with_team_protect so the saved_bytes>0 path is taken and
            # save_messages actually runs.
            full = _valid_session()
            pruned = full[:-2]  # drop the last 2 entries so saved_bytes > 0
            empty_team = TeamState()

            def _raise_save(*a, **kw):
                raise PruneValidationError(
                    reason="post-append delta orphan",
                    evidence={"failed_check": "C1", "dangling_uuid": "ghost"},
                )

            with mock.patch(
                "cozempic.guard.prune_with_team_protect",
                return_value=(pruned, [], empty_team),
            ), mock.patch(
                "cozempic.guard.save_messages", side_effect=_raise_save,
            ):
                result = guard_prune_cycle(
                    session_path=session_path,
                    rx_name="gentle",
                    config={},
                    auto_reload=False,
                    cwd=str(tmpdir),
                    session_id="test",
                )
            self.assertEqual(result.get("saved_mb", -1.0), 0.0)
            self.assertFalse(result.get("reloading", True))
            self.assertTrue(
                result.get("validation_failed", False),
                f"expected validation_failed=True; result={result}",
            )


# ─── A.3 — singleton tag survives exception in run_prescription ──────────────


class TestSingletonTagStrippedInFinally(unittest.TestCase):
    """A.3: if anything between tag and strip raises, the strip must STILL
    run (try/finally). Otherwise the caller's input list keeps the internal
    __cozempic_metadata_singleton__ flag and saves it to disk."""

    def test_input_messages_have_no_singleton_tag_after_exception(self):
        from cozempic.executor import run_prescription
        from cozempic.registry import strategy, STRATEGIES
        from cozempic.types import PruneAction, StrategyResult
        from cozempic.safety import PruneValidationError

        @strategy("test-a3-drop-all-users", "Test only", "gentle", "n/a")
        def _drop(messages, config):  # noqa: ANN001
            actions = []
            for idx, m, size in messages:
                if m.get("type") == "user":
                    actions.append(PruneAction(
                        line_index=idx, action="remove",
                        reason="t", original_bytes=size, pruned_bytes=0,
                    ))
            return StrategyResult(
                strategy_name="test-a3-drop-all-users", actions=actions,
                original_bytes=0, pruned_bytes=0,
                messages_affected=len(actions),
                messages_removed=len(actions), messages_replaced=0,
                summary="",
            )

        try:
            messages = _valid_session()
            try:
                # enable_floor=False → validation fires on C3; raises.
                run_prescription(messages, ["test-a3-drop-all-users"], {},
                                 enable_floor=False)
            except PruneValidationError:
                pass
            # The caller's input list MUST not carry the internal tag.
            for _, m, _ in messages:
                self.assertNotIn(
                    "__cozempic_metadata_singleton__", m,
                    msg=f"leaked singleton tag on uuid {m.get('uuid')!r}",
                )
        finally:
            STRATEGIES.pop("test-a3-drop-all-users", None)


# ─── A.7 — cmd_diagnose / cmd_current loop survives validation failure ───────


class TestDiagnoseLoopContinuesOnValidationError(unittest.TestCase):
    """A.7: cmd_current and cmd_diagnose loop over every prescription.
    A single PruneValidationError from one prescription must NOT abort the
    loop or crash the CLI — diagnostic dry-runs should report all
    prescriptions so the operator sees the full picture."""

    def test_run_prescription_failures_dont_propagate_in_diagnostic_loops(self):
        # We do not invoke cmd_diagnose end-to-end (that requires session
        # resolution). Instead we mirror its loop shape and confirm the
        # try/except around run_prescription is in place.
        from cozempic import cli as cli_mod
        import inspect
        src = inspect.getsource(cli_mod.cmd_diagnose)
        # The fix wraps the run_prescription call in try/except
        # PruneValidationError; the source must contain BOTH the call and
        # the exception type by name.
        self.assertIn("run_prescription", src)
        self.assertIn(
            "PruneValidationError", src,
            msg="cmd_diagnose must catch PruneValidationError on its "
                "per-prescription dry-run loop (A.7)",
        )

        src_cur = inspect.getsource(cli_mod.cmd_current)
        if "run_prescription" in src_cur:
            self.assertIn(
                "PruneValidationError", src_cur,
                msg="cmd_current must catch PruneValidationError on its "
                    "per-prescription dry-run loop (A.7)",
            )


# ─── A.8 — validation_failed must not advance K-counter ──────────────────────


class TestValidationFailedDoesNotAdvanceKCounter(unittest.TestCase):
    """A.8: the outer guard daemon loop has a carve-out for
    active_session_refused that skips the K-counter increment. The same
    carve-out must apply to validation_failed — a refused prune cycle is
    semantically the same regardless of which gate refused it."""

    def test_outer_loop_carveout_includes_validation_failed(self):
        # Source-level check: the carve-out condition must mention both
        # active_session_refused AND validation_failed. We don't run the
        # daemon end-to-end (too heavy); the source is the contract.
        from cozempic import guard as guard_mod
        import inspect
        src = inspect.getsource(guard_mod.start_guard)
        self.assertIn(
            "active_session_refused", src,
            msg="missing active_session_refused carve-out",
        )
        self.assertIn(
            "validation_failed", src,
            msg="A.8: outer loop must skip K-counter on validation_failed too",
        )


# ─── A.15 — team-protected tag stripped even on PruneValidationError ─────────


class TestPruneWithTeamProtectStripsTagOnException(unittest.TestCase):
    """A.15: prune_with_team_protect tags messages with
    __cozempic_team_protected__ before calling run_prescription. If
    run_prescription raises, the cleanup pass at the bottom is skipped and
    the tag persists on the in-memory messages → leaks to disk on the next
    save."""

    def test_team_protected_tag_stripped_when_run_prescription_raises(self):
        from cozempic.guard import prune_with_team_protect
        from cozempic.team import TEAM_TOOL_NAMES
        from cozempic.safety import PruneValidationError

        # A tool_use with name=Task qualifies as a team message.
        team_tool_name = next(iter(TEAM_TOOL_NAMES))
        messages = [
            _msg(0, {"type": "user", "uuid": "u0", "parentUuid": None,
                     "message": {"role": "user", "content": "hi"}}),
            _msg(1, {"type": "assistant", "uuid": "a1", "parentUuid": "u0",
                     "message": {"role": "assistant", "content": [
                         {"type": "tool_use", "id": "tu0",
                          "name": team_tool_name, "input": {}},
                     ]}}),
            _msg(2, {"type": "user", "uuid": "u1", "parentUuid": "a1",
                     "message": {"role": "user", "content": [
                         {"type": "tool_result", "tool_use_id": "tu0",
                          "content": "done"},
                     ]}}),
        ]

        def _raise(*a, **kw):
            raise PruneValidationError(
                reason="forced", evidence={"failed_check": "C1"},
            )

        with mock.patch("cozempic.guard.run_prescription", side_effect=_raise):
            with self.assertRaises(PruneValidationError):
                prune_with_team_protect(messages, rx_name="gentle", config={})

        # After the exception, no surviving message in the caller's input
        # list may still carry the team-protected tag.
        for _, m, _ in messages:
            self.assertNotIn(
                "__cozempic_team_protected__", m,
                msg=f"leaked team-protected tag on uuid {m.get('uuid')!r}",
            )


if __name__ == "__main__":
    unittest.main()
