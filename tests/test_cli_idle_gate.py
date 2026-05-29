"""RED tests for P4 — idle gate + --force on cmd_strategy --execute and
doctor.fix_orphaned_tool_results (PLAN.md §2.6, §3.5).

RED at current HEAD:
- cmd_strategy --execute has no idle gate and p_strat has no --force argument.
- doctor.fix_orphaned_tool_results skips active sessions on PruneLockError but
  NOT on active-mtime (i.e., no assert_session_idle_or_force guard).
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d, separators=(",", ":")) + "\n")


def _set_mtime(path: Path, seconds_ago: float) -> None:
    now = time.time()
    target = now - seconds_ago
    os.utime(path, (target, target))


def _minimal_session() -> list[dict]:
    """Minimal session with orphan-friendly structure."""
    return [
        {"type": "user", "uuid": "u1", "parentUuid": None,
         "message": {"role": "user", "content": "hello"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "message": {"role": "assistant",
                     "content": [{"type": "text", "text": "hi"}]}},
        # Orphaned tool_result: tool_use_id not present in any tool_use block
        {"type": "user", "uuid": "u2", "parentUuid": "a1",
         "message": {"role": "user", "content": [
             {"type": "tool_result",
              "tool_use_id": "orphan-id-001",
              "content": "result"}
         ]}},
        {"type": "assistant", "uuid": "a2", "parentUuid": "u2",
         "message": {"role": "assistant",
                     "content": [{"type": "text", "text": "done"}]}},
        {"type": "ai-title", "uuid": "t1", "parentUuid": "a2",
         "title": "Test"},
        {"type": "last-prompt", "uuid": "lp1", "parentUuid": "a2",
         "text": "hello"},
        {"type": "permission-mode", "uuid": "pm1", "parentUuid": "a2",
         "mode": "default"},
    ]


class TestCmdStrategyIdleGate(unittest.TestCase):
    """cmd_strategy --execute must refuse active sessions (force=False) and
    accept active sessions with --force.

    RED at current HEAD: cmd_strategy --execute has no idle gate.
    """

    def _run_strategy(self, path: Path, force: bool = False) -> None:
        """Invoke cmd_strategy with metadata-strip strategy --execute via argparse."""
        from cozempic.cli import build_parser, cmd_strategy  # type: ignore

        parser = build_parser()
        args = parser.parse_args(
            ["strategy", "metadata-strip", str(path), "--execute"]
            + (["--force"] if force else [])
        )
        cmd_strategy(args)

    def test_cmd_strategy_execute_refuses_active_session(self):
        """cmd_strategy --execute on fresh mtime (no --force) → SystemExit(4).

        RED: no idle gate today — cmd_strategy --execute proceeds regardless of mtime.
        """
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.jsonl"
            _write_jsonl(path, _minimal_session())
            _set_mtime(path, seconds_ago=60)  # 1 minute ago — active

            with patch.dict(os.environ, {"COZEMPIC_MIN_IDLE_HOURS": "24"}), \
                 patch("cozempic.session.find_sessions",
                       return_value=[{"path": path, "session_id": path.stem,
                                      "size": 100, "project": "p"}]), \
                 patch("cozempic.session.find_current_session",
                       return_value={"path": path, "session_id": path.stem,
                                     "size": 100, "project": "p"}):
                with self.assertRaises(SystemExit) as ctx:
                    self._run_strategy(path, force=False)

            self.assertEqual(
                ctx.exception.code, 4,
                msg=(
                    f"cmd_strategy --execute on active session must SystemExit(4). "
                    f"Got SystemExit({ctx.exception.code!r}). "
                    f"P4 not implemented: no idle gate in cmd_strategy."
                ),
            )

    def test_cmd_strategy_execute_with_force_proceeds(self):
        """cmd_strategy --execute --force on fresh mtime → proceeds, no SystemExit(4).

        RED: --force is not wired to cmd_strategy today.
        """
        from cozempic.cli import build_parser  # type: ignore

        parser = build_parser()
        # --force must be a valid argument for p_strat
        try:
            args = parser.parse_args(["strategy", "metadata-strip", "some-session",
                                      "--execute", "--force"])
        except SystemExit as exc:
            self.fail(
                f"p_strat does not accept --force: {exc}. "
                f"P4 not implemented: --force not added to p_strat argparse."
            )

        self.assertTrue(
            getattr(args, "force", False),
            msg="args.force must be True when --force is passed to strategy subcommand.",
        )

    def test_p_strat_argparse_accepts_force_flag(self):
        """The strategy subparser must expose --force in its argparse definition.

        RED: p_strat (cli.py:1296) has no --force argument today.
        """
        from cozempic.cli import build_parser  # type: ignore

        parser = build_parser()
        # Parse with --force; must not raise SystemExit
        try:
            ns = parser.parse_args(["strategy", "metadata-strip", "fake-session",
                                    "--force"])
        except SystemExit:
            self.fail(
                "p_strat argparse rejects --force. "
                "P4 not implemented: add --force to p_strat."
            )

        self.assertTrue(
            ns.force,
            "strategy --force must set args.force=True.",
        )


class TestDoctorIdleGate(unittest.TestCase):
    """doctor.fix_orphaned_tool_results must skip sessions with fresh mtime.

    RED at current HEAD: doctor has no assert_session_idle_or_force guard.
    """

    def test_doctor_fix_orphaned_skips_active_session(self):
        """fix_orphaned_tool_results on a fresh-mtime session → skipped (not modified).

        RED: doctor currently does NOT check mtime before fixing — it will modify
        active sessions. After P4 fix: active sessions are skipped silently.
        """
        from cozempic.doctor import fix_orphaned_tool_results  # type: ignore

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.jsonl"
            _write_jsonl(path, _minimal_session())
            _set_mtime(path, seconds_ago=60)  # active session
            original_content = path.read_bytes()

            with patch("cozempic.doctor.find_sessions",
                       return_value=[{
                           "path": path,
                           "session_id": path.stem,
                           "size": path.stat().st_size,
                       }]), \
                 patch.dict(os.environ, {"COZEMPIC_MIN_IDLE_HOURS": "24"}):
                result = fix_orphaned_tool_results()

            # File must not have been modified
            after_content = path.read_bytes()
            self.assertEqual(
                original_content, after_content,
                msg=(
                    f"doctor.fix_orphaned_tool_results modified an active session "
                    f"(mtime 60s ago, threshold 24h). P4 requires an idle gate here. "
                    f"Result: {result!r}"
                ),
            )
            # Result message should mention the skip
            self.assertIn(
                "skip", result.lower(),
                msg=(
                    f"Result must mention the skipped session. Got: {result!r}. "
                    f"Check that fix_orphaned_tool_results reports skipped_sessions "
                    f"when the idle gate fires."
                ),
            )


if __name__ == "__main__":
    unittest.main()
