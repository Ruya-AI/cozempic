"""RED→GREEN tests for PR #138 review fixes.

Fix 1 — guard.py: `_persisted_tokens_saved(pre, post)` — maximal prune to post==0
  must return pre, not 0 (the `and post_te.total` bug).

Fix 2 — team.py: `extract_team_state` must not crash on non-str `text` fields
  in tool-result sub-blocks or in assistant lead-summary blocks.

Fix 3 — overflow.py: reactive safe-point gate must FAIL-CLOSED (Exception →
  defer, not → kill).
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1: _persisted_tokens_saved helper (extracted from guard.py inline expr)
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistedTokensSaved(unittest.TestCase):
    """Unit-tests for the pure helper `_persisted_tokens_saved(pre, post)`.

    RED at base: the OLD inline expression
        `pre - post if pre and post else 0`
    returns 0 when post==0 (because `and post` is falsy), losing the savings.

    GREEN: the helper returns pre when pre>0, post==0 (maximal prune).
    """

    def _call(self, pre: int, post: int) -> int:
        from cozempic.guard import _persisted_tokens_saved
        return _persisted_tokens_saved(pre, post)

    def test_maximal_prune_to_zero_returns_pre(self):
        """Maximal prune: pre=100, post=0 → savings==100 (was 0 at base)."""
        result = self._call(100, 0)
        self.assertEqual(result, 100,
            "A maximal prune (post==0) should report pre tokens saved, not 0")

    def test_normal_prune_returns_difference(self):
        """Normal prune: pre=1000, post=400 → 600."""
        self.assertEqual(self._call(1000, 400), 600)

    def test_zero_pre_returns_zero(self):
        """pre==0 means nothing to measure; both OLD and NEW should return 0."""
        self.assertEqual(self._call(0, 0), 0)
        self.assertEqual(self._call(0, 100), 0)

    def test_equal_tokens_returns_zero(self):
        """No prune: pre==post → 0 saved."""
        self.assertEqual(self._call(500, 500), 0)

    def test_large_maximal_prune(self):
        """Large maximal prune: pre=200_000, post=0 → 200_000."""
        result = self._call(200_000, 0)
        self.assertEqual(result, 200_000)


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2: extract_team_state survives non-str `text` fields
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractTeamStateNonStrText(unittest.TestCase):
    """Prove that extract_team_state raises TypeError at base on non-str text.

    (a) tool-result sub-block with text=99999 (int) → _extract_block_text join crashes
    (b) assistant team message with text=None in block → [:300] slice crashes
    """

    def _write_session(self, lines: list[dict]) -> Path:
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        p = Path(d) / "session.jsonl"
        p.write_text("\n".join(json.dumps(ln) for ln in lines) + "\n")
        return p

    def _load_and_extract(self, session_path: Path):
        from cozempic.session import load_messages
        from cozempic.team import extract_team_state
        msgs = load_messages(session_path)
        return extract_team_state(msgs)

    # ── (a) non-str text in tool-result sub-block ──────────────────────────
    def test_tool_result_int_text_does_not_crash(self):
        """tool_result with text=99999 (int) must not raise TypeError."""
        # A Task tool_use followed by a tool_result whose content block
        # carries a numeric text field — the old `"".join(sub.get("text","")...)`
        # raises TypeError: sequence item N: expected str instance, int found.
        session = self._write_session([
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "name": "Task", "id": "tid1",
                         "input": {"subagent_type": "builder", "prompt": "go"}}
                    ]
                }
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tid1",
                         "content": [{"type": "text", "text": 99999}]}
                    ]
                }
            },
        ])
        # Must not raise — returns a TeamState (possibly empty or partial)
        try:
            state = self._load_and_extract(session)
        except TypeError as exc:
            self.fail(f"extract_team_state raised TypeError on int text: {exc}")

    def test_tool_result_none_text_does_not_crash(self):
        """tool_result with text=None must not raise TypeError."""
        session = self._write_session([
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "name": "Task", "id": "tid2",
                         "input": {"subagent_type": "builder", "prompt": "go"}}
                    ]
                }
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tid2",
                         "content": [{"type": "text", "text": None}]}
                    ]
                }
            },
        ])
        try:
            self._load_and_extract(session)
        except TypeError as exc:
            self.fail(f"extract_team_state raised TypeError on None text: {exc}")

    def test_tool_result_list_text_does_not_crash(self):
        """tool_result with text=[] (list) must not raise TypeError."""
        session = self._write_session([
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "name": "Task", "id": "tid3",
                         "input": {"subagent_type": "builder", "prompt": "go"}}
                    ]
                }
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tid3",
                         "content": [{"type": "text", "text": [1, 2, 3]}]}
                    ]
                }
            },
        ])
        try:
            self._load_and_extract(session)
        except TypeError as exc:
            self.fail(f"extract_team_state raised TypeError on list text: {exc}")

    # ── (b) non-str text in assistant lead-summary block ───────────────────
    def test_assistant_team_msg_int_text_does_not_crash(self):
        """assistant team message with text=42 must not crash the lead-summary builder."""
        # Inject a TeamCreate tool_use so the message is recognized as a team
        # message, then give the same assistant message a text block with text=42.
        session = self._write_session([
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "name": "TeamCreate", "id": "tc1",
                         "input": {"name": "my-team"}},
                        {"type": "text", "text": 42}
                    ]
                }
            },
        ])
        try:
            self._load_and_extract(session)
        except TypeError as exc:
            self.fail(
                f"extract_team_state raised TypeError on int text in lead-summary: {exc}")

    def test_assistant_team_msg_none_text_does_not_crash(self):
        """assistant team message with text=None must not crash."""
        session = self._write_session([
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "name": "TeamCreate", "id": "tc2",
                         "input": {"name": "team-b"}},
                        {"type": "text", "text": None}
                    ]
                }
            },
        ])
        try:
            self._load_and_extract(session)
        except TypeError as exc:
            self.fail(
                f"extract_team_state raised TypeError on None text in lead-summary: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Fix 3: reactive safe-point gate must FAIL-CLOSED
# ─────────────────────────────────────────────────────────────────────────────

class TestOverflowSafePointFailClosed(unittest.TestCase):
    """Prove that an exception inside the safe-point gate causes deferral (not kill).

    At BASE: Exception → _safe=True → _terminate_and_resume IS called (SIGKILL).
    After fix: Exception → _safe=False → _terminate_and_resume NOT called,
               checkpoint_team IS called.
    """

    def _make_recovery(self, tmp: str, session_path: Path):
        from cozempic.overflow import CircuitBreaker, OverflowRecovery
        breaker = CircuitBreaker(
            session_id="test-fc",
            max_recoveries=5,
            window_seconds=300,
        )
        breaker.reset()
        return OverflowRecovery(
            session_path=session_path,
            session_id="test-fc",
            cwd=tmp,
            breaker=breaker,
            danger_threshold_mb=100.0,
            claude_pid=9999,
        )

    def _prune_result_safe(self, session_path: Path) -> dict:
        """Return a guard_prune_cycle result that passes the pre-flight check."""
        # small _final_bytes → after_mb well below 100 MB → not still_dangerous
        return {
            "saved_mb": 5.0,
            "original_tokens": 50_000,
            "final_tokens": 10_000,
            "_final_bytes": 1024,          # ~0.001 MB — not dangerous
            "_deferred_writer": None,
            "_write_holder": {"written": False},
        }

    def test_gate_exception_causes_deferral_not_kill(self):
        """When extract_team_state raises, the gate must FAIL-CLOSED (no kill).

        _do_recover imports lazily: `from .guard import checkpoint_team, guard_prune_cycle,
        _terminate_and_resume` and `from .team import extract_team_state`.
        Patching the source modules intercepts the lazy import bindings.
        """
        with tempfile.TemporaryDirectory() as tmp:
            session_path = Path(tmp) / "session.jsonl"
            session_path.write_text(
                json.dumps({"type": "user", "message": {"content": "x"}}) + "\n"
            )

            rec = self._make_recovery(tmp, session_path)
            prune_result = self._prune_result_safe(session_path)

            mock_terminate = MagicMock()
            mock_checkpoint = MagicMock()

            with (
                patch("cozempic.guard.guard_prune_cycle",
                      return_value=prune_result),
                patch("cozempic.team.extract_team_state",
                      side_effect=RuntimeError("boom")),
                patch("cozempic.guard._terminate_and_resume", mock_terminate),
                patch("cozempic.guard.checkpoint_team", mock_checkpoint),
            ):
                rec._do_recover()

            # FAIL-CLOSED: exception → deferred, not killed
            mock_terminate.assert_not_called()
            mock_checkpoint.assert_called()

    def test_gate_load_messages_exception_causes_deferral(self):
        """When load_messages raises inside the gate, must FAIL-CLOSED."""
        with tempfile.TemporaryDirectory() as tmp:
            session_path = Path(tmp) / "session.jsonl"
            session_path.write_text(
                json.dumps({"type": "user", "message": {"content": "y"}}) + "\n"
            )

            rec = self._make_recovery(tmp, session_path)
            prune_result = self._prune_result_safe(session_path)

            mock_terminate = MagicMock()
            mock_checkpoint = MagicMock()

            with (
                patch("cozempic.guard.guard_prune_cycle",
                      return_value=prune_result),
                patch("cozempic.guard.safe_to_reload",
                      side_effect=OSError("disk gone")),
                patch("cozempic.guard._terminate_and_resume", mock_terminate),
                patch("cozempic.guard.checkpoint_team", mock_checkpoint),
            ):
                rec._do_recover()

            mock_terminate.assert_not_called()
            mock_checkpoint.assert_called()

    def test_normal_safe_path_still_kills(self):
        """Control: when the gate succeeds with _safe=True, kill still proceeds."""
        with tempfile.TemporaryDirectory() as tmp:
            session_path = Path(tmp) / "session.jsonl"
            session_path.write_text(
                json.dumps({"type": "user", "message": {"content": "z"}}) + "\n"
            )

            rec = self._make_recovery(tmp, session_path)
            prune_result = self._prune_result_safe(session_path)

            mock_terminate = MagicMock()

            with (
                patch("cozempic.guard.guard_prune_cycle",
                      return_value=prune_result),
                patch("cozempic.guard.safe_to_reload",
                      return_value=(True, "")),
                patch("cozempic.guard._terminate_and_resume", mock_terminate),
                patch("cozempic.session.find_claude_pid", return_value=None),
                # _ReloadLock imported lazily from .reload_lock inside _do_recover
                patch("cozempic.reload_lock._ReloadLock") as mock_lock,
            ):
                mock_lock.return_value.__enter__ = MagicMock(return_value=None)
                mock_lock.return_value.__exit__ = MagicMock(return_value=False)
                rec._do_recover()

            # With safe=True and explicit pid=9999, _terminate_and_resume is called
            mock_terminate.assert_called()


if __name__ == "__main__":
    unittest.main()
