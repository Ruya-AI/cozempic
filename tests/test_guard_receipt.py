"""Guard/overflow auto-prunes emit a committed receipt (via _emit_guard_receipt)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cozempic import guard
from cozempic.metrics import validate_receipt
from cozempic.tokens import TokenEstimate
from cozempic.types import StrategyResult


def _args(**over):
    pre = TokenEstimate(2000, 20.0, "exact", "high", "claude-opus-4-8", 200000)
    post = TokenEstimate(1500, 15.0, "heuristic", "medium", "claude-opus-4-8", 200000)
    a = dict(
        session_path=Path("/x/sessZ.jsonl"),
        session_id="sessZ",
        cwd="/proj",
        rx_name="standard",
        trigger_source="guard",
        results=[StrategyResult("tool-output-trim", [], 1000, 200, 1, 0, 1, "")],
        pruned_messages=[(0, {"type": "user"}, 100)] * 9,
        original_msgs=[(0, {"type": "user"}, 100)] * 10,
        pre_te=pre,
        post_te=post,
        original_bytes=5000,
    )
    a.update(over)
    return a


def _only_receipt(home):
    d = Path(home) / ".cozempic" / "receipts"
    files = [p for p in d.glob("*.jsonl") if p.name != "index.jsonl"]
    return json.loads(files[0].read_text().splitlines()[0])


class TestGuardReceipt(unittest.TestCase):
    def setUp(self):
        os.environ.pop("COZEMPIC_NO_RECEIPTS", None)

    def test_guard_prune_emits_committed_receipt(self):
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
                os.environ.pop("COZEMPIC_NO_RECEIPTS", None)
                guard._emit_guard_receipt(**_args())
                rec = _only_receipt(home)
                validate_receipt(rec)
                self.assertEqual(rec["outcome"], "committed")
                self.assertEqual(rec["trigger"]["source"], "guard")
                self.assertEqual(rec["tokens"]["reclaimed"], 500)
                self.assertEqual(rec["model"]["name"], "claude-opus-4-8")
                self.assertNotIn("sessZ", json.dumps(rec))  # session id hashed

    def test_overflow_source_tagged(self):
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
                os.environ.pop("COZEMPIC_NO_RECEIPTS", None)
                guard._emit_guard_receipt(**_args(trigger_source="overflow"))
                self.assertEqual(_only_receipt(home)["trigger"]["source"], "overflow")

    def test_never_raises_on_bad_input(self):
        # a malformed pre_te must not propagate out of the guard hot path AND must
        # not persist a garbage receipt
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
                os.environ.pop("COZEMPIC_NO_RECEIPTS", None)
                guard._emit_guard_receipt(**_args(pre_te=None))  # must not raise
                recdir = Path(home) / ".cozempic" / "receipts"
                self.assertFalse(recdir.exists() and any(
                    p for p in recdir.glob("*.jsonl") if p.name != "index.jsonl"))

    def test_optout_suppresses(self):
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home, "USERPROFILE": home, "COZEMPIC_NO_RECEIPTS": "1"}):
                guard._emit_guard_receipt(**_args())
                self.assertFalse((Path(home) / ".cozempic" / "receipts").exists())


class TestGuardReceiptIntegration(unittest.TestCase):
    """Drive the REAL guard_prune_cycle -> deferred writer -> _emit_guard_receipt
    path (a deleted/mis-gated emit call must fail a test here, not slip through)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cz_guard_rcpt_"))
        self.session = self.tmp / "fake.jsonl"
        self.session.write_text('{"type":"user","message":{"content":"' + "x" * 100 + '"}}\n' * 500)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _drive(self, *, home, read_only_live, invoke_writer, trigger_source="guard"):
        from cozempic.guard import guard_prune_cycle
        from cozempic.team import TeamState

        team = MagicMock(spec=TeamState)
        team.is_empty.return_value = True
        team.team_name = None
        team.message_count = 0
        orig = [(0, {"type": "user"}, 100_000)]
        pruned = [(0, {"type": "user"}, 40_000)]  # 60% saving, past the futile floor
        results = [StrategyResult("tool-output-trim", [], 100_000, 40_000, 1, 0, 1, "")]
        totals = iter([100_000, 40_000])  # pre > post -> real token saving

        def est(*a, **k):
            try:
                total = next(totals)
            except StopIteration:
                total = 40_000
            return SimpleNamespace(total=total, context_pct=0.0, method="exact",
                                   confidence="high", model="claude-opus-4-8",
                                   context_window=200000)

        scratch = self.tmp / "scratch"
        scratch.mkdir(exist_ok=True)
        # Build the cycle (returns the deferred writer). Mirrors the guard's current
        # mock surface: load_messages_and_snapshot + _guard_tmp_root (post-#138).
        with patch("cozempic.guard._guard_tmp_root", return_value=scratch), \
                patch("cozempic.guard.load_messages_and_snapshot", return_value=(orig, MagicMock())), \
                patch("cozempic.guard.load_messages", return_value=orig), \
                patch("cozempic.guard.prune_with_team_protect", return_value=(pruned, results, team)), \
                patch("cozempic.tokens.estimate_session_tokens", side_effect=est), \
                patch("cozempic.tokens.calibrate_ratio", return_value=0.5):
            result = guard_prune_cycle(
                session_path=self.session, rx_name="gentle", config=None,
                auto_reload=False, read_only_live=read_only_live, trigger_source=trigger_source,
                session_id="testsess",
            )
        # Invoke the deferred writer (post-death) under patched I/O + temp HOME, and
        # patch record_savings — _SAVINGS_FILE is a frozen module constant a HOME
        # patch can't redirect, so it must be patched to spare the real ledger.
        writer = result.get("_deferred_writer")
        if invoke_writer and writer is not None:
            with patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}), \
                    patch("cozempic.guard._PruneLock"), \
                    patch("cozempic.guard.save_messages", return_value=None), \
                    patch("cozempic.guard.cleanup_old_backups"), \
                    patch("cozempic.helpers.record_savings"):
                os.environ.pop("COZEMPIC_NO_RECEIPTS", None)
                writer()
        recdir = Path(home) / ".cozempic" / "receipts"
        if not recdir.exists():
            return []
        return [p for p in recdir.glob("*.jsonl") if p.name != "index.jsonl"]

    def test_confirmed_write_emits_committed_receipt(self):
        with tempfile.TemporaryDirectory() as home:
            files = self._drive(home=home, read_only_live=False, invoke_writer=True)
            self.assertEqual(len(files), 1)
            rec = json.loads(files[0].read_text().splitlines()[0])
            validate_receipt(rec)
            self.assertEqual(rec["outcome"], "committed")
            self.assertEqual(rec["trigger"]["source"], "guard")
            self.assertEqual(rec["trigger"]["tier"], "gentle")

    def test_read_only_emits_nothing(self):
        with tempfile.TemporaryDirectory() as home:
            # read-only tier never writes the live file -> no committed receipt
            self.assertEqual(self._drive(home=home, read_only_live=True, invoke_writer=True), [])

    def test_unwritten_deferred_prune_emits_nothing(self):
        with tempfile.TemporaryDirectory() as home:
            # deferred writer not invoked (prune never persisted) -> no receipt
            self.assertEqual(self._drive(home=home, read_only_live=False, invoke_writer=False), [])

    def test_overflow_source_through_real_path(self):
        with tempfile.TemporaryDirectory() as home:
            files = self._drive(home=home, read_only_live=False, invoke_writer=True,
                                trigger_source="overflow")
            self.assertEqual(len(files), 1)
            rec = json.loads(files[0].read_text().splitlines()[0])
            self.assertEqual(rec["trigger"]["source"], "overflow")


if __name__ == "__main__":
    unittest.main()
