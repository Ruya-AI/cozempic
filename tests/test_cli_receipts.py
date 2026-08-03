"""D1 integration: the cli prune helper actually emits a receipt under ~/.cozempic."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cozempic import cli
from cozempic.metrics import validate_receipt
from cozempic.types import PrescriptionResult, StrategyResult


def _pr():
    s = StrategyResult("tool-output-trim", [], 1000, 200, 1, 0, 1, "")
    return PrescriptionResult("standard", [s], 5000, 4000, 10, 9, 2000, 1500,
                              "exact", "claude-opus-4-8", 200000)


class TestCliEmitsReceipt(unittest.TestCase):
    def test_committed_receipt_written_under_home(self):
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
                os.environ.pop("COZEMPIC_NO_RECEIPTS", None)
                cli._emit_prune_receipt(
                    Path(home) / "sess-xyz.jsonl", _pr(),
                    source="manual", outcome="committed", cwd=home,
                )
                rec_dir = Path(home) / ".cozempic" / "receipts"
                session_files = [p for p in rec_dir.glob("*.jsonl") if p.name != "index.jsonl"]
                self.assertEqual(len(session_files), 1)
                rec = json.loads(session_files[0].read_text().splitlines()[0])
                validate_receipt(rec)  # cli-emitted receipt passes the contract guard
                self.assertEqual(rec["outcome"], "committed")
                self.assertEqual(rec["trigger"]["source"], "manual")
                self.assertEqual(rec["trigger"]["tier"], "standard")  # known tier, not "custom"
                self.assertEqual(rec["tokens"]["reclaimed"], 500)
                # the real session id must not appear raw
                self.assertNotIn("sess-xyz", json.dumps(rec))

    def test_deferred_path_emits_valid_receipt(self):
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
                os.environ.pop("COZEMPIC_NO_RECEIPTS", None)
                cli._emit_prune_receipt(
                    Path(home) / "s.jsonl", _pr(), source="manual",
                    outcome="deferred", cwd=home, defer_reason="prune_lock",
                )
                rec_dir = Path(home) / ".cozempic" / "receipts"
                sf = [p for p in rec_dir.glob("*.jsonl") if p.name != "index.jsonl"][0]
                rec = json.loads(sf.read_text().splitlines()[0])
                validate_receipt(rec)
                self.assertEqual(rec["outcome"], "deferred")
                self.assertTrue(rec["validation"]["deferred"])
                self.assertEqual(rec["validation"]["defer_reason"], "prune_lock")

    def test_helper_never_raises_on_bad_pr(self):
        # a malformed result must not propagate out of the cli helper, and must
        # not persist a garbage receipt
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
                os.environ.pop("COZEMPIC_NO_RECEIPTS", None)
                cli._emit_prune_receipt(None, object(), source="manual", outcome="committed")
                rec_dir = Path(home) / ".cozempic" / "receipts"
                # no parseable garbage receipt persisted (dir absent or empty)
                self.assertFalse(rec_dir.exists() and any(rec_dir.iterdir()))


if __name__ == "__main__":
    unittest.main()
