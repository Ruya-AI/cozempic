"""Tests for the lifetime savings ledger band on the dashboard."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cozempic import cli
from cozempic.dashboard.lifetime import load_lifetime
from cozempic.dashboard.render import render_html


def _ledger_file(tmp, data):
    p = Path(tmp) / ".cozempic_savings.json"
    p.write_text(json.dumps(data))
    return p


class TestLoadLifetime(unittest.TestCase):
    def test_normalizes_and_computes_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _ledger_file(tmp, {"tokens_saved": 456170685, "tokens_processed": 2601916262,
                                   "prune_count": 3309, "turns_gained": 23394, "since": "2026-04-09"})
            lg = load_lifetime(p)
            self.assertEqual(lg["tokens_saved"], 456170685)
            self.assertEqual(lg["prune_count"], 3309)
            self.assertEqual(lg["savings_rate_pct"], 17.5)  # 456.2M / 2.60B
            self.assertEqual(lg["since"], "2026-04-09")

    def test_missing_file_returns_none(self):
        self.assertIsNone(load_lifetime(Path("/nonexistent/x.json")))

    def test_garbage_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.json"
            p.write_text("{ not json")
            self.assertIsNone(load_lifetime(p))

    def test_zero_or_missing_saved_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_lifetime(_ledger_file(tmp, {"tokens_saved": 0})))
            self.assertIsNone(load_lifetime(_ledger_file(tmp, {"prune_count": 5})))

    def test_no_processed_means_no_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            lg = load_lifetime(_ledger_file(tmp, {"tokens_saved": 1000}))
            self.assertIsNone(lg["savings_rate_pct"])

    def test_non_int_fields_coerced(self):
        with tempfile.TemporaryDirectory() as tmp:
            lg = load_lifetime(_ledger_file(tmp, {"tokens_saved": 1000, "prune_count": "x",
                                                  "turns_gained": None, "since": 5}))
            self.assertEqual(lg["prune_count"], 0)
            self.assertEqual(lg["turns_gained"], 0)
            self.assertIsNone(lg["since"])

    def test_rate_over_100_suppressed(self):
        # corrupt ledger (processed < saved) -> impossible rate dropped, band stays
        with tempfile.TemporaryDirectory() as tmp:
            lg = load_lifetime(_ledger_file(tmp, {"tokens_saved": 456170685, "tokens_processed": 1000}))
            self.assertIsNotNone(lg)
            self.assertIsNone(lg["savings_rate_pct"])

    def test_float_fields_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            lg = load_lifetime(_ledger_file(tmp, {"tokens_saved": 1000.7, "tokens_processed": 5000.0}))
            self.assertIsNotNone(lg)  # float must NOT drop the whole band
            self.assertEqual(lg["tokens_saved"], 1000)
            self.assertEqual(lg["savings_rate_pct"], 20.0)


class TestSessionMultiplier(unittest.TestCase):
    def test_multiplier_uses_tracked_prunes(self):
        with tempfile.TemporaryDirectory() as tmp:
            # rate=17.5%, tracked_prunes/sessions=20/10=2 -> 1 + 2*0.175 = 1.35
            lg = load_lifetime(_ledger_file(tmp, {"tokens_saved": 175, "tokens_processed": 1000,
                                                  "tracked_prunes": 20, "sessions": 10}))
            self.assertEqual(lg["session_multiplier_x"], 1.35)
            self.assertEqual(lg["sessions"], 10)

    def test_lifetime_prune_count_NOT_used(self):
        # the real-world mismatch: huge lifetime prune_count, tiny forward window.
        # Must use tracked_prunes (10/5=2 -> 1.35x), NOT prune_count (3309/5 -> 117x).
        with tempfile.TemporaryDirectory() as tmp:
            lg = load_lifetime(_ledger_file(tmp, {"tokens_saved": 175, "tokens_processed": 1000,
                                                  "prune_count": 3309, "tracked_prunes": 10,
                                                  "sessions": 5}))
            self.assertEqual(lg["session_multiplier_x"], 1.35)
            self.assertLess(lg["session_multiplier_x"], 2)  # never the absurd 117x

    def test_multiplier_none_below_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            lg = load_lifetime(_ledger_file(tmp, {"tokens_saved": 175, "tokens_processed": 1000,
                                                  "tracked_prunes": 5, "sessions": 3}))
            self.assertIsNone(lg["session_multiplier_x"])  # too few sessions to be stable

    def test_no_tracking_means_no_multiplier(self):
        # existing ledger with lifetime prune_count but no forward tracking yet
        with tempfile.TemporaryDirectory() as tmp:
            lg = load_lifetime(_ledger_file(tmp, {"tokens_saved": 175, "tokens_processed": 1000,
                                                  "prune_count": 3309}))
            self.assertEqual(lg["sessions"], 0)
            self.assertIsNone(lg["session_multiplier_x"])  # honest-blank until measured


class TestRecordSavingsSessions(unittest.TestCase):
    def test_distinct_pruned_sessions_counted(self):
        from cozempic import helpers

        with tempfile.TemporaryDirectory() as tmp:
            sav = Path(tmp) / "sav.json"
            with patch.object(helpers, "_SAVINGS_FILE", sav), \
                 patch.dict(os.environ, {"COZEMPIC_NO_TELEMETRY": "1"}):
                helpers.record_savings(100, total_tokens=1000, turn_count=10, session_id="A")
                helpers.record_savings(100, total_tokens=1000, turn_count=10, session_id="A")  # repeat
                helpers.record_savings(100, total_tokens=1000, turn_count=10, session_id="B")
                helpers.record_savings(100, total_tokens=1000, turn_count=10)  # no session id
            data = json.loads(sav.read_text())
            self.assertEqual(data["prune_count"], 4)  # lifetime: all 4 prunes
            self.assertEqual(data["tracked_prunes"], 3)  # forward: only the 3 with a session_id
            self.assertEqual(data["sessions"], 2)  # A and B distinct; repeat A not recounted
            self.assertEqual(len(data["_pruned_session_hashes"]), 2)
            # raw session id never stored (hashed)
            self.assertNotIn("A", data["_pruned_session_hashes"])


class TestLifetimeBand(unittest.TestCase):
    _LEDGER = {"tokens_saved": 456170685, "tokens_processed": 2601916262,
               "prune_count": 3309, "turns_gained": 23394, "since": "2026-04-09",
               "savings_rate_pct": 17.5}

    def test_band_rendered_with_ledger(self):
        h = render_html({"lifetime": {"prunes_total": 0}}, generated_ts="t", ledger=self._LEDGER)
        self.assertIn("456.2M", h)
        self.assertIn("Tokens Reclaimed", h)
        self.assertIn("3,309", h)
        self.assertIn("~23,394", h)  # tilde = estimate, not a hard count
        self.assertIn("Est. Extra Turns", h)
        self.assertIn("17.5%", h)
        self.assertIn("Reclaimed of Processed", h)  # honest label, not "avg reclaimed"
        self.assertIn("since 2026-04-09", h)
        # band shows even when there are no receipts yet
        self.assertIn("No prunes recorded yet", h)

    def test_no_band_without_ledger(self):
        h = render_html({"lifetime": {"prunes_total": 0}}, generated_ts="t", ledger=None)
        self.assertNotIn('<section class="lifetime">', h)  # no band section emitted
        self.assertNotIn("Running totals from", h)  # band footer absent (no band)

    def test_band_shows_session_multiplier_chip(self):
        h = render_html({"lifetime": {"prunes_total": 0}}, generated_ts="t",
                        ledger={"tokens_saved": 1000, "session_multiplier_x": 1.35})
        self.assertIn("1.35×", h)
        self.assertIn("Longer Per Pruned Session", h)

    def test_band_escapes_since(self):
        h = render_html({"lifetime": {"prunes_total": 0}}, generated_ts="t",
                        ledger={"tokens_saved": 10, "since": "<b>x</b>"})
        self.assertNotIn("<b>x</b>", h)
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", h)


class TestCliLifetimeLine(unittest.TestCase):
    def test_prints_lifetime_when_ledger_present(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
                _ledger_file(home, {"tokens_saved": 456170685, "prune_count": 3309,
                                    "since": "2026-04-09"})
                with patch("webbrowser.open"):
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        cli.cmd_dashboard(argparse.Namespace(no_open=True, agent=None))
                    out = buf.getvalue()
                    self.assertIn("456.2M tokens reclaimed", out)
                    self.assertIn("3,309 prunes", out)

    def test_agent_filter_suppresses_global_ledger(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
                _ledger_file(home, {"tokens_saved": 456170685, "prune_count": 3309})
                with patch("webbrowser.open"):
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        cli.cmd_dashboard(argparse.Namespace(no_open=True, agent="codex"))
                    # the global ledger must NOT show next to an agent-scoped view
                    self.assertNotIn("Lifetime:", buf.getvalue())
                html = (Path(home) / ".cozempic" / "dashboard.html").read_text()
                self.assertNotIn('<section class="lifetime">', html)


if __name__ == "__main__":
    unittest.main()
