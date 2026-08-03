"""Tests for the static-HTML dashboard renderer + cli command — D3/D4."""

from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cozempic import cli
from cozempic.dashboard.render import (
    dashboard_path,
    render_dashboard,
    render_html,
    write_dashboard,
)
from cozempic.metrics import ClaudeMetricsAdapter, TriggerInfo, build_receipt
from cozempic.receipts import write_receipt
from cozempic.types import PrescriptionResult, StrategyResult


def _data(**over):
    d = {
        "lifetime": {"prunes_total": 2, "committed": 1, "deferred": 1, "noop": 0,
                     "failed": 0, "deferral_rate": 0.5, "tokens_reclaimed": 1500,
                     "bytes_reclaimed": 4096, "sessions": 1,
                     "first_ts": "2026-06-16T09:00:00Z", "last_ts": "2026-06-16T09:05:00Z"},
        "per_strategy": [{"id": "tool-output-trim", "tier": "standard",
                          "tokens_reclaimed": 1200, "bytes_reclaimed": 3000, "count": 1},
                         {"id": "thinking", "tier": "gentle",
                          "tokens_reclaimed": 300, "bytes_reclaimed": 1096, "count": 1}],
        "per_agent": [{"agent": "claude", "prunes": 2, "committed": 1, "tokens_reclaimed": 1500}],
        "by_tier": {"standard": 1, "gentle": 1},
        "per_session": [{"session": "abc123def456", "agent": "claude", "prunes": 2,
                         "tokens_reclaimed": 1500,
                         "timeline": [{"ts": "2026-06-16T09:00:00Z", "context_pct_after": 40.0},
                                      {"ts": "2026-06-16T09:05:00Z", "context_pct_after": 22.0}]}],
    }
    d.update(over)
    return d


class TestRenderHtml(unittest.TestCase):
    def test_well_formed_document(self):
        h = render_html(_data(), generated_ts="2026-06-16T09:05:00Z")
        self.assertTrue(h.startswith("<!DOCTYPE html>"))
        self.assertIn("</html>", h)
        self.assertIn("cozempic", h)

    def test_empty_state(self):
        h = render_html({"lifetime": {"prunes_total": 0}}, generated_ts="t")
        self.assertIn("No prunes recorded yet", h)
        self.assertTrue(h.startswith("<!DOCTYPE html>"))

    def test_renders_metrics_and_chart(self):
        h = render_html(_data(), generated_ts="t")
        self.assertIn("Tokens Reclaimed", h)
        self.assertIn("1.5K", h)  # 1500 tokens formatted
        self.assertIn("Tool Output Trim", h)  # strategy slug -> Title Case name
        self.assertIn("<svg", h)  # session sparkline rendered
        self.assertIn("polyline", h)

    def test_self_contained_no_external_refs(self):
        h = render_html(_data(), generated_ts="t")
        for needle in ("http://", "https://", "cdn", "<script"):
            self.assertNotIn(needle, h)  # no network deps, no JS

    def test_escapes_dynamic_values(self):
        evil = _data(per_strategy=[{"id": "<script>x</script>", "tier": "t",
                                    "tokens_reclaimed": 10, "bytes_reclaimed": 1, "count": 1}])
        h = render_html(evil, generated_ts="t")
        self.assertNotIn("<script>x</script>", h)
        self.assertIn("&lt;script&gt;x&lt;/script&gt;", h)

    def test_sparkline_na_for_short_timeline(self):
        d = _data(per_session=[{"session": "s", "agent": "claude", "prunes": 1,
                                "tokens_reclaimed": 0,
                                "timeline": [{"ts": "t", "context_pct_after": 5.0}]}])
        h = render_html(d, generated_ts="t")
        self.assertIn("—", h)  # single point -> no line

    def test_non_string_tier_key_does_not_crash(self):
        # D2 can emit a numeric tier key if a receipt's trigger.tier is a number
        h = render_html(_data(by_tier={5: 1, "gentle": 2}), generated_ts="t")
        self.assertIn("Gentle", h)  # no TypeError on sorted(); tier slug -> Title Case

    def test_all_zero_bars_no_divzero(self):
        d = _data(per_strategy=[{"id": "a", "tier": "t", "tokens_reclaimed": 0,
                                 "bytes_reclaimed": 0, "count": 1}])
        h = render_html(d, generated_ts="t")
        self.assertIn("width:0.0%", h)

    def test_negative_values_formatted(self):
        # negative reclaim formats correctly (tested at the formatter level since
        # the summary cards were removed as redundant with the lifetime band)
        from cozempic.dashboard.render import _fmt_bytes, _fmt_tokens

        self.assertEqual(_fmt_tokens(-5000), "-5.0K")
        self.assertEqual(_fmt_bytes(-2048), "-2.0 KB")

    def test_deferral_rate_none_safe(self):
        d = _data()
        d["lifetime"]["deferral_rate"] = None
        h = render_html(d, generated_ts="t")  # no TypeError
        self.assertIn("0%", h)

    def test_sparkline_clamps_negative_pct_into_viewbox(self):
        import re

        d = _data(per_session=[{"session": "s", "agent": "claude", "prunes": 2,
                                "tokens_reclaimed": 0,
                                "timeline": [{"ts": "a", "context_pct_after": -10.0},
                                             {"ts": "b", "context_pct_after": -50.0}]}])
        h = render_html(d, generated_ts="t")
        m = re.search(r'points="([^"]+)"', h)
        self.assertIsNotNone(m)
        ys = [float(pt.split(",")[1]) for pt in m.group(1).split()]
        self.assertTrue(all(0 <= y <= 24 for y in ys))  # within the 24px viewBox


class TestWriteDashboard(unittest.TestCase):
    def test_atomic_write_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_dashboard("<html>ok</html>", base_dir=Path(tmp))
            self.assertEqual(p, dashboard_path(Path(tmp)))
            self.assertEqual(p.read_text(), "<html>ok</html>")
            # no leftover temp files
            leftovers = [x for x in p.parent.iterdir() if x.name.startswith(".dashboard-")]
            self.assertEqual(leftovers, [])

    def test_render_dashboard_from_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ.pop("COZEMPIC_NO_RECEIPTS", None)
            base = Path(tmp)
            s = StrategyResult("tool-output-trim", [], 1000, 200, 1, 0, 1, "")
            res = PrescriptionResult("standard", [s], 5000, 4000, 10, 9, 2000, 1500,
                                     "exact", "claude-opus-4-8", 200000)
            rec = build_receipt(res, adapter=ClaudeMetricsAdapter(), session_id="s1",
                                trigger=TriggerInfo("manual", "standard", "standard"),
                                ts="2026-06-16T09:00:00Z", receipt_id="r1")
            write_receipt(rec, base_dir=base)
            h = render_dashboard(base, generated_ts="now")
            self.assertIn("Tool Output Trim", h)
            self.assertIn("500", h)  # tokens reclaimed shows up somewhere

    def test_write_failure_cleans_temp_and_reraises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("os.replace", side_effect=OSError("boom")):
                with self.assertRaises(OSError):
                    write_dashboard("<html>x</html>", base_dir=Path(tmp))
            leftovers = list(Path(tmp).glob(".dashboard-*"))
            self.assertEqual(leftovers, [])  # temp file cleaned on failure


class TestCliDashboardCommand(unittest.TestCase):
    def test_command_writes_file_no_open(self):
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
                with patch("webbrowser.open") as wb:
                    cli.cmd_dashboard(argparse.Namespace(no_open=True))
                    wb.assert_not_called()  # --no-open suppresses browser
                out = Path(home) / ".cozempic" / "dashboard.html"
                self.assertTrue(out.exists())
                self.assertTrue(out.read_text().startswith("<!DOCTYPE html>"))

    def test_default_opens_browser(self):
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
                with patch("webbrowser.open", return_value=True) as wb:
                    cli.cmd_dashboard(argparse.Namespace(no_open=False))
                    wb.assert_called_once()

    def test_empty_prints_no_prunes(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
                with patch("webbrowser.open"):
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        cli.cmd_dashboard(argparse.Namespace(no_open=True))
                    self.assertIn("No prunes recorded yet", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
