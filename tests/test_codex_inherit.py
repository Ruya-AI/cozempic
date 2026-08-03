"""D5: a Codex adapter inherits the metrics contract + dashboard with zero changes.

Proves the seam is genuinely agent-agnostic — a CodexMetricsAdapter emits the SAME
PruneReceipt, which flows through write -> load -> aggregate -> render unchanged,
and the --agent filter isolates one agent.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cozempic import cli
from cozempic.dashboard.aggregate import aggregate, load_receipts
from cozempic.dashboard.render import render_dashboard
from cozempic.metrics import (
    ClaudeMetricsAdapter,
    CodexMetricsAdapter,
    MetricsAdapter,
    TriggerInfo,
    build_receipt,
    validate_receipt,
)
from cozempic.receipts import write_receipt
from cozempic.types import PrescriptionResult, StrategyResult


def _res():
    s = StrategyResult("tool-output-offload", [], 1000, 200, 1, 0, 1, "")
    return PrescriptionResult("standard", [s], 5000, 4000, 10, 9, 3000, 1000,
                              "exact", "gpt-5.5", 272000)


def _codex_receipt(session="cx1", ts="2026-06-16T09:00:00Z"):
    return build_receipt(
        _res(), adapter=CodexMetricsAdapter(agent_version="0.139.0"),
        session_id=session, trigger=TriggerInfo("guard", "standard", "standard"),
        ts=ts, receipt_id="rx" + session, mode="edit_resume",
    )


class TestCodexAdapter(unittest.TestCase):
    def test_satisfies_protocol(self):
        self.assertIsInstance(CodexMetricsAdapter(), MetricsAdapter)

    def test_count_tokens_reads_real_nested_token_count(self):
        # the REAL codex 0.139 shape: info.total_token_usage.total_tokens + window
        a = CodexMetricsAdapter()
        rollout = [
            {"type": "response_item", "payload": {"type": "message"}},
            {"type": "event_msg", "payload": {"type": "token_count",
             "info": {"total_token_usage": {"total_tokens": 24590},
                      "model_context_window": 258400}}},
        ]
        tc = a.count_tokens(rollout)
        self.assertEqual(tc.total, 24590)
        self.assertEqual(tc.method, "exact")
        self.assertEqual(a.context_window(rollout), 258400)  # real window, not the default

    def test_count_tokens_flat_alias_and_bool_guard(self):
        a = CodexMetricsAdapter()
        # flat total_tokens alias (older/synthetic) still resolves
        self.assertEqual(a.count_tokens([{"payload": {"type": "token_count", "total_tokens": 700}}]).total, 700)
        # a boolean is not a real count -> heuristic fallback
        self.assertEqual(a.count_tokens([{"payload": {"type": "token_count", "total": True}}]).method, "heuristic")

    def test_context_window_default_without_event(self):
        self.assertEqual(CodexMetricsAdapter().context_window([{"payload": {"type": "message"}}]), 272000)

    def test_count_tokens_heuristic_fallback(self):
        a = CodexMetricsAdapter()
        tc = a.count_tokens([{"type": "response_item", "payload": {"type": "message"}}])
        self.assertGreaterEqual(tc.total, 0)
        self.assertEqual(tc.method, "heuristic")

    def test_entry_bytes_on_rollout_line(self):
        self.assertGreater(CodexMetricsAdapter().entry_bytes({"type": "x", "payload": {}}), 0)

    def test_build_receipt_stamps_codex(self):
        r = _codex_receipt()
        validate_receipt(r)  # same contract guard as Claude
        self.assertEqual(r["agent"], {"name": "codex", "version": "0.139.0",
                                      "adapter_schema_version": "1"})
        self.assertEqual(r["tokens"]["reclaimed"], 2000)
        self.assertEqual(r["model"]["context_window"], 272000)


class TestDashboardInheritance(unittest.TestCase):
    def setUp(self):
        os.environ.pop("COZEMPIC_NO_RECEIPTS", None)

    def _claude_receipt(self, session="cl1"):
        s = StrategyResult("tool-output-trim", [], 1000, 200, 1, 0, 1, "")
        res = PrescriptionResult("standard", [s], 5000, 4000, 10, 9, 2000, 1500,
                                 "exact", "claude-opus-4-8", 200000)
        return build_receipt(res, adapter=ClaudeMetricsAdapter(), session_id=session,
                             trigger=TriggerInfo("manual", "standard", "standard"),
                             ts="2026-06-16T08:00:00Z", receipt_id="rc" + session)

    def test_codex_receipt_aggregates_and_renders_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_receipt(self._claude_receipt(), base_dir=base)
            write_receipt(_codex_receipt(), base_dir=base)
            data = aggregate(load_receipts(base))
            agents = {a["agent"] for a in data["per_agent"]}
            self.assertEqual(agents, {"claude", "codex"})
            html = render_dashboard(base, generated_ts="now")
            self.assertIn("Codex", html)  # agent name rendered Title Case
            self.assertIn("Claude", html)

    def test_agent_filter_isolates_codex(self):
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
                write_receipt(self._claude_receipt(session="claudesess"))
                write_receipt(_codex_receipt(session="codexsess"))
                # data-level isolation: both load, filter keeps only codex
                allrecs = load_receipts()
                self.assertEqual(len(allrecs), 2)
                codex_only = [r for r in allrecs if r["agent"]["name"] == "codex"]
                data = aggregate(codex_only)
                self.assertEqual({a["agent"] for a in data["per_agent"]}, {"codex"})
                self.assertEqual(data["lifetime"]["prunes_total"], 1)
                # render via cmd_dashboard and prove claude did NOT leak through
                with patch("webbrowser.open"):
                    cli.cmd_dashboard(argparse.Namespace(no_open=True, agent="codex"))
                html = (Path(home) / ".cozempic" / "dashboard.html").read_text()
                self.assertIn("agent=codex", html)
                self.assertIn("codex", html)
                self.assertNotIn("claude", html)  # the real isolation check

    def test_agent_filter_unknown_shows_empty(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
                write_receipt(_codex_receipt(session="cx"))
                with patch("webbrowser.open"):
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        cli.cmd_dashboard(argparse.Namespace(no_open=True, agent="nonexistent"))
                    self.assertIn("No prunes recorded for agent 'nonexistent'", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
