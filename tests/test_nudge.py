"""1.8.22 — tiered nudge (cozempic nudge, Stop hook). Non-blocking, once-per-tier."""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _transcript(tmp: Path, total: int, model: str = "claude-sonnet-4-6") -> Path:
    p = tmp / "t.jsonl"
    rec = {"type": "assistant", "message": {"role": "assistant", "model": model,
           "content": [{"type": "text", "text": "x"}],
           "usage": {"input_tokens": total, "cache_creation_input_tokens": 0,
                     "cache_read_input_tokens": 0, "output_tokens": 0}}}
    p.write_text(json.dumps(rec) + "\n")
    return p


class TestNudge(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cozempic_nudge_"))
        self.home = Path(tempfile.mkdtemp(prefix="cozempic_home_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        shutil.rmtree(self.home, ignore_errors=True)

    def _run(self, total: int, session: str = "s1", env: dict | None = None):
        from cozempic.cli import cmd_nudge
        t = _transcript(self.tmp, total)
        payload = json.dumps({"transcript_path": str(t), "session_id": session})
        out = io.StringIO()
        # These tests isolate via patch(Path.home)=self.home, so the read side must
        # NOT inherit an ambient CLAUDE_CONFIG_DIR (e.g. the conftest autouse fixture)
        # or get_claude_dir would diverge from where record_session wrote the sidecar.
        e = {k: v for k, v in os.environ.items()
             if not k.startswith("COZEMPIC_NUDGE") and k != "CLAUDE_CONFIG_DIR"}
        if env:
            e.update(env)
        with patch("sys.stdin", io.StringIO(payload)), patch("sys.stdout", out), \
             patch("pathlib.Path.home", return_value=self.home), \
             patch.dict(os.environ, e, clear=True):
            cmd_nudge(None)
        s = out.getvalue()
        return json.loads(s)["systemMessage"] if s.strip() else None

    def test_silent_below_25(self):
        self.assertIsNone(self._run(100_000))  # 10% of 1M

    def test_25_tier_informational(self):
        m = self._run(260_000, "s25")  # 26%
        self.assertIsNotNone(m)
        self.assertIn("26%", m)
        self.assertIn("Optional", m)
        self.assertNotIn("auto-reload", m.lower())  # 25% has no reload-pending line

    def test_55_tier_actionable(self):
        m = self._run(560_000, "s55")  # 56%
        self.assertIsNotNone(m)
        self.assertIn("56%", m)
        # Must point at the SLASH command, not the bare shell command: this is an
        # in-TUI systemMessage, and a bare `cozempic reload` typed in the prompt goes
        # to the model (does NOT run the CLI). Regression guard for that UX bug.
        self.assertIn("/cozempic reload", m)
        self.assertNotRegex(m, r"(?<!/)`cozempic reload`")
        self.assertIn("pause between turns", m.lower())

    def test_all_tiers_use_slash_command_not_bare(self):
        # No tier may tell the user to run the bare `cozempic reload` (it would be
        # routed to the model in the Claude Code TUI, not the shell).
        for size, sid in ((260_000, "a25"), (560_000, "a55"), (820_000, "a80")):
            m = self._run(size, sid)
            self.assertIsNotNone(m)
            if "reload" in m:
                self.assertNotRegex(
                    m, r"(?<!/)`cozempic reload`",
                    f"tier message uses bare `cozempic reload` (should be /cozempic reload): {m!r}")

    def test_80_tier_urgent_no_emergency_word(self):
        m = self._run(820_000, "s80")  # 82%
        self.assertIsNotNone(m)
        self.assertIn("82%", m)
        self.assertIn("autocompact wall", m)
        self.assertNotIn("emergency", m.lower())  # user explicitly removed "emergency"

    def test_once_per_tier(self):
        self.assertIsNotNone(self._run(560_000, "sx"))
        self.assertIsNone(self._run(560_000, "sx"))  # silent on repeat in same tier

    def test_rearm_after_drop_then_recross(self):
        self.assertIsNotNone(self._run(560_000, "sy"))   # cross 55 → fires
        self.assertIsNone(self._run(560_000, "sy"))       # latched → silent
        self.assertIsNone(self._run(100_000, "sy"))       # dropped to 10% (< all tiers) → silent, clears latch
        self.assertIsNotNone(self._run(560_000, "sy"))    # re-cross 55 → re-fires

    def test_hysteresis_no_refire_on_small_dip(self):
        # 56%→54%→56% must fire the 55 nudge only ONCE (dip stays within the
        # hysteresis band, so the tier doesn't re-arm).
        self.assertIsNotNone(self._run(560_000, "hy"))  # 56% fires 55
        self.assertIsNone(self._run(540_000, "hy"))      # 54% — within band, silent
        self.assertIsNone(self._run(560_000, "hy"))      # back to 56% — still latched

    def test_jump_then_dip_no_stale_lower_tier(self):
        # Firing 55 latches 25 too; a later dip to the 25 band must NOT fire a
        # stale 25% FYI (context went down, not up).
        self.assertIsNotNone(self._run(560_000, "jd"))  # 56% fires 55 (+latches 25)
        self.assertIsNone(self._run(300_000, "jd"))      # 30% — 25 already latched, silent

    def test_off_env_silences(self):
        self.assertIsNone(self._run(560_000, "soff", env={"COZEMPIC_NUDGE_OFF": "1"}))

    def test_non_blocking_bad_stdin(self):
        from cozempic.cli import cmd_nudge
        out = io.StringIO()
        with patch("sys.stdin", io.StringIO("not json")), patch("sys.stdout", out), \
             patch("pathlib.Path.home", return_value=self.home):
            cmd_nudge(None)  # must not raise
        self.assertEqual(out.getvalue().strip(), "")

    def test_non_blocking_wrong_shape_stdin(self):
        # valid JSON but not an object (list/int/str) must not crash — "always
        # exit 0" contract (the Stop hook swallows stderr, but the contract holds).
        from cozempic.cli import cmd_nudge
        for payload in ("[1,2,3]", "42", '"hello"', "null"):
            out = io.StringIO()
            with patch("sys.stdin", io.StringIO(payload)), patch("sys.stdout", out), \
                 patch("pathlib.Path.home", return_value=self.home):
                cmd_nudge(None)  # must not raise
            self.assertEqual(out.getvalue().strip(), "", payload)

    def test_non_blocking_malformed_state_file(self):
        # A corrupt/foreign nudge-state.json must not crash the nudge.
        from cozempic.cli import cmd_nudge
        sf = self.home / ".claude" / "cozempic-metrics" / "nudge-state.json"
        sf.parent.mkdir(parents=True, exist_ok=True)
        for bad in ('[1,2,3]', '{"s1": "not-a-dict"}', '{"s1": {"tiers_fired": ["x","y"]}}'):
            sf.write_text(bad)
            t = _transcript(self.tmp, 560_000)
            payload = json.dumps({"transcript_path": str(t), "session_id": "s1"})
            out = io.StringIO()
            import os
            e = {k: v for k, v in os.environ.items()
                 if not k.startswith("COZEMPIC_NUDGE") and k != "CLAUDE_CONFIG_DIR"}
            with patch("sys.stdin", io.StringIO(payload)), patch("sys.stdout", out), \
                 patch("pathlib.Path.home", return_value=self.home), \
                 patch.dict(os.environ, e, clear=True):
                cmd_nudge(None)  # must not raise (and should still fire the 55 nudge)
            self.assertIn("56%", out.getvalue(), bad)

    def test_state_file_honors_claude_config_dir(self):
        profile = self.tmp / "profile"
        profile.mkdir()
        self._run(560_000, "profile-state", env={"CLAUDE_CONFIG_DIR": str(profile)})
        self.assertTrue((profile / "cozempic-metrics" / "nudge-state.json").exists())
        self.assertFalse((self.home / ".claude" / "cozempic-metrics" / "nudge-state.json").exists())

    def test_tiers_from_sidecar(self):
        # The guard records its resolved reload-tier fractions; the nudge fires at
        # THOSE points (so a raised threshold moves the nudges), not the defaults.
        from cozempic.session import record_session
        with patch("pathlib.Path.home", return_value=self.home), \
             patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                     if k != "CLAUDE_CONFIG_DIR"}, clear=True):
            record_session("sc1", "/tmp/x", 1_000_000, nudge_tiers=[0.40, 0.70, 0.90])
        # 30% < custom SOFT 40% → silent (default 25% tier would have fired)
        self.assertIsNone(self._run(300_000, "sc1"))
        # 42% >= custom SOFT 40% → fires
        self.assertIsNotNone(self._run(420_000, "sc1"))

    def test_inflight_softens_copy(self):
        # When a background Agent is in flight, the 55% copy must NOT promise an
        # idle reload — it says the reload waits for agents/tools to finish.
        from cozempic.cli import cmd_nudge
        p = self.tmp / "t.jsonl"
        assist = {"type": "assistant", "message": {"role": "assistant",
                  "model": "claude-sonnet-4-6", "content": [{"type": "text", "text": "x"}],
                  "usage": {"input_tokens": 560_000, "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0, "output_tokens": 0}}}
        agent = {"type": "user", "message": {"role": "user", "content": [
                 {"type": "tool_result",
                  "content": "Async agent launched successfully. agentId: zz1 (internal ID)"}]}}
        p.write_text(json.dumps(assist) + "\n" + json.dumps(agent) + "\n")
        out = io.StringIO()
        e = {k: v for k, v in os.environ.items() if not k.startswith("COZEMPIC_NUDGE")}
        with patch("sys.stdin", io.StringIO(json.dumps({"transcript_path": str(p), "session_id": "if1"}))), \
             patch("sys.stdout", out), patch("pathlib.Path.home", return_value=self.home), \
             patch.dict(os.environ, e, clear=True):
            cmd_nudge(None)
        m = json.loads(out.getvalue())["systemMessage"]
        self.assertIn("agents/tools finish", m)
        self.assertNotIn("pause between turns", m)

    def test_projected_pct_from_armed_sentinel(self):
        # when the daemon armed with a projected_pct, the 55% message shows it
        from cozempic import guard
        with patch("cozempic.guard._guard_tmp_root", return_value=self.tmp):
            guard.write_armed("sp1", None, 55, 38.0)
            m = self._run(560_000, "sp1")
        self.assertIsNotNone(m)
        self.assertIn("~38%", m)


if __name__ == "__main__":
    unittest.main()
