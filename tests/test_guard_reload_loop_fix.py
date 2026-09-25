"""1.8.21 — confirmed-write reload-loop fix.

Two complementary guards stop a session that re-bloats to the HARD threshold
from churning kill→resume→re-bloat forever (each cycle a confirmed prune+ping,
invisible to the per-process circuit breaker which is reborn at 0 on respawn):

  * P0a — post-prune TOKEN-PROGRESS gate: if a prune frees real bytes but reduces
    *tokens* by less than _MIN_PRUNE_RATIO, skip the reload (futile) instead of
    resuming a session whose token count barely moved (re-fires HARD at once).
    Gating on PROGRESS (not an absolute floor) does NOT block a prune that frees
    real headroom but lands just above the trigger.
  * P0b — disk-backed reload-rate ledger: cap reloads-per-window per session,
    surviving daemon respawn (the in-process breaker cannot).

All tests patch _guard_tmp_root to a per-test scratch dir so the disk reload
ledger never leaks into the shared /tmp (which would make order/timing-dependent
flakes across runs).
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


def _make_session_file(tmpdir: Path, size_bytes: int = 100_000) -> Path:
    path = tmpdir / "fake_session.jsonl"
    line = '{"type":"user","message":{"content":"' + "x" * 100 + '"}}\n'
    n = max(1, size_bytes // len(line.encode()))
    path.write_text(line * n)
    return path


class TestPostPruneTokenProgressGate(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cozempic_loopfix_"))
        self.session_path = _make_session_file(self.tmpdir, 100_000)
        self.scratch = Path(tempfile.mkdtemp(prefix="cozempic_tmproot_"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        shutil.rmtree(self.scratch, ignore_errors=True)

    def _run(self, pre_total, post_total, savings_calls):
        from cozempic.team import TeamState
        from cozempic.guard import guard_prune_cycle
        team = MagicMock(spec=TeamState)
        team.is_empty.return_value = True
        team.team_name = None
        team.message_count = 0
        orig = [(0, {"type": "user"}, 100_000)]
        pruned = [(0, {"type": "user"}, 40_000)]
        _totals = iter([pre_total, post_total])

        def _est(*a, **k):
            try:
                return MagicMock(total=next(_totals))
            except StopIteration:
                return MagicMock(total=post_total)

        with patch("cozempic.guard._guard_tmp_root", return_value=self.scratch), \
             patch("cozempic.guard.load_messages", return_value=orig), \
             patch("cozempic.guard.prune_with_team_protect", return_value=(pruned, {}, team)), \
             patch("cozempic.guard.save_messages", side_effect=lambda *a, **k: None), \
             patch("cozempic.guard.snapshot_session", return_value=MagicMock()), \
             patch("cozempic.guard._terminate_and_resume",
                   side_effect=lambda *a, **k: k.get("write_pruned", lambda: None)()), \
             patch("cozempic.helpers.record_savings",
                   side_effect=lambda *a, **k: savings_calls.append(a)), \
             patch("cozempic.tokens.estimate_session_tokens", side_effect=_est), \
             patch("cozempic.tokens.calibrate_ratio", return_value=0.5):
            return guard_prune_cycle(
                session_path=self.session_path, rx_name="aggressive", config=None,
                auto_reload=True, claude_pid=999999, session_id="abcdef012345",
            )

    def test_gate_fires_when_token_progress_below_min_ratio(self):
        # saved 5k of 100k = 5% < 10% _MIN_PRUNE_RATIO -> futile, no reload/ping
        calls = []
        r = self._run(pre_total=100_000, post_total=95_000, savings_calls=calls)
        self.assertTrue(r.get("token_progress_insufficient"))
        self.assertTrue(r.get("futile_reload_skipped"))
        self.assertFalse(r.get("reloading"))
        self.assertEqual(r.get("saved_mb"), 0.0)
        self.assertEqual(calls, [], "futile (low token progress) must not record savings")

    def test_gate_passes_with_real_progress_even_if_still_high(self):
        # saved 20k of 100k = 20% >= 10% -> reload proceeds, even though post (80k)
        # is still high. This is the over-block guard: real progress is NOT blocked.
        calls = []
        r = self._run(pre_total=100_000, post_total=80_000, savings_calls=calls)
        self.assertFalse(r.get("token_progress_insufficient"))
        self.assertTrue(r.get("reloading"))
        self.assertEqual(len(calls), 1)

    def test_gate_boundary_exactly_min_ratio_proceeds(self):
        # saved exactly 10% -> not < 10% -> proceeds (boundary)
        calls = []
        r = self._run(pre_total=100_000, post_total=90_000, savings_calls=calls)
        self.assertFalse(r.get("token_progress_insufficient"))
        self.assertTrue(r.get("reloading"))


class TestReloadRateLedger(unittest.TestCase):
    def test_ledger_write_replaces_symlink_without_touching_target(self):
        from cozempic.guard import _reload_rate_exceeded

        d = Path(tempfile.mkdtemp(prefix="cozempic_ledger_"))
        try:
            victim = d / "victim"
            victim.write_text("keep")
            ledger = d / "h.json"
            os.symlink(victim, ledger)
            _reload_rate_exceeded(ledger, now=1000.0)
            self.assertFalse(ledger.is_symlink())
            self.assertEqual(victim.read_text(), "keep")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_helper_caps_after_max_in_window(self):
        from cozempic.guard import _reload_rate_exceeded
        d = Path(tempfile.mkdtemp(prefix="cozempic_ledger_"))
        try:
            p = d / "h.json"
            for i in range(3):
                cap, n = _reload_rate_exceeded(p, now=1000.0 + i)
                self.assertFalse(cap)
            cap, n = _reload_rate_exceeded(p, now=1004.0)
            self.assertTrue(cap)
            self.assertEqual(n, 3)
            cap, n = _reload_rate_exceeded(p, now=1000.0 + 100000)  # window reset
            self.assertFalse(cap)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_helper_survives_respawn_same_path(self):
        # "respawn" = fresh _reload_rate_exceeded calls against the SAME on-disk
        # path. The 4th within the window must cap from persisted state.
        from cozempic.guard import _reload_rate_exceeded
        d = Path(tempfile.mkdtemp(prefix="cozempic_ledger_"))
        try:
            p = d / "h.json"
            caps = [_reload_rate_exceeded(p, now=2000.0 + i)[0] for i in range(4)]
            self.assertEqual(caps, [False, False, False, True])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_corrupt_ledger_degrades_open(self):
        from cozempic.guard import _reload_rate_exceeded
        d = Path(tempfile.mkdtemp(prefix="cozempic_ledger_"))
        try:
            p = d / "h.json"
            p.write_text("{not json")
            cap, n = _reload_rate_exceeded(p, now=1000.0)
            self.assertFalse(cap, "corrupt ledger must never block a reload")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_malicious_session_id_confined(self):
        from cozempic.guard import _reload_ledger_path
        scratch = Path(tempfile.mkdtemp(prefix="cozempic_tmproot_"))
        try:
            with patch("cozempic.guard._guard_tmp_root", return_value=scratch):
                p = _reload_ledger_path("../../etc/passwd", Path("/x/s.jsonl"))
                # must stay inside the tmp root, sanitized
                self.assertEqual(p.parent, scratch)
                self.assertNotIn("..", p.name)
                self.assertNotIn("/", p.name.replace("cozempic_reload_", ""))
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def test_guard_cycle_caps_reload_when_ledger_full(self):
        from cozempic.team import TeamState
        from cozempic.guard import guard_prune_cycle, _reload_ledger_path, _reload_rate_exceeded
        tmpdir = Path(tempfile.mkdtemp(prefix="cozempic_loopfix2_"))
        scratch = Path(tempfile.mkdtemp(prefix="cozempic_tmproot_"))
        try:
            session_path = _make_session_file(tmpdir, 100_000)
            sid = "fedcba987654"
            team = MagicMock(spec=TeamState)
            team.is_empty.return_value = True
            team.team_name = None
            team.message_count = 0
            orig = [(0, {"type": "user"}, 100_000)]
            pruned = [(0, {"type": "user"}, 40_000)]
            terminate_called = []
            # Real token progress (100k -> 40k = 60%) so the token-progress gate
            # PASSES and we exercise the LEDGER path (pre-filled to the cap).
            _totals = iter([100_000, 40_000])

            def _est(*a, **k):
                try:
                    return MagicMock(total=next(_totals))
                except StopIteration:
                    return MagicMock(total=40_000)

            with patch("cozempic.guard._guard_tmp_root", return_value=scratch):
                # Pre-fill the ledger (in the scratch tmp root) to the cap.
                ledger = _reload_ledger_path(sid, session_path)
                import time
                for _ in range(3):
                    _reload_rate_exceeded(ledger, now=time.time())
                with patch("cozempic.guard.load_messages", return_value=orig), \
                     patch("cozempic.guard.prune_with_team_protect", return_value=(pruned, {}, team)), \
                     patch("cozempic.guard.save_messages", side_effect=lambda *a, **k: None), \
                     patch("cozempic.guard.snapshot_session", return_value=MagicMock()), \
                     patch("cozempic.guard._terminate_and_resume",
                           side_effect=lambda *a, **k: terminate_called.append(True)), \
                     patch("cozempic.tokens.estimate_session_tokens", side_effect=_est), \
                     patch("cozempic.tokens.calibrate_ratio", return_value=0.5):
                    r = guard_prune_cycle(
                        session_path=session_path, rx_name="standard", config=None,
                        auto_reload=True, claude_pid=999999, session_id=sid,
                    )
            self.assertTrue(r.get("reload_rate_capped"), "ledger cap must fire")
            self.assertTrue(r.get("futile_reload_skipped"), "capped cycle accounts to breaker")
            self.assertEqual(terminate_called, [], "capped cycle must not terminate Claude")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            shutil.rmtree(scratch, ignore_errors=True)


class TestVersionedPruneCounter(unittest.TestCase):
    def test_record_savings_pings_versioned_counter(self):
        from cozempic import helpers, __version__
        urls = []

        def _fake_urlopen(req, *a, **k):
            urls.append(req.full_url if hasattr(req, "full_url") else str(req))
            return MagicMock()

        with patch("cozempic.helpers._HostFileLock"), \
             patch("cozempic.helpers.atomic_write_text"), \
             patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            helpers.record_savings(123_456, total_tokens=500_000, turn_count=10)
        vtag = "".join(c if (c.isalnum() or c == "_") else "_" for c in __version__.replace(".", "_"))
        self.assertTrue(any("/counter/prunes/up" in u for u in urls), "base prune counter pinged")
        self.assertTrue(any(f"/counter/prunes_v{vtag}/up" in u for u in urls),
                        f"versioned counter prunes_v{vtag} must be pinged; got {urls}")

    def test_no_telemetry_env_suppresses_all_pings(self):
        from cozempic import helpers
        import os
        urls = []
        with patch.dict(os.environ, {"COZEMPIC_NO_TELEMETRY": "1"}), \
             patch("cozempic.helpers._HostFileLock"), \
             patch("cozempic.helpers.atomic_write_text"), \
             patch("urllib.request.urlopen", side_effect=lambda *a, **k: urls.append(1)):
            helpers.record_savings(123_456, total_tokens=500_000, turn_count=10)
        self.assertEqual(urls, [], "COZEMPIC_NO_TELEMETRY must suppress all pings")


if __name__ == "__main__":
    unittest.main()
