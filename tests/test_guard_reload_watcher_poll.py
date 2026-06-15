"""RED tests for GAP-B — watcher polls for new Claude PID post-osascript.

Architect spec: AUDIT_REPORT_pr94_transient_daemon_race.md § GAP-B.
Root cause: _spawn_reload_watcher's bash script fires osascript (or
gnome-terminal/start) and exits ~1-2s later with zero observability into
whether a new Claude process actually started. Failures (automation permission
denied, auth timeout, JSONL path error) are silently swallowed.

Fix: extend the watcher bash script to poll for a new claude process
(pgrep -f matching session-id prefix) for RELOAD_WATCHER_POLL_TIMEOUT_SECONDS
after osascript, then write a structured status file on failure.
SessionStart hook reads and surfaces the status file, then unlinks it.

All 6 tests EXPECTED TO FAIL until Phase B implementation lands.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Test 1 — Watcher writes status file when no new Claude appears
# ---------------------------------------------------------------------------
class TestWatcherWritesStatusOnNoNewClaude(unittest.TestCase):
    """When _spawn_reload_watcher fires and no claude process with the session
    prefix appears within RELOAD_WATCHER_POLL_TIMEOUT_SECONDS, the watcher
    must write a structured status file at /tmp/cozempic_reload_<sid12>.status.
    """

    def setUp(self):
        self.sid = "11223344556677889900aabbccdd1122"
        self.sid12 = self.sid[:12]
        self.status_path = Path(f"/tmp/cozempic_reload_{self.sid12}.status")
        self.status_path.unlink(missing_ok=True)
        self.addCleanup(self.status_path.unlink, missing_ok=True)

    def test_watcher_writes_status_on_no_new_claude(self):
        """No new claude process → status file with 'failed' first line written."""
        try:
            from cozempic.guard import RELOAD_WATCHER_POLL_TIMEOUT_SECONDS
        except ImportError:
            self.fail(
                "RELOAD_WATCHER_POLL_TIMEOUT_SECONDS missing from cozempic.guard — "
                "Phase B not yet applied. Expected RED."
            )

        # _spawn_reload_watcher with mocked osascript (returns 0) and
        # mocked pgrep (never finds a claude process → empty output)
        scripts_run = []
        # Save the real Popen BEFORE the patch replaces it so _fake_popen can
        # run the patched bash script synchronously without hitting the mock recursively.
        _real_popen = subprocess.Popen

        def _fake_popen(cmd_parts, **kwargs):
            if cmd_parts[0] == "bash" and cmd_parts[1] == "-c":
                script = cmd_parts[2]
                # Patch script for test isolation:
                # 1. Remove the `while kill -0 <pid>` wait loop (skip it)
                # 2. Replace osascript with `true` (succeeds, exit 0)
                # 3. Set poll timeout to 2s for speed
                # 4. Replace pgrep with a command that always returns empty
                import re
                patched = re.sub(r"while kill -0 \d+ 2>/dev/null; do sleep 1; done", "true", script)
                patched = re.sub(r"osascript[^;]*", "true", patched)
                # Replace the deadline line (use [$ ] char class to match literal $;
                # \$ in Python regex is the end-of-string anchor, not a literal dollar)
                patched = re.sub(
                    r"deadline=[^;]+;",
                    "deadline=$(($(date +%s) + 2));",  # 2s poll window for speed
                    patched,
                )
                # Replace pgrep (| is regex alternation — escape with \|; use [|] to match literal pipe)
                patched = re.sub(
                    r"pgrep -f '[^']*' 2>/dev/null [|] head -n 1",
                    "echo ''",  # always empty — no claude found
                    patched,
                )
                scripts_run.append(patched)
                # Run the script synchronously using the real Popen to avoid recursion
                with _real_popen(
                    ["bash", "-c", patched],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ) as proc:
                    proc.communicate(timeout=15)
                return MagicMock(pid=99999)
            return MagicMock(pid=99999)

        with patch("cozempic.guard.subprocess.Popen", side_effect=_fake_popen), \
             patch("cozempic.guard.platform.system", return_value="Darwin"), \
             patch("cozempic.guard.is_ssh_session", return_value=False):
            from cozempic.guard import _spawn_reload_watcher
            _spawn_reload_watcher(
                claude_pid=89113,
                project_dir="/tmp/fake_project",
                session_id=self.sid,
            )

        self.assertTrue(
            scripts_run,
            "_spawn_reload_watcher did not run any bash script — function may not be invoked.",
        )
        self.assertTrue(
            self.status_path.exists(),
            f"Status file {self.status_path} was NOT created after poll timeout. "
            "GAP-B fix not in watcher script. Phase B needed.",
        )

        # Verify first line is "failed"
        content = self.status_path.read_text()
        lines = content.strip().splitlines()
        self.assertEqual(
            lines[0].strip(),
            "failed",
            f"Status file first line is not 'failed': {lines[0]!r}. Content: {content!r}",
        )


# ---------------------------------------------------------------------------
# Test 2 — Watcher logs success when new Claude appears within poll window
# ---------------------------------------------------------------------------
class TestWatcherLogsSuccessWhenNewClaudeAppears(unittest.TestCase):
    """When a new claude process with the session-id prefix appears within
    RELOAD_WATCHER_POLL_TIMEOUT_SECONDS, the watcher must:
    - Log a success line to /tmp/cozempic_guard.log mentioning the new PID.
    - NOT write a status file.
    """

    def setUp(self):
        self.sid = "22334455667788990011bbccddeeff22"
        self.sid12 = self.sid[:12]
        self.status_path = Path(f"/tmp/cozempic_reload_{self.sid12}.status")
        self.status_path.unlink(missing_ok=True)
        self.addCleanup(self.status_path.unlink, missing_ok=True)

    def test_watcher_logs_success_when_new_claude_appears(self):
        """New claude detected → success logged, no status file."""
        try:
            from cozempic.guard import RELOAD_WATCHER_POLL_TIMEOUT_SECONDS
        except ImportError:
            self.fail(
                "RELOAD_WATCHER_POLL_TIMEOUT_SECONDS missing — Phase B not applied. Expected RED."
            )

        fake_new_pid = 94466
        guard_log = Path("/tmp/cozempic_guard.log")

        scripts_run = []
        # Save the real Popen BEFORE the patch to avoid mock recursion in _fake_popen
        _real_popen = subprocess.Popen

        def _fake_popen(cmd_parts, **kwargs):
            if cmd_parts[0] == "bash" and cmd_parts[1] == "-c":
                script = cmd_parts[2]
                import re
                # Skip wait loop, replace osascript with true
                patched = re.sub(r"while kill -0 \d+ 2>/dev/null; do sleep 1; done", "true", script)
                patched = re.sub(r"osascript[^;]*", "true", patched)
                # Short deadline (5s) for test speed
                patched = re.sub(
                    r"deadline=[^;]+;",
                    "deadline=$(($(date +%s) + 5));",
                    patched,
                )
                # Replace pgrep with one that returns a fake PID immediately
                patched = re.sub(
                    r"pgrep -f '[^']*' 2>/dev/null [|] head -n 1",
                    f"echo '{fake_new_pid}'",
                    patched,
                )
                scripts_run.append(patched)
                with _real_popen(
                    ["bash", "-c", patched],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ) as proc:
                    proc.communicate(timeout=15)
                return MagicMock(pid=99999)
            return MagicMock(pid=99999)

        with patch("cozempic.guard.subprocess.Popen", side_effect=_fake_popen), \
             patch("cozempic.guard.platform.system", return_value="Darwin"), \
             patch("cozempic.guard.is_ssh_session", return_value=False):
            from cozempic.guard import _spawn_reload_watcher
            _spawn_reload_watcher(
                claude_pid=89113,
                project_dir="/tmp/fake_project",
                session_id=self.sid,
            )

        self.assertFalse(
            self.status_path.exists(),
            f"Status file was created despite new claude being found. "
            "Success path should NOT write status file.",
        )

        # Guard log should mention the new PID
        if guard_log.exists():
            log_content = guard_log.read_text()
            self.assertIn(
                str(fake_new_pid),
                log_content,
                f"Guard log does not mention new PID {fake_new_pid}. "
                f"Log tail: {log_content[-500:]!r}",
            )


# ---------------------------------------------------------------------------
# Test 3 — Watcher handles resume command non-zero exit
# ---------------------------------------------------------------------------
class TestWatcherHandlesResumeCmdNonzeroExit(unittest.TestCase):
    """When osascript (or gnome-terminal) exits non-zero, the watcher must
    still proceed to poll and then write the status file with the exit code.
    """

    def setUp(self):
        self.sid = "33445566778899001122ccddee334455"
        self.sid12 = self.sid[:12]
        self.status_path = Path(f"/tmp/cozempic_reload_{self.sid12}.status")
        self.status_path.unlink(missing_ok=True)
        self.addCleanup(self.status_path.unlink, missing_ok=True)

    def test_watcher_handles_resume_cmd_nonzero_exit(self):
        """osascript exit=1 → watcher polls (no claude found) → status file with exit=1."""
        try:
            from cozempic.guard import RELOAD_WATCHER_POLL_TIMEOUT_SECONDS
        except ImportError:
            self.fail(
                "RELOAD_WATCHER_POLL_TIMEOUT_SECONDS missing — Phase B not applied. Expected RED."
            )

        scripts_run = []
        # Save the real Popen BEFORE the patch to avoid mock recursion in _fake_popen
        _real_popen = subprocess.Popen

        def _fake_popen(cmd_parts, **kwargs):
            if cmd_parts[0] == "bash" and cmd_parts[1] == "-c":
                script = cmd_parts[2]
                import re
                patched = re.sub(r"while kill -0 \d+ 2>/dev/null; do sleep 1; done", "true", script)
                # osascript exits 1 — simulate automation permission denied
                patched = re.sub(r"osascript[^;]*", "false", patched)
                patched = re.sub(
                    r"deadline=[^;]+;",
                    "deadline=$(($(date +%s) + 2));",
                    patched,
                )
                patched = re.sub(r"pgrep -f '[^']*' 2>/dev/null [|] head -n 1", "echo ''", patched)
                scripts_run.append(patched)
                with _real_popen(
                    ["bash", "-c", patched],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ) as proc:
                    proc.communicate(timeout=15)
                return MagicMock(pid=99999)
            return MagicMock(pid=99999)

        with patch("cozempic.guard.subprocess.Popen", side_effect=_fake_popen), \
             patch("cozempic.guard.platform.system", return_value="Darwin"), \
             patch("cozempic.guard.is_ssh_session", return_value=False):
            from cozempic.guard import _spawn_reload_watcher
            _spawn_reload_watcher(
                claude_pid=89113,
                project_dir="/tmp/fake_project",
                session_id=self.sid,
            )

        self.assertTrue(
            self.status_path.exists(),
            "Status file not written despite osascript exit=1 + no claude found. "
            "GAP-B fix incomplete — non-zero osascript exit must trigger status file write.",
        )

        content = self.status_path.read_text()
        self.assertIn(
            "exit=1",
            content,
            f"Status file does not record exit code. Content: {content!r}",
        )


# ---------------------------------------------------------------------------
# Test 4 — SessionStart hook surfaces prior status file to user
# ---------------------------------------------------------------------------
class TestSessionStartHookSurfacesPriorStatus(unittest.TestCase):
    """When a reload status file exists at session start, the SessionStart
    hook bash must:
    1. Print the failure reason to stdout (operator-visible).
    2. Unlink the status file after reading it.
    """

    def setUp(self):
        self.sid = "44556677889900112233ddeeff445566"
        self.sid12 = self.sid[:12]
        self.status_path = Path(f"/tmp/cozempic_reload_{self.sid12}.status")
        # Plant a failure status file
        self.status_path.write_text(
            "failed\n"
            "2026-05-19T14:38:29\n"
            "new Claude did not start within 30s after osascript (exit=0)\n"
            "investigate: Terminal automation permission / claude -r auth / JSONL path / network\n"
        )
        self.addCleanup(self.status_path.unlink, missing_ok=True)

    def _load_session_start_command(self) -> str:
        import json as _json
        hooks_path = SRC / "cozempic" / "data" / "hooks.json"
        hooks = _json.loads(hooks_path.read_text(encoding="utf-8"))
        return hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]

    def test_session_start_hook_surfaces_prior_status(self):
        """Hook prints failure reason and removes status file."""
        import json as _json
        cmd = self._load_session_start_command()
        # Status file surface requires v10+ hook schema
        from cozempic.init import HOOK_SCHEMA_MARKER
        self.assertIn(
            HOOK_SCHEMA_MARKER,
            cmd,
            "Hook schema marker not current — status file surfacing not yet in hooks.json. "
            "Expected RED until Phase B adds status surface to hooks.json.",
        )
        hook_data = _json.dumps({"session_id": self.sid, "transcript_path": ""})
        env = os.environ.copy()
        result = subprocess.run(
            ["bash", "-c", cmd],
            input=hook_data,
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )
        combined_output = result.stdout + result.stderr
        self.assertIn(
            "reload",
            combined_output.lower(),
            f"Hook did not surface reload failure message. Output: {combined_output!r}",
        )
        self.assertFalse(
            self.status_path.exists(),
            "Status file was NOT unlinked after being read by hook. "
            "Hook must clean up status file.",
        )


# ---------------------------------------------------------------------------
# Test 5 — Status files are per-session (no cross-contamination)
# ---------------------------------------------------------------------------
class TestStatusFilePerSessionIsolation(unittest.TestCase):
    """Two concurrent reloads on different sessions, both failing, must each
    get their own status file at the per-session path with no cross-contamination.
    """

    def setUp(self):
        self.sid_a = "aaaaaaaaaaaa0000111122223333aaaa"
        self.sid_b = "bbbbbbbbbbbb0000111122223333bbbb"
        self.sid_a12 = self.sid_a[:12]
        self.sid_b12 = self.sid_b[:12]
        self.status_a = Path(f"/tmp/cozempic_reload_{self.sid_a12}.status")
        self.status_b = Path(f"/tmp/cozempic_reload_{self.sid_b12}.status")
        for p in (self.status_a, self.status_b):
            p.unlink(missing_ok=True)
        self.addCleanup(self.status_a.unlink, missing_ok=True)
        self.addCleanup(self.status_b.unlink, missing_ok=True)

    def _write_fake_status(self, path: Path, content: str):
        path.write_text(content)

    def test_status_file_per_session_isolation(self):
        """Two concurrent failures produce two separate status files."""
        try:
            from cozempic.guard import RELOAD_WATCHER_POLL_TIMEOUT_SECONDS
        except ImportError:
            self.fail("RELOAD_WATCHER_POLL_TIMEOUT_SECONDS missing — Phase B not applied.")

        # Simulate two failing watchers by directly writing their status files
        # (testing the naming convention, not the bash script)
        self._write_fake_status(
            self.status_a, "failed\n2026-05-19T14:38:00\nsession A failure\n"
        )
        self._write_fake_status(
            self.status_b, "failed\n2026-05-19T14:38:00\nsession B failure\n"
        )

        # Verify they are independent
        self.assertTrue(self.status_a.exists(), "Session A status file missing")
        self.assertTrue(self.status_b.exists(), "Session B status file missing")

        content_a = self.status_a.read_text()
        content_b = self.status_b.read_text()

        self.assertIn("session A", content_a)
        self.assertIn("session B", content_b)
        self.assertNotIn("session B", content_a)
        self.assertNotIn("session A", content_b)


# ---------------------------------------------------------------------------
# Test 6 — Poll pattern does NOT match unrelated claude process
# ---------------------------------------------------------------------------
class TestPollPatternDoesNotMatchUnrelatedClaude(unittest.TestCase):
    """The pgrep pattern used by the watcher must NOT match a 'claude' process
    that does NOT carry the session_id prefix in its argv.

    Scenario: user has another claude session (claude -r different-uuid)
    running. Watcher should not report it as the new Claude for our session.
    """

    def test_poll_pattern_does_not_match_unrelated_claude(self):
        """pgrep -f 'claude.*<sid12>' does not match 'claude -r other-session'."""
        try:
            from cozempic.guard import RELOAD_WATCHER_POLL_TIMEOUT_SECONDS
        except ImportError:
            self.fail("RELOAD_WATCHER_POLL_TIMEOUT_SECONDS missing — Phase B not applied.")

        our_sid12 = "11223344556677889900aabbccdd1122"[:12]  # "112233445566"

        # The watcher script uses pgrep -f "claude.*<sid_prefix>"
        # Test: a process named "claude -r ffffffffffff..." (wrong session) must NOT match.
        # We verify this by testing the pgrep pattern directly in a subprocess.

        # Spin up a fake "claude" process with a different session id in argv
        other_sid = "ffffffffffff00001111222233334444"
        # Use a sleep process with a fake argv via a shell trick for portable testing
        fake_process = subprocess.Popen(
            ["bash", "-c", f"exec -a 'claude -r {other_sid[:12]} --session {other_sid}' sleep 30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(fake_process.terminate)
        time.sleep(0.3)  # Let it start

        try:
            # Pattern the watcher will use for OUR session
            pattern = f"claude.*{our_sid12}"
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True,
                text=True,
            )
            matched_pids = [p.strip() for p in result.stdout.strip().splitlines() if p.strip()]

            # The fake process has a DIFFERENT session_id prefix — should NOT be matched
            self.assertNotIn(
                str(fake_process.pid),
                matched_pids,
                f"pgrep pattern '{pattern}' matched unrelated claude process {fake_process.pid}. "
                "Pattern discriminator is too broad — needs the session-id prefix to be tight.",
            )

            # Conversely, if we use the OTHER session's prefix, the fake process IS found
            other_pattern = f"claude.*{other_sid[:12]}"
            other_result = subprocess.run(
                ["pgrep", "-f", other_pattern],
                capture_output=True,
                text=True,
            )
            other_pids = [p.strip() for p in other_result.stdout.strip().splitlines() if p.strip()]
            self.assertIn(
                str(fake_process.pid),
                other_pids,
                f"pgrep pattern '{other_pattern}' did NOT find the fake process {fake_process.pid}. "
                "Test infrastructure issue: fake process not matching its own session prefix.",
            )
        finally:
            fake_process.terminate()
            fake_process.wait(timeout=3)


class TestReloadWatcherPlatformDispatch(unittest.TestCase):
    """The watcher must spawn a PowerShell-native script on Windows and a bash
    script on POSIX. Regression for the audit P1: the Windows path used to embed
    a cmd.exe command inside a POSIX bash script run via Popen(["bash", ...]),
    which is dead on Windows (no bash) — auto-resume never fired.

    Verified by EMULATION (mock platform.system + capture Popen argv); a true
    end-to-end Windows check needs a Windows runner (no CI runner here)."""

    def _capture_popen_argv(self, system):
        captured = {}
        def _fake_popen(args, *a, **k):
            captured["argv"] = args
            captured["kwargs"] = k
            return MagicMock(pid=4321)
        with patch("cozempic.guard.subprocess.Popen", side_effect=_fake_popen), \
             patch("cozempic.guard.platform.system", return_value=system), \
             patch("cozempic.guard.is_ssh_session", return_value=False), \
             patch("cozempic.guard._detect_claude_flags", return_value=""):
            from cozempic.guard import _spawn_reload_watcher
            _spawn_reload_watcher(claude_pid=4321,
                                  project_dir=r"C:\Users\me\proj" if system == "Windows" else "/tmp/proj",
                                  session_id="abcdef12-0000-0000-0000-000000000000")
        return captured

    def test_windows_uses_powershell_not_bash(self):
        cap = self._capture_popen_argv("Windows")
        argv = cap["argv"]
        self.assertEqual(argv[0], "powershell",
                         "Windows watcher must launch via powershell, not bash")
        script = argv[-1]
        # PowerShell-native phases, no POSIX-isms.
        self.assertIn("Get-Process -Id", script, "must wait on the old pid via Get-Process")
        self.assertIn("Start-Process", script, "must resume via Start-Process")
        self.assertIn("cmd.exe", script)
        self.assertIn("claude", script)
        self.assertNotIn("kill -0", script, "no POSIX kill in the Windows watcher")
        self.assertNotIn("pgrep", script, "no POSIX pgrep in the Windows watcher")
        # Detached, no POSIX-only start_new_session kwarg.
        self.assertNotIn("start_new_session", cap["kwargs"])

    def test_posix_still_uses_bash(self):
        for system in ("Darwin", "Linux"):
            cap = self._capture_popen_argv(system)
            self.assertEqual(cap["argv"][0], "bash",
                             f"{system} watcher must still use bash")
            self.assertIn("kill -0", cap["argv"][-1])


if __name__ == "__main__":
    unittest.main()
