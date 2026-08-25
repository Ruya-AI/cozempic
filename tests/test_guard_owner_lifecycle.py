"""GREEN tests for the guard owner-lifecycle fix (ISSUE-AGT-637).

Two defects were observed on macOS with Cozempic 1.8.39:
  1. The SessionStart hook backgrounds the whole updater/spawn subshell, so by
     the time ``cozempic guard --daemon`` runs, the process tree has been
     reparented under PID 1. ``find_claude_pid()`` then returns None, the
     daemon argv lacks ``--claude-pid``, and the guard never notices Claude
     exited — leaking a PPID-1 guard per session.
  2. The guard only checks owner exit when ``claude_pid`` is truthy.

Fix under test:
  * The hook captures an identity-verified Claude PID (a process whose ``comm``
    matches ``claude``/``node``, same check as ``find_claude_pid``) in the hook's
    MAIN shell, BEFORE the background boundary, and threads it as
    ``guard --daemon --claude-pid <pid>``.
  * ``start_guard`` stops (final checkpoint + break) when that PID dies.

Test 1 is the mutation-proof hook-path proof: a fake ``claude`` ancestor process
(compiled C binary so its ``comm`` contains "claude") drives the real hook
command through a backgrounded spawn. If the capture/threading is deleted or
moved inside the background boundary, the daemon argv loses ``--claude-pid``
and the test fails.

Test 2 is the mutation-proof owner-exit proof: ``start_guard`` launched with a
live owner PID must exit (break) when that PID dies on the first poll. Delete
the ``if claude_pid and claude_alive`` block and the thread never joins.

No giant fixtures, no real ~/.claude access, no killing of live processes.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
import textwrap
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent

_FAKE_CLAUDE_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(int argc, char **argv) {
    /* Fake "claude" process for hook tests: argv[1] = file containing the hook
       command, argv[2] = file containing the hook payload. Spawn
       `bash -c <cmd>` as a child with the payload on stdin and wait, so the
       hook runs as a descendant of THIS process (whose comm contains
       "claude"). */
    if (argc < 3) return 2;
    FILE *f = fopen(argv[1], "r");
    if (!f) return 3;
    char cmd[65536];
    size_t n = fread(cmd, 1, sizeof(cmd) - 1, f);
    cmd[n] = '\0';
    fclose(f);
    FILE *pf = fopen(argv[2], "r");
    char payload[16384];
    size_t pn = pf ? fread(payload, 1, sizeof(payload) - 1, pf) : 0;
    if (pf) fclose(pf);
    payload[pn] = '\0';
    int pipefd[2];
    if (pipe(pipefd) != 0) return 4;
    pid_t pid = fork();
    if (pid == 0) {
        dup2(pipefd[0], 0);
        close(pipefd[0]); close(pipefd[1]);
        execl("/bin/bash", "bash", "-c", cmd, (char *)NULL);
        _exit(127);
    }
    close(pipefd[0]);
    if (pn) write(pipefd[1], payload, pn);
    close(pipefd[1]);
    int status;
    waitpid(pid, &status, 0);
    return 0;
}
"""


def _session_start_command() -> str:
    hooks = json.loads(
        (REPO_ROOT / "src/cozempic/data/hooks.json").read_text(encoding="utf-8")
    )
    return hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]


class TestHookThreadsClaudePid(unittest.TestCase):
    """The real SessionStart hook must capture the Claude PID before the
    background boundary and pass it to the guard daemon."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cozempic_owner_"))
        self.session_id = "abc12345-1111-2222-3333-444455556666"
        self.sid12 = self.session_id[:12]
        self.fake_tmp = self.tmpdir / "tmp"
        self.fake_tmp.mkdir()
        self.pid_file = Path(f"/tmp/cozempic_guard_{self.sid12}.pid")
        self.hook_lock = Path(f"/tmp/cozempic_hook_{self.sid12}.lock")
        for p in (self.pid_file, self.hook_lock):
            if p.exists():
                p.unlink()

        self.cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
        if not self.cc:
            self.skipTest("no C compiler available for the fake-claude helper")

        self.invocation_log = self.tmpdir / "invocations.log"
        self.bin_dir = self.tmpdir / "bin"
        self.bin_dir.mkdir()
        self._install_stub_cozempic()
        self.fake_claude = self._compile_fake_claude()

    def tearDown(self):
        self._kill_pid_file_daemon()
        for p in (self.pid_file, self.hook_lock):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _compile_fake_claude(self) -> Path:
        src = self.tmpdir / "fake_claude.c"
        src.write_text(_FAKE_CLAUDE_C)
        out = self.tmpdir / "fakebin"
        out.mkdir()
        binary = out / "claude"
        subprocess.run(
            [self.cc, "-o", str(binary), str(src)],
            check=True,
            capture_output=True,
        )
        return binary

    def _install_stub_cozempic(self):
        stub = self.bin_dir / "cozempic"
        stub.write_text(
            textwrap.dedent(f"""\
            #!/usr/bin/env bash
            printf '%s\\n' "$$ $*" >> {self.invocation_log!s}
            case "$1" in
              --version) echo "cozempic 99.0.0" ;;
              guard)
                if [[ "$2" == "--daemon" || "$2" == "--reload-self" ]]; then
                  nohup sleep 30 </dev/null >/dev/null 2>&1 &
                  printf '%s' "$!" > {self.pid_file!s}
                fi
                ;;
              *) : ;;
            esac
            exit 0
        """)
        )
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        for name in ("uv", "pip"):
            s = self.bin_dir / name
            s.write_text("#!/usr/bin/env bash\nexit 0\n")
            s.chmod(s.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    def _kill_pid_file_daemon(self):
        if not self.pid_file.exists():
            return
        try:
            pid = int(self.pid_file.read_text().splitlines()[0].strip())
            os.kill(pid, 9)
        except (ValueError, ProcessLookupError, OSError, IndexError):
            pass

    def _wait_for_background_spawn(self):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            time.sleep(0.2)
            if self.invocation_log.exists():
                size = self.invocation_log.stat().st_size
                time.sleep(0.5)
                if self.invocation_log.stat().st_size == size:
                    return

    def _spawn_log_lines(self):
        if not self.invocation_log.exists():
            return []
        return self.invocation_log.read_text().splitlines()

    def test_daemon_receives_original_claude_pid(self):
        """The hook (backgrounding its spawn subshell) must thread the
        identity-verified Claude PID captured BEFORE the boundary."""
        cmd = _session_start_command()
        env = os.environ.copy()
        env["PATH"] = f"{self.bin_dir}:{env.get('PATH', '')}"
        env["PYTHONPATH"] = ""
        payload = json.dumps(
            {
                "session_id": self.session_id,
                "transcript_path": f"/tmp/{self.session_id}.jsonl",
                "hook_event_name": "SessionStart",
                "source": "startup",
            }
        )
        cmd_file = self.tmpdir / "hook_cmd.txt"
        payload_file = self.tmpdir / "hook_payload.txt"
        cmd_file.write_text(cmd)
        payload_file.write_text(payload)

        proc = subprocess.Popen(
            [str(self.fake_claude), str(cmd_file), str(payload_file)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        fake_claude_pid = proc.pid
        proc.communicate(timeout=30)
        self._wait_for_background_spawn()

        daemon_lines = [l for l in self._spawn_log_lines() if "guard --daemon" in l]
        self.assertGreater(len(daemon_lines), 0, "no guard --daemon spawn recorded")
        self.assertIn(
            f"--claude-pid {fake_claude_pid}",
            daemon_lines[0],
            "the daemon argv must carry the identity-verified Claude PID captured "
            "before the SessionStart background boundary. Mutation guard: delete "
            "the hook's PID capture/threading and this test fails.",
        )


class TestGuardStopsWhenOwnerExits(unittest.TestCase):
    """start_guard must stop when the threaded owner PID dies (mutation-proof)."""

    def _run_guard_until_owner_death(self):
        import io
        import threading

        import cozempic.guard as g

        owner = subprocess.Popen(["sleep", "60"])
        owner_pid = owner.pid
        with tempfile.TemporaryDirectory() as td:
            jsonl = Path(td) / "session.jsonl"
            jsonl.write_text("{}\n")
            fake_session = {
                "session_id": "owner-test-1111-2222-3333-444455556666",
                "session_path": str(jsonl),
                "path": jsonl,
                "project_dir": td,
            }
            out = io.StringIO()
            thread_done = threading.Event()
            thread_err = []

            def runner():
                try:
                    with redirect_stdout(out):
                        g.start_guard(
                            session_id=fake_session["session_id"],
                            claude_pid=owner_pid,
                            interval=30,
                            reactive=False,
                        )
                except BaseException as e:  # noqa: BLE001
                    thread_err.append(e)
                finally:
                    thread_done.set()

            killed = {"n": 0}

            def fake_sleep(secs):
                # On the loop's first poll, kill the owner so the owner-exit
                # check on the same iteration sees it gone.
                if killed["n"] == 0:
                    killed["n"] = 1
                    owner.kill()
                    owner.wait()
                return None

            with (
                mock.patch("cozempic.guard.time.sleep", side_effect=fake_sleep),
                mock.patch(
                    "cozempic.guard._resolve_session_by_id", return_value=fake_session
                ),
                mock.patch("cozempic.guard.find_claude_pid", return_value=None),
                mock.patch("cozempic.guard._record_claude_identity"),
                mock.patch("cozempic.guard.load_messages", return_value=[]),
                mock.patch("cozempic.guard.quick_token_estimate", return_value=None),
                mock.patch(
                    "cozempic.tokens.detect_context_window", return_value=200000
                ),
                mock.patch(
                    "cozempic.tokens.default_token_thresholds_4tier",
                    return_value=(50000, 110000, 160000),
                ),
                mock.patch("cozempic.session.record_session"),
                mock.patch("cozempic.guard._cleanup_stale_watchers"),
                mock.patch("cozempic.guard.ping_install_if_new"),
                mock.patch("cozempic.guard.maybe_auto_update"),
                mock.patch("cozempic.guard._safe_unlink_session_pidfile"),
                # signal.signal can only run in the main thread; the guard loop
                # runs here in a worker thread.
                mock.patch("cozempic.guard.signal.signal"),
            ):
                t = threading.Thread(target=runner)
                t.start()
                t.join(timeout=20)

            self.assertFalse(
                t.is_alive(), "start_guard did not exit after the owner PID died"
            )
            self.assertEqual(thread_err, [])
            captured = out.getvalue()
            self.assertIn(
                "Guard stopping (Claude exited).",
                captured,
                "the owner-exit path must fire. Mutation guard: delete the "
                "`if claude_pid and claude_alive` watchdog block and this test "
                "hangs/fails.",
            )

    def test_guard_exits_when_owner_pid_dies(self):
        self._run_guard_until_owner_death()


if __name__ == "__main__":
    unittest.main()
