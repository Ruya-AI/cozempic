"""Tests for the cross-process daemon spawn lock.

Companion to ``test_guard_race_2026_05_18.py::TestR1_*`` (the original RED
test). These tests pin down:

  - Three-process contention (extends R1's two-process race).
  - The "no placeholder PID 0 ever visible" invariant — closes Bug 3.
  - FileNotFoundError recovery on the log file open path.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _race_worker(
    barrier_handle,
    result_queue,
    session_id: str,
    cwd: str,
    worker_index: int,
):
    """Mirror of the R1 worker — fakes Popen + claude-pid so the test stays
    on the spawn-lock contract, not the daemon lifecycle."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    import subprocess as _subprocess
    from unittest.mock import patch as _patch

    from cozempic.guard import start_guard_daemon

    _real_popen = _subprocess.Popen

    class _DummyProc:
        def __init__(self, pid):
            self.pid = pid

    def _fake_popen(cmd_parts, **kwargs):
        # Fake ONLY the daemon-spawn Popen (python -m cozempic.cli guard).
        # Every other subprocess use — notably the `ps` identity probe inside
        # _is_cozempic_guard_process, which subprocess.run drives via
        # `with Popen(...) as p: p.communicate()` — must run for real; the
        # global Popen patch otherwise hands `ps` a _DummyProc with no
        # __enter__/communicate, raising AttributeError and turning every
        # loser worker into an 'undefined' outcome (the deterministically-dead
        # gate this test is supposed to be).
        parts = cmd_parts if isinstance(cmd_parts, (list, tuple)) else [cmd_parts]
        is_daemon_spawn = any("cozempic.cli" in str(p) for p in parts) or (
            any("cozempic" in str(p) for p in parts) and any(str(p) == "guard" for p in parts)
        )
        if is_daemon_spawn:
            return _DummyProc(900_000 + worker_index)
        return _real_popen(cmd_parts, **kwargs)

    with (
        _patch("cozempic.guard.subprocess.Popen", side_effect=_fake_popen),
        _patch("cozempic.guard.find_claude_pid", return_value=12345),
        _patch("cozempic.guard._cleanup_legacy_pid"),
    ):
        try:
            barrier_handle.wait(timeout=10.0)
        except Exception as e:
            result_queue.put(
                {"error": f"barrier failed: {e!r}", "worker": worker_index}
            )
            return

        try:
            r = start_guard_daemon(
                cwd=cwd,
                session_id=session_id,
                threshold_tokens=1000,
            )
        except Exception as e:
            r = {"error": repr(e)}
        r["worker"] = worker_index
        result_queue.put(r)


class TestThreeProcessContention(unittest.TestCase):
    """Three processes race for the same session — exactly ONE must win."""

    SESSION_ID = "fade1234-5678-9abc-def0-2026051811cc"

    def setUp(self):
        from cozempic.guard import _pid_file_for_session

        self.pid_path = _pid_file_for_session(self.SESSION_ID)
        self.pid_path.unlink(missing_ok=True)
        self.log_path = self.pid_path.with_suffix(".log")
        self.log_path.unlink(missing_ok=True)

    def tearDown(self):
        self.pid_path.unlink(missing_ok=True)
        self.log_path.unlink(missing_ok=True)

    def test_three_process_contention(self):

        ctx = mp.get_context("spawn")
        N = 3
        ITERATIONS = 20
        cwd = os.getcwd()

        failures = []
        for it in range(ITERATIONS):
            self.pid_path.unlink(missing_ok=True)
            self.log_path.unlink(missing_ok=True)

            barrier = ctx.Barrier(N)
            queue = ctx.Queue()
            procs = [
                ctx.Process(
                    target=_race_worker,
                    args=(barrier, queue, self.SESSION_ID, cwd, i),
                    name=f"race-3-{i}",
                )
                for i in range(N)
            ]
            for p in procs:
                p.start()

            results = []
            for _ in range(N):
                try:
                    results.append(queue.get(timeout=15.0))
                except Exception as e:
                    results.append({"error": f"queue.get failed: {e!r}"})
            for p in procs:
                p.join(timeout=5.0)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2.0)

            started = [r for r in results if r.get("started") is True]
            already = [r for r in results if r.get("already_running") is True]

            if len(started) != 1 or len(already) != (N - 1):
                failures.append({"iteration": it, "results": results})

        if failures:
            self.fail(
                f"3-process race produced bad outcomes in "
                f"{len(failures)}/{ITERATIONS} iterations. "
                f"First 2: {failures[:2]!r}"
            )


class TestV4TenProcessContention(unittest.TestCase):
    """V4 regression: 10 contending processes × 30 iterations.

    The V4 stress (race-reproducer, 2026-05-18) exposed the original
    ``fcntl.flock`` implementation's unlink-on-release race: peers that
    O_CREAT'd new inodes after the holder's unlink got flocks on
    different kernel objects and both proceeded to spawn. The fix replaces
    flock with ``O_CREAT|O_EXCL`` on the PID file directly — POSIX
    guarantees exactly one winner per file create.

    This test is heavier than ``TestThreeProcessContention`` (10 procs ×
    30 iter ≈ 8s on a healthy laptop) and serves as the per-PR regression
    gate for any change to spawn_lock.py or start_guard_daemon.
    """

    SESSION_ID = "feedbeef-1234-5678-9abc-2026051811ff"
    N = 10
    ITERATIONS = 30

    def setUp(self):
        from cozempic.guard import _pid_file_for_session

        self.pid_path = _pid_file_for_session(self.SESSION_ID)
        self.pid_path.unlink(missing_ok=True)
        self.log_path = self.pid_path.with_suffix(".log")
        self.log_path.unlink(missing_ok=True)

    def tearDown(self):
        self.pid_path.unlink(missing_ok=True)
        self.log_path.unlink(missing_ok=True)
        self.pid_path.with_suffix(".pid.tmp").unlink(missing_ok=True)

    def test_ten_process_contention_30x(self):
        ctx = mp.get_context("spawn")
        cwd = os.getcwd()

        failures = []
        for it in range(self.ITERATIONS):
            self.pid_path.unlink(missing_ok=True)
            self.log_path.unlink(missing_ok=True)

            barrier = ctx.Barrier(self.N)
            queue = ctx.Queue()
            procs = [
                ctx.Process(
                    target=_race_worker,
                    args=(barrier, queue, self.SESSION_ID, cwd, i),
                    name=f"v4-{i}",
                )
                for i in range(self.N)
            ]
            for p in procs:
                p.start()

            results = []
            for _ in range(self.N):
                try:
                    results.append(queue.get(timeout=20.0))
                except Exception as e:
                    results.append({"error": f"queue.get: {e!r}"})
            for p in procs:
                p.join(timeout=5.0)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2.0)

            started = [r for r in results if r.get("started") is True]
            already = [r for r in results if r.get("already_running") is True]
            undefined = [
                r
                for r in results
                if r.get("started") is not True and r.get("already_running") is not True
            ]

            if len(started) != 1 or len(already) != (self.N - 1) or undefined:
                failures.append(
                    {
                        "iter": it,
                        "started_count": len(started),
                        "already_count": len(already),
                        "undefined_count": len(undefined),
                        "started_workers": [r.get("worker") for r in started],
                        "first_undefined": undefined[0] if undefined else None,
                    }
                )

        if failures:
            self.fail(
                f"V4 stress (10p × {self.ITERATIONS}it) produced "
                f"{len(failures)}/{self.ITERATIONS} bad outcomes. "
                f"First 3: {failures[:3]!r}"
            )


class TestNoPlaceholderPidVisible(unittest.TestCase):
    """The pidfile must NEVER hold '0' between any two reads — closes Bug 3.

    After the v1.8.14 refactor the file is created via atomic-rename, so
    only "no file" or "file with real PID > 0" should ever be observable.
    """

    SESSION_ID = "babe1234-5678-9abc-def0-2026051811dd"

    def setUp(self):
        from cozempic.guard import _pid_file_for_session

        self.pid_path = _pid_file_for_session(self.SESSION_ID)
        self.pid_path.unlink(missing_ok=True)
        self.log_path = self.pid_path.with_suffix(".log")
        self.log_path.unlink(missing_ok=True)

    def tearDown(self):
        self.pid_path.unlink(missing_ok=True)
        self.log_path.unlink(missing_ok=True)
        self.pid_path.with_suffix(".pid.tmp").unlink(missing_ok=True)

    def test_no_placeholder_pid_ever_visible(self):
        """Run start_guard_daemon while a tight reader loop observes the
        pidfile. Any observation of '0' is a Bug 3 regression."""
        from cozempic.guard import start_guard_daemon

        observed_values: list[str] = []
        stop = {"flag": False}

        def reader():
            while not stop["flag"]:
                try:
                    if self.pid_path.exists():
                        observed_values.append(self.pid_path.read_text().strip())
                except OSError:
                    pass

        import threading

        t = threading.Thread(target=reader, daemon=True)
        t.start()

        with (
            patch("cozempic.guard.subprocess.Popen") as mock_popen,
            patch("cozempic.guard.find_claude_pid", return_value=12345),
            patch("cozempic.guard._cleanup_legacy_pid"),
        ):
            mock_popen.return_value.pid = 88888
            result = start_guard_daemon(
                cwd=os.getcwd(),
                session_id=self.SESSION_ID,
                threshold_tokens=1000,
            )

        # Let the reader catch the post-spawn state too
        time.sleep(0.05)
        stop["flag"] = True
        t.join(timeout=1.0)

        self.assertTrue(result.get("started"), f"spawn failed: {result!r}")
        # The reader may have caught the file empty or with the real PID, but
        # NEVER with "0".
        self.assertNotIn(
            "0",
            observed_values,
            f"Placeholder PID 0 observed on disk — Bug 3 regression. "
            f"Observations: {observed_values[:10]}",
        )


class TestFreshWindowEnvVarClamping(unittest.TestCase):
    """DA round 3 N2 regression test: the ``COZEMPIC_PIDFILE_FRESH_SECONDS``
    env var must be clamped to ``(0, _FRESH_MAX]``. Operator typos like
    ``inf``, ``1e10``, ``-1``, ``0``, ``NaN``, or non-numeric junk must
    fall back to the default rather than silently disabling staleness
    recovery (an inf/1e10 fresh window would treat ANY pidfile as fresh
    forever; a 0/negative window would never classify anything fresh).
    """

    ENV_VAR = "COZEMPIC_PIDFILE_FRESH_SECONDS"

    def _reload_with_env(self, value):
        """Set or unset the env var, reload spawn_lock, and return the
        active _FRESH_PIDFILE_SECONDS value."""
        import importlib

        import cozempic.spawn_lock as sl

        prior = os.environ.get(self.ENV_VAR)
        try:
            if value is None:
                os.environ.pop(self.ENV_VAR, None)
            else:
                os.environ[self.ENV_VAR] = value
            importlib.reload(sl)
            return sl._FRESH_PIDFILE_SECONDS
        finally:
            # Restore prior env state so other tests aren't affected.
            if prior is None:
                os.environ.pop(self.ENV_VAR, None)
            else:
                os.environ[self.ENV_VAR] = prior
            importlib.reload(sl)

    def test_env_var_clamps_inf(self):
        """``inf`` is not finite → must fall back to default. Without the
        clamp, an inf fresh window would treat every pidfile as fresh
        forever — peer-claim recovery would never fire."""
        from cozempic.spawn_lock import _DEFAULT_FRESH

        for val in ("inf", "Infinity", "-inf", "1e400"):  # 1e400 → inf
            with self.subTest(value=val):
                self.assertEqual(
                    self._reload_with_env(val), _DEFAULT_FRESH,
                    f"{val!r} should clamp to _DEFAULT_FRESH",
                )

    def test_env_var_clamps_zero_and_negative(self):
        """``0``, ``-1``, ``-100`` are <=0 → must fall back to default.
        Without the clamp, a 0/negative fresh window would never classify
        anything as fresh, defeating the peer-protection window."""
        from cozempic.spawn_lock import _DEFAULT_FRESH

        for val in ("0", "0.0", "-1", "-100", "-0.001"):
            with self.subTest(value=val):
                self.assertEqual(
                    self._reload_with_env(val), _DEFAULT_FRESH,
                    f"{val!r} should clamp to _DEFAULT_FRESH",
                )

    def test_env_var_clamps_above_max(self):
        """Values > _FRESH_MAX (300s) must clamp to default. Reason:
        legitimate slow-Popen scenarios live well under 5 minutes; values
        higher are almost certainly typos (e.g. ``COZEMPIC_PIDFILE_FRESH_SECONDS=300000``
        meaning "1e10 ns" or similar) that would let stale pidfiles from
        crashed prior spawns block new ones for hours."""
        from cozempic.spawn_lock import _DEFAULT_FRESH, _FRESH_MAX

        for val in (str(_FRESH_MAX + 1), "1e10", "999999"):
            with self.subTest(value=val):
                self.assertEqual(
                    self._reload_with_env(val), _DEFAULT_FRESH,
                    f"{val!r} should clamp to _DEFAULT_FRESH",
                )

    def test_env_var_nan_clamps(self):
        """``nan`` is not finite per IEEE-754 → must fall back to default."""
        from cozempic.spawn_lock import _DEFAULT_FRESH

        self.assertEqual(
            self._reload_with_env("nan"), _DEFAULT_FRESH,
            "nan should clamp to _DEFAULT_FRESH",
        )

    def test_env_var_garbage_clamps(self):
        """Non-numeric junk (``abc``, empty-after-strip, mixed) → default."""
        from cozempic.spawn_lock import _DEFAULT_FRESH

        for val in ("abc", "5seconds", "five", "  "):
            with self.subTest(value=val):
                self.assertEqual(
                    self._reload_with_env(val), _DEFAULT_FRESH,
                    f"{val!r} should clamp to _DEFAULT_FRESH",
                )

    def test_env_var_valid_value_respected(self):
        """A value in the allowed (0, _FRESH_MAX] range must be honored
        unchanged. This is the happy path operators rely on for slow CI."""
        for val in ("1", "10", "30", "300", "0.1"):
            with self.subTest(value=val):
                self.assertEqual(
                    self._reload_with_env(val), float(val),
                    f"{val!r} should be respected as-is",
                )

    def test_env_var_unset_uses_default(self):
        """No env var set → default applies (regression check on the
        ``raw is None`` branch)."""
        from cozempic.spawn_lock import _DEFAULT_FRESH

        self.assertEqual(self._reload_with_env(None), _DEFAULT_FRESH)


class TestSymlinkDefense(unittest.TestCase):
    """Round-3 C1 regression test: the post-Popen ``.pid.tmp`` write must
    NOT follow symlinks. A local user with write access to ``/tmp`` (a
    real condition on shared dev boxes / CI runners) could pre-plant
    ``/tmp/cozempic_guard_<sid12>.pid.tmp`` as a symlink to ``~/.zshrc``
    or ``~/.ssh/authorized_keys``; the prior ``Path.write_text`` followed
    the symlink and overwrote the target with the PID number.

    Post-fix the write uses ``os.open(O_CREAT|O_EXCL|O_WRONLY|O_NOFOLLOW)``,
    which:
      - rejects symlinks with ``OSError`` (ELOOP on Linux, EEXIST on macOS
        since O_EXCL fires first when the path exists at all)
      - rejects pre-existing regular files too (orphan ``.pid.tmp`` from
        a crashed prior spawn)
      - leaves the victim file untouched in both cases
    """

    SESSION_ID = "ca7ec0de-1234-5678-9abc-2026051811f0"

    def setUp(self):
        from cozempic.guard import _pid_file_for_session

        self.pid_path = _pid_file_for_session(self.SESSION_ID)
        self.tmp_pid_path = self.pid_path.with_suffix(".pid.tmp")
        self.log_path = self.pid_path.with_suffix(".log")
        # Clean slate
        for p in (self.pid_path, self.tmp_pid_path, self.log_path):
            try:
                p.unlink()
            except (FileNotFoundError, OSError):
                pass
        # Victim file lives in a separate tmpdir so the test doesn't
        # depend on any real file in /tmp.
        import tempfile

        self.tmpdir = Path(tempfile.mkdtemp(prefix="cozempic_symlink_test_"))
        self.victim = self.tmpdir / "victim.txt"
        self.victim.write_text("ORIGINAL\n", encoding="utf-8")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)
        for p in (self.pid_path, self.tmp_pid_path, self.log_path):
            try:
                p.unlink()
            except (FileNotFoundError, OSError):
                pass

    def test_pid_tmp_write_rejects_symlink(self):
        """Attacker pre-plants .pid.tmp as a symlink to a victim file.
        The post-Popen atomic-rename write path MUST refuse it and leave
        the victim untouched."""
        from unittest.mock import patch

        from cozempic.guard import start_guard_daemon

        # Plant the symlink BEFORE start_guard_daemon runs.
        os.symlink(str(self.victim), str(self.tmp_pid_path))
        self.assertTrue(
            self.tmp_pid_path.is_symlink(),
            "test setup failed: symlink not in place",
        )

        class _DummyProc:
            def __init__(self, pid):
                self.pid = pid

        with (
            patch(
                "cozempic.guard.subprocess.Popen",
                side_effect=lambda *a, **k: _DummyProc(99999),
            ),
            patch("cozempic.guard.find_claude_pid", return_value=12345),
            patch("cozempic.guard._cleanup_legacy_pid"),
        ):
            result = start_guard_daemon(
                cwd=str(self.tmpdir),
                session_id=self.SESSION_ID,
                threshold_tokens=1000,
            )

        # The spawn body raised OSError on the os.open(O_EXCL|O_NOFOLLOW)
        # and surfaced a structured failure (started=False, reason="pidfile:
        # ..."). The DaemonSpawnClaim's __exit__ unlinked the parent-PID
        # claim file we wrote at _claim time (handed_off=False).
        self.assertFalse(
            result.get("started"),
            f"Spawn should have FAILED — symlink defense breached. " f"Got: {result!r}",
        )
        self.assertIn(
            "reason",
            result,
            f"Failure path must carry a structured reason; got {result!r}",
        )
        # CRITICAL: the victim file must be untouched.
        self.assertEqual(
            self.victim.read_text(encoding="utf-8"),
            "ORIGINAL\n",
            "Victim file was overwritten — C1 symlink TOCTOU regression. "
            "The .pid.tmp write must use O_NOFOLLOW (or fail closed on "
            "EEXIST when the path already exists, which O_EXCL guarantees).",
        )


class TestFileNotFoundErrorRecovery(unittest.TestCase):
    """If the log file's parent dir vanishes mid-spawn, the daemon must
    recover with one retry — not crash the SessionStart hook."""

    SESSION_ID = "feed1234-5678-9abc-def0-2026051811ee"

    def setUp(self):
        from cozempic.guard import _pid_file_for_session

        self.pid_path = _pid_file_for_session(self.SESSION_ID)
        self.pid_path.unlink(missing_ok=True)
        self.log_path = self.pid_path.with_suffix(".log")
        self.log_path.unlink(missing_ok=True)

    def tearDown(self):
        self.pid_path.unlink(missing_ok=True)
        self.log_path.unlink(missing_ok=True)
        self.pid_path.with_suffix(".pid.tmp").unlink(missing_ok=True)

    def test_filenotfounderror_recovery(self):
        """First open(log_file) raises FileNotFoundError, second succeeds."""
        from cozempic.guard import start_guard_daemon

        # We track open() calls on the log_file path; the FIRST call raises,
        # the rest are passed through. Mock os.makedirs to confirm the retry
        # path is taken.
        real_open = open
        call_state = {"n": 0}

        def fake_open(path, *args, **kwargs):
            if str(path) == str(self.log_path):
                call_state["n"] += 1
                if call_state["n"] == 1:
                    raise FileNotFoundError(2, "No such file or directory", str(path))
            return real_open(path, *args, **kwargs)

        with (
            patch("cozempic.guard.subprocess.Popen") as mock_popen,
            patch("cozempic.guard.find_claude_pid", return_value=12345),
            patch("cozempic.guard._cleanup_legacy_pid"),
            patch("builtins.open", side_effect=fake_open),
            patch("cozempic.guard.os.makedirs") as mock_makedirs,
        ):
            mock_popen.return_value.pid = 77777
            result = start_guard_daemon(
                cwd=os.getcwd(),
                session_id=self.SESSION_ID,
                threshold_tokens=1000,
            )

        self.assertTrue(
            result.get("started"),
            f"Daemon failed to recover from missing log dir: {result!r}",
        )
        self.assertEqual(
            call_state["n"],
            2,
            "Expected exactly 2 open() calls on log file (1 fail + 1 retry); "
            f"got {call_state['n']}.",
        )
        mock_makedirs.assert_called_once()


class TestDaemonSpawnLockUnit(unittest.TestCase):
    """Direct contract tests on the spawn_lock module."""

    def test_lock_yields_path(self):
        """Post-V4-rework: the spawn claim writes to the .pid file directly
        (the PID file IS the lock, no separate sentinel). This pins that
        contract."""
        from cozempic.spawn_lock import daemon_spawn_lock

        sid = "deadbeef-1234-5678-9abc-de00deadbeef"
        with daemon_spawn_lock(sid) as lock_path:
            self.assertIsInstance(lock_path, Path)
            self.assertIn("cozempic_guard_", lock_path.name)
            self.assertTrue(
                lock_path.name.endswith(".pid"),
                f"V4 rework: claim path must be the .pid file, got {lock_path.name}",
            )

    def test_double_acquire_raises(self):
        """A second concurrent acquire MUST raise DaemonAlreadyStarting."""
        from cozempic.spawn_lock import DaemonAlreadyStarting, daemon_spawn_lock

        sid = "deadbeef-1234-5678-9abc-de00cafebabe"
        with daemon_spawn_lock(sid):
            with self.assertRaises(DaemonAlreadyStarting):
                with daemon_spawn_lock(sid):
                    pass

    def test_release_allows_reacquire(self):
        """After the first lock exits, a fresh acquire succeeds."""
        from cozempic.spawn_lock import daemon_spawn_lock

        sid = "deadbeef-1234-5678-9abc-de00f00dbabe"
        with daemon_spawn_lock(sid):
            pass
        # Should not raise
        with daemon_spawn_lock(sid):
            pass


class TestC2_SlugConvergence(unittest.TestCase):
    """Round-3 C2 regression test: the bash hook session-id sanitizer and the
    Python ``_pid_file_for_session`` validator MUST agree on every input —
    same accept/reject decision, and same 12-char slug when accepted.

    The post-Round-3 implementer's intent (per commit e124614 message and
    the Option-B path agreed with team-lead + code-auditor) was to
    converge by RELAXING Python rather than tightening bash. Today:

      - bash:   ``re.sub(r'[^a-z0-9_-]', '_', s.lower())`` then ``[:12]``
                (always returns a slug; non-``[a-z0-9_-]`` chars become ``_``)
      - python: ``re.fullmatch(r'^[a-z0-9][a-z0-9_-]{11,}$', s.lower())``
                (rejects if shape doesn't match; raises ValueError)

    These two procedures CONVERGE on inputs already in the ``[a-z0-9_-]``
    alphabet (≥12 chars, alphanumeric first char) and DIVERGE on inputs
    containing other characters (bash substitutes to ``_``, Python rejects).
    This test pins BOTH halves of the contract so any future drift in
    either side surfaces immediately.
    """

    @staticmethod
    def _bash_hook_slug(session_id: str) -> str:
        """Reproduce the bash hook's actual Python one-liner verbatim:

            s = json.load(sys.stdin).get('session_id','').lower()
            print(re.sub(r'[^a-z0-9_-]', '_', s))

        Then the shell takes ``SLUG=$(printf '%.12s' "$SESSION_ID")`` as the
        slug (POSIX-safe, #168). So the composite contract is unchanged:
        lowercase → sub non-``[a-z0-9_-]`` with
        ``_`` → take first 12 chars. Always returns a string (possibly
        empty if input is empty); the shell's ``[ -n "$SESSION_ID" ]``
        gate is what skips the spawn for empty inputs.
        """
        import re

        s = (session_id or "").lower()
        return re.sub(r"[^a-z0-9_-]", "_", s)[:12]

    @staticmethod
    def _python_slug(session_id: str) -> str | None:
        """Reproduce the Python validator's accept/reject contract:

        _pid_file_for_session raises ValueError on bad shape; the
        caller-facing ``_is_guard_running_for_session`` catches that
        and returns None. We report None ↔ ValueError, else the
        12-char prefix that ends up in the pid path.
        """
        from cozempic.guard import _pid_file_for_session

        try:
            p = _pid_file_for_session(session_id)
        except ValueError:
            return None
        # Path is /tmp/cozempic_guard_<sid12>.pid — extract sid12.
        name = p.name
        prefix = "cozempic_guard_"
        suffix = ".pid"
        self_assert_invariant = name.startswith(prefix) and name.endswith(suffix)
        assert self_assert_invariant, f"unexpected pid path shape: {name!r}"
        return name[len(prefix) : -len(suffix)]

    # ─────────────────────────────────────────────────────────────────────
    # Inputs in the SHARED alphabet (alphanumeric + `_` + `-`, leading
    # alphanumeric, ≥12 chars). For these, bash and python MUST produce
    # the SAME 12-char slug. Any divergence here is a real C2 regression.
    # ─────────────────────────────────────────────────────────────────────
    SHARED_ALPHABET_TABLE = [
        # (description, session_id, expected_slug)
        ("canonical uuid v4", "aabbccdd-1122-4455-8899-2026051811bb", "aabbccdd-112"),
        (
            "uppercase uuid → lowercased",
            "AABBCCDD-1122-4455-8899-2026051811BB",
            "aabbccdd-112",
        ),
        ("hex-only minimum length (12)", "0123456789ab", "0123456789ab"),
        ("hex-only longer", "0123456789abcdef", "0123456789ab"),
        # `_` is in the shared alphabet — bash accepts and python accepts.
        ("underscores (in shared alphabet)", "abcd_1234_5678", "abcd_1234_56"),
        # Leading alphanumeric is required by Python; bash's `[:12]` would
        # still produce a slug here but it would be a leading-dash slug that
        # collides with truncation of unrelated inputs (the dash-collision
        # security property pinned by TestPolishV2_SessionIdRegexRequiresHexFirstChar).
        # So a leading-dash input goes in the DIVERGENT_TABLE below, not here.
    ]

    # ─────────────────────────────────────────────────────────────────────
    # Inputs OUTSIDE the shared alphabet. The Round-3 implementer chose
    # NOT to fully converge: bash produces a slug (substitutes non-alphabet
    # chars with `_`), Python raises ValueError. This table documents the
    # accepted divergence so a future change that ACCIDENTALLY converges
    # one way without the other is detected. The test asserts the CURRENT
    # asymmetry holds — if both sides flip to accept-or-reject this needs
    # an explicit update + a team-lead review.
    # ─────────────────────────────────────────────────────────────────────
    DIVERGENT_TABLE = [
        # (description, session_id, expected_bash_slug, expected_python_slug_or_None)
        (
            "contains slash (path traversal attempt)",
            "abcd/1234/5678",
            "abcd_1234_56",
            None,
        ),
        (
            "contains dot (extension attempt)",
            "abcd.1234.5678",
            "abcd_1234_56",
            None,
        ),
        ("contains space", "abcd 1234 5678", "abcd_1234_56", None),
        # Uppercase letters that are NOT hex digits get lowercased first by
        # bash, so they're already valid lowercase alphanumeric and python
        # accepts them too (post-Round-3 relaxation). Confirm.
        ("uppercase Z lowercased to z", "abcdef1234ZZ", "abcdef1234zz", "abcdef1234zz"),
        ("uppercase G lowercased to g", "abcdef1234GG", "abcdef1234gg", "abcdef1234gg"),
        ("starts with dash (anchor violation)", "-abcdef123456", "-abcdef12345", None),
        ("path traversal ..", "../etc/passwd", "___etc_passw", None),
        ("absolute path", "/etc/passwd00", "_etc_passwd0", None),
        ("non-ascii unicode", "abcdefÿ12345678", "abcdef_12345", None),
    ]

    # Inputs that BOTH sides effectively reject (bash via the empty-string
    # gate `[ -n "$SESSION_ID" ]`, python via ValueError).
    BOTH_REJECT_TABLE = [
        ("empty string", "", "", None),  # bash returns "" → fails -n gate
        ("too short (11 chars)", "0123456789a", "0123456789a", None),
        ("just one char", "a", "a", None),
    ]

    def test_shared_alphabet_inputs_produce_same_slug(self):
        """For inputs in the shared alphabet, bash and python MUST produce
        byte-equal slugs. Divergence here = C2 regression."""
        divergences = []
        for desc, sid, expected in self.SHARED_ALPHABET_TABLE:
            bash_out = self._bash_hook_slug(sid)
            py_out = self._python_slug(sid)
            if bash_out != py_out or py_out != expected:
                divergences.append(
                    {
                        "desc": desc,
                        "input": sid,
                        "bash": bash_out,
                        "python": py_out,
                        "expected": expected,
                    }
                )
        if divergences:
            self.fail(
                f"C2 regression in SHARED alphabet ({len(divergences)}/"
                f"{len(self.SHARED_ALPHABET_TABLE)}):\n"
                + "\n".join(repr(d) for d in divergences)
            )

    def test_divergent_inputs_match_current_round3_intent(self):
        """For inputs outside the shared alphabet, the Round-3 fix-pack
        deliberately keeps bash substituting (→ slug) while python rejects
        (→ ValueError). Pin that contract so a future tightening of bash
        OR loosening of python OR drift in either is caught explicitly.
        Any change here requires a team-lead review of the security
        implications.
        """
        divergences = []
        for desc, sid, expected_bash, expected_py in self.DIVERGENT_TABLE:
            bash_out = self._bash_hook_slug(sid)
            py_out = self._python_slug(sid)
            if bash_out != expected_bash or py_out != expected_py:
                divergences.append(
                    {
                        "desc": desc,
                        "input": sid,
                        "bash": bash_out,
                        "expected_bash": expected_bash,
                        "python": py_out,
                        "expected_python": expected_py,
                    }
                )
        if divergences:
            self.fail(
                f"DIVERGENT-table contract drift ({len(divergences)}/"
                f"{len(self.DIVERGENT_TABLE)}):\n"
                + "\n".join(repr(d) for d in divergences)
            )

    def test_both_reject_table(self):
        """Inputs both bash (via empty-string gate) and python (via
        ValueError) reject. Bash returns a substring; python returns None."""
        divergences = []
        for desc, sid, expected_bash, expected_py in self.BOTH_REJECT_TABLE:
            bash_out = self._bash_hook_slug(sid)
            py_out = self._python_slug(sid)
            if bash_out != expected_bash or py_out != expected_py:
                divergences.append(
                    {
                        "desc": desc,
                        "input": sid,
                        "bash": bash_out,
                        "expected_bash": expected_bash,
                        "python": py_out,
                        "expected_python": expected_py,
                    }
                )
        if divergences:
            self.fail(
                f"BOTH-REJECT contract drift ({len(divergences)}/"
                f"{len(self.BOTH_REJECT_TABLE)}):\n"
                + "\n".join(repr(d) for d in divergences)
            )

    def test_real_hook_json_carries_expected_sanitizer(self):
        """Drift-detection: read the actual hooks.json shipped in the package
        and assert the bash side still uses the substitute-pattern
        ``re.sub(r'[^a-z0-9_-]', '_', s.lower())``. If a future commit
        retightens bash to use re.match instead, the DIVERGENT_TABLE
        contract above will silently change shape — this assertion catches
        the drift at the source-of-truth level.
        """
        # cozempic.data is a namespace package — __file__ is None.
        # Use cozempic.__file__ then walk to the sibling data/ dir.
        import cozempic

        hooks_path = Path(cozempic.__file__).parent / "data" / "hooks.json"
        self.assertTrue(
            hooks_path.exists(),
            f"hooks.json not found at {hooks_path}",
        )
        body = hooks_path.read_text(encoding="utf-8")
        self.assertIn(
            r"re.sub(r'[^a-z0-9_-]','_',s)",
            body,
            "hooks.json bash sanitiser drifted away from the documented "
            "substitute-pattern. Re-check Round-3 C2 contract and update "
            "DIVERGENT_TABLE expectations.",
        )


if __name__ == "__main__":
    unittest.main()
