"""Tests for JsonlWatcher bugs W1-W5.

W1 — kqueue fd must be closed deterministically (not relying on GC / CPython refcount)
W2 — FileNotFoundError crash when file missing at start
W3 — _last_size not reset after shrink → growth blindness post-prune
W4 — on_growth exceptions silently swallowed; must log to stderr (one-shot)
W5 — kqueue watches orphaned inode after os.replace → blind post-prune (kqueue only)
W6 — module-level _HAS_KQUEUE constant (no __import__ idiom in __init__)
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, call, patch

from cozempic.watcher import JsonlWatcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_watcher(path: str, callback, *, use_kqueue: bool) -> JsonlWatcher:
    """Create a JsonlWatcher with an overridden _use_kqueue flag."""
    w = JsonlWatcher(path, callback)
    w._use_kqueue = use_kqueue
    return w


def _start_thread(watcher: JsonlWatcher) -> threading.Thread:
    t = threading.Thread(target=watcher.start, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# W1 — kqueue fd must close deterministically; must not rely on GC
# ---------------------------------------------------------------------------

@unittest.skipUnless(sys.platform == "darwin", "kqueue macOS only")
class TestJsonlWatcherFdLeak(unittest.TestCase):
    """W1: kq.close() must be called in finally — not left to GC.

    Tests use mock-spy so they are deterministic and RED if kq.close() is
    removed from the finally block (CPython GC happens to call __del__ in
    the simple case, making lsof-count tests unreliable as regression guards).
    """

    def test_kqueue_close_called_after_normal_stop(self):
        """kq.close() must be called deterministically when watcher stops normally.

        select.kqueue is a C extension — its .close attribute is read-only, so we
        cannot monkey-patch the method on the instance directly. Instead, wrap the
        real kqueue object in a MagicMock that proxies control() and tracks close().
        """
        import select as _sel

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as f:
            path = f.name
        try:
            close_called = threading.Event()

            real_kqueue_cls = _sel.kqueue

            def spy_kqueue():
                real_kq = real_kqueue_cls()
                mock_kq = MagicMock()
                # Proxy control() through to the real kqueue so the watcher loop works
                mock_kq.control.side_effect = real_kq.control
                # Track close() and also close the real fd to avoid leaking it
                def tracked_close():
                    close_called.set()
                    real_kq.close()
                mock_kq.close.side_effect = tracked_close
                return mock_kq

            with patch("select.kqueue", side_effect=spy_kqueue):
                w = _make_watcher(path, lambda p, s: None, use_kqueue=True)
                t = _start_thread(w)
                time.sleep(0.2)
                w.stop()
                t.join(timeout=2)

            self.assertTrue(close_called.is_set(),
                "kq.close() never called after normal stop. "
                "W1 not fixed: deterministic close absent (matters on PyPy / ref-cycles).")
        finally:
            os.unlink(path)

    def test_kqueue_fd_closed_if_kqueue_constructor_raises(self):
        """If select.kqueue() raises, the file fd opened before it must still be closed.

        This is H-2: `kq = select.kqueue()` sits outside the try whose finally
        closes fd — if kqueue() raises (EMFILE / OSError), fd leaks.
        Fix requires kq = None sentinel so finally always closes fd.
        """
        import select as _sel

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as f:
            path = f.name
        try:
            fd_closed = threading.Event()
            real_os_close = os.close

            def spy_close(fd: int) -> None:
                fd_closed.set()
                real_os_close(fd)

            with patch("select.kqueue", side_effect=OSError("EMFILE simulated")), \
                 patch("os.close", side_effect=spy_close):
                w = _make_watcher(path, lambda p, s: None, use_kqueue=True)
                t = _start_thread(w)
                t.join(timeout=2)

            self.assertTrue(fd_closed.is_set(),
                "os.close(fd) not called after select.kqueue() raised OSError. "
                "H-2: fd leaks when kqueue constructor fails.")
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# W2 — missing-file crash / fallback to poll
# ---------------------------------------------------------------------------

@unittest.skipUnless(sys.platform == "darwin", "kqueue macOS only")
class TestJsonlWatcherMissingFileKqueue(unittest.TestCase):
    """W2 (kqueue): os.open on missing file must fall back to poll, not crash."""

    def test_kqueue_falls_back_to_poll_when_file_missing(self):
        """Watcher thread must stay alive when file doesn't exist at start (kqueue)."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nonexistent.jsonl")
            w = _make_watcher(path, lambda p, s: None, use_kqueue=True)
            t = _start_thread(w)
            t.join(timeout=1.0)
            # Thread should still be alive (fell back to poll loop, which returns 0 for OSError)
            self.assertTrue(t.is_alive(),
                "Watcher thread died immediately on missing file (W2 not fixed: os.open unguarded)")
            w.stop()
            t.join(timeout=2)


class TestJsonlWatcherMissingFilePoll(unittest.TestCase):
    """W2 (poll): poll path already handles missing file — regression guard."""

    def test_poll_handles_missing_file_gracefully(self):
        """Poll-mode watcher must stay alive when file doesn't exist."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nonexistent.jsonl")
            w = _make_watcher(path, lambda p, s: None, use_kqueue=False)
            t = _start_thread(w)
            time.sleep(0.35)
            self.assertTrue(t.is_alive(),
                "Poll watcher thread died on missing file")
            w.stop()
            t.join(timeout=2)


# ---------------------------------------------------------------------------
# W3 — shrink-reset: _last_size must reset after file shrinks
# ---------------------------------------------------------------------------

class TestJsonlWatcherShrinkResetPoll(unittest.TestCase):
    """W3 (poll path): growth after shrink must fire callback."""

    def test_growth_detected_after_file_shrinks_poll(self):
        """After prune shrinks file, next growth must trigger on_growth."""
        fired: list[int] = []

        def cb(path: str, size: int) -> None:
            fired.append(size)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl",
                                         mode="wb") as f:
            path = f.name
            f.write(b"x" * 1000)

        try:
            w = _make_watcher(path, cb, use_kqueue=False)
            # _last_size captured at __init__ = 1000

            t = _start_thread(w)
            time.sleep(0.05)  # let thread enter loop

            # Simulate prune: shrink file to 100 bytes
            with open(path, "wb") as f:
                f.write(b"y" * 100)
            time.sleep(0.5)  # wait for poll to detect shrink

            # Now grow: append 200 more bytes (total 300)
            with open(path, "ab") as f:
                f.write(b"z" * 200)
            time.sleep(0.5)  # wait for growth detection

            w.stop()
            t.join(timeout=2)

            self.assertTrue(
                any(s >= 300 for s in fired),
                f"on_growth did not fire after shrink+regrowth. "
                f"fired={fired}. W3 not fixed: _last_size not reset on shrink.",
            )
        finally:
            os.unlink(path)

    def test_shrink_alone_does_not_fire_callback(self):
        """Shrinking the file must NOT trigger on_growth."""
        fired: list[int] = []

        def cb(path: str, size: int) -> None:
            fired.append(size)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl",
                                         mode="wb") as f:
            path = f.name
            f.write(b"x" * 1000)

        try:
            w = _make_watcher(path, cb, use_kqueue=False)
            t = _start_thread(w)
            time.sleep(0.05)

            # Only shrink
            with open(path, "wb") as f:
                f.write(b"y" * 100)
            time.sleep(0.5)

            w.stop()
            t.join(timeout=2)

            self.assertEqual(fired, [],
                f"on_growth fired on shrink (false positive). fired={fired}")
        finally:
            os.unlink(path)


@unittest.skipUnless(sys.platform == "darwin", "kqueue macOS only")
class TestJsonlWatcherShrinkResetKqueue(unittest.TestCase):
    """W3 (kqueue path): growth after shrink must fire callback."""

    def test_kqueue_growth_detected_after_file_shrinks(self):
        """After prune shrinks file, kqueue path must detect next growth."""
        fired: list[int] = []
        event = threading.Event()

        def cb(path: str, size: int) -> None:
            fired.append(size)
            event.set()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl",
                                         mode="wb") as f:
            path = f.name
            f.write(b"x" * 1000)

        try:
            w = _make_watcher(path, cb, use_kqueue=True)
            t = _start_thread(w)
            time.sleep(0.1)

            # Shrink file
            with open(path, "wb") as f:
                f.write(b"y" * 100)
            time.sleep(0.2)

            # Grow past 100 bytes
            with open(path, "ab") as f:
                f.write(b"z" * 300)

            event.wait(timeout=3.0)
            w.stop()
            t.join(timeout=2)

            self.assertTrue(
                any(s >= 300 for s in fired),
                f"kqueue on_growth not fired after shrink+regrowth. "
                f"fired={fired}. W3 not fixed in kqueue path.",
            )
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# W4 — callback errors must be logged to stderr (one-shot)
# ---------------------------------------------------------------------------

class TestJsonlWatcherCallbackErrorLoggedPoll(unittest.TestCase):
    """W4 (poll): on_growth exception must print to stderr, not silently pass."""

    def test_callback_error_logged_to_stderr_poll(self):
        """First callback error must appear on stderr."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl",
                                         mode="wb") as f:
            path = f.name
            f.write(b"a" * 100)

        try:
            logged = threading.Event()

            def bad_cb(path: str, size: int) -> None:
                raise RuntimeError("boom from test")

            w = _make_watcher(path, bad_cb, use_kqueue=False)

            captured = io.StringIO()
            # Patch sys.stderr in the watcher module so the background thread sees it
            import cozempic.watcher as watcher_mod
            old_stderr = watcher_mod.sys.stderr
            watcher_mod.sys.stderr = captured

            try:
                t = _start_thread(w)
                with open(path, "ab") as f:
                    f.write(b"b" * 200)
                time.sleep(0.6)
                w.stop()
                t.join(timeout=2)
            finally:
                watcher_mod.sys.stderr = old_stderr

            err_output = captured.getvalue()
            self.assertIn("[watcher]", err_output,
                "Expected '[watcher]' in stderr. W4 not fixed: error silently swallowed.")
            self.assertIn("on_growth error", err_output,
                "Expected 'on_growth error' in stderr. W4 not fixed.")
        finally:
            os.unlink(path)

    def test_callback_error_does_not_crash_watcher_poll(self):
        """Watcher thread must survive a callback that raises."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl",
                                         mode="wb") as f:
            path = f.name
            f.write(b"a" * 100)

        try:
            def bad_cb(path: str, size: int) -> None:
                raise RuntimeError("crash me")

            w = _make_watcher(path, bad_cb, use_kqueue=False)
            t = _start_thread(w)
            time.sleep(0.05)

            with open(path, "ab") as f:
                f.write(b"b" * 200)
            time.sleep(0.5)

            self.assertTrue(t.is_alive(),
                "Watcher thread died after callback raised (W4 regression)")
            w.stop()
            t.join(timeout=2)
        finally:
            os.unlink(path)

    def test_one_shot_error_log_not_repeated_on_each_growth(self):
        """Repeated callback errors must only log ONCE (one-shot pattern)."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl",
                                         mode="wb") as f:
            path = f.name
            f.write(b"a" * 50)

        try:
            def bad_cb(path: str, size: int) -> None:
                raise RuntimeError("persistent failure")

            w = _make_watcher(path, bad_cb, use_kqueue=False)

            captured = io.StringIO()
            import cozempic.watcher as watcher_mod
            old_stderr = watcher_mod.sys.stderr
            watcher_mod.sys.stderr = captured

            try:
                t = _start_thread(w)
                for i in range(3):
                    with open(path, "ab") as f:
                        f.write(b"x" * 100)
                    time.sleep(0.4)
                w.stop()
                t.join(timeout=2)
            finally:
                watcher_mod.sys.stderr = old_stderr

            err_output = captured.getvalue()
            occurrences = err_output.count("[watcher]")
            self.assertEqual(occurrences, 1,
                f"Expected exactly 1 '[watcher]' log line, got {occurrences}. "
                "One-shot suppression (W4) not implemented.")
        finally:
            os.unlink(path)

    def test_error_log_resets_after_successful_callback(self):
        """_cb_error_logged must reset to False after a successful on_growth call.

        Without reset, one transient error permanently silences logging for
        an hours-long daemon. Reset on success re-enables visibility after recovery.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl",
                                         mode="wb") as f:
            path = f.name
            f.write(b"a" * 50)

        try:
            call_count = [0]
            # First call raises, second call succeeds, third call raises again
            def intermittent_cb(path: str, size: int) -> None:
                n = call_count[0]
                call_count[0] += 1
                if n == 0 or n == 2:
                    raise RuntimeError(f"transient error #{n}")
                # n == 1: success — should reset _cb_error_logged

            w = _make_watcher(path, intermittent_cb, use_kqueue=False)

            captured = io.StringIO()
            import cozempic.watcher as watcher_mod
            old_stderr = watcher_mod.sys.stderr
            watcher_mod.sys.stderr = captured

            try:
                t = _start_thread(w)
                # Trigger 3 growth events: error, success, error
                for i in range(3):
                    with open(path, "ab") as f:
                        f.write(b"x" * 100)
                    time.sleep(0.4)
                w.stop()
                t.join(timeout=2)
            finally:
                watcher_mod.sys.stderr = old_stderr

            err_output = captured.getvalue()
            occurrences = err_output.count("[watcher]")
            # First error logged, success resets flag, third error logged again → 2
            self.assertEqual(occurrences, 2,
                f"Expected 2 '[watcher]' log lines (reset on success), got {occurrences}. "
                "M-3 not fixed: _cb_error_logged not reset after successful callback.")
        finally:
            os.unlink(path)


@unittest.skipUnless(sys.platform == "darwin", "kqueue macOS only")
class TestJsonlWatcherCallbackErrorLoggedKqueue(unittest.TestCase):
    """W4 (kqueue): callback errors must log to stderr."""

    def test_callback_error_logged_to_stderr_kqueue(self):
        """kqueue path: first callback error must appear on stderr."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl",
                                         mode="wb") as f:
            path = f.name
            f.write(b"a" * 100)

        try:
            def bad_cb(path: str, size: int) -> None:
                raise RuntimeError("kqueue boom")

            w = _make_watcher(path, bad_cb, use_kqueue=True)

            captured = io.StringIO()
            import cozempic.watcher as watcher_mod
            old_stderr = watcher_mod.sys.stderr
            watcher_mod.sys.stderr = captured

            try:
                t = _start_thread(w)
                with open(path, "ab") as f:
                    f.write(b"b" * 500)
                time.sleep(1.5)
                w.stop()
                t.join(timeout=2)
            finally:
                watcher_mod.sys.stderr = old_stderr

            err_output = captured.getvalue()
            self.assertIn("[watcher]", err_output,
                "Expected '[watcher]' in stderr from kqueue path. W4 not fixed.")
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# W5 — inode replacement: kqueue must survive os.replace, fall back to poll
# ---------------------------------------------------------------------------

@unittest.skipUnless(sys.platform == "darwin", "kqueue macOS only")
class TestJsonlWatcherInodeReplacement(unittest.TestCase):
    """W5: after os.replace (atomic prune), watcher must still detect growth."""

    def test_kqueue_detects_growth_after_atomic_replace(self):
        """Growth after os.replace must fire on_growth (via poll fallback).

        The new file after os.replace is grown ABOVE the original size (800B > 500B)
        so the callback fires unambiguously regardless of what _last_size inherited
        from the kqueue phase.
        """
        fired: list[int] = []
        growth_after_replace = threading.Event()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl",
                                         mode="wb") as f:
            path = f.name
            f.write(b"x" * 500)  # original: 500 bytes

        try:
            def cb(p: str, size: int) -> None:
                fired.append(size)
                if size > 500:
                    growth_after_replace.set()

            w = _make_watcher(path, cb, use_kqueue=True)
            # _last_size = 500 at __init__
            t = _start_thread(w)
            time.sleep(0.2)  # let kqueue register

            # Simulate atomic prune: replace with 50B (shrink), then grow to 800B
            tmp_path = path + ".tmp"
            with open(tmp_path, "wb") as f:
                f.write(b"y" * 50)
            os.replace(tmp_path, path)   # DELETE/RENAME event → poll takeover
            time.sleep(0.3)              # let poll takeover complete

            # Grow new file well above original 500B
            with open(path, "ab") as f:
                f.write(b"z" * 750)     # total ~800B > 500B original

            growth_after_replace.wait(timeout=3.0)
            w.stop()
            t.join(timeout=2)

            self.assertTrue(
                growth_after_replace.is_set(),
                f"on_growth did not fire with size>500 after os.replace+growth. "
                f"fired={fired}. W5 not fixed: kqueue watches orphaned inode.",
            )
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            try:
                os.unlink(path + ".tmp")
            except FileNotFoundError:
                pass

    def test_kqueue_registers_delete_rename_fflags(self):
        """kqueue kevent fflags arg must include KQ_NOTE_DELETE and KQ_NOTE_RENAME.

        Uses mock to intercept select.kevent construction and inspect the fflags
        keyword argument. RED at base if KQ_NOTE_DELETE/RENAME absent from fflags.
        """
        import select as _sel

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl",
                                         mode="wb") as f:
            path = f.name
            f.write(b"x" * 100)

        try:
            kevent_calls: list[dict] = []
            real_kevent = _sel.kevent

            def spy_kevent(*args, **kwargs):
                kevent_calls.append(kwargs)
                return real_kevent(*args, **kwargs)

            with patch("select.kevent", side_effect=spy_kevent):
                w = _make_watcher(path, lambda p, s: None, use_kqueue=True)
                t = _start_thread(w)
                time.sleep(0.2)
                w.stop()
                t.join(timeout=2)

            self.assertTrue(kevent_calls, "select.kevent was never called (test setup issue)")
            fflags = kevent_calls[0].get("fflags", 0)
            self.assertTrue(fflags & _sel.KQ_NOTE_DELETE,
                f"KQ_NOTE_DELETE not in fflags={fflags:#x}. W5 incomplete.")
            self.assertTrue(fflags & _sel.KQ_NOTE_RENAME,
                f"KQ_NOTE_RENAME not in fflags={fflags:#x}. W5 incomplete.")
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# W6 — __import__ idiom: module-level constant must be used
# ---------------------------------------------------------------------------

class TestJsonlWatcherImportIdiom(unittest.TestCase):
    """W6: __import__("select") in __init__ must be replaced with module-level constant."""

    def test_module_has_has_kqueue_constant(self):
        """After fix, watcher module must expose _HAS_KQUEUE at module level."""
        import cozempic.watcher as watcher_mod
        self.assertTrue(
            hasattr(watcher_mod, "_HAS_KQUEUE"),
            "_HAS_KQUEUE module-level constant missing. W6 not fixed: "
            "__import__('select') idiom still in __init__.",
        )

    def test_init_uses_module_constant_not_import_idiom(self):
        """JsonlWatcher._use_kqueue must be derived from _HAS_KQUEUE, not inline __import__."""
        import cozempic.watcher as watcher_mod
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as f:
            path = f.name
        try:
            w = JsonlWatcher(path, lambda p, s: None)
            expected = watcher_mod._HAS_KQUEUE  # type: ignore[attr-defined]
            self.assertEqual(w._use_kqueue, expected,
                "_use_kqueue does not match _HAS_KQUEUE module constant.")
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Round-3 — OSError-robustness: any kqueue-path OSError must degrade to poll
# ---------------------------------------------------------------------------

@unittest.skipUnless(sys.platform == "darwin", "kqueue macOS only")
class TestJsonlWatcherOSErrorRobustness(unittest.TestCase):
    """kqueue path must fall back to poll on any OSError, not crash the daemon thread.

    Three scenarios:
    R1 — os.open raises PermissionError (subclass of OSError, not FileNotFoundError)
    R2 — select.kqueue() raises OSError (EMFILE) → poll fallback, thread stays alive
    R3 — kq.close() raises → os.close(fd) still runs (nested finally)
    """

    def test_permission_error_on_open_falls_back_to_poll(self):
        """PermissionError on os.open must fall back to poll, not crash the thread.

        Current code catches only FileNotFoundError; PermissionError propagates
        uncaught → thread dies. Fix: catch OSError (parent of both).
        """
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "watch.jsonl")
            with open(path, "wb") as f:
                f.write(b"x" * 100)

            with patch("os.open", side_effect=PermissionError("EACCES simulated")):
                w = _make_watcher(path, lambda p, s: None, use_kqueue=True)
                t = _start_thread(w)
                t.join(timeout=1.5)

            # Thread must still be alive — fell back to poll loop
            self.assertTrue(t.is_alive(),
                "Watcher thread died on PermissionError from os.open. "
                "R1 not fixed: only FileNotFoundError caught, not OSError.")
            w.stop()
            t.join(timeout=2)

    def test_kqueue_oserror_falls_back_to_poll(self):
        """OSError from select.kqueue() (e.g. EMFILE) must fall back to poll.

        Current behaviour: OSError propagates out of the try/finally that closes fd,
        then bubbles up through start() → thread dies. Fix: catch OSError around the
        kqueue+control-loop block and fall back to poll.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl",
                                          mode="wb") as f:
            path = f.name
            f.write(b"x" * 100)
        try:
            with patch("select.kqueue", side_effect=OSError("EMFILE simulated")):
                w = _make_watcher(path, lambda p, s: None, use_kqueue=True)
                t = _start_thread(w)
                t.join(timeout=1.5)

            # Thread must still be alive — fell back to poll loop
            self.assertTrue(t.is_alive(),
                "Watcher thread died on select.kqueue() OSError. "
                "R2 not fixed: OSError not caught; no poll fallback after kqueue failure.")
            w.stop()
            t.join(timeout=2)
        finally:
            os.unlink(path)

    def test_fd_closed_even_if_kq_close_raises(self):
        """os.close(fd) must run even if kq.close() raises an exception.

        Current finally block: kq.close() then os.close(fd) — if kq.close() raises,
        os.close(fd) is skipped. Fix: nest them so os.close(fd) is unconditional.
        """
        import select as _sel

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl",
                                          mode="wb") as f:
            path = f.name
            f.write(b"x" * 100)
        try:
            fd_closed = threading.Event()
            real_os_close = os.close

            def spy_close(fd: int) -> None:
                fd_closed.set()
                real_os_close(fd)

            real_kqueue_cls = _sel.kqueue

            def spy_kqueue():
                real_kq = real_kqueue_cls()
                mock_kq = MagicMock()
                mock_kq.control.side_effect = real_kq.control
                # kq.close() raises — simulates a misbehaving kqueue object
                def raising_close():
                    real_kq.close()
                    raise OSError("kq.close() failed")
                mock_kq.close.side_effect = raising_close
                return mock_kq

            with patch("select.kqueue", side_effect=spy_kqueue), \
                 patch("os.close", side_effect=spy_close):
                w = _make_watcher(path, lambda p, s: None, use_kqueue=True)
                t = _start_thread(w)
                time.sleep(0.2)
                w.stop()
                t.join(timeout=2)

            self.assertTrue(fd_closed.is_set(),
                "os.close(fd) not called when kq.close() raised. "
                "R3 not fixed: kq.close() must be in a nested try so os.close(fd) "
                "always runs unconditionally.")
        finally:
            os.unlink(path)

    def test_kq_close_oserror_does_not_kill_thread_and_polls(self):
        """kq.close() raising OSError in finally must NOT kill the thread.

        Gap in the current nested try/finally (R3):
            try:
                if kq is not None: kq.close()   # raises OSError here
            finally:
                os.close(fd)                     # still runs (R3 ok)
        ...but kq.close()'s OSError propagates out of the outer finally,
        killing the thread and skipping the poll-fallback tail line.

        Fix: swallow cleanup-close errors. Both closes are attempted and neither
        propagates; the poll fallback at the end of _watch_kqueue is always reached.

        This test combines with a kqueue()-EMFILE scenario so kqueue_error is True
        when kq.close() raises — ensuring poll fallback IS entered (b), not just
        that the thread survives a graceful stop.

        Asserts:
        (a) os.close(fd) is still called
        (b) _watch_poll IS entered (poll fallback runs even when cleanup raises)
        (c) thread stays alive / joins cleanly (does not die from cleanup exception)
        """
        import select as _sel

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl",
                                          mode="wb") as f:
            path = f.name
            f.write(b"x" * 100)
        try:
            fd_closed = threading.Event()
            poll_entered = threading.Event()
            real_os_close = os.close

            def spy_close(fd: int) -> None:
                fd_closed.set()
                real_os_close(fd)

            real_kqueue_cls = _sel.kqueue
            kqueue_call_count = [0]

            def spy_kqueue():
                kqueue_call_count[0] += 1
                real_kq = real_kqueue_cls()
                mock_kq = MagicMock()
                # First control() call raises OSError → kqueue_error=True → poll fallback
                # This ensures fall_back_to_poll is True when we reach the finally block
                call_n = [0]
                def raising_control(events, max_events, timeout):
                    call_n[0] += 1
                    if call_n[0] == 1:
                        raise OSError("kq.control() failed — EMFILE simulation")
                    return real_kq.control(events, max_events, timeout)
                mock_kq.control.side_effect = raising_control
                # kq.close() also raises — the key condition being tested
                def raising_close():
                    real_kq.close()
                    raise OSError("kq.close() misbehaved in cleanup")
                mock_kq.close.side_effect = raising_close
                return mock_kq

            w = _make_watcher(path, lambda p, s: None, use_kqueue=True)

            # Spy on _watch_poll to detect poll-fallback entry
            real_watch_poll = w._watch_poll
            def spy_poll():
                poll_entered.set()
                real_watch_poll()
            w._watch_poll = spy_poll

            with patch("select.kqueue", side_effect=spy_kqueue), \
                 patch("os.close", side_effect=spy_close):
                t = _start_thread(w)
                # Wait for poll fallback to be entered (or timeout)
                poll_entered.wait(timeout=2.0)
                w.stop()
                t.join(timeout=2)

            # (a) fd was closed despite kq.close() raising
            self.assertTrue(fd_closed.is_set(),
                "os.close(fd) not called when kq.close() raised. "
                "Cleanup swallow not implemented (addendum fix).")
            # (b) poll fallback was entered despite the cleanup exception
            self.assertTrue(poll_entered.is_set(),
                "_watch_poll not entered after kq.close() OSError. "
                "Addendum not fixed: OSError in cleanup kills thread before "
                "poll-fallback tail line is reached.")
            # (c) thread joined cleanly
            self.assertFalse(t.is_alive(),
                "Thread still alive after stop(). May be stuck in unexpected state.")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
