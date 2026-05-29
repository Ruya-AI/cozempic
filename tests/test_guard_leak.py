"""RED tests for guard memory leak fix (PR #89).

Captures the contracts that The incremental loader (`load_messages_incremental` + bounded cache)
and `_is_claude_process` mtime-recency fallback must satisfy.

Why these tests are RED on current main:
  - `cozempic.session.load_messages_incremental` does not exist yet (ImportError
    at test-collection time for every Test_* class that imports it).
  - `cozempic.guard._is_claude_process` does not accept `session_path=` yet.

Bug map (from AUDIT_REPORT_leak.md §5):
  Incremental-loader contracts
    - equivalence on append (cache returns same shape as full read)
    - rewrite detection via inode (os.replace)
    - size-shrink detection (in-place truncation)
    - partial-line tail safety (mid-write file)
    - bounded per-session cache (MAX_CACHED_MESSAGES eviction)
    - empirical RSS growth bound under hot-loop (checkpoint_team pattern)
    - thread-serialized cache access (no dup / no corruption)

  Claude-process mtime-fallback contracts
    - mtime-fresh JSONL corroborates liveness when ps drifts
    - aged mtime does not falsely corroborate

All tests use tmp_path fixtures; nothing touches real ~/.claude or ~/.claudes.
All thresholds are DELTAS, never absolutes — machine-portable.
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from cozempic.session import MAX_LINE_BYTES, load_messages


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _rss_bytes() -> int:
    """Current process RSS in bytes, platform-normalized.

    ru_maxrss is bytes on macOS, kilobytes on Linux. This returns bytes on both.
    Used for DELTA checks only; absolute value is not portable.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(raw)
    # Linux, BSD: kilobytes → bytes
    return int(raw) * 1024


def _write_jsonl(path: Path, n_lines: int, payload_bytes: int = 400) -> None:
    """Write `n_lines` synthetic JSONL messages of roughly `payload_bytes` each."""
    filler = "x" * max(payload_bytes - 64, 1)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_lines):
            f.write(json.dumps({"role": "user", "content": f"{i}:{filler}"}) + "\n")


def _append_jsonl(path: Path, n_lines: int, start_index: int, payload_bytes: int = 400) -> None:
    """Append `n_lines` messages to an existing JSONL, preserving newline boundary."""
    filler = "x" * max(payload_bytes - 64, 1)
    with open(path, "a", encoding="utf-8") as f:
        for i in range(start_index, start_index + n_lines):
            f.write(json.dumps({"role": "user", "content": f"{i}:{filler}"}) + "\n")


# ─── memory growth under hot-loop pattern ──────────────────────────


class Test_MemoryGrowth(unittest.TestCase):
    """Empirical RSS bound: 50 cycles of checkpoint-style read on a 20 MB JSONL.

    Pre-team repro (`/tmp/leak_repro_v2.py`): 120 cycles → +587 MB.
    50 cycles → +100 MB, flat.

    We assert the incremental-read bound (≤ 200 MB delta over 50 cycles) because the
    alternative (leak-proving that current load_messages grows >300 MB) cannot
    be RED at collection time — it would actually pass on current code iff the
    leak is deterministic, but the mechanical RED signal is the ImportError on
    `load_messages_incremental` (no such symbol on main). Keep the memory
    assertion as a GREEN-side regression guard once the impl lands.
    """

    def test_incremental_keeps_rss_bounded_over_many_cycles(self):
        from cozempic.session import load_messages_incremental  # RED: not defined on main

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "hot.jsonl"
            _write_jsonl(jsonl, n_lines=50_000, payload_bytes=400)  # ~20 MB

            baseline = _rss_bytes()
            for _ in range(50):
                load_messages_incremental(jsonl)
            delta = _rss_bytes() - baseline

        # Conservative upper bound — empirical is ~100 MB; give 2× headroom.
        self.assertLess(
            delta, 200 * 1024 * 1024,
            f"Incremental loader leaked {delta/1e6:.1f} MB over 50 cycles",
        )


# ─── incremental append contract ────────────────────────────────────


class Test_IncrementalAppend(unittest.TestCase):
    """Equivalence on append + minimal re-parsing.

    Audit contract 1 (§5.1): after growing, incremental result must equal
    full-read result, tuple-by-tuple (line_index, msg_dict, byte_size).

    Minimality contract: once the cache is warm, subsequent calls after an
    append must parse ONLY the new lines, not the whole file. We enforce this
    by mocking `json.loads` and counting calls on the second read.
    """

    def test_incremental_matches_full_read_after_append(self):
        from cozempic.session import load_messages_incremental  # RED: not defined on main

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "s.jsonl"
            _write_jsonl(jsonl, n_lines=10)
            first = load_messages_incremental(jsonl)
            self.assertEqual(len(first), 10)

            _append_jsonl(jsonl, n_lines=5, start_index=10)
            second = load_messages_incremental(jsonl)

            full = load_messages(jsonl)
            self.assertEqual(second, full, "incremental must match load_messages exactly after append")
            self.assertEqual(len(second), 15)

    def test_incremental_parses_only_new_lines_on_second_call(self):
        import json as _json
        from cozempic.session import load_messages_incremental  # RED: not defined on main

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "s.jsonl"
            _write_jsonl(jsonl, n_lines=10)
            load_messages_incremental(jsonl)  # warm cache

            _append_jsonl(jsonl, n_lines=3, start_index=10)

            real_loads = _json.loads
            call_count = {"n": 0}

            def counting_loads(s, *a, **kw):
                call_count["n"] += 1
                return real_loads(s, *a, **kw)

            with patch("cozempic.session.json.loads", side_effect=counting_loads):
                load_messages_incremental(jsonl)

            # Only 3 appended lines should be parsed — not the full 13.
            self.assertEqual(
                call_count["n"], 3,
                f"incremental re-parsed too much: {call_count['n']} json.loads calls (expected 3)",
            )


# ─── rewrite detection ──────────────────────────────────────────────


class Test_RewriteDetection(unittest.TestCase):
    """Inode change (os.replace) and size shrink must trigger full re-read.

    Audit contracts 2 & 3 (§5.2, §5.3). Prune cycle rewrites JSONL via
    os.replace → new inode. Rare in-place truncations → same inode, smaller
    size. Both must invalidate the cache.
    """

    def test_os_replace_triggers_full_reread(self):
        from cozempic.session import load_messages_incremental  # RED: not defined on main

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "s.jsonl"
            _write_jsonl(jsonl, n_lines=20)
            first = load_messages_incremental(jsonl)
            self.assertEqual(len(first), 20)

            # Replace atomically with a smaller file (simulates prune cycle).
            replacement = Path(tmp) / "s.new.jsonl"
            _write_jsonl(replacement, n_lines=7)
            os.replace(replacement, jsonl)

            second = load_messages_incremental(jsonl)
            self.assertEqual(len(second), 7, "cache must invalidate on inode change")
            self.assertEqual(second, load_messages(jsonl))

    def test_in_place_truncation_triggers_full_reread(self):
        from cozempic.session import load_messages_incremental  # RED: not defined on main

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "s.jsonl"
            _write_jsonl(jsonl, n_lines=20)
            load_messages_incremental(jsonl)  # warm

            # Truncate to first 5 lines WITHOUT changing inode.
            with open(jsonl, "r", encoding="utf-8") as f:
                lines = f.readlines()
            truncated = "".join(lines[:5])
            with open(jsonl, "r+", encoding="utf-8") as f:
                f.write(truncated)
                f.truncate()

            second = load_messages_incremental(jsonl)
            self.assertEqual(len(second), 5, "cache must invalidate on size shrink")


# ─── bounded cache ──────────────────────────────────────────────────


class Test_CacheBounded(unittest.TestCase):
    """Per-session cache must cap at MAX_CACHED_MESSAGES (audit §2, §5.5).

    Growing past the cap should not leak: the in-memory list is trimmed to
    the newest N entries. Byte-offset invariants are preserved regardless.
    """

    def test_cache_list_is_bounded(self):
        from cozempic.session import (
            MAX_CACHED_MESSAGES,  # RED: constant not exported on main
            _INCR_CACHE,            # RED: cache dict not present on main
            load_messages_incremental,
        )

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "big.jsonl"
            total = MAX_CACHED_MESSAGES + 1000
            _write_jsonl(jsonl, n_lines=total, payload_bytes=200)

            load_messages_incremental(jsonl)

            entry = _INCR_CACHE[jsonl.resolve()]
            self.assertLessEqual(
                len(entry.messages), MAX_CACHED_MESSAGES,
                f"cache held {len(entry.messages)} messages (cap {MAX_CACHED_MESSAGES})",
            )


# ─── partial-line tail safety + concurrency ────────────────────────


class Test_ConcurrentLoad(unittest.TestCase):
    """Parallel incremental readers must not race or double-parse.

    Audit §6 risk row: module-global lock serializes cache access.
    Worst-case behavior is serialized latency — never corruption, never
    duplicate messages.
    """

    def test_four_threads_get_consistent_view(self):
        from cozempic.session import load_messages_incremental  # RED: not defined on main

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "s.jsonl"
            _write_jsonl(jsonl, n_lines=500)

            results: list[list] = []
            errors: list[BaseException] = []
            barrier = threading.Barrier(4)

            def worker():
                try:
                    barrier.wait(timeout=5)
                    results.append(load_messages_incremental(jsonl))
                except BaseException as e:  # noqa: BLE001 — test surface
                    errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            self.assertEqual(errors, [], f"thread errors: {errors!r}")
            self.assertEqual(len(results), 4)
            # All threads see the same 500 messages — no dup, no corruption.
            for r in results:
                self.assertEqual(len(r), 500)
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0], results[2])
            self.assertEqual(results[0], results[3])


class Test_PartialLineTailSafe(unittest.TestCase):
    """Partial (mid-write) trailing line must be ignored until complete.

    Audit contract 4 (§5.4). If Claude is mid-write, the file ends without a
    newline. The cache must stop at the last `\\n` and pick up the remainder
    next cycle — no partial-JSON parsing, no offset drift.
    """

    def test_partial_trailing_line_is_skipped_then_picked_up(self):
        from cozempic.session import load_messages_incremental  # RED: not defined on main

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "s.jsonl"
            _write_jsonl(jsonl, n_lines=5)
            # Append a half-written line (no trailing newline).
            with open(jsonl, "a", encoding="utf-8") as f:
                f.write('{"role":"user","content":"partial')

            first = load_messages_incremental(jsonl)
            self.assertEqual(len(first), 5, "partial trailing line must be excluded")

            # Claude finishes the write: complete the line.
            with open(jsonl, "a", encoding="utf-8") as f:
                f.write('"}' + "\n")

            second = load_messages_incremental(jsonl)
            self.assertEqual(len(second), 6)
            self.assertEqual(second[-1][1]["content"], "partial")


# ─── ps drift + mtime corroboration ─────────────────────────────────


@pytest.mark.skipif(sys.platform == "win32", reason="mtime fallback is POSIX-only")
class Test_ClaudeProcessMtimeFallback(unittest.TestCase):
    """_is_claude_process must corroborate with JSONL mtime when ps drifts.

    Audit §4. Live PID 58060 proved ps-based detection can return False for
    a genuinely-alive Claude forking a subshell. A fresh JSONL mtime
    (< 60 s) overrides ps=False; an aged mtime leaves ps=False untouched.
    """

    def test_fresh_mtime_overrides_ps_false(self):
        from cozempic.guard import _is_claude_process  # RED: signature lacks session_path on main

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "session.jsonl"
            jsonl.write_text('{"role":"user","content":"recent"}\n')
            # mtime is already fresh (just-written).

            fake_ps = subprocess.CompletedProcess(
                args=["ps"], returncode=0, stdout="/usr/bin/zsh -l\n", stderr="",
            )
            with patch("cozempic.guard.subprocess.run", return_value=fake_ps):
                result = _is_claude_process(12345, session_path=jsonl)

            self.assertTrue(
                result,
                "fresh JSONL mtime must corroborate liveness when ps args don't match claude",
            )

    def test_aged_mtime_does_not_corroborate(self):
        from cozempic.guard import _is_claude_process  # RED: signature lacks session_path on main

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "session.jsonl"
            jsonl.write_text('{"role":"user","content":"old"}\n')
            # Age the file to 5 minutes old — well beyond the 60 s fallback window.
            aged = time.time() - 300
            os.utime(jsonl, (aged, aged))

            fake_ps = subprocess.CompletedProcess(
                args=["ps"], returncode=0, stdout="/usr/bin/zsh -l\n", stderr="",
            )
            with patch("cozempic.guard.subprocess.run", return_value=fake_ps):
                result = _is_claude_process(12345, session_path=jsonl)

            self.assertFalse(
                result,
                "aged JSONL mtime must NOT corroborate — Claude is dead",
            )


# ─── same-size in-place rewrite must invalidate cache ───────────


class Test_SameSizeRewriteInvalidation(unittest.TestCase):
    """Adversarial review surfaced: an in-place rewrite via `open('r+')` preserves inode
    and size while advancing mtime. The original inode/size/mtime_ns predicate
    missed this case and returned stale cached content, violating the audit §3
    equivalence contract.

    Fix contract: mtime advancing with size unchanged is itself a cache-miss
    signal — full re-read required.
    """

    def test_in_place_same_size_rewrite_returns_fresh_content(self):
        from cozempic.session import (
            _INCR_CACHE,
            load_messages,
            load_messages_incremental,
        )

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _INCR_CACHE.clear()
            p = Path(tmp) / "s.jsonl"
            p.write_text('{"a":1}\n{"a":2}\n{"a":3}\n')
            r1 = load_messages_incremental(p)
            self.assertEqual(len(r1), 3)

            # Ensure the subsequent write lands with an advancing mtime.
            time.sleep(0.01)
            # r+ preserves inode; exactly matching byte count keeps size identical.
            with open(p, "r+", encoding="utf-8") as f:
                f.write('{"z":9}\n{"z":8}\n{"z":7}\n')

            r2 = load_messages_incremental(p)
            self.assertEqual(
                r2, load_messages(p),
                "cache must invalidate on mtime advance with unchanged size",
            )


# ─── _INCR_CACHE LRU bound across sessions ──────────────────────


class Test_CacheLRUAcrossSessions(unittest.TestCase):
    """Adversarial review surfaced: _INCR_CACHE accreted one entry per distinct Path,
    unbounded. Single-session guard is fine, but library-API consumers
    (find_sessions() → loop) leaked.

    Fix contract: cap at MAX_CACHE_SESSIONS with LRU eviction.
    """

    def test_cache_bounded_by_max_cache_sessions(self):
        from cozempic.session import (
            MAX_CACHE_SESSIONS,
            _INCR_CACHE,
            load_messages_incremental,
        )

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _INCR_CACHE.clear()
            for i in range(MAX_CACHE_SESSIONS + 10):
                p = Path(tmp) / f"s{i}.jsonl"
                _write_jsonl(p, n_lines=3, payload_bytes=100)
                load_messages_incremental(p)

            self.assertLessEqual(
                len(_INCR_CACHE), MAX_CACHE_SESSIONS,
                f"cache held {len(_INCR_CACHE)} entries "
                f"(cap {MAX_CACHE_SESSIONS})",
            )

    def test_cache_evicts_least_recently_used(self):
        from cozempic.session import (
            MAX_CACHE_SESSIONS,
            _INCR_CACHE,
            load_messages_incremental,
        )

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _INCR_CACHE.clear()
            paths = []
            for i in range(MAX_CACHE_SESSIONS):
                p = Path(tmp) / f"s{i}.jsonl"
                _write_jsonl(p, n_lines=3, payload_bytes=100)
                load_messages_incremental(p)
                paths.append(p.resolve())

            # Touch paths[0] so it becomes most-recently-used.
            load_messages_incremental(paths[0])

            # Insert one more — expect eviction of paths[1] (oldest), NOT paths[0].
            extra = Path(tmp) / "extra.jsonl"
            _write_jsonl(extra, n_lines=3, payload_bytes=100)
            load_messages_incremental(extra)

            self.assertIn(paths[0], _INCR_CACHE, "most-recently-used path was evicted")
            self.assertNotIn(paths[1], _INCR_CACHE, "least-recently-used path was NOT evicted")


# ─── _terminate_and_resume accepts session_path ──────────────────


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only fallback path")
class Test_TerminateAndResumeAcceptsSessionPath(unittest.TestCase):
    """Adversarial review surfaced: watchdog passed session_path to _is_claude_process
    (the mtime fallback), but _terminate_and_resume's 8 internal calls did not. Result:
    watchdog said alive, reload path said dead, reload silently skipped.

    Fix contract: _terminate_and_resume accepts session_path and plumbs it
    into every _is_claude_process call inside it.
    """

    def test_terminate_and_resume_signature_accepts_session_path(self):
        import inspect
        from cozempic.guard import _terminate_and_resume
        sig = inspect.signature(_terminate_and_resume)
        self.assertIn(
            "session_path", sig.parameters,
            "_terminate_and_resume must accept session_path to corroborate "
            "liveness when ps drifts",
        )

    def test_terminate_and_resume_plumbs_session_path_to_identity_check(self):
        """Static source contract: every _is_claude_process call inside
        _terminate_claude MUST carry session_path.

        PR #102 P1: _terminate_and_resume is now a thin composition wrapper
        (_terminate_claude + _resume_claude). The identity checks live in
        _terminate_claude; _terminate_and_resume passes session_path through.
        Test updated to inspect _terminate_claude (where the checks now live).
        """
        import inspect
        import re as _re
        from cozempic.guard import _terminate_claude
        src = inspect.getsource(_terminate_claude)
        # Find every _is_claude_process( ... ) invocation and confirm each
        # references session_path. Matches across linebreaks.
        calls = _re.findall(
            r"_is_claude_process\(([^()]*(?:\([^()]*\)[^()]*)*)\)",
            src, flags=_re.DOTALL,
        )
        self.assertGreater(len(calls), 0, "no identity check found in _terminate_claude")
        missing = [c.strip() for c in calls if "session_path" not in c]
        self.assertEqual(
            missing, [],
            "these _is_claude_process calls in _terminate_claude lack "
            f"session_path: {missing!r}",
        )
