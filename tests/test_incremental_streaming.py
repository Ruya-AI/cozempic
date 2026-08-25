"""GREEN tests for the bounded streaming read in load_messages_incremental.

ISSUE-AGT-637: the guard's 30s read-only checkpoint path read the WHOLE unread
byte range into one ``raw_bytes`` allocation. On a 100-180 MiB transcript that
meant a ~700 MiB transient per poll and hundreds of MiB to ~1 GiB RSS. The fix
reads at most ``_MAX_INCREMENTAL_READ_BYTES`` per call, parses complete lines,
defers the unterminated tail, and resumes from the byte offset on the next poll.

These tests are deterministic (offset/LRU spies, no wall-clock RSS asserts) and
use sparse/synthetic transcripts — no giant committed fixture:
  * a >200 MiB SPARSE file proves the first read is budget-bounded
  * a >10 MB single line must be skipped whole, not buffered unboundedly
  * a huge PARTIAL trailing line must not defeat the bound (skip mode)
  * convergence must reproduce load_messages() exactly
  * partial-line deferral, surrogateescape, LRU and rewrite invalidation hold
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from cozempic.session import (
    MAX_CACHE_SESSIONS,
    MAX_LINE_BYTES,
    _INCR_CACHE,
    _MAX_INCREMENTAL_READ_BYTES,
    load_messages,
    load_messages_incremental,
)


def _write_jsonl(
    path: Path, n_lines: int, payload_bytes: int = 400, start: int = 0
) -> None:
    filler = "x" * max(payload_bytes - 64, 1)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(start, start + n_lines):
            f.write(json.dumps({"role": "user", "content": f"{i}:{filler}"}) + "\n")


def _append_jsonl(
    path: Path, n_lines: int, start_index: int, payload_bytes: int = 400
) -> None:
    filler = "x" * max(payload_bytes - 64, 1)
    with open(path, "a", encoding="utf-8") as f:
        for i in range(start_index, start_index + n_lines):
            f.write(json.dumps({"role": "user", "content": f"{i}:{filler}"}) + "\n")


def _converge(p: Path) -> list:
    """Call load_messages_incremental until the cache has consumed the file."""
    load_messages_incremental(p)  # seed the cache entry
    guard = 0
    entry = _INCR_CACHE[p.resolve()]
    while entry.offset < os.path.getsize(p) and guard < 1000:
        load_messages_incremental(p)
        guard += 1
    return load_messages_incremental(p)


class TestBoundedFirstRead(unittest.TestCase):
    """A cache miss on a >200 MiB transcript must not allocate the whole file."""

    def test_first_read_is_bounded_on_large_sparse_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "big.jsonl"
            with open(p, "wb") as f:
                f.write(b'{"role":"user","content":"first"}\n')
                f.seek(250 * 1024 * 1024)
                f.write(b'{"role":"user","content":"last"}\n')
            self.assertGreater(p.stat().st_size, 200 * 1024 * 1024)

            _INCR_CACHE.clear()
            r = load_messages_incremental(p)
            entry = _INCR_CACHE[p.resolve()]

            # Only the first line is parsed, and only a bounded prefix was read.
            self.assertEqual([m[1].get("content") for m in r], ["first"])
            self.assertLessEqual(
                entry.offset,
                _MAX_INCREMENTAL_READ_BYTES,
                f"first read consumed {entry.offset} bytes, exceeding the {_MAX_INCREMENTAL_READ_BYTES} budget",
            )
            # The unread remainder is still pending — convergence is deferred.
            self.assertLess(entry.offset, p.stat().st_size)

    def test_convergence_matches_full_read(self):
        """Repeated bounded calls must eventually reproduce load_messages()."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "med.jsonl"
            # ~12 MiB of lines — several budget-sized reads on a full miss
            # (well above the 8 MiB per-call budget).
            _write_jsonl(p, n_lines=25_000, payload_bytes=500)
            _INCR_CACHE.clear()
            first = load_messages_incremental(p)
            entry = _INCR_CACHE[p.resolve()]
            self.assertLess(
                entry.offset, p.stat().st_size
            )  # not fully consumed in one call

            result = _converge(p)
            # The incremental loader is capped at the newest MAX_CACHED_MESSAGES;
            # compare against the newest N of the full read.
            from cozempic.session import MAX_CACHED_MESSAGES

            self.assertEqual(
                result,
                load_messages(p)[-MAX_CACHED_MESSAGES:],
                "bounded convergence != newest-N of full read",
            )


class TestOversizedLines(unittest.TestCase):
    """A single oversized JSONL line must not defeat the byte bound."""

    def test_huge_complete_line_skipped_whole(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "huge.jsonl"
            with open(p, "w", encoding="utf-8") as f:
                f.write('{"role":"user","content":"before"}\n')
                f.write(
                    '{"role":"user","content":"'
                    + "y" * (MAX_LINE_BYTES + 1000)
                    + '"}\n'
                )
                f.write('{"role":"user","content":"after"}\n')
            _INCR_CACHE.clear()
            result = _converge(p)
            contents = [m[1].get("content") for m in result]
            self.assertEqual(contents, ["before", "after"])
            # The giant line still consumes a line index (0 before, 1 giant, 2 after).
            self.assertEqual([m[0] for m in result], [0, 2])
            # Nothing oversized is buffered.
            entry = _INCR_CACHE[p.resolve()]
            self.assertEqual(entry.pending, b"")
            self.assertFalse(entry.skipping)

    def test_huge_partial_trailing_line_bounded(self):
        """A giant line still being written must not accumulate in pending."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "partial_huge.jsonl"
            with open(p, "w", encoding="utf-8") as f:
                f.write('{"role":"user","content":"ok"}\n')
                f.write('{"role":"user","content":"' + "z" * 1000)
            _INCR_CACHE.clear()
            load_messages_incremental(p)
            # Grow the partial past the per-line limit.
            with open(p, "a", encoding="utf-8") as f:
                f.write("z" * (MAX_LINE_BYTES + 100))
            load_messages_incremental(p)
            entry = _INCR_CACHE[p.resolve()]
            # The oversized partial is NOT buffered wholesale — either capped or in skip mode.
            self.assertTrue(
                len(entry.pending) <= MAX_LINE_BYTES or entry.skipping,
                f"pending grew to {len(entry.pending)} bytes",
            )
            # Complete the line + append a small one; convergence yields both valid lines.
            with open(p, "a", encoding="utf-8") as f:
                f.write('"}\n')
                f.write('{"role":"user","content":"after"}\n')
            result = _converge(p)
            self.assertEqual([m[1].get("content") for m in result], ["ok", "after"])


class TestStreamingStatePreserved(unittest.TestCase):
    """Partial-line deferral, surrogateescape and LRU semantics must hold."""

    def test_partial_trailing_line_deferred_then_completed(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "partial.jsonl"
            _write_jsonl(p, n_lines=3)
            with open(p, "a", encoding="utf-8") as f:
                f.write('{"role":"user","content":"pa')
            r = load_messages_incremental(p)
            self.assertEqual(len(r), 3, "partial trailing line must be deferred")
            with open(p, "a", encoding="utf-8") as f:
                f.write('rtial"}\n')
            r = load_messages_incremental(p)
            self.assertEqual(len(r), 4)
            self.assertEqual(r[-1][1]["content"], "partial")

    def test_surrogateescape_bytes_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "surrogate.jsonl"
            # A structurally-invalid-UTF-8 line (raw 0xff byte) must decode via
            # surrogateescape and keep byte_len correct, matching load_messages.
            with open(p, "wb") as f:
                f.write(b'{"role":"user","content":"ok"}\n')
                f.write(b'{"role":"user","content":"bad \xff byte"}\n')
            _INCR_CACHE.clear()
            result = _converge(p)
            self.assertEqual(result, load_messages(p), "surrogateescape divergence")

    def test_rewrite_invalidation_still_full_reread(self):
        """os.replace (prune) still invalidates the streaming cache."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rw.jsonl"
            _write_jsonl(p, n_lines=20)
            _INCR_CACHE.clear()
            load_messages_incremental(p)
            replacement = Path(td) / "rw.new.jsonl"
            _write_jsonl(replacement, n_lines=7)
            os.replace(replacement, p)
            r = load_messages_incremental(p)
            self.assertEqual(len(r), 7, "inode change must trigger full re-read")
            self.assertEqual(r, load_messages(p))

    def test_lru_across_sessions_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            _INCR_CACHE.clear()
            for i in range(MAX_CACHE_SESSIONS + 5):
                p = Path(td) / f"s{i}.jsonl"
                _write_jsonl(p, n_lines=3, payload_bytes=100)
                load_messages_incremental(p)
            self.assertLessEqual(len(_INCR_CACHE), MAX_CACHE_SESSIONS)


if __name__ == "__main__":
    unittest.main()
