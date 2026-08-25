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
    """A cache miss on a >200 MiB transcript must not allocate the whole file,
    and (ISSUE-AGT-637 follow-up) must not require multiple polls to see the
    tail — the first result must already reflect the newest retained bytes.
    """

    def test_first_read_is_bounded_and_shows_tail_on_large_sparse_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "big.jsonl"
            with open(p, "wb") as f:
                f.write(b'{"role":"user","content":"first"}\n')
                # A real newline INSIDE the tail window, well before "last", so
                # the tail-alignment scan has a genuine line boundary to latch
                # onto rather than only the sparse zero-gap (which — being
                # newline-free — would otherwise make "last"'s own terminator
                # the first newline found, and get trimmed away with it).
                # "spacer" and "last" are written back-to-back (no further
                # seek) so nothing but real JSONL sits between the alignment
                # point and EOF — a real transcript never has a NUL-byte gap
                # between two adjacent lines; that's purely a sparse-file test
                # artifact this avoids.
                f.seek(246 * 1024 * 1024)
                f.write(b'{"role":"user","content":"spacer"}\n')
                f.write(b'{"role":"user","content":"last"}\n')
            self.assertGreater(p.stat().st_size, 200 * 1024 * 1024)

            _INCR_CACHE.clear()
            r = load_messages_incremental(p)
            entry = _INCR_CACHE[p.resolve()]

            # The FIRST call already sees the tail — the old bug required many
            # 30s-cadence polls (minutes) to forward-scan from byte 0 to here.
            self.assertEqual([m[1].get("content") for m in r], ["last"])
            # Fully converged in one call (tail window reaches EOF by construction).
            self.assertEqual(entry.offset, p.stat().st_size)

    def test_convergence_matches_full_read(self):
        """A tail-anchored cache miss must reproduce load_messages()'s newest N."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "med.jsonl"
            # ~12 MiB of lines — above the 8 MiB per-call budget, so the tail
            # window covers only the newest slice of the file on a cache miss.
            _write_jsonl(p, n_lines=25_000, payload_bytes=500)
            _INCR_CACHE.clear()
            load_messages_incremental(p)
            entry = _INCR_CACHE[p.resolve()]
            # Tail-anchored miss: [start_offset, EOF) always fits one budget,
            # so a single call reaches EOF (no multi-poll forward scan needed).
            self.assertEqual(
                entry.offset,
                p.stat().st_size,
                "tail-anchored cache miss should reach EOF in one call",
            )

            result = _converge(p)
            # The incremental loader is capped at the newest MAX_CACHED_MESSAGES;
            # compare against the newest N of the full read. Line indices are
            # NOT compared: a tail-anchored miss starts counting from the jump
            # point (it never scans the skipped prefix), so indices are only
            # meaningful relative to each other within one incremental cache,
            # not equal to load_messages()'s true absolute file-line numbers.
            from cozempic.session import MAX_CACHED_MESSAGES

            full_tail = load_messages(p)[-MAX_CACHED_MESSAGES:]
            self.assertEqual(
                [(msg, size) for _idx, msg, size in result],
                [(msg, size) for _idx, msg, size in full_tail],
                "bounded convergence != newest-N of full read (by content)",
            )
            # Indices are still monotonically increasing within the cache.
            self.assertEqual(
                [idx for idx, _msg, _size in result],
                sorted(idx for idx, _msg, _size in result),
            )


class TestOversizedLines(unittest.TestCase):
    """A single oversized JSONL line must not defeat the byte bound."""

    def test_huge_complete_line_at_tail_boundary_realigns(self):
        """The oversized line (>budget) always straddles the tail-anchor window
        on a cache miss. The read must realign to the line's own terminating
        newline rather than parse a mid-line fragment as garbage — proving
        tail-anchoring stays newline-aligned even when it lands inside a giant
        line.
        """
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
            # "before" and the giant line predate the tail window and are not
            # retained after a cache miss — only "after" (post-realignment) is.
            self.assertEqual(contents, ["after"])
            # No parse-error entries from a mis-aligned mid-line read.
            self.assertFalse(any(m[1].get("_parse_error") for m in result))
            entry = _INCR_CACHE[p.resolve()]
            self.assertEqual(entry.pending, b"")
            self.assertFalse(entry.skipping)

    def test_huge_complete_line_via_append_still_skipped_whole(self):
        """A giant line arriving via a plain forward append (base file well
        under budget, so no tail-anchor is involved) must still be skipped
        whole via the existing skip-mode machinery."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "huge_append.jsonl"
            _write_jsonl(p, n_lines=1, payload_bytes=50)  # line 0: small, cached
            _INCR_CACHE.clear()
            load_messages_incremental(p)
            with open(p, "a", encoding="utf-8") as f:
                f.write(
                    '{"role":"user","content":"'
                    + "y" * (MAX_LINE_BYTES + 1000)
                    + '"}\n'
                )
                f.write('{"role":"user","content":"after"}\n')
            result = _converge(p)
            contents = [m[1].get("content") for m in result]
            self.assertEqual(contents[-1], "after")
            self.assertNotIn("y" * 10, str(contents))  # giant line never surfaces
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


class TestTailAnchoredCacheMiss(unittest.TestCase):
    """ISSUE-AGT-637 follow-up: checkpoint_team/start_guard derive agents_active
    from load_messages_incremental's FIRST result. On a large transcript, a
    cache miss (first call, inode replacement, truncation, same-size rewrite)
    must show tail evidence on that very first call, not several polls later.
    """

    def _line(self, i, payload_bytes=400):
        filler = "x" * max(payload_bytes - 64, 1)
        return json.dumps({"role": "user", "content": f"{i}:{filler}"})

    def _task_tool_use_line(self, tool_id="tail-task-1"):
        """A bare Task tool_use with no matching tool_result — extract_team_state
        marks this an active ("running") subagent (cozempic.team.extract_team_state)."""
        return json.dumps({
            "message": {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": tool_id,
                    "name": "Task",
                    "input": {
                        "description": "tail agent",
                        "subagent_type": "general-purpose",
                        "prompt": "do the tail work",
                    },
                }],
            }
        })

    def test_active_subagent_only_in_final_chunk_visible_on_first_call(self):
        """A Task spawn recorded only in the last (retained) chunk of a large
        transcript must be visible to extract_team_state on the FIRST
        load_messages_incremental call — the old forward-from-0 scan could
        take many 30s-cadence guard cycles (minutes) to reach it."""
        from cozempic.team import extract_team_state

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "session.jsonl"
            with open(p, "w", encoding="utf-8") as f:
                # Enough filler to push the file well past the read budget —
                # none of this is team-related.
                for i in range(30_000):
                    f.write(self._line(i, payload_bytes=400) + "\n")
                # The active-agent evidence lands only in the final chunk.
                f.write(self._task_tool_use_line() + "\n")
            self.assertGreater(p.stat().st_size, _MAX_INCREMENTAL_READ_BYTES)

            _INCR_CACHE.clear()
            first_call_messages = load_messages_incremental(p)
            state = extract_team_state(first_call_messages)

            self.assertFalse(
                state.is_empty(), "active subagent must be visible on the FIRST call"
            )
            self.assertTrue(
                any(s.status == "running" for s in state.subagents),
                "the tail Task spawn must be detected as an active subagent",
            )

    def test_guard_agents_active_true_on_first_checkpoint(self):
        """End-to-end: checkpoint_team()/_compute_agents_active() must report
        active on the FIRST checkpoint for a large transcript whose only team
        evidence is at the tail — this is the exact contract start_guard's
        reload gate relies on."""
        from cozempic.guard import _compute_agents_active, checkpoint_team

        with tempfile.TemporaryDirectory() as td:
            session_dir = Path(td)
            p = session_dir / "session.jsonl"
            with open(p, "w", encoding="utf-8") as f:
                for i in range(30_000):
                    f.write(self._line(i, payload_bytes=400) + "\n")
                f.write(self._task_tool_use_line() + "\n")
            self.assertGreater(p.stat().st_size, _MAX_INCREMENTAL_READ_BYTES)

            _INCR_CACHE.clear()
            state = checkpoint_team(session_path=p, quiet=True)
            self.assertTrue(
                _compute_agents_active(state),
                "agents_active must be True on the first checkpoint, not several polls later",
            )

    def test_first_read_is_byte_bounded_and_newline_aligned(self):
        """The tail-anchor window must never straddle into a mid-line offset:
        every parsed message on the first call is a real, complete line."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "aligned.jsonl"
            # Fixed-width lines so the tail window boundary is guaranteed to
            # land mid-line for at least one line (line width does not evenly
            # divide the budget).
            line_width = len(self._line(0, payload_bytes=777)) + 1  # +1 for \n
            n_lines = (_MAX_INCREMENTAL_READ_BYTES // line_width) * 2 + 3
            with open(p, "w", encoding="utf-8") as f:
                for i in range(n_lines):
                    f.write(self._line(i, payload_bytes=777) + "\n")
            self.assertGreater(p.stat().st_size, _MAX_INCREMENTAL_READ_BYTES)

            _INCR_CACHE.clear()
            result = load_messages_incremental(p)
            entry = _INCR_CACHE[p.resolve()]

            # Bounded: the read never needed a second call to reach EOF.
            self.assertEqual(entry.offset, p.stat().st_size)
            # Aligned: no parse-error entries from a mid-line fragment, and
            # every returned message's content round-trips cleanly.
            self.assertTrue(result, "first call must already return messages")
            for _idx, msg, _size in result:
                self.assertNotIn("_parse_error", msg)
                self.assertRegex(msg["content"], r"^\d+:x+$")
            # The last retained line is the true last line of the file.
            self.assertEqual(result[-1][1]["content"].split(":")[0], str(n_lines - 1))

    def test_truncation_and_rewrite_also_tail_anchor(self):
        """os.replace / truncation / same-size in-place rewrite all route
        through the same needs_full_read branch as a fresh cache miss — a
        large replacement transcript must also show tail evidence immediately,
        not after a forward re-scan."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rewritten.jsonl"
            _write_jsonl(p, n_lines=5)
            _INCR_CACHE.clear()
            load_messages_incremental(p)  # warm a small entry

            # Replace with a large file whose only distinguishing tail content
            # is a final unique marker line.
            replacement = Path(td) / "rewritten.new.jsonl"
            with open(replacement, "w", encoding="utf-8") as f:
                for i in range(30_000):
                    f.write(self._line(i, payload_bytes=400) + "\n")
                f.write(json.dumps({"role": "user", "content": "TAIL_MARKER"}) + "\n")
            os.replace(replacement, p)
            self.assertGreater(p.stat().st_size, _MAX_INCREMENTAL_READ_BYTES)

            r = load_messages_incremental(p)
            self.assertEqual(
                r[-1][1]["content"],
                "TAIL_MARKER",
                "inode-replacement full-read must also tail-anchor, not forward-scan",
            )


if __name__ == "__main__":
    unittest.main()
