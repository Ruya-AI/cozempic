"""Round-6 mission-critical regression tests for PR #138.

Pins the R6 findings — a SECOND surrogate P0 I introduced in R5, and a broad
sweep of unhashable-field sibling-misses across the prune/guard/safety/executor
paths — to their production entry paths.

The dominant lesson: guard the whole CLASS, not named sites. The unhashable test
drives a single max-poison transcript (unhashable uuid / parentUuid / tool_use id /
name / compactedToolIds element + non-dict content blocks + unhashable tool_result
id) through EVERY prune/guard entry point and asserts none crash.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from cozempic.session import load_messages, load_messages_and_snapshot, save_messages
from cozempic.executor import run_prescription


class TestLoneSurrogateSaveNoCrashP0(unittest.TestCase):
    """R5's ensure_ascii=False crashed (UnicodeEncodeError) on a lone HIGH surrogate
    (U+D800-U+DBFF, which CC's JSON.stringify can emit for a sliced astral char) —
    on the guard path that kills Claude without resuming. _jsonl_line falls back to
    ensure_ascii=True for any surrogate the surrogateescape file can't encode."""

    def _save_ok(self, content_bytes_literal: bytes):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.jsonl"
            p.write_bytes(
                b'{"type":"user","uuid":"u0","message":{"role":"user","content":"keep ' + b"z" * 300 + b'"}}\n'
                + content_bytes_literal
            )
            msgs, snap = load_messages_and_snapshot(p)
            out, _ = run_prescription(msgs, ["standard"], {})
            save_messages(p, out, create_backup=False)  # must NOT raise
            reloaded = load_messages(p)  # must reload cleanly
            return reloaded

    def test_lone_high_surrogate_does_not_crash_save(self):
        # \ud83d (lone high surrogate) written as a JSON escape — json.loads accepts it.
        self._save_ok(b'{"type":"assistant","uuid":"a1","message":{"role":"assistant","content":"frag \\ud83d end"}}\n')

    def test_lone_low_surrogate_below_dc80_does_not_crash(self):
        self._save_ok(b'{"type":"assistant","uuid":"a2","message":{"role":"assistant","content":"x \\udc00 y"}}\n')

    def test_real_byte_still_byte_exact(self):
        # In-range surrogate from a real non-UTF-8 byte must stay byte-exact.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.jsonl"
            p.write_bytes(
                b'{"type":"user","uuid":"u0","message":{"role":"user","content":"keep ' + b"z" * 300 + b'"}}\n'
                b'{"type":"assistant","uuid":"a1","message":{"role":"assistant","content":"caf\xe9.txt"}}\n'
            )
            msgs, snap = load_messages_and_snapshot(p)
            out, _ = run_prescription(msgs, ["standard"], {})
            save_messages(p, out, create_backup=False)
            self.assertIn(b'caf\xe9.txt', p.read_bytes())


class TestUnhashableFieldSweep(unittest.TestCase):
    """A single max-poison transcript driven through EVERY prune/guard entry point —
    none may crash (the R6 sibling-miss class: an unhashable list/dict used as a set
    member / dict key raised TypeError -> guard respawn storm / aborted prune)."""

    POISON = [
        {"type": "user", "uuid": "u0", "parentUuid": None, "message": {"role": "user", "content": "hi " + "z" * 200}},
        # unhashable uuid + parentUuid -> loader wraps as _parse_error
        {"type": "assistant", "uuid": ["badid"], "parentUuid": ["bad"],
         "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "Task",
                     "input": {"subagent_type": "r", "prompt": "go"}}]}},
        # unhashable block id + name + non-dict content elements
        {"type": "assistant", "uuid": "a2", "parentUuid": "u0",
         "message": {"role": "assistant", "content": [{"type": "tool_use", "id": ["evil"], "name": ["bad"], "input": {}}, 42, "barestr"]}},
        # unhashable compactedToolIds element
        {"type": "assistant", "uuid": "a3", "parentUuid": "a2",
         "message": {"role": "assistant", "content": [{"type": "system", "subtype": "microcompact_boundary", "compactedToolIds": [["bad"], "ok"]}]}},
        # unhashable tool_result tool_use_id
        {"type": "user", "uuid": "u2", "parentUuid": "a3",
         "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": ["x"], "content": "r"}]}},
    ]

    def _write(self, d):
        p = Path(d) / "s.jsonl"
        p.write_text("\n".join(json.dumps(x) for x in self.POISON) + "\n")
        return p

    def test_loader_wraps_nonstr_uuid_line(self):
        with tempfile.TemporaryDirectory() as d:
            msgs = load_messages(self._write(d))
            # the unhashable-uuid line is opaque-wrapped, not a DAG-linked dict
            self.assertTrue(any(m.get("_parse_error") is True for _, m, _ in msgs))

    def test_no_path_crashes_on_max_poison(self):
        from cozempic.guard import prune_with_team_protect, detect_in_flight, safe_to_reload
        from cozempic.team import extract_team_state
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d)
            msgs = load_messages(p)
            with contextlib.redirect_stderr(io.StringIO()):  # per-strategy isolation logs to stderr
                for rx in ("gentle", "standard", "aggressive"):
                    run_prescription(load_messages(p), [rx], {})
                prune_with_team_protect(load_messages(p), "aggressive", {})
                detect_in_flight(msgs)
                safe_to_reload(extract_team_state(msgs), msgs, p)
            # reaching here = no crash on any path


if __name__ == "__main__":
    unittest.main()
