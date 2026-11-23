"""Streamed JSONL reads must equal the old whole-file read, byte for byte.

load_messages_and_snapshot and load_messages_incremental used to read the whole
region into one bytes object, decode it into one str and split it into a list
of lines: ~8.5x the file size at peak, held by the guard daemon for its
lifetime. They now stream line by line. These tests pin the streamed result to
the whole-file semantics (_split_physical_lines over the decoded text) on the
inputs where a line-by-line reader could plausibly diverge.
"""
import hashlib
import os

import pytest

from cozempic import session as S

CASES = {
    "lf": b'{"a":1}\n{"b":2}\n',
    "crlf": b'{"a":1}\r\n{"b":2}\r\n',
    "lone_cr": b'{"a":1}\r{"b":2}\n',
    "trailing_cr": b'{"a":1}\r',
    "u2028_raw": '{"t":"x y z"}\n{"u":"é\U0001F600"}\n'.encode(),
    "invalid_utf8": b'{"t":"\xff\xfe"}\n{"ok":1}\n',
    "unterminated_tail": b'{"a":1}\n{"b":2}',
    "blank_lines": b'\n\n{"a":1}\n\n',
    "not_json": b'not json\n{"a":1}\n',
    "empty": b"",
}


def _whole_file_reference(raw: bytes):
    out = []
    for i, line in enumerate(S._split_physical_lines(raw.decode("utf-8", "surrogateescape"))):
        parsed = S._parse_one_line(line, i)
        if parsed is not None:
            out.append(parsed)
    return out


@pytest.mark.parametrize("name", sorted(CASES))
def test_snapshot_load_matches_whole_file_read(tmp_path, name):
    raw = CASES[name]
    p = tmp_path / f"{name}.jsonl"
    p.write_bytes(raw)
    messages, snap = S.load_messages_and_snapshot(p)
    assert messages == _whole_file_reference(raw)
    assert snap.size == len(raw)
    assert snap.content_hash == hashlib.md5(raw).hexdigest()
    assert snap.inode == os.stat(p).st_ino
    assert snap.classify(p) == "unchanged"


@pytest.mark.parametrize("name", sorted(CASES))
def test_incremental_load_matches_whole_file_read(tmp_path, name):
    raw = CASES[name]
    p = tmp_path / f"{name}.jsonl"
    p.write_bytes(raw)
    S._INCR_CACHE.clear()
    # The incremental reader defers an unterminated final line.
    complete = raw[: raw.rfind(b"\n") + 1]
    assert S.load_messages_incremental(p) == _whole_file_reference(complete)
    assert S._INCR_CACHE[p.resolve()].offset == len(complete)


def test_incremental_append_across_partial_line(tmp_path):
    p = tmp_path / "grow.jsonl"
    p.write_bytes(b'{"a":1}\n{"b":')
    S._INCR_CACHE.clear()
    assert [m[1] for m in S.load_messages_incremental(p)] == [{"a": 1}]
    with open(p, "ab") as f:
        f.write(b'2}\r\n{"c":"\xc3\xa9"}\n')
    final = S.load_messages_incremental(p)
    assert final == _whole_file_reference(p.read_bytes())
    assert [m[1] for m in final] == [{"a": 1}, {"b": 2}, {"c": "é"}]


def test_incremental_full_read_keeps_newest_only(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "MAX_CACHED_MESSAGES", 3)
    p = tmp_path / "big.jsonl"
    p.write_bytes(b"".join(b'{"n":%d}\n' % i for i in range(20)))
    S._INCR_CACHE.clear()
    msgs = S.load_messages_incremental(p)
    assert [m[1]["n"] for m in msgs] == [17, 18, 19]
    assert [m[0] for m in msgs] == [17, 18, 19]  # line indices still absolute


@pytest.mark.parametrize("loader", ["snapshot", "incremental"])
def test_peak_memory_is_not_a_multiple_of_the_file(tmp_path, loader):
    """One non-Latin-1 character used to widen the WHOLE decoded file to 4 bytes
    per char, on top of the raw bytes and the split lines (7x the file here,
    8.5x on a real 81 MB transcript). Streamed, peak stays near what the parsed
    messages themselves hold."""
    import tracemalloc
    p = tmp_path / "wide.jsonl"
    body = "x" * 10_000
    with open(p, "wb") as f:
        f.write('{"t":"\U0001F600"}\n'.encode())
        for i in range(400):
            f.write(('{"n":%d,"t":"%s"}\n' % (i, body)).encode())
    size = p.stat().st_size
    S._INCR_CACHE.clear()
    tracemalloc.start()
    try:
        if loader == "snapshot":
            result = S.load_messages_and_snapshot(p)
        else:
            result = S.load_messages_incremental(p)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del result
    assert peak < 3 * size, f"peak {peak / size:.1f}x the file size"
