"""Shared test fixtures for the session-pruner regression suite.

Provides `make_realistic_session(target_size_mb, with_compact_boundary)` —
a deterministic generator that produces a ~1MB JSONL message stream mimicking
the shape of the corrupted `fannyugc` session from the bug handoff.

The generator is exposed both as a pytest fixture and as a standalone
callable so e2e tests can build per-test fixtures inline.
"""

from __future__ import annotations

import json
import random
import string
from pathlib import Path
from typing import Callable

import pytest

from cozempic.helpers import msg_bytes


def _deterministic_text(rng: random.Random, n_chars: int) -> str:
    """Produce a deterministic blob of n_chars text-safe characters."""
    alphabet = string.ascii_letters + string.digits + "    \n"
    return "".join(rng.choice(alphabet) for _ in range(n_chars))


def make_realistic_session(
    target_size_mb: float = 1.0,
    with_compact_boundary: bool = True,
    seed: int = 1234,
) -> list[dict]:
    """Generate a deterministic synthetic session.

    Layout — matches the bug handoff's per-type damage table:
      - 1 root user (parentUuid=null)
      - ~600 user entries (chained)
      - ~1000 assistant entries
      - ~8000 attachment entries (small, inflate count)
      - ~300 ai-title entries
      - ~300 last-prompt entries
      - ~300 permission-mode entries
      - ~200 system entries (no subtype)
      - 1 system entry with subtype=compact_boundary at ~95% through file
      - ~50 file-history-snapshot entries

    Total ~10 000 entries; target ~1MB on-disk after JSONL serialization.

    Per-entry sizes are tuned so the JSONL roughly hits target_size_mb. The
    output is deterministic given `seed` — used so the regression test
    asserts on stable counts.

    Returns a list of message dicts (NOT (idx, dict, size) tuples). Caller is
    responsible for converting via `enumerate` or writing to JSONL.
    """
    rng = random.Random(seed)

    counts = {
        "user": 600,
        "assistant": 1000,
        "attachment": 8000,
        "ai-title": 300,
        "last-prompt": 300,
        "permission-mode": 300,
        "system": 200,
        "file-history-snapshot": 50,
    }

    # Per-type body padding (bytes-ish) calibrated for ~1MB total.
    # target_size_mb=1 → ~100 bytes/entry average across 10k entries.
    # Heavy types (assistant) get more.
    pad = {
        "user": 40,
        "assistant": 80,
        "attachment": 30,
        "ai-title": 20,
        "last-prompt": 30,
        "permission-mode": 25,
        "system": 30,
        "file-history-snapshot": 60,
    }
    # Scale padding by requested size ratio.
    scale = target_size_mb / 1.0
    pad = {k: max(10, int(v * scale)) for k, v in pad.items()}

    messages: list[dict] = []
    prev_uuid: str | None = None

    def _uuid_for(t: str, i: int) -> str:
        return f"{t}_{i:05d}"

    # Root user always first.
    root_uuid = "root_user"
    messages.append({
        "type": "user",
        "uuid": root_uuid,
        "parentUuid": None,
        "message": {
            "role": "user",
            "content": _deterministic_text(rng, pad["user"]),
        },
    })
    prev_uuid = root_uuid

    # Build a deterministic interleaving by counts. We round-robin over types
    # using their target counts as a weighting; the final layout matches the
    # bug handoff's per-type proportions and places the compact_boundary at
    # ~95% through the file.
    total_entries = sum(counts.values())
    remaining = dict(counts)
    remaining["user"] -= 1  # account for root_user above
    order = sorted(remaining.keys())

    # Walk: at each step, pick the type that is the most "behind" relative to
    # its share. This produces a deterministic stable interleave.
    cumulative = 0
    boundary_pos = int(total_entries * 0.95) if with_compact_boundary else -1
    next_idx: dict[str, int] = {k: 0 for k in counts}

    for step in range(total_entries):
        if step == boundary_pos and with_compact_boundary:
            messages.append({
                "type": "system",
                "subtype": "compact_boundary",
                "uuid": "compact_boundary_marker",
                "parentUuid": prev_uuid,
                "hasPreservedSegment": False,
            })
            prev_uuid = "compact_boundary_marker"
            cumulative += 1
            continue

        # Pick the type with the highest (target / remaining_total) ratio
        # among non-exhausted types.
        candidates = [t for t in order if remaining[t] > 0]
        if not candidates:
            break
        # Sort by deficit so the order is stable & deterministic.
        candidates.sort(key=lambda t: -remaining[t] / counts[t])
        t = candidates[0]

        idx = next_idx[t]
        next_idx[t] += 1
        remaining[t] -= 1

        uid = _uuid_for(t, idx)

        if t == "user":
            payload = {
                "type": "user",
                "uuid": uid,
                "parentUuid": prev_uuid,
                "message": {
                    "role": "user",
                    "content": _deterministic_text(rng, pad["user"]),
                },
            }
            prev_uuid = uid
        elif t == "assistant":
            payload = {
                "type": "assistant",
                "uuid": uid,
                "parentUuid": prev_uuid,
                "message": {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": _deterministic_text(rng, pad["assistant"]),
                    }],
                },
            }
            prev_uuid = uid
        elif t == "attachment":
            payload = {
                "type": "attachment",
                "uuid": uid,
                "parentUuid": prev_uuid,
                "filename": f"f{idx}.txt",
                "size": pad["attachment"],
            }
            # Attachments do not advance the conversational chain
        elif t == "ai-title":
            payload = {
                "type": "ai-title",
                "uuid": uid,
                "parentUuid": prev_uuid,
                "title": _deterministic_text(rng, pad["ai-title"]),
            }
        elif t == "last-prompt":
            payload = {
                "type": "last-prompt",
                "uuid": uid,
                "parentUuid": prev_uuid,
                "text": _deterministic_text(rng, pad["last-prompt"]),
            }
        elif t == "permission-mode":
            payload = {
                "type": "permission-mode",
                "uuid": uid,
                "parentUuid": prev_uuid,
                "mode": "default",
                "_pad": _deterministic_text(rng, pad["permission-mode"]),
            }
        elif t == "system":
            payload = {
                "type": "system",
                "uuid": uid,
                "parentUuid": prev_uuid,
                "subtype": "info",
                "text": _deterministic_text(rng, pad["system"]),
            }
        elif t == "file-history-snapshot":
            payload = {
                "type": "file-history-snapshot",
                "uuid": uid,
                "parentUuid": prev_uuid,
                "messageId": f"mid_{idx % 25}",
                "isSnapshotUpdate": (idx % 3 == 0),
                "_pad": _deterministic_text(rng, pad["file-history-snapshot"]),
            }
        else:
            # Should never happen given our counts dict.
            continue

        messages.append(payload)
        cumulative += 1

    return messages


def write_jsonl(messages: list[dict], path: Path) -> None:
    """Write a list of message dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m, separators=(",", ":")) + "\n")


@pytest.fixture
def realistic_session_factory() -> Callable[..., list[dict]]:
    """Factory fixture: callable that builds a fresh session per test."""
    return make_realistic_session
