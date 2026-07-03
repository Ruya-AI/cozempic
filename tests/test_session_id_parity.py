"""Bash↔Python slug parity regression test for C2 (DA round 1 finding).

C2 was: the bash hook sanitiser and Python ``_pid_file_for_session``
accepted different character sets for ``session_id``. Non-UUID inputs
would compute one path in bash and a different one (or ValueError) in
Python → silent guard disablement.

Round-3 fix per code-auditor's Option B sign-off: relax Python's
``_SESSION_ID_RE`` from ``^[0-9a-f][0-9a-f-]{11,}$`` to
``^[a-z0-9][a-z0-9_-]{11,}$``. This matches the char-class of the bash
sanitiser ``re.sub(r'[^a-z0-9_-]','_', s.lower())`` AND
``reload_lock._slug_for``/``spawn_lock._slug_for``.

This test pins the parity contract: for a fixed set of synthetic
session_ids representative of UUIDs, non-hex names, underscores, mixed
case, etc., the bash sanitiser's first-12-chars output MUST equal the
slug Python derives via ``_pid_file_for_session``. If either side
regresses to a stricter or laxer char class, this test fails.

Test set chosen to cover:
- canonical UUIDs (the production case)
- non-hex letters (`t`, `g`, `z` — would have been rejected pre-Option-B)
- underscores (allowed in both)
- mixed case (lowercased by both)
- short suffix (≥12 chars is the only length constraint after truncation)

Inputs that should be REJECTED by both (leading dash, too short, special
chars) are covered elsewhere — see
``test_guard_hardening.py::TestPolishV2_SessionIdRegexRequiresHexFirstChar``.
This file is solely about parity on ACCEPTED inputs.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _bash_slug(session_id: str) -> str:
    """Pure-Python mirror of the bash SessionStart hook's slug logic.

    Bash hook does:
        SESSION_ID=$(python3 -c "
            import sys, json, re
            s = json.load(sys.stdin).get('session_id','').lower()
            print(re.sub(r'[^a-z0-9_-]', '_', s))
        ")
        SLUG=$(printf '%.12s' "$SESSION_ID")      # POSIX-safe, #168
        GUARD_PID_FILE="/tmp/cozempic_guard_${SLUG}.pid"

    So the bash slug is: ``re.sub(r'[^a-z0-9_-]', '_', s.lower())[:12]``.
    """
    return re.sub(r"[^a-z0-9_-]", "_", session_id.lower())[:12]


class TestSessionIdSlugParity(unittest.TestCase):
    """C2 parity regression: bash slug == Python slug for every valid input."""

    # Synthetic test set per code-auditor's spec (with two extras for
    # underscore + mixed-case coverage).
    PARITY_INPUTS = [
        "5d53e013-32d9-4e72-9a3a-deadbeefcafe",     # canonical UUID
        "deadbeef-cafe-1234-5678-abcdef012345",     # another canonical
        "test-session-id-long-enough",              # non-hex letters
        "abc_123_def_456789",                       # underscore in body
        "Test-Session-Id-Mixed-Case-Long",          # mixed case → lower
        "z9z9z9z9z9z9z9z9z9z9",                     # non-hex 'z' repeats
        "0123456789ab",                             # exactly 12 chars
    ]

    def test_bash_python_slug_parity(self):
        from cozempic.guard import _pid_file_for_session

        failures = []
        for sid in self.PARITY_INPUTS:
            bash_slug = _bash_slug(sid)
            try:
                python_path = _pid_file_for_session(sid)
            except ValueError as exc:
                failures.append(
                    f"{sid!r} → bash slug={bash_slug!r}, "
                    f"Python REJECTED ({exc!s})"
                )
                continue
            python_slug = (
                python_path.name
                .removeprefix("cozempic_guard_")
                .removesuffix(".pid")
            )
            if bash_slug != python_slug:
                failures.append(
                    f"{sid!r} → bash={bash_slug!r} python={python_slug!r}"
                )

        if failures:
            self.fail(
                "Bash↔Python slug divergence (C2 regression).\n"
                "If you tightened Python's _SESSION_ID_RE or the bash "
                "sanitiser, both sides must change together.\n"
                "Failures:\n  " + "\n  ".join(failures)
            )

    def test_uuid_canonical_unchanged(self):
        """Canonical UUID inputs (the production case) must produce a slug
        that's just the first 12 chars of the lowercased UUID — no
        substitution should fire for any character."""
        from cozempic.guard import _pid_file_for_session

        for sid in (
            "5d53e013-32d9-4e72-9a3a-deadbeefcafe",
            "5D53E013-32D9-4E72-9A3A-DEADBEEFCAFE",  # uppercase variant
        ):
            python_path = _pid_file_for_session(sid)
            self.assertEqual(
                python_path.name,
                "cozempic_guard_5d53e013-32d.pid",
                f"Canonical UUID slug regressed for {sid!r}: got {python_path.name!r}",
            )


if __name__ == "__main__":
    unittest.main()
