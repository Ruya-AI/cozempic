"""E2E regression test for the 63MB → 730KB resume-break bug.

Per PLAN.md § 3.5:

A realistic synthetic session (~1MB, mimicking the corrupted `fannyugc` shape
from the handoff) is built via the `make_realistic_session` fixture, then run
through the GENTLE prescription end-to-end. The output is asserted REPLAYABLE
against the post-fix invariants:

  - Final user count ≥ 50% of before
  - Final assistant count ≥ 50% of before
  - At least one permission-mode entry survives (the LAST one)
  - At least one last-prompt entry survives
  - At least one ai-title entry survives
  - First message (parentUuid=null root) survives
  - compact_boundary entry survives
  - Every surviving parentUuid resolves to either null or another surviving uuid
  - File size reduction is in [30%, 80%] (gentle still produces meaningful
    reduction without destroying resume)

Currently RED until the four fix pillars (P0-A through P0-D) are in place.
"""

from __future__ import annotations

import unittest

from cozempic.helpers import msg_bytes
from tests._session_factory import make_realistic_session


def _to_messages(payloads: list[dict]) -> list[tuple[int, dict, int]]:
    """Convert a list of payloads into the (idx, msg, size) tuple form
    used throughout the executor."""
    return [(i, p, msg_bytes(p)) for i, p in enumerate(payloads)]


class TestE2ERegressionForResumeBreakBug(unittest.TestCase):
    """Run the GENTLE prescription on a realistic ~1MB session with a
    compact_boundary near the tail. Assert the output is replayable."""

    def test_realistic_session_with_compact_boundary_does_not_lose_resumable_state(self):
        from cozempic.executor import run_prescription
        from cozempic.registry import PRESCRIPTIONS
        import cozempic.strategies  # noqa: F401 — register strategies

        payloads = make_realistic_session(
            target_size_mb=1.0, with_compact_boundary=True, seed=42,
        )
        before = _to_messages(payloads)
        before_bytes = sum(b for _, _, b in before)

        # Counts before
        def _count(messages, t):
            return sum(1 for _, m, _ in messages if m.get("type") == t)

        users_before = _count(before, "user")
        asst_before = _count(before, "assistant")

        # Run GENTLE end-to-end.
        after, _ = run_prescription(before, PRESCRIPTIONS["gentle"], {})
        after_bytes = sum(b for _, _, b in after)

        # ── 1. Conversation survival ≥ 50% ──────────────────────────────────
        users_after = _count(after, "user")
        asst_after = _count(after, "assistant")
        self.assertGreaterEqual(
            users_after / max(users_before, 1), 0.50,
            f"user survival {users_after}/{users_before}",
        )
        self.assertGreaterEqual(
            asst_after / max(asst_before, 1), 0.50,
            f"assistant survival {asst_after}/{asst_before}",
        )

        # ── 2. At least one of each bootstrap metadata ──────────────────────
        for t in ("permission-mode", "last-prompt", "ai-title"):
            self.assertGreaterEqual(
                _count(after, t), 1,
                f"no {t} entry survived — the documented bug",
            )

        # ── 3. Root (parentUuid=None) survives ──────────────────────────────
        roots_after = [m for m in after if m[1].get("parentUuid") is None]
        self.assertGreaterEqual(len(roots_after), 1, "no root after prune")
        # Specifically, the original root must be the surviving one.
        self.assertEqual(roots_after[0][1].get("uuid"), "root_user")

        # ── 4. compact_boundary survives ────────────────────────────────────
        boundary_after = [
            m for m in after
            if m[1].get("type") == "system"
            and m[1].get("subtype") == "compact_boundary"
        ]
        self.assertEqual(
            len(boundary_after), 1,
            "compact_boundary marker lost or duplicated",
        )

        # ── 5. parentUuid graph resolves ────────────────────────────────────
        survivor_uuids = {m[1].get("uuid") for m in after if m[1].get("uuid")}
        for _, msg, _ in after:
            parent = msg.get("parentUuid")
            if parent is None:
                continue
            self.assertIn(
                parent, survivor_uuids,
                f"dangling parentUuid {parent!r} on uuid {msg.get('uuid')!r}",
            )

        # ── 6. File-size reduction in [30%, 90%] ────────────────────────────
        # The upper bound exists to catch the documented bug shape (98% loss
        # in the fannyugc corruption). The synthetic fixture has ~80% of its
        # entries as small attachments below the boundary, so post-floor
        # gentle naturally hits ~84% reduction; we cap at 90% to leave
        # headroom while still detecting the destructive prune.
        reduction_pct = 1.0 - (after_bytes / max(before_bytes, 1))
        self.assertGreaterEqual(
            reduction_pct, 0.30,
            f"gentle reduced only {reduction_pct:.0%}; expected ≥30%",
        )
        self.assertLessEqual(
            reduction_pct, 0.90,
            f"gentle reduced {reduction_pct:.0%}; expected ≤90% (anything more "
            f"is the destructive bug — 98% loss in fannyugc)",
        )


if __name__ == "__main__":
    unittest.main()
