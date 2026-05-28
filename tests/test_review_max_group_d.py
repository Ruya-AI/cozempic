"""RED test for review-max Group D — thread messages_before_prune through
save_messages (D.12).

H-3's narrow scope used ``merged_before == merged_after`` so only C1
(dangling parentUuid) fired on the append delta. C2–C7 became trivially
true because the synthesized "before" was identical to the "after".

Fix: extend ``save_messages`` with a new keyword argument
``messages_before_prune: list[Message] | None = None``. When provided AND
``state == "appended"``, call
``validate_post_prune(messages_before_prune, list(messages) + delta)`` so
the pre-prune original is the "before" and C2–C7 fire materially.

Default ``None`` preserves backward compat for any third-party caller —
falls back to the existing weak C1-only check.

Scenario tested: the pruner dropped the LAST permission-mode entry
(violating C5 against the original); the post-prune validation INSIDE
run_prescription would have caught it, but the pruner is malformed in this
test and ships an invalid set. Then Claude appends an irrelevant entry.
With messages_before_prune fed, save_messages' post-append re-validation
catches the C5 violation and rejects the save.
"""

from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cozempic.helpers import msg_bytes


def _msg(idx: int, payload: dict) -> tuple[int, dict, int]:
    return (idx, payload, msg_bytes(payload))


def _write_jsonl(path: Path, messages: list[tuple[int, dict, int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for _, m, _ in messages:
            f.write(json.dumps(m, separators=(",", ":")) + "\n")


def _original_session() -> list[tuple[int, dict, int]]:
    return [
        _msg(0, {"type": "user", "uuid": "u0", "parentUuid": None,
                 "message": {"role": "user", "content": "hi"}}),
        _msg(1, {"type": "assistant", "uuid": "a1", "parentUuid": "u0",
                 "message": {"role": "assistant",
                             "content": [{"type": "text", "text": "hello"}]}}),
        _msg(2, {"type": "user", "uuid": "u1", "parentUuid": "a1",
                 "message": {"role": "user", "content": "next"}}),
        _msg(3, {"type": "assistant", "uuid": "a2", "parentUuid": "u1",
                 "message": {"role": "assistant",
                             "content": [{"type": "text", "text": "world"}]}}),
        _msg(4, {"type": "permission-mode", "uuid": "pm1", "parentUuid": "a2",
                 "mode": "default"}),
        _msg(5, {"type": "last-prompt", "uuid": "lp1", "parentUuid": "a2",
                 "text": "next"}),
    ]


class TestSaveMessagesAcceptsMessagesBeforePrune(unittest.TestCase):
    """D.12 — save_messages signature must accept messages_before_prune
    keyword and use it as the "before" in post-append validation."""

    def test_save_messages_signature_has_messages_before_prune(self):
        import inspect
        from cozempic.session import save_messages
        sig = inspect.signature(save_messages)
        self.assertIn(
            "messages_before_prune", sig.parameters,
            msg="save_messages must accept messages_before_prune kwarg (D.12)",
        )
        # Default must be None for backward compat.
        param = sig.parameters["messages_before_prune"]
        self.assertEqual(
            param.default, None,
            msg="messages_before_prune default must be None for back-compat",
        )

    def test_post_append_validation_uses_pre_prune_set_when_provided(self):
        """C5 (last permission-mode) fires only when messages_before_prune is
        threaded through. Without it, the merged_before==merged_after weak
        check misses the C5 violation."""
        from cozempic.session import (
            save_messages, snapshot_session,
        )
        from cozempic.safety import PruneValidationError

        with TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "session.jsonl"
            original = _original_session()
            _write_jsonl(session_path, original)
            snap = snapshot_session(session_path)

            # Claude appends an irrelevant assistant entry (valid parent
            # pointing to a2 which DOES survive — no C1 violation).
            extra = {
                "type": "assistant", "uuid": "a_late", "parentUuid": "a2",
                "message": {"role": "assistant",
                            "content": [{"type": "text", "text": "race"}]},
            }
            with open(session_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(extra, separators=(",", ":")) + "\n")

            # The pruner produced an output that drops the last
            # permission-mode (violates C5 against the ORIGINAL).
            pruned = [m for m in original if m[1].get("uuid") != "pm1"]

            # Without messages_before_prune the weak check passes (delta has
            # no perm-mode, merged_before==merged_after so C5 is vacuous).
            # With messages_before_prune the check uses the real original
            # → C5 fires.
            with self.assertRaises(PruneValidationError) as ctx:
                save_messages(
                    session_path, pruned,
                    create_backup=False, snapshot=snap,
                    messages_before_prune=original,
                )
            self.assertEqual(ctx.exception.evidence.get("failed_check"), "C5")


if __name__ == "__main__":
    unittest.main()
