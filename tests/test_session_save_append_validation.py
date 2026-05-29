"""RED test for review finding H-3 — post-append validation re-run.

When ``save_messages`` enters its ``state == "appended"`` branch, the new
bytes that Claude wrote to the live file between the snapshot moment and
the impending ``os.replace`` are appended to the tmp file VERBATIM. Those
appended lines were never seen by ``validate_post_prune`` (which ran
inside ``run_prescription`` before save). A malformed entry written by
Claude in that window — e.g., one whose ``parentUuid`` points to a uuid
the pruner just removed — would ship in the saved JSONL without ever
being structurally checked.

The fix: after the append-delta merge in ``save_messages``, re-parse the
tmp file's tail and re-run ``validate_post_prune`` on the merged
``(pruned_messages + delta_messages)`` set. On failure: unlink the tmp
file, leave the live file + backup untouched, propagate the exception.

This test SHOULD fail RED on the pre-fix code (save_messages succeeds
silently with the orphan in place) and GREEN after the post-append
validation is added.
"""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cozempic.helpers import msg_bytes
from cozempic.session import (
    _FileSnapshot,
    load_messages,
    save_messages,
    snapshot_session,
)


def _msg(idx: int, payload: dict) -> tuple[int, dict, int]:
    return (idx, payload, msg_bytes(payload))


def _valid_session() -> list[tuple[int, dict, int]]:
    return [
        _msg(0, {"type": "user", "uuid": "u0", "parentUuid": None,
                 "message": {"role": "user", "content": "hi"}}),
        _msg(1, {"type": "assistant", "uuid": "a1", "parentUuid": "u0",
                 "message": {"role": "assistant",
                             "content": [{"type": "text", "text": "hello"}]}}),
        _msg(2, {"type": "user", "uuid": "u1", "parentUuid": "a1",
                 "message": {"role": "user", "content": "follow"}}),
        _msg(3, {"type": "assistant", "uuid": "a2", "parentUuid": "u1",
                 "message": {"role": "assistant",
                             "content": [{"type": "text", "text": "world"}]}}),
        _msg(4, {"type": "permission-mode", "uuid": "pm1", "parentUuid": "a2",
                 "mode": "default"}),
        _msg(5, {"type": "last-prompt", "uuid": "lp1", "parentUuid": "a2",
                 "text": "follow"}),
    ]


class TestSaveMessagesRevalidatesAppendDelta(unittest.TestCase):
    """H-3 — save_messages must re-validate the merged file after appending
    Claude's mid-prune delta lines.
    """

    def test_save_messages_rejects_orphaned_parent_in_append_delta(self):
        from cozempic.safety import PruneValidationError

        with TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "session.jsonl"
            # 1. Write the original session.
            messages = _valid_session()
            with open(session_path, "w", encoding="utf-8") as f:
                for _, m, _ in messages:
                    f.write(json.dumps(m, separators=(",", ":")) + "\n")

            # 2. Snapshot at the same moment as a pruner would.
            snap = snapshot_session(session_path)

            # 3. Simulate Claude appending a structurally-invalid entry mid-
            #    prune: parentUuid points to a uuid we are about to remove.
            #    (We pretend the strategy will drop u1 below.)
            orphan_entry = {
                "type": "assistant", "uuid": "a_late", "parentUuid": "u1",
                "message": {"role": "assistant",
                            "content": [{"type": "text", "text": "race"}]},
            }
            with open(session_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(orphan_entry, separators=(",", ":")) + "\n")

            # 4. Pruner removes u1 from its in-memory view; calls save_messages.
            #    Because we are exercising save_messages directly, validate_
            #    post_prune already ran on the pruned set inside run_prescription
            #    (we skip that for this isolated test). The pruned set is:
            pruned = [m for m in messages if m[1].get("uuid") != "u1"]

            # PR #102 C-1 fix: C1 is now unconditionally baseline-relative.
            # `messages_before_prune` must be the REAL pre-prune messages so
            # `before_uuids` contains u1 (which was pruned). Without it,
            # merged_before = pruned + delta (no u1), so before_uuids lacks u1,
            # and C1 skips the a_late parentUuid as a cross-session pointer.
            # This mirrors how guard_prune_cycle and cmd_treat actually call
            # save_messages (they always thread the real pre-prune list).
            with self.assertRaises(PruneValidationError):
                save_messages(
                    session_path, pruned,
                    create_backup=False, snapshot=snap,
                    messages_before_prune=messages,  # real pre-prune state (required for C-1)
                )

            # 5. The live file must remain at its appended state (NOT replaced
            #    by the prune output), so the operator can retry next cycle.
            reloaded = load_messages(session_path)
            uuids = {m[1].get("uuid") for m in reloaded}
            # u1 must still be present (the failed save did NOT replace).
            self.assertIn("u1", uuids,
                          "save_messages must not overwrite the live file on "
                          "post-append validation failure")
            # The appended orphan should also still be there in the live file.
            self.assertIn("a_late", uuids)


if __name__ == "__main__":
    unittest.main()
