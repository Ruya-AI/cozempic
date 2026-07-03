"""Regression tests for shared-task-list keying in extract_team_state.

Bug: TaskCreate's tool *input* carries no id — the system assigns it and returns
it only in the tool_result ("Task #N created"). The extractor used to key created
tasks positionally (str(len(seen_tasks)) -> "0","1",...), but TaskUpdate keys by
the real system id (taskId, 1-based). The two namespaces diverge, so:
  - a TaskUpdate(completed) lands on the wrong slot,
  - the first-created task (key "0") is never matched by any update,
  - high-id updates fabricate phantom blank tasks,
so genuinely-completed tasks render as pending in the recovered checkpoint (which
is injected back into context post-compact).

Fix: re-key each created task to its real id parsed from the "Task #N created"
tool_result; reused ids (task-store resets) overwrite older generations.

These tests fail on the positional-keying base and pass after the fix.
"""

import unittest
from unittest.mock import patch


def _tool_use(id_, name, inp):
    return {"message": {"role": "assistant", "content": [
        {"type": "tool_use", "id": id_, "name": name, "input": inp}
    ]}}


def _tool_result(tool_use_id, text):
    return {"message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}
    ]}}


def _extract(msgs):
    """extract_team_state with config isolation (no live ~/.claude/teams/ merge)."""
    from cozempic.team import extract_team_state
    with patch("cozempic.team.load_team_configs", return_value=[]):
        return extract_team_state(msgs)


def _create(idx, uid, real_id, subject):
    """A TaskCreate tool_use immediately followed by its 'Task #N created' result."""
    return [
        (idx, _tool_use(uid, "TaskCreate", {"subject": subject}), 100),
        (idx + 1, _tool_result(uid, f"Task #{real_id} created successfully: {subject}"), 80),
    ]


def _update(idx, uid, real_id, status):
    return [(idx, _tool_use(uid, "TaskUpdate", {"taskId": str(real_id), "status": status}), 80)]


def _by_subject(state, subject):
    return [t for t in state.tasks if (t.subject or "").strip() == subject]


class TestTaskListKeying(unittest.TestCase):
    def test_completed_task_not_rendered_pending(self):
        """A created-then-completed task must read 'completed', not 'pending'."""
        msgs = []
        msgs += _create(0, "c5", 5, "Ship the widget")
        msgs += _update(2, "u5", 5, "completed")
        state = _extract(msgs)

        matches = _by_subject(state, "Ship the widget")
        self.assertEqual(len(matches), 1, "task should appear exactly once")
        self.assertEqual(matches[0].status, "completed",
                         "completed task must not render as pending")

    def test_no_phantom_blank_task_from_update(self):
        """The TaskUpdate must not fabricate a blank-subject phantom task."""
        msgs = []
        msgs += _create(0, "c1", 1, "Only task")
        msgs += _update(2, "u1", 1, "completed")
        state = _extract(msgs)
        blanks = [t for t in state.tasks if not (t.subject or "").strip()]
        self.assertEqual(blanks, [], f"no phantom blank tasks expected, got {blanks}")

    def test_first_created_task_not_orphaned(self):
        """With several tasks all completed, none should remain pending."""
        msgs = []
        msgs += _create(0, "c1", 1, "First")
        msgs += _create(2, "c2", 2, "Second")
        msgs += _create(4, "c3", 3, "Third")
        msgs += _update(6, "u1", 1, "completed")
        msgs += _update(7, "u2", 2, "completed")
        msgs += _update(8, "u3", 3, "completed")
        state = _extract(msgs)
        pending = [t.subject for t in state.tasks
                   if (t.status or "") not in ("completed", "deleted")
                   and (t.subject or "").strip()]
        self.assertEqual(pending, [], f"all tasks completed; none should be pending, got {pending}")

    def test_id_reuse_latest_generation_wins(self):
        """A task-store reset reuses id #1; the latest generation must win."""
        msgs = []
        msgs += _create(0, "a1", 1, "Old gen task")
        msgs += _update(2, "ua1", 1, "completed")
        # store reset — id 1 reused for a brand-new task
        msgs += _create(4, "b1", 1, "New gen task")
        msgs += _update(6, "ub1", 1, "in_progress")
        state = _extract(msgs)

        new = _by_subject(state, "New gen task")
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0].status, "in_progress",
                         "latest-generation task #1 must reflect its own status")
        self.assertEqual(_by_subject(state, "Old gen task"), [],
                         "superseded generation must not linger")


class TestTaskListKeyingTornTranscripts(unittest.TestCase):
    """Hardening for torn/reordered/reworded transcripts (#167 review nits).

    The re-key depends on the create's tool_result existing AND matching
    `Task #N created`. When ordering or text drift breaks that, the fix must
    still not regress: never render a completed task as pending, never leak the
    internal `__pending_create_<uid>` sentinel, and recover if a good result
    arrives later.
    """

    def test_update_before_result_keeps_completed(self):
        """Out-of-order: TaskUpdate(completed) precedes the create-result.

        The re-key must NOT clobber the authoritative 'completed' with the
        create's 'pending'; it must backfill the subject onto the real entry.
        """
        msgs = [
            (0, _tool_use("c1", "TaskCreate", {"subject": "Build feature X"}), 100),
            (1, _tool_use("u1", "TaskUpdate", {"taskId": "1", "status": "completed", "owner": "alice"}), 80),
            (2, _tool_result("c1", "Task #1 created successfully: Build feature X"), 80),
        ]
        state = _extract(msgs)
        matches = _by_subject(state, "Build feature X")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].status, "completed",
                         "out-of-order completion must survive the re-key")
        self.assertEqual(matches[0].owner, "alice", "owner from the update must survive")

    def test_missing_result_no_sentinel_leak(self):
        """A create whose result was truncated away must not leak the temp key."""
        msgs = [(0, _tool_use("c1", "TaskCreate", {"subject": "Orphan task"}), 100)]
        state = _extract(msgs)
        leaked = [t for t in state.tasks if t.task_id.startswith("__pending_create_")]
        self.assertEqual(leaked, [], f"internal sentinel must not reach output, got {leaked}")
        self.assertEqual(_by_subject(state, "Orphan task")[0].status, "pending")

    def test_regex_miss_then_duplicate_result_recovers(self):
        """A reworded first result must not consume the mapping — a later good
        (duplicate) result must still recover the real id."""
        msgs = [
            (0, _tool_use("c1", "TaskCreate", {"subject": "Retry task"}), 100),
            (1, _tool_result("c1", "Created task 1 (reworded, no canonical phrase)"), 80),
            (2, _tool_result("c1", "Task #1 created successfully: Retry task"), 80),
            (3, _tool_use("u1", "TaskUpdate", {"taskId": "1", "status": "completed"}), 80),
        ]
        state = _extract(msgs)
        matches = _by_subject(state, "Retry task")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].status, "completed",
                         "duplicate good result must recover the id after a regex miss")
        self.assertEqual(matches[0].task_id, "1")


if __name__ == "__main__":
    unittest.main()
