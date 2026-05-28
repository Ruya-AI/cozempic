"""RED tests for last-of-type metadata protection (P0-D).

Per PLAN.md § 2.5:
- `executor._tag_last_of_metadata_types(messages)` tags the LAST occurrence of
  each protected-singleton type (`ai-title`, `last-prompt`, `permission-mode`)
  with `__cozempic_metadata_singleton__=True`. Called as the FIRST step in
  `run_prescription`.
- `helpers.is_protected()` is extended to honor the tag, so any strategy that
  respects `is_protected` (all of them) skips the tagged singleton.
- After strategies finish, `run_prescription` strips the tag before returning —
  it MUST NOT persist to disk.
- `gentle.py:_META_TYPES` adds `permission-mode` as defense in depth.

These tests SHOULD FAIL until P0-D lands:
- `test_last_permission_mode_protected` fails on current code because
  `permission-mode` is NOT in gentle._META_TYPES.
- `test_singleton_protection_survives_compact_summary_collapse` fails for the
  same reason.
"""

from __future__ import annotations

import unittest

from cozempic.helpers import msg_bytes


def _msg(idx: int, payload: dict) -> tuple[int, dict, int]:
    return (idx, payload, msg_bytes(payload))


def _build_session_with_metadata_scattered(
    n_ai_title: int = 100,
    n_last_prompt: int = 100,
    n_permission_mode: int = 100,
    include_compact_boundary: bool = False,
) -> list[tuple[int, dict, int]]:
    """Build a session with `n_*` of each metadata type scattered through.

    Layout (line indices):
      0       root user
      1..N    interleaved metadata + conversation + (optional) boundary
      N+1..   final assistant
    """
    msgs: list[tuple[int, dict, int]] = []
    line = 0
    msgs.append(_msg(line, {
        "type": "user", "uuid": "u0", "parentUuid": None,
        "message": {"role": "user", "content": "root"},
    }))
    line += 1
    prev = "u0"

    # Interleave metadata + a few conversational entries.
    for i in range(max(n_ai_title, n_last_prompt, n_permission_mode)):
        if i < n_ai_title:
            uid = f"t{i}"
            msgs.append(_msg(line, {
                "type": "ai-title", "uuid": uid, "parentUuid": prev,
                "title": f"title {i}",
            }))
            line += 1
        if i < n_last_prompt:
            uid = f"lp{i}"
            msgs.append(_msg(line, {
                "type": "last-prompt", "uuid": uid, "parentUuid": prev,
                "text": f"prompt {i}",
            }))
            line += 1
        if i < n_permission_mode:
            uid = f"pm{i}"
            msgs.append(_msg(line, {
                "type": "permission-mode", "uuid": uid, "parentUuid": prev,
                "mode": f"mode_{i}",
            }))
            line += 1
        # Inject conversation turn every few iterations.
        if i % 10 == 0:
            uid = f"u{i+1}"
            msgs.append(_msg(line, {
                "type": "user", "uuid": uid, "parentUuid": prev,
                "message": {"role": "user", "content": f"u {i}"},
            }))
            prev = uid
            line += 1
            aid = f"a{i}"
            msgs.append(_msg(line, {
                "type": "assistant", "uuid": aid, "parentUuid": prev,
                "message": {"role": "assistant",
                            "content": [{"type": "text", "text": f"a {i}"}]},
            }))
            prev = aid
            line += 1

    if include_compact_boundary:
        msgs.append(_msg(line, {
            "type": "system", "subtype": "compact_boundary",
            "uuid": "cb1", "parentUuid": prev,
        }))
        line += 1

    # Final assistant + last metadata anchors POST-boundary if requested.
    msgs.append(_msg(line, {
        "type": "assistant", "uuid": "aF", "parentUuid": prev,
        "message": {"role": "assistant",
                    "content": [{"type": "text", "text": "final"}]},
    }))
    line += 1
    return msgs


class TestLastOfTypeMetadataProtected(unittest.TestCase):
    """The LAST entry of each metadata-singleton type survives every prescription."""

    def test_last_ai_title_protected(self):
        from cozempic.executor import run_prescription
        from cozempic.registry import PRESCRIPTIONS
        import cozempic.strategies  # noqa: F401

        before = _build_session_with_metadata_scattered(
            n_ai_title=100, n_last_prompt=10, n_permission_mode=10,
            include_compact_boundary=True,
        )
        ai_titles = [m for m in before if m[1].get("type") == "ai-title"]
        last_ai_uuid = ai_titles[-1][1]["uuid"]

        after, _ = run_prescription(before, PRESCRIPTIONS["gentle"], {})
        surviving = {m[1].get("uuid") for m in after if m[1].get("type") == "ai-title"}
        self.assertIn(last_ai_uuid, surviving,
                      "last ai-title was dropped by gentle prescription")

    def test_last_last_prompt_protected(self):
        from cozempic.executor import run_prescription
        from cozempic.registry import PRESCRIPTIONS
        import cozempic.strategies  # noqa: F401

        before = _build_session_with_metadata_scattered(
            n_ai_title=10, n_last_prompt=100, n_permission_mode=10,
            include_compact_boundary=True,
        )
        lps = [m for m in before if m[1].get("type") == "last-prompt"]
        last_lp_uuid = lps[-1][1]["uuid"]

        after, _ = run_prescription(before, PRESCRIPTIONS["gentle"], {})
        surviving = {m[1].get("uuid") for m in after if m[1].get("type") == "last-prompt"}
        self.assertIn(last_lp_uuid, surviving)

    def test_last_permission_mode_protected(self):
        """RED on current code: gentle._META_TYPES does NOT include permission-mode.

        This is the bug from the handoff (-98% permission-mode loss). After fix:
        the singleton-protection pre-pass tags the last permission-mode, and
        compact-summary-collapse honors is_protected() to skip it.
        """
        from cozempic.executor import run_prescription
        from cozempic.registry import PRESCRIPTIONS
        import cozempic.strategies  # noqa: F401

        before = _build_session_with_metadata_scattered(
            n_ai_title=10, n_last_prompt=10, n_permission_mode=100,
            include_compact_boundary=True,
        )
        pms = [m for m in before if m[1].get("type") == "permission-mode"]
        last_pm_uuid = pms[-1][1]["uuid"]

        after, _ = run_prescription(before, PRESCRIPTIONS["gentle"], {})
        surviving = {m[1].get("uuid")
                     for m in after if m[1].get("type") == "permission-mode"}
        self.assertIn(last_pm_uuid, surviving,
                      "last permission-mode was dropped — the documented bug")


class TestPreBoundaryMetadataDroppedWhenReplacedPostBoundary(unittest.TestCase):
    """When the LAST metadata of a type appears POST-boundary, pre-boundary entries
    of that same type can be dropped — matches existing gentle.py:45 exemption logic.

    Specifically: 50 ai-titles pre-boundary + 5 post-boundary. Post-boundary ones
    are LATER → tagged as singletons; pre-boundary ones get collapsed.
    """

    def test_pre_boundary_metadata_dropped_when_replaced_post_boundary(self):
        from cozempic.executor import run_prescription
        from cozempic.registry import PRESCRIPTIONS
        import cozempic.strategies  # noqa: F401

        msgs: list[tuple[int, dict, int]] = []
        line = 0
        # Root user
        msgs.append(_msg(line, {
            "type": "user", "uuid": "u0", "parentUuid": None,
            "message": {"role": "user", "content": "root"},
        }))
        line += 1
        prev = "u0"

        # 50 pre-boundary ai-titles
        pre_uuids = []
        for i in range(50):
            uid = f"pre_t{i}"
            pre_uuids.append(uid)
            msgs.append(_msg(line, {
                "type": "ai-title", "uuid": uid, "parentUuid": prev,
                "title": f"pre {i}",
            }))
            line += 1

        # A user/assistant pair to keep C3 satisfied.
        msgs.append(_msg(line, {
            "type": "user", "uuid": "u1", "parentUuid": prev,
            "message": {"role": "user", "content": "u1"},
        }))
        line += 1
        prev = "u1"
        msgs.append(_msg(line, {
            "type": "assistant", "uuid": "a1", "parentUuid": prev,
            "message": {"role": "assistant",
                        "content": [{"type": "text", "text": "a1"}]},
        }))
        line += 1
        prev = "a1"

        # Compact boundary
        msgs.append(_msg(line, {
            "type": "system", "subtype": "compact_boundary",
            "uuid": "cb1", "parentUuid": prev,
        }))
        line += 1

        # 5 post-boundary ai-titles
        post_uuids = []
        for i in range(5):
            uid = f"post_t{i}"
            post_uuids.append(uid)
            msgs.append(_msg(line, {
                "type": "ai-title", "uuid": uid, "parentUuid": prev,
                "title": f"post {i}",
            }))
            line += 1

        # Final user+assistant + last-prompt + permission-mode (so C3/C5/C6 pass).
        msgs.append(_msg(line, {
            "type": "user", "uuid": "u2", "parentUuid": prev,
            "message": {"role": "user", "content": "u2"},
        }))
        line += 1
        msgs.append(_msg(line, {
            "type": "assistant", "uuid": "a2", "parentUuid": "u2",
            "message": {"role": "assistant",
                        "content": [{"type": "text", "text": "a2"}]},
        }))
        line += 1
        msgs.append(_msg(line, {
            "type": "last-prompt", "uuid": "lp1", "parentUuid": "a2",
            "text": "lp",
        }))
        line += 1
        msgs.append(_msg(line, {
            "type": "permission-mode", "uuid": "pm1", "parentUuid": "a2",
            "mode": "default",
        }))
        line += 1

        after, _ = run_prescription(msgs, PRESCRIPTIONS["gentle"], {})
        surviving = {m[1].get("uuid")
                     for m in after if m[1].get("type") == "ai-title"}

        # Last post-boundary ai-title must survive (singleton).
        self.assertIn(post_uuids[-1], surviving)
        # Pre-boundary ai-titles may be dropped (compact-summary-collapse).
        # Assert at least the OLDEST few are dropped to confirm collapse fired.
        self.assertNotIn(pre_uuids[0], surviving)


class TestSingletonProtectionSurvivesCompactSummaryCollapse(unittest.TestCase):
    """When metadata appears ONLY pre-boundary (not post), the last one must
    still survive via singleton protection. Current code FAILS for permission-mode."""

    def test_singleton_protection_survives_when_no_post_boundary_replacement(self):
        from cozempic.executor import run_prescription
        from cozempic.registry import PRESCRIPTIONS
        import cozempic.strategies  # noqa: F401

        msgs: list[tuple[int, dict, int]] = []
        line = 0
        msgs.append(_msg(line, {
            "type": "user", "uuid": "u0", "parentUuid": None,
            "message": {"role": "user", "content": "root"},
        }))
        line += 1
        prev = "u0"
        # 50 permission-mode pre-boundary
        pre_pm = []
        for i in range(50):
            uid = f"pm{i}"
            pre_pm.append(uid)
            msgs.append(_msg(line, {
                "type": "permission-mode", "uuid": uid, "parentUuid": prev,
                "mode": f"m{i}",
            }))
            line += 1

        # User+assistant pair to satisfy C3
        msgs.append(_msg(line, {
            "type": "user", "uuid": "u1", "parentUuid": prev,
            "message": {"role": "user", "content": "u1"},
        }))
        line += 1
        msgs.append(_msg(line, {
            "type": "assistant", "uuid": "a1", "parentUuid": "u1",
            "message": {"role": "assistant",
                        "content": [{"type": "text", "text": "a1"}]},
        }))
        line += 1

        # Compact boundary
        msgs.append(_msg(line, {
            "type": "system", "subtype": "compact_boundary",
            "uuid": "cb1", "parentUuid": "a1",
        }))
        line += 1

        # 0 post-boundary permission-mode entries (intentional). Need
        # last-prompt to satisfy C6.
        msgs.append(_msg(line, {
            "type": "user", "uuid": "u2", "parentUuid": "cb1",
            "message": {"role": "user", "content": "u2"},
        }))
        line += 1
        msgs.append(_msg(line, {
            "type": "assistant", "uuid": "a2", "parentUuid": "u2",
            "message": {"role": "assistant",
                        "content": [{"type": "text", "text": "a2"}]},
        }))
        line += 1
        msgs.append(_msg(line, {
            "type": "last-prompt", "uuid": "lp1", "parentUuid": "a2",
            "text": "lp",
        }))
        line += 1

        after, _ = run_prescription(msgs, PRESCRIPTIONS["gentle"], {})
        surviving = {m[1].get("uuid")
                     for m in after if m[1].get("type") == "permission-mode"}

        # The LAST pre-boundary permission-mode must survive.
        self.assertIn(pre_pm[-1], surviving,
                      "last permission-mode was lost — the documented bug")


class TestSingletonTagStrippedBeforeReturn(unittest.TestCase):
    """run_prescription must strip __cozempic_metadata_singleton__ before
    returning — the tag is internal-only and MUST NOT persist to disk."""

    def test_tag_not_present_in_output(self):
        from cozempic.executor import run_prescription
        from cozempic.registry import PRESCRIPTIONS
        import cozempic.strategies  # noqa: F401

        before = _build_session_with_metadata_scattered(
            n_ai_title=5, n_last_prompt=5, n_permission_mode=5,
            include_compact_boundary=False,
        )
        after, _ = run_prescription(before, PRESCRIPTIONS["gentle"], {})
        for _, msg, _ in after:
            self.assertNotIn(
                "__cozempic_metadata_singleton__", msg,
                f"internal tag leaked to output: uuid={msg.get('uuid')!r}",
            )


if __name__ == "__main__":
    unittest.main()
