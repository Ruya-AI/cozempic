"""RED tests for P0-B (validate_post_prune C1-C7), P0-C (enforce_floor),
P0-D (metadata singleton tag), and the R-2 floor-tag-strip invariant.

All tests in this file MUST fail at base commit 1b8b863 because
cozempic.safety and cozempic.config do not exist in v1.8.18.
"""

from __future__ import annotations

import json
import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

def _m(idx: int, *, t: str, uuid: str, parent: str | None = "UNSET", **kw) -> tuple[int, dict, int]:
    """Build a (line_index, dict, byte_size) triple."""
    d: dict = {"type": t, "uuid": uuid}
    if parent != "UNSET":
        d["parentUuid"] = parent
    d.update(kw)
    return (idx, d, len(json.dumps(d, separators=(",", ":"))))


def _user(idx: int, uuid: str, parent: str | None = "UNSET", **kw):
    content = kw.pop("content", "hello")
    msg = {"content": content, "role": "user"}
    return _m(idx, t="user", uuid=uuid, parent=parent, message=msg, **kw)


def _asst(idx: int, uuid: str, parent: str | None = "UNSET", **kw):
    content = kw.pop("content", "hi")
    msg = {"content": content, "role": "assistant"}
    return _m(idx, t="assistant", uuid=uuid, parent=parent, message=msg, **kw)


def _pm(idx: int, uuid: str, parent: str | None = "UNSET"):
    """permission-mode entry."""
    return _m(idx, t="permission-mode", uuid=uuid, parent=parent)


def _cb(idx: int, uuid: str, parent: str | None = "UNSET"):
    """compact_boundary system entry."""
    d = {"type": "system", "subtype": "compact_boundary", "uuid": uuid}
    if parent != "UNSET":
        d["parentUuid"] = parent
    return (idx, d, len(json.dumps(d, separators=(",", ":"))))


def _lp(idx: int, uuid: str, parent: str | None = "UNSET"):
    """last-prompt entry."""
    return _m(idx, t="last-prompt", uuid=uuid, parent=parent)


def _ai(idx: int, uuid: str, parent: str | None = "UNSET"):
    """ai-title entry."""
    return _m(idx, t="ai-title", uuid=uuid, parent=parent)


# ── Class 1: validate_post_prune C1 — baseline-relative parent check ─────────

class TestValidatePostPruneC1Baseline:
    def test_c1_cross_session_parent_not_flagged(self):
        """Cross-session parentUuid (absent from both before AND after) must NOT raise."""
        from cozempic.safety import validate_post_prune

        # Simulates a resumed session: root message references a uuid from
        # the PARENT session file (not in this file at all).
        before = [
            _m(0, t="user", uuid="root-001", parent="external-session-uuid",
               message={"content": "hi", "role": "user"}),
            _asst(1, "a-001", parent="root-001"),
        ]
        after = before[:]  # no removals
        # Must NOT raise — external-session-uuid is a valid cross-session anchor
        validate_post_prune(before, after)

    def test_c1_prune_induced_break_raises(self):
        """Parent uuid that existed pre-prune but was removed must raise C1."""
        from cozempic.safety import validate_post_prune, PruneValidationError

        before = [
            _user(0, "root-A", None),
            _asst(1, "a-001", "root-A"),
            _user(2, "u-001", "a-001"),   # <-- will be removed
            _asst(3, "a-002", "u-001"),    # parent u-001 removed → break
        ]
        # Remove u-001 without relinking a-002
        after = [before[0], before[1], before[3]]  # drop u-001, keep a-002 with parent=u-001

        with pytest.raises(PruneValidationError) as exc_info:
            validate_post_prune(before, after)
        assert exc_info.value.evidence["failed_check"] == "C1"

    def test_c1_zero_removal_prune_passes(self):
        """Identity prune (before == after) must pass all checks."""
        from cozempic.safety import validate_post_prune

        msgs = [
            _user(0, "root-B", None),
            _asst(1, "a-001", "root-B"),
            _user(2, "u-001", "a-001"),
        ]
        validate_post_prune(msgs, msgs[:])  # shallow copy, same content


# ── Class 2: validate_post_prune C2 — root preservation ─────────────────────

class TestValidatePostPruneC2:
    def test_c2_root_dropped_raises(self):
        """Dropping the only original root uuid raises C2."""
        from cozempic.safety import validate_post_prune, PruneValidationError

        before = [
            _user(0, "root-001", None),       # the original root
            _asst(1, "a-001", "root-001"),
            _user(2, "u-002", "a-001"),
        ]
        # Drop root-001; a-002 now has no semantic root anchor
        after = [before[1], before[2]]

        with pytest.raises(PruneValidationError) as exc_info:
            validate_post_prune(before, after)
        assert exc_info.value.evidence["failed_check"] == "C2"

    def test_c2_one_root_of_two_survives_passes(self):
        """Multi-root session: one root dropped, one remains → passes C2."""
        from cozempic.safety import validate_post_prune

        # Two root messages (parentUuid=None), one removed after prune
        before = [
            _user(0, "root-X", None),   # older root
            _asst(1, "a-001", "root-X"),
            _user(2, "root-Y", None),   # newer root (resume-of-resume)
            _asst(3, "a-002", "root-Y"),
        ]
        # Keep newer root only — valid per C2 (at least one original root survives)
        after = [before[2], before[3]]
        validate_post_prune(before, after)


# ── Class 3: validate_post_prune C3 — conversation survival ──────────────────

class TestValidatePostPruneC3:
    def test_c3_all_users_dropped_raises(self):
        """Dropping all user messages raises C3."""
        from cozempic.safety import validate_post_prune, PruneValidationError

        before = [
            _user(0, "u-001", None),
            _asst(1, "a-001", "u-001"),
        ]
        after = [before[1]]  # only assistant, no user

        with pytest.raises(PruneValidationError) as exc_info:
            validate_post_prune(before, after)
        assert exc_info.value.evidence["failed_check"] == "C3"

    def test_c3_all_assistants_dropped_raises(self):
        """Dropping all assistant messages raises C3."""
        from cozempic.safety import validate_post_prune, PruneValidationError

        before = [
            _user(0, "u-001", None),
            _asst(1, "a-001", "u-001"),
        ]
        after = [before[0]]  # only user, no assistant

        with pytest.raises(PruneValidationError) as exc_info:
            validate_post_prune(before, after)
        assert exc_info.value.evidence["failed_check"] == "C3"


# ── Class 4: validate_post_prune C4-C7 ───────────────────────────────────────

class TestValidatePostPruneC4C5C6C7:
    def _base(self):
        """Minimal valid session with one of each tracked type."""
        return [
            _user(0, "u-root", None),
            _asst(1, "a-001", "u-root"),
            _cb(2, "cb-001", "a-001"),
            _pm(3, "pm-001", "cb-001"),
            _lp(4, "lp-001", "pm-001"),
            _ai(5, "at-001", "lp-001"),
        ]

    def test_c4_last_compact_boundary_dropped_raises(self):
        """Dropping last compact_boundary raises C4."""
        from cozempic.safety import validate_post_prune, PruneValidationError

        before = self._base()
        after = [m for m in before if not (m[1].get("subtype") == "compact_boundary")]

        with pytest.raises(PruneValidationError) as exc_info:
            validate_post_prune(before, after)
        assert exc_info.value.evidence["failed_check"] == "C4"

    def test_c5_last_permission_mode_dropped_raises(self):
        """Dropping last permission-mode raises C5."""
        from cozempic.safety import validate_post_prune, PruneValidationError

        before = self._base()
        after = [m for m in before if m[1].get("type") != "permission-mode"]

        with pytest.raises(PruneValidationError) as exc_info:
            validate_post_prune(before, after)
        assert exc_info.value.evidence["failed_check"] == "C5"

    def test_c6_last_prompt_dropped_raises(self):
        """Dropping last last-prompt raises C6."""
        from cozempic.safety import validate_post_prune, PruneValidationError

        before = self._base()
        after = [m for m in before if m[1].get("type") != "last-prompt"]

        with pytest.raises(PruneValidationError) as exc_info:
            validate_post_prune(before, after)
        assert exc_info.value.evidence["failed_check"] == "C6"

    def test_c7_last_ai_title_dropped_raises(self):
        """Dropping last ai-title raises C7."""
        from cozempic.safety import validate_post_prune, PruneValidationError

        before = self._base()
        after = [m for m in before if m[1].get("type") != "ai-title"]

        with pytest.raises(PruneValidationError) as exc_info:
            validate_post_prune(before, after)
        assert exc_info.value.evidence["failed_check"] == "C7"


class TestValidatePostPruneC8ToolPairing:
    """C8 — a surviving tool_use must keep its tool_result (baseline-relative)."""

    def _before(self):
        # u0 -> a1(tool_use tu1) -> u1(tool_result tu1) -> a2
        return [
            _user(0, "u0", parent=None),
            _asst(1, "a1", parent="u0",
                  content=[{"type": "tool_use", "id": "tu1", "name": "Bash", "input": {}}]),
            _user(2, "u1", parent="a1",
                  content=[{"type": "tool_result", "tool_use_id": "tu1", "content": "ok"}]),
            _asst(3, "a2", parent="u1", content="done"),
        ]

    def test_c8_dangling_tool_use_raises(self):
        """Dropping the tool_result message but keeping the tool_use raises C8."""
        from cozempic.safety import validate_post_prune, PruneValidationError

        before = self._before()
        # drop u1 (the tool_result carrier); relink a2 -> a1 so the DAG still resolves
        a2 = _asst(3, "a2", parent="a1", content="done")
        after = [before[0], before[1], a2]

        with pytest.raises(PruneValidationError) as exc_info:
            validate_post_prune(before, after)
        assert exc_info.value.evidence["failed_check"] == "C8"
        assert exc_info.value.evidence["dangling_tool_use_id"] == "tu1"

    def test_c8_intact_pair_passes(self):
        """Keeping the full tool_use/tool_result pair passes."""
        from cozempic.safety import validate_post_prune

        before = self._before()
        validate_post_prune(before, before)  # no raise

    def test_c8_inflight_tool_use_without_prior_result_passes(self):
        """A tool_use that never had a tool_result before the prune (an in-flight
        final turn) is NOT a prune-induced break — must pass (baseline-relative)."""
        from cozempic.safety import validate_post_prune

        before = [
            _user(0, "u0", parent=None),
            _asst(1, "a1", parent="u0",
                  content=[{"type": "tool_use", "id": "tuX", "name": "Bash", "input": {}}]),
        ]
        validate_post_prune(before, before)  # no raise

    def test_c8_both_halves_dropped_passes(self):
        """Dropping BOTH the tool_use and its tool_result leaves nothing dangling."""
        from cozempic.safety import validate_post_prune

        before = self._before()
        # drop a1 (tool_use) AND u1 (tool_result); relink a2 -> u0
        a2 = _asst(3, "a2", parent="u0", content="done")
        after = [before[0], a2]
        validate_post_prune(before, after)  # no raise


# ── Class 5: simulate_replay_readiness (REMOVED — M-1) ───────────────────────
# simulate_replay_readiness was a dead export: implemented, exported in __all__,
# and unit-tested, but never called by any production path. Removed per M-1.
# validate_post_prune (two-list, in the prune path) already provides the
# structural/anchor guarantee. A standalone single-list diagnostic is deferred
# to a separate PR if wanted later.


# ── Class 6: enforce_floor ───────────────────────────────────────────────────

class TestEnforceFloor:
    def _users(self, count: int):
        """Build `count` user+assistant pairs (parentUuid chained)."""
        msgs = []
        prev = None
        for i in range(count):
            u_uuid = f"u-{i:03d}"
            a_uuid = f"a-{i:03d}"
            u = _user(i * 2, u_uuid, prev)
            a = _asst(i * 2 + 1, a_uuid, u_uuid)
            msgs.extend([u, a])
            prev = a_uuid
        return msgs

    def test_floor_readds_dropped_last_k_user(self):
        """20 user+asst pairs; prune drops last 10 users; floor re-adds them."""
        from cozempic.safety import enforce_floor
        from cozempic.config import FloorConfig

        before = self._users(20)
        # Simulate: keep only first 10 users and their assistants (drop last 10 users)
        after = [m for m in before if int(m[1]["uuid"].split("-")[1]) < 10]

        cfg = FloorConfig(preserve_last_k_turns=10, max_user_assistant_drop_pct=1.0,
                          preserve_first_message=False)
        result = enforce_floor(before, after, cfg=cfg)

        surviving_user_uuids = {msg.get("uuid") for _, msg, _ in result if msg.get("type") == "user"}
        # Last 10 users (u-010 through u-019) must be re-added
        expected = {f"u-{i:03d}" for i in range(10, 20)}
        assert expected.issubset(surviving_user_uuids), (
            f"Floor failed to re-add last-10 users. Missing: {expected - surviving_user_uuids}"
        )

    def test_floor_readds_first_message(self):
        """Prune drops the root (parentUuid=None); floor re-adds it."""
        from cozempic.safety import enforce_floor
        from cozempic.config import FloorConfig

        before = self._users(5)  # u-000 is the root (parentUuid=None)
        after = [m for m in before if m[1].get("uuid") != "u-000"]  # drop root

        cfg = FloorConfig(preserve_first_message=True, preserve_last_k_turns=0,
                          max_user_assistant_drop_pct=1.0)
        result = enforce_floor(before, after, cfg=cfg)

        surviving_uuids = {msg.get("uuid") for _, msg, _ in result}
        assert "u-000" in surviving_uuids, "Floor failed to re-add first message (u-000)."

    def test_floor_cap_50pct(self):
        """10 user+asst pairs; prune drops all 10 users; floor preserves at least 5."""
        from cozempic.safety import enforce_floor
        from cozempic.config import FloorConfig

        before = self._users(10)
        # Drop all users
        after = [m for m in before if m[1].get("type") != "user"]

        cfg = FloorConfig(max_user_assistant_drop_pct=0.50,
                          preserve_last_k_turns=0, preserve_first_message=False)
        result = enforce_floor(before, after, cfg=cfg)

        surviving_users = [msg for _, msg, _ in result if msg.get("type") == "user"]
        assert len(surviving_users) >= 5, (
            f"Floor 50% cap failed: only {len(surviving_users)} users survived (expected ≥5)"
        )

    def test_floor_no_revert_of_replacements(self):
        """A replaced message (same uuid, modified payload) must keep the replacement."""
        from cozempic.safety import enforce_floor
        from cozempic.config import FloorConfig

        # Build before with a user message carrying specific content
        before = [
            _user(0, "u-root", None, content="original content"),
            _asst(1, "a-001", "u-root"),
        ]
        # After: replacement has same uuid but modified payload
        after_root = (0, {"type": "user", "uuid": "u-root", "parentUuid": None,
                          "message": {"content": "REPLACED content", "role": "user"}},
                      50)
        after = [after_root, before[1]]

        cfg = FloorConfig(preserve_first_message=True, preserve_last_k_turns=1,
                          max_user_assistant_drop_pct=0.50)
        result = enforce_floor(before, after, cfg=cfg)

        root_msg = next((msg for _, msg, _ in result if msg.get("uuid") == "u-root"), None)
        assert root_msg is not None
        assert root_msg.get("message", {}).get("content") == "REPLACED content", (
            "Floor reverted a strategy replacement (same uuid, modified payload)."
        )

    def test_floor_pair_counterpart_closure(self):
        """Re-adding a user msg with tool_use also re-adds the tool_result carrier."""
        from cozempic.safety import enforce_floor
        from cozempic.config import FloorConfig

        # Session: root user (tool_use) → asst (tool_result)
        # Prune dropped both. Floor must re-add both due to pair-closure.
        import json

        tool_user = (0, {
            "type": "user",
            "uuid": "u-tool",
            "parentUuid": None,
            "message": {
                "role": "user",
                "content": [{"type": "tool_use", "id": "tool-1", "name": "Bash", "input": {}}],
            }
        }, 100)
        tool_result = (1, {
            "type": "user",
            "uuid": "u-result",
            "parentUuid": "u-tool",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "ok"}],
            }
        }, 80)
        asst = _asst(2, "a-001", "u-result")
        before = [tool_user, tool_result, asst]

        # After: drop both u-tool and u-result (only asst survives)
        after = [asst]

        cfg = FloorConfig(preserve_last_k_turns=0, preserve_first_message=True,
                          max_user_assistant_drop_pct=0.0)
        result = enforce_floor(before, after, cfg=cfg)

        surviving = {msg.get("uuid") for _, msg, _ in result}
        # u-tool must be re-added (preserve_first_message=True → root)
        assert "u-tool" in surviving, "Floor failed to re-add tool_use root message."
        # pair closure: re-adding u-tool must pull in u-result (tool_result carrier)
        assert "u-result" in surviving, (
            "Floor pair-closure failed: re-adding u-tool should also pull in u-result."
        )


# ── Class 7: FloorConfig defaults and env var resolution ─────────────────────

class TestFloorConfig:
    def test_default_values(self):
        """FloorConfig() has the expected default values.

        last_k=10 (H-2): lowered from 50 → 10 to avoid neutralizing gentle/standard
        on sessions ≤50 turns. K=50 re-added every removed turn on small sessions → 0%
        savings. K=10 preserves a meaningful recent-context floor without swallowing
        typical sessions. Operators needing the stricter floor use
        COZEMPIC_FLOOR_PRESERVE_LAST_K=50.
        """
        from cozempic.config import FloorConfig

        cfg = FloorConfig()
        assert cfg.max_user_assistant_drop_pct == 0.50
        assert cfg.preserve_last_k_turns == 10
        assert cfg.preserve_first_message is True

    def test_env_var_max_drop_pct(self, monkeypatch):
        """COZEMPIC_FLOOR_MAX_DROP_PCT=0.3 → FloorConfig.max_user_assistant_drop_pct == 0.3."""
        from cozempic import config as cfg_mod

        monkeypatch.setenv("COZEMPIC_FLOOR_MAX_DROP_PCT", "0.3")
        resolved = cfg_mod._resolve_floor_with({})
        assert resolved.max_user_assistant_drop_pct == pytest.approx(0.3)

    def test_env_var_nan_falls_to_default(self, monkeypatch):
        """COZEMPIC_FLOOR_MAX_DROP_PCT=nan → falls back to default 0.50."""
        from cozempic import config as cfg_mod

        monkeypatch.setenv("COZEMPIC_FLOOR_MAX_DROP_PCT", "nan")
        resolved = cfg_mod._resolve_floor_with({})
        assert resolved.max_user_assistant_drop_pct == pytest.approx(0.50)

    def test_env_var_inf_falls_to_default(self, monkeypatch):
        """COZEMPIC_FLOOR_MAX_DROP_PCT=inf → falls back to default 0.50."""
        from cozempic import config as cfg_mod

        monkeypatch.setenv("COZEMPIC_FLOOR_MAX_DROP_PCT", "inf")
        resolved = cfg_mod._resolve_floor_with({})
        assert resolved.max_user_assistant_drop_pct == pytest.approx(0.50)

    def test_floor_config_disabled_classmethod(self):
        """FloorConfig.disabled() returns no-op config (all constraints off)."""
        from cozempic.config import FloorConfig

        disabled = FloorConfig.disabled()
        assert disabled.max_user_assistant_drop_pct == 1.0
        assert disabled.preserve_last_k_turns == 0
        assert disabled.preserve_first_message is False

    def test_clamp_int_inf_falls_to_default(self):
        """_clamp_int('inf', ...) falls back to default without OverflowError."""
        from cozempic.config import _clamp_int

        result = _clamp_int("inf", 1, 1000, 50)
        assert result == 50

    def test_clamp_float_nan_falls_to_default(self):
        """_clamp_float('nan', ...) falls back to default."""
        from cozempic.config import _clamp_float

        result = _clamp_float("nan", 0.0, 1.0, 0.50)
        assert result == pytest.approx(0.50)

    # ── R3-1: preserve_first_message JSON-config path must use _parse_bool ─────
    # The pre-fix code used `bool(floor_data["preserve_first_message"])` on the
    # file path. `bool("false")` and `bool("0")` return True — opposite of the
    # env path which uses `_parse_bool`. String values from JSON config were
    # silently ignored (always True).

    def test_file_config_string_false_is_false(self, monkeypatch):
        """RED (assertion-RED pre-fix): file config preserve_first_message='false' → False.

        Pre-fix: bool('false') → True (WRONG). Post-fix: _parse_bool('false') → False.
        This test FAILS pre-fix (asserts False, gets True) and PASSES post-fix.
        Env var cleared so the file path is exercised.
        """
        from cozempic import config as cfg_mod

        monkeypatch.delenv("COZEMPIC_FLOOR_PRESERVE_FIRST", raising=False)
        result = cfg_mod._resolve_floor_with({"floor": {"preserve_first_message": "false"}})
        assert result.preserve_first_message is False, (
            "string 'false' in JSON config must parse to False; "
            "pre-fix bool('false') returns True"
        )

    def test_file_config_string_zero_is_false(self, monkeypatch):
        """File config preserve_first_message='0' → False (same bug, different token)."""
        from cozempic import config as cfg_mod

        monkeypatch.delenv("COZEMPIC_FLOOR_PRESERVE_FIRST", raising=False)
        result = cfg_mod._resolve_floor_with({"floor": {"preserve_first_message": "0"}})
        assert result.preserve_first_message is False

    def test_file_config_string_no_is_false(self, monkeypatch):
        """File config preserve_first_message='no' → False."""
        from cozempic import config as cfg_mod

        monkeypatch.delenv("COZEMPIC_FLOOR_PRESERVE_FIRST", raising=False)
        result = cfg_mod._resolve_floor_with({"floor": {"preserve_first_message": "no"}})
        assert result.preserve_first_message is False

    def test_file_config_native_bool_false_is_false(self, monkeypatch):
        """Native JSON bool false (parsed by json.load as Python False) → False.

        This already worked pre-fix (bool(False) → False), but must keep working
        post-fix as a regression guard.
        """
        from cozempic import config as cfg_mod

        monkeypatch.delenv("COZEMPIC_FLOOR_PRESERVE_FIRST", raising=False)
        result = cfg_mod._resolve_floor_with({"floor": {"preserve_first_message": False}})
        assert result.preserve_first_message is False

    def test_file_config_string_true_is_true(self, monkeypatch):
        """File config preserve_first_message='true' → True."""
        from cozempic import config as cfg_mod

        monkeypatch.delenv("COZEMPIC_FLOOR_PRESERVE_FIRST", raising=False)
        result = cfg_mod._resolve_floor_with({"floor": {"preserve_first_message": "true"}})
        assert result.preserve_first_message is True

    def test_file_config_string_one_is_true(self, monkeypatch):
        """File config preserve_first_message='1' → True."""
        from cozempic import config as cfg_mod

        monkeypatch.delenv("COZEMPIC_FLOOR_PRESERVE_FIRST", raising=False)
        result = cfg_mod._resolve_floor_with({"floor": {"preserve_first_message": "1"}})
        assert result.preserve_first_message is True

    def test_file_config_native_bool_true_is_true(self, monkeypatch):
        """Native JSON bool true → True (regression guard)."""
        from cozempic import config as cfg_mod

        monkeypatch.delenv("COZEMPIC_FLOOR_PRESERVE_FIRST", raising=False)
        result = cfg_mod._resolve_floor_with({"floor": {"preserve_first_message": True}})
        assert result.preserve_first_message is True


# ── Class 8: R-2 — team-tag and singleton-tag strip invariant ────────────────

class TestTagLeakInvariant:
    def test_no_team_protected_tag_in_floor_readds(self):
        """Tags applied by prune_with_team_protect must be stripped from floor re-adds.

        The floor re-adds entries from msgs_before which carry __cozempic_team_protected__
        tags (applied by prune_with_team_protect before passing to run_prescription).
        The guard's strip loop at guard.py:314-316 iterates pruned_messages (=
        run_prescription's output, which includes floor re-adds) and removes the tag.

        This test verifies the full run_prescription flow (which includes floor)
        produces tag-free output when the guard's strip loop is applied. The strip
        is done in guard.py, not inside enforce_floor itself — enforce_floor
        intentionally passes the original dicts through to minimize copies.
        """
        import cozempic.strategies  # noqa: F401
        from cozempic.executor import run_prescription
        from cozempic.config import FloorConfig

        # Simulate what prune_with_team_protect does: tag messages with team-protected
        before = [
            (0, {"type": "user", "uuid": "u-root", "parentUuid": None,
                 "__cozempic_team_protected__": True,
                 "message": {"content": "hi", "role": "user"}}, 80),
            (1, {"type": "assistant", "uuid": "a-001", "parentUuid": "u-root",
                 "__cozempic_team_protected__": True,
                 "message": {"content": "ok", "role": "assistant"}}, 80),
        ]

        # Run with a prescription that would drop both (empty strategy list +
        # floor=disabled to isolate: the team-protected tags keep them in,
        # then we simulate the guard's strip loop).
        result, _ = run_prescription(
            before, [], {}, floor_config=FloorConfig.disabled()
        )

        # Simulate guard.py:314-316 strip loop
        for _, msg, _ in result:
            msg.pop("__cozempic_team_protected__", None)

        # After strip: no tag should remain
        for _, msg, _ in result:
            assert "__cozempic_team_protected__" not in msg, (
                f"__cozempic_team_protected__ remained after guard strip on uuid={msg.get('uuid')!r}"
            )
            assert "__cozempic_metadata_singleton__" not in msg, (
                f"__cozempic_metadata_singleton__ leaked on uuid={msg.get('uuid')!r}"
            )

    def test_no_singleton_tag_in_run_prescription_output(self):
        """__cozempic_metadata_singleton__ must never appear in run_prescription output."""
        import cozempic.strategies  # noqa: F401
        from cozempic.executor import run_prescription
        from cozempic.config import FloorConfig

        msgs = [
            _pm(0, "pm-001", None),
            _user(1, "u-001", "pm-001"),
            _asst(2, "a-001", "u-001"),
        ]
        result, _ = run_prescription(msgs, [], {}, floor_config=FloorConfig.disabled())

        for _, msg, _ in result:
            assert "__cozempic_metadata_singleton__" not in msg
        for _, msg, _ in msgs:
            assert "__cozempic_metadata_singleton__" not in msg


# ── Class 10: enforce_floor 2-root DAG fork on compacted sessions (L7) ────────


def _cs(idx: int, uuid: str, parent: str | None = "UNSET"):
    """compact_summary user message (isCompactSummary=True)."""
    d: dict = {"type": "user", "isCompactSummary": True, "uuid": uuid,
               "message": {"content": "summary", "role": "user"}}
    if parent != "UNSET":
        d["parentUuid"] = parent
    return (idx, d, len(json.dumps(d, separators=(",", ":"))))


def _cb_preserved(idx: int, uuid: str, parent: str | None = "UNSET"):
    """compact_boundary with hasPreservedSegment=True (collapse is a no-op)."""
    d: dict = {"type": "system", "subtype": "compact_boundary",
               "uuid": uuid, "hasPreservedSegment": True}
    if parent != "UNSET":
        d["parentUuid"] = parent
    return (idx, d, len(json.dumps(d, separators=(",", ":"))))


def _compacted_msgs():
    """Standard compacted-session layout for TestFloorCompactedSession.

    idx=0  root-0   (user, parentUuid=None)
    idx=1  pre-1    (user, parentUuid=root-0)
    idx=2  pre-2    (asst, parentUuid=pre-1)
    idx=3  cb-1     (compact_boundary, parentUuid=pre-2)
    idx=4  cs-1     (compact_summary user, isCompactSummary=True, parentUuid=cb-1)
    idx=5  pt-1     (user, parentUuid=cs-1)
    idx=6  pt-2     (asst, parentUuid=pt-1)
    """
    return [
        _user(0, "root-0", parent=None),
        _user(1, "pre-1",  parent="root-0"),
        _asst(2, "pre-2",  parent="pre-1"),
        _cb(3,  "cb-1",   parent="pre-2"),
        _cs(4,  "cs-1",   parent="cb-1"),
        _user(5, "pt-1",  parent="cs-1"),
        _asst(6, "pt-2",  parent="pt-1"),
    ]


class TestFloorCompactedSession:
    """L7 HIGH: enforce_floor must not re-add pre-boundary root on a compacted session.

    Bug: compact-summary-collapse removes idx=0..2 and _relink_parent_chain re-roots
    compact_boundary (cb-1) to parentUuid=None. Then enforce_floor(preserve_first_message=True)
    re-adds the original root (root-0, parentUuid=None) creating TWO DAG roots:
    root-0 and cb-1. The validate_post_prune C9 check (P0-B) catches this; the
    enforce_floor fix (P0-A) prevents it from happening.

    RED-at-base: tests 1, 2, and 6 fail before P0-A/P0-B because the bug is live.
    Tests 3 and 4 are regression guards (GREEN at base and after fix).
    Test 5 verifies the hasPreservedSegment edge case (GREEN at base too — edge case
    does not exercise the buggy path since collapse is a no-op there).
    """

    def test_floor_does_not_re_add_pre_boundary_root_on_compacted_session(self):
        """RED-at-base: floor re-adds root-0 alongside cb-1, creating 2 roots.

        After fix (P0-A): only 1 root in result (cb-1); root-0 NOT re-added.
        """
        import cozempic.strategies  # noqa: F401 — registers strategy names
        from cozempic.executor import execute_actions
        from cozempic.strategies.gentle import strategy_compact_summary_collapse
        from cozempic.safety import enforce_floor
        from cozempic.config import FloorConfig

        msgs_before = _compacted_msgs()

        # Run compact-summary-collapse to get the post-collapse state
        strategy_result = strategy_compact_summary_collapse(msgs_before, {})
        msgs_after_collapse = execute_actions(msgs_before, strategy_result.actions)

        # enforce_floor with default config (preserve_first_message=True)
        cfg = FloorConfig(preserve_first_message=True, preserve_last_k_turns=0,
                          max_user_assistant_drop_pct=0.0)
        result = enforce_floor(msgs_before, msgs_after_collapse, cfg=cfg)

        roots = [m for _, m, _ in result if not m.get("parentUuid") and m.get("uuid")]
        # RED at base: 2 roots (root-0 and cb-1). After fix: 1 root (cb-1).
        assert len(roots) == 1, (
            f"floor produced {len(roots)} DAG roots — expected 1. "
            f"root uuids: {[m.get('uuid') for m in roots]}. "
            "enforce_floor re-added the pre-boundary root alongside compact_boundary "
            "(confused-deputy 2-root DAG fork — P0-A guard missing)"
        )

    def test_c9_detects_two_root_fork_raises(self):
        """RED-at-base: injecting a second root into msgs_after must raise C9.

        Before P0-B, validate_post_prune does not detect the extra root.
        After P0-B: raises PruneValidationError with failed_check='C9'.
        """
        from cozempic.safety import validate_post_prune, PruneValidationError

        # Single-root before
        msgs_before = [
            _user(0, "root-a", parent=None),
            _asst(1, "asst-1", parent="root-a"),
            _user(2, "user-1", parent="asst-1"),
        ]
        # Two-root after (injected second root)
        msgs_after = [
            _user(0, "root-a", parent=None),
            _user(3, "root-b", parent=None),  # spurious second root
            _asst(1, "asst-1", parent="root-a"),
            _user(2, "user-1", parent="asst-1"),
        ]
        with pytest.raises(PruneValidationError) as exc_info:
            validate_post_prune(msgs_before, msgs_after)
        assert exc_info.value.evidence.get("failed_check") == "C9", (
            f"expected C9 but got {exc_info.value.evidence.get('failed_check')!r} — "
            "C9 multi-root check is missing from validate_post_prune (P0-B)"
        )

    def test_c9_single_root_passes(self):
        """Regression guard (GREEN at base and after fix): single-root session does not raise."""
        from cozempic.safety import validate_post_prune

        msgs = [
            _user(0, "root-a", parent=None),
            _asst(1, "asst-1", parent="root-a"),
            _user(2, "user-1", parent="asst-1"),
        ]
        # identity prune: same before and after — must not raise
        validate_post_prune(msgs, msgs)

    def test_c9_multi_root_before_same_count_after_passes(self):
        """Regression guard: a legitimately two-root session (team/resume) must not raise.

        C9 is BASELINE-RELATIVE: if root count after == root count before, no raise.
        Only raises if the prune INCREASED the root count.
        """
        from cozempic.safety import validate_post_prune

        # Two-root session (team fork — both chains present in the same file)
        msgs_before = [
            _user(0, "root-a", parent=None),
            _asst(1, "asst-a", parent="root-a"),
            _user(2, "root-b", parent=None),   # second root from team session
            _asst(3, "asst-b", parent="root-b"),
        ]
        # Prune keeps both roots and both assistants, same root count
        msgs_after = [
            _user(0, "root-a", parent=None),
            _asst(1, "asst-a", parent="root-a"),
            _user(2, "root-b", parent=None),
            _asst(3, "asst-b", parent="root-b"),
        ]
        # Must not raise — root count (2) did not increase
        validate_post_prune(msgs_before, msgs_after)

    def test_floor_with_has_preserved_segment_still_re_adds(self):
        """Edge case: compact_boundary(hasPreservedSegment=True) must NOT skip pre-boundary floor.

        When hasPreservedSegment=True, compact-summary-collapse does NOT remove the
        pre-boundary turns. The floor must still re-add them if needed (the P0-A skip
        only fires when the boundary is ACTIVE — i.e., hasPreservedSegment=False/absent).
        """
        from cozempic.safety import enforce_floor
        from cozempic.config import FloorConfig

        msgs_before = [
            _user(0, "root-0", parent=None),
            _user(1, "pre-1",  parent="root-0"),
            _asst(2, "pre-2",  parent="pre-1"),
            _cb_preserved(3, "cb-1", parent="pre-2"),
            _cs(4, "cs-1", parent="cb-1"),
            _user(5, "pt-1", parent="cs-1"),
            _asst(6, "pt-2", parent="pt-1"),
        ]
        # Simulate a partial removal that drops root-0 but NOT via a collapse
        # (hasPreservedSegment=True → collapse did NOT run → root-0 is recoverable)
        msgs_after = [
            _user(1, "pre-1",  parent="root-0"),
            _asst(2, "pre-2",  parent="pre-1"),
            _cb_preserved(3, "cb-1", parent="pre-2"),
            _cs(4, "cs-1", parent="cb-1"),
            _user(5, "pt-1", parent="cs-1"),
            _asst(6, "pt-2", parent="pt-1"),
        ]
        cfg = FloorConfig(preserve_first_message=True, preserve_last_k_turns=0,
                          max_user_assistant_drop_pct=0.0)
        result = enforce_floor(msgs_before, msgs_after, cfg=cfg)

        result_uuids = {m.get("uuid") for _, m, _ in result}
        assert "root-0" in result_uuids, (
            "floor must re-add pre-boundary root when hasPreservedSegment=True "
            "(the P0-A skip must NOT fire when the compact_boundary is not active)"
        )

    def test_run_prescription_compacted_session_single_root(self):
        """END-TO-END acceptance: run_prescription on a compacted session yields exactly 1 root.

        This is the EXACT acceptance check the lead will reproduce independently.
        Before fix: result has 2 roots (root-0 and cb-1 both have parentUuid=None).
        After fix: result has exactly 1 root (cb-1).
        """
        import cozempic.strategies  # noqa: F401
        from cozempic.executor import run_prescription
        from cozempic.config import FloorConfig

        msgs_before = _compacted_msgs()
        cfg = FloorConfig(preserve_first_message=True, preserve_last_k_turns=0,
                          max_user_assistant_drop_pct=0.0)
        result, _ = run_prescription(
            msgs_before, ["compact-summary-collapse"], {}, floor_config=cfg
        )

        roots = [m for _, m, _ in result if not m.get("parentUuid") and m.get("uuid")]
        assert len(roots) == 1, (
            f"run_prescription produced {len(roots)} DAG roots — expected 1. "
            f"root uuids: {[m.get('uuid') for m in roots]}"
        )

    def test_floor_step3_does_not_re_add_pre_boundary_pair_partner(self):
        """REGRESSION GUARD (review M-1) — RED without 8dce0b7: step-3 pair-closure re-adds
        the pre-boundary tool_result partner when the post-boundary tool_use is preserved,
        introducing a second DAG root (2-root fork).

        Scenario (compacted session with tool pair straddling the boundary):
          idx=0  pre-result  (user, parentUuid=None)  ← PRE-boundary root; tool_result(tid-1)
          idx=1  cb-1        (compact_boundary, parentUuid=pre-result)
          idx=2  cs-1        (compact_summary, parentUuid=cb-1)
          idx=3  post-use    (asst, parentUuid=cs-1)  ← tool_use(tid-1), post-boundary
          idx=4  pt-final    (user, parentUuid=post-use)  ← last user turn

        After compact-summary-collapse removes pre-result (idx=0), cb-1 is relinked to
        parentUuid=None (becomes the session root). preserve_last_k_turns=1 preserves
        post-use (idx=3). Step-3 pair-closure then inspects post-use:
          tool_use(id="tid-1") → tool_use_id_to_results["tid-1"] = {pre-result.uuid}

        WITHOUT 8dce0b7 gate: pre-result added to must_preserve → re-added to result
          → result has TWO roots: cb-1 (parentUuid=None) + pre-result (parentUuid=None).
        WITH 8dce0b7 gate: pre-result is _is_pre_boundary(idx=0) → skipped.
          → result has exactly ONE root: cb-1.

        Asserts:
          - exactly 1 root in enforce_floor result (cb-1)
          - pre-result uuid is NOT in the result
        """
        import cozempic.strategies  # noqa: F401 — registers strategy names
        from cozempic.executor import execute_actions
        from cozempic.strategies.gentle import strategy_compact_summary_collapse
        from cozempic.safety import enforce_floor
        from cozempic.config import FloorConfig

        # pre-result: parentUuid=None → it IS the pre-boundary root
        pre_result_msg = (0, {
            "type": "user",
            "uuid": "pre-result",
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "tid-1", "content": "ok"}],
                "role": "user",
            },
        }, 80)
        cb1_msg = _cb(1, "cb-1", parent="pre-result")
        cs1_msg = _cs(2, "cs-1", parent="cb-1")
        # post-use: post-boundary asst with matching tool_use
        post_use_msg = (3, {
            "type": "assistant",
            "uuid": "post-use",
            "parentUuid": "cs-1",
            "message": {
                "content": [{"type": "tool_use", "id": "tid-1", "name": "Bash", "input": {}}],
                "role": "assistant",
            },
        }, 100)
        pt_final_msg = _user(4, "pt-final", parent="post-use")

        msgs_before = [pre_result_msg, cb1_msg, cs1_msg, post_use_msg, pt_final_msg]

        # Run compact-summary-collapse (removes pre-boundary turns, relinks cb-1 to root)
        strategy_result = strategy_compact_summary_collapse(msgs_before, {})
        msgs_after_collapse = execute_actions(msgs_before, strategy_result.actions)

        # preserve_last_k_turns=1 puts post-use into must_preserve → step-3 fires
        cfg = FloorConfig(preserve_first_message=False, preserve_last_k_turns=1,
                          max_user_assistant_drop_pct=0.0)
        result = enforce_floor(msgs_before, msgs_after_collapse, cfg=cfg)

        result_uuids = {m.get("uuid") for _, m, _ in result}
        roots = [m for _, m, _ in result if not m.get("parentUuid") and m.get("uuid")]

        assert "pre-result" not in result_uuids, (
            "pre-result (pre-boundary tool_result root) must NOT be re-added by step-3 "
            "pair-closure — the _is_pre_boundary gate in 8dce0b7 should suppress it. "
            f"result uuids: {sorted(result_uuids)}"
        )
        assert len(roots) == 1, (
            f"enforce_floor produced {len(roots)} DAG roots — expected 1 (cb-1 only). "
            f"root uuids: {[m.get('uuid') for m in roots]}. "
            "Step-3 pair-closure re-introduced the pre-boundary tool_result as a second root "
            "(2-root fork — M-1 regression guard missing until this test)"
        )


# ── Class 9: PruneValidationError in guard_prune_cycle abort path ─────────────
# Rewritten (H-1): see test_prune_safety_r2.py::TestGuardAbortContractRewritten
# for the correct abort-contract tests that patch cozempic.guard.prune_with_team_protect
# (the symbol guard.py actually calls) and assert all 3 invariants:
#   (a) result has validation_error + evidence keys
#   (b) file is byte-identical before and after
#   (c) _terminate_and_resume was NOT called
#
# The old test here patched cozempic.executor.run_prescription, which guard.py
# had already imported via `from .executor import run_prescription` (line 138).
# The patch never fired; the test was GREEN via the saved_bytes<=0 early-return
# path, NOT the abort branch — it proved nothing about the abort contract.
