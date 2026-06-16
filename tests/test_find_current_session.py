"""Tests for find_current_session strict mode and sidecar session store."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from cozempic.session import find_current_session


def _write_session(proj_dir: Path, session_id: str, content: str = "") -> Path:
    proj_dir.mkdir(parents=True, exist_ok=True)
    p = proj_dir / f"{session_id}.jsonl"
    p.write_text(content or json.dumps({"message": {"role": "user", "content": "hi"}}) + "\n",
                 encoding="utf-8")
    return p


@contextmanager
def _isolated_session_home(projects_dir: Path, process_session_id: str | None = None):
    """Isolate find_current_session from the developer's real ~/.claude state.

    Patches three seams that would otherwise let a live Claude session bleed in:
    - get_projects_dir  → tmp projects dir (isolates Strategy 2/4/5)
    - _session_id_from_process → controllable (isolates Strategy 2)
    - find_claude_pid   → None (isolates Strategy 1: lookup_active_transcript)
    """
    with (
        patch("cozempic.session.get_projects_dir", return_value=projects_dir),
        patch("cozempic.session._session_id_from_process", return_value=process_session_id),
        patch("cozempic.session.find_claude_pid", return_value=None),
    ):
        yield


class TestStrictMode:
    def test_strict_returns_none_when_only_fallback_available(self, tmp_path):
        """With no process or CWD match, strict=True returns None instead of guessing."""
        proj = tmp_path / "projects" / "-some-other-path"
        _write_session(proj, "aaaa1111-0000-0000-0000-000000000000")

        with _isolated_session_home(tmp_path / "projects"):
            result = find_current_session(cwd="/unrelated/path", strict=True)

        assert result is None

    def test_non_strict_returns_fallback(self, tmp_path):
        """With no process or CWD match, strict=False still returns most recent session."""
        proj = tmp_path / "projects" / "-some-other-path"
        _write_session(proj, "aaaa1111-0000-0000-0000-000000000000")

        with _isolated_session_home(tmp_path / "projects"):
            result = find_current_session(cwd="/unrelated/path", strict=False)

        assert result is not None
        assert result["session_id"] == "aaaa1111-0000-0000-0000-000000000000"

    def test_strict_succeeds_when_process_detected(self, tmp_path):
        """Process-based detection (Strategy 2) satisfies strict mode."""
        session_id = "bbbb2222-0000-0000-0000-000000000000"
        proj = tmp_path / "projects" / "-some-path"
        _write_session(proj, session_id)

        # Pass session_id as process_session_id so Strategy 2 fires; Strategy 1 stays off.
        with _isolated_session_home(tmp_path / "projects", process_session_id=session_id):
            result = find_current_session(strict=True)

        assert result is not None
        assert result["session_id"] == session_id

    def test_strict_succeeds_on_cwd_slug_match(self, tmp_path):
        """CWD slug match (Strategy 4) satisfies strict mode — underscore and dot variants.

        Fixture dirs use HARDCODED literal names (what Claude Code actually writes to disk),
        independent of cwd_to_project_slug. If the slug formula regresses, the computed
        slug won't match the literal dir → strict=True returns None → test FAILS.
        """
        # Underscore cwd (was the original bug trigger).
        # Literal dir: '/Users/foo/topstep_automation' → '-Users-foo-topstep-automation'
        cwd_under = "/Users/foo/topstep_automation"
        literal_dir_under = "-Users-foo-topstep-automation"
        proj_under = tmp_path / "projects" / literal_dir_under
        sess_under = "cccc3333-0000-0000-0000-000000000000"
        _write_session(proj_under, sess_under)

        with _isolated_session_home(tmp_path / "projects"):
            result = find_current_session(cwd=cwd_under, strict=True)

        assert result is not None, (
            f"Strategy 4 did not find underscore-cwd project. "
            f"Expected slug '-Users-foo-topstep-automation' to match literal dir."
        )
        assert result["session_id"] == sess_under

    def test_strict_succeeds_on_dot_cwd_slug_match(self, tmp_path):
        """Dot-path cwd (double-dash slug) is found by Strategy 4 in strict mode.

        Fixture dir uses HARDCODED literal name.
        '/Users/foo/.claude' → '-Users-foo--claude' (dot→dash produces double-dash).
        """
        cwd_dot = "/Users/foo/.claude"
        literal_dir_dot = "-Users-foo--claude"   # double-dash from dot replacement
        proj_dot = tmp_path / "projects" / literal_dir_dot
        sess_dot = "eeee5555-0000-0000-0000-000000000000"
        _write_session(proj_dot, sess_dot)

        with _isolated_session_home(tmp_path / "projects"):
            result = find_current_session(cwd=cwd_dot, strict=True)

        assert result is not None, (
            f"Strategy 4 did not find dot-path project. "
            f"Expected slug '-Users-foo--claude' to match literal dir."
        )
        assert result["session_id"] == sess_dot

    def test_no_sessions_returns_none_regardless_of_strict(self, tmp_path):
        projects = tmp_path / "projects"
        projects.mkdir()

        with _isolated_session_home(projects):
            assert find_current_session(strict=True) is None
            assert find_current_session(strict=False) is None

    def test_trailing_slash_cwd_resolves_project(self, tmp_path):
        """cwd with trailing slash must find the project after normpath strips it.

        Without normpath: slug('-Users-x-proj-') ≠ dir('-Users-x-proj') → None.
        With normpath: '/Users/x/proj/' → '/Users/x/proj' → slug='-Users-x-proj' → found.

        RED-at-base (815485d): no normpath → trailing dash mismatch → strict=True → None.
        """
        literal_dir = "-Users-x-proj"
        proj = tmp_path / "projects" / literal_dir
        session_id = "ffff6666-0000-0000-0000-000000000000"
        _write_session(proj, session_id)

        with _isolated_session_home(tmp_path / "projects"):
            result = find_current_session(cwd="/Users/x/proj/", strict=True)

        assert result is not None, (
            "find_current_session(cwd='/Users/x/proj/', strict=True) returned None. "
            "normpath is not stripping the trailing slash before slug computation."
        )
        assert result["session_id"] == session_id


# ---------------------------------------------------------------------------
# TestStrategy4ExactMatch — Bug B: substring → exact-match in Strategy 4
# ---------------------------------------------------------------------------

class TestStrategy4ExactMatch:
    """Strategy 4 (CWD slug) must use exact-match on slug, not substring."""

    def test_underscore_project_found_after_slug_fix(self, tmp_path):
        """Strategy 4 must find an underscore-path project via its exact slug.

        Fixture uses the LITERAL dir name '-Users-x-topstep-automation' (not derived
        from cwd_to_project_slug) so the test is independent of the function under test.
        strict=True disables Strategy 5 — if Strategy 4 misses, returns None (RED).

        RED-at-base (815485d): broken slug '-Users-x-topstep_automation' (keeps '_')
        ≠ literal dir '-Users-x-topstep-automation' → Strategy 4 finds 0 matches →
        strict=True → None → assertIsNotNone FAILS.
        """
        cwd = "/Users/x/topstep_automation"
        # Hardcoded literal — the dir name Claude Code actually creates on disk.
        # NOT derived from cwd_to_project_slug() so this test is not tautological.
        literal_dir = "-Users-x-topstep-automation"
        proj = tmp_path / "projects" / literal_dir
        session_id = "dddd4444-0000-0000-0000-000000000000"
        _write_session(proj, session_id)

        with _isolated_session_home(tmp_path / "projects"):
            result = find_current_session(cwd=cwd, strict=True)

        assert result is not None, (
            "find_current_session(strict=True) returned None for underscore cwd. "
            "The slug normalization is not replacing '_' with '-'. "
            "Expected: computed slug '-Users-x-topstep-automation' matches literal dir."
        )
        assert result["session_id"] == session_id

    def test_strategy3_no_prefix_collision(self, tmp_path):
        """Slug '-Users-x-foo' must NOT match project '-Users-x-foobar'."""
        from cozempic.session import cwd_to_project_slug
        cwd_foo = "/Users/x/foo"
        slug_foo = cwd_to_project_slug(cwd_foo)    # "-Users-x-foo"

        # Create BOTH project dirs
        proj_foo = tmp_path / "projects" / slug_foo
        proj_foobar = tmp_path / "projects" / f"{slug_foo}bar"
        sess_foo = "eeee5555-0000-0000-0000-000000000000"
        sess_foobar = "ffff6666-0000-0000-0000-000000000000"
        _write_session(proj_foo, sess_foo)
        _write_session(proj_foobar, sess_foobar)

        with _isolated_session_home(tmp_path / "projects"):
            result = find_current_session(cwd=cwd_foo, strict=True)

        assert result is not None
        assert result["session_id"] == sess_foo, (
            f"Expected session for '-Users-x-foo', got {result['session_id']!r}. "
            "Prefix collision: Strategy 4 is still using substring match."
        )

    def test_strategy4_strict_returns_none_on_no_match(self, tmp_path):
        """strict=True blocks Strategy 5 fallback when no slug matches."""
        proj = tmp_path / "projects" / "-some-other-path"
        _write_session(proj, "aaaa1111-0000-0000-0000-000000000000")

        with _isolated_session_home(tmp_path / "projects"):
            result = find_current_session(cwd="/unrelated/underscore_path", strict=True)

        assert result is None, (
            "strict=True must return None when no slug matches. "
            "Strategy 5 fallback is leaking through."
        )


class TestSlugRoundTrip:
    def test_simple_path_round_trips(self):
        from cozempic.session import cwd_to_project_slug, project_slug_to_path
        path = "/Users/foo/myproject"
        slug = cwd_to_project_slug(path)
        assert project_slug_to_path(slug) == path

    def test_hyphenated_path_slug_is_ambiguous(self):
        """Slug reversal is known-ambiguous for hyphenated paths.

        '/Users/foo/my-project' and '/Users/foo/my/project' produce the same
        slug.  This is intentionally not fixed in project_slug_to_path() —
        callers that need an exact path must use get_session_cwd() (sidecar).
        """
        from cozempic.session import cwd_to_project_slug, project_slug_to_path
        slug_a = cwd_to_project_slug("/Users/foo/my-project")
        slug_b = cwd_to_project_slug("/Users/foo/my/project")
        # The ambiguity: both slugs are identical
        assert slug_a == slug_b
        # And reversal cannot recover the original — this is expected behaviour
        assert project_slug_to_path(slug_a) != "/Users/foo/my-project"


class TestSidecarStore:
    def test_record_and_retrieve(self, tmp_path):
        """record_session persists cwd; get_session_cwd retrieves it."""
        from cozempic.session import get_session_cwd, record_session
        sid = "aaaa1111-0000-0000-0000-000000000000"
        with patch("cozempic.session.get_sidecar_path", return_value=tmp_path / "sidecar.json"):
            record_session(sid, "/Users/foo/my-project", context_window=200_000)
            assert get_session_cwd(sid) == "/Users/foo/my-project"

    def test_hyphenated_path_survives_round_trip(self, tmp_path):
        """Sidecar stores exact cwd — hyphenated paths are not mangled."""
        from cozempic.session import get_session_cwd, record_session
        sid = "bbbb2222-0000-0000-0000-000000000000"
        path = "/Users/foo/my-hyphenated-project"
        with patch("cozempic.session.get_sidecar_path", return_value=tmp_path / "sidecar.json"):
            record_session(sid, path)
            assert get_session_cwd(sid) == path

    def test_context_window_persisted(self, tmp_path):
        from cozempic.session import get_session_context_window, record_session
        sid = "cccc3333-0000-0000-0000-000000000000"
        with patch("cozempic.session.get_sidecar_path", return_value=tmp_path / "sidecar.json"):
            record_session(sid, "/some/path", context_window=1_000_000)
            assert get_session_context_window(sid) == 1_000_000

    def test_context_window_preserved_on_refresh(self, tmp_path):
        """Refreshing last_seen_at without a context_window keeps the old value."""
        from cozempic.session import get_session_context_window, record_session
        sid = "dddd4444-0000-0000-0000-000000000000"
        with patch("cozempic.session.get_sidecar_path", return_value=tmp_path / "sidecar.json"):
            record_session(sid, "/some/path", context_window=200_000)
            record_session(sid, "/some/path")  # refresh without context_window
            assert get_session_context_window(sid) == 200_000

    def test_unknown_session_returns_none(self, tmp_path):
        from cozempic.session import get_session_cwd
        with patch("cozempic.session.get_sidecar_path", return_value=tmp_path / "sidecar.json"):
            assert get_session_cwd("no-such-session") is None

    def test_evicts_oldest_when_full(self, tmp_path):
        """Sidecar is capped at _SIDECAR_MAX_ENTRIES; oldest entry is evicted."""
        from datetime import datetime as real_datetime
        from cozempic.session import _SIDECAR_MAX_ENTRIES, get_session_cwd, record_session

        sidecar = tmp_path / "sidecar.json"
        call_count = [0]

        def mock_now():
            n = call_count[0]
            call_count[0] += 1
            return real_datetime(2026, 1, 1, n // 3600, (n % 3600) // 60, n % 60)

        with (
            patch("cozempic.session.get_sidecar_path", return_value=sidecar),
            patch("cozempic.session.datetime") as mock_dt,
        ):
            mock_dt.now.side_effect = mock_now
            # First entry gets the smallest timestamp (oldest)
            first_id = f"{'0' * 8}-0000-0000-0000-{'0' * 12}"
            record_session(first_id, "/first", context_window=200_000)
            for i in range(1, _SIDECAR_MAX_ENTRIES + 1):
                sid = f"{i:08x}-0000-0000-0000-{'0' * 12}"
                record_session(sid, f"/path/{i}")
            # Oldest entry should have been evicted
            assert get_session_cwd(first_id) is None

    def test_missing_sidecar_returns_none(self, tmp_path):
        from cozempic.session import get_session_cwd
        with patch("cozempic.session.get_sidecar_path", return_value=tmp_path / "nonexistent.json"):
            assert get_session_cwd("any-session") is None

    def test_corrupt_sidecar_returns_none(self, tmp_path):
        from cozempic.session import get_session_cwd
        sidecar = tmp_path / "sidecar.json"
        sidecar.write_text("not valid json", encoding="utf-8")
        with patch("cozempic.session.get_sidecar_path", return_value=sidecar):
            assert get_session_cwd("any-session") is None
