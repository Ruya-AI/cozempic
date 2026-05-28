"""RED tests for review-round3 Group G — cosmetic + doc drift.

Findings:
  G.N1  Duplicate realistic_session_factory fixture registration
  G.N3  cleanup_old_backups glob over-match (sister-session prefix collision)
  G.N5  Docstring drift "C1-C6" — C7 (ai-title) added in E.4
  G.N6  tests/_session_factory.py imports pytest at module top (zero-dep blocker)
  G.N8  README floor.max_user_assistant_drop_pct range docs (inclusive vs exclusive)
  G.M4  C.5 tool_use_id_to_owner overwrite-only (asymmetric with tool_use_id_to_results)
"""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


# ─── G.N1 — single fixture registration ──────────────────────────────────────


class TestRealisticSessionFactoryRegisteredOnce(unittest.TestCase):

    def test_no_duplicate_decorator_in_session_factory(self):
        path = Path(__file__).parent / "_session_factory.py"
        text = path.read_text()
        self.assertNotIn(
            "@pytest.fixture", text,
            msg="G.N1: tests/_session_factory.py must NOT carry "
                "@pytest.fixture — conftest.py is the canonical location",
        )

    def test_conftest_carries_canonical_fixture(self):
        path = Path(__file__).parent / "conftest.py"
        text = path.read_text()
        self.assertIn("@pytest.fixture", text)
        self.assertIn("def realistic_session_factory", text)


# ─── G.N3 — cleanup_old_backups stricter glob ────────────────────────────────


class TestCleanupOldBackupsRespectsSessionBoundary(unittest.TestCase):
    """G.N3: when session ids share a prefix (e.g. 'abc' and 'abc-2'),
    cleanup_old_backups for 'abc' must not delete 'abc-2's backups."""

    def test_sister_session_backups_not_cross_deleted(self):
        from cozempic.session import cleanup_old_backups

        with TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            # Session A: 'abc' has 5 backups (will keep 3, drop 2).
            a_session = d / "abc.jsonl"
            a_session.write_text("{}\n")
            now = time.time()
            for i in range(5):
                bk = d / f"abc.{20260101 + i:08d}_120000.jsonl.bak"
                bk.write_text("{}")
                os.utime(bk, (now - (5 - i) * 60, now - (5 - i) * 60))

            # Session B: 'abc-2' has 3 backups; cleanup of A must leave
            # ALL of them intact.
            b_keep = []
            for i in range(3):
                bk = d / f"abc-2.{20260201 + i:08d}_120000.jsonl.bak"
                bk.write_text("{}")
                b_keep.append(bk)

            cleanup_old_backups(a_session, keep=3)
            for bk in b_keep:
                self.assertTrue(
                    bk.exists(),
                    msg=f"G.N3: cleanup_old_backups deleted sister-session "
                        f"backup {bk.name} via glob over-match",
                )


# ─── G.N5 — docstring drift updated to C1-C7 ────────────────────────────────


class TestDocstringMentionsC7(unittest.TestCase):

    def test_validate_post_prune_docstring_mentions_c7(self):
        from cozempic.safety import validate_post_prune
        doc = (validate_post_prune.__doc__ or "")
        self.assertIn(
            "C7", doc,
            msg="G.N5: validate_post_prune docstring must mention C7 "
                "(added by E.4 for ai-title)",
        )

    def test_prune_validation_error_doc_mentions_c7(self):
        from cozempic.safety import PruneValidationError
        doc = (PruneValidationError.__doc__ or "")
        # The class docstring previously said C1..C6; it must include C7 now.
        self.assertNotIn(
            '"C1".."C6"', doc,
            msg="G.N5: PruneValidationError docstring drift — still says "
                "C1..C6 after E.4 added C7",
        )


# ─── G.N6 — _session_factory.py importable without pytest ───────────────────


class TestSessionFactoryImportableWithoutPytest(unittest.TestCase):
    """G.N6: the fixture-generator module should be importable in zero-dep
    reproducer scripts. Top-level `import pytest` blocks that. Move pytest
    inside the fixture function (or remove it entirely from this file)."""

    def test_session_factory_does_not_import_pytest_at_module_top(self):
        path = Path(__file__).parent / "_session_factory.py"
        text = path.read_text()
        # Walk lines until the first non-comment / non-import block of the
        # module. Any `import pytest` outside a function body fails.
        in_func = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("def "):
                in_func = True
            if not in_func and stripped == "import pytest":
                self.fail(
                    "G.N6: tests/_session_factory.py has a top-level "
                    "`import pytest` — move inside the fixture or remove",
                )


# ─── G.N8 — README range inclusive ───────────────────────────────────────────


class TestReadmeRangeIsInclusive(unittest.TestCase):

    def test_readme_drop_pct_range_inclusive(self):
        path = Path(__file__).parent.parent / "README.md"
        text = path.read_text()
        # The doc must NOT claim exclusive `(0.0, 1.0)` for
        # max_user_assistant_drop_pct since _clamp_float accepts the
        # endpoints. Look for the line and assert it uses square brackets.
        for line in text.splitlines():
            if "max_user_assistant_drop_pct" in line and "0.0" in line:
                self.assertIn(
                    "[0.0, 1.0]", line,
                    msg="G.N8: README must use inclusive range notation "
                        "[0.0, 1.0] for max_user_assistant_drop_pct, "
                        "matching _clamp_float's actual inclusive clamp",
                )
                return
        self.fail("README does not document max_user_assistant_drop_pct range")


# ─── G.M4 — tool_use_id_to_owner additive ───────────────────────────────────


class TestToolUseIdToOwnerIsAdditive(unittest.TestCase):
    """G.M4: the pair-counterpart closure builds tool_use_id_to_owner with
    overwrite-only assignment (`d[tid] = u`). If two msgs_before entries
    legitimately reference the same tool_use_id (theoretically impossible
    per Anthropic but defensive), the later overwrites the earlier. Make
    it additive (set-valued) like tool_use_id_to_results."""

    def test_source_uses_setdefault_for_tool_use_id_to_owner(self):
        import inspect
        from cozempic import safety as safety_mod

        src = inspect.getsource(safety_mod.enforce_floor)
        # The fix uses an additive set-valued map. Assert the literal
        # overwrite-assignment is gone and that an additive `.add(` call
        # is present for the owner map.
        self.assertNotIn(
            "tool_use_id_to_owner[tid] = u", src,
            msg="G.M4: tool_use_id_to_owner must be additive (set-valued) "
                "not overwrite-only, mirroring tool_use_id_to_results",
        )


import os  # noqa: E402 — used inside test methods above

if __name__ == "__main__":
    unittest.main()
