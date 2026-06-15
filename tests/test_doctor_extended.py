"""Extended doctor tests covering previously untested paths."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cozempic.doctor import (
    check_stale_backups,
    fix_stale_backups,
    run_doctor,
)


class TestStaleBackupsScope(unittest.TestCase):
    """fix_stale_backups must only touch *.jsonl.bak, not arbitrary *.bak files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.projects_dir = Path(self.tmpdir) / "projects" / "my-proj"
        self.projects_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _patch_claude_dir(self):
        return patch("cozempic.doctor.get_claude_dir", return_value=Path(self.tmpdir))

    def test_fix_only_deletes_jsonl_bak(self):
        """Only *.jsonl.bak files are deleted; other .bak files survive."""
        cozempic_bak = self.projects_dir / "abc123.20240101_120000.jsonl.bak"
        other_bak = self.projects_dir / "something_else.bak"
        cozempic_bak.write_text("cozempic backup")
        other_bak.write_text("not a cozempic backup")

        with self._patch_claude_dir():
            result = fix_stale_backups()

        assert not cozempic_bak.exists(), "cozempic *.jsonl.bak should be deleted"
        assert other_bak.exists(), "non-cozempic *.bak should survive"
        assert "1" in result  # deleted 1 file

    def test_check_only_counts_jsonl_bak(self):
        """check_stale_backups only counts *.jsonl.bak, not arbitrary *.bak files."""
        (self.projects_dir / "foreign.bak").write_text("x" * (200 * 1024 * 1024))  # 200MB
        (self.projects_dir / "cozempic.20240101.jsonl.bak").write_text("small")

        with self._patch_claude_dir():
            result = check_stale_backups()

        # Should only count the jsonl.bak (small), not the large foreign.bak
        assert "1 backup" in result.message
        assert result.status == "ok"  # small size — no warning

    def test_fix_no_op_when_no_jsonl_bak(self):
        """Reports nothing to clean when only non-cozempic .bak files exist."""
        (self.projects_dir / "foreign.bak").write_text("data")

        with self._patch_claude_dir():
            result = fix_stale_backups()

        assert "No backup" in result

    def test_fix_no_op_when_empty(self):
        with self._patch_claude_dir():
            result = fix_stale_backups()
        assert "No backup" in result


class TestRunDoctorFixFalse(unittest.TestCase):
    """run_doctor(fix=False) must never invoke any fix_fn."""

    def test_fix_fn_not_called_when_fix_false(self):
        """Even with fixable issues present, fix_fn should not run in audit mode."""
        called = []

        def fake_check():
            from cozempic.doctor import CheckResult
            return CheckResult(
                name="fake",
                status="issue",
                message="something is broken",
                fix_description="run fix_fake()",
            )

        def fake_fix():
            called.append(True)
            return "fixed!"

        from cozempic.doctor import ALL_CHECKS
        original = list(ALL_CHECKS)
        ALL_CHECKS.clear()
        ALL_CHECKS.append(("fake", fake_check, fake_fix))

        try:
            results = run_doctor(fix=False)
        finally:
            ALL_CHECKS.clear()
            ALL_CHECKS.extend(original)

        assert not called, "fix_fn was called despite fix=False"
        assert results[0].status == "issue"

    def test_fix_fn_called_when_fix_true(self):
        """Sanity check: fix_fn IS called when fix=True."""
        called = []

        def fake_check():
            from cozempic.doctor import CheckResult
            return CheckResult(
                name="fake",
                status="issue",
                message="broken",
                fix_description="do fix",
            )

        def fake_fix():
            called.append(True)
            return "done"

        from cozempic.doctor import ALL_CHECKS
        original = list(ALL_CHECKS)
        ALL_CHECKS.clear()
        ALL_CHECKS.append(("fake", fake_check, fake_fix))

        try:
            run_doctor(fix=True)
        finally:
            ALL_CHECKS.clear()
            ALL_CHECKS.extend(original)

        assert called, "fix_fn was NOT called despite fix=True"


class TestRunDoctorHonestFixedStatus(unittest.TestCase):
    """run_doctor must only report 'fixed' when the issue is actually gone
    (re-run the check), not unconditionally after calling fix_fn (audit P1)."""

    def _run_with(self, check_fn, fix_fn):
        from cozempic.doctor import ALL_CHECKS
        original = list(ALL_CHECKS)
        ALL_CHECKS.clear()
        ALL_CHECKS.append(("fake", check_fn, fix_fn))
        try:
            return run_doctor(fix=True)
        finally:
            ALL_CHECKS.clear()
            ALL_CHECKS.extend(original)

    def test_noop_fix_does_not_report_fixed(self):
        from cozempic.doctor import CheckResult
        # Check ALWAYS returns "issue" (the fix was a no-op / couldn't resolve it).
        def check():
            return CheckResult(name="fake", status="issue", message="still broken",
                               fix_description="try fix")
        def fix():
            return "Skipped 2 sessions (nothing changed)"
        results = self._run_with(check, fix)
        self.assertNotEqual(results[0].status, "fixed",
                            "a no-op/failed fix must NOT report 'fixed'")
        self.assertIn("not fully resolved", results[0].message)

    def test_real_fix_reports_fixed(self):
        from cozempic.doctor import CheckResult
        state = {"broken": True}
        def check():
            return CheckResult(name="fake",
                               status="issue" if state["broken"] else "ok",
                               message="broken" if state["broken"] else "clean",
                               fix_description="do fix")
        def fix():
            state["broken"] = False  # actually resolves the issue
            return "repaired 3 blocks"
        results = self._run_with(check, fix)
        self.assertEqual(results[0].status, "fixed",
                         "a fix that resolves the issue must report 'fixed'")
        self.assertIn("repaired 3 blocks", results[0].message)
