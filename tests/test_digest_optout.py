"""COZEMPIC_NO_DIGEST opt-out: every digest write path must no-op when set."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cozempic.digest import (
    DigestRule,
    DigestStore,
    digest_enabled,
    flush_digest,
    recover_digest,
    sync_to_memdir,
    update_digest,
)


def _store_with_active_rule() -> DigestStore:
    rule = DigestRule(
        id="R001",
        rule="Do not force-push to main",
        priority="hard",
        status="active",
        occurrence_count=2,
        evidence="never force-push to main",
    )
    return DigestStore(strategy_rules=[rule])


class TestDigestEnabledTruthTable(unittest.TestCase):
    def test_unset_means_enabled(self):
        with patch.dict(os.environ):
            os.environ.pop("COZEMPIC_NO_DIGEST", None)
            self.assertTrue(digest_enabled())

    def test_empty_and_falsy_mean_enabled(self):
        for value in ("", "  ", "0", "false", "no", "off"):
            with patch.dict(os.environ, {"COZEMPIC_NO_DIGEST": value}):
                self.assertTrue(digest_enabled(), f"value={value!r}")

    def test_truthy_means_disabled(self):
        for value in ("1", "true", "yes", "on", "TRUE"):
            with patch.dict(os.environ, {"COZEMPIC_NO_DIGEST": value}):
                self.assertFalse(digest_enabled(), f"value={value!r}")


class TestWritePathsGated(unittest.TestCase):
    def test_update_and_flush_return_zeros(self):
        with patch.dict(os.environ, {"COZEMPIC_NO_DIGEST": "1"}):
            self.assertEqual(update_digest([], project_dir="/tmp"), (0, 0, 0))
            self.assertEqual(flush_digest([], project_dir="/tmp"), (0, 0, 0))

    def test_recover_returns_zero(self):
        with patch.dict(os.environ, {"COZEMPIC_NO_DIGEST": "1"}):
            self.assertEqual(recover_digest(project_dir="/tmp"), 0)

    def test_sync_to_memdir_writes_nothing_when_disabled(self):
        store = _store_with_active_rule()
        with tempfile.TemporaryDirectory() as home:
            memdir = Path(home) / ".claude" / "projects" / "-tmp" / "memory"
            memdir.mkdir(parents=True)
            with patch.dict(
                os.environ, {"HOME": home, "COZEMPIC_NO_DIGEST": "1"}
            ):
                with patch(
                    "cozempic.digest._get_memdir", return_value=memdir
                ):
                    self.assertEqual(sync_to_memdir(store, cwd="/tmp"), 0)
            self.assertEqual(list(memdir.iterdir()), [])

    def test_sync_to_memdir_writes_when_enabled(self):
        """Control case: same setup with the gate off must produce the file."""
        store = _store_with_active_rule()
        with tempfile.TemporaryDirectory() as home:
            memdir = Path(home) / ".claude" / "projects" / "-tmp" / "memory"
            memdir.mkdir(parents=True)
            with patch.dict(os.environ, {"HOME": home}):
                os.environ.pop("COZEMPIC_NO_DIGEST", None)
                with patch(
                    "cozempic.digest._get_memdir", return_value=memdir
                ):
                    synced = sync_to_memdir(store, cwd="/tmp")
            self.assertEqual(synced, 1)
            self.assertTrue((memdir / "cozempic_digest.md").exists())


if __name__ == "__main__":
    unittest.main()
