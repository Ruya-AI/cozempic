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
    remove_synced_memory,
    save_digest_store,
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

    def test_unrecognized_nonempty_fails_closed(self):
        """Ambiguous opt-out intent disables — same fail-safe as receipts."""
        for value in ("disabled", "ture", "2", "please"):
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

    def test_save_digest_store_gated(self):
        """The store choke point must not write (covers migration-on-read)."""
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(
                os.environ, {"HOME": home, "COZEMPIC_NO_DIGEST": "1"}
            ):
                save_digest_store(_store_with_active_rule())
            self.assertFalse((Path(home) / ".cozempic").exists())

    def test_disabling_removes_previously_synced_memory(self):
        """Opt-out after a sync must delete the memory file and index line."""
        with tempfile.TemporaryDirectory() as home:
            memdir = Path(home) / ".claude" / "projects" / "-tmp" / "memory"
            memdir.mkdir(parents=True)
            (memdir / "cozempic_digest.md").write_text("stale rules")
            (memdir / "MEMORY.md").write_text(
                "# Memory\n\n- [Other](other.md) — keep me\n"
                "- [Cozempic Behavioral Digest](cozempic_digest.md) — rules\n"
                "- user note: why cozempic_digest.md was disabled — keep me too\n"
            )
            with patch.dict(
                os.environ, {"HOME": home, "COZEMPIC_NO_DIGEST": "1"}
            ):
                with patch(
                    "cozempic.digest._get_memdir", return_value=memdir
                ):
                    self.assertEqual(
                        sync_to_memdir(_store_with_active_rule(), cwd="/tmp"), 0
                    )
            self.assertFalse((memdir / "cozempic_digest.md").exists())
            index = (memdir / "MEMORY.md").read_text()
            self.assertNotIn(
                "[Cozempic Behavioral Digest](cozempic_digest.md)", index
            )
            self.assertIn("keep me", index)
            # User-authored lines survive even when they mention the filename.
            self.assertIn("keep me too", index)

    def test_remove_synced_memory_noop_when_nothing_synced(self):
        with tempfile.TemporaryDirectory() as home:
            memdir = Path(home) / ".claude" / "projects" / "-tmp" / "memory"
            memdir.mkdir(parents=True)
            with patch("cozempic.digest._get_memdir", return_value=memdir):
                self.assertFalse(remove_synced_memory("/tmp"))

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
