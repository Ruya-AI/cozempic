"""#158: bake an absolute cozempic interpreter into the hook fallback so the
guard daemon resolves even when bare `cozempic` isn't on the hook's PATH."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cozempic import init as cz


class TestResolveAndBake(unittest.TestCase):
    def test_bake_replaces_python3_fallback_with_abs(self):
        hooks = {"SessionStart": [{"hooks": [
            {"type": "command",
             "command": "{ cozempic guard 2>/dev/null || python3 -m cozempic guard 2>/dev/null; } # cozempic-hook-schema=v13"},
        ]}]}
        out = cz._bake_cozempic_path(hooks, "/opt/uvtools/cozempic/bin/python3.12")
        cmd = out["SessionStart"][0]["hooks"][0]["command"]
        self.assertIn("/opt/uvtools/cozempic/bin/python3.12 -m cozempic guard", cmd)
        self.assertNotIn("|| python3 -m cozempic", cmd)   # bare python3 fallback gone
        self.assertIn("cozempic guard 2>/dev/null ||", cmd)  # on-PATH primary kept first
        self.assertIn("cozempic-hook-schema=v13", cmd)       # marker untouched

    def test_bake_does_not_mutate_input(self):
        hooks = {"E": [{"hooks": [{"command": "python3 -m cozempic x"}]}]}
        cz._bake_cozempic_path(hooks, "/abs/py")
        self.assertEqual(hooks["E"][0]["hooks"][0]["command"], "python3 -m cozempic x")  # original intact

    def test_bake_quotes_paths_with_spaces(self):
        out = cz._bake_cozempic_path({"E": [{"hooks": [{"command": "python3 -m cozempic x"}]}]},
                                     "/Users/a b/py")
        self.assertIn("'/Users/a b/py' -m cozempic x", out["E"][0]["hooks"][0]["command"])

    def test_bake_tolerates_malformed_entries(self):
        # non-dict entries / missing command must not crash
        cz._bake_cozempic_path({"E": [{"hooks": ["nope", {"x": 1}]}, "bad"]}, "/p")

    def test_resolve_uses_sys_executable_unmodified(self):
        # must NOT realpath — a uv-tool venv python must be returned as-is so its
        # site-packages (which has cozempic) is used by `-m cozempic`.
        p = "/Users/x/.local/share/uv/tools/cozempic/bin/python3.12"
        with patch("cozempic.init.sys.executable", p):
            got, _ = cz._resolve_cozempic_python()
            self.assertEqual(got, p)

    def test_resolve_flags_uvx_ephemeral(self):
        for p in ("/Users/x/.cache/uv/environments-v2/abc/bin/python3",
                  "/private/var/folders/xx/T/uvx-tmp/bin/python"):
            with patch("cozempic.init.sys.executable", p):
                _, eph = cz._resolve_cozempic_python()
                self.assertTrue(eph, f"{p} should be ephemeral")

    def test_resolve_flags_windows_uvx_ephemeral(self):
        with patch("cozempic.init.sys.executable", r"C:\Users\x\.cache\uv\environments-v2\abc\python.exe"):
            _, ephemeral = cz._resolve_cozempic_python()
        self.assertTrue(ephemeral)

    def test_resolve_degrades_when_sys_executable_empty(self):
        # exotic frozen interpreters can have an empty sys.executable — must NOT
        # return "" (which would bake an empty `'' -m cozempic` no-op command).
        with patch("cozempic.init.sys.executable", ""), \
                patch("cozempic.init.shutil.which", return_value="/usr/bin/python3"):
            got, _ = cz._resolve_cozempic_python()
            self.assertEqual(got, "/usr/bin/python3")

    def test_bake_skips_when_abs_python_empty(self):
        hooks = {"E": [{"hooks": [{"command": "python3 -m cozempic x"}]}]}
        out = cz._bake_cozempic_path(hooks, "")
        self.assertEqual(out["E"][0]["hooks"][0]["command"], "python3 -m cozempic x")  # untouched

    def test_resolve_flags_persistent_not_ephemeral(self):
        for p in ("/Users/x/.local/share/uv/tools/cozempic/bin/python3.12",
                  "/opt/homebrew/bin/python3", "/usr/local/bin/python3.11"):
            with patch("cozempic.init.sys.executable", p):
                _, eph = cz._resolve_cozempic_python()
                self.assertFalse(eph, f"{p} should be persistent")

    def test_settings_scalar_returns_error_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".claude" / "settings.json"
            path.parent.mkdir(parents=True)
            path.write_text("[]")
            self.assertTrue(cz.cozempic_hook_schema_state(path).error)
            self.assertIn("JSON object", cz.wire_hooks(tmp)["error"])
            self.assertIn("JSON object", cz.uninstall_hooks(tmp)["error"])

    def test_null_hook_containers_are_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".claude" / "settings.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"hooks": {"SessionStart": null}}')
            result = cz.wire_hooks(tmp)
            self.assertFalse(result.get("error"))
            settings = json.loads(path.read_text())
            self.assertIsInstance(settings["hooks"]["SessionStart"], list)

    def test_wire_hooks_repairs_malformed_nested_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".claude" / "settings.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"hooks": {"SessionStart": [null, {"matcher": "", "hooks": null}]}}')
            result = cz.wire_hooks(tmp)
            self.assertFalse(result.get("error"))
            settings = json.loads(path.read_text())
            entries = settings["hooks"]["SessionStart"]
            self.assertTrue(entries)
            self.assertTrue(all(isinstance(entry, dict) for entry in entries))
            self.assertTrue(all(isinstance(entry.get("hooks"), list) for entry in entries))

    def test_hook_state_tolerates_truthy_non_list_containers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".claude" / "settings.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"hooks": {"SessionStart": [{"hooks": 5}]}}')
            self.assertFalse(cz.cozempic_hook_schema_state(path).error)

    def test_hook_commands_tolerate_non_string_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".claude" / "settings.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"hooks": {"SessionStart": [{"hooks": [{"command": null}]}]}}')
            self.assertFalse(cz.wire_hooks(tmp).get("error"))
            self.assertFalse(cz.uninstall_hooks(tmp).get("error"))

    def test_init_replaces_stale_local_hooks_with_project_hooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / ".claude" / "settings.local.json"
            local.parent.mkdir(parents=True)
            local.write_text('{"hooks": {"SessionStart": [{"hooks": [{"command": "{ cozempic guard; } # cozempic-hook-schema=v1"}]}]}}')
            cz.run_init(tmp, skip_slash=True)
            self.assertNotIn("cozempic", local.read_text())
            settings = json.loads((Path(tmp) / ".claude" / "settings.json").read_text())
            self.assertIn("cozempic", json.dumps(settings))


class TestWireHooksBakes(unittest.TestCase):
    def test_wire_hooks_writes_absolute_fallback_and_reports_ephemeral(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("cozempic.init._resolve_cozempic_python",
                       return_value=("/abs/uvtool/bin/python3.12", False)):
                res = cz.wire_hooks(tmp)
            self.assertFalse(res["ephemeral"])
            self.assertEqual(res["cozempic_python"], "/abs/uvtool/bin/python3.12")
            settings = json.loads((Path(tmp) / ".claude" / "settings.json").read_text())
            cmds = [h["command"] for entries in settings["hooks"].values()
                    for e in entries for h in e["hooks"]]
            # at least one hook now carries the absolute interpreter fallback
            self.assertTrue(any("/abs/uvtool/bin/python3.12 -m cozempic" in c for c in cmds))
            # and none still falls back to bare python3
            self.assertFalse(any("|| python3 -m cozempic" in c for c in cmds))

    def test_wire_hooks_propagates_ephemeral_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("cozempic.init._resolve_cozempic_python",
                       return_value=("/tmp/uvx/bin/python", True)):
                res = cz.wire_hooks(tmp)
            self.assertTrue(res["ephemeral"])


if __name__ == "__main__":
    unittest.main()
