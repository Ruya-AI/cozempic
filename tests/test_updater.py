"""Tests for auto-update logic."""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestVersionTuple(unittest.TestCase):
    def test_parses_version(self):
        from cozempic.updater import _version_tuple
        self.assertEqual(_version_tuple("1.2.0"), (1, 2, 0))
        self.assertEqual(_version_tuple("2.0.0"), (2, 0, 0))

    def test_bad_version_returns_zeros(self):
        from cozempic.updater import _version_tuple
        self.assertEqual(_version_tuple("bad"), (0,))


class TestShouldCheck(unittest.TestCase):
    def test_no_cache_file_means_should_check(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / ".cozempic_update_check"
            with patch("cozempic.updater._CACHE_FILE", cache):
                from cozempic.updater import _should_check
                self.assertTrue(_should_check())

    def test_recent_check_means_skip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / ".cozempic_update_check"
            cache.write_text(str(time.time()))
            with patch("cozempic.updater._CACHE_FILE", cache):
                from cozempic.updater import _should_check
                self.assertFalse(_should_check())

    def test_old_check_means_should_check(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / ".cozempic_update_check"
            cache.write_text(str(time.time() - 90000))  # 25 hours ago
            with patch("cozempic.updater._CACHE_FILE", cache):
                from cozempic.updater import _should_check
                self.assertTrue(_should_check())


class _EnvIsolated(unittest.TestCase):
    """Save → clear → restore the auto-update opt-out env vars around each test.

    A maintainer who PINS carries COZEMPIC_PIN in their shell (the feature's own
    target audience), so any test touching maybe_auto_update must neutralize these
    or it fails order-dependently. We save-and-restore (not pop) so a pre-existing
    ambient value survives the run (#123 QA P3 — was a destructive pop)."""

    _OPT_VARS = ("COZEMPIC_NO_AUTO_UPDATE", "COZEMPIC_PIN")

    def setUp(self):
        self._saved_env = {k: os.environ.pop(k, None) for k in self._OPT_VARS}

    def tearDown(self):
        for k in self._OPT_VARS:
            os.environ.pop(k, None)
            if self._saved_env.get(k) is not None:
                os.environ[k] = self._saved_env[k]


class TestMaybeAutoUpdate(_EnvIsolated):
    def test_skips_when_env_var_set(self):
        """COZEMPIC_NO_AUTO_UPDATE=1 disables all update activity."""
        with patch.dict(os.environ, {"COZEMPIC_NO_AUTO_UPDATE": "1"}):
            with patch("cozempic.updater._should_check") as mock_check:
                from cozempic.updater import maybe_auto_update
                maybe_auto_update()
                mock_check.assert_not_called()

    def test_works_without_tty(self):
        """Auto-update should work even without TTY (hooks, daemons)."""
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            with patch("cozempic.updater._should_check", return_value=False) as mock_check:
                from cozempic.updater import maybe_auto_update
                maybe_auto_update()
                mock_check.assert_called()  # Should still check (TTY no longer blocks)

    def test_skips_when_already_checked(self):
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = True
            with patch("cozempic.updater._should_check", return_value=False):
                with patch("cozempic.updater._get_latest_version") as mock_get:
                    from cozempic.updater import maybe_auto_update
                    maybe_auto_update()
                    mock_get.assert_not_called()

    def test_skips_when_already_up_to_date(self):
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = True
            with patch("cozempic.updater._should_check", return_value=True), \
                 patch("cozempic.updater._mark_checked"), \
                 patch("cozempic.updater._get_latest_version", return_value="0.0.1"), \
                 patch("cozempic.updater._do_upgrade") as mock_upgrade:
                from cozempic.updater import maybe_auto_update
                maybe_auto_update()
                mock_upgrade.assert_not_called()

    def test_upgrades_when_newer_version_available(self, capsys=None):
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = True
            with patch("cozempic.updater._should_check", return_value=True), \
                 patch("cozempic.updater._mark_checked"), \
                 patch("cozempic.updater._get_latest_version", return_value="99.99.99"), \
                 patch("cozempic.updater._do_upgrade", return_value=True) as mock_upgrade:
                from cozempic.updater import maybe_auto_update
                maybe_auto_update()
                mock_upgrade.assert_called_once_with("99.99.99")

    def test_prints_failure_message_on_upgrade_error(self):
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = True
            calls = []
            mock_stdout.write = lambda s: calls.append(s)
            with patch("cozempic.updater._should_check", return_value=True), \
                 patch("cozempic.updater._mark_checked"), \
                 patch("cozempic.updater._get_latest_version", return_value="99.99.99"), \
                 patch("cozempic.updater._do_upgrade", return_value=False), \
                 patch("builtins.print") as mock_print:
                from cozempic.updater import maybe_auto_update
                maybe_auto_update()
                printed = " ".join(str(a) for call in mock_print.call_args_list for a in call[0])
                self.assertIn("auto-update failed", printed)

    def test_no_op_when_pypi_unreachable(self):
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = True
            with patch("cozempic.updater._should_check", return_value=True), \
                 patch("cozempic.updater._mark_checked"), \
                 patch("cozempic.updater._get_latest_version", return_value=None), \
                 patch("cozempic.updater._do_upgrade") as mock_upgrade:
                from cozempic.updater import maybe_auto_update
                maybe_auto_update()
                mock_upgrade.assert_not_called()


class TestInstallMethodDetection(unittest.TestCase):
    """1.8.22: auto-update must pick a mechanism that matches the install method —
    pip can't upgrade a Homebrew keg or a `uv tool` install (the binary on PATH
    never moves), which is why brew/uvx users silently stayed behind."""

    def _method_for(self, path):
        from cozempic import updater
        with patch.object(updater, "__file__", path):
            return updater._install_method()

    def test_detects_brew_keg(self):
        self.assertEqual(self._method_for(
            "/opt/homebrew/Cellar/cozempic/1.8.22/libexec/lib/python3.12/site-packages/cozempic/updater.py"),
            "brew")

    def test_detects_uv_tool(self):
        self.assertEqual(self._method_for(
            "/Users/x/.local/share/uv/tools/cozempic/lib/python3.12/site-packages/cozempic/updater.py"),
            "uv-tool")

    def test_detects_pipx(self):
        self.assertEqual(self._method_for(
            "/Users/x/.local/pipx/venvs/cozempic/lib/python3.12/site-packages/cozempic/updater.py"),
            "pipx")

    def test_defaults_to_pip(self):
        self.assertEqual(self._method_for(
            "/Users/x/proj/.venv/lib/python3.12/site-packages/cozempic/updater.py"),
            "pip")


class TestDoUpgradeDispatch(unittest.TestCase):
    def test_brew_never_autoruns(self):
        from cozempic import updater
        with patch.object(updater, "_install_method", return_value="brew"), \
             patch("cozempic.updater._sp.run") as run:
            self.assertFalse(updater._do_upgrade("9.9.9"))
            run.assert_not_called()

    def test_uv_tool_runs_uv_tool_upgrade(self):
        from cozempic import updater
        with patch.object(updater, "_install_method", return_value="uv-tool"), \
             patch("cozempic.updater.shutil.which", return_value="/usr/bin/uv"), \
             patch("cozempic.updater._sp.run",
                   return_value=MagicMock(returncode=0)) as run:
            self.assertTrue(updater._do_upgrade("9.9.9"))
            self.assertEqual(run.call_args[0][0], ["uv", "tool", "upgrade", "cozempic"])

    def test_pip_uses_install_chain(self):
        from cozempic import updater
        with patch.object(updater, "_install_method", return_value="pip"), \
             patch("cozempic.updater.shutil.which", return_value=None), \
             patch("cozempic.updater._sp.run",
                   return_value=MagicMock(returncode=0)) as run:
            self.assertTrue(updater._do_upgrade("9.9.9"))
            # first attempt is `pip install cozempic==…` via sys.executable
            self.assertIn("install", run.call_args[0][0])


class TestAutoUpdateOptOuts(_EnvIsolated):
    """#123: the documented kill switch must actually stop the Python updater,
    and COZEMPIC_PIN holds a reviewed version without auto-installing it."""

    def test_no_auto_update_skips_everything(self):
        from cozempic import updater
        os.environ["COZEMPIC_NO_AUTO_UPDATE"] = "1"
        with patch("cozempic.updater._should_check") as sc, \
             patch("cozempic.updater._do_upgrade") as up:
            updater.maybe_auto_update(force=True)
            sc.assert_not_called()   # returns before even checking
            up.assert_not_called()

    def test_pin_disables_autoupdate(self):
        from cozempic import updater
        os.environ["COZEMPIC_PIN"] = updater.__version__  # pinned to current
        with patch("cozempic.updater._get_latest_version", return_value="99.0.0"), \
             patch("cozempic.updater._do_upgrade") as up, \
             patch("cozempic.updater._should_check", return_value=True):
            updater.maybe_auto_update(force=True)
            up.assert_not_called()   # never upgrades while pinned

    def test_pin_warns_on_drift(self):
        import io
        from cozempic import updater
        os.environ["COZEMPIC_PIN"] = "1.0.0"  # != current installed version
        buf = io.StringIO()
        with patch("sys.stdout", buf), \
             patch("cozempic.updater._should_check", return_value=True), \
             patch("cozempic.updater._mark_checked"), \
             patch("cozempic.updater._do_upgrade") as up:
            updater.maybe_auto_update(force=True)
            up.assert_not_called()
        out = buf.getvalue()
        self.assertIn("pinned to 1.0.0", out)
        self.assertIn("pip install 'cozempic==1.0.0'", out)

    def test_pin_matching_current_is_silent(self):
        import io
        from cozempic import updater
        os.environ["COZEMPIC_PIN"] = updater.__version__
        buf = io.StringIO()
        with patch("sys.stdout", buf), \
             patch("cozempic.updater._should_check", return_value=True), \
             patch("cozempic.updater._do_upgrade"):
            updater.maybe_auto_update(force=True)
        self.assertEqual(buf.getvalue(), "", "no drift warning when pin == current")

    def test_pinned_version_helper(self):
        from cozempic import updater
        self.assertIsNone(updater._pinned_version())
        os.environ["COZEMPIC_PIN"] = " 1.8.30 "
        self.assertEqual(updater._pinned_version(), "1.8.30")  # trimmed

    def test_whitespace_pin_disables_update_matching_shell(self):
        # #123 QA P3: a whitespace-only pin must be PINNED (auto-update OFF), the
        # same as the hook's `[ -z "$COZEMPIC_PIN" ]` (non-empty → skip upgrade),
        # not fall through to auto-update as the old .strip()->None did.
        from cozempic import updater
        os.environ["COZEMPIC_PIN"] = "   "
        self.assertTrue(updater._pinned_version())  # truthy → counts as pinned
        with patch("cozempic.updater._should_check", return_value=True), \
             patch("cozempic.updater._do_upgrade") as up:
            updater.maybe_auto_update(force=True)
            up.assert_not_called()

    def test_v_prefix_pin_does_not_false_warn(self):
        # #123 QA P3: COZEMPIC_PIN=v<current> must not warn against <current>.
        import io
        from cozempic import updater
        os.environ["COZEMPIC_PIN"] = "v" + updater.__version__
        buf = io.StringIO()
        with patch("sys.stdout", buf), \
             patch("cozempic.updater._should_check", return_value=True), \
             patch("cozempic.updater._do_upgrade"):
            updater.maybe_auto_update(force=True)
        self.assertEqual(buf.getvalue(), "", "leading-v pin equal to current must be silent")

    def test_garbage_pin_emits_no_unrunnable_command(self):
        # #123 QA P3: a non-version pin still disables update but must NOT print a
        # copy-paste `pip install 'cozempic==garbage'` that errors.
        import io
        from cozempic import updater
        os.environ["COZEMPIC_PIN"] = "garbage"
        buf = io.StringIO()
        with patch("sys.stdout", buf), \
             patch("cozempic.updater._should_check", return_value=True), \
             patch("cozempic.updater._mark_checked"), \
             patch("cozempic.updater._do_upgrade") as up:
            updater.maybe_auto_update(force=True)
            up.assert_not_called()       # still pinned → no upgrade
        self.assertNotIn("pip install", buf.getvalue())  # but no broken command

    def test_unicode_digit_pin_emits_no_command(self):
        # #123 QA re-verify P3: a unicode-digit pin (fullwidth / Arabic-Indic) is
        # NOT a pip-installable version — _VERSION_SHAPE (re.A) must reject it so we
        # never print an un-runnable `pip install 'cozempic==１.８.３２'`.
        import io
        from cozempic import updater
        for v in ("１.８.３２", "١.٨.٣٢"):
            os.environ["COZEMPIC_PIN"] = v
            buf = io.StringIO()
            with patch("sys.stdout", buf), \
                 patch("cozempic.updater._should_check", return_value=True), \
                 patch("cozempic.updater._mark_checked"), \
                 patch("cozempic.updater._do_upgrade") as up:
                updater.maybe_auto_update(force=True)
                up.assert_not_called()                    # still pinned → no upgrade
            self.assertNotIn("pip install", buf.getvalue())  # no un-runnable command


class TestHookHonorsOptOuts(unittest.TestCase):
    """#123 Defect 1: the SessionStart hook's shell upgrade must be gated on the
    SAME env vars the README advertises, not bypass them."""

    def test_sessionstart_upgrade_is_guarded(self):
        import json
        from pathlib import Path
        import cozempic
        hooks = json.loads((Path(cozempic.__file__).parent / "data" / "hooks.json").read_text())
        cmd = hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]
        # The pip --upgrade must be inside a guard that checks both opt-outs.
        guard = 'if [ -z "$COZEMPIC_NO_AUTO_UPDATE" ] && [ -z "$COZEMPIC_PIN" ]; then'
        self.assertIn(guard, cmd)
        # And the guard must come BEFORE the upgrade in the command string.
        self.assertLess(cmd.index(guard), cmd.index("pip install --upgrade cozempic"))

    def test_npm_install_decide_honors_optouts(self):
        # #123 QA P1/P2: BEHAVIORAL — exercise install.js's decideInstall(env) via
        # node, not a static text check (the static-only test gave false confidence
        # and missed a whitespace-pin divergence). Asserts parity with the hook /
        # Python paths: ANY non-empty pin (incl. whitespace/garbage) drops --upgrade.
        import json
        import shutil
        import subprocess
        from pathlib import Path
        node = shutil.which("node")
        if not node:
            self.skipTest("node not available")
        install_js = Path(__file__).parent.parent / "npm" / "install.js"
        script = (
            "const {decideInstall}=require(process.argv[1]);"
            "const cases=JSON.parse(process.argv[2]);"
            "console.log(JSON.stringify(cases.map(decideInstall)));"
        )
        cases = [
            {}, {"COZEMPIC_NO_AUTO_UPDATE": "1"}, {"COZEMPIC_PIN": "1.8.30"},
            {"COZEMPIC_PIN": "v1.8.30"}, {"COZEMPIC_PIN": "garbage"},
            {"COZEMPIC_PIN": "   "}, {"COZEMPIC_PIN": "1.0.0; rm -rf /"}, {"COZEMPIC_PIN": ""},
        ]
        out = subprocess.run([node, "-e", script, str(install_js), json.dumps(cases)],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        res = json.loads(out.stdout)
        # default + empty pin → --upgrade; everything else (opt-out) → no --upgrade.
        self.assertEqual(res[0]["up"], ["--upgrade"], "default must still upgrade")
        self.assertEqual(res[7]["up"], ["--upgrade"], "empty pin is not an opt-out")
        for i in (1, 2, 3, 4, 5, 6):
            self.assertEqual(res[i]["up"], [], f"case {i} ({cases[i]}) must drop --upgrade")
        # version-shaped pin → exact spec; malformed/injection pin → bare cozempic.
        self.assertEqual(res[2]["spec"], "cozempic==1.8.30")
        self.assertEqual(res[3]["spec"], "cozempic==1.8.30")  # leading v stripped
        self.assertEqual(res[4]["spec"], "cozempic")           # garbage not used as spec
        self.assertEqual(res[6]["spec"], "cozempic")           # injection rejected


class TestMaybeAutoUpdateBrew(_EnvIsolated):
    def test_brew_prints_hint_and_does_not_attempt(self):
        import io
        from cozempic import updater
        buf = io.StringIO()
        with patch("sys.stdout", buf), \
             patch("cozempic.updater._should_check", return_value=True), \
             patch("cozempic.updater._mark_checked"), \
             patch("cozempic.updater._get_latest_version", return_value="99.0.0"), \
             patch.object(updater, "_install_method", return_value="brew"), \
             patch("cozempic.updater._do_upgrade") as up:
            updater.maybe_auto_update()
            up.assert_not_called()
        out = buf.getvalue()
        # Fully-qualified so brew's untrusted-tap gate doesn't block the upgrade.
        self.assertIn("brew upgrade Ruya-AI/cozempic/cozempic", out)
        self.assertNotIn("updating", out.lower())
