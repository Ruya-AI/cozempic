"""#168: the SessionStart hook must be POSIX-sh clean.

Claude Code invokes hooks with `/bin/sh`. On Debian/Ubuntu that is dash, which
does NOT support bash's `${var:offset:length}` substring expansion — the hook
used `${SESSION_ID:0:12}` and aborted with "Bad substitution", silently killing
the guard-daemon spawn on those systems. The fix computes the slug once with the
POSIX-safe `SLUG=$(printf '%.12s' "$SESSION_ID")`.

These tests pin: (1) no bash-only substring expansion survives in any shipped
hook, (2) the POSIX slug is defined and used, and (3) the construct runs clean
under a strict POSIX shell (dash if present) producing the same 12-char slug.
"""

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHIPPED_HOOKS = [
    ROOT / "src" / "cozempic" / "data" / "hooks.json",
    ROOT / "plugin" / "hooks" / "hooks.json",
]

# bash-only ${var:offset[:length]} substring expansion — a colon FOLLOWED BY A
# DIGIT. Deliberately does NOT match the POSIX forms ${var:-default},
# ${var:+alt}, ${var:=default} (colon followed by -, +, =).
_BASH_SUBSTRING = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*:[0-9]")


class TestSessionStartHookIsPosix(unittest.TestCase):
    def _ss_cmd(self, path: Path) -> str:
        data = json.loads(path.read_text())
        return data["hooks"]["SessionStart"][0]["hooks"][0]["command"]

    def test_no_bash_only_substring_expansion(self):
        for f in SHIPPED_HOOKS:
            cmd = self._ss_cmd(f)
            m = _BASH_SUBSTRING.search(cmd)
            self.assertIsNone(
                m, f"{f.name}: bash-only substring expansion "
                f"{m.group(0) if m else ''!r} breaks under dash (#168)")

    def test_defines_and_uses_posix_slug(self):
        for f in SHIPPED_HOOKS:
            cmd = self._ss_cmd(f)
            self.assertIn(
                "SLUG=$(printf '%.12s' \"$SESSION_ID\")", cmd,
                f"{f.name}: must define the slug via POSIX printf (#168)")
            self.assertIn(
                "${SLUG}", cmd,
                f"{f.name}: tmp paths must reference the portable ${{SLUG}}")

    def test_slug_runs_clean_under_posix_sh(self):
        """The POSIX slug construct yields the same 12-char prefix as bash's
        ${SESSION_ID:0:12} under a strict POSIX shell, with no error."""
        sh = shutil.which("dash") or "/bin/sh"
        snippet = ('SESSION_ID=5d53e013-32d9-4e72-9a3a-deadbeefcafe; '
                   'SLUG=$(printf "%.12s" "$SESSION_ID"); printf %s "$SLUG"')
        out = subprocess.run([sh, "-c", snippet], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, f"{sh}: {out.stderr}")
        self.assertNotIn("Bad substitution", out.stderr)
        self.assertEqual(out.stdout, "5d53e013-32d",
                         "POSIX slug must equal bash ${SESSION_ID:0:12}")


if __name__ == "__main__":
    unittest.main()
