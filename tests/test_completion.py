"""Tests for shell completion generation."""
import unittest
from cozempic.completion import bash_completion, zsh_completion


class TestBashCompletion(unittest.TestCase):
    def test_contains_complete_directive(self):
        self.assertIn("complete -F _cozempic cozempic", bash_completion())

    def test_includes_subcommands(self):
        script = bash_completion()
        for cmd in ["list", "treat", "guard", "doctor", "completions"]:
            self.assertIn(cmd, script)

    def test_includes_prescriptions(self):
        script = bash_completion()
        for rx in ["gentle", "standard", "aggressive"]:
            self.assertIn(rx, script)


class TestZshCompletion(unittest.TestCase):
    def test_contains_compdef(self):
        self.assertIn("#compdef cozempic", zsh_completion())


if __name__ == "__main__":
    unittest.main()
