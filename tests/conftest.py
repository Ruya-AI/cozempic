"""Shared pytest fixtures.

Hermetic ``~/.claude``: point ``CLAUDE_CONFIG_DIR`` at a fresh temp dir for every
test so nothing reads the developer's real Claude config — in particular the
active-transcript store (``cozempic-active-sessions.json``) that
``find_current_session`` consults as Strategy 1. Without this, those tests
non-deterministically pick up the live session's real record (they patch
``get_projects_dir`` but not the active store), which is exactly the latent
non-hermetic-test leak the 1.8.30 detection work introduced.

A test that needs a specific config dir still overrides this — its own
``monkeypatch.setenv`` / ``patch("...get_claude_dir")`` wins.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_claude_config_dir(tmp_path_factory, monkeypatch):
    d = tmp_path_factory.mktemp("claude_home")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(d))
    yield


@pytest.fixture(autouse=True)
def _disable_real_receipts(monkeypatch):
    """Never write prune receipts to the developer's real ``~/.cozempic`` from a
    test. cmd_treat/cmd_reload emit receipts on every execute (D1), so any
    treat/reload test that doesn't isolate HOME would otherwise leak a receipt
    into the real receipts dir. Receipt-specific tests opt back in by popping
    ``COZEMPIC_NO_RECEIPTS`` inside an isolated HOME/base_dir.

    NOTE: this is an autouse *pytest* fixture, so the hermetic guarantee holds
    only under pytest. Run the suite with pytest (not bare ``python -m unittest``)
    or a treat/reload test relying solely on this guard could write to real
    ~/.cozempic. (Receipt-writer tests stay safe under bare unittest — they
    isolate base_dir/HOME themselves.)"""
    monkeypatch.setenv("COZEMPIC_NO_RECEIPTS", "1")
    yield


@pytest.fixture(autouse=True)
def _default_digest_enabled(monkeypatch):
    """Run the suite with the digest in its default (enabled) state even on a
    developer machine that exports ``COZEMPIC_NO_DIGEST`` — the pre-existing
    digest tests assert enabled-path behavior. Opt-out tests set the variable
    explicitly inside their own scope (test_digest_optout.py)."""
    monkeypatch.delenv("COZEMPIC_NO_DIGEST", raising=False)
    yield
