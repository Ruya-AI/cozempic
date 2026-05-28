"""Pytest fixtures shared across the session-pruner regression suite."""

from __future__ import annotations

from typing import Callable

import pytest

from tests._session_factory import make_realistic_session  # noqa: F401


@pytest.fixture
def realistic_session_factory() -> Callable[..., list[dict]]:
    """Factory fixture: callable that builds a fresh session per test."""
    return make_realistic_session
