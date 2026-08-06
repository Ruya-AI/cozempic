"""Windows-quiet subprocess wrappers.

On Windows, a console program (``ps``, ``taskkill``, ``git``, ``uv``, the
guard-daemon relaunch, …) spawned by a GUI or detached parent *without* the
``CREATE_NO_WINDOW`` creation flag briefly allocates a console host window — a
visible flash. cozempic shells out ~two dozen times, so running each of those
through ``subprocess`` directly makes a window flash on every guard spawn, PID
lookup, auto-update check, etc. These thin wrappers inject ``CREATE_NO_WINDOW``
by default so the child stays windowless.

Everything here is a no-op on POSIX: ``CREATE_NO_WINDOW`` doesn't exist there
(``getattr`` resolves to 0), and ``creationflags`` is a documented ``Popen``
keyword on all platforms that must be 0 on POSIX — which is exactly what
``_quiet_flags`` returns off Windows.

Use these in place of ``subprocess.run`` / ``.Popen`` / ``.call``. Callers that
genuinely want a visible console can pass ``creationflags=subprocess.CREATE_NEW_CONSOLE``
and it is honored (the two flags are mutually exclusive, so we don't add
CREATE_NO_WINDOW on top).
"""
from __future__ import annotations

import os
import subprocess

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)


def _quiet_flags(creationflags: int = 0) -> int:
    """Return ``creationflags`` with CREATE_NO_WINDOW OR-ed in on Windows.

    No-op on POSIX. Left unchanged when the caller explicitly requested a new
    console (CREATE_NEW_CONSOLE), since that is mutually exclusive with
    CREATE_NO_WINDOW.
    """
    if os.name != "nt":
        return creationflags
    if creationflags & _CREATE_NEW_CONSOLE:
        return creationflags
    return creationflags | _CREATE_NO_WINDOW


def run(*args, **kwargs):
    """``subprocess.run`` that is windowless-by-default on Windows."""
    kwargs["creationflags"] = _quiet_flags(kwargs.get("creationflags", 0))
    return subprocess.run(*args, **kwargs)


def popen(*args, **kwargs):
    """``subprocess.Popen`` that is windowless-by-default on Windows."""
    kwargs["creationflags"] = _quiet_flags(kwargs.get("creationflags", 0))
    return subprocess.Popen(*args, **kwargs)


def call(*args, **kwargs):
    """``subprocess.call`` that is windowless-by-default on Windows."""
    kwargs["creationflags"] = _quiet_flags(kwargs.get("creationflags", 0))
    return subprocess.call(*args, **kwargs)
