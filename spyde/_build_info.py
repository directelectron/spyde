"""Build metadata for the About box, update checks, and diagnostics.

In a released build, ``tools/write_build_info.py`` overwrites the constants
below with the real version / git sha / channel / build date. In a dev checkout
the values are resolved lazily from ``spyde._version`` + git so the module still
works (and the About box reads "dev").
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone

# These are overwritten in CI by tools/write_build_info.py. The sentinels let a
# dev checkout fall back to live resolution.
VERSION = "0.0.0+dev"
GIT_SHA = "unknown"
CHANNEL = "dev"          # "stable" | "beta" | "dev"
BUILD_DATE = ""          # ISO8601 UTC


def _dev_version() -> str:
    try:
        from spyde._version import __version__
        return f"{__version__}+dev"
    except Exception:
        return "0.0.0+dev"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def build_info() -> dict:
    """Resolved build info, filling dev fallbacks when CI didn't stamp it."""
    version = VERSION
    sha = GIT_SHA
    channel = CHANNEL
    date = BUILD_DATE
    if version == "0.0.0+dev":          # not stamped → dev checkout
        version = _dev_version()
        sha = _git_sha()
        channel = "dev"
        date = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "version": version,
        "git_sha": sha,
        "channel": channel,
        "build_date": date,
    }


def version_string() -> str:
    """Human-friendly one-liner, e.g. 'SpyDE 0.1.0 (a1b2c3d, stable)'."""
    info = build_info()
    return f"SpyDE {info['version']} ({info['git_sha']}, {info['channel']})"
