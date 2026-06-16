#!/usr/bin/env python
"""Stamp spyde/_build_info.py with real build metadata (run in CI before bundling).

Usage:
    python tools/write_build_info.py [--channel stable|beta]

Version comes from spyde/_version.py; git sha from the current checkout; the
channel defaults to "stable" (or "beta" if the version has a pre-release tag).
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_INFO = ROOT / "spyde" / "_build_info.py"


def _version() -> str:
    ns: dict = {}
    exec((ROOT / "spyde" / "_version.py").read_text(encoding="utf-8"), ns)
    return ns["__version__"]


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default=None,
                    choices=["stable", "beta", "dev"])
    ap.add_argument("--version", default=None,
                    help="override the stamped version (e.g. the git tag, "
                         "with any leading 'v' stripped)")
    args = ap.parse_args()

    version = (args.version or _version()).lstrip("v")
    channel = args.channel or (
        "beta" if any(t in version for t in ("rc", "a", "b", "dev")) else "stable")
    sha = _git_sha()
    date = datetime.now(timezone.utc).isoformat(timespec="seconds")

    text = BUILD_INFO.read_text(encoding="utf-8")
    repl = {
        'VERSION = "0.0.0+dev"': f'VERSION = "{version}"',
        'GIT_SHA = "unknown"': f'GIT_SHA = "{sha}"',
        'CHANNEL = "dev"': f'CHANNEL = "{channel}"',
        'BUILD_DATE = ""': f'BUILD_DATE = "{date}"',
    }
    for old, new in repl.items():
        if old not in text:
            print(f"warning: marker not found: {old}", file=sys.stderr)
        text = text.replace(old, new, 1)
    BUILD_INFO.write_text(text, encoding="utf-8")
    print(f"stamped {BUILD_INFO.name}: {version} ({sha}, {channel}) {date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
