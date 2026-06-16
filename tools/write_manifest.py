#!/usr/bin/env python
"""Generate latest.json — the update manifest published with each release.

The in-app updater currently reads the GitHub Releases API directly, but a
static manifest gives us a stable, cacheable channel definition (and room for
min-supported-version / staged rollout later). Published as a release asset.

Usage:
    python tools/write_manifest.py --channel stable --out dist/latest.json \
        [--asset NAME=URL ...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _version() -> str:
    ns: dict = {}
    exec((ROOT / "spyde" / "_version.py").read_text(encoding="utf-8"), ns)
    return ns["__version__"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="stable")
    ap.add_argument("--out", default="dist/latest.json")
    ap.add_argument("--asset", action="append", default=[],
                    help="NAME=URL pairs for downloadable artifacts")
    ap.add_argument("--notes-url", default="")
    args = ap.parse_args()

    assets = {}
    for pair in args.asset:
        if "=" in pair:
            name, url = pair.split("=", 1)
            assets[name.strip()] = url.strip()

    manifest = {
        "schema": 1,
        "channel": args.channel,
        "version": _version(),
        "released": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_supported_version": "0.1.0",
        "assets": assets,
        "notes_url": args.notes_url,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out}: {manifest['version']} ({manifest['channel']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
