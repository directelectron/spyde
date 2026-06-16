#!/usr/bin/env python
"""Build a uv-managed SpyDE.app bundle (+ .dmg) for macOS.

The .app wraps the same uv-managed payload as the Windows installer: the
project + bundled uv + installer/launch.py live in Contents/Resources, and a
small shell launcher in Contents/MacOS runs launch.py via the bundled uv. The
icon (spyde/icon.icns) and a proper Info.plist make it a real, double-clickable,
Dock-pinnable app.

Usage (CI, on macOS):
    python tools/write_build_info.py --channel stable
    python tools/build_macos_app.py            # -> dist/SpyDE.app, dist/SpyDE.dmg
"""
from __future__ import annotations

import os
import plistlib
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
APP = DIST / "SpyDE.app"
DMG = DIST / "SpyDE.dmg"

UV_RELEASE = (
    "https://github.com/astral-sh/uv/releases/latest/download/"
    "uv-aarch64-apple-darwin.tar.gz"
)

EXCLUDE = {"__pycache__", "tests", ".venv", ".venv2"}
INCLUDE_FILES = ["pyproject.toml", "uv.lock", "main.py"]
INCLUDE_DIRS = ["spyde", "installer"]


def _copy_payload(resources: Path) -> None:
    for name in INCLUDE_FILES:
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, resources / name)
    for d in INCLUDE_DIRS:
        src = ROOT / d
        if not src.exists():
            continue
        for item in src.rglob("*"):
            if any(p in EXCLUDE or p.endswith(".egg-info") for p in item.parts):
                continue
            if item.is_dir():
                continue
            rel = item.relative_to(ROOT)
            out = resources / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, out)


def _bundle_uv(resources: Path) -> None:
    dst = resources / "uv"
    local = shutil.which("uv")
    if local:
        shutil.copy2(local, dst)
    else:
        import tarfile
        tmp = DIST / "_uv.tar.gz"
        urllib.request.urlretrieve(UV_RELEASE, tmp)
        with tarfile.open(tmp) as t:
            for m in t.getmembers():
                if m.name.endswith("/uv") or m.name == "uv":
                    m.name = "uv"
                    t.extract(m, resources)
                    break
        tmp.unlink(missing_ok=True)
    os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _write_launcher(macos: Path) -> None:
    launcher = macos / "SpyDE"
    launcher.write_text(
        '#!/bin/bash\n'
        'DIR="$(cd "$(dirname "$0")/../Resources" && pwd)"\n'
        'export SPYDE_HOME="$DIR"\n'
        'exec "$DIR/uv" run --frozen --project "$DIR" "$DIR/main.py"\n',
        encoding="utf-8")
    os.chmod(launcher, 0o755)


def _write_plist(contents: Path, version: str) -> None:
    plist = {
        "CFBundleName": "SpyDE",
        "CFBundleDisplayName": "SpyDE",
        "CFBundleIdentifier": "com.directelectron.spyde",
        "CFBundleVersion": version,
        "CFBundleShortVersionString": version,
        "CFBundleExecutable": "SpyDE",
        "CFBundleIconFile": "icon.icns",
        "CFBundlePackageType": "APPL",
        "LSMinimumSystemVersion": "11.0",
        "NSHighResolutionCapable": True,
    }
    with open(contents / "Info.plist", "wb") as f:
        plistlib.dump(plist, f)


def _version() -> str:
    ns: dict = {}
    exec((ROOT / "spyde" / "_version.py").read_text(encoding="utf-8"), ns)
    return ns["__version__"]


def main() -> int:
    if APP.exists():
        shutil.rmtree(APP)
    contents = APP / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    for d in (macos, resources):
        d.mkdir(parents=True)

    _copy_payload(resources)
    _bundle_uv(resources)
    icns = ROOT / "spyde" / "icon.icns"
    if icns.exists():
        shutil.copy2(icns, resources / "icon.icns")
    _write_launcher(macos)
    _write_plist(contents, _version())

    # Package into a .dmg (with the icon as the volume icon when available).
    if DMG.exists():
        DMG.unlink()
    try:
        subprocess.check_call([
            "create-dmg", "--volname", "SpyDE",
            "--volicon", str(icns) if icns.exists() else "",
            "--icon", "SpyDE.app", "175", "190",
            "--app-drop-link", "425", "190",
            "--window-size", "600", "400",
            str(DMG), str(APP),
        ])
    except Exception:
        # fallback: plain hdiutil image
        subprocess.check_call([
            "hdiutil", "create", "-volname", "SpyDE",
            "-srcfolder", str(APP), "-ov", "-format", "UDZO", str(DMG)])

    print(f"built {APP} and {DMG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
