#!/usr/bin/env python
"""Stage the uv-managed install payload for the NSIS installer.

Produces dist/installer_payload/ containing everything the installer ships:
    pyproject.toml, uv.lock, main.py, spyde/, installer/launch.py,
    uv.exe (bundled), Spyde.ico, SpyDE.exe (windowed launcher stub)

The launcher stub is built with PyInstaller from installer/launch.py so a
double-click runs `uv sync` (first run) then the app, with no console flash.

Usage (CI, on Windows):
    python tools/write_build_info.py --channel stable
    python tools/build_installer_payload.py
    makensis /DVERSION=<v> installer/spyde.nsi
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
PAYLOAD = DIST / "installer_payload"

# Files/dirs copied verbatim into the payload.
INCLUDE = ["pyproject.toml", "uv.lock", "main.py"]
INCLUDE_DIRS = ["spyde", "installer"]
EXCLUDE_DIR_NAMES = {"__pycache__", "tests", ".venv", ".venv2", "egg-info"}

UV_RELEASE = (
    "https://github.com/astral-sh/uv/releases/latest/download/"
    "uv-x86_64-pc-windows-msvc.zip"
)


def _copy_tree(src: Path, dst: Path) -> None:
    for item in src.rglob("*"):
        if any(part in EXCLUDE_DIR_NAMES or part.endswith(".egg-info")
               for part in item.parts):
            continue
        if item.is_dir():
            continue
        rel = item.relative_to(src.parent)
        out = PAYLOAD / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, out)


def _ensure_uv() -> None:
    """Place uv.exe in the payload (download the pinned release if not present)."""
    dst = PAYLOAD / "uv.exe"
    local = shutil.which("uv")
    if local and local.lower().endswith(".exe"):
        shutil.copy2(local, dst)
        return
    print("downloading uv release…")
    tmp = DIST / "_uv.zip"
    urllib.request.urlretrieve(UV_RELEASE, tmp)
    with zipfile.ZipFile(tmp) as z:
        for name in z.namelist():
            if name.endswith("uv.exe"):
                with z.open(name) as src, open(dst, "wb") as f:
                    shutil.copyfileobj(src, f)
                break
    tmp.unlink(missing_ok=True)


def _build_launcher_stub() -> None:
    """SpyDE.exe — a windowed PyInstaller one-file stub of installer/launch.py."""
    ico = ROOT / "spyde" / "Spyde.ico"
    cmd = [
        sys.executable, "-m", "PyInstaller", "--onefile", "--noconsole",
        "--name", "SpyDE", "--distpath", str(PAYLOAD),
        "--workpath", str(DIST / "_pyi_work"),
        "--specpath", str(DIST / "_pyi_spec"),
    ]
    if ico.exists():
        cmd += ["--icon", str(ico)]
    cmd.append(str(ROOT / "installer" / "launch.py"))
    subprocess.check_call(cmd)


def main() -> int:
    if PAYLOAD.exists():
        shutil.rmtree(PAYLOAD)
    PAYLOAD.mkdir(parents=True)

    for name in INCLUDE:
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, PAYLOAD / name)
    for d in INCLUDE_DIRS:
        if (ROOT / d).exists():
            _copy_tree(ROOT / d, PAYLOAD / d)

    ico = ROOT / "spyde" / "Spyde.ico"
    if ico.exists():
        shutil.copy2(ico, PAYLOAD / "Spyde.ico")

    _ensure_uv()
    _build_launcher_stub()

    print(f"payload staged at {PAYLOAD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
