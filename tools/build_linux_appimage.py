#!/usr/bin/env python
"""Build a uv-managed SpyDE AppImage for Linux.

Assembles an AppDir wrapping the uv-managed payload (project + bundled uv +
installer/launch.py), with a .desktop entry and the Spyde.png icon, then runs
appimagetool to produce dist/SpyDE.AppImage.

The AppRun launcher runs launch.py via the bundled uv, which on first run does
`uv sync` to build the managed venv next to the extracted app (or in
$SPYDE_HOME). Note: AppImages are read-only at runtime, so the venv is created
under $HOME/.local/share/SpyDE (set via SPYDE_HOME) rather than inside the image.

Usage (CI, on Linux):
    python tools/write_build_info.py --channel stable
    python tools/build_linux_appimage.py        # -> dist/SpyDE.AppImage
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
APPDIR = DIST / "SpyDE.AppDir"
OUT = DIST / "SpyDE.AppImage"

UV_RELEASE = (
    "https://github.com/astral-sh/uv/releases/latest/download/"
    "uv-x86_64-unknown-linux-gnu.tar.gz"
)
APPIMAGETOOL = (
    "https://github.com/AppImage/appimagetool/releases/download/continuous/"
    "appimagetool-x86_64.AppImage"
)

EXCLUDE = {"__pycache__", "tests", ".venv", ".venv2"}
INCLUDE_FILES = ["pyproject.toml", "uv.lock", "main.py"]
INCLUDE_DIRS = ["spyde", "installer"]


def _copy_payload(root: Path) -> None:
    usr = root / "usr" / "share" / "spyde"
    usr.mkdir(parents=True, exist_ok=True)
    for name in INCLUDE_FILES:
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, usr / name)
    for d in INCLUDE_DIRS:
        src = ROOT / d
        if not src.exists():
            continue
        for item in src.rglob("*"):
            if any(p in EXCLUDE or p.endswith(".egg-info") for p in item.parts):
                continue
            if item.is_dir():
                continue
            out = usr / item.relative_to(ROOT)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, out)


def _bundle_uv(root: Path) -> None:
    import tarfile
    dst = root / "usr" / "bin" / "uv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    local = shutil.which("uv")
    if local:
        shutil.copy2(local, dst)
    else:
        tmp = DIST / "_uv.tar.gz"
        urllib.request.urlretrieve(UV_RELEASE, tmp)
        with tarfile.open(tmp) as t:
            for m in t.getmembers():
                if m.name.endswith("/uv") or m.name == "uv":
                    m.name = "uv"
                    t.extract(m, dst.parent)
                    break
        tmp.unlink(missing_ok=True)
    os.chmod(dst, 0o755)


def _write_apprun(root: Path) -> None:
    apprun = root / "AppRun"
    apprun.write_text(
        '#!/bin/bash\n'
        'HERE="$(dirname "$(readlink -f "$0")")"\n'
        'APP="$HERE/usr/share/spyde"\n'
        '# AppImage is read-only; keep the managed venv in the user data dir.\n'
        'export SPYDE_HOME="${SPYDE_HOME:-$HOME/.local/share/SpyDE}"\n'
        'mkdir -p "$SPYDE_HOME"\n'
        '# sync project sources into the writable home on first run / version change\n'
        'rsync -a --delete --exclude ".venv" "$APP/" "$SPYDE_HOME/" 2>/dev/null '
        '|| cp -ru "$APP/." "$SPYDE_HOME/"\n'
        'exec "$HERE/usr/bin/uv" run --frozen --project "$SPYDE_HOME" '
        '"$SPYDE_HOME/main.py"\n',
        encoding="utf-8")
    os.chmod(apprun, 0o755)


def _write_desktop_and_icon(root: Path) -> None:
    (root / "spyde.desktop").write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=SpyDE\n"
        "Exec=AppRun\n"
        "Icon=spyde\n"
        "Categories=Science;Education;\n"
        "Comment=Electron microscopy visualization & analysis\n",
        encoding="utf-8")
    icon = ROOT / "spyde" / "Spyde.png"
    if icon.exists():
        shutil.copy2(icon, root / "spyde.png")
        # AppImage also wants the icon under usr/share/icons
        icons = root / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps"
        icons.mkdir(parents=True, exist_ok=True)
        shutil.copy2(icon, icons / "spyde.png")


def _appimagetool() -> str:
    tool = DIST / "appimagetool"
    if not tool.exists():
        urllib.request.urlretrieve(APPIMAGETOOL, tool)
        os.chmod(tool, 0o755)
    return str(tool)


def main() -> int:
    if APPDIR.exists():
        shutil.rmtree(APPDIR)
    APPDIR.mkdir(parents=True)

    _copy_payload(APPDIR)
    _bundle_uv(APPDIR)
    _write_apprun(APPDIR)
    _write_desktop_and_icon(APPDIR)

    if OUT.exists():
        OUT.unlink()
    env = dict(os.environ, ARCH="x86_64")
    subprocess.check_call([_appimagetool(), str(APPDIR), str(OUT)], env=env)
    print(f"built {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
