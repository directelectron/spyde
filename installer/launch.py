"""uv-managed launcher for an installed SpyDE.

The installer lays down, under the install root:
    pyproject.toml, uv.lock, spyde/, main.py, uv(.exe), launch.py

On launch this script ensures the managed venv is synced (first run, or after an
update changed the lock) and then runs the app via uv. Keeping the venv next to
the install (not baked into the artifact) is what makes installs small and
updates incremental.

A marker file records the lock hash the venv was last synced for, so we only pay
`uv sync` when the lock actually changed.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYNC_MARKER = ROOT / ".venv" / ".spyde_sync_hash"


def _uv() -> str:
    for name in ("uv.exe", "uv"):
        cand = ROOT / name
        if cand.exists():
            return str(cand)
    import shutil
    found = shutil.which("uv")
    if found:
        return found
    raise SystemExit("uv not found next to SpyDE; reinstall.")


def _lock_hash() -> str:
    lock = ROOT / "uv.lock"
    if not lock.exists():
        return ""
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def _needs_sync() -> bool:
    if not (ROOT / ".venv").exists():
        return True
    try:
        return SYNC_MARKER.read_text(encoding="utf-8").strip() != _lock_hash()
    except Exception:
        return True


def main() -> int:
    uv = _uv()
    env = dict(os.environ)
    env.setdefault("SPYDE_HOME", str(ROOT))

    if _needs_sync():
        # First run or post-update: build/refresh the managed venv with the
        # GPU-correct torch wheel. Streams to stdout (a splash/console can show
        # it). --frozen keeps it reproducible from the shipped lock.
        rc = subprocess.call(
            [uv, "sync", "--frozen", "--torch-backend=auto"],
            cwd=str(ROOT), env=env)
        if rc != 0:
            # fall back to a non-frozen resolve if the frozen lock is stale
            rc = subprocess.call(
                [uv, "sync", "--torch-backend=auto"], cwd=str(ROOT), env=env)
        if rc != 0:
            return rc
        try:
            SYNC_MARKER.parent.mkdir(parents=True, exist_ok=True)
            SYNC_MARKER.write_text(_lock_hash(), encoding="utf-8")
        except Exception:
            pass

    # Launch the app inside the synced environment.
    return subprocess.call(
        [uv, "run", "--frozen", "main.py"], cwd=str(ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
