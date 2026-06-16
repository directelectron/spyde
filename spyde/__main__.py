"""
spyde/__main__.py — entry point for the Electron-backed app.

The Electron main process spawns this as a subprocess and communicates via
stdin/stdout JSON lines (the PLOTAPP: protocol from anyplotlib._electron).

When run standalone for debugging:
    uv run python -m spyde
"""
from __future__ import annotations


def main() -> None:
    from spyde.backend.app import run
    run()


if __name__ == "__main__":
    main()
