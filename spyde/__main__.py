"""
spyde/__main__.py — entry point for the Electron-backed app.

The Electron main process spawns this as a subprocess and communicates via
stdin/stdout JSON lines (the PLOTAPP: protocol from anyplotlib._electron).

When run standalone for debugging:
    uv run python -m spyde
"""
from __future__ import annotations


def main() -> None:
    # CRITICAL (packaged Windows app): Dask spawns each worker via
    # multiprocessing 'spawn', which re-executes THIS frozen entrypoint. Without
    # freeze_support(), every worker subprocess re-runs the full app bootstrap
    # (heavy hyperspy/pyxem/torch imports — and even start_dask) before
    # multiprocessing can hijack it, so spawning 11 workers blocked the
    # LocalCluster constructor for ~71s. freeze_support() makes a worker child
    # detect it is a multiprocessing bootstrap and short-circuit to the worker
    # entry instead of re-running main(). No-op in a normal (non-frozen) run.
    import multiprocessing
    multiprocessing.freeze_support()

    from spyde.backend.app import run
    run()


if __name__ == "__main__":
    main()
