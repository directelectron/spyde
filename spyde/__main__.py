"""
spyde/__main__.py — entry point for the Electron-backed app.

The Electron main process spawns this as a subprocess and communicates via
stdin/stdout JSON lines (the PLOTAPP: protocol from anyplotlib._electron).

When run standalone for debugging:
    uv run python -m spyde
"""
from __future__ import annotations


def _set_mac_neural_env() -> None:
    """Mac-only: set the neural-inference robustness env BEFORE anything imports
    torch (so it takes effect before torch initialises MPS) AND before dask spawns
    workers (spawn copies this process's environment, so workers inherit these
    without relying solely on the worker plugin). No-op off Mac; setdefault so an
    explicit user override wins.

    - PYTORCH_ENABLE_MPS_FALLBACK=1 (fix 1): unsupported MPS ops → CPU per-op.
    - SPYDE_FV_GPU_CONC=1 (fix 3): one MPS forward at a time per process."""
    import os
    import sys
    if sys.platform != "darwin":
        return
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("SPYDE_FV_GPU_CONC", "1")


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

    # Set Mac MPS-fallback / device-serialisation env BEFORE torch is imported so
    # it takes effect before MPS initialises, and before workers are spawned (they
    # inherit this environment). No-op off Mac.
    _set_mac_neural_env()

    from spyde.backend.app import run
    run()


if __name__ == "__main__":
    main()
