"""
PyCrucible entry point.
Delegates straight to the installed spyde package so all relative imports work.
"""

if __name__ == "__main__":
    # Must run BEFORE importing the heavy spyde stack: in the packaged app, Dask
    # spawns workers by re-executing this frozen entrypoint via multiprocessing.
    # freeze_support() lets a worker child short-circuit to the multiprocessing
    # bootstrap instead of re-running the whole app (the ~71s 11-worker spawn
    # stall). No-op in a normal (non-frozen) run.
    import multiprocessing
    multiprocessing.freeze_support()

    from spyde.__main__ import main
    main()

