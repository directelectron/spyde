"""
benchmark_drift_latency.py — where the drift caret's wall-clock actually goes.

Run directly, never under pytest (CLAUDE.md § Benchmarking)::

    uv run python -m spyde.tests.benchmark_drift_latency
    uv run python -m spyde.tests.benchmark_drift_latency --size 4096 --frames 120

The complaint this exists to answer was "the ROI spawns a drift check image which
takes 60 seconds" and "the Correct Drift button just seems to do nothing". A
green functional suite could not see either, because it runs on a 96x112 fixture
where every stage is fast for reasons that do not survive a real movie.

This drives the REAL staged handlers (``drift_open`` → ``drift_run`` →
``drift_commit``) against a real ``Session`` and a real lazy movie on disk, with
``SPYDE_ACTION_PROFILE`` on, and prints the per-stage breakdown each one logs.

**It runs Dask-free** (``SPYDE_NO_DASK=1``), which is a deliberate limitation
worth stating: the distributed round-trip a live cluster adds to every frame read
is measured separately by ``repro_drift_frame_read.py`` (2.5x on a lazy .hspy
movie) because a real ``LocalCluster`` cannot be spun up here. So these numbers
are the floor — the app pays these PLUS that multiplier on the read stages.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time

os.environ.setdefault("SPYDE_NO_DASK", "1")
os.environ["SPYDE_ACTION_PROFILE"] = "1"
os.environ.setdefault(
    "SPYDE_SETTINGS_DIR", tempfile.mkdtemp(prefix="spyde-drift-latency-"))

import numpy as np  # noqa: E402


def _movie(path: str, size: int, frames: int):
    """A real file on disk, loaded lazily at 1 frame/chunk — an in-situ movie."""
    import hyperspy.api as hs
    if not os.path.exists(path):
        rng = np.random.default_rng(0)
        block = rng.integers(0, 4000, (frames, size, size), dtype=np.uint16)
        # A moving bright band so the drift solve has something to lock onto.
        for i in range(frames):
            x = int(size * 0.3) + 2 * i
            block[i, :, x:x + 24] = 20000
        hs.signals.Signal2D(block).save(path)
        del block
    sig = hs.load(path, lazy=True)
    sig.data = sig.data.rechunk((1, size, size))
    ax = sig.axes_manager.navigation_axes[0]
    ax.name, ax.units, ax.scale = "time", "s", 0.05
    sig.set_signal_type("insitu")
    return sig


def _wait(pred, timeout=1800.0, what="condition"):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for {what}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args(argv)

    # The profile lines are INFO on the backend logger; send them to stdout so
    # running this prints the answer instead of hiding it.
    # A Windows console is cp1252 and this file prints box-drawing characters;
    # without this the whole run dies on the first heading, after paying for the
    # movie load. errors="replace" so a stray glyph degrades instead of aborting.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    for noisy in ("hyperspy", "rsciio", "distributed", "asyncio", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from spyde.backend.session import Session
    import spyde.actions.drift_action as dr

    path = os.path.join(tempfile.gettempdir(),
                        f"spyde-drift-latency-{args.size}-{args.frames}.hspy")
    print(f"movie: {args.frames} x {args.size}^2 uint16 lazy, 1 frame/chunk")
    t0 = time.perf_counter()
    sig = _movie(path, args.size, args.frames)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s\n")

    session = Session(n_workers=1, threads_per_worker=1)
    try:
        session._add_signal(sig, source_path=path)
        plot = next(p for p in session._plots
                    if not p.is_navigator and p.plot_state is not None)
        tree = plot.signal_tree

        print("── drift_open (caret mount → check image + first preview) ──")
        dr.drift_open(session, plot, {"upsample": 8, "max_shift": 32})
        _wait(lambda: getattr(tree, "_drift_wizard", None) is not None
              and tree._drift_wizard.window_id is not None,
              what="the Drift Check window")
        wiz = tree._drift_wizard
        _wait(lambda: wiz.preview is not None, what="the first ROI preview")

        print("\n── drift_run (Correct Drift) ──")
        dr.drift_run(session, plot, {"upsample": 8, "max_shift": 32})
        _wait(lambda: wiz.model is not None, what="the solve")

        print("\n── drift_commit (Apply) ──")
        t = time.perf_counter()
        node = wiz.commit()
        dt = time.perf_counter() - t
        print(f"  Apply returned {'a node' if node is not None else 'None'} "
              f"in {dt * 1e3:.0f}ms")
    finally:
        session.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
