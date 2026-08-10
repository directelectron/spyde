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
import faulthandler
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
    """A real .mrc on disk, loaded lazily — MEMMAP reads, no decode.

    The backing is the measurement. An earlier version of this benchmark used
    ``.hspy`` and reported ~180 ms/frame, which is HDF5 DECODE cost — a real
    in-situ movie is .mrc / .de5 / raw binary, where a frame is a memmap slice
    and the only cost is moving bytes. Measuring the wrong container inflates
    every read stage and points the fix at the wrong thing.
    """
    import hyperspy.api as hs
    from spyde.backend._session_testharness import TestHarnessMixin
    rng = np.random.default_rng(0)
    block = rng.integers(0, 4000, (frames, size, size), dtype=np.uint16)
    # A moving bright band so the drift solve has something to lock onto.
    for i in range(frames):
        x = int(size * 0.3) + 2 * i
        block[i, :, x:x + 24] = 20000
    path = TestHarnessMixin._write_movie_mrc(block)
    del block
    sig = hs.load(path, lazy=True)
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
    ap.add_argument("--assert-budget", action="store_true",
                    help="exit non-zero if a stage misses its budget (CI gate)")
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

    # If a stage WEDGES rather than merely being slow, a timing print never
    # arrives and the run looks identical to "still working". Dump every thread's
    # stack periodically so a hang names its own line instead of being guessed at.
    faulthandler.dump_traceback_later(90, repeat=True, exit=False, file=sys.stderr)

    from spyde.backend.session import Session
    import spyde.actions.drift_action as dr
    from spyde.backend import action_profile as _ap

    # Capture every profile line so --assert-budget can judge them. The lines are
    # the contract the budgets are written against, so read them rather than
    # re-timing independently: a gate that measures something other than what the
    # app reports would drift away from it silently.
    seen: list[tuple[str, float]] = []
    _orig_done = _ap.ActionProfile.done

    def _done(self, extra: str = "") -> None:
        if getattr(self, "_on", False):
            total = (time.perf_counter() - self._t0) * 1e3
            seen.append((self._label, total))
        _orig_done(self, extra)

    _ap.ActionProfile.done = _done

    path = ""   # _movie picks the (reused) .mrc path
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

    if args.assert_budget:
        # Caret-open is the number the maintainer set: 200 ms from click to the
        # Drift Check image being on screen. The others are tripwires an order of
        # magnitude above where they sit, to catch a pathology rather than noise.
        budgets = {"drift_open": 200.0, "drift_commit": 500.0, "drift_run": 30_000.0}
        worst = {}
        for label, total in seen:
            worst[label] = max(worst.get(label, 0.0), total)
        bad = []
        print("")
        print("budget check")
        for label, limit in budgets.items():
            got = worst.get(label)
            if got is None:
                print(f"  {label:<16} MISSING (never ran)")
                bad.append(label)
                continue
            ok = got <= limit
            print(f"  {label:<16} {got:8.1f} ms   budget {limit:8.1f} ms   "
                  f"{'ok' if ok else 'OVER'}")
            if not ok:
                bad.append(label)
        if bad:
            print("")
            print(f"FAILED: {', '.join(bad)}")
            return 1
        print("")
        print("all stages inside budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
