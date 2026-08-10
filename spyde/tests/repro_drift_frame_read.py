"""
repro_drift_frame_read.py — what a per-frame ``.compute()`` costs with a live
Dask client, measured rather than argued.

Run it directly (needs a REAL cluster, so it will not run under the test suite
or inside an agent sandbox — CLAUDE.md § Testing)::

    uv run python -m spyde.tests.repro_drift_frame_read

It builds a memmap-backed lazy movie (the shape a real ``.mrc`` in-situ movie
has: one frame per chunk) and times ONE frame read three ways:

    local            explicit ``scheduler="threads"`` — what frames.py does now
    ambient client   bare ``.compute()`` with a distributed Client alive — what
                     it did before, and what the running app therefore did
    memmap floor     the raw numpy read, i.e. the cost that is actually inherent

Then it multiplies by the frame counts the drift caret really uses (64 for the
check image, 20 for the ROI preview, N for the solve) so the numbers land in the
units of the complaint: seconds of dead UI.
"""
from __future__ import annotations

import os
import tempfile
import time

import numpy as np

FRAMES = 24
SIZE = 2048
CHECK_FRAMES = 64      # drift_action._SUM_MAX_FRAMES
PREVIEW_FRAMES = 20    # drift_action._PREVIEW_FRAMES


def _movie(path: str):
    """A REAL file loaded lazily — the graph a user's in-situ movie actually has.

    Deliberately NOT ``da.from_array(memmap)``: that embeds the whole source array
    in the graph, so every ``.compute()`` also ships hundreds of MB to the
    scheduler and the distributed number comes out flattering-to-the-argument.
    hyperspy's lazy reader builds a proper reader graph, which is what the app
    hands to ``frame_source``.
    """
    import hyperspy.api as hs
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 4000, (FRAMES, SIZE, SIZE), dtype=np.uint16)
    hs.signals.Signal2D(frames).save(path, overwrite=True)
    del frames
    # 1 frame/chunk, the storage-aligned shape an in-situ movie has
    # (CLAUDE.md Live-Display §1).
    sig = hs.load(path, lazy=True)
    # 1 frame/chunk, the storage-aligned shape an in-situ movie has (CLAUDE.md
    # Live-Display §1). Rechunking a LAZY array only rebuilds the graph.
    return None, sig.data.rechunk((1, SIZE, SIZE))


def _time(fn, n=8):
    fn(0)                                    # warm
    t0 = time.perf_counter()
    for i in range(n):
        fn(i % FRAMES)
    return (time.perf_counter() - t0) / n


def main() -> int:
    tmp = os.path.join(tempfile.gettempdir(), "spyde-drift-read-repro.hspy")
    print(f"building {FRAMES} x {SIZE}^2 uint16 movie ({FRAMES*SIZE*SIZE*2/1e9:.1f} GB)")
    _unused, arr = _movie(tmp)

    # np.asarray on a memmap slice is a VIEW — it reads nothing. Force the read.
    floor = _time(lambda i: np.array(arr[i].compute(scheduler="single-threaded")))
    local = _time(lambda i: np.asarray(arr[i].compute(scheduler="threads")))

    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=2, threads_per_worker=2, processes=True,
                           dashboard_address=None)
    client = Client(cluster)
    try:
        ambient = _time(lambda i: np.asarray(arr[i].compute()))
    finally:
        client.close()
        cluster.close()

    print(f"\n{'read path':<28}{'ms/frame':>10}")
    print(f"{'single-threaded floor':<28}{floor*1e3:>10.1f}")
    print(f"{'local threaded scheduler':<28}{local*1e3:>10.1f}")
    print(f"{'ambient distributed client':<28}{ambient*1e3:>10.1f}   "
          f"({ambient/max(local,1e-9):.0f}x the local read)")

    print(f"\n{'drift stage':<28}{'frames':>8}{'before':>12}{'after':>12}")
    for label, k in (("check image (drift_open)", CHECK_FRAMES),
                     ("ROI preview", PREVIEW_FRAMES),
                     (f"solve ({FRAMES} frames)", FRAMES)):
        print(f"{label:<28}{k:>8}{ambient*k:>11.1f}s{local*k:>11.1f}s")
    try:
        os.remove(tmp)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
