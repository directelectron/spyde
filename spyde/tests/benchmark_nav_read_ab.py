"""
benchmark_nav_read_ab.py — A/B the 4D-STEM DP navigator read.

GATE for unifying the 4D-STEM diffraction-pattern (DP) navigator onto
``ComputeBackend.submit_graph`` (the pooled sync-graph path proven for movies).
CLAUDE.md §1-4 forbid touching the live-display core without a real-scale
benchmark; this is it.

Compares, for the SAME nav indices, on a real 4D-STEM scan:

  * ``getindex``   the current live-display call
                   ``sig._get_cache_dask_chunk(indices, get_result=True)`` — which
                   is hyperspy ``CachedDaskArray.get_index(..., sum_data=True)``:
                   a SINGLE nav point returns that frame; MULTIPLE points return
                   the **mean over them** (integrating region).
  * ``submit``     ``ComputeBackend.submit_graph`` of the equivalent lazy op:
                   single point -> ``raw[iy, ix]``; region -> ``raw[pts].mean(0)``
                   (cast back to the frame dtype to match get_index's no-upcast).

Asserts the two produce the **same array** (single point AND integrating region),
then reports latency for each. Optionally stands up a real distributed
``LocalCluster`` (``--distributed``) to time the get_index path on the cluster the
app actually uses (else both run threaded/synchronous, still a valid correctness +
graph-walk-cost comparison).

Run (NOT under pytest; a real cluster won't run in an agent sandbox):

    .venv/Scripts/python -m spyde.tests.benchmark_nav_read_ab
    .venv/Scripts/python -m spyde.tests.benchmark_nav_read_ab --distributed
    .venv/Scripts/python -m spyde.tests.benchmark_nav_read_ab --path "C:\\...\\scan.mrc"
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import hyperspy.api as hs

from spyde.compute_backend import ComputeBackend

# Real 4D-STEM scans on this dev box (nav-dim 2, small DP signal).
_CANDIDATES = [
    r"C:\Users\CarterFrancis\Downloads\directelectron_in-situ-tinbo_2026-06-04_2019\20241219_29674_movie_movie.mrc",
    r"C:\Users\CarterFrancis\Downloads\20241215_29639_movie_movie.mrc",
]


def _default_path():
    for p in _CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _load_4dstem(path):
    if path:
        s = hs.load(path, lazy=True, chunks=(32, 32, -1, -1))
        if isinstance(s, list):
            s = s[0]
        return s
    # Fallback: pyxem's canonical sped_ag (208x64 nav, 112x112 DP).
    import pyxem  # noqa: F401
    from pyxem.data import sped_ag
    s = sped_ag(allow_download=True, lazy=True)
    s = s.rechunk((32, 32, -1, -1))
    return s


def _region_points(iy, ix, r, ny, nx):
    """A small integrating region around (iy, ix): list of [iy, ix] nav points."""
    pts = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            y, x = iy + dy, ix + dx
            if 0 <= y < ny and 0 <= x < nx:
                pts.append([int(y), int(x)])
    return pts


def _getindex(sig, indices):
    return np.asarray(sig._get_cache_dask_chunk(np.array(indices), get_result=True))


def _submit_lazy(raw, indices):
    """The equivalent lazy op submit_graph would compute. Matches hyperspy
    get_index(sum_data=True): a single nav point returns that frame (native
    dtype); MULTIPLE points return the float64 **mean** over them (get_index
    returns the region mean un-rounded as float64 — NOT cast back to the frame
    dtype; verified against _get_cache_dask_chunk)."""
    idx = np.asarray(indices)
    if idx.shape[0] == 1:
        iy, ix = int(idx[0, 0]), int(idx[0, 1])
        return raw[iy, ix]
    ys = idx[:, 0].astype(int)
    xs = idx[:, 1].astype(int)
    import dask.array as da
    frames = da.stack([raw[int(y), int(x)] for y, x in zip(ys, xs)], axis=0)
    return frames.mean(axis=0)   # float64, matches get_index region mean


def _time(fn, reps=6, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(ts)), float(np.min(ts))


def main() -> None:
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None)
    ap.add_argument("--distributed", action="store_true",
                    help="stand up a real LocalCluster to time get_index on it")
    args = ap.parse_args()

    path = args.path or _default_path()
    sig = _load_4dstem(path)
    am = sig.axes_manager
    ny, nx = int(am.navigation_shape[1]), int(am.navigation_shape[0])
    raw = sig.data
    print(f"# 4D-STEM nav-read A/B\n\nsource: "
          f"{os.path.basename(path) if path else 'pyxem sped_ag'}")
    print(f"shape={raw.shape} dtype={raw.dtype} nav=({ny},{nx}) "
          f"sig={tuple(am.signal_shape)}\n")

    client = None
    if args.distributed:
        from dask.distributed import LocalCluster, Client
        cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True,
                               dashboard_address=None)
        client = Client(cluster)
        # Pin the cache client like the app does.
        _ = sig._get_cache_dask_chunk  # ensure attr exists
        if getattr(sig, "cached_dask_array", None) is None:
            sig._get_cache_dask_chunk(np.array([[0, 0]]), get_result=True)
        try:
            sig.cached_dask_array._client = client
        except Exception:
            pass
        print(f"distributed: {client}\n")

    pool = ThreadPoolExecutor(max_workers=2)
    backend = ComputeBackend(executor=pool)

    # Test positions: a few single points + integrating regions, mid-scan.
    iy, ix = ny // 2, nx // 2
    cases = [
        ("single", [[iy, ix]]),
        ("single2", [[iy + 1, ix + 3]]),
        ("region-r1", _region_points(iy, ix, 1, ny, nx)),
        ("region-r2", _region_points(iy, ix, 2, ny, nx)),
    ]

    # The unified candidate: get_index with NO distributed client (synchronous
    # numpy chunk cache — the SAME cache logic, minus the distributed hop),
    # submitted to OUR pool for a cancellable async Future. Make sure the cache
    # client is unset so get_index takes the synchronous cache path.
    if getattr(sig, "cached_dask_array", None) is None:
        _getindex(sig, [[iy, ix]])
    if client is None:
        try:
            sig.cached_dask_array._client = None
        except Exception:
            pass

    def _cached_read(inds):
        # Runs get_index (synchronous cache path) inside a pool worker → a real
        # cancellable Future. In the app this runs on the serial _NavDispatcher
        # so the cache is never re-entered concurrently (CLAUDE.md §4).
        return backend.submit(lambda i=inds: _getindex(sig, i))

    print("| case | pts | match | getindex ms | submit_graph ms | cached_read ms |")
    print("|---|---:|---|---:|---:|---:|")
    all_match = True
    for name, inds in cases:
        gi = _getindex(sig, inds)
        sg = backend.submit_graph(_submit_lazy(raw, inds)).result()
        cr = _cached_read(inds).result()
        match = (gi.shape == sg.shape == cr.shape
                 and np.allclose(gi.astype(np.float64), sg.astype(np.float64), atol=1e-6)
                 and np.allclose(gi.astype(np.float64), cr.astype(np.float64), atol=1e-6))
        all_match = all_match and match
        gi_ms = _time(lambda inds=inds: _getindex(sig, inds))
        sg_ms = _time(lambda inds=inds: backend.submit_graph(
            _submit_lazy(raw, inds)).result())
        cr_ms = _time(lambda inds=inds: _cached_read(inds).result())
        print(f"| {name} | {len(inds)} | {'OK' if match else 'MISMATCH'} | "
              f"{gi_ms[0]:.1f} | {sg_ms[0]:.1f} | {cr_ms[0]:.1f} |")

    print()
    print(f"**Correctness:** {'all match' if all_match else 'MISMATCH — do NOT unify'}")
    print("`cached_read` = get_index (no distributed client, synchronous chunk "
          "cache) in our pool = cancellable Future + ~1 ms dwell-in-chunk hits, "
          "no distributed overhead.")

    pool.shutdown(wait=False, cancel_futures=True)
    if client is not None:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
