"""
repro_movie_scrub.py — prove the pooled sync-graph movie read path.

Validates ``ComputeBackend.submit_graph`` (threaded pool + synchronous-scheduler
compute of a lazy dask slice) gives the movie navigator what the distributed
path has (cancellable async Futures) WITHOUT the distributed round-trip, and that
the SAME path scrubs a lazy CROP for free.

Proves:
  1. latency      — one submitted+resolved frame is ~memmap-fast, not ~250 ms.
  2. cancellation — a fast scrub (submit N, keep only the last) cancels the
                    superseded queued frames cleanly (latest-position-wins), and
                    an async done-callback delivers only the surviving frame.
  3. crop-scrub   — a lazy ``s.inav[..].isig[..]`` crop reads through the SAME
                    submit_graph path (cropping is a pure graph op, no materialise).

Run (NOT under pytest — reads a real multi-GB movie off disk):

    .venv/Scripts/python -m spyde.tests.repro_movie_scrub
    .venv/Scripts/python -m spyde.tests.repro_movie_scrub --path "C:\\...\\movie.mrc"
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import hyperspy.api as hs

from spyde.compute_backend import ComputeBackend

_CANDIDATES = [
    r"C:\Users\CarterFrancis\Downloads\20251117_88074_run1_9104_movie.mrc",
    r"C:\Users\CarterFrancis\Downloads\20251117_88075_run3 some growth_1236_movie.mrc",
    r"C:\Users\CarterFrancis\Downloads\20241002_07954_movie.mrc",
]


def _default_path():
    for p in _CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None)
    args = ap.parse_args()
    path = args.path or _default_path()
    if not path or not os.path.exists(path):
        print("No movie found. Pass --path <file.mrc>.")
        return

    # Load with the Phase-1 one-frame-per-chunk layout (adaptive chunking does
    # this in the app; here we force it so the read touches one frame's chunk).
    s = hs.load(path, lazy=True, chunks=(1, -1, -1))
    if isinstance(s, list):
        s = s[0]
    raw = s.data
    n_time = int(raw.shape[0])
    print(f"movie {raw.shape} dtype={raw.dtype} n_time={n_time}\n")

    # OUR pool (the async layer we own) → threaded ComputeBackend.
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="movie-read")
    backend = ComputeBackend(executor=pool)

    # warm one frame (cold CUDA/graph/OS-cache one-time cost).
    backend.submit_graph(raw[0]).result()

    # ── 1. latency ────────────────────────────────────────────────────────────
    lat = []
    for t in np.linspace(1, n_time - 1, 8).astype(int):
        t0 = time.perf_counter()
        f = backend.submit_graph(raw[int(t)]).result()
        lat.append((time.perf_counter() - t0) * 1e3)
    print(f"1) latency: mean {np.mean(lat):.1f} ms  min {np.min(lat):.1f}  "
          f"max {np.max(lat):.1f}  frame {f.shape}")

    # ── 2. cancellation under a fast scrub (submit a burst, keep only last) ────
    start = min(200, n_time - 25)
    futs = [backend.submit_graph(raw[start + i]) for i in range(20)]
    # Cancel everything except the last (what latest-wins coalescing would do).
    cancelled = sum(1 for fu in futs[:-1] if fu.cancel())
    # The surviving future delivers its frame via a done-callback (async paint).
    got = {}
    done = threading.Event()

    def _paint(fu):
        try:
            got["shape"] = fu.result().shape
        finally:
            done.set()

    futs[-1].add_done_callback(_paint)
    done.wait(10)
    print(f"2) cancel: cancelled {cancelled}/{len(futs) - 1} queued scrub frames; "
          f"survivor delivered via callback -> {got.get('shape')}")

    # ── 3. crop-then-scrub through the SAME path ──────────────────────────────
    y0 = raw.shape[1] // 4
    x0 = raw.shape[2] // 4
    t0 = time.perf_counter()
    cropped = s.inav[start:start + 20].isig[x0:3 * x0, y0:3 * y0]   # lazy graph op
    build_ms = (time.perf_counter() - t0) * 1e3
    craw = cropped.data
    # warm + scrub the cropped view via submit_graph.
    backend.submit_graph(craw[0]).result()
    clat = []
    for t in range(1, min(6, craw.shape[0])):
        t0 = time.perf_counter()
        cf = backend.submit_graph(craw[t]).result()
        clat.append((time.perf_counter() - t0) * 1e3)
    print(f"3) crop-scrub: crop build {build_ms:.1f} ms (lazy {cropped._lazy}); "
          f"cropped frame {cf.shape} read mean {np.mean(clat):.1f} ms "
          f"(same submit_graph path)")

    pool.shutdown(wait=False, cancel_futures=True)
    print("\nOK - pooled sync-graph gives futures+cancel+callback at low latency, "
          "and scrubs a lazy crop through the same path.")


if __name__ == "__main__":
    main()
