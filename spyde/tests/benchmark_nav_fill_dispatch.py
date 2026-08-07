"""Navigator fill A/B: the retired per-chunk submit loop vs the shared
dispatcher (``compute_dispatch.dispatch_chunks``).

Run directly (slow, needs a real file + a real LocalCluster)::

    python -m spyde.tests.benchmark_nav_fill_dispatch --impl new
    python -m spyde.tests.benchmark_nav_fill_dispatch --impl old
    python -m spyde.tests.benchmark_nav_fill_dispatch --impl both --warm

Default target is the 15.3 GB / 977-frame in-situ MRC.  What it reports:

  submit    wall time until the fill call RETURNS.  The old path made one
            blocking ``client.compute()`` scheduler round trip per nav chunk
            with the GIL held in the client process, so for 977 chunks nothing
            else in the backend could run for the whole of this window — the
            navigator sat blank and even the paint threads went silent.
  1st chunk wall time until the first per-chunk callback fires (the first pixel
            the user sees).
  total     wall time until the whole navigator is assembled.

PAGE CACHE: a 15.3 GB file fits in this box's 128 GB RAM, so a second run reads
from RAM, not disk — the classic page-cache benchmark trap.  ``--purge``
evicts the file's cached pages via FILE_FLAG_NO_BUFFERING before each run;
``--warm`` deliberately does the opposite (one throwaway pass first) so the two
implementations are compared with I/O held constant and only the dispatch
overhead differs.
"""
from __future__ import annotations

import argparse
import ctypes
import itertools
import os
import threading
import time

import numpy as np

DEFAULT_PATH = r"D:\InsituElectroChemistry\20251117_88075_run3 some growth_1236_movie.mrc"


# ── page cache ────────────────────────────────────────────────────────────────

def purge_cache(path: str) -> None:
    """Evict ``path``'s pages from the Windows file cache.

    Opening a handle with FILE_FLAG_NO_BUFFERING makes the cache manager purge
    the cached view of that file.  Without this, a "cold" benchmark run after
    any earlier run measures RAM, not disk — which has produced a green
    storage benchmark whose real-world answer was yellow.
    """
    if os.name != "nt":
        try:
            os.system("sync")
        except Exception:
            pass
        return
    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    OPEN_EXISTING = 3
    FILE_FLAG_NO_BUFFERING = 0x20000000
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = ctypes.c_void_p
    h = k32.CreateFileW(
        ctypes.c_wchar_p(path), ctypes.c_uint(GENERIC_READ),
        ctypes.c_uint(FILE_SHARE_READ | FILE_SHARE_WRITE), None,
        ctypes.c_uint(OPEN_EXISTING), ctypes.c_uint(FILE_FLAG_NO_BUFFERING), None,
    )
    if h in (None, -1, 2 ** 64 - 1):
        print(f"  [purge] CreateFileW failed ({ctypes.get_last_error()})")
        return
    k32.CloseHandle(ctypes.c_void_p(h))


# ── the retired implementation, kept verbatim for the A/B ─────────────────────

def _old_unbounded_loop(result_array, nav_shape, client, on_chunk_done):
    """The path this change replaced: one blocking ``client.compute()`` per nav
    chunk, ALL of them up front, plus a second whole-array graph at the end."""
    nav_ndim = len(nav_shape)
    trailing = (slice(None),) * (result_array.ndim - nav_ndim)
    axes_ranges = []
    for axis_chunks in result_array.chunks[:nav_ndim]:
        positions, start = [], 0
        for size in axis_chunks:
            positions.append((start, size))
            start += size
        axes_ranges.append(positions)

    futures, slices = [], []
    for combo in itertools.product(*axes_ranges):
        nav_sl = tuple(slice(s, s + n) for s, n in combo)
        futures.append(client.compute(result_array[nav_sl + trailing]))
        slices.append(nav_sl)

    for fut, nav_sl in zip(futures, slices):
        def _cb(f, _sl=nav_sl):
            try:
                on_chunk_done(f.result(), _sl)
            except Exception:
                pass
        fut.add_done_callback(_cb)

    return client.compute(result_array)          # the redundant second pass


# ── harness ───────────────────────────────────────────────────────────────────

def _load(path):
    from spyde.backend.heavy_imports import ensure_heavy_imports
    ensure_heavy_imports()
    from spyde.backend.session import Session
    t0 = time.perf_counter()
    sig = Session.load_aligned(path)
    if isinstance(sig, list):
        sig = sig[0]
    dt = time.perf_counter() - t0
    nav_dim = sig.axes_manager.navigation_dimension
    data = sig.data
    nav_dask = data.sum(axis=tuple(range(nav_dim, data.ndim)))
    print(f"  loaded in {dt:.2f} s  shape={data.shape} dtype={data.dtype} "
          f"nav_chunks={len(nav_dask.chunks[0])}")
    return sig, nav_dask


def _run_one(impl, nav_dask, client):
    from spyde.drawing.update_functions import compute_with_live_buffer

    nav_shape = tuple(nav_dask.shape)
    first = [None]
    n_chunks = [0]
    lock = threading.Lock()
    t0 = time.perf_counter()

    def _on_chunk(chunk, nav_sl):
        with lock:
            n_chunks[0] += 1
            if first[0] is None:
                first[0] = time.perf_counter() - t0

    if impl == "old":
        fut = _old_unbounded_loop(nav_dask, nav_shape, client, _on_chunk)
    else:
        fut = compute_with_live_buffer(nav_dask, nav_shape, client, shm_name="",
                                       on_chunk_done=_on_chunk)
    t_submit = time.perf_counter() - t0

    res = fut.result()
    t_total = time.perf_counter() - t0
    arr = np.asarray(res, dtype=np.float64)
    return {
        "submit": t_submit,
        "first": first[0],
        "total": t_total,
        "chunks": n_chunks[0],
        "checksum": float(np.nansum(arr)),
        "finite": int(np.isfinite(arr).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--impl", default="new", choices=["new", "old", "both"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--purge", action="store_true",
                    help="evict the file from the page cache before each run")
    ap.add_argument("--warm", action="store_true",
                    help="one throwaway pass first so both runs read from RAM")
    args = ap.parse_args()

    from dask.distributed import Client, LocalCluster
    t0 = time.perf_counter()
    cluster = LocalCluster(n_workers=args.workers, threads_per_worker=2,
                           processes=True, dashboard_address=None)
    client = Client(cluster)
    client.wait_for_workers(args.workers, timeout=120)
    info = client.scheduler_info(n_workers=-1)["workers"]
    threads = sum(int(w.get("nthreads", 1)) for w in info.values())
    print(f"cluster up in {time.perf_counter() - t0:.1f} s "
          f"({len(info)} workers, {threads} threads)")

    impls = ["old", "new"] if args.impl == "both" else [args.impl]
    results = {}
    try:
        if args.warm:
            print("warm-up pass (fills the page cache) ...")
            _sig, nav = _load(args.path)
            t = time.perf_counter()
            nav.compute()
            print(f"  warm-up nav sum {time.perf_counter() - t:.1f} s")
            del _sig, nav

        for impl in impls:
            if args.purge:
                print(f"purging page cache for {os.path.basename(args.path)} ...")
                purge_cache(args.path)
            print(f"\n=== {impl.upper()} ===")
            sig, nav = _load(args.path)
            r = _run_one(impl, nav, client)
            results[impl] = r
            print(f"  submit    {r['submit']:8.2f} s   "
                  f"(client-side, GIL held; {len(nav.chunks[0])} nav chunks)")
            print(f"  1st chunk {r['first']:8.2f} s" if r["first"] is not None
                  else "  1st chunk      n/a")
            print(f"  total     {r['total']:8.2f} s")
            print(f"  chunks streamed {r['chunks']}, finite cells "
                  f"{r['finite']}/{int(np.prod(nav.shape))}, "
                  f"checksum {r['checksum']:.6e}")
            del sig, nav

        if len(results) == 2:
            o, n = results["old"], results["new"]
            print("\n=== A/B ===")
            print(f"  submit  {o['submit']:.2f} s -> {n['submit']:.2f} s "
                  f"({o['submit'] / max(n['submit'], 1e-9):.0f}x)")
            print(f"  total   {o['total']:.2f} s -> {n['total']:.2f} s")
            same = abs(o["checksum"] - n["checksum"]) <= 1e-6 * abs(o["checksum"])
            print(f"  checksums {'MATCH' if same else 'DIFFER'}")
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
