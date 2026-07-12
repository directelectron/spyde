"""
Does the DP signal's CachedDaskArray resolve the DISTRIBUTED client when the
Client was created on a DIFFERENT (background) thread — as in the real app, where
DaskManager builds the Client on the 'dask-startup' thread and the navigator
update runs on a threading.Timer thread?

The app trace showed `cache.client=THREADED/none` → the cache silently used the
threaded scheduler for DP updates. This isolates whether get_client()'s ambient
lookup fails across threads when as_default isn't the calling thread.

    .venv/Scripts/python -m spyde.tests.repro_cache_client_thread
"""
from __future__ import annotations
import os, sys, threading, time
import numpy as np


def log(m): print(f"{time.monotonic()-T0:6.1f}s {m}", file=sys.stderr, flush=True)
T0 = time.monotonic()


def make_lazy(seed=0):
    import dask.array as da
    import hyperspy.api as hs
    d = da.from_array(
        np.random.RandomState(seed).randint(0, 100, (24, 24, 32, 32), dtype=np.int16),
        chunks=(8, 8, 32, 32),
    )
    return hs.signals.Signal2D(d).as_lazy()


def cache_client_kind(sig):
    from distributed import Future
    res = sig._get_cache_dask_chunk([(5, 5)], get_result=False, return_future=True)
    cache = sig.cached_dask_array
    cli = cache.client
    kind = ("distributed" if cli is not None and type(cli).__name__ == "Client"
            else ("THREADED/none" if cli is None else type(cli).__name__))
    return kind, type(res).__name__


def main():
    from distributed import Client, LocalCluster

    result = {}

    def build_cluster_on_bg_thread():
        # EXACTLY like DaskManager._run: Client created on a NON-main thread.
        cl = LocalCluster(n_workers=2, threads_per_worker=2, processes=True,
                          dashboard_address=None)
        c = Client(cl)
        result["client"] = c
        result["cluster"] = cl
        log(f"bg thread: client created, as_default default. addr={c.scheduler.address}")

    t = threading.Thread(target=build_cluster_on_bg_thread, name="dask-startup")
    t.start(); t.join()
    client = result["client"]

    log("=== MAIN thread cache lookup ===")
    k, r = cache_client_kind(make_lazy(0))
    log(f"  main-thread: cache.client={k} chunk_result={r}")

    log("=== threading.Timer thread cache lookup (like the selector fire) ===")
    out = {}
    def on_timer():
        out["k"], out["r"] = cache_client_kind(make_lazy(1))
    timer = threading.Timer(0.01, on_timer)
    timer.start(); timer.join()
    log(f"  timer-thread: cache.client={out['k']} chunk_result={out['r']}")

    log("=== fresh plain worker thread cache lookup ===")
    out2 = {}
    def worker():
        out2["k"], out2["r"] = cache_client_kind(make_lazy(2))
    w = threading.Thread(target=worker, name="some-worker")
    w.start(); w.join()
    log(f"  worker-thread: cache.client={out2['k']} chunk_result={out2['r']}")

    # Does setting _client explicitly fix it (the candidate fix)?
    log("=== with cache._client set explicitly, from timer thread ===")
    out3 = {}
    def on_timer2():
        sig = make_lazy(3)
        sig._get_cache_dask_chunk([(5, 5)], get_result=False, return_future=True)
        sig.cached_dask_array._client = client   # the fix
        k, r = cache_client_kind(sig)
        out3["k"], out3["r"] = k, r
    timer2 = threading.Timer(0.01, on_timer2)
    timer2.start(); timer2.join()
    log(f"  timer+explicit _client: cache.client={out3['k']} chunk_result={out3['r']}")

    try:
        client.close(); result["cluster"].close()
    except Exception:
        pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
