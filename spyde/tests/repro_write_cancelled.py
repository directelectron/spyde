"""
Isolate why the signal DP's write_shared_array future ends CANCELLED on the
distributed path (worker drops cancelled futures → DP never paints).

Mirrors update_from_navigation_selection's distributed branch exactly:
  get_inds_fut = cache.get_index(..., return_future=True)   # depends on a block fut
  write_fut = client.submit(write_shared_array, get_inds_fut, shm_name, priority=10)
  current_img = write_fut   # get_inds_fut local var goes out of scope

Then we drop the get_inds reference and check write_fut's terminal status.

    .venv/Scripts/python -m spyde.tests.repro_write_cancelled
"""
from __future__ import annotations
import os, sys, time
import numpy as np


def log(m): print(f"{time.monotonic()-T0:6.1f}s {m}", file=sys.stderr, flush=True)
T0 = time.monotonic()


def main():
    from distributed import Client, LocalCluster
    from multiprocessing import shared_memory
    import dask.array as da
    import hyperspy.api as hs
    from spyde.drawing.update_functions import write_shared_array

    cluster = LocalCluster(n_workers=2, threads_per_worker=2, processes=True,
                           dashboard_address=None)
    client = Client(cluster)
    log(f"cluster up: {client.scheduler.address}")

    d = da.from_array(
        np.random.RandomState(0).randint(0, 255, (24, 24, 32, 32), dtype=np.int16),
        chunks=(8, 8, 32, 32))
    sig = hs.signals.Signal2D(d).as_lazy()

    # Pin the cache client (the spyde fix).
    sig._get_cache_dask_chunk([(5, 5)], get_result=False, return_future=True)
    sig.cached_dask_array._client = client

    # Pre-create the shm segment like Plot.shared_memory does.
    shm_name = "repro_plot_buffer"
    nbytes = 32 * 32 * 8 + 1024
    try:
        shm = shared_memory.SharedMemory(name=shm_name, create=True, size=nbytes)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=shm_name, create=False)

    def one_move(iy, ix, label):
        # EXACT real-code order: cancel_surrounding() BEFORE the chunk request.
        ca = sig.cached_dask_array
        if ca is not None and hasattr(ca, "cancel_surrounding"):
            ca.cancel_surrounding()
        get_inds_fut = sig._get_cache_dask_chunk([(iy, ix)], get_result=False,
                                                 return_future=True)
        log(f"[{label}] get_inds={type(get_inds_fut).__name__} key="
            f"{getattr(get_inds_fut,'key',None)}")
        write_fut = client.submit(write_shared_array, get_inds_fut, shm_name,
                                  priority=10)
        wkey = write_fut.key
        # Drop the get_inds reference exactly like the real code (current_img=fut).
        del get_inds_fut
        # Poll write_fut terminal status.
        t_end = time.monotonic() + 5
        while time.monotonic() < t_end:
            st = write_fut.status
            if st in ("finished", "error", "cancelled"):
                break
            time.sleep(0.05)
        log(f"[{label}] write {wkey} FINAL status={write_fut.status}")
        return write_fut.status

    # First move (cache hit on (5,5) region — block already resident).
    s1 = one_move(5, 5, "move-1 same-chunk")
    # Cross-chunk move.
    s2 = one_move(18, 18, "move-2 cross-chunk")
    # Another.
    s3 = one_move(2, 18, "move-3 cross-chunk")

    log(f"RESULTS: {s1}, {s2}, {s3}")
    try:
        shm.close(); shm.unlink()
    except Exception:
        pass
    try:
        client.close(); cluster.close()
    except Exception:
        pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
