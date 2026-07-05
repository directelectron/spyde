"""
repro_batch_stall.py — standalone repro for the "chunk tasks sit unscheduled
until any client RPC arrives" stall (and the exactly-60.0s LocalCluster build).

App-side evidence (Playwright probe `_probe_fv_stall.spec.ts`): submit the
find-vectors batch, touch nothing → tasks stay {waiting, processing} with all
workers idle indefinitely; fire ONE client round-trip at t=X → compute done at
t=X+4s, for X=10 and X=60. Scheduler answers RPCs in milliseconds throughout —
its timer-driven work (batched-send flushes etc.) simply never fires without
incoming I/O.

This script mimics the app EXACTLY (same cluster construction as DaskManager,
per-chunk client.compute from a worker thread, mild worker restrictions) but
WITHOUT Electron. Outcomes:

  * "STALL REPRODUCED" here → distributed/env problem on this box (report
    upstream with this script).
  * completes promptly here → the pathology is specific to the
    Electron-spawned backend process (stdio/env/loop interaction) — dig there.

Run it yourself (real LocalCluster(processes=True) — not under pytest/agents):

    uv run python -m spyde.tests.repro_batch_stall
"""
from __future__ import annotations

import sys
import threading
import time


def main() -> int:
    import numpy as np
    import dask.array as da
    from dask.distributed import Client, LocalCluster

    t0 = time.monotonic()

    def log(msg):
        print(f"[{time.monotonic() - t0:7.1f}s] {msg}", flush=True)

    log("building LocalCluster(n_workers=0) …")
    t = time.monotonic()
    cluster = LocalCluster(n_workers=0, threads_per_worker=1)
    log(f"scheduler up in {time.monotonic() - t:.1f}s")
    client = Client(cluster)
    cluster.scale(4)
    client.wait_for_workers(4, timeout=120)
    log("4 workers up")
    workers = list(client.scheduler_info(n_workers=-1)["workers"].keys())

    # A small chunked compute, submitted per chunk from a WORKER THREAD with a
    # worker restriction — the exact shape of the find-vectors batch dispatch.
    arr = da.random.random((8, 8, 64, 64), chunks=(4, 4, 64, 64))
    sums = [arr[i:i + 4, j:j + 4].sum() for i in (0, 4) for j in (0, 4)]

    futures = []
    submitted = threading.Event()

    def _submit():
        futs = client.compute(sums, workers=workers[1:], allow_other_workers=True)
        futures.extend(futs)
        submitted.set()

    threading.Thread(target=_submit, daemon=True, name="repro-submit").start()
    assert submitted.wait(30), "submission thread never returned"
    log(f"submitted {len(futures)} chunk futures from a worker thread")

    # Phase 1: NO client traffic — poll future.done() locally only.
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if all(f.done() for f in futures):
            log("ALL FUTURES DONE with zero client traffic — stall NOT reproduced")
            client.close(); cluster.close()
            return 0
        time.sleep(0.5)
    log("45s with zero traffic: futures still pending — firing ONE client RPC …")

    # Phase 2: one RPC — in the app this unsticks everything within ~4 s.
    client.scheduler_info(n_workers=-1)
    t_poke = time.monotonic()
    while time.monotonic() - t_poke < 30:
        if all(f.done() for f in futures):
            log(f"STALL REPRODUCED: done {time.monotonic() - t_poke:.1f}s after "
                "the poke (was stuck 45s without it)")
            client.close(); cluster.close()
            return 2
        time.sleep(0.5)
    log("still pending 30s after the poke — different failure mode")
    client.close(); cluster.close()
    return 3


if __name__ == "__main__":
    sys.exit(main())
