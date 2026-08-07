"""
benchmark_nav_submit_batching.py — how many times do we call the scheduler?

Run directly::

    python -m spyde.tests.benchmark_nav_submit_batching
    python -m spyde.tests.benchmark_nav_submit_batching --chunks 977

The navigator fill on a big dataset looked like "the cluster is using one
worker". It was not a placement problem: `dispatch_chunks` tops up on EVERY
completion, so on the unpinned lane `lane_cap - outstanding` is 1 in steady
state and the batch size collapsed to ONE task per submit. `submit_batch=8`
only ever applied to the first fill.

Each of those submits is a blocking scheduler round trip with the GIL held in
the client process, so the client — not the cluster — became the bottleneck,
and it got worse with more chunks. Hence "only big datasets".

What this measures, and why not just wall-clock: on a synthetic cluster the
per-task WORK is trivial, so total time understates the problem. The honest
metric is **how many times we call `client.compute`** (each is a round trip
that scales with graph size), plus time-to-first-chunk, which is what the user
actually watches.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import dask.array as da
    from dask.distributed import Client, LocalCluster
    import spyde.compute_dispatch as cd

    cluster = LocalCluster(n_workers=a.workers, threads_per_worker=a.threads,
                           processes=True, dashboard_address=None, silence_logs=50)
    client = Client(cluster)
    client.wait_for_workers(a.workers)
    print(f"cluster: {a.workers} workers x {a.threads} threads   "
          f"nav chunks: {a.chunks}\n")

    # A 1-D navigator sum, one task per chunk — the movie shape.
    arr = da.zeros((a.chunks, 64, 64), chunks=(1, 64, 64), dtype=np.float32)
    nav = arr.sum(axis=(1, 2))

    out: dict = {"chunks": a.chunks, "workers": a.workers, "threads": a.threads}

    # `submit_batch=1` reproduces the OLD steady-state behaviour exactly: the
    # window refilled one task at a time. Comparing against it isolates the
    # batching change from everything else in the dispatcher.
    for label, kw in (("one-at-a-time (the bug)", dict(batch_unpinned=False, cap=8)),
                      ("batched (fixed)", dict())):
        calls = {"n": 0}
        real = client.compute

        def counting(x, *args, **kwargs):
            if isinstance(x, (list, tuple)):
                calls["n"] += 1
            return real(x, *args, **kwargs)

        client.compute = counting
        first = {"t": None}
        t0 = time.perf_counter()

        def assemble(res, sl, val):
            if first["t"] is None:
                first["t"] = time.perf_counter() - t0
            res[sl] = val

        try:
            cd.dispatch_chunks(client, nav, 1, [], None, assemble=assemble,
                               fill_value=np.nan, label=label,
                               lane_default_mode="off", **kw)
        finally:
            client.compute = real
        el = time.perf_counter() - t0
        print(f"  {label:24}  submits={calls['n']:>4}  "
              f"first-chunk={first['t'] * 1e3:>6.0f} ms  total={el:>6.2f}s")
        out[label] = {"submits": calls["n"], "first_ms": first["t"] * 1e3,
                      "total_s": el}

    a_, b_ = out["one-at-a-time (the bug)"], out["batched (fixed)"]
    print(f"\n  submits: {a_['submits']} -> {b_['submits']}  "
          f"({a_['submits'] / max(b_['submits'], 1):.0f}x fewer round trips)")
    print("  NB each saved round trip is ~14 ms with the GIL held on a real "
          "graph, which a synthetic graph does not charge us.")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=float)
    client.close()
    cluster.close()
    sys.stdout.flush()      # _exit skips stdio flushing
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
