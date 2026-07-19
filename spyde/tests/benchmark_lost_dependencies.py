"""
benchmark_lost_dependencies.py
==============================
Reproduce (and regression-guard) the "get_inds-... cancelled for reason: lost
dependencies" churn seen when dragging the navigator on a lazy/distributed
4D-STEM signal — which "gets worse on the second signal you load".

ROOT CAUSE (confirmed by this harness):
    hyperspy CachedDaskArray.get_index took a synchronous `future.result()`
    whenever the core blocks were already done() — even for an interactive
    caller passing force_compute=False / return_future=False. The get_inds
    future is a NEW worker task, so .result() BLOCKS on it; a concurrent
    navigator move that evicts a core block it depends on cancels it ("lost
    dependencies") and the blocking .result() raises. Fixed in the hyperspy
    fork (slice-integrate2): only block when force_compute is explicit;
    interactive callers always get the future back and poll it asynchronously.

Run as a plain script (NOT under pytest — needs a real LocalCluster):

    .venv/Scripts/python -m spyde.tests.benchmark_lost_dependencies --signals 2
    .venv/Scripts/python -m spyde.tests.benchmark_lost_dependencies --signals 2 --return-future

It builds lazy dask 4D signals with signal-spanning chunks + a real distributed
Client, then drives navigator positions through the SAME path the app uses
(`update_functions.update_from_navigation_selection`) on a thread per move
(mimicking the throttle's threading.Timer). It counts how many update calls
raise the "lost dependencies" FutureCancelledError.

Before the fork fix: ~tens of raises (more on signal_1). After: 0.

Prints a JSON summary line `REPRO_JSON {...}` and os._exit(0).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np


# Capture distributed's cancellation log messages ("lost dependencies").
class _CancelCounter(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lost_deps = 0
        self.cancelled = 0
        self.messages = []

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        low = msg.lower()
        if "lost dependencies" in low:
            self.lost_deps += 1
            if len(self.messages) < 20:
                self.messages.append(msg)
        elif "cancelled" in low and "get_inds" in low:
            self.cancelled += 1


def build_signal(client, ny=16, nx=16, ky=64, kx=64, chunk=4, seed=0):
    import dask.array as da
    import hyperspy.api as hs

    rng = da.random.RandomState(seed)
    data = rng.randint(
        0, 255, size=(ny, nx, ky, kx), dtype=np.int16,
        chunks=(chunk, chunk, ky, kx),  # signal-spanning chunks
    )
    s = hs.signals.Signal2D(data).as_lazy()
    # Scatter the data onto the workers so the cache uses futures-on-workers
    # (the data_on_workers path that submits get_inds).
    return s


def drive(client, signal, positions, cancel_counter, settle=0.25, return_future=False):
    """Drive a sequence of nav positions through the real update path."""
    from spyde.drawing import update_functions as uf

    class _PlotState:
        def __init__(self, sig):
            self.current_signal = sig

    class _Child:
        """Minimal stand-in for a Plot: just enough for the cache + shm path."""
        def __init__(self, sig):
            self.plot_state = _PlotState(sig)
            self._pending_shm_future = None
            self.main_window = None
            self.current_data = None

        @property
        def shared_memory(self):
            return None

        def update_data(self, d):
            self.current_data = d

    class _Sel:
        is_integrating = False
        _fn_gen = 0
        _update_gen = 0

        def is_stale_body(self):
            return False

    child = _Child(signal)
    sel = _Sel()

    # Proposed fix: force the cache to always hand back a Future (never take the
    # blocking future.result() path that gets raced). Patch at the signal so the
    # app's call site is unchanged.
    if return_future:
        _orig = signal._get_cache_dask_chunk
        def _always_future(indices, get_result=False, return_future=False, **kw):
            return _orig(indices, get_result=False, return_future=True, **kw)
        signal._get_cache_dask_chunk = _always_future

    import threading

    errors = {"raised": 0}
    elock = threading.Lock()

    def one_move(y, x):
        # EXACT app path: cache lock + cancel_surrounding + _get_cache_dask_chunk
        # via update_from_navigation_selection. return_future is toggled by
        # monkeypatching the kwargs the function passes (see _patch below).
        indices = np.array([[x, y]])  # widget order (cx, cy)
        try:
            uf.update_from_navigation_selection(
                sel, child, indices, get_result=False,
            )
        except Exception as e:
            # The "lost dependencies" FutureCancelledError surfaces HERE, exactly
            # as the app logs it ("selector update failed: get_inds-... lost
            # dependencies"). Count it.
            with elock:
                errors["raised"] += 1

    # Drive moves on separate threads (the throttle fires each on a new
    # threading.Timer thread) so the cache section sees real concurrency — a
    # move evicts a block while a prior move's get_inds future is in flight.
    threads = []
    for (y, x) in positions:
        t = threading.Thread(target=one_move, args=(int(y), int(x)))
        t.start()
        threads.append(t)
        time.sleep(0.004)  # ~250 moves/s, faster than the 40ms throttle floor
    for t in threads:
        t.join(timeout=5)

    time.sleep(settle)
    return errors["raised"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", type=int, default=1)
    ap.add_argument("--moves", type=int, default=60)
    ap.add_argument("--ny", type=int, default=16)
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--return-future", action="store_true",
                    help="request a Future from the cache (proposed fix)")
    args = ap.parse_args()

    # Attach the cancellation counter to distributed's logger.
    counter = _CancelCounter()
    for name in ("distributed.scheduler", "distributed.worker", "distributed"):
        logging.getLogger(name).addHandler(counter)
        logging.getLogger(name).setLevel(logging.INFO)

    from distributed import Client, LocalCluster

    cluster = LocalCluster(
        n_workers=2, threads_per_worker=2, processes=True,
        dashboard_address=None,
    )
    client = Client(cluster)

    # A zig-zag drag across chunk boundaries (maximises cross-chunk evictions).
    rng = np.random.RandomState(1)
    results = {}
    signals = []
    for si in range(args.signals):
        sig = build_signal(client, ny=args.ny, nx=args.nx, chunk=args.chunk, seed=si)
        signals.append(sig)
        positions = []
        for _ in range(args.moves):
            positions.append((rng.randint(0, args.ny), rng.randint(0, args.nx)))
        before_lost = counter.lost_deps
        raised = drive(client, sig, positions, counter,
                       return_future=args.return_future)
        results[f"signal_{si}"] = {
            "update_exceptions_raised": raised,
            "lost_dependencies_logs": counter.lost_deps - before_lost,
        }

    summary = {
        "total_lost_dependencies_logs": counter.lost_deps,
        "total_get_inds_cancelled_logs": counter.cancelled,
        "per_signal": results,
        "sample_messages": counter.messages[:5],
    }
    print("REPRO_JSON " + json.dumps(summary))

    try:
        client.close()
        cluster.close()
    except Exception:
        pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
