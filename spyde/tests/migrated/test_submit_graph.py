"""
ComputeBackend.submit_graph — the cancellable, low-latency, dask-graph movie read.

The movie navigator needs cancellable async Futures (latest-position-wins scrub)
WITHOUT the distributed-scheduler round-trip. submit_graph runs a lazy dask slice
on OUR ThreadPoolExecutor with the ``synchronous`` scheduler, so we get a real
concurrent.futures.Future (cancel + done_callback) while dask does not spawn a
nested pool. The SAME path reads a lazy CROP (a pure graph op), so crop-then-scrub
just works. See repro_movie_scrub.py for the real-movie timing proof.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import dask.array as da
import hyperspy.api as hs

from spyde.compute_backend import ComputeBackend


def _movie(n=12, frame=(64, 64)):
    data = np.arange(n, dtype=np.float32).reshape(n, 1, 1) * np.ones((1,) + frame,
                                                                     dtype=np.float32)
    return da.from_array(data, chunks=(1,) + frame)


class TestSubmitGraph:
    def test_returns_materialised_frame(self):
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            be = ComputeBackend(executor=pool)
            raw = _movie()
            f = be.submit_graph(raw[5]).result()
            assert isinstance(f, np.ndarray)
            assert f.shape == (64, 64)
            assert float(f[0, 0]) == 5.0     # frame t carries value t
        finally:
            pool.shutdown(wait=False)

    def test_future_is_cancellable(self):
        # A busy single-worker pool + a queued task → the queued future cancels.
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            be = ComputeBackend(executor=pool)
            raw = _movie()
            block = threading.Event()
            # Occupy the one worker so subsequent submits queue.
            busy = pool.submit(block.wait)
            queued = [be.submit_graph(raw[i]) for i in range(3)]
            cancelled = [fu.cancel() for fu in queued]
            block.set()
            busy.result(timeout=5)
            assert all(cancelled), "queued submit_graph futures must be cancellable"
            assert all(fu.cancelled() for fu in queued)
        finally:
            pool.shutdown(wait=False)

    def test_done_callback_delivers_async(self):
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            be = ComputeBackend(executor=pool)
            raw = _movie()
            got = {}
            done = threading.Event()
            def cb(fu):
                got["v"] = float(fu.result()[0, 0]); done.set()
            be.submit_graph(raw[7]).add_done_callback(cb)
            assert done.wait(5)
            assert got["v"] == 7.0
        finally:
            pool.shutdown(wait=False)

    def test_reads_a_lazy_crop_through_same_path(self):
        # Cropping is a pure graph op; submit_graph reads the cropped view with
        # no materialise of the source.
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            be = ComputeBackend(executor=pool)
            s = hs.signals.Signal2D(_movie(n=12, frame=(64, 64))).as_lazy()
            cropped = s.inav[4:8].isig[16:48, 16:48]      # lazy: 4 frames, 32x32
            assert cropped._lazy is True
            craw = cropped.data
            f = be.submit_graph(craw[0]).result()          # first cropped frame = t=4
            assert f.shape == (32, 32)
            assert float(f[0, 0]) == 4.0
        finally:
            pool.shutdown(wait=False)

    def test_synchronous_scheduler_no_nested_pool(self):
        # Sanity: the frame computes correctly (the synchronous scheduler walks
        # the graph on the worker thread — no nested dask pool contending).
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            be = ComputeBackend(executor=pool)
            raw = _movie(n=4, frame=(8, 8))
            results = [be.submit_graph(raw[t]).result() for t in range(4)]
            assert [float(r[0, 0]) for r in results] == [0.0, 1.0, 2.0, 3.0]
        finally:
            pool.shutdown(wait=False)
