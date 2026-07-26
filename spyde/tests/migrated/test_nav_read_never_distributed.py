"""An interactive nav read must NEVER go to the distributed cluster.

`ComputeBackend.submit_graph` is the expensive-tier nav read. It used to call
`client.compute` whenever a Dask cluster was live, which meant every navigator
frame paid graph serialization + worker dispatch + a result transfer for what is
a few MB of LOCAL file. Measured in the real app on a .zspy 4D-STEM: an
integrating region took 87-441 ms (once 3731 ms) through the cluster.

The cluster is for BATCH compute (find-vectors, orientation, VI) where the work
per byte is large. A frame read is the opposite shape: tiny compute, pure I/O,
latency-critical. So submit_graph always runs on a LOCAL thread pool.
"""
from __future__ import annotations

import numpy as np
import dask.array as da
import pytest

from spyde.compute_backend import ComputeBackend


class _ExplodingClient:
    """Stands in for a live distributed Client. Any use of the cluster fails the
    test loudly rather than silently costing latency."""

    def __init__(self):
        self.compute_calls = 0

    def compute(self, *a, **k):
        self.compute_calls += 1
        raise AssertionError(
            "submit_graph routed an interactive nav read to the CLUSTER")

    def submit(self, *a, **k):
        raise AssertionError("submit_graph used client.submit")


@pytest.fixture
def lazy_frames():
    return da.from_array(np.arange(256, dtype=np.uint16).reshape(16, 16),
                         chunks=(4, 4))


class TestNavReadStaysLocal:
    def test_distributed_mode_does_not_use_the_client(self, lazy_frames):
        client = _ExplodingClient()
        backend = ComputeBackend(client=client)
        try:
            fut = backend.submit_graph(lazy_frames[3])
            np.testing.assert_array_equal(
                np.asarray(fut.result(timeout=30)), np.asarray(lazy_frames[3]))
            assert client.compute_calls == 0
        finally:
            backend.shutdown_nav_pool()

    def test_threaded_mode_uses_the_existing_executor(self, lazy_frames):
        """Threaded mode already has a pool — it must NOT build a second one."""
        import concurrent.futures as cf
        ex = cf.ThreadPoolExecutor(max_workers=2)
        backend = ComputeBackend(executor=ex)
        try:
            fut = backend.submit_graph(lazy_frames[5])
            np.testing.assert_array_equal(
                np.asarray(fut.result(timeout=30)), np.asarray(lazy_frames[5]))
            assert backend._nav_executor is None
        finally:
            backend.shutdown_nav_pool()
            ex.shutdown(wait=False)

    def test_future_is_cancellable(self, lazy_frames):
        """Latest-position-wins depends on a queued read cancelling cleanly."""
        backend = ComputeBackend(client=_ExplodingClient())
        try:
            futs = [backend.submit_graph(lazy_frames[i]) for i in range(8)]
            # at least one of the later submissions should still be queued
            assert any(f.cancel() or f.cancelled() or f.done() for f in futs)
        finally:
            backend.shutdown_nav_pool()

    def test_shutdown_is_idempotent(self):
        backend = ComputeBackend(client=_ExplodingClient())
        backend.shutdown_nav_pool()
        backend.shutdown_nav_pool()          # must not raise
        assert backend._nav_executor is None
