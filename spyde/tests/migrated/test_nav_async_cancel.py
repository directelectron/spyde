"""Expensive-tier async nav read: supersede-cancel + latest-wins paint (Stage 2).

An expensive nav read (large region / cold large frame / derived view) is
submitted OFF the serial dispatcher via ComputeBackend.submit_graph and cancelled
when a newer position supersedes it, so it never blocks the navigator and only the
latest frame paints. These tests drive _submit_async_nav_read directly against a
real threaded ComputeBackend with a fake Plot/session.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import dask.array as da

from spyde.compute_backend import ComputeBackend
from spyde.drawing.update_functions import _submit_async_nav_read


class _FakeAxesManager:
    def __init__(self, nav_dim):
        self.navigation_dimension = nav_dim


class _FakeSignal:
    def __init__(self, data, nav_dim):
        self.data = data
        self._lazy = True
        self.axes_manager = _FakeAxesManager(nav_dim)


class _FakeSession:
    def __init__(self, backend, inline=True):
        self._backend = backend
        self._inline = inline
        self._queued = []

    @property
    def compute_backend(self):
        return self._backend

    def _dispatch_to_main(self, fn):
        # Inline apply (like early startup / tests) — the callback marshals here.
        if self._inline:
            fn()
        else:
            self._queued.append(fn)

    def drain(self):
        while self._queued:
            self._queued.pop(0)()


class _FakePlot:
    def __init__(self, session):
        self.session = session
        self._nav_future = None
        self.current_data = None
        self.painted = []

    def update(self):
        self.painted.append(np.asarray(self.current_data).copy())


class _Prof:
    def done(self, *a, **k):
        pass


def _movie(n=64, frame=(8, 8), seed=0):
    arr = np.random.RandomState(seed).rand(n, *frame).astype(np.float32)
    return da.from_array(arr, chunks=(1, -1, -1)), arr


class TestAsyncNavRead:
    def test_single_frame_paints_correct_data(self):
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            be = ComputeBackend(executor=pool)
            sess = _FakeSession(be)
            plot = _FakePlot(sess)
            data, arr = _movie()
            sig = _FakeSignal(data, nav_dim=1)
            armed = _submit_async_nav_read(plot, sig, np.array([7]), False, _Prof())
            assert armed is True
            # Wait for the async paint.
            end = time.monotonic() + 3.0
            while time.monotonic() < end and not plot.painted:
                time.sleep(0.01)
            assert plot.painted, "async read never painted"
            np.testing.assert_array_equal(plot.painted[-1], arr[7])
            assert plot._nav_future is None  # cleared after paint
        finally:
            pool.shutdown(wait=True)

    def test_region_mean_matches_numpy(self):
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            be = ComputeBackend(executor=pool)
            sess = _FakeSession(be)
            plot = _FakePlot(sess)
            data, arr = _movie()
            sig = _FakeSignal(data, nav_dim=1)
            pts = np.array([[i] for i in range(3, 12)])  # region of 9 frames
            armed = _submit_async_nav_read(plot, sig, pts, False, _Prof())
            assert armed is True
            end = time.monotonic() + 3.0
            while time.monotonic() < end and not plot.painted:
                time.sleep(0.01)
            assert plot.painted
            expected = arr[3:12].mean(axis=0)
            np.testing.assert_allclose(plot.painted[-1], expected, rtol=1e-5)
        finally:
            pool.shutdown(wait=True)

    def test_int_region_mean_is_rounded(self):
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            be = ComputeBackend(executor=pool)
            sess = _FakeSession(be)
            plot = _FakePlot(sess)
            arr = np.random.RandomState(1).randint(0, 1000, (32, 8, 8)).astype(np.uint16)
            data = da.from_array(arr, chunks=(1, -1, -1))
            sig = _FakeSignal(data, nav_dim=1)
            pts = np.array([[i] for i in range(0, 5)])
            _submit_async_nav_read(plot, sig, pts, False, _Prof())
            end = time.monotonic() + 3.0
            while time.monotonic() < end and not plot.painted:
                time.sleep(0.01)
            assert plot.painted
            expected = np.rint(arr[0:5].mean(axis=0)).astype(np.uint16)
            np.testing.assert_array_equal(plot.painted[-1], expected)
            assert plot.painted[-1].dtype == np.uint16
        finally:
            pool.shutdown(wait=True)

    def test_supersede_cancels_prior_future(self):
        """A newer submission cancels the prior in-flight future and only the
        latest position paints. Block the pool so the first future stays queued,
        then submit a second — the first must be cancelled."""
        pool = ThreadPoolExecutor(max_workers=1)  # 1 worker → 2nd submit queues
        gate = threading.Event()
        try:
            be = ComputeBackend(executor=pool)
            sess = _FakeSession(be)
            plot = _FakePlot(sess)
            data, arr = _movie()
            sig = _FakeSignal(data, nav_dim=1)

            # Occupy the single pool worker with an unrelated blocking task so the
            # nav futures must queue (and can be cancelled).
            blocker = pool.submit(lambda: gate.wait(2.0))

            _submit_async_nav_read(plot, sig, np.array([1]), False, _Prof())
            first = plot._nav_future
            _submit_async_nav_read(plot, sig, np.array([2]), False, _Prof())
            second = plot._nav_future

            assert first is not second
            assert first.cancelled(), "prior nav future was not cancelled on supersede"

            gate.set()
            blocker.result(timeout=2.0)
            # Only the latest (frame 2) paints.
            end = time.monotonic() + 3.0
            while time.monotonic() < end and not plot.painted:
                time.sleep(0.01)
            assert plot.painted, "latest async read never painted"
            np.testing.assert_array_equal(plot.painted[-1], arr[2])
            # The cancelled first future's callback must not have painted frame 1.
            for f in plot.painted:
                assert not np.array_equal(f, arr[1]), "cancelled frame 1 painted"
        finally:
            gate.set()
            pool.shutdown(wait=True)

    def test_settle_flag_requests_full_res(self):
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            be = ComputeBackend(executor=pool)
            sess = _FakeSession(be)
            plot = _FakePlot(sess)
            data, arr = _movie()
            sig = _FakeSignal(data, nav_dim=1)
            _submit_async_nav_read(plot, sig, np.array([9]), True, _Prof())
            end = time.monotonic() + 3.0
            while time.monotonic() < end and not plot.painted:
                time.sleep(0.01)
            assert plot.painted
        finally:
            pool.shutdown(wait=True)

    def test_failed_read_releases_slot(self):
        """If the async compute raises, _apply must RELEASE _nav_future (not wedge
        it on a dead future) so the next read's supersede-cancel still works, and
        it must NOT paint. current_data is never set to the future, so a later
        repaint isn't stuck on a non-ndarray."""
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            be = ComputeBackend(executor=pool)
            sess = _FakeSession(be)
            plot = _FakePlot(sess)
            # A lazy array whose COMPUTE raises (simulates a torn/failed read).
            # dtype= is given so map_blocks doesn't eagerly probe _boom at build.
            def _boom(_block):
                raise RuntimeError("simulated read failure")
            data = da.from_array(
                np.zeros((16, 8, 8), np.float32), chunks=(1, -1, -1)
            ).map_blocks(_boom, dtype=np.float32)
            sig = _FakeSignal(data, nav_dim=1)
            armed = _submit_async_nav_read(plot, sig, np.array([3]), False, _Prof())
            assert armed is True
            # The future resolves (with an exception); wait for the callback.
            end = time.monotonic() + 3.0
            while time.monotonic() < end and plot._nav_future is not None:
                time.sleep(0.01)
            assert plot._nav_future is None, "failed read wedged the _nav_future slot"
            assert not plot.painted, "a failed read must not paint"
            # current_data must NOT be a Future (it was never set to one).
            from concurrent.futures import Future as _CF
            assert not isinstance(plot.current_data, _CF)
        finally:
            pool.shutdown(wait=True)

    def test_current_data_not_set_to_future(self):
        """The async submit must NOT park a Future in current_data (only the
        eventual ndarray), so an interleaved non-nav repaint never no-ops on a
        Future."""
        pool = ThreadPoolExecutor(max_workers=1)
        gate = threading.Event()
        try:
            be = ComputeBackend(executor=pool)
            sess = _FakeSession(be)
            plot = _FakePlot(sess)
            data, arr = _movie()
            sig = _FakeSignal(data, nav_dim=1)
            blocker = pool.submit(lambda: gate.wait(2.0))  # keep the read queued
            _submit_async_nav_read(plot, sig, np.array([5]), False, _Prof())
            from concurrent.futures import Future as _CF
            # While the read is still in flight, current_data must not be a Future.
            assert not isinstance(plot.current_data, _CF)
            gate.set()
            blocker.result(timeout=2.0)
        finally:
            gate.set()
            pool.shutdown(wait=True)


class TestComputeBackendShutdownGuard:
    def test_backend_none_after_shutdown_no_executor_recreate(self):
        """After shutdown(), compute_backend must return None and must NOT lazily
        recreate the nav executor — a queued dispatcher/settle update firing post
        teardown would otherwise leak a fresh ThreadPoolExecutor."""
        import os
        os.environ.setdefault("SPYDE_NO_DASK", "1")
        from spyde.backend.session import Session
        sess = Session(n_workers=1, threads_per_worker=1)
        # Touch the property so the threaded executor is created (no cluster).
        be = sess.compute_backend
        assert be is not None
        assert sess._nav_executor is not None
        sess.shutdown()
        # Post-shutdown: property returns None and does NOT respawn the executor.
        assert sess.compute_backend is None
        assert sess._nav_executor is None
