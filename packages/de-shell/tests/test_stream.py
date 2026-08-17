"""
test_stream.py — newest-wins painting, and painting from a future.

The behaviours here are the ones a hand-rolled version gets subtly wrong: a
producer faster than the renderer must not backlog, a superseded future must not
paint over a newer frame, and nothing may paint after close.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from de_shell.plotting.stream import FrameStream


class FakeView:
    """Records what it was asked to paint. `ok=False` makes show() fail."""

    def __init__(self, ok: bool = True) -> None:
        self.painted: list[np.ndarray] = []
        self.clims: list = []
        self.ok = ok

    def show(self, frame, *, clim=None):
        if not self.ok:
            raise RuntimeError("paint failed")
        self.painted.append(frame)
        self.clims.append(clim)
        return True


class Deferred:
    """Collects dispatched callables so a test drives the 'main thread'."""

    def __init__(self) -> None:
        self.queue: list = []

    def __call__(self, fn):
        self.queue.append(fn)

    def run_all(self) -> int:
        n = 0
        while self.queue:
            self.queue.pop(0)()
            n += 1
        return n


def frame(v: int) -> np.ndarray:
    return np.full((2, 2), v, dtype=np.uint16)


class TestNewestWins:
    def test_a_submitted_frame_paints_on_the_main_thread(self):
        view, main = FakeView(), Deferred()
        s = FrameStream(view, main)
        s.submit(frame(1))
        assert view.painted == [], "painted on the producer's thread"
        main.run_all()
        assert [f[0, 0] for f in view.painted] == [1]

    def test_a_burst_schedules_ONE_paint_and_shows_the_newest(self):
        # THE property this exists for. One callback per frame would back the
        # main thread up behind an ever-growing queue of stale frames.
        view, main = FakeView(), Deferred()
        s = FrameStream(view, main)
        for i in range(5):
            s.submit(frame(i))
        assert len(main.queue) == 1, "a paint was queued per frame"
        main.run_all()
        assert [f[0, 0] for f in view.painted] == [4]
        assert s.shown == 1 and s.dropped == 4

    def test_dropping_is_counted(self):
        # Dropping is correct; invisible dropping makes "it feels laggy"
        # unfalsifiable.
        view, main = FakeView(), Deferred()
        s = FrameStream(view, main)
        s.submit(frame(1)); s.submit(frame(2)); s.submit(frame(3))
        main.run_all()
        assert (s.shown, s.dropped) == (1, 2)

    def test_a_later_submission_schedules_again(self):
        view, main = FakeView(), Deferred()
        s = FrameStream(view, main)
        s.submit(frame(1)); main.run_all()
        s.submit(frame(2)); main.run_all()
        assert [f[0, 0] for f in view.painted] == [1, 2]

    def test_clim_rides_with_the_frame(self):
        view, main = FakeView(), Deferred()
        s = FrameStream(view, main)
        s.submit(frame(1), clim=(0.0, 10.0))
        main.run_all()
        assert view.clims == [(0.0, 10.0)]

    def test_on_painted_runs_only_for_a_successful_paint(self):
        seen = []
        view, main = FakeView(ok=False), Deferred()
        s = FrameStream(view, main, on_painted=seen.append)
        s.submit(frame(1))
        main.run_all()
        assert seen == [] and s.shown == 0

    def test_a_failed_paint_reports_and_does_not_raise(self):
        errs = []
        view, main = FakeView(ok=False), Deferred()
        s = FrameStream(view, main, on_error=errs.append)
        s.submit(frame(1))
        main.run_all()      # the paint fails, then the error is dispatched
        main.run_all()
        assert len(errs) == 1 and isinstance(errs[0], RuntimeError)


class TestFutures:
    def test_paints_when_the_future_completes(self):
        view, main = FakeView(), Deferred()
        s = FrameStream(view, main)
        fut: Future = Future()
        s.submit_future(fut)
        assert view.painted == []
        fut.set_result(frame(7))
        main.run_all()
        assert [f[0, 0] for f in view.painted] == [7]

    def test_a_superseded_future_is_cancelled(self):
        view, main = FakeView(), Deferred()
        s = FrameStream(view, main)
        old: Future = Future()
        new: Future = Future()
        s.submit_future(old)
        s.submit_future(new)
        assert old.cancelled(), "the superseded future was left running"

    def test_an_ALREADY_RUNNING_superseded_future_does_not_paint(self):
        # A queued future cancels; one already running cannot be stopped, so its
        # callback must no-op on the identity check instead of painting a frame
        # the user has moved past. This is the whole latest-wins guarantee.
        view, main = FakeView(), Deferred()
        s = FrameStream(view, main)
        old: Future = Future()
        old.set_running_or_notify_cancel()      # now un-cancellable
        s.submit_future(old)
        new: Future = Future()
        s.submit_future(new)

        old.set_result(frame(1))                # lands late
        new.set_result(frame(2))
        main.run_all()
        assert [f[0, 0] for f in view.painted] == [2]

    def test_a_failed_future_reports_rather_than_painting(self):
        errs = []
        view, main = FakeView(), Deferred()
        s = FrameStream(view, main, on_error=errs.append)
        fut: Future = Future()
        s.submit_future(fut)
        fut.set_exception(ValueError("compute failed"))
        main.run_all()
        assert view.painted == [] and isinstance(errs[0], ValueError)

    def test_a_cancelled_future_is_silent(self):
        errs = []
        view, main = FakeView(), Deferred()
        s = FrameStream(view, main, on_error=errs.append)
        fut: Future = Future()
        s.submit_future(fut)
        fut.cancel()
        main.run_all()
        assert view.painted == [] and errs == []

    def test_a_future_resolving_to_None_paints_nothing(self):
        # A read that found nothing to show is not an error.
        view, main = FakeView(), Deferred()
        s = FrameStream(view, main)
        fut: Future = Future()
        s.submit_future(fut)
        fut.set_result(None)
        main.run_all()
        assert view.painted == []

    def test_works_with_a_real_thread_pool(self):
        # The realistic shape: work runs on `de_shell.compute.ThreadCompute`'s
        # pool and the frame paints when it lands. The pool is shut down before
        # draining, which guarantees the done-callback has already run — polling
        # for it would be a race dressed up as a test.
        view, main = FakeView(), Deferred()
        s = FrameStream(view, main)
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            s.submit_future(pool.submit(lambda: frame(3)))
        finally:
            pool.shutdown(wait=True)
        assert main.run_all() == 1
        assert [f[0, 0] for f in view.painted] == [3]


class TestClose:
    def test_nothing_paints_after_close(self):
        view, main = FakeView(), Deferred()
        s = FrameStream(view, main)
        s.submit(frame(1))
        s.close()
        main.run_all()
        assert view.painted == []

    def test_submitting_after_close_is_a_no_op(self):
        view, main = FakeView(), Deferred()
        s = FrameStream(view, main)
        s.close()
        s.submit(frame(1))
        s.submit_future(Future())
        assert main.queue == []

    def test_close_cancels_an_outstanding_future(self):
        view, main = FakeView(), Deferred()
        s = FrameStream(view, main)
        fut: Future = Future()
        s.submit_future(fut)
        s.close()
        assert fut.cancelled()

    def test_close_is_idempotent(self):
        s = FrameStream(FakeView(), Deferred())
        s.close()
        s.close()
