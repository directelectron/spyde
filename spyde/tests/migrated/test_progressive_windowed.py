"""The windowed VI/FFT progressive compute (``compute_with_live_buffer``).

The original path submitted every nav chunk at once PLUS a duplicate full-graph
future, and an ROI move could not cancel the per-chunk futures ("the VI is a
completely wild dask task": whole-dataset prefetch → spill, zombie computes
stacking per drag tick). It then grew its own bounded submit loop, and the
navigator fill grew a different one — both are now the SHARED dispatcher
(``compute_dispatch.dispatch_chunks``; see ``test_chunk_dispatch_guard.py``).

These tests pin the contract that must survive that move, with a fake client:
bounded in-flight futures, per-chunk callbacks, client-side assembly, and a
cancel that actually stops everything.
"""
from __future__ import annotations

import threading
import time

import dask.array as da
import numpy as np


def _submitted_and_armed(client, at_least: int) -> bool:
    """``at_least`` futures exist AND every one of them has its done-callback.

    ``dispatch_chunks`` appends the futures FIRST and registers the callbacks in
    a following loop, so a predicate on the COUNT alone can wake inside that
    gap. Firing a future whose ``_cbs`` is still empty drops the completion on
    the floor: the bulk submit never goes out and the test times out five
    seconds later, looking like a dispatcher bug.
    """
    futs = list(client.created)
    return len(futs) >= at_least and all(f._cbs for f in futs)


def _wait(pred, timeout=10.0, required=True):
    t0 = time.monotonic()
    evt = threading.Event()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        evt.wait(0.005)
    if required:
        raise AssertionError("timed out waiting for condition")
    return False


class _FakeFuture:
    """Manually-fired future so the test controls completion order."""

    def __init__(self, value_fn):
        self._value_fn = value_fn
        self._cbs = []
        self.cancelled = False
        self._done = False

    def add_done_callback(self, cb):
        self._cbs.append(cb)

    @property
    def fired(self):
        """Already completed. The tests fire ONE future to release the bulk
        submit and then fire "the rest"; without this guard the first one runs
        its done-callback twice and the assembly counts 17 chunks out of 16."""
        return self._done

    def done(self):
        return self._done

    def result(self):
        if self.cancelled:
            raise RuntimeError("cancelled")
        return self._value_fn()

    def cancel(self):
        self.cancelled = True

    def release(self):
        pass

    def fire(self):
        self._done = True
        for cb in self._cbs:
            cb(self)


class _FakeClient:
    def __init__(self):
        self.created: list[_FakeFuture] = []
        self.submit_calls = 0

    def scheduler_info(self, n_workers=None):
        # 2 threads → unpinned window = max(4, 2 // 2) = 4
        return {"workers": {"a": {"nthreads": 2}}}

    def _make(self, arr):
        fut = _FakeFuture(lambda a=arr: a.compute(scheduler="synchronous"))
        self.created.append(fut)
        return fut

    def compute(self, arrays, **kwargs):
        self.submit_calls += 1
        if isinstance(arrays, (list, tuple)):
            return [self._make(a) for a in arrays]
        return self._make(arrays)

    def run_on_scheduler(self, *a, **k):
        return {}

    def call_stack(self, *a, **k):
        return {}


def _run(nav=(8, 8), chunks=(2, 2), stop_event=None):
    from spyde.drawing.update_functions import compute_with_live_buffer
    src = da.from_array(np.arange(np.prod(nav), dtype=np.float32).reshape(nav),
                        chunks=chunks)
    client = _FakeClient()
    chunks_seen = []
    handle = compute_with_live_buffer(
        src, nav, client, shm_name="",
        on_chunk_done=lambda res, sl: chunks_seen.append((res, sl)),
        windowed=True, stop_event=stop_event,
    )
    # The dispatcher runs on its own thread — wait for the first window to be
    # submitted AND armed (see _submitted_and_armed).
    _wait(lambda: _submitted_and_armed(client, 4))
    return src, client, handle, chunks_seen


class TestVirtualImageMaskedSum:
    def test_einsum_path_matches_product_reference(self, monkeypatch):
        """The per-chunk einsum contraction (no ``data*mask`` product
        intermediate — the VI spill fix) must equal the old broadcast-product
        reduce for sum AND mean, on lazy uint16 data and the numpy fast-path."""
        import hyperspy.api as hs
        import spyde.actions.virtual_image as vi_mod

        rng = np.random.default_rng(0)
        raw = (rng.random((3, 4, 8, 8)) * 1000).astype(np.uint16)
        mask = (rng.random((8, 8)) > 0.4).astype(np.float32)
        monkeypatch.setattr(vi_mod, "widget_to_mask", lambda w, s: mask)

        class _Sel:
            roi = object()

        inst = object.__new__(vi_mod.VirtualImageAction)
        ref = (raw.astype(np.float64) * mask).sum(axis=(-2, -1))

        sig = hs.signals.Signal2D(raw).as_lazy()
        sig.data = sig.data.rechunk((2, 2, 8, 8))     # full-frame chunks
        out_sum = inst._virtual_image_array(sig, _Sel(), calculation="sum")
        np.testing.assert_allclose(out_sum.compute(), ref, rtol=1e-4)
        out_mean = inst._virtual_image_array(sig, _Sel(), calculation="mean")
        np.testing.assert_allclose(out_mean.compute(), ref / mask.sum(), rtol=1e-4)

        out_np = inst._virtual_image_array(hs.signals.Signal2D(raw), _Sel(),
                                           calculation="sum")
        np.testing.assert_allclose(np.asarray(out_np), ref, rtol=1e-4)


class TestWindowedProgressive:
    def test_primes_small_then_sends_the_rest_in_one_submit(self):
        """The UNPINNED lane submits in TWO calls: a small priming batch, then
        everything else.

        It used to keep a bounded in-flight window and top up on every
        completion — which meant `lane_cap - outstanding` was 1 in steady state
        and the batch collapsed to ONE task per submit, i.e. a blocking
        GIL-held round trip per chunk. Measured on 977 chunks: 970 submits,
        12.4 s. There is nothing to balance on an unpinned lane (no placement
        decision), so the window bought nothing; `distributed` queues root
        tasks at the scheduler anyway.

        The priming batch is why this is two calls and not one: going straight
        to a single all-at-once submit doubled time-to-first-chunk (642 ->
        1292 ms) because the client serialises the whole graph before anything
        returns. Priming keeps the first paint fast — measured 45 ms, better
        than the windowed version's 656 ms — for one extra round trip.
        """
        src, client, handle, seen = _run()
        # The priming batch only — the bulk submit follows the first completion
        # (or the wait loop's next top-up), which is what keeps first paint fast.
        primed = len(client.created)
        assert primed <= 8, f"priming batch was {primed}, expected <= 8"
        client.created[0].fire()
        _wait(lambda: len(client.created) == 16, timeout=5.0)
        assert client.submit_calls <= 2, (
            f"{client.submit_calls} submits for 16 chunks — the unpinned lane "
            f"is back to per-completion top-ups")
        for f in list(client.created):
            if not f.fired:
                f.fire()
        np.testing.assert_array_equal(handle.result(timeout=10.0), src.compute())
        assert len(seen) == 16

    def test_cancel_cancels_every_outstanding_chunk(self):
        """Cancellation still stops the work.

        With the whole job submitted up front there is no pending queue left to
        withhold, so "stop" means CANCELLING the outstanding futures rather than
        declining to submit more. That is the property that actually matters —
        a superseded VI stream must not keep computing through an ROI drag —
        and dask cancels queued tasks it has not started.
        """
        src, client, handle, seen = _run()
        client.created[0].fire()          # let the bulk submit go out
        _wait(lambda: len(client.created) == 16, timeout=5.0)
        handle.cancel()
        # Cancel is IMMEDIATE (dispatch_chunks' on_start hook wakes the wait
        # loop) — a superseded VI stream must not keep computing through an ROI
        # drag while the next tick's stream is already running.
        # Every future that had NOT already completed. The one fired above to
        # release the bulk submit is done, and a completed future is not
        # cancellable — asserting "all cancelled" would just be wrong.
        _wait(lambda: all(f.cancelled for f in client.created if not f.fired))
        # Firing the cancelled futures must not submit more work.
        for f in list(client.created):
            if not f.fired:
                f.fire()
        _wait(handle.done)
        assert len(client.created) == 16, "firing cancelled futures resubmitted work"

    def test_stop_event_halts_topups(self):
        stop = threading.Event()
        src, client, handle, seen = _run(stop_event=stop)
        stop.set()
        for f in list(client.created):
            f.fire()
        _wait(handle.done)
        # No BULK submit after the stop: only the priming batch was ever sent.
        # (It was 4 when the lane kept a window; the priming batch is 8.)
        assert len(client.created) <= 8, "work was submitted after the stop"

    def test_error_propagates_via_result(self):
        from spyde.drawing.update_functions import compute_with_live_buffer
        src = da.from_array(np.ones((4, 4), np.float32), chunks=(2, 2))

        class _BoomClient(_FakeClient):
            def _make(self, arr):
                fut = _FakeFuture(lambda: (_ for _ in ()).throw(ValueError("boom")))
                self.created.append(fut)
                return fut

        client = _BoomClient()
        handle = compute_with_live_buffer(src, (4, 4), client, shm_name="",
                                          windowed=True)
        _wait(lambda: _submitted_and_armed(client, 4))
        for f in list(client.created):
            f.fire()
        _wait(handle.done)
        try:
            handle.result(timeout=5.0)
            assert False, "error did not propagate"
        except ValueError:
            pass
