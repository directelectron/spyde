"""The windowed VI/FFT progressive compute (update_functions._windowed_progressive).

The old path submitted every nav chunk at once PLUS a duplicate full-graph
future, and an ROI move could not cancel the per-chunk futures ("the VI is a
completely wild dask task": whole-dataset prefetch → spill, zombie computes
stacking per drag tick). These tests pin the new contract with a fake client:
bounded in-flight futures, per-chunk callbacks, client-side assembly, and a
cancel that actually stops everything.
"""
from __future__ import annotations

import threading

import dask.array as da
import numpy as np


class _FakeFuture:
    """Manually-fired future so the test controls completion order."""

    def __init__(self, value_fn):
        self._value_fn = value_fn
        self._cbs = []
        self.cancelled = False

    def add_done_callback(self, cb):
        self._cbs.append(cb)

    def result(self):
        if self.cancelled:
            raise RuntimeError("cancelled")
        return self._value_fn()

    def cancel(self):
        self.cancelled = True

    def fire(self):
        for cb in self._cbs:
            cb(self)


class _FakeClient:
    def __init__(self):
        self.created: list[_FakeFuture] = []

    def scheduler_info(self, n_workers=None):
        return {"workers": {"a": {"nthreads": 2}}}   # → window = max(4, 4) = 4

    def compute(self, arr):
        fut = _FakeFuture(lambda a=arr: a.compute(scheduler="synchronous"))
        self.created.append(fut)
        return fut


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
    def test_bounded_in_flight_and_assembly(self):
        src, client, handle, seen = _run()
        # 16 chunks total, but only the window (4) submitted up front.
        assert len(client.created) == 4
        assert not handle.done()
        # Completing futures tops up the window without ever exceeding it.
        fired = 0
        while fired < len(client.created):
            client.created[fired].fire()
            fired += 1
            in_flight = len(client.created) - fired
            assert in_flight <= 4
        assert handle.done()
        # Every chunk streamed a callback and the assembly equals the source.
        assert len(seen) == 16
        np.testing.assert_array_equal(handle.result(), src.compute())

    def test_cancel_stops_submission_and_outstanding(self):
        src, client, handle, seen = _run()
        assert len(client.created) == 4
        handle.cancel()
        assert all(f.cancelled for f in client.created)
        # Firing the cancelled futures must not submit more work.
        for f in list(client.created):
            f.fire()
        assert len(client.created) == 4
        assert handle.done()

    def test_stop_event_halts_topups(self):
        stop = threading.Event()
        src, client, handle, seen = _run(stop_event=stop)
        stop.set()
        for f in list(client.created):
            f.fire()
        assert len(client.created) == 4        # no top-ups after stop
        assert handle.done()

    def test_error_propagates_via_result(self):
        from spyde.drawing.update_functions import compute_with_live_buffer
        src = da.from_array(np.ones((4, 4), np.float32), chunks=(2, 2))

        class _BoomClient(_FakeClient):
            def compute(self, arr):
                fut = _FakeFuture(lambda: (_ for _ in ()).throw(ValueError("boom")))
                self.created.append(fut)
                return fut

        client = _BoomClient()
        handle = compute_with_live_buffer(src, (4, 4), client, shm_name="",
                                          windowed=True)
        for f in list(client.created):
            f.fire()
        assert handle.done()
        try:
            handle.result()
            assert False, "error did not propagate"
        except ValueError:
            pass
