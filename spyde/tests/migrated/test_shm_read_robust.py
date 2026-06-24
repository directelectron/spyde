"""
Robustness of the shared-memory navigator-image path under the latest-wins,
future-cancel update model.

When a newer navigator position cancels the in-flight write_shared_array future,
its buffer may be unwritten. The worker does NOT special-case cancelled futures
(skipping them dropped every distributed DP frame, since hyperspy GC-cancels the
write's get_inds dependency); instead read_shared_array must reject an unwritten
buffer cleanly (ValueError, not the cryptic "Data type '' not understood"), the
worker captures that as the result, and _on_plot_ready drops an Exception result.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.drawing.update_functions import (
    read_shared_array, write_shared_array, ensure_live_buffer,
)


class _Shm:
    """A shared-memory-buffer stand-in backed by a bytearray."""
    def __init__(self, n=4096):
        self.buf = memoryview(bytearray(n))


class TestReadSharedArrayRobust:
    def test_unwritten_buffer_raises_clean_valueerror(self):
        # all-zero buffer → dtype_length 0 → must raise ValueError, NOT the
        # cryptic "Data type '' not understood".
        shm = _Shm()
        with pytest.raises(ValueError):
            read_shared_array(shm)

    def test_roundtrip_after_write(self):
        shm = _Shm()
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)

        # write_shared_array attaches by NAME; emulate by writing into our buffer
        # via the same byte layout it uses.
        buf = shm.buf
        off = 0
        dtype_bytes = arr.dtype.str.encode("utf-8")
        buf[off:off+4] = len(dtype_bytes).to_bytes(4, "little"); off += 4
        buf[off:off+len(dtype_bytes)] = dtype_bytes; off += len(dtype_bytes)
        buf[off:off+4] = arr.ndim.to_bytes(4, "little"); off += 4
        for d in arr.shape:
            buf[off:off+8] = int(d).to_bytes(8, "little"); off += 8
        raw = arr.tobytes()
        buf[off:off+len(raw)] = raw
        out = read_shared_array(shm)
        assert out.shape == (3, 4)
        assert np.array_equal(out, arr)


class _FakeCancelledFuture:
    key = "write_shared_array-abc"
    def done(self): return True
    def cancelled(self): return True
    def result(self): raise AssertionError("should not read a cancelled future")


class _FakeDoneFuture:
    key = "write_shared_array-xyz"
    def __init__(self): self._cancelled = False
    def done(self): return True
    def cancelled(self): return False


def test_future_from_signal_no_ambiguous_truth():
    """_future_from_signal must not raise on a multi-element ndarray (the
    'truth value of an array is ambiguous' bug from `and data`)."""
    from spyde.workers.plot_update_worker import PlotUpdateWorker
    w = PlotUpdateWorker.__new__(PlotUpdateWorker)

    class _Sig:
        pass

    # plain multi-element numeric ndarray → no future, no exception
    s = _Sig(); s.data = np.zeros((4, 4, 8, 8), dtype=np.float32)
    assert w._future_from_signal(s) is None

    # empty ndarray
    s2 = _Sig(); s2.data = np.zeros((0,), dtype=np.float32)
    assert w._future_from_signal(s2) is None

    # length-1 object array NOT holding a Future
    s3 = _Sig(); s3.data = np.array([np.zeros((2, 2))], dtype=object)
    assert w._future_from_signal(s3) is None


def test_worker_does_not_skip_cancelled_future_but_result_is_harmless():
    """A cancelled future is NOT skipped (the QT worker didn't skip, and skipping
    ALL cancelled futures silently dropped every distributed DP frame because
    hyperspy GC-cancels the write's get_inds dependency). Instead the shm read is
    attempted; a torn/unwritten buffer raises cleanly, the worker captures it as
    the RESULT (an Exception), and emits — `_on_plot_ready` then drops an Exception
    result harmlessly. This test asserts: emitter IS called, and with an Exception
    result (never a bogus frame)."""
    from spyde.workers.plot_update_worker import PlotUpdateWorker

    w = PlotUpdateWorker.__new__(PlotUpdateWorker)
    w._seen = set()
    w._dispatch = None   # run the emit inline (no main loop in the test)
    w._emit_timing = False

    class _Sig:
        def emit(self, *a): pass
    w.debug_print = _Sig()

    # A plot whose shared_memory read raises (torn/unwritten buffer for the
    # cancelled write) — mirrors the real read_shared_array ValueError.
    class _Plot:
        @property
        def shared_memory(self):
            raise ValueError("shared-memory buffer not yet written")

    import spyde.workers.plot_update_worker as mod
    real_Future = mod.Future
    try:
        mod.Future = (_FakeCancelledFuture, _FakeDoneFuture)
        captured = {}
        w._maybe_emit_future(
            _FakeCancelledFuture(),
            emitter=lambda plot, result, fut: captured.update(result=result),
            plot=_Plot(),
        )
        # Emitter WAS called (not skipped), and the result is the Exception from
        # the torn read — never a real frame.
        assert "result" in captured, "cancelled future was wrongly skipped"
        assert isinstance(captured["result"], Exception)
    finally:
        mod.Future = real_Future
