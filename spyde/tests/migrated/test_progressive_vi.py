"""
Virtual image must produce a NON-BLACK output (the "it's just black" bug).

The VI computes via `VirtualImageAction.reduce` — a Dask Future for lazy data
with a client (the PlotUpdateWorker polls it and pushes the result), or a
synchronous numpy compute otherwise. (An earlier per-chunk poll-thread stream
raced with the selector's own `update_data(blank)` and clobbered the output back
to a blank frame.)
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _wait(pred, timeout=8.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _bright_4d(nav=(4, 5), sig=(8, 8)):
    """Uniformly-bright data so ANY detector ROI integrates non-zero signal
    (the output must not be black)."""
    rng = np.random.RandomState(0)
    data = (rng.rand(*nav, *sig).astype(np.float32) + 1.0) * 100.0
    s = hs.signals.Signal2D(data)
    s.set_signal_type("electron_diffraction")
    return s


class TestVirtualImageDisplay:
    def test_eager_virtual_image_is_nonzero(self, window):
        session = window["window"]
        session._add_signal(_bright_4d())          # eager numpy
        time.sleep(0.4)
        src = _signal_plot(session)
        before = list(session._plots)
        session._dispatch_toolbar_action(
            src, "add_virtual_image", {"type": "disk", "calculation": "mean"})
        out = next((p for p in session._plots if p not in before), None)
        assert out is not None
        got = _wait(lambda: isinstance(getattr(out, "current_data", None), np.ndarray)
                    and np.isfinite(out.current_data).any()
                    and float(np.nanmax(out.current_data)) > 0)
        assert got, f"virtual image output is black: {getattr(out, 'current_data', None)!r}"

    def test_lazy_virtual_image_is_nonzero(self, window):
        session = window["window"]
        session._add_signal(_bright_4d().as_lazy())  # lazy, no client → sync compute
        time.sleep(0.4)
        src = _signal_plot(session)
        before = list(session._plots)
        session._dispatch_toolbar_action(
            src, "add_virtual_image", {"type": "disk", "calculation": "mean"})
        out = next((p for p in session._plots if p not in before), None)
        assert out is not None
        got = _wait(lambda: isinstance(getattr(out, "current_data", None), np.ndarray)
                    and np.isfinite(out.current_data).any()
                    and float(np.nanmax(out.current_data)) > 0)
        assert got, f"virtual image output is black: {getattr(out, 'current_data', None)!r}"
