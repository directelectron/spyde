"""
Level-of-detail decimation of large frames before transport (Phase 3).

Plot._set_array decimates a 2-D frame whose longest side exceeds _LOD_MAX_PX
before it is serialised to the renderer (a 4k/8k movie frame is far bigger than
any panel and the base64-in-JSON transport). The axes are subsampled by the same
stride so the scale bar / extent stay calibrated. A DP-sized frame is untouched.

We assert on the array actually handed to anyplotlib's Plot2D.set_data (the wire
boundary) by stubbing it, so we test the real paint path without a renderer.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.drawing.plots.plot import _lod_stride, _LOD_MAX_PX


class TestLodStride:
    def test_small_dp_untouched(self):
        assert _lod_stride(128, 128) == 1
        assert _lod_stride(256, 256) == 1
        assert _lod_stride(_LOD_MAX_PX, _LOD_MAX_PX) == 1

    def test_large_frames_decimate(self):
        assert _lod_stride(4096, 4096) == 3      # -> 1365
        assert _lod_stride(8192, 8192) == 6      # -> 1365
        assert _lod_stride(_LOD_MAX_PX + 1, _LOD_MAX_PX + 1) == 2

    def test_non_square_uses_longest_side(self):
        # A 300x4096 frame decimates by the 4096 side (stride 3) on both axes,
        # preserving aspect ratio.
        assert _lod_stride(300, 4096) == 3


class _FakePlot2D:
    """Captures the (data, axes) handed to set_data — the wire boundary."""
    def __init__(self):
        self.last = None
        self._state = {}
    def set_data(self, data, x_axis=None, y_axis=None, units=None, clim=None):
        self.last = {"data": np.asarray(data), "x_axis": x_axis, "y_axis": y_axis}
    def set_extent(self, x, y):
        pass


def _make_plot(sig_shape, units="nm", scale=0.5):
    """A minimal Plot wired just enough to run _set_array with a calibrated
    signal, a stubbed Plot2D, and no real figure."""
    from spyde.drawing.plots.plot import Plot
    import hyperspy.api as hs

    plot = Plot.__new__(Plot)
    # Attributes _set_array / _axes_info / _robust_levels touch.
    plot._plot2d = _FakePlot2D()
    plot.is_navigator = False
    plot.window_id = "t"
    plot.needs_auto_level = True
    plot._last_levels = None
    plot._last_extent_key = None
    plot._fv_transform_active = False
    plot._fv_paint_token = False

    # A signal whose signal axes are calibrated so _axes_info returns real axes.
    data = np.zeros((2,) + sig_shape, dtype=np.float32)
    s = hs.signals.Signal2D(data)
    for ax in s.axes_manager.signal_axes:
        ax.units = units
        ax.scale = scale

    class _PS:
        current_signal = s
    plot.plot_state = _PS()

    # Stub the figure-ensuring + histogram so _set_array runs headless.
    plot._ensure_figure = lambda dims: None
    plot._emit_histogram = lambda *a, **k: None
    plot._is_navigated_frame = lambda: True
    return plot


class TestLodInSetArray:
    def test_large_frame_is_decimated_at_the_wire(self):
        plot = _make_plot((4096, 4096))
        frame = np.random.RandomState(0).rand(4096, 4096).astype(np.float32)
        plot._set_array(frame)
        out = plot._plot2d.last["data"]
        assert max(out.shape) <= _LOD_MAX_PX, f"frame not decimated: {out.shape}"
        # 4096 with stride 3 -> ceil(4096/3) = 1366 samples (numpy slice length).
        assert out.shape == (1366, 1366)

    def test_axes_subsampled_to_match_decimated_frame(self):
        plot = _make_plot((4096, 4096), units="nm", scale=0.5)
        frame = np.random.RandomState(1).rand(4096, 4096).astype(np.float32)
        plot._set_array(frame)
        last = plot._plot2d.last
        out = last["data"]
        # The axis arrays handed to set_data must match the decimated width/height
        # (otherwise anyplotlib draws a wrong scale bar / extent).
        assert last["x_axis"] is not None
        assert len(last["x_axis"]) == out.shape[1]
        assert len(last["y_axis"]) == out.shape[0]
        # Calibration preserved: the physical span is (approximately) the same —
        # first/last axis samples still span the frame.
        assert abs(float(last["x_axis"][0])) < 1e-6           # starts at 0
        # last sample ~ (n-1)*stride*scale, i.e. still near the full 4096*0.5 span
        assert float(last["x_axis"][-1]) > 4096 * 0.5 * 0.9

    def test_dp_frame_not_decimated(self):
        plot = _make_plot((128, 128))
        frame = np.random.RandomState(2).rand(128, 128).astype(np.float32)
        plot._set_array(frame)
        out = plot._plot2d.last["data"]
        assert out.shape == (128, 128)     # untouched

    def test_navigator_image_not_decimated(self):
        # A large NAVIGATOR image (e.g. a 4D-STEM real-space nav) must NOT be
        # decimated — its 2-D selector maps clicks by displayed-pixel coords, so
        # decimation would offset every nav selection by the stride.
        plot = _make_plot((4096, 4096))
        plot.is_navigator = True
        frame = np.random.RandomState(3).rand(4096, 4096).astype(np.float32)
        plot._set_array(frame)
        out = plot._plot2d.last["data"]
        assert out.shape == (4096, 4096)   # full-res, not decimated
