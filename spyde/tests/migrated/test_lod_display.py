"""SpyDE does NO level-of-detail decimation — anyplotlib tile mode owns all
downscaling.

Plot._set_array hands the FULL frame to anyplotlib's ``Plot2D.set_data`` every time:
a signal image with ``tile="auto"`` (anyplotlib tiles it above its threshold —
overview base + on-zoom detail tile), a navigator image with ``tile=False`` (its 2-D
selector maps clicks by displayed px, so it must stay a 1:1 full-frame image). The old
SpyDE-side stride/thumbnail decimation (``_lod_stride`` / ``_lod_downsample`` / the
``_LOD_MAX_PX*`` caps) is gone.

We assert on the array + tile flag actually handed to set_data (the wire boundary) by
stubbing it, so we test the real paint path without a renderer.
"""
from __future__ import annotations

import numpy as np

from spyde.drawing.plots.plot import Plot


class _FakePlot2D:
    """Captures the (data, axes, tile) handed to set_data — the wire boundary."""
    def __init__(self):
        self.last = None
        self._state = {}
    def set_data(self, data, x_axis=None, y_axis=None, units=None, clim=None,
                 tile=None):
        self.last = {"data": np.asarray(data), "x_axis": x_axis, "y_axis": y_axis,
                     "clim": clim, "tile": tile}
    def set_extent(self, x, y):
        pass


def _make_plot(sig_shape, units="nm", scale=0.5, is_navigator=False):
    """A minimal Plot wired just enough to run _set_array with a calibrated
    signal, a stubbed Plot2D, and no real figure."""
    import hyperspy.api as hs

    plot = Plot.__new__(Plot)
    plot._plot2d = _FakePlot2D()
    plot.is_navigator = is_navigator
    plot.window_id = "t"
    plot.needs_auto_level = True
    plot._last_levels = None
    plot._last_extent_key = None
    plot._fv_transform_active = False
    plot._fv_paint_token = False

    data = np.zeros((2,) + sig_shape, dtype=np.float32)
    s = hs.signals.Signal2D(data)
    for ax in s.axes_manager.signal_axes:
        ax.units = units
        ax.scale = scale

    class _PS:
        current_signal = s
    plot.plot_state = _PS()

    plot._ensure_figure = lambda dims: None
    plot._emit_histogram = lambda *a, **k: None
    plot._is_navigated_frame = lambda: True
    return plot


class TestNoDecimation:
    def test_large_signal_frame_sent_full_res(self):
        # A 4096² signal frame must reach set_data at FULL resolution (no stride) —
        # anyplotlib tiles it, SpyDE does not shrink it.
        plot = _make_plot((4096, 4096))
        frame = np.random.RandomState(0).rand(4096, 4096).astype(np.float32)
        plot._set_array(frame)
        out = plot._plot2d.last["data"]
        assert out.shape == (4096, 4096), f"frame was decimated: {out.shape}"

    def test_large_signal_frame_uses_tile_auto(self):
        plot = _make_plot((4096, 4096))
        plot._set_array(np.random.RandomState(1).rand(4096, 4096).astype(np.float32))
        assert plot._plot2d.last["tile"] == "auto"

    def test_small_frame_sent_full_and_auto(self):
        # A small DP frame is also handed full-res with tile="auto" (anyplotlib just
        # won't tile it, being under its threshold — SpyDE doesn't decide that).
        plot = _make_plot((128, 128))
        plot._set_array(np.random.RandomState(2).rand(128, 128).astype(np.float32))
        assert plot._plot2d.last["data"].shape == (128, 128)
        assert plot._plot2d.last["tile"] == "auto"

    def test_navigator_frame_full_res_tile_false(self):
        # A large NAVIGATOR image must NOT tile (its 2-D selector maps clicks by
        # displayed-pixel coords) and must NOT be decimated (that would offset every
        # nav selection). So: full-res + tile=False.
        plot = _make_plot((4096, 4096), is_navigator=True)
        plot._set_array(np.random.RandomState(3).rand(4096, 4096).astype(np.float32))
        assert plot._plot2d.last["data"].shape == (4096, 4096)
        assert plot._plot2d.last["tile"] is False

    def test_axes_full_length_match_full_frame(self):
        # The calibrated axes handed to set_data span the full frame (no subsample) —
        # length must equal the (undecimated) frame dimensions.
        plot = _make_plot((4096, 4096), units="nm", scale=0.5)
        plot._set_array(np.random.RandomState(4).rand(4096, 4096).astype(np.float32))
        last = plot._plot2d.last
        assert last["x_axis"] is not None
        assert len(last["x_axis"]) == 4096 and len(last["y_axis"]) == 4096

    def test_clim_passed_atomically(self):
        # The display range rides the SAME set_data push (no separate set_clim → no
        # contrast flash).
        plot = _make_plot((256, 256))
        plot._set_array(np.random.RandomState(5).rand(256, 256).astype(np.float32))
        clim = plot._plot2d.last["clim"]
        assert clim is not None and clim[1] > clim[0]
