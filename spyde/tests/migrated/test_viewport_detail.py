"""SpyDE large-signal frames route through anyplotlib TILE mode via set_data.

A big (≥ TILE_THRESHOLD) signal frame is handed to anyplotlib's tiled display by
calling ``plot2d.set_data(native, clim=...)`` (see Plot._set_array). anyplotlib's
set_data AUTO-ENABLES tile mode on the first large frame, swaps the source on each
subsequent one (keeping zoom/detail), and derives the contrast — SpyDE no longer
wraps a NumpyTileBackend or calls enable_tile itself (the old hand-rolled
_maybe_tile_signal drifted out of sync with set_data and is gone).

These assert the anyplotlib seam SpyDE depends on, using a real Plot2D (no browser).
"""
import os
os.environ.setdefault("SPYDE_NO_DASK", "1")

import numpy as np
import anyplotlib as apl


def _plot2d(n=64):
    """A small (untiled) Plot2D — the placeholder SpyDE creates before real frames."""
    fig, axes = apl.subplots(1, 1)
    ax = axes[0][0] if isinstance(axes, list) else axes
    return ax.imshow(np.zeros((n, n), np.float32), vmin=0, vmax=1, gpu="auto")


class TestSetDataDrivesTiling:
    def test_large_frame_auto_enables_tile_mode(self):
        p2d = _plot2d()
        assert p2d._state["tile_enabled"] is False        # starts small/untiled
        frame = np.random.RandomState(0).rand(4096, 4096).astype(np.float32)
        p2d.set_data(frame, clim=(0.0, 1.0))
        assert p2d._state["tile_enabled"] is True
        assert p2d._state["image_width"] == 4096           # logical
        assert 0 < p2d._state["base_width"] <= 1024         # overview base

    def test_small_frame_stays_untiled(self):
        p2d = _plot2d()
        frame = np.random.RandomState(1).rand(512, 512).astype(np.float32)
        p2d.set_data(frame, clim=(0.0, 1.0))
        assert p2d._state["tile_enabled"] is False
        assert p2d._state["image_width"] == 512

    def test_live_update_swaps_frame_keeps_view(self):
        from anyplotlib.callbacks import Event
        p2d = _plot2d()
        a = np.zeros((4096, 4096), np.float32)
        b = np.full((4096, 4096), 0.9, np.float32)
        b[1408:2688, 1408:2688] = 0.5
        p2d.set_data(a, clim=(0.0, 1.0))                    # enable with frame A
        # zoom → a detail tile of the current region
        p2d.callbacks.fire(Event("view_changed", zoom=4.0, center_x=0.5, center_y=0.5,
                                 display_width=1000, display_height=1000))
        reg = list(p2d._state["detail_region"])
        assert len(reg) == 4
        # next nav frame via set_data → view persists, tile source swaps to B
        p2d.set_data(b, clim=(0.0, 1.0))
        assert list(p2d._state["detail_region"]) == reg     # zoom/subselection kept
        assert p2d._state["tile_enabled"] is True
        x0, x1, y0, y1 = reg
        crop = p2d._tile_backend.sample(x0, x1, y0, y1, 100, 100, "mean")
        # backend swapped a→b: the over-fetched region mixes b's 0.9 background with
        # its 0.5 centre patch, so the mean sits well above a's all-zero frame.
        assert 0.5 <= float(crop.mean()) <= 0.9

    def test_contrast_from_full_res_not_overview(self):
        # No explicit clim: the display range must come from the FULL-RES frame (native
        # extremes), NOT the averaged overview — else a zoom detail tile blows out to
        # white. Full-range random data → display range must span ~[0, 1].
        p2d = _plot2d()
        frame = np.random.RandomState(3).rand(4096, 4096).astype(np.float32)
        p2d.set_data(frame)                                 # NO clim
        assert p2d._state["tile_enabled"] is True
        assert p2d._state["display_min"] < 0.05             # true min, not overview-mean
        assert p2d._state["display_max"] > 0.95             # true max, not overview-mean
