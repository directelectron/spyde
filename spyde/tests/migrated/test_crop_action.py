"""
Crop action (Phase 7) — trim a dataset to a spatial box + optional time range.

CropAction is a TransformAction (like Rebin) that adds a lazy "Cropped" node to
the same tree via hyperspy isig/inav slicing — a dask-view (no materialise), so a
multi-GB movie crops for free. Zero ranges keep the full extent on that axis.
"""
from __future__ import annotations

import time

import numpy as np
import dask.array as da
import hyperspy.api as hs

from spyde.actions.base import CropAction, _crop_signal


class TestCropSlicing:
    def _movie(self, n=20, frame=(256, 256)):
        return hs.signals.Signal2D(
            da.zeros((n,) + frame, dtype=np.float32, chunks=(1,) + frame)).as_lazy()

    def test_spatial_crop_only_lazy(self):
        s = self._movie()
        c = _crop_signal(s, x0=50, x1=150, y0=30, y1=100)
        assert c._lazy and isinstance(c.data, da.Array)   # no materialise
        assert c.data.shape == (20, 70, 100)              # (t, y1-y0, x1-x0)

    def test_spatial_and_time_crop(self):
        s = self._movie()
        c = _crop_signal(s, x0=50, x1=150, y0=30, y1=100, t0=5, t1=12)
        assert c.data.shape == (7, 70, 100)
        assert c._lazy

    def test_zero_spatial_ranges_keep_full_frame(self):
        s = self._movie()
        c = _crop_signal(s, t0=2, t1=8)          # time-only crop
        assert c.data.shape == (6, 256, 256)     # full frame kept

    def test_out_of_range_end_is_clamped(self):
        s = self._movie(frame=(64, 64))
        c = _crop_signal(s, x0=10, x1=9999, y0=0, y1=0)   # x1 too big, y full
        assert c.data.shape == (20, 64, 54)      # x: 10..64, y: full

    def test_start_at_zero_is_a_real_crop_not_full(self):
        # y0=0, y1=15 must crop to the first 15 rows (only end<=0 means "full").
        s = self._movie(frame=(64, 64))
        c = _crop_signal(s, y0=0, y1=15)
        assert c.data.shape == (20, 15, 64)

    def test_all_zero_crop_is_a_noop(self):
        s = self._movie()
        c = _crop_signal(s)                 # every bound 0
        assert c is s                       # returned unchanged, no new object

    def test_crop_preserves_values(self):
        # A real (eager) frame with a marker so we can check the crop window.
        data = np.zeros((4, 32, 32), dtype=np.float32)
        data[:, 10, 20] = 7.0                    # (row=10, col=20) per frame
        s = hs.signals.Signal2D(data)
        c = _crop_signal(s, x0=15, x1=25, y0=5, y1=15)   # cols 15..25, rows 5..15
        arr = np.asarray(c.data)
        assert arr.shape == (4, 10, 10)
        # marker at (row 10, col 20) -> in crop it's (row 10-5=5, col 20-15=5).
        assert float(arr[0, 5, 5]) == 7.0


class TestCropThroughAction:
    def test_run_adds_a_cropped_node(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        tree = session.signal_trees[0]
        plot = next(p for p in session._plots
                    if not p.is_navigator and p.plot_state is not None)
        root = tree.root

        # ASYMMETRIC box so an x/y transpose bug in the run() flow is caught:
        # x 2..14 (=12 cols), y 4..10 (=6 rows) → signal_shape (12, 6).
        act = CropAction.for_plot(plot, x0=2, x1=14, y0=4, y1=10)
        new = act.run()
        time.sleep(0.2)

        assert new is not None
        assert tuple(new.axes_manager.signal_shape) == (12, 6)
        # Nav (scan) shape unchanged (no t-range given).
        assert tuple(new.axes_manager.navigation_shape) == \
            tuple(root.axes_manager.navigation_shape)

    def test_run_keeps_a_lazy_movie_lazy(self, movie_dataset):
        # The headline memory-safety claim, guarded through the real run() flow:
        # cropping a LAZY movie stays a dask view (no materialise).
        session = movie_dataset["window"]
        plot = next(p for p in session._plots
                    if not p.is_navigator and p.plot_state is not None)
        act = CropAction.for_plot(plot, x0=4, x1=20, y0=6, y1=18)
        new = act.run()
        time.sleep(0.2)
        assert new is not None
        assert new._lazy is True
        assert isinstance(new.data, da.Array), "cropped movie must stay lazy"
        assert tuple(new.axes_manager.signal_shape) == (16, 12)


class TestCropInToolbar:
    def test_crop_available_on_a_2d_signal_plot(self, stem_4d_dataset):
        from spyde.drawing.toolbars.plot_control_toolbar import (
            get_toolbar_actions_for_plot,
        )
        session = stem_4d_dataset["window"]
        plot = next(p for p in session._plots
                    if not p.is_navigator and p.plot_state is not None)
        names = get_toolbar_actions_for_plot(plot.plot_state)[2]
        assert "Crop" in names, f"Crop missing from toolbar actions: {names}"
