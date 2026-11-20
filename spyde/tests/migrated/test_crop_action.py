"""
CropAction — the scripting form of Crop: the keyword fields (``x0``…``t1``,
``scan_*``) and a lazy "Cropped" node in the same tree, via hyperspy isig/inav
slicing — a dask view (no materialise), so a multi-GB movie crops for free.
Zero ranges keep the full extent on that axis.

The app's Crop panel is covered by test_crop_panel.py.
"""
from __future__ import annotations

import numpy as np
import pytest
import dask.array as da
import hyperspy.api as hs

from spyde.actions.base import CropAction, _crop_signal
from spyde.tests.migrated.conftest import _settle


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

    def test_time_on_a_time_series_of_scans_is_the_slowest_axis(self):
        """Hyperspy lists navigation axes fastest first, so on a 5-D
        (time, y, x | ky, kx) stack the first one is the scan's x. ``t0:t1``
        used to cut that, and a time crop came back as a narrower scan."""
        s = hs.signals.Signal2D(np.zeros((6, 5, 7, 4, 4), dtype=np.float32))
        c = _crop_signal(s, t0=2, t1=5)
        assert c.data.shape == (3, 5, 7, 4, 4)

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
        _settle(session)

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
        _settle(session)
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


class TestCroppingAndBinningTheScan:
    """The scan axes, not just the detector.

    A 4-D dataset is usually far bigger in the scan than the detector, and
    neither action could touch it: Rebin built ``[1] * nav + [x, y]`` and Crop
    reached only the FIRST navigation axis. So a scan could not be cut to a
    region of interest, nor reduced, nor therefore saved smaller.
    """

    def _scan(self):
        import hyperspy.api as hs
        return hs.signals.Signal2D(
            np.arange(8 * 10 * 6 * 4, dtype=np.uint16).reshape(8, 10, 6, 4))

    def test_the_scan_box_crops_both_axes(self):
        from spyde.actions.base import _crop_signal

        signal = self._scan()
        out = _crop_signal(signal, scan_x0=2, scan_x1=8,
                           scan_y0=1, scan_y1=6)
        assert out.data.shape == (5, 6, 6, 4)
        # The right pixels, not merely the right count.
        assert np.array_equal(out.data, signal.inav[2:8, 1:6].data)

    def test_the_scan_box_does_not_take_the_detector_box(self):
        """The bug this had: the detector branch binds `sx0`/`sy0` as LOCALS,
        so arguments of that name were overwritten before the scan branch read
        them and the scan came back cropped to the detector's bounds — the
        wrong region, at a shape plausible enough to keep."""
        from spyde.actions.base import _crop_signal

        signal = self._scan()
        out = _crop_signal(signal, scan_x0=2, scan_x1=8, scan_y0=1, scan_y1=6,
                           x0=1, x1=3, y0=0, y1=4)
        assert out.data.shape == (5, 6, 4, 2)
        assert np.array_equal(out.data,
                              signal.inav[2:8, 1:6].isig[1:3, 0:4].data)

    def test_nothing_asked_for_changes_nothing(self):
        from spyde.actions.base import _crop_signal

        signal = self._scan()
        assert _crop_signal(signal) is signal

    def test_rebin_reduces_the_scan(self):
        from spyde.actions.base import Rebin2DAction

        signal = self._scan()
        kwargs = Rebin2DAction.build_kwargs(
            None, signal, scale_x=1, scale_y=1, scan_x=2, scan_y=2)
        assert signal.rebin(**kwargs).data.shape == (4, 5, 6, 4)

    def test_rebin_reduces_both_spaces_at_once(self):
        from spyde.actions.base import Rebin2DAction

        signal = self._scan()
        kwargs = Rebin2DAction.build_kwargs(
            None, signal, scale_x=2, scale_y=2, scan_x=2, scan_y=2)
        assert signal.rebin(**kwargs).data.shape == (4, 5, 3, 2)

    def test_rebin_defaults_leave_the_scan_alone(self):
        """Binning the detector was what this did before, and still is."""
        from spyde.actions.base import Rebin2DAction

        signal = self._scan()
        kwargs = Rebin2DAction.build_kwargs(None, signal)
        assert kwargs["scale"] == [1, 1, 2, 2]

    def test_a_factor_that_does_not_divide_says_which_axis(self):
        """hyperspy raises on this too, naming neither the axis nor the size."""
        from spyde.actions.base import Rebin2DAction

        signal = self._scan()
        with pytest.raises(RuntimeError, match="scan x is 10 px"):
            Rebin2DAction.build_kwargs(None, signal, scale_x=1, scale_y=1,
                                       scan_x=3, scan_y=1)
