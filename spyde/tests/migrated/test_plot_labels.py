"""Axis calibration + labels reach the anyplotlib panel.

Covers the three "labels are missing / don't update" bugs:

- A 1-D line plot must get its x-axis coordinates + x-axis label (units/name)
  AND a y-axis label ("Intensity" by default, or the signal's quantity) — the
  old ``_set_array`` dims==1 branch called ``set_data(data)`` with nothing, so
  the panel had a bare 0..N-1 x-axis and no y-label.
- Editing the axes (scale/units) re-pushes the plot; the 1-D branch re-reads the
  signal every paint, so the edit propagates.
- A 2-D signal plot shows the dataset NAME as the panel title (drawn in
  anyplotlib's title strip), not only on the Electron window chrome.

We stub the anyplotlib Plot1D/Plot2D and assert on what reaches set_data /
set_title — the wire boundary — so no renderer is needed.
"""
from __future__ import annotations

import numpy as np
import hyperspy.api as hs

from spyde.drawing.plots.plot import Plot


class _FakePlot1D:
    def __init__(self):
        self.last = None
        self._state = {"title": ""}
    def set_data(self, data, x_axis=None, units=None, y_units=None):
        self.last = {"data": np.asarray(data), "x_axis": x_axis,
                     "units": units, "y_units": y_units}
    def set_title(self, title):
        self._state["title"] = title


class _FakePlot2D:
    def __init__(self):
        self.last = None
        self._state = {"title": "", "units": "px"}
    def set_data(self, data, x_axis=None, y_axis=None, units=None, clim=None,
                 tile=None):
        self.last = {"data": np.asarray(data), "x_axis": x_axis, "y_axis": y_axis,
                     "units": units, "clim": clim, "tile": tile}
    def set_extent(self, x, y, units=None):
        if units is not None:
            self._state["units"] = units
    def set_title(self, title):
        self._state["title"] = title


def _make_1d_plot(signal):
    plot = Plot.__new__(Plot)
    plot._plot1d = _FakePlot1D()
    plot._plot2d = None
    plot.is_navigator = False
    plot.window_id = "t"
    plot.needs_auto_level = True
    plot._last_levels = None
    plot._last_extent_key = None
    plot._fv_transform_active = False
    plot._fv_paint_token = False
    plot._y_label_override = None

    class _PS:
        current_signal = signal
    plot.plot_state = _PS()
    plot._ensure_figure = lambda dims: None
    plot._emit_histogram = lambda *a, **k: None
    return plot


def _make_2d_plot(signal, title=""):
    plot = Plot.__new__(Plot)
    plot._plot2d = _FakePlot2D()
    plot._plot1d = None
    plot.is_navigator = False
    plot.window_id = "t"
    signal.metadata.set_item("General.title", title)

    class _PS:
        current_signal = signal
    plot.plot_state = _PS()
    return plot


class TestLine1DCalibration:
    def test_calibrated_x_axis_reaches_set_data(self):
        s = hs.signals.Signal1D(np.arange(64, dtype=np.float32))
        ax = s.axes_manager.signal_axes[0]
        ax.units = "eV"
        ax.scale = 0.5
        ax.offset = 100.0
        plot = _make_1d_plot(s)
        plot._set_array(s.data)
        out = plot._plot1d.last
        assert out["x_axis"] is not None
        # x-axis spans 100..100+63*0.5 (the calibrated coordinate array)
        assert out["x_axis"][0] == 100.0
        assert out["x_axis"][-1] == 100.0 + 63 * 0.5
        assert "eV" in out["units"]

    def test_default_y_label_is_intensity(self):
        s = hs.signals.Signal1D(np.arange(10, dtype=np.float32))
        plot = _make_1d_plot(s)
        plot._set_array(s.data)
        assert plot._plot1d.last["y_units"] == "Intensity"

    def test_signal_quantity_overrides_default_y_label(self):
        s = hs.signals.Signal1D(np.arange(10, dtype=np.float32))
        s.metadata.set_item("Signal.quantity", "Counts")
        plot = _make_1d_plot(s)
        plot._set_array(s.data)
        assert plot._plot1d.last["y_units"] == "Counts"

    def test_edit_units_repropagates_on_repaint(self):
        # An Axes-editor edit mutates the axes_manager then re-pushes via
        # p.update() → _set_array. The 1-D branch re-reads the signal each paint,
        # so the new units reach set_data.
        s = hs.signals.Signal1D(np.arange(32, dtype=np.float32))
        ax = s.axes_manager.signal_axes[0]
        ax.units = "px"
        plot = _make_1d_plot(s)
        plot._set_array(s.data)
        # "px" is suppressed → empty x label
        assert plot._plot1d.last["units"] == ""
        ax.units = "1/nm"
        ax.scale = 2.0
        plot._set_array(s.data)
        assert "1/nm" in plot._plot1d.last["units"] or "nm" in plot._plot1d.last["units"]

    def test_length_mismatch_keeps_default_x_axis(self):
        # Line-profile output: the placeholder signal's axis is length 10 but the
        # data is a different length. Don't push a bad x-axis; keep 0..N-1.
        s = hs.signals.Signal1D(np.zeros(10, dtype=np.float32))
        plot = _make_1d_plot(s)
        plot._set_array(np.arange(50, dtype=np.float32))  # 50 != 10
        assert plot._plot1d.last["x_axis"] is None
        # y-label still applies
        assert plot._plot1d.last["y_units"] == "Intensity"


class TestPlot2DTitle:
    def test_apply_plot_title_pushes_dataset_name(self):
        s = hs.signals.Signal2D(np.zeros((8, 8), dtype=np.float32))
        plot = _make_2d_plot(s, title="my_scan.hspy")
        plot._apply_plot_title()
        assert plot._plot2d._state["title"] == "my_scan.hspy"

    def test_empty_title_leaves_strip_blank(self):
        s = hs.signals.Signal2D(np.zeros((8, 8), dtype=np.float32))
        plot = _make_2d_plot(s, title="")
        plot._apply_plot_title()
        assert plot._plot2d._state["title"] == ""


class TestNavAxisEdit:
    """Editing a NAVIGATION axis must recalibrate the navigator plot. The
    navigator displays a derived `root.sum(signal_axes)` signal whose axes are a
    decoupled copy; the fix makes the navigator plot read the ROOT navigation
    axes directly (Plot._display_axes), so a set_axis edit reaches it."""

    def _nav_plot(self, session):
        for p in session._plots:
            if getattr(p, "is_navigator", False) and p.plot_state is not None:
                return p
        return None

    def test_nav_axis_edit_reaches_navigator_display_axes(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        nav = self._nav_plot(session)
        assert nav is not None
        tree = nav.signal_tree
        # Edit EVERY navigation axis of the root → nm units + scale 2.0.
        am = tree.root.axes_manager
        for i, ax in enumerate(am._axes):
            if ax.navigate:
                session._set_axis(nav, {"index": i, "field": "units", "value": "nm"})
                session._set_axis(nav, {"index": i, "field": "scale", "value": "2.0"})
        # The navigator plot's DISPLAY axes are the root nav axes and now carry
        # the edit (before the fix they were the stale derived-signal copy).
        disp = nav._display_axes()
        assert all(str(a.units) == "nm" for a in disp)
        assert all(float(a.scale) == 2.0 for a in disp)
        # And _axes_info (2-D navigator) returns the calibrated axes + units.
        ns = am.navigation_shape
        frame = np.zeros((ns[1], ns[0]), dtype=np.float32)
        axes, units = nav._axes_info(frame)
        assert units == "nm"
        assert axes is not None and len(axes[0]) == ns[0] and len(axes[1]) == ns[1]

    def test_signal_plot_still_uses_signal_axes(self, stem_4d_dataset):
        # The is_navigator branch must NOT change the signal plot: it still reads
        # its current_signal's signal axes.
        session = stem_4d_dataset["window"]
        sig_plot = next(p for p in session._plots
                        if not getattr(p, "is_navigator", False)
                        and p.plot_state is not None)
        disp = sig_plot._display_axes()
        expected = list(sig_plot.plot_state.current_signal.axes_manager.signal_axes)
        assert disp == expected


class TestSetTitle:
    """The breadcrumb rename → set_title writes the shared root title + emits a
    lightweight window_title update to every window of the tree."""

    def test_set_title_updates_root_and_emits(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        messages = stem_4d_dataset["messages"]
        plot = next(p for p in session._plots if p.plot_state is not None)
        tree = plot.signal_tree
        session._set_title(plot, {"title": "my_new_name"})
        assert tree.root.metadata.get_item("General.title") == "my_new_name"
        # A window_title message went out for the whole tree.
        wt = [m for m in messages if m.get("type") == "window_title"]
        assert wt, "no window_title message emitted"
        assert wt[-1]["title"] == "my_new_name"
        assert len(wt[-1]["window_ids"]) >= 1
        # The in-panel title strip was re-applied on the 2-D plots.
        for p in session._plots:
            p2 = getattr(p, "_plot2d", None)
            if p2 is not None and not p.is_navigator:
                assert p2._state.get("title") == "my_new_name"

    def test_set_title_ignores_blank(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        plot = next(p for p in session._plots if p.plot_state is not None)
        before = plot.signal_tree.root.metadata.get_item("General.title", default="")
        session._set_title(plot, {"title": "   "})
        assert plot.signal_tree.root.metadata.get_item(
            "General.title", default="") == before
