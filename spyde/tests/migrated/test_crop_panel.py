"""
The Crop panel (spyde/actions/crop.py) on 2-D, 3-D, 4-D and 5-D data.

One set of ranges, many views of it: the panel's fields and sliders, a box on
every plot that draws an axis. What has to hold for every dimensionality:

* every axis is offered, filed under what it is (time, scan, image/detector);
* every plot drawing an axis carries a box, placed on the axis's pixel EDGES;
* moving one view moves the others;
* the result is a NEW dataset holding exactly the cut, whose navigators have
  the cut's extent and calibration, and the source is left alone.
"""
from __future__ import annotations

import types

import dask.array as da
import hyperspy.api as hs
import numpy as np
import pytest

from spyde.actions import crop as crop_module
from spyde.actions.crop import (
    crop_open, crop_close, crop_reset, crop_run, crop_set_range, crop_signal,
)
from spyde.tests.migrated.conftest import _settle


def _signal(shape, *, lazy=False, time_index=None, time_scale=0.05):
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    if lazy:
        chunks = (1,) * (len(shape) - 2) + tuple(shape[-2:])
        signal = hs.signals.Signal2D(da.from_array(data, chunks=chunks)).as_lazy()
    else:
        signal = hs.signals.Signal2D(data)
    signal.metadata.General.title = "sample"
    if time_index is not None:
        axis = signal.axes_manager.navigation_axes[time_index]
        axis.name, axis.units, axis.scale = "time", "s", time_scale
    return signal, data


# name: (shape, lazy, time axis, groups the panel files the axes under)
CASES = {
    "2D": ((20, 30), False, None, ["Image", "Image"]),
    "3D": ((6, 20, 30), True, 0, ["Time", "Image", "Image"]),
    "4D": ((5, 7, 20, 30), False, None, ["Scan", "Scan", "Detector", "Detector"]),
    "5D": ((4, 5, 7, 20, 30), True, 2,
           ["Time", "Scan", "Scan", "Detector", "Detector"]),
}
# The boxes each dimensionality gets: (on a navigator?, the axes it covers).
BOXES = {
    "2D": [(False, ["sig0", "sig1"])],
    "3D": [(False, ["sig0", "sig1"]), (True, ["nav0"])],
    "4D": [(False, ["sig0", "sig1"]), (True, ["nav0", "nav1"])],
    "5D": [(False, ["sig0", "sig1"]), (True, ["nav2"]), (True, ["nav0", "nav1"])],
}


@pytest.fixture
def opened(window, monkeypatch):
    """Load one CASE, open the panel on its signal plot; yields a namespace."""
    states = []
    monkeypatch.setattr(crop_module, "emit", states.append)
    session = window["window"]

    def load(name):
        shape, lazy, time_index, _groups = CASES[name]
        signal, data = _signal(shape, lazy=lazy, time_index=time_index)
        tree = session._add_signal(signal, source_path=None)
        _settle(session, 4.0)
        plot = next(p for p in session._plots
                    if not p.is_navigator and p.signal_tree is tree)
        crop_open(session, plot, {})
        return types.SimpleNamespace(
            session=session, tree=tree, plot=plot, data=data, signal=signal,
            controller=tree._crop_controller, states=states)
    return load


def _drag(box, released=True):
    box.handler(types.SimpleNamespace(
        event_type="pointer_up" if released else "pointer_move"))


class TestCropSignal:
    @pytest.mark.parametrize("name", sorted(CASES))
    def test_every_axis_cuts_to_its_range(self, name):
        shape = CASES[name][0]
        signal, data = _signal(shape)
        am = signal.axes_manager
        ranges = {f"nav{i}": (1, int(a.size) - 1)
                  for i, a in enumerate(am.navigation_axes)}
        ranges.update({f"sig{i}": (2, int(a.size) - 3)
                       for i, a in enumerate(am.signal_axes)})
        out = crop_signal(signal, ranges)
        # Data order is the reverse of display order.
        expected = data[tuple(
            [slice(1, n - 1) for n in shape[:-2]]
            + [slice(2, shape[-2] - 3), slice(2, shape[-1] - 3)])]
        assert np.array_equal(np.asarray(out.data), expected)

    def test_a_whole_range_is_no_crop(self):
        signal, _data = _signal((4, 5, 6))
        assert crop_signal(signal, {"nav0": (0, 4), "sig1": (0, 5)}) is signal

    def test_a_lazy_crop_stays_lazy(self):
        signal, _data = _signal((6, 20, 30), lazy=True)
        out = crop_signal(signal, {"nav0": (1, 4)})
        assert out._lazy and isinstance(out.data, da.Array)

    def test_a_range_is_clamped_to_its_axis(self):
        signal, _data = _signal((6, 20, 30))
        out = crop_signal(signal, {"nav0": (4, 99), "sig0": (-5, 10)})
        assert out.data.shape == (2, 20, 10)


class TestPanel:
    @pytest.mark.parametrize("name", sorted(CASES))
    def test_every_axis_is_offered_under_its_kind(self, opened, name):
        view = opened(name)
        axes = view.states[-1]["axes"]
        assert [axis["group"] for axis in axes] == CASES[name][3]
        assert all((axis["start"], axis["stop"]) == (0, axis["size"]) for axis in axes)
        assert view.states[-1]["is_crop"] is False

    @pytest.mark.parametrize("name", sorted(CASES))
    def test_every_plot_of_an_axis_carries_a_box(self, opened, name):
        view = opened(name)
        boxes = [(box.plot.is_navigator, list(box.keys))
                 for box in view.controller.boxes]
        assert boxes == BOXES[name]

    @pytest.mark.parametrize("name", sorted(CASES))
    def test_a_field_moves_every_box(self, opened, name):
        view = opened(name)
        ranges = {axis["key"]: (1, axis["size"] - 1)
                  for axis in view.states[-1]["axes"]}
        crop_set_range(view.session, view.plot, {"ranges": ranges})
        for box in view.controller.boxes:
            for key, edges in box.read().items():
                assert edges == pytest.approx(ranges[key]), key
        assert view.states[-1]["is_crop"] is True

    def test_units_read_as_the_plots_draw_them(self, opened):
        """pyxem writes a reciprocal detector as ``$\\AA^{-1}$``; the panel
        shows what the axis ticks show."""
        view = opened("4D")
        crop_close(view.session, view.plot)
        for axis in view.signal.axes_manager.signal_axes:
            axis.units = "$\\AA^{-1}$"
        crop_open(view.session, view.plot, {})
        units = {axis["key"]: axis["units"] for axis in view.states[-1]["axes"]}
        assert units["sig0"] == "Å⁻¹"

    def test_a_box_sits_on_pixel_edges(self, opened):
        """anyplotlib puts integer coordinates at pixel CENTRES, so keeping
        columns 3..9 is a box from 2.5 to 9.5 — not 3 to 10, which shows
        half a pixel of the neighbour and drops half of the last kept one."""
        view = opened("2D")
        crop_set_range(view.session, view.plot,
                       {"ranges": {"sig0": (3, 10), "sig1": (4, 8)}})
        widget = view.controller.boxes[0].widget
        assert (float(widget.x), float(widget.w)) == (2.5, 7.0)
        assert (float(widget.y), float(widget.h)) == (3.5, 4.0)

    def test_a_time_span_is_in_the_navigators_units(self, opened):
        """The movie's time line is drawn in seconds; the span edges sit half
        a frame either side of the first and last kept frame."""
        view = opened("3D")
        crop_set_range(view.session, view.plot, {"key": "nav0", "start": 1, "stop": 4})
        span = next(box for box in view.controller.boxes if box.keys == ["nav0"])
        assert (float(span.widget.x0), float(span.widget.x1)) == pytest.approx(
            (0.025, 0.175))

    @pytest.mark.parametrize("name", ["3D", "4D"])
    def test_a_moved_box_is_in_the_panel_state(self, opened, name):
        """A targeted widget move redraws the box but leaves the panel state
        the figure keeps; a click in the plot reloads that state, the box
        jumped back to where it was created, and a drag started from there.
        Seen in the app: grab a typed-in time span, get the whole movie."""
        import json

        view = opened(name)
        crop_set_range(view.session, view.plot, {"key": "nav0", "start": 1, "stop": 3})
        box = next(each for each in view.controller.boxes if each.plot.is_navigator)
        panel = box.widget._plot
        state = json.loads(getattr(panel._fig, f"panel_{panel._id}_json"))
        stored = next(w for w in state["overlay_widgets"] if w["id"] == box.widget.id)
        key = "x0" if "x0" in stored else "x"
        assert stored[key] == pytest.approx(getattr(box.widget, key))

    def test_dragging_a_box_moves_the_ranges_and_the_other_views(self, opened):
        view = opened("4D")
        scan = next(box for box in view.controller.boxes if box.plot.is_navigator)
        scan.widget.set(_notify=False, x=1.5, y=0.5, w=3.0, h=2.0)
        _drag(scan)
        assert view.controller.ranges["nav0"] == (2, 5)
        assert view.controller.ranges["nav1"] == (1, 3)
        state = view.states[-1]
        assert state["after"] == "nav 3 × 2 · sig 30 × 20"

    def test_a_box_moved_past_the_edge_keeps_its_size(self, opened):
        view = opened("4D")
        crop_set_range(view.session, view.plot, {"ranges": {"nav0": (1, 4)}})
        scan = next(box for box in view.controller.boxes if box.plot.is_navigator)
        scan.widget.set(_notify=False, x=float(scan.widget.x) - 2.0)
        _drag(scan)
        assert view.controller.ranges["nav0"] == (0, 3)

    def test_a_drag_snaps_to_whole_pixels_on_release(self, opened):
        view = opened("2D")
        box = view.controller.boxes[0]
        box.widget.set(_notify=False, x=2.7, y=0.9, w=7.1, h=5.3)
        _drag(box, released=False)
        assert float(box.widget.x) == 2.7          # left alone under the cursor
        _drag(box)
        assert float(box.widget.x) == 2.5          # snapped on release
        assert view.controller.ranges["sig0"] == (3, 10)

    def test_reset_restores_every_axis(self, opened):
        view = opened("5D")
        crop_set_range(view.session, view.plot, {"key": "nav2", "start": 1, "stop": 3})
        crop_reset(view.session, view.plot)
        assert view.states[-1]["is_crop"] is False

    def test_close_takes_every_box_down(self, opened):
        view = opened("5D")
        widgets = [box.widget for box in view.controller.boxes]
        crop_close(view.session, view.plot)
        assert getattr(view.tree, "_crop_controller", None) is None
        assert all(widget.visible is False for widget in widgets)

    def test_closing_the_tree_takes_the_boxes_down(self, opened):
        view = opened("4D")
        widgets = [box.widget for box in view.controller.boxes]
        view.tree.close()
        assert all(widget.visible is False for widget in widgets)


class TestResult:
    @pytest.mark.parametrize("name", sorted(CASES))
    def test_the_crop_opens_as_a_new_dataset(self, opened, name):
        view = opened(name)
        ranges = {axis["key"]: (1, axis["size"] - 1)
                  for axis in view.states[-1]["axes"]}
        crop_set_range(view.session, view.plot, {"ranges": ranges})
        trees = len(view.session.signal_trees)
        crop_run(view.session, view.plot)
        _settle(view.session, 4.0)

        assert len(view.session.signal_trees) == trees + 1
        result = view.session.signal_trees[-1].root
        shape = CASES[name][0]
        expected = view.data[tuple(slice(1, n - 1) for n in shape)]
        assert np.array_equal(np.asarray(result.data), expected)
        assert result.metadata.General.title == "sample (cropped)"
        assert view.tree.root.data.shape == shape, "the source was changed"
        assert view.controller._closed and not view.controller.boxes

    def test_a_cropped_movie_navigates_only_its_frames(self, opened):
        """The bug that started this: the source's navigator kept all 6
        frames, and frames outside the crop showed the nearest edge one."""
        view = opened("3D")
        crop_set_range(view.session, view.plot, {"key": "nav0", "start": 1, "stop": 4})
        crop_run(view.session, view.plot)
        _settle(view.session, 4.0)
        result = view.session.signal_trees[-1]
        time_axis = result.root.axes_manager.navigation_axes[0]
        assert (time_axis.size, time_axis.offset) == (3, pytest.approx(0.05))
        navigator = next(p for p in view.session._plots
                         if p.is_navigator and p.signal_tree is result)
        x, _units, _label = navigator._axes_info_1d(np.zeros(3))
        assert x == pytest.approx([0.05, 0.10, 0.15])

    def test_a_scan_crop_reuses_the_finished_navigator(self, opened):
        """The navigator sums over the detector, so with the detector whole it
        is the source's, cut — no pass over the cropped data."""
        view = opened("4D")
        source_navigator = np.asarray(
            view.tree.navigator_signals["base"][-1].data).copy()
        crop_set_range(view.session, view.plot, {"ranges": {
            "nav0": (2, 6), "nav1": (1, 4)}})
        crop_run(view.session, view.plot)
        result = view.session.signal_trees[-1]
        navigator = np.asarray(result.navigator_signals["base"][-1].data)
        assert np.array_equal(navigator, source_navigator[1:4, 2:6])

    def test_a_time_series_of_scans_keeps_both_navigators(self, opened):
        view = opened("5D")
        crop_set_range(view.session, view.plot, {"ranges": {
            "nav2": (1, 3), "nav0": (2, 6), "nav1": (1, 4)}})
        crop_run(view.session, view.plot)
        _settle(view.session, 4.0)
        result = view.session.signal_trees[-1]
        navigators = sorted(
            (len(p.current_data.shape) if p.current_data is not None else 0)
            for p in view.session._plots
            if p.is_navigator and p.signal_tree is result)
        assert navigators == [1, 2], "a 5-D result needs its time line and scan image"
        assert result.root.axes_manager.navigation_shape == (4, 3, 2)

    def test_nothing_to_cut_opens_nothing(self, opened):
        view = opened("4D")
        trees = len(view.session.signal_trees)
        crop_run(view.session, view.plot)
        assert len(view.session.signal_trees) == trees
        assert not view.controller._closed, "the panel stays up to try again"


class TestTheTimeNavigatorOfAScanSeries:
    def test_is_calibrated_by_the_time_axis(self, window):
        """A 5-D stack's line navigator draws its slowest axis. It used to read
        the first one (the scan's x): ticks in the scan's units, or none."""
        session = window["window"]
        signal, _data = _signal((4, 5, 7, 6, 6), lazy=True, time_index=2,
                                time_scale=0.5)
        tree = session._add_signal(signal, source_path=None)
        _settle(session, 4.0)
        line = next(p for p in session._plots
                    if p.is_navigator and p.signal_tree is tree
                    and getattr(p, "_plot1d", None) is not None)
        x, units, _label = line._axes_info_1d(np.zeros(4))
        assert x == pytest.approx([0.0, 0.5, 1.0, 1.5])
        assert units == "time (s)"


class TestFollowsTheDisplayedNode:
    def test_a_node_switch_rebuilds_the_panel(self, opened):
        from spyde.actions.base import Rebin2DAction
        from spyde.actions.context import ActionContext

        view = opened("4D")
        first = view.controller
        params = {"scale_x": 2, "scale_y": 2, "scan_x": 1, "scan_y": 1}
        Rebin2DAction(ActionContext(plot=view.plot, params=params,
                                    action_name="Rebin")).run(**params)
        assert first._closed
        controller = view.tree._crop_controller
        assert controller is not None and controller is not first
        sizes = {axis["key"]: axis["size"] for axis in view.states[-1]["axes"]}
        assert (sizes["sig0"], sizes["sig1"]) == (15, 10)
