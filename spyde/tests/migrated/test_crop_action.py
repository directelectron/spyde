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

from spyde.actions.base import (
    CropAction, _crop_signal, crop_open, crop_close, crop_set_region,
)


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


class TestCropROI:
    """Crop (laundry item #4): activating the action shows a draggable
    rectangle widget on the source plot; the user adjusts it; Run/Apply reads
    the WIDGET's live geometry (the primary input, same as the CZB
    search-window precedent), not stale typed fields. Non-square + partially
    out-of-bounds regions are exercised, and the widget tears down on close /
    after a successful crop."""

    def _plot(self, session):
        return next(p for p in session._plots
                    if not p.is_navigator and p.plot_state is not None)

    def test_open_shows_draggable_widget_covering_full_frame(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        plot = self._plot(session)
        tree = plot.signal_tree
        signal = plot.plot_state.current_signal
        w, h = (int(ax.size) for ax in signal.axes_manager.signal_axes)

        crop_open(session, plot, {})
        widget = getattr(tree, "_crop_widget", None)
        assert widget is not None, "no crop widget was created"
        # Draggable/resizable, not a static marker.
        assert hasattr(widget, "add_event_handler")
        assert hasattr(widget, "set") and hasattr(widget, "hide")
        assert (float(widget.x), float(widget.y)) == (0.0, 0.0)
        assert (float(widget.w), float(widget.h)) == (float(w), float(h))

        crop_close(session, plot, {})
        assert getattr(tree, "_crop_widget", None) is None

    def test_widget_drives_a_nonsquare_crop(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        plot = self._plot(session)
        tree = plot.signal_tree

        crop_open(session, plot, {})
        widget = tree._crop_widget
        # Simulate the user dragging the rectangle to an ASYMMETRIC box:
        # x 2..14 (12 cols), y 4..10 (6 rows).
        widget.set(x=2.0, y=4.0, w=12.0, h=6.0, _push=False)

        # Run picks up the WIDGET's geometry — typed fields are stale/absent.
        act = CropAction.for_plot(plot)
        new = act.run()
        time.sleep(0.2)

        assert new is not None
        assert tuple(new.axes_manager.signal_shape) == (12, 6)
        # A successful crop tears the (now stale) box down.
        assert getattr(tree, "_crop_widget", None) is None

    def test_widget_partially_out_of_bounds_is_clamped(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        plot = self._plot(session)
        tree = plot.signal_tree
        signal = plot.plot_state.current_signal
        w, h = (int(ax.size) for ax in signal.axes_manager.signal_axes)

        crop_open(session, plot, {})
        widget = tree._crop_widget
        # Drag a corner past the top-left AND past the bottom-right edges —
        # negative x/y and w/h that overflow past (w, h).
        widget.set(x=-5.0, y=-3.0, w=float(w + 20), h=float(h + 20), _push=False)

        act = CropAction.for_plot(plot)
        new = act.run()
        time.sleep(0.2)

        assert new is not None
        # Clamped to the full valid frame (0..w, 0..h) rather than raising or
        # producing an inverted/out-of-range slice.
        assert tuple(new.axes_manager.signal_shape) == (w, h)

    def test_typed_field_edit_moves_the_widget(self, stem_4d_dataset):
        # Bidirectional sync: a typed-field edit (crop_set_region) repositions
        # the on-plot widget to match (cheap — no recompute involved).
        session = stem_4d_dataset["window"]
        plot = self._plot(session)
        tree = plot.signal_tree

        crop_open(session, plot, {})
        crop_set_region(session, plot, {"x0": 1, "x1": 9, "y0": 2, "y1": 8})
        widget = tree._crop_widget
        assert (float(widget.x), float(widget.y)) == (1.0, 2.0)
        assert (float(widget.w), float(widget.h)) == (8.0, 6.0)

    def test_close_tears_down_widget_without_running(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        plot = self._plot(session)
        tree = plot.signal_tree

        crop_open(session, plot, {})
        widget = tree._crop_widget
        assert widget.visible is True

        crop_close(session, plot, {})
        assert getattr(tree, "_crop_widget", None) is None
        assert widget.visible is False

    def test_full_frame_widget_is_a_noop_run(self, stem_4d_dataset):
        # Opening Crop and hitting Run without adjusting the box must not add a
        # redundant "Cropped" node identical to the source.
        session = stem_4d_dataset["window"]
        plot = self._plot(session)
        tree = plot.signal_tree
        root = tree.root

        crop_open(session, plot, {})
        act = CropAction.for_plot(plot)
        new = act.run()
        assert new is root, "an unadjusted full-frame box must be a no-op"
        # And no redundant tree node was created (add_transformation has no
        # identity check — CropAction.run must short-circuit BEFORE it).
        assert not any("Cropped" in k for k in tree.get_node(root).children)
        # The box stays up so the user can adjust and try again.
        assert getattr(tree, "_crop_widget", None) is not None

    def test_drag_clamp_handler_runs_once_per_event(self, stem_4d_dataset):
        # REGRESSION (reviewer repro): anyplotlib Widget.set() fires
        # pointer_move UNCONDITIONALLY (even for a no-change write, regardless
        # of _push), so the clamp handler's own set() used to re-enter the
        # handler recursively — ONE JS drag frame recursed ~2000 deep and only
        # survived via a bumped recursionlimit + a swallowing try/except,
        # spamming hundreds of redundant pushes. Drive the REAL JS→Python path
        # (_update_from_js) and count the clamp's set() executions.
        session = stem_4d_dataset["window"]
        plot = self._plot(session)
        tree = plot.signal_tree

        crop_open(session, plot, {})
        widget = tree._crop_widget

        calls = []
        real_set = widget.set
        def counting_set(*a, **kw):
            calls.append(1)
            return real_set(*a, **kw)
        # NB object.__setattr__: a plain `widget.set = ...` would be routed
        # through Widget.__setattr__ into set(set=...) instead of rebinding.
        object.__setattr__(widget, "set", counting_set)

        widget._update_from_js(
            {"x": -5.0, "y": -3.0, "w": 100.0, "h": 100.0}, "pointer_move")

        assert len(calls) == 1, \
            f"clamp must run exactly ONCE per JS event, ran {len(calls)}x"
        # And the clamp genuinely applied (16x16 frame).
        assert (float(widget.x), float(widget.y)) == (0.0, 0.0)
        assert (float(widget.w), float(widget.h)) == (16.0, 16.0)

    def test_node_switch_tears_down_widget(self, stem_4d_dataset):
        # Switching the displayed node (show_tree_node — fired by ANY
        # transform or the Workflow panel) must hide the crop box: the caret
        # stays mounted across a node switch (crop_close never fires), and
        # the box no longer describes the node now on screen.
        from spyde.actions.lifecycle import show_tree_node
        session = stem_4d_dataset["window"]
        plot = self._plot(session)
        tree = plot.signal_tree

        crop_open(session, plot, {})
        widget = tree._crop_widget
        assert widget is not None

        show_tree_node(plot, tree, tree.root)
        assert getattr(tree, "_crop_widget", None) is None
        assert getattr(tree, "_crop_widget_handler", None) is None
        assert widget.visible is False

    def test_tree_close_tears_down_widget(self, stem_4d_dataset):
        # BaseSignalTree.close() is the teardown authority — the crop widget
        # must be in its attr sweep (it was a hard-coded list).
        session = stem_4d_dataset["window"]
        plot = self._plot(session)
        tree = plot.signal_tree

        crop_open(session, plot, {})
        widget = tree._crop_widget
        tree.close()
        assert getattr(tree, "_crop_widget", None) is None
        assert getattr(tree, "_crop_widget_handler", None) is None
        assert widget.visible is False


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
