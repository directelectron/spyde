"""The Plot Control dock describes the node on screen, not the tree's root.

Reported after binning a scan: with the Binned node selected the dock still
said "nav 24 × 24" and showed scale 1.00 on the binned axes. Two things kept
it on the root: the axes table read ``tree.root``, and nothing re-sent the
table or the dataset summary when the displayed node changed.

Once the table shows a node's axes, an edit made there has to land on that
node's axes, and the same calibration has to reach every other node of the
tree. The crosshair finds a binned or cropped node's position through the
axes' calibration, so a scale typed into one node only would leave the
navigator pointing at the wrong frames.
"""
from __future__ import annotations

from unittest.mock import patch

import hyperspy.api as hs
import numpy as np
from de_shell import ipc

from spyde.tests.migrated.conftest import close_session, make_session

SCAN_ROWS, SCAN_COLUMNS, DETECTOR = 6, 10, 4


def _session():
    data = np.zeros((SCAN_ROWS, SCAN_COLUMNS, DETECTOR, DETECTOR), np.uint16)
    for row in range(SCAN_ROWS):
        for column in range(SCAN_COLUMNS):
            data[row, column, 0, 0] = 100 * row + column
    signal = hs.signals.Signal2D(data)
    signal.set_signal_type("electron_diffraction")
    session = make_session()
    session._add_signal(signal)
    return session


def _data_plot(session):
    return next(plot for plot in session._plots
                if not getattr(plot, "is_navigator", False)
                and getattr(getattr(plot, "plot_state", None),
                            "current_signal", None) is not None)


def _run(action_class, plot, **params):
    from spyde.actions.context import ActionContext
    action = action_class(ActionContext(plot=plot, params=params,
                                        action_name=action_class.name))
    new = action.run(**params)
    assert new is not None, f"{action_class.name} produced no node"
    return new


def _bin_scan_x(plot):
    from spyde.actions.base import Rebin2DAction
    return _run(Rebin2DAction, plot, scan_x=2, scan_y=1, scale_x=1, scale_y=1)


class _Captured:
    """The dock's two messages, as last sent."""

    def __init__(self):
        self.axes = None
        self.metadata = None
        self._forward = ipc.emit

    def __call__(self, message):
        if isinstance(message, dict):
            if message.get("type") == "axes_info":
                self.axes = message["axes"]
            elif message.get("type") == "metadata":
                self.metadata = message["metadata"]
        self._forward(message)


def _row(axes, navigate, position):
    """The ``position``-th navigation or signal row, in array order."""
    return [row for row in axes if row["navigate"] == navigate][position]


class TestTheDockDescribesTheSelectedNode:
    def test_the_axes_table_is_the_binned_nodes(self):
        from spyde.metadata_extract import build_axes_list
        session = _session()
        try:
            plot = _data_plot(session)
            _bin_scan_x(plot)
            axes = build_axes_list(plot.signal_tree)
            # Array order: scan y, then scan x (binned by 2).
            assert [row["size"] for row in axes] == [6, 5, 4, 4]
            assert _row(axes, True, 1)["scale"] == 2.0
            assert _row(axes, True, 0)["scale"] == 1.0
        finally:
            close_session(session)

    def test_running_rebin_resends_the_table_and_the_shape(self):
        captured = _Captured()
        session = _session()
        try:
            plot = _data_plot(session)
            with patch.object(ipc, "emit", captured):
                _bin_scan_x(plot)
            assert captured.axes is not None, "no axes table was sent"
            assert [row["size"] for row in captured.axes] == [6, 5, 4, 4]
            assert captured.metadata is not None, "no dataset summary was sent"
            assert captured.metadata["Dataset"]["Shape"] == "nav 5 × 6 · sig 4 × 4"
        finally:
            close_session(session)

    def test_switching_back_to_the_root_describes_the_root(self):
        captured = _Captured()
        session = _session()
        try:
            plot = _data_plot(session)
            root = plot.signal_tree.root
            _bin_scan_x(plot)
            with patch.object(ipc, "emit", captured):
                session._select_signal_node(plot, id(root))
            assert [row["size"] for row in captured.axes] == [6, 10, 4, 4]
            assert captured.metadata["Dataset"]["Shape"] == "nav 10 × 6 · sig 4 × 4"
        finally:
            close_session(session)

    def test_a_cropped_node_shows_its_offset(self):
        from spyde.actions.base import CropAction
        from spyde.metadata_extract import build_axes_list
        session = _session()
        try:
            plot = _data_plot(session)
            _run(CropAction, plot, scan_x0=2, scan_x1=8, scan_y0=1, scan_y1=5)
            axes = build_axes_list(plot.signal_tree)
            assert [row["size"] for row in axes] == [4, 6, 4, 4]
            assert _row(axes, True, 0)["offset"] == 1.0
            assert _row(axes, True, 1)["offset"] == 2.0
        finally:
            close_session(session)


class TestAnEditReachesEveryNode:
    def _x_index(self, plot):
        from spyde.metadata_extract import build_axes_list
        return _row(build_axes_list(plot.signal_tree), True, 1)["index"]

    def test_a_scale_typed_on_the_binned_node_lands_on_it_and_the_root(self):
        session = _session()
        try:
            plot = _data_plot(session)
            root = plot.signal_tree.root
            binned = _bin_scan_x(plot)
            session._set_axis(plot, {"index": self._x_index(plot),
                                     "field": "scale", "value": "0.5"})
            assert binned.axes_manager.navigation_axes[0].scale == 0.5
            assert root.axes_manager.navigation_axes[0].scale == 0.25
        finally:
            close_session(session)

    def test_a_scale_typed_on_the_root_reaches_the_binned_node(self):
        session = _session()
        try:
            plot = _data_plot(session)
            root = plot.signal_tree.root
            binned = _bin_scan_x(plot)
            session._select_signal_node(plot, id(root))
            session._set_axis(plot, {"index": self._x_index(plot),
                                     "field": "scale", "value": "3"})
            assert root.axes_manager.navigation_axes[0].scale == 3.0
            assert binned.axes_manager.navigation_axes[0].scale == 6.0
        finally:
            close_session(session)

    def test_an_offset_typed_on_the_binned_node_shifts_every_node(self):
        session = _session()
        try:
            plot = _data_plot(session)
            root = plot.signal_tree.root
            binned = _bin_scan_x(plot)
            before = binned.axes_manager.navigation_axes[0].offset
            session._set_axis(plot, {"index": self._x_index(plot),
                                     "field": "offset", "value": str(before + 7)})
            assert root.axes_manager.navigation_axes[0].offset == 7.0
        finally:
            close_session(session)

    def test_the_crosshair_still_reads_the_frame_under_it_after_an_edit(self):
        from spyde.drawing.update_functions import \
            update_from_navigation_selection
        session = _session()
        try:
            plot = _data_plot(session)
            binned = _bin_scan_x(plot)
            session._set_axis(plot, {"index": self._x_index(plot),
                                     "field": "scale", "value": "0.3"})
            selectors = plot.signal_tree.navigator_plot_manager.navigation_selectors
            selector = next(each for group in selectors.values() for each in group)
            frame = update_from_navigation_selection(
                selector, plot, np.array([[6, 2]]))
            assert int(np.asarray(frame)[0, 0]) == \
                int(np.asarray(binned.data)[2, 3, 0, 0])
        finally:
            close_session(session)


class TestTheRebinCaretGroupsItsFields:
    def test_scan_and_diffraction_fields_are_separate_groups(self):
        from spyde import TOOLBAR_ACTIONS
        parameters = TOOLBAR_ACTIONS["functions"]["Rebin"]["parameters"]
        groups = [spec.get("group") for spec in parameters.values()]
        assert groups[0] == groups[1] and groups[2] == groups[3], groups
        assert groups[0] and groups[2] and groups[0] != groups[2], groups
