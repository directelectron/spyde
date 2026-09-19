"""
Multi-Angle 4-D STEM: switching the DISPLAYED node between a 5-D root and its
4-D child, inside ONE signal tree.

The planned loader builds one tree::

    Aligned Stack   (angle, y, x | ky, kx)   5-D   root
    └── Summed      (y, x | ky, kx)          4-D   sum over the angle axis

and the dock's Workflow section switches between them (``select_signal_node``).

Those two nodes are navigated DIFFERENTLY. A 5-D root opens with two chained
navigators — an angle line driving a real-space image driving the diffraction
pattern — so the innermost selector composes a THREE-coordinate navigation
index ``(angle, x, y)``. The 4-D child has two navigation dimensions and wants
``(x, y)``. The navigator chain is built once, from the ROOT's navigation
dimension, and no node switch rebuilds it — so the composed index still carries
its angle coordinate after the switch.

``update_functions._prepare_nav_indices`` therefore trims the composed index to
the DISPLAYED node's navigation dimension, keeping its last columns. Without
that trim the angle coordinate is consumed as the 4-D node's scan row and the
window silently paints one ROW of a diffraction pattern instead of the pattern —
no exception, no error message. These tests pin both halves: the frame the 4-D
node displays, and that the angle navigator can no longer move a node that has
no angle axis.
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs

from spyde.drawing.selectors.base_selector import _nav_dispatcher

ANGLES, SCAN_ROWS, SCAN_COLUMNS = 3, 4, 4
DETECTOR_ROWS, DETECTOR_COLUMNS = 8, 8

# The navigation position both navigators are pinned to for every test here.
ANGLE_INDEX, SCAN_ROW, SCAN_COLUMN = 1, 3, 2


def _aligned_stack():
    """``(angle, y, x | ky, kx)`` — a tiny stand-in for the aligned stack.

    The values are random so a frame read at the wrong rank, or from the wrong
    navigation position, cannot coincidentally match the expected one.
    """
    data = np.random.default_rng(0).random(
        (ANGLES, SCAN_ROWS, SCAN_COLUMNS, DETECTOR_ROWS, DETECTOR_COLUMNS)
    ).astype(np.float32)
    signal = hs.signals.Signal2D(data)
    signal.set_signal_type("electron_diffraction")
    # The slowest navigation axis is the leading data axis — the angle.
    signal.axes_manager.navigation_axes[-1].name = "angle"
    return signal


def summed_over_angles(signal):
    """The 4-D node: integrate the leading angle axis away."""
    return signal.sum(signal.axes_manager.navigation_axes[-1])


def _settle_navigator(timeout: float = 5.0) -> bool:
    """Wait until the serial navigator dispatcher has run every queued update.

    Idle has to hold TWICE: a chained update (angle → real space → diffraction
    pattern) is submitted from inside the update it follows, so a single sample
    can land in the gap between the two and read the display one frame early.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _nav_dispatcher.idle():
            time.sleep(0.05)
            if _nav_dispatcher.idle():
                return True
        time.sleep(0.01)
    return _nav_dispatcher.idle()


def _signal_plot(session):
    """The diffraction-pattern plot — the only non-navigator window of a 5-D
    tree (the real-space image is an intermediate NAVIGATOR)."""
    return next(plot for plot in session._plots
                if not plot.is_navigator and plot.plot_state is not None)


def _navigation_selectors(tree):
    """``(angle_selector, scan_selector)``, the two composite selectors of the
    chain, outermost first."""
    by_name = {type(selector).__name__: selector
               for selector in tree.navigator_plot_manager.all_navigation_selectors}
    return by_name["IntegratingSelector1D"], by_name["IntegratingSSelector2D"]


def _open_multiangle(session):
    """Open the 5-D stack, add the 4-D Summed child, pin both navigators.

    Returns ``(tree, root_signal, summed_signal, signal_plot)`` with the
    diffraction pattern already showing the root's frame at the pinned
    position.
    """
    root_signal = _aligned_stack()
    session._add_signal(root_signal, source_path=None)
    tree = session.signal_trees[0]
    summed_signal = tree.add_transformation(
        root_signal, function=summed_over_angles, node_name="Summed")

    angle_selector, scan_selector = _navigation_selectors(tree)
    # Pin both navigators by reporting a fixed widget position, so the expected
    # frame is a known array rather than wherever the widgets happen to sit.
    # The 2-D selector reports widget order (column, row).
    angle_selector.selector._get_selected_indices = \
        lambda: np.array([[ANGLE_INDEX]])
    scan_selector.selector._get_selected_indices = \
        lambda: np.array([[SCAN_COLUMN, SCAN_ROW]])
    scan_selector.selector._run_update(force=True)
    _settle_navigator()
    return tree, root_signal, summed_signal, _signal_plot(session)


class TestMultiAngleNodeSwitch:
    def test_the_stack_opens_with_two_chained_navigators(self, window):
        """The premise: a 5-D root is a two-level navigator chain, and its
        innermost selector composes a three-coordinate index."""
        session = window["window"]
        tree, root_signal, _, signal_plot = _open_multiangle(session)

        assert tree.navigator_plot_manager.navigation_depth == 2
        assert len(tree.signal_plots) == 1
        _, scan_selector = _navigation_selectors(tree)
        assert scan_selector.get_selected_indices().tolist() == \
            [[ANGLE_INDEX, SCAN_COLUMN, SCAN_ROW]]
        assert np.allclose(
            signal_plot.current_data,
            root_signal.data[ANGLE_INDEX, SCAN_ROW, SCAN_COLUMN])

    def test_the_summed_node_is_selectable_from_the_workflow_panel(self, window):
        """The dock's pick reaches the 4-D node: it has a PlotState on the
        diffraction window, the handler finds it by signal id, and the tree is
        re-emitted with it marked active."""
        session = window["window"]
        messages = window["messages"]
        _, _, summed_signal, signal_plot = _open_multiangle(session)

        assert summed_signal.axes_manager.navigation_dimension == 2
        assert summed_signal in signal_plot.plot_states

        messages.clear()
        session._select_signal_node(signal_plot, id(summed_signal))
        _settle_navigator()

        assert signal_plot.plot_state.current_signal is summed_signal
        trees = [m for m in messages if m.get("type") == "signal_tree"]
        assert trees and trees[-1]["active_signal_id"] == id(summed_signal)

    def test_the_4d_node_displays_its_diffraction_pattern(self, window):
        """The whole pattern, at the scan position the inner navigator holds —
        not the one-row slice an untrimmed ``(angle, y, x)`` index reads out of
        a 4-D array."""
        session = window["window"]
        _, _, summed_signal, signal_plot = _open_multiangle(session)

        session._select_signal_node(signal_plot, id(summed_signal))
        _settle_navigator()

        assert signal_plot.current_data.shape == (DETECTOR_ROWS, DETECTOR_COLUMNS)
        assert np.allclose(signal_plot.current_data,
                           summed_signal.data[SCAN_ROW, SCAN_COLUMN])

    def test_the_angle_navigator_cannot_move_the_4d_frame(self, window):
        """The 4-D node integrated the angle axis away, so the angle navigator
        has nothing left to select in it — the frame must not change."""
        session = window["window"]
        tree, _, summed_signal, signal_plot = _open_multiangle(session)

        session._select_signal_node(signal_plot, id(summed_signal))
        _settle_navigator()
        at_first_angle = np.array(signal_plot.current_data, copy=True)

        angle_selector, _ = _navigation_selectors(tree)
        angle_selector.selector._get_selected_indices = \
            lambda: np.array([[ANGLE_INDEX + 1]])
        angle_selector.selector._run_update(force=True)
        _settle_navigator()

        assert np.array_equal(signal_plot.current_data, at_first_angle)
        assert np.allclose(signal_plot.current_data,
                           summed_signal.data[SCAN_ROW, SCAN_COLUMN])

    def test_switching_back_to_the_5d_root_restores_its_frame(self, window):
        """The round trip is clean: going back to the root shows the root's own
        frame at the unchanged position again — the trim reads the index the
        displayed node needs without consuming the one the chain composes."""
        session = window["window"]
        _, root_signal, summed_signal, signal_plot = _open_multiangle(session)

        session._select_signal_node(signal_plot, id(summed_signal))
        _settle_navigator()
        session._select_signal_node(signal_plot, id(root_signal))
        _settle_navigator()

        assert signal_plot.plot_state.current_signal is root_signal
        assert np.allclose(
            signal_plot.current_data,
            root_signal.data[ANGLE_INDEX, SCAN_ROW, SCAN_COLUMN])
