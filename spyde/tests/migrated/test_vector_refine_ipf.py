"""Double-click masking on the vector orientation heat map.

The heat map shows how well every sampled orientation explains the pattern
under the crosshair. Double-clicking a triangle draws a circle that RESTRICTS
the match to the orientations inside it — the gesture and the meaning are the
dense refine heat map's, and the geometry is literally its code
(``rot_mask_from_circles``).

What is new here, and what these cover, is the join: a circle is in one phase's
stereographic coordinates, while the matcher wants one keep-array per phase
over that phase's own zone axes. The mask itself is exercised against a real
plan in ``test_quantem_adapter.TestZoneMask``; these are the wiring.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.actions.vector_refine_ipf import VectorRefineIpfController


class _StubFitter:
    """Records what the controller hands the matcher."""

    def __init__(self, counts):
        self._counts = list(counts)
        self.masks = []

    @property
    def zone_counts(self):
        return list(self._counts)

    def set_zone_mask(self, keep_per_phase):
        self.masks.append(keep_per_phase)


def _info(phase_index, xs, ys, lib_idx):
    """The geometry entry ``build_zone_ipf`` produces per phase."""
    return {
        "phase_index": phase_index,
        "xs": np.asarray(xs, float),
        "ys": np.asarray(ys, float),
        "lib_idx": np.asarray(lib_idx, int),
        "mins": np.array([0.0, 0.0]),
        "maxs": np.array([1.0, 1.0]),
    }


def _controller(infos, counts, fit_overlay=None):
    controller = VectorRefineIpfController(
        vectors=None, fitter=_StubFitter(counts), infos=infos, panels=[],
        fit_overlay=fit_overlay)
    # attach() is what wires the overlay and the double-click; these tests are
    # about what happens after a click, so the node stays absent and refresh()
    # is a no-op unless a stub tree is set.
    return controller


class _StubTree:
    """Records which overlay nodes were told their inputs moved."""

    def __init__(self):
        self.refreshed = []

    def replace_overlay_static(self, node, **static):
        self.refreshed.append(node)


class TestToggleCircle:
    def test_a_click_adds_a_circle_and_a_second_click_inside_removes_it(self):
        controller = _controller([_info(0, [0.1], [0.1], [0])], [1])
        controller.toggle_circle(0, 0.5, 0.5)
        assert len(controller.circles[0]) == 1
        cx, cy, radius = controller.circles[0][0]
        assert (cx, cy) == (0.5, 0.5) and radius > 0

        controller.toggle_circle(0, cx + radius * 0.5, cy)   # inside
        assert controller.circles[0] == []

    def test_a_click_outside_adds_a_second_circle(self):
        controller = _controller([_info(0, [0.1], [0.1], [0])], [1])
        controller.toggle_circle(0, 0.2, 0.2)
        controller.toggle_circle(0, 0.9, 0.9)
        assert len(controller.circles[0]) == 2

    def test_a_click_on_an_unknown_phase_is_ignored(self):
        controller = _controller([_info(0, [0.1], [0.1], [0])], [1])
        controller.toggle_circle(7, 0.5, 0.5)
        assert controller.circles == {0: []}


class TestMaskReachesTheMatcher:
    """The join: circles in one phase's triangle → one keep-array per phase,
    each over that phase's own zone axes."""

    def _two_phases(self):
        # Phase 0 has 3 zone axes (global 0..2), phase 1 has 2 (global 3..4) —
        # the order build_zone_ipf stacks them in.
        infos = [
            _info(0, [0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0, 1, 2]),
            _info(1, [0.0, 1.0], [0.0, 1.0], [3, 4]),
        ]
        return _controller(infos, [3, 2])

    def test_no_circles_means_no_restriction(self):
        controller = self._two_phases()
        controller.apply_mask()
        assert controller.fitter.masks == [None]

    def test_a_circle_keeps_only_what_it_covers_in_that_phase(self):
        controller = self._two_phases()
        controller.toggle_circle(0, 0.5, 0.5)       # covers phase 0's middle
        keep = controller.fitter.masks[-1]

        assert len(keep) == 2, "one keep-array per phase"
        np.testing.assert_array_equal(keep[0], [False, True, False])
        # A phase with NO circle keeps everything — a restriction drawn on one
        # phase must not quietly rule the other one out.
        np.testing.assert_array_equal(keep[1], [True, True])

    def test_each_phase_is_cut_from_its_own_block_of_the_global_mask(self):
        """The mask is built over the phases' zone axes end to end, so the
        split has to use each phase's length. Getting the offset wrong masks a
        real orientation in the wrong crystal, which looks like a bad fit
        rather than like a bug."""
        controller = self._two_phases()
        controller.toggle_circle(1, 1.0, 1.0)       # phase 1's second zone
        keep = controller.fitter.masks[-1]

        np.testing.assert_array_equal(keep[0], [True, True, True])
        np.testing.assert_array_equal(keep[1], [False, True])
        assert [len(k) for k in keep] == controller.fitter.zone_counts

    def test_removing_the_last_circle_clears_the_restriction(self):
        controller = self._two_phases()
        controller.toggle_circle(0, 0.5, 0.5)
        controller.toggle_circle(0, 0.5, 0.5)       # same spot → removes it
        assert controller.circles[0] == []
        assert controller.fitter.masks[-1] is None


class TestWiring:
    def test_every_panel_gets_a_double_click_handler(self):
        handlers = []

        class _XY:
            def add_event_handler(self, fn, event):
                handlers.append((fn, event))

        controller = _controller([_info(0, [0.1], [0.1], [0]),
                                  _info(1, [0.2], [0.2], [1])], [1, 1])
        controller.panels = [{"info": i, "xy": _XY()} for i in controller.infos]
        for panel in controller.panels:
            controller._wire_double_click(panel)

        assert [event for _fn, event in handlers] == ["double_click"] * 2

    def test_the_handler_reads_the_click_coordinates_from_either_shape(self):
        """anyplotlib hands an event object; a plain dict is what the tests and
        some hosts send. Both must reach the same circle."""
        controller = _controller([_info(0, [0.1], [0.1], [0])], [1])
        captured = []

        class _XY:
            def add_event_handler(self, fn, event):
                captured.append(fn)

        controller._wire_double_click({"info": controller.infos[0], "xy": _XY()})
        on_double_click = captured[0]

        on_double_click({"xdata": 0.25, "ydata": 0.75})
        assert controller.circles[0][0][:2] == (0.25, 0.75)

        controller.circles[0].clear()
        on_double_click(type("E", (), {"xdata": 0.4, "ydata": 0.6})())
        assert controller.circles[0][0][:2] == (0.4, 0.6)

    def test_a_click_with_no_coordinates_does_nothing(self):
        controller = _controller([_info(0, [0.1], [0.1], [0])], [1])
        captured = []

        class _XY:
            def add_event_handler(self, fn, event):
                captured.append(fn)

        controller._wire_double_click({"info": controller.infos[0], "xy": _XY()})
        captured[0](None)
        assert controller.circles[0] == []


class TestTheMaskReachesThePattern:
    """The heat map and the matched-pattern overlay fit through ONE matcher, so
    a restriction drawn on the triangle governs the green pattern too. They are
    separate overlay nodes with separate cached readers, though, so both have
    to be invalidated — otherwise the user rules an orientation out and the
    pattern carries on showing it until the navigator happens to move.
    """

    def _wired(self):
        heat_map_node, pattern_node = object(), object()
        controller = _controller([_info(0, [0.1], [0.1], [0])], [1],
                                 fit_overlay=pattern_node)
        controller.tree = _StubTree()
        controller.node = heat_map_node
        return controller, heat_map_node, pattern_node

    def test_both_overlays_are_re_evaluated(self):
        controller, heat_map_node, pattern_node = self._wired()
        controller.toggle_circle(0, 0.5, 0.5)
        assert controller.tree.refreshed == [heat_map_node, pattern_node]

    def test_without_a_pattern_overlay_only_the_heat_map_is_refreshed(self):
        controller, heat_map_node, _pattern = self._wired()
        controller.fit_overlay = None
        controller.toggle_circle(0, 0.5, 0.5)
        assert controller.tree.refreshed == [heat_map_node]

    def test_a_refresh_failure_on_one_node_does_not_skip_the_other(self):
        """A heat map that failed to redraw must not also leave the pattern
        showing a masked orientation."""
        controller, heat_map_node, pattern_node = self._wired()
        tree = controller.tree

        def _explode(node, **static):
            if node is heat_map_node:
                raise RuntimeError("boom")
            tree.refreshed.append(node)

        tree.replace_overlay_static = _explode
        controller.toggle_circle(0, 0.5, 0.5)
        assert tree.refreshed == [pattern_node]


class TestClosingLiftsTheRestriction:
    """The mask lives on the matcher, which Compute Maps and the pattern
    overlay both go on using. Closing the heat map takes away the circles that
    would lift it, so the restriction has to be lifted with them — otherwise
    every later fit is silently confined to a region nobody can see."""

    def _masked(self):
        pattern_node = object()
        controller = _controller([_info(0, [0.1], [0.1], [0])], [1],
                                 fit_overlay=pattern_node)
        controller.tree = _StubTree()
        controller.node = object()
        controller.toggle_circle(0, 0.5, 0.5)
        assert controller.fitter.masks[-1] is not None, "precondition"
        controller.tree.refreshed.clear()
        return controller, pattern_node

    def test_closing_clears_the_mask(self):
        controller, _pattern = self._masked()
        controller.close()
        assert controller.fitter.masks[-1] is None
        assert controller.circles == {0: []}

    def test_closing_redraws_the_pattern_without_the_restriction(self):
        controller, pattern_node = self._masked()
        controller.close()
        assert controller.tree.refreshed == [pattern_node]

    def test_closing_without_a_mask_touches_nothing(self):
        controller = _controller([_info(0, [0.1], [0.1], [0])], [1],
                                 fit_overlay=object())
        controller.tree = _StubTree()
        controller.node = object()
        controller.close()
        assert controller.fitter.masks == []
        assert controller.tree.refreshed == []

    def test_a_closing_tree_is_not_made_to_re_fit(self):
        """This same close runs on tree teardown. Lifting the mask there still
        has to happen — the matcher may outlive the window — but re-evaluating
        would spend a fit on a navigator that is going away."""
        controller, _pattern = self._masked()
        controller.tree._spyde_closed = True
        controller.close()
        assert controller.fitter.masks[-1] is None, "the mask must still lift"
        assert controller.tree.refreshed == [], "no fit on a closing tree"
