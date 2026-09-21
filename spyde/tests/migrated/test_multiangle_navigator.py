"""
The POLAR ANGLE NAVIGATOR — a multi-angle acquisition's 5th dimension, drawn.

``spyde.actions.multiangle_navigator`` opens a bare-figure window showing the
acquisition the way it physically happened: one ring per shell, one point per
member at ``(tilt·sin θ, tilt·cos θ)``. Its whole job is to be the picture AND
the control — so these tests check the two halves that can silently disagree:

* the **picture** matches the model (one point per member, at its polar
  position; shells at distinct radii), because a ring drawn from a copy of the
  angles would look right while naming the wrong member;
* the **binding runs both ways through the tree's REAL 1-D angle selector** — a
  pick moves that selector (never a private position of the window's own), and a
  move from anywhere else moves the ring's highlight.

Plus the two states the user toggles between with one click: every angle lit on
the Summed node, and one angle highlighted on the Aligned Stack — where clicking
on the sum is what switches the display to the stack.

The members are written to disk and opened through ``Session.open_multiangle``,
so the tree under test is the one a user gets, selector chain included.
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs
import pytest

from spyde.actions import multiangle_navigator as ring
from spyde.actions import registry
from spyde.drawing.selectors.base_selector import _nav_dispatcher
from spyde.multiangle import make_multiangle
from spyde.multiangle.model import MultiAngleModel, assign_shells
from spyde.multiangle.recipe import recipe_for
from spyde.tests.migrated._async import quiesce, wait_until, why_busy

SCAN = (20, 22)
DETECTOR = (16, 16)


def _wait(pred, timeout=30.0):
    return wait_until(pred, timeout)


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def two_shells(tmp_path_factory):
    """Five members at two tilts — unequal shells, which is the general case."""
    data = make_multiangle(shells=((1.0, 3), (0.5, 2)), scan_shape=SCAN,
                           detector_shape=DETECTOR, seed=3)
    directory = tmp_path_factory.mktemp("ring-two-shells")
    paths = []
    for index, member in enumerate(data.members):
        signal = hs.signals.Signal2D(member)
        signal.set_signal_type("electron_diffraction")
        path = str(directory / f"member{index:02d}.hspy")
        signal.save(path)
        paths.append(path)
    return data, paths


def _settle(session) -> None:
    """Wait for the session AND the navigator dispatcher to have nothing left.

    Idle has to hold TWICE: a chained update (angle → real space → diffraction
    pattern) is submitted from inside the update it follows, so a single sample
    can land in the gap between the two.
    """
    assert quiesce(session), why_busy(session)

    def _idle_twice():
        if not _nav_dispatcher.idle():
            return False
        time.sleep(0.05)
        return _nav_dispatcher.idle()

    assert _wait(_idle_twice), "the navigator dispatcher never went idle"


def _signal_plot(session):
    return next(plot for plot in session._plots if not plot.is_navigator)


def _open(session, acquisition):
    """Open the acquisition and wait until the Summed node is on screen."""
    data, paths = acquisition
    session.open_multiangle(paths, data.tilts.tolist(), data.azimuths.tolist(),
                            reference=data.reference)
    assert _wait(lambda: bool(session.signal_trees)
                 and "Summed" in session.signal_trees[0].root_node.children), \
        "the multi-angle tree never appeared"
    tree = session.signal_trees[0]
    summed = tree.root_node.children["Summed"].signal
    plot = _signal_plot(session)
    assert _wait(lambda: plot.plot_state is not None
                 and plot.plot_state.current_signal is summed), \
        "the Summed node never became the displayed one"
    _settle(session)
    return tree, summed, plot


def _open_ring(session, acquisition):
    """The acquisition open, with its polar angle navigator up."""
    tree, summed, plot = _open(session, acquisition)
    controller = ring.open_multiangle_navigator(session, tree)
    assert controller is not None, "the angle navigator never opened"
    return tree, summed, plot, controller


def _tree_angle_selector(tree):
    """The tree's own 1-D angle selector, found WITHOUT asking the window."""
    model = ring.multiangle_model(tree)
    selectors = [selector
                 for selector in tree.navigator_plot_manager.all_navigation_selectors
                 if ring._selector_axis_size(selector) == model.n_members]
    assert len(selectors) == 1, f"expected one angle selector, got {selectors}"
    return selectors[0]


def _selector_member(selector) -> int | None:
    indices = getattr(selector, "current_indices", None)
    return None if indices is None else int(np.asarray(indices).ravel()[0])


def model_tilt(tree, member: int) -> float:
    return float(ring.multiangle_model(tree).tilts[member])


def model_azimuth(tree, member: int) -> float:
    return float(ring.multiangle_model(tree).azimuths[member])


def _ring_hooks(selector, controller) -> list[int]:
    """How many of *controller*'s index hooks sit on each half of *selector*.

    A 1-D navigation selector is a composite whose crosshair and integrating
    span keep separate hook lists, and the ring hooks BOTH — so the count that
    matters is one per half, never two on the same one.
    """
    return [sum(hook is controller._index_hook for hook in target.index_hooks)
            for target in ring._index_hook_targets(selector)]


def _click(controller, x: float, y: float) -> None:
    """Fire the pointer event the figure's own handler receives on a click."""
    from anyplotlib.callbacks import Event

    controller._axes.callbacks.fire(
        Event("pointer_down", source=controller._axes,
              xdata=float(x), ydata=float(y), button=0, buttons=1))


# ── the picture ──────────────────────────────────────────────────────────────

class TestTheRingPicture:
    """What is drawn must come from the model, member by member."""

    def test_one_point_per_member_at_its_polar_position(self, window, two_shells):
        data, _paths = two_shells
        _tree, summed, _plot, controller = _open_ring(window["window"], two_shells)
        model = recipe_for(summed).model

        assert controller.positions.shape == (model.n_members, 2)
        azimuths = np.radians(model.azimuths)
        expected = np.column_stack([model.tilts * np.sin(azimuths),
                                    model.tilts * np.cos(azimuths)])
        assert np.allclose(controller.positions, expected)
        # Every member's distance from the centre IS its tilt — that is what
        # makes a gap in a ring read as a missing angle.
        assert np.allclose(np.linalg.norm(controller.positions, axis=1),
                           data.tilts)

    def test_shells_become_distinct_radii(self, window, two_shells):
        _tree, summed, _plot, controller = _open_ring(window["window"], two_shells)
        model = recipe_for(summed).model

        assert len(controller.radii) == model.n_shells == 2
        assert sorted(round(r, 6) for r in controller.radii.values()) == [0.5, 1.0]
        for shell, members in model.shells.items():
            radius = controller.radii[shell]
            assert np.allclose(
                np.linalg.norm(controller.positions[list(members)], axis=1),
                radius)

    def test_members_sharing_an_azimuth_are_addressed_separately(self):
        """A pick resolves to a MEMBER, never to an azimuth.

        Built by hand: the synthetic acquisition staggers its shells on purpose,
        so the case this guards against cannot occur there.
        """
        tilts = np.array([1.0, 1.0, 0.5, 0.5])
        azimuths = np.array([0.0, 90.0, 0.0, 90.0])   # shared across shells
        model = MultiAngleModel(
            paths=[f"member{i}" for i in range(4)], tilts=tilts,
            azimuths=azimuths, shell_ids=assign_shells(tilts),
            nav_offsets=np.zeros((4, 2), int), dp_offsets=np.zeros((4, 2), int))

        positions = ring.member_positions(model)
        assert len({tuple(p) for p in np.round(positions, 6)}) == 4
        for member, (x, y) in enumerate(positions):
            assert ring.nearest_member(positions, x, y) == member

    def test_the_summed_node_lights_every_angle_and_says_so(self, window,
                                                            two_shells):
        _tree, summed, _plot, controller = _open_ring(window["window"], two_shells)
        model = recipe_for(summed).model

        assert controller.showing_one_angle() is False
        assert controller.drawn_member_colors() == [ring.LIT_COLOR] * model.n_members
        # No single angle is the live one, so there is nothing to mark and the
        # drag handle is put away.
        assert controller._handle.visible is False
        assert controller.drawn_highlight().shape == (0, 2)
        assert controller.badge == "Σ all angles"


# ── picking drives the tree ──────────────────────────────────────────────────

class TestPickingAnAngle:
    """A pick must move the tree's own selector — the window keeps no copy."""

    def test_the_window_holds_the_trees_real_selector(self, window, two_shells):
        tree, _summed, _plot, controller = _open_ring(window["window"], two_shells)

        assert controller.selector is _tree_angle_selector(tree)

    def test_a_click_moves_the_real_selector(self, window, two_shells):
        session = window["window"]
        tree, _summed, _plot, controller = _open_ring(session, two_shells)
        selector = _tree_angle_selector(tree)
        target = 3

        _click(controller, *controller.positions[target])
        _settle(session)

        assert _selector_member(selector) == target

    def test_a_click_near_a_point_picks_that_member(self, window, two_shells):
        session = window["window"]
        tree, _summed, _plot, controller = _open_ring(session, two_shells)
        selector = _tree_angle_selector(tree)
        target = 1
        # Nudged off the point: the ring snaps to the nearest angle, which is
        # what makes dragging around it feel like scrubbing.
        x, y = controller.positions[target]
        _click(controller, x + 0.02, y - 0.02)
        _settle(session)

        assert _selector_member(selector) == target

    def test_a_click_on_the_summed_node_switches_to_the_stack(self, window,
                                                              two_shells):
        session = window["window"]
        tree, summed, plot, controller = _open_ring(session, two_shells)
        selector = _tree_angle_selector(tree)
        target = 2

        assert plot.plot_state.current_signal is summed
        _click(controller, *controller.positions[target])
        _settle(session)

        assert plot.plot_state.current_signal is tree.root, \
            "clicking an angle on the sum must show that angle's stack node"
        assert _selector_member(selector) == target
        # And the picture flips to the one-angle state it now describes.
        assert controller.showing_one_angle() is True
        assert np.allclose(controller.drawn_highlight(),
                           controller.positions[target].reshape(1, 2))
        assert controller.badge == (
            f"{ring._format_degrees(model_tilt(tree, target))} · "
            f"{ring._format_degrees(model_azimuth(tree, target))}")
        colors = controller.drawn_member_colors()
        assert colors[target] == ring.LIT_COLOR
        assert all(color == ring.DIM_COLOR
                   for index, color in enumerate(colors) if index != target)

    def test_a_second_pick_on_the_stack_scrubs_without_switching(self, window,
                                                                 two_shells):
        session = window["window"]
        tree, _summed, plot, controller = _open_ring(session, two_shells)
        selector = _tree_angle_selector(tree)

        _click(controller, *controller.positions[2])
        _settle(session)
        _click(controller, *controller.positions[4])
        _settle(session)

        assert plot.plot_state.current_signal is tree.root
        assert _selector_member(selector) == 4

    def test_dragging_the_handle_scrubs_the_angle(self, window, two_shells):
        """The stack node's live angle carries a draggable handle; dragging it
        round the ring is the other half of the gesture."""
        session = window["window"]
        tree, _summed, plot, controller = _open_ring(session, two_shells)
        selector = _tree_angle_selector(tree)

        session._select_signal_node(plot, id(tree.root))
        _settle(session)
        assert controller._handle.visible is True

        target = 3
        # A drag: the handle's own position changes and it reports it, exactly
        # as the renderer does mid-drag.
        x, y = controller.positions[target]
        controller._handle.set(x=float(x) + 0.03, y=float(y) - 0.03)
        _settle(session)

        assert _selector_member(selector) == target

    def test_the_handle_snaps_onto_the_angle_when_the_drag_settles(
            self, window, two_shells):
        """Mid-drag the pointer owns the handle, so it is only put back onto
        the angle it picked once the drag ends."""
        from anyplotlib.callbacks import Event

        session = window["window"]
        tree, _summed, plot, controller = _open_ring(session, two_shells)
        session._select_signal_node(plot, id(tree.root))
        _settle(session)

        target = 2
        x, y = controller.positions[target]
        controller._handle.set(x=float(x) + 0.04, y=float(y))
        _settle(session)
        assert controller._dragging is True
        assert not np.allclose(controller.drawn_highlight(),
                               controller.positions[target].reshape(1, 2))

        controller._handle.callbacks.fire(
            Event("pointer_up", source=controller._handle))
        _settle(session)

        assert controller._dragging is False
        assert np.allclose(controller.drawn_highlight(),
                           controller.positions[target].reshape(1, 2))

    def test_the_staged_pick_action_goes_through_the_same_path(self, window,
                                                               two_shells):
        session = window["window"]
        tree, _summed, plot, controller = _open_ring(session, two_shells)
        selector = _tree_angle_selector(tree)

        registry.resolve_staged("multiangle_pick_angle")(
            session, plot, {"member": 1, "window_id": controller.window_id})
        _settle(session)

        assert _selector_member(selector) == 1


# ── the selector drives the ring ─────────────────────────────────────────────

class TestTheSelectorDrivesTheRing:
    """A move from anywhere else must reach the highlight — that is the whole
    point of hanging off the selector's own update instead of the pick."""

    def test_an_external_selector_move_moves_the_highlight(self, window,
                                                           two_shells):
        session = window["window"]
        tree, _summed, plot, controller = _open_ring(session, two_shells)
        selector = _tree_angle_selector(tree)

        # Show the stack first: on a sum no single angle is the live one.
        session._select_signal_node(plot, id(tree.root))
        _settle(session)

        target = 4
        # Move the selector the way playback or a keyboard step does — nothing
        # here touches the ring window.
        scale, offset = ring._selector_calibration(selector)
        ring._selector_widget(selector).x = target * scale + offset
        selector.delayed_update_data(force=True)
        _settle(session)

        assert _wait(lambda: np.allclose(
            controller.drawn_highlight(),
            controller.positions[target].reshape(1, 2)), timeout=5.0), \
            f"the ring highlight never followed to member {target}"
        assert controller.current_member == target

    def test_an_integrating_angle_span_lights_every_angle_in_it(self, window,
                                                                two_shells):
        """Integrate on the angle axis sums several members, so several are in
        the frame — and the ring lights all of them. Its hook has to be on the
        span half of the selector too, or it stops following the moment
        Integrate goes on."""
        session = window["window"]
        tree, _summed, plot, controller = _open_ring(session, two_shells)
        selector = _tree_angle_selector(tree)

        session._select_signal_node(plot, id(tree.root))
        _settle(session)
        selector.set_integrating(True)
        _settle(session)

        summed_members = tuple(np.asarray(selector.current_indices).ravel())
        assert len(summed_members) > 1, "Integrate did not select a span"
        assert _wait(lambda: controller.current_members == summed_members,
                     timeout=5.0), "the ring never followed the integrating span"
        # Every angle in the span is lit; the handle marks where it starts.
        colors = controller.drawn_member_colors()
        assert [index for index, color in enumerate(colors)
                if color == ring.LIT_COLOR] == sorted(summed_members)
        assert np.allclose(controller.drawn_highlight(),
                           controller.positions[summed_members[0]].reshape(1, 2))
        n_members = ring.multiangle_model(tree).n_members
        assert controller.badge == (
            "Σ all angles" if len(summed_members) == n_members
            else f"Σ {len(summed_members)} angles")

    def test_the_ring_is_bound_to_the_selector_exactly_once(self, window,
                                                            two_shells):
        session = window["window"]
        tree, _summed, _plot, controller = _open_ring(session, two_shells)
        selector = _tree_angle_selector(tree)

        bound = _ring_hooks(selector, controller)
        assert bound and all(count == 1 for count in bound)
        # A second open raises the existing window rather than binding again.
        again = ring.open_multiangle_navigator(session, tree)
        assert again is controller
        assert _ring_hooks(selector, controller) == bound


# ── teardown ─────────────────────────────────────────────────────────────────

class TestTeardown:
    """Opening, closing and re-opening must leave exactly one of everything."""

    def test_the_window_registers_a_controller(self, window, two_shells):
        session = window["window"]
        tree, _summed, _plot, controller = _open_ring(session, two_shells)

        assert session.controller_by_window_id(controller.window_id) is controller
        assert getattr(tree, ring._WINDOW_ATTRIBUTE) is controller

    def test_close_leaves_nothing_registered(self, window, two_shells):
        session = window["window"]
        tree, _summed, _plot, controller = _open_ring(session, two_shells)
        selector = _tree_angle_selector(tree)
        window_id = controller.window_id

        session._forget_window(window_id)

        assert session.controller_by_window_id(window_id) is None
        assert getattr(tree, ring._WINDOW_ATTRIBUTE) is None
        assert all(count == 0 for count in _ring_hooks(selector, controller)), \
            "a closed ring must stop following the real selector"

    def test_a_closed_ring_ignores_a_later_selector_move(self, window,
                                                         two_shells):
        session = window["window"]
        tree, _summed, plot, controller = _open_ring(session, two_shells)
        selector = _tree_angle_selector(tree)
        session._select_signal_node(plot, id(tree.root))
        _settle(session)
        before = controller.current_members

        session._forget_window(controller.window_id)
        ring._selector_widget(selector).x = 4.0
        selector.delayed_update_data(force=True)
        _settle(session)

        assert _selector_member(selector) == 4, "the selector did not move"
        assert controller.current_members == before, \
            "a closed ring still followed the selector"
        assert controller.drawn_highlight().shape == (0, 2)

    def test_open_close_open_leaves_one_controller(self, window, two_shells):
        session = window["window"]
        tree, _summed, _plot, first = _open_ring(session, two_shells)
        selector = _tree_angle_selector(tree)

        session._forget_window(first.window_id)
        second = ring.open_multiangle_navigator(session, tree)

        assert second is not None and second is not first
        assert second.window_id != first.window_id
        live = [controller
                for controller in session._window_controllers.values()
                if isinstance(controller, ring.MultiAngleNavigatorController)]
        assert live == [second]
        assert all(count == 0 for count in _ring_hooks(selector, first))
        bound = _ring_hooks(selector, second)
        assert bound and all(count == 1 for count in bound)


# ── the opening seam ─────────────────────────────────────────────────────────

class TestOpening:
    def test_the_acquisition_opens_with_its_ring_already_up(self, window,
                                                            two_shells):
        """The ring is part of opening an acquisition, not something to go and
        find — the loader raises it alongside the data."""
        session = window["window"]
        tree, _summed, _plot = _open(session, two_shells)
        assert getattr(tree, ring._WINDOW_ATTRIBUTE) is not None

    def test_the_staged_action_reopens_a_closed_window(self, window,
                                                       two_shells):
        session = window["window"]
        tree, _summed, plot = _open(session, two_shells)

        # Close the one the loader raised, so what is asserted below is the
        # action DRAWING a window rather than an already-open one lingering.
        session._forget_window(
            getattr(tree, ring._WINDOW_ATTRIBUTE).window_id)
        assert getattr(tree, ring._WINDOW_ATTRIBUTE, None) is None
        messages = window["messages"]
        messages.clear()

        registry.resolve_staged("multiangle_show_angles")(session, plot, {})

        controller = getattr(tree, ring._WINDOW_ATTRIBUTE)
        assert controller is not None
        figures = [m for m in messages if m.get("type") == "figure"
                   and m.get("window_id") == controller.window_id]
        assert figures, "the angle navigator emitted no figure"
        assert "Angles" in figures[-1]["title"]

    def test_an_ordinary_tree_gets_no_ring(self, window, stem_4d_dataset):
        """A 4-D STEM scan is not a multi-angle acquisition — refuse, loudly,
        rather than open a window with nothing in it."""
        session = stem_4d_dataset["window"]
        tree = session.signal_trees[0]

        assert ring.multiangle_model(tree) is None
        assert ring.open_multiangle_navigator(session, tree) is None


class TestTheRingOnAReopenedAcquisition:
    """A reopened tree has no recipes. The ring asked only the recipe whether
    one angle was on screen, so on every reopened dataset it stayed on "all
    angles": every member lit, no handle, a node switch on every pick."""

    def _saved(self, tmp_path):
        from spyde.signals.multiangle import MULTIANGLE_METADATA

        data = np.random.default_rng(3).integers(
            0, 400, (4, 5, 6, 4, 4), dtype=np.uint16)
        signal = hs.signals.Signal2D(data)
        signal.metadata.set_item(MULTIANGLE_METADATA, {
            "n_members": 4, "n_shells": 1, "tilts": [1.0] * 4,
            "azimuths": [0.0, 90.0, 180.0, 270.0], "shell_ids": [0] * 4,
            "reference": 0})
        signal.set_signal_type("electron_diffraction")
        path = tmp_path / "acquisition.zspy"
        signal.save(str(path))
        return path

    def test_a_pick_shows_one_angle_and_says_so(self, tmp_path):
        import time

        from spyde.tests.migrated.conftest import close_session, make_session

        session = make_session()
        try:
            session.open_file(str(self._saved(tmp_path)))
            deadline = time.time() + 60.0
            while time.time() < deadline and not session.signal_trees:
                time.sleep(0.2)
            tree = session.signal_trees[0]
            time.sleep(1.0)
            controller = ring.open_multiangle_navigator(session, tree)
            assert controller is not None
            assert controller.showing_one_angle() is False
            controller.select_member(2)
            time.sleep(0.5)
            assert controller.showing_one_angle() is True, \
                "the stack is on screen and the ring does not know"
            assert controller.live_members() == (2,)
            assert len(controller.drawn_highlight()) > 0, "no handle drawn"
            displayed = controller.displayed_signal()
            assert displayed is tree.root_node.signal
        finally:
            close_session(session)

    def test_a_pick_stays_on_a_binned_stack(self, tmp_path):
        import time

        from spyde.actions.base import Rebin2DAction
        from spyde.actions.context import ActionContext
        from spyde.actions.lifecycle import show_tree_node
        from spyde.tests.migrated.conftest import close_session, make_session

        session = make_session()
        try:
            session.open_file(str(self._saved(tmp_path)))
            deadline = time.time() + 60.0
            while time.time() < deadline and not session.signal_trees:
                time.sleep(0.2)
            tree = session.signal_trees[0]
            time.sleep(1.0)
            plot = next(p for p in tree.signal_plots)
            show_tree_node(plot, tree, tree.root_node.signal)
            params = {"scale_x": 2, "scale_y": 2, "scan_x": 1, "scan_y": 1}
            binned = Rebin2DAction(ActionContext(
                plot=plot, params=params, action_name="Rebin")).run(**params)
            assert binned.data.shape == (4, 5, 6, 2, 2)
            controller = ring.open_multiangle_navigator(session, tree)
            controller.select_member(1)
            time.sleep(0.5)
            assert controller.displayed_signal() is binned, \
                "the pick threw the user back onto the full-resolution root"
        finally:
            close_session(session)

