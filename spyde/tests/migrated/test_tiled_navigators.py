"""
Tiled 2-D navigators: every panel's crosshair is a copy of a navigation
selector's crosshair, and that selector's own crosshair on the live navigator
is the one master holding the position (GitHub #171).

⇧-clicking two navigator chips of a 4D-STEM tree tiles the navigator images
side by side. These tests build a real Session, tile two navigators the way the
chip strip does, and drive the tiled crosshairs with the same event messages the
renderer sends, asserting that:

  (a) the tiled crosshairs start at the master's position, not the image centre;
  (b) moving the master moves every tiled copy;
  (c) dragging one copy moves the master and the other copies, and nothing
      snaps back once the navigation update has run;
  (d) adding or removing a selector while tiled rebuilds the figure with one
      crosshair per selector on every panel, each in its selector's colour.
"""
from __future__ import annotations

import json
import time

import numpy as np
import anyplotlib._electron as electron

from spyde.actions import navigator_views
from spyde.tests.migrated.conftest import _settle


# ── helpers ──────────────────────────────────────────────────────────────────

def _navigator_plot(session):
    return next(p for p in session._plots if getattr(p, "is_navigator", False))


def _manager(session):
    return session.signal_trees[0].navigator_plot_manager


def _selectors(session):
    plot = _navigator_plot(session)
    return list(_manager(session).navigation_selectors[plot.plot_window])


def _master(selector):
    return selector._crosshair_selector._widget


def _tile(session, messages):
    """Register a second navigator image and ⇧-click both chips; returns the
    tiled figure the renderer would show."""
    tree = session.signal_trees[0]
    base_name = next(iter(tree.navigator_signals))
    base = tree.navigator_signals[base_name]
    base = base[0] if isinstance(base, (list, tuple)) else base
    doubled = base.deepcopy()
    doubled.data = np.asarray(doubled.data) * 2
    tree.add_navigator_signal("doubled", doubled)
    plot = _navigator_plot(session)
    messages.clear()
    navigator_views.select_navigator(
        session, plot, {"names": [base_name, "doubled"], "window_id": plot.window_id})
    return _last_tiled_figure(messages)


def _last_tiled_figure(messages):
    tiled = [m for m in messages
             if m.get("type") == "figure" and m.get("view_kind") == "tiled"]
    assert tiled, "no tiled navigator figure emitted"
    return tiled[-1]["fig_id"], electron._figures[tiled[-1]["fig_id"]]


def _panels(figure):
    """[(panel_id, [crosshair widgets on that panel])] in panel order."""
    return [
        (panel_id, [w for w in plot._widgets.values() if w._type == "crosshair"])
        for panel_id, plot in figure._plots_map.items()
    ]


def _position(widget):
    return float(widget.get("cx")), float(widget.get("cy"))


def _drag(fig_id, panel_id, widget, cx, cy):
    """Send the renderer's drag messages for one tiled crosshair."""
    for event_type in ("pointer_move", "pointer_up"):
        electron.dispatch_event(fig_id, json.dumps({
            "event_type": event_type, "panel_id": panel_id,
            "widget_id": widget._id, "cx": cx, "cy": cy,
        }))


def _wait_for(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


# ── (a) seeding ──────────────────────────────────────────────────────────────

class TestTiledSeeding:
    def test_tiled_crosshairs_start_at_the_master_position(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        (selector,) = _selectors(session)
        _master(selector).set(cx=3.0, cy=1.0)
        _settle(session)

        _fig_id, figure = _tile(session, stem_4d_dataset["messages"])
        panels = _panels(figure)
        assert len(panels) == 2
        for _panel_id, crosshairs in panels:
            assert len(crosshairs) == 1
            assert _position(crosshairs[0]) == (3.0, 1.0)


# ── (b) master → copies ──────────────────────────────────────────────────────

class TestMasterDrivesCopies:
    def test_moving_the_master_moves_every_tiled_crosshair(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        (selector,) = _selectors(session)
        _fig_id, figure = _tile(session, stem_4d_dataset["messages"])

        _master(selector).set(cx=1.0, cy=3.0)   # a drag on the live navigator
        _settle(session)

        for _panel_id, (crosshair,) in _panels(figure):
            assert _wait_for(lambda: _position(crosshair) == (1.0, 3.0)), \
                f"tiled crosshair stayed at {_position(crosshair)}"


# ── (c) copy → master → other copies ─────────────────────────────────────────

class TestCopyForwardsToMaster:
    def test_dragging_one_tile_moves_the_master_and_the_other_tiles(
            self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        (selector,) = _selectors(session)
        fig_id, figure = _tile(session, stem_4d_dataset["messages"])
        (first_id, (first,)), (second_id, (second,)) = _panels(figure)

        _drag(fig_id, second_id, second, 2.0, 1.0)
        assert _position(_master(selector)) == (2.0, 1.0)
        assert _position(first) == (2.0, 1.0)

        # Once the navigation update has run, nothing snaps back.
        _settle(session)
        assert _wait_for(lambda: np.array_equal(selector.current_indices, [[2, 1]]))
        time.sleep(0.2)
        for crosshair in (first, second):
            assert _position(crosshair) == (2.0, 1.0)
        assert _position(_master(selector)) == (2.0, 1.0)


# ── (d) adding / removing a selector while tiled ─────────────────────────────

class TestSelectorsWhileTiled:
    def test_adding_a_selector_adds_its_crosshair_to_every_panel(
            self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        messages = stem_4d_dataset["messages"]
        _tile(session, messages)
        plot = _navigator_plot(session)

        messages.clear()
        _manager(session).add_navigation_selector_and_signal_plot(plot.plot_window)
        _settle(session)

        selectors = _selectors(session)
        assert len(selectors) == 2
        fig_id, figure = _last_tiled_figure(messages)
        colours = sorted(str(s.color) for s in selectors)
        for _panel_id, crosshairs in _panels(figure):
            assert len(crosshairs) == 2
            assert sorted(str(w.get("color")) for w in crosshairs) == colours

        # Each copy follows its own selector.
        second = selectors[1]
        panel_id, crosshairs = _panels(figure)[0]
        copy = next(w for w in crosshairs if w.get("color") == second.color)
        _drag(fig_id, panel_id, copy, 0.0, 2.0)
        assert _position(_master(second)) == (0.0, 2.0)
        assert _position(_master(selectors[0])) != (0.0, 2.0)

    def test_removing_a_selector_drops_its_crosshair(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        messages = stem_4d_dataset["messages"]
        plot = _navigator_plot(session)
        _manager(session).add_navigation_selector_and_signal_plot(plot.plot_window)
        _settle(session)
        _tile(session, messages)

        closed = _selectors(session)[1]
        signal_window = next(child for child in closed.children).window_id
        messages.clear()
        session.dispatch_action({"action": "close_window", "window_id": signal_window})

        _fig_id, figure = _last_tiled_figure(messages)
        for _panel_id, crosshairs in _panels(figure):
            assert len(crosshairs) == 1

    def test_single_chip_stops_rebuilding_the_tiles(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        messages = stem_4d_dataset["messages"]
        _tile(session, messages)
        plot = _navigator_plot(session)
        tree = session.signal_trees[0]
        navigator_views.select_navigator(
            session, plot, {"names": [next(iter(tree.navigator_signals))],
                            "window_id": plot.window_id})

        messages.clear()
        _manager(session).add_navigation_selector_and_signal_plot(plot.plot_window)
        _settle(session)
        assert not [m for m in messages if m.get("view_kind") == "tiled"]


# ── (e) integrate mode: the tiles follow the integrating rectangle ───────────

def _rectangle(selector):
    return selector._rect_selector._widget


def _rectangle_centre(selector):
    rectangle = _rectangle(selector)
    return (float(rectangle.get("x")) + float(rectangle.get("w")) / 2,
            float(rectangle.get("y")) + float(rectangle.get("h")) / 2)


def _hook_counts(selector, hook):
    """How many copies of *hook* sit on each half of the composite selector."""
    return [sum(h is hook for h in half.index_hooks)
            for half in (selector._crosshair_selector, selector._rect_selector)]


class TestTiledIntegrate:
    def _assert_tiles_on_rectangle(self, figure, selector):
        centre = _rectangle_centre(selector)
        for _panel_id, (crosshair,) in _panels(figure):
            assert _wait_for(lambda: _position(crosshair) == centre), \
                f"tiled crosshair at {_position(crosshair)}, rectangle centre {centre}"

    def test_built_while_integrating_tiles_follow_the_rectangle(
            self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        (selector,) = _selectors(session)
        selector.set_integrating(True)
        _settle(session)
        _fig_id, figure = _tile(session, stem_4d_dataset["messages"])
        self._assert_tiles_on_rectangle(figure, selector)

        _rectangle(selector).set(x=1.0, y=2.0)   # a drag on the live navigator
        _settle(session)
        self._assert_tiles_on_rectangle(figure, selector)

    def test_toggling_integrate_after_tiling_keeps_tiles_following(
            self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        (selector,) = _selectors(session)
        _fig_id, figure = _tile(session, stem_4d_dataset["messages"])

        selector.set_integrating(True)
        _settle(session)
        _rectangle(selector).set(x=2.0, y=1.0)
        _settle(session)
        self._assert_tiles_on_rectangle(figure, selector)

    def test_dragging_a_tile_centres_the_rectangle_there(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        (selector,) = _selectors(session)
        selector.set_integrating(True)
        _settle(session)
        fig_id, figure = _tile(session, stem_4d_dataset["messages"])
        size = (float(_rectangle(selector).get("w")), float(_rectangle(selector).get("h")))
        (first_id, (first,)), (second_id, (second,)) = _panels(figure)

        _drag(fig_id, second_id, second, 3.5, 3.0)
        _settle(session)

        assert _rectangle_centre(selector) == (3.5, 3.0)
        assert (float(_rectangle(selector).get("w")),
                float(_rectangle(selector).get("h"))) == size
        assert _position(first) == (3.5, 3.0)

    def test_hooks_both_halves_and_close_removes_them_after_a_toggle(
            self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        (selector,) = _selectors(session)
        _tile(session, stem_4d_dataset["messages"])
        plot = _navigator_plot(session)
        (cursor,) = session._nav_view_cursors[plot.window_id]
        hook = cursor._index_hook
        assert _hook_counts(selector, hook) == [1, 1]

        selector.set_integrating(True)      # the composite now forwards to the rectangle
        _settle(session)
        tree = session.signal_trees[0]
        navigator_views.select_navigator(
            session, plot, {"names": [next(iter(tree.navigator_signals))],
                            "window_id": plot.window_id})

        assert _hook_counts(selector, hook) == [0, 0]

    def test_dragging_a_tile_to_the_edge_keeps_the_rectangle_inside(
            self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        (selector,) = _selectors(session)
        selector.set_integrating(True)
        _settle(session)
        fig_id, figure = _tile(session, stem_4d_dataset["messages"])
        (panel_id, (crosshair,)), _second = _panels(figure)

        _drag(fig_id, panel_id, crosshair, 100.0, -100.0)
        _settle(session)

        rectangle = _rectangle(selector)
        state = rectangle._plot._state
        assert float(rectangle.get("x")) + float(rectangle.get("w")) <= state["image_width"]
        assert float(rectangle.get("y")) == 0.0
        # The copies settle on where the region actually is.
        assert _wait_for(lambda: _position(crosshair) == _rectangle_centre(selector))
