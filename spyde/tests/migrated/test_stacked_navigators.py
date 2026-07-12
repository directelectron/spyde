"""
Stacked 1-D navigators with a shared, linked time cursor (in-situ movies).

A movie tree's navigation space is 1-D (time). ⇧-clicking two or more navigator
chips STACKS the selected 1-D traces as rows in ONE anyplotlib figure with a
shared x (time) axis and a single logical time cursor — one draggable vertical
line per row, all kept in sync, wired to the tree's REAL 1-D navigation selector.

These tests build a real Session via the ``movie_dataset`` fixture, register a
second 1-D navigator on the tree, drive ``select_navigator`` the way the chip
strip does, and assert on the emitted figure + the shared-cursor behaviour:

  (a) ⇧-click 2 chips → a stacked figure + one VLine per row + a cursor object;
  (b) a line drag on row 2 → the real navigation selector's index moves and the
      signal plot repaints (normal nav cascade);
  (c) a PROGRAMMATIC selector move (translate_pixels + delayed_update_data) →
      every row's line syncs to the new frame (the sync hangs off the selector's
      own update path, so playback moves the lines too);
  (d) teardown: switching back to a single navigator / closing the window
      removes the cursor and detaches its index hook from the real selector.
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs

from spyde.actions import navigator_views as nv


# ── helpers ──────────────────────────────────────────────────────────────────

def _nav_window_id(session):
    """The movie tree's navigator plot window id (the one carrying the real
    1-D navigation selector)."""
    tree = session.signal_trees[0]
    mgr = tree.navigator_plot_manager
    pw = list(mgr.plot_windows.keys())[0]
    return pw.window_id


def _add_second_navigator(session, name="peak", n=8):
    """Register a second 1-D navigator trace of the same nav length on the movie
    tree (a plain Signal1D — max/other summary of the movie)."""
    tree = session.signal_trees[0]
    trace = hs.signals.Signal1D((np.arange(n) ** 2).astype(np.float32))
    tree.add_navigator_signal(name, trace)
    return tree


def _real_selector(session):
    wid = _nav_window_id(session)
    return session._nav_selectors.get(wid)


def _settle(session, sel):
    """Let the serial nav dispatcher run the queued update, then any inline
    main-thread sync (in tests _dispatch_to_main runs inline)."""
    # The dispatcher is a background thread; give it a moment to run _run_update
    # (which fires the index hooks that sync the lines).
    for _ in range(40):
        time.sleep(0.02)
        if sel.current_indices is not None:
            break
    time.sleep(0.1)


# ── (a) building the stacked view ────────────────────────────────────────────

class TestStackedBuild:
    def test_movie_navigator_is_1d(self, movie_dataset):
        session = movie_dataset["window"]
        tree = session.signal_trees[0]
        assert tree.nav_dim == 1
        assert nv._tree_nav_is_1d(tree) is True

    def test_shift_click_two_1d_navigators_builds_stacked_figure(self, movie_dataset):
        session = movie_dataset["window"]
        msgs = movie_dataset["messages"]
        _add_second_navigator(session, "peak")
        wid = _nav_window_id(session)
        plot = session._plot_by_window_id(wid)

        msgs.clear()
        nv.select_navigator(session, plot, {"names": ["base", "peak"], "window_id": wid})

        # A stacked figure is emitted on the navigator window, tagged stacked.
        figs = [m for m in msgs if m.get("type") == "figure"
                and m.get("view_kind") == "stacked"]
        assert figs, "no stacked navigator figure emitted"
        fig = figs[-1]
        assert fig["window_id"] == wid
        assert fig["view_label"] == nv.STACKED_LABEL
        assert fig["is_navigator"] is True

        # The cursor object is registered for this navigator window.
        cursor = session._stacked_nav_cursors.get(wid)
        assert cursor is not None
        # One VLine widget per stacked row (2 navigators → 2 rows/lines).
        assert len(cursor.widgets) == 2

    def test_stacked_cursor_starts_at_current_selector_index(self, movie_dataset):
        session = movie_dataset["window"]
        _add_second_navigator(session, "peak")
        wid = _nav_window_id(session)
        plot = session._plot_by_window_id(wid)
        sel = _real_selector(session)

        # Move the real selector to frame 3 first.
        sel._widget.x = 3 * 0.1     # scale is 0.1 (time axis)
        sel.delayed_update_data(force=True)
        _settle(session, sel)

        nv.select_navigator(session, plot, {"names": ["base", "peak"], "window_id": wid})
        cursor = session._stacked_nav_cursors[wid]
        # Each row's line starts on the live frame (x = index * scale).
        for w in cursor.widgets:
            assert abs(float(w.get("x")) - 3 * 0.1) < 1e-6

    def test_single_chip_does_not_build_stacked(self, movie_dataset):
        session = movie_dataset["window"]
        msgs = movie_dataset["messages"]
        _add_second_navigator(session, "peak")
        wid = _nav_window_id(session)
        plot = session._plot_by_window_id(wid)

        msgs.clear()
        nv.select_navigator(session, plot, {"names": ["base"], "window_id": wid})
        stacked = [m for m in msgs if m.get("type") == "figure"
                   and m.get("view_kind") == "stacked"]
        assert not stacked
        assert session._stacked_nav_cursors.get(wid) is None


# ── (b) dragging a row's line drives the real selector ───────────────────────

class TestStackedDrag:
    def test_drag_row2_line_moves_real_selector_index(self, movie_dataset):
        session = movie_dataset["window"]
        _add_second_navigator(session, "peak")
        wid = _nav_window_id(session)
        plot = session._plot_by_window_id(wid)
        sel = _real_selector(session)

        nv.select_navigator(session, plot, {"names": ["base", "peak"], "window_id": wid})
        cursor = session._stacked_nav_cursors[wid]
        assert len(cursor.widgets) == 2

        # Simulate a drag on ROW 2's line: set its x to frame 5 (x = 5 * 0.1),
        # which fires pointer_move → the stacked-cursor drag handler.
        row2 = cursor.widgets[1]
        row2.set(x=5 * 0.1)
        _settle(session, sel)

        # The real 1-D navigation selector committed frame 5 → the DP repaints.
        assert sel.current_indices is not None
        assert int(np.asarray(sel.current_indices).ravel()[0]) == 5

        # And the real selector's own VLine widget followed the drag.
        real_w = nv._selector_vline_widget(sel)
        assert abs(float(real_w.x) - 5 * 0.1) < 1e-6

    def test_drag_mirrors_to_the_other_row(self, movie_dataset):
        session = movie_dataset["window"]
        _add_second_navigator(session, "peak")
        wid = _nav_window_id(session)
        plot = session._plot_by_window_id(wid)
        sel = _real_selector(session)

        nv.select_navigator(session, plot, {"names": ["base", "peak"], "window_id": wid})
        cursor = session._stacked_nav_cursors[wid]

        row1, row2 = cursor.widgets
        row2.set(x=4 * 0.1)      # drag row 2
        _settle(session, sel)
        # Row 1's line mirrors the same x (one logical cursor).
        assert abs(float(row1.get("x")) - 4 * 0.1) < 1e-6


# ── (c) a programmatic selector move syncs all rows ──────────────────────────

class TestStackedProgrammaticSync:
    def test_programmatic_step_syncs_every_row_line(self, movie_dataset):
        session = movie_dataset["window"]
        _add_second_navigator(session, "peak")
        wid = _nav_window_id(session)
        plot = session._plot_by_window_id(wid)
        sel = _real_selector(session)

        nv.select_navigator(session, plot, {"names": ["base", "peak"], "window_id": wid})
        cursor = session._stacked_nav_cursors[wid]

        # This is exactly what the playback clock does: translate the real
        # selector's pixel position then request an update. The stacked cursor is
        # NOT involved — the sync must come purely from the selector's own update
        # path (index_hook).
        start = int(np.asarray(sel.get_selected_indices()).ravel()[0])
        sel.translate_pixels(2)
        sel.delayed_update_data(force=True)
        _settle(session, sel)

        target = start + 2
        assert int(np.asarray(sel.current_indices).ravel()[0]) == target
        # Every row's line snapped to the new frame's x.
        for w in cursor.widgets:
            assert abs(float(w.get("x")) - target * 0.1) < 1e-6, \
                f"row line at {w.get('x')} did not sync to frame {target}"

    def test_index_hook_is_installed_on_the_real_selector(self, movie_dataset):
        session = movie_dataset["window"]
        _add_second_navigator(session, "peak")
        wid = _nav_window_id(session)
        plot = session._plot_by_window_id(wid)
        sel = _real_selector(session)

        nv.select_navigator(session, plot, {"names": ["base", "peak"], "window_id": wid})
        cursor = session._stacked_nav_cursors[wid]
        assert cursor._index_hook in sel.index_hooks


# ── (d) teardown ─────────────────────────────────────────────────────────────

class TestStackedTeardown:
    def test_switching_back_to_single_navigator_tears_down(self, movie_dataset):
        session = movie_dataset["window"]
        _add_second_navigator(session, "peak")
        wid = _nav_window_id(session)
        plot = session._plot_by_window_id(wid)
        sel = _real_selector(session)

        nv.select_navigator(session, plot, {"names": ["base", "peak"], "window_id": wid})
        cursor = session._stacked_nav_cursors[wid]
        hook = cursor._index_hook
        assert hook in sel.index_hooks

        # Click a single chip → back to one navigator; the cursor is gone and its
        # index hook is detached from the real selector.
        nv.select_navigator(session, plot, {"names": ["base"], "window_id": wid})
        assert session._stacked_nav_cursors.get(wid) is None
        assert hook not in sel.index_hooks
        assert cursor._closed is True

    def test_closing_the_window_detaches_the_hook(self, movie_dataset):
        session = movie_dataset["window"]
        _add_second_navigator(session, "peak")
        wid = _nav_window_id(session)
        plot = session._plot_by_window_id(wid)
        sel = _real_selector(session)

        nv.select_navigator(session, plot, {"names": ["base", "peak"], "window_id": wid})
        cursor = session._stacked_nav_cursors[wid]
        hook = cursor._index_hook
        assert hook in sel.index_hooks

        session._forget_window(wid)
        assert session._stacked_nav_cursors.get(wid) is None
        assert hook not in sel.index_hooks

    def test_rebuild_replaces_the_prior_cursor(self, movie_dataset):
        session = movie_dataset["window"]
        _add_second_navigator(session, "peak")
        _add_second_navigator(session, "peak2")  # a third 1-D navigator
        wid = _nav_window_id(session)
        plot = session._plot_by_window_id(wid)
        sel = _real_selector(session)

        nv.select_navigator(session, plot, {"names": ["base", "peak"], "window_id": wid})
        first = session._stacked_nav_cursors[wid]
        first_hook = first._index_hook

        # Re-⇧-click a different set → the old cursor is torn down, a new one built
        # (no leaked/duplicated index hooks on the real selector).
        nv.select_navigator(session, plot,
                            {"names": ["base", "peak", "peak2"], "window_id": wid})
        second = session._stacked_nav_cursors[wid]
        assert second is not first
        assert first._closed is True
        assert first_hook not in sel.index_hooks
        assert second._index_hook in sel.index_hooks
        # Exactly one stacked-cursor hook is present.
        stacked_hooks = [h for h in sel.index_hooks
                         if h in (first_hook, second._index_hook)]
        assert stacked_hooks == [second._index_hook]
        assert len(second.widgets) == 3
