"""
Closing a SIGNAL plot window (not the navigator) must deregister the
navigator selector that drove it — otherwise the Plot Control dock keeps
showing a selector row for a window that's gone, and the closed (but never
removed) selector object lingers in ``MultiplotManager.navigation_selectors``
forever.

Companion to test_close.py (window-close SCOPING); this covers the selector
bookkeeping that scoping alone doesn't touch.
"""
from __future__ import annotations


def _nav_wid(session):
    for p in session._plots:
        if getattr(p, "is_navigator", False) and p.window_id is not None:
            return p.window_id
    return None


def _signal_wids(session):
    return sorted({
        p.window_id for p in session._plots
        if not getattr(p, "is_navigator", False) and p.window_id is not None
    })


def _nav_plot(session):
    for p in session._plots:
        if getattr(p, "is_navigator", False):
            return p
    return None


def _mm(session):
    """The MultiplotManager for the (only) open tree."""
    tree = session.signal_trees[0]
    return tree.navigator_plot_manager


class TestSelectorCloseCleanup:
    def test_closing_signal_window_removes_selector_from_multiplot_manager(
        self, stem_4d_dataset,
    ):
        session = stem_4d_dataset["window"]
        mm = _mm(session)
        nav = _nav_wid(session)
        sig_wids = _signal_wids(session)
        assert nav is not None and sig_wids

        before = list(mm.all_navigation_selectors)
        assert before, "expected at least one navigator selector"

        target = sig_wids[0]
        session.dispatch_action({"action": "close_window", "window_id": target})

        after = mm.all_navigation_selectors
        assert len(after) == len(before) - 1, (
            "the selector driving the closed signal window must be removed "
            "from navigation_selectors"
        )

    def test_closing_signal_window_emits_selector_removed(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        msgs = stem_4d_dataset["messages"]
        sig_wids = _signal_wids(session)
        target = sig_wids[0]

        # Capture the selector_id the dock would have shown for this window's
        # driving selector before we close it.
        infos = [m for m in msgs if m.get("type") == "selector_info"]
        assert infos, "expected selector_info to have been emitted on creation"

        msgs.clear()
        session.dispatch_action({"action": "close_window", "window_id": target})

        removed = [m for m in msgs if m.get("type") == "selector_removed"]
        assert removed, "closing a signal window must emit selector_removed"
        assert isinstance(removed[0].get("selector_id"), int)

    def test_closed_selector_dropped_from_session_lookup_tables(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        sig_wids = _signal_wids(session)
        target = sig_wids[0]

        # Find the selector object driving `target` BEFORE closing it.
        plot = next(p for p in session._plots if p.window_id == target)
        pw = plot.plot_window
        sel = pw.parent_selector
        assert sel is not None
        sid = id(sel)
        assert sid in getattr(session, "_nav_selectors_by_id", {})

        session.dispatch_action({"action": "close_window", "window_id": target})

        assert sid not in getattr(session, "_nav_selectors_by_id", {}), (
            "closed selector must be dropped from _nav_selectors_by_id"
        )
        assert sel not in getattr(session, "_nav_selectors", {}).values()

    def test_navigator_and_other_signal_window_survive(self, stem_4d_dataset):
        """Closing ONE signal window must not disturb the navigator's OWN
        (still-live) selector bookkeeping — only the closed window's selector
        is pruned."""
        session = stem_4d_dataset["window"]
        mm = _mm(session)
        nav = _nav_wid(session)
        sig_wids = _signal_wids(session)
        assert nav is not None and sig_wids

        target = sig_wids[0]
        session.dispatch_action({"action": "close_window", "window_id": target})

        # The navigator window itself is untouched.
        assert nav in {
            p.window_id for p in session._plots if p.window_id is not None
        }
        # The tree is still open (navigator survives a signal-window close).
        assert session.signal_trees
        # Whatever selectors remain are still tracked consistently (no stale
        # dict entries pointing at the closed window).
        for sel_list in mm.navigation_selectors.values():
            for sel in sel_list:
                assert id(sel) in session._nav_selectors_by_id

    def test_queued_update_for_closed_selector_does_not_raise(self, stem_4d_dataset):
        """A stale/queued navigator update against an already-closed selector
        must not crash the dispatcher (mirrors the _NavDispatcher's own
        try/except contract — this asserts closing + a subsequent forced
        update on the SAME selector object is a safe no-crash no-op)."""
        session = stem_4d_dataset["window"]
        sig_wids = _signal_wids(session)
        target = sig_wids[0]

        plot = next(p for p in session._plots if p.window_id == target)
        sel = plot.plot_window.parent_selector
        assert sel is not None

        session.dispatch_action({"action": "close_window", "window_id": target})

        # Directly invoke the update body (bypassing the dispatcher thread) —
        # this must not raise even though the selector was just closed and
        # deregistered.
        sel._run_update(force=True)
