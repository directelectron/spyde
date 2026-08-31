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

import time

import numpy as np
import hyperspy.api as hs
from spyde.tests.migrated.conftest import _settle, make_session


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

    def test_second_selector_survives_closing_first_signal_window(
        self, stem_4d_dataset,
    ):
        """With TWO selectors on one navigator (what the add_selector toolbar
        action creates), closing ONE signal window must prune ONLY its own
        selector — the sibling stays registered everywhere and its dock row
        is not re-emitted or removed. (The single-selector survive test above
        can't see this: after the only signal window closes, the consistency
        loop runs over an empty list.)"""
        session = stem_4d_dataset["window"]
        msgs = stem_4d_dataset["messages"]
        mm = _mm(session)
        nav = _nav_plot(session)
        navpw = nav.plot_window

        # Second signal window + selector off the SAME navigator (the
        # add_selector toolbar action's body — see spyde.actions.base).
        mm.add_navigation_selector_and_signal_plot(navpw)
        _settle(session)

        sels = list(mm.navigation_selectors[navpw])
        assert len(sels) == 2, "expected two selectors on the navigator"
        sig_wids = _signal_wids(session)
        assert len(sig_wids) == 2, "expected two signal windows"

        target = sig_wids[0]
        target_plot = next(p for p in session._plots if p.window_id == target)
        closed_sel = target_plot.plot_window.parent_selector
        assert closed_sel is not None
        survivor = next(s for s in sels if s is not closed_sel)
        assert id(survivor) in session._nav_selectors_by_id

        msgs.clear()
        session.dispatch_action({"action": "close_window", "window_id": target})

        # The closed window's selector is gone...
        assert closed_sel not in mm.all_navigation_selectors
        removed_ids = [m.get("selector_id") for m in msgs
                       if m.get("type") == "selector_removed"]
        assert id(closed_sel) in removed_ids
        # ...the SIBLING selector survives, fully registered...
        assert survivor in mm.all_navigation_selectors
        assert id(survivor) in session._nav_selectors_by_id
        # ...with NO selector_removed for it and NO selector_info re-emit
        # that would rewrite its dock row.
        assert id(survivor) not in removed_ids
        assert not any(
            m.get("type") == "selector_info"
            and m.get("selector_id") == id(survivor)
            for m in msgs
        ), "the survivor's dock row payload must be untouched"
        # The second signal window itself stays open.
        assert sig_wids[1] in _signal_wids(session)

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


def _make_5d_session():
    """5-D stack (time → spatial → DP): a 2-level navigator chain whose DP is
    driven by a COMPOSITE (IntegratingSSelector2D). Mirrors test_nav_chain_5d."""
    s = hs.signals.Signal2D(
        np.random.RandomState(0).rand(2, 4, 5, 8, 8).astype(np.float32))
    s.set_signal_type("electron_diffraction")
    sess = make_session()
    sess._add_signal(s, source_path=None)
    _settle(sess)
    return sess


class TestComposite5DClose:
    """Closing the DP of a 5-D chained-navigator tree must deregister the
    COMPOSITE spatial selector (the DP window's parent_selector points at the
    composite, not an inner sub-selector — the nav-chain fix) and tear down
    BOTH inner sub-selectors' settle timers."""

    def test_closing_dp_removes_composite_and_cancels_inner_settles(
        self, captured_messages,
    ):
        sess = _make_5d_session()
        try:
            npm = sess.signal_trees[0].navigator_plot_manager
            sels = {type(x).__name__: x for x in npm.all_navigation_selectors}
            comp = sels["IntegratingSSelector2D"]       # spatial → drives the DP
            time_comp = sels["IntegratingSelector1D"]   # time → drives spatial nav

            dp = next(p for p in sess._plots
                      if not getattr(p, "is_navigator", False)
                      and p.window_id is not None)
            assert dp.plot_window.parent_selector is comp

            # Arm both inner settle timers so close() has live timers to
            # cancel (the construction-time ones fired long ago).
            comp._crosshair_selector._arm_settle()
            comp._rect_selector._arm_settle()
            t_ch = comp._crosshair_selector._settle_timer
            t_rect = comp._rect_selector._settle_timer
            assert t_ch is not None and t_rect is not None

            captured_messages.clear()
            sess.dispatch_action(
                {"action": "close_window", "window_id": dp.window_id})

            # The COMPOSITE is deregistered everywhere...
            assert comp not in npm.all_navigation_selectors
            assert id(comp) not in sess._nav_selectors_by_id
            removed_ids = [m.get("selector_id") for m in captured_messages
                           if m.get("type") == "selector_removed"]
            assert id(comp) in removed_ids, (
                "selector_removed must carry the COMPOSITE's id"
            )
            # ...while the upstream TIME composite (driving the still-open
            # spatial navigator) survives untouched.
            assert time_comp in npm.all_navigation_selectors
            assert id(time_comp) in sess._nav_selectors_by_id
            assert id(time_comp) not in removed_ids

            # Both inner sub-selectors' settle timers are cancelled + cleared
            # (composite close() → each inner BaseSelector.close()).
            assert comp._crosshair_selector._settle_timer is None
            assert comp._rect_selector._settle_timer is None
            assert t_ch.finished.is_set(), "crosshair settle timer must be cancelled"
            assert t_rect.finished.is_set(), "rect settle timer must be cancelled"
        finally:
            sess.shutdown()
