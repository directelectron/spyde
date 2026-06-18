"""
Closing a subwindow must be correctly SCOPED:
  * the NAVIGATOR's X closes the whole tree (all signals share its dataset),
  * a signal window's X closes ONLY that signal (and its selectors), and
  * a lone signal window (no navigator) closes its tree once nothing's left.

The renderer removes a window on the singular `window_closed`/`window_id`
(the backend used to emit an unhandled `windows_closed`/`tree_id` → close button
did nothing).
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


def _all_wids(session):
    return sorted({p.window_id for p in session._plots if p.window_id is not None})


def _closed(msgs):
    return sorted(m["window_id"] for m in msgs if m.get("type") == "window_closed")


class TestCloseWindow:
    def test_navigator_close_closes_the_whole_tree(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        msgs = stem_4d_dataset["messages"]
        nav = _nav_wid(session)
        assert nav is not None
        all_wids = _all_wids(session)

        msgs.clear()
        session.dispatch_action({"action": "close_window", "window_id": nav})

        assert _closed(msgs) == all_wids, "navigator X must close every window"
        assert session.signal_trees == []
        assert session._plots == []
        assert not any(m.get("type") == "windows_closed" for m in msgs)

    def test_signal_close_closes_only_that_window(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        msgs = stem_4d_dataset["messages"]
        nav = _nav_wid(session)
        sig_wids = _signal_wids(session)
        assert nav is not None and sig_wids, "expected a navigator + a signal window"
        target = sig_wids[0]

        msgs.clear()
        session.dispatch_action({"action": "close_window", "window_id": target})

        # Only the signal window is reported closed; navigator stays.
        assert _closed(msgs) == [target]
        assert session.signal_trees, "tree must remain open (navigator still up)"
        assert nav in _all_wids(session)
        assert target not in _all_wids(session)

    def test_lone_signal_window_closes_its_tree(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        msgs = tem_2d_dataset["messages"]
        wids = _all_wids(session)
        assert len(wids) == 1  # 2-D image, no navigator
        msgs.clear()
        session.dispatch_action({"action": "close_window", "window_id": wids[0]})
        assert _closed(msgs) == wids
        assert session._plots == []
        assert session.signal_trees == []

    def test_closing_unknown_window_still_notifies_renderer(self, window):
        session = window["window"]
        msgs = window["messages"]
        msgs.clear()
        session.dispatch_action({"action": "close_window", "window_id": 9999})
        assert any(
            m.get("type") == "window_closed" and m.get("window_id") == 9999
            for m in msgs
        )
