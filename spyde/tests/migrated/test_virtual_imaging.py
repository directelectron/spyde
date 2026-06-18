"""
Virtual Imaging multi-VI sub-toolbar (Qt parity).

"Virtual Imaging" is a submenu; "Add Virtual Image" adds a colour-cycled ROI VI
(its own output window) and emits a `sub_item` so the sub-toolbar lists it. You
can add several (colours cycle red→green→blue→…) and remove one via
`set_action_active(name, active=False)`, which closes its window + drops the chip.
"""
from __future__ import annotations

import time


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _sub_items(msgs):
    return [m for m in msgs if m.get("type") == "sub_item"]


def _add_vi(session, src):
    session._dispatch_toolbar_action(
        src, "add_virtual_image", {"type": "disk", "calculation": "mean"})
    time.sleep(0.3)


class TestVirtualImagingSubToolbar:
    def test_add_virtual_image_creates_colored_vi_and_emits_chip(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        msgs = stem_4d_dataset["messages"]
        src = _signal_plot(session)
        n_plots = len(session._plots)

        msgs.clear()
        _add_vi(session, src)

        # A new output window was opened.
        assert len(session._plots) == n_plots + 1
        # A sub_item chip was emitted for the Virtual Imaging sub-toolbar.
        chips = [m for m in _sub_items(msgs) if m.get("active")]
        assert len(chips) == 1
        chip = chips[0]
        assert chip["action"] == "Virtual Imaging"
        assert chip["color"] == "red"          # first colour in the cycle
        assert chip["window_id"] == src.window_id
        # Tracked as an artifact for removal.
        assert (src.window_id, chip["name"]) in session._action_artifacts

    def test_colors_cycle_across_adds(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        src = _signal_plot(session)
        for _ in range(3):
            _add_vi(session, src)
        colors = [it["color"] for it in src._vi_items]
        assert colors == ["red", "green", "blue"]
        assert len(src._vi_items) == 3

    def test_update_vi_applies_caret_params_live(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        msgs = stem_4d_dataset["messages"]
        src = _signal_plot(session)
        _add_vi(session, src)
        vi = src._vi_items[0]
        assert vi["calculation"] == "mean"
        act = session._action_artifacts[(src.window_id, vi["name"])]["action"]

        msgs.clear()
        session.dispatch_action({
            "action": "update_vi", "window_id": src.window_id,
            "payload": {"name": vi["name"], "params": {"calculation": "sum"}},
        })
        time.sleep(0.2)

        # The VI's live params changed (so the next recompute uses them) and the
        # chip is re-emitted with the new value.
        assert act._live_params["calculation"] == "sum"
        assert src._vi_items[0]["calculation"] == "sum"
        assert any(m.get("type") == "sub_item" and m.get("name") == vi["name"]
                   and m.get("calculation") == "sum" for m in msgs)

    def test_type_change_rebuilds_the_roi_shape(self, stem_4d_dataset):
        from spyde.drawing.selectors import CircleSelector, AnnularSelector
        session = stem_4d_dataset["window"]
        src = _signal_plot(session)
        _add_vi(session, src)                      # default type = disk
        vi = src._vi_items[0]
        art = session._action_artifacts[(src.window_id, vi["name"])]
        assert isinstance(art["action"]._selector, CircleSelector)
        old_sel = art["action"]._selector

        # Switch detector type → the on-plot ROI is rebuilt as a ring.
        session.dispatch_action({
            "action": "update_vi", "window_id": src.window_id,
            "payload": {"name": vi["name"], "params": {"type": "annular"}},
        })
        time.sleep(0.2)
        assert isinstance(art["action"]._selector, AnnularSelector)
        assert art["action"]._selector is not old_sel
        assert art["selector"] is art["action"]._selector   # removal ref refreshed

        # A calc-only change must NOT rebuild the selector.
        sel_after_type = art["action"]._selector
        session.dispatch_action({
            "action": "update_vi", "window_id": src.window_id,
            "payload": {"name": vi["name"], "params": {"calculation": "sum"}},
        })
        time.sleep(0.2)
        assert art["action"]._selector is sel_after_type

    def test_remove_chip_closes_window_and_drops_it(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        msgs = stem_4d_dataset["messages"]
        src = _signal_plot(session)
        _add_vi(session, src)
        vi = src._vi_items[0]
        out_wids = list(session._action_artifacts[(src.window_id, vi["name"])]["out_wids"])
        assert out_wids

        msgs.clear()
        session.dispatch_action({
            "action": "set_action_active", "window_id": src.window_id,
            "payload": {"name": vi["name"], "active": False},
        })

        # Output window closed, chip removed (sub_item active:false), list emptied.
        closed = {m["window_id"] for m in msgs if m.get("type") == "window_closed"}
        assert set(out_wids) <= closed
        assert any(m.get("type") == "sub_item" and m.get("active") is False
                   and m.get("name") == vi["name"] for m in msgs)
        assert src._vi_items == []
        assert (src.window_id, vi["name"]) not in session._action_artifacts
