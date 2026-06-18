"""
Deselecting a toolbar action (and closing a window) must hide the output window
and ROI selector it created — Qt parity for "an unchecked action removes its
artifacts".
"""
from __future__ import annotations

import time


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


# FFT is a directly-dispatched RegionAction (auto-tracked via the generic
# action_active path) — Virtual Imaging is now a submenu (see test_virtual_imaging).
ACTION = "FFT"


def _run_action(session):
    src = _signal_plot(session)
    assert src is not None
    session._dispatch_toolbar_action(src, ACTION, {})
    time.sleep(0.4)
    return src


class TestActionDeselect:
    def test_running_region_action_tracks_artifact_and_marks_active(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        msgs = stem_4d_dataset["messages"]
        msgs.clear()
        src = _run_action(session)

        key = (src.window_id, ACTION)
        assert key in session._action_artifacts
        assert session._action_artifacts[key]["out_wids"], "no output window tracked"
        assert any(m.get("type") == "action_active" and m.get("active") is True
                   and m.get("name") == ACTION for m in msgs)

    def test_deselect_hides_output_window_and_clears_artifact(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        msgs = stem_4d_dataset["messages"]
        src = _run_action(session)
        key = (src.window_id, ACTION)
        out_wids = list(session._action_artifacts[key]["out_wids"])
        assert out_wids

        msgs.clear()
        session.dispatch_action({
            "action": "set_action_active", "window_id": src.window_id,
            "payload": {"name": ACTION, "active": False},
        })

        closed = {m["window_id"] for m in msgs if m.get("type") == "window_closed"}
        assert set(out_wids) <= closed, f"output {out_wids} not closed (got {closed})"
        assert any(m.get("type") == "action_active" and m.get("active") is False
                   for m in msgs)
        assert key not in session._action_artifacts
        # The output plot is really gone.
        assert all(p.window_id not in out_wids for p in session._plots)

    def test_closing_output_window_unmarks_source_action(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        msgs = stem_4d_dataset["messages"]
        src = _run_action(session)
        key = (src.window_id, ACTION)
        out_wid = session._action_artifacts[key]["out_wids"][0]

        msgs.clear()
        # User closes the output window directly.
        session.dispatch_action({"action": "close_window", "window_id": out_wid})

        # The source toolbar is told to un-highlight the action, artifact cleared.
        assert any(m.get("type") == "action_active" and m.get("active") is False
                   and m.get("name") == ACTION for m in msgs)
        assert key not in session._action_artifacts
