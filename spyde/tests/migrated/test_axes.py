"""
Editable axes panel — data flow.

The backend must (1) emit an `axes_info` message describing each axis when a
signal opens, and (2) on a `set_axis` action, write the new value back to the
real axes_manager AND re-emit `axes_info` so the table reflects it.
"""
from __future__ import annotations


def _axes_msgs(msgs):
    return [m for m in msgs if m.get("type") == "axes_info"]


class TestAxes:
    def test_axes_info_emitted_on_load(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        msgs = stem_4d_dataset["messages"]
        axm = _axes_msgs(msgs)
        assert axm, "no axes_info emitted on load"
        rows = axm[-1]["axes"]
        # 4D STEM → 2 navigation + 2 signal axes.
        assert len(rows) == 4
        for r in rows:
            assert {"index", "name", "size", "scale", "offset", "units", "navigate"} <= set(r)
        assert sum(1 for r in rows if r["navigate"]) == 2
        assert sum(1 for r in rows if not r["navigate"]) == 2
        # Window ids target the tree's windows.
        assert axm[-1]["window_ids"]

    def test_set_axis_scale_writes_back_and_reemits(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        msgs = stem_4d_dataset["messages"]
        tree = session.signal_trees[0]
        # Target a signal window's plot so set_axis resolves the tree.
        wid = sorted({p.window_id for p in session._plots if p.window_id is not None})[0]

        msgs.clear()
        session.dispatch_action({
            "action": "set_axis", "window_id": wid,
            "payload": {"index": 0, "field": "scale", "value": "0.25"},
        })

        # Written back to the real dataset.
        assert tree.root.axes_manager._axes[0].scale == 0.25
        # Table re-emitted with the new value.
        axm = _axes_msgs(msgs)
        assert axm, "set_axis did not re-emit axes_info"
        assert axm[-1]["axes"][0]["scale"] == 0.25

    def test_set_axis_name_and_units(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        tree = session.signal_trees[0]
        wid = sorted({p.window_id for p in session._plots if p.window_id is not None})[0]
        session.dispatch_action({
            "action": "set_axis", "window_id": wid,
            "payload": {"index": 3, "field": "units", "value": "1/nm"},
        })
        session.dispatch_action({
            "action": "set_axis", "window_id": wid,
            "payload": {"index": 3, "field": "name", "value": "kx"},
        })
        ax = tree.root.axes_manager._axes[3]
        assert ax.units == "1/nm"
        assert ax.name == "kx"

    def test_set_axis_ignores_garbage_numeric(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        tree = session.signal_trees[0]
        wid = sorted({p.window_id for p in session._plots if p.window_id is not None})[0]
        before = tree.root.axes_manager._axes[0].scale
        session.dispatch_action({
            "action": "set_axis", "window_id": wid,
            "payload": {"index": 0, "field": "scale", "value": "1.2e"},  # mid-typing
        })
        # Non-numeric input is ignored, not crashing or zeroing the scale.
        assert tree.root.axes_manager._axes[0].scale == before
