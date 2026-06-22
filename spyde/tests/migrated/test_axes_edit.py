"""
Axes table edits write back to the dataset AND reflect in the plot immediately.

`set_axis` mutates the real axes_manager, re-pushes every plot in the tree (so the
extent / scale bar update at once), and re-emits the axes table + metadata. The
dataset shape now lives in the Metadata panel (`Dataset` section), not the table.
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


class TestAxesEdit:
    def test_set_axis_writes_back_and_re_emits(self):
        import spyde.backend.session as sess_mod
        from spyde.backend.session import Session
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            s = hs.signals.Signal2D(np.zeros((4, 5, 24, 24), np.float32))
            s.set_signal_type("electron_diffraction")
            session._add_signal(s)
            time.sleep(0.3)
            plot = _signal_plot(session)
            tree = plot.signal_tree

            captured = []
            orig = sess_mod.emit          # session.py binds `emit` at import
            sess_mod.emit = lambda m: captured.append(m)
            try:
                # Edit signal-axis 3 (kx) scale → 0.25.
                session._set_axis(plot, {"index": 3, "field": "scale", "value": "0.25"})
            finally:
                sess_mod.emit = orig

            # Written to the real axes_manager (immediate, in-memory).
            assert abs(float(tree.root.axes_manager._axes[3].scale) - 0.25) < 1e-9
            # Re-emitted the axes table with the new value …
            axes_msgs = [m for m in captured if m.get("type") == "axes_info"]
            assert axes_msgs
            scales = [a["scale"] for a in axes_msgs[-1]["axes"]]
            assert any(abs(sc - 0.25) < 1e-9 for sc in scales)
            # … and re-emitted metadata (Dataset shape lives there now).
            md = [m for m in captured if m.get("type") == "metadata"]
            assert md and "Dataset" in md[-1]["metadata"]
            assert "Shape" in md[-1]["metadata"]["Dataset"]
        finally:
            session.shutdown()

    def test_scale_edit_preserves_origin_pixel(self):
        """Changing scale rescales the offset so the (0,0) data point stays on
        the SAME pixel — recalibrating pixel size must not move the marked
        origin / crosshair centre."""
        import hyperspy.api as hs
        from spyde.backend.session import Session
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            s = hs.signals.Signal2D(np.zeros((4, 5, 20, 20), np.float32))
            s.set_signal_type("electron_diffraction")
            # _axes order is nav-first; signal kx is index 3 (matches the test
            # above). Configure that axis so the edit targets it.
            s.axes_manager._axes[3].scale = 0.1
            s.axes_manager._axes[3].offset = -1.0   # origin pixel = 1.0/0.1 = 10
            session._add_signal(s)
            time.sleep(0.3)
            plot = _signal_plot(session)
            axx = plot.signal_tree.root.axes_manager._axes[3]
            origin_px = -float(axx.offset) / float(axx.scale)
            assert abs(origin_px - 10.0) < 1e-9

            # double the scale → offset should double so origin pixel stays 10
            session._set_axis(plot, {"index": 3, "field": "scale", "value": "0.2"})
            assert abs(float(axx.scale) - 0.2) < 1e-9
            assert abs(float(axx.offset) - (-2.0)) < 1e-9
            new_origin_px = -float(axx.offset) / float(axx.scale)
            assert abs(new_origin_px - origin_px) < 1e-9
        finally:
            session.shutdown()
