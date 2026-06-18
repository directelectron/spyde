"""Scale bar: physical signal-axis units must reach the panel state so
anyplotlib draws its (automatic) calibrated scale bar."""
import json
import time

import numpy as np
import hyperspy.api as hs


def _panel_units(messages):
    """Return the set of (units, scale_x>0) seen in panel state pushes."""
    seen = []
    for m in messages:
        if m.get("type") == "state_update" and str(m.get("key", "")).startswith("panel"):
            d = json.loads(m["value"]) if isinstance(m["value"], str) else m["value"]
            if "units" in d:
                seen.append((d.get("units"), float(d.get("scale_x", 0)) > 0))
    return seen


class TestScaleBar:
    def test_physical_units_reach_panel_state(self, captured_messages):
        from spyde.backend.session import Session

        # 2-D image whose axes are calibrated in nm.
        s = hs.signals.Signal2D(np.random.RandomState(0).rand(32, 32).astype(np.float32))
        for ax in s.axes_manager.signal_axes:
            ax.units = "nm"
            ax.scale = 0.5
        sess = Session(n_workers=1, threads_per_worker=1)
        sess._add_signal(s, source_path=None)
        time.sleep(0.8)
        sess.shutdown()

        units = _panel_units(captured_messages)
        # The scale bar renders when units != 'px' and scale_x > 0.
        assert any(u == "nm" and has_scale for (u, has_scale) in units), \
            f"no calibrated panel push; got {set(units)}"

    def test_pixel_units_give_no_scale(self, captured_messages):
        from spyde.backend.session import Session

        # Uncalibrated image → units stay 'px' → no scale bar (correct).
        s = hs.signals.Signal2D(np.random.RandomState(1).rand(16, 16).astype(np.float32))
        sess = Session(n_workers=1, threads_per_worker=1)
        sess._add_signal(s, source_path=None)
        time.sleep(0.6)
        sess.shutdown()

        units = _panel_units(captured_messages)
        assert all(u != "nm" for (u, _) in units)
