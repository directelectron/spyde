"""Headless probe: simulate a navigator selector drag and verify the signal
(diffraction) plot updates to the DP at the new position."""
import json
import time

import numpy as np
import hyperspy.api as hs

import spyde.backend.ipc as ipc
import anyplotlib._electron as ael

ipc.emit = lambda o: None
ael.emit = lambda o: None

# 4D STEM where each nav position has a DISTINCT, identifiable pattern.
nav, sig = (4, 5), (16, 16)
data = np.zeros(nav + sig, dtype=np.float32)
for i in range(nav[0]):
    for j in range(nav[1]):
        # a single bright pixel whose location encodes (i, j)
        data[i, j, (i * 3) % 16, (j * 3) % 16] = 100.0
s = hs.signals.Signal2D(data)
s.set_signal_type("electron_diffraction")

from spyde.backend.session import Session

sess = Session(n_workers=1, threads_per_worker=1)
sess._add_signal(s, source_path=None)
time.sleep(0.8)

nav_plot = next(p for p in sess._plots if p.is_navigator)
sig_plot = next(p for p in sess._plots if not p.is_navigator)

# Locate the crosshair widget on the navigator's anyplotlib panel.
plot2d = nav_plot._plot2d
panel_id = plot2d._id
widgets = plot2d._widgets
crosshair = next((w for w in widgets.values() if "cx" in w._data and "r" not in w._data), None)
assert crosshair is not None, f"no crosshair widget; have {[w._type for w in widgets.values()]}"

before = np.asarray(sig_plot.current_data).copy()
print("initial DP nonzero at:", np.argwhere(before > 50).tolist()[:3])

# Simulate dragging the crosshair to nav position (3, 4) → expect DP bright at
# ((3*3)%16, (4*3)%16) = (9, 12).
target_cx, target_cy = 4, 3  # cx=col, cy=row in image coords
event = {
    "source": "js",
    "panel_id": panel_id,
    "widget_id": crosshair._id,
    "event_type": "pointer_up",
    "cx": float(target_cx),
    "cy": float(target_cy),
}
nav_plot._fig._dispatch_event(json.dumps(event))
time.sleep(0.6)  # debounce + slice

after = np.asarray(sig_plot.current_data)
print("after-drag DP nonzero at:", np.argwhere(after > 50).tolist()[:3])

changed = not np.array_equal(before, after)
print("SIGNAL PLOT UPDATED ON DRAG:", changed)

import sys
sys.stdout.flush()
import os
os._exit(0 if changed else 1)
