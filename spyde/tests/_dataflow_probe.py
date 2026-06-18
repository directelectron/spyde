"""Headless data-flow probe: load a synthetic 4D dataset through Session and
verify real image data reaches both the navigator and the diffraction plot."""
import json
import time

import numpy as np
import hyperspy.api as hs

import spyde.backend.ipc as ipc
import anyplotlib._electron as ael

captured = []


def cap(obj):
    captured.append(obj)


ipc.emit = cap
ael.emit = cap

nav = (4, 5)
sig = (8, 8)
data = np.zeros(nav + sig, dtype=np.float32)
yy, xx = np.mgrid[0:8, 0:8]
disk = ((xx - 4) ** 2 + (yy - 4) ** 2 <= 4).astype(np.float32)
for i in range(nav[0]):
    for j in range(nav[1]):
        data[i, j] = disk * (i * 5 + j + 1)

s = hs.signals.Signal2D(data)

from spyde.backend.session import Session

sess = Session(n_workers=1, threads_per_worker=1)
sess._add_signal(s, source_path=None)
time.sleep(1.5)

final = {}
for c in captured:
    if c.get("type") == "state_update" and str(c.get("key", "")).startswith("panel"):
        d = json.loads(c["value"]) if isinstance(c["value"], str) else c["value"]
        final[c["fig_id"]] = d

print("=== final pushed image per figure ===")
nav_ok = dp_ok = False
for fid, d in final.items():
    w, h = d["image_width"], d["image_height"]
    dmin, dmax = d["display_min"], d["display_max"]
    nw = len(d["overlay_widgets"])
    print(f"fig={fid[:6]} img={w}x{h} disp=[{dmin:.0f},{dmax:.0f}] widgets={nw}")
    if w == 5 and h == 4 and dmax > 1:
        nav_ok = True
    if w == 8 and h == 8 and dmax > 1:
        dp_ok = True

print("NAV filled with real data:", nav_ok)
print("DP filled with real data:", dp_ok)
import sys
sys.stdout.flush()
sys.stderr.flush()
import os
os._exit(0 if (nav_ok and dp_ok) else 1)
