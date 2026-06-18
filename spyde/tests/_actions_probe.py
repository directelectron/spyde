"""Headless probe: exercise the template toolbar actions (Virtual Image, FFT,
Line Profile) end-to-end through the dispatcher and verify each produces a new
output plot with real data."""
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

# 4D STEM: bright disk per pattern, brightness gradient across nav.
nav, sig = (4, 5), (16, 16)
data = np.zeros(nav + sig, dtype=np.float32)
yy, xx = np.mgrid[0:16, 0:16]
disk = ((xx - 8) ** 2 + (yy - 8) ** 2 <= 9).astype(np.float32)
for i in range(nav[0]):
    for j in range(nav[1]):
        data[i, j] = disk * (i * 5 + j + 1)
s = hs.signals.Signal2D(data)
s.set_signal_type("electron_diffraction")

from spyde.backend.session import Session

sess = Session(n_workers=1, threads_per_worker=1)
sess._add_signal(s, source_path=None)
time.sleep(1.0)

# The signal (DP) window is the non-navigator one.
sig_plot = None
for p in sess._plots:
    if not p.is_navigator and p.plot_state is not None:
        sig_plot = p
        break
assert sig_plot is not None, "no signal plot found"

n_windows_before = len({c["window_id"] for c in captured
                        if c.get("type") == "window_opened"})


def run_action(name, params):
    captured.clear()
    sess._dispatch_toolbar_action(sig_plot, name, params)
    time.sleep(0.6)
    opened = [c for c in captured if c.get("type") == "window_opened"]
    errors = [c.get("text") for c in captured if c.get("type") == "error"]
    pushes = []
    for c in captured:
        if c.get("type") == "state_update" and str(c.get("key", "")).startswith("panel"):
            d = json.loads(c["value"]) if isinstance(c["value"], str) else c["value"]
            if "image_width" in d:
                pushes.append((d["image_width"], d["image_height"], round(float(d.get("display_max", 0)), 1)))
            elif d.get("kind") == "1d":
                pushes.append(("1d", len(d.get("y", d.get("data", []))), 0))
    if errors:
        print(f"  {name} ERRORS: {errors}")
    return opened, pushes


def fmt(pushes):
    return sorted(set(str(p) for p in pushes))

vi_open, vi_push = run_action("Virtual Imaging", {"type": "disk", "calculation": "mean"})
print("Virtual Imaging: windows_opened=%d pushes=%s" % (len(vi_open), fmt(vi_push)))

fft_open, fft_push = run_action("FFT", {})
print("FFT: windows_opened=%d pushes=%s" % (len(fft_open), fmt(fft_push)))

lp_open, lp_push = run_action("Line Profile", {})
print("Line Profile: windows_opened=%d pushes=%s" % (len(lp_open), fmt(lp_push)))

ok = bool(vi_open) and bool(fft_open) and bool(lp_open)
print("ALL ACTIONS OPENED OUTPUT WINDOWS:", ok)

import sys
sys.stdout.flush()
sys.stderr.flush()
import os
os._exit(0 if ok else 1)
