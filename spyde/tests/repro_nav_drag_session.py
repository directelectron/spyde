"""
Pure-Python reproduction of the distributed crosshair-drag DP-update bug, with a
REAL Session + REAL Dask cluster + REAL signal tree/selectors — no Electron, no
iframe harness. Drives the navigator crosshair across chunk boundaries and checks
the SIGNAL plot's painted frame (current_data) changes per move.

    .venv/Scripts/python -m spyde.tests.repro_nav_drag_session

Prints DRAG_JSON {...} and os._exit(0).
"""
from __future__ import annotations
import asyncio, os, sys, threading, time
import numpy as np


def log(m): print(f"{time.monotonic()-T0:6.1f}s {m}", file=sys.stderr, flush=True)
T0 = time.monotonic()


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s",
                        stream=sys.stderr)
    os.environ["SPYDE_NAV_TIMING"] = "1"

    import hyperspy.api as hs
    import dask.array as da
    from spyde.backend.session import Session

    # A real asyncio loop on a background thread → marshaling target (main-thread
    # applies). Without it _dispatch_to_main runs inline (also fine to compare).
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True, name="main-loop")
    loop_thread.start()

    session = Session(n_workers=2, threads_per_worker=2)
    session.set_main_loop(loop)
    session.start_dask()
    # Wait for the cluster.
    for _ in range(120):
        if session.dask_manager.client is not None:
            break
        time.sleep(0.5)
    log(f"cluster client={session.dask_manager.client is not None}")

    # Match the USER's real scan: 300×648 nav, 128×128 signal, nav chunks of 64
    # (→ 5×11 chunk grid, "size 5" on axis 0), and CALIBRATED nav axes (nm scale)
    # like the user's 3 nm scan — the edge cross-chunk move is where "Index N out
    # of bounds for axis 0 with size N" appears. Keep ky/kx small so memory is
    # sane. (The widget reports PIXEL coords, so calibration no longer affects the
    # index; the scale is kept only to mirror the real dataset.)
    ny, nx, ky, kx = 300, 648, 16, 16
    rng = np.random.RandomState(0)
    arr = da.random.randint(0, 255, size=(ny, nx, ky, kx), dtype=np.int16,
                            chunks=(64, 64, ky, kx))
    s = hs.signals.Signal2D(arr).as_lazy()
    s.set_signal_type("electron_diffraction")
    # Calibrate the NAVIGATION axes (3 nm step), as in the real scan.
    for ax in s.axes_manager.navigation_axes:
        ax.scale = 3.0
        ax.units = "nm"
    log(f"nav axes scale={[ax.scale for ax in s.axes_manager.navigation_axes]} "
        f"nav_shape={s.axes_manager.navigation_shape}")
    session._add_signal(s, source_path="repro")
    time.sleep(4.0)   # let nav compute + first DP settle

    tree = session.signal_trees[-1]
    mgr = tree.navigator_plot_manager
    pw = next(iter(mgr.navigation_selectors.keys()))
    sel = mgr.navigation_selectors[pw][0]
    cross = getattr(sel, "_crosshair_selector", sel)
    child = next(iter(sel.children.keys()))
    log(f"selector ready, crosshair widget={cross._widget is not None}")

    def frame_sig():
        d = getattr(child, "current_data", None)
        if isinstance(d, np.ndarray) and d.size:
            return (int(d.argmax()), float(d.sum()))
        return None

    def cd_kind():
        return type(getattr(child, "current_data", None)).__name__

    # Targets in DATA (nm) coordinates — including the FAR EDGES/CORNERS where the
    # cross-chunk "Index N out of bounds" appears. nav is 648(x)×300(y) px at 3nm:
    # x_max ≈ 647*3 = 1941nm, y_max ≈ 299*3 = 897nm. Push slightly PAST the edge
    # (1944, 900) to mimic the widget reporting a position at the image boundary.
    sx = float(s.axes_manager.navigation_axes[0].scale)
    sy = float(s.axes_manager.navigation_axes[1].scale)
    targets = [
        (10 * sx, 10 * sy),           # interior
        (647 * sx, 299 * sy),         # exact far corner (last valid pixel)
        (648 * sx, 300 * sy),         # ONE PAST → the suspected OOB
        (648 * sx, 299 * sy),         # x one past
        (647 * sx, 300 * sy),         # y one past
        (320 * sx, 256 * sy),         # interior of last y-chunk (y=256..299)
        (60 * sx, 60 * sy),
    ]
    prev = frame_sig()
    results = []
    for (x, y) in targets:
        cross._widget.cx = float(x)
        cross._widget.cy = float(y)
        sel.delayed_update_data(force=True)
        cur, t_end = prev, time.monotonic() + 4
        while time.monotonic() < t_end:
            cur = frame_sig()
            if cur is not None and cur != prev:
                break
            time.sleep(0.03)
        changed = cur is not None and cur != prev
        results.append({"xy": (x, y), "changed": bool(changed), "cd_kind": cd_kind(),
                        "sig": cur})
        log(f"move ({x},{y}) changed={changed} cd_kind={cd_kind()} sig={cur}")
        prev = cur

    import json
    n_changed = sum(1 for r in results if r["changed"])
    print("DRAG_JSON " + json.dumps({"changed": n_changed, "total": len(targets),
                                     "results": results}))
    sys.stdout.flush()
    try:
        session.dask_manager.shutdown()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
