"""
benchmark_drift_real_scale.py — the drift caret on a REAL-SHAPE movie.

245 frames x 4096^2 uint8, lazy, memmap-backed .mrc — the shape of the
maintainer's "In-situ Electrochemistry Growth" dataset. Run directly::

    uv run python -m spyde.tests.benchmark_drift_real_scale

**Why the shape is the whole point.** Every earlier drift measurement here used
60 x 2048^2, which is 16x fewer pixels per frame and 4x fewer frames. It could
not have caught what a real dataset shows, and it did not: on real data the
corrected-sum panel came back solid BLACK, both ROI panels blank WHITE, and the
sharpness gain 0.3x — i.e. the "corrected" sum three times WORSE than the raw.
None of those are latency bugs and none of them were visible at 2048^2.

So this dumps the CONTENT of every array that reaches a panel — NaN fraction,
min/max, constancy — rather than only timing the code that produces it. A panel
is blank because the array behind it is all-NaN or constant, and that is a fact
about the array, not about the renderer.

Ground truth: frames are the base image rolled by a known integer drift, so the
solved shifts can be checked rather than eyeballed. Integer rolls (not a Fourier
ramp) because 245 sub-pixel resamples of 4096^2 costs more than it teaches here —
sub-pixel accuracy is covered by test_drift_translation.py.
"""
from __future__ import annotations

import argparse
import logging
import os
import struct
import sys
import tempfile
import time

os.environ.setdefault("SPYDE_NO_DASK", "1")
os.environ["SPYDE_ACTION_PROFILE"] = "1"
os.environ.setdefault(
    "SPYDE_SETTINGS_DIR", tempfile.mkdtemp(prefix="spyde-real-scale-"))

import numpy as np  # noqa: E402

SIZE = 4096
FRAMES = 245
DRIFT_PX = 40          # total excursion across the movie


def _truth(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    dy = DRIFT_PX * t ** 1.2
    dx = -0.5 * DRIFT_PX * t
    s = np.stack([dy, dx], 1).round().astype(int)
    return s - s[0]


def _write_mrc_uint8(path: str, size: int, frames: int) -> np.ndarray:
    """Minimal MRC2014, mode 0 (int8/uint8). Returns the ground-truth shifts."""
    truth = _truth(frames)
    want = 1024 + frames * size * size
    if os.path.exists(path) and os.path.getsize(path) == want:
        print(f"  reusing {path}")
        return truth
    print(f"  building {frames} x {size}^2 uint8 ({want / 2**30:.1f} GiB)")
    rng = np.random.default_rng(0)
    # Structure at several scales so a correlation has something to lock onto,
    # and so gradient energy (the gain) means something.
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = (60 + 40 * np.sin(yy / 37.0) * np.cos(xx / 41.0)
            + 25 * np.sin((yy + xx) / 11.0))
    for _ in range(40):                       # bright particles
        cy, cx = rng.uniform(0.05, 0.95, 2) * size
        r = rng.uniform(20, 70)
        m = (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
        base[m] = 220
    base = np.clip(base, 0, 255).astype(np.uint8)

    hdr = bytearray(1024)
    struct.pack_into("<iii", hdr, 0, size, size, frames)
    struct.pack_into("<i", hdr, 12, 0)                    # mode 0 = int8/uint8
    struct.pack_into("<iii", hdr, 28, size, size, frames)
    struct.pack_into("<fff", hdr, 40, float(size), float(size), float(frames))
    struct.pack_into("<fff", hdr, 52, 90.0, 90.0, 90.0)
    struct.pack_into("<iii", hdr, 64, 1, 2, 3)
    struct.pack_into("<i", hdr, 92, 0)
    hdr[208:212] = b"MAP "
    hdr[212:216] = bytes([0x44, 0x44, 0, 0])
    tmp = path + ".part"
    t0 = time.perf_counter()
    with open(tmp, "wb") as fh:
        fh.write(hdr)
        for i, (dy, dx) in enumerate(truth):
            fh.write(np.roll(base, (int(dy), int(dx)), axis=(0, 1)).tobytes())
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{frames}  {time.perf_counter() - t0:.0f}s")
    os.replace(tmp, path)
    return truth


def _describe(name: str, a) -> str:
    """What a panel would actually SHOW. Blank white and solid black are both
    'the array behind it has no structure', and that is checkable."""
    if a is None:
        return f"    {name:<16} None  (never painted)"
    a = np.asarray(a, np.float64)
    finite = np.isfinite(a)
    nan_pct = 100.0 * (1.0 - finite.mean())
    if not finite.any():
        return f"    {name:<16} shape={a.shape} ALL NaN  -> blank panel"
    v = a[finite]
    flat = "  CONSTANT -> blank panel" if v.max() - v.min() < 1e-9 else ""
    return (f"    {name:<16} shape={a.shape} nan={nan_pct:5.1f}%  "
            f"min={v.min():.4g} max={v.max():.4g} mean={v.mean():.4g}{flat}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--normal-load", action="store_true",
                    help="wait for the session's torch prewarm before opening the "
                         "caret — what a user who spent a few seconds looking at "
                         "the data actually experiences. Without it the caret RACES "
                         "the prewarm, which is a harness artefact, not the app.")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    for noisy in ("hyperspy", "rsciio", "distributed", "asyncio", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    import hyperspy.api as hs
    from spyde.backend.session import Session
    import spyde.actions.drift_action as dr

    path = os.path.join(tempfile.gettempdir(),
                        f"spyde-real-{args.frames}x{args.size}.mrc")
    truth = _write_mrc_uint8(path, args.size, args.frames)
    sig = hs.load(path, lazy=True)
    ax = sig.axes_manager.navigation_axes[0]
    ax.name, ax.units, ax.scale = "time", "s", 0.26
    sig.set_signal_type("insitu")
    print(f"  loaded: nav {sig.axes_manager.navigation_shape} "
          f"sig {sig.axes_manager.signal_shape} {sig.data.dtype} lazy")

    session = Session(n_workers=1, threads_per_worker=1)
    painted: dict = {}
    try:
        session._add_signal(sig, source_path=path)
        plot = next(p for p in session._plots
                    if not p.is_navigator and p.plot_state is not None)
        tree = plot.signal_tree

        if args.normal_load:
            # A user opens a 3.8 GiB movie, looks at it, THEN reaches for the
            # caret — by which time the session's background torch prewarm has
            # long finished. Opening the caret microseconds after the load races
            # that prewarm, which is a harness artefact and not what anyone
            # experiences. Both numbers are worth having; only one is the user's.
            from spyde.backend.heavy_imports import wait_for_torch
            _t = time.perf_counter()
            wait_for_torch(180.0)
            print(f"  (waited {time.perf_counter() - _t:.1f}s for the session's "
                  f"torch prewarm, as a browsing user would have)")

        print("\n== drift_open ==")
        t0 = time.perf_counter()
        dr.drift_open(session, plot, {})
        end = time.time() + 900
        while time.time() < end and (
                getattr(tree, "_drift_wizard", None) is None
                or tree._drift_wizard.window_id is None):
            time.sleep(0.05)
        wiz = tree._drift_wizard
        # Record exactly what reaches the panels.
        real_show, real_check = wiz.show_preview, wiz.update_check
        wiz.show_preview = lambda r: (painted.update(preview=r), real_show(r))
        wiz.update_check = lambda **kw: (painted.update(after=kw.get("after")),
                                         real_check(**kw))
        end = time.time() + 900
        while time.time() < end and wiz.preview is None:
            time.sleep(0.05)
        print(f"  open+first preview wall = {time.perf_counter() - t0:.1f}s")
        print(f"  box = {wiz.roi_box()}   preview_frames param = "
              f"{wiz.params['preview_frames']}")
        print(_describe("raw sum", wiz._before_sum))

        print("\n== a preview step (what the ROI panels get) ==")
        r = dr._preview_step(wiz)
        print(f"    frames={r['frames']}  gain={r['gain']:.3f}  "
              f"max_abs_shift={r['max_abs_shift']:.2f}px")
        print(_describe("ROI raw", r["raw"]))
        print(_describe("ROI aligned", r["aligned"]))

        print("\n== drift_run ==")
        t0 = time.perf_counter()
        dr.drift_run(session, plot, {})
        end = time.time() + 3600
        while time.time() < end and wiz.model is None:
            time.sleep(0.2)
        print(f"  Correct Drift wall = {time.perf_counter() - t0:.1f}s")
        got = np.asarray(wiz.model.shifts, float)
        # frame_i = roll(base, +truth[i]), so the correction to ADD is -truth[i]
        # (DriftModel sign convention). Comparing against +truth reports an error
        # of ~2x the drift for a PERFECT solve — which is what it did, and which
        # is why this line is spelled out rather than inlined.
        expect = -truth[:len(got)]
        err = np.abs(got - expect)
        print(f"  shifts vs ground truth: max err {np.nanmax(err):.2f} px  "
              f"rms {np.sqrt(np.nanmean(err ** 2)):.2f} px")
        print(_describe("corrected sum", painted.get("after")))
    finally:
        session.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
