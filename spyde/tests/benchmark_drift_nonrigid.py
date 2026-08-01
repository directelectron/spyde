"""
benchmark_drift_nonrigid.py — what does a non-rigid solve cost on a real movie?

Run directly (torch-CUDA segfaults under pytest on Windows -- CLAUDE.md)::

    python -m spyde.tests.benchmark_drift_nonrigid
    python -m spyde.tests.benchmark_drift_nonrigid --device cpu --frames 100

The question is 4096x4096 x hundreds of frames, and the first thing to say
about it is that the stack CANNOT be held: 100 x 4096^2 float32 is 6.7 GB and
300 is 20 GB. So the solve is necessarily two separate costs and they are
measured separately here, because they scale differently and only one of them
is paid per frame:

1. **The FIT**, on a DECIMATED stack. A drift field is smooth by construction --
   that is the entire modelling assumption -- so it does not need full
   resolution to be measured. The fit is over a handful of parameters per frame
   (2 x n_knots, or 2 x gh x gw), and the decimation factor is the dominant
   cost knob.
2. **The APPLY**, at FULL resolution, once per frame, in the display/export
   path. This is the number that decides whether scrubbing stays interactive.

Reporting one blended figure would hide exactly the thing a caller needs to
decide: how much to decimate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def _synth(n: int, h: int, w: int, seed: int = 0) -> np.ndarray:
    """A textured stack with a known slow-axis distortion, at fit resolution."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.zeros((h, w), np.float32)
    for _ in range(40):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        s = rng.uniform(h / 60, h / 20)
        base += rng.uniform(0.5, 1.5) * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * s * s))
    base += 0.05 * rng.standard_normal((h, w)).astype(np.float32)

    from spyde.drift import nonrigid as nr
    import torch
    rows = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    out = np.empty((n, h, w), np.float32)
    t = torch.as_tensor(base)[None]
    for i in range(n):
        a = 3.0 * (0.3 + 0.7 * i / max(n - 1, 1))
        dy = np.repeat((a * rows)[:, None], w, axis=1)
        g = nr.warp_frame(torch, t, torch.as_tensor(dy)[None],
                          torch.as_tensor(np.zeros((h, w), np.float32))[None],
                          fill_nan=False)
        out[i] = g[0].numpy()
    return out


def _sync(device: str) -> None:
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--full", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import torch
    from spyde.drift import nonrigid as nr

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}   frames: {a.frames}   full frame: {a.full}^2   steps: {a.steps}")
    print(f"(the full stack would be "
          f"{a.frames * a.full * a.full * 4 / 1e9:.1f} GB -- hence decimation)\n")
    out: dict = {"device": dev, "frames": a.frames, "full": a.full, "steps": a.steps}

    # ── 1. fit cost vs decimated size ────────────────────────────────────────
    print("=== FIT (decimated stack, whole movie at once) ===")
    print(f"{'fit size':>10} {'decim':>7} {'model':>10} {'build':>8} {'fit':>9} {'per-frame':>10}")
    out["fit"] = {}
    for side in (128, 256, 512):
        t0 = time.perf_counter()
        stack = _synth(a.frames, side, side)
        build = time.perf_counter() - t0
        for model, kw in ((nr.SCAN_KNOT, dict(n_knots=3)), (nr.DENSE, dict(grid=(6, 6)))):
            _sync(dev)
            t0 = time.perf_counter()
            nr.solve_nonrigid(stack, model=model, steps=a.steps, device=dev, **kw)
            _sync(dev)
            el = time.perf_counter() - t0
            print(f"{side:>7}^2 {a.full // side:>6}x {model:>10} {build:>7.2f}s "
                  f"{el:>8.2f}s {el / a.frames * 1e3:>8.1f} ms")
            out["fit"][f"{side}/{model}"] = {"s": el, "per_frame_ms": el / a.frames * 1e3}
        del stack

    # ── 2. apply cost at FULL resolution, per frame ──────────────────────────
    print(f"\n=== APPLY, {a.full}^2, per frame (display/export path) ===")
    out["apply"] = {}
    small = _synth(4, 128, 128)
    frame = np.ascontiguousarray(
        np.random.default_rng(1).random((a.full, a.full), dtype=np.float32))
    for model, kw in ((nr.SCAN_KNOT, dict(n_knots=3)), (nr.DENSE, dict(grid=(6, 6)))):
        m = nr.solve_nonrigid(small, model=model, steps=10, device=dev, **kw)
        # Re-target the fitted field to the full frame: the parameters are
        # resolution-independent by construction, which is the property that
        # makes fit-small/apply-large legitimate rather than a shortcut.
        m.extra["field_shape"] = (a.full, a.full)
        nr.apply_nonrigid(frame, m, 0)                       # warm
        t0 = time.perf_counter()
        reps = 3
        for _ in range(reps):
            nr.apply_nonrigid(frame, m, 0)
        el = (time.perf_counter() - t0) / reps
        print(f"  {model:>10}  {el * 1e3:>7.0f} ms/frame   "
              f"-> {el * a.frames:>6.1f} s for {a.frames} frames")
        out["apply"][model] = {"ms": el * 1e3, "movie_s": el * a.frames}

    print("\n" + json.dumps(out, indent=2, default=float))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=float)
    # FLUSH BEFORE `os._exit`. `_exit` skips atexit and does NOT flush stdio, so
    # with output redirected to a file (buffered, not line-buffered) every print
    # above is discarded and the run looks like it produced nothing. Invisible
    # on a terminal, which is why it is easy to write and easy to miss.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)          # skip torch/CUDA teardown crash (CLAUDE.md)


if __name__ == "__main__":
    main()
