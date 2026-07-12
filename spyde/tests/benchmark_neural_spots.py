"""
benchmark_neural_spots.py
=========================
Real-scale spot-finding benchmark for the neural (SpotUNet) detector vs DoG vs
NXCORR on ``pyxem.data.sped_ag()`` — a real 4D-STEM SpEd Ag scan
(208x64 = 13,312 patterns of 112x112). This is the canonical target from the
Benchmarking section of CLAUDE.md.

Per-frame detection throughput is what matters for live preview + batch compute,
so this times the *detector core* on a flat stack of all patterns (the nav-blur /
ghost-overlap plumbing is shared across methods and benchmarked elsewhere). For
the neural and torch paths it follows the GPU timing rules: ``cuda.synchronize``
around the timed region and the first (cold CUDA init / kernel JIT) run discarded.

Run (NOT under pytest — torch/numba CUDA segfaults in the pytest process on
Windows):

    .venv/Scripts/python -m spyde.tests.benchmark_neural_spots
    .venv/Scripts/python -m spyde.tests.benchmark_neural_spots --frames 2000
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def _load_frames(n_frames):
    import pyxem  # noqa: F401  (registers the signal types)
    from pyxem.data import sped_ag

    s = sped_ag(allow_download=True)
    data = np.asarray(s.data, dtype=np.float32)
    flat = data.reshape(-1, data.shape[-2], data.shape[-1])
    if n_frames and n_frames < len(flat):
        flat = flat[:n_frames]
    return flat


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _time_neural(flat, batch=512):
    from spyde import models
    model, device = models.get_model()

    def run():
        out = []
        for i in range(0, len(flat), batch):
            out.extend(models.detect_batch(model, flat[i:i + batch], device, thresh=0.3))
        return out

    _ = run(); _sync()                       # warm-up (cold CUDA / JIT) — discarded
    t0 = time.perf_counter(); peaks = run(); _sync()
    dt = time.perf_counter() - t0
    return dt, peaks


def _time_method(flat, method):
    from spyde.actions.find_vectors import _find_peaks_single_frame

    params = {"method": method, "threshold": 0.5 if method == "nxcorr" else 10.0,
              "kernel_radius": 5, "min_distance": 5, "dog_sigma1": 0.8,
              "dog_sigma2": 2.0, "subpixel": True}

    def run():
        return [_find_peaks_single_frame(f, params) for f in flat]

    _ = run()[:1]                            # warm-up import/JIT
    t0 = time.perf_counter(); peaks = run()
    return time.perf_counter() - t0, peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=0, help="limit frames (0 = all 13,312)")
    args = ap.parse_args()

    print("Loading sped_ag()…")
    flat = _load_frames(args.frames)
    print(f"{len(flat)} patterns of {flat.shape[1]}x{flat.shape[2]}")

    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"
    print(f"torch device: {dev}\n")

    rows = []
    dt, pk = _time_neural(flat)
    rows.append(("neural", dt, np.mean([len(p) for p in pk])))
    for method in ("dog", "nxcorr"):
        dt, pk = _time_method(flat, method)
        rows.append((method, dt, np.mean([len(p) for p in pk])))

    n = len(flat)
    print(f"{'method':10s} {'total (s)':>10s} {'ms/frame':>10s} {'fps':>10s} {'mean peaks':>12s}")
    print("-" * 56)
    for name, dt, mean_pk in rows:
        print(f"{name:10s} {dt:10.2f} {dt / n * 1e3:10.2f} {n / dt:10.0f} {mean_pk:12.1f}")


if __name__ == "__main__":
    main()
