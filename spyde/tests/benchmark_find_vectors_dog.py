"""
benchmark_find_vectors_dog.py
=============================
GPU vs CPU parity + speed for the Difference-of-Gaussians (DoG) find-vectors
method, on the real 3 nm DESEMCam 4D-STEM scan.

torch-CUDA under the pytest process segfaults on Windows (a harness interaction,
see CLAUDE.md), so this runs as a plain script, NOT under pytest:

    .venv/Scripts/python spyde/tests/benchmark_find_vectors_dog.py
    .venv/Scripts/python spyde/tests/benchmark_find_vectors_dog.py --frames 512

It prints a JSON summary line (``BENCH_JSON {...}``) and ``os._exit(0)`` after,
so a parent process can parse the result and skip the torch/CUDA teardown crash.

Reports:
  * GPU ms/frame (warm) and CPU ms/frame, with the speedup;
  * parity: for frames where GPU and CPU agree on peak COUNT, the median /max
    nearest-neighbour position distance (sub-px refinement differs slightly
    between the scipy and torch boundary handling — same as the NXCORR path);
  * count agreement rate (off-by-one near threshold is expected and benign).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--sigma1", type=float, default=0.8)
    ap.add_argument("--sigma2", type=float, default=2.0)
    ap.add_argument("--threshold", type=float, default=10.0)
    args = ap.parse_args()

    import spyde.tests.benchmark_3nm_spots as B
    from spyde.actions.find_vectors import (
        _find_vectors_single_frame_dog, detect_beamstop)
    from spyde.actions.find_vectors_torch import (
        find_vectors_dog_torch_batch, torch_gpu_device)

    path = args.path or B.DEFAULT_MRC
    mm = B.open_memmap(path)
    s, bs_simple = B.build_sum_and_beamstop(mm)
    bs = detect_beamstop(s, frac=0.15, dilate=5)
    idx = B.pick_spotty_frames(mm, bs_simple, args.frames)
    frames = np.stack([mm[int(i)] for i in idx]).astype(np.float32)

    dev = torch_gpu_device()
    s1, s2, thr = args.sigma1, args.sigma2, args.threshold

    result = {"device": str(dev), "n_frames": len(frames),
              "beamstop_px": int(bs.sum()) if bs is not None else 0}

    # ── GPU timing (warm) ────────────────────────────────────────────────────
    if dev is not None:
        import torch
        find_vectors_dog_torch_batch(frames[:8], s1, s2, thr, 3, beamstop_mask=bs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu = find_vectors_dog_torch_batch(frames, s1, s2, thr, 3, beamstop_mask=bs)
        torch.cuda.synchronize()
        gpu_ms = (time.perf_counter() - t0) / len(frames) * 1e3
        result["gpu_ms_per_frame"] = round(gpu_ms, 4)
    else:
        gpu = None
        result["gpu_ms_per_frame"] = None

    # ── CPU timing on a subset ───────────────────────────────────────────────
    ncpu = min(64, len(frames))
    t0 = time.perf_counter()
    cpu = [_find_vectors_single_frame_dog(frames[i], s1, s2, thr, 3,
                                          beamstop_mask=bs)[2]
           for i in range(ncpu)]
    cpu_ms = (time.perf_counter() - t0) / ncpu * 1e3
    result["cpu_ms_per_frame"] = round(cpu_ms, 4)
    if result["gpu_ms_per_frame"]:
        result["speedup"] = round(cpu_ms / result["gpu_ms_per_frame"], 1)

    # ── Parity (only where GPU is available) ─────────────────────────────────
    if gpu is not None:
        count_match, pos_max = 0, []
        for i in range(ncpu):
            g, c = gpu[i][:, :2], cpu[i][:, :2]
            if len(g) == len(c):
                count_match += 1
                if len(g):
                    d = np.sqrt(((c[:, None] - g[None]) ** 2).sum(-1)).min(1)
                    pos_max.append(float(d.max()))
        result["count_agree_rate"] = round(count_match / ncpu, 3)
        result["pos_max_px"] = round(max(pos_max), 3) if pos_max else None
        result["pos_median_px"] = round(float(np.median(pos_max)), 3) if pos_max else None
        result["mean_peaks"] = round(float(np.mean([len(c) for c in cpu])), 1)

    print("BENCH_JSON " + json.dumps(result))
    print("\n=== DoG find-vectors benchmark ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    # Skip torch/CUDA teardown (segfaults on exit on Windows).
    os._exit(0)


if __name__ == "__main__":
    main()
