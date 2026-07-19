"""Memory-scaling test for ``detect_batch`` sub-batching (infer.py).

``_neural_block`` used to call ``models.detect_batch`` with EVERY frame in a nav
chunk (1000+ for a real 4D-STEM scan) in ONE forward pass — one (N,1,H,W) CUDA
tensor whose activations scaled with N and ballooned per-worker memory to
~15 GB. ``detect_batch`` now loops in fixed sub-batches of
``SPYDE_NEURAL_BATCH`` frames (default 64), freeing CUDA blocks between
sub-batches, so peak activation memory scales with K, not N.

Run in a SUBPROCESS that prints a JSON result and ``os._exit(0)`` after, matching
``test_neural_detect.py`` / ``test_vector_orientation_gpu.py``: torch-CUDA
teardown can segfault at interpreter exit inside the pytest process on Windows.
This test specifically needs a real CUDA device (the memory-scaling claim is
about the CUDA caching allocator) — it skips cleanly when none is available.

Covers:
  - peak CUDA memory with a small sub-batch (K=32) is substantially lower
    (>3x) than with the whole stack in one pass (K=N=256);
  - the DETECTED PEAKS are bit-identical between the two K values — the
    per-batch scale/bg_sigma/NMS-window are computed ONCE from frame[0]
    (outside the sub-batch loop), so chunking the forward pass must not
    change results.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

_DRIVER = textwrap.dedent(r"""
    import json, sys, os
    import numpy as np
    from scipy.ndimage import gaussian_filter

    def planted_stack(n, shape=(128, 128), amp=400.0, sigma=2.2, seed=0):
        rng = np.random.default_rng(seed)
        frames = np.zeros((n,) + shape, np.float32)
        for i in range(n):
            f = np.zeros(shape, np.float32)
            # A couple of planted disks per frame, jittered so frames differ
            # slightly (closer to real data) but the physical disk size is
            # constant across the stack (the shared-scale/bg_sigma invariant).
            for (cy, cx) in [(50, 60), (80, 40)]:
                dy, dx = rng.integers(-2, 3), rng.integers(-2, 3)
                f[cy + dy, cx + dx] = amp
            frames[i] = gaussian_filter(f, sigma)
        return frames

    mode = sys.argv[1]
    out = {}

    if mode == "check_cuda":
        import torch
        out["cuda"] = bool(torch.cuda.is_available())
        print("RESULT_JSON", json.dumps(out))
        sys.stdout.flush()
        os._exit(0)

    elif mode == "scaling":
        import torch
        from spyde import models

        device = torch.device("cuda")
        model, dev = models.get_model()
        assert dev.type == "cuda", f"model did not load onto cuda: {dev}"

        N = 256
        frames = planted_stack(N)

        def run_with_batch(k):
            os.environ["SPYDE_NEURAL_BATCH"] = str(k)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            peaks = models.detect_batch(model, frames, device, thresh=0.3)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            return peak, peaks

        # Warm-up run (cold CUDA init / kernel JIT) discarded from the measurement,
        # but keep its K so the very first call doesn't pollute the comparison.
        run_with_batch(32)

        peak_small, peaks_small = run_with_batch(32)
        peak_large, peaks_large = run_with_batch(N)  # K=N: whole stack in one pass

        out["peak_small_bytes"] = int(peak_small)
        out["peak_large_bytes"] = int(peak_large)
        out["ratio"] = float(peak_large) / float(peak_small) if peak_small else float("inf")

        # Bit-identical results: same per-frame peak count and positions.
        same_n = len(peaks_small) == len(peaks_large) == N
        out["same_frame_count"] = same_n
        max_diff = 0.0
        counts_match = True
        if same_n:
            for a, b in zip(peaks_small, peaks_large):
                a = np.asarray(a, np.float32).reshape(-1, 3)
                b = np.asarray(b, np.float32).reshape(-1, 3)
                if len(a) != len(b):
                    counts_match = False
                    continue
                if len(a):
                    # Sort both by (y, x) so sub-batch boundary reordering
                    # (there shouldn't be any) can't spuriously fail the compare.
                    ao = a[np.lexsort((a[:, 1], a[:, 0]))]
                    bo = b[np.lexsort((b[:, 1], b[:, 0]))]
                    max_diff = max(max_diff, float(np.abs(ao - bo).max()))
        out["counts_match"] = counts_match
        out["max_pos_diff"] = max_diff

        print("RESULT_JSON", json.dumps(out))
        sys.stdout.flush()
        os._exit(0)
""")


def _run(mode, timeout=300):
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER, mode],
        capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, (
        f"subprocess failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON"))
    return json.loads(line[len("RESULT_JSON "):])


class TestNeuralBatchMemory:
    def test_peak_memory_scales_with_batch_not_stack_size(self):
        cuda = _run("check_cuda")
        if not cuda["cuda"]:
            import pytest
            pytest.skip("no CUDA device available in this environment")

        out = _run("scaling")
        assert out["peak_small_bytes"] > 0
        assert out["peak_large_bytes"] > 0
        # K=32 must use substantially less peak CUDA memory than K=N=256.
        assert out["ratio"] > 3.0, out

        # Sub-batching must not change WHAT is detected: scale/bg_sigma/NMS
        # window are resolved once from frame[0], outside the sub-batch loop.
        assert out["same_frame_count"], out
        assert out["counts_match"], out
        assert out["max_pos_diff"] <= 1e-3, out
