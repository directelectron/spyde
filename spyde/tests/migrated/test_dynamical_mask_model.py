"""The bundled mask-head detector (``spotunet-dynamical-mask8-v1``) end to end.

Runs in ONE subprocess that prints a ``RESULT_JSON`` line and ``os._exit(0)``s: torch
CUDA teardown can segfault inside the pytest process on Windows (see
test_neural_detect.py). The detections are compared between CPU and CUDA when a GPU
is present.

Covers:
  - the registry entry loads with its mask head, normalisation floor and threshold;
  - planted dynamical-looking disks (a bright crescent inside a sharp aperture) are
    found and centred on the aperture, not on the crescent;
  - ``detect`` and ``detect_batch`` agree, and CPU and CUDA agree;
  - a sparse counted frame (most pixels zero) gives a handful of detections, not
    one per count.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

_DRIVER = textwrap.dedent(r"""
    import json, os
    import numpy as np
    import torch
    if torch.cuda.is_available():
        import torch.nn.functional as F
        F.linear(torch.zeros(1, 1, device="cuda"), torch.zeros(1, 1, device="cuda"))
        torch.cuda.synchronize()
    from spyde.models import infer, registry

    MODEL_ID = "spotunet-dynamical-mask8-v1"

    def frame(centres, radius=11.0, shape=(256, 256), seed=0):
        # flat background, disks whose fill is a crescent pushed to one side
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
        lam = np.full(shape, 0.5, np.float32)
        for k, (cy, cx) in enumerate(centres):
            r = np.hypot(yy - cy, xx - cx)
            aperture = np.clip(radius - r + 0.5, 0, 1)
            angle = 0.9 * k
            ramp = 1 + 0.9 * (((xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)) / radius)
            lam += 60 * aperture * np.clip(ramp, 0.05, None)
        return rng.poisson(lam).astype(np.float32)

    centres = [(70.3, 64.6), (70.8, 128.2), (134.1, 96.5), (190.6, 160.4), (128.4, 190.9)]
    frames = np.stack([frame(centres, seed=s) for s in range(3)])
    out = {}
    entry = next(m for m in registry.list_models() if m["id"] == MODEL_ID)
    path = registry.ensure_local(MODEL_ID)
    cpu_model, _ = infer.load_model(path, device=torch.device("cpu"), arch=entry.get("arch"))
    out["mask_head"] = cpu_model.head_mask is not None
    out["mad_floor"] = cpu_model.mad_floor
    out["threshold"] = cpu_model.default_threshold
    kwargs = dict(thresh=0.3, min_distance=5, spot_diameter=22.0)
    single = infer.detect(cpu_model, frames[0], torch.device("cpu"), **kwargs)
    batch = infer.detect_batch(cpu_model, frames, torch.device("cpu"), **kwargs)

    def errors(peaks):
        d = [np.hypot(peaks[:, 0] - cy, peaks[:, 1] - cx).min() if len(peaks) else 1e9 for cy, cx in centres]
        return [float(v) for v in d]

    out["errors"] = errors(batch[0])
    out["count"] = int(len(batch[0]))
    a = single[np.lexsort(single[:, :2].T)]
    b = batch[0][np.lexsort(batch[0][:, :2].T)]
    out["single_batch_diff"] = float(np.abs(a[:, :2] - b[:, :2]).max()) if len(a) == len(b) else 1e9
    if torch.cuda.is_available():
        gpu_model, gpu = infer.load_model(path, device=torch.device("cuda"), arch=entry.get("arch"))
        on_gpu = infer.detect_batch(gpu_model, frames, gpu, **kwargs)
        c = on_gpu[0][np.lexsort(on_gpu[0][:, :2].T)]
        out["cpu_gpu_diff"] = float(np.abs(c[:, :2] - b[:, :2]).max()) if len(c) == len(b) else 1e9
    sparse = np.random.default_rng(5).poisson(np.full((256, 256), 0.01)).astype(np.float32)
    out["sparse_count"] = int(len(infer.detect_batch(cpu_model, sparse[None], torch.device("cpu"),
                                                     thresh=0.3, min_distance=5, spot_diameter=22.0)[0]))
    print("RESULT_JSON " + json.dumps(out), flush=True)
    os._exit(0)
""")


@pytest.fixture(scope="module")
def result():
    proc = subprocess.run([sys.executable, "-c", _DRIVER], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON "))
    return json.loads(line.split(" ", 1)[1])


class TestDynamicalMaskModel:
    def test_loads_with_its_operating_point(self, result):
        assert result["mask_head"]
        assert result["mad_floor"] == pytest.approx(0.05)
        assert result["threshold"] == pytest.approx(0.2)

    def test_centres_on_the_aperture(self, result):
        # each crescent's intensity centroid is ~2-3 px off its aperture centre
        assert max(result["errors"]) < 1.0, result
        assert result["count"] <= 8, result

    def test_single_matches_batch(self, result):
        assert result["single_batch_diff"] < 0.05, result

    def test_cpu_matches_cuda(self, result):
        if "cpu_gpu_diff" not in result:
            pytest.skip("no CUDA device")
        assert result["cpu_gpu_diff"] < 0.05, result

    def test_sparse_counts_do_not_flood(self, result):
        assert result["sparse_count"] < 10, result
