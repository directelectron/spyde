"""The neural batch moves each chunk's frames to the GPU once, into reused buffers.

``_neural_block`` uploads a chunk's frames into buffers each worker keeps
(``spyde.models.device_frames``), and the detector and both centre steps read
them from there. These tests pin the two things that could go wrong:

- the vectors are the same as the path that casts on the host and uploads the
  frames again for each centre step (bit for bit: the device cast of an integer
  or float32 frame is exact, and every later operation sees the same values in
  the same order);
- after the first chunk, a chunk of the same shape allocates no device memory
  from the driver.

Run in a SUBPROCESS that prints a JSON result and ``os._exit(0)`` after, as in
``test_neural_detect.py``: torch-CUDA work can segfault inside the pytest
process on Windows. Skips without CUDA.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")

_DRIVER = textwrap.dedent(r"""
    import json, os, sys
    import numpy as np
    import torch
    import torch.nn.functional as F

    # Prime cublasLt before any convolution (Pascal late-init failure).
    F.linear(torch.zeros(1, 1, device="cuda"), torch.zeros(1, 1, device="cuda"))
    torch.cuda.synchronize()

    import pyxem.data as pxd
    from spyde import models
    import spyde.actions.find_vectors_neural as fvn

    data = np.asarray(pxd.si_grains().data)
    data = np.clip(data, 0, None)
    # A Find Vectors run's parameters: refine and Friedel on, a spot size both
    # centre steps accept.
    spot_radius = 7.0
    kwargs = dict(threshold=0.3, min_dist=5, subpixel=True, beamstop_mask=None,
                  model_id=None, bg_sigma=12.0, persistence=False,
                  spot_radius=spot_radius, refine_centres=True, friedel_partner=True)

    def reference(block):
        # Cast on the host, upload inside detect_batch, upload again per step.
        model, device = models.get_model(None)
        flat = block.reshape(-1, *block.shape[2:]).astype(np.float32)
        raw = models.detect_batch(model, flat, device, thresh=0.3, min_distance=5,
                                  bg_sigma=12.0, spot_diameter=2 * spot_radius,
                                  with_width=True)
        counts_radius = fvn._counts_radius(flat[0], spot_radius)
        preds = [np.asarray(p, np.float32).reshape(-1, 4) for p in raw]
        preds, sigmas = fvn._refined(flat, preds, counts_radius, True, True, device)
        return [fvn._neural_peaks(flat[i], p, counts_radius, sigmas[i])
                for i, p in enumerate(preds)]

    def as_lists(out, nx):
        lists = []
        for i in range(out.shape[0] * out.shape[1]):
            rows = out[i // nx, i % nx]
            lists.append(rows[~np.isnan(rows[:, 0])])
        return lists

    mode = sys.argv[1]
    result = {}
    if mode == "same_vectors":
        for name, block in (("float32", data.astype(np.float32)),
                            ("uint16", np.rint(data * 1000).astype(np.uint16))):
            expected = reference(block)
            got = as_lists(fvn._neural_block(block, **kwargs), block.shape[1])
            count = sum(len(e) for e in expected)
            identical = all(e.shape == g.shape and np.array_equal(e, g, equal_nan=True)
                            for e, g in zip(expected, got))
            refined = sum(int(np.isfinite(e[:, 4]).sum()) for e in expected)
            result[name] = dict(vectors=count, refined=refined, identical=identical)

    elif mode == "upload":
        from spyde.models.device_frames import FrameBuffers
        buffers = FrameBuffers()
        device = torch.device("cuda")
        rng = np.random.default_rng(0)
        exact = {}
        for dtype in (np.uint8, np.uint16, np.uint32, np.int32, np.float32, np.float64):
            frames = (rng.random((5, 32, 48)) * 250).astype(dtype)
            got = buffers.upload(frames, device).cpu().numpy()
            exact[np.dtype(dtype).name] = bool(np.array_equal(got, frames.astype(np.float32)))
        first = buffers.upload(np.zeros((8, 32, 48), np.uint16), device)
        smaller = buffers.upload(np.ones((3, 32, 48), np.uint16), device)
        # A non-contiguous view, as a sliced chunk can arrive.
        strided = rng.integers(0, 4000, (12, 32, 48)).astype(np.uint16)[::2]
        got = buffers.upload(strided, device).cpu().numpy()
        result = dict(exact=exact,
                      reused=first.data_ptr() == smaller.data_ptr(),
                      shape=list(smaller.shape),
                      strided=bool(np.array_equal(got, strided)))

    elif mode == "allocations":
        block = np.rint(np.tile(data, (3, 3, 1, 1)) * 1000).astype(np.uint16)
        device_allocations, pool_allocations = [], []
        for _ in range(4):
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            fvn._neural_block(block, **kwargs)
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            device_allocations.append(after["num_device_alloc"] - before["num_device_alloc"])
            pool_allocations.append(after["allocation.all.allocated"]
                                    - before["allocation.all.allocated"])
        result = dict(device_allocations=device_allocations,
                      pool_allocations=pool_allocations)

    print("RESULT_JSON", json.dumps(result))
    sys.stdout.flush()
    os._exit(0)
""")


def _run(mode, timeout=600):
    proc = subprocess.run([sys.executable, "-c", _DRIVER, mode],
                          capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, (
        f"subprocess failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON"))
    return json.loads(line[len("RESULT_JSON "):])


class TestNeuralBatchBuffers:
    def test_upload_casts_exactly_and_reuses_the_buffer(self):
        out = _run("upload")
        assert all(out["exact"].values()), out
        assert out["reused"] and out["shape"] == [3, 32, 48], out
        assert out["strided"], out

    def test_vectors_match_the_uploading_path(self):
        out = _run("same_vectors")
        for name, row in out.items():
            assert row["vectors"] > 0, name
            # The centre steps must have run, or the reused frames were never read.
            assert row["refined"] > 0, name
            assert row["identical"], (name, row)

    def test_later_chunks_allocate_no_device_memory(self):
        out = _run("allocations")
        assert out["device_allocations"][0] >= 0
        assert out["device_allocations"][1:] == [0, 0, 0], out
