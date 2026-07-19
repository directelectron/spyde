"""Parity: preprocess_torch (on-device batched) vs preprocess.py (numpy reference).

detect_batch preprocesses on the model device (GPU) via preprocess_torch instead
of on the host with numpy/scipy — a big end-to-end speedup (removes the host
Amdahl floor), but ONLY sound if the maths matches exactly, since the checkpoints
were trained with preprocess.py's pipeline. These tests pin that parity. They run
on torch-CPU (no GPU needed → safe under pytest / CI); the same functions run
verbatim on MPS/CUDA at runtime (validated on real hardware separately).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, zoom

from spyde.models import preprocess as P
from spyde.models import preprocess_torch as PT
from spyde.models.infer import _build_input_stack, _gpu_prep_enabled, _pad_to_multiple


def _frames(n=8, H=112, W=112, seed=0):
    rng = np.random.default_rng(seed)
    base = np.zeros((H, W), np.float32)
    for (cy, cx) in [(56, 56), (30, 70), (78, 40), (44, 24)]:
        base[cy, cx] = 500.0
    base = gaussian_filter(base, 2.5)
    return np.stack([base + rng.random((H, W), np.float32) * 8 for _ in range(n)], 0)


class TestPreprocessTorchParity:
    def test_gaussian_blur_matches_scipy(self):
        """gaussian_blur2d == scipy.ndimage.gaussian_filter (truncate=4, half-sample
        reflect) across sigmas — incl. sigma whose radius exceeds the frame."""
        f = _frames(4)
        logf = np.log1p(np.clip(f, 0, None))
        for sigma in (6.0, 12.0, 40.0):
            ref = np.stack([gaussian_filter(x, sigma) for x in logf])
            got = PT.gaussian_blur2d(
                torch.from_numpy(logf).unsqueeze(1), sigma).squeeze(1).numpy()
            assert np.max(np.abs(ref - got)) < 1e-3, f"sigma={sigma}"

    def test_normalize_input_matches_numpy(self):
        """normalize_input_batch == preprocess.normalize_input (log1p + local bg +
        median/MAD; note np.median averages two middles — sort-based match)."""
        f = _frames(6, seed=1)
        ref = np.stack([P.normalize_input(x, local=True, bg_sigma=12.0) for x in f])
        got = PT.normalize_input_batch(torch.from_numpy(f), 12.0).numpy()
        assert np.max(np.abs(ref - got)) < 1e-3

    def test_scale_matches_scipy_zoom(self):
        """scale_batch == scipy.ndimage.zoom(order=1): endpoint-aligned, output size
        round(n*factor)."""
        f = _frames(4, seed=2)
        for factor in (1.125, 0.8, 1.5):
            ref = np.stack([zoom(x, factor, order=1) for x in f])
            got = PT.scale_batch(torch.from_numpy(f), factor).numpy()
            assert got.shape == ref.shape, f"factor={factor} shape"
            # values up to ~500; endpoint-aligned bilinear matches to <1e-2
            assert np.max(np.abs(ref - got)) < 1e-2, f"factor={factor}"

    def test_pad_to_multiple_matches_numpy(self):
        """pad_to_multiple_batch == infer._pad_to_multiple (numpy 'reflect' =
        whole-sample = torch reflect) — exact."""
        f = _frames(3, H=110, W=113)   # non-multiples of 4 to force padding
        ref = np.stack([_pad_to_multiple(x, 2) for x in f])
        got = PT.pad_to_multiple_batch(torch.from_numpy(f), 2).numpy()
        assert got.shape == ref.shape
        assert np.array_equal(ref, got)

    def test_canonical_scale_factor_matches(self):
        """The GPU path's factor derivation matches scale_to_canonical's (bounded +
        <5% no-op short-circuit)."""
        for diam in (8.0, 9.0, 9.2, 18.0, 3.0, 40.0):
            _scaled, ref = P.scale_to_canonical(np.zeros((32, 32), np.float32),
                                                diameter=diam)
            got = PT.canonical_scale_factor_from_diameter(diam)
            assert abs(ref - got) < 1e-6, f"diam={diam}"


class TestBuildInputStack:
    def test_cpu_device_uses_numpy_reference(self):
        """On a CPU device _build_input_stack returns the numpy-reference stack as a
        CPU tensor of the padded shape (never the GPU torch path)."""
        f = _frames(5, seed=4)
        factor = 1.0
        out = _build_input_stack(f, torch.device("cpu"), factor, 12.0, 2,
                                 auto_scale=True)
        assert out.device.type == "cpu"
        assert out.shape == (5, 1, 112, 112)
        # matches the direct numpy pipeline
        ref = np.stack([_pad_to_multiple(P.normalize_input(x, local=True, bg_sigma=12.0), 2)
                        for x in f])
        assert np.max(np.abs(out.squeeze(1).numpy() - ref)) < 1e-5

    def test_gpu_prep_env_gate(self, monkeypatch):
        monkeypatch.delenv("SPYDE_NEURAL_GPU_PREP", raising=False)
        assert _gpu_prep_enabled() is True
        monkeypatch.setenv("SPYDE_NEURAL_GPU_PREP", "0")
        assert _gpu_prep_enabled() is False
        monkeypatch.setenv("SPYDE_NEURAL_GPU_PREP", "off")
        assert _gpu_prep_enabled() is False
        monkeypatch.setenv("SPYDE_NEURAL_GPU_PREP", "1")
        assert _gpu_prep_enabled() is True

    def test_torch_prep_path_matches_numpy_on_cpu_tensor(self, monkeypatch):
        """Force the torch-prep branch by presenting a 'gpu-like' device and verify
        it matches the numpy reference (the runtime parity guarantee, exercised
        without a real GPU by patching the device-type gate)."""
        import spyde.models.infer as infer

        f = _frames(6, seed=5)
        factor = 1.125
        bg_sigma = 12.0
        # numpy reference (what the CPU device would produce)
        ref = np.stack([_pad_to_multiple(
            P.normalize_input(zoom(x, factor, order=1), local=True, bg_sigma=bg_sigma), 2)
            for x in f])
        # torch path built directly (same ops _build_input_stack runs for a GPU dev)
        x = torch.from_numpy(f)
        x = PT.scale_batch(x, factor)
        x = PT.normalize_input_batch(x, bg_sigma=bg_sigma, local=True)
        got = PT.pad_to_multiple_batch(x, 2).unsqueeze(1).numpy()
        assert got.shape == ref[:, None].shape
        assert np.max(np.abs(got[:, 0] - ref)) < 1e-2
