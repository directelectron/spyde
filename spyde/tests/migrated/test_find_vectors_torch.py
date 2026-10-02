"""
Torch (MPS/CUDA) batched find-vectors path — the whole pipeline (NXCORR +
local-max + 2D-quadratic subpixel FIT) runs on the GPU. Skipped when no torch GPU
is present (CI / CPU-only boxes use the numpy path).

This path is deliberately a touch MORE correct than the numpy reference, not
bit-identical: it uses the true linear cross-correlation (the reference's
circular-FFT correlation has a boundary/normalisation quirk) and fits the
correlation peak with a 2D quadratic least-squares surface (the surface vertex is
the subpixel maximum) instead of a center-of-mass — far less biased. So we test
ACCURACY against known subpixel centres, not equality with the CPU CoM.

torch FORWARD ops (conv/fft, no autograd) are safe under pytest on MPS.
"""
from __future__ import annotations

import numpy as np
import pytest

torch_gpu = None
try:
    from spyde.actions.find_vectors_torch import (
        torch_gpu_device, find_vectors_torch_batch)
    torch_gpu = torch_gpu_device()
except Exception:
    torch_gpu = None

pytestmark = pytest.mark.skipif(torch_gpu is None, reason="no torch GPU (MPS/CUDA)")


def _gauss_frame(centers, H=112, W=112, sigma=2.2, amp=100.0, seed=3):
    yy, xx = np.mgrid[0:H, 0:W]
    f = np.zeros((H, W), np.float32)
    for (cy, cx) in centers:
        f += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))).astype(np.float32) * amp
    f += np.random.RandomState(seed).rand(H, W).astype(np.float32) * 1.0
    return f


class TestFindVectorsTorch:
    def test_quadratic_fit_recovers_subpixel_centers(self):
        """The 2D-quadratic fit recovers KNOWN subpixel disk centres to <0.1 px —
        and beats the numpy CoM, which is biased to ~0.8 px on the same data."""
        from spyde.actions.find_vectors import _find_vectors_single_frame
        truth = [(56.3, 55.7), (40.8, 72.2), (72.1, 40.6)]
        f = _gauss_frame(truth)
        tr = find_vectors_torch_batch(f[None], 5, 0.4, 6, subpixel=True)[0]
        cp = _find_vectors_single_frame(f, 5, 0.4, 6, subpixel=True)[2]

        def err(p):
            return np.array([np.sqrt((p[:, 0] - cy) ** 2 + (p[:, 1] - cx) ** 2).min()
                             for (cy, cx) in truth])
        torch_err = float(err(tr).mean())
        com_err = float(err(cp).mean())
        assert torch_err < 0.1, f"quad-fit error {torch_err:.3f} px"
        assert torch_err < com_err            # the fit is more accurate than CoM

    def test_finds_all_disks_batched(self):
        """Batched over many frames: every true disk is found in every frame."""
        truth = [(56, 56), (30, 80), (80, 30), (40, 70)]
        frames = np.stack([_gauss_frame(truth, sigma=2.0, seed=s) for s in range(12)])
        out = find_vectors_torch_batch(frames, 5, 0.4, 6, subpixel=True)
        assert len(out) == 12
        for i in range(12):
            assert len(out[i]) >= len(truth)
            for (cy, cx) in truth:
                d = np.sqrt((out[i][:, 0] - cy) ** 2 + (out[i][:, 1] - cx) ** 2)
                assert d.min() < 0.6, f"frame {i} missing disk {(cy, cx)}"

    def test_subpixel_off_returns_integer_peaks(self):
        truth = [(56, 56), (40, 72)]
        f = _gauss_frame(truth, sigma=2.0)
        out = find_vectors_torch_batch(f[None], 5, 0.4, 6, subpixel=False)[0]
        assert len(out) >= 2
        assert np.allclose(out[:, :2], np.round(out[:, :2]))   # integer coords
