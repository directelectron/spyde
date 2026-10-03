"""NXCORR recovers the subpixel centre of a flat-topped disk on every path.

The statistics window is a pixel larger than the disk template, so near a disk
centre the score exceeds 1. Clipping it to [-1, 1] before locating the peak
turned the top into a flat plateau and the parabolic vertex snapped to an
integer pixel (0.85 px RMS for a 6 px disk). The disks here are noise-free,
rendered exactly at known subpixel centres; every path must recover them to a
few hundredths of a pixel.
"""
import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.special import erfc

from spyde.actions.find_vectors.detectors import _find_vectors_single_frame, _make_disk

FRAME = 64
SHIFTS = np.linspace(-0.5, 0.5, 9)
TOLERANCE_RMS = 0.08
TOLERANCE_MAX = 0.12


def _disk_frame(centre_y, centre_x, radius, supersample=5):
    """One flat-topped disk (erf edge, 0.5 px wide, 0.6 px detector blur) whose
    centre is a continuous parameter, integrated over each pixel."""
    sub = (np.arange(FRAME * supersample) + 0.5) / supersample - 0.5
    y, x = np.meshgrid(sub, sub, indexing="ij")
    disk = 0.5 * erfc((np.hypot(y - centre_y, x - centre_x) - radius) / (math.sqrt(2) * 0.5))
    pixels = disk.reshape(FRAME, supersample, FRAME, supersample).mean((1, 3))
    return (gaussian_filter(pixels, 0.6) * 100 + 2).astype(np.float32)


def _truth_and_frames(radius):
    truth = np.array([(31.3 + shift, 32.0 + 0.37 * shift) for shift in SHIFTS])
    frames = np.stack([_disk_frame(cy, cx, radius) for cy, cx in truth])
    return truth, frames


def _errors(peaks_per_frame, truth):
    errors = []
    for peaks, (cy, cx) in zip(peaks_per_frame, truth):
        assert len(peaks), "disk not detected"
        distance = np.hypot(peaks[:, 0] - cy, peaks[:, 1] - cx)
        errors.append(distance.min())
    return np.array(errors)


def _assert_subpixel(errors, path, radius):
    rms = float(np.sqrt(np.mean(errors ** 2)))
    assert rms < TOLERANCE_RMS and errors.max() < TOLERANCE_MAX, (
        f"{path} R={radius}: RMS {rms:.3f} px, max {errors.max():.3f} px")


def _disk_stats(radius):
    disk = _make_disk(radius)
    n = disk.size
    mean = float(disk.mean())
    return n, mean, float(np.sqrt(np.sum((disk - mean) ** 2) / n))


@pytest.mark.parametrize("radius", [3, 6, 10])
class TestNxcorrSubpixelDisks:
    def test_cpu(self, radius):
        truth, frames = _truth_and_frames(radius)
        peaks = [_find_vectors_single_frame(f, radius, 0.5, max(1, radius // 2))[2] for f in frames]
        _assert_subpixel(_errors(peaks, truth), "cpu", radius)

    def test_torch_on_cpu(self, radius):
        torch = pytest.importorskip("torch")
        from spyde.actions.find_vectors_torch import find_vectors_torch_batch
        truth, frames = _truth_and_frames(radius)
        peaks = find_vectors_torch_batch(frames, radius, 0.5, max(1, radius // 2),
                                         device=torch.device("cpu"))
        _assert_subpixel(_errors(peaks, truth), "torch", radius)

    def test_gpu_chunk(self, radius):
        """numba.cuda kernels (or CuPy when installed); skipped without a CUDA device."""
        numba = pytest.importorskip("numba")
        from numba import cuda
        if not cuda.is_available():
            pytest.skip("no CUDA device")
        from spyde.actions.find_vectors.chunk import _find_vectors_chunk_gpu
        from spyde.actions.find_vectors.kernels import _GPU_KERNELS_AVAILABLE
        if not _GPU_KERNELS_AVAILABLE:
            pytest.skip("numba CUDA kernels unavailable")
        truth, frames = _truth_and_frames(radius)
        block = frames.reshape(3, 3, FRAME, FRAME)
        out = _find_vectors_chunk_gpu(block, 0, 2, 0.0, radius, 0.5, max(1, radius // 2),
                                      True, None, _disk_stats(radius))
        flat = out.reshape(9, out.shape[2], 3)
        peaks = [p[np.isfinite(p[:, 0])] for p in flat]
        _assert_subpixel(_errors(peaks, truth), "gpu", radius)
