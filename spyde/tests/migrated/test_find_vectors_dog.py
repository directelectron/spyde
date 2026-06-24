"""
Tests for the Difference-of-Gaussians (DoG) find-vectors method.

Covers:
  * single-frame DoG correctness (finds known Gaussian spots, sub-pixel, raw
    intensity column, beam-stop exclusion + background fill);
  * the method dispatcher (_find_peaks_single_frame routes nxcorr vs dog);
  * end-to-end batch via _do_compute_vectors on numpy and lazy dask data;
  * memory safety: the DoG batch + beamstop_auto sampling never compute the full
    lazy dataset;
  * auto beam-stop detection (detect_beamstop / _auto_beamstop_from_signal).

GPU note: torch-CUDA segfaults under pytest on Windows, so these exercise the
numpy DoG path (the GPU path is validated for parity separately in
benchmark_find_vectors_dog.py, which runs in a subprocess).
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import dask.array as da
import hyperspy.api as hs
import pytest

from spyde.actions.find_vectors import (
    _find_vectors_single_frame_dog, _find_peaks_single_frame,
    _do_compute_vectors, detect_beamstop, _auto_beamstop_from_signal,
    METHOD_DOG, METHOD_NXCORR,
)


def _spotty_frame(H=64, W=64, spots=((20, 20), (44, 44), (20, 44), (44, 20),
                                     (32, 33)), amp=300.0, sig=1.2, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.normal(50.0, 3.0, (H, W)).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for (cy, cx) in spots:
        f += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig * sig))
    return f, spots


class TestDoGSingleFrame:
    def test_finds_all_spots(self):
        f, spots = _spotty_frame()
        _, _, peaks = _find_vectors_single_frame_dog(f, 0.8, 2.0, 8.0, 3)
        assert len(peaks) == len(spots)
        found = {(round(p[0]), round(p[1])) for p in peaks}
        for (cy, cx) in spots:
            assert (cy, cx) in found

    def test_intensity_column_is_raw(self):
        f, _ = _spotty_frame()
        _, _, peaks = _find_vectors_single_frame_dog(f, 0.8, 2.0, 8.0, 3)
        # value column = raw frame intensity (~background+amp), NOT the SNR
        assert peaks[:, 2].min() > 100.0

    def test_subpixel_shifts_off_integer(self):
        f, _ = _spotty_frame(spots=((32, 32),))
        # shift the single spot by a known sub-pixel amount
        yy, xx = np.mgrid[0:64, 0:64]
        f = (50 + 300 * np.exp(-((yy - 32.4) ** 2 + (xx - 31.7) ** 2) / (2 * 1.2 ** 2))
             ).astype(np.float32)
        _, _, peaks = _find_vectors_single_frame_dog(f, 0.8, 2.0, 6.0, 3, subpixel=True)
        assert len(peaks) == 1
        assert abs(peaks[0, 0] - 32.4) < 0.4
        assert abs(peaks[0, 1] - 31.7) < 0.4

    def test_beamstop_excludes_and_no_rim_peak(self):
        # a bright stripe (beam-stop rim surrogate) plus a real spot far away,
        # on a realistic noisy background (real detectors always have shot noise)
        rng = np.random.default_rng(3)
        f = rng.normal(50.0, 3.0, (64, 64)).astype(np.float32)
        f[:, 30:34] = 5000.0                 # bright vertical bar
        yy, xx = np.mgrid[0:64, 0:64]
        f += 300 * np.exp(-((yy - 12) ** 2 + (xx - 12) ** 2) / (2 * 1.2 ** 2))
        mask = np.zeros((64, 64), bool)
        mask[:, 30:34] = True
        _, _, peaks = _find_vectors_single_frame_dog(
            f, 0.8, 2.0, 6.0, 3, beamstop_mask=mask)
        # no peak inside the (background-filled) mask
        if len(peaks):
            iy = np.clip(peaks[:, 0].round().astype(int), 0, 63)
            ix = np.clip(peaks[:, 1].round().astype(int), 0, 63)
            assert not mask[iy, ix].any()
        # the real spot is still found
        found = {(round(p[0]), round(p[1])) for p in peaks}
        assert (12, 12) in found


class TestDispatcher:
    def test_dog_route(self):
        f, spots = _spotty_frame()
        peaks = _find_peaks_single_frame(
            f, dict(method=METHOD_DOG, threshold=8.0, min_distance=3,
                    dog_sigma1=0.8, dog_sigma2=2.0))
        assert len(peaks) == len(spots)

    def test_nxcorr_route_still_works(self):
        f, _ = _spotty_frame()
        peaks = _find_peaks_single_frame(
            f, dict(method=METHOD_NXCORR, kernel_radius=2, threshold=0.4,
                    min_distance=3))
        assert len(peaks) >= 5


class TestDoGBatch:
    def _lazy_4d(self, nav=(8, 8), sig=(48, 48), chunk_nav=4):
        ny, nx = nav
        ky, kx = sig
        f, _ = _spotty_frame(ky, kx, spots=((12, 12), (36, 36), (12, 36),
                                            (36, 12)), seed=1)
        data = np.broadcast_to(f, (ny, nx, ky, kx)).copy()
        da_data = da.from_array(data, chunks=(chunk_nav, chunk_nav, ky, kx))
        return hs.signals.Signal2D(da_data)

    def _params(self):
        return dict(method=METHOD_DOG, sigma=0.0, kernel_radius=5,
                    threshold=8.0, min_distance=3, subpixel=True,
                    dog_sigma1=0.8, dog_sigma2=2.0)

    def test_numpy_batch(self):
        sig = self._lazy_4d()
        sig = hs.signals.Signal2D(np.asarray(sig.data))  # eager
        vecs = _do_compute_vectors(sig, self._params(), None, None)
        assert vecs.nav_shape == (8, 8)
        # 4 spots per pattern × 64 patterns
        assert len(vecs.flat_buffer) == 4 * 64

    def test_lazy_batch_memory_safe(self):
        sig = self._lazy_4d(sig=(128, 128), chunk_nav=4)
        full_shape = sig.data.shape
        orig = da.Array.compute
        full_hit = [False]

        def _spy(self, *a, **k):
            if self.shape == full_shape:
                full_hit[0] = True
                raise AssertionError("full-dataset compute in DoG path")
            return orig(self, *a, **k)

        with patch.object(da.Array, "compute", _spy):
            vecs = _do_compute_vectors(sig, self._params(), None, None)
        assert not full_hit[0]
        assert vecs.nav_shape == (8, 8)

    def test_beamstop_auto_memory_safe(self):
        sig = self._lazy_4d(sig=(128, 128), chunk_nav=4)
        full_shape = sig.data.shape
        orig = da.Array.compute
        full_hit = [False]

        def _spy(self, *a, **k):
            if self.shape == full_shape:
                full_hit[0] = True
                raise AssertionError("full compute during beamstop_auto")
            return orig(self, *a, **k)

        p = self._params()
        p["beamstop_auto"] = True
        with patch.object(da.Array, "compute", _spy):
            vecs = _do_compute_vectors(sig, p, None, None)
        assert not full_hit[0]


class TestBeamstopDetection:
    def test_detect_beamstop_lollipop(self):
        # synthetic mean pattern: bright field with a dark stop
        m = np.full((64, 64), 1000.0, np.float64)
        m[0:40, 30:34] = 1.0          # bar
        m[36:50, 26:38] = 1.0         # blob
        mask = detect_beamstop(m, frac=0.15, dilate=3)
        assert mask is not None
        assert mask[20, 31]           # on the bar
        assert mask.sum() > 0
        # dilation extends past the geometric stop
        assert mask.sum() > ((m < 150).sum())

    def test_detect_beamstop_none_when_featureless(self):
        m = np.full((64, 64), 1000.0, np.float64)
        assert detect_beamstop(m) is None

    def test_auto_from_signal_samples_sparsely(self):
        ny, nx, ky, kx = 10, 10, 48, 48
        f = np.full((ky, kx), 1000.0, np.float32)
        f[:30, 22:26] = 1.0           # dark bar in every pattern
        data = np.broadcast_to(f, (ny, nx, ky, kx)).copy()
        sig = hs.signals.Signal2D(da.from_array(data, chunks=(5, 5, ky, kx)))
        mask = _auto_beamstop_from_signal(sig, nav_dim=2, max_samples=20)
        assert mask is not None
        assert mask[10, 23]
