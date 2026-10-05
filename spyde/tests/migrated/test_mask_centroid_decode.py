"""The pieces a mask-head SpotUNet adds to inference, on torch-CPU (no model weights).

- the floor on the normalisation scale: ``None`` leaves the original maths untouched,
  and the numpy and on-device paths agree with a floor too;
- the mask-centroid decode returns the geometric centre of the predicted coverage,
  wherever the intensity sits inside the disk, and falls back to the offset head when
  the mask has nothing at a peak;
- a SpotUNet without the head keeps its two-output contract and its state-dict keys.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from spyde.models import decode as D
from spyde.models import preprocess as P
from spyde.models import preprocess_torch as PT
from spyde.models.unet import SpotUNet


def _sparse_counts(seed=0, shape=(192, 192)):
    """A counted frame where most pixels are zero: a few disks of ~20 counts."""
    rng = np.random.default_rng(seed)
    lam = np.full(shape, 0.0005, np.float32)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    for cy, cx in ((30, 30), (30, 66), (66, 48)):
        lam[(yy - cy) ** 2 + (xx - cx) ** 2 <= 16] += 0.4
    return rng.poisson(lam).astype(np.float32)


def _coverage_logits(shape, centre, radius):
    """Logits of an anti-aliased disk coverage, i.e. what a trained mask head emits."""
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
    coverage = np.clip(radius - np.hypot(yy - centre[0], xx - centre[1]) + 0.5, 1e-4, 1 - 1e-4)
    return torch.from_numpy(np.log(coverage / (1 - coverage)))[None, None]


class TestNormalisationFloor:
    def test_no_floor_is_the_original_maths(self):
        frame = gaussian_filter(np.random.default_rng(1).random((64, 64)).astype(np.float32) * 50, 1.0)
        x = np.log1p(frame)
        x = x - gaussian_filter(x, 12.0)
        med = np.median(x)
        original = (x - med) / (1.4826 * (np.median(np.abs(x - med)) + 1e-6))
        assert np.allclose(P.normalize_input(frame, bg_sigma=12.0), original, atol=1e-6)

    def test_floor_bounds_sparse_frames(self):
        frame = _sparse_counts()
        assert np.abs(P.normalize_input(frame, bg_sigma=12.0)).max() > 1e3
        assert np.abs(P.normalize_input(frame, bg_sigma=12.0, mad_floor=0.05)).max() < 200

    def test_floor_numpy_matches_torch(self):
        frames = np.stack([_sparse_counts(seed) for seed in range(4)])
        reference = np.stack([P.normalize_input(f, bg_sigma=12.0, mad_floor=0.05) for f in frames])
        on_device = PT.normalize_input_batch(torch.from_numpy(frames), 12.0, mad_floor=0.05).numpy()
        assert np.max(np.abs(reference - on_device)) < 1e-2


class TestMaskCentroidDecode:
    def test_centroid_is_the_disk_centre(self):
        shape = (64, 64)
        for centre in ((31.3, 28.7), (20.55, 40.2), (44.0, 44.9)):
            logits = _coverage_logits(shape, centre, 4.5)
            seed_y = torch.tensor([round(centre[0]) + 0.8])
            seed_x = torch.tensor([round(centre[1]) - 0.6])
            y, x, area = D.mask_centroid(logits, torch.tensor([0]), seed_y, seed_x, 4.5)
            assert abs(float(y) - centre[0]) < 0.02 and abs(float(x) - centre[1]) < 0.02
            assert abs(float(area) - np.pi * 4.5 ** 2) < 3.0

    def test_decode_uses_mask_and_falls_back_without_one(self):
        shape = (64, 64)
        centre = (30.4, 33.7)
        heatmap = torch.full((1, 1) + shape, -8.0)
        heatmap[0, 0, 30, 34] = 4.0                     # the peak pixel
        heatmap[0, 0, 12, 12] = 4.0                     # a peak with no disk under it
        offsets = torch.zeros((1, 2) + shape)
        mask = _coverage_logits(shape, centre, 4.5)
        mask[0, 0, :20, :20] = -10.0
        peaks = D.decode_batch(heatmap, offsets, thresh=0.3, min_distance=2, mask_logits=mask, radius=4.5)
        peaks = peaks[np.argsort(peaks[:, 1])]
        assert np.allclose(peaks[0, 1:3], (12.0, 12.0))                  # offset head kept
        assert np.allclose(peaks[1, 1:3], centre, atol=0.02)             # mask centroid
        single = D.decode(heatmap[0], offsets[0], thresh=0.3, min_distance=2, mask_logits=mask[0], radius=4.5)
        single = single[np.argsort(single[:, 0])]
        assert np.allclose(single[:, :2], peaks[:, 1:3], atol=1e-4)


class TestSpotUNetHeads:
    def test_without_mask_head_contract_is_unchanged(self):
        model = SpotUNet(base=8, levels=3).eval()
        assert not any(k.startswith('head_mask') for k in model.state_dict())
        with torch.no_grad():
            outputs = model(torch.zeros(1, 1, 32, 32))
        assert len(outputs) == 2

    def test_mask_head_adds_one_output(self):
        model = SpotUNet(base=8, levels=3, mask_head=True).eval()
        with torch.no_grad():
            heatmap, offsets, mask = model(torch.zeros(2, 1, 32, 32))
        assert mask.shape == heatmap.shape == (2, 1, 32, 32)
        assert offsets.shape == (2, 2, 32, 32)


class TestBackToFrame:
    def test_endpoint_aligned_zoom_maps_back_exactly(self):
        """A point drawn in the frame lands where the zoomed image puts it, mapped back
        through the zoom's own geometry, not through 1 / factor."""
        from spyde.models.infer import _to_frame
        shape, factor = (512, 512), 9 / 22
        frame = np.zeros(shape, np.float32)
        frame[440:461, 50:71] = 1.0                      # a block centred on (450, 60)
        zoomed = PT.scale_batch(torch.from_numpy(frame)[None], factor)[0].numpy()
        rows, columns = np.mgrid[:zoomed.shape[0], :zoomed.shape[1]]
        y, x = (zoomed * rows).sum() / zoomed.sum(), (zoomed * columns).sum() / zoomed.sum()
        back = _to_frame(np.array([[y, x, 1.0]], np.float32), factor, shape)
        assert abs(back[0, 0] - 450) < 0.2 and abs(back[0, 1] - 60) < 0.2
        naive = np.array([y, x]) / factor
        assert abs(naive[0] - 450) > 2.0                 # the old mapping is off by ~2 px here
