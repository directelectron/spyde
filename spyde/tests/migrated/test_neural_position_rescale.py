"""The neural detector reports a disk at its true position in the ORIGINAL frame.

Frames are rescaled so disks reach the model's canonical size; positions found
in that working frame must be mapped back with the inverse of the same
endpoint-aligned rescale. The model is replaced by a stub that finds each disk
in the working frame by its centroid, so any error left is the mapping's.
``p / factor`` put disks across a 128 px frame 0.6-0.7 px off; the exact
inverse is within the stub's own accuracy everywhere.
"""
import math

import numpy as np
import pytest
import torch
from scipy.special import erfc

from spyde.models import infer, preprocess

FRAME = 128
DIAMETER = 16.0          # factor 9/16 -> a 72 px working frame (127/71 != 16/9)
TRUTH = [(9.4, 118.6), (118.3, 8.7), (64.2, 63.6)]


def _frame():
    """Three flat-topped disks with continuous (subpixel) centres."""
    supersample = 5
    sub = (np.arange(FRAME * supersample) + 0.5) / supersample - 0.5
    y, x = np.meshgrid(sub, sub, indexing="ij")
    image = np.zeros_like(y)
    for cy, cx in TRUTH:
        image += 0.5 * erfc((np.hypot(y - cy, x - cx) - DIAMETER / 2) / (math.sqrt(2) * 0.5))
    pixels = image.reshape(FRAME, supersample, FRAME, supersample).mean((1, 3))
    return (pixels * 100 + 2).astype(np.float32)


class _CentroidModel(torch.nn.Module):
    """Heatmap + offset 'model' that places each disk at the centroid of its
    blob in the rescaled RAW frame (the normalised input the real model sees is
    background-subtracted, which would bias a centroid near the frame edge).
    The rescale is the detector's own, so the only thing under test is the way
    working-frame positions are mapped back."""
    levels = 2

    def __init__(self, frame):
        super().__init__()
        self.working, _ = preprocess.scale_to_canonical(frame, diameter=DIAMETER)

    def forward(self, x):
        from scipy.ndimage import center_of_mass, label
        heatmap = torch.full_like(x, -20.0)
        offset = torch.zeros(x.shape[0], 2, *x.shape[2:])
        signal = np.clip(self.working - 0.5 * self.working.max(), 0, None)
        blobs, count = label(signal > 0)
        for index in range(1, count + 1):
            cy, cx = center_of_mass(signal * (blobs == index))
            iy, ix = int(round(cy)), int(round(cx))
            heatmap[:, 0, iy, ix] = 20.0
            offset[:, 0, iy, ix] = cy - iy
            offset[:, 1, iy, ix] = cx - ix
        return heatmap, offset


def _errors(peaks):
    return np.array([np.hypot(peaks[:, 0] - cy, peaks[:, 1] - cx).min() for cy, cx in TRUTH])


class TestNeuralPositionRescale:
    def test_detect_maps_back_to_frame_coordinates(self):
        peaks = infer.detect(_CentroidModel(_frame()), _frame(), torch.device("cpu"),
                             spot_diameter=DIAMETER)
        errors = _errors(peaks)
        assert errors.max() < 0.1, f"position errors {np.round(errors, 3)} px"

    def test_detect_batch_maps_back_to_frame_coordinates(self, monkeypatch):
        monkeypatch.setenv("SPYDE_NEURAL_GPU_PREP", "0")
        frames = np.stack([_frame(), _frame()])
        for peaks in infer.detect_batch(_CentroidModel(frames[0]), frames, torch.device("cpu"),
                                        spot_diameter=DIAMETER):
            errors = _errors(peaks)
            assert errors.max() < 0.1, f"position errors {np.round(errors, 3)} px"

    @pytest.mark.parametrize("size,factor", [(128, 0.5625), (112, 1.5), (256, 0.45)])
    def test_inverse_matches_the_rescale(self, size, factor):
        """A pixel-centre position survives rescale -> inverse exactly."""
        out = int(round(size * factor))
        working = np.array([[0.0, 0.0, 1.0], [out - 1.0, out - 1.0, 1.0]])
        frame = infer._to_frame_coordinates(working, (size, size), factor)
        np.testing.assert_allclose(frame[:, :2], [[0, 0], [size - 1, size - 1]])
