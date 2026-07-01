"""
fft_action.py — live FFT of a selected region, on the RegionAction template.

Place a rectangle on the source image; the linked output plot shows the
log-magnitude FFT of the selected sub-region, recomputed as the rectangle moves.
Host-agnostic (Electron + Jupyter).
"""
from __future__ import annotations

import numpy as np

from spyde.actions.action import RegionAction
from spyde.actions._common import widget_region


class FFTAction(RegionAction):
    """``image + rectangle ROI -> live FFT of the selected sub-region``."""

    name = "FFT"
    output_dims = 2
    output_node_name = "FFT"

    def selector_for_params(self, **params):
        from spyde.drawing.selectors import RectangleSelector
        return RectangleSelector

    def reduce(self, signal, selector, indices, **params):
        # Source image is whatever the parent plot currently displays.
        img = getattr(self.plot, "current_data", None)
        if not isinstance(img, np.ndarray) or img.ndim != 2:
            return None

        region = widget_region(selector, img)
        if region.size == 0 or min(region.shape) < 2:
            region = img

        fft = np.fft.fftshift(np.fft.fft2(region.astype(np.float64)))
        return np.log1p(np.abs(fft)).astype(np.float32)
