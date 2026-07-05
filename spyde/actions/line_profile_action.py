"""
line_profile_action.py — integrated line profile on the RegionAction template.

The ROI is a :class:`~spyde.drawing.selectors.LineProfileSelector`: a SOLID
line (the profile path) between two draggable endpoint handles, plus a DASHED
perpendicular line whose length is the integration WIDTH. The profile samples
the image along the solid line (bilinear) and averages across the width band —
the classic integrated line profile. Output is a 1-D plot that updates live.
Host-agnostic (Electron + Jupyter).
"""
from __future__ import annotations

import numpy as np

from spyde.actions.action import RegionAction


class LineProfileAction(RegionAction):
    """``image + line ROI -> 1-D profile averaged across the width band``."""

    name = "Line Profile"
    output_dims = 1
    output_node_name = "Line Profile"

    def selector_for_params(self, **params):
        from spyde.drawing.selectors import LineProfileSelector
        return LineProfileSelector

    def reduce(self, signal, selector, indices, **params):
        img = getattr(self.plot, "current_data", None)
        if not isinstance(img, np.ndarray) or img.ndim != 2:
            return None
        try:
            (x0, y0), (x1, y1) = selector.endpoints
            width = max(1.0, float(getattr(selector, "width", 1.0)))
        except Exception:
            return None

        length = float(np.hypot(x1 - x0, y1 - y0))
        if length < 1.0:
            return None

        # Sample points along the line (~1 px apart), then average across the
        # perpendicular width band (~1 px lanes), bilinear interpolation.
        n = int(np.ceil(length)) + 1
        t = np.linspace(0.0, 1.0, n, dtype=np.float64)
        xs = x0 + (x1 - x0) * t
        ys = y0 + (y1 - y0) * t
        nx, ny = -(y1 - y0) / length, (x1 - x0) / length
        lanes = max(1, int(round(width)))
        offs = (np.linspace(-width / 2.0, width / 2.0, lanes)
                if lanes > 1 else np.zeros(1))
        sample_x = xs[None, :] + nx * offs[:, None]
        sample_y = ys[None, :] + ny * offs[:, None]

        from scipy import ndimage
        vals = ndimage.map_coordinates(
            img.astype(np.float32, copy=False),
            [sample_y.ravel(), sample_x.ravel()],
            order=1, mode="nearest",
        ).reshape(lanes, n)
        return vals.mean(axis=0).astype(np.float32)
