"""
line_profile_action.py — integrated line profile on the RegionAction template.

anyplotlib has no line-ROI primitive, so this uses a rectangle and integrates
(averages) over its short axis to produce a 1-D profile — the common
"box / integrated line profile". Output is a 1-D plot that updates live.
Host-agnostic (Electron + Jupyter).
"""
from __future__ import annotations

import numpy as np

from spyde.actions.action import RegionAction
from spyde.actions._common import widget_region


class LineProfileAction(RegionAction):
    """``image + rectangle ROI -> 1-D profile integrated across the box``."""

    name = "Line Profile"
    output_dims = 1
    output_node_name = "Line Profile"

    def selector_for_params(self, **params):
        from spyde.drawing.selectors import RectangleSelector
        return RectangleSelector

    def reduce(self, signal, selector, indices, **params):
        img = getattr(self.plot, "current_data", None)
        if not isinstance(img, np.ndarray) or img.ndim != 2:
            return None

        region = widget_region(selector, img)
        if region.size == 0:
            return None

        # Integrate across the shorter axis so the profile runs along the
        # rectangle's longer dimension.
        axis = 0 if region.shape[0] <= region.shape[1] else 1
        return region.mean(axis=axis).astype(np.float32)
