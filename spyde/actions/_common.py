"""Small shared helpers used across several action modules.

Centralizes a few snippets that were copy-pasted in 2+ action modules
(reciprocal-radius from signal calibration, the strain component/title
constants, and the rectangular-ROI -> image-region slice). Keep this module
dependency-light (numpy + plain helpers only) so any action can import it
without pulling heavy deps.
"""
from __future__ import annotations

import numpy as np

# ── Strain component constants (shared by strain_action + strain_display) ──────
# Canonical order of strain-tensor components and their display titles.
STRAIN_COMPONENTS: tuple[str, ...] = ("exx", "eyy", "exy", "omega")
STRAIN_TITLES: dict[str, str] = {
    "exx": "εxx", "eyy": "εyy", "exy": "εxy", "omega": "ω",
}


def reciprocal_radius(signal) -> float:
    """Max reciprocal radius from the signal-axis calibration (Å⁻¹).

    The smallest half-extent across the signal axes — i.e. the largest radius
    that still fits inside the detector in every signal dimension.
    """
    sig_axes = signal.axes_manager.signal_axes
    return float(min(ax.scale * ax.size / 2.0 for ax in sig_axes))


def widget_region(selector, img: np.ndarray) -> np.ndarray:
    """Slice the rectangular-ROI region of ``img`` selected by ``selector``.

    Reads the 2-D widget bounds (x, y, w, h in image pixels — see the
    anyplotlib-widget-pixel-coords convention), clamps to the image, and returns
    the sub-array. Falls back to the full image when there is no usable ROI.
    """
    widget = getattr(selector, "roi", None)
    if widget is not None and hasattr(widget, "_data") and "w" in widget._data:
        x0 = int(round(float(widget.x)))
        y0 = int(round(float(widget.y)))
        x1 = int(round(float(widget.x) + float(widget.w)))
        y1 = int(round(float(widget.y) + float(widget.h)))
        x0, x1 = sorted((max(0, x0), min(img.shape[1], x1)))
        y0, y1 = sorted((max(0, y0), min(img.shape[0], y1)))
        region = img[y0:y1, x0:x1]
    else:
        region = img
    return region
