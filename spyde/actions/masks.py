"""
masks.py — host-agnostic detector-mask builders for region actions.

Replaces the pyqtgraph ``roi_to_mask``: instead of reading geometry off a
pyqtgraph ROI, these read it off anyplotlib widgets (which expose ``.x/.y/.w/.h``,
``.cx/.cy/.r``, ``.cx/.cy/.r_outer/.r_inner``).  The returned mask matches the
signal's last two axes (ky, kx) so it composes with the same tensordot used
for virtual imaging.
"""
from __future__ import annotations

import numpy as np


def _signal_k_grids(signal):
    """Return (xx, yy, ky_size, kx_size) pixel-index grids matching the signal's
    last two axes.

    **Coordinate system.** anyplotlib overlay widgets report their geometry
    (``cx/cy/r``, ``x/y/w/h``) in *image-pixel* coordinates — the column/row
    index of the displayed image, 0..image_width — with **no** axis scale or
    offset applied (extent only relabels the ticks; see ``_imgToCanvas2d`` in
    anyplotlib's ``figure_esm.js``).  So the mask grid must be built in the same
    pixel-index space, *not* physical units (``pixel*scale + offset``).  Building
    it in physical units made the ROI miss the grid entirely on any calibrated
    axis (scale != 1) → empty mask → black virtual image.  This regressed
    silently because the synthetic test data used scale=1.

    ``xx`` is the horizontal/column index (the fast, innermost signal axis = kx)
    and ``yy`` is the vertical/row index (the slow axis = ky).  ``widget.cx`` is
    horizontal so it compares against ``xx``; ``widget.cy`` against ``yy``.  The
    returned grids have shape ``(ky_size, kx_size)`` so the mask composes
    directly with ``signal.data[..., ky, kx]``.
    """
    sig_axes = signal.axes_manager.signal_axes
    kx_axis = sig_axes[0]   # fast / innermost → image columns (horizontal, cx)
    ky_axis = sig_axes[1]   # slow / next-innermost → image rows (vertical, cy)
    kx_size, ky_size = kx_axis.size, ky_axis.size

    col_idx = np.arange(kx_size)   # x / horizontal / kx
    row_idx = np.arange(ky_size)   # y / vertical   / ky
    xx, yy = np.meshgrid(col_idx, row_idx)   # both (ky_size, kx_size)
    return xx, yy, ky_size, kx_size


def widget_to_mask(widget, signal) -> np.ndarray:
    """Build a float32 detector mask from an anyplotlib overlay *widget*.

    Supports rectangle, circle and annular widgets (the same shapes the old
    pyqtgraph path handled). The widget's type is detected from its attributes.
    Geometry is interpreted in image-pixel coordinates (see ``_signal_k_grids``).
    """
    xx, yy, ky_size, kx_size = _signal_k_grids(signal)
    data = getattr(widget, "_data", {})

    # Annular (ring): cx, cy, r_outer, r_inner
    if "r_outer" in data and "r_inner" in data:
        cx, cy = float(widget.cx), float(widget.cy)
        r_in, r_out = float(widget.r_inner), float(widget.r_outer)
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask_bool = (dist2 >= r_in ** 2) & (dist2 <= r_out ** 2)

    # Circle / disk: cx, cy, r
    elif "r" in data and "cx" in data:
        cx, cy, r = float(widget.cx), float(widget.cy), float(widget.r)
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask_bool = dist2 <= r ** 2

    # Rectangle: x, y, w, h
    elif "w" in data and "h" in data:
        x0, y0 = float(widget.x), float(widget.y)
        x1, y1 = x0 + float(widget.w), y0 + float(widget.h)
        mask_bool = (
            (xx >= x0) & (xx < x1) &
            (yy >= y0) & (yy < y1)
        )

    else:
        raise TypeError(f"Unsupported widget geometry: {list(data)}")

    return mask_bool.astype(np.float32)
