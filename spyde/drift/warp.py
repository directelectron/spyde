"""
warp.py — apply a drift correction to ONE frame.

Per-frame by design. This is the function a lazy signal-tree node calls, so the
aligned movie is never materialised (``spyde/drift/__init__.py``), and it is also
what the derived-view reader would call if the per-frame shift transform is ever
added to ``array_cache/readers/per_frame.py`` — which is a signal-tree read-path
change and therefore gated on review, NOT done here.

Edge policy (locked, plan §A7): the frame keeps its full size and uncovered
pixels become **NaN**, with :func:`coverage_mask` giving the validity map.
Nothing is cropped and nothing is filled with invented data.

**Downstream contract:** segmentation MUST respect the coverage mask. A NaN-padded
border is the single most likely integration bug in this feature — a threshold
applied to NaN, or a NaN-to-zero conversion, invents a large "particle" along the
edge that then nucleates a spurious track.
"""
from __future__ import annotations

import numpy as np


def _split_shift(shift) -> tuple[np.ndarray, bool]:
    s = np.asarray(shift, dtype=np.float64).reshape(-1)
    if s.size != 2:
        raise ValueError(f"shift must be (dy, dx); got {np.shape(shift)}")
    if not np.all(np.isfinite(s)):
        raise ValueError(f"shift must be finite; got {s!r}")
    return s, bool(np.all(s == np.round(s)))


def coverage_mask(shape: tuple[int, int], shift) -> np.ndarray:
    """Boolean map of which output pixels come from real source data.

    A pixel is covered when its source coordinate falls inside the source frame.
    For a sub-pixel shift the border row/column that would need to interpolate
    against off-frame data is treated as **uncovered** — bilinear interpolation
    there would silently blend in the fill value.
    """
    h, w = int(shape[0]), int(shape[1])
    s, integral = _split_shift(shift)
    dy, dx = s

    mask = np.zeros((h, w), dtype=bool)
    if integral:
        y0, y1 = max(0, int(dy)), min(h, h + int(dy))
        x0, x1 = max(0, int(dx)), min(w, w + int(dx))
    else:
        # Output pixel y draws from source y - dy; it needs floor and floor+1,
        # so require 0 <= y - dy and y - dy + 1 <= h - 1.
        y0 = int(np.ceil(max(0.0, dy)))
        y1 = int(np.floor(min(float(h), h + dy - 1.0))) + 1
        x0 = int(np.ceil(max(0.0, dx)))
        x1 = int(np.floor(min(float(w), w + dx - 1.0))) + 1
    if y1 > y0 and x1 > x0:
        mask[y0:y1, x0:x1] = True
    return mask


def shift_frame(
    frame: np.ndarray,
    shift,
    *,
    order: int = 1,
    fill: float = np.nan,
    preserve_dtype: bool = False,
) -> np.ndarray:
    """Shift *frame* by ``(dy, dx)``, padding uncovered pixels with *fill*.

    Parameters
    ----------
    frame
        2-D source frame, any dtype.
    shift
        ``(dy, dx)`` correction — the value ADDED to coordinates. See
        :mod:`spyde.drift.model` for the sign convention.
    order
        Interpolation order for a sub-pixel shift (1 = bilinear, the default;
        3 = cubic). Ignored for a whole-pixel shift, which is done exactly.
    fill
        Value for uncovered pixels. NaN by default, which forces a float result.
    preserve_dtype
        Keep the source dtype. Only honoured for a whole-pixel shift with a
        non-NaN *fill* — a uint16 frame cannot hold NaN, and interpolation
        cannot be exact in an integer type. Raises otherwise rather than
        silently returning something lossy.

    Notes
    -----
    A whole-pixel shift takes an exact slice-copy path: no interpolation, no
    float promotion, and bit-identical to the source pixels. This matters because
    the common case for a well-behaved stage IS an integer shift, and running it
    through ``scipy.ndimage.shift`` would resample (and blur) data that did not
    need to move sub-pixel at all.
    """
    src = np.asarray(frame)
    if src.ndim != 2:
        raise ValueError(f"frame must be 2-D; got shape {src.shape}")
    s, integral = _split_shift(shift)
    dy, dx = s
    h, w = src.shape

    if preserve_dtype and not (integral and np.isfinite(fill)):
        raise ValueError(
            "preserve_dtype=True requires a whole-pixel shift and a finite fill "
            f"(got shift={s.tolist()}, fill={fill!r}); a sub-pixel shift must "
            "interpolate and NaN padding cannot be stored in an integer dtype"
        )

    out_dtype = src.dtype if preserve_dtype else np.float32

    if integral:
        out = np.full((h, w), fill, dtype=out_dtype)
        iy, ix = int(dy), int(dx)
        # Destination window, and the matching source window.
        dy0, dy1 = max(0, iy), min(h, h + iy)
        dx0, dx1 = max(0, ix), min(w, w + ix)
        if dy1 > dy0 and dx1 > dx0:
            out[dy0:dy1, dx0:dx1] = src[dy0 - iy:dy1 - iy, dx0 - ix:dx1 - ix]
        return out

    from scipy.ndimage import shift as ndi_shift

    # ndimage cannot propagate NaN through its spline filter without smearing it,
    # so interpolate with a finite sentinel and stamp the fill on afterwards using
    # the analytic coverage map. This keeps the padded border crisp instead of
    # letting a NaN bleed `order` pixels into real data.
    work = src.astype(np.float32, copy=False)
    out = ndi_shift(work, s, order=order, mode="constant", cval=0.0, prefilter=order > 1)
    out = out.astype(out_dtype, copy=False)
    cov = coverage_mask((h, w), s)
    out[~cov] = fill
    return out
