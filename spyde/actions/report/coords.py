"""coords.py — DATA → PIXEL conversion for report figure annotations.

Report annotation dicts (``PanelSpec.annotations``) store ``offsets`` and sizes
in calibrated DATA coordinates (the same coords the on-disk spec/YAML uses —
a calibration-aware recipe). anyplotlib's 2-D marker path renders those offsets
as IMAGE PIXELS (``drawMarkers2d`` → ``_imgToCanvas2d`` is a pure pixel
transform; there is no data→px conversion on the marker path). So a panel
calibrated with real-world axes (e.g. 0–12 nm) needs its annotation dicts
converted to pixel space at render time — this module is that conversion,
kept separate from ``compose.py`` (which owns the on-disk spec / INDEX→DATA
conversion) to avoid a circular import from ``figure_builder.py``.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def axis_offset_scale(axis_vals):
    """(offset, scale) for a calibrated 1-D axis array (offset = first sample,
    scale = uniform step). None when the array is missing / too short.

    Mirrors ``spyde.actions.report.compose._axis_offset_scale`` — duplicated
    (not imported) to avoid a circular import between ``compose`` and
    ``figure_builder``."""
    try:
        a = np.asarray(axis_vals, dtype=float)
    except (TypeError, ValueError):
        return None
    if a.ndim != 1 or a.size < 1:
        return None
    offset = float(a[0])
    scale = float(a[1] - a[0]) if a.size >= 2 else 1.0
    return offset, scale


def _panel_data_to_pixel_scale(axes):
    """``(ox, sx, oy, sy)`` for a panel's ``axes`` dict (``{units, x_axis,
    y_axis}``), or None if unusable (missing/short axes) → caller should treat
    the panel as uncalibrated (identity, index == pixel)."""
    if not axes:
        return None
    xo_sc = axis_offset_scale(axes.get("x_axis"))
    yo_sc = axis_offset_scale(axes.get("y_axis"))
    if xo_sc is None or yo_sc is None:
        return None
    ox, sx = xo_sc
    oy, sy = yo_sc
    if sx == 0 or sy == 0:
        return None
    return ox, sx, oy, sy


def _convert_point(x, y, ox, sx, oy, sy):
    return (x - ox) / sx, (y - oy) / sy


def _convert_offsets(offsets, ox, sx, oy, sy):
    arr = np.asarray(offsets, dtype=float)
    single = arr.ndim == 1
    if single:
        arr = arr[np.newaxis, :]
    out = np.empty_like(arr)
    out[:, 0] = (arr[:, 0] - ox) / sx
    out[:, 1] = (arr[:, 1] - oy) / sy
    if single:
        return out[0].tolist()
    return out.tolist()


def _convert_size(val, denom):
    """Divide a scalar or per-element size (radius/widths/heights) by *denom*."""
    if val is None:
        return val
    arr = np.asarray(val, dtype=float)
    out = arr / denom
    if arr.ndim == 0:
        return float(out)
    return out.tolist()


def _convert_segments(segments, ox, sx, oy, sy):
    arr = np.asarray(segments, dtype=float)
    out = arr.copy()
    out[..., 0] = (arr[..., 0] - ox) / sx
    out[..., 1] = (arr[..., 1] - oy) / sy
    return out.tolist()


def annotation_data_to_pixel(ann: dict, axes) -> dict:
    """Return a COPY of annotation dict *ann* with data-coordinate fields
    (offsets/sizes/segments/U/V) converted to pixel space using panel *axes*
    (``{units, x_axis, y_axis}`` or None/unusable → identity passthrough,
    since an uncalibrated panel already has index == pixel). Never mutates
    *ann*."""
    out = dict(ann)
    scale = _panel_data_to_pixel_scale(axes)
    if scale is None:
        return out
    ox, sx, oy, sy = scale
    size_denom = (abs(sx) + abs(sy)) / 2.0

    if "offsets" in out and out["offsets"] is not None:
        out["offsets"] = _convert_offsets(out["offsets"], ox, sx, oy, sy)
    if "radius" in out and out["radius"] is not None and size_denom:
        out["radius"] = _convert_size(out["radius"], size_denom)
    if "widths" in out and out["widths"] is not None and sx:
        out["widths"] = _convert_size(out["widths"], abs(sx))
    if "heights" in out and out["heights"] is not None and sy:
        out["heights"] = _convert_size(out["heights"], abs(sy))
    if "U" in out and out["U"] is not None and sx:
        out["U"] = _convert_size(out["U"], sx)
    if "V" in out and out["V"] is not None and sy:
        out["V"] = _convert_size(out["V"], sy)
    if "segments" in out and out["segments"] is not None:
        out["segments"] = _convert_segments(out["segments"], ox, sx, oy, sy)
    return out


# ── INVERSE: PIXEL → DATA (the edit-mode drag write-back) ─────────────────────
#
# When a report figure is in EDIT MODE its annotations are rendered as draggable
# anyplotlib WIDGETS (image-pixel coords). A drag's final geometry (pixel) is
# converted BACK to the DATA coords the spec stores, using the panel's axes.
# These are the exact inverse of ``annotation_data_to_pixel``'s per-field maps:
#   point:  data = origin + pixel * scale        (inverse of (data-origin)/scale)
#   radius: data = pixel * ((|sx|+|sy|)/2)        (inverse of / size_denom)
#   width:  data = pixel * |sx|                   (inverse of / |sx|)
#   height: data = pixel * |sy|                   (inverse of / |sy|)
#   U:      data = pixel * sx  (signed)           (inverse of / sx)
#   V:      data = pixel * sy  (signed)           (inverse of / sy)
# ``axes=None``/unusable → identity passthrough (uncalibrated panel: px == data).


def pixel_to_data_point(x, y, axes):
    """Convert one IMAGE-PIXEL point ``(x, y)`` to DATA coords using panel *axes*.
    Identity when the panel is uncalibrated (axes None/unusable)."""
    scale = _panel_data_to_pixel_scale(axes)
    if scale is None:
        return float(x), float(y)
    ox, sx, oy, sy = scale
    return ox + float(x) * sx, oy + float(y) * sy


def pixel_to_data_radius(r, axes):
    """Convert a pixel radius to DATA using the mean axis scale. Identity when
    uncalibrated."""
    scale = _panel_data_to_pixel_scale(axes)
    if scale is None or r is None:
        return None if r is None else float(r)
    _ox, sx, _oy, sy = scale
    size_denom = (abs(sx) + abs(sy)) / 2.0
    return float(r) * size_denom if size_denom else float(r)


def pixel_to_data_width(w, axes):
    """Convert a pixel width to DATA (scaled by ``|sx|``). Identity when
    uncalibrated."""
    scale = _panel_data_to_pixel_scale(axes)
    if scale is None or w is None:
        return None if w is None else float(w)
    _ox, sx, _oy, _sy = scale
    return float(w) * abs(sx) if sx else float(w)


def pixel_to_data_height(hh, axes):
    """Convert a pixel height to DATA (scaled by ``|sy|``). Identity when
    uncalibrated."""
    scale = _panel_data_to_pixel_scale(axes)
    if scale is None or hh is None:
        return None if hh is None else float(hh)
    _ox, _sx, _oy, sy = scale
    return float(hh) * abs(sy) if sy else float(hh)


def pixel_to_data_u(u, axes):
    """Convert a pixel ``U`` displacement to DATA (SIGNED, scaled by ``sx``).
    Identity when uncalibrated."""
    scale = _panel_data_to_pixel_scale(axes)
    if scale is None or u is None:
        return None if u is None else float(u)
    _ox, sx, _oy, _sy = scale
    return float(u) * sx if sx else float(u)


def pixel_to_data_v(v, axes):
    """Convert a pixel ``V`` displacement to DATA (SIGNED, scaled by ``sy``).
    Identity when uncalibrated."""
    scale = _panel_data_to_pixel_scale(axes)
    if scale is None or v is None:
        return None if v is None else float(v)
    _ox, _sx, _oy, sy = scale
    return float(v) * sy if sy else float(v)
