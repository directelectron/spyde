"""data_embed.py — put the NUMBERS behind an exported figure into the page.

An exported report carried pixels and nothing else. The image is 8-bit codes
quantised over the display window (``anyplotlib._utils._normalize_image``), so
anything outside the contrast band is destroyed and what survives is recoverable
to about 1/256 of the range — fine to look at, useless to re-analyse. A reader
months later, or a colleague who has never opened SpyDE, could not get a strain
value out of a strain map.

The diffraction-vectors explorer already proved the shape: pack the dataset into
the page as one base64 blob and let the page work from it
(``vectors_embed.pack_vectors``). This is the same bargain for an ordinary 2-D
map or 1-D curve — full float32 precision, the calibrated axes, and the value
units, handed over as an ``.npy`` from the figure's hover toolbar
(:mod:`figure_chrome`). Not CSV: a 2-D map as text is a huge, awkward file that
nobody reloads, and the point of this is re-analysis.

Deliberately NOT downsampled past the cap. A quietly decimated array looks
exactly like the real one and would be worse than no data at all, so a panel over
:data:`MAX_EMBED_VALUES` is omitted and the page says so.
"""
from __future__ import annotations

import base64
import logging

import numpy as np

log = logging.getLogger(__name__)

#: Values per panel that will be embedded. 2 M float32 is 8 MB raw, ~11 MB as
#: base64 — already a heavy page, and a 4k × 4k map is 16 M values, which is not
#: a document. Past this the panel exports pixels only, with a visible note.
MAX_EMBED_VALUES = 2_000_000


def pack_panel_data(spec, snapshots: dict) -> "dict | None":
    """``{"panels": [...]}`` — every scalar layer of *spec*, or None when there
    is nothing embeddable.

    Each panel entry carries the base64 float32 values, the shape, the
    calibrated axis coordinates and units, and the value units, so a reader
    knows what the numbers are without the report beside them.
    """
    panels = []
    for panel in getattr(spec, "panels", []) or []:
        if str(getattr(panel, "kind", "image")) == "scene3d":
            continue                    # a point cloud, already in the figure
        axes = dict(getattr(panel, "axes", None) or {})
        for layer in getattr(panel, "layers", []) or []:
            array = snapshots.get((panel.id, layer.id))
            if array is None:
                continue
            array = np.asarray(array)
            if array.ndim == 3 and array.shape[-1] in (3, 4):
                continue                # RGB is a picture, not a measurement
            if array.ndim not in (1, 2):
                continue
            name = str(getattr(layer.source, "title", "") or panel.title
                       or "data")
            entry = {
                "panel": str(panel.id),
                "layer": str(layer.id),
                "name": name,
                "shape": [int(n) for n in array.shape],
                "axis_units": str(axes.get("units", "") or ""),
                "value_units": str(axes.get("value_units", "") or ""),
            }
            if array.size > MAX_EMBED_VALUES:
                # Say so rather than shipping a decimated copy that looks whole.
                entry["omitted"] = True
                entry["reason"] = (f"{array.size:,} values exceeds the "
                                   f"{MAX_EMBED_VALUES:,} embed cap")
                panels.append(entry)
                continue
            values = np.ascontiguousarray(array, dtype=np.float32)
            entry["b64"] = base64.b64encode(values.tobytes()).decode("ascii")
            for key in ("x_axis", "y_axis"):
                coords = axes.get(key)
                if coords is not None:
                    entry[key] = [float(v) for v in coords]
            panels.append(entry)
    return {"panels": panels} if panels else None
