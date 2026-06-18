"""
vector_virtual_imaging.py — Electron-native Vector Virtual Imaging.

Builds virtual images from a ``SpyDEDiffractionVectors`` result tree (the output
of Find Diffraction Vectors). REUSES the raw Virtual-Imaging machinery: it is the
same multi-VI sub-toolbar UX (a "＋" adds a colour-cycled detector ROI on the
vectors diffraction pattern → its own output window that recomputes live), built
on the host-agnostic :class:`~spyde.actions.virtual_image.VirtualImageAction`
template. Only :meth:`reduce` differs — instead of a Dask reduction over the raw
4D dataset, each image is built in-memory from the CSR flat buffer via
``vecs.virtual_image_from_roi_gpu`` (O(N_frame), recomputes live on every drag).

Each ROI is intensity-weighted (sums NXCORR peak scores) by default, matching how
the vectors were detected; a "count" weighting tallies vectors instead.

No Qt — mirrors :mod:`spyde.actions.virtual_image` so it runs in the Electron
backend (and a notebook) unchanged.
"""
from __future__ import annotations

import numpy as np

from spyde.actions.virtual_image import VirtualImageAction, VI_COLORS


class VectorVirtualImageAction(VirtualImageAction):
    """A virtual image computed from the tree's diffraction vectors (not the raw
    4D data). Inherits the nav-shaped placeholder + live-recompute flow from
    :class:`VirtualImageAction`; overrides the selector mapping and the reduce."""

    name = "Vector Virtual Image"
    output_node_name = "Vector Virtual Image"

    # Same param KEYS as the raw VI (``type``/``calculation``) so the multi-VI
    # sub-toolbar UI (shape icon + per-VI caret) is reused verbatim — only the
    # option lists/meaning differ (calculation = vector weighting here).
    parameters = {
        "type": {
            "name": "Detector shape",
            "type": "enum",
            "default": "disk",
            "options": ["disk", "annular", "rectangle"],
        },
        "calculation": {
            "name": "Weighting",
            "type": "enum",
            "default": "intensity",
            "options": ["intensity", "count"],
        },
    }

    def selector_for_params(self, **params):
        from spyde.drawing.selectors import (
            CircleSelector, AnnularSelector, RectangleSelector,
        )
        return {
            "disk": CircleSelector,
            "annular": AnnularSelector,
            "rectangle": RectangleSelector,
        }.get(params.get("type", "disk"), CircleSelector)

    def _current_t(self):
        vecs = getattr(self.signal_tree, "diffraction_vectors", None)
        if vecs is None or getattr(vecs, "n_time", 0) <= 0:
            return None
        try:
            return int(self.signal_tree.root.axes_manager.indices[0])
        except Exception:
            return None

    def reduce(self, signal, selector, indices, **params):
        """Build the virtual image for the current detector ROI from the vectors.

        anyplotlib ROI widgets report geometry in *image-pixel* coords; the
        vectors store kx,ky in *calibrated* units, so convert pixel → calibrated
        (``k = px*scale + offset``) before querying the CSR buffer. Returns a
        nav-space numpy image (in-memory, synchronous — sub-ms for typical data)."""
        vecs = getattr(self.signal_tree, "diffraction_vectors", None)
        widget = getattr(selector, "roi", None)
        if vecs is None or widget is None:
            return None

        sig_axes = signal.axes_manager.signal_axes
        sx, ox = float(sig_axes[0].scale), float(sig_axes[0].offset)
        sy, oy = float(sig_axes[1].scale), float(sig_axes[1].offset)
        weighted = params.get("calculation", "intensity") != "count"
        t = self._current_t()
        data = getattr(widget, "_data", {})

        try:
            if "w" in data and "h" in data:                 # rectangle
                x0 = float(widget.x) * sx + ox
                y0 = float(widget.y) * sy + oy
                x1 = (float(widget.x) + float(widget.w)) * sx + ox
                y1 = (float(widget.y) + float(widget.h)) * sy + oy
                img = vecs.virtual_image_from_rect(
                    x0, y0, x1, y1, t=t, intensity_weighted=weighted)
            else:                                            # disk / annulus
                cx = float(widget.cx) * sx + ox
                cy = float(widget.cy) * sy + oy
                if "r_outer" in data:
                    r_out = float(widget.r_outer) * sx
                    r_in = float(widget.r_inner) * sx
                else:
                    r_out = float(widget.r) * sx
                    r_in = 0.0
                img = vecs.virtual_image_from_roi_gpu(
                    cx, cy, r_out, r_in, t=t, intensity_weighted=weighted)
        except Exception:
            return None
        return np.asarray(img, dtype=np.float32)

    def reduce_to(self, signal, selector, child, indices, **params):
        return self.reduce(signal, selector, indices, **params)


def vector_virtual_imaging(ctx, action_name: str = "Vector Virtual Imaging", **kwargs):
    """Parent submenu action — a no-op; the toolbar opens its sub-toolbar
    ("Add Vector Virtual Image") instead of dispatching this."""
    return None


def add_vector_virtual_image(ctx, action_name: str = "Add Vector Virtual Image", **params):
    """Add ONE more vector virtual image (multi-VI sub-toolbar, Qt parity): a
    colour-cycled detector ROI on the vectors diffraction pattern → its own
    output window that recomputes live from the vectors as the ROI is dragged.
    Cycles red→…→magenta and is listed as a removable chip in the Vector Virtual
    Imaging sub-toolbar."""
    from spyde.backend.ipc import emit

    plot = ctx.plot
    session = ctx.session
    if getattr(getattr(plot, "signal_tree", None), "diffraction_vectors", None) is None:
        from spyde.backend.ipc import emit_error
        emit_error("Vector Virtual Imaging: this signal has no diffraction vectors")
        return None

    items = getattr(plot, "_vi_items", [])
    n = len(items)
    color = VI_COLORS[n % len(VI_COLORS)]

    vtype = params.get("type", "disk")
    calc = params.get("calculation", "intensity")
    act = VectorVirtualImageAction(ctx)
    act.roi_color = color
    selector = act.run(type=vtype, calculation=calc)

    vi_name = f"Vector Image {n + 1} ({color})"
    out_wids = sorted({
        c.window_id for c in getattr(selector, "active_children", [])
        if getattr(c, "window_id", None) is not None
    })
    # Same chip schema as raw VI (`type`/`calculation`) so the SubToolbar shape
    # icon + per-VI caret are reused verbatim; `parent_action` routes caret edits
    # (update_vi) to the Vector Virtual Imaging bar, not the raw one.
    items.append({"name": vi_name, "color": color, "type": vtype,
                  "calculation": calc, "out_wids": out_wids,
                  "parent_action": "Vector Virtual Imaging"})
    plot._vi_items = items

    src_wid = getattr(plot, "window_id", None)
    if src_wid is not None and session is not None:
        session._action_artifacts[(src_wid, vi_name)] = {
            "selector": selector, "out_wids": out_wids, "vi_source": src_wid,
            "action": act,
        }
        emit({
            "type": "sub_item", "window_id": src_wid, "action": "Vector Virtual Imaging",
            "name": vi_name, "color": color, "vtype": vtype,
            "calculation": calc, "active": True,
        })
    return None
