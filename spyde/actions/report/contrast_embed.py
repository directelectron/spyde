"""contrast_embed.py — a working contrast control in an EXPORTED report.

A reader opening a saved report could not adjust a figure's contrast at all. The
window was whatever the author had set when they wrote the report, and it was
baked into the pixels: anyplotlib quantises a scalar frame to 8-bit codes over
its clim, so everything outside that band is saturated to 0/255 before the file
is written. Widening was not unimplemented, it was impossible.

Two things make it possible, and both are needed:

* the export quantises over a WIDE robust band and moves the window with
  ``Plot2D.set_display_window`` instead of ``set_clim``
  (``figure_builder._render_panel``), so the codes keep headroom either side of
  what is on screen;
* this module measures the histogram and the window bounds; the figure's hover
  toolbar (:mod:`figure_chrome`) draws them and drives the figure through
  anyplotlib's ``awi_state`` message.

The host page, not the figure iframe, because the iframe is a sandboxed
``srcdoc`` and its JS is anyplotlib's, not ours. Re-sending the whole
``panel_<id>_json`` trait per drag tick is cheap for the same reason the export
is self-contained: ``_resolve_pixels_for_standalone`` splits the pixels onto a
separate ``panel_<id>_geom`` trait, so the light state carries display window,
colormap name and labels — no image.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def pack_contrast(spec, snapshots: dict, light_state: dict) -> "dict | None":
    """``{"panels": [...]}`` — the histogram + window bounds for every scalar
    panel of a cell, or None when none of them can be re-windowed.

    A panel is only included when its serialised state has real headroom
    (``raw_min``/``raw_max`` wider than the display window). Offering a control
    that cannot move anything is worse than offering none: the reader drags it
    and concludes the report is broken.
    """
    from de_shell.plotting.figure import histogram_payload

    panels = []
    for panel in getattr(spec, "panels", []) or []:
        if str(getattr(panel, "kind", "image")) != "image":
            continue
        if not panel.layers:
            continue
        entry = light_state.get(str(panel.id))
        if not entry:
            continue
        state = entry.get("state") or {}
        raw_lo, raw_hi = state.get("raw_min"), state.get("raw_max")
        disp_lo, disp_hi = state.get("display_min"), state.get("display_max")
        if None in (raw_lo, raw_hi, disp_lo, disp_hi) or raw_hi <= raw_lo:
            continue
        # No headroom → set_clim quantised over the window (an older anyplotlib,
        # or an RGB panel). The LUT is the identity; a control would be a lie.
        if float(raw_lo) >= float(disp_lo) and float(raw_hi) <= float(disp_hi):
            continue
        array = snapshots.get((panel.id, panel.layers[0].id))
        if array is None:
            continue
        array = np.asarray(array)
        if array.ndim != 2:
            continue
        hist = histogram_payload(array, float(disp_lo), float(disp_hi),
                                 (float(raw_lo), float(raw_hi)))
        if hist is None:
            continue
        panels.append({
            # The DISPATCH id — it names the `panel_<id>_json` trait the page
            # re-sends, which is not the spec id this entry was looked up by.
            "panel": str(entry.get("dispatch") or panel.id),
            "title": str(panel.title or ""),
            "counts": hist["counts"],
            "edges": hist["edges"],
            "vmin": float(disp_lo),
            "vmax": float(disp_hi),
            "raw_min": float(raw_lo),
            "raw_max": float(raw_hi),
            "state": state,
        })
    return {"panels": panels} if panels else None
