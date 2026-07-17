"""overlay_embed.py — embed a tinted-overlay BLENDER in an HTML report export.

A figure cell whose panels carry tinted overlay layers (``LayerSpec.tint``)
exports, in INTERACTIVE mode, a self-contained blender page instead of the
static anyplotlib iframe: the base layer renders grayscale to a Canvas2D, each
tinted overlay composites over it as a clear→tint intensity ramp, and one
``<input type=range>`` per overlay drives that layer's opacity LIVE in the
exported file — no backend, no network, no anyplotlib runtime (plain canvas
compositing; the ramp math mirrors anyplotlib's tint LUT: per-texel alpha =
intensity/255, multiplied by the layer alpha).

Structurally modeled on ``vectors_embed.py`` (the self-contained interactive
HTML precedent): pack numpy → base64 u8 planes + a JSON header, inline CSS +
a module script, flag ``data-ready`` for the browser tests, and hook into
``export_html._render_body`` exactly like the vectors-viewer swap (which takes
PRECEDENCE — a vectors cell stays a vectors explorer). Static/PDF exports are
unchanged (baked blend).

Packing: per GRID panel, the base layer's snapshot normalized to uint8 over
its clim (absent clim → the same robust min→99.5-percentile window
``bake_fallback_png`` uses) plus each TINTED overlay's u8 plane and its
``{tint, alpha, visible, name}`` metadata. Any plane whose long side exceeds
``MAX_EDGE`` is stride-downsampled first (base + overlays share the stride —
they share the shape by the overlay contract). Multi-panel cells stack panel
blocks vertically. Returns None (→ the caller falls through to the live-figure
iframe) when no panel has a tinted overlay.
"""
from __future__ import annotations

import base64
import html as _html
import json
import logging

import numpy as np

log = logging.getLogger(__name__)

# Long-side cap for embedded planes: a report blender is a visual aid, not a
# data container — 1024 px is plenty and keeps the b64 payload ~1 MB/plane.
MAX_EDGE = 1024


def _grid_panels(spec):
    """The panels occupying a GRID cell (mirrors compose._grid_panels): every
    panel NOT referenced as a callout inset — a floating inset stacked as its
    own blender block would misread as a separate figure."""
    inset_ids = set()
    for p in spec.panels:
        for ins in (p.insets or []):
            if ins.get("panel"):
                inset_ids.add(ins["panel"])
    return [p for p in spec.panels if p.id not in inset_ids]


def _downsample(arr: np.ndarray, max_edge: int = MAX_EDGE) -> np.ndarray:
    """Stride-downsample so the long side fits *max_edge* (cheap, no float
    work — the same approach bake_fallback_png takes before normalizing)."""
    long_edge = max(arr.shape[0], arr.shape[1])
    if long_edge <= max_edge:
        return arr
    stride = int(np.ceil(long_edge / max_edge))
    return arr[::stride, ::stride] if arr.ndim == 2 else arr[::stride, ::stride, :]


def _to_u8(arr, clim) -> "np.ndarray | None":
    """Normalize a 2-D scalar plane to uint8 over ``clim`` ([lo, hi]); an
    absent/unusable clim auto-ranges over the finite data with the SAME robust
    window ``bake_fallback_png`` uses (min → 99.5th percentile). An RGB(A)
    base (e.g. an IPF map) collapses to channel-mean gray — the blender's base
    is grayscale by design. Returns None for an unusable array."""
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        a = np.asarray(a[..., :3], dtype=np.float64).mean(axis=-1)
    if a.ndim != 2:
        return None
    a = np.asarray(a, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if clim is not None and clim[0] is not None and clim[1] is not None:
        lo, hi = float(clim[0]), float(clim[1])
    elif finite.size:
        lo = float(np.nanmin(finite))
        hi = float(np.nanpercentile(finite, 99.5))
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    a = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    return (a * 255.0 + 0.5).astype(np.uint8)


def _b64_u8(u8: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(u8).tobytes()).decode("ascii")


def _pack_cell(mgr, cell) -> "dict | None":
    """The blender payload ``{"panels": [...]}`` for *cell*, or None when the
    spec has no tinted overlay layer (the gate — legacy/untinted cells keep the
    live-figure iframe)."""
    spec = getattr(cell, "spec", None)
    if spec is None or not spec.panels:
        return None
    snap = mgr.snapshot_map(cell.id)
    panels_out: list[dict] = []
    any_tint = False
    for panel in _grid_panels(spec):
        if not panel.layers:
            continue
        base_layer = panel.layers[0]
        base_arr = snap.get((panel.id, base_layer.id))
        if base_arr is None:
            continue
        base_u8 = _to_u8(_downsample(np.asarray(base_arr)), base_layer.clim)
        if base_u8 is None:
            continue
        h, w = base_u8.shape
        overlays: list[dict] = []
        for layer in panel.layers[1:]:
            if not getattr(layer, "tint", None):
                continue        # cmap overlays have no ramp to blend — skipped
            arr = snap.get((panel.id, layer.id))
            if arr is None:
                continue
            u8 = _to_u8(_downsample(np.asarray(arr)), layer.clim)
            if u8 is None or u8.shape != (h, w):
                continue
            overlays.append({
                "tint": str(layer.tint),
                "alpha": float(layer.alpha),
                "visible": bool(layer.visible),
                "name": str(getattr(layer.source, "title", "") or "Overlay"),
                "b64": _b64_u8(u8),
            })
        if overlays:
            any_tint = True
        panels_out.append({
            "title": str(panel.title or ""),
            "w": int(w), "h": int(h),
            "base_b64": _b64_u8(base_u8),
            "overlays": overlays,
        })
    if not any_tint or not panels_out:
        return None
    return {"panels": panels_out}


_BLENDER_CSS = """
:root { color-scheme: light; }
body { margin: 0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       background: #fff; color: #1a1a1a; font-size: 12px; }
.ovb-wrap { display: flex; flex-direction: column; gap: 14px; padding: 6px; }
.ovb-panel { display: flex; flex-direction: column; gap: 6px; }
.ovb-panel h4 { margin: 0; font-size: 12px; font-weight: 600; color: #444; }
.ovb-canvas { width: 100%; max-width: 360px; image-rendering: pixelated;
              border: 1px solid #e2e2e6; border-radius: 4px; }
.ovb-row { display: flex; gap: 8px; align-items: center; max-width: 360px; }
.ovb-swatch { width: 12px; height: 12px; border-radius: 3px; flex: none;
              border: 1px solid rgba(0,0,0,.35); }
.ovb-name { font-size: 11px; color: #444; min-width: 60px; max-width: 140px;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ovb-slider { flex: 1; }
.ovb-val { font-size: 11px; color: #666; min-width: 30px; text-align: right; }
.ovb-meta { color: #666; font-size: 11px; max-width: 700px; }
"""

# The glue script. Decodes the u8 planes, composites base + clear→tint ramps
# onto each panel's Canvas2D, and recomposites on every slider input. Kept a
# plain module (no imports, no anyplotlib) so the page works anywhere forever.
# `window.__ovb` mirrors slider state for the browser tests.
_BLENDER_JS = r"""
const DATA = JSON.parse(document.getElementById('ovb-data').textContent);

function u8(b64, n) {
  const s = atob(b64);
  const a = new Uint8Array(n);
  for (let i = 0; i < n; i++) a[i] = s.charCodeAt(i);
  return a;
}
function hexRgb(h) {
  let s = h.replace('#', '');
  if (s.length === 3) s = s.split('').map((c) => c + c).join('');
  return [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16),
          parseInt(s.slice(4, 6), 16)];
}

window.__ovb = { panels: [] };

DATA.panels.forEach((pn, pi) => {
  const canvas = document.getElementById('ovb-canvas-' + pi);
  const ctx = canvas.getContext('2d');
  const n = pn.w * pn.h;
  const base = u8(pn.base_b64, n);
  const panelEl = canvas.closest('.ovb-panel');
  // DOM order of the sliders inside this panel block == overlays order.
  const sliders = panelEl.querySelectorAll('.ovb-slider');
  const vals = panelEl.querySelectorAll('.ovb-val');
  const ovs = pn.overlays.map((o, oi) => ({
    px: u8(o.b64, n),
    rgb: hexRgb(o.tint),
    visible: o.visible,
    slider: sliders[oi],
    valEl: vals[oi],
  }));

  function composite() {
    const img = ctx.createImageData(pn.w, pn.h);
    const d = img.data;
    for (let i = 0; i < n; i++) {
      let r = base[i], g = base[i], b = base[i];
      for (const o of ovs) {
        if (!o.visible || o.val <= 0) continue;
        // Clear→tint ramp: per-texel alpha = intensity/255, times the layer
        // opacity — the same law as anyplotlib's tint LUT compositor.
        const a = (o.px[i] / 255) * o.val;
        if (a <= 0) continue;
        r = r * (1 - a) + o.rgb[0] * a;
        g = g * (1 - a) + o.rgb[1] * a;
        b = b * (1 - a) + o.rgb[2] * a;
      }
      d[4 * i] = r; d[4 * i + 1] = g; d[4 * i + 2] = b; d[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  for (const o of ovs) {
    o.val = Number(o.slider.value);
    o.slider.addEventListener('input', () => {
      o.val = Number(o.slider.value);
      if (o.valEl) o.valEl.textContent = o.val.toFixed(2);
      composite();
    });
  }
  composite();
  window.__ovb.panels.push({ ovs, composite });
});

document.getElementById('ovb-root').dataset.ready = '1';
"""


def overlay_blender_html(mgr, cell, caption: str = "") -> "str | None":
    """The self-contained blender page for *cell* (goes into the report's
    sandboxed ``<iframe srcdoc>`` like any interactive figure), or None when the
    cell's spec has no overlay layer with a tint — the caller then falls
    through to the live-figure iframe."""
    payload = _pack_cell(mgr, cell)
    if payload is None:
        return None
    cap = _html.escape(caption or "")

    blocks: list[str] = []
    for pi, pn in enumerate(payload["panels"]):
        title = _html.escape(pn["title"] or "")
        rows: list[str] = []
        for oi, ov in enumerate(pn["overlays"]):
            name = _html.escape(ov["name"])
            tint = _html.escape(ov["tint"])
            val = max(0.0, min(1.0, float(ov["alpha"])))
            rows.append(
                "<div class=\"ovb-row\">"
                f"<span class=\"ovb-swatch\" style=\"background:{tint}\"></span>"
                f"<span class=\"ovb-name\" title=\"{name}\">{name}</span>"
                f"<input type=\"range\" class=\"ovb-slider\" "
                f"id=\"ovb-slider-{pi}-{oi}\" min=\"0\" max=\"1\" "
                f"step=\"0.01\" value=\"{val:.2f}\">"
                f"<span class=\"ovb-val\">{val:.2f}</span>"
                "</div>"
            )
        blocks.append(
            "<div class=\"ovb-panel\">"
            + (f"<h4>{title}</h4>" if title else "")
            + f"<canvas id=\"ovb-canvas-{pi}\" class=\"ovb-canvas\" "
              f"width=\"{pn['w']}\" height=\"{pn['h']}\"></canvas>"
            + "".join(rows)
            + "</div>"
        )

    # </script> can't appear inside a script element — escape the slash (the
    # vectors_embed convention; base64/JSON content can't otherwise collide).
    data_json = json.dumps(payload).replace("</", "<\\/")
    return (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<style>{_BLENDER_CSS}</style></head><body>"
        "<div id=\"ovb-root\" class=\"ovb-wrap\">"
        + "".join(blocks)
        + (f"<div class=\"ovb-meta\">{cap}</div>" if cap else "")
        + "</div>"
        f"<script type=\"application/json\" id=\"ovb-data\">{data_json}</script>"
        f"<script type=\"module\">{_BLENDER_JS}</script>"
        "</body></html>"
    )
