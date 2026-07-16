"""vectors_embed.py — embed a FULL diffraction-vectors dataset in an HTML report.

A find-vectors result is a compact CSR flat buffer (per vector: nav x/y, kx,
ky, intensity) — small enough to inline into a self-contained HTML page and
rich enough to recompute VIRTUAL IMAGES client-side. The exported panel is
built from REAL anyplotlib figures (the same widget UX as the app — user
feedback: "the scroll is weird, just use the anyplotlib circle"):

- k-space figure: the log density of every vector, with a draggable /
  handle-resizable **circle** (or **annulus** — a radio swaps the detector
  shape) widget selecting the detector region;
- virtual-image figure: the intensity-sum (or counts) per nav position of the
  vectors inside the detector, recomputed live in the page, with a draggable
  **rectangle** widget selecting a REAL-SPACE region — the k-space density
  then shows only the diffraction spots from that region.

The glue rides anyplotlib's standalone machinery (``mount(el, state,
{onEvent})`` + ``handle.applyUpdate``): widget drags arrive as
``pointer_move/up`` events carrying the widget geometry in IMAGE PIXELS; the
script recomputes a uint8 frame and pushes it back through the figure's own
panel state. No backend, no network — the single .html file works years later.

Packing (little-endian, one base64 blob):
    uint16  x[n]        nav column index
    uint16  y[n]        nav row index
    float32 kx[n], ky[n], intensity[n]
16 bytes/vector → ~1 M vectors ≈ 16 MB (≈21 MB as base64). Above
``MAX_EMBED_VECTORS`` the embed is refused (returns None) and the export falls
back to the baked static image.

Hooked into ``export_html._render_body`` (interactive mode): a figure cell
whose resolved source tree carries ``diffraction_vectors`` exports this
explorer instead of the anyplotlib figure iframe.
"""
from __future__ import annotations

import base64
import html as _html
import json
import logging

import numpy as np

log = logging.getLogger(__name__)

MAX_EMBED_VECTORS = 3_000_000
DENSITY_BINS = 256          # k-space density image size (px)
FIG_PX = 340                # each anyplotlib figure's CSS size


def _axis_extent(ax) -> tuple[float, float]:
    """(lo, hi) data extent of a (hyperspy or _AxisLite) axis."""
    scale = float(getattr(ax, "scale", 1.0) or 1.0)
    offset = float(getattr(ax, "offset", 0.0) or 0.0)
    size = int(getattr(ax, "size", 0) or 0)
    return offset, offset + scale * max(0, size - 1)


def pack_vectors(vecs) -> "dict | None":
    """Pack a SpyDEDiffractionVectors into the embed payload
    ``{header: dict, b64: str}`` — or None when too large / empty."""
    from spyde.signals.diffraction_vectors import COL_KX, COL_KY, COL_INTENSITY

    buf = np.asarray(vecs.flat_buffer, dtype=np.float32)
    n = int(buf.shape[0])
    if n == 0 or n > MAX_EMBED_VECTORS:
        if n:
            log.info("[report] vectors embed skipped: %d vectors > cap %d",
                     n, MAX_EMBED_VECTORS)
        return None

    nav_shape = tuple(int(s) for s in vecs.nav_shape)          # (ny, nx)
    x = buf[:, 0].astype(np.uint16)
    y = buf[:, 1].astype(np.uint16)
    kx = np.ascontiguousarray(buf[:, COL_KX], dtype="<f4")
    ky = np.ascontiguousarray(buf[:, COL_KY], dtype="<f4")
    inten = np.ascontiguousarray(buf[:, COL_INTENSITY], dtype="<f4")

    blob = (x.astype("<u2").tobytes() + y.astype("<u2").tobytes()
            + kx.tobytes() + ky.tobytes() + inten.tobytes())

    sig_axes = list(getattr(vecs, "sig_axes", []) or [])
    if len(sig_axes) >= 2:
        kx_lo, kx_hi = _axis_extent(sig_axes[0])
        ky_lo, ky_hi = _axis_extent(sig_axes[1])
    else:   # uncalibrated fallback: span the data
        kx_lo, kx_hi = float(kx.min()), float(kx.max())
        ky_lo, ky_hi = float(ky.min()), float(ky.max())
    units = str(getattr(sig_axes[0], "units", "") or "") if sig_axes else ""

    header = {
        "n": n,
        "nav": [nav_shape[-2], nav_shape[-1]],
        "k": {"kx": [kx_lo, kx_hi], "ky": [ky_lo, ky_hi], "units": units},
        "bins": DENSITY_BINS,
        "r0": float(getattr(vecs, "kernel_radius_data", 0.0) or 0.0),
    }
    return {"header": header, "b64": base64.b64encode(blob).decode("ascii")}


def _k_density_u8(kx, ky, inten, header) -> np.ndarray:
    """Log-scaled uint8 density image of ALL vectors (the initial k panel)."""
    bins = header["bins"]
    (kx_lo, kx_hi) = header["k"]["kx"]
    (ky_lo, ky_hi) = header["k"]["ky"]
    hist, _, _ = np.histogram2d(
        ky, kx, bins=bins,
        range=[[ky_lo, ky_hi or 1], [kx_lo, kx_hi or 1]], weights=inten)
    img = np.log1p(hist)
    mx = float(img.max()) or 1.0
    return (255 * img / mx).astype(np.uint8)


def _build_figures(vecs, payload) -> "tuple[dict, dict, str] | None":
    """Build the two anyplotlib figures (k-space + virtual image, with their
    widgets) and return ``(state_k, state_vi, esm_text)``."""
    try:
        import anyplotlib as apl
        from anyplotlib import embed as apl_embed
        from spyde.signals.diffraction_vectors import (
            COL_KX, COL_KY, COL_INTENSITY,
        )
    except Exception as e:
        log.debug("[report] anyplotlib unavailable for vectors embed: %s", e)
        return None

    hdr = payload["header"]
    buf = np.asarray(vecs.flat_buffer, dtype=np.float32)
    density = _k_density_u8(buf[:, COL_KX], buf[:, COL_KY],
                            buf[:, COL_INTENSITY], hdr)

    ny, nx = hdr["nav"]
    bins = hdr["bins"]

    fig_k, ax_k = apl.subplots(1, 1, figsize=(FIG_PX, FIG_PX))
    # Fixed 0..255 levels: the page pushes fresh uint8 frames, and auto-levels
    # from the INITIAL data (all-zero for the VI) would pin display_max at 0.
    p_k = ax_k.imshow(density, cmap="gray", vmin=0, vmax=255)
    # Detector widgets in IMAGE-PIXEL coords (the widget event convention).
    # BOTH shapes are added so their exact serialized dicts exist in the panel
    # state — the page's shape radio then keeps only the active one.
    p_k.add_widget("circle", color="#e01b24",
                   cx=bins / 2, cy=bins / 2, r=bins / 8)
    p_k.add_widget("annular", color="#e01b24",
                   cx=bins / 2, cy=bins / 2,
                   r_outer=bins / 5, r_inner=bins / 10)

    fig_vi, ax_vi = apl.subplots(1, 1, figsize=(FIG_PX, FIG_PX))
    p_vi = ax_vi.imshow(np.zeros((ny, nx), dtype=np.uint8), cmap="gray",
                        vmin=0, vmax=255)
    # Real-space region selector: k-space then shows only this region's spots.
    p_vi.add_widget("rectangle", color="#33d17a",
                    x=0, y=0, w=nx, h=ny)

    state_k = apl_embed.figure_state(fig_k)
    state_vi = apl_embed.figure_state(fig_vi)

    esm = apl_embed.esm_path().read_text(encoding="utf-8")
    return state_k, state_vi, esm


_EXPLORER_CSS = """
:root { color-scheme: light; }
body { margin: 0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       background: #fff; color: #1a1a1a; font-size: 12px; }
.vx-wrap { display: flex; gap: 10px; padding: 6px; align-items: flex-start;
           flex-wrap: wrap; }
.vx-col { display: flex; flex-direction: column; gap: 4px; }
.vx-col h4 { margin: 0; font-size: 12px; font-weight: 600; color: #444; }
.vx-controls { display: flex; gap: 12px; align-items: center; font-size: 11px;
               flex-wrap: wrap; }
.vx-meta { color: #666; font-size: 11px; max-width: 700px; }
"""

# The glue module. Runs after both figures mount; owns detector/region state;
# recomputes uint8 frames and pushes them via handle.applyUpdate on the
# figure's own panel-state key. `window.__vx` is the test hook (the browser
# spec drives geometry through the SAME apply() the widget events use).
_EXPLORER_JS_TMPL = r"""
const HDR = JSON.parse(document.getElementById('vx-header').textContent);
const B64 = document.getElementById('vx-data').textContent.trim();
const STATE_K = JSON.parse(document.getElementById('vx-state-k').textContent);
const STATE_VI = JSON.parse(document.getElementById('vx-state-vi').textContent);
const ESM = document.getElementById('vx-esm').textContent;

const esmUrl = URL.createObjectURL(new Blob([ESM], { type: 'text/javascript' }));
const { mount } = await import(esmUrl);

const ab = await (await fetch('data:application/octet-stream;base64,' + B64))
  .arrayBuffer();
const n = HDR.n, navH = HDR.nav[0], navW = HDR.nav[1], BINS = HDR.bins;
let off = 0;
const X = new Uint16Array(ab, off, n); off += 2 * n;
const Y = new Uint16Array(ab, off, n); off += 2 * n;
const KX = new Float32Array(ab, off, n); off += 4 * n;
const KY = new Float32Array(ab, off, n); off += 4 * n;
const IN = new Float32Array(ab, off, n);
const kxLo = HDR.k.kx[0], kxHi = HDR.k.kx[1];
const kyLo = HDR.k.ky[0], kyHi = HDR.k.ky[1];
// image px <-> k data (density image is BINS x BINS over the k extent)
const pxToKx = (px) => kxLo + (px / (BINS - 1)) * (kxHi - kxLo);
const pxToKy = (py) => kyLo + (py / (BINS - 1)) * (kyHi - kyLo);
const kSpanX = (kxHi - kxLo) / (BINS - 1), kSpanY = (kyHi - kyLo) / (BINS - 1);

// Panel-state plumbing: find each figure's panel_<id>_json key; keep parsed
// copies we own (the widgets' source of truth lives HERE after any drag).
function panelKey(state) {
  return Object.keys(state).find((k) => /^panel_.+_json$/.test(k));
}
const keyK = panelKey(STATE_K), keyVI = panelKey(STATE_VI);
// Pixels live in the sibling _geom key (image_b64 etc. are geom-stripped from
// the light panel json — anyplotlib's zoom-perf design); widgets + display
// levels live in the light json. We patch each through its own key.
const geomKeyK = keyK.replace(/_json$/, '_geom');
const geomKeyVI = keyVI.replace(/_json$/, '_geom');
const panelK = JSON.parse(STATE_K[keyK]);
const panelVI = JSON.parse(STATE_VI[keyVI]);
const geomK = JSON.parse(STATE_K[geomKeyK]);
const geomVI = JSON.parse(STATE_VI[geomKeyVI]);
// The exact widget dicts as anyplotlib serialized them.
const circleDef = (panelK.overlay_widgets || []).find((w) => w.type === 'circle');
const annularDef = (panelK.overlay_widgets || []).find((w) => w.type === 'annular');
const rectDef = (panelVI.overlay_widgets || []).find((w) => w.type === 'rectangle');
panelK.overlay_widgets = [circleDef];              // start as a disk detector
STATE_K[keyK] = JSON.stringify(panelK);

let hK = null, hVI = null;
const det = { shape: 'circle', cx: circleDef.cx, cy: circleDef.cy,
              r: circleDef.r, rIn: annularDef.r_inner, rOut: annularDef.r_outer };
const region = { on: false, x: 0, y: 0, w: navW, h: navH };
const stats = { hit: 0, leftMean: 0, rightMean: 0 };   // test/readout mirror
const readout = document.getElementById('vx-readout');

function b64OfU8(u8) {
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  }
  return btoa(s);
}

function pushImage(handle, jsonKey, geomKey, panel, geom, u8) {
  geom.image_b64 = b64OfU8(u8);
  panel.display_min = 0; panel.display_max = 255;
  panel._geom_rev = (panel._geom_rev || 0) + 1;   // re-render insurance
  handle.applyUpdate(geomKey, JSON.stringify(geom));
  handle.applyUpdate(jsonKey, JSON.stringify(panel));
}

function inRegion(i) {
  return !region.on || (X[i] >= region.x && X[i] < region.x + region.w
                        && Y[i] >= region.y && Y[i] < region.y + region.h);
}

function computeVI() {
  const acc = new Float32Array(navW * navH);
  const sum = document.querySelector('input[name=vx-mode]:checked').value === 'sum';
  const cx = pxToKx(det.cx), cy = pxToKy(det.cy);
  const rO = (det.shape === 'circle' ? det.r : det.rOut) * kSpanX;
  const rI = det.shape === 'circle' ? 0 : det.rIn * kSpanX;
  const rO2 = rO * rO, rI2 = rI * rI;
  let hit = 0;
  for (let i = 0; i < n; i++) {
    const dx = KX[i] - cx, dy = KY[i] - cy;
    const d2 = dx * dx + dy * dy;
    if (d2 <= rO2 && d2 >= rI2) { acc[Y[i] * navW + X[i]] += sum ? IN[i] : 1; hit++; }
  }
  let m = 0;
  for (let i = 0; i < acc.length; i++) if (acc[i] > m) m = acc[i];
  const u8 = new Uint8Array(navW * navH);
  const s = m > 0 ? 255 / m : 0;
  for (let i = 0; i < acc.length; i++) u8[i] = Math.round(acc[i] * s);
  // Halved means for the readout AND the browser test hook.
  let lS = 0, rS = 0, nL = 0, nR = 0;
  for (let yy = 0; yy < navH; yy++) {
    for (let xx = 0; xx < navW; xx++) {
      const v = u8[yy * navW + xx];
      if (xx < navW / 2) { lS += v; nL++; } else { rS += v; nR++; }
    }
  }
  stats.hit = hit; stats.leftMean = lS / nL; stats.rightMean = rS / nR;
  pushImage(hVI, keyVI, geomKeyVI, panelVI, geomVI, u8);
  readout.textContent = 'detector: (' + cx.toFixed(3) + ', ' + cy.toFixed(3)
    + ') ' + (det.shape === 'circle'
        ? 'r=' + rO.toFixed(3)
        : 'r=' + rI.toFixed(3) + '..' + rO.toFixed(3))
    + (HDR.k.units ? ' ' + HDR.k.units : '') + ' — ' + hit + ' of ' + n
    + ' vectors' + (region.on ? ' (region-filtered k view)' : '');
}

function computeK() {
  const binsArr = new Float32Array(BINS * BINS);
  const sxb = (BINS - 1) / (kxHi - kxLo || 1), syb = (BINS - 1) / (kyHi - kyLo || 1);
  for (let i = 0; i < n; i++) {
    if (!inRegion(i)) continue;
    const bx = ((KX[i] - kxLo) * sxb) | 0, by = ((KY[i] - kyLo) * syb) | 0;
    if (bx >= 0 && bx < BINS && by >= 0 && by < BINS) binsArr[by * BINS + bx] += IN[i];
  }
  let mx = 0;
  for (let i = 0; i < binsArr.length; i++) if (binsArr[i] > mx) mx = binsArr[i];
  const lmax = Math.log1p(mx) || 1;
  const u8 = new Uint8Array(BINS * BINS);
  for (let i = 0; i < binsArr.length; i++) {
    u8[i] = Math.round(255 * Math.log1p(binsArr[i]) / lmax);
  }
  pushImage(hK, keyK, geomKeyK, panelK, geomK, u8);
}

let pend = 0;
function refresh(k) {           // k: also recompute the k-density (rect moved)
  pend |= k ? 3 : 1;
  requestAnimationFrame(() => {
    if (pend & 2) computeK();
    if (pend & 1) computeVI();
    pend = 0;
  });
}

// Widget-drag events carry the widget geometry (image px) spread into the
// payload — anyplotlib's own event contract.
function onKEvent(ev) {
  if (!ev.widget_id) return;
  if (ev.event_type !== 'pointer_move' && ev.event_type !== 'pointer_up') return;
  if (det.shape === 'circle') {
    if (typeof ev.cx === 'number') { det.cx = ev.cx; det.cy = ev.cy; }
    if (typeof ev.r === 'number') det.r = ev.r;
  } else {
    if (typeof ev.cx === 'number') { det.cx = ev.cx; det.cy = ev.cy; }
    if (typeof ev.r_outer === 'number') det.rOut = ev.r_outer;
    if (typeof ev.r_inner === 'number') det.rIn = ev.r_inner;
  }
  refresh(false);
}

function onVIEvent(ev) {
  if (!ev.widget_id) return;
  if (ev.event_type !== 'pointer_move' && ev.event_type !== 'pointer_up') return;
  if (typeof ev.x === 'number') {
    region.on = true;
    region.x = ev.x; region.y = ev.y; region.w = ev.w; region.h = ev.h;
    refresh(true);
  }
}

function pushDetectorWidget() {
  const w = det.shape === 'circle'
    ? Object.assign({}, circleDef, { cx: det.cx, cy: det.cy, r: det.r })
    : Object.assign({}, annularDef,
        { cx: det.cx, cy: det.cy, r_outer: det.rOut, r_inner: det.rIn });
  panelK.overlay_widgets = [w];
  hK.applyUpdate(keyK, JSON.stringify(panelK));
}

function setShape(shape) {
  det.shape = shape;
  pushDetectorWidget();
  refresh(false);
}

hK = mount(document.getElementById('vx-figk'), STATE_K, { onEvent: onKEvent });
hVI = mount(document.getElementById('vx-figvi'), STATE_VI, { onEvent: onVIEvent });

for (const rb of document.querySelectorAll('input[name=vx-shape]')) {
  rb.addEventListener('change', () => setShape(rb.value));
}
for (const rb of document.querySelectorAll('input[name=vx-mode]')) {
  rb.addEventListener('change', () => refresh(false));
}

// Test hook: geometry in IMAGE PIXELS, same code path as the widget events —
// and the on-figure widget follows, so hook-driven moves stay visible.
window.__vx = {
  setDetector(d) { Object.assign(det, d); pushDetectorWidget(); refresh(false); },
  setRegion(r) { Object.assign(region, r, { on: true }); refresh(true); },
  det, region, stats,
};

refresh(true);
document.getElementById('vx-root').dataset.ready = '1';
"""


def vectors_explorer_html(vecs, caption: str = "") -> "str | None":
    """The self-contained explorer page for one vectors dataset (goes into the
    report's sandboxed ``<iframe srcdoc>`` like any interactive figure), or
    None when the dataset is empty / over the embed cap."""
    payload = pack_vectors(vecs)
    if payload is None:
        return None
    built = _build_figures(vecs, payload)
    if built is None:
        return None
    state_k, state_vi, esm = built
    hdr = payload["header"]
    cap = _html.escape(caption or "")

    def _json_script(el_id: str, obj) -> str:
        # </script> can't appear inside a script element — escape the slash.
        txt = json.dumps(obj).replace("</", "<\\/")
        return f"<script type=\"application/json\" id=\"{el_id}\">{txt}</script>"

    esm_safe = esm.replace("</script>", "<\\/script>")
    return (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
        f"<style>{_EXPLORER_CSS}</style></head><body>"
        "<div id=\"vx-root\" class=\"vx-wrap\">"
        "<div class=\"vx-col\"><h4>Diffraction vectors — drag the detector</h4>"
        "<div id=\"vx-figk\"></div>"
        "<div class=\"vx-controls\">"
        "<span>detector:</span>"
        "<label><input type=\"radio\" name=\"vx-shape\" value=\"circle\" checked> disk</label>"
        "<label><input type=\"radio\" name=\"vx-shape\" value=\"annular\"> annulus</label>"
        "</div></div>"
        "<div class=\"vx-col\"><h4>Virtual image — drag the region to filter k-space</h4>"
        "<div id=\"vx-figvi\"></div>"
        "<div class=\"vx-controls\">"
        "<label><input type=\"radio\" name=\"vx-mode\" value=\"sum\" checked> intensity</label>"
        "<label><input type=\"radio\" name=\"vx-mode\" value=\"count\"> counts</label>"
        "</div></div>"
        f"<div id=\"vx-readout\" class=\"vx-meta\"></div>"
        f"<div class=\"vx-meta\">{cap}</div>"
        "</div>"
        f"{_json_script('vx-header', hdr)}"
        f"<script type=\"application/json\" id=\"vx-data\">{payload['b64']}</script>"
        f"{_json_script('vx-state-k', state_k)}"
        f"{_json_script('vx-state-vi', state_vi)}"
        f"<script type=\"text/plain\" id=\"vx-esm\">{esm_safe}</script>"
        f"<script type=\"module\">{_EXPLORER_JS_TMPL}</script>"
        "</body></html>"
    )


def vectors_for_cell(session, cell) -> "object | None":
    """The SpyDEDiffractionVectors behind a figure cell's base layer, or None.
    Resolution goes through the cell spec's SignalRef → live plot → tree."""
    spec = getattr(cell, "spec", None)
    if spec is None or not getattr(spec, "panels", None):
        return None
    try:
        layers = spec.panels[0].layers
        if not layers:
            return None
        plot = layers[0].source.resolve(session)
        if plot is None:
            return None
        tree = getattr(plot, "signal_tree", None)
        return getattr(tree, "diffraction_vectors", None)
    except Exception as e:
        log.debug("[report] vectors resolve for cell failed: %s", e)
        return None
