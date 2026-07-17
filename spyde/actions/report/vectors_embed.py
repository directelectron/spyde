"""vectors_embed.py — embed a FULL diffraction-vectors dataset in an HTML report.

A find-vectors result is a compact CSR flat buffer (per vector: nav x/y, kx,
ky, intensity) — small enough to inline into a self-contained HTML page and
rich enough to re-render the DIFFRACTION PATTERN at any nav position entirely
client-side. The explorer mirrors the MDI app's vector view: ONE anyplotlib
figure with TWO panels — a NAVIGATOR (count map) driving a DIFFRACTION-PATTERN
panel that shows the pointed position's vectors as intensity DISKS.

- Navigator panel: the count map, with a draggable **crosshair** (POINTER
  mode — one nav position) and a **rectangle** (INTEGRATE mode — a nav region
  whose disk patterns are SUMMED). A mode radio swaps which is active.
- DP panel: the rendered disks for the current pointer / region, recomputed
  live in the page. Only POINTS travel to JS — the disks are rasterised in the
  browser (mirroring ``_render_disks_block`` / ``render_region``), so no
  rendered frames are ever shipped.

Because it's a single figure/single ``mount()``, it renders in the report
SIDEBAR for free via the SeamlessFigureFrame (the old two-figure layout showed
only a plain snapshot there). The glue rides anyplotlib's standalone machinery
(``mount(el, state, {onEvent})`` + ``handle.applyUpdate``): widget drags arrive
as ``pointer_move/up`` events carrying the widget geometry in IMAGE PIXELS (==
nav index on the nav-shaped panel); the script re-renders the DP frame and
pushes it via an overlay canvas over the DP panel. No backend, no network — the
single .html file works years later.

Packing (little-endian, one base64 blob) — see :func:`pack_vectors` for the
byte offsets:
    uint16  x[n], y[n]                 nav column / row index
    float32 kx[n], ky[n], intensity[n]
    uint32  nav_off[ny*nx+1]           OPTIONAL run-length index (O(1) slice)
16 bytes/vector (+ the small offset tail) → ~1 M vectors ≈ 16 MB (≈21 MB as
base64). Above ``MAX_EMBED_VECTORS`` the embed is refused (returns None) and the
export falls back to the baked static image.

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


def _nav_offsets_flat(vecs, ny: int, nx: int, n: int) -> "np.ndarray | None":
    """A ``(ny*nx + 1,)`` uint32 run-length index over the packed point block so
    the page can O(1)-slice one nav position's vectors (pointer mode) instead of
    scanning all ``n``.

    The point block is packed in ``flat_buffer`` row order (sorted
    outermost-nav-first). For a 4-D scan that order groups each ``(iy, ix)``
    position contiguously, so ``nav_offsets[-1]`` (or the legacy spatial
    ``offsets``) IS the run-length index directly. Returns None when a valid
    contiguous per-position index can't be produced (e.g. a 5-D stack, where one
    ``(iy, ix)`` position's vectors are spread across time steps) — the page then
    falls back to a full scan, still correct, just O(n)."""
    want = ny * nx + 1
    # 5-D stacks (an outer nav dim above the 2-D scan) aren't contiguous per
    # (iy, ix) — bail so the page scans instead of mis-slicing.
    full = tuple(int(s) for s in getattr(vecs, "full_nav_shape", vecs.nav_shape))
    if len(full) > 2:
        return None
    for cand in (getattr(vecs, "nav_offsets", None), None):
        arr = None
        if cand is not None and len(cand):
            arr = np.asarray(cand[-1])
        if arr is not None and arr.shape[0] == want:
            off = np.ascontiguousarray(arr, dtype="<u4")
            # Sanity: monotonic and ends at n (the full point count).
            if int(off[0]) == 0 and int(off[-1]) == n and np.all(np.diff(off) >= 0):
                return off
    legacy = getattr(vecs, "offsets", None)
    if legacy is not None and np.asarray(legacy).shape[0] == want:
        off = np.ascontiguousarray(np.asarray(legacy), dtype="<u4")
        if int(off[0]) == 0 and int(off[-1]) == n and np.all(np.diff(off) >= 0):
            return off
    return None


def pack_vectors(vecs) -> "dict | None":
    """Pack a SpyDEDiffractionVectors into the embed payload
    ``{header: dict, b64: str}`` — or None when too large / empty.

    Blob layout (little-endian, one base64 string), byte offsets for ``n``
    vectors and (when present) a ``ny*nx+1`` run-length index::

        uint16  x[n]         nav column index          bytes [0,           2n)
        uint16  y[n]         nav row index                   [2n,          4n)
        float32 kx[n]                                        [4n,          8n)
        float32 ky[n]                                        [8n,         12n)
        float32 intensity[n]                                 [12n,        16n)
        uint32  nav_off[m]   m = ny*nx+1 (OPTIONAL)          [16n,  16n + 4m)

    The 16-byte/vector point block is UNCHANGED; ``nav_off`` (header key
    ``nav_off`` = its length ``m``, absent when the index couldn't be built) is
    appended after it so old readers that stop at ``16n`` still work."""
    from spyde.signals.diffraction_vectors import COL_KX, COL_KY, COL_INTENSITY

    buf = np.asarray(vecs.flat_buffer, dtype=np.float32)
    n = int(buf.shape[0])
    if n == 0 or n > MAX_EMBED_VECTORS:
        if n:
            log.info("[report] vectors embed skipped: %d vectors > cap %d",
                     n, MAX_EMBED_VECTORS)
        return None

    nav_shape = tuple(int(s) for s in vecs.nav_shape)          # (ny, nx)
    ny, nx = nav_shape[-2], nav_shape[-1]
    x = buf[:, 0].astype(np.uint16)
    y = buf[:, 1].astype(np.uint16)
    kx = np.ascontiguousarray(buf[:, COL_KX], dtype="<f4")
    ky = np.ascontiguousarray(buf[:, COL_KY], dtype="<f4")
    inten = np.ascontiguousarray(buf[:, COL_INTENSITY], dtype="<f4")

    blob = (x.astype("<u2").tobytes() + y.astype("<u2").tobytes()
            + kx.tobytes() + ky.tobytes() + inten.tobytes())

    nav_off = _nav_offsets_flat(vecs, ny, nx, n)
    nav_off_len = int(nav_off.shape[0]) if nav_off is not None else 0
    if nav_off is not None:
        blob += nav_off.tobytes()

    sig_axes = list(getattr(vecs, "sig_axes", []) or [])
    if len(sig_axes) >= 2:
        kx_lo, kx_hi = _axis_extent(sig_axes[0])
        ky_lo, ky_hi = _axis_extent(sig_axes[1])
        # DP frame pixel dims: axis 0 (kx) ↔ width (columns), axis 1 (ky) ↔ height.
        sig_w = int(getattr(sig_axes[0], "size", 0) or 0)
        sig_h = int(getattr(sig_axes[1], "size", 0) or 0)
    else:   # uncalibrated fallback: span the data
        kx_lo, kx_hi = float(kx.min()), float(kx.max())
        ky_lo, ky_hi = float(ky.min()), float(ky.max())
        sig_w = sig_h = DENSITY_BINS
    units = str(getattr(sig_axes[0], "units", "") or "") if sig_axes else ""

    header = {
        "n": n,
        "nav": [ny, nx],
        "sig": [sig_h, sig_w],          # DP frame [H, W] in pixels
        "k": {"kx": [kx_lo, kx_hi], "ky": [ky_lo, ky_hi], "units": units},
        "bins": DENSITY_BINS,
        "r0": float(getattr(vecs, "kernel_radius_data", 0.0) or 0.0),
        "r_px": float(getattr(vecs, "kernel_radius_px", 0.0) or 0.0),
        "nav_off": nav_off_len,         # 0 = no run-length index (scan instead)
    }
    return {"header": header, "b64": base64.b64encode(blob).decode("ascii")}


def _count_map_u8(vecs, ny: int, nx: int) -> np.ndarray:
    """Log-scaled uint8 count-map image — the navigator background (mirrors the
    MDI vector window's count-map navigator)."""
    try:
        cm = np.asarray(vecs.count_map(), dtype=np.float32)
    except Exception as e:
        log.debug("[report] count_map failed, using zeros nav: %s", e)
        cm = np.zeros((ny, nx), dtype=np.float32)
    if cm.shape != (ny, nx):
        cm = cm.reshape(ny, nx) if cm.size == ny * nx else np.zeros((ny, nx), np.float32)
    img = np.log1p(cm)
    mx = float(img.max()) or 1.0
    return (255 * img / mx).astype(np.uint8)


def _build_figure(vecs, payload) -> "tuple[dict, str, str, str] | None":
    """Build ONE anyplotlib figure with TWO panels — navigator | DP — and return
    ``(state, nav_panel_id, dp_panel_id, esm_text)``.

    A single figure/single ``mount()`` renders in the report SIDEBAR for free via
    the SeamlessFigureFrame (the previous two-figure layout showed only a plain
    snapshot there). The navigator (panel 0) shows the count map with a draggable
    CROSSHAIR (pointer mode) and a RECTANGLE (integrate mode); the DP panel
    (panel 1) starts as zeros and is repainted client-side via the overlay-canvas
    push. Panel ids are discovered from the serialized state (the state dict does
    NOT guarantee navigator-first ordering), keyed by which panel carries the
    widgets and by image size."""
    try:
        import json as _json
        import anyplotlib as apl
        from anyplotlib import embed as apl_embed
    except Exception as e:
        log.debug("[report] anyplotlib unavailable for vectors embed: %s", e)
        return None

    hdr = payload["header"]
    ny, nx = hdr["nav"]
    H, W = hdr["sig"]
    r_px = max(1.0, float(hdr.get("r_px", 1.0)))

    nav_u8 = _count_map_u8(vecs, ny, nx)

    fig, axs = apl.subplots(1, 2, figsize=(2 * FIG_PX + 20, FIG_PX))
    ax_nav, ax_dp = axs[0], axs[1]

    # NAVIGATOR (panel 0): count map + crosshair (pointer) + rectangle
    # (integrate). BOTH widgets are serialized so their exact dicts exist; the
    # page's mode radio keeps only the active one visible.
    p_nav = ax_nav.imshow(nav_u8, cmap="gray", vmin=0, vmax=255)
    p_nav.add_widget("crosshair", color="#f6c177",
                     cx=nx / 2, cy=ny / 2)
    p_nav.add_widget("rectangle", color="#89b4fa",
                     x=max(0, nx / 2 - 2), y=max(0, ny / 2 - 2),
                     w=min(nx, 4), h=min(ny, 4))

    # DP (panel 1): fixed 0..255 levels — the page pushes fresh uint8 frames, and
    # auto-levels from the initial all-zero frame would pin display_max at 0.
    ax_dp.imshow(np.zeros((H, W), dtype=np.uint8), cmap="gray", vmin=0, vmax=255)

    state = apl_embed.figure_state(fig)

    # Identify which panel_<id>_json is the navigator (carries the widgets) vs
    # the DP (matches the DP image size / no widgets).
    nav_id = dp_id = None
    for k in state:
        if not (k.startswith("panel_") and k.endswith("_json")):
            continue
        pid = k[len("panel_"):-len("_json")]
        try:
            pj = _json.loads(state[k])
        except Exception:
            continue
        ws = pj.get("overlay_widgets") or []
        if any(w.get("type") in ("crosshair", "rectangle") for w in ws):
            nav_id = pid
        else:
            dp_id = pid
    if nav_id is None or dp_id is None:
        log.debug("[report] could not identify nav/DP panels for vectors embed")
        return None

    esm = apl_embed.esm_path().read_text(encoding="utf-8")
    return state, nav_id, dp_id, esm


_EXPLORER_CSS = """
:root { color-scheme: light; }
body { margin: 0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       background: #fff; color: #1a1a1a; font-size: 12px; }
.vx-wrap { display: flex; flex-direction: column; gap: 6px; padding: 6px;
           align-items: flex-start; }
.vx-col h4 { margin: 0; font-size: 12px; font-weight: 600; color: #444; }
.vx-controls { display: flex; gap: 12px; align-items: center; font-size: 11px;
               flex-wrap: wrap; }
.vx-meta { color: #666; font-size: 11px; max-width: 720px; }
#vx-fig { width: 100%; }
"""

# The glue module. Runs after the single 2-panel figure mounts; owns the
# pointer crosshair / integrate rectangle state on the NAVIGATOR panel, renders
# the DP frame client-side (points → disks, mirroring _render_disks_block /
# render_region), and pushes it via an overlay canvas over the DP panel.
# `window.__vx` is the test hook (browser spec drives geometry through the SAME
# code path the widget events use).
_EXPLORER_JS_TMPL = r"""
const HDR = JSON.parse(document.getElementById('vx-header').textContent);
const B64 = document.getElementById('vx-data').textContent.trim();
const STATE = JSON.parse(document.getElementById('vx-state').textContent);
const NAV_ID = document.getElementById('vx-navid').textContent.trim();
const DP_ID = document.getElementById('vx-dpid').textContent.trim();
const ESM = document.getElementById('vx-esm').textContent;

const esmUrl = URL.createObjectURL(new Blob([ESM], { type: 'text/javascript' }));
const { mount } = await import(esmUrl);

const n = HDR.n, navH = HDR.nav[0], navW = HDR.nav[1];
const DPH = HDR.sig[0], DPW = HDR.sig[1];              // DP frame [H, W] in px
const R_PX = Math.max(1, Math.round(HDR.r_px || 1));   // disk radius (px)
const NAV_OFF_LEN = HDR.nav_off || 0;                  // 0 = no run-length index

const ab = await (await fetch('data:application/octet-stream;base64,' + B64))
  .arrayBuffer();
let off = 0;
const X = new Uint16Array(ab, off, n); off += 2 * n;
const Y = new Uint16Array(ab, off, n); off += 2 * n;
const KX = new Float32Array(ab, off, n); off += 4 * n;
const KY = new Float32Array(ab, off, n); off += 4 * n;
const IN = new Float32Array(ab, off, n); off += 4 * n;
// Optional uint32 run-length index (ny*nx+1): NAV_OFF[iy*navW+ix] .. +1 bound a
// position's rows in the point block. Absent → scan (still correct, O(n)).
const NAV_OFF = NAV_OFF_LEN ? new Uint32Array(ab, off, NAV_OFF_LEN) : null;

const kxLo = HDR.k.kx[0], kxHi = HDR.k.kx[1];
const kyLo = HDR.k.ky[0], kyHi = HDR.k.ky[1];
// kx/ky (calibrated) → DP pixel. Mirrors _render_disks_block: axis 0 (kx) ↔
// column (width), axis 1 (ky) ↔ row (height). Extents span the frame edges.
const kSpanX = (kxHi - kxLo) / (DPW - 1 || 1);
const kSpanY = (kyHi - kyLo) / (DPH - 1 || 1);
const kxToCol = (kx) => Math.round((kx - kxLo) / (kSpanX || 1));
const kyToRow = (ky) => Math.round((ky - kyLo) / (kSpanY || 1));

// The single mount handle + panel objects (looked up by id from the render API).
let H = null, navPanel = null, dpPanel = null;

// The DP repaint lands on a PASS-THROUGH OVERLAY canvas over the DP panel's
// image rect. Pushing pixels through the figure's own state keys proved
// undebuggable in the standalone shim (bytes accepted, blit cache rebuilt, a
// correct drawImage even observed — frame still black); a plain canvas
// drawImage on top is a primitive we verified directly. The anyplotlib figure
// keeps the widget interactions, axes and theme. (Trade-off: the overlay tracks
// the UNZOOMED fit rect, so figure zoom/pan is a no-op on the DP — fine here.)
function makeOverlay(panel, iw, ih) {
  const host = panel.plotCanvas.parentElement;
  const scale = Math.min(panel.imgW / iw, panel.imgH / ih);
  const w = iw * scale, h = ih * scale;
  const x = (panel.imgW - w) / 2, y = (panel.imgH - h) / 2;
  const c = document.createElement('canvas');
  c.width = iw; c.height = ih;
  // z-index 2: above the plot canvas (z 1, opaque), below markers/widgets.
  c.style.cssText = 'position:absolute;pointer-events:none;z-index:2;'
    + 'image-rendering:pixelated;'
    + `left:${panel.plotCanvas.offsetLeft + x}px;`
    + `top:${panel.plotCanvas.offsetTop + y}px;`
    + `width:${w}px;height:${h}px;`;
  if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
  host.insertBefore(c, panel.plotCanvas.nextSibling);
  return c.getContext('2d');
}

function pushImage(ctx, u8, iw, ih) {
  const img = ctx.createImageData(iw, ih);
  const d = img.data;
  for (let i = 0; i < u8.length; i++) {
    const v = u8[i];
    d[4 * i] = v; d[4 * i + 1] = v; d[4 * i + 2] = v; d[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// Row range [s, e) of the vectors at nav (iy, ix): O(1) via the run-length
// index, else a full scan collecting matching rows.
function rowsAt(iy, ix, out) {
  out.length = 0;
  if (iy < 0 || iy >= navH || ix < 0 || ix >= navW) return out;
  if (NAV_OFF) {
    const p = iy * navW + ix;
    const s = NAV_OFF[p], e = NAV_OFF[p + 1];
    for (let i = s; i < e; i++) out.push(i);
    return out;
  }
  for (let i = 0; i < n; i++) if (Y[i] === iy && X[i] === ix) out.push(i);
  return out;
}

// Splat one position's vectors into a scratch DP frame with INTRA-frame MAX
// (mirror _render_disks_block: disk mask (dy^2+dx^2)<=r^2, value=intensity,
// overlapping disks keep the max). Additive into `acc` gives render_region's
// max-then-sum. `scratch` is reused (zeroed) per position.
function splatPosition(rowIdx, acc, scratch) {
  const r = R_PX, r2 = r * r;
  // Zero only the touched pixels — but simplest correct: track a dirty list.
  const dirty = [];
  for (const i of rowIdx) {
    const cc = kxToCol(KX[i]), cr = kyToRow(KY[i]);
    const inten = IN[i];
    for (let dy = -r; dy <= r; dy++) {
      const ry = cr + dy;
      if (ry < 0 || ry >= DPH) continue;
      const dy2 = dy * dy;
      for (let dx = -r; dx <= r; dx++) {
        if (dy2 + dx * dx > r2) continue;
        const rx = cc + dx;
        if (rx < 0 || rx >= DPW) continue;
        const p = ry * DPW + rx;
        if (inten > scratch[p]) { if (scratch[p] === 0) dirty.push(p); scratch[p] = inten; }
      }
    }
  }
  for (const p of dirty) { acc[p] += scratch[p]; scratch[p] = 0; }
}

const mode = { integrate: false };
const cross = { ix: (navW / 2) | 0, iy: (navH / 2) | 0 };
const region = { x: 0, y: 0, w: navW, h: navH };
const stats = { hit: 0, leftMean: 0, rightMean: 0, max: 0 };  // test/readout mirror
const readout = document.getElementById('vx-readout');
let ovDP = null;

function computeDP() {
  const acc = new Float32Array(DPW * DPH);
  const scratch = new Float32Array(DPW * DPH);
  const idx = [];
  let y0, y1, x0, x1;
  if (mode.integrate) {
    x0 = Math.max(0, Math.round(region.x));
    y0 = Math.max(0, Math.round(region.y));
    x1 = Math.min(navW, Math.round(region.x + region.w));
    y1 = Math.min(navH, Math.round(region.y + region.h));
    if (x1 <= x0) x1 = Math.min(navW, x0 + 1);
    if (y1 <= y0) y1 = Math.min(navH, y0 + 1);
  } else {
    x0 = Math.max(0, Math.min(navW - 1, cross.ix)); x1 = x0 + 1;
    y0 = Math.max(0, Math.min(navH - 1, cross.iy)); y1 = y0 + 1;
  }
  let hit = 0;
  for (let iy = y0; iy < y1; iy++) {
    for (let ix = x0; ix < x1; ix++) {
      rowsAt(iy, ix, idx);
      hit += idx.length;
      splatPosition(idx, acc, scratch);
    }
  }
  let m = 0;
  for (let i = 0; i < acc.length; i++) if (acc[i] > m) m = acc[i];
  const u8 = new Uint8Array(DPW * DPH);
  const s = m > 0 ? 255 / m : 0;
  for (let i = 0; i < acc.length; i++) u8[i] = Math.round(acc[i] * s);
  stats.hit = hit; stats.max = m;
  // Left/right halved means over the DP frame (test hook parity).
  let lS = 0, rS = 0, nL = 0, nR = 0;
  for (let ry = 0; ry < DPH; ry++) {
    for (let rx = 0; rx < DPW; rx++) {
      const v = u8[ry * DPW + rx];
      if (rx < DPW / 2) { lS += v; nL++; } else { rS += v; nR++; }
    }
  }
  stats.leftMean = lS / (nL || 1); stats.rightMean = rS / (nR || 1);
  if (ovDP) pushImage(ovDP, u8, DPW, DPH);
  readout.textContent = mode.integrate
    ? ('integrate: nav [' + y0 + ':' + y1 + ', ' + x0 + ':' + x1 + '] — '
       + hit + ' vectors summed')
    : ('pointer: nav (' + y0 + ', ' + x0 + ') — ' + hit + ' vectors');
}

let pend = false;
function refresh() {
  if (pend) return;
  pend = true;
  requestAnimationFrame(() => { pend = false; computeDP(); });
}

// Navigator widget-drag events carry the widget geometry (image px == nav
// index on the nav-shaped panel) spread into the payload. The crosshair drives
// pointer mode; the rectangle drives integrate mode — the active MODE decides
// which geometry the DP uses, and any drag on the matching widget refreshes.
function onNavEvent(ev) {
  if (ev.event_type !== 'pointer_move' && ev.event_type !== 'pointer_up') return;
  if (ev.type === 'crosshair' || typeof ev.cx === 'number') {
    if (typeof ev.cx === 'number') { cross.ix = Math.round(ev.cx); cross.iy = Math.round(ev.cy); }
    if (!mode.integrate) refresh();
  }
  if (ev.type === 'rectangle' || typeof ev.w === 'number') {
    if (typeof ev.x === 'number') {
      region.x = ev.x; region.y = ev.y; region.w = ev.w; region.h = ev.h;
    }
    if (mode.integrate) refresh();
  }
}

function setMode(integrate) {
  mode.integrate = integrate;
  // Show only the active widget on the navigator.
  const live = liveNav();
  const want = integrate ? 'rectangle' : 'crosshair';
  live.overlay_widgets = (allWidgets || []).filter((w) => w.type === want);
  H.applyUpdate(navKey, JSON.stringify(live));
  refresh();
}

// Navigator panel-state plumbing — patch through the LIVE model state
// (handle.get), never the page-load parse (a stale json apply resets the
// fitted view/layout → 0-width canvas).
const navKey = 'panel_' + NAV_ID + '_json';
let allWidgets = null;
function liveNav() {
  try { return JSON.parse(H.get(navKey)); }
  catch (e) { return JSON.parse(STATE[navKey]); }
}

H = mount(document.getElementById('vx-fig'), STATE, { onEvent: onNavEvent });
// Capture the exact serialized widget dicts, then start in pointer mode
// (crosshair only).
allWidgets = (JSON.parse(STATE[navKey]).overlay_widgets || []).map((w) => Object.assign({}, w));
// Give the layout a beat to settle so the overlay measures the final DP rect.
await new Promise((res) => requestAnimationFrame(() => setTimeout(res, 30)));
navPanel = H.api.panels.get(NAV_ID);
dpPanel = H.api.panels.get(DP_ID);
if (dpPanel) ovDP = makeOverlay(dpPanel, DPW, DPH);
setMode(false);

// AUTO-FIT: the figure is built at a FIXED native size (~2:1, ~700px wide). In
// a WIDE container (the HTML export's ~736px article iframe) it fits natively;
// in a NARROW one (the report SIDEBAR cell, ~390px) it would overflow — and
// anyplotlib's own cell-scaling doesn't engage in this standalone embed. So
// scale the mounted figure's outer element down with a CSS transform to fit the
// container width (never up past 1). The overlay canvas lives inside this outer
// element, so it scales in lockstep and stays registered on the DP panel. A
// negative marginBottom pulls the controls up under the shrunk figure (scale
// doesn't affect layout height). transform-origin is top-left.
function fitFigure() {
  const host = document.getElementById('vx-fig');
  const outer = host && host.firstElementChild;
  if (!outer) return;
  outer.style.transformOrigin = 'top left';
  // The figure's grid is a FIXED-px layout (fig_width/fig_height from the model);
  // the outer element itself gets width-constrained by the container (its
  // offsetWidth == container width, NOT max-content, in this standalone embed),
  // so the fixed grid OVERFLOWS + clips rather than reporting a wide offsetWidth.
  // Scale off the TRUE native size (the model's fig_width) — stable across fires
  // (no feedback loop, it never changes) — falling back to scrollWidth.
  const nativeW = Number(H.get('fig_width')) || outer.scrollWidth || outer.offsetWidth;
  const nativeH = Number(H.get('fig_height')) || outer.scrollHeight || outer.offsetHeight;
  const avail = host.clientWidth;
  if (!nativeW || !avail) return;
  const s = Math.min(1, avail / nativeW);
  outer.style.transform = s < 1 ? ('scale(' + s + ')') : '';
  // scale() doesn't change layout height — the outer still occupies nativeH in
  // flow. When shrunk, pull the controls up by the removed height; also cap the
  // host height so the shrunk figure doesn't leave a tall empty gap.
  outer.style.marginBottom = (s < 1 && nativeH)
    ? Math.round(nativeH * (s - 1)) + 'px' : '';
}
fitFigure();
if (typeof ResizeObserver !== 'undefined') {
  let ft = null;
  new ResizeObserver(() => {
    if (ft) cancelAnimationFrame(ft);
    ft = requestAnimationFrame(fitFigure);
  }).observe(document.getElementById('vx-fig'));
}

// TOUCH → MOUSE shim: anyplotlib's widget drags listen to mouse events only,
// so phone touches would dead-end. Re-dispatch single-finger touches on the
// figure container as synthetic mouse events (and stop the page scrolling while
// dragging a widget); two-finger gestures pass through untouched.
{
  const el = document.getElementById('vx-fig');
  const relay = (type) => (ev) => {
    if (ev.touches && ev.touches.length > 1) return;
    const t = ev.touches[0] || ev.changedTouches[0];
    if (!t) return;
    ev.preventDefault();
    const target = document.elementFromPoint(t.clientX, t.clientY) || el;
    (type === 'mousemove' ? document : target).dispatchEvent(
      new MouseEvent(type, {
        bubbles: true, cancelable: true, view: window,
        clientX: t.clientX, clientY: t.clientY, button: 0, buttons: 1,
      }));
  };
  el.addEventListener('touchstart', relay('mousedown'), { passive: false });
  el.addEventListener('touchmove', relay('mousemove'), { passive: false });
  el.addEventListener('touchend', relay('mouseup'), { passive: false });
}

for (const rb of document.querySelectorAll('input[name=vx-mode]')) {
  rb.addEventListener('change', () => setMode(rb.value === 'integrate'));
}

// Test hook: nav geometry in IMAGE PIXELS (== nav index), same code path as the
// widget events. setPointer moves the crosshair; setRegion moves the rectangle.
window.__vx = {
  setPointer(p) {
    if (typeof p.ix === 'number') { cross.ix = p.ix; cross.iy = p.iy; }
    if (typeof p.cx === 'number') { cross.ix = Math.round(p.cx); cross.iy = Math.round(p.cy); }
    if (!mode.integrate) refresh();
  },
  setRegion(r) { Object.assign(region, r); if (mode.integrate) refresh(); },
  setMode,
  cross, region, mode, stats,
  _h: () => ({ H, navKey, NAV_ID, DP_ID, navPanel, dpPanel }),
};

refresh();
document.getElementById('vx-root').dataset.ready = '1';
"""


def vectors_explorer_html(vecs, caption: str = "") -> "str | None":
    """The self-contained explorer page for one vectors dataset (goes into the
    report's sandboxed ``<iframe srcdoc>`` like any interactive figure), or
    None when the dataset is empty / over the embed cap.

    ONE anyplotlib figure with TWO panels (navigator | DP): the navigator's
    crosshair (pointer mode) / rectangle (integrate mode) drives a client-side
    disk render of the diffraction pattern — points, not frames, are embedded."""
    payload = pack_vectors(vecs)
    if payload is None:
        return None
    built = _build_figure(vecs, payload)
    if built is None:
        return None
    state, nav_id, dp_id, esm = built
    hdr = payload["header"]
    cap = _html.escape(caption or "")

    def _json_script(el_id: str, obj) -> str:
        # </script> can't appear inside a script element — escape the slash.
        txt = json.dumps(obj).replace("</", "<\\/")
        return f"<script type=\"application/json\" id=\"{el_id}\">{txt}</script>"

    esm_safe = esm.replace("</script>", "<\\/script>")
    return (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<style>{_EXPLORER_CSS}</style></head><body>"
        "<div id=\"vx-root\" class=\"vx-wrap\">"
        "<h4>Diffraction vectors — navigator (left) drives the pattern (right)</h4>"
        "<div id=\"vx-fig\"></div>"
        "<div class=\"vx-controls\">"
        "<span>mode:</span>"
        "<label><input type=\"radio\" name=\"vx-mode\" value=\"pointer\" checked>"
        " pointer</label>"
        "<label><input type=\"radio\" name=\"vx-mode\" value=\"integrate\">"
        " integrate (sum region)</label>"
        "</div>"
        f"<div id=\"vx-readout\" class=\"vx-meta\"></div>"
        f"<div class=\"vx-meta\">{cap}</div>"
        "</div>"
        f"{_json_script('vx-header', hdr)}"
        f"<script type=\"application/json\" id=\"vx-data\">{payload['b64']}</script>"
        f"{_json_script('vx-state', state)}"
        f"<script type=\"text/plain\" id=\"vx-navid\">{nav_id}</script>"
        f"<script type=\"text/plain\" id=\"vx-dpid\">{dp_id}</script>"
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
