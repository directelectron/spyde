"""embed_export.py — the PNG-harvest contract for SpyDE's bespoke embed pages.

A report cell's static export (article HTML, PDF, the saved zip's
``assets/<id>.png``) comes from a PNG the RENDERER harvests out of the live
figure: ``requestFigurePng`` posts ``anyplotlib_export_png`` into the cell's
iframe and waits for ``anyplotlib_export_png_result``.

anyplotlib's own standalone page answers that message. The pages SpyDE builds by
hand — the diffraction-vectors explorer, the orientation explorer, the tinted-
overlay blender — call ``mount()`` (or no anyplotlib at all) and so never
installed the listener. The harvest therefore timed out, and the static export
fell back to an Agg bake of the base image: a DIFFERENT picture from the one on
screen, at 1.5 s of the 3 s snapshot budget per cell.

This module carries the shared half. A page inlines :data:`EXPORT_PNG_JS` and
defines ``globalThis.__spydeExportPng(opts)``; the helpers here do the
compositing, because on every one of these pages the interesting pixels are on
canvases anyplotlib does not own:

* the explorers paint recomputed frames onto pass-through OVERLAY canvases
  layered over the plot canvas (pushing them through the figure's state ends up
  black — see vectors_embed's module docstring), so ``handle.exportPNG()`` alone
  returns the placeholder background;
* the blender has no anyplotlib runtime at all, just a grid of plain canvases.

Both cases are the same operation: draw a set of canvases into one output,
positioned by their on-screen box relative to a root element — which is exactly
how anyplotlib's own ``exportPNG`` composites, so the two agree.
"""
from __future__ import annotations

# The protocol shim + the two compositors. Inserted verbatim into a page's
# <script type="module"> — no .format(), so JS braces need no escaping.
EXPORT_PNG_JS = r"""
// ── PNG export protocol (SpyDE report harvest) ───────────────────────────────
// Mirrors anyplotlib's standalone-page contract exactly:
//   → { type: 'anyplotlib_export_png', requestId, opts }
//   ← { type: 'anyplotlib_export_png_result', requestId, dataUrl, width, height }
//   ← { type: 'anyplotlib_export_png_result', requestId, error }
// The page supplies globalThis.__spydeExportPng(opts) -> Promise<{dataUrl,…}>.
window.addEventListener('message', (e) => {
  if (!e.data || e.data.type !== 'anyplotlib_export_png') return;
  const requestId = e.data.requestId;
  const source = e.source;
  const reply = (msg) => {
    try {
      if (source && typeof source.postMessage === 'function') {
        source.postMessage(Object.assign(
          { type: 'anyplotlib_export_png_result', requestId }, msg), '*');
      }
    } catch (_) {}
  };
  try {
    const fn = globalThis.__spydeExportPng;
    if (typeof fn !== 'function') {
      reply({ error: 'page not ready (__spydeExportPng undefined)' });
      return;
    }
    Promise.resolve(fn(e.data.opts || {}))
      .then((res) => reply({
        dataUrl: res.dataUrl, width: res.width, height: res.height }))
      .catch((err) => reply({ error: String((err && err.message) || err) }));
  } catch (err) {
    reply({ error: String((err && err.message) || err) });
  }
});

// Draw `canvases` into `out`, each at its on-screen box relative to `rootRect`,
// scaled by `k`. Same geometry anyplotlib's exportPNG uses, so an overlay lands
// exactly where it sits on screen. Deriving `k` from the MEASURED root width
// (rather than devicePixelRatio * opts.scale) keeps this correct when the page
// has CSS-transform-scaled the figure to fit its container, which the explorers
// do.
function __spydeDrawCanvases(out, rootRect, canvases, k) {
  const ctx = out.getContext('2d');
  if (!ctx) return;
  ctx.imageSmoothingEnabled = false;
  for (const c of canvases) {
    if (!c || !c.width || !c.height) continue;
    if (c.style && c.style.display === 'none') continue;
    const r = c.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    const dx = Math.round((r.left - rootRect.left) * k);
    const dy = Math.round((r.top - rootRect.top) * k);
    const dw = Math.round((r.right - rootRect.left) * k) - dx;
    const dh = Math.round((r.bottom - rootRect.top) * k) - dy;
    try { ctx.drawImage(c, dx, dy, dw, dh); } catch (_) {}
  }
}

// An anyplotlib figure (via its mount handle) PLUS the page's own overlay
// canvases. The figure exports itself first — that is the axes, ticks, colorbar,
// widgets and panel backgrounds — then the overlays go on top.
//
// `root` must be the element anyplotlib measures its own export against: the
// figure background div, reachable from a figure-level overlay canvas. Falling
// back to the mount container is close enough to keep the export usable if that
// internal ever moves.
async function __spydeExportFigureWithOverlays(handle, mountEl, canvases, opts) {
  const res = await handle.exportPNG(Object.assign({ includeWidgets: true }, opts || {}));
  const api = handle.api || {};
  const root = (api.figMarkerCanvas && api.figMarkerCanvas.parentElement)
    || (api.calloutCanvas && api.calloutCanvas.parentElement)
    || mountEl;
  const live = (canvases || []).filter(Boolean);
  if (!live.length) return res;
  const img = await __spydeLoadImage(res.dataUrl);
  const out = document.createElement('canvas');
  out.width = res.width; out.height = res.height;
  const ctx = out.getContext('2d');
  if (!ctx) return res;
  ctx.drawImage(img, 0, 0);
  const rootRect = root.getBoundingClientRect();
  const k = rootRect.width ? (res.width / rootRect.width) : 1;
  __spydeDrawCanvases(out, rootRect, live, k);
  return { dataUrl: out.toDataURL('image/png'), width: out.width, height: out.height };
}

// A page with NO anyplotlib figure — composite plain canvases against a
// container element, at the container's own on-screen size times `scale`.
function __spydeExportCanvasGrid(root, canvases, opts) {
  const o = opts || {};
  const scale = (o.scale != null && o.scale > 0) ? o.scale : 1;
  const k = (window.devicePixelRatio || 1) * scale;
  const rootRect = root.getBoundingClientRect();
  const out = document.createElement('canvas');
  out.width = Math.max(1, Math.round(rootRect.width * k));
  out.height = Math.max(1, Math.round(rootRect.height * k));
  const ctx = out.getContext('2d');
  if (!ctx) return Promise.reject(new Error('export: 2D context unavailable'));
  // Paint the page background first so transparent gaps do not export black.
  const bg = getComputedStyle(root).backgroundColor;
  ctx.fillStyle = (bg && bg !== 'rgba(0, 0, 0, 0)') ? bg : '#ffffff';
  ctx.fillRect(0, 0, out.width, out.height);
  __spydeDrawCanvases(out, rootRect, (canvases || []).filter(Boolean), k);
  return Promise.resolve({
    dataUrl: out.toDataURL('image/png'), width: out.width, height: out.height });
}

// Tell the host page how tall this embed actually is, so it can size the iframe
// instead of guessing. A report embeds these in a fixed-height frame, which
// leaves a dead band under the controls — worst on a phone, where the figure
// scales down and the gap is most of the screen. `vxHeight` is the original key
// (the vectors explorer's, which external scripts already read); the neutral one
// is what new hosts should listen for. Harmless where nobody listens.
function __spydeReportHeight(rootId) {
  const root = document.getElementById(rootId);
  if (!root) return;
  let last = 0;
  const report = () => {
    const h = Math.ceil(root.getBoundingClientRect().height) + 16;
    if (h > 0 && Math.abs(h - last) > 2) {
      last = h;
      try {
        parent.postMessage({ spydeEmbedHeight: h, vxHeight: h }, '*');
      } catch (e) { /* no host */ }
    }
  };
  report();
  requestAnimationFrame(() => setTimeout(report, 60));
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(report).observe(root);
}

function __spydeLoadImage(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('export: figure PNG failed to decode'));
    img.src = dataUrl;
  });
}
"""
