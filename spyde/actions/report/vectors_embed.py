"""vectors_embed.py — embed a FULL diffraction-vectors dataset in an HTML report.

A find-vectors result is a compact CSR flat buffer (per vector: nav x/y, kx,
ky, intensity) — small enough to inline into a self-contained HTML page and
rich enough to recompute VIRTUAL IMAGES client-side. The exported panel shows
the k-space density of every vector with a draggable/resizable circular
detector; a script recomputes the virtual image (intensity-sum or counts per
nav position) live in the browser — no backend, no network, works years later
from the single .html file.

Packing (little-endian, one base64 blob):
    uint16  x[n]        nav column index
    uint16  y[n]        nav row index
    float32 kx[n], ky[n], intensity[n]
16 bytes/vector → ~1 M vectors ≈ 16 MB (≈21 MB as base64). Above
``MAX_EMBED_VECTORS`` the embed is refused (returns None) and the export falls
back to the baked static image — a 20 M-vector scan would be a quarter-GB
page.

Hooked into ``export_html._render_body`` (interactive mode): a figure cell
whose resolved source tree carries ``diffraction_vectors`` exports this
explorer instead of the anyplotlib iframe.
"""
from __future__ import annotations

import base64
import html as _html
import json
import logging

import numpy as np

log = logging.getLogger(__name__)

MAX_EMBED_VECTORS = 3_000_000


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
        "r0": float(getattr(vecs, "kernel_radius_data", 0.0) or 0.0),
    }
    return {"header": header, "b64": base64.b64encode(blob).decode("ascii")}


# ── the self-contained explorer page ─────────────────────────────────────────
# Plain ES5-ish script, no dependencies: decodes the blob, renders the k-space
# density once, and recomputes the VI per detector move (rAF-throttled — a
# few million typed-array ops is ~10 ms).

_EXPLORER_CSS = """
:root { color-scheme: light; }
body { margin: 0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       background: #fff; color: #1a1a1a; font-size: 12px; }
.vx-wrap { display: flex; gap: 12px; padding: 8px; align-items: flex-start;
           flex-wrap: wrap; }
.vx-col { display: flex; flex-direction: column; gap: 4px; }
.vx-col h4 { margin: 0; font-size: 12px; font-weight: 600; color: #444; }
canvas { border: 1px solid #d0d0d6; border-radius: 4px; }
#vx-k { cursor: crosshair; touch-action: none; }
#vx-vi { image-rendering: pixelated; }
.vx-meta { color: #666; font-size: 11px; max-width: 320px; }
.vx-controls { display: flex; gap: 10px; align-items: center; font-size: 11px; }
"""

_EXPLORER_JS = r"""
(function () {
  var hdr = JSON.parse(document.getElementById('vx-header').textContent);
  var b64 = document.getElementById('vx-data').textContent.trim();

  function decode(cb) {
    fetch('data:application/octet-stream;base64,' + b64)
      .then(function (r) { return r.arrayBuffer(); }).then(cb)
      .catch(function () {          // very old engines: atob fallback
        var s = atob(b64), a = new Uint8Array(s.length);
        for (var i = 0; i < s.length; i++) a[i] = s.charCodeAt(i);
        cb(a.buffer);
      });
  }

  decode(function (ab) {
    var n = hdr.n, navH = hdr.nav[0], navW = hdr.nav[1];
    var off = 0;
    var X = new Uint16Array(ab, off, n); off += 2 * n;
    var Y = new Uint16Array(ab, off, n); off += 2 * n;
    var KX = new Float32Array(ab, off, n); off += 4 * n;
    var KY = new Float32Array(ab, off, n); off += 4 * n;
    var IN = new Float32Array(ab, off, n);

    var kxLo = hdr.k.kx[0], kxHi = hdr.k.kx[1];
    var kyLo = hdr.k.ky[0], kyHi = hdr.k.ky[1];
    var KW = 280, KH = 280;
    var kCan = document.getElementById('vx-k');
    kCan.width = KW; kCan.height = KH;
    var kCtx = kCan.getContext('2d');

    // k-space density (log) — rendered ONCE to an offscreen canvas.
    var bins = new Float32Array(KW * KH);
    var sx = (KW - 1) / (kxHi - kxLo || 1), sy = (KH - 1) / (kyHi - kyLo || 1);
    for (var i = 0; i < n; i++) {
      var bx = ((KX[i] - kxLo) * sx) | 0, by = ((KY[i] - kyLo) * sy) | 0;
      if (bx >= 0 && bx < KW && by >= 0 && by < KH) bins[by * KW + bx] += IN[i];
    }
    var mx = 0;
    for (i = 0; i < bins.length; i++) if (bins[i] > mx) mx = bins[i];
    var kImg = document.createElement('canvas');
    kImg.width = KW; kImg.height = KH;
    var kd = kImg.getContext('2d').createImageData(KW, KH);
    var lmax = Math.log1p(mx) || 1;
    for (i = 0; i < bins.length; i++) {
      var v = 255 - Math.round(255 * Math.log1p(bins[i]) / lmax);   // dark spots
      kd.data[4 * i] = v; kd.data[4 * i + 1] = v; kd.data[4 * i + 2] = v;
      kd.data[4 * i + 3] = 255;
    }
    kImg.getContext('2d').putImageData(kd, 0, 0);

    // Detector state (data coords). Default: centred, r0 or 10% of the span.
    var det = {
      cx: (kxLo + kxHi) / 2, cy: (kyLo + kyHi) / 2,
      r: hdr.r0 > 0 ? 3 * hdr.r0 : 0.1 * Math.abs(kxHi - kxLo),
    };

    var viCan = document.getElementById('vx-vi');
    viCan.width = navW; viCan.height = navH;
    var scale = Math.max(1, Math.floor(280 / Math.max(navW, navH)));
    viCan.style.width = (navW * scale) + 'px';
    viCan.style.height = (navH * scale) + 'px';
    var viCtx = viCan.getContext('2d');
    var acc = new Float32Array(navW * navH);
    var readout = document.getElementById('vx-readout');

    function mode() {
      return document.querySelector('input[name=vx-mode]:checked').value;
    }

    function drawK() {
      kCtx.drawImage(kImg, 0, 0);
      kCtx.strokeStyle = '#e01b24'; kCtx.lineWidth = 1.5;
      kCtx.beginPath();
      kCtx.ellipse((det.cx - kxLo) * sx, (det.cy - kyLo) * sy,
                   det.r * sx, det.r * sy, 0, 0, 2 * Math.PI);
      kCtx.stroke();
    }

    function computeVI() {
      acc.fill(0);
      var r2 = det.r * det.r, cx = det.cx, cy = det.cy, hit = 0;
      var sum = mode() === 'sum';
      for (var i = 0; i < n; i++) {
        var dx = KX[i] - cx, dy = KY[i] - cy;
        if (dx * dx + dy * dy <= r2) {
          acc[Y[i] * navW + X[i]] += sum ? IN[i] : 1;
          hit++;
        }
      }
      var m = 0;
      for (i = 0; i < acc.length; i++) if (acc[i] > m) m = acc[i];
      var img = viCtx.createImageData(navW, navH);
      var s = m > 0 ? 255 / m : 0;
      for (i = 0; i < acc.length; i++) {
        var v = Math.round(acc[i] * s);
        img.data[4 * i] = v; img.data[4 * i + 1] = v; img.data[4 * i + 2] = v;
        img.data[4 * i + 3] = 255;
      }
      viCtx.putImageData(img, 0, 0);
      readout.textContent = 'detector: (' + det.cx.toFixed(3) + ', '
        + det.cy.toFixed(3) + ') r=' + det.r.toFixed(3)
        + (hdr.k.units ? ' ' + hdr.k.units : '') + ' — ' + hit
        + ' of ' + n + ' vectors';
    }

    var pending = false;
    function refresh() {
      drawK();
      if (!pending) {
        pending = true;
        requestAnimationFrame(function () { pending = false; computeVI(); });
      }
    }

    var dragging = false;
    function toData(ev) {
      var r = kCan.getBoundingClientRect();
      return { x: kxLo + (ev.clientX - r.left) / sx,
               y: kyLo + (ev.clientY - r.top) / sy };
    }
    kCan.addEventListener('pointerdown', function (ev) {
      dragging = true; kCan.setPointerCapture(ev.pointerId);
      var p = toData(ev); det.cx = p.x; det.cy = p.y; refresh();
    });
    kCan.addEventListener('pointermove', function (ev) {
      if (!dragging) return;
      var p = toData(ev); det.cx = p.x; det.cy = p.y; refresh();
    });
    kCan.addEventListener('pointerup', function () { dragging = false; });
    kCan.addEventListener('wheel', function (ev) {
      ev.preventDefault();
      det.r *= ev.deltaY < 0 ? 1.1 : 1 / 1.1;
      refresh();
    }, { passive: false });
    var radios = document.querySelectorAll('input[name=vx-mode]');
    for (var ri = 0; ri < radios.length; ri++) {
      radios[ri].addEventListener('change', refresh);
    }

    refresh();
    document.getElementById('vx-root').dataset.ready = '1';
  });
})();
"""


def vectors_explorer_html(vecs, caption: str = "") -> "str | None":
    """The self-contained explorer page for one vectors dataset (goes into the
    report's sandboxed ``<iframe srcdoc>`` like any interactive figure), or
    None when the dataset is empty / over the embed cap."""
    payload = pack_vectors(vecs)
    if payload is None:
        return None
    hdr = payload["header"]
    cap = _html.escape(caption or "")
    return (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
        f"<style>{_EXPLORER_CSS}</style></head><body>"
        f"<div id=\"vx-root\" class=\"vx-wrap\">"
        "<div class=\"vx-col\"><h4>Diffraction vectors (drag detector, wheel = radius)</h4>"
        "<canvas id=\"vx-k\"></canvas>"
        "<div class=\"vx-controls\">"
        "<label><input type=\"radio\" name=\"vx-mode\" value=\"sum\" checked> intensity</label>"
        "<label><input type=\"radio\" name=\"vx-mode\" value=\"count\"> counts</label>"
        "</div></div>"
        "<div class=\"vx-col\"><h4>Virtual image</h4>"
        "<canvas id=\"vx-vi\"></canvas>"
        f"<div id=\"vx-readout\" class=\"vx-meta\"></div>"
        f"<div class=\"vx-meta\">{cap}</div>"
        "</div></div>"
        f"<script type=\"application/json\" id=\"vx-header\">{json.dumps(hdr)}</script>"
        f"<script type=\"application/json\" id=\"vx-data\">{payload['b64']}</script>"
        f"<script>{_EXPLORER_JS}</script>"
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
