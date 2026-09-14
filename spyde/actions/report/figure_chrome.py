"""figure_chrome.py — the exported figure's hover toolbar.

The app's report cell reveals a small chrome pill when you hover a figure, and a
click on ◐ drops a contrast caret under it. An exported report should read the
same way: the two are the same document, and a reader who has used one should
not have to learn the other.

The first version put a permanently-visible histogram strip and a row of
download buttons under every figure. That is a lot of furniture for controls
most readers never touch, and it pushed the actual figures apart. This replaces
both with the app's gesture — hover, then click.

What the toolbar offers per figure:

* **◐ contrast** — a histogram with black/white handles, when the figure was
  encoded with room to re-window (see :mod:`contrast_embed`).
* **⤓ data** — the values behind the figure as ``.npy``, when they were small
  enough to embed (see :mod:`data_embed`).
* **⧉ png** — the figure exactly as it looks right now, contrast included, via
  anyplotlib's own ``anyplotlib_export_png`` protocol. Always offered: it needs
  nothing embedded beyond the figure itself.

A figure with neither gets no toolbar at all rather than an empty pill.
"""
from __future__ import annotations

import html as _html
import json
import logging

log = logging.getLogger(__name__)


def chrome_block_html(cell_id: str, contrast: "dict | None",
                      data: "dict | None") -> str:
    """The ``<script>`` payload for one figure cell's toolbar, or ``""``.

    Emitted INSIDE the cell's ``.fig-box`` so the toolbar can position against
    it and the hover target is the figure itself."""
    if not contrast and not data:
        return ""
    payload = {"contrast": (contrast or {}).get("panels", []),
               "data": (data or {}).get("panels", [])}
    if not payload["contrast"] and not payload["data"]:
        return ""
    blob = json.dumps(payload).replace("</", "<\\/")
    element_id = _html.escape(str(cell_id))
    return (f"<script type=\"application/json\" class=\"spyde-chrome\" "
            f"id=\"chrome-{element_id}\">{blob}</script>")


CHROME_CSS = """
/* Hover toolbar on an exported figure — the app's report-cell chrome, same
   gesture. Hidden until the figure is hovered (or the caret is open), and gone
   entirely in print, where a button means nothing. */
.fig-chrome { position: absolute; top: 6px; right: 6px; display: flex; gap: 3px;
  padding: 2px; border-radius: 6px; background: rgba(20,20,28,0.82);
  opacity: 0; transition: opacity 120ms ease; pointer-events: none; z-index: 4; }
.fig-box:hover .fig-chrome, .fig-chrome[data-open="1"] { opacity: 1;
  pointer-events: auto; }
.fig-chrome button { background: none; border: none; color: #cdd6f4;
  cursor: pointer; font-size: 14px; line-height: 1; padding: 4px 7px;
  border-radius: 4px; font-family: inherit; }
.fig-chrome button:hover { background: rgba(255,255,255,0.14); }
.fig-chrome button[data-active="1"] { background: #89b4fa; color: #11111b; }
.fig-caret { position: absolute; top: 34px; right: 6px; z-index: 5;
  background: rgba(24,24,37,0.97); border: 1px solid #89b4fa; border-radius: 8px;
  padding: 6px 8px; box-shadow: 0 6px 22px rgba(0,0,0,0.55);
  display: flex; flex-direction: column; gap: 4px; text-align: left; }
.fig-caret-title { color: #a6adc8; font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.04em; display: flex; align-items: center; gap: 6px; }
.fig-caret-title button { margin-left: auto; background: none; border: none;
  color: #a6adc8; cursor: pointer; font-size: 13px; line-height: 1; }
.fig-caret svg { touch-action: none; cursor: ew-resize; display: block; }
.fig-caret-row { display: flex; justify-content: space-between;
  font-size: 11px; color: #a6adc8; }
.fig-caret-btns { display: flex; gap: 4px; }
.fig-caret-btns button { flex: 1; background: #1e1e2e; color: #a6adc8;
  border: 1px solid #313244; border-radius: 4px; padding: 4px 8px;
  font-size: 11px; cursor: pointer; font-family: inherit; }
.fig-caret-btns button:hover { background: #313244; color: #cdd6f4; }
.fig-note { color: #8a8a92; font-size: 0.72rem; font-style: italic;
  text-align: center; margin-top: 0.3rem; }
@media print { .fig-chrome, .fig-caret { display: none !important; } }
"""

# One script for the whole page: find every figure that has a payload, build its
# toolbar, and wire the caret. Kept in the HOST page (not the figure's sandboxed
# iframe) because the iframe's JS is anyplotlib's, and because an
# `allow-scripts` sandbox without `allow-downloads` cannot start a download.
CHROME_JS = r"""<script>
(function () {
  var W = 320, H = 96;

  function decode(b64, n) {
    var s = atob(b64), bytes = new Uint8Array(s.length);
    for (var i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
    return new Float32Array(bytes.buffer, 0, n);
  }

  // A .npy needs only its 128-byte v1 header, so the reader gets a file numpy
  // opens directly rather than a text grid they have to re-parse and re-shape.
  function npy(values, shape) {
    var dict = "{'descr': '<f4', 'fortran_order': False, 'shape': ("
      + shape.map(function (n) { return n + ','; }).join(' ') + '), }';
    var pad = 64 - ((10 + dict.length + 1) % 64);
    var header = dict + new Array(pad + 1).join(' ') + '\n';
    var out = new Uint8Array(10 + header.length + values.byteLength);
    out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0);
    out[8] = header.length & 0xff; out[9] = (header.length >> 8) & 0xff;
    for (var i = 0; i < header.length; i++) out[10 + i] = header.charCodeAt(i);
    out.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
            10 + header.length);
    return out;
  }

  function save(bytes, filename) {
    var url = URL.createObjectURL(new Blob([bytes],
      { type: 'application/octet-stream' }));
    var a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
  }
  function safe(name) { return String(name).replace(/[^\w.-]+/g, '_') || 'data'; }

  // Save the figure AS IT LOOKS — the reader's contrast, not the author's, and
  // whatever they have zoomed to. anyplotlib's standalone page already answers
  // this request; the host only has to ask and turn the data URL into a file
  // (the sandboxed iframe cannot start a download itself).
  var _pngSeq = 0;
  function savePng(frame, name) {
    if (!frame || !frame.contentWindow) return;
    var id = 'png' + (++_pngSeq);
    var done = false;
    function onMsg(e) {
      var d = e.data || {};
      if (d.type !== 'anyplotlib_export_png_result' || d.requestId !== id) return;
      window.removeEventListener('message', onMsg);
      done = true;
      if (!d.dataUrl) return;
      var bin = atob(d.dataUrl.split(',')[1] || '');
      var bytes = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      save(bytes, safe(name) + '.png');
    }
    window.addEventListener('message', onMsg);
    // Never leave the listener behind if the figure does not answer.
    setTimeout(function () {
      if (!done) window.removeEventListener('message', onMsg);
    }, 15000);
    try {
      frame.contentWindow.postMessage({
        type: 'anyplotlib_export_png', requestId: id,
        opts: { scale: 2, includeWidgets: true },
      }, '*');
    } catch (e) { window.removeEventListener('message', onMsg); }
  }

  function push(frame, entry) {
    if (!frame || !frame.contentWindow) return;
    entry.state.display_min = entry.vmin;
    entry.state.display_max = entry.vmax;
    try {
      frame.contentWindow.postMessage({
        type: 'awi_state',
        key: 'panel_' + entry.panel + '_json',
        value: JSON.stringify(entry.state),
      }, '*');
    } catch (e) { /* frame gone */ }
  }

  function drawHist(svg, entry) {
    var lo = entry.edges[0], hi = entry.edges[entry.edges.length - 1];
    var span = (hi - lo) || 1;
    svg._xOf = function (v) { return ((v - lo) / span) * W; };
    svg._vOf = function (x) { return lo + (x / W) * span; };
    var max = 1;
    for (var i = 0; i < entry.counts.length; i++) {
      if (entry.counts[i] > max) max = entry.counts[i];
    }
    var lmax = Math.log1p(max);
    var parts = ['<rect x="0" y="0" width="' + W + '" height="' + H
      + '" fill="#181825"/>'];
    var bw = W / entry.counts.length;
    for (var b = 0; b < entry.counts.length; b++) {
      var h = (Math.log1p(entry.counts[b]) / lmax) * (H - 3);
      if (h <= 0) continue;
      parts.push('<rect x="' + (b * bw).toFixed(2) + '" y="' + (H - h).toFixed(2)
        + '" width="' + Math.max(1, bw - 0.5).toFixed(2) + '" height="'
        + h.toFixed(2) + '" fill="#7f8dbb"/>');
    }
    var x0 = Math.max(0, Math.min(W, svg._xOf(entry.vmin)));
    var x1 = Math.max(0, Math.min(W, svg._xOf(entry.vmax)));
    parts.push('<rect x="' + x0.toFixed(2) + '" y="0" width="'
      + Math.max(0, x1 - x0).toFixed(2) + '" height="' + H
      + '" fill="rgba(137,180,250,0.18)"/>');
    parts.push('<line x1="' + x0.toFixed(2) + '" y1="0" x2="' + x0.toFixed(2)
      + '" y2="' + H + '" stroke="#f38ba8" stroke-width="2"/>');
    parts.push('<line x1="' + x1.toFixed(2) + '" y1="0" x2="' + x1.toFixed(2)
      + '" y2="' + H + '" stroke="#f38ba8" stroke-width="2"/>');
    svg.innerHTML = parts.join('');
  }

  function fmt(v) {
    var a = Math.abs(v);
    return (a !== 0 && (a < 0.01 || a >= 1e5)) ? v.toExponential(1)
      : String(Math.round(v * 1000) / 1000);
  }

  function buildCaret(box, entry, frame, onClose) {
    var caret = document.createElement('div');
    caret.className = 'fig-caret';
    caret.setAttribute('data-testid', 'fig-caret-contrast');
    // innerHTML, not createElementNS: the export must contain no absolute URL
    // scheme, and the SVG namespace URI is one. The HTML parser namespaces an
    // <svg> tag on its own.
    caret.innerHTML =
      '<div class="fig-caret-title">Contrast<button title="Close">&times;</button></div>'
      + '<svg width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '"></svg>'
      + '<div class="fig-caret-row"><span></span><span></span></div>'
      + '<div class="fig-caret-btns"><button data-act="auto">&#9686; Auto</button>'
      + '<button data-act="reset">&#8635; Reset</button></div>';
    var svg = caret.querySelector('svg');
    var spans = caret.querySelectorAll('.fig-caret-row span');
    var initial = [entry.vmin, entry.vmax];

    function refresh() {
      drawHist(svg, entry);
      spans[0].textContent = fmt(entry.vmin);
      spans[1].textContent = fmt(entry.vmax);
    }
    refresh();

    var dragging = null;
    svg.addEventListener('pointerdown', function (e) {
      var r = svg.getBoundingClientRect();
      var x = ((e.clientX - r.left) / r.width) * W;
      dragging = Math.abs(x - svg._xOf(entry.vmin))
        <= Math.abs(x - svg._xOf(entry.vmax)) ? 'min' : 'max';
      svg.setPointerCapture(e.pointerId);
      move(e);
    });
    svg.addEventListener('pointermove', move);
    svg.addEventListener('pointerup', function (e) {
      dragging = null;
      try { svg.releasePointerCapture(e.pointerId); } catch (_) {}
    });
    function move(e) {
      if (!dragging) return;
      var r = svg.getBoundingClientRect();
      var v = svg._vOf(((e.clientX - r.left) / r.width) * W);
      if (dragging === 'min') entry.vmin = Math.min(v, entry.vmax - 1e-9);
      else entry.vmax = Math.max(v, entry.vmin + 1e-9);
      refresh();
      push(frame, entry);
    }
    caret.querySelector('[data-act="auto"]').addEventListener('click', function () {
      // "Auto" with no backend is the window the report was written with — the
      // author's judgement, which is the best answer available offline.
      entry.vmin = initial[0]; entry.vmax = initial[1];
      refresh(); push(frame, entry);
    });
    caret.querySelector('[data-act="reset"]').addEventListener('click', function () {
      // …and "Reset" is the full encoded band: everything the file still holds.
      entry.vmin = entry.raw_min; entry.vmax = entry.raw_max;
      refresh(); push(frame, entry);
    });
    caret.querySelector('.fig-caret-title button')
      .addEventListener('click', onClose);
    box.appendChild(caret);
    return caret;
  }

  Array.prototype.forEach.call(
    document.querySelectorAll('script.spyde-chrome'), function (tag) {
      var box = tag.closest ? tag.closest('.fig-box') : null;
      if (!box) return;
      var payload;
      try { payload = JSON.parse(tag.textContent); } catch (e) { return; }
      var frame = box.querySelector('iframe');
      var contrast = (payload.contrast || [])[0];
      var data = payload.data || [];
      var downloads = data.filter(function (d) { return !d.omitted; });
      var omitted = data.filter(function (d) { return d.omitted; });
      var name = (contrast && contrast.title)
        || (data[0] && data[0].name) || 'figure';
      if (!frame && !downloads.length && !omitted.length) return;

      var bar = document.createElement('div');
      bar.className = 'fig-chrome';
      var caret = null;

      if (contrast && frame) {
        var cb = document.createElement('button');
        cb.textContent = '◐';
        cb.title = 'Contrast — adjust the display range';
        cb.setAttribute('data-testid', 'fig-chrome-contrast');
        cb.addEventListener('click', function () {
          if (caret) { closeCaret(); return; }
          caret = buildCaret(box, contrast, frame, closeCaret);
          cb.setAttribute('data-active', '1');
          bar.setAttribute('data-open', '1');
        });
        bar.appendChild(cb);
        var closeCaret = function () {
          if (caret) { caret.remove(); caret = null; }
          cb.removeAttribute('data-active');
          bar.removeAttribute('data-open');
        };
      }

      downloads.forEach(function (entry) {
        var db = document.createElement('button');
        db.textContent = '⤓';
        db.title = 'Download the values behind this figure (.npy)';
        db.setAttribute('data-testid', 'fig-chrome-data');
        db.addEventListener('click', function () {
          var n = entry.shape.reduce(function (a, b) { return a * b; }, 1);
          save(npy(decode(entry.b64, n), entry.shape), safe(entry.name) + '.npy');
        });
        bar.appendChild(db);
      });

      if (frame) {
        var pb = document.createElement('button');
        pb.textContent = '⧉';
        pb.title = 'Save this figure as a PNG, exactly as it looks now';
        pb.setAttribute('data-testid', 'fig-chrome-png');
        pb.addEventListener('click', function () { savePng(frame, name); });
        bar.appendChild(pb);
      }

      if (bar.children.length) box.appendChild(bar);

      // An oversized panel still says so — quietly, under the figure. The
      // alternative is a reader assuming the data is there and finding no way
      // to reach it.
      omitted.forEach(function (entry) {
        var note = document.createElement('div');
        note.className = 'fig-note';
        note.textContent = entry.name + ': values not embedded (' + entry.reason + ')';
        if (box.parentNode) box.parentNode.appendChild(note);
      });
    });
})();
</script>
"""
