"""
export_html.py — the Report Builder export handlers (Phase 3).

Three export forms, all sharing the SAME snapshot-harvest handshake as
``report_save`` (so ``<img>``s / figures are current):

* ``report_export_html {mode:'static'|'interactive', path}`` — one self-contained
  HTML file (a clean neutral article; print-safe for Electron ``printToPDF``).
    - **static**: figure cells become ``<figure><img src="data:image/png;…">``;
      no iframes, no external fetches.
    - **interactive**: figure cells embed their LIVE anyplotlib figure in a
      sandboxed ``<iframe srcdoc>`` (rebuilt via ``build_cell_figure`` so the
      pixels are inlined — no ``\\x00bin:`` tokens); a cell that can't rebuild
      (offline) falls back to the static ``<img>``.
* ``report_export_markdown {path}`` — write the UNZIPPED container (``report.md``
  + ``figures/*.yaml`` + ``assets/*.png``) into a target DIRECTORY.
* ``report_paste_cell {cell}`` — insert a serialized cell (renderer clipboard):
  a markdown cell verbatim, or a figure cell rebuilt LIVE from the resolved
  source (offline → the provided ``png`` data URL as the baked fallback).

On success each export emits
``{"type":"report_exported","kind":<k>,"path":<str>}`` where ``<k>`` is one of
``"html-static" | "html-interactive" | "markdown-folder"``. Failures go through
``emit_error``.
"""
from __future__ import annotations

import base64
import html as _html
import logging
import os

import numpy as np

from spyde.backend import ipc
from spyde.actions.report.handlers import (
    _decode_data_url, _manager, harvest_snapshots,
)
from spyde.actions.report.model import (
    Cell, FigureSpec, dir_is_safe_md_target, new_cell_id, write_report_dir,
)

log = logging.getLogger(__name__)


# ── the page skeleton + article CSS ───────────────────────────────────────────

# A clean neutral article stylesheet: readable column, works when printed. The
# print block forces black-on-white (this exact file is what Electron
# printToPDF consumes) so a dark UI theme never bleeds into the PDF.
_ARTICLE_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2.5rem 1.25rem;
  background: #ffffff; color: #1a1a1a;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica,
    Arial, sans-serif;
  line-height: 1.6; font-size: 16px;
}
.report-article { max-width: 46rem; margin: 0 auto; }
.report-article h1 { font-size: 2rem; line-height: 1.2; margin: 0 0 1.5rem;
  font-weight: 700; }
.report-article h2 { font-size: 1.5rem; margin: 2rem 0 0.75rem; }
.report-article h3 { font-size: 1.2rem; margin: 1.5rem 0 0.5rem; }
.report-article p { margin: 0 0 1rem; }
.report-article a { color: #1a5fb4; }
.report-article code { font-family: ui-monospace, SFMono-Regular, Menlo,
  Consolas, monospace; font-size: 0.9em;
  background: #f2f2f4; padding: 0.1em 0.35em; border-radius: 4px; }
.report-article pre { background: #f6f6f8; padding: 1rem; border-radius: 8px;
  overflow-x: auto; }
.report-article pre code { background: none; padding: 0; }
.report-article pre.md-src { white-space: pre-wrap; }
.report-article blockquote { margin: 0 0 1rem; padding: 0 1rem;
  border-left: 4px solid #d0d0d6; color: #555; }
.report-article li input[type="checkbox"] { margin-right: 0.4em; }
/* KaTeX math ships as MathML (output:'mathml') — no KaTeX CSS/fonts needed;
   browsers and printToPDF render MathML Core natively. */
.report-article .katex { font-size: 1.08em; }
.report-article .katex-display { display: block; margin: 1rem 0;
  text-align: center; overflow-x: auto; overflow-y: hidden; }
.report-article table { border-collapse: collapse; margin: 0 0 1rem;
  display: block; overflow-x: auto; }
.report-article th, .report-article td { border: 1px solid #d0d0d6;
  padding: 0.4rem 0.6rem; }
.report-article img { max-width: 100%; height: auto; }
figure.report-figure { margin: 1.75rem 0; text-align: center; }
figure.report-figure img { max-width: 100%; height: auto;
  border: 1px solid #e2e2e6; border-radius: 6px; }
figure.report-figure iframe { width: 100%; border: 1px solid #e2e2e6;
  border-radius: 6px; }
figure.report-figure figcaption { margin-top: 0.6rem; font-size: 0.9rem;
  color: #555; font-style: italic; }
@media print {
  body { background: #fff !important; color: #000 !important;
    padding: 0; }
  .report-article { max-width: none; }
  figure.report-figure img, figure.report-figure iframe { border: none; }
}
"""

# Aspect box for an interactive figure iframe — a self-contained figure sizes
# itself, but the srcdoc iframe needs an explicit height. A 4:3-ish default keeps
# a diffraction-pattern square figure fully visible without scroll.
_IFRAME_HEIGHT_PX = 480


def _page(title: str, body_html: str) -> str:
    """Wrap the article body in a self-contained HTML page (no external fetches)."""
    esc_title = _html.escape(title or "Report")
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{esc_title}</title>\n"
        f"<style>{_ARTICLE_CSS}</style>\n"
        "</head>\n<body>\n"
        f"<article class=\"report-article\">\n<h1>{esc_title}</h1>\n"
        f"{body_html}\n</article>\n</body>\n</html>\n"
    )


# ── cell → HTML fragment ──────────────────────────────────────────────────────


def _markdown_cell_html(cell: Cell) -> str:
    """The rendered-HTML fragment for a markdown cell.

    The renderer is the single markdown engine (marked+DOMPurify): it sends the
    sanitized fragment on every commit and we cache it on ``cell.html``. When
    that cache is absent (never edited this session / reloaded), fall back to the
    raw markdown source escaped inside ``<pre class="md-src">`` — correctness
    over beauty."""
    frag = (cell.html or "").strip()
    if frag:
        return frag
    return f"<pre class=\"md-src\">{_html.escape(cell.source or '')}</pre>"


def _figure_img_html(caption: str, png: "bytes | None") -> str:
    """A static ``<figure><img data:image/png;…></figure>`` for a figure cell.
    Returns ``""`` when there are no pixels (placeholder / unbaked)."""
    if not png:
        return ""
    b64 = base64.b64encode(png).decode("ascii")
    cap = _html.escape(caption or "")
    figcap = f"<figcaption>{cap}</figcaption>" if cap else ""
    return (
        "<figure class=\"report-figure\">"
        f"<img src=\"data:image/png;base64,{b64}\" alt=\"{cap}\">"
        f"{figcap}</figure>"
    )


def _figure_iframe_html(caption: str, figure_html: str) -> str:
    """A sandboxed ``<iframe srcdoc>`` embedding a cell's self-contained
    interactive figure HTML (pixels already inlined). The srcdoc content is
    HTML-escaped so the attribute can't be broken out of."""
    srcdoc = _html.escape(figure_html, quote=True)
    cap = _html.escape(caption or "")
    figcap = f"<figcaption>{cap}</figcaption>" if cap else ""
    return (
        "<figure class=\"report-figure\">"
        f"<iframe sandbox=\"allow-scripts\" srcdoc=\"{srcdoc}\" "
        f"style=\"height:{_IFRAME_HEIGHT_PX}px;\" loading=\"lazy\"></iframe>"
        f"{figcap}</figure>"
    )


def _build_interactive_figure_html(mgr, cell: Cell) -> "str | None":
    """Rebuild a figure cell's LIVE anyplotlib figure and return its
    self-contained standalone HTML (pixels materialised via ``build_cell_figure``
    → ``_resolve_pixels_for_standalone`` so no binary tokens leak), or None when
    the cell has no snapshot to rebuild (offline)."""
    if cell.spec is None:
        return None
    snap_map = mgr.snapshot_map(cell.id)
    if not snap_map:
        return None
    try:
        from spyde.actions.report.figure_builder import build_cell_figure
        # standalone=True → the JS bundle is INLINED (no machine-local file:// ESM
        # reference), so the sandboxed srcdoc iframe renders on any machine/browser.
        _fig, _fig_id, html_str = build_cell_figure(
            cell.spec, snap_map, standalone=True)
        return html_str
    except Exception as e:
        log.debug("interactive figure rebuild failed for cell %s: %s", cell.id, e)
        return None


def _render_cell_html(mgr, cell: Cell, assets: dict, *, interactive: bool,
                      session=None) -> str:
    """The HTML fragment for ONE cell (markdown or figure), shared by the
    article body AND the slides shell. A placeholder figure → ``""`` (skipped).

    Figure handling mirrors :func:`_render_body`'s per-cell logic: interactive
    mode tries the vectors explorer, then the tinted-overlay blender, then the
    live-figure iframe; anything that can't rebuild falls back to the static
    ``<img>``. Static mode is always the ``<img>``."""
    if cell.cell_type == "markdown":
        return _markdown_cell_html(cell)
    if cell.cell_type != "figure" or cell.placeholder:
        return ""
    html_frag = ""
    if interactive:
        # Drop-time choice: vectors_mode == "image" pins the static
        # snapshot even when the tree carries diffraction vectors.
        if getattr(cell.spec, "vectors_mode", "") != "image":
            try:
                from spyde.actions.report.vectors_embed import (
                    vectors_explorer_html, vectors_for_cell,
                )
                vecs = vectors_for_cell(session, cell)
                if vecs is not None:
                    vx_html = vectors_explorer_html(vecs, caption=cell.caption)
                    if vx_html is not None:
                        html_frag = _figure_iframe_html(cell.caption, vx_html)
            except Exception as e:
                log.debug("vectors embed for cell %s failed: %s", cell.id, e)
        # Tinted-overlay blender (vectors swap above wins when both
        # apply — a vectors cell stays a vectors explorer).
        if not html_frag:
            try:
                from spyde.actions.report.overlay_embed import (
                    overlay_blender_html,
                )
                ov_html = overlay_blender_html(mgr, cell, caption=cell.caption)
                if ov_html is not None:
                    html_frag = _figure_iframe_html(cell.caption, ov_html)
            except Exception as e:
                log.debug("overlay blender embed for cell %s failed: %s",
                          cell.id, e)
        if not html_frag:
            fig_html = _build_interactive_figure_html(mgr, cell)
            if fig_html is not None:
                html_frag = _figure_iframe_html(cell.caption, fig_html)
    if not html_frag:
        # Static path (also the interactive OFFLINE fallback).
        html_frag = _figure_img_html(cell.caption, assets.get(cell.id))
    return html_frag


def _render_body(mgr, assets: dict, *, interactive: bool, session=None) -> str:
    """Assemble the article body: each cell in order → its HTML fragment. Figure
    placeholders are skipped. For interactive mode a figure with no rebuildable
    live figure falls back to the static ``<img>``.

    A figure cell whose resolved source tree carries ``diffraction_vectors``
    exports the FULL vectors dataset as the interactive explorer instead of an
    anyplotlib iframe — the reader recomputes virtual images from the embedded
    vectors right in the page (see vectors_embed.py). Over the embed cap /
    offline → the usual static image.

    A figure cell with TINTED overlay layers (and no vectors explorer — the
    vectors swap takes precedence) exports the overlay BLENDER instead: base
    grayscale + clear→tint ramps with a LIVE opacity slider per overlay (see
    overlay_embed.py). No tinted overlay → the live-figure iframe as before."""
    blocks: list[str] = []
    for c in mgr.doc.cells:
        frag = _render_cell_html(mgr, c, assets, interactive=interactive,
                                 session=session)
        if frag:
            blocks.append(frag)
    return "\n".join(blocks)


# ── slides deck (portable, self-contained, no CDN) ────────────────────────────

# A minimal reveal.js-STYLE deck: full-viewport dark stage, one `.slide` shown at
# a time, a tiny vanilla-JS switcher (arrow / space / pagedown advance, Home/End,
# a slide counter). No external fetches — the interactive figure embeds are the
# SAME self-contained srcdoc iframes the interactive HTML export emits, so they
# work here too with zero runtime Python. Print falls back to showing every slide
# stacked (so a browser "Print to PDF" of the deck yields one slide per page-ish).
_SLIDES_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; height: 100%; }
body {
  background: #14141f; color: #e8e8f0; overflow: hidden;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica,
    Arial, sans-serif;
  line-height: 1.6; font-size: 22px;
}
#deck { position: fixed; inset: 0; }
.slide {
  position: absolute; inset: 0; display: none;
  flex-direction: column; justify-content: center;
  padding: 5vh 8vw; overflow-y: auto;
}
.slide.active { display: flex; }
.slide-inner { max-width: 60rem; margin: 0 auto; width: 100%; }
.slide h1 { font-size: 2.4rem; line-height: 1.15; margin: 0 0 1.2rem; font-weight: 700; }
.slide h2 { font-size: 1.8rem; margin: 1.4rem 0 0.7rem; }
.slide h3 { font-size: 1.35rem; margin: 1.1rem 0 0.5rem; }
.slide p { margin: 0 0 0.9rem; }
.slide a { color: #89b4fa; }
.slide code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.9em; background: #22222f; padding: 0.1em 0.35em; border-radius: 4px; }
.slide pre { background: #1c1c28; padding: 1rem; border-radius: 8px; overflow-x: auto; }
.slide pre code { background: none; padding: 0; }
.slide pre.md-src { white-space: pre-wrap; }
.slide blockquote { margin: 0 0 1rem; padding: 0 1rem; border-left: 4px solid #45475a;
  color: #a6adc8; }
.slide table { border-collapse: collapse; margin: 0 0 1rem; display: block; overflow-x: auto; }
.slide th, .slide td { border: 1px solid #45475a; padding: 0.4rem 0.6rem; }
.slide .katex-display { display: block; margin: 1rem 0; text-align: center;
  overflow-x: auto; overflow-y: hidden; }
figure.report-figure { margin: 1rem 0; text-align: center; }
figure.report-figure img { max-width: 100%; max-height: 62vh; height: auto;
  border-radius: 6px; }
figure.report-figure iframe { width: 100%; height: 62vh; border: 1px solid #313244;
  border-radius: 6px; }
figure.report-figure figcaption { margin-top: 0.5rem; font-size: 0.85rem;
  color: #a6adc8; font-style: italic; }
#deck-counter {
  position: fixed; bottom: 14px; right: 18px; z-index: 10;
  font-size: 0.8rem; color: #7f849c; background: rgba(20,20,31,0.7);
  padding: 3px 10px; border-radius: 12px; user-select: none;
}
#deck-hint {
  position: fixed; bottom: 14px; left: 18px; z-index: 10;
  font-size: 0.72rem; color: #585b70; user-select: none;
}
@media print {
  body { overflow: visible; height: auto; background: #fff; color: #000; }
  #deck-counter, #deck-hint { display: none; }
  .slide { position: static; display: flex !important; page-break-after: always;
    min-height: 90vh; }
}
"""

_SLIDES_JS = """
(function () {
  var slides = Array.prototype.slice.call(document.querySelectorAll('.slide'));
  var counter = document.getElementById('deck-counter');
  var i = 0;
  function show(n) {
    if (!slides.length) return;
    i = Math.max(0, Math.min(slides.length - 1, n));
    for (var k = 0; k < slides.length; k++) {
      slides[k].classList.toggle('active', k === i);
    }
    if (counter) counter.textContent = (i + 1) + ' / ' + slides.length;
    try { location.hash = 'slide-' + (i + 1); } catch (e) {}
  }
  function next() { show(i + 1); }
  function prev() { show(i - 1); }
  document.addEventListener('keydown', function (e) {
    var k = e.key;
    // A presentation clicker sends these arrow / PageUp/PageDown keys.
    if (k === 'ArrowRight' || k === 'PageDown' || k === ' ' || k === 'Spacebar') {
      e.preventDefault(); next();
    } else if (k === 'ArrowLeft' || k === 'PageUp') {
      e.preventDefault(); prev();
    } else if (k === 'Home') { e.preventDefault(); show(0); }
    else if (k === 'End') { e.preventDefault(); show(slides.length - 1); }
  });
  // Click the right two-thirds → next, the left third → prev (tap-friendly).
  document.getElementById('deck').addEventListener('click', function (e) {
    if (e.target.closest('a, iframe, button, input, figure.report-figure')) return;
    if (e.clientX < window.innerWidth / 3) prev(); else next();
  });
  var m = /slide-(\\d+)/.exec(location.hash || '');
  show(m ? parseInt(m[1], 10) - 1 : 0);
})();
"""


def _slides_page(title: str, slides_html: str) -> str:
    """Wrap the rendered slides in the self-contained deck shell (inline CSS +
    JS, no external fetches)."""
    esc_title = _html.escape(title or "Presentation")
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{esc_title}</title>\n"
        f"<style>{_SLIDES_CSS}</style>\n"
        "</head>\n<body>\n"
        f"<div id=\"deck\">\n{slides_html}\n</div>\n"
        "<div id=\"deck-counter\"></div>\n"
        "<div id=\"deck-hint\">← → / Space to navigate</div>\n"
        f"<script>{_SLIDES_JS}</script>\n"
        "</body>\n</html>\n"
    )


def _render_slides(mgr, assets: dict, *, interactive: bool, session=None) -> str:
    """Render the report as slide `<section class="slide">` blocks, grouped by
    the same ``slide_break`` flag :meth:`ReportDoc.slides` uses. Each slide holds
    every one of its cells' HTML fragments (reusing :func:`_render_cell_html`),
    so a slide's interactive embeds work exactly as in the interactive HTML
    export. A slide whose cells all render empty (e.g. a lone placeholder) is
    dropped rather than shown blank."""
    blocks: list[str] = []
    for group in mgr.doc.slides():
        frags = [
            _render_cell_html(mgr, c, assets, interactive=interactive,
                              session=session)
            for c in group
        ]
        frags = [f for f in frags if f]
        if not frags:
            continue
        inner = "\n".join(frags)
        blocks.append(
            f"<section class=\"slide\">\n<div class=\"slide-inner\">\n"
            f"{inner}\n</div>\n</section>")
    return "\n".join(blocks)


# ── handlers ───────────────────────────────────────────────────────────────────


def _exported_msg(kind: str, path: str, token) -> dict:
    """The ``report_exported`` message body. Echoes ``token`` VERBATIM when the
    request supplied one (any non-None value, incl. 0 / ""), and OMITS the key
    entirely otherwise — so the contract stays backward compatible."""
    msg = {"type": "report_exported", "kind": kind, "path": path}
    if token is not None:
        msg["token"] = token
    return msg


def report_export_html(session, plot, payload) -> None:
    """Export the open report as ONE self-contained HTML file.

    ``mode`` is ``"static"`` (baked ``<img>``s only), ``"interactive"`` (live
    figures in sandboxed ``srcdoc`` iframes), or ``"slides"`` (a portable
    reveal.js-STYLE deck — the same self-contained cells wrapped in a thin
    slide-navigable shell, grouped by ``slide_break``; the interactive figure
    embeds work in the deck too, zero runtime Python). Runs the snapshot-harvest
    handshake first so the images are fresh, then writes the file and emits
    ``report_exported``.

    ``temp:true`` (static only) writes to a UNIQUE file under the OS temp
    directory instead of ``path`` and emits ``report_exported`` with THAT path.
    This is the first leg of the PDF export flow: the renderer awaits the emitted
    temp path, then hands it to Electron ``printToPDF``. ``path`` is ignored when
    ``temp`` is set."""
    mgr = _manager(session)
    if not mgr.open:
        ipc.emit_error("report_export_html: no open report.")
        return
    temp = bool(payload.get("temp"))
    if temp:
        import tempfile
        import uuid
        path = os.path.join(
            tempfile.gettempdir(), f"spyde-report-{uuid.uuid4().hex}.html")
    else:
        path = payload.get("path")
        if not path:
            ipc.emit_error("report_export_html: no path.")
            return
    mode = str(payload.get("mode", "static")).lower()
    slides = mode == "slides"
    interactive = mode == "interactive"
    if slides:
        kind = "html-slides"
    elif interactive:
        kind = "html-interactive"
    else:
        kind = "html-static"
    # A slides deck embeds the interactive figures (the whole point of a portable
    # deck the reader can drive) — so it renders figure cells interactively.
    render_interactive = interactive or slides
    # OPTIONAL correlation token echoed verbatim in report_exported so the renderer
    # can match an export reply to the request it issued (e.g. the PDF flow awaits a
    # specific temp-export). Backward compatible: absent → absent in the reply.
    token = payload.get("token")

    def finish(harvested: dict) -> None:
        try:
            assets = mgr.assemble_assets(harvested)
            if slides:
                body = _render_slides(mgr, assets,
                                      interactive=render_interactive,
                                      session=session)
                page = _slides_page(mgr.doc.title, body)
            else:
                body = _render_body(mgr, assets, interactive=render_interactive,
                                    session=session)
                page = _page(mgr.doc.title, body)
            with open(path, "w", encoding="utf-8") as f:
                f.write(page)
        except Exception as e:
            ipc.emit_error(f"Exporting HTML failed: {e}")
            log.exception("report_export_html failed")
            return
        ipc.emit(_exported_msg(kind, path, token))

    harvest_snapshots(session, mgr, finish)


def report_export_markdown(session, plot, payload) -> None:
    """Export the open report as an UNZIPPED markdown folder (``report.md`` +
    ``figures/*.yaml`` + ``assets/*.png``) into the target directory ``path``.

    Refuses a directory that already holds content that doesn't look like a prior
    export (conservative — never clobber the user's data). Runs the same snapshot
    handshake as save so the baked PNGs are fresh."""
    mgr = _manager(session)
    if not mgr.open:
        ipc.emit_error("report_export_markdown: no open report.")
        return
    path = payload.get("path")
    if not path:
        ipc.emit_error("report_export_markdown: no path.")
        return
    if os.path.isfile(path):
        ipc.emit_error("report_export_markdown: target is a file, not a directory.")
        return
    if not dir_is_safe_md_target(path):
        ipc.emit_error(
            "report_export_markdown: target directory is not empty and doesn't "
            "look like a previous export — pick an empty directory.")
        return
    # OPTIONAL correlation token echoed verbatim in report_exported (see
    # report_export_html). Backward compatible: absent → absent in the reply.
    token = payload.get("token")

    def finish(harvested: dict) -> None:
        try:
            assets = mgr.assemble_assets(harvested)
            mgr.doc.touch()
            write_report_dir(mgr.doc, path, assets=assets)
        except Exception as e:
            ipc.emit_error(f"Exporting markdown folder failed: {e}")
            log.exception("report_export_markdown failed")
            return
        ipc.emit(_exported_msg("markdown-folder", path, token))

    harvest_snapshots(session, mgr, finish)


def report_paste_cell(session, plot, payload) -> None:
    """Insert a serialized cell from the renderer's internal clipboard.

    ``cell`` is ``{cell_type:'markdown', source}`` or
    ``{cell_type:'figure', caption, figure:<FigureSpec dict>, png?:<dataURL>}``.
    A markdown cell is inserted verbatim (fresh id). A figure cell gets FRESH ids
    for its cell / panels / layers, and each layer's SignalRef is resolved like
    ``report_open`` does: all-resolvable → rebuilt LIVE (re-snapshotted from the
    resolved plots' current_data); otherwise an OFFLINE cell whose baked fallback
    is the provided ``png`` data URL."""
    from spyde.actions.report.handlers import _ensure_open, _insert_cell
    mgr = _ensure_open(session)
    spec_cell = payload.get("cell") or {}
    cell_type = str(spec_cell.get("cell_type", "markdown"))
    index = payload.get("index")

    if cell_type == "markdown":
        cell = Cell(id=new_cell_id(), cell_type="markdown",
                    source=str(spec_cell.get("source", "") or ""))
        if spec_cell.get("html") is not None:
            cell.html = str(spec_cell.get("html") or "")
        _insert_cell(mgr.doc, cell, index)
        mgr.dirty = True
        mgr.emit_state()
        return

    if cell_type != "figure":
        ipc.emit_error(f"report_paste_cell: unsupported cell_type {cell_type!r}.")
        return

    # Figure cell — rebuild the spec with fresh ids so a paste never collides with
    # the source cell's ids.
    spec = FigureSpec.from_dict(spec_cell.get("figure") or {})
    _freshen_spec_ids(spec)
    caption = str(spec_cell.get("caption", "") or "")
    cell = Cell(id=new_cell_id(), cell_type="figure", caption=caption,
                placeholder=False, spec=spec)

    # Resolve every layer against open trees/files; a live rebuild needs a snapshot
    # for EACH layer (same all-or-offline rule as report_open). A scene3d panel
    # rebinds by RECOMPUTING its point cloud from the resolved orientation
    # result (no image layer to read) — same rule as report_open's rebind.
    from spyde.actions.report.handlers import _scene3d_snap_entries
    snap_map: dict = {}
    all_resolved = bool(spec.panels)
    for panel in spec.panels:
        if str(getattr(panel, "kind", "")) == "scene3d":
            entries = _scene3d_snap_entries(session, panel)
            if entries is None:
                all_resolved = False
            else:
                snap_map.update(entries)
            continue
        for layer in panel.layers:
            src_plot = layer.source.resolve(session) if layer.source else None
            arr = None
            if src_plot is not None:
                frame = getattr(src_plot, "current_data", None)
                if isinstance(frame, np.ndarray) and frame.dtype != object:
                    arr = np.array(frame, copy=True)
            if arr is None:
                all_resolved = False
            else:
                snap_map[(panel.id, layer.id)] = arr

    _insert_cell(mgr.doc, cell, index)

    if all_resolved and snap_map:
        mgr._snapshots[cell.id] = snap_map
        mgr._baked.pop(cell.id, None)
        mgr._offline.discard(cell.id)
        mgr.build_figure_window(cell)
    else:
        # Offline: keep the provided PNG as the baked fallback so the renderer can
        # still show the snapshot (its data URL rides along in report_state).
        png = _decode_data_url(spec_cell.get("png"))
        mgr._snapshots.pop(cell.id, None)
        if png is not None:
            mgr._baked[cell.id] = png
        mgr._offline.add(cell.id)

    mgr.dirty = True
    mgr.emit_state()


def _freshen_spec_ids(spec: FigureSpec) -> None:
    """Assign fresh panel + layer ids to a pasted FigureSpec (in place), keeping
    inset ``panel:`` back-references pointing at the renamed panels so callouts
    survive the paste."""
    from spyde.actions.report.model import new_layer_id
    panel_remap: dict = {}
    for i, panel in enumerate(spec.panels):
        old_pid = panel.id
        new_pid = f"p{i + 1}"
        panel_remap[old_pid] = new_pid
        panel.id = new_pid
        for layer in panel.layers:
            layer.id = new_layer_id()
    # Re-point inset panel references at the renamed panels.
    for panel in spec.panels:
        for ins in (panel.insets or []):
            ref = ins.get("panel")
            if ref in panel_remap:
                ins["panel"] = panel_remap[ref]
