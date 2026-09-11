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

from de_shell import ipc
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
/* The figure box owns the SHAPE (width 100% of the column, height from an
   aspect-ratio matching the sidebar cell) and the figure is SCALED to fill it.
   Scaling rather than resizing because a saved figure cannot be resized: the
   renderer lays out from `layout_json`, which carries fig_width/fig_height and
   every panel's size, and only the Python side recomputes that. So the iframe
   keeps its natural pixel size, nothing inside is ever clipped, and CSS maps it
   onto the box, which stays responsive with no JS relayout. */
figure.report-figure .fig-box { position: relative; line-height: 0;
  overflow: hidden; border: 1px solid #e2e2e6; border-radius: 6px;
  /* The figure's own background, so a letterbox left by a box whose aspect
     differs from the figure's reads as part of the figure. */
  background: #1e1e2e; }
figure.report-figure .fig-box iframe { display: block; border: none;
  transform-origin: top left; }
figure.report-figure figcaption { margin-top: 0.6rem; font-size: 0.9rem;
  color: #555; font-style: italic; }
figure.report-figure video { max-width: 100%; height: auto;
  border: 1px solid #e2e2e6; border-radius: 6px; }
/* A movie exported as its poster still: the badge is what tells the reader this
   is one frame of a movie and not a static figure. */
.report-movie .movie-still { position: relative; display: inline-block;
  max-width: 100%; }
.report-movie .movie-badge { position: absolute; left: 50%; top: 50%;
  transform: translate(-50%, -50%); width: 3rem; height: 3rem;
  display: flex; align-items: center; justify-content: center;
  border-radius: 50%; background: rgba(0, 0, 0, 0.55); color: #fff;
  font-size: 1.2rem; line-height: 1; pointer-events: none; }
.report-movie .movie-note { margin-top: 0.4rem; font-size: 0.8rem;
  color: #8a8a92; }
.report-figure--missing .missing-box { border: 1px dashed #c8c8ce;
  border-radius: 6px; padding: 2.5rem 1rem; color: #8a8a92;
  font-size: 0.9rem; background: #fafafb; }
/* Split block (Wave A): a text side BESIDE a figure/photo side. The two columns
   are vertically centered against each other; stacks to one column on a narrow
   viewport so a phone still reads it. The figure column's own figure sizes to its
   column. */
.report-article .split-block { display: grid; grid-template-columns: 1fr 1fr;
  gap: 1.5rem; align-items: center; margin: 1.75rem 0; }
.report-article .split-block--stacked { grid-template-columns: 1fr;
  grid-auto-rows: auto; }
.report-article .split-col { min-width: 0; }
.report-article .split-fig figure.report-figure { margin: 0; }
@media (max-width: 720px) {
  .report-article .split-block { grid-template-columns: 1fr; }
}
@media print {
  body { background: #fff !important; color: #000 !important;
    padding: 0; }
  .report-article { max-width: none; }
  figure.report-figure img, figure.report-figure iframe { border: none; }
}
"""

# The exported figure box is sized the SAME way the sidebar cell is: width 100%
# of the column, height from a CSS aspect-ratio derived from the panel grid. A
# fixed pixel height made every exported figure the same tall box regardless of
# its shape, so a wide 1x3 row was letterboxed and a square pattern stretched.
#
# These MIRROR ReportFigureCell.tsx (PANEL_ASPECT / figureAspectRatio) and are
# pinned against it by test_report_export, because a silent drift between two
# copies is won by whichever one the reader happens to be looking at.
_PANEL_ASPECT = 4 / 3
_DEFAULT_ASPECT = 16 / 10
_VECTORS_ASPECT = 3 / 2
# A tall grid (many rows, one column) would otherwise produce a box metres long.
_MAX_IFRAME_HEIGHT_PX = 720

EMBED_BUDGET_BYTES = 100 * 2**20
"""The ceiling on anything one export inlines, so a self-contained report stays
an openable document rather than a blob no browser will parse. A cell whose
asset is over it exports as its still plus a note naming both the size and the
budget, so the reader knows a bigger original exists."""


def _megabytes(size_bytes: int) -> str:
    """``size_bytes`` as the ``"12.5 MB"`` string the over-budget note prints."""
    return f"{size_bytes / 2**20:.1f} MB"


def _figure_aspect(spec) -> float:
    """The width:height ratio for a cell's figure box, the mirror of
    ``figureAspectRatio`` in ReportFigureCell.tsx."""
    try:
        mode = str(getattr(spec, "vectors_mode", "") or "")
        if mode not in ("", "image"):
            return _VECTORS_ASPECT       # the 2-panel explorer plus its chrome
        layout = getattr(spec, "layout", None) or {}
        if str(layout.get("kind", "")) != "grid":
            return _DEFAULT_ASPECT
        rows = max(1, int(layout.get("rows") or 1))
        cols = max(1, int(layout.get("cols") or 1))
        return (_PANEL_ASPECT * cols) / rows
    except Exception as e:
        log.debug("figure aspect from spec failed: %s", e)
        return _DEFAULT_ASPECT


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
        f"{body_html}\n</article>\n{_IFRAME_AUTOSIZE_JS}</body>\n</html>\n"
    )


# An embed that knows its own height says so; without this the page held every
# interactive figure in one fixed box, leaving a dead band under the controls,
# worst on a phone where the figure scales down and the gap is most of the
# screen. Sandboxed srcdoc iframes are cross-origin, so the frame is identified
# by matching event.source against each contentWindow.
_IFRAME_AUTOSIZE_JS = """<script>
(function () {
  var frames = function () {
    return document.querySelectorAll('figure.report-figure iframe');
  };

  // An embed that measures itself wins: it knows about its own controls below
  // the figure, which no outside measurement can account for.
  window.addEventListener('message', function (e) {
    var d = e.data || {};
    var h = d.spydeEmbedHeight || d.vxHeight;
    if (!h || !isFinite(h)) return;
    var f = frames();
    for (var i = 0; i < f.length; i++) {
      if (f[i].contentWindow === e.source) {
        f[i].dataset.selfSized = '1';
        f[i].style.height = Math.max(120, Math.round(h)) + 'px';
        return;
      }
    }
  });

  // Everything else is SCALED to its box. The iframe carries the figure at its
  // natural pixel size, so nothing inside is ever clipped, and a CSS transform
  // maps it onto the shaped box. Not a resize: the renderer lays out from
  // `layout_json`, which carries fig_width/fig_height AND every panel's size,
  // and only the Python side recomputes it.
  function fit() {
    var boxes = document.querySelectorAll('figure.report-figure .fig-box');
    for (var i = 0; i < boxes.length; i++) {
      var box = boxes[i];
      var f = box.querySelector('iframe');
      if (!f || f.dataset.selfSized === '1') continue;
      var natW = parseFloat(f.style.width), natH = parseFloat(f.style.height);
      if (!natW || !natH) continue;                 // width:100%, nothing to map
      var r = box.getBoundingClientRect();
      if (r.width < 40) continue;
      // Contain, so a box whose aspect differs from the figure's letterboxes
      // rather than distorting or cropping it.
      var k = Math.min(r.width / natW, r.height / natH);
      f.style.transform = 'scale(' + k.toFixed(4) + ')';
      f.style.marginLeft = Math.max(0, (r.width - natW * k) / 2).toFixed(1) + 'px';
      f.style.marginTop = Math.max(0, (r.height - natH * k) / 2).toFixed(1) + 'px';
    }
  }
  window.addEventListener('load', fit);
  if (document.readyState === 'complete') fit();
  // Follow the container, not just the window: an aspect-ratio box changes
  // height whenever the column width does.
  if (typeof ResizeObserver !== 'undefined') {
    var ro = new ResizeObserver(fit);
    document.querySelectorAll('figure.report-figure .fig-box')
      .forEach(function (b) { ro.observe(b); });
  } else {
    window.addEventListener('resize', fit);
  }
})();
</script>
"""


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

    With no pixels (a scene3d cell nobody harvested, an offline figure that never
    baked) the CAPTION still renders, above an empty framed box; ``""`` only when
    there is no caption either. A silent hole reads as "the author wrote
    nothing", which is worse than a visibly missing image."""
    if not png:
        cap = _html.escape(caption or "")
        if not cap:
            return ""
        return (
            "<figure class=\"report-figure report-figure--missing\">"
            "<div class=\"missing-box\">image unavailable</div>"
            f"<figcaption>{cap}</figcaption></figure>"
        )
    b64 = base64.b64encode(png).decode("ascii")
    cap = _html.escape(caption or "")
    figcap = f"<figcaption>{cap}</figcaption>" if cap else ""
    return (
        "<figure class=\"report-figure\">"
        f"<img src=\"data:image/png;base64,{b64}\" alt=\"{cap}\">"
        f"{figcap}</figure>"
    )


def _image_cell_html(cell: Cell, data: "bytes | None") -> str:
    """A static ``<figure><img data:image/<ext>;…></figure>`` for an IMAGE (photo)
    cell — the raw image bytes inlined as a data URL so the exported page is
    self-contained (same principle as the figure PNGs). Returns ``""`` when there
    are no bytes."""
    if not data:
        return ""
    ext = (cell.image_ext or "png").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    b64 = base64.b64encode(data).decode("ascii")
    cap = _html.escape(cell.caption or "")
    figcap = f"<figcaption>{cap}</figcaption>" if cap else ""
    return (
        "<figure class=\"report-figure\">"
        f"<img src=\"data:image/{mime};base64,{b64}\" alt=\"{cap}\">"
        f"{figcap}</figure>"
    )


def _movie_cell_html(mgr, cell: Cell, poster: "bytes | None", *,
                     interactive: bool) -> str:
    """A MOVIE cell's fragment.

    Every export mode gets at least the poster still plus the caption, badged so
    a reader can tell it is a frame OF a movie rather than a static figure. An
    interactive export upgrades to a real ``<video>`` when the movie was rendered
    to a file this session and that file is inside
    :data:`EMBED_BUDGET_BYTES`; a report stays self-contained, so a path
    reference, which would not travel with the HTML, is never emitted."""
    cap = _html.escape(cell.caption or "")
    figcap = f"<figcaption>{cap}</figcaption>" if cap else ""
    src, mime, note = (_inline_movie_src(mgr, cell) if interactive
                       else ("", "", ""))
    if src:
        # An animated GIF plays in an <img>; <video> does not accept one.
        element = (f"<img src=\"{src}\" alt=\"{cap}\">" if mime == "image/gif" else
                   f"<video controls loop playsinline preload=\"metadata\" "
                   f"src=\"{src}\"></video>")
        return (
            "<figure class=\"report-figure report-movie\">"
            f"{element}{figcap}</figure>"
        )
    note_html = f"<div class=\"movie-note\">{_html.escape(note)}</div>" if note else ""
    if not poster:
        if not cap:
            return ""
        return (
            "<figure class=\"report-figure report-movie report-figure--missing\">"
            "<div class=\"missing-box\">movie not yet rendered</div>"
            f"{figcap}</figure>"
        )
    b64 = base64.b64encode(poster).decode("ascii")
    return (
        "<figure class=\"report-figure report-movie\">"
        "<div class=\"movie-still\">"
        f"<img src=\"data:image/png;base64,{b64}\" alt=\"{cap}\">"
        "<span class=\"movie-badge\" aria-label=\"movie\">&#9654;</span>"
        "</div>"
        f"{note_html}{figcap}</figure>"
    )


def _inline_movie_src(mgr, cell: Cell) -> "tuple[str, str, str]":
    """``(data URL, mime, note)`` for the file this cell's movie was rendered to.

    The data URL is ``""`` when there is no such file, when it has since been
    moved or deleted, or when it is over :data:`EMBED_BUDGET_BYTES`. In that last
    case the note is the sentence the export prints under the poster still,
    naming the size and the budget; it is ``""`` otherwise."""
    path = getattr(mgr, "_movie_files", {}).get(cell.id)
    if not path:
        return "", "", ""
    try:
        size = os.path.getsize(path)
        if size > EMBED_BUDGET_BYTES:
            return "", "", (f"Movie not embedded: {_megabytes(size)} is over the "
                            f"{_megabytes(EMBED_BUDGET_BYTES)} export budget.")
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as e:
        log.debug("movie file for cell %s unreadable: %s", cell.id, e)
        return "", "", ""
    mime = "image/gif" if str(path).lower().endswith(".gif") else "video/mp4"
    return (f"data:{mime};base64," + base64.b64encode(raw).decode("ascii"),
            mime, "")


def _figure_iframe_html(caption: str, figure_html: str, *,
                        aspect: float = _DEFAULT_ASPECT,
                        natural_width: int = 0,
                        natural_height: int = 0) -> str:
    """A sandboxed ``<iframe srcdoc>`` embedding a cell's self-contained
    interactive figure HTML (pixels already inlined). The srcdoc content is
    HTML-escaped so the attribute can't be broken out of.

    Sized by ``aspect-ratio`` like the sidebar cell, not a fixed height, so the
    box matches the figure's shape. ``max-height`` catches a tall grid; an embed
    that measures itself overrides both by posting its height (see
    ``_IFRAME_AUTOSIZE_JS``).

    ``natural_width``/``natural_height`` are the figure's own laid-out size. The
    iframe is given exactly that and the page's fit script scales it onto the
    box, so nothing inside is ever clipped and the mapping follows the column at
    any window width."""
    srcdoc = _html.escape(figure_html, quote=True)
    cap = _html.escape(caption or "")
    figcap = f"<figcaption>{cap}</figcaption>" if cap else ""
    natural = (f"width:{int(natural_width)}px;height:{int(natural_height)}px;"
               if natural_width and natural_height else "width:100%;")
    return (
        "<figure class=\"report-figure\">"
        f"<div class=\"fig-box\" style=\"aspect-ratio:{aspect:.4f};"
        f"max-height:{_MAX_IFRAME_HEIGHT_PX}px;\">"
        f"<iframe sandbox=\"allow-scripts\" srcdoc=\"{srcdoc}\" "
        f"style=\"{natural}\" loading=\"lazy\"></iframe>"
        f"</div>{figcap}</figure>"
    )


def _build_interactive_figure_html(mgr, cell: Cell) -> "tuple[str | None, tuple]":
    """``(standalone HTML, natural size)`` for a figure cell's LIVE anyplotlib
    figure — pixels materialised via ``build_cell_figure`` →
    ``_resolve_pixels_for_standalone`` so no binary tokens leak. ``(None, (0, 0))``
    when the cell has no snapshot to rebuild (offline).

    The natural size is the figure's own laid-out pixel size, which is what lets
    the export's box fit the figure exactly instead of letterboxing it."""
    if cell.spec is None:
        return None, (0, 0)
    snap_map = mgr.snapshot_map(cell.id)
    if not snap_map:
        return None, (0, 0)
    try:
        from spyde.actions.report.figure_builder import build_cell_figure
        # standalone=True → the JS bundle is INLINED (no machine-local file:// ESM
        # reference), so the sandboxed srcdoc iframe renders on any machine/browser.
        fig, _fig_id, html_str = build_cell_figure(
            cell.spec, snap_map, standalone=True)
        # Plus the grid padding the renderer adds around the panels.
        width = int(getattr(fig, "fig_width", 0) or 0)
        height = int(getattr(fig, "fig_height", 0) or 0)
        size = (width + 16, height + 16) if (width and height) else (0, 0)
        return html_str, size
    except Exception as e:
        log.debug("interactive figure rebuild failed for cell %s: %s", cell.id, e)
        return None, (0, 0)


def _render_figure_side_html(mgr, cell: Cell, assets: dict, *, interactive: bool,
                             session=None) -> str:
    """The ``<figure>`` fragment for a cell's FIGURE/PHOTO side — the FIGURE-cell
    rendering logic, factored out so a figure cell AND a split cell's figure side
    share ONE path. Interactive mode tries the vectors explorer, then the
    tinted-overlay blender, then the live-figure iframe; anything that can't
    rebuild falls back to the static ``<img>``. A cell with a photo side (a spec-
    less split, or an image cell) inlines the raw ``<img>`` in every mode. Returns
    ``""`` when there are no pixels."""
    # A photo side (no FigureSpec) — always the inlined <img>, in every mode.
    if cell.spec is None:
        return _image_cell_html(cell, assets.get(cell.id))
    aspect = _figure_aspect(cell.spec)
    html_frag = ""
    if interactive:
        # Drop-time choice: vectors_mode == "image" pins the static
        # snapshot even when the tree carries diffraction vectors.
        if cell.spec.vectors_mode != "image":
            try:
                from spyde.actions.report.vectors_embed import (
                    vectors_explorer_html, vectors_for_cell,
                )
                vecs = vectors_for_cell(session, cell)
                if vecs is not None:
                    vx_html = vectors_explorer_html(vecs, caption=cell.caption)
                    if vx_html is not None:
                        html_frag = _figure_iframe_html(cell.caption, vx_html,
                                                        aspect=aspect)
            except Exception as e:
                log.debug("vectors embed for cell %s failed: %s", cell.id, e)
        # ORIENTATION explorer — the same swap for a tree carrying an
        # orientation result. After vectors, because a vector-OM tree carries
        # BOTH and the vectors explorer is the one that cell was dragged from.
        # getattr, not attribute access: a spec here is whatever the caller
        # built, and the vectors path's own stubs predate this field.
        if not html_frag and getattr(cell.spec, "orientation_mode", "") != "image":
            try:
                from spyde.actions.report.orientation_embed import (
                    orientation_explorer_html, orientation_for_cell,
                )
                result = orientation_for_cell(session, cell)
                if result is not None:
                    ox_html = orientation_explorer_html(result,
                                                        caption=cell.caption)
                    if ox_html is not None:
                        html_frag = _figure_iframe_html(cell.caption, ox_html,
                                                        aspect=aspect)
            except Exception as e:
                log.debug("orientation embed for cell %s failed: %s", cell.id, e)
        # Tinted-overlay blender (vectors swap above wins when both
        # apply — a vectors cell stays a vectors explorer).
        if not html_frag:
            try:
                from spyde.actions.report.overlay_embed import (
                    overlay_blender_html,
                )
                ov_html = overlay_blender_html(mgr, cell, caption=cell.caption)
                if ov_html is not None:
                    html_frag = _figure_iframe_html(cell.caption, ov_html,
                                                    aspect=aspect)
            except Exception as e:
                log.debug("overlay blender embed for cell %s failed: %s",
                          cell.id, e)
        if not html_frag:
            fig_html, natural = _build_interactive_figure_html(mgr, cell)
            if fig_html is not None:
                # The figure's OWN shape when we know it, so the box fits it with
                # no letterbox; the grid-derived ratio is the fallback for an
                # embed whose natural size we cannot read.
                shape = (natural[0] / natural[1]) if all(natural) else aspect
                html_frag = _figure_iframe_html(
                    cell.caption, fig_html, aspect=shape,
                    natural_width=natural[0], natural_height=natural[1])
    if not html_frag:
        # Static path (also the interactive OFFLINE fallback).
        html_frag = _figure_img_html(cell.caption, assets.get(cell.id))
    return html_frag


def _split_cell_html(mgr, cell: Cell, assets: dict, *, interactive: bool,
                     session=None) -> str:
    """A SPLIT cell (Wave A) → a 2-column ``.split-block`` grid: the TEXT side
    (its markdown) BESIDE the FIGURE/PHOTO side, ordered by ``split_layout``
    (``text-left`` → text then figure; ``text-right`` → figure then text). The
    figure side reuses :func:`_render_figure_side_html` (interactive iframe / baked
    PNG / photo data URL — all self-contained). An empty figure side just renders
    the text beside an empty column. Reused by the article/static export AND (via
    :func:`_render_slide_rows`) the slides deck."""
    from spyde.actions.report.model import _normalize_split_layout
    text_html = _markdown_cell_html(cell)
    fig_html = _render_figure_side_html(mgr, cell, assets, interactive=interactive,
                                        session=session)
    layout = _normalize_split_layout(cell.split_layout)
    text_col = f"<div class=\"split-col split-text\">\n{text_html}\n</div>"
    fig_col = f"<div class=\"split-col split-fig\">\n{fig_html}\n</div>"
    # Text-first for left/top; figure-first for right/bottom. A "--stacked"
    # modifier switches the block from 2 columns to 2 rows (CSS below).
    text_first = layout in ("text-left", "text-top")
    stacked = layout in ("text-top", "text-bottom")
    first, second = (text_col, fig_col) if text_first else (fig_col, text_col)
    cls = "split-block split-block--stacked" if stacked else "split-block"
    return f"<div class=\"{cls}\">\n{first}\n{second}\n</div>"


def _render_cell_html(mgr, cell: Cell, assets: dict, *, interactive: bool,
                      session=None) -> str:
    """The HTML fragment for ONE cell (markdown, figure, image, split or movie),
    shared by the article body AND the slides shell. Every cell type the document
    model can hold has a branch here: an unhandled type exports as nothing,
    caption and all, which a reader cannot tell from an empty report. A
    placeholder figure → ``""`` (skipped).

    Figure handling mirrors :func:`_render_body`'s per-cell logic (see
    :func:`_render_figure_side_html`). A SPLIT cell renders as a 2-column
    ``.split-block`` (text beside figure/photo)."""
    if cell.cell_type == "markdown":
        return _markdown_cell_html(cell)
    if cell.cell_type == "image":
        # A photo — always the inlined <img> (self-contained), in every mode.
        return _image_cell_html(cell, assets.get(cell.id))
    if cell.cell_type == "split":
        return _split_cell_html(mgr, cell, assets, interactive=interactive,
                                session=session)
    if cell.cell_type == "movie":
        return _movie_cell_html(mgr, cell, assets.get(cell.id),
                                interactive=interactive)
    if cell.cell_type != "figure" or cell.placeholder:
        return ""
    return _render_figure_side_html(mgr, cell, assets, interactive=interactive,
                                    session=session)


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
    overlay_embed.py). No tinted overlay → the live-figure iframe as before.

    A SPLIT cell renders as a self-contained 2-column ``.split-block`` (text
    beside figure/photo — see :func:`_split_cell_html`); every other cell renders
    full width in document order."""
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
figure.report-figure video { max-width: 100%; max-height: 62vh; height: auto;
  border: 1px solid #313244; border-radius: 6px; }
.report-movie .movie-still { position: relative; display: inline-block;
  max-width: 100%; }
.report-movie .movie-badge { position: absolute; left: 50%; top: 50%;
  transform: translate(-50%, -50%); width: 3rem; height: 3rem;
  display: flex; align-items: center; justify-content: center;
  border-radius: 50%; background: rgba(0, 0, 0, 0.55); color: #fff;
  font-size: 1.2rem; line-height: 1; pointer-events: none; }
.report-movie .movie-note { margin-top: 0.4rem; font-size: 0.8rem;
  color: #a6adc8; }
.report-figure--missing .missing-box { border: 1px dashed #45475a;
  border-radius: 6px; padding: 2.5rem 1rem; color: #a6adc8;
  font-size: 0.85rem; background: #1c1c28; }
figure.report-figure figcaption { margin-top: 0.5rem; font-size: 0.85rem;
  color: #a6adc8; font-style: italic; }
/* Split block (Wave A) — the self-contained text-beside-figure cell. A 2-col
   grid; the column ORDER is baked by the export (text-left vs text-right) so no
   CSS reordering is needed. Stacks to one column on a narrow / portrait viewport
   so a phone still reads it. */
.split-block { display: grid; grid-template-columns: 1fr 1fr; gap: 2.5vw;
  align-items: center; }
.split-block--stacked { grid-template-columns: 1fr; grid-auto-rows: auto;
  gap: 1.5vh; }
.split-col { min-width: 0; }
.split-fig figure.report-figure { margin: 0.5rem 0; }
.split-fig figure.report-figure img { max-height: 74vh; }
.split-fig figure.report-figure iframe { height: 56vh; }
@media (max-width: 720px), (orientation: portrait) {
  .split-block { grid-template-columns: 1fr; }
}
/* ── presentation polish: TITLE / SECTION slides ──────────────────────────────
   A title slide (data-kind="title") centers a large title block — the whole
   markdown is scaled up + centered, first heading huge, the rest a muted
   subtitle. Selecting the .slide-inner keeps it inside the normal padded stage. */
.slide[data-kind="title"] { text-align: center; }
.slide[data-kind="title"] .slide-inner {
  max-width: 48rem; display: flex; flex-direction: column;
  justify-content: center; gap: 0.4rem;
}
.slide[data-kind="title"] h1 {
  font-size: 4.2rem; line-height: 1.08; margin: 0 0 0.6rem; font-weight: 800;
  letter-spacing: -0.01em;
}
.slide[data-kind="title"] h2 { font-size: 2.2rem; margin: 0.2rem 0; font-weight: 600;
  color: #cdd6f4; }
.slide[data-kind="title"] h3 { font-size: 1.6rem; color: #a6adc8; font-weight: 500; }
.slide[data-kind="title"] p { font-size: 1.6rem; color: #a6adc8; margin: 0.3rem 0; }
.slide[data-kind="title"] .present-md, .slide[data-kind="title"] .spyde-md { text-align: center; }
/* An accent rule under the title — a subtle branded flourish for section slides. */
.slide[data-kind="title"] h1::after {
  content: ""; display: block; width: 4rem; height: 3px; margin: 1.2rem auto 0;
  background: #89b4fa; border-radius: 2px;
}
/* ── per-slide background/heading presets ─────────────────────────────────────*/
.slide-style-plain { background: #0e0e16; }
.slide-style-accent {
  background: radial-gradient(ellipse at 50% 30%, rgba(137,180,250,0.18), transparent 70%), #14141f;
}
.slide-style-accent h1, .slide-style-accent h2 { color: #b4c6fb; }
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


def _render_slide_rows(mgr, group, assets: dict, *, interactive: bool,
                       session=None) -> "list[str]":
    """Render ONE slide's cells into ordered HTML row blocks
    (:func:`slide_columns`): a ``full`` row is a plain block; a ``split`` row is
    a self-contained SPLIT cell (:func:`_render_cell_html` emits its own
    ``.split-block`` 2-column grid). Reuses :func:`_render_cell_html` for every
    cell so interactive embeds work exactly as before. Rows whose cell renders
    empty are dropped."""
    from spyde.actions.report.model import slide_columns

    def _frags(cells) -> "list[str]":
        return [f for f in (
            _render_cell_html(mgr, c, assets, interactive=interactive,
                              session=session)
            for c in cells) if f]

    rows: list[str] = []
    for row in slide_columns(group):
        # Both "full" and "split" render a single cell — a split cell's
        # _render_cell_html already emits the .split-block grid (text beside
        # figure, ordered by split_layout).
        fr = _frags([row["cell"]])
        if fr:
            rows.append("\n".join(fr))
    return rows


def _render_slides(mgr, assets: dict, *, interactive: bool, session=None) -> str:
    """Render the report as slide `<section class="slide">` blocks, grouped by
    the same ``slide_break`` flag :meth:`ReportDoc.slides` uses. Each slide holds
    every one of its cells' HTML fragments (reusing :func:`_render_cell_html`),
    so a slide's interactive embeds work exactly as in the interactive HTML
    export. Within a slide, a SPLIT cell renders as a self-contained 2-column
    ``.split-block`` (see :func:`_render_slide_rows`). A slide whose cells all
    render empty (e.g. a lone placeholder) is dropped rather than shown blank.

    Presentation POLISH: the slide's per-slide ``slide_kind`` / ``slide_style``
    (read off its first cell via :func:`slide_meta`) are stamped as
    ``data-kind="title"`` + a ``slide-style-<preset>`` class on the ``<section>``,
    which the deck CSS turns into the big-centered title treatment / the
    background preset. ``data-kind``/style are OMITTED when default so an older
    deck's markup is byte-for-byte unchanged.

    SPEAKER NOTES stay AUDIENCE-INVISIBLE: notes are never rendered as slide
    content (they live only on the model, not in any cell's HTML). For a possible
    future web presenter view we stash them as a hidden ``data-notes`` attribute
    (HTML-escaped) on the ``<section>`` — the deck CSS/JS never surfaces it, so
    the audience deck shows nothing. The attribute is OMITTED when a slide has no
    notes, keeping a notes-free deck's markup unchanged."""
    from spyde.actions.report.model import slide_meta

    blocks: list[str] = []
    for group in mgr.doc.slides():
        rows = _render_slide_rows(mgr, group, assets, interactive=interactive,
                                  session=session)
        if not rows:
            continue
        inner = "\n".join(rows)
        meta = slide_meta(group)
        cls = "slide"
        if meta["style"]:
            cls += f" slide-style-{meta['style']}"
        kind_attr = ' data-kind="title"' if meta["kind"] == "title" else ""
        # Speaker notes → a hidden data-notes attribute (escaped so it can't break
        # out of the attribute or inject markup); NEVER rendered as visible slide
        # content, so the audience deck shows nothing. Omitted when there are none.
        notes = str(meta.get("notes") or "")
        notes_attr = (f' data-notes="{_html.escape(notes, quote=True)}"'
                      if notes else "")
        blocks.append(
            f"<section class=\"{cls}\"{kind_attr}{notes_attr}>\n"
            f"<div class=\"slide-inner\">\n{inner}\n</div>\n</section>")
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
    resolved plots' displayed images); otherwise an OFFLINE cell whose baked fallback
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

    if cell_type == "image":
        # A pasted photo carries its bytes as an ``image`` data URL. A fresh id +
        # held bytes, exactly like report_add_image_cell.
        from spyde.actions.report.model import IMAGE_EXTS
        data = _decode_data_url(spec_cell.get("image"))
        if not data:
            ipc.emit_error("report_paste_cell: image cell has no image bytes.")
            return
        ext = str(spec_cell.get("image_ext", "") or "").lower().lstrip(".")
        if ext == "jpeg":
            ext = "jpg"
        if ext not in IMAGE_EXTS:
            ext = "png"
        cell = Cell(id=new_cell_id(), cell_type="image",
                    caption=str(spec_cell.get("caption", "") or ""), image_ext=ext)
        mgr._images[cell.id] = data
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
        if str(panel.kind) == "scene3d":
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
                frame = getattr(src_plot, "displayed_data", None)
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
