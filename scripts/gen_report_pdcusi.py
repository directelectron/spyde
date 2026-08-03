"""gen_report_pdcusi.py — build the PdCuSi crystallization report page.

    python -m scripts.gen_report_pdcusi [out.html]

Reads the vectors written by ``scripts/compute_pdcusi_vectors.py`` (that compute
is 733,200 patterns — far too slow to run inside a docs build) plus a handful of
frames straight from the example file, and writes ONE self-contained HTML page
into ``docs-site/public/media/reports/``. The docs site lists it on its Reports
tab; the file also stands alone if you just open it.

It is a SpyDE report page, not a bespoke one: the skeleton, the article CSS and
the figure/iframe wrappers are ``spyde.actions.report.export_html``'s own, and
the interactive panel is the same ``vectors_explorer_html`` the app's
"interactive" HTML export embeds. What differs is only that the prose is authored
here rather than typed into report cells.
"""
from __future__ import annotations

import html as _html
import io
import os
import sys

import numpy as np

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "docs-site", "public",
    "media", "reports", "pdcusi-crystallization.html")

TITLE = "PdCuSi metallic glass — crystallization, in situ"

# How much of the point block the page carries. 16 bytes per vector, +33% again
# as base64, so the page runs ~21 bytes per embedded vector — the full 6.2 M
# result would be a 130 MB page. Over this budget the SERIES axis is decimated,
# never the scan axes: dropping probe positions puts holes in the count map,
# while dropping series steps only coarsens the time resolution of something
# sampled 400 times. The static crystallization curve always uses every step.
#
# ~900 k lands the page near 20 MB. That is a lot to download, and the reason it
# is acceptable is that the page RENDERS BEFORE IT ARRIVES: HTML parses
# incrementally, the point block is emitted last (inside the explorer's srcdoc at
# the end of the article), and everything above it — prose, the summed patterns,
# the crystallization curve — paints while the rest streams in behind a loading
# card. A poster reader gets a readable page in the first second either way.
EMBED_VECTOR_BUDGET = 900_000


# ── figures ───────────────────────────────────────────────────────────────────

def _fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    return buf.read()


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": "#1a1a1a", "axes.labelcolor": "#1a1a1a",
        "xtick.color": "#555", "ytick.color": "#555",
        "axes.edgecolor": "#d0d0d6", "font.size": 9,
    })
    return plt


def _patterns_figure(sig, steps) -> bytes:
    """One summed diffraction pattern per chosen series step — the amorphous →
    crystalline transition, which is the whole point of the dataset.

    Summed over the scan so a single frame's shot noise doesn't hide the change;
    that is one nav CHUNK per step (the file is chunked one series step at a
    time), so this reads a few hundred MB, not the whole 5.7 GB."""
    plt = _style()
    fig, axs = plt.subplots(1, len(steps), figsize=(3.0 * len(steps), 3.2))
    axs = np.atleast_1d(axs)
    kx = sig.axes_manager.signal_axes[0]
    ext = [kx.offset, kx.offset + kx.scale * (kx.size - 1)] * 2
    for ax, t in zip(axs, steps):
        frame = np.asarray(sig.data[t].sum(axis=(0, 1)).compute(), np.float64)
        ax.imshow(np.log1p(frame), cmap="gray",
                  extent=[ext[0], ext[1], ext[3], ext[2]])
        ax.set_title(f"step {t}", fontsize=10)
        ax.set_xlabel(f"$k_x$ ({kx.units})")
        if ax is axs[0]:
            ax.set_ylabel(f"$k_y$ ({kx.units})")
        else:
            ax.set_yticklabels([])
    fig.tight_layout()
    png = _fig_png(fig)
    plt.close(fig)
    return png


def _series_figure(vecs) -> bytes:
    """Vectors found per series step — the crystallization curve."""
    plt = _style()
    series = np.asarray(vecs.count_map_series(), dtype=np.float64)
    totals = series.reshape(series.shape[0], -1).sum(axis=1)
    n_pos = series.shape[1] * series.shape[2]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(np.arange(totals.size), totals / n_pos, color="#1a5fb4", lw=1.4)
    ax.set_xlabel("series step")
    ax.set_ylabel("vectors per probe position")
    ax.grid(alpha=0.25, lw=0.6)
    ax.margins(x=0.01)
    fig.tight_layout()
    png = _fig_png(fig)
    plt.close(fig)
    return png


def _count_maps_figure(vecs, steps) -> bytes:
    """Where the vectors are, in real space, at each chosen step."""
    plt = _style()
    fig, axs = plt.subplots(1, len(steps), figsize=(2.6 * len(steps), 3.0))
    axs = np.atleast_1d(axs)
    maps = [np.asarray(vecs.count_map_at_t(t), np.float64) for t in steps]
    vmax = max(1.0, max(float(m.max()) for m in maps))
    for ax, t, m in zip(axs, steps, maps):
        im = ax.imshow(m, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"step {t}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=list(axs), fraction=0.03, pad=0.02,
                 label="vectors per position")
    png = _fig_png(fig)
    plt.close(fig)
    return png


# ── embed budget ──────────────────────────────────────────────────────────────

def _decimate_series(vecs, budget: int):
    """Keep every k-th series step until the point block fits ``budget``.

    Returns ``(vectors, keep_every)``. The flat buffer is sorted (t, iy, ix), so
    selecting whole t values preserves that order and ``from_arrays`` can rebuild
    the offsets — the time column is renumbered 0..k-1 so the page's slider still
    reads as consecutive slices."""
    from spyde.signals.diffraction_vectors import (
        SpyDEDiffractionVectors, COL_TIME)

    n = int(len(vecs.flat_buffer))
    n_t = int(vecs.n_time)
    if n <= budget or n_t <= 1:
        return vecs, 1
    keep_every = int(np.ceil(n / budget))
    keep = np.arange(0, n_t, keep_every)
    buf = vecs.flat_buffer
    mask = np.isin(buf[:, COL_TIME].astype(np.int64), keep)
    sub = buf[mask].copy()
    remap = {int(t): i for i, t in enumerate(keep)}
    sub[:, COL_TIME] = [remap[int(t)] for t in sub[:, COL_TIME]]
    ny, nx = vecs.nav_shape
    out = SpyDEDiffractionVectors.from_arrays(
        flat_buffer=sub, full_nav_shape=(len(keep), ny, nx),
        sig_shape=vecs.sig_shape, sig_axes=vecs.sig_axes,
        kernel_radius_px=vecs.kernel_radius_px,
        kernel_radius_data=vecs.kernel_radius_data,
        params=dict(getattr(vecs, "params", {}) or {}),
    )
    return out, keep_every


# ── page ──────────────────────────────────────────────────────────────────────

def _explorer_block(explorer_html: str, caption: str, mb: float) -> str:
    """The explorer figure, with a LOADING CARD in front of it.

    The point block is the last thing in the document, so a reader on a phone
    over conference wifi has the whole article — text, patterns, the
    crystallization curve — long before the explorer's bytes finish arriving.
    Without a placeholder that reads as a page that has stopped loading; with
    one it reads as a panel on its way, which is what is actually happening.

    The card is plain markup that renders as the parser reaches it. The iframe's
    `load` fires once its srcdoc is parsed, and that is when they swap. No
    percentage is claimed: the explorer rides inside the page's own download, and
    a page cannot honestly measure its own transfer while it is still arriving.
    """
    srcdoc = _html.escape(explorer_html, quote=True)
    cap = _html.escape(caption or "")
    figcap = f"<figcaption>{cap}</figcaption>" if cap else ""
    return (
        "<figure class=\"report-figure vx-figure\">"
        "<div class=\"vx-loading\" id=\"vx-loading\">"
        "<div class=\"vx-loading-bar\"><span></span></div>"
        "<div class=\"vx-loading-text\">Loading the interactive explorer — "
        f"about {mb:.0f}&nbsp;MB of diffraction vectors. The rest of the report "
        "is ready to read.</div>"
        "</div>"
        f"<iframe id=\"vx-frame\" sandbox=\"allow-scripts\" srcdoc=\"{srcdoc}\" "
        "style=\"height:520px;visibility:hidden;position:absolute;"
        "left:-99999px;\"></iframe>"
        f"{figcap}"
        "<script>(function(){"
        "var f=document.getElementById('vx-frame'),"
        "l=document.getElementById('vx-loading');"
        "function show(){if(!f)return;"
        "f.style.position='';f.style.left='';f.style.visibility='';"
        "if(l&&l.parentNode)l.parentNode.removeChild(l);}"
        # `load` is the honest signal; the timeout is a backstop so a browser
        # that never fires it (or an explorer that throws) still shows the frame
        # rather than leaving a spinner up forever.
        "if(f){f.addEventListener('load',show);setTimeout(show,20000);}"
        # The explorer reports its real height (see vectors_embed's postMessage
        # block). Without this the frame is a fixed 520 px and the panels, which
        # scale to the container width, leave a dead band underneath — most of
        # the screen on a phone. Only messages from THIS frame are honoured.
        "window.addEventListener('message',function(e){"
        "if(!f||e.source!==f.contentWindow)return;"
        "var h=e.data&&e.data.vxHeight;"
        "if(typeof h==='number'&&h>120&&h<4000)f.style.height=h+'px';});"
        "})();</script>"
        "</figure>"
    )


_LOADING_CSS = """
<style>
figure.vx-figure { position: relative; }
.vx-loading { border: 1px solid #e2e2e6; border-radius: 6px; padding: 1.5rem 1.25rem;
  background: #fafafc; text-align: left; }
.vx-loading-bar { height: 4px; border-radius: 2px; background: #e4e4ea;
  overflow: hidden; }
.vx-loading-bar span { display: block; height: 100%; width: 35%;
  border-radius: 2px; background: #1a5fb4;
  animation: vx-slide 1.1s ease-in-out infinite; }
@keyframes vx-slide {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(340%); }
}
.vx-loading-text { margin-top: 0.75rem; font-size: 0.9rem; color: #555; }
@media (prefers-reduced-motion: reduce) {
  .vx-loading-bar span { animation: none; width: 100%; opacity: 0.5; }
}
</style>
"""


def _p(text: str) -> str:
    return f"<p>{text}</p>"


def _h2(text: str) -> str:
    return f"<h2>{_html.escape(text)}</h2>"


def build_page(vecs, sig, *, params: dict) -> str:
    from spyde.actions.report.export_html import _figure_img_html, _page
    from spyde.actions.report.vectors_embed import vectors_explorer_html

    am = sig.axes_manager
    nav = list(am.navigation_axes)          # axes-manager order: fastest first
    sx, sy = am.signal_axes[0], am.signal_axes[1]
    n_t, ny, nx = (int(s) for s in vecs.full_nav_shape)
    n_pat = n_t * ny * nx
    n_vec = int(len(vecs.flat_buffer))
    series = np.asarray(vecs.count_map_series(), np.float64)
    totals = series.reshape(series.shape[0], -1).sum(axis=1) / (ny * nx)
    lo_i, hi_i = int(np.argmin(totals)), int(np.argmax(totals))
    early = float(totals[: max(1, n_t // 10)].mean())
    late = float(totals[-max(1, n_t // 10):].mean())
    step_pts = sorted({0, n_t // 3, 2 * n_t // 3, n_t - 1})

    embed_vecs, keep_every = _decimate_series(vecs, EMBED_VECTOR_BUDGET)
    explorer = vectors_explorer_html(
        embed_vecs,
        caption=("Navigate the scan and the series; the pattern is redrawn from "
                 "the vectors themselves, in your browser."))

    body = []

    body.append(_h2("The dataset"))
    body.append(_p(
        "A Pd–Cu–Si metallic glass crystallizing under the beam, recorded as a "
        "4D-STEM series at 200&nbsp;kV. Every one of the "
        f"<strong>{n_pat:,}</strong> probe positions carries a full "
        f"{sx.size}&times;{sy.size} diffraction pattern, so the run is a "
        "<em>five-dimensional</em> dataset: a series axis, two scan axes, and "
        "two reciprocal-space axes."))
    body.append(
        "<table><tbody>"
        f"<tr><th>Series steps</th><td>{n_t}</td></tr>"
        f"<tr><th>Scan</th><td>{ny}&nbsp;&times;&nbsp;{nx} positions, "
        f"{nav[0].scale:g}&nbsp;{_html.escape(nav[0].units or '')} step</td></tr>"
        f"<tr><th>Detector</th><td>{sx.size}&nbsp;&times;&nbsp;{sy.size} px, "
        f"{sx.scale:g}&nbsp;{_html.escape(sx.units or '')} per px</td></tr>"
        f"<tr><th>Patterns</th><td>{n_pat:,}</td></tr>"
        "<tr><th>Source</th><td>em-database "
        "<code>PdCuSiCrystallization</code>, Carter Francis "
        "(University of Wisconsin–Madison)</td></tr>"
        "</tbody></table>")
    body.append(_p(
        "One caveat worth stating rather than papering over: the series axis in "
        f"the file is calibrated in {_html.escape(nav[-1].units or '?')} at "
        f"{nav[-1].scale:g} per step, which looks like a scan calibration "
        "carried onto the series axis rather than a real time base. Steps are "
        "therefore numbered here, not dated."))

    body.append(_h2("What the patterns show"))
    body.append(_figure_img_html(
        "Diffraction summed over the whole scan at four points in the series, "
        "log-scaled. The diffuse amorphous halo gives way to discrete "
        "crystalline reflections.",
        _patterns_figure(sig, step_pts)))

    body.append(_h2("Finding the vectors"))
    body.append(_p(
        "Every pattern went through SpyDE's neural disk detector — a small U-Net "
        "trained to find diffraction disks — at a spot size of "
        f"<strong>{params['kernel_radius']}&nbsp;px</strong> and a confidence "
        f"threshold of <strong>{params['threshold']}</strong>, with sub-pixel "
        "refinement on. Spot size is the one scale knob: it sets the "
        "canonical rescale the model sees and the non-maximum-suppression "
        f"distance ({params['min_distance']}&nbsp;px)."))
    if params.get("persistence"):
        body.append(_p(
            "The threshold is deliberately permissive, because a second pass "
            "then re-scores every peak against the peaks found at its "
            "<strong>scan neighbours</strong> and drops the ones no neighbour "
            "confirms. A real reflection persists across adjacent probe "
            "positions; a detector artefact does not. Asking the scan itself "
            "for a second opinion is a stricter filter than raising the "
            "confidence bar, and it uses information a single-frame detector "
            "cannot see."))
    body.append(_p(
        f"That yields <strong>{n_vec:,} diffraction vectors</strong> — "
        f"{n_vec / n_pat:.2f} per pattern averaged over the run. They are stored "
        "as a flat CSR buffer of "
        "<code>(nav_x, nav_y, k<sub>x</sub>, k<sub>y</sub>, step, intensity)</code>, "
        "which is what makes the interactive panel below possible: the whole "
        "result is a few tens of MB of points, not a stack of frames."))

    body.append(_h2("Crystallization shows up as a count"))
    body.append(_figure_img_html(
        "Vectors found per probe position, against series step.",
        _series_figure(vecs)))
    body.append(_p(
        "Nothing here fitted a model or picked a phase — this is just how many "
        "disks the detector found. The count rises from "
        f"{early:.2f} vectors per position over the first tenth of the series to "
        f"{late:.2f} over the last, peaking at step {hi_i} "
        f"({totals[hi_i]:.2f}) and at its lowest at step {lo_i} "
        f"({totals[lo_i]:.2f}). An amorphous pattern has a diffuse halo and "
        "few disks to find; a crystal has sharp reflections. The count is a "
        "crude but honest order parameter."))

    body.append(_figure_img_html(
        "Vectors per probe position in real space, at the same four steps. "
        "Crystallization is not uniform — it starts somewhere.",
        _count_maps_figure(vecs, step_pts)))

    body.append(_h2("Explore it"))
    note = ""
    if keep_every > 1:
        note = (f" Every {keep_every}th series step is embedded "
                f"({embed_vecs.n_time} of {n_t}) to keep the page a reasonable "
                "size; the figures above use all of them.")
    body.append(_p(
        "The panel below is the same explorer SpyDE's interactive HTML export "
        "produces, and it mirrors the app's own window: the left plot picks the "
        "series step, the middle one is the real-space vector count map, and the "
        "right one redraws the diffraction pattern for wherever you point. "
        "Drag the green detector on the pattern to form a virtual image on the "
        "count map. It runs entirely in your browser — no server, no Python. "
        "On a phone it drops the series plot and gives you a slider instead, so "
        "the two images that matter get the width."
        + note))
    if explorer:
        body.append(_explorer_block(
            explorer, "", len(explorer.encode("utf-8")) / 1e6))
    else:
        body.append(_p(
            "<em>(The interactive panel was not built for this page — the vector "
            "count exceeded the embed budget.)</em>"))

    body.append(_h2("Reproducing this"))
    body.append(_p(
        "Everything above is a SpyDE session. Open the dataset from "
        "<strong>Examples → In-situ TEM → PdCuSiCrystallization</strong>, run "
        "<strong>Find Diffraction Vectors</strong> with the settings above, "
        "then drag the vectors window into a report from the sidebar, write "
        "around it, and use <strong>File → Export → HTML (interactive)</strong>. "
        "The page you are reading is that export — the same explorer, the same "
        "self-contained file."))

    return _page(TITLE, _LOADING_CSS + "\n".join(body))


def main(vec_path: str, out_path: str) -> int:
    import hyperspy.api as hs
    from spyde.backend.heavy_imports import ensure_heavy_imports
    from spyde.backend import example_catalogue as catalogue
    from spyde.signals.diffraction_vectors import SpyDEDiffractionVectors

    ensure_heavy_imports()
    if not os.path.exists(vec_path):
        raise SystemExit(
            f"no vectors at {vec_path} — run "
            "`python -m scripts.compute_pdcusi_vectors` first")
    vecs = SpyDEDiffractionVectors.load(vec_path)
    print(f"[report] {len(vecs.flat_buffer):,} vectors, "
          f"full_nav_shape={vecs.full_nav_shape}", flush=True)

    ds = catalogue.resolve("PdCuSiCrystallization")
    path = ds.filepath() if ds else None
    if not path or not os.path.exists(path):
        raise SystemExit("PdCuSiCrystallization is not downloaded")
    sig = hs.load(path, lazy=True)

    from scripts.compute_pdcusi_vectors import PARAMS
    page = build_page(vecs, sig, params=PARAMS)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"[report] wrote {out_path} ({len(page) / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    from scripts.compute_pdcusi_vectors import default_out as _vec_path
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    out = args[0] if args else DEFAULT_OUT
    rc = main(os.path.abspath(_vec_path()), os.path.abspath(out))
    sys.stdout.flush()
    os._exit(rc)
