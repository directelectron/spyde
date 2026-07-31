"""
build_spyde_overview.py — generate ``spyde-overview.spyde-report``, the ~12-minute
conference talk *about* SpyDE, authored *as* a SpyDE presentation.

The deck is a real ``.spyde-report`` container (a zip of ``report.md`` +
``assets/*.png``), built through :mod:`spyde.actions.report.model` — the SAME
writer ``report_save`` uses — so the artifact the app opens is the artifact this
script writes. Slides are plain markdown cells (and SPLIT cells for the
text-beside-screenshot slides); the screenshots are IMAGE cells, so the deck
carries no live signal bindings and opens standalone with no data loaded.

Rebuild after editing the SLIDES table below::

    python doc/presentations/build_spyde_overview.py

Open it in the app: the report sidebar's **Open** button (or the backend action
``report_open`` with ``{"path": ...}``), then **Present**.

Screenshots live in ``doc/presentations/media/`` and are captured by
``electron/tests/talk_screenshots.spec.ts`` (a capture run, not a regression
test). They are downscaled to :data:`IMAGE_WIDTH` here so the committed zip
stays small.
"""
from __future__ import annotations

import io
import os
import sys

# Import spyde from the repo checkout when run in-place (no install needed).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from spyde.actions.report.model import (  # noqa: E402
    Cell, ReportDoc, new_cell_id, write_report,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "media")
OUT = os.path.join(HERE, "spyde-overview.spyde-report")

# Screenshots are captured at the app's full window size (~2800 px wide). Present
# mode never shows a slide image wider than ~1100 px (a split pane is ~half that),
# so downscale before embedding — this is what keeps the committed zip ~1 MB
# instead of ~4 MB.
IMAGE_WIDTH = 1600

# Present mode caps a slide's content column at 60rem, so a split slide gives its
# figure only ~460 CSS px. A raw window capture (2800 px wide, half of it empty
# desktop) is unreadable at that size, so each shot is CROPPED to the region that
# carries meaning — the plot windows — BEFORE it is scaled down. Boxes are
# (left, top, right, bottom) in the captured image's own pixels.
CROPS: dict[str, tuple[int, int, int, int]] = {
    "01-navigator-and-dp.png":    (20, 95, 1450, 770),
    "02-find-vectors-wizard.png": (20, 95, 1450, 1350),
    "03-find-vectors-result.png": (20, 95, 2070, 1430),
    "04-virtual-imaging.png":     (20, 95, 2075, 1165),
    "05-eels.png":                (20, 95, 1350, 790),
}


def _png(name: str) -> bytes:
    """Read ``media/<name>``, crop it per :data:`CROPS`, and cap its width at
    :data:`IMAGE_WIDTH`."""
    path = os.path.join(MEDIA, name)
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        from PIL import Image
    except ImportError:                     # Pillow is a core dep; be forgiving.
        return raw
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    box = CROPS.get(name)
    if box:
        # Clamp to the real image so a re-capture at a different window size
        # degrades to "less cropped" instead of raising.
        left, top, right, bottom = box
        img = img.crop((min(left, img.width - 1), min(top, img.height - 1),
                        min(right, img.width), min(bottom, img.height)))
    if img.width > IMAGE_WIDTH:
        height = round(img.height * IMAGE_WIDTH / img.width)
        img = img.resize((IMAGE_WIDTH, height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ── the deck ──────────────────────────────────────────────────────────────────
#
# Each entry is one SLIDE:
#   text     — the slide's markdown (a split slide's TEXT side)
#   image    — a media/ filename; present → the slide carries a screenshot
#   layout   — "full" (image below the text) or "text-left"/"text-right" (split)
#   kind     — "title" for a title/section slide, "" for a content slide
#   style    — "" (default stage) | "plain" | "accent"
#   notes    — speaker notes (presenter view only, never shown to the audience)
#   seconds  — the time budget; the total is asserted at the bottom of this file
#
# TOTAL BUDGET: ~12 minutes.

SLIDES: list[dict] = [
    # ── 1 ──────────────────────────────────────────────────────────────────────
    dict(
        kind="title", style="accent", seconds=20,
        text=(
            "# SpyDE\n\n"
            "## Interactive analysis for electron microscopy\n\n"
            "HyperSpy · pyxem · anyplotlib · Electron\n\n"
            "*[TODO: your name · affiliation · venue · date]*\n"
        ),
        notes=(
            "Introduce yourself. One sentence on the plan: what SpyDE is, what it\n"
            "stands on, how it's built, and where it's going.\n\n"
            "Timing: this deck is budgeted at ~12.5 min of talking. The two\n"
            "'Active development' slides are yours to fill — they are the slack."
        ),
    ),
    # ── 2 ──────────────────────────────────────────────────────────────────────
    dict(
        seconds=45,
        text=(
            "## Big data, small patience\n\n"
            "- A modern 4D-STEM scan is **hundreds of gigabytes**. A notebook asks "
            "you to decide what to look at *before* you have looked at it.\n"
            "- The loop that actually matters — *move the probe, see the pattern* — "
            "is the one a script is worst at.\n"
            "- HyperSpy already had the data model and the science. What was missing "
            "was a **responsive GUI** that never asks you to down-sample first.\n\n"
            "> \"No data should be too big to analyze.\" — SpyDE docs, FAQ\n"
        ),
        notes=(
            "The pitch: exploration is interactive, and interactivity is an\n"
            "engineering problem, not a science problem.\n\n"
            "If you want a concrete anecdote here, describe the last time you\n"
            "waited on a re-run because you cropped to the wrong region.\n\n"
            "The quote is verbatim from doc/intro.rst (FAQ)."
        ),
    ),
    # ── 3 ──────────────────────────────────────────────────────────────────────
    dict(
        seconds=50, image="01-navigator-and-dp.png", layout="text-left",
        text=(
            "## What SpyDE is\n\n"
            "A **desktop application** for visualising and analysing electron "
            "microscopy data — TEM, STEM, Cryo-EM, 4D-STEM, EELS.\n\n"
            "- Navigator on the left, signal on the right, **live** — including on "
            "lazy data\n"
            "- Opens `.hspy` `.zspy` `.mrc` `.tif` `.tiff` `.de5`\n"
            "- GPL-3.0-or-later, with original support from **Direct Electron**\n"
            "- ~63k lines of Python + ~25k lines of TypeScript\n"
        ),
        notes=(
            "Point at the screenshot: navigator (N-) and signal (S-) windows, the\n"
            "green crosshair, the calibrated k-axis and scale bar on the pattern,\n"
            "the Plot Control dock on the right (histogram/contrast, colormap,\n"
            "signal type, workflow chip, axes table).\n\n"
            "Line counts are `find`+`wc` over spyde/ excluding tests, and\n"
            "electron/src/. Extensions are SUPPORTED_EXTS in\n"
            "spyde/backend/_session_files.py."
        ),
    ),
    # ── 4 ──────────────────────────────────────────────────────────────────────
    dict(
        seconds=50,
        text=(
            "## HyperSpy is the data model\n\n"
            "Everything in SpyDE **is** a HyperSpy `BaseSignal`.\n\n"
            "- **Navigation vs signal axes** — a 4D-STEM scan is 2 nav × 2 signal. "
            "That split is what makes \"move the probe, show the pattern\" a *slice* "
            "rather than a special case.\n"
            "- **Calibrated axes** — scale / offset / units ride with the data, so a "
            "scale bar and a reciprocal-space axis come for free.\n"
            "- **Metadata + signal type** — `electron_diffraction`, `EELS`, … the "
            "type decides which actions are even offered.\n"
            "- **Lazy = Dask** — a signal can be backed by a dask array; nothing "
            "forces it into RAM.\n\n"
            "SpyDE tracks a HyperSpy fork (`cssfrancis/hyperspy`, pinned to a "
            "commit) — the navigator's cached-chunk read lives there.\n"
        ),
        notes=(
            "This is the slide that earns the rest of the talk: SpyDE did not\n"
            "invent a data model, it adopted one.\n\n"
            "The fork is a pinned COMMIT in pyproject.toml, not a branch. The delta\n"
            "is the CachedDaskArray / get_index cache logic the navigator reads\n"
            "through — worth saying it is intended to go upstream if that's the\n"
            "plan. [TODO: say whether upstreaming is planned]"
        ),
    ),
    # ── 5 ──────────────────────────────────────────────────────────────────────
    dict(
        seconds=40,
        text=(
            "## Transformations form a tree, not a script\n\n"
            "`BaseSignalTree` tracks a DAG of signal transformations.\n\n"
            "- **Non-breaking** (filter, centre the direct beam) updates the current "
            "plot in place\n"
            "- **Breaking** (azimuthal integration) branches a **new node**\n"
            "- Every window shows its position in the tree — the *workflow* chip\n"
            "- A finished result (virtual image, strain, orientation) is "
            "**committed** to a new tree, carrying its provenance\n\n"
            "You can walk back up the tree and compare states instead of re-running "
            "a cell and losing the previous one.\n"
        ),
        notes=(
            "Contrast with the notebook: in a notebook, re-running a cell destroys\n"
            "the previous state. Here both states stay addressable.\n\n"
            "'Commit to New Tree' is visible in the virtual-imaging screenshot two\n"
            "slides from now — you can forward-reference it."
        ),
    ),
    # ── 6 ──────────────────────────────────────────────────────────────────────
    dict(
        seconds=55,
        text=(
            "## Don't reimplement the science — inherit it\n\n"
            "| package | what it brings |\n"
            "|---|---|\n"
            "| **pyxem** | 4D-STEM / diffraction: template matching, orientation, strain |\n"
            "| **exspy** | EELS + EDS: edges, models, quantification |\n"
            "| **kikuchipy** | EBSD pattern indexing |\n"
            "| **orix** | orientations, symmetry, IPF colouring |\n"
            "| **atomap** | atomic column finding |\n\n"
            "**pyxem** is a core dependency. **exspy / kikuchipy / atomap** are "
            "extras — `pip install spyde[eels,ebsd,atoms]`; **orix** arrives through "
            "pyxem and kikuchipy.\n\n"
            "A missing extra **hides the buttons** instead of raising — the toolbar's "
            "`requires_package` gate.\n"
        ),
        notes=(
            "The ecosystem argument: these packages are where the domain expertise\n"
            "lives, and they already share HyperSpy's data model, so adopting them\n"
            "costs almost nothing.\n\n"
            "Accuracy note: orix is NOT declared in pyproject.toml — it comes in\n"
            "transitively via pyxem and kikuchipy. lumispy is not a dependency at\n"
            "all; it is deliberately absent from this table.\n"
            "[TODO: mention lumispy / any other package only if you actually use it]\n\n"
            "The requires_package gate is a nice detail: the UI adapts to what is\n"
            "installed rather than erroring."
        ),
    ),
    # ── 7 ──────────────────────────────────────────────────────────────────────
    dict(
        seconds=40, image="05-eels.png", layout="text-right",
        text=(
            "## One app, several techniques\n\n"
            "The signal type drives the UI, so the same shell serves very different "
            "data.\n\n"
            "- A **4D-STEM** scan gets a diffraction toolbar and a k-calibrated "
            "pattern\n"
            "- An **EELS** spectrum image gets an energy-loss axis, edges and "
            "model fitting\n"
            "- Same navigator, same tree, same report\n\n"
            "*Synthetic EELS spectrum image — nav 16 × 16, 1024 channels, "
            "power-law background with C / N / O K edges.*\n"
        ),
        notes=(
            "Point at the energy-loss axis in eV, the acceleration voltage and\n"
            "convergence angle picked up from metadata, and the signal type set to\n"
            "EELS in the dock.\n\n"
            "This is bundled synthetic data (spyde.data.eels_si) whose ground truth\n"
            "is stored on the metadata, so a fit can be scored against the numbers\n"
            "the data was built from.\n\n"
            "[TODO: swap in a screenshot of YOUR real EELS data if you'd rather "
            "show that]"
        ),
    ),
    # ── 8 ──────────────────────────────────────────────────────────────────────
    dict(
        seconds=55,
        text=(
            "## Two processes, one line protocol\n\n"
            "```\n"
            "  Electron main (Node)  ──spawn──▶   python -m spyde\n"
            "         │                                 │\n"
            "    IPC / preload                  asyncio stdin/stdout\n"
            "         │                           PLOTAPP: JSON lines\n"
            "         ▼                                 │\n"
            "  React + TypeScript renderer  ◀───────────┘\n"
            "```\n\n"
            "- The **scientific stack stays in Python** — HyperSpy, pyxem, Dask, torch\n"
            "- The **UI is a modern web stack** — React 18, TypeScript 5, Electron 34\n"
            "- The boundary is a **line protocol**, so the backend tests headlessly "
            "and the frontend tests under Playwright\n"
            "- Image pixels bypass JSON entirely (raw binary frames)\n"
        ),
        notes=(
            "Why not PyQt: SpyDE started as a PySide6/pyqtgraph app and was migrated.\n"
            "The split buys a modern UI toolkit without dragging the scientific\n"
            "stack into it, and a hard testable seam between the two.\n\n"
            "The protocol is `PLOTAPP:`-prefixed JSON lines on stdout, from\n"
            "anyplotlib._electron. Binary frames ride a separate PLOTBIN path.\n\n"
            "[TODO: add a sentence on the migration if the audience knows the old "
            "Qt app]"
        ),
    ),
    # ── 9 ──────────────────────────────────────────────────────────────────────
    dict(
        seconds=50, image="04-virtual-imaging.png", layout="text-left",
        text=(
            "## anyplotlib — the plotting layer\n\n"
            "Figures are **HTML embedded in the renderer**, not a native widget.\n\n"
            "- Interactive widgets — crosshair, ROI, span — are anyplotlib's; SpyDE "
            "binds them to navigation axes\n"
            "- **Tiled display** for large frames: a downsampled overview plus a "
            "hi-res detail tile of exactly what's on screen — crisp zoom without "
            "shipping 16 megapixels\n"
            "- Co-developed alongside SpyDE, released on PyPI (`anyplotlib>=0.4.2`)\n\n"
            "One renderer serves the app, the exported HTML report, and the docs.\n"
        ),
        notes=(
            "The screenshot: a virtual detector (the red disk) dropped on the\n"
            "diffraction pattern, and the virtual image on the right building live\n"
            "as you drag it. That interaction is an anyplotlib widget bound to a\n"
            "HyperSpy ROI.\n\n"
            "Browser-native was the right call because the figure you interact with\n"
            "and the figure you export are literally the same object."
        ),
    ),
    # ── 10 ─────────────────────────────────────────────────────────────────────
    dict(
        seconds=55,
        text=(
            "## The hard part: staying live on lazy data\n\n"
            "Moving the probe has to feel instant on a dataset that does not fit in "
            "RAM.\n\n"
            "- **Storage-aligned chunking** — load with chunks that span whole "
            "signal frames, so one pattern is one chunk read. Never `rechunk()` a "
            "multi-gigabyte array to fix it after the fact.\n"
            "- **One serial dispatcher, latest-position-wins** — no locks, no "
            "thread per move; a superseded position is dropped before it ever runs.\n"
            "- **Two caches** — decoded frames, and decoded navigation *blocks*, "
            "because a compressed chunk is atomic: reading one frame costs what "
            "reading all of them costs.\n"
            "- **Tiered reads** — cheap reads run synchronously on the dispatcher; "
            "only reads that would genuinely freeze the UI go async and cancellable.\n\n"
            "Region integration on a 64 × 64 × 256² scan: **2850 ms → ~5 ms** per "
            "drag step.\n"
        ),
        notes=(
            "This is the engineering slide — the one that separates 'a GUI over\n"
            "HyperSpy' from 'a GUI that stays live'.\n\n"
            "The honest framing: almost every obvious fix here was tried and made\n"
            "it worse. Per-update threads raced the chunk cache. A lock held across\n"
            "the compute wedged. A one-entry block memo re-decoded on every chunk\n"
            "crossing. What survives is serial + latest-wins + LRU caches.\n\n"
            "Numbers are from the project's own benchmarks (CLAUDE.md /\n"
            "benchmarks.md).\n"
            "[TODO: confirm the machine these were measured on if asked]"
        ),
    ),
    # ── 11 ─────────────────────────────────────────────────────────────────────
    dict(
        seconds=50, image="03-find-vectors-result.png", layout="text-right",
        text=(
            "## GPU where it pays\n\n"
            "- **Peak finding** — classical (difference-of-Gaussians, normalised "
            "cross-correlation) *and* a neural detector, both on torch\n"
            "- **Orientation mapping** — the whole scan is fit **at once**: every "
            "pattern packed into one batched tensor, coarse-seeded by angular "
            "cross-correlation, refined with Adam. No dask, no per-pattern loop.\n"
            "- Rewriting the coarse seed from a Python loop over templates into one "
            "batched FFT correlation: **289 s → 1.6 s**\n"
            "- CUDA *and* Apple MPS, always with a working CPU fallback\n"
        ),
        notes=(
            "The screenshot is a real run on bundled synthetic Si grains — 701\n"
            "diffraction vectors found (see the status bar), overlaid in red on the\n"
            "pattern, with the vector count map as a new window.\n\n"
            "The lesson worth stating out loud: when a GPU step is slow the cause is\n"
            "almost always a Python loop launching tiny kernels, not the arithmetic.\n"
            "289s -> 1.6s was a restructuring, not a faster card.\n\n"
            "[TODO: confirm the GPU these numbers were measured on]"
        ),
    ),
    # ── 12 ─────────────────────────────────────────────────────────────────────
    dict(
        seconds=40, image="02-find-vectors-wizard.png", layout="text-left",
        text=(
            "## Interaction is the feature\n\n"
            "Heavy compute is staged behind a **live preview**: tune the parameters "
            "against the pattern under the crosshair, *then* commit to the full "
            "scan.\n\n"
            "- Preview updates as you move the navigator — before any full-dataset "
            "compute\n"
            "- The same shape serves virtual imaging, FFT, line profiles, strain and "
            "orientation mapping\n"
            "- Results open **early** and fill in progressively\n"
        ),
        notes=(
            "Point at the red circles on the pattern — that is the peak finder\n"
            "running live under the crosshair with the parameters currently in the\n"
            "wizard. You never launch a 20-minute job on a guess.\n\n"
            "Note the neural detector (SpotUNet) in the Method dropdown alongside\n"
            "the classical methods.\n\n"
            "This is the 'Wizard' shape in spyde/actions/README.md: open, tune, run,\n"
            "commit, close."
        ),
    ),
    # ── 13 ─────────────────────────────────────────────────────────────────────
    dict(
        seconds=45,
        text=(
            "## …and this talk is a SpyDE document\n\n"
            "The Report Builder turns a session into a document. Drag a figure out "
            "of a window into the report and it stays **live and re-bindable** — "
            "reopen the report with the data loaded and the figure re-renders from "
            "the signal.\n\n"
            "- `.spyde-report` is a plain **zip**: `report.md` + `figures/*.yaml` + "
            "`assets/*.png` — valid markdown you can unzip and hand to pandoc\n"
            "- One document, several surfaces: scrolling report, **slide deck**, "
            "movie editor\n"
            "- Exports to static HTML, interactive HTML, PDF, or a markdown folder\n"
            "- Present mode has a presenter view with speaker notes — press **S**\n\n"
            "**This deck is `doc/presentations/spyde-overview.spyde-report`.**\n"
        ),
        notes=(
            "The reveal. Worth pausing on: you are looking at the feature.\n\n"
            "Press S here if you want to show the presenter view live — it is the\n"
            "same screen (SpyDE is a single Electron window), showing the current\n"
            "slide, the next one, these notes, and a timer.\n\n"
            "The 'no JSON anywhere' choice is deliberate: the document stays\n"
            "readable and diffable in git."
        ),
    ),
    # ── 14 ── section divider ──────────────────────────────────────────────────
    dict(
        kind="title", style="accent", seconds=10,
        text="# Active development\n\n## Where this is going\n",
        notes="Breath. Transition into the part of the talk that is yours.",
    ),
    # ── 15 ── OPEN PLACEHOLDER ─────────────────────────────────────────────────
    dict(
        seconds=55,
        # OPEN PLACEHOLDER — deliberately unwritten. A bare "-" renders as an
        # invisible empty <li>, so the blanks are "…" instead: visible bullets
        # that unmistakably read as "fill me in" on the projected slide.
        text=(
            "## Active development\n\n"
            "- …\n"
            "- …\n"
            "- …\n"
            "- …\n\n"
            "*[TODO: fill these in — the speaker notes suggest a shape.]*\n"
        ),
        notes=(
            "[TODO: FILL IN — this slide is deliberately empty.]\n\n"
            "Suggested shape: what is being built right now, who is using it, what\n"
            "changed since the last time this audience saw it. Add or remove bullets\n"
            "freely; the deck's time budget gives this slide ~60 s."
        ),
    ),
    # ── 16 ── OPEN PLACEHOLDER ─────────────────────────────────────────────────
    dict(
        seconds=40,
        # OPEN PLACEHOLDER — see the note on the previous slide.
        text=(
            "## Roadmap\n\n"
            "- …\n"
            "- …\n"
            "- …\n\n"
            "*[TODO: fill these in — the speaker notes suggest a shape.]*\n"
        ),
        notes=(
            "[TODO: FILL IN — this slide is deliberately empty.]\n\n"
            "Suggested shape: the next milestone, what you want help with, and how\n"
            "someone in the room could contribute. Budgeted ~45 s."
        ),
    ),
    # ── 17 ─────────────────────────────────────────────────────────────────────
    dict(
        seconds=25,
        text=(
            "## Try it\n\n"
            "- **Download** — signed macOS build, Windows installer, Linux AppImage: "
            "`github.com/CSSFrancis/spyde/releases`\n"
            "- **From source** — Node 18+ and `uv`, then `uv sync --extra tests` and "
            "`npm run dev`\n"
            "- **Docs** — `directelectron.github.io/spyde`\n"
            "- Python **3.10–3.13** · **GPL-3.0-or-later**\n\n"
            "Built on the work of the HyperSpy and pyxem communities.\n"
        ),
        notes=(
            "First launch bootstraps its own Python environment with uv, including\n"
            "the GPU-correct torch wheel, so 'download and run' really is the path.\n"
            "Windows installers are currently unsigned — expect a SmartScreen\n"
            "warning.\n\n"
            "[TODO: confirm which release/version you want to point people at]"
        ),
    ),
    # ── 18 ─────────────────────────────────────────────────────────────────────
    dict(
        kind="title", style="accent", seconds=10,
        text=(
            "# Thank you\n\n"
            "## Questions?\n\n"
            "`github.com/CSSFrancis/spyde`\n"
        ),
        notes="[TODO: add your contact details / acknowledgements]",
    ),
]


def build() -> ReportDoc:
    """Assemble :data:`SLIDES` into a presentation ``ReportDoc`` + its assets."""
    doc = ReportDoc(title="SpyDE — an overview", doc_type="presentation")
    assets: dict[str, bytes] = {}

    for i, s in enumerate(SLIDES):
        cid = new_cell_id()
        image = s.get("image")
        layout = s.get("layout", "full")
        # A slide with an image and a split layout is ONE atomic split cell (text
        # beside the screenshot). Anything else is a markdown cell, optionally
        # followed by a full-width image cell.
        is_split = bool(image) and layout in ("text-left", "text-right")
        cell = Cell(
            id=cid,
            cell_type="split" if is_split else "markdown",
            source=s["text"],
            # Present-mode per-slide attributes ride on the slide's FIRST cell.
            # A leading break on cell 0 is a harmless no-op (see ReportDoc.slides).
            slide_break=True,
            slide_kind=s.get("kind", ""),
            slide_style=s.get("style", ""),
            notes=s.get("notes", ""),
        )
        if is_split:
            cell.split_layout = layout
            cell.image_ext = "png"
            cell.caption = ""
            assets[cid] = _png(image)
        doc.cells.append(cell)

        if image and not is_split:
            img_id = new_cell_id()
            doc.cells.append(Cell(id=img_id, cell_type="image",
                                  image_ext="png", caption=""))
            assets[img_id] = _png(image)

        del i
    return doc, assets


def main() -> None:
    doc, assets = build()
    write_report(doc, OUT, assets)

    total = sum(int(s["seconds"]) for s in SLIDES)
    n_slides = len(doc.slides())
    size_kb = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT}")
    print(f"  {n_slides} slides · {len(doc.cells)} cells · {size_kb:.0f} KB")
    print(f"  budget {total} s = {total / 60:.1f} min")
    assert n_slides == len(SLIDES), (
        f"slide grouping mismatch: {n_slides} groups vs {len(SLIDES)} entries")


if __name__ == "__main__":
    main()
