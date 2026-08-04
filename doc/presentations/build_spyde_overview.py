"""
build_spyde_overview.py — generate ``spyde-overview.spyde-report``, the ~12-minute
conference talk *about* SpyDE, authored *as* a SpyDE presentation.

The deck is a real ``.spyde-report`` container (a zip of ``report.md`` +
``assets/*.png``), built through :mod:`spyde.actions.report.model` — the SAME
writer ``report_save`` uses — so the artifact the app opens is the artifact this
script writes. Slides are plain markdown cells (and SPLIT cells for the
text-beside-screenshot slides); the screenshots are IMAGE cells, so the deck
carries no live signal bindings and opens standalone with no data loaded.

Three tables are the whole file, and they are the things you edit:

* :data:`SPEAKER` — who is giving the talk. It feeds the title slide, the
  closing slide AND the theme's footer bar, so a venue change is one edit.
* :data:`THEME`   — the deck's look, written into the document's ``theme:``
  front matter. It travels with the file: hand the deck to a colleague and it
  still looks like this.
* :data:`SLIDES`  — the talk itself.

Rebuild after editing any of them::

    python doc/presentations/build_spyde_overview.py

Open it in the app: the report sidebar's **Open** button (or the backend action
``report_open`` with ``{"path": ...}``), then **Present**.

Screenshots live in ``doc/presentations/media/`` and are captured by
``electron/tests/talk_screenshots.spec.ts`` (a capture run, not a regression
test). They are downscaled to :data:`IMAGE_WIDTH` here so the committed zip
stays small.
"""
from __future__ import annotations

import base64
import io
import os
import sys

# Import spyde from the repo checkout when run in-place (no install needed).
# THREE levels up: <repo>/doc/presentations/<this file>. Two levels lands on
# doc/, which still imported fine from a repo-root cwd — and silently made the
# logo lookup below miss.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from spyde.actions.report.model import (  # noqa: E402
    Cell, ReportDoc, new_cell_id, normalize_theme, write_report,
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


# ── who is giving the talk ────────────────────────────────────────────────────
#
# One place, three consumers: the title slide, the closing slide, and the theme
# footer that runs along the bottom of every content slide. VENUE and DATE are
# blank by default — set them for a specific booking and the title slide picks
# them up; leave them blank and the line is simply omitted rather than printing
# an empty separator.

SPEAKER = {
    "name": "Carter Francis",
    "role": "R&D Scientist",
    "org": "Direct Electron",
    "email": "cfrancis@directelectron.com",
    "venue": "",       # e.g. "M&M 2026"
    "date": "",        # e.g. "August 2026"
}

# The links the audience is asked to write down. Both are verified live; the
# directelectron.github.io/spyde and github.com/directelectron/spyde URLs in
# pyproject.toml are NOT yet published, so pointing the room at them would send
# it to a 404. Change these here if the project moves.
REPO_URL = "github.com/CSSFrancis/spyde"
DOCS_URL = "cssfrancis.github.io/spyde"


# ── the deck's look ───────────────────────────────────────────────────────────
#
# Serialized into the document's ``theme:`` front matter, so it travels with the
# file. Colours reach the slide markdown as CSS custom properties on the deck
# root; the footer is drawn on every slide EXCEPT title/section cards, which
# carry their own attribution.
#
# The palette is the app's own: every screenshot in this deck is a dark SpyDE
# window, so a light deck would frame each one in a bright box. The accent is
# SpyDE's blue (#89b4fa) for the same reason — the slide headings and the app
# chrome in the screenshots then agree.

#: Footer logo. The dark-background variant, because the deck is dark — the
#: light `icon.png` carries black web lines that vanish on a dark bar.
LOGO_SRC = os.path.join(_REPO, "spyde", "SpydeDark.png")
LOGO_PX = 96          # the logo is drawn ~28 px tall; 96 px covers 3× displays

THEME = {
    "bg": "#12121c",
    "text": "#e9ecf3",
    "muted": "#a6adc8",
    "accent": "#89b4fa",
    # Interface stacks in preference order; every entry after the first is a
    # fallback for a machine that lacks it, so the deck degrades to the host's
    # own UI font rather than to Times.
    "font": ('Inter, "Segoe UI", -apple-system, BlinkMacSystemFont, '
             'system-ui, Roboto, "Helvetica Neue", Arial, sans-serif'),
    "logo_height": 28,
    "footer_show": True,
    "footer_name": SPEAKER["name"],
    "footer_email": SPEAKER["email"],
    "footer_note": SPEAKER["org"],
    "slide_numbers": True,
}


def _logo_data_url() -> str:
    """The footer logo as a ``data:`` URL, downscaled to :data:`LOGO_PX`.

    A data URL rather than a path because the deck has to survive being emailed
    to someone whose disk has never had this repo on it."""
    try:
        from PIL import Image
    except ImportError:                      # Pillow is a core dep; be forgiving.
        return ""
    if not os.path.exists(LOGO_SRC):
        return ""
    img = Image.open(LOGO_SRC).convert("RGBA")
    if img.width > LOGO_PX:
        img = img.resize((LOGO_PX, round(img.height * LOGO_PX / img.width)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


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


def _title_slide_text() -> str:
    """The title card, composed from :data:`SPEAKER` so a venue change is one
    edit. Blank fields drop out instead of leaving dangling separators."""
    who = " · ".join(x for x in (SPEAKER["name"], SPEAKER["role"],
                                 SPEAKER["org"]) if x)
    where = " · ".join(x for x in (SPEAKER["venue"], SPEAKER["date"]) if x)
    lines = [
        "# SpyDE\n",
        "## Interactive analysis for electron microscopy\n",
        f"{who}\n",
        f"{SPEAKER['email']}\n",
    ]
    if where:
        lines.append(f"*{where}*\n")
    return "\n".join(lines)


def _closing_slide_text() -> str:
    return (
        "# Thank you\n\n"
        "## Questions?\n\n"
        f"{SPEAKER['name']} · {SPEAKER['email']}\n\n"
        # Plain, not a code span: on a title card the code chip's pink is the
        # only pink on the slide and pulls the eye off the address.
        f"{REPO_URL}\n"
    )


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
# TOTAL BUDGET: ~12.5 minutes. Nothing on a slide is a placeholder — anything
# still to be decided lives in the speaker notes, where the audience can't read
# it off a projector.

SLIDES: list[dict] = [
    # ── 1 ──────────────────────────────────────────────────────────────────────
    dict(
        kind="title", style="accent", seconds=20,
        text=_title_slide_text(),
        notes=(
            "Introduce yourself, then set the plan in one sentence: where this\n"
            "came from, what it is built on, how it works, and where it is going.\n\n"
            "Timing: the deck is budgeted at ~12.5 min of talking, printed by the\n"
            "build script on every rebuild. If you are given 15, the two slides\n"
            "with the most give are 'Active development' and 'Roadmap'.\n\n"
            "Set SPEAKER['venue'] / SPEAKER['date'] in the build script and the\n"
            "line under your name appears; leave them blank and it is omitted."
        ),
    ),
    # ── 2 ──────────────────────────────────────────────────────────────────────
    dict(
        seconds=45,
        text=(
            "## Where this comes from\n\n"
            "**Graduate school.** A PhD with Paul Voyles at UW–Madison, measuring "
            "disorder in glasses with 4D-STEM. The microscope was ready years "
            "before the analysis was.\n\n"
            "**Open source.** So I wrote some of the analysis, and then helped "
            "maintain it — **HyperSpy** and **pyxem**, which is where most of my "
            "open-source time still goes.\n\n"
            "**Direct Electron.** Now the same problem from the other side: a "
            "modern detector produces data faster than any person can look at it.\n\n"
            "SpyDE is what those two jobs look like when you do them at the same "
            "time.\n"
        ),
        notes=(
            "Keep this to about 45 seconds — it is context, not a CV.\n\n"
            "The honest through-line: every job I have had has been the same\n"
            "complaint from a different chair. In grad school the data outran the\n"
            "tools; on the detector side the tools have to keep up with hardware\n"
            "that got much faster.\n\n"
            "If the room is mostly pyxem/HyperSpy users, this is also the moment\n"
            "to say that SpyDE is not a competitor to either — it is a front end\n"
            "for both, and the fixes go upstream."
        ),
    ),
    # ── 3 ──────────────────────────────────────────────────────────────────────
    dict(
        seconds=35,
        text=(
            "## Big data, small patience\n\n"
            "- A modern 4D-STEM scan is **hundreds of gigabytes**. A script asks "
            "you to decide what to look at *before* you have looked at it.\n"
            "- The loop that actually matters — *move the probe, see the pattern* "
            "— is the one a notebook is worst at.\n"
            "- HyperSpy already had the data model and the science. What was "
            "missing was a **responsive interface** that never asks you to "
            "down-sample first.\n\n"
            "> \"No data should be too big to analyze.\"\n"
        ),
        notes=(
            "The pitch: exploration is interactive, and interactivity is an\n"
            "engineering problem rather than a science problem.\n\n"
            "A concrete anecdote lands well here — the last time you waited on a\n"
            "re-run because you had cropped to the wrong region.\n\n"
            "The quote is verbatim from doc/intro.rst (FAQ)."
        ),
    ),
    # ── 4 ──────────────────────────────────────────────────────────────────────
    dict(
        seconds=45, image="01-navigator-and-dp.png", layout="text-left",
        text=(
            "## What SpyDE is\n\n"
            "A **desktop application** for visualising and analysing electron "
            "microscopy data — TEM, STEM, cryo-EM, 4D-STEM, EELS.\n\n"
            "- Navigator beside the signal, **live**, including on data far larger "
            "than memory\n"
            "- Opens `.hspy` `.zspy` `.mrc` `.tif` `.de5`, and DE's sparse `.csb` "
            "event streams\n"
            "- Every operation is a HyperSpy operation, so nothing you do here "
            "traps you here\n"
            "- Free software — **GPL-3.0-or-later**, developed at Direct Electron\n"
        ),
        notes=(
            "Point at the screenshot: the navigator (N-) and signal (S-) windows,\n"
            "the green crosshair, the calibrated k-axis and scale bar on the\n"
            "pattern, and the Plot Control dock on the right — histogram and\n"
            "contrast, colormap, signal type, workflow chip, axes table.\n\n"
            "'Nothing traps you here' is worth saying slowly: the signal tree is\n"
            "HyperSpy objects, so you can drop into the built-in Python console\n"
            "at any point and keep working in code.\n\n"
            "Extensions are SUPPORTED_EXTS in spyde/backend/_session_files.py.\n"
            "Scale, if asked: ~65k lines of Python plus ~26k of TypeScript."
        ),
    ),
    # ── 5 ──────────────────────────────────────────────────────────────────────
    dict(
        seconds=50,
        text=(
            "## HyperSpy is the data model\n\n"
            "Everything in SpyDE **is** a HyperSpy `BaseSignal`.\n\n"
            "- **Navigation vs signal axes** — a 4D-STEM scan is 2 nav × 2 signal, "
            "which makes \"move the probe, show the pattern\" a *slice* rather than "
            "a special case\n"
            "- **Calibrated axes and metadata** — scale, offset, units and signal "
            "type ride with the data; the signal type decides which actions are "
            "even offered\n"
            "- **Lazy = Dask** — nothing forces a dataset into RAM\n\n"
            "Transformations form a **tree, not a script**. Non-breaking steps "
            "update the plot in place; breaking ones branch a new node; a finished "
            "result is committed with its provenance. You can walk back and compare "
            "states instead of re-running a cell and losing the previous one.\n"
        ),
        notes=(
            "This is the slide that earns the rest of the talk: SpyDE did not\n"
            "invent a data model, it adopted one. That is why five other packages\n"
            "drop straight in two slides from now.\n\n"
            "The tree is the contrast with a notebook: re-running a cell destroys\n"
            "the previous state, and here both states stay addressable.\n\n"
            "SpyDE currently tracks a HyperSpy fork pinned to a commit — the delta\n"
            "is the cached-chunk read the navigator goes through, and it is meant\n"
            "to go upstream. Say so if anyone asks; don't volunteer it."
        ),
    ),
    # ── 6 ──────────────────────────────────────────────────────────────────────
    dict(
        seconds=40,
        text=(
            "## Don't reimplement the science — inherit it\n\n"
            "| package | what it brings |\n"
            "|---|---|\n"
            "| **pyxem** | 4D-STEM: template matching, orientation, strain |\n"
            "| **exspy** | EELS + EDS: edges, models, quantification |\n"
            "| **kikuchipy** | EBSD pattern indexing |\n"
            "| **orix** | orientations, symmetry, IPF colouring |\n"
            "| **atomap** | atomic column finding |\n\n"
            "An enormous amount of excellent work already exists. SpyDE's job is to "
            "put one consistent interface on it and hand it back to the "
            "community — **free**.\n\n"
            "`pip install spyde[eels,ebsd,atoms]`; a missing extra **hides the "
            "buttons** rather than raising.\n"
        ),
        notes=(
            "The ecosystem argument: these packages are where the domain expertise\n"
            "lives, and they already share HyperSpy's data model, so adopting them\n"
            "costs almost nothing.\n\n"
            "Accuracy, in case it comes up: pyxem is a core dependency. orix is\n"
            "not declared directly — it arrives through pyxem and kikuchipy.\n"
            "lumispy is not a dependency at all, which is why it is not listed.\n\n"
            "The requires_package gate is a good detail: the UI adapts to what is\n"
            "installed instead of erroring at the user."
        ),
    ),
    # ── 7 ──────────────────────────────────────────────────────────────────────
    dict(
        seconds=30, image="05-eels.png", layout="text-right",
        text=(
            "## One app, several techniques\n\n"
            "The signal type drives the interface, so one shell serves very "
            "different data.\n\n"
            "- A **4D-STEM** scan gets a diffraction toolbar and a k-calibrated "
            "pattern\n"
            "- An **EELS** spectrum image gets an energy-loss axis, edges and model "
            "fitting\n"
            "- Same navigator, same tree, same report\n"
        ),
        notes=(
            "Point at the energy-loss axis in eV, the acceleration voltage and\n"
            "convergence angle picked up from metadata, and the signal type set to\n"
            "EELS in the dock.\n\n"
            "This is bundled synthetic data (spyde.data.eels_si) — nav 16 x 16,\n"
            "1024 channels, power-law background with C / N / O K edges — whose\n"
            "ground truth is stored on the metadata, so a fit can be scored against\n"
            "the numbers the data was built from.\n\n"
            "Swap in a screenshot of your own EELS data if you would rather show\n"
            "that; re-crop in CROPS afterwards."
        ),
    ),
    # ── 8 ──────────────────────────────────────────────────────────────────────
    dict(
        seconds=40,
        text=(
            "## Open, or it isn't reproducible\n\n"
            "There are two ways to ship software. **Apple's** — polished, closed, "
            "and you take its word for it. **Linux's** — you can read every line, "
            "and it still runs in ten years.\n\n"
            "- A number you cannot re-derive is an anecdote. If the analysis lives "
            "in a black box, *\"processed in version 4.2\"* is the whole methods "
            "section.\n"
            "- Science needs the other property: open the same file years later, "
            "run the same pipeline, get the same answer — or see exactly what "
            "changed.\n"
            "- Every layer here is readable — the data model, the algorithms, the "
            "file format, the application itself.\n\n"
            "SpyDE wants the polish of the first and the guarantees of the second.\n"
        ),
        notes=(
            "This is the argument slide. Deliver the Apple/Linux line as a\n"
            "compliment to both — the point is not that closed software is badly\n"
            "made, it is that 'well made' and 'checkable' are different properties\n"
            "and science needs the second one.\n\n"
            "Concrete backing if someone pushes: the report format is markdown in\n"
            "a zip, dependency versions are pinned in a lock file, and the deck on\n"
            "screen is itself a file in the repository that anyone can rebuild.\n\n"
            "A good place to name the failure mode out loud: a student graduates,\n"
            "and two years later nobody can reproduce the figure."
        ),
    ),
    # ── 9 ──────────────────────────────────────────────────────────────────────
    dict(
        seconds=40,
        text=(
            "## Direct Electron's bet on open source\n\n"
            "DE pays for this work and then gives it away. That is a position, not "
            "charity.\n\n"
            "- **SpyDE is free and GPL-3.0** — for any detector, not only ours. No "
            "paid tier, no licence server.\n"
            "- **The work goes upstream** — into HyperSpy, pyxem and RosettaSciIO, "
            "where it outlives any one application.\n"
            "- **`deapi`**, our detector control API, is MIT and on PyPI. So is "
            "**`anyplotlib`**.\n"
            "- A camera is only as useful as what you can do with the data. Locking "
            "that up helps nobody — least of all the person who bought it.\n"
        ),
        notes=(
            "Say the commitment plainly, because the audience will assume a vendor\n"
            "talk otherwise: the licence is GPL-3.0, the repository is public, and\n"
            "SpyDE opens other manufacturers' formats.\n\n"
            "The business case, if asked: detectors are the product, and every\n"
            "hour a customer spends fighting file formats is an hour the detector\n"
            "isn't earning its keep. Open tooling is the cheapest way to make the\n"
            "hardware worth more.\n\n"
            "deapi: github.com/directelectron/deapi (MIT, pip install deapi)."
        ),
    ),
    # ── 10 ─────────────────────────────────────────────────────────────────────
    dict(
        seconds=40,
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
            "- The **scientific stack stays in Python** — HyperSpy, pyxem, Dask, "
            "torch\n"
            "- The **interface is a modern web stack** — React, TypeScript, "
            "Electron\n"
            "- The boundary is a **line protocol**, so the backend tests headlessly "
            "and the frontend tests under Playwright\n"
            "- Image pixels bypass JSON entirely, as raw binary frames\n"
        ),
        notes=(
            "Why not PyQt: SpyDE started as a PySide6/pyqtgraph app and was\n"
            "migrated. The split buys a modern UI toolkit without dragging the\n"
            "scientific stack into it, and a hard, testable seam between the two.\n\n"
            "The protocol is PLOTAPP:-prefixed JSON lines on stdout, from\n"
            "anyplotlib._electron. Binary frames ride a separate PLOTBIN path — a\n"
            "base64 round-trip per frame was measurably slower.\n\n"
            "Add a sentence on the Qt migration only if the audience saw the old\n"
            "app."
        ),
    ),
    # ── 11 ─────────────────────────────────────────────────────────────────────
    dict(
        seconds=45, image="04-virtual-imaging.png", layout="text-left",
        text=(
            "## anyplotlib — plotting that stays interactive\n\n"
            "matplotlib's object-oriented API, rendered in the **browser** instead "
            "of in Python. `apl.subplots()`, `ax.imshow()`, `ax.plot()` — switching "
            "is often a one-line change.\n\n"
            "- Pan, zoom and drag **never touch the kernel**, so interaction stays "
            "at frame rate on large data\n"
            "- Widgets — crosshair, ROI, span — report positions back to Python; "
            "SpyDE binds them to navigation axes\n"
            "- Deliberately light: `anywidget`, `numpy`, `traitlets`, `colorcet`. "
            "**No matplotlib required.**\n"
        ),
        notes=(
            "The screenshot: a virtual detector — the red disk — dropped on the\n"
            "diffraction pattern, and the virtual image on the right building live\n"
            "as you drag it. That interaction is an anyplotlib widget bound to a\n"
            "HyperSpy ROI.\n\n"
            "Be fair to matplotlib: it is still the right tool for print-quality\n"
            "vector figures, and anyplotlib deliberately does not try to be. The\n"
            "trade is the opposite one — raster canvas, browser rendering, and\n"
            "interactivity that does not degrade with data size.\n\n"
            "ipympl is the honest comparison: it re-renders on the Python side\n"
            "every frame, which is exactly the round-trip this avoids."
        ),
    ),
    # ── 12 ─────────────────────────────────────────────────────────────────────
    dict(
        seconds=35,
        text=(
            "## anyplotlib is a bridge\n\n"
            "Written **for** SpyDE. Not owned by it — MIT, on PyPI, with no SpyDE "
            "anywhere inside it.\n\n"
            "| where it runs | how |\n"
            "|---|---|\n"
            "| **Jupyter Lab** | an `anywidget` — the design target |\n"
            "| **SpyDE** | the renderer mounted directly in Electron |\n"
            "| **`deapi`** | live plots while a detector is running |\n"
            "| **HyperSpy** | the goal: an interactive backend for `s.plot()` |\n"
            "| **A report, a website** | `fig.save_html()` — one self-contained "
            "file, still interactive, no kernel |\n\n"
            "The figure you explored is the figure you ship.\n"
        ),
        notes=(
            "This is the slide to linger on if the room is Jupyter-heavy. The\n"
            "argument: the same plotting layer serves the notebook, the desktop\n"
            "app, the detector's live view and the exported document, so a widget\n"
            "written once shows up in all four.\n\n"
            "Status, stated accurately: the anywidget, Electron and save_html\n"
            "paths all ship today. The HyperSpy backend is intent, not released —\n"
            "call it a goal, not a feature.\n\n"
            "The Sphinx extension is a nice aside for the docs-minded: figures in\n"
            "the gallery are live in the browser via Pyodide, with no server."
        ),
    ),
    # ── 13 ─────────────────────────────────────────────────────────────────────
    dict(
        seconds=40,
        text=(
            "## The hard part: staying live on lazy data\n\n"
            "Moving the probe has to feel instant on a dataset that does not fit in "
            "RAM.\n\n"
            "- **Storage-aligned chunking** — load with chunks that span whole "
            "signal frames, so one pattern is one chunk read. Never re-chunk a "
            "multi-gigabyte array to fix it afterwards.\n"
            "- **One serial dispatcher, latest-position-wins** — no locks and no "
            "thread per move; a superseded position is dropped before it ever runs.\n"
            "- **Two caches** — decoded frames, and decoded navigation *blocks*, "
            "because a compressed chunk is atomic: one frame costs what all of them "
            "cost.\n\n"
            "Region integration on a 64 × 64 × 256² scan: **2850 ms → ~5 ms** per "
            "drag step.\n"
        ),
        notes=(
            "The engineering slide — what separates 'a GUI over HyperSpy' from 'a\n"
            "GUI that stays live'.\n\n"
            "The honest framing, and the part people remember: almost every obvious\n"
            "fix here was tried and made it worse. Per-update threads raced the\n"
            "chunk cache. A lock held across the compute wedged the UI. A\n"
            "one-entry block cache re-decoded on every chunk crossing. What\n"
            "survived is serial, latest-wins, and two LRU caches.\n\n"
            "Numbers are from the project's own benchmarks (benchmarks.md). If\n"
            "asked what they were measured on, say the dev workstation rather\n"
            "than inventing a spec."
        ),
    ),
    # ── 14 ─────────────────────────────────────────────────────────────────────
    dict(
        seconds=40, image="03-find-vectors-result.png", layout="text-right",
        text=(
            "## GPU where it pays\n\n"
            "- **Peak finding** — classical (difference-of-Gaussians, normalised "
            "cross-correlation) *and* a neural detector, both on torch\n"
            "- **Orientation mapping** — the whole scan is fit **at once**: every "
            "pattern packed into one batched tensor, coarse-seeded by angular "
            "cross-correlation, refined with Adam. No Dask, no per-pattern loop.\n"
            "- Rewriting that coarse seed from a Python loop over templates into "
            "one batched FFT correlation: **289 s → 1.6 s**\n"
            "- CUDA *and* Apple Metal, always with a working CPU fallback\n"
        ),
        notes=(
            "The screenshot is a real run on bundled synthetic Si grains — 701\n"
            "diffraction vectors found, overlaid in red on the pattern, with the\n"
            "vector count map opened as a new window.\n\n"
            "The lesson worth saying out loud: when a GPU step is slow, the cause\n"
            "is almost always a Python loop launching tiny kernels, not the\n"
            "arithmetic. 289 s -> 1.6 s was a restructuring, not a faster card.\n\n"
            "Apple Metal has a sharp edge worth a sentence if there are Mac users\n"
            "in the room: it is not thread-safe, so every torch call site in the\n"
            "app takes one shared device lock."
        ),
    ),
    # ── 15 ─────────────────────────────────────────────────────────────────────
    dict(
        seconds=30, image="02-find-vectors-wizard.png", layout="text-left",
        text=(
            "## Interaction is the feature\n\n"
            "Heavy compute is staged behind a **live preview**: tune the parameters "
            "against the pattern under the crosshair, *then* commit to the full "
            "scan.\n\n"
            "- The preview follows the navigator, before any full-dataset compute\n"
            "- The same shape serves virtual imaging, FFT, line profiles, strain "
            "and orientation mapping\n"
            "- Results open **early** and fill in progressively\n"
        ),
        notes=(
            "Point at the red circles on the pattern — the peak finder running\n"
            "live under the crosshair with the parameters currently in the wizard.\n"
            "You never launch a twenty-minute job on a guess.\n\n"
            "Note the neural detector in the Method dropdown alongside the\n"
            "classical ones — same preview, same commit.\n\n"
            "This is the 'Wizard' shape in spyde/actions/README.md: open, tune,\n"
            "run, commit, close."
        ),
    ),
    # ── 16 ─────────────────────────────────────────────────────────────────────
    dict(
        seconds=40,
        text=(
            "## …and this talk is a SpyDE document\n\n"
            "The Report Builder turns a session into a document. Drag a figure out "
            "of a window into the report and it stays **live and re-bindable** — "
            "reopen the report with the data loaded and it re-renders from the "
            "signal.\n\n"
            "- `.spyde-report` is a plain **zip**: `report.md` + `figures/*.yaml` + "
            "`assets/*.png` — valid markdown you can unzip and hand to pandoc\n"
            "- One document, several surfaces: scrolling report, **slide deck**, "
            "movie editor\n"
            "- Themed in the document, so a deck still looks like yours on someone "
            "else's machine\n"
            "- Exports to static HTML, interactive HTML, PDF, or a markdown folder\n\n"
            "**This deck is `doc/presentations/spyde-overview.spyde-report`.**\n"
        ),
        notes=(
            "The reveal. Worth a pause: the audience is looking at the feature.\n\n"
            "Press S here to show the presenter view live — same screen, since\n"
            "SpyDE is a single Electron window: current slide, next slide, these\n"
            "notes, and a timer.\n\n"
            "The 'no JSON anywhere' choice is deliberate — the document stays\n"
            "readable and diffable in git, which is the same reproducibility\n"
            "argument from a few slides ago applied to the write-up."
        ),
    ),
    # ── 17 ── section divider ──────────────────────────────────────────────────
    dict(
        kind="title", style="accent", seconds=10,
        text="# What's next\n\n## Active development and roadmap\n",
        notes="Breath, and a change of gear from 'what it is' to 'where it goes'.",
    ),
    # ── 18 ─────────────────────────────────────────────────────────────────────
    dict(
        seconds=45,
        text=(
            "## Active development\n\n"
            "- **Spectroscopy** — EELS and EDS model fitting across a whole "
            "spectrum image: edges, backgrounds, quantification\n"
            "- **EBSD** — kikuchipy indexing, IPF maps and refinement in the same "
            "shell\n"
            "- **Atomic resolution** — column finding through atomap\n"
            "- **In situ** — drift correction, particle segmentation and tracking, "
            "and DE's sparse `.csb` event streams, re-cut at any exposure without "
            "re-reading the movie\n"
            "- **Apple silicon** — the fitting and EBSD paths run on Metal\n"
        ),
        notes=(
            "Pick the two items closest to this audience and spend the time there\n"
            "rather than reading all five.\n\n"
            "The .csb point is the one that surprises people: the file is an event\n"
            "stream, not a frame stack, so an image only exists once you choose an\n"
            "exposure — and changing that choice is cheap, because the per-frame\n"
            "totals come from the block table without reading any payload.\n\n"
            "This slide is a snapshot of the tree as of v0.3.0; refresh it before\n"
            "you give the talk."
        ),
    ),
    # ── 19 ─────────────────────────────────────────────────────────────────────
    dict(
        seconds=40,
        text=(
            "## Roadmap\n\n"
            "**quantem** — a PyTorch-native toolkit for quantitative EM: "
            "ptychographic phase retrieval, HAADF tomography, neural object "
            "representations. It makes the same bet SpyDE's GPU paths already "
            "make — put the whole problem on the device and batch it.\n\n"
            "- Bring those reconstructions in as SpyDE actions, so a phase map is "
            "another **node on the tree** rather than another script\n"
            "- **anyplotlib as an interactive backend for HyperSpy** — the same "
            "figures in the notebook and in the app\n"
            "- **Analyse during acquisition, not after it** — the detector API is "
            "already open\n"
        ),
        notes=(
            "quantem is the next-generation piece: github.com/electronmicroscopy/\n"
            "quantem, pip install quantem. Ptychography and tomography are exactly\n"
            "the workloads SpyDE has no answer for today, and its PyTorch backend\n"
            "means it already thinks in batched device tensors — the integration\n"
            "is a data-model question, not a rewrite.\n\n"
            "Be clear that these are directions, not shipped features, and say\n"
            "which one you want help with. This is the slide that turns a talk\n"
            "into a collaboration.\n\n"
            "Refresh before each delivery — a roadmap slide ages fastest."
        ),
    ),
    # ── 20 ─────────────────────────────────────────────────────────────────────
    dict(
        seconds=25,
        text=(
            "## Try it\n\n"
            f"- **Download** — macOS, Windows and Linux builds: `{REPO_URL}/releases`\n"
            "- **From source** — Node 18+ and `uv`, then `uv sync --extra tests` "
            "and `npm run dev`\n"
            f"- **Docs** — `{DOCS_URL}`\n"
            "- Python **3.10–3.13** · **GPL-3.0-or-later**\n\n"
            "Built on the work of the HyperSpy and pyxem communities — and given "
            "back to them.\n"
        ),
        notes=(
            "First launch bootstraps its own Python environment with uv, including\n"
            "the GPU-correct torch wheel, so 'download and run' really is the path.\n"
            "The macOS build is signed and notarised; Windows installers are not\n"
            "yet signed, so warn people about the SmartScreen prompt.\n\n"
            "Confirm the version you want to point people at before you present —\n"
            "REPO_URL and DOCS_URL are constants at the top of the build script."
        ),
    ),
    # ── 21 ─────────────────────────────────────────────────────────────────────
    dict(
        kind="title", style="accent", seconds=10,
        text=_closing_slide_text(),
        notes=(
            "Leave this up for questions.\n\n"
            "Acknowledgements worth naming out loud: Paul Voyles and the Voyles\n"
            "group, the HyperSpy and pyxem maintainers, and Direct Electron for\n"
            "funding the work and agreeing to give it away."
        ),
    ),
]


def build() -> tuple[ReportDoc, dict[str, bytes]]:
    """Assemble :data:`SLIDES` into a presentation ``ReportDoc`` + its assets."""
    theme = normalize_theme({**THEME, "logo": _logo_data_url()})
    doc = ReportDoc(title="SpyDE — an overview", doc_type="presentation",
                    theme=theme)
    assets: dict[str, bytes] = {}

    for s in SLIDES:
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
    print(f"  theme  {doc.theme['bg']} / accent {doc.theme['accent']} · "
          f"footer {'on' if doc.theme['footer_show'] else 'off'} · "
          f"logo {'embedded' if doc.theme['logo'] else 'MISSING'}")
    assert n_slides == len(SLIDES), (
        f"slide grouping mismatch: {n_slides} groups vs {len(SLIDES)} entries")
    # The talk has a slot. Fail the build rather than discover it on stage.
    assert 660 <= total <= 810, (
        f"budget {total} s is outside the ~11-13.5 min slot this deck targets")


if __name__ == "__main__":
    main()
