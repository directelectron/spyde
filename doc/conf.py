# Configuration file for the Sphinx documentation builder.
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

_HERE = Path(__file__).parent


# -- Project information -----------------------------------------------------
project = "spyde"
copyright = "2025, Direct Electron"
author = "Direct Electron"

release = "0.0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx_design",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Tutorial screenshots ----------------------------------------------------
# doc/tutorials/*.rst is GENERATED from the guides/ walkthroughs (the single
# source the in-app guided tour also renders) by
# ``node scripts/gen_guide_docs.mjs``. Its ``.. image::`` targets live under
# ``doc/tutorials/media/<guide>/`` — but the PNGs themselves are produced by the
# Playwright run ``guide_screenshots.spec.ts``, which writes them into
# ``docs-site/public/media/`` for the docs website.
#
# Rather than commit the same PNGs twice, mirror that tree into the doc source
# at build time. A missing source dir is not an error: a step whose screenshot
# has not been captured yet simply renders without one.
def _copy_guide_media(app=None):
    src = _HERE.parent / "docs-site" / "public" / "media"
    dst = _HERE / "tutorials" / "media"
    if not src.is_dir():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.html"))


# -- Published reports -------------------------------------------------------
# A report is a whole analysis of a real dataset, exported by SpyDE as ONE
# self-contained .html with its figures baked in and its explorer running
# client-side. Unlike the tutorial media above — where *.html is deliberately
# skipped because a walkthrough embed only makes sense inside the docs-site
# React page — a report IS the page, and it needs a stable public URL of its
# own: the poster QR codes point straight at it.
#
# `html_extra_path` copies a tree into the build output VERBATIM (no Sphinx
# parsing), so staging the report files into `doc/_extra/reports/` publishes
# them at `<pages-url>/reports/<file>.html`. Staged rather than committed twice:
# the one copy lives under docs-site/public/media/reports/, which is also what
# the docs-site Reports tab serves.
def _stage_reports() -> None:
    # Called at CONF IMPORT, not from a `builder-inited` hook: Sphinx validates
    # `html_extra_path` while reading the config, so a directory created later
    # gets a "does not exist" warning and is silently never copied. Import time
    # is the only point early enough.
    src = _HERE.parent / "docs-site" / "public" / "media" / "reports"
    dst = _HERE / "_extra" / "reports"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)     # keep the extra path valid even
    if not src.is_dir():                       # with nothing to publish yet
        return
    for pat in ("*.html", "*.svg", "*.png"):
        for f in src.glob(pat):
            shutil.copy2(f, dst / f.name)


_stage_reports()
html_extra_path = ["_extra"]


def setup(app):
    app.connect("builder-inited", _copy_guide_media)
    return {"parallel_read_safe": True, "parallel_write_safe": True}

# -- Options for HTML output -------------------------------------------------
# Ensure pydata-sphinx-theme is available
if importlib.util.find_spec("pydata_sphinx_theme") is None:
    raise RuntimeError("pydata-sphinx-theme is not installed in this environment")

html_theme = "pydata_sphinx_theme"
# html_static_path = ["_static"]
html_theme_options = {
    "logo": {
        "image_light": "_static/spyde_banner_light.svg",
        "image_dark": "_static/spyde_banner_dark.svg",
    }
}

html_favicon = "_static/icon.svg"


master_doc = "index"

# -- Autodoc / Autosummary ---------------------------------------------------
autosummary_ignore_module_all = False
autosummary_imported_members = True
autodoc_typehints_format = "short"
autodoc_default_options = {"show-inheritance": True}
autosummary_generate = True

# -- Sphinx Gallery ----------------------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "examples",
    "filename_pattern": "^((?!sgskip).)*$",
    "ignore_pattern": "_sgskip.py",
    "backreferences_dir": "spyde",
    "doc_module": ("spyde",),
    "reference_url": {"spyde": None},
    # Default matplotlib scraper. (The old "spyde.qt_scrapper.qt_sg_scraper"
    # captured screenshots of the retired Qt MainWindow; that module is gone
    # now that the UI is Electron/anyplotlib.)
    #
    # TODO (docs Phase 5 follow-up): swap in anyplotlib's AnywidgetScraper
    # (`from anyplotlib.sphinx_anywidget import AnywidgetScraper`) alongside the
    # matplotlib one so example scripts that build an anyplotlib figure render as
    # LIVE interactive widgets in the gallery — the same precompute-embed model
    # the docs-site walkthrough embeds use (see spyde/tests/gen_guide_embeds.py
    # and docs-site/ InteractiveEmbed). Gated on the anyplotlib version pinned in
    # pyproject exposing sphinx_anywidget in every doc-build env; wire it as
    #   "image_scrapers": ("matplotlib", AnywidgetScraper()),
    # and add an examples/general/plot_*.py that builds an anyplotlib Plot2D.
    "image_scrapers": ("matplotlib",),
    "capture_repr": (),  # Disable text output capture
}
