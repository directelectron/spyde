# Configuration file for the Sphinx documentation builder.
from __future__ import annotations

import importlib.util


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
