"""
Register the CSB reader with RosettaSciIO at runtime.

WHAT
    Appends a plugin spec for Direct Electron ``.csb`` centroid-streaming files
    to ``rsciio.IO_PLUGINS``, pointing at :mod:`spyde.external.rsciio_csb`, so
    ``hs.load("movie.csb", lazy=True)`` resolves to our ``file_reader`` through
    the ordinary extension dispatch. Nothing in SpyDE's load path changes.

WHY
    This is a stretch of what ``spyde.external`` is for — the rest of this
    package PATCHES upstream, and this ADDS a format. It is here because it is
    the only place SpyDE can teach an installed rosettasciio about a format it
    does not ship, and the alternative (special-casing ``.csb`` in
    ``_session_files``) would put format dispatch somewhere it does not belong.

    The reader itself is deliberately NOT written as SpyDE code: it lives in
    :mod:`spyde.external.rsciio_csb`, laid out exactly as an rsciio plugin
    (``__init__`` re-exporting ``file_reader``, ``_api.py``,
    ``specifications.yaml``) and importing nothing from SpyDE.

WHEN TO REMOVE
    When the plugin lands in RosettaSciIO (or the cssfrancis fork) as
    ``rsciio/csb/``. That is a directory copy of ``rsciio_csb`` — rsciio
    auto-discovers ``specifications.yaml``, so registration becomes unnecessary
    and this module and its entry in ``__init__`` are simply deleted.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_applied = False

#: Mirrors rsciio_csb/specifications.yaml. Duplicated rather than read from the
#: yaml because rsciio builds IO_PLUGINS from those files at import time and we
#: are appending after the fact; the yaml stays authoritative for the day this
#: moves upstream. `test_csb_reader.py` asserts the two agree.
_SPEC = {
    "name": "CSB",
    "name_aliases": ["DE_CSB"],
    "description": ("Direct Electron CSB (compressed sparse block) "
                    "centroid-streaming files: a sparse event stream rather "
                    "than a dense frame stack. Reading integrates events over "
                    "a time window per plane."),
    "full_support": False,
    "file_extensions": ["csb", "CSB"],
    "default_extension": 0,
    "writes": False,
    "non_uniform_axis": False,
    "api": "spyde.external.rsciio_csb",
}


def apply() -> bool:
    """Append the CSB spec to ``rsciio.IO_PLUGINS`` (idempotent)."""
    global _applied
    if _applied:
        return True
    try:
        import rsciio
    except Exception as e:                                  # pragma: no cover
        log.warning("spyde.external.rosettasciio.csb_format: rsciio import "
                    "failed (%s) — .csb files will not open", e)
        return False

    plugins = getattr(rsciio, "IO_PLUGINS", None)
    if not isinstance(plugins, list):
        # Upstream changed shape. Never raise from a startup patch.
        log.warning("spyde.external.rosettasciio.csb_format: rsciio.IO_PLUGINS "
                    "is %s, not a list — cannot register the CSB reader; .csb "
                    "files will not open", type(plugins).__name__)
        return False

    if any(isinstance(p, dict) and p.get("name") == "CSB" for p in plugins):
        _applied = True
        return True

    plugins.append(dict(_SPEC))
    _applied = True
    log.debug("spyde.external.rosettasciio.csb_format: registered the CSB "
              "reader for %s", _SPEC["file_extensions"])
    return True
