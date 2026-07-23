"""
Per-page lazy TIFF chunking — the DEFINED HOME for the multi-page .tif read fix.

WHAT (the deficiency this module owns)
--------------------------------------
RosettaSciIO's lazy TIFF reader wraps a whole multi-page stack in ONE
``dask.delayed(handle.asarray)`` — a single monolithic chunk spanning every page —
and REJECTS a ``chunks=`` argument (raises ``TypeError``). So SpyDE's
``Session._signal_spanning_chunks`` reload silently no-ops for ``.tif``, and
reading ONE frame decodes the ENTIRE stack. On a real in-situ
``.frames.mrc.tif`` (20 x 8192^2 LZW pages) every navigator move took ~4.8 s
(decode all 20 pages) instead of ~0.3 s (the one page shown).

The fix (``_maybe_page_chunk_tiff``): rebuild the lazy array with one dask task
per PAGE-BLOCK (``tifffile`` page read) so a frame read decodes only the page(s)
shown. Block size follows the same frame-byte target as
``_signal_spanning_chunks`` (a big movie frame -> 1 page/chunk; small frames pack
up to ``_NAV_CHUNK_MAX``). Pixel-identical to the reader (same
``tf.pages[i].asarray``). Gated to ``.tif``/``.tiff``, 1-D-nav 2-D-signal,
monolithic-single-chunk only.

STATUS IN THIS BRANCH
---------------------
The implementation currently lives ONLY on the unmerged branch
``fix/tiff-perpage-lazy`` (commit ``fbe565e``, as ``_maybe_page_chunk_tiff`` in
``spyde/backend/_session_files.py`` + ``test_movie_chunking.py``
``TestPerPageTiffChunking``). It is NOT present on this branch, so there is
nothing to move here yet and :func:`apply` is a **documented no-op**.

When ``fix/tiff-perpage-lazy`` lands (or is rebased onto this line), MOVE
``_maybe_page_chunk_tiff`` out of ``_session_files.py`` into this module: keep it
as a plain callable the loader invokes on a freshly-loaded ``.tif`` signal (it is
a per-signal transform, not a global monkeypatch, so unlike the hyperspy/MRC
patches it is applied at LOAD time on one signal — not via :func:`apply` at
startup). ``apply`` should then verify the rsciio TIFF reader still returns the
monolithic single chunk it targets, and warn if a rsciio bump fixed it upstream
(at which point this whole module can be deleted).

WHY it is not a startup monkeypatch: unlike ``CachedDaskArray.client`` (a class
attribute), this transforms the dask graph of one just-loaded signal. It belongs
in the load path, called per-file — this module is its documented home and the
place its rsciio-reader assumptions are recorded, not a global ``apply()`` target.

WHEN TO REMOVE
--------------
When RosettaSciIO's lazy TIFF reader honours ``chunks=`` (or itself splits pages
into separate dask tasks). Track upstream rosettasciio TIFF reader; no SpyDE
issue link yet.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def is_monolithic_tiff_reader() -> bool:
    """Best-effort probe of whether the installed rsciio TIFF reader still wraps a
    multi-page stack in ONE dask task (the deficiency this module targets).

    Currently returns True unconditionally as a conservative default — the real
    per-page splitter (``_maybe_page_chunk_tiff``) does its own per-signal
    single-chunk gate at load time, so this probe only informs :func:`apply`'s
    log. Wire a concrete rsciio-version/reader check here alongside the moved
    implementation."""
    return True


def apply() -> bool:
    """No-op in this branch (see module docstring): the per-page TIFF splitter
    lives on ``fix/tiff-perpage-lazy`` and is applied per-file at load time, not
    as a startup monkeypatch. Registered so the deficiency has a discoverable
    home; returns True (nothing to apply)."""
    log.debug("spyde.external.rosettasciio.tiff: per-page TIFF chunking is a "
              "load-time per-signal transform (see docstring); startup apply() "
              "is a no-op in this branch")
    return True
