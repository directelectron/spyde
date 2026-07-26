"""
Fast Windows MRC binary read — VERIFY the stock fast path is active.

WHAT
----
This module does NOT monkeypatch the MRC reader. It verifies that the installed
RosettaSciIO exposes ``rsciio.utils.file.memmap_distributed`` (the
``np.memmap`` + ``dask.map_blocks(slice_memmap)`` chunked lazy reader that the MRC
reader uses on the lazy path) and logs a one-time warning if it does not — so a
regression to a slow full-file read is visible instead of silent.

WHY (history)
-------------
SpyDE previously pinned a ``cssfrancis/rosettasciio@win32-binary-read`` fork whose
only delta was an MRC-read speed/memory optimisation. As of the current
pyproject.toml that fork is DROPPED in favour of stock ``rosettasciio>=0.14.0``,
whose ``memmap_distributed`` already reads each dask chunk by opening a fresh
``np.memmap(..., mode="r")`` and slicing only the requested region
(``rsciio.utils._distributed.slice_memmap``). So reading one navigator frame
touches only that frame's bytes (via the OS page cache), not the whole file.

Measured on this dev box (stock rsciio 0.14.0), on real DE .mrc files:

  ``scan4_035displace.mrc``  (11.25 GB, 85808 x 256 x 256 uint16, nav-dim 1):
    hs.load(lazy=True)        ~5.3 s   (graph build + hyperspy warmup)
    single frame  (cold)      ~6.8 ms
    single frame  (warm)      ~3.4 ms
    scrub 5 scattered frames  ~2.6-3.1 ms each

  ``20241214_29632_movie_movie.mrc`` (11.37 GB, 86784 x 256 x 256 uint16):
    same ~6 ms/frame ONCE the DE-metadata scan is bypassed. (A filename
    containing "movie" makes the reader eagerly load related virtual images via
    ``find_related_de_files`` — that is a *separate* slowness in the DE movie
    load path, NOT the binary read this module is about.)

Conclusion: the stock reader IS the fast path; no MRC monkeypatch is required.
This module is the documented "ensure/verify it's active" the external-patch
contract allows when upstream already has the capability.

WHEN TO REMOVE
--------------
Never strictly required (it only logs). Drop it if/when SpyDE stops caring whether
the MRC lazy read is memmap-chunked, or if a future rosettasciio renames
``memmap_distributed`` and the check becomes noise. If a slow read regresses, the
fix is a real per-chunk patch here (a ``file_reader``/``memmap_distributed``
wrapper), NOT reviving the retired fork.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_verified = False


def is_fast_mrc_read_available() -> bool:
    """True iff the installed RosettaSciIO exposes the chunked memmap reader the
    MRC lazy path relies on (``rsciio.utils.file.memmap_distributed``)."""
    try:
        from rsciio.utils import file as _rsfile
        return callable(getattr(_rsfile, "memmap_distributed", None))
    except Exception:
        return False


def apply() -> bool:
    """Verify (do not patch) that the stock fast MRC read path is present.

    Returns True if the chunked memmap reader is available, False (with a
    one-time warning) if it is missing — in which case MRC lazy reads may fall
    back to a slow monolithic path and want investigation."""
    global _verified
    if _verified:
        return True
    ok = is_fast_mrc_read_available()
    if ok:
        log.debug("spyde.external.rosettasciio.mrc: stock memmap_distributed MRC "
                  "read is active (fast per-chunk lazy read)")
        _verified = True
        return True
    log.warning("spyde.external.rosettasciio.mrc: rsciio.utils.file."
                "memmap_distributed is missing — MRC lazy reads may use a slow "
                "path. If large .mrc scrubbing is slow, add a per-chunk memmap "
                "patch here (see this module's docstring).")
    return False
