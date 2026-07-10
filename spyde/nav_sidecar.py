"""Navigator sidecar cache — persist a computed navigator NEXT TO its source
file so the next open loads it instead of re-reading the whole dataset.

Computing a navigator for a large lazy dataset reads every byte of it (a 16 GB
in-situ movie's per-frame sum, a 4D-STEM scan's DP sum) — minutes of disk time
on every open. The result is tiny (nav-shaped, ~KB). So on the FIRST successful
fill we write ``<file>.spyde-nav.npz`` beside the source, fingerprinted by the
source's size + mtime + the navigator shape; on a later open a matching sidecar
short-circuits the compute entirely (``BaseSignalTree._compute_navigator``).

The fingerprint intentionally includes the navigator SHAPE: a raw .mrc folded to
a different scan grid via the nav-shape prompt produces a different-shaped
navigator and must recompute (the stale sidecar is then overwritten).

A sidecar is best-effort both ways: failure to write (read-only dir) or a stale/
corrupt file just falls back to the normal compute.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

log = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".spyde-nav.npz"
_VERSION = 1


def sidecar_path(data_path: str) -> str:
    """``<data_path>.spyde-nav.npz`` (works for .zspy/.zarr directory stores too,
    landing beside the store directory)."""
    return data_path + SIDECAR_SUFFIX


def _fingerprint(data_path: str, nav_shape: tuple) -> dict:
    st = os.stat(data_path)
    return {
        "version": _VERSION,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "nav_shape": [int(s) for s in nav_shape],
    }


def save_nav_sidecar(data_path: str, nav: np.ndarray) -> bool:
    """Write ``nav`` (the COMPLETED navigator, array order) beside ``data_path``.
    Atomic (tmp + replace) so a crash never leaves a torn sidecar. Returns True
    on success; failures are logged and swallowed (best-effort cache)."""
    try:
        if not os.path.exists(data_path):
            return False
        nav = np.asarray(nav)
        if nav.size == 0 or not np.all(np.isfinite(nav)):
            # A partial fill (NaN holes) must never be cached as authoritative.
            return False
        meta = _fingerprint(data_path, nav.shape)
        out = sidecar_path(data_path)
        # NB np.savez appends ".npz" to a name that lacks it — keep the suffix.
        tmp = out + ".tmp.npz"
        np.savez_compressed(tmp, nav=nav, meta=np.asarray(json.dumps(meta)))
        os.replace(tmp, out)
        log.info("saved navigator sidecar %s (%s, %.1f KB)",
                 os.path.basename(out), nav.shape, os.path.getsize(out) / 1e3)
        return True
    except Exception as e:
        log.debug("saving navigator sidecar for %s failed: %s", data_path, e)
        return False


def load_nav_sidecar(data_path: str, nav_shape: tuple) -> "np.ndarray | None":
    """Return the cached navigator for ``data_path`` if a sidecar exists AND its
    fingerprint (source size + mtime + navigator shape) still matches; else None.
    Never raises."""
    try:
        p = sidecar_path(data_path)
        if not (os.path.exists(p) and os.path.exists(data_path)):
            return None
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta != _fingerprint(data_path, nav_shape):
                log.debug("navigator sidecar %s is stale (fingerprint mismatch)", p)
                return None
            nav = np.asarray(z["nav"])
        if tuple(nav.shape) != tuple(int(s) for s in nav_shape):
            return None
        return nav
    except Exception as e:
        log.debug("loading navigator sidecar for %s failed: %s", data_path, e)
        return None
