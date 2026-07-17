"""
slicing.py — fresh-slice frame reads for report callout insets.

A fresh-slice callout stores WHERE it was sliced (``nav_indices`` — hyperspy
x-first index order — or ``time_index``); this module turns that stored
position back into a 2-D frame by slicing the LIVE signal.

MEMORY-SAFETY (CLAUDE.md rule): the navigation slice happens FIRST
(``sig.inav[...]``) and ``.compute()`` runs on THE SLICE ONLY — never on the
full dataset. Runs synchronously on the asyncio main thread from the
``repfig_*`` handlers; it deliberately never touches the navigator's
``_NavDispatcher`` / ``CachedDaskArray`` machinery (Live-Display §3 is the MDI
hot path — the report slices the plain hyperspy signal instead).
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def read_frame_at(plot, nav_indices) -> "np.ndarray | None":
    """The 2-D signal frame of *plot*'s current signal at *nav_indices*.

    ``nav_indices`` is in hyperspy ``axes_manager`` order (x-first) — the SAME
    order ``axes_manager.indices`` reports and ``sig.inav[...]`` consumes — so
    a 1-D nav (movie) is ``[t]`` and a 2-D nav is ``[ix, iy]``. Returns a
    detached 2-D ndarray, or None on ANY failure (rank mismatch, index out of
    range, unreadable signal) — callers treat None as "skip"; the cause is
    logged at debug.
    """
    try:
        sig = plot.plot_state.current_signal
        am = sig.axes_manager
        nav_dim = int(am.navigation_dimension)
        if nav_dim == 0 or len(nav_indices) != nav_dim:
            log.debug("read_frame_at: rank mismatch (indices %s vs nav_dim %s)",
                      nav_indices, nav_dim)
            return None
        idx = tuple(int(i) for i in nav_indices)
        # Slice FIRST; compute ONLY the slice (never the full dataset).
        frame = sig.inav[idx]
        data = frame.data
        if hasattr(data, "compute"):
            data = data.compute()
        arr = np.asarray(data)
        # Squeeze 1-length leading dims (a slice can keep a unit nav axis).
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            log.debug("read_frame_at: sliced frame is %s-D, not 2-D", arr.ndim)
            return None
        return np.array(arr, copy=True)   # detach from the signal's buffer
    except Exception as e:
        log.debug("read_frame_at failed at %s: %s", nav_indices, e)
        return None
