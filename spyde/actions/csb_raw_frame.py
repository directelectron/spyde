"""One RAW camera frame under the point, instead of its integrated plane.

WHY
    A ``.csb`` loads as integrated planes — 8 raw frames each by default —
    because a single 390 µs frame of an 8192² detector carries ~314k events
    across 67M pixels (0.5% occupancy) and is mostly noise. Integrating is the
    right thing to LOOK at, and stays the default. But it also hides what the
    format is actually streaming, so the Point selector can drop to a single
    raw frame to see one.

HOW
    A raw frame is not a slice of the loaded signal — the signal is a stack of
    PLANES, and a frame is a different integration.
    ``original_metadata.csb.plane_frame_bounds`` maps each plane to its
    ``[f0, f1)`` raw range, so the first frame of the plane under the cursor is
    ``integrate_plane(path, backend, f0, f0 + 1, bin, dtype)`` — the very call
    the plane stack itself is built from. Same binning, same shape, same cache.

    It is installed by swapping the child's entry in ``selector.children``,
    which is the seam the vectors tree already uses to COMPUTE a frame rather
    than slice one; ``_run_update`` paints whatever the function returns. So
    nothing in the navigator read path changes — the point selector is simply
    pointed at a different producer while raw mode is on, and back again when
    it is off.

COST
    Bounded and known. The plane readback is ~27 ms and does not depend on the
    exposure, so one raw frame costs about what one plane costs (~31 ms vs
    ~39 ms measured on the 8192² test movie). Dropping to raw makes a single
    frame no dearer to look at; it is only reading the WHOLE movie at raw
    cadence that multiplies out, which is why this is a viewing mode and not a
    load option.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

#: `frames` value on the wire that means "one raw camera frame", chosen
#: because every real width is >= 1 plane and 0 is otherwise meaningless.
RAW = 0


def _csb_meta(signal):
    """``original_metadata.csb`` as a plain dict, or None if not a CSB signal."""
    try:
        om = signal.original_metadata
        if "csb" not in om:
            return None
        return om.csb.as_dictionary()
    except Exception as e:
        log.debug("reading CSB metadata failed: %s", e)
        return None


def raw_frames_per_plane(signal) -> int:
    """How many raw camera frames one loaded plane integrates.

    0 when this is not a CSB signal, or when a plane IS a single frame
    already — in both cases there is nothing below the plane to offer.
    """
    meta = _csb_meta(signal)
    if not meta:
        return 0
    try:
        fpp = list(meta.get("frames_per_plane") or [])
    except Exception:
        return 0
    if not fpp:
        return 0
    n = int(fpp[0])
    return n if n > 1 else 0


def raw_frame_update(selector, child, indices, get_result: bool = False):
    """Return ONE raw camera frame for the plane under *indices*.

    Returning None lets `_run_update` skip the paint and leave the last good
    frame up, which is what should happen if this is ever installed on a
    signal that cannot serve it.
    """
    try:
        signal = child.plot_state.current_signal
    except Exception as e:
        log.debug("raw frame: no current signal (%s)", e)
        return None

    meta = _csb_meta(signal)
    if not meta:
        return None

    try:
        bounds = meta["plane_frame_bounds"]
        plane = int(np.ravel(np.asarray(indices))[0])
        plane = max(0, min(plane, len(bounds) - 1))
        f0 = int(bounds[plane][0])
        # Which frame WITHIN the plane, so the caret can walk across a plane's
        # own frames later without this needing to change.
        offset = int(getattr(selector, "raw_frame_offset", 0) or 0)
        f1 = int(bounds[plane][1])
        f0 = min(f0 + max(0, offset), f1 - 1)

        from spyde.external.rsciio_csb._api import integrate_plane
        img = integrate_plane(str(meta["path"]), str(meta.get("backend", "auto")),
                              f0, f0 + 1, int(meta.get("bin", 1) or 1),
                              signal.data.dtype)
    except Exception as e:
        log.debug("raw frame read failed: %s", e)
        return None

    arr = np.asarray(img)
    return arr[0] if arr.ndim == 3 else arr


def install(selector, on: bool) -> bool:
    """Point *selector*'s children at the raw-frame producer, or back.

    Returns True when the selector ends up in raw mode. The default function
    is stashed per child rather than assumed, so a selector that already had a
    custom producer (a vectors tree's render hook) gets its own back.
    """
    inner = getattr(selector, "selector", None) or selector
    if on:
        saved = getattr(inner, "_pre_raw_children", None)
        if saved is None:
            inner._pre_raw_children = dict(inner.children)
        for chld in list(inner.children):
            inner.children[chld] = raw_frame_update
        inner.raw_frame = True
        return True

    saved = getattr(inner, "_pre_raw_children", None)
    if saved:
        for chld, fn in saved.items():
            if chld in inner.children:
                inner.children[chld] = fn
    inner._pre_raw_children = None
    inner.raw_frame = False
    return False
