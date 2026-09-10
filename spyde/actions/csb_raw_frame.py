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

    That makes it a reader rather than a slice, so raw mode pins
    :class:`RawFrameReader` on the tree for each of the selector's windows and
    unpins it again when the mode goes off. Nothing in the navigator read path
    changes: the read asks the tree which reader answers for the window and the
    node it displays, and while raw mode is on this one does.

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
        frames_per_plane = list(meta.get("frames_per_plane") or [])
    except Exception:
        return 0
    if not frames_per_plane:
        return 0
    count = int(frames_per_plane[0])
    return count if count > 1 else 0


class RawFrameReader:
    """One raw camera frame of the plane under a navigation position.

    ``selector`` supplies ``raw_frame_offset``, which frame within the plane to
    read, so a caret can walk a plane's own frames without this changing.
    """

    def __init__(self, selector, signal):
        self.selector = selector
        self.signal = signal

    @property
    def frame_bytes(self) -> int:
        data = self.signal.data
        navigation_dimension = self.signal.axes_manager.navigation_dimension
        return (int(np.prod(data.shape[navigation_dimension:]))
                * data.dtype.itemsize)

    def read_frame(self, indices):
        """The raw frame at ``indices``, or None when this signal cannot serve
        one, so the navigator paints nothing and the last frame stays up."""
        meta = _csb_meta(self.signal)
        if not meta:
            return None
        try:
            bounds = meta["plane_frame_bounds"]
            plane = int(np.ravel(np.asarray(indices))[0])
            plane = max(0, min(plane, len(bounds) - 1))
            first, last = int(bounds[plane][0]), int(bounds[plane][1])
            offset = int(getattr(self.selector, "raw_frame_offset", 0) or 0)
            first = min(first + max(0, offset), last - 1)

            from spyde.external.rsciio_csb._api import integrate_plane
            image = integrate_plane(
                str(meta["path"]), str(meta.get("backend", "auto")),
                first, first + 1, int(meta.get("bin", 1) or 1),
                self.signal.data.dtype)
        except Exception as e:
            log.debug("raw frame read failed: %s", e)
            return None

        frame = np.asarray(image)
        return frame[0] if frame.ndim == 3 else frame


def install(selector, on: bool) -> bool:
    """Read one raw camera frame under *selector*'s point, or go back to the
    integrated plane. Returns True when the selector ends up in raw mode.

    The reader is pinned for the WINDOW, not the signal: two windows can show
    the same movie, and raw is a way of looking at it rather than a property of
    it. Turning raw off releases only a reader this put there, so a window
    whose frames come from somewhere else entirely keeps answering the way it
    did.
    """
    inner = getattr(selector, "selector", None) or selector
    for child in list(inner.children):
        tree = getattr(child, "signal_tree", None)
        state = getattr(child, "plot_state", None)
        signal = getattr(state, "current_signal", None) if state is not None else None
        if tree is None or signal is None:
            continue
        if on:
            tree.set_reader_override(signal, RawFrameReader(selector, signal),
                                     plot=child)
        elif isinstance(tree.reader_override_for(signal, child), RawFrameReader):
            tree.set_reader_override(signal, None, plot=child)
    inner.raw_frame = bool(on)
    return bool(on)
