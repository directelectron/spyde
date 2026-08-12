"""
tile.py — `get_result` as anyplotlib's tile source.

The viewer does not fetch a frame and paint it. anyplotlib asks for the region
currently on screen at the resolution it needs, and that request goes straight
to the server:

    TileBackend.sample(x0, x1, y0, y1, out_w, out_h)      anyplotlib asks
        ↓
    get_result(centerX, centerY, windowWidth=out_w,
               windowHeight=out_h, zoom=out_w/(x1-x0))     server crops + scales

So only the pixels actually on screen cross the wire, which is the difference
between usable and unusable on a 4096² detector — and the same call returns the
frame statistics, so the stats strip costs nothing extra.

## What the parameters mean

Established by measurement against deapi's simulated server, not from the
docstring:

* ``windowWidth`` / ``windowHeight`` are the **output** size in pixels.
* ``zoom`` sets how much **source** that output covers:
  ``source_extent = windowWidth / zoom``. Requesting a 256-px window at
  ``zoom=0.25`` returned the whole 1024² frame decimated to 256² (correlation
  0.98 against a numpy reference); at ``zoom=1`` it returns a 256-px crop.

So for a region ``[x0, x1)`` rendered into ``out_w`` pixels,
``zoom = out_w / (x1 - x0)`` — which is just "output pixels per source pixel".

## What the simulator cannot tell us

`FakeServer` reads ``center_x`` and ``center_y`` off the wire and then never
uses them — they appear nowhere else in its source, and requesting
``centerX=200`` returns a byte-identical image to ``centerX=0``. **Panning is
therefore untested.** The centre is computed here the obvious way and needs
confirming against real hardware or an improved simulator before anyone trusts
a pan. Zoom, output size and statistics are all genuinely exercised.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

#: Frame type requested for the live/last image.
DEFAULT_FRAME_TYPE = "singleframe_integrated"


class DeapiTileBackend:
    """anyplotlib :class:`TileBackend` backed by a DE Server connection.

    Parameters
    ----------
    instrument
        The :class:`~de_groundcrew.instrument.Instrument` owning the connection.
        Every request is routed through its io thread.
    shape
        ``(H, W)`` of the full sensor image.
    frame_type
        deapi frame type to request.
    """

    def __init__(self, instrument, shape: tuple[int, int],
                 frame_type: str = DEFAULT_FRAME_TYPE,
                 pixel_format: str = "UINT16") -> None:
        self._inst = instrument
        self._shape = (int(shape[0]), int(shape[1]))
        self._frame_type = frame_type
        self._pixel_format = pixel_format
        #: Statistics from the most recent sample — the same round trip that
        #: fetched the pixels, so the stats strip needs no second call.
        self.last_stats: dict = {}

    # ── TileBackend protocol ──────────────────────────────────────────────────

    @property
    def full_shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.uint16)

    @property
    def origin(self) -> str:
        return "upper"

    def extent(self):
        """Pixel coordinates — the detector has no calibrated axes of its own
        until the TEM channel supplies a pixel size."""
        return None

    def sample(self, x0: int, x1: int, y0: int, y1: int,
               out_w: int, out_h: int, method: str = "mean") -> np.ndarray:
        """Return source region ``[y0:y1, x0:x1]`` resampled to ``(out_h, out_w)``.

        ``method`` is accepted for protocol compatibility and ignored: the
        server does the resampling, and it does not take a method. Down-sampling
        an already-downsampled tile again here would only add blur.
        """
        src_w = max(1, int(x1) - int(x0))
        out_w = max(1, int(out_w))
        out_h = max(1, int(out_h))

        fut = self._inst.call(lambda c: self._fetch(
            c, cx=(int(x0) + int(x1)) // 2, cy=(int(y0) + int(y1)) // 2,
            out_w=out_w, out_h=out_h, zoom=out_w / src_w))
        arr, stats = fut.result()
        self.last_stats = stats

        # The server is the authority on size, but anyplotlib indexes the array
        # it gets back — so a short read must be corrected here rather than
        # raising out of a paint.
        if arr.shape != (out_h, out_w):
            log.debug("tile came back %s, expected %s", arr.shape, (out_h, out_w))
            arr = _fit(arr, out_h, out_w)
        return arr

    # ── Internals (io thread) ─────────────────────────────────────────────────

    def _fetch(self, client, *, cx: int, cy: int, out_w: int, out_h: int,
               zoom: float):
        from deapi import Attributes, Histogram

        attrs = Attributes()
        attrs.centerX, attrs.centerY = int(cx), int(cy)
        attrs.windowWidth, attrs.windowHeight = int(out_w), int(out_h)
        attrs.zoom = float(zoom)

        hist = Histogram()
        hist.bins = 256

        result = client.get_result(self._frame_type, self._pixel_format, attrs, hist)
        arr = np.asarray(result[0] if isinstance(result, tuple) else result)

        # Statistics come from the HISTOGRAM, not from `attrs`. Against the
        # simulator the attribute fields carry its configuration rather than
        # any measurement — imageMin/Max/Mean/Std read 0 / 32768 / 100 / 5
        # (exactly 2^15 and two round numbers) while the pixels that arrived in
        # the same call ran 1…25 with a mean of 1.98. The histogram matched the
        # array exactly. Reporting the attributes would have put "MAX 32768"
        # under a picture whose brightest pixel was 25.
        stats = _histogram_stats(hist)
        stats["frame"] = _int(attrs.imageIndex)
        # Exposure-rate fields have no second source, so they are passed through
        # as the server gives them and are suspect on the simulator for the same
        # reason. Real hardware is expected to populate them.
        stats.update({
            "eppix": _num(attrs.eppix), "eppixps": _num(attrs.eppixps),
            "over": _num(attrs.overExposureRate),
            "under": _num(attrs.underExposureRate),
        })
        return arr, stats


def _histogram_stats(hist) -> dict:
    """Frame statistics and a robust display range, from the server histogram.

    Free: the histogram arrives on the same `get_result` as the pixels, and it
    describes the WHOLE frame — so the display range does not shift as the user
    pans across a tiled image, which is what deriving it from the visible tile
    would do.

    ``levels`` is the 2nd–98th percentile, the same robust range SpyDE uses. A
    plain min/max is unusable on a detector: one hot pixel at saturation drives
    the whole image black.
    """
    empty = {"min": None, "max": None, "mean": None, "std": None, "levels": None}
    try:
        counts = np.asarray(hist.data, dtype=np.float64)
        lo, hi = float(hist.min), float(hist.max)
    except (AttributeError, TypeError, ValueError):
        return empty
    total = counts.sum()
    if counts.size == 0 or total <= 0 or not np.isfinite([lo, hi]).all():
        return empty

    # Bin CENTRES: a count in bin i means "values in [edge_i, edge_i+1)", and
    # the centre is the least-wrong single representative.
    width = (hi - lo) / counts.size
    centres = lo + (np.arange(counts.size) + 0.5) * width

    mean = float((centres * counts).sum() / total)
    var = float((counts * (centres - mean) ** 2).sum() / total)

    cum = np.cumsum(counts)
    p_lo = float(centres[int(np.searchsorted(cum, 0.02 * total))])
    p_hi = float(centres[int(min(np.searchsorted(cum, 0.98 * total), counts.size - 1))])
    if p_hi <= p_lo:                      # a frame in one bin — a flat field
        p_hi = p_lo + max(width, 1.0)

    return {"min": lo, "max": hi, "mean": mean, "std": float(np.sqrt(max(var, 0.0))),
            "levels": (p_lo, p_hi)}


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # The server uses out-of-band sentinels for "not measured"; NaN and inf
    # would otherwise reach the UI and render as "NaN".
    return f if np.isfinite(f) else None


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _fit(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Crop or edge-pad *arr* to exactly ``(h, w)``.

    Only ever corrects an off-by-a-row disagreement with the server about
    rounding. It is not a resampler, and a large mismatch means the zoom maths
    is wrong rather than that this needs to be cleverer.
    """
    out = np.zeros((h, w), dtype=arr.dtype)
    hh, ww = min(h, arr.shape[0]), min(w, arr.shape[1])
    out[:hh, :ww] = arr[:hh, :ww]
    if hh < h and hh > 0:
        out[hh:, :ww] = out[hh - 1, :ww]
    if ww < w and ww > 0:
        out[:, ww:] = out[:, ww - 1:ww]
    return out
