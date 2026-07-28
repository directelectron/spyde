"""
ebsd_overlay.py — the live Kikuchi band overlay on the EBSD pattern.

The EBSD counterpart of ``vector_overlay.OrientationOverlay``. Both do the same
thing — index the pattern under the crosshair and draw the matched orientation
back on top of it — and both hang off the same ``_DPOverlay`` chrome, so the
navigator wiring, the seeding, the show/hide and the teardown are shared. The
difference is entirely in what gets drawn:

* a 4D-STEM template match is a set of SPOTS, so that overlay draws circles;
* an EBSD match is a set of BANDS, and a band's centre is a straight LINE on a
  flat detector (:mod:`spyde.ebsd.bands`), so this one draws line segments.

Both the geometry and the reflector set come from the wizard's dictionary, so
the lines are by construction the centres of the bands the dictionary entry was
rendered with — a line beside a band means the ORIENTATION is wrong, which is
the whole point of showing it.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from spyde.actions.vector_overlay import _DPOverlay

log = logging.getLogger(__name__)

BAND_COLOR = "#30ff60"     # same green as the matched-template spot overlay
ZONE_COLOR = "#fab387"     # orange, the strain overlay's accent


class EbsdBandOverlay(_DPOverlay):
    """Live Kikuchi bands for the best-matching orientation, on the EBSD plot.

    Per navigator move: correct the pattern the way the dictionary expects,
    match it against the resident dictionary (one mat-vec), project that
    orientation's bands onto the detector, push the segments.

    Runs on the overlay engine's worker thread (``_overlay_mode = "thread"``):
    the match is milliseconds but it is not free, and the navigator's update
    path is serialised — computing here would gate the pattern display at the
    overlay's rate (see ``live_overlay`` for why).
    """

    _overlay_mode = "thread"
    name = "ebsd_bands"

    def __init__(self, dp_plot, signal, indexer, reflectors, *,
                 detector, pc, correct=None, n_bands: int = 12,
                 show_zone_axes: bool = False, linewidth: float = 1.2,
                 color: str = BAND_COLOR, on_match=None):
        self.dp_plot = dp_plot
        self.signal = signal
        self.indexer = indexer
        self.reflectors = reflectors
        self.detector = (int(detector[0]), int(detector[1]))
        self.pc = tuple(float(v) for v in pc)
        self.correct = correct
        self.n_bands = int(n_bands)
        self.show_zone_axes = bool(show_zone_axes)
        self.linewidth = float(linewidth)
        self._color = color
        self.on_match = on_match          # callback(euler, score) for the caret
        self._mg = None
        self._mg_za = None
        self._selectors: list = []
        self._last_iyix = (0, 0)
        self._radius_px = 3.0
        # The matcher touches one resident torch tensor; a nav move (engine
        # thread) and a Refine slider (dispatch thread) can both land in here.
        self._match_lock = threading.Lock()

    # ── parameters (the Refine tab) ───────────────────────────────────────────
    def set_refine_params(self, **params) -> None:
        """Live-update the overlay knobs and redraw at the CURRENT crosshair
        position — the wizard's Refine tab."""
        if params.get("n_bands") is not None:
            self.n_bands = max(1, int(params["n_bands"]))
        if params.get("show_zone_axes") is not None:
            self.show_zone_axes = bool(params["show_zone_axes"])
        if params.get("linewidth") is not None:
            self.linewidth = max(0.2, float(params["linewidth"]))
        if params.get("pc") is not None:
            self.pc = tuple(float(v) for v in params["pc"])
        if self._engine is not None:
            self._engine.request(*self._last_iyix)
        else:                                        # not attached to a figure
            self._render_payload(self._offsets_for(*self._last_iyix))

    # ── drawing primitives ────────────────────────────────────────────────────
    def _empty(self):
        return (np.zeros((0, 2, 2), np.float32), np.zeros((0, 2), np.float32))

    def _make_markers(self, plot2d):
        """Lines for the bands, plus a circle group for the zone axes. Both are
        created up front and simply fed empty arrays when off — adding a marker
        group later, from the overlay worker thread, would race the figure."""
        self._mg_za = plot2d.add_circles(
            np.zeros((0, 2), np.float32), name=f"{self.name}_zone_axes",
            radius=3.0, edgecolors=ZONE_COLOR, facecolors=None,
            linewidths=1.2, alpha=1.0, transform="data",
        )
        return plot2d.add_lines(
            np.zeros((0, 2, 2), np.float32), name=self.name,
            edgecolors=self._color, linewidths=self.linewidth, transform="data",
        )

    def _push(self, payload) -> None:
        """Push a ``(segments, zone_axis_points)`` payload. A bare array means
        "clear" — that is what ``_DPOverlay.set_visible(False)`` sends."""
        segs, za = payload if isinstance(payload, tuple) else self._empty()
        if self._mg is not None:
            try:
                self._mg.set(segments=segs, linewidths=self.linewidth)
            except Exception as e:
                log.debug("pushing EBSD band segments failed: %s", e)
        if self._mg_za is not None:
            try:
                self._mg_za.set(offsets=za)
            except Exception as e:
                log.debug("pushing EBSD zone axes failed: %s", e)

    def remove(self):
        super().remove()
        if self._mg_za is not None:
            try:
                self._mg_za.remove()
            except Exception as e:
                log.debug("removing EBSD zone-axis markers failed: %s", e)
            self._mg_za = None

    # ── the per-position compute ──────────────────────────────────────────────
    def _frame(self, iy, ix) -> np.ndarray:
        frame = self.signal.data[iy, ix]
        if hasattr(frame, "compute"):        # lazy/dask: one small pattern only
            frame = frame.compute()
        return np.asarray(frame, dtype=float)

    def _offsets_for(self, iy, ix):
        from spyde.ebsd.bands import band_lines, zone_axis_points
        try:
            frame = self._frame(iy, ix)
            if self.correct is not None:
                frame = self.correct(frame)
            with self._match_lock:
                euler, score = self.indexer.best(frame)
        except Exception as e:
            # Don't swallow blind: a failed match must be distinguishable from
            # an orientation whose bands simply miss the detector.
            log.debug("[overlay:ebsd] indexing FAILED nav=(%s,%s): %r", iy, ix, e)
            return self._empty()

        try:
            segs, _w = band_lines(euler, self.reflectors, self.detector, self.pc,
                                  max_bands=self.n_bands)
            za = (zone_axis_points(euler, self.reflectors.brightest(self.n_bands),
                                   self.detector, self.pc)
                  if self.show_zone_axes else np.zeros((0, 2), np.float32))
        except Exception as e:
            log.debug("[overlay:ebsd] projecting bands FAILED nav=(%s,%s): %r",
                      iy, ix, e)
            return self._empty()

        log.debug("[overlay:ebsd] nav=(%s,%s) -> %d bands, ncc=%.4f",
                  iy, ix, len(segs), score)
        if self.on_match is not None:
            try:
                self.on_match(euler, score)
            except Exception as e:
                log.debug("[overlay:ebsd] on_match callback failed: %s", e)
        return segs, za


def attach_ebsd_band_overlay(dp_plot, signal, indexer, reflectors, tree, *,
                             detector, pc, correct=None, n_bands: int = 12,
                             show_zone_axes: bool = False,
                             on_match=None) -> EbsdBandOverlay:
    """Add the live band overlay to ``dp_plot``, wired to ``tree``'s navigator
    selectors. Returns the :class:`EbsdBandOverlay`."""
    return EbsdBandOverlay(
        dp_plot, signal, indexer, reflectors, detector=detector, pc=pc,
        correct=correct, n_bands=n_bands, show_zone_axes=show_zone_axes,
        on_match=on_match,
    ).attach(tree)
