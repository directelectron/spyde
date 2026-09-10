"""The live Kikuchi band overlay on the EBSD pattern.

The counterpart of the orientation template overlay: index the pattern under
the crosshair and draw the matched orientation back on top of it. A 4D-STEM
template match is a set of SPOTS, so that overlay draws circles; an EBSD match
is a set of BANDS, and a band's centre is a straight LINE on a flat detector
(:mod:`spyde.ebsd.bands`), so this one draws line segments.

Both the geometry and the reflector set come from the wizard's dictionary, so
the lines are by construction the centres of the bands the dictionary entry was
rendered with: a line beside a band means the ORIENTATION is wrong, which is
the whole point of showing it.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

BAND_COLOR = "#30ff60"     # same green as the matched-template spot overlay
ZONE_COLOR = "#fab387"     # orange, the strain overlay's accent

_NO_ZONE_AXES = np.zeros((0, 2), np.float32)


def ebsd_bands(frame, *, indexer, reflectors, detector, pc, correct, n_bands,
               show_zone_axes, linewidth) -> dict:
    """The Kikuchi bands of the best-matching orientation for one pattern.

    The pattern is corrected the way the dictionary expects, matched against
    the resident dictionary, and that orientation's bands are projected onto
    the detector. ``match`` carries the orientation and its score for the
    caret's readout."""
    from spyde.ebsd.bands import band_lines, zone_axis_points

    frame = np.asarray(frame, dtype=float)
    if correct is not None:
        frame = correct(frame)
    euler, score = indexer.best(frame)
    segments, _weights = band_lines(euler, reflectors, detector, pc,
                                    max_bands=int(n_bands))
    zone_axes = (zone_axis_points(euler, reflectors.brightest(int(n_bands)),
                                  detector, pc)
                 if show_zone_axes else _NO_ZONE_AXES)
    return {"bands": {"data": segments, "linewidths": float(linewidth)},
            "zone": zone_axes, "match": (euler, score)}


def attach_ebsd_band_overlay(signal, indexer, reflectors, tree, *,
                             detector, pc, correct=None, n_bands: int = 12,
                             show_zone_axes: bool = False,
                             linewidth: float = 1.2, color: str = BAND_COLOR,
                             on_match=None):
    """Draw the matched orientation's bands on the windows showing ``signal``,
    re-indexed at every navigator position. Returns the node.

    The match is milliseconds but not free, so the node is expensive: it runs
    off the navigator thread and the pattern display never waits for it."""
    from spyde.actions.vector_overlay import _add_overlay

    return _add_overlay(
        tree, signal, ebsd_bands, name="ebsd_bands", expensive=True,
        groups={"bands": ("lines", {"edgecolors": color}),
                "zone": ("circles", {"radius": 3.0, "edgecolors": ZONE_COLOR,
                                     "facecolors": None, "linewidths": 1.2,
                                     "alpha": 1.0})},
        static={"indexer": indexer, "reflectors": reflectors,
                "detector": (int(detector[0]), int(detector[1])),
                "pc": tuple(float(v) for v in pc), "correct": correct,
                "n_bands": int(n_bands),
                "show_zone_axes": bool(show_zone_axes),
                "linewidth": float(linewidth)},
        on_value=(None if on_match is None
                  else lambda value: _report_match(on_match, value)),
    )


def _report_match(on_match, value) -> None:
    match = value.get("match") if isinstance(value, dict) else None
    if match is None:
        return
    try:
        on_match(*match)
    except Exception as e:
        log.debug("[overlay:ebsd] the match callback failed: %s", e)


def set_ebsd_refine_params(tree, node, **params) -> None:
    """Live-update the band-overlay knobs and redraw at the current crosshair.

    The projection centre belongs here because it is the one parameter you
    cannot set from first principles: you nudge it until the drawn lines sit on
    the bands."""
    from spyde.drawing.overlays import refresh_overlays_for

    static = {}
    if params.get("n_bands") is not None:
        static["n_bands"] = max(1, int(params["n_bands"]))
    if params.get("show_zone_axes") is not None:
        static["show_zone_axes"] = bool(params["show_zone_axes"])
    if params.get("pc") is not None:
        static["pc"] = tuple(float(v) for v in params["pc"])
    if params.get("linewidth") is not None:
        static["linewidth"] = max(0.2, float(params["linewidth"]))
    if static:
        tree.replace_overlay_static(node, **static)
    else:
        refresh_overlays_for(tree)
