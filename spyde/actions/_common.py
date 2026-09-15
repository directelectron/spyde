"""Small shared helpers used across several action modules.

Centralizes a few snippets that were copy-pasted in 2+ action modules
(reciprocal-radius from signal calibration, the strain component/title
constants, and the rectangular-ROI -> image-region slice). Keep this module
dependency-light (numpy + plain helpers only) so any action can import it
without pulling heavy deps.
"""
from __future__ import annotations

import numpy as np

# ── Strain component constants (shared by strain_action + strain_display) ──────
# Canonical order of strain-tensor components and their display titles.
STRAIN_COMPONENTS: tuple[str, ...] = ("exx", "eyy", "exy", "omega")
STRAIN_TITLES: dict[str, str] = {
    "exx": "εxx", "eyy": "εyy", "exy": "εxy", "omega": "ω",
}
# What a strain map SHOWS. The fit is fractional strain and radians
# (``StrainField``); the display multiplies by these so the colorbar, the
# histogram handles and a committed tree all read in percent strain and
# degrees of rotation — the numbers people quote.
STRAIN_DISPLAY_SCALE: dict[str, float] = {
    "exx": 100.0, "eyy": 100.0, "exy": 100.0, "omega": 180.0 / np.pi,
}
STRAIN_UNITS: dict[str, str] = {"exx": "%", "eyy": "%", "exy": "%", "omega": "°"}


def strain_quantity(component: str) -> str:
    """What a displayed strain component's numbers are — ``"εxx (%)"``,
    ``"ω (°)"`` — the colorbar label and the committed ``Signal.quantity``."""
    return f"{STRAIN_TITLES[component]} ({STRAIN_UNITS[component]})"


def symmetric_range(previous, vmin, vmax) -> tuple[float, float]:
    """Keep a signed map's display range centred on zero.

    The dock's two handles arrive as an ``(vmin, vmax)`` pair; for a quantity
    whose zero means something the two are one number. Whichever handle the
    user moved sets the magnitude and the other follows it, so dragging the
    lower handle inward narrows the range instead of snapping back. With no
    *previous* range (or both ends changed at once) the larger magnitude wins.
    """
    vmin, vmax = float(vmin), float(vmax)
    if previous is not None:
        moved_min = abs(vmin - float(previous[0])) > 1e-12
        moved_max = abs(vmax - float(previous[1])) > 1e-12
        if moved_min and not moved_max:
            magnitude = abs(vmin)
        elif moved_max and not moved_min:
            magnitude = abs(vmax)
        else:
            magnitude = max(abs(vmin), abs(vmax))
    else:
        magnitude = max(abs(vmin), abs(vmax))
    magnitude = magnitude or 1e-9
    return (-magnitude, magnitude)


def robust_map_limits(array: np.ndarray, *, symmetric: bool = False
                      ) -> tuple[float, float]:
    """Display limits for a per-position RESULT map, ignoring failed positions.

    Percentiles rather than the extremes: a handful of failed fits or edge
    positions otherwise stretch the scale until the real structure is one flat
    mid-tone. Non-finite values are dropped, so a map still filling in during a
    progressive pass neither blanks nor rescales.

    ``symmetric`` is for a quantity whose ZERO means something — a strain
    component, a field component, divergence, curl. Those want ``(-v, v)`` so
    the midpoint of a diverging colormap lands on zero; an unsigned magnitude
    wants the plain 2nd-to-98th spread.

    Distinct from ``de_shell.plotting.figure.robust_levels``, which is for a
    detector FRAME on the paint path: that one subsamples for speed and has no
    notion of a signed quantity, because a frame's zero is just its floor.
    """
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return (-1.0, 1.0)
    if symmetric:
        magnitude = float(np.percentile(np.abs(finite), 98)) or 1.0
        return (-magnitude, magnitude)
    low = float(np.percentile(finite, 2))
    high = float(np.percentile(finite, 98))
    return (low, high) if high > low else (low, low + 1.0)


def reciprocal_radius(signal) -> float:
    """Max reciprocal radius from the signal-axis calibration, in Å⁻¹.

    The smallest half-extent across the signal axes — the largest radius that
    still fits inside the detector in every signal dimension, and so the outer
    radius a simulated template library should be generated to.

    Always Å⁻¹, whatever the axes are labelled: diffsims reads this number as
    Å⁻¹ and does not check, so a scan calibrated in nm⁻¹ that returned its own
    axis units would build a library ten times too large — thousands of
    reflections that cannot reach the detector — and still produce a map.
    Raises when the detector has no reciprocal calibration at all, because
    there is no defensible number to return.
    """
    from spyde.reciprocal_units import reciprocal_extent

    extent = reciprocal_extent(signal)
    if extent is None:
        raise ValueError(
            "This dataset's detector axes carry no reciprocal calibration, so "
            "a template library cannot be sized. Set the detector scale in the "
            "Plot Control dock (or a beam energy, for axes in mrad).")
    return extent


def widget_region(selector, img: np.ndarray) -> np.ndarray:
    """Slice the rectangular-ROI region of ``img`` selected by ``selector``.

    Reads the 2-D widget bounds (x, y, w, h in image pixels — see the
    anyplotlib-widget-pixel-coords convention), clamps to the image, and returns
    the sub-array. Falls back to the full image when there is no usable ROI.
    """
    widget = getattr(selector, "roi", None)
    if widget is not None and hasattr(widget, "_data") and "w" in widget._data:
        x0 = int(round(float(widget.x)))
        y0 = int(round(float(widget.y)))
        x1 = int(round(float(widget.x) + float(widget.w)))
        y1 = int(round(float(widget.y) + float(widget.h)))
        x0, x1 = sorted((max(0, x0), min(img.shape[1], x1)))
        y0, y1 = sorted((max(0, y0), min(img.shape[0], y1)))
        region = img[y0:y1, x0:x1]
    else:
        region = img
    return region
