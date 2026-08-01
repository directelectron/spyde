"""
measure.py — turn a label image into calibrated particle property rows.

ParticleSpy's measured-property set, converted to physical units exactly once,
here. Every downstream consumer — the table, the histogram, the kymograph, the
CSV — reads the already-calibrated numbers, so there is one place a unit can be
wrong instead of a dozen.

The property table itself no longer comes from ``regionprops_table``
--------------------------------------------------------------------
``regionprops_table`` walks a Python object per region, so its cost is per
PARTICLE — and a real 4096² in-situ growth frame has **26 566** of them. Measured
(``benchmarks.md``): 53.5 s to measure a frame against 3.0 s to segment it, of
which ``solidity`` alone was 30.5 s and the axis/eccentricity trio 12 s. Whole
frame now: **1.37 s**.

Four modules replace it, keeping skimage's definitions to the bit:

* :mod:`spyde.particles.props` — every column that is a **label-wise reduction
  over the raster** (``bincount``: area, centroid, bbox, equivalent diameter, the
  second central moments behind the axes and eccentricity, and the perimeter's
  border-crossing weights). 43.7 s → 1.1 s.
* :mod:`spyde.particles.hull` — ``solidity``, the one property that is a convex
  hull per region rather than a reduction, in a numba kernel that does every
  region in one ``prange`` with the GIL released. 30.2 s → 0.18 s, and the hull
  is reproduced in exact integer arithmetic, so ``area_convex`` matches skimage
  on **all 26 566** regions with zero differing pixels.
* :mod:`spyde.particles.intensity` — the intensity statistics as the same
  ``bincount`` the moments turned out to be, and the local background RING (a
  dilation per particle, which overlapping neighbours may each claim, so it is
  not a partition and no ``bincount`` expresses it) as a second numba kernel.
  4.92 s → 0.19 s, and all four columns are **bit-identical** on the real frame.
* :mod:`spyde.particles.contours` — marching squares and the segment assembly
  behind them, for every region in one ``prange``. 4.77 s → 0.15 s, and the
  FILLED polygon — which is what ``render_frame`` and ``mask_at`` consume — is
  identical on **26 566 of 26 566** regions. The vertices are NOT, and cannot be:
  a closed contour is a cycle and the two tracers cut it at different vertices.
  See that module for why the fill is the right gate and vertex identity is not.

``SPYDE_PARTICLE_PROPS=legacy`` (or ``measure_frame(..., fast=False)``) restores
the ``regionprops_table`` + ``find_contours`` path. It is what the parity tests
compare against, and what runs if numba cannot compile.

Two things that look like details and are not:

* **NaN in the intensity image is respected, not coerced.** A drift-corrected
  frame has a NaN-padded border (``spyde.drift.warp``). Letting ``np.nan`` reach
  a plain ``mean`` makes every particle touching the border report NaN intensity;
  coercing NaN to zero invents a dark rim that biases the measurement instead.
  Both are wrong, so intensity statistics are computed over finite pixels only,
  and a particle with no finite pixels is dropped (see :func:`measure_frame`).
* **Circularity uses ParticleSpy's convention**, ``4*pi*area / perimeter^2``,
  which is 1 for a perfect disc. It is computed in PIXELS before calibration —
  it is dimensionless, so calibrating area and perimeter separately and then
  dividing would introduce a spurious ``scale`` factor.
"""
from __future__ import annotations

import os

import numpy as np

from spyde.signals.particles import COL, N_COLUMNS

# regionprops names we ask for. Kept minimal: each extra property is another pass
# over every region, and the ones omitted here are cheap to derive from these.
# This is now the LEGACY path's property list and the parity test's reference —
# `spyde.particles.props.label_props` produces exactly these columns.
_PROPS = (
    "label",
    "centroid",
    "area",
    "equivalent_diameter_area",
    "major_axis_length",
    "minor_axis_length",
    "perimeter",
    "eccentricity",
    "solidity",
    "bbox",
)


def _fast_default() -> bool:
    """Whether the vectorised property path is on. ``SPYDE_PARTICLE_PROPS=legacy``
    turns it off, for a bug report or an A/B against skimage."""
    return os.environ.get("SPYDE_PARTICLE_PROPS", "").lower() not in (
        "legacy", "skimage", "0", "off")


def property_table(labels: np.ndarray, *, fast: bool | None = None) -> dict:
    """The per-region property table, as ``regionprops_table`` would return it.

    Split out from :func:`measure_frame` so the parity test can compare the two
    paths column by column on a real label image, which is the only reason it is
    safe to have replaced the reference implementation at all.
    """
    if fast is None:
        fast = _fast_default()
    if fast:
        from spyde.particles.props import label_props
        return label_props(labels)
    from skimage.measure import regionprops_table
    return regionprops_table(labels, properties=_PROPS)


def warm_kernels() -> None:
    """Compile every numba kernel ``measure_frame`` uses, before the first frame.

    Three of them now (hull, ring, contours), and paying any of their first-use
    compiles inside a measured frame makes that frame look like a regression.
    Idempotent; each module short-circuits once compiled.
    """
    from spyde.particles.contours import warmup as warm_contours
    from spyde.particles.hull import warmup as warm_hull
    from spyde.particles.intensity import warmup as warm_ring

    warm_hull()
    warm_ring()
    warm_contours()


def measure_frame(
    labels: np.ndarray,
    intensity: np.ndarray | None = None,
    *,
    t: int = 0,
    scale: float = 1.0,
    background_ring: int = 3,
    min_area_px: int = 0,
    fast: bool | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Measure every instance in *labels*.

    Parameters
    ----------
    labels
        ``(h, w)`` integer label image; 0 is background.
    intensity
        Optional source frame for intensity statistics. May contain NaN (a
        drift-corrected border does); NaN pixels are excluded rather than coerced.
    t
        Frame index stamped into the ``t`` column.
    scale
        Pixel size in physical units. Lengths are multiplied by it and areas by
        its square; dimensionless quantities are untouched.
    background_ring
        Width in pixels of the dilated ring outside each particle used for the
        local ``background`` measurement. 0 disables it (leaves NaN).
    min_area_px
        Drop instances smaller than this many pixels. Applied here, in pixels,
        because it is a detector-resolution question, not a physical one.
    fast
        Force the vectorised property path on/off. ``None`` reads
        ``SPYDE_PARTICLE_PROPS``. The two paths are asserted equal column by
        column in ``test_particles_props_parity.py``.

    Returns
    -------
    (rows, contours)
        ``rows`` is ``(n, N_COLUMNS)`` float32 matching
        :data:`spyde.signals.particles.COLUMNS`, with ``track_id`` set to -1.
        ``contours`` is a list of ``(k, 2)`` int16 ``(y, x)`` outlines, one per
        row and in the same order — the 1:1 correspondence
        ``SpyDEParticles.from_frames`` requires.
    """
    lab = np.asarray(labels)
    if lab.ndim != 2:
        raise ValueError(f"labels must be 2-D; got shape {lab.shape}")
    if intensity is not None:
        inten = np.asarray(intensity, dtype=np.float64)
        if inten.shape != lab.shape:
            raise ValueError(
                f"intensity shape {inten.shape} != labels shape {lab.shape}"
            )
    else:
        inten = None

    if lab.max() <= 0:
        return np.zeros((0, N_COLUMNS), np.float32), []

    tbl = property_table(lab, fast=fast)
    n = len(tbl["label"])

    area_px = tbl["area"].astype(np.float64)
    perim_px = tbl["perimeter"].astype(np.float64)

    keep = area_px >= float(min_area_px)

    # Dimensionless, so computed in pixels BEFORE calibration.
    with np.errstate(divide="ignore", invalid="ignore"):
        circularity = np.where(perim_px > 0,
                               4.0 * np.pi * area_px / perim_px ** 2, np.nan)

    rows = np.zeros((n, N_COLUMNS), dtype=np.float32)
    rows[:, COL["t"]] = float(t)
    rows[:, COL["label"]] = tbl["label"]
    rows[:, COL["y"]] = tbl["centroid-0"] * scale
    rows[:, COL["x"]] = tbl["centroid-1"] * scale
    rows[:, COL["area"]] = area_px * scale ** 2
    rows[:, COL["equiv_diameter"]] = tbl["equivalent_diameter_area"] * scale
    rows[:, COL["major_axis"]] = tbl["major_axis_length"] * scale
    rows[:, COL["minor_axis"]] = tbl["minor_axis_length"] * scale
    rows[:, COL["perimeter"]] = perim_px * scale
    rows[:, COL["circularity"]] = circularity
    rows[:, COL["eccentricity"]] = tbl["eccentricity"]
    rows[:, COL["solidity"]] = tbl["solidity"]
    rows[:, COL["bbox_y0"]] = tbl["bbox-0"]
    rows[:, COL["bbox_x0"]] = tbl["bbox-1"]
    rows[:, COL["bbox_y1"]] = tbl["bbox-2"]
    rows[:, COL["bbox_x1"]] = tbl["bbox-3"]
    rows[:, COL["track_id"]] = -1.0
    rows[:, COL["intensity_mean"]] = np.nan
    rows[:, COL["intensity_max"]] = np.nan
    rows[:, COL["intensity_std"]] = np.nan
    rows[:, COL["background"]] = np.nan

    if inten is not None:
        _fill_intensity(rows, lab, inten, tbl, keep, background_ring, fast=fast)

    contours = _contours(lab, tbl, fast=fast)

    rows = rows[keep]
    contours = [c for c, k in zip(contours, keep) if k]
    # Score LAST: it is derived from the intensity columns filled above, so it
    # costs no extra pass over the frame. That is what lets the caret filter on
    # it without re-segmenting — see `particle_scores`.
    rows[:, COL["score"]] = particle_scores(rows)
    return np.ascontiguousarray(rows), contours


def _table_bboxes(tbl) -> np.ndarray:
    """``(n, 4)`` int64 ``(y0, x0, y1, x1)`` from the property table's columns."""
    return np.stack([np.asarray(tbl[f"bbox-{k}"], np.int64) for k in range(4)],
                    axis=1)


def _fill_intensity(rows, lab, inten, tbl, keep, ring: int, *,
                    fast: bool | None = None) -> None:
    """Intensity statistics over FINITE pixels only, plus a local background ring.

    ``intensity_mean/max/std`` are label-wise reductions over the foreground and
    go through :func:`spyde.particles.intensity.label_intensity_stats`; the
    background ring is a per-particle dilation that overlapping neighbours may
    each claim, so it goes through that module's numba kernel instead. The
    per-region loop below is the definition both are checked against, and the
    path that runs when numba is unavailable.
    """
    if fast is None:
        fast = _fast_default()
    if fast and _fill_intensity_fast(rows, lab, inten, tbl, keep, ring):
        return

    from scipy.ndimage import binary_dilation

    h, w = lab.shape
    for i in range(len(tbl["label"])):
        if not keep[i]:
            continue
        lbl = int(tbl["label"][i])
        y0, x0 = int(tbl["bbox-0"][i]), int(tbl["bbox-1"][i])
        y1, x1 = int(tbl["bbox-2"][i]), int(tbl["bbox-3"][i])

        # Pad by the ring width so the dilated boundary fits inside the crop.
        py0, px0 = max(0, y0 - ring - 1), max(0, x0 - ring - 1)
        py1, px1 = min(h, y1 + ring + 1), min(w, x1 + ring + 1)
        sub_lab = lab[py0:py1, px0:px1]
        sub_int = inten[py0:py1, px0:px1]
        m = sub_lab == lbl

        vals = sub_int[m]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            rows[i, COL["intensity_mean"]] = vals.mean()
            rows[i, COL["intensity_max"]] = vals.max()
            # Normalised by the max, matching ParticleSpy, so it is comparable
            # between particles of very different absolute brightness.
            mx = vals.max()
            rows[i, COL["intensity_std"]] = (vals.std() / mx) if mx else np.nan

        if ring > 0:
            grown = binary_dilation(m, iterations=int(ring))
            # The ring is what the dilation added, minus anything belonging to a
            # NEIGHBOURING particle — otherwise a touching particle's body is
            # measured as this one's background, which is exactly backwards.
            ring_mask = grown & ~m & (sub_lab == 0)
            bvals = sub_int[ring_mask]
            bvals = bvals[np.isfinite(bvals)]
            if bvals.size:
                rows[i, COL["background"]] = bvals.mean()


def _fill_intensity_fast(rows, lab, inten, tbl, keep, ring: int) -> bool:
    """The vectorised half of :func:`_fill_intensity`. False if numba is missing.

    Returns False WITHOUT writing anything when the ring kernel is unavailable,
    so the caller can run the per-region loop instead — a half-filled row would
    be worse than a slow one.
    """
    from spyde.particles.intensity import label_intensity_stats, ring_backgrounds

    labels = np.asarray(tbl["label"], np.int64)
    if labels.size == 0:
        return True
    bg = ring_backgrounds(lab, inten, labels, _table_bboxes(tbl), int(ring))
    if bg is None:
        return False

    mean, mx, std = label_intensity_stats(lab, inten, labels)
    keep = np.asarray(keep, bool)
    rows[keep, COL["intensity_mean"]] = mean[keep]
    rows[keep, COL["intensity_max"]] = mx[keep]
    rows[keep, COL["intensity_std"]] = std[keep]
    if int(ring) > 0:
        rows[keep, COL["background"]] = bg[keep]
    return True


def _contours(lab: np.ndarray, tbl, *, fast: bool | None = None
              ) -> list[np.ndarray]:
    """One int16 outline per region, in ``tbl`` order.

    Traced inside each region's padded bbox crop, not on the whole frame: a
    frame-wide ``find_contours`` would return every region's outline in arbitrary
    order with no label attached, and re-associating them is both slow and
    ambiguous where particles touch.

    :mod:`spyde.particles.contours` does exactly that, for every region in one
    ``prange``, and produces the same FILLED polygon (the thing ``render_frame``
    and ``mask_at`` consume) per region. The loop below is the definition it is
    checked against, and the path that runs without numba.
    """
    if fast is None:
        fast = _fast_default()
    if fast:
        from spyde.particles.contours import label_contours

        got = label_contours(lab, np.asarray(tbl["label"], np.int64),
                             _table_bboxes(tbl),
                             np.asarray(tbl["area"], np.int64))
        if got is not None:
            return got

    from skimage.measure import find_contours

    h, w = lab.shape
    out: list[np.ndarray] = []
    for i in range(len(tbl["label"])):
        lbl = int(tbl["label"][i])
        y0, x0 = int(tbl["bbox-0"][i]), int(tbl["bbox-1"][i])
        y1, x1 = int(tbl["bbox-2"][i]), int(tbl["bbox-3"][i])
        py0, px0 = max(0, y0 - 1), max(0, x0 - 1)
        py1, px1 = min(h, y1 + 1), min(w, x1 + 1)
        sub = (lab[py0:py1, px0:px1] == lbl).astype(np.float32)
        cs = find_contours(sub, 0.5)
        if not cs:
            # Degenerate (single pixel, or a region the tracer cannot close):
            # fall back to the bbox corners so every row still has an outline and
            # the 1:1 correspondence holds.
            out.append(np.array(
                [[y0, x0], [y0, x1 - 1], [y1 - 1, x1 - 1], [y1 - 1, x0]],
                dtype=np.int16))
            continue
        c = max(cs, key=len)                      # outer boundary
        c = np.rint(c).astype(np.int32)
        c[:, 0] += py0
        c[:, 1] += px0
        np.clip(c[:, 0], 0, h - 1, out=c[:, 0])
        np.clip(c[:, 1], 0, w - 1, out=c[:, 1])
        out.append(c.astype(np.int16))
    return out


def particle_scores(rows: np.ndarray) -> np.ndarray:
    """Per-particle confidence in [0, 1] — "is this a particle, or texture?"

    Derived from columns already measured, which is the whole point: scoring
    costs no extra pass over the frame, so the caret can filter on it without
    re-segmenting and a slider drag is instant.

    The statistic is CONTRAST-TO-NOISE against the particle's own dilated
    background ring::

        |intensity_mean - background| / spread

    with *spread* the particle's own intensity spread. That is the quantity
    that separates the two populations in the failure this exists for: a real
    particle sits well away from its immediate surroundings, while a fragment
    of textured support film is, by construction, the same brightness as the
    texture around it however large or round it happens to be. Size, circularity
    and solidity all fail here — over-split speckle is often small AND round.

    Squashed to [0, 1] with ``cnr / (cnr + 1)`` so the caret's slider is a plain
    0-100% control with no dataset-dependent range to explain: 0.5 means "as far
    from its background as its own noise", which is a weak particle, and real
    ones land well above it.

    NaN (no background ring measured — ``background_ring=0``, or a particle
    whose ring fell entirely outside the frame) scores 1.0 rather than 0.0.
    An unmeasurable particle must not be silently filtered away; the slider is
    a "hide the marginal ones" control, and something with no evidence against
    it is not marginal.
    """
    if rows.size == 0:
        return np.zeros((0,), np.float32)
    mean = rows[:, COL["intensity_mean"]].astype(np.float64)
    bg = rows[:, COL["background"]].astype(np.float64)
    std = rows[:, COL["intensity_std"]].astype(np.float64)

    contrast = np.abs(mean - bg)
    # `intensity_std` is already normalised by the particle's max (see
    # _fill_intensity), so put it back on the intensity scale before using it
    # as a noise estimate. Guard the degenerate flat particle.
    noise = np.where(np.isfinite(std) & (std > 0), std * np.abs(mean), np.nan)
    floor = np.nanmedian(noise) if np.isfinite(noise).any() else 1.0
    if not np.isfinite(floor) or floor <= 0:
        floor = 1.0
    noise = np.where(np.isfinite(noise) & (noise > 0), noise, floor)

    with np.errstate(invalid="ignore", divide="ignore"):
        cnr = contrast / noise
    score = cnr / (cnr + 1.0)
    # Unmeasurable => not marginal => keep. See the docstring.
    score = np.where(np.isfinite(score), score, 1.0)
    return np.clip(score, 0.0, 1.0).astype(np.float32)
