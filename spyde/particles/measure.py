"""
measure.py — turn a label image into calibrated particle property rows.

ParticleSpy's measured-property set, computed with ``regionprops_table`` (one
vectorised pass, never a Python loop over regions) and converted to physical units
exactly once, here. Every downstream consumer — the table, the histogram, the
kymograph, the CSV — reads the already-calibrated numbers, so there is one place
a unit can be wrong instead of a dozen.

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

import numpy as np

from spyde.signals.particles import COL, N_COLUMNS

# regionprops names we ask for. Kept minimal: each extra property is another pass
# over every region, and the ones omitted here are cheap to derive from these.
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


def measure_frame(
    labels: np.ndarray,
    intensity: np.ndarray | None = None,
    *,
    t: int = 0,
    scale: float = 1.0,
    background_ring: int = 3,
    min_area_px: int = 0,
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

    Returns
    -------
    (rows, contours)
        ``rows`` is ``(n, N_COLUMNS)`` float32 matching
        :data:`spyde.signals.particles.COLUMNS`, with ``track_id`` set to -1.
        ``contours`` is a list of ``(k, 2)`` int16 ``(y, x)`` outlines, one per
        row and in the same order — the 1:1 correspondence
        ``SpyDEParticles.from_frames`` requires.
    """
    from skimage.measure import regionprops_table

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

    tbl = regionprops_table(lab, properties=_PROPS)
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
        _fill_intensity(rows, lab, inten, tbl, keep, background_ring)

    contours = _contours(lab, tbl)

    rows = rows[keep]
    contours = [c for c, k in zip(contours, keep) if k]
    return np.ascontiguousarray(rows), contours


def _fill_intensity(rows, lab, inten, tbl, keep, ring: int) -> None:
    """Intensity statistics over FINITE pixels only, plus a local background ring.

    Done with per-particle bbox crops rather than one pass per statistic over the
    whole frame: a crop is tiny, and it is also the only way to compute the ring
    without dilating a full-frame mask once per particle (which at hundreds of
    particles on a 4096^2 frame is the dominant cost of the whole measure step).
    """
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


def _contours(lab: np.ndarray, tbl) -> list[np.ndarray]:
    """One int16 outline per region, in ``tbl`` order.

    Traced inside each region's padded bbox crop, not on the whole frame: a
    frame-wide ``find_contours`` would return every region's outline in arbitrary
    order with no label attached, and re-associating them is both slow and
    ambiguous where particles touch.
    """
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
