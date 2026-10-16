"""
measure.py — one row of properties per instance, in calibrated units.

Region properties come from scikit-image; this module only chooses the
columns, applies the pixel calibration exactly once, and fixes the column
names and dtypes the :class:`~spyde.signals.particles.Particles` store is
built from.
"""
from __future__ import annotations

import numpy as np

#: Measured columns and their numpy dtype, in store order. ``field`` (the
#: index of the field the instance was found in) is prepended by the store.
COLUMNS: tuple[tuple[str, str], ...] = (
    ("label", "i4"),
    ("y", "f4"), ("x", "f4"),
    ("area", "f4"),
    ("equivalent_diameter", "f4"),
    ("major_axis", "f4"), ("minor_axis", "f4"),
    ("perimeter", "f4"),
    ("eccentricity", "f4"),
    ("solidity", "f4"),
    ("intensity_mean", "f4"), ("intensity_max", "f4"), ("intensity_std", "f4"),
    ("bbox_y0", "i4"), ("bbox_x0", "i4"), ("bbox_y1", "i4"), ("bbox_x1", "i4"),
)

#: Which columns scale with length and which with area when calibrating.
LENGTH_COLUMNS = ("y", "x", "equivalent_diameter", "major_axis", "minor_axis", "perimeter")
AREA_COLUMNS = ("area",)

_PROPERTIES = ("label", "area", "centroid", "bbox", "perimeter", "eccentricity",
               "solidity", "axis_major_length", "axis_minor_length",
               "equivalent_diameter_area")
_INTENSITY_PROPERTIES = ("intensity_mean", "intensity_max", "intensity_std")


def measure_instances(labels: np.ndarray, intensity: np.ndarray | None = None, *,
                      scale: float = 1.0) -> dict[str, np.ndarray]:
    """Measure every instance in a label image.

    Parameters
    ----------
    labels
        ``(H, W)`` integer label image; 0 is background.
    intensity
        Optional field the labels came from, for the intensity columns.
        Non-finite pixels (a drift-corrected border) are excluded.
    scale
        Pixel size in the field's units: lengths are multiplied by it, areas
        by its square, dimensionless columns are untouched.

    Returns
    -------
    A dict of equal-length 1-D arrays, one per :data:`COLUMNS` entry.
    """
    from skimage.measure import regionprops_table

    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2-D; got shape {labels.shape}")
    if labels.max() <= 0:
        return empty_table()

    table = regionprops_table(labels, properties=_PROPERTIES)
    label_ids = table["label"].astype(np.int32)
    scale = float(scale)
    rows = {
        "label": label_ids,
        "y": table["centroid-0"] * scale,
        "x": table["centroid-1"] * scale,
        "area": table["area"] * scale ** 2,
        "equivalent_diameter": table["equivalent_diameter_area"] * scale,
        "major_axis": table["axis_major_length"] * scale,
        "minor_axis": table["axis_minor_length"] * scale,
        "perimeter": table["perimeter"] * scale,
        "eccentricity": table["eccentricity"],
        "solidity": table["solidity"],
        "bbox_y0": table["bbox-0"], "bbox_x0": table["bbox-1"],
        "bbox_y1": table["bbox-2"], "bbox_x1": table["bbox-3"],
        **intensity_columns(labels, intensity, label_ids),
    }
    return {name: np.ascontiguousarray(rows[name], dtype=dtype) for name, dtype in COLUMNS}


def intensity_columns(labels: np.ndarray, intensity: np.ndarray | None,
                      label_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Mean, max and standard deviation of the intensity under each label, over
    the FINITE pixels only. NaN where a label has none, or without an image."""
    n_rows = len(label_ids)
    if intensity is None:
        return {name: np.full(n_rows, np.nan) for name in _INTENSITY_PROPERTIES}
    image = np.asarray(intensity, np.float64)
    if image.shape != labels.shape:
        raise ValueError(f"intensity {image.shape} does not match labels {labels.shape}")
    finite = np.isfinite(image) & (labels > 0)
    ids, values = labels[finite].astype(np.int64), image[finite]
    n_bins = int(labels.max()) + 1
    count = np.bincount(ids, minlength=n_bins)
    total = np.bincount(ids, values, minlength=n_bins)
    squares = np.bincount(ids, values * values, minlength=n_bins)
    maximum = np.full(n_bins, -np.inf)
    np.maximum.at(maximum, ids, values)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        variance = np.maximum(squares / count - mean * mean, 0.0)
    maximum[count == 0] = np.nan
    return {"intensity_mean": mean[label_ids], "intensity_max": maximum[label_ids],
            "intensity_std": np.sqrt(variance)[label_ids]}


def empty_table() -> dict[str, np.ndarray]:
    return {name: np.zeros(0, dtype) for name, dtype in COLUMNS}
