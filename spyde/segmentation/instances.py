"""
instances.py — cut a foreground probability map into labelled instances.

Two routes, chosen by whether the classifier learnt a boundary class:

* **Boundary painted** — the cores ``foreground & ~boundary`` are labelled by
  connected components and then grown back out over the boundary ring with a
  flat-elevation watershed, so every foreground pixel is claimed and touching
  particles are separated where the classifier said they touch. No distance
  transform runs. This is the faster and, on touching particles, the more
  accurate route: the user told the model where the seams are.
* **No boundary** — the classical recipe: a distance transform, smoothed
  local maxima as markers, and a watershed of the negative distance.

Both end the same way: instances below ``min_size`` pixels are dropped and the
labels are renumbered 1..n.
"""
from __future__ import annotations

import numpy as np


def split_instances(foreground: np.ndarray, boundary: np.ndarray | None = None, *,
                    min_size: int = 20, split_touching: bool = True,
                    min_separation: int = 3, marker_smooth: float = 1.0,
                    clear_border: bool = False) -> np.ndarray:
    """Label the instances in a foreground map.

    Parameters
    ----------
    foreground
        ``(H, W)`` boolean mask or a probability in ``[0, 1]`` (cut at 0.5).
    boundary
        Optional ``(H, W)`` mask or probability of the seam between touching
        particles. An all-zero boundary counts as absent.
    min_size
        Instances smaller than this many pixels are dropped.
    split_touching
        Run the watershed on the no-boundary route. Off, touching particles
        stay one instance.
    min_separation
        Minimum distance between watershed markers, in pixels.
    marker_smooth
        Gaussian sigma on the distance transform before finding markers.
    clear_border
        Drop instances touching the edge of the field.

    Returns
    -------
    ``(H, W)`` int32 label image, 0 for background, instances numbered 1..n.
    """
    from scipy import ndimage
    from skimage.segmentation import clear_border as _clear_border, relabel_sequential

    mask = np.asarray(foreground)
    if mask.ndim != 2:
        raise ValueError(f"foreground must be 2-D; got shape {mask.shape}")
    if mask.dtype != bool:
        mask = mask > 0.5
    seam = _as_mask(boundary, mask.shape)

    if not mask.any():
        return np.zeros(mask.shape, np.int32)
    if seam is not None:
        labels = _split_by_boundary(mask, seam)
    elif split_touching:
        labels = _split_by_watershed(mask, min_separation, marker_smooth)
    else:
        labels, _count = ndimage.label(mask)

    labels = np.asarray(labels, np.int32)
    if clear_border:
        labels = _clear_border(labels)
    labels = drop_small(labels, min_size)
    return relabel_sequential(labels)[0].astype(np.int32)


def _as_mask(boundary, shape: tuple[int, int]) -> np.ndarray | None:
    if boundary is None:
        return None
    seam = np.asarray(boundary)
    if seam.shape != tuple(shape):
        raise ValueError(f"boundary {seam.shape} does not match foreground {shape}")
    if seam.dtype != bool:
        seam = seam > 0.5
    return seam if seam.any() else None


def _split_by_boundary(mask: np.ndarray, seam: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    from skimage.segmentation import watershed
    cores, count = ndimage.label(mask & ~seam)
    if count == 0:
        return np.zeros(mask.shape, np.int32)
    # A flat elevation floods each core outward at the same rate, so a seam
    # pixel goes to the nearest core: the ring is given back without a
    # distance transform.
    return watershed(np.zeros(mask.shape, np.uint8), cores, mask=mask)


def _split_by_watershed(mask: np.ndarray, min_separation: int,
                        marker_smooth: float) -> np.ndarray:
    from scipy import ndimage
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    distance = ndimage.distance_transform_edt(mask)
    smoothed = ndimage.gaussian_filter(distance, marker_smooth) if marker_smooth > 0 else distance
    components, _count = ndimage.label(mask)
    peaks = peak_local_max(smoothed, min_distance=max(1, int(min_separation)),
                           labels=components, exclude_border=False)
    if not len(peaks):
        return components
    markers = np.zeros(mask.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return watershed(-distance, markers, mask=mask)


def drop_small(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Zero every instance with fewer than *min_size* pixels."""
    if min_size <= 1 or labels.max() <= 0:
        return labels
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_size)
    keep[0] = False
    return np.where(keep[labels], labels, 0).astype(np.int32)
