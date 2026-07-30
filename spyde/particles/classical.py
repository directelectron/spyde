"""
classical.py — the always-available segmentation engine, and the SHARED
instance-split every engine uses.

Two halves, and the split between them is the important part:

* :func:`threshold_mask` / :func:`segment_frame` — the classical engine, a port of
  ParticleSpy's ``segptcls.process``. Parameter *names* are kept identical so the
  caret is recognisable to anyone arriving from ParticleSpy.
* :func:`split_instances` — takes a foreground **probability or mask** and splits
  it into individual particles. This is used by all three engines (classical,
  scribble, prompt), which is why it lives here as its own function rather than
  inside the classical pipeline. It is the only place in the package that imports
  skimage's segmentation machinery.

Sensitivity is one axis, deliberately
--------------------------------------
Plan §0.9 makes detection sensitivity the priority over instance splitting, and
notes that sensitivity and separation trade off against each other — a threshold
loose enough to catch a faint particle also merges neighbours. So
:attr:`SegmentParams.sensitivity` is a single 0..1 control that biases the
threshold, and the splitting parameters are secondary. Exposing "threshold offset"
and "split aggressiveness" as independent knobs invites the user to chase one with
the other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Thresholding methods, matching ParticleSpy's vocabulary.
THRESHOLD_METHODS: tuple[str, ...] = (
    "otsu", "mean", "minimum", "yen", "isodata", "li",
    "local", "local_otsu", "niblack", "sauvola",
)

_LOCAL_METHODS = frozenset({"local", "local_otsu", "niblack", "sauvola"})


@dataclass
class SegmentParams:
    """Classical-pipeline parameters. ParticleSpy names where they correspond."""

    threshold: str = "otsu"
    #: 0..1. 0.5 is the method's own threshold; >0.5 is more sensitive (lower
    #: threshold, catches fainter particles, merges more); <0.5 is stricter.
    sensitivity: float = 0.5
    rb_kernel: int = 0          # rolling-ball radius, 0 = off
    gaussian: float = 0.0       # pre-blur sigma, 0 = off
    invert: bool = False        # dark particles on a bright background
    local_size: int = 31        # window for the local threshold methods (odd)
    watershed: bool = True      # split touching particles
    #: Minimum distance between watershed markers, px. **This replaces
    #: ParticleSpy's ``watershed_size``**, whose semantics do not transfer: it
    #: filtered markers by AREA, which works for their thresholded-distance
    #: markers but silently deletes every small particle when markers are local
    #: maxima (a 3x3 particle's marker is one pixel, so any area floor erases it).
    #: That is precisely the sensitivity failure plan §0.9 exists to prevent.
    min_separation: int = 3
    #: Gaussian sigma applied to the distance transform BEFORE peak finding.
    #: Suppresses spurious maxima from a ragged boundary without merging genuinely
    #: separate particles. 0 disables.
    marker_smooth: float = 1.0
    watershed_erosion: int = 0  # erosions before the distance transform
    #: Grid decimation for the SPLIT geometry only. 0 = auto (see
    #: :func:`_split_factor`), 1 = never decimate.
    #:
    #: The distance transform is **61% of a 4096² segmentation** (3.93 s of
    #: 6.40 s measured; watershed itself is only 9%), and it is used for two
    #: things — finding markers and supplying the watershed's elevation — neither
    #: of which needs full resolution. Computing it on a decimated grid and
    #: upsampling the elevation gives **2.8–2.9× on 4k with identical particle
    #: counts and identical median areas** (0.0% difference, on both touching and
    #: isolated fields).
    #:
    #: This does NOT weaken detection, and the distinction matters: the threshold
    #: still runs at full resolution, so *which* bodies are found is unchanged —
    #: plan §0.9's faint-particle sensitivity is untouched. Only the cut BETWEEN
    #: two touching bodies moves, by about ``factor`` pixels.
    split_decimation: int = 0
    min_size: int = 20          # discard instances smaller than this, px
    max_size: int = 0           # discard instances larger than this, px; 0 = off
    clear_border: bool = False  # drop instances touching the frame edge
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold not in THRESHOLD_METHODS:
            raise ValueError(
                f"unknown threshold {self.threshold!r}; expected one of "
                f"{', '.join(THRESHOLD_METHODS)}"
            )
        if not 0.0 <= self.sensitivity <= 1.0:
            raise ValueError(f"sensitivity must be in 0..1; got {self.sensitivity}")
        if self.threshold in _LOCAL_METHODS and self.local_size % 2 == 0:
            # skimage requires an odd window; silently bumping it would make the
            # caret's number disagree with what actually ran.
            raise ValueError(
                f"local_size must be odd for threshold={self.threshold!r}; "
                f"got {self.local_size}"
            )


# ── preprocessing ────────────────────────────────────────────────────────────

def _rolling_ball(img: np.ndarray, radius: int) -> np.ndarray:
    """Background flatten via white top-hat, as ParticleSpy does it.

    Note this is the morphological top-hat, not skimage's newer
    ``restoration.rolling_ball``. Keeping ParticleSpy's actual operation matters
    for the parity gate — the two give visibly different backgrounds on a sloping
    carbon film.
    """
    from skimage.morphology import square, white_tophat
    if radius <= 0:
        return img
    return white_tophat(img, footprint=square(int(radius)))


def _prepare(frame: np.ndarray, p: SegmentParams) -> np.ndarray:
    """Rolling-ball → gaussian → invert, returning float32.

    NaN is filled with the finite minimum BEFORE filtering. A drift-corrected
    frame carries a NaN border (``spyde.drift.warp``), and every skimage filter
    propagates NaN outward, which would erase a band of real data around the edge.
    Filling with the minimum makes the padding read as background — the one value
    guaranteed not to threshold as a particle.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.util import invert as sk_invert

    img = np.asarray(frame, dtype=np.float32)
    bad = ~np.isfinite(img)
    if bad.any():
        finite = img[~bad]
        img = img.copy()
        img[bad] = finite.min() if finite.size else 0.0

    if p.rb_kernel > 0:
        img = _rolling_ball(img, p.rb_kernel)
    if p.gaussian > 0:
        img = gaussian_filter(img, float(p.gaussian))
    if p.invert:
        # sk_invert on a float image maps x -> -x, which is all thresholding needs.
        img = sk_invert(img)
    return img


# ── thresholding ─────────────────────────────────────────────────────────────

def _sensitivity_offset(img: np.ndarray, sensitivity: float) -> float:
    """Convert 0..1 sensitivity into an additive threshold offset.

    Scaled by the image's own robust spread (5-95 percentile), so the control
    behaves the same on a uint16 frame and a normalised float one. At 0.5 the
    offset is exactly zero, i.e. the method's own threshold is used unmodified —
    which keeps the default path bit-identical to plain Otsu and makes the parity
    test against ParticleSpy meaningful.
    """
    if sensitivity == 0.5:
        return 0.0
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return 0.0
    lo, hi = np.percentile(finite, [5, 95])
    spread = float(hi - lo)
    if spread <= 0:
        return 0.0
    # sensitivity 1.0 lowers the threshold by half the spread; 0.0 raises it.
    return -(float(sensitivity) - 0.5) * spread


def threshold_mask(img: np.ndarray, p: SegmentParams) -> np.ndarray:
    """Boolean foreground mask for a prepared image.

    Raises
    ------
    ValueError
        If the chosen method cannot be computed on this image. Some methods
        genuinely have preconditions — ``minimum`` needs a bimodal histogram and
        skimage raises ``RuntimeError`` when it cannot find two maxima, which
        happens on a sparse field of small bright particles (nearly all background,
        so the histogram is one spike). Re-raised here with the method named and a
        suggestion, because the bare skimage error gives the user nothing to act on.
    """
    from skimage import filters

    offset = _sensitivity_offset(img, p.sensitivity)

    try:
        return _apply_threshold(img, p, offset)
    except RuntimeError as exc:
        raise ValueError(
            f"threshold method {p.threshold!r} failed on this frame ({exc}). "
            "It requires a clearly bimodal intensity histogram; a sparse field of "
            "small particles does not have one. Try 'otsu', 'yen' or 'li', or a "
            "local method such as 'sauvola'."
        ) from exc


def _apply_threshold(img: np.ndarray, p: SegmentParams, offset: float) -> np.ndarray:
    from skimage import filters

    if p.threshold == "otsu":
        t = filters.threshold_otsu(img)
    elif p.threshold == "mean":
        t = filters.threshold_mean(img)
    elif p.threshold == "minimum":
        t = filters.threshold_minimum(img)
    elif p.threshold == "yen":
        t = filters.threshold_yen(img)
    elif p.threshold == "isodata":
        t = filters.threshold_isodata(img)
    elif p.threshold == "li":
        t = filters.threshold_li(img)
    elif p.threshold == "local":
        t = filters.threshold_local(img, block_size=p.local_size)
    elif p.threshold == "niblack":
        t = filters.threshold_niblack(img, window_size=p.local_size)
    elif p.threshold == "sauvola":
        t = filters.threshold_sauvola(img, window_size=p.local_size)
    elif p.threshold == "local_otsu":
        from skimage.filters.rank import otsu as rank_otsu
        from skimage.morphology import disk
        from skimage.util import img_as_ubyte
        lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
        norm = np.zeros_like(img) if hi <= lo else (img - lo) / (hi - lo)
        t_u8 = rank_otsu(img_as_ubyte(norm), disk(max(1, p.local_size // 2)))
        t = lo + (t_u8.astype(np.float32) / 255.0) * (hi - lo)
    else:                                        # pragma: no cover — guarded above
        raise ValueError(f"unknown threshold {p.threshold!r}")

    return img > (np.asarray(t, dtype=np.float32) + np.float32(offset))


# ── the shared instance split ────────────────────────────────────────────────

def split_instances(
    foreground: np.ndarray,
    p: SegmentParams,
    *,
    distance_from: np.ndarray | None = None,
) -> np.ndarray:
    """Split a foreground mask (or probability map) into labelled instances.

    **Shared by every engine.** ``foreground`` may be boolean, or a float
    probability in 0..1 (thresholded at 0.5), which is what the scribble and
    prompt engines produce.

    Parameters
    ----------
    distance_from
        Optional alternative to the binary distance transform for seeding
        watershed markers — e.g. a learned boundary-class probability. Passing the
        classifier's own notion of "interior" separates touching particles better
        than geometry does, which is why the hook exists.

    Returns
    -------
    ``(h, w)`` int32 label image, relabelled 1..n with no gaps.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import binary_erosion, disk
    from skimage.segmentation import clear_border, watershed

    fg = np.asarray(foreground)
    if fg.ndim != 2:
        raise ValueError(f"foreground must be 2-D; got shape {fg.shape}")
    if fg.dtype != bool:
        fg = fg > 0.5

    if p.min_size > 0 and fg.any():
        # Own implementation rather than skimage's `remove_small_objects`: that
        # function is mid-deprecation (its replacement removes objects smaller
        # than OR EQUAL to the threshold, a silent off-by-one against the
        # documented `min_size` meaning), and we already need `_drop_small` for
        # the post-watershed pass.
        lab0, _ = ndi.label(fg)
        fg = _drop_small(lab0, int(p.min_size)) > 0

    if not fg.any():
        return np.zeros(fg.shape, dtype=np.int32)

    if p.watershed:
        seed_src = fg
        if p.watershed_erosion > 0:
            seed_src = fg.copy()
            for _ in range(int(p.watershed_erosion)):
                seed_src = binary_erosion(seed_src, disk(1))

        if distance_from is not None:
            dist = np.asarray(distance_from, dtype=np.float32) * seed_src
            markers = _distance_markers(dist, fg, p)
        else:
            dist, markers = _distance_and_markers(seed_src, fg, p)

        if markers.max() > 0:
            labels = watershed(-dist, markers, mask=fg)
        else:
            labels, _ = ndi.label(fg)
    else:
        labels, _ = ndi.label(fg)

    labels = np.asarray(labels, dtype=np.int32)

    if p.clear_border:
        labels = clear_border(labels)
    if p.max_size > 0:
        labels = _drop_large(labels, int(p.max_size))
    if p.min_size > 0:
        labels = _drop_small(labels, int(p.min_size))

    return _relabel_sequential(labels)


#: Pixel count above which the split geometry decimates by default. Below roughly
#: this the distance transform is already sub-100 ms and decimating buys nothing
#: worth the (small) boundary shift.
_SPLIT_DECIMATE_ABOVE = 2 * 1024 * 1024


def _split_factor(shape: tuple[int, int], p: SegmentParams) -> int:
    """Decimation factor for the split geometry. See ``split_decimation``."""
    if p.split_decimation:
        return max(1, int(p.split_decimation))
    n = int(shape[0]) * int(shape[1])
    if n <= _SPLIT_DECIMATE_ABOVE:
        return 1
    # One step per 4x in area, capped: past 4 the markers of a small particle
    # start to merge, and the whole point is that detection is unaffected.
    return 2 if n <= 4 * _SPLIT_DECIMATE_ABOVE else 4


def _distance_and_markers(seed_src: np.ndarray, fg: np.ndarray,
                          p: SegmentParams) -> tuple[np.ndarray, np.ndarray]:
    """``(elevation, markers)`` for the watershed, decimating when it pays.

    The distance transform dominates a large-frame segmentation — 61% of a 4096²
    run — and is needed only to seed markers and to give watershed an elevation.
    Neither wants full resolution, so above a threshold both are computed on a
    decimated grid and the elevation is bilinearly upsampled (and rescaled by the
    factor, so it stays in pixel units).

    Measured on 4096²: 5.7 s → 2.0 s, with the SAME particle count and the same
    median area to 0.0%.
    """
    from scipy import ndimage as ndi

    factor = _split_factor(fg.shape, p)
    if factor <= 1:
        dist = ndi.distance_transform_edt(seed_src)
        return dist, _distance_markers(dist, fg, p)

    small = seed_src[::factor, ::factor]
    d_small = ndi.distance_transform_edt(small).astype(np.float32)
    markers_small = _distance_markers(d_small, small, p, factor=factor)

    # Nearest-neighbour for the MARKERS (a label must not be interpolated into a
    # value that names a different particle) and bilinear for the ELEVATION.
    h, w = fg.shape
    markers = np.repeat(np.repeat(markers_small, factor, 0), factor, 1)[:h, :w]
    markers = markers * fg

    from scipy.ndimage import zoom
    elev = zoom(d_small, factor, order=1, grid_mode=True, mode="nearest")
    elev = elev[:h, :w] * float(factor)      # back into pixel units
    if elev.shape != fg.shape:               # odd sizes: pad the last row/col
        pad = np.zeros(fg.shape, np.float32)
        pad[:elev.shape[0], :elev.shape[1]] = elev
        elev = pad
    return elev, np.asarray(markers, dtype=np.int32)


def _distance_markers(dist: np.ndarray, fg: np.ndarray,
                      p: SegmentParams, factor: int = 1) -> np.ndarray:
    """One marker per particle, from the local maxima of the distance transform.

    Two failure modes have to be avoided at once, and they pull in opposite
    directions:

    * **Thresholding the distance map merges neighbours.** Taking connected
      components of ``dist > k`` looks appealing and is wrong: two discs whose
      edges overlap have ``dist > k`` everywhere in the join, so the union is one
      connected core and watershed is handed a single marker — it then cannot
      split anything. This was the original implementation here and the touching-
      discs test caught it.
    * **Raw peak maxima over-split a round particle.** A disc's distance maximum
      is a flat plateau, so a maximum filter marks every plateau pixel; treated as
      separate markers, one disc becomes a pie chart of wedges.

    ``peak_local_max`` followed by ``ndi.label`` resolves both: the plateau's
    pixels are contiguous, so labelling collapses them into ONE marker, while two
    genuinely separate maxima stay separate. Smoothing the distance map first
    removes the boundary-roughness maxima that would otherwise fragment an
    irregular particle.

    No marker is ever dropped for being small — see
    :attr:`SegmentParams.min_separation`.
    """
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max

    if dist.max() <= 0:
        return np.zeros(fg.shape, dtype=np.int32)

    dist_pk = dist
    if p.marker_smooth > 0:
        from scipy.ndimage import gaussian_filter
        dist_pk = gaussian_filter(dist.astype(np.float32), float(p.marker_smooth))

    # `min_separation` is in FULL-frame pixels, so on a decimated grid it has to
    # come down by the same factor or a 3 px separation becomes an effective 6
    # and two close particles merge into one marker.
    coords = peak_local_max(
        dist_pk,
        min_distance=max(1, int(p.min_separation) // max(1, int(factor))),
        labels=fg,
        exclude_border=False,
    )
    peaks = np.zeros(fg.shape, dtype=bool)
    if len(coords):
        peaks[tuple(coords.T)] = True
    markers, _ = ndi.label(peaks)
    return np.asarray(markers, dtype=np.int32)


def _drop_small(labels: np.ndarray, min_size: int) -> np.ndarray:
    counts = np.bincount(labels.ravel())
    bad = np.flatnonzero(counts < min_size)
    bad = bad[bad > 0]
    if bad.size:
        labels = labels.copy()
        labels[np.isin(labels, bad)] = 0
    return labels


def _drop_large(labels: np.ndarray, max_size: int) -> np.ndarray:
    counts = np.bincount(labels.ravel())
    bad = np.flatnonzero(counts > max_size)
    bad = bad[bad > 0]
    if bad.size:
        labels = labels.copy()
        labels[np.isin(labels, bad)] = 0
    return labels


def _relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Renumber to 1..n with no gaps, so ``label`` is a usable per-frame index."""
    present = np.unique(labels)
    present = present[present > 0]
    if present.size == 0:
        return np.zeros(labels.shape, dtype=np.int32)
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lut[present] = np.arange(1, present.size + 1, dtype=np.int32)
    return lut[labels]


# ── the classical engine ─────────────────────────────────────────────────────

def segment_frame(frame: np.ndarray, p: SegmentParams | None = None) -> np.ndarray:
    """Classical segmentation of one frame → int32 label image.

    ``prepare → threshold → split_instances``. The whole ParticleSpy pipeline,
    with the instance step factored out so the other two engines share it.
    """
    p = p or SegmentParams()
    prepared = _prepare(frame, p)
    fg = threshold_mask(prepared, p)
    return split_instances(fg, p)
