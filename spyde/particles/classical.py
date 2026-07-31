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

Two routes through the split, and which one runs is the performance story
-------------------------------------------------------------------------
``split_instances`` has a **watershed** route and a **boundary** route.

The watershed route is geometry-only: it needs a distance transform to find
markers and to give the flood an elevation. That geometry — the transform, the
marker/elevation upsample and the flood itself — is **1.62 s of a 1.78 s split at
4096², and that split is most of a 2.7 s frame** (``benchmarks.md``). Every engine
funnels into it, so no amount of work on any one engine changes what a big frame
costs. (Do NOT go optimising the distance transform on the strength of this: since
``split_decimation`` shipped it is 4% of the frame, and the cost moved to the
upsample and the flood. ``benchmarks.md`` has the post-decimation breakdown.)

The boundary route is taken when the caller can supply a **boundary mask** — the
third class in the ilastik convention (particle / background / boundary). A
classifier told where the joins are hands back particles that are already
separated, so instances are plain connected components and **neither the distance
transform nor the watershed runs at all**. Only the scribble engine can produce
that mask, which is why the boundary class lives in
:mod:`~spyde.particles.scribble` and the route it unlocks lives here.

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
    #: Cap on the passes of label growth that reclaim the BOUNDARY ring back into
    #: the instances either side of it, on the boundary split path only (see
    #: :func:`_split_by_boundary`). **0 = grow until nothing more can be
    #: assigned**, which is the default and the only setting that reproduces the
    #: watershed's areas.
    #:
    #: Every boundary pixel is real particle, so ``ndi.label(fg & ~boundary)`` on
    #: its own under-reports area by the seam's width. Measured against the
    #: watershed on touching 44 px discs, median area error by seam width and
    #: pass count::
    #:
    #:     seam width   1 px    2 px    4 px    8 px
    #:     1 pass       0.0%   +0.0%   -1.2%   -4.0%
    #:     2 passes     0.0%    0.0%   +0.0%   -2.7%
    #:     converged    0.0%    0.0%    0.0%    0.0%
    #:
    #: A pass grows every label by one pixel, so it takes ``ceil(width/2)`` of
    #: them for the two sides to meet in the middle — and converging costs
    #: nothing, because each pass only visits the pixels still unassigned.
    #: **The particle COUNT is correct at every setting**, including 1; only the
    #: areas move, which is why this is a cap and not a correctness switch.
    boundary_reclaim: int = 0
    #: Grid decimation for the SPLIT geometry only. 0 = auto (see
    #: :func:`_split_factor`), 1 = never decimate.
    #:
    #: The distance transform **was 61% of a 4096² segmentation** before this
    #: shipped (3.93 s of 6.40 s measured; watershed itself only 9%), and it is
    #: used for two things — finding markers and supplying the watershed's
    #: elevation — neither of which needs full resolution. Computing it on a
    #: decimated grid and
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
    boundary: np.ndarray | None = None,
    distance_from: np.ndarray | None = None,
) -> np.ndarray:
    """Split a foreground mask (or probability map) into labelled instances.

    **Shared by every engine.** ``foreground`` may be boolean, or a float
    probability in 0..1 (thresholded at 0.5), which is what the scribble and
    prompt engines produce.

    Parameters
    ----------
    boundary
        Optional ``(h, w)`` mask or probability of the **inter-particle
        boundary** — the ilastik third class. When given (and non-empty) the
        instances come from plain connected components of ``foreground &
        ~boundary`` and **the distance transform and watershed never run**; see
        :func:`_split_by_boundary` for why that is the whole point.
    distance_from
        Optional alternative to the binary distance transform for seeding
        watershed markers — e.g. a learned boundary-class probability. Passing the
        classifier's own notion of "interior" separates touching particles better
        than geometry does, which is why the hook exists. Ignored when
        *boundary* is supplied, since that path has no markers to seed.

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

    bnd = _as_boundary(boundary, fg.shape)
    if bnd is not None:
        # The whole reason the boundary class exists — no EDT, no watershed.
        return _finalize_labels(_split_by_boundary(fg, bnd, p), p)

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

    return _finalize_labels(labels, p)


# ── the boundary split: connected components instead of a watershed ──────────

def _as_boundary(boundary, shape: tuple[int, int]) -> np.ndarray | None:
    """Coerce a boundary argument to a boolean mask, or None if there is none.

    An all-False boundary is treated as **absent**, not as "a boundary with no
    pixels". That is what makes the wizard's automatic switch safe: a user who
    has added the boundary class but not painted it yet gets the watershed route
    rather than a silent downgrade to unsplit connected components.
    """
    if boundary is None:
        return None
    b = np.asarray(boundary)
    if b.ndim != 2:
        raise ValueError(f"boundary must be 2-D; got shape {b.shape}")
    if tuple(b.shape) != tuple(shape):
        raise ValueError(
            f"boundary is {b.shape} but foreground is {shape} — they must "
            "describe the same frame")
    if b.dtype != bool:
        b = b > 0.5
    return b if b.any() else None


def _split_by_boundary(fg: np.ndarray, bnd: np.ndarray,
                       p: SegmentParams) -> np.ndarray:
    """Instances from ``fg & ~bnd`` — the reason the boundary class exists.

    The watershed route costs a **global** distance transform, a marker/elevation
    upsample and a flood — together 1.62 s of a 1.78 s split at 4096²
    (``benchmarks.md``). All of it is doing geometry's best guess at a question
    the classifier can simply be *told*: the
    ilastik convention paints particle / background / **boundary**, and a head
    trained on boundaries hands back touching particles already separated. Then
    instance extraction is one ``ndi.label`` and both the EDT and the watershed
    are skipped outright.

    Two things this does that a bare ``ndi.label(fg & ~bnd)`` does not:

    * **The ring is given back.** Every boundary pixel is real particle, so
      labelling only the cores under-reports area by the ring's width — 1.2% on a
      2 px seam, 5.2% on an 8 px one, measured against the watershed on touching
      44 px discs. :func:`_reclaim_boundary` grows the cores back out into it,
      recovering the same total foreground the watershed claimed, pixel for
      pixel. Where the two routes *cut* can still differ by a pixel or two —
      watershed cuts equidistant from two markers, growth breaks a tie toward
      the higher label id — which measured as 0.0% on those 44 px discs and 0.7%
      on a tighter 14 px pair.
    * **A component with no boundary through it is untouched.** An isolated
      particle has no boundary pixels, so its core *is* the whole particle and it
      passes through unchanged. The boundary path is therefore not a different
      answer for isolated bodies, only for touching ones.

    The pre-``min_size`` filter the watershed route runs is deliberately skipped:
    it exists to keep specks out of the watershed, and there is no watershed
    here. :func:`_finalize_labels` applies the same floor to the result.
    """
    from scipy import ndimage as ndi

    core = fg & ~bnd
    if not core.any():
        return np.zeros(fg.shape, dtype=np.int32)
    labels, _n = ndi.label(core)
    labels = np.asarray(labels, dtype=np.int32)
    return _reclaim_boundary(labels, fg, int(p.boundary_reclaim))


#: 8-neighbour offsets, as (dy, dx). 8 and not 4 so a diagonal step across a
#: thin boundary still reaches the core on the other side.
_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
               (0, 1), (1, -1), (1, 0), (1, 1))


def _reclaim_boundary(labels: np.ndarray, fg: np.ndarray,
                      max_passes: int) -> np.ndarray:
    """Grow *labels* out into the unlabelled foreground until it is all claimed.

    Only the **unassigned foreground** pixels are ever visited — the boundary
    ring, ~1% of a frame — so this is eight gathers over a shrinking index list
    rather than a morphological pass over the raster. A full-frame
    ``grey_dilation`` would be ~200 ms *per pass* at 4096²; the whole convergence
    there measures **82 ms**.

    The update is **synchronous** (Jacobi): every pixel's new label is computed
    from the state at the start of the pass and written afterwards, so one pass
    grows every label by exactly one pixel and the result does not depend on
    array order. A pixel reachable from two instances at the same distance goes
    to the higher label id — an arbitrary but *deterministic* tie-break, which is
    the property that matters (the alternative, nearest-by-Euclidean-distance,
    is a distance transform, and avoiding that is the entire point of this path).

    Termination is guaranteed without a cap: a pixel leaves the list as soon as
    it is claimed and never returns, and the loop stops the first time a whole
    pass claims nothing — which is exactly when the pixels left over are
    unreachable (an island of foreground with no core of its own, entirely
    fenced in by boundary). Those stay 0, and that is honest: they belong to no
    instance, and inventing an owner would be worse than leaving them out.

    *max_passes* is a cap for callers who want the seam left partly unassigned;
    0 means run to convergence.
    """
    if max_passes < 0:
        return labels
    h, w = labels.shape
    todo_y, todo_x = np.nonzero(fg & (labels == 0))
    if not todo_y.size:
        return labels

    # Pad by one so a neighbour offset can never wrap around a row edge; the
    # border ring stays 0, so it contributes nothing to the max.
    pw = w + 2
    padded = np.zeros((h + 2, pw), dtype=labels.dtype)
    padded[1:-1, 1:-1] = labels
    flat = padded.reshape(-1)
    todo = (todo_y + 1).astype(np.int64) * pw + (todo_x + 1)
    offsets = [dy * pw + dx for dy, dx in _NEIGHBOURS]

    passes = 0
    while todo.size:
        if max_passes and passes >= int(max_passes):
            break
        best = flat[todo + offsets[0]]
        for off in offsets[1:]:
            best = np.maximum(best, flat[todo + off])
        won = best > 0
        if not won.any():
            break                     # nothing adjacent to a label; never will be
        flat[todo[won]] = best[won]   # written AFTER the whole pass — see above
        todo = todo[~won]
        passes += 1
    return padded[1:-1, 1:-1]


# ── the size filter and the sequential relabel, fused ────────────────────────

def _finalize_labels(labels: np.ndarray, p: SegmentParams) -> np.ndarray:
    """Apply ``min_size``/``max_size`` and renumber to 1..n, in ONE pass.

    Equivalent to ``_relabel_sequential(_drop_small(_drop_large(labels)))`` and
    bit-identical to it (a test pins that), but it reads the label raster twice
    instead of six times. At 4096² the old chain was **302 ms** — 93 ms of
    ``bincount`` for ``_drop_small``, its ``np.isin`` write-back, then
    ``_relabel_sequential``'s ``np.unique`` (a full 16.7 M-element sort, 139 ms
    on its own) and a second gather. Here one ``bincount`` decides every label's
    fate and one LUT gather writes the final ids: **164 ms**.

    Dropping and renumbering cannot be separated without paying for the raster
    twice, which is why they are one function rather than two composed ones.
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels.ravel())
    # `counts > 0` and not `slice(1, None)`: a label id absent from the raster
    # (a gap left by clear_border) must not be handed an output number, which is
    # exactly what `np.unique` gave for free and a bare size test would not.
    keep = counts > 0
    keep[0] = False                                  # 0 is background, never an id
    if p.min_size > 0:
        keep &= counts >= int(p.min_size)
    if p.max_size > 0:
        keep &= counts <= int(p.max_size)

    n = int(keep.sum())
    lut = np.zeros(counts.size, dtype=np.int32)
    if n:
        # Ascending original id, which is the order `np.unique` produced.
        lut[keep] = np.arange(1, n + 1, dtype=np.int32)
    return lut[labels]


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

    The distance transform used to dominate a large-frame segmentation — 61% of a
    4096² run — and is needed only to seed markers and to give watershed an
    elevation. Neither wants full resolution, so above a threshold both are
    computed on a decimated grid and the elevation is bilinearly upsampled (and
    rescaled by the factor, so it stays in pixel units). That is what makes the
    61% historical: the transform is now 4% of the frame and the cost sits in the
    upsample and the flood.

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
    """Unused on the hot path — :func:`_finalize_labels` folds this in. Kept as
    the readable reference that test pins ``_finalize_labels`` against."""
    counts = np.bincount(labels.ravel())
    bad = np.flatnonzero(counts > max_size)
    bad = bad[bad > 0]
    if bad.size:
        labels = labels.copy()
        labels[np.isin(labels, bad)] = 0
    return labels


def _relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Renumber to 1..n with no gaps, so ``label`` is a usable per-frame index.

    Also off the hot path now (see :func:`_finalize_labels`) and kept for the
    same reason: it is the obvious-and-correct version the fused one is checked
    against.
    """
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
