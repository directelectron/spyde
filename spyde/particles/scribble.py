"""
scribble.py — the scribble-trained pixel classifier. Plan step B3, the workhorse.

The user paints a few strokes on a frame; this learns *their* data and produces a
per-frame foreground probability map that :func:`spyde.particles.instances.
split_instances` turns into instances. Same shared downstream stage as the
classical and prompt engines — the only thing that changes is how the first box
in the plan's §0.2 pipeline gets filled.

Why this and not a threshold
----------------------------
Plan §0.9 makes detection **sensitivity** the priority: missing a particle's first
appearance destroys the nucleation event, which is the most interesting thing in
the movie. No single global threshold spans a nucleation sequence — one that
catches a 3σ particle at t=0 is not the one that works at t=end. Measured on the
``particle_movie()`` fixture, whose two ``p_faint`` probes are exactly this case:
the classical engine at its default sensitivity finds **0 of 2**
(``test_particle_movie_fixture.py::test_default_sensitivity_misses_the_faint_probes``
pins that), and this classifier trained on eleven scribbles finds **2 of 2** while
keeping all seven bright ones.

Design decisions that are not obvious
-------------------------------------
**Multi-class, and class 0 is not special.** Classes are user-defined
``(id, name, colour)`` triples and a softmax head costs nothing over a sigmoid. In
EM "background" is genuinely two or three different things — carbon film, vacuum,
beam-stop — and forcing them into one class makes the head spend its capacity
proving they are the same rather than separating either from a particle. Which
classes count as foreground is a per-class flag (:attr:`ScribbleClass.particle`),
so several particle *phases* can be labelled separately and still sum into one
probability map.

**A BOUNDARY class, and it is a performance feature.** The ilastik convention is
particle / background / **boundary**, and the third one is not a refinement — it
is the only way out of the cost that dominates a large frame. Every engine ends
at :func:`spyde.particles.instances.split_instances`, whose watershed route needs
a global distance transform, a marker/elevation upsample and a flood — together
1.62 s of a 1.78 s split at 4096². All of that exists to *guess* where two
touching particles should be cut. A head that has
been shown a few strokes along the joins does not have to guess: it returns them
already separated, so the split degenerates to one ``ndi.label`` and both the
distance transform and the watershed are skipped. Measured on 4096²: **1.78 s →
0.33 s for the split**, and on that field it also found the exactly-correct 162
bodies where the watershed found 173.

:attr:`ScribbleClass.boundary` marks such a class, only this engine can produce
one, and :meth:`ScribbleClassifier.segment` passes it down automatically —
falling back to the watershed when nothing was painted, so not using the class
costs nothing.

**What "boundary" means is the whole art of it, and getting it wrong is silent.**
It is the seam BETWEEN two bodies, never the outline of one. A head taught
outlines learns "shrink everything": on the fixture's merge frame that MERGED the
touching pair and lost 40% of the median area, while still reporting a trained
boundary class. A boundary trained on a handful of pixels is no better — 30 px of
seam took the fast route and returned 81 bodies where the watershed found 162.
The per-class pixel counts in the caret are what surface this, which is why
:meth:`LabelStore.counts` lists classes with no pixels at all.

**Labels accumulate across frames, sparsely.** :class:`LabelStore` is keyed by
frame index, so painting on frame 0 and again on frame 400 trains one model from
both. It stores flat pixel indices, not label images: a scribble is a few thousand
pixels, and a dense ``int16`` map is 32 MB *per labelled frame* at 4096² — 320 MB
for ten labelled frames, to hold ~50 000 useful values.

**A torch MLP, with sklearn's RandomForest kept as the parity reference.** One
hidden layer, class-balanced cross-entropy. The forest is what ParticleSpy and
ilastik use and it is a *better* classifier out of the box on this kind of
tabular problem — but it cannot live on the GPU next to the feature stack, and
the interaction budget is train+apply under 1 s on the displayed frame. So the
MLP is the shipped head and agreement with the forest on identical labels and
identical features is the acceptance gate
(``test_particles_scribble.py::TestRandomForestParity``).

**The NaN border is forced to zero probability.** Plan trap 2: a drift-corrected
frame has a NaN-padded border, and segmentation that ignores it invents a large
"particle" along the edge which then nucleates a spurious track.
:func:`spyde.particles.features.prepare_frame` reports validity and every
prediction here is masked by it.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from spyde.device_lock import accelerator_lock
from spyde.particles.features import (
    FeatureSpec,
    import_torch,
    map_feature_bands,
    prepare_frame,
    sample_features,
    select_device,
)

log = logging.getLogger(__name__)

FORMAT_VERSION = 1

#: Sentinel for "no class" in a label map or a predicted label image. -1 rather
#: than 0 because class id 0 is a perfectly ordinary user class here.
UNLABELLED = -1

#: Default classes for a fresh session: a particle class, two backgrounds and a
#: boundary. Two backgrounds and not one because that is what EM actually looks
#: like, and because the second one is free — see the module docstring. Colours
#: are the renderer's accent family so the floating brush strip (plan B0) needs
#: no palette of its own.
#:
#: ``(id, name, colour, particle, boundary)``. The **boundary** class is the
#: ilastik convention and it is here for speed as much as for quality: painting
#: the joins between touching particles lets
#: :func:`~spyde.particles.instances.split_instances` take its connected-
#: components route and skip the distance transform and watershed entirely —
#: 1.78 s down to 0.33 s at 4096². It is not a particle class — a boundary
#: pixel is the seam, not the body — so it does not enter the foreground sum.
DEFAULT_CLASSES: tuple[tuple[int, str, str, bool, bool], ...] = (
    (0, "particle", "#f9a03f", True, False),
    (1, "support film", "#89b4fa", False, False),
    (2, "vacuum", "#585b70", False, False),
    (3, "boundary", "#f38ba8", False, True),
)


# ── classes and the label store ──────────────────────────────────────────────

@dataclass(frozen=True)
class ScribbleClass:
    """One user-defined class.

    Parameters
    ----------
    id
        Stable integer key. Referenced by :class:`LabelStore` and by the trained
        head's output column order, so it must not be reused after a delete.
    name, colour
        For the caret's class list and the in-canvas brush strip. ``colour`` is a
        CSS hex string — the renderer's units, not the backend's.
    particle
        Whether this class counts toward the foreground probability map. Several
        classes may set it (two particle phases, say) and their probabilities sum.
    boundary
        Whether this class marks the **seam between touching particles**. Summed
        the same way *particle* is, into a separate map that
        :func:`~spyde.particles.instances.split_instances` uses to skip the
        watershed. A class is one or the other, never both: a boundary pixel is
        not part of any body, and counting it as foreground would glue the two
        bodies it separates back together — which is the exact failure the class
        exists to prevent.
    """

    id: int
    name: str
    colour: str = "#ffffff"
    particle: bool = False
    boundary: bool = False

    def __post_init__(self) -> None:
        if self.particle and self.boundary:
            raise ValueError(
                f"class {self.id} ({self.name!r}) is marked both particle and "
                "boundary — a seam is not part of a body, so counting it as "
                "both would merge every pair of touching particles it separates")

    def to_dict(self) -> dict[str, Any]:
        return {"id": int(self.id), "name": self.name, "colour": self.colour,
                "particle": bool(self.particle),
                "boundary": bool(self.boundary)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScribbleClass":
        # `boundary` defaults False, so a session or a model saved before the
        # class existed loads as the particle/background-only setup it was.
        return cls(int(d["id"]), str(d["name"]), str(d.get("colour", "#ffffff")),
                   bool(d.get("particle", False)), bool(d.get("boundary", False)))


def default_classes() -> list[ScribbleClass]:
    """A fresh copy of :data:`DEFAULT_CLASSES`."""
    return [ScribbleClass(i, n, c, p, b) for i, n, c, p, b in DEFAULT_CLASSES]


@dataclass(eq=False)
class LabelStore:
    """Scribble labels, keyed by frame index and accumulating across frames.

    Painted pixels are held as flat indices into a ``(h, w)`` frame plus a
    parallel class-id array — see the module docstring for why this is not a dense
    label image. Repainting a pixel *replaces* its class (last write wins), which
    is what a user expects from a brush, and erasing removes it entirely rather
    than assigning it to a background class.

    Parameters
    ----------
    frame_shape
        ``(h, w)``. Fixed for the life of the store: a flat index means nothing
        without it, and silently accepting a differently-shaped frame would
        scatter the labels across the image.
    classes
        The class list. Mutate through :meth:`add_class` / :meth:`remove_class`
        so ids stay unique and a removed class takes its pixels with it.

    Notes
    -----
    ``eq=False``: the generated ``__eq__`` would compare a dict of numpy arrays
    and raise on the ambiguous truth value. Compare :meth:`to_dict` instead, which
    is what the round-trip test does.
    """

    frame_shape: tuple[int, int]
    classes: list[ScribbleClass] = field(default_factory=default_classes)
    #: frame index -> (flat indices int64, class ids int16)
    _frames: dict[int, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.frame_shape = (int(self.frame_shape[0]), int(self.frame_shape[1]))
        ids = [c.id for c in self.classes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate class ids: {ids}")

    # -- classes -------------------------------------------------------------

    @property
    def class_ids(self) -> list[int]:
        return [c.id for c in self.classes]

    def class_by_id(self, cid: int) -> ScribbleClass:
        for c in self.classes:
            if c.id == int(cid):
                return c
        raise KeyError(
            f"no class with id {cid}; have {self.class_ids}")

    def add_class(self, name: str, colour: str = "#ffffff", *,
                  particle: bool = False, boundary: bool = False,
                  id: int | None = None) -> ScribbleClass:
        """Append a class. Its id is one past the current maximum unless given."""
        cid = int(id) if id is not None else (
            max(self.class_ids) + 1 if self.classes else 0)
        if cid in self.class_ids:
            raise ValueError(f"class id {cid} already exists")
        c = ScribbleClass(cid, str(name), str(colour), bool(particle),
                          bool(boundary))
        self.classes.append(c)
        return c

    def remove_class(self, cid: int) -> None:
        """Drop a class **and every pixel labelled with it**.

        Leaving orphaned pixels behind would train a head with a column for a
        class the user has deleted, and the pixel counts in the caret would stop
        adding up to the labelled total — which is the one number that tells the
        user a class is under-trained.
        """
        cid = int(cid)
        self.class_by_id(cid)                      # raises if unknown
        self.classes = [c for c in self.classes if c.id != cid]
        for t in list(self._frames):
            idx, cls = self._frames[t]
            keep = cls != cid
            if keep.all():
                continue
            if keep.any():
                self._frames[t] = (idx[keep], cls[keep])
            else:
                del self._frames[t]

    # -- painting ------------------------------------------------------------

    def _flatten(self, where) -> np.ndarray:
        """Coerce a mask / ``(k, 2)`` yx / flat-index argument to flat indices."""
        h, w = self.frame_shape
        a = np.asarray(where)
        if a.dtype == bool:
            if a.shape != (h, w):
                raise ValueError(
                    f"mask shape {a.shape} != frame_shape {self.frame_shape}")
            return np.flatnonzero(a.reshape(-1)).astype(np.int64)
        if a.ndim == 2 and a.shape[1] == 2:
            ys = np.rint(a[:, 0]).astype(np.int64)
            xs = np.rint(a[:, 1]).astype(np.int64)
            inside = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
            return (ys[inside] * w + xs[inside])
        flat = a.reshape(-1).astype(np.int64)
        return flat[(flat >= 0) & (flat < h * w)]

    def paint(self, t: int, where, class_id: int) -> int:
        """Label pixels on frame *t*. Returns how many pixels the store now holds
        for that frame.

        *where* may be a boolean mask the shape of the frame, a ``(k, 2)``
        ``(y, x)`` array, or flat indices. Out-of-frame coordinates are dropped
        rather than raising — a brush stroke that runs off the edge is normal.
        """
        cid = int(class_id)
        self.class_by_id(cid)
        t = int(t)
        add = self._flatten(where)
        if not add.size:
            return len(self._frames.get(t, (np.empty(0),))[0])

        cur_idx, cur_cls = self._frames.get(
            t, (np.zeros(0, np.int64), np.zeros(0, np.int16)))
        idx = np.concatenate([cur_idx, add])
        cls = np.concatenate([cur_cls, np.full(add.size, cid, np.int16)])
        self._frames[t] = _dedup_last_wins(idx, cls)
        return int(self._frames[t][0].size)

    def paint_disc(self, t: int, y: float, x: float, radius: float,
                   class_id: int) -> int:
        """Label a filled disc — one brush dab, and what a click paints."""
        return self.paint(t, _disc_indices(self.frame_shape, y, x, radius),
                          class_id)

    def paint_stroke(self, t: int, points: Sequence[Sequence[float]],
                     class_id: int, *, brush: float = 3.0) -> int:
        """Label a brush stroke: a polyline of ``(y, x)`` points, *brush* px wide.

        This is the door the anyplotlib brush widget (plan B0) comes through, and
        the coordinates arrive in **image pixels** with no scale or offset applied
        — plan trap 6. Do not multiply by the axis scale on the way in.

        The polyline is densified to half-pixel steps before dabbing, because the
        widget emits a pointer sample per frame and a fast stroke can jump 20 px
        between them; dabbing only at the samples leaves a dotted line.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if not len(pts):
            return int(self._frames.get(t, (np.empty(0),))[0].size)
        r = max(0.5, float(brush) / 2.0)
        dense = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            n = int(np.ceil(np.hypot(*(b - a)) * 2.0))
            if n > 1:
                dense.extend(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
            else:
                dense.append(b)
        idx = np.unique(np.concatenate(
            [_disc_indices(self.frame_shape, py, px, r) for py, px in dense]))
        return self.paint(t, idx, class_id)

    def erase(self, t: int, where) -> int:
        """Unlabel pixels on frame *t* (the brush widget's eraser)."""
        t = int(t)
        got = self._frames.get(t)
        if got is None:
            return 0
        idx, cls = got
        drop = np.isin(idx, self._flatten(where))
        if drop.all():
            del self._frames[t]
            return 0
        self._frames[t] = (idx[~drop], cls[~drop])
        return int(self._frames[t][0].size)

    def clear_frame(self, t: int) -> None:
        self._frames.pop(int(t), None)

    def clear(self) -> None:
        self._frames.clear()

    # -- inspection ----------------------------------------------------------

    def labelled_frames(self) -> list[int]:
        """Frame indices carrying labels, ascending — the caret's revisit list."""
        return sorted(self._frames)

    def at(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """``(flat_indices, class_ids)`` for frame *t*; empty arrays if none."""
        return self._frames.get(
            int(t), (np.zeros(0, np.int64), np.zeros(0, np.int16)))

    def label_map(self, t: int) -> np.ndarray:
        """Frame *t*'s labels as a dense ``(h, w)`` int16 map, :data:`UNLABELLED`
        where unpainted. Built on demand for display; never stored (see the
        module docstring)."""
        idx, cls = self.at(t)
        out = np.full(self.frame_shape, UNLABELLED, dtype=np.int16)
        if idx.size:
            out.reshape(-1)[idx] = cls
        return out

    def counts(self) -> dict[int, int]:
        """Labelled pixels per class id, across every frame.

        This is what the caret's class list shows, and it is how a user notices a
        class is under-trained — so classes with zero pixels are present in the
        result rather than absent from it.
        """
        out = {c.id: 0 for c in self.classes}
        for idx, cls in self._frames.values():
            if not cls.size:
                continue
            vals, n = np.unique(cls, return_counts=True)
            for v, k in zip(vals.tolist(), n.tolist()):
                out[int(v)] = out.get(int(v), 0) + int(k)
        return out

    def __len__(self) -> int:
        return int(sum(idx.size for idx, _ in self._frames.values()))

    @property
    def n_classes_used(self) -> int:
        return int(sum(1 for v in self.counts().values() if v > 0))

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe, so a session's scribbles survive a save/reload.

        Indices go out as lists. That is fine at scribble scale (a few thousand
        per frame) and is what keeps the format inspectable; a binary side-car
        would only pay off at a density no human paints.
        """
        return {
            "frame_shape": list(self.frame_shape),
            "classes": [c.to_dict() for c in self.classes],
            "frames": {str(t): {"index": idx.tolist(), "class": cls.tolist()}
                       for t, (idx, cls) in sorted(self._frames.items())},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LabelStore":
        store = cls(
            frame_shape=tuple(d["frame_shape"]),
            classes=[ScribbleClass.from_dict(c) for c in d["classes"]],
        )
        for key, blk in (d.get("frames") or {}).items():
            store._frames[int(key)] = (
                np.asarray(blk["index"], dtype=np.int64),
                np.asarray(blk["class"], dtype=np.int16),
            )
        return store


def _dedup_last_wins(idx: np.ndarray, cls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse repeated indices, keeping the LAST class written to each.

    ``np.unique`` keeps the *first* occurrence of each value, so the arrays are
    reversed going in and the result flipped back — a repaint over an existing
    stroke has to change its class, not be ignored.
    """
    if idx.size < 2:
        return idx.astype(np.int64), cls.astype(np.int16)
    ridx, rcls = idx[::-1], cls[::-1]
    uniq, first = np.unique(ridx, return_index=True)
    return uniq.astype(np.int64), rcls[first].astype(np.int16)


def _disc_indices(shape: tuple[int, int], y: float, x: float,
                  radius: float) -> np.ndarray:
    """Flat indices of a filled disc, clipped to the frame.

    Built from the bounding box rather than a full-frame distance map: a brush dab
    is a handful of pixels and a 4096² ``mgrid`` per dab would make painting
    unusable.
    """
    h, w = int(shape[0]), int(shape[1])
    r = max(0.5, float(radius))
    y0, y1 = max(0, int(math.floor(y - r))), min(h - 1, int(math.ceil(y + r)))
    x0, x1 = max(0, int(math.floor(x - r))), min(w - 1, int(math.ceil(x + r)))
    if y1 < y0 or x1 < x0:
        return np.zeros(0, np.int64)
    ys = np.arange(y0, y1 + 1)[:, None]
    xs = np.arange(x0, x1 + 1)[None, :]
    inside = (ys - float(y)) ** 2 + (xs - float(x)) ** 2 <= r * r
    yy, xx = np.nonzero(inside)
    return ((yy + y0).astype(np.int64) * w + (xx + x0).astype(np.int64))


# ── the SAM/prompt bootstrap (plan §0.4) ─────────────────────────────────────

def masks_to_labels(
    masks,
    *,
    t: int = 0,
    frame_shape: tuple[int, int] | None = None,
    store: LabelStore | None = None,
    particle_class: int = 0,
    background_class: int = 1,
    gap: int = 2,
    background_dilation: int = 8,
    erode: int = 1,
) -> LabelStore:
    """Turn promptable-segmentation masks into scribble labels. Plan §0.4.

    "Click four particles with the prompt model. Those masks — plus their dilated
    surroundings as background — become the scribble classifier's training labels.
    Train, apply to all N frames. No painting at all." This is that handoff, and
    it is a first-class feature rather than a convenience: it is what makes the
    dense result adapted to the data instead of to COCO.

    Parameters
    ----------
    masks
        A boolean ``(h, w)`` mask, a sequence of them, or an ``(n, h, w)`` array.
    store
        Accumulate into an existing store (so several prompt clicks on several
        frames build one training set). A new one is created when omitted, which
        needs *frame_shape* or at least one mask to infer it from.
    gap
        Pixels either side of the mask boundary left **unlabelled**. The boundary
        is where the prompt model is least certain and where a particle's own
        soft edge lives; labelling it either way teaches the head the wrong
        thing about edges, which shows up as systematically over- or
        under-sized instances downstream.
    background_dilation
        Outer radius of the background ring, in pixels. It must be wider than
        *gap* or there is no ring at all.
    erode
        Pixels eroded off the mask interior before it becomes a particle label,
        for the same reason as *gap*. **Skipped for any mask the erosion would
        empty** — a 3 px particle is exactly the object plan §0.9 is about, and
        silently dropping it here would defeat the whole bootstrap.

    Returns
    -------
    LabelStore
        The store, with *particle_class* painted on the mask interiors and
        *background_class* on the surrounding rings. Rings never cover another
        mask, so particle A is not taught as background for particle B.

    Notes
    -----
    Each mask gets its own full-frame dilation, which is O(n_masks) frame-sized
    boolean passes. That is fine and deliberate at the scale this runs at — a
    handful of prompt clicks — and it is why this is not the batch path: the batch
    path is "train once, apply to N frames", which never comes back here.
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    arr = np.asarray(masks)
    if arr.dtype != bool:
        arr = arr.astype(bool)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(
            f"masks must be (h, w) or (n, h, w) boolean; got shape {arr.shape}")
    if background_dilation <= gap:
        raise ValueError(
            f"background_dilation ({background_dilation}) must exceed gap "
            f"({gap}) or the background ring is empty")

    shape = tuple(arr.shape[1:]) if frame_shape is None else tuple(frame_shape)
    if store is None:
        store = LabelStore(frame_shape=shape)
    elif tuple(store.frame_shape) != shape:
        raise ValueError(
            f"masks are {shape} but the store holds {store.frame_shape} labels")
    store.class_by_id(particle_class)
    store.class_by_id(background_class)

    union = arr.any(axis=0)
    # Everything within `gap` of ANY mask is off-limits as background — computed
    # once over the union rather than per mask, so overlapping prompts agree.
    near_any = binary_dilation(union, iterations=int(gap)) if gap > 0 else union

    for m in arr:
        if not m.any():
            continue
        inner = m
        if erode > 0:
            shrunk = binary_erosion(m, iterations=int(erode))
            if shrunk.any():
                inner = shrunk
        store.paint(t, inner, particle_class)
        ring = binary_dilation(m, iterations=int(background_dilation)) & ~near_any
        if ring.any():
            store.paint(t, ring, background_class)
    return store


# ── the head ─────────────────────────────────────────────────────────────────

def _frame_getter(frames) -> Callable[[int], np.ndarray]:
    """``t -> 2-D frame`` for a callable, a mapping, a stack, or a single frame.

    Delegates the stack cases to :func:`spyde.drift.frames.frame_source`, which
    already handles a HyperSpy signal, a dask array, a numpy array and a sequence
    of frames and — importantly — reads exactly one frame at a time. Training
    over a long movie must never materialise it (CLAUDE.md memory-safety rule),
    and re-deriving that accessor here would be a second place to get it wrong.
    """
    if callable(frames):
        return frames
    if isinstance(frames, dict):
        return lambda t: np.asarray(frames[int(t)])
    arr = frames if hasattr(frames, "ndim") else None
    if arr is not None and arr.ndim == 2:
        return lambda t, _a=arr: np.asarray(_a)
    from spyde.drift.frames import frame_source
    _n, get_frame, _shape = frame_source(frames)
    return get_frame


class ScribbleClassifier:
    """Multi-class pixel classifier over the torch feature stack.

    Parameters
    ----------
    spec
        The feature stack to classify. Saved with the weights — a model and the
        channels it was trained on are meaningless apart.
    hidden
        Hidden-layer width. 64 is where agreement with the RandomForest reference
        stops improving — measured IoU on the fixture: 16 → 0.915, 32 → 0.939,
        **64 → 0.941**, 128 → 0.936, 256 → 0.934 — and the fit is flat in width
        (0.45–0.58 s across all of those) because the cost is per-step dispatch,
        not arithmetic, so there is nothing to buy by going narrower.
    epochs, lr, weight_decay
        Full-batch Adam. 300 steps, again from the parity measurement:
        100 → 0.832, 200 → 0.899, **300 → 0.941**, 500 → 0.864 (it starts
        over-tightening the boundary past 300). One step is ~1.5 ms at *any* torch
        thread count from 1 to 24, i.e. entirely per-step overhead, so the epoch
        count is a fixed ~0.45 s independent of frame size and of how much was
        painted.

        Full-batch rather than mini-batch because the training set is *thousands*
        of rows by design (plan B3): a mini-batch loop would add a shuffle order,
        and therefore a seed dependence the user would perceive as the model
        changing when they changed nothing, in exchange for no speed at this size.
    seed
        Weight initialisation. Same seed + same labels → same model, bit for bit
        (``TestDeterminism``).
    device
        ``None`` auto-selects CUDA/MPS/CPU. Pass ``"cpu"`` in tests: torch-CUDA
        work segfaults under the pytest process on Windows (CLAUDE.md).
    """

    def __init__(
        self,
        spec: FeatureSpec | None = None,
        *,
        hidden: int = 64,
        epochs: int = 300,
        lr: float = 0.05,
        weight_decay: float = 1e-4,
        seed: int = 0,
        device=None,
    ) -> None:
        self.spec = spec or FeatureSpec()
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.device = select_device(device) if not hasattr(device, "type") else device
        self.classes: list[ScribbleClass] = []
        self._net = None
        self._mean = None            # (C,) feature standardisation, from training
        self._std = None
        self.report: dict[str, Any] = {}

    # -- state ---------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._net is not None

    @property
    def particle_class_ids(self) -> list[int]:
        return [c.id for c in self.classes if c.particle]

    @property
    def boundary_class_ids(self) -> list[int]:
        """Trained classes marked :attr:`ScribbleClass.boundary`.

        Empty when the user never painted a boundary — ``fit`` drops classes with
        no labelled pixels, so this answers "was a boundary actually taught",
        not "does a boundary class exist in the list". That distinction is what
        :meth:`segment` switches on.
        """
        return [c.id for c in self.classes if c.boundary]

    @property
    def has_boundary(self) -> bool:
        """True when a boundary class carries trained weight — i.e. when
        :meth:`segment` will take the connected-components route."""
        return bool(self.boundary_class_ids)

    def _require_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError(
                "this ScribbleClassifier has not been trained — call fit() with a "
                "LabelStore first")

    # -- training ------------------------------------------------------------

    def fit(self, store: LabelStore, frames, *,
            progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        """Train on every labelled pixel in *store*.

        Parameters
        ----------
        store
            The accumulated scribbles. Every labelled frame contributes; classes
            with no labelled pixels are dropped from the head (a column that
            never sees a positive example would only ever emit noise).
        frames
            How to get frame *t*: a callable, a ``{t: frame}`` mapping, a 3-D
            stack, a HyperSpy signal, or a single 2-D frame (for a one-frame
            store). Read one frame at a time — never materialised.
        progress
            ``progress(done, total)`` over the labelled frames, then once more at
            completion. Featurising is the slow part, so it is reported per frame
            rather than per training epoch.

        Returns
        -------
        dict
            Training report: per-class pixel counts, final loss and training
            accuracy, the wall-clock split between featurising and fitting, and
            the device used. This is what the caret shows, and the featurise/fit
            split is what tells a user whether adding a labelled frame or
            widening the head is what costs them.

        Notes
        -----
        Cost is ``one featurise per labelled frame`` plus a fixed fit. Measured on
        the 96×112 fixture, CPU, default spec: **14 ms featurise + 457 ms fit**, so
        the interaction budget is met with room, and re-training after adding a
        stroke on a *new* frame costs one more featurise (≈1.1 s at 2048²). There is
        deliberately **no cached feature sampler**: the cache would have to be
        invalidated when the underlying frames change (a re-drift, a different
        node), and nothing in this API can observe that — a silently stale
        training set is far worse than a re-featurise the report already shows the
        cost of.
        """
        torch = import_torch()
        if len(store) == 0:
            raise ValueError("nothing painted yet — the label store is empty")

        get_frame = _frame_getter(frames)
        t_frames = store.labelled_frames()

        # ONE lock acquisition around the whole fit, not one per stage. Every line
        # below submits to the device — the per-frame featurise, the cat, the
        # standardisation, the optimiser loop, the accuracy read — and MPS needs
        # all of them serialised (CLAUDE.md § GPU Computing). Held across the fit
        # for the same reason `drift.translation.solve_translation` holds it
        # across a solve: this completes in well under a second by contract, so a
        # concurrent preview waits a bounded time, and a partial hold would leave
        # exactly the gaps that took the backend down last time. Reentrant, so
        # `sample_features` taking it again is free. Null context off MPS.
        t0 = time.perf_counter()
        with accelerator_lock(self.device):
            xs, ys = [], []
            for i, t in enumerate(t_frames):
                idx, cls = store.at(t)
                if not idx.size:
                    continue
                frame = np.asarray(get_frame(t))
                if tuple(frame.shape) != tuple(store.frame_shape):
                    raise ValueError(
                        f"frame {t} is {frame.shape} but the label store holds "
                        f"{store.frame_shape} labels — the flat indices would "
                        "land in the wrong pixels")
                xs.append(sample_features(frame, idx, self.spec,
                                          device=self.device))
                ys.append(torch.as_tensor(cls.astype(np.int64),
                                          device=self.device))
                if progress is not None:
                    progress(i + 1, len(t_frames))
            t_feat = time.perf_counter() - t0

            X = torch.cat(xs, dim=0)
            y_raw = torch.cat(ys, dim=0)

            # Only classes that actually carry labels become head columns,
            # remapped to a contiguous 0..K-1 so cross-entropy has no dead output.
            present = sorted({int(v) for v in y_raw.unique().tolist()})
            if len(present) < 2:
                raise ValueError(
                    f"only one class is painted (id {present[0]}) — a classifier "
                    "needs at least two, e.g. a particle and some background")
            self.classes = [store.class_by_id(c) for c in present]
            lut = torch.full((max(present) + 1,), -1, dtype=torch.long,
                             device=self.device)
            for k, cid in enumerate(present):
                lut[cid] = k
            y = lut[y_raw]

            t1 = time.perf_counter()
            self._mean = X.mean(dim=0)
            # Standardise per channel. A guard on the std and not an epsilon: a
            # constant channel (a rank filter on a flat region) divided by a tiny
            # number amplifies float noise to unit scale, and the head then fits
            # it — the same failure the drift solver's phase FLOOR exists for.
            std = X.std(dim=0, unbiased=False)
            self._std = torch.where(std > 1e-6, std, torch.ones_like(std))
            Xn = (X - self._mean) / self._std

            self._net = _build_mlp(torch, X.shape[1], self.hidden, len(present),
                                   self.seed, self.device)
            counts = torch.bincount(y, minlength=len(present)).to(Xn.dtype)
            # Class-balanced loss. Scribbles are wildly unbalanced by nature — a
            # user paints a few dabs on particles and sweeps whole regions of
            # background — and unweighted cross-entropy on a 40:1 split simply
            # learns "background", which reads as the classifier not working.
            weight = X.shape[0] / (len(present) * counts.clamp_min(1.0))
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
            opt = torch.optim.Adam(self._net.parameters(), lr=self.lr,
                                   weight_decay=self.weight_decay)
            loss = float("nan")
            for _ in range(self.epochs):
                opt.zero_grad(set_to_none=True)
                out = self._net(Xn)
                lo = loss_fn(out, y)
                lo.backward()
                opt.step()
                loss = float(lo.detach())
            with torch.no_grad():
                acc = float((self._net(Xn).argmax(dim=1) == y).to(Xn.dtype).mean())
        t_fit = time.perf_counter() - t1

        self.report = {
            "device": str(self.device),
            "n_pixels": int(X.shape[0]),
            "n_channels": int(X.shape[1]),
            "n_classes": len(present),
            # Whether a boundary class carries trained weight, i.e. whether the
            # split will take its connected-components route. Surfaced in the
            # report because it is the difference between a 0.33 s and a 1.78 s
            # split at 4096², and the user is the one who decides it by painting.
            "has_boundary": bool([c for c in self.classes if c.boundary]),
            "labelled_frames": list(t_frames),
            "pixels_per_class": {str(c.id): int(n) for c, n in
                                 zip(self.classes, counts.tolist())},
            "loss": loss,
            "train_accuracy": acc,
            "featurise_s": t_feat,
            "fit_s": t_fit,
        }
        if progress is not None:
            progress(len(t_frames), len(t_frames))
        log.info("[scribble] trained on %d px x %d ch, %d classes: acc %.3f "
                 "(featurise %.2f s, fit %.2f s, %s)", X.shape[0], X.shape[1],
                 len(present), acc, t_feat, t_fit, self.device)
        return self.report

    # -- prediction ----------------------------------------------------------

    def predict_class_proba(self, frame) -> np.ndarray:
        """``(K, H, W)`` float32 softmax over the trained classes.

        Column *k* is ``self.classes[k]``. Pixels that were non-finite in *frame*
        get probability 0 in **every** class, so the columns do not sum to 1
        there — that is deliberate and is what the NaN-border contract means: an
        invalid pixel is not "probably background", it is not a measurement.
        """
        self._require_trained()
        torch = import_torch()
        prepared = prepare_frame(frame, self.spec)
        h, w = prepared.image.shape

        # The lock spans the allocation, the per-band head evaluation AND the
        # read-back: a device->host copy is a submission too, and doing it after
        # releasing is exactly the kind of gap that reopens the MPS crash.
        with accelerator_lock(self.device):
            out = torch.zeros((len(self.classes), h, w), device=self.device,
                              dtype=torch.float32)

            def band(y0: int, y1: int, stack) -> None:
                # `stack` is a row slice of a larger tensor and so not contiguous;
                # `reshape` copies when it must, which `view` would refuse to do.
                flat = stack.reshape(stack.shape[0], -1).t()          # (rows*w, C)
                with torch.no_grad():
                    p = torch.softmax(
                        self._net((flat - self._mean) / self._std), dim=1)
                out[:, y0:y1] = p.t().reshape(len(self.classes), y1 - y0, w)

            map_feature_bands(prepared, self.spec, device=self.device, fn=band)
            proba = out.detach().cpu().numpy()

        proba[:, ~prepared.valid] = 0.0
        return proba

    def predict_proba(self, frame) -> np.ndarray:
        """``(H, W)`` float32 **foreground** probability, 0..1.

        The sum over every class with :attr:`ScribbleClass.particle` set, which is
        the per-frame probability map the plan's §0.2 spine consumes — hand it
        straight to :func:`spyde.particles.instances.split_instances`.

        Zero inside a NaN-padded region (plan trap 2).
        """
        return self.predict_foreground_boundary(frame)[0]

    def predict_boundary_proba(self, frame) -> np.ndarray | None:
        """``(H, W)`` float32 **boundary** probability, or None if untrained.

        None and not a zero map, because the two mean different things to
        :func:`~spyde.particles.instances.split_instances`: "no boundary was
        taught, use the watershed" versus "a boundary was taught and this frame
        has none of it", which would leave every touching pair merged.
        """
        return self.predict_foreground_boundary(frame)[1]

    def predict_foreground_boundary(
        self, frame
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """``(foreground, boundary)`` from **one** pass over the frame.

        Both maps come out of the same softmax, and that softmax is the whole
        cost of this engine — 1.2 s of a 1.5 s prediction at 4096². Calling
        :meth:`predict_proba` and :meth:`predict_boundary_proba` separately would
        featurise the frame twice for two views of one result, so the pair is the
        primitive and the two singular accessors are the wrappers.

        *boundary* is None when no trained class is marked
        :attr:`ScribbleClass.boundary`.
        """
        self._require_trained()
        proba = self.predict_class_proba(frame)
        wanted = [k for k, c in enumerate(self.classes) if c.particle]
        if not wanted:
            raise RuntimeError(
                "no trained class is marked as a particle class, so there is no "
                "foreground to report — set ScribbleClass.particle on at least "
                "one of "
                f"{[c.name for c in self.classes]} and retrain")
        edge = [k for k, c in enumerate(self.classes) if c.boundary]
        return _sum_planes(proba, wanted), (_sum_planes(proba, edge) if edge
                                            else None)

    def predict_labels(self, frame) -> np.ndarray:
        """``(H, W)`` int16 argmax over classes, as **class ids**.

        :data:`UNLABELLED` (-1) where the source pixel was non-finite. Class ids,
        not column indices, so the value indexes straight into the user's class
        list and its colour.
        """
        proba = self.predict_class_proba(frame)
        ids = np.asarray([c.id for c in self.classes], dtype=np.int16)
        out = ids[proba.argmax(axis=0)]
        out[proba.sum(axis=0) <= 0.0] = UNLABELLED
        return out

    def segment(self, frame, params=None) -> np.ndarray:
        """Convenience: probability → labelled instances via the shared split.

        The whole point of plan §0.2 is that this engine stops at a probability
        map and the instance stage is written once, so this is a short forwarder
        — kept here only so the caller does not have to remember which module
        owns the split.

        **The boundary is passed on when there is one**, and that is what makes
        this engine fast rather than merely accurate: with a taught boundary the
        split takes its connected-components route and never runs the distance
        transform or the watershed. Without one it falls back to the watershed,
        so a user who has not painted any boundary gets exactly the behaviour
        they had before — never a silently worse split.
        """
        from spyde.particles.instances import SegmentParams, split_instances
        fg, bnd = self.predict_foreground_boundary(frame)
        return split_instances(fg, params or SegmentParams(), boundary=bnd)

    # -- serialisation -------------------------------------------------------

    def save(self, path: str) -> None:
        """Write weights + :class:`FeatureSpec` + classes to one ``.npz``.

        One file, never two: a recipe is the spec *and* the weights *and* the
        feature standardisation, and any of them alone predicts nonsense. Written
        with ``np.savez_compressed`` and read back with ``allow_pickle=False``,
        matching :meth:`spyde.signals.particles.SpyDEParticles.save` — a model
        file should not be able to execute code on load.
        """
        self._require_trained()
        with accelerator_lock(self.device):
            state = {k: v.detach().cpu().numpy()
                     for k, v in self._net.state_dict().items()}
            mean = self._mean.detach().cpu().numpy()
            std = self._std.detach().cpu().numpy()
        meta = {
            "format_version": FORMAT_VERSION,
            "spec": self.spec.to_dict(),
            "classes": [c.to_dict() for c in self.classes],
            "hidden": self.hidden,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "state_keys": sorted(state),
            "report": _jsonable(self.report),
        }
        np.savez_compressed(
            path,
            meta=np.array(json.dumps(meta)),
            feature_mean=mean,
            feature_std=std,
            **{f"w_{k}": v for k, v in state.items()},
        )

    @classmethod
    def load(cls, path: str, *, device=None) -> "ScribbleClassifier":
        """Read a model written by :meth:`save`.

        The host→device transfers are taken under the accelerator lock. A cold
        model load is the *specific* unlocked call site CLAUDE.md names — the
        neural detector's ``load_model`` had exactly this hole and it was one of
        the two threads in the MPS crash.
        """
        torch = import_torch()
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"].item()))
            ver = meta.get("format_version")
            if ver != FORMAT_VERSION:
                raise ValueError(
                    f"unsupported scribble model format version {ver!r} "
                    f"(this build reads {FORMAT_VERSION})")
            model = cls(
                FeatureSpec.from_dict(meta["spec"]),
                hidden=int(meta["hidden"]),
                epochs=int(meta.get("epochs", 300)),
                lr=float(meta.get("lr", 0.05)),
                weight_decay=float(meta.get("weight_decay", 1e-4)),
                seed=int(meta.get("seed", 0)),
                device=device,
            )
            model.classes = [ScribbleClass.from_dict(c) for c in meta["classes"]]
            model.report = meta.get("report") or {}
            n_in = int(np.size(z["feature_mean"]))
            if n_in != model.spec.n_channels:
                raise ValueError(
                    f"model expects {n_in} feature channels but its saved "
                    f"FeatureSpec produces {model.spec.n_channels} — the spec and "
                    "the weights have come apart")
            with accelerator_lock(model.device):
                model._mean = torch.as_tensor(z["feature_mean"],
                                              device=model.device)
                model._std = torch.as_tensor(z["feature_std"],
                                             device=model.device)
                net = _build_mlp(torch, n_in, model.hidden, len(model.classes),
                                 model.seed, model.device)
                net.load_state_dict(
                    {k: torch.as_tensor(z[f"w_{k}"], device=model.device)
                     for k in meta["state_keys"]})
                model._net = net
        return model


def _sum_planes(proba: np.ndarray, cols: list[int]) -> np.ndarray:
    """Sum the given class planes of a ``(K, H, W)`` softmax, float32.

    The single-column case is the overwhelmingly common one (one particle class,
    one boundary class) and it is taken by a plain slice: ``proba[[k]].sum(0)``
    on a 4096² map builds a fancy-indexed copy and then reduces it, which
    measured **113 ms** for the pair against ~0 ms for two views.
    """
    if len(cols) == 1:
        plane = proba[cols[0]]
        return plane if plane.dtype == np.float32 else plane.astype(np.float32)
    return proba[cols].sum(axis=0).astype(np.float32)


def _build_mlp(torch, n_in: int, hidden: int, n_out: int, seed: int, device):
    """One hidden layer, ReLU, deterministically initialised from *seed*.

    The weights are drawn from an explicitly seeded ``torch.Generator`` and copied
    in, rather than letting ``nn.Linear`` use the global RNG: the global stream is
    shared with everything else in the process (the neural detector, a dask
    worker), so "same seed, same labels, same model" would otherwise depend on
    what else happened to run first. ``torch.nn.init`` has no generator argument
    on this torch version, which is why the scaling is written out.

    The construction is nevertheless wrapped in ``fork_rng``, because
    ``nn.Linear.__init__`` draws its own default initialisation from the global
    stream *before* it is overwritten here — so without the fork, training a
    scribble model would silently shift every other consumer's random sequence.
    ``devices=[]`` forks the CPU generator only; forking CUDA's would initialise
    the CUDA context as a side effect.
    """
    g = torch.Generator().manual_seed(int(seed))
    with torch.random.fork_rng(devices=[]):
        net = torch.nn.Sequential(
            torch.nn.Linear(n_in, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_out),
        )
    with torch.no_grad():
        for lin, fan_in in ((net[0], n_in), (net[2], hidden)):
            w = torch.randn(lin.weight.shape, generator=g) / math.sqrt(fan_in)
            lin.weight.copy_(w)
            lin.bias.zero_()
    return net.to(device)


def _jsonable(obj):
    """Coerce a report dict to something ``json.dumps`` accepts."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ── the sklearn parity reference (test-only, kept next to what it checks) ─────

def random_forest_reference(
    store: LabelStore,
    frames,
    spec: FeatureSpec | None = None,
    *,
    n_estimators: int = 100,
    seed: int = 0,
    device=None,
):
    """Train ``sklearn.ensemble.RandomForestClassifier`` on the SAME features.

    This is the acceptance reference for the whole engine (plan B3): the forest is
    what ParticleSpy and ilastik use, so agreement on identical labels and
    identical channels is what says the torch head is a re-implementation rather
    than a different algorithm that happens to produce pictures.

    It lives here, beside the head it checks, rather than in the test file — the
    two must read the same feature stack through the same sampler, and the moment
    the reference has its own copy of that plumbing the comparison stops being
    apples to apples.

    Returns
    -------
    (model, predict)
        ``predict(frame)`` gives an ``(H, W)`` float32 foreground probability, on
        the same convention as
        :meth:`ScribbleClassifier.predict_proba` including the NaN-border zeroing.
    """
    from sklearn.ensemble import RandomForestClassifier

    spec = spec or FeatureSpec()
    device = select_device(device) if not hasattr(device, "type") else device
    get_frame = _frame_getter(frames)

    xs, ys = [], []
    with accelerator_lock(device):
        for t in store.labelled_frames():
            idx, cls = store.at(t)
            if not idx.size:
                continue
            xs.append(sample_features(get_frame(t), idx, spec,
                                      device=device).detach().cpu().numpy())
            ys.append(cls.astype(np.int64))
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(X, y)
    particle_ids = {c.id for c in store.classes if c.particle}
    cols = [i for i, c in enumerate(rf.classes_) if int(c) in particle_ids]

    def predict(frame) -> np.ndarray:
        from spyde.particles.features import feature_stack
        prepared = prepare_frame(frame, spec)
        with accelerator_lock(device):
            # The forest is CPU-only, so unlike `predict_class_proba` this one
            # genuinely does materialise the whole stack (numpy) before scoring —
            # which is the other reason the forest is a reference and not the
            # shipped head.
            stack = feature_stack(prepared, spec, device=device)
        c, h, w = stack.shape
        p = rf.predict_proba(stack.reshape(c, -1).T)
        out = p[:, cols].sum(axis=1).reshape(h, w).astype(np.float32)
        out[~prepared.valid] = 0.0
        return out

    return rf, predict
