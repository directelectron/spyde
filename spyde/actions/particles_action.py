"""
particles_action.py — the Segment Particles wizard (``seg_`` staged actions).

Plan B7, honouring the §0.8 interaction contract literally:

    seg_open        caret mounted → preview the CURRENT frame, emit caret state
    seg_close       caret unmounted → clear the overlay, drop the controller
    seg_set_method  classical | scribble | prompt
    seg_tune        debounced param change → re-preview the CURRENT frame only
    seg_paint       one brush stroke → LabelStore
    seg_train       fit the scribble classifier on the accumulated labels
    seg_run         whole movie on a worker: progressive, cancellable
    seg_commit      snapshot the previewed frame as a one-frame particle tree

The three things that shape this module, none of them cosmetic:

**1. Preview is one frame, always.** ``seg_tune`` reads exactly the frame the
navigator is sitting on. A movie is thousands of frames at the plan's target
scale, so a tune that touched more than one would put the interaction budget
(plan B3: train + apply under ~1 s) out of reach on the first drag of a slider.

**2. The run opens its result tree EARLY and attaches nothing until it
finalizes.** ``open_particle_tree`` sets ``tree.particles`` at construction —
which is right for its own contract (a finished segmentation) and wrong for a
progressive run, because ``requires_particles`` would unlock downstream actions
against a store holding zero particles. So the run hands it a placeholder,
immediately clears ``tree.particles`` back to ``None``, and sets it at
``_finalize``. That is the attach gap ``lifecycle.wait_for_particles`` and
``lifecycle.seg_batch_running`` exist to cover, and it only means anything if
the flag and the attribute move at the right moments.

The placeholder is then MUTATED IN PLACE rather than replaced, because the lazy
label movie ``open_particle_tree`` built closes over that exact object — a fresh
``SpyDEParticles`` would leave the movie rendering the placeholder's zeros
forever.

**3. min_size is floored, and the floor is reported.** Plan §0.9 measured it: at
``min_size=0`` a classifier taught faint contrast produced 33 instances where 9
were real, and ``min_size=10`` removed 24 of the 25 spurious ones. A user who
zeroes it to "catch the small ones" gets the opposite. So it is floored — and
the EFFECTIVE value goes back to the caret in every preview, because silently
running a number different from the one on screen is the failure mode
``classical.SegmentParams`` refuses for ``local_size`` and this must not
reintroduce it.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np

from spyde.actions.context import current_signal as _current_signal
from spyde.actions.context import src_plot_tree as _src_plot_tree
from spyde.actions.lifecycle import bump_generation, is_current, run_on_worker
from spyde.actions.wizard import WizardController
from spyde.backend.ipc import emit, emit_error, emit_progress, emit_status

log = logging.getLogger(__name__)

#: The three mask sources of plan §0.2. ``prompt`` is declared here so the caret
#: can render its tab from the schema before the engine lands (plan B4); every
#: code path that would run it emits a "not installed yet" status instead.
METHODS: tuple[str, ...] = ("classical", "scribble", "prompt")

#: Floor applied to ``min_size``. NOT a taste default — plan §0.9's measurement:
#: on the fixture, one faint scribble added to bright-only labels gave 33
#: instances at min_size=0 (25 of them spurious) and 9 at min_size=10. The
#: classifier is not what buys specificity; this filter is.
MIN_SIZE_FLOOR = 10

#: Cap on the per-frame area list shipped to the caret's size histogram. A
#: histogram of a few hundred bodies is already at its useful resolution and the
#: PLOTAPP line protocol is shared with the nav painter thread — a 20k-element
#: list per slider tick is exactly the traffic plan B0 rejects for brush strokes.
_MAX_AREAS_SENT = 2000

#: Minimum wall-clock gap between progressive count-trace paints during a run.
#: Matches ``live_fill_poller``'s default: the fill is a reassurance signal, not
#: an animation, and every paint is a marshal onto the asyncio main thread.
_PROGRESS_INTERVAL = 0.35

DEFAULTS: dict[str, Any] = dict(
    method="classical",
    # The one sensitivity axis of plan §0.9. Everything else about the split is
    # secondary and sits below it in the caret.
    sensitivity=0.5,
    threshold="otsu",
    min_size=20,
    max_size=0,
    watershed=True,
    min_separation=3,
    marker_smooth=1.0,
    gaussian=0.0,
    rb_kernel=0,
    invert=False,
    local_size=31,
    clear_border=False,
    # Outlines are what make the overlay and the label movie possible, so they
    # default ON; plan §0.5 turns them off for very long movies.
    store_masks=True,
    track=True,
    max_dist=10.0,
    brush=3.0,
    # The paint state the ClassStrip owns. These MUST be real parameters: the
    # brush widget lives in Python, so the strip's choices only reach the paint
    # by travelling through here. They were read but never declared or set, which
    # pinned every stroke to class 0 and made the eraser a no-op.
    active_class=0,
    erase=False,
)


class SegmentWizard(WizardController):
    """Owns the segmentation caret's state: engine, parameters, scribbles,
    trained head, the last preview, and the run's result tree."""

    key = "seg"

    # Declared parameter schema — the single source of truth every host renders
    # from (registry.wizard_parameters("seg")). `tab` mirrors plan B7's three
    # engine tabs; `method` is the tab bar itself and so carries no tab.
    parameters = {
        "method": {
            "name": "Engine", "type": "enum", "default": DEFAULTS["method"],
            "choices": list(METHODS),
        },
        "sensitivity": {
            "name": "Sensitivity", "type": "float", "default": DEFAULTS["sensitivity"],
            "min": 0.0, "max": 1.0, "step": 0.01, "tab": "Classical",
        },
        "threshold": {
            "name": "Threshold", "type": "enum", "default": DEFAULTS["threshold"],
            "choices": ["otsu", "mean", "minimum", "yen", "isodata", "li",
                        "local", "local_otsu", "niblack", "sauvola"],
            "tab": "Classical",
        },
        "gaussian": {
            "name": "Pre-blur σ (px)", "type": "float", "default": DEFAULTS["gaussian"],
            "min": 0.0, "max": 10.0, "step": 0.1, "tab": "Classical",
        },
        "rb_kernel": {
            "name": "Rolling ball (px)", "type": "int", "default": DEFAULTS["rb_kernel"],
            "min": 0, "max": 256, "tab": "Classical",
        },
        "invert": {
            "name": "Dark particles", "type": "bool", "default": DEFAULTS["invert"],
            "tab": "Classical",
        },
        "local_size": {
            "name": "Local window (px, odd)", "type": "int",
            "default": DEFAULTS["local_size"], "min": 3, "max": 255,
            "tab": "Classical",
        },
        # min_size sits next to sensitivity deliberately (plan §0.9: the two are
        # coupled and must not be in separate tabs).
        "min_size": {
            "name": "Min size (px)", "type": "int", "default": DEFAULTS["min_size"],
            "min": 0, "max": 100000, "tab": "Split",
        },
        "max_size": {
            "name": "Max size (px, 0=off)", "type": "int",
            "default": DEFAULTS["max_size"], "min": 0, "max": 10000000,
            "tab": "Split",
        },
        "watershed": {
            "name": "Split touching", "type": "bool", "default": DEFAULTS["watershed"],
            "tab": "Split",
        },
        "min_separation": {
            "name": "Min separation (px)", "type": "int",
            "default": DEFAULTS["min_separation"], "min": 1, "max": 100,
            "tab": "Split",
        },
        "marker_smooth": {
            "name": "Marker smoothing", "type": "float",
            "default": DEFAULTS["marker_smooth"], "min": 0.0, "max": 10.0,
            "step": 0.1, "tab": "Split",
        },
        "clear_border": {
            "name": "Drop edge particles", "type": "bool",
            "default": DEFAULTS["clear_border"], "tab": "Split",
        },
        "brush": {
            "name": "Brush (px)", "type": "float", "default": DEFAULTS["brush"],
            "min": 1.0, "max": 64.0, "step": 1.0, "tab": "Scribble",
        },
        "store_masks": {
            "name": "Store outlines", "type": "bool",
            "default": DEFAULTS["store_masks"], "tab": "Run",
        },
        "track": {
            "name": "Link tracks", "type": "bool", "default": DEFAULTS["track"],
            "tab": "Run",
        },
        "max_dist": {
            "name": "Link radius (units)", "type": "float",
            "default": DEFAULTS["max_dist"], "min": 0.1, "max": 1000.0,
            "step": 0.1, "tab": "Run",
        },
    }

    def __init__(self, session, tree, src_plot):
        super().__init__(session, tree)
        self.src_plot = src_plot
        self.window_id = getattr(src_plot, "window_id", None)
        self.params: dict[str, Any] = dict(DEFAULTS)
        self.labels = None                 # LabelStore, built on first stroke
        self.classifier = None             # ScribbleClassifier, after seg_train
        #: ``{"frame", "labels", "rows", "contours", "count", "areas"}`` for the
        #: frame the caret is showing — what ``commit()`` snapshots.
        self.preview: dict[str, Any] | None = None
        self.result_tree = None

    # ── the signal this wizard segments ──────────────────────────────────────

    def signal(self):
        """The DISPLAYED node, not the root — segmenting a rebinned or cropped
        view must segment what the user is looking at."""
        return _current_signal(self.src_plot) or self.tree.root

    def frames(self):
        """``(n_frames, get_frame, (h, w))`` — one frame at a time, never the
        stack (CLAUDE.md memory safety)."""
        return frames_of(self.signal())

    def scale_units(self) -> tuple[float, str]:
        try:
            ax = self.signal().axes_manager.signal_axes[0]
            return float(ax.scale), str(ax.units or "px")
        except Exception as exc:
            log.debug("[seg] reading signal calibration failed: %s", exc)
            return 1.0, "px"

    def frame_index(self) -> int:
        """Where the navigator is sitting.

        Read from the navigation SELECTOR, not the Plot: ``current_indices``
        lives on the selector, and looking for it on the Plot silently returns
        None — the bug that made "fit spectrum" fit the navigation mean
        (``fit_action.current_indices``).
        """
        npm = getattr(self.tree, "navigator_plot_manager", None)
        if npm is None:
            return 0
        for sels in (getattr(npm, "navigation_selectors", {}) or {}).values():
            for sel in sels:
                idx = getattr(sel, "current_indices", None)
                if idx is None:
                    continue
                try:
                    return int(np.atleast_1d(np.asarray(idx)).ravel()[0])
                except Exception as exc:
                    log.debug("[seg] reading navigator index failed: %s", exc)
        return 0

    # ── scribbles ────────────────────────────────────────────────────────────

    def label_store(self):
        """The accumulating :class:`LabelStore`, built on first use.

        Built lazily because it needs the frame shape, and a flat index means
        nothing without one — a store made against the wrong shape scatters
        every stroke across the image.
        """
        if self.labels is None:
            from spyde.particles import LabelStore
            _n, _get, shape = self.frames()
            self.labels = LabelStore(frame_shape=shape)
        return self.labels

    def class_report(self) -> list[dict[str, Any]]:
        """Per-class labelled-pixel counts for the caret's class list.

        Not decoration (plan B3): under-training a class is *the* failure mode
        and these counts are how a user notices, so a class with zero pixels is
        present in the list rather than absent from it.
        """
        if self.labels is None:
            from spyde.particles import default_classes
            return [dict(c.to_dict(), pixels=0) for c in default_classes()]
        counts = self.labels.counts()
        return [dict(c.to_dict(), pixels=int(counts.get(c.id, 0)))
                for c in self.labels.classes]

    # ── lifecycle ────────────────────────────────────────────────────────────

    def remove(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.set_overlay(None)
        if getattr(self.tree, "_seg_wizard", None) is self:
            self.tree._seg_wizard = None

    def set_overlay(self, labels) -> None:
        """Show (or clear) the previewed instances as a translucent mask on the
        source plot. Client-side compositing — no image re-push."""
        plot = self.src_plot
        if plot is None or not hasattr(plot, "set_overlay_mask"):
            return
        try:
            plot.set_overlay_mask(None if labels is None else (labels > 0),
                                  color="#a6e3a1", alpha=0.35)
        except Exception as exc:
            log.debug("[seg] overlay push failed: %s", exc)

    def commit(self):
        """Snapshot the PREVIEWED frame as a committed label-image tree.

        ``seg_run`` is the whole-movie door and goes through
        ``open_particle_tree`` progressively; Commit is the other half of the
        §0.8 contract — you tuned on one frame, and this keeps that frame's
        result as a dataset without paying for the movie. It is also the whole
        of the single-2-D-image shape in plan §0.10, where there is no movie to
        run over.

        It goes through ``commit_result_tree`` rather than
        ``open_particle_tree`` because a ONE-frame particle tree is not
        constructible: ``open_particle_tree`` builds a ``(1, h, w)`` label
        movie, hyperspy reads the leading axis as a size-1 navigation axis, and
        ``MultiplotManager`` has no selector for one — it raises. A single frame
        is a 2-D label IMAGE, which is exactly what ``commit_result_tree``
        expects, and the store rides along in ``attrs`` so
        ``requires_particles`` unlocks the same downstream actions.
        """
        prev = self.preview
        if prev is None or self.session is None:
            emit_error("Segment Particles: nothing previewed to commit")
            return None
        return commit_single_frame(
            self.session, self, prev["labels"], prev["rows"], prev["contours"],
            int(prev["frame"]))


# ── parameters ───────────────────────────────────────────────────────────────

def _coerce(payload: dict | None) -> dict:
    """Payload → a complete, valid parameter dict.

    Every out-of-range value is corrected rather than raised on: these arrive
    from a slider mid-drag, and a caret that errors while you are moving it is
    unusable. The corrections that change what the user asked for
    (``min_size``, ``local_size``) are echoed back in every preview so the
    number on screen is the number that ran.
    """
    from spyde.particles import THRESHOLD_METHODS

    p = dict(DEFAULTS)
    payload = payload or {}
    for k, default in DEFAULTS.items():
        v = payload.get(k)
        if v is None or v == "":
            continue
        try:
            p[k] = bool(v) if isinstance(default, bool) else type(default)(v)
        except (TypeError, ValueError) as exc:
            log.debug("[seg] param %r=%r not coercible, keeping default: %s",
                      k, v, exc)

    p["method"] = str(p["method"]).lower()
    if p["method"] not in METHODS:
        p["method"] = DEFAULTS["method"]
    p["threshold"] = str(p["threshold"]).lower()
    if p["threshold"] not in THRESHOLD_METHODS:
        p["threshold"] = DEFAULTS["threshold"]
    p["sensitivity"] = float(min(1.0, max(0.0, p["sensitivity"])))

    # skimage's local thresholds require an odd window and SegmentParams raises
    # rather than bumping it silently. Bump here (the caret gets the effective
    # value back) so a slider that lands on an even number doesn't error.
    p["local_size"] = max(3, int(p["local_size"]))
    if p["local_size"] % 2 == 0:
        p["local_size"] += 1

    p["min_size"] = int(p["min_size"])
    p["min_size_floored"] = p["min_size"] < MIN_SIZE_FLOOR
    if p["min_size_floored"]:
        p["min_size"] = MIN_SIZE_FLOOR
    return p


def _segment_params(p: dict):
    from spyde.particles import SegmentParams
    return SegmentParams(
        threshold=p["threshold"], sensitivity=p["sensitivity"],
        rb_kernel=int(p["rb_kernel"]), gaussian=float(p["gaussian"]),
        invert=bool(p["invert"]), local_size=int(p["local_size"]),
        watershed=bool(p["watershed"]), min_separation=int(p["min_separation"]),
        marker_smooth=float(p["marker_smooth"]), min_size=int(p["min_size"]),
        max_size=int(p["max_size"]), clear_border=bool(p["clear_border"]),
    )


def frames_of(signal):
    """``(n_frames, get_frame, (h, w))`` for a movie **or** a single image.

    Delegates to :func:`spyde.drift.frame_source`, which is the shared streaming
    accessor for exactly this — it exists so callers cannot reach ``.data`` and
    accidentally compute the stack. It requires a 1-D navigation axis, so the
    plain-2-D-image case (plan §0.10) is wrapped here as a one-frame stack
    rather than duplicating the accessor.
    """
    am = signal.axes_manager
    if int(am.signal_dimension) != 2:
        raise TypeError(
            "Segment Particles needs 2-D image frames; got signal_dimension="
            f"{int(am.signal_dimension)}")
    nav = int(am.navigation_dimension)
    if nav > 1:
        # frame_source raises for this too, but with a message about drift.
        raise TypeError(
            "needs a movie (1-D time navigation) or a single image; got "
            f"navigation_dimension={nav}. Reduce a 4D-STEM scan to a virtual "
            "image first — plan §0.10.")
    if nav == 0:
        data = signal.data

        def get_frame(i: int, _d=data) -> np.ndarray:
            arr = _d.compute() if hasattr(_d, "compute") else _d
            return np.asarray(arr)

        h, w = int(signal.data.shape[-2]), int(signal.data.shape[-1])
        return 1, get_frame, (h, w)

    from spyde.drift import frame_source
    return frame_source(signal)


# ── the three engines, behind one call ───────────────────────────────────────

def _engine(wiz: SegmentWizard, p: dict) -> Callable[[np.ndarray], np.ndarray] | None:
    """A ``frame → int32 labels`` callable for the selected engine, or None when
    the engine cannot run yet (the caller has already been told why)."""
    sp = _segment_params(p)
    method = p["method"]

    if method == "classical":
        from spyde.particles import segment_frame
        return lambda frame: segment_frame(frame, sp)

    if method == "scribble":
        clf = wiz.classifier
        if clf is None or not clf.is_trained:
            emit_status("Segment Particles: paint a few scribbles — including at "
                        "least one FAINT particle — then press Train.")
            return None
        return lambda frame: clf.segment(frame, sp)

    # plan B4: EfficientSAM-Ti through the existing model registry. The tab
    # exists so the caret can render it; the engine does not.
    emit_status("Segment Particles: prompt segmentation is not installed yet — "
                "use the Classical or Scribble tab.")
    return None


# ── controller resolution ────────────────────────────────────────────────────

def _wizard(session, plot) -> SegmentWizard | None:
    _src, tree = _src_plot_tree(session, plot)
    wiz = getattr(tree, "_seg_wizard", None) if tree is not None else None
    return wiz if (wiz is not None and not wiz._closed) else None


def _emit(wiz: SegmentWizard, msg: dict) -> None:
    msg.setdefault("window_id", wiz.window_id)
    emit(msg)


def _emit_state(wiz: SegmentWizard) -> None:
    """The caret's authoritative state: engine, classes + pixel counts, which
    frames carry labels, and the EFFECTIVE parameters."""
    try:
        n_frames, _get, shape = wiz.frames()
    except Exception:
        n_frames, shape = 1, (0, 0)
    _emit(wiz, {
        "type": "seg_state",
        "method": wiz.params["method"],
        "frame": wiz.frame_index(),
        "n_frames": int(n_frames),
        "frame_shape": [int(shape[0]), int(shape[1])],
        "classes": wiz.class_report(),
        "labelled_frames": (wiz.labels.labelled_frames() if wiz.labels else []),
        "trained": bool(wiz.classifier is not None and wiz.classifier.is_trained),
        "params": {k: v for k, v in wiz.params.items()},
    })


# ── preview (the CURRENT frame, and only the current frame) ──────────────────

#: Pixel budget for ONE preview segmentation. Above this the preview runs on a
#: centred CROP at full resolution instead of the whole frame.
#:
#: Measured, one frame, classical path (segment + measure):
#:
#:   ===========  =========
#:   frame         cost
#:   ===========  =========
#:   256^2          0.27 s
#:   1024^2         0.57 s
#:   2048^2         1.69 s
#:   **4096^2**   **8.36 s**
#:   ===========  =========
#:
#: ``segment_frame`` is 7.6 s of that 8.36 s — watershed over 16.7 M pixels. The
#: caret re-previews on every sensitivity nudge, so on a real 4k in-situ movie the
#: whole caret reads as hung. It was not hung; it was doing 8 seconds of work per
#: keystroke.
#:
#: **Crop, do NOT downsample.** Downsampling would be cheaper still, but it makes
#: the preview a DIFFERENT computation from the run it is supposed to predict:
#: plan §0.9 records that the fine feature scales are what find small faint
#: particles, and a preview that silently detects a different population is worse
#: than a slow one. A crop at full resolution runs the identical algorithm on
#: identical pixels, so what it shows is exactly what the run will do there.
#:
#: 1024^2 keeps a preview near half a second — inside the interaction budget with
#: room for a slower machine.
_PREVIEW_PIXEL_BUDGET = 1024 * 1024


def _preview_window(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """``(frame_or_crop, box)`` — bound one preview's cost by AREA, not by scale.

    Returns the frame untouched (and ``box=None``) when it already fits the
    budget, which is the common case for a tutorial-sized movie and means the
    fast path pays nothing for this. Otherwise a centred crop of the same aspect
    ratio, at full resolution, plus its ``(y0, x0, h, w)`` so the caller can tell
    the user what it measured.
    """
    h, w = frame.shape[:2]
    if h * w <= _PREVIEW_PIXEL_BUDGET:
        return frame, None
    shrink = (h * w / _PREVIEW_PIXEL_BUDGET) ** 0.5
    ch = max(64, int(h / shrink))
    cw = max(64, int(w / shrink))
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    return frame[y0:y0 + ch, x0:x0 + cw], (y0, x0, ch, cw)


def _preview(wiz: SegmentWizard, gen: int) -> None:
    """Segment the displayed frame on a worker and paint the result.

    Generation-guarded at BOTH ends: a superseded tune must neither paint nor
    leave a stale overlay behind (the caret can be closed mid-compute).
    """
    p = dict(wiz.params)
    engine = _engine(wiz, p)
    if engine is None:
        return
    t = wiz.frame_index()
    scale, units = wiz.scale_units()

    def _work():
        _n, get_frame, _shape = wiz.frames()
        full = np.asarray(get_frame(t))
        frame, box = _preview_window(full)
        t0 = time.perf_counter()
        labels = engine(frame)
        from spyde.particles import measure_frame
        rows, contours = measure_frame(labels, frame, t=t, scale=scale)
        return {"frame": t, "labels": labels, "rows": rows,
                "contours": contours, "elapsed": time.perf_counter() - t0,
                "box": box, "full_shape": full.shape}

    def _done(res):
        if not wiz.still(gen) or wiz._closed:
            return
        rows = res["rows"]
        from spyde.signals.particles import COL
        areas = (rows[:, COL["area"]] if len(rows)
                 else np.zeros(0, np.float32))
        wiz.preview = {"frame": res["frame"], "labels": res["labels"],
                       "rows": rows, "contours": res["contours"],
                       "count": int(len(rows)), "areas": areas,
                       "box": res.get("box")}
        # An overlay of a CROP cannot be painted onto the full frame as-is — it
        # would land in the corner instead of over the region it describes. Place
        # it back at its offset, leaving the rest unlabelled (which is honest:
        # nothing was segmented out there).
        labels = res["labels"]
        box = res.get("box")
        if box is not None:
            y0, x0, ch, cw = box
            placed = np.zeros(res["full_shape"][:2], labels.dtype)
            placed[y0:y0 + ch, x0:x0 + cw] = labels
            labels = placed
        wiz.set_overlay(labels)
        # The count reported is the count AFTER the size filter (plan §0.9b) —
        # `split_instances` applies min_size last, so `rows` is already filtered
        # and this number is the one the histogram below describes.
        _emit(wiz, {
            "type": "seg_preview",
            "frame": int(res["frame"]),
            "count": int(len(rows)),
            "areas": [float(a) for a in areas[:_MAX_AREAS_SENT]],
            "median_area": (float(np.median(areas)) if areas.size else 0.0),
            "units": units,
            "method": p["method"],
            "min_size": int(p["min_size"]),
            "min_size_floored": bool(p["min_size_floored"]),
            "elapsed_ms": round(1000.0 * res["elapsed"], 1),
            # Present only when the frame was too big to preview whole. The caret
            # must say so: otherwise "12 particles on this frame" is a lie about a
            # 4096² frame where only the middle megapixel was looked at.
            "preview_box": (None if res.get("box") is None
                            else [int(v) for v in res["box"]]),
        })
        if p["min_size_floored"]:
            emit_status(
                f"Segment Particles: min size raised to {MIN_SIZE_FLOOR} px — "
                "at 0 the split returns background speckle as particles "
                "(measured: 33 instances where 9 are real).")

    def _fail(exc):
        if wiz.still(gen):
            emit_error(f"Segment Particles preview failed: {exc}")

    run_on_worker(wiz.session, _work, name="seg-preview",
                  on_done=_done, on_error=_fail)


# ── staged handlers ──────────────────────────────────────────────────────────

def seg_open(session, plot, payload) -> None:
    """Caret mounted: build the controller and preview the displayed frame."""
    src, tree = _src_plot_tree(session, plot)
    if src is None or tree is None:
        emit_error("Segment Particles: no active dataset")
        return
    try:
        frames_of(_current_signal(src) or tree.root)
    except TypeError as exc:
        emit_error(f"Segment Particles: {exc}")
        return

    existing = getattr(tree, "_seg_wizard", None)
    if existing is not None and not existing._closed:
        # Idempotent re-open: adopt the new parameters and re-preview rather
        # than building a second controller with its own scribbles.
        existing.params = _coerce(payload)
        gen = existing.guard()
        _emit_state(existing)
        _preview(existing, gen)
        return

    wiz = SegmentWizard(session, tree, src)
    wiz.params = _coerce(payload)
    # BEFORE anything deferred: React StrictMode fires open/close/open
    # synchronously, so the close's bump must be able to invalidate this open.
    gen = wiz.guard()
    tree._seg_wizard = wiz
    _emit_state(wiz)
    _preview(wiz, gen)


def seg_close(session, plot, payload=None) -> None:
    """Caret unmounted: invalidate in-flight work FIRST, then tear down."""
    _src, tree = _src_plot_tree(session, plot)
    if tree is None:
        return
    # This IS WizardController.cancel_inflight (same `_seg_run_gen` key), done
    # on the TREE so it fires even when there is no controller yet: a StrictMode
    # open whose worker has not landed must still be cancelled.
    bump_generation(tree, "_seg_run_gen")
    wiz = getattr(tree, "_seg_wizard", None)
    if wiz is not None:
        _detach_brush(wiz)
        wiz.remove()


def seg_set_method(session, plot, payload) -> None:
    """Switch engine (classical | scribble | prompt) and re-preview."""
    wiz = _wizard(session, plot)
    if wiz is None:
        return
    method = str((payload or {}).get("method", "")).lower()
    if method not in METHODS:
        emit_error(f"Segment Particles: unknown engine {method!r}")
        return
    wiz.params["method"] = method
    # The brush exists only while Scribble is the engine: it floats over the
    # image, so leaving it armed on Classical would put a paint cursor over data
    # with nothing to paint into.
    if method == "scribble":
        _arm_brush(wiz)
    else:
        _detach_brush(wiz)
    gen = wiz.guard()
    _emit_state(wiz)
    _preview(wiz, gen)


def seg_tune(session, plot, payload) -> None:
    """Debounced parameter change → re-preview the CURRENT frame only."""
    wiz = _wizard(session, plot)
    if wiz is None:
        return
    wiz.params = _coerce({**wiz.params, **(payload or {})})
    # Paint state must reach the WIDGET, not just the params dict — the widget is
    # what tags a stroke, so a class change that stops here paints the old colour.
    _sync_brush(wiz)
    gen = wiz.guard()
    _preview(wiz, gen)


def seg_paint(session, plot, payload) -> None:
    """One brush stroke into the :class:`LabelStore`.

    Payload: ``{frame, points: [[y, x], …], class_id, erase, brush}``. Points
    arrive in **image pixels** with no scale or offset applied (plan trap 6 —
    anyplotlib 2-D widgets report pixels), so nothing is converted here.

    Synchronous: a stroke is a few thousand indices and the caret's class counts
    must be correct by the time the user lifts the brush.
    """
    wiz = _wizard(session, plot)
    if wiz is None:
        return
    payload = payload or {}
    points = payload.get("points") or []
    if not len(points):
        return
    if _paint_stroke(wiz, int(payload.get("frame", wiz.frame_index())), points,
                     int(payload.get("class_id", 0)),
                     bool(payload.get("erase")),
                     float(payload.get("brush", wiz.params["brush"]))):
        _emit_state(wiz)


def _paint_stroke(wiz, t: int, points, class_id: int, erase: bool,
                  brush: float) -> int:
    """Rasterise ONE stroke into the label store. Returns pixels changed.

    Shared by the on-plot brush (:func:`_on_stroke`) and the renderer's
    ``seg_paint``, deliberately: two rasterisers would let the brush and the
    eraser — or the widget and a scripted stroke — disagree about which pixels a
    given path covers, and that divergence is invisible until a trained model
    behaves oddly.

    *points* are ``(y, x)`` in **image pixels**, no scale or offset applied
    (anyplotlib 2-D widgets report pixels).
    """
    store = wiz.label_store()
    before = int(sum(store.counts().values())) if hasattr(store, "counts") else -1
    try:
        if erase:
            # No `erase_stroke` on LabelStore, and the eraser must cover exactly
            # what the brush would paint. Rasterise the stroke into a scratch
            # store with the same geometry and erase by the indices it produced,
            # so the two can never drift apart.
            from spyde.particles import LabelStore, ScribbleClass
            scratch = LabelStore(frame_shape=store.frame_shape,
                                 classes=[ScribbleClass(0, "scratch")])
            scratch.paint_stroke(0, points, 0, brush=brush)
            store.erase(t, scratch.at(0)[0])
        else:
            store.paint_stroke(t, points, int(class_id), brush=brush)
    except (KeyError, ValueError) as exc:
        emit_error(f"Segment Particles: {exc}")
        return 0
    after = int(sum(store.counts().values())) if hasattr(store, "counts") else -1
    # -1 when the store cannot report counts: assume it painted rather than
    # swallowing a stroke the user definitely made.
    return max(1, abs(after - before)) if before >= 0 else 1


# ── the on-plot brush ────────────────────────────────────────────────────────
#
# The scribble engine needs strokes, and the ONLY thing that can produce them is
# an anyplotlib brush widget living on the signal plot. Two things were wrong
# before this existed, and both made painting impossible rather than merely
# awkward:
#
#   1. Nothing ever called ``add_brush_widget``, so there was no brush on the
#      plot at all — Shift+drag had nothing to hit.
#   2. The caret listened for a RENDERER-side ``spyde:figure_event`` carrying a
#      points array. A brush stroke does not travel that way: anyplotlib emits it
#      to PYTHON (``event_json`` → ``Figure._dispatch_event`` →
#      ``Widget._update_from_js`` → ``plot.callbacks.fire``), which the renderer
#      never sees. So even with a brush present, no stroke could have arrived.
#
# The widget is therefore created, owned and read HERE. The renderer's job shrinks
# to telling us which class is active and how fat the brush is; ``seg_paint``
# survives only as the programmatic/test door.
_BRUSH_COLORS = ("#f9a03f", "#89b4fa", "#585b70", "#f38ba8", "#a6e3a1", "#cba6f7")


def _brush_supported() -> bool:
    """Whether the installed anyplotlib has the brush widget.

    It landed in 0.5.0 (CSSFrancis/anyplotlib#47); SpyDE's floor is still 0.4.2,
    so a user on PyPI's 0.4.2 has no brush and must be TOLD that rather than left
    dragging at an image that never responds.
    """
    try:
        from anyplotlib.plot2d._plot2d import Plot2D
        return hasattr(Plot2D, "add_brush_widget")
    except Exception:
        return False


def _attach_brush(wiz) -> bool:
    """Put a brush on the source plot and wire its strokes into the label store.

    Idempotent: an existing brush is re-shown and re-armed rather than duplicated,
    so switching tabs or re-tuning never stacks two brushes on one plot.

    Returns True when a brush is live.
    """
    if not _brush_supported():
        return False
    src = wiz.src_plot
    plot2d = getattr(src, "_plot2d", None) if src is not None else None
    if plot2d is None:
        return False

    existing = getattr(wiz.tree, "_seg_brush", None)
    if existing is not None:
        try:
            existing.set(active=True, visible=True)
            return True
        except Exception as exc:
            log.debug("[seg] re-arming the existing brush failed: %s", exc)

    try:
        classes = wiz.class_report()
        brush = plot2d.add_brush_widget(
            radius=float(wiz.params.get("brush", 6.0)),
            colors=[c.get('colour', '#f9a03f') for c in classes] or list(_BRUSH_COLORS),
            class_id=int(wiz.params.get("active_class", 0)),
            alpha=0.55,
            active=True,
        )
    except Exception as exc:
        log.debug("[seg] add_brush_widget failed: %s", exc)
        return False

    from spyde.drawing.selectors.base_selector import event_handler_fn

    # pointer_up ONLY. The brush deliberately emits once per finished stroke
    # rather than per pointer_move — a growing stroke re-serialised every frame is
    # quadratic over one drag (see anyplotlib's BrushWidget docs) — so there is
    # nothing to listen for mid-stroke and listening would fire on other widgets.
    handler = event_handler_fn(lambda event: _on_stroke(wiz, event))
    try:
        brush.add_event_handler(handler, "pointer_up")
    except Exception as exc:
        log.debug("[seg] wiring the brush handler failed: %s", exc)
        return False

    wiz.tree._seg_brush = brush
    wiz.tree._seg_brush_handler = handler      # weak callbacks: keep a hard ref
    wiz._brush_seen = 0
    return True


def _sync_brush(wiz) -> None:
    """Push the caret's paint state onto the live brush widget.

    The widget tags each stroke with its OWN ``class_id`` at paint time in JS, so
    a class change that only reaches ``wiz.params`` paints the previous colour
    forever — which is precisely what "I can only scribble one colour" was.
    Likewise ``erase``: the eraser is a widget mode, not something the handler can
    decide after the fact, because the stroke has already been tagged.

    No-op when there is no brush (Classical, or an anyplotlib without one).
    """
    brush = getattr(wiz.tree, "_seg_brush", None)
    if brush is None:
        return
    try:
        brush.set(
            class_id=int(wiz.params.get("active_class", 0)),
            radius=float(wiz.params.get("brush", 3.0)),
            erase=bool(wiz.params.get("erase", False)),
        )
    except Exception as exc:
        log.debug("[seg] syncing brush state failed: %s", exc)


def _arm_brush(wiz) -> None:
    """Attach the brush and TELL THE USER how to use it.

    The instruction is not optional. A brush that arms silently on Shift+drag is
    undiscoverable — there is no cursor change to find it by and no affordance on
    the image, so the honest outcome is a user dragging at a picture that never
    responds. That is exactly what happened on first use.
    """
    if _attach_brush(wiz):
        emit_status("Shift+drag on the image to paint labels · plain drag still "
                    "pans · pick the class and brush size on the strip beside "
                    "the plot")
        _emit(wiz, {"type": "seg_brush", "available": True,
                    "hint": "Shift+drag to paint"})
        return
    # No brush: say WHY, with the actionable part. Silence here reads as a bug.
    if not _brush_supported():
        import anyplotlib as apl
        emit_error(
            f"Painting needs anyplotlib 0.5.0 or newer for the brush widget; "
            f"this environment has {getattr(apl, '__version__', 'unknown')}. "
            "Until then, use the Classical engine, or install the brush build.")
    else:
        emit_error("Painting could not attach a brush to this plot.")
    _emit(wiz, {"type": "seg_brush", "available": False,
                "hint": "brush unavailable — see the log"})


def _detach_brush(wiz) -> None:
    """Remove the brush. Called from ``seg_close`` and on leaving Scribble.

    Best-effort and idempotent — teardown must never raise, or closing the caret
    leaves the wizard half-torn-down.
    """
    brush = getattr(wiz.tree, "_seg_brush", None)
    if brush is not None:
        for attempt in ("remove", "hide"):
            fn = getattr(brush, attempt, None)
            if fn is None:
                continue
            try:
                fn()
                break
            except Exception as exc:
                log.debug("[seg] brush %s() failed: %s", attempt, exc)
    for attr in ("_seg_brush", "_seg_brush_handler"):
        if hasattr(wiz.tree, attr):
            try:
                setattr(wiz.tree, attr, None)
            except Exception as exc:                      # pragma: no cover
                log.debug("[seg] clearing %s failed: %s", attr, exc)


def _on_stroke(wiz, event) -> None:
    """A finished brush stroke → label pixels → re-preview.

    Reads the widget rather than the event payload. ``Widget._update_from_js``
    has already merged the JS fields into ``_data`` by the time a plot-level
    handler runs, so ``brush.strokes`` is authoritative and the event's own copy
    is redundant.

    Only strokes we have not consumed are applied: the widget accumulates them
    for as long as it lives, so replaying the whole list on every stroke would
    re-paint everything and make the pixel counts grow quadratically.
    """
    if wiz._closed:
        return
    brush = getattr(wiz.tree, "_seg_brush", None)
    if brush is None:
        return
    try:
        strokes = list(getattr(brush, "strokes", ()) or ())
        classes = list(getattr(brush, "stroke_classes", ()) or ())
    except Exception as exc:
        log.debug("[seg] reading brush strokes failed: %s", exc)
        return

    seen = int(getattr(wiz, "_brush_seen", 0))
    fresh = strokes[seen:]
    if not fresh:
        return
    wiz._brush_seen = len(strokes)

    erase = bool(wiz.params.get("erase", False))
    radius = float(getattr(brush, "radius", wiz.params.get("brush", 6.0)) or 6.0)
    default_cls = int(wiz.params.get("active_class", 0))

    painted = 0
    for i, stroke in enumerate(fresh, start=seen):
        cls = int(classes[i]) if i < len(classes) else default_cls
        # anyplotlib gives [[x, y], ...] in IMAGE PIXELS; LabelStore works in
        # (y, x). Swapping these silently mirrors every scribble about the
        # diagonal, which on a non-square frame also puts half of them outside
        # the image — see spyde/actions/masks.py for this bug class.
        pts = [[float(p[1]), float(p[0])] for p in (stroke or ()) if len(p) >= 2]
        if pts:
            painted += _paint_stroke(wiz, wiz.frame_index(), pts, cls,
                                    erase, radius)

    if painted:
        _emit_state(wiz)
        _preview(wiz, wiz.guard())


def seg_train(session, plot, payload) -> None:
    """Fit the scribble classifier on every accumulated label.

    Trains on labelled PIXELS only (thousands, not millions) — plan B3's hard
    interaction budget. The store is snapshotted before it crosses onto the
    worker: ``seg_paint`` runs on the main thread and a stroke landing mid-fit
    would mutate the arrays the fit is iterating.
    """
    wiz = _wizard(session, plot)
    if wiz is None:
        emit_error("Segment Particles: the caret is not open")
        return
    if wiz.labels is None or len(wiz.labels) == 0:
        emit_error("Segment Particles: nothing painted yet — scribble on a "
                   "particle and on the background first.")
        return

    from spyde.particles import LabelStore
    snapshot = LabelStore.from_dict(wiz.labels.to_dict())
    device = (payload or {}).get("device")
    gen = wiz.guard()
    emit_status("Segment Particles: training…")

    def _work():
        from spyde.particles import ScribbleClassifier
        _n, get_frame, _shape = wiz.frames()
        clf = ScribbleClassifier(device=device)
        report = clf.fit(snapshot, get_frame,
                         progress=lambda d, n: emit_progress(d, n, "Training"))
        return clf, report

    def _done(res):
        clf, report = res
        if not wiz.still(gen) or wiz._closed:
            return
        wiz.classifier = clf
        wiz.params["method"] = "scribble"
        _emit(wiz, {"type": "seg_trained", "report": report})
        emit_status(
            f"Segment Particles: trained on {report['n_pixels']} px across "
            f"{report['n_classes']} classes "
            f"(accuracy {report['train_accuracy']:.3f})")
        _emit_state(wiz)
        _preview(wiz, gen)

    def _fail(exc):
        emit_error(f"Segment Particles: training failed — {exc}")

    run_on_worker(session, _work, name="seg-train", on_done=_done, on_error=_fail)


# ── the single-frame result ──────────────────────────────────────────────────

def commit_single_frame(session, wiz: SegmentWizard, labels, rows, contours,
                        frame: int):
    """Commit one frame's segmentation as a label-image tree with the store
    attached. Shared by ``seg_commit`` and by ``seg_run`` on a 2-D image.

    See :meth:`SegmentWizard.commit` for why this door and not
    ``open_particle_tree``.
    """
    from spyde.actions.commit import commit_result_tree
    from spyde.signals.particles import COL, SpyDEParticles

    scale, units = wiz.scale_units()
    _n, _get, shape = wiz.frames()
    rows = np.asarray(rows, np.float32)
    if len(rows):
        # The store is one frame long, so every row's `t` must be 0 or the CSR
        # offsets disagree with the rows about which frame they live in.
        rows = rows.copy()
        rows[:, COL["t"]] = 0.0
    params = dict(wiz.params, frame=int(frame), mode="single_frame")
    parts = SpyDEParticles.from_frames(
        [rows], frame_shape=shape,
        contours_per_frame=([list(contours)] if wiz.params["store_masks"] else None),
        scale=scale, units=units, params=params,
        provenance={"action": "segment_particles", "params": dict(wiz.params)},
    )
    tree = commit_result_tree(
        session, title=f"Particles — frame {int(frame)} ({len(rows)})",
        primary=np.asarray(labels, np.float32), primary_label="labels",
        levels=None, cmap="gray",
        attrs={"particles": parts, "source_node": wiz.signal(),
               "source_tree": wiz.tree, "particle_events": [],
               "nav_map": np.zeros(1, np.int64)},
        provenance={"action": "segment_particles", "params": params,
                    "frame": int(frame)},
    )
    _rebuild_toolbars(tree)
    emit_status(f"Committed {len(rows)} particles from frame {int(frame)}")
    return tree


# ── the batch run ────────────────────────────────────────────────────────────

def seg_run(session, plot, payload) -> None:
    """Segment every frame on a worker: progressive, cancellable.

    The result window opens IMMEDIATELY with an empty particle store, its count
    trace fills as frames complete, and ``tree.particles`` attaches only at
    ``_finalize`` — see the module docstring for why that ordering is the whole
    point of the attach gap.
    """
    src, tree = _src_plot_tree(session, plot)
    if src is None or tree is None:
        emit_error("Segment Particles: no active dataset")
        return
    wiz = _wizard(session, plot)
    if wiz is None:
        wiz = SegmentWizard(session, tree, src)
        tree._seg_wizard = wiz
    wiz.params = _coerce({**wiz.params, **(payload or {})})
    p = dict(wiz.params)

    engine = _engine(wiz, p)
    if engine is None:
        return
    try:
        n_frames, get_frame, shape = wiz.frames()
    except TypeError as exc:
        emit_error(f"Segment Particles: {exc}")
        return
    scale, units = wiz.scale_units()
    source = wiz.signal()

    if n_frames < 2:
        # Plan §0.10's single-image shape: there is no movie to fill
        # progressively, and a 1-frame particle tree is not constructible (see
        # SegmentWizard.commit). Segment it and commit the label image.
        _run_single_frame(session, wiz, get_frame, engine, scale)
        return

    from spyde.actions.particle_tree import open_particle_tree
    from spyde.signals.particles import N_COLUMNS, SpyDEParticles

    empty = [np.zeros((0, N_COLUMNS), np.float32)] * n_frames
    placeholder = SpyDEParticles.from_frames(
        empty, frame_shape=shape, scale=scale, units=units, params=dict(p))
    # attach=False: `requires_particles` must not unlock the particle toolbar
    # against an empty store, so the tree publishes `particles` only at
    # _finalize. The placeholder is still handed in because the lazy label movie
    # renders from THAT object — see open_particle_tree's `attach` docs — which
    # is why _finalize mutates it in place rather than swapping in a new store.
    result = open_particle_tree(
        session, particles=placeholder, source_node=source, source_tree=tree,
        params=dict(p), title=f"Particles — {n_frames} frames", attach=False)
    wiz.result_tree = result

    result._seg_batch_running = True
    tree._seg_batch_running = True

    stopped = [False]
    for t_ in {id(tree): tree, id(result): result}.values():
        if hasattr(t_, "register_cancel"):
            t_.register_cancel(flag=stopped)

    emit_status(f"Segmenting {n_frames} frames…")
    gen = bump_generation(result, "_seg_batch_gen")

    counts = np.zeros(n_frames, np.float32)
    last_paint = [0.0]

    def _paint_counts():
        if not is_current(result, "_seg_batch_gen", gen):
            return
        _paint_count_trace(result, counts.copy())

    def _work():
        per_frame: list[np.ndarray] = []
        contours: list[list[np.ndarray]] = []
        from spyde.particles import measure_frame
        done = 0
        for t_i in range(n_frames):
            if stopped[0]:
                break
            frame = np.asarray(get_frame(t_i))
            labels = engine(frame)
            rows, cs = measure_frame(labels, frame, t=t_i, scale=scale)
            per_frame.append(rows)
            contours.append(cs)
            counts[t_i] = float(len(rows))
            done = t_i + 1
            now = time.monotonic()
            if now - last_paint[0] >= _PROGRESS_INTERVAL:
                last_paint[0] = now
                emit_progress(done, n_frames, "Segmenting")
                _to_main(session, _paint_counts)
        # Frames never reached keep an empty block, so the CSR store always
        # spans the movie and a cancelled run reads as "no particles after
        # frame N" rather than a shorter movie.
        while len(per_frame) < n_frames:
            per_frame.append(np.zeros((0, N_COLUMNS), np.float32))
            contours.append([])
        return per_frame, contours, done

    def _done(res):
        per_frame, contours, done = res
        try:
            if getattr(result, "_spyde_closed", False):
                return                      # window torn down mid-run
            _finalize(session, result, placeholder, per_frame, contours,
                      p, scale, units, done, n_frames, cancelled=stopped[0])
        finally:
            _teardown_batch(tree, result, stopped)

    def _fail(exc):
        emit_error(f"Segment Particles failed: {exc}")
        log.exception("segmentation batch failed")
        _teardown_batch(tree, result, stopped)

    run_on_worker(session, _work, name="seg-batch", on_done=_done, on_error=_fail)


def _run_single_frame(session, wiz, get_frame, engine, scale) -> None:
    """Segment the only frame there is, on a worker, then commit it."""
    def _work():
        from spyde.particles import measure_frame
        frame = np.asarray(get_frame(0))
        labels = engine(frame)
        rows, contours = measure_frame(labels, frame, t=0, scale=scale)
        return labels, rows, contours

    def _done(res):
        labels, rows, contours = res
        if wiz._closed:
            return
        commit_single_frame(session, wiz, labels, rows, contours, 0)

    run_on_worker(session, _work, name="seg-single",
                  on_done=_done,
                  on_error=lambda e: emit_error(f"Segment Particles failed: {e}"))


def _teardown_batch(tree, result, stopped) -> None:
    result._seg_batch_running = False
    tree._seg_batch_running = False
    for t_ in {id(tree): tree, id(result): result}.values():
        if hasattr(t_, "unregister_cancel"):
            try:
                t_.unregister_cancel(flag=stopped)
            except Exception as exc:
                log.debug("[seg] unregister_cancel failed: %s", exc)


def _finalize(session, result, placeholder, per_frame, contours, p,
              scale, units, done, n_frames, *, cancelled: bool) -> None:
    """Attach the finished store, repaint, and unlock the particle actions."""
    from spyde.actions.particle_tree import _navigator_traces
    from spyde.signals.particles import SpyDEParticles

    params = dict(p)
    if cancelled:
        # Recorded, not just narrated: a partial result that only says so in a
        # transient status line is indistinguishable from a complete one later.
        params["cancelled_after_frame"] = int(done)
    final = SpyDEParticles.from_frames(
        per_frame, frame_shape=placeholder.frame_shape,
        contours_per_frame=(contours if p["store_masks"] else None),
        scale=scale, units=units, params=params,
        provenance={"action": "segment_particles", "params": dict(p)},
    )

    events = []
    if p["track"] and final.n_particles:
        try:
            from spyde.particles import link
            res = link(final, max_dist=float(p["max_dist"]), apply=True)
            events = list(res.events)
        except Exception as exc:
            log.debug("[seg] linking failed, keeping untracked particles: %s", exc)

    _adopt(placeholder, final)
    result.particles = placeholder
    result.particle_events = events
    result.nav_traces = _navigator_traces(placeholder, events or None)

    # The signal plot's CachedDaskArray captured the placeholder's zeros when
    # the window first rendered; drop it so the label movie re-slices the real
    # contours (the same stale-cache fix find_vectors_action._finalize makes).
    try:
        result.root.cached_dask_array = None
        result.root._clear_cache_dask_data()
    except Exception as exc:
        log.debug("[seg] clearing stale cached dask array failed: %s", exc)

    _paint_count_trace(result, placeholder.count_series())
    _repaint_label_movie(result)
    _rebuild_toolbars(result)

    # TERMINAL progress. Without it `state.loading.busy` never clears in the
    # renderer: the StatusBar spinner spins forever and — because it prefers
    # `loading.text` while busy (StatusBar.tsx) — "Segmenting (33%)" permanently
    # masks the "Found N particles" line emitted immediately below. Found by
    # driving the real UI; no headless test could see it.
    emit_progress(n_frames, n_frames, "Segmenting")

    n = placeholder.n_particles
    if cancelled:
        emit_status(f"Segmentation cancelled — found {n} particles in the "
                    f"first {done} of {n_frames} frames")
    else:
        emit_status(f"Found {n} particles in {n_frames} frames")


def _repaint_label_movie(result) -> None:
    """Push the finalized frame to the label-movie window.

    Clearing the stale cached dask array (above) makes the NEXT read correct, but
    nothing triggers a read — so the window keeps showing the placeholder's zeros
    and the result looks empty until the user happens to scrub. Re-read the
    currently-displayed frame and paint it.

    Best-effort: a failure here costs a stale frame until the next scrub, which is
    exactly the state we were in before, so it must never take the finalize down
    with it.
    """
    from spyde.actions.lifecycle import paint_signal_plots

    try:
        particles = result.particles
        if particles is None or particles.n_frames == 0:
            return
        t = 0
        for plot in getattr(result, "plots", None) or ():
            idx = getattr(plot, "current_indices", None)
            if idx:
                t = int(np.atleast_1d(idx)[0])
                break
        t = max(0, min(t, particles.n_frames - 1))
        paint_signal_plots(result, particles.render_frame(t, value="track"))
    except Exception as exc:
        log.debug("[seg] repainting the label movie after finalize failed: %s", exc)


def _adopt(placeholder, final) -> None:
    """Move *final*'s arrays onto *placeholder*, in place.

    The lazy label movie built by ``open_particle_tree`` closes over the
    placeholder OBJECT and calls ``render_frame`` at graph-execution time, so
    swapping ``tree.particles`` for a new store would leave the movie rendering
    zeros forever. Mutating is what makes the early window fill in.
    """
    placeholder.flat_buffer = final.flat_buffer
    placeholder.t_offsets = final.t_offsets
    placeholder.contours = final.contours
    placeholder.contour_offsets = final.contour_offsets
    placeholder.params = final.params
    placeholder.provenance = final.provenance


def _to_main(session, fn) -> None:
    """Run *fn* on the asyncio main thread; inline when there is no loop (bare
    handler tests) — the same fallback ``lifecycle.run_on_worker`` makes."""
    dispatch = getattr(session, "_dispatch_to_main", None)
    if dispatch is None:
        fn()
        return
    dispatch(fn)


def _nav_plots(tree) -> list:
    npm = getattr(tree, "navigator_plot_manager", None)
    if npm is None:
        return []
    out = []
    for pw in list(npm.plot_windows.keys()):
        out.extend(npm.plots.get(pw, []))
    return out


def _paint_count_trace(tree, counts: np.ndarray) -> None:
    """Paint particle-count-vs-time onto the result tree's 1-D navigator.

    Shape-matched rather than "the first navigator": a tree can carry more than
    one navigator plot, and painting an (n_frames,) trace onto the wrong one
    leaves it blank — the bug find_vectors_action._spatial_nav_plot exists for.
    """
    want = int(counts.shape[0])
    for nav in _nav_plots(tree):
        cur = getattr(nav, "current_data", None)
        if cur is None or getattr(cur, "ndim", 0) != 1 or int(cur.shape[0]) != want:
            continue
        try:
            nav.needs_auto_level = True
            nav.set_data(np.asarray(counts, np.float32))
        except Exception as exc:
            log.debug("[seg] painting count trace failed: %s", exc)


def _rebuild_toolbars(tree) -> None:
    """Re-send the toolbar config so ``requires_particles`` actions appear.

    This is the moment the gate flips — without it the buttons stay hidden until
    something else happens to rebuild the toolbar, and the e2e specs wait on
    exactly this appearing.
    """
    for sp in list(getattr(tree, "signal_plots", []) or []):
        try:
            state = getattr(sp, "plot_state", None)
            if state is not None and hasattr(state, "_send_toolbar_config"):
                state._send_toolbar_config()
        except Exception as exc:
            log.debug("[seg] re-sending toolbar config failed: %s", exc)


def seg_commit(session, plot, payload=None) -> None:
    """Commit the previewed frame as a one-frame particle tree."""
    wiz = _wizard(session, plot)
    if wiz is None:
        emit_error("Segment Particles: nothing to commit")
        return
    wiz.commit()


def segment_particles(ctx, action_name: str = "Segment Particles", **params):
    """Toolbar entry — a no-op parent; the Electron toolbar opens the staged
    caret, which drives the ``seg_*`` handlers (README §4)."""
    return None
