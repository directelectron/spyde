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
from spyde.actions.lifecycle import (
    bump_generation, is_current, run_on_worker, window_computing,
)
from spyde.actions.particle_overlay import _navigator_selectors_for, _push_groups
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
    # THE control for "too many particles". Filters the measured instances by
    # their confidence score (spyde.particles.measure.particle_scores), which is
    # computed WITH the measurement — so this re-filters an existing result and
    # never re-segments, and a drag is instant. It also means the same thing on
    # every engine, because it acts on the OUTPUT rather than on any one
    # method's parameters. 0 keeps everything (the old behaviour).
    min_score=0.0,
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
        #: The live preview outline group, and the navigator hooks that keep it
        #: on the displayed frame. See :meth:`set_overlay` / :meth:`wire_navigator`.
        self._ov_group = None
        self._ov_box_group = None
        self._ov_selectors: list = []
        #: Latest frame the navigator asked for, and whether a preview for it is
        #: already running — the latest-wins pair, see :func:`_preview_for_nav`.
        self._nav_frame: int | None = None
        self._nav_busy = False

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
        self._unwire_navigator()
        self._drop_overlay()
        self._ov_cleared = False
        self._ov_box_state = None
        if getattr(self.tree, "_seg_wizard", None) is self:
            self.tree._seg_wizard = None

    def set_overlay(self, contours=None, box=None, labels=None,
                    full_shape=None) -> None:
        """Draw (or clear) the previewed instances as OUTLINES on the source plot.

        Outlines and not a translucent raster mask, for two independent reasons —
        the first is a bug, the second is why the bug could not just be patched:

        * **A raster mask does not survive GPU tile mode.** A signal frame at or
          above 1024 px is handed to anyplotlib's tiled display, whose base image
          is drawn by WebGPU; ``set_overlay_mask`` composites onto the Canvas2D
          context underneath it and is simply not visible. That is the whole
          "106 particles and no overlay" report — the mask WAS being pushed
          (``[plot] overlay mask set: N px`` in the log), it just never appeared.
          Markers draw over the GPU base correctly, which is why the brush strokes
          were visible in the same screenshot that had no overlay.
        * **A full-resolution mask cannot follow the navigator.** At 4096² the
          mask is 16.7 M px — ~16 MB down the PLOTAPP line protocol *per frame*.
          This overlay re-draws on every navigator move, so the payload has to be
          proportional to the number of PARTICLES, not to the number of pixels.
          The same 106 particles are ~100 kB of polygon, and they stay crisp when
          the user zooms in, which a mask rasterised at frame resolution does not.

        *contours* are ``(k, 2)`` arrays of ``(y, x)`` **crop** pixels as
        :func:`spyde.particles.measure_frame` returns them; *box* is the
        ``(y0, x0, h, w)`` preview window they were measured in, or None when the
        whole frame was segmented. The offset is applied here rather than by the
        caller so that whatever ``_preview`` decided to crop stays in one place.
        """
        plot2d = getattr(self.src_plot, "_plot2d", None)
        if plot2d is None:
            return
        group = self._overlay_group(plot2d)
        if group is None:
            return
        # Hundreds of polygons is what makes the whole app sluggish — each is a
        # path the renderer re-transforms every pan/zoom frame. Above the
        # threshold draw ONE mask instead; the particle COUNT and every
        # measurement are unaffected, only how they are drawn.
        n = len(contours) if contours is not None else 0
        if n > _RASTER_ABOVE and labels is not None:
            if self._set_raster_overlay(labels, box, full_shape):
                self._ov_raster = True
                # Drop the vector outlines so the two do not double up.
                updates = {group: {"vertices_list": []}}
                frame_group = self._window_group(plot2d)
                if frame_group is not None:
                    updates[frame_group] = {"vertices_list": _box_poly(box)}
                try:
                    _push_groups(plot2d, updates)
                    self._ov_cleared = False
                    self._ov_box_state = None
                except Exception as exc:
                    log.debug("[seg] overlay push failed: %s", exc)
                return
        self._clear_raster_overlay()
        polys = _contour_polys(contours, box)
        updates = {group: {"vertices_list": polys}}
        # The preview window's own outline. Without it "only the middle of my
        # 4k frame has any segmentation" reads as a BUG rather than as the
        # documented 1-megapixel budget: the caret says "preview window
        # 1024x1024 px" in small print, but it cannot say WHERE, and on a 4096²
        # frame the window is 1/16 of the area sitting in the middle of an
        # otherwise untouched image. Drawing the boundary makes the empty region
        # obviously "not looked at yet" instead of "looked at and found
        # nothing".
        frame_group = self._window_group(plot2d)
        if frame_group is not None:
            updates[frame_group] = {"vertices_list": _box_poly(box)}
        try:
            # `vertices_list` is the polygon group's own key — the same one
            # `particle_overlay._payload` writes. A key the group does not know
            # is accepted silently by `_push_groups` and simply never drawn.
            _push_groups(plot2d, updates)
            # Real outlines are up now, so the no-engine path must redraw rather
            # than skip on its cached "already cleared" state.
            self._ov_cleared = False
            self._ov_box_state = None
        except Exception as exc:
            log.debug("[seg] overlay push failed: %s", exc)

    def _set_raster_overlay(self, labels, box, full_shape) -> bool:
        """Draw the instances as ONE mask instead of N polygons. True if drawn.

        Uses ``Plot2D.set_overlay_mask``, which composites client-side onto the
        transparent 2-D canvas that sits ABOVE the WebGPU canvas — so this works
        on a GPU-rendered base, which a marker-layer raster would not.

        THE TILING TRAP. When a large frame is in tile mode the renderer's mask
        check is ``bytes.length === iw * ih`` where ``iw = base_width ||
        image_width`` — i.e. the OVERVIEW size, not the native frame. A
        native-resolution mask therefore fails that check and is dropped
        SILENTLY: no error, no overlay, and nothing in the log to say why. So
        the mask is built at the frame size and then reduced to the overview
        grid whenever ``base_width`` is set.

        The reduction is a block ANY, not a subsample: particles here are often
        a few pixels across, and a strided sample of a 4096² mask at 1024²
        drops three quarters of them at random.
        """
        plot2d = getattr(self.src_plot, "_plot2d", None)
        if plot2d is None or not hasattr(plot2d, "set_overlay_mask"):
            return False
        try:
            lab = np.asarray(labels)
            if lab.ndim != 2 or not lab.any():
                return False
            fh, fw = (int(full_shape[0]), int(full_shape[1])) if full_shape \
                else lab.shape
            mask = np.zeros((fh, fw), bool)
            if box is not None:
                y0, x0, h, w = (int(v) for v in box)
                mask[y0:y0 + h, x0:x0 + w] = lab[:h, :w] > 0
            else:
                mask[:lab.shape[0], :lab.shape[1]] = lab > 0

            state = getattr(plot2d, "_state", {}) or {}
            bw, bh = int(state.get("base_width") or 0), int(state.get("base_height") or 0)
            if bw > 0 and bh > 0 and (bw, bh) != (fw, fh):
                ys = max(1, fh // bh)
                xs = max(1, fw // bw)
                # Block ANY via a reshape-reduce, then pad/crop to exactly the
                # overview grid — the renderer's length check is exact.
                cut = mask[:(fh // ys) * ys, :(fw // xs) * xs]
                small = cut.reshape(fh // ys, ys, fw // xs, xs).any(axis=(1, 3))
                out = np.zeros((bh, bw), bool)
                sh, sw = min(bh, small.shape[0]), min(bw, small.shape[1])
                out[:sh, :sw] = small[:sh, :sw]
                mask = out
            plot2d.set_overlay_mask(mask, color=_PREVIEW_COLOR, alpha=_RASTER_ALPHA)
            return True
        except Exception as exc:
            log.debug("[seg] raster overlay failed (%s); using outlines", exc)
            return False

    def _clear_raster_overlay(self) -> None:
        plot2d = getattr(self.src_plot, "_plot2d", None)
        if plot2d is None or not hasattr(plot2d, "set_overlay_mask"):
            return
        if not getattr(self, "_ov_raster", False):
            return
        try:
            plot2d.set_overlay_mask(None)
        except Exception as exc:
            log.debug("[seg] clearing the raster overlay failed: %s", exc)
        self._ov_raster = False

    def show_preview_window(self) -> None:
        """Draw the preview-window box ALONE, with no instance outlines.

        For the states that have no engine to run: an untrained Scribble, or
        Prompt before anything is prompted. Callers that DO have a result go
        through :meth:`set_overlay`, which draws both.

        Clearing the outlines here is the other half of the fix — switching to
        an untrained engine must not leave the previous one's instances on
        screen looking like the new engine's answer.
        """
        plot2d = getattr(self.src_plot, "_plot2d", None)
        if plot2d is None:
            return
        try:
            _n, get_frame, _shape = self.frames()
            _frame, box = _preview_window(np.asarray(get_frame(self.frame_index())))
        except Exception as exc:
            log.debug("[seg] preview-window box: no frame to size it from: %s", exc)
            return
        # Only push when something actually CHANGES. `seg_tune` fires on every
        # slider tick and lands here whenever there is no engine, and
        # `_push_groups` falls back to `MarkerGroup.set`, which re-serialises
        # the panel — so an unconditional push would put a full serialisation
        # on every tick of a drag to redraw a rectangle that did not move.
        state = (tuple(box) if box is not None else None,
                 getattr(self, "_ov_cleared", False))
        if state == getattr(self, "_ov_box_state", None):
            return
        updates = {}
        group = self._overlay_group(plot2d)
        if group is not None and not getattr(self, "_ov_cleared", False):
            updates[group] = {"vertices_list": []}          # no instances yet
        frame_group = self._window_group(plot2d)
        if frame_group is not None:
            updates[frame_group] = {"vertices_list": _box_poly(box)}
        if not updates:
            return
        try:
            _push_groups(plot2d, updates)
            self._ov_cleared = True
            self._ov_box_state = (tuple(box) if box is not None else None, True)
        except Exception as exc:
            log.debug("[seg] preview-window push failed: %s", exc)

    def _overlay_group(self, plot2d):
        """The lazily-created polygon group the preview outlines live in."""
        if self._ov_group is not None:
            return self._ov_group
        try:
            self._ov_group = plot2d.add_polygons(
                [], name="seg_preview_outline",
                facecolors=_PREVIEW_COLOR, edgecolors=_PREVIEW_COLOR,
                linewidths=_PREVIEW_WIDTH, alpha=_PREVIEW_ALPHA,
                transform="data")
        except Exception as exc:
            log.debug("[seg] creating the preview overlay group failed: %s", exc)
        return self._ov_group

    def _window_group(self, plot2d):
        """The lazily-created outline of the PREVIEW WINDOW (empty when whole)."""
        if self._ov_box_group is not None:
            return self._ov_box_group
        try:
            self._ov_box_group = plot2d.add_polygons(
                [], name="seg_preview_window",
                facecolors=None, edgecolors=_PREVIEW_WINDOW_COLOR,
                linewidths=1.0, transform="data")
        except Exception as exc:
            log.debug("[seg] creating the preview-window outline failed: %s", exc)
        return self._ov_box_group

    def _drop_overlay(self) -> None:
        self._clear_raster_overlay()
        for attr in ("_ov_group", "_ov_box_group"):
            group = getattr(self, attr, None)
            if group is None:
                continue
            try:
                group.remove()
            except Exception as exc:
                log.debug("[seg] removing %s failed: %s", attr, exc)
            setattr(self, attr, None)

    # ── following the navigator ──────────────────────────────────────────────

    def wire_navigator(self) -> None:
        """Re-preview when the navigator moves, so the outlines follow the frame.

        Nothing subscribed to the navigator before this: ``frame_index()`` only
        READ the selector when something else asked for a preview, so scrolling
        through a movie left the outlines describing whichever frame was showing
        when you last touched the caret.
        """
        self._ov_selectors = _navigator_selectors_for(self.tree, self.src_plot)
        for sel in self._ov_selectors:
            if self._on_indices not in sel.index_hooks:
                sel.index_hooks.append(self._on_indices)
        if not self._ov_selectors:
            log.debug("[seg] no navigator selectors — the preview will not "
                      "follow the frame (a single image has none, which is fine)")

    def _unwire_navigator(self) -> None:
        for sel in self._ov_selectors:
            if self._on_indices in sel.index_hooks:
                sel.index_hooks.remove(self._on_indices)
        self._ov_selectors = []

    def _on_indices(self, indices) -> None:
        """Navigation moved. **Runs on the ``_NavDispatcher`` thread.**

        Does no figure work here and submits no compute here — both are the main
        thread's business (CLAUDE.md's threading contract), so this only records
        the frame and marshals.
        """
        if self._closed:
            return
        try:
            frame = int(np.asarray(indices).ravel()[0])
        except Exception:
            return
        if frame == self._nav_frame:
            return
        self._nav_frame = frame
        dispatch = getattr(self.session, "_dispatch_to_main", None)
        if dispatch is None:
            _preview_for_nav(self)
            return
        dispatch(lambda: _preview_for_nav(self))

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
    p["min_score"] = float(min(0.99, max(0.0, p.get("min_score", 0.0) or 0.0)))
    p["min_size_floored"] = p["min_size"] < MIN_SIZE_FLOOR
    if p["min_size_floored"]:
        p["min_size"] = MIN_SIZE_FLOOR
    return p


def _segment_kwargs(p: dict) -> dict:
    """The ``SegmentParams`` fields, as a plain dict.

    A dict and not the dataclass because this crosses to dask workers inside
    :class:`~spyde.particles.batch.EngineSpec`, and a spec that pickles without
    dragging the segmentation modules onto the client's import path is one less
    thing to go wrong in a worker with a different import order.
    """
    return dict(
        threshold=p["threshold"], sensitivity=p["sensitivity"],
        rb_kernel=int(p["rb_kernel"]), gaussian=float(p["gaussian"]),
        invert=bool(p["invert"]), local_size=int(p["local_size"]),
        watershed=bool(p["watershed"]), min_separation=int(p["min_separation"]),
        marker_smooth=float(p["marker_smooth"]), min_size=int(p["min_size"]),
        max_size=int(p["max_size"]), clear_border=bool(p["clear_border"]),
    )


def _segment_params(p: dict):
    from spyde.particles import SegmentParams
    return SegmentParams(**_segment_kwargs(p))


def _movie_array(signal):
    """The ``(n, h, w)`` array behind *signal*, or None when there isn't one.

    None is a legitimate answer (a frame source that is a callable or a
    sequence), and :func:`~spyde.particles.batch.segment_movie` falls back to
    the streaming accessor for it. Never touches the data itself — reading
    ``.data`` on a lazy signal hands back the dask graph, not the movie.
    """
    data = getattr(signal, "data", None)
    return data if getattr(data, "ndim", 0) == 3 else None


#: How long the batch waits for the cluster before running locally. The cluster
#: is built on a background thread, so a run fired seconds after a load can
#: arrive first; falling straight through would silently cost the whole fan-out.
_CLIENT_WAIT_S = 30.0


def _batch_client(session, stopped=None):
    """The distributed client for the batch, waiting briefly for it to come up.

    Mirrors ``_do_compute_vectors``: we are already on a worker thread, so
    blocking here doesn't freeze the UI, and the alternative — silently taking
    the local thread-pool path — is exactly the "why is this slow" report this
    work exists to fix. Returns None under ``SPYDE_NO_DASK=1`` (the migrated-test
    mode, where the manager exists but never starts).
    """
    import os
    if os.environ.get("SPYDE_NO_DASK") == "1":
        return None
    dm = getattr(session, "dask_manager", None)
    if dm is None:
        return None
    client = getattr(dm, "client", None)
    if client is not None:
        return client
    from spyde.compute_dispatch import reliable_sleep
    deadline = time.monotonic() + _CLIENT_WAIT_S
    while client is None and time.monotonic() < deadline:
        if stopped is not None and stopped[0]:
            return None
        reliable_sleep(0.1)
        client = getattr(dm, "client", None)
    if client is None:
        log.warning("[seg] no Dask client after %.0f s — segmenting locally",
                    _CLIENT_WAIT_S)
    return client


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


#: The live preview outline: a SATURATED green, filled at the same 25% the
#: committed result overlay uses (``particle_overlay.FILL_ALPHA``) so tuning and
#: result look like the same thing.
#:
#: Green because the particle brush class paints ORANGE and both are on screen
#: at once while you scribble — "what I told it" and "what it found" have to be
#: separable at a glance. Saturated rather than the retired mask's pale sage
#: (``#a6e3a1``): that was chosen for a 35%-alpha area fill, and as a 1 px
#: outline on grey EM data it is very close to invisible.
_PREVIEW_COLOR = "#40e070"
_PREVIEW_ALPHA = 0.25
_PREVIEW_WIDTH = 1.0

#: The preview WINDOW's outline. Deliberately a neutral light grey rather than
#: any of the data colours: it is UI chrome saying "this is the region that was
#: looked at", and must not be mistaken for a found particle (green) or for a
#: painted class (orange / blue / grey / pink).
_PREVIEW_WINDOW_COLOR = "#cdd6f4"

#: Above this many instances the overlay switches from one polygon per particle
#: to a SINGLE raster mask. Measured symptom: several hundred vector contours
#: make the whole app sluggish, because every one is a path the renderer
#: re-transforms on each pan/zoom frame. A mask is one image however many
#: particles it contains.
#:
#: Below the threshold the vector outlines stay, and they are worth keeping:
#: they are crisp at any zoom and each is a real object the UI can hover.
_RASTER_ABOVE = 100

#: The raster overlay's colour/opacity. Deliberately the same green as the
#: vector outlines so crossing the threshold does not look like a mode change.
_RASTER_ALPHA = 0.45


def _box_poly(box) -> list:
    """The preview window as a one-polygon list, or empty when there is none.

    ``box`` is ``(y0, x0, h, w)``; markers want ``(x, y)``. An absent box means
    the WHOLE frame was segmented, and then the outline must be cleared rather
    than left showing the previous frame's window — the caret can switch between
    cropped and whole (a small frame, or a changed budget) without closing.
    """
    if box is None:
        return []
    y0, x0, h, w = (float(v) for v in box)
    return [np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]],
                     dtype=np.float32)]


def _contour_polys(contours, box) -> list:
    """``measure_frame`` contours → marker polygons, offset back onto the frame.

    Two conversions, both of which are silent-wrong-picture bugs if skipped:

    * **(y, x) → (x, y).** Contours are stored row-major like the array they came
      from; marker offsets are ``(x, y)``. Getting this wrong transposes every
      outline about the diagonal, which on a square frame still *looks* like
      plausible particles (``particle_overlay.contour_xy`` is the same swap).
    * **The crop offset.** A preview of a big frame is measured on a 1024² CROP
      (``_preview_window``), so its contours start at ``(0, 0)`` of that window.
      Drawn unshifted they pile up in the frame's corner instead of over the
      region they describe — the same trap the mask path documented.
    """
    if contours is None:
        return []
    dy, dx = (float(box[0]), float(box[1])) if box is not None else (0.0, 0.0)
    polys = []
    for c in contours:
        arr = np.asarray(c, dtype=np.float32)
        if arr.ndim != 2 or len(arr) < 3:
            continue                       # a polygon needs three vertices
        polys.append(np.column_stack([arr[:, 1] + dx, arr[:, 0] + dy])
                     .astype(np.float32))
    return polys


def _preview_for_nav(wiz: SegmentWizard) -> None:
    """Re-preview because the navigator moved. LATEST FRAME WINS.

    A 4096² preview is ~600 ms and a scroll emits an index change per step, so
    firing one preview per step would queue dozens of them and paint the frames
    out of order as they landed. Only ONE is ever in flight; a frame requested
    while it runs replaces any other waiting frame and is fired when it lands.

    This is Live-Display §2's latest-wins coalescing, not a queue and not a
    self-pacing gate: ``_nav_busy`` is cleared by BOTH the success and the
    failure path of the preview it guards, so a failing frame cannot wedge it.
    """
    if wiz._closed or wiz._nav_busy:
        return                            # the in-flight one will re-fire
    wiz._nav_busy = True
    # `current_gen()`, NOT `guard()`. Scrolling is not a new interaction, and
    # bumping the generation here cancels whatever the user actually started —
    # a navigator move during `seg_train` made the trained classifier land on a
    # stale generation and get dropped, so Train silently never finished.
    _preview(wiz, wiz.current_gen())


def _chase_nav(wiz: SegmentWizard) -> None:
    """After a preview lands: if the navigator has moved on, go again.

    Called only once the new ``wiz.preview`` is in place, because the comparison
    is "is what we are now showing the frame that was last asked for" — run
    before the assignment it would read the PREVIOUS frame and re-fire forever.
    Releasing the gate is deliberately NOT done here (see the call sites): it has
    to happen even when the result is superseded or failed, and this function
    returns early in both cases.
    """
    if wiz._closed or wiz._nav_frame is None:
        return
    if (wiz.preview or {}).get("frame") == wiz._nav_frame:
        return                # what just landed IS the newest frame — nothing to chase
    _preview_for_nav(wiz)


def filter_by_score(rows, contours, min_score: float):
    """Keep only instances scoring at or above *min_score*.

    Separate from the measurement on purpose: this is what the caret's single
    "Confidence" slider calls, and it must be able to run WITHOUT re-segmenting
    — the score already lives in the row (see
    :func:`spyde.particles.measure.particle_scores`), so a drag is a numpy mask
    over a few hundred rows rather than a fresh segmentation of a 1-megapixel
    window.

    ``min_score <= 0`` returns the inputs untouched, so the default costs
    nothing and behaves exactly as before this control existed.
    """
    import numpy as _np
    from spyde.signals.particles import COL as _COL

    if min_score is None or float(min_score) <= 0.0 or len(rows) == 0:
        return rows, contours
    keep = _np.asarray(rows[:, _COL["score"]] >= float(min_score), bool)
    if keep.all():
        return rows, contours
    kept_rows = _np.ascontiguousarray(rows[keep])
    kept_contours = ([c for c, k in zip(contours, keep) if k]
                     if contours is not None else contours)
    return kept_rows, kept_contours


def _preview(wiz: SegmentWizard, gen: int) -> None:
    """Segment the displayed frame on a worker and paint the result.

    Generation-guarded at BOTH ends: a superseded tune must neither paint nor
    leave a stale overlay behind (the caret can be closed mid-compute).
    """
    p = dict(wiz.params)
    engine = _engine(wiz, p)
    if engine is None:
        # No engine yet — an untrained Scribble, or Prompt before any prompt.
        # Still show WHERE the preview window is, and clear any outlines the
        # previous engine left.
        #
        # The window box used to be drawn only as a side effect of a successful
        # segmentation, which made it lie in three ways: it appeared only under
        # Classical (the one engine that always has a solver), it SURVIVED a
        # switch to an untrained Scribble because this early return painted
        # nothing and cleared nothing, and after toggling the caret off and on
        # it never came back. The box documents where the 1-megapixel budget
        # looks; that is true whenever the caret is open and has nothing to do
        # with whether an engine is trained.
        wiz.show_preview_window()
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
        n_all = len(rows)
        rows, contours = filter_by_score(rows, contours, p.get("min_score", 0.0))
        return {"frame": t, "labels": labels, "rows": rows, "n_all": n_all,
                "contours": contours, "elapsed": time.perf_counter() - t0,
                "box": box, "full_shape": full.shape}

    def _done(res):
        # Release the navigator gate BEFORE the generation guard: a superseded or
        # closed preview still has to hand the gate back, or the next navigator
        # move finds `_nav_busy` set forever and the outlines stop following the
        # frame. The retired self-pacing gates in Live-Display §2 wedged exactly
        # like this.
        wiz._nav_busy = False
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
        # The outlines carry the crop offset themselves (`_contour_polys`), so
        # unlike the retired mask path nothing has to be re-rasterised onto a
        # full-frame array first — at 4096² that array alone was 16 MB per frame.
        wiz.set_overlay(res["contours"], res.get("box"),
                        labels=res.get("labels"),
                        full_shape=res.get("full_shape"))
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
        _chase_nav(wiz)

    def _fail(exc):
        wiz._nav_busy = False              # never wedge the gate — see `_done`
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
    # Follow the navigator from the moment the caret opens, not from the first
    # Train: scrolling with the caret up should keep the outlines on the frame
    # you are looking at whichever engine is selected.
    wiz.wire_navigator()
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
    state = (int(wiz.params.get("active_class", 0)),
             float(wiz.params.get("brush", 3.0)),
             bool(wiz.params.get("erase", False)))
    if state == getattr(wiz, "_brush_state", None):
        return
    wiz._brush_state = state
    try:
        brush.set(class_id=state[0], radius=state[1], erase=state[2])
    except Exception as exc:
        log.debug("[seg] syncing brush state failed: %s", exc)
        return
    _force_widget_push(wiz)


def _force_widget_push(wiz) -> None:
    """Make the brush's new state AUTHORITATIVE in the panel JSON.

    ``Widget.set`` reaches JS through ``Figure._push_widget``, which writes
    ``event_json`` **only** — deliberately, because re-serialising a whole panel
    per drag frame is the cost that path exists to avoid. The documented
    consequence is that ``panel_<id>_json`` keeps stale widget state between
    plot-level pushes, and the brush's *drawing* reads exactly that: JS takes
    ``w.class_id`` from ``p.state.overlay_widgets`` at stroke start
    (``figure_esm.js::_brushLiveBegin``) and paints in ``colors[class_id]``.

    So a class switch that only goes through ``set()`` leaves every stroke
    painting in the PREVIOUS class's colour — reported twice, as "I can only
    scribble one colour" and then "support film still doesn't change the
    painting colour". The strokes were landing in the right class the whole
    time (the caret's per-class counts prove it); only the colour was stale,
    which makes it look like nothing switched.

    A full ``_push`` is the fix, and it is gated on the brush state having
    actually CHANGED (``_sync_brush``'s ``_brush_state`` compare) because
    ``seg_tune`` also fires for every sensitivity-slider tick — re-serialising a
    4096² panel, image bytes and all, on each one would trade a colour bug for a
    much worse drag. Painting itself still goes through the cheap targeted path
    untouched; this costs one push per class/eraser/size click.
    """
    plot2d = getattr(wiz.src_plot, "_plot2d", None)
    push = getattr(plot2d, "_push", None)
    if push is None:
        return
    try:
        push()
    except Exception as exc:
        log.debug("[seg] forcing the panel push failed: %s", exc)


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
    except Exception as exc:
        log.debug("[seg] reading brush strokes failed: %s", exc)
        return

    seen = int(getattr(wiz, "_brush_seen", 0))
    fresh = strokes[seen:]
    if not fresh:
        return
    wiz._brush_seen = len(strokes)

    # PYTHON is the authority for all three, and the class one is load-bearing.
    #
    # `stroke_classes` is what the JS widget tagged the stroke with, and reading
    # it made class switching silently not work: `Figure._push_widget` sends a
    # targeted update that never writes `panel_<id>_json`, so a Python-side
    # `brush.class_id = 1` does not reliably reach the widget before the next
    # stroke — anyplotlib's own brush test has to press Shift BEFORE pushing the
    # class to dodge exactly this.
    #
    # The natural experiment that proved it: `erase` read from `wiz.params` and
    # WORKED, `class` read from `stroke_classes` and did not, in the same handler
    # on the same stroke. So the strip → seg_tune → params path is sound; the
    # Python → JS widget push is what is unreliable.
    #
    # Nothing is lost by preferring params: a stroke cannot change class midway,
    # so the active class when it completes IS its class. The widget push stays
    # (see `_sync_brush`) purely so the stroke DRAWS in the right colour while
    # the user paints — cosmetic, and no longer load-bearing.
    erase = bool(wiz.params.get("erase", False))
    radius = float(getattr(brush, "radius", wiz.params.get("brush", 6.0)) or 6.0)
    active_cls = int(wiz.params.get("active_class", 0))

    painted = 0
    for i, stroke in enumerate(fresh, start=seen):
        cls = active_cls
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
        # Say which split route the training just selected. A painted boundary
        # class is what lets `split_instances` skip the distance transform and
        # the watershed (1.78 s -> 0.33 s at 4096²), and the user is the one who
        # decides it by painting — so it has to be visible that they did.
        route = ("boundary class painted — touching particles split by "
                 "connected components, no watershed"
                 if report.get("has_boundary") else
                 "no boundary painted — touching particles split by watershed")
        emit_status(
            f"Segment Particles: trained on {report['n_pixels']} px across "
            f"{report['n_classes']} classes "
            f"(accuracy {report['train_accuracy']:.3f}); {route}")
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

    # RAISE THE "Calculating…" OVERLAY NOW — synchronously, the statement after
    # the window exists and BEFORE any of the setup below.
    #
    # It used to be raised nowhere at all, and the natural place to add it (the
    # worker, next to the first progress emission) is far too late: the window
    # opens, then the placeholder store is built, the cancel flags registered,
    # the generation bumped, the worker scheduled, the thread hop paid, and the
    # first frame COMPUTED — on a 4096² frame that is seconds of a window that
    # looks finished and empty. Every one of those steps is between the window
    # appearing and the user learning it is working, which is the lag reported.
    computing = window_computing(_result_window_id(result))
    computing.start()

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
        # The batch fans out over the cluster (spyde.particles.batch): one dask
        # task per block of frames, dual-lane so the GPU workers run the torch
        # head while the rest run the CPU path in parallel. It used to be a
        # plain serial for-loop on this one thread — 90 minutes for 900 frames
        # of 4096², with 47 cores and 78% of the GPU idle.
        from spyde.particles.batch import (EngineSpec, drop_engine_model,
                                           save_engine_model, segment_movie)
        done = [0]
        # Frames finished so far, indexed by frame so an out-of-order block
        # lands in the right slot; the unfinished ones stay empty, which renders
        # as "nothing found there yet" rather than as a shorter movie.
        live_rows = [np.zeros((0, N_COLUMNS), np.float32)] * n_frames
        live_contours: list = [[] for _ in range(n_frames)]
        model_path = None
        if p["method"] == "scribble":
            # The trained head crosses to the workers as a FILE, never as a
            # pickled CUDA tensor — see EngineSpec.
            model_path = save_engine_model(wiz.classifier)
        spec = EngineSpec(method=p["method"], params=_segment_kwargs(p),
                          model_path=model_path)

        def _on_frames(t0, t1, vals):
            """One block landed. Runs on a dask/worker callback thread, so it
            only touches `counts` and marshals the paint (CLAUDE.md threading)."""
            for i, (rows, cs) in enumerate(vals):
                counts[t0 + i] = float(len(rows))
                # Keep the finished frames so the label movie can render them
                # WHILE the rest compute. Blocks land out of order under the
                # fan-out, so index by frame rather than appending.
                live_rows[t0 + i] = rows
                live_contours[t0 + i] = cs
            done[0] += int(t1 - t0)
            now = time.monotonic()
            if now - last_paint[0] >= _PROGRESS_INTERVAL:
                last_paint[0] = now
                emit_progress(done[0], n_frames, "Segmenting")
                _to_main(session, _paint_counts)
                # The navigator's count trace shows that SOMETHING is happening;
                # the signal shows whether it is happening CORRECTLY. That is
                # the difference between abandoning a bad 900-frame run in the
                # first ten seconds and after it finishes. Snapshot the lists —
                # the callback thread keeps writing into them.
                snap = (list(live_rows), list(live_contours), int(t1) - 1)
                _to_main(session, lambda s=snap: _publish_partial(
                    session, result, placeholder, s[0], s[1],
                    p, scale, units, s[2]))

        try:
            # Frames never reached keep an empty block, so the CSR store always
            # spans the movie and a cancelled run reads as "no particles after
            # frame N" rather than a shorter movie — segment_movie guarantees
            # both lists are n_frames long.
            per_frame, contours, n_done = segment_movie(
                _movie_array(source), spec, n_frames=n_frames,
                get_frame=get_frame, scale=scale,
                store_masks=bool(p["store_masks"]),
                client=_batch_client(session, stopped), stopped=stopped,
                on_frames=_on_frames)
        finally:
            drop_engine_model(model_path)
        return per_frame, contours, n_done

    def _done(res):
        per_frame, contours, done = res
        try:
            if getattr(result, "_spyde_closed", False):
                return                      # window torn down mid-run
            _finalize(session, result, placeholder, per_frame, contours,
                      p, scale, units, done, n_frames, cancelled=stopped[0])
        finally:
            _teardown_batch(tree, result, stopped, computing)

    def _fail(exc):
        emit_error(f"Segment Particles failed: {exc}")
        log.exception("segmentation batch failed")
        _teardown_batch(tree, result, stopped, computing)

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


def _teardown_batch(tree, result, stopped, computing=None) -> None:
    # The Calculating chip comes down HERE, the one place every exit path goes
    # through — success, failure and cancellation alike. Clearing it only on the
    # success path is how an overlay ends up spinning forever over a window that
    # stopped working, which is the failure `window_computing`'s pairing
    # contract exists to prevent.
    if computing is not None:
        try:
            computing.stop()
        except Exception as exc:
            log.debug("[seg] clearing the computing overlay failed: %s", exc)
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
        # "{done} of {n_frames}", NOT "the first {done}": under the block
        # fan-out the finished frames are not a contiguous prefix — blocks land
        # out of order, so a cancelled run has holes rather than a clean cut.
        # Saying "first" would misdescribe which frames actually have particles.
        emit_status(f"Segmentation cancelled — found {n} particles in "
                    f"{done} of {n_frames} frames")
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


def _result_window_id(tree):
    """The result tree's SIGNAL window — what the Calculating chip sits on.

    The signal plot and not the navigator: the navigator fills in visibly as the
    count trace grows, so it is self-evidently working; the signal plot is the
    one that sits there looking empty and finished.
    """
    for plot in (getattr(tree, "signal_plots", None) or []):
        wid = getattr(plot, "window_id", None)
        if wid is not None:
            return wid
    return None


def _publish_partial(session, result, placeholder, per_frame, contours,
                     p, scale, units, t_latest: int) -> None:
    """Adopt the frames finished SO FAR so the label movie renders them live.

    Same three steps as :func:`_finalize`, minus the tracking and the attach:
    build a store from what exists, adopt it into the placeholder the lazy label
    movie closed over, and drop the stale cached dask array so the next slice
    re-reads. Without the cache drop the movie keeps serving the zeros it
    captured when the window first rendered, which is exactly the bug
    ``_finalize`` documents.

    Deliberately NOT attached to ``result.particles``: the particle toolbar is
    ``requires_particles``-gated and must not unlock against a store that is
    still filling — that is the attach gap the module docstring is about. This
    only makes the movie SHOW the work.

    Then paint the MOST RECENTLY COMPUTED frame, not whatever frame the
    navigator happens to sit on — "show the newest result" is the whole point,
    and on a 900-frame run the navigator is parked on frame 0 the entire time,
    so painting its frame would show one image for twenty minutes.

    Cheap enough to run on the feedback clock (a few times a second): the rows
    are small float arrays and the rebuild is a concatenate, not a recompute.
    """
    from spyde.actions.lifecycle import paint_signal_plots
    from spyde.signals.particles import SpyDEParticles

    # EVERYTHING here is inside the guard, deliberately. This runs on the
    # asyncio MAIN thread (marshalled from the batch's callback), so an
    # exception does not merely lose one preview frame — it escapes into the
    # event loop. A cosmetic live fill must never be able to damage the run it
    # is describing, and the correct result is still produced by `_finalize`
    # whatever happens here.
    try:
        if getattr(result, "_spyde_closed", False):
            return                      # window torn down mid-run
        partial = SpyDEParticles.from_frames(
            per_frame, frame_shape=placeholder.frame_shape,
            contours_per_frame=(contours if p["store_masks"] else None),
            scale=scale, units=units, params=dict(p))
        _adopt(placeholder, partial)
        try:
            result.root.cached_dask_array = None
            result.root._clear_cache_dask_data()
        except Exception as exc:
            log.debug("[seg] clearing the cache for the live fill failed: %s", exc)
        t = max(0, min(int(t_latest), placeholder.n_frames - 1))
        paint_signal_plots(result, placeholder.render_frame(t, value="track"))
    except Exception as exc:
        log.debug("[seg] live label-movie fill failed: %s", exc)


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
