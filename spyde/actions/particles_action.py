"""
particles_action.py — the Segment Particles wizard (``seg_`` staged actions).

Plan B7, honouring the §0.8 interaction contract literally:

    seg_open        caret mounted → preview the CURRENT frame, emit caret state
    seg_close       caret unmounted → clear the overlay, drop the controller
    seg_set_method  scribble | prompt (one engine today; plan B4 adds the other)
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
``instances.SegmentParams`` refuses for its own fields, and this must not
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

#: The mask sources. ``classical`` (a port of ParticleSpy's threshold pipeline)
#: was REMOVED, not deprecated: measured on the low-contrast in-situ data this
#: feature exists for, a global threshold has no bimodal histogram to find, so
#: otsu lands inside the noise and the split shatters the support film — 4873
#: instances at 39% coverage on one 1024² frame, and no parameter combination
#: recovered the real particles. It also made the caret lie, because otsu on the
#: preview's centred crop (120.0) is not otsu on the whole frame (146.0), so
#: tuning the preview did not control the run. See
#: :mod:`spyde.particles.instances` and ``benchmarks.md``.
#:
#: ``prompt`` is declared so the caret can offer it from the schema before the
#: engine lands (plan B4); the code path that would run it emits a "not
#: installed yet" status instead.
METHODS: tuple[str, ...] = ("scribble", "prompt")

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
    method="scribble",
    # THE control for "too many particles". Filters the measured instances by
    # their confidence score (spyde.particles.measure.particle_scores), which is
    # computed WITH the measurement — so this re-filters an existing result and
    # never re-segments, and a drag is instant. It also means the same thing on
    # every engine, because it acts on the OUTPUT rather than on any one
    # method's parameters. 0 keeps everything (the old behaviour).
    min_score=0.0,
    # THE two face controls, both in NANOMETRES because a distance in the image
    # is something the eye can judge and a 0-1 confidence is not.
    #   merge_nm  — pieces closer than this are ONE particle
    #   min_nm    — smallest particle worth keeping (diameter)
    # Both are converted to pixels with the signal's own scale at dispatch, so
    # they mean the same thing at any magnification. 0 disables either.
    merge_nm=0.0,
    min_nm=0.0,
    min_size=20,
    max_size=0,
    watershed=True,
    min_separation=3,
    marker_smooth=1.0,
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

    def set_params(self, payload: dict | None, *, merge: bool = False) -> None:
        """Coerce *payload* into ``self.params``, WITH the signal's scale.

        The one place params are assigned, because of the second line. The face
        controls `merge_nm` / `min_nm` are physical, and :func:`_nm_to_px`
        converts them with ``p["scale"]`` — but :func:`_coerce` builds `p` from
        the ``DEFAULTS`` keys alone, so nothing put a scale there and the
        conversion took its uncalibrated branch on EVERY signal. The slider said
        "50 nm" and the backend merged at 50 *pixels*: silent, and wrong by
        exactly the magnification. Stashing it at dispatch is what `_nm_to_px`
        already documents ("stashed on the params at dispatch"); it just was
        never done.

        Assigning `wiz.params = _coerce(...)` directly re-opens that hole, so
        don't — the scale is only correct if it is refreshed here, on the
        DISPLAYED node, which is what a rebinned or cropped view changes.

        The stashed value is **nm per pixel**, not the raw axis scale, and it is
        0.0 when the axis is not a real-space length at all. A signal calibrated
        in `nm^-1` reported a perfectly good positive scale, so converting
        against it produced a merge radius that was silently wrong by the camera
        length; 0.0 routes those signals to the pixel fallback instead, and
        `_face_units` relabels the sliders so the caret does not claim nm.
        """
        base = {**self.params, **(payload or {})} if merge else payload
        p = _coerce(base)
        scale, units = self.scale_units()
        p["scale"] = _length_nm_per_px(scale, units)
        self.params = p

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
                    full_shape=None, n_instances=None) -> None:
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
        # The true instance count, which is NOT `len(contours)`: the preview
        # skips outline extraction above the draw cap precisely because this
        # branch is about to discard them. Keying the decision off the contour
        # list would then read "0 instances", fall through to the polygon path,
        # draw nothing, and leave the frame bare in exactly the case that most
        # needs a mask.
        n = n_instances if n_instances is not None else (
            len(contours) if contours is not None else 0)
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
        # HARD CAP, independent of `_RASTER_ABOVE`. Reaching here with a huge n
        # means the raster path was unavailable or failed, and the honest answer
        # is then "too many to draw" — NOT 14028 paths the renderer must
        # re-transform on every pan and zoom frame. That is a hang, and because
        # thousands of translucent fills overlap into one flat sheet it is not
        # even a legible one: the user sees a solid green block and no data.
        # The COUNT still reports every instance; only the drawing is dropped,
        # and the caret says so rather than leaving a silently empty frame.
        if len(polys) > _MAX_OUTLINE_POLYS:
            log.warning("[seg] %d outlines exceeds the %d draw cap and the raster "
                        "overlay was unavailable; drawing none",
                        len(polys), _MAX_OUTLINE_POLYS)
            polys = []
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

    def set_mask_overlay(self, mask, box, full_shape) -> None:
        """Paint a boolean mask, clearing any outlines the full preview left.

        The mask-only live preview's drawing path. It reuses
        :meth:`_set_raster_overlay` unchanged — that method already thresholds
        ``> 0``, so a boolean array is a valid ``labels`` argument, and reusing it
        means the tiled-frame contract (hand anyplotlib a NATIVE-resolution mask,
        let IT reduce) has exactly one implementation rather than a second one to
        get wrong the same way.

        The outlines must be cleared explicitly: switching from a full preview to
        a mask leaves the previous frame's polygons on the figure otherwise, and
        they would sit there looking like a current result.
        """
        plot2d = getattr(self.src_plot, "_plot2d", None)
        if plot2d is None:
            return
        group = self._overlay_group(plot2d)
        updates = {}
        if group is not None:
            updates[group] = {"vertices_list": []}
        # NO PREVIEW-WINDOW BOX on this path, and the existing one is cleared.
        #
        # The box existed to admit that only the middle megapixel of a large
        # frame had been looked at — without it, an untouched 15/16 of a 4096²
        # frame reads as "found nothing" rather than "not examined". The mask
        # preview covers a full 4096² frame (`_PREVIEW_PIXEL_BUDGET_MASK`), so
        # there is nothing left to disclaim and the box is just a rectangle drawn
        # over the data.
        #
        # It is CLEARED rather than merely not drawn, because switching from the
        # full preview to the mask must take the old box with it.
        #
        # Above the mask budget a crop is still possible (an 8192² cryo movie),
        # and that case is still reported — `preview_box` goes out in the
        # `seg_preview` payload and the caret says so in text.
        frame_group = self._window_group(plot2d)
        if frame_group is not None:
            updates[frame_group] = {"vertices_list": []}
        if updates:
            try:
                _push_groups(plot2d, updates)
            except Exception as exc:
                log.debug("[seg] overlay clear failed: %s", exc)
        if not self._set_raster_overlay(mask, box, full_shape):
            # No raster path available (an old anyplotlib, or a plot with no
            # `set_overlay_mask`). Say so — silently drawing nothing is how the
            # "106 particles and no overlay" report happened.
            log.warning("[seg] raster overlay unavailable; the mask preview has "
                        "nothing to draw with")
        self._ov_cleared = False
        self._ov_box_state = None

    def _set_raster_overlay(self, labels, box, full_shape) -> bool:
        """Draw the instances as ONE mask instead of N polygons. True if drawn.

        Uses ``Plot2D.set_overlay_mask``, which composites client-side onto the
        transparent 2-D canvas that sits ABOVE the WebGPU canvas — so this works
        on a GPU-rendered base, which a marker-layer raster would not.

        THE TILING TRAP, and why the reduction is NOT done here any more. When a
        large frame is in tile mode the renderer sizes the mask against
        ``base_width || image_width`` — the OVERVIEW grid — while tile mode sets
        ``image_width`` to the FULL native frame. This method used to reduce the
        mask to the overview grid itself, which was right for the renderer and
        REJECTED by ``set_overlay_mask``'s own shape check (it compared against
        the image shape). The ``ValueError`` landed in the ``except`` below, was
        logged at DEBUG, and every large-frame preview fell back to N polygons —
        so on a 4096² frame the raster path could never run, which is precisely
        the case it was added for. At N=14028 that fallback is what hangs the
        renderer and paints the frame a solid sheet of green.

        anyplotlib now accepts a full-resolution mask in tile mode and does the
        block-ANY reduction itself (``_reduce_mask_any``), so there is ONE owner
        of that arithmetic and it cannot disagree with the shape check sitting
        next to it. Hand it the native-resolution mask and let it decide.
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
            plot2d.set_overlay_mask(mask, color=_PREVIEW_COLOR, alpha=_RASTER_ALPHA)
            # Register for teardown HERE, in the method that actually draws,
            # rather than in the caller. `_clear_raster_overlay` refuses to clear
            # unless this is set, so a second drawing caller that forgot it left
            # a mask the caret could not take down with it — which is exactly
            # what `set_mask_overlay` did when the mask-only preview landed.
            # One owner: whoever draws it, arms the teardown.
            self._ov_raster = True
            return True
        except Exception as exc:
            # WARNING, not debug. The fallback from here is N polygons, and N is
            # only large enough to be here because it was already too large to
            # draw — so a silent failure trades a missing overlay for a hung
            # renderer. If this line appears, that is the bug.
            log.warning("[seg] raster overlay failed (%s); falling back to outlines",
                        exc)
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

        BOTH drawing routes, which is the half that was missing. This cleared
        the vector outlines and left any RASTER mask in place, so above
        `_RASTER_ABOVE` instances the previous engine's result survived the
        switch untouched. On the Scribble tab that is not merely stale, it is
        disabling: the mask covers the image you have to paint on, and the
        reported symptom was exactly "moved to scribble, immediately unusable".
        """
        plot2d = getattr(self.src_plot, "_plot2d", None)
        if plot2d is None:
            return
        self._clear_raster_overlay()
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
        # No preview-window box: the mask preview covers a whole 4096² frame, so
        # there is no untouched remainder left to disclaim. See `set_mask_overlay`.
        frame_group = self._window_group(plot2d)
        if frame_group is not None:
            updates[frame_group] = {"vertices_list": []}
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
        # `SpyDEParticles.from_frames` requires one contour PER ROW, in order.
        # The preview skips outline extraction above the draw cap, so committing
        # that preview would build a store whose outlines silently do not
        # correspond to its rows. Refuse, and say why — the only way to be here
        # is a preview with thousands of instances, which is the failed-threshold
        # case (`_threshold_failed`) and not something worth committing anyway.
        if len(prev["contours"]) != len(prev["rows"]):
            emit_error(
                f"Segment Particles: {len(prev['rows'])} instances is too many "
                "to commit as outlines — this is usually a threshold that "
                "landed in the noise. Tighten the size filter, or train the "
                "Scribble classifier, then commit.")
            return None
        return commit_single_frame(
            self.session, self, prev["labels"], prev["rows"], prev["contours"],
            int(prev["frame"]))


# ── parameters ───────────────────────────────────────────────────────────────

def _coerce(payload: dict | None) -> dict:
    """Payload → a complete, valid parameter dict.

    Every out-of-range value is corrected rather than raised on: these arrive
    from a slider mid-drag, and a caret that errors while you are moving it is
    unusable. The correction that changes what the user asked for
    (``min_size``) is echoed back in every preview so the number on screen is
    the number that ran.

    Keys are taken from ``DEFAULTS`` only, so a payload — or a reloaded
    provenance dict — carrying a parameter of the deleted classical engine is
    dropped here rather than raising.
    """
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
    p["min_separation"] = max(1, int(p["min_separation"]))
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

    Naming EVERY field explicitly is also what makes a stale saved result load:
    a `.spyde` written before the classical engine was removed carries
    ``threshold`` / ``sensitivity`` / ``gaussian`` / ``rb_kernel`` / ``invert``
    / ``local_size`` in its provenance, and they are simply not read here, so
    they can never reach ``SegmentParams`` and raise.
    """
    return dict(
        watershed=bool(p["watershed"]), min_separation=int(p["min_separation"]),
        marker_smooth=float(p["marker_smooth"]), min_size=_min_size_px(p),
        max_size=int(p["max_size"]), clear_border=bool(p["clear_border"]),
        merge_distance=_nm_to_px(p.get("merge_nm", 0.0), p),
    )


#: Axis units that are a REAL-SPACE LENGTH, and their size in nanometres. The
#: face controls are in nm, so an axis calibrated in µm or Å converts through
#: this rather than being divided by a raw number in the wrong unit.
#:
#: Anything NOT in here — `nm^-1` and friends from a reciprocal-space signal,
#: `mrad`, `px`, an empty unit — is not a length, and then there is no physical
#: distance to convert to. Those signals fall back to PIXELS and the caret
#: relabels the two sliders accordingly (`_face_units`). A reciprocal axis is
#: the case that made this necessary: dividing a nanometre by a value in nm⁻¹
#: is dimensionally meaningless, and it silently produced a merge radius wrong
#: by whatever the camera length happened to be.
_NM_PER: dict[str, float] = {
    "m": 1e9, "cm": 1e7, "mm": 1e6,
    "um": 1e3, "µm": 1e3, "μm": 1e3, "micron": 1e3,
    "nm": 1.0,
    "a": 0.1, "å": 0.1, "ang": 0.1, "angstrom": 0.1,
    "pm": 1e-3,
}


def _length_nm_per_px(scale: float, units: str) -> float:
    """*scale* in nm/px, or 0.0 when the axis is not a real-space length."""
    try:
        s = float(scale)
    except (TypeError, ValueError):
        return 0.0
    if not (s > 0):
        return 0.0
    factor = _NM_PER.get(str(units or "").strip().lower())
    return s * factor if factor else 0.0


def _face_units(scale: float, units: str) -> str:
    """What the two face sliders are actually in: 'nm' or 'px'."""
    return "nm" if _length_nm_per_px(scale, units) > 0 else "px"


def _nm_to_px(value_nm: float, p: dict) -> float:
    """A face control in nanometres -> pixels, using the signal's own scale.

    The face controls are physical so they mean the same thing at any
    magnification, and so the number matches what the scale bar says. `scale` is
    stashed on the params at dispatch **already converted to nm per pixel**
    (:meth:`SegmentWizard.set_params`); a missing or zero scale means the signal
    is uncalibrated *or its axis is not a length at all*, and then the value IS
    pixels rather than being silently divided by a number in the wrong unit.
    """
    try:
        v = float(value_nm or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if v <= 0:
        return 0.0
    try:
        scale = float(p.get("scale") or 0.0)
    except (TypeError, ValueError):
        scale = 0.0
    return v / scale if scale > 0 else v


def _min_size_px(p: dict) -> int:
    """`min_nm` (a DIAMETER) wins over the raw pixel `min_size` when set.

    Area, not diameter, is what the size filter compares -- a particle of
    diameter d covers ~pi/4 d^2 pixels -- so converting the physical control
    means going through the area, not handing the diameter straight over.
    """
    d_px = _nm_to_px(p.get("min_nm", 0.0), p)
    if d_px > 0:
        return max(1, int(round(3.14159265 / 4.0 * d_px * d_px)))
    return int(p["min_size"])


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

    if method == "scribble":
        clf = wiz.classifier
        if clf is None or not clf.is_trained:
            emit_status("Segment Particles: paint a few scribbles — including at "
                        "least one FAINT particle — then press Train.")
            return None

        # SAME DEVICE POLICY AS THE BATCH, which is find-vectors' policy.
        #
        # This path had none of it, and that was invisible while `classical`
        # was the default engine: classical is pure scipy and never touched the
        # device, so the interactive preview never submitted to the GPU.
        # Deleting it made EVERY preview a torch submission — one per tune, one
        # per navigator move, each on its own `run_on_worker` thread — with
        # nothing bounding how many occupy the device at once. That is exactly
        # the opportunistic-GPU collapse `_gpu_task_allowed` was written for on
        # the vectors path: the kernels serialise, the allocator thrashes, and
        # the feature stack's band budget (sized against *free* VRAM) is
        # divided by however many submitters happen to be in flight.
        #
        # What is borrowed is `_gpu_slots()` — segmentation's process-wide
        # device semaphore (`PARTICLE_DEVICE_CONC`, `SPYDE_SEG_GPU_CONC`) — and
        # ONLY that.
        #
        # It is the SAME semaphore the batch's workers take, deliberately: this
        # process is not in the dask lane (`_gpu_task_allowed` returns True off
        # a worker), so a preview fired while a batch runs is a feeder the lane
        # never counted. Sharing the semaphore is what stops the caret adding
        # one. See `PARTICLE_DEVICE_CONC` for the VRAM arithmetic — every
        # feeder claims 25% of *free* device memory, so they multiply.
        #
        # NOT `_cap_torch_threads()`, deliberately, even though the batch calls
        # it: it is documented as a no-op off a dask worker precisely because
        # "the interactive preview and the training fit are single,
        # latency-sensitive calls that should keep the whole machine". Capping
        # intra-op threads is right for nine workers × four task slots and
        # wrong for one preview the user is waiting on. The two paths want the
        # same DEVICE policy and opposite CPU policies.
        #
        # The split stays OUTSIDE the slot: it is numpy/scipy and holding a
        # device slot across it is what turns N feeders back into one.
        from spyde.particles.batch import _gpu_slots
        from spyde.particles.instances import split_instances

        def _engine_fn(frame, _clf=clf, _sp=sp):
            with _gpu_slots():
                fg, bnd = _clf.predict_foreground_boundary(frame)
            return split_instances(fg, _sp, boundary=bnd)

        return _engine_fn

    # plan B4: EfficientSAM-Ti through the existing model registry.
    emit_status("Segment Particles: prompt segmentation is not installed yet — "
                "paint a few scribbles and press Train instead.")
    return None


def _mask_engine(wiz: SegmentWizard, p: dict) -> Callable[[np.ndarray], np.ndarray] | None:
    """A ``frame → bool mask`` callable: the classifier, and NOTHING after it.

    This is the live-preview path, and it exists because the split and the
    measurement — neither of which a live preview needs — are 87% of it.
    Measured per stage on one 4096² frame, warm:

        FrameNorm                 48 ms
        featurise + head         162 ms
        softmax + device->host    52 ms
        split_instances          732 ms   <- skipped here
        measure_frame + contours 958 ms   <- skipped here
        ------------------------------
        full preview            1953 ms       mask only ~262 ms

    "Is my scribble good enough yet?" is a question about the CLASSIFICATION, so
    the answer does not require deciding which pixels belong to which instance,
    nor measuring any of them. Instance identity and the size distribution come
    from the real run (and from `seg_commit`), where they are worth their price.

    Shares :func:`_gpu_slots` with :func:`_engine` and the batch, for the reasons
    written there.
    """
    if p["method"] != "scribble":
        return None
    clf = wiz.classifier
    if clf is None or not clf.is_trained:
        return None
    from spyde.particles.batch import _gpu_slots

    def _mask_fn(frame, _clf=clf):
        with _gpu_slots():
            fg, _bnd = _clf.predict_foreground_boundary(frame)
        return np.asarray(fg) > 0.5

    return _mask_fn


# ── controller resolution ────────────────────────────────────────────────────

def _wizard(session, plot) -> SegmentWizard | None:
    _src, tree = _src_plot_tree(session, plot)
    wiz = getattr(tree, "_seg_wizard", None) if tree is not None else None
    return wiz if (wiz is not None and not wiz._closed) else None


def _emit(wiz: SegmentWizard, msg: dict) -> None:
    """Send a caret message, resolving the window id AT SEND TIME.

    `SegmentWizard.window_id` is copied from `src_plot.window_id` when the
    controller is built, and `Plot.window_id` is itself copied from
    `plot_window` when the PLOT is built — so a plot that acquired its window
    afterwards carries None, the wizard copies the None, and every message this
    caret sends is addressed to nobody. The renderer filters `seg_state` by
    window id (`useWizardEvent`), so the caret then shows an empty class list
    and no brush strip: it opens, and silently cannot be used.

    Re-reading the plot each time costs nothing and fixes the ordering. The
    stale captured value is the fallback, not the source of truth.
    """
    wid = getattr(wiz.src_plot, "window_id", None)
    if wid is None:
        wid = wiz.window_id
    else:
        wiz.window_id = wid           # keep the cached one honest
    if wid is None:
        # Unroutable. Never silent: the caret will look merely empty.
        log.warning("[seg] %s has no window_id — the caret cannot receive it",
                    msg.get("type", "message"))
    msg.setdefault("window_id", wid)
    emit(msg)


def _emit_state(wiz: SegmentWizard) -> None:
    """The caret's authoritative state: engine, classes + pixel counts, which
    frames carry labels, and the EFFECTIVE parameters."""
    try:
        n_frames, _get, shape = wiz.frames()
    except Exception:
        n_frames, shape = 1, (0, 0)
    # The CLASS LIST is the load-bearing part of this message: the caret gates
    # its brush strip on `classes.length > 0`, so a `seg_state` that does not
    # arrive — or arrives without classes — is a caret that opens, shows "—"
    # where the classes should be, and has nothing to paint with. That is
    # indistinguishable from "segmentation is broken" and it is silent.
    #
    # So every OTHER field is best-effort, and the message goes out regardless.
    try:
        classes = wiz.class_report()
    except Exception as exc:
        log.warning("[seg] class report failed (%s); falling back to defaults",
                    exc)
        from spyde.particles import default_classes
        classes = [dict(c.to_dict(), pixels=0) for c in default_classes()]
    try:
        frame = wiz.frame_index()
    except Exception:
        frame = 0
    _emit(wiz, {
        "type": "seg_state",
        "method": wiz.params.get("method", DEFAULTS["method"]),
        "frame": frame,
        "n_frames": int(n_frames),
        "frame_shape": [int(shape[0]), int(shape[1])],
        "classes": classes,
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

#: Area budget for the MASK-ONLY preview, which does not split or measure.
#:
#: The 1-megapixel figure above was set by what a FULL preview cost: 8.36 s on a
#: 4096² frame, 7.6 s of it the watershed. That is why a 4k frame only ever
#: previewed its middle sixteenth, with a drawn box saying so — the rest of the
#: frame read as "not looked at" because it genuinely had not been.
#:
#: The mask path does not split and does not measure, and a whole 4096² frame
#: through it measures ~262 ms (FrameNorm 48 + featurise/head 162 + softmax and
#: readback 52). So the crop no longer buys anything worth its cost in
#: comprehensibility, and the budget is raised to cover a full 4096² frame with
#: headroom. Above this — an 8192² cryo movie — the crop still applies, and the
#: preview-window box still tells the truth about what was looked at.
_PREVIEW_PIXEL_BUDGET_MASK = 20 * 1024 * 1024


def _preview_window(frame: np.ndarray, budget: int | None = None
                    ) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """``(frame_or_crop, box)`` — bound one preview's cost by AREA, not by scale.

    Returns the frame untouched (and ``box=None``) when it already fits the
    budget, which is the common case for a tutorial-sized movie and means the
    fast path pays nothing for this. Otherwise a centred crop of the same aspect
    ratio, at full resolution, plus its ``(y0, x0, h, w)`` so the caller can tell
    the user what it measured.

    *budget* overrides :data:`_PREVIEW_PIXEL_BUDGET`. The mask-only preview passes
    :data:`_PREVIEW_PIXEL_BUDGET_MASK` — see there for why the 1-megapixel figure
    stopped being the right one.
    """
    h, w = frame.shape[:2]
    budget = int(budget or _PREVIEW_PIXEL_BUDGET)
    if h * w <= budget:
        return frame, None
    shrink = (h * w / budget) ** 0.5
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

#: The absolute ceiling on polygons handed to the renderer, whatever route got
#: us here. `_RASTER_ABOVE` chooses the nicer drawing; this one is the seatbelt
#: for when the raster is unavailable or fails, because the fallback used to be
#: unbounded — a report of 14028 instances on a real in-situ frame meant 14028
#: filled paths, which hangs the renderer and composites into one flat green
#: sheet that shows nothing at all. Well above any legitimate outline count.
_MAX_OUTLINE_POLYS = 1500

#: The raster overlay's colour/opacity. Deliberately the same green as the
#: vector outlines so crossing the threshold does not look like a mode change.
_RASTER_ALPHA = 0.45

#: A preview that calls this much of the window foreground, AND shatters it into
#: this many pieces, is a FAILED THRESHOLD being reported as a result.
#:
#: Measured on a synthetic stand-in for the reported frame (low-contrast noisy
#: support film, 8 faint dark particles, 1024²) — a global threshold has no
#: bimodal histogram to find here, so otsu lands in the middle of the noise:
#:
#:     defaults ............................. 4873 instances, 39% coverage
#:     min_size=200 .......................... 208 instances, 14% coverage
#:     min_size=2000 .......................... 17 instances, 7.2% coverage
#:     watershed off .......................... 751 instances, 40% coverage
#:     gaussian=2 + min_size=200 + no watershed . 8 instances, 52% coverage
#:
#: The last row is the trap and the reason the test is on BOTH numbers: 8
#: instances looks like the 8 real particles, but at 52% coverage those 8 bodies
#: ARE the film. Neither number alone is diagnostic — a genuinely crowded frame
#: can be 40% covered by real particles, and a coarse threshold can find 8 real
#: objects. Together they are: thousands of pieces AND half the frame is the
#: signature of noise being segmented.
#:
#: Deliberately NOT a silent auto-correction. The plan's §0.9 answer to this
#: data is the scribble classifier, and no threshold tweak substitutes for it —
#: so the caret's job is to say the threshold failed and point at Scribble, not
#: to quietly pick different parameters that fail differently.
_FAIL_COVERAGE = 0.25
_FAIL_COUNT = 500

#: Coverage at which the count stops mattering. A field of REAL particles can be
#: 40% of a frame — crowded, but a legitimate answer — so the shatter test above
#: needs the count as well. Nothing legitimate is most of the frame: at 60% the
#: "particles" are the support film whether there are 200 of them or 20 000.
#:
#: This second branch exists because the first one MISSED the failure it was
#: written for, in the only way that matters — on screen. A head mis-trained to
#: call film "particle" (`seg_autolabel {"mislabel": true}`) returned **228**
#: instances covering essentially the whole frame: a solid green sheet, plainly
#: wrong, and under `_FAIL_COUNT`. The two engines fail with different SHAPES —
#: a threshold shatters the film into thousands of fragments, a bad classifier
#: merges it into a few hundred blobs — and a rule tuned on one shape does not
#: see the other.
_FAIL_COVERAGE_ALONE = 0.60


def _threshold_failed(count: int, coverage: float) -> bool:
    """True when a preview is the support film being reported as particles.

    Two shapes of the same failure; see `_FAIL_COVERAGE` and
    `_FAIL_COVERAGE_ALONE`. Deliberately NOT about thresholds any more, despite
    the name it kept: it reads the measured output, so it catches a bad
    classifier as readily as it caught a bad threshold.
    """
    if count <= 0:
        return False
    if coverage >= _FAIL_COVERAGE_ALONE:
        return True
    return count >= _FAIL_COUNT and coverage >= _FAIL_COVERAGE


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


def _preview_mask(wiz: SegmentWizard, gen: int, p: dict, t: int, units: str,
                  mask_fn) -> None:
    """The live preview: classify the displayed frame and paint the mask.

    No ``split_instances``, no ``measure_frame``, no contours — see
    :func:`_mask_engine` for the measurement that motivates it. The mask goes up
    through the SAME raster overlay the >100-instance path already uses, so
    nothing new had to learn about tiled frames (the raster path must be handed a
    NATIVE-resolution mask and let anyplotlib reduce it — reducing it here is the
    bug that made the overlay unreachable on every frame it was written for).
    """
    def _work():
        _n, get_frame, _shape = wiz.frames()
        full = np.asarray(get_frame(t))
        frame, box = _preview_window(full, _PREVIEW_PIXEL_BUDGET_MASK)
        t0 = time.perf_counter()
        mask = mask_fn(frame)
        return {"frame": t, "mask": mask, "box": box,
                "full_shape": full.shape,
                "coverage": float(mask.mean()) if mask.size else 0.0,
                "elapsed": time.perf_counter() - t0}

    def _done(res):
        # Hand the navigator gate back BEFORE the generation guard, exactly as
        # the full preview does — a superseded or closed preview that keeps it
        # wedges every later navigator move.
        wiz._nav_busy = False
        if not wiz.still(gen) or wiz._closed:
            return
        wiz.preview = {"frame": res["frame"], "mask": res["mask"],
                       "count": -1, "areas": np.zeros(0, np.float32),
                       "box": res.get("box")}
        wiz.set_mask_overlay(res["mask"], res.get("box"), res.get("full_shape"))
        _emit(wiz, {
            "type": "seg_preview",
            "frame": int(res["frame"]),
            # -1 is "not counted", NOT "found nothing". The caret must render
            # these differently or a good mask reads as a failed segmentation.
            "count": -1,
            "mask_only": True,
            "areas": [],
            "median_area": 0.0,
            "units": units,
            "method": p["method"],
            "min_size": int(p["min_size"]),
            "min_size_floored": bool(p["min_size_floored"]),
            "elapsed_ms": round(1000.0 * res["elapsed"], 1),
            "preview_box": (None if res.get("box") is None
                            else [int(v) for v in res["box"]]),
            "coverage": round(float(res.get("coverage") or 0.0), 4),
            # Coverage alone still catches the threshold landing in the noise —
            # it is the half of `_threshold_failed` that does not need a count.
            "threshold_failed": bool(float(res.get("coverage") or 0.0) > 0.35),
            "face_units": _face_units(*wiz.scale_units()),
        })
        _chase_nav(wiz)

    def _fail(exc):
        wiz._nav_busy = False
        if wiz.still(gen):
            emit_error(f"Segment Particles preview failed: {exc}")

    run_on_worker(wiz.session, _work, name="seg-preview-mask",
                  on_done=_done, on_error=_fail)


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

    # THE LIVE PREVIEW IS MASK-ONLY. See `_mask_engine` for the per-stage
    # measurement: the split and the measurement are 87% of a full preview and a
    # live preview needs neither, so this path stops at the classification.
    # `count` is reported as -1, meaning "not counted" — the caret shows coverage
    # instead, and the real count comes from the run.
    mask_fn = _mask_engine(wiz, p)
    if mask_fn is not None:
        _preview_mask(wiz, gen, p, t, units, mask_fn)
        return

    def _work():
        _n, get_frame, _shape = wiz.frames()
        full = np.asarray(get_frame(t))
        frame, box = _preview_window(full)
        t0 = time.perf_counter()
        labels = engine(frame)
        from spyde.particles import measure_frame
        # Outlines only when they can actually be drawn. `labels.max()` is the
        # instance count for the price of one pass, and above the draw cap the
        # overlay discards the polygons anyway — extracting thousands of them
        # first is pure latency on the tune the user is waiting for (measured:
        # ~half of a 1353 ms preview at 4873 instances). The measured PROPERTIES
        # are all still computed, so the count, the histogram, the median and
        # the confidence filter are unaffected.
        n_lab = int(np.asarray(labels).max()) if np.asarray(labels).size else 0
        rows, contours = measure_frame(
            labels, frame, t=t, scale=scale,
            want_contours=n_lab <= _MAX_OUTLINE_POLYS)
        n_all = len(rows)
        rows, contours = filter_by_score(rows, contours, p.get("min_score", 0.0))
        # What FRACTION of the previewed window was called foreground. This is
        # the one number that separates "found the particles" from "the
        # threshold landed inside the noise" — see `_threshold_failed`.
        coverage = float((np.asarray(labels) > 0).mean()) if labels.size else 0.0
        return {"frame": t, "labels": labels, "rows": rows, "n_all": n_all,
                "contours": contours, "elapsed": time.perf_counter() - t0,
                "coverage": coverage, "box": box, "full_shape": full.shape}

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
                        full_shape=res.get("full_shape"),
                        n_instances=int(len(rows)))
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
            # The threshold-failure verdict (see `_threshold_failed`). Reported
            # rather than silently swallowed: the count is still true, it just
            # is not an ANSWER, and the caret has to say which of the two it is
            # holding.
            "coverage": round(float(res.get("coverage") or 0.0), 4),
            "threshold_failed": _threshold_failed(
                int(len(rows)), float(res.get("coverage") or 0.0)),
            # What the two FACE sliders are in — 'nm' on a real-space axis, 'px'
            # when the axis is not a length (a reciprocal-space signal reports a
            # healthy positive scale in nm^-1, and labelling that 'nm' is a
            # claim about the scale bar that is simply false).
            "face_units": _face_units(scale, units),
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
        existing.set_params(payload)
        _arm_brush(existing)
        gen = existing.guard()
        _emit_state(existing)
        _preview(existing, gen)
        return

    wiz = SegmentWizard(session, tree, src)
    wiz.set_params(payload)
    # BEFORE anything deferred: React StrictMode fires open/close/open
    # synchronously, so the close's bump must be able to invalidate this open.
    gen = wiz.guard()
    tree._seg_wizard = wiz
    # Follow the navigator from the moment the caret opens, not from the first
    # Train: scrolling with the caret up should keep the outlines on the frame
    # you are looking at whichever engine is selected.
    wiz.wire_navigator()
    # ARM THE BRUSH HERE, not on an engine switch.
    #
    # It used to be armed only by `seg_set_method("scribble")`, which the caret
    # sent when you clicked the Scribble tab. Deleting the classical engine
    # deleted the tab row, so nothing sent that verb any more and the brush was
    # never attached: the caret opened, told you to paint, and there was nothing
    # to paint with. Reported as "I can't scribble".
    #
    # Opening is the right trigger now. There is one engine, painting is the
    # only way to teach it, and the caret's own UI already assumes this — the
    # ClassStrip renders unconditionally.
    _arm_brush(wiz)
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
    """Switch engine and re-preview. One engine ships today; the verb stays
    because plan B4's prompt head is the second."""
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
    wiz.set_params(payload, merge=True)
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


def seg_autolabel(session, plot, payload=None) -> None:
    """TEST DOOR: paint labels from a synthetic fixture's stamped ground truth.

    With the classical engine gone there is no result until something is
    trained, so every spec whose subject is something ELSE — the overlay
    reaching a tiled frame, the batch opening a result window, the caret's
    layout — would otherwise have to hand-place brush strokes at coordinates it
    has no way to know, and would silently stop testing its own subject the day
    the fixture moved a particle.

    So this replaces the PAINTING only. It rasterises through ``_paint_stroke``,
    the same function the real brush and ``seg_paint`` use, and it stops there:
    the spec then presses Train like a user, and ``seg_train`` is untouched.
    What is faked is *where the strokes are*, which is exactly the part a spec
    about the overlay has no opinion about.

    Refuses on a signal with no stamped truth — this must never quietly become
    a way to get a trained head on real data, where the labels would be wrong
    and the test would pass anyway.

    Payload: ``{"frame": int, "mislabel": bool}``. ``mislabel`` SWAPS the two
    classes — the support film is painted as *particle* and the particles as
    *film* — which is the realistic way a user breaks this: one careless dab on
    the background and the head learns that the film is what you are looking
    for. It is how a spec reproduces the over-segmentation the coverage verdict
    (:func:`_threshold_failed`) exists to name, now that the classical engine
    that used to produce it by accident is gone. A correctly trained head does
    NOT over-segment even a heavily noisy frame — measured, which is the whole
    argument for having deleted the other engine.
    """
    wiz = _wizard(session, plot)
    if wiz is None:
        return
    from spyde.data.synthetic import ground_truth, particle_truth_at

    signal = wiz.signal()
    try:
        gt = ground_truth(signal)
    except (ValueError, AttributeError) as exc:
        emit_error(f"seg_autolabel is a test door and needs a synthetic "
                   f"fixture's stamped ground truth: {exc}")
        return

    payload = payload or {}
    t = int(payload.get("frame", wiz.frame_index()))
    # Swapped, so "paint a dab on each particle" teaches FILM and the sweeps
    # across the background teach PARTICLE.
    c_particle, c_film = (1, 0) if payload.get("mislabel") else (0, 1)
    pos, radii, present = particle_truth_at(gt, t)
    h, w = (int(v) for v in gt["frame_shape"])
    idx = [int(i) for i in np.flatnonzero(present)]
    if not idx:
        emit_error(f"seg_autolabel: no particles present at frame {t}")
        return

    # PARTICLE: a dab at each centre, well inside the body so a radius that is
    # slightly off cannot label film as particle.
    painted = 0
    for i in idx:
        cy, cx = float(pos[i, 0]), float(pos[i, 1])
        painted += _paint_stroke(wiz, t, [[cy, cx]], c_particle, False,
                                 max(1.5, float(radii[i]) * 0.5))

    # FILM: sweeps that clear every particle by a margin. Painting background is
    # not optional politeness — a head that has never seen a not-quite-particle
    # pixel returns visibly fat masks (measured in test_particles_scribble).
    yy, xx = np.mgrid[0:h, 0:w]
    clear = np.ones((h, w), bool)
    for i in idx:
        d2 = (yy - pos[i, 0]) ** 2 + (xx - pos[i, 1]) ** 2
        clear &= d2 > (float(radii[i]) + max(3.0, 0.05 * h)) ** 2
    band = max(2, h // 24)
    sweeps = np.zeros((h, w), bool)
    sweeps[:band, :] = True
    sweeps[-band:, :] = True
    sweeps[:, :band] = True
    sweeps[h // 2:h // 2 + band, :] = True
    store = wiz.label_store()
    store.paint(t, sweeps & clear, c_film)
    painted += int((sweeps & clear).sum())

    log.info("[seg] autolabel: %d px across %d particles + film on frame %d",
             painted, len(idx), t)
    _emit_state(wiz)


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
        from spyde.particles import FastScribbleClassifier
        _n, get_frame, _shape = wiz.frames()
        clf = FastScribbleClassifier(device=device)
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
    wiz.set_params(payload, merge=True)
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
            _report_stage_costs(session, n_frames)
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


def _report_stage_costs(session, n_frames: int) -> None:
    """Log where a finished batch actually spent its time, per stage per lane.

    Every worker already records ``(t0, n, device, engine_s, measure_s,
    block_s)`` for every task (``batch._STAGE_LOG``) — and until now only the
    benchmark ever drained it, so a real run threw the numbers away and left
    "it's slow" with nothing attached.

    That matters here more than usual because the three stages have completely
    different fixes and roughly comparable costs on a 4096² frame (measured:
    ~1.0 s featurise+head, ~1.8 s split, ~1.4 s measure). Optimising the wrong
    one is the default outcome of guessing, and CLAUDE.md's benchmarking rule
    says so in as many words: time each stage separately, because "it's slow"
    is usually one of them.

    Best-effort and never fatal: this is diagnostics on the way out of a run
    that has already produced its result.
    """
    try:
        from spyde.particles.batch import drain_stage_log

        client = _batch_client(session)
        recs: list = []
        if client is not None:
            for per_worker in (client.run(drain_stage_log) or {}).values():
                recs.extend(per_worker or [])
        else:
            recs.extend(drain_stage_log())
        if not recs:
            return

        by_dev: dict[str, list] = {}
        for _t0, n, dev, eng, meas, block in recs:
            by_dev.setdefault(str(dev), []).append((n, eng, meas, block))
        for dev, rows in sorted(by_dev.items()):
            frames = sum(r[0] for r in rows)
            eng = sum(r[1] for r in rows)
            meas = sum(r[2] for r in rows)
            block = sum(r[3] for r in rows)
            other = max(0.0, block - eng - meas)
            if frames <= 0:
                continue
            log.info(
                "[seg-batch] %s: %d frames in %d tasks | per frame: "
                "engine %.2fs (%.0f%%), measure %.2fs (%.0f%%), other %.2fs "
                "(%.0f%%) | %.3f frames/s",
                dev, frames, len(rows),
                eng / frames, 100.0 * eng / max(block, 1e-9),
                meas / frames, 100.0 * meas / max(block, 1e-9),
                other / frames, 100.0 * other / max(block, 1e-9),
                frames / max(block, 1e-9))
    except Exception as exc:                                  # pragma: no cover
        log.debug("[seg-batch] stage-cost report unavailable: %s", exc)


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
