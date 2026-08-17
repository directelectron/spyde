"""
drift_action.py — the Drift Correction wizard (``drift_`` staged actions).

Plan A8, rewritten under plan §0.9a (*"the caret shows ONE control; everything
else is Advanced"*) after the first review: **"way too complicated. Too many
options. Information overload."**

    drift_open        caret mounted → Drift Check window, the alignment ROI on
                      the movie, and the first discovery preview
    drift_close       caret unmounted → tear all of it down
    drift_set_method  rigid | rigid+affine (lives in Advanced now)
    drift_tune        a toggle/parameter changed → re-run the discovery preview
    drift_run         solve the movie on a worker; opens the dy/dx window and
                      fills it progressively from the solver's ``on_shift``
    drift_discard     drop the solved model (and stop a solve in flight)
    drift_commit      add the LAZY corrected node to the tree

**The caret carries the TASK, not the algorithm.** Its default face is two
toggles and one button. Reference mode, sub-pixel factor, max shift,
interpolation order and the model tabs are all real and all still here — they
sit behind a collapsed *Advanced* in the caret, and the schema below (the one
source of truth, mirrored by ``registry._WIZARD_SCHEMAS``) tags them so any
host renders the same split. Nothing was deleted; provenance still records
every parameter.

**Discovery comes before commitment.** The centrepiece is a draggable
rectangle on the movie plus a live drift-corrected sum of just that box over
~20 frames (:data:`_PREVIEW_FRAMES`). A good landmark sums sharp, a bad one
blurs, and the *gain* number (:func:`_gradient_energy` of the aligned sum over
the raw sum, measured on the SAME pixels) puts a figure on it. So the user sees
whether alignment works on a subset before paying for the whole movie — and the
"Use ROI for alignment" toggle then feeds that exact rectangle to
``solve_translation(roi=…)``, which is often the more CORRECT answer anyway:
whole-frame correlation is contaminated by the sample's own motion (see
``spyde/drift/translation.py``'s ``roi`` docs).

**Geometry is in IMAGE PIXELS end to end.** anyplotlib's 2-D widgets report
``x/y/w/h`` in image pixels with no scale/offset applied, and
``solve_translation``'s ``roi=(y0, x0, h, w)`` is in pixels too, so the two meet
with no conversion. Do not add one "for consistency" — see
``spyde/actions/masks.py::_signal_k_grids`` for that bug class.

**Two windows, each with one job.** The *Drift Check* window is the evidence:
the whole-movie raw/corrected sums on top, the discovery pair (ROI raw vs ROI
aligned) beneath. The *Drift dy/dx* window is the curve, opened when the solve
starts and filled progressively from ``on_shift`` — it is a normal figure
window, not caret furniture. Both are bare ``figure`` windows (NOT registered
``Plot``s), so each registers a controller via ``own_window`` and keeps its
figure referenced through ``figure_registry.keep_alive``, per
``actions/README.md`` §6.

**Nothing here materialises the movie.** ``solve_translation`` streams one
frame at a time; the check sums stream over a bounded subset
(:data:`_SUM_MAX_FRAMES`); the preview reads one full frame at a time and keeps
only the small crop, under a byte cap (:data:`_PREVIEW_MAX_BYTES`); and
``drift_commit`` adds a ``map_blocks`` node so the corrected movie is a lazy
view, never a copy (plan §0.7). The corrected node is tagged ``local=True``
because a rigid shift is exactly per-frame, which is what lets the existing
``LocalTransformReader`` scrub it.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from spyde.actions.context import current_signal as _current_signal
from spyde.actions.context import src_plot_tree as _src_plot_tree
from spyde.actions.lifecycle import (
    bump_generation, is_current, run_on_worker, show_tree_node,
)
from de_shell.actions.wizard import WizardController
from de_shell.ipc import emit, emit_error, emit_progress, emit_status

log = logging.getLogger(__name__)

#: Solver families. ``rigid`` is the only one ``spyde.drift`` implements today;
#: ``rigid_affine`` is declared so a host can render the choice, and it falls
#: back to ``rigid`` with an explicit status rather than silently doing
#: something the user did not ask for. It lives inside Advanced (§0.9a).
METHODS: tuple[str, ...] = ("rigid", "rigid_affine")

_UNAVAILABLE = {
    "rigid_affine": ("the affine drift search (plan A4) is not implemented in "
                     "spyde.drift yet"),
}

#: Frames summed for the whole-movie before/after check images. A sum is a
#: SHARPNESS test, not a measurement — a few dozen frames already show the blur
#: unambiguously, and the cap is what keeps the check window responsive on a
#: movie whose full pass costs as much as the solve itself. Evenly spaced, and
#: the SAME indices for both sums, or the comparison means nothing.
_SUM_MAX_FRAMES = 64

# Frames per streamed drift-trace message / per dy-dx repaint. One message per
# frame would flood the PLOTAPP line protocol at the plan's target scale
# (thousands of frames) for a curve the eye cannot follow at that resolution;
# batching by 16 keeps the trace visibly live while cutting the message count
# by the same factor.
_TRACE_BATCH = 16

# …but a COUNT alone is not enough, and the first screenshot showed why: a
# 12-frame movie never reaches 16, so the curve stayed empty for the whole solve
# and appeared complete at the end — the exact opposite of "fills in as it is
# computed". Flush on whichever comes first, so the trace is live at any movie
# length and still capped at ~7 messages/s on a fast one.
_TRACE_MAX_INTERVAL = 0.15

#: Figure geometry for the two bare-figure windows.
#:
#: A bare figure never receives ``resize_figure`` (that path resolves a
#: registered ``Plot``), so its INITIAL px size is the one it keeps and anything
#: outside it is CLIPPED by the subwindow — which is what cut the check
#: window's bottom row in half. The renderer sizes a new window from the
#: ``aspect`` field as ``inner_h = clamp(460 / aspect, 130, 300)`` then
#: ``inner_w = inner_h * aspect`` (``MDIArea.windowSize``). At the height cap
#: the first clamp is active for any aspect below 460/300, so a figure exactly
#: :data:`_FIG_HEIGHT` tall lands pixel-for-pixel in its window at any width up
#: to 460 — pick the width, derive the aspect.
#:
#: The width is deliberately the renderer's OWN default (340). Widening the
#: check window to 460 made it no longer fit beside the movie, so the free-slot
#: packer wrapped it to the next row — straight on top of the caret, which is an
#: overlay the packer cannot see. Keeping the default width keeps the placement
#: the packer already gets right.
_FIG_WIDTH = 340
_FIG_HEIGHT = 300


def _figure_geometry(width: int = _FIG_WIDTH) -> tuple[tuple[int, int], float]:
    """``(figsize, aspect)`` that opens a bare-figure window with no clipping."""
    w = int(min(460, max(190, width)))
    return (w, _FIG_HEIGHT), w / float(_FIG_HEIGHT)


#: Frames the discovery preview aligns. ~20 is the brief's number and it is a
#: DEFAULT, not a law — ``preview_frames`` in Advanced moves it.
#:
#: Sampled EVENLY OVER THE WHOLE MOVIE, not the first 20 in a row. The question
#: a preview answers is "does this landmark survive the FULL excursion", and 20
#: consecutive frames of a 3000-frame movie drift by almost nothing — a
#: contiguous window would answer "looks fine" for every box, including the
#: useless ones. The same reasoning (and the same spacing) as
#: :meth:`DriftWizard.sum_indices`.
_PREVIEW_FRAMES = 20

#: Byte ceiling on the preview's retained crop stack. The preview reads one
#: FULL frame at a time and keeps only the (usually small) ROI crop, so this
#: bounds the only thing that accumulates. With no ROI the crop IS the frame,
#: which is how 20 frames of a 4096² movie would otherwise become 1.3 GB;
#: over the cap the sampled frame count is thinned rather than the read being
#: abandoned. Never a reason to touch the full dataset (CLAUDE.md).
_PREVIEW_MAX_BYTES = 192 * 1024 * 1024

#: Settle delay for a preview re-solve driven by an ROI DRAG. The widget's
#: pointer_move fires at renderer frame rate; re-solving 20 frames on each one
#: would queue solves faster than they finish. ``drift_tune`` is NOT debounced
#: here — the renderer's ``useDebouncedAction`` already settles it, and
#: debouncing twice just adds latency.
_PREVIEW_SETTLE_S = 0.25

#: Smallest alignment box, in image pixels. MUST stay >= the solver's own
#: ``spyde.drift.translation._MIN_ROI``, which REJECTS a smaller box rather
#: than clamping it (a silently shrunk ROI would correlate somewhere the user
#: did not drag). Pinned by ``test_drift_wizard.py``.
_ROI_MIN_PX = 16

#: Default alignment box: this fraction of each frame dimension, centred. Half
#: the frame is deliberately generous — the ROI is FIXED in frame coordinates,
#: so the landmark drifts within it and the box wants to be comfortably larger
#: than the total excursion.
_ROI_DEFAULT_FRACTION = 0.5

_ROI_COLOR = "#94e2d5"

#: **Off by default, and that is a measurement, not caution.** A guessed centre
#: box is NOT automatically the better correlation: on the ``particle_movie``
#: fixture (96×112 frames, the default half-frame box = 48×56) the ROI solve
#: comes back 1.03 px from the stamped ground truth where the whole-frame solve
#: is 0.25 px — a quarter of the pixels is a quarter of the correlation signal,
#: and the Tukey taper eats a larger fraction of a small box. So the default
#: stays the answer we already know is right, and the ROI is what the user
#: reaches for when the whole frame is the problem (a moving sample, a mostly
#: featureless field). The preview runs on the box either way — that is the
#: discovery step, and it is what tells you the box is worth committing to.
DEFAULTS: dict[str, Any] = dict(
    use_roi=False,
    reject_outliers=True,
    method="rigid",
    upsample=8,
    max_shift=32.0,
    reference="running",
    apodize=True,
    normalize=True,
    order=1,
    preview_frames=_PREVIEW_FRAMES,
)


class DriftWizard(WizardController):
    """Owns the drift caret's state: parameters, the alignment ROI and its live
    preview, the solved model, and the two figure windows."""

    key = "drift"

    #: One source of truth (mirrored by ``registry._WIZARD_SCHEMAS``). Entries
    #: WITHOUT a ``tab`` are the caret's default face; everything tagged
    #: ``"Advanced"`` renders behind the collapsed disclosure (§0.9a).
    parameters = {
        "use_roi": {
            "name": "Use ROI for alignment", "type": "bool",
            "default": DEFAULTS["use_roi"],
        },
        "reject_outliers": {
            "name": "Ignore bad frames", "type": "bool",
            "default": DEFAULTS["reject_outliers"],
        },
        "method": {
            "name": "Model", "type": "enum", "default": DEFAULTS["method"],
            "choices": list(METHODS), "tab": "Advanced",
        },
        "reference": {
            "name": "Reference", "type": "enum", "default": DEFAULTS["reference"],
            "choices": ["running", "sequential", "first"], "tab": "Advanced",
        },
        "upsample": {
            "name": "Sub-pixel factor", "type": "int", "default": DEFAULTS["upsample"],
            "min": 1, "max": 64, "tab": "Advanced",
        },
        "max_shift": {
            "name": "Max shift (px)", "type": "float", "default": DEFAULTS["max_shift"],
            "min": 1.0, "max": 4096.0, "step": 1.0, "tab": "Advanced",
        },
        "apodize": {
            "name": "Edge taper", "type": "bool", "default": DEFAULTS["apodize"],
            "tab": "Advanced",
        },
        "normalize": {
            "name": "Phase correlation", "type": "bool",
            "default": DEFAULTS["normalize"], "tab": "Advanced",
        },
        "order": {
            "name": "Interpolation order", "type": "int", "default": DEFAULTS["order"],
            "min": 0, "max": 3, "tab": "Advanced",
        },
        "preview_frames": {
            "name": "Preview frames", "type": "int",
            "default": DEFAULTS["preview_frames"], "min": 4, "max": 200,
            "tab": "Advanced",
        },
    }

    def __init__(self, session, tree, src_plot):
        super().__init__(session, tree)
        self.src_plot = src_plot
        self.src_window_id = getattr(src_plot, "window_id", None)
        self.params: dict[str, Any] = dict(DEFAULTS)
        self.model = None
        #: The Drift Check window (a bare figure) and its four panels.
        self.window_id: int | None = None
        self._panels: dict[str, Any] = {}
        self._sum_indices: np.ndarray | None = None
        self._before_sum: np.ndarray | None = None
        #: The dy/dx window — opened by the solve, filled from ``on_shift``.
        self.trace_window_id: int | None = None
        self._trace: dict[str, Any] = {}
        #: The alignment ROI (discovery): widget + last preview result.
        self._roi_widget = None
        self._roi_handler = None
        self._roi_clamping = False
        self._frame_shape: tuple[int, int] | None = None
        self._settle: threading.Timer | None = None
        self.preview: dict[str, Any] | None = None
        #: Cancel flag of the solve in flight (Discard/Stop flips it).
        self._stop: list[bool] = [False]

    # ── the movie ────────────────────────────────────────────────────────────

    def signal(self):
        return _current_signal(self.src_plot) or self.tree.root

    def frames(self):
        """``(n_frames, get_frame, (h, w))`` — one frame at a time."""
        from spyde.drift import frame_source
        return frame_source(self.signal())

    def sum_indices(self, n_frames: int) -> np.ndarray:
        if self._sum_indices is None or self._sum_indices.size == 0:
            k = min(int(n_frames), _SUM_MAX_FRAMES)
            self._sum_indices = np.unique(
                np.linspace(0, max(0, n_frames - 1), max(1, k)).round().astype(int))
        return self._sum_indices

    # ── the alignment ROI (the discovery feature) ────────────────────────────

    def _plot2d(self):
        return getattr(self.src_plot, "_plot2d", None) if self.src_plot else None

    def ensure_roi_widget(self, shape: tuple[int, int]) -> None:
        """Draw the draggable alignment box on the source movie (idempotent).

        Geometry is IMAGE PIXELS — anyplotlib 2-D widgets report ``x/y/w/h``
        that way, and that is exactly what ``solve_translation(roi=…)`` wants.
        A raw ``add_rectangle_widget`` rather than ``RectangleSelector``: the
        selector caps itself at ``MAX_REGION_EXTENT_PER_DIM`` (16 px) because
        it drives a nav-space region integrate, and a 16 px alignment box is
        below the solver's own floor.
        """
        h, w = int(shape[0]), int(shape[1])
        self._frame_shape = (h, w)
        if self._roi_widget is not None:
            return
        plot2d = self._plot2d()
        if plot2d is None:
            return
        if min(h, w) < 2 * _ROI_MIN_PX:
            # Nothing sensible to drag; the whole frame IS the ROI.
            return
        bw = max(_ROI_MIN_PX, min(w, int(round(w * _ROI_DEFAULT_FRACTION))))
        bh = max(_ROI_MIN_PX, min(h, int(round(h * _ROI_DEFAULT_FRACTION))))
        try:
            widget = plot2d.add_rectangle_widget(
                x=float((w - bw) // 2), y=float((h - bh) // 2),
                w=float(bw), h=float(bh), color=_ROI_COLOR, show_handles=True,
            )
            from spyde.drawing.selectors.base_selector import event_handler_fn
            handler = event_handler_fn(lambda event: self._on_roi_drag())
            widget.add_event_handler(handler, "pointer_move", "pointer_up")
            self._roi_widget = widget
            self._roi_handler = handler      # keep a ref alive (weak callback)
        except Exception as exc:
            log.debug("[drift] alignment ROI widget failed: %s", exc)

    def _on_roi_drag(self) -> None:
        """Clamp the box to the frame, then arm the settle timer.

        RE-ENTRANCY GUARD: anyplotlib ``Widget.set()`` fires ``pointer_move``
        UNCONDITIONALLY (even on a no-change write), so the clamp below
        re-invokes this handler synchronously — unguarded, ONE JS drag frame
        recursed ~2000 deep before RecursionError in the Crop box (see
        ``actions/base.py``). A hard flag breaks the cycle; compare-before-set
        is NOT sufficient.
        """
        if self._roi_clamping or self._closed:
            return
        self._roi_clamping = True
        try:
            self._clamp_roi()
        finally:
            self._roi_clamping = False
        self.schedule_preview()

    def _clamp_roi(self) -> None:
        """Keep the box inside the frame and above the solver's floor.

        COMPARE BEFORE SET, with slack. ``Widget.set()`` pushes geometry back to
        the renderer, and writing on every ``pointer_move`` echoes
        python-sourced geometry into a live drag — the same failure the 1-D span
        cap documents in CLAUDE.md (Live-Display §3). A box already resting on a
        bound must be left alone.
        """
        widget, shape = self._roi_widget, self._frame_shape
        if widget is None or shape is None:
            return
        h, w = shape
        try:
            ww = min(max(float(widget.w), float(_ROI_MIN_PX)), float(w))
            hh = min(max(float(widget.h), float(_ROI_MIN_PX)), float(h))
            x = min(max(float(widget.x), 0.0), float(w) - ww)
            y = min(max(float(widget.y), 0.0), float(h) - hh)
            now = (float(widget.x), float(widget.y),
                   float(widget.w), float(widget.h))
            if max(abs(a - b) for a, b in zip(now, (x, y, ww, hh))) > 1e-6:
                widget.set(x=x, y=y, w=ww, h=hh)
        except Exception as exc:
            log.debug("[drift] clamping the alignment ROI failed: %s", exc)

    def roi_box(self) -> tuple[int, int, int, int] | None:
        """``(y0, x0, h, w)`` in IMAGE PIXELS, or None when there is no usable
        box — the shape ``solve_translation``'s ``roi`` takes, with no scale or
        offset applied because neither side has any."""
        widget, shape = self._roi_widget, self._frame_shape
        if widget is None or shape is None:
            return None
        fh, fw = shape
        try:
            x0 = int(round(float(widget.x)))
            y0 = int(round(float(widget.y)))
            bw = int(round(float(widget.w)))
            bh = int(round(float(widget.h)))
        except Exception as exc:
            log.debug("[drift] reading the alignment ROI failed: %s", exc)
            return None
        # SIZE first, then origin. The other order looks equivalent and is not:
        # clamping the origin to the frame edge and only then applying the
        # minimum size pushes the box back OUT past the edge, and
        # solve_translation rejects an out-of-frame roi outright.
        bw = max(_ROI_MIN_PX, min(bw, fw))
        bh = max(_ROI_MIN_PX, min(bh, fh))
        if bw > fw or bh > fh:
            return None                  # frame smaller than the solver's floor
        x0 = max(0, min(x0, fw - bw))
        y0 = max(0, min(y0, fh - bh))
        return (y0, x0, bh, bw)

    def active_roi(self) -> tuple[int, int, int, int] | None:
        """The ROI the FULL SOLVE should use — None unless the toggle is on (and
        None when no box is usable, which is a whole-frame correlation).

        The preview deliberately does NOT go through here: it always aligns the
        box, toggle or not, because that is the question it exists to answer
        ("is this landmark worth committing to?"). The toggle is the commitment.
        """
        return self.roi_box() if self.params.get("use_roi") else None

    def remove_roi_widget(self) -> None:
        widget, self._roi_widget = self._roi_widget, None
        self._roi_handler = None
        if widget is not None:
            try:
                widget.hide()          # widgets have no remove(), only hide()
            except Exception as exc:
                log.debug("[drift] hiding the alignment ROI failed: %s", exc)

    # ── preview scheduling (latest-wins, cancellable) ────────────────────────

    def schedule_preview(self, delay: float = _PREVIEW_SETTLE_S) -> None:
        """(Re-)arm the settle timer for a drag-driven preview re-solve.

        Latest-wins in two places: the timer is restarted per pointer event so
        only the RESTING geometry ever solves, and the solve that does run
        carries a ``_drift_preview_gen`` generation so a superseded result is
        dropped on arrival instead of painting over a newer one.
        """
        self.cancel_preview()
        if self._closed:
            return
        timer = threading.Timer(max(0.0, float(delay)), self._fire_preview)
        timer.daemon = True
        self._settle = timer
        timer.start()

    def cancel_preview(self) -> None:
        timer, self._settle = self._settle, None
        if timer is not None:
            try:
                timer.cancel()
            except Exception as exc:
                log.debug("[drift] cancelling the preview timer failed: %s", exc)

    def _fire_preview(self) -> None:
        """Timer thread → main thread → the worker. Reading widget geometry and
        spawning the compute both belong on the main thread (thread marshal,
        README §6); only the arithmetic runs on the worker."""
        self._settle = None
        if self._closed:
            return
        dispatch = getattr(self.session, "_dispatch_to_main", None)
        if dispatch is None:
            _run_preview(self)
        else:
            dispatch(lambda: (None if self._closed else _run_preview(self)))

    # ── the check window ─────────────────────────────────────────────────────

    def open_check_window(self, before: np.ndarray, n_frames: int) -> None:
        """Emit the bare-figure Drift Check window and register this controller
        for it, so ✕ and ``Session._forget_window`` reach the wizard.

        Top row = the whole movie, raw and corrected (the solve's evidence).
        Bottom row = the DISCOVERY pair, raw and aligned over ~20 frames of
        whatever is being correlated (the ROI, or the whole frame when the
        toggle is off). Side by side, because "is this landmark good" is
        answered by comparing two sums, not by staring at one.
        """
        import anyplotlib as apl
        import anyplotlib._electron as _electron
        from de_shell.actions.figure_registry import keep_alive
        from spyde.drawing.plots.plot import finalize_figure_html

        figsize, aspect = _figure_geometry()
        fig, axes = apl.subplots(2, 2, figsize=figsize)
        ax = np.array(axes, dtype=object).ravel()
        before = np.asarray(before, np.float32)
        zeros = np.zeros_like(before)

        panels = {
            "before": ax[0].imshow(before, cmap="gray"),
            "after": ax[1].imshow(zeros, cmap="gray"),
            "roi_raw": ax[2].imshow(zeros, cmap="gray"),
            "roi_aligned": ax[3].imshow(zeros, cmap="gray"),
        }
        titles = {"before": "Raw sum", "after": "Corrected sum",
                  "roi_raw": "ROI raw", "roi_aligned": "ROI aligned"}
        self._panels = panels
        for key, title in titles.items():
            self._set_panel_title(key, title)

        wid = self.session.next_window_id()
        fig_id = _electron.register(fig)
        html = finalize_figure_html(fig, fig_id)
        keep_alive(int(wid), fig)
        emit({"type": "figure", "fig_id": fig_id, "window_id": int(wid),
              "html": html, "title": "Drift Check", "is_navigator": False,
              "aspect": float(aspect)})
        self.window_id = int(wid)
        self._before_sum = before
        self.own_window(wid)

    def _set_panel_title(self, key: str, title: str) -> None:
        panel = self._panels.get(key)
        if panel is None:
            return
        try:
            panel.set_title(title)
        except Exception as exc:
            log.debug("[drift] set_title(%s) failed: %s", key, exc)

    def update_check(self, *, after=None) -> None:
        """Paint the whole-movie corrected sum (main thread only)."""
        if after is None or not self._panels:
            return
        try:
            self._panels["after"].set_data(np.asarray(after, np.float32))
        except Exception as exc:
            log.debug("[drift] painting the corrected sum failed: %s", exc)

    def show_preview(self, result: dict) -> None:
        """Paint the discovery pair + its titles (main thread only)."""
        if not self._panels:
            return
        what = "ROI" if result.get("roi") is not None else "Whole frame"
        n = int(result.get("frames", 0))
        gain = float(result.get("gain", float("nan")))
        for key, arr, title in (
            ("roi_raw", result.get("raw"), f"{what} raw · {n} frames"),
            ("roi_aligned", result.get("aligned"),
             f"{what} aligned · {gain:.1f}x sharper" if np.isfinite(gain)
             else f"{what} aligned"),
        ):
            if arr is None:
                continue
            try:
                self._panels[key].set_data(np.asarray(arr, np.float32))
            except Exception as exc:
                log.debug("[drift] painting the %s panel failed: %s", key, exc)
            self._set_panel_title(key, title)

    # ── the dy/dx window ─────────────────────────────────────────────────────

    def open_trace_window(self, n_frames: int) -> None:
        """Open (or reset) the dy/dx figure window.

        Its OWN window, not caret furniture: the curve is the measurement, and
        a 40 px inline sparkline could show that the stage crept but never
        which frame jumped. One panel with two labelled lines rather than two
        panels — drift is anisotropic, and a shared y-scale is what makes
        "mostly x" readable at a glance.
        """
        n = max(2, int(n_frames))
        if self.trace_window_id is not None and self._trace:
            self.reset_trace(n)
            return
        import anyplotlib as apl
        import anyplotlib._electron as _electron
        from de_shell.actions.figure_registry import keep_alive
        from spyde.drawing.plots.plot import finalize_figure_html

        figsize, aspect = _figure_geometry()
        fig, axes = apl.subplots(1, 1, figsize=figsize)
        ax = np.array(axes, dtype=object).ravel()[0]
        x0 = np.zeros(1, dtype=np.float64)
        y0 = np.zeros(1, dtype=np.float64)
        panel = ax.plot(y0, axes=[x0], units="frame", y_units="shift (px)",
                        color="#89b4fa", label="dy")
        dx_line = panel.add_line(y0, x_axis=x0, color="#f38ba8", label="dx")
        for setter, text in (("set_title", "Drift dy / dx"),
                             ("set_xlabel", "frame"),
                             ("set_ylabel", "shift (px)")):
            try:
                getattr(panel, setter)(text)
            except Exception as exc:
                log.debug("[drift] trace %s failed: %s", setter, exc)

        wid = self.session.next_window_id()
        fig_id = _electron.register(fig)
        html = finalize_figure_html(fig, fig_id)
        keep_alive(int(wid), fig)
        emit({"type": "figure", "fig_id": fig_id, "window_id": int(wid),
              "html": html, "title": "Drift dy/dx", "is_navigator": False,
              "aspect": float(aspect)})
        self.trace_window_id = int(wid)
        self._trace = {"panel": panel, "dx": dx_line}
        self.reset_trace(n)
        self.own_window(wid)

    def reset_trace(self, n_frames: int) -> None:
        n = max(2, int(n_frames))
        self._trace["dy_data"] = np.full(n, np.nan, np.float64)
        self._trace["dx_data"] = np.full(n, np.nan, np.float64)
        self._trace["filled"] = 0
        self.push_trace([(0, 0.0, 0.0)])

    def push_trace(self, points) -> None:
        """Append a batch of ``(index, dy, dx)`` and repaint (main thread only).

        Only the SOLVED PREFIX is pushed, so the curve grows left to right and
        the y-scale tracks what has actually been measured — pushing the whole
        NaN-padded array would make anyplotlib's auto-range see one point.
        """
        if not self._trace:
            return
        dy = self._trace.get("dy_data")
        dx = self._trace.get("dx_data")
        if dy is None or dx is None:
            return
        hi = int(self._trace.get("filled", 0))
        for i, y, x in points:
            i = int(i)
            if 0 <= i < dy.size:
                dy[i] = float(y)
                dx[i] = float(x)
                hi = max(hi, i + 1)
        self._trace["filled"] = hi
        if hi < 1:
            return
        xs = np.arange(hi, dtype=np.float64)
        try:
            self._trace["panel"].set_data(np.nan_to_num(dy[:hi]), x_axis=xs)
            self._trace["dx"].set_data(np.nan_to_num(dx[:hi]), x_axis=xs)
        except Exception as exc:
            log.debug("[drift] painting the dy/dx trace failed: %s", exc)

    def close_trace_window(self) -> None:
        self._trace = {}
        wid, self.trace_window_id = self.trace_window_id, None
        self._close_window(wid)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _close_window(self, wid: int | None) -> None:
        if wid is None:
            return
        forget = getattr(self.session, "_forget_window", None)
        if forget is not None:
            try:
                forget(int(wid))
            except Exception as exc:
                log.debug("[drift] forgetting window %s failed: %s", wid, exc)
            return
        # Bare / stub session: emit + unregister by hand.
        try:
            emit({"type": "window_closed", "window_id": int(wid)})
        except Exception as exc:
            log.debug("[drift] closing window %s failed: %s", wid, exc)
        reg = getattr(self.session, "_window_controllers", None)
        if isinstance(reg, dict):
            reg.pop(int(wid), None)

    def close(self) -> None:
        """WindowController protocol — ``Session._forget_window`` calls this for
        EITHER owned window, with no way to say which.

        Only the Drift Check window is the wizard's life; closing the dy/dx
        window just drops the curve. ``_forget_window`` pops the controller for
        the window that went away BEFORE calling here, so "is the check window
        still registered?" identifies it exactly — and the programmatic path
        (:meth:`close_trace_window`) clears ``trace_window_id`` first, so this
        re-entry is a no-op rather than a recursion.
        """
        reg = getattr(self.session, "_window_controllers", None) or {}
        if self.window_id is not None and reg.get(int(self.window_id)) is self:
            self._trace = {}
            self.trace_window_id = None
            return
        self.remove()

    def remove(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop[0] = True             # stop a solve in flight
        self.cancel_preview()
        self.remove_roi_widget()
        self._panels = {}
        self._trace = {}
        wid, self.window_id = self.window_id, None
        twid, self.trace_window_id = self.trace_window_id, None
        self._close_window(wid)
        self._close_window(twid)
        if getattr(self.tree, "_drift_wizard", None) is self:
            self.tree._drift_wizard = None

    def commit(self):
        """Add the lazy corrected node — see :func:`drift_commit`."""
        return _commit(self)


# ── parameters ───────────────────────────────────────────────────────────────

def _coerce(payload: dict | None) -> dict:
    p = dict(DEFAULTS)
    payload = payload or {}
    for k, default in DEFAULTS.items():
        v = payload.get(k)
        if v is None or v == "":
            continue
        try:
            p[k] = bool(v) if isinstance(default, bool) else type(default)(v)
        except (TypeError, ValueError) as exc:
            log.debug("[drift] param %r=%r not coercible, keeping default: %s",
                      k, v, exc)
    p["method"] = str(p["method"]).lower()
    if p["method"] not in METHODS:
        p["method"] = DEFAULTS["method"]
    if p["reference"] not in ("running", "sequential", "first"):
        p["reference"] = DEFAULTS["reference"]
    p["upsample"] = max(1, int(p["upsample"]))
    p["max_shift"] = max(1.0, float(p["max_shift"]))
    p["order"] = int(min(3, max(0, p["order"])))
    p["preview_frames"] = int(min(200, max(4, p["preview_frames"])))
    return p


def _solver_kwargs(p: dict, roi=None) -> dict:
    return dict(upsample=int(p["upsample"]), max_shift=float(p["max_shift"]),
                reference=str(p["reference"]), apodize=bool(p["apodize"]),
                normalize=bool(p["normalize"]),
                reject_outliers=bool(p["reject_outliers"]),
                roi=None if roi is None else tuple(int(v) for v in roi))


def _wizard(session, plot) -> DriftWizard | None:
    """Resolve the live wizard from any of its windows.

    The check and dy/dx windows are bare figures, so ``_plot_by_window_id``
    returns None for them and the plot-based lookup finds nothing — resolve by
    window id through the controller registry first (README §6), then fall back
    to the source tree's back-reference.
    """
    wid = getattr(plot, "window_id", None) if plot is not None else None
    lookup = getattr(session, "controller_by_window_id", None)
    if wid is not None and lookup is not None:
        ctrl = lookup(int(wid))
        if isinstance(ctrl, DriftWizard) and not ctrl._closed:
            return ctrl
    _src, tree = _src_plot_tree(session, plot)
    wiz = getattr(tree, "_drift_wizard", None) if tree is not None else None
    return wiz if (wiz is not None and not wiz._closed) else None


def _emit_state(wiz: DriftWizard, **extra) -> None:
    roi = wiz.roi_box()
    msg = {"type": "drift_state",
           "window_id": wiz.src_window_id,
           "check_window_id": wiz.window_id,
           "trace_window_id": wiz.trace_window_id,
           "method": wiz.params["method"],
           "solved": wiz.model is not None,
           "use_roi": bool(wiz.params["use_roi"]),
           "roi": None if roi is None else [int(v) for v in roi],
           "params": dict(wiz.params)}
    msg.update(extra)
    emit(msg)


# ── streaming sums + the sharpness number ────────────────────────────────────

def _stack_sum(get_frame, indices, shifts=None, *, order: int = 1) -> np.ndarray:
    """Mean of the selected frames, optionally drift-corrected first.

    Streams: one frame resident at a time plus one float64 accumulator, so this
    is safe at the plan's target scale however long the movie is. NaN padding
    from :func:`spyde.drift.warp.shift_frame` is excluded per pixel rather than
    zero-filled — a zero-filled border reads as a dark rim that looks like real
    data and would be segmented as one.
    """
    acc = None
    hits = None
    for i in indices:
        frame = np.asarray(get_frame(int(i)), dtype=np.float32)
        if shifts is not None:
            s = shifts[int(i)]
            if np.all(np.isfinite(s)):
                from spyde.drift import shift_frame
                frame = shift_frame(frame, s, order=order)
        if acc is None:
            acc = np.zeros(frame.shape, np.float64)
            hits = np.zeros(frame.shape, np.int32)
        good = np.isfinite(frame)
        acc[good] += frame[good]
        hits[good] += 1
    if acc is None:
        return np.zeros((1, 1), np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(hits > 0, acc / np.maximum(hits, 1), np.nan)
    return out.astype(np.float32)


def _gradient_energy(img, mask=None) -> float:
    """Mean squared forward-difference gradient over the valid pixels.

    The sharpness number, and it is NaN-aware by construction rather than by
    ``nan_to_num``: an aligned sum's uncovered border is NaN (plan A7 — nothing
    is cropped, nothing is invented), and zero-filling it manufactures a step
    at the border whose gradient energy dwarfs the image's own, which would
    make every ROI look brilliantly sharp. Differences touching a non-finite
    (or masked-out) pixel are excluded from BOTH the sum and the count, so the
    raw and aligned sums are measured over exactly the same pixels.
    """
    a = np.asarray(img, np.float64)
    ok = np.isfinite(a)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    a = np.where(ok, a, 0.0)
    total = 0.0
    count = 0
    if a.shape[0] > 1:
        m = ok[1:, :] & ok[:-1, :]
        d = (a[1:, :] - a[:-1, :])[m]
        total += float(np.sum(d * d))
        count += int(m.sum())
    if a.shape[1] > 1:
        m = ok[:, 1:] & ok[:, :-1]
        d = (a[:, 1:] - a[:, :-1])[m]
        total += float(np.sum(d * d))
        count += int(m.sum())
    return total / count if count else float("nan")


def _preview_indices(n_frames: int, k: int, frame_bytes: int) -> np.ndarray:
    """Evenly spaced sample of the movie for the preview, thinned to fit the
    byte cap. See :data:`_PREVIEW_FRAMES` for why evenly spaced and not the
    first *k* in a row."""
    n = max(1, int(n_frames))
    k = max(2, min(int(k), n))
    cap = max(2, int(_PREVIEW_MAX_BYTES // max(1, int(frame_bytes))))
    k = min(k, cap)
    return np.unique(np.linspace(0, n - 1, k).round().astype(int))


def preview_alignment(get_frame, indices, roi, *, params) -> dict:
    """Align *indices* on *roi* alone and report how much sharper the sum got.

    Reads one FULL frame at a time and keeps only the crop, so the resident set
    is one frame plus ``len(indices)`` crops (bounded by
    :data:`_PREVIEW_MAX_BYTES` at the caller). The crops ARE the region to
    correlate, so the solve runs with ``roi=None`` on them.

    Returns ``{roi, frames, raw, aligned, gain, raw_energy, aligned_energy,
    max_abs_shift}``. *gain* is the whole point: > 1 means aligning this region
    genuinely sharpened it, ~1 means alignment changes nothing here (a
    featureless box), and it is measured on the pixels both sums cover.
    """
    from spyde.drift import solve_translation

    crops: list[np.ndarray] = []
    for i in indices:
        frame = np.asarray(get_frame(int(i)), np.float32)
        if roi is not None:
            y0, x0, h, w = (int(v) for v in roi)
            frame = frame[y0:y0 + h, x0:x0 + w]
        crops.append(np.ascontiguousarray(frame, dtype=np.float32))

    model = solve_translation(crops, **_solver_kwargs(params))
    take = range(len(crops))
    raw = _stack_sum(crops.__getitem__, take)
    aligned = _stack_sum(crops.__getitem__, take, model.shifts,
                         order=int(params["order"]))
    both = np.isfinite(raw) & np.isfinite(aligned)
    e_raw = _gradient_energy(raw, both)
    e_aligned = _gradient_energy(aligned, both)
    gain = (e_aligned / e_raw) if (np.isfinite(e_raw) and e_raw > 0) \
        else float("nan")
    return {"roi": None if roi is None else tuple(int(v) for v in roi),
            "frames": len(crops), "raw": raw, "aligned": aligned,
            "gain": float(gain), "raw_energy": float(e_raw),
            "aligned_energy": float(e_aligned),
            "max_abs_shift": float(model.max_abs_shift)}


# ── staged handlers ──────────────────────────────────────────────────────────

def drift_open(session, plot, payload) -> None:
    """Caret mounted: build the controller, open the Drift Check window, draw
    the alignment ROI, and run the first discovery preview.

    Nothing SOLVES here — drift correction is deliberately opt-in:
    and never runs on load. The compute is the bounded raw sum plus the
    ~20-frame preview of the default box, which is what makes the caret's first
    frame informative instead of an empty panel and a button.
    """
    src, tree = _src_plot_tree(session, plot)
    if src is None or tree is None:
        emit_error("Drift Correction: no active dataset")
        return

    existing = getattr(tree, "_drift_wizard", None)
    if existing is not None and not existing._closed:
        existing.params = _coerce({**existing.params, **(payload or {})})
        _emit_state(existing)
        return

    wiz = DriftWizard(session, tree, src)
    wiz.params = _coerce(payload)
    try:
        n_frames, get_frame, shape = wiz.frames()
    except TypeError as exc:
        emit_error(f"Drift Correction: {exc}")
        return
    # BEFORE the worker: StrictMode fires open/close/open synchronously and the
    # close's bump has to be able to invalidate this open's deferred build.
    gen = wiz.guard()
    tree._drift_wizard = wiz
    _emit_state(wiz, n_frames=int(n_frames))

    def _work():
        return _stack_sum(get_frame, wiz.sum_indices(n_frames))

    def _done(raw_sum):
        if not wiz.still(gen) or wiz._closed:
            return
        wiz.open_check_window(raw_sum, int(n_frames))
        wiz.ensure_roi_widget(shape)
        _emit_state(wiz, n_frames=int(n_frames))
        emit_status("Drift Correction: drag the box onto a landmark to test it, "
                    "then Correct Drift.")
        _run_preview(wiz)

    def _fail(exc):
        emit_error(f"Drift Correction: reading the movie failed — {exc}")

    run_on_worker(session, _work, name="drift-open", on_done=_done, on_error=_fail)


def drift_close(session, plot, payload=None) -> None:
    """Caret unmounted: invalidate in-flight work FIRST, then tear down."""
    _src, tree = _src_plot_tree(session, plot)
    wiz = _wizard(session, plot)
    if tree is not None:
        # The same `_drift_run_gen` key WizardController.cancel_inflight bumps,
        # done on the TREE so it fires even when there is no controller yet: a
        # StrictMode open whose worker has not landed must still be cancelled.
        bump_generation(tree, "_drift_run_gen")
        bump_generation(tree, "_drift_preview_gen")
    if wiz is not None:
        # Harmlessly re-bumps when the tree resolved above; the point is the
        # case where it did not (the wizard was found through one of the
        # figure windows' controller registry).
        wiz.cancel_inflight()
        wiz.remove()


def drift_set_method(session, plot, payload) -> None:
    """Select the drift model (Advanced).

    Only ``rigid`` has a solver in ``spyde.drift`` today. Selecting the other
    says so and stays on rigid — running a rigid solve while the caret claims
    "rigid+affine" would put a wrong ``kind`` into the model's provenance,
    which is worse than the missing feature.
    """
    wiz = _wizard(session, plot)
    if wiz is None:
        return
    method = str((payload or {}).get("method", "")).lower()
    if method not in METHODS:
        emit_error(f"Drift Correction: unknown model {method!r}")
        return
    reason = _UNAVAILABLE.get(method)
    if reason:
        emit_status(f"Drift Correction: {reason} — staying on the rigid solve.")
        method = "rigid"
    wiz.params["method"] = method
    _emit_state(wiz)


def drift_tune(session, plot, payload) -> None:
    """A toggle or Advanced parameter changed → re-run the discovery preview.

    NOT debounced here: the renderer's ``useDebouncedAction`` already settles
    the send, and debouncing twice only adds latency. The drag path IS
    debounced, on the backend, because widget pointer events arrive at renderer
    frame rate (:meth:`DriftWizard.schedule_preview`).
    """
    wiz = _wizard(session, plot)
    if wiz is None:
        return
    wiz.params = _coerce({**wiz.params, **(payload or {})})
    _emit_state(wiz)
    _run_preview(wiz)


def _run_preview(wiz: DriftWizard) -> None:
    """Align ~20 sampled frames on the current box and report the gain.

    Latest-wins on ``_drift_preview_gen``: a drag that outruns the solve drops
    the stale result rather than painting it over the newer one. The preview
    never touches ``tree.drift`` or the caret's solved state — it is a question,
    not an answer.
    """
    if wiz._closed:
        return
    tree = wiz.tree
    gen = bump_generation(tree, "_drift_preview_gen")
    params = dict(wiz.params)
    roi = wiz.roi_box()          # the BOX, toggle or not — see active_roi()
    try:
        n_frames, get_frame, shape = wiz.frames()
    except TypeError as exc:
        log.debug("[drift] preview skipped: %s", exc)
        return
    if n_frames < 2:
        return
    h, w = (roi[2], roi[3]) if roi is not None else (int(shape[0]), int(shape[1]))
    indices = _preview_indices(n_frames, params["preview_frames"], h * w * 4)

    def _work():
        return preview_alignment(get_frame, indices, roi, params=params)

    def _done(result):
        if not is_current(tree, "_drift_preview_gen", gen) or wiz._closed:
            return
        wiz.preview = result
        wiz.show_preview(result)
        emit({"type": "drift_preview", "window_id": wiz.src_window_id,
              "roi": None if result["roi"] is None else list(result["roi"]),
              "frames": int(result["frames"]),
              "gain": float(result["gain"]),
              "max_abs_shift": float(result["max_abs_shift"]),
              "params": dict(params)})

    def _fail(exc):
        if is_current(tree, "_drift_preview_gen", gen):
            emit_error(f"Drift preview failed: {exc}")

    run_on_worker(wiz.session, _work, name="drift-preview",
                  on_done=_done, on_error=_fail)


def drift_run(session, plot, payload) -> None:
    """Solve the whole movie on a worker: progress-reported and cancellable.

    Opens the dy/dx window FIRST and fills it from ``solve_translation``'s
    ``on_shift`` stream, so the curve draws while it solves rather than
    appearing whole at the end.

    Cancellation goes through ``BaseSignalTree.register_cancel`` so closing the
    tree stops the solve, and ``solve_translation``'s own ``cancel()`` hook
    polls the same flag — a cancelled solve leaves NaN shifts for the frames it
    never reached, which is why a partial model is detectable rather than
    silently wrong. Stop/Discard flips the same flag.
    """
    src, tree = _src_plot_tree(session, plot)
    if src is None or tree is None:
        emit_error("Drift Correction: no active dataset")
        return
    wiz = _wizard(session, plot)
    if wiz is None:
        emit_error("Drift Correction: the caret is not open")
        return
    wiz.params = _coerce({**wiz.params, **(payload or {})})
    p = dict(wiz.params)
    reason = _UNAVAILABLE.get(p["method"])
    if reason:
        emit_status(f"Drift Correction: {reason} — solving rigid instead.")
        p["method"] = wiz.params["method"] = "rigid"

    try:
        n_frames, get_frame, _shape = wiz.frames()
    except TypeError as exc:
        emit_error(f"Drift Correction: {exc}")
        return
    if n_frames < 2:
        emit_error("Drift Correction needs at least two frames")
        return

    roi = wiz.active_roi()
    if p["use_roi"] and roi is None:
        emit_status("Drift Correction: no usable alignment box — correlating "
                    "the whole frame.")

    gen = wiz.guard()
    stopped = [False]
    wiz._stop = stopped
    if hasattr(tree, "register_cancel"):
        tree.register_cancel(flag=stopped)
    wiz.open_trace_window(int(n_frames))
    _emit_state(wiz)
    emit_status(f"Solving drift over {n_frames} frames…")
    dispatch = getattr(session, "_dispatch_to_main", None)

    def _work():
        from spyde.drift import solve_translation

        def _progress(done, total):
            emit_progress(int(done), int(total), "Drift")
            emit({"type": "drift_progress", "window_id": wiz.src_window_id,
                  "done": int(done), "total": int(total)})

        # Stream the curve as it solves. `progress` carries only a count and the
        # shift array is solver-local until the return, so without this callback
        # the caret could show a bar but not a trace. Batched rather than per
        # frame: at thousands of frames one message each would flood the PLOTAPP
        # line protocol for a curve the eye cannot follow that finely. The PAINT
        # is marshalled — `on_shift` runs on the solver thread and figures are
        # main-thread only (README §6).
        pending: list[tuple[int, float, float]] = []
        last_flush = [time.monotonic()]

        def _flush():
            if not pending:
                return
            batch = pending[:]
            pending.clear()
            last_flush[0] = time.monotonic()
            emit({"type": "drift_trace", "window_id": wiz.src_window_id,
                  "points": batch})
            if not wiz.still(gen):
                return
            if dispatch is None:
                wiz.push_trace(batch)
            else:
                dispatch(lambda b=batch: (None if wiz._closed or not wiz.still(gen)
                                          else wiz.push_trace(b)))

        def _on_shift(i, dy, dx, _sharp):
            pending.append((int(i), float(dy), float(dx)))
            if (len(pending) >= _TRACE_BATCH
                    or time.monotonic() - last_flush[0] >= _TRACE_MAX_INTERVAL):
                _flush()

        model = solve_translation(
            wiz.signal(), progress=_progress, on_shift=_on_shift,
            cancel=lambda: stopped[0],
            provenance={"action": "Drift Correction", "params": dict(p),
                        "roi": None if roi is None else [int(v) for v in roi]},
            **_solver_kwargs(p, roi))
        _flush()
        if stopped[0]:
            return model, None, float("nan")
        # One extra streaming pass over the SAME bounded subset the raw sum
        # used, so the two check images are comparable.
        after = _stack_sum(get_frame, wiz.sum_indices(n_frames), model.shifts,
                           order=int(p["order"]))
        # The same number the discovery preview reports, now for the whole
        # movie — measured on the worker because a 4096² gradient energy is
        # ~100 ms and the main thread is the navigator's.
        before = wiz._before_sum
        gain = float("nan")
        if before is not None and before.shape == after.shape:
            both = np.isfinite(before) & np.isfinite(after)
            e_before = _gradient_energy(before, both)
            if np.isfinite(e_before) and e_before > 0:
                gain = _gradient_energy(after, both) / e_before
        return model, after, gain

    def _done(res):
        model, after, gain = res
        try:
            if not wiz.still(gen) or wiz._closed:
                return
            wiz.model = model
            tree.drift = model
            wiz.update_check(after=after)
            emit({"type": "drift_result", "window_id": wiz.src_window_id,
                  "shifts": [[float(a), float(b)] for a, b in model.shifts],
                  "kind": model.kind, "reference": model.reference,
                  "roi": None if roi is None else [int(v) for v in roi],
                  "max_abs_shift": float(model.max_abs_shift),
                  "gain": float(gain),
                  "rejected": int(model.params.get("rejected_from_reference", 0)),
                  "cancelled": bool(stopped[0])})
            _emit_state(wiz)
            solved = int(np.isfinite(model.shifts).all(axis=1).sum())
            if stopped[0]:
                emit_status(f"Drift solve stopped after {solved} of "
                            f"{n_frames} frames")
            else:
                emit_status(f"Drift solved: max shift "
                            f"{model.max_abs_shift:.2f} px over {n_frames} frames")
        finally:
            if hasattr(tree, "unregister_cancel"):
                try:
                    tree.unregister_cancel(flag=stopped)
                except Exception as exc:
                    log.debug("[drift] unregister_cancel failed: %s", exc)

    def _fail(exc):
        emit_error(f"Drift Correction failed: {exc}")
        log.exception("drift solve failed")
        if hasattr(tree, "unregister_cancel"):
            try:
                tree.unregister_cancel(flag=stopped)
            except Exception as e2:
                log.debug("[drift] unregister_cancel failed: %s", e2)

    run_on_worker(session, _work, name="drift-run", on_done=_done, on_error=_fail)


def drift_discard(session, plot, payload=None) -> None:
    """Stop a solve in flight and/or throw the solved model away.

    One handler for both because they are the same user intent ("no, not
    that"): the button reads *Stop* while the bar is moving and *Discard*
    once there is a result. Bumping the run generation FIRST means a solve that
    finishes anyway lands on a stale generation and never installs itself.
    """
    wiz = _wizard(session, plot)
    if wiz is None:
        return
    wiz._stop[0] = True
    wiz.cancel_inflight()
    wiz.model = None
    if getattr(wiz.tree, "drift", None) is not None:
        wiz.tree.drift = None
    wiz.close_trace_window()
    if wiz._panels and wiz._before_sum is not None:
        try:
            wiz._panels["after"].set_data(np.zeros_like(wiz._before_sum))
        except Exception as exc:
            log.debug("[drift] clearing the corrected sum failed: %s", exc)
    _emit_state(wiz)
    emit_status("Drift result discarded.")


# ── the corrected node ───────────────────────────────────────────────────────

def drift_corrected(signal, *, model, order: int = 1, fill: float = float("nan")):
    """A LAZY drift-corrected view of *signal*. Plan §0.7.

    Parameters
    ----------
    signal
        The source movie (1-D navigation, 2-D signal).
    model
        The :class:`~spyde.drift.model.DriftModel` to apply. ``shifts[i]`` is the
        correction ADDED to frame *i* — go through the model rather than writing
        the arithmetic out; the inverted sign doubles the drift and still looks
        plausible (``spyde/drift/model.py``).
    order
        Interpolation order for sub-pixel shifts. A whole-pixel model takes an
        exact slice-copy path inside :func:`~spyde.drift.warp.shift_frame`.
    fill
        Uncovered-pixel value. NaN by default, per the plan A7 edge policy —
        nothing is cropped and nothing is filled with invented data.

    Notes
    -----
    Built with ``map_blocks`` over the source's OWN chunking, deliberately: this
    never calls ``.rechunk()`` and never computes anything, so a multi-GB movie
    costs a graph and nothing else (CLAUDE.md memory-safety rule, and Live-
    Display §1 on not reshuffling storage chunks). Each block warps its own
    frames using ``block_info`` to recover their absolute indices, so a movie
    stored several frames per chunk works unchanged.
    """
    import dask.array as da
    from spyde.drift import shift_frame

    data = signal.data
    if getattr(data, "ndim", 0) != 3:
        raise ValueError(
            f"drift correction needs a (n, h, w) frame stack; got shape "
            f"{getattr(data, 'shape', None)}")
    shifts = np.asarray(model.shifts, dtype=np.float32)
    if shifts.shape[0] != int(data.shape[0]):
        raise ValueError(
            f"the drift model covers {shifts.shape[0]} frames but the signal has "
            f"{int(data.shape[0])} — solve again on this node")

    if not isinstance(data, da.Array):
        # Already resident; wrapping it costs nothing and keeps the node lazy so
        # the whole tree reads through one path.
        data = da.from_array(data, chunks=(1,) + tuple(int(s) for s in data.shape[1:]))

    def _block(blk, block_info=None):
        t0 = (0 if block_info is None
              else int(block_info[0]["array-location"][0][0]))
        out = np.empty(blk.shape, np.float32)
        for k in range(blk.shape[0]):
            s = shifts[t0 + k]
            if not np.all(np.isfinite(s)):
                # A frame the solve never reached (cancelled) keeps its raw
                # pixels rather than becoming an all-NaN hole.
                out[k] = np.asarray(blk[k], np.float32)
            else:
                out[k] = shift_frame(blk[k], s, order=int(order), fill=fill)
        return out

    warped = da.map_blocks(_block, data, dtype=np.float32,
                           meta=np.zeros((0, 0, 0), np.float32))
    new = signal._deepcopy_with_new_data(warped)
    if not new._lazy:
        new._lazy = True
        new._assign_subclass()
    return new


def _commit(wiz: DriftWizard):
    if wiz.model is None:
        emit_error("Drift Correction: solve first, then Apply")
        return None
    parent = wiz.signal()
    try:
        new_signal = wiz.tree.add_transformation(
            parent, function=drift_corrected, node_name="Drift corrected",
            local=True, model=wiz.model, order=int(wiz.params["order"]))
    except Exception as exc:
        emit_error(f"Drift Correction: applying the model failed — {exc}")
        log.exception("drift commit failed")
        return None
    if new_signal is None:
        return None
    wiz.tree.drift = wiz.model
    try:
        new_signal.metadata.set_item(
            "General.spyde_provenance",
            {"action": "Drift Correction", "params": dict(wiz.params),
             "kind": wiz.model.kind, "reference": wiz.model.reference,
             "roi": wiz.model.params.get("roi")})
    except Exception as exc:
        log.debug("[drift] stamping provenance failed: %s", exc)
    show_tree_node(wiz.src_plot, wiz.tree, new_signal)
    emit_status(f"Drift corrected node added (max shift "
                f"{wiz.model.max_abs_shift:.2f} px)")
    return new_signal


def drift_commit(session, plot, payload=None) -> None:
    """Add the lazy corrected node to the tree and show it."""
    wiz = _wizard(session, plot)
    if wiz is None:
        emit_error("Drift Correction: nothing to apply")
        return
    wiz.commit()


def drift_correction(ctx, action_name: str = "Drift Correction", **params):
    """Toolbar entry — a no-op parent; the Electron toolbar opens the staged
    caret, which drives the ``drift_*`` handlers (README §4)."""
    return None
