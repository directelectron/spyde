"""
drift_action.py — the Drift Correction wizard (``drift_`` staged actions).

Plan A8:

    drift_open        caret mounted → open the Drift Check window, show the
                      uncorrected sum, empty shift trace
    drift_close       caret unmounted → tear the check window down
    drift_set_method  rigid | rigid+affine | non-rigid
    drift_tune        debounced re-tune → re-solve the FIRST PAIR only
    drift_run         solve on a worker, cancellable, progress-reported
    drift_commit      add the LAZY corrected node to the tree

**The verification surface is a separate window, not the caret.** A 240 px caret
cannot show a sum image at a size where sharpness is judgeable, and judging
sharpness is the entire point of the check: an aligned stack sums sharp, a
misaligned one blurs. So the check window holds before/after sums side by side
with dy(t) / dx(t) beneath, and the caret holds the method tabs, the parameters
and Commit. The window is a bare ``figure`` — NOT a registered ``Plot`` — so it
registers a controller (``own_window``) and keeps its figure referenced through
``figure_registry.keep_alive``, per ``actions/README.md`` §6.

**Nothing here materialises the movie.** ``solve_translation`` streams one frame
at a time; the check sums stream too (and over a bounded subset — see
``_SUM_MAX_FRAMES``); and ``drift_commit`` adds a ``map_blocks`` node so the
corrected movie is a lazy view, never a copy (plan §0.7). The corrected node is
tagged ``local=True`` because a rigid shift is exactly per-frame, which is what
lets the existing ``LocalTransformReader`` scrub it.

**Known gap, stated rather than worked around:** ``solve_translation`` returns
its shifts only when it finishes — ``progress(done, total)`` carries no partial
trace and the array is local to the solver. So the run reports PROGRESS
progressively and draws the shift trace once, at the end. Making the trace fill
incrementally needs an ``on_shift(i, dy, dx)`` callback (or a caller-supplied
output array) in ``spyde/drift/translation.py``; inventing one here by solving
in segments would change the answer, because the running Fourier reference is
accumulated across the whole stack.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from spyde.actions.context import current_signal as _current_signal
from spyde.actions.context import src_plot_tree as _src_plot_tree
from spyde.actions.lifecycle import bump_generation, run_on_worker, show_tree_node
from spyde.actions.wizard import WizardController
from spyde.backend.ipc import emit, emit_error, emit_progress, emit_status

log = logging.getLogger(__name__)

#: Solver families. ``rigid`` is the only one ``spyde.drift`` implements today;
#: the other two are declared so the caret can render its method tabs, and both
#: fall back to ``rigid`` with an explicit status rather than silently doing
#: something the user did not ask for.
METHODS: tuple[str, ...] = ("rigid", "rigid_affine", "nonrigid")

_UNAVAILABLE = {
    "rigid_affine": ("the affine drift search (plan A4) is not implemented in "
                     "spyde.drift yet"),
    "nonrigid": ("non-rigid warping (plan A5) is not implemented in spyde.drift "
                 "yet"),
}

#: Frames summed for the before/after check images. A sum is a SHARPNESS test,
#: not a measurement — a few dozen frames already show the blur unambiguously,
#: and the cap is what keeps the check window responsive on a movie whose full
#: pass costs as much as the solve itself. Evenly spaced, and the SAME indices
#: for both sums, or the comparison means nothing.
_SUM_MAX_FRAMES = 64

# Frames per streamed drift-trace message. One message per frame would flood the
# PLOTAPP line protocol at the plan's target scale (thousands of frames) for a
# curve the eye cannot follow at that resolution; batching by 16 keeps the trace
# visibly live while cutting the message count by the same factor.
_TRACE_BATCH = 16

DEFAULTS: dict[str, Any] = dict(
    method="rigid",
    upsample=8,
    max_shift=32.0,
    reference="running",
    apodize=True,
    normalize=True,
    reject_outliers=True,
    order=1,
)


class DriftWizard(WizardController):
    """Owns the drift caret's state: parameters, the solved model, and the
    separate Drift Check window."""

    key = "drift"

    parameters = {
        "method": {
            "name": "Model", "type": "enum", "default": DEFAULTS["method"],
            "choices": list(METHODS),
        },
        "reference": {
            "name": "Reference", "type": "enum", "default": DEFAULTS["reference"],
            "choices": ["running", "sequential", "first"], "tab": "Solve",
        },
        "upsample": {
            "name": "Sub-pixel factor", "type": "int", "default": DEFAULTS["upsample"],
            "min": 1, "max": 64, "tab": "Solve",
        },
        "max_shift": {
            "name": "Max shift (px)", "type": "float", "default": DEFAULTS["max_shift"],
            "min": 1.0, "max": 4096.0, "step": 1.0, "tab": "Solve",
        },
        "apodize": {
            "name": "Edge taper", "type": "bool", "default": DEFAULTS["apodize"],
            "tab": "Solve",
        },
        "normalize": {
            "name": "Phase correlation", "type": "bool",
            "default": DEFAULTS["normalize"], "tab": "Solve",
        },
        "reject_outliers": {
            "name": "Reject bad frames", "type": "bool",
            "default": DEFAULTS["reject_outliers"], "tab": "Solve",
        },
        "order": {
            "name": "Interpolation order", "type": "int", "default": DEFAULTS["order"],
            "min": 0, "max": 3, "tab": "Apply",
        },
    }

    def __init__(self, session, tree, src_plot):
        super().__init__(session, tree)
        self.src_plot = src_plot
        self.src_window_id = getattr(src_plot, "window_id", None)
        self.params: dict[str, Any] = dict(DEFAULTS)
        self.model = None
        #: The Drift Check window (a bare figure) and its four plot handles.
        self.window_id: int | None = None
        self._panels: dict[str, Any] = {}
        self._sum_indices: np.ndarray | None = None
        self._before_sum: np.ndarray | None = None

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

    # ── the check window ─────────────────────────────────────────────────────

    def open_check_window(self, before: np.ndarray, n_frames: int) -> None:
        """Emit the bare-figure Drift Check window and register this controller
        for it, so ✕ and ``Session._forget_window`` reach the wizard."""
        import anyplotlib as apl
        import anyplotlib._electron as _electron
        from spyde.actions.figure_registry import keep_alive
        from spyde.drawing.plots.plot import finalize_figure_html

        fig, axes = apl.subplots(2, 2)
        ax = np.array(axes, dtype=object).ravel()
        zeros = np.zeros_like(before)
        t_axis = np.arange(max(1, n_frames), dtype=np.float64)
        nan_trace = np.zeros(max(1, n_frames), dtype=np.float32)

        panels = {
            "before": ax[0].imshow(before.astype(np.float32), cmap="gray"),
            "after": ax[1].imshow(zeros.astype(np.float32), cmap="gray"),
            "dy": ax[2].plot(nan_trace, axes=[t_axis], label="dy"),
            "dx": ax[3].plot(nan_trace, axes=[t_axis], label="dx"),
        }
        for key, title in (("before", "Raw sum"), ("after", "Corrected sum"),
                           ("dy", "dy (px)"), ("dx", "dx (px)")):
            try:
                panels[key].set_title(title)
            except Exception as exc:
                log.debug("[drift] set_title(%s) failed: %s", key, exc)

        wid = self.session.next_window_id()
        fig_id = _electron.register(fig)
        html = finalize_figure_html(fig, fig_id)
        keep_alive(int(wid), fig)
        emit({"type": "figure", "fig_id": fig_id, "window_id": int(wid),
              "html": html, "title": "Drift Check", "is_navigator": False})
        self.window_id = int(wid)
        self._panels = panels
        self._before_sum = before
        self.own_window(wid)

    def update_check(self, *, after=None, shifts=None) -> None:
        """Live-update the check window in place (main thread only)."""
        if not self._panels:
            return
        if after is not None:
            try:
                self._panels["after"].set_data(np.asarray(after, np.float32))
            except Exception as exc:
                log.debug("[drift] painting corrected sum failed: %s", exc)
        if shifts is not None:
            s = np.asarray(shifts, np.float32)
            for key, col in (("dy", 0), ("dx", 1)):
                try:
                    self._panels[key].set_data(
                        np.nan_to_num(s[:, col]).astype(np.float32),
                        x_axis=np.arange(s.shape[0], dtype=np.float64))
                except Exception as exc:
                    log.debug("[drift] painting %s trace failed: %s", key, exc)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def remove(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._panels = {}
        wid, self.window_id = self.window_id, None
        if wid is not None:
            forget = getattr(self.session, "_forget_window", None)
            if forget is not None:
                try:
                    forget(int(wid))
                except Exception as exc:
                    log.debug("[drift] forgetting check window failed: %s", exc)
            else:
                # Bare / stub session: emit + unregister by hand.
                try:
                    emit({"type": "window_closed", "window_id": int(wid)})
                except Exception as exc:
                    log.debug("[drift] closing check window failed: %s", exc)
                reg = getattr(self.session, "_window_controllers", None)
                if isinstance(reg, dict):
                    reg.pop(int(wid), None)
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
    return p


def _solver_kwargs(p: dict) -> dict:
    return dict(upsample=int(p["upsample"]), max_shift=float(p["max_shift"]),
                reference=str(p["reference"]), apodize=bool(p["apodize"]),
                normalize=bool(p["normalize"]),
                reject_outliers=bool(p["reject_outliers"]))


def _wizard(session, plot) -> DriftWizard | None:
    """Resolve the live wizard from either window.

    The Drift Check window is a bare figure, so ``_plot_by_window_id`` returns
    None for it and the plot-based lookup finds nothing — resolve by window id
    through the controller registry first (README §6), then fall back to the
    source tree's back-reference.
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
    msg = {"type": "drift_state",
           "window_id": wiz.src_window_id,
           "check_window_id": wiz.window_id,
           "method": wiz.params["method"],
           "solved": wiz.model is not None,
           "params": dict(wiz.params)}
    msg.update(extra)
    emit(msg)


# ── streaming sums (the check window's evidence) ─────────────────────────────

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


# ── staged handlers ──────────────────────────────────────────────────────────

def drift_open(session, plot, payload) -> None:
    """Caret mounted: build the controller and open the Drift Check window.

    Nothing solves here — plan A8 is explicit that drift correction is opt-in
    and never runs on load. The only compute is the raw sum, over at most
    ``_SUM_MAX_FRAMES`` frames, so the window opens with real evidence of the
    uncorrected blur instead of an empty panel.
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
        n_frames, get_frame, _shape = wiz.frames()
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
        _emit_state(wiz, n_frames=int(n_frames))
        emit_status("Drift Correction: tune the solver, then Solve.")

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
    if wiz is not None:
        # Harmlessly re-bumps when the tree resolved above; the point is the
        # case where it did not (the wizard was found through the check
        # window's controller registry).
        wiz.cancel_inflight()
        wiz.remove()


def drift_set_method(session, plot, payload) -> None:
    """Select the drift model.

    Only ``rigid`` has a solver in ``spyde.drift`` today. Selecting either of
    the others says so and stays on rigid — running a rigid solve while the
    caret claims "rigid+affine" would put a wrong ``kind`` into the model's
    provenance, which is worse than the missing feature.
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
    """Debounced parameter change → re-solve the FIRST PAIR only.

    Two frames, so it is two FFTs and lands inside a slider drag. It answers the
    only question a tune can answer cheaply — whether ``max_shift`` and
    ``upsample`` are in the right range for this movie — without paying for the
    stack. Generation-guarded: a superseded tune must not emit.
    """
    wiz = _wizard(session, plot)
    if wiz is None:
        return
    wiz.params = _coerce({**wiz.params, **(payload or {})})
    p = dict(wiz.params)
    gen = wiz.guard()

    def _work():
        from spyde.drift import solve_translation
        n_frames, get_frame, _shape = wiz.frames()
        if n_frames < 2:
            return None
        pair = [get_frame(0), get_frame(1)]
        kwargs = _solver_kwargs(p)
        # A 2-frame stack has no running average to speak of, so the reference
        # mode is forced to the only one that is meaningful for a pair.
        kwargs["reference"] = "first"
        return solve_translation(pair, **kwargs)

    def _done(model):
        if model is None or not wiz.still(gen) or wiz._closed:
            return
        dy, dx = (float(v) for v in model.shifts[1])
        emit({"type": "drift_preview", "window_id": wiz.src_window_id,
              "dy": dy, "dx": dx,
              "sharpness": float(model.residuals[1])
              if model.residuals is not None else float("nan"),
              "params": dict(p)})

    def _fail(exc):
        if wiz.still(gen):
            emit_error(f"Drift Correction preview failed: {exc}")

    run_on_worker(session, _work, name="drift-tune", on_done=_done, on_error=_fail)


def drift_run(session, plot, payload) -> None:
    """Solve the whole movie on a worker: progress-reported and cancellable.

    Cancellation goes through ``BaseSignalTree.register_cancel`` so closing the
    tree stops the solve, and ``solve_translation``'s own ``cancel()`` hook
    polls the same flag — a cancelled solve leaves NaN shifts for the frames it
    never reached, which is why a partial model is detectable rather than
    silently wrong.
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

    gen = wiz.guard()
    stopped = [False]
    if hasattr(tree, "register_cancel"):
        tree.register_cancel(flag=stopped)
    emit_status(f"Solving drift over {n_frames} frames…")

    def _work():
        from spyde.drift import solve_translation

        def _progress(done, total):
            emit_progress(int(done), int(total), "Drift")
            emit({"type": "drift_progress", "window_id": wiz.src_window_id,
                  "done": int(done), "total": int(total)})

        # Stream the curve as it solves. `progress` carries only a count and the
        # shift array is solver-local until the return, so without this callback
        # the caret can show a bar but not a trace. Batched rather than emitted
        # per frame: at thousands of frames one message each would flood the
        # PLOTAPP line protocol for a curve the eye cannot follow that finely.
        pending: list[tuple[int, float, float]] = []

        def _on_shift(i, dy, dx, _sharp):
            pending.append((int(i), float(dy), float(dx)))
            if len(pending) >= _TRACE_BATCH:
                emit({"type": "drift_trace", "window_id": wiz.src_window_id,
                      "points": pending[:]})
                pending.clear()

        model = solve_translation(
            wiz.signal(), progress=_progress, on_shift=_on_shift,
            cancel=lambda: stopped[0],
            provenance={"action": "Drift Correction", "params": dict(p)},
            **_solver_kwargs(p))
        if pending:
            emit({"type": "drift_trace", "window_id": wiz.src_window_id,
                  "points": pending[:]})
            pending.clear()
        if stopped[0]:
            return model, None
        # One extra streaming pass over the SAME bounded subset the raw sum
        # used, so the two check images are comparable.
        after = _stack_sum(get_frame, wiz.sum_indices(n_frames), model.shifts,
                           order=int(p["order"]))
        return model, after

    def _done(res):
        model, after = res
        try:
            if not wiz.still(gen) or wiz._closed:
                return
            wiz.model = model
            tree.drift = model
            wiz.update_check(after=after, shifts=model.shifts)
            emit({"type": "drift_result", "window_id": wiz.src_window_id,
                  "shifts": [[float(a), float(b)] for a, b in model.shifts],
                  "kind": model.kind, "reference": model.reference,
                  "max_abs_shift": float(model.max_abs_shift),
                  "rejected": int(model.params.get("rejected_from_reference", 0)),
                  "cancelled": bool(stopped[0])})
            _emit_state(wiz)
            solved = int(np.isfinite(model.shifts).all(axis=1).sum())
            if stopped[0]:
                emit_status(f"Drift solve cancelled after {solved} of "
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
             "kind": wiz.model.kind, "reference": wiz.model.reference})
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
