"""background_action.py — background removal from a dragged window (#64).

The user drags a span over a region that is background ONLY — the pre-edge
window in EELS, a line-free stretch in EDS — and the model is fitted to just
that span, then extrapolated across the whole axis and subtracted.

Why the window is the primary input rather than a typed pair of energies: which
stretch of a spectrum is "background" is a judgement about that spectrum, made
by looking at it. The numbers are a consequence, not the decision. (Same reason
Crop and Center Zero Beam take their geometry from a widget.)

Two things this gets right that a naive version does not:

* **The fit uses ONLY the window; the subtraction uses the whole axis.** That is
  the entire point of a background model — it says what the background would
  have been under the peaks, where it cannot be measured.
* **It fits every navigation position, not the displayed one.** A spectrum image
  has a different background per pixel (thickness varies), so subtracting one
  pixel's fit everywhere would leave a thickness map imprinted on the result.
  The batched engine makes this affordable.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.actions.context import src_plot_tree as _src_plot_tree
from de_shell.actions.wizard import WizardController
from de_shell import ipc

log = logging.getLogger(__name__)

# Offered background shapes. PowerLaw first: it is the EELS default and what a
# core-loss background actually looks like.
MODELS = ("PowerLaw", "Polynomial", "Offset", "Exponential")

PARAMETERS = {
    "model": {"name": "Background", "type": "enum", "default": "PowerLaw",
              "choices": list(MODELS)},
}


class BackgroundWizard(WizardController):
    key = "bg"
    parameters = PARAMETERS

    def __init__(self, session, tree, plot):
        super().__init__(session, tree)
        self.plot = plot
        self.model_kind = "PowerLaw"
        self.window: tuple[float, float] | None = None
        self._widget = None
        self._widget_cb = None
        self._line = None

    # ── the signal ────────────────────────────────────────────────────────
    @property
    def signal(self):
        return getattr(self.tree, "current_signal", None) or self.tree.root

    def axis(self) -> np.ndarray:
        return np.asarray(self.signal.axes_manager.signal_axes[0].axis, float)

    def default_window(self) -> tuple[float, float]:
        """The first fifth of the axis — in EELS that is pre-edge by
        construction, and it is somewhere to start rather than a claim."""
        x = self.axis()
        lo, hi = float(x[0]), float(x[-1])
        return lo, lo + 0.2 * (hi - lo)

    # ── the dragged window ────────────────────────────────────────────────
    def add_widget(self) -> None:
        p1 = getattr(self.plot, "_plot1d", None)
        if p1 is None:
            return
        from spyde.drawing.selectors.base_selector import event_handler_fn
        x0, x1 = self.window or self.default_window()
        try:
            self._widget = p1.add_range_widget(x0=x0, x1=x1, color="#a6e3a1")
            self._widget_cb = event_handler_fn(self._on_drag)
            self._widget.add_event_handler(self._widget_cb, "pointer_move",
                                           "pointer_up")
            self.window = (x0, x1)
        except Exception as e:
            log.debug("adding the background window failed: %s", e)

    def _on_drag(self, event) -> None:
        """The band moved — refit and redraw, live.

        The geometry is on ``event.source``, NOT on the event. anyplotlib's
        ``Event`` carries x/y and a handful of scalars; a range widget's
        ``x0``/``x1`` live on the widget it hands back. Reading them off the
        event gave None every time, so this returned at the first check and the
        background preview never followed the band at all — the same mistake
        the Fit handles made (see ``_on_widget_drag``).
        """
        src = getattr(event, "source", None) or event
        get = (lambda k: src.get(k) if isinstance(src, dict)
               else getattr(src, k, None))
        x0, x1 = get("x0"), get("x1")
        if x0 is None or x1 is None:
            return
        self.window = (min(float(x0), float(x1)), max(float(x0), float(x1)))
        # A pointer_move redraws the curve; the state message waits for the
        # release. Sending the whole state per pointer frame is what made the
        # Fit caret lag behind its own handles.
        live = str(getattr(event, "event_type", "") or "") != "pointer_up"
        self.preview(live=live)

    # ── fit + preview ─────────────────────────────────────────────────────
    def build_spec(self):
        """A one-component model, restricted to the dragged window.

        The channel mask is what makes the fit use only the window; the same
        spec then evaluates over the FULL axis to give the curve to subtract.
        """
        from spyde.fitting import ModelSpec
        from spyde.actions.fit_action import new_component_spec

        # Shared with the Fit caret's picker: the spec for a kind is read off
        # one prototype per process, not rebuilt on every window drag.
        cspec = new_component_spec(
            self.model_kind, 2 if self.model_kind == "Polynomial" else None)
        spec = ModelSpec(components=[cspec])

        x = self.axis()
        lo, hi = self.window or self.default_window()
        spec.channel_mask = (x >= lo) & (x <= hi)
        if spec.channel_mask.sum() < 4:
            raise ValueError("the background window covers fewer than 4 "
                             "channels — drag it wider")

        from spyde.actions.fit_action import scale_to_data
        # Seed from the data INSIDE the window, not the whole spectrum: a
        # pre-edge window is far below the peak, so scaling to the global max
        # would start the background an order of magnitude too high.
        inside = self.current_spectrum()[spec.channel_mask]
        if self.model_kind == "PowerLaw":
            # `origin` is the reference point, and 0 is HyperSpy's convention
            # and what every downstream tool assumes. But the curve is drawn —
            # and subtracted — across the WHOLE axis, not just the window, so
            # if the axis reaches the origin the extrapolation is a
            # singularity: a window on the far side of a peak fitted to 3e9 at
            # the left edge, which blew the plot's y-range away and hid the
            # spectrum entirely. Same rule as `fit_action.seed_background`.
            cspec["origin"].value = (
                0.0 if float(x[0]) > 0.0
                else float(x[0]) - 0.25 * (float(x[-1]) - float(x[0])))
            cspec["origin"].free = False
            cspec["r"].value = 3.0
        scale_to_data(cspec, x[spec.channel_mask], inside, fraction=1.0)
        return spec

    def current_spectrum(self) -> np.ndarray:
        data = np.asarray(self.signal.data, float)
        idx = getattr(self.plot, "current_indices", None)
        try:
            if idx is not None and data.ndim > 1:
                return data[tuple(int(i) for i in idx)]
        except Exception as e:
            log.debug("reading the current spectrum failed: %s", e)
        return data.reshape(-1, data.shape[-1]).mean(0)

    def fit_background(self, whole_scan: bool):
        """Fit the background. ``whole_scan`` fits every position (Apply);
        otherwise just the displayed spectrum (the live preview)."""
        import torch
        from spyde.fitting import components as tcomp
        from spyde.fitting.engine import fit_batched

        spec = self.build_spec()
        x = self.axis()
        data = (np.asarray(self.signal.data, float) if whole_scan
                else self.current_spectrum()[None, :])
        result = fit_batched(spec, data, x, device="cpu" if not whole_scan else None,
                             max_iter=80)
        values = torch.as_tensor(result.values)
        curve = tcomp.evaluate(spec, torch.as_tensor(x), values).numpy()
        return spec, result, curve

    def preview(self, live: bool = False) -> None:
        p1 = getattr(self.plot, "_plot1d", None)
        if p1 is None:
            return
        try:
            _spec, _res, curve = self.fit_background(whole_scan=False)
        except Exception as e:
            if not live:            # a mid-drag window can be briefly invalid
                ipc.emit_status(f"Background: {e}")
            return
        y = np.asarray(curve[0], np.float32)
        try:
            if self._line is None:
                self._line = p1.add_line(y, x_axis=self.axis(),
                                         label="background",
                                         color="#a6e3a1", linewidth=1.8)
            else:
                # UPDATE IN PLACE. Removing and re-adding the line every drag
                # frame is heavy AND does not repaint during the drag — the
                # curve simply does not follow the band. Same lesson as the fit
                # preview's `rebuild_lines` vs `refresh_lines`.
                self._line.set_data(y)
        except Exception as e:
            log.debug("drawing the background preview failed: %s", e)
        if not live:
            self.emit_state()

    def emit_state(self, status: str | None = None) -> None:
        lo, hi = self.window or self.default_window()
        ipc.emit({"type": "bg_state",
                  "window_id": getattr(self.plot, "window_id", None),
                  "model": self.model_kind, "x0": float(lo), "x1": float(hi),
                  "status": status})

    # ── teardown ──────────────────────────────────────────────────────────
    def remove(self) -> None:
        if self._closed:
            return
        self._closed = True
        p1 = getattr(self.plot, "_plot1d", None)
        try:
            if p1 is not None and self._line is not None:
                p1.remove_line(self._line)
            if p1 is not None and self._widget is not None:
                p1.remove_widget(self._widget.id)
        except Exception as e:
            log.debug("background teardown failed: %s", e)
        self._line = self._widget = None
        if getattr(self.tree, "_bg_wizard", None) is self:
            self.tree._bg_wizard = None


def _wizard(session, plot):
    src, tree = _src_plot_tree(session, plot)
    return (getattr(tree, "_bg_wizard", None) if tree is not None else None), tree


def bg_toolbar(ctx, action_name: str = "Remove Background", **params) -> None:
    bg_open(ctx.session, ctx.plot, params or {})


def bg_open(session, plot, payload=None) -> None:
    src, tree = _src_plot_tree(session, plot)
    if src is None or tree is None:
        ipc.emit_error("Remove Background: no active dataset")
        return
    wiz = getattr(tree, "_bg_wizard", None)
    if wiz is not None and not wiz._closed:
        wiz.emit_state()
        return
    wiz = BackgroundWizard(session, tree, src)
    wiz.guard()
    tree._bg_wizard = wiz
    wiz.add_widget()
    wiz.preview()
    wiz.emit_state("Drag the green band over a background-only region.")


def bg_close(session, plot, payload=None) -> None:
    wiz, _tree = _wizard(session, plot)
    if wiz is not None:
        wiz.cancel_inflight()
        wiz.remove()


def bg_set_model(session, plot, payload) -> None:
    wiz, _tree = _wizard(session, plot)
    if wiz is None:
        return
    kind = (payload or {}).get("model")
    if kind not in MODELS:
        ipc.emit_error(f"Remove Background: unknown model {kind!r}")
        return
    wiz.model_kind = kind
    wiz.preview()
    wiz.emit_state(f"Background model: {kind}.")


def bg_set_region(session, plot, payload) -> None:
    """Typed edit of the window — keeps the widget in step with the fields."""
    wiz, _tree = _wizard(session, plot)
    if wiz is None:
        return
    p = payload or {}
    try:
        lo, hi = float(p["x0"]), float(p["x1"])
    except (KeyError, TypeError, ValueError):
        return
    wiz.window = (min(lo, hi), max(lo, hi))
    try:
        if wiz._widget is not None:
            wiz._widget.set_range(wiz.window[0], wiz.window[1])
    except Exception as e:
        log.debug("moving the background widget failed: %s", e)
    wiz.preview()


def bg_apply(session, plot, payload=None) -> None:
    """Fit EVERY position and subtract, into a new node on the same tree."""
    wiz, tree = _wizard(session, plot)
    if wiz is None:
        ipc.emit_error("Remove Background: open the tool first")
        return

    from spyde.actions.lifecycle import supersede
    handle = supersede(getattr(wiz, "_apply_compute", None), tree)
    wiz._apply_compute = handle

    def _work():
        if handle.stopped:
            return None
        _spec, result, curve = wiz.fit_background(whole_scan=True)
        data = np.asarray(wiz.signal.data, float)
        return data - curve.reshape(data.shape), result

    def _done(out):
        handle.retire()
        if out is None or handle.stopped or wiz._closed:
            return
        subtracted, result = out
        try:
            import hyperspy.api as hs
            new = wiz.signal.deepcopy()
            new.data = subtracted.astype(wiz.signal.data.dtype, copy=False)
            new.metadata.General.title = (
                f"{getattr(wiz.signal.metadata.General, 'title', 'signal')} "
                f"− {wiz.model_kind} background")
            tree.add_node(wiz.signal, new, "background removed")
            tree.update_plot_states(new)
            session._reemit_signal_tree(tree)
        except Exception as e:
            ipc.emit_error(f"Remove Background: could not add the result ({e})")
            return
        pct = 100.0 * result.convergence_rate
        ipc.emit_status(f"Background removed — {pct:.0f}% of positions converged")
        wiz.emit_state("Background removed into a new node.")

    ipc.emit_status("Fitting the background over every position…")
    wiz.run_on_worker(_work, name="bg-apply", on_done=_done)
