"""fit_action.py — the Fit wizard (#55, #56, #58).

A staged caret over :mod:`spyde.fitting`. "Add a fit" opens a caret group where
components are added line by line; each edit redraws a live model curve on the
spectrum; Run fits every navigation position in one batched call; Commit turns
each component's integrated area into a map.

Follows the wizard protocol in ``spyde/actions/README.md`` §2 exactly — the
generation guard is opened in ``fit_open`` before any worker and bumped FIRST
in ``fit_close``, because React StrictMode fires open/close/open synchronously.

Three things specific to this wizard:

* **The preview is ONE spectrum, not the grid.** A model edit has to feel
  instant, and fitting 65k positions on every keystroke would not. The live
  curve is the model evaluated at the current navigator position; ``fit_run``
  is the only thing that touches the whole scan.
* **The component catalogue ships its SHAPES** (#56). "Add component" shows
  what each one looks like rather than only its name, so the backend samples
  every available component at default parameters over the current axis and
  sends a small polyline the renderer draws as a sparkline.
* **Commit produces one map per component** (#58) through
  ``commit.commit_result_tree``, which already gives the strain-style toggle
  (click one, cmd-click to tile). No new display code.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from spyde.actions.commit import commit_result_tree
from spyde.actions.context import src_plot_tree as _src_plot_tree
from spyde.actions.wizard import WizardController
# Imported as a MODULE, not `from ... import emit`. The test fixture patches
# `ipc.emit` to capture outgoing messages, and a from-import binds the original
# function at import time — the patch would then never be seen and every
# message assertion would silently observe nothing.
from spyde.backend import ipc

log = logging.getLogger(__name__)

_SQRT_2PI = math.sqrt(2.0 * math.pi)

# Offered in the picker, in the order a user is likely to want them: the
# backgrounds first, then the peak shapes, then the steps.
CATALOGUE = [
    ("Offset", "Flat background"),
    ("PowerLaw", "Power-law background (EELS)"),
    ("Polynomial", "Polynomial background"),
    ("Exponential", "Exponential decay"),
    ("Gaussian", "Gaussian peak"),
    ("Lorentzian", "Lorentzian peak"),
    ("GaussianHF", "Gaussian (height / FWHM)"),
    ("Erf", "Smeared step"),
    ("Arctan", "Arctan step"),
]

_PREVIEW_POINTS = 64

# How each component maps onto the on-plot drag handles (#57).
#
#   pos    the parameter the POINT widget's x drives
#   width  the parameter the RANGE widget's span drives
#   amp    the parameter the POINT widget's y drives
#   kind   how amp relates to the curve's HEIGHT at the peak, because for most
#          components the fitted amplitude is an AREA, not a height — dragging
#          a handle to y and storing y as `A` would jump the curve by a factor
#          of sigma*sqrt(2*pi).
#
# A component with no `pos` (a background) gets no handles: there is nothing on
# the plot to point at.
_DRAG = {
    "Gaussian":   {"pos": "centre", "width": "sigma", "amp": "A", "amp_kind": "area_gauss"},
    "Lorentzian": {"pos": "centre", "width": "gamma", "amp": "A", "amp_kind": "area_lorentz"},
    "GaussianHF": {"pos": "centre", "width": "fwhm", "amp": "height", "amp_kind": "height"},
    "Erf":        {"pos": "origin", "width": "sigma", "amp": "A", "amp_kind": "height"},
    "Arctan":     {"pos": "x0", "width": None, "amp": "A", "amp_kind": "height"},
}

# Half-width of the RANGE widget, per width parameter, so the band a user drags
# corresponds to something they recognise (a Gaussian's band is its FWHM).
_WIDTH_TO_HALF = {"sigma": 1.1774, "gamma": 1.0, "fwhm": 0.5}


def _amp_from_height(kind_info, height: float, width: float) -> float:
    """Curve height at the peak -> the component's amplitude parameter."""
    k = kind_info["amp_kind"]
    if k == "area_gauss":
        return float(height) * max(width, 1e-9) * _SQRT_2PI
    if k == "area_lorentz":
        return float(height) * math.pi * max(width, 1e-9)
    return float(height)


def _height_from_amp(kind_info, amp: float, width: float) -> float:
    k = kind_info["amp_kind"]
    if k == "area_gauss":
        return float(amp) / max(width * _SQRT_2PI, 1e-9)
    if k == "area_lorentz":
        return float(amp) / max(math.pi * width, 1e-9)
    return float(amp)


class FitWizard(WizardController):
    key = "fit"

    parameters = {
        "max_iter": {"name": "Max iterations", "type": "int", "default": 60,
                     "min": 5, "max": 500, "step": 5},
        "seeded": {"name": "Seeded propagation", "type": "bool",
                   "default": True},
        "weighting": {"name": "Weighting", "type": "enum", "default": "none",
                      "choices": ["none", "poisson"]},
    }

    def __init__(self, session, tree, plot):
        super().__init__(session, tree)
        from spyde.fitting import ModelSpec
        self.plot = plot
        # The MODEL and the FIT live on the TREE, not on this controller.
        # They are RESULTS (the ownership map in actions/README.md §3 puts
        # results on the tree and controllers alongside them), and a fit is
        # expensive — closing the caret to get it out of the way should not
        # throw away a model the user spent minutes building or a scan that
        # took a minute to fit. `BaseSignalTree.close()` still disposes both.
        if getattr(tree, "fit_spec", None) is None:
            tree.fit_spec = ModelSpec()
        self._preview_label = "fit_preview"
        self._preview_line = None
        # widget id -> (component name, role) for the on-plot drag handles.
        self._widgets: dict = {}
        self._widget_cb = None

    @property
    def spec(self):
        return self.tree.fit_spec

    @spec.setter
    def spec(self, value):
        self.tree.fit_spec = value

    @property
    def result(self):
        return getattr(self.tree, "fit_result", None)

    @result.setter
    def result(self, value):
        self.tree.fit_result = value

    # ── the signal being fitted ───────────────────────────────────────────
    @property
    def signal(self):
        return getattr(self.tree, "current_signal", None) or self.tree.root

    def axis(self) -> np.ndarray:
        return np.asarray(self.signal.axes_manager.signal_axes[0].axis, float)

    def current_spectrum(self) -> np.ndarray:
        """The spectrum under the navigator right now — what the preview fits.

        Falls back to the mean over navigation when there is no navigator
        position yet, which is better than an arbitrary corner pixel.
        """
        data = self.signal.data
        idx = getattr(self.plot, "current_indices", None)
        try:
            if idx is not None and np.ndim(data) > 1:
                return np.asarray(data[tuple(int(i) for i in idx)], float)
        except Exception as e:
            log.debug("reading the current spectrum failed: %s", e)
        arr = np.asarray(data, float)
        return arr.reshape(-1, arr.shape[-1]).mean(0)

    # ── live preview ──────────────────────────────────────────────────────
    def draw_preview(self) -> None:
        """Redraw the model curve on the spectrum plot."""
        p1 = getattr(self.plot, "_plot1d", None)
        if p1 is None or not len(self.spec):
            return
        try:
            import torch
            from spyde.fitting import components as tcomp
            x = self.axis()
            values = torch.as_tensor(self.spec.flat_values()[None, :],
                                     dtype=torch.float64)
            y = tcomp.evaluate(self.spec, torch.as_tensor(x), values).numpy()[0]
        except Exception as e:
            log.debug("evaluating the fit preview failed: %s", e)
            return
        try:
            # `label`, NOT `name` — Plot1D.add_line takes `label`, and passing
            # `name` raised a TypeError this method's own except swallowed, so
            # the preview silently never drew.
            self.clear_preview()
            self._preview_line = p1.add_line(
                np.asarray(y, np.float32), x_axis=self.axis(),
                label=self._preview_label, color="#f5a97f", linewidth=1.8)
        except Exception as e:
            log.debug("drawing the fit preview failed: %s", e)

    def clear_preview(self) -> None:
        """Remove the previous preview line.

        Keeps the HANDLE ``add_line`` returned — ``remove_line`` takes an id or
        a ``Line1D``, NOT the label. Passing the label raises a KeyError that
        was swallowed here, so every redraw stacked another line and the legend
        filled up with `fit_preview` entries.
        """
        p1 = getattr(self.plot, "_plot1d", None)
        line = getattr(self, "_preview_line", None)
        if p1 is None or line is None:
            return
        try:
            p1.remove_line(line)
        except Exception as e:
            log.debug("clearing the fit preview failed: %s", e)
        finally:
            self._preview_line = None

    # ── on-plot drag handles (#57) ────────────────────────────────────────
    def sync_widgets(self) -> None:
        """Put a POINT handle (centre + height) and a RANGE handle (width) on
        each component that has a position on the axis.

        Rebuilt wholesale rather than diffed: a model is a handful of
        components, and reconciling handle-to-component identity across an
        add/remove is exactly the kind of bookkeeping that leaves an orphaned
        handle dragging a component that no longer exists.
        """
        p1 = getattr(self.plot, "_plot1d", None)
        if p1 is None:
            return
        self.clear_widgets()
        from spyde.drawing.selectors.base_selector import event_handler_fn
        self._widget_cb = event_handler_fn(self._on_widget_drag)

        for comp in self.spec.active_components:
            info = _DRAG.get(comp.kind)
            if info is None:
                continue                      # a background has nothing to point at
            try:
                pos = float(comp[info["pos"]].value)
                width = (float(comp[info["width"]].value)
                         if info["width"] else (self.axis().ptp() / 20.0))
                height = _height_from_amp(info, float(comp[info["amp"]].value),
                                          width)
                pw = p1.add_point_widget(x=pos, y=height, color="#f5a97f")
                self._widgets[pw.id] = (comp.name, "point")
                pw.add_event_handler(self._widget_cb, "pointer_move", "pointer_up")
                if info["width"]:
                    half = width * _WIDTH_TO_HALF.get(info["width"], 1.0)
                    rw = p1.add_range_widget(x0=pos - half, x1=pos + half,
                                             color="#89b4fa", y=height / 2)
                    self._widgets[rw.id] = (comp.name, "range")
                    rw.add_event_handler(self._widget_cb, "pointer_move",
                                         "pointer_up")
            except Exception as e:
                log.debug("adding drag handles for %s failed: %s", comp.name, e)

    def update_widgets(self, skip_id=None) -> None:
        """MOVE the existing handles to match the model, in place.

        Distinct from :meth:`sync_widgets`, which tears them down and rebuilds.
        Rebuilding on every parameter keystroke destroys and recreates every
        widget several times a second — the handles flicker, a widget the user
        is reaching for vanishes under the cursor, and the plot spends its time
        re-pushing widget state instead of redrawing the curve. Rebuild only
        when the component LIST changes; move them otherwise.

        *skip_id* is the widget currently being dragged: writing a position
        back to the handle under the user's finger fights the drag.
        """
        p1 = getattr(self.plot, "_plot1d", None)
        if p1 is None:
            return
        by_name: dict = {}
        for wid, (name, role) in self._widgets.items():
            by_name.setdefault(name, {})[role] = wid

        for comp in self.spec.active_components:
            info = _DRAG.get(comp.kind)
            roles = by_name.get(comp.name)
            if info is None or not roles:
                continue
            try:
                pos = float(comp[info["pos"]].value)
                width = (float(comp[info["width"]].value)
                         if info["width"] else (self.axis().ptp() / 20.0))
                height = _height_from_amp(info, float(comp[info["amp"]].value),
                                          width)
                pid = roles.get("point")
                if pid is not None and pid != skip_id:
                    w = p1.get_widget(pid)
                    if w is not None:
                        w.set(x=pos, y=height)
                rid = roles.get("range")
                if rid is not None and rid != skip_id and info["width"]:
                    half = width * _WIDTH_TO_HALF.get(info["width"], 1.0)
                    w = p1.get_widget(rid)
                    if w is not None:
                        w.set(x0=pos - half, x1=pos + half, y=height / 2)
            except Exception as e:
                log.debug("moving handles for %s failed: %s", comp.name, e)

    def clear_widgets(self) -> None:
        p1 = getattr(self.plot, "_plot1d", None)
        for wid in list(self._widgets):
            try:
                if p1 is not None:
                    p1.remove_widget(wid)
            except Exception as e:
                log.debug("removing fit widget %s failed: %s", wid, e)
        self._widgets.clear()

    def _on_widget_drag(self, event) -> None:
        """A handle moved — write it back into the model and redraw.

        Handles are the PRIMARY input for shape (the anyplotlib
        interactive-fitting example's idiom): the numeric fields stay editable
        and are refreshed from here, so both directions agree.
        """
        try:
            wid = getattr(event, "id", None) or (event or {}).get("id")
            target = self._widgets.get(wid)
            if target is None:
                return
            name, role = target
            comp = self.spec[name]
            info = _DRAG.get(comp.kind)
            if info is None:
                return
            get = (lambda k: getattr(event, k, None)
                   if not isinstance(event, dict) else event.get(k))

            if role == "point":
                x, y = get("x"), get("y")
                if x is not None:
                    comp[info["pos"]].value = float(x)
                if y is not None and info["amp"]:
                    width = (float(comp[info["width"]].value)
                             if info["width"] else 1.0)
                    comp[info["amp"]].value = _amp_from_height(info, float(y),
                                                               width)
            else:
                x0, x1 = get("x0"), get("x1")
                if x0 is not None and x1 is not None and info["width"]:
                    half = abs(float(x1) - float(x0)) / 2.0
                    factor = _WIDTH_TO_HALF.get(info["width"], 1.0)
                    # Keep the AMPLITUDE fixed as the width changes: for an
                    # area-parameterised component, changing sigma alone would
                    # otherwise change the peak HEIGHT under the user's cursor,
                    # which reads as the curve fighting the drag.
                    height = _height_from_amp(
                        info, float(comp[info["amp"]].value),
                        float(comp[info["width"]].value))
                    comp[info["width"]].value = max(half / max(factor, 1e-9),
                                                    1e-6)
                    comp[info["amp"]].value = _amp_from_height(
                        info, height, float(comp[info["width"]].value))
            self.result = None            # the old fit no longer describes it
            self.draw_preview()
            self.update_widgets(skip_id=wid)   # never fight the dragged handle
            self.emit_state()
        except Exception as e:
            log.debug("fit widget drag failed: %s", e)

    def emit_state(self, status: str | None = None) -> None:
        """Send the whole model to the caret — components, parameters, values.

        One message rather than incremental patches: a model is small, and a
        renderer that rebuilds from the truth cannot drift out of step with the
        backend after a failed edit.
        """
        ipc.emit({
            "type": "fit_state",
            "window_id": getattr(self.plot, "window_id", None),
            "components": [
                {"name": c.name, "kind": c.kind, "active": bool(c.active),
                 "parameters": [
                     {"name": p.name, "value": float(p.value),
                      "free": bool(p.free), "linear": bool(p.linear)}
                     for p in c.scalar_parameters]}
                for c in self.spec.components],
            "fitted": self.result is not None,
            "status": status,
        })

    # ── teardown ──────────────────────────────────────────────────────────
    def remove(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.clear_preview()
        self.clear_widgets()
        # The MODEL and the RESULT deliberately SURVIVE — they live on the
        # tree, so reopening the caret restores what the user built. Only the
        # controller and its on-plot artefacts go.
        if getattr(self.tree, "_fit_wizard", None) is self:
            self.tree._fit_wizard = None

    # ── commit ────────────────────────────────────────────────────────────
    def commit(self):
        if self.result is None or self.session is None:
            ipc.emit_error("Fit: run the fit before committing")
            return None
        maps = component_area_maps(self.spec, self.result, self.axis(),
                                   self.nav_shape())
        if not maps:
            ipc.emit_error("Fit: no component produced a map")
            return None
        names = list(maps)
        return commit_result_tree(
            self.session, title="Fit components",
            primary=maps[names[0]], primary_label=names[0],
            views=[(n, maps[n]) for n in names[1:]],
            levels=None, cmap="viridis",
            attrs={"fit_spec": self.spec, "fit_result": self.result},
            provenance={"action": "Fit",
                        "params": {"components": [c.kind for c in self.spec]},
                        "source_title": getattr(
                            self.signal.metadata.General, "title", "")},
        )

    def nav_shape(self):
        try:
            nav = tuple(int(n) for n in
                        self.signal.axes_manager.navigation_shape)
            return tuple(reversed(nav))
        except Exception:
            return None


def component_area_maps(spec, result, x, nav_shape=None) -> dict[str, np.ndarray]:
    """Integrated area under each component, per navigation position (#58).

    Each component is evaluated ALONE with the fitted parameters and integrated
    over the signal axis. Area rather than peak height: it is what scales with
    how much of a thing is present, and it is insensitive to a slightly wider
    or narrower fit, so two positions are comparable.
    """
    import torch
    from spyde.fitting import components as tcomp

    xt = torch.as_tensor(np.asarray(x, float))
    values = torch.as_tensor(np.asarray(result.values, float))
    out: dict[str, np.ndarray] = {}
    i = 0
    for c in spec.active_components:
        n = len(c.scalar_parameters)
        try:
            comp = tcomp.component_for(c) if hasattr(tcomp, "component_for") \
                else tcomp.get_component(c.kind, n_params=n)
            y = comp(xt, values[:, i:i + n]).numpy()
            area = np.trapezoid(y, np.asarray(x, float), axis=1) \
                if hasattr(np, "trapezoid") else np.trapz(y, x, axis=1)
            out[c.name] = (area.reshape(nav_shape) if nav_shape else area)
        except Exception as e:
            log.debug("area map for %s failed: %s", c.name, e)
        i += n
    return out


def component_catalogue(x: np.ndarray) -> list[dict]:
    """Every offerable component with a sampled SHAPE (#56).

    The picker shows what a component looks like, not just its name. Each
    preview is the component at defaults over the CURRENT axis, normalised —
    the sparkline is about shape, and an un-normalised power law would render
    as a spike beside a flat gaussian.
    """
    import torch
    from spyde.fitting import components as tcomp
    from spyde.fitting.spec import spec_from_component

    lo, hi = float(np.min(x)), float(np.max(x))
    xs = np.linspace(lo, hi, _PREVIEW_POINTS)
    out = []
    for kind, description in CATALOGUE:
        try:
            import hyperspy.components1d as c1d
            comp = (c1d.Polynomial(order=2) if kind == "Polynomial"
                    else getattr(c1d, kind)())
            cspec = spec_from_component(comp)
            _seed_for_preview(cspec, lo, hi)
            n = len(cspec.scalar_parameters)
            batched = tcomp.get_component(kind, n_params=n)
            vals = np.array([[p.value for p in cspec.scalar_parameters]])
            y = batched(torch.as_tensor(xs), torch.as_tensor(vals)).numpy()[0]
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            span = float(np.ptp(y))
            y = (y - y.min()) / span if span > 0 else np.zeros_like(y)
            out.append({"kind": kind, "description": description,
                        "preview": [round(float(v), 4) for v in y]})
        except Exception as e:
            log.debug("preview for %s failed: %s", kind, e)
    return out


def scale_to_data(cspec, x: np.ndarray, y: np.ndarray, fraction: float = 0.5) -> None:
    """Set a component's LINEAR amplitude so it is visible against the data.

    Without this a new component arrives with an amplitude of 1 against counts
    of 1e5 — the preview curve is a flat line on the axis, the drag handles sit
    at y=0, and the fit starts five orders of magnitude away from its answer.
    All three read as "the model does nothing".

    Generic rather than per-component: evaluate the component with its linear
    parameter at 1, then scale so its peak reaches *fraction* of the data's
    range. Works for any component the batched library can evaluate, including
    ones added later.
    """
    import torch
    from spyde.fitting import components as tcomp

    linear = next((p for p in cspec.scalar_parameters if p.linear), None)
    if linear is None:
        return
    try:
        comp = tcomp.get_component(cspec.kind,
                                   n_params=len(cspec.scalar_parameters))
        original, linear.value = linear.value, 1.0
        vals = np.array([[p.value for p in cspec.scalar_parameters]])
        unit = comp(torch.as_tensor(np.asarray(x, float)),
                    torch.as_tensor(vals)).numpy()[0]
        unit = np.nan_to_num(unit, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(unit)))
        target = float(np.nanmax(np.asarray(y, float))) * float(fraction)
        linear.value = (target / peak) if peak > 0 else original
    except Exception as e:
        log.debug("scaling %s to the data failed: %s", cspec.kind, e)


def _seed_for_preview(cspec, lo: float, hi: float) -> None:
    """Put a component somewhere visible on THIS axis.

    A default Gaussian sits at 0 with sigma 1, which is off-screen on a 200-800
    eV axis and would preview as a flat line — the picker would then show every
    peak shape as identical nothing.
    """
    mid, width = (lo + hi) / 2, (hi - lo) / 8
    for p in cspec.parameters:
        if p.name in ("centre", "origin", "x0"):
            p.value = mid
        elif p.name in ("sigma", "gamma", "fwhm"):
            p.value = width
        elif p.name == "tau":
            p.value = max(hi / 3.0, 1.0)
        elif p.name == "k":
            p.value = 4.0 / max(hi - lo, 1.0)
        elif p.name in ("A", "height", "a", "intensity"):
            p.value = 1.0
        elif p.name == "r":
            p.value = 3.0


# ─────────────────────────────────────────────────────────────────────────────
# staged handlers — fn(session, plot, payload)
# ─────────────────────────────────────────────────────────────────────────────

def fit_toolbar(ctx, action_name: str = "Fit", **params) -> None:
    """Toolbar entry — the Electron toolbar opens the caret, which sends
    ``fit_open``. This exists so the YAML has a resolvable ``function:``
    (see ``vector_orientation_mapping`` for the same no-op parent pattern)."""
    fit_open(ctx.session, ctx.plot, params or {})


def _wizard(session, plot):
    src, tree = _src_plot_tree(session, plot)
    return (getattr(tree, "_fit_wizard", None) if tree is not None else None), tree


def fit_open(session, plot, payload=None) -> None:
    src, tree = _src_plot_tree(session, plot)
    if src is None or tree is None:
        ipc.emit_error("Fit: no active dataset")
        return
    wiz = getattr(tree, "_fit_wizard", None)
    if wiz is not None and not wiz._closed:
        wiz.emit_state()                                  # idempotent re-open
        return
    wiz = FitWizard(session, tree, src)
    wiz.guard()
    tree._fit_wizard = wiz
    try:
        ipc.emit({"type": "fit_catalogue",
              "window_id": getattr(src, "window_id", None),
              "components": component_catalogue(wiz.axis())})
    except Exception as e:
        log.debug("emitting the component catalogue failed: %s", e)
    wiz.draw_preview()
    wiz.sync_widgets()
    wiz.emit_state("Add a component to begin." if not len(wiz.spec)
                   else "Model restored.")


def fit_close(session, plot, payload=None) -> None:
    wiz, _tree = _wizard(session, plot)
    if wiz is not None:
        wiz.cancel_inflight()
        wiz.remove()


def fit_add_component(session, plot, payload) -> None:
    wiz, _tree = _wizard(session, plot)
    if wiz is None:
        return
    kind = (payload or {}).get("kind")
    if not kind:
        ipc.emit_error("Fit: no component kind given")
        return
    try:
        import hyperspy.components1d as c1d
        from spyde.fitting.spec import spec_from_component
        comp = (c1d.Polynomial(order=int((payload or {}).get("order", 2)))
                if kind == "Polynomial" else getattr(c1d, kind)())
        cspec = spec_from_component(comp)
    except Exception as e:
        ipc.emit_error(f"Fit: cannot add {kind} ({e})")
        return

    x = wiz.axis()
    _seed_for_preview(cspec, float(np.min(x)), float(np.max(x)))
    # Scale the amplitude to THIS spectrum, or the component arrives five
    # orders of magnitude below the data and looks like it does nothing.
    scale_to_data(cspec, x, wiz.current_spectrum())
    # A unique name per instance — two Gaussians must be separately addressable
    # by the caret and produce two distinct area maps at commit.
    existing = {c.name for c in wiz.spec.components}
    base, n = kind, 1
    while cspec.name in existing:
        n += 1
        cspec.name = f"{base} {n}"
    wiz.spec.append(cspec)
    wiz.result = None                       # the old fit no longer describes it
    wiz.draw_preview()
    wiz.sync_widgets()
    wiz.emit_state(f"Added {cspec.name}.")


def fit_from_composition(session, plot, payload) -> None:
    """Populate the model from the elements present (#65, on top of #62).

    The point of the wave: type "Fe, Ni, Cu" and get a model, then drag the
    lines where they belong. Everything after this call is the ordinary Fit
    caret — the drag handles from #57 work on an edge or an X-ray line exactly
    as they do on a hand-placed gaussian, which is why #65 is mostly wiring.

    EELS models are TABULATED on the way in (#63), because ``EELSCLEdge`` has no
    batched port: without that step the model is correct but falls back to
    HyperSpy's one-pixel-at-a-time fitting, which is the difference between
    seconds and minutes on a real scan.
    """
    wiz, tree = _wizard(session, plot)
    if wiz is None:
        ipc.emit_error("Fit: open the Fit caret first")
        return
    p = payload or {}
    raw = p.get("elements")
    elements = ([e.strip() for e in raw.replace(",", " ").split() if e.strip()]
                if isinstance(raw, str) else list(raw or []))

    try:
        from spyde.spectroscopy import (
            MissingExtra, model_for_composition, tabulate_model,
        )
    except ImportError:
        ipc.emit_error('Composition models need exspy — '
                       'pip install "spyde[eels]"')
        return

    try:
        spec, info = model_for_composition(wiz.signal, elements or None)
    except MissingExtra as e:
        ipc.emit_error(str(e))
        return
    except Exception as e:
        ipc.emit_error(f"Fit: could not build a model for {elements or 'this signal'} ({e})")
        return

    note = ""
    if not info.get("engine_supported") and bool(p.get("tabulate", True)):
        try:
            spec, tinfo = tabulate_model(spec, wiz.signal)
            if tinfo["tabulated"]:
                note = (f" {len(tinfo['tabulated'])} edge(s) tabulated — "
                        f"fine structure is fixed.")
        except Exception as e:
            log.info("tabulating the composition model failed (%s); the fit "
                     "will fall back to hyperspy", e)

    wiz.spec = spec
    wiz.result = None
    wiz.draw_preview()
    wiz.sync_widgets()
    dropped = info.get("dropped") or []
    wiz.emit_state(
        f"Built {len(spec)} components for {', '.join(info['elements'])}."
        + (f" {len(dropped)} outside the range dropped." if dropped else "")
        + note)


def fit_remove_component(session, plot, payload) -> None:
    wiz, _tree = _wizard(session, plot)
    if wiz is None:
        return
    name = (payload or {}).get("name")
    wiz.spec.components = [c for c in wiz.spec.components if c.name != name]
    wiz.result = None
    wiz.clear_preview()
    wiz.draw_preview()
    wiz.sync_widgets()
    wiz.emit_state(f"Removed {name}.")


def fit_set_param(session, plot, payload) -> None:
    wiz, _tree = _wizard(session, plot)
    if wiz is None:
        return
    p = payload or {}
    try:
        comp = wiz.spec[p["component"]]
        par = comp[p["parameter"]]
        if "value" in p:
            par.value = float(p["value"])
        if "free" in p:
            par.free = bool(p["free"])
    except (KeyError, TypeError, ValueError) as e:
        log.debug("fit_set_param %s failed: %s", p, e)
        return
    wiz.draw_preview()
    wiz.update_widgets()      # MOVE, do not rebuild — see update_widgets
    wiz.emit_state()


def fit_tune(session, plot, payload=None) -> None:
    """Debounced redraw — the caret's live edit path."""
    wiz, _tree = _wizard(session, plot)
    if wiz is not None:
        wiz.draw_preview()


def fit_current(session, plot, payload=None) -> None:
    """Fit ONLY the spectrum on screen — the iterate-quickly button.

    Building a model is a loop: place a component, see where it lands, nudge it.
    Fitting the whole scan to check one guess is the wrong unit of work — it
    costs seconds to minutes and answers a question about one pixel. This fits
    the displayed spectrum, writes the result back into the model, and redraws,
    so the next nudge starts from a fitted position.

    It is the same engine and the same spec as :func:`fit_run`; the only
    difference is that the data is one row.
    """
    wiz, tree = _wizard(session, plot)
    if wiz is None:
        ipc.emit_error("Fit: open the Fit caret first")
        return
    if not len(wiz.spec):
        ipc.emit_error("Fit: add at least one component first")
        return

    from spyde.fitting import components as tcomp
    if not tcomp.supports(wiz.spec):
        unsupported = sorted({c.kind for c in wiz.spec.active_components
                              if c.kind not in tcomp.available()})
        ipc.emit_error(f"Fit: {', '.join(unsupported)} has no batched "
                       f"implementation yet")
        return

    from spyde.fitting.engine import fit_batched
    try:
        res = fit_batched(wiz.spec, wiz.current_spectrum()[None, :], wiz.axis(),
                          device="cpu", max_iter=int((payload or {}).get(
                              "max_iter", 120)))
        # Write the fitted values back into the MODEL so the caret, the handles
        # and the next fit all start from them. This is the difference between
        # a preview and a step in the workflow.
        wiz.spec.set_flat_values(res.values[0])
    except Exception as e:
        ipc.emit_error(f"Fit: fitting this spectrum failed ({e})")
        return

    wiz.result = None          # a single-spectrum fit is NOT a scan result
    wiz.draw_preview()
    wiz.update_widgets()
    ok = "converged" if bool(res.converged[0]) else "did not converge"
    wiz.emit_state(f"This spectrum {ok} (chi2 {res.chisq[0]:.3g}).")


def fit_run(session, plot, payload=None) -> None:
    """Fit EVERY navigation position, batched, on a worker."""
    wiz, tree = _wizard(session, plot)
    if wiz is None:
        ipc.emit_error("Fit: open the Fit caret first")
        return
    if not len(wiz.spec):
        ipc.emit_error("Fit: add at least one component first")
        return

    from spyde.fitting import components as tcomp
    if not tcomp.supports(wiz.spec):
        unsupported = sorted({c.kind for c in wiz.spec.active_components
                              if c.kind not in tcomp.available()})
        ipc.emit_error(f"Fit: {', '.join(unsupported)} has no batched "
                   f"implementation yet")
        return

    p = payload or {}
    max_iter = int(p.get("max_iter", 60))
    seeded = bool(p.get("seeded", True))
    weights = p.get("weighting", "none")
    weights = None if weights in (None, "none") else weights

    spec = wiz.spec.copy()
    x = wiz.axis()
    data = np.asarray(wiz.signal.data, float)
    nav_shape = wiz.nav_shape()
    gen = wiz.guard()

    def _fit():
        from spyde.fitting.engine import fit_batched
        from spyde.fitting.seeding import fit_seeded
        fn = fit_seeded if (seeded and data.ndim > 2) else fit_batched
        return fn(spec, data, x, max_iter=max_iter, weights=weights,
                  progress=lambda d, t: ipc.emit_status(
                      f"Fitting {d}/{t} spectra…"))

    def _done(result):
        if not wiz.still(gen) or wiz._closed:
            return
        wiz.result = result
        wiz.spec = spec
        # Show the fit at the CURRENT position, so the preview reflects the
        # result rather than the pre-run guess.
        try:
            idx = getattr(wiz.plot, "current_indices", None)
            flat = 0
            if idx is not None and nav_shape:
                flat = int(np.ravel_multi_index(
                    tuple(int(i) for i in reversed(idx)), nav_shape))
            spec.set_flat_values(result.values[flat])
        except Exception as e:
            log.debug("seeding the post-fit preview failed: %s", e)
        wiz.draw_preview()
        pct = 100.0 * result.convergence_rate
        ipc.emit_status(f"Fit complete — {pct:.0f}% converged "
                    f"({result.n_iter} iterations)")
        wiz.emit_state(f"{pct:.0f}% converged. Commit to make component maps.")

    wiz.run_on_worker(_fit, name="fit-run", on_done=_done)


def fit_commit(session, plot, payload=None) -> None:
    wiz, _tree = _wizard(session, plot)
    if wiz is None:
        ipc.emit_error("Fit: nothing to commit")
        return
    wiz.commit()
