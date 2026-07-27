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
from spyde.drawing.selectors.base_selector import event_handler_fn
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
        # Position tuple -> fitted parameter vector, filled in as the user
        # explores. Sparse on purpose: a full (P, n) array would claim every
        # position had been fitted when only a handful have.
        if getattr(tree, "fit_store", None) is None:
            tree.fit_store = {}
        # One overlay line per component, plus the dashed sum.
        self._comp_lines: dict = {}
        self._sum_line = None
        # component name -> {"point": widget, "range": widget, "info": ...}
        self._widgets: dict = {}
        # anyplotlib registers callbacks WEAKLY — a handler this object does
        # not hold is collected, and the handle goes dead when grabbed.
        self._widget_cbs: list = []
        # Guards the widget -> model -> widget round trip (the example's
        # `_syncing`): moving a handle in response to its own drag re-enters.
        self._syncing = False

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

    # ── where the navigator is, and what was fitted there ─────────────────
    def current_indices(self):
        """The navigator's position, as a tuple, or None.

        Read from the SELECTOR, not the plot: ``current_indices`` lives on the
        navigation selector. Looking for it on the Plot (as this first did)
        always returned None, which is what made "Fit spectrum" silently fit
        the navigation mean.
        """
        npm = getattr(self.tree, "navigator_plot_manager", None)
        if npm is None:
            return None
        for sels in (getattr(npm, "navigation_selectors", {}) or {}).values():
            for sel in sels:
                idx = getattr(sel, "current_indices", None)
                if idx is None:
                    continue
                try:
                    flat = np.atleast_1d(np.asarray(idx)).ravel()
                    return tuple(int(v) for v in flat)
                except Exception as e:
                    log.debug("reading navigator indices failed: %s", e)
        return None

    def remember(self, values) -> None:
        """Store the fitted parameters for the CURRENT navigator position.

        The store is keyed by position and lives on the tree beside the model,
        so scrubbing back to a pixel shows the fit that was made there rather
        than whatever the last pixel left behind. It is deliberately SPARSE —
        a dict, not a (P, n) array — because it fills in as the user explores
        and a full allocation would claim every position had been fitted.
        """
        key = self.current_indices()
        if key is None:
            return
        self.tree.fit_store[key] = np.asarray(values, float).copy()

    def recall(self) -> bool:
        """Load this position's stored fit into the model. True if there was one."""
        key = self.current_indices()
        stored = self.tree.fit_store.get(key) if key is not None else None
        if stored is None or len(stored) != len(self.spec.parameter_names()):
            return False
        self.spec.set_flat_values(stored)
        return True

    def forget_all(self) -> None:
        """Drop every stored fit.

        Called whenever the component LIST changes: the stored vectors are
        positional, so after an add or remove they would be silently
        reinterpreted against the wrong parameters.
        """
        self.tree.fit_store.clear()

    def current_spectrum(self) -> np.ndarray:
        """The spectrum ON SCREEN — what the preview and "Fit spectrum" fit.

        ``plot.current_data`` is the authority: it is literally the array the
        plot is displaying, already resolved through whatever navigator,
        region-integration or derived-view path produced it.

        Reconstructing it instead from ``signal.data`` and a navigator index was
        wrong in a way that LOOKED like it worked. The index was not where this
        expected, so it silently fell through to the mean over navigation — the
        fit then converged happily against a spectrum nobody was looking at, and
        the drawn model came out about half the height of the data with a
        "converged" status next to it.
        """
        n = len(self.axis())
        data = getattr(self.plot, "current_data", None)
        if isinstance(data, np.ndarray):
            arr = np.asarray(data, float).squeeze()
            if arr.ndim == 1 and arr.size == n:
                return arr

        # No painted data yet (the caret can open before the first frame
        # lands). The nav mean is a defensible stand-in for a PREVIEW, and the
        # log line says so, because a fit against it is not what was asked for.
        raw = np.asarray(self.signal.data, float)
        if raw.ndim > 1:
            log.debug("no painted spectrum yet — falling back to the "
                      "navigation mean")
            return raw.reshape(-1, raw.shape[-1]).mean(0)
        return raw

    # ── live preview: ONE LINE PER COMPONENT + a sum line ─────────────────
    # Follows anyplotlib's interactive-fitting example. Two things there that
    # this got wrong at first, and that matter more than they look:
    #
    #   * lines are updated with ``Line1D.set_data`` IN PLACE. Removing and
    #     re-adding a line every drag frame is heavy AND does not repaint
    #     during the drag — the curve simply did not follow the handle.
    #   * a widget's drag event carries the widget on ``event.source``:
    #     ``event.source.x``, not ``event.x``. Reading ``event.x`` gives None,
    #     so every drag silently did nothing at all.
    #
    # Per-component lines rather than one summed curve, also from the example:
    # with several overlapping peaks a single sum tells you the total is wrong
    # but not WHICH component to grab.
    _COMP_COLORS = ("#f5a97f", "#a6da95", "#c6a0f6", "#eed49f", "#8bd5ca",
                    "#f0c6c6")

    def rebuild_lines(self) -> None:
        """One overlay line per active component, plus the sum. Called when
        the component LIST changes."""
        p1 = getattr(self.plot, "_plot1d", None)
        if p1 is None:
            return
        self.clear_preview()
        x = self.axis()
        blank = np.zeros(len(x), np.float32)
        for i, comp in enumerate(self.spec.active_components):
            try:
                self._comp_lines[comp.name] = p1.add_line(
                    blank.copy(), x_axis=x, label=comp.name, linewidth=1.6,
                    color=self._COMP_COLORS[i % len(self._COMP_COLORS)])
            except Exception as e:
                log.debug("adding a line for %s failed: %s", comp.name, e)
        if len(self.spec):
            try:
                self._sum_line = p1.add_line(
                    blank.copy(), x_axis=x, label="model", color="#cdd6f4",
                    linewidth=1.8, linestyle="dashed")
            except Exception as e:
                log.debug("adding the sum line failed: %s", e)
        self.refresh_lines()

    def refresh_lines(self) -> None:
        """Re-evaluate and ``set_data`` every line — cheap enough per drag frame."""
        if not self._comp_lines and self._sum_line is None:
            return
        try:
            import torch
            from spyde.fitting import components as tcomp
            xt = torch.as_tensor(self.axis())
            total = None
            for comp in self.spec.active_components:
                vals = torch.as_tensor(
                    np.array([[p.value for p in comp.scalar_parameters]]))
                y = tcomp.component_for(comp)(xt, vals).numpy()[0]
                total = y if total is None else total + y
                line = self._comp_lines.get(comp.name)
                if line is not None:
                    line.set_data(np.asarray(y, np.float32))
            if self._sum_line is not None and total is not None:
                self._sum_line.set_data(np.asarray(total, np.float32))
        except Exception as e:
            log.debug("refreshing the model lines failed: %s", e)

    def draw_preview(self) -> None:
        """Refresh in place; rebuild only when the line set is out of date."""
        if list(self._comp_lines) != [c.name for c in self.spec.active_components]:
            self.rebuild_lines()
        else:
            self.refresh_lines()

    def clear_preview(self) -> None:
        """Remove every overlay line.

        ``remove_line`` takes an id or a ``Line1D`` HANDLE, not a label —
        passing the label raises a KeyError that used to be swallowed here, so
        redraws stacked lines until the legend filled up.
        """
        p1 = getattr(self.plot, "_plot1d", None)
        if p1 is not None:
            for line in list(self._comp_lines.values()) + [self._sum_line]:
                if line is None:
                    continue
                try:
                    p1.remove_line(line)
                except Exception as e:
                    log.debug("removing a model line failed: %s", e)
        self._comp_lines.clear()
        self._sum_line = None

    # ── on-plot drag handles (#57) ────────────────────────────────────────
    def sync_widgets(self) -> None:
        """One POINT handle (centre + height) and one RANGE band (width) per
        positioned component. Rebuilt when the component LIST changes.

        Handles are kept as OBJECTS, not ids: the widget's own ``set()`` moves
        it and its drag event hands it back on ``event.source``, so the object
        is what everything here needs.
        """
        p1 = getattr(self.plot, "_plot1d", None)
        if p1 is None:
            return
        self.clear_widgets()
        for i, comp in enumerate(self.spec.active_components):
            info = _DRAG.get(comp.kind)
            if info is None:
                continue                  # a background has nothing to point at
            try:
                colour = self._COMP_COLORS[i % len(self._COMP_COLORS)]
                pos, width, height = self._geometry(comp, info)
                pw = p1.add_point_widget(pos, height, color=colour,
                                         show_crosshair=False)
                self._widgets[comp.name] = {"point": pw, "info": info}
                self._wire(pw, comp.name, "point")
                if info["width"]:
                    half = width * _WIDTH_TO_HALF.get(info["width"], 1.0)
                    rw = p1.add_range_widget(pos - half, pos + half,
                                             y=height / 2.0, color=colour,
                                             style="fwhm")
                    self._widgets[comp.name]["range"] = rw
                    self._wire(rw, comp.name, "range")
            except Exception as e:
                log.debug("adding drag handles for %s failed: %s", comp.name, e)

    def _geometry(self, comp, info):
        """(position, width, peak height) for a component's handles."""
        pos = float(comp[info["pos"]].value)
        width = (float(comp[info["width"]].value) if info["width"]
                 else float(np.ptp(self.axis())) / 20.0)
        height = _height_from_amp(info, float(comp[info["amp"]].value), width)
        return pos, width, height

    def _wire(self, widget, name: str, role: str) -> None:
        """Register the drag handlers, MOVE and UP separately.

        They do different amounts of work, and that is the whole point. A
        pointer_move fires at pointer rate; every one of them crossing the IPC
        boundary to re-send the full model state is what made dragging feel
        like it was catching. A move now redraws the curve and nothing else;
        the state message, the partner handle and any refit wait for the
        release.
        """
        move = event_handler_fn(
            lambda event, n=name, r=role: self._on_widget_drag(n, r, event,
                                                              live=True))
        up = event_handler_fn(
            lambda event, n=name, r=role: self._on_widget_drag(n, r, event,
                                                              live=False))
        # Hold the references: anyplotlib registers callbacks weakly, so a
        # handler owned only by this call is collected and the handle goes dead
        # the moment it is grabbed.
        self._widget_cbs.extend((move, up))
        widget.add_event_handler(move, "pointer_move")
        widget.add_event_handler(up, "pointer_up")

    def update_widgets(self, skip: str | None = None) -> None:
        """MOVE the handles to match the model, in place.

        Distinct from :meth:`sync_widgets`, which rebuilds. Rebuilding on every
        keystroke destroys and recreates every widget several times a second —
        they flicker, and one can vanish from under the cursor mid-reach.
        *skip* is the role being dragged: writing a position back to the handle
        under the user's finger fights the drag.
        """
        for comp in self.spec.active_components:
            entry = self._widgets.get(comp.name)
            if not entry:
                continue
            info = entry["info"]
            try:
                pos, width, height = self._geometry(comp, info)
                if skip != "point" and entry.get("point") is not None:
                    entry["point"].set(x=pos, y=height)
                if skip != "range" and entry.get("range") is not None:
                    half = width * _WIDTH_TO_HALF.get(info["width"], 1.0)
                    entry["range"].set(x0=pos - half, x1=pos + half,
                                       y=height / 2.0)
            except Exception as e:
                log.debug("moving handles for %s failed: %s", comp.name, e)

    def clear_widgets(self) -> None:
        p1 = getattr(self.plot, "_plot1d", None)
        for entry in self._widgets.values():
            for role in ("point", "range"):
                w = entry.get(role)
                if w is None or p1 is None:
                    continue
                try:
                    p1.remove_widget(w.id)
                except Exception as e:
                    log.debug("removing a fit widget failed: %s", e)
        self._widgets.clear()
        self._widget_cbs.clear()

    def _on_widget_drag(self, name: str, role: str, event, live: bool = False) -> None:
        """A handle moved — write it back into the model and redraw.

        The dragged widget arrives on ``event.source``. Reading ``event.x``
        gives None and the drag does nothing at all, which is exactly how this
        failed the first time: the handle moved on screen because the widget
        draws itself, while the model never heard about it.
        """
        if self._syncing:
            return                      # the example's guard against feedback
        self._syncing = True
        try:
            src = getattr(event, "source", None) or event
            comp = self.spec[name]
            info = _DRAG.get(comp.kind)
            if info is None:
                return

            def get(k):
                return (src.get(k) if isinstance(src, dict)
                        else getattr(src, k, None))

            if role == "point":
                x, y = get("x"), get("y")
                if x is not None:
                    comp[info["pos"]].value = float(x)
                if y is not None:
                    width = (float(comp[info["width"]].value)
                             if info["width"] else 1.0)
                    comp[info["amp"]].value = _amp_from_height(info, float(y),
                                                               width)
            else:
                x0, x1 = get("x0"), get("x1")
                if x0 is not None and x1 is not None and info["width"]:
                    factor = _WIDTH_TO_HALF.get(info["width"], 1.0)
                    # Hold the peak HEIGHT as the width changes: for an
                    # area-parameterised component, changing sigma alone moves
                    # the curve away under the cursor.
                    height = _height_from_amp(
                        info, float(comp[info["amp"]].value),
                        float(comp[info["width"]].value))
                    comp[info["width"]].value = max(
                        abs(float(x1) - float(x0)) / 2.0 / max(factor, 1e-9),
                        1e-6)
                    comp[info["pos"]].value = (float(x0) + float(x1)) / 2.0
                    comp[info["amp"]].value = _amp_from_height(
                        info, height, float(comp[info["width"]].value))

            self.result = None          # the old fit no longer describes it
            # DURING the drag: redraw the curve and nothing else. Moving the
            # partner handle and re-sending the model both cross the IPC
            # boundary, and doing either at pointer rate is what made the curve
            # lag behind the cursor.
            self.refresh_lines()
            if not live:
                self.update_widgets(skip=role)
                self.emit_state()
        except Exception as e:
            log.debug("fit widget drag failed: %s", e)
        finally:
            self._syncing = False

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
    wiz.forget_all()   # stored vectors are positional; the parameters changed
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
    wiz.forget_all()   # stored vectors are positional; the parameters changed
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
    wiz.forget_all()   # stored vectors are positional; the parameters changed
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
    wiz.remember(res.values[0])          # this position's answer, kept
    wiz.draw_preview()
    wiz.update_widgets()
    ok = "converged" if bool(res.converged[0]) else "did not converge"
    wiz.emit_state(f"This spectrum {ok} (chi2 {res.chisq[0]:.3g}). "
                   f"{len(wiz.tree.fit_store)} position(s) fitted.")


def fit_navigated(session, plot, payload=None) -> None:
    """The navigator moved — show this position's fit.

    Two behaviours, in order:

    1. If this position has been fitted before, RECALL it. Scrubbing back to a
       pixel should show what was found there, not whatever the last pixel left
       in the model.
    2. Otherwise, if adaptive fitting is on, fit this spectrum now — seeded
       from the model as it stands, which after step 1 is a neighbouring
       position's answer and therefore a good starting point (the same reason
       seeded propagation works for the whole scan, #54).

    With adaptive off and nothing stored, only the preview is redrawn: the
    model stays put and the user sees it against the new spectrum.
    """
    wiz, _tree = _wizard(session, plot)
    if wiz is None or not len(wiz.spec):
        return
    if wiz.recall():
        wiz.draw_preview()
        wiz.update_widgets()
        wiz.emit_state("Recalled this position's fit.")
        return
    if bool((payload or {}).get("adaptive")):
        fit_current(session, plot, payload)
        return
    wiz.draw_preview()          # same model, new spectrum underneath


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
