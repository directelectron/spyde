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
from spyde.fitting.components import EELS_EDGE_KIND
from spyde.fitting.store import FitStore
from spyde.actions.context import src_plot_tree as _src_plot_tree
from de_shell.actions.wizard import WizardController
from spyde.drawing.selectors.base_selector import event_handler_fn
# Imported as a MODULE, not `from ... import emit`. The test fixture patches
# `ipc.emit` to capture outgoing messages, and a from-import binds the original
# function at import time — the patch would then never be seen and every
# message assertion would silently observe nothing.
from de_shell import ipc

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

# exspy's core-loss edge is deliberately NOT in CATALOGUE, and cannot be: every
# entry there is a bare `Kind()` the picker draws as one button, whereas an edge
# is `EELSCLEdge("Fe_L3")` — one button per SUBSHELL, only on an EELS signal,
# and only once the microscope is described. `eels_offer` builds that half of
# the picker; see spyde/spectroscopy/edges.py.

_PREVIEW_POINTS = 64

# ── build hyperspy components ONCE ───────────────────────────────────────────
# Constructing one costs ~23 ms, because hyperspy's Expression components run
# their formula through sympy and lambdify it on every instantiation — there is
# no caching upstream. Building the nine-component palette therefore cost 6.2 s
# on the first caret open and ~130 ms on every one after, which is what "the
# fitting is slow" actually was: the caret opening, not the fit.
#
# These are read-only TEMPLATES. `spec_from_component` only reads them, and
# every caller turns the result into its own ComponentSpec dataclass, so a
# shared instance is safe. Nothing may mutate one — a parameter written here
# would leak into every component added afterwards.
_PROTOTYPES: dict[str, object] = {}
# The ComponentSpec read off each prototype — the structure of a kind never
# changes, so this is read once and deep-copied per use.
_SPECS: dict[str, object] = {}
# Sampled palette shapes, keyed by the axis they were sampled on.
_CATALOGUE_CACHE: dict[tuple, list] = {}


def prototype(kind: str, order: int | None = None):
    """A shared, read-only hyperspy component of *kind*.

    Racy by design: two threads asking at once may each build one and the
    second overwrite the first in the dict. Both are equally valid read-only
    templates, so the only cost is 20 ms of duplicated work — cheaper than a
    lock on a path the catalogue worker and the caret both take.
    """
    key = kind if order is None else f"{kind}:{order}"
    comp = _PROTOTYPES.get(key)
    if comp is None:
        import hyperspy.components1d as c1d
        comp = (c1d.Polynomial(order=int(order if order is not None else 2))
                if kind == "Polynomial" else getattr(c1d, kind)())
        _PROTOTYPES[key] = comp
    return comp


def new_component_spec(kind: str, order: int | None = None):
    """A FRESH ``ComponentSpec`` for *kind*, built from the shared prototype.

    The single place anything in SpyDE turns a component NAME into a spec.
    Everywhere that used to construct its own hyperspy component to read the
    shape of one — the picker, "add component", the background wizard — comes
    through here, so the ~20-80 ms build happens once per kind per process.

    The returned spec is a deep copy and the caller owns it: it is about to get
    a name, seeded values and a scale, and two components of the same kind in
    one model must not share parameter objects.
    """
    from copy import deepcopy
    from spyde.fitting.spec import spec_from_component
    key = kind if order is None else f"{kind}:{order}"
    cspec = _SPECS.get(key)
    if cspec is None:
        cspec = spec_from_component(prototype(kind, order))
        _SPECS[key] = cspec
    return deepcopy(cspec)

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

# BACKGROUNDS get handles too, they just cannot use the peak/width pair — there
# is no peak to point at. Instead they carry ANCHOR handles at fixed fractions
# of the axis, and dragging one makes the curve pass through it. Two anchors
# determine both parameters of a power law or an exponential exactly, which is
# also how you would place one by eye: pin it at each end.
#
# `solve` names the closed form that turns the anchor positions back into
# parameters. Least-squares would work for any number of anchors, but an exact
# solve means the curve passes EXACTLY through the handle the user is holding —
# a curve that lands near your cursor rather than under it reads as broken.
_ANCHORS = {
    "Offset":      {"at": (0.5,), "solve": "offset"},
    "Polynomial":  {"at": (0.5,), "solve": "shift"},
    "PowerLaw":    {"at": (0.25, 0.75), "solve": "powerlaw"},
    "Exponential": {"at": (0.25, 0.75), "solve": "exponential"},
}


def evaluate_component(comp, xs) -> np.ndarray:
    """A component's own curve at *xs*, through the torch components."""
    import torch
    from spyde.fitting import components as tcomp
    vals = np.array([[p.value for p in comp.scalar_parameters]])
    return tcomp.component_for(comp)(
        torch.as_tensor(np.asarray(xs, float)),
        torch.as_tensor(vals)).numpy()[0]


def _clip_to_bounds(param, value: float, span: float = 50.0) -> float:
    lo, hi = param.bounds()
    return float(np.clip(value, max(lo, -span), min(hi, span)))


def _solve_anchors(comp, kind: str, xs, ys) -> bool:
    """Write parameters so the component passes through the anchor points.

    False when the points do not determine the component — a power law needs
    positive values inside its domain, an exponential needs two DIFFERENT
    ones. Writing anyway would put a nan in the model and every curve drawn
    afterwards would vanish.
    """
    solve = _ANCHORS[kind]["solve"]
    if solve == "offset":
        comp["offset"].value = float(ys[0])
        return True
    if solve == "shift":
        # Move the whole polynomial vertically through its constant term. The
        # higher coefficients are its SHAPE and are left to the fit — a single
        # handle cannot determine order+1 coefficients, and pretending it can
        # would make the curve lurch.
        y_now = float(evaluate_component(comp, [xs[0]])[0])
        comp["a0"].value = float(comp["a0"].value) + float(ys[0]) - y_now
        return True

    (x1, x2), (y1, y2) = (float(xs[0]), float(xs[1])), (float(ys[0]), float(ys[1]))
    if solve == "powerlaw":
        origin = float(comp["origin"].value)
        d1, d2 = x1 - origin, x2 - origin
        if d1 <= 0 or d2 <= 0 or y1 <= 0 or y2 <= 0 or d1 == d2:
            return False                # outside the curve's domain
        comp["r"].value = _clip_to_bounds(
            comp["r"], math.log(y1 / y2) / math.log(d2 / d1))
        comp["A"].value = float(y1 * d1 ** comp["r"].value)
        return True
    if solve == "exponential":
        if y1 <= 0 or y2 <= 0 or x1 == x2 or y1 == y2:
            return False
        tau = (x2 - x1) / math.log(y1 / y2)
        comp["tau"].value = float(tau)
        comp["A"].value = float(y1 * math.exp(x1 / tau))
        return True
    return False


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
        # The per-position parameters live in a real HyperSpy model's own
        # `parameter.map` arrays — see spyde/fitting/store.py. They exist from
        # the moment the caret opens, empty, and fill in as positions are
        # fitted. `_ensure_store` (re)builds it whenever the component list
        # changes, because the packed width changes with it.
        self._ensure_store()
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

        …and from the selector's LIVE geometry, not its ``current_indices``
        snapshot. That attribute is written by ``_run_update`` on the
        ``_NavDispatcher`` THREAD, while this runs on the asyncio main thread —
        and the renderer sends ``fit_navigated`` off the same pointer event that
        started the navigator update, so a handler can easily arrive before the
        dispatcher has committed the new position. It then recalls the PREVIOUS
        pixel's fit, and nothing re-fires to correct it: the caret sits on a
        stale model until the next navigator move. ``get_selected_indices`` is
        the same pure geometry call ``_run_update`` itself makes (the snapshot is
        just its last result), so reading it here is the same number, only never
        behind.

        Which selector wins is unchanged — still the first with a committed
        position — so this cannot resurrect the fit-the-navigation-mean bug
        above; only the VALUE is made current.
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
                    live = sel.get_selected_indices()
                    if live is not None:
                        idx = live
                except Exception as e:
                    log.debug("live navigator indices unavailable (%s); "
                              "using the dispatcher's snapshot", e)
                try:
                    flat = np.atleast_1d(np.asarray(idx)).ravel()
                    return tuple(int(v) for v in flat)
                except Exception as e:
                    log.debug("reading navigator indices failed: %s", e)
        return None

    # ── the per-position store ────────────────────────────────────────────
    @property
    def store(self):
        return getattr(self.tree, "fit_store", None)

    def _ensure_store(self, force: bool = False):
        """(Re)build the per-position store for the CURRENT component list.

        Rebuilt whenever that list changes: the store's width is the packed
        parameter count, so an add or remove would otherwise reinterpret every
        stored vector against the wrong parameters. Nothing is carried over —
        that is the same reason the old dict was cleared on an edit.
        """
        from spyde.fitting.store import FitStore
        want = len(self.spec.parameter_names())
        store = getattr(self.tree, "fit_store", None)
        if not force and store is not None and store.n_params == want \
                and store.spec is self.spec:
            return store
        try:
            store = FitStore(self.spec, self.signal)
        except Exception as e:
            log.debug("building the fit store failed: %s", e)
            store = None
        self.tree.fit_store = store
        return store

    def remember(self, values, chisq: float | None = None) -> None:
        """Store the fitted parameters for the CURRENT navigator position."""
        store = self.store
        if store is not None:
            store.put(self.current_indices(), values, chisq=chisq)

    def record_run(self, result, nav_shape=None) -> int:
        """Record a whole-scan fit. Returns how many positions landed.

        EVERY position, not only the converged ones. Moving the navigator has
        to show the fit that was made there — a position left out shows
        whatever the last one left in the model instead, which reads as the fit
        being stale. A position that fitted poorly is still that position's
        answer; ``poor_positions`` is how it gets found again.
        """
        store = self.store
        if store is None:
            return 0
        return store.put_all(result.values,
                             chisq=getattr(result, "chisq", None))

    def poor_positions(self) -> int:
        """How many positions fit worse than their neighbours."""
        store = self.store
        if store is None:
            return 0
        try:
            from spyde.fitting.polish import neighbour_index, poor_mask
            chisq = store.chisq.ravel()
            done = store.set_mask().ravel()
            if not done.any():
                return 0
            mask = poor_mask(np.where(done, chisq, np.nanmedian(chisq[done])),
                             neighbour_index(store.nav_shape))
            return int((mask & done).sum())
        except Exception as e:
            log.debug("counting the poor positions failed: %s", e)
            return 0

    def recall(self) -> bool:
        """Load this position's stored fit into the model. True if there was one."""
        store = self.store
        stored = store.get(self.current_indices()) if store is not None else None
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
        store = self.store
        if store is not None:
            store.clear()

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
        """Re-evaluate every line and push the result ONCE.

        The arithmetic here is nothing — 0.37 ms for two components over 1024
        channels, measured. The cost is the transport: ``Line1D.set_data``
        recomputes the axis range and pushes the WHOLE plot state, so calling it
        per line meant N+1 full state pushes per pointer frame, and that is what
        made dragging lag. The lines are written together and pushed once
        instead, which is one push per frame no matter how many components the
        model has.

        This reaches into anyplotlib's line entries because there is no public
        batched update; the public per-line path is kept as the fallback, so a
        change upstream costs speed rather than correctness.
        """
        if not self._comp_lines and self._sum_line is None:
            return
        try:
            import torch
            from spyde.fitting import components as tcomp
            xt = torch.as_tensor(self.axis())
            updates, total = [], None
            for comp in self.spec.active_components:
                vals = torch.as_tensor(
                    np.array([[p.value for p in comp.scalar_parameters]]))
                y = tcomp.component_for(comp)(xt, vals).numpy()[0]
                total = y if total is None else total + y
                line = self._comp_lines.get(comp.name)
                if line is not None:
                    updates.append((line, y))
            if self._sum_line is not None and total is not None:
                updates.append((self._sum_line, total))
        except Exception as e:
            log.debug("evaluating the model lines failed: %s", e)
            return
        self._push_lines(updates)

    def _push_lines(self, updates) -> None:
        """Write several lines' data and push the plot once."""
        p1 = getattr(self.plot, "_plot1d", None)
        if p1 is None or not updates:
            return
        try:
            for line, y in updates:
                line._entry()["data"] = np.asarray(y, float)
            p1._recompute_data_range()
            p1._push()
        except Exception as e:
            log.debug("batched line push unavailable (%s); falling back to "
                      "one push per line", e)
            for line, y in updates:
                try:
                    line.set_data(np.asarray(y, np.float32))
                except Exception as e2:
                    log.debug("set_data fallback failed: %s", e2)

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
        """Give EVERY component handles. Rebuilt when the component LIST changes.

        A positioned component (a peak, a step) gets a POINT handle for its
        centre and height plus a RANGE band for its width. A background has no
        peak to point at, so it gets ANCHOR points instead — see ``_ANCHORS``.
        Either way there is something on the curve to grab, which is the point:
        a component you can only edit through the caret's number boxes is a
        component you cannot see yourself editing.

        Handles are kept as OBJECTS, not ids: the widget's own ``set()`` moves
        it and its drag event hands it back on ``event.source``, so the object
        is what everything here needs.
        """
        p1 = getattr(self.plot, "_plot1d", None)
        if p1 is None:
            return
        self.clear_widgets()
        for i, comp in enumerate(self.spec.active_components):
            colour = self._COMP_COLORS[i % len(self._COMP_COLORS)]
            try:
                info = _DRAG.get(comp.kind)
                if info is not None:
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
                elif comp.kind in _ANCHORS:
                    xs = self._anchor_x(comp)
                    ys = self._eval_at(comp, xs)
                    entry = {"anchors": [], "at": list(xs), "info": None}
                    for j, (ax, ay) in enumerate(zip(xs, ys)):
                        w = p1.add_point_widget(float(ax), float(ay),
                                                color=colour,
                                                show_crosshair=False)
                        entry["anchors"].append(w)
                        self._wire(w, comp.name, f"anchor:{j}")
                    self._widgets[comp.name] = entry
            except Exception as e:
                log.debug("adding drag handles for %s failed: %s", comp.name, e)

    def _geometry(self, comp, info):
        """(position, width, peak height) for a component's handles."""
        pos = float(comp[info["pos"]].value)
        width = (float(comp[info["width"]].value) if info["width"]
                 else float(np.ptp(self.axis())) / 20.0)
        height = _height_from_amp(info, float(comp[info["amp"]].value), width)
        return pos, width, height

    def _anchor_x(self, comp):
        """Default x positions for a background's anchor handles."""
        x = self.axis()
        lo, hi = float(x[0]), float(x[-1])
        return [lo + f * (hi - lo) for f in _ANCHORS[comp.kind]["at"]]

    def _eval_at(self, comp, xs) -> np.ndarray:
        """The component's own curve at *xs* — how the anchors stay ON it."""
        return evaluate_component(comp, xs)

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
        """MOVE the handles back onto the curve, in place.

        Distinct from :meth:`sync_widgets`, which rebuilds. Rebuilding on every
        keystroke destroys and recreates every widget several times a second —
        they flicker, and one can vanish from under the cursor mid-reach.
        *skip* is the role being dragged: writing a position back to the handle
        under the user's finger fights the drag.

        This runs DURING a drag as well as after it. Skipping it mid-drag left
        the partner handle where the drag started while the curve moved out
        from under it, so the handles drifted off the component and only
        snapped back on release. A widget ``set`` is a TARGETED push of that
        widget's few fields, not a plot re-render, so paying it per frame is
        what keeps the handles glued on.

        **Call this BEFORE the redraw, never after.** A widget ``set`` is a
        TARGETED push on the event channel; ``refresh_lines`` ends in a FULL
        panel push that serialises every widget's geometry. Redrawing first
        means that push carries the OLD handle positions and the new ones go
        out targeted-only — invisible to anything reading panel state, and one
        extra push per moved handle. Every caller here is ordered that way.

        **Guarded, because ``set`` FIRES a ``pointer_move``.** anyplotlib
        notifies on a programmatic move exactly as it does on a dragged one,
        so every handle written here re-enters ``_on_widget_drag`` — which
        treats it as a user edit and discards the fit. Moving the handles onto
        a freshly fitted model therefore threw that fit away: `result` went
        None, the maps would not open ("run the fit before committing") and
        the Commit button vanished. During a real drag the flag is already
        held by the drag itself, so this only adds the missing case.
        """
        was_syncing = self._syncing
        self._syncing = True
        try:
            self._update_widgets(skip)
        finally:
            self._syncing = was_syncing

    def _update_widgets(self, skip: str | None = None) -> None:
        for comp in self.spec.active_components:
            entry = self._widgets.get(comp.name)
            if not entry:
                continue
            try:
                if entry.get("anchors"):
                    ys = self._eval_at(comp, entry["at"])
                    for j, (w, ax) in enumerate(zip(entry["anchors"],
                                                    entry["at"])):
                        if skip == f"anchor:{j}":
                            continue
                        w.set(x=float(ax), y=float(ys[j]))
                    continue
                info = entry["info"]
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
            widgets = [entry.get("point"), entry.get("range")]
            widgets += entry.get("anchors") or []
            for w in widgets:
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

            def get(k):
                return (src.get(k) if isinstance(src, dict)
                        else getattr(src, k, None))

            if role.startswith("anchor:"):
                entry = self._widgets.get(name)
                if entry is None or comp.kind not in _ANCHORS:
                    return
                j = int(role.split(":", 1)[1])
                x, y = get("x"), get("y")
                if x is None or y is None:
                    return
                entry["at"][j] = float(x)
                # The OTHER anchors stay where they are; solving through all of
                # them together is what makes one handle move the curve without
                # throwing the rest off it.
                ys = list(self._eval_at(comp, entry["at"]))
                ys[j] = float(y)
                _solve_anchors(comp, comp.kind, entry["at"], ys)
            elif info is None:
                return
            elif role == "point":
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
            # Every frame: move the other handles onto the curve AND redraw it,
            # so nothing drifts off the component mid-drag. What still waits for
            # the release is `emit_state` — that one re-sends the entire model
            # to the caret, and doing THAT at pointer rate is what made the
            # curve lag behind the cursor.
            #
            # HANDLES FIRST, then the lines. `refresh_lines` ends in a FULL
            # panel push, which serialises every widget's current geometry;
            # `update_widgets` only issues TARGETED per-widget pushes. Doing
            # the lines first meant the frame's full push carried the OLD
            # partner position and the new one went out on the targeted
            # channel alone — invisible to anything reading panel state, and
            # one push per moved handle instead of none.
            self.update_widgets(skip=role)
            self.refresh_lines()
            if not live:
                self.emit_state()
        except Exception as e:
            log.debug("fit widget drag failed: %s", e)
        finally:
            self._syncing = False

    def contributions(self) -> dict[str, float]:
        """Each component's peak height as a fraction of the model's.

        Because the caret shows a gaussian's ``A``, which is its AREA. A
        component one SIXTH the height of its neighbour has one SIXTY-FIFTH
        the area once its width is a tenth — so a correct fit of a narrow peak
        on a broad one reads as "A=888 next to A=57471", i.e. as a component
        that has been suppressed to zero. It has not; you just cannot see it
        at that scale on a linear axis. This is the number that says so.
        """
        try:
            x = self.axis()
            peaks = {c.name: float(np.max(np.abs(np.nan_to_num(
                evaluate_component(c, x), nan=0.0, posinf=0.0, neginf=0.0))))
                for c in self.spec.active_components}
        except Exception as e:
            log.debug("computing component contributions failed: %s", e)
            return {}
        total = max(peaks.values(), default=0.0)
        return {k: (v / total if total > 0 else 0.0) for k, v in peaks.items()}

    def emit_state(self, status: str | None = None) -> None:
        """Send the whole model to the caret — components, parameters, values.

        One message rather than incremental patches: a model is small, and a
        renderer that rebuilds from the truth cannot drift out of step with the
        backend after a failed edit.
        """
        share = self.contributions()
        done, total = (self.store.coverage() if self.store is not None
                       else (0, self.nav_total()))
        ipc.emit({
            "type": "fit_state",
            "window_id": getattr(self.plot, "window_id", None),
            "components": [
                {"name": c.name, "kind": c.kind, "active": bool(c.active),
                 "share": share.get(c.name, 0.0),
                 "parameters": [
                     {"name": p.name, "value": float(p.value),
                      "free": bool(p.free), "linear": bool(p.linear)}
                     for p in c.scalar_parameters]}
                for c in self.spec.components],
            "fitted": self.result is not None,
            # Which positions have an answer. Without this the store is
            # invisible: you cannot tell a position you skipped from one you
            # fitted, and "why didn't adaptive fill this in" has no answer on
            # screen.
            "fitted_count": done,
            "nav_total": total,
            "position_fitted": bool(self.store is not None and
                                    self.store.is_set(self.current_indices())),
            # How many positions fit worse than their neighbours. The honest
            # headline for a scan fit: "99% converged" can still hide a patch
            # where the model fell over.
            "poor_count": self.poor_positions(),
            # Models already stored on this signal, so the caret can offer them
            # without a round trip. These are HyperSpy's own — they save and
            # load with the dataset.
            "stored_models": (self.store.stored_names()
                              if self.store is not None else []),
            "status": status,
        })

    def nav_total(self) -> int:
        try:
            nav = self.signal.axes_manager.navigation_shape
            return int(np.prod([int(n) for n in nav])) if len(nav) else 1
        except Exception:
            return 0

    # ── teardown ──────────────────────────────────────────────────────────
    def remove(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.clear_preview()
        self.clear_widgets()
        # The live maps window belongs to the CARET, so it closes with it —
        # it is a preview of the model as it currently stands, and one left
        # behind is a window that silently stops tracking. "Commit components"
        # is how you get one that stays.
        tree = getattr(self, "_maps_tree", None)
        self._maps_tree = None
        self._maps_labels = None
        if tree is not None and not getattr(tree, "_spyde_closed", False):
            try:
                self.session._close_tree(tree)
            except Exception as e:
                log.debug("closing the fit maps window failed: %s", e)
        # The MODEL and the per-position STORE deliberately SURVIVE — they live
        # on the tree, so reopening the caret restores what the user built and
        # every position already fitted. Only the controller and its on-plot
        # artefacts go.
        if getattr(self.tree, "_fit_wizard", None) is self:
            self.tree._fit_wizard = None

    # ── the component maps, live ──────────────────────────────────────────
    def result_maps(self) -> dict[str, np.ndarray]:
        """One map per component, plus chi-squared, read from the STORE.

        From the store and not from a finished ``FitResult``, because that is
        what makes them exist before there is a result: every map is NaN where
        the position has not been fitted, so the window can open flat when the
        caret does and fill in as positions are fitted — a point at a time from
        "Fit spectrum", or all at once from "Fit all Spectra".

        chi-squared belongs beside the components, not in a status line: it is
        the map that says WHERE the model failed, and a component map read
        without it can be a picture of the fit falling over rather than of the
        sample.
        """
        store = self.store
        if store is None:
            return {}
        return store.maps(self.axis())

    def show_maps(self):
        """Open the component-maps window if it is not open, else REFRESH it.

        One window for the life of the caret, updated in place. Creating a new
        one per fit stacked identical-looking windows with no way to tell which
        was current; and creating it only when a fit finished meant there was
        nothing to watch fill in.

        It is titled "(live)" to separate it from the snapshot commit() keeps.
        Both used to be "Fit components", which is the same
        no-way-to-tell-which-is-which problem one paragraph up — after a
        commit the user had two identical titles side by side, and
        fit_wizard.spec.ts could not target either (its locator matched both
        and failed strict mode).
        """
        maps = self.result_maps()
        if not maps:
            return None
        labels = list(maps)
        tree = getattr(self, "_maps_tree", None)
        alive = tree is not None and not getattr(tree, "_spyde_closed", False)
        if alive and labels == getattr(self, "_maps_labels", None):
            self._refresh_maps(tree, maps)
            return tree
        # The SET of maps changed — a component was added or removed. The
        # window's primary map and its chips are fixed at creation, so this
        # rebuilds rather than repaints: refreshing would leave the first
        # component with no chip of its own and chi-squared as the primary.
        if alive:
            try:
                self.session._close_tree(tree)
            except Exception as e:
                log.debug("closing the previous maps window failed: %s", e)
        self._maps_tree = self._open_maps(maps, self.LIVE_MAPS_TITLE)
        self._maps_labels = labels
        return self._maps_tree

    #: The caret's live maps window, refreshed in place as positions fit.
    LIVE_MAPS_TITLE = "Fit components (live)"
    #: The snapshot commit() keeps. Plain name: it is the one that persists.
    COMMITTED_MAPS_TITLE = "Fit components"

    def _open_maps(self, maps, title: str = COMMITTED_MAPS_TITLE):
        names = list(maps)
        return commit_result_tree(
            self.session, title=title,
            primary=maps[names[0]], primary_label=names[0],
            views=[(n, maps[n]) for n in names[1:]],
            levels=None, cmap="viridis",
            attrs={"fit_spec": self.spec, "fit_result": self.result},
            provenance={"action": "Fit",
                        "params": {"components": [c.kind for c in self.spec]},
                        "source_title": getattr(
                            self.signal.metadata.General, "title", "")},
        )

    def _refresh_maps(self, tree, maps) -> None:
        """Repaint an already-open maps window with the current store."""
        from spyde.actions.views import emit_view_figure, register_views
        mats = [(n, np.nan_to_num(np.asarray(m, np.float32)))
                for n, m in maps.items()]
        sp = next(iter(getattr(tree, "signal_plots", []) or []), None)
        if sp is None:
            return
        try:
            tree.root.data[...] = mats[0][1]
        except Exception as e:
            log.debug("writing the maps root failed: %s", e)
        try:
            sp.needs_auto_level = True
            sp.set_data(mats[0][1])
        except Exception as e:
            log.debug("repainting the maps plot failed: %s", e)
        wid = getattr(sp, "window_id", None)
        if wid is None:
            return
        try:
            register_views(wid, mats, cmap="viridis", levels=None)
            for lbl, m in mats[1:]:
                emit_view_figure(wid, m, lbl, kind="2d", cmap="viridis",
                                 levels=None)
        except Exception as e:
            log.debug("re-registering the maps views failed: %s", e)

    def commit(self):
        """Keep the current maps as a tree of their own.

        The live window belongs to the caret and is refreshed as more
        positions are fitted; this makes a snapshot that does not move.
        """
        if self.session is None:
            return None
        if self.store is None or not self.store.coverage()[0]:
            ipc.emit_error("Fit: fit something before keeping the maps")
            return None
        maps = self.result_maps()
        if not maps:
            ipc.emit_error("Fit: no component produced a map")
            return None
        return self._open_maps(maps)

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

    lo, hi = float(np.min(x)), float(np.max(x))
    # The shapes depend only on the axis they are sampled over, so reopening the
    # caret on the same signal costs nothing.
    cache_key = (round(lo, 6), round(hi, 6))
    hit = _CATALOGUE_CACHE.get(cache_key)
    if hit is not None:
        return hit

    xs = np.linspace(lo, hi, _PREVIEW_POINTS)
    out = []
    for kind, description in CATALOGUE:
        try:
            cspec = new_component_spec(kind, 2 if kind == "Polynomial" else None)
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
    _CATALOGUE_CACHE[cache_key] = out
    return out


def eels_offer(signal) -> dict:
    """The EELS half of the picker: which edges can be added, and what blocks them.

    Always returns the same keys, so the renderer never has to guess why the
    edge section is empty — an EELS signal without exspy gets the install line,
    one without the microscope gets the field names, and a non-EELS signal gets
    ``eels: False`` and no section at all.

    Every failure here is swallowed: the picker's analytic components must
    still arrive if the edge lookup goes wrong.
    """
    out = {"eels": False, "exspy": False, "edges": [],
           "microscope_missing": [], "install_hint": ""}
    try:
        from spyde.drawing.toolbars.plot_control_toolbar import (
            install_hint, package_available,
        )
        from spyde.spectroscopy import edges as eels_edges
    except Exception as e:                                # pragma: no cover
        log.debug("the EELS edge offer is unavailable: %s", e)
        return out

    if signal is None or not eels_edges.is_eels(signal):
        return out
    out["eels"] = True
    # `package_available` uses find_spec — asking does not pay exspy's import.
    if not package_available("exspy"):
        out["install_hint"] = install_hint("exspy")
        return out
    out["exspy"] = True
    try:
        out["microscope_missing"] = eels_edges.missing_microscope_parameters(signal)
    except Exception as e:                                # pragma: no cover
        log.debug("reading the microscope parameters failed: %s", e)
    try:
        out["edges"] = eels_edges.available_edges(signal)
    except Exception as e:
        log.info("listing the EELS edges for the picker failed: %s", e)
    return out


def scale_to_data(cspec, x: np.ndarray, y: np.ndarray, fraction: float = 0.5) -> None:
    """Set a component's LINEAR amplitude so it is visible against the data.

    Without this a new component arrives with an amplitude of 1 against counts
    of 1e5 — the preview curve is a flat line on the axis, the drag handles sit
    at y=0, and the fit starts five orders of magnitude away from its answer.
    All three read as "the model does nothing".

    Evaluate the component with its linear parameter at 1, then scale so its
    peak reaches *fraction* of the data's range.

    A BACKGROUND does not go through here — see :func:`seed_background`.
    Scaling one by its peak matches a power law's SINGULAR left edge (on an
    axis starting at zero the unit curve peaks at 1/dx**r), which left the
    background at ~3e-5 across the whole spectrum; scaling it by its median
    level instead put 3.7e10 at that same edge and blew up the y-axis. Neither
    is a scaling problem: a background needs its SHAPE seeded from the data,
    not just its amplitude.
    """
    import torch
    from spyde.fitting import components as tcomp

    linear = next((p for p in cspec.scalar_parameters if p.linear), None)
    if linear is None:
        return
    try:
        # `component_for`, not `get_component`: an EELS edge is a supported
        # kind that resolves only from its SPEC (the precomputed GOS curves
        # live there). Looking it up by kind alone raises, which this except
        # swallowed — so an added edge kept intensity=1 against counts of 1e5
        # and drew as a flat line on the axis.
        comp = tcomp.component_for(cspec)
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


def seed_background(cspec, x: np.ndarray, y: np.ndarray) -> bool:
    """Put a background ON the data, by solving it through two points of it.

    The same closed form the anchor handles use (:func:`_solve_anchors`), fed
    from the data instead of from the cursor — so a background arrives roughly
    where the user would have dragged it, and its handles start on the curve
    they are about to move.

    This replaces seeding a shape constant and then scaling: with a fixed
    ``r=3`` a power law spanning a decade of axis is far steeper than any real
    background, so whichever amplitude you then choose it is either invisible
    at one end or astronomical at the other. Fitting r to the data's own decay
    is the only thing that gets both ends right.

    Returns False if *cspec* is not a background, or if the data does not
    determine one (flat data cannot fix an exponential's tau), so the caller
    falls back to :func:`scale_to_data` and the component is at least visible.
    """
    if cspec.kind not in _ANCHORS:
        return False
    x = np.asarray(x, float)
    ydata = np.asarray(y, float)
    lo, hi = float(x[0]), float(x[-1])
    # A LOW PERCENTILE over a WIDE window, not the local value or its median.
    # A background belongs on the data's low envelope, and the anchor
    # fractions land wherever they land: on this test spectrum the 50% anchor
    # sits on a peak, and a median there seeded a flat background at 765 when
    # the baseline was 5. A wide window is what steps past a peak; a low
    # percentile is what ignores the part of it still inside the window.
    at = _ANCHORS[cspec.kind]["at"]
    # A single-anchor background is a LEVEL, so its window is the whole
    # spectrum: a window centred mid-axis is centred on whatever peak happens
    # to be there, and seeded a flat background at 734 on a baseline of 17.
    half = len(x) if len(at) == 1 else max(len(x) // 6, 1)
    xs, ys = [], []
    for f in at:
        i = int(np.clip(round(f * (len(x) - 1)), 0, len(x) - 1))
        j0, j1 = max(i - half, 0), min(i + half + 1, len(x))
        window = ydata[j0:j1]
        window = window[np.isfinite(window)]
        if not len(window):
            return False
        xs.append(float(x[i]))
        ys.append(float(np.percentile(window, 15.0)))
    if cspec.kind == "PowerLaw":
        # Hyperspy's convention, and the reference point every downstream tool
        # assumes. It is also where the anchors need it: `origin` above the
        # first anchor puts that handle outside the curve's domain.
        cspec["origin"].value = 0.0
        cspec["origin"].free = False
        if lo <= 0.0:
            # ...except that the axis then contains the singularity. Move the
            # reference off-screen so the curve is finite where it is drawn,
            # rather than clipping it and drawing a wall. A QUARTER of the
            # span, not a nudge: at 2% the left edge is only 2% of the way to
            # the first anchor, so the same fitted r puts 16000 there against
            # data of 300. A quarter keeps the whole axis on the gentle part
            # of the curve, which is what a background over this range is.
            cspec["origin"].value = lo - 0.25 * (hi - lo)
    return _solve_anchors(cspec, cspec.kind, xs, ys)


def spread_repeats(cspec, spec, lo: float, hi: float) -> None:
    """Place repeats of one kind APART, symmetrically about the seed.

    Every component is seeded mid-axis, so a second gaussian used to land
    exactly on the first: identical centre, identical sigma, and therefore
    two IDENTICAL Jacobian columns (measured corr(centre1, centre2) =
    1.000000, cond(J) = 1e16). A perfectly degenerate pair can never
    separate — the solver moves both the same way forever — so it grows one
    and shrinks the other, which is what "it forces the second gaussian to 0"
    looks like. On two well-separated peaks the pair instead ran away
    together to sigma = 170 on a 100-wide axis, chisq 6.3e7.

    The family is re-placed as a whole and spaced by ONE of their own widths.
    Symmetric so the pair still straddles where the seed put it; one width so
    the columns are distinct without moving a genuinely co-centred pair far —
    on hyperspy's ``two_gaussians``, where both true centres ARE 50, this
    reproduces the co-located fit exactly (chisq 5.245e5, unchanged).

    Wider spacing is worse, not better: at two widths the co-centred case
    fails outright (chisq 6.3e5 against 1.3e-21). Nothing here is a
    universally right guess — which is why the components have drag handles.
    """
    info = _DRAG.get(cspec.kind)
    if info is None:
        return                       # a background has no position to spread
    family = [c for c in spec.components if c.kind == cspec.kind] + [cspec]
    if len(family) < 2:
        return
    width = (float(cspec[info["width"]].value) if info["width"]
             else (hi - lo) / 8.0)
    base = float(cspec[info["pos"]].value)
    for j, comp in enumerate(family):
        offset = (j - (len(family) - 1) / 2.0) * width
        comp[info["pos"]].value = float(np.clip(base + offset, lo, hi))


def clamp_to_axis(cspec, lo: float, hi: float) -> None:
    """Keep a peak's position and width inside the data.

    A component centred off the axis, or wider than the whole of it, is not
    modelling anything the user can see — and those are exactly the runaway
    directions the solver walks down when a start is poor: a centre at 750 on
    a 0-100 axis, a sigma of 170 on the same. Bounding them is not enough on
    its own (a degenerate pair just pins itself at the cap instead) but it is
    necessary: with the spread above and without these, a noisy pair of
    near-overlapping peaks still diverged to chisq 1.1e8.

    Only components with a position ON the curve. A background is bounded by
    nothing here — a PowerLaw's ``origin`` is a reference point that is
    deliberately placed OUTSIDE the axis when the axis would otherwise
    contain its singularity.
    """
    info = _DRAG.get(cspec.kind)
    if info is None:
        return
    span = hi - lo
    for name, bmin, bmax in ((info["pos"], lo, hi),
                             (info["width"], span / 1000.0, span)):
        if not name:
            continue
        try:
            p = cspec[name]
        except KeyError:
            continue
        p.bmin = bmin if p.bmin is None else max(p.bmin, bmin)
        p.bmax = bmax if p.bmax is None else min(p.bmax, bmax)
        p.value = float(np.clip(p.value, p.bmin, p.bmax))


def _seed_for_preview(cspec, lo: float, hi: float) -> None:
    """Put a component somewhere visible on THIS axis.

    A default Gaussian sits at 0 with sigma 1, which is off-screen on a 200-800
    eV axis and would preview as a flat line — the picker would then show every
    peak shape as identical nothing.
    """
    mid, width = (lo + hi) / 2, (hi - lo) / 8
    # A PowerLaw's `origin` is its reference point, NOT a position on the
    # curve, and hyperspy returns zero below it. Seeding it mid-axis like a
    # peak centre gave a background that was identically zero over the left
    # half of the spectrum — and left its anchor handles outside the domain,
    # where an exact solve has nothing to solve. `seed_background` owns it.
    for p in cspec.parameters:
        if p.name == "origin" and cspec.kind == "PowerLaw":
            continue
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


def _send_catalogue(session, wiz, src, gen: int) -> None:
    """Sample the palette OFF the main thread and send it when it is ready.

    The first call in a process builds nine hyperspy components (~680 ms of
    sympy lambdify) and it used to sit in the middle of ``fit_open``, so the
    caret took most of a second to appear the first time. The renderer already
    receives ``fit_catalogue`` as its own event and renders an empty picker
    until it arrives, so nothing here has to be synchronous — the caret opens
    now and the sparklines fill in.
    """
    from spyde.actions.lifecycle import run_on_worker
    x = wiz.axis()
    signal = wiz.signal

    def _build():
        # The EELS edges ride the SAME worker call as the sparklines. They are
        # cheap (a table lookup), but the first `import exspy` is not, and
        # paying it on the main thread would put the caret's open back where
        # the sympy lambdify used to have it.
        return component_catalogue(x), eels_offer(signal)

    def _emit(result):
        if not wiz.still(gen):
            return                      # the caret closed while we sampled
        components, offer = result
        ipc.emit({"type": "fit_catalogue",
                  "window_id": getattr(src, "window_id", None),
                  "components": components, **offer})

    run_on_worker(session, _build,
                  name="fit-catalogue", on_done=_emit,
                  on_error=lambda e: log.debug("component catalogue: %s", e))


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
    gen = wiz.guard()
    tree._fit_wizard = wiz
    _send_catalogue(session, wiz, src, gen)
    wiz.draw_preview()
    wiz.sync_widgets()
    wiz.show_maps()          # flat now, filled in as positions are fitted
    wiz.emit_state("Add a component to begin." if not len(wiz.spec)
                   else "Model restored.")


def fit_close(session, plot, payload=None) -> None:
    wiz, _tree = _wizard(session, plot)
    if wiz is not None:
        wiz.cancel_inflight()
        wiz.remove()


def _eels_edge_spec(wiz, payload):
    """A ready-to-append ComponentSpec for one ``EELSCLEdge``, or None.

    Every rejection path emits its own message, because the native failures are
    unactionable: no exspy is an ``ImportError`` on a package the user has
    never heard of, and no microscope geometry is
    ``AttributeError('Acquisition_instrument')`` raised from inside exspy's
    ``model.append``.
    """
    subshell = ((payload or {}).get("element_subshell")
                or (payload or {}).get("subshell"))
    if not subshell:
        ipc.emit_error("Fit: no edge given — pick one (e.g. O_K) from the list")
        return None
    try:
        from spyde.fitting import ModelSpec
        from spyde.spectroscopy import edges as eels_edges
        from spyde.spectroscopy import prepare_eels_edges
    except ImportError as e:                              # pragma: no cover
        ipc.emit_error(f"Fit: EELS edges are unavailable ({e})")
        return None

    try:
        cspec = eels_edges.edge_component_spec(wiz.signal, subshell)
    except (eels_edges.MissingExtra,
            eels_edges.MissingMicroscopeParameters, ValueError) as e:
        ipc.emit_error(f"Fit: {e}")
        return None
    except Exception as e:
        ipc.emit_error(f"Fit: cannot add the {subshell} edge ({e})")
        return None

    # Attach the batched GOS curves NOW rather than at fit time. The integral
    # depends on the element and the microscope, not the pixel, so it is
    # computed once — and without it the whole model drops off the batched
    # engine onto HyperSpy's one-pixel-at-a-time path (#63). Best-effort: a
    # raw edge still fits, just slowly.
    try:
        prepared, _info = prepare_eels_edges(
            ModelSpec(components=[cspec]), wiz.signal)
        cspec = prepared.components[0]
    except Exception as e:
        log.info("preparing the %s edge for the batched engine failed (%s); "
                 "the fit will fall back to hyperspy", subshell, e)
    return cspec


def fit_add_component(session, plot, payload) -> None:
    wiz, _tree = _wizard(session, plot)
    if wiz is None:
        return
    kind = (payload or {}).get("kind")
    if not kind:
        ipc.emit_error("Fit: no component kind given")
        return

    x = wiz.axis()
    lo, hi = float(np.min(x)), float(np.max(x))

    if kind == EELS_EDGE_KIND:
        cspec = _eels_edge_spec(wiz, payload)
        if cspec is None:
            return
        # NO `_seed_for_preview`: an edge's position is the tabulated onset,
        # which is the whole point of picking it by subshell. Moving it to
        # mid-axis like a gaussian's centre would put O-K wherever the window
        # happens to be centred.
    else:
        try:
            cspec = new_component_spec(
                kind, int((payload or {}).get("order", 2))
                if kind == "Polynomial" else None)
        except Exception as e:
            ipc.emit_error(f"Fit: cannot add {kind} ({e})")
            return
        _seed_for_preview(cspec, lo, hi)
    # Put it on THIS spectrum, or the component arrives five orders of
    # magnitude below the data and looks like it does nothing. A background
    # needs its shape solved through the data, not just its amplitude scaled.
    spectrum = wiz.current_spectrum()
    if not seed_background(cspec, x, spectrum):
        scale_to_data(cspec, x, spectrum)
    # Two of a kind must not start in the same place (degenerate, unfittable)
    # and no peak may wander off the data.
    spread_repeats(cspec, wiz.spec, lo, hi)
    clamp_to_axis(cspec, lo, hi)
    # A unique name per instance — two Gaussians must be separately addressable
    # by the caret and produce two distinct area maps at commit.
    existing = {c.name for c in wiz.spec.components}
    base, n = kind, 1
    while cspec.name in existing:
        n += 1
        cspec.name = f"{base} {n}"
    wiz.spec.append(cspec)
    wiz.result = None                       # the old fit no longer describes it
    wiz._ensure_store(force=True)   # the packed width changed with the model
    wiz.draw_preview()
    wiz.sync_widgets()
    wiz.show_maps()          # one more (or one fewer) map to fill in
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
            MissingExtra, model_for_composition, prepare_eels_edges,
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
    if not info.get("engine_supported"):
        # An EELS edge's GOS integral depends on the element and the
        # microscope, not the pixel, so it is computed once here and the edge
        # becomes batchable. The component stays an `EELSCLEdge` — the model
        # remains a real exspy model and stores on its own signal.
        try:
            spec, einfo = prepare_eels_edges(
                spec, wiz.signal,
                fit_fine_structure=bool(p.get("fit_fine_structure", True)))
            n_coeff = sum(einfo["coefficients"].values())
            if einfo["prepared"]:
                note = (f" {len(einfo['prepared'])} edge(s) batched"
                        + (f", {n_coeff} fine-structure coefficients fitted."
                           if n_coeff else "."))
        except Exception as e:
            log.info("preparing the EELS edges failed (%s); the fit will fall "
                     "back to hyperspy", e)

    wiz.spec = spec
    wiz.result = None
    wiz._ensure_store(force=True)   # the packed width changed with the model
    wiz.draw_preview()
    wiz.sync_widgets()
    wiz.show_maps()          # one more (or one fewer) map to fill in
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
    wiz._ensure_store(force=True)   # the packed width changed with the model
    wiz.clear_preview()
    wiz.draw_preview()
    wiz.sync_widgets()
    wiz.show_maps()          # one more (or one fewer) map to fill in
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
    wiz.update_widgets()      # MOVE, do not rebuild — see update_widgets
    wiz.draw_preview()        # ...and BEFORE the redraw — see _on_widget_drag
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
    wiz.remember(res.values[0], chisq=float(res.chisq[0]))
    wiz.update_widgets()      # handles first, then the push — see _on_widget_drag
    wiz.draw_preview()
    wiz.show_maps()           # one more position filled in
    ok = "converged" if bool(res.converged[0]) else "did not converge"
    wiz.emit_state(f"This spectrum {ok} (chi2 {res.chisq[0]:.3g}). "
                   f"{wiz.store.coverage()[0]} position(s) fitted.")


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

    EVERY path here ends in an ``emit_state``. The caret coalesces navigator
    moves by keeping ONE of these in flight and waiting for the state to come
    back, so a branch that returned silently would stall the next move until a
    2-second wedge timer expired — the pause-and-snap this coalescer exists to
    remove, reintroduced by the back door.
    """
    wiz, _tree = _wizard(session, plot)
    if wiz is None:
        return                  # the caret is gone; nothing is listening
    if not len(wiz.spec):
        wiz.emit_state()
        return
    if wiz.recall():
        wiz.update_widgets()  # handles first, then the push — see _on_widget_drag
        wiz.draw_preview()
        wiz.emit_state("Recalled this position's fit.")
        return
    if bool((payload or {}).get("adaptive")):
        fit_current(session, plot, payload)     # emits its own state
        return
    wiz.draw_preview()          # same model, new spectrum underneath
    wiz.emit_state()


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

    polish = bool(p.get("polish", True))

    def _fit():
        from spyde.fitting.engine import fit_batched
        from spyde.fitting.polish import polish_scan
        from spyde.fitting.relabel import relabel_scan
        from spyde.fitting.seeding import fit_seeded
        use_seeded = seeded and data.ndim > 2
        fn = fit_seeded if use_seeded else fit_batched
        res = fn(spec, data, x, max_iter=max_iter, weights=weights,
                 progress=lambda d, t: ipc.emit_status(
                     f"Fitting {d}/{t} spectra…"))
        if not use_seeded and nav_shape:
            # `fit_seeded` relabels internally (it has to, so the seeds agree
            # before they propagate). A cold batched fit has had no such pass,
            # and two components of a kind land in whichever slot each position
            # happens to pick — measured at 43% agreement on a real scan.
            res.values = relabel_scan(spec, res.values, nav_shape,
                                      converged=res.converged)
        if polish and nav_shape:
            # Rescue the positions that landed somewhere else, each from its
            # best neighbour. Measured 27 -> 1 positions above 1.5x the noise
            # floor, for 0.1 s a pass against 2.8 s for the fit.
            ipc.emit_status("Refitting the positions that fit poorly…")
            res = polish_scan(spec, data, x, res, nav_shape=nav_shape,
                              weights=weights)
            res.values = relabel_scan(spec, res.values, nav_shape,
                                      converged=res.converged)
        return res

    from spyde.actions.lifecycle import supersede
    handle = supersede(getattr(wiz, "_fit_handle", None), tree)
    wiz._fit_handle = handle

    def _done(result):
        handle.retire()
        if handle.stopped or not wiz.still(gen) or wiz._closed:
            return
        wiz.result = result
        wiz.spec = spec
        try:
            wiz.record_run(result, nav_shape)
        except Exception as e:
            log.debug("recording the run into the fit store failed: %s", e)
        # Show the fit at the CURRENT position, so the preview reflects the
        # result rather than the pre-run guess. `current_indices` lives on the
        # navigation SELECTOR, not the plot — reading it off the plot always
        # gave None, so this silently showed position 0's parameters.
        try:
            idx = wiz.current_indices()
            flat = 0
            if idx is not None and nav_shape:
                flat = int(np.ravel_multi_index(
                    tuple(int(i) for i in reversed(idx)), nav_shape))
            spec.set_flat_values(result.values[flat])
        except Exception as e:
            log.debug("seeding the post-fit preview failed: %s", e)
        wiz.update_widgets()  # the curves moved; the handles go with them
        wiz.draw_preview()    # ...and the full push must come AFTER them
        # Show the maps NOW, not only on Commit. "Fit all spectra" produced a
        # result you could not look at: the only way to see where the fit
        # succeeded was to scrub the navigator one pixel at a time.
        try:
            wiz.show_maps()
        except Exception as e:
            log.debug("opening the fit result maps failed: %s", e)
        pct = 100.0 * result.convergence_rate
        rescued = int(getattr(result, "polish_improved", 0) or 0)
        extra = f", {rescued} rescued by refitting" if rescued else ""
        ipc.emit_status(f"Fit complete — {pct:.0f}% converged "
                        f"({result.n_iter} iterations{extra})")
        wiz.emit_state(f"{pct:.0f}% converged{extra}. "
                       f"{wiz.poor_positions()} position(s) fit poorly.")

    wiz.run_on_worker(_fit, name="fit-run", on_done=_done)


def fit_save_model(session, plot, payload=None) -> None:
    """Store the model — components AND every position's fit — on the signal.

    Straight through HyperSpy's ``m.store(name)``, so it lands in the signal's
    own ``models`` and travels with the dataset: save the .hspy/.zspy and the
    fit is inside it. No SpyDE format, nothing to keep in step.
    """
    wiz, _tree = _wizard(session, plot)
    if wiz is None or wiz.store is None:
        ipc.emit_error("Fit: open the Fit caret first")
        return
    name = str((payload or {}).get("name") or "").strip() or "spyde fit"
    try:
        wiz.store.save_as(name)
    except Exception as e:
        ipc.emit_error(f"Fit: could not store the model ({e})")
        return
    done, total = wiz.store.coverage()
    ipc.emit_status(f"Model '{name}' stored on the signal ({done}/{total} "
                    f"positions) — it saves with the dataset")
    wiz.emit_state(f"Stored as '{name}'. Save the dataset to keep it.")


def fit_load_model(session, plot, payload=None) -> None:
    """Restore a model stored on this signal, per-position fits and all."""
    from spyde.fitting import ModelSpec
    wiz, tree = _wizard(session, plot)
    if wiz is None:
        ipc.emit_error("Fit: open the Fit caret first")
        return
    names = wiz.store.stored_names() if wiz.store is not None else []
    name = str((payload or {}).get("name") or "").strip()
    if not name:
        name = names[-1] if names else ""
    if not name:
        ipc.emit_error("Fit: this signal has no stored models")
        return
    try:
        spec, store = FitStore.restore(ModelSpec, wiz.signal, name)
    except Exception as e:
        ipc.emit_error(f"Fit: could not restore '{name}' ({e})")
        return
    wiz.spec = spec
    tree.fit_store = store
    wiz.result = None
    wiz.recall()
    wiz.sync_widgets()
    wiz.update_widgets()
    wiz.draw_preview()
    wiz.show_maps()
    done, total = store.coverage()
    wiz.emit_state(f"Restored '{name}' — {done}/{total} positions already fitted.")


def fit_refit_poor(session, plot, payload=None) -> None:
    """Refit only the positions that fit worse than their neighbours.

    Each is restarted from its best neighbour's answer rather than from the
    model's defaults — a spectrum image is smooth, so next door is nearly the
    answer here. Measured on hyperspy's two_gaussians: positions worse than
    1.5x the noise floor go 27 -> 1, total chisq down 9.7%, in 0.1 s a pass.

    Run automatically at the end of `fit_run`; this is the button for doing it
    again after changing the model, or for pushing harder on what is left.
    """
    wiz, tree = _wizard(session, plot)
    if wiz is None or wiz.result is None:
        ipc.emit_error("Fit: fit all spectra before refitting the poor ones")
        return
    nav_shape = wiz.nav_shape()
    if not nav_shape:
        ipc.emit_error("Fit: this signal has no navigation grid to compare")
        return

    spec = wiz.spec.copy()
    x = wiz.axis()
    data = np.asarray(wiz.signal.data, float)
    result = wiz.result
    factor = float((payload or {}).get("factor", 1.5))
    gen = wiz.guard()

    from spyde.actions.lifecycle import supersede
    handle = supersede(getattr(wiz, "_fit_handle", None), tree)
    wiz._fit_handle = handle

    def _work():
        if handle.stopped:
            return None
        from spyde.fitting.polish import polish_scan
        from spyde.fitting.relabel import relabel_scan
        res = polish_scan(spec, data, x, result, nav_shape=nav_shape,
                          factor=factor)
        res.values = relabel_scan(spec, res.values, nav_shape,
                                  converged=res.converged)
        return res

    def _done(res):
        handle.retire()
        if (res is None or handle.stopped or not wiz.still(gen)
                or wiz._closed):
            return
        wiz.result = res
        wiz.record_run(res, nav_shape)
        rescued = int(getattr(res, "polish_improved", 0) or 0)
        wiz.recall()
        wiz.update_widgets()  # handles first, then the push — see _on_widget_drag
        wiz.draw_preview()
        try:
            wiz.show_maps()
        except Exception as e:
            log.debug("re-opening the fit result maps failed: %s", e)
        wiz.emit_state(f"Refit {rescued} poor position(s). "
                       f"{wiz.poor_positions()} still poor.")

    ipc.emit_status("Refitting the positions that fit poorly…")
    wiz.run_on_worker(_work, name="fit-refit-poor", on_done=_done)


def fit_commit(session, plot, payload=None) -> None:
    wiz, _tree = _wizard(session, plot)
    if wiz is None:
        ipc.emit_error("Fit: nothing to commit")
        return
    wiz.commit()
