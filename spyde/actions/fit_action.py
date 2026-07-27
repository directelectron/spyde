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
        self.spec = ModelSpec()
        self.result = None
        self._preview_name = "fit_preview"

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
            p1.add_line(np.asarray(y, np.float32), x_axis=self.axis(),
                        name=self._preview_name, color="#f5a97f", linewidth=1.6)
        except Exception as e:
            log.debug("drawing the fit preview failed: %s", e)

    def clear_preview(self) -> None:
        p1 = getattr(self.plot, "_plot1d", None)
        if p1 is None:
            return
        for meth in ("remove_line", "remove_lines"):
            fn = getattr(p1, meth, None)
            if fn is None:
                continue
            try:
                fn(self._preview_name)
                return
            except Exception as e:
                log.debug("clearing the fit preview via %s failed: %s", meth, e)

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
    wiz.emit_state("Add a component to begin.")


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
    wiz.emit_state(f"Added {cspec.name}.")


def fit_remove_component(session, plot, payload) -> None:
    wiz, _tree = _wizard(session, plot)
    if wiz is None:
        return
    name = (payload or {}).get("name")
    wiz.spec.components = [c for c in wiz.spec.components if c.name != name]
    wiz.result = None
    wiz.clear_preview()
    wiz.draw_preview()
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
    wiz.emit_state()


def fit_tune(session, plot, payload=None) -> None:
    """Debounced redraw — the caret's live edit path."""
    wiz, _tree = _wizard(session, plot)
    if wiz is not None:
        wiz.draw_preview()


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
