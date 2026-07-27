"""store.py — the fitted parameters at every navigation position.

**This is HyperSpy's own store, not a parallel one.** A HyperSpy parameter
already carries ``parameter.map``, a structured array over the navigation grid
with ``values`` / ``std`` / ``is_set``, and a model already carries ``chisq``
as a signal over the same grid. That is exactly "hold the parameters for the
model at each position", it is what ``m.multifit()`` fills, and it is what
``m.store()`` / ``s.models.restore()`` persist.

So ``FitStore`` wraps a live model built from the spec and writes into those
maps. What SpyDE adds is only the packing: the batched engine works in the flat
column order of :meth:`ModelSpec.parameter_names`, so this converts between
that vector and the per-parameter maps. Nothing here duplicates the storage.

What this replaced was a dict keyed by position tuple, which had none of it:
no ``is_set`` (so "fitted" and "fitted to zero" were the same thing), no
``std``, no chi-squared, no way to hand the result to HyperSpy, and it was
built up sparsely as the user explored rather than existing from the start.

**The maps exist from the moment the caret opens**, empty. A position that has
not been fitted reads NaN, which is what draws the "not yet" holes in the
component maps as they fill in — the same shape as every other progressive
calculation in SpyDE.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def _scalar_params(model) -> list:
    """A live model's scalar parameters, in ``parameter_names()`` order.

    Vector parameters (EELSCLEdge's fine_structure_coeff) are skipped, exactly
    as ``ComponentSpec.scalar_parameters`` skips them — the packed vector holds
    one scalar per column, so the two orders must agree.
    """
    out = []
    for comp in model:
        if not getattr(comp, "active", True):
            continue
        for par in comp.parameters:
            if int(getattr(par, "_number_of_elements", 1) or 1) == 1:
                out.append(par)
    return out


class FitStore:
    """Fitted parameters per navigation position, in HyperSpy's parameter maps.

    ``indices`` throughout are the NAVIGATOR's order (x first, as
    ``axes_manager.indices`` gives them). HyperSpy's maps are indexed the other
    way round (y first, like the data), so every lookup reverses — getting that
    backwards transposes the whole scan, and on a square scan nothing would
    look wrong.
    """

    def __init__(self, spec, signal):
        self.model = spec.to_model(signal)
        self.spec = spec
        self._signal = signal
        # The parameter objects in the SAME order as `spec.parameter_names()`,
        # which is the engine's column order. Anything that maps a fitted
        # vector onto storage has to use that order rather than re-deriving one.
        # `to_model` appends every component of the spec in order, so the live
        # model's i-th component is the spec's i-th; the ACTIVE ones, in that
        # order, are what `parameter_names()` packs.
        live = list(self.model)
        self._params = []
        for i, cspec in enumerate(spec.components):
            if not cspec.active or i >= len(live):
                continue
            for pspec in cspec.scalar_parameters:
                par = getattr(live[i], pspec.name, None)
                if par is None:
                    log.debug("component %s has no parameter %s",
                              cspec.kind, pspec.name)
                self._params.append(par)
        self._chisq = np.full(self.nav_shape, np.nan, dtype=np.float64)

    # ── geometry ──────────────────────────────────────────────────────────
    @property
    def nav_shape(self) -> tuple:
        """The maps' shape — y first, matching the data and HyperSpy's maps."""
        try:
            nav = tuple(int(n) for n in
                        self._signal.axes_manager.navigation_shape)
            return tuple(reversed(nav)) or (1,)
        except Exception:
            return (1,)

    @property
    def n_positions(self) -> int:
        return int(np.prod(self.nav_shape))

    @property
    def n_params(self) -> int:
        return len(self._params)

    def _key(self, indices):
        """Navigator indices -> map index. THE SAME WAY THE DISPLAY READS THEM.

        ``_build_nav_lazy_slice`` / ``get_local_frame`` do ``data[point]`` with
        the selector's indices exactly as given, so the spectrum on screen at
        crosshair ``(cx, cy)`` is ``data[cx, cy]``. The store has to agree,
        because its whole job is to answer "what was fitted to THAT spectrum".

        Reversing here — reasoning from ``axes_manager.navigation_shape``
        rather than from what the display actually does — transposed the whole
        scan. It was invisible on the 32x32 tutorial: the shapes matched, the
        coverage read 1024/1024, every recall succeeded, and the diagonal
        positions were even correct. Every OTHER position quietly showed its
        transpose's fit, which is a plausible-looking curve that misses the
        data (measured: 20% median misfit against the spectrum, 47% worst).
        """
        if indices is None:
            return None
        key = tuple(int(v) for v in tuple(indices))
        if len(key) != len(self.nav_shape):
            return None
        if any(not (0 <= k < n) for k, n in zip(key, self.nav_shape)):
            return None
        return key

    # ── one position ──────────────────────────────────────────────────────
    def put(self, indices, values, chisq: float | None = None,
            std=None) -> bool:
        key = self._key(indices)
        if key is None:
            return False
        values = np.asarray(values, float).ravel()
        if values.size != self.n_params:
            log.debug("store: %d values for %d parameters — refused",
                      values.size, self.n_params)
            return False
        for par, v, i in zip(self._params, values, range(len(values))):
            if par is None:
                continue
            par.map["values"][key] = float(v)
            par.map["is_set"][key] = True
            if std is not None:
                par.map["std"][key] = float(np.asarray(std).ravel()[i])
        if chisq is not None:
            self._chisq[key] = float(chisq)
        return True

    def get(self, indices):
        """This position's parameters, or None if it has not been fitted."""
        key = self._key(indices)
        if key is None:
            return None
        out = np.empty(self.n_params, dtype=np.float64)
        for i, par in enumerate(self._params):
            if par is None or not bool(par.map["is_set"][key]):
                return None
            out[i] = float(par.map["values"][key])
        return out

    def is_set(self, indices) -> bool:
        return self.get(indices) is not None

    def chisq_at(self, indices):
        key = self._key(indices)
        return None if key is None else float(self._chisq[key])

    # ── the whole scan ────────────────────────────────────────────────────
    def put_all(self, values, chisq=None, std=None) -> int:
        """Write a whole-scan result. Returns how many positions landed."""
        values = np.asarray(values, float)
        if values.ndim != 2 or values.shape[1] != self.n_params:
            log.debug("store: whole-scan values %s do not match %d parameters",
                      values.shape, self.n_params)
            return 0
        n = min(values.shape[0], self.n_positions)
        shaped = values[:n].reshape(*self.nav_shape, self.n_params) \
            if n == self.n_positions else None
        if shaped is None:
            log.debug("store: %d rows for %d positions", n, self.n_positions)
            return 0
        for i, par in enumerate(self._params):
            if par is None:
                continue
            par.map["values"][...] = shaped[..., i]
            par.map["is_set"][...] = True
            if std is not None:
                par.map["std"][...] = np.asarray(std, float).reshape(
                    *self.nav_shape, self.n_params)[..., i]
        if chisq is not None:
            self._chisq[...] = np.asarray(chisq, float).reshape(self.nav_shape)
        return self.n_positions

    def clear(self) -> None:
        """Forget every position. Called when the component LIST changes: the
        stored vectors are positional, so after an add or remove they would be
        silently reinterpreted against the wrong parameters."""
        for par in self._params:
            if par is not None:
                par.map["is_set"][...] = False
                par.map["values"][...] = 0.0
        self._chisq[...] = np.nan

    def forget(self, indices) -> None:
        key = self._key(indices)
        if key is None:
            return
        for par in self._params:
            if par is not None:
                par.map["is_set"][key] = False
        self._chisq[key] = np.nan

    # ── what has been fitted ──────────────────────────────────────────────
    def set_mask(self) -> np.ndarray:
        """True where every parameter has a stored value."""
        if not self._params or self._params[0] is None:
            return np.zeros(self.nav_shape, bool)
        mask = np.ones(self.nav_shape, bool)
        for par in self._params:
            if par is not None:
                mask &= par.map["is_set"]
        return mask

    def coverage(self) -> tuple[int, int]:
        return int(self.set_mask().sum()), self.n_positions

    @property
    def chisq(self) -> np.ndarray:
        """chi-squared per position; NaN where nothing has been fitted."""
        return self._chisq

    def values_array(self) -> np.ndarray:
        """(P, n) of every position's parameters, NaN where unset."""
        out = np.full((self.n_positions, self.n_params), np.nan)
        for i, par in enumerate(self._params):
            if par is None:
                continue
            v = np.asarray(par.map["values"], float).ravel()
            s = np.asarray(par.map["is_set"], bool).ravel()
            out[s, i] = v[s]
        return out

    # ── save / load, through HyperSpy ─────────────────────────────────────
    # `m.store(name)` puts the model — components, parameters AND every
    # position's map — into the signal's own `models`, so it travels with the
    # dataset: save the .hspy/.zspy and the fit is in it; `hs.load` then
    # `signal.models.restore(name)` brings it all back, `is_set` included.
    # Nothing here writes a format of its own.
    def save_as(self, name: str) -> None:
        self.model.store(str(name))

    def stored_names(self) -> list[str]:
        try:
            return list(self._signal.models._models.as_dictionary().keys())
        except Exception as e:
            log.debug("listing stored models failed: %s", e)
            return []

    @classmethod
    def restore(cls, spec_cls, signal, name: str):
        """Load a stored model back into a (spec, store) pair.

        The SPEC is read from the restored model rather than kept alongside it:
        the model is the thing that was saved, so it is the thing that decides
        what the components are.
        """
        model = signal.models.restore(str(name))
        spec = spec_cls.from_model(model)
        store = cls(spec, signal)
        # `from_model` reads the CURRENT position's values; the per-position
        # maps are on the restored model, so copy them across parameter by
        # parameter rather than refitting anything.
        for dst, src in zip(store._params, _scalar_params(model)):
            if dst is not None and src is not None:
                dst.map[...] = src.map
        return spec, store

    # ── the maps a user looks at ──────────────────────────────────────────
    def maps(self, x) -> dict[str, np.ndarray]:
        """Integrated area under each component, plus chi-squared.

        NaN at a position that has not been fitted — that is what leaves the
        holes visible while the scan fills in, rather than drawing zeros and
        claiming the fit found nothing there.
        """
        import torch
        from spyde.fitting import components as tcomp

        values = self.values_array()
        done = self.set_mask().ravel()
        out: dict[str, np.ndarray] = {}
        xt = torch.as_tensor(np.asarray(x, float))
        i = 0
        for c in self.spec.active_components:
            n = len(c.scalar_parameters)
            m = np.full(self.n_positions, np.nan)
            if done.any():
                try:
                    block = np.nan_to_num(values[done, i:i + n])
                    y = tcomp.component_for(c)(
                        xt, torch.as_tensor(block)).numpy()
                    area = (np.trapezoid(y, np.asarray(x, float), axis=1)
                            if hasattr(np, "trapezoid")
                            else np.trapz(y, x, axis=1))
                    m[done] = area
                except Exception as e:
                    log.debug("area map for %s failed: %s", c.name, e)
            out[c.name] = m.reshape(self.nav_shape)
            i += n
        out["chi squared"] = self._chisq.copy()
        return out
