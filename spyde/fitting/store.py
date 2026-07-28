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
    """Fitted parameters per navigation position, mirrored to HyperSpy's maps.

    ``indices`` throughout are the NAVIGATOR's order (x first, as
    ``axes_manager.indices`` gives them). HyperSpy's maps are indexed the other
    way round (y first, like the data), so every lookup reverses — getting that
    backwards transposes the whole scan, and on a square scan nothing would
    look wrong.
    """

    def __init__(self, spec, signal):
        self.spec = spec
        self._signal = signal
        nav = self.nav_shape
        n = len(spec.parameter_names())
        # (values, std, is_set) per position — the same triple HyperSpy keeps
        # per parameter, held once for the whole model in `parameter_names()`
        # order, which is the engine's column order.
        #
        # Held HERE and mirrored into a model, rather than living inside one,
        # because not every component SpyDE fits has a HyperSpy counterpart: a
        # tabulated EELS edge (#63) exists only here, so `to_model` cannot
        # rebuild it. A store that lived inside a model would fail to exist for
        # exactly the models the composition path produces — which is how this
        # first showed up, as `'NoneType' has no attribute 'coverage'` the
        # moment you fitted one.
        self._n = n
        self._values = np.zeros((n,) + nav, dtype=np.float64)
        self._std = np.full((n,) + nav, np.nan, dtype=np.float64)
        self._set = np.zeros(nav, dtype=bool)
        self._chisq = np.full(nav, np.nan, dtype=np.float64)
        # Built when it CAN be. This is what `save_as` pushes the arrays into,
        # so a model HyperSpy can represent still stores on the signal as its
        # own and travels with the dataset.
        try:
            self.model = spec.to_model(signal)
        except Exception as e:
            log.info("no HyperSpy model for this spec (%s) — the fit is held "
                     "here and cannot be stored on the signal", e)
            self.model = None

    def _model_params(self) -> list:
        """The model's scalar parameters in `parameter_names()` order, or []."""
        return _scalar_params(self.model) if self.model is not None else []

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
        return self._n

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
        if values.size != self._n:
            log.debug("store: %d values for %d parameters — refused",
                      values.size, self._n)
            return False
        self._values[(slice(None),) + key] = values
        self._set[key] = True
        if std is not None:
            self._std[(slice(None),) + key] = np.asarray(std, float).ravel()
        if chisq is not None:
            self._chisq[key] = float(chisq)
        return True

    def get(self, indices):
        """This position's parameters, or None if it has not been fitted."""
        key = self._key(indices)
        if key is None or not bool(self._set[key]):
            return None
        return np.array(self._values[(slice(None),) + key])

    def is_set(self, indices) -> bool:
        key = self._key(indices)
        return bool(key is not None and self._set[key])

    def chisq_at(self, indices):
        key = self._key(indices)
        return None if key is None else float(self._chisq[key])

    # ── the whole scan ────────────────────────────────────────────────────
    def put_all(self, values, chisq=None, std=None) -> int:
        """Write a whole-scan result. Returns how many positions landed."""
        values = np.asarray(values, float)
        if values.ndim != 2 or values.shape[1] != self._n:
            log.debug("store: whole-scan values %s do not match %d parameters",
                      values.shape, self._n)
            return 0
        if values.shape[0] != self.n_positions:
            log.debug("store: %d rows for %d positions", values.shape[0],
                      self.n_positions)
            return 0
        # Row r is the fit to `data[r // nx, r % nx]`, and the arrays are
        # (n,) + nav — so this is a reshape plus moving the parameter axis to
        # the front, and it lands where `get` reads it back.
        self._values[...] = np.moveaxis(
            values.reshape(self.nav_shape + (self._n,)), -1, 0)
        self._set[...] = True
        if std is not None:
            self._std[...] = np.moveaxis(
                np.asarray(std, float).reshape(self.nav_shape + (self._n,)),
                -1, 0)
        if chisq is not None:
            self._chisq[...] = np.asarray(chisq, float).reshape(self.nav_shape)
        return self.n_positions

    def clear(self) -> None:
        """Forget every position. Called when the component LIST changes: the
        stored vectors are positional, so after an add or remove they would be
        silently reinterpreted against the wrong parameters."""
        self._set[...] = False
        self._values[...] = 0.0
        self._std[...] = np.nan
        self._chisq[...] = np.nan

    def forget(self, indices) -> None:
        key = self._key(indices)
        if key is not None:
            self._set[key] = False
            self._chisq[key] = np.nan

    # ── what has been fitted ──────────────────────────────────────────────
    def set_mask(self) -> np.ndarray:
        """True where this position has a stored answer."""
        return self._set

    def coverage(self) -> tuple[int, int]:
        return int(self._set.sum()), self.n_positions

    @property
    def chisq(self) -> np.ndarray:
        """chi-squared per position; NaN where nothing has been fitted."""
        return self._chisq

    def values_array(self) -> np.ndarray:
        """(P, n) of every position's parameters, NaN where unset."""
        out = np.moveaxis(self._values, 0, -1).reshape(
            self.n_positions, self._n).astype(np.float64, copy=True)
        out[~self._set.ravel()] = np.nan
        return out

    # ── save / load, through HyperSpy ─────────────────────────────────────
    # `m.store(name)` puts the model — components, parameters AND every
    # position's map — into the signal's own `models`, so it travels with the
    # dataset: save the .hspy/.zspy and the fit is in it; `hs.load` then
    # `signal.models.restore(name)` brings it all back, `is_set` included.
    # Nothing here writes a format of its own.
    def save_as(self, name: str) -> None:
        """Push every position into the model's own maps and store it.

        Raises if this spec has no HyperSpy model — a tabulated EELS edge (#63)
        exists only in SpyDE, so there is genuinely nothing for HyperSpy to
        store. Saying so beats writing a private format that only SpyDE can
        read back.
        """
        if self.model is None:
            raise ValueError(
                "this model has components HyperSpy cannot represent "
                "(tabulated edges) — it cannot be stored on the signal")
        params = self._model_params()
        if len(params) != self._n:
            raise ValueError(
                f"the model has {len(params)} scalar parameters but the fit "
                f"holds {self._n} — refusing to store a mismatched pair")
        for i, par in enumerate(params):
            par.map["values"][...] = self._values[i]
            par.map["std"][...] = self._std[i]
            par.map["is_set"][...] = self._set
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
        # maps are on the restored model, so pull them across rather than
        # refitting anything. `is_set` comes with them — without it every
        # unfitted hole in a component map would come back a confident zero.
        src = _scalar_params(model)
        if len(src) == store._n:
            for i, par in enumerate(src):
                store._values[i] = par.map["values"]
                store._std[i] = par.map["std"]
            store._set[...] = np.logical_and.reduce(
                [np.asarray(p.map["is_set"], bool) for p in src])
        else:
            log.warning("restored model has %d parameters, the store expects "
                        "%d — values not carried across", len(src), store._n)
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
