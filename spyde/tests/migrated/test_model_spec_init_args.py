"""ModelSpec must rebuild components that take CONSTRUCTOR arguments.

Two real bugs this closes, found by trying to round-trip a real EELS model:

* ``EELSCLEdge`` raises outright — ``element_subshell`` is required, so
  ``to_model`` could not rebuild any EELS model at all.
* ``Polynomial`` is worse because it does NOT raise. Rebuilt bare it comes back
  at its default order, so an order-6 EDS background silently loses a3..a6 —
  ``to_model`` skips the parameters that no longer exist and returns a
  different model that still fits and still looks plausible.

The original round-trip tests missed both because they only ever went
model -> spec, never spec -> model with a component of this kind.
"""
from __future__ import annotations

import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy.components1d import Polynomial

from spyde.fitting import ModelSpec
from spyde.fitting.spec import ComponentSpec, ParameterSpec


def _signal(n=512, lo=200.0, hi=800.0):
    s = hs.signals.Signal1D(np.random.default_rng(0).random((2, 2, n)) + 1.0)
    s.axes_manager.signal_axes[0].offset = lo
    s.axes_manager.signal_axes[0].scale = (hi - lo) / n
    return s


class TestPolynomialOrder:
    @pytest.mark.parametrize("order", [1, 2, 6])
    def test_order_survives_the_round_trip(self, order):
        sig = _signal()
        m = sig.create_model()
        m.append(Polynomial(order=order))
        for k in range(order + 1):
            getattr(m[0], f"a{k}").value = 0.5 * (k + 1)

        spec = ModelSpec.from_model(m)
        assert spec[0].init_args == {"order": order}

        back = ModelSpec.from_model(spec.to_model(sig))
        assert len(back[0].parameters) == order + 1, \
            "rebuilt Polynomial lost coefficients"
        np.testing.assert_allclose(back.flat_values(), spec.flat_values())

    def test_high_order_coefficients_are_not_silently_dropped(self):
        """The specific failure: a bare rebuild keeps only the default order's
        coefficients, and every later one is quietly skipped."""
        sig = _signal()
        m = sig.create_model()
        m.append(Polynomial(order=6))
        for k in range(7):
            getattr(m[0], f"a{k}").value = float(k + 1)
        spec = ModelSpec.from_model(m)
        rebuilt = spec.to_model(sig)
        names = [p.name for p in rebuilt[0].parameters]
        assert "a6" in names, names
        assert float(np.ravel(rebuilt[0].a6.value)[0]) == pytest.approx(7.0)

    def test_order_survives_json(self):
        sig = _signal()
        m = sig.create_model()
        m.append(Polynomial(order=4))
        spec = ModelSpec.from_model(m)
        back = ModelSpec.from_dict(spec.to_dict())
        assert back[0].init_args == {"order": 4}
        assert len(back.to_model(sig)[0].parameters) == 5


class TestEelsEdge:
    def test_edge_round_trips(self):
        pytest.importorskip("exspy", reason="needs the eels extra")
        from spyde.data import eels_si
        from spyde.spectroscopy import model_for_composition

        s = eels_si(nav=(2, 2), n_channels=256)
        spec, _ = model_for_composition(s, ["C", "N", "O"])
        edges = [c for c in spec if c.kind == "EELSCLEdge"]
        assert edges, "no edges in the composition model"
        for e in edges:
            assert e.init_args.get("element_subshell"), e.name

        # The whole point: this used to raise.
        rebuilt = spec.to_model(s)
        kinds = [getattr(c, "_id_name", "") for c in rebuilt]
        assert kinds.count("EELSCLEdge") == len(edges)

    def test_edge_keeps_its_subshell_identity(self):
        pytest.importorskip("exspy", reason="needs the eels extra")
        from spyde.data import eels_si
        from spyde.spectroscopy import model_for_composition

        s = eels_si(nav=(2, 2), n_channels=256)
        spec, _ = model_for_composition(s, ["C", "N", "O"])
        rebuilt = ModelSpec.from_model(spec.to_model(s))
        assert {c.name for c in rebuilt if c.kind == "EELSCLEdge"} == \
               {c.name for c in spec if c.kind == "EELSCLEdge"}

    def test_edge_onsets_survive(self):
        pytest.importorskip("exspy", reason="needs the eels extra")
        from spyde.data import eels_si
        from spyde.spectroscopy import model_for_composition

        s = eels_si(nav=(2, 2), n_channels=256)
        spec, _ = model_for_composition(s, ["C", "N", "O"])
        before = {c.name: c["onset_energy"].value
                  for c in spec if c.kind == "EELSCLEdge"}
        back = ModelSpec.from_model(spec.to_model(s))
        after = {c.name: c["onset_energy"].value
                 for c in back if c.kind == "EELSCLEdge"}
        assert before == pytest.approx(after)

    def test_edge_round_trips_through_json(self):
        pytest.importorskip("exspy", reason="needs the eels extra")
        import json
        from spyde.data import eels_si
        from spyde.spectroscopy import model_for_composition

        s = eels_si(nav=(2, 2), n_channels=256)
        spec, _ = model_for_composition(s, ["C", "O"])
        revived = ModelSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        assert revived.to_model(s)          # must not raise


class TestUnrebuildableComponent:
    def test_missing_init_args_give_an_actionable_error(self):
        """If a component needs constructor arguments nobody captured, say so
        and say where to fix it — not a bare TypeError from deep inside
        hyperspy."""
        pytest.importorskip("exspy", reason="needs the eels extra")
        spec = ModelSpec(components=[ComponentSpec(
            kind="EELSCLEdge", name="O_K",
            parameters=[ParameterSpec("intensity", 1.0)])])
        with pytest.raises(ValueError, match="_INIT_ARGS"):
            spec.to_model(_signal())

    def test_components_without_init_args_are_unaffected(self):
        sig = _signal()
        m = sig.create_model()
        from hyperspy.components1d import Gaussian, Offset
        m.extend([Offset(), Gaussian()])
        spec = ModelSpec.from_model(m)
        assert all(c.init_args == {} for c in spec)
        assert len(ModelSpec.from_model(spec.to_model(sig))) == 2
