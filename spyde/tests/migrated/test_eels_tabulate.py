"""Tabulated EELS edges through the batched engine (#63).

The claim under test: an EELS model that could only be fitted by HyperSpy, one
pixel at a time, becomes fittable by the batched engine — and the answer still
means what it meant before.

So the tests check both halves. That the tabulated shape REPRODUCES the
hyperspy edge (otherwise the speed is worthless), and that fitting it recovers
the intensities the data was built from.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("exspy", reason="needs the eels extra")

from spyde.data import eels_si
from spyde.fitting import ModelSpec
from spyde.fitting import components as tc
from spyde.fitting.components import TABULATED_KIND, tabulated_component
from spyde.fitting.engine import fit_batched
from spyde.fitting.spec import ComponentSpec, ParameterSpec
from spyde.spectroscopy import model_for_composition, onset_energies, tabulate_model


@pytest.fixture(scope="module")
def eels():
    return eels_si(nav=(3, 3), n_channels=512)


class TestTabulatedComponent:
    def test_reproduces_the_table_at_unit_intensity(self):
        x = np.linspace(0.0, 10.0, 101)
        table = np.sin(x) ** 2
        comp = tabulated_component(table, x[0], x[1] - x[0])
        got = comp(torch.as_tensor(x), torch.tensor([[1.0, 0.0]],
                                                    dtype=torch.float64))
        np.testing.assert_allclose(got.numpy()[0], table, atol=1e-12)

    def test_intensity_scales_linearly(self):
        x = np.linspace(0.0, 10.0, 101)
        comp = tabulated_component(np.sin(x) ** 2, x[0], x[1] - x[0])
        xt = torch.as_tensor(x)
        one = comp(xt, torch.tensor([[1.0, 0.0]], dtype=torch.float64))
        three = comp(xt, torch.tensor([[3.0, 0.0]], dtype=torch.float64))
        torch.testing.assert_close(three, 3 * one)

    def test_onset_shift_moves_the_shape_up_in_energy(self):
        """A POSITIVE shift must move the edge to higher energy — the sign a
        user expects. Getting this backwards still fits, just with the shift
        inverted, and nothing else would catch it."""
        x = np.linspace(0.0, 100.0, 501)
        table = (x > 50.0).astype(float)          # a step at 50
        comp = tabulated_component(table, x[0], x[1] - x[0])
        y = comp(torch.as_tensor(x),
                 torch.tensor([[1.0, 10.0]], dtype=torch.float64)).numpy()[0]
        assert x[int(np.argmax(y > 0.5))] == pytest.approx(60.0, abs=0.5)

    def test_holds_the_end_value_outside_the_table(self):
        """Extrapolating a GOS tail would invent signal where the measurement
        has none."""
        x = np.linspace(0.0, 10.0, 51)
        comp = tabulated_component(np.ones_like(x) * 2.0, x[0], x[1] - x[0])
        far = comp(torch.as_tensor(np.array([-50.0, 60.0])),
                   torch.tensor([[1.0, 0.0]], dtype=torch.float64))
        np.testing.assert_allclose(far.numpy()[0], [2.0, 2.0])

    def test_analytic_gradient_matches_autodiff_on_a_smooth_table(self):
        """Checked on a SMOOTH table, where the two definitions coincide.

        The shift derivative here is a deliberate central difference, not the
        exact piecewise-linear slope: the exact one is a step function that
        jumps at every segment boundary, so LM chatters as the shift crosses a
        channel. On a table with a KINK the two therefore differ within one
        channel of it — by design — so a kinked table would be testing the
        artefact rather than the machinery.
        """
        from torch.func import jacfwd
        x = np.linspace(200.0, 800.0, 301)
        table = np.exp(-0.5 * ((x - 500.0) / 40.0) ** 2)
        comp = tabulated_component(table, x[0], x[1] - x[0])
        xt = torch.as_tensor(x)
        v = torch.tensor([[2.0, 1.5]], dtype=torch.float64)
        auto = jacfwd(lambda p: comp(xt, p.unsqueeze(0)).squeeze(0))(v[0])
        got = comp.grad(xt, v)[0]

        # The INTENSITY column is exact — it is just the table.
        torch.testing.assert_close(got[:, 0], auto[:, 0], rtol=1e-9, atol=1e-9)

        # The SHIFT column is smoothed over one channel, so it is compared as a
        # small perturbation of the autodiff gradient rather than as an
        # equality. The largest disagreement sits at the peak, where the left
        # and right segment slopes cancel in the central difference but
        # autodiff returns one of them — a real property of the smoothing, not
        # an error.
        # Measured at ~1.5% of the gradient's own scale on this table; the
        # bound is the O(dx * f'') error a one-channel central difference must
        # have, not a number tuned until it passed.
        scale = auto[:, 1].abs().max()
        assert (got[:, 1] - auto[:, 1]).abs().max() < 0.05 * scale
        # And it must still point the same way wherever the slope is real.
        steep = auto[:, 1].abs() > 0.1 * scale
        assert (torch.sign(got[steep, 1]) == torch.sign(auto[steep, 1])).all()

    def test_shift_gradient_has_the_right_sign_on_a_kinked_table(self):
        """A kinked (edge-like) table is the real case, so the gradient still
        has to point the right way there even though it is smoothed."""
        x = np.linspace(200.0, 800.0, 301)
        table = np.clip(x - 400.0, 0, None) ** 0.5
        comp = tabulated_component(table, x[0], x[1] - x[0])
        g = comp.grad(torch.as_tensor(x),
                      torch.tensor([[2.0, 0.0]], dtype=torch.float64))[0]
        above = x > 420.0
        # Shifting the edge UP in energy reduces intensity above the onset,
        # where the table is rising.
        assert (g[above, 1] < 0).all()

    def test_missing_table_is_an_actionable_error(self):
        spec = ComponentSpec(kind=TABULATED_KIND, name="O_K",
                             parameters=[ParameterSpec("intensity", 1.0)])
        with pytest.raises(ValueError, match="tabulate"):
            tc.component_for(spec)


class TestTabulateModel:
    def test_edges_become_tabulated_components(self, eels):
        spec, info = model_for_composition(eels, ["C", "N", "O"])
        assert info["engine_supported"] is False          # before

        tab, tinfo = tabulate_model(spec, eels)
        assert set(tinfo["tabulated"]) == {"C_K", "N_K", "O_K"}
        assert not tinfo["skipped"]
        assert [c.kind for c in tab].count(TABULATED_KIND) == 3

    def test_the_batched_engine_can_now_fit_it(self, eels):
        """The headline: an EELS model that HyperSpy alone could fit now goes
        through the batched engine."""
        spec, _ = model_for_composition(eels, ["C", "N", "O"])
        tab, _ = tabulate_model(spec, eels)
        assert tc.supports(tab) is True
        assert tc.has_analytic_grad(tab) is True

    def test_the_table_reproduces_the_hyperspy_edge(self, eels):
        """If the tabulated shape is not the edge, the speed is worthless."""
        spec, _ = model_for_composition(eels, ["O"])
        tab, tinfo = tabulate_model(spec, eels)
        x = np.asarray(eels.axes_manager.signal_axes[0].axis, float)

        model = spec.to_model(eels)
        edge = [c for c in model if getattr(c, "_id_name", "") == "EELSCLEdge"][0]
        edge.intensity.value = 1.0
        want = np.nan_to_num(np.asarray(edge.function(x), float))

        comp = [c for c in tab if c.kind == TABULATED_KIND][0]
        got = tc.component_for(comp)(
            torch.as_tensor(x), torch.tensor([[1.0, 0.0]], dtype=torch.float64))
        np.testing.assert_allclose(got.numpy()[0], want, rtol=1e-9, atol=1e-12)

    def test_intensity_is_marked_linear(self, eels):
        """Variable projection and the seeding path both key off this."""
        spec, _ = model_for_composition(eels, ["O"])
        tab, _ = tabulate_model(spec, eels)
        comp = [c for c in tab if c.kind == TABULATED_KIND][0]
        assert comp["intensity"].linear is True
        assert comp["onset_shift"].linear is False

    def test_non_edge_components_are_left_alone(self, eels):
        spec, _ = model_for_composition(eels, ["C", "N", "O"])
        tab, _ = tabulate_model(spec, eels)
        assert any(c.kind == "PowerLaw" for c in tab), \
            "the background was tabulated or lost"

    def test_reports_what_it_froze(self, eels):
        """The approximation must be stated, not buried — fine structure and
        effective angle stop being fitted."""
        spec, _ = model_for_composition(eels, ["O"])
        _, info = tabulate_model(spec, eels)
        assert "fine_structure_coeff" in info["frozen"]
        assert "effective_angle" in info["frozen"]

    def test_a_non_uniform_axis_is_rejected(self, eels):
        """Tabulation interpolates by index arithmetic, which silently gives
        wrong energies on a non-uniform axis."""
        s = eels.deepcopy()
        spec, _ = model_for_composition(s, ["O"])
        ax = s.axes_manager.signal_axes[0]
        import hyperspy.axes as hax
        s.axes_manager.remove(ax)
        s.axes_manager._axes.append(hax.DataAxis(axis=np.geomspace(200, 800, 512)))
        with pytest.raises(ValueError, match="UNIFORM"):
            tabulate_model(spec, s)


class TestFittingATabulatedModel:
    def test_recovers_the_intensity_it_was_given(self):
        """End to end on data built from the tabulated shape itself, so the
        only thing under test is the FIT."""
        x = np.linspace(200.0, 800.0, 512)
        table = np.clip(x - 400.0, 0, None) ** 0.5
        truth = np.array([2.0, 5.0, 9.0])
        data = np.stack([a * table for a in truth])

        spec = ModelSpec(components=[ComponentSpec(
            kind=TABULATED_KIND, name="edge",
            init_args={"x0": float(x[0]), "dx": float(x[1] - x[0])},
            data=table,
            parameters=[ParameterSpec("intensity", 1.0, linear=True, bmin=0.0),
                        ParameterSpec("onset_shift", 0.0, bmin=-20.0, bmax=20.0)])])
        got = fit_batched(spec, data, x, device="cpu", max_iter=80)
        col = spec.parameter_names().index("edge.intensity")
        np.testing.assert_allclose(got.values[:, col], truth, rtol=1e-4)

    def test_recovers_an_onset_shift(self):
        x = np.linspace(200.0, 800.0, 512)
        table = np.clip(x - 400.0, 0, None) ** 0.5
        shifted = np.clip(x - 412.0, 0, None) ** 0.5      # +12 eV

        spec = ModelSpec(components=[ComponentSpec(
            kind=TABULATED_KIND, name="edge",
            init_args={"x0": float(x[0]), "dx": float(x[1] - x[0])},
            data=table,
            parameters=[ParameterSpec("intensity", 1.0, linear=True, bmin=0.0),
                        ParameterSpec("onset_shift", 0.0, bmin=-30.0, bmax=30.0)])])
        got = fit_batched(spec, shifted[None, :], x, device="cpu", max_iter=150)
        col = spec.parameter_names().index("edge.onset_shift")
        assert got.values[0, col] == pytest.approx(12.0, abs=1.5)

    def test_fits_a_real_composition_model_on_real_synthetic_data(self, eels):
        """The whole path: composition -> model -> tabulate -> batched fit."""
        spec, _ = model_for_composition(eels, ["C", "N", "O"])
        tab, _ = tabulate_model(spec, eels)
        x = np.asarray(eels.axes_manager.signal_axes[0].axis, float)
        got = fit_batched(tab, eels.data, x, device="cpu", max_iter=60)
        assert got.values.shape[0] == 9
        assert np.isfinite(got.values).all()

    def test_edge_intensity_tracks_the_known_concentration(self, eels):
        """The measurement that matters: more of an element must fit a bigger
        edge intensity, scored against the map the data was built from."""
        from spyde.data import ground_truth

        spec, _ = model_for_composition(eels, ["C", "N", "O"])
        tab, _ = tabulate_model(spec, eels)
        x = np.asarray(eels.axes_manager.signal_axes[0].axis, float)
        got = fit_batched(tab, eels.data, x, device="cpu", max_iter=80)

        names = tab.parameter_names()
        conc = np.asarray(ground_truth(eels)["concentration"]["O_K"]).ravel()
        fitted = got.values[:, names.index("O_K.intensity")]
        assert np.corrcoef(fitted, conc)[0, 1] > 0.7, \
            f"fitted O_K intensity does not track its concentration map"


class TestOnsetEnergies:
    def test_reports_an_absolute_energy(self, eels):
        """onset_shift alone is relative to where the table was sampled, which
        is not a number a user can read."""
        spec, _ = model_for_composition(eels, ["O"])
        tab, _ = tabulate_model(spec, eels)
        comp = [c for c in tab if c.kind == TABULATED_KIND][0]
        comp["onset_shift"].value = 5.0
        got = onset_energies(tab)
        assert got["O_K"] == pytest.approx(
            comp.init_args["onset_energy"] + 5.0)

    def test_ignores_untabulated_components(self, eels):
        spec, _ = model_for_composition(eels, ["O"])
        tab, _ = tabulate_model(spec, eels)
        assert set(onset_energies(tab)) == {"O_K"}
