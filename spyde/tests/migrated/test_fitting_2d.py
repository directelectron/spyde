"""2-D model support (#59) — Gaussian2D through the same batched engine.

A 2-D model is not a separate code path: the image is flattened to H*W sample
points exactly as a spectrum is C channels, and the only difference is that the
"axis" carries (x, y) PAIRS. Everything downstream — packing, the Jacobian, the
LM solve, bounds — is the 1-D machinery unchanged, and these tests exist to
prove that rather than assume it.

This is also what Wave 4 needs: refining atom positions IS a batched 2-D
gaussian fit (#77).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import hyperspy.components2d as c2d

from spyde.fitting import ModelSpec
from spyde.fitting import components as tc
from spyde.fitting.engine import fit_batched
from spyde.fitting.spec import ComponentSpec, ParameterSpec

SHAPE = (24, 28)          # deliberately non-square: catches a transposed grid


def _truth(A=500.0, cx=13.0, cy=10.0, sx=2.5, sy=3.5):
    return dict(A=A, centre_x=cx, centre_y=cy, sigma_x=sx, sigma_y=sy)


def _hyperspy_image(**vals):
    g = c2d.Gaussian2D()
    for k, v in vals.items():
        getattr(g, k).value = v
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]].astype(float)
    return np.asarray(g.function(xx, yy), float)


def _spec(**vals):
    order = ("A", "centre_x", "centre_y", "sigma_x", "sigma_y")
    return ModelSpec(components=[ComponentSpec(
        kind="Gaussian2D",
        parameters=[ParameterSpec(n, float(vals[n])) for n in order])])


class TestGaussian2DParity:
    def test_parameter_order_matches_hyperspy(self):
        hs = tuple(p.name for p in c2d.Gaussian2D().parameters)
        assert tc.get_component("Gaussian2D").params == hs

    def test_linear_flag_matches_hyperspy(self):
        hs = tuple(bool(getattr(p, "_linear", False))
                   for p in c2d.Gaussian2D().parameters)
        assert tc.get_component("Gaussian2D").linear == hs

    def test_values_match_hyperspy(self):
        """A is the VOLUME, not the peak height — the same convention trap as
        the 1-D Gaussian."""
        vals = _truth()
        want = _hyperspy_image(**vals).ravel()
        xy = tc.image_coordinates(SHAPE, dtype=torch.float64)
        comp = tc.get_component("Gaussian2D")
        got = comp(xy, torch.as_tensor(
            [[vals[n] for n in comp.params]], dtype=torch.float64))
        np.testing.assert_allclose(got.numpy()[0], want, rtol=1e-10, atol=1e-12)

    def test_analytic_gradient_matches_autodiff(self):
        from torch.func import jacfwd
        vals = _truth()
        comp = tc.get_component("Gaussian2D")
        xy = tc.image_coordinates(SHAPE, dtype=torch.float64)
        v = torch.as_tensor([[vals[n] for n in comp.params]],
                            dtype=torch.float64)
        auto = jacfwd(lambda p: comp(xy, p.unsqueeze(0)).squeeze(0))(v[0])
        torch.testing.assert_close(comp.grad(xy, v)[0], auto,
                                   rtol=1e-9, atol=1e-9)


class TestCoordinates:
    def test_shape_and_ordering(self):
        xy = tc.image_coordinates(SHAPE)
        assert xy.shape == (SHAPE[0] * SHAPE[1], 2)
        # Row-major: x runs fastest, matching an image flattened with .ravel().
        assert xy[0].tolist() == [0, 0]
        assert xy[1].tolist() == [1, 0]
        assert xy[SHAPE[1]].tolist() == [0, 1]

    def test_matches_a_ravelled_mgrid(self):
        yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
        xy = tc.image_coordinates(SHAPE).numpy()
        np.testing.assert_array_equal(xy[:, 0], xx.ravel())
        np.testing.assert_array_equal(xy[:, 1], yy.ravel())


class TestBatching:
    def test_each_row_gets_its_own_surface(self):
        comp = tc.get_component("Gaussian2D")
        xy = tc.image_coordinates(SHAPE, dtype=torch.float64)
        vals = torch.tensor([[500.0, 5.0, 5.0, 2.0, 2.0],
                             [500.0, 20.0, 18.0, 2.0, 2.0]],
                            dtype=torch.float64)
        y = comp(xy, vals).reshape(2, *SHAPE)
        peak0 = np.unravel_index(int(y[0].argmax()), SHAPE)
        peak1 = np.unravel_index(int(y[1].argmax()), SHAPE)
        assert peak0 == (5, 5)          # (row=y, col=x)
        assert peak1 == (18, 20)

    def test_non_square_images_are_not_transposed(self):
        """SHAPE is deliberately non-square — a transposed coordinate grid
        would still run and put every peak in the wrong place."""
        comp = tc.get_component("Gaussian2D")
        xy = tc.image_coordinates(SHAPE, dtype=torch.float64)
        y = comp(xy, torch.tensor([[500.0, 22.0, 3.0, 1.5, 1.5]],
                                  dtype=torch.float64)).reshape(SHAPE)
        assert np.unravel_index(int(y.argmax()), SHAPE) == (3, 22)


class TestFittingAnImage:
    def test_recovers_the_generating_parameters(self):
        """End to end: a 2-D model fits through the SAME engine as a spectrum."""
        vals = _truth()
        img = _hyperspy_image(**vals).ravel()[None, :]
        spec = _spec(A=400.0, centre_x=12.0, centre_y=11.0,
                     sigma_x=2.0, sigma_y=3.0)
        xy = tc.image_coordinates(SHAPE, dtype=torch.float64).numpy()

        got = fit_batched(spec, img, xy, device="cpu", max_iter=200)
        names = spec.parameter_names()
        for key, want in vals.items():
            col = names.index(f"Gaussian2D.{key}")
            assert got.values[0, col] == pytest.approx(want, rel=1e-4), key
        assert got.converged.all()

    def test_fits_many_images_at_once(self):
        """The point of the engine: N independent 2-D fits in one call."""
        centres = [(8.0, 7.0), (14.0, 12.0), (20.0, 16.0)]
        imgs = np.stack([_hyperspy_image(**_truth(cx=cx, cy=cy)).ravel()
                         for cx, cy in centres])
        spec = _spec(A=400.0, centre_x=13.0, centre_y=11.0,
                     sigma_x=2.0, sigma_y=3.0)
        xy = tc.image_coordinates(SHAPE, dtype=torch.float64).numpy()

        got = fit_batched(spec, imgs, xy, device="cpu", max_iter=200)
        names = spec.parameter_names()
        cx = got.values[:, names.index("Gaussian2D.centre_x")]
        cy = got.values[:, names.index("Gaussian2D.centre_y")]
        np.testing.assert_allclose(cx, [c[0] for c in centres], atol=1e-3)
        np.testing.assert_allclose(cy, [c[1] for c in centres], atol=1e-3)

    def test_bounds_apply_in_2d_too(self):
        """Bounds are engine-level, not per-component — but a 2-D model going
        through a different shaping path is exactly where that could break."""
        img = _hyperspy_image(**_truth()).ravel()[None, :]
        spec = _spec(A=400.0, centre_x=12.0, centre_y=11.0,
                     sigma_x=2.0, sigma_y=3.0)
        spec[0]["centre_x"].bmin = 5.0
        spec[0]["centre_x"].bmax = 9.0
        xy = tc.image_coordinates(SHAPE, dtype=torch.float64).numpy()
        got = fit_batched(spec, img, xy, device="cpu", max_iter=120)
        col = spec.parameter_names().index("Gaussian2D.centre_x")
        assert 5.0 - 1e-9 <= got.values[0, col] <= 9.0 + 1e-9

    def test_a_fixed_parameter_stays_fixed_in_2d(self):
        img = _hyperspy_image(**_truth()).ravel()[None, :]
        spec = _spec(A=400.0, centre_x=12.0, centre_y=11.0,
                     sigma_x=2.0, sigma_y=3.5)
        spec[0]["sigma_y"].free = False
        xy = tc.image_coordinates(SHAPE, dtype=torch.float64).numpy()
        got = fit_batched(spec, img, xy, device="cpu", max_iter=120)
        col = spec.parameter_names().index("Gaussian2D.sigma_y")
        assert got.values[0, col] == pytest.approx(3.5, rel=1e-12)


class TestRoundTrip:
    def test_gaussian2d_round_trips_through_modelspec(self):
        """A 2-D model must save and reopen like any other."""
        import hyperspy.api as hs

        sig = hs.signals.Signal2D(np.zeros(SHAPE))
        m = sig.create_model()
        m.append(c2d.Gaussian2D())
        for k, v in _truth().items():
            getattr(m[0], k).value = v
        spec = ModelSpec.from_model(m)
        assert spec[0].kind == "Gaussian2D"
        back = ModelSpec.from_model(spec.to_model(sig))
        np.testing.assert_allclose(back.flat_values(), spec.flat_values())

    def test_the_engine_reports_2d_support(self):
        assert tc.supports(_spec(**_truth())) is True
        assert tc.has_analytic_grad(_spec(**_truth())) is True
