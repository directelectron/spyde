"""The per-position parameter store — HyperSpy's own, not a parallel one.

A HyperSpy parameter already carries ``parameter.map``: a structured array over
the navigation grid with ``values`` / ``std`` / ``is_set``. That IS "hold the
parameters for the model at each position", it is what ``m.multifit()`` fills,
and it is what ``m.store()`` persists. These tests pin that SpyDE writes into
those arrays rather than keeping a dict beside them.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import hyperspy.api as hs

from spyde.actions.fit_action import new_component_spec
from spyde.fitting import ModelSpec
from spyde.fitting.store import FitStore


def _signal(ny=4, nx=5, nc=128):
    """Deliberately NON-square: a transposed store looks identical on a
    square scan, and only shows up as every position holding its
    neighbour's answer."""
    s = hs.signals.Signal1D(np.random.default_rng(0).random((ny, nx, nc)) * 100)
    ax = s.axes_manager.signal_axes[0]
    ax.offset, ax.scale = 0.0, 50.0 / (nc - 1)
    return s


def _spec(n=2):
    spec = ModelSpec()
    for _ in range(n):
        c = new_component_spec("Gaussian")
        while any(e.name == c.name for e in spec.components):
            c.name += "'"
        spec.append(c)
    return spec


@pytest.fixture
def store():
    return FitStore(_spec(), _signal())


class TestItIsHyperspysStore:
    def test_the_values_land_in_the_parameter_map(self, store):
        store.put((3, 1), np.arange(6.0))
        par = store._params[0]
        assert par.map["values"][1, 3] == 0.0
        assert bool(par.map["is_set"][1, 3])
        assert store._params[1].map["values"][1, 3] == 1.0

    def test_it_is_a_real_hyperspy_model(self, store):
        assert len(store.model) == 2
        assert all(hasattr(c, "parameters") for c in store.model)

    def test_the_parameter_order_is_the_engines_column_order(self, store):
        names = store.spec.parameter_names()
        assert len(store._params) == len(names)
        for full, par in zip(names, store._params):
            assert full.endswith("." + par.name)


class TestIndexing:
    def test_navigator_indices_are_x_first_and_maps_are_y_first(self, store):
        """The hazard: on a square scan a transposed store is invisible."""
        assert store.nav_shape == (4, 5)
        store.put((3, 1), np.arange(6.0))
        assert store.get((3, 1)) is not None
        assert store.get((1, 3)) is None

    def test_an_out_of_range_position_is_refused(self, store):
        assert store.put((99, 0), np.arange(6.0)) is False
        assert store.get((99, 0)) is None

    def test_a_wrong_width_vector_is_refused(self, store):
        assert store.put((0, 0), [1.0, 2.0]) is False
        assert store.get((0, 0)) is None


class TestIsSet:
    def test_an_unfitted_position_reads_nothing(self, store):
        assert store.get((2, 2)) is None
        assert store.is_set((2, 2)) is False

    def test_fitted_to_zero_is_not_the_same_as_unfitted(self, store):
        """The thing a plain dict of values could not express, and the reason
        the maps can show holes rather than a field of zeros."""
        store.put((2, 2), np.zeros(6))
        assert store.is_set((2, 2)) is True
        assert np.array_equal(store.get((2, 2)), np.zeros(6))

    def test_coverage_counts_positions(self, store):
        assert store.coverage() == (0, 20)
        store.put((0, 0), np.arange(6.0))
        store.put((1, 0), np.arange(6.0))
        assert store.coverage() == (2, 20)

    def test_clear_forgets_everything(self, store):
        store.put((0, 0), np.arange(6.0))
        store.clear()
        assert store.coverage() == (0, 20)
        assert store.get((0, 0)) is None


class TestWholeScan:
    def test_put_all_fills_every_position(self, store):
        values = np.arange(20 * 6, dtype=float).reshape(20, 6)
        assert store.put_all(values, chisq=np.arange(20.0)) == 20
        assert store.coverage() == (20, 20)

    def test_put_all_lands_in_the_right_places(self, store):
        """Flat row i must end up at the position `get` reads back for i —
        getting this reversed transposes the entire scan."""
        values = np.arange(20 * 6, dtype=float).reshape(20, 6)
        store.put_all(values)
        for iy in range(4):
            for ix in range(5):
                flat = iy * 5 + ix
                assert np.array_equal(store.get((ix, iy)), values[flat])

    def test_values_array_round_trips(self, store):
        values = np.arange(20 * 6, dtype=float).reshape(20, 6)
        store.put_all(values)
        assert np.array_equal(store.values_array(), values)

    def test_a_wrong_width_scan_is_refused(self, store):
        assert store.put_all(np.zeros((20, 3))) == 0
        assert store.coverage()[0] == 0


class TestMaps:
    def test_every_map_starts_empty(self, store):
        x = np.linspace(0.0, 50.0, 128)
        maps = store.maps(x)
        assert set(maps) == {"Gaussian", "Gaussian'", "chi squared"}
        for m in maps.values():
            assert m.shape == (4, 5)
            assert np.isnan(m).all()

    def test_a_fitted_position_becomes_finite(self, store):
        x = np.linspace(0.0, 50.0, 128)
        store.put((3, 1), [1000.0, 25.0, 4.0, 500.0, 30.0, 6.0], chisq=9.0)
        maps = store.maps(x)
        assert np.isfinite(maps["Gaussian"]).sum() == 1
        assert np.isfinite(maps["Gaussian"][1, 3])
        assert maps["chi squared"][1, 3] == 9.0
        # ...and the rest is still a hole, not a zero. A zero would read as
        # "fitted, and the component is absent here".
        assert np.isnan(maps["Gaussian"][0, 0])

    def test_the_area_map_tracks_the_amplitude(self, store):
        x = np.linspace(0.0, 50.0, 128)
        store.put((0, 0), [1000.0, 25.0, 4.0, 0.0, 30.0, 6.0])
        store.put((1, 0), [2000.0, 25.0, 4.0, 0.0, 30.0, 6.0])
        maps = store.maps(x)
        assert maps["Gaussian"][0, 1] == pytest.approx(
            2 * maps["Gaussian"][0, 0], rel=1e-6)
