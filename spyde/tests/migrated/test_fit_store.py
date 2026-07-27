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
        assert par.map["values"][3, 1] == 0.0
        assert bool(par.map["is_set"][3, 1])
        assert store._params[1].map["values"][3, 1] == 1.0

    def test_it_is_a_real_hyperspy_model(self, store):
        assert len(store.model) == 2
        assert all(hasattr(c, "parameters") for c in store.model)

    def test_the_parameter_order_is_the_engines_column_order(self, store):
        names = store.spec.parameter_names()
        assert len(store._params) == len(names)
        for full, par in zip(names, store._params):
            assert full.endswith("." + par.name)


class TestIndexing:
    """The store must index the way the DISPLAY does, and nothing else.

    `_build_nav_lazy_slice` / `get_local_frame` do `data[point]` with the
    selector's indices exactly as given, so the spectrum on screen at
    crosshair (cx, cy) is `data[cx, cy]`. The store answers "what was fitted
    to THAT spectrum", so it has to agree — reasoning instead from
    `axes_manager.navigation_shape` transposed the whole scan.
    """

    def test_the_key_matches_how_the_display_reads_the_data(self, store):
        store.put((3, 1), np.arange(6.0))
        par = store._params[0]
        # data[3, 1] is what the display shows at this crosshair, so the map
        # entry has to be [3, 1] too.
        assert bool(par.map["is_set"][3, 1]), "the store is transposed"
        assert not bool(par.map["is_set"][1, 3])

    def test_a_transposed_position_is_a_different_position(self, store):
        """On a SQUARE scan this is the ONLY thing that catches a transpose —
        the shapes match, coverage looks complete, and every recall succeeds
        while quietly returning another pixel's answer."""
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

    def test_put_all_lands_where_the_display_looks(self, store):
        """A whole-scan fit flattens `data` in C order, so row r is
        `data[r // nx, r % nx]`. The display shows `data[point]` at crosshair
        `point`, so `get(point)` must return the row for `data[point]` — a
        self-consistent-but-transposed convention passes a test that only
        checks the round trip, and shows every position its neighbour's fit."""
        ny, nx = 4, 5
        values = np.arange(ny * nx * 6, dtype=float).reshape(ny * nx, 6)
        store.put_all(values)
        for a in range(ny):
            for b in range(nx):
                flat = a * nx + b        # the row fitted to data[a, b]
                assert np.array_equal(store.get((a, b)), values[flat])

    def test_values_array_round_trips(self, store):
        values = np.arange(20 * 6, dtype=float).reshape(20, 6)
        store.put_all(values)
        assert np.array_equal(store.values_array(), values)

    def test_a_wrong_width_scan_is_refused(self, store):
        assert store.put_all(np.zeros((20, 3))) == 0
        assert store.coverage()[0] == 0


class TestSaveAndLoad:
    """Straight through HyperSpy's own model store, so a fit travels with the
    dataset: `m.store(name)` puts the components AND every position's map into
    the signal's `models`, and saving the file saves them with it."""

    def test_a_stored_model_survives_a_save_and_reload(self, tmp_path):
        from spyde.fitting import ModelSpec
        sig = _signal()
        store = FitStore(_spec(), sig)
        values = np.tile(np.array([100.0, 25.0, 4.0, 50.0, 30.0, 6.0]), (20, 1))
        store.put_all(values, chisq=np.arange(20.0))
        store.save_as("spyde_fit")

        path = tmp_path / "with_a_model.hspy"
        sig.save(str(path))
        reloaded = hs.load(str(path))

        spec, restored = FitStore.restore(ModelSpec, reloaded, "spyde_fit")
        assert [c.kind for c in spec.components] == ["Gaussian", "Gaussian"]
        assert restored.coverage() == (20, 20)
        assert np.array_equal(restored.values_array(), values)

    def test_the_names_are_listed(self):
        store = FitStore(_spec(), _signal())
        assert store.stored_names() == []
        store.save_as("first")
        store.save_as("second")
        assert set(store.stored_names()) == {"first", "second"}

    def test_an_unfitted_position_stays_unfitted_across_the_round_trip(self, tmp_path):
        """`is_set` is the thing a values-only format would lose — and losing
        it turns every hole in a component map into a confident zero."""
        from spyde.fitting import ModelSpec
        sig = _signal()
        store = FitStore(_spec(), sig)
        store.put((1, 2), np.arange(6.0))
        store.save_as("partial")
        path = tmp_path / "partial.hspy"
        sig.save(str(path))

        _spec_out, restored = FitStore.restore(
            ModelSpec, hs.load(str(path)), "partial")
        assert restored.coverage() == (1, 20)
        assert restored.get((1, 2)) is not None
        assert restored.get((0, 0)) is None


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
        assert np.isfinite(maps["Gaussian"][3, 1])
        assert maps["chi squared"][3, 1] == 9.0
        # ...and the rest is still a hole, not a zero. A zero would read as
        # "fitted, and the component is absent here".
        assert np.isnan(maps["Gaussian"][0, 0])

    def test_the_area_map_tracks_the_amplitude(self, store):
        x = np.linspace(0.0, 50.0, 128)
        store.put((0, 0), [1000.0, 25.0, 4.0, 0.0, 30.0, 6.0])
        store.put((0, 1), [2000.0, 25.0, 4.0, 0.0, 30.0, 6.0])
        maps = store.maps(x)
        assert maps["Gaussian"][0, 1] == pytest.approx(
            2 * maps["Gaussian"][0, 0], rel=1e-6)
