"""ModelSpec round-trips against HyperSpy and packs parameters in a stable order.

Two contracts:

1. **HyperSpy is the storage format.** A spec taken from a live model and put
   back must reproduce the same component kinds, values, bounds, free flags and
   channel range — otherwise a model saved by SpyDE would not reopen in plain
   HyperSpy, and the parity tests in #53 would be comparing different models.
2. **Packed parameter order is part of the API.** Column j of every array (and
   of the engine's tensors) always means the same parameter, and
   `parameter_names()` is that order. Anything reporting per-parameter results
   relies on it.
"""
from __future__ import annotations

import numpy as np
import pytest
import hyperspy.api as hs
from hyperspy.components1d import Gaussian, Offset, PowerLaw

from spyde.fitting import ComponentSpec, ModelSpec, ParameterSpec


def _signal(n=64, ny=3, nx=3):
    return hs.signals.Signal1D(np.random.default_rng(0).random((ny, nx, n)) + 1.0)


def _model():
    m = _signal().create_model()
    m.extend([PowerLaw(), Gaussian(centre=30.0, sigma=4.0, A=10.0)])
    return m


class TestFromModel:
    def test_reads_components_and_kinds(self):
        spec = ModelSpec.from_model(_model())
        assert [c.kind for c in spec] == ["PowerLaw", "Gaussian"]

    def test_reads_parameter_values(self):
        spec = ModelSpec.from_model(_model())
        g = spec["Gaussian"]
        assert g["centre"].value == pytest.approx(30.0)
        assert g["sigma"].value == pytest.approx(4.0)
        assert g["A"].value == pytest.approx(10.0)

    def test_reads_bounds_including_absent_ones(self):
        spec = ModelSpec.from_model(_model())
        g = spec["Gaussian"]
        assert g["A"].bmin == pytest.approx(0.0)     # HyperSpy bounds A >= 0
        assert g["centre"].bmin is None              # unbounded
        assert g["centre"].bounds() == (-np.inf, np.inf)

    def test_reads_the_linear_flag(self):
        """The variable-projection path (#53) keys off this, so it has to come
        from HyperSpy rather than a hand-maintained list of our own."""
        spec = ModelSpec.from_model(_model())
        assert spec["Gaussian"]["A"].linear is True
        assert spec["Gaussian"]["centre"].linear is False

    def test_reads_free_flags(self):
        m = _model()
        m[1].sigma.free = False
        spec = ModelSpec.from_model(m)
        assert spec["Gaussian"]["sigma"].free is False
        assert spec["Gaussian"]["A"].free is True

    def test_all_channels_active_is_stored_as_none(self):
        """A full mask is the default; carrying an all-True array of signal
        length through JSON on every model would be pure waste."""
        assert ModelSpec.from_model(_model()).channel_mask is None

    def test_reads_a_restricted_signal_range(self):
        m = _model()
        m.set_signal_range(10, 40)
        spec = ModelSpec.from_model(m)
        assert spec.channel_mask is not None
        assert spec.channel_mask.sum() < spec.channel_mask.size
        assert spec.channel_mask[20] and not spec.channel_mask[5]


class TestToModel:
    def test_round_trip_preserves_kinds_and_values(self):
        spec = ModelSpec.from_model(_model())
        back = ModelSpec.from_model(spec.to_model(_signal()))
        assert [c.kind for c in back] == [c.kind for c in spec]
        assert back.flat_values() == pytest.approx(spec.flat_values())

    def test_round_trip_preserves_bounds_and_free_flags(self):
        m = _model()
        m[1].sigma.free = False
        m[1].centre.bmin, m[1].centre.bmax = 20.0, 40.0
        spec = ModelSpec.from_model(m)
        back = ModelSpec.from_model(spec.to_model(_signal()))
        assert back["Gaussian"]["sigma"].free is False
        assert back["Gaussian"]["centre"].bmin == pytest.approx(20.0)
        assert back["Gaussian"]["centre"].bmax == pytest.approx(40.0)

    def test_round_trip_preserves_the_signal_range(self):
        m = _model()
        m.set_signal_range(10, 40)
        spec = ModelSpec.from_model(m)
        back = ModelSpec.from_model(spec.to_model(_signal()))
        assert np.array_equal(back.channel_mask, spec.channel_mask)

    def test_bounds_are_set_before_values(self):
        """HyperSpy clips an assignment that falls outside the bounds, so a
        to_model() that wrote the value first would silently corrupt it."""
        spec = ModelSpec(components=[ComponentSpec(
            kind="Gaussian", parameters=[
                ParameterSpec("A", value=500.0, bmin=0.0, bmax=1000.0),
                ParameterSpec("centre", value=30.0),
                ParameterSpec("sigma", value=4.0, bmin=0.0, bmax=10.0),
            ])])
        m = spec.to_model(_signal())
        assert float(np.ravel(m[0].A.value)[0]) == pytest.approx(500.0)

    def test_inactive_component_survives_the_round_trip(self):
        m = _model()
        m[1].active = False
        back = ModelSpec.from_model(ModelSpec.from_model(m).to_model(_signal()))
        assert back["Gaussian"].active is False

    def test_replaces_any_prepopulated_components(self):
        """create_model() can pre-populate (an EELS model adds a background and
        the declared edges). The spec is authoritative, so those must go."""
        spec = ModelSpec(components=[ComponentSpec(
            kind="Offset", parameters=[ParameterSpec("offset", value=1.0)])])
        m = spec.to_model(_signal())
        assert [getattr(c, "_id_name", "") for c in m] == ["Offset"]

    def test_unknown_kind_is_a_clear_error(self):
        spec = ModelSpec(components=[ComponentSpec(kind="Flurbulator")])
        with pytest.raises(ValueError, match="Flurbulator"):
            spec.to_model(_signal())


class TestPackedOrder:
    def test_parameter_names_follow_component_then_parameter_order(self):
        spec = ModelSpec.from_model(_model())
        names = spec.parameter_names()
        assert names[0].startswith("PowerLaw.")
        assert names[-1].startswith("Gaussian.")
        assert len(names) == len(spec.flat_values())

    def test_masks_line_up_with_values(self):
        spec = ModelSpec.from_model(_model())
        n = len(spec.flat_values())
        assert len(spec.free_mask()) == n
        assert len(spec.linear_mask()) == n
        lo, hi = spec.bounds_arrays()
        assert len(lo) == len(hi) == n

    def test_set_flat_values_is_the_inverse_of_flat_values(self):
        spec = ModelSpec.from_model(_model())
        new = spec.flat_values() + 1.5
        spec.set_flat_values(new)
        assert spec.flat_values() == pytest.approx(new)

    def test_set_flat_values_rejects_the_wrong_width(self):
        spec = ModelSpec.from_model(_model())
        with pytest.raises(ValueError, match="expected"):
            spec.set_flat_values(np.zeros(3))

    def test_inactive_components_occupy_no_columns(self):
        """An inactive component contributes nothing to the model, so it must
        not take a column — otherwise the engine fits a parameter of something
        that is not being evaluated."""
        spec = ModelSpec.from_model(_model())
        wide = len(spec.flat_values())
        spec["Gaussian"].active = False
        narrow = len(spec.flat_values())
        assert narrow == wide - 3
        assert all(n.startswith("PowerLaw.") for n in spec.parameter_names())

    def test_component_slices_isolate_each_component(self):
        spec = ModelSpec.from_model(_model())
        sl = spec.component_slices()
        vals = spec.flat_values()
        assert set(sl) == {"PowerLaw", "Gaussian"}
        assert len(vals[sl["Gaussian"]]) == 3
        # Slices must tile the vector exactly, with no gap or overlap.
        covered = sorted(i for s in sl.values() for i in range(*s.indices(len(vals))))
        assert covered == list(range(len(vals)))


class TestJson:
    def test_dict_round_trip(self):
        spec = ModelSpec.from_model(_model())
        back = ModelSpec.from_dict(spec.to_dict())
        assert back.parameter_names() == spec.parameter_names()
        assert back.flat_values() == pytest.approx(spec.flat_values())

    def test_dict_is_json_serialisable(self):
        import json
        m = _model()
        m.set_signal_range(10, 40)
        json.dumps(ModelSpec.from_model(m).to_dict())      # must not raise

    def test_channel_mask_survives_as_ranges(self):
        m = _model()
        m.set_signal_range(10, 40)
        spec = ModelSpec.from_model(m)
        back = ModelSpec.from_dict(spec.to_dict())
        assert np.array_equal(back.channel_mask, spec.channel_mask)

    def test_two_disjoint_ranges_survive(self):
        """remove_signal_range can leave a hole in the middle; the compact
        range encoding has to keep both runs."""
        mask = np.zeros(64, bool)
        mask[5:15] = True
        mask[40:50] = True
        spec = ModelSpec(components=[ComponentSpec(kind="Offset")],
                         channel_mask=mask)
        assert len(spec.to_dict()["channel_ranges"]) == 2
        assert np.array_equal(ModelSpec.from_dict(spec.to_dict()).channel_mask,
                              mask)


class TestCopy:
    def test_copy_is_deep(self):
        spec = ModelSpec.from_model(_model())
        clone = spec.copy()
        clone["Gaussian"]["A"].value = 999.0
        assert spec["Gaussian"]["A"].value != 999.0

    def test_lookup_errors_name_what_is_available(self):
        spec = ModelSpec.from_model(_model())
        with pytest.raises(KeyError, match="Gaussian"):
            spec["NoSuchComponent"]
        with pytest.raises(KeyError, match="centre"):
            spec["Gaussian"]["no_such_parameter"]
