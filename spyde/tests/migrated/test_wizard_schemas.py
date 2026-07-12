"""
test_wizard_schemas.py — every wizard declares a valid parameter schema.

The three-host parity contract (NOTEBOOK_PARITY_PLAN §6) requires ONE source
of truth for each wizard's parameters, resolvable host-agnostically via
``registry.wizard_parameters(key)`` and expressed in the toolbars.yaml
``parameters:`` dict spec. This suite enforces:

* every registered wizard key resolves to a non-empty schema,
* every entry is well-formed (type/name/default; bounds contain the default;
  enum choices contain the default; file entries declare extensions),
* schema defaults stay in lock-step with the backend handler DEFAULTS they
  mirror (the drift this contract exists to prevent).
"""
from __future__ import annotations

import pytest

from spyde.actions import registry

VALID_TYPES = {"int", "float", "bool", "enum", "file"}


class TestSchemaCompleteness:
    def test_every_wizard_key_has_a_schema(self):
        assert set(registry.wizard_keys()) >= {"fv", "om", "strain", "vom", "czb"}
        for key in registry.wizard_keys():
            schema = registry.wizard_parameters(key)
            assert isinstance(schema, dict) and schema, \
                f"wizard {key!r} has no declared parameter schema"

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            registry.wizard_parameters("nope")

    def test_schema_is_a_copy(self):
        a = registry.wizard_parameters("strain")
        a["component"] = "mutated"
        assert registry.wizard_parameters("strain")["component"] != "mutated"


class TestSchemaValidity:
    @pytest.mark.parametrize("key", ["fv", "om", "strain", "vom", "czb", "mvx"])
    def test_entries_well_formed(self, key):
        schema = registry.wizard_parameters(key)
        for pname, spec in schema.items():
            assert isinstance(spec, dict), f"{key}.{pname} is not a dict"
            ptype = spec.get("type")
            assert ptype in VALID_TYPES, \
                f"{key}.{pname}: type {ptype!r} not in {VALID_TYPES}"
            assert spec.get("name"), f"{key}.{pname}: missing display name"
            assert "default" in spec, f"{key}.{pname}: missing default"
            d = spec["default"]
            if ptype in ("int", "float"):
                assert isinstance(d, (int, float)) and not isinstance(d, bool), \
                    f"{key}.{pname}: numeric type with non-numeric default {d!r}"
                if "min" in spec:
                    assert spec["min"] <= d, f"{key}.{pname}: default < min"
                if "max" in spec:
                    assert d <= spec["max"], f"{key}.{pname}: default > max"
            elif ptype == "bool":
                assert isinstance(d, bool), \
                    f"{key}.{pname}: bool type with default {d!r}"
            elif ptype == "enum":
                choices = spec.get("choices") or spec.get("options")
                assert choices, f"{key}.{pname}: enum without choices"
                assert d in choices, \
                    f"{key}.{pname}: default {d!r} not in choices {choices}"
            elif ptype == "file":
                assert spec.get("extensions"), \
                    f"{key}.{pname}: file entry without extensions"


class TestSchemaBackendLockstep:
    """Schema defaults must match the handler DEFAULTS they describe."""

    def test_czb_defaults(self):
        from spyde.actions import center_zero_beam as czb
        schema = registry.wizard_parameters("czb")
        for k in ("method", "half_square_width", "make_flat_field"):
            assert schema[k]["default"] == czb.DEFAULTS[k], \
                f"czb schema/{k} drifted from center_zero_beam.DEFAULTS"

    def test_mvx_defaults(self):
        from spyde.actions.movie_export.handlers import DEFAULTS
        schema = registry.wizard_parameters("mvx")
        for k in ("fps", "downsample", "stride", "cmap", "timestamp", "scalebar"):
            assert schema[k]["default"] == DEFAULTS[k], \
                f"mvx schema/{k} drifted from movie_export.handlers.DEFAULTS"

    def test_vom_defaults(self):
        from spyde.actions import vector_orientation_om as vom
        schema = registry.wizard_parameters("vom")
        for k in ("accelerating_voltage", "resolution", "minimum_intensity",
                  "strain_cap", "smooth"):
            assert schema[k]["default"] == vom.DEFAULTS[k], \
                f"vom schema/{k} drifted from vector_orientation_om.DEFAULTS"

    def test_vom_fit_defaults(self):
        from spyde.actions.vector_orientation import DEFAULTS as FIT
        schema = registry.wizard_parameters("vom")
        assert schema["strain_cap"]["default"] == FIT["strain_cap"]
        assert schema["sink_bw"]["default"] == FIT["sink_bw"]

    def test_strain_components(self):
        from spyde.actions._common import STRAIN_COMPONENTS
        schema = registry.wizard_parameters("strain")
        assert tuple(schema["component"]["choices"]) == STRAIN_COMPONENTS

    def test_fv_defaults_match_action(self):
        from spyde.actions.find_vectors_action import DEFAULTS
        schema = registry.wizard_parameters("fv")
        for k in ("sigma", "kernel_radius", "threshold", "min_distance",
                  "method", "dog_sigma1", "dog_sigma2"):
            assert schema[k]["default"] == DEFAULTS[k], \
                f"fv yaml schema/{k} drifted from find_vectors_action.DEFAULTS"

    def test_om_defaults_match_action(self):
        from spyde.actions.orientation_action import DEFAULTS
        schema = registry.wizard_parameters("om")
        for k in ("accelerating_voltage", "resolution", "n_best",
                  "minimum_intensity"):
            assert schema[k]["default"] == DEFAULTS[k], \
                f"om yaml schema/{k} drifted from orientation_action.DEFAULTS"

    def test_controllers_declare_parameters(self):
        from spyde.actions.strain_action import StrainController
        from spyde.actions.vector_orientation_om import VomWizard
        for cls in (StrainController, VomWizard):
            assert cls.parameters, \
                f"{cls.__name__} must declare its parameter schema"
