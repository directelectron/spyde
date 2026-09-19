"""The ``beta:`` toolbar key.

An action marked ``beta:`` works and is offered exactly as any other — the key
is not a gate. It only travels to the renderer, which draws a badge on the
button and a ribbon across the caret, so what a user is promised about an
action matches what we are willing to keep stable.

The flag is declared once, in the toolbar schema. The renderer reads it off the
message rather than keeping its own copy, so there is no second value to drift;
what these tests protect is the delivery — a key that stops being forwarded
would silently un-mark every beta action.
"""
from __future__ import annotations

import types

from spyde import TOOLBAR_ACTIONS
from spyde.drawing.toolbars.plot_control_toolbar import (
    _action_matches_plot,
    get_toolbar_config_for_plot,
)


class _FakeVectorPlot:
    """A plot showing a diffraction-vectors image with vectors attached — what
    the vector actions gate on."""

    def __init__(self):
        signal = types.SimpleNamespace(
            _signal_type="spyde_diffraction_vectors_image")
        self.plot_state = types.SimpleNamespace(
            current_signal=signal, dimensions=2, plot=self)
        self.signal_tree = types.SimpleNamespace(
            diffraction_vectors=object(), root=None)
        self.is_navigator = False


def _descriptor(actions, name):
    return next((a for a in actions if a["name"] == name), None)


class TestBetaTravelsToTheRenderer:
    def test_a_beta_action_is_marked(self):
        actions = get_toolbar_config_for_plot(_FakeVectorPlot().plot_state)
        vom = _descriptor(actions, "Vector Orientation Mapping")
        assert vom is not None, "the vectors plot should offer vector OM"
        assert vom["beta"] is True

    def test_an_ordinary_action_is_not_marked(self):
        actions = get_toolbar_config_for_plot(_FakeVectorPlot().plot_state)
        for descriptor in actions:
            if descriptor["name"] != "Vector Orientation Mapping":
                assert descriptor["beta"] is False, descriptor["name"]

    def test_every_descriptor_carries_the_key(self):
        """Absent is not the same as False to a renderer reading `a.beta`."""
        actions = get_toolbar_config_for_plot(_FakeVectorPlot().plot_state)
        assert actions, "fixture should offer at least one action"
        assert all("beta" in descriptor for descriptor in actions)


class TestBetaIsNotAGate:
    def test_marking_an_action_beta_does_not_hide_it(self):
        """The point of the flag is to ship something and say so. If it ever
        started filtering, a beta action would vanish instead of being labelled.
        """
        plot = _FakeVectorPlot()
        meta = dict(TOOLBAR_ACTIONS["functions"]["Vector Orientation Mapping"])
        assert meta.get("beta") is True
        assert _action_matches_plot("Vector Orientation Mapping", meta, plot.plot_state)

        meta.pop("beta")
        assert _action_matches_plot("Vector Orientation Mapping", meta, plot.plot_state)


class TestSchemaDeclaresIt:
    def test_vector_orientation_mapping_is_beta_in_the_schema(self):
        """Pins the declaration itself: this action is being rebuilt on the
        quantem matcher, so its results and controls are expected to move."""
        meta = TOOLBAR_ACTIONS["functions"]["Vector Orientation Mapping"]
        assert meta.get("beta") is True
