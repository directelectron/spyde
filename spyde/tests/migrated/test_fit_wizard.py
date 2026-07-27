"""The Fit wizard's staged handlers (#55, #56, #58).

Handlers are called directly, as `spyde/actions/README.md` §7 prescribes. This
covers the CONTRACT — that the model the caret shows is the model the backend
fits, that the palette carries shapes, that commit produces one map per
component — but it is explicitly NOT verification that the caret works. That
needs the real app and a screenshot (CLAUDE.md), and `fit_wizard.spec.ts` does
it.

The double-fire test is the one that matters most for a wizard: React
StrictMode fires open/close/open synchronously, and a wizard that leaves two
live controllers behind produces two of everything downstream.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import hyperspy.api as hs

from spyde.actions import fit_action
from spyde.actions.fit_action import (
    CATALOGUE,
    component_area_maps,
    component_catalogue,
    fit_add_component,
    fit_close,
    fit_commit,
    fit_open,
    fit_remove_component,
    fit_run,
    fit_set_param,
)


def _spectrum_image(ny=4, nx=5, nc=256):
    """Offset + one gaussian whose amplitude varies across the scan."""
    x = np.linspace(0.0, 50.0, nc)
    amp = np.linspace(20.0, 60.0, ny * nx).reshape(ny, nx)
    data = np.empty((ny, nx, nc))
    for iy in range(ny):
        for ix in range(nx):
            data[iy, ix] = 5.0 + amp[iy, ix] / (3 * np.sqrt(2 * np.pi)) * \
                np.exp(-((x - 25.0) ** 2) / 18.0)
    s = hs.signals.Signal1D(data)
    s.axes_manager.signal_axes[0].offset = x[0]
    s.axes_manager.signal_axes[0].scale = x[1] - x[0]
    s.metadata.General.title = "fit test"
    return s, amp


@pytest.fixture
def fitted(window):
    session = window["window"]
    sig, amp = _spectrum_image()
    session._add_signal(sig)
    tree = window["signal_trees"][0]
    plot = next(iter(tree.signal_plots))
    return session, plot, tree, amp


def _messages_of(window, kind):
    return [m for m in window["messages"] if m.get("type") == kind]


class TestOpenClose:
    def test_open_creates_a_wizard_and_sends_state(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        assert tree._fit_wizard is not None
        assert _messages_of(window, "fit_state")

    def test_open_sends_the_component_palette(self, window, fitted):
        """#56 — the picker needs SHAPES, not just names."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        cat = _messages_of(window, "fit_catalogue")
        assert cat, "no catalogue was sent"
        kinds = {c["kind"] for c in cat[-1]["components"]}
        assert {"Gaussian", "PowerLaw", "Offset"} <= kinds
        for c in cat[-1]["components"]:
            assert len(c["preview"]) > 8, c["kind"]

    def test_close_tears_the_wizard_down(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_close(session, plot, {})
        assert getattr(tree, "_fit_wizard", None) is None

    def test_double_fire_leaves_exactly_one_controller(self, window, fitted):
        """React StrictMode fires open/close/open synchronously. Two live
        controllers means two of everything downstream."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        first = tree._fit_wizard
        fit_close(session, plot, {})
        fit_open(session, plot, {})
        assert tree._fit_wizard is not None
        assert tree._fit_wizard is not first
        assert not tree._fit_wizard._closed

    def test_reopen_without_close_is_idempotent(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        wiz = tree._fit_wizard
        fit_open(session, plot, {})
        assert tree._fit_wizard is wiz

    def test_close_without_open_does_not_raise(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_close(session, plot, {})


class TestPersistence:
    """The model and the fit live on the TREE, not the controller.

    A fit costs minutes; closing the caret to get it out of the way must not
    throw it away. Only `tree.close()` disposes them.
    """

    def test_the_model_survives_close_and_reopen(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        fit_set_param(session, plot, {"component": "Gaussian",
                                      "parameter": "centre", "value": 25.0})
        fit_close(session, plot, {})
        fit_open(session, plot, {})
        state = _messages_of(window, "fit_state")[-1]
        assert [c["kind"] for c in state["components"]] == ["Gaussian"]
        assert tree._fit_wizard.spec["Gaussian"]["centre"].value == 25.0

    def test_the_fit_survives_close_and_reopen(self, window, fitted):
        """So Commit is still offered after the caret has been closed."""
        session, plot, tree, amp = fitted
        from spyde.fitting.engine import fit_batched
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        wiz = tree._fit_wizard
        wiz.result = fit_batched(wiz.spec, wiz.signal.data, wiz.axis(),
                                 device="cpu", max_iter=20)
        fit_close(session, plot, {})
        fit_open(session, plot, {})
        assert tree._fit_wizard.result is not None
        assert _messages_of(window, "fit_state")[-1]["fitted"] is True

    def test_the_model_lives_on_the_tree_not_the_controller(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        fit_close(session, plot, {})
        assert getattr(tree, "_fit_wizard", None) is None   # controller gone
        assert len(tree.fit_spec) == 1                       # model stayed

    def test_reopening_reports_the_restored_model(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        fit_close(session, plot, {})
        fit_open(session, plot, {})
        assert "restored" in (_messages_of(window, "fit_state")[-1]["status"] or "")


class TestDragHandles:
    """The on-plot handles (#57).

    The conversions are the substance: for most components the fitted amplitude
    is an AREA, so storing a dragged height straight into `A` would jump the
    curve by a factor of sigma*sqrt(2*pi).
    """

    def test_height_and_amplitude_round_trip(self):
        from spyde.actions.fit_action import (
            _DRAG, _amp_from_height, _height_from_amp,
        )
        for kind, info in _DRAG.items():
            width = 3.0
            amp = _amp_from_height(info, 7.0, width)
            assert _height_from_amp(info, amp, width) == pytest.approx(7.0), kind

    def test_a_gaussians_amplitude_is_an_area_not_a_height(self):
        """The specific trap: dragging a Gaussian handle to y=1 must store an
        AREA, not 1 — otherwise the curve leaps as soon as it is touched."""
        from spyde.actions.fit_action import _DRAG, _amp_from_height
        got = _amp_from_height(_DRAG["Gaussian"], 1.0, 2.0)
        assert got == pytest.approx(2.0 * np.sqrt(2 * np.pi))

    def test_a_height_parameterised_component_is_left_alone(self):
        from spyde.actions.fit_action import _DRAG, _amp_from_height
        assert _amp_from_height(_DRAG["GaussianHF"], 5.0, 2.0) == 5.0

    def test_only_positioned_components_get_handles(self):
        """A background has nothing on the plot to point at."""
        from spyde.actions.fit_action import _DRAG
        assert "Gaussian" in _DRAG and "Lorentzian" in _DRAG
        for background in ("Offset", "PowerLaw", "Polynomial", "Exponential"):
            assert background not in _DRAG

    def test_every_drag_target_names_real_parameters(self):
        """A typo in the drag table would silently disable the handles for that
        component — the KeyError is swallowed per-component."""
        import hyperspy.components1d as c1d
        from spyde.actions.fit_action import _DRAG
        for kind, info in _DRAG.items():
            names = {p.name for p in getattr(c1d, kind)().parameters}
            assert info["pos"] in names, kind
            assert info["amp"] in names, kind
            if info["width"]:
                assert info["width"] in names, kind

    def test_drag_writes_back_into_the_model(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard
        # Stand in for the widget: register a handle and feed it a drag event.
        wiz._widgets["w1"] = ("Gaussian", "point")
        wiz._on_widget_drag({"id": "w1", "x": 31.5, "y": 2.0})
        assert wiz.spec["Gaussian"]["centre"].value == pytest.approx(31.5)

    def test_dragging_invalidates_a_previous_fit(self, window, fitted):
        """Moving a component means the fitted parameters no longer describe
        the model — offering Commit would export a stale map."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard
        wiz.result = object()
        wiz._widgets["w1"] = ("Gaussian", "point")
        wiz._on_widget_drag({"id": "w1", "x": 20.0, "y": 1.0})
        assert wiz.result is None

    def test_an_unknown_widget_id_is_ignored(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        before = tree._fit_wizard.spec["Gaussian"]["centre"].value
        tree._fit_wizard._on_widget_drag({"id": "nope", "x": 999.0})
        assert tree._fit_wizard.spec["Gaussian"]["centre"].value == before

    def test_width_drag_keeps_the_peak_height(self, window, fitted):
        """Changing sigma alone would change the peak HEIGHT under the cursor
        for an area-parameterised component, which reads as the curve fighting
        the drag."""
        from spyde.actions.fit_action import _DRAG, _height_from_amp
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard
        comp = wiz.spec["Gaussian"]
        info = _DRAG["Gaussian"]
        before = _height_from_amp(info, comp["A"].value, comp["sigma"].value)

        wiz._widgets["r1"] = ("Gaussian", "range")
        centre = comp["centre"].value
        wiz._on_widget_drag({"id": "r1", "x0": centre - 9.0, "x1": centre + 9.0})

        after = _height_from_amp(info, comp["A"].value, comp["sigma"].value)
        assert after == pytest.approx(before, rel=1e-6)
        assert comp["sigma"].value != pytest.approx(3.0)


class TestBuildingTheModel:
    def test_add_component_appears_in_the_state(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        state = _messages_of(window, "fit_state")[-1]
        assert [c["kind"] for c in state["components"]] == ["Gaussian"]

    def test_two_of_a_kind_get_distinct_names(self, window, fitted):
        """Two Gaussians must be separately addressable by the caret and
        produce two distinct area maps at commit."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        names = [c["name"] for c in _messages_of(window, "fit_state")[-1]["components"]]
        assert len(set(names)) == 2, names

    def test_components_are_seeded_onto_the_current_axis(self, window, fitted):
        """A default Gaussian sits at 0 with sigma 1, which is off-screen on a
        0-50 axis — the preview would be a flat line and the fit would start
        nowhere near the data."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        comp = tree._fit_wizard.spec["Gaussian"]
        assert 0.0 < comp["centre"].value < 50.0
        assert comp["sigma"].value > 0.5

    def test_remove_component(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        fit_remove_component(session, plot, {"name": "Gaussian"})
        assert _messages_of(window, "fit_state")[-1]["components"] == []

    def test_set_param_updates_the_model(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        fit_set_param(session, plot, {"component": "Gaussian",
                                      "parameter": "centre", "value": 25.0})
        assert tree._fit_wizard.spec["Gaussian"]["centre"].value == 25.0

    def test_set_param_can_fix_a_parameter(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        fit_set_param(session, plot, {"component": "Gaussian",
                                      "parameter": "sigma", "free": False})
        assert tree._fit_wizard.spec["Gaussian"]["sigma"].free is False

    def test_a_bad_edit_does_not_kill_the_wizard(self, window, fitted):
        """The caret rebuilds from fit_state, so a rejected edit must leave the
        backend's model intact rather than half-applied."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        before = tree._fit_wizard.spec["Gaussian"]["centre"].value
        fit_set_param(session, plot, {"component": "NoSuch",
                                      "parameter": "centre", "value": 1.0})
        fit_set_param(session, plot, {"component": "Gaussian",
                                      "parameter": "centre", "value": "abc"})
        assert tree._fit_wizard.spec["Gaussian"]["centre"].value == before
        assert not tree._fit_wizard._closed

    def test_unknown_component_is_reported_not_added(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Flurbulator"})
        assert len(tree._fit_wizard.spec) == 0
        assert _messages_of(window, "error")


class TestRun:
    def test_run_without_components_is_refused(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_run(session, plot, {})
        assert any("component" in (m.get("text") or "").lower()
                   for m in _messages_of(window, "error"))

    def test_run_fits_every_position(self, window, fitted):
        session, plot, tree, amp = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        fit_set_param(session, plot, {"component": "Gaussian",
                                      "parameter": "centre", "value": 25.0})
        fit_set_param(session, plot, {"component": "Gaussian",
                                      "parameter": "sigma", "value": 3.0})

        wiz = tree._fit_wizard
        # Call the compute inline rather than through the worker: the marshal
        # is covered by test_lifecycle, and this test is about the fit.
        from spyde.fitting.engine import fit_batched
        wiz.result = fit_batched(wiz.spec, wiz.signal.data, wiz.axis(),
                                 device="cpu", max_iter=80)
        assert wiz.result.values.shape[0] == 20
        assert wiz.result.convergence_rate > 0.5


class TestCommit:
    def test_commit_without_a_fit_is_refused(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        fit_commit(session, plot, {})
        assert any("run the fit" in (m.get("text") or "").lower()
                   for m in _messages_of(window, "error"))

    def test_commit_makes_one_tree_with_a_view_per_component(self, window, fitted):
        """#58 — the maps ride commit_result_tree, which already gives the
        strain-style toggle. No new display code."""
        session, plot, tree, amp = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        fit_set_param(session, plot, {"component": "Gaussian",
                                      "parameter": "centre", "value": 25.0})
        fit_set_param(session, plot, {"component": "Gaussian",
                                      "parameter": "sigma", "value": 3.0})
        wiz = tree._fit_wizard
        from spyde.fitting.engine import fit_batched
        wiz.result = fit_batched(wiz.spec, wiz.signal.data, wiz.axis(),
                                 device="cpu", max_iter=80)

        before = len(window["signal_trees"])
        fit_commit(session, plot, {})
        assert len(window["signal_trees"]) == before + 1


class TestComponentAreaMaps:
    def test_area_tracks_the_amplitude(self, fitted):
        """The map has to MEAN something: a bigger gaussian must give a bigger
        area, scored against the amplitudes the data was built from."""
        session, plot, tree, amp = fitted
        from spyde.fitting.engine import fit_batched
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard
        wiz.spec["Gaussian"]["centre"].value = 25.0
        wiz.spec["Gaussian"]["sigma"].value = 3.0
        res = fit_batched(wiz.spec, wiz.signal.data, wiz.axis(), device="cpu",
                          max_iter=80)
        maps = component_area_maps(wiz.spec, res, wiz.axis(), amp.shape)
        assert set(maps) == {"Offset", "Gaussian"}
        r = np.corrcoef(maps["Gaussian"].ravel(), amp.ravel())[0, 1]
        assert r > 0.99, f"component area does not track amplitude (r={r:.3f})"

    def test_one_map_per_component_shaped_to_the_scan(self, fitted):
        session, plot, tree, amp = fitted
        from spyde.fitting.engine import fit_batched
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        wiz = tree._fit_wizard
        res = fit_batched(wiz.spec, wiz.signal.data, wiz.axis(), device="cpu",
                          max_iter=20)
        maps = component_area_maps(wiz.spec, res, wiz.axis(), amp.shape)
        assert maps["Offset"].shape == amp.shape


class TestCatalogue:
    def test_every_offered_component_previews(self):
        """A palette entry that fails to sample would render as a blank button
        — worse than not offering it."""
        x = np.linspace(200.0, 800.0, 256)
        got = {c["kind"] for c in component_catalogue(x)}
        assert got == {k for k, _ in CATALOGUE}

    def test_previews_are_normalised_and_finite(self):
        """The sparkline is about SHAPE; an un-normalised power law would
        render as a spike beside a flat gaussian."""
        for c in component_catalogue(np.linspace(200.0, 800.0, 256)):
            p = np.asarray(c["preview"], float)
            assert np.isfinite(p).all(), c["kind"]
            assert p.min() >= -1e-9 and p.max() <= 1.0 + 1e-9, c["kind"]

    def test_peak_shapes_are_actually_peaks(self):
        """Seeding must put a peak IN the axis range — the real failure mode is
        a palette where every peak previews as a flat line."""
        cat = {c["kind"]: np.asarray(c["preview"], float)
               for c in component_catalogue(np.linspace(200.0, 800.0, 256))}
        for kind in ("Gaussian", "Lorentzian", "GaussianHF"):
            p = cat[kind]
            assert p.max() - p.min() > 0.5, f"{kind} previews flat"
            peak = int(np.argmax(p))
            assert 5 < peak < len(p) - 5, f"{kind} peaks at the edge"


class TestToolbarGating:
    def test_fit_is_offered_on_1d_plots_only(self):
        from spyde import TOOLBAR_ACTIONS
        meta = TOOLBAR_ACTIONS["functions"]["Fit"]
        assert meta["plot_dim"] == [1]
        assert meta["navigation"] is False

    def test_every_staged_handler_is_registered(self):
        from spyde.actions import registry
        for stage in ("open", "close", "add_component", "remove_component",
                      "set_param", "tune", "run", "commit"):
            assert registry.resolve_staged(f"fit_{stage}") is not None, stage

    def test_the_parameter_schema_resolves(self):
        """Three-host parity: the caret, a notebook form and the docs all
        resolve the schema through the registry."""
        from spyde.actions import registry
        schema = registry.wizard_parameters("fit")
        assert set(schema) == {"max_iter", "seeded", "weighting"}


class TestFitCurrentSpectrum:
    """The "Fit spectrum" button (#57 workflow).

    Building a model is a loop — place, look, nudge. Fitting the whole scan to
    check one guess is the wrong unit of work, so this fits only what is on
    screen and writes the answer back into the model.
    """

    def test_fits_only_the_displayed_spectrum(self, window, fitted):
        from spyde.actions.fit_action import fit_current
        session, plot, tree, amp = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard
        wiz.spec["Gaussian"]["centre"].value = 25.0
        wiz.spec["Gaussian"]["sigma"].value = 3.0

        fit_current(session, plot, {})
        # The OFFSET is the easy check: the data sits on a constant 5.0.
        assert wiz.spec["Offset"]["offset"].value == pytest.approx(5.0, abs=0.5)

    def test_writes_the_result_back_into_the_model(self, window, fitted):
        """So the next nudge starts from a fitted position, and the handles
        move to it — the difference between a preview and a workflow step."""
        from spyde.actions.fit_action import fit_current
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        before = tree._fit_wizard.spec["Offset"]["offset"].value
        fit_current(session, plot, {})
        assert tree._fit_wizard.spec["Offset"]["offset"].value != before

    def test_does_not_produce_a_scan_result(self, window, fitted):
        """One spectrum is not a map. Offering Commit after it would export a
        component map built from a single pixel repeated everywhere."""
        from spyde.actions.fit_action import fit_current
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        fit_current(session, plot, {})
        assert tree._fit_wizard.result is None
        assert _messages_of(window, "fit_state")[-1]["fitted"] is False

    def test_refuses_with_no_components(self, window, fitted):
        from spyde.actions.fit_action import fit_current
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_current(session, plot, {})
        assert any("component" in (m.get("text") or "").lower()
                   for m in _messages_of(window, "error"))

    def test_reports_whether_it_converged(self, window, fitted):
        from spyde.actions.fit_action import fit_current
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        fit_current(session, plot, {})
        assert "converge" in (_messages_of(window, "fit_state")[-1]["status"] or "")


class TestParameterEditsMoveTheHandles:
    """A typed value and a dragged handle must agree — both directions.

    Handles are MOVED in place rather than rebuilt: rebuilding on every
    keystroke destroys and recreates every widget several times a second, which
    makes them flicker and can pull one out from under the cursor.
    """

    def test_editing_a_parameter_moves_its_handle(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard

        moved = {}

        class _FakeWidget:
            def set(self, **kw):
                moved.update(kw)

        class _FakeP1:
            def get_widget(self, wid):
                return _FakeWidget()

        wiz.plot._plot1d = _FakeP1()
        wiz._widgets = {"p1": ("Gaussian", "point")}
        fit_set_param(session, plot, {"component": "Gaussian",
                                      "parameter": "centre", "value": 41.0})
        assert moved.get("x") == pytest.approx(41.0)

    def test_a_parameter_edit_does_not_rebuild_the_handles(self, window, fitted):
        """Rebuilding is for a changed component LIST, not for a keystroke."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard
        wiz._widgets = {"p1": ("Gaussian", "point")}
        fit_set_param(session, plot, {"component": "Gaussian",
                                      "parameter": "centre", "value": 30.0})
        assert "p1" in wiz._widgets, "handles were torn down by a value edit"
