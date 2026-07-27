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

    def test_every_offered_component_has_handles(self):
        """No component is editable only through the caret's number boxes.

        A background has no peak to point at, so it takes the ANCHOR route
        instead of the point/range pair — but it does take one.
        """
        from spyde.actions.fit_action import _ANCHORS, _DRAG
        for kind, _description in CATALOGUE:
            assert kind in _DRAG or kind in _ANCHORS, kind

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

    def test_a_drag_event_carries_the_widget_on_event_source(self, window, fitted):
        """THE bug this class exists for.

        anyplotlib hands the dragged widget back on ``event.source``, so the
        position is ``event.source.x`` — not ``event.x``. Reading ``event.x``
        returns None and the drag does NOTHING, while the handle still moves on
        screen because the widget draws itself. The model silently never hears
        about it, which is exactly how it looked in the app.
        """
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard

        class _Src:
            x, y = 31.5, 2.0

        class _Event:
            source = _Src()

        wiz._on_widget_drag("Gaussian", "point", _Event())
        assert wiz.spec["Gaussian"]["centre"].value == pytest.approx(31.5)

    def test_dragging_invalidates_a_previous_fit(self, window, fitted):
        """Moving a component means the fitted parameters no longer describe
        the model — offering Commit would export a stale map."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard
        wiz.result = object()
        wiz._on_widget_drag("Gaussian", "point", {"x": 20.0, "y": 1.0})
        assert wiz.result is None

    def test_an_unknown_component_is_ignored(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        before = tree._fit_wizard.spec["Gaussian"]["centre"].value
        tree._fit_wizard._on_widget_drag("NoSuch", "point", {"x": 999.0})
        assert tree._fit_wizard.spec["Gaussian"]["centre"].value == before

    def test_a_reentrant_drag_is_guarded(self, window, fitted):
        """Moving the handles in response to a drag re-enters the handler. The
        example guards this with `_syncing`; without it the widget and the
        model chase each other."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard
        wiz._syncing = True
        before = wiz.spec["Gaussian"]["centre"].value
        wiz._on_widget_drag("Gaussian", "point", {"x": 12.0})
        assert wiz.spec["Gaussian"]["centre"].value == before

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
        before_h = _height_from_amp(info, comp["A"].value, comp["sigma"].value)
        before_sigma = comp["sigma"].value

        centre = comp["centre"].value
        wiz._on_widget_drag("Gaussian", "range",
                            {"x0": centre - 9.0, "x1": centre + 9.0})

        after_h = _height_from_amp(info, comp["A"].value, comp["sigma"].value)
        assert after_h == pytest.approx(before_h, rel=1e-6)
        assert comp["sigma"].value != pytest.approx(before_sigma)


class TestBackgroundAnchors:
    """Backgrounds get handles too — anchor points on the curve.

    The contract is that the curve passes THROUGH the handle you are holding,
    exactly. A curve that lands near your cursor rather than under it reads as
    broken, so these assert to 1e-9, not to a tolerance.
    """

    @pytest.mark.parametrize("kind", ["Offset", "Polynomial", "PowerLaw",
                                      "Exponential"])
    def test_dragging_an_anchor_puts_the_curve_under_it(self, window, fitted,
                                                        kind):
        from spyde.actions.fit_action import evaluate_component
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": kind})
        wiz = tree._fit_wizard
        name = wiz.spec.components[-1].name
        comp = wiz.spec[name]
        entry = wiz._widgets[name]

        x0 = entry["at"][0]
        target = float(evaluate_component(comp, [x0])[0]) * 1.7
        wiz._on_widget_drag(name, "anchor:0", {"x": x0, "y": target})

        assert float(evaluate_component(comp, [x0])[0]) == pytest.approx(
            target, rel=1e-9)

    @pytest.mark.parametrize("kind", ["PowerLaw", "Exponential"])
    def test_the_anchor_you_are_not_holding_stays_put(self, window, fitted,
                                                      kind):
        """Two anchors determine both parameters exactly. If the solve only
        used the dragged one the curve would pivot out from under the other."""
        from spyde.actions.fit_action import evaluate_component
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": kind})
        wiz = tree._fit_wizard
        name = wiz.spec.components[-1].name
        comp = wiz.spec[name]
        entry = wiz._widgets[name]

        x0, x1 = entry["at"]
        held = float(evaluate_component(comp, [x1])[0])
        wiz._on_widget_drag(
            name, "anchor:0",
            {"x": x0, "y": float(evaluate_component(comp, [x0])[0]) * 2.0})

        assert float(evaluate_component(comp, [x1])[0]) == pytest.approx(
            held, rel=1e-9)

    def test_a_background_gets_widgets_on_the_plot(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "PowerLaw"})
        wiz = tree._fit_wizard
        entry = wiz._widgets[wiz.spec.components[-1].name]
        assert len(entry["anchors"]) == 2

    def test_a_nonsense_drag_leaves_the_model_alone(self, window, fitted):
        """A power law is undefined at y<=0. Solving anyway would write a nan
        into the model and every curve afterwards would vanish."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "PowerLaw"})
        wiz = tree._fit_wizard
        name = wiz.spec.components[-1].name
        before = [p.value for p in wiz.spec[name].scalar_parameters]
        wiz._on_widget_drag(name, "anchor:0",
                            {"x": wiz._widgets[name]["at"][0], "y": -5.0})
        assert [p.value for p in wiz.spec[name].scalar_parameters] == before

    def test_anchors_follow_a_parameter_edit(self, window, fitted):
        """Typing a value in the caret must move the handles onto the new
        curve, the same way it does for a gaussian."""
        from spyde.actions.fit_action import evaluate_component
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        wiz = tree._fit_wizard
        name = wiz.spec.components[-1].name
        fit_set_param(session, plot, {"component": name, "parameter": "offset",
                                      "value": 123.0})
        widget = wiz._widgets[name]["anchors"][0]
        assert widget.get("y") == pytest.approx(123.0)
        assert evaluate_component(wiz.spec[name], [0.0])[0] == pytest.approx(123.0)


class TestARunFillsTheStore:
    """A whole-scan run has fitted every position — so say so, and remember it.

    It used to report "0 fitted" straight after fitting the entire scan, and
    navigating afterwards refit positions it had just solved. It also left the
    handles where they were while the curves jumped to the fitted values, and
    seeded the post-run preview from position 0 because it read
    `current_indices` off the PLOT (the same mistake that made "Fit spectrum"
    fit the navigation mean).
    """

    def test_position_of_inverts_the_flat_index(self, window, fitted):
        """On a square scan a transposed key looks identical, so this is
        checked on a NON-square one."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        wiz = tree._fit_wizard
        nav = (4, 5)
        for iy in range(nav[0]):
            for ix in range(nav[1]):
                flat = int(np.ravel_multi_index((iy, ix), nav))
                # `remember` keys on current_indices(), which the run path
                # ravels as reversed(indices) — so the key is (ix, iy).
                assert wiz.position_of(flat, nav) == (ix, iy)

    @staticmethod
    def _run(window, fitted):
        """Fit the whole scan and record it, without the worker marshal.

        `fit_run`'s `_done` is dispatched onto the asyncio main thread, which
        does not turn under pytest — the existing run test sidesteps it the
        same way. The compute is the engine's; what is under test here is what
        the wizard does with the result.
        """
        from spyde.fitting.engine import fit_batched
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        wiz = tree._fit_wizard
        result = fit_batched(wiz.spec, wiz.signal.data, wiz.axis(),
                             device="cpu", max_iter=60)
        wiz.record_run(result, wiz.nav_shape())
        return wiz, result

    def test_a_run_records_every_converged_position(self, window, fitted):
        _session, _plot, tree, _ = fitted
        wiz, result = self._run(window, fitted)
        assert tree.fit_store, "a whole-scan run recorded nothing"
        assert len(tree.fit_store) == int(np.asarray(result.converged).sum())
        for key, values in tree.fit_store.items():
            assert len(key) == 2
            assert len(values) == len(tree.fit_spec.parameter_names())

    def test_an_unconverged_position_is_not_remembered(self, window, fitted):
        """A failed fit is not an answer. Storing it would make an adaptive
        pass recall the failure instead of retrying from a better seed."""
        _session, _plot, tree, _ = fitted
        wiz, result = self._run(window, fitted)
        result.converged[:] = False
        wiz.record_run(result, wiz.nav_shape())
        assert tree.fit_store == {}

    def test_navigating_after_a_run_recalls_instead_of_refitting(self, window,
                                                                 fitted):
        _session, _plot, tree, _ = fitted
        wiz, _result = self._run(window, fitted)
        key = next(iter(tree.fit_store))
        _fake_nav(tree, key)
        assert wiz.recall() is True

    def test_a_run_reports_its_coverage(self, window, fitted):
        _session, _plot, tree, _ = fitted
        wiz, _result = self._run(window, fitted)
        wiz.emit_state()
        state = _messages_of(window, "fit_state")[-1]
        assert state["fitted_count"] == len(tree.fit_store) > 0


class TestTwoOfAKindCanBeFitted:
    """A second component of a kind must not start ON the first.

    Every component is seeded mid-axis, so two gaussians used to arrive with
    identical centres AND sigmas — two IDENTICAL Jacobian columns
    (corr = 1.000000, cond(J) = 1e16). A perfectly degenerate pair can never
    separate: the solver moves both the same way forever, so it grows one and
    shrinks the other, which reads as "it forces the second gaussian to 0".
    On two well-separated peaks the pair instead ran away together to a sigma
    larger than the whole axis.
    """

    @staticmethod
    def _separated(n=1024):
        x = np.linspace(0.0, 100.0, n)
        y = (800 * np.exp(-0.5 * ((x - 30.0) / 5.0) ** 2)
             + 800 * np.exp(-0.5 * ((x - 70.0) / 5.0) ** 2) + 20.0)
        return x, y

    @staticmethod
    def _caret_model(x, y, kinds):
        """What fit_add_component builds, without needing a Session."""
        from spyde.actions.fit_action import (
            _seed_for_preview, clamp_to_axis, new_component_spec,
            scale_to_data, spread_repeats,
        )
        from spyde.fitting import ModelSpec
        lo, hi = float(x.min()), float(x.max())
        spec = ModelSpec()
        for kind in kinds:
            c = new_component_spec(kind)
            _seed_for_preview(c, lo, hi)
            scale_to_data(c, x, y)
            spread_repeats(c, spec, lo, hi)
            clamp_to_axis(c, lo, hi)
            while any(e.name == c.name for e in spec.components):
                c.name += "'"
            spec.append(c)
        return spec

    def test_two_of_a_kind_do_not_share_a_centre(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        centres = [float(c["centre"].value)
                   for c in tree.fit_spec.active_components]
        assert centres[0] != centres[1], "a degenerate pair cannot be fitted"

    def test_the_pair_still_straddles_the_seed(self, window, fitted):
        """Symmetric, so a genuinely co-centred pair still starts co-centred —
        which is what makes hyperspy's two_gaussians (both true centres 50)
        fit exactly as well as it did before."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        before = float(tree.fit_spec["Gaussian"]["centre"].value)
        fit_add_component(session, plot, {"kind": "Gaussian"})
        centres = [float(c["centre"].value)
                   for c in tree.fit_spec.active_components]
        assert sum(centres) / 2 == pytest.approx(before, abs=1e-6)

    def test_separated_peaks_are_actually_recovered(self):
        """The end of it: this fit used to reach chisq 6e7 with both
        components pinned at a sigma wider than the whole axis."""
        from spyde.fitting.engine import fit_batched
        x, y = self._separated()
        spec = self._caret_model(x, y, ("Offset", "Gaussian", "Gaussian"))
        r = fit_batched(spec, y[None, :], x, device="cpu", max_iter=200)
        spec.set_flat_values(r.values[0])
        got = sorted((round(float(c["centre"].value), 2),
                      round(float(c["sigma"].value), 2))
                     for c in spec.active_components if c.kind == "Gaussian")
        assert got == [(30.0, 5.0), (70.0, 5.0)]
        assert float(r.chisq[0]) < 1e-6

    def test_a_co_centred_pair_is_still_recovered(self):
        """The other half of the trade, and the one the spacing was chosen
        for: hyperspy's two_gaussians is genuinely two co-centred peaks, a
        wide one and a narrow one at 1.7% of its height."""
        from spyde.fitting.engine import fit_batched
        x = np.linspace(0.0, 100.0, 1024)
        y = (900 * np.exp(-0.5 * ((x - 50.0) / 25.0) ** 2)
             + 150 * np.exp(-0.5 * ((x - 50.0) / 2.5) ** 2) + 20.0)
        spec = self._caret_model(x, y, ("Offset", "Gaussian", "Gaussian"))
        r = fit_batched(spec, y[None, :], x, device="cpu", max_iter=200)
        spec.set_flat_values(r.values[0])
        got = sorted((round(float(c["centre"].value), 1),
                      round(float(c["sigma"].value), 1))
                     for c in spec.active_components if c.kind == "Gaussian")
        assert got == [(50.0, 2.5), (50.0, 25.0)]

    def test_a_peak_cannot_wander_off_the_data(self):
        """Bounds are NOT redundant with the spread: with the spread and
        without them, a noisy near-overlapping pair still diverged to a
        centre off the axis and chisq 1.1e8."""
        from spyde.actions.fit_action import clamp_to_axis, new_component_spec
        c = new_component_spec("Gaussian")
        clamp_to_axis(c, 200.0, 800.0)
        assert (c["centre"].bmin, c["centre"].bmax) == (200.0, 800.0)
        assert c["sigma"].bmax == 600.0
        assert c["sigma"].bmin > 0.0

    def test_a_background_is_not_bounded_to_the_axis(self):
        """A PowerLaw's `origin` is a reference point, deliberately placed
        OUTSIDE the axis when the axis would otherwise contain its
        singularity. Clamping it there would undo `seed_background`."""
        from spyde.actions.fit_action import (
            clamp_to_axis, new_component_spec, seed_background,
        )
        x = np.linspace(0.0, 100.0, 512)
        c = new_component_spec("PowerLaw")
        seed_background(c, x, 1000.0 / (x + 30.0) + 5.0)
        origin = float(c["origin"].value)
        assert origin < 0.0, "the singularity was not moved off the axis"
        clamp_to_axis(c, 0.0, 100.0)
        assert float(c["origin"].value) == origin


class TestBackgroundSeeding:
    """A new background arrives ON the data.

    Found in the app, invisible here until these existed: a PowerLaw was
    seeded with `origin` at the axis MIDPOINT (the rule that puts a gaussian's
    centre there also matched `origin`), so it was identically zero over the
    left half of the spectrum and its anchors sat outside its own domain. The
    two obvious repairs both failed at the other end — scaling by peak matches
    a power law's singular left edge and leaves ~3e-5 everywhere else; scaling
    by median level puts 3.7e10 at that same edge. A background needs its
    SHAPE solved from the data, which is what `seed_background` does.
    """

    @staticmethod
    def _spectrum():
        x = np.linspace(0.0, 102.3, 1024)
        y = (1000 * np.exp(-0.5 * ((x - 40) / 6.0) ** 2)
             + 800 * np.exp(-0.5 * ((x - 60) / 9.0) ** 2)
             + 300 * np.exp(-x / 30.0) + 5.0)
        return x, y

    @pytest.mark.parametrize("kind", ["PowerLaw", "Exponential", "Offset",
                                      "Polynomial"])
    def test_a_seeded_background_is_the_same_size_as_the_data(self, kind):
        from spyde.actions.fit_action import (
            evaluate_component, new_component_spec, seed_background,
            _seed_for_preview,
        )
        x, y = self._spectrum()
        cspec = new_component_spec(kind, 2 if kind == "Polynomial" else None)
        _seed_for_preview(cspec, float(x[0]), float(x[-1]))
        assert seed_background(cspec, x, y) is True
        curve = evaluate_component(cspec, x)
        assert np.isfinite(curve).all(), f"{kind} seeded to a non-finite curve"
        # Within an order of magnitude of the data at both ends — the two
        # failures this replaced were 5 orders low and 7 orders high.
        assert 0.1 * np.median(y) < np.max(curve) < 10 * np.max(y), kind

    def test_a_background_is_not_seeded_onto_a_peak(self):
        """The single anchor of a flat background sits mid-axis, which on this
        spectrum is a peak. Sampling there seeded an Offset at 734 against a
        baseline of 17."""
        from spyde.actions.fit_action import (
            new_component_spec, seed_background,
        )
        x, y = self._spectrum()
        cspec = new_component_spec("Offset")
        assert seed_background(cspec, x, y)
        baseline = float(np.percentile(y, 5))
        assert cspec["offset"].value < 4 * baseline

    def test_a_peak_is_not_treated_as_a_background(self):
        from spyde.actions.fit_action import new_component_spec, seed_background
        x, y = self._spectrum()
        assert seed_background(new_component_spec("Gaussian"), x, y) is False

    def test_flat_data_cannot_fix_an_exponential(self):
        """Two equal points do not determine tau. Returning True there would
        leave nan in the model; returning False sends the caller to
        `scale_to_data`, which at least makes it visible."""
        from spyde.actions.fit_action import new_component_spec, seed_background
        x = np.linspace(0.0, 20.0, 512)
        assert seed_background(new_component_spec("Exponential"), x,
                               np.full_like(x, 30.0)) is False

    def test_a_power_law_keeps_hyperspys_origin_when_it_can(self):
        """origin=0 is the convention every downstream tool assumes; it is
        only moved when the axis would otherwise contain the singularity."""
        from spyde.actions.fit_action import new_component_spec, seed_background
        e = np.linspace(200.0, 800.0, 512)
        cspec = new_component_spec("PowerLaw")
        seed_background(cspec, e, 5e6 * e ** -3.2 + 20)
        assert cspec["origin"].value == 0.0
        assert cspec["origin"].free is False

    def test_adding_a_background_puts_its_handles_on_the_curve(self, window,
                                                               fitted):
        """The end-to-end version: the handles must not start at y=0."""
        from spyde.actions.fit_action import evaluate_component
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "PowerLaw"})
        wiz = tree._fit_wizard
        name = wiz.spec.components[-1].name
        entry = wiz._widgets[name]
        ys = evaluate_component(wiz.spec[name], entry["at"])
        assert all(v > 0 for v in ys), "a background arrived flat on the axis"


class TestHandlesStayOnTheComponent:
    """Every handle sits on the curve, INCLUDING mid-drag.

    Updating the partner only on release left it where the drag started while
    the curve moved out from under it, so the handles drifted off the component
    and snapped back when you let go.
    """

    def test_the_range_band_tracks_a_live_point_drag(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard
        band = wiz._widgets["Gaussian"]["range"]

        wiz._on_widget_drag("Gaussian", "point", {"x": 40.0, "y": 3.0},
                            live=True)

        centre = (band.get("x0") + band.get("x1")) / 2.0
        assert centre == pytest.approx(40.0), "the band lagged behind the point"

    def test_the_point_tracks_a_live_width_drag(self, window, fitted):
        from spyde.actions.fit_action import _DRAG, _height_from_amp
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard
        point = wiz._widgets["Gaussian"]["point"]

        wiz._on_widget_drag("Gaussian", "range", {"x0": 10.0, "x1": 30.0},
                            live=True)

        comp = wiz.spec["Gaussian"]
        assert point.get("x") == pytest.approx(20.0)
        assert point.get("y") == pytest.approx(_height_from_amp(
            _DRAG["Gaussian"], comp["A"].value, comp["sigma"].value))

    def test_the_handle_being_dragged_is_not_written_back(self, window, fitted):
        """Writing a position back to the handle under the user's finger fights
        the drag — it is the one thing update_widgets must skip."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard
        point = wiz._widgets["Gaussian"]["point"]
        point.set(x=-999.0, y=-999.0)      # where the widget thinks it is

        wiz.update_widgets(skip="point")

        assert point.get("x") == -999.0

    def test_a_live_drag_still_does_not_re_send_the_model(self, window, fitted):
        """The partner handle is a targeted widget push; `fit_state` is the
        whole model. Only the cheap one belongs on the pointer path."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard
        before = len(_messages_of(window, "fit_state"))
        wiz._on_widget_drag("Gaussian", "point", {"x": 30.0}, live=True)
        assert len(_messages_of(window, "fit_state")) == before


class TestSpecsAreBuiltOnce:
    """Constructing a hyperspy component costs 20-110 ms of sympy lambdify and
    the structure of a kind never changes, so it is read once per process."""

    def test_the_prototype_is_shared(self):
        from spyde.actions.fit_action import prototype
        assert prototype("Gaussian") is prototype("Gaussian")

    def test_each_caller_gets_its_own_spec(self):
        """Shared PROTOTYPE, independent SPEC — two Gaussians in one model must
        not share parameter objects, or moving one moves the other."""
        from spyde.actions.fit_action import new_component_spec
        a, b = new_component_spec("Gaussian"), new_component_spec("Gaussian")
        assert a is not b and a["centre"] is not b["centre"]
        a["centre"].value = 42.0
        assert b["centre"].value != 42.0

    def test_a_polynomial_keeps_its_order(self):
        from spyde.actions.fit_action import new_component_spec
        assert len(new_component_spec("Polynomial", 4).scalar_parameters) == 5
        assert len(new_component_spec("Polynomial", 2).scalar_parameters) == 3

    def test_the_catalogue_is_cached_per_axis(self):
        """Reopening the caret on the same signal must cost nothing."""
        x = np.linspace(0.0, 50.0, 128)
        first = component_catalogue(x)
        assert component_catalogue(x) is first
        assert component_catalogue(np.linspace(0.0, 900.0, 128)) is not first

    def test_the_background_wizard_shares_the_cache(self, window, fitted):
        """Same specs, one build — the two wizards offer the same components."""
        from spyde.actions import background_action, fit_action
        calls = []
        real = fit_action.new_component_spec

        def counted(kind, order=None):
            calls.append(kind)
            return real(kind, order)

        fit_action.new_component_spec = counted
        try:
            session, plot, tree, _ = fitted
            background_action.bg_open(session, plot, {})
            wiz = tree._bg_wizard
            wiz.model_kind = "PowerLaw"
            wiz.build_spec()
        finally:
            fit_action.new_component_spec = real
        assert "PowerLaw" in calls


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

        from spyde.actions.fit_action import _DRAG
        moved = {}

        class _FakeWidget:
            id = "w"

            def set(self, **kw):
                moved.update(kw)

        wiz._widgets = {"Gaussian": {"point": _FakeWidget(),
                                     "info": _DRAG["Gaussian"]}}
        fit_set_param(session, plot, {"component": "Gaussian",
                                      "parameter": "centre", "value": 41.0})
        assert moved.get("x") == pytest.approx(41.0)

    def test_a_parameter_edit_does_not_rebuild_the_handles(self, window, fitted):
        """Rebuilding is for a changed component LIST, not for a keystroke.

        Rebuilding several times a second while typing makes the handles
        flicker and can pull one out from under the cursor.
        """
        from spyde.actions.fit_action import _DRAG
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard

        class _FakeWidget:
            id = "w"

            def set(self, **kw):
                pass

        sentinel = _FakeWidget()
        wiz._widgets = {"Gaussian": {"point": sentinel,
                                     "info": _DRAG["Gaussian"]}}
        fit_set_param(session, plot, {"component": "Gaussian",
                                      "parameter": "centre", "value": 30.0})
        assert wiz._widgets["Gaussian"]["point"] is sentinel, \
            "handles were torn down and rebuilt by a value edit"


class TestFitsWhatIsOnScreen:
    """"Fit spectrum" must fit the spectrum being DISPLAYED.

    `plot.current_data` is the authority — it is the array the plot is showing,
    already resolved through whatever navigator or region path produced it.
    Reconstructing it from `signal.data` plus a navigator index failed in the
    worst way available: it fell through to the mean over navigation, so the
    fit converged happily against a spectrum nobody was looking at and the
    drawn model came out at about half the data's height with "converged"
    beside it.
    """

    def test_uses_the_painted_spectrum(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        wiz = tree._fit_wizard
        painted = np.linspace(1.0, 2.0, len(wiz.axis()))
        plot.current_data = painted
        np.testing.assert_allclose(wiz.current_spectrum(), painted)

    def test_does_not_silently_use_the_navigation_mean(self, window, fitted):
        """The specific failure. The mean is a fine PREVIEW stand-in before the
        first frame lands, but once something is painted it must win."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        wiz = tree._fit_wizard
        nav_mean = np.asarray(wiz.signal.data, float).reshape(
            -1, len(wiz.axis())).mean(0)
        plot.current_data = np.asarray(wiz.signal.data, float)[0, 0]
        got = wiz.current_spectrum()
        assert not np.allclose(got, nav_mean), \
            "fell back to the navigation mean despite painted data"
        np.testing.assert_allclose(got, np.asarray(wiz.signal.data, float)[0, 0])

    def test_ignores_a_stale_shape(self, window, fitted):
        """current_data can hold a dask Future or a differently-shaped frame
        mid-transition; neither is this signal's spectrum."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        wiz = tree._fit_wizard
        for bad in (object(), np.zeros((4, 4)), np.zeros(7)):
            plot.current_data = bad
            assert len(wiz.current_spectrum()) == len(wiz.axis())

    def test_fit_current_matches_the_displayed_pixel(self, window, fitted):
        """End to end: the fitted amplitude must track the pixel on screen, not
        an average of the scan."""
        from spyde.actions.fit_action import fit_current
        session, plot, tree, amp = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard
        wiz.spec["Gaussian"]["centre"].value = 25.0
        wiz.spec["Gaussian"]["sigma"].value = 3.0

        # The BRIGHTEST pixel — its amplitude is well above the scan mean, so a
        # fit against the mean cannot pass this.
        iy, ix = np.unravel_index(int(np.argmax(amp)), amp.shape)
        plot.current_data = np.asarray(wiz.signal.data, float)[iy, ix]
        fit_current(session, plot, {})
        assert wiz.spec["Gaussian"]["A"].value == pytest.approx(
            float(amp[iy, ix]), rel=0.05)


def _fake_nav(tree, idx):
    """Stand in for the navigation selector.

    `current_indices` lives on the SELECTOR, not the plot — reading it off the
    plot is what made "Fit spectrum" silently fit the navigation mean, so the
    fake deliberately mirrors the real shape.
    """
    class _Sel:
        current_indices = idx

    class _NPM:
        navigation_selectors = {0: [_Sel()]}

    tree.navigator_plot_manager = _NPM()


class TestPerPositionMemory:
    """Each navigator position remembers its own fit.

    Scrubbing back to a pixel should show what was found THERE, not whatever
    the last pixel left in the model.
    """

    def test_reads_the_position_from_the_selector(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        _fake_nav(tree, (2, 3))
        assert tree._fit_wizard.current_indices() == (2, 3)

    def test_no_selector_means_no_position(self, window, fitted):
        """With nothing to ask, the answer is None — and `remember` then stores
        nothing rather than inventing a key that would collide with every other
        position that also had no selector."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        tree.navigator_plot_manager = None
        wiz = tree._fit_wizard
        assert wiz.current_indices() is None
        wiz.remember([1.0])
        assert not tree.fit_store

    def test_a_real_session_exposes_a_position(self, window, fitted):
        """The accessor works against the REAL selector, not just the fake —
        this is the half that was wrong before (it read the plot, not the
        selector, and so always saw None)."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        assert tree._fit_wizard.current_indices() is not None

    def test_remembers_and_recalls_per_position(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        wiz = tree._fit_wizard

        _fake_nav(tree, (0, 0))
        wiz.remember([11.0])
        _fake_nav(tree, (1, 1))
        wiz.remember([22.0])

        _fake_nav(tree, (0, 0))
        assert wiz.recall() is True
        assert wiz.spec["Offset"]["offset"].value == pytest.approx(11.0)
        _fake_nav(tree, (1, 1))
        assert wiz.recall() is True
        assert wiz.spec["Offset"]["offset"].value == pytest.approx(22.0)

    def test_an_unvisited_position_recalls_nothing(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        wiz = tree._fit_wizard
        _fake_nav(tree, (0, 0))
        wiz.remember([11.0])
        _fake_nav(tree, (3, 3))
        assert wiz.recall() is False

    def test_changing_the_model_clears_the_store(self, window, fitted):
        """Stored vectors are POSITIONAL. After an add or remove they would be
        reinterpreted against the wrong parameters — a stored sigma arriving as
        an amplitude — and nothing would look obviously wrong."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        wiz = tree._fit_wizard
        _fake_nav(tree, (0, 0))
        wiz.remember([11.0])
        assert tree.fit_store

        fit_add_component(session, plot, {"kind": "Gaussian"})
        assert not tree.fit_store, "stale positional vectors survived a model change"

    def test_a_wrong_length_vector_is_refused(self, window, fitted):
        """Belt and braces for the same hazard."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        wiz = tree._fit_wizard
        _fake_nav(tree, (0, 0))
        tree.fit_store[(0, 0)] = np.array([1.0, 2.0, 3.0])   # wrong width
        assert wiz.recall() is False

    def test_the_store_survives_close_and_reopen(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        _fake_nav(tree, (0, 0))
        tree._fit_wizard.remember([7.0])
        fit_close(session, plot, {})
        fit_open(session, plot, {})
        assert tree._fit_wizard.recall() is True


class TestAdaptiveFit:
    def test_navigating_recalls_before_it_refits(self, window, fitted):
        """A stored answer wins over a fresh fit — same computation, already
        done, and re-running it could land somewhere slightly different."""
        from spyde.actions.fit_action import fit_navigated
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        wiz = tree._fit_wizard
        _fake_nav(tree, (0, 0))
        wiz.remember([42.0])
        wiz.spec["Offset"]["offset"].value = 0.0

        fit_navigated(session, plot, {"adaptive": True})
        assert wiz.spec["Offset"]["offset"].value == pytest.approx(42.0)

    def test_adaptive_off_leaves_the_model_alone(self, window, fitted):
        from spyde.actions.fit_action import fit_navigated
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        wiz = tree._fit_wizard
        _fake_nav(tree, (3, 3))
        before = wiz.spec["Offset"]["offset"].value
        fit_navigated(session, plot, {"adaptive": False})
        assert wiz.spec["Offset"]["offset"].value == before

    def test_adaptive_on_fits_an_unvisited_position(self, window, fitted):
        from spyde.actions.fit_action import fit_navigated
        session, plot, tree, amp = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Offset"})
        wiz = tree._fit_wizard
        _fake_nav(tree, (0, 0))
        plot.current_data = np.asarray(wiz.signal.data, float)[0, 0]
        wiz.spec["Offset"]["offset"].value = 0.0

        fit_navigated(session, plot, {"adaptive": True})
        # The data sits on a constant 5.0.
        assert wiz.spec["Offset"]["offset"].value == pytest.approx(5.0, abs=0.5)
        assert (0, 0) in tree.fit_store, "an adaptive fit was not remembered"

    def test_navigating_with_no_model_does_nothing(self, window, fitted):
        from spyde.actions.fit_action import fit_navigated
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_navigated(session, plot, {"adaptive": True})   # must not raise


class TestDragSmoothness:
    """A pointer_move must do LESS work than a pointer_up.

    Every move crossing the IPC boundary to re-send the whole model is what
    made dragging feel like it was catching. A move redraws the curve; the
    state message and the partner handle wait for the release.
    """

    def test_a_live_move_does_not_emit_state(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard
        before = len(_messages_of(window, "fit_state"))
        wiz._on_widget_drag("Gaussian", "point", {"x": 30.0}, live=True)
        assert len(_messages_of(window, "fit_state")) == before

    def test_the_release_emits_state(self, window, fitted):
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard
        before = len(_messages_of(window, "fit_state"))
        wiz._on_widget_drag("Gaussian", "point", {"x": 30.0}, live=False)
        assert len(_messages_of(window, "fit_state")) > before

    def test_a_live_move_still_updates_the_model(self, window, fitted):
        """Cheaper, not inert — the curve must still follow the cursor."""
        session, plot, tree, _ = fitted
        fit_open(session, plot, {})
        fit_add_component(session, plot, {"kind": "Gaussian"})
        wiz = tree._fit_wizard
        wiz._on_widget_drag("Gaussian", "point", {"x": 33.0}, live=True)
        assert wiz.spec["Gaussian"]["centre"].value == pytest.approx(33.0)
