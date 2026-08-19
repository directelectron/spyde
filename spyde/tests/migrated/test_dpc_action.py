"""
The DPC wizard backend (``dpc_*`` staged handlers).

Handlers are called directly as ``fn(session, plot, payload)`` and polled with
``_wait``, the shape ``test_find_vectors_wizard.py`` establishes — the beam-shift
pass runs on a worker thread.

The physics lives in ``test_dpc.py``. What this suite covers is the *wiring*,
where a green unit suite proves nothing:

:class:`TestOpen`
    One expensive measure, one result window, and a ``dpc_state`` message that
    tells the caret whether the Center step is needed at all.
:class:`TestCentering`
    Each reference mode puts its OWN furniture on the right window — corner
    boxes on the navigator (they select scan positions), the crosshair on the
    diffraction pattern (it selects a detector position) — and takes the other
    mode's furniture away. Overlays left behind on a mode switch are the classic
    version of this bug.
:class:`TestLive`
    Rotation, view and field-mode changes must be pure re-derivation. If any of
    them re-measured, the slider would be unusable on a real scan — so the test
    counts measures, not milliseconds.
:class:`TestCommit`
    Every component becomes a real child node, not just a picture.
:class:`TestTeardown`
    README §6: the result window is a bare ``figure``, so it must be reachable
    through ``controller_by_window_id`` and must actually disappear.
:class:`TestDoubleFire`
    README §4 / StrictMode: open, close, open leaves exactly ONE wizard and ONE
    window.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from spyde.actions import dpc_action as dpca


@pytest.fixture
def _capture_module_emit(window, monkeypatch):
    """Route ``dpc_action``'s own ``emit`` into the captured list.

    The module does ``from spyde.backend.ipc import emit`` at import, so
    conftest's patch of ``ipc.emit`` never reaches that binding — the same hazard
    conftest documents for ``session.py``, and the same fix.
    """
    monkeypatch.setattr(dpca, "emit", window["messages"].append)


def _wait(pred, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _navigator_plot(session):
    return next((p for p in session._plots if p.is_navigator), None)


def _dataset(window, **payload):
    session = window["window"]
    session._load_test_data_dpc({"nav": 16, "sig": 32, **payload})
    assert _wait(lambda: _signal_plot(session) is not None), \
        "the DPC fixture never produced a signal plot"
    plot = _signal_plot(session)
    return session, plot, plot.signal_tree


def _opened(window, **params):
    session, plot, tree = _dataset(window)
    dpca.dpc_open(session, plot, dict(params))
    assert _wait(lambda: getattr(tree, "_dpc_wizard", None) is not None
                 and tree._dpc_wizard.shifts is not None
                 and tree._dpc_wizard.window_id is not None), \
        "the DPC result window never opened"
    return session, plot, tree, tree._dpc_wizard


def _of_type(messages, kind):
    return [m for m in messages if isinstance(m, dict) and m.get("type") == kind]


def _errors(messages):
    """The error strings. ``emit_error`` puts the text under ``text``, not
    ``message`` — reading the wrong key makes every "did it complain?" assertion
    silently vacuous."""
    return [str(m.get("text", "")) for m in _of_type(messages, "error")]


def _markers(plot2d):
    """Marker-group NAMES on a plot (``list_markers`` returns descriptor dicts)."""
    try:
        return [m.get("name") if isinstance(m, dict) else m
                for m in plot2d.list_markers()]
    except Exception:
        return []


@pytest.mark.usefixtures("_capture_module_emit")
class TestOpen:
    def test_opening_measures_once_and_opens_a_window(self, window):
        session, _plot, _tree, wiz = _opened(window)
        assert wiz.shifts.shape == (16, 16, 2)
        figs = _of_type(window["messages"], "figure")
        assert any(f.get("window_id") == wiz.window_id for f in figs), \
            "no figure message for the DPC result window"
        assert session.controller_by_window_id(wiz.window_id) is wiz

    def test_state_reports_the_descan_so_the_step_can_be_skipped(self, window):
        """The fixture bakes in a known offset AND ramp, so ``centered`` must be
        False and both numbers must be recognisable. A caret that cannot tell
        the difference makes the user apply a correction blind."""
        _s, _p, _t, _w = _opened(window)
        states = _of_type(window["messages"], "dpc_state")
        assert states, "no dpc_state message reached the caret"
        c = states[-1]["centering"]
        assert c is not None and c["centered"] is False
        assert c["offset"][0] == pytest.approx(1.5, abs=0.3)
        assert c["offset"][1] == pytest.approx(-1.0, abs=0.3)
        assert c["worst"] > c["tol_px"]

    def test_an_already_centered_scan_says_so(self, window):
        session, plot, _tree = _dataset(window, offset_x=0.0, offset_y=0.0,
                                        ramp_x=0.0, ramp_y=0.0, amplitude=0.0)
        dpca.dpc_open(session, plot, {})
        assert _wait(lambda: _of_type(window["messages"], "dpc_state"))
        c = _of_type(window["messages"], "dpc_state")[-1]["centering"]
        assert c["centered"] is True, \
            f"a scan with no descan should not need centering (worst={c['worst']})"

    def test_only_real_4d_scans_are_offered_as_vacuum_candidates(self, window):
        """A vacuum reference needs a beam position at every scan point, so only
        a 2-D scan over a 2-D detector qualifies.

        The list used to be every open tree, which offered this action's own
        committed result maps as "vacuum scans" — a choice that was never valid
        and produced a failed measure when taken.
        """
        import hyperspy.api as hs
        session, plot, tree = _dataset(window)
        good = hs.signals.Signal2D(np.zeros((4, 4, 8, 8), np.float32))
        good.metadata.General.title = "Vacuum scan"
        session._add_signal(good)
        flat = hs.signals.Signal2D(np.zeros((8, 8), np.float32))
        flat.metadata.General.title = "A committed map"
        session._add_signal(flat)

        dpca.dpc_open(session, plot, {})
        assert _wait(lambda: _of_type(window["messages"], "dpc_state"))
        titles = [d["title"] for d in
                  _of_type(window["messages"], "dpc_state")[-1]["datasets"]]
        assert any(t.startswith("Vacuum scan") for t in titles), titles
        assert not any(t.startswith("A committed map") for t in titles), titles
        # The shape disambiguates near-duplicate titles (a sample scan and its
        # vacuum scan usually share a name).
        assert any("(4×4)" in t for t in titles), titles

    def test_the_candidate_list_refreshes_when_the_mode_changes(self, window):
        """A vacuum scan opened AFTER the caret mounted must still be offered —
        a list captured once shows an empty picker with no way to refresh it."""
        import hyperspy.api as hs
        session, plot, _tree, _wiz = _opened(window)
        assert not _of_type(window["messages"], "dpc_state")[-1]["datasets"]
        later = hs.signals.Signal2D(np.zeros((4, 4, 8, 8), np.float32))
        later.metadata.General.title = "Opened later"
        session._add_signal(later)
        dpca.dpc_set_center(session, plot, {"center_mode": "vacuum"})
        titles = [d["title"] for d in
                  _of_type(window["messages"], "dpc_state")[-1]["datasets"]]
        assert any(t.startswith("Opened later") for t in titles), titles

    def test_a_non_4d_dataset_is_refused_with_a_reason(self, window):
        """DPC needs a 2-D scan. Say so rather than failing somewhere downstream
        with a shape error the user cannot act on."""
        session = window["window"]
        import hyperspy.api as hs
        s = hs.signals.Signal2D(np.zeros((6, 8, 8), np.float32))
        s.set_signal_type("electron_diffraction")
        session._add_signal(s)
        assert _wait(lambda: _signal_plot(session) is not None)
        dpca.dpc_open(session, _signal_plot(session), {})
        errors = _errors(window["messages"])
        assert any("navigation dimension" in e for e in errors), errors


@pytest.mark.usefixtures("_capture_module_emit")
class TestCentering:
    def test_corner_boxes_land_on_the_navigator(self, window):
        """They select SCAN positions, so the navigator is the only window they
        can mean anything on."""
        session, plot, _tree, wiz = _opened(window, center_mode="corners")
        nav = _navigator_plot(session)
        assert nav is not None
        assert wiz._corner_mg is not None
        assert "dpc_corners" in _markers(nav._plot2d)
        assert "dpc_corners" not in _markers(plot._plot2d), \
            "the corner boxes belong on the navigator, not the pattern"

    def test_the_box_size_slider_resizes_them_in_place(self, window):
        session, plot, _tree, wiz = _opened(window, center_mode="corners")
        first = wiz._corner_mg
        dpca.dpc_set_center(session, plot, {"center_mode": "corners",
                                            "corner_fraction": 0.25})
        assert wiz._corner_mg is first, "resizing must not rebuild the markers"
        assert wiz.params["corner_fraction"] == 0.25

    def test_switching_mode_takes_the_previous_furniture_away(self, window):
        """Overlays that outlive their mode are the classic version of this bug:
        boxes still on screen describing a reference no longer in use."""
        session, plot, _tree, wiz = _opened(window, center_mode="corners")
        nav = _navigator_plot(session)
        assert "dpc_corners" in _markers(nav._plot2d)

        dpca.dpc_set_center(session, plot, {"center_mode": "manual"})
        assert wiz._corner_mg is None
        assert "dpc_corners" not in _markers(nav._plot2d)
        assert wiz._cross is not None, "Manual mode must offer a crosshair"

        dpca.dpc_set_center(session, plot, {"center_mode": "none"})
        assert wiz._cross is None and wiz._corner_mg is None

    def test_picking_the_crosshair_sets_a_constant_reference(self, window):
        session, plot, _tree, wiz = _opened(window, center_mode="manual")
        assert wiz._cross is not None
        wiz._cross.cx, wiz._cross.cy = 20.0, 12.0
        dpca.dpc_pick_center(session, plot, {})
        assert wiz.params["cx"] == 20.0 and wiz.params["cy"] == 12.0
        ref = wiz.reference()
        # (centre − picked), constant over the scan — the same sign convention
        # get_direct_beam_position uses.
        assert ref[..., 0] == pytest.approx(32 / 2.0 - 20.0)
        assert ref[..., 1] == pytest.approx(32 / 2.0 - 12.0)

    def test_picking_without_a_crosshair_errors_instead_of_guessing(self, window):
        session, plot, _tree, wiz = _opened(window, center_mode="none")
        dpca.dpc_pick_center(session, plot, {})
        assert any("crosshair" in e for e in _errors(window["messages"]))

    def test_a_vacuum_dataset_becomes_the_reference(self, window):
        """A second scan with no field is pure descan, so it IS the reference."""
        session, plot, tree, wiz = _opened(window)
        session._load_test_data_dpc({"nav": 16, "sig": 32, "amplitude": 0.0})
        assert _wait(lambda: len(session.signal_trees) > 1)
        index = len(session.signal_trees) - 1
        assert session.signal_trees[index] is not tree

        dpca.dpc_load_vacuum(session, plot, {"tree_index": index})
        assert _wait(lambda: wiz.vacuum_shifts is not None), \
            "the vacuum scan was never measured"
        assert wiz.params["center_mode"] == "vacuum"
        ref = wiz.reference()
        assert ref is not None and ref.shape == wiz.shifts.shape
        # Both scans carry the same descan, so the reference must reproduce it.
        assert np.abs(ref - wiz.vacuum_shifts).max() < 0.5

    def test_vacuum_before_a_dataset_is_picked_is_not_an_error(self, window):
        """Sitting on Vacuum with nothing chosen yet is mid-interaction. The
        window must keep rendering (uncorrected) rather than blanking or
        raising — the same reason `reference()` is non-strict."""
        session, plot, _tree, wiz = _opened(window)
        dpca.dpc_set_center(session, plot, {"center_mode": "vacuum"})
        assert wiz.reference() is None
        assert wiz.result is not None
        assert not _errors(window["messages"])

    def test_the_corner_boxes_are_the_pixels_that_get_fitted(self, window):
        """The overlay is a claim about the fit. Check the claim, on the live
        wizard, not just on the pure function."""
        from spyde.actions import dpc as _dpc
        _s, _p, _t, wiz = _opened(window, center_mode="corners",
                                  corner_fraction=0.2)
        boxes = _dpc.corner_boxes(wiz._nav_shape(), 0.2)
        mask = _dpc.corner_mask(wiz._nav_shape(), 0.2)
        drawn = np.ones(wiz._nav_shape(), bool)
        for (x, y, w, h) in boxes:
            drawn[int(y):int(y + h), int(x):int(x + w)] = False
        assert np.array_equal(drawn, mask)


@pytest.mark.usefixtures("_capture_module_emit")
class TestLive:
    def test_tuning_the_rotation_never_re_measures(self, window):
        """The one architectural claim of this wizard: measure once, tune
        forever. A re-measure hidden in ``dpc_tune`` would be invisible on a
        16x16 fixture and fatal on a real scan, so count calls rather than
        trusting the timing."""
        from spyde.actions import dpc as _dpc
        session, plot, _tree, wiz = _opened(window)
        calls = {"n": 0}
        real = _dpc.measure_beam_shifts

        def counted(*a, **kw):
            calls["n"] += 1
            return real(*a, **kw)

        import spyde.actions.dpc as dpc_mod
        dpc_mod.measure_beam_shifts = counted
        try:
            for angle in (10.0, 45.0, 200.0):
                dpca.dpc_tune(session, plot, {"rotation": angle})
            dpca.dpc_set_view(session, plot, {"view": "divergence"})
            dpca.dpc_tune(session, plot, {"flip": True})
        finally:
            dpc_mod.measure_beam_shifts = real
        assert calls["n"] == 0, "a live parameter change re-measured the dataset"
        assert wiz.params["rotation"] == 200.0 and wiz.params["flip"] is True

    def test_re_measure_is_the_only_thing_that_re_measures(self, window):
        session, plot, _tree, wiz = _opened(window)
        first = wiz.shifts.copy()
        dpca.dpc_run(session, plot, {"method": "center_of_mass",
                                     "half_square_width": 8})
        assert _wait(lambda: wiz.params["half_square_width"] == 8)
        assert wiz.shifts is not None and wiz.shifts.shape == first.shape

    def test_solving_the_rotation_reports_its_own_confidence(self, window):
        """The fixture's field is curl-free, so ``mode="electric"`` should find
        the baked-in 25° and say the residual collapsed."""
        session, plot, _tree, wiz = _opened(window, center_mode="corners",
                                            corner_fraction=0.125,
                                            mode="electric")
        dpca.dpc_auto_rotation(session, plot, {})
        assert _wait(lambda: _of_type(window["messages"], "dpc_estimate"))
        est = _of_type(window["messages"], "dpc_estimate")[-1]
        err = min(abs((est["angle"] - 25.0) % 180.0),
                  180.0 - abs((est["angle"] - 25.0) % 180.0))
        assert err < 3.0, f"solved {est['angle']}°, truth 25°"
        assert est["improvement"] > 5.0
        assert wiz.params["rotation"] == est["angle"]

    def test_the_wheel_is_a_hover_KEY_not_an_inset(self, window):
        """The legend is a `Plot2D.add_key` overlay — the same primitive as the
        IPF colour triangle and the scale bar.

        It was an ``add_inset`` first, which is a floating window with a title
        bar and its own canvas stack: it read as a panel sitting ON the map
        rather than as part of the figure, and its picture had to be re-pushed.
        A key floats in screen space with no chrome, appears on hover, and is
        still baked into a PNG export.
        """
        session, plot, _tree, wiz = _opened(window)
        assert wiz.wheel is not None, "no colour-wheel legend was attached"
        assert "dpc_wheel" in [k.name for k in wiz.plot.list_keys()]
        assert wiz.plot.get_key("dpc_wheel") is wiz.wheel
        d = wiz.wheel.to_dict()
        assert d["hover_only"] is True, "the legend must stay out of the way"
        assert d["visible"] is True
        assert not hasattr(wiz.wheel, "imshow"), \
            "the wheel is a KeyOverlay, not a plot in an inset"

    def test_the_wheel_hides_for_a_scalar_view(self, window):
        """A hue legend left over a divergence map describes something that is
        not on screen."""
        session, plot, _tree, wiz = _opened(window)
        dpca.dpc_set_view(session, plot, {"view": "divergence"})
        assert wiz.wheel.to_dict()["visible"] is False
        dpca.dpc_set_view(session, plot, {"view": "rgb"})
        assert wiz.wheel.to_dict()["visible"] is True

    def test_every_view_paints_without_error(self, window):
        from spyde.actions import dpc_display
        session, plot, _tree, wiz = _opened(window)
        for view in dpc_display.VIEWS:
            dpca.dpc_set_view(session, plot, {"view": view})
            assert wiz.params["view"] == view
        assert not _errors(window["messages"])

    def test_an_unknown_view_is_ignored(self, window):
        session, plot, _tree, wiz = _opened(window)
        dpca.dpc_set_view(session, plot, {"view": "sideways"})
        assert wiz.params["view"] == "rgb"

    def test_switching_field_mode_drops_a_stale_rotation_estimate(self, window):
        """The two modes assert different symmetries, so an estimate carried
        across would describe the wrong physics while looking authoritative."""
        session, plot, _tree, wiz = _opened(window, mode="electric")
        dpca.dpc_auto_rotation(session, plot, {})
        assert _wait(lambda: wiz.estimate is not None)
        dpca.dpc_tune(session, plot, {"mode": "magnetic"})
        assert wiz.estimate is None

    def test_result_messages_carry_the_units(self, window):
        session, plot, _tree, _wiz = _opened(window)
        dpca.dpc_tune(session, plot, {"rotation": 30.0})
        results = _of_type(window["messages"], "dpc_result")
        assert results and results[-1]["units"] in ("px", "mrad", "MV/cm")
        assert results[-1]["rotation"] == 30.0

    def test_transport_keys_never_reach_the_parameters(self, window):
        """``window_id`` rides on every staged message; letting it into params
        would put transport plumbing into the committed provenance."""
        session, plot, _tree, wiz = _opened(window)
        dpca.dpc_tune(session, plot, {"rotation": 12.0, "window_id": 999,
                                      "nonsense": True})
        assert "window_id" not in wiz.params and "nonsense" not in wiz.params
        assert wiz.params["rotation"] == 12.0


@pytest.mark.usefixtures("_capture_module_emit")
class TestCommit:
    def test_every_component_becomes_a_real_child_node(self, window):
        """A committed tree must carry the DATA, not a picture of it — the same
        lesson the Strain commit learned (a saved tree that held only εxx)."""
        from spyde.actions import dpc as _dpc
        session, plot, _tree, wiz = _opened(window)
        before = len(session.signal_trees)
        dpca.dpc_commit(session, plot, {})
        assert len(session.signal_trees) == before + 1
        new = session.signal_trees[-1]
        names = {n.name for n in _nodes(new)}
        titles = _dpc.component_titles(wiz.result.mode, wiz.result.units)
        for comp in _dpc.COMPONENTS:
            assert titles[comp] in names, f"{comp} is missing from the tree"

    def test_the_rgb_primary_is_not_labelled_as_a_component(self, window):
        """The primary map is the direction+magnitude IMAGE. Labelling it "Ex"
        put a chip beside the real "Ex (MV/cm)" view claiming to be the same
        thing — two chips, one name, different data."""
        from spyde.actions import dpc as _dpc
        session, plot, _tree, wiz = _opened(window)
        dpca.dpc_commit(session, plot, {})
        titles = _dpc.component_titles(wiz.result.mode, wiz.result.units)
        chips = [m for m in _of_type(window["messages"], "view_figure")]
        labels = [c.get("label") for c in chips]
        assert len(labels) == len(set(labels)), f"duplicate view chips: {labels}"
        assert titles["fx"] not in {"Ex", "Bx"}, \
            "the component title should carry its units"

    def test_provenance_records_the_orientation(self, window):
        """Rotation, handedness and reverse are the parameters a reader most
        needs to reproduce (or distrust) a DPC figure."""
        session, plot, _tree, _wiz = _opened(window)
        dpca.dpc_tune(session, plot, {"rotation": 42.0, "flip": True,
                                      "reverse": True})
        dpca.dpc_commit(session, plot, {})
        prov = session.signal_trees[-1]._commit_provenance
        assert prov["action"] == "DPC"
        assert prov["params"]["rotation"] == 42.0
        assert prov["params"]["flip"] is True
        assert prov["params"]["reverse"] is True
        assert "units" in prov["params"]

    def test_committing_nothing_errors(self, window):
        session, plot, _tree = _dataset(window)
        dpca.dpc_commit(session, plot, {})
        assert any("DPC" in e for e in _errors(window["messages"]))


def _nodes(tree):
    """Every SignalNode in *tree* (``children`` is a dict, so iterate values)."""
    stack, out = [tree.root_node], []
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend((getattr(node, "children", None) or {}).values())
    return out


@pytest.mark.usefixtures("_capture_module_emit")
class TestTeardown:
    def test_closing_removes_the_window_and_the_overlays(self, window):
        session, plot, tree, wiz = _opened(window, center_mode="corners")
        wid = wiz.window_id
        nav = _navigator_plot(session)
        assert "dpc_corners" in _markers(nav._plot2d)

        dpca.dpc_close(session, plot, {})
        assert wiz._closed
        assert getattr(tree, "_dpc_wizard", None) is None
        assert session.controller_by_window_id(wid) is None
        assert "dpc_corners" not in _markers(nav._plot2d)
        assert any(m.get("type") == "window_closed" and m.get("window_id") == wid
                   for m in window["messages"])

    def test_closing_twice_is_a_no_op(self, window):
        session, plot, _tree, _wiz = _opened(window)
        dpca.dpc_close(session, plot, {})
        dpca.dpc_close(session, plot, {})     # must not raise

    def test_forgetting_the_window_tears_the_wizard_down(self, window):
        """README §6 — the window can go away for reasons the caret never sees
        (the user closes it), and ``_forget_window`` must reach ``close()``."""
        session, _plot, tree, wiz = _opened(window)
        session._forget_window(wiz.window_id)
        assert wiz._closed and getattr(tree, "_dpc_wizard", None) is None


@pytest.mark.usefixtures("_capture_module_emit")
class TestDoubleFire:
    def test_open_close_open_leaves_exactly_one_wizard(self, window):
        """React StrictMode fires the three synchronously, before the first
        measure lands — so the idempotence check cannot see the in-flight call
        and the generation guard has to."""
        session, plot, tree = _dataset(window)
        dpca.dpc_open(session, plot, {})
        dpca.dpc_close(session, plot, {})
        dpca.dpc_open(session, plot, {})
        assert _wait(lambda: getattr(tree, "_dpc_wizard", None) is not None
                     and tree._dpc_wizard.window_id is not None)
        time.sleep(0.5)          # let any superseded measure land and be dropped
        wizards = [t for t in session.signal_trees
                   if getattr(t, "_dpc_wizard", None) is not None]
        assert len(wizards) == 1
        dpc_windows = {m["window_id"] for m in _of_type(window["messages"], "figure")
                       if str(m.get("title", "")).startswith("DPC")}
        assert len(dpc_windows) == 1, f"{len(dpc_windows)} DPC windows opened"

    def test_re_opening_a_live_wizard_does_not_build_a_second(self, window):
        session, plot, tree, wiz = _opened(window)
        dpca.dpc_open(session, plot, {"rotation": 15.0})
        assert tree._dpc_wizard is wiz
        assert wiz.params["rotation"] == 15.0


class TestToolbarGating:
    """The button must be offered on a diffraction pattern and nowhere else.

    Both filter paths matter (README §6): ``get_toolbar_actions_for_plot``
    resolves the function, ``_action_matches_plot`` backs
    ``get_toolbar_config_for_plot`` and imports nothing. A gate added to only
    one renders a button that never dispatches, or vice versa.
    """

    @staticmethod
    def _signal(signal_type: str):
        import hyperspy.api as hs
        s = hs.signals.Signal2D(np.zeros((4, 4, 8, 8), dtype=np.float32))
        s.set_signal_type(signal_type)
        return s

    @staticmethod
    def _plot(signal, *, is_navigator=False):
        """The `_FakePlot` shape `test_vector_vvi_action` uses — the filters
        reach back from the PlotState to `plot_state.plot.is_navigator`."""
        import types
        plot = types.SimpleNamespace(
            signal_tree=types.SimpleNamespace(diffraction_vectors=None),
            is_navigator=is_navigator)
        plot.plot_state = types.SimpleNamespace(
            current_signal=signal, dimensions=2, plot=plot)
        return plot

    @pytest.mark.parametrize("signal_type,offered", [
        ("electron_diffraction", True),
        ("spyde_diffraction_vectors_image", False),   # a vectors RESULT window
        ("insitu", False),                            # a movie, not a 4D scan
    ])
    def test_offered_only_on_dense_diffraction(self, signal_type, offered):
        from spyde.drawing.toolbars.plot_control_toolbar import (
            get_toolbar_actions_for_plot,
        )
        plot = self._plot(self._signal(signal_type))
        names = get_toolbar_actions_for_plot(plot.plot_state)[2]
        assert ("DPC" in names) is offered, sorted(names)

    def test_both_filter_paths_agree(self):
        """One gate, two enforcement sites — they must give the same answer."""
        from spyde.drawing.toolbars.plot_control_toolbar import (
            _action_matches_plot, get_toolbar_actions_for_plot,
        )
        import spyde
        spec = next(group["DPC"] for group in spyde.TOOLBAR_ACTIONS.values()
                    if isinstance(group, dict) and "DPC" in group)
        for signal_type in ("electron_diffraction",
                            "spyde_diffraction_vectors_image", "insitu"):
            plot = self._plot(self._signal(signal_type))
            resolved = "DPC" in get_toolbar_actions_for_plot(plot.plot_state)[2]
            matched = _action_matches_plot("DPC", spec, plot.plot_state)
            assert resolved == matched, (
                f"the two toolbar filters disagree about DPC on "
                f"{signal_type}: resolved={resolved} matched={matched}")
