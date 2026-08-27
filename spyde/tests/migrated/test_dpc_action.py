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

    The module does ``from de_shell.ipc import emit`` at import, so conftest's
    patch of ``ipc.emit`` never reaches that binding — the same hazard conftest
    documents for ``session.py``, and the same fix.
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


class _PointerEvent:
    """What anyplotlib hands a widget handler. Only ``event_type`` is read."""

    def __init__(self, event_type):
        self.event_type = event_type


def _drag_frame():
    return _PointerEvent("pointer_move")


def _release():
    return _PointerEvent("pointer_up")


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

    def test_the_drawn_boxes_cover_exactly_the_fitted_pixels(self, window):
        """The overlay is a promise about WHICH scan positions the plane is
        fitted through, so it has to be the same pixels, to the edge.

        ``corner_boxes`` gives pixel INDICES; ``add_rectangles`` takes centres.
        Pixel ``i`` covers ``[i - 0.5, i + 0.5]``, so a block over indices 0..1
        spans ``[-0.5, 1.5]`` and is centred on 0.5 — not on ``x + w/2``, which
        is 1.0. Getting that wrong shifts every box half a pixel toward the
        bottom-right: a visible gap inside the top-left corner, an overhang past
        the bottom-right edge, and boxes that no longer mark what is measured.
        """
        from spyde.actions import dpc as _dpc
        session, plot, _tree, wiz = _opened(window, center_mode="corners")
        drawn = wiz._corner_mg._data
        got = sorted(
            (float(cx) - float(w) / 2, float(cx) + float(w) / 2,
             float(cy) - float(h) / 2, float(cy) + float(h) / 2)
            for (cx, cy), w, h in zip(drawn["offsets"], drawn["widths"],
                                      drawn["heights"]))
        # The fit's own slices, converted to the pixel EDGES they cover.
        want = sorted(
            (cols.start - 0.5, cols.stop - 0.5, rows.start - 0.5, rows.stop - 0.5)
            for rows, cols in _dpc.corner_slices(
                wiz._nav_shape(), float(wiz.params["corner_fraction"])))
        assert got == pytest.approx(want), \
            "the drawn corner boxes do not cover the pixels the plane is fitted through"

    def test_the_box_size_slider_resizes_them_in_place(self, window):
        session, plot, _tree, wiz = _opened(window, center_mode="corners")
        first = wiz._corner_mg
        dpca.dpc_set_center(session, plot, {"center_mode": "corners",
                                            "corner_fraction": 0.25})
        assert wiz._corner_mg is first, "resizing must not rebuild the markers"
        assert wiz.params["corner_fraction"] == 0.25

    def test_switching_mode_takes_the_previous_furniture_away(self, window):
        """Overlays that outlive their mode are the classic version of this bug:
        boxes still on screen describing a reference no longer in use.

        The beam region is deliberately NOT mode-scoped — it defines the centre
        of mass for every reference mode — so it must survive a mode switch that
        clears the corner boxes.
        """
        session, plot, _tree, wiz = _opened(window, center_mode="corners",
                                            beam_shape="circle")
        nav = _navigator_plot(session)
        assert "dpc_corners" in _markers(nav._plot2d)
        assert wiz._beam_selector.roi is not None

        dpca.dpc_set_center(session, plot, {"center_mode": "manual"})
        assert wiz._corner_mg is None
        assert "dpc_corners" not in _markers(nav._plot2d)
        assert wiz._beam_selector.roi is not None, \
            "the beam region is not owned by a Center mode"

        dpca.dpc_set_center(session, plot, {"center_mode": "none"})
        assert wiz._corner_mg is None and wiz._beam_selector.roi is not None

    def test_the_region_centre_becomes_the_manual_reference(self, window):
        """Drag the region onto the beam and Manual is already answered — no
        second marker to place, and no way for the two to disagree."""
        session, plot, _tree, wiz = _opened(window, center_mode="manual",
                                            beam_shape="circle")
        assert wiz._beam_selector.roi is not None
        wiz.params.update({"beam_cx": 20.0, "beam_cy": 12.0, "beam_r": 6.0})
        dpca.dpc_pick_center(session, plot, {})
        assert wiz.params["cx"] == 20.0 and wiz.params["cy"] == 12.0
        ref = wiz.reference()
        # (centre − picked), constant over the scan — the same sign convention
        # get_direct_beam_position uses.
        assert ref[..., 0] == pytest.approx(32 / 2.0 - 20.0)
        assert ref[..., 1] == pytest.approx(32 / 2.0 - 12.0)

    def test_the_reference_follows_the_region_without_an_explicit_pick(self, window):
        """`manual_center` falls back to the region, so a user who never presses
        the button still gets the centre they dragged."""
        session, plot, _tree, wiz = _opened(window, center_mode="manual",
                                            beam_shape="circle")
        wiz.params.update({"beam_cx": 18.0, "beam_cy": 14.0, "beam_r": 5.0})
        assert wiz.manual_center() == (18.0, 14.0)
        assert wiz.reference()[..., 0] == pytest.approx(16.0 - 18.0)

    def test_picking_before_the_region_has_a_radius_errors(self, window):
        """There is no "off" shape any more, so the only way to have no usable
        region is the moment before the detector size has filled the radii in —
        which is what ``BeamRegion.active`` still guards."""
        session, plot, _tree, wiz = _opened(window, center_mode="none")
        wiz.params["beam_r"] = 0.0
        dpca.dpc_pick_center(session, plot, {})
        assert any("beam region" in e for e in _errors(window["messages"]))

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

    def test_the_beam_region_selector_matches_the_shape(self, window):
        """Circle and ring are different anyplotlib widget TYPES, so switching
        rebuilds the SELECTOR rather than mutating it — and the widget on screen
        must be the one the mask is computed from.

        The region is a real ``BaseSelector`` (the same one a virtual image puts
        on this plot), which is what gives it the navigator's latest-wins
        dispatch and settle re-fire instead of a hand-rolled debounce.
        """
        from spyde.drawing.selectors import AnnularSelector, CircleSelector

        session, plot, _tree, wiz = _opened(window, beam_shape="circle")
        assert isinstance(wiz._beam_selector, CircleSelector)
        assert type(wiz._beam_selector.roi).__name__ == "CircleWidget"
        assert wiz.region().shape == "circle" and wiz.region().r > 0

        dpca.dpc_set_beam(session, plot, {"beam_shape": "ring"})
        assert _wait(lambda: isinstance(wiz._beam_selector, AnnularSelector))
        assert type(wiz._beam_selector.roi).__name__ == "AnnularWidget"
        assert wiz.region().shape == "ring"
        assert 0 < wiz.region().r_inner < wiz.region().r

    def test_the_region_is_always_on_the_pattern(self, window):
        """There is no "off". The region IS what the centre of mass is taken
        over, so switching it off only meant taking the whole frame with no
        handle to grab — a control whose useful setting was "not that one"."""
        session, _plot, _tree, wiz = _opened(window)
        assert wiz._beam_selector is not None
        assert wiz.region().active, "the region opened without a usable radius"
        assert "off" not in dpca._dpc.BEAM_SHAPES
        assert dpca.DEFAULTS["beam_shape"] in dpca._dpc.BEAM_SHAPES

    def test_radii_are_filled_in_from_the_detector_size(self, window):
        """They cannot be declared in DEFAULTS — a sensible radius is a fraction
        of a detector whose size is unknown until a dataset is open. Opening the
        wizard is therefore what fills them in, since the region is always on."""
        session, plot, _tree, wiz = _opened(window)
        sy, sx = wiz._sig_shape()
        # The INSCRIBED circle: the region is always on, so its default is what
        # an unattended scan is measured with, and a smaller one clips the beam
        # and under-reads the shift. See dpc.default_beam_region for the numbers.
        assert wiz.params["beam_r"] == pytest.approx(0.5 * min(sy, sx))
        assert (wiz.params["beam_cx"], wiz.params["beam_cy"]) == (sx / 2, sy / 2)

    def test_a_moved_region_re_measures_through_the_selector(self, window):
        """The map tracks the region as it moves, the way a virtual image tracks
        its detector ROI — and for the same reason: the region IS a
        ``BaseSelector``, so every move goes through the navigator's serial
        latest-wins dispatcher and cancels the compute it replaces.

        Waiting for ``pointer_up`` left the map frozen for the whole gesture
        (measured in the real app: zero repaints across a 6.4 s drag).
        """
        session, plot, _tree, wiz = _opened(window, beam_shape="circle")
        passes: list = []
        wiz.measure = lambda **kwargs: passes.append(wiz.region().r)

        # Driven by moving the WIDGET, not by calling the hook: what is under
        # test is that the selector carries a move all the way to a re-measure
        # on its own, which is the whole point of it being a BaseSelector.
        for r in (8.0, 9.0, 10.0, 11.0):         # a drag, frame by frame
            before = len(passes)
            wiz._beam_selector.roi.set(r=r)
            assert _wait(lambda: len(passes) > before, timeout=10.0), \
                f"moving the region to r={r} never re-measured"

        assert passes[-1] == pytest.approx(11.0), \
            "the last pass measured a region the widget had already moved off"
        assert wiz.params["beam_r"] == pytest.approx(11.0), \
            "the widget geometry must track the pointer every frame"

    def test_a_superseded_pass_is_cancelled_not_left_to_finish(self, window):
        """Each move cancels the pass it replaces. A beam-shift pass reads the
        whole scan, so one nobody is waiting for costs the cluster the entire
        dataset — the same contract ``virtual_image`` keeps."""
        session, plot, _tree, wiz = _opened(window, beam_shape="circle")
        first = [False]
        wiz._track_measure(first)
        assert wiz._measure_stop is first

        wiz._beam_selector.roi.set(r=9.0)
        assert _wait(lambda: first[0], timeout=10.0), \
            "moving the region left the previous pass running"
        assert wiz._measure_stop is not first, \
            "the new pass did not take the slot from the one it superseded"


    def test_moving_the_region_does_not_read_a_frame_for_the_brightness(self, window):
        """The brightness readout reads a FRAME — a dask compute on a lazy scan
        — so one per pointer frame queues work faster than it drains, and the
        caret's own radius keeps climbing after the pointer stops. Re-measuring
        has no such problem (it dispatches and returns, and a superseded pass is
        cancelled), which is why the two are not on the same cadence: the
        readout refreshes when a pass LANDS.
        """
        import threading

        import spyde.actions.dpc as dpc_mod
        reads: list = []
        real_brightness = dpc_mod.region_brightness
        dpc_mod.region_brightness = lambda *a, **k: (
            reads.append(threading.current_thread()) or real_brightness(*a, **k))
        try:
            session, plot, _tree, wiz = _opened(window, beam_shape="circle")
            wiz.measure = lambda **kwargs: None      # not what this test is about
            reads.clear()
            for r in (8.0, 9.0, 10.0, 11.0):
                wiz._beam_selector.roi.set(r=r)
                wiz._on_region_moved()
            assert reads == [], \
                "moving the region read a frame for the brightness readout"

            wiz.emit_region(with_brightness=True)    # what a landed pass does
            assert _wait(lambda: len(reads) >= 1, timeout=10.0), \
                "a landed pass never refreshed the brightness"
            assert reads[0] is not threading.current_thread(), \
                "the brightness probe ran on the caller's thread, stalling the drag"
        finally:
            dpc_mod.region_brightness = real_brightness

    def test_a_measure_abandoned_by_close_does_not_shout(self, window):
        """Closing the caret mid-pass fails the measure on the way down (the
        executor it submits into is gone). Reporting that tells the user
        "locating the direct beam failed" for something they did on purpose and
        that has no consequence."""
        session, plot, _tree, wiz = _opened(window)
        gen = wiz.guard()
        wiz._measure_failed(gen, RuntimeError("still live — should report"))
        assert any("locating the direct beam failed" in e
                   for e in _errors(window["messages"]))

        before = len(_errors(window["messages"]))
        wiz.remove()
        wiz._measure_failed(gen, RuntimeError("cannot schedule new futures"))
        assert len(_errors(window["messages"])) == before, \
            "a measure abandoned by close reported an error to the user"

    def test_a_superseded_measure_does_not_shout(self, window):
        """Same for a run replaced by a newer one — nobody is waiting for it."""
        session, plot, _tree, wiz = _opened(window)
        stale = wiz.guard()
        wiz.guard()                      # a newer run supersedes it
        before = len(_errors(window["messages"]))
        wiz._measure_failed(stale, RuntimeError("boom"))
        assert len(_errors(window["messages"])) == before

    def test_the_settle_re_fire_cannot_fire_after_teardown(self, window):
        """A timer that survives close would re-measure a torn-down wizard on a
        worker thread. The timer belongs to the SELECTOR now, so closing the
        wizard has to close the selector — which is the one thing that could
        quietly stop happening when the region stopped being a raw widget."""
        session, plot, _tree, wiz = _opened(window, beam_shape="circle")
        selector = wiz._beam_selector
        selector.update_data()                       # arms the settle re-fire
        assert _wait(lambda: selector._settle_timer is not None, timeout=5.0)
        dpca.dpc_close(session, plot, {})
        assert selector._settle_timer is None and wiz._closed
        assert wiz._beam_selector is None

    def test_the_region_is_echoed_back_for_the_caret(self, window):
        session, plot, _tree, wiz = _opened(window, beam_shape="circle")
        wiz._on_region_moved()
        msgs = _of_type(window["messages"], "dpc_region")
        assert msgs, "no dpc_region echo reached the caret"
        last = msgs[-1]
        assert last["shape"] == "circle"
        assert last["window_id"] == getattr(plot, "window_id", None), \
            "must address the SOURCE window — useWizardEvent filters on it"
        assert last["r"] == pytest.approx(wiz.params["beam_r"])

    def test_a_region_off_the_beam_is_reported(self, window):
        """A region on empty detector still returns a plausible centroid, so the
        only way the user learns is if we say so.

        The warning keys on BRIGHTNESS, not on whether the found beam is inside
        the region — a disc always contains its own centroid, so that test is
        vacuous for the commonest shape (see ``dpc.beam_inside_region``).
        """
        session, plot, _tree, wiz = _opened(window, beam_shape="circle")
        # A corner of the frame: the synthetic beam sits at the centre, so this
        # region is on the disc's faint tail and nothing else.
        dpca.dpc_set_beam(session, plot, {"beam_shape": "circle", "beam_cx": 2.0,
                                          "beam_cy": 2.0, "beam_r": 1.5})
        assert _wait(lambda: any("not on the direct beam" in str(m.get("text", ""))
                                 for m in window["messages"]
                                 if isinstance(m, dict)
                                 and m.get("type") in ("status", "error"))), \
            [m.get("text") for m in window["messages"]
             if isinstance(m, dict) and m.get("type") in ("status", "error")][-4:]
        # Close before finishing. `dpc_action.emit` is re-bound to the NEXT
        # test's capture list by the `_capture_module_emit` fixture, so a
        # measure still in flight here would deliver THIS test's deliberate
        # warning into that one's messages — which is exactly what made
        # `test_a_well_placed_region_is_not_warned_about` fail, but only in
        # some orderings.
        dpca.dpc_close(session, plot, {})

    def test_a_well_placed_region_is_not_warned_about(self, window):
        """The warning has to be quiet when things are right, or it is noise."""
        session, plot, _tree = _dataset(window)
        # Count only what THIS wizard emits. A late emit from a previous test's
        # session would otherwise be indistinguishable from a warning about
        # this region (see the note in the test above).
        start = len(window["messages"])
        dpca.dpc_open(session, plot, {"beam_shape": "circle"})
        wiz = _wait(lambda: getattr(plot.signal_tree, "_dpc_wizard", None)
                    is not None) and plot.signal_tree._dpc_wizard
        assert _wait(lambda: wiz.shifts is not None)
        assert not any("not on the direct beam" in str(m.get("text", ""))
                       for m in window["messages"][start:] if isinstance(m, dict))

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

    def test_a_lazy_scan_fills_in_progressively(self, window):
        """On lazy data the pass streams per nav chunk and the map repaints as
        each lands — the difference between a filling field and a spinner on a
        scan that takes minutes.

        Asserted on the PROGRESS stream, which reports what each landing chunk
        triggered. NOT on how much of the field was finite when a repaint ran:
        the repaints are marshalled to the event loop and the pass now finishes
        in milliseconds, so every one of them can legitimately see a complete
        field and still have been driven by a separate chunk. That is the
        compute being fast, not the stream having collapsed — measuring it that
        way made this test fail on a 37x speedup.
        """
        import dask.array as da
        import hyperspy.api as hs
        session = window["window"]
        nav, sig, chunk = 16, 24, 8
        gy, gx = np.mgrid[0:sig, 0:sig].astype(np.float32)
        frame = 1000.0 / (1 + np.exp((np.hypot(gx - 14, gy - 10) - 5) / 1.5))
        arr = da.from_array(np.tile(frame, (nav, nav, 1, 1)),
                            chunks=(chunk, chunk, sig, sig))
        s = hs.signals.Signal2D(arr).as_lazy()
        s.set_signal_type("electron_diffraction")
        session._add_signal(s)
        assert _wait(lambda: _signal_plot(session) is not None)
        plot = _signal_plot(session)

        seen_partial = []
        real_refresh = dpca.DpcWizard.refresh

        def spy(self):
            if self.shifts is not None:
                finite = int(np.isfinite(self.shifts).all(axis=-1).sum())
                seen_partial.append(finite)
            return real_refresh(self)

        total_positions = nav * nav
        dpca.DpcWizard.refresh = spy
        try:
            dpca.dpc_open(session, plot, {})
            _wait(lambda: getattr(plot.signal_tree, "_dpc_wizard", None)
                  is not None)
            # Wait on the recorded PAINT, not on `wiz.shifts`: `_finish` assigns
            # the shifts and only THEN calls `refresh`, so a wait on the array
            # can return between the two — leaving the last entry in
            # `seen_partial` a partial paint, and restoring the spy in `finally`
            # before the full one was ever recorded.
            assert _wait(lambda: seen_partial
                         and seen_partial[-1] == total_positions,
                         timeout=60.0), \
                f"the streamed pass never completed: {seen_partial}"
        finally:
            dpca.DpcWizard.refresh = real_refresh

        assert seen_partial, "the map was never repainted"
        assert seen_partial[-1] == total_positions, \
            f"the streamed pass left holes in the field: {seen_partial}"

        # The whole claim: the map was repainted MORE THAN ONCE, once per nav
        # chunk that landed, rather than a single paint at the end.
        chunks = (nav // chunk) ** 2
        streamed = [m for m in window["messages"]
                    if isinstance(m, dict) and m.get("type") == "progress"
                    and "DPC" in str(m.get("label", ""))
                    and m.get("total") == chunks]
        assert len(streamed) >= 2, \
            f"the pass did not stream — one paint at the end: {streamed}"
        assert any(0 < m["done"] < chunks for m in streamed), \
            f"no chunk reported partial progress: {streamed}"
        assert streamed[0]["total"] == chunks, \
            "progress should count NAV CHUNKS, matching the storage layout"

    def test_an_eager_scan_does_not_pretend_to_stream(self, window):
        """Nothing to stream when the data is already in RAM — it runs in one
        pass rather than through the chunk dispatcher."""
        session, plot, _tree, wiz = _opened(window)
        assert not getattr(wiz.signal, "_lazy", False)
        assert wiz.shifts is not None and np.isfinite(wiz.shifts).all()

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
        # Shown ALWAYS, not on hover: a direction map's hues mean nothing
        # without its key, so hiding the key until the pointer arrives hides
        # what makes the picture readable.
        assert d["hover_only"] is False, "the direction key must always be up"
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
class TestMeasureIsolation:
    """A measure runs hyperspy ``map`` on a worker thread, and two measures
    overlap whenever a second open lands before the first pass finishes. They
    must not share a signal object — hyperspy mutates the one it is called on
    (``dpc.private_view``)."""

    def _measured_signals(self, window, opens=1):
        """``(tree, seen)``, each ``seen`` entry the ``(signal, data)`` handed to
        the worker, recorded AT CALL TIME.

        The buffer is recorded in the spy rather than read back afterwards. The
        wait below only catches the measure STARTING, so a later read races a
        pass that is still running and rebinding ``.data`` as it goes — which
        made this assertion fail on macOS and Windows CI while passing
        everywhere else. What the contract is about is the object handed over,
        and that is knowable exactly once: at the call.
        """
        session, plot, tree = _dataset(window)
        seen = []
        real = dpca._dpc.measure_beam_shifts

        def spy(signal, **kw):
            seen.append((signal, signal.data))
            return real(signal, **kw)

        dpca._dpc.measure_beam_shifts = spy
        try:
            for _ in range(opens):
                dpca.dpc_open(session, plot, {})
                if opens > 1:
                    dpca.dpc_close(session, plot, {})
            _wait(lambda: len(seen) >= opens, timeout=30.0)
        finally:
            dpca._dpc.measure_beam_shifts = real
        return tree, seen

    def test_the_worker_never_gets_the_trees_own_signal(self, window):
        tree, seen = self._measured_signals(window)
        assert seen, "the measure never ran"
        handed, handed_data = seen[0]
        live = dpca._current_signal(_signal_plot(window["window"]))
        assert handed is not live, \
            "the worker was handed the tree's live signal — two passes would race"
        assert handed_data is live.data, "the view must share the data buffer"

    def test_every_pass_gets_its_own_object_and_they_cannot_overlap(self, window):
        """Two guarantees, and the second is what makes the first sufficient.

        Each pass is handed its own view, so two of them cannot read each
        other's state. And they are set up on a SINGLE-thread lane, so two
        set-ups never run at once — which is the stronger property, because
        ``private_view`` is itself the operation that is unsafe to overlap
        (hyperspy parks a length-1 placeholder on ``.data`` while it copies).

        Passes are run sequentially here rather than raced: a superseded pass is
        now cancelled before the lane ever picks it up, so open/close/open no
        longer produces two calls to compare.
        """
        session, plot, _tree, wiz = _opened(window)
        seen = []
        real = dpca._dpc.measure_beam_shifts

        def spy(signal, **kw):
            seen.append(signal)
            return real(signal, **kw)

        dpca._dpc.measure_beam_shifts = spy
        try:
            for _ in range(3):
                before = len(seen)
                wiz.measure()
                assert _wait(lambda: len(seen) > before, timeout=30.0), \
                    "a measure never reached the worker"
        finally:
            dpca._dpc.measure_beam_shifts = real

        live = dpca._current_signal(_signal_plot(session))
        assert all(s is not live for s in seen), \
            "a pass was handed the tree's live signal — two would race"
        assert len({id(s) for s in seen}) == len(seen), \
            "two measures shared one signal object"
        assert wiz._pass_lane()._max_workers == 1, \
            "the pass lane must be serial — private_view cannot overlap itself"


@pytest.mark.usefixtures("_capture_module_emit")
class TestDoubleFire:
    def test_open_close_open_leaves_exactly_one_wizard(self, window):
        """React StrictMode fires the three synchronously. Whether the first
        measure lands before or after the close is a race, and both orderings
        must leave exactly one live wizard and one open window: land late and
        the generation guard stops the window ever opening; land early and it
        opens, then teardown closes it."""
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
        # Count what is still OPEN, not what was ever emitted: a window the
        # early-landing measure opened is matched by a window_closed from
        # teardown, and counting opens alone reads that as a leak.
        opened = {m["window_id"] for m in _of_type(window["messages"], "figure")
                  if str(m.get("title", "")).startswith("DPC")}
        closed = {m["window_id"] for m in _of_type(window["messages"],
                                                   "window_closed")}
        assert len(opened - closed) == 1, (
            f"{len(opened - closed)} DPC windows left open "
            f"(opened {sorted(opened)}, closed {sorted(closed)})")

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
