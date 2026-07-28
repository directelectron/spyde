"""The staged EBSD-Indexing wizard (``spyde/actions/ebsd_action.py``).

Handlers are called directly (``fn(session, plot, payload)``) against a real
Qt-free Session, as ``spyde/actions/README.md`` §7 prescribes, and polled with
``_wait`` because each stage hands off to a worker thread.

Note the toolbar GATE is asserted here too: the action is keyed on the ``EBSD``
signal type, which only exists when kikuchipy is installed, and both filter
paths have to agree or the button renders and then dispatches into nothing.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spyde.actions.ebsd_action import (
    DEFAULTS, EbsdWizard, ebsd_build_dictionary, ebsd_refine, ebsd_run,
)
from spyde.data import ebsd_patterns, ground_truth


def _wait(pred, timeout=120.0, interval=0.05):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def ebsd_session(captured_messages, monkeypatch):
    """A Session holding the bundled synthetic EBSD scan.

    Small on purpose (8x8 of 40x40) — the wizard is what is under test, and a
    real-size dictionary would make every test a minute long. Accuracy is
    covered by ``test_ebsd_indexing`` / ``test_ebsd_refine``.
    """
    from spyde.backend.session import Session
    import spyde.actions.ebsd_action as ebsd_mod

    # ebsd_action binds `emit` at import (`from ...ipc import emit`), so
    # captured_messages' patch of ipc.emit doesn't reach its direct calls —
    # the same reason conftest patches session.emit separately. emit_status /
    # emit_error resolve `emit` as a module global inside ipc, so those are
    # already covered.
    monkeypatch.setattr(ebsd_mod, "emit", captured_messages.append)

    session = Session(n_workers=1, threads_per_worker=1)
    s = ebsd_patterns(nav=(8, 8), detector=(40, 40))
    session._add_signal(s, source_path=None)
    time.sleep(0.8)             # let the selector debounce timers fire
    yield {"session": session, "signal": s, "truth": ground_truth(s),
           "messages": captured_messages,
           "trees": session.signal_trees, "plots": session._plots}
    session.shutdown()


def _signal_plot(session):
    """The pattern (non-navigator) plot — the one the caret sits on."""
    for plot in session._plots:
        if not getattr(plot, "is_navigator", False):
            return plot
    return session._plots[0]


def _build(ctx, **over):
    """Run stage 2 and wait for the wizard to exist."""
    session, plot = ctx["session"], _signal_plot(ctx["session"])
    tree = plot.signal_tree
    payload = {"step_deg": 12.0, "background": "dynamic",
               "background_sigma": 6.0, "n_bands": 8}
    payload.update(over)
    ebsd_build_dictionary(session, plot, payload)
    assert _wait(lambda: getattr(tree, "_ebsd_wizard", None) is not None), \
        "the dictionary never built"
    return session, plot, tree, tree._ebsd_wizard


class TestBuildDictionary:
    def test_builds_a_wizard_with_a_resident_dictionary(self, ebsd_session):
        _s, _p, tree, wiz = _build(ebsd_session)
        assert isinstance(wiz, EbsdWizard)
        assert len(wiz.indexer) > 10
        assert len(wiz.euler) == len(wiz.indexer)
        assert wiz.detector == (40, 40)

    def test_adopts_the_projection_centre_the_data_records(self, ebsd_session):
        """The synthetic scan stamps the PC it was rendered with. Without this
        the first overlay is drawn with a guessed geometry and every line is
        off, which reads as broken indexing."""
        _s, _p, _t, wiz = _build(ebsd_session)
        assert np.allclose(wiz.pc, ebsd_session["truth"]["pc"])

    def test_announces_itself_to_the_caret(self, ebsd_session):
        _build(ebsd_session)
        msgs = [m for m in ebsd_session["messages"]
                if m.get("type") == "ebsd_dictionary_ready"]
        assert msgs, "no ebsd_dictionary_ready — the caret stays locked"
        assert msgs[-1]["n_orientations"] > 10
        assert len(msgs[-1]["pc"]) == 3

    def test_the_dictionary_is_filtered_like_the_data(self, ebsd_session):
        """Both sides of a cross-correlation must go through the SAME filter.
        High-passing only the experimental patterns leaves the dictionary
        carrying low frequencies its counterpart no longer has — scores stay
        mediocre and it looks like bad indexing (see test_ebsd_indexing)."""
        _s, _p, _t, wiz = _build(ebsd_session, background="dynamic",
                                 background_sigma=6.0)
        assert wiz.sim_sigma == 6.0
        from spyde.ebsd.bands import simulate_patterns
        # A raw simulated pattern carries a big DC term; the dictionary entry
        # for the same orientation has been high-passed and no longer does.
        raw = simulate_patterns(wiz.euler[0], wiz.reflectors,
                                wiz.detector, wiz.pc)[0]
        assert raw.mean() > 0.1
        _e, score = wiz.indexer.best(wiz.correct(
            np.asarray(ebsd_session["signal"].data[0, 0], float)))
        assert score > 0.5, f"live match scored only {score:.3f}"

    def test_no_background_means_no_simulated_filter(self, ebsd_session):
        _s, _p, _t, wiz = _build(ebsd_session, background="none")
        assert wiz.sim_sigma is None

    def test_static_only_does_not_filter_the_dictionary(self, ebsd_session):
        """`static` subtracts a DETECTOR artefact, which simulated patterns do
        not have — applying it to them would be a different image, not the
        same correction."""
        _s, _p, _t, wiz = _build(ebsd_session, background="static")
        assert wiz.sim_sigma is None
        assert wiz.static_ref is not None

    def test_rebuilding_replaces_the_previous_wizard(self, ebsd_session):
        """Otherwise a second Build stacks a second overlay on the pattern and
        both redraw on every navigator move."""
        _s, _p, tree, first = _build(ebsd_session)
        session, plot = ebsd_session["session"], _signal_plot(ebsd_session["session"])
        ebsd_build_dictionary(session, plot, {"step_deg": 15.0})
        assert _wait(lambda: getattr(tree, "_ebsd_wizard", None) is not first)
        assert first._closed and first.overlay is None


class TestBandOverlay:
    def test_draws_line_segments_for_the_matched_orientation(self, ebsd_session):
        _s, _p, _t, wiz = _build(ebsd_session)
        ov = wiz.overlay
        assert ov is not None, "no band overlay attached"
        segs, za = ov._offsets_for(2, 3)
        assert segs.ndim == 3 and segs.shape[1:] == (2, 2), \
            "not the (N,2,2) shape anyplotlib add_lines needs"
        assert len(segs) > 0, "the matched orientation drew no bands"
        assert len(za) == 0, "zone axes are off by default"

    def test_streams_the_match_to_the_caret(self, ebsd_session):
        _s, _p, _t, wiz = _build(ebsd_session)
        wiz.overlay._offsets_for(2, 3)
        hits = [m for m in ebsd_session["messages"] if m.get("type") == "ebsd_match"]
        assert hits and hits[-1]["ok"]
        assert 0.0 <= hits[-1]["score"] <= 1.0
        assert set(hits[-1]) >= {"phi1", "Phi", "phi2", "score"}

    def test_a_different_position_gives_different_bands(self, ebsd_session):
        """The two grains have genuinely different orientations, so an overlay
        that ignored the navigator would be caught here."""
        _s, _p, _t, wiz = _build(ebsd_session)
        mask = np.asarray(ebsd_session["truth"]["grain2_mask"], bool)
        ys, xs = np.nonzero(mask)
        ys2, xs2 = np.nonzero(~mask)
        a, _ = wiz.overlay._offsets_for(int(ys[0]), int(xs[0]))
        b, _ = wiz.overlay._offsets_for(int(ys2[0]), int(xs2[0]))
        assert a.shape != b.shape or not np.allclose(a, b)

    def test_hiding_clears_both_marker_groups(self, ebsd_session):
        """set_visible(False) pushes a bare array, not the (segments, points)
        tuple the overlay normally renders — it has to survive that."""
        _s, _p, _t, wiz = _build(ebsd_session)
        wiz.overlay.set_visible(False)
        assert wiz.overlay._hidden
        wiz.overlay.set_visible(True)
        assert not wiz.overlay._hidden


class TestRefineStage:
    def test_band_count_and_zone_axes_apply_live(self, ebsd_session):
        session, plot, _t, wiz = _build(ebsd_session)
        ebsd_refine(session, plot, {"n_bands": 3, "show_zone_axes": True})
        assert _wait(lambda: wiz.overlay.n_bands == 3
                     and wiz.overlay.show_zone_axes)
        segs, za = wiz.overlay._offsets_for(2, 3)
        assert len(segs) <= 3
        assert za.ndim == 2 and za.shape[1] == 2

    def test_moving_the_projection_centre_moves_the_lines(self, ebsd_session):
        """The PC is the one parameter you can only set by looking — nudging it
        has to redraw, or the Refine tab does nothing."""
        session, plot, _t, wiz = _build(ebsd_session)
        before, _ = wiz.overlay._offsets_for(2, 3)
        ebsd_refine(session, plot, {"pc_x": 0.62})
        assert _wait(lambda: abs(wiz.overlay.pc[0] - 0.62) < 1e-9)
        after, _ = wiz.overlay._offsets_for(2, 3)
        assert before.shape != after.shape or not np.allclose(before, after)
        assert abs(wiz.pc[0] - 0.62) < 1e-9

    def test_is_a_no_op_before_the_dictionary_exists(self, ebsd_session):
        session = ebsd_session["session"]
        ebsd_refine(session, _signal_plot(session), {"n_bands": 3})   # must not raise


class TestRunStage:
    def test_indexes_the_scan_into_an_ipf_window(self, ebsd_session):
        session, plot, tree, _wiz = _build(ebsd_session)
        before = len(session.signal_trees)
        ebsd_run(session, plot, {"keep": 4, "refine": False})
        assert _wait(lambda: getattr(tree, "orientation_map", None) is not None,
                     timeout=180), "no orientation map attached"
        om = tree.orientation_map
        assert om.nav_shape == (8, 8)
        assert om.quats.shape == (8, 8, 1, 4)
        rgb = om.ipf_color_map("z")
        assert rgb.shape == (8, 8, 3) and rgb.dtype == np.uint8
        assert len(session.signal_trees) > before, "no result window opened"

    def test_the_ipf_map_shows_the_two_grains(self, ebsd_session):
        """The end-to-end check: the wedge grain must come out a different
        colour from the drifting background grain. This is also the only test
        that would notice the orientations being handed to orix transposed —
        no, it would not; see test_ebsd_bands for that. It notices indexing
        that lost the grain structure entirely."""
        session, plot, tree, _wiz = _build(ebsd_session)
        ebsd_run(session, plot, {"keep": 4, "refine": False})
        assert _wait(lambda: getattr(tree, "orientation_map", None) is not None,
                     timeout=180)
        rgb = tree.orientation_map.ipf_color_map("z").astype(float)
        mask = np.asarray(ebsd_session["truth"]["grain2_mask"], bool)
        assert np.linalg.norm(rgb[mask].mean(0) - rgb[~mask].mean(0)) > 20, \
            "the two grains came out the same colour"

    def test_refuses_before_the_dictionary_is_built(self, ebsd_session):
        session = ebsd_session["session"]
        ebsd_run(session, _signal_plot(session), {})
        assert any("build the dictionary first" in str(m.get("text", "")).lower()
                   for m in ebsd_session["messages"] if m.get("type") == "error")


class TestWiring:
    def test_the_schema_is_registered_and_matches_the_defaults(self):
        """One source of truth for every host — a schema default that drifts
        from the handler default means the caret sends one thing and a
        notebook another."""
        from spyde.actions.registry import wizard_parameters
        schema = wizard_parameters("ebsd")
        assert schema, "the ebsd wizard has no registered schema"
        for key, spec in schema.items():
            if key in DEFAULTS:
                assert spec["default"] == DEFAULTS[key], \
                    f"{key}: schema default {spec['default']!r} != handler " \
                    f"default {DEFAULTS[key]!r}"

    def test_every_stage_resolves(self):
        from spyde.actions.registry import resolve_staged
        for name in ("ebsd_build_dictionary", "ebsd_refine", "ebsd_run"):
            assert callable(resolve_staged(name)), f"{name} is not registered"

    def test_the_toolbar_entry_is_gated_on_the_ebsd_signal_type(self):
        """Both filter paths apply the same gates in two places; a gate added
        to one alone renders a button that never dispatches, or vice versa."""
        import spyde
        meta = None
        for group in spyde.TOOLBAR_ACTIONS.values():
            if isinstance(group, dict) and "EBSD Indexing" in group:
                meta = group["EBSD Indexing"]
        assert meta is not None, "EBSD Indexing is not in toolbars.yaml"
        assert meta["signal_types"] == ["EBSD"]
        assert meta["function"].endswith("ebsd_action.ebsd_indexing")

    def test_the_overlay_toggle_reaches_the_wizard(self, ebsd_session):
        """The caret shows/hides the overlay through Session._set_overlay,
        which resolves it by ACTION NAME — a name mismatch silently no-ops."""
        session, plot, _t, wiz = _build(ebsd_session)
        session._set_overlay(plot, "EBSD Indexing", False)
        assert wiz.overlay._hidden
        session._set_overlay(plot, "EBSD Indexing", True)
        assert not wiz.overlay._hidden
