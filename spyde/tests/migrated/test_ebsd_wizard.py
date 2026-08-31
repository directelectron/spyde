"""The staged EBSD-Indexing wizard (``spyde/actions/ebsd_action.py``).

Handlers are called directly (``fn(session, plot, payload)``) against a real
Qt-free Session, as ``spyde/actions/README.md`` §7 prescribes, and polled with
``_wait`` because each stage hands off to a worker thread.

Note the toolbar GATE is asserted here too: the action is keyed on the ``EBSD``
signal type, which only exists when kikuchipy is installed, and both filter
paths have to agree or the button renders and then dispatches into nothing.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spyde.actions.ebsd_action import (
    DEFAULTS, EbsdWizard, ebsd_build_dictionary, ebsd_refine, ebsd_run,
)
from spyde.data import ebsd_patterns, ground_truth
from spyde.tests.migrated.conftest import _settle, close_session, make_session


@pytest.fixture(autouse=True)
def _cpu_device(monkeypatch):
    """Force the EBSD device to CPU for this file — WIRING tests, like every
    sibling (test_ebsd_indexing / test_ebsd_refine pin ``device="cpu"`` on
    every call; kernel accuracy is theirs, not this file's).

    This file never pinned a device, so on an Apple-Silicon runner every
    handler resolved ``default_device()`` -> "mps" — the only place in the
    whole migrated suite that touched Metal. Under pytest that means the
    build worker, the band-overlay engine thread, up to 8 dask threads and
    per-test teardown of MPS-resident tensors all churn the device across 7
    Session lifecycles — the multi-threaded Metal profile CLAUDE.md documents
    as fatally racy — and macOS CI died with SIGABRT here when the
    macos-latest image rolled. Accelerator work belongs in a subprocess (the
    test_vector_orientation_gpu.py pattern), not under the pytest harness.

    Overridable: a maintainer reproducing on a Mac sets SPYDE_EBSD_DEVICE=mps.
    """
    if not os.environ.get("SPYDE_EBSD_DEVICE"):
        monkeypatch.setenv("SPYDE_EBSD_DEVICE", "cpu")


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

    session = make_session()
    s = ebsd_patterns(nav=(8, 8), detector=(40, 40))
    session._add_signal(s, source_path=None)
    _settle(session)            # let the selector debounce timers fire
    yield {"session": session, "signal": s, "truth": ground_truth(s),
           "messages": captured_messages,
           "trees": session.signal_trees, "plots": session._plots}
    close_session(session)


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

        # It adopts the projection centre the data records.  The synthetic
        # scan stamps the PC it was rendered with; without this the first
        # overlay is drawn with a guessed geometry and every line is off,
        # which reads as broken indexing.
        assert np.allclose(wiz.pc, ebsd_session["truth"]["pc"])

        # And it announces itself to the caret.
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
    def test_draws_line_segments_and_streams_the_match(self, ebsd_session):
        _s, _p, _t, wiz = _build(ebsd_session)
        ov = wiz.overlay
        assert ov is not None, "no band overlay attached"
        segs, za = ov._offsets_for(2, 3)
        assert segs.ndim == 3 and segs.shape[1:] == (2, 2), \
            "not the (N,2,2) shape anyplotlib add_lines needs"
        assert len(segs) > 0, "the matched orientation drew no bands"
        assert len(za) == 0, "zone axes are off by default"

        # The draw call above also streams the match to the caret.
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

        # Hiding clears both marker groups: set_visible(False) pushes a bare
        # array, not the (segments, points) tuple the overlay normally
        # renders — it has to survive that.
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

        # And the IPF map shows the two grains.  The end-to-end check: the
        # wedge grain must come out a different colour from the drifting
        # background grain. This is also the only test that would notice the
        # orientations being handed to orix transposed — no, it would not;
        # see test_ebsd_bands for that. It notices indexing that lost the
        # grain structure entirely.
        grain_rgb = rgb.astype(float)
        mask = np.asarray(ebsd_session["truth"]["grain2_mask"], bool)
        assert np.linalg.norm(
            grain_rgb[mask].mean(0) - grain_rgb[~mask].mean(0)) > 20, \
            "the two grains came out the same colour"

    def test_refuses_before_the_dictionary_is_built(self, ebsd_session):
        session = ebsd_session["session"]
        ebsd_run(session, _signal_plot(session), {})
        assert any("build the dictionary first" in str(m.get("text", "")).lower()
                   for m in ebsd_session["messages"] if m.get("type") == "error")


class TestNeverMaterialisesTheScan:
    """Memory safety (CLAUDE.md rule #1) — the scan is read in navigation
    BLOCKS and the whole thing is never resident.

    An EBSD pattern is small, which makes "just load it" tempting and wrong:
    at float32 a 512x512 map of 60x60 patterns is 3.8 GB, and a real detector
    is far bigger than 60x60. kikuchipy indexes over a chunked dask array for
    the same reason.

    This is guarded rather than reasoned about, because the failure is silent
    until someone opens a big scan: the guard below raises if anything ever
    computes a dask array the shape of the full dataset.
    """

    NAV, DET = (8, 8), (40, 40)

    @pytest.fixture
    def lazy_session(self, captured_messages, monkeypatch):
        from spyde.backend.session import Session
        import spyde.actions.ebsd_action as ebsd_mod

        monkeypatch.setattr(ebsd_mod, "emit", captured_messages.append)

        session = make_session()
        s = ebsd_patterns(nav=self.NAV, detector=self.DET).as_lazy()
        # Several nav chunks, so a correct implementation never asks for a
        # slice the shape of the whole scan (which on a single-chunk 8-row scan
        # it legitimately would, and the guard could not tell the difference).
        s.data = s.data.rechunk((2, 4, -1, -1))
        session._add_signal(s, source_path=None)
        _settle(session)
        yield {"session": session, "signal": s, "messages": captured_messages,
               "trees": session.signal_trees, "plots": session._plots}
        close_session(session)

    @pytest.fixture
    def no_full_compute(self, monkeypatch):
        """Raise if any dask array the shape of the whole dataset is computed."""
        import dask.array as da

        full = tuple(self.NAV) + tuple(self.DET)
        original = da.Array.compute
        seen: list[tuple] = []

        def guard(self, *args, **kwargs):
            if tuple(self.shape) == full:
                raise AssertionError(
                    f"computed the FULL dataset {full} — the scan must be read "
                    f"in navigation blocks (CLAUDE.md memory-safety rule)")
            seen.append(tuple(self.shape))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(da.Array, "compute", guard)
        return seen

    def test_indexing_reads_the_scan_chunk_by_chunk(self, lazy_session,
                                                    no_full_compute):
        session, plot = lazy_session["session"], _signal_plot(lazy_session["session"])
        tree = plot.signal_tree
        ebsd_build_dictionary(session, plot, {"step_deg": 12.0,
                                              "background": "dynamic"})
        assert _wait(lambda: getattr(tree, "_ebsd_wizard", None) is not None)

        ebsd_run(session, plot, {"keep": 4, "refine": False})
        assert _wait(lambda: getattr(tree, "orientation_map", None) is not None,
                     timeout=180), "indexing never finished"
        assert tree.orientation_map.nav_shape == self.NAV

    def test_the_static_reference_is_streamed_too(self, lazy_session,
                                                  no_full_compute):
        """The static background reference is a whole-scan MEAN — exactly the
        kind of reduction that invites reading everything to compute it."""
        session, plot = lazy_session["session"], _signal_plot(lazy_session["session"])
        tree = plot.signal_tree
        ebsd_build_dictionary(session, plot, {"step_deg": 12.0,
                                              "background": "both"})
        assert _wait(lambda: getattr(tree, "_ebsd_wizard", None) is not None)
        wiz = tree._ebsd_wizard
        assert wiz.static_ref is not None
        assert wiz.static_ref.shape == self.DET

        # And it is the real WHOLE-scan mean, not one block's. The reference
        # comes from a fresh eager copy (the generator is deterministic) rather
        # than from computing the lazy signal — which the guard above would,
        # correctly, refuse.
        expected = np.asarray(ebsd_patterns(nav=self.NAV, detector=self.DET).data,
                              np.float32).mean(axis=(0, 1))
        assert np.allclose(wiz.static_ref, expected, atol=1e-3)

    def test_the_adp_map_is_seamless_across_chunks(self):
        """Everything else in the run is per-pattern, so chunking cannot change
        it. The ADP map is the exception — it compares each position with its
        four NEIGHBOURS, so the rows either side of every chunk boundary would
        be computed against nothing above/below them without the halo that
        ``map_overlap`` adds. That shows up as faint lines at the seams, which
        is easy to mistake for real structure.
        """
        import dask.array as da
        from spyde.actions.ebsd_action import _adp_graph
        from spyde.ebsd import average_dot_product_map

        s = ebsd_patterns(nav=(8, 8), detector=self.DET)
        raw = np.asarray(s.data, np.float32)

        class _Passthrough:            # the halo, not the correction, is the test
            background = "none"

            def correct(self, patterns):
                return np.asarray(patterns, float)

        whole = average_dot_product_map(raw)
        for chunk in (8, 4, 3, 1):     # 1 chunk, exact split, ragged, per-row
            scan = da.from_array(raw, chunks=(chunk, chunk, -1, -1))
            got = _adp_graph(scan, _Passthrough()).compute(scheduler="threads")
            assert got.shape == whole.shape
            assert np.allclose(got, whole, atol=1e-5), \
                f"the ADP map has seams at {chunk}-row chunk boundaries"

    def test_indexing_is_one_lazy_graph_over_the_chunks(self):
        """The index is a dask graph, so nothing is read by BUILDING it — the
        patterns only exist inside a worker, a chunk at a time."""
        import dask.array as da
        from spyde.actions.ebsd_action import _lazy_scan, _packed_index_graph

        s = ebsd_patterns(nav=(8, 8), detector=self.DET).as_lazy()
        scan = _lazy_scan(s)
        assert isinstance(scan, da.Array)

        class _Wiz:
            euler = np.zeros((2, 3))

        graph = _packed_index_graph(scan, _Wiz(), keep=3, refine=False, steps=1)
        assert isinstance(graph, da.Array)
        assert graph.shape == (8, 8, 4 + 3)
        # Nav chunking is carried through, so the result assembles per chunk.
        assert graph.chunks[:2] == scan.chunks[:2]

    def test_eager_data_still_gets_a_bounded_chunking(self):
        """An eager scan has no chunks of its own; it must still be given a
        nav chunking, sized by BYTES, so a bigger detector means fewer rows
        rather than the same rows and more memory.

        _nav_block_rows / _lazy_scan are pure bytes/shape math, so a 16x16
        nav grid proves the same three properties a 64x64 one did (which
        cost ~9 s of pattern rendering).  The budget shrinks with it so BOTH
        row counts stay bytes-derived rather than ny-clamped: at 1 MiB the
        40x40 detector gives 10 rows and the 80x80 gives 2."""
        from spyde.actions.ebsd_action import _lazy_scan, _nav_block_rows
        small = ebsd_patterns(nav=(16, 16), detector=(40, 40))
        big = ebsd_patterns(nav=(16, 16), detector=(80, 80))
        budget = 1 << 20
        assert _nav_block_rows(big, budget) < _nav_block_rows(small, budget)
        assert _nav_block_rows(small, 1 << 40) == 16        # never more than ny
        scan = _lazy_scan(small)
        assert len(scan.chunks[2]) == 1 and len(scan.chunks[3]) == 1, \
            "patterns must not be split across chunks"

    def test_patterns_split_across_chunks_are_merged(self):
        """A chunk holding PART of a pattern makes every read useless. EBSD
        readers do not normally split the detector, but if one does the graph
        has to fix it rather than index garbage."""
        from spyde.actions.ebsd_action import _lazy_scan

        s = ebsd_patterns(nav=(8, 8), detector=(40, 40)).as_lazy()
        s.rechunk(nav_chunks=(4, 4), sig_chunks=(20, 20), inplace=True)
        scan = _lazy_scan(s)
        assert len(scan.chunks[2]) == 1 and len(scan.chunks[3]) == 1

    def test_the_lazy_signals_own_chunking_is_left_alone(self):
        """Two things at once, both load-bearing.

        The nav chunking a lazy signal arrived with is KEPT — reshuffling a
        multi-GB array to a theoretical optimum is the mistake CLAUDE.md's
        live-display notes record (419 s against 184 s), and storage alignment
        beats any after-the-fact rechunk.

        And the signal itself is not repointed: the navigator reads through
        the same array, so preparing the compute must not mutate it.
        """
        from spyde.actions.ebsd_action import _lazy_scan

        s = ebsd_patterns(nav=(8, 8), detector=(40, 40)).as_lazy()
        s.rechunk(nav_chunks=(4, 4), sig_chunks=(20, 20), inplace=True)
        before = s.data.chunks

        scan = _lazy_scan(s)
        assert s.data.chunks == before, "the live signal's chunking was mutated"
        assert scan.chunks[:2] == before[:2], "the nav chunking was reshuffled"


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
