"""An overlay belongs to the node the window displays, and reads it the way the
base pattern is read.

Centre the diffraction pattern, then Find Vectors: the circles must land on the
disks of the CENTRED frame the window shows. Before this contract, Find Vectors
computed on the tree root and its circles sat where the root's disks were,
2 px off the pattern on screen. The same pin covers the live preview's frame
read on a lazy mapped node, which used to compute the whole dask block per
navigator move instead of the node's recipe on the parent's resident block.
"""
from __future__ import annotations

import time

import dask.array as da
import numpy as np
import hyperspy.api as hs

from spyde.tests.migrated.conftest import _settle
from spyde.tests.migrated.test_center_zero_beam import (
    _off_center_4d, _signal_plot, _wait,
)

FIND_VECTORS_PARAMS = {"sigma": 1.0, "kernel_radius": 5, "threshold": 0.4,
                       "min_distance": 3, "subpixel": True}
BEAM = (18, 14)          # column, row of the un-centred disk
CENTRE = (16.0, 16.0)    # where centring puts it on a 32x32 frame


def _off_centre_lazy(nav=(8, 8), sig=(32, 32), beam=BEAM, chunk=4):
    """Like ``_off_center_4d`` but lazy, in ``chunk`` x ``chunk`` navigation
    blocks, so a mapped child has a real block to avoid computing."""
    yy, xx = np.mgrid[0:sig[0], 0:sig[1]]
    disk = ((xx - beam[0]) ** 2 + (yy - beam[1]) ** 2 <= 9).astype(np.float32)
    data = np.zeros(nav + sig, dtype=np.float32)
    for idx in np.ndindex(*nav):
        data[idx] = disk * 100.0
    signal = hs.signals.Signal2D(data).as_lazy()
    signal.data = signal.data.rechunk((chunk, chunk, -1, -1))
    signal.set_signal_type("electron_diffraction")
    return signal


def _centre_of_mass(frame):
    frame = np.asarray(frame, dtype=np.float64)
    yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    total = frame.sum()
    return float((xx * frame).sum() / total), float((yy * frame).sum() / total)


def _centre(session, src):
    """Run Center Zero Beam on ``src`` and return the centred node's signal."""
    from spyde.actions.center_zero_beam import czb_run
    before = src.plot_state.current_signal
    czb_run(session, src, {"method": "center_of_mass"})
    assert _wait(lambda: src.plot_state.current_signal is not before, 30), \
        "centering never produced a new signal"
    centred = src.plot_state.current_signal
    assert _wait(lambda: (src.current_data is not None
                          and np.allclose(_centre_of_mass(src.current_data), CENTRE, atol=0.5)),
                 20), "the window never painted the centred frame"
    return centred


def _find_vectors(session, src, tree):
    """Dispatch Find Diffraction Vectors and wait for the source overlay AND the
    vectors to attach (the attach gap: CLAUDE.md, Testing)."""
    session._dispatch_toolbar_action(src, "Find Diffraction Vectors", dict(FIND_VECTORS_PARAMS))
    assert _wait(lambda: getattr(tree, "_vector_overlay", None) is not None, 60), \
        "overlay never attached to the source tree"
    result_tree = session.signal_trees[-1]
    assert _wait(lambda: getattr(result_tree, "diffraction_vectors", None) is not None, 60), \
        "vectors never attached"
    return result_tree


def _overlay_offsets(overlay, iy, ix):
    overlay._on_indices(np.array([[ix, iy]]))
    return np.asarray(overlay._mg._data["offsets"], dtype=np.float64)


class TestCentreThenFindVectors:
    def _assert_circles_on_displayed_frame(self, session, src):
        tree = src.signal_tree
        centred = _centre(session, src)
        result_tree = _find_vectors(session, src, tree)
        overlay = tree._vector_overlay

        offsets = _overlay_offsets(overlay, 0, 0)
        assert len(offsets) == 1, offsets
        shown = _centre_of_mass(src.current_data)
        assert np.allclose(offsets[0], shown, atol=0.5), (offsets[0], shown)
        # And not where the root's disk is.
        assert not np.allclose(offsets[0], BEAM, atol=0.5)

        provenance = getattr(result_tree, "_commit_provenance", None) or {}
        assert str(provenance.get("source_node", "")).startswith("Centered"), provenance
        assert overlay.signal is centred

    def test_circles_land_on_the_displayed_frame(self):
        from spyde.backend.session import Session
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_off_center_4d(beam=BEAM))
            _settle(session)
            self._assert_circles_on_displayed_frame(session, _signal_plot(session))
        finally:
            session.shutdown()

    def test_circles_land_on_the_displayed_frame_lazy(self):
        from spyde.backend.session import Session
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_off_centre_lazy())
            _settle(session)
            self._assert_circles_on_displayed_frame(session, _signal_plot(session))
        finally:
            session.shutdown()

    def test_overlay_draws_only_while_its_node_is_displayed(self):
        from spyde.backend.session import Session
        from spyde.actions.lifecycle import show_tree_node
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_off_center_4d(beam=BEAM))
            _settle(session)
            src = _signal_plot(session)
            tree = src.signal_tree
            root = tree.root
            centred = _centre(session, src)
            _find_vectors(session, src, tree)
            overlay = tree._vector_overlay
            assert len(_overlay_offsets(overlay, 0, 0)) == 1

            show_tree_node(src, tree, root)
            assert _wait(lambda: src.plot_state.current_signal is root, 10)
            assert len(_overlay_offsets(overlay, 0, 0)) == 0, \
                "an overlay for the centred node was drawn over the root"

            show_tree_node(src, tree, centred)
            assert _wait(lambda: src.plot_state.current_signal is centred, 10)
            assert len(_overlay_offsets(overlay, 0, 0)) == 1
        finally:
            session.shutdown()


class TestOverlayReadsThroughThePlottingPath:
    def test_frame_at_matches_the_block_and_uses_the_plots_reader(self):
        from spyde.backend.session import Session
        from spyde.actions.vector_overlay import frame_at
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_off_centre_lazy())
            _settle(session)
            src = _signal_plot(session)
            centred = _centre(session, src)

            frame = frame_at(src, centred, 1, 2)
            expected = np.asarray(centred.data[1, 2].compute())
            assert np.array_equal(frame, expected)
            reader = src._local_transform_readers.get(id(centred))
            assert reader is not None and type(reader).__name__ == "RecipeReader", reader
        finally:
            session.shutdown()

    def test_preview_never_computes_a_block_on_a_warm_chunk(self):
        """Five moves inside one navigation chunk of a centred lazy scan: the
        preview's frame reads must be the recipe on the resident block, never a
        dask compute of anything larger than one frame."""
        from spyde.backend.session import Session
        from spyde.actions.find_vectors_action import fv_open
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_off_centre_lazy())
            _settle(session)
            src = _signal_plot(session)
            tree = src.signal_tree
            centred = _centre(session, src)

            fv_open(session, src, dict(FIND_VECTORS_PARAMS, method="dog"))
            assert _wait(lambda: getattr(tree, "_fv_preview", None) is not None, 30)
            preview = tree._fv_preview
            assert preview.signal is centred
            preview._on_indices(np.array([[1, 1]]))
            preview._offsets_for(1, 1)          # warm the chunk

            nav_dim = centred.axes_manager.navigation_dimension
            block_computes = []
            original = da.Array.compute

            def counting_compute(self, *args, **kwargs):
                if int(np.prod(self.shape[:nav_dim])) > 1:
                    block_computes.append(self.shape)
                return original(self, *args, **kwargs)

            da.Array.compute = counting_compute
            try:
                for iy, ix in [(1, 1), (2, 2), (1, 2), (2, 1), (1, 1)]:
                    preview._offsets_for(iy, ix)
            finally:
                da.Array.compute = original
            assert block_computes == [], block_computes
        finally:
            session.shutdown()
