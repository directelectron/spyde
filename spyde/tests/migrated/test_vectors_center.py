"""
Center Zero Beam on a VECTORS tree: the beam is the brightest vector in the
box at each position, a plane through it across the scan, every vector moved
by centre minus plane — no frame touched. The caret is the same one the
frames use; on a vectors window its Run centres the vectors into a new tree.
"""
from __future__ import annotations

import numpy as np

from spyde.signals.diffraction_vectors import (
    COL_INTENSITY, COL_KX, COL_KY, N_COLS, SpyDEDiffractionVectors,
)


def _vectors(ny=4, nx=6, drift=(0.3, -0.2), beam=(1.5, -1.0), scale=0.5):
    """A scan whose beam sits off centre and drifts linearly across it: the
    beam vector (bright) plus three weaker reflections at fixed offsets from
    it. Units are the vectors' own (a 0.5/px axis over 64 px, centred at 0)."""
    rows, offsets = [], [0]
    truth = np.empty((ny, nx, 2))
    for iy in range(ny):
        for ix in range(nx):
            bx = beam[0] + drift[0] * ix
            by = beam[1] + drift[1] * iy
            truth[iy, ix] = (bx, by)
            rows.append([ix, iy, bx, by, -1.0, 100.0])
            for dx, dy in ((5.0, 0.0), (0.0, 5.0), (-3.5, -3.5)):
                rows.append([ix, iy, bx + dx, by + dy, -1.0, 20.0])
            offsets.append(len(rows))
    flat = np.asarray(rows, dtype=np.float32).reshape(-1, N_COLS)
    off = np.asarray(offsets, dtype=np.int64)
    vecs = SpyDEDiffractionVectors(
        flat_buffer=flat, nav_offsets=[np.arange(ny + 1) * nx, off],
        nav_shape=(ny, nx), full_nav_shape=(ny, nx), sig_shape=(64, 64),
        sig_axes=None, kernel_radius_px=2.0, kernel_radius_data=1.0, offsets=off)
    return vecs, truth


class TestCentringTheVectors:
    def test_the_beam_is_the_brightest_vector_in_the_box(self):
        from spyde.actions.vectors_center import beam_per_position
        vecs, truth = _vectors()
        beam = beam_per_position(vecs, centre_xy=(0.0, 0.0), half_width=4.0)
        assert np.allclose(beam, truth, atol=1e-5)

    def test_a_box_that_misses_gives_nan(self):
        from spyde.actions.vectors_center import beam_per_position
        vecs, _ = _vectors()
        beam = beam_per_position(vecs, centre_xy=(30.0, 30.0), half_width=1.0)
        assert np.isnan(beam).all()

    def test_the_plane_recovers_a_linear_drift(self):
        from spyde.actions.vectors_center import fit_plane
        ny, nx = 4, 6
        iy, ix = np.mgrid[0:ny, 0:nx]
        field = 1.5 + 0.3 * ix - 0.2 * iy
        noisy = field.copy()
        noisy[1, 2] = np.nan
        assert np.allclose(fit_plane(noisy), field, atol=1e-9)

    def test_centred_vectors_put_the_beam_on_the_centre(self):
        from spyde.actions.vectors_center import center_vectors
        vecs, truth = _vectors()
        centred, shift, beam = center_vectors(vecs, (0.0, 0.0), 4.0, plane=True)
        assert np.allclose(shift, -truth, atol=1e-4), "the plane is the drift"
        for iy in range(4):
            for ix in range(6):
                rows = np.asarray(centred.at(iy, ix))
                brightest = rows[np.argmax(rows[:, COL_INTENSITY])]
                assert abs(brightest[COL_KX]) < 1e-3 and abs(brightest[COL_KY]) < 1e-3
                # The reflections moved with the beam, so the pattern is intact.
                assert np.allclose(sorted(rows[:, COL_KX] - brightest[COL_KX]),
                                   sorted([0.0, 5.0, 0.0, -3.5]), atol=1e-3)
        # The source is untouched.
        assert np.allclose(np.asarray(vecs.at(0, 0))[0, COL_KX:COL_KY + 1], truth[0, 0])
        assert centred.nav_shape == vecs.nav_shape
        assert np.asarray(centred.count_map()).sum() == np.asarray(vecs.count_map()).sum()

    def test_without_the_plane_each_position_keeps_its_own_beam(self):
        from spyde.actions.vectors_center import center_vectors
        vecs, truth = _vectors()
        # Knock one position's beam a little off the plane.
        vecs.flat_buffer[0, COL_KX] += 0.4
        centred, shift, beam = center_vectors(vecs, (0.0, 0.0), 4.0, plane=False)
        rows = np.asarray(centred.at(0, 0))
        brightest = rows[np.argmax(rows[:, COL_INTENSITY])]
        assert abs(brightest[COL_KX]) < 1e-3, "its own beam, not the plane's"


class TestTheCaretOnAVectorsWindow:
    def test_run_makes_a_centred_vectors_tree(self):
        from spyde.tests.migrated.conftest import _settle, close_session, make_session
        from spyde.tests.migrated._async import wait_until
        from spyde.actions.center_zero_beam import czb_run
        from spyde.actions.find_vectors_action import build_vectors_result_tree
        session = make_session()
        try:
            vecs, truth = _vectors()
            # Axis records so the vectors tree can calibrate its window: 64 px
            # at 0.5 per px, centred — the units the vectors were made in.
            import types
            axis = lambda name: types.SimpleNamespace(  # noqa: E731
                scale=0.5, offset=-16.0, units="1/nm", name=name, size=64)
            vecs.sig_axes = [axis("kx"), axis("ky")]
            vecs.nav_axes = [types.SimpleNamespace(scale=1.0, offset=0.0, units="nm", name="x", size=6),
                             types.SimpleNamespace(scale=1.0, offset=0.0, units="nm", name="y", size=4)]
            tree = build_vectors_result_tree(session, vecs, title="Scan — Vectors")
            _settle(session)
            plot = next(p for p in session._plots
                        if not p.is_navigator and getattr(p, "signal_tree", None) is tree
                        and p.plot_state is not None)
            n_before = len(session.signal_trees)
            czb_run(session, plot, {"method": "center_of_mass", "half_square_width": 8,
                                    "make_flat_field": True})
            assert wait_until(lambda: len(session.signal_trees) > n_before, 30), \
                "no centred vectors tree appeared"
            assert wait_until(lambda: getattr(session.signal_trees[-1],
                                              "diffraction_vectors", None) is not None, 30)
            centred = session.signal_trees[-1].diffraction_vectors
            assert "Centered" in session.signal_trees[-1].root.metadata.get_item("General.title")
            rows = np.asarray(centred.at(2, 3))
            brightest = rows[np.argmax(rows[:, COL_INTENSITY])]
            assert abs(brightest[COL_KX]) < 1e-2 and abs(brightest[COL_KY]) < 1e-2
            assert tree.diffraction_vectors is vecs, "the source tree keeps its vectors"
        finally:
            close_session(session)
