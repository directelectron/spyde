"""
Center Zero Beam on a VECTORS tree: the beam is the brightest vector in the
box at each position, a plane through it across the scan, every vector moved
by centre minus plane — no frame touched. The caret is the same one the
frames use; on a vectors window its Run centres the vectors into a new tree.
"""
from __future__ import annotations

import numpy as np

from spyde.signals.diffraction_vectors import (
    COL_INTENSITY, COL_KX, COL_KY, LEGACY_N_COLS, SpyDEDiffractionVectors,
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
    flat = np.asarray(rows, dtype=np.float32).reshape(-1, LEGACY_N_COLS)
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


class TestCornersOnly:
    def test_the_corner_plane_ignores_the_middle_of_the_scan(self):
        """On a 4 x 6 scan the 2-px corner floor makes the corners columns
        0-1 and 4-5; a beam knocked off in columns 2-3 must not move the
        plane."""
        from spyde.actions.vectors_center import centring_shifts
        from spyde.signals.diffraction_vectors import COL_NAV_X
        vecs, truth = _vectors()
        middle = np.isin(vecs.flat_buffer[:, COL_NAV_X], (2, 3))
        bright = vecs.flat_buffer[:, COL_INTENSITY] == 100.0
        vecs.flat_buffer[middle & bright, COL_KX] += 2.0
        whole_scan, _beam = centring_shifts(vecs, (0.0, 0.0), 4.0, plane=True)
        corners, _beam = centring_shifts(vecs, (0.0, 0.0), 4.0, plane=True,
                                         corner_fraction=0.05)
        assert np.abs(whole_scan + truth).max() > 0.5
        assert np.allclose(corners, -truth, atol=1e-4)


class TestTheOriginMovesToTheBeam:
    def test_the_beam_reads_zero_on_an_axis_that_starts_at_zero(self):
        """Vectors found on an uncentered detector axis (offset 0) center on the
        middle pixel, and that pixel becomes the origin: the beam reads (0, 0)
        while every vector stays on the pixel it was on."""
        import types
        from spyde.actions.vectors_center import center_vectors, with_origin_at
        vecs, truth = _vectors()
        middle = (16.0, 16.0)       # 0.5 per px over 64 px, starting at 0
        vecs.flat_buffer[:, COL_KX] += middle[0]
        vecs.flat_buffer[:, COL_KY] += middle[1]
        axes = [types.SimpleNamespace(scale=0.5, offset=0.0, units="1/nm", name=n, size=64)
                for n in ("kx", "ky")]
        centered, _, _ = center_vectors(vecs, middle, 4.0, plane=True)
        centered = with_origin_at(centered, middle, axes)

        rows = np.asarray(centered.at(2, 3))
        brightest = rows[np.argmax(rows[:, COL_INTENSITY])]
        assert abs(brightest[COL_KX]) < 1e-3 and abs(brightest[COL_KY]) < 1e-3
        assert [a.offset for a in centered.sig_axes] == [-16.0, -16.0]
        # The beam is still drawn on the middle pixel.
        pixel = (brightest[COL_KX] - centered.sig_axes[0].offset) / centered.sig_axes[0].scale
        assert abs(pixel - 32.0) < 1e-2
        # The source's axis records are not touched.
        assert [a.offset for a in axes] == [0.0, 0.0]


class TestFindVectorsCentersOnTheBeam:
    """Find Vectors moves its result so the zero beam is the origin, unless a
    beam stop hides the beam — then the brightest vector is a Bragg spot."""

    @staticmethod
    def _found_on_an_uncentered_detector():
        import types
        vecs, truth = _vectors()
        vecs.flat_buffer[:, COL_KX] += 16.0     # the middle of a 0.5/px, 64 px axis
        vecs.flat_buffer[:, COL_KY] += 16.0
        vecs.sig_axes = [types.SimpleNamespace(scale=0.5, offset=0.0, units="1/nm",
                                               name=n, size=64) for n in ("kx", "ky")]
        return vecs

    def test_the_beam_becomes_the_origin(self):
        from spyde.actions.find_vectors_action import DEFAULTS, _auto_center
        vecs = self._found_on_an_uncentered_detector()
        result, note = _auto_center(None, vecs, dict(DEFAULTS))
        rows = np.asarray(result.at(2, 3))
        brightest = rows[np.argmax(rows[:, COL_INTENSITY])]
        assert abs(brightest[COL_KX]) < 1e-3 and abs(brightest[COL_KY]) < 1e-3
        assert [axis.offset for axis in result.sig_axes] == [-16.0, -16.0]
        assert note.startswith("Zero beam centered")

    def test_unchecked_leaves_the_vectors_alone(self):
        from spyde.actions.find_vectors_action import DEFAULTS, _auto_center
        vecs = self._found_on_an_uncentered_detector()
        params = {**DEFAULTS, "center_beam": False}
        assert _auto_center(None, vecs, params) == (vecs, None)

    def test_a_hidden_beam_leaves_the_vectors_alone(self):
        """With the beam behind a stop, the brightest vector near the middle is
        a Bragg spot as bright as its Friedel partner — nothing to center on."""
        from spyde.actions.find_vectors_action import DEFAULTS, _auto_center
        vecs = self._found_on_an_uncentered_detector()
        # The beam rows dim to the strongest reflection's level.
        beam_rows = vecs.flat_buffer[:, COL_INTENSITY] == 100.0
        vecs.flat_buffer[beam_rows, COL_INTENSITY] = 21.0
        result, note = _auto_center(None, vecs, dict(DEFAULTS))
        assert result is vecs and "uncentered" in note

    def test_the_beam_found_twice_still_stands_out(self):
        """A second detection of the beam's own disk is not another spot: the
        beam still counts as the brightest disk and the vectors are centered."""
        from spyde.actions.find_vectors_action import DEFAULTS, _auto_center
        vecs = self._found_on_an_uncentered_detector()
        beam_rows = vecs.flat_buffer[vecs.flat_buffer[:, COL_INTENSITY] == 100.0]
        twice = np.concatenate([vecs.flat_buffer, beam_rows])
        order = np.lexsort((twice[:, 0], twice[:, 1]))     # keep the scan order
        vecs = self._rebuild(vecs, twice[order])
        result, note = _auto_center(None, vecs, dict(DEFAULTS))
        assert result is not vecs and note.startswith("Zero beam centered")

    @staticmethod
    def _rebuild(vecs, rows):
        """*vecs* with its rows replaced, offsets recomputed per position."""
        from spyde.signals.diffraction_vectors import SpyDEDiffractionVectors
        ny, nx = vecs.nav_shape
        counts = np.bincount((rows[:, 1] * nx + rows[:, 0]).astype(int), minlength=ny * nx)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return SpyDEDiffractionVectors(
            flat_buffer=rows.astype(np.float32), nav_offsets=[np.arange(ny + 1) * nx, offsets],
            nav_shape=(ny, nx), full_nav_shape=(ny, nx), sig_shape=vecs.sig_shape,
            sig_axes=vecs.sig_axes, kernel_radius_px=vecs.kernel_radius_px,
            kernel_radius_data=vecs.kernel_radius_data, offsets=offsets)

    def test_a_masked_beam_stop_leaves_the_vectors_alone(self):
        from spyde.actions.find_vectors_action import DEFAULTS, _auto_center
        vecs = self._found_on_an_uncentered_detector()
        params = {**DEFAULTS, "beamstop_auto": True}
        result, _ = _auto_center(None, vecs, params)
        assert result is vecs

    def test_the_caret_checkbox_is_on_by_default_like_the_backend(self):
        from pathlib import Path
        from spyde.actions.find_vectors_action import DEFAULTS
        source = (Path(__file__).resolve().parents[3] / "electron" / "src" / "renderer"
                  / "src" / "components" / "FindVectorsWizard.tsx").read_text(encoding="utf-8")
        assert DEFAULTS["center_beam"] is True
        assert "useState(saved?.centerBeam ?? true)" in source
        assert "center_beam: v.centerBeam" in source
