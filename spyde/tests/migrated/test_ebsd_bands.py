"""Kikuchi band geometry (:mod:`spyde.ebsd.bands`) — the overlay's foundation.

Two things are worth pinning here and they are not the same thing:

* **the lines land on the bands** — the overlay's whole claim. Checked by
  sampling the simulated pattern ALONG each drawn line and requiring it to be
  brighter than the frame, and by requiring a WRONG orientation's lines not to
  be;
* **the rotation convention agrees with orix** — which no score-based test can
  see. Indexing is self-consistent whichever way round the rotation goes, so a
  transposed simulator matches its own dictionary perfectly and then colours
  the IPF map from the inverse orientation. The tell is symmetry: a CRYSTAL
  symmetry operation must leave the pattern alone and a SAMPLE rotation must
  not.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.ebsd.bands import (
    Reflectors, band_lines, cubic_reflectors, detector_directions,
    euler_to_matrix, normals_to_sample, simulate_patterns, wavelength,
    zone_axis_points,
)

DET = (60, 60)
PC = (0.5, 0.5, 0.55)
EULER = np.deg2rad([33.3, 30.0, 15.0])


def _sample_along(pattern, segments, n=40, offset=0.0):
    """Mean pattern intensity along each segment, optionally shifted *offset*
    pixels perpendicular to it."""
    out = []
    h, w = pattern.shape
    for (x0, y0), (x1, y1) in segments:
        t = np.linspace(0.05, 0.95, n)
        v = np.array([x1 - x0, y1 - y0], float)
        nrm = np.array([-v[1], v[0]]) / max(np.linalg.norm(v), 1e-12) * offset
        xs = np.clip(np.round(x0 + t * (x1 - x0) + nrm[0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(y0 + t * (y1 - y0) + nrm[1]).astype(int), 0, h - 1)
        out.append(float(pattern[ys, xs].mean()))
    return np.asarray(out)


def _perpendicular_profile(pattern, segments, half=6):
    """Intensity across each band: ``(n_bands, 2*half+1)`` sampled at
    perpendicular offsets ``-half … +half`` pixels.

    Comparing a line against the WHOLE-FRAME mean does not actually test
    anything about placement — a weak band grazing a dim corner sits below the
    frame mean while being drawn perfectly, and a line through the bright
    middle beats the mean while being drawn wrong. What the overlay claims is
    LOCAL: the pattern is brightest ON the line and falls away to either side.
    """
    offs = np.arange(-half, half + 1, dtype=float)
    return np.column_stack([_sample_along(pattern, segments, offset=o)
                            for o in offs]), offs


class TestConventionAgreesWithOrix:
    """The Euler angles this module simulates from are handed straight to orix
    to colour the IPF map, so the two must mean the same thing."""

    def test_euler_matrix_matches_orix(self):
        from orix.quaternion import Rotation
        ours = euler_to_matrix(*EULER)
        theirs = Rotation.from_euler(EULER[None]).to_matrix()[0]
        assert np.allclose(ours, theirs, atol=1e-9), (
            "Bunge matrix disagrees with orix — every orientation leaving this "
            "module would be transposed")

    def test_crystal_symmetry_leaves_the_pattern_alone(self):
        """phi2 + 90 deg is a cubic 4-fold about the crystal z: the SAME
        crystal orientation, so the same pattern."""
        from orix.quaternion import Orientation, symmetry
        a = Orientation.from_euler(EULER[None], symmetry.Oh)
        b = Orientation.from_euler((EULER + np.deg2rad([0, 0, 90]))[None],
                                   symmetry.Oh)
        assert float(np.rad2deg(a.angle_with(b))[0]) < 1e-6, \
            "orix says these are different orientations; the premise is wrong"

        p0 = simulate_patterns(EULER, detector=DET, pc=PC)[0]
        p1 = simulate_patterns(EULER + np.deg2rad([0, 0, 90]),
                               detector=DET, pc=PC)[0]
        assert np.allclose(p0, p1, atol=1e-6), \
            "a crystal symmetry operation changed the pattern — the normals " \
            "are being rotated by the transpose"

    def test_a_sample_rotation_does_change_the_pattern(self):
        """The other half of the same check — without it, a simulator that
        ignores the rotation entirely would pass the test above."""
        p0 = simulate_patterns(EULER, detector=DET, pc=PC)[0]
        p1 = simulate_patterns(EULER + np.deg2rad([90, 0, 0]),
                               detector=DET, pc=PC)[0]
        assert not np.allclose(p0, p1, atol=1e-3)

    def test_normals_to_sample_is_the_transpose(self):
        rot = euler_to_matrix(*EULER)
        n = cubic_reflectors().normals
        assert np.allclose(normals_to_sample(n, rot), (rot.T @ n.T).T)


class TestBandLinesLandOnBands:
    def test_every_line_sits_on_a_local_intensity_peak(self):
        """The overlay's whole claim: each drawn line is the CENTRE of a band,
        so the pattern peaks on it and falls away to either side."""
        pat = simulate_patterns(EULER, detector=DET, pc=PC)[0]
        segs, weights = band_lines(EULER, detector=DET, pc=PC)
        assert len(segs) > 3, "no bands crossed the detector"
        assert len(weights) == len(segs)

        profile, offs = _perpendicular_profile(pat, segs)
        peak = offs[profile.argmax(axis=1)]
        assert np.abs(peak).max() <= 1.0, \
            f"band lines are offset from the bands by up to {np.abs(peak).max()}px"
        # And the peak is a real one, not a plateau.
        assert (profile.max(1) > profile[:, [0, -1]].max(1)).all()

    def test_a_wrong_orientation_does_not_land_on_them(self):
        """Without this, lines drawn anywhere down a bright band would pass."""
        pat = simulate_patterns(EULER, detector=DET, pc=PC)[0]
        wrong = np.deg2rad([70.0, 55.0, 40.0])
        segs, _ = band_lines(wrong, detector=DET, pc=PC)
        profile, offs = _perpendicular_profile(pat, segs)
        peak = offs[profile.argmax(axis=1)]
        assert np.abs(peak).max() > 1.0, \
            "a wrong orientation's lines also landed on band centres"
        assert _sample_along(pat, segs).mean() < pat.mean()

    def test_segments_are_the_line_collection_shape(self):
        """anyplotlib add_lines takes (N, 2, 2) and raises on anything else."""
        segs, _ = band_lines(EULER, detector=DET, pc=PC)
        assert segs.ndim == 3 and segs.shape[1:] == (2, 2)

    def test_segments_stay_inside_the_detector(self):
        segs, _ = band_lines(EULER, detector=DET, pc=PC)
        assert segs.min() >= -0.5 - 1e-6
        assert segs[..., 0].max() <= DET[1] - 0.5 + 1e-6
        assert segs[..., 1].max() <= DET[0] - 0.5 + 1e-6

    def test_pixel_convention_is_centre_on_integer(self):
        """A band through the pattern centre must be drawn through the centre
        pixel, not half a pixel off — the overlay is drawn in the image-pixel
        coordinates anyplotlib's transform="data" markers use."""
        # [001] out of the detector: the (200) plane containing the beam gives
        # a band straight through the pattern centre.
        segs, _ = band_lines(np.zeros(3), detector=DET, pc=(0.5, 0.5, 0.55))
        centre = (DET[1] - 1) / 2.0
        mid = segs.mean(axis=1)                    # (N, 2) segment midpoints
        assert np.abs(mid - centre).min() < 1e-6, \
            "no band passes through the detector centre for the identity " \
            "orientation with a centred PC"

    def test_max_bands_keeps_the_strongest(self):
        segs_all, w_all = band_lines(EULER, detector=DET, pc=PC)
        segs_few, w_few = band_lines(EULER, detector=DET, pc=PC, max_bands=3)
        assert len(segs_few) <= 3 < len(segs_all)
        assert w_few.min() >= np.sort(w_all)[::-1][:len(w_few)].min() - 1e-9

    def test_a_band_that_misses_the_detector_is_dropped(self):
        """A tiny detector sees fewer bands — silently returning off-frame
        segments would draw lines pinned to the edge."""
        few, _ = band_lines(EULER, detector=(60, 60), pc=(0.5, 0.5, 4.0))
        many, _ = band_lines(EULER, detector=(60, 60), pc=(0.5, 0.5, 0.4))
        assert len(few) < len(many)


class TestReflectors:
    def test_cubic_set_is_friedel_unique(self):
        """4x{111} + 3x{200} + 6x{220} = 13 bands, one per +-g pair."""
        r = cubic_reflectors()
        assert len(r) == 13
        dots = np.abs(r.normals @ r.normals.T)
        np.fill_diagonal(dots, 0.0)
        assert dots.max() < 1 - 1e-9, "a band appears twice as +g and -g"

    def test_widths_and_weights_are_per_reflector(self):
        r = cubic_reflectors()
        assert r.widths.shape == r.weights.shape == (len(r),)
        assert (r.widths > 0).all()

    def test_brightest_is_a_subset(self):
        r = cubic_reflectors()
        sub = r.brightest(5)
        assert len(sub) == 5
        assert sub.weights.min() >= np.sort(r.weights)[::-1][:5].min() - 1e-12

    def test_brightest_none_is_a_no_op(self):
        r = cubic_reflectors()
        assert len(r.brightest(None)) == len(r)
        assert len(r.brightest(999)) == len(r)

    def test_from_a_real_phase(self):
        """A .cif-backed phase gives real reflectors with Bragg-angle widths —
        an order of magnitude narrower than the synthetic set's exaggerated
        ones, which is why width is a per-reflector property."""
        pytest.importorskip("diffsims")
        from diffpy.structure import Atom, Lattice, Structure
        from orix.crystal_map import Phase
        from spyde.ebsd.bands import reflectors_from_phase

        phase = Phase(name="ni", space_group=225,
                      structure=Structure(atoms=[Atom("Ni", [0, 0, 0])],
                                          lattice=Lattice(3.52, 3.52, 3.52,
                                                          90, 90, 90)))
        r = reflectors_from_phase(phase, min_dspacing=0.8, voltage_kv=20.0)
        assert len(r) > 5
        assert np.isfinite(r.weights).all() and r.weights.max() > 0
        assert (r.widths > 0).all() and r.widths.max() < 0.1
        assert np.allclose(np.linalg.norm(r.normals, axis=1), 1.0)

    def test_a_structureless_phase_falls_back(self):
        """A bare Phase(space_group=…) has no atoms, so no structure factors —
        the wizard must still work, with the generic cubic band set."""
        from orix.crystal_map import Phase
        from spyde.ebsd.bands import reflectors_from_phase
        r = reflectors_from_phase(Phase(name="p", space_group=225))
        assert len(r) == len(cubic_reflectors())


class TestZoneAxes:
    def test_axes_fall_where_bands_cross(self):
        segs, _ = band_lines(EULER, detector=DET, pc=PC)
        za = zone_axis_points(EULER, detector=DET, pc=PC)
        assert len(za) >= 1
        # Every zone axis must sit ON at least two of the drawn band lines.
        for p in za:
            d = []
            for (x0, y0), (x1, y1) in segs:
                v = np.array([x1 - x0, y1 - y0], float)
                n = np.array([-v[1], v[0]]) / max(np.linalg.norm(v), 1e-12)
                d.append(abs(float((p - np.array([x0, y0])) @ n)))
            assert sorted(d)[1] < 1.0, f"zone axis {p} lies on fewer than 2 bands"

    def test_points_stay_on_the_detector(self):
        za = zone_axis_points(EULER, detector=DET, pc=PC)
        assert ((za >= -0.5) & (za <= np.array([DET[1], DET[0]]) - 0.5)).all()


class TestGeometryBasics:
    def test_detector_directions_are_unit_vectors(self):
        r = detector_directions(DET, PC)
        assert r.shape == (DET[0], DET[1], 3)
        assert np.allclose(np.linalg.norm(r, axis=-1), 1.0)

    def test_detector_y_is_flipped(self):
        """Detector row 0 is the TOP, which is +y in the sample frame. Getting
        this backwards mirrors every pattern and is invisible in a symmetric
        one — which is why the synthetic data is deliberately asymmetric."""
        r = detector_directions(DET, PC)
        assert r[0, DET[1] // 2, 1] > 0 > r[-1, DET[1] // 2, 1]

    def test_wavelength_is_relativistic(self):
        # 20 kV: 0.0859 A relativistic, 0.0867 A if the correction is dropped.
        assert abs(wavelength(20.0) - 0.0859) < 5e-4

    def test_reflectors_dataclass_broadcasts_scalars(self):
        r = Reflectors(np.eye(3), 1.0, 0.05)
        assert r.weights.shape == (3,) and r.widths.shape == (3,)
