"""The SpyDE side of the vendored quantem matcher.

Everything here guards a seam between our data and upstream's: the peak
protocol their matcher reads, the quaternion convention that bridges their
rotations to orix, and the zone-angle cache that makes a per-pattern fit fast
enough to run under the crosshair. The physics is upstream's and is covered by
``benchmark_quantem_orientation`` on real data; these are the joins.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.signals.diffraction_vectors import COL_KX, COL_KY, COL_INTENSITY

pytest.importorskip("ase", reason="quantem's crystal.py needs ase + spglib")
pytest.importorskip("spglib", reason="quantem's crystal.py needs ase + spglib")

from spyde.actions.vector_orientation_quantem import (  # noqa: E402
    PEAK_FIELDS, PeaksAdapter, QUAT_CONVENTIONS, quantem_quats_to_orix,
)


class _StubVectors:
    """The slice of SpyDEDiffractionVectors the adapter reads."""

    n_time = 0

    def __init__(self, rows_by_position, nav_shape):
        self._rows = rows_by_position
        self.nav_shape = nav_shape

    def at(self, row, column):
        return self._rows[(row, column)]


def _rows(*peaks):
    """(N, 6) flat-buffer rows from (kx, ky, intensity) triples."""
    out = np.zeros((len(peaks), 6), dtype=np.float32)
    for i, (kx, ky, intensity) in enumerate(peaks):
        out[i, COL_KX], out[i, COL_KY], out[i, COL_INTENSITY] = kx, ky, intensity
    return out


@pytest.fixture
def vectors():
    return _StubVectors({
        (0, 0): _rows((0.1, 0.2, 10.0), (-0.3, 0.4, 20.0)),
        (0, 1): _rows((0.5, -0.6, 30.0)),
        (1, 0): _rows(),                       # a position with no peaks
        (1, 1): _rows((0.7, 0.8, 40.0), (0.9, -1.0, 50.0), (1.1, 1.2, 60.0)),
    }, nav_shape=(2, 2))


class TestPeaksProtocol:
    """Their OrientationMap duck-types upstream's ragged container. If any of
    this drifts the matcher does not fail, it reads the wrong thing."""

    def test_shape_and_fields(self, vectors):
        peaks = PeaksAdapter(vectors)
        assert peaks.shape == (2, 2)
        assert tuple(peaks.fields) == PEAK_FIELDS

    def test_rows_are_qx_qy_intensity_in_that_order(self, vectors):
        peaks = PeaksAdapter(vectors)
        array = peaks[0, 0].array
        assert array.shape == (2, 3)
        np.testing.assert_allclose(array[0], [0.1, 0.2, 10.0], rtol=1e-6)
        np.testing.assert_allclose(array[1], [-0.3, 0.4, 20.0], rtol=1e-6)

    def test_a_position_with_no_peaks_is_an_empty_array_not_an_error(self, vectors):
        assert PeaksAdapter(vectors)[1, 0].array.shape == (0, 3)

    def test_indexing_is_row_then_column(self, vectors):
        peaks = PeaksAdapter(vectors)
        assert peaks[0, 1].array.shape[0] == 1
        assert peaks[1, 1].array.shape[0] == 3

    def test_flatten_returns_every_peak(self, vectors):
        flat = PeaksAdapter(vectors).select_fields(*PEAK_FIELDS).flatten()
        assert flat.shape == (6, 3)

    def test_peak_counts_follow_scan_order(self, vectors):
        np.testing.assert_array_equal(
            PeaksAdapter(vectors).peak_counts, [2, 1, 0, 3])


class TestUnitConversion:
    """Vectors are stored in the detector's own units; the templates are always
    inverse angstroms. A scan calibrated in nm^-1 is out by ten if this is
    dropped — silently, because every ring simply lands on the wrong shell."""

    def test_coordinates_are_scaled_but_intensity_is_not(self, vectors):
        peaks = PeaksAdapter(vectors, inverse_angstrom_factor=10.0)
        array = peaks[0, 0].array
        np.testing.assert_allclose(array[0, :2], [1.0, 2.0], rtol=1e-6)
        np.testing.assert_allclose(array[0, 2], 10.0, rtol=1e-6)

    def test_unit_factor_defaults_to_no_conversion(self, vectors):
        np.testing.assert_allclose(
            PeaksAdapter(vectors)[0, 0].array[0, :2], [0.1, 0.2], rtol=1e-6)


class TestQuaternionConvention:
    def test_conjugate_is_the_default(self):
        quats = np.array([[0.5, 0.5, 0.5, 0.5]])
        np.testing.assert_allclose(
            quantem_quats_to_orix(quats), quantem_quats_to_orix(quats, "conjugate"))

    def test_conjugate_negates_only_the_vector_part(self):
        out = quantem_quats_to_orix(np.array([[0.5, 0.1, 0.2, 0.3]]), "conjugate")
        np.testing.assert_allclose(out, [[0.5, -0.1, -0.2, -0.3]], rtol=1e-6)

    def test_identity_passes_through(self):
        quats = np.array([[0.5, 0.1, 0.2, 0.3]])
        np.testing.assert_allclose(
            quantem_quats_to_orix(quats, "identity"), quats, rtol=1e-6)

    def test_an_unknown_convention_is_refused(self):
        with pytest.raises(ValueError, match="unknown convention"):
            quantem_quats_to_orix(np.zeros((1, 4)), "sideways")

    def test_both_conventions_are_offered_to_the_benchmark_sweep(self):
        assert set(QUAT_CONVENTIONS) == {"identity", "conjugate"}


class _StubMap:
    """The slice of a matched OrientationMap that ``combine_phases`` reads."""

    def __init__(self, correlation, quats, reliability, angle_deg, counts):
        import torch

        self.corr = torch.as_tensor(correlation, dtype=torch.float64)[..., None]
        self.quats = torch.as_tensor(quats, dtype=torch.float64)[..., None, :]
        self.reliability = torch.as_tensor(reliability, dtype=torch.float64)
        self._angle = torch.as_tensor(angle_deg, dtype=torch.float64)
        self.peaks = type("P", (), {"peak_counts": np.asarray(counts)})()

    def in_plane_angle_deg(self):
        return self._angle


def _stub(correlation, quats, reliability=None, angle_deg=None):
    correlation = np.asarray(correlation, float)
    shape = correlation.shape
    if reliability is None:
        reliability = np.zeros(shape)
    if angle_deg is None:
        angle_deg = np.zeros(shape)
    return _StubMap(correlation, np.asarray(quats, float), reliability,
                    angle_deg, np.full(correlation.size, 5))


class TestPhaseSelection:
    """Each phase is matched against its own plan — the polar shells are that
    crystal's own lattice radii, so nothing is shareable. What makes combining
    them afterwards legitimate is that their correlation is a normalised cosine
    similarity, comparable across crystals, so a per-position argmax is a real
    comparison rather than a contest of library sizes."""

    def test_the_higher_correlation_wins_each_position(self):
        from spyde.actions.vector_orientation_quantem import combine_phases

        phase_a = _stub([[0.9, 0.1]], [[[1, 0, 0, 0], [1, 0, 0, 0]]])
        phase_b = _stub([[0.2, 0.8]], [[[0, 1, 0, 0], [0, 1, 0, 0]]])
        result = combine_phases([phase_a, phase_b], [{"name": "a"}, {"name": "b"}])
        np.testing.assert_array_equal(result.phase_idx, [[0, 1]])
        np.testing.assert_allclose(result.coarse_score, [[0.9, 0.8]], rtol=1e-6)

    def test_the_winner_supplies_the_orientation(self):
        from spyde.actions.vector_orientation_quantem import combine_phases

        phase_a = _stub([[0.9, 0.1]], [[[1, 0, 0, 0], [1, 0, 0, 0]]])
        phase_b = _stub([[0.2, 0.8]], [[[0, 1, 0, 0], [0, 1, 0, 0]]])
        result = combine_phases([phase_a, phase_b], [{"name": "a"}, {"name": "b"}])
        # conjugate convention negates the vector part
        np.testing.assert_allclose(result.quats[0, 0], [1, 0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(result.quats[0, 1], [0, -1, 0, 0], atol=1e-6)

    def test_the_winner_supplies_the_strain(self):
        from spyde.actions.vector_orientation_quantem import combine_phases

        phase_a = _stub([[0.9, 0.1]], [[[1, 0, 0, 0], [1, 0, 0, 0]]])
        phase_b = _stub([[0.2, 0.8]], [[[1, 0, 0, 0], [1, 0, 0, 0]]])
        strain_a = np.full((1, 2, 3), 0.01)
        strain_b = np.full((1, 2, 3), 0.02)
        result = combine_phases([phase_a, phase_b], [{"name": "a"}, {"name": "b"}],
                                strains=[strain_a, strain_b])
        np.testing.assert_allclose(result.strain[0, 0], 0.01, rtol=1e-5)
        np.testing.assert_allclose(result.strain[0, 1], 0.02, rtol=1e-5)

    def test_reliability_comes_from_the_winner(self):
        from spyde.actions.vector_orientation_quantem import combine_phases

        phase_a = _stub([[0.9]], [[[1, 0, 0, 0]]], reliability=[[0.4]])
        phase_b = _stub([[0.2]], [[[1, 0, 0, 0]]], reliability=[[0.7]])
        result = combine_phases([phase_a, phase_b], [{"name": "a"}, {"name": "b"}])
        np.testing.assert_allclose(result.reliability, [[0.4]], rtol=1e-6)

    def test_an_unmatched_position_is_left_unfitted(self):
        """corr 0 everywhere means nothing was matched — it must not silently
        report phase 0 with a strain."""
        from spyde.actions.vector_orientation_quantem import combine_phases

        phase_a = _stub([[0.0]], [[[1, 0, 0, 0]]])
        result = combine_phases([phase_a], [{"name": "a"}],
                                strains=[np.full((1, 1, 3), 0.01)])
        assert result.coarse_score[0, 0] == 0.0
        assert np.isnan(result.strain[0, 0]).all()

    def test_one_phase_is_just_the_single_phase_case(self):
        from spyde.actions.vector_orientation_quantem import combine_phases, to_result

        phase_a = _stub([[0.9, 0.3]], [[[1, 0, 0, 0], [0, 0, 1, 0]]])
        direct = to_result(phase_a, [{"name": "a"}])
        combined = combine_phases([phase_a], [{"name": "a"}])
        np.testing.assert_array_equal(direct.phase_idx, combined.phase_idx)
        np.testing.assert_allclose(direct.quats, combined.quats, atol=1e-6)


class TestStrainRecoversWhatWasApplied:
    """The strain gate. Real data cannot settle this — on sped_ag both our fit
    and theirs sit at sd ~0.01 against a ~0.02 peak-finding noise floor, so two
    independent noise realisations correlate at zero whether or not either is
    right. Here the strain is applied by hand and must come back.

    It also pins the sign. A real-space deformation ``F = I + eps`` sends
    reciprocal vectors to ``q @ inv(F)``, so the real-space branch must return
    ``+eps`` and the reciprocal branch its negation. Getting this backwards
    reports every compression as a tension, which no test on unstrained data
    would ever catch.
    """

    APPLIED = np.array([[0.025, 0.008], [0.008, -0.018]])

    @staticmethod
    def _simulate(crystal, quat, strain, noise, rng, sigma):
        from spyde.external.quantem.diffraction.rotations import quat_to_matrix

        rotation = quat_to_matrix(quat[None])[0].numpy()
        g = crystal.g_vec.numpy() @ rotation.T
        wavelength = 0.0251
        gz, g2 = g[:, 2], (g ** 2).sum(1)
        excitation = (2 * gz - wavelength * g2) / (2 - 2 * wavelength * gz)
        keep = np.abs(excitation) < sigma
        q = g[keep, :2] @ np.linalg.inv(np.eye(2) + strain)
        if noise:
            q = q + rng.normal(0, noise, q.shape)
        intensity = crystal.struct_factors_int.numpy()[keep]
        return np.column_stack([q, intensity]).astype(np.float64)

    @pytest.fixture(scope="class")
    def recovered(self):
        """Median recovered strain, both branches, from a strained simulation."""
        import os

        from orix.crystal_map import Phase

        from spyde.actions.vector_orientation_quantem import (
            build_orientation_map, orientation_map_class, phase_to_crystal,
            strain_from_orientation_map,
        )
        from spyde.tests.migrated import conftest  # noqa: F401  (path setup)

        cif = os.path.join(os.path.dirname(__file__), "..", "Silver__0011135.cif")
        sigma, k_max = 0.04, 1.5
        crystal = phase_to_crystal(Phase.from_cif(cif), name="Ag")
        crystal.calculate_structure_factors(k_max=k_max)
        rng = np.random.RandomState(0)

        class _Peaks:
            fields = list(PEAK_FIELDS)

            def __init__(self, cells):
                self.shape = (1, len(cells))
                self.metadata = {}
                self._cells = [type("C", (), {"array": a})() for a in cells]

            def __getitem__(self, index):
                return self._cells[index[1]]

            def select_fields(self, *names):
                return self

            def flatten(self):
                return np.vstack([c.array for c in self._cells])

        base = build_orientation_map(
            _Peaks([np.zeros((0, 3))]), crystal, energy_ev=200e3, k_max=k_max,
            angle_step_zone_axis_deg=4.0, verbose=False)
        picks = rng.choice(base.zone_quats.shape[0], 8, replace=False)
        cells = [self._simulate(crystal, base.zone_quats[i], self.APPLIED, 0.0,
                                rng, sigma) for i in picks]

        om = orientation_map_class("cpu").from_vectors(
            _Peaks(cells), crystal, base.energy_ev)
        for attr in ("device", "dtype", "cdtype", "corr_kernel_size",
                     "sigma_excitation", "power_radial", "power_intensity",
                     "zone_axes", "zone_quats", "zone_step_deg", "zone_nbr_idx",
                     "zone_nbr_pos", "zone_nbr_valid", "plan_fft", "shell_radii",
                     "num_gamma", "gamma", "detector_mask", "plan_norm_shift",
                     "plan_frac_shift"):
            setattr(om, attr, getattr(base, attr))
        om.metadata = dict(base.metadata)
        om.match_orientations(progress_bar=False)
        om.refine_orientations(progress_bar=False, neighbor_rescue=False)

        out = {}
        for name, reciprocal in (("real", False), ("reciprocal", True)):
            strain, _pairs = strain_from_orientation_map(
                om, reciprocal=reciprocal, sigma_excitation=sigma)
            flat = strain.reshape(-1, 3)
            usable = flat[np.isfinite(flat).all(1)]
            assert usable.size, "no position produced a strain"
            out[name] = np.median(usable, axis=0)
        return out

    def test_real_space_recovers_the_applied_strain(self, recovered):
        expected = [self.APPLIED[0, 0], self.APPLIED[1, 1], self.APPLIED[0, 1]]
        np.testing.assert_allclose(recovered["real"], expected, atol=2e-3)

    def test_the_reciprocal_branch_is_its_negation(self, recovered):
        np.testing.assert_allclose(
            recovered["reciprocal"], -recovered["real"], atol=2e-3)

    def test_the_sign_is_not_merely_symmetric(self, recovered):
        """exx positive and eyy negative, as applied — a fit that returned the
        magnitudes with both signs flipped would pass a symmetric check."""
        assert recovered["real"][0] > 0.015
        assert recovered["real"][1] < -0.010


def _ag_fitter(applied_strain, zone_axis=(1.0, 1.0, 1.0), seed=1,
               angle_step_zone_axis_deg=4.0):
    """A real silver fitter and one simulated pattern's rows.

    Shared by the fit tests and the zone-mask tests so they exercise the same
    plan — the mask rides the plan's detector-aperture correction, so a stub
    plan would not test the thing that makes it work.
    """
    import os

    import torch
    from orix.crystal_map import Phase

    from spyde.actions.vector_orientation_quantem import (
        SinglePatternFitter, phase_to_crystal,
    )
    from spyde.external.quantem.diffraction.rotations import quat_from_zone_axis

    cif = os.path.join(os.path.dirname(__file__), "..", "Silver__0011135.cif")
    crystal = phase_to_crystal(Phase.from_cif(cif), name="Ag")
    crystal.calculate_structure_factors(k_max=1.5)

    rng = np.random.RandomState(seed)
    zone = quat_from_zone_axis(
        torch.tensor([list(zone_axis)], dtype=torch.float64))[0]
    peaks = TestStrainRecoversWhatWasApplied._simulate(
        crystal, zone, applied_strain, 0.0, rng, 0.04)
    rows = _rows(*[(q[0], q[1], q[2]) for q in peaks])

    # The plan measures the detector footprint from the peaks, so it is built
    # from the scan's peaks, not from an empty stand-in.
    class _Scan:
        fields = list(PEAK_FIELDS)
        shape = (1, 1)
        metadata: dict = {}

        def __getitem__(self, index):
            return type("C", (), {"array": peaks})()

        def select_fields(self, *names):
            return self

        def flatten(self):
            return peaks

        @property
        def peak_counts(self):
            return np.array([peaks.shape[0]])

    fitter = SinglePatternFitter(
        [crystal], _Scan(), energy_ev=200e3, k_max=1.5,
        angle_step_zone_axis_deg=angle_step_zone_axis_deg)
    return fitter, rows


class TestSinglePatternFitter:
    """The unit the live overlay consumes. A whole-field fit takes tens of
    seconds and cannot follow a navigator; one pattern against an existing plan
    can, and this is what it has to return to be worth wiring up."""

    APPLIED = np.array([[0.02, 0.005], [0.005, -0.015]])

    @pytest.fixture(scope="class")
    def fitted(self):
        fitter, rows = _ag_fitter(self.APPLIED)
        return fitter, rows, fitter.fit(rows)

    def test_it_fits(self, fitted):
        _fitter, _rows, fit = fitted
        assert fit is not None
        # Normalised to about [0, 1] but not bounded by 1 — see
        # SinglePatternFit.correlation. A near-perfect fit can land just over.
        assert 0.0 < fit.correlation <= 1.1

    def test_too_few_peaks_returns_none_rather_than_a_bad_fit(self, fitted):
        fitter, _rows, _fit = fitted
        assert fitter.fit(_rows_few()) is None

    def test_it_reports_a_reliability(self, fitted):
        """The discriminability metric our own path has no field for at all."""
        _fitter, _rows, fit = fitted
        assert np.isfinite(fit.reliability)

    def test_it_recovers_the_applied_strain(self, fitted):
        _fitter, _rows, fit = fitted
        expected = [self.APPLIED[0, 0], self.APPLIED[1, 1], self.APPLIED[0, 1]]
        np.testing.assert_allclose(fit.strain, expected, atol=4e-3)

    def test_every_measured_peak_has_a_spot_drawn_on_it(self, fitted):
        """The overlay's whole job.

        Asserted in this direction on purpose. A simulated pattern carries far
        more reflections than a peak finder detects — 29 against 8 here — so
        most drawn spots having no measured partner is correct, and requiring
        it would fail a working overlay. What must hold is the converse: no
        measured peak is left unexplained.
        """
        _fitter, rows, fit = fitted
        measured = np.column_stack([rows[:, COL_KX], rows[:, COL_KY]])
        distance = np.linalg.norm(
            fit.spots[:, None, :] - measured[None, :, :], axis=-1).min(axis=0)
        explained = (distance < 0.02).mean()
        assert explained > 0.95, f"only {explained:.0%} of peaks have a spot"

    def test_the_spots_are_deformed_by_the_fit_not_left_ideal(self, fitted):
        """Pins the direction of the deformation. Drawing the unstrained
        pattern still explains most peaks on a lightly strained crystal, so
        only the residual distance separates right from nearly-right: applying
        the fitted map lands the spots an order of magnitude closer, and the
        transpose or the inverse of it lands them much further away.
        """
        import torch

        fitter, rows, fit = fitted
        measured = np.column_stack([rows[:, COL_KX], rows[:, COL_KY]])
        orientation_map = fitter._maps[fit.phase_index]
        pattern = orientation_map.generate_pattern(0, 0)
        ideal = torch.stack((pattern["qx"], pattern["qy"]), dim=1).numpy()

        def residual(spots):
            return np.median(np.linalg.norm(
                spots[:, None, :] - measured[None, :, :], axis=-1).min(axis=0))

        assert residual(fit.spots) < residual(ideal) / 4

    def test_spots_and_intensities_agree_in_length(self, fitted):
        _fitter, _rows, fit = fitted
        assert fit.spots.shape[0] == fit.intensities.shape[0]
        assert fit.spots.shape[1] == 2


def _rows_few():
    return _rows((0.1, 0.1, 1.0), (0.2, 0.2, 1.0))


class TestZoneAngleCache:
    """The cache exists for speed and must not change a single number."""

    def test_cached_table_equals_the_uncached_one(self):
        import torch

        from spyde.actions import vector_orientation_quantem as adapter
        from spyde.external.quantem.diffraction import orientation as qorient
        from spyde.external.quantem.diffraction.rotations import (
            symmetry_reduced_zone_angles as uncached,
        )

        adapter.install_zone_angle_cache()
        zone_axes = torch.nn.functional.normalize(
            torch.randn(24, 3, dtype=torch.float64), dim=-1)
        symmetry = torch.tensor([[1.0, 0, 0, 0], [0.0, 0, 0, 1]], dtype=torch.float64)

        expected = uncached(zone_axes, symmetry)
        got = qorient.symmetry_reduced_zone_angles(zone_axes, symmetry)
        assert torch.equal(got, expected)

    def test_a_repeat_call_is_served_from_the_cache(self):
        import torch

        from spyde.actions import vector_orientation_quantem as adapter
        from spyde.external.quantem.diffraction import orientation as qorient

        adapter.install_zone_angle_cache()
        zone_axes = torch.nn.functional.normalize(
            torch.randn(16, 3, dtype=torch.float64), dim=-1)
        symmetry = torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64)

        first = qorient.symmetry_reduced_zone_angles(zone_axes, symmetry)
        second = qorient.symmetry_reduced_zone_angles(zone_axes, symmetry)
        assert second is first, "a repeat call should not recompute the table"

    def test_installing_twice_is_harmless(self):
        from spyde.actions import vector_orientation_quantem as adapter
        from spyde.external.quantem.diffraction import orientation as qorient

        adapter.install_zone_angle_cache()
        once = qorient.symmetry_reduced_zone_angles
        adapter.install_zone_angle_cache()
        assert qorient.symmetry_reduced_zone_angles is once, \
            "re-installing must not wrap the wrapper"


class TestConstantsTrackUpstream:
    """The adapter restates three of upstream's numbers. Two are re-exported
    and cannot drift; the third is copied out of a signature and can."""

    def test_pairing_and_intensity_are_upstream_s_own_objects(self):
        from spyde.actions import vector_orientation_quantem as adapter
        from spyde.external.quantem.diffraction import defaults

        assert adapter.PAIR_DISTANCE is defaults.PAIR_DISTANCE
        assert adapter.MIN_SIM_INTENSITY_REL is defaults.MIN_SIM_INTENSITY_REL

    def test_the_detector_fraction_still_matches_the_signature_it_copies(self):
        """A masked zone is one with zero on-detector weight, so the mask only
        works while this threshold is above zero AND agrees with the matcher.
        Upstream keeps it in a signature, so nothing but this test would notice
        it moving — and a mask that silently stops restricting anything is the
        kind of failure a user reads as the feature being broken."""
        import inspect

        from spyde.actions.vector_orientation_quantem import MIN_DETECTOR_FRACTION
        from spyde.external.quantem.diffraction.orientation import OrientationMap

        upstream = inspect.signature(
            OrientationMap.match_orientations
        ).parameters["min_detector_fraction"].default
        assert MIN_DETECTOR_FRACTION == upstream
        assert MIN_DETECTOR_FRACTION > 0


class TestTheHeatMapDoesNotDisturbAFitInProgress:
    """The heat map runs inline on the navigator thread; the matched-pattern
    overlay fits on the overlay lane. They share one matcher, and every stage
    of a fit (match, refine, strain) re-reads ``orientation_map.peaks``.

    So the heat map must not assign it. It used to, which rebound the pattern
    out from under a fit already running — the fit stayed self-consistent only
    because a superseded one is discarded, which is three unrelated mechanisms
    away from being a guarantee. It never needed to: ``_polar_images`` takes
    its arrays as an argument.
    """

    def test_zone_correlations_leaves_peaks_alone(self):
        import numpy as np

        from spyde.actions.vector_orientation_quantem import SinglePatternFitter

        sentinel = object()
        seen = []

        class _Map:
            device = "cpu"

            def __init__(self):
                self.peaks = sentinel

            def _polar_images(self, arrays, ix):
                # self, not the class: the bug assigned an INSTANCE attribute,
                # which reading through the class would not have seen.
                seen.append(self.peaks)
                raise RuntimeError("far enough")

        fitter = SinglePatternFitter.__new__(SinglePatternFitter)
        fitter.min_peaks = 1
        fitter.inverse_angstrom_factor = 1.0
        fitter._metadata = {}
        shared_map = _Map()
        fitter._maps = [shared_map]

        with pytest.raises(RuntimeError, match="far enough"):
            fitter.zone_correlations(np.zeros((6, 6), np.float64))

        assert seen == [sentinel], "the heat map rebound the shared peaks"
        assert shared_map.peaks is sentinel
