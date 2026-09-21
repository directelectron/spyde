"""
Tests for spyde.multiangle — the two integer alignments of a multi-angle
acquisition, plus the model that carries them.

The acceptance gate is the planted answer: :func:`make_multiangle` builds
members displaced by offsets it returns, and both solvers must reproduce those
integers EXACTLY. Nothing here eyeballs an image.

The sign convention gets its own class, because that is the one bug that looks
plausible while being exactly backwards (see :mod:`spyde.drift.model`). Those
tests do not compare the recovered numbers to the planted ones — they APPLY the
recovered offsets and assert the members actually line up, and that the inverted
offsets do not.

Qt-free and dask-free (bar one streaming guard), pure compute on small
synthetic members.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from spyde.multiangle import (
    MultiAngleModel,
    assign_shells,
    make_multiangle,
    solve_real_space,
    solve_reciprocal,
)
from spyde.multiangle.align import (
    REGISTRATION_CANDIDATES,
    REGISTRATION_PREFILTERS,
    best_real_space,
)

# Mixed shells on purpose: 1 centre member, 6 at one tilt, 3 at another. Equal
# member counts per shell would let a solver that assumed them pass.
SHELLS = ((0.0, 1), (1.0, 6), (0.5, 3))

#: A format version this build cannot possibly know how to read.
FUTURE_VERSION = 99


def _normalised_mismatch(region: np.ndarray, reference: np.ndarray) -> float:
    """Largest relative disagreement between two overlap regions.

    Normalised by the mean so a member's overall brightness — which differs with
    noise and with where its window fell — cannot masquerade as misalignment.
    """
    return float(np.abs(region / region.mean()
                        - reference / reference.mean()).max())


@pytest.fixture(scope="module")
def members():
    return make_multiangle(shells=SHELLS, reference=0)


@pytest.fixture(scope="module")
def navigators(members):
    return members.navigators()


@pytest.fixture(scope="module")
def mean_patterns(members):
    return members.mean_patterns()


@pytest.fixture(scope="module")
def real_space_solution(members, navigators):
    return solve_real_space(navigators, reference=members.reference)


@pytest.fixture(scope="module")
def reciprocal_solution(members, mean_patterns):
    return solve_reciprocal(mean_patterns, reference=members.reference)


@pytest.fixture(scope="module")
def model(members, real_space_solution, reciprocal_solution):
    nav_offsets, nav_residuals = real_space_solution
    dp_offsets, dp_residuals = reciprocal_solution
    return MultiAngleModel(
        paths=members.paths,
        tilts=members.tilts,
        azimuths=members.azimuths,
        shell_ids=members.shell_ids,
        nav_offsets=nav_offsets,
        dp_offsets=dp_offsets,
        reference=members.reference,
        nav_residuals=nav_residuals,
        dp_residuals=dp_residuals,
        provenance={"solver": "phase correlation", "upsample": 8},
    )


class TestSyntheticFixture:
    def test_member_shapes(self, members):
        assert members.n_members == 10
        for member in members.members:
            assert member.shape == (32, 36, 32, 32)
            assert member.dtype == np.uint16

    def test_reference_member_is_the_origin(self, members):
        assert tuple(members.nav_offsets[members.reference]) == (0, 0)
        assert tuple(members.dp_offsets[members.reference]) == (0, 0)

    def test_offsets_are_not_proportional(self, members):
        """The fixture's own discriminating power.

        If the scan and detector offsets were multiples of one another, a solver
        that registered the wrong space — or swapped dy for dx — would still
        reproduce both sets of numbers.
        """
        nav = members.nav_offsets.astype(np.float64)
        dp = members.dp_offsets.astype(np.float64)
        assert not np.array_equal(np.sign(nav), np.sign(dp))
        assert not np.array_equal(np.sign(nav), np.sign(dp[:, ::-1]))

    def test_tilts_and_azimuths_match_the_shells(self, members):
        assert members.tilts.shape == (10,)
        assert members.azimuths.shape == (10,)
        assert sorted(members.tilts.tolist()) == sorted(
            [0.0] + [1.0] * 6 + [0.5] * 3)

    def test_explicit_offsets_are_planted(self):
        planted = np.array([[0, 0], [2, -3], [-1, 4]], dtype=np.int64)
        data = make_multiangle(shells=((1.0, 3),), nav_offsets=planted)
        assert np.array_equal(data.nav_offsets, planted)

    def test_noise_free_fixture_is_deterministic(self):
        first = make_multiangle(shells=((1.0, 2),), noise=0.0)
        second = make_multiangle(shells=((1.0, 2),), noise=0.0)
        assert np.array_equal(first.members[1], second.members[1])


class TestRealSpaceAlignment:
    def test_recovers_planted_offsets(self, members, real_space_solution):
        offsets, _ = real_space_solution
        assert np.array_equal(offsets, members.nav_offsets)

    def test_offsets_are_integral(self, real_space_solution):
        offsets, _ = real_space_solution
        assert np.issubdtype(offsets.dtype, np.integer)
        assert offsets.shape == (10, 2)

    def test_residuals_are_reported(self, real_space_solution):
        _, residuals = real_space_solution
        assert residuals.shape == (10, 2)
        # An exactly-integer planted shift leaves nothing behind, but the array
        # must still be there for a caller to report.
        assert np.all(np.abs(residuals) <= 0.5 + 1e-6)

    def test_reference_member_gets_zero(self, members, real_space_solution):
        offsets, _ = real_space_solution
        assert tuple(offsets[members.reference]) == (0, 0)

    def test_non_zero_reference_member(self):
        data = make_multiangle(shells=SHELLS, reference=4)
        offsets, _ = solve_real_space(data.navigators(), reference=4)
        assert tuple(offsets[4]) == (0, 0)
        assert np.array_equal(offsets, data.nav_offsets)

    def test_rejects_reference_out_of_range(self, navigators):
        with pytest.raises(ValueError, match="outside"):
            solve_real_space(navigators, reference=99)

    def test_rejects_a_solver_reference_mode(self, navigators):
        with pytest.raises(ValueError, match="member index"):
            solve_real_space(navigators, reference="running")

    def test_rejects_a_single_image(self):
        with pytest.raises(TypeError, match="3-D"):
            solve_real_space(np.zeros((8, 9)))

    def test_reads_one_frame_at_a_time(self, navigators):
        """The Memory-Safety rule: never compute the whole stack."""
        dask_array = pytest.importorskip("dask.array")
        stack = dask_array.from_array(navigators, chunks=(1,) + navigators.shape[1:])
        whole_stack = {"count": 0}
        real_compute = dask_array.Array.compute

        def guard(self, *args, **kwargs):
            if self.shape == navigators.shape:
                whole_stack["count"] += 1
            return real_compute(self, *args, **kwargs)

        dask_array.Array.compute = guard
        try:
            offsets, _ = solve_real_space(stack, reference=0)
        finally:
            dask_array.Array.compute = real_compute
        assert whole_stack["count"] == 0
        assert offsets.shape == (10, 2)


class TestReciprocalAlignment:
    def test_recovers_planted_offsets(self, members, reciprocal_solution):
        offsets, _ = reciprocal_solution
        assert np.array_equal(offsets, members.dp_offsets)

    def test_offsets_are_integral(self, reciprocal_solution):
        offsets, _ = reciprocal_solution
        assert np.issubdtype(offsets.dtype, np.integer)

    def test_residuals_are_reported(self, reciprocal_solution):
        _, residuals = reciprocal_solution
        assert residuals.shape == (10, 2)
        assert np.all(np.abs(residuals) <= 0.5 + 1e-6)

    def test_non_zero_reference_member(self):
        data = make_multiangle(shells=SHELLS, reference=7)
        offsets, _ = solve_reciprocal(data.mean_patterns(), reference=7)
        assert tuple(offsets[7]) == (0, 0)
        assert np.array_equal(offsets, data.dp_offsets)

    def test_survives_a_larger_beam_displacement(self):
        data = make_multiangle(shells=((1.0, 4),), detector_shape=(48, 48),
                               max_dp_offset=6)
        offsets, _ = solve_reciprocal(data.mean_patterns(), reference=0)
        assert np.array_equal(offsets, data.dp_offsets)


class TestSignConvention:
    """Applying the answer must ALIGN the members, not double the misalignment.

    An inverted sign reproduces a perfectly plausible set of offsets, so these
    tests never compare numbers — they read the members through the offsets and
    look at what comes out.
    """

    def test_applying_the_offsets_aligns_the_members(self, model, navigators):
        scan_shape = navigators.shape[1:]
        reference = navigators[model.reference][
            model.nav_slices(model.reference, scan_shape)]
        for index in range(model.n_members):
            region = navigators[index][model.nav_slices(index, scan_shape)]
            assert _normalised_mismatch(region, reference) < 0.05

    def test_inverted_offsets_do_not_align(self, model, navigators):
        """The same comparison with the sign flipped must FAIL loudly."""
        scan_shape = navigators.shape[1:]
        inverted = MultiAngleModel(
            paths=model.paths,
            tilts=model.tilts,
            azimuths=model.azimuths,
            shell_ids=model.shell_ids,
            nav_offsets=-model.nav_offsets,
            dp_offsets=-model.dp_offsets,
            reference=model.reference,
        )
        reference = navigators[inverted.reference][
            inverted.nav_slices(inverted.reference, scan_shape)]
        worst = max(
            _normalised_mismatch(
                navigators[index][inverted.nav_slices(index, scan_shape)],
                reference)
            for index in range(inverted.n_members)
        )
        assert worst > 0.5

    def test_applying_the_offset_centres_the_direct_beam(self):
        """An odd detector so "the centre" is one pixel, not two."""
        data = make_multiangle(shells=SHELLS, detector_shape=(33, 33),
                               noise=0.0)
        patterns = data.mean_patterns()
        offsets, _ = solve_reciprocal(patterns, reference=data.reference)
        for index in range(data.n_members):
            aligned = np.roll(patterns[index],
                              tuple(int(v) for v in offsets[index]),
                              axis=(0, 1))
            assert np.unravel_index(int(np.argmax(aligned)),
                                    aligned.shape) == (16, 16)

    def test_uncorrected_beams_are_not_already_centred(self):
        """Guards the test above: the fixture must actually displace the beam."""
        data = make_multiangle(shells=SHELLS, detector_shape=(33, 33),
                               noise=0.0)
        patterns = data.mean_patterns()
        peaks = {np.unravel_index(int(np.argmax(pattern)), pattern.shape)
                 for pattern in patterns}
        assert len(peaks) > 1


class TestResiduals:
    """What rounding discarded, on a stack whose true shift is NOT an integer."""

    @staticmethod
    def _subpixel_stack(shifts):
        base = make_multiangle(shells=((0.0, 1),), scan_shape=(64, 72),
                               detector_shape=(8, 8), noise=0.0).images[0]
        height, width = base.shape
        spectrum = np.fft.fft2(base)
        row_frequency = np.fft.fftfreq(height)[:, None]
        column_frequency = np.fft.fftfreq(width)[None, :]
        frames = []
        for row_shift, column_shift in shifts:
            # Displace by -shift, so +shift is the correction.
            ramp = np.exp(2j * np.pi * (row_shift * row_frequency
                                        + column_shift * column_frequency))
            frames.append(np.real(np.fft.ifft2(spectrum * ramp)))
        return np.stack(frames).astype(np.float32)

    def test_offset_plus_residual_is_the_true_shift(self):
        shifts = np.array([[0.0, 0.0], [2.3, -1.4], [-3.7, 0.65]])
        offsets, residuals = solve_real_space(
            self._subpixel_stack(shifts), reference=0)
        assert np.array_equal(offsets, np.rint(shifts).astype(np.int64))
        assert offsets + residuals == pytest.approx(shifts, abs=0.05)

    def test_residual_never_exceeds_half_a_pixel(self):
        shifts = np.array([[0.0, 0.0], [2.3, -1.4], [-3.7, 0.65]])
        _, residuals = solve_real_space(
            self._subpixel_stack(shifts), reference=0)
        assert np.all(np.abs(residuals) <= 0.5 + 1e-6)


class TestShells:
    def test_groups_equal_tilts(self):
        shell_ids = assign_shells([1.0, 1.0, 1.0, 0.5, 0.5])
        assert shell_ids.tolist() == [1, 1, 1, 0, 0]

    def test_numbered_from_the_smallest_tilt_up(self):
        shell_ids = assign_shells([2.5, 0.0, 1.0])
        assert shell_ids.tolist() == [2, 0, 1]

    def test_mixed_member_counts_per_shell(self):
        tilts = [1.0] * 6 + [0.5] * 4
        shell_ids = assign_shells(tilts)
        counts = np.bincount(shell_ids)
        assert counts.tolist() == [4, 6]

    def test_non_uniform_tilt_spacing(self):
        tilts = [0.0, 0.3, 0.3, 1.0, 1.0, 1.0, 2.5]
        shell_ids = assign_shells(tilts)
        assert np.bincount(shell_ids).tolist() == [1, 2, 3, 1]

    def test_tolerance_merges_jittered_tilts(self):
        shell_ids = assign_shells([1.0, 1.005, 0.998])
        assert np.unique(shell_ids).size == 1

    def test_tolerance_does_not_chain(self):
        """Each member is compared with the shell's opening tilt, not its last.

        A run of small steps would otherwise swallow arbitrarily different tilts
        into one shell, and every per-shell average would then be a mixture.
        """
        shell_ids = assign_shells([1.0, 1.008, 1.016])
        assert np.unique(shell_ids).size == 2

    def test_model_groups_members_by_shell(self, model):
        assert model.n_shells == 3
        shells = model.shells
        assert sorted(len(indices) for indices in shells.values()) == [1, 3, 6]
        for shell, indices in shells.items():
            assert {int(model.shell_ids[i]) for i in indices} == {shell}

    def test_a_shell_is_one_tilt_magnitude(self, model):
        for indices in model.shells.values():
            tilts = model.tilts[list(indices)]
            assert np.ptp(tilts) < 1e-9


class TestMultiAngleModel:
    def test_round_trip(self, model, tmp_path):
        path = str(tmp_path / "multiangle.npz")
        model.save(path)
        loaded = MultiAngleModel.load(path)
        assert loaded.paths == model.paths
        assert np.array_equal(loaded.tilts, model.tilts)
        assert np.array_equal(loaded.azimuths, model.azimuths)
        assert np.array_equal(loaded.shell_ids, model.shell_ids)
        assert np.array_equal(loaded.nav_offsets, model.nav_offsets)
        assert np.array_equal(loaded.dp_offsets, model.dp_offsets)
        assert np.array_equal(loaded.nav_residuals, model.nav_residuals)
        assert np.array_equal(loaded.dp_residuals, model.dp_residuals)
        assert loaded.reference == model.reference
        assert loaded.provenance == model.provenance
        assert np.issubdtype(loaded.nav_offsets.dtype, np.integer)

    def test_round_trip_without_residuals(self, model, tmp_path):
        bare = MultiAngleModel(
            paths=model.paths, tilts=model.tilts, azimuths=model.azimuths,
            shell_ids=model.shell_ids, nav_offsets=model.nav_offsets,
            dp_offsets=model.dp_offsets, reference=model.reference)
        path = str(tmp_path / "bare.npz")
        bare.save(path)
        loaded = MultiAngleModel.load(path)
        assert loaded.nav_residuals is None
        assert loaded.dp_residuals is None
        assert loaded.provenance is None

    def test_rejects_an_unknown_format_version(self, model, tmp_path):
        path = str(tmp_path / "future.npz")
        model.save(path)
        with np.load(path, allow_pickle=False) as stored:
            arrays = {name: stored[name] for name in stored.files}
        meta = json.loads(str(arrays["meta"].item()))
        meta["format_version"] = FUTURE_VERSION
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **arrays)
        with pytest.raises(ValueError, match="format version"):
            MultiAngleModel.load(path)

    def test_rejects_non_integer_offsets(self, model):
        offsets = model.nav_offsets.astype(np.float64)
        offsets[1, 0] += 0.5
        with pytest.raises(ValueError, match="whole pixels"):
            MultiAngleModel(
                paths=model.paths, tilts=model.tilts, azimuths=model.azimuths,
                shell_ids=model.shell_ids, nav_offsets=offsets,
                dp_offsets=model.dp_offsets)

    def test_accepts_float_offsets_that_are_already_whole(self, model):
        whole = MultiAngleModel(
            paths=model.paths, tilts=model.tilts, azimuths=model.azimuths,
            shell_ids=model.shell_ids,
            nav_offsets=model.nav_offsets.astype(np.float64),
            dp_offsets=model.dp_offsets)
        assert np.issubdtype(whole.nav_offsets.dtype, np.integer)

    def test_rejects_offset_shape_mismatch(self, model):
        with pytest.raises(ValueError, match="nav_offsets must be"):
            MultiAngleModel(
                paths=model.paths, tilts=model.tilts, azimuths=model.azimuths,
                shell_ids=model.shell_ids,
                nav_offsets=model.nav_offsets[:3],
                dp_offsets=model.dp_offsets)

    def test_rejects_tilt_length_mismatch(self, model):
        with pytest.raises(ValueError, match="tilts must be"):
            MultiAngleModel(
                paths=model.paths, tilts=model.tilts[:3],
                azimuths=model.azimuths, shell_ids=model.shell_ids,
                nav_offsets=model.nav_offsets, dp_offsets=model.dp_offsets)

    def test_rejects_residual_shape_mismatch(self, model):
        with pytest.raises(ValueError, match="nav_residuals must be"):
            MultiAngleModel(
                paths=model.paths, tilts=model.tilts, azimuths=model.azimuths,
                shell_ids=model.shell_ids, nav_offsets=model.nav_offsets,
                dp_offsets=model.dp_offsets,
                nav_residuals=np.zeros((3, 2)))

    def test_rejects_reference_out_of_range(self, model):
        with pytest.raises(ValueError, match="reference"):
            MultiAngleModel(
                paths=model.paths, tilts=model.tilts, azimuths=model.azimuths,
                shell_ids=model.shell_ids, nav_offsets=model.nav_offsets,
                dp_offsets=model.dp_offsets, reference=model.n_members)

    def test_rejects_an_empty_acquisition(self):
        with pytest.raises(ValueError, match="at least one member"):
            MultiAngleModel(
                paths=[], tilts=np.zeros(0), azimuths=np.zeros(0),
                shell_ids=np.zeros(0, dtype=np.int64),
                nav_offsets=np.zeros((0, 2), dtype=np.int64),
                dp_offsets=np.zeros((0, 2), dtype=np.int64))

    def test_rejects_non_finite_offsets(self, model):
        offsets = model.nav_offsets.astype(np.float64)
        offsets[2, 1] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            MultiAngleModel(
                paths=model.paths, tilts=model.tilts, azimuths=model.azimuths,
                shell_ids=model.shell_ids, nav_offsets=offsets,
                dp_offsets=model.dp_offsets)

    def test_max_abs_offsets(self, model):
        assert model.max_abs_nav_offset == int(np.max(np.abs(model.nav_offsets)))
        assert model.max_abs_dp_offset == int(np.max(np.abs(model.dp_offsets)))

    def test_overlap_shape_shrinks_by_the_offset_spread(self, model):
        spread = (model.nav_offsets.max(axis=0) - model.nav_offsets.min(axis=0))
        assert model.overlap_shape((32, 36)) == (
            32 - int(spread[0]), 36 - int(spread[1]))

    def test_overlap_shape_is_never_negative(self, model):
        assert model.overlap_shape((2, 2)) == (0, 0)

    def test_overlap_shape_rejects_a_non_2d_shape(self, model):
        with pytest.raises(ValueError, match="scan shape"):
            model.overlap_shape((32, 36, 32, 32))

    def test_nav_slices_are_in_bounds_and_the_same_size(self, model):
        scan_shape = (32, 36)
        expected = model.overlap_shape(scan_shape)
        for index in range(model.n_members):
            rows, columns = model.nav_slices(index, scan_shape)
            assert rows.start >= 0 and columns.start >= 0
            assert rows.stop <= scan_shape[0] and columns.stop <= scan_shape[1]
            assert (rows.stop - rows.start,
                    columns.stop - columns.start) == expected

    def test_nav_slices_land_on_one_region_of_the_reference_grid(self, model):
        scan_shape = (32, 36)
        origins = {
            (model.nav_slices(index, scan_shape)[0].start
             + int(model.nav_offsets[index, 0]),
             model.nav_slices(index, scan_shape)[1].start
             + int(model.nav_offsets[index, 1]))
            for index in range(model.n_members)
        }
        assert origins == {model.overlap_origin}

    def test_nav_slices_rejects_an_unknown_member(self, model):
        with pytest.raises(ValueError, match="member_index"):
            model.nav_slices(model.n_members, (32, 36))

    def test_repr_names_the_shape_of_the_acquisition(self, model):
        text = repr(model)
        assert "n_members=10" in text and "n_shells=3" in text


class TestDetectorRemap:
    """The reciprocal-space counterpart of the real-space remap.

    The detector is CROPPED to the region every member covers rather than padded
    to keep full extent, so every surviving pixel has all N members behind it.
    A padded border would carry a step in contributor count exactly where vector
    finding and strain read.
    """

    def test_detector_shape_shrinks_by_the_offset_spread(self, model):
        spread = model.dp_offsets.max(axis=0) - model.dp_offsets.min(axis=0)
        assert model.detector_shape((32, 32)) == (
            32 - int(spread[0]), 32 - int(spread[1]))

    def test_detector_shape_is_never_negative(self, model):
        assert model.detector_shape((1, 1)) == (0, 0)

    def test_detector_shape_rejects_a_non_2d_shape(self, model):
        with pytest.raises(ValueError, match="detector shape"):
            model.detector_shape((32, 32, 4))

    def test_detector_slices_are_in_bounds_and_the_same_size(self, model):
        detector_shape = (32, 32)
        expected = model.detector_shape(detector_shape)
        for index in range(model.n_members):
            rows, columns = model.detector_slices(index, detector_shape)
            assert rows.start >= 0 and columns.start >= 0
            assert rows.stop <= detector_shape[0]
            assert columns.stop <= detector_shape[1]
            assert (rows.stop - rows.start,
                    columns.stop - columns.start) == expected

    def test_detector_slices_reject_an_unknown_member(self, model):
        with pytest.raises(ValueError, match="member_index"):
            model.detector_slices(model.n_members, (32, 32))

    def test_the_reference_member_is_cropped_but_not_moved(self, model):
        """The reference has offset (0, 0), so its slice starts at the origin —
        the crop takes the border off every member including this one."""
        rows, columns = model.detector_slices(model.reference, (32, 32))
        assert (rows.start, columns.start) == model.detector_origin

    @staticmethod
    def _beam_centroid(pattern):
        """Intensity-weighted centre of a cropped pattern.

        Not ``argmax``: the synthetic beam is symmetric about a half-integer
        centre, so the brightest pixel is a near-tie that sensor noise breaks
        differently in each member. That measures the tie-break, not the
        alignment. A centroid is stable to well under a pixel.
        """
        values = np.asarray(pattern, dtype=np.float64)
        rows, columns = np.indices(values.shape)
        total = values.sum()
        return (float((rows * values).sum() / total),
                float((columns * values).sum() / total))

    def _centroids(self, members, model):
        return [self._beam_centroid(
                    pattern[model.detector_slices(index, pattern.shape)])
                for index, pattern in enumerate(members.mean_patterns())]

    def test_cropping_aligns_the_direct_beams(self, members, model):
        """The point of the whole exercise: read every member's mean pattern
        through its own detector slice and the beams coincide."""
        centroids = np.asarray(self._centroids(members, model))
        spread = centroids.max(axis=0) - centroids.min(axis=0)
        assert np.all(spread < 0.25), f"beam centroids spread by {spread} px"

    def test_an_inverted_offset_scatters_the_beams(self, members, model):
        """Guards the sign convention the way the real-space tests do: negating
        the offsets must break the alignment, or the test above proves nothing."""
        inverted = MultiAngleModel(
            paths=model.paths, tilts=model.tilts, azimuths=model.azimuths,
            shell_ids=model.shell_ids, nav_offsets=model.nav_offsets,
            dp_offsets=-model.dp_offsets, reference=model.reference)
        centroids = np.asarray(self._centroids(members, inverted))
        spread = centroids.max(axis=0) - centroids.min(axis=0)
        assert np.any(spread > 1.0), (
            f"inverting the offsets left the beams aligned ({spread} px) — "
            "the alignment test above cannot be distinguishing sign")


class TestChoosingBySharpness:
    """`best_real_space` tries several registrations and keeps the sharpest.

    The trap is that "sharpest" is measured on the summed members, and on a
    small or low-contrast image that measurement cannot resolve the last pixel
    — answers a pixel apart score within a percent of each other. Letting the
    highest number win outright turns that noise into a wrong answer, which is
    why the choice has to be decisive before it is taken.
    """

    def _images(self):
        data = make_multiangle(shells=SHELLS)
        return data, np.asarray(data.navigators())

    def test_a_score_that_cannot_decide_leaves_the_offsets_right(self):
        """Noise in the score must not move the answer off the truth.

        Several candidates miss the planted offsets by a pixel here, and a
        score jittered by a few percent will rank one of them top about as
        often as not.
        """
        data, images = self._images()
        jitter = np.random.default_rng(0)

        def score(offsets):
            return 1.0 + 0.03 * jitter.random()

        offsets, _residuals, report = best_real_space(
            images, reference=data.reference, score=score)
        assert np.array_equal(offsets, data.nav_offsets), (
            "a score with no real signal in it changed the answer")
        assert report["decisive"] is False

    def test_a_decisive_score_is_taken(self):
        """The margin must not be a way of ignoring the score entirely."""
        data, images = self._images()
        default, _residuals = solve_real_space(
            images, reference=data.reference, **REGISTRATION_CANDIDATES[0])

        def score(offsets):
            return 1.0 if np.array_equal(offsets, default) else 2.0

        offsets, _residuals, report = best_real_space(
            images, reference=data.reference, score=score)
        assert not np.array_equal(offsets, default)
        assert report["decisive"] is True

    def test_the_prefilters_leave_the_planted_offsets_alone(self):
        """Registering on intensity CHANGE must still find a planted shift.

        The members come from one scene, so a prefilter has nothing to correct
        here — it may not cost accuracy for that.
        """
        data, images = self._images()
        for _label, prefilter in REGISTRATION_PREFILTERS:
            if prefilter is None:
                continue
            prepared = np.stack([prefilter(image) for image in images])
            offsets, _residuals = solve_real_space(
                prepared, reference=data.reference,
                **REGISTRATION_CANDIDATES[0])
            assert np.array_equal(offsets, data.nav_offsets), _label

    def test_every_candidate_is_reported(self):
        data, images = self._images()
        _offsets, _residuals, report = best_real_space(
            images, reference=data.reference, score=lambda o: 1.0)
        assert len(report["attempts"]) == (len(REGISTRATION_PREFILTERS)
                                           * len(REGISTRATION_CANDIDATES))
        assert {named["on"] for named, _gain, _error in report["attempts"]} == \
            {label for label, _prefilter in REGISTRATION_PREFILTERS}
