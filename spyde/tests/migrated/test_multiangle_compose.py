"""
Composing aligned members into the 5-D stack and the 4-D sum.

The contract under test is that composition is nothing but slicing: the result
must equal the hand-computed sum of the shifted members, and building it must
not read a single member. The second half is the Memory-Safety rule from
CLAUDE.md — these arrays are tens of gigabytes in the field, and a stray
``.compute()`` on one would take the machine down rather than run slowly.
"""
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest

from spyde.multiangle.compose import (
    aligned_load_chunks, aligned_member, compose_stack, compose_sum,
    composed_axis_offsets, composed_nav_chunks, member_nav_chunks,
    sum_dtype, _aligned_members,
)
from spyde.multiangle.model import MultiAngleModel

SCAN = (24, 28)
DETECTOR = (16, 16)
NAV_CHUNK = 8


def _model(nav_offsets, dp_offsets):
    n_members = len(nav_offsets)
    return MultiAngleModel(
        paths=[f"member{index}.mrc" for index in range(n_members)],
        tilts=np.zeros(n_members),
        azimuths=np.linspace(0.0, 360.0, n_members, endpoint=False),
        shell_ids=np.zeros(n_members, dtype=np.int64),
        nav_offsets=np.asarray(nav_offsets),
        dp_offsets=np.asarray(dp_offsets),
    )


@pytest.fixture
def members_and_model():
    """Four members whose values identify the member AND the position, so a
    misplaced slice shows up as a wrong number rather than a plausible one."""
    nav_offsets = [(0, 0), (2, -1), (-3, 2), (1, 3)]
    dp_offsets = [(0, 0), (1, 1), (-2, 0), (0, -1)]
    model = _model(nav_offsets, dp_offsets)

    generator = np.random.default_rng(1234)
    raw = [generator.integers(0, 500, SCAN + DETECTOR, dtype=np.uint16)
           for _ in nav_offsets]
    members = [da.from_array(array, chunks=(NAV_CHUNK, NAV_CHUNK, -1, -1))
               for array in raw]
    return members, raw, model


class TestSumDtype:
    def test_unsigned_widens_far_enough_to_never_overflow(self):
        dtype = sum_dtype(np.uint16, 10)
        assert np.iinfo(dtype).max >= np.iinfo(np.uint16).max * 10

    def test_a_huge_member_count_still_finds_an_accumulator(self):
        dtype = sum_dtype(np.uint16, 100_000)
        assert np.iinfo(dtype).max >= np.iinfo(np.uint16).max * 100_000

    def test_signed_stays_signed(self):
        assert np.dtype(sum_dtype(np.int16, 8)).kind == "i"

    def test_float32_is_left_alone(self):
        assert sum_dtype(np.float32, 10) == np.dtype(np.float32)

    def test_half_precision_is_widened(self):
        assert sum_dtype(np.float16, 10) == np.dtype(np.float32)

    def test_a_non_numeric_dtype_is_refused(self):
        with pytest.raises(TypeError):
            sum_dtype(np.dtype("U4"), 4)


class TestAlignment:
    def test_the_sum_equals_the_hand_shifted_members(self, members_and_model):
        members, raw, model = members_and_model
        composed = compose_sum(members, model).compute()

        expected = np.zeros(composed.shape, dtype=np.uint32)
        for index, array in enumerate(raw):
            scan_rows, scan_columns = model.nav_slices(index, SCAN)
            detector_rows, detector_columns = model.detector_slices(
                index, DETECTOR)
            expected += array[scan_rows, scan_columns,
                              detector_rows, detector_columns]
        assert np.array_equal(composed, expected)

    def test_each_angle_of_the_stack_is_that_member(self, members_and_model):
        members, raw, model = members_and_model
        stack = compose_stack(members, model).compute()
        for index, array in enumerate(raw):
            scan_rows, scan_columns = model.nav_slices(index, SCAN)
            detector_rows, detector_columns = model.detector_slices(
                index, DETECTOR)
            assert np.array_equal(
                stack[index],
                array[scan_rows, scan_columns, detector_rows, detector_columns])

    def test_the_stack_summed_over_angle_is_the_sum(self, members_and_model):
        """The two compositions are two views of one thing, so they must agree."""
        members, _raw, model = members_and_model
        stack = compose_stack(members, model)
        assert np.array_equal(
            stack.sum(axis=0, dtype=np.uint32).compute(),
            compose_sum(members, model).compute())

    def test_the_composed_shape_is_the_common_region(self, members_and_model):
        members, _raw, model = members_and_model
        composed = compose_sum(members, model)
        assert composed.shape[:2] == model.overlap_shape(SCAN)
        assert composed.shape[2:] == model.detector_shape(DETECTOR)

    def test_aligned_member_is_a_pure_slice_of_its_member(self, members_and_model):
        members, raw, model = members_and_model
        for index in range(model.n_members):
            scan_rows, scan_columns = model.nav_slices(index, SCAN)
            detector_rows, detector_columns = model.detector_slices(
                index, DETECTOR)
            assert np.array_equal(
                aligned_member(members[index], model, index).compute(),
                raw[index][scan_rows, scan_columns,
                           detector_rows, detector_columns])


class TestChunking:
    """Alignment is done at LOAD time, so nothing is ever rechunked.

    Each member is cropped from a different start, so members loaded on one
    uniform grid have their boundaries knocked out of step and summing them
    makes dask unify to the union of all of them. Giving each member a first
    chunk exactly as long as the margin its crop discards puts every later
    boundary on the same grid — for free, because a lazy load only builds a
    graph. Rechunking afterwards would move bytes at compute time instead.
    """

    def test_the_signal_axes_span_the_full_detector(self, members_and_model):
        members, _raw, model = members_and_model
        for composed in (compose_sum(members, model),
                         compose_stack(members, model)):
            assert all(len(axis) == 1 for axis in composed.chunks[-2:])

    def test_load_time_chunks_give_every_member_the_same_grid(
            self, members_and_model):
        _members, raw, model = members_and_model
        tailored = [da.from_array(array, chunks=spec)
                    for array, spec in
                    zip(raw, aligned_load_chunks(model, SCAN, NAV_CHUNK))]
        assert composed_nav_chunks(_aligned_members(tailored, model)) is not None

    def test_load_time_chunks_leave_no_rechunk_in_the_graph(
            self, members_and_model):
        _members, raw, model = members_and_model
        tailored = [da.from_array(array, chunks=spec)
                    for array, spec in
                    zip(raw, aligned_load_chunks(model, SCAN, NAV_CHUNK))]
        composed = compose_sum(tailored, model)
        offenders = [name for name in composed.dask.layers
                     if "rechunk" in name or "shuffle" in name]
        assert offenders == []

    def test_a_uniform_load_is_what_fragments_the_grid(self, members_and_model):
        """The failure this exists to avoid, pinned so nobody reintroduces it."""
        members, raw, model = members_and_model
        uniform = compose_sum(members, model)
        tailored = compose_sum(
            [da.from_array(array, chunks=spec) for array, spec in
             zip(raw, aligned_load_chunks(model, SCAN, NAV_CHUNK))], model)
        assert tailored.npartitions < uniform.npartitions
        assert all(max(axis) <= NAV_CHUNK for axis in tailored.chunks[:2])

    def test_the_chunking_does_not_change_the_values(self, members_and_model):
        members, raw, model = members_and_model
        tailored = [da.from_array(array, chunks=spec)
                    for array, spec in
                    zip(raw, aligned_load_chunks(model, SCAN, NAV_CHUNK))]
        assert np.array_equal(compose_sum(members, model).compute(),
                              compose_sum(tailored, model).compute())

    def test_a_member_spec_covers_its_whole_extent(self, members_and_model):
        _members, _raw, model = members_and_model
        for member_index in range(model.n_members):
            rows, columns = member_nav_chunks(
                model, member_index, SCAN, NAV_CHUNK)
            assert sum(rows) == SCAN[0]
            assert sum(columns) == SCAN[1]

    def test_the_discarded_margin_is_its_own_leading_chunk(
            self, members_and_model):
        """The head chunk IS the crop's offset, which is what makes every
        boundary after it land on the common grid."""
        _members, _raw, model = members_and_model
        for member_index in range(model.n_members):
            scan_rows, scan_columns = model.nav_slices(member_index, SCAN)
            rows, columns = member_nav_chunks(
                model, member_index, SCAN, NAV_CHUNK)
            if scan_rows.start:
                assert rows[0] == scan_rows.start
            if scan_columns.start:
                assert columns[0] == scan_columns.start

    def test_a_nonsense_chunk_size_is_refused(self, members_and_model):
        _members, _raw, model = members_and_model
        with pytest.raises(ValueError, match="nav_chunk"):
            member_nav_chunks(model, 0, SCAN, 0)

    def test_the_stack_keeps_one_chunk_per_angle(self, members_and_model):
        """Scrubbing to angle *a* must touch member *a* and nothing else."""
        members, _raw, model = members_and_model
        assert compose_stack(members, model).chunks[0] == (1,) * len(members)


class TestMemorySafety:
    """Building a composition must never read a member (CLAUDE.md)."""

    def test_composing_reads_nothing(self, members_and_model):
        members, _raw, model = members_and_model
        with patch.object(da.Array, "compute",
                          side_effect=AssertionError(
                              "composition computed a dask array")):
            compose_stack(members, model)
            compose_sum(members, model)

    def test_reading_one_frame_does_not_materialise_the_dataset(
            self, members_and_model):
        members, _raw, model = members_and_model
        composed = compose_sum(members, model)
        computed_shapes = []
        original = da.Array.compute

        def record(self, *args, **kwargs):
            computed_shapes.append(self.shape)
            return original(self, *args, **kwargs)

        with patch.object(da.Array, "compute", record):
            composed[3, 4].compute()
        assert composed.shape not in computed_shapes


class TestGuards:
    def test_a_member_count_mismatch_is_refused(self, members_and_model):
        members, _raw, model = members_and_model
        with pytest.raises(ValueError, match="describes"):
            compose_sum(members[:-1], model)

    def test_members_that_do_not_overlap_are_refused(self):
        model = _model([(0, 0), (40, 0)], [(0, 0), (0, 0)])
        members = [da.zeros(SCAN + DETECTOR, dtype=np.uint16) for _ in range(2)]
        with pytest.raises(ValueError, match="no common region"):
            compose_sum(members, model)

    def test_a_member_that_is_not_4d_is_refused(self, members_and_model):
        members, _raw, model = members_and_model
        with pytest.raises(ValueError, match="4-D"):
            compose_sum([members[0][0]] + list(members[1:]), model)


class TestComposedAxisOffsets:
    def test_the_origins_move_by_the_crop(self, members_and_model):
        _members, _raw, model = members_and_model
        scan_origin, detector_origin = composed_axis_offsets(
            model, scan_offsets=(0.0, 0.0), scan_scales=(2.0, 2.0),
            detector_offsets=(-1.0, -1.0), detector_scales=(0.1, 0.1))
        assert scan_origin == (model.overlap_origin[0] * 2.0,
                               model.overlap_origin[1] * 2.0)
        assert detector_origin == pytest.approx(
            (-1.0 + model.detector_origin[0] * 0.1,
             -1.0 + model.detector_origin[1] * 0.1))

    def test_an_uncropped_acquisition_leaves_the_origins_alone(self):
        model = _model([(0, 0), (0, 0)], [(0, 0), (0, 0)])
        scan_origin, detector_origin = composed_axis_offsets(
            model, scan_offsets=(5.0, 6.0), scan_scales=(2.0, 2.0),
            detector_offsets=(-1.0, -1.0), detector_scales=(0.1, 0.1))
        assert scan_origin == (5.0, 6.0)
        assert detector_origin == (-1.0, -1.0)
