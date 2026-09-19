"""
Loading multi-angle members already aligned, straight off the disk.

The claim under test is the one the whole design rests on: alignment costs
nothing because it is a choice about WHICH FRAME to read, not a shift applied to
frames that were already read. So these tests write real binary files, load them
through the aligned path, and compare against the frames a naive
"load-everything-then-slice" would have produced. They must agree exactly.

The second claim is that members loaded this way share one chunk grid, so
composing them introduces no rechunk. That one is pinned by inspecting the
graph, because it is invisible in the values.
"""
import numpy as np
import pytest

import dask.array as da

from spyde.multiangle.compose import stack_aligned, sum_aligned
from spyde.multiangle.load import (
    frame_chunked_source, load_aligned_store_member,
    aligned_member_chunks, frame_index_map, load_aligned_member,
    load_aligned_members,
)
from spyde.multiangle.model import MultiAngleModel

SCAN = (12, 16)
DETECTOR = (6, 8)
NAV_CHUNK = 4
N_MEMBERS = 4


@pytest.fixture
def written_members(tmp_path):
    """Four members on disk, each frame stamped with its member and position so
    a frame read from the wrong place is a wrong NUMBER, not a plausible one."""
    nav_offsets = np.array([(0, 0), (2, -1), (-3, 2), (1, 3)])
    dp_offsets = np.array([(0, 0), (1, 1), (-2, 0), (0, -1)])
    model = MultiAngleModel(
        paths=[str(tmp_path / f"member{index}.raw") for index in range(N_MEMBERS)],
        tilts=np.array([1.0, 1.0, 0.5, 0.5]),
        azimuths=np.array([0.0, 180.0, 90.0, 270.0]),
        shell_ids=np.array([0, 0, 1, 1]),
        nav_offsets=nav_offsets,
        dp_offsets=dp_offsets,
    )

    generator = np.random.default_rng(99)
    arrays = []
    for member_index, path in enumerate(model.paths):
        array = generator.integers(
            0, 9000, SCAN + DETECTOR, dtype=np.uint16)
        # A per-position stamp on top of noise: a misread frame cannot coincide.
        for row in range(SCAN[0]):
            for column in range(SCAN[1]):
                array[row, column, 0, 0] = (
                    member_index * 10000 + row * 100 + column)
        array.reshape(-1, *DETECTOR).tofile(path)
        arrays.append(array)
    return model, arrays


def _expected(array, model, member_index):
    scan_rows, scan_columns = model.nav_slices(member_index, SCAN)
    detector_rows, detector_columns = model.detector_slices(
        member_index, DETECTOR)
    return array[scan_rows, scan_columns, detector_rows, detector_columns]


class TestFrameIndexMap:
    def test_each_entry_is_the_flat_index_of_the_right_frame(
            self, written_members):
        model, _arrays = written_members
        for member_index in range(N_MEMBERS):
            scan_rows, scan_columns = model.nav_slices(member_index, SCAN)
            index_map = frame_index_map(model, member_index, SCAN)
            assert index_map.shape == model.overlap_shape(SCAN)
            for row in range(index_map.shape[0]):
                for column in range(index_map.shape[1]):
                    member_row = scan_rows.start + row
                    member_column = scan_columns.start + column
                    assert index_map[row, column] == (
                        member_row * SCAN[1] + member_column)

    def test_the_map_never_points_outside_the_member(self, written_members):
        model, _arrays = written_members
        for member_index in range(N_MEMBERS):
            index_map = frame_index_map(model, member_index, SCAN)
            assert index_map.min() >= 0
            assert index_map.max() < SCAN[0] * SCAN[1]

    def test_the_reference_member_reads_its_own_crop(self, written_members):
        """With offset (0, 0) the reference is still cropped — every member
        loses the margin the others do not cover."""
        model, _arrays = written_members
        scan_rows, _ = model.nav_slices(model.reference, SCAN)
        index_map = frame_index_map(model, model.reference, SCAN)
        assert index_map[0, 0] == scan_rows.start * SCAN[1] + \
            model.nav_slices(model.reference, SCAN)[1].start


class TestAlignedLoad:
    def test_a_member_loads_exactly_the_frames_it_should(self, written_members):
        model, arrays = written_members
        for member_index, path in enumerate(model.paths):
            loaded = load_aligned_member(
                path, model, member_index, dtype=np.uint16,
                member_scan_shape=SCAN, detector_shape=DETECTOR,
                nav_chunk=NAV_CHUNK).compute()
            assert np.array_equal(
                loaded, _expected(arrays[member_index], model, member_index)), (
                    f"member {member_index} read the wrong frames")

    def test_the_position_stamp_confirms_the_remap(self, written_members):
        """Read the stamp back: output (0, 0) must carry the member position the
        alignment says belongs there, not the member's own (0, 0)."""
        model, _arrays = written_members
        for member_index, path in enumerate(model.paths):
            scan_rows, scan_columns = model.nav_slices(member_index, SCAN)
            loaded = load_aligned_member(
                path, model, member_index, dtype=np.uint16,
                member_scan_shape=SCAN, detector_shape=DETECTOR,
                nav_chunk=NAV_CHUNK)
            detector_rows, detector_columns = model.detector_slices(
                member_index, DETECTOR)
            # The stamp lives at detector (0, 0), which the crop may have moved.
            if detector_rows.start == 0 and detector_columns.start == 0:
                stamp = int(loaded[0, 0, 0, 0].compute())
                assert stamp == (member_index * 10000
                                 + scan_rows.start * 100 + scan_columns.start)

    def test_a_file_with_a_header_offset_still_reads_correctly(self, tmp_path):
        model = MultiAngleModel(
            paths=[str(tmp_path / "with_header.raw")],
            tilts=np.zeros(1), azimuths=np.zeros(1),
            shell_ids=np.zeros(1, dtype=np.int64),
            nav_offsets=np.zeros((1, 2), dtype=np.int64),
            dp_offsets=np.zeros((1, 2), dtype=np.int64))
        array = np.random.default_rng(3).integers(
            0, 5000, SCAN + DETECTOR, dtype=np.uint16)
        header = np.full(37, 7, dtype=np.uint8)
        with open(model.paths[0], "wb") as handle:
            handle.write(header.tobytes())
            handle.write(array.reshape(-1, *DETECTOR).tobytes())
        loaded = load_aligned_member(
            model.paths[0], model, 0, dtype=np.uint16, member_scan_shape=SCAN,
            detector_shape=DETECTOR, nav_chunk=NAV_CHUNK,
            offset=header.nbytes).compute()
        assert np.array_equal(loaded, array)

    def test_every_member_comes_back_on_one_chunk_grid(self, written_members):
        model, _arrays = written_members
        members = load_aligned_members(
            model.paths, model, dtype=np.uint16, member_scan_shape=SCAN,
            detector_shape=DETECTOR, nav_chunk=NAV_CHUNK)
        grids = {member.chunks for member in members}
        assert len(grids) == 1

    def test_the_grid_is_the_one_we_chose_not_the_members(self, written_members):
        model, _arrays = written_members
        members = load_aligned_members(
            model.paths, model, dtype=np.uint16, member_scan_shape=SCAN,
            detector_shape=DETECTOR, nav_chunk=NAV_CHUNK)
        expected = aligned_member_chunks(
            model, SCAN, DETECTOR, NAV_CHUNK, np.uint16)
        assert members[0].chunks[:2] == expected[:2]
        assert all(len(axis) == 1 for axis in members[0].chunks[-2:])

    def test_a_path_count_mismatch_is_refused(self, written_members):
        model, _arrays = written_members
        with pytest.raises(ValueError, match="describes"):
            load_aligned_members(
                model.paths[:-1], model, dtype=np.uint16,
                member_scan_shape=SCAN, detector_shape=DETECTOR)


class TestComposingAlignedMembers:
    def test_composing_needs_no_rechunk(self, written_members):
        """The whole point: members born on one grid combine with no repair."""
        model, _arrays = written_members
        members = load_aligned_members(
            model.paths, model, dtype=np.uint16, member_scan_shape=SCAN,
            detector_shape=DETECTOR, nav_chunk=NAV_CHUNK)
        for composed in (sum_aligned(members),
                         stack_aligned(members)):
            offenders = [name for name in composed.dask.layers
                         if "rechunk" in name or "shuffle" in name]
            assert offenders == []

    def test_the_composed_sum_matches_the_hand_computed_one(
            self, written_members):
        model, arrays = written_members
        members = load_aligned_members(
            model.paths, model, dtype=np.uint16, member_scan_shape=SCAN,
            detector_shape=DETECTOR, nav_chunk=NAV_CHUNK)
        composed = sum_aligned(members).compute()

        expected = np.zeros(composed.shape, dtype=np.uint32)
        for member_index, array in enumerate(arrays):
            expected += _expected(array, model, member_index)
        assert np.array_equal(composed, expected)

    def test_the_composed_grid_stays_the_chosen_one(self, written_members):
        model, _arrays = written_members
        members = load_aligned_members(
            model.paths, model, dtype=np.uint16, member_scan_shape=SCAN,
            detector_shape=DETECTOR, nav_chunk=NAV_CHUNK)
        composed = sum_aligned(members)
        assert all(max(axis) <= NAV_CHUNK for axis in composed.chunks[:2])
        assert all(len(axis) == 1 for axis in composed.chunks[-2:])


class TestMemorySafety:
    def test_building_the_aligned_load_reads_nothing(self, written_members):
        model, _arrays = written_members
        computed = []
        original = da.Array.compute
        da.Array.compute = lambda self, *a, **k: computed.append(self.shape)
        try:
            load_aligned_members(
                model.paths, model, dtype=np.uint16, member_scan_shape=SCAN,
                detector_shape=DETECTOR, nav_chunk=NAV_CHUNK)
        finally:
            da.Array.compute = original
        assert computed == []

    def test_one_frame_does_not_materialise_the_member(self, written_members):
        model, _arrays = written_members
        member = load_aligned_member(
            model.paths[0], model, 0, dtype=np.uint16, member_scan_shape=SCAN,
            detector_shape=DETECTOR, nav_chunk=NAV_CHUNK)
        shapes = []
        original = da.Array.compute

        def record(self, *args, **kwargs):
            shapes.append(self.shape)
            return original(self, *args, **kwargs)

        da.Array.compute = record
        try:
            member[2, 2].compute()
        finally:
            da.Array.compute = original
        assert member.shape not in shapes


class TestFrameChunkedStore:
    """A `.zspy`/`.hspy` compressed FRAME BY FRAME can be aligned the same way.

    It has no flat run of frames to index, but it does not need one: with a
    chunk per frame every dask grid is equally cheap to read, so the member is
    rebuilt on a grid whose blocks line up after cropping. What must NOT happen
    is one dask chunk per frame — that would put one task per scan position in
    the graph.
    """

    @staticmethod
    def _store(tmp_path, name, array, nav_chunk):
        import zarr

        store = zarr.open(
            str(tmp_path / name), mode="w", shape=array.shape,
            chunks=(nav_chunk, nav_chunk) + array.shape[2:], dtype=array.dtype)
        store[:] = array
        return store

    def _members(self, tmp_path, written_members, nav_chunk=1):
        model, arrays = written_members
        members = [
            da.from_array(self._store(tmp_path, f"store{index}.zarr",
                                      array, nav_chunk))
            for index, array in enumerate(arrays)
        ]
        return model, arrays, members

    def test_a_frame_chunked_store_is_recognised(self, tmp_path,
                                                 written_members):
        _model, _arrays, members = self._members(tmp_path, written_members)
        assert frame_chunked_source(members[0]) is not None

    def test_a_nav_chunked_store_is_not(self, tmp_path, written_members):
        """Big navigation chunks mean an arbitrary frame is NOT cheap, so this
        path must decline rather than straddle storage chunks."""
        _model, _arrays, members = self._members(
            tmp_path, written_members, nav_chunk=4)
        assert frame_chunked_source(members[0]) is None

    def test_a_plain_dask_array_is_not_a_store(self, written_members):
        _model, arrays = written_members
        assert frame_chunked_source(da.from_array(arrays[0])) is None

    def test_the_aligned_store_member_reads_the_right_frames(
            self, tmp_path, written_members):
        model, arrays, members = self._members(tmp_path, written_members)
        for member_index, member in enumerate(members):
            aligned = load_aligned_store_member(
                member, model, member_index, nav_chunk=NAV_CHUNK)
            assert aligned is not None
            assert np.array_equal(
                aligned.compute(),
                _expected(arrays[member_index], model, member_index))

    def test_it_agrees_with_the_binary_path(self, tmp_path, written_members):
        """Two routes to the same answer — they must not disagree."""
        model, _arrays, members = self._members(tmp_path, written_members)
        for member_index, member in enumerate(members):
            from_store = load_aligned_store_member(
                member, model, member_index, nav_chunk=NAV_CHUNK).compute()
            from_binary = load_aligned_member(
                model.paths[member_index], model, member_index,
                dtype=np.uint16, member_scan_shape=SCAN,
                detector_shape=DETECTOR, nav_chunk=NAV_CHUNK).compute()
            assert np.array_equal(from_store, from_binary)

    def test_blocks_are_not_one_frame_each(self, tmp_path, written_members):
        """The store is chunked per frame; the DASK grid must not copy that."""
        model, _arrays, members = self._members(tmp_path, written_members)
        aligned = load_aligned_store_member(
            members[0], model, 0, nav_chunk=NAV_CHUNK)
        assert max(aligned.chunks[0]) > 1
        assert aligned.npartitions < np.prod(aligned.shape[:2])

    def test_every_member_lands_on_the_same_grid(self, tmp_path,
                                                 written_members):
        model, _arrays, members = self._members(tmp_path, written_members)
        aligned = [load_aligned_store_member(member, model, index,
                                             nav_chunk=NAV_CHUNK)
                   for index, member in enumerate(members)]
        assert len({member.chunks for member in aligned}) == 1

    def test_composing_them_needs_no_rechunk(self, tmp_path, written_members):
        model, _arrays, members = self._members(tmp_path, written_members)
        aligned = [load_aligned_store_member(member, model, index,
                                             nav_chunk=NAV_CHUNK)
                   for index, member in enumerate(members)]
        composed = sum_aligned(aligned)
        offenders = [name for name in composed.dask.layers
                     if "rechunk" in name or "shuffle" in name]
        assert offenders == []

    def test_the_composed_sum_is_right(self, tmp_path, written_members):
        model, arrays, members = self._members(tmp_path, written_members)
        aligned = [load_aligned_store_member(member, model, index,
                                             nav_chunk=NAV_CHUNK)
                   for index, member in enumerate(members)]
        expected = np.zeros(
            model.overlap_shape(SCAN) + model.detector_shape(DETECTOR),
            dtype=np.uint32)
        for member_index, array in enumerate(arrays):
            expected += _expected(array, model, member_index)
        assert np.array_equal(sum_aligned(aligned).compute(), expected)


class TestAlignedFromAnOpenedSignal:
    """The loader opens members through HyperSpy, then re-expresses them as
    aligned reads of their own files.

    This is the path a real `.mrc`/`.de5` acquisition takes, so these tests
    build members exactly the way RosettaSciIO's distributed memmap does and
    assert the alignment really is done by the READ — not by cropping afterwards.
    """

    @staticmethod
    def _opened(path, scan_shape, detector_shape, nav_chunk=4):
        """A lazy signal shaped like one RosettaSciIO hands back for a binary file."""
        from hyperspy._signals.signal2d import LazySignal2D
        from rsciio.utils._distributed import memmap_distributed

        return LazySignal2D(memmap_distributed(
            str(path), np.dtype(np.uint16),
            shape=tuple(scan_shape) + tuple(detector_shape),
            chunks=(nav_chunk, nav_chunk, -1, -1)))

    def _members(self, written_members):
        model, arrays = written_members
        return model, arrays, [self._opened(path, SCAN, DETECTOR)
                               for path in model.paths]

    def test_a_memmap_backed_member_takes_the_aligned_read(self,
                                                           written_members):
        from spyde.multiangle.load import aligned_from_signal

        model, _arrays, members = self._members(written_members)
        for index, member in enumerate(members):
            assert aligned_from_signal(
                member, model, index, nav_chunk=NAV_CHUNK) is not None

    def test_it_reads_the_same_frames_as_cropping_would(self, written_members):
        from spyde.multiangle.load import aligned_from_signal

        model, arrays, members = self._members(written_members)
        for index, member in enumerate(members):
            aligned = aligned_from_signal(
                member, model, index, nav_chunk=NAV_CHUNK)
            assert np.array_equal(aligned.compute(),
                                  _expected(arrays[index], model, index))

    def test_every_member_lands_on_one_grid(self, written_members):
        from spyde.multiangle.load import aligned_from_signal

        model, _arrays, members = self._members(written_members)
        aligned = [aligned_from_signal(member, model, index,
                                       nav_chunk=NAV_CHUNK)
                   for index, member in enumerate(members)]
        assert len({member.chunks for member in aligned}) == 1

    def test_composing_them_carries_no_rechunk(self, written_members):
        """The point of doing it at read time rather than after."""
        from spyde.multiangle.load import aligned_from_signal

        model, _arrays, members = self._members(written_members)
        aligned = [aligned_from_signal(member, model, index,
                                       nav_chunk=NAV_CHUNK)
                   for index, member in enumerate(members)]
        composed = sum_aligned(aligned)
        assert [name for name in composed.dask.layers
                if "rechunk" in name or "shuffle" in name] == []

    def test_a_member_that_is_not_memmap_backed_declines(self,
                                                         written_members):
        from spyde.multiangle.load import aligned_from_signal

        model, arrays, _members = self._members(written_members)
        plain = da.from_array(arrays[0], chunks=(NAV_CHUNK, NAV_CHUNK, -1, -1))
        assert aligned_from_signal(plain, model, 0) is None

    def test_the_session_loader_uses_it(self, written_members):
        """Wiring check: `_aligned_arrays` must PREFER the aligned read.

        The composed grid is the visible difference — cropping after loading
        leaves it the union of the members' boundaries.
        """
        from spyde.backend._session_multiangle import _aligned_arrays

        model, _arrays, members = self._members(written_members)
        aligned = _aligned_arrays(members, model)
        assert len({member.chunks for member in aligned}) == 1
        assert [name for name in sum_aligned(aligned).dask.layers
                if "rechunk" in name or "shuffle" in name] == []
