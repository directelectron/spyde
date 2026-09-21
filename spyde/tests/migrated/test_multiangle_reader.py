"""
The multi-angle display reader must agree with the dask composition EXACTLY.

The reader exists only because asking dask for one composed frame pulls a
navigation chunk out of every member. It is therefore a second implementation of
the same answer, and the only thing that keeps a second implementation honest is
a parity test against the first. ``array_equal``, never ``allclose``: these are
integer counts read through an index remap, so a frame that is merely close is a
frame read from the wrong place.
"""
import dask.array as da
import hyperspy.api as hs
import numpy as np
import pytest

from spyde.array_cache.readers.multiangle import build_multiangle_reader
from spyde.multiangle.compose import (
    aligned_member, compose_stack, compose_sum, sum_dtype,
)
from spyde.multiangle.model import MultiAngleModel
from spyde.multiangle.recipe import MultiAngleRecipe, attach_recipe, recipe_for

SCAN = (20, 24)
DETECTOR = (12, 12)
NAV_CHUNK = 8
N_MEMBERS = 4


@pytest.fixture
def acquisition():
    """Four lazy members with distinct content and non-trivial offsets."""
    nav_offsets = np.array([(0, 0), (2, -1), (-3, 2), (1, 3)])
    dp_offsets = np.array([(0, 0), (1, 1), (-2, 0), (0, -1)])
    model = MultiAngleModel(
        paths=[f"member{index}.mrc" for index in range(N_MEMBERS)],
        tilts=np.array([1.0, 1.0, 0.5, 0.5]),
        azimuths=np.array([0.0, 180.0, 90.0, 270.0]),
        shell_ids=np.array([0, 0, 1, 1]),
        nav_offsets=nav_offsets,
        dp_offsets=dp_offsets,
    )

    generator = np.random.default_rng(7)
    members = []
    for _ in range(N_MEMBERS):
        raw = generator.integers(0, 4000, SCAN + DETECTOR, dtype=np.uint16)
        signal = hs.signals.Signal2D(raw).as_lazy()
        signal.data = da.from_array(raw, chunks=(NAV_CHUNK, NAV_CHUNK, -1, -1))
        members.append(signal)
    return members, model


def _composed_signal(array, members, model, *, has_angle_axis, dtype=None):
    signal = hs.signals.Signal2D(np.zeros((1, 1, 1, 1), dtype=np.uint8)).as_lazy()
    signal.data = array
    return attach_recipe(signal, MultiAngleRecipe(
        members=tuple(members), model=model,
        has_angle_axis=has_angle_axis, dtype=dtype))


class TestSummedParity:
    def test_every_position_matches_the_dask_composition(self, acquisition):
        members, model = acquisition
        composed = compose_sum(members, model)
        signal = _composed_signal(composed, members, model,
                                  has_angle_axis=False, dtype=composed.dtype)
        reader = build_multiangle_reader(signal, composed)
        assert reader is not None

        reference = composed.compute()
        for scan_row in range(composed.shape[0]):
            for scan_column in range(composed.shape[1]):
                assert np.array_equal(
                    reader.read_frame((scan_row, scan_column)),
                    reference[scan_row, scan_column]), (
                        f"mismatch at ({scan_row}, {scan_column})")

    def test_the_frame_dtype_is_the_composed_dtype(self, acquisition):
        members, model = acquisition
        composed = compose_sum(members, model)
        signal = _composed_signal(composed, members, model,
                                  has_angle_axis=False, dtype=composed.dtype)
        reader = build_multiangle_reader(signal, composed)
        frame = reader.read_frame((2, 2))
        assert frame.dtype == composed.dtype == sum_dtype(np.uint16, N_MEMBERS)

    def test_the_sum_does_not_overflow_the_member_type(self, acquisition):
        """Four members near uint16 max must exceed it, not wrap to a small
        number — the failure a narrow accumulator would produce silently."""
        members, model = acquisition
        for member in members:
            member.data = da.full(SCAN + DETECTOR, 60000, dtype=np.uint16,
                                  chunks=(NAV_CHUNK, NAV_CHUNK, -1, -1))
        composed = compose_sum(members, model)
        signal = _composed_signal(composed, members, model,
                                  has_angle_axis=False, dtype=composed.dtype)
        reader = build_multiangle_reader(signal, composed)
        assert reader.read_frame((2, 2))[0, 0] == 60000 * N_MEMBERS

    def test_frame_bytes_describes_the_composed_frame(self, acquisition):
        members, model = acquisition
        composed = compose_sum(members, model)
        signal = _composed_signal(composed, members, model,
                                  has_angle_axis=False, dtype=composed.dtype)
        reader = build_multiangle_reader(signal, composed)
        assert reader.frame_bytes == (
            int(np.prod(composed.shape[2:])) * composed.dtype.itemsize)


class TestStackParity:
    def test_every_angle_and_position_matches(self, acquisition):
        members, model = acquisition
        composed = compose_stack(members, model)
        signal = _composed_signal(composed, members, model, has_angle_axis=True)
        reader = build_multiangle_reader(signal, composed)
        assert reader is not None

        reference = composed.compute()
        for angle in range(composed.shape[0]):
            for scan_row in (0, composed.shape[1] // 2, composed.shape[1] - 1):
                for scan_column in (0, composed.shape[2] - 1):
                    assert np.array_equal(
                        reader.read_frame((angle, scan_row, scan_column)),
                        reference[angle, scan_row, scan_column]), (
                            f"mismatch at angle {angle}, "
                            f"({scan_row}, {scan_column})")

    def test_an_angle_outside_the_members_is_refused(self, acquisition):
        members, model = acquisition
        composed = compose_stack(members, model)
        signal = _composed_signal(composed, members, model, has_angle_axis=True)
        reader = build_multiangle_reader(signal, composed)
        with pytest.raises(IndexError):
            reader.read_frame((N_MEMBERS, 0, 0))

    def test_each_angle_reads_only_its_own_member(self, acquisition):
        """The point of the stack node: angle *a* is member *a*, untouched by
        the others. A frame equal to another member's would mean a lost offset."""
        members, model = acquisition
        composed = compose_stack(members, model)
        signal = _composed_signal(composed, members, model, has_angle_axis=True)
        reader = build_multiangle_reader(signal, composed)
        frames = [reader.read_frame((angle, 3, 3)) for angle in range(N_MEMBERS)]
        for left in range(N_MEMBERS):
            for right in range(left + 1, N_MEMBERS):
                assert not np.array_equal(frames[left], frames[right])


class TestDeclining:
    """A specific reader kind must decline rather than serve wrong pixels."""

    def test_a_node_with_no_recipe_is_declined(self, acquisition):
        members, model = acquisition
        composed = compose_sum(members, model)
        signal = hs.signals.Signal2D(
            np.zeros((1, 1, 1, 1), dtype=np.uint8)).as_lazy()
        signal.data = composed
        assert recipe_for(signal) is None
        assert build_multiangle_reader(signal, composed) is None

    def test_a_recipe_whose_frames_are_the_wrong_shape_is_declined(
            self, acquisition, caplog):
        members, model = acquisition
        composed = compose_sum(members, model)
        # A node claiming an uncropped detector: the recipe crops, so the two
        # disagree and the reader must refuse rather than serve a smaller frame.
        mismatched = da.zeros(composed.shape[:2] + DETECTOR, dtype=composed.dtype)
        signal = _composed_signal(mismatched, members, model,
                                  has_angle_axis=False, dtype=composed.dtype)
        assert build_multiangle_reader(signal, mismatched) is None

    def test_a_stack_whose_angle_axis_is_the_wrong_length_is_declined(
            self, acquisition):
        members, model = acquisition
        composed = compose_stack(members, model)
        truncated = composed[:-1]
        signal = _composed_signal(truncated, members, model, has_angle_axis=True)
        assert build_multiangle_reader(signal, truncated) is None

    def test_an_empty_member_list_is_declined(self, acquisition):
        members, model = acquisition
        composed = compose_sum(members, model)
        signal = hs.signals.Signal2D(
            np.zeros((1, 1, 1, 1), dtype=np.uint8)).as_lazy()
        signal.data = composed
        attach_recipe(signal, MultiAngleRecipe(
            members=(), model=model, has_angle_axis=False, dtype=None))
        assert build_multiangle_reader(signal, composed) is None


class TestMemorySafety:
    def test_reading_a_frame_never_computes_the_composition(self, acquisition):
        """The reader's whole reason to exist: one composed frame must not send
        the composed graph to dask (CLAUDE.md memory-safety rule)."""
        members, model = acquisition
        composed = compose_sum(members, model)
        signal = _composed_signal(composed, members, model,
                                  has_angle_axis=False, dtype=composed.dtype)
        reader = build_multiangle_reader(signal, composed)

        computed_shapes = []
        original = da.Array.compute

        def record(self, *args, **kwargs):
            computed_shapes.append(self.shape)
            return original(self, *args, **kwargs)

        da.Array.compute = record
        try:
            reader.read_frame((4, 5))
        finally:
            da.Array.compute = original
        assert composed.shape not in computed_shapes


class TestSubsetParity:
    """A node holding a SUBSET of the members — a per-shell sum.

    Its members sit at model indices 2 and 3, not 0 and 1, and their offsets
    must be looked up by the model index. Getting that wrong serves a real
    frame from the wrong scan position, which no shape or dtype check catches.
    """

    def _shell_node(self, members, model, member_indices):
        from spyde.multiangle.signals import build_summed_signal

        aligned = [aligned_member(member, model, index)
                   for index, member in enumerate(members)]
        return build_summed_signal(
            aligned, members, model, member_indices=member_indices)

    def test_a_trailing_shell_matches_the_dask_truth(self, acquisition):
        members, model = acquisition
        member_indices = model.shells[1]
        assert member_indices == (2, 3), "fixture no longer exercises a subset"
        signal = self._shell_node(members, model, member_indices)
        reader = build_multiangle_reader(signal, signal.data)
        assert reader is not None

        reference = signal.data.compute()
        for scan_row in range(reference.shape[0]):
            for scan_column in range(reference.shape[1]):
                assert np.array_equal(
                    reader.read_frame((scan_row, scan_column)),
                    reference[scan_row, scan_column]), (
                        f"shell frame wrong at ({scan_row}, {scan_column})")

    def test_a_leading_shell_matches_too(self, acquisition):
        members, model = acquisition
        signal = self._shell_node(members, model, model.shells[0])
        reader = build_multiangle_reader(signal, signal.data)
        reference = signal.data.compute()
        assert np.array_equal(reader.read_frame((2, 2)), reference[2, 2])

    def test_the_two_shells_are_not_the_same_frame(self, acquisition):
        """Guards the test above: if both shells returned the same pixels the
        parity checks would pass while reading the wrong members."""
        members, model = acquisition
        left = self._shell_node(members, model, model.shells[0])
        right = self._shell_node(members, model, model.shells[1])
        left_frame = build_multiangle_reader(left, left.data).read_frame((2, 2))
        right_frame = build_multiangle_reader(right, right.data).read_frame((2, 2))
        assert not np.array_equal(left_frame, right_frame)

    def test_the_recipe_records_model_indices_not_positions(self, acquisition):
        members, model = acquisition
        signal = self._shell_node(members, model, model.shells[1])
        assert recipe_for(signal).indices == (2, 3)


class TestReadAhead:
    """The read-ahead asks a reader where the next decode will happen; a
    composed node answers with the intersection of its members' chunks."""

    def test_the_span_holds_its_point_and_everything_in_it_is_resident(
            self, acquisition):
        from spyde.array_cache.block_cache import BlockCache

        members, model = acquisition
        composed = compose_sum(members, model)
        signal = _composed_signal(composed, members, model,
                                  has_angle_axis=False, dtype=composed.dtype)
        reader = build_multiangle_reader(signal, composed,
                                         block_cache=BlockCache(1 << 28))
        point = (5, 9)
        assert not reader.is_chunk_resident(point)
        reader.read_frame(point)
        span = reader.chunk_span(point)
        assert span is not None
        (row_start, row_stop), (column_start, column_stop) = span
        assert row_start <= point[0] < row_stop
        assert column_start <= point[1] < column_stop
        for row in range(row_start, row_stop):
            for column in range(column_start, column_stop):
                assert reader.is_chunk_resident((row, column)), (row, column)
        past = (row_stop, point[1])
        if past[0] < composed.shape[0]:
            assert not reader.is_chunk_resident(past), \
                "the position past the span should still need a decode"

    def test_the_stack_node_spans_one_angle(self, acquisition):
        members, model = acquisition
        stacked = compose_stack(members, model)
        signal = _composed_signal(stacked, members, model, has_angle_axis=True)
        reader = build_multiangle_reader(signal, stacked)
        span = reader.chunk_span((2, 5, 9))
        assert span is not None and span[0] == (2, 3)
