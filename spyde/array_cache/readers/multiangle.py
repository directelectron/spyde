"""Reader kind 6: a composed multi-angle node, read through its MEMBERS.

A multi-angle acquisition is N separate 4-D files aligned by an integer index
remap (:mod:`spyde.multiangle`). The composed node the user displays is either
their sum or the stack that keeps the angle axis, and in both cases one frame of
it is a function of one frame from each member.

Why not just ask dask. The composed array is a real lazy array and
``composed[y, x].compute()`` answers correctly — by pulling a navigation CHUNK
out of every one of the N members to keep one frame from each. That is the same
trap the rebin and `map` paths hit (``readers/per_frame.py``, ``readers/
recipe.py`` measure 2403 ms and 2196 ms for the equivalent question), except
multiplied by N files. Going through the members' own readers instead means each
member decodes through ITS block cache, so a dwell inside a chunk costs a numpy
slice per member and a chunk crossing costs one decode per member — the same
cost a single dataset pays, N times, and no worse.

The member readers share the owning plot's :class:`BlockCache`, so N members
compete for ONE byte budget rather than each keeping a private one. That is
deliberate: the budget exists to hold the few storage chunks a read spans, and a
multi-angle read spans N of them at once.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class MultiAngleReader:
    """Frames of a composed multi-angle node, read one member at a time.

    Resolved once per (plot, node); everything member-specific — the readers,
    the per-member scan origin, the per-member detector crop — is computed in
    ``__init__`` so ``read_frame`` is a loop of slices and one add.
    """

    def __init__(self, signal, data, recipe, block_cache=None) -> None:
        from spyde.array_cache.resolve import resolve_reader

        self.signal = signal
        self.data = data
        self._recipe = recipe
        self._has_angle_axis = bool(recipe.has_angle_axis)
        self._nav_ndim = 3 if self._has_angle_axis else 2

        model = recipe.model
        self._n_members = len(recipe.members)
        self._member_readers = []
        self._scan_origins = []
        self._detector_crops = []
        # A node rebuilt from a saved stack reads its members as PLANES of
        # that stack, through the stack's one store reader; a node composed
        # from files reads each member through its own. The plane of a member
        # is its index in the model, because the stack was written in model
        # order — and that stays true for a per-shell subset.
        self._stack_reader = None
        self._planes = tuple(recipe.indices)
        stack = getattr(recipe, "stack", None)
        if stack is not None:
            self._stack_reader = resolve_reader(
                stack, getattr(stack, "data", stack), block_cache)
        # Offsets are looked up by the member's index IN THE MODEL, which is
        # not its position here when the node holds a subset (a per-shell sum).
        for model_index, member in zip(recipe.indices, recipe.members):
            member_data = getattr(member, "data", member)
            scan_shape = tuple(int(size) for size in member_data.shape[:2])
            detector_shape = tuple(int(size) for size in member_data.shape[2:])
            scan_rows, scan_columns = model.nav_slices(model_index, scan_shape)
            self._scan_origins.append((scan_rows.start, scan_columns.start))
            self._detector_crops.append(
                model.detector_slices(model_index, detector_shape))
            self._member_readers.append(
                None if self._stack_reader is not None
                else resolve_reader(member, member_data, block_cache))

        first = getattr(recipe.members[0], "data", recipe.members[0])
        self._frame_shape = model.detector_shape(
            tuple(int(size) for size in first.shape[2:]))
        self._dtype = np.dtype(
            recipe.dtype if recipe.dtype is not None else data.dtype)

    @property
    def frame_bytes(self) -> int:
        return int(np.prod(self._frame_shape)) * self._dtype.itemsize

    def _member_frame(self, member_index: int, scan_row: int,
                      scan_column: int) -> np.ndarray:
        """One member's contribution at a composed scan position.

        The composed grid is the members' common region, so a composed position
        sits at a DIFFERENT index in each member — that offset is the alignment,
        and applying it here is what "remap rather than move" means at read time.
        """
        frame = self._member_reader(member_index).read_frame(
            self._member_point(member_index, scan_row, scan_column))
        detector_rows, detector_columns = self._detector_crops[member_index]
        return frame[detector_rows, detector_columns]

    def _member_reader(self, member_index: int):
        if self._stack_reader is not None:
            return self._stack_reader
        return self._member_readers[member_index]

    def _member_point(self, member_index: int, scan_row: int,
                      scan_column: int) -> tuple:
        """Where a composed position lives in this member's own reader."""
        origin_row, origin_column = self._scan_origins[member_index]
        point = (origin_row + scan_row, origin_column + scan_column)
        if self._stack_reader is not None:
            return (self._planes[member_index],) + point
        return point

    def _members_at(self, point) -> tuple:
        """(members involved, scan row, scan column) for a node position."""
        if self._has_angle_axis:
            angle, scan_row, scan_column = point
            if not 0 <= angle < self._n_members:
                raise IndexError(
                    f"angle {angle} outside 0..{self._n_members - 1}")
            return (angle,), scan_row, scan_column
        scan_row, scan_column = point
        return tuple(range(self._n_members)), scan_row, scan_column

    def read_frame(self, indices) -> np.ndarray:
        point = tuple(int(value) for value in indices[:self._nav_ndim])
        members, scan_row, scan_column = self._members_at(point)
        if self._has_angle_axis:
            return self._member_frame(members[0], scan_row, scan_column)
        # Accumulate in the composed dtype rather than the members' own: N
        # members of an integer type overflow it, which numpy would wrap
        # silently (spyde.multiangle.compose.sum_dtype picks the width).
        total = self._member_frame(members[0], scan_row, scan_column).astype(
            self._dtype, copy=True)
        for member_index in members[1:]:
            total += self._member_frame(member_index, scan_row, scan_column)
        return total

    def chunk_span(self, point):
        """``(start, stop)`` per navigation axis of the stretch of positions
        that costs no decode from ``point``, or None when unknown.

        The read-ahead aims at the first position past it. A composed
        position sits in a different chunk of each member, so the stretch is
        the INTERSECTION of the members' chunks, translated back into the
        composed grid: the nearest boundary in either direction is where the
        next decode happens, whichever member it belongs to.
        """
        point = tuple(int(value) for value in point[:self._nav_ndim])
        members, scan_row, scan_column = self._members_at(point)
        rows, columns = (0, None), (0, None)
        for member_index in members:
            span = getattr(self._member_reader(member_index), "chunk_span", None)
            span = span(self._member_point(member_index, scan_row,
                                           scan_column)) if span else None
            if span is None:
                return None
            origin_row, origin_column = self._scan_origins[member_index]
            (row_start, row_stop), (column_start, column_stop) = span[-2:]
            rows = (max(rows[0], row_start - origin_row),
                    row_stop - origin_row if rows[1] is None
                    else min(rows[1], row_stop - origin_row))
            columns = (max(columns[0], column_start - origin_column),
                       column_stop - origin_column if columns[1] is None
                       else min(columns[1], column_stop - origin_column))
        if self._has_angle_axis:
            return ((point[0], point[0] + 1), rows, columns)
        return (rows, columns)

    def is_chunk_resident(self, indices) -> bool:
        """Would ``read_frame`` be a numpy slice — every member's chunk
        already decoded? Side-effect-free."""
        try:
            point = tuple(int(value) for value in indices[:self._nav_ndim])
            members, scan_row, scan_column = self._members_at(point)
            for member_index in members:
                probe = getattr(self._member_reader(member_index),
                                "is_chunk_resident", None)
                if probe is None or not probe(self._member_point(
                        member_index, scan_row, scan_column)):
                    return False
            return True
        except Exception:
            return False


def build_multiangle_reader(signal, data, block_cache=None):
    """A :class:`MultiAngleReader` for *signal*, or None if it is not composed.

    Declines rather than guesses, like every other specific reader kind: a node
    with no recipe, or whose recipe does not reproduce the node's own shape, is
    left to the universal path. A slow frame is recoverable; a wrong one, drawn
    from the wrong member, is not.
    """
    from spyde.multiangle.recipe import recipe_for

    recipe = recipe_for(signal)
    if recipe is None or not recipe.members:
        return None
    try:
        reader = MultiAngleReader(signal, data, recipe, block_cache=block_cache)
    except Exception:
        log.warning("a multi-angle recipe would not build a reader; falling "
                    "back to the universal reader for this node.", exc_info=True)
        return None

    expected_nav = tuple(int(size) for size in data.shape[:reader._nav_ndim])
    expected_frame = tuple(int(size) for size in data.shape[reader._nav_ndim:])
    if tuple(reader._frame_shape) != expected_frame:
        log.warning(
            "a multi-angle recipe yields %s frames but the node's own shape "
            "says %s; declining rather than serving the wrong pixels.",
            tuple(reader._frame_shape), expected_frame)
        return None
    if recipe.has_angle_axis and expected_nav[0] != recipe.n_members:
        log.warning(
            "a multi-angle recipe has %d members but the node's angle axis is "
            "%d long; declining.", recipe.n_members, expected_nav[0])
        return None
    return reader
