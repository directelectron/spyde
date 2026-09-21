"""
compose.py — the aligned members combined into ONE lazy array.

Two shapes come out of it:

* :func:`stack_aligned` — ``(N, y, x, ky, kx)``, every member kept separate on a
  common grid. This is the 5-D node: the angle axis survives, so a user can
  scrub through the individual angles.
* :func:`sum_aligned` — ``(y, x, ky, kx)``, the stack summed over the angle
  axis. This is what the analysis wants, and the node a loaded acquisition
  displays first.

**Where the alignment happens.** For a binary member it happens at LOAD time —
:mod:`spyde.multiangle.load` chooses which frame is read at which output
position, so members arrive already on a common grid and these functions only
add them. That is the preferred path: it moves nothing, and every member is born
on the same chunk grid so there is no rechunk anywhere in the graph.

:func:`compose_stack` and :func:`compose_sum` are the fallback for members that
could NOT be loaded that way — a chunked, compressed store has no memmap to
index, so it is loaded whole and cropped afterwards. Pair those with
:func:`member_nav_chunks`, which keeps the crops landing on one grid even though
the cropping happens after the read. Either way nothing is ever rechunked, which
is the constraint Live-Display §1 imposes: a rechunk moves bytes between tasks at
compute time, on data that is tens of gigabytes in the field.

Both paths crop the scan AND the detector to the region every member covers, so
each surviving element of the sum has all N members behind it. Keeping the full
extent and padding instead would leave a border where fewer members contribute,
and a step in contributor count is indistinguishable from real structure to
anything reading the edge.
"""
from __future__ import annotations

import logging

import numpy as np

#: Summing N members of an integer type overflows that type, so the sum is
#: accumulated in a wider one. Chosen over float32 because these are photon or
#: electron counts: an integer accumulator is exact, whereas float32 stops being
#: exact above 2**24 and a long shell sum can reach that.
_INTEGER_ACCUMULATORS = (np.uint16, np.uint32, np.uint64)
_SIGNED_ACCUMULATORS = (np.int16, np.int32, np.int64)


def sum_dtype(source_dtype, n_members: int):
    """A dtype that cannot overflow when *n_members* frames are summed.

    A float source keeps its own type (float16 is widened to float32, which has
    range to spare and is what every downstream consumer expects anyway). An
    integer source moves to the narrowest integer that still holds
    ``n_members * max``, because the sum of counts is itself a count and
    rounding it into a float would throw away exactness for nothing.
    """
    dtype = np.dtype(source_dtype)
    if dtype.kind == "f":
        return np.dtype(np.float32) if dtype.itemsize < 4 else dtype
    if dtype.kind not in ("u", "i"):
        raise TypeError(
            f"cannot sum members of dtype {dtype}; expected an integer or "
            "floating point type"
        )
    if n_members < 1:
        raise ValueError(f"n_members must be at least 1; got {n_members}")
    if n_members == 1:
        return dtype

    largest = int(np.iinfo(dtype).max) * int(n_members)
    smallest = int(np.iinfo(dtype).min) * int(n_members)
    candidates = _INTEGER_ACCUMULATORS if dtype.kind == "u" else _SIGNED_ACCUMULATORS
    for candidate in candidates:
        info = np.iinfo(candidate)
        if np.dtype(candidate).itemsize < dtype.itemsize:
            continue
        if info.min <= smallest and largest <= info.max:
            return np.dtype(candidate)
    # Nothing is wider than 64 bits. A 64-bit source is already the widest
    # accumulator there is, and refusing it cost a whole acquisition: a
    # rebinned stack came back as uint64, this raised, and the file reopened
    # as a bare 5-D array. Real detector counts sit nowhere near 2**63, so the
    # sum stays exact in practice, and a stack the loader can show beats one
    # it cannot.
    logging.getLogger(__name__).warning(
        "summing %d members of %s in %s: a value above %.3g would wrap",
        n_members, dtype, dtype, np.iinfo(dtype).max / n_members)
    return dtype


def _member_array(member):
    """The backing array of a member, whether it arrived as a signal or an array.

    Duck-typed on ``data`` so this module never imports hyperspy — the same
    reason :mod:`spyde.drift.frames` avoids it, and what lets a test hand over a
    plain dask array.
    """
    # A HyperSpy signal keeps its array in `.data`; an array IS the array. Test
    # for the signal, because numpy's own `.data` is a memoryview — which has an
    # `ndim` of its own and so passes the check below before failing obscurely.
    data = member.data if hasattr(member, "axes_manager") else member
    if getattr(data, "ndim", None) != 4:
        raise ValueError(
            "each member must be 4-D (scan_y, scan_x, detector_y, detector_x); "
            f"got ndim={getattr(data, 'ndim', None)}"
        )
    return data


def aligned_member(member, model, member_index: int):
    """One member, sliced onto the common scan grid and the common detector.

    The returned array is a VIEW expressed as a slice of the member's own lazy
    array; reading it reads the member. This single function is the alignment —
    everything else in this module stacks or sums what it returns.
    """
    data = _member_array(member)
    scan_rows, scan_columns = model.nav_slices(member_index, data.shape[:2])
    detector_rows, detector_columns = model.detector_slices(
        member_index, data.shape[2:])
    return data[scan_rows, scan_columns, detector_rows, detector_columns]


def _aligned_members(members, model):
    if len(members) != model.n_members:
        raise ValueError(
            f"the model describes {model.n_members} members but {len(members)} "
            "were given"
        )
    # Members of differing raw shape slice to differing aligned shapes, which
    # _check_aligned catches for both entry points.
    return [aligned_member(member, model, index)
            for index, member in enumerate(members)]


def member_nav_chunks(model, member_index: int, member_scan_shape,
                      nav_chunk: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Navigation chunking to LOAD member *member_index* with, so that its
    aligned crop lands on the grid every other member's does.

    This is the whole answer to chunk alignment, and it costs nothing.

    Each member is cropped from a different start, so members loaded on a
    uniform grid have their boundaries knocked out of step by exactly their
    offset, and summing them forces dask to unify to the UNION of all of them.
    Measured on a 256x256 scan with 10 members and 32-position chunks: **1849
    chunks where one member has 64**, 70 of them slivers one or two scan
    positions wide.

    Fixing that afterwards with ``rechunk`` is the wrong tool — it moves bytes
    between tasks at compute time, on data that is tens of gigabytes in the
    field, and Live-Display §1 rules it out. Fixing it at LOAD time does not
    move anything: make the member's FIRST chunk exactly as long as the part the
    crop discards, and every chunk after it starts where the common grid starts.
    A lazy reload with these chunks only rebuilds the dask graph.

    So for a member whose crop starts at ``S`` and runs for ``L``, out of an
    extent ``E``, the chunks are ``(S, C, C, …, L % C, E - S - L)`` — the head
    and tail are the discarded margins and the body is the common grid.
    """
    scan_rows, scan_columns = model.nav_slices(member_index, member_scan_shape)
    extent = tuple(int(size) for size in member_scan_shape)
    return tuple(
        _axis_chunks(int(axis_slice.start),
                     int(axis_slice.stop - axis_slice.start),
                     axis_extent, int(nav_chunk))
        for axis_slice, axis_extent in zip((scan_rows, scan_columns), extent)
    )


def _axis_chunks(start: int, length: int, extent: int,
                 nav_chunk: int) -> tuple[int, ...]:
    if nav_chunk < 1:
        raise ValueError(f"nav_chunk must be at least 1; got {nav_chunk}")
    head = (start,) if start > 0 else ()
    body = (nav_chunk,) * (length // nav_chunk)
    remainder = length % nav_chunk
    if remainder:
        body += (remainder,)
    tail_length = extent - start - length
    if tail_length < 0:
        raise ValueError(
            f"the crop [{start}, {start + length}) does not fit in an extent "
            f"of {extent}")
    tail = (tail_length,) if tail_length > 0 else ()
    return head + body + tail


def aligned_load_chunks(model, member_scan_shape, nav_chunk: int,
                        detector_ndim: int = 2):
    """One ``chunks=`` argument per member, ready to pass to a lazy load.

    Signal axes are ``-1`` — a chunk must span the full detector (Live-Display
    §1), and cropping the detector by a constant cannot break that.
    """
    return [
        member_nav_chunks(model, member_index, member_scan_shape, nav_chunk)
        + (-1,) * detector_ndim
        for member_index in range(model.n_members)
    ]


def composed_nav_chunks(aligned) -> tuple[tuple[int, ...], ...] | None:
    """The navigation chunking the aligned members agree on, or None.

    Returned so a caller can assert the load-time alignment actually worked —
    a member loaded on the wrong grid is not an error, just a slow graph, and
    that is precisely the kind of regression that goes unnoticed.
    """
    grids = {getattr(array, "chunks", (None,))[:2] for array in aligned}
    return grids.pop() if len(grids) == 1 else None


def _check_aligned(aligned):
    """Every aligned member must be the same shape, and non-empty."""
    shapes = {tuple(int(size) for size in array.shape) for array in aligned}
    if not shapes:
        raise ValueError("no members to compose")
    if len(shapes) != 1:
        raise ValueError(
            f"members do not align to a common shape; got {sorted(shapes)}")
    if 0 in shapes.pop():
        raise ValueError(
            "the members have no common region — the offsets exceed the "
            "extent on at least one axis, so nothing overlaps"
        )
    return aligned


def stack_aligned(aligned):
    """``(N, y, x, ky, kx)`` from members that are ALREADY on the common grid.

    The entry point for members loaded through :mod:`spyde.multiangle.load`,
    which did the alignment while reading. The angle axis gets one chunk per
    member, so scrubbing to angle *a* touches member *a* and nothing else.
    """
    import dask.array as da

    return da.stack(_check_aligned(list(aligned)), axis=0)


def sum_aligned(aligned, *, dtype=None):
    """``(y, x, ky, kx)`` from members that are ALREADY on the common grid.

    Summed as a tree over the members rather than by reducing
    :func:`stack_aligned`, so no N-member axis is ever materialised — reading
    one output frame reads one frame from each member and adds them.
    """
    aligned = _check_aligned(list(aligned))
    if dtype is None:
        dtype = sum_dtype(aligned[0].dtype, len(aligned))
    dtype = np.dtype(dtype)

    total = aligned[0].astype(dtype)
    for array in aligned[1:]:
        total = total + array
    return total


def compose_stack(members, model):
    """``(N, y, x, ky, kx)`` from RAW members, sliced onto the common grid here.

    For members that were not loaded already aligned — a chunked, compressed
    store has no memmap to index, so it is cropped after loading instead. Pair
    it with :func:`member_nav_chunks` so the crops still land on one grid;
    members loaded through :mod:`spyde.multiangle.load` are already aligned and
    go to :func:`stack_aligned` instead.
    """
    return stack_aligned(_aligned_members(members, model))


def compose_sum(members, model, *, dtype=None):
    """``(y, x, ky, kx)`` from RAW members, sliced onto the common grid here.

    The counterpart of :func:`compose_stack`; see it for when to use this
    rather than :func:`sum_aligned`. ``dtype`` defaults to :func:`sum_dtype` of
    the members' own type, which cannot overflow.
    """
    return sum_aligned(_aligned_members(members, model), dtype=dtype)


def composed_axis_offsets(model, scan_offsets, scan_scales,
                          detector_offsets, detector_scales):
    """Where the composed array's axes start, in the reference member's units.

    Cropping to the common region moves every origin: the composed scan begins
    at :attr:`~spyde.multiangle.model.MultiAngleModel.overlap_origin` scan
    positions into the reference member, and the composed detector at
    ``detector_origin`` pixels into it. Calibration that ignores this is wrong
    by exactly the crop, which puts the reciprocal-space origin off the direct
    beam — subtly enough to survive a glance and corrupt every g-vector.

    Returns ``(scan_origin, detector_origin)``, each a ``(y, x)`` pair in the
    axes' own units.
    """
    scan_rows, scan_columns = model.overlap_origin
    detector_rows, detector_columns = model.detector_origin
    return (
        (scan_offsets[0] + scan_rows * scan_scales[0],
         scan_offsets[1] + scan_columns * scan_scales[1]),
        (detector_offsets[0] + detector_rows * detector_scales[0],
         detector_offsets[1] + detector_columns * detector_scales[1]),
    )
