"""
load.py — load a member ALREADY ALIGNED, by choosing which frame goes where.

This is the alignment, and it happens at load time rather than after it.

A binary 4-D dataset on disk is a flat run of frames; which scan position a
frame belongs to is a convention, not a property of the bytes. So aligning a
member to a common grid is not a shift of anything — it is a different answer to
"which frame goes at output position (y, x)". :func:`frame_index_map` computes
that answer, and it is one line of arithmetic:

    frame = (output_row + start_row) * member_width + output_column + start_column

RosettaSciIO's distributed memmap utility already reads a file that way:
:func:`rsciio.utils._distributed.slice_memmap` takes a block-shaped array of
frame indices and fancy-indexes the memmap with it, so one dask block reads
exactly its own frames (``-1`` yields zeros for a position no frame covers).
We hand it our map instead of the identity one a normal load would use.

Why this rather than slicing after loading. Every member is cropped from a
different start, so members loaded on one uniform grid have their chunk
boundaries knocked out of step, and summing them makes dask unify to the UNION
of all of them — measured at 1849 chunks where a member has 64, with 70 slivers
one or two scan positions wide. Repairing that with ``rechunk`` moves bytes
between tasks at compute time on data that is tens of gigabytes in the field,
which Live-Display §1 rules out. Choosing the output grid here means every
member is born on the SAME grid and there is nothing to repair: no rechunk
layer, no shuffle, no copy.

The detector crop is applied as an ordinary slice afterwards. Signal axes are a
single chunk by construction, so slicing them cannot fragment anything, and a
raw file cannot serve a sub-rectangle of a frame more cheaply than the whole
frame anyway.

Two kinds of member can be aligned this way, and the test is the same one in
both cases: is an ARBITRARY frame cheap to read?

* A flat binary file (``.mrc``, ``.de5``, raw) — :func:`load_aligned_member`.
  Any frame is a seek, so the frame-index map above does it.
* A store compressed FRAME BY FRAME (a ``.zspy``/``.hspy`` chunked
  ``(1, 1, ky, kx)``) — :func:`load_aligned_store_member`. There is no flat run
  of frames to index, but there does not need to be: with a chunk per frame
  every dask grid is equally cheap to read, so the member is rebuilt on a grid
  whose blocks line up with the others' after cropping. Note the dask grid is
  NOT the storage grid — one dask chunk per frame would put a task in the graph
  for every scan position.

A store with BIG navigation chunks is the one case neither covers, and
:func:`frame_chunked_source` declines it deliberately: a block that straddles
storage chunks decodes all of them to keep part of each, which is the ~100x
mistake CLAUDE.md records for compressed data. Those members are loaded whole
and cropped afterwards, with
:func:`spyde.multiangle.compose.member_nav_chunks` keeping the crops on one grid.
"""
from __future__ import annotations

import numpy as np


def frame_index_map(model, member_index: int, member_scan_shape) -> np.ndarray:
    """``(height, width)`` of FILE frame indices for one member's aligned view.

    Entry ``(row, column)`` is the index, in the member's own flat run of
    frames, of the frame belonging at that position of the COMMON grid. This is
    the entire alignment: nothing is shifted, a different frame is simply read.

    Frames outside the common region get no entry, because the map only covers
    the common region — so the margin this member does not share with the others
    is never read at all, rather than read and discarded.
    """
    scan_rows, scan_columns = model.nav_slices(member_index, member_scan_shape)
    member_width = int(member_scan_shape[1])
    rows = np.arange(scan_rows.start, scan_rows.stop, dtype=np.int64)
    columns = np.arange(scan_columns.start, scan_columns.stop, dtype=np.int64)
    return rows[:, None] * member_width + columns[None, :]


def aligned_member_chunks(model, member_scan_shape, detector_shape,
                          nav_chunk: int, dtype):
    """The output chunk grid, which is the SAME for every member.

    Chosen once here rather than inherited from each member's own layout — that
    is what makes the composition need no rechunk. Signal axes are one chunk
    (Live-Display §1).
    """
    import dask.array as da

    height, width = model.overlap_shape(member_scan_shape)
    return da.core.normalize_chunks(
        (nav_chunk, nav_chunk, -1, -1),
        shape=(height, width) + tuple(int(size) for size in detector_shape),
        dtype=np.dtype(dtype),
    )


def load_aligned_member(path, model, member_index: int, *, dtype,
                        member_scan_shape, detector_shape, nav_chunk: int = 32,
                        offset: int = 0, order: str = "C", key=None):
    """One member of a multi-angle acquisition, read from *path* already aligned.

    Returns a lazy array of the member's frames on the COMMON scan grid, with
    the common detector crop applied. Nothing is read here — this builds the
    graph, and each block reads only its own frames when it is computed.

    Parameters
    ----------
    path
        The member's binary file.
    model, member_index
        Which member this is, and the alignment it belongs to.
    dtype
        The file's frame dtype, as :class:`numpy.memmap` would take it. A
        structured dtype needs *key* naming the field holding the image.
    member_scan_shape, detector_shape
        The member's OWN scan and detector shape, before alignment.
    nav_chunk
        Scan positions per chunk on each navigation axis.
    offset, order, key
        Passed through to the memmap, for a file with a header, non-C ordering
        or a structured dtype.
    """
    import dask.array as da
    from rsciio.utils._distributed import slice_memmap

    dtype = np.dtype(dtype)
    array_dtype = dtype[key].base if dtype.names is not None else dtype.base

    index_map = frame_index_map(model, member_index, member_scan_shape)
    chunks = aligned_member_chunks(
        model, member_scan_shape, detector_shape, nav_chunk, array_dtype)
    # The trailing (1, 1) is what slice_memmap expands into the detector axes:
    # one block of the index map becomes one block of frames.
    chunked_index_map = da.from_array(
        index_map[..., None, None],
        chunks=(chunks[0], chunks[1], 1, 1),
    )

    n_frames = int(np.prod([int(size) for size in member_scan_shape]))
    frames = da.map_blocks(
        slice_memmap,
        chunked_index_map,
        file=str(path),
        dtype=array_dtype,
        shape=(n_frames,) + tuple(int(size) for size in detector_shape),
        order=order,
        mode="r",
        dtypes=dtype,
        offset=offset,
        chunks=chunks,
        drop_axis=None,
        positions=True,
        key=key,
    )

    detector_rows, detector_columns = model.detector_slices(
        member_index, detector_shape)
    return frames[:, :, detector_rows, detector_columns]


def load_aligned_members(paths, model, *, dtype, member_scan_shape,
                         detector_shape, nav_chunk: int = 32, offset: int = 0,
                         order: str = "C", key=None):
    """Every member of *model*, each already on the common grid.

    The members come back sharing one chunk grid, so
    :func:`spyde.multiangle.compose.compose_sum` and
    :func:`~spyde.multiangle.compose.compose_stack` combine them with no
    rechunk layer at all.
    """
    if len(paths) != model.n_members:
        raise ValueError(
            f"the model describes {model.n_members} members but {len(paths)} "
            "paths were given")
    return [
        load_aligned_member(
            path, model, member_index, dtype=dtype,
            member_scan_shape=member_scan_shape, detector_shape=detector_shape,
            nav_chunk=nav_chunk, offset=offset, order=order, key=key)
        for member_index, path in enumerate(paths)
    ]


def frame_chunked_source(member):
    """The live store behind *member* if it is compressed FRAME BY FRAME, else None.

    "Frame by frame" means the store's own chunks are one navigation position
    each, spanning the whole detector — ``(1, 1, ky, kx)``. That is the property
    that matters, because it decides whether an arbitrary frame is cheap: with a
    chunk per frame, reading any set of frames costs exactly those frames, so the
    dask grid can be chosen freely. With big navigation chunks it cannot — a
    block that straddles storage chunks decodes all of them to keep part of each,
    which is the ~100x mistake CLAUDE.md records for compressed data.

    Reuses :func:`~spyde.array_cache.readers.source_array.find_source_array`,
    which already knows how to recover the store from a lazy signal's graph and
    which already declines a derived view whose graph still carries the source's
    layer — a gate this needs for exactly the same reason.
    """
    from spyde.array_cache.readers.source_array import find_source_array

    source = find_source_array(getattr(member, "data", member))
    if source is None:
        return None
    chunks = getattr(source, "chunks", None)
    if chunks is None:                      # contiguous store: no chunk concept
        return None
    shape = tuple(int(size) for size in getattr(source, "shape", ()))
    if len(shape) != 4:
        return None
    if tuple(int(size) for size in chunks[:2]) != (1, 1):
        return None
    if tuple(int(size) for size in chunks[2:]) != shape[2:]:
        return None
    return source


def load_aligned_store_member(member, model, member_index: int, *,
                              nav_chunk: int = 32):
    """One member of a frame-chunked store, aligned — or None if not applicable.

    The store has no flat run of frames to index the way a memmap does, but it
    does not need one: with a chunk per frame, EVERY dask grid is equally cheap
    to read, so the member is simply rebuilt on a grid whose blocks line up with
    the others' after cropping. ``da.from_array`` on an existing store is a graph
    operation — it reads nothing and moves nothing, the same way a lazy reload
    with different chunks does.

    The grid comes from :func:`~spyde.multiangle.compose.member_nav_chunks`: a
    leading chunk exactly as long as the margin this member's crop discards, so
    every boundary after it sits on the common grid. Dask blocks stay a sensible
    size — the point is NOT to make every frame its own dask chunk, which would
    give the graph one task per scan position.
    """
    import dask.array as da

    from spyde.multiangle.compose import aligned_member, member_nav_chunks

    source = frame_chunked_source(member)
    if source is None:
        return None

    scan_shape = tuple(int(size) for size in source.shape[:2])
    chunk_specification = member_nav_chunks(
        model, member_index, scan_shape, nav_chunk) + (-1, -1)
    rebuilt = da.from_array(source, chunks=chunk_specification)
    return aligned_member(rebuilt, model, member_index)


def aligned_from_signal(member, model, member_index: int, *,
                        nav_chunk: int = 32):
    """Re-read an already-opened MEMMAP-BACKED member, aligned — or None.

    The loader opens each member through HyperSpy to learn its shape and
    calibration, which leaves it holding a lazy array rather than the file
    layout :func:`load_aligned_member` needs. That layout has not been lost: a
    member read through RosettaSciIO's distributed memmap carries its own
    ``slice_memmap`` arguments in its dask graph, and
    :func:`~spyde.array_cache.readers.binary.find_memmap_source` reads them back
    out. So the member can be re-expressed as an aligned read of the same file
    without re-opening or re-reading anything.

    Declines — returning None for the caller to fall back on cropping — when the
    member is not memmap-backed, or when its layout is one this cannot flatten
    to a run of frames. ``positions`` mode indexes a FLAT frame axis, which is
    the same bytes as ``(scan_y, scan_x, ...)`` only in C order.
    """
    from spyde.array_cache.readers.binary import find_memmap_source

    data = member.data if hasattr(member, "axes_manager") else member
    arguments = find_memmap_source(data)
    if arguments is None:
        return None
    if str(arguments.get("order", "C")).upper() != "C":
        return None

    dtype = np.dtype(arguments["dtypes"])
    key = arguments.get("key")
    shape = tuple(int(size) for size in arguments["shape"])
    if key is not None:
        # Structured dtype: `shape` is the scan only, the field supplies the frame.
        scan_shape = shape[:2]
        detector_shape = tuple(int(size) for size in dtype[key].shape)
    else:
        if len(shape) != 4:
            return None
        scan_shape, detector_shape = shape[:2], shape[2:]
    if len(scan_shape) != 2 or len(detector_shape) != 2:
        return None
    if tuple(int(size) for size in data.shape[:2]) != scan_shape:
        return None

    return load_aligned_member(
        arguments["file"], model, member_index, dtype=dtype,
        member_scan_shape=scan_shape, detector_shape=detector_shape,
        nav_chunk=nav_chunk, offset=int(arguments.get("offset", 0) or 0),
        order="C", key=key)
