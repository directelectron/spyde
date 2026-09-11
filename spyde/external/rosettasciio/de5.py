"""
Direct Electron ``.de5`` files: rsciio's EMD reader cannot parse their axes,
reads the whole datacube into RAM even when asked for a lazy load, and hands
the result back transposed with no navigation axes.

WHAT
----
A ``.de5`` is an EMD (HDF5) file, read by :mod:`rsciio.emd`'s Berkeley-variant
reader. That reader differs from what the camera writes in three ways, and this
module wraps three of its methods to absorb them:

1. Each calibration axis (``dim1`` … ``dim4``) is a column vector of shape
   ``(N, 1)`` rather than the ``(N,)`` vector the EMD spec shows.
   ``EMD_NCEM._parse_axis`` does ``float(axis_data[0])`` on it, which on a
   ``(1,)`` row is a ``TypeError`` under NumPy ≥ 1.25 ("only 0-dimensional
   arrays can be converted to Python scalars"), so every load died in
   ``hs.load``. The wrapper flattens the axis first.

2. ``EMD_NCEM._read_dataset`` does ``dataset[:]`` — it reads the ENTIRE HDF5
   array into memory — and ``lazy=True`` only wraps that in-RAM array in dask
   afterwards. An acquisition that was stopped early has its dataset declared
   at the planned scan size, so opening it tries to allocate the whole planned
   scan (hundreds of GB) and dies with "Unable to allocate". Measured on a
   302 MB file: a "lazy" load grew the process by 308 MB. The wrapper, on a
   lazy read, hands dask the ``h5py.Dataset`` itself (as hyperspy's own
   ``.hspy`` reader does), in blocks of whole frames sized to about 64 MB —
   the storage-aligned shape CLAUDE.md Live-Display §1 requires, and NOT the
   camera's one-frame-per-chunk grid, which drowns dask in tasks (see
   :func:`lazy_dataset_chunks` for the measurements). This applies to every
   EMD file the Berkeley reader opens lazily, since lazy should mean lazy for
   all of them.

3. The datacube is stored in C order, scan axes first: ``(R_y, R_x, Q_y, Q_x)``
   with one whole frame per HDF5 chunk, and ``dim1`` … ``dim4`` name the array
   axes in that same order (the py4DSTEM convention). The reader transposes
   every non-prismatic dataset (it expects Fortran-ordered EMD), yielding a
   ``(Q_x, Q_y, R_x, R_y)`` array, and marks every axis as a signal axis. SpyDE
   would then show a 12×12 "pattern" over a 1024×1024 "scan". After
   ``EMD_NCEM.read_file`` runs, the wrapper transposes the data back to storage
   order and flags all but the last two axes as navigation. This applies only
   when the file carries a ``DirectElectronInfo`` group.

WHY
---
All three fixes belong upstream, but a user with a ``.de5`` cannot wait for a
rosettasciio release. The wrappers are the smallest changes that make the
stock reader accept the files the camera actually writes.

WHEN TO REMOVE
--------------
When the minimum ``rosettasciio`` in ``pyproject.toml`` flattens the axis in
``_parse_axis``, wraps the live dataset for a lazy read, and recognises Direct
Electron files as C-ordered datacubes. ``apply()`` probes the installed
``_parse_axis`` with a ``(3, 1)`` array first, so once that half is fixed
upstream it becomes a logged no-op rather than a double patch.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_applied = False
_PATCH_SENTINEL = "_spyde_de5_patch"
_DIRECT_ELECTRON_GROUP = "DirectElectronInfo"


def _upstream_parses_column_axis(reader_class) -> bool:
    """True if the installed ``_parse_axis`` already accepts a ``(N, 1)`` axis."""
    try:
        offset, scale = reader_class._parse_axis(np.arange(3.0).reshape(3, 1))
    except Exception:
        return False
    return offset == 0.0 and scale == 1.0


def is_direct_electron_file(file) -> bool:
    """True if any top-level group of the open HDF5 file holds Direct Electron's
    camera-info group."""
    try:
        return any(
            hasattr(group, "keys") and _DIRECT_ELECTRON_GROUP in group
            for group in file.values()
        )
    except Exception:
        return False


CHUNK_TARGET_BYTES = 64 << 20
NAV_BLOCK_MAX = 32


def _largest_power_of_two_at_most(value: int) -> int:
    return 1 << max(0, int(value).bit_length() - 1)


def lazy_dataset_chunks(dataset) -> tuple:
    """The dask chunk grid for a lazily wrapped HDF5 dataset.

    Whole frames, in blocks of scan positions sized to about 64 MB. The camera
    stores one frame per HDF5 chunk, and taking that grid as the dask grid was
    measured on a 1 GB, 128×64 scan of 256² frames: 24,577 tasks, a navigator
    fill of one compute per 128 KB chunk (10 ms × 8192 ≈ 80 s) and 12.5 s for
    the sum as a single compute. The same dataset in 8×8 blocks: 129 tasks and
    1.2 s, the speed of a plain numpy loop over h5py. A block is a single
    hyperslab read for HDF5 whatever its own chunk grid, and a per-frame read
    never goes through dask at all (see :mod:`spyde.array_cache.readers.
    source_array`), so the dask grid only has to suit whole-scan computes.

    The block edge is a power of two, capped at 32 positions per axis, and a
    multiple of the file's own chunk edge where that is larger than one, so a
    dask chunk never straddles an HDF5 chunk. The frame axes are always whole:
    splitting them (dask's ``"auto"`` cubes) makes one frame cost several
    partial reads."""
    shape = tuple(int(n) for n in dataset.shape)
    if len(shape) <= 2:
        return shape
    nav_shape, frame_shape = shape[:-2], shape[-2:]
    frame_bytes = max(1, int(np.prod(frame_shape)) * np.dtype(dataset.dtype).itemsize)
    frames_per_chunk = max(1, CHUNK_TARGET_BYTES // frame_bytes)
    edge = _largest_power_of_two_at_most(
        int(frames_per_chunk ** (1.0 / len(nav_shape))))
    edge = min(NAV_BLOCK_MAX, edge)
    storage = dataset.chunks
    block = []
    for axis, size in enumerate(nav_shape):
        axis_edge = edge
        if storage is not None and storage[axis] > 1:
            axis_edge = max(storage[axis], axis_edge - axis_edge % storage[axis])
        block.append(min(size, axis_edge))
    return tuple(block) + frame_shape


def _wrapped_source(data):
    """The live HDF5 dataset a ``da.from_array`` wrap reads from, when ``data``
    is that wrap (possibly transposed) and nothing else; None otherwise."""
    try:
        for name, layer in data.dask.layers.items():
            if not name.startswith("original-"):
                continue
            values = list(layer.values())
            if len(values) != 1:
                return None
            source = values[0]
            if isinstance(source, np.ndarray) or not hasattr(source, "__getitem__"):
                return None
            if tuple(int(n) for n in source.shape) != tuple(data.shape)[::-1]:
                return None
            return source
    except Exception:
        return None
    return None


def restore_storage_order(dictionary: dict) -> None:
    """Undo the reader's transpose on one rsciio signal dictionary, in place.

    The data is reversed back to the file's C order and the axes list with it,
    with ``index_in_array`` renumbered. The last two axes stay the signal
    (detector) axes and everything before them navigates: the scan for a
    datacube, time for a frame stack, nothing for a single image.

    For a lazy load the array is rebuilt as a plain ``da.from_array`` of the
    HDF5 dataset in storage order rather than transposed back: two transpose
    layers on top of the wrap hide the dataset from SpyDE's frame readers,
    which then serve every frame through a dask compute (6 ms) instead of an
    h5py read (0.2 ms)."""
    data = dictionary.get("data")
    axes = dictionary.get("axes") or []
    ndim = getattr(data, "ndim", 0)
    if ndim < 3 or len(axes) != ndim:
        return
    source = _wrapped_source(data) if hasattr(data, "dask") else None
    if source is not None:
        import dask.array as da
        dictionary["data"] = da.from_array(
            source, chunks=tuple(reversed(data.chunks)))
    else:
        dictionary["data"] = data.transpose()
    restored = []
    for position, axis in enumerate(reversed(axes)):
        axis = dict(axis)
        axis["index_in_array"] = position
        axis["navigate"] = position < ndim - 2
        restored.append(axis)
    dictionary["axes"] = restored


def apply() -> bool:
    """Patch ``EMD_NCEM`` so Direct Electron ``.de5`` files load (idempotent)."""
    global _applied
    if _applied:
        return True
    try:
        import h5py
        from rsciio.emd._emd_ncem import EMD_NCEM
    except Exception as e:                                    # pragma: no cover
        log.warning("spyde.external.rosettasciio.de5: rsciio.emd import failed "
                    "(%s) — .de5 files may not open", e)
        return False

    original_parse_axis = getattr(EMD_NCEM, "_parse_axis", None)
    original_read_dataset = getattr(EMD_NCEM, "_read_dataset", None)
    original_read_file = getattr(EMD_NCEM, "read_file", None)
    if None in (original_parse_axis, original_read_dataset, original_read_file):
        log.warning("spyde.external.rosettasciio.de5: EMD_NCEM lacks _parse_axis, "
                    "_read_dataset or read_file; upstream changed shape, leaving "
                    "the reader alone")
        return False
    if getattr(original_read_file, _PATCH_SENTINEL, False):
        _applied = True
        return True

    if _upstream_parses_column_axis(EMD_NCEM):
        log.debug("spyde.external.rosettasciio.de5: upstream already flattens the "
                  "axis; only the lazy-read and storage-order patches are applied")
    else:
        def _parse_axis(axis_data):
            if getattr(axis_data, "ndim", 0) > 1:
                axis_data = np.asarray(axis_data).ravel()
            return original_parse_axis(axis_data)

        EMD_NCEM._parse_axis = staticmethod(_parse_axis)

    def _read_dataset(self, dataset):
        # Upstream's is a staticmethod; binding it here is what gives the
        # wrapper access to the lazy flag read_file stored on the instance.
        lazy = getattr(self, "lazy", False)
        if not lazy or h5py.check_string_dtype(dataset.dtype):
            return original_read_dataset(dataset)
        return dataset, lazy_dataset_chunks(dataset)

    def read_file(self, file, *args, **kwargs):
        original_read_file(self, file, *args, **kwargs)
        if is_direct_electron_file(file):
            for dictionary in getattr(self, "dictionaries", []):
                restore_storage_order(dictionary)

    setattr(read_file, _PATCH_SENTINEL, True)
    EMD_NCEM._read_dataset = _read_dataset
    EMD_NCEM.read_file = read_file
    _applied = True
    return True
