"""
Direct Electron ``.de5`` files: rsciio's EMD reader cannot parse their axes, and
when it can, it hands the datacube back transposed with no navigation axes.

WHAT
----
A ``.de5`` is an EMD (HDF5) file, read by :mod:`rsciio.emd`'s Berkeley-variant
reader. Direct Electron's writer differs from that reader's assumptions in two
ways, and this module wraps two of its methods to absorb them:

1. Each calibration axis (``dim1`` … ``dim4``) is a column vector of shape
   ``(N, 1)`` rather than the ``(N,)`` vector the EMD spec shows.
   ``EMD_NCEM._parse_axis`` does ``float(axis_data[0])`` on it, which on a
   ``(1,)`` row is a ``TypeError`` under NumPy ≥ 1.25 ("only 0-dimensional
   arrays can be converted to Python scalars"), so every load died in
   ``hs.load``. The wrapper flattens the axis first.

2. The datacube is stored in C order, scan axes first: ``(R_y, R_x, Q_y, Q_x)``
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
Both fixes belong upstream, but a user with a ``.de5`` cannot wait for a
rosettasciio release. The wrappers are the smallest changes that make the
stock reader accept the files the camera actually writes.

WHEN TO REMOVE
--------------
When the minimum ``rosettasciio`` in ``pyproject.toml`` flattens the axis in
``_parse_axis`` and recognises Direct Electron files as C-ordered datacubes.
``apply()`` probes the installed ``_parse_axis`` with a ``(3, 1)`` array first,
so once that half is fixed upstream it becomes a logged no-op rather than a
double patch.
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


def restore_storage_order(dictionary: dict) -> None:
    """Undo the reader's transpose on one rsciio signal dictionary, in place.

    The data is reversed back to the file's C order and the axes list with it,
    with ``index_in_array`` renumbered. The last two axes stay the signal
    (detector) axes and everything before them navigates: the scan for a
    datacube, time for a frame stack, nothing for a single image."""
    data = dictionary.get("data")
    axes = dictionary.get("axes") or []
    ndim = getattr(data, "ndim", 0)
    if ndim < 3 or len(axes) != ndim:
        return
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
        from rsciio.emd._emd_ncem import EMD_NCEM
    except Exception as e:                                    # pragma: no cover
        log.warning("spyde.external.rosettasciio.de5: rsciio.emd import failed "
                    "(%s) — .de5 files may not open", e)
        return False

    original_parse_axis = getattr(EMD_NCEM, "_parse_axis", None)
    original_read_file = getattr(EMD_NCEM, "read_file", None)
    if original_parse_axis is None or original_read_file is None:
        log.warning("spyde.external.rosettasciio.de5: EMD_NCEM lacks _parse_axis "
                    "or read_file; upstream changed shape, leaving the reader alone")
        return False
    if getattr(original_read_file, _PATCH_SENTINEL, False):
        _applied = True
        return True

    if _upstream_parses_column_axis(EMD_NCEM):
        log.debug("spyde.external.rosettasciio.de5: upstream already flattens the "
                  "axis; only the storage-order patch is applied")
    else:
        def _parse_axis(axis_data):
            if getattr(axis_data, "ndim", 0) > 1:
                axis_data = np.asarray(axis_data).ravel()
            return original_parse_axis(axis_data)

        EMD_NCEM._parse_axis = staticmethod(_parse_axis)

    def read_file(self, file, *args, **kwargs):
        original_read_file(self, file, *args, **kwargs)
        if is_direct_electron_file(file):
            for dictionary in getattr(self, "dictionaries", []):
                restore_storage_order(dictionary)

    setattr(read_file, _PATCH_SENTINEL, True)
    EMD_NCEM.read_file = read_file
    _applied = True
    return True
