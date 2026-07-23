"""
spyde.external.rosettasciio — patches / read-path verification for RosettaSciIO
(the ``.mrc`` / ``.tif`` / ``.hspy`` reader stack).

History: SpyDE used to pin a ``cssfrancis/rosettasciio@win32-binary-read`` fork
for a fast MRC read. As of the current pyproject.toml that fork is DROPPED — we
resolve stock ``rosettasciio[hdf5,image]>=0.14.0`` from PyPI, whose
``memmap_distributed`` (``np.memmap`` + ``dask.map_blocks(slice_memmap)``) already
gives a fast per-chunk lazy read (measured ~6 ms cold / ~3 ms warm per 256x256
frame on an 11 GB in-situ .mrc — see :mod:`spyde.external.rosettasciio.mrc`).

Modules:
* :mod:`~spyde.external.rosettasciio.mrc` — VERIFIES the stock fast MRC read is
  active (no monkeypatch needed with 0.14.0); logs a warning if a future
  rosettasciio drops ``memmap_distributed`` so the slow path can be spotted.
* :mod:`~spyde.external.rosettasciio.tiff` — home for the per-page lazy TIFF
  chunking patch (rsciio's lazy TIFF returns ONE monolithic chunk). Currently a
  documented no-op in this branch — see the module docstring.
"""
from __future__ import annotations

from spyde.external import register
from spyde.external.rosettasciio.mrc import apply as _apply_mrc
from spyde.external.rosettasciio.tiff import apply as _apply_tiff

register("rosettasciio", _apply_mrc)
register("rosettasciio", _apply_tiff)

__all__ = ["_apply_mrc", "_apply_tiff"]
