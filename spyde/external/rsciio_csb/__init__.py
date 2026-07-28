"""CSB — Direct Electron compressed-sparse-block centroid streams.

**This package is shaped as a RosettaSciIO plugin, not as SpyDE code.** Its
layout is rsciio's exactly — ``__init__.py`` re-exporting ``file_reader``,
the implementation in ``_api.py``, and a ``specifications.yaml`` beside them —
so moving it upstream is a directory copy into ``rsciio/csb/`` plus deleting
SpyDE's runtime registration (``spyde.external.rosettasciio.csb_format``);
rsciio auto-discovers the yaml and needs no registration at all.

Nothing in here imports SpyDE. Keep it that way — the moment it does, the
directory copy stops working.

``_core`` and ``_sparse`` are vendored verbatim from the ``de-csb`` project
(``csb.py`` / ``csb_sparse.py``); the only edit is the import between them
being made relative.
"""
from ._api import file_reader

__all__ = [
    "file_reader",
]


def __dir__():
    return sorted(__all__)
