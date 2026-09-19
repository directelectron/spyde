"""Additions to the em-database example catalogue.

Unlike the rest of ``spyde.external``, which patches upstream behaviour, this
subpackage ADDS a dataset — the same stretch of the package's remit that
``rosettasciio.csb_format`` makes for a file format, and for the same reason:
it is the only place SpyDE can teach an installed em-database about something
it does not yet ship, and the alternative is special-casing one dataset inside
the Examples menu, where dataset declarations do not belong.
"""
from __future__ import annotations

from spyde.external import register
from spyde.external.emdatabase.sped_ag import apply as _apply_sped_ag

register("em-database", _apply_sped_ag)

__all__ = ["_apply_sped_ag"]
