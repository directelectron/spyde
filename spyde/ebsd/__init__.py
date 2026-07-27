"""spyde.ebsd — EBSD indexing and refinement (0.3.0 Wave 3, GitHub #68).

**Scope decision.** kikuchipy supplies the things that are hard to get right and
easy to get from a maintained library: signal classes, vendor IO, detector
geometry and master-pattern simulation. Dictionary indexing and refinement are
implemented here in torch on purpose — those are the slow steps, and making
them fast is the entire reason this wave exists.

The display side is almost free: SpyDE already depends on **orix** and already
has IPF views, orientation maps and a 3D IPF toolbar from the 4D-STEM work, so
an indexed ``CrystalMap`` feeds straight into existing code (#73).

Nothing in this package requires the ``ebsd`` extra — indexing is plain
torch + numpy, so it can be developed and tested without kikuchipy installed.
Only the IO/geometry/master-pattern layer (#69) needs it, and that is gated
with ``requires_package``.
"""
from __future__ import annotations

from spyde.ebsd.indexing import (
    IndexingResult,
    dictionary_index,
    sample_orientations,
)
from spyde.ebsd.preprocess import average_dot_product_map, remove_background
from spyde.ebsd.crystal_map import (
    ipf_colors,
    merge_phases,
    orientation_similarity_map,
    to_crystal_map,
)
from spyde.ebsd.refine import RefinementResult, refine_orientations

__all__ = ["dictionary_index", "sample_orientations", "IndexingResult",
           "remove_background", "average_dot_product_map",
           "refine_orientations", "RefinementResult",
           "to_crystal_map", "ipf_colors", "orientation_similarity_map",
           "merge_phases"]
