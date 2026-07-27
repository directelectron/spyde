"""spyde.atoms — atom position mapping (0.3.0 Wave 4, GitHub #74).

**atomap owns the structure, SpyDE owns the refinement.** Initial peak finding,
sublattices, nearest neighbours, zone axes and dumbbell pairing are atomap's,
because reimplementing them would mean diverging from published atomap results.
Refinement is ours because refining atom positions *is* a batched 2-D gaussian
fit, and :mod:`spyde.fitting` already does the whole field in one Levenberg-
Marquardt where atomap fits one atom at a time with scipy.

atomap's GUI functions (``select_atoms_with_gui``, ``add_atoms_with_gui``,
``toggle_atom_refine_position_with_gui``) are matplotlib-based, so the
*interaction* is reimplemented over SpyDE's own anyplotlib overlay (#76) —
the same machinery ``actions/vector_overlay.py`` already uses for diffraction
vectors.

Only :func:`~spyde.atoms.finding.find_atoms` needs the ``atoms`` extra;
refinement and the property maps are plain numpy/torch, so they are testable
and usable without it.
"""
from __future__ import annotations

from spyde.atoms.finding import (
    MissingExtra,
    find_atoms,
    refine_atoms,
    refine_center_of_mass,
    refine_gaussian,
)
from spyde.atoms.properties import (
    displacement_from_ideal,
    displacement_magnitude,
    ellipticity,
    ellipticity_angle,
    intensity,
    nearest_neighbour_distance,
    property_maps,
    to_map,
)

__all__ = [
    "find_atoms", "refine_atoms", "refine_center_of_mass", "refine_gaussian",
    "MissingExtra",
    "ellipticity", "ellipticity_angle", "intensity",
    "nearest_neighbour_distance", "displacement_from_ideal",
    "displacement_magnitude", "property_maps", "to_map",
]
