"""Reading a crystal phase from a ``.cif``, with its atoms where the file put them.

``orix.crystal_map.Phase.from_cif`` re-expresses the structure in orix's own
Cartesian frame (``a`` along x, ``c*`` along z) by moving the lattice into that
frame while keeping each atom's Cartesian position. For a cell whose ``a`` is
not already along x in the parser's frame — every hexagonal, trigonal and
monoclinic/triclinic cell — the lattice turns and the atoms do not, so their
fractional coordinates come out wrong. Hexagonal BaTiO3 (COD 2009488) puts
Ba at (0.577, 0.577, z) instead of (1/3, 2/3, z), and the structure no longer
has the symmetry of its own space group.

Fractional coordinates do not depend on the Cartesian frame, so the file's own
fractions are the right ones in orix's frame too. Restoring them is a no-op for
the cells orix does not turn (cubic, tetragonal, orthorhombic) and stays a no-op
if orix moves the atoms itself one day.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


def phase_from_cif(path):
    """An orix ``Phase`` read from the ``.cif`` at *path*, named after the file,
    with the atoms' fractional coordinates exactly as the file gives them."""
    from diffpy.structure.parsers import p_cif
    from orix.crystal_map import Phase

    path = Path(path)
    parser = p_cif.P_cif()
    read = getattr(parser, "parse_file", None) or parser.parseFile
    structure = read(str(path))
    space_group = getattr(getattr(parser, "spacegroup", None), "number", None)
    if space_group is None:
        warnings.warn(f"Could not read the space group from {str(path)!r}")
    phase = Phase(path.stem, space_group, structure=structure)
    for atom, as_read in zip(phase.structure, structure):
        atom.xyz = np.array(as_read.xyz, dtype=float)
    return phase
