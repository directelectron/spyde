"""unit_cell.py — a phase's unit cell as atoms and edges, in the crystal frame.

Cartesian coordinates come from the same orix/diffpy lattice the band model's
reflectors are built on (``Phase.structure.lattice.base``), so a cell drawn
with these coordinates and the band sphere turn together under one rotation.
"""
from __future__ import annotations

import itertools

import numpy as np

#: Display colours for common elements (Jmol's scheme); others get a grey.
ELEMENT_COLOURS = {
    "H": "#ffffff", "C": "#909090", "N": "#3050f8", "O": "#ff0d0d",
    "F": "#90e050", "Na": "#ab5cf2", "Mg": "#8aff00", "Al": "#bfa6a6",
    "Si": "#f0c8a0", "P": "#ff8000", "S": "#ffff30", "Cl": "#1ff01f",
    "K": "#8f40d4", "Ca": "#3dff00", "Ti": "#bfc2c7", "V": "#a6a6ab",
    "Cr": "#8a99c7", "Mn": "#9c7ac7", "Fe": "#e06633", "Co": "#f090a0",
    "Ni": "#50d050", "Cu": "#c88033", "Zn": "#7d80b0", "Ga": "#c28f8f",
    "Ge": "#668f8f", "As": "#bd80e3", "Zr": "#94e0e0", "Nb": "#73c2c9",
    "Mo": "#54b5b5", "Ag": "#c0c0c0", "Sn": "#668080", "Hf": "#4dc2ff",
    "Ta": "#4da6ff", "W": "#2194d6", "Pt": "#d0d0e0", "Au": "#ffd123",
    "Pb": "#575961",
}


def element_symbol(name: str) -> str:
    """``"Ag"`` from a CIF atom label like ``"Ag1"`` or ``"Fe2+"``."""
    letters = "".join(ch for ch in str(name) if ch.isalpha())
    return letters[:2].capitalize() if len(letters) > 1 and letters[1].islower() \
        else letters[:1].upper()


def unit_cell(phase) -> tuple[np.ndarray, list[str], np.ndarray]:
    """``(positions, elements, edges)`` of *phase*'s unit cell.

    *positions* ``(N, 3)`` are Cartesian, in Å, every atom of the cell plus
    its periodic images on the faces, edges and corners — so a cubic cell
    shows its 8 corner atoms rather than one. *edges* is ``(12, 2, 3)``.
    """
    structure = phase.structure
    base = np.asarray(structure.lattice.base, float)       # rows: a, b, c
    fractional, elements = [], []
    for atom in structure:
        xyz = np.mod(np.asarray(atom.xyz, float), 1.0)
        xyz[np.isclose(xyz, 1.0, atol=1e-6)] = 0.0
        # An atom on a face, edge or corner has images on the opposite side.
        shifts = [(0.0, 1.0) if np.isclose(v, 0.0, atol=1e-6) else (0.0,)
                  for v in xyz]
        for shift in itertools.product(*shifts):
            image = xyz + shift
            if not any(np.allclose(image, seen, atol=1e-6) for seen in fractional):
                fractional.append(image)
                elements.append(element_symbol(atom.element))
    positions = np.asarray(fractional) @ base

    corners = np.array(list(itertools.product((0, 1), repeat=3)), float)
    edges = np.array([[corners[i] @ base, corners[j] @ base]
                      for i, j in itertools.combinations(range(8), 2)
                      if np.abs(corners[i] - corners[j]).sum() == 1])
    return positions, elements, edges
