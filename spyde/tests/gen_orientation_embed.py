"""gen_orientation_embed.py — build a synthetic orientation map + its
report-embed IPF explorer HTML (the shared fixture for the unit tests and the
real-browser spec).

    python -m spyde.tests.gen_orientation_embed <out.html>

Designed for crisp assertions, the same way ``gen_vectors_embed`` is: the LEFT
half of the nav grid is ONE crystal orientation and the RIGHT half is a
different one, with a small per-pixel jitter so the map is not a solid block of
two colours. So a pick in the left half and a pick in the right half must land
in visibly different places on the triangle AND swing the sphere's camera — if
either stays put, the pick is not driving the view.
"""
from __future__ import annotations

import sys

import numpy as np


def synthetic_orientation_map(nav=(16, 16)):
    from orix.crystal_map import Phase
    from orix.quaternion import Rotation
    from diffpy.structure import Atom, Lattice, Structure
    from spyde.signals.orientation_map import SpyDEOrientationMap, phase_to_dict

    ny, nx = nav
    structure = Structure(atoms=[Atom("Al", [0, 0, 0])],
                          lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90))
    phase = Phase(name="Al", space_group=225, structure=structure)

    # Two well-separated orientations, one per nav half. Euler angles rather
    # than raw quaternions so the two really are a large rotation apart.
    left = Rotation.from_euler(np.deg2rad([[0.0, 0.0, 0.0]]))
    right = Rotation.from_euler(np.deg2rad([[35.0, 42.0, 15.0]]))

    rng = np.random.default_rng(0)
    quats = np.zeros((ny, nx, 1, 4), np.float32)
    for iy in range(ny):
        for ix in range(nx):
            base = (left if ix < nx // 2 else right).data[0]
            # ±1.5° of jitter: enough to spread the cloud, far too little to
            # blur the two halves into each other.
            jit = Rotation.from_euler(np.deg2rad(rng.normal(0, 1.5, 3))).data[0]
            q = Rotation(np.asarray(base)) * Rotation(np.asarray(jit))
            quats[iy, ix, 0] = np.asarray(q.data[0], np.float32)

    corr = np.ones((ny, nx, 1), np.float32)
    phase_idx = np.zeros((ny, nx, 1), np.int16)
    mirror = np.ones((ny, nx, 1), np.int8)
    return SpyDEOrientationMap(quats, corr, phase_idx, mirror,
                               [phase_to_dict(phase)])


def main(out_path: str) -> None:
    from spyde.actions.report.orientation_embed import orientation_explorer_html

    html = orientation_explorer_html(synthetic_orientation_map(),
                                     caption="synthetic orientation embed")
    assert html is not None
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {out_path} ({len(html)} bytes)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "orientation_embed.html")
