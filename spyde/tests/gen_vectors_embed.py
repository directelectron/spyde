"""gen_vectors_embed.py — build a synthetic vectors dataset + its report-embed
explorer HTML (the shared fixture for the unit tests and the real-browser spec).

    python -m spyde.tests.gen_vectors_embed <out.html>

The synthetic set is designed for crisp assertions: cluster A at k=(-0.5, 0)
exists ONLY in the LEFT half of the nav grid; cluster B at k=(+0.5, 0) ONLY in
the RIGHT half. Because kx maps to the DP column, a POINTER on a left-half nav
position renders its disks in the LEFT of the diffraction pattern (and a
right-half position in the RIGHT); an INTEGRATE region over the left half sums
all of cluster A's disks into a bright left-side blob.
"""
from __future__ import annotations

import sys

import numpy as np


def synthetic_vectors(nav=(16, 16)):
    from spyde.signals.diffraction_vectors import (
        COL_INTENSITY, COL_KX, COL_KY, COL_TIME, N_COLS,
        SpyDEDiffractionVectors, _AxisLite,
    )

    ny, nx = nav
    rows = []
    rng = np.random.default_rng(0)
    for iy in range(ny):
        for ix in range(nx):
            k = (-0.5, 0.0) if ix < nx // 2 else (0.5, 0.0)
            for _ in range(3):                       # 3 vectors per position
                r = np.zeros(N_COLS, np.float32)
                r[0], r[1] = ix, iy
                r[COL_TIME] = -1.0
                r[COL_KX] = k[0] + rng.normal(0, 0.02)
                r[COL_KY] = k[1] + rng.normal(0, 0.02)
                r[COL_INTENSITY] = 100.0 + 10 * rng.random()
                rows.append(r)
    flat = np.stack(rows).astype(np.float32)

    ax = _AxisLite(scale=2.0 / 255, offset=-1.0, size=256, units="1/A", name="k")
    return SpyDEDiffractionVectors.from_arrays(
        flat_buffer=flat, full_nav_shape=(ny, nx), sig_shape=(256, 256),
        sig_axes=[ax, ax], kernel_radius_px=4.0, kernel_radius_data=0.0,
        params={}, nav_axes=None,
    )


def synthetic_vectors_5d(nt=4, nav=(16, 16)):
    """A 5-D STACK fixture built for the same kind of crisp assertion, but along
    the STACK axis: slice ``t``'s vectors sit at ``kx`` swept left→right across
    the stack, and each position holds ``t + 1`` of them.

    So a page showing the wrong slice is unmistakable twice over — the DP energy
    is on the wrong side AND the vector count is wrong. A page that ignores the
    stack axis entirely (the old behaviour: every slice piled onto one position)
    shows disks on BOTH sides and a count of ``nt(nt+1)/2``.
    """
    from spyde.signals.diffraction_vectors import (
        COL_INTENSITY, COL_KX, COL_KY, COL_TIME, N_COLS,
        SpyDEDiffractionVectors, _AxisLite,
    )

    ny, nx = nav
    rows = []
    rng = np.random.default_rng(1)
    for t in range(nt):
        kx = -0.75 + 1.5 * (t / max(1, nt - 1))     # sweeps left → right
        for iy in range(ny):
            for ix in range(nx):
                for _ in range(t + 1):              # count identifies the slice
                    r = np.zeros(N_COLS, np.float32)
                    r[0], r[1] = ix, iy
                    r[COL_TIME] = t
                    r[COL_KX] = kx + rng.normal(0, 0.02)
                    r[COL_KY] = rng.normal(0, 0.02)
                    r[COL_INTENSITY] = 100.0 + 10 * rng.random()
                    rows.append(r)
    flat = np.stack(rows).astype(np.float32)

    ax = _AxisLite(scale=2.0 / 255, offset=-1.0, size=256, units="1/A", name="k")
    return SpyDEDiffractionVectors.from_arrays(
        flat_buffer=flat, full_nav_shape=(nt, ny, nx), sig_shape=(256, 256),
        sig_axes=[ax, ax], kernel_radius_px=4.0, kernel_radius_data=0.0,
        params={}, nav_axes=None,
    )


def main(out_path: str, stack: bool = False) -> None:
    from spyde.actions.report.vectors_embed import vectors_explorer_html

    vecs = synthetic_vectors_5d() if stack else synthetic_vectors()
    html = vectors_explorer_html(
        vecs, caption="synthetic stack embed" if stack else "synthetic embed")
    assert html is not None
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {out_path} ({len(html)} bytes)")


if __name__ == "__main__":
    main(sys.argv[1], stack="--5d" in sys.argv[2:])
