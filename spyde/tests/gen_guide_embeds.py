"""gen_guide_embeds.py — build the INTERACTIVE HTML embeds for the docs website
walkthroughs (Phase 5 of the docs overhaul).

    python -m spyde.tests.gen_guide_embeds            # write all embeds
    python -m spyde.tests.gen_guide_embeds <out.html> # write one embed to a path

The docs site (docs-site/) renders each guide step; a step with an ``embed``
field mounts a self-contained interactive .html in a sandboxed iframe INSTEAD of
a static screenshot. This script GENERATES those .html files so they're
reproducible, not hand-authored — it builds a small synthetic diffraction-vectors
dataset and runs it through the SAME builder the report export uses
(``spyde.actions.report.vectors_embed.vectors_explorer_html``), which packs the
vectors into the page and inlines the anyplotlib ESM. The result is a single
.html that navigates, integrates, and virtual-images ENTIRELY in the browser with
ZERO runtime Python (the precompute-embed model — no pyodide, no torch, no
backend).

Runs anywhere: NO GPU / torch needed. The vectors are built directly with
``SpyDEDiffractionVectors.from_arrays`` (the same CPU path
``gen_vectors_embed.synthetic_vectors`` uses), so nothing calls the GPU
find-vectors kernels. The synthetic set here is a step up from that fixture's
two-cluster toy: a reciprocal LATTICE of Bragg spots per scan position with a
grain boundary down the middle (the left grain and the right grain sit at
different lattice orientations), so the explorer's three interactions each tell a
real story on the docs page:

  • NAVIGATE (pointer): moving the crosshair across the boundary swaps which
    lattice's spots render in the diffraction-pattern panel.
  • INTEGRATE: a region over one grain sums that grain's lattice into a crisp
    spot pattern; a region straddling the boundary shows both.
  • VIRTUAL IMAGING: parking the DP detector on a spot unique to one grain lights
    up only that grain in the navigator VI — the classic dark/bright-field
    grain-contrast demo, live.

Output paths (docs-site/public/media/<guide>/<name>.html) are declared in
``EMBEDS`` and wired into the guide steps via ``embed: '<name>.html'``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# docs-site media dir, relative to the repo root (this file is spyde/tests/…).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MEDIA_ROOT = _REPO_ROOT / "docs-site" / "public" / "media"


def grain_lattice_vectors(nav=(14, 14)):
    """A synthetic 4D-STEM diffraction-vectors set with a GRAIN BOUNDARY.

    Two crystal grains meet down the middle of the scan: the left grain's
    reciprocal lattice is rotated one way, the right grain's the other. Each scan
    position carries a small lattice of Bragg spots (the direct beam + first-order
    reflections) at that grain's orientation, so:

      • the diffraction pattern rendered at a left-grain position differs
        (rotated spots) from a right-grain position — navigation is meaningful;
      • a spot that exists ONLY in one grain's lattice makes a clean VI
        grain-contrast demo.

    Vectors are calibrated in 1/Å over a 256×256 DP frame (k = 0 centred), disk
    radius 5 px. Built purely with numpy + ``SpyDEDiffractionVectors.from_arrays``
    — no GPU, no find-vectors kernel, no file. Returns a
    ``SpyDEDiffractionVectors``.
    """
    from spyde.signals.diffraction_vectors import (
        COL_INTENSITY, COL_KX, COL_KY, COL_TIME, N_COLS,
        SpyDEDiffractionVectors, _AxisLite,
    )

    ny, nx = nav
    rng = np.random.default_rng(7)

    # First-order reciprocal-lattice spots (excluding the direct beam) for a
    # simple square lattice, in 1/Å. |g| ~ 0.45 gives spots well inside the
    # ±1 1/Å frame; the two grains differ by a rotation so their spots don't
    # overlap.
    g = 0.45
    base = np.array([
        (g, 0.0), (-g, 0.0), (0.0, g), (0.0, -g),
        (g, g), (-g, -g), (g, -g), (-g, g),
    ], dtype=np.float32)

    def _rotate(pts, deg):
        t = np.deg2rad(deg)
        c, s = np.cos(t), np.sin(t)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        return pts @ R.T

    left_spots = _rotate(base, +12.0)    # left grain: +12° lattice
    right_spots = _rotate(base, -18.0)   # right grain: −18° lattice

    rows = []
    for iy in range(ny):
        for ix in range(nx):
            in_left = ix < nx // 2
            spots = left_spots if in_left else right_spots
            # Direct beam (k≈0) always present + bright.
            beam = np.zeros(N_COLS, np.float32)
            beam[0], beam[1] = ix, iy
            beam[COL_TIME] = -1.0
            beam[COL_KX] = rng.normal(0, 0.01)
            beam[COL_KY] = rng.normal(0, 0.01)
            beam[COL_INTENSITY] = 255.0
            rows.append(beam)
            # First-order reflections at this grain's orientation. A gentle
            # intensity ramp across the scan gives the VI real per-pixel contrast.
            ramp = 0.6 + 0.4 * (iy / max(1, ny - 1))
            for (kx, ky) in spots:
                r = np.zeros(N_COLS, np.float32)
                r[0], r[1] = ix, iy
                r[COL_TIME] = -1.0
                r[COL_KX] = kx + rng.normal(0, 0.008)
                r[COL_KY] = ky + rng.normal(0, 0.008)
                r[COL_INTENSITY] = float(np.clip((110.0 + 30 * rng.random()) * ramp,
                                                 1, 255))
                rows.append(r)

    flat = np.stack(rows).astype(np.float32)
    # 256-px DP frame spanning k ∈ [−1, 1] 1/Å (offset −1, scale 2/255).
    ax = _AxisLite(scale=2.0 / 255, offset=-1.0, size=256, units="1/A", name="k")
    return SpyDEDiffractionVectors.from_arrays(
        flat_buffer=flat, full_nav_shape=(ny, nx), sig_shape=(256, 256),
        sig_axes=[ax, ax], kernel_radius_px=5.0, kernel_radius_data=0.0,
        params={}, nav_axes=None,
    )


# Each embed: which guide it belongs to, its filename, and the caption shown
# under the explorer. One compelling interactive embed per applicable guide.
_FV_CAPTION = (
    "Interactive — drag the crosshair across the scan to swap which grain's "
    "diffraction pattern you see; switch to Integrate to sum a region; drag the "
    "green detector on the pattern to virtual-image a spot back onto the scan."
)
_VI_CAPTION = (
    "Interactive — drag the green detector over a diffraction spot: the scan map "
    "lights up wherever that spot appears (a live virtual image). Move it to a "
    "spot from the other grain and the contrast flips."
)

EMBEDS = [
    {"guide": "find-vectors", "name": "vectors-explorer.html", "caption": _FV_CAPTION},
    {"guide": "virtual-imaging", "name": "vectors-explorer.html", "caption": _VI_CAPTION},
]


def build_embed(caption: str) -> str:
    """Build ONE interactive explorer page (self-contained HTML string) from the
    synthetic grain-lattice vectors, via the report export's explorer builder."""
    from spyde.actions.report.vectors_embed import vectors_explorer_html

    html = vectors_explorer_html(grain_lattice_vectors(), caption=caption)
    if html is None:
        raise RuntimeError("vectors_explorer_html returned None (empty/over-cap)")
    return html


def _write(path: Path, caption: str) -> None:
    html = build_embed(caption)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"wrote {path} ({len(html)} bytes)")


def main(argv: "list[str]") -> None:
    if len(argv) >= 1:
        # Single-output mode (used by the spec): write one embed to the given path.
        _write(Path(argv[0]), _FV_CAPTION)
        return
    # Default: (re)generate every declared embed into the docs media dir.
    for spec in EMBEDS:
        _write(_MEDIA_ROOT / spec["guide"] / spec["name"], spec["caption"])


if __name__ == "__main__":
    main(sys.argv[1:])
