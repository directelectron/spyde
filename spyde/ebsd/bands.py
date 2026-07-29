"""bands.py — Kikuchi band geometry: the projection shared by every consumer.

Three parts of Wave 3 need the same projection and must not disagree about it:

* the synthetic data generator (:mod:`spyde.data.synthetic`), which renders
  patterns from known orientations,
* the dictionary simulator (:class:`spyde.ebsd.refine.BandSimulator`), which
  renders the entries an experimental pattern is matched against,
* the **live overlay**, which draws the indexed orientation's bands back on top
  of the experimental pattern.

If any two of those drift the overlay silently lies: the lines sit beside the
bands and there is nothing to tell you whether the indexing was wrong or only
the drawing. So the geometry lives here, once, and the other three import it.

The geometry itself is the flat-detector gnomonic projection. A detector pixel
sees the crystal along a direction ``r``; a band appears where ``r`` is nearly
perpendicular to a plane normal ``n``. That gives two things:

* **the band centre**, ``r·n = 0``. Because ``r`` is linear in the pixel
  coordinates and the ``|r|`` normalisation is a positive scalar, this is
  exactly a straight LINE on the detector — which is why a band overlay is
  drawn with line segments and not with a sampled curve.
* **the band edges**, ``r̂·n = ±w``, which are conics. They are not drawn: the
  centre line is what tells you whether the orientation is right, and a real
  pattern's band edges are diffuse anyway. This is also what kikuchipy's
  geometrical simulation draws.

Nothing here needs torch or the ``ebsd`` extra — it is numpy and, only for
:func:`reflectors_from_phase`, diffsims (which SpyDE already depends on).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# Electron wavelength in Å at an accelerating voltage in kV (relativistic).
_H, _M0, _E, _C = 6.62607015e-34, 9.1093837015e-31, 1.602176634e-19, 299792458.0


def wavelength(voltage_kv: float) -> float:
    """Relativistic electron wavelength in Å. EBSD runs at 10–30 kV, where the
    non-relativistic form is already ~1% off, so the correction is kept."""
    v = float(voltage_kv) * 1e3
    lam = _H / np.sqrt(2 * _M0 * _E * v * (1 + _E * v / (2 * _M0 * _C ** 2)))
    return float(lam * 1e10)


@dataclass
class Reflectors:
    """The set of diffracting planes a pattern is built from.

    normals : (N, 3) unit plane normals in the CRYSTAL cartesian frame, one per
        Friedel pair — ``+g`` and ``-g`` are the same band, and keeping both
        would draw every line twice.
    weights : (N,) relative band intensity.
    widths : (N,) band half-width, in units of ``r̂·n`` (i.e. the sine of the
        angle between the beam direction and the plane). Only the simulator
        uses this; the overlay draws centre lines.
    hkl : (N, 3) Miller indices, kept for labelling. Optional.
    """

    normals: np.ndarray
    weights: np.ndarray
    widths: np.ndarray
    hkl: np.ndarray | None = None

    def __post_init__(self):
        self.normals = np.asarray(self.normals, float).reshape(-1, 3)
        n = len(self.normals)
        self.weights = np.broadcast_to(np.asarray(self.weights, float), (n,)).copy()
        self.widths = np.broadcast_to(np.asarray(self.widths, float), (n,)).copy()
        if self.hkl is not None:
            self.hkl = np.asarray(self.hkl).reshape(n, 3)

    def __len__(self) -> int:
        return len(self.normals)

    def brightest(self, n: int | None) -> "Reflectors":
        """The *n* strongest bands. Drawing every reflector of a real phase
        turns the pattern into a grey mesh, so the overlay draws a subset — and
        the strongest bands are the ones actually visible in the data."""
        if n is None or n >= len(self):
            return self
        keep = np.argsort(self.weights)[::-1][:max(1, int(n))]
        keep = np.sort(keep)
        return Reflectors(self.normals[keep], self.weights[keep],
                          self.widths[keep],
                          None if self.hkl is None else self.hkl[keep])


def _dedupe_friedel(normals: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Indices of one normal per ``±n`` pair, keeping the first occurrence."""
    keep: list[int] = []
    kept: list[np.ndarray] = []
    for i, v in enumerate(normals):
        if any(abs(abs(float(v @ k)) - 1.0) < tol for k in kept):
            continue
        keep.append(i)
        kept.append(v)
    return np.asarray(keep, int)


def cubic_reflectors() -> Reflectors:
    """The {111}/{200}/{220} band set of a generic cubic crystal.

    The default when no crystal structure has been loaded, and the exact set
    :func:`spyde.data.synthetic.ebsd_patterns` renders — so the overlay can be
    verified against the bundled synthetic scan pixel for pixel. The widths are
    deliberately WIDER than a real 20 kV pattern's: they are what makes bands
    visible on a 60 px detector, and a dictionary built from these has to match
    data drawn with these.
    """
    fams = [((1, 1, 1), 1.00), ((2, 0, 0), 0.70), ((2, 2, 0), 0.45)]
    normals, weights, hkls = [], [], []
    for hkl, w in fams:
        seen: set[tuple[int, int, int]] = set()
        h, k, l = hkl
        for perm in {(h, k, l), (h, l, k), (k, h, l), (k, l, h), (l, h, k), (l, k, h)}:
            for sx in (1, -1):
                for sy in (1, -1):
                    for sz in (1, -1):
                        v = (perm[0] * sx, perm[1] * sy, perm[2] * sz)
                        if v == (0, 0, 0) or tuple(-x for x in v) in seen:
                            continue
                        seen.add(v)
        for v in sorted(seen):
            g = np.array(v, float)
            normals.append(g / np.linalg.norm(g))
            # Band width goes as 1/|g|: bigger d-spacing -> wider band.
            weights.append(w / np.linalg.norm(g))
            hkls.append(v)
    weights_arr = np.array(weights)
    widths = 0.055 * (weights_arr / weights_arr.max()) + 0.012
    return Reflectors(np.array(normals), weights_arr, widths, np.array(hkls))


def reflectors_from_phase(phase, *, min_dspacing: float = 0.7,
                          voltage_kv: float = 20.0,
                          max_bands: int | None = 60) -> Reflectors:
    """The reflectors of a real crystal structure, via diffsims.

    ``phase`` is an orix :class:`~orix.crystal_map.Phase` — normally
    ``Phase.from_cif(path)``, the same door the 4D-STEM orientation wizard uses.
    Band intensity is ``|F_hkl|`` and the half-width is the Bragg angle
    ``λ / 2d``, so at EBSD voltages the bands come out an order of magnitude
    narrower than :func:`cubic_reflectors`' exaggerated synthetic ones — which
    is correct, and the reason width is a per-reflector property rather than a
    constant.

    Falls back to :func:`cubic_reflectors` if the phase carries no structure
    (a ``Phase(space_group=…)`` with no atoms cannot give structure factors).
    """
    from diffsims.crystallography import ReciprocalLatticeVector

    try:
        rlv = ReciprocalLatticeVector.from_min_dspacing(
            phase, min_dspacing=float(min_dspacing))
        rlv = rlv.unique(use_symmetry=True).symmetrise()
        rlv.sanitise_phase()
        rlv.calculate_structure_factor()
        # |F_hkl| — the structure factor is COMPLEX, and asarray(..., float)
        # would silently take the real part (a reflector whose phase is near
        # 90 degrees would come out extinct).
        factor = np.abs(np.asarray(rlv.structure_factor)).astype(float)
        normals = np.asarray(rlv.unit.data, float).reshape(-1, 3)
        d = np.asarray(rlv.dspacing, float).reshape(-1)
        hkl = np.asarray(rlv.hkl, float).reshape(-1, 3)
    except Exception as e:
        log.warning("reflectors for %s could not be computed from the "
                    "structure (%s) — falling back to a generic cubic band set",
                    getattr(phase, "name", phase), e)
        return cubic_reflectors()

    finite = np.isfinite(factor) & np.isfinite(d) & (d > 0)
    finite &= np.isfinite(normals).all(1)
    # A structureless Phase — `Phase(space_group=225)` with no atoms — still
    # yields hkl, but every structure factor is ZERO, so the "reflectors" are
    # all extinct and include ones a real fcc crystal cannot show ({100},
    # {110}). Weightless bands would draw a lattice of lines with nothing
    # behind them, so treat it as no structure at all.
    if not finite.any() or float(factor[finite].max()) <= 0:
        log.debug("phase %s has no usable structure factors — using the "
                  "generic cubic band set", getattr(phase, "name", phase))
        return cubic_reflectors()
    normals, factor, d, hkl = normals[finite], factor[finite], d[finite], hkl[finite]

    keep = _dedupe_friedel(normals)
    normals, factor, d, hkl = normals[keep], factor[keep], d[keep], hkl[keep]

    lam = wavelength(voltage_kv)
    widths = np.clip(lam / (2.0 * d), 1e-4, 0.25)
    weights = factor / max(float(factor.max()), 1e-12)
    refl = Reflectors(normals, weights, widths, hkl.astype(int))
    return refl.brightest(max_bands)


def detector_directions(detector=(60, 60), pc=(0.5, 0.5, 0.55)) -> np.ndarray:
    """Unit vectors from the sample to each detector pixel (gnomonic).

    ``(dy, dx, 3)``. ``pc`` is ``(pcx, pcy, L)`` in the fractional convention:
    the pattern centre as a fraction of the detector width/height, and the
    detector distance as a fraction of the width.
    """
    pcx, pcy, L = float(pc[0]), float(pc[1]), float(pc[2])
    dy, dx = int(detector[0]), int(detector[1])
    gy, gx = np.mgrid[0:dy, 0:dx].astype(float)
    rx = (gx + 0.5) / dx - pcx
    ry = pcy - (gy + 0.5) / dy                    # detector y is flipped
    r = np.stack([rx, ry, np.full_like(rx, L)], -1)
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def euler_to_matrix(phi1, Phi, phi2) -> np.ndarray:
    """Bunge ZXZ Euler angles -> rotation matrices, batched over the leading
    axes. Returns ``(..., 3, 3)``."""
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    m = np.empty(np.shape(phi1) + (3, 3), float)
    m[..., 0, 0] = c1 * c2 - s1 * s2 * c
    m[..., 0, 1] = s1 * c2 + c1 * s2 * c
    m[..., 0, 2] = s2 * s
    m[..., 1, 0] = -c1 * s2 - s1 * c2 * c
    m[..., 1, 1] = -s1 * s2 + c1 * c2 * c
    m[..., 1, 2] = c2 * s
    m[..., 2, 0] = s1 * s
    m[..., 2, 1] = -c1 * s
    m[..., 2, 2] = c
    return m


def normals_to_sample(normals, rot):
    """Plane normals from the CRYSTAL frame into the SAMPLE frame.

    ``rot`` is the Bunge matrix from :func:`euler_to_matrix`, which maps SAMPLE
    components to CRYSTAL components (the orix convention — the two agree, and
    ``test_ebsd_bands`` pins that). So going the other way needs its TRANSPOSE,
    which for row-vector normals is a plain ``normals @ rot``.

    Getting this backwards is invisible in an indexing score — the dictionary
    and the data would simply share the mistake and match each other perfectly
    — and shows up only where the Euler angles leave this module: the IPF map
    would then be coloured from the INVERSE orientation, still showing grains
    and gradients, just the wrong colours. The tell is symmetry: the simulated
    pattern must be unchanged by a CRYSTAL symmetry operation (Bunge phi2 + 90
    degrees for a cubic 4-fold) and must change under a SAMPLE rotation
    (phi1 + 90). Transposed, that is exactly reversed.
    """
    return np.asarray(normals) @ np.asarray(rot)


def simulate_patterns(euler, reflectors: Reflectors | None = None,
                      detector=(60, 60), pc=(0.5, 0.5, 0.55), *,
                      background: bool = False) -> np.ndarray:
    """Kikuchi patterns for a list of orientations -> ``(N, dy, dx)`` float32.

    *euler* is ``(N, 3)`` Bunge angles in radians. Pure numpy and looped over
    N: this is for a handful of patterns (a preview, a test). Building a
    DICTIONARY of thousands goes through
    :class:`spyde.ebsd.refine.BandSimulator`, which is the same arithmetic
    batched in torch.

    ``background=False`` by default because a dictionary is matched by
    normalised cross-correlation, which is invariant to the smooth gradient a
    real detector adds — and the experimental side has background removal
    applied before matching anyway (:mod:`spyde.ebsd.preprocess`).
    """
    refl = reflectors if reflectors is not None else cubic_reflectors()
    euler = np.atleast_2d(np.asarray(euler, float))
    rot = euler_to_matrix(euler[:, 0], euler[:, 1], euler[:, 2])    # (N, 3, 3)
    r = detector_directions(detector, pc)
    dy, dx = r.shape[:2]
    flat_r = r.reshape(-1, 3)

    out = np.empty((len(euler), dy, dx), np.float32)
    for i in range(len(euler)):
        n_rot = normals_to_sample(refl.normals, rot[i])
        d = flat_r @ n_rot.T
        band = np.exp(-0.5 * (d / refl.widths) ** 2) * refl.weights
        out[i] = band.sum(1).reshape(dy, dx).astype(np.float32)
    if background:
        gy, gx = np.mgrid[0:dy, 0:dx].astype(np.float32)
        out = out / max(out.max(), 1e-9) + (
            0.35 + 0.30 * ((gy / dy) * 0.6 + (gx / dx) * 0.4))
    return out


def _clip_line_to_box(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                      width: float, height: float):
    """Clip the lines ``a·u + b·v + c = 0`` to ``[0, width] x [0, height]``.

    Vectorised over the bands. Returns ``(segments (M, 2, 2), keep (N,) bool)``
    — the ``[[u0, v0], [u1, v1]]`` line-collection convention anyplotlib's
    ``add_lines`` takes — and *keep* says which input lines crossed the box.

    Done by intersecting with all four edges and taking the two intersections
    that lie inside, rather than by a Liang-Barsky parametric walk: a band line
    can be exactly axis-parallel (a zone axis on the pattern centre makes this
    common, not a corner case), and the edge-intersection form degrades to
    "no intersection with the parallel edges" instead of dividing by zero.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)
    n = len(a)
    eps = 1e-12

    pts = np.full((n, 4, 2), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        # u = 0 and u = width  ->  v = -(c + a*u) / b
        for j, u in enumerate((0.0, width)):
            v = np.where(np.abs(b) > eps, -(c + a * u) / np.where(np.abs(b) > eps, b, 1.0), np.nan)
            ok = np.isfinite(v) & (v >= -1e-9) & (v <= height + 1e-9)
            pts[ok, j, 0] = u
            pts[ok, j, 1] = v[ok]
        # v = 0 and v = height  ->  u = -(c + b*v) / a
        for j, v in enumerate((0.0, height), start=2):
            u = np.where(np.abs(a) > eps, -(c + b * v) / np.where(np.abs(a) > eps, a, 1.0), np.nan)
            ok = np.isfinite(u) & (u >= -1e-9) & (u <= width + 1e-9)
            pts[ok, j, 0] = u[ok]
            pts[ok, j, 1] = v

    segs = np.zeros((n, 2, 2))
    keep = np.zeros(n, bool)
    for i in range(n):
        p = pts[i][np.isfinite(pts[i]).all(1)]
        if len(p) < 2:
            continue
        # A line through a corner hits two edges at the same point; take the
        # two furthest-apart intersections so the segment spans the box.
        d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
        i0, i1 = np.unravel_index(int(np.argmax(d2)), d2.shape)
        if d2[i0, i1] <= 1e-9:
            continue
        segs[i] = (p[i0], p[i1])
        keep[i] = True
    return segs[keep], keep


def band_lines(euler, reflectors: Reflectors | None = None,
               detector=(60, 60), pc=(0.5, 0.5, 0.55), *,
               max_bands: int | None = None):
    """Band CENTRE lines for one orientation -> ``(M, 2, 2)`` detector pixels.

    Returns ``(segments, weights)`` where each segment is
    ``[[x0, y0], [x1, y1]]`` — the line-collection shape anyplotlib's
    ``add_lines`` takes — in the image-pixel convention its ``transform="data"``
    markers use (pixel centres on integers, so the frame spans
    ``-0.5 … size-0.5``). *weights* is the matching band intensity, so a caller
    can style by band strength.

    Only bands that actually cross the detector are returned, so ``M <= N``.
    """
    refl = reflectors if reflectors is not None else cubic_reflectors()
    if max_bands is not None:
        refl = refl.brightest(max_bands)
    euler = np.asarray(euler, float).reshape(3)
    rot = euler_to_matrix(*euler)                            # (3, 3)
    n_rot = normals_to_sample(refl.normals, rot)             # (N, 3) sample frame

    dy, dx = int(detector[0]), int(detector[1])
    pcx, pcy, L = float(pc[0]), float(pc[1]), float(pc[2])
    nx, ny, nz = n_rot[:, 0], n_rot[:, 1], n_rot[:, 2]

    # r·n = 0 with r = ((u/dx - pcx), (pcy - v/dy), L) and u, v the continuous
    # pixel coordinates. Linear in (u, v) — the band centre IS a straight line.
    a = nx / dx
    b = -ny / dy
    c = L * nz - pcx * nx + pcy * ny

    segs, keep = _clip_line_to_box(a, b, c, float(dx), float(dy))
    # u, v measure from the frame edge; anyplotlib pixel coordinates put pixel
    # 0's CENTRE at 0, so the whole frame shifts by half a pixel.
    return (segs - 0.5).astype(np.float32), refl.weights[keep].astype(np.float32)


def zone_axis_points(euler, reflectors: Reflectors | None = None,
                     detector=(60, 60), pc=(0.5, 0.5, 0.55), *,
                     max_axes: int = 12, min_bands: int = 3):
    """Where band lines meet — the zone axes -> ``(M, 2)`` detector pixels.

    A zone axis is a crystal direction several planes share, and on a real
    pattern it is the bright junction the eye locks onto first, which makes it
    the most useful thing to draw after the bands themselves. Found from the
    band set rather than from a separate direction list: any pair of normals
    defines their zone axis ``n_i x n_j``, and the axes worth drawing are the
    ones many pairs agree on.
    """
    refl = reflectors if reflectors is not None else cubic_reflectors()
    euler = np.asarray(euler, float).reshape(3)
    rot = euler_to_matrix(*euler)
    n_rot = normals_to_sample(refl.normals, rot)
    n = len(n_rot)
    if n < 2:
        return np.zeros((0, 2), np.float32)

    i, j = np.triu_indices(n, k=1)
    axes = np.cross(n_rot[i], n_rot[j])
    norm = np.linalg.norm(axes, axis=1)
    axes = axes[norm > 1e-6] / norm[norm > 1e-6, None]
    if not len(axes):
        return np.zeros((0, 2), np.float32)
    axes *= np.sign(axes[:, 2:3] + 1e-30)          # keep the +z hemisphere

    # Cluster the pairwise axes: a direction that many band pairs share is a
    # real zone axis, one that a single pair gives is just two lines crossing.
    order = np.lexsort((axes[:, 2], axes[:, 1], axes[:, 0]))
    axes = axes[order]
    uniq: list[np.ndarray] = []
    counts: list[int] = []
    for v in axes:
        for k, u in enumerate(uniq):
            if float(v @ u) > 0.9995:
                counts[k] += 1
                break
        else:
            uniq.append(v)
            counts.append(1)
    uniq_arr = np.asarray(uniq)
    counts_arr = np.asarray(counts)
    # k planes through one axis give k*(k-1)/2 pairs; min_bands=3 -> 3 pairs.
    want = max(1, min_bands * (min_bands - 1) // 2)
    sel = counts_arr >= want
    if not sel.any():
        return np.zeros((0, 2), np.float32)
    uniq_arr, counts_arr = uniq_arr[sel], counts_arr[sel]
    keep = np.argsort(counts_arr)[::-1][:int(max_axes)]
    uniq_arr = uniq_arr[keep]

    dy, dx = int(detector[0]), int(detector[1])
    pcx, pcy, L = float(pc[0]), float(pc[1]), float(pc[2])
    tz = uniq_arr[:, 2]
    front = tz > 1e-6                              # behind the detector = invisible
    uniq_arr, tz = uniq_arr[front], tz[front]
    u = dx * (pcx + uniq_arr[:, 0] * L / tz) - 0.5
    v = dy * (pcy - uniq_arr[:, 1] * L / tz) - 0.5
    inside = (u >= -0.5) & (u <= dx - 0.5) & (v >= -0.5) & (v <= dy - 0.5)
    return np.column_stack([u[inside], v[inside]]).astype(np.float32)
