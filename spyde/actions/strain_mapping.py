"""
strain_mapping.py — strain mapping from a set of diffraction vectors.

Given a ``SpyDEDiffractionVectors`` (the per-pixel found g-vectors) and a
**reference** (unstrained) lattice, fit a 2-D deformation gradient ``T`` per
pixel from ``g_measured ≈ T · g_reference`` and decompose it into the strain
components εxx / εyy / εxy and the lattice rotation ω. The principal strains
(ε1, ε2, θ) drive the strain-ellipse glyph overlay.

Two things make this robust (per the design discussion):

* **−g ≡ g (Friedel).** Each reference reflection is matched against the measured
  peak at *either* ``+g`` or ``−g``, and the fit carries a translation term — so
  a diffraction pattern whose centre is slightly off does not leak into the
  strain (the offset is absorbed by the translation, and the ±g matching keeps
  the constraints centrosymmetric).
* **Multi-ring.** The reference naturally spans every reflection ring present at
  the reference pixel, so the least-squares fit is over-constrained by all of
  them, not a single ring.

The measured vectors come from Find Vectors, which already refines peak
positions to sub-pixel — so the fit operates on sub-pixel inputs.

No Qt. Host-agnostic (Electron + Jupyter).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StrainField:
    """Per-pixel strain maps over the navigation grid ``(ny, nx)``."""
    exx: np.ndarray
    eyy: np.ndarray
    exy: np.ndarray
    omega: np.ndarray          # lattice rotation (radians)
    coverage: np.ndarray       # fraction of reference reflections matched (0–1)

    @property
    def nav_shape(self) -> tuple:
        return self.exx.shape


def _median_nn(g: np.ndarray) -> float:
    """Median nearest-neighbour distance of a point set — a natural length scale
    for the match tolerance. Returns 0.0 for < 2 points."""
    if g is None or len(g) < 2:
        return 0.0
    from scipy.spatial import cKDTree
    d, _ = cKDTree(g).query(g, k=2)
    nn = d[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    return float(np.median(nn)) if nn.size else 0.0


def fit_pattern_strain(g_meas: np.ndarray, g_ref: np.ndarray, *, tol: float):
    """Fit one pixel: ``g_meas ≈ T · g_ref + t`` with ±g (Friedel) matching.

    Returns ``(exx, eyy, exy, omega, coverage)`` or ``None`` when fewer than two
    reflections match (an undetermined fit). ``coverage`` is the fraction of the
    reference reflections that found a measured partner.
    """
    g_meas = np.asarray(g_meas, dtype=float).reshape(-1, 2)
    g_ref = np.asarray(g_ref, dtype=float).reshape(-1, 2)
    if len(g_meas) < 2 or len(g_ref) < 2:
        return None

    from scipy.spatial import cKDTree
    # Centre both sets: a centrosymmetric ({g, −g}) peak set's centroid IS the
    # diffraction-pattern-centre offset, so removing it makes the ±g match robust
    # to a badly-centred pattern (the −g=g point). The affine translation below
    # then mops up any residual from missing reflections.
    g_meas = g_meas - g_meas.mean(axis=0)
    g_ref = g_ref - g_ref.mean(axis=0)

    # Friedel: a reference reflection may show up at +g OR −g.
    ref_aug = np.vstack([g_ref, -g_ref])                       # (2M, 2)
    src_idx = np.concatenate([np.arange(len(g_ref))] * 2)      # back to ref id
    # Match each MEASURED peak to its nearest augmented reference within tol.
    d, idx = cKDTree(ref_aug).query(g_meas, distance_upper_bound=tol)
    ok = np.isfinite(d)
    if ok.sum() < 2:
        return None
    G_ref = ref_aug[idx[ok]]                                   # (K, 2)
    G_meas = g_meas[ok]                                        # (K, 2)
    if np.linalg.matrix_rank(G_ref - G_ref.mean(0)) < 2:       # collinear → ill-posed
        return None

    # Affine least squares: [G_ref | 1] @ M = G_meas, M = [[T^T],[t^T]] (3×2).
    A = np.hstack([G_ref, np.ones((len(G_ref), 1))])
    sol, *_ = np.linalg.lstsq(A, G_meas, rcond=None)
    T = sol[:2, :].T                                           # 2×2: g_meas ≈ T·g_ref + t

    # Report REAL-SPACE lattice strain, not reciprocal: the measured g map as
    # g = F⁻ᵀ·g_ref, so the real-space deformation gradient is F = T⁻ᵀ. This makes
    # a stretched lattice POSITIVE strain (its diffraction vectors are smaller) —
    # the physical convention.
    try:
        F = np.linalg.inv(T).T
    except np.linalg.LinAlgError:
        return None
    # Polar decomposition F = R · U (U = (FᵀF)^½, the rotation-free right stretch).
    # Strain = U − I; the rotation lives in R — so a finite lattice rotation is
    # NOT mistaken for strain (matches the vector-OM convention).
    w, V = np.linalg.eigh(F.T @ F)
    U = (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T
    e = U - np.eye(2)
    try:
        R = F @ np.linalg.inv(U)
        omega = float(np.arctan2(R[1, 0], R[0, 0]))
    except np.linalg.LinAlgError:
        omega = 0.0
    coverage = len(np.unique(src_idx[idx[ok]])) / len(g_ref)
    return float(e[0, 0]), float(e[1, 1]), float(e[0, 1]), omega, float(coverage)


def cif_g_families(phase, *, min_dspacing: float = 0.7) -> np.ndarray:
    """Allowed reflection |g| families (1/Å, ascending) of an orix ``Phase`` —
    structure-factor-filtered so fcc/bcc-forbidden rings are excluded."""
    from diffsims.crystallography import ReciprocalLatticeVector
    rlv = ReciprocalLatticeVector.from_min_dspacing(phase, min_dspacing=min_dspacing)
    rlv.sanitise_phase()
    rlv.calculate_structure_factor()
    F = np.abs(np.asarray(rlv.structure_factor))
    g = np.asarray(rlv.gspacing)
    allowed = (g > 0) & (F > 1e-3 * (F.max() or 1.0))
    return np.unique(np.round(g[allowed], 4))


def snap_reference_to_cif(sample_g, families, *, tol_frac: float = 0.2) -> np.ndarray:
    """Build an ABSOLUTE reference lattice from a measured vector set: keep each
    vector's direction but snap its magnitude to the nearest CIF |g| family
    (within ``tol_frac``). Strain is then measured against the ideal spacing —
    so a flat region is not needed (the −g=g family identity from the DP scale)."""
    sample_g = np.asarray(sample_g, dtype=float).reshape(-1, 2)
    families = np.asarray(families, dtype=float)
    mag = np.linalg.norm(sample_g, axis=1)
    out = []
    for v, m in zip(sample_g, mag):
        if m <= 0 or families.size == 0:
            continue
        j = int(np.argmin(np.abs(families - m)))
        if abs(families[j] - m) / m <= tol_frac:
            out.append(v / m * families[j])
    return np.asarray(out, dtype=float).reshape(-1, 2)


def compute_strain_field(vecs, ref_yx=None, *, ref_vectors=None,
                         tol: float | None = None) -> StrainField:
    """Strain field of ``vecs`` (a ``SpyDEDiffractionVectors``) measured against a
    reference lattice — either the vectors at reference pixel ``ref_yx = (ry, rx)``
    (relative strain; ε = 0 there by construction) OR an explicit ``ref_vectors``
    set, e.g. the CIF-snapped absolute reference (absolute strain).

    ``tol`` (reciprocal units) is the ±g match radius; defaults to ¼ of the
    reference lattice's nearest-neighbour spacing.
    """
    ny, nx = vecs.nav_shape
    if ref_vectors is not None:
        g_ref = np.asarray(ref_vectors, dtype=float).reshape(-1, 2)
    else:
        ry, rx = int(ref_yx[0]), int(ref_yx[1])
        g_ref = np.asarray(vecs.kxy_at(ry, rx), dtype=float).reshape(-1, 2)
    if tol is None:
        nn = _median_nn(g_ref)
        tol = 0.25 * nn if nn > 0 else np.inf

    exx = np.full((ny, nx), np.nan, dtype=np.float32)
    eyy = np.full((ny, nx), np.nan, dtype=np.float32)
    exy = np.full((ny, nx), np.nan, dtype=np.float32)
    omega = np.full((ny, nx), np.nan, dtype=np.float32)
    cov = np.zeros((ny, nx), dtype=np.float32)

    for iy in range(ny):
        for ix in range(nx):
            r = fit_pattern_strain(vecs.kxy_at(iy, ix), g_ref, tol=tol)
            if r is not None:
                exx[iy, ix], eyy[iy, ix], exy[iy, ix], omega[iy, ix], cov[iy, ix] = r
    return StrainField(exx, eyy, exy, omega, cov)


def principal_strain(exx: np.ndarray, eyy: np.ndarray, exy: np.ndarray):
    """Per-pixel principal strains and direction for the ellipse glyphs.

    Returns ``(e1, e2, theta)`` where ``e1 >= e2`` are the eigenvalues of the
    symmetric strain tensor ``[[exx, exy], [exy, eyy]]`` and ``theta`` (radians)
    is the orientation of the ``e1`` axis.
    """
    exx = np.asarray(exx, float)
    eyy = np.asarray(eyy, float)
    exy = np.asarray(exy, float)
    half_tr = 0.5 * (exx + eyy)
    diff = 0.5 * (exx - eyy)
    rad = np.sqrt(diff ** 2 + exy ** 2)
    e1 = half_tr + rad
    e2 = half_tr - rad
    theta = 0.5 * np.arctan2(2.0 * exy, exx - eyy)
    return e1, e2, theta
