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


def group_rings(g_vectors, *, rel_tol: float = 0.06):
    """Cluster a vector set into reflection rings by |g|. Returns
    ``(ring_g ascending, ring_index per vector)`` — for peak/ring selection
    (use all rings, or toggle a ring out of the fit)."""
    g = np.asarray(g_vectors, dtype=float).reshape(-1, 2)
    mag = np.linalg.norm(g, axis=1)
    rings: list[float] = []
    idx = np.full(len(g), -1, dtype=int)
    for i in np.argsort(mag):
        m = float(mag[i])
        if m <= 0:
            continue
        hit = next((r for r, rg in enumerate(rings) if abs(m - rg) / rg <= rel_tol), None)
        if hit is None:
            rings.append(m)
            hit = len(rings) - 1
        idx[i] = hit
    return np.asarray(rings, dtype=float), idx


def _strain_from_T(T: np.ndarray):
    """Vectorized strain decomposition of a stack of 2×2 deformations.

    ``T`` is ``(P, 2, 2)`` (g_meas ≈ T·g_ref). Returns ``(exx, eyy, exy, omega)``
    each ``(P,)``, with NaN where T is singular. This is the batched, closed-form
    equivalent of the per-pixel ``inv/eigh/polar`` block in :func:`fit_pattern_strain`
    — a real 2×2 symmetric eigendecomposition has a closed form, so no Python loop
    or per-matrix ``eigh`` is needed.
    """
    a, b = T[:, 0, 0], T[:, 0, 1]
    c, d = T[:, 1, 0], T[:, 1, 1]
    det = a * d - b * c
    bad = ~np.isfinite(det) | (np.abs(det) < 1e-12)
    det_safe = np.where(bad, 1.0, det)
    # F = inv(T).T  → real-space deformation gradient (see fit_pattern_strain).
    inv = np.empty_like(T)
    inv[:, 0, 0], inv[:, 0, 1] = d / det_safe, -b / det_safe
    inv[:, 1, 0], inv[:, 1, 1] = -c / det_safe, a / det_safe
    F = np.transpose(inv, (0, 2, 1))

    # M = FᵀF (symmetric, SPD). Closed-form right stretch U = M^½ and strain U − I.
    f00, f01 = F[:, 0, 0], F[:, 0, 1]
    f10, f11 = F[:, 1, 0], F[:, 1, 1]
    m00 = f00 * f00 + f10 * f10
    m01 = f00 * f01 + f10 * f11
    m11 = f01 * f01 + f11 * f11
    tr, dt = m00 + m11, m00 * m11 - m01 * m01
    disc = np.sqrt(np.clip(tr * tr - 4.0 * dt, 0.0, None))
    l1 = 0.5 * (tr + disc)            # eigenvalues of M (= squared stretches)
    l2 = 0.5 * (tr - disc)
    s1, s2 = np.sqrt(np.clip(l1, 0.0, None)), np.sqrt(np.clip(l2, 0.0, None))
    # U = s1·P1 + s2·P2 where P1,P2 are the eigen-projectors of M. For a 2×2
    # symmetric M this collapses to U = (M + sqrt(dt)·I) / (s1 + s2)   [since
    # M^½ has det = sqrt(dt) and trace = s1+s2], avoiding eigenvector assembly.
    sdt = np.sqrt(np.clip(dt, 0.0, None))
    denom = s1 + s2
    denom_safe = np.where(denom < 1e-12, 1.0, denom)
    u00 = (m00 + sdt) / denom_safe
    u01 = m01 / denom_safe
    u11 = (m11 + sdt) / denom_safe
    exx = u00 - 1.0
    eyy = u11 - 1.0
    exy = u01
    # R = F·U⁻¹ ; ω = atan2(R10, R00). U⁻¹ = [[u11,-u01],[-u01,u00]]/det(U).
    udet = u00 * u11 - u01 * u01
    udet_safe = np.where(np.abs(udet) < 1e-12, 1.0, udet)
    r00 = (f00 * u11 - f01 * u01) / udet_safe
    r10 = (f10 * u11 - f11 * u01) / udet_safe
    omega = np.arctan2(r10, r00)

    for arr in (exx, eyy, exy, omega):
        arr[bad] = np.nan
    return exx, eyy, exy, omega


def compute_strain_field(vecs, ref_yx=None, *, ref_vectors=None,
                         tol: float | None = None) -> StrainField:
    """Strain field of ``vecs`` (a ``SpyDEDiffractionVectors``) measured against a
    reference lattice — either the vectors at reference pixel ``ref_yx = (ry, rx)``
    (relative strain; ε = 0 there by construction) OR an explicit ``ref_vectors``
    set, e.g. the CIF-snapped absolute reference (absolute strain).

    ``tol`` (reciprocal units) is the ±g match radius; defaults to ¼ of the
    reference lattice's nearest-neighbour spacing.

    Fits EVERY nav pixel in one vectorized pass (one global KDTree match + batched
    per-pixel normal-equation solve + closed-form 2×2 polar decomposition) rather
    than a per-pixel Python ``scipy`` loop — the loop ran ~13k tiny lstsq/eigh
    calls on a real scan and made the strain window take seconds. The 5-D (time)
    case falls back to the per-pixel reference path (rare; kxy_at aggregates time).
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

    # Vectorized path needs the flat CSR layout with one segment per (iy, ix); a
    # 5-D dataset's innermost offsets are per (t, iy, ix), so fall back there.
    flat = getattr(vecs, "flat_buffer", None)
    offs = getattr(vecs, "nav_offsets", None)
    n_time = getattr(vecs, "n_time", 0)
    if (flat is None or offs is None or n_time != 0 or len(g_ref) < 2):
        return _compute_strain_field_loop(vecs, g_ref, tol, ny, nx)

    return _compute_strain_field_vectorized(flat, offs[-1], g_ref, tol, ny, nx)


def _compute_strain_field_loop(vecs, g_ref, tol, ny, nx) -> StrainField:
    """Per-pixel reference path (5-D / non-CSR / degenerate reference)."""
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


def _compute_strain_field_vectorized(flat_buffer, x_off, g_ref, tol, ny, nx) -> StrainField:
    """Whole-field strain in one pass — see :func:`compute_strain_field`.

    Mirrors :func:`fit_pattern_strain` exactly, batched over all P = ny·nx pixels:
    per-pixel mean-centre → one ±g (Friedel) KDTree match for every vector at once
    → per-pixel affine normal equations (AᵀA, AᵀB) via scatter-add → batched 3×3
    solve → closed-form 2×2 polar decomposition.
    """
    from scipy.spatial import cKDTree

    P = ny * nx
    from spyde.signals.diffraction_vectors import COL_KX, COL_KY
    kxy = np.ascontiguousarray(flat_buffer[:, COL_KX:COL_KY + 1], dtype=float)  # (Ntot, 2)
    x_off = np.asarray(x_off)
    counts = np.diff(x_off[:P + 1]).astype(np.int64)                # vectors per pixel
    Ntot = int(x_off[P])
    kxy = kxy[:Ntot]
    # Per-vector owning pixel id (segment id from the CSR row pointers).
    pix = np.repeat(np.arange(P), counts)

    # Per-pixel mean-centre (the −g=g centroid removal in fit_pattern_strain).
    sums = np.zeros((P, 2)); np.add.at(sums, pix, kxy)
    cnt = np.maximum(counts, 1)[:, None]
    centroids = sums / cnt
    kc = kxy - centroids[pix]                                        # centred measured

    g_ref = g_ref - g_ref.mean(axis=0)                              # centred reference
    ref_aug = np.vstack([g_ref, -g_ref])                           # (2M, 2), Friedel
    src_idx = np.concatenate([np.arange(len(g_ref))] * 2)          # back to ref id

    # One KDTree match for EVERY measured vector across the whole scan.
    d, idx = cKDTree(ref_aug).query(kc, distance_upper_bound=tol)
    ok = np.isfinite(d)
    pid = pix[ok]
    Gr = ref_aug[idx[ok]]                                           # (K, 2) matched ref
    Gm = kc[ok]                                                     # (K, 2) measured

    # Per-pixel affine normal equations for [gx,gy,1]·M = g_meas. Accumulate the
    # six unique AᵀA entries + the AᵀB (3×2) via scatter-add keyed by pixel.
    ax, ay = Gr[:, 0], Gr[:, 1]
    AtA = np.zeros((P, 3, 3))
    AtB = np.zeros((P, 3, 2))
    # AᵀA = Σ [ax,ay,1]ᵀ[ax,ay,1]
    np.add.at(AtA[:, 0, 0], pid, ax * ax)
    np.add.at(AtA[:, 0, 1], pid, ax * ay)
    np.add.at(AtA[:, 0, 2], pid, ax)
    np.add.at(AtA[:, 1, 1], pid, ay * ay)
    np.add.at(AtA[:, 1, 2], pid, ay)
    np.add.at(AtA[:, 2, 2], pid, np.ones_like(ax))
    AtA[:, 1, 0] = AtA[:, 0, 1]
    AtA[:, 2, 0] = AtA[:, 0, 2]
    AtA[:, 2, 1] = AtA[:, 1, 2]
    np.add.at(AtB[:, 0, :], pid, ax[:, None] * Gm)
    np.add.at(AtB[:, 1, :], pid, ay[:, None] * Gm)
    np.add.at(AtB[:, 2, :], pid, Gm)

    # Matched ref count + distinct-ref coverage per pixel.
    matched = np.zeros(P, dtype=np.int64); np.add.at(matched, pid, 1)
    # Coverage = #distinct reference reflections matched / len(g_ref).
    cov = np.zeros((ny, nx), dtype=np.float32)
    if len(pid):
        ref_of = src_idx[idx[ok]]
        seen = {}
        # distinct (pixel, ref) pairs → count per pixel (small K; vectorized via unique)
        pairs = pid.astype(np.int64) * len(g_ref) + ref_of
        upairs = np.unique(pairs)
        dpix = (upairs // len(g_ref)).astype(np.int64)
        dcov = np.zeros(P, dtype=np.float32); np.add.at(dcov, dpix, 1.0)
        cov = (dcov / len(g_ref)).reshape(ny, nx)

    # A pixel is well-posed only with ≥2 matches AND a non-collinear, invertible
    # normal matrix. Solve all good pixels' 3×3 systems at once.
    exx = np.full(P, np.nan); eyy = np.full(P, np.nan)
    exy = np.full(P, np.nan); omega = np.full(P, np.nan)
    detAtA = np.linalg.det(AtA)
    good = (matched >= 2) & np.isfinite(detAtA) & (np.abs(detAtA) > 1e-12)
    if good.any():
        sol = np.linalg.solve(AtA[good], AtB[good])                # (G, 3, 2): rows = M
        T = np.transpose(sol[:, :2, :], (0, 2, 1))                 # (G, 2, 2): g≈T·g_ref
        gx, gy, gz, gw = _strain_from_T(T)
        exx[good], eyy[good], exy[good], omega[good] = gx, gy, gz, gw

    return StrainField(
        exx.reshape(ny, nx).astype(np.float32),
        eyy.reshape(ny, nx).astype(np.float32),
        exy.reshape(ny, nx).astype(np.float32),
        omega.reshape(ny, nx).astype(np.float32),
        cov.astype(np.float32),
    )


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
