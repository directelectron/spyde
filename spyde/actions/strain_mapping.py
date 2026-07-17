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
from typing import Optional

import numpy as np


@dataclass
class StrainField:
    """Per-pixel strain maps over the navigation grid ``(ny, nx)``."""
    exx: np.ndarray
    eyy: np.ndarray
    exy: np.ndarray
    omega: np.ndarray          # lattice rotation (radians)
    coverage: np.ndarray       # fraction of reference reflections matched (0–1)
    # Per-pixel fit quality (drives the confidence-weighted display):
    residual: Optional[np.ndarray] = None    # RMS match residual of the final fit
    n_matched: Optional[np.ndarray] = None   # matches used in the final fit
    # Provenance record ({"action", "params", "spyde_version"}) — same dict
    # convention as commit._stamp_provenance (script/app interchangeable).
    provenance: Optional[dict] = None

    @property
    def nav_shape(self) -> tuple:
        return self.exx.shape


def default_reference(vecs) -> tuple:
    """A sensible unstrained reference: the pixel with the most vectors (the
    best-determined local lattice)."""
    try:
        cm = np.asarray(vecs.count_map())
        iy, ix = np.unravel_index(int(np.argmax(cm)), cm.shape)
        return int(iy), int(ix)
    except Exception:
        return 0, 0


def zero_beam_filtered(g_ref) -> np.ndarray:
    """Reference spots with the central/direct (zero) beam removed.

    The zero beam (|g|≈0) carries no lattice information and would pin the fit's
    translation/centroid, so it's excluded from every strain reference. Threshold
    = 25% of the median nonzero |g| (well below the first ring, above numerical
    noise at the centre)."""
    g = np.asarray(g_ref, dtype=float).reshape(-1, 2)
    if len(g) == 0:
        return g
    mag = np.linalg.norm(g, axis=1)
    nz = mag[mag > 0]
    thresh = 0.25 * float(np.median(nz)) if nz.size else 0.0
    return g[mag > thresh]


def region_reference(vecs, ref_yx, radius: int = 2, min_frac: float = 0.5,
                     tol: float | None = None) -> np.ndarray:
    """Consensus reference built from ALL peaks in the ``(2·radius+1)²`` scan
    neighbourhood around ``ref_yx`` — the noise-robust replacement for trusting a
    single frame's peak set.

    Vectors from every frame in the window are pooled and clustered by proximity
    (greedy, densest-first, radius = the same ¼-NN-spacing scale the strain match
    uses). A cluster is kept only when it appears in ≥ ``min_frac`` of the frames
    — so a false positive detected in one or two frames is DROPPED — and its
    position is the MEDIAN over members, so per-frame subpixel noise averages
    down ~√N. ``radius=0`` returns the single-pixel set (old behaviour).

    Returns (K, 2) vectors (zero-beam NOT yet filtered — callers filter)."""
    ry, rx = int(ref_yx[0]), int(ref_yx[1])
    centre = np.asarray(vecs.kxy_at(ry, rx), dtype=float).reshape(-1, 2)
    if radius <= 0:
        return centre
    ny, nx = vecs.nav_shape
    frames = []
    for iy in range(max(0, ry - radius), min(ny, ry + radius + 1)):
        for ix in range(max(0, rx - radius), min(nx, rx + radius + 1)):
            g = np.asarray(vecs.kxy_at(iy, ix), dtype=float).reshape(-1, 2)
            if len(g):
                frames.append(g)
    if len(frames) <= 1:
        return centre
    if tol is None:
        nn = _median_nn(centre if len(centre) >= 2 else np.vstack(frames))
        tol = 0.25 * nn if nn > 0 else 0.0
    if tol <= 0:
        return centre
    from scipy.spatial import cKDTree
    pool = np.vstack(frames)
    fid = np.repeat(np.arange(len(frames)), [len(f) for f in frames])
    neigh = cKDTree(pool).query_ball_point(pool, r=tol)
    order = np.argsort([-len(n) for n in neigh])          # densest clusters first
    used = np.zeros(len(pool), bool)
    need = max(2, int(np.ceil(min_frac * len(frames))))
    out = []
    for i in order:
        if used[i]:
            continue
        members = [j for j in neigh[i] if not used[j]]
        used[members] = True
        if len({int(fid[j]) for j in members}) >= need:   # persists across frames
            out.append(np.median(pool[members], axis=0))
    if len(out) < 2:
        return centre                                     # degenerate → old behaviour
    return np.asarray(out, dtype=float)


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


# A pixel's affine fit (2×2 T + translation = 6 unknowns) is EXACTLY determined by
# 3 matches and UNDER-determined by 2 (lstsq then returns an arbitrary min-norm T →
# a wild strain value: the "bright dot" pixels). 3 is the mathematical minimum; the
# residual trim below is what protects a 3-match pixel from a single FP (dropping
# the FP leaves 2 → below the gate → honestly masked instead of garbage).
DEFAULT_MIN_MATCHES = 3
# Residual trimming: after the first fit, matches with residual > 2.5× the pixel's
# RMS (with a small absolute floor so exact synthetic fits keep everything, and an
# absolute cap of tol/2) are dropped and the pixel refit once — one FP vector inside
# the match radius no longer drags T.
_TRIM_SIGMA = 2.5
_TRIM_FLOOR = 1e-6


def _trim_keep(r: np.ndarray, rms: float, tol: float) -> np.ndarray:
    keep = r <= max(_TRIM_SIGMA * rms, _TRIM_FLOOR)
    if np.isfinite(tol):
        keep &= r <= 0.5 * tol
    return keep


def _fit_pattern_strain_full(g_meas, g_ref, *, tol, min_matches=DEFAULT_MIN_MATCHES,
                             trim=True):
    """fit_pattern_strain + fit-quality extras:
    ``(exx, eyy, exy, omega, coverage, residual_rms, n_matched)`` or None."""
    g_meas = np.asarray(g_meas, dtype=float).reshape(-1, 2)
    g_ref = np.asarray(g_ref, dtype=float).reshape(-1, 2)
    if len(g_meas) < 2 or len(g_ref) < 2:
        return None

    from scipy.spatial import cKDTree
    # TWO candidate alignments; the one matching MORE spots wins (per pixel):
    #   raw      — trust the pipeline's pattern centre (kxy are centre-relative).
    #              Essential for ASYMMETRIC detections (missing Friedel partners,
    #              e.g. faint precipitate patterns): their centroid is NOT the
    #              centre — the bias reaches ~|g|/3 > tol, so the old
    #              centroid-only alignment matched NOTHING and a frame with six
    #              good spots read as a failed fit.
    #   centroid — remove each set's mean (a CENTROSYMMETRIC set's centroid IS
    #              the centre offset): robust to a badly-centred pattern.
    # The affine translation term mops up the residual either way; strict '>'
    # keeps the centroid behaviour on ties.
    src_idx = np.concatenate([np.arange(len(g_ref))] * 2)      # aug → ref id
    cands = []
    for centre in (False, True):
        gm = g_meas - (g_meas.mean(axis=0) if centre else 0.0)
        gr = g_ref - (g_ref.mean(axis=0) if centre else 0.0)
        ref_aug = np.vstack([gr, -gr])                         # (2M, 2) Friedel
        d, idx = cKDTree(ref_aug).query(gm, distance_upper_bound=tol)
        ok = np.isfinite(d)
        cands.append((int(ok.sum()), gm, ref_aug, idx, ok))
    n_raw, n_cen = cands[0][0], cands[1][0]
    _, gm, ref_aug, idx, ok = cands[0] if n_raw > n_cen else cands[1]
    G_ref = ref_aug[idx[ok]]                                   # (K, 2)
    G_meas = gm[ok]                                            # (K, 2)
    ref_ids = src_idx[idx[ok]]

    sol = None
    resid = None
    for it in range(2 if trim else 1):
        if len(G_ref) < max(2, min_matches):
            return None
        if np.linalg.matrix_rank(G_ref - G_ref.mean(0)) < 2:   # collinear → ill-posed
            return None
        # Affine least squares: [G_ref | 1] @ M = G_meas, M = [[T^T],[t^T]] (3×2).
        A = np.hstack([G_ref, np.ones((len(G_ref), 1))])
        sol, *_ = np.linalg.lstsq(A, G_meas, rcond=None)
        resid = np.linalg.norm(G_meas - A @ sol, axis=1)
        if it == 0 and trim:
            rms = float(np.sqrt(np.mean(resid ** 2)))
            keep = _trim_keep(resid, rms, tol)
            if keep.all():
                break
            G_ref, G_meas, ref_ids = G_ref[keep], G_meas[keep], ref_ids[keep]
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
    coverage = len(np.unique(ref_ids)) / len(g_ref)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return (float(e[0, 0]), float(e[1, 1]), float(e[0, 1]), omega, float(coverage),
            rms, int(len(G_ref)))


def fit_pattern_strain(g_meas: np.ndarray, g_ref: np.ndarray, *, tol: float,
                       min_matches: int = DEFAULT_MIN_MATCHES, trim: bool = True):
    """Fit one pixel: ``g_meas ≈ T · g_ref + t`` with ±g (Friedel) matching.

    Returns ``(exx, eyy, exy, omega, coverage)`` or ``None`` when fewer than
    ``min_matches`` reflections match (an under-determined fit — the 6-unknown
    affine needs redundancy, not just solvability). One residual-trimming pass
    (``trim``) drops outlier matches (FP vectors inside the match radius) and
    refits. ``coverage`` is the fraction of the reference reflections that found
    a measured partner.
    """
    r = _fit_pattern_strain_full(g_meas, g_ref, tol=tol, min_matches=min_matches,
                                 trim=trim)
    return None if r is None else r[:5]


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
                         tol: float | None = None,
                         min_matches: int = DEFAULT_MIN_MATCHES, trim: bool = True,
                         ref_radius: int = 0) -> StrainField:
    """Strain field of ``vecs`` (a ``SpyDEDiffractionVectors``) measured against a
    reference lattice — either the vectors at reference pixel ``ref_yx = (ry, rx)``
    (relative strain; ε = 0 there by construction) OR an explicit ``ref_vectors``
    set, e.g. the CIF-snapped absolute reference (absolute strain).

    ``tol`` (reciprocal units) is the ±g match radius; defaults to ¼ of the
    reference lattice's nearest-neighbour spacing. ``min_matches`` / ``trim`` are
    the fit-robustness knobs (see :func:`fit_pattern_strain`). ``ref_radius`` > 0
    pools the reference over a ``(2r+1)²`` neighbourhood of ``ref_yx``
    (:func:`region_reference` — noise-robust consensus; 0 = the single pixel).

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
        g_ref = region_reference(vecs, ref_yx, radius=int(ref_radius))
    if tol is None:
        nn = _median_nn(g_ref)
        tol = 0.25 * nn if nn > 0 else np.inf

    # Vectorized path needs the flat CSR layout with one segment per (iy, ix); a
    # 5-D dataset's innermost offsets are per (t, iy, ix), so fall back there.
    flat = getattr(vecs, "flat_buffer", None)
    offs = getattr(vecs, "nav_offsets", None)
    n_time = getattr(vecs, "n_time", 0)
    if (flat is None or offs is None or n_time != 0 or len(g_ref) < 2):
        return _compute_strain_field_loop(vecs, g_ref, tol, ny, nx,
                                          min_matches=min_matches, trim=trim)

    return _compute_strain_field_vectorized(flat, offs[-1], g_ref, tol, ny, nx,
                                            min_matches=min_matches, trim=trim)


def _compute_strain_field_loop(vecs, g_ref, tol, ny, nx,
                               min_matches=DEFAULT_MIN_MATCHES, trim=True) -> StrainField:
    """Per-pixel reference path (5-D / non-CSR / degenerate reference)."""
    exx = np.full((ny, nx), np.nan, dtype=np.float32)
    eyy = np.full((ny, nx), np.nan, dtype=np.float32)
    exy = np.full((ny, nx), np.nan, dtype=np.float32)
    omega = np.full((ny, nx), np.nan, dtype=np.float32)
    cov = np.zeros((ny, nx), dtype=np.float32)
    res = np.full((ny, nx), np.nan, dtype=np.float32)
    nm = np.zeros((ny, nx), dtype=np.int32)
    for iy in range(ny):
        for ix in range(nx):
            r = _fit_pattern_strain_full(vecs.kxy_at(iy, ix), g_ref, tol=tol,
                                         min_matches=min_matches, trim=trim)
            if r is not None:
                (exx[iy, ix], eyy[iy, ix], exy[iy, ix], omega[iy, ix],
                 cov[iy, ix], res[iy, ix], nm[iy, ix]) = r
    return StrainField(exx, eyy, exy, omega, cov, residual=res, n_matched=nm)


def _compute_strain_field_vectorized(flat_buffer, x_off, g_ref, tol, ny, nx,
                                     min_matches=DEFAULT_MIN_MATCHES,
                                     trim=True) -> StrainField:
    """Whole-field strain in one pass — see :func:`compute_strain_field`.

    Mirrors :func:`fit_pattern_strain` exactly (incl. the min-match gate and the
    one residual-trimming pass), batched over all P = ny·nx pixels: per-pixel
    mean-centre → one ±g (Friedel) KDTree match for every vector at once →
    per-pixel affine normal equations (AᵀA, AᵀB) via scatter-add → batched 3×3
    solve → residual trim + refit → closed-form 2×2 polar decomposition.
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

    src_idx = np.concatenate([np.arange(len(g_ref))] * 2)          # aug → ref id
    # TWO candidate alignments (see _fit_pattern_strain_full): raw (asymmetric
    # detections — the centroid is biased) vs centroid-removed (badly-centred
    # patterns). One KDTree match each for EVERY vector across the whole scan;
    # each pixel keeps whichever alignment matched MORE of its spots (strict '>'
    # keeps the centroid behaviour on ties — identical rule to the loop path).
    ref_aug_raw = np.vstack([g_ref, -g_ref])
    g_ref_cen = g_ref - g_ref.mean(axis=0)
    ref_aug_cen = np.vstack([g_ref_cen, -g_ref_cen])
    d0, idx0 = cKDTree(ref_aug_raw).query(kxy, distance_upper_bound=tol)
    d1, idx1 = cKDTree(ref_aug_cen).query(kc, distance_upper_bound=tol)
    ok0, ok1 = np.isfinite(d0), np.isfinite(d1)
    n0 = np.zeros(P, np.int64); np.add.at(n0, pix[ok0], 1)
    n1 = np.zeros(P, np.int64); np.add.at(n1, pix[ok1], 1)
    use_raw = n0 > n1                                               # per pixel
    take0 = ok0 & use_raw[pix]
    take1 = ok1 & ~use_raw[pix]
    pid = np.concatenate([pix[take0], pix[take1]])
    Gr = np.vstack([ref_aug_raw[idx0[take0]], ref_aug_cen[idx1[take1]]])
    Gm = np.vstack([kxy[take0], kc[take1]])
    rid = np.concatenate([src_idx[idx0[take0]], src_idx[idx1[take1]]])

    min_m = max(2, int(min_matches))

    def _normal_eqs(pid, Gr, Gm):
        """Per-pixel affine normal equations for [gx,gy,1]·M = g_meas: the six
        unique AᵀA entries + AᵀB (3×2) via scatter-add keyed by pixel."""
        ax, ay = Gr[:, 0], Gr[:, 1]
        AtA = np.zeros((P, 3, 3))
        AtB = np.zeros((P, 3, 2))
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
        matched = np.zeros(P, dtype=np.int64); np.add.at(matched, pid, 1)
        detAtA = np.linalg.det(AtA)
        good = (matched >= min_m) & np.isfinite(detAtA) & (np.abs(detAtA) > 1e-12)
        sol = np.zeros((P, 3, 2))
        if good.any():
            sol[good] = np.linalg.solve(AtA[good], AtB[good])       # rows of M
        return sol, good, matched

    def _residuals(sol, pid, Gr, Gm):
        A = np.hstack([Gr, np.ones((len(Gr), 1))])                  # (K, 3)
        model = np.einsum('kj,kjl->kl', A, sol[pid])                # (K, 2)
        return np.linalg.norm(Gm - model, axis=1)

    sol, good, matched = _normal_eqs(pid, Gr, Gm)
    if trim and len(pid):
        # One trimming pass (same thresholds as _fit_pattern_strain_full): drop
        # matches whose residual exceeds 2.5× their pixel's RMS (abs floor) or
        # tol/2, then refit. Matches on not-yet-good pixels are left alone —
        # their pixels stay NaN either way (mirrors the loop returning None).
        r = _residuals(sol, pid, Gr, Gm)
        ss = np.zeros(P); nn = np.zeros(P)
        np.add.at(ss, pid, r * r); np.add.at(nn, pid, 1.0)
        rms = np.sqrt(ss / np.maximum(nn, 1.0))
        keep = r <= np.maximum(_TRIM_SIGMA * rms[pid], _TRIM_FLOOR)
        if np.isfinite(tol):
            keep &= r <= 0.5 * tol
        keep |= ~good[pid]
        if not keep.all():
            pid, Gr, Gm, rid = pid[keep], Gr[keep], Gm[keep], rid[keep]
            sol, good, matched = _normal_eqs(pid, Gr, Gm)

    # Coverage = #distinct reference reflections matched / len(g_ref), from the
    # FINAL (post-trim) match set.
    cov = np.zeros((ny, nx), dtype=np.float32)
    if len(pid):
        pairs = pid.astype(np.int64) * len(g_ref) + rid
        dpix = (np.unique(pairs) // len(g_ref)).astype(np.int64)
        dcov = np.zeros(P, dtype=np.float32); np.add.at(dcov, dpix, 1.0)
        cov = (dcov / len(g_ref)).reshape(ny, nx)

    # Per-pixel RMS residual of the FINAL fit (the confidence/error map).
    res = np.full(P, np.nan)
    if len(pid):
        r = _residuals(sol, pid, Gr, Gm)
        ss = np.zeros(P); nn = np.zeros(P)
        np.add.at(ss, pid, r * r); np.add.at(nn, pid, 1.0)
        with np.errstate(invalid="ignore"):
            res = np.where(nn > 0, np.sqrt(ss / np.maximum(nn, 1.0)), np.nan)

    exx = np.full(P, np.nan); eyy = np.full(P, np.nan)
    exy = np.full(P, np.nan); omega = np.full(P, np.nan)
    if good.any():
        T = np.transpose(sol[good][:, :2, :], (0, 2, 1))            # (G, 2, 2)
        gx, gy, gz, gw = _strain_from_T(T)
        exx[good], eyy[good], exy[good], omega[good] = gx, gy, gz, gw
    res[~good] = np.nan

    return StrainField(
        exx.reshape(ny, nx).astype(np.float32),
        eyy.reshape(ny, nx).astype(np.float32),
        exy.reshape(ny, nx).astype(np.float32),
        omega.reshape(ny, nx).astype(np.float32),
        cov.astype(np.float32),
        residual=res.reshape(ny, nx).astype(np.float32),
        n_matched=matched.reshape(ny, nx).astype(np.int32),
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
