"""
index_and_calibrate.py — phase identification + reciprocal-space calibration +
per-pattern centering, from a found set of diffraction vectors.

The pixel size (Å⁻¹/px) is usually unknown for a 4D-STEM scan, and on a
beam-stopped pattern there is no direct beam to center on.  This module solves
both from the vectors alone:

  1. **Centering** — each pattern's origin is found by Friedel (g↔−g) symmetry
     (``sem_geometry.friedel_center``), which works even when the direct beam is
     occluded by a beam stop.  Produces a per-pattern center map plus a robust
     global center.

  2. **Ring extraction** — the |g| of every centered vector is pooled across
     clean (single-grain) patterns and the characteristic ring radii (in px)
     are found from the radial histogram.

  3. **Phase ID + calibration** — the *ratios* of the ring radii are
     scale-independent (they need no pixel size), so they are matched against
     the allowed-reflection d-spacing ratios of candidate phases.  The
     best-matching phase fixes the absolute scale at the same time:
     ``scale [Å⁻¹/px] = (1/d_hkl) / R_px`` for the matched ring.

The candidate set defaults to the common Ti-Nb-O polymorphs (Nb-doped TiO₂
anatase & rutile, both tetragonal; TiNb₂O₇ monoclinic) but any
:class:`Phase` list can be supplied.

This is a Qt-free analysis core (numpy only); a thin action wrapper can call
``index_and_calibrate_vectors`` and surface the result.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Candidate phases
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Phase:
    """A candidate crystal phase: name, lattice, and a reflection-condition
    predicate.  ``system`` is one of 'tetragonal' | 'orthorhombic' |
    'cubic' | 'hexagonal' | 'monoclinic' (monoclinic uses β)."""
    name: str
    a: float
    b: float = None
    c: float = None
    system: str = "tetragonal"
    beta_deg: float = 90.0
    # f(h,k,l) -> True if the reflection is allowed (space-group extinctions)
    allowed: object = None

    def __post_init__(self):
        if self.b is None:
            self.b = self.a
        if self.c is None:
            self.c = self.a

    def d_spacing(self, h: int, k: int, l: int) -> float:
        a, b, c = self.a, self.b, self.c
        s = self.system
        if s in ("tetragonal", "cubic"):
            inv2 = (h * h + k * k) / (a * a) + (l * l) / (c * c)
        elif s == "orthorhombic":
            inv2 = h * h / (a * a) + k * k / (b * b) + l * l / (c * c)
        elif s == "hexagonal":
            inv2 = 4.0 / 3.0 * (h * h + h * k + k * k) / (a * a) + l * l / (c * c)
        elif s == "monoclinic":  # unique axis b
            beta = np.radians(self.beta_deg)
            sb = np.sin(beta)
            inv2 = (1.0 / sb ** 2) * (
                h * h / (a * a) + k * k * sb * sb / (b * b) + l * l / (c * c)
                - 2.0 * h * l * np.cos(beta) / (a * c))
        else:
            raise ValueError(f"unknown system {s!r}")
        return 1.0 / np.sqrt(inv2) if inv2 > 0 else np.inf

    def d_list(self, max_index: int = 4, n_keep: int = 12) -> np.ndarray:
        """Unique d-spacings (Å), largest first, for allowed reflections up to
        ``max_index``.  Near-degenerate d's (within 0.5 %) are merged so the
        ratio fingerprint reflects distinct rings, not multiplicities."""
        ds = []
        rng = range(-max_index, max_index + 1)
        for h in rng:
            for k in rng:
                for l in rng:
                    if h == 0 and k == 0 and l == 0:
                        continue
                    if self.allowed is not None and not self.allowed(h, k, l):
                        continue
                    ds.append(self.d_spacing(h, k, l))
        ds = np.array(sorted(set(round(float(d), 4) for d in ds), reverse=True))
        # merge near-degenerate rings
        merged = []
        for d in ds:
            if not merged or abs(merged[-1] - d) / merged[-1] > 0.005:
                merged.append(d)
        return np.array(merged[:n_keep])


# Reflection conditions (extinctions) -----------------------------------------
def _anatase_allowed(h, k, l):
    # I4₁/amd: body-centred (h+k+l even) is the dominant condition used for ratios
    return (h + k + l) % 2 == 0


def _rutile_allowed(h, k, l):
    # P4₂/mnm — primitive; no integral extinction (zonal conditions ignored for
    # the ring-ratio fingerprint, which only uses |g| magnitudes)
    return True


def _primitive_allowed(h, k, l):
    return True


def default_ti_nb_o_phases() -> list[Phase]:
    """Common Ti-Nb-O polymorphs (Nb lightly substitutes Ti, so the TiO₂ cells
    are a good first approximation; TiNb₂O₇ included as the Nb-rich phase)."""
    return [
        Phase("anatase TiO2", a=3.7845, c=9.5143, system="tetragonal",
              allowed=_anatase_allowed),
        Phase("rutile TiO2", a=4.5933, c=2.9592, system="tetragonal",
              allowed=_rutile_allowed),
        # TiNb2O7 monoclinic C2/m (approximate cell); primitive-ratio proxy
        Phase("TiNb2O7", a=20.35, b=3.801, c=11.93, system="monoclinic",
              beta_deg=120.2, allowed=_primitive_allowed),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Centering
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CenterResult:
    center_map: np.ndarray          # (nav_y, nav_x, 2) [cx, cy] px; NaN if none
    global_center: np.ndarray       # (2,) robust [cx, cy] px
    n_centered: int
    n_total: int


def center_vectors_friedel(vecs, *, min_vectors: int = 6) -> CenterResult:
    """Per-pattern Friedel center for every nav position (beam-stop safe).

    ``vecs`` is a :class:`SpyDEDiffractionVectors`.  Returns a per-pattern
    center map (px, in the signal pixel grid) and a robust global center
    (median over patterns with enough vectors)."""
    from spyde.actions.sem_geometry import friedel_center
    ny, nx = vecs.nav_shape
    sig_ax = vecs.sig_axes
    # calibrated -> px conversion for kx (axis0) and ky (axis1)
    kx_scale, kx_off = float(sig_ax[0].scale), float(sig_ax[0].offset)
    ky_scale, ky_off = float(sig_ax[1].scale), float(sig_ax[1].offset)

    cmap = np.full((ny, nx, 2), np.nan, dtype=np.float32)
    centers = []
    for iy in range(ny):
        for ix in range(nx):
            kxy = vecs.kxy_at(iy, ix)
            if len(kxy) < min_vectors:
                continue
            # to pixels
            px = (kxy[:, 0] - kx_off) / kx_scale
            py = (kxy[:, 1] - ky_off) / ky_scale
            c = friedel_center(np.column_stack([px, py]))
            if c is None:
                continue
            cmap[iy, ix] = c
            centers.append(c)
    if centers:
        gc = np.median(np.asarray(centers), axis=0).astype(np.float32)
    else:
        gc = np.array([np.nan, np.nan], dtype=np.float32)
    return CenterResult(cmap, gc, len(centers), ny * nx)


# ─────────────────────────────────────────────────────────────────────────────
# Per-grain 2D reciprocal-basis fit
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BasisFit:
    g1_px: float
    g2_px: float
    angle_deg: float            # angle between g1 and g2
    ratio: float                # |g2|/|g1|
    inlier_frac: float          # fraction of vectors that index to integer (h,k)
    n_vectors: int


def fit_reciprocal_basis(rel_px: np.ndarray, *, tol: float = 0.22,
                         min_inliers: int = 5) -> Optional[BasisFit]:
    """Fit the 2D reciprocal-lattice basis (g1, g2) from one pattern's centered
    vectors ``rel_px`` (px, relative to the pattern center).

    A zone-axis single-crystal pattern is a 2D net v = h·g1 + k·g2; this is the
    correct model (NOT 1D powder rings).  g1 = shortest vector, g2 = shortest
    non-collinear vector; we index every vector to its nearest integer (h,k) and
    least-squares refine the basis on the inliers.  Returns None if the net does
    not explain enough vectors (multi-grain / junk)."""
    r = np.hypot(rel_px[:, 0], rel_px[:, 1])
    o = np.argsort(r)
    rel = rel_px[o]
    r = r[o]
    rel = rel[r > 3.0]
    if len(rel) < min_inliers:
        return None
    g1 = rel[0]
    g2 = None
    for v in rel[1:]:
        cs = abs(np.dot(g1, v) / (np.linalg.norm(g1) * np.linalg.norm(v) + 1e-9))
        if cs < 0.85:
            g2 = v
            break
    if g2 is None:
        return None
    B0 = np.array([g1, g2]).T                  # columns are basis vectors
    try:
        hk = np.linalg.solve(B0, rel.T).T
    except np.linalg.LinAlgError:
        return None
    hki = np.round(hk)
    good = np.abs(hk - hki).max(1) < tol
    if good.sum() < min_inliers:
        return None
    Bt, *_ = np.linalg.lstsq(hki[good], rel[good], rcond=None)   # (2,2)
    g1r, g2r = Bt[0], Bt[1]
    n1, n2 = np.linalg.norm(g1r), np.linalg.norm(g2r)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    # ensure g1 is the shorter
    if n2 < n1:
        n1, n2, g1r, g2r = n2, n1, g2r, g1r
    ang = np.degrees(np.arccos(np.clip(abs(np.dot(g1r, g2r) / (n1 * n2)), 0, 1)))
    return BasisFit(float(n1), float(n2), float(ang), float(n2 / n1),
                    float(good.mean()), int(len(rel)))


def survey_grain_bases(vecs, center_result: CenterResult, *,
                       min_vectors: int = 8, min_inlier_frac: float = 0.5
                       ) -> list[BasisFit]:
    """Fit a 2D reciprocal basis for every pattern with enough vectors; keep the
    fits that index cleanly (single-grain zone-axis patterns).  The spread of
    the returned bases tells you whether the scan is single-phase / single-grain
    or a mix."""
    ny, nx = vecs.nav_shape
    sig_ax = vecs.sig_axes
    kx_scale, kx_off = float(sig_ax[0].scale), float(sig_ax[0].offset)
    ky_scale, ky_off = float(sig_ax[1].scale), float(sig_ax[1].offset)
    gc = center_result.global_center
    fits = []
    for iy in range(ny):
        for ix in range(nx):
            kxy = vecs.kxy_at(iy, ix)
            if len(kxy) < min_vectors:
                continue
            c = center_result.center_map[iy, ix]
            if not np.isfinite(c).all():
                c = gc
            if not np.isfinite(c).all():
                continue
            px = (kxy[:, 0] - kx_off) / kx_scale - c[0]
            py = (kxy[:, 1] - ky_off) / ky_scale - c[1]
            fb = fit_reciprocal_basis(np.column_stack([px, py]))
            if fb is not None and fb.inlier_frac >= min_inlier_frac:
                fits.append(fb)
    return fits


# ─────────────────────────────────────────────────────────────────────────────
# Ring extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_ring_radii(vecs, center_result: CenterResult, *,
                       min_vectors: int = 8, max_r: float = 120.0,
                       bin_px: float = 1.5, n_rings: int = 6,
                       prominence_frac: float = 0.15) -> np.ndarray:
    """Pool |g| (px) of centered vectors over all patterns and return the
    characteristic ring radii (peaks of the radial histogram), smallest first.

    Each pattern is centered by its own Friedel center (falling back to the
    global center) so rings stay sharp across grains at different positions."""
    from scipy.ndimage import gaussian_filter1d
    ny, nx = vecs.nav_shape
    sig_ax = vecs.sig_axes
    kx_scale, kx_off = float(sig_ax[0].scale), float(sig_ax[0].offset)
    ky_scale, ky_off = float(sig_ax[1].scale), float(sig_ax[1].offset)
    gc = center_result.global_center

    radii = []
    for iy in range(ny):
        for ix in range(nx):
            kxy = vecs.kxy_at(iy, ix)
            if len(kxy) < min_vectors:
                continue
            c = center_result.center_map[iy, ix]
            if not np.isfinite(c).all():
                c = gc
            if not np.isfinite(c).all():
                continue
            px = (kxy[:, 0] - kx_off) / kx_scale
            py = (kxy[:, 1] - ky_off) / ky_scale
            r = np.hypot(px - c[0], py - c[1])
            radii.extend(r[r > 3.0].tolist())
    if not radii:
        return np.zeros(0, dtype=np.float32)

    radii = np.asarray(radii)
    edges = np.arange(0, max_r + bin_px, bin_px)
    hist, _ = np.histogram(radii, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sm = gaussian_filter1d(hist.astype(float), 1.0)

    # local maxima above a prominence floor
    peaks = []
    thr = prominence_frac * sm.max()
    for i in range(1, len(sm) - 1):
        if sm[i] > sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] >= thr:
            peaks.append((centers[i], sm[i]))
    peaks.sort(key=lambda t: -t[1])           # by strength
    peaks = sorted(peaks[:n_rings], key=lambda t: t[0])   # then by radius
    return np.array([p[0] for p in peaks], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Phase matching + calibration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PhaseMatch:
    phase: str
    scale_inv_angstrom_per_px: float
    residual: float                 # mean |Δ(ratio)| over matched rings
    n_matched: int
    ring_assignment: list = field(default_factory=list)  # (R_px, hkl_d, 1/d)


def _ratio_match(ring_radii: np.ndarray, d_list: np.ndarray):
    """Match measured ring radii to a phase's d-spacings, scale-free.

    Ring radius R ∝ g = 1/d.  A single-grain zone-axis pattern usually shows
    only a SUBSET of the powder rings, so we do not assume the smallest measured
    ring is the largest d-spacing.  Instead we try every candidate scale formed
    by anchoring the smallest ring R₀ to each predicted reflection g_a, and for
    each anchor greedily match all rings to predicted |g|, keeping the anchor
    with the lowest mean residual.  This makes the fit (and the recovered scale)
    robust to which orders are actually present.

    Returns (mean_residual_in_g_units_normalised, scale [Å⁻¹/px], assignment)."""
    if len(ring_radii) < 2 or len(d_list) < 2:
        return np.inf, np.nan, []
    R = np.sort(ring_radii)
    g_pred = np.sort(1.0 / d_list)              # ascending predicted |g| (Å⁻¹)

    best = (np.inf, np.nan, [])
    for a in range(len(g_pred)):
        scale = g_pred[a] / R[0]                 # anchor smallest ring to g_pred[a]
        if not np.isfinite(scale) or scale <= 0:
            continue
        g_meas = R * scale                       # measured |g| under this scale
        resid = 0.0
        assign = []
        used = set()
        ok = True
        for gm in g_meas:
            diffs = np.abs(g_pred - gm)
            for j in np.argsort(diffs):
                if j in used:
                    continue
                used.add(int(j))
                # residual normalised by the predicted |g| (relative error)
                resid += diffs[j] / g_pred[j]
                assign.append((float(gm), int(j)))
                break
            else:
                ok = False
        if not ok or not assign:
            continue
        resid /= len(g_meas)
        if resid < best[0]:
            d_sorted = np.sort(d_list)            # ascending d? we want by g
            full = [(float(gm / scale), float(1.0 / g_pred[j]), float(g_pred[j]))
                    for (gm, j) in assign]
            best = (resid, float(scale), full)
    return best


def match_phase(ring_radii: np.ndarray,
                phases: Sequence[Phase] = None,
                max_index: int = 4) -> list[PhaseMatch]:
    """Rank candidate phases by how well their d-spacing-ratio fingerprint
    matches the measured ring radii.  Returns matches sorted best-first."""
    if phases is None:
        phases = default_ti_nb_o_phases()
    out = []
    for ph in phases:
        d_list = ph.d_list(max_index=max_index)
        resid, scale, assign = _ratio_match(ring_radii, d_list)
        out.append(PhaseMatch(ph.name, scale, resid, len(assign), assign))
    out.sort(key=lambda m: m.residual)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Top-level
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class IndexCalibResult:
    center: CenterResult
    ring_radii_px: np.ndarray
    grain_bases: list             # list[BasisFit] from clean single-grain frames
    phase_matches: list           # list[PhaseMatch], best first
    best: Optional[PhaseMatch]
    single_grain_consistent: bool # True if grain bases cluster (one phase/zone)
    confidence: str               # 'high' | 'medium' | 'low' verdict

    @property
    def scale(self) -> float:
        """Best-phase reciprocal-space calibration (Å⁻¹/px).  Trust this only
        when ``confidence`` is not 'low'."""
        return self.best.scale_inv_angstrom_per_px if self.best else float("nan")

    def angstrom_per_px_from_dspacing(self, ring_index: int, d_angstrom: float
                                      ) -> float:
        """Direct calibration from ONE user-supplied (ring → d-spacing)
        assignment — the reliable route when the phase is known.  Returns the
        Å⁻¹/px scale: g = 1/d, scale = g / R_px."""
        R = float(np.sort(self.ring_radii_px)[ring_index])
        return (1.0 / float(d_angstrom)) / R if R > 0 else float("nan")


def _grain_consistency(fits: list) -> tuple[bool, str]:
    """Judge whether the grain bases cluster (single phase/zone) or scatter
    (multi-grain).  Returns (is_consistent, confidence_label)."""
    if len(fits) < 3:
        return False, "low"
    ang = np.array([f.angle_deg for f in fits])
    ratio = np.array([f.ratio for f in fits])
    g1 = np.array([f.g1_px for f in fits])
    # coefficient of variation of |g1| and spread of angle
    cv_g1 = np.std(g1) / (np.mean(g1) + 1e-9)
    ang_iqr = np.percentile(ang, 75) - np.percentile(ang, 25)
    if cv_g1 < 0.12 and ang_iqr < 12:
        return True, "high"
    if cv_g1 < 0.25 and ang_iqr < 25:
        return True, "medium"
    return False, "low"


def index_and_calibrate_vectors(vecs, phases: Sequence[Phase] = None, *,
                                max_index: int = 4) -> IndexCalibResult:
    """Full pipeline.

    1. Friedel-center every pattern (beam-stop safe).
    2. Fit a 2D reciprocal basis per clean single-grain pattern (the correct
       model for zone-axis single-crystal data) and judge whether the scan is
       single-phase/single-grain or a mix — this governs how much to trust the
       phase ID.
    3. Pool ring radii and rank candidate phases by scale-free d-ratio.

    **Honesty note:** ratio-only phase ID from a few rings is weak, and a
    large-cell phase can over-fit by sheer reflection density.  When
    ``confidence`` is 'low' (scattered grain bases), treat the phase ranking as
    tentative and calibrate instead from a *known* (ring → d-spacing) assignment
    via :meth:`IndexCalibResult.angstrom_per_px_from_dspacing`.
    """
    center = center_vectors_friedel(vecs)
    bases = survey_grain_bases(vecs, center)
    consistent, confidence = _grain_consistency(bases)
    rings = extract_ring_radii(vecs, center)
    matches = match_phase(rings, phases=phases, max_index=max_index)
    best = matches[0] if matches else None
    return IndexCalibResult(center, rings, bases, matches, best,
                            consistent, confidence)
