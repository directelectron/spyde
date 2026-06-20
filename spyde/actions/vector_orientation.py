"""
vector_orientation.py — orientation + strain mapping on diffraction vectors.

Operates on the sparse `(kx, ky, intensity)` peaks of a
`SpyDEDiffractionVectors` rather than dense patterns. Per pattern:

  1. **coarse seed** — pick the discrete orientation branch by running the
     proven pyxem polar matcher on the vectors rasterised onto its exact grid
     (seed only; never the strain fit).
  2. **continuous refine** — Levenberg-Marquardt over pose `(theta, A, t)` with
     `v ≈ A · Rot(theta) · g_template + t`, using an intensity-weighted
     soft-assign + no-match-sink cost (robust to missing/spurious peaks). The
     2×2 affine `A` yields orientation + strain directly.

Design & validation: `VECTOR_ORIENTATION_MAPPING_PLAN.md` §7b–7d. Validated on
synthetic known-strain sets and real sped_ag (residual ~1.4 px, strain ~2-3%,
~42 ms/pattern).

No Qt imports — importable on dask workers and in tests. The matching primitives
reuse the dense OM library generation in `orientation_compute.py`.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from spyde.signals.diffraction_vectors import COL_KX, COL_KY, COL_INTENSITY

log = logging.getLogger(__name__)


# Pose vector layout: [theta, a00, a01, a10, a11, tx, ty]
_P_THETA = 0
_P_A = slice(1, 5)
_P_T = slice(5, 7)


# ─────────────────────────────────────────────────────────────────────────────
# Defaults (validated on sped_ag; see plan §7d). All overridable via params.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    strain_cap=0.05,        # |strain| bound (symmetric-part singular values)
    sink_bw=0.04,           # no-match sink bandwidth, Å⁻¹
    sigma_schedule=(0.06, 0.03, 0.015),   # soft-assign anneal, Å⁻¹
    strain_penalty=50.0,    # weight on the in-residual strain-band penalty
    max_nfev=200,
    n_seed=1,               # coarse-search candidate branches to refine
    coarse_NR=100,
    coarse_NA=360,
)


# ─────────────────────────────────────────────────────────────────────────────
# Library → per-template spot tables (reuse dense generation)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TemplateLibrary:
    """Per-template diffraction spots in calibrated Å⁻¹ plus the dense polar
    matching cache used only for the coarse seed."""
    spots_xy: List[np.ndarray]      # list of (M_i, 2) float32, per template
    spots_I: List[np.ndarray]       # list of (M_i,) float32
    template_quats: np.ndarray      # (N, 4) float64
    template_phase: np.ndarray      # (N,) int16
    phases_meta: list               # [{'name','point_group'}, ...]
    cache: dict                     # build_matching_cache output (coarse seed)
    radial_range: tuple             # (r0, r1) Å⁻¹ for the coarse grid
    r_max: float


def build_template_library(sim, calibration_signal, r_max: float,
                           coarse_NR: int = 100, coarse_NA: int = 360
                           ) -> TemplateLibrary:
    """
    Extract per-template spots from a diffsims Simulation2D and build the dense
    polar matching cache for coarse seeding.

    Parameters
    ----------
    sim : diffsims Simulation2D (from orientation_compute.generate_library_from_phases)
    calibration_signal : an ElectronDiffraction2D whose signal axes carry the
        real data's calibration (scale/offset) — used to build the coarse grid
        on the same (r, theta) axes the matcher expects.
    r_max : outer reciprocal radius (Å⁻¹) to keep spots within.
    """
    from spyde.actions.orientation_compute import (
        build_matching_cache, template_tables, sim_phases_list,
    )
    from spyde.signals.orientation_map import phase_to_dict

    rots = sim.rotations
    n = rots.size if hasattr(rots, "size") else len(rots)

    spots_xy: List[np.ndarray] = []
    spots_I: List[np.ndarray] = []
    for i in range(n):
        _r, _p, dv = sim.get_simulation(i)
        # Raw template spots (kx, ky) in Å⁻¹. Unlike the dense overlay path
        # (pyxem._get_best_fit_spots) we deliberately do NOT apply pyxem's
        # mirror/rotate/negate display convention here: the fit's free Rot(θ)
        # and affine A absorb in-plane rotation and mirror, so the matcher must
        # see the un-transformed template (audit 2026-06-15, plan §10).
        xy = np.asarray(dv.data[:, :2], dtype=np.float32)
        inten = np.asarray(dv.intensity, dtype=np.float32)
        rr = np.sqrt((xy ** 2).sum(1))
        keep = (rr > 1e-3) & (rr < r_max)
        spots_xy.append(xy[keep])
        spots_I.append(inten[keep])

    template_quats, template_phase = template_tables(sim)
    phases_meta = [phase_to_dict(p) for p in sim_phases_list(sim)]

    cache = build_matching_cache(calibration_signal, sim)
    cache["NR"], cache["NA"] = coarse_NR, coarse_NA
    # Rebuild cache at requested resolution if it differs from build default.
    if (cache.get("NR") != coarse_NR) or (cache.get("NA") != coarse_NA):
        cache = build_matching_cache(calibration_signal, sim)
    _sl, _f, _fs, radial_range = calibration_signal.calibration.get_slices2d(
        cache["NR"], cache["NA"]
    )

    return TemplateLibrary(
        spots_xy=spots_xy, spots_I=spots_I,
        template_quats=template_quats, template_phase=template_phase,
        phases_meta=phases_meta, cache=cache,
        radial_range=(float(radial_range[0]), float(radial_range[1])),
        r_max=float(r_max),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pose model + cost
# ─────────────────────────────────────────────────────────────────────────────

def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def project_spots(params: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Project template spots g (M,2) through pose params → (M,2)."""
    A = params[_P_A].reshape(2, 2)
    t = params[_P_T]
    return (g @ _rot(params[_P_THETA]).T) @ A.T + t


def strain_from_pose(params: np.ndarray) -> np.ndarray:
    """Rotation-free symmetric strain tensor (2,2) from a pose.

    The full template→measured linear map is `M = A · Rot(theta)`. LM is free to
    split the total rotation between `theta` and `A`, so `0.5(A+Aᵀ)−I` is NOT
    pure strain (it leaks residual rotation). Polar-decompose `M = R · S` (S
    symmetric positive-definite) and report `S − I` — the rotation R is absorbed
    into the orientation, leaving only the physical stretch/strain.
    """
    A = params[_P_A].reshape(2, 2)
    M = A @ _rot(params[_P_THETA])
    # right polar: M = R S, S = sqrt(MᵀM)
    U, sv, Vt = np.linalg.svd(M)
    S = (Vt.T * sv) @ Vt
    return S - np.eye(2)


def project_strain_bound(A: np.ndarray, cap: float = 0.05) -> np.ndarray:
    """Clamp strain to ±cap and forbid reflections (det must stay > 0).

    A reflection here is a mirror alias of a wrong seed, not physical strain;
    the SVD clamp alone would preserve it, so flip a singular value sign first.
    """
    U, sv, Vt = np.linalg.svd(A)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        U[:, -1] *= -1
        sv[-1] *= -1
    sv = np.clip(sv, 1.0 - cap, 1.0 + cap)
    return U @ np.diag(sv) @ Vt


def _residual(params, g, gI, v, vI, sigma, sink_bw, cap, pen):
    """LM residual: template→measured soft-assign + sink + strain-band penalty.

    Template→measured (each template spot pulled to its soft-nearest measured
    vector) means the affine can't shrink templates to a point — every template
    spot must land on data. The sink lets template spots with no measured
    support (missing reflections) opt out; weighting by measured intensity makes
    spurious low-I peaks contribute little. The penalty band keeps singular
    values of A within [1±cap] so LM never explores the collapse.
    """
    p = project_spots(params, g)                          # (M, 2)
    d2 = ((p[:, None, :] - v[None, :, :]) ** 2).sum(-1)   # (M, Nv)
    w = np.exp(-d2 / (2 * sigma ** 2)) * vI[None, :]
    raw = w.sum(1)
    wn = w / (raw[:, None] + 1e-9)
    target = (wn[..., None] * v[None, :, :]).sum(1)       # soft-nearest measured
    sink = np.exp(-(sink_bw ** 2) / (2 * sigma ** 2))
    conf = raw / (raw + sink)                             # matched-ness in [0,1)
    r_match = ((p - target) * (np.sqrt(gI) * conf)[:, None]).ravel()

    A = params[_P_A].reshape(2, 2)
    sv = np.linalg.svd(A, compute_uv=False)
    excess = np.clip(np.abs(sv - 1.0) - cap, 0.0, None)
    return np.concatenate([r_match, pen * excess])


# ─────────────────────────────────────────────────────────────────────────────
# Per-pattern fit
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatternFit:
    template_idx: int
    quat: np.ndarray            # (4,) resolved orientation (w,x,y,z)
    phase_idx: int
    theta: float                # in-plane angle, rad
    affine: np.ndarray          # (2,2) template→measured
    strain: np.ndarray          # (2,2) symmetric strain tensor (A_sym − I)
    translation: np.ndarray     # (2,) beam-center residual, Å⁻¹
    residual: float             # mean |match residual|, Å⁻¹
    friedel_asym: float         # g/−g residual asymmetry QC, Å⁻¹ (nan if no pair)
    n_matched: int              # template spots with measured support
    coarse_score: float


def _rasterize_to_polar(meas_xy, meas_I, cache, radial_range):
    """Splat sparse vectors onto the matcher's (NA, NR) grid with a 3×3
    Gaussian per spot (templates are smoothed too)."""
    NR, NA = cache["NR"], cache["NA"]
    r0, r1 = radial_range
    r = np.sqrt((meas_xy ** 2).sum(1))
    th = np.arctan2(meas_xy[:, 1], meas_xy[:, 0])
    polar = np.zeros((NA, NR), dtype=float)
    span = (r1 - r0) if (r1 - r0) else 1.0
    for k in range(len(meas_xy)):
        rb = (r[k] - r0) / span * NR
        if not (0 <= rb < NR):
            continue
        ri = int(rb)
        ai = int((th[k] + np.pi) / (2 * np.pi) * NA) % NA
        for da in (-1, 0, 1):
            for dr in (-1, 0, 1):
                rj = ri + dr
                if 0 <= rj < NR:
                    polar[(ai + da) % NA, rj] += meas_I[k] * np.exp(
                        -(da * da + dr * dr) / 2.0)
    return polar


def coarse_seed(meas_xy, meas_I, lib: TemplateLibrary, n_seed: int = 1,
                gamma: float = 0.5):
    """Return up to n_seed (template_idx, seed_angle_rad, score), best first."""
    from pyxem.utils.indexation_utils import _mixed_matching_lib_to_polar
    cache = lib.cache
    polar = _rasterize_to_polar(meas_xy, meas_I, cache, lib.radial_range)
    polar = np.nan_to_num(polar ** gamma)
    result = _mixed_matching_lib_to_polar(
        polar,
        integrated_templates=cache["integrated"],
        r_templates=cache["r_templates"],
        theta_templates=cache["theta_templates"],
        intensities_templates=cache["intensities_norm"],
        n_keep=None, frac_keep=1.0, n_best=max(1, n_seed), transpose=False,
    )
    rows = np.atleast_2d(result)[:max(1, n_seed)]
    NA = cache["NA"]
    out = []
    for row in rows:
        bt = int(row[0])
        angle = np.deg2rad(row[2] / NA * 360.0 - 180.0)
        out.append((bt, angle, float(row[1])))
    return out


def _friedel_asymmetry(params, g, v, sigma=0.02):
    """Mean ||res(g) + res(−g)|| over matched centrosymmetric template pairs.
    Real strain is centrosymmetric ⇒ this cancels; a high value flags skewed
    vector finding (miscentered beam / detector distortion), not strain."""
    p = project_spots(params, g)
    d2 = ((p[:, None, :] - v[None, :, :]) ** 2).sum(-1)
    j = d2.argmin(1)
    res = v[j] - p
    matched = np.sqrt(d2[np.arange(len(p)), j]) < 3 * sigma
    gd2 = ((g[:, None, :] + g[None, :, :]) ** 2).sum(-1)
    opp = gd2.argmin(1)
    is_pair = gd2[np.arange(len(g)), opp] < 1e-3
    vals = [np.linalg.norm(res[i] + res[opp[i]])
            for i in range(len(g))
            if is_pair[i] and matched[i] and matched[opp[i]]]
    return float(np.mean(vals)) if vals else float("nan")


def fit_pattern(meas_xy: np.ndarray, meas_I: np.ndarray,
                lib: TemplateLibrary, params: Optional[dict] = None,
                seed: Optional[Tuple[int, float, float]] = None
                ) -> Optional[PatternFit]:
    """
    Fit orientation + strain for one pattern's vectors.

    seed : optional (template_idx, seed_angle_rad, score) to skip the coarse
        search — used for whole-field warm-start propagation. If None, the
        coarse seed is computed here.
    Returns None if there are too few vectors to fit.
    """
    from scipy.optimize import least_squares
    P = {**DEFAULTS, **(params or {})}

    if len(meas_xy) < 4:
        return None
    meas_xy = np.asarray(meas_xy, dtype=np.float64)
    meas_I = np.asarray(meas_I, dtype=np.float64)

    if seed is not None:
        candidates = [seed]
    elif lib.cache:
        candidates = coarse_seed(meas_xy, meas_I, lib, P["n_seed"])
    else:
        # No coarse cache (e.g. a single-template library): seed every template
        # at angle 0 and let the refine pick the best by residual. Fine for tiny
        # libraries; large ones should always carry a cache for the pyxem seed.
        candidates = [(i, 0.0, 0.0) for i in range(len(lib.spots_xy))]

    cap = P["strain_cap"]
    best = None
    for (bt, seed_angle, cscore) in candidates:
        g = np.asarray(lib.spots_xy[bt], dtype=np.float64)
        gI = np.asarray(lib.spots_I[bt], dtype=np.float64)
        if len(g) < 3:
            continue
        p0 = np.array([seed_angle, 1, 0, 0, 1, 0, 0], dtype=np.float64)
        nfev = 0
        for sigma in P["sigma_schedule"]:
            sol = least_squares(
                _residual, p0, method="lm", max_nfev=P["max_nfev"],
                args=(g, gI, meas_xy, meas_I, sigma, P["sink_bw"], cap,
                      P["strain_penalty"]),
            )
            p0 = sol.x
            nfev += sol.nfev
        p0[_P_A] = project_strain_bound(p0[_P_A].reshape(2, 2), cap).ravel()

        # score the converged fit: mean match residual at the finest sigma
        p = project_spots(p0, g)
        d2 = ((p[:, None, :] - meas_xy[None, :, :]) ** 2).sum(-1)
        mind = np.sqrt(d2.min(1))
        matched = mind < 3 * P["sigma_schedule"][-1]
        resid = float(mind[matched].mean()) if matched.any() else float("inf")

        if best is None or resid < best[0]:
            best = (resid, bt, p0, cscore, int(matched.sum()), g)

    if best is None:
        return None

    resid, bt, p0, cscore, n_matched, g = best
    A = p0[_P_A].reshape(2, 2)
    strain = strain_from_pose(p0)
    quat = _resolve_one_quat(bt, p0[_P_THETA], lib)
    fa = _friedel_asymmetry(p0, g, meas_xy)
    return PatternFit(
        template_idx=bt, quat=quat,
        phase_idx=int(lib.template_phase[bt]),
        theta=float(p0[_P_THETA]), affine=A.astype(np.float32),
        strain=strain.astype(np.float32),
        translation=p0[_P_T].astype(np.float32),
        residual=resid, friedel_asym=fa, n_matched=n_matched,
        coarse_score=cscore,
    )


def _resolve_one_quat(template_idx: int, theta: float, lib: TemplateLibrary
                      ) -> np.ndarray:
    """Resolve one template + in-plane angle into a full orientation quaternion,
    matching orientation_compute.resolve_quaternions for a single entry."""
    from orix.quaternion import Orientation
    base = lib.template_quats[int(template_idx)]
    euler = Orientation(base).to_euler(degrees=True)
    euler = np.atleast_2d(euler)
    euler[:, 0] = np.rad2deg(theta)
    return Orientation.from_euler(euler, degrees=True).data[0].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Batch driver over a SpyDEDiffractionVectors (whole-field, warm-start)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VectorOrientationResult:
    """Per-position orientation + strain maps from a vector-OM run."""
    quats: np.ndarray           # (ny, nx, 4) float32
    phase_idx: np.ndarray       # (ny, nx) int16
    theta: np.ndarray           # (ny, nx) float32 (rad)
    strain: np.ndarray          # (ny, nx, 3) float32 [exx, eyy, exy]
    residual: np.ndarray        # (ny, nx) float32
    friedel_asym: np.ndarray    # (ny, nx) float32
    n_matched: np.ndarray       # (ny, nx) int16
    coarse_score: np.ndarray    # (ny, nx) float32
    phases_meta: list
    nav_shape: tuple
    params: dict = field(default_factory=dict)

    def strain_map(self, component: str = "exx") -> np.ndarray:
        idx = {"exx": 0, "eyy": 1, "exy": 2}[component]
        return self.strain[..., idx]

    def dilatation_map(self) -> np.ndarray:
        return self.strain[..., 0] + self.strain[..., 1]

    def shear_map(self) -> np.ndarray:
        return self.strain[..., 2]

    def smoothed_strain(self, method: str = "tv", weight: float = 0.03,
                        size: int = 3) -> np.ndarray:
        """(ny, nx, 3) edge-preserving denoised strain field.

        The per-pattern strain has a ~0.02 noise floor (peak-finding limited).
        Neighbouring positions share the true strain, so field-level denoising
        recovers it. Benchmark (plan §7h, benchmarks.md):

          - ``method="tv"`` (default) — total-variation (Chambolle). Most robust;
            the gap over the raw fit *widens* with noise/dropout (6x better at
            high noise) because its piecewise-constant prior matches grains.
            ``weight`` ≈ λ, higher = smoother.
          - ``method="median"`` — median filter of ``size``. Good at low noise,
            plateaus as noise rises. No skimage dependency.

        Per-pattern re-fitting with smoothed seeds (iterated / joint) was tried
        and is WORSE — the affine re-absorbs noise, undoing the smoothing. So
        this is strictly a post-process; the raw ``strain`` is always kept.
        NaNs (unfit positions) are preserved.
        """
        comp_nan = [np.isnan(self.strain[..., c]) for c in range(3)]
        if method == "tv":
            try:
                from skimage.restoration import denoise_tv_chambolle
                out = np.empty_like(self.strain)
                for c in range(self.strain.shape[-1]):
                    comp = self.strain[..., c].astype(float)
                    fill = np.nanmedian(comp) if np.isnan(comp).any() else 0.0
                    filled = np.where(np.isnan(comp), fill, comp)
                    out[..., c] = denoise_tv_chambolle(filled, weight=weight)
                    out[..., c][comp_nan[c]] = np.nan
                return out
            except Exception:
                method = "median"  # skimage missing → fall back
        from scipy.ndimage import median_filter
        out = np.empty_like(self.strain)
        for c in range(self.strain.shape[-1]):
            comp = self.strain[..., c]
            filled = np.where(np.isnan(comp), np.nanmedian(comp), comp)
            out[..., c] = median_filter(filled, size=size)
            out[..., c][comp_nan[c]] = np.nan
        return out

    def to_orientation_map(self):
        """Wrap as a SpyDEOrientationMap (n_best=1) for IPF/correlation maps,
        save/load, and the existing orientation-map machinery. The vector path's
        strain stays on this result object; the OM container handles orientation."""
        from spyde.signals.orientation_map import SpyDEOrientationMap
        ny, nx = self.nav_shape
        return SpyDEOrientationMap(
            quats=self.quats[:, :, np.newaxis, :],
            corr=self.coarse_score[:, :, np.newaxis].astype(np.float32),
            phase_idx=self.phase_idx[:, :, np.newaxis].astype(np.int16),
            mirror=np.ones((ny, nx, 1), np.int8),
            phases=self.phases_meta,
            params=dict(self.params),
        )

    def ipf_color_map(self, direction: str = "z") -> np.ndarray:
        """(ny, nx, 3) uint8 IPF color map of the best orientation per position."""
        return self.to_orientation_map().ipf_color_map(direction)


def _snake_order(ny: int, nx: int):
    """Boustrophedon nav order so each step's neighbor (the previous fit) is
    spatially adjacent — the warm-start seed stays valid."""
    for iy in range(ny):
        xs = range(nx) if iy % 2 == 0 else range(nx - 1, -1, -1)
        for ix in xs:
            yield iy, ix


# Live preview buffer layout: 12 channels per nav position.
#   [0:9]  = IPF RGB for X | Y | Z (uint8-range float, 0..255)
#   [9:12] = strain εxx, εyy, εxy (signed float)
_PREVIEW_CHANNELS = 12


def fit_rows_block(rows_list, lib: TemplateLibrary, params: dict,
                   warm_start: bool = False):
    """Fit a flat list of per-position vector rows. Pure/headless — the unit of
    work for both the serial driver and the chunked/parallel one.

    rows_list : list of (Ni, 6) flat-buffer slices (one per nav position, in
        scan order). Entries with <4 vectors yield a None fit.
    Returns a list of (PatternFit | None) the same length as rows_list.
    """
    out = []
    prev_seed = None
    reseed_resid = 2.0 * params.get("sigma_schedule", DEFAULTS["sigma_schedule"])[-1]
    reseed_strain = 0.8 * params.get("strain_cap", DEFAULTS["strain_cap"])
    for rows in rows_list:
        if rows is None or len(rows) < 4:
            out.append(None)
            prev_seed = None
            continue
        mxy = rows[:, [COL_KX, COL_KY]].astype(np.float64)
        mI = rows[:, COL_INTENSITY].astype(np.float64)
        fit = None
        if warm_start and prev_seed is not None:
            fit = fit_pattern(mxy, mI, lib, params, seed=prev_seed)
            if (fit is None or fit.residual > reseed_resid
                    or np.abs(fit.strain).max() > reseed_strain):
                fit = None
        if fit is None:
            fit = fit_pattern(mxy, mI, lib, params)
        out.append(fit)
        if fit is not None:
            prev_seed = (fit.template_idx, fit.theta, fit.coarse_score)
        else:
            prev_seed = None
    return out


def _fits_to_arrays(fits, ny, nx):
    """Assemble a (ny*nx) list of fits (scan order) into per-field arrays."""
    quats = np.zeros((ny, nx, 4), np.float32); quats[..., 0] = 1.0
    phase_idx = np.zeros((ny, nx), np.int16)
    theta = np.zeros((ny, nx), np.float32)
    strain = np.full((ny, nx, 3), np.nan, np.float32)
    residual = np.full((ny, nx), np.nan, np.float32)
    friedel = np.full((ny, nx), np.nan, np.float32)
    n_matched = np.zeros((ny, nx), np.int16)
    coarse = np.zeros((ny, nx), np.float32)
    k = 0
    for iy in range(ny):
        for ix in range(nx):
            fit = fits[k]; k += 1
            if fit is None:
                continue
            quats[iy, ix] = fit.quat
            phase_idx[iy, ix] = fit.phase_idx
            theta[iy, ix] = fit.theta
            strain[iy, ix] = (fit.strain[0, 0], fit.strain[1, 1],
                              fit.strain[0, 1])
            residual[iy, ix] = fit.residual
            friedel[iy, ix] = fit.friedel_asym
            n_matched[iy, ix] = fit.n_matched
            coarse[iy, ix] = fit.coarse_score
    return dict(quats=quats, phase_idx=phase_idx, theta=theta, strain=strain,
                residual=residual, friedel_asym=friedel,
                n_matched=n_matched, coarse_score=coarse)


def _block_preview_rgb(block_arrays, lib, ny, nx):
    """(ny, nx, 12) live-preview block: IPF RGB X|Y|Z + strain εxx,εyy,εxy."""
    from spyde.signals.orientation_map import SpyDEOrientationMap
    a = block_arrays
    om = SpyDEOrientationMap(
        quats=a["quats"][:, :, np.newaxis, :],
        corr=a["coarse_score"][:, :, np.newaxis].astype(np.float32),
        phase_idx=a["phase_idx"][:, :, np.newaxis].astype(np.int16),
        mirror=np.ones((ny, nx, 1), np.int8),
        phases=lib.phases_meta,
    )
    buf = np.zeros((ny, nx, _PREVIEW_CHANNELS), np.float32)
    for di, d in enumerate(("x", "y", "z")):
        buf[..., 3 * di:3 * di + 3] = om.ipf_color_map(d).astype(np.float32)
    buf[..., 9:12] = a["strain"]
    return buf


def _vector_chunk(rows_list, lib, params, warm_start,
                  block_origin, block_shape, shm_name, nav_2d_shape):
    """Fit a rectangular nav block (rows pre-sliced) and, if shm_name is set,
    write its 12-channel preview into the live buffer at block_origin. Runs on
    a dask worker. Returns (ny, nx, 9) packed fit fields for final assembly:
    [quat(4), corr(1), phase(1), strain(3)]."""
    ny, nx = block_shape
    fits = fit_rows_block(rows_list, lib, params, warm_start=warm_start)
    arr = _fits_to_arrays(fits, ny, nx)
    if shm_name is not None and nav_2d_shape is not None:
        try:
            from multiprocessing import shared_memory as _shm
            sh = _shm.SharedMemory(name=shm_name, create=False)
            try:
                full = np.ndarray(tuple(nav_2d_shape) + (_PREVIEW_CHANNELS,),
                                  dtype=np.float32, buffer=sh.buf)
                preview = _block_preview_rgb(arr, lib, ny, nx)
                y0, x0 = block_origin
                full[y0:y0 + ny, x0:x0 + nx, :] = preview
                del full
            finally:
                sh.close()
        except Exception as e:
            log.debug("live vector-orientation block preview write failed: %s", e)
    # pack fit fields: quat(4) corr(1) phase(1) strain(3) residual(1)
    # friedel(1) n_matched(1) = 12
    packed = np.full((ny, nx, 12), np.nan, np.float32)
    packed[..., 0:4] = arr["quats"]
    packed[..., 4] = arr["coarse_score"]
    packed[..., 5] = arr["phase_idx"]
    packed[..., 6:9] = arr["strain"]
    packed[..., 9] = arr["residual"]
    packed[..., 10] = arr["friedel_asym"]
    packed[..., 11] = arr["n_matched"]
    return packed


def compute_vector_orientation(
    vectors, lib: TemplateLibrary, params: Optional[dict] = None,
    t: Optional[int] = None, warm_start: bool = False,
    progress=None, stopped_flag=None,
) -> VectorOrientationResult:
    """
    Fit orientation + strain for every position of a SpyDEDiffractionVectors.

    warm_start : seed each pattern from its converged spatial neighbour (snake
        order), falling back to a fresh coarse search when the warm fit is poor.

        OFF by default: on real sped_ag the independent path is both faster
        (~59 vs ~100 ms/pattern) and more accurate (strain median 0.008 vs
        0.046). The bounded affine can fit a slightly-wrong neighbour-seeded
        orientation by absorbing the mismatch as spurious strain, so a low
        residual doesn't guarantee the right branch — and the residual-gated
        reseed then pays for both the warm AND the cold fit. Warm-start is kept
        for datasets with very large libraries where the cold coarse search
        dominates, but it must be validated per-dataset (see plan §7e).
        A warm fit is rejected (→ fresh coarse search) when its residual OR its
        strain magnitude exceeds threshold — large strain flags a wrong branch.
    t : time index for 5D vectors (None → 4D / all-time at (iy, ix)).
    progress : optional callable(done, total).
    """
    P = {**DEFAULTS, **(params or {})}
    ny, nx = vectors.nav_shape
    total = ny * nx

    quats = np.zeros((ny, nx, 4), np.float32)
    quats[..., 0] = 1.0
    phase_idx = np.zeros((ny, nx), np.int16)
    theta = np.zeros((ny, nx), np.float32)
    strain = np.zeros((ny, nx, 3), np.float32)
    residual = np.full((ny, nx), np.nan, np.float32)
    friedel = np.full((ny, nx), np.nan, np.float32)
    n_matched = np.zeros((ny, nx), np.int16)
    coarse = np.zeros((ny, nx), np.float32)

    # a warm-started fit is rejected (→ fresh coarse search) when residual or
    # strain magnitude exceeds these — large strain flags a wrong branch the
    # affine absorbed rather than a genuine fit.
    reseed_resid = 2.0 * P["sigma_schedule"][-1]
    reseed_strain = 0.8 * P["strain_cap"]

    prev_seed = None
    done = 0
    for iy, ix in _snake_order(ny, nx):
        if stopped_flag is not None and stopped_flag[0]:
            break
        rows = (vectors.at_t(iy, ix, t) if (t is not None and vectors.n_time > 0)
                else vectors.at(iy, ix))
        done += 1
        if progress is not None and (done % 64 == 0 or done == total):
            progress(done, total)
        if len(rows) < 4:
            prev_seed = None
            continue
        mxy = rows[:, [COL_KX, COL_KY]].astype(np.float64)
        mI = rows[:, COL_INTENSITY].astype(np.float64)

        fit = None
        if warm_start and prev_seed is not None:
            fit = fit_pattern(mxy, mI, lib, P,
                              seed=(prev_seed[0], prev_seed[1], prev_seed[2]))
            if (fit is None or fit.residual > reseed_resid
                    or np.abs(fit.strain).max() > reseed_strain):
                fit = None  # warm start unreliable → fresh coarse search below
        if fit is None:
            fit = fit_pattern(mxy, mI, lib, P)
        if fit is None:
            prev_seed = None
            continue

        quats[iy, ix] = fit.quat
        phase_idx[iy, ix] = fit.phase_idx
        theta[iy, ix] = fit.theta
        strain[iy, ix] = (fit.strain[0, 0], fit.strain[1, 1], fit.strain[0, 1])
        residual[iy, ix] = fit.residual
        friedel[iy, ix] = fit.friedel_asym
        n_matched[iy, ix] = fit.n_matched
        coarse[iy, ix] = fit.coarse_score
        prev_seed = (fit.template_idx, fit.theta, fit.coarse_score)

    return VectorOrientationResult(
        quats=quats, phase_idx=phase_idx, theta=theta, strain=strain,
        residual=residual, friedel_asym=friedel, n_matched=n_matched,
        coarse_score=coarse, phases_meta=lib.phases_meta,
        nav_shape=(ny, nx), params=dict(P),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chunked / parallel batch driver (mirrors orientation_compute._do_compute_*)
# ─────────────────────────────────────────────────────────────────────────────

def _slice_chunk_rows(vectors, y0, y1, x0, x1, t):
    """Per-position row slices for nav block [y0:y1, x0:x1], scan order.
    The CSR buffer makes each .at()/.at_t() an O(1) slice (no copy of the
    whole dataset)."""
    rows_list = []
    use_t = t is not None and vectors.n_time > 0
    for iy in range(y0, y1):
        for ix in range(x0, x1):
            rows = (vectors.at_t(iy, ix, t) if use_t else vectors.at(iy, ix))
            rows_list.append(np.asarray(rows, dtype=np.float32))
    return rows_list


def compute_vector_orientation_chunked(
    vectors, lib: TemplateLibrary, params: Optional[dict] = None,
    t: Optional[int] = None, warm_start: bool = False,
    main_window=None, signal_tree=None, shm_name: Optional[str] = None,
    chunk: int = 16, stopped_flag=None, progress=None,
) -> Optional[VectorOrientationResult]:
    """
    Whole-field vector OM, parallelised across the cluster with a live preview —
    the same architecture as the dense ``_do_compute_orientations``.

    The per-pattern LM fit is CPU-bound Python, so the speed-up comes from
    running nav-CHUNKS concurrently across every worker (the serial
    ``compute_vector_orientation`` used one core). Each chunk writes its
    12-channel preview (IPF X|Y|Z + strain) into ``shm_name`` as it finishes,
    so the GUI paints the map in progressively.

    Returns a VectorOrientationResult, or None if stopped. Falls back to a
    local thread pool when no distributed client is reachable (e.g. tests).
    """
    import dask

    P = {**DEFAULTS, **(params or {})}
    ny, nx = vectors.nav_shape

    # Client (same env policy as find_vectors / orientation_compute).
    client = None
    if signal_tree is not None:
        client = getattr(signal_tree, "client", None)
    if client is None and main_window is not None:
        client = getattr(getattr(main_window, "dask_manager", None),
                         "client", None)

    # Enumerate nav chunks → (y0, y1, x0, x1) blocks.
    cy, cx = max(1, min(chunk, ny)), max(1, min(chunk, nx))
    blocks = [(y0, min(y0 + cy, ny), x0, min(x0 + cx, nx))
              for y0 in range(0, ny, cy) for x0 in range(0, nx, cx)]
    packed = np.full((ny, nx, 12), np.nan, np.float32)

    def _do_block(blk):
        """Slice rows for a block and fit them (worker-side). Pre-slicing the
        CSR rows here keeps the whole vectors object off the wire when a client
        scatters the closure — only this block's small rows travel."""
        y0, y1, x0, x1 = blk
        rows_list = _slice_chunk_rows(vectors, y0, y1, x0, x1, t)
        return blk, _vector_chunk(
            rows_list, lib, P, warm_start, (y0, x0), (y1 - y0, x1 - x0),
            shm_name, (ny, nx))

    tic = time.time()
    total = len(blocks)
    done = 0
    if client is not None:
        # Scatter the (small) library once; submit one task per nav chunk so
        # every worker fits chunks concurrently (vs the serial single-core
        # path). Collect as they finish for live progress.
        from distributed import as_completed as _as_completed
        try:
            lib_f = client.scatter(lib, broadcast=True)
            vecs_f = client.scatter(vectors, broadcast=True)
        except Exception:
            lib_f, vecs_f = lib, vectors

        def _remote_block(blk, _vectors, _lib):
            y0, y1, x0, x1 = blk
            rows_list = _slice_chunk_rows(_vectors, y0, y1, x0, x1, t)
            return blk, _vector_chunk(
                rows_list, _lib, P, warm_start, (y0, x0),
                (y1 - y0, x1 - x0), shm_name, (ny, nx))

        futures = [client.submit(_remote_block, blk, vecs_f, lib_f,
                                 pure=False) for blk in blocks]
        for fut in _as_completed(futures):
            if stopped_flag is not None and stopped_flag[0]:
                for f in futures:
                    try:
                        f.cancel()
                    except Exception as e:
                        log.debug("cancelling vector-orientation future failed: %s", e)
                return None
            blk, pk = fut.result()
            y0, y1, x0, x1 = blk
            packed[y0:y1, x0:x1, :] = pk
            done += 1
            if progress is not None:
                progress(done, total)
    else:
        # Local fallback: a thread pool over chunks (tests / no cluster).
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor() as ex:
            futs = [ex.submit(_do_block, blk) for blk in blocks]
            for fut in as_completed(futs):
                if stopped_flag is not None and stopped_flag[0]:
                    return None
                blk, pk = fut.result()
                y0, y1, x0, x1 = blk
                packed[y0:y1, x0:x1, :] = pk
                done += 1
                if progress is not None:
                    progress(done, total)

    if stopped_flag is not None and stopped_flag[0]:
        return None
    print(f"[vector-orientation] matched {ny * nx} patterns in "
          f"{time.time() - tic:.1f} s")

    # Unpack the (ny, nx, 12) assembly.
    quats = np.nan_to_num(packed[..., 0:4], nan=0.0).astype(np.float32)
    quats[(quats == 0).all(-1)] = (1.0, 0.0, 0.0, 0.0)
    coarse = np.nan_to_num(packed[..., 4]).astype(np.float32)
    phase_idx = np.nan_to_num(packed[..., 5]).astype(np.int16)
    strain = packed[..., 6:9].astype(np.float32)
    residual = packed[..., 9].astype(np.float32)
    friedel = packed[..., 10].astype(np.float32)
    n_matched = np.nan_to_num(packed[..., 11]).astype(np.int16)
    return VectorOrientationResult(
        quats=quats, phase_idx=phase_idx,
        theta=np.zeros((ny, nx), np.float32), strain=strain,
        residual=residual, friedel_asym=friedel,
        n_matched=n_matched, coarse_score=coarse,
        phases_meta=lib.phases_meta, nav_shape=(ny, nx), params=dict(P),
    )
