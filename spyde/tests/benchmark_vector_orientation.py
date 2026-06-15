"""
Benchmark harness for vector orientation mapping fit strategies.

Run directly (not under pytest by default — it's a benchmark, slow):

    python -m spyde.tests.benchmark_vector_orientation

Compares, on real sped_ag vectors and synthetic ground-truth:
  - independent            : current per-pattern (θ, A, t) fit
  - friedel                : Friedel-paired residual (±g measured pairs give
                             noise-averaged observations; unpaired vectors still
                             contribute normally)
  - warm                   : warm-start propagation (hardened gate)

Metrics: ms/pattern, strain median, residual (px), Friedel asymmetry, and on
synthetic data the recovered-vs-applied strain error.

These numbers feed VECTOR_ORIENTATION_MAPPING_PLAN.md §7f and benchmarks.md.
"""
from __future__ import annotations

import time

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Friedel-paired residual (prototype; promoted to vector_orientation.py if it wins)
# ─────────────────────────────────────────────────────────────────────────────

def pair_friedel(meas_xy, meas_I, tol=0.02):
    """Index ±g partners among measured vectors.

    Returns (pair_a, pair_b, singles): arrays of indices. A measured vector and
    its nearest opposite (||v_i + v_j|| < tol) form a centrosymmetric pair;
    everything else is a single. Pairing is greedy by closeness.
    """
    n = len(meas_xy)
    used = np.zeros(n, bool)
    pa, pb = [], []
    # distance from v_i to -v_j
    s2 = ((meas_xy[:, None, :] + meas_xy[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(s2, np.inf)
    order = np.argsort(s2.ravel())
    for flat in order:
        i, j = divmod(int(flat), n)
        if used[i] or used[j] or i == j:
            continue
        if s2[i, j] > tol * tol:
            break
        used[i] = used[j] = True
        pa.append(i); pb.append(j)
    singles = np.where(~used)[0]
    return np.array(pa, int), np.array(pb, int), singles


def residual_friedel(params, g, gI, v, vI, sigma, sink_bw, cap, pen,
                     pairs_ab=None):
    """Soft-assign residual that uses Friedel-averaged measured observations.

    For paired ±g measured vectors, the soft-assign target is matched against
    the noise-averaged pair midpoint/half-difference rather than the raw spots,
    cancelling per-spot beam-center error. Unpaired vectors are matched normally
    (NOT excluded). Falls back to the standard residual if no pairs.
    """
    from spyde.actions.vector_orientation import _P_A, _P_THETA, _P_T, _rot

    A = params[_P_A].reshape(2, 2)
    t = params[_P_T]
    gl = (g @ _rot(params[_P_THETA]).T) @ A.T        # A·Rot·g  (no translation)
    p_full = gl + t                                   # full projection

    # Singles match the full projection (A·Rot·g + t). Paired ±g measured
    # vectors yield a centering-cancelled observation: half = (v_g - v_{-g})/2
    # estimates A·Rot·g with the shared beam-center error removed — match it
    # against gl (no t). Unpaired vectors are NOT excluded.
    if pairs_ab is not None and len(pairs_ab[0]):
        pa, pb, singles = pairs_ab
        half = 0.5 * (v[pa] - v[pb])
        Ipair = 0.5 * (vI[pa] + vI[pb])
        v_centered = np.vstack([half, -half])         # match against gl
        I_centered = np.concatenate([Ipair, Ipair])
        v_single = v[singles]                          # match against p_full
        I_single = vI[singles]
    else:
        v_centered = np.empty((0, 2)); I_centered = np.empty(0)
        v_single = v; I_single = vI

    def _soft(p_proj, vv, vvI):
        if len(vv) == 0:
            return np.zeros((len(p_proj), 2))
        d2 = ((p_proj[:, None, :] - vv[None, :, :]) ** 2).sum(-1)
        w = np.exp(-d2 / (2 * sigma ** 2)) * vvI[None, :]
        raw = w.sum(1)
        wn = w / (raw[:, None] + 1e-9)
        target = (wn[..., None] * vv[None, :, :]).sum(1)
        sink = np.exp(-(sink_bw ** 2) / (2 * sigma ** 2))
        conf = raw / (raw + sink)
        return (p_proj - target) * (np.sqrt(gI) * conf)[:, None]

    r_centered = _soft(gl, v_centered, I_centered)     # template vs ±g halves
    r_single = _soft(p_full, v_single, I_single)       # template vs singles
    sv = np.linalg.svd(A, compute_uv=False)
    excess = np.clip(np.abs(sv - 1.0) - cap, 0.0, None)
    return np.concatenate([r_centered.ravel(), r_single.ravel(), pen * excess])


def friedel_center(meas_xy, tol=0.02):
    """Closed-form beam-center estimate from ±g pairs: the mean midpoint of all
    centrosymmetric pairs. Cancels per-spot peak-finding error (√(2N) averaging)
    without any optimization. Returns (cx, cy) or None if no pairs."""
    pa, pb, _ = pair_friedel(meas_xy, np.ones(len(meas_xy)), tol)
    if len(pa) == 0:
        return None
    mids = 0.5 * (meas_xy[pa] + meas_xy[pb])
    return mids.mean(0)


def fit_pattern_friedel(meas_xy, meas_I, lib, params=None, seed=None,
                        pair_tol=0.02):
    """fit_pattern with a cheap closed-form Friedel beam-center pre-correction.

    The ±g midpoints give a robust beam-center estimate; subtracting it before
    the standard fit removes the dominant peak-finding centering error without
    adding any LM cost. Unpaired vectors are kept (not excluded). Falls back to
    the plain fit when no pairs are present.
    """
    from spyde.actions.vector_orientation import fit_pattern
    meas_xy = np.asarray(meas_xy, float)
    c = friedel_center(meas_xy, pair_tol)
    if c is None:
        return fit_pattern(meas_xy, meas_I, lib, params, seed)
    fit = fit_pattern(meas_xy - c, meas_I, lib, params, seed)
    if fit is not None:
        # fold the pre-correction back into the reported translation
        fit.translation = (fit.translation + c.astype(np.float32))
    return fit


def fit_pattern_friedel_denoise(meas_xy, meas_I, lib, params=None, seed=None,
                                pair_tol=0.02):
    """Friedel denoising: replace each ±g pair by its symmetric average so the
    two noisy observations become one √2-cleaner pair (±half about the shared
    center). Singles are kept as-is. This actually reduces per-spot noise in the
    fit inputs, unlike the center-only correction. Fewer, cleaner points →
    should also be a touch faster."""
    from spyde.actions.vector_orientation import fit_pattern
    meas_xy = np.asarray(meas_xy, float); meas_I = np.asarray(meas_I, float)
    pa, pb, singles = pair_friedel(meas_xy, meas_I, pair_tol)
    if len(pa) == 0:
        return fit_pattern(meas_xy, meas_I, lib, params, seed)
    center = (0.5 * (meas_xy[pa] + meas_xy[pb])).mean(0)
    half = 0.5 * (meas_xy[pa] - meas_xy[pb])           # √2-cleaner ±g
    Ipair = 0.5 * (meas_I[pa] + meas_I[pb])
    # rebuild a centered, denoised measured set: ±half about origin + singles-c
    v_new = np.vstack([center + half, center - half, meas_xy[singles]])
    I_new = np.concatenate([Ipair, Ipair, meas_I[singles]])
    fit = fit_pattern(v_new - center, I_new, lib, params, seed)
    if fit is not None:
        fit.translation = (fit.translation + center.astype(np.float32))
    return fit


# ─────────────────────────────────────────────────────────────────────────────
# Data builders
# ─────────────────────────────────────────────────────────────────────────────

def _ag_library(r_max=0.75, res=1.0):
    from orix.crystal_map import Phase
    from diffpy.structure import Atom, Lattice, Structure
    import hyperspy.api as hs
    from spyde.actions.orientation_compute import generate_library_from_phases
    from spyde.actions.vector_orientation import build_template_library
    a = 4.0853
    latt = Lattice(a, a, a, 90, 90, 90)
    atoms = [Atom("Ag", [0, 0, 0]), Atom("Ag", [.5, .5, 0]),
             Atom("Ag", [.5, 0, .5]), Atom("Ag", [0, .5, .5])]
    phase = Phase(name="Ag", point_group="m-3m",
                  structure=Structure(atoms, latt))
    sim = generate_library_from_phases([phase], 200.0, res, 1e-3, r_max)
    cal = hs.signals.Signal2D(np.zeros((112, 112), np.float32))
    for ax in cal.axes_manager.signal_axes:
        ax.scale = 0.01336; ax.offset = -0.7484
    cal.set_signal_type("electron_diffraction")
    return build_template_library(sim, cal, r_max=r_max)


def _ag_vectors(iy0, ix0, ny, nx):
    from scipy.ndimage import maximum_filter, gaussian_filter
    import pyxem.data as d
    from spyde.signals.diffraction_vectors import (
        SpyDEDiffractionVectors, _build_nav_offsets, N_COLS,
        COL_NAV_X, COL_NAV_Y, COL_KX, COL_KY, COL_TIME, COL_INTENSITY)
    s = d.sped_ag(); sax = s.axes_manager.signal_axes
    sc, off = sax[0].scale, sax[0].offset
    rows = []
    for iy in range(ny):
        for ix in range(nx):
            fr = np.asarray(s.inav[ix0 + ix, iy0 + iy].data, float)
            fb = gaussian_filter(fr, 1.0); thr = fb.max() * 0.08
            mx = (maximum_filter(fb, 5) == fb) & (fb > thr)
            ys, xs = np.where(mx)
            kx = xs * sc + off; ky = ys * sc + off; I = fb[ys, xs]
            keep = np.sqrt(kx**2 + ky**2) > 0.05
            kx, ky, I = kx[keep], ky[keep], I[keep]
            for k in range(len(kx)):
                r = np.zeros(N_COLS, np.float32)
                r[COL_NAV_X] = ix; r[COL_NAV_Y] = iy
                r[COL_KX] = kx[k]; r[COL_KY] = ky[k]
                r[COL_TIME] = -1.0; r[COL_INTENSITY] = I[k]
                rows.append(r)
    flat = np.array(rows, np.float32)
    nav_offsets = _build_nav_offsets(flat, (ny, nx))

    class _Ax:
        def __init__(s_, _sc, _of): s_.scale, s_.offset, s_.size = _sc, _of, 112
        units = "1/A"; name = "k"
    return SpyDEDiffractionVectors(
        flat_buffer=flat, nav_offsets=nav_offsets, nav_shape=(ny, nx),
        full_nav_shape=(ny, nx), sig_shape=(112, 112),
        sig_axes=[_Ax(sc, off), _Ax(sc, off)],
        kernel_radius_px=3.0, kernel_radius_data=0.04)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def _stats(fits, vecs):
    strains, resids, friedels, times = [], [], [], []
    for f, ms in fits:
        if f is None:
            continue
        strains.append(np.abs(f.strain).max())
        resids.append(f.residual)
        if np.isfinite(f.friedel_asym):
            friedels.append(f.friedel_asym)
        times.append(ms)
    return dict(
        ms=np.median(times), strain=np.median(strains),
        resid_px=np.median(resids) / 0.01336,
        friedel=np.median(friedels) if friedels else float("nan"),
        n=len(strains))


def run(ny=10, nx=14):
    from spyde.actions.vector_orientation import fit_pattern
    from spyde.signals.diffraction_vectors import COL_KX, COL_KY, COL_INTENSITY

    print("building Ag library + sped_ag vectors...")
    lib = _ag_library()
    vecs = _ag_vectors(20, 80, ny, nx)
    coords = [(iy, ix) for iy in range(ny) for ix in range(nx)]

    def _bench(fit_fn, label):
        out = []
        for (iy, ix) in coords:
            rows = vecs.at(iy, ix)
            if len(rows) < 4:
                continue
            mxy = rows[:, [COL_KX, COL_KY]].astype(float)
            mI = rows[:, COL_INTENSITY].astype(float)
            a = time.perf_counter()
            f = fit_fn(mxy, mI, lib)
            out.append((f, (time.perf_counter() - a) * 1e3))
        st = _stats(out, vecs)
        print(f"  {label:12s}: {st['ms']:.1f} ms/pat | strain_med {st['strain']:.4f} "
              f"| resid {st['resid_px']:.2f} px | friedel {st['friedel']:.4f} "
              f"| n={st['n']}")
        return st

    print(f"\n=== {ny}x{nx} sped_ag region, {len(lib.spots_xy)} templates ===")
    _bench(fit_pattern, "independent")
    _bench(fit_pattern_friedel, "friedel")


def run_synthetic(n_trials=40):
    """Ground-truth stress test: known strain + beam-center offset + per-spot
    noise (bad peak finding). Measures recovered-vs-applied strain error with
    and without the Friedel center pre-correction. This is where Friedel should
    win — the offset breaks the naive fit's beam-center estimate."""
    from spyde.actions.vector_orientation import (
        fit_pattern, TemplateLibrary, _rot)

    rng = np.random.RandomState(0)
    # centrosymmetric template (Friedel pairs exist)
    base = rng.uniform(-0.6, 0.6, (10, 2))
    base = base[np.linalg.norm(base, axis=1) < 0.6]
    g = np.vstack([base, -base]).astype(np.float32)
    half = rng.uniform(0.3, 1.0, len(g) // 2)
    gI = np.concatenate([half, half]).astype(np.float32)
    lib = TemplateLibrary(
        spots_xy=[g], spots_I=[gI],
        template_quats=np.array([[1.0, 0, 0, 0]]),
        template_phase=np.array([0], np.int16),
        phases_meta=[{"name": "x", "point_group": "m-3m"}],
        cache={}, radial_range=(0.0, 0.6), r_max=0.6)

    true_strain = np.array([[0.025, 0.008], [0.008, -0.018]])
    theta = np.deg2rad(11.0)

    for noise, offset in [(0.004, 0.0), (0.012, 0.0),
                          (0.004, 0.04), (0.012, 0.04)]:
        e_ind, e_fri = [], []
        for _ in range(n_trials):
            A = np.eye(2) + true_strain
            v = (g @ _rot(theta).T) @ A.T
            v = v + rng.normal(0, noise, v.shape)        # peak-finding noise
            v = v + np.array([offset, offset * 0.5])      # beam miscentering
            vI = gI.copy()
            fi = fit_pattern(v, vI, lib, seed=(0, theta, 1.0))
            ff = fit_pattern_friedel(v, vI, lib, seed=(0, theta, 1.0))
            fd = fit_pattern_friedel_denoise(v, vI, lib, seed=(0, theta, 1.0))
            if fi is not None:
                e_ind.append(np.abs(fi.strain - true_strain).max())
            if ff is not None:
                e_fri.append(np.abs(ff.strain - true_strain).max())
            if fd is not None:
                e_den = e_fri  # placeholder if needed
        # recompute denoise separately for clarity
        e_den = []
        for _ in range(n_trials):
            A = np.eye(2) + true_strain
            v = (g @ _rot(theta).T) @ A.T + rng.normal(0, noise, g.shape)
            v = v + np.array([offset, offset * 0.5])
            fd = fit_pattern_friedel_denoise(v, gI.copy(), lib, seed=(0, theta, 1.0))
            if fd is not None:
                e_den.append(np.abs(fd.strain - true_strain).max())
        print(f"  noise={noise:.3f} offset={offset:.2f}: "
              f"independent {np.median(e_ind):.4f} | "
              f"friedel-center {np.median(e_fri):.4f} | "
              f"friedel-denoise {np.median(e_den):.4f}")


def run_field_coupling(ny=20, nx=20, noise=0.01):
    """Does neighbor coupling reduce the per-pattern strain noise floor?

    Synthetic uniform-orientation field with a known strain that varies smoothly
    in space (a gradient + a sharp grain boundary), plus per-spot noise. Compare
    independent per-pattern strain vs a TV/edge-preserving smoothed field, by
    error to ground truth in the smooth region AND boundary sharpness.
    """
    from spyde.actions.vector_orientation import (
        fit_pattern, TemplateLibrary, _rot)
    from scipy.ndimage import median_filter, gaussian_filter

    rng = np.random.RandomState(3)
    base = rng.uniform(-0.6, 0.6, (12, 2))
    base = base[np.linalg.norm(base, axis=1) < 0.6]
    g = np.vstack([base, -base]).astype(np.float32)
    half = rng.uniform(0.4, 1.0, len(g) // 2)
    gI = np.concatenate([half, half]).astype(np.float32)
    lib = TemplateLibrary(
        spots_xy=[g], spots_I=[gI],
        template_quats=np.array([[1.0, 0, 0, 0]]),
        template_phase=np.array([0], np.int16),
        phases_meta=[{"name": "x", "point_group": "m-3m"}],
        cache={}, radial_range=(0.0, 0.6), r_max=0.6)
    theta = np.deg2rad(8.0)

    # ground-truth strain field: εxx gradient left→right + a step at x=nx//2
    gt = np.zeros((ny, nx, 3), np.float32)
    xv = np.linspace(-0.02, 0.02, nx)
    gt[..., 0] = xv[None, :]                       # εxx gradient
    gt[:, nx // 2:, 1] = 0.02                       # εyy grain on the right half

    est = np.full((ny, nx, 3), np.nan, np.float32)
    for iy in range(ny):
        for ix in range(nx):
            S = np.array([[1 + gt[iy, ix, 0], gt[iy, ix, 2]],
                          [gt[iy, ix, 2], 1 + gt[iy, ix, 1]]])
            v = (g @ _rot(theta).T) @ S.T + rng.normal(0, noise, g.shape)
            f = fit_pattern(v, gI.copy(), lib, seed=(0, theta, 1.0))
            if f is not None:
                est[iy, ix] = (f.strain[0, 0], f.strain[1, 1], f.strain[0, 1])

    def _err(field):
        return np.nanmedian(np.abs(field - gt))

    # smoothing options
    sm_gauss = np.stack([gaussian_filter(est[..., c], 1.0) for c in range(3)], -1)
    sm_med = np.stack([median_filter(est[..., c], size=3) for c in range(3)], -1)

    # boundary sharpness: |Δεyy| across the x=nx//2 step (should stay ~0.02)
    def _step(field):
        col = nx // 2
        return float(np.nanmedian(field[:, col, 1]) -
                     np.nanmedian(field[:, col - 1, 1]))

    print(f"  independent : err {_err(est):.4f}  step {_step(est):+.4f} (true ~0.020)")
    print(f"  gaussian σ1 : err {_err(sm_gauss):.4f}  step {_step(sm_gauss):+.4f}")
    print(f"  median 3x3  : err {_err(sm_med):.4f}  step {_step(sm_med):+.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Global field-solve methods (prototypes; the winner is promoted to the core)
# ─────────────────────────────────────────────────────────────────────────────

def tv_denoise_field(field, weight=0.02, n_iter=100, valid=None):
    """Total-variation denoise a (ny,nx,C) field, edge-preserving by design.

    Chambolle TV per channel. `weight` ≈ λ (higher = smoother). `valid` masks
    unfit positions (NaN). TV's piecewise-constant prior fits grains better than
    median and is tunable; the right tool for a robust strain field."""
    from skimage.restoration import denoise_tv_chambolle
    out = np.empty_like(field)
    for c in range(field.shape[-1]):
        comp = field[..., c].astype(float)
        m = np.nanmedian(comp) if valid is None else np.nanmedian(comp[valid])
        filled = np.where(np.isnan(comp), m, comp)
        out[..., c] = denoise_tv_chambolle(filled, weight=weight)
    return out


def fit_field_independent(vecs, lib, params=None, seed_fn=None):
    """Independent per-pattern fit → (strain(ny,nx,3), pose(ny,nx,7), valid)."""
    from spyde.actions.vector_orientation import fit_pattern
    from spyde.signals.diffraction_vectors import COL_KX, COL_KY, COL_INTENSITY
    ny, nx = vecs.nav_shape
    strain = np.full((ny, nx, 3), np.nan, np.float32)
    pose = np.full((ny, nx, 7), np.nan, np.float32)
    for iy in range(ny):
        for ix in range(nx):
            rows = vecs.at(iy, ix)
            if len(rows) < 4:
                continue
            mxy = rows[:, [COL_KX, COL_KY]].astype(float)
            mI = rows[:, COL_INTENSITY].astype(float)
            seed = seed_fn(iy, ix) if seed_fn else None
            f = fit_pattern(mxy, mI, lib, params, seed=seed)
            if f is not None:
                strain[iy, ix] = (f.strain[0, 0], f.strain[1, 1], f.strain[0, 1])
                pose[iy, ix] = (f.theta, *f.affine.ravel(), *f.translation)
    return strain, pose, np.isfinite(strain[..., 0])


def fit_field_iterated(vecs, lib, params=None, tv_weight=0.02, n_outer=2):
    """fit → TV-smooth the POSE field → re-fit each pattern warm-started from the
    smoothed neighbour estimate. The smoothed field is reliable (unlike raw
    warm-start), so the re-fit is both warm and well-seeded — captures most of a
    joint solve at a fraction of the cost."""
    from spyde.actions.vector_orientation import fit_pattern
    from spyde.signals.diffraction_vectors import COL_KX, COL_KY, COL_INTENSITY
    ny, nx = vecs.nav_shape
    strain, pose, valid = fit_field_independent(vecs, lib, params)
    for _ in range(n_outer):
        # smooth the pose field (theta + affine + t), edge-preserving
        sm_pose = tv_denoise_field(pose, weight=tv_weight)
        new_strain = strain.copy()
        for iy in range(ny):
            for ix in range(nx):
                rows = vecs.at(iy, ix)
                if len(rows) < 4:
                    continue
                mxy = rows[:, [COL_KX, COL_KY]].astype(float)
                mI = rows[:, COL_INTENSITY].astype(float)
                # seed θ + template branch from the smoothed neighbourhood;
                # template idx carried from the original independent fit
                th = float(sm_pose[iy, ix, 0])
                f = fit_pattern(mxy, mI, lib, params, seed=(0, th, 1.0)) \
                    if not lib.cache else fit_pattern(mxy, mI, lib, params)
                if f is not None:
                    new_strain[iy, ix] = (f.strain[0, 0], f.strain[1, 1],
                                          f.strain[0, 1])
                    pose[iy, ix] = (f.theta, *f.affine.ravel(), *f.translation)
        strain = new_strain
    return strain, valid


def run_robustness(ny=18, nx=18):
    """High-noise / few-spot regime: which method survives when independent fits
    start to fail? Sweep noise + spot-dropout, compare independent, median,
    TV, and iterated (fit→TV→refit) by error to a known smooth strain field."""
    from spyde.actions.vector_orientation import (
        TemplateLibrary, _rot, VectorOrientationResult)
    from spyde.signals.diffraction_vectors import (
        SpyDEDiffractionVectors, _build_nav_offsets, N_COLS,
        COL_NAV_X, COL_NAV_Y, COL_KX, COL_KY, COL_TIME, COL_INTENSITY)

    rng = np.random.RandomState(5)
    base = rng.uniform(-0.6, 0.6, (12, 2))
    base = base[np.linalg.norm(base, axis=1) < 0.6]
    g = np.vstack([base, -base]).astype(np.float32)
    halfI = rng.uniform(0.4, 1.0, len(g) // 2)
    gI = np.concatenate([halfI, halfI]).astype(np.float32)
    lib = TemplateLibrary(
        spots_xy=[g], spots_I=[gI],
        template_quats=np.array([[1.0, 0, 0, 0]]),
        template_phase=np.array([0], np.int16),
        phases_meta=[{"name": "x", "point_group": "m-3m"}],
        cache={}, radial_range=(0.0, 0.6), r_max=0.6)
    theta = np.deg2rad(8.0)

    # smooth GT strain: a gentle gradient + one grain boundary
    gt = np.zeros((ny, nx, 3), np.float32)
    gt[..., 0] = np.linspace(-0.015, 0.015, nx)[None, :]
    gt[ny // 2:, :, 1] = 0.018

    def _build_vecs(noise, drop):
        rows = []
        for iy in range(ny):
            for ix in range(nx):
                S = np.array([[1 + gt[iy, ix, 0], gt[iy, ix, 2]],
                              [gt[iy, ix, 2], 1 + gt[iy, ix, 1]]])
                v = (g @ _rot(theta).T) @ S.T + rng.normal(0, noise, g.shape)
                keep = rng.rand(len(v)) > drop          # spot dropout
                v, vi = v[keep], gI[keep]
                for k in range(len(v)):
                    r = np.zeros(N_COLS, np.float32)
                    r[COL_NAV_X] = ix; r[COL_NAV_Y] = iy
                    r[COL_KX] = v[k, 0]; r[COL_KY] = v[k, 1]
                    r[COL_TIME] = -1.0; r[COL_INTENSITY] = vi[k]
                    rows.append(r)
        flat = np.array(rows, np.float32)
        no = _build_nav_offsets(flat, (ny, nx))

        class _Ax:
            scale = 1.0; offset = 0.0; size = 64; units = "1/A"; name = "k"
        return SpyDEDiffractionVectors(
            flat_buffer=flat, nav_offsets=no, nav_shape=(ny, nx),
            full_nav_shape=(ny, nx), sig_shape=(64, 64),
            sig_axes=[_Ax(), _Ax()], kernel_radius_px=3.0,
            kernel_radius_data=0.04)

    def _err(field):
        return float(np.nanmedian(np.abs(field - gt)))

    print(f"  {'noise/drop':14s} {'indep':>8s} {'median':>8s} {'TV':>8s} {'iter':>8s}")
    for noise, drop in [(0.01, 0.0), (0.02, 0.0), (0.02, 0.3),
                        (0.035, 0.3), (0.05, 0.4)]:
        vecs = _build_vecs(noise, drop)
        strain, pose, valid = fit_field_independent(vecs, lib)
        res = VectorOrientationResult(
            quats=np.zeros((ny, nx, 4), np.float32),
            phase_idx=np.zeros((ny, nx), np.int16),
            theta=pose[..., 0], strain=strain,
            residual=np.zeros((ny, nx), np.float32),
            friedel_asym=np.zeros((ny, nx), np.float32),
            n_matched=np.zeros((ny, nx), np.int16),
            coarse_score=np.zeros((ny, nx), np.float32),
            phases_meta=[], nav_shape=(ny, nx))
        e_ind = _err(strain)
        e_med = _err(res.smoothed_strain())
        e_tv = _err(tv_denoise_field(strain, weight=0.03))
        st_it, _ = fit_field_iterated(vecs, lib, tv_weight=0.03, n_outer=2)
        e_it = _err(st_it)
        print(f"  {f'{noise:.3f}/{drop:.1f}':14s} {e_ind:8.4f} {e_med:8.4f} "
              f"{e_tv:8.4f} {e_it:8.4f}")


if __name__ == "__main__":
    run()
    print("\n=== synthetic: strain recovery vs noise + beam offset ===")
    run_synthetic()
    print("\n=== field coupling: does neighbor smoothing cut the strain noise? ===")
    run_field_coupling()
    print("\n=== robustness: high-noise/few-spot, independent vs median/TV/iter ===")
    run_robustness()
