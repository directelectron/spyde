"""
vector_orientation_gpu.py — batched GPU fit of the whole vector-OM field at once.

The vectors and the template library are tiny (a few MB), and every nav position
runs the *same* differentiable soft-assign cost — so the entire scan is one
batched optimisation, not a per-pattern loop. We pack all P patterns onto the
GPU and:

  1. **coarse seed (batched)** — score every (pattern, template, in-plane-angle)
     combination with the soft-assign overlap in a few big tensor ops; argmax
     gives each pattern's orientation branch + seed angle. No per-pattern Python,
     no pyxem matcher, no dask.
  2. **refine (batched)** — per-pattern pose (theta, log-strain, beam-shift)
     optimised together with Adam over a sigma-anneal schedule. The 2x2 affine is
     parametrised as Rot(theta) . exp(symmetric strain) so the strain bound is
     by-construction (no penalty term) and the rotation/strain split is clean.

This replaces the serial/dask `compute_vector_orientation*` for the common case;
falls back to the CPU path when torch+CUDA is unavailable. Same result container.

The cost mirrors `vector_orientation._residual` exactly (template->measured
soft-assign + no-match sink, intensity weighted) so GPU and CPU agree.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from spyde.signals.diffraction_vectors import COL_KX, COL_KY, COL_INTENSITY

log = logging.getLogger(__name__)
from spyde.actions.vector_orientation import (
    DEFAULTS, TemplateLibrary, VectorOrientationResult,
)

import warnings as _warnings
# torch's MPS irfft reuses an internal out-tensor and emits a benign deprecation
# UserWarning ("An output with one or more elements was resized…") — the result
# is correct; silence it so it doesn't spam the console on every coarse seed.
_warnings.filterwarnings(
    "ignore", message=".*output with one or more elements was resized.*")


def select_device():
    """Best available torch device: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU.

    torch gives one batched/autograd API across all three, so the same fit runs
    GPU-accelerated on Windows/Linux+CUDA and on Apple-Silicon Macs via Metal,
    and still-batched (far faster than the serial scipy path) on plain CPU.
    Returns a torch.device or None if torch isn't importable.
    """
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except Exception:
        return None


def gpu_available() -> bool:
    """True when a hardware-accelerated torch device (CUDA or Apple-MPS) exists.

    CPU-only torch returns False here — the batched torch path still *runs* on
    CPU (and is used as the fast fallback), but this gate is what the UI uses to
    decide whether to expect interactive speed.
    """
    dev = select_device()
    return dev is not None and dev.type in ("cuda", "mps")


def torch_available() -> bool:
    """True when the batched torch path can run at all (any device, incl. CPU)."""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def gpu_unavailable_reason() -> str:
    """Human-readable reason the accelerated path is off — for surfacing in the
    UI/logs so a silent fall-through to the slow CPU fit is never a mystery."""
    try:
        import torch
    except Exception as e:
        return f"torch not importable ({e})"
    try:
        if torch.cuda.is_available():
            return "CUDA available"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "Apple MPS available"
        return ("torch present but no CUDA/MPS device "
                f"(torch {torch.__version__}; CPU-only build or no GPU visible)")
    except Exception as e:
        return f"device check raised: {e}"


_AUTOGRAD_WARMED = False


def warmup_autograd() -> None:
    """Initialise the CUDA autograd engine on the *calling* thread.

    torch's CUDA autograd backward segfaults on Windows the first time it runs
    on a thread whose engine hasn't been initialised. The GPU compute runs on a
    daemon worker, so we run one trivial backward on the GUI/main thread first
    (idempotent) — afterwards the worker thread's backward is safe.
    """
    global _AUTOGRAD_WARMED
    if _AUTOGRAD_WARMED:
        return
    try:
        import torch
        if torch.cuda.is_available():
            x = torch.zeros(1, device="cuda", requires_grad=True)
            (x * 2).sum().backward()
        _AUTOGRAD_WARMED = True
    except Exception as e:
        log.debug("CUDA autograd warmup skipped: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Packing
# ─────────────────────────────────────────────────────────────────────────────

def _pack_patterns(vectors, t, device, dtype):
    """All patterns → padded (P, Vmax, 2) coords + (P, Vmax) intensity + mask.

    P = ny*nx in scan order. Patterns with <3 vectors get an all-False mask row
    and are skipped in decode. Pure CSR slicing, then one host→GPU transfer.
    """
    import torch
    ny, nx = vectors.nav_shape
    use_t = t is not None and vectors.n_time > 0
    rows_list = []
    counts = []
    for iy in range(ny):
        for ix in range(nx):
            rows = (vectors.at_t(iy, ix, t) if use_t else vectors.at(iy, ix))
            rows_list.append(rows)
            counts.append(len(rows))
    P = len(rows_list)
    Vmax = max(counts) if counts else 0
    Vmax = max(Vmax, 1)
    xy = np.zeros((P, Vmax, 2), np.float32)
    inten = np.zeros((P, Vmax), np.float32)
    mask = np.zeros((P, Vmax), bool)
    for i, rows in enumerate(rows_list):
        n = len(rows)
        if n:
            xy[i, :n, 0] = rows[:, COL_KX]
            xy[i, :n, 1] = rows[:, COL_KY]
            inten[i, :n] = rows[:, COL_INTENSITY]
            mask[i, :n] = True
    # Per-pattern unit-mean normalisation — the stored intensity is RAW image
    # counts (for virtual imaging); the soft-assign sink gating expects O(1)
    # weights. Mirrors the CPU fit_pattern normalisation so CPU/GPU agree.
    pcnt = mask.sum(axis=1).astype(np.float32)            # (P,)
    pmean = inten.sum(axis=1) / np.maximum(pcnt, 1.0)     # mean over real rows
    inten = inten / np.maximum(pmean, 1e-12)[:, None]
    return (
        torch.as_tensor(xy, device=device, dtype=dtype),
        torch.as_tensor(inten, device=device, dtype=dtype),
        torch.as_tensor(mask, device=device),
        np.asarray(counts, np.int32),
    )


def _pack_templates(lib: TemplateLibrary, device, dtype):
    """Library spots → padded (T, Mmax, 2) + (T, Mmax) intensity + mask."""
    import torch
    T = len(lib.spots_xy)
    Mmax = max((len(s) for s in lib.spots_xy), default=1)
    Mmax = max(Mmax, 1)
    g = np.zeros((T, Mmax, 2), np.float32)
    gI = np.zeros((T, Mmax), np.float32)
    gmask = np.zeros((T, Mmax), bool)
    for ti in range(T):
        s = lib.spots_xy[ti]
        m = len(s)
        if m:
            g[ti, :m] = s
            gI[ti, :m] = lib.spots_I[ti]
            gmask[ti, :m] = True
    return (
        torch.as_tensor(g, device=device, dtype=dtype),
        torch.as_tensor(gI, device=device, dtype=dtype),
        torch.as_tensor(gmask, device=device),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batched cost (template → measured soft-assign + no-match sink)
# ─────────────────────────────────────────────────────────────────────────────

def _rot_mat(theta):
    """theta (P,) → (P, 2, 2) rotation matrices."""
    import torch
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s], -1),
                        torch.stack([s, c], -1)], -2)


def _affine_from_logstrain(eps, cap):
    """eps (P,3) [exx,eyy,exy] (pre-clamp, in log space) → (P,2,2) symmetric
    stretch S = expm(E) with |strain| bounded by cap via tanh. Symmetric
    positive-definite by construction, so no reflection / collapse possible."""
    import torch
    b = cap * torch.tanh(eps)             # bounded symmetric-strain components
    exx, eyy, exy = b[:, 0], b[:, 1], b[:, 2]
    # 2x2 symmetric E = [[exx,exy],[exy,eyy]]; S = I + E (small-strain) is fine
    # within +-cap; use I+E (cheaper than expm and within bound stays SPD).
    P = eps.shape[0]
    S = torch.zeros(P, 2, 2, device=eps.device, dtype=eps.dtype)
    S[:, 0, 0] = 1.0 + exx
    S[:, 1, 1] = 1.0 + eyy
    S[:, 0, 1] = exy
    S[:, 1, 0] = exy
    return S, torch.stack([exx, eyy, exy], -1)


def _batched_cost(theta, eps, tvec, g, gI, gmask, v, vI, vmask,
                  sigma, sink_bw, cap):
    """Per-pattern soft-assign + sink cost. Returns (P,) cost.

    Mirrors vector_orientation._residual: each template spot is pulled to its
    intensity-weighted soft-nearest MEASURED vector; the sink lets unmatched
    template spots opt out. Summed (squared, gI*conf weighted) over template
    spots per pattern.
    """
    import torch
    S, _ = _affine_from_logstrain(eps, cap)               # (P,2,2)
    R = _rot_mat(theta)                                    # (P,2,2)
    M = torch.bmm(S, R)                                    # A = S R  (P,2,2)
    # project template spots: p = g @ M^T + t   → (P, Mg, 2)
    p = torch.einsum("pmd,pcd->pmc", g, M) + tvec[:, None, :]
    # distances template→measured: (P, Mg, Vv)
    d2 = ((p[:, :, None, :] - v[:, None, :, :]) ** 2).sum(-1)
    big = torch.tensor(1e6, device=d2.device, dtype=d2.dtype)
    d2 = d2 + (~vmask[:, None, :]) * big                   # ignore pad vectors
    w = torch.exp(-d2 / (2 * sigma * sigma)) * vI[:, None, :]
    raw = w.sum(-1)                                        # (P, Mg)
    wn = w / (raw[..., None] + 1e-9)
    target = (wn[..., None] * v[:, None, :, :]).sum(-2)    # (P, Mg, 2)
    sink = float(np.exp(-(sink_bw ** 2) / (2 * sigma ** 2)))
    conf = raw / (raw + sink)                              # (P, Mg)
    sqdiff = ((p - target) ** 2).sum(-1)                   # (P, Mg)
    wgt = gI * conf * gmask.to(gI.dtype)                   # (P, Mg)
    return (sqdiff * wgt).sum(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Batched coarse seed
# ─────────────────────────────────────────────────────────────────────────────

def _polar_hist(xy, I, mask, n_r, n_a, r_max, sigma_a):
    """Bin spots into a smoothed (n_r, n_a) polar histogram per item.

    xy   (N, Smax, 2), I (N, Smax), mask (N, Smax) bool.
    Each spot adds its intensity to its (radius-bin, angle-bin), with the angle
    smeared by a small Gaussian (circular) of width sigma_a bins so the later
    cross-correlation is smooth. Returns (N, n_r, n_a).
    """
    import torch
    dev = xy.device
    N, S, _ = xy.shape
    r = torch.sqrt((xy ** 2).sum(-1) + 1e-12)             # (N,S)
    a = torch.atan2(xy[..., 1], xy[..., 0])               # (N,S) in (-pi,pi]
    rb = torch.clamp((r / r_max * n_r).long(), 0, n_r - 1)  # (N,S)
    ab = (a + np.pi) / (2 * np.pi) * n_a                  # (N,S) float angle bin
    w = (I * mask.to(I.dtype))                            # (N,S)

    # circular angular smear: spread each spot over all angle bins by a wrapped
    # Gaussian, then matmul-scatter — fully vectorised, no python loop.
    abins = torch.arange(n_a, device=dev).view(1, 1, n_a)  # (1,1,n_a)
    da = abins - ab[..., None]                             # (N,S,n_a)
    da = (da + n_a / 2) % n_a - n_a / 2                    # wrap to (-n_a/2,n_a/2]
    ang_w = torch.exp(-(da ** 2) / (2 * sigma_a ** 2))     # (N,S,n_a)
    ang_w = ang_w * w[..., None]                           # weight by intensity

    hist = xy.new_zeros(N, n_r, n_a)
    # scatter-add over the radius bin: flatten (r,a) → index, add ang_w
    idx = (rb[..., None] * n_a + abins).reshape(N, -1)     # (N, S*n_a)
    hist.view(N, -1).scatter_add_(1, idx, ang_w.reshape(N, -1))
    return hist                                            # (N, n_r, n_a)


def _coarse_seed_batched(g, gI, gmask, v, vI, vmask, n_angles, sigma):
    """Best (template, in-plane angle) per pattern via polar-histogram angular
    cross-correlation — O((T+P)·n_r·n_a) instead of O(P·T·A·M·V).

    Both templates and patterns are binned into smoothed polar histograms; an
    in-plane rotation is a circular shift along the angle axis, so the best
    (template, angle) per pattern is the argmax of the circular cross-correlation
    of their radial-angular signatures. The correlation over all template/angle
    pairs is one batched FFT product — no per-template or per-angle Python loop.
    Returns (best_template (P,), best_theta (P,), best_score (P,)).
    """
    import torch
    dev = v.device
    T = g.shape[0]
    P = v.shape[0]
    # angle resolution = n_angles (the requested in-plane search granularity);
    # radius bins kept modest — the signature only needs to separate shells.
    n_a = int(n_angles)
    r_max = float(torch.sqrt((g ** 2).sum(-1)).max().clamp_min(1e-6))
    n_r = 24
    sig_a = max(1.0, n_a * (sigma * 4))   # angular smear ~ a few bins

    gh = _polar_hist(g, gI, gmask, n_r, n_a, r_max, sig_a)     # (T,n_r,n_a)
    vh = _polar_hist(v, vI, vmask, n_r, n_a, r_max, sig_a)     # (P,n_r,n_a)

    # circular cross-correlation along angle, summed over radius shells, via FFT:
    #   corr[p,t,Δ] = Σ_r Σ_a vh[p,r,a] · gh[t,r,(a-Δ)]
    Gf = torch.fft.rfft(gh, dim=-1)                            # (T,n_r,F)
    Vf = torch.fft.rfft(vh, dim=-1)                            # (P,n_r,F)

    best_score = torch.full((P,), -1.0, device=dev)
    best_t = torch.zeros(P, dtype=torch.long, device=dev)
    best_a = torch.zeros(P, device=dev)
    # chunk over patterns so the (Pc,T,n_a) correlation stays bounded (~a few
    # hundred MB); one batched FFT product per chunk, no per-template loop.
    pchunk = max(1, int(40_000_000 / max(1, T * n_a)))
    for p0 in range(0, P, pchunk):
        p1 = min(p0 + pchunk, P)
        Vc = Vf[p0:p1]                                        # (Pc,n_r,F)
        cross = torch.fft.irfft(
            (Vc[:, None] * Gf[None].conj()).sum(2), n=n_a, dim=-1)  # (Pc,T,n_a)
        sc, idx = cross.reshape(p1 - p0, -1).max(-1)
        best_score[p0:p1] = sc
        best_t[p0:p1] = idx // n_a
        best_a[p0:p1] = (idx % n_a).to(v.dtype) * (2 * np.pi / n_a)
    # shift Δ bins → rotation angle; wrap to (-π,π].
    best_a = (best_a + np.pi) % (2 * np.pi) - np.pi
    return best_t, best_a, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Full batched fit
# ─────────────────────────────────────────────────────────────────────────────

def compute_vector_orientation_gpu(
    vectors, lib: TemplateLibrary, params: Optional[dict] = None,
    t: Optional[int] = None, progress=None, stopped_flag=None,
    shm_name: Optional[str] = None, n_seed_angles: int = 72,
    refine_steps: int = 60, on_yield=None,
) -> Optional[VectorOrientationResult]:
    """Fit the whole field on the GPU in one batched pass. See module docstring.

    progress(done, total) is called a handful of times (seed, each anneal
    stage, decode) so the GUI can show coarse progress + paint the live buffer.

    on_yield (if given) is called after each anneal stage. The whole fit is
    short (~1-2s) and MUST run on the GUI/main thread — torch's CUDA autograd
    backward segfaults off the main thread on Windows — so the caller passes
    ``QApplication.processEvents`` here to keep the UI responsive and let the
    shm-preview poll timer paint partial results.
    """
    import torch
    P_params = {**DEFAULTS, **(params or {})}
    cap = float(P_params["strain_cap"])
    sink_bw = float(P_params["sink_bw"])
    sched = P_params["sigma_schedule"]
    ny, nx = vectors.nav_shape
    total_pat = ny * nx

    # CUDA → Apple-MPS → CPU. Even on CPU the batched/vectorised solve is far
    # faster than the serial per-pattern scipy-LM path.
    dev = select_device() or torch.device("cpu")
    dt = torch.float32

    v, vI, vmask, counts = _pack_patterns(vectors, t, dev, dt)
    g, gI, gmask = _pack_templates(lib, dev, dt)
    P = v.shape[0]
    valid_bool = torch.as_tensor(counts >= 3, device=dev)
    # Float weight (not bool) for the loss: backprop through a bool*float Mul can
    # trip a CUDA illegal-access on some torch/driver combos.
    valid = valid_bool.to(dt)

    def _report(frac):
        if progress is not None:
            progress(int(frac * total_pat), total_pat)

    # ── 1. Batched coarse seed ──────────────────────────────────────────────
    if stopped_flag is not None and stopped_flag[0]:
        return None
    best_t, best_a, _ = _coarse_seed_batched(
        g, gI, gmask, v, vI, vmask, n_seed_angles, sched[0])
    _report(0.25)

    # The stretch S is SPD-bounded (I+E within ±cap), so it cannot represent the
    # −S that a θ≈±180° branch would demand: at that branch the SPD optimiser is
    # stuck and reports garbage strain. A diffraction pattern's 180° in-plane
    # ambiguity is physical (Friedel), so θ and θ±π are equivalent *matches* —
    # collapse the seed into (−π/2, π/2] so the refine stays in the SPD-valid
    # basin. (Mirrors the CPU path, whose free-2×2 A absorbs the sign.)
    best_a = torch.remainder(best_a + (np.pi / 2), np.pi) - (np.pi / 2)

    # gather each pattern's seed template spots
    gp = g[best_t]            # (P, Mmax, 2)
    gIp = gI[best_t]
    gmp = gmask[best_t]

    # ── 2. Batched refine (Adam over the sigma anneal) ──────────────────────
    theta = best_a.clone().requires_grad_(True)
    eps = torch.zeros(P, 3, device=dev, dtype=dt, requires_grad=True)
    tvec = torch.zeros(P, 2, device=dev, dtype=dt, requires_grad=True)
    opt = torch.optim.Adam([theta, eps, tvec], lr=0.02)
    n_stages = len(sched)
    # The autograd engine's background worker thread segfaults under CUDA on
    # Windows when backward() runs off the main thread (we run on a daemon
    # worker). Pin backward to the calling thread for the whole refine.
    _prev_mt = torch.autograd.is_multithreading_enabled()
    torch.autograd.set_multithreading_enabled(False)
    try:
        for si, sigma in enumerate(sched):
            # At wide sigma the Gaussian soft-assign is minimised by *shrinking*
            # the template (a spurious negative-strain basin that runs to the
            # cap). So fit a rigid pose (theta + beam-shift) through the coarse
            # stages and only release the strain DOF at the finest sigma, where
            # the true strain is the global minimum. Keeps strain honest and
            # agrees with the CPU LM path.
            fit_strain = si == n_stages - 1
            for step in range(refine_steps):
                if stopped_flag is not None and stopped_flag[0]:
                    return None
                opt.zero_grad()
                cost = _batched_cost(theta, eps, tvec, gp, gIp, gmp,
                                     v, vI, vmask, float(sigma), sink_bw, cap)
                (cost * valid).sum().backward()
                if not fit_strain and eps.grad is not None:
                    eps.grad.zero_()      # freeze strain during the rigid stage
                opt.step()
                # Yield to the GUI a few times *within* each stage so the window
                # never freezes for seconds and the progress bar advances
                # smoothly (each stage is otherwise one long blocking call).
                if on_yield is not None and (step % 12) == 11:
                    frac = (si + (step + 1) / refine_steps) / n_stages
                    _report(0.25 + 0.6 * frac)
                    on_yield()
            _report(0.25 + 0.6 * (si + 1) / n_stages)
            _live_preview(si, shm_name, eps, cap, ny, nx, theta, best_t,
                          valid_bool, lib)
            if on_yield is not None:
                on_yield()
    finally:
        torch.autograd.set_multithreading_enabled(_prev_mt)

    # ── 3. Decode → arrays ──────────────────────────────────────────────────
    with torch.no_grad():
        S, strain3 = _affine_from_logstrain(eps, cap)      # (P,2,2),(P,3)
        resid = _final_residual(theta, eps, tvec, gp, gIp, gmp,
                                v, vI, vmask, sched[-1], cap)
    theta_np = theta.detach().cpu().numpy()
    strain_np = strain3.detach().cpu().numpy()
    bt_np = best_t.detach().cpu().numpy()
    valid_np = valid_bool.detach().cpu().numpy()
    resid_np = resid.detach().cpu().numpy()
    _report(0.92)

    res = _assemble_result(
        ny, nx, theta_np, strain_np, bt_np, valid_np, resid_np, lib,
        dict(P_params))
    _report(1.0)
    if shm_name is not None:
        _write_final_preview(shm_name, ny, nx, res)
    return res


def _final_residual(theta, eps, tvec, g, gI, gmask, v, vI, vmask, sigma, cap):
    """Per-pattern mean nearest template→measured distance (Å⁻¹), matched only."""
    import torch
    S, _ = _affine_from_logstrain(eps, cap)
    R = _rot_mat(theta)
    M = torch.bmm(S, R)
    p = torch.einsum("pmd,pcd->pmc", g, M) + tvec[:, None, :]
    d2 = ((p[:, :, None, :] - v[:, None, :, :]) ** 2).sum(-1)
    d2 = d2 + (~vmask[:, None, :]) * 1e6
    mind = d2.min(-1).values.clamp_min(0).sqrt()           # (P, Mg)
    matched = (mind < 3 * sigma) & gmask
    cnt = matched.sum(-1).clamp_min(1)
    return (mind * matched).sum(-1) / cnt


def _assemble_result(ny, nx, theta_np, strain_np, bt_np, valid_np, resid_np,
                     lib, params) -> VectorOrientationResult:
    """Build the result container. Quaternions are resolved in ONE vectorised
    orix call (resolve_quaternions over the whole field), not per pattern."""
    from spyde.actions.orientation_compute import resolve_quaternions
    strain = strain_np.reshape(ny, nx, 3).astype(np.float32)
    valid2d = valid_np.reshape(ny, nx)
    strain[~valid2d] = np.nan
    theta = theta_np.reshape(ny, nx).astype(np.float32)
    bt = np.clip(bt_np.reshape(ny, nx), 0, len(lib.template_phase) - 1)
    phase_idx = lib.template_phase[bt].astype(np.int16)
    residual = resid_np.reshape(ny, nx).astype(np.float32)
    residual[~valid2d] = np.nan

    # result4 rows = [lib_idx, corr, angle_deg, mirror]; mirror=+1 (vector fit
    # has no mirror — the affine Rot absorbs in-plane rotation).
    result4 = np.zeros((ny, nx, 4), np.float32)
    result4[..., 0] = bt
    result4[..., 1] = valid2d.astype(np.float32)
    result4[..., 2] = np.rad2deg(theta)
    result4[..., 3] = 1.0
    quats = resolve_quaternions(result4, lib.template_quats)

    return VectorOrientationResult(
        quats=quats, phase_idx=phase_idx, theta=theta, strain=strain,
        residual=residual, friedel_asym=np.full((ny, nx), np.nan, np.float32),
        n_matched=np.zeros((ny, nx), np.int16),
        coarse_score=valid2d.astype(np.float32),
        phases_meta=lib.phases_meta, nav_shape=(ny, nx), params=params,
    )


# ── live preview into the 12-channel shm buffer ──────────────────────────────

_PREVIEW_CH = 12


def _live_preview(si, shm_name, eps, cap, ny, nx, theta, best_t, valid_bool, lib):
    """Paint the current strain estimate into the shm buffer after a stage.

    Only the strain channels (9:12) are written live — that is the progressive
    signal the user watches fill in. The IPF orientation RGB needs an orix
    quaternion resolve (`_assemble_result`), which is comparatively slow and is
    deferred to the single authoritative final paint. (Resolving orix per stage
    inside the live CUDA graph is also fragile in-process.)
    """
    import torch
    if shm_name is None:
        return
    with torch.no_grad():
        _, strain3_live = _affine_from_logstrain(eps, cap)
    strain = strain3_live.cpu().numpy().reshape(ny, nx, 3).astype(np.float32)
    valid2d = valid_bool.detach().cpu().numpy().reshape(ny, nx)
    strain[~valid2d] = np.nan
    try:
        from multiprocessing import shared_memory as _shm
        sh = _shm.SharedMemory(name=shm_name, create=False)
        try:
            buf = np.ndarray((ny, nx, _PREVIEW_CH), np.float32, buffer=sh.buf)
            buf[..., 9:12] = strain
            # mark channel 0 finite where valid so the poll's "any finite" gate
            # fires and the strain panels repaint as the field fills in.
            buf[..., 0] = np.where(valid2d, 0.0, np.nan).astype(np.float32)
        finally:
            sh.close()
    except Exception as e:
        log.debug("live vector-orientation preview write failed: %s", e)


def _write_final_preview(shm_name, ny, nx, res: VectorOrientationResult):
    try:
        from multiprocessing import shared_memory as _shm
        sh = _shm.SharedMemory(name=shm_name, create=False)
        try:
            buf = np.ndarray((ny, nx, _PREVIEW_CH), np.float32, buffer=sh.buf)
            for di, d in enumerate(("x", "y", "z")):
                buf[..., 3 * di:3 * di + 3] = res.ipf_color_map(d).astype(np.float32)
            buf[..., 9:12] = res.strain
        finally:
            sh.close()
    except Exception as e:
        log.debug("final vector-orientation preview write failed: %s", e)
