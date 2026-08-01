"""
nonrigid.py — non-rigid drift: scan-knot and dense-field, one solver (plan A2-A5).

Rigid translation (:mod:`spyde.drift.translation`) removes the part of the drift
that moves the whole frame. What is left is real and is what this module fits:

* **Scan distortion.** A scanned frame is not acquired instantaneously — the
  stage keeps moving while the beam rasters, so each ROW is displaced by a
  different amount. The distortion is therefore a function of the SLOW scan
  coordinate, which is why one displacement per row (smoothed) is the natural
  parameterisation and not a general 2-D field.
* **Sample deformation.** The specimen itself bends, and parts of the field
  move independently of each other. No function of the scan coordinate can
  express that, so it needs displacements that vary in both directions.

Both causes are real, so the model is SELECTABLE rather than assumed
(:func:`solve_nonrigid`'s ``model=`` argument). They share the warp, the
solver and the regularisation; only the parameter -> displacement map differs,
which is the whole reason both are affordable.

Sign convention
---------------
Identical to :class:`~spyde.drift.model.DriftModel`: a displacement is the
correction you ADD to a pixel's coordinate to bring it into the reference. The
rigid ``shifts`` stay in the model unchanged and the non-rigid field is the
RESIDUAL on top of them, so a ``kind="scan_knot"`` model applied without its
extra parameters still degrades gracefully to the rigid answer rather than to
nonsense.

Why gather (``grid_sample``) and not the KDE scatter the plan sketched
---------------------------------------------------------------------
quantem resamples with a KDE scatter (``index_put_(accumulate=True)`` over the
four bilinear neighbours plus a weight image) because it is *building* a
reconstruction from many scans, where several source pixels legitimately land on
one output pixel and must accumulate.

Here the job is the inverse: one frame, resampled onto the reference grid. That
is a GATHER — for each output pixel, read the input at a computed coordinate —
and ``torch.nn.functional.grid_sample`` does exactly that, differentiably, with
a fused CUDA kernel. A scatter would need its own normalisation pass, leaves
holes wherever no source pixel lands, and is strictly more code for a worse
result on this problem. The scatter formulation is still the right one if this
ever grows into multi-scan reconstruction; it is not needed to correct a movie.

Out-of-bounds samples come back NaN, matching :mod:`spyde.drift.warp`'s locked
edge policy (nothing cropped, nothing invented) — see :func:`warp_frame`.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable

import numpy as np

from spyde.drift.model import DriftModel

log = logging.getLogger(__name__)

# Parameterisation names, also the DriftModel.kind values they produce.
SCAN_KNOT = "scan_knot"
DENSE = "dense"
MODELS = (SCAN_KNOT, DENSE)


# ── torch plumbing ───────────────────────────────────────────────────────────

def _torch():
    try:
        import torch
    except ImportError as e:                                  # pragma: no cover
        raise RuntimeError(
            "non-rigid drift needs torch; install it or use solve_translation"
        ) from e
    return torch


def gpu_available() -> bool:
    """True when a CUDA device is usable. Mirrors the other GPU paths."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_device(device: str | None):
    torch = _torch()
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── parameterisations: parameters -> per-pixel displacement ──────────────────

def _bezier_basis(torch, n_rows: int, n_knots: int, device, dtype):
    """``(n_rows, n_knots)`` Bernstein/Bézier basis over the slow scan axis.

    Bézier rather than a free per-row displacement because scan distortion is
    SMOOTH in the slow coordinate — the stage does not jerk row to row. A free
    per-row fit has one parameter per row and happily absorbs sample motion and
    noise into the "scan" term, which is precisely the failure this model is
    supposed to avoid. ``n_knots=1`` (a constant offset per frame, i.e. degree 0)
    is quantem's documented default for uniform distortion; 2-4 covers the
    curved-drift case.
    """
    n_knots = max(1, int(n_knots))
    t = torch.linspace(0.0, 1.0, n_rows, device=device, dtype=dtype)
    if n_knots == 1:
        return torch.ones((n_rows, 1), device=device, dtype=dtype)
    deg = n_knots - 1
    ks = torch.arange(n_knots, device=device, dtype=dtype)
    # C(deg, k) t^k (1-t)^(deg-k), built in log space so deg>10 stays finite.
    logc = (math.lgamma(deg + 1)
            - torch.lgamma(ks + 1) - torch.lgamma(torch.tensor(float(deg), device=device, dtype=dtype) - ks + 1))
    tt = t[:, None].clamp(1e-7, 1 - 1e-7)
    return torch.exp(logc[None, :] + ks[None, :] * torch.log(tt)
                     + (deg - ks)[None, :] * torch.log1p(-tt))


def scan_knot_field(torch, knots, shape, scan_direction_degrees: float = 0.0):
    """Per-pixel ``(dy, dx)`` from scan knots.

    ``knots`` is ``(N, 2, n_knots)`` — per frame, per scan axis (fast, slow),
    per knot. The displacement varies only along the SLOW scan coordinate, then
    is projected onto image axes using the scan direction, so a rotated scan is
    handled without a second parameterisation.
    """
    n, _, n_knots = knots.shape
    h, w = shape
    dev, dt = knots.device, knots.dtype
    ang = math.radians(float(scan_direction_degrees))
    # Fast axis unit vector in (y, x); slow axis is perpendicular.
    fy, fx = math.sin(ang), math.cos(ang)
    sy, sx = math.cos(ang), -math.sin(ang)

    basis = _bezier_basis(torch, h, n_knots, dev, dt)          # (h, n_knots)
    # (N, 2, h): displacement magnitude along each scan axis, per row.
    mag = torch.einsum("nck,rk->ncr", knots, basis)
    fast, slow = mag[:, 0, :], mag[:, 1, :]                    # (N, h) each
    dy = fast * fy + slow * sy
    dx = fast * fx + slow * sx
    # Constant across the fast axis (a row is acquired at one slow coordinate).
    return dy[:, :, None].expand(n, h, w), dx[:, :, None].expand(n, h, w)


def dense_field(torch, control, shape):
    """Per-pixel ``(dy, dx)`` from a coarse control-point grid.

    ``control`` is ``(N, 2, gh, gw)``. Upsampled bicubically to the frame, which
    is the free-form-deformation standard: the grid is coarse (so the model
    cannot chase noise) and the interpolation is smooth (so the recovered field
    has no control-point creases).
    """
    n, _, gh, gw = control.shape
    h, w = shape
    up = torch.nn.functional.interpolate(
        control, size=(h, w), mode="bicubic", align_corners=True)
    return up[:, 0], up[:, 1]


# ── the differentiable warp ──────────────────────────────────────────────────

def warp_frame(torch, frame, dy, dx, *, fill_nan: bool = True):
    """Resample ``frame`` at ``(y + dy, x + dx)``. Differentiable in dy/dx.

    ``frame`` is ``(N, H, W)``; ``dy``/``dx`` are ``(N, H, W)``. Out-of-bounds
    samples become NaN when *fill_nan*, matching the locked edge policy in
    :mod:`spyde.drift.warp` — nothing is cropped and nothing is invented.
    """
    n, h, w = frame.shape
    dev, dt = frame.device, frame.dtype
    yy = torch.arange(h, device=dev, dtype=dt)[None, :, None]
    xx = torch.arange(w, device=dev, dtype=dt)[None, None, :]
    sy = yy + dy
    sx = xx + dx
    # grid_sample wants normalised [-1, 1] with align_corners=True.
    gy = 2.0 * sy / max(h - 1, 1) - 1.0
    gx = 2.0 * sx / max(w - 1, 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1)                       # (N, H, W, 2)
    out = torch.nn.functional.grid_sample(
        frame[:, None], grid, mode="bilinear",
        padding_mode="zeros", align_corners=True)[:, 0]
    if not fill_nan:
        return out
    inside = (gy.abs() <= 1.0) & (gx.abs() <= 1.0)
    return torch.where(inside, out, torch.full_like(out, float("nan")))


# ── regularisation ───────────────────────────────────────────────────────────

def _bending_energy(torch, field):
    """Second-difference energy of a control grid — the FFD smoothness term.

    Penalising CURVATURE rather than magnitude is deliberate: a uniform or
    linearly-varying displacement is exactly what a real drift looks like and
    must not be taxed, while a grid that folds or oscillates between neighbouring
    control points is not a physical deformation.
    """
    e = field.new_zeros(())
    if field.shape[-2] >= 3:
        e = e + (field[..., :-2, :] - 2 * field[..., 1:-1, :] + field[..., 2:, :]).pow(2).mean()
    if field.shape[-1] >= 3:
        e = e + (field[..., :, :-2] - 2 * field[..., :, 1:-1] + field[..., :, 2:]).pow(2).mean()
    return e


def _temporal_energy(torch, params):
    """Penalise frame-to-frame CHANGE of the parameters.

    Drift is continuous in time — the distortion in frame *i* is nearly that of
    frame *i-1*. This is what lets a noisy frame borrow support from its
    neighbours, and it is the term that keeps a single bad frame from acquiring
    its own wild field.
    """
    if params.shape[0] < 2:
        return params.new_zeros(())
    return (params[1:] - params[:-1]).pow(2).mean()


# ── the solver ───────────────────────────────────────────────────────────────

def solve_nonrigid(
    frames,
    *,
    model: str = SCAN_KNOT,
    reference=None,
    rigid: DriftModel | None = None,
    n_knots: int = 2,
    grid: tuple[int, int] = (4, 4),
    scan_direction_degrees: float = 0.0,
    steps: int = 120,
    lr: float = 0.5,
    smooth_weight: float = 1.0,
    temporal_weight: float = 1.0,
    max_displacement: float | None = 32.0,
    device: str | None = None,
    progress: Callable[[int, int], None] | None = None,
    on_yield: Callable[[], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    provenance: dict[str, Any] | None = None,
) -> DriftModel:
    """Fit a non-rigid correction on top of a rigid solve.

    Parameters
    ----------
    frames
        ``(N, H, W)`` array — already rigid-corrected, or pass *rigid* and it is
        applied here. Small enough to hold: this is a fit over a few
        parameters, so callers are expected to pass a decimated or cropped
        stack, not a 900x4096x4096 movie.
    model
        ``"scan_knot"`` (default) or ``"dense"``. See the module docstring for
        which physical cause each one describes.
    reference
        ``(H, W)`` target. Defaults to the mean of *frames*, which is the right
        default for drift: the mean of an already-rigid-aligned stack is the
        sharpest thing available without picking a privileged frame.
    n_knots, grid
        Model size. ``n_knots`` for scan-knot; ``grid`` for dense.
    smooth_weight, temporal_weight
        Regularisation strengths. Both default to 1.0 against a mean-squared
        data term, i.e. deliberately NOT free — an unregularised dense field
        will happily fit noise.
    max_displacement
        Hard clamp on the fitted field, in pixels. ``None`` disables.

    Returns
    -------
    DriftModel
        ``kind`` is *model*, ``shifts`` carries the rigid component (zeros if
        none was given), and ``extra`` holds the parameters plus everything
        needed to rebuild the field (``field_shape``, ``n_knots``/``grid``,
        ``scan_direction_degrees``).

    Notes
    -----
    The Windows CUDA-autograd mitigations are load-bearing and both are applied:
    ``backward()`` segfaults the first time it runs on a thread whose autograd
    engine is uninitialised, so a warm-up backward runs on THIS thread before
    the loop, and multithreaded autograd is disabled around it. See CLAUDE.md.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}; got {model!r}")

    torch = _torch()
    from spyde.device_lock import accelerator_lock

    arr = np.asarray(frames)
    if arr.ndim != 3:
        raise ValueError(f"frames must be (N, H, W); got {arr.shape}")
    n, h, w = arr.shape
    dev = _resolve_device(device)

    # Warm the autograd engine on the CALLING thread — see Notes.
    _warmup_autograd(torch, dev)

    with accelerator_lock(dev):
        f = torch.as_tensor(np.ascontiguousarray(arr, dtype=np.float32), device=dev)

        if rigid is not None and rigid.n_frames == n:
            sh = torch.as_tensor(np.asarray(rigid.shifts, np.float32), device=dev)
            f = warp_frame(torch, f,
                           -sh[:, 0, None, None].expand(n, h, w),
                           -sh[:, 1, None, None].expand(n, h, w),
                           fill_nan=False)

        ref = (torch.as_tensor(np.asarray(reference, np.float32), device=dev)
               if reference is not None else f.mean(0))
        ref = ref[None]

        # Standardise so the loss scale (and therefore the regularisation
        # weights) does not depend on the detector's units.
        mu, sd = f.mean(), f.std().clamp_min(1e-6)
        f = (f - mu) / sd
        ref = (ref - mu) / sd

        if model == SCAN_KNOT:
            p = torch.zeros((n, 2, max(1, int(n_knots))), device=dev, requires_grad=True)
            def field(pp):
                return scan_knot_field(torch, pp, (h, w), scan_direction_degrees)
        else:
            gh, gw = (max(2, int(grid[0])), max(2, int(grid[1])))
            p = torch.zeros((n, 2, gh, gw), device=dev, requires_grad=True)
            def field(pp):
                return dense_field(torch, pp, (h, w))

        opt = torch.optim.Adam([p], lr=float(lr))
        total = max(1, int(steps))
        prev_mt = None
        try:
            prev_mt = torch.autograd.is_multithreading_enabled()
            torch.autograd.set_multithreading_enabled(False)
        except Exception:                                       # pragma: no cover
            prev_mt = None

        try:
            for it in range(total):
                if cancel is not None and cancel():
                    log.info("non-rigid drift cancelled at step %d/%d", it, total)
                    break
                opt.zero_grad(set_to_none=True)
                dy, dx = field(p)
                if max_displacement is not None:
                    m = float(max_displacement)
                    dy = dy.clamp(-m, m)
                    dx = dx.clamp(-m, m)
                moved = warp_frame(torch, f, dy, dx, fill_nan=False)
                data = (moved - ref).pow(2).mean()
                reg = smooth_weight * (_bending_energy(torch, p) if model == DENSE
                                       else _knot_energy(torch, p))
                reg = reg + temporal_weight * _temporal_energy(torch, p)
                loss = data + reg
                loss.backward()
                opt.step()

                if progress is not None and (it % 8 == 0 or it == total - 1):
                    progress(it + 1, total)
                # Yield INSIDE the loop, not per stage — otherwise the window
                # freezes for seconds and the progress bar looks stuck.
                if on_yield is not None and it % 12 == 0:
                    _yield_device(torch, dev, on_yield)
        finally:
            if prev_mt is not None:
                try:
                    torch.autograd.set_multithreading_enabled(prev_mt)
                except Exception:                               # pragma: no cover
                    pass

        with torch.no_grad():
            dy, dx = field(p)
            if max_displacement is not None:
                m = float(max_displacement)
                dy, dx = dy.clamp(-m, m), dx.clamp(-m, m)
            final = float((warp_frame(torch, f, dy, dx, fill_nan=False) - ref)
                          .pow(2).mean().item())
            params_np = p.detach().float().cpu().numpy()
            # Per-frame mean displacement — a compact, inspectable summary and
            # what a 1-D "how much did this frame deform" plot shows.
            mean_dy = dy.mean(dim=(1, 2)).float().cpu().numpy()
            mean_dx = dx.mean(dim=(1, 2)).float().cpu().numpy()

    shifts = (np.asarray(rigid.shifts, np.float32) if rigid is not None and rigid.n_frames == n
              else np.zeros((n, 2), np.float32))
    extra = {
        "params": params_np,
        "field_shape": (int(h), int(w)),
        "mean_dy": np.asarray(mean_dy, np.float32),
        "mean_dx": np.asarray(mean_dx, np.float32),
        "final_mse": final,
        "scan_direction_degrees": float(scan_direction_degrees),
    }
    if model == SCAN_KNOT:
        extra["n_knots"] = int(max(1, n_knots))
    else:
        extra["grid"] = (int(max(2, grid[0])), int(max(2, grid[1])))

    return DriftModel(
        shifts=shifts,
        kind=model,
        reference="mean" if reference is None else "given",
        params={
            "model": model, "steps": int(steps), "lr": float(lr),
            "smooth_weight": float(smooth_weight),
            "temporal_weight": float(temporal_weight),
            "max_displacement": max_displacement,
            "n_knots": int(n_knots), "grid": tuple(grid),
            "scan_direction_degrees": float(scan_direction_degrees),
        },
        provenance=provenance,
        extra=extra,
    )


def _knot_energy(torch, knots):
    """Smoothness across knots — the scan-knot analogue of bending energy."""
    if knots.shape[-1] < 3:
        return knots.new_zeros(())
    return (knots[..., :-2] - 2 * knots[..., 1:-1] + knots[..., 2:]).pow(2).mean()


def _warmup_autograd(torch, device) -> None:
    """One trivial backward on THIS thread before any worker touches autograd.

    CUDA-gated and a no-op elsewhere. On Windows the first ``backward()`` on a
    thread whose autograd engine is uninitialised segfaults — uncatchably — and
    the solve is dispatched to a daemon worker. See CLAUDE.md.
    """
    if getattr(device, "type", None) != "cuda":
        return
    try:
        x = torch.zeros(1, device=device, requires_grad=True)
        (x * x).sum().backward()
    except Exception as e:                                      # pragma: no cover
        log.debug("autograd warm-up failed (continuing): %s", e)


def _yield_device(torch, device, on_yield) -> None:
    """Hand the accelerator back at a yield point, then take it again.

    Always synchronise BEFORE releasing: handing off while kernels are still in
    flight lets the next thread submit into a live encoder, which is the MPS
    race the device lock exists to prevent.
    """
    try:
        if getattr(device, "type", None) == "mps":
            torch.mps.synchronize()
        elif getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize()
    except Exception as e:                                      # pragma: no cover
        log.debug("device sync before yield failed: %s", e)
    try:
        on_yield()
    except Exception as e:                                      # pragma: no cover
        log.debug("on_yield raised (ignored): %s", e)


# ── applying a fitted model ──────────────────────────────────────────────────

def displacement_for_frame(model: DriftModel, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the ``(dy, dx)`` field for one frame of a fitted model.

    Returns arrays of the frame's shape. Raises for a rigid-only model — the
    caller wants :mod:`spyde.drift.warp` for those, and silently returning zeros
    would turn "this model has no non-rigid part" into "this frame did not
    deform", which is a different claim.
    """
    if model.kind not in MODELS:
        raise ValueError(
            f"model.kind is {model.kind!r}, not a non-rigid fit; use spyde.drift.warp"
        )
    torch = _torch()
    p = np.asarray(model.extra["params"], np.float32)
    if not 0 <= index < p.shape[0]:
        raise IndexError(f"frame {index} out of range for {p.shape[0]} fitted frames")
    shape = tuple(model.extra["field_shape"])
    t = torch.as_tensor(p[index: index + 1])
    if model.kind == SCAN_KNOT:
        dy, dx = scan_knot_field(
            torch, t, shape, float(model.extra.get("scan_direction_degrees", 0.0)))
    else:
        dy, dx = dense_field(torch, t, shape)
    return (dy[0].numpy().copy(), dx[0].numpy().copy())


def apply_nonrigid(frame, model: DriftModel, index: int) -> np.ndarray:
    """Apply a fitted non-rigid model to ONE frame. NaN outside coverage.

    Per-frame by design, like :func:`spyde.drift.warp.shift_frame` — the aligned
    movie is never materialised.
    """
    torch = _torch()
    dy, dx = displacement_for_frame(model, index)
    f = np.asarray(frame, np.float32)
    if f.shape != dy.shape:
        raise ValueError(f"frame is {f.shape}, model was fitted at {dy.shape}")
    t = torch.as_tensor(f)[None]
    out = warp_frame(torch, t,
                     torch.as_tensor(dy)[None], torch.as_tensor(dx)[None],
                     fill_nan=True)
    return out[0].numpy().copy()
