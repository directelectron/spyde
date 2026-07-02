"""
spyde.api — the script-parity layer.

Every SpyDE scientific action, callable from a plain Python script (or a
Jupyter kernel) with **no Session, no Electron, no display**: signal/vectors
in, self-contained result object out. These are thin, typed wrappers over the
SAME compute cores the app's toolbar actions dispatch to, so a scripted
result and an in-app result are numerically identical.

    import hyperspy.api as hs
    from spyde import api

    sig  = hs.load("scan.zspy", lazy=True, chunks=(32, 32, -1, -1))
    vecs = api.find_vectors(sig, method="nxcorr", threshold=0.5)
    sf   = api.strain_map(vecs)                       # auto reference pixel
    om   = api.orientation_map(sig, "austenite.cif")

Design rules (enforced by ``test_api_layer.py``):

* **This module never imports ``spyde.backend`` or ``spyde.drawing``** — it
  must stay importable and runnable in an environment with no UI stack.
* Heavy imports happen inside functions (importing ``spyde.api`` is cheap).
* Every result carries a ``provenance`` record
  ``{"action", "params", "spyde_version"}`` — the same dict convention the
  app's Commit action stamps (``commit._stamp_provenance``), so scripted
  results and committed trees are interchangeable.
* ``client=`` (a ``dask.distributed.Client``) is optional everywhere a batch
  compute can use one; without it computes fall back to the local threaded
  scheduler.

The three-host parity contract (script ↔ Jupyter ↔ SpyDE) is documented in
``NOTEBOOK_PARITY_PLAN.md``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:  # names only — no runtime imports of the heavy stack
    from spyde.actions.strain_mapping import StrainField
    from spyde.actions.vector_orientation import (
        TemplateLibrary, VectorOrientationResult,
    )
    from spyde.signals.diffraction_vectors import SpyDEDiffractionVectors
    from spyde.signals.orientation_map import SpyDEOrientationMap

__all__ = [
    "find_vectors",
    "orientation_map",
    "vector_orientation_map",
    "strain_map",
    "center_zero_beam",
    "virtual_image",
    "vector_virtual_image",
]


def _provenance(action: str, params: dict) -> dict:
    import spyde
    clean = {k: v for k, v in params.items()
             if not isinstance(v, np.ndarray)}          # keep the record small
    return {"action": action, "params": clean,
            "spyde_version": getattr(spyde, "__version__", "unknown")}


def _as_phases(phases) -> list:
    """Normalize ``phases``: CIF path(s) and/or orix ``Phase`` object(s) →
    ``[Phase]``."""
    from orix.crystal_map import Phase
    if isinstance(phases, (str,)) or hasattr(phases, "__fspath__"):
        return [Phase.from_cif(str(phases))]
    if isinstance(phases, Phase):
        return [phases]
    out = []
    for p in phases:
        out.extend(_as_phases(p))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Find Diffraction Vectors
# ─────────────────────────────────────────────────────────────────────────────

def find_vectors(
    signal,
    *,
    method: str = "nxcorr",
    sigma: float = 1.0,
    kernel_radius: int = 5,
    threshold: Optional[float] = None,
    min_distance: int = 5,
    subpixel: bool = True,
    dog_sigma1: float = 0.8,
    dog_sigma2: float = 2.0,
    beamstop_auto: bool = False,
    beamstop_mask: Optional[np.ndarray] = None,
    client=None,
) -> "SpyDEDiffractionVectors":
    """Detect diffraction spots across the whole scan.

    The batch core is ``find_vectors.orchestrate._do_compute_vectors`` — the
    exact compute the app's Find Diffraction Vectors action runs (ghost-padded
    nav chunks, never materialises the full dataset).

    Parameters
    ----------
    signal : hyperspy/pyxem 4D (or 5D) signal, numpy or lazy dask.
    method : "nxcorr" (disk cross-correlation) | "dog" (band-pass, for small /
        beam-stopped spots) | "neural" where available.
    threshold : method-scaled score cut. Default 0.5 for nxcorr (score in
        [-1, 1]) and 10.0 for dog (SNR units) — the same method-aware default
        the app applies.
    beamstop_auto : detect a static beam stop from sampled frames and mask it.
    beamstop_mask : explicit (ky, kx) bool mask (overrides beamstop_auto).
    client : optional ``dask.distributed.Client`` for parallel/GPU dispatch;
        None → local threaded compute.
    """
    from spyde.actions.find_vectors import (
        DEFAULT_DOG_THRESHOLD, _auto_beamstop_from_signal, _do_compute_vectors,
    )

    if threshold is None:
        threshold = DEFAULT_DOG_THRESHOLD if method == "dog" else 0.5
    params = dict(
        method=str(method).lower(), sigma=float(sigma),
        kernel_radius=int(kernel_radius), threshold=float(threshold),
        min_distance=int(min_distance), subpixel=bool(subpixel),
        dog_sigma1=float(dog_sigma1), dog_sigma2=float(dog_sigma2),
    )
    if beamstop_mask is None and beamstop_auto:
        nav_dim = signal.axes_manager.navigation_dimension
        beamstop_mask = _auto_beamstop_from_signal(signal, nav_dim)

    vecs = _do_compute_vectors(signal, params, beamstop_mask=beamstop_mask,
                               client=client)
    if vecs is not None:
        vecs.provenance = _provenance(
            "find_vectors", {**params, "beamstop": beamstop_mask is not None})
    return vecs


# ─────────────────────────────────────────────────────────────────────────────
# Orientation Mapping (dense)
# ─────────────────────────────────────────────────────────────────────────────

def orientation_map(
    signal,
    phases,
    *,
    accelerating_voltage: float = 200.0,
    resolution: float = 1.0,
    minimum_intensity: float = 1e-4,
    reciprocal_radius: Optional[float] = None,
    max_excitation_error: float = 0.1,
    n_best: int = 5,
    gamma: float = 1.0,
    normalize_templates: bool = True,
    client=None,
) -> "SpyDEOrientationMap":
    """Template-match orientations over a dense 4D-STEM scan.

    Builds the diffsims template library from ``phases`` (CIF path(s) or orix
    ``Phase`` object(s)) and runs the app's batch matcher
    (``orientation_compute._do_compute_orientations``).

    ``reciprocal_radius`` defaults to the largest radius that fits the
    detector (from the signal-axis calibration) — the same choice the app
    makes.
    """
    from spyde.actions._common import reciprocal_radius as _recip
    from spyde.actions.orientation_compute import (
        _do_compute_orientations, generate_library_from_phases,
    )

    phase_list = _as_phases(phases)
    if reciprocal_radius is None:
        reciprocal_radius = _recip(signal)
    try:
        signal.set_signal_type("electron_diffraction")   # pyxem calibration
    except Exception:
        pass

    sim = generate_library_from_phases(
        phase_list, accelerating_voltage, resolution, minimum_intensity,
        reciprocal_radius, max_excitation_error=max_excitation_error,
    )
    params = dict(n_best=int(n_best), gamma=float(gamma),
                  normalize_templates=bool(normalize_templates))
    om = _do_compute_orientations(signal, sim, params, client=client)
    if om is not None:
        om.provenance = _provenance("orientation_map", {
            **params,
            "phases": [getattr(p, "name", "?") for p in phase_list],
            "accelerating_voltage": accelerating_voltage,
            "resolution": resolution,
            "minimum_intensity": minimum_intensity,
            "reciprocal_radius": reciprocal_radius,
            "max_excitation_error": max_excitation_error,
        })
    return om


# ─────────────────────────────────────────────────────────────────────────────
# Vector Orientation Mapping
# ─────────────────────────────────────────────────────────────────────────────

def vector_orientation_map(
    vectors,
    library_or_phases: Union["TemplateLibrary", object],
    *,
    calibration_signal=None,
    accelerating_voltage: float = 200.0,
    resolution: float = 1.0,
    minimum_intensity: float = 1e-4,
    r_max: Optional[float] = None,
    gpu: Union[str, bool] = "auto",
    progress=None,
    client=None,
    **fit_params,
) -> "VectorOrientationResult":
    """Fit orientation + strain per pattern from detected vectors (sparse OM).

    ``library_or_phases`` is either a prebuilt
    ``vector_orientation.TemplateLibrary`` or CIF path(s)/``Phase`` object(s);
    building from phases needs ``calibration_signal`` (an
    ElectronDiffraction2D whose signal axes carry the data's calibration —
    typically the source signal the vectors were found on).

    ``gpu="auto"`` uses the whole-field batched torch fit when a CUDA/MPS
    device exists (the app's production path), otherwise the chunked CPU fit
    (thread pool without a ``client``). ``fit_params`` are forwarded to the
    fit (see ``vector_orientation.DEFAULTS``: strain_cap, sigma_schedule, …).
    """
    from spyde.actions.vector_orientation import (
        TemplateLibrary, build_template_library,
        compute_vector_orientation_chunked,
    )
    from spyde.actions.vector_orientation_gpu import (
        compute_vector_orientation_gpu, gpu_available,
    )

    if isinstance(library_or_phases, TemplateLibrary):
        lib = library_or_phases
        lib_meta = "prebuilt"
    else:
        if calibration_signal is None:
            raise ValueError(
                "building a template library from phases needs "
                "calibration_signal= (the source ElectronDiffraction2D); "
                "alternatively pass a prebuilt TemplateLibrary")
        from spyde.actions._common import reciprocal_radius as _recip
        from spyde.actions.orientation_compute import (
            generate_library_from_phases,
        )
        phase_list = _as_phases(library_or_phases)
        recip_r = _recip(calibration_signal)
        sim = generate_library_from_phases(
            phase_list, accelerating_voltage, resolution, minimum_intensity,
            recip_r)
        lib = build_template_library(
            sim, calibration_signal, r_max if r_max is not None else recip_r)
        lib_meta = [getattr(p, "name", "?") for p in phase_list]

    use_gpu = gpu_available() if gpu == "auto" else bool(gpu)
    if use_gpu:
        res = compute_vector_orientation_gpu(vectors, lib, fit_params or None,
                                             progress=progress)
    else:
        res = compute_vector_orientation_chunked(
            vectors, lib, fit_params or None, progress=progress,
            client=client)
    if res is not None:
        res.provenance = _provenance("vector_orientation_map", {
            **(fit_params or {}), "gpu": use_gpu, "library": lib_meta})
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Strain Mapping
# ─────────────────────────────────────────────────────────────────────────────

def strain_map(
    vectors,
    *,
    ref_yx: Optional[tuple] = None,
    ref_vectors: Optional[np.ndarray] = None,
    cif: Optional[Union[str, object]] = None,
    tol: Optional[float] = None,
    min_dspacing: float = 0.7,
    tol_frac: float = 0.2,
) -> "StrainField":
    """Whole-field strain from detected vectors — the app's Strain Mapping.

    The reference lattice, in priority order:

    * ``ref_vectors`` — explicit (N, 2) ``(kx, ky)`` array;
    * ``ref_yx`` — a navigation pixel whose vectors become the reference
      (default: the pixel with the most vectors, the wizard's choice);
    * ``cif`` — additionally snaps the reference magnitudes to the CIF's
      allowed |g| families (absolute strain; the wizard's CIF mode).

    Every reference is zero-beam-filtered — the SAME physics as the wizard
    (``strain_mapping.zero_beam_filtered``).
    """
    from spyde.actions.strain_mapping import (
        cif_g_families, compute_strain_field, default_reference,
        snap_reference_to_cif, zero_beam_filtered,
    )

    if ref_vectors is None:
        if ref_yx is None:
            ref_yx = default_reference(vectors)
        ry, rx = int(ref_yx[0]), int(ref_yx[1])
        ref_vectors = vectors.kxy_at(ry, rx)
    ref_vectors = zero_beam_filtered(ref_vectors)

    if cif is not None:
        phase = _as_phases(cif)[0]
        families = cif_g_families(phase, min_dspacing=min_dspacing)
        ref_vectors = snap_reference_to_cif(ref_vectors, families,
                                            tol_frac=tol_frac)

    sf = compute_strain_field(vectors, ref_vectors=ref_vectors, tol=tol)
    sf.provenance = _provenance("strain_map", {
        "ref_yx": tuple(ref_yx) if ref_yx is not None else None,
        "cif": str(cif) if cif is not None else None,
        "tol": tol, "min_dspacing": min_dspacing, "tol_frac": tol_frac,
        "n_ref": int(len(ref_vectors)),
    })
    return sf


# ─────────────────────────────────────────────────────────────────────────────
# Center Zero Beam
# ─────────────────────────────────────────────────────────────────────────────

def center_zero_beam(
    signal,
    *,
    method: str = "center_of_mass",
    half_square_width: int = 0,
    plane_fit: bool = False,
    inplace: bool = False,
):
    """Estimate the direct-beam position per pattern and centre it — the same
    pyxem calls as the app's automatic Center Zero Beam
    (``get_direct_beam_position`` → optional linear-plane flat field →
    ``center_direct_beam``). Returns the centered signal (or the input when
    ``inplace=True``)."""
    try:
        signal.set_signal_type("electron_diffraction")
    except Exception:
        pass
    kw = {"method": str(method), "lazy_output": False}
    if int(half_square_width) > 0:
        kw["half_square_width"] = int(half_square_width)
    shifts = signal.get_direct_beam_position(**kw)
    if getattr(shifts, "_lazy", False):
        shifts.compute()
    if plane_fit:
        lp = shifts.get_linear_plane()
        if lp is not None:
            shifts = lp
    out = signal.center_direct_beam(shifts=shifts, inplace=inplace)
    result = signal if inplace else out
    try:
        result.metadata.set_item(
            "General.spyde_provenance",
            _provenance("center_zero_beam", {
                "method": method, "half_square_width": half_square_width,
                "plane_fit": plane_fit}))
    except Exception:
        pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Virtual Imaging
# ─────────────────────────────────────────────────────────────────────────────

def virtual_image(
    signal,
    *,
    cx: float,
    cy: float,
    r: Optional[float] = None,
    r_inner: float = 0.0,
    kind: str = "disk",
    calculation: str = "sum",
    client=None,
):
    """Virtual image from a geometric detector on the signal (k-space) plane.

    ``cx``/``cy``/``r``/``r_inner`` are in DATA coordinates (the signal-axis
    units, e.g. Å⁻¹), matching ``vector_virtual_image``. ``kind`` is "disk" or
    "annulus" (annulus = ``r_inner`` > 0 implied). Returns a hyperspy
    ``Signal2D`` over the navigation grid, with the source's navigation-axis
    calibration. The reduction is per-chunk (lazy-safe — never materialises
    the dataset)."""
    import hyperspy.api as hs

    if r is None:
        raise ValueError("virtual_image needs r= (outer radius, data units)")
    sig_ax = signal.axes_manager.signal_axes    # (kx, ky) hyperspy order
    ky_ax, kx_ax = sig_ax[1], sig_ax[0]
    ky = ky_ax.offset + ky_ax.scale * np.arange(ky_ax.size)
    kx = kx_ax.offset + kx_ax.scale * np.arange(kx_ax.size)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    rho2 = (KX - float(cx)) ** 2 + (KY - float(cy)) ** 2
    mask = rho2 <= float(r) ** 2
    if kind == "annulus" or float(r_inner) > 0:
        mask &= rho2 >= float(r_inner) ** 2
    if not mask.any():
        raise ValueError("virtual detector selects no pixels — check cx/cy/r "
                         "against the signal-axis calibration")

    data = signal.data * mask                    # broadcast over nav dims
    vi = data.sum(axis=(-2, -1))
    if calculation == "mean":
        vi = vi / float(mask.sum())
    if hasattr(vi, "compute"):                   # lazy dask reduction
        vi = client.compute(vi).result() if client is not None \
            else vi.compute(scheduler="threads")

    out = hs.signals.Signal2D(vi)
    for dst, src in zip(out.axes_manager.signal_axes[::-1],
                        signal.axes_manager.navigation_axes[::-1]):
        dst.scale, dst.offset = src.scale, src.offset
        dst.name, dst.units = src.name, src.units
    out.metadata.set_item("General.spyde_provenance", _provenance(
        "virtual_image", {"cx": cx, "cy": cy, "r": r, "r_inner": r_inner,
                          "kind": kind, "calculation": calculation}))
    return out


def vector_virtual_image(
    vectors,
    *,
    cx: float,
    cy: float,
    r: float,
    r_inner: float = 0.0,
    t: Optional[int] = None,
    intensity_weighted: bool = True,
    gpu: bool = False,
) -> np.ndarray:
    """Virtual image from detected vectors (no raw dataset needed) — a
    passthrough to ``SpyDEDiffractionVectors.virtual_image_from_roi`` (same
    ``(kx, ky)`` data coordinates as the vectors). Returns a (nav_y, nav_x)
    array."""
    fn = vectors.virtual_image_from_roi_gpu if gpu else \
        vectors.virtual_image_from_roi
    return fn(cx, cy, r, r_inner=r_inner, t=t,
              intensity_weighted=intensity_weighted)
