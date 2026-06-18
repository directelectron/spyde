"""
orientation_compute.py — batch orientation mapping compute for SpyDE.

Chunked template matching over a 4D-STEM scan, following the find_vectors
batch architecture (storage-aligned chunks, GPU/CPU lane dispatch, live shm
preview) but simpler: patterns are independent, so there are no ghost zones
and no rechunk shuffles.

The live preview buffer holds the IPF **RGB** maps for all three sample
directions stacked channel-wise — (nav_y, nav_x, 9) = X RGB | Y RGB | Z RGB —
so the GUI can paint the X/Y/Z orientation maps in chunk by chunk.

No Qt imports in this module: the chunk function runs on dask workers and
everything here is unit-testable headless.
"""
from __future__ import annotations

import functools
import time
from typing import Optional

import numpy as np

from spyde.signals.orientation_map import (
    SpyDEOrientationMap, phase_to_dict,
)

# pyxem OrientationMap column convention, used throughout:
COL_LIB_IDX = 0   # global template index
COL_CORR = 1      # correlation score
COL_ANGLE = 2     # in-plane rotation, degrees in [-180, 180)
COL_MIRROR = 3    # +-1 template mirror factor


# ─────────────────────────────────────────────────────────────────────────────
# Library / cache helpers (pure — shared with the GUI caret)
# ─────────────────────────────────────────────────────────────────────────────

def generate_library_from_phases(phases, accelerating_voltage, resolution,
                                 minimum_intensity, reciprocal_radius,
                                 max_excitation_error=0.1):
    """Generate a diffsims Simulation2D library from orix Phase objects."""
    from diffsims.generators.simulation_generator import SimulationGenerator
    from orix.sampling import get_sample_reduced_fundamental

    generator = SimulationGenerator(
        accelerating_voltage, minimum_intensity=minimum_intensity
    )
    rotations = [
        get_sample_reduced_fundamental(
            resolution=resolution, point_group=phase.point_group
        )
        for phase in phases
    ]
    return generator.calculate_diffraction2d(
        phases if len(phases) > 1 else phases[0],
        rotation=rotations if len(rotations) > 1 else rotations[0],
        max_excitation_error=max_excitation_error,
        reciprocal_radius=reciprocal_radius,
        with_direct_beam=False,
    )


def build_matching_cache(signal, sim) -> dict:
    """
    Pre-compute polar slices and templates (geometry- and library-dependent
    only).  Identical maths to the refine-step cache in actions/pyxem.py.
    """
    from pyxem.utils.indexation_utils import (
        _get_integrated_polar_templates, _norm_rows,
    )

    NR, NA = 100, 360
    slices, factors, factors_slice, radial_range = \
        signal.calibration.get_slices2d(NR, NA)

    r0, r1 = float(radial_range[0]), float(radial_range[1])
    radial_axis = r0 + (r1 - r0) / NR * np.arange(NR)
    azim_axis = np.linspace(-np.pi, np.pi, NA, endpoint=False)

    r_templates, theta_templates, intensities_templates = \
        sim.polar_flatten_simulations(
            radial_axes=radial_axis, azimuthal_axes=azim_axis,
        )
    integrated = _get_integrated_polar_templates(
        NR, r_templates, intensities_templates, True
    )
    intensities_raw = intensities_templates.copy().astype(float)
    intensities_norm = _norm_rows(intensities_raw.copy())

    return {
        "slices": slices,
        "factors": factors,
        "factors_slice": factors_slice,
        "r_templates": r_templates,
        "theta_templates": theta_templates,
        "intensities_norm": intensities_norm,
        "intensities_raw": intensities_raw,
        "integrated": integrated,
        "NR": NR, "NA": NA,
    }


def template_tables(sim):
    """
    (template_quats (n_templates, 4) float64, template_phase (n_templates,)
    int16) flattened in the same phase-major order as
    polar_flatten_simulations / sim.get_simulation.
    """
    rots = sim.rotations
    # Multiphase sims store rotations as a numpy OBJECT array of per-phase
    # orix Rotation instances; single-phase sims store the Rotation directly.
    if isinstance(rots, np.ndarray) and rots.dtype == object:
        rot_list = list(rots.flat)
    elif isinstance(rots, (list, tuple)):
        rot_list = list(rots)
    else:
        rot_list = [rots]
    quats = np.concatenate(
        [np.atleast_2d(r.data) for r in rot_list], axis=0
    ).astype(np.float64)
    phase_of = np.concatenate([
        np.full(int(np.atleast_2d(r.data).shape[0]), i, dtype=np.int16)
        for i, r in enumerate(rot_list)
    ])
    return quats, phase_of


def sim_phases_list(sim):
    """Phases of a simulation as a list (single-phase sims store a scalar)."""
    phases = sim.phases
    if hasattr(phases, "__len__") and not hasattr(phases, "point_group"):
        return list(phases)
    return [phases]


def best_match_spots(pattern_data, sim, matching_cache, *, gamma: float = 0.5,
                     max_radius: float | None = None,
                     normalize_templates: bool = True,
                     scale_override: float | None = None,
                     original_scale: float | None = None,
                     min_intensity: float = 0.0) -> np.ndarray:
    """Simulated diffraction spots (Å⁻¹, centred on the direct beam) for the
    best-matching template of a SINGLE pattern.

    Qt-free port of ``actions.pyxem._get_best_fit_spots`` fast path (the matching
    cache makes it ~5 ms/call) — used to overlay the matched template on the live
    diffraction pattern as the navigator moves. Returns an ``(N, 2)`` array of
    ``[kx, ky]`` spot coordinates; the same flip/rotate/mirror as
    ``vectors_from_orientation_map`` so the spots line up with the data.

    Refine knobs (Qt "3 Refine" tab): ``scale_override`` rescales the template
    coords by ``scale_override/original_scale`` (correct a miscalibrated camera
    length); ``min_intensity`` (0–1, fraction of the brightest spot) drops faint
    spots.
    """
    from pyxem.utils.indexation_utils import _mixed_matching_lib_to_polar
    from pyxem.utils._azimuthal_integrations import _slice_radial_integrate

    slices = matching_cache["slices"]
    factors = matching_cache["factors"]
    factors_slice = matching_cache["factors_slice"]
    r_tmpl = matching_cache["r_templates"]
    theta_tmpl = matching_cache["theta_templates"]
    int_norm = matching_cache["intensities_norm"]
    integrated = matching_cache["integrated"]
    NR, NA = matching_cache["NR"], matching_cache["NA"]

    pattern_data = np.asarray(pattern_data, dtype=float)
    polar = _slice_radial_integrate(
        pattern_data, factors, factors_slice, slices, NR, NA, mean=True
    )
    polar = np.nan_to_num(polar ** gamma).T.astype(float)   # (NA, NR)

    int_templates = int_norm if normalize_templates else matching_cache["intensities_raw"]
    result = _mixed_matching_lib_to_polar(
        polar,
        integrated_templates=integrated,
        r_templates=r_tmpl,
        theta_templates=theta_tmpl,
        intensities_templates=int_templates,
        n_keep=None, frac_keep=1.0, n_best=integrated.shape[0], transpose=False,
    )
    row = result[0]                       # best match (sorted desc by corr)
    lib_idx = int(row[0])
    rot_idx = int(row[2])
    mirror = float(row[3])

    _rot, _phase_idx, coords_dv = sim.get_simulation(lib_idx)
    raw = coords_dv.data[:, :2].copy().astype(float)
    inten = np.array(coords_dv.intensity, dtype=float)

    # vectors_from_orientation_map: flip y, rotate by mirror*angle, negate, mirror*y
    angle_deg = rot_idx / NA * 360.0 - 180.0
    a = np.deg2rad(mirror * angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    rx = raw[:, 0]; ry = -raw[:, 1]
    kx = rx * cos_a - ry * sin_a
    ky = rx * sin_a + ry * cos_a
    kx, ky = -kx, -ky
    ky = mirror * ky
    coords = np.stack([kx, ky], axis=1)

    # Scale refine: rescale template coords to where spots actually land.
    if scale_override and original_scale:
        coords = coords * (float(scale_override) / float(original_scale))

    # Min-intensity refine: drop spots fainter than a fraction of the brightest.
    if min_intensity > 0.0 and len(inten) > 0:
        imax = float(inten.max()) or 1.0
        keep = (inten / imax) >= float(min_intensity)
        coords = coords[keep]
        inten = inten[keep]

    if max_radius is not None and len(coords) > 0:
        keep = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2) <= max_radius
        coords = coords[keep]
        inten = inten[keep]
    return coords


def resolve_quaternions(result4: np.ndarray, template_quats: np.ndarray
                        ) -> np.ndarray:
    """
    Resolve [lib_idx, corr, angle_deg, mirror] rows into full orientation
    quaternions, replicating pyxem's rotation_from_orientation_map exactly:
    euler(template) * mirror, then euler[0] <- in-plane angle.

    result4 : (..., 4) — any leading shape
    Returns (..., 4) float32 quaternions (w, x, y, z).
    """
    from orix.quaternion import Orientation

    flat = result4.reshape(-1, 4)
    idx = np.clip(flat[:, COL_LIB_IDX].astype(int), 0,
                  len(template_quats) - 1)
    ori = template_quats[idx]
    euler = Orientation(ori).to_euler(degrees=True) \
        * flat[:, COL_MIRROR][..., np.newaxis]
    euler[:, 0] = flat[:, COL_ANGLE]
    quats = Orientation.from_euler(euler, degrees=True).data
    return quats.reshape(result4.shape[:-1] + (4,)).astype(np.float32)


def _chunk_ipf_rgb(result4: np.ndarray, template_quats: np.ndarray,
                   template_phase: np.ndarray, phases_meta: list,
                   direction: str = "z") -> np.ndarray:
    """(ny, nx, 3) uint8 IPF colors for a chunk's best matches."""
    best = result4[..., 0, :]
    quats = resolve_quaternions(best, template_quats)
    pidx = template_phase[
        np.clip(best[..., COL_LIB_IDX].astype(int), 0,
                len(template_phase) - 1)
    ]
    om = SpyDEOrientationMap(
        quats=quats[..., np.newaxis, :],
        corr=best[..., COL_CORR][..., np.newaxis].astype(np.float32),
        phase_idx=pidx[..., np.newaxis].astype(np.int16),
        mirror=best[..., COL_MIRROR][..., np.newaxis].astype(np.int8),
        phases=phases_meta,
    )
    return om.ipf_color_map(direction)


# ─────────────────────────────────────────────────────────────────────────────
# Chunk function (runs on workers)
# ─────────────────────────────────────────────────────────────────────────────

def _match_chunk(
    block: np.ndarray,
    cache: dict,
    block_info=None,
    n_best: int = 5,
    gamma: float = 0.5,
    normalize_templates: bool = True,
    rot_mask=None,
    template_quats=None,
    template_phase=None,
    phases_meta=None,
    shm_name: str = None,
    nav_2d_shape: tuple = None,
) -> np.ndarray:
    """
    Template-match every pattern in a (ny, nx, KY, KX) block.

    Returns (ny, nx, n_best, 4) float32 rows [lib_idx, corr, angle_deg,
    mirror] (pyxem OrientationMap layout, angle already in degrees).

    When shm_name is set, also writes this block's IPF RGB for the X, Y and
    Z directions into the live (nav_y, nav_x, 9) shared-memory buffer at the
    block's location so the GUI sees all three orientation maps paint in.
    """
    # Zero-size blocks: dask meta inference calls chunk fns on empty arrays
    # in the CLIENT process — return the right structure, do no work.
    if block.size == 0:
        return np.empty((0, 0, n_best, 4), dtype=np.float32)

    from pyxem.utils.indexation_utils import _mixed_matching_lib_to_polar
    from pyxem.utils._azimuthal_integrations import _slice_radial_integrate

    slices = cache["slices"]
    factors = cache["factors"]
    factors_slice = cache["factors_slice"]
    r_tmpl = cache["r_templates"]
    theta_tmpl = cache["theta_templates"]
    integrated = cache["integrated"]
    int_templates = (cache["intensities_norm"] if normalize_templates
                     else cache["intensities_raw"])
    NR, NA = cache["NR"], cache["NA"]

    if rot_mask is not None and np.asarray(rot_mask).any():
        mask_idx = np.where(np.asarray(rot_mask))[0]
        integrated = integrated[mask_idx]
        r_tmpl = r_tmpl[mask_idx]
        theta_tmpl = theta_tmpl[mask_idx]
        int_templates = int_templates[mask_idx]
    else:
        mask_idx = None

    n_templates = integrated.shape[0]
    k = min(int(n_best), n_templates)

    ny, nx = block.shape[0], block.shape[1]
    out = np.zeros((ny, nx, int(n_best), 4), dtype=np.float32)
    out[..., COL_MIRROR] = 1.0

    for iy in range(ny):
        for ix in range(nx):
            pattern = np.asarray(block[iy, ix], dtype=float)
            polar = _slice_radial_integrate(
                pattern, factors, factors_slice, slices, NR, NA, mean=True
            )
            polar = np.nan_to_num(polar ** gamma).T.astype(float)
            result = _mixed_matching_lib_to_polar(
                polar,
                integrated_templates=integrated,
                r_templates=r_tmpl,
                theta_templates=theta_tmpl,
                intensities_templates=int_templates,
                n_keep=None, frac_keep=1.0, n_best=k, transpose=False,
            )
            rows = np.atleast_2d(result)[:k]
            lib = rows[:, 0].astype(int)
            if mask_idx is not None:
                lib = mask_idx[lib]
            out[iy, ix, :k, COL_LIB_IDX] = lib
            out[iy, ix, :k, COL_CORR] = rows[:, 1]
            out[iy, ix, :k, COL_ANGLE] = rows[:, 2] / NA * 360.0 - 180.0
            out[iy, ix, :k, COL_MIRROR] = rows[:, 3]

    # ── Live IPF RGB preview (X | Y | Z stacked channel-wise) ────────────────
    if shm_name is not None and block_info and 0 in block_info:
        try:
            loc = block_info[0]["array-location"]
            ys, xs = loc[0], loc[1]
            from multiprocessing import shared_memory as _shm_mod
            shm = _shm_mod.SharedMemory(name=shm_name, create=False)
            try:
                buf = np.ndarray(tuple(nav_2d_shape) + (9,),
                                 dtype=np.float32, buffer=shm.buf)
                for di, direction in enumerate(("x", "y", "z")):
                    rgb = _chunk_ipf_rgb(out, template_quats, template_phase,
                                         phases_meta, direction)
                    buf[ys[0]:ys[1], xs[0]:xs[1], 3 * di:3 * di + 3] = \
                        rgb.astype(np.float32)
                del buf
            finally:
                shm.close()
        except Exception:
            pass

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Batch driver
# ─────────────────────────────────────────────────────────────────────────────

def _do_compute_orientations(
    signal,
    sim,
    params: dict,
    main_window,
    signal_tree,
    shm_name: Optional[str] = None,
    stopped_flag=None,
    cache: Optional[dict] = None,
) -> Optional[SpyDEOrientationMap]:
    """
    Batch orientation mapping over a (nav_y, nav_x, KY, KX) signal.

    params: n_best (int), gamma (float), normalize_templates (bool),
            rot_mask (bool array over templates | None).

    Same submission strategy as find_vectors: GPU/CPU lane dispatch when a
    distributed client with a designated GPU worker is reachable, single
    future otherwise, local scheduler without a client (tests).  NEVER
    computes the full dataset (per-chunk slices only).
    """
    import dask.array as da

    tic = time.time()
    nav_dim = signal.axes_manager.navigation_dimension
    if nav_dim != 2:
        raise NotImplementedError(
            "Orientation mapping currently supports 2D navigation only"
        )
    sig_dim = signal.axes_manager.signal_dimension

    if cache is None:
        cache = build_matching_cache(signal, sim)
    template_quats, template_phase = template_tables(sim)
    phases_meta = [phase_to_dict(p) for p in sim_phases_list(sim)]

    n_best = int(params.get("n_best", 5))
    gamma = float(params.get("gamma", 0.5))
    normalize_templates = bool(params.get("normalize_templates", True))
    rot_mask = params.get("rot_mask")

    raw = signal.data
    nav_shape = tuple(raw.shape[:nav_dim])

    # ── Chunked dask array: storage-aligned, no ghosts needed ────────────────
    if isinstance(raw, np.ndarray):
        frame_mb = raw.shape[-2] * raw.shape[-1] * 4 / 1e6
        target = max(4, int(np.sqrt(96.0 / max(frame_mb, 1e-3))))
        chunks = (min(target, nav_shape[0]), min(target, nav_shape[1])) \
            + raw.shape[nav_dim:]
        da_data = da.from_array(raw, chunks=chunks)
    else:
        sig_ok = all(len(c) == 1 for c in raw.chunks[nav_dim:])
        if sig_ok:
            da_data = raw
        else:
            da_data = raw.rechunk(
                raw.chunks[:nav_dim] + tuple(raw.shape[nav_dim:])
            )

    # ── Client / lanes (shared helpers; same env policy as find_vectors) ─────
    client = None
    if signal_tree is not None:
        client = getattr(signal_tree, "client", None)
    if client is None and main_window is not None:
        client = getattr(getattr(main_window, "dask_manager", None),
                         "client", None)

    # Big read-only payloads: scatter once per cluster instead of pickling
    # them into every task.
    cache_arg = cache
    if client is not None:
        try:
            cache_arg = client.scatter(cache, broadcast=True)
        except Exception:
            cache_arg = cache

    chunk_fn = functools.partial(
        _match_chunk,
        n_best=n_best,
        gamma=gamma,
        normalize_templates=normalize_templates,
        rot_mask=rot_mask,
        template_quats=template_quats,
        template_phase=template_phase,
        phases_meta=phases_meta,
        shm_name=shm_name,
        nav_2d_shape=tuple(nav_shape),
    )

    sig_axes_idx = list(range(nav_dim, nav_dim + sig_dim))
    result_da = da.map_blocks(
        chunk_fn,
        da_data,
        cache_arg,
        dtype=np.float32,
        drop_axis=sig_axes_idx,
        new_axis=[nav_dim, nav_dim + 1],
        chunks=da_data.chunks[:nav_dim] + ((n_best,), (4,)),
        meta=np.empty((0,) * (nav_dim + 2), dtype=np.float32),
    )
    print(f"[orientation] graph built in {time.time() - tic:.1f} s "
          f"({len(template_quats)} templates, n_best={n_best})")

    if stopped_flag is not None and stopped_flag[0]:
        return None

    # ── Submit ────────────────────────────────────────────────────────────────
    tic = time.time()
    if client is not None:
        from spyde.compute_dispatch import split_workers_for_gpu, \
            dispatch_chunks
        gpu_addrs, cpu_addrs = split_workers_for_gpu(client)
        if gpu_addrs and cpu_addrs:
            result4 = dispatch_chunks(
                client, result_da, nav_dim, gpu_addrs, cpu_addrs,
                stopped_flag=stopped_flag, fill_value=0.0,
                label="orientation",
            )
            if result4 is None:
                return None
        else:
            future = client.compute(result_da)
            while not future.done():
                if stopped_flag is not None and stopped_flag[0]:
                    try:
                        future.cancel()
                    except Exception:
                        pass
                    return None
                time.sleep(0.1)
            result4 = future.result()
    else:
        # Pin the threaded scheduler: a bare .compute() silently runs on any
        # ambient distributed Client (dask makes a live Client the global
        # default), changing where — and with what thread count — the
        # matcher's parallel tie-breaking executes.
        result4 = result_da.compute(scheduler="threads")
    print(f"[orientation] matched {nav_shape[0] * nav_shape[1]} patterns "
          f"in {time.time() - tic:.1f} s")

    if stopped_flag is not None and stopped_flag[0]:
        return None

    # ── Resolve quaternions and build the container ──────────────────────────
    quats = resolve_quaternions(result4, template_quats)
    lib = np.clip(result4[..., COL_LIB_IDX].astype(int), 0,
                  len(template_phase) - 1)
    om = SpyDEOrientationMap(
        quats=quats,
        corr=result4[..., COL_CORR].astype(np.float32),
        phase_idx=template_phase[lib].astype(np.int16),
        mirror=result4[..., COL_MIRROR].astype(np.int8),
        phases=phases_meta,
        nav_axes=list(signal.axes_manager.navigation_axes),
        params=dict(params),
    )

    # Final authoritative shm write (covers any chunk whose live write failed)
    if shm_name is not None:
        try:
            from multiprocessing import shared_memory as _shm_mod
            shm = _shm_mod.SharedMemory(name=shm_name, create=False)
            try:
                buf = np.ndarray(tuple(nav_shape) + (9,), dtype=np.float32,
                                 buffer=shm.buf)
                for di, direction in enumerate(("x", "y", "z")):
                    buf[..., 3 * di:3 * di + 3] = \
                        om.ipf_color_map(direction).astype(np.float32)
                del buf
            finally:
                shm.close()
        except Exception:
            pass

    return om
