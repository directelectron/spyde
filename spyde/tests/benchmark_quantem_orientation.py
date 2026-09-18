"""
benchmark_quantem_orientation.py — our vector orientation fit vs the vendored
quantem ACOM matcher, on the SAME real ``pyxem.data.sped_ag`` vectors.

  * vector OM   : our sparse batched pose fit (`compute_vector_orientation_gpu`)
  * quantem     : polar correlation match + pairwise refine
                  (`spyde.external.quantem`, driven through
                  `spyde.actions.vector_orientation_quantem`)

Both are handed the peaks OUR disk detection found, on the same region with the
same calibration, so the comparison is of the matching, not of the peak finding.

Two questions, and they must be answered in this order:

1. **Which quaternion convention reconciles them?** Both sides compose an
   in-plane spin onto a zone-axis orientation, but nothing establishes that
   their rotation runs the direction orix reads. So the script sweeps the
   candidates and reports IPF colour agreement for each — colour is the
   convention-safe metric, because both results render through the SAME
   `SpyDEOrientationMap.ipf_color_map`. A misorientation number cannot settle
   this; two different conventions produce two self-consistent fields.
2. **What does it cost?** Each stage is timed separately. Their refinement runs
   chunk-vectorized in float64 on the CPU by design, so the interesting number
   is wall clock at real scale, not on a 4x4 toy.

Run directly (NOT pytest — downloads the real dataset, uses the GPU)::

    uv run python -m spyde.tests.benchmark_quantem_orientation --ny 12 --nx 12
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

CIF = os.path.join(os.path.dirname(__file__), "Silver__0011135.cif")
# The true sped_ag reciprocal calibration (pyxem's default is exactly half).
SPED_AG_SCALE = 0.00668207597 * 4
SPED_AG_OFFSET = -0.374196254 * 4

#: Separates rounding from a real difference in what was computed. In float64
#: the main refinement pass is bit-exact against the vendored one; in float32
#: it lands a few 1e-5 degrees away.
#:
#: The neighbour-rescue pass is a different matter and will always show a small
#: diverged population. Upstream sweeps the outliers sequentially, writing each
#: result back into ``quats`` as it goes, so a later outlier reads whichever of
#: its neighbours have already been rescued. A batch cannot see its own
#: updates, so it works from the pre-rescue snapshot. Those positions are the
#: ones where several neighbour orientations score within the pass's 2 percent
#: acceptance margin, so the choice was near-arbitrary either way -- which is
#: why the report also referees them on upstream's own score.
PARITY_TOLERANCE_DEG = 0.05

#: Everything ``build_plan`` populates. Reusing it across maps is what lets the
#: two refinements start from an identical match instead of two of them.
_PLAN_ATTRS = (
    "device", "dtype", "cdtype", "corr_kernel_size", "sigma_excitation",
    "power_radial", "power_intensity", "zone_axes", "zone_quats",
    "zone_step_deg", "zone_nbr_idx", "zone_nbr_pos", "zone_nbr_valid",
    "plan_fft", "shell_radii", "num_gamma", "gamma", "detector_mask",
    "plan_norm_shift", "plan_frac_shift",
)


def _clone_with_plan(source, refine_device, refine_dtype=None):
    """A second map over the same plan and the same match, refined elsewhere."""
    import copy

    from spyde.actions.vector_orientation_quantem import orientation_map_class

    clone = orientation_map_class(refine_device, refine_dtype=refine_dtype).from_vectors(
        source.peaks, source.crystal, source.energy_ev)
    for attr in _PLAN_ATTRS:
        setattr(clone, attr, getattr(source, attr))
    clone.metadata = copy.deepcopy(source.metadata)
    clone.quats = source.quats.clone()
    clone.corr = source.corr.clone()
    clone.corr_second = source.corr_second.clone()
    clone.reliability = source.reliability.clone()
    clone.mirror = source.mirror.clone()
    return clone


def _pairing_score(om, chunk=512):
    """Total paired measured intensity of each position's best orientation.

    This is the quantity the rescue pass maximises, recomputed here in float64
    independently of either refinement so it can referee between them.
    """
    import torch

    from spyde.external.quantem.diffraction.rotations import quat_to_matrix

    refine = om.metadata["refine"]
    delta = float(refine["pair_distance"])
    sigma = float(refine["sigma_excitation"])
    rows, columns = om.quats.shape[:2]
    field_index = [om.peaks.fields.index(f) for f in ("qx", "qy", "intensity")]

    cells = [om.peaks[r, c].array for r, c in np.ndindex(rows, columns)]
    max_peaks = max(1, max(cell.shape[0] for cell in cells))
    coordinates = np.full((len(cells), max_peaks, 2), 1e6)
    weights = np.zeros((len(cells), max_peaks))
    for i, cell in enumerate(cells):
        n = cell.shape[0]
        if n == 0:
            continue
        coordinates[i, :n] = cell[:, field_index[:2]]
        w = np.clip(cell[:, field_index[2]], 0.0, None)
        weights[i, :n] = w / max(float(w.max()), 1e-12)
    coordinates = torch.as_tensor(coordinates)
    weights = torch.as_tensor(weights)

    reflections = om.crystal.g_vec
    wavelength = om.wavelength
    quats = om.quats[..., 0, :].reshape(-1, 4).to(torch.float64)
    scores = torch.zeros(quats.shape[0], dtype=torch.float64)
    for start in range(0, quats.shape[0], chunk):
        stop = min(start + chunk, quats.shape[0])
        g = torch.einsum("bij,gj->bgi", quat_to_matrix(quats[start:stop]), reflections)
        gz, g2 = g[..., 2], (g ** 2).sum(dim=-1)
        excitation = (2 * gz - wavelength * g2) / (2 - 2 * wavelength * gz)
        distance = torch.cdist(g[..., :2], coordinates[start:stop])
        nearest_distance, nearest = distance.min(dim=-1)
        paired = (torch.abs(excitation) < 2 * sigma) & (nearest_distance < delta)
        weight = torch.gather(weights[start:stop], 1, nearest) * (
            1 - nearest_distance / delta).clamp_min(0)
        scores[start:stop] = (weight * paired).sum(dim=1)
    return scores.reshape(rows, columns).numpy()


def _misorientation_deg(q1, q2, symmetry=None):
    """Per-pixel symmetry-reduced disorientation (deg) between two quat fields."""
    from orix.quaternion import Orientation
    from orix.quaternion.symmetry import Oh

    shape = q1.shape[:-1]
    o1 = Orientation(q1.reshape(-1, 4), symmetry=symmetry or Oh)
    o2 = Orientation(q2.reshape(-1, 4), symmetry=symmetry or Oh)
    try:
        angle = o1.angle_with(o2, degrees=True)
    except TypeError:                                    # older orix: radians
        angle = np.rad2deg(o1.angle_with(o2))
    return np.asarray(angle).reshape(shape)


def _torch_dtype(name):
    if name is None:
        return None
    import torch

    return {"float32": torch.float32, "float64": torch.float64}[name]


def _t(label, fn):
    start = time.time()
    out = fn()
    print(f"  [{time.time() - start:6.2f}s] {label}", flush=True)
    return out


def _calibrate(signal):
    for axis in signal.axes_manager.signal_axes:
        axis.scale = SPED_AG_SCALE
        axis.offset = SPED_AG_OFFSET
        axis.units = "1/A"
    return signal


def run(ny=12, nx=12, iy0=28, ix0=96, resolution=1.0, voltage=200.0,
        min_intensity=1e-4, zone_step=1.0, in_plane_step=5.0,
        precession_deg=0.0, device="cpu", refine_device="cpu",
        refine_parity=False, refine_dtype=None):
    import pyxem.data as pxd
    from orix.crystal_map import Phase

    from spyde.actions.find_vectors import _do_compute_vectors
    from spyde.actions.orientation_action import _reciprocal_radius
    from spyde.actions.orientation_compute import generate_library_from_phases
    from spyde.actions.vector_orientation import build_template_library
    from spyde.actions.vector_orientation_gpu import compute_vector_orientation_gpu
    from spyde.actions.vector_orientation_quantem import (
        PeaksAdapter, QUAT_CONVENTIONS, build_orientation_map, phase_to_crystal,
        to_result,
    )
    from spyde.reciprocal_units import axis_unit_factor
    from spyde.signals.orientation_map import phase_to_dict

    print(f"\n=== quantem vs vector-OM — sped_ag {ny}x{nx} @ ({iy0},{ix0}) ===",
          flush=True)
    phase = _t("Load Silver.cif", lambda: Phase.from_cif(CIF))
    signal = _t("Load sped_ag (lazy)",
                lambda: pxd.sped_ag(allow_download=True, lazy=True))
    _calibrate(signal)
    reciprocal_radius = _reciprocal_radius(signal)
    print(f"        recip_r={reciprocal_radius:.4f} 1/A", flush=True)

    region = signal.inav[ix0:ix0 + nx, iy0:iy0 + ny]
    region_eager = _t("materialise region", lambda: region.deepcopy())
    region_eager.data = np.asarray(region.data.compute(), np.float32)
    region_eager._lazy = False

    # ── peaks: OUR disk detection, shared by both matchers ───────────────────
    vectors = _t("find diffraction vectors",
                 lambda: _do_compute_vectors(
                     region_eager,
                     dict(sigma=1.0, kernel_radius=5, threshold=0.4,
                          min_distance=3, subpixel=True),
                     main_window=None, signal_tree=None))

    # ── our path ─────────────────────────────────────────────────────────────
    simulation = _t("our: generate template library",
                    lambda: generate_library_from_phases(
                        [phase], voltage, resolution, min_intensity,
                        reciprocal_radius))
    library = _t("our: build vector template library",
                 lambda: build_template_library(simulation, region_eager,
                                                r_max=reciprocal_radius))
    print(f"        {len(library.spots_xy)} templates", flush=True)
    ours = _t("our: compute_vector_orientation_gpu",
              lambda: compute_vector_orientation_gpu(vectors, library,
                                                     dict(strain_cap=0.05)))
    assert ours is not None

    # ── quantem path ─────────────────────────────────────────────────────────
    unit_factor = float(axis_unit_factor(region_eager) or 1.0)
    peaks = PeaksAdapter(vectors, inverse_angstrom_factor=unit_factor)
    counts = peaks.peak_counts
    print(f"        peaks/pattern: median={np.median(counts):.0f} "
          f"min={counts.min()} max={counts.max()} "
          f"({int((counts >= 5).sum())}/{len(counts)} pass their "
          f"min_number_peaks=5)", flush=True)

    crystal = _t("quantem: Phase -> Crystal",
                 lambda: phase_to_crystal(phase, name="Ag"))
    orientation_map = _t(
        "quantem: structure factors + orientation plan",
        lambda: build_orientation_map(
            peaks, crystal, energy_ev=voltage * 1e3,
            k_max=reciprocal_radius,
            angle_step_zone_axis_deg=zone_step,
            angle_step_in_plane_deg=in_plane_step,
            device=device, refine_device=refine_device, verbose=True))
    if precession_deg:
        orientation_map.metadata["precession_deg"] = float(precession_deg)
    print(f"        {orientation_map.zone_axes.shape[0]} zone axes x "
          f"{orientation_map.gamma.shape[0]} in-plane x "
          f"{orientation_map.shell_radii.shape[0]} shells", flush=True)

    _t("quantem: match_orientations",
       lambda: orientation_map.match_orientations(progress_bar=False))

    if refine_parity:
        # Same matched starting point, refined both ways. The override is a
        # transcription of upstream's method onto a device, so anything but
        # agreement here is a porting bug, not a tuning choice.
        cpu_map = _clone_with_plan(orientation_map, "cpu")
        gpu_map = _clone_with_plan(orientation_map, refine_device, refine_dtype)
        _t("quantem: refine_orientations (cpu, vendored)",
           lambda: cpu_map.refine_orientations(progress_bar=False))
        _t(f"quantem: refine_orientations ({refine_device}, override)",
           lambda: gpu_map.refine_orientations(progress_bar=False))
        drift = _misorientation_deg(
            cpu_map.quats[..., 0, :].cpu().numpy(),
            gpu_map.quats[..., 0, :].cpu().numpy())
        diverged = drift > PARITY_TOLERANCE_DEG
        print(f"  refine parity: median={np.median(drift):.2e} deg  "
              f"max={drift.max():.2e} deg  "
              f"above {PARITY_TOLERANCE_DEG} deg: {int(diverged.sum())}"
              f"/{drift.size} ({diverged.mean():.2%})", flush=True)
        if diverged.any():
            # Expected, and not a precision effect -- see PARITY_TOLERANCE_DEG.
            # These positions chose between genuinely different neighbour
            # orientations on a 2 percent margin, so the question is not
            # whether the two agree but whether ours is worse by upstream's
            # own criterion: the paired-intensity score the rescue maximises.
            # A porting bug makes most of them worse; a sequencing difference
            # splits them about evenly.
            cpu_score = _pairing_score(cpu_map)
            gpu_score = _pairing_score(gpu_map)
            worse = (gpu_score < cpu_score * 0.98) & diverged
            better = (gpu_score > cpu_score * 1.02) & diverged
            print(f"    of those, ours scores lower: {int(worse.sum())}, "
                  f"higher: {int(better.sum())}, level: "
                  f"{int(diverged.sum() - worse.sum() - better.sum())}",
                  flush=True)
        orientation_map = gpu_map
    else:
        _t(f"quantem: refine_orientations ({refine_device})",
           lambda: orientation_map.refine_orientations(progress_bar=False))

    phases_meta = [phase_to_dict(phase)]
    correlation = orientation_map.corr[..., 0].cpu().numpy()
    reliability = orientation_map.reliability.cpu().numpy()
    valid = correlation > 0
    print(f"\n  quantem correlation : median={np.median(correlation[valid]):.3f} "
          f"min={correlation[valid].min():.3f} max={correlation[valid].max():.3f}")
    print(f"  quantem reliability : median={np.median(reliability[valid]):.3f} "
          f"(best minus best-outside-15deg; our corr channel is currently "
          f"a 0/1 validity flag)")

    # ── compare, per candidate convention ────────────────────────────────────
    # IPF colour is the convention-safe metric: both results render through the
    # same SpyDEOrientationMap machinery, so equal orientation means equal
    # colour. Compare every axis — Z alone only proves the out-of-plane axis
    # agrees; X and Y agreeing too proves the in-plane angle does.
    # The requested region is clipped against the scan edge, so report the
    # shape actually fitted rather than the one asked for.
    both_valid = valid & (np.linalg.norm(ours.quats, axis=-1) > 0)
    fitted_ny, fitted_nx = ours.nav_shape
    print(f"\n  comparable pixels: {int(both_valid.sum())}/"
          f"{fitted_ny * fitted_nx} (region {fitted_ny}x{fitted_nx})", flush=True)

    results = {}
    for convention in QUAT_CONVENTIONS:
        theirs = to_result(orientation_map, phases_meta, convention=convention)
        line = [f"  convention={convention:<10}"]
        for direction in ("x", "y", "z"):
            ours_rgb = ours.ipf_color_map(direction).astype(float)
            theirs_rgb = theirs.ipf_color_map(direction).astype(float)
            difference = np.abs(ours_rgb - theirs_rgb).mean(-1)
            same = (difference[both_valid] < 25).mean()
            line.append(f"IPF-{direction.upper()} {same:>4.0%}")
        print("  ".join(line), flush=True)
        results[convention] = theirs

    best = max(QUAT_CONVENTIONS, key=lambda c: (
        np.abs(ours.ipf_color_map("z").astype(float)
               - results[c].ipf_color_map("z").astype(float)
               ).mean(-1)[both_valid] < 25).mean())
    print(f"\n  => best convention by IPF-Z: {best}", flush=True)
    return dict(ours=ours, theirs=results, orientation_map=orientation_map,
                vectors=vectors, valid=both_valid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ny", type=int, default=12)
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--iy0", type=int, default=28)
    parser.add_argument("--ix0", type=int, default=96)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--zone-step", type=float, default=1.0)
    parser.add_argument("--in-plane-step", type=float, default=5.0)
    parser.add_argument("--precession-deg", type=float, default=0.0,
                        help="sped_ag is a precession dataset; their library "
                             "can average the excitation envelope over it")
    parser.add_argument("--device", default="cpu",
                        help="device for the polar correlation (their plan)")
    parser.add_argument("--refine-device", default="cpu",
                        help="device for the pairwise refinement; anything but "
                             "cpu uses the override in vector_orientation_quantem")
    parser.add_argument("--refine-dtype", default=None,
                        choices=("float32", "float64"),
                        help="precision of the refinement override; float64 "
                             "removes the threshold flips in the rescue pass")
    parser.add_argument("--refine-parity", action="store_true",
                        help="refine the same match both ways and report the "
                             "disorientation between them")
    args = parser.parse_args()
    run(ny=args.ny, nx=args.nx, iy0=args.iy0, ix0=args.ix0,
        resolution=args.resolution, zone_step=args.zone_step,
        in_plane_step=args.in_plane_step,
        precession_deg=args.precession_deg, device=args.device,
        refine_device=args.refine_device, refine_parity=args.refine_parity,
        refine_dtype=_torch_dtype(args.refine_dtype))
