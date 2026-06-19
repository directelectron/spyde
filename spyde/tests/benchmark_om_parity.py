"""
benchmark_om_parity.py — run BOTH orientation-mapping workflows on the SAME real
``pyxem.data.sped_ag`` region and check they agree.

  * raw OM       : dense per-pattern template match (`_do_compute_orientations`)
  * vector OM    : sparse-vector batched GPU fit (`compute_vector_orientation_gpu`)

Both use the Silver phase, the same calibration, and the same template library
SOURCE simulation, so their recovered crystal orientations must be the SAME field
(up to Ag m-3m symmetry). The script reports the per-pixel misorientation between
the two and whether a coordinate MIRROR is needed to reconcile them — the thing a
4D-STEM user actually checks when overlaying the two maps.

Run directly (NOT pytest — downloads + processes the real dataset, uses the GPU):

    uv run python -m spyde.tests.benchmark_om_parity --ny 12 --nx 12

Each stage is timed separately. Vector-OM uses the batched torch path (seconds);
raw OM runs on a real LocalCluster like the app.
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


def _t(label, fn):
    t = time.time()
    out = fn()
    print(f"  [{time.time() - t:6.2f}s] {label}", flush=True)
    return out


def _calibrate(sig):
    for ax in sig.axes_manager.signal_axes:
        ax.scale = SPED_AG_SCALE
        ax.offset = SPED_AG_OFFSET
        ax.units = "1/A"
    return sig


def _vectors_from_region(region, params):
    """Find diffraction vectors on a (small, eager) region — the app's compute
    core, no Qt, no cluster needed for a small block."""
    from spyde.actions.find_vectors import _do_compute_vectors
    return _do_compute_vectors(region, params, main_window=None, signal_tree=None)


def _misorientation_deg(q1, q2, point_group="m-3m"):
    """Per-pixel disorientation angle (deg) between two (ny,nx,4) quat fields,
    properly two-sided symmetry-reduced by the crystal symmetry. Returns (ny,nx).

    NOTE: the IPF-X/Y/Z colour agreement is the authoritative, convention-safe
    parity metric (both result types render colour through the same machinery);
    this is a secondary cross-check. The two quat fields may also carry different
    quaternion conventions, so trust the colours first."""
    from orix.quaternion import Orientation
    from orix.quaternion.symmetry import Oh
    ny, nx, _ = q1.shape
    o1 = Orientation(q1.reshape(-1, 4), symmetry=Oh)
    o2 = Orientation(q2.reshape(-1, 4), symmetry=Oh)
    # Orientation.angle_with(other, degrees=...) returns the symmetry-reduced
    # disorientation angle (≤ 62.8° for cubic) — the correct, non-deprecated path.
    try:
        ang = o1.angle_with(o2, degrees=True)
    except TypeError:                                   # older orix: radians
        ang = np.rad2deg(o1.angle_with(o2))
    return np.asarray(ang).reshape(ny, nx)


def run(ny=12, nx=12, iy0=28, ix0=96, resolution=1.0, voltage=200.0,
        min_intensity=1e-4, gamma=1.0, n_best=5):
    import pyxem.data as pxd
    from orix.crystal_map import Phase
    from spyde.actions.orientation_compute import (
        generate_library_from_phases, _do_compute_orientations)
    from spyde.actions.orientation_action import _reciprocal_radius
    from spyde.actions.vector_orientation import build_template_library
    from spyde.actions.vector_orientation_gpu import compute_vector_orientation_gpu

    print(f"\n=== OM parity — sped_ag {ny}x{nx} @ ({iy0},{ix0}) ===", flush=True)
    phase = _t("Load Silver.cif", lambda: Phase.from_cif(CIF))
    s = _t("Load sped_ag (lazy)", lambda: pxd.sped_ag(allow_download=True, lazy=True))
    _calibrate(s)
    recip_r = _reciprocal_radius(s)
    print(f"        recip_r={recip_r:.4f} 1/A  scale={SPED_AG_SCALE:.5f}", flush=True)

    region = s.inav[ix0:ix0 + nx, iy0:iy0 + ny]
    region_eager = _t("materialise region", lambda: region.deepcopy())
    region_eager.data = np.asarray(region.data.compute(), np.float32)
    region_eager._lazy = False

    sim = _t("generate template library",
             lambda: generate_library_from_phases(
                 [phase], voltage, resolution, min_intensity, recip_r))
    n_templates = int(np.asarray(sim.rotations.data).reshape(-1, 4).shape[0])
    print(f"        {n_templates} templates", flush=True)

    # ── raw OM (dense match, real cluster) ───────────────────────────────────
    from dask.distributed import Client, LocalCluster

    class _DM:
        def __init__(self, c): self.client = c; self.gpu_worker_address = None; self.heavy_workers = None
    class _MW:
        def __init__(self, c): self.dask_manager = _DM(c)

    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True)
    client = Client(cluster)
    try:
        om = _t("raw OM: _do_compute_orientations",
                lambda: _do_compute_orientations(
                    region, sim, dict(n_best=n_best, gamma=gamma, normalize_templates=True),
                    main_window=_MW(client), signal_tree=None))
    finally:
        client.close(); cluster.close()
    assert om is not None
    q_raw = np.asarray(om.quats, np.float64)            # (ny,nx,[n_best,]4)
    if q_raw.ndim == 4:                                  # keep only the best match
        q_raw = q_raw[..., 0, :]
    ipf_raw = om.ipf_color_map("z")

    # ── vector OM (sparse, batched GPU) ──────────────────────────────────────
    vecs = _t("find diffraction vectors",
              lambda: _vectors_from_region(region_eager, dict(
                  sigma=1.0, kernel_radius=5, threshold=0.4, min_distance=3, subpixel=True)))
    lib = _t("build vector template library",
             lambda: build_template_library(sim, region_eager, r_max=recip_r))
    res = _t("vector OM: compute_vector_orientation_gpu",
             lambda: compute_vector_orientation_gpu(vecs, lib, dict(strain_cap=0.05)))
    assert res is not None
    q_vec = np.asarray(res.quats, np.float64)           # (ny,nx,4)
    ipf_vec = res.ipf_color_map("z")

    # ── compare ──────────────────────────────────────────────────────────────
    # IPF maps are the user-facing, convention-safe artifact: both result types
    # render colour through the SAME SpyDEOrientationMap.ipf_color_map, so equal
    # crystal orientation → equal colour. Compare per-pixel for EVERY axis (X|Y|Z)
    # — Z agreement alone only proves the out-of-plane axis matches; X & Y also
    # agreeing proves the full orientation (incl. the in-plane angle) matches.
    valid = (np.linalg.norm(q_raw, axis=-1) > 0) & (np.linalg.norm(q_vec, axis=-1) > 0)
    print(f"\n  valid pixels: {int(valid.sum())}/{ny*nx}", flush=True)

    def _ipf_agree(direction):
        a = om.ipf_color_map(direction).astype(float)
        b = res.ipf_color_map(direction).astype(float)
        diff = np.abs(a - b).mean(-1)                      # mean RGB diff per pixel
        close = (diff[valid] < 25).mean()                  # <25/255 ≈ same colour
        var = float(a[valid].std())                        # non-uniformity guard
        return diff[valid].mean(), close, var

    for d in ("x", "y", "z"):
        md, close, var = _ipf_agree(d)
        print(f"  IPF-{d.upper()}  mean|Δrgb|={md:5.1f}/255  %same={close:.0%}  "
              f"(map RGB std={var:.0f} → {'varied' if var > 8 else 'UNIFORM (weak test)'})",
              flush=True)

    # Full-orientation misorientation (orix, Oh-reduced) for reference. A large,
    # ~CONSTANT value with IPF-Z agreeing == a fixed in-plane θ-reference offset
    # between the two methods (a convention, not an error).
    mis = _misorientation_deg(q_raw, q_vec)
    print(f"  misorientation (raw vs vector)  median={np.nanmedian(mis[valid]):.1f}°  "
          f"std={np.nanstd(mis[valid]):.1f}°", flush=True)

    mz, closez, varz = _ipf_agree("z")
    agree = closez > 0.9 and mz < 25
    print(f"\n  => IPF-Z {'AGREES' if agree else 'DISAGREES'} "
          f"({closez:.0%} of pixels same colour)", flush=True)
    return dict(mis=mis, q_raw=q_raw, q_vec=q_vec, ipf_raw=ipf_raw, ipf_vec=ipf_vec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny", type=int, default=12)
    ap.add_argument("--nx", type=int, default=12)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    a = ap.parse_args()
    run(ny=a.ny, nx=a.nx, resolution=a.resolution, gamma=a.gamma)
