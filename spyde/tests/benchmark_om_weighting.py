"""
benchmark_om_weighting.py — validate the vector-OM reflection weighting on the
real ``pyxem.data.sped_ag`` scan.

Computes the dense full-pattern OM ONCE as an independent reference, then runs
the sparse vector OM under several (gamma, k_power) weightings and reports how
well each agrees with the reference (IPF %same colour + misorientation scatter).

Background: the vector-OM refine weights each template spot's residual by
``(gI**gamma)·(|g|**k_power)·conf``.
  * gamma   compresses peak intensity (like the raw-OM gamma) so bright low-k
            beams don't dominate. Lower gamma lets the dim high-k reflections —
            whose larger |g| lever arm pins the orientation more tightly — carry
            their proper weight.
  * k_power adds an EXPLICIT reciprocal lever arm on top.

Finding on sped_ag (18×28 slab): gamma=0.5 tightens the orientation scatter vs
the full-pattern reference from ~8.4° (gamma=1, intensity-dominated) to ~7.4°
(~12%); the explicit k_power lever arm was neutral (the |g|² leverage is already
implicit once gamma flattens the weights). Hence the defaults gamma=0.5,
k_power=0.0.

Run directly (NOT pytest — downloads + processes the real dataset, uses the GPU):

    uv run python -m spyde.tests.benchmark_om_weighting
"""
from __future__ import annotations

import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

from spyde.tests.benchmark_om_parity import (
    _calibrate, _vectors_from_region, _misorientation_deg, CIF)

NY, NX, IY0, IX0 = 18, 28, 20, 80     # a multi-grain slab of sped_ag
SETTINGS = [(1.0, 0.0), (0.5, 0.0), (1.0, 1.0), (0.5, 1.0), (0.3, 1.5)]


def main(ny=NY, nx=NX):
    import pyxem.data as pxd
    from orix.crystal_map import Phase
    from spyde.actions.orientation_compute import (
        generate_library_from_phases, _do_compute_orientations)
    from spyde.actions.orientation_action import _reciprocal_radius
    from spyde.actions.vector_orientation import build_template_library
    from spyde.actions.vector_orientation_gpu import compute_vector_orientation_gpu
    from dask.distributed import Client, LocalCluster

    phase = Phase.from_cif(CIF)
    s = _calibrate(pxd.sped_ag(allow_download=True, lazy=True))
    recip_r = _reciprocal_radius(s)
    region = s.inav[IX0:IX0 + nx, IY0:IY0 + ny]
    region_eager = region.deepcopy()
    region_eager.data = np.asarray(region.data.compute(), np.float32)
    region_eager._lazy = False
    sim = generate_library_from_phases([phase], 200.0, 1.0, 1e-4, recip_r)
    n_t = int(np.asarray(sim.rotations.data).reshape(-1, 4).shape[0])
    print(f"region {ny}x{nx}  templates={n_t}", flush=True)

    class _DM:
        def __init__(s_, c): s_.client = c; s_.gpu_worker_address = None; s_.heavy_workers = None
    class _MW:
        def __init__(s_, c): s_.dask_manager = _DM(c)

    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True)
    client = Client(cluster)
    try:
        om = _do_compute_orientations(
            region, sim, dict(n_best=5, gamma=1.0, normalize_templates=True),
            main_window=_MW(client), signal_tree=None)
    finally:
        client.close(); cluster.close()
    q_raw = np.asarray(om.quats, np.float64)
    if q_raw.ndim == 4:
        q_raw = q_raw[..., 0, :]

    vecs = _vectors_from_region(region_eager, dict(
        sigma=1.0, kernel_radius=5, threshold=0.4, min_distance=3, subpixel=True))
    lib = build_template_library(sim, region_eager, r_max=recip_r)

    def agree(res):
        q_vec = np.asarray(res.quats, np.float64)
        valid = (np.linalg.norm(q_raw, axis=-1) > 0) & (np.linalg.norm(q_vec, axis=-1) > 0)
        same = {d: (np.abs(om.ipf_color_map(d).astype(float)
                          - res.ipf_color_map(d).astype(float)).mean(-1)[valid] < 25).mean()
                for d in ("x", "y", "z")}
        mis = _misorientation_deg(q_raw, q_vec)[valid]
        return same, float(np.nanstd(mis))

    print(f"\n{'gamma':>6} {'k_pow':>6} | {'IPF-X':>6} {'IPF-Y':>6} {'IPF-Z':>6} | {'mis std':>8}", flush=True)
    print("-" * 56, flush=True)
    for gamma, kpow in SETTINGS:
        res = compute_vector_orientation_gpu(
            vecs, lib, dict(strain_cap=0.05, gamma=gamma, k_power=kpow))
        same, mstd = agree(res)
        tag = "  <- default" if (gamma, kpow) == (0.5, 0.0) else ""
        print(f"{gamma:6.1f} {kpow:6.1f} | {same['x']:6.0%} {same['y']:6.0%} "
              f"{same['z']:6.0%} | {mstd:7.1f}°{tag}", flush=True)


if __name__ == "__main__":
    main()
