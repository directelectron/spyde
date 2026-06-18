"""
benchmark_orientation_ag.py — recreate the Qt "Ag Silver" Orientation-Mapping
workflow end-to-end on the real ``pyxem.data.sped_ag`` SpEd Ag 4D-STEM scan,
mirroring the 4-tab wizard:

  1 Load     — load Silver__0011135.cif → orix Phase; accelerating voltage.
  2 Library  — generate the diffsims template library (angle resolution,
               minimum intensity) + the matching cache.
  3 Refine   — single-pattern template match under the "crosshair"
               (``best_match_spots``) with the gamma / scale sliders, returning
               the best-fit template spots to overlay on the diffraction pattern.
  4 Run      — full-field template match (``_do_compute_orientations``) over a
               navigation region → SpyDEOrientationMap + IPF-Z colour map.

Run directly (NOT under pytest — it downloads + processes the real dataset):

    uv run python -m spyde.tests.benchmark_orientation_ag
    uv run python -m spyde.tests.benchmark_orientation_ag --ny 16 --nx 16

Each stage is timed separately (per the benchmarking guidance) so a slow stage
is obvious. The first ``best_match_spots`` pays a one-time pyxem/numba JIT cost.
"""
from __future__ import annotations

import argparse
import os
import time

# Force single-threaded numba/BLAS BEFORE importing pyxem (which pulls in numba).
# The no-cluster compute here runs the matching on dask threads, and numba's
# default workqueue layer is not thread-safe (the real app runs this on separate
# distributed worker PROCESSES, so it never hits this). Must be set pre-import.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

CIF = os.path.join(os.path.dirname(__file__), "Silver__0011135.cif")


def _t(label, fn):
    t = time.time()
    out = fn()
    print(f"  [{time.time() - t:6.2f}s] {label}")
    return out


def run(ny: int = 8, nx: int = 8, iy0: int = 28, ix0: int = 96,
        resolution: float = 1.0, voltage: float = 200.0,
        min_intensity: float = 1e-4, gamma: float = 0.5,
        n_best: int = 5, normalize: bool = True) -> None:
    import pyxem.data as pxd
    from orix.crystal_map import Phase
    from spyde.actions.orientation_compute import (
        generate_library_from_phases, build_matching_cache, best_match_spots,
        _do_compute_orientations,
    )
    from spyde.actions.orientation_action import _reciprocal_radius

    print(f"\n=== Ag Silver orientation mapping — sped_ag {ny}x{nx} region ===")

    # ── 1 Load ──────────────────────────────────────────────────────────────
    phase = _t("1 Load: Silver.cif → Phase", lambda: Phase.from_cif(CIF))
    print(f"        phase={phase.name}  point group={phase.point_group.name}")
    s = _t("1 Load: sped_ag (lazy)",
           lambda: pxd.sped_ag(allow_download=True, lazy=True))
    recip_r = _reciprocal_radius(s)

    # ── 2 Library ───────────────────────────────────────────────────────────
    sim = _t("2 Library: generate templates",
             lambda: generate_library_from_phases(
                 [phase], voltage, resolution, min_intensity, recip_r))
    n_templates = int(np.asarray(sim.rotations.data).reshape(-1, 4).shape[0])
    print(f"        {n_templates} templates  (resolution={resolution}°, "
          f"voltage={voltage}kV, min_int={min_intensity})")
    cache = _t("2 Library: build matching cache",
               lambda: build_matching_cache(s, sim))

    # ── 3 Refine (single-pattern match under the crosshair) ──────────────────
    iy, ix = iy0 + ny // 2, ix0 + nx // 2
    pat = np.asarray(s.data[iy, ix].compute(), dtype=float)
    # warm up the pyxem/numba JIT (first call is the one-time cost)
    best_match_spots(pat, sim, cache, gamma=gamma, max_radius=recip_r)
    spots = _t(f"3 Refine: best_match_spots @({iy},{ix})",
               lambda: best_match_spots(pat, sim, cache, gamma=gamma,
                                        max_radius=recip_r, normalize_templates=normalize))
    print(f"        {len(spots)} template spots (gamma={gamma}) to overlay on the DP")

    # ── 4 Run (full-field match over the region) ─────────────────────────────
    # Run on a real LocalCluster, exactly like the app: the matcher executes on
    # separate worker PROCESSES, avoiding the in-process numba workqueue clash the
    # no-cluster threaded path hits.
    from dask.distributed import Client, LocalCluster
    region = s.inav[ix0:ix0 + nx, iy0:iy0 + ny]

    class _DM:
        def __init__(self, client):
            self.client = client
            self.gpu_worker_address = None
            self.heavy_workers = None
    class _MW:
        def __init__(self, client):
            self.dask_manager = _DM(client)

    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True)
    client = Client(cluster)
    try:
        om = _t("4 Run: _do_compute_orientations (region, cluster)",
                lambda: _do_compute_orientations(
                    region, sim,
                    dict(n_best=n_best, gamma=gamma, normalize_templates=normalize),
                    main_window=_MW(client), signal_tree=None))
    finally:
        client.close(); cluster.close()
    assert om is not None, "orientation compute returned None"
    ipf = om.ipf_color_map("z")
    corr = om.correlation_map()
    print(f"        IPF-Z map {ipf.shape}  mean corr={float(corr.mean()):.3f}  "
          f"max corr={float(corr.max()):.3f}")
    print("=== Ag Silver workflow OK ===\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    a = ap.parse_args()
    run(ny=a.ny, nx=a.nx, resolution=a.resolution, gamma=a.gamma)
