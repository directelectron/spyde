"""
benchmark_quantem_orientation.py — the vector orientation matcher at real scale.

Times each stage of the correlation match on the REAL ``sped_ag`` scan and
reports what the result says about itself: the correlation, the reliability
(best minus best outside the exclusion ball) and the strain.

It used to compare against SpyDE's own pose fit, which is how the quaternion
convention was settled and how the two were measured against each other. That
fit has since been deleted — the matcher does the job — so what is left is the
measurement of one method rather than a comparison of two. The convention is
pinned by ``test_quantem_adapter.py`` and the strain by the known-strain gate
there; this is for wall clock and for looking at real numbers.

Run directly (NOT pytest — downloads the real dataset, uses the GPU)::

    uv run python -m spyde.tests.benchmark_quantem_orientation --ny 64 --nx 208
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


def _t(label, fn):
    start = time.time()
    out = fn()
    print(f"  [{time.time() - start:6.2f}s] {label}", flush=True)
    return out


def run(ny=64, nx=208, iy0=0, ix0=0, voltage=200.0, zone_step=1.0,
        in_plane_step=5.0, device="cuda"):
    from orix.crystal_map import Phase

    from spyde.actions.find_vectors import _do_compute_vectors
    from spyde.actions.orientation_action import _reciprocal_radius
    from spyde.actions.vector_orientation_quantem import (
        PeaksAdapter, build_orientation_map, phase_to_crystal,
        strain_from_orientation_map,
    )
    from spyde.backend._session_testharness import TestHarnessMixin
    from spyde.reciprocal_units import axis_unit_factor

    print(f"\n=== vector orientation matcher — sped_ag {ny}x{nx} ===", flush=True)
    phase = _t("Load Silver.cif", lambda: Phase.from_cif(CIF))

    # The CALIBRATED scan. pyxem and em-database both still pin a record whose
    # copy is at half the true reciprocal scale, which no crystal can be fitted
    # to; the harness loader is the one that fetches the corrected file.
    import pooch

    from spyde.external.emdatabase.sped_ag import DATASET
    import hyperspy.api as hs

    path = pooch.retrieve(url=f"{DATASET['source']}/{DATASET['file']}",
                          known_hash=DATASET["checksum"],
                          fname="SPED-Ag-calibrated.zspy",
                          path=pooch.os_cache("spyde"))
    signal = _t("Load SPED-Ag (calibrated)", lambda: hs.load(path, lazy=True))
    scale = float(signal.axes_manager.signal_axes[0].scale)
    assert abs(scale - TestHarnessMixin._SPED_AG_SCALE) < 1e-9, \
        f"not the calibrated copy: {scale} Å⁻¹/px"
    reciprocal_radius = _reciprocal_radius(signal)
    print(f"        recip_r={reciprocal_radius:.4f} 1/A  scale={scale:.6f}",
          flush=True)

    region = signal.inav[ix0:ix0 + nx, iy0:iy0 + ny]
    region_eager = _t("materialise region", lambda: region.deepcopy())
    region_eager.data = np.asarray(region.data.compute(), np.float32)
    region_eager._lazy = False

    vectors = _t("find diffraction vectors",
                 lambda: _do_compute_vectors(
                     region_eager,
                     dict(sigma=1.0, kernel_radius=5, threshold=0.4,
                          min_distance=3, subpixel=True),
                     main_window=None, signal_tree=None))

    unit_factor = float(axis_unit_factor(region_eager) or 1.0)
    peaks = PeaksAdapter(vectors, inverse_angstrom_factor=unit_factor)
    counts = peaks.peak_counts
    print(f"        peaks/pattern: median={np.median(counts):.0f} "
          f"min={counts.min()} max={counts.max()} "
          f"({int((counts >= 5).sum())}/{len(counts)} have the five the "
          f"matcher needs)", flush=True)

    crystal = _t("Phase -> Crystal", lambda: phase_to_crystal(phase, name="Ag"))
    orientation_map = _t(
        "structure factors + orientation plan",
        lambda: build_orientation_map(
            peaks, crystal, energy_ev=voltage * 1e3, k_max=reciprocal_radius,
            angle_step_zone_axis_deg=zone_step,
            angle_step_in_plane_deg=in_plane_step,
            device=device, refine_device=device, verbose=True))
    print(f"        {orientation_map.zone_axes.shape[0]} zone axes x "
          f"{orientation_map.gamma.shape[0]} in-plane x "
          f"{orientation_map.shell_radii.shape[0]} shells", flush=True)

    _t("match_orientations",
       lambda: orientation_map.match_orientations(progress_bar=False))
    _t("refine_orientations",
       lambda: orientation_map.refine_orientations(progress_bar=False))
    strain, pairs = _t(
        "strain",
        lambda: strain_from_orientation_map(orientation_map, device=device))

    correlation = orientation_map.corr[..., 0].cpu().numpy()
    reliability = orientation_map.reliability.cpu().numpy()
    fitted = correlation > 0
    print(f"\n  correlation : median={np.median(correlation[fitted]):.3f} "
          f"min={correlation[fitted].min():.3f} max={correlation[fitted].max():.3f}")
    print(f"  reliability : median={np.median(reliability[fitted]):.3f} "
          f"(best minus best outside the exclusion ball — what a grain-boundary "
          f"mask is thresholded on)")
    usable = fitted & np.isfinite(strain).all(-1)
    print(f"  strain      : {int(usable.sum())}/{usable.size} positions, "
          f"median {int(np.median(pairs[fitted]))} paired reflections")
    for index, label in enumerate(("exx", "eyy", "exy")):
        values = strain[..., index][usable]
        if values.size:
            print(f"    {label}: median {np.median(values):+.4f}  "
                  f"sd {values.std():.4f}")
    return dict(orientation_map=orientation_map, strain=strain, vectors=vectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nx", type=int, default=208)
    parser.add_argument("--iy0", type=int, default=0)
    parser.add_argument("--ix0", type=int, default=0)
    parser.add_argument("--zone-step", type=float, default=1.0)
    parser.add_argument("--in-plane-step", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(ny=args.ny, nx=args.nx, iy0=args.iy0, ix0=args.ix0,
        zone_step=args.zone_step, in_plane_step=args.in_plane_step,
        device=args.device)
