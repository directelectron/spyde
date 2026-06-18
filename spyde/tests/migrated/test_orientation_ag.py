"""
Ag Silver orientation-mapping workflow (the Qt 4-tab wizard), fast/CI slice.

Exercises the Ag-specific stages on the REAL Silver .cif (no sped_ag download,
no full-field Run — the batch compute is covered by test_orientation_port /
test_orientation_overlay / orientation_lazy.spec.ts):

  1 Load    — Silver__0011135.cif → orix Phase (m-3m).
  2 Library — generate the diffsims template library + matching cache (coarse
              angle resolution so it's quick).
  3 Refine  — single-pattern template match (best_match_spots) with the gamma
              slider → best-fit spots to overlay on the diffraction pattern.

The full real-scale run is `spyde/tests/benchmark_orientation_ag.py`.
"""
from __future__ import annotations

import os

import numpy as np
import hyperspy.api as hs

CIF = os.path.join(os.path.dirname(__file__), "..", "Silver__0011135.cif")


def _calibrated_dp_4d(nav=(3, 3), sig=(64, 64), scale=0.0134):
    """A small calibrated 4D-STEM signal (sped_ag-like axes) so the polar
    matching geometry is valid."""
    rng = np.random.RandomState(0)
    s = hs.signals.Signal2D(rng.rand(*nav, *sig).astype(np.float32))
    s.set_signal_type("electron_diffraction")
    for ax in s.axes_manager.signal_axes:
        ax.scale = scale
        ax.offset = -(ax.size / 2.0) * scale
        ax.units = "$A^{-1}$"
    return s


class TestAgSilverWorkflow:
    def test_load_cif_library_refine(self):
        from orix.crystal_map import Phase
        from spyde.actions.orientation_compute import (
            generate_library_from_phases, build_matching_cache, best_match_spots,
        )
        from spyde.actions.orientation_action import _reciprocal_radius

        # 1 Load
        assert os.path.isfile(CIF), CIF
        phase = Phase.from_cif(CIF)
        assert phase.point_group.name == "m-3m"      # fcc silver

        s = _calibrated_dp_4d()
        rr = _reciprocal_radius(s)
        assert rr > 0

        # 2 Library (coarse resolution → quick) + matching cache
        sim = generate_library_from_phases(
            phases=[phase], accelerating_voltage=200.0, resolution=10.0,
            minimum_intensity=1e-4, reciprocal_radius=rr,
        )
        n_templates = int(np.asarray(sim.rotations.data).reshape(-1, 4).shape[0])
        assert n_templates > 0
        cache = build_matching_cache(s, sim)
        assert cache["NR"] == 100 and cache["NA"] == 360

        # 3 Refine: single-pattern best-fit spots (the gamma slider feeds this)
        pat = np.asarray(s.data[1, 1], dtype=float)
        spots = best_match_spots(pat, sim, cache, gamma=0.5, max_radius=rr)
        assert isinstance(spots, np.ndarray)
        assert spots.ndim == 2 and spots.shape[1] == 2      # (N, 2) [kx, ky]
        # Spots are within the reciprocal radius (they overlay on the DP).
        if len(spots):
            r = np.sqrt((spots ** 2).sum(axis=1))
            assert float(r.max()) <= rr + 1e-6
