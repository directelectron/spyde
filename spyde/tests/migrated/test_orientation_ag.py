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

    def test_template_spots_match_pyxem(self):
        """The matched-template overlay must land on the data with pyxem's exact
        handedness. ``best_match_spots`` builds its overlay spots via
        ``_template_spots_pyxem`` → pyxem's authoritative
        ``vectors_from_orientation_map`` (a 3-D CRYSTAL rotation). The historical
        hand-rolled 2-D rotation port agreed with pyxem only for special angles;
        for a general in-plane rotation it produced an x-/y-mirror of the real
        spots — the reported green-template x-flip (2026-06-23). This pins the
        production helper spot-for-spot against an independent pyxem call across
        rotations/mirrors that the 2-D port got WRONG (e.g. rot_idx=137)."""
        from orix.crystal_map import Phase
        from spyde.actions.orientation_compute import (
            generate_library_from_phases, _template_spots_pyxem,
        )
        from spyde.actions.orientation_action import _reciprocal_radius
        from pyxem.utils.indexation_utils import phase2dict
        from pyxem.signals.indexation_results import vectors_from_orientation_map

        phase = Phase.from_cif(CIF)
        if phase.space_group is None:
            phase.space_group = 225                  # fcc (needed by phase2dict)
        s = _calibrated_dp_4d()
        rr = _reciprocal_radius(s)
        sim = generate_library_from_phases(
            phases=[phase], accelerating_voltage=200.0, resolution=10.0,
            minimum_intensity=1e-4, reciprocal_radius=rr,
        )
        NA = 360
        n = sim.rotations.size

        # Independent reference inputs (NOT the cached ones the helper builds).
        data = np.array([sim.get_simulation(k)[2].data for k in range(n)],
                        dtype=object)
        hkl = np.array([np.asarray(sim.get_simulation(k)[2].hkl)
                        for k in range(n)], dtype=object)
        inten = np.array([np.asarray(sim.get_simulation(k)[2].intensity, float)
                          for k in range(n)], dtype=object)
        phases_dicts = [phase2dict(list(sim.phases)[0]
                                   if hasattr(sim.phases, "__len__")
                                   else sim.phases)]
        phase_index = np.zeros(n, dtype=int)

        checked = 0
        for lib_idx in sorted({0, 3, min(7, n - 1), n - 1}):
            for rot_idx in (0, 45, 137, 290):       # 137/290 broke the 2-D port
                for mirror in (1.0, -1.0):
                    angle_deg = rot_idx / NA * 360.0 - 180.0
                    spy = _template_spots_pyxem(sim, lib_idx, angle_deg, mirror)
                    assert spy is not None          # space group set → no fallback
                    row = np.array([[lib_idx, 1.0, angle_deg, mirror]])
                    ref = np.asarray(vectors_from_orientation_map(
                        row, data, phases_dicts, phase_index, hkl, inten,
                        n_best_index=0, return_object=True).data[:, :2], float)
                    assert spy.shape == ref.shape, (lib_idx, rot_idx, mirror)
                    if len(spy) == 0:
                        continue
                    assert float(np.abs(spy - ref).max()) < 1e-9, (
                        lib_idx, rot_idx, mirror)
                    checked += 1
        assert checked > 0
