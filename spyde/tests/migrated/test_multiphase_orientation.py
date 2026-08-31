"""
Multi-phase orientation mapping.

The OM pipeline (`generate_library_from_phases` → `_do_compute_orientations` →
`ipf_color_map` / `ipf_sphere_points`) already supports >1 phase; the Electron
`om_generate_library` accepts several `.cif` paths. These tests verify the
two-phase path produces an orientation map spanning both phases + a clean 2-D IPF
RGB map, and that the wizard handler accepts multiple CIFs.
"""
from __future__ import annotations

import os
import time

import numpy as np
import hyperspy.api as hs
from spyde.tests.migrated.conftest import _settle, close_session, make_session
from spyde.tests.migrated._async import wait_until


def _wait(pred, timeout=40.0):
    return wait_until(pred, timeout)

CIF = os.path.join(os.path.dirname(__file__), "..", "Silver__0011135.cif")


def _signal_plot(session, tree=None):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None
                 and (tree is None or getattr(p, "signal_tree", None) is tree)), None)


def _two_phases():
    from orix.crystal_map import Phase
    from diffpy.structure import Atom, Lattice, Structure
    al = Phase(name="Al", space_group=225, structure=Structure(
        atoms=[Atom("Al", [0, 0, 0])], lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90)))
    fe = Phase(name="Fe", space_group=229, structure=Structure(
        atoms=[Atom("Fe", [0, 0, 0])], lattice=Lattice(2.87, 2.87, 2.87, 90, 90, 90)))
    return [al, fe]


def _diffraction_4d(nav=(3, 3), sig=(64, 64), scale=0.03):
    rng = np.random.RandomState(0)
    s = hs.signals.Signal2D(rng.rand(*nav, *sig).astype(np.float32))
    s.set_signal_type("electron_diffraction")
    for ax in s.axes_manager.signal_axes:
        ax.scale = scale
        ax.offset = -(ax.size / 2.0) * scale
        ax.units = "$A^{-1}$"
    return s


class TestMultiPhaseOrientation:
    def test_two_phase_map_and_2d_ipf(self):
        from spyde.actions.orientation_action import run_orientation
        session = make_session()
        try:
            session._add_signal(_diffraction_4d())
            _settle(session)
            src = _signal_plot(session)
            tree = src.signal_tree
            before = len(session.signal_trees)

            run_orientation(
                session, tree.root, tree, _two_phases(),
                dict(accelerating_voltage=200.0, resolution=10.0),
                dict(n_best=2, gamma=0.5), src_dp_plot=src,
            )
            assert _wait(lambda: len(session.signal_trees) > before, timeout=90), \
                "orientation IPF window never opened"
            otree = next((t for t in session.signal_trees
                          if getattr(t, "orientation_map", None) is not None), None)
            assert otree is not None
            om = otree.orientation_map
            assert om.n_phases == 2
            # A clean 2-D IPF RGB map: shape (ny, nx, 3), every pixel coloured.
            rgb = om.ipf_color_map("z")
            assert rgb.shape == tuple(om.nav_shape) + (3,) and rgb.dtype == np.uint8
            assert (rgb.sum(-1) > 0).all()
        finally:
            close_session(session)

    def test_generate_library_accepts_multiple_cifs(self):
        from spyde.actions.orientation_action import om_generate_library
        session = make_session()
        try:
            session._add_signal(_diffraction_4d())
            _settle(session)
            src = _signal_plot(session)
            tree = src.signal_tree
            # Two CIFs (degenerate here, but exercises the multi-phase code path).
            om_generate_library(session, src, {
                "cif_paths": [CIF, CIF], "accelerating_voltage": 200.0,
                "resolution": 12.0, "minimum_intensity": 1e-4,
            })
            assert _wait(lambda: getattr(tree, "_om_wizard", None) is not None
                         and tree._om_wizard.sim is not None)
            wiz = tree._om_wizard
            assert len(wiz.phases) == 2
            # Live refine overlay is single-phase only → skipped for multi-phase.
            assert wiz.overlay is None
        finally:
            close_session(session)
