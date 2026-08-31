"""
Staged Orientation-Mapping wizard backend (Qt 4-tab parity):

  Generate Library  → `om_generate_library` builds the library + cache and
                       activates the LIVE refine overlay on the source DP.
  Refine            → `om_refine` updates gamma/scale/normalize and re-draws the
                       matched template at the current crosshair (no re-run).
  Compute Map       → `om_run` runs the full-field match using the already-built
                       library → IPF-Z window + attached orientation map.
"""
from __future__ import annotations

import os
import time

import numpy as np
import hyperspy.api as hs

from spyde.tests.migrated.conftest import _settle
from spyde.tests.migrated._async import quiesce, why_busy
from spyde.tests.migrated._async import wait_until


def _wait(pred, timeout=30.0):
    return wait_until(pred, timeout)

CIF = os.path.join(os.path.dirname(__file__), "..", "Silver__0011135.cif")


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _calibrated_4d(nav=(3, 3), sig=(64, 64), scale=0.0134):
    rng = np.random.RandomState(0)
    s = hs.signals.Signal2D(rng.rand(*nav, *sig).astype(np.float32))
    s.set_signal_type("electron_diffraction")
    for ax in s.axes_manager.signal_axes:
        ax.scale = scale
        ax.offset = -(ax.size / 2.0) * scale
        ax.units = "$A^{-1}$"
    return s


class TestOrientationWizard:
    def test_generate_refine_run(self):
        from spyde.backend.session import Session
        from spyde.actions.orientation_action import (
            om_generate_library, om_refine, om_run,
        )
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_calibrated_4d())
            _settle(session)
            src = _signal_plot(session)
            tree = src.signal_tree

            # ── Pre-generate: Compute Map without a library must error
            #    gracefully — no new tree ──────────────────────────────────────
            before = len(session.signal_trees)
            om_run(session, src, {"n_best": 3})   # no library generated
            assert quiesce(session), why_busy(session)
            assert len(session.signal_trees) == before   # nothing created

            # ── Generate Library (coarse resolution → quick) ─────────────────
            om_generate_library(session, src, {
                "cif_path": CIF, "accelerating_voltage": 200.0,
                "resolution": 10.0, "minimum_intensity": 1e-4,
            })
            assert _wait(lambda: getattr(tree, "_om_wizard", None) is not None
                         and tree._om_wizard.overlay is not None), \
                "library/overlay never became ready"
            wiz = tree._om_wizard
            assert wiz.sim is not None and wiz.cache is not None
            overlay = wiz.overlay
            assert overlay._mg is not None          # live template overlay attached

            # ── Refine: change gamma + normalize live ────────────────────────
            om_refine(session, src, {"gamma": 0.3, "normalize_templates": True})
            assert _wait(lambda: abs(overlay.gamma - 0.3) < 1e-9)
            assert overlay.normalize_templates is True

            # scale override is applied too
            om_refine(session, src, {"gamma": 0.3, "scale_override": 0.014})
            assert _wait(lambda: overlay.scale_override is not None)

            # ── Compute Map (reuses the built library) ───────────────────────
            # The IPF window now opens BLANK up front (progressive live fill-in)
            # and the orientation map is attached when the compute finishes.
            before = len(session.signal_trees)
            om_run(session, src, {"n_best": 3, "gamma": 0.5,
                                  "normalize_templates": False})
            assert _wait(lambda: len(session.signal_trees) == before + 1,
                         timeout=60), "IPF window never opened"
            otree = session.signal_trees[-1]
            assert _wait(lambda: getattr(otree, "orientation_map", None) is not None,
                         timeout=60), "orientation map never attached"
        finally:
            session.shutdown()

    # test_run_without_library_errors_gracefully was folded into
    # test_generate_refine_run as its pre-generate stage: same session build,
    # and the error path must run against a genuinely library-less tree, which
    # is exactly the state before om_generate_library.
