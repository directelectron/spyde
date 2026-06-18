"""
Staged Find-Diffraction-Vectors wizard backend (Qt parity):

  fv_preview → attaches a LIVE found-peaks preview overlay to the source DP so
               red circles update as you tune the sliders / move the navigator.
  fv_tune    → live-updates the preview sliders (σ / kernel radius / threshold /
               min distance / subpixel) and redraws at the current crosshair —
               NO full-dataset compute.
  fv_run     → full-dataset batch with the tuned params → a new vectors-image
               window with `tree.diffraction_vectors` attached.
  fv_stop    → removes the live preview overlay (caret closed).

Memory safety: the preview only slices/computes a small nav window (radius
ceil(3σ)) around the crosshair — never the full dataset (see
test_find_vectors_memory for the batch-compute contract).
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _wait(pred, timeout=25.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def _calibrated_diffraction_4d(nav=(4, 5), sig=(24, 24), scale=0.1):
    """A 4D-STEM stack: every pattern has a bright disk at the centre (so the
    NXCORR peak finder reliably finds at least one peak)."""
    data = np.zeros(nav + sig, dtype=np.float32)
    yy, xx = np.mgrid[0:sig[0], 0:sig[1]]
    disk = ((xx - 12) ** 2 + (yy - 12) ** 2 <= 16).astype(np.float32)
    for idx in np.ndindex(*nav):
        data[idx] = disk * 100.0
    s = hs.signals.Signal2D(data)
    s.set_signal_type("electron_diffraction")
    for ax in s.axes_manager.signal_axes:
        ax.scale = scale
        ax.offset = 0.0
        ax.units = "1/nm"
    return s


class TestFindVectorsWizard:
    def test_preview_tune_run(self):
        from spyde.backend.session import Session
        from spyde.actions.find_vectors_action import (
            fv_preview, fv_tune, fv_run, fv_stop,
        )
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_calibrated_diffraction_4d(scale=0.1))
            time.sleep(0.4)
            src = _signal_plot(session)
            tree = src.signal_tree

            # ── Tune: start the live preview ─────────────────────────────────
            fv_preview(session, src, {
                "sigma": 1.0, "kernel_radius": 5, "threshold": 0.4,
                "min_distance": 3, "subpixel": True,
            })
            assert _wait(lambda: getattr(tree, "_fv_preview", None) is not None), \
                "live preview overlay never attached"
            prev = tree._fv_preview
            assert prev._mg is not None              # circle marker group exists

            # The centred disk produces at least one peak at the current crosshair.
            assert _wait(lambda: prev._offsets_for(0, 0).shape[0] >= 1), \
                "preview found no peaks on a disk pattern"

            # ── Tune: change params live (no new window) ─────────────────────
            before_trees = len(session.signal_trees)
            fv_tune(session, src, {
                "sigma": 1.0, "kernel_radius": 7, "threshold": 0.3,
                "min_distance": 4, "subpixel": False,
            })
            assert _wait(lambda: prev.kernel_radius == 7 and prev.subpixel is False)
            assert abs(prev.threshold - 0.3) < 1e-9
            assert len(session.signal_trees) == before_trees   # tune never computes

            # ── Compute: full-dataset batch → a new vectors window ───────────
            fv_run(session, src, {
                "sigma": 1.0, "kernel_radius": 5, "threshold": 0.4,
                "min_distance": 3, "subpixel": True,
            })
            assert _wait(lambda: len(session.signal_trees) == before_trees + 1,
                         timeout=40), "vectors window never opened"
            vtree = session.signal_trees[-1]
            assert _wait(lambda: getattr(vtree, "diffraction_vectors", None) is not None,
                         timeout=40), "diffraction_vectors never attached"
            assert int(vtree.diffraction_vectors.count_map().sum()) > 0

            # Running drops the live preview (the final overlay replaces it).
            assert _wait(lambda: getattr(tree, "_fv_preview", None) is None)
            # …and attaches the persistent found-vector overlay on the source DP.
            assert _wait(lambda: getattr(tree, "_vector_overlay", None) is not None)
        finally:
            session.shutdown()

    def test_stop_removes_preview(self):
        from spyde.backend.session import Session
        from spyde.actions.find_vectors_action import fv_preview, fv_stop
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_calibrated_diffraction_4d(scale=0.1))
            time.sleep(0.4)
            src = _signal_plot(session)
            tree = src.signal_tree

            fv_preview(session, src, {"sigma": 1.0, "kernel_radius": 5,
                                      "threshold": 0.4, "min_distance": 3,
                                      "subpixel": True})
            assert _wait(lambda: getattr(tree, "_fv_preview", None) is not None)
            prev = tree._fv_preview

            fv_stop(session, src, {})
            assert _wait(lambda: getattr(tree, "_fv_preview", None) is None)
            assert prev._mg is None                  # marker group removed
        finally:
            session.shutdown()
