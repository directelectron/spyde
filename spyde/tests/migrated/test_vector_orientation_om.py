"""
Vector Orientation Mapping (Electron, reuse OM wizard pattern).

On a `diffraction_vectors` tree, the staged Vector-Orientation handlers must:
  vom_generate_library → build the diffsims template library (cached on the tree),
  vom_run             → fit orientation + strain for the field and open an IPF-Z
                        window plus εxx / εyy / εxy strain windows, attaching the
                        VectorOrientationResult.

Mirrors `om_generate_library` / `om_run`; uses the sparse-vector matcher.
"""
from __future__ import annotations

import os
import time

import numpy as np
import hyperspy.api as hs

CIF = os.path.join(os.path.dirname(__file__), "..", "Silver__0011135.cif")


def _signal_plot(session, tree):
    return next((p for p in session._plots
                 if not p.is_navigator and getattr(p, "signal_tree", None) is tree
                 and p.plot_state is not None), None)


def _wait(pred, timeout=40.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def _multi_disk_4d(nav=(3, 3), sig=(48, 48), scale=0.05):
    """Four disks per pattern (≥4 vectors → the per-pattern fit actually runs)."""
    yy, xx = np.mgrid[0:sig[0], 0:sig[1]]
    cy, cx = sig[0] / 2, sig[1] / 2
    spots = [(cx, cy), (cx + 10, cy + 4), (cx - 8, cy + 9), (cx + 3, cy - 11)]
    pat = np.zeros(sig, np.float32)
    for sxx, syy in spots:
        pat += ((xx - sxx) ** 2 + (yy - syy) ** 2 <= 6).astype(np.float32)
    data = np.zeros(nav + sig, np.float32)
    for idx in np.ndindex(*nav):
        data[idx] = pat * 100.0
    s = hs.signals.Signal2D(data)
    s.set_signal_type("electron_diffraction")
    for ax in s.axes_manager.signal_axes:
        ax.scale = scale
        ax.offset = -(ax.size / 2.0) * scale
        ax.units = "1/nm"
    return s


def _make_vectors_tree(session):
    session._add_signal(_multi_disk_4d())
    time.sleep(0.4)
    src = next((p for p in session._plots
                if not p.is_navigator and p.plot_state is not None), None)
    session._dispatch_toolbar_action(
        src, "Find Diffraction Vectors",
        {"sigma": 0.6, "kernel_radius": 3, "threshold": 0.3,
         "min_distance": 2, "subpixel": True},
    )
    assert _wait(lambda: getattr(session.signal_trees[-1], "diffraction_vectors", None) is not None), \
        "Find Vectors never attached diffraction_vectors"
    return session.signal_trees[-1]


class TestVectorOrientationOM:
    def test_generate_then_run(self):
        from spyde.backend.session import Session
        from spyde.actions.vector_orientation_om import vom_generate_library, vom_run
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            vtree = _make_vectors_tree(session)
            vplot = _signal_plot(session, vtree)
            assert vplot is not None

            # ── Generate Library (coarse resolution → quick) ────────────────
            vom_generate_library(session, vplot, {
                "cif_path": CIF, "accelerating_voltage": 200.0,
                "resolution": 12.0, "minimum_intensity": 1e-4,
            })
            assert _wait(lambda: getattr(vtree, "_vom_wizard", None) is not None
                         and vtree._vom_wizard.get("lib") is not None), \
                "template library never built"
            assert len(vtree._vom_wizard["lib"].spots_xy) > 0

            # ── Compute Maps → IPF-Z + 3 strain windows ─────────────────────
            n_before = len(session.signal_trees)
            vom_run(session, vplot, {"strain_cap": 0.05, "smooth": False})
            # 4 new trees: IPF-Z + εxx + εyy + εxy.
            assert _wait(lambda: len(session.signal_trees) >= n_before + 4,
                         timeout=90), "orientation/strain windows never opened"
            ipf_tree = next((t for t in session.signal_trees
                             if getattr(t, "vector_orientation", None) is not None), None)
            assert ipf_tree is not None
            res = ipf_tree.vector_orientation
            assert res.nav_shape == tuple(vtree.diffraction_vectors.nav_shape)
            assert res.strain.shape[-1] == 3
        finally:
            session.shutdown()

    def test_generate_activates_live_refine_overlay(self):
        from spyde.backend.session import Session
        from spyde.actions.vector_orientation_om import vom_generate_library
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            vtree = _make_vectors_tree(session)
            vplot = _signal_plot(session, vtree)
            vom_generate_library(session, vplot, {
                "cif_path": CIF, "accelerating_voltage": 200.0,
                "resolution": 12.0, "minimum_intensity": 1e-4,
            })
            assert _wait(lambda: getattr(vtree, "_vom_wizard", None) is not None
                         and vtree._vom_wizard.get("overlay") is not None), \
                "live refine overlay never attached"
            ov = vtree._vom_wizard["overlay"]
            # Two marker groups: measured (red) + fitted template (green).
            assert ov._mg_meas is not None and ov._mg_tmpl is not None
            # At a position with ≥4 vectors: measured points drawn, template fit too.
            vecs = vtree.diffraction_vectors
            cm = vecs.count_map()
            ys, xs = np.nonzero(cm >= 4)
            if len(ys):
                meas, tmpl = ov._offsets_for(int(ys[0]), int(xs[0]))
                assert meas.shape[0] >= 4
                assert tmpl.shape[1] == 2     # a fitted template was produced
        finally:
            session.shutdown()

    def test_run_without_library_errors_gracefully(self):
        from spyde.backend.session import Session
        from spyde.actions.vector_orientation_om import vom_run
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            vtree = _make_vectors_tree(session)
            vplot = _signal_plot(session, vtree)
            before = len(session.signal_trees)
            vom_run(session, vplot, {"strain_cap": 0.05})   # no library yet
            time.sleep(0.4)
            assert len(session.signal_trees) == before
        finally:
            session.shutdown()
