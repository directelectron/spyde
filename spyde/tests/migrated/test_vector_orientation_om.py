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
    def test_generate_then_run(self, monkeypatch):
        from spyde.backend.session import Session
        from spyde.actions.vector_orientation_om import vom_generate_library, vom_run
        # Force the CPU fit path: this exercises the WIRING (handlers → result →
        # windows). The batched-torch GPU path is validated separately (subprocess
        # GPU test + the sped_ag benchmark) — running torch autograd under pytest
        # is slow (cold JIT) and segfaults on Windows+CUDA.
        import spyde.actions.vector_orientation_gpu as _gpu
        monkeypatch.setattr(_gpu, "select_device", lambda: None)
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
                         and vtree._vom_wizard.lib is not None), \
                "template library never built"
            assert len(vtree._vom_wizard.lib.spots_xy) > 0

            # Generate now ALSO fits the whole field and opens the live IPF
            # heatmap window (Qt parity — the orientation map appears while you
            # refine, before Compute Maps).
            assert _wait(lambda: getattr(vtree, "_vom_field", None) is not None,
                         timeout=90), "live field fit / IPF heatmap never produced"
            # The IPF heatmap tree is created AFTER _vom_field is set (the worker
            # sets _vom_field, then calls _build_ipf_heatmap which sets
            # .vector_orientation). Poll for it rather than asserting immediately —
            # on slow runners the worker is still between those two steps here.
            assert _wait(lambda: any(getattr(t, "vector_orientation", None) is not None
                                     for t in session.signal_trees),
                         timeout=90), "no IPF heatmap window"

            # ── Compute Maps → reuses the field, adds ONE unified Strain window
            #    (εxx is its signal plot; εyy / εxy are chip-selectable view
            #    figures emitted into the same window, not new signal trees) ──
            n_before = len(session.signal_trees)
            vom_run(session, vplot, {"strain_cap": 0.05, "smooth": False})
            assert _wait(lambda: len(session.signal_trees) >= n_before + 1,
                         timeout=90), "strain window never opened"
            ipf_tree = next((t for t in session.signal_trees
                             if getattr(t, "vector_orientation", None) is not None), None)
            assert ipf_tree is not None
            res = ipf_tree.vector_orientation
            assert res.nav_shape == tuple(vtree.diffraction_vectors.nav_shape)
            assert res.strain.shape[-1] == 3

            # The unified Strain window tags its signal plot as the εxx chip view
            # (εyy / εxy ride along as extra view figures in the same window).
            def _strain_tree():
                return next((t for t in session.signal_trees
                             if "Strain" in t.root.metadata.get_item(
                                 "General.title", "")), None)

            # The Strain tree is added (passing the wait above) BEFORE the worker
            # tags its signal plot's view_label "εxx" (_build_result_windows adds
            # the tree, then calls sp.set_view_tag). Poll for the tagged plot
            # rather than reading view_label immediately — on slow runners the
            # tag hasn't landed yet at this point.
            def _strain_sp():
                st = _strain_tree()
                return next(iter(getattr(st, "signal_plots", [])), None) if st else None

            assert _wait(lambda: getattr(_strain_sp(), "view_label", None) == "εxx",
                         timeout=90), "Strain window εxx view never tagged"
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
                         and vtree._vom_wizard.overlay is not None), \
                "live refine overlay never attached"
            ov = vtree._vom_wizard.overlay
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

    def test_fit_field_prefers_gpu_then_falls_back(self, monkeypatch):
        """`_fit_field` must dispatch the BATCHED GPU path first (Qt parity — the
        serial CPU fit is ~30 min on a real 13k-pattern scan) and fall back to CPU
        only when torch is unavailable or the GPU fit raises."""
        import spyde.actions.vector_orientation_om as vom
        import spyde.actions.vector_orientation_gpu as gpu
        import spyde.actions.vector_orientation as cpu

        class _Vecs:
            nav_shape = (4, 5)
        calls = []

        # (1) GPU available + succeeds → CPU never called.
        monkeypatch.setattr(gpu, "torch_available", lambda: True)
        monkeypatch.setattr(gpu, "select_device", lambda: type("D", (), {"type": "mps"})())
        monkeypatch.setattr(gpu, "compute_vector_orientation_gpu",
                            lambda *a, **k: (calls.append("gpu"), "RESULT")[1])
        monkeypatch.setattr(cpu, "compute_vector_orientation",
                            lambda *a, **k: calls.append("cpu"))
        assert vom._fit_field(_Vecs(), object(), {}) == "RESULT"
        assert calls == ["gpu"]

        # (2) GPU raises → CPU fallback runs.
        calls.clear()
        monkeypatch.setattr(gpu, "compute_vector_orientation_gpu",
                            lambda *a, **k: (calls.append("gpu"), (_ for _ in ()).throw(RuntimeError("boom")))[1])
        monkeypatch.setattr(cpu, "compute_vector_orientation",
                            lambda *a, **k: (calls.append("cpu"), "CPU_RESULT")[1])
        assert vom._fit_field(_Vecs(), object(), {}) == "CPU_RESULT"
        assert calls == ["gpu", "cpu"]

        # (3) torch unavailable → straight to CPU.
        calls.clear()
        monkeypatch.setattr(gpu, "torch_available", lambda: False)
        assert vom._fit_field(_Vecs(), object(), {}) == "CPU_RESULT"
        assert calls == ["cpu"]

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
