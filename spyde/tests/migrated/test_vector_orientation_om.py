"""
Vector Orientation Mapping (Electron, reuse OM wizard pattern).

On a `diffraction_vectors` tree, the staged Vector-Orientation handlers must:
  vom_generate_library → build one correlation plan per phase (cached on the tree),
  vom_run             → fit orientation + strain for the field and open an IPF-Z
                        window plus εxx / εyy / εxy strain windows, attaching the
                        VectorOrientationResult.

Mirrors `om_generate_library` / `om_run`; uses the correlation matcher.
"""
from __future__ import annotations

import os
import types

import pytest
import time

import numpy as np
import hyperspy.api as hs

from spyde.tests.migrated.conftest import _settle, close_session, make_session
from spyde.tests.migrated._async import wait_until


def _wait(pred, timeout=40.0):
    return wait_until(pred, timeout)

CIF = os.path.join(os.path.dirname(__file__), "..", "Silver__0011135.cif")


def _signal_plot(session, tree):
    return next((p for p in session._plots
                 if not p.is_navigator and getattr(p, "signal_tree", None) is tree
                 and p.plot_state is not None), None)


def _multi_disk_4d(nav=(3, 3), sig=(48, 48), scale=0.5):
    """Five disks per pattern, so the per-pattern fit actually runs.

    Five and not four: the correlation matcher will not attempt a pattern with
    fewer than five peaks (upstream's MIN_NUMBER_PEAKS), where the pose fit it
    replaced needed four. A four-disk fixture leaves every position unmatched
    and the overlay correctly draws no template — which reads as the wiring
    being broken.

    Calibrated in nm⁻¹ **and meaning it**: 0.5 nm⁻¹/px over 48 px is a
    half-extent of 12 nm⁻¹ = 1.2 Å⁻¹, which comfortably holds silver's
    reflections ({111} at 0.42 Å⁻¹). The fixture used to say ``1/nm`` while
    carrying a scale that only made sense as Å⁻¹ — harmless while nothing read
    the label, and a library ten times too small the moment something did. Kept
    in nm⁻¹ rather than relabelled, so this wiring test also crosses the
    conversion the way a real nm⁻¹ dataset does.
    """
    yy, xx = np.mgrid[0:sig[0], 0:sig[1]]
    cy, cx = sig[0] / 2, sig[1] / 2
    spots = [(cx, cy), (cx + 10, cy + 4), (cx - 8, cy + 9), (cx + 3, cy - 11),
             (cx - 12, cy - 6)]
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
    _settle(session)
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
        from spyde.actions.vector_orientation_om import vom_generate_library, vom_run
        # Force the CPU fit path: this exercises the WIRING (handlers → result →
        # windows). The batched-torch GPU path is validated separately (subprocess
        # GPU test + the sped_ag benchmark) — running torch autograd under pytest
        # is slow (cold JIT) and segfaults on Windows+CUDA.
        import spyde.torch_device as _gpu
        monkeypatch.setattr(_gpu, "select_device", lambda: None)
        session = make_session()
        try:
            vtree = _make_vectors_tree(session)
            vplot = _signal_plot(session, vtree)
            assert vplot is not None

            # ── Pre-generate: Compute Maps without a library must error
            #    gracefully — no new tree (the natural staged-wizard ordering,
            #    so it runs against a genuinely library-less tree) ───────────
            before = len(session.signal_trees)
            vom_run(session, vplot, {})   # no library yet
            time.sleep(0.4)
            assert len(session.signal_trees) == before

            # ── Generate Library (coarse resolution → quick) ────────────────
            vom_generate_library(session, vplot, {
                "cif_path": CIF, "accelerating_voltage": 200.0,
                "resolution": 12.0, "minimum_intensity": 1e-4,
            })
            assert _wait(lambda: getattr(vtree, "_vom_wizard", None) is not None
                         and vtree._vom_wizard.fitter is not None), \
                "correlation plan never built"
            assert vtree._vom_wizard.fitter._maps, "no plan per phase"

            # ── Generate also activates the live refine overlay ─────────────
            assert _wait(lambda: vtree._vom_wizard.overlay is not None), \
                "live refine overlay never attached"
            node = vtree._vom_wizard.overlay
            # Two marker groups: measured (red) + fitted template (green).
            assert set(node.groups) == {"measured", "template"}
            assert (id(node), "measured") in vplot._overlay_groups
            assert (id(node), "template") in vplot._overlay_groups
            # At a position the matcher will attempt (five peaks or more):
            # measured points drawn, matched pattern too.
            from spyde.array_cache import reader_for_overlay
            vecs = vtree.diffraction_vectors
            cm = vecs.count_map()
            ys, xs = np.nonzero(cm >= 5)
            if len(ys):
                value = reader_for_overlay(vplot, node).read_frame(
                    (int(ys[0]), int(xs[0])))
                assert value["measured"].shape[0] >= 5
                assert value["template"].shape[1] == 2   # a matched pattern

            # Generate stops at the library and the previews. It does NOT fit
            # the field: that is a scan's compute before the user has chosen
            # anything, and the orientation map is Run's to produce.
            assert not any(getattr(t, "vector_orientation", None) is not None
                           for t in session.signal_trees), \
                "Generate produced an orientation map; that is Run's job"

            # ── Compute Maps → fits the field, opens the orientation map and
            #    ONE unified Strain window
            #    (εxx is its signal plot; εyy / εxy are chip-selectable view
            #    figures emitted into the same window, not new signal trees) ──
            n_before = len(session.signal_trees)
            vom_run(session, vplot, {"smooth": False})
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
            close_session(session)

    # test_generate_activates_live_refine_overlay and
    # test_run_without_library_errors_gracefully were folded into
    # test_generate_then_run above: they re-ran the identical
    # _make_vectors_tree + vom_generate_library flow to read other properties
    # of the same wizard (and the overlay variant did not force the CPU fit,
    # so its background field fit could hit CUDA in-process on dev boxes).

    def test_fit_field_matches_and_does_not_fall_back(self, monkeypatch):
        """The field is fitted by the correlation matcher, and a failure
        surfaces instead of being fitted some other way.

        The matcher and the per-pattern pose fit it replaced report strain in
        opposite senses, so a silent fallback would have made the sign of every
        strain map depend on which one happened to succeed — a wrong answer
        rather than a slow one.
        """
        import spyde.actions.vector_orientation_om as vom
        import spyde.actions.vector_orientation_quantem as quantem

        class _Vecs:
            nav_shape = (4, 5)

        wizard = types.SimpleNamespace(
            lib=types.SimpleNamespace(inverse_angstrom_factor=1.0),
            phases=[object()], voltage=200.0, recip_r=1.5)
        calls = []

        monkeypatch.setattr(quantem, "compute_vector_orientation_quantem",
                            lambda *a, **k: (calls.append("match"), "RESULT")[1])
        assert vom._fit_field(_Vecs(), wizard, {}) == "RESULT"
        assert calls == ["match"]

        calls.clear()
        monkeypatch.setattr(
            quantem, "compute_vector_orientation_quantem",
            lambda *a, **k: (calls.append("match"),
                             (_ for _ in ()).throw(RuntimeError("boom")))[1])
        with pytest.raises(RuntimeError):
            vom._fit_field(_Vecs(), wizard, {})
        assert calls == ["match"], "a failure must surface, not fall back"

    def test_the_pose_fit_modules_are_gone(self):
        """There is nothing left to fall back TO.

        The strongest form of the rule above: the per-pattern scipy pose fit
        and its batched GPU twin are deleted, so no future edit can quietly
        reinstate the opposite strain sign by importing one.
        """
        import importlib

        for name in ("spyde.actions.vector_orientation",
                     "spyde.actions.vector_orientation_gpu"):
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module(name)

    def test_refine_settings_reach_the_field_fit(self, monkeypatch):
        """Compute Maps produces the map of the fit the crosshair was showing,
        so it runs with whatever Refine was last set to rather than defaults."""
        import spyde.actions.vector_orientation_om as vom
        import spyde.actions.vector_orientation_quantem as quantem

        class _Vecs:
            nav_shape = (2, 2)

        seen = {}
        monkeypatch.setattr(
            quantem, "compute_vector_orientation_quantem",
            lambda *a, **k: (seen.update(k), "RESULT")[1])
        wizard = types.SimpleNamespace(
            lib=types.SimpleNamespace(inverse_angstrom_factor=1.0),
            phases=[object()], voltage=200.0, recip_r=1.5)
        vom._fit_field(_Vecs(), wizard,
                       {"pair_distance": 0.07, "sigma_excitation": 0.03})
        assert seen["params"] == {"pair_distance": 0.07, "sigma_excitation": 0.03}



class TestRefineIpfTogglesWithTheAction:
    """Closing the IPF heat map must not retire it.

    Its window is registered with a controller so ✕ tears down the overlay,
    which is right — nothing should keep correlating for a window nobody is
    looking at. But that left the heat map gone until the library was rebuilt,
    which is a minute of work to undo a click. Re-selecting the action brings
    it back, which is what a toggle is supposed to mean.
    """

    @staticmethod
    def _wizard(refine_ipf, fitter=object()):
        from spyde.actions.vector_orientation_om import VomWizard

        tree = types.SimpleNamespace(
            root=object(), diffraction_vectors=object(), _vom_wizard=None)
        wizard = VomWizard(
            session=None, tree=tree, phases=[], overlay=None,
            voltage=200.0, recip_r=1.5,
            refine_ipf=refine_ipf, fitter=fitter)
        return wizard

    def _patched(self, monkeypatch):
        """Count re-opens instead of building a plan and a window."""
        opened = []
        import spyde.actions.vector_refine_ipf as module

        def fake_open(session, signal, fitter, phases, vectors, tree):
            opened.append(True)
            return types.SimpleNamespace(_closed=False, node=object())

        monkeypatch.setattr(module, "open_refine_ipf", fake_open)
        return opened

    def test_a_closed_heat_map_is_reopened(self, monkeypatch):
        opened = self._patched(monkeypatch)
        wizard = self._wizard(types.SimpleNamespace(_closed=True, node=None))
        wizard.ensure_refine_ipf(session=None)
        assert len(opened) == 1
        assert wizard.refine_ipf._closed is False

    def test_a_heat_map_never_opened_is_opened(self, monkeypatch):
        opened = self._patched(monkeypatch)
        wizard = self._wizard(None)
        wizard.ensure_refine_ipf(session=None)
        assert len(opened) == 1

    def test_a_live_heat_map_is_left_alone(self, monkeypatch):
        """Re-selecting the action with the window already up must not stack a
        second one on top of it."""
        opened = self._patched(monkeypatch)
        live = types.SimpleNamespace(_closed=False, node=object())
        wizard = self._wizard(live)
        wizard.ensure_refine_ipf(session=None)
        assert opened == []
        assert wizard.refine_ipf is live

    def test_nothing_is_reopened_without_a_fitter(self, monkeypatch):
        """No matcher means no correlations to draw; a window would be empty."""
        opened = self._patched(monkeypatch)
        wizard = self._wizard(None, fitter=None)
        wizard.ensure_refine_ipf(session=None)
        assert opened == []

    def test_a_torn_down_wizard_reopens_nothing(self, monkeypatch):
        """The tree is closing; re-selecting must not resurrect its windows."""
        opened = self._patched(monkeypatch)
        wizard = self._wizard(None)
        wizard._closed = True
        wizard.ensure_refine_ipf(session=None)
        assert opened == []
