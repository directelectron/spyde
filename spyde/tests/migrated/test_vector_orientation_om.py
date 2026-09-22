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
                template = value["template"]
                assert template["data"].shape[1] == 2   # a matched pattern
                # Each spot draws at its own intensity, as an alpha.
                assert len(template["edgecolors"]) == template["data"].shape[0]
                assert all(c.startswith("rgba(48,255,96,") for c in template["edgecolors"])

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
            # The orientation window opens BEFORE the fit (blank, to fill in
            # band by band), so a new tree proves nothing; the result landing
            # on it does.
            assert _wait(lambda: len(session.signal_trees) >= n_before + 1,
                         timeout=90), "orientation window never opened"

            def _ipf_tree():
                return next((t for t in session.signal_trees
                             if getattr(t, "vector_orientation", None) is not None),
                            None)

            assert _wait(lambda: _ipf_tree() is not None, timeout=90), \
                "orientation map never landed"
            ipf_tree = _ipf_tree()
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



class TestTheMatchIsBanded:
    """The whole-field match runs a band of rows at a time so the map fills
    in as it goes. The match is per position, so this must be the same answer
    the scan gives matched whole — banding is a display affordance, not a
    different fit."""

    @staticmethod
    def _vectors(ny=3, nx=4):
        """A scan of one silver-like pattern: six spots at Ag {111}/{200}
        radii (Å⁻¹), the same at every position, so every band has peaks to
        match."""
        from spyde.signals.diffraction_vectors import (
            SpyDEDiffractionVectors, N_COLS,
        )
        angles = np.deg2rad(np.arange(0, 360, 60))
        spots = [(0.0, 0.0)] + [(0.42 * np.cos(a), 0.42 * np.sin(a))
                                for a in angles]
        rows, offsets = [], [0]
        for iy in range(ny):
            for ix in range(nx):
                for kx, ky in spots:
                    rows.append([ix, iy, kx, ky, -1.0, 1.0])
                offsets.append(len(rows))
        flat = np.asarray(rows, dtype=np.float32).reshape(-1, N_COLS)
        off = np.asarray(offsets, dtype=np.int64)
        return SpyDEDiffractionVectors(
            flat_buffer=flat, nav_offsets=[np.arange(ny + 1) * nx, off],
            nav_shape=(ny, nx), full_nav_shape=(ny, nx), sig_shape=(64, 64),
            sig_axes=None, kernel_radius_px=1.0, kernel_radius_data=0.02,
            offsets=off)

    def test_banded_equals_whole(self):
        from orix.crystal_map import Phase

        from spyde.actions.vector_orientation_quantem import (
            compute_vector_orientation_quantem,
        )

        vectors = self._vectors()
        phases = [Phase.from_cif(CIF)]
        bands = []

        def run(band_rows, on_band=None):
            return compute_vector_orientation_quantem(
                vectors, phases, energy_ev=200e3, k_max=1.2,
                angle_step_zone_axis_deg=12.0, device="cpu",
                band_rows=band_rows, on_band=on_band)

        whole = run(band_rows=vectors.nav_shape[0])
        banded = run(band_rows=1, on_band=lambda *a: bands.append(a))

        assert whole is not None and banded is not None
        np.testing.assert_array_equal(banded.phase_idx, whole.phase_idx)
        np.testing.assert_allclose(banded.coarse_score, whole.coarse_score,
                                   rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(banded.quats, whole.quats, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(banded.strain, whole.strain,
                                   rtol=1e-5, atol=1e-5, equal_nan=True)

        # One band per row, each carrying that row's orientations.
        assert [(a[0], a[1]) for a in bands] == [(r, r + 1) for r in range(3)]
        for row_start, row_stop, quats, phase_index in bands:
            assert quats.shape == (1, 4, 4)
            assert phase_index.shape == (1, 4)

    def test_progress_counts_positions(self):
        """The count moves with the bands — 'fitting… 33%' for a whole
        minute was the match stage reporting nothing until it finished."""
        from orix.crystal_map import Phase

        from spyde.actions.vector_orientation_quantem import (
            compute_vector_orientation_quantem,
        )

        seen = []
        compute_vector_orientation_quantem(
            self._vectors(), [Phase.from_cif(CIF)], energy_ev=200e3, k_max=1.2,
            angle_step_zone_axis_deg=12.0, device="cpu", band_rows=1,
            progress=lambda done, total: seen.append((done, total)))
        done = [d for d, _ in seen]
        assert done == sorted(done), "progress must not go backwards"
        # Three bands of four positions matched, then the refinement.
        assert done[:3] == [4, 8, 12]
        assert seen[-1][0] == seen[-1][1]


class TestASingularPositionIsOneNaN:
    """One position whose paired peaks are degenerate must not take the
    scan's strain down. It used to: the batch inverse raised on the one
    singular member ("batch element 1552 … The input matrix is singular")
    and Compute Maps failed as a whole."""

    def test_real_space_deformation_skips_the_singular_member(self):
        import torch
        from spyde.actions.vector_orientation_quantem import real_space_deformation

        affine = torch.tensor([
            [[1.02, 0.01], [0.00, 0.98]],       # a fine map
            [[1.00, 2.00], [0.50, 1.00]],       # rank one: singular
            [[float("nan"), 0.0], [0.0, 1.0]],  # too few pairs
            [[1e-3, 0.0], [0.0, 1e-3]],         # small but well conditioned
        ], dtype=torch.float64)
        real = real_space_deformation(affine)
        expected = torch.linalg.inv(affine[0]).T
        assert torch.allclose(real[0], expected)
        assert torch.isnan(real[1]).all()
        assert torch.isnan(real[2]).all()
        assert torch.allclose(real[3], torch.linalg.inv(affine[3]).T)

    def test_the_field_strain_carries_on_past_it(self, monkeypatch):
        import torch
        import spyde.actions.vector_orientation_quantem as quantem

        affine = torch.full((2, 3, 2, 2), float("nan"), dtype=torch.float64)
        affine[0, 0] = torch.eye(2, dtype=torch.float64) * 1.01
        affine[1, 2] = torch.tensor([[1.0, 2.0], [0.5, 1.0]])   # singular
        pairs = torch.full((2, 3), 6, dtype=torch.long)
        monkeypatch.setattr(quantem, "reciprocal_affine",
                            lambda *a, **k: (affine, pairs))
        strain, _ = quantem.strain_from_orientation_map(object())
        assert strain.shape == (2, 3, 3)
        assert np.isfinite(strain[0, 0]).all()
        assert np.isnan(strain[1, 2]).all(), "the singular position is NaN"
        assert np.isnan(strain[0, 1]).all()


class TestThePhaseMapAndItsChips:
    @staticmethod
    def _result(phase_idx, score):
        from spyde.signals.orientation_map import VectorOrientationResult
        phase_idx = np.asarray(phase_idx, np.int16)
        ny, nx = phase_idx.shape
        return VectorOrientationResult(
            quats=np.tile(np.array([1, 0, 0, 0], np.float32), (ny, nx, 1)),
            phase_idx=phase_idx, theta=np.zeros((ny, nx), np.float32),
            strain=np.full((ny, nx, 3), np.nan, np.float32),
            residual=np.full((ny, nx), np.nan, np.float32),
            friedel_asym=np.full((ny, nx), np.nan, np.float32),
            n_matched=np.zeros((ny, nx), np.int16),
            coarse_score=np.asarray(score, np.float32),
            phases_meta=[{"name": "Cu", "point_group": "m-3m"},
                         {"name": "Nb", "point_group": "m-3m"}],
            nav_shape=(ny, nx))

    def test_a_matched_position_is_coloured_by_its_phase(self):
        """The matcher reports no residual; a position with a correlation is
        fitted. Keyed on the residual alone, every position was grey."""
        from spyde.actions.vector_orientation_om import phase_map_rgb, _PHASE_COLORS, _UNFIT_COLOR
        result = self._result([[0, 1], [1, 0]], [[0.9, 0.8], [0.0, 0.7]])
        rgb = phase_map_rgb(result)
        assert tuple(rgb[0, 0]) == _PHASE_COLORS[0]
        assert tuple(rgb[0, 1]) == _PHASE_COLORS[1]
        assert tuple(rgb[1, 0]) == _UNFIT_COLOR, "no correlation, no phase"

    def test_each_phase_gets_its_own_map(self):
        from spyde.actions.vector_orientation_om import phase_views, _UNFIT_COLOR
        result = self._result([[0, 1], [1, 0]], [[0.9, 0.8], [0.0, 0.7]])
        ipf = np.full((2, 2, 3), 200, np.uint8)
        views = dict(phase_views(result, ipf))
        assert list(views) == ["Phase", "Cu", "Nb"]
        assert tuple(views["Cu"][0, 0]) == (200, 200, 200)
        assert tuple(views["Cu"][0, 1]) == _UNFIT_COLOR
        assert tuple(views["Nb"][0, 1]) == (200, 200, 200)
        assert tuple(views["Nb"][1, 0]) == _UNFIT_COLOR, "unfit is nobody's"

    def test_one_phase_has_no_chips(self):
        from spyde.actions.vector_orientation_om import phase_views
        result = self._result([[0, 0]], [[0.9, 0.8]])
        result.phases_meta = result.phases_meta[:1]
        assert phase_views(result, np.zeros((1, 2, 3), np.uint8)) == []


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
        """Record re-opens instead of building a plan and a window."""
        opened = []
        import spyde.actions.vector_refine_ipf as module

        def fake_open(session, signal, fitter, phases, vectors, tree, **kwargs):
            opened.append(kwargs)
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

    def test_a_reopened_heat_map_can_still_restrict_the_pattern(self, monkeypatch):
        """A mask drawn on the triangle has to redraw the matched pattern, so
        the re-opened heat map needs that overlay — not just the first one
        built at Generate."""
        opened = self._patched(monkeypatch)
        overlay = object()
        wizard = self._wizard(None)
        wizard.overlay = overlay
        wizard.ensure_refine_ipf(session=None)
        assert opened[0].get("fit_overlay") is overlay

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
