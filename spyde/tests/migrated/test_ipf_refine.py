"""
test_ipf_refine.py — the live IPF correlation heatmap + region-mask compute for
the OM refine step (`spyde.actions.ipf_refine`), on the real Silver .cif library.
"""
import os

import numpy as np
import hyperspy.api as hs

CIF = os.path.join(os.path.dirname(__file__), "..", "Silver__0011135.cif")


def _sig(nav=(3, 3), sig=(64, 64), scale=0.0134):
    rng = np.random.RandomState(0)
    s = hs.signals.Signal2D(rng.rand(*nav, *sig).astype(np.float32))
    s.set_signal_type("electron_diffraction")
    for ax in s.axes_manager.signal_axes:
        ax.scale = scale
        ax.offset = -(ax.size / 2.0) * scale
        ax.units = "$A^{-1}$"
    return s


def _lib():
    from orix.crystal_map import Phase
    from spyde.actions.orientation_compute import (
        generate_library_from_phases, build_matching_cache, template_tables,
    )
    from spyde.actions.orientation_action import _reciprocal_radius
    phase = Phase.from_cif(CIF)
    s = _sig()
    rr = _reciprocal_radius(s)
    sim = generate_library_from_phases(
        phases=[phase], accelerating_voltage=200.0, resolution=10.0,
        minimum_intensity=1e-4, reciprocal_radius=rr)
    cache = build_matching_cache(s, sim)
    n_templates = template_tables(sim)[0].shape[0]
    return s, sim, cache, n_templates


class TestIpfRefine:
    def test_build_phase_ipf_geometry(self):
        from spyde.actions import ipf_refine
        _s, sim, _c, n = _lib()
        infos = ipf_refine.build_phase_ipf(sim)
        assert len(infos) == 1                       # single phase (Ag)
        info = infos[0]
        assert info["lib_idx"].shape[0] == info["xs"].shape[0] == n
        assert info["tri_xy"].shape[1] == 2 and len(info["tri_xy"]) >= 3
        assert info["labels"]                        # [001]/[101]/[111]-style corners
        # the templates project INSIDE the triangle bounding box
        assert (info["xs"] >= info["mins"][0] - 1e-6).all()
        assert (info["xs"] <= info["maxs"][0] + 1e-6).all()

    def test_match_correlations_full_length(self):
        from spyde.actions import ipf_refine
        s, sim, cache, n = _lib()
        pat = np.asarray(s.data[1, 1], float)
        corr, best = ipf_refine.match_correlations(pat, sim, cache, gamma=0.5)
        assert corr.shape == (n,)
        assert corr.max() > 0 and np.isfinite(corr).all()
        assert int(best[0]) == int(np.argmax(corr))   # best row == argmax template

    def test_native_heatmap_colours_vary(self):
        """The native PlotXY render paints one polygon per inside-sector cell and
        colours them by correlation — a real heatmap (varied), not flat."""
        from spyde.actions import ipf_refine
        from spyde.actions.ipf_refine_render import (
            build_refine_figure, update_panels, best_xy_for, _mesh_geometry,
        )
        s, sim, cache, _n = _lib()
        infos = ipf_refine.build_phase_ipf(sim)
        corr, best = ipf_refine.match_correlations(
            np.asarray(s.data[0, 0], float), sim, cache, gamma=0.5)

        _fig, _fid, _html, panels = build_refine_figure(infos)
        # the mesh covers the inside-sector cells (some cells, not the whole grid)
        verts, cells = _mesh_geometry(infos[0])
        assert 0 < len(cells) < ipf_refine.GRID_N ** 2

        update_panels(panels, corr, {i["phase_index"]: [] for i in infos},
                      best_xy_for(infos, int(best[0])))
        faces = panels[0]["mesh"]._data.get("facecolors")
        assert faces is not None and len(faces) == len(cells)
        assert len(set(faces)) > 3                      # colour variation
        # the best-match marker got placed
        assert len(np.asarray(panels[0]["best"]._data.get("offsets"))) == 1

    def test_rot_mask_from_circles_restricts(self):
        from spyde.actions import ipf_refine
        s, sim, cache, n = _lib()
        infos = ipf_refine.build_phase_ipf(sim)
        info = infos[0]
        corr, best = ipf_refine.match_correlations(np.asarray(s.data[1, 1], float),
                                                   sim, cache, gamma=0.5)
        # A small circle around the best-match template's IPF point.
        bi = int(np.argmax(corr))
        j = int(np.where(info["lib_idx"] == bi)[0][0])
        cx, cy = float(info["xs"][j]), float(info["ys"][j])
        mask = ipf_refine.rot_mask_from_circles(infos, {0: [(cx, cy, 0.05)]}, n)
        assert mask is not None and mask.dtype == bool and mask.shape == (n,)
        assert mask[bi]                                 # best match kept
        assert 0 < mask.sum() < n                       # but it's a real restriction
        # matching still works under the mask, and stays within the kept set.
        corr2, best2 = ipf_refine.match_correlations(
            np.asarray(s.data[1, 1], float), sim, cache, gamma=0.5, rot_mask=mask)
        assert mask[int(best2[0])]

    def test_no_circles_returns_none(self):
        from spyde.actions import ipf_refine
        infos = ipf_refine.build_phase_ipf(_lib()[1])
        assert ipf_refine.rot_mask_from_circles(infos, {}, 100) is None


class TestRefineController:
    def test_controller_recompute_and_double_click_toggle(self):
        from spyde.actions import ipf_refine
        from spyde.actions.ipf_refine_render import (
            build_refine_figure, RefineIpfController,
        )
        s, sim, cache, _n = _lib()
        infos = ipf_refine.build_phase_ipf(sim)
        _fig, _fid, _html, panels = build_refine_figure(infos)

        ctrl = RefineIpfController(None, s, sim, cache, infos, panels,
                                   gamma=0.6, normalize=False)
        ctrl._last_iyix = (1, 1)
        ctrl._recompute()                                # live paint — no error
        assert ctrl.circles == {0: []}

        info = infos[0]
        cx = float(info["mins"][0] + 0.4 * (info["maxs"][0] - info["mins"][0]))
        cy = float(info["mins"][1] + 0.3 * (info["maxs"][1] - info["mins"][1]))
        ctrl.toggle_circle(0, cx, cy)                    # double-click adds a region
        assert len(ctrl.circles[0]) == 1
        ctrl.toggle_circle(0, cx, cy)                    # double-click inside removes it
        assert len(ctrl.circles[0]) == 0

        ctrl.set_refine_params(gamma=0.9, normalize=True)
        assert ctrl.gamma == 0.9 and ctrl.normalize is True
