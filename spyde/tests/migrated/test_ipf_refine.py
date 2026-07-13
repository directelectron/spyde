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

    def test_build_uses_single_raster_not_polygons(self):
        """The heatmap is ONE RGBA raster (not ~9k recoloured polygons), and the
        static markers (mask circles, outline, corner labels, best-match scatter)
        still exist alongside it."""
        from spyde.actions import ipf_refine
        from spyde.actions.ipf_refine_render import build_refine_figure

        _s, sim, _cache, _n = _lib()
        infos = ipf_refine.build_phase_ipf(sim)
        _fig, _fid, _html, panels = build_refine_figure(infos)
        panel = panels[0]

        # Heatmap is a single raster marker of the right shape (GRID_N × GRID_N).
        raster = panel["raster"]
        assert raster._type == "raster"
        assert raster._data["image_width"] == ipf_refine.GRID_N
        assert raster._data["image_height"] == ipf_refine.GRID_N
        assert raster._data.get("image_b64")               # pixels present
        assert raster._data.get("clip_path") is not None   # clipped to the sector
        assert panel.get("mesh") is None                   # the polygon mesh is gone

        # The cheap decorations are untouched.
        assert panel["circle_grp"]._type == "polygons"     # mask circles
        assert panel["best"] is not None                   # best-match scatter
        info = panel["info"]
        assert info["tri_xy"].shape[1] == 2                # outline
        assert info["labels"]                              # corner labels

    def test_corr_rgba_lut_alpha_and_orientation(self):
        """`_corr_rgba`: LUT-maps correlation to colour, sets outside-sector cells
        to alpha 0, and orients rows so grid (0,0)/(0,n-1) land at the extent
        corners the old polygon layout used (row 0 = bottom = mins[1])."""
        from spyde.actions.ipf_refine_render import _corr_rgba, _lut

        lut = _lut()
        n = 4
        # Everything inside except one outside cell; distinct values at the two
        # bottom corners so we can check orientation unambiguously.
        outside = np.zeros((n, n), dtype=bool)
        outside[2, 2] = True
        vals = np.zeros((n, n), dtype=float)
        vals[0, 0] = 1.0                # grid bottom-left (mins[0], mins[1])
        vals[0, n - 1] = 0.5            # grid bottom-right (maxs[0], mins[1])

        rgba = _corr_rgba(vals, outside, lut)
        assert rgba.shape == (n, n, 4) and rgba.dtype == np.uint8

        # LUT colour: value 1.0 → lut[255], value 0.5 → lut[~127].
        top_val = np.clip(np.round(1.0 * 255), 0, 255).astype(int)
        half_val = np.clip(np.round(0.5 * 255), 0, 255).astype(int)

        # Orientation: raster row 0 = TOP of extent (origin="upper"), grid row 0 =
        # BOTTOM (mins[1]). So grid (0, j) → raster IMAGE row (n-1).
        # grid (0,0)  → image (n-1, 0)     value 1.0
        assert np.array_equal(rgba[n - 1, 0, :3], lut[top_val])
        assert rgba[n - 1, 0, 3] == 255
        # grid (0,n-1) → image (n-1, n-1)  value 0.5
        assert np.array_equal(rgba[n - 1, n - 1, :3], lut[half_val])
        assert rgba[n - 1, n - 1, 3] == 255

        # Outside cell grid (2,2) → image row (n-1-2)=1, col 2 → alpha 0.
        assert rgba[n - 1 - 2, 2, 3] == 0
        # All other cells opaque.
        opaque = rgba[..., 3] == 255
        assert opaque.sum() == n * n - 1

    def test_corr_rgba_matches_polygon_cell_position(self):
        """The raster places each correlation cell at the same (x, y) the old
        `_mesh_geometry` polygon for that cell centred on — using the real sector
        geometry: cell (i, j) sits at y = linspace(mins[1], maxs[1], n)[i], and
        the raster (origin='upper') maps it to image row (n-1-i)."""
        from spyde.actions import ipf_refine
        from spyde.actions.ipf_refine_render import _corr_rgba, _lut, _mesh_geometry

        _s, sim, _cache, _n = _lib()
        info = ipf_refine.build_phase_ipf(sim)[0]
        n = int(info["grid_n"])
        outside = np.asarray(info["outside"]).reshape(n, n)
        lut = _lut()

        verts, cells = _mesh_geometry(info)
        assert 0 < len(cells) < ipf_refine.GRID_N ** 2   # a real triangular subset

        # Put a unique ramp value at each inside cell; check the raster pixel at
        # the y-flipped row/col carries that cell's colour, and its data-space
        # centre matches the polygon centroid for that cell.
        vals = np.zeros((n, n), dtype=float)
        for k, (i, j) in enumerate(cells):
            vals[i, j] = (k % 200 + 1) / 255.0          # in (0, 1], distinct-ish
        rgba = _corr_rgba(vals, outside, lut)

        ex = np.linspace(float(info["mins"][0]), float(info["maxs"][0]), n + 1)
        ey = np.linspace(float(info["mins"][1]), float(info["maxs"][1]), n + 1)
        # spot-check a handful of inside cells (verts[k] ↔ cells[k], same order)
        step = max(1, len(cells) // 20)
        for k in range(0, len(cells), step):
            i, j = int(cells[k][0]), int(cells[k][1])
            code = np.clip(np.round(vals[i, j] * 255), 0, 255).astype(int)
            img_row = n - 1 - i                          # origin="upper" flip
            assert np.array_equal(rgba[img_row, j, :3], lut[code])
            assert rgba[img_row, j, 3] == 255
            # polygon centroid for this cell (mesh spanned [ex[j],ex[j+1]] × [ey[i],ey[i+1]])
            cx, cy = np.asarray(verts[k]).mean(0)
            assert ex[j] <= cx <= ex[j + 1]
            assert ey[i] <= cy <= ey[i + 1]

    def test_update_panels_pushes_image_not_facecolors(self):
        """The live update swaps the raster's pixels (`image_b64`) — NOT per-cell
        facecolors — and still moves the best-match marker + mask circles."""
        from spyde.actions import ipf_refine
        from spyde.actions.ipf_refine_render import (
            build_refine_figure, update_panels, best_xy_for,
        )
        s, sim, cache, _n = _lib()
        infos = ipf_refine.build_phase_ipf(sim)
        corr, best = ipf_refine.match_correlations(
            np.asarray(s.data[0, 0], float), sim, cache, gamma=0.5)

        _fig, _fid, _html, panels = build_refine_figure(infos)
        panel = panels[0]
        blank_b64 = panel["raster"]._data.get("image_b64")

        # A mask circle so the circle-update path is exercised too.
        circles = {i["phase_index"]: [] for i in infos}
        circles[infos[0]["phase_index"]] = [
            (float(infos[0]["xs"][0]), float(infos[0]["ys"][0]), 0.05)]

        update_panels(panels, corr, circles, best_xy_for(infos, int(best[0])))

        # Heatmap: the raster's stored bytes changed and NO facecolors were pushed.
        painted_b64 = panel["raster"]._data.get("image_b64")
        assert painted_b64 is not None and painted_b64 != blank_b64
        assert "facecolors" not in panel["raster"]._data
        assert "edgecolors" not in panel["raster"]._data
        assert panel["raster"]._data["image_width"] == ipf_refine.GRID_N

        # Decoded pixels vary (a real heatmap, not flat) and carry transparency.
        import base64
        raw = np.frombuffer(base64.b64decode(painted_b64), dtype=np.uint8)
        img = raw.reshape(ipf_refine.GRID_N, ipf_refine.GRID_N, 4)
        assert len(np.unique(img[..., :3].reshape(-1, 3), axis=0)) > 3
        assert (img[..., 3] == 0).any() and (img[..., 3] == 255).any()

        # The best-match marker got placed; the mask circle got drawn.
        assert len(np.asarray(panel["best"]._data.get("offsets"))) == 1
        assert len(panel["circle_grp"]._data.get("vertices_list")) == 1

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
