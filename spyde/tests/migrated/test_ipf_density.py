"""
IPF density heatmap (inverse pole density function): `ipf_density` builds a
native-anyplotlib raster (resampled from orix's equal-area grid) of the
orientation density across the fundamental sector (one triangle per phase) and
emits it as a `view="density"` figure for the IPF window — the 3rd toggle next
to the 2-D map and 3-D sphere.
"""
from __future__ import annotations

import numpy as np


def _al_orientation_map(ny=6, nx=7, n_phases=1):
    from orix.crystal_map import Phase
    from diffpy.structure import Atom, Lattice, Structure
    from spyde.signals.orientation_map import SpyDEOrientationMap, phase_to_dict

    structure = Structure(atoms=[Atom("Al", [0, 0, 0])],
                          lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90))
    phase = Phase(name="Al", space_group=225, structure=structure)

    rng = np.random.RandomState(0)
    q = rng.randn(ny, nx, 1, 4).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    corr = np.ones((ny, nx, 1), np.float32)
    phase_idx = np.zeros((ny, nx, 1), np.int16)
    if n_phases == 2:                       # split the map across two phases
        phase_idx[:, nx // 2:, 0] = 1
    mirror = np.ones((ny, nx, 1), np.int8)
    phases = [phase_to_dict(phase)] * n_phases
    return SpyDEOrientationMap(q, corr, phase_idx, mirror, phases)


class TestIpfDensity:
    def test_build_density_figure_has_mesh(self):
        from spyde.actions.ipf_density import build_ipf_density_figure
        fig, fig_id, html = build_ipf_density_figure(_al_orientation_map(), "z")
        assert isinstance(fig_id, str) and fig_id
        assert isinstance(html, str) and "<body>" in html
        # The density is now a single stretched RGBA raster (add_raster) — a
        # `raster` marker, not thousands of polygons — plus the white sector
        # outline (a lines group). Both serialise into the state.
        assert "raster" in html
        assert "lines" in html

    def test_density_uses_raster_not_polygon_mesh(self):
        # The equal-area orix grid is resampled onto a regular raster and drawn
        # as ONE `add_raster` image instead of one polygon per histogram cell.
        from spyde.actions.ipf_density import build_ipf_density_figure
        fig, _id, _html = build_ipf_density_figure(_al_orientation_map(), "z")
        plot = list(fig._plots_map.values())[0]
        markers = plot.to_state_dict().get("markers", [])
        rasters = [m for m in markers if m.get("type") == "raster"]
        assert rasters, "expected a raster marker (equal-area grid resampled)"
        r = rasters[0]
        assert r["image_width"] > 1 and r["image_height"] > 1
        assert "clip_path" in r                       # clipped to the sector
        polys = [m for m in markers if m.get("type") == "polygons"]
        assert not polys, "should not also draw the slow per-cell polygon mesh"

    def test_density_raster_known_region_maps_to_sane_color(self):
        # A known high-density direction (the mode of a tight, seeded cluster of
        # crystal directions) should map to a warm/high-value colour in the
        # resampled raster, not to background/transparent — a basic fidelity
        # check on the nearest-neighbour resample.
        from spyde.actions.ipf_density import (
            _resample_density_to_raster, _sector_limits,
        )
        from spyde.signals.orientation_map import ipf_triangle_xy
        from orix.crystal_map import Phase
        from orix.measure import pole_density_function
        from orix.quaternion import Rotation
        from orix.vector import Vector3d
        from diffpy.structure import Atom, Lattice, Structure

        structure = Structure(atoms=[Atom("Al", [0, 0, 0])],
                              lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90))
        phase = Phase(name="Al", space_group=225, structure=structure)

        # A tight cluster of near-identical rotations -> one hot spot.
        rng = np.random.RandomState(0)
        base = np.array([1.0, 0.0, 0.0, 0.0])
        q = base[None, :] + rng.randn(500, 4) * 1e-3
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        t = Rotation(q) * Vector3d.zvector()
        hist, (x, y) = pole_density_function(
            t, symmetry=phase.point_group, resolution=2.0, sigma=5.0,
            log=False, hemisphere="upper",
        )
        xy_edges, _label_xy, _labels = ipf_triangle_xy(phase)
        xlim, ylim = _sector_limits(xy_edges)
        raster = _resample_density_to_raster(x, y, hist, xlim, ylim, "fire", None)
        assert raster is not None
        rgba, extent = raster
        assert rgba.shape[-1] == 4
        # The brightest cell (the resampled hot spot) should be far from
        # black/transparent — a basic fidelity check on the nearest-neighbour
        # resample (not washed out to background).
        brightness = rgba[..., :3].sum(axis=-1)
        assert int(brightness.max()) > 60

    def test_emit_density_message(self):
        import spyde.backend.ipc as ipc
        from spyde.actions.ipf_density import emit_ipf_density
        captured, orig = [], ipc.emit
        ipc.emit = lambda m: captured.append(m)
        try:
            assert emit_ipf_density(91, _al_orientation_map(), "z") is True
        finally:
            ipc.emit = orig
        figs = [m for m in captured if m.get("type") == "figure"]
        assert figs and figs[-1]["view"] == "density"
        assert figs[-1]["window_id"] == 91
        assert "<body>" in figs[-1]["html"]

    def test_multiphase_builds_one_axis_per_phase(self):
        # Two phases present → two sector triangles (subplots(1, 2)).
        from spyde.actions.ipf_density import build_ipf_density_figure
        fig, _id, html = build_ipf_density_figure(_al_orientation_map(n_phases=2), "z")
        assert len(fig._plots_map) == 2                       # one triangle per phase

    def test_attach_ipf_3d_also_emits_density(self):
        # attach_ipf_3d wires the density heatmap onto the IPF window alongside
        # the 3-D explorer (so the frontend gets the PDF toggle).
        import spyde.backend.ipc as ipc
        from spyde.actions.ipf_view import attach_ipf_3d

        class _SP:
            window_id = 55
            _plot2d = None
        class _Tree:
            signal_plots = [_SP()]
        captured, orig = [], ipc.emit
        ipc.emit = lambda m: captured.append(m)
        try:
            attach_ipf_3d(_Tree(), _al_orientation_map(), "z")
        finally:
            ipc.emit = orig
        views = {m.get("view") for m in captured if m.get("type") == "figure"}
        assert "density" in views and "3d" in views
