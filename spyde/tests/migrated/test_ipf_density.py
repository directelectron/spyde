"""
IPF density heatmap (inverse pole density function): `ipf_density` builds a
native-anyplotlib `pcolormesh` of the orientation density across the fundamental
sector (one triangle per phase) and emits it as a `view="density"` figure for the
IPF window — the 3rd toggle next to the 2-D map and 3-D sphere.
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
        # The density is drawn as a polygon quad mesh (pcolormesh) with the
        # white sector outline (a lines group) — both serialise into the state.
        assert "polygons" in html
        assert "lines" in html

    def test_density_mesh_has_colored_cells(self):
        # Introspect the PlotXY: the mesh is one polygons group with per-cell
        # fill colours (a PathCollection), clipped to the sector (>0 cells).
        from spyde.actions.ipf_density import build_ipf_density_figure
        fig, _id, _html = build_ipf_density_figure(_al_orientation_map(), "z")
        plot = list(fig._plots_map.values())[0]
        markers = plot.to_state_dict().get("markers", [])
        polys = [m for m in markers if m.get("type") == "polygons"]
        assert polys, "expected a polygon mesh"
        mesh = polys[0]
        assert len(mesh["vertices_list"]) > 0
        assert isinstance(mesh.get("fill_color"), list)      # per-cell colours
        assert len(mesh["fill_color"]) == len(mesh["vertices_list"])
        assert len(set(mesh["fill_color"])) > 3              # a density gradient

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
