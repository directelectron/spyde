"""
3-D IPF explorer backend: `SpyDEOrientationMap.ipf_sphere_points` returns reduced
crystal directions ON the unit sphere + matching IPF RGB, and `ipf_view`
builds/emits a `view="3d"` scatter figure for the IPF window.
"""
from __future__ import annotations

import numpy as np


def _al_orientation_map(ny=4, nx=5):
    from orix.crystal_map import Phase
    from orix.quaternion import Rotation
    from diffpy.structure import Atom, Lattice, Structure
    from spyde.signals.orientation_map import SpyDEOrientationMap, phase_to_dict

    structure = Structure(atoms=[Atom("Al", [0, 0, 0])],
                          lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90))
    phase = Phase(name="Al", space_group=225, structure=structure)

    rng = np.random.RandomState(0)
    # Random unit quaternions so the sphere points spread over the sector.
    q = rng.randn(ny, nx, 1, 4).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    corr = np.ones((ny, nx, 1), np.float32)
    phase_idx = np.zeros((ny, nx, 1), np.int16)
    mirror = np.ones((ny, nx, 1), np.int8)
    return SpyDEOrientationMap(q, corr, phase_idx, mirror, [phase_to_dict(phase)])


class TestIpf3D:
    def test_sphere_points_on_unit_sphere(self):
        om = _al_orientation_map()
        xyz, rgb = om.ipf_sphere_points("z")
        assert xyz.shape[1] == 3 and rgb.shape[1] == 3
        assert xyz.shape[0] == rgb.shape[0] > 0
        # Every reduced direction is a UNIT vector (point on the sphere).
        norms = np.linalg.norm(xyz, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3)
        assert rgb.dtype == np.uint8

    def test_build_ipf_key_figure(self):
        # The colour-key triangle legend builds as a native anyplotlib figure.
        from spyde.actions.ipf_view import build_ipf_key_figure
        fig, fig_id, html = build_ipf_key_figure(_al_orientation_map(), "z")
        assert isinstance(fig_id, str) and fig_id
        assert isinstance(html, str) and "<body>" in html
        assert len(html) > 2000                      # a real figure, not empty

    def test_emit_ipf_key_message(self):
        # emit_ipf_key posts a `figure` message tagged view="ipf_key" (the legend
        # is now a native anyplotlib figure, not a matplotlib PNG data URL).
        import spyde.backend.ipc as ipc
        from spyde.actions.ipf_view import emit_ipf_key
        captured, orig = [], ipc.emit
        ipc.emit = lambda m: captured.append(m)
        try:
            assert emit_ipf_key(77, _al_orientation_map(), "z") is True
        finally:
            ipc.emit = orig
        figs = [m for m in captured if m.get("type") == "figure"
                and m.get("view") == "ipf_key"]
        assert figs and figs[-1]["window_id"] == 77
        assert "<body>" in figs[-1]["html"]

    def test_build_3d_figure_html(self):
        om = _al_orientation_map()
        xyz, rgb = om.ipf_sphere_points("z")
        from spyde.actions.ipf_view import build_ipf_3d_figure
        fig, fig_id, html, p3d = build_ipf_3d_figure(xyz, rgb)
        assert isinstance(html, str) and len(html) > 500
        assert isinstance(fig_id, str) and fig_id
        assert p3d is not None                       # the live Plot3D (for set_highlight)

    def test_emit_3d_figure_message(self):
        # emit_ipf_3d posts a `figure` message tagged view="3d".
        import spyde.backend.ipc as ipc
        captured = []
        orig = ipc.emit
        ipc.emit = lambda msg: captured.append(msg)
        try:
            from spyde.actions.ipf_view import emit_ipf_3d
            ok = emit_ipf_3d(7, _al_orientation_map(), "z")
        finally:
            ipc.emit = orig
        assert ok is True
        figs = [m for m in captured if m.get("type") == "figure"]
        assert len(figs) == 1
        assert figs[0]["view"] == "3d" and figs[0]["window_id"] == 7
        assert "<body>" in figs[0]["html"]
