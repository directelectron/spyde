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

    def test_sphere_points_not_capped_at_20000(self):
        # A scan bigger than the OLD 20000-point subsample cap must now return
        # every valid pixel (WebGPU instanced points handle up to ~1M).
        om = _al_orientation_map(ny=200, nx=200)          # 40,000 pixels
        xyz, rgb = om.ipf_sphere_points("z")
        assert len(xyz) == len(rgb) == 40_000
        assert len(xyz) > 20_000

    def test_sphere_points_still_caps_at_absurd_size(self):
        # The max_points safety ceiling still strides down an over-large input
        # so it can't blow up memory/transport.  The cap branch is
        # input-size-relative (input > max_points engages the stride-down), so
        # a 40,000-pixel map against max_points=1000 exercises the same path
        # a 1.44M-pixel map against the 1M default did — without 1.44M random
        # quats through orix symmetry reduction (~9 s).
        om = _al_orientation_map(ny=200, nx=200)           # 40,000 pixels
        xyz, rgb = om.ipf_sphere_points("z", max_points=1_000)
        assert len(xyz) == len(rgb) <= 1_000
        assert len(xyz) > 0

    def test_ipf_key_overlay_is_an_rgba_image(self):
        """add_key takes a picture, and the sector mask has to live in ALPHA —
        that is what lets the triangle sit on the map with no square card."""
        from spyde.actions.ipf_view import ipf_key_overlay
        rgba, _labels = ipf_key_overlay(_al_orientation_map(), "z")
        assert rgba.ndim == 3 and rgba.shape[2] == 4
        assert rgba.dtype == np.uint8
        alpha = rgba[..., 3]
        assert (alpha == 255).any() and (alpha == 0).any()
        # Coloured by crystal DIRECTION, so the sector is not one flat colour.
        assert len(np.unique(rgba[alpha > 0][:, :3], axis=0)) > 1

    def test_key_labels_are_fractions_of_the_image(self):
        """add_key positions labels in fractions of the KEY image, not in data
        coordinates — that is what makes them track a panel resize."""
        from spyde.actions.ipf_view import ipf_key_overlay
        _rgba, labels = ipf_key_overlay(_al_orientation_map(), "z")
        assert labels, "the [hkl] corner indices should still be drawn"
        for d in labels:
            assert 0.0 <= d["x"] <= 1.0 and 0.0 <= d["y"] <= 1.0, d
            assert isinstance(d["text"], str) and d["text"]

    def test_edge_labels_align_inwards_so_they_do_not_clip(self):
        """Two of the cubic sector's three corners sit hard against the key's
        edge; centre-aligned text there is cut off by the panel."""
        from spyde.actions.ipf_view import ipf_key_overlay
        _rgba, labels = ipf_key_overlay(_al_orientation_map(), "z")
        for d in labels:
            if d["x"] > 0.75:
                assert d["align"] == "right", d
            elif d["x"] < 0.25:
                assert d["align"] == "left", d
            assert 0.02 <= d["x"] <= 0.98, d
            assert 0.06 <= d["y"] <= 0.94, d

    def test_attach_ipf_key_registers_one_key_on_the_plot(self):
        """The key rides on the map figure now, instead of being a second
        figure floated over the window by the renderer."""
        import anyplotlib as apl
        from spyde.actions.ipf_view import attach_ipf_key

        fig, axes = apl.subplots(1, 1)
        ax = axes[0][0] if isinstance(axes, list) else axes
        p = ax.imshow(np.zeros((8, 8), np.float32))
        assert attach_ipf_key(p, _al_orientation_map(), "z") is True
        keys = p.list_keys()
        assert len(keys) == 1

    def test_the_key_is_hover_only_by_default(self):
        """It should not sit permanently on top of the data."""
        import anyplotlib as apl
        from spyde.actions.ipf_view import attach_ipf_key

        fig, axes = apl.subplots(1, 1)
        ax = axes[0][0] if isinstance(axes, list) else axes
        p = ax.imshow(np.zeros((8, 8), np.float32))
        attach_ipf_key(p, _al_orientation_map(), "z")
        assert p.get_key("ipf_key").hover_only is True

    def test_no_ipf_key_figure_is_emitted_any_more(self):
        """The separate `view="ipf_key"` figure is gone — a regression here
        means two colour keys on screen, or a stray iframe with no renderer
        left to place it."""
        import spyde.backend.ipc as ipc
        from spyde.actions.ipf_view import attach_ipf_3d

        class _Plot:
            window_id = 77

            def set_view_tag(self, *a, **k):
                pass

        class _Tree:
            signal_plots = [_Plot()]

        captured, orig = [], ipc.emit
        ipc.emit = lambda m: captured.append(m)
        try:
            attach_ipf_3d(_Tree(), _al_orientation_map(), "z")
        finally:
            ipc.emit = orig
        assert captured, "the fallback path should still emit its view figures"
        assert not [m for m in captured if m.get("view") == "ipf_key"]

    def test_build_3d_figure_html(self):
        om = _al_orientation_map()
        xyz, rgb = om.ipf_sphere_points("z")
        from spyde.actions.ipf_view import build_ipf_3d_figure
        fig, fig_id, html, p3d = build_ipf_3d_figure(xyz, rgb)
        assert isinstance(html, str) and len(html) > 500
        assert isinstance(fig_id, str) and fig_id
        assert p3d is not None                       # the live Plot3D (for set_highlight)

    def test_build_3d_figure_forces_gpu(self):
        # scatter3d(..., gpu=True) must reach Plot3D so it renders on the
        # WebGPU instanced-points pipeline instead of gpu="auto" (which would
        # fall back to Canvas2D below anyplotlib's ~20k-point threshold — and
        # the sphere now carries every nav pixel, not a 20k subsample).
        om = _al_orientation_map()
        xyz, rgb = om.ipf_sphere_points("z")
        from spyde.actions.ipf_view import build_ipf_3d_figure
        _fig, _fig_id, _html, p3d = build_ipf_3d_figure(xyz, rgb)
        assert p3d._state["gpu_mode"] == "always"

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
