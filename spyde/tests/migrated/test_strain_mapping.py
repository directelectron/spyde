"""
Strain mapping from diffraction vectors (`spyde.actions.strain_mapping`):
the −g=g, center-robust deformation-gradient fit and the per-pixel field.
"""
from __future__ import annotations

import numpy as np

from spyde.actions.strain_mapping import (
    fit_pattern_strain, compute_strain_field, principal_strain, StrainField,
)

# A small multi-ring reference lattice (square, 1st + 2nd ring; non-collinear).
G_REF = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
                  [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])


def _strained(T, g=G_REF, *, offset=(0.0, 0.0)):
    return (g @ np.asarray(T).T) + np.asarray(offset)


class TestStrainFit:
    def test_recovers_known_strain(self):
        T = np.array([[1.01, 0.0], [0.0, 0.995]])      # +1% x, −0.5% y
        exx, eyy, exy, omega, cov = fit_pattern_strain(_strained(T), G_REF, tol=0.3)
        assert abs(exx - 0.01) < 1e-4
        assert abs(eyy + 0.005) < 1e-4
        assert abs(exy) < 1e-4
        assert cov == 1.0

    def test_center_offset_does_not_leak_into_strain(self):
        # An off-centre diffraction pattern: every peak shifted by a constant.
        # The translation term must absorb it — strain unchanged (the −g=g point).
        T = np.array([[1.008, 0.002], [0.002, 0.996]])
        clean = fit_pattern_strain(_strained(T), G_REF, tol=0.3)
        shifted = fit_pattern_strain(_strained(T, offset=(0.35, -0.22)), G_REF, tol=0.3)
        for a, b in zip(clean[:4], shifted[:4]):
            assert abs(a - b) < 1e-6                    # identical despite the offset

    def test_friedel_matches_minus_g(self):
        # Pattern shows only the −g half of each reflection; ±g matching recovers T.
        T = np.array([[1.02, 0.0], [0.0, 1.0]])
        g_meas = _strained(T, g=-G_REF)
        exx, eyy, exy, omega, cov = fit_pattern_strain(g_meas, G_REF, tol=0.3)
        assert abs(exx - 0.02) < 1e-4 and abs(eyy) < 1e-4

    def test_pure_rotation_is_not_strain(self):
        th = np.deg2rad(2.0)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        exx, eyy, exy, omega, cov = fit_pattern_strain(_strained(R), G_REF, tol=0.3)
        assert abs(exx) < 1e-4 and abs(eyy) < 1e-4 and abs(exy) < 1e-4
        assert abs(omega - th) < 1e-3                   # rotation captured separately

    def test_too_few_matches_returns_none(self):
        assert fit_pattern_strain(np.array([[5.0, 5.0]]), G_REF, tol=0.1) is None


class _MockVecs:
    """Duck-typed SpyDEDiffractionVectors: nav_shape + kxy_at(iy, ix)."""
    def __init__(self, nav_shape, T_of):
        self.nav_shape = nav_shape
        self._T_of = T_of

    def kxy_at(self, iy, ix):
        return _strained(self._T_of(iy, ix))


class TestStrainField:
    def test_linear_gradient_field(self):
        ny, nx = 4, 5
        # εxx grows linearly with ix; reference at (0,0) is unstrained.
        T_of = lambda iy, ix: np.array([[1.0 + 0.01 * ix, 0.0], [0.0, 1.0]])
        field = compute_strain_field(_MockVecs((ny, nx), T_of), (0, 0), tol=0.3)
        assert isinstance(field, StrainField) and field.nav_shape == (ny, nx)
        assert abs(field.exx[0, 0]) < 1e-4                       # reference unstrained
        assert field.exx[0, 4] > field.exx[0, 1] > field.exx[0, 0]   # gradient
        assert np.allclose(field.exx[2, :], field.exx[0, :], atol=1e-5)  # no y dependence
        assert np.nanmax(field.coverage) == 1.0

    def test_principal_strain_axes(self):
        e1, e2, theta = principal_strain(np.array([0.02]), np.array([0.0]), np.array([0.0]))
        assert abs(e1[0] - 0.02) < 1e-9 and abs(e2[0]) < 1e-9
        assert abs(theta[0]) < 1e-9                              # ε1 along x
        # 45° pure shear → principal axes at 45°
        e1, e2, theta = principal_strain(np.array([0.0]), np.array([0.0]), np.array([0.01]))
        assert abs(e1[0] - 0.01) < 1e-9 and abs(e2[0] + 0.01) < 1e-9
        assert abs(abs(theta[0]) - np.deg2rad(45)) < 1e-6


class TestStrainDisplay:
    def _field(self, ny=12, nx=12):
        rng = np.random.RandomState(0)
        return StrainField(
            (0.01 * rng.rand(ny, nx)).astype("f4"),
            (-0.01 * rng.rand(ny, nx)).astype("f4"),
            (0.005 * rng.rand(ny, nx)).astype("f4"),
            (0.01 * rng.rand(ny, nx)).astype("f4"),
            np.ones((ny, nx), "f4"))

    def test_build_strain_figure_map_glyphs_and_ref(self):
        from spyde.actions.strain_display import build_strain_figure
        fig, fid, html, p, g = build_strain_figure(
            self._field(), component="exx", ref_yx=(1, 1), glyph_step=3)
        assert isinstance(fid, str) and fid and isinstance(html, str) and len(html) > 500
        types = {m["type"] for m in p.list_markers()}
        assert "ellipses" in types          # principal-strain glyphs
        assert "lines" in types             # the reference crosshair

    def test_each_component_builds(self):
        from spyde.actions.strain_display import build_strain_figure
        for comp in ("exx", "eyy", "exy", "omega"):
            fig, fid, html, p, g = build_strain_figure(
                self._field(), component=comp, glyphs=False)
            assert fid and "ellipses" not in {m["type"] for m in p.list_markers()}


class _MockVecsCM(_MockVecs):
    """_MockVecs + count_map (for the default-reference pick)."""
    def __init__(self, nav_shape, T_of, npk=8):
        super().__init__(nav_shape, T_of)
        self._npk = npk

    def count_map(self):
        return np.full(self.nav_shape, self._npk, dtype=int)


class TestStrainAction:
    def test_strain_run_emits_window_and_attaches_controller(self):
        import spyde.backend.ipc as ipc
        from spyde.actions.strain_action import strain_run

        vecs = _MockVecsCM((6, 6), lambda iy, ix: np.array([[1 + 0.01 * ix, 0.0],
                                                            [0.0, 1.0]]))
        tree = type("T", (), {"diffraction_vectors": vecs})()
        plot = type("P", (), {"signal_tree": tree})()
        session = type("S", (), {"_w": 0,
                                 "next_window_id": lambda self: setattr(self, "_w", self._w + 1) or self._w})()

        cap, orig = [], ipc.emit
        ipc.emit = lambda m: cap.append(m)
        try:
            strain_run(session, plot, {})
        finally:
            ipc.emit = orig

        figs = [m for m in cap if m.get("type") == "figure"]
        assert figs and "Strain" in figs[-1]["title"]
        assert figs[-1]["strain_components"] == ["exx", "eyy", "exy", "omega"]

        ctrl = getattr(tree, "_strain_controller", None)
        assert ctrl is not None and ctrl.field is not None
        ctrl.set_component("eyy")                       # toggle — no error
        ctrl.set_reference(2, 3)                        # move reference — recompute
        assert ctrl.ref_yx == (2, 3) and ctrl.component == "eyy"

    def test_strain_run_without_vectors_errors(self):
        import spyde.backend.ipc as ipc
        from spyde.actions.strain_action import strain_run
        tree = type("T", (), {"diffraction_vectors": None})()
        plot = type("P", (), {"signal_tree": tree})()
        cap, orig = [], ipc.emit
        ipc.emit = lambda m: cap.append(m)
        try:
            strain_run(object(), plot, {})
        finally:
            ipc.emit = orig
        assert not [m for m in cap if m.get("type") == "figure"]
        assert any(m.get("type") == "error" for m in cap)
