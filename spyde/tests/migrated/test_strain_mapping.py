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


def _strained(F, g=G_REF, *, offset=(0.0, 0.0)):
    """Measured reciprocal vectors of a lattice under REAL deformation ``F``:
    g_meas = (F⁻ᵀ)·g_ref (+ DP-centre offset). So the fit must recover ``F``'s
    real-space strain (a stretched real lattice ⇒ positive ε)."""
    T = np.linalg.inv(np.asarray(F, dtype=float)).T          # reciprocal transform
    return (g @ T.T) + np.asarray(offset)


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


def _real_vecs(ny, nx, T_of, *, noise=0.0, seed=0):
    """A real SpyDEDiffractionVectors 4D container (so the CSR/vectorized strain
    path is exercised, not the _MockVecs loop fallback)."""
    from spyde.signals.diffraction_vectors import SpyDEDiffractionVectors, N_COLS
    rng = np.random.default_rng(seed)
    rows, offsets = [], [0]
    for iy in range(ny):
        for ix in range(nx):
            g = _strained(T_of(iy, ix))
            if noise:
                g = g + rng.normal(0.0, noise, g.shape)
            for kx, ky in g:
                rows.append([ix, iy, kx, ky, -1.0, 1.0])
            offsets.append(len(rows))
    flat = np.asarray(rows, dtype=np.float32).reshape(-1, N_COLS)
    off = np.asarray(offsets, dtype=np.int64)
    return SpyDEDiffractionVectors(
        flat_buffer=flat, nav_offsets=[np.arange(ny + 1) * nx, off],
        nav_shape=(ny, nx), full_nav_shape=(ny, nx), sig_shape=(64, 64),
        sig_axes=None, kernel_radius_px=1.0, kernel_radius_data=1.0, offsets=off)


class TestStrainFieldVectorized:
    """The whole-field vectorized path (real CSR container) must match the
    per-pixel scipy loop bit-for-bit and recover known strain."""

    def _T_of(self, iy, ix):
        return np.array([[1.0 + 0.01 * ix / 8.0, 0.003],
                         [0.003, 1.0 - 0.005 * iy / 8.0]])

    def test_vectorized_matches_loop(self):
        from spyde.actions.strain_mapping import (
            _compute_strain_field_loop, _median_nn,
        )
        v = _real_vecs(8, 8, self._T_of, noise=0.002, seed=3)
        ref = np.asarray(v.kxy_at(0, 0), float).reshape(-1, 2)
        nn = _median_nn(ref)
        tol = 0.25 * nn if nn > 0 else np.inf

        vec = compute_strain_field(v, ref_vectors=ref)       # CSR → vectorized
        loop = _compute_strain_field_loop(v, ref, tol, 8, 8)  # per-pixel reference
        for name in ("exx", "eyy", "exy", "omega", "coverage"):
            a, b = getattr(vec, name), getattr(loop, name)
            assert np.allclose(a, b, atol=1e-5, equal_nan=True), f"{name} mismatch"

    def test_recovers_known_gradient(self):
        v = _real_vecs(6, 10, lambda iy, ix: np.array([[1.0 + 0.01 * ix, 0.0],
                                                       [0.0, 1.0]]))
        field = compute_strain_field(v, (0, 0))
        assert field.nav_shape == (6, 10)
        assert abs(field.exx[0, 0]) < 1e-4                    # reference unstrained
        assert field.exx[0, 9] > field.exx[0, 1] > field.exx[0, 0]   # gradient
        assert np.allclose(field.exx[3, :], field.exx[0, :], atol=1e-5)  # no y dep
        assert np.nanmax(field.coverage) == 1.0


class TestCifReference:
    def _al_phase(self):
        from orix.crystal_map import Phase
        from diffpy.structure import Atom, Lattice, Structure
        st = Structure(atoms=[Atom("Al", [0, 0, 0])],
                       lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90))
        return Phase(name="Al", space_group=225, structure=st)

    def test_families_exclude_forbidden(self):
        from spyde.actions.strain_mapping import cif_g_families
        fam = cif_g_families(self._al_phase(), min_dspacing=0.9)
        assert np.any(np.abs(fam - np.sqrt(3) / 4.05) < 2e-3)   # {111} allowed
        assert np.any(np.abs(fam - 2.0 / 4.05) < 2e-3)          # {200} allowed
        assert not np.any(np.abs(fam - 1.0 / 4.05) < 2e-3)      # {100} forbidden (fcc)

    def test_snap_corrects_magnitude_to_ideal(self):
        from spyde.actions.strain_mapping import cif_g_families, snap_reference_to_cif
        fam = cif_g_families(self._al_phase(), min_dspacing=0.9)
        g = float(fam[np.argmin(np.abs(fam - np.sqrt(3) / 4.05))])
        meas = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], float) * (g * 1.02)
        ref = snap_reference_to_cif(meas, fam)
        assert len(ref) == 4
        assert np.allclose(np.linalg.norm(ref, axis=1), g, atol=1e-3)

    def test_cif_reference_gives_absolute_strain(self):
        # No flat region needed: snap to the ideal CIF spacing → absolute strain.
        from spyde.actions.strain_mapping import (
            cif_g_families, snap_reference_to_cif, fit_pattern_strain)
        fam = cif_g_families(self._al_phase(), min_dspacing=0.6)
        g = float(fam[np.argmin(np.abs(fam - np.sqrt(3) / 4.05))])
        dirs = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], float) * g
        meas = _strained(np.diag([1.01, 0.99]), g=dirs)         # real +1% x, −1% y
        ref = snap_reference_to_cif(meas, fam)                  # ideal magnitudes
        exx, eyy, exy, omega, cov = fit_pattern_strain(meas, ref, tol=0.05)
        assert abs(exx - 0.01) < 3e-3 and abs(eyy + 0.01) < 3e-3


class TestStrainDisplay:
    def _field(self, ny=12, nx=12):
        rng = np.random.RandomState(0)
        return StrainField(
            (0.01 * rng.rand(ny, nx)).astype("f4"),
            (-0.01 * rng.rand(ny, nx)).astype("f4"),
            (0.005 * rng.rand(ny, nx)).astype("f4"),
            (0.01 * rng.rand(ny, nx)).astype("f4"),
            np.ones((ny, nx), "f4"))

    def test_build_strain_figure_map_and_ref(self):
        from spyde.actions.strain_display import build_strain_figure
        fig, fid, html, p = build_strain_figure(
            self._field(), component="exx", ref_yx=(1, 1))
        assert isinstance(fid, str) and fid and isinstance(html, str) and len(html) > 500
        types = {m["type"] for m in p.list_markers()}
        assert "ellipses" not in types      # glyph overlay removed
        assert "lines" in types             # the reference crosshair

    def test_each_component_builds(self):
        from spyde.actions.strain_display import build_strain_figure
        for comp in ("exx", "eyy", "exy", "omega"):
            fig, fid, html, p = build_strain_figure(self._field(), component=comp)
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

    def test_strict_mode_double_mount_builds_only_one_controller(self):
        """React StrictMode mounts the wizard TWICE synchronously (mount →
        cleanup → remount) on every open, firing strain_run, strain_stop,
        strain_run right in a row — before either strain_run's worker thread
        has finished (see strain_run's _strain_run_gen comment). Both calls
        must NOT end up building a live StrainController: the generation
        counter should let only the LATEST strain_run's window survive."""
        import threading
        import spyde.backend.ipc as ipc
        from spyde.actions.strain_action import strain_run, strain_stop, _CONTROLLERS

        vecs = _MockVecsCM((6, 6), lambda iy, ix: np.array([[1 + 0.01 * ix, 0.0],
                                                            [0.0, 1.0]]))
        tree = type("T", (), {"diffraction_vectors": vecs})()
        plot = type("P", (), {"signal_tree": tree})()

        class _Session:
            _w = 0
            signal_trees: list = []
            def next_window_id(self):
                self._w += 1
                return self._w
            def _dispatch_to_main(self, fn):
                fn()      # inline, like the real Session with a registered loop

        session = _Session()

        cap, orig = [], ipc.emit
        ipc.emit = lambda m: cap.append(m)
        try:
            # The exact StrictMode sequence: run, stop, run — all synchronous,
            # before any of strain_run's background "strain-run" threads land.
            strain_run(session, plot, {})
            strain_stop(session, plot, {})
            strain_run(session, plot, {})
            # Let both strain_run compute threads finish and dispatch back.
            for t in threading.enumerate():
                if t.name == "strain-run":
                    t.join(timeout=5.0)
        finally:
            ipc.emit = orig

        fig_msgs = [m for m in cap if m.get("type") == "figure"]
        # Exactly one strain-map window's figure must have been emitted and
        # left registered — not two.
        assert len(fig_msgs) == 1, f"expected 1 strain figure, got {len(fig_msgs)}"
        ctrl = getattr(tree, "_strain_controller", None)
        assert ctrl is not None
        assert len(_CONTROLLERS) == 1
        assert _CONTROLLERS[ctrl.window_id] is ctrl

    def test_controller_cif_reference_then_region(self):
        import anyplotlib as apl
        from orix.crystal_map import Phase
        from diffpy.structure import Atom, Lattice, Structure
        from spyde.actions.strain_action import StrainController

        st = Structure(atoms=[Atom("Al", [0, 0, 0])],
                       lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90))
        phase = Phase(name="Al", space_group=225, structure=st)
        g111 = float(np.sqrt(3) / 4.05)
        dirs = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], float) * g111

        class _V:
            nav_shape = (4, 4)
            def kxy_at(self, iy, ix):
                return dirs * (1.0 + 0.004 * ix)        # mild strain gradient
            def count_map(self):
                return np.ones((4, 4), int)

        fig, ax = apl.subplots()
        p = ax.imshow(np.zeros((4, 4), "f4"))
        ctrl = StrainController(_V(), p, ref_yx=(0, 0))
        ctrl.attach()
        ctrl.set_cif_reference(phase)                   # absolute CIF reference
        assert ctrl.cif_mode is True
        assert ctrl.field is not None and ctrl.field.nav_shape == (4, 4)
        ctrl.set_reference(2, 2)                         # crosshair → region mode
        assert ctrl.cif_mode is False

    def test_reference_excludes_zero_beam(self):
        # The reference uses every spot EXCEPT the central/direct (zero) beam.
        import anyplotlib as apl
        from spyde.actions.strain_action import StrainController

        # G_REF (4 ring spots) + a zero-beam point at the centre.
        with_zero = np.vstack([G_REF, [[0.0, 0.0]]])

        class _V:
            nav_shape = (4, 4)
            def kxy_at(self, iy, ix):
                return with_zero
            def count_map(self):
                return np.ones((4, 4), int)

        fig, ax = apl.subplots()
        p = ax.imshow(np.zeros((4, 4), "f4"))
        ctrl = StrainController(_V(), p, ref_yx=(0, 0))
        ctrl.attach()
        ref = ctrl._selected_reference()
        # The zero beam was dropped; all kept spots have |g| > 0.
        assert len(ref) == len(G_REF)
        assert np.all(np.linalg.norm(ref, axis=1) > 0)

    def test_reference_selector_uses_distinct_color_and_suppresses_toolbar(self):
        """The dedicated reference crosshair must be visually distinguishable
        from the main navigator crosshair (default green) — a different color
        — and its linked DP window must have NO toolbar (it exists solely to
        host the crosshair + the click-to-select overlay, not to run actions)."""
        import anyplotlib as apl
        import spyde.backend.ipc as ipc
        from spyde.actions.strain_action import StrainController

        class _FakeSelector:
            def __init__(self, child):
                self.active_children = [child]
                self.index_hooks = []
                class _W:
                    cx = cy = 0.0
                self._widget = _W()
            def update_data(self):
                pass

        class _FakeWindow:
            def __init__(self, plot):
                self.current_plot_item = plot
                self.window_id = plot.window_id

        class _FakeNPM:
            """Records the color passed to add_navigation_selector_and_signal_plot,
            mirroring MultiplotManager's contract without the full plot stack."""
            def __init__(self, ref_plot):
                self.plot_windows = {object(): {}}
                self.navigation_selectors = {}
                self.seen_color = None
                self._ref_plot = ref_plot

            def add_navigation_selector_and_signal_plot(self, nav_window, color=None):
                self.seen_color = color
                sel = _FakeSelector(self._ref_plot)
                self.navigation_selectors[nav_window] = [sel]
                return _FakeWindow(self._ref_plot)

        fig, ax = apl.subplots()
        ref_p = ax.imshow(np.zeros((4, 4), "f4"))
        ref_p.window_id = 999

        vecs = _MockVecsCM((4, 4), lambda iy, ix: np.array([[1.0, 0.0], [0.0, 1.0]]))
        npm = _FakeNPM(ref_p)
        tree = type("T", (), {"navigator_plot_manager": npm, "signal_plots": []})()

        cap, orig = [], ipc.emit
        ipc.emit = lambda m: cap.append(m)
        try:
            ctrl = StrainController(vecs, ax.imshow(np.zeros((4, 4), "f4")),
                                    ref_yx=(0, 0), src_tree=tree)
            ctrl._attach_reference_selector()
        finally:
            ipc.emit = orig

        assert npm.seen_color == StrainController._REF_CROSSHAIR_COLOR
        assert npm.seen_color != "green"     # distinct from the main nav crosshair

        toolbar_msgs = [m for m in cap if m.get("type") == "toolbar_config"
                        and m.get("window_id") == 999]
        assert toolbar_msgs and toolbar_msgs[-1]["toolbar_actions"] == []

    def test_commit_makes_new_signal_tree(self, window):
        # Submit freezes the live field as a new SignalTree (εxx signal plot +
        # εyy/εxy/ω view figures).
        import anyplotlib as apl
        from spyde.actions.strain_action import StrainController, strain_commit
        session = window["window"]
        n_before = len(session.signal_trees)

        vecs = _MockVecsCM((4, 4), lambda iy, ix: np.array([[1 + 0.01 * ix, 0.0],
                                                            [0.0, 1.0]]))
        from spyde.actions.strain_mapping import compute_strain_field
        fig, ax = apl.subplots()
        p = ax.imshow(np.zeros((4, 4), "f4"))
        ctrl = StrainController(vecs, p, window_id=4242, ref_yx=(0, 0),
                                session=session)
        ctrl.field = compute_strain_field(vecs, (0, 0))   # pre-set like strain_run
        ctrl.attach()                                     # registers in _CONTROLLERS (skips recompute)
        assert ctrl.field is not None
        # Dispatch by window_id (the strain window is not a registered Plot, so
        # plot is None — the handler resolves the controller from the registry).
        strain_commit(session, None, {"window_id": 4242})
        assert len(session.signal_trees) == n_before + 1
        new_tree = session.signal_trees[-1]
        assert "Strain" in new_tree.root.metadata.get_item("General.title", "")


class _Axis:
    def __init__(self, scale=1.0, offset=0.0, size=64):
        self.scale, self.offset, self.size = scale, offset, size


class _Group:
    """Records the last marker push for one group."""
    def __init__(self):
        self.kw = {}
    def set(self, **kw):
        self.kw.update(kw)
    def remove(self):
        pass


class _Plot2D:
    """Minimal anyplotlib Plot2D stand-in: records circle/arrow groups + the
    click handler so a test can fire a synthetic click."""
    def __init__(self):
        self.groups = {}
        self.handler = None
        self.handler_events: tuple = ()
    def add_circles(self, offsets, name=None, **kw):
        g = _Group(); g.kw["offsets"] = offsets; self.groups[name] = g; return g
    def add_arrows(self, offsets, U, V, name=None, **kw):
        g = _Group(); g.kw.update(offsets=offsets, U=U, V=V); self.groups[name] = g; return g
    def add_event_handler(self, fn, *events):
        self.handler = fn
        self.handler_events = events


class _OverlayVecs:
    """vecs with sig_axes + kxy_at_nav for the selection overlay."""
    def __init__(self, ref_spots, frame_peaks, *, scale=1.0, offset=0.0):
        self.sig_axes = [_Axis(scale=scale, offset=offset),
                         _Axis(scale=scale, offset=offset)]
        self.kernel_radius_px = 4.0
        self.nav_shape = (3, 3)
        self._ref = np.asarray(ref_spots, float)
        self._frame = np.asarray(frame_peaks, float)
    def kxy_at_nav(self, iy, ix, lead=()):
        return self._ref if (iy, ix) == (0, 0) else self._frame


class _Evt:
    """A real anyplotlib ``double_click`` Event only ever carries xdata/ydata
    (the calibrated data-space coordinate) — there is NO img_x/img_y field on
    the Python Event dataclass (see callbacks.py). _on_click hit-tests
    directly against self.ref_spots (already calibrated kx,ky), so tests pass
    the CALIBRATED click position here, not a pixel position."""
    def __init__(self, xdata, ydata):
        self.xdata, self.ydata = xdata, ydata


class TestStrainSelectionOverlay:
    """The interactive reference-spot selection + displacement overlay
    (green/grey circles on the reference pixel; arrows off it)."""

    def _overlay(self, on_toggle=None, *, scale=1.0, offset=0.0):
        from spyde.actions.vector_overlay import StrainSelectionOverlay
        ref = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [-5.0, 0.0]])  # incl. zero beam
        frame = ref + np.array([0.4, -0.3])                               # shifted peaks
        vecs = _OverlayVecs(ref, frame, scale=scale, offset=offset)
        dp = type("DP", (), {"_plot2d": _Plot2D()})()
        ov = StrainSelectionOverlay(dp, vecs, ref_yx=(0, 0),
                                    ref_spots=ref[1:],   # zero beam already excluded by ctrl
                                    match_radius_px=3.0, on_toggle=on_toggle)
        # attach without a tree → no navigator selectors, but groups + handler wire up.
        ov.attach(tree=type("T", (), {"navigator_plot_manager": None})())
        return ov, dp._plot2d

    def test_reference_pixel_shows_selected_circles(self):
        ov, p2d = self._overlay()
        # On the reference pixel all 3 spots are selected (green), none excluded.
        assert len(p2d.groups["strain_selected"].kw["offsets"]) == 3
        assert len(p2d.groups["strain_excluded"].kw["offsets"]) == 0

    def test_click_handler_listens_on_double_click(self):
        # A single click is ambiguous with panning on an anyplotlib 2-D panel,
        # so pick/toggle interactions use "double_click" — same as the
        # anyplotlib Particle Picker example (add_event_handler(fn, "double_click")).
        ov, p2d = self._overlay()
        assert "double_click" in p2d.handler_events

    def test_click_toggles_selection_and_fires_callback(self):
        hits = []
        ov, p2d = self._overlay(on_toggle=lambda: hits.append(1))
        # Double-click at the CALIBRATED position of the spot at (5,0) → toggle it OFF.
        p2d.handler(_Evt(5.0, 0.0))
        assert hits == [1]
        assert ov.selected.sum() == 2          # one dropped
        assert len(p2d.groups["strain_selected"].kw["offsets"]) == 2
        assert len(p2d.groups["strain_excluded"].kw["offsets"]) == 1
        assert len(ov.selected_reference()) == 2
        # Click it again → back ON.
        p2d.handler(_Evt(5.0, 0.0))
        assert ov.selected.sum() == 3

    def test_click_hit_test_is_scale_independent(self):
        # A trivial scale=1/offset=0 axis makes calibrated units == pixel
        # units by coincidence — the real bug this guards against (comparing
        # a calibrated click position against pixel-space markers) was
        # invisible under exactly that coincidence. Use a realistic
        # calibration (e.g. 0.01 Å⁻¹/px, offset -0.64, like a real diffraction
        # pattern's kx/ky axes) so the hit-test must genuinely work in
        # calibrated space, not just line up by luck.
        hits = []
        ov, p2d = self._overlay(on_toggle=lambda: hits.append(1),
                                scale=0.01, offset=-0.64)
        # The spot at calibrated (5.0, 0.0) — click exactly there, NOT at its
        # (very different) pixel position.
        p2d.handler(_Evt(5.0, 0.0))
        assert hits == [1]
        assert ov.selected.sum() == 2

    def test_off_reference_draws_displacement_arrows(self):
        ov, p2d = self._overlay()
        ov._last_iyix = (1, 1)                 # move off the reference pixel
        ov._redraw()
        arrows = p2d.groups["strain_displacement"].kw
        # 3 selected spots each match a shifted frame peak within radius → 3 arrows.
        assert len(arrows["offsets"]) == 3
        assert np.allclose(arrows["U"], 0.4, atol=1e-5)
        assert np.allclose(arrows["V"], -0.3, atol=1e-5)
        # And the circle groups are cleared off-reference.
        assert len(p2d.groups["strain_selected"].kw["offsets"]) == 0

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
