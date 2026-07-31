"""
The TWO-WINDOW orientation result layout.

Window 1 (the orientation MAP window) carries the IPF-X / IPF-Y / IPF-Z
projections as chip views; window 2 (the IPF EXPLORER) is a bare-figure window
holding the four ``[2D|3D] × [Points|Heatmap]`` figures, driven by the map's
crosshair — which marks the picked orientation AND rotates both spheres to face
it.

Same plumbing for every producer (raw OM, vector OM, EBSD): they all go through
``ipf_view.attach_ipf_3d(..., session=…)``.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.tests.migrated.test_ipf_3d import _al_orientation_map


class _FakePlot:
    """Just enough of a Plot for the attach helpers: a window id, a view tag and
    a paintable surface."""

    def __init__(self, window_id=41):
        self.window_id = window_id
        self.needs_auto_level = False
        self.view_label = None
        self.view_kind = None
        self.data = None
        self.signal_tree = None

    def set_view_tag(self, label, kind="2d"):
        self.view_label, self.view_kind = label, kind

    def set_data(self, d):
        self.data = np.asarray(d)


class _FakeTree:
    def __init__(self, plot=None, title="Si — Orientation (IPF-Z)"):
        import hyperspy.api as hs
        self.signal_plots = [plot] if plot is not None else []
        self.root = hs.signals.Signal2D(np.zeros((4, 5), np.float32))
        self.root.metadata.General.title = title
        self.orientation_map = None


class _FakeSession:
    def __init__(self):
        self._next = 100
        self.controllers = {}
        self.forgotten = []

    def next_window_id(self):
        self._next += 1
        return self._next

    def register_window_controller(self, wid, ctrl):
        self.controllers[int(wid)] = ctrl

    def controller_by_window_id(self, wid):
        return self.controllers.get(wid)

    def _forget_window(self, wid):
        self.forgotten.append(wid)
        ctrl = self.controllers.pop(wid, None)
        if ctrl is not None:
            ctrl.close()


@pytest.fixture()
def emitted(monkeypatch):
    """Capture every `emit` the IPF modules make (they each import it lazily
    from spyde.backend.ipc, so patching the source module is enough)."""
    msgs: list[dict] = []
    from spyde.backend import ipc
    monkeypatch.setattr(ipc, "emit", lambda m: msgs.append(m))
    return msgs


class TestFaceCamera:
    def test_aims_the_camera_down_the_vector(self):
        # The turntable camera looks along -view_dir; face_camera(v) must give
        # angles whose camera direction IS v (round-trip through the same
        # spherical convention the renderer uses).
        from spyde.actions.ipf_window import face_camera

        for v in ([0, 0, 1], [1, 0, 0], [0, 1, 0],
                  list(np.array([1.0, 1.0, 1.0]) / np.sqrt(3))):
            v = np.asarray(v, float)
            az, el = face_camera(v)
            a, e = np.radians(az), np.radians(el)
            back = np.array([np.cos(e) * np.sin(a), -np.cos(e) * np.cos(a),
                             np.sin(e)])
            assert np.allclose(back, v, atol=1e-6), (v, az, el)

    def test_returns_degrees_in_range(self):
        from spyde.actions.ipf_window import face_camera
        az, el = face_camera([0.0, 0.0, 1.0])
        assert -180.0 <= az <= 180.0 and -90.0 <= el <= 90.0


class TestIpfFigureBuilders:
    def test_points_2d_figure_has_a_marker_per_phase(self):
        from spyde.actions.ipf_window import build_ipf_points_2d_figure
        fig, fig_id, html, panels = build_ipf_points_2d_figure(
            _al_orientation_map(6, 7), "z")
        assert fig_id and "<" in html
        assert set(panels) == {0}
        assert "marker" in panels[0] and "xy" in panels[0]

    def test_points_2d_subsamples_a_huge_map(self):
        # The IPF is a DISTRIBUTION — a million-point EBSD map must not ship a
        # million scatter offsets to the renderer.
        from spyde.actions import ipf_window as iw
        om = _al_orientation_map(200, 200)                       # 40k positions
        _f, _i, html, _p = iw.build_ipf_points_2d_figure(om, "z", max_points=500)
        assert len(html) < 2_000_000

    def _grid(self, resolution=2.0):
        from spyde.actions.ipf_window import _density_sphere_grid
        om = _al_orientation_map(20, 25)
        grid = _density_sphere_grid(om, 0, "z", resolution=resolution,
                                    sigma=5.0, cmap="fire")
        assert grid is not None
        return grid

    def test_density_3d_grid_lies_on_the_unit_sphere(self):
        X, Y, Z, _rgba = self._grid()
        r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        assert np.allclose(r, 1.0, atol=1e-4)

    def test_density_3d_grid_is_a_surface_not_a_point_list(self):
        """The whole point of the texture path: the grid keeps its 2-D shape,
        so it IS a mesh and the raster lines up with it index for index."""
        X, Y, Z, rgba = self._grid()
        assert X.ndim == 2 and X.shape == Y.shape == Z.shape
        assert rgba.shape == X.shape + (4,)

    def test_every_vertex_is_finite(self):
        """A NaN vertex tears the mesh. Cells outside the projection's unit
        disk have no inverse, so they must be given a finite placeholder and
        masked in alpha instead of being left undefined."""
        X, Y, Z, _rgba = self._grid()
        assert np.isfinite(X).all() and np.isfinite(Y).all()
        assert np.isfinite(Z).all()

    def test_the_sector_is_masked_in_alpha(self):
        """The fundamental sector is not rectangular and the grid is, so the
        mask has to live in the texture rather than the geometry."""
        _X, _Y, _Z, rgba = self._grid()
        alpha = rgba[..., 3]
        assert (alpha == 255).any(), "nothing painted"
        assert (alpha == 0).any(), "nothing masked — the sector filled the grid"
        assert set(np.unique(alpha)) <= {0, 255}

    def test_painted_cells_are_coloured_by_density(self):
        _X, _Y, _Z, rgba = self._grid()
        lit = rgba[..., 3] > 0
        assert len(np.unique(rgba[lit][:, :3], axis=0)) > 1, \
            "the painted sector is one flat colour"

    def test_density_3d_figure_builds(self):
        from spyde.actions.ipf_window import build_ipf_density_3d_figure
        built = build_ipf_density_3d_figure(_al_orientation_map(20, 25), "z",
                                            resolution=2.0)
        assert built is not None
        _fig, fig_id, html, plots = built
        assert fig_id and "<" in html and set(plots) == {0}


class TestIpfExplorerWindow:
    def _open(self, emitted, direction="z"):
        from spyde.actions.ipf_window import open_ipf_window
        om = _al_orientation_map(6, 7)
        tree = _FakeTree(_FakePlot())
        sess = _FakeSession()
        ctrl = open_ipf_window(sess, tree, om, direction)
        return ctrl, om, tree, sess

    def test_opens_a_separate_window_with_all_four_views(self, emitted):
        ctrl, _om, tree, sess = self._open(emitted)
        assert ctrl is not None
        # A NEW window id — not the map window's (41).
        assert ctrl.window_id != 41
        figs = [m for m in emitted if m.get("type") == "figure"
                and m.get("window_id") == ctrl.window_id]
        assert {f["view"] for f in figs} == {"ipf2d", "density", "3d", "density3d"}
        # Registered for ✕-teardown + bare-window action dispatch, and cached.
        assert sess.controller_by_window_id(ctrl.window_id) is ctrl
        assert tree._ipf_window is ctrl

    def test_window_is_titled(self, emitted):
        # Every figure it emits carries a `view` tag, and the reducer refuses to
        # let a tagged view rename a window — so the title must be sent
        # explicitly or the window shows up nameless.
        ctrl, *_ = self._open(emitted)
        titles = [m for m in emitted if m.get("type") == "window_title"
                  and ctrl.window_id in (m.get("window_ids") or [])]
        assert titles and titles[-1]["title"] == "Si — IPF"

    def test_reopening_reuses_the_same_window(self, emitted):
        from spyde.actions.ipf_window import open_ipf_window
        ctrl, om, tree, sess = self._open(emitted)
        again = open_ipf_window(sess, tree, om, "z")
        assert again is ctrl                      # no duplicate window piles up
        assert len(sess.controllers) == 1

    def test_show_orientation_marks_and_rotates(self, emitted):
        from spyde.actions.ipf_window import face_camera
        ctrl, om, _tree, _s = self._open(emitted)
        assert ctrl.show_orientation(2, 3) is True

        v = om.ipf_xyz(2, 3, 0, "z")[0]
        az, el = face_camera(v)
        for handles in (ctrl._p3d_points, ctrl._p3d_density):
            for p3d in handles.values():
                hl = p3d._state.get("highlight")
                assert hl is not None
                assert np.allclose([hl["x"], hl["y"], hl["z"]], v, atol=1e-5)
                # THE new behaviour: the sphere rotates to face that direction.
                assert p3d._state["azimuth"] == pytest.approx(az, abs=1e-6)
                assert p3d._state["elevation"] == pytest.approx(el, abs=1e-6)

    def test_show_orientation_moves_the_2d_marker(self, emitted):
        ctrl, om, _tree, _s = self._open(emitted)
        marker = ctrl._panels_2d[0]["marker"]
        before = np.asarray(marker._data["offsets"], dtype=float)
        ctrl.show_orientation(1, 1)
        after = np.asarray(marker._data["offsets"], dtype=float)
        assert not np.array_equal(after, before)
        want = om.ipf_xy(1, 1, "z")[0][0]
        assert np.allclose(after[0], want, atol=1e-6)

    def test_show_orientation_does_not_re_emit_figures(self, emitted):
        # Picking must be a targeted push — re-emitting would reload the iframe
        # and throw away the camera the user orbited to.
        ctrl, *_ = self._open(emitted)
        n = len(emitted)
        ctrl.show_orientation(0, 0)
        assert len(emitted) == n

    def test_set_direction_recolours_every_view(self, emitted):
        ctrl, *_ = self._open(emitted)
        emitted.clear()
        ctrl.set_direction("x")
        assert ctrl.direction == "x"
        views = {m["view"] for m in emitted if m.get("type") == "figure"}
        assert views == {"ipf2d", "density", "3d", "density3d"}

    def test_close_is_idempotent_and_clears_the_tree(self, emitted):
        ctrl, _om, tree, _s = self._open(emitted)
        ctrl.close()
        ctrl.close()
        assert tree._ipf_window is None
        assert ctrl.show_orientation(0, 0) is False


class TestMapWindowProjections:
    def test_attach_projections_registers_three_chips(self, emitted):
        from spyde.actions import views
        from spyde.actions.ipf_view import attach_ipf_projections
        plot = _FakePlot(window_id=55)
        tree = _FakeTree(plot)
        assert attach_ipf_projections(tree, _al_orientation_map(4, 5), "z")
        # The painted map itself is one chip; the other two ride along.
        assert plot.view_label == "IPF-Z"
        labels = [m["view_label"] for m in emitted
                  if m.get("type") == "figure" and m.get("view_label")]
        assert set(labels) == {"IPF-X", "IPF-Y"}
        assert views._VIEW_DATA[55]["order"] == ["IPF-X", "IPF-Y", "IPF-Z"]

    def test_projection_maps_actually_differ(self, emitted):
        # A real bug this guards: emitting the SAME colour map three times
        # would look right in the chip strip and be useless.
        om = _al_orientation_map(6, 7)
        x, y, z = (om.ipf_color_map(d) for d in "xyz")
        assert not np.array_equal(x, y)
        assert not np.array_equal(y, z)

    def test_register_views_append_merges(self):
        from spyde.actions.views import register_views, _VIEW_DATA
        a = np.zeros((3, 3), np.float32)
        register_views(7, [("IPF-X", a), ("IPF-Z", a)])
        register_views(7, [("NCC", a)], append=True)
        assert _VIEW_DATA[7]["order"] == ["IPF-X", "IPF-Z", "NCC"]
        # Without append it still REPLACES (the re-run path).
        register_views(7, [("only", a)])
        assert _VIEW_DATA[7]["order"] == ["only"]


class TestDirectionDispatch:
    def test_bare_window_direction_reaches_the_controller(self, emitted):
        # The X/Y/Z buttons live on the EXPLORER window, which has no
        # registered Plot — dispatch hands the handler plot=None and the
        # injected payload["window_id"] is the only way back.
        from spyde.actions.ipf_view import ipf_set_direction
        from spyde.actions.ipf_window import open_ipf_window
        sess = _FakeSession()
        tree = _FakeTree(_FakePlot())
        ctrl = open_ipf_window(sess, tree, _al_orientation_map(4, 5), "z")
        ipf_set_direction(sess, None, {"direction": "y",
                                       "window_id": ctrl.window_id})
        assert ctrl.direction == "y"

    def test_map_window_direction_forwards_to_the_explorer(self, emitted):
        from spyde.actions.ipf_view import ipf_set_direction
        from spyde.actions.ipf_window import open_ipf_window
        sess = _FakeSession()
        plot = _FakePlot()
        tree = _FakeTree(plot)
        plot.signal_tree = tree
        om = _al_orientation_map(4, 5)
        tree.orientation_map = om
        ctrl = open_ipf_window(sess, tree, om, "z")
        ipf_set_direction(sess, plot, {"direction": "x"})
        assert ctrl.direction == "x"
        assert plot.data is not None and plot.data.shape == (4, 5, 3)


class TestSharedPlumbing:
    @pytest.mark.parametrize("caller", [
        "spyde.actions.orientation_action",       # raw OM (4D-STEM)
        "spyde.actions.vector_orientation_om",    # vector OM (4D-STEM)
        "spyde.actions.ebsd_action",              # EBSD
    ])
    def test_every_producer_passes_the_session(self, caller):
        # One plumbing point for all four combinations — if a producer forgets
        # `session=` it silently falls back to the old single-window layout.
        import importlib, inspect
        src = inspect.getsource(importlib.import_module(caller))
        assert "attach_ipf_3d(" in src
        for line in src.splitlines():
            if "attach_ipf_3d(" in line and "import" not in line and "def " not in line:
                assert "session=" in line, line

    def test_ebsd_finalize_builds_the_two_windows_and_keeps_its_chips(self, emitted):
        """EBSD's own finalize, driven directly.

        Its E2E (``ebsd_workflow.spec.ts``) needs ``kikuchipy`` to register the
        ``EBSD`` signal type for the toolbar gate, so it can't run in a dev env
        without that optional extra — this covers the same finalize function.
        It also pins the ``register_views(append=True)`` fix: EBSD registers its
        NCC / Similarity / ADP quality maps AFTER the IPF projections, and a
        plain (replacing) register_views silently dropped the projections from
        the ⌘-tile set.
        """
        from spyde.actions import ebsd_action, views

        class _EbsdTree(_FakeTree):
            def __init__(self, plot):
                super().__init__(plot, title="AlNi — Orientation (IPF-Z)")
                self.nodes = []

            def add_node(self, parent, child, label):
                self.nodes.append(label)

            def update_plot_states(self, child):
                pass

        class _EbsdSession(_FakeSession):
            def _reemit_signal_tree(self, tree):
                pass

        plot = _FakePlot(window_id=88)
        tree = _EbsdTree(plot)
        sess = _EbsdSession()
        om = _al_orientation_map(4, 5)
        q = np.ones((4, 5), np.float32)
        ebsd_action._finalize_ipf_window(sess, tree, om, score=q, osm=q, adp=q)

        # Window 2 opened, separate from the map window.
        ctrl = getattr(tree, "_ipf_window", None)
        assert ctrl is not None and ctrl.window_id != 88
        views_on_2 = {m["view"] for m in emitted if m.get("type") == "figure"
                      and m.get("window_id") == ctrl.window_id}
        assert views_on_2 == {"ipf2d", "density", "3d", "density3d"}
        # Window 1 kept BOTH chip families.
        order = views._VIEW_DATA[88]["order"]
        assert order == ["IPF-X", "IPF-Y", "IPF-Z", "NCC", "Similarity", "ADP"]

    def test_attach_without_a_session_keeps_the_legacy_layout(self, emitted):
        # Older callers / unit tests get everything on the one window.
        from spyde.actions.ipf_view import attach_ipf_3d
        plot = _FakePlot(window_id=63)
        tree = _FakeTree(plot)
        assert attach_ipf_3d(tree, _al_orientation_map(4, 5), "z") is True
        wins = {m["window_id"] for m in emitted if m.get("type") == "figure"}
        assert wins == {63}
        assert getattr(tree, "_ipf_window", None) is None
