"""
test_strain_scan_frame.py — a strain map's x is the SCAN's x, and its colours
mean the same thing after Commit as they did in the live window.

Three things a reader of a strain map takes for granted that nothing used to
guarantee:

* "εxx" is strain along the scan's x. The fit works in the detector's frame,
  and the detector sits at some angle (sometimes with the opposite handedness)
  to the scan; the tensor has to be turned, the way DPC turns its beam shifts.
* Zero is white. A diverging map is meaningless if the range is not centred on
  zero, and dragging one contrast handle used to drift it.
* The diverging map survives Commit and every node switch, and each component
  gets a range of its own — εxx at ±0.5 % beside ω at ±3° on one shared scale
  was a flat εxx.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.actions._common import symmetric_range
from spyde.actions.strain_mapping import (
    StrainField, compute_strain_field, rotate_strain_basis,
)
from spyde.tests.migrated.test_strain_mapping import _MockVecs


def _rotation(deg):
    theta = np.deg2rad(deg)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def _field_for(T_of, ny=3, nx=4):
    return compute_strain_field(_MockVecs((ny, nx), T_of), (0, 0), tol=0.3)


class TestRotateStrainBasis:
    @pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, -45.0, 137.0])
    def test_matches_refitting_in_the_turned_frame(self, angle):
        # The ground truth: fit the SAME lattice seen by a detector turned by
        # `angle` — its deformation is R T Rᵀ — and compare with turning the
        # original fit. The two must agree component for component.
        R = _rotation(angle)
        T_of = lambda iy, ix: np.array([[1.0 + 0.01 * ix, 0.003 * iy],
                                        [0.003 * iy, 1.0 - 0.004 * ix]])
        turned = _field_for(lambda iy, ix: R @ T_of(iy, ix) @ R.T)
        rotated = rotate_strain_basis(_field_for(T_of), angle)
        for name in ("exx", "eyy", "exy", "omega"):
            np.testing.assert_allclose(getattr(rotated, name), getattr(turned, name),
                                       atol=2e-5, err_msg=name)

    def test_ninety_degrees_swaps_the_normal_components(self):
        field = StrainField(exx=np.full((2, 2), 0.02), eyy=np.zeros((2, 2)),
                            exy=np.zeros((2, 2)), omega=np.zeros((2, 2)),
                            coverage=np.ones((2, 2)))
        rotated = rotate_strain_basis(field, 90.0)
        assert np.allclose(rotated.exx, 0.0, atol=1e-12)
        assert np.allclose(rotated.eyy, 0.02)

    def test_a_flip_swaps_the_axes_and_reverses_the_rotation_sense(self):
        field = StrainField(exx=np.full((2, 2), 0.02), eyy=np.full((2, 2), -0.01),
                            exy=np.full((2, 2), 0.005), omega=np.full((2, 2), 0.3),
                            coverage=np.ones((2, 2)))
        flipped = rotate_strain_basis(field, 0.0, flip=True)
        assert np.allclose(flipped.exx, -0.01) and np.allclose(flipped.eyy, 0.02)
        assert np.allclose(flipped.exy, 0.005)
        assert np.allclose(flipped.omega, -0.3)

    def test_the_fit_itself_is_never_touched(self):
        field = _field_for(lambda iy, ix: np.array([[1.0 + 0.01 * ix, 0.0], [0.0, 1.0]]))
        before = field.exx.copy()
        rotate_strain_basis(field, 45.0, flip=True)
        assert np.array_equal(field.exx, before)


class TestControllerFrame:
    def _controller(self, session=None, **kw):
        import anyplotlib as apl
        from spyde.actions.strain_action import StrainController
        fig, ax = apl.subplots()
        p = ax.imshow(np.zeros((3, 4), "f4"))
        ctrl = StrainController(None, p, window_id=4343, ref_yx=(0, 0),
                                session=session, **kw)
        ctrl.set_raw_field(_field_for(
            lambda iy, ix: np.array([[1.0 + 0.01 * ix, 0.0], [0.0, 1.0]])))
        return ctrl

    def test_the_window_shows_the_scan_frame(self):
        ctrl = self._controller(rotation=90.0)
        assert np.allclose(ctrl.field.exx, 0.0, atol=1e-6)
        assert np.allclose(ctrl.field.eyy, ctrl.raw_field.exx, atol=1e-6)

    def test_changing_the_angle_re_expresses_without_a_refit(self):
        ctrl = self._controller()
        raw = ctrl.raw_field
        ctrl.set_rotation(90.0)
        assert ctrl.raw_field is raw
        assert np.allclose(ctrl.field.eyy, raw.exx, atol=1e-6)
        ctrl.set_rotation(flip=True)
        assert np.allclose(ctrl.field.exx, raw.exx, atol=1e-6)

    def test_from_dpc_takes_the_run_that_solved_the_scan(self, window):
        session = window["window"]
        dpc = type("R", (), {"rotation": 25.0, "flip": True})()
        tree = type("T", (), {"dpc_result": dpc})()
        session.signal_trees.append(tree)
        try:
            ctrl = self._controller(session=session)
            assert ctrl.rotation_from_dpc() == (25.0, True)
        finally:
            session.signal_trees.remove(tree)

    def test_nothing_to_take_from(self, window):
        ctrl = self._controller(session=window["window"])
        assert ctrl.rotation_from_dpc() is None

    def test_commit_records_the_frame(self, window):
        ctrl = self._controller(session=window["window"], rotation=30.0, flip=True)
        tree = ctrl.commit()
        params = tree._commit_provenance["params"]
        assert params["rotation"] == 30.0 and params["flip"] is True


class TestSymmetricRange:
    def test_the_moved_handle_sets_the_magnitude(self):
        assert symmetric_range((-2.0, 2.0), -1.0, 2.0) == (-1.0, 1.0)
        assert symmetric_range((-2.0, 2.0), -2.0, 3.0) == (-3.0, 3.0)

    def test_without_history_the_larger_wins(self):
        assert symmetric_range(None, -1.0, 4.0) == (-4.0, 4.0)

    def test_never_degenerate(self):
        lo, hi = symmetric_range(None, 0.0, 0.0)
        assert hi > lo


class TestCommittedContrast:
    def _commit(self, session):
        from spyde.actions.commit import commit_result_tree
        exx = np.full((4, 5), 0.5, np.float32)
        exx[0, 0] = -0.5
        omega = np.full((4, 5), 3.0, np.float32)
        omega[0, 0] = -3.0
        return commit_result_tree(
            session, title="Strain", primary=exx, primary_label="εxx",
            views=[("ω", omega)], cmap="coolwarm")

    def test_each_view_has_its_own_zero_centred_range(self, window):
        from spyde.actions import views
        tree = self._commit(window["window"])
        levels = views._VIEW_DATA[tree.signal_plots[0].window_id]["levels"]
        assert levels["εxx"][0] == -levels["εxx"][1]
        assert levels["ω"][0] == -levels["ω"][1]
        assert levels["εxx"][1] < levels["ω"][1], "ω's scale flattened εxx"

    def test_the_diverging_map_is_on_the_plot_and_in_every_node(self, window):
        tree = self._commit(window["window"])
        plot = tree.signal_plots[0]
        assert plot._colormap_name == "coolwarm"
        for sig in tree.signals():
            assert sig.metadata.get_item("Spyde.display.colormap") == "coolwarm"
            assert sig.metadata.get_item("Spyde.display.symmetric") is True

    def test_dragging_one_handle_keeps_zero_at_the_middle(self, window):
        tree = self._commit(window["window"])
        plot = tree.signal_plots[0]
        plot.set_clim(-0.5, 0.2)          # the upper handle moved
        assert plot._last_levels == (-0.2, 0.2)
        plot.set_clim(-0.1, 0.2)          # then the lower one
        assert plot._last_levels == (-0.1, 0.1)

    def test_auto_and_reset_stay_symmetric(self, window):
        tree = self._commit(window["window"])
        plot = tree.signal_plots[0]
        plot.auto_clim("full")
        assert plot._last_levels == (-0.5, 0.5)
        plot.auto_clim("robust")
        lo, hi = plot._last_levels
        assert lo == -hi

    def test_the_histogram_tells_the_dock(self, window):
        tree = self._commit(window["window"])
        plot = tree.signal_plots[0]
        window["messages"].clear()
        plot.auto_clim("robust")
        hist = [m for m in window["messages"] if m.get("type") == "histogram"][-1]
        assert hist["symmetric"] is True and hist["colormap"] == "coolwarm"

    def test_switching_to_a_child_node_keeps_the_map(self, window):
        from spyde.actions.lifecycle import show_tree_node
        tree = self._commit(window["window"])
        plot = tree.signal_plots[0]
        plot.set_colormap("gray")            # as if the figure had drifted
        child = next(s for s in tree.signals() if s is not tree.root)
        show_tree_node(plot, tree, child)
        assert plot._colormap_name == "coolwarm"
        assert plot._symmetric_contrast() is True


class TestUnsignedCommitsAreUntouched:
    def test_a_plain_map_keeps_a_plain_range(self, window):
        from spyde.actions.commit import commit_result_tree
        tree = commit_result_tree(window["window"], title="Plain",
                                  primary=np.arange(20, dtype=np.float32).reshape(4, 5),
                                  levels=None)
        plot = tree.signal_plots[0]
        assert plot._symmetric_contrast() is False
        plot.set_clim(2.0, 9.0)
        assert plot._last_levels == (2.0, 9.0)


class TestScanAxesGlyph:
    """The x/y arrows on the reference pattern ARE the rotation control."""

    def _pattern_plot(self, size=64):
        import anyplotlib as apl
        fig, ax = apl.subplots()
        plot2d = ax.imshow(np.zeros((size, size), "f4"))
        return type("P", (), {"_plot2d": plot2d})()

    def test_at_zero_the_arrows_are_the_detector_axes(self):
        from spyde.actions.strain_axes_glyph import ScanAxesGlyph
        glyph = ScanAxesGlyph(self._pattern_plot())
        assert glyph.x_arrow.u > 0 and abs(glyph.x_arrow.v) < 1e-9
        assert abs(glyph.y_arrow.u) < 1e-9 and glyph.y_arrow.v > 0
        assert (glyph.x_arrow.x, glyph.x_arrow.y) == (32.0, 32.0)

    def test_the_arrows_follow_the_frame_the_tensor_uses(self):
        # rotate_strain_basis(90°) puts the detector's x strain into the scan's
        # y: so at 90° the scan's y arrow must point along the detector's x.
        from spyde.actions.strain_axes_glyph import ScanAxesGlyph
        glyph = ScanAxesGlyph(self._pattern_plot(), rotation=90.0)
        assert glyph.y_arrow.u > 0 and abs(glyph.y_arrow.v) < 1e-9
        assert abs(glyph.x_arrow.u) < 1e-9 and glyph.x_arrow.v < 0
        glyph.set_rotation(0.0, flip=True)
        assert abs(glyph.x_arrow.u) < 1e-9 and glyph.x_arrow.v > 0

    def test_dragging_an_arrow_head_turns_the_frame_and_re_pins_it(self):
        from spyde.actions.strain_axes_glyph import ScanAxesGlyph
        seen = []
        glyph = ScanAxesGlyph(self._pattern_plot(), on_change=seen.append)
        # The user drags the x head straight UP on screen (v negative) — and
        # drags the whole arrow off its origin while doing it.
        glyph.x_arrow.set(_notify=False, x=10.0, y=10.0, u=0.0, v=-20.0)
        glyph._on_x_drag()
        assert seen and seen[-1] == pytest.approx(90.0)
        assert (glyph.x_arrow.x, glyph.x_arrow.y) == (32.0, 32.0)   # re-pinned
        assert glyph.y_arrow.u > 0 and abs(glyph.y_arrow.v) < 1e-9     # y followed
        # Dragging the y head is the same control from the other end.
        glyph.y_arrow.set(_notify=False, u=0.0, v=20.0)
        glyph._on_y_drag()
        assert seen[-1] == pytest.approx(0.0)

    def test_round_trip_through_the_drag_maths(self):
        from spyde.actions.strain_axes_glyph import (
            rotation_from_direction, scan_axes_on_detector,
        )
        for flip in (False, True):
            for angle in (-150.0, -30.0, 0.0, 45.0, 120.0):
                (xu, xv), (yu, yv) = scan_axes_on_detector(angle, flip)
                assert rotation_from_direction(xu, xv, axis="x", flip=flip) == \
                    pytest.approx(angle)
                assert rotation_from_direction(yu, yv, axis="y", flip=flip) == \
                    pytest.approx(angle)

    def test_the_controller_owns_the_glyph_and_tells_the_caret(self, window):
        import de_shell.ipc as ipc
        from spyde.actions.strain_action import StrainController
        session = window["window"]
        dp = self._pattern_plot()
        ctrl = StrainController(None, self._pattern_plot()._plot2d, window_id=4444,
                                ref_yx=(0, 0), session=session, src_dp_plot=dp)
        ctrl.caret_window_id = 7
        ctrl.set_raw_field(_field_for(
            lambda iy, ix: np.array([[1.0 + 0.01 * ix, 0.0], [0.0, 1.0]])))
        ctrl._attach_axes_glyph()
        assert ctrl.axes_glyph is not None
        window["messages"].clear()
        ctrl.axes_glyph.x_arrow.set(_notify=False, u=0.0, v=-20.0)
        ctrl.axes_glyph._on_x_drag()
        assert ctrl.rotation == pytest.approx(90.0)
        assert np.allclose(ctrl.field.eyy, ctrl.raw_field.exx, atol=1e-6)
        basis = [m for m in window["messages"] if m.get("type") == "strain_rotation"]
        assert basis and basis[-1]["window_id"] == 7
        assert basis[-1]["rotation"] == pytest.approx(90.0)
        # The caret's slider moves the arrows too.
        ctrl.set_rotation(0.0)
        assert ctrl.axes_glyph.x_arrow.u > 0
        ctrl.remove()
        assert ctrl.axes_glyph is None


class TestDpcSourcePicker:
    def _controller(self, session):
        import anyplotlib as apl
        from spyde.actions.strain_action import StrainController
        fig, ax = apl.subplots()
        ctrl = StrainController(None, ax.imshow(np.zeros((3, 4), "f4")),
                                window_id=4545, ref_yx=(0, 0), session=session)
        ctrl.caret_window_id = 9
        return ctrl

    def test_lists_live_wizards_and_committed_trees_by_name(self, window):
        import hyperspy.api as hs
        session = window["window"]
        dpc = type("R", (), {"rotation": 25.0, "flip": True})()
        root = hs.signals.Signal2D(np.zeros((2, 2)))
        root.metadata.General.title = "Vacuum scan"
        tree = type("T", (), {"dpc_result": dpc, "root": root})()
        live = type("C", (), {"result": type("R", (), {"rotation": -12.0, "flip": False})(),
                              "tree": tree})()
        session.signal_trees.append(tree)
        session._window_controllers[999] = live
        try:
            ctrl = self._controller(session)
            labels = [label for label, _r, _f in ctrl.dpc_sources()]
            assert labels == ["Vacuum scan (live) — -12.0°", "Vacuum scan — 25.0°, flipped"]
            assert ctrl.rotation_from_dpc(1) == (25.0, True)
            assert ctrl.rotation_from_dpc(5) is None
            window["messages"].clear()
            ctrl.emit_basis(sources=True)
            msg = [m for m in window["messages"] if m.get("type") == "strain_rotation"][-1]
            assert [s["label"] for s in msg["dpc_sources"]] == labels
        finally:
            session.signal_trees.remove(tree)
            session._window_controllers.pop(999, None)
