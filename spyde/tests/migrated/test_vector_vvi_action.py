"""
Gating + wiring for the Vector Virtual Imaging toolbar action.

The action is requires_vectors-gated: it must be absent until a signal tree
has diffraction_vectors attached, and present after PlotState.rebuild_toolbars()
re-runs the filter. Also checks the action module's compute helper produces a
correct intensity-weighted image from a real vectors object.
"""
import types

import numpy as np
import pytest
import dask.array as da
import hyperspy.api as hs

from spyde.actions.find_vectors import _do_compute_vectors
from spyde.drawing.toolbars.plot_control_toolbar import get_toolbar_actions_for_plot


def _params():
    return {"sigma": 0.5, "kernel_radius": 3, "threshold": 0.3,
            "min_distance": 3, "subpixel": False}


def _make_vecs():
    ny, nx, ky, kx = 4, 4, 32, 32
    data = np.zeros((ny, nx, ky, kx), dtype=np.float32)
    data[:, :, 14:18, 14:18] = 100.0
    s = hs.signals.Signal2D(da.from_array(data, chunks=(2, 2, ky, kx)))
    for ax in s.axes_manager.signal_axes:
        ax.scale = 0.01
        ax.offset = -ky * 0.005
    v = _do_compute_vectors(s, _params(), None, None)
    if len(v.flat_buffer) == 0:
        pytest.skip("No vectors found")
    return v


class _FakePlot:
    def __init__(self, signal, tree, is_navigator=False):
        self.plot_state = types.SimpleNamespace(
            current_signal=signal, dimensions=2, plot=self,
        )
        self.signal_tree = tree
        self.is_navigator = is_navigator


class _FakeTree:
    diffraction_vectors = None


class _FakeTreeWithVectors:
    diffraction_vectors = object()      # any non-None value unlocks the gate


def _signal2d():
    """Plain 4D diffraction signal (raw data)."""
    s = hs.signals.Signal2D(np.zeros((4, 4, 8, 8), dtype=np.float32))
    s.set_signal_type("electron_diffraction")
    return s


def _vectors_image():
    """A vectors-result image (the type the gating keys on)."""
    s = hs.signals.Signal2D(np.zeros((4, 4, 8, 8), dtype=np.float32))
    s.set_signal_type("spyde_diffraction_vectors_image")
    return s


class TestVectorVVIGating:
    def test_absent_on_raw_diffraction(self):
        plot = _FakePlot(_signal2d(), _FakeTree())
        names = get_toolbar_actions_for_plot(plot.plot_state)[2]
        assert "Vector Virtual Imaging" not in names
        assert "Vector Orientation Mapping" not in names

    def test_present_on_vectors_image(self):
        plot = _FakePlot(_vectors_image(), _FakeTreeWithVectors())
        names = get_toolbar_actions_for_plot(plot.plot_state)[2]
        assert "Vector Virtual Imaging" in names
        assert "Vector Orientation Mapping" in names

    def test_hidden_until_vectors_attach(self):
        """requires_vectors: the vector actions stay hidden on the vectors-image
        window while the batch is still computing (diffraction_vectors=None) and
        appear when _finalize attaches them + re-sends the toolbar config."""
        tree = _FakeTree()
        plot = _FakePlot(_vectors_image(), tree)
        before = get_toolbar_actions_for_plot(plot.plot_state)[2]
        for name in ("Vector Virtual Imaging", "Vector Orientation Mapping",
                     "Strain Mapping"):
            assert name not in before, f"{name} must wait for the vectors attach"
        tree.diffraction_vectors = object()          # the batch finalizes
        after = get_toolbar_actions_for_plot(plot.plot_state)[2]
        for name in ("Vector Virtual Imaging", "Vector Orientation Mapping",
                     "Strain Mapping"):
            assert name in after, f"{name} should appear once vectors attach"

    def test_dense_actions_excluded_on_vectors_image(self):
        # the vectors-result image is a Diffraction2D subclass, but the dense
        # diffraction actions must NOT appear on it.
        plot = _FakePlot(_vectors_image(), _FakeTreeWithVectors())
        names = get_toolbar_actions_for_plot(plot.plot_state)[2]
        for dense in ("Virtual Imaging", "Orientation Mapping",
                      "Find Diffraction Vectors"):
            assert dense not in names, f"{dense} should be excluded"

    def test_dense_actions_present_on_raw_diffraction(self):
        plot = _FakePlot(_signal2d(), _FakeTree())
        names = get_toolbar_actions_for_plot(plot.plot_state)[2]
        for dense in ("Virtual Imaging", "Orientation Mapping",
                      "Find Diffraction Vectors"):
            assert dense in names, f"{dense} should be present on raw data"

    def test_other_actions_unaffected(self):
        plot = _FakePlot(_signal2d(), _FakeTree())
        names = get_toolbar_actions_for_plot(plot.plot_state)[2]
        assert "Reset" in names and "Zoom In" in names


class TestVectorVVICompute:
    def test_roi_image_intensity_weighted(self):
        vecs = _make_vecs()
        img = vecs.virtual_image_from_roi_gpu(
            0.0, 0.0, vecs.kernel_radius_data * 4, 0.0,
            t=None, intensity_weighted=True,
        )
        assert img.shape == vecs.nav_shape
        assert img.sum() > 0

    def test_rect_image(self):
        vecs = _make_vecs()
        # a rectangle covering the whole detector equals a big disk total
        full_rect = vecs.virtual_image_from_rect(-1, -1, 1, 1,
                                                 intensity_weighted=True)
        full_disk = vecs.virtual_image_from_roi(0, 0, 2.0, 0.0,
                                                intensity_weighted=True)
        assert full_rect.shape == vecs.nav_shape
        assert np.isclose(full_rect.sum(), full_disk.sum())
        # a tiny rectangle away from any spot picks up nothing
        empty = vecs.virtual_image_from_rect(0.5, 0.5, 0.51, 0.51)
        assert empty.sum() == 0

    def test_rect_handles_reversed_corners(self):
        vecs = _make_vecs()
        a = vecs.virtual_image_from_rect(-1, -1, 1, 1)
        b = vecs.virtual_image_from_rect(1, 1, -1, -1)   # corners swapped
        assert np.array_equal(a, b)


def _make_vecs_5d():
    """A 5-D (stack, y, x, ky, kx) vectors object."""
    nt, ny, nx, ky, kx = 2, 4, 4, 32, 32
    data = np.zeros((nt, ny, nx, ky, kx), dtype=np.float32)
    data[:, :, :, 14:18, 14:18] = 100.0
    s = hs.signals.Signal2D(da.from_array(data, chunks=(1, 2, 2, ky, kx)))
    for ax in s.axes_manager.signal_axes:
        ax.scale = 0.01
        ax.offset = -ky * 0.005
    v = _do_compute_vectors(s, _params(), None, None)
    if len(v.flat_buffer) == 0:
        pytest.skip("No vectors found")
    return v


class _DiskWidget:
    """Minimal anyplotlib-circle-ROI stand-in for the action's reduce()."""
    def __init__(self, cx, cy, r):
        self.cx, self.cy, self.r = cx, cy, r
        self._data = {"r": r}


class TestVectorVVI5D:
    """Regression: making the vector VI return a 3-D series for a 5-D stack broke
    the action (the RegionAction output is a single non-navigable plot, so a 3-D
    result can't display → 'breaks everything'). reduce() must return a 2-D image
    for the CURRENT stack slice and must not raise."""

    def _action_for(self, vecs, parked_stack=1):
        from spyde.actions.vector_virtual_imaging import VectorVirtualImageAction
        sig = _vectors_image()                       # the displayed vectors image
        tree = _FakeTree()
        tree.diffraction_vectors = vecs
        # root carries the stack index the source navigator is parked on.
        tree.root = types.SimpleNamespace(
            axes_manager=types.SimpleNamespace(indices=(parked_stack, 0, 0)))
        plot = _FakePlot(sig, tree)
        # signal_tree/signal/plot are read-only properties → drive them via ctx.
        ctx = types.SimpleNamespace(plot=plot, params={})
        act = VectorVirtualImageAction(ctx)
        return act, sig

    def test_reduce_returns_2d_current_slice_for_stack(self):
        vecs = _make_vecs_5d()
        assert vecs.n_time == 2
        act, sig = self._action_for(vecs)
        sel = types.SimpleNamespace(
            roi=_DiskWidget(cx=16, cy=16, r=8))         # pixel-space ROI on centre
        img = act.reduce(sig, sel, None, calculation="intensity")
        assert img is not None
        assert img.ndim == 2                            # NOT a 3-D series
        assert tuple(img.shape) == vecs.nav_shape       # (y, x)

    def test_current_t_reads_source_index(self):
        vecs = _make_vecs_5d()
        act, _ = self._action_for(vecs, parked_stack=1)
        assert act._current_t() == 1                    # the parked stack slice

    def test_current_t_none_for_4d(self):
        vecs = _make_vecs()                             # 4-D
        sig = _vectors_image()
        tree = _FakeTree(); tree.diffraction_vectors = vecs
        plot = _FakePlot(sig, tree)
        from spyde.actions.vector_virtual_imaging import VectorVirtualImageAction
        act = VectorVirtualImageAction(types.SimpleNamespace(plot=plot, params={}))
        assert act._current_t() is None


class _RectWidget:
    """Minimal anyplotlib-rectangle-ROI stand-in."""
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self._data = {"w": w, "h": h}


class TestVectorVVI5DSeries:
    """The 5-D vector VI spawns a NAVIGABLE 3-D result (navigator = stack/time,
    signal = the per-slice VI map). The detector ROI on the source DP recomputes
    the FULL (n_t, nav_y, nav_x) stack every move via _series_array, which
    replaces the result tree's root data. These pin the compute half (the
    tree-spawning half needs a real Session — covered by the GUI test)."""

    def _action_for(self, vecs, parked_stack=1):
        from spyde.actions.vector_virtual_imaging import VectorVirtualImageAction
        sig = _vectors_image()
        # The displayed vectors image carries the DP (signal-axis) calibration;
        # _roi_geom reads it to convert the pixel-space ROI → calibrated k. Match
        # the vecs calibration so a centred pixel ROI maps to k≈0 (where the
        # synthetic spots are), as in the real app.
        for ax in sig.axes_manager.signal_axes:
            ax.scale, ax.offset = 0.01, -32 * 0.005
        tree = _FakeTree()
        tree.diffraction_vectors = vecs
        tree.root = types.SimpleNamespace(
            axes_manager=types.SimpleNamespace(indices=(parked_stack, 0, 0)))
        plot = _FakePlot(sig, tree)
        ctx = types.SimpleNamespace(plot=plot, params={})
        return VectorVirtualImageAction(ctx), sig

    def test_series_array_returns_full_stack(self):
        vecs = _make_vecs_5d()
        assert vecs.n_time == 2
        act, sig = self._action_for(vecs)
        sel = types.SimpleNamespace(roi=_DiskWidget(cx=16, cy=16, r=8))
        stack = act._series_array(sig, sel, calculation="intensity")
        assert stack is not None
        assert stack.ndim == 3                          # (n_t, y, x)
        assert tuple(stack.shape) == (vecs.n_time, *vecs.nav_shape)
        assert stack.sum() > 0

    def test_series_slice_matches_reduce_for_each_t(self):
        """Each slice of the full series equals the single-slice reduce for that
        t — the navigable stack is consistent with scrubbing the source nav."""
        vecs = _make_vecs_5d()
        act, sig = self._action_for(vecs)
        sel = types.SimpleNamespace(roi=_DiskWidget(cx=16, cy=16, r=8))
        stack = act._series_array(sig, sel, calculation="intensity")
        for t in range(vecs.n_time):
            single = vecs.virtual_image_from_roi_gpu(
                0.0, 0.0, 8 * 0.01, 0.0, t=t, intensity_weighted=True)
            assert np.allclose(stack[t], single)

    def test_series_array_rectangle(self):
        vecs = _make_vecs_5d()
        act, sig = self._action_for(vecs)
        sel = types.SimpleNamespace(roi=_RectWidget(x=0, y=0, w=32, h=32))
        stack = act._series_array(sig, sel, calculation="intensity")
        assert stack is not None
        assert tuple(stack.shape) == (vecs.n_time, *vecs.nav_shape)
        assert stack.sum() > 0

    def test_reduce_to_5d_pushes_and_returns_none(self):
        """In 5-D mode reduce_to recomputes the stack, pushes it to the tree, and
        returns None (the tree's navigator owns the signal plot — no direct
        slice push)."""
        vecs = _make_vecs_5d()
        act, sig = self._action_for(vecs)
        pushed = {}

        class _Tree:
            signal_plots = []
        act.vi_tree = _Tree()
        act._push_stack_to_tree = lambda stack: pushed.update(shape=stack.shape)

        sel = types.SimpleNamespace(roi=_DiskWidget(cx=16, cy=16, r=8))
        out = act.reduce_to(sig, sel, None, None, calculation="intensity")
        assert out is None
        assert pushed["shape"] == (vecs.n_time, *vecs.nav_shape)

    def test_count_weighting_differs_from_intensity(self):
        vecs = _make_vecs_5d()
        act, sig = self._action_for(vecs)
        sel = types.SimpleNamespace(roi=_DiskWidget(cx=16, cy=16, r=8))
        inten = act._series_array(sig, sel, calculation="intensity")
        count = act._series_array(sig, sel, calculation="count")
        assert count.sum() > 0
        # intensity weighting (~100 per vector) ≫ unit counts
        assert inten.sum() > count.sum()


class TestVectorVVI5DSpawnTree:
    """End-to-end (real Session): on a 5-D vectors result, Add Vector Virtual
    Image spawns a NAVIGABLE 3-D result tree (nav_dim 1 = stack/time, signal =
    the VI map) and fills its root data from the detector ROI."""

    def _vectors_result_tree(self, session):
        """Build a 5-D vectors result tree wired like find_vectors does: a
        vectors-image root + diffraction_vectors attached, opened in the session
        so it has real navigator/signal plots."""
        import time
        vecs = _make_vecs_5d()
        nt, (ny, nx) = vecs.n_time, vecs.nav_shape
        ky = int(vecs.sig_axes[1].size); kx = int(vecs.sig_axes[0].size)
        root = hs.signals.Signal2D(
            np.zeros((nt, ny, nx, ky, kx), dtype=np.float32))
        root.set_signal_type("spyde_diffraction_vectors_image")
        for ax in root.axes_manager.signal_axes:
            ax.scale, ax.offset = 0.01, -32 * 0.005
        tree = session._add_signal(root)
        tree.diffraction_vectors = vecs
        time.sleep(0.6)
        return tree, vecs

    def test_add_vvi_spawns_navigable_3d_tree(self, window):
        from spyde.actions.context import ActionContext
        from spyde.actions.vector_virtual_imaging import add_vector_virtual_image
        session = window["window"]
        tree, vecs = self._vectors_result_tree(session)
        n_trees_before = len(session.signal_trees)

        sig_plot = next(p for p in tree.signal_plots)
        ctx = ActionContext(plot=sig_plot, params={}, action_name="Add Vector Virtual Image")
        add_vector_virtual_image(ctx, type="disk", calculation="intensity")
        import time; time.sleep(0.6)

        # A new signal tree was spawned for the VI result …
        assert len(session.signal_trees) == n_trees_before + 1
        vi_tree = session.signal_trees[-1]
        # … and it is a navigable 3-D signal: nav_dim 1 (stack/time), sig_dim 2.
        assert vi_tree.root.data.ndim == 3
        assert vi_tree.root.axes_manager.navigation_dimension == 1
        assert tuple(vi_tree.root.data.shape) == (vecs.n_time, *vecs.nav_shape)

    def test_roi_move_recomputes_full_stack(self, window):
        """Moving the detector ROI replaces the VI tree's root data with the
        freshly-computed full stack (both navigator + current slice refresh).
        Simulate a move by repositioning the ROI widget onto the spots and
        re-running the action's update path."""
        from spyde.actions.context import ActionContext
        from spyde.actions.vector_virtual_imaging import (
            add_vector_virtual_image, VectorVirtualImageAction,
        )
        session = window["window"]
        tree, vecs = self._vectors_result_tree(session)
        sig_plot = next(p for p in tree.signal_plots)
        ctx = ActionContext(plot=sig_plot, params={}, action_name="Add Vector Virtual Image")
        add_vector_virtual_image(ctx, type="disk", calculation="intensity")
        import time; time.sleep(0.6)

        vi_tree = session.signal_trees[-1]
        act = next(v["action"] for v in session._action_artifacts.values()
                   if isinstance(v.get("action"), VectorVirtualImageAction))

        # The default ROI lands tiny + at the corner → empty. "Move" it onto the
        # central spots (pixel-space, which _roi_geom converts via the DP scale).
        roi = act._selector.roi
        sx = float(act.signal.axes_manager.signal_axes[0].scale)
        ox = float(act.signal.axes_manager.signal_axes[0].offset)
        roi.cx = (0.0 - ox) / sx                 # k=0 (centre) in pixels
        roi.cy = (0.0 - ox) / sx
        roi.r = 0.08 / sx                        # 0.08 Å⁻¹ radius in pixels

        out = act.reduce_to(act.signal, act._selector, None, None,
                            calculation="intensity")
        assert out is None                       # tree navigator owns the slice
        time.sleep(0.3)
        data = np.asarray(vi_tree.root.data)
        assert data.shape == (vecs.n_time, *vecs.nav_shape)
        assert data.sum() > 0                    # the moved ROI now picks up spots
