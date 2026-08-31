"""
SpyDEDiffractionVectors 5-D (stacked) per-slice access.

A 5-D stack stores vectors with full_nav_shape = (n_stack, nav_y, nav_x); the
outer axis is the stack/time dimension. These exercise the per-slice helpers used
by the stack display: count_map_series, at_nav/kxy_at_nav (current slice only),
and the rect/disk virtual-image series.
"""
from __future__ import annotations

import numpy as np

from spyde.signals.diffraction_vectors import (
    SpyDEDiffractionVectors, N_COLS, COL_NAV_X, COL_NAV_Y, COL_KX, COL_KY,
    COL_TIME, COL_INTENSITY,
)
from spyde.tests.migrated.conftest import make_session


class _Ax:
    def __init__(self, size, scale=1.0, offset=0.0):
        self.size = size
        self.scale = scale
        self.offset = offset
        self.units = ""
        self.name = ""


def _make_5d():
    """stack=2, nav_y=2, nav_x=3 detector 8x8. One vector per (stack,y,x):
    intensity encodes the slice so we can tell slices apart; all peaks at kx=ky=4
    so an ROI at the centre catches them. Sorted outermost-first (stack,y,x)."""
    rows = []
    for st in range(2):
        for y in range(2):
            for x in range(3):
                r = np.zeros(N_COLS, dtype=np.float32)
                r[COL_NAV_X] = x
                r[COL_NAV_Y] = y
                r[COL_TIME] = st
                r[COL_KX] = 4.0
                r[COL_KY] = 4.0
                r[COL_INTENSITY] = 10.0 * (st + 1)   # slice 0 → 10, slice 1 → 20
                rows.append(r)
    flat = np.array(rows, dtype=np.float32)
    sig_axes = (_Ax(8), _Ax(8))
    return SpyDEDiffractionVectors.from_arrays(
        flat_buffer=flat, full_nav_shape=(2, 2, 3),
        sig_shape=(8, 8), sig_axes=sig_axes,
        kernel_radius_px=1.0, kernel_radius_data=1.0,
    )


class TestVectors5D:
    def test_n_time_and_shapes(self):
        v = _make_5d()
        assert v.n_time == 2
        assert v.nav_shape == (2, 3)
        assert v.full_nav_shape == (2, 2, 3)

    def test_count_map_series_is_3d_per_slice(self):
        v = _make_5d()
        series = v.count_map_series()
        assert series.shape == (2, 2, 3)          # (stack, y, x)
        # One vector per position in every slice.
        assert np.all(series == 1)
        # And it differs from the 2-D summed map (which sums over stack → 2).
        assert v.count_map().shape == (2, 3)
        assert np.all(v.count_map() == 2)

    def test_count_map_at_t_matches_series_and_no_crash(self):
        v = _make_5d()
        # Regression: count_map_at_t walked nav_offsets[0] (vector offsets) as row
        # indices → IndexError on real data. Now it indexes the series.
        for t in range(v.n_time):
            cm = v.count_map_at_t(t)
            assert cm.shape == (2, 3)
            np.testing.assert_array_equal(cm, v.count_map_series()[t])
        # Out-of-range t is clamped, not an IndexError.
        assert v.count_map_at_t(99).shape == (2, 3)

    def test_at_nav_picks_current_slice_only(self):
        v = _make_5d()
        # at() (no lead) returns BOTH slices' vectors at (y=1, x=2).
        both = v.at(1, 2)
        assert both.shape[0] == 2
        # at_nav with lead=(stack,) returns ONLY that slice's single vector.
        s0 = v.at_nav(1, 2, lead=(0,))
        s1 = v.at_nav(1, 2, lead=(1,))
        assert s0.shape[0] == 1 and s1.shape[0] == 1
        assert float(s0[0, COL_INTENSITY]) == 10.0   # slice 0
        assert float(s1[0, COL_INTENSITY]) == 20.0   # slice 1

    def test_kxy_at_nav_shape(self):
        v = _make_5d()
        kxy = v.kxy_at_nav(0, 0, lead=(1,))
        assert kxy.shape == (1, 2)
        np.testing.assert_allclose(kxy[0], [4.0, 4.0])

    def test_at_nav_stale_lead_falls_back(self):
        v = _make_5d()
        # Wrong-length lead (e.g. a stale 2-D position) must not raise.
        out = v.at_nav(0, 0, lead=(0, 0))   # 2 lead coords but only 1 outer dim
        assert out.shape[1] == N_COLS

    def test_virtual_image_series_disk_is_3d(self):
        v = _make_5d()
        series = v.virtual_image_series(cx=4.0, cy=4.0, r_outer=2.0, r_inner=0.0,
                                        intensity_weighted=True)
        assert series.shape == (2, 2, 3)             # (stack, y, x)
        # Slice 0 sums intensity 10 per position, slice 1 sums 20.
        assert np.allclose(series[0], 10.0)
        assert np.allclose(series[1], 20.0)

    def test_virtual_image_series_rect_is_3d(self):
        v = _make_5d()
        series = v.virtual_image_series_rect(x0=2.0, y0=2.0, x1=6.0, y1=6.0,
                                             intensity_weighted=False)
        assert series.shape == (2, 2, 3)
        # count weighting → 1 vector per position per slice.
        assert np.all(series == 1.0)


class TestFiveDResultDisplay:
    """The 5-D vectors RESULT WINDOW, which used to stop at slice 0.

    Two separate faults, both visible in the app on a real 5-D stack:

      * the 1-D TIME navigator was never painted — the paint loop skipped every
        nav plot whose shape wasn't the 2-D spatial grid — so it sat on an
        all-zero flat line, showing neither how many vectors a slice held nor
        which slice was current;
      * `count_map_at_t(0)` was painted once at attach and had exactly ONE call
        site in the whole app, so scrubbing time left the count map on slice 0
        while the DP and the vector overlay moved on.
    """

    class _Plot:
        """Minimal nav-plot stand-in: records what was painted onto it."""

        def __init__(self, shape):
            self.current_data = np.zeros(shape, dtype=np.float32)
            self.painted = []
            self.needs_auto_level = False

        def set_data(self, arr):
            self.painted.append(np.asarray(arr))
            self.current_data = np.asarray(arr)

    class _Sel:
        def __init__(self):
            self.index_hooks = []
            self.active_children = []
            self.current_indices = None

    def _tree(self, vecs):
        """A tree carrying both nav plots a 5-D stack opens, plus one selector."""
        from types import SimpleNamespace
        spatial = self._Plot(vecs.nav_shape)          # 2-D count map
        timeline = self._Plot((vecs.n_time,))         # 1-D stack navigator
        sel = self._Sel()
        npm = SimpleNamespace(
            plot_windows={"w": None}, plots={"w": [spatial, timeline]},
            all_navigation_selectors=[sel])
        tree = SimpleNamespace(navigator_plot_manager=npm, signal_plots=[object()])
        return tree, spatial, timeline, sel

    def test_the_time_navigator_gets_per_slice_totals(self):
        """The flat-zero line: the 1-D navigator must show vectors-per-slice."""
        from spyde.actions.find_vectors_action import _all_nav_plots
        v = _make_5d()
        tree, _spatial, timeline, _sel = self._tree(v)
        assert set(_all_nav_plots(tree)) >= {timeline}

        series = np.asarray(v.count_map_series(), dtype=np.float32)
        per_slice = series.reshape(series.shape[0], -1).sum(axis=1)
        assert per_slice.shape == (v.n_time,)
        assert per_slice.sum() == len(v.flat_buffer), \
            "per-slice totals must account for every vector"
        assert (per_slice > 0).all(), "a flat-zero time navigator is the bug"

    def test_moving_time_repaints_the_count_map(self):
        from spyde.actions.find_vectors_action import _attach_time_slice_repaint
        v = _make_5d()
        tree, spatial, _timeline, sel = self._tree(v)
        _attach_time_slice_repaint(tree, v)
        assert sel.index_hooks, "no hook registered on the navigator selector"

        # A 5-D index is (t, iy, ix); the leading coord is the slice.
        sel.index_hooks[0](np.array([[1, 0, 0]]))
        assert spatial.painted, "moving the time axis repainted nothing"
        np.testing.assert_array_equal(spatial.painted[-1], v.count_map_at_t(1))

    def test_a_repeat_of_the_same_slice_does_not_repaint(self):
        """The hook fires on every nav move, not just time changes."""
        from spyde.actions.find_vectors_action import _attach_time_slice_repaint
        v = _make_5d()
        tree, spatial, _t, sel = self._tree(v)
        _attach_time_slice_repaint(tree, v)
        sel.index_hooks[0](np.array([[1, 0, 0]]))
        n = len(spatial.painted)
        sel.index_hooks[0](np.array([[1, 2, 2]]))     # same t, different x/y
        assert len(spatial.painted) == n

    def test_reattaching_does_not_double_paint(self):
        """Re-running Find Vectors must not leave two hooks painting per move."""
        from spyde.actions.find_vectors_action import _attach_time_slice_repaint
        v = _make_5d()
        tree, spatial, _t, sel = self._tree(v)
        _attach_time_slice_repaint(tree, v)
        _attach_time_slice_repaint(tree, v)
        assert len(sel.index_hooks) == 1
        sel.index_hooks[0](np.array([[1, 0, 0]]))
        assert len(spatial.painted) == 1

    def test_a_4d_result_registers_no_hook(self):
        """4-D has no time axis — nothing to slice, nothing to subscribe to."""
        from spyde.actions.find_vectors_action import _attach_time_slice_repaint
        from types import SimpleNamespace
        flat = np.zeros((4, N_COLS), dtype=np.float32)
        flat[:, COL_KX] = 4.0
        flat[:, COL_KY] = 4.0
        v4 = SpyDEDiffractionVectors.from_arrays(
            flat_buffer=flat, full_nav_shape=(2, 2), sig_shape=(8, 8),
            sig_axes=(_Ax(8), _Ax(8)), kernel_radius_px=1.0,
            kernel_radius_data=1.0)
        assert v4.n_time == 0
        tree, _s, _t, sel = self._tree(v4)
        _attach_time_slice_repaint(tree, v4)
        assert sel.index_hooks == []


class TestFiveDResultWiringOnARealTree:
    """The 5-D result window built by the REAL MultiplotManager.

    ``TestFiveDResultDisplay`` above stands the nav plots up by hand, and that is
    exactly why it could not see these: both faults are in how the real
    three-level window (1-D stack → 2-D real space → DP) is WIRED.

      * ``MultiplotManager.add_navigation_selector_and_signal_plot`` filed the
        intermediate real-space NAVIGATOR in ``tree.signal_plots``. Find-Vectors
        then installed the DP's render-frame slice function on the stack
        selector, so the real-space window drew diffraction patterns — and every
        other consumer of ``signal_plots[0]`` ("the DP" to strain / IPF / EBSD /
        fit / commit) addressed the wrong window too.
      * plots were matched by ``current_data.shape``, which is None until the
        first async paint lands. The miss pushed a 2-D count map onto the 1-D
        stack navigator, whose ``set_data`` then raised — leaving it a flat zero
        line for the rest of the session.
    """

    def _session_with_5d(self):
        import hyperspy.api as hs
        session = make_session()
        data = np.zeros((2, 2, 3, 8, 8), dtype=np.float32)
        data[..., 4, 4] = 100.0
        s = hs.signals.Signal2D(data)
        s.metadata.General.title = "stack"
        tree = session._add_signal(s)
        return session, tree

    def test_the_intermediate_navigator_is_not_a_signal_plot(self):
        session, tree = self._session_with_5d()
        try:
            assert tree.root.axes_manager.navigation_dimension == 3
            navs = [p for p in tree.signal_plots
                    if getattr(p, "is_navigator", False)]
            assert navs == [], (
                "the real-space navigator was filed as a signal plot: "
                f"{[(id(p), getattr(p, 'is_navigator', None)) for p in tree.signal_plots]}")
            assert len(tree.signal_plots) == 1, \
                "a 5-D tree has exactly one signal plot (the DP)"
        finally:
            session.shutdown()

    def test_render_fn_goes_to_the_dp_and_the_count_map_to_the_navigator(self):
        from spyde.actions.find_vectors_action import (
            _install_render_display, _all_nav_plots,
        )
        session, tree = self._session_with_5d()
        try:
            v = _make_5d()
            _install_render_display(tree, v)
            dp = tree.signal_plots[0]
            spatial = [p for p in _all_nav_plots(tree)
                       if getattr(p, "is_navigator", False)
                       and getattr(p, "current_data", None) is not None
                       and np.asarray(p.current_data).shape == v.nav_shape]

            installed = {}
            for sel in tree.navigator_plot_manager.all_navigation_selectors:
                for child, fn in sel.children.items():
                    installed[id(child)] = getattr(fn, "__name__", str(fn))

            assert installed.get(id(dp)) == "_fn", (
                "the DP is not driven by the vectors renderer: "
                f"{installed.get(id(dp))}")
            for nav in spatial:
                assert installed.get(id(nav)) != "_fn", (
                    "the real-space navigator got the DP renderer — this is the "
                    "'real space shows a diffraction pattern' bug")
                assert installed.get(id(nav)) == "_count_fn", (
                    "the real-space navigator should slice the per-slice count "
                    f"map, got {installed.get(id(nav))}")
        finally:
            session.shutdown()

    def test_the_count_fn_returns_that_slices_count_map(self):
        from spyde.actions.find_vectors_action import _install_render_display
        session, tree = self._session_with_5d()
        try:
            v = _make_5d()
            _install_render_display(tree, v)
            count_fn = None
            for sel in tree.navigator_plot_manager.all_navigation_selectors:
                for fn in sel.children.values():
                    if getattr(fn, "__name__", "") == "_count_fn":
                        count_fn = fn
            assert count_fn is not None, "no count-map slice function installed"
            for t in range(v.n_time):
                out = count_fn(None, None, np.array([[t]]))
                np.testing.assert_array_equal(out, v.count_map_at_t(t))
        finally:
            session.shutdown()

    def test_an_unpainted_stack_navigator_is_still_found(self):
        """_display_shape must resolve before the first async paint lands —
        matching on current_data alone is what stranded the time navigator."""
        from spyde.actions.find_vectors_action import (
            _display_shape, _nav_plot_with_shape, _all_nav_plots,
        )
        session, tree = self._session_with_5d()
        try:
            for p in _all_nav_plots(tree):
                p.current_data = None
            assert _nav_plot_with_shape(tree, (2, 3)) is not None, \
                "the 2-D count-map navigator was not found before its first paint"
            shapes = {_display_shape(p) for p in _all_nav_plots(tree)}
            assert (2,) in shapes, f"the 1-D stack navigator is missing: {shapes}"
        finally:
            session.shutdown()

    def test_finalize_never_pushes_a_2d_map_onto_the_1d_navigator(self):
        from spyde.actions.find_vectors_action import _finalize, _all_nav_plots
        from spyde.actions.find_vectors_action import _display_shape
        session, tree = self._session_with_5d()
        try:
            v = _make_5d()
            painted: dict[int, list] = {}
            want: dict[int, tuple] = {}
            for p in _all_nav_plots(tree):
                p.current_data = None            # nothing has painted yet
                want[id(p)] = _display_shape(p) or ()
                real = p.set_data

                def _cap(arr, _p=p, _real=real):
                    painted.setdefault(id(_p), []).append(np.asarray(arr).shape)
                    return _real(arr)

                p.set_data = _cap
            _finalize(tree, v)

            assert painted, "_finalize painted no navigator at all"
            for pid, shapes in painted.items():
                for got in shapes:
                    assert len(got) == len(want[pid]), (
                        f"a {len(got)}-D array was painted onto a "
                        f"{len(want[pid])}-D navigator (shape {want[pid]}) — "
                        "the flat-zero-line bug")
        finally:
            session.shutdown()
