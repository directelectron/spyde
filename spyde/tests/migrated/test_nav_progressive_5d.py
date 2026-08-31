"""
The progressive navigator fill must run for 5-D datasets — and fill BOTH
navigators from ONE pass.

A 5-D stack (t, y, x | ky, kx) opens two navigators: a 1-D time line whose
selector drives a 2-D real-space image (MultiplotManager's
``navigation_depth == 2`` branch). Both are reductions of the SAME deep nav-sum
``deep[t, y, x] = data[t, y, x].sum()``:

    child (real space) = deep[t_selected]
    top   (time line)  = deep.sum(axis=(-2, -1))

The bug: ``_preprocess_navigator`` handed ``_compute_navigator`` the ALREADY
REDUCED 1-D time sum, so the progressive fill's "chunks" were whole time steps —
it read the entire dataset to paint one point of a short line, showed no visible
progress, and never filled the real-space navigator at all (that one waited on
the selector's own blocking whole-time-slice read).

The fix stashes the DEEP array and derives the top navigator from the real-space
planes the fill already computed — one pass, two navigators. These tests pin:
per-chunk progressive paints on BOTH plots, correct final values on both, the
deep array committed in RAM (so a later time move is a numpy slice), and no
regression of the 4-D single-navigator path.
"""
from __future__ import annotations

import os
import time

import numpy as np
import dask.array as da
import hyperspy.api as hs
import pytest
from spyde.tests.migrated._async import wait_until
from spyde.tests.migrated.conftest import make_session


def _wait(pred, timeout=20.0):
    return wait_until(pred, timeout)


def _make_session():
    return make_session()


def _stack_5d(nt=3, ny=8, nx=8, ky=4, kx=4, chunk_nav=4):
    """Lazy 5-D stack with a per-(t, y, x) signature so every reduction is
    checkable, chunked one time step × a nav block × the WHOLE frame."""
    rs = np.random.RandomState(0)
    base = rs.rand(nt, ny, nx, ky, kx).astype(np.float32) + 1.0
    for t in range(nt):
        base[t] *= (t + 1)          # strictly increasing time ramp
    arr = da.from_array(base, chunks=(1, chunk_nav, chunk_nav, ky, kx))
    s = hs.signals.Signal2D(arr).as_lazy()
    s.set_signal_type("electron_diffraction")
    return s, base


@pytest.fixture
def paint_tracker(monkeypatch):
    """Record every navigator ``Plot.set_data`` as ``(plot, array_copy)``."""
    from spyde.drawing.plots.plot import Plot
    paints: list[tuple] = []
    orig = Plot.set_data

    def _rec(self, data, *a, **k):
        try:
            if getattr(self, "is_navigator", False):
                paints.append((self, np.array(data, copy=True)))
        except Exception:
            pass
        return orig(self, data, *a, **k)

    monkeypatch.setattr(Plot, "set_data", _rec)
    return paints


class _StubPlot:
    """Minimal Plot stand-in for the paint-guard unit tests."""

    is_navigator = True

    def __init__(self):
        self.current_data = None
        self.painted: list[np.ndarray] = []
        self.histograms = 0

    def set_data(self, data, levels=None):
        self.current_data = np.asarray(data)
        self.painted.append(np.array(data, copy=True))

    def _emit_histogram(self, data, vmin, vmax):
        self.histograms += 1


class _StubSelector:
    def __init__(self, index=0):
        self.index = index

    def get_selected_indices(self):
        return np.array([[self.index]])


class TestProgressiveNavigator5D:
    def test_deep_array_is_what_the_fill_computes(self, monkeypatch):
        """The stash handed to the progressive compute must be the DEEP
        ``(t, y, x)`` nav-sum, not the reduced 1-D time sum — that is the whole
        bug (a 1-D stash makes each 'chunk' a whole-dataset read)."""
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        session = _make_session()
        try:
            seen = {}
            from spyde.signal_tree import BaseSignalTree
            orig = BaseSignalTree._start_progressive_nav_compute

            def _spy(self, nav_dask=None, deep=None):
                if nav_dask is not None:
                    seen["shape"] = tuple(nav_dask.shape)
                    seen["chunks"] = nav_dask.chunks
                    seen["deep"] = bool(deep)
                return orig(self, nav_dask, deep=deep)

            monkeypatch.setattr(
                BaseSignalTree, "_start_progressive_nav_compute", _spy)

            s, base = _stack_5d()
            session._add_signal(s, source_path=None)
            assert _wait(lambda: "shape" in seen), "the fill never started for 5-D"
            assert seen["deep"] is True
            assert seen["shape"] == base.shape[:3]
            # …and it has a real chunk grid to walk (3 t × 2 × 2 here), not one
            # whole-dataset block per point of the time line.
            assert int(np.prod([len(c) for c in seen["chunks"]])) > len(base)
        finally:
            session.shutdown()

    def test_both_navigators_fill_from_one_pass(self, monkeypatch, paint_tracker):
        """ONE pass over the deep array lands the correct image on BOTH
        navigators, and the top (time) one is painted PROGRESSIVELY — multiple
        paints with a growing finite-point count."""
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        session = _make_session()
        try:
            s, base = _stack_5d()
            session._add_signal(s, source_path=None)

            tree = session.signal_trees[-1]
            assert _wait(lambda: tree._deep_nav_targets() is not None)
            top, child, _sel = tree._deep_nav_targets()

            want_time = base.sum(axis=(1, 2, 3, 4))
            want_space = base[0].sum(axis=(2, 3))

            assert _wait(lambda: isinstance(top.current_data, np.ndarray)
                         and np.isfinite(top.current_data).all()), \
                "the 1-D time navigator never filled"
            assert _wait(lambda: isinstance(child.current_data, np.ndarray)
                         and np.isfinite(child.current_data).all()), \
                "the 2-D real-space navigator never filled"

            np.testing.assert_allclose(
                np.asarray(top.current_data).ravel(), want_time, rtol=1e-4)
            np.testing.assert_allclose(
                np.asarray(child.current_data), want_space, rtol=1e-4)

            counts = [int(np.isfinite(a).sum())
                      for p, a in paint_tracker if p is top]
            assert len(counts) >= 2, (
                f"time navigator painted {len(counts)}× — not progressive")
            assert max(counts) == len(want_time), (
                f"time navigator never fully filled "
                f"({max(counts)}/{len(want_time)} finite)")
            assert min(counts) < len(want_time), (
                "time navigator only ever painted a complete line — "
                "the fill was not progressive")
            # The child navigator's own progressive fill is exercised by
            # test_child_navigator_fills_progressively: on data this small its
            # selector's whole-time-slice read finishes first, so the fill
            # (correctly) declines to overwrite a complete frame with a partial
            # one. On a real stack that read takes minutes and the fill wins.
        finally:
            session.shutdown()

    def test_time_navigator_never_shows_a_partial_sum(self, monkeypatch,
                                                      paint_tracker):
        """A half-summed time point is a WRONG value, not a dim one — the
        derived line keeps it NaN until its real-space plane is complete, so
        every finite value EVER painted already equals the final one."""
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        session = _make_session()
        try:
            s, base = _stack_5d()
            session._add_signal(s, source_path=None)
            tree = session.signal_trees[-1]
            assert _wait(lambda: tree._deep_nav_targets() is not None)
            top, _child, _sel = tree._deep_nav_targets()
            want = base.sum(axis=(1, 2, 3, 4))
            assert _wait(lambda: isinstance(top.current_data, np.ndarray)
                         and np.isfinite(top.current_data).all())

            painted = [a.ravel() for p, a in paint_tracker if p is top]
            assert painted, "the time navigator was never painted"
            for arr in painted:
                finite = np.isfinite(arr)
                np.testing.assert_allclose(
                    arr[finite], want[finite], rtol=1e-4,
                    err_msg="the time navigator showed a PARTIAL sum "
                            f"({arr} vs final {want})")
        finally:
            session.shutdown()

    def test_child_navigator_fills_progressively(self):
        """The child navigator is repainted every time the fill's plane holds
        MORE finite pixels — the real-data case, where its own selector read
        (a whole time slice) has not landed yet."""
        from spyde.signal_tree import BaseSignalTree

        tree = BaseSignalTree.__new__(BaseSignalTree)
        top, child, sel = _StubPlot(), _StubPlot(), _StubSelector(0)
        acc = np.full((2, 2, 4), np.nan, dtype=np.float32)
        for col in range(4):
            acc[0, :, col] = float(col + 1)
            tree._paint_deep_nav(acc, (top, child, sel))
        assert [int(np.isfinite(a).sum()) for a in child.painted] == [2, 4, 6, 8]
        assert child.histograms == 0
        tree._paint_deep_nav(acc, (top, child, sel), final=True)
        assert child.histograms == 1

    def test_partial_plane_never_clobbers_a_complete_frame(self):
        """The guard that makes the above safe: once the child's own selector
        read has painted a COMPLETE frame, a half-filled plane from the fill is
        dropped rather than painted over it."""
        from spyde.signal_tree import BaseSignalTree

        tree = BaseSignalTree.__new__(BaseSignalTree)
        top, child, sel = _StubPlot(), _StubPlot(), _StubSelector(0)
        complete = np.arange(4, dtype=np.float32).reshape(2, 2)
        child.current_data = complete
        acc = np.full((2, 2, 2), np.nan, dtype=np.float32)
        acc[0, 0, 0] = 99.0
        tree._paint_deep_nav(acc, (top, child, sel))
        assert child.painted == []
        np.testing.assert_array_equal(child.current_data, complete)

    def test_reduce_deep_nav_masks_incomplete_planes(self):
        """Unit: the derived reduction sums complete planes and leaves an
        incomplete one NaN (rather than reporting a partial sum as a value)."""
        from spyde.signal_tree import BaseSignalTree
        acc = np.full((3, 2, 2), np.nan, dtype=np.float32)
        acc[0] = [[1.0, 2.0], [3.0, 4.0]]     # complete
        acc[1, 0, 0] = 5.0                    # partial
        out = BaseSignalTree._reduce_deep_nav(acc)
        assert out.shape == (3,)
        assert out[0] == pytest.approx(10.0)
        assert np.isnan(out[1]) and np.isnan(out[2])

    def test_deep_signal_is_committed_in_ram(self, monkeypatch):
        """When the fill finishes, the child navigator's own signal holds the
        completed array EAGERLY — so a later time-slider move is a numpy slice,
        not another whole-time-slice dask read. Leaving ``_lazy`` set with numpy
        data would route the read down the lazy branch, which needs dask."""
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        session = _make_session()
        try:
            s, base = _stack_5d()
            session._add_signal(s, source_path=None)
            tree = session.signal_trees[-1]
            nav_signals = tree.navigator_signals["base"]
            assert len(nav_signals) == 2

            assert _wait(lambda: isinstance(nav_signals[1].data, np.ndarray)), \
                "the deep navigator signal was never committed"
            assert nav_signals[1]._lazy is False
            np.testing.assert_allclose(
                nav_signals[1].data, base.sum(axis=(3, 4)), rtol=1e-4)
            np.testing.assert_allclose(
                np.asarray(nav_signals[0].data).ravel(),
                base.sum(axis=(1, 2, 3, 4)), rtol=1e-4)
        finally:
            session.shutdown()

    def test_time_move_reads_the_committed_array(self, monkeypatch):
        """After the fill, moving the time selector repaints the real-space
        navigator from the committed array — the recursive result is what the
        chain reads, so no second pass over the dataset happens."""
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        session = _make_session()
        try:
            s, base = _stack_5d()
            session._add_signal(s, source_path=None)
            tree = session.signal_trees[-1]
            assert _wait(lambda: tree._deep_nav_targets() is not None)
            _top, child, sel = tree._deep_nav_targets()
            assert _wait(lambda: isinstance(
                tree.navigator_signals["base"][1].data, np.ndarray))

            # Drive the time selector to the LAST time step, the way the other
            # nav tests do (pin the reported index, then force an update).
            t_last = base.shape[0] - 1
            inner = getattr(sel, "selector", sel)
            inner._get_selected_indices = lambda: np.array([[t_last]])
            sel.delayed_update_data(force=True)

            want = base[t_last].sum(axis=(2, 3))
            assert _wait(lambda: isinstance(child.current_data, np.ndarray)
                         and np.allclose(child.current_data, want, rtol=1e-4)), (
                f"real-space navigator did not follow the time axis; got "
                f"{np.asarray(child.current_data)!r}")
        finally:
            session.shutdown()


    def test_sidecar_round_trip_skips_the_compute_entirely(self, monkeypatch,
                                                           tmp_path):
        """The sidecar caches the DEEP array, so a re-open serves BOTH
        navigators from RAM with no compute at all — the whole point of caching
        the unreduced array rather than the 1-D line."""
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        # The sidecar fingerprints the source's size + mtime + shape; the file
        # only has to exist.
        src = tmp_path / "stack5d.mrc"
        src.write_bytes(b"x" * 4096)

        def _open():
            session = _make_session()
            s, base = _stack_5d()
            session._add_signal(s, source_path=str(src))
            return session, base

        session, base = _open()
        try:
            assert _wait(lambda: isinstance(
                session.signal_trees[-1].navigator_signals["base"][1].data,
                np.ndarray))
            from spyde.nav_sidecar import sidecar_path
            # Wait for the SIDECAR, not just for the navigator data. The two are
            # different events: the nav array becoming an ndarray means the
            # compute produced it, while the sidecar is written afterwards by
            # the same background work — so asserting the file the instant the
            # array appears is a race the test loses on a loaded runner
            # ("no sidecar written" on macos-py3.13, green on every other job
            # and on a local full run of the same platform + Python).
            assert _wait(lambda: os.path.exists(sidecar_path(str(src)))), \
                "no sidecar written"
        finally:
            session.shutdown()

        session, base = _open()
        try:
            tree = session.signal_trees[-1]
            # Nothing was stashed for the progressive fill — the cache IS the
            # navigator.
            assert tree._pending_nav_dask is None
            assert tree._pending_nav_deep is False
            nav_signals = tree.navigator_signals["base"]
            np.testing.assert_allclose(
                np.asarray(nav_signals[0].data).ravel(),
                base.sum(axis=(1, 2, 3, 4)), rtol=1e-4)
            np.testing.assert_allclose(
                np.asarray(nav_signals[1].data), base.sum(axis=(3, 4)), rtol=1e-4)
            assert nav_signals[1]._lazy is False
        finally:
            session.shutdown()


class TestFourDNotRegressed:
    def test_4d_still_uses_the_single_navigator_path(self, monkeypatch):
        """A 4-D dataset has ONE navigator: the stash must stay the reduced
        nav image and the deep flag must be off."""
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        session = _make_session()
        try:
            seen = {}
            from spyde.signal_tree import BaseSignalTree
            orig = BaseSignalTree._start_progressive_nav_compute

            def _spy(self, nav_dask=None, deep=None):
                if nav_dask is not None:
                    seen["shape"] = tuple(nav_dask.shape)
                    seen["deep"] = bool(deep)
                return orig(self, nav_dask, deep=deep)

            monkeypatch.setattr(
                BaseSignalTree, "_start_progressive_nav_compute", _spy)

            base = np.arange(8 * 8 * 4 * 4, dtype=np.float32).reshape(8, 8, 4, 4)
            arr = da.from_array(base, chunks=(4, 4, 4, 4))
            s = hs.signals.Signal2D(arr).as_lazy()
            s.set_signal_type("electron_diffraction")
            session._add_signal(s, source_path=None)

            assert _wait(lambda: "shape" in seen)
            assert seen["deep"] is False
            assert seen["shape"] == (8, 8)

            tree = session.signal_trees[-1]
            nav_plot = tree.navigator_plot_manager.plots[
                list(tree.navigator_plot_manager.plot_windows)[0]][0]
            want = base.sum(axis=(2, 3))
            assert _wait(lambda: isinstance(nav_plot.current_data, np.ndarray)
                         and np.isfinite(nav_plot.current_data).all())
            np.testing.assert_allclose(nav_plot.current_data, want, rtol=1e-4)
        finally:
            session.shutdown()
