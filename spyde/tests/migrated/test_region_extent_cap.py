"""Integrating-region extent cap (Stage 0a).

An integrating ROI (2-D rectangle) / 1-D span must never grow beyond
MAX_REGION_EXTENT_PER_DIM nav positions PER navigation dimension — the widget
geometry is clamped on resize (the ROI visibly stops growing) so a region read
can never accidentally sum a huge number of nav positions. A belt-and-suspenders
clamp in _get_selected_indices bounds the index count even if the widget geometry
wasn't clamped (e.g. a programmatic set).
"""
import time

import numpy as np
import pytest
import hyperspy.api as hs

from spyde.drawing.selectors.base_selector import MAX_REGION_EXTENT_PER_DIM


def _make_4d_session():
    """A 4-D STEM scan → 2-D nav → the navigator's composite exposes a rectangle."""
    from spyde.backend.session import Session
    # Nav big enough (32x32) that a >16 rectangle is possible.
    s = hs.signals.Signal2D(
        np.random.RandomState(0).rand(32, 32, 8, 8).astype(np.float32))
    s.set_signal_type("electron_diffraction")
    sess = Session(n_workers=1, threads_per_worker=1)
    sess._add_signal(s, source_path=None)
    time.sleep(0.6)
    return sess


def _make_movie_session():
    """A 3-D in-situ movie → 1-D time nav → composite exposes a linear span."""
    from spyde.backend.session import Session
    s = hs.signals.Signal2D(
        np.random.RandomState(1).rand(64, 8, 8).astype(np.float32))
    sess = Session(n_workers=1, threads_per_worker=1)
    sess._add_signal(s, source_path=None)
    time.sleep(0.6)
    return sess


def _widget_supports_max_extent() -> bool:
    """Does the installed anyplotlib accept ``max_extent=``?

    SpyDE floors at anyplotlib>=0.4.0, but the widget-side cap landed after the
    0.4.0 release, so a valid environment can lack it. The selectors handle that
    (TypeError -> uncapped widget), and these tests skip rather than fail."""
    try:
        from anyplotlib.widgets import RectangleWidget
        RectangleWidget(lambda: None, x=0, y=0, w=1, h=1, max_extent=1)
        return True
    except TypeError:
        return False
    except Exception:
        return False


def _composite(sess):
    return next(iter(sess._nav_selectors.values()))


class TestRegionExtentCap:
    def test_rectangle_geometry_clamped_on_resize(self):
        sess = _make_4d_session()
        try:
            rect = _composite(sess)._rect_selector
            w = rect._widget
            assert w is not None, "expected a real rectangle widget"
            # Drag the far corner way past the cap.
            w.x = 2.0
            w.y = 3.0
            w.w = 100.0
            w.h = 80.0
            rect._clamp_extent()
            # The rectangle physically stops growing at the cap.
            assert float(w.w) == float(MAX_REGION_EXTENT_PER_DIM)
            assert float(w.h) == float(MAX_REGION_EXTENT_PER_DIM)
            # And the anchor (x/y) is unchanged — only the extent is pinned.
            assert float(w.x) == 2.0 and float(w.y) == 3.0
        finally:
            sess.shutdown()

    def test_rectangle_indices_bounded_per_dim(self):
        sess = _make_4d_session()
        try:
            rect = _composite(sess)._rect_selector
            w = rect._widget
            # Bypass the geometry clamp (simulate a programmatic set) to prove the
            # _get_selected_indices safety net independently.
            w.x, w.y, w.w, w.h = 0.0, 0.0, 100.0, 100.0
            idx = rect._get_selected_indices()
            assert idx.ndim == 2 and idx.shape[1] == 2
            xs = np.unique(idx[:, 0])
            ys = np.unique(idx[:, 1])
            assert len(xs) <= MAX_REGION_EXTENT_PER_DIM
            assert len(ys) <= MAX_REGION_EXTENT_PER_DIM
            # Worst case is exactly the per-dim cap squared.
            assert idx.shape[0] <= MAX_REGION_EXTENT_PER_DIM ** 2
        finally:
            sess.shutdown()

    def test_span_geometry_clamped_on_resize(self):
        sess = _make_movie_session()
        try:
            comp = _composite(sess)
            region = comp._linear_region_selector
            w = region._widget
            assert w is not None, "expected a real range widget"
            from spyde.drawing.selectors.selector1d import _signal_axis
            scale, _ = _signal_axis(region)
            w.x0 = 5.0 * scale
            w.x1 = 5.0 * scale + 90.0 * scale  # 90 indices wide → past the cap
            region._clamp_extent()
            span_indices = (float(w.x1) - float(w.x0)) / scale
            assert span_indices <= MAX_REGION_EXTENT_PER_DIM + 1e-6
            # Lower edge unchanged.
            assert abs(float(w.x0) - 5.0 * scale) < 1e-6
        finally:
            sess.shutdown()

    def test_span_indices_bounded(self):
        sess = _make_movie_session()
        try:
            region = _composite(sess)._linear_region_selector
            w = region._widget
            from spyde.drawing.selectors.selector1d import _signal_axis
            scale, offset = _signal_axis(region)
            # Programmatic oversize set, bypassing the resize clamp.
            w.x0 = offset
            w.x1 = offset + 90.0 * scale
            idx = region._get_selected_indices()
            assert idx.ndim == 2 and idx.shape[1] == 1
            assert idx.shape[0] <= MAX_REGION_EXTENT_PER_DIM
        finally:
            sess.shutdown()


class TestWidgetSideCap:
    """The cap is now ALSO pushed into the anyplotlib widget as ``max_extent``.

    That is what makes the ROI physically stop under the cursor mid-drag. The
    Python-side ``_clamp_extent`` remains as the fallback for geometry that never
    went through a drag, but it anchors on the lower edge — which can move the
    edge the user is holding, and reads as the selection jumping. With the widget
    enforcing the cap itself, that fallback should not fire during normal use.
    """

    def test_rectangle_widget_carries_the_cap(self):
        """Skips on an anyplotlib without ``max_extent``.

        The selectors degrade deliberately (they catch TypeError and build an
        uncapped widget), so asserting unconditionally would turn a supported
        configuration into a red suite. _clamp_extent still bounds the geometry
        there — that path is covered by TestRegionExtentCap above."""
        if not _widget_supports_max_extent():
            pytest.skip("anyplotlib predates widget max_extent")
        sess = _make_4d_session()
        try:
            w = _composite(sess)._rect_selector._widget
            assert w is not None
            assert float(w.max_w) == float(MAX_REGION_EXTENT_PER_DIM)
            assert float(w.max_h) == float(MAX_REGION_EXTENT_PER_DIM)
        finally:
            sess.shutdown()

    def test_span_widget_cap_tracks_the_axis_scale(self):
        """The 1-D span is in DATA units, so the cap is index-cap * scale — and
        the signal usually attaches AFTER the widget is built, so a cap fixed at
        construction would be wrong for any calibrated axis. _clamp_extent
        re-derives it. Skips on an anyplotlib without ``max_extent`` — see
        test_rectangle_widget_carries_the_cap."""
        if not _widget_supports_max_extent():
            pytest.skip("anyplotlib predates widget max_extent")
        sess = _make_movie_session()
        try:
            region = _composite(sess)._linear_region_selector
            w = region._widget
            assert w is not None
            from spyde.drawing.selectors.selector1d import _signal_axis
            scale, _ = _signal_axis(region)
            region._clamp_extent()
            expected = abs(MAX_REGION_EXTENT_PER_DIM * scale)
            assert float(w.max_extent) == expected

            # A recalibrated axis must move the cap with it.
            sig = region.current_plot.plot_state.current_signal
            sig.axes_manager.signal_axes[0].scale = float(scale) * 4.0
            region._clamp_extent()
            assert float(w.max_extent) == abs(MAX_REGION_EXTENT_PER_DIM * scale * 4.0)
        finally:
            sess.shutdown()
