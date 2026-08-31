"""Integrating-region extent cap (Stage 0a).

An integrating ROI (2-D rectangle) / 1-D span must never grow beyond
MAX_REGION_EXTENT_PER_DIM nav positions PER navigation dimension — the widget
geometry is clamped on resize (the ROI visibly stops growing) so a region read
can never accidentally sum a huge number of nav positions. A belt-and-suspenders
clamp in _get_selected_indices bounds the index count even if the widget geometry
wasn't clamped (e.g. a programmatic set).
"""
import numpy as np
import hyperspy.api as hs

from spyde.drawing.selectors.base_selector import (
    DEFAULT_REGION_EXTENT_PER_DIM, MAX_REGION_EXTENT_PER_DIM,
)
from spyde.tests.migrated.conftest import _settle, make_session


def _make_4d_session():
    """A 4-D STEM scan → 2-D nav → the navigator's composite exposes a rectangle."""
    # Nav big enough (32x32) that a >16 rectangle is possible.
    s = hs.signals.Signal2D(
        np.random.RandomState(0).rand(32, 32, 8, 8).astype(np.float32))
    s.set_signal_type("electron_diffraction")
    sess = make_session()
    sess._add_signal(s, source_path=None)
    _settle(sess)
    return sess


def _make_movie_session():
    """A 3-D in-situ movie → 1-D time nav → composite exposes a linear span."""
    s = hs.signals.Signal2D(
        np.random.RandomState(1).rand(64, 8, 8).astype(np.float32))
    sess = make_session()
    sess._add_signal(s, source_path=None)
    _settle(sess)
    return sess


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


class TestDefaultSpanOnFirstIntegrate:
    """Switching to integrate mode for the FIRST time must give a small region,
    not the whole recording.

    The 1-D span widget is constructed before the signal attaches, with x0=0 and
    x1=10 in DATA units. On a calibrated movie axis (0.05 s/frame) that is the
    entire 4-second recording, and nothing clamped it: `max_extent` only governs
    interactive drags, so the box covered everything while _get_selected_indices
    quietly capped the READ at MAX_REGION_EXTENT_PER_DIM — the drawn region and
    the displayed frame disagreed.
    """

    @staticmethod
    def _movie_session(scale=0.05, n=80):
        s = hs.signals.Signal2D(
            np.random.RandomState(2).rand(n, 8, 8).astype(np.float32))
        tax = s.axes_manager.navigation_axes[0]
        tax.name, tax.units, tax.scale = "time", "s", scale
        sess = make_session()
        sess._add_signal(s, source_path=None)
        _settle(sess)
        return sess

    def test_first_integrate_gives_the_default_width_not_the_whole_axis(self):
        sess = self._movie_session()
        try:
            comp = _composite(sess)
            region = comp._linear_region_selector
            # As built: the full-axis span that caused the bug.
            assert region._widget is not None
            comp.set_integrating(True)
            idx = region._get_selected_indices()
            assert idx.shape[0] == DEFAULT_REGION_EXTENT_PER_DIM, (
                f"first integrate produced {idx.shape[0]} frames, expected "
                f"{DEFAULT_REGION_EXTENT_PER_DIM}")
            assert idx.shape[0] < MAX_REGION_EXTENT_PER_DIM, \
                "the default must leave room to grow as well as shrink"
        finally:
            sess.shutdown()

    def test_the_drawn_span_matches_the_frames_actually_read(self):
        """The real defect: the box said 'everything', the read said 16."""
        sess = self._movie_session()
        try:
            comp = _composite(sess)
            region = comp._linear_region_selector
            comp.set_integrating(True)
            from spyde.drawing.selectors.selector1d import _signal_axis
            scale, offset = _signal_axis(region)
            w = region._widget
            drawn = abs(float(w.x1) - float(w.x0)) / abs(scale)
            read = region._get_selected_indices().shape[0]
            assert abs(drawn - read) <= 1, (
                f"the drawn span covers {drawn:g} frames but the read uses "
                f"{read} — the region shown disagrees with the frame displayed")
        finally:
            sess.shutdown()

    def test_it_centres_on_the_crosshair(self):
        sess = self._movie_session()
        try:
            comp = _composite(sess)
            line, region = comp._inf_line_selector, comp._linear_region_selector
            from spyde.drawing.selectors.selector1d import _signal_axis
            scale, offset = _signal_axis(line)
            line._widget.x = offset + 40 * scale       # park the crosshair at 40
            comp.set_integrating(True)
            idx = region._get_selected_indices().ravel()
            centre = (int(idx.min()) + int(idx.max())) / 2.0
            assert abs(centre - 40) <= 1.5, (
                f"region centred at {centre}, expected ~40 (the crosshair)")
        finally:
            sess.shutdown()

    def test_a_usable_span_survives_toggling_off_and_on(self):
        """Only an UNUSABLE span gets reseeded — a width the user chose must not
        be thrown away by toggling the mode."""
        sess = self._movie_session()
        try:
            comp = _composite(sess)
            region = comp._linear_region_selector
            comp.set_integrating(True)
            from spyde.drawing.selectors.selector1d import _signal_axis
            scale, offset = _signal_axis(region)
            region._widget.x0 = offset + 10 * scale     # user picks 3 frames
            region._widget.x1 = offset + 13 * scale
            comp.set_integrating(False)
            comp.set_integrating(True)
            idx = region._get_selected_indices().ravel()
            assert int(idx.min()) == 10 and idx.shape[0] == 3, (
                f"the user's 3-frame span at 10 was reseeded to "
                f"{idx.shape[0]} frames at {int(idx.min())}")
        finally:
            sess.shutdown()

    def test_a_short_movie_cannot_seed_past_the_end(self):
        sess = self._movie_session(n=5)
        try:
            comp = _composite(sess)
            comp.set_integrating(True)
            idx = comp._linear_region_selector._get_selected_indices().ravel()
            assert int(idx.min()) >= 0 and int(idx.max()) <= 4, \
                f"seeded span {idx.min()}..{idx.max()} runs off a 5-frame movie"
        finally:
            sess.shutdown()

    def test_uncalibrated_axis_still_gets_the_default(self):
        """scale=1.0 is the case the old x1=10 accidentally almost worked for —
        make sure the fix does not regress it."""
        sess = self._movie_session(scale=1.0)
        try:
            comp = _composite(sess)
            comp.set_integrating(True)
            idx = comp._linear_region_selector._get_selected_indices()
            assert idx.shape[0] == DEFAULT_REGION_EXTENT_PER_DIM
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
        re-derives it."""
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
