"""Navigator selector mode toggle (crosshair point vs integrating region).

Only one sub-selector may be visible at a time, and the dock toggle must switch
between them via set_selector_mode.
"""
import time

import numpy as np
import hyperspy.api as hs
from spyde.tests.migrated.conftest import _settle, make_session


def _make_4d_session():
    s = hs.signals.Signal2D(np.random.RandomState(0).rand(4, 5, 8, 8).astype(np.float32))
    s.set_signal_type("electron_diffraction")
    sess = make_session()
    sess._add_signal(s, source_path=None)
    _settle(sess)
    return sess


def _composite(sess):
    return next(iter(sess._nav_selectors.values()))


class TestCompositeSharesChildren:
    """The composite and BOTH sub-selectors must share ONE children mapping.

    Updates run on the ACTIVE SUB-SELECTOR (`delayed_update_data` delegates to
    `self.selector`), so `_run_update` iterates the sub-selector's dict — but
    every caller that swaps a child's update function reaches for the composite,
    because that is what `all_navigation_selectors` yields:

        sel.children[child] = render_fn

    With separate dicts those writes landed where nothing read them. On a 5-D
    vectors result the diffraction pattern kept slicing the lazy zero
    placeholder and stayed blank while the vector overlay, which rides
    `index_hooks` on the sub-selector, tracked the cursor — circles moving over
    a dead image.
    """

    def test_the_three_views_are_one_dict(self):
        sess = _make_4d_session()
        comp = _composite(sess)
        try:
            assert comp.children is comp._crosshair_selector.children
            assert comp.children is comp._rect_selector.children
            assert comp.active_children is comp._crosshair_selector.active_children
            assert comp.active_children is comp._rect_selector.active_children
        finally:
            sess.shutdown()

    def test_the_1d_composite_shares_with_both_sub_selectors(self):
        """The 1-D composite shared the POINT selector's dict but not the
        REGION one, so the same write survived in point mode and vanished in
        integrate mode."""
        import numpy as np
        import hyperspy.api as hs
        sess = make_session()
        try:
            s = hs.signals.Signal2D(
                np.random.RandomState(0).rand(6, 8, 8).astype(np.float32))
            sess._add_signal(s, source_path=None)
            _settle(sess)
            comp = next(iter(sess._nav_selectors.values()))
            assert comp.children is comp._inf_line_selector.children
            assert comp.children is comp._linear_region_selector.children

            child = next(iter(comp.children))

            def _marker(selector, plot, indices):
                return None

            comp.children[child] = _marker
            comp.set_integrating(True)
            assert comp.selector.children[child] is _marker, \
                "integrate mode would still call the old function"
        finally:
            sess.shutdown()

    def test_a_write_through_the_composite_reaches_the_active_selector(self):
        """The exact call every render-display installer makes."""
        sess = _make_4d_session()
        comp = _composite(sess)
        try:
            child = next(iter(comp.children))

            def _marker(selector, plot, indices):
                return None

            comp.children[child] = _marker
            assert comp.selector.children[child] is _marker, \
                "the update path would still call the old function"
            # …and it survives a mode switch, which swaps which sub-selector runs.
            comp.set_integrating(True)
            assert comp.selector.children[child] is _marker
            comp.set_integrating(False)
            assert comp.selector.children[child] is _marker
        finally:
            sess.shutdown()


class TestSelectorMode:
    def test_selector_info_emitted(self, captured_messages):
        sess = _make_4d_session()
        infos = [m for m in captured_messages if m.get("type") == "selector_info"]
        sess.shutdown()
        assert infos and infos[0]["mode"] == "crosshair"

    def test_only_crosshair_visible_initially(self, captured_messages):
        sess = _make_4d_session()
        comp = _composite(sess)
        ch = comp._crosshair_selector._widget
        rect = comp._rect_selector._widget
        sess.shutdown()
        assert ch.visible is True
        assert rect.visible is False

    def test_toggle_to_integrate_swaps_visibility(self, captured_messages):
        sess = _make_4d_session()
        comp = _composite(sess)
        wid = next(iter(sess._nav_selectors.keys()))

        sess.set_selector_mode(wid, integrate=True)
        assert comp._crosshair_selector._widget.visible is False
        assert comp._rect_selector._widget.visible is True
        assert comp.is_integrating is True

        sess.set_selector_mode(wid, integrate=False)
        assert comp._crosshair_selector._widget.visible is True
        assert comp._rect_selector._widget.visible is False
        sess.shutdown()

    def test_toggle_emits_updated_info(self, captured_messages):
        sess = _make_4d_session()
        wid = next(iter(sess._nav_selectors.keys()))
        captured_messages.clear()
        sess.set_selector_mode(wid, integrate=True)
        sess.shutdown()
        infos = [m for m in captured_messages if m.get("type") == "selector_info"]
        assert infos and infos[-1]["mode"] == "integrate"
