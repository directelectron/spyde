"""Navigator selector mode toggle (crosshair point vs integrating region).

Only one sub-selector may be visible at a time, and the dock toggle must switch
between them via set_selector_mode.
"""
import time

import numpy as np
import hyperspy.api as hs


def _make_4d_session():
    from spyde.backend.session import Session
    s = hs.signals.Signal2D(np.random.RandomState(0).rand(4, 5, 8, 8).astype(np.float32))
    s.set_signal_type("electron_diffraction")
    sess = Session(n_workers=1, threads_per_worker=1)
    sess._add_signal(s, source_path=None)
    time.sleep(0.6)
    return sess


def _composite(sess):
    return next(iter(sess._nav_selectors.values()))


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
