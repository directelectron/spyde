"""Tuning Find-Vectors parameters re-renders the live preview peaks even when
the navigator has not moved.

A slider tweak replaces the overlay node's static arguments and re-runs the
position the navigator is already on, which is the same path a navigator move
takes: there is no second, synchronous redraw to keep in step with the first.
"""
from __future__ import annotations

import numpy as np
import hyperspy.api as hs

from spyde.actions.find_vectors_action import fv_close, fv_open, fv_tune
from spyde.actions.vector_overlay import overlay_static
from spyde.drawing.overlays import refresh_overlays
from spyde.tests.migrated._async import wait_until
from spyde.tests.migrated.conftest import _settle, close_session, make_session

PARAMS = {"method": "nxcorr", "sigma": 0.0, "kernel_radius": 3,
          "threshold": 0.5, "min_distance": 3, "subpixel": False}


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _close_disks_4d():
    """Disks close enough together that the found-peak count is sensitive to
    min_distance (a clean disk correlates at ~1, so the threshold does not
    discriminate between them; the minimum separation does)."""
    frame = np.zeros((64, 64), np.float32)
    yy, xx = np.mgrid[0:64, 0:64]
    for cy, cx in [(28, 28), (28, 36), (36, 28), (36, 36), (20, 32), (44, 32)]:
        frame += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.5 ** 2)))
    data = np.broadcast_to(frame, (2, 2, 64, 64)).copy()
    signal = hs.signals.Signal2D(data)
    signal.set_signal_type("electron_diffraction")
    return signal


def _drawn(plot, node):
    handle = plot._overlay_groups.get((id(node), "peaks"))
    if handle is None:
        return np.zeros((0, 2))
    return np.asarray(handle._data["offsets"])


class TestPreviewParameters:
    def test_tuning_min_distance_redraws_the_peaks(self):
        session = make_session()
        try:
            session._add_signal(_close_disks_4d(), source_path=None)
            _settle(session)
            plot = _signal_plot(session)
            tree = plot.signal_tree

            fv_open(session, plot, dict(PARAMS))
            assert wait_until(lambda: getattr(tree, "_fv_preview", None) is not None, 30)
            node = tree._fv_preview
            assert wait_until(lambda: len(_drawn(plot, node)) > 0, 20), \
                "the preview found no peaks on a disk pattern"

            fv_tune(session, plot, dict(PARAMS, min_distance=2))
            assert wait_until(
                lambda: overlay_static(node)["params"]["min_distance"] == 2, 20)
            assert wait_until(lambda: len(_drawn(plot, node)) > 0, 20)
            close = len(_drawn(plot, node))

            fv_tune(session, plot, dict(PARAMS, min_distance=30))
            assert wait_until(
                lambda: overlay_static(node)["params"]["min_distance"] == 30, 20)
            assert wait_until(lambda: len(_drawn(plot, node)) < close, 20), \
                f"a min_distance change did not update the peaks (still {close})"
        finally:
            close_session(session)

    def test_a_navigator_move_redraws_the_peaks(self):
        session = make_session()
        try:
            session._add_signal(_close_disks_4d(), source_path=None)
            _settle(session)
            plot = _signal_plot(session)
            tree = plot.signal_tree

            fv_open(session, plot, dict(PARAMS))
            assert wait_until(lambda: getattr(tree, "_fv_preview", None) is not None, 30)
            node = tree._fv_preview
            fv_close(session, plot, {})
            assert wait_until(lambda: getattr(tree, "_fv_preview", None) is None, 20)
            assert wait_until(lambda: len(_drawn(plot, node)) == 0, 20)

            fv_open(session, plot, dict(PARAMS))
            assert wait_until(lambda: getattr(tree, "_fv_preview", None) is not None, 30)
            node = tree._fv_preview
            refresh_overlays(plot, np.array([[1, 1]]))
            assert wait_until(lambda: len(_drawn(plot, node)) > 0, 20), \
                "a navigator move did not render the preview peaks"
        finally:
            close_session(session)
