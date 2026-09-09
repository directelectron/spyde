"""
Pattern marker overlays hide when their action is deselected.

The found-vectors / orientation-template / vector-OM-refine overlays draw on the
diffraction pattern only while their toolbar action (caret) is SELECTED. Closing
the caret hides the overlay (markers cleared, nav moves don't redraw); reopening
redraws at the current frame. Driven by the `set_overlay` action, which reaches
`BaseSignalTree.set_overlay_visible`.
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs
from spyde.tests.migrated.conftest import _settle, close_session, make_session
from spyde.tests.migrated._async import wait_until


def _wait(pred, timeout=25.0):
    return wait_until(pred, timeout)


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _drawn(plot, node, group="found"):
    """What the plot's marker group is currently showing."""
    handle = plot._overlay_groups.get((id(node), group))
    if handle is None:
        return np.zeros((0, 2))
    return np.asarray(handle._data["offsets"])


def _calibrated_diffraction_4d(nav=(4, 5), sig=(24, 24), scale=0.1):
    data = np.zeros(nav + sig, dtype=np.float32)
    yy, xx = np.mgrid[0:sig[0], 0:sig[1]]
    disk = ((xx - 12) ** 2 + (yy - 12) ** 2 <= 16).astype(np.float32)
    for idx in np.ndindex(*nav):
        data[idx] = disk * 100.0
    s = hs.signals.Signal2D(data)
    s.set_signal_type("electron_diffraction")
    for ax in s.axes_manager.signal_axes:
        ax.scale = scale
        ax.offset = 0.0
        ax.units = "1/nm"
    return s


class TestOverlayVisibility:
    def test_found_vectors_overlay_hides_on_deselect(self):
        session = make_session()
        try:
            session._add_signal(_calibrated_diffraction_4d(scale=0.1))
            _settle(session)
            src_plot = _signal_plot(session)
            src_tree = src_plot.signal_tree

            session._dispatch_toolbar_action(
                src_plot, "Find Diffraction Vectors",
                {"sigma": 1.0, "kernel_radius": 5, "threshold": 0.4,
                 "min_distance": 3, "subpixel": True},
            )
            assert _wait(lambda: getattr(src_tree, "_vector_overlay", None) is not None)
            node = src_tree._vector_overlay

            # Seed at a position that has vectors so a visible push is non-empty.
            from spyde.drawing.overlays import refresh_overlays
            vecs = session.signal_trees[-1].diffraction_vectors
            ys, xs = np.nonzero(vecs.count_map())
            iy, ix = int(ys[0]), int(xs[0])
            refresh_overlays(src_plot, np.array([[ix, iy]]))
            assert _wait(lambda: len(_drawn(src_plot, node)) > 0)

            # Deselect → markers CLEARED.
            session._set_overlay(src_plot, "Find Diffraction Vectors", False)
            assert node.visible is False
            assert _wait(lambda: len(_drawn(src_plot, node)) == 0)

            # While hidden, navigating does NOT redraw.
            refresh_overlays(src_plot, np.array([[ix, iy]]))
            assert not _wait(lambda: len(_drawn(src_plot, node)) > 0, 1.0)

            # Reselect → redrawn at the current frame (non-empty).
            session._set_overlay(src_plot, "Find Diffraction Vectors", True)
            assert node.visible is True
            assert _wait(lambda: len(_drawn(src_plot, node)) > 0)
        finally:
            close_session(session)

    def test_set_overlay_unknown_is_noop(self):
        session = make_session()
        try:
            session._add_signal(_calibrated_diffraction_4d(scale=0.1))
            _settle(session)
            src_plot = _signal_plot(session)
            # No overlay yet / unknown action → must not raise.
            session._set_overlay(src_plot, "Find Diffraction Vectors", False)
            session._set_overlay(src_plot, "Nonexistent", True)
            session._set_overlay(None, "Find Diffraction Vectors", True)
        finally:
            close_session(session)
