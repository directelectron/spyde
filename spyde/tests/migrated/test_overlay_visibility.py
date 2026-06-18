"""
DP marker overlays hide when their action is deselected.

The found-vectors / orientation-template / vector-OM-refine overlays draw on the
diffraction pattern only while their toolbar action (caret) is SELECTED. Closing
the caret hides the overlay (markers cleared, nav moves don't redraw); reopening
redraws at the current frame. Driven by the `set_overlay` action / the overlays'
`set_visible`.
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _wait(pred, timeout=25.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


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
        from spyde.backend.session import Session
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_calibrated_diffraction_4d(scale=0.1))
            time.sleep(0.4)
            src_plot = _signal_plot(session)
            src_tree = src_plot.signal_tree

            session._dispatch_toolbar_action(
                src_plot, "Find Diffraction Vectors",
                {"sigma": 1.0, "kernel_radius": 5, "threshold": 0.4,
                 "min_distance": 3, "subpixel": True},
            )
            assert _wait(lambda: getattr(src_tree, "_vector_overlay", None) is not None)
            overlay = src_tree._vector_overlay

            # Seed at a position that has vectors so a visible push is non-empty.
            vecs = session.signal_trees[-1].diffraction_vectors
            ys, xs = np.nonzero(vecs.count_map())
            iy, ix = int(ys[0]), int(xs[0])
            overlay._on_indices(np.array([[ix, iy]]))

            # Spy on what gets pushed to the marker group.
            pushed = []
            orig = overlay._mg.set
            overlay._mg.set = lambda **kw: pushed.append(
                np.asarray(kw.get("offsets", np.zeros((0, 2)))))

            # Deselect → markers CLEARED.
            session._set_overlay(src_plot, "Find Diffraction Vectors", False)
            assert overlay._hidden is True
            assert pushed and pushed[-1].shape[0] == 0

            # While hidden, navigating does NOT redraw.
            pushed.clear()
            overlay._on_indices(np.array([[ix, iy]]))
            assert pushed == []

            # Reselect → redrawn at the current frame (non-empty).
            session._set_overlay(src_plot, "Find Diffraction Vectors", True)
            assert overlay._hidden is False
            assert pushed and pushed[-1].shape[0] > 0
        finally:
            session.shutdown()

    def test_set_overlay_unknown_is_noop(self):
        from spyde.backend.session import Session
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_calibrated_diffraction_4d(scale=0.1))
            time.sleep(0.3)
            src_plot = _signal_plot(session)
            # No overlay yet / unknown action → must not raise.
            session._set_overlay(src_plot, "Find Diffraction Vectors", False)
            session._set_overlay(src_plot, "Nonexistent", True)
            session._set_overlay(None, "Find Diffraction Vectors", True)
        finally:
            session.shutdown()
