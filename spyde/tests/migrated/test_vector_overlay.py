"""
Found-vector marker overlay on the diffraction pattern (Qt parity).

After Find Diffraction Vectors runs, the found peaks must be drawn as circle
markers on the SOURCE diffraction pattern and track the navigator — like the Qt
scatter overlay. These tests verify:

  * the overlay is attached to the source tree,
  * its marker offsets are the calibrated vectors converted to *image-pixel*
    coordinates (the same convention anyplotlib widgets/markers use), and
  * moving the navigator (firing the selector index hook) re-pushes offsets.

Uses CALIBRATED signal axes (scale=0.1) so the kx,ky→pixel conversion is
actually exercised (a scale=1 dataset would hide a calibration error).
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


def _calibrated_diffraction_4d(scale=0.1):
    nav, sig = (4, 5), (24, 24)
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


class TestVectorOverlay:
    def test_overlay_attached_with_pixel_offsets(self):
        from spyde.backend.session import Session
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_calibrated_diffraction_4d(scale=0.1), source_path=None)
            time.sleep(0.4)
            src_plot = _signal_plot(session)
            src_tree = src_plot.signal_tree

            session._dispatch_toolbar_action(
                src_plot, "Find Diffraction Vectors",
                {"sigma": 1.0, "kernel_radius": 5, "threshold": 0.4,
                 "min_distance": 3, "subpixel": True},
            )
            assert _wait(lambda: getattr(src_tree, "_vector_overlay", None) is not None), \
                "overlay never attached to the source tree"
            overlay = src_tree._vector_overlay
            vecs = session.signal_trees[-1].diffraction_vectors
            assert overlay._mg is not None

            # Pick a nav position that has vectors and push it through the hook.
            cm = vecs.count_map()
            ys, xs = np.nonzero(cm)
            assert len(ys) > 0
            iy, ix = int(ys[0]), int(xs[0])
            overlay._on_indices(np.array([[ix, iy]]))

            offsets = np.asarray(overlay._mg._data["offsets"], dtype=np.float64)
            assert len(offsets) == int(cm[iy, ix])

            # Offsets must be calibrated kx,ky converted back to PIXELS:
            #   px = (k - offset) / scale, in [0, sig_size).
            kxy = np.asarray(vecs.kxy_at(iy, ix), dtype=np.float64)
            x_scale = float(vecs.sig_axes[0].scale)
            y_scale = float(vecs.sig_axes[1].scale)
            exp_x = (kxy[:, 0] - float(vecs.sig_axes[0].offset)) / x_scale
            exp_y = (kxy[:, 1] - float(vecs.sig_axes[1].offset)) / y_scale
            assert np.allclose(offsets[:, 0], exp_x, atol=1e-3)
            assert np.allclose(offsets[:, 1], exp_y, atol=1e-3)
            # A bright disk at pixel (12,12) → markers near the frame centre.
            assert offsets.min() >= 0 and offsets.max() < 24
            assert abs(offsets[:, 0].mean() - 12) < 3
            assert abs(offsets[:, 1].mean() - 12) < 3
        finally:
            session.shutdown()

    def test_overlay_updates_when_navigator_moves(self):
        from spyde.backend.session import Session
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_calibrated_diffraction_4d(scale=0.1), source_path=None)
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
            vecs = session.signal_trees[-1].diffraction_vectors

            # The hook is registered on the navigator selector; firing it via the
            # selector (not directly) is what happens on a real drag.
            sels = [s for s in src_tree.navigator_plot_manager.all_navigation_selectors
                    if overlay._on_indices in s.index_hooks]
            assert sels, "overlay hook was not registered on any navigator selector"

            # Two positions with vectors → the pushed offset count tracks them.
            cm = vecs.count_map()
            positions = list(zip(*np.nonzero(cm)))
            assert len(positions) >= 1
            for (iy, ix) in positions[:2]:
                overlay._on_indices(np.array([[ix, iy]]))
                pushed = np.asarray(overlay._mg._data["offsets"])
                assert len(pushed) == int(cm[iy, ix])
        finally:
            session.shutdown()
