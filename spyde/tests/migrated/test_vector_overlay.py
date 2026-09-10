"""Found-vector marker overlay on the diffraction pattern.

After Find Diffraction Vectors runs, the found peaks must be drawn as circle
markers on the SOURCE diffraction pattern and track the navigator. These tests
verify:

  * the overlay node is attached to the source tree as a child of the node the
    vectors were found on,
  * its marker offsets are the calibrated vectors converted to *image-pixel*
    coordinates (the same convention anyplotlib widgets/markers use), and
  * a navigator move re-evaluates it and pushes the new offsets.

Uses CALIBRATED signal axes (scale=0.1) so the kx,ky→pixel conversion is
actually exercised (a scale=1 dataset would hide a calibration error).
"""
from __future__ import annotations

import numpy as np
import hyperspy.api as hs

from spyde.array_cache import reader_for_overlay
from spyde.drawing.overlays import refresh_overlays
from spyde.tests.migrated.conftest import _settle, close_session, make_session
from spyde.tests.migrated._async import wait_until


def _wait(pred, timeout=25.0):
    return wait_until(pred, timeout)


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


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


def _drawn_offsets(plot, node):
    handle = plot._overlay_groups.get((id(node), "found"))
    if handle is None:
        return np.zeros((0, 2))
    return np.asarray(handle._data["offsets"], dtype=np.float64)


def _find_vectors(session, plot, tree):
    session._dispatch_toolbar_action(
        plot, "Find Diffraction Vectors",
        {"sigma": 1.0, "kernel_radius": 5, "threshold": 0.4,
         "min_distance": 3, "subpixel": True},
    )
    assert _wait(lambda: getattr(tree, "_vector_overlay", None) is not None), \
        "overlay never attached to the source tree"
    return tree._vector_overlay


class TestVectorOverlay:
    def test_overlay_attached_with_pixel_offsets(self):
        session = make_session()
        try:
            session._add_signal(_calibrated_diffraction_4d(scale=0.1), source_path=None)
            _settle(session)
            src_plot = _signal_plot(session)
            src_tree = src_plot.signal_tree
            node = _find_vectors(session, src_plot, src_tree)
            vecs = session.signal_trees[-1].diffraction_vectors

            assert node.overlay and node.groups["found"][0] == "circles"
            assert node.parent.signal is src_tree.root

            # Pick a nav position that has vectors and read the node there.
            cm = vecs.count_map()
            ys, xs = np.nonzero(cm)
            assert len(ys) > 0
            iy, ix = int(ys[0]), int(xs[0])
            value = reader_for_overlay(src_plot, node).read_frame((iy, ix))
            offsets = np.asarray(value["found"], dtype=np.float64)
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
            close_session(session)

    def test_overlay_updates_when_navigator_moves(self):
        session = make_session()
        try:
            session._add_signal(_calibrated_diffraction_4d(scale=0.1), source_path=None)
            _settle(session)
            src_plot = _signal_plot(session)
            src_tree = src_plot.signal_tree
            node = _find_vectors(session, src_plot, src_tree)
            vecs = session.signal_trees[-1].diffraction_vectors

            # The overlay is a child of the displayed node, which is what makes
            # the navigator refresh find it.
            assert node in src_tree.overlay_children(src_plot.plot_state.current_signal)

            # Two positions with vectors → the DRAWN offset count tracks them,
            # through the same call a navigator move makes.
            cm = vecs.count_map()
            positions = list(zip(*np.nonzero(cm)))
            assert len(positions) >= 1
            for (iy, ix) in positions[:2]:
                def _drawn_here(iy=iy, ix=ix):
                    refresh_overlays(src_plot, np.array([[ix, iy]]))
                    return len(_drawn_offsets(src_plot, node)) == int(cm[iy, ix])

                assert _wait(_drawn_here, 10)
        finally:
            close_session(session)
