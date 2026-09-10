"""
Staged Find-Diffraction-Vectors wizard backend (Qt parity):

  fv_open → attaches a LIVE found-peaks preview overlay to the source DP so
               red circles update as you tune the sliders / move the navigator.
  fv_tune    → live-updates the preview sliders (σ / kernel radius / threshold /
               min distance / subpixel) and redraws at the current crosshair —
               NO full-dataset compute.
  fv_run     → full-dataset batch with the tuned params → a new vectors-image
               window with `tree.diffraction_vectors` attached.
  fv_close    → removes the live preview overlay (caret closed).

Memory safety: the preview only slices/computes a small nav window (radius
ceil(3σ)) around the crosshair — never the full dataset (see
test_find_vectors_memory for the batch-compute contract).
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import hyperspy.api as hs
from spyde.tests.migrated._async import quiesce, why_busy
from spyde.tests.migrated._async import wait_until
from spyde.actions.vector_overlay import overlay_static
from spyde.array_cache import reader_for_overlay
from spyde.tests.migrated.conftest import make_session, close_session


def _wait(pred, timeout=25.0):
    return wait_until(pred, timeout)


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _calibrated_diffraction_4d(nav=(4, 5), sig=(24, 24), scale=0.1):
    """A 4D-STEM stack: every pattern has a bright disk at the centre (so the
    NXCORR peak finder reliably finds at least one peak)."""
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


class TestFindVectorsWizard:
    def test_preview_tune_run(self):
        from spyde.actions.find_vectors_action import (
            fv_open, fv_tune, fv_run, fv_close,
        )
        session = make_session()
        try:
            session._add_signal(_calibrated_diffraction_4d(scale=0.1))
            assert quiesce(session), why_busy(session)
            src = _signal_plot(session)
            tree = src.signal_tree

            # ── Tune: start the live preview ─────────────────────────────────
            fv_open(session, src, {
                "sigma": 1.0, "kernel_radius": 5, "threshold": 0.4,
                "min_distance": 3, "subpixel": True,
            })
            assert _wait(lambda: getattr(tree, "_fv_preview", None) is not None), \
                "live preview overlay never attached"
            prev = tree._fv_preview
            assert (id(prev), "peaks") in src._overlay_groups   # circles exist

            # The centred disk produces at least one peak at the current
            # crosshair, read the way a navigator move reads the overlay.
            reader = reader_for_overlay(src, prev)
            assert _wait(lambda: len(reader.read_frame((0, 0))["peaks"]) >= 1), \
                "preview found no peaks on a disk pattern"

            # ── Tune: change params live (no new window) ─────────────────────
            before_trees = len(session.signal_trees)
            fv_tune(session, src, {
                "sigma": 1.0, "kernel_radius": 7, "threshold": 0.3,
                "min_distance": 4, "subpixel": False,
            })
            assert _wait(lambda: overlay_static(prev)["params"]["kernel_radius"] == 7
                         and overlay_static(prev)["params"]["subpixel"] is False)
            assert abs(overlay_static(prev)["params"]["threshold"] - 0.3) < 1e-9
            assert len(session.signal_trees) == before_trees   # tune never computes

            # ── Compute: full-dataset batch → a new vectors window ───────────
            fv_run(session, src, {
                "sigma": 1.0, "kernel_radius": 5, "threshold": 0.4,
                "min_distance": 3, "subpixel": True,
            })
            assert _wait(lambda: len(session.signal_trees) == before_trees + 1,
                         timeout=40), "vectors window never opened"
            vtree = session.signal_trees[-1]
            assert _wait(lambda: getattr(vtree, "diffraction_vectors", None) is not None,
                         timeout=40), "diffraction_vectors never attached"
            assert int(vtree.diffraction_vectors.count_map().sum()) > 0

            # Running drops the live preview (the final overlay replaces it).
            assert _wait(lambda: getattr(tree, "_fv_preview", None) is None)
            # …and attaches the persistent found-vector overlay on the source DP.
            assert _wait(lambda: getattr(tree, "_vector_overlay", None) is not None)
        finally:
            close_session(session)

    def test_stop_removes_preview(self):
        from spyde.actions.find_vectors_action import fv_open, fv_close
        session = make_session()
        try:
            session._add_signal(_calibrated_diffraction_4d(scale=0.1))
            assert quiesce(session), why_busy(session)
            src = _signal_plot(session)
            tree = src.signal_tree

            fv_open(session, src, {"sigma": 1.0, "kernel_radius": 5,
                                      "threshold": 0.4, "min_distance": 3,
                                      "subpixel": True})
            assert _wait(lambda: getattr(tree, "_fv_preview", None) is not None)
            prev = tree._fv_preview

            fv_close(session, src, {})
            assert _wait(lambda: getattr(tree, "_fv_preview", None) is None)
            assert (id(prev), "peaks") not in src._overlay_groups   # circles gone
        finally:
            close_session(session)


class TestPreviewBeamstop:
    """The beam stop the preview excludes is a static argument of the overlay
    node: detected once from a sample of frames, cached on the tree, and
    dilated on demand without re-detecting."""

    class _Tree:
        """Only the attribute the detection caches itself on."""

    def _signal_with_a_bar(self, ny=6, nx=6, ky=32, kx=32):
        frame = np.full((ky, kx), 1000.0, np.float32)
        frame[:24, 14:18] = 1.0                   # a dark bar in every pattern
        return hs.signals.Signal2D(
            np.broadcast_to(frame, (ny, nx, ky, kx)).copy())

    def test_the_mask_is_detected_once_and_dilated_on_demand(self):
        from spyde.actions.vector_overlay import _beamstop_for
        signal, tree = self._signal_with_a_bar(), self._Tree()

        assert _beamstop_for(tree, signal, {"beamstop_auto": False}) is None
        assert not hasattr(tree, "_fv_beamstop_raw"), "detection ran while off"

        mask = _beamstop_for(tree, signal, {"beamstop_auto": True,
                                            "beamstop_dilate": 5})
        assert mask is not None, "the beam stop was never detected"
        assert mask[10, 15]                        # on the bar
        raw = tree._fv_beamstop_raw
        assert raw is not None and int(mask.sum()) >= int(raw.sum())

        # A bigger dilation grows the mask from the SAME cached detection.
        wider = _beamstop_for(tree, signal, {"beamstop_auto": True,
                                             "beamstop_dilate": 12})
        assert int(wider.sum()) > int(mask.sum())
        assert tree._fv_beamstop_raw is raw, "the beam stop was detected twice"

        # Toggling off clears it without dropping the cache.
        assert _beamstop_for(tree, signal, {"beamstop_auto": False}) is None
        assert tree._fv_beamstop_raw is raw

    def test_the_mask_is_drawn_on_nav_paint_and_goes_with_the_node(self):
        """The stop is a group of the preview node like any other: its value
        rides every evaluation, it is pushed on the painter thread, and
        removing the node clears it — nothing keeps a mask alive past the
        caret that put it there."""
        from spyde.actions.find_vectors_action import fv_open, fv_close

        session = make_session()
        try:
            session._add_signal(self._signal_with_a_bar().as_lazy(),
                                source_path=None)
            assert quiesce(session), why_busy(session)
            plot = _signal_plot(session)
            tree = plot.signal_tree

            drawn = []
            plot._set_overlay_mask = lambda mask, **style: drawn.append(
                (threading.current_thread().name,
                 None if mask is None else int(np.count_nonzero(mask))))

            fv_open(session, plot, {"method": "dog", "sigma": 0.0,
                                    "kernel_radius": 3, "threshold": 8.0,
                                    "min_distance": 3, "subpixel": False,
                                    "beamstop_auto": True})
            assert _wait(lambda: getattr(tree, "_fv_preview", None) is not None, 30)
            node = tree._fv_preview
            assert _wait(lambda: any(count for _thread, count in drawn), 30),                 "the beam stop never reached the pattern"
            assert {thread for thread, _count in drawn} == {"nav-paint"}, drawn

            fv_close(session, plot, {})
            assert _wait(lambda: drawn[-1][1] is None, 10), drawn
            assert (id(node), "mask") not in plot._overlay_groups
        finally:
            close_session(session)


    def test_the_detection_never_runs_on_the_callers_thread(self, monkeypatch):
        """Detecting the stop reads frames, which is far too slow for the
        thread the caret is dispatched on. It runs on the compute backend's
        overlay lane and the mask reaches the recipe afterwards."""
        import threading
        import spyde.actions.find_vectors as find_vectors
        from spyde.actions.find_vectors_action import fv_open
        from spyde.actions.vector_overlay import tune_find_vectors_preview

        params = {"method": "dog", "sigma": 0.0, "kernel_radius": 3,
                  "threshold": 8.0, "min_distance": 3, "subpixel": False}
        session = make_session()
        try:
            session._add_signal(self._signal_with_a_bar().as_lazy(),
                                source_path=None)
            assert quiesce(session), why_busy(session)
            plot = _signal_plot(session)
            tree = plot.signal_tree
            fv_open(session, plot, dict(params))
            assert _wait(lambda: getattr(tree, "_fv_preview", None) is not None, 30)
            node = tree._fv_preview
            assert overlay_static(node)["beamstop_mask"] is None

            scans = []
            detect = find_vectors._auto_beamstop_from_signal

            def _slow_detect(signal, navigation_dimension, **kwargs):
                scans.append(threading.current_thread().name)
                time.sleep(0.3)
                return detect(signal, navigation_dimension, **kwargs)

            monkeypatch.setattr(find_vectors, "_auto_beamstop_from_signal",
                                _slow_detect)

            started = time.monotonic()
            tune_find_vectors_preview(tree, node,
                                      dict(params, beamstop_auto=True))
            elapsed = time.monotonic() - started
            assert elapsed < 0.05, f"the tune blocked for {elapsed:.3f}s"

            assert _wait(
                lambda: overlay_static(node)["beamstop_mask"] is not None, 20), \
                "the beam-stop mask never reached the recipe"
            assert scans, "the beam stop was never detected"
            assert all(name.startswith("overlay-eval") for name in scans), scans
            assert overlay_static(node)["beamstop_mask"][10, 15]      # on the bar
        finally:
            close_session(session)


class TestPreviewTransformView:
    def test_the_value_carries_the_peaks_and_the_response(self):
        """With the transform view on, one evaluation produces BOTH the peak
        markers and the image the detector found them in, with the contrast
        window the response needs."""
        from spyde.actions.vector_overlay import find_vectors_preview

        ky, kx = 32, 32
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[0:ky, 0:kx]
        frame = rng.normal(50, 3, (ky, kx)).astype(np.float32)
        for cy, cx in ((10, 10), (22, 22)):
            frame += 300 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.3 ** 2))
        params = {"method": "dog", "kernel_radius": 5, "threshold": 8.0,
                  "min_distance": 3, "subpixel": True,
                  "dog_sigma1": 0.8, "dog_sigma2": 2.0}

        value = find_vectors_preview(frame, None, params=params, sigma=0.0,
                                     beamstop_mask=None, show_transform=True)
        assert len(value["peaks"]["data"]) >= 2
        # The circle radius rides the value, so the slider needs no rebuild.
        assert value["peaks"]["radius"] == pytest.approx(np.sqrt(2.0) * 0.8)
        image, (low, high) = value["transform"]["data"], value["transform"]["levels"]
        assert image.shape == (ky, kx)
        assert low == 8.0                          # the display floor is the threshold
        assert high > low

        # With the view off there is no response to pay for.
        off = find_vectors_preview(frame, None, params=params, sigma=0.0,
                                   beamstop_mask=None, show_transform=False)
        assert off["transform"] is None
        assert len(off["peaks"]["data"]) >= 2

    def test_the_correlation_ceiling_never_moves_with_the_threshold(self):
        """Moving the threshold in correlation view must NOT move the ceiling:
        tying the two together blew the window open and washed the image to
        white. An NXCORR score has a fixed ceiling of 1.0; only the floor
        moves."""
        from spyde.actions.vector_overlay import _transform_levels

        response = np.linspace(-1, 0.64, 32 * 32).reshape(32, 32).astype(np.float32)
        for threshold in (0.5, 0.6, 0.7, 0.9):
            low, high = _transform_levels(response, "nxcorr", threshold)
            assert high == 1.0, f"the correlation ceiling moved to {high}"
            assert abs(low - threshold) < 1e-6
        # A DoG SNR has no fixed scale, so its ceiling comes from the response.
        low, high = _transform_levels(response, "dog", 0.5)
        assert high == float(np.percentile(response, 99.0))

    def test_the_painter_applies_the_contrast_window_the_value_carries(self):
        """A transform value with levels reaches the plot as those levels, not
        as an auto-levelled repaint, and holds the base frame back while it
        is showing."""
        from spyde.drawing.plots.plot import Plot

        node = type("Node", (), {})()
        painted, restored = [], []

        class _Plot:
            needs_auto_level = True
            current_data = np.ones((8, 8), np.float32)

            def set_transform_image(self, data, levels=None):
                painted.append((data.shape, levels))

            def _set_array(self, data, levels=None):
                restored.append(data.shape)

        plot = _Plot()
        plot._overlay_groups = {(id(node), "transform"): None}
        plot._live_transform_groups = set()
        image = np.zeros((8, 8), np.float32)
        key = (id(node), "transform")

        Plot._push_overlay_group(plot, node, "transform", "transform",
                                 {"data": image, "levels": (0.3, 1.0)})
        assert painted == [((8, 8), (0.3, 1.0))]
        assert plot.needs_auto_level is False
        assert Plot.has_live_transform(plot) and key in plot._live_transform_groups

        # A bare array is still a transform value; it just brings no levels.
        Plot._push_overlay_group(plot, node, "transform", "transform", image)
        assert painted[-1] == ((8, 8), None)

        # Clearing it hands the plot back to the frame the navigator read.
        Plot._push_overlay_group(plot, node, "transform", "transform", None)
        assert not Plot.has_live_transform(plot)
        assert restored == [(8, 8)]


class TestPreviewHistogram:
    def test_the_threshold_is_marked_on_the_histogram(self):
        from spyde.actions.vector_overlay import _emit_preview_histogram
        marked = []

        class _Plot:
            def _emit_histogram(self, image, low, high, threshold=None):
                marked.append((low, high, threshold))

        plot = _Plot()
        image = np.zeros((4, 4), np.float32)
        _emit_preview_histogram(plot, {"transform": {"data": image,
                                                     "levels": (0.4, 1.0)},
                                       "threshold": 0.4})
        assert marked == [(0.4, 1.0, 0.4)]
        _emit_preview_histogram(plot, {"transform": None, "threshold": 0.4})
        assert len(marked) == 1
