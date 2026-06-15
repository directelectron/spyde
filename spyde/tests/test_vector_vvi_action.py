"""
Gating + wiring for the Vector Virtual Imaging toolbar action.

The action is requires_vectors-gated: it must be absent until a signal tree
has diffraction_vectors attached, and present after PlotState.rebuild_toolbars()
re-runs the filter. Also checks the action module's compute helper produces a
correct intensity-weighted image from a real vectors object.
"""
import types

import numpy as np
import pytest
import dask.array as da
import hyperspy.api as hs

from spyde.actions.find_vectors import _do_compute_vectors
from spyde.drawing.toolbars.plot_control_toolbar import get_toolbar_actions_for_plot


def _params():
    return {"sigma": 0.5, "kernel_radius": 3, "threshold": 0.3,
            "min_distance": 3, "subpixel": False}


def _make_vecs():
    ny, nx, ky, kx = 4, 4, 32, 32
    data = np.zeros((ny, nx, ky, kx), dtype=np.float32)
    data[:, :, 14:18, 14:18] = 100.0
    s = hs.signals.Signal2D(da.from_array(data, chunks=(2, 2, ky, kx)))
    for ax in s.axes_manager.signal_axes:
        ax.scale = 0.01
        ax.offset = -ky * 0.005
    v = _do_compute_vectors(s, _params(), None, None)
    if len(v.flat_buffer) == 0:
        pytest.skip("No vectors found")
    return v


class _FakePlot:
    def __init__(self, signal, tree, is_navigator=False):
        self.plot_state = types.SimpleNamespace(
            current_signal=signal, dimensions=2, plot=self,
        )
        self.signal_tree = tree
        self.is_navigator = is_navigator


class _FakeTree:
    diffraction_vectors = None


def _signal2d():
    """Plain 4D diffraction signal (raw data)."""
    s = hs.signals.Signal2D(np.zeros((4, 4, 8, 8), dtype=np.float32))
    s.set_signal_type("electron_diffraction")
    return s


def _vectors_image():
    """A vectors-result image (the type the gating keys on)."""
    s = hs.signals.Signal2D(np.zeros((4, 4, 8, 8), dtype=np.float32))
    s.set_signal_type("spyde_diffraction_vectors_image")
    return s


class TestVectorVVIGating:
    def test_absent_on_raw_diffraction(self):
        plot = _FakePlot(_signal2d(), _FakeTree())
        names = get_toolbar_actions_for_plot(plot.plot_state)[2]
        assert "Vector Virtual Imaging" not in names
        assert "Vector Orientation Mapping" not in names

    def test_present_on_vectors_image(self):
        plot = _FakePlot(_vectors_image(), _FakeTree())
        names = get_toolbar_actions_for_plot(plot.plot_state)[2]
        assert "Vector Virtual Imaging" in names
        assert "Vector Orientation Mapping" in names

    def test_dense_actions_excluded_on_vectors_image(self):
        # the vectors-result image is a Diffraction2D subclass, but the dense
        # diffraction actions must NOT appear on it.
        plot = _FakePlot(_vectors_image(), _FakeTree())
        names = get_toolbar_actions_for_plot(plot.plot_state)[2]
        for dense in ("Virtual Imaging", "Orientation Mapping",
                      "Find Diffraction Vectors"):
            assert dense not in names, f"{dense} should be excluded"

    def test_dense_actions_present_on_raw_diffraction(self):
        plot = _FakePlot(_signal2d(), _FakeTree())
        names = get_toolbar_actions_for_plot(plot.plot_state)[2]
        for dense in ("Virtual Imaging", "Orientation Mapping",
                      "Find Diffraction Vectors"):
            assert dense in names, f"{dense} should be present on raw data"

    def test_other_actions_unaffected(self):
        plot = _FakePlot(_signal2d(), _FakeTree())
        names = get_toolbar_actions_for_plot(plot.plot_state)[2]
        assert "Reset" in names and "Zoom In" in names


class TestVectorVVICompute:
    def test_roi_image_intensity_weighted(self):
        vecs = _make_vecs()
        img = vecs.virtual_image_from_roi_gpu(
            0.0, 0.0, vecs.kernel_radius_data * 4, 0.0,
            t=None, intensity_weighted=True,
        )
        assert img.shape == vecs.nav_shape
        assert img.sum() > 0

    def test_rect_image(self):
        vecs = _make_vecs()
        # a rectangle covering the whole detector equals a big disk total
        full_rect = vecs.virtual_image_from_rect(-1, -1, 1, 1,
                                                 intensity_weighted=True)
        full_disk = vecs.virtual_image_from_roi(0, 0, 2.0, 0.0,
                                                intensity_weighted=True)
        assert full_rect.shape == vecs.nav_shape
        assert np.isclose(full_rect.sum(), full_disk.sum())
        # a tiny rectangle away from any spot picks up nothing
        empty = vecs.virtual_image_from_rect(0.5, 0.5, 0.51, 0.51)
        assert empty.sum() == 0

    def test_rect_handles_reversed_corners(self):
        vecs = _make_vecs()
        a = vecs.virtual_image_from_rect(-1, -1, 1, 1)
        b = vecs.virtual_image_from_rect(1, 1, -1, -1)   # corners swapped
        assert np.array_equal(a, b)
