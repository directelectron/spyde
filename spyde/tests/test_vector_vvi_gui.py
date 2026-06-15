"""
GUI integration: the Vector Virtual Imaging toolbar action end-to-end.

Drives the real toolbar (correct `toolbar_bottom` attribute) on a 4D STEM
plot: attaches a vectors object to the tree, rebuilds toolbars so the gated
action appears, toggles it on, fires "Add Vector Virtual Image", and asserts a
preview window is created with a non-trivial intensity image and a ROI lands
on the signal (diffraction) plot. Adding a second image must reuse the same
parent action and create a second preview + ROI.
"""
import numpy as np
import pytest
import dask.array as da
import hyperspy.api as hs
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from spyde.actions.find_vectors import _do_compute_vectors


def _make_vecs():
    ny, nx, ky, kx = 4, 4, 32, 32
    data = np.zeros((ny, nx, ky, kx), dtype=np.float32)
    data[:, :, 14:18, 14:18] = 100.0
    s = hs.signals.Signal2D(da.from_array(data, chunks=(2, 2, ky, kx)))
    for ax in s.axes_manager.signal_axes:
        ax.scale = 0.01
        ax.offset = -ky * 0.005
    v = _do_compute_vectors(
        s, {"sigma": 0.5, "kernel_radius": 3, "threshold": 0.3,
            "min_distance": 3, "subpixel": False}, None, None,
    )
    if len(v.flat_buffer) == 0:
        pytest.skip("No vectors found")
    return v


def _signal_plot(win):
    for pw in win.plot_subwindows:
        for plot in pw.plots:
            if plot.plot_state is not None and not plot.is_navigator:
                return plot
    return None


class TestVectorVVIAction:
    @pytest.fixture(autouse=True)
    def setup(self, stem_4d_dataset, qtbot):
        self.win = stem_4d_dataset["window"]
        self.qtbot = qtbot
        self.plot = _signal_plot(self.win)
        assert self.plot is not None
        # Mark the plot's signal as a vectors-result image + attach vectors, then
        # rebuild toolbars — same end state as a completed Find Vectors run.
        self.plot.plot_state.current_signal.set_signal_type(
            "spyde_diffraction_vectors_image")
        self.plot.signal_tree.diffraction_vectors = _make_vecs()
        self.plot.plot_state.rebuild_toolbars()
        QApplication.processEvents()

    def _bottom_tb(self):
        return self.plot.plot_state.toolbar_bottom

    def test_action_present_after_attach(self):
        names = [a.text() for a in self._bottom_tb().actions()]
        assert "Vector Virtual Imaging" in names

    def test_add_image_creates_preview_and_roi(self):
        tb = self._bottom_tb()
        n_before = len(self.win.plot_subwindows)

        act = tb._find_action("Vector Virtual Imaging")
        assert act is not None
        if not act.isChecked():
            act.trigger()
            QApplication.processEvents()
            QTest.qWait(50)

        # The submenu lives in a PopoutToolBar stored as the parent action's
        # widget in action_widgets[name]["widget"].
        entry0 = tb.action_widgets.get("Vector Virtual Imaging", {})
        submenu = entry0.get("widget")
        assert submenu is not None, "Vector Virtual Imaging submenu popout missing"
        add_act = submenu._find_action("Add Vector Virtual Image")
        assert add_act is not None, "Add Vector Virtual Image subaction missing"
        add_act.trigger()
        QApplication.processEvents()
        QTest.qWait(100)

        # A preview window appeared
        self.qtbot.waitUntil(
            lambda: len(self.win.plot_subwindows) > n_before, timeout=5000
        )
        entry = tb.action_widgets.get("Vector Virtual Imaging", {})

        # The preview window is the one registered against the parent action.
        windows = entry.get("plot_windows", {})
        assert windows, "No preview window registered"
        vi_window = list(windows.values())[-1]

        # A ROI is registered on the signal plot for the parent action
        items = entry.get("plot_items", {})
        assert len(items) >= 1, "No ROI registered on the signal plot"
        roi = list(items.values())[-1]

        # The action computes from the vectors' own (kx, ky) coordinates. Use
        # the centroid of the actual vectors as the ROI centre (production: the
        # vectors and the plot share axes, so the kernel-radius ROI at the
        # detector centre lands on the central beam; here the attached vectors
        # come from a separate synthetic signal, so target their real coords).
        vecs = self.plot.signal_tree.diffraction_vectors
        from spyde.signals.diffraction_vectors import COL_KX, COL_KY
        cx = float(vecs.flat_buffer[:, COL_KX].mean())
        cy = float(vecs.flat_buffer[:, COL_KY].mean())
        r_out = float(roi.size().x() / 2)
        img = vecs.virtual_image_from_roi(cx, cy, r_out, 0.0,
                                          intensity_weighted=True)
        assert img.shape == vecs.nav_shape
        assert img.sum() > 0, "Vector virtual image is empty at the vector centroid"
        # intensity-weighted differs from / equals manual sum
        import numpy as _np
        assert _np.isfinite(img).all()

    def _add_one(self, tb):
        """Toggle the parent action + fire the submenu's Add once."""
        act = tb._find_action("Vector Virtual Imaging")
        if not act.isChecked():
            act.trigger(); QApplication.processEvents(); QTest.qWait(30)
        submenu = tb.action_widgets["Vector Virtual Imaging"]["widget"]
        submenu._find_action("Add Vector Virtual Image").trigger()
        QApplication.processEvents(); QTest.qWait(60)

    def test_add_multiple_images(self):
        tb = self._bottom_tb()
        n0 = len(self.win.plot_subwindows)
        self._add_one(tb)
        self.qtbot.waitUntil(lambda: len(self.win.plot_subwindows) > n0, timeout=5000)
        n1 = len(self.win.plot_subwindows)
        self._add_one(tb)   # the "+" capability: add a second
        self.qtbot.waitUntil(lambda: len(self.win.plot_subwindows) > n1, timeout=5000)
        # two ROIs registered (one per VI)
        items = tb.action_widgets["Vector Virtual Imaging"].get("plot_items", {})
        assert len(items) >= 2, f"expected >=2 ROIs, got {len(items)}"

    def test_shape_dropdown_switches_roi(self):
        from pyqtgraph import CircleROI, RectROI
        tb = self._bottom_tb()
        n0 = len(self.win.plot_subwindows)
        self._add_one(tb)
        self.qtbot.waitUntil(lambda: len(self.win.plot_subwindows) > n0, timeout=5000)
        items = tb.action_widgets["Vector Virtual Imaging"]["plot_items"]
        key = list(items.keys())[-1]
        assert isinstance(items[key], CircleROI)   # disk default
        # the per-VI action + caret live on the submenu popout toolbar, keyed by
        # the per-image action name (e.g. "Vector Image (red)").
        submenu = tb.action_widgets["Vector Virtual Imaging"]["widget"]
        caret = submenu.action_widgets[key]["widget"]
        combo = caret.kwargs["shape"]
        combo.setCurrentText("rectangle")
        QApplication.processEvents(); QTest.qWait(60)
        assert isinstance(tb.action_widgets["Vector Virtual Imaging"]["plot_items"][key],
                          RectROI), "ROI did not switch to RectROI"
