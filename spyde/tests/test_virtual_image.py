"""Tests for the virtual image compute kernel."""
import numpy as np
import dask.array as da
import pytest

from distributed import Future


class TestVirtualImageKernel:
    @pytest.fixture(autouse=True)
    def client(self, stem_4d_dataset):
        self.win = stem_4d_dataset["window"]
        self.client = self.win.dask_manager.client

    def _mask(self, nkx=8, nky=8):
        mask = np.zeros((nkx, nky), dtype=np.float32)
        mask[2:6, 2:6] = 1.0
        return mask

    def test_4d_output_shape(self):
        from spyde.drawing.update_functions import compute_virtual_image_kernel
        data = da.ones((4, 4, 8, 8), dtype=np.float32, chunks=(2, 2, 8, 8))
        mask = self._mask()
        future = compute_virtual_image_kernel(data, mask, self.client, None)
        assert isinstance(future, Future)
        result = future.result()
        assert result.shape == (4, 4)

    def test_4d_values_match_reference(self):
        from spyde.drawing.update_functions import compute_virtual_image_kernel
        rng = np.random.default_rng(0)
        data_np = rng.random((4, 4, 8, 8)).astype(np.float32)
        mask = self._mask()
        data = da.from_array(data_np, chunks=(2, 2, 8, 8))
        future = compute_virtual_image_kernel(data, mask, self.client, None)
        result = future.result()
        expected = (data_np * mask).sum(axis=(-2, -1))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_4d_values_chunked_signal_axes(self):
        """Signal axes chunked (typical for tiled STEM data) must give correct result."""
        from spyde.drawing.update_functions import compute_virtual_image_kernel
        rng = np.random.default_rng(1)
        data_np = rng.random((4, 4, 8, 8)).astype(np.float32)
        mask = self._mask()
        # Signal axes chunked into 4×4 tiles — this was the failing case
        data = da.from_array(data_np, chunks=(1, 1, 4, 4))
        future = compute_virtual_image_kernel(data, mask, self.client, None)
        result = future.result()
        expected = (data_np * mask).sum(axis=(-2, -1))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_3d_input(self):
        from spyde.drawing.update_functions import compute_virtual_image_kernel
        data = da.ones((4, 8, 8), dtype=np.float32, chunks=(2, 8, 8))
        mask = self._mask()
        future = compute_virtual_image_kernel(data, mask, self.client, None)
        result = future.result()
        assert result.shape == (4,)

    def test_5d_input(self):
        from spyde.drawing.update_functions import compute_virtual_image_kernel
        data = da.ones((2, 4, 4, 8, 8), dtype=np.float32, chunks=(1, 2, 2, 8, 8))
        mask = self._mask()
        future = compute_virtual_image_kernel(data, mask, self.client, None)
        result = future.result()
        assert result.shape == (2, 4, 4)

    def test_6d_input(self):
        from spyde.drawing.update_functions import compute_virtual_image_kernel
        data = da.ones((2, 3, 4, 4, 8, 8), dtype=np.float32, chunks=(1, 1, 2, 2, 8, 8))
        mask = self._mask()
        future = compute_virtual_image_kernel(data, mask, self.client, None)
        result = future.result()
        assert result.shape == (2, 3, 4, 4)

    def test_numpy_input_works(self):
        from spyde.drawing.update_functions import compute_virtual_image_kernel
        data_np = np.ones((4, 4, 8, 8), dtype=np.float32)
        mask = self._mask()
        data = da.from_array(data_np)
        future = compute_virtual_image_kernel(data, mask, self.client, None)
        result = future.result()
        assert result.shape == (4, 4)

    def test_gpu_annotation_branch_returns_future(self):
        from spyde.drawing.update_functions import compute_virtual_image_kernel
        data = da.ones((4, 4, 8, 8), dtype=np.float32, chunks=(2, 2, 8, 8))
        mask = self._mask()
        # Pass a fake address — annotation is set on the graph but scheduling
        # falls through to CPU workers when no GPU worker matches.
        future = compute_virtual_image_kernel(data, mask, self.client, "tcp://fake:8786")
        assert isinstance(future, Future)
        result = future.result()
        assert result.shape == (4, 4)


class TestVirtualImageEndToEnd:
    """Integration test: roi_to_mask + compute_virtual_image_kernel on a known dataset.

    Dataset: 4x4 nav grid. Some positions have a 4x4 rectangle of ones in the
    centre of a 16x16 diffraction pattern, others are all zeros.

    Pattern:
        (0,0)=filled  (0,1)=empty  (0,2)=filled  (0,3)=empty
        (1,0)=empty   (1,1)=filled (1,2)=empty   (1,3)=filled
        (2,0)=filled  (2,1)=empty  (2,2)=filled  (2,3)=empty
        (3,0)=empty   (3,1)=filled (3,2)=empty   (3,3)=filled

    The rectangle covers diffraction pixels [6:10, 6:10] (4x4 = 16 pixels of value 1.0).
    A RectROI placed exactly over that region should give virtual_image[ny,nx] = 16
    for filled positions and 0 for empty ones.
    """

    @pytest.fixture(autouse=True)
    def client(self, stem_4d_dataset):
        self.win = stem_4d_dataset["window"]
        self.client = self.win.dask_manager.client

    def _make_signal(self, chunks):
        """4D STEM signal: filled positions have a 4x4 rectangle in diffraction centre."""
        import hyperspy.api as hs
        nav_y, nav_x, ky, kx = 4, 4, 16, 16
        rect_y, rect_x = slice(6, 10), slice(6, 10)  # 4x4 centre rectangle

        data = np.zeros((nav_y, nav_x, ky, kx), dtype=np.float32)
        for ny in range(nav_y):
            for nx in range(nav_x):
                if (ny + nx) % 2 == 0:  # checkerboard of filled/empty
                    data[ny, nx, rect_y, rect_x] = 1.0

        sig = hs.signals.Signal2D(da.from_array(data, chunks=chunks))
        # Set unit scale and zero offset on signal axes so ROI coords == pixel coords
        sig.axes_manager.signal_axes[0].scale = 1.0   # kx
        sig.axes_manager.signal_axes[0].offset = 0.0
        sig.axes_manager.signal_axes[1].scale = 1.0   # ky
        sig.axes_manager.signal_axes[1].offset = 0.0
        return sig, data

    def _make_rect_roi(self):
        """RectROI exactly covering the 4x4 rectangle at diffraction pixels [6:10, 6:10].

        pyqtgraph scene-x = ky (data axis 0 of signal slice).
        pyqtgraph scene-y = kx (data axis 1 of signal slice).
        Rectangle runs ky 6..10 (scene-x) and kx 6..10 (scene-y).
        RectROI pos=(scene_x0, scene_y0), size=(scene_dx, scene_dy).
        """
        from pyqtgraph import RectROI
        return RectROI(pos=(6, 6), size=(4, 4))

    def _expected(self, data):
        """Expected virtual image: 16 for filled positions, 0 for empty."""
        expected = np.zeros((4, 4), dtype=np.float32)
        for ny in range(4):
            for nx in range(4):
                expected[ny, nx] = data[ny, nx, 6:10, 6:10].sum()
        return expected

    def _run(self, chunks):
        from spyde.actions.pyxem import roi_to_mask
        from spyde.drawing.update_functions import compute_virtual_image_kernel

        sig, data_np = self._make_signal(chunks)
        roi = self._make_rect_roi()
        mask = roi_to_mask(roi, sig)

        # Mask must be (ky_size, kx_size) = (16, 16)
        assert mask.shape == (16, 16), f"mask shape {mask.shape}"
        # Mask must select exactly the rectangle pixels [6:10, 6:10]
        assert mask[6:10, 6:10].sum() == 16, "mask should select 16 pixels"
        assert mask[:6, :].sum() == 0, "pixels outside rect should be 0"
        assert mask[10:, :].sum() == 0, "pixels outside rect should be 0"

        future = compute_virtual_image_kernel(sig.data, mask, self.client, None)
        result = future.result()

        expected = self._expected(data_np)
        np.testing.assert_allclose(
            result, expected, atol=1e-4,
            err_msg=f"chunks={chunks}: result mismatch\n{result}\n!=\n{expected}"
        )

    def test_single_chunk(self):
        self._run(chunks=(4, 4, 16, 16))

    def test_nav_chunked(self):
        """Nav axes chunked into rows of 1 — typical real STEM chunking."""
        self._run(chunks=(1, 4, 16, 16))

    def test_nav_fully_chunked(self):
        """Both nav axes chunked individually."""
        self._run(chunks=(1, 1, 16, 16))

    def test_signal_axes_chunked(self):
        """Signal axes also chunked — was the original failing case."""
        self._run(chunks=(1, 1, 4, 4))

    def test_mixed_chunk_sizes(self):
        """Uneven nav chunks like a real large dataset (22-row tiles, single signal chunk)."""
        self._run(chunks=(2, 3, 16, 16))


class TestGPUWorkerSetup:
    def test_probe_gpus_returns_zero_when_absent(self):
        """_probe_gpus returns 0 when nvidia-smi is not found."""
        import unittest.mock as mock
        from spyde.dask_manager import _probe_gpus
        with mock.patch("spyde.dask_manager.subprocess.run", side_effect=FileNotFoundError):
            assert _probe_gpus() == 0

    def test_probe_gpus_returns_zero_on_timeout(self):
        import unittest.mock as mock
        import subprocess
        from spyde.dask_manager import _probe_gpus
        with mock.patch("spyde.dask_manager.subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 3)):
            assert _probe_gpus() == 0

    def test_probe_gpus_returns_count_from_mocked_output(self):
        import unittest.mock as mock
        from spyde.dask_manager import _probe_gpus
        fake_result = mock.Mock()
        fake_result.returncode = 0
        fake_result.stdout = b"NVIDIA GeForce RTX 3080\nNVIDIA GeForce RTX 3080\n"
        with mock.patch("spyde.dask_manager.subprocess.run", return_value=fake_result):
            assert _probe_gpus() == 2

    def test_gpu_worker_address_is_none_when_no_gpu(self, stem_4d_dataset):
        """_gpu_worker_address is None when no GPU is present (default on CI)."""
        win = stem_4d_dataset["window"]
        assert hasattr(win, "dask_manager")
        import subprocess
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, timeout=3,
            )
            has_gpu = r.returncode == 0 and r.stdout.strip()
        except Exception:
            has_gpu = False
        if not has_gpu:
            assert win.dask_manager.gpu_worker_address is None


class TestComputeStatusIndicator:
    def test_import(self):
        from spyde.qt.compute_status_indicator import ComputeStatusIndicator
        assert ComputeStatusIndicator is not None

    def test_states(self, qtbot):
        from spyde.qt.compute_status_indicator import ComputeStatusIndicator
        w = ComputeStatusIndicator()
        qtbot.addWidget(w)
        w.show()

        w.set_idle()
        assert w._state == "idle"

        w.set_computing(total_tasks=10)
        assert w._state == "computing"
        assert w._total_tasks == 10

        w.update_progress(5)
        assert w._completed_tasks == 5

        w.set_done()
        assert w._state == "done"

    def test_color_matches_roi_color(self, qtbot):
        """Indicator color must match the color string passed at construction."""
        from PySide6.QtGui import QColor
        from spyde.qt.compute_status_indicator import ComputeStatusIndicator
        w = ComputeStatusIndicator(color="red")
        qtbot.addWidget(w)
        assert w._color == QColor("red")

        w2 = ComputeStatusIndicator(color="cyan")
        qtbot.addWidget(w2)
        assert w2._color == QColor("cyan")

    def test_indicator_on_preview_window_has_roi_color(self, qtbot, stem_4d_dataset):
        """The indicator attached to a virtual preview window must use the ROI's color."""
        from PySide6.QtGui import QColor
        win = stem_4d_dataset["window"]
        tb, vi_widget, action_name, caret_box, roi, preview_window = _add_virtual_detector(qtbot, win)

        assert preview_window is not None
        indicator = preview_window._compute_indicator
        assert indicator is not None

        # The action name encodes the color: "Virtual Image (red)", "Virtual Image (green)", etc.
        color_name = action_name.split("(")[1].rstrip(")")
        assert indicator._color == QColor(color_name), (
            f"Indicator color {indicator._color.name()} does not match ROI color {color_name}"
        )


def _add_virtual_detector(qtbot, win):
    """
    Shared helper: enable Virtual Imaging, add one detector, return
    (toolbar_bottom, vi_widget, new_action_name, caret_box, roi, preview_plot_window).
    """
    nav, sig = win.plots
    tb = sig.plot_state.toolbar_bottom
    vi_action = None
    for a in tb.actions():
        if a.text() == "Virtual Imaging":
            vi_action = a
            break
    assert vi_action is not None, "Virtual Imaging action not found in toolbar"

    vi_action.trigger()
    qtbot.wait(200)

    vi_widget = tb.action_widgets["Virtual Imaging"]["widget"]
    for a in vi_widget.actions():
        if a.text() == "Add Virtual Image":
            a.trigger()
            break
    qtbot.wait(300)

    new_action = vi_widget.actions()[-1]
    new_action.trigger()
    qtbot.wait(300)

    action_name = new_action.text()
    caret_box = vi_widget.action_widgets[action_name]["widget"]
    roi = tb.action_widgets["Virtual Imaging"]["plot_items"][action_name]
    preview_window = tb.action_widgets["Virtual Imaging"].get("plot_windows", {}).get(action_name)

    return tb, vi_widget, action_name, caret_box, roi, preview_window


class TestVirtualImageLivePreview:
    """End-to-end tests: ROI placement → preview window → live update → indicator."""

    def test_preview_window_is_frameless(self, qtbot, stem_4d_dataset):
        """The virtual preview PlotWindow must use FramelessWindowHint (no Qt border)."""
        from PySide6.QtCore import Qt
        win = stem_4d_dataset["window"]
        tb, vi_widget, action_name, caret_box, roi, preview_window = _add_virtual_detector(qtbot, win)

        assert preview_window is not None, "No preview PlotWindow registered with the toolbar"
        flags = preview_window.windowFlags()
        assert flags & Qt.WindowType.FramelessWindowHint, (
            "Preview PlotWindow does not have FramelessWindowHint — Qt border will be shown"
        )

    def test_preview_window_added_to_plot_subwindows(self, qtbot, stem_4d_dataset):
        """The preview window must appear in main_window.plot_subwindows."""
        win = stem_4d_dataset["window"]
        n_before = len(win.plot_subwindows)
        tb, vi_widget, action_name, caret_box, roi, preview_window = _add_virtual_detector(qtbot, win)
        assert len(win.plot_subwindows) == n_before + 1, (
            "Preview PlotWindow was not appended to main_window.plot_subwindows"
        )

    def test_subwindow_activation_with_preview_window_does_not_crash(self, qtbot, stem_4d_dataset):
        """Activating any subwindow after adding a virtual detector must not raise AttributeError.

        Regression test for: all_plot_windows iterating the preview PlotWindow
        (which has current_plot_state=None) and calling .toolbar_right on None.
        """
        win = stem_4d_dataset["window"]
        tb, vi_widget, action_name, caret_box, roi, preview_window = _add_virtual_detector(qtbot, win)

        subwindows = win.mdi_area.subWindowList()
        assert len(subwindows) >= 3, "Expected at least 3 subwindows (nav, sig, preview)"

        # Activate each subwindow in turn — this triggers on_subwindow_activated →
        # all_plot_windows, which previously crashed when it hit the preview window.
        for sw in subwindows:
            win.mdi_area.setActiveSubWindow(sw)
            qtbot.wait(100)  # let Qt process the activation event

        # If we reach here without an exception, the fix is working.

    def test_roi_is_on_signal_diffraction_plot(self, qtbot, stem_4d_dataset):
        """The virtual ROI must be on the signal (diffraction pattern) plot, not the navigator."""
        win = stem_4d_dataset["window"]
        nav, sig = win.plots
        tb, vi_widget, action_name, caret_box, roi, preview_window = _add_virtual_detector(qtbot, win)

        assert roi in sig.items, (
            "ROI is not on the diffraction (signal) plot — it may be on the wrong plot"
        )
        assert roi not in nav.items, "ROI should not be on the navigator plot"

    def test_compute_updates_preview_image(self, qtbot, stem_4d_dataset):
        """Computing must produce a rendered image in the preview plot's image_item."""
        win = stem_4d_dataset["window"]
        tb, vi_widget, action_name, caret_box, roi, preview_window = _add_virtual_detector(qtbot, win)

        assert preview_window is not None
        preview_plot = preview_window.plots[0]

        # Live recompute is off; trigger via the title-bar Compute button
        preview_window.title_bar.compute_button.click()

        # Wait until the future resolves and the image is rendered
        qtbot.waitUntil(
            lambda: (
                preview_plot.current_data is not None
                and not isinstance(preview_plot.current_data, Future)
            ),
            timeout=10000,
        )

        img = preview_plot.image_item.image
        assert img is not None, "image_item.image is None — image was never rendered to the scene"
        assert img.ndim == 2, f"Expected 2D image, got shape {img.shape}"
        assert img.shape[0] > 0 and img.shape[1] > 0, "Image has zero-size dimension"

    def test_different_roi_positions_produce_different_images(self, qtbot, stem_4d_dataset):
        """Moving the ROI to a different position must produce a different virtual image."""
        win = stem_4d_dataset["window"]
        tb, vi_widget, action_name, caret_box, roi, preview_window = _add_virtual_detector(qtbot, win)

        assert preview_window is not None
        preview_plot = preview_window.plots[0]

        # First position — compute and wait for result
        preview_window.title_bar.compute_button.click()
        qtbot.waitUntil(
            lambda: (
                preview_plot.current_data is not None
                and not isinstance(preview_plot.current_data, Future)
            ),
            timeout=10000,
        )
        first_image = preview_plot.image_item.image.copy()

        # Move ROI to a substantially different position and compute again
        old_pos = roi.pos()
        sig_plot = win.plots[1]  # diffraction plot
        image_item = sig_plot.image_item
        # Move ~30% of image width
        shift = image_item.width() * 0.3
        roi.setPos(old_pos.x() + shift, old_pos.y() + shift)
        preview_window.title_bar.compute_button.click()

        qtbot.waitUntil(
            lambda: (
                preview_plot.current_data is not None
                and not isinstance(preview_plot.current_data, Future)
                and not np.array_equal(preview_plot.image_item.image, first_image)
            ),
            timeout=10000,
        )

        second_image = preview_plot.image_item.image
        assert not np.array_equal(first_image, second_image), (
            "Virtual image did not change after moving the ROI"
        )

    def test_indicator_transitions_idle_computing_done(self, qtbot, stem_4d_dataset):
        """ComputeStatusIndicator must go idle→computing after Compute, then done."""
        win = stem_4d_dataset["window"]
        tb, vi_widget, action_name, caret_box, roi, preview_window = _add_virtual_detector(qtbot, win)

        assert preview_window is not None
        indicator = preview_window._compute_indicator
        assert indicator is not None, "No ComputeStatusIndicator attached to preview window"

        assert indicator._state == "idle", f"Expected idle before computation, got {indicator._state}"

        preview_window.title_bar.compute_button.click()
        qtbot.waitUntil(lambda: indicator._state == "computing", timeout=2000)

        qtbot.waitUntil(lambda: indicator._state in ("done", "idle"), timeout=10000)

    def test_virtual_imaging_toggle_hides_roi_and_preview(self, qtbot, stem_4d_dataset):
        """Toggling the Virtual Imaging toolbar button hides ROI and preview window together."""
        win = stem_4d_dataset["window"]
        nav, sig = win.plots
        tb, vi_widget, action_name, caret_box, roi, preview_window = _add_virtual_detector(qtbot, win)

        vi_action = None
        for a in tb.actions():
            if a.text() == "Virtual Imaging":
                vi_action = a
                break

        assert roi.isVisible(), "ROI should be visible after adding detector"
        assert preview_window.isVisible(), "Preview window should be visible after adding detector"

        # Toggle OFF
        vi_action.trigger()
        qtbot.wait(200)
        assert not roi.isVisible(), "ROI should be hidden after toggling Virtual Imaging OFF"
        assert not preview_window.isVisible(), "Preview window should be hidden after toggling OFF"

        # Toggle ON
        vi_action.trigger()
        qtbot.wait(200)
        assert roi.isVisible(), "ROI should be visible after toggling Virtual Imaging ON again"
        assert preview_window.isVisible(), "Preview window should be visible after toggling ON again"

    def test_roi_move_does_not_trigger_computation(self, qtbot, stem_4d_dataset):
        """Live recompute is permanently off: moving the ROI must NOT launch a
        dask graph (computing on every drag made the app appear frozen)."""
        win = stem_4d_dataset["window"]
        tb, vi_widget, action_name, caret_box, roi, preview_window = _add_virtual_detector(qtbot, win)

        assert preview_window is not None
        preview_plot = preview_window.plots[0]

        initial_data = preview_plot.current_data  # None — no computation yet

        # Move ROI and emit
        roi.sigRegionChangeFinished.emit(roi)
        qtbot.wait(2000)

        assert preview_plot.current_data is initial_data, (
            "Computation was triggered by an ROI move (live should be off)"
        )

    def test_title_bar_compute_button_triggers_one_computation(self, qtbot, stem_4d_dataset):
        """The title-bar Compute button must trigger the VI computation."""
        win = stem_4d_dataset["window"]
        tb, vi_widget, action_name, caret_box, roi, preview_window = _add_virtual_detector(qtbot, win)

        assert preview_window is not None
        preview_plot = preview_window.plots[0]

        compute_btn = preview_window.title_bar.compute_button
        assert compute_btn.isVisibleTo(preview_window.title_bar), (
            "Title-bar Compute button is not shown on the VI preview window"
        )
        compute_btn.click()

        qtbot.waitUntil(
            lambda: (
                preview_plot.current_data is not None
                and not isinstance(preview_plot.current_data, Future)
            ),
            timeout=10000,
        )

        img = preview_plot.image_item.image
        assert img is not None, "No image rendered after pressing Compute"
        assert img.ndim == 2

    def test_compute_and_commit_buttons_side_by_side(self, qtbot, stem_4d_dataset):
        """Compute and Commit live together on the preview title bar; the
        caret's Submit button is hidden (redundant)."""
        win = stem_4d_dataset["window"]
        tb, vi_widget, action_name, caret_box, roi, preview_window = _add_virtual_detector(qtbot, win)

        title_bar = preview_window.title_bar
        assert title_bar.compute_button.parent() is title_bar
        assert title_bar.commit_button.parent() is title_bar
        assert title_bar.compute_button.isVisibleTo(title_bar)
        assert title_bar.commit_button.isVisibleTo(title_bar)
        assert not caret_box.submit_button.isVisibleTo(caret_box), (
            "Caret Submit button should be hidden — Commit handles it"
        )

    def test_roi_type_switch_removes_old_roi_from_plot(self, qtbot, stem_4d_dataset):
        """Switching detector type must remove the old ROI from the signal plot scene."""
        from pyqtgraph import CircleROI, RectROI
        win = stem_4d_dataset["window"]
        nav, sig = win.plots
        tb, vi_widget, action_name, caret_box, roi, preview_window = _add_virtual_detector(qtbot, win)

        # Initial ROI is a CircleROI (disk is the default)
        assert isinstance(roi, CircleROI), f"Expected CircleROI, got {type(roi)}"
        assert roi in sig.items, "Initial CircleROI not in signal plot"

        old_roi = roi

        # Switch to rectangle
        type_widget = caret_box.get_parameter_widget("type")
        type_widget.setCurrentText("rectangle")
        qtbot.wait(200)

        # Old CircleROI must be gone from the scene
        assert old_roi not in sig.items, (
            "Old CircleROI is still in the signal plot after switching to rectangle"
        )
        # New RectROI must be in the scene
        new_roi = tb.action_widgets["Virtual Imaging"]["plot_items"][action_name]
        assert isinstance(new_roi, RectROI), f"Expected RectROI after switch, got {type(new_roi)}"
        assert new_roi in sig.items, "New RectROI not in signal plot after type switch"


class TestVirtualImageCommit:
    """End-to-end commit tests using the title-bar Commit button."""

    def _setup_with_preview(self, qtbot, win):
        """Add detector, trigger first computation, wait for preview image."""
        tb, vi_widget, action_name, caret_box, roi, preview_window = _add_virtual_detector(qtbot, win)
        preview_plot = preview_window.plots[0]

        preview_window.title_bar.compute_button.click()
        qtbot.waitUntil(
            lambda: (
                preview_plot.current_data is not None
                and not isinstance(preview_plot.current_data, Future)
            ),
            timeout=10000,
        )
        return caret_box, roi, preview_plot, preview_window

    def test_commit_button_disabled_before_first_computation(self, qtbot, stem_4d_dataset):
        """Commit button in title bar must be disabled immediately after adding a detector."""
        win = stem_4d_dataset["window"]
        tb, vi_widget, action_name, caret_box, roi, preview_window = _add_virtual_detector(qtbot, win)
        assert not preview_window.title_bar.commit_button.isEnabled(), (
            "Commit button should be disabled before any computation"
        )

    def test_commit_button_enabled_after_preview_completes(self, qtbot, stem_4d_dataset):
        """Commit button must become enabled once the first preview computation finishes."""
        win = stem_4d_dataset["window"]
        caret_box, roi, preview_plot, preview_window = self._setup_with_preview(qtbot, win)
        assert preview_window.title_bar.commit_button.isEnabled(), (
            "Commit button should be enabled after preview computation"
        )

    def test_commit_adds_new_signal_tree(self, qtbot, stem_4d_dataset):
        """Clicking title-bar Commit must add exactly one new root to main_window.signal_trees."""
        win = stem_4d_dataset["window"]
        n_before = len(win.signal_trees)
        caret_box, roi, preview_plot, preview_window = self._setup_with_preview(qtbot, win)

        preview_window.title_bar.commit_button.click()

        qtbot.waitUntil(lambda: len(win.signal_trees) == n_before + 1, timeout=10000)
        assert len(win.signal_trees) == n_before + 1

    def test_committed_signal_is_virtual_dark_field(self, qtbot, stem_4d_dataset):
        """The committed signal tree root must be a VirtualDarkFieldImage."""
        from pyxem.signals import VirtualDarkFieldImage
        win = stem_4d_dataset["window"]
        n_before = len(win.signal_trees)
        caret_box, roi, preview_plot, preview_window = self._setup_with_preview(qtbot, win)

        preview_window.title_bar.commit_button.click()

        qtbot.waitUntil(lambda: len(win.signal_trees) == n_before + 1, timeout=10000)
        new_signal = win.signal_trees[n_before].root
        assert isinstance(new_signal, VirtualDarkFieldImage), (
            f"Expected VirtualDarkFieldImage, got {type(new_signal)}"
        )

    def test_committed_signal_axes_match_parent_nav(self, qtbot, stem_4d_dataset):
        """The VDF signal axes must carry the scale/offset of the source navigation axes."""
        win = stem_4d_dataset["window"]
        nav, sig = win.plots
        source_signal = sig.plot_state.current_signal

        n_before = len(win.signal_trees)
        caret_box, roi, preview_plot, preview_window = self._setup_with_preview(qtbot, win)

        preview_window.title_bar.commit_button.click()

        qtbot.waitUntil(lambda: len(win.signal_trees) == n_before + 1, timeout=10000)
        vdf = win.signal_trees[n_before].root

        src_nav = list(source_signal.axes_manager.navigation_axes)
        vdf_sig = list(vdf.axes_manager.signal_axes)
        assert len(vdf_sig) == len(src_nav)
        for i, src_ax in enumerate(src_nav):
            vdf_ax = vdf_sig[i]
            assert abs(vdf_ax.scale - src_ax.scale) < 1e-9
            assert abs(vdf_ax.offset - src_ax.offset) < 1e-9

    def test_committed_signal_has_roi_metadata(self, qtbot, stem_4d_dataset):
        """The committed VDF must carry ROI geometry in metadata.Signal.virtual_detector."""
        win = stem_4d_dataset["window"]
        n_before = len(win.signal_trees)
        caret_box, roi, preview_plot, preview_window = self._setup_with_preview(qtbot, win)

        preview_window.title_bar.commit_button.click()

        qtbot.waitUntil(lambda: len(win.signal_trees) == n_before + 1, timeout=10000)
        vdf = win.signal_trees[n_before].root

        assert vdf.metadata.Signal.virtual_detector is not None
        assert "type" in vdf.metadata.Signal.virtual_detector

    def test_preview_window_remains_open_after_commit(self, qtbot, stem_4d_dataset):
        """The live preview PlotWindow must stay open after committing."""
        win = stem_4d_dataset["window"]
        caret_box, roi, preview_plot, preview_window = self._setup_with_preview(qtbot, win)

        n_trees = len(win.signal_trees)
        preview_window.title_bar.commit_button.click()
        qtbot.waitUntil(lambda: len(win.signal_trees) > n_trees - 1, timeout=10000)
        qtbot.wait(500)

        assert preview_window.isVisible()

    def test_two_commits_produce_two_independent_trees(self, qtbot, stem_4d_dataset):
        """Committing twice must produce two independent signal trees."""
        win = stem_4d_dataset["window"]
        n_before = len(win.signal_trees)
        caret_box, roi, preview_plot, preview_window = self._setup_with_preview(qtbot, win)

        preview_window.title_bar.commit_button.click()
        qtbot.waitUntil(lambda: len(win.signal_trees) == n_before + 1, timeout=10000)
        qtbot.waitUntil(lambda: preview_window.title_bar.commit_button.isEnabled(), timeout=5000)

        preview_window.title_bar.commit_button.click()
        qtbot.waitUntil(lambda: len(win.signal_trees) == n_before + 2, timeout=10000)

        tree1 = win.signal_trees[n_before]
        tree2 = win.signal_trees[n_before + 1]
        assert tree1 is not tree2
        assert tree1.root is not tree2.root

    def test_closing_preview_removes_roi_from_signal_plot(self, qtbot, stem_4d_dataset):
        """Closing the virtual image preview window must remove the ROI from the diffraction plot."""
        win = stem_4d_dataset["window"]
        nav, sig = win.plots
        tb, vi_widget, action_name, caret_box, roi, preview_window = _add_virtual_detector(qtbot, win)

        assert roi in sig.items, "ROI should be on signal plot before close"

        preview_window.close()
        qtbot.wait(200)

        assert roi not in sig.items, "ROI should be removed from signal plot after preview window closes"

    def test_closing_preview_removes_action_from_toolbar(self, qtbot, stem_4d_dataset):
        """Closing the virtual image preview window must remove its action from the Virtual Imaging toolbar."""
        win = stem_4d_dataset["window"]
        tb, vi_widget, action_name, caret_box, roi, preview_window = _add_virtual_detector(qtbot, win)

        action_names_before = [a.text() for a in vi_widget.actions()]
        assert action_name in action_names_before, "Action should exist in toolbar before close"

        preview_window.close()
        qtbot.wait(200)

        action_names_after = [a.text() for a in vi_widget.actions()]
        assert action_name not in action_names_after, (
            "Action should be removed from Virtual Imaging toolbar after preview window closes"
        )


def test_action_preview_stays_hidden_when_action_unchecked(qtbot, stem_4d_dataset):
    """An action-preview window with controlling_action unchecked must stay hidden after activation."""
    from PySide6.QtWidgets import QApplication
    win = stem_4d_dataset["window"]
    nav_window = stem_4d_dataset["subwindows"][0]
    win.mdi_area.setActiveSubWindow(nav_window)
    qtbot.wait(100)

    # Add a virtual image selector to get an action-preview window with a controlling_action
    nav, sig = win.plots
    tb = nav.plot_state.toolbar_top
    # Find a checkable action that registers a plot window (e.g. virtual imaging)
    target_action = None
    for a in tb.actions():
        if a.isCheckable():
            target_action = a
            break
    if target_action is None:
        pytest.skip("No checkable action found on toolbar_top")

    # Check the action to open its preview window
    n_before = len(win.plot_subwindows)
    target_action.setChecked(True)
    QApplication.processEvents()
    qtbot.wait(100)

    action_windows = [pw for pw in win.plot_subwindows[n_before:]
                      if getattr(pw, 'controlling_action', None) is target_action]
    if not action_windows:
        pytest.skip("No preview window registered with controlling_action for this action")
    preview = action_windows[0]
    assert preview.isVisible(), "Preview should be visible when action is checked"

    # Uncheck the action — window should hide
    target_action.setChecked(False)
    QApplication.processEvents()
    qtbot.wait(50)
    assert not preview.isVisible(), "Preview should hide when action is unchecked"

    # Trigger on_subwindow_activated — must NOT re-show the preview
    win.on_subwindow_activated(nav_window)
    assert not preview.isVisible(), (
        "Preview must stay hidden after on_subwindow_activated when controlling_action is unchecked"
    )


@pytest.mark.gpu
class TestVirtualImageKernelGPU:

    @pytest.fixture(autouse=True)
    def skip_if_no_gpu(self, gpu_available):
        if not gpu_available:
            pytest.skip("No NVIDIA GPU detected")

    @pytest.fixture(autouse=True)
    def client(self, stem_4d_dataset):
        self.win = stem_4d_dataset["window"]
        self.client = self.win.dask_manager.client
        self.gpu_address = self.win.dask_manager.gpu_worker_address

    def _mask(self):
        mask = np.zeros((8, 8), dtype=np.float32)
        mask[2:6, 2:6] = 1.0
        return mask

    def test_4d_gpu_matches_cpu(self):
        from spyde.drawing.update_functions import compute_virtual_image_kernel
        rng = np.random.default_rng(1)
        data_np = rng.random((4, 4, 8, 8)).astype(np.float32)
        mask = self._mask()
        data = da.from_array(data_np, chunks=(2, 2, 8, 8))

        cpu_result = compute_virtual_image_kernel(data, mask, self.client, None).result()
        gpu_result = compute_virtual_image_kernel(data, mask, self.client, self.gpu_address).result()

        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-4)

    def test_5d_gpu_matches_cpu(self):
        from spyde.drawing.update_functions import compute_virtual_image_kernel
        rng = np.random.default_rng(2)
        data_np = rng.random((2, 4, 4, 8, 8)).astype(np.float32)
        mask = self._mask()
        data = da.from_array(data_np, chunks=(1, 2, 2, 8, 8))

        cpu_result = compute_virtual_image_kernel(data, mask, self.client, None).result()
        gpu_result = compute_virtual_image_kernel(data, mask, self.client, self.gpu_address).result()

        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-4)
        assert gpu_result.shape == (2, 4, 4)
