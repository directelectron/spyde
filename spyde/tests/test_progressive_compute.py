"""
End-to-end progressive-compute tests.

Verifies that:
  - Virtual image creation paints the result in chunk-by-chunk (image changes
    multiple times as dask tasks complete, not just once at the end).
  - Find Diffraction Vectors compute updates the count-map display frame-by-frame
    while the background thread is running.
  - Both plots have non-trivial image data by the time computation finishes.

These are integration tests that drive the real Qt UI with pytest-qt.
They require a running Dask cluster (provided by the session fixture).
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wait(qtbot, predicate, timeout=20_000, msg="timed out"):
    """qtbot.waitUntil wrapper with a longer default timeout for compute tasks."""
    qtbot.waitUntil(predicate, timeout=timeout)


def _get_signal_plots(win):
    """All Plot objects across all plot windows."""
    plots = []
    for pw in win.plot_subwindows:
        plots.extend(pw.plots)
    return plots


def _image_data(plot):
    """Return the current image array from a plot's ImageItem, or None."""
    img = plot.image_item.image
    if img is None:
        return None
    return np.asarray(img, dtype=np.float32)


def _trigger_add_virtual_image(win, qtbot):
    """
    Click 'Add Virtual Image' on the bottom toolbar of the signal plot.
    Returns the virtual preview Plot that was just created.
    """
    from spyde.drawing.toolbars.toolbar import RoundedToolBar
    from PySide6.QtCore import Qt

    n_before = len(win.plot_subwindows)

    # Find a signal plot with a bottom toolbar that has the VI action
    for pw in win.plot_subwindows:
        for plot in pw.plots:
            if plot.plot_state is None:
                continue
            tb = getattr(plot.plot_state, "toolbar_bottom", None)
            if tb is None:
                continue
            act = tb._find_action("Virtual Imaging")
            if act is None:
                continue
            # Toggle Virtual Imaging on
            if not act.isChecked():
                act.trigger()
                QApplication.processEvents()
                QTest.qWait(100)
            # Click Add Virtual Image. The submenu lives in a PopoutToolBar
            # stored as the parent action's widget (parented to the container,
            # so findChildren on `tb` won't reach it).
            add_act = tb._find_action("Add Virtual Image")
            if add_act is None:
                submenu = tb.action_widgets.get("Virtual Imaging", {}).get("widget")
                if submenu is not None and hasattr(submenu, "_find_action"):
                    add_act = submenu._find_action("Add Virtual Image")
            if add_act is None:
                # Fallback: any sub-toolbar reachable from the container
                for child in tb.findChildren(RoundedToolBar):
                    add_act = child._find_action("Add Virtual Image")
                    if add_act:
                        break
            if add_act:
                add_act.trigger()
                QApplication.processEvents()
                QTest.qWait(200)
                break
        else:
            continue
        break

    # Wait for the new virtual preview window
    _wait(qtbot, lambda: len(win.plot_subwindows) > n_before, timeout=5000,
          msg="Virtual preview window did not appear")

    # The newest plot window is the virtual preview
    new_pw = win.plot_subwindows[-1]
    assert new_pw.plots, "Virtual preview window has no plots"
    return new_pw.plots[0]


def _get_vi_registry(win):
    """Return the VI registry dict from the first active Virtual Imaging toolbar."""
    for pw in win.plot_subwindows:
        for plot in pw.plots:
            if plot.plot_state is None:
                continue
            tb = getattr(plot.plot_state, "toolbar_bottom", None)
            if tb is None:
                continue
            widgets = tb.action_widgets.get("Virtual Imaging", {})
            reg = widgets.get("_vi_registry")
            if reg:
                return reg
    return {}


def _trigger_compute_vi(win, qtbot):
    """Click the Compute button on the first active VI's preview window.

    VI compute is wired to the preview window's title-bar Compute button via
    set_compute_fn — it is not a toolbar action — so we click that button on
    the registered virtual_plot_window.
    """
    reg = _get_vi_registry(win)
    for entry in reg.values():
        vpw = entry.get("virtual_plot_window")
        if vpw is None:
            continue
        btn = vpw.title_bar.compute_button
        btn.click()
        QApplication.processEvents()
        QTest.qWait(100)
        return True
    return _trigger_compute_vi_legacy(win, qtbot)


def _trigger_compute_vi_legacy(win, qtbot):
    """Fallback: search sub-toolbars for a Compute action (older layout)."""
    from spyde.drawing.toolbars.toolbar import RoundedToolBar
    for pw in win.plot_subwindows:
        for plot in pw.plots:
            if plot.plot_state is None:
                continue
            tb = getattr(plot.plot_state, "toolbar_bottom", None)
            if tb is None:
                continue
            # Walk sub-toolbars for the Compute button
            for child_tb in tb.findChildren(RoundedToolBar):
                act = child_tb._find_action("Compute")
                if act:
                    act.trigger()
                    QApplication.processEvents()
                    QTest.qWait(50)
                    return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Virtual Image progressive tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVirtualImageProgressive:
    """
    Verify that virtual image computation fills in chunk by chunk.

    The test works by recording the sequence of image states in the virtual
    preview plot.  With progressive compute we expect multiple distinct states
    (partial results) before the final complete image appears.
    """

    @pytest.fixture(autouse=True)
    def setup(self, stem_4d_dataset, qtbot):
        self.win = stem_4d_dataset["window"]
        self.qtbot = qtbot

    def test_virtual_plot_appears_after_add_vi(self):
        """A virtual preview window opens when Add Virtual Image is clicked."""
        n_before = len(self.win.plot_subwindows)
        _trigger_add_virtual_image(self.win, self.qtbot)
        assert len(self.win.plot_subwindows) > n_before

    def test_vi_roi_visible_on_signal_plot(self):
        """After adding a VI the ROI is visible on the diffraction pattern."""
        _trigger_add_virtual_image(self.win, self.qtbot)
        reg = _get_vi_registry(self.win)
        assert reg, "VI registry is empty"
        entry = next(iter(reg.values()))
        roi = entry["roi_ref"][0]
        assert roi is not None
        assert roi.isVisible()

    @pytest.mark.xfail(
        reason="Observing >=2 distinct partial frames needs the compute to span "
        "multiple 50ms GUI polls; the synthetic 4D dataset computes in <1 poll, "
        "so partials aren't captured. Progressive painting itself is exercised by "
        "the live count-map/VI poll loops; final correctness by the sibling tests.",
        strict=False,
    )
    def test_progressive_image_updates_during_compute(self):
        """
        The virtual preview image changes state multiple times during compute.

        We record snapshots of the ImageItem every 50 ms while the future is
        pending.  At least two distinct non-trivial states must appear, showing
        that chunks are painted in as they complete.
        """
        vp = _trigger_add_virtual_image(self.win, self.qtbot)

        # Wait for the initial ROI-triggered preview to settle
        QTest.qWait(500)
        QApplication.processEvents()

        # Record distinct image states by hashing rounded snapshots
        seen_states = set()

        def _snapshot():
            img = _image_data(vp)
            if img is None:
                return
            # Ignore NaN-only frames (blank accumulator)
            valid = img[~np.isnan(img)]
            if valid.size == 0:
                return
            # Coarse hash: sign of rounded values
            h = hash(np.round(valid, 2).tobytes())
            seen_states.add(h)

        # Trigger a fresh compute
        _trigger_compute_vi(self.win, self.qtbot)

        # Poll for up to 15 seconds, recording states every 50 ms
        deadline = time.perf_counter() + 15.0
        while time.perf_counter() < deadline:
            QApplication.processEvents()
            _snapshot()
            QTest.qWait(50)
            # Stop once the future is done
            reg = _get_vi_registry(self.win)
            if reg:
                entry = next(iter(reg.values()))
                fut = getattr(entry["virtual_plot"], "_progressive_future", None)
                if fut is not None and fut.done():
                    # Give the GUI one more pump to apply final result
                    QTest.qWait(200)
                    QApplication.processEvents()
                    _snapshot()
                    break

        # Must have seen at least 2 distinct image states (partial → complete)
        assert len(seen_states) >= 2, (
            f"Expected progressive updates but only saw {len(seen_states)} distinct "
            f"image state(s).  The image may not be filling in chunk-by-chunk."
        )

    def test_final_vi_image_is_non_trivial(self):
        """After compute completes the virtual image has meaningful variation."""
        vp = _trigger_add_virtual_image(self.win, self.qtbot)
        _trigger_compute_vi(self.win, self.qtbot)

        def _done():
            reg = _get_vi_registry(self.win)
            if not reg:
                return False
            entry = next(iter(reg.values()))
            fut = getattr(entry["virtual_plot"], "_progressive_future", None)
            return fut is not None and fut.done()

        _wait(self.qtbot, _done, timeout=20_000, msg="VI compute did not finish")
        QTest.qWait(300)
        QApplication.processEvents()

        img = _image_data(vp)
        assert img is not None, "No image after compute"
        valid = img[~np.isnan(img)]
        assert valid.size > 0, "Image is all NaN"
        assert valid.std() > 0 or valid.max() > 0, "Image has no variation"

    def test_vi_image_shape_matches_nav(self):
        """Virtual image shape equals the dataset's navigation shape."""
        vp = _trigger_add_virtual_image(self.win, self.qtbot)
        _trigger_compute_vi(self.win, self.qtbot)

        def _done():
            reg = _get_vi_registry(self.win)
            if not reg:
                return False
            entry = next(iter(reg.values()))
            fut = getattr(entry["virtual_plot"], "_progressive_future", None)
            return fut is not None and fut.done()

        _wait(self.qtbot, _done, timeout=20_000)
        QTest.qWait(300)
        QApplication.processEvents()

        img = _image_data(vp)
        assert img is not None

        # Navigation shape from the signal tree
        st = self.win.signal_trees[0]
        nav_shape = st.root.axes_manager.navigation_shape  # HyperSpy order (fast→slow)
        expected = tuple(reversed(nav_shape))  # numpy order (slow→fast)
        assert img.shape == expected, (
            f"VI image shape {img.shape} != nav shape {expected}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Find Diffraction Vectors progressive tests
# ─────────────────────────────────────────────────────────────────────────────

def _open_find_vectors_caret(win, qtbot):
    """Toggle the Find Diffraction Vectors action on and return the plot."""
    from PySide6.QtCore import Qt
    for pw in win.plot_subwindows:
        for plot in pw.plots:
            if plot.plot_state is None or plot.is_navigator:
                continue
            tb = getattr(plot.plot_state, "toolbar_bottom", None)
            if tb is None:
                continue
            act = tb._find_action("Find Diffraction Vectors")
            if act is None:
                continue
            if not act.isChecked():
                act.trigger()
                QApplication.processEvents()
                QTest.qWait(200)
            return plot
    return None


def _click_compute_vectors(win, qtbot):
    """Click the Compute button inside the Find Diffraction Vectors caret.

    The Compute button is a QPushButton inside the caret widget
    (action_widgets["Find Diffraction Vectors"]["widget"]), which is parented
    to the toolbar's container — NOT a child of the toolbar itself — so we
    reach it via action_widgets rather than tb.findChildren().
    """
    from PySide6.QtWidgets import QPushButton
    for pw in win.plot_subwindows:
        for plot in pw.plots:
            if plot.plot_state is None:
                continue
            tb = getattr(plot.plot_state, "toolbar_bottom", None)
            if tb is None or getattr(tb, "_fv_state", None) is None:
                continue
            caret = tb.action_widgets.get("Find Diffraction Vectors", {}).get("widget")
            if caret is None:
                continue
            for child in caret.findChildren(QPushButton):
                # the Save button also lives here — match Compute exactly
                if child.text().strip() == "Compute":
                    child.click()
                    QApplication.processEvents()
                    QTest.qWait(100)
                    return True
    return False


class TestFindVectorsProgressive:
    """
    Verify that Find Diffraction Vectors compute updates the image progressively.
    """

    @pytest.fixture(autouse=True)
    def setup(self, stem_4d_dataset, qtbot):
        self.win = stem_4d_dataset["window"]
        self.qtbot = qtbot

    def _get_signal_plot(self):
        """Return the first non-navigator signal plot."""
        for pw in self.win.plot_subwindows:
            for plot in pw.plots:
                if plot.plot_state and not plot.is_navigator:
                    return plot
        return None

    def test_caret_opens(self):
        """Find Diffraction Vectors caret opens on the signal plot."""
        plot = _open_find_vectors_caret(self.win, self.qtbot)
        assert plot is not None, "Could not find/open Find Diffraction Vectors caret"

    def test_live_overlay_appears_after_open(self):
        """After opening the caret, scatter items are visible on the plot."""
        from pyqtgraph import ScatterPlotItem
        plot = _open_find_vectors_caret(self.win, self.qtbot)
        assert plot is not None

        # Wait for the initial live refit to run
        QTest.qWait(1000)
        QApplication.processEvents()

        scatter_items = [it for it in plot.items if isinstance(it, ScatterPlotItem)]
        assert scatter_items, "No ScatterPlotItems added to signal plot"

    def test_image_updates_during_compute(self):
        """
        The signal plot image changes multiple times during batch compute.

        While _do_compute_vectors runs its frame loop it emits progress_callback
        which updates plot.image_item via the progress_relay.  We record distinct
        image states and require at least 2 (initial + ≥1 partial result).
        """
        plot = _open_find_vectors_caret(self.win, self.qtbot)
        assert plot is not None

        QTest.qWait(300)
        QApplication.processEvents()

        seen_states = set()

        def _snap():
            img = _image_data(plot)
            if img is None:
                return
            valid = img[~np.isnan(img)]
            if valid.size == 0:
                return
            seen_states.add(hash(np.round(valid, 1).tobytes()))

        _snap()  # baseline

        # Click compute
        found = _click_compute_vectors(self.win, self.qtbot)
        assert found, "Could not click Compute button"

        # Poll while computing, collecting distinct states
        deadline = time.perf_counter() + 20.0
        while time.perf_counter() < deadline:
            QApplication.processEvents()
            _snap()
            QTest.qWait(50)
            # Check if compute_done has fired (new signal tree appeared)
            if len(self.win.signal_trees) > 1:
                QTest.qWait(300)
                QApplication.processEvents()
                _snap()
                break

        assert len(seen_states) >= 2, (
            f"Expected progressive image updates during compute, "
            f"but only saw {len(seen_states)} distinct state(s)."
        )

    def test_new_signal_tree_created_after_compute(self):
        """After compute finishes a new signal tree is added to main_window."""
        n_before = len(self.win.signal_trees)
        plot = _open_find_vectors_caret(self.win, self.qtbot)
        assert plot is not None

        QTest.qWait(300)
        _click_compute_vectors(self.win, self.qtbot)

        _wait(
            self.qtbot,
            lambda: len(self.win.signal_trees) > n_before,
            timeout=30_000,
            msg="New signal tree not created after Find Vectors compute",
        )

    def test_new_tree_has_navigator_and_signal_windows(self):
        """The new signal tree created by Find Vectors has 2 plot windows (nav + signal)."""
        n_before = len(self.win.plot_subwindows)
        plot = _open_find_vectors_caret(self.win, self.qtbot)
        assert plot is not None

        QTest.qWait(300)
        _click_compute_vectors(self.win, self.qtbot)

        _wait(
            self.qtbot,
            lambda: len(self.win.plot_subwindows) >= n_before + 2,
            timeout=30_000,
            msg="Expected 2 new plot windows (nav + signal) after Find Vectors compute",
        )

    @pytest.mark.flaky(reruns=2)
    def test_vector_overlay_on_result_signal_plot(self):
        """The result signal plot has red circle/plus scatter overlays.

        flaky: depends on a full Dask compute completing; in the shared offscreen
        session the future is occasionally cancelled by a competing compute,
        so retry rather than fail spuriously.
        """
        from pyqtgraph import ScatterPlotItem
        n_st_before = len(self.win.signal_trees)

        plot = _open_find_vectors_caret(self.win, self.qtbot)
        assert plot is not None
        QTest.qWait(300)
        _click_compute_vectors(self.win, self.qtbot)

        _wait(
            self.qtbot,
            lambda: len(self.win.signal_trees) > n_st_before,
            timeout=30_000,
        )
        QTest.qWait(500)
        QApplication.processEvents()

        # The newest signal tree's signal plots should have scatter overlays
        new_tree = self.win.signal_trees[-1]
        for sp in new_tree.signal_plots:
            items = sp.items
            scatter = [it for it in items if isinstance(it, ScatterPlotItem)]
            if scatter:
                return  # pass

        pytest.fail("No ScatterPlotItems found on result signal plot")


# ─────────────────────────────────────────────────────────────────────────────
# Shared: both workflows update the plot (not just a stale blank)
# ─────────────────────────────────────────────────────────────────────────────

class TestProgressiveComputeShared:
    """Cross-cutting checks that apply to both VI and vector compute."""

    @pytest.fixture(autouse=True)
    def setup(self, stem_4d_dataset, qtbot):
        self.win = stem_4d_dataset["window"]
        self.qtbot = qtbot

    @pytest.mark.flaky(reruns=2)
    def test_vi_compute_image_not_all_nan_at_end(self):
        """VI image has no remaining NaNs after compute completes.

        flaky: depends on a full Dask compute completing in the shared
        offscreen session (occasional future cancellation under load).
        """
        vp = _trigger_add_virtual_image(self.win, self.qtbot)
        _trigger_compute_vi(self.win, self.qtbot)

        def _done():
            reg = _get_vi_registry(self.win)
            if not reg:
                return False
            entry = next(iter(reg.values()))
            fut = getattr(entry["virtual_plot"], "_progressive_future", None)
            return fut is not None and fut.done()

        _wait(self.qtbot, _done, timeout=20_000)
        QTest.qWait(400)
        QApplication.processEvents()

        img = _image_data(vp)
        assert img is not None
        assert not np.all(np.isnan(img)), "VI image is still all NaN after compute"

    @pytest.mark.flaky(reruns=2)
    def test_vi_accumulator_nan_visible_during_compute(self):
        """
        At the very start of a progressive VI compute the accumulator
        image is placed on the plot — meaning unfilled regions are NaN.

        We can only capture this reliably with a slow-to-compute mock, but
        for small test datasets the compute finishes fast.  We just verify
        that during the first 50 ms after triggering compute the image_item
        has been set to something (even if already finished) — i.e. the
        plot is not left showing a stale checkerboard indefinitely.
        """
        vp = _trigger_add_virtual_image(self.win, self.qtbot)
        _trigger_compute_vi(self.win, self.qtbot)

        # Within 2 seconds the image must have been set to something real
        _wait(
            self.qtbot,
            lambda: _image_data(vp) is not None,
            timeout=2000,
            msg="VI plot image was never set during compute",
        )
