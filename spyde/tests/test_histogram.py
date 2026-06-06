"""Tests for histogram widget behaviour across plot switches."""


class TestHistogram:

    def test_histogram_resets_range_on_plot_switch(self, qtbot, stem_4d_dataset):
        """Zooming the histogram axis then switching to another plot must reset the range."""
        win = stem_4d_dataset["window"]
        nav_window, sig_window = win.plot_subwindows

        # Activate nav plot so the histogram is bound to it.
        # Clear the cached item so on_subwindow_activated always rebinds.
        win._histogram_image_item = None
        win.mdi_area.setActiveSubWindow(nav_window)
        qtbot.wait(300)

        hist_item = win.histogram.item

        # Record the natural auto-range for the nav image
        natural_range = hist_item.getHistogramRange()
        natural_span = natural_range[1] - natural_range[0]
        assert natural_span > 0, "Histogram has no data range after activation"

        # Simulate the user zooming into the histogram by setting a very narrow range
        zoomed_mn = natural_range[0] + natural_span * 0.4
        zoomed_mx = natural_range[0] + natural_span * 0.6
        hist_item.setHistogramRange(zoomed_mn, zoomed_mx)
        qtbot.wait(100)

        zoomed_range = hist_item.getHistogramRange()
        zoomed_span = zoomed_range[1] - zoomed_range[0]
        assert zoomed_span < natural_span * 0.5, "setHistogramRange did not narrow the view"

        # Now activate a different plot window — this should reset the histogram range.
        # Clear cache so the switch always triggers rebind + range reset.
        win._histogram_image_item = None
        win.mdi_area.setActiveSubWindow(sig_window)
        qtbot.wait(300)

        reset_range = hist_item.getHistogramRange()
        reset_span = reset_range[1] - reset_range[0]

        assert reset_span > zoomed_span * 1.5, (
            f"Histogram range was not reset after switching plots: "
            f"zoomed span={zoomed_span:.4g}, reset span={reset_span:.4g}"
        )

    def test_histogram_bound_to_active_plot(self, qtbot, stem_4d_dataset):
        """After switching plots the histogram must be bound to the new plot's image item."""
        win = stem_4d_dataset["window"]
        nav_window, sig_window = win.plot_subwindows
        nav_plot = nav_window.current_plot_item
        sig_plot = sig_window.current_plot_item

        win.mdi_area.setActiveSubWindow(nav_window)
        qtbot.wait(300)
        assert win._histogram_image_item is nav_plot.image_item, (
            "Histogram not bound to nav plot image item"
        )

        win.mdi_area.setActiveSubWindow(sig_window)
        qtbot.wait(300)
        assert win._histogram_image_item is sig_plot.image_item, (
            "Histogram not re-bound to sig plot image item after switch"
        )
