"""An index naming several navigation positions is READ as several positions.

A selector reports the mode of its own widget. In two cases the index it
composes names more positions than that mode admits to:

* a chained navigator whose OUTER selector is an integrating span — the
  innermost selector composes the cartesian product of every selector above it,
  so a crosshair on a 5-D stack whose angle span covers N angles reports N rows
  while calling itself a point;
* a 1-D point selector given a width (``sum_frames``) — one pointer, n index
  rows, which is what the width means and what a region selector emits.

Reducing either to a mean index truncates (``int(0.5) == 0``) and displays one
of the selected positions with no error, so the span integrates one angle and a
requested exposure of n frames shows one frame.
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs

from spyde.drawing.selectors.base_selector import _nav_dispatcher

ANGLES, SCAN_ROWS, SCAN_COLUMNS = 4, 4, 4
DETECTOR_ROWS, DETECTOR_COLUMNS = 8, 8
MOVIE_FRAMES = 6

SPAN_FIRST_ANGLE, SPAN_LAST_ANGLE = 1, 2
SCAN_ROW, SCAN_COLUMN = 3, 2


def _stack():
    """``(angle, y, x | ky, kx)`` with values that make a wrong angle visible."""
    data = np.random.default_rng(0).random(
        (ANGLES, SCAN_ROWS, SCAN_COLUMNS, DETECTOR_ROWS, DETECTOR_COLUMNS)
    ).astype(np.float32)
    signal = hs.signals.Signal2D(data)
    signal.set_signal_type("electron_diffraction")
    return signal


def _movie():
    """``(time | y, x)`` — the in-situ movie a width widens the exposure of."""
    data = np.random.default_rng(1).random(
        (MOVIE_FRAMES, DETECTOR_ROWS, DETECTOR_COLUMNS)).astype(np.float32)
    return hs.signals.Signal2D(data)


def _settle_navigator(timeout: float = 5.0) -> bool:
    """Wait until the serial navigator dispatcher has run every queued update.

    Idle has to hold TWICE: a chained update is submitted from inside the update
    it follows, so a single sample can land in the gap between the two.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _nav_dispatcher.idle():
            time.sleep(0.05)
            if _nav_dispatcher.idle():
                return True
        time.sleep(0.01)
    return _nav_dispatcher.idle()


def _signal_plot(session):
    return next(plot for plot in session._plots
                if not plot.is_navigator and plot.plot_state is not None)


def _navigation_selectors(tree):
    """``(angle_selector, scan_selector)`` of a 5-D chain, outermost first."""
    by_name = {type(selector).__name__: selector
               for selector in tree.navigator_plot_manager.all_navigation_selectors}
    return by_name["IntegratingSelector1D"], by_name["IntegratingSSelector2D"]


class TestOuterSpanIntegrates:
    """The outer navigator of a chain is a span; the inner one is a crosshair."""

    def _open(self, session):
        signal = _stack()
        session._add_signal(signal, source_path=None)
        tree = session.signal_trees[0]

        angle_selector, scan_selector = _navigation_selectors(tree)
        # Report fixed widget positions so the expected frame is a known array:
        # the angle selector integrates a span of two angles, the scan selector
        # stays a crosshair (it reports widget order — column, row).
        angle_selector.set_integrating(True)
        angle_selector.selector._get_selected_indices = lambda: np.array(
            [[angle] for angle in range(SPAN_FIRST_ANGLE, SPAN_LAST_ANGLE + 1)])
        scan_selector.selector._get_selected_indices = \
            lambda: np.array([[SCAN_COLUMN, SCAN_ROW]])
        scan_selector.selector._run_update(force=True)
        _settle_navigator()
        return signal, _signal_plot(session)

    def test_a_span_of_two_angles_shows_both(self, window):
        signal, signal_plot = self._open(window["window"])

        expected = signal.data[SPAN_FIRST_ANGLE:SPAN_LAST_ANGLE + 1,
                               SCAN_ROW, SCAN_COLUMN].mean(axis=0)
        assert signal_plot.current_data.shape == (DETECTOR_ROWS, DETECTOR_COLUMNS)
        assert np.allclose(signal_plot.current_data, expected)

    def test_the_span_is_not_truncated_to_its_first_angle(self, window):
        """The failure this pins is silent, so name it: a mean of the span's
        index columns rounds down to the span's first angle."""
        signal, signal_plot = self._open(window["window"])

        first_angle_only = signal.data[SPAN_FIRST_ANGLE, SCAN_ROW, SCAN_COLUMN]
        assert not np.allclose(signal_plot.current_data, first_angle_only)


class TestPointWidthIntegrates:
    """A 1-D point selector with ``sum_frames`` set reads a window of frames."""

    def _open(self, session, width):
        signal = _movie()
        session._add_signal(signal, source_path=None)
        tree = session.signal_trees[0]
        selector = tree.navigator_plot_manager.all_navigation_selectors[0]
        selector.sum_frames = width
        selector.selector._run_update(force=True)
        _settle_navigator()
        return signal, _signal_plot(session)

    def test_a_width_of_one_reads_one_frame(self, window):
        """The baseline: a plain pointer still shows a single raw frame."""
        signal, signal_plot = self._open(window["window"], width=1)

        assert signal_plot.current_data.shape == (DETECTOR_ROWS, DETECTOR_COLUMNS)
        assert any(np.allclose(signal_plot.current_data, frame)
                   for frame in signal.data)

    def test_a_width_of_four_reads_four_frames(self, window):
        signal, signal_plot = self._open(window["window"], width=4)

        # Whichever four consecutive frames the pointer's window lands on, the
        # displayed frame is their mean — and never one frame on its own.
        windows = [signal.data[start:start + 4].mean(axis=0)
                   for start in range(MOVIE_FRAMES - 3)]
        assert any(np.allclose(signal_plot.current_data, mean)
                   for mean in windows)
        assert not any(np.allclose(signal_plot.current_data, frame)
                       for frame in signal.data)
