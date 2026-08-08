"""
The "second-signal IndexError", console flavour — a navigator update must never
index a signal whose ``.data`` is hyperspy's transient deepcopy placeholder.

Mechanism (found via console_preview.spec.ts CI evidence: ``NAV-DEBUG eager
index RAISED: indices=[0, 0] data.shape=(1,) nav_shape=(6, 6)``): EVERY
hyperspy signal operation (arithmetic, comparison, ``sum``, ``deepcopy``) goes
through ``BaseSignal._deepcopy_with_new_data``, which TRANSIENTLY rebinds
``self.data = None`` on the live signal object while it deep-copies — and the
data setter's ``np.atleast_1d(np.asanyarray(None))`` turns that into an
``array([None], dtype=object)`` of shape ``(1,)``. The math console evaluates
user expressions (``s1 + 0`` — the eye-toggle live preview, re-run on every
nav commit via ``NAV_CHANGE_HOOKS``) against the SAME bound root-signal objects
on the console thread, so a navigator update on the serial ``_NavDispatcher``
thread can land inside that window: the signal still reports its real
nav_shape, but its data is the 1-element placeholder → "too many indices for
array: array is 1-dimensional, but 2 were indexed". ``_pending_future_data``
cannot catch it (``data[0]`` is None, not a future).

The fix (``update_functions``): capture ``.data`` ONCE per read, skip the frame
when the captured binding cannot satisfy the nav indices (``_nav_readable_data``
— the last good frame stays up, exactly like the pending-future skip), and
clamp + index that SAME captured reference so a post-capture swap still reads
the coherent pre-swap array. No locks, no generation counters — the serial
dispatcher model is untouched.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

from spyde.drawing import update_functions as uf
from spyde.drawing.selectors.base_selector import _nav_dispatcher


def _nav_error_records(caplog):
    """ERROR+ records from the nav-read module (the spec's failure signal)."""
    return [r for r in caplog.records
            if r.name == "spyde.drawing.update_functions"
            and r.levelno >= logging.ERROR]


def _wait_dispatcher_idle(timeout: float = 3.0) -> None:
    """Wait until the serial dispatcher has drained its pending queue (from
    test_navigator_race.py)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        with _nav_dispatcher._lock:
            empty = not _nav_dispatcher._pending
        if empty:
            time.sleep(0.05)   # let the in-flight job finish
            return
        time.sleep(0.01)


class TestNavSecondSignalRace:
    """Pin: a shape-(1,) placeholder on a 2-D-navigated signal is SKIPPED (no
    ERROR, no exception, last good frame stays) and the read recovers."""

    @staticmethod
    def _sel_and_child(session):
        """The tree's navigation selector (composite + inner crosshair) and its
        signal-plot child — the same lever _test_nav_drag uses."""
        tree = session.signal_trees[0]
        mgr = tree.navigator_plot_manager
        pw = next(iter(mgr.navigation_selectors.keys()))
        sel = mgr.navigation_selectors[pw][0]
        inner = getattr(sel, "selector", None) or sel
        child = next(iter(sel.children.keys()))
        return tree, sel, inner, child

    @staticmethod
    def _drive(sel, inner, x, y, wait: bool = True):
        """Park the crosshair at widget position (x, y) and run one forced
        update through the REAL dispatcher path."""
        pos = np.array([[int(x), int(y)]])
        inner.get_selected_indices = lambda: pos
        sel.delayed_update_data(force=True)
        if wait:
            _wait_dispatcher_idle()

    @staticmethod
    def _wait_frame(child, expected, timeout: float = 3.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            cd = child.current_data
            if cd is not None and np.array_equal(np.asarray(cd), expected):
                return True
            time.sleep(0.01)
        return False

    def test_placeholder_window_skips_frame_without_error(
            self, stem_4d_dataset, caplog):
        """A nav update landing inside the deepcopy window is skipped quietly:
        no ERROR log, the last good frame stays painted — and the SAME position
        paints normally once the window closes (the main signal still updates).
        """
        session = stem_4d_dataset["window"]
        tree, sel, inner, child = self._sel_and_child(session)
        root = tree.root
        assert child.plot_state.current_signal is root
        expected = np.array(root.data)   # the real (4, 5, 16, 16) array

        # Painted baseline at data position (1, 1).
        self._drive(sel, inner, 1, 1)
        assert self._wait_frame(child, expected[1, 1]), \
            "baseline frame at (1, 1) never painted"
        before = child.current_data

        # Open EXACTLY the transient window _deepcopy_with_new_data opens on
        # the live signal: data becomes array([None], dtype=object), shape (1,),
        # while axes_manager still reports nav (5, 4).
        old = root.data
        root.data = None
        assert root.data.shape == (1,) and root.data.dtype == object
        try:
            self._drive(sel, inner, 2, 3)    # → data order (3, 2)
        finally:
            root.data = old

        assert not _nav_error_records(caplog), \
            f"nav read errored inside the deepcopy window: {caplog.text}"
        assert child.current_data is before, \
            "the skipped read must leave the last good frame painted"

        # The window has closed — the same position now reads and paints.
        self._drive(sel, inner, 2, 3)
        assert self._wait_frame(child, expected[3, 2]), \
            "the nav read did not recover after the placeholder window closed"
        assert not _nav_error_records(caplog)

    def test_update_returns_none_mid_window(self, stem_4d_dataset):
        """Direct-call determinism: update_from_navigation_selection with the
        placeholder parked returns None (skip) instead of raising."""
        session = stem_4d_dataset["window"]
        tree, sel, inner, child = self._sel_and_child(session)
        root = tree.root

        old = root.data
        root.data = None
        try:
            out = uf.update_from_navigation_selection(
                inner, child, np.array([[0, 0]]))
        finally:
            root.data = old
        assert out is None

    def test_console_shaped_op_race_never_errors(self, stem_4d_dataset, caplog):
        """The real race shape: hammer ``root + 0`` — the exact expression the
        console's eye-toggle live preview evaluates against the LIVE bound
        signal on its own thread — while driving the dispatcher through many
        positions. Before the fix this logged the second-signal IndexError on
        a large fraction of moves (119/300 measured); with it, never."""
        session = stem_4d_dataset["window"]
        tree, sel, inner, child = self._sel_and_child(session)
        root = tree.root

        stop = threading.Event()

        def hammer():
            while not stop.is_set():
                _ = root + 0   # _deepcopy_with_new_data window on every call

        t = threading.Thread(target=hammer, daemon=True)
        t.start()
        try:
            for i in range(150):
                self._drive(sel, inner, i % 5, i % 4, wait=False)
                time.sleep(0.003)
        finally:
            stop.set()
            t.join(timeout=2.0)
        _wait_dispatcher_idle()

        assert not _nav_error_records(caplog), (
            f"{len(_nav_error_records(caplog))} nav-read ERRORs under the "
            f"console-op race:\n{caplog.text[:2000]}")
