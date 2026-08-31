"""
Per-window "Calculating…" overlay lifecycle (laundry item #3).

The backend brackets a long compute that paints into a plot window with
``window_computing`` start/stop messages (``spyde.actions.lifecycle.
window_computing`` / ``de_shell.ipc.emit_window_computing``) — the
renderer uses these to show/hide a floating translucent chip centered on that
plot (WindowContent.tsx). This suite pins the backend contract on the
progressive navigator fill (``BaseSignalTree._start_progressive_nav_compute``,
the THREADED/no-cluster path exercised under ``SPYDE_NO_DASK=1``):

  1. a ``window_computing`` message with ``computing: True`` is emitted before
     the fill starts,
  2. a matching ``computing: False`` is emitted once the fill finishes,
  3. both messages carry the SAME ``window_id`` (the navigator plot's window),
  4. a compute that raises PARTWAY THROUGH still emits the stop message (the
     ``finally``/context-manager guarantee — a failed compute must not leave
     the renderer's overlay stuck forever).
"""
from __future__ import annotations

import time

import numpy as np
import dask.array as da
import hyperspy.api as hs
import pytest
from spyde.tests.migrated.conftest import make_session


def _make_session():
    return make_session()


def _lazy_4d(nav=(4, 4), sig=(8, 8), chunks=(2, 2, 8, 8)):
    ny, nx = nav
    base = np.arange(ny * nx * sig[0] * sig[1], dtype=np.float32).reshape(ny, nx, *sig)
    arr = da.from_array(base, chunks=chunks)
    s = hs.signals.Signal2D(arr).as_lazy()
    s.set_signal_type("electron_diffraction")
    return s


def _computing_messages(messages, window_id=None):
    out = [m for m in messages if m.get("type") == "window_computing"]
    if window_id is not None:
        out = [m for m in out if m.get("window_id") == window_id]
    return out


def _wait(pred, timeout=8.0, interval=0.05):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(interval)
    return False


class TestWindowComputingOverlayNavFill:
    """Happy-path: start before the fill, stop once it completes."""

    def test_start_then_stop_bracket_the_progressive_nav_fill(self, monkeypatch, captured_messages):
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        session = _make_session()
        try:
            s = _lazy_4d()
            session._add_signal(s, source_path=None)

            # Wait for the fill to finish: a stop (computing:False) message lands.
            assert _wait(lambda: any(
                m.get("computing") is False for m in _computing_messages(captured_messages)
            )), "no window_computing stop message ever arrived"

            tree = session.signal_trees[-1]
            mgr = tree.navigator_plot_manager
            assert mgr is not None
            nav_pw = next(iter(mgr.plot_windows.keys()))
            nav_plot = mgr.plots[nav_pw][0]
            wid = nav_plot.window_id
            assert wid is not None

            msgs = _computing_messages(captured_messages, window_id=wid)
            assert len(msgs) >= 2, (
                f"expected >=2 window_computing messages for window {wid}, got {msgs}"
            )
            # First message for this window must be the start.
            assert msgs[0]["computing"] is True
            # A stop must follow somewhere after it.
            assert any(m["computing"] is False for m in msgs[1:])

            # And every started window_id got a matching stop — a second
            # property of the same captured message list (this ran as its own
            # test with an identical session + fill).
            starts = [m["window_id"] for m in _computing_messages(captured_messages)
                      if m["computing"] is True]
            stops = [m["window_id"] for m in _computing_messages(captured_messages)
                     if m["computing"] is False]
            assert starts, "no start message emitted"
            assert stops, "no stop message emitted"
            assert set(starts) <= set(stops), (
                f"started windows {set(starts)} missing a stop in {set(stops)}"
            )
        finally:
            session.shutdown()


class TestWindowComputingOverlayErrorPath:
    """A compute that raises partway through must still clear the overlay."""

    def test_failed_fill_still_emits_stop(self, monkeypatch, captured_messages):
        monkeypatch.setenv("SPYDE_NO_DASK", "1")

        # Make every dask block compute raise, so the threaded per-chunk fill
        # (_bg_nav in signal_tree.py) hits its except-block instead of
        # finishing normally — the window_computing stop must STILL fire
        # (it lives in a `finally`, not after the try body).
        def _boom(block, block_info=None):
            raise RuntimeError("synthetic nav-fill failure")

        session = _make_session()
        try:
            ny = nx = 4
            base = np.zeros((ny, nx, 8, 8), dtype=np.float32)
            arr = da.from_array(base, chunks=(2, 2, 8, 8))
            arr = arr.map_blocks(_boom, dtype=np.float32)
            s = hs.signals.Signal2D(arr).as_lazy()
            s.set_signal_type("electron_diffraction")

            session._add_signal(s, source_path=None)

            # A stop message must still arrive despite the compute failing.
            assert _wait(lambda: any(
                m.get("computing") is False for m in _computing_messages(captured_messages)
            ), timeout=8.0), (
                "a failed progressive nav fill left the window_computing overlay "
                "stuck (no stop message) — the compute's error path must still "
                "clear it via a finally/context-manager, not just the happy path"
            )

            # And a start must have preceded it (otherwise this is a vacuous pass).
            starts = [m for m in captured_messages
                      if m.get("type") == "window_computing" and m.get("computing") is True]
            assert starts, "no start message emitted before the failure"
        finally:
            session.shutdown()


class TestWindowComputingLifecycleHelper:
    """Unit-level pin on spyde.actions.lifecycle.window_computing itself —
    the shared context-manager every call site relies on for the
    guaranteed-stop contract."""

    def test_context_manager_emits_start_then_stop(self, monkeypatch):
        from spyde.actions.lifecycle import window_computing

        calls = []
        monkeypatch.setattr(
            "de_shell.ipc.emit_window_computing",
            lambda window_id, computing: calls.append((window_id, computing)),
        )

        with window_computing(42):
            assert calls == [(42, True)]
        assert calls == [(42, True), (42, False)]

    def test_context_manager_emits_stop_on_exception(self, monkeypatch):
        from spyde.actions.lifecycle import window_computing

        calls = []
        monkeypatch.setattr(
            "de_shell.ipc.emit_window_computing",
            lambda window_id, computing: calls.append((window_id, computing)),
        )

        with pytest.raises(ValueError):
            with window_computing(7):
                raise ValueError("boom")
        assert calls == [(7, True), (7, False)]

    def test_none_window_id_is_a_silent_noop(self, monkeypatch):
        from spyde.actions.lifecycle import window_computing

        calls = []
        monkeypatch.setattr(
            "de_shell.ipc.emit_window_computing",
            lambda window_id, computing: calls.append((window_id, computing)),
        )

        # emit_window_computing itself no-ops on None; the context manager
        # still calls through to it (no special-casing needed at call sites).
        with window_computing(None):
            pass
        assert calls == [(None, True), (None, False)]
