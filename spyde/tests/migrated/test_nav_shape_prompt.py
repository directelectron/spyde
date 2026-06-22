"""
Navigation-shape + step-size prompt on load.

A navigated 2-D-signal dataset (4D-STEM scan, or a flat stack of diffraction
patterns) should NOT silently open with whatever shape/calibration the reader
guessed. The backend emits a ``nav_shape_prompt`` with the inferred shape +
current step size; the frontend confirms/overrides, and ``confirm_nav_shape``
reshapes (lazily) + calibrates before opening.

This was lost in the Qt→Electron port (MRC scans opened with scale=1 and no
shape confirmation), so a 3 nm-step scan had the wrong real-space calibration.
"""
from __future__ import annotations

import time

import numpy as np
import dask.array as da
import hyperspy.api as hs
import pytest


def _make_session():
    from spyde.backend.session import Session
    return Session(n_workers=1, threads_per_worker=1)


def _msgs_of(session_msgs, mtype):
    return [m for m in session_msgs if m.get("type") == mtype]


def _lazy_4d(nav=(5, 4), sig=(8, 8)):
    ny, nx = nav
    base = np.zeros((ny, nx) + sig, dtype=np.float32)
    for iy in range(ny):
        for ix in range(nx):
            base[iy, ix] = float(iy * nx + ix + 1)
    arr = da.from_array(base, chunks=(ny, nx) + sig)
    s = hs.signals.Signal2D(arr).as_lazy()
    s.set_signal_type("electron_diffraction")
    return s


def _lazy_stack(n=12, sig=(8, 8)):
    """A flat stack of n images (navigation_dimension == 1)."""
    arr = da.from_array(np.arange(n * sig[0] * sig[1], dtype=np.float32)
                        .reshape((n,) + sig), chunks=(n,) + sig)
    return hs.signals.Signal2D(arr).as_lazy()


class TestNavShapePrompt:
    def test_4d_load_emits_prompt_and_does_not_open(self, captured_messages, monkeypatch):
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        session = _make_session()
        try:
            # Drive the load synchronously (skip the thread) for determinism.
            sig = _lazy_4d(nav=(5, 4))   # data nav (ny=5, nx=4) → display (x=4, y=5)
            session._prompt_nav_shape(sig, "scan.mrc")

            prompts = _msgs_of(captured_messages, "nav_shape_prompt")
            assert prompts, "no nav_shape_prompt emitted"
            p = prompts[-1]
            assert p["nav_shape"] == [4, 5]          # display (x, y)
            assert p["n_patterns"] == 20
            assert p["signal_shape"] == [8, 8]
            # Nothing opened yet — we're waiting on the user.
            assert not _msgs_of(captured_messages, "window_opened")
            assert session._pending_load is not None
        finally:
            session.shutdown()

    def test_confirm_applies_step_size_and_opens(self, captured_messages, monkeypatch):
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        session = _make_session()
        try:
            sig = _lazy_4d(nav=(5, 4))
            session._prompt_nav_shape(sig, "scan.mrc")
            session._confirm_nav_shape({"nav_shape": [5, 4], "step_size": 3.0, "units": "nm"})
            time.sleep(0.4)

            tree = session.signal_trees[-1]
            nav_axes = tree.root.axes_manager.navigation_axes
            assert all(abs(ax.scale - 3.0) < 1e-9 for ax in nav_axes), "step size not applied"
            assert all(ax.units == "nm" for ax in nav_axes)
            assert _msgs_of(captured_messages, "window_opened"), "did not open after confirm"
            assert session._pending_load is None
        finally:
            session.shutdown()

    def test_confirm_reshapes_flat_stack_lazily(self, captured_messages, monkeypatch):
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        session = _make_session()
        try:
            sig = _lazy_stack(n=12)               # nav-dim 1, 12 frames
            assert sig.axes_manager.navigation_shape == (12,)
            session._prompt_nav_shape(sig, "stack.mrc")
            # Fold 12 → (4, 3) display scan grid with a 2 nm step.
            session._confirm_nav_shape({"nav_shape": [4, 3], "step_size": 2.0, "units": "nm"})
            time.sleep(0.4)

            root = session.signal_trees[-1].root
            assert root.axes_manager.navigation_shape == (4, 3), \
                f"reshape failed: {root.axes_manager.navigation_shape}"
            assert root.axes_manager.signal_shape == (8, 8)
            assert root._lazy, "reshape materialised the data (not lazy!)"
        finally:
            session.shutdown()

    def test_confirm_with_wrong_frame_count_falls_back(self, captured_messages, monkeypatch):
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        session = _make_session()
        try:
            sig = _lazy_stack(n=12)
            session._prompt_nav_shape(sig, "stack.mrc")
            # 5×3 = 15 ≠ 12 → reshape rejected; opens as-loaded + emits an error.
            session._confirm_nav_shape({"nav_shape": [5, 3], "step_size": 1.0, "units": "nm"})
            time.sleep(0.4)
            assert _msgs_of(captured_messages, "error"), "no error on bad shape"
            assert _msgs_of(captured_messages, "window_opened"), "should still open as-loaded"
        finally:
            session.shutdown()

    def test_open_file_emits_busy_then_clears(self, captured_messages, monkeypatch, tmp_path):
        """A file open emits loading busy=True up front (so the UI shows a
        spinner during the cold-cache read) and busy=False when the read is
        done — so a big first-open never just looks hung."""
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        session = _make_session()
        try:
            # Write a small real .hspy so open_file's load path runs end-to-end.
            p = tmp_path / "img.hspy"
            hs.signals.Signal2D(np.zeros((8, 8), dtype=np.float32)).save(str(p))
            session.open_file(str(p))
            time.sleep(0.8)

            loading = [m for m in captured_messages if m.get("type") == "loading"]
            assert any(m.get("busy") for m in loading), "no busy=True on open"
            assert any(not m.get("busy") for m in loading), "busy never cleared"
            # busy=True must come before busy=False.
            busy_idx = next(i for i, m in enumerate(loading) if m.get("busy"))
            clear_idx = next(i for i, m in enumerate(loading) if not m.get("busy"))
            assert busy_idx < clear_idx
        finally:
            session.shutdown()

    def test_plain_2d_image_does_not_prompt(self, monkeypatch):
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        from spyde.backend.session import Session
        s = hs.signals.Signal2D(np.zeros((16, 16), dtype=np.float32))
        assert Session._wants_nav_prompt(s) is False
        s4d = _lazy_4d()
        assert Session._wants_nav_prompt(s4d) is True
