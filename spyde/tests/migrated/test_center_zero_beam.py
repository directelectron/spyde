"""
Center Zero Beam (Electron, two-tab parity).

  Automatic — `czb_run` estimates the beam position (pyxem
              `get_direct_beam_position`) and applies `center_direct_beam`, adding
              a "Centered" tree node; the displayed pattern becomes centred.
  Manual    — `czb_open` drops a draggable crosshair; `czb_pick`
              centres by the picked position (constant shift).
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _wait(pred, timeout=25.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def _off_center_4d(nav=(3, 3), sig=(32, 32), beam=(18, 14)):
    """A disk offset from the detector centre (beam at col=18,row=14)."""
    yy, xx = np.mgrid[0:sig[0], 0:sig[1]]
    disk = ((xx - beam[0]) ** 2 + (yy - beam[1]) ** 2 <= 9).astype(np.float32)
    data = np.zeros(nav + sig, dtype=np.float32)
    for idx in np.ndindex(*nav):
        data[idx] = disk * 100.0
    s = hs.signals.Signal2D(data)
    s.set_signal_type("electron_diffraction")
    return s


def _com(frame):
    frame = np.asarray(frame, dtype=np.float64)
    yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    tot = frame.sum()
    return (xx * frame).sum() / tot, (yy * frame).sum() / tot   # (col, row)


class TestCenterZeroBeam:
    def test_auto_centers_beam(self):
        from spyde.backend.session import Session
        from spyde.actions.center_zero_beam import czb_run
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_off_center_4d(beam=(18, 14)))
            time.sleep(0.4)
            src = _signal_plot(session)
            before = src.plot_state.current_signal

            czb_run(session, src, {"method": "center_of_mass"})
            assert _wait(lambda: src.plot_state.current_signal is not before,
                         timeout=20), "centering never produced a new signal"
            centered = src.plot_state.current_signal
            frame = centered.inav[0, 0].data
            if hasattr(frame, "compute"):
                frame = frame.compute()
            cx, cy = _com(frame)
            assert abs(cx - 16) < 2 and abs(cy - 16) < 2, (cx, cy)
            # The tree records the step.
            node = session.signal_trees[0].get_node(before)
            assert any("Centered" in k for k in node.children)
        finally:
            session.shutdown()

    def test_manual_center_from_crosshair(self):
        from spyde.backend.session import Session
        from spyde.actions.center_zero_beam import czb_open, czb_pick
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_off_center_4d(beam=(18, 14)))
            time.sleep(0.4)
            src = _signal_plot(session)
            tree = src.signal_tree
            before = src.plot_state.current_signal

            czb_open(session, src, {})
            assert getattr(tree, "_czb_cross", None) is not None
            # Simulate the user dragging the crosshair onto the beam (18, 14).
            tree._czb_cross.set(cx=18.0, cy=14.0)

            czb_pick(session, src, {})
            assert _wait(lambda: src.plot_state.current_signal is not before,
                         timeout=20), "manual centering never produced a new signal"
            centered = src.plot_state.current_signal
            frame = centered.inav[0, 0].data
            if hasattr(frame, "compute"):
                frame = frame.compute()
            cx, cy = _com(frame)
            assert abs(cx - 16) < 2 and abs(cy - 16) < 2, (cx, cy)
            # Crosshair is cleared after applying.
            assert getattr(tree, "_czb_cross", None) is None
        finally:
            session.shutdown()
