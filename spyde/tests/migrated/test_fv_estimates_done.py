"""The Find Vectors caret holds Compute until ``fv_estimates_done`` arrives.

So that message has to arrive on every way an open can end — estimates sent,
skipped, cached, or failed — or Compute stays greyed out for good. It must
also come AFTER the estimates it vouches for, and never from an open that a
later one superseded, which would enable Compute before the later open's
estimates land.
"""
from __future__ import annotations

import numpy as np
import pytest
import hyperspy.api as hs

import spyde.actions.find_vectors_action as fva
import spyde.actions.find_vectors_neural as fvn
import spyde.actions.vector_overlay as vector_overlay
from spyde.tests.migrated._async import quiesce, why_busy, wait_until
from spyde.tests.migrated.conftest import make_session, close_session

NEURAL = {"method": "neural", "sigma": 0.0, "kernel_radius": 5,
          "threshold": 0.3, "min_distance": 3}
NXCORR = {"method": "nxcorr", "sigma": 1.0, "kernel_radius": 5,
          "threshold": 0.4, "min_distance": 3}


def _diffraction_4d(nav=(4, 5), sig=(24, 24)):
    yy, xx = np.mgrid[0:sig[0], 0:sig[1]]
    disk = ((xx - 12) ** 2 + (yy - 12) ** 2 <= 16).astype(np.float32) * 100.0
    s = hs.signals.Signal2D(np.broadcast_to(disk, nav + sig).copy())
    s.set_signal_type("electron_diffraction")
    return s


def _signal_plot(session):
    return next(p for p in session._plots
                if not p.is_navigator and p.plot_state is not None)


@pytest.fixture
def opened(monkeypatch):
    """A session showing a 4D-STEM dataset, every message the action module
    sends recorded in order, and the neural model kept out of it: these tests
    are about which messages an open sends, not about the detector."""
    sent = []
    monkeypatch.setattr(fva, "emit", sent.append)
    monkeypatch.setattr(fva, "emit_status", lambda *a, **k: None)
    monkeypatch.setattr(fva, "emit_error", lambda text, *a, **k: sent.append(
        {"type": "error", "text": text}))
    monkeypatch.setattr(fva, "_ensure_model_local", lambda p: None)
    monkeypatch.setattr(fvn, "calibrate_neural", lambda frames, **kw: {
        "bg_sigma": 8.0, "thresh": 0.22, "scale_factor": 1.0})
    session = make_session()
    try:
        session._add_signal(_diffraction_4d())
        assert quiesce(session), why_busy(session)
        yield session, _signal_plot(session), sent
    finally:
        close_session(session)


def _types(sent):
    return [message["type"] for message in sent]


def _done(sent):
    return [m for m in sent if m["type"] == "fv_estimates_done"]


class TestEveryOpenEnds:
    def test_after_the_estimates_it_vouches_for(self, opened):
        session, plot, sent = opened
        fva.fv_open(session, plot, NEURAL)
        assert wait_until(lambda: _done(sent), 30), _types(sent)
        types = _types(sent)
        assert types.index("fv_auto_params") < types.index("fv_estimates_done")
        assert types.index("fv_calibration") < types.index("fv_estimates_done")
        assert _done(sent)[0]["window_id"] == plot.window_id

    def test_when_the_method_has_no_calibration(self, opened):
        session, plot, sent = opened
        fva.fv_open(session, plot, NXCORR)
        assert wait_until(lambda: _done(sent), 30), _types(sent)
        assert "fv_calibration" not in _types(sent)

    def test_when_the_calibration_fails(self, opened, monkeypatch):
        session, plot, sent = opened

        def broken(frames, **kw):
            raise RuntimeError("no model")

        monkeypatch.setattr(fvn, "calibrate_neural", broken)
        fva.fv_open(session, plot, NEURAL)
        assert wait_until(lambda: _done(sent), 30), _types(sent)
        assert "fv_calibration" not in _types(sent)

    def test_when_the_frame_is_too_large_to_calibrate(self, opened, monkeypatch):
        session, plot, sent = opened
        monkeypatch.setattr(fva, "_CAL_MAX_FRAME_PX", 1)
        fva.fv_open(session, plot, NEURAL)
        assert wait_until(lambda: _done(sent), 30), _types(sent)
        assert "fv_calibration" not in _types(sent)

    def test_when_the_preview_cannot_attach(self, opened, monkeypatch):
        session, plot, sent = opened

        def broken(*args, **kwargs):
            raise RuntimeError("no preview")

        monkeypatch.setattr(vector_overlay, "attach_find_vectors_preview", broken)
        fva.fv_open(session, plot, NEURAL)
        assert wait_until(lambda: _done(sent), 30), _types(sent)

    def test_when_a_reopen_finds_everything_already_estimated(self, opened):
        """The spot size is estimated once per dataset and the calibration is
        cached, so a second open sends at most the cached calibration."""
        session, plot, sent = opened
        fva.fv_open(session, plot, NEURAL)
        assert wait_until(lambda: _done(sent), 30), _types(sent)
        fva.fv_close(session, plot, {})
        sent.clear()
        fva.fv_open(session, plot, NEURAL)
        assert wait_until(lambda: _done(sent), 30), _types(sent)
        assert "fv_auto_params" not in _types(sent)

    def test_when_the_dataset_is_not_a_4d_scan(self, monkeypatch):
        sent = []
        monkeypatch.setattr(fva, "emit", sent.append)
        monkeypatch.setattr(fva, "emit_error", lambda *a, **k: None)
        session = make_session()
        try:
            session._add_signal(hs.signals.Signal2D(np.zeros((16, 16), np.float32)))
            assert quiesce(session), why_busy(session)
            fva.fv_open(session, _signal_plot(session), NXCORR)
            assert _done(sent), "a refused open left Compute waiting"
        finally:
            close_session(session)


class TestASupersededOpenStaysQuiet:
    def test_open_close_open_sends_one_done_after_the_live_estimates(
            self, opened, monkeypatch):
        """React mounts the caret twice: open, close, open, before either
        worker runs. Only the second open is live."""
        session, plot, sent = opened
        fva.fv_open(session, plot, NEURAL)
        fva.fv_close(session, plot, {})
        fva.fv_open(session, plot, NEURAL)
        assert wait_until(lambda: _done(sent), 30), _types(sent)
        assert quiesce(session), why_busy(session)
        assert len(_done(sent)) == 1
        types = _types(sent)
        assert types.index("fv_calibration") < types.index("fv_estimates_done")

    def test_closing_mid_calibration_sends_nothing(self, opened, monkeypatch):
        import threading
        session, plot, sent = opened
        calibrating, finish = threading.Event(), threading.Event()

        def slow(frames, **kw):
            calibrating.set()
            finish.wait(10)
            return {"bg_sigma": 8.0, "thresh": 0.22, "scale_factor": 1.0}

        monkeypatch.setattr(fvn, "calibrate_neural", slow)
        fva.fv_open(session, plot, NEURAL)
        assert calibrating.wait(30), "the calibration never started"
        fva.fv_close(session, plot, {})
        finish.set()
        assert quiesce(session), why_busy(session)
        assert not _done(sent), "a closed caret was told its estimates landed"
        assert "fv_calibration" not in _types(sent)
