"""test_batch_feedback.py — the standard "this is computing" surface.

``lifecycle.batch_feedback`` bundles the three things a long batch run has to
show: the window's Calculating overlay, rate-limited status-bar progress, and
the most recent RESULT painted live. The third is the one that matters most and
is the easiest to get subtly wrong — a progress bar advancing through a run that
is quietly finding nothing looks exactly like one that is working.

These pin the behaviours a caller depends on:
  * the overlay is ALWAYS paired, including when the batch raises;
  * progress is rate-limited, but the FINAL result is never dropped;
  * the paint is marshalled to the main thread (CLAUDE.md's threading rule);
  * a failing paint cannot take the batch down.
"""
from __future__ import annotations

import time

import pytest

from spyde.actions import lifecycle


class _Session:
    """Records what got marshalled instead of running a loop."""

    def __init__(self, run=True):
        self.dispatched = []
        self._run = run

    def _dispatch_to_main(self, fn):
        self.dispatched.append(fn)
        if self._run:
            fn()


@pytest.fixture
def emitted(monkeypatch):
    """Capture ipc emissions at the module the helper imports them from."""
    out = {"progress": [], "computing": []}
    import spyde.backend.ipc as ipc
    monkeypatch.setattr(ipc, "emit_progress",
                        lambda d, t, label="": out["progress"].append((d, t, label)))
    monkeypatch.setattr(ipc, "emit_window_computing",
                        lambda wid, on: out["computing"].append((wid, bool(on))))
    return out


class TestOverlayPairing:
    def test_start_and_stop_bracket_the_run(self, emitted):
        with lifecycle.batch_feedback(_Session(), 7, "Segmenting", 3):
            pass
        assert emitted["computing"] == [(7, True), (7, False)]

    def test_the_overlay_stops_even_when_the_batch_raises(self, emitted):
        """The whole point of the pairing contract: a failed run must not leave
        a Calculating chip spinning over a window that has stopped working."""
        with pytest.raises(RuntimeError):
            with lifecycle.batch_feedback(_Session(), 7, "Segmenting", 3):
                raise RuntimeError("boom")
        assert emitted["computing"] == [(7, True), (7, False)]

    def test_no_window_sends_no_message(self, monkeypatch):
        """A plot with no window yet must not put a malformed message on the
        wire. The guard lives in ``emit_window_computing`` and is DELEGATED to,
        so this has to capture at ``ipc.emit`` — stubbing
        ``emit_window_computing`` (as the other tests here do) replaces the very
        guard under test and the assertion becomes vacuous.
        """
        import spyde.backend.ipc as ipc
        sent = []
        monkeypatch.setattr(ipc, "emit", lambda msg: sent.append(msg))
        with lifecycle.batch_feedback(_Session(), None, "Segmenting", 3):
            pass
        assert [m for m in sent if m.get("type") == "window_computing"] == []

    def test_a_real_window_id_does_send(self, monkeypatch):
        """The other half of the guard — otherwise "sends nothing" would pass
        for a helper that never emits at all."""
        import spyde.backend.ipc as ipc
        sent = []
        monkeypatch.setattr(ipc, "emit", lambda msg: sent.append(msg))
        with lifecycle.batch_feedback(_Session(), 7, "Segmenting", 3):
            pass
        assert [(m["window_id"], m["computing"]) for m in sent
                if m.get("type") == "window_computing"] == [(7, True), (7, False)]


class TestProgress:
    def test_rate_limited_between_steps(self, emitted):
        fb = lifecycle.batch_feedback(_Session(), 1, "Seg", 100, min_interval=60.0)
        fb.step(1)
        fb.step(2)
        fb.step(3)
        # Only the first got through; the rest are inside the interval. A
        # 900-frame run emitting per frame would put 900 messages on the same
        # stdout line protocol the nav painter uses.
        assert emitted["progress"] == [(1, 100, "Seg")]

    def test_force_defeats_the_rate_limit(self, emitted):
        fb = lifecycle.batch_feedback(_Session(), 1, "Seg", 100, min_interval=60.0)
        fb.step(1)
        fb.step(2, force=True)
        assert emitted["progress"] == [(1, 100, "Seg"), (2, 100, "Seg")]

    def test_the_last_step_is_never_rate_limited(self, emitted):
        """`done == total` always emits, so the bar reaches the end."""
        fb = lifecycle.batch_feedback(_Session(), 1, "Seg", 3, min_interval=60.0)
        fb.step(1)
        fb.step(2)
        fb.step(3)
        assert emitted["progress"][-1] == (3, 3, "Seg")

    def test_finish_emits_a_terminal_tick(self, emitted):
        """Without it an early-finishing run leaves the spinner mid-way and a
        finished job reads as hung."""
        fb = lifecycle.batch_feedback(_Session(), 1, "Seg", 10, min_interval=60.0)
        fb.step(4)
        fb.finish()
        assert emitted["progress"][-1] == (10, 10, "Seg")


class TestLiveResult:
    def test_the_result_is_published(self, emitted):
        seen = []
        fb = lifecycle.batch_feedback(_Session(), 1, "Seg", 3, publish=seen.append)
        fb.step(1, result="frame-1")
        assert seen == ["frame-1"]

    def test_the_paint_is_marshalled_to_the_main_thread(self, emitted):
        """Figure updates must not happen on the worker thread that computed
        them — CLAUDE.md's threading contract."""
        session = _Session(run=False)
        seen = []
        fb = lifecycle.batch_feedback(session, 1, "Seg", 3, publish=seen.append)
        fb.step(1, result="frame-1")
        assert seen == [], "painted inline instead of marshalling"
        assert len(session.dispatched) == 1
        session.dispatched[0]()
        assert seen == ["frame-1"]

    def test_a_rate_limited_step_publishes_nothing(self, emitted):
        seen = []
        fb = lifecycle.batch_feedback(_Session(), 1, "Seg", 100,
                                      publish=seen.append, min_interval=60.0)
        fb.step(1, result="a")
        fb.step(2, result="b")
        assert seen == ["a"]

    def test_the_final_result_survives_the_rate_limit(self, emitted):
        """The last frame is the one left on screen, so it must never be the
        one the rate limiter drops."""
        seen = []
        fb = lifecycle.batch_feedback(_Session(), 1, "Seg", 3,
                                      publish=seen.append, min_interval=60.0)
        fb.step(1, result="a")
        fb.step(2, result="b")
        fb.step(3, result="c")
        assert seen[-1] == "c"

    def test_a_failing_paint_does_not_break_the_batch(self, emitted):
        def boom(_r):
            raise ValueError("bad frame")

        fb = lifecycle.batch_feedback(_Session(), 1, "Seg", 3, publish=boom)
        fb.step(1, result="a")            # must not raise
        assert emitted["progress"] == [(1, 3, "Seg")]

    def test_step_without_a_result_still_reports_progress(self, emitted):
        seen = []
        fb = lifecycle.batch_feedback(_Session(), 1, "Seg", 3, publish=seen.append)
        fb.step(1)
        assert seen == []
        assert emitted["progress"] == [(1, 3, "Seg")]


class TestRealClock:
    def test_the_interval_actually_elapses(self, emitted):
        """The limiter is wall-clock, not a call counter.

        The margin is deliberately enormous (10x) rather than just over the
        interval: Windows' ``time.monotonic`` granularity is ~15.6 ms, so a
        sleep(0.06) against a 0.05 interval can measure as 0.047 and drop the
        second emission. That version passed alone and failed in the file — a
        real flake, not a real bug.
        """
        fb = lifecycle.batch_feedback(_Session(), 1, "Seg", 100, min_interval=0.02)
        fb.step(1)
        time.sleep(0.2)
        fb.step(2)
        assert emitted["progress"] == [(1, 100, "Seg"), (2, 100, "Seg")]
