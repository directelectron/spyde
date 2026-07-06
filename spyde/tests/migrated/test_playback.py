"""
Movie playback controller (Phase 6).

MoviePlaybackController advances the 1-D time navigator on a frame clock, driving
the existing selector step (translate_pixels + delayed_update_data). These tests
use a fake selector/tree so no real dataset or dispatcher is needed — they pin the
clock's stepping, bounds, pause, fast-forward, and the no-movie no-op.
"""
from __future__ import annotations

import time
import threading

import numpy as np

from spyde.actions.playback import MoviePlaybackController


class _FakeWidget:
    def __init__(self, x=0.0):
        self.x = float(x)          # 1-D VLineWidget has .x (no .cx)


class _FakeSelector:
    """A 1-D time selector: .x is the frame index (scale 1). translate_pixels
    moves it; get_selected_indices reports the rounded index."""
    def __init__(self):
        self._widget = _FakeWidget(0.0)
        self.selector = self               # composite delegates to itself
        self.fires = 0
        self._lock = threading.Lock()

    def translate_pixels(self, shift_x):
        with self._lock:
            self._widget.x += float(shift_x)

    def get_selected_indices(self):
        with self._lock:
            return np.array([[int(round(self._widget.x))]])

    def delayed_update_data(self, force=False):
        with self._lock:
            self.fires += 1


class _FakeTree:
    def __init__(self, n):
        class _Ax:
            navigation_shape = (n,)
        class _AM:
            axes_manager = _Ax()
        class _Root:
            axes_manager = _Ax()
        self.root = _Root()
        self._n = n
        self.navigator_plot_manager = self

    # MoviePlaybackController iterates mgr.all_navigation_selectors
    @property
    def all_navigation_selectors(self):
        return [self._sel]


class _FakeSession:
    def __init__(self, n, sel):
        tree = _FakeTree(n)
        tree._sel = sel
        self.signal_trees = [tree]


def _controller(n=50):
    sel = _FakeSelector()
    session = _FakeSession(n, sel)
    return MoviePlaybackController(session), sel


class TestPlayback:
    def test_no_movie_selector_is_noop(self):
        # A session with no 1-D selector → play() returns False.
        class _EmptySession:
            signal_trees = []
        pb = MoviePlaybackController(_EmptySession())
        assert pb.play() is False
        assert pb.is_playing is False

    def test_play_advances_frames(self):
        pb, sel = _controller(n=50)
        assert pb.play(fps=50) is True     # 50 fps → ~20ms/frame
        time.sleep(0.35)
        pb.pause()
        assert sel.fires >= 3, f"expected several frames, got {sel.fires}"
        idx = int(round(sel._widget.x))
        assert idx > 0, "playback did not advance the frame index"

    def test_pause_stops_advancing(self):
        pb, sel = _controller(n=50)
        pb.play(fps=50)
        time.sleep(0.2)
        pb.pause()
        assert pb.is_playing is False
        fires_after_pause = sel.fires
        time.sleep(0.2)
        assert sel.fires == fires_after_pause, "frames advanced after pause"

    def test_stops_at_last_frame_without_loop(self):
        pb, sel = _controller(n=6)          # tiny movie
        pb.play(fps=100, loop=False)
        time.sleep(0.5)                      # plenty of time to reach the end
        assert pb.is_playing is False, "playback should stop at the last frame"
        assert int(round(sel._widget.x)) <= 6

    def test_loop_wraps_to_start(self):
        pb, sel = _controller(n=6)
        pb.play(fps=100, loop=True)
        time.sleep(0.4)
        looping = pb.is_playing
        pb.pause()
        assert looping is True, "loop playback should still be running"

    def test_fast_forward_uses_larger_step(self):
        pb, sel = _controller(n=200)
        pb.play(fps=50, step=10)
        time.sleep(0.25)
        pb.pause()
        # With step 10 the index advances much faster than 1/frame.
        assert int(round(sel._widget.x)) >= 20

    def test_single_step(self):
        pb, sel = _controller(n=50)
        sel.translate_pixels(0)             # at 0
        # Simulate the backend single-step handler.
        s, _ = pb._time_selector()
        s.translate_pixels(1)
        s.delayed_update_data(force=True)
        assert int(round(sel._widget.x)) == 1
        assert sel.fires == 1

    def test_restart_does_not_desync_playing_state(self):
        # play → play (restart): the superseded first clock exiting must NOT
        # clobber _playing back to False while the second clock runs (the thread
        # leak bug). After several rapid restarts, is_playing stays True and one
        # pause() truly stops everything.
        pb, sel = _controller(n=1000)
        for _ in range(5):
            pb.play(fps=60)
            time.sleep(0.03)
        # A superseded clock may have exited by now; state must still be "playing".
        assert pb.is_playing is True
        time.sleep(0.1)
        assert pb.is_playing is True, "restart race clobbered _playing"
        pb.pause()
        assert pb.is_playing is False
        fires = sel.fires
        time.sleep(0.15)
        assert sel.fires == fires, "a leaked clock thread kept advancing after pause"

    def test_does_not_match_a_2d_rectangle_selector(self):
        # A 2-D RectangleWidget has .x (x/y/w/h) and no .cx — the OLD heuristic
        # false-matched it. It must NOT be picked as a time selector.
        class _RectWidget:
            def __init__(self):
                self.x = 0.0; self.y = 0.0; self.w = 5.0; self.h = 5.0
        class _RectSel:
            def __init__(self):
                self._widget = _RectWidget()
                self.selector = self
            def translate_pixels(self, dx, dy=0):
                pass
        assert MoviePlaybackController._is_time_selector(_RectSel()) is False

    def test_matches_a_1d_line_and_range_selector(self):
        class _VLine:
            def __init__(self): self.x = 0.0
        class _Range:
            def __init__(self): self.x0 = 0.0; self.x1 = 3.0
        class _Sel:
            def __init__(self, w):
                self._widget = w; self.selector = self
            def translate_pixels(self, dx): pass
        assert MoviePlaybackController._is_time_selector(_Sel(_VLine())) is True
        assert MoviePlaybackController._is_time_selector(_Sel(_Range())) is True

    def test_does_not_match_a_2d_crosshair(self):
        class _Cross:
            def __init__(self): self.cx = 0.0; self.cy = 0.0
        class _Sel:
            def __init__(self): self._widget = _Cross(); self.selector = self
            def translate_pixels(self, dx, dy=0): pass
        assert MoviePlaybackController._is_time_selector(_Sel()) is False


class TestPlaybackSessionWiring:
    """The session exposes a lazy `playback` controller and routes the
    `playback` action + single-step through it."""

    def test_session_playback_property_and_dispatch(self, movie_dataset):
        session = movie_dataset["window"]
        pb = session.playback
        assert pb is session.playback          # cached (same instance)

        # A single-frame step through the real dispatch path advances the movie's
        # 1-D time navigator (no crash; the movie fixture has a 1-D nav).
        sel, tree = pb._time_selector()
        assert sel is not None, "movie fixture should have a 1-D time selector"
        start = pb._current_index(sel)
        session.dispatch_action(
            {"action": "playback", "payload": {"command": "step", "step": 1}})
        import time as _t
        _t.sleep(0.2)
        assert pb._current_index(sel) == start + 1

    def test_play_then_pause_via_dispatch(self, movie_dataset):
        session = movie_dataset["window"]
        session.dispatch_action(
            {"action": "playback", "payload": {"command": "play", "fps": 40}})
        import time as _t
        _t.sleep(0.15)
        assert session.playback.is_playing is True
        session.dispatch_action(
            {"action": "playback", "payload": {"command": "pause"}})
        assert session.playback.is_playing is False
