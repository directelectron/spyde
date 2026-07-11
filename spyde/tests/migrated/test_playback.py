"""
Movie playback controller — REAL-TIME pacing + speed multiplier.

MoviePlaybackController advances the 1-D time navigator on a WALL-CLOCK frame clock:
1 s of playback = 1 s of experiment time (from the time-axis scale/units). Fast
Forward is a 2x→4x→8x→1x speed cycle. These tests use a fake selector/tree carrying
a configurable time axis (scale + units) so no real dataset or dispatcher is needed —
they pin the real-time pacing, unit conversion, the garbage-scale fallback, the speed
cycle, frame-skipping under a slow pipeline, and the classic bounds/pause/no-op.
"""
from __future__ import annotations

import time
import threading

import numpy as np

from spyde.actions.playback import MoviePlaybackController, DEFAULT_FPS


class _FakeWidget:
    def __init__(self, x=0.0):
        self.x = float(x)          # 1-D VLineWidget has .x (no .cx)


class _FakeSelector:
    """A 1-D time selector: .x is the frame index (scale 1). translate_pixels
    moves it; get_selected_indices reports the rounded index. `fire_delay`
    optionally makes each paint SLOW (to test frame-skipping)."""
    def __init__(self, fire_delay=0.0):
        self._widget = _FakeWidget(0.0)
        self.selector = self               # composite delegates to itself
        self.fires = 0
        self.fire_delay = fire_delay
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
        if self.fire_delay:
            time.sleep(self.fire_delay)


class _FakeAxis:
    def __init__(self, n, scale, units):
        self.scale = scale
        self.units = units
        self._n = n


class _FakeAxesManager:
    def __init__(self, n, scale, units):
        self.navigation_shape = (n,)
        self.navigation_axes = [_FakeAxis(n, scale, units)]


class _FakeRoot:
    def __init__(self, n, scale, units, signal_type="insitu"):
        self.axes_manager = _FakeAxesManager(n, scale, units)
        self._signal_type = signal_type


class _FakeTree:
    def __init__(self, n, scale=1.0, units="s", signal_type="insitu"):
        self.root = _FakeRoot(n, scale, units, signal_type)
        self._n = n
        self.navigator_plot_manager = self

    # MoviePlaybackController iterates mgr.all_navigation_selectors
    @property
    def all_navigation_selectors(self):
        return [self._sel]


class _FakeSession:
    def __init__(self, tree):
        self.signal_trees = [tree]


def _controller(n=50, scale=1.0, units="s", fire_delay=0.0, signal_type="insitu"):
    sel = _FakeSelector(fire_delay=fire_delay)
    tree = _FakeTree(n, scale=scale, units=units, signal_type=signal_type)
    tree._sel = sel
    return MoviePlaybackController(_FakeSession(tree)), sel


class TestPlayback:
    def test_no_movie_selector_is_noop(self):
        # A session with no 1-D selector → play() returns False.
        class _EmptySession:
            signal_trees = []
        pb = MoviePlaybackController(_EmptySession())
        assert pb.play() is False
        assert pb.is_playing is False

    def test_play_advances_frames(self):
        # scale 0.02 s → 50 fps at 1×. Poll rather than a fixed sleep: a starved
        # CI runner (macOS) delivered only 2 timer ticks in 0.35 s.
        pb, sel = _controller(n=500, scale=0.02, units="s")
        assert pb.play() is True
        deadline = time.time() + 10.0
        while sel.fires < 3 and time.time() < deadline:
            time.sleep(0.02)
        pb.pause()
        assert sel.fires >= 3, f"expected several frames, got {sel.fires}"
        idx = int(round(sel._widget.x))
        assert idx > 0, "playback did not advance the frame index"

    def test_pause_stops_advancing(self):
        pb, sel = _controller(n=500, scale=0.02, units="s")
        pb.play()
        time.sleep(0.2)
        pb.pause()
        assert pb.is_playing is False
        fires_after_pause = sel.fires
        time.sleep(0.2)
        assert sel.fires == fires_after_pause, "frames advanced after pause"

    def test_stops_at_last_frame_without_loop(self):
        pb, sel = _controller(n=6, scale=0.005, units="s")   # tiny fast movie
        pb.play(loop=False)
        deadline = time.time() + 10.0       # poll — CI-runner-speed independent
        while pb.is_playing and time.time() < deadline:
            time.sleep(0.02)
        assert pb.is_playing is False, "playback should stop at the last frame"
        assert int(round(sel._widget.x)) == 5, "should land ON the last frame"

    def test_loop_wraps_to_start(self):
        pb, sel = _controller(n=6, scale=0.005, units="s")
        pb.play(loop=True)
        time.sleep(0.4)
        looping = pb.is_playing
        pb.pause()
        assert looping is True, "loop playback should still be running"

    # ── real-time pacing ────────────────────────────────────────────────────

    def test_realtime_pacing_from_scale(self):
        # scale 0.05 s → 20 fps. Over ~0.6 s wall clock we expect ≈12 frames.
        pb, sel = _controller(n=1000, scale=0.05, units="s")
        assert pb.play() is True
        dt = 0.6
        time.sleep(dt)
        pb.pause()
        idx = int(round(sel._widget.x))
        expected = dt / 0.05                # ≈ 12
        lo, hi = expected * 0.6, expected * 1.4   # ±40% CI-safe band
        assert lo <= idx <= hi, \
            f"real-time pacing off: idx={idx}, expected≈{expected:.0f} ({lo:.0f}-{hi:.0f})"

    def test_units_ms_behaves_like_seconds(self):
        # scale=50 ms == 0.05 s → 20 fps (same as the seconds test above).
        pb, sel = _controller(n=1000, scale=50.0, units="ms")
        assert pb.play() is True
        dt = 0.6
        time.sleep(dt)
        pb.pause()
        idx = int(round(sel._widget.x))
        expected = dt / 0.05
        lo, hi = expected * 0.6, expected * 1.4
        assert lo <= idx <= hi, \
            f"ms units mis-converted: idx={idx}, expected≈{expected:.0f} ({lo:.0f}-{hi:.0f})"

    def test_effective_scale_seconds_helpers(self):
        pb, _ = _controller(n=10, scale=0.05, units="s")
        tree = pb._session.signal_trees[0]
        assert abs(pb._scale_seconds(tree) - 0.05) < 1e-9
        assert abs(pb._effective_scale_seconds(tree) - 0.05) < 1e-9
        # ms conversion
        pb2, _ = _controller(n=10, scale=50.0, units="ms")
        tree2 = pb2._session.signal_trees[0]
        assert abs(pb2._scale_seconds(tree2) - 0.05) < 1e-9

    def test_fallback_to_default_fps_on_garbage_scale(self):
        # Zero/absent scale → legacy default fps.
        pb, _ = _controller(n=10, scale=0.0, units="")
        tree = pb._session.signal_trees[0]
        assert abs(pb._effective_scale_seconds(tree) - 1.0 / DEFAULT_FPS) < 1e-9

        # Absurdly small scale (implies >1000 fps) → default.
        pb2, _ = _controller(n=10, scale=1e-9, units="s")
        tree2 = pb2._session.signal_trees[0]
        assert abs(pb2._effective_scale_seconds(tree2) - 1.0 / DEFAULT_FPS) < 1e-9

        # Unitless garbage scale is TREATED AS SECONDS (not rejected) when in-band:
        # scale 0.1 unitless → 10 fps, kept as-is.
        pb3, _ = _controller(n=10, scale=0.1, units="")
        tree3 = pb3._session.signal_trees[0]
        assert abs(pb3._effective_scale_seconds(tree3) - 0.1) < 1e-9

    def test_garbage_scale_still_plays_at_default(self):
        # A movie with no usable scale still plays (at 10 fps) — never a no-op.
        pb, sel = _controller(n=500, scale=0.0, units="")
        assert pb.play() is True
        deadline = time.time() + 10.0       # poll — CI-runner-speed independent
        while int(round(sel._widget.x)) <= 0 and time.time() < deadline:
            time.sleep(0.02)
        pb.pause()
        assert int(round(sel._widget.x)) > 0

    # ── fast-forward = speed cycle ──────────────────────────────────────────

    def test_fast_forward_cycles_speed(self):
        # Slow movie so it never auto-stops during the cycle.
        pb, _ = _controller(n=100000, scale=1.0, units="s")
        # From stopped → starts at 2×.
        assert pb.fast_forward() is True
        assert pb.is_playing is True
        assert pb.speed == 2
        # Then 2→4→8→1 while playing.
        pb.fast_forward(); assert pb.speed == 4
        pb.fast_forward(); assert pb.speed == 8
        pb.fast_forward(); assert pb.speed == 1     # wraps, still playing
        assert pb.is_playing is True
        pb.fast_forward(); assert pb.speed == 2
        pb.pause()

    def test_fast_forward_emits_speed_in_state(self):
        # The playback_state emit carries the current speed.
        emitted = []
        pb, _ = _controller(n=100000, scale=1.0, units="s")
        pb._emit_state = lambda: emitted.append(pb.speed)   # capture speed at emit
        pb.fast_forward()
        pb.fast_forward()
        assert 2 in emitted and 4 in emitted, f"speeds not emitted: {emitted}"
        pb.pause()

    def test_play_toggle_pauses_while_fast_forwarding(self):
        # Play is a plain on/off toggle: pressing it while FF-playing pauses.
        pb, _ = _controller(n=100000, scale=1.0, units="s")
        pb.fast_forward()                    # playing at 2×
        assert pb.is_playing is True
        assert pb.toggle() is False          # Play toggles it off
        assert pb.is_playing is False

    def test_speed_multiplier_advances_faster(self):
        # At 4× a movie advances ~4× as many frames in the same wall time.
        pb1, sel1 = _controller(n=100000, scale=0.02, units="s")
        pb1.play(speed=1)
        time.sleep(0.4)
        pb1.pause()
        idx1 = int(round(sel1._widget.x))

        pb4, sel4 = _controller(n=100000, scale=0.02, units="s")
        pb4.play(speed=4)
        time.sleep(0.4)
        pb4.pause()
        idx4 = int(round(sel4._widget.x))

        # 4× should be roughly 4× further; generous lower bound for CI jitter.
        assert idx4 > idx1 * 2.2, f"speed 4× not faster: idx1={idx1}, idx4={idx4}"

    # ── frame skipping (no drift) ───────────────────────────────────────────

    def test_frame_skipping_tracks_wall_time(self):
        # A SLOW paint (each fire sleeps) must not slow the CLOCK — the reached
        # index tracks WALL time (frames are skipped, not paced to render).
        pb, sel = _controller(n=100000, scale=0.01, units="s", fire_delay=0.05)
        pb.play()                            # target 100 fps; each paint takes 50ms
        dt = 0.6
        time.sleep(dt)
        pb.pause()
        idx = int(round(sel._widget.x))
        # If it PACED to render completion (50ms/frame) it'd reach only ~12.
        # Tracking wall time at 100 fps it should reach far past that.
        expected_walltime = dt / 0.01       # ≈ 60
        assert idx >= expected_walltime * 0.4, \
            f"frame-skipping failed: idx={idx} tracked render not wall time"
        # And far more than the number of (slow) paints that completed.
        assert idx > sel.fires, \
            f"index ({idx}) should outpace completed paints ({sel.fires}) via skipping"

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
        pb, sel = _controller(n=1000000, scale=0.01, units="s")
        for _ in range(5):
            pb.play()
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

    # ── signal-type gate ────────────────────────────────────────────────────

    def test_non_insitu_tree_is_not_played(self):
        # A 1-D-nav tree that is NOT insitu must not be picked as a movie.
        pb, _ = _controller(n=50, signal_type="electron_diffraction")
        assert pb.play() is False
        assert pb._time_selector()[0] is None

    # ── selector-type discrimination (unchanged contract) ───────────────────

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
            {"action": "playback", "payload": {"command": "play"}})
        import time as _t
        _t.sleep(0.15)
        assert session.playback.is_playing is True
        session.dispatch_action(
            {"action": "playback", "payload": {"command": "pause"}})
        assert session.playback.is_playing is False

    def test_fast_forward_via_dispatch_cycles_speed(self, movie_dataset):
        session = movie_dataset["window"]
        pb = session.playback
        session.dispatch_action(
            {"action": "playback", "payload": {"command": "fast_forward"}})
        assert pb.is_playing is True
        assert pb.speed == 2
        session.dispatch_action(
            {"action": "playback", "payload": {"command": "fast_forward"}})
        assert pb.speed == 4
        session.dispatch_action(
            {"action": "playback", "payload": {"command": "pause"}})
        assert pb.is_playing is False
