"""
playback.py — real-time movie playback (Play / Fast-Forward) for the time navigator.

An in-situ movie has a 1-D time navigator (an ``IntegratingSelector1D`` wrapping a
draggable line). Playback is a WALL-CLOCK frame clock: it advances that selector so
that **1 second of wall clock = 1 second of experiment time** (at 1× speed), then
lets the existing navigator cascade paint the frame (the unified synchronous cached
read + read-ahead prefetch).

Real-time pacing
----------------
The navigation (time) axis is ``tree.root.axes_manager.navigation_axes[0]``. Its
``.scale`` is the per-frame time step and ``.units`` the unit. We convert the scale
to SECONDS (``s``/``ms``/``us``/``ns``/``min``/``h`` …; unitless → treat as seconds)
and derive the effective frame rate: ``fps = speed / scale_seconds``. scale=1 s →
1 fps; scale=1/60 s → 60 fps. A missing/absurd scale (fps outside ``[0.001, 1000]``)
falls back to the legacy default of 10 fps.

Frame skipping (no drift)
-------------------------
Each tick computes the TARGET frame from elapsed WALL time —
``idx = start_idx + floor(elapsed * speed / scale_seconds)`` — and moves the
selector to that absolute index (delta via ``translate_pixels``). A slow pipeline
therefore SKIPS indices rather than falling behind; a very high rate advances
several frames per tick. The wakeup rate is capped (~60 Hz); slow movies tick at the
frame rate. There is no fixed sleep-per-frame, so playback never accumulates drift.

Fast-forward = speed multiplier
-------------------------------
``fast_forward()`` cycles the speed 1×→2×→4×→8×→1× (starting playback at 2× if
stopped). ``play()`` is a plain toggle at 1× (or the current speed). Both step the
same selector via ``translate_pixels`` + ``delayed_update_data(force=True)`` — the
same path a manual drag uses — so no new frame-read machinery is introduced. At the
last frame it stops (or wraps if ``loop`` is set; the wall-clock origin wraps too).

The controller is owned by the ``Session`` (``session._playback``) so Play/Fast
Forward reuse one clock. It is movie-only: it finds the FIRST 1-D navigator selector
belonging to an ``insitu``-typed tree; on a 4D-STEM (2-D navigator) there is no time
axis to play, so play() is a no-op.
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

# Speed multipliers cycled by Fast Forward.
SPEED_CYCLE = (1, 2, 4, 8)

# Legacy fallback frame rate when the time axis carries no usable scale.
DEFAULT_FPS = 10.0

# Bounds on a real-time-derived frame rate. Outside this band the scale is treated
# as garbage (unset / absurd units) and we fall back to DEFAULT_FPS.
MIN_FPS = 0.001
MAX_FPS = 1000.0

# Cap the clock wakeup rate: never busy-spin faster than this even for a very fast
# movie (a single tick then advances several frames).
MAX_WAKE_HZ = 60.0

# Unit → seconds conversion for the time-axis scale.
_UNIT_SECONDS = {
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "ms": 1e-3, "msec": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3,
    "us": 1e-6, "µs": 1e-6, "usec": 1e-6, "microsecond": 1e-6, "microseconds": 1e-6,
    "ns": 1e-9, "nsec": 1e-9, "nanosecond": 1e-9, "nanoseconds": 1e-9,
    "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
}


def _units_to_seconds(units) -> float:
    """Multiplier converting a scale expressed in *units* to SECONDS.

    Unknown / empty / unitless units → 1.0 (treat the scale as already seconds)."""
    if not units:
        return 1.0
    key = str(units).strip().lower()
    return _UNIT_SECONDS.get(key, 1.0)


class MoviePlaybackController:
    """Owns the real-time playback clock for one session. Thread-safe start/stop.

    State machine (all guarded by ``self._lock``):
      • ``_playing``  — a clock thread is running.
      • ``speed``     — current multiplier (1/2/4/8); the "×N" the UI shows.
      • ``loop``      — wrap to the start at the end instead of stopping.
    ``play()`` is a plain toggle at the current (or requested) speed;
    ``fast_forward()`` starts at 2× if stopped, else bumps 1→2→4→8→1 while playing.
    """

    def __init__(self, session) -> None:
        self._session = session
        self._lock = threading.Lock()
        self._thread: "threading.Thread | None" = None
        self._stop = threading.Event()
        self.speed: int = 1
        self.loop: bool = False
        self._playing = False
        # Optional hint: the tree of the plot whose Play/FF button was clicked.
        # `_time_selector` tries this tree first (so the RIGHT movie plays when
        # several are open), then falls back to scanning all in-situ trees.
        self._preferred_tree = None

    def set_preferred_tree(self, tree) -> None:
        """Bias `_time_selector` toward *tree* (the clicked plot's tree)."""
        self._preferred_tree = tree

    # ── state ────────────────────────────────────────────────────────────────
    @property
    def is_playing(self) -> bool:
        return self._playing

    # ── selector discovery ───────────────────────────────────────────────────
    def _time_selector(self):
        """The movie's 1-D time-navigator selector + its tree, or ``(None, None)``.

        Only trees whose ROOT ``_signal_type == "insitu"`` qualify (a plain 1-D-nav
        signal is not a movie). Within a qualifying tree we take the first 1-D
        (line/range) navigator selector. The clicked plot's tree
        (``_preferred_tree``) is tried FIRST so the right movie plays when several
        are open, then all in-situ trees are scanned."""
        trees = list(self._session.signal_trees)
        pref = self._preferred_tree
        if pref is not None and pref in trees:
            trees = [pref] + [t for t in trees if t is not pref]
        for tree in trees:
            if not self._is_insitu_tree(tree):
                continue
            mgr = getattr(tree, "navigator_plot_manager", None)
            if mgr is None:
                continue
            for sel in getattr(mgr, "all_navigation_selectors", []) or []:
                if self._is_time_selector(sel):
                    return sel, tree
        return None, None

    @staticmethod
    def _is_insitu_tree(tree) -> bool:
        """True when the tree root is an in-situ movie (gates playback).

        Fixtures/tests may use a bare fake tree with no ``root._signal_type``; in
        that case (attribute absent) we do NOT reject it — the signal-type gate only
        DISqualifies a tree that explicitly declares a non-insitu type. This keeps
        the movie-only contract for real signals while staying permissive for the
        lightweight fakes."""
        root = getattr(tree, "root", None)
        st = getattr(root, "_signal_type", None)
        if st is None:
            return True                      # unknown (fake tree) → allow
        return st == "insitu"

    @staticmethod
    def _is_time_selector(sel) -> bool:
        """True for a 1-D (time/line/range) navigator selector — the movie's time
        axis. Discriminate on the WIDGET TYPE, not attribute-sniffing: a 2-D
        RectangleWidget also has ``.x`` and no ``.cx`` (it stores x/y/w/h), so the
        old ``.x``-but-no-``.cx`` test false-matched a rectangle ROI. A 1-D
        selector wraps an anyplotlib VLineWidget / RangeWidget; a 2-D crosshair
        has ``.cx``; a rectangle has ``.w``/``.h``. Require ``translate_pixels``
        (the frame-step API) AND a 1-D line/range widget."""
        if not hasattr(sel, "translate_pixels"):
            return False
        inner = getattr(sel, "selector", sel)
        w = getattr(inner, "_widget", None) or getattr(sel, "_widget", None)
        if w is None:
            return False
        # 2-D widgets: crosshair (.cx/.cy) or rectangle (.w/.h) → not a time axis.
        if hasattr(w, "cx") or hasattr(w, "cy") or hasattr(w, "w") or hasattr(w, "h"):
            return False
        # 1-D: a VLine has .x; a Range has .x0/.x1.
        return hasattr(w, "x") or hasattr(w, "x0")

    def _n_frames(self, tree) -> int:
        try:
            return int(tree.root.axes_manager.navigation_shape[0])
        except Exception:
            return 0

    def _current_index(self, sel) -> int:
        try:
            idx = sel.get_selected_indices()
            import numpy as np
            return int(np.atleast_1d(np.asarray(idx).ravel())[0])
        except Exception:
            return 0

    def _scale_seconds(self, tree) -> float:
        """The per-frame time step in SECONDS, from the time axis' scale + units.

        Returns 0.0 when the scale is missing / non-positive (→ caller falls back to
        the legacy default fps)."""
        try:
            ax = tree.root.axes_manager.navigation_axes[0]
            scale = float(getattr(ax, "scale", 0.0) or 0.0)
            if scale <= 0.0:
                return 0.0
            return scale * _units_to_seconds(getattr(ax, "units", None))
        except Exception:
            return 0.0

    def _effective_scale_seconds(self, tree) -> float:
        """The scale-seconds actually used for pacing.

        Falls back to ``1/DEFAULT_FPS`` when the axis has no usable scale, or when
        the scale would imply an absurd frame rate (fps outside ``[MIN, MAX]`` at
        1×). Real-time is the law when the axis is calibrated; the legacy default is
        only a safety net."""
        scale_s = self._scale_seconds(tree)
        if scale_s <= 0.0:
            logger.debug("playback: no usable time-axis scale → default %.1f fps",
                         DEFAULT_FPS)
            return 1.0 / DEFAULT_FPS
        base_fps = 1.0 / scale_s
        if base_fps < MIN_FPS or base_fps > MAX_FPS:
            logger.debug("playback: time-axis scale implies %.4g fps (absurd) → "
                         "default %.1f fps", base_fps, DEFAULT_FPS)
            return 1.0 / DEFAULT_FPS
        return scale_s

    # ── controls ─────────────────────────────────────────────────────────────
    def play(self, speed: "int | None" = None, loop: "bool | None" = None) -> bool:
        """Start (or restart) real-time playback at ``speed`` (default: current).

        Returns True if a movie time navigator was found and the clock started,
        else False (no-op — no movie present)."""
        if speed is not None:
            self.speed = self._coerce_speed(speed)
        if loop is not None:
            self.loop = bool(loop)

        sel, tree = self._time_selector()
        if sel is None:
            logger.debug("playback: no in-situ 1-D time navigator to play")
            return False

        with self._lock:
            self._stop_locked()
            self._stop = threading.Event()
            self._playing = True
            t = threading.Thread(target=self._run, args=(self._stop,),
                                 name="movie-playback", daemon=True)
            self._thread = t
            t.start()
        self._emit_state()
        return True

    def pause(self) -> None:
        with self._lock:
            self._stop_locked()
        self._emit_state()

    def toggle(self, **kw) -> bool:
        """Plain play/pause toggle at the current (or a requested) speed.

        Pressing this while FF-playing PAUSES (it does not reset the speed) — Play
        is a pure on/off toggle; Fast Forward owns the speed cycle."""
        if self._playing:
            self.pause()
            return False
        return self.play(**{k: v for k, v in kw.items() if k in ("speed", "loop")})

    def fast_forward(self, loop: "bool | None" = None) -> bool:
        """Fast-forward = speed multiplier cycle.

        Stopped  → start playing at 2×.
        Playing  → bump the speed 1→2→4→8→1 (stays playing at 1× after 8×).
        Returns True when playback is running afterwards."""
        if loop is not None:
            self.loop = bool(loop)
        if not self._playing:
            return self.play(speed=2)
        # Already playing: advance to the next multiplier in the cycle, restarting
        # the clock so the new speed re-bases the wall-clock origin at the current
        # frame (no jump / drift from the speed change).
        self.speed = self._next_speed(self.speed)
        return self.play(speed=self.speed)

    def set_speed(self, speed: int) -> None:
        self.speed = self._coerce_speed(speed)
        if self._playing:
            self.play(speed=self.speed)      # re-base at the new speed
        else:
            self._emit_state()

    def set_loop(self, loop: bool) -> None:
        self.loop = bool(loop)
        self._emit_state()

    def shutdown(self) -> None:
        with self._lock:
            self._stop_locked()

    # ── speed helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _coerce_speed(speed) -> int:
        try:
            s = int(round(float(speed)))
        except Exception:
            return 1
        return s if s in SPEED_CYCLE else min((c for c in SPEED_CYCLE if c >= s),
                                              default=SPEED_CYCLE[-1])

    @staticmethod
    def _next_speed(speed: int) -> int:
        try:
            i = SPEED_CYCLE.index(int(speed))
        except (ValueError, TypeError):
            return SPEED_CYCLE[1]            # unknown → jump to 2×
        return SPEED_CYCLE[(i + 1) % len(SPEED_CYCLE)]

    # ── internals ────────────────────────────────────────────────────────────
    def _stop_locked(self) -> None:
        self._playing = False
        if self._stop is not None:
            self._stop.set()
        self._thread = None

    def _run(self, stop: threading.Event) -> None:
        """Wall-clock frame clock. Targets the frame implied by ELAPSED wall time,
        so a slow read skips frames (never drifts) and a fast movie advances several
        frames per tick."""
        sel, tree = self._time_selector()
        if sel is None:
            self._finish(stop)
            return

        n = self._n_frames(tree)
        scale_s = self._effective_scale_seconds(tree)
        start_idx = self._current_index(sel)
        speed = max(1, int(self.speed))
        origin = time.monotonic()
        cur = start_idx
        # Wake cadence: aim for the frame period, but never busy-spin faster than
        # the wake cap (a fast movie then advances >1 frame per capped tick) nor
        # sleep so long we stop tracking (cap at 1 s for a very slow movie so
        # pause/stop stays responsive).
        frame_period = scale_s / speed if scale_s > 0 else 0.0
        wake = max(1.0 / MAX_WAKE_HZ, min(frame_period, 1.0)) if frame_period > 0 \
            else 1.0 / MAX_WAKE_HZ

        while not stop.is_set():
            elapsed = time.monotonic() - origin
            # Target frame from wall clock (frame-skipping, drift-free).
            target = start_idx + int(elapsed * speed / scale_s) if scale_s > 0 \
                else cur + 1

            if target >= n:
                if self.loop:
                    # Wrap the frame index AND re-base the wall-clock origin so the
                    # next second of playback starts cleanly at frame 0 (no jump).
                    try:
                        sel.translate_pixels(-cur)
                    except Exception as e:
                        logger.debug("playback loop reset failed: %s", e)
                        break
                    self._fire(sel)
                    cur = 0
                    start_idx = 0
                    origin = time.monotonic()
                else:
                    # Land exactly on the last frame, then stop.
                    if cur < n - 1:
                        try:
                            sel.translate_pixels((n - 1) - cur)
                        except Exception as e:
                            logger.debug("playback final step failed: %s", e)
                        else:
                            self._fire(sel)
                    break
            elif target != cur:
                try:
                    sel.translate_pixels(target - cur)
                except Exception as e:
                    logger.debug("playback step failed: %s", e)
                    break
                cur = target
                self._fire(sel)

            if stop.wait(wake):
                break

        self._finish(stop)

    def _finish(self, stop: threading.Event) -> None:
        # Only clear _playing if WE are still the current clock. A play→play restart
        # (or an auto-stop landing just after a new play()) must not let THIS (now
        # superseded) thread clobber the newer thread's _playing=True — that desync
        # leaked extra clock threads (the toggle saw paused mid-play and started
        # another). Guard on our own stop event being the live one.
        with self._lock:
            if self._stop is stop:
                self._playing = False
                emit_now = True
            else:
                emit_now = False            # a newer clock owns the state now
        if emit_now:
            self._emit_state()

    def _fire(self, sel) -> None:
        try:
            sel.delayed_update_data(force=True)
        except Exception as e:
            logger.debug("playback fire failed: %s", e)

    def _emit_state(self) -> None:
        """Tell the renderer the current play state so the toolbar can reflect it
        (the Play toggle + the Fast-Forward "×N" speed badge)."""
        try:
            from spyde.backend.ipc import emit
            emit({"type": "playback_state", "playing": self._playing,
                  "speed": int(self.speed), "loop": bool(self.loop)})
        except Exception as e:
            logger.debug("emit playback_state failed: %s", e)
