"""
playback.py — movie playback (Play / Pause / Fast-Forward) for the time navigator.

An in-situ movie has a 1-D time navigator (an ``IntegratingSelector1D`` wrapping a
draggable line). Playback is just a frame clock that advances that selector one
step every ``1/fps`` seconds and lets the existing navigator cascade paint the
frame (the unified synchronous cached read + read-ahead prefetch). Fast-forward =
a larger step and/or higher fps.

The clock runs on a single daemon ``threading.Timer``-style loop and steps the
selector via ``translate_pixels`` + ``delayed_update_data(force=True)`` — the same
path a manual drag uses — so no new frame-read machinery is introduced. When it
reaches the last frame it stops (or loops if ``loop`` is set).

The controller is owned by the ``Session`` (``session._playback``) so Play/Pause
toggles reuse one clock. It is movie-only: it finds the FIRST 1-D navigator
selector; on a 4D-STEM (2-D navigator) there is no time axis to play, so play()
is a no-op.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class MoviePlaybackController:
    """Owns the playback clock for one session. Thread-safe start/stop."""

    def __init__(self, session) -> None:
        self._session = session
        self._lock = threading.Lock()
        self._thread: "threading.Thread | None" = None
        self._stop = threading.Event()
        self.fps: float = 10.0
        self.step: int = 1
        self.loop: bool = False
        self._playing = False

    # ── state ────────────────────────────────────────────────────────────────
    @property
    def is_playing(self) -> bool:
        return self._playing

    # ── selector discovery ───────────────────────────────────────────────────
    def _time_selector(self):
        """The movie's 1-D time-navigator selector, or None (no movie / 2-D nav).

        A 1-D navigator selector exposes ``translate_pixels`` (the frame step) and
        a widget with a single ``.x`` (via its active inner selector). We detect it
        by the presence of ``translate_pixels`` + a 1-D widget (``.x`` but no
        ``.cx``)."""
        for tree in self._session.signal_trees:
            mgr = getattr(tree, "navigator_plot_manager", None)
            if mgr is None:
                continue
            for sel in mgr.all_navigation_selectors:
                if self._is_time_selector(sel):
                    return sel, tree
        return None, None

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

    # ── controls ─────────────────────────────────────────────────────────────
    def play(self, fps: "float | None" = None, step: "int | None" = None,
             loop: "bool | None" = None) -> bool:
        """Start (or restart) playback. Returns True if a movie time navigator was
        found and the clock started, else False (no-op)."""
        if fps is not None:
            self.fps = max(0.5, float(fps))
        if step is not None:
            self.step = max(1, int(step))
        if loop is not None:
            self.loop = bool(loop)

        sel, tree = self._time_selector()
        if sel is None:
            logger.debug("playback: no 1-D time navigator to play")
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
        if self._playing:
            self.pause()
            return False
        return self.play(**kw)

    def set_fps(self, fps: float) -> None:
        self.fps = max(0.5, float(fps))
        self._emit_state()

    def set_step(self, step: int) -> None:
        self.step = max(1, int(step))
        self._emit_state()

    def shutdown(self) -> None:
        with self._lock:
            self._stop_locked()

    # ── internals ────────────────────────────────────────────────────────────
    def _stop_locked(self) -> None:
        self._playing = False
        if self._stop is not None:
            self._stop.set()
        self._thread = None

    def _run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            sel, tree = self._time_selector()
            if sel is None:
                break
            n = self._n_frames(tree)
            cur = self._current_index(sel)
            nxt = cur + self.step
            if nxt >= n:
                if self.loop:
                    # Jump back to the start.
                    try:
                        sel.translate_pixels(-cur)
                    except Exception as e:
                        logger.debug("playback loop reset failed: %s", e)
                    self._fire(sel)
                else:
                    break                       # reached the end → stop
            else:
                try:
                    sel.translate_pixels(self.step)
                except Exception as e:
                    logger.debug("playback step failed: %s", e)
                    break
                self._fire(sel)
            # Pace to the target fps. A slow frame read lowers the effective rate
            # (the read is synchronous on the dispatcher); at high fps the
            # dispatcher coalesces + drops stale positions, so playback SKIPS
            # frames rather than pacing to render completion — fine for a scrubber.
            if stop.wait(1.0 / self.fps):
                break
        # Only clear _playing if WE are still the current clock. A play→play
        # restart (or an auto-stop landing just after a new play()) must not let
        # THIS (now superseded) thread clobber the newer thread's _playing=True —
        # that desync leaked extra clock threads (the toggle saw paused mid-play
        # and started another). Guard on our own stop event being the live one.
        with self._lock:
            if self._stop is stop:
                self._playing = False
                emit_now = True
            else:
                emit_now = False           # a newer clock owns the state now
        if emit_now:
            self._emit_state()

    def _fire(self, sel) -> None:
        try:
            sel.delayed_update_data(force=True)
        except Exception as e:
            logger.debug("playback fire failed: %s", e)

    def _emit_state(self) -> None:
        """Tell the renderer the current play state so the toolbar can reflect it."""
        try:
            from spyde.backend.ipc import emit
            emit({"type": "playback_state", "playing": self._playing,
                  "fps": self.fps, "step": self.step, "loop": self.loop})
        except Exception as e:
            logger.debug("emit playback_state failed: %s", e)
