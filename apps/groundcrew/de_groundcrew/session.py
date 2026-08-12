"""
session.py — Ground Crew's backend coordinator.

A thin app layer on `de_shell.session.SessionBase`: the base owns the window
registry, main-thread marshalling and the settings file, and
`de_shell.plotting.figure.FigureView` owns the image pane. What is left here is
the camera and the acquisition loop — which is the whole point.

Actions arrive from the renderer as `{"type": "action", "action": …, "payload":
…}` and are dispatched by name through `_ACTIONS`. That is deliberately a plain
dict rather than SpyDE's YAML-driven registry — Ground Crew has a handful of
fixed controls, and the staged-wizard machinery would be scaffolding around
nothing. If it grows, `de_shell.actions.registry` is there to adopt.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from de_shell.ipc import emit, emit_error, emit_status
from de_shell.session import SessionBase

from de_groundcrew.camera import AcquisitionLoop, SimulatedCamera
from de_shell.plotting.figure import FigureView

log = logging.getLogger(__name__)


class GroundCrewSession(SessionBase):
    """One instance per app lifetime."""

    def __init__(self, settings_dir: str) -> None:
        super().__init__(settings_dir=settings_dir)

        self.camera = SimulatedCamera()
        self._exposure_s = 0.05
        self._viewer = FigureView(self.next_window_id(), title="Live view")
        self.register_window_controller(self._viewer.window_id, self._viewer)

        self._loop = AcquisitionLoop(
            self.camera, on_frame=self._on_frame, exposure_s=self._exposure_s)

        # The newest frame, and a flag saying one is waiting to be painted.
        # The acquisition thread produces faster than the main thread can paint,
        # so a queue would grow without bound and show ever-staler frames.
        # Newest-wins with a single slot is the same shape as SpyDE's nav
        # painter: drop what you could not keep up with, never lag behind.
        self._pending_frame: np.ndarray | None = None
        self._paint_scheduled = False
        self._frame_lock = threading.Lock()
        self._frames_shown = 0
        self._frames_dropped = 0

    # ── Startup ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the viewer window and begin free-running the camera."""
        self._viewer.open(self.camera.shape)
        emit_status(f"Camera ready — {self.camera.shape[1]}×{self.camera.shape[0]}")
        self._loop.start()

    # ── Acquisition → paint ───────────────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray) -> None:
        """Acquisition thread: stash the frame and ask the main thread to paint.

        Only ever schedules ONE pending paint. Without the `_paint_scheduled`
        latch, a camera running faster than the renderer would queue a callback
        per frame onto the event loop and the main thread would fall
        progressively further behind while the queue grew.
        """
        with self._frame_lock:
            if self._pending_frame is not None:
                self._frames_dropped += 1
            self._pending_frame = frame
            if self._paint_scheduled:
                return
            self._paint_scheduled = True
        self._dispatch_to_main(self._paint_pending)

    def _paint_pending(self) -> None:
        """Main thread: paint the newest frame."""
        with self._frame_lock:
            frame, self._pending_frame = self._pending_frame, None
            self._paint_scheduled = False
        if frame is None or self._closed:
            return
        try:
            self._viewer.show(frame)
            self._frames_shown += 1
            self._emit_stats(frame)
        except Exception as e:
            log.exception("painting frame failed")
            emit_error(f"Display failed: {e}")

    def _emit_stats(self, frame: np.ndarray) -> None:
        """The stats strip under the image (the PySide6 app's image_stats_panel)."""
        emit({
            "type": "frame_stats",
            "frame": int(getattr(self.camera, "frame_index", self._frames_shown)),
            "shown": self._frames_shown,
            "dropped": self._frames_dropped,
            "min": float(frame.min()),
            "max": float(frame.max()),
            "mean": round(float(frame.mean()), 2),
            "dtype": str(frame.dtype),
            "shape": list(frame.shape),
        })

    # ── Actions ───────────────────────────────────────────────────────────────

    def dispatch_action(self, msg: dict) -> None:
        """Route `{"type": "action", ...}` from the renderer. Called by
        `de_shell.app`'s loop on the main thread."""
        name = str(msg.get("action", ""))
        payload = msg.get("payload") or {}
        handler = _ACTIONS.get(name)
        if handler is None:
            log.warning("unknown action: %s", name)
            return
        handler(self, payload)

    def _act_start(self, payload: dict) -> None:
        self._loop.start()
        emit_status("Acquiring")
        self._emit_acq_state()

    def _act_stop(self, payload: dict) -> None:
        self._loop.stop()
        emit_status("Stopped")
        self._emit_acq_state()

    def _act_single(self, payload: dict) -> None:
        """One exposure, off the main thread so a long one can't freeze the UI."""
        if self._loop.running:
            emit_status("Stop the live view before a single acquisition")
            return

        def _grab() -> None:
            try:
                self._on_frame(self.camera.acquire(self._exposure_s))
            except Exception as e:
                log.exception("single acquisition failed")
                self._dispatch_to_main(lambda: emit_error(f"Acquisition failed: {e}"))

        threading.Thread(target=_grab, daemon=True, name="groundcrew-single").start()

    def _act_set_exposure(self, payload: dict) -> None:
        try:
            seconds = float(payload.get("seconds", 0.05))
        except (TypeError, ValueError):
            emit_error(f"Invalid exposure: {payload.get('seconds')!r}")
            return
        self._exposure_s = max(seconds, 1e-3)
        self._loop.set_exposure(self._exposure_s)
        emit_status(f"Exposure {self._exposure_s * 1000:.0f} ms")
        self._emit_acq_state()

    def _act_set_colormap(self, payload: dict) -> None:
        self._viewer.set_colormap(str(payload.get("name", "gray")))

    def _emit_acq_state(self) -> None:
        emit({
            "type": "acq_state",
            "running": self._loop.running,
            "exposure_s": self._exposure_s,
        })

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        # Acquisition first: it is the only thing still producing work, and
        # stopping it before the viewer means no frame can arrive for a window
        # that has already gone.
        self._loop.stop()
        try:
            self._viewer.close()
        except Exception as e:
            log.debug("viewer close failed: %s", e)
        super().shutdown()


_ACTIONS = {
    "start_acquisition": GroundCrewSession._act_start,
    "stop_acquisition": GroundCrewSession._act_stop,
    "single_acquisition": GroundCrewSession._act_single,
    "set_exposure": GroundCrewSession._act_set_exposure,
    "set_colormap": GroundCrewSession._act_set_colormap,
}
