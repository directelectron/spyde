"""
session.py — Autopilot's backend coordinator.

`de_shell.session.SessionBase` owns the window registry, main-thread marshalling
and settings; `de_shell.plotting.figure.FigureView` owns the image pane. What is
here is the recipe, its runner, and turning runner callbacks into IPC.

Progress goes through the shell's `emit_progress`, so the renderer's progress
handling is the shared one rather than something this app invented.
"""
from __future__ import annotations

import logging

import numpy as np

from de_shell.ipc import emit, emit_error, emit_progress, emit_status
from de_shell.plotting.figure import FigureView
from de_shell.plotting.stream import FrameStream
from de_shell.session import SessionBase

from de_autopilot.recipe import Detector, RecipeRunner, Stage, default_recipe

log = logging.getLogger(__name__)


class AutopilotSession(SessionBase):
    """One instance per app lifetime."""

    def __init__(self, settings_dir: str) -> None:
        super().__init__(settings_dir=settings_dir)

        self.stage = Stage()
        self.detector = Detector()
        self.recipe = default_recipe()

        self._viewer = FigureView(self.next_window_id(), title="Last acquisition")
        self.register_window_controller(self._viewer.window_id, self._viewer)

        self._runner = self._new_runner()

        # Newest-wins painting, marshalled onto the main thread — the shell's,
        # the same one Ground Crew uses.
        self._frames = FrameStream(
            self._viewer, self._dispatch_to_main,
            on_painted=self._emit_frame_stats,
            on_error=lambda e: emit_error(f"Display failed: {e}"),
        )

    def _new_runner(self) -> RecipeRunner:
        return RecipeRunner(
            self.recipe, self.stage, self.detector,
            on_frame=self._on_frame, on_step=self._on_step, on_state=self._on_state,
        )

    # ── Startup ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the viewer and publish the recipe. Does NOT begin a run —
        unattended acquisition starts when the operator says so."""
        self._viewer.open(self.detector.shape)
        self._emit_recipe()
        emit_status(f"Ready — {len(self.recipe.steps)} steps")

    # ── Runner callbacks (runner thread) ──────────────────────────────────────

    def _on_frame(self, frame: np.ndarray) -> None:
        """Runner thread → the stream, which marshals and paints."""
        self._frames.submit(frame)

    def _emit_frame_stats(self, frame: np.ndarray) -> None:
        """Main thread, after a frame painted."""
        emit({
            "type": "frame_stats",
            "acquired": self._frames.shown,
            "stage": {"x": self.stage.x, "y": self.stage.y},
            "min": float(frame.min()), "max": float(frame.max()),
            "mean": round(float(frame.mean()), 2),
            "shape": list(frame.shape),
        })

    def _on_step(self, index: int, state: str) -> None:
        self._dispatch_to_main(lambda: self._emit_step(index, state))

    def _emit_step(self, index: int, state: str) -> None:
        emit({"type": "step_state", "index": index, "state": state})
        if state == "done":
            # The shell's progress channel, not a bespoke one — the renderer
            # already knows how to show it.
            emit_progress(index + 1, len(self.recipe.steps), "Running recipe")

    def _on_state(self, state: str) -> None:
        self._dispatch_to_main(lambda: self._emit_run_state(state))

    def _emit_run_state(self, state: str) -> None:
        emit({"type": "run_state", "state": state})
        emit_status({
            "running": "Running…", "paused": "Paused",
            "done": "Recipe complete", "stopped": "Stopped",
            "failed": "Recipe failed — see the log",
        }.get(state, state))
        if state in ("done", "stopped", "failed"):
            emit_progress(0, 0, "")     # clear the progress bar

    # ── Actions ───────────────────────────────────────────────────────────────

    def dispatch_action(self, msg: dict) -> None:
        name = str(msg.get("action", ""))
        payload = msg.get("payload") or {}
        handler = _ACTIONS.get(name)
        if handler is None:
            log.warning("unknown action: %s", name)
            return
        handler(self, payload)

    def _emit_recipe(self) -> None:
        emit({"type": "recipe", **self.recipe.to_json()})

    def _act_run(self, payload: dict) -> None:
        if self._runner.running:
            if self._runner.state == "paused":
                self._runner.resume()
            return
        # A finished runner is not restartable — its thread has exited — so a
        # second Run builds a fresh one rather than silently doing nothing.
        self._runner = self._new_runner()
        self._frames.shown = 0
        self._runner.start()

    def _act_pause(self, payload: dict) -> None:
        self._runner.pause()

    def _act_stop(self, payload: dict) -> None:
        self._runner.stop()
        self._emit_run_state("stopped")

    def _act_get_recipe(self, payload: dict) -> None:
        self._emit_recipe()

    def _act_set_colormap(self, payload: dict) -> None:
        self._viewer.set_colormap(str(payload.get("name", "gray")))

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        # The runner first: it is the only thing still producing frames, so
        # stopping it before the viewer means none can arrive for a dead window.
        try:
            self._runner.stop()
        except Exception as e:
            log.debug("runner stop failed: %s", e)
        # The stream next: it cancels anything outstanding, so a frame already
        # in flight cannot paint into a window that is about to close.
        self._frames.close()
        try:
            self._viewer.close()
        except Exception as e:
            log.debug("viewer close failed: %s", e)
        super().shutdown()


_ACTIONS = {
    "run_recipe": AutopilotSession._act_run,
    "pause_recipe": AutopilotSession._act_pause,
    "stop_recipe": AutopilotSession._act_stop,
    "get_recipe": AutopilotSession._act_get_recipe,
    "set_colormap": AutopilotSession._act_set_colormap,
}
