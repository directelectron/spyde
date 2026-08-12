"""
recipe.py — the acquisition sequence and the thing that runs it.

A *recipe* is an ordered list of steps; a *runner* walks them on a daemon
thread, reporting each frame and each state change through callbacks. The
callbacks fire on the RUNNER's thread, so anything that touches a figure has to
marshal onto the asyncio main thread — see `session.py`.

The design constraint worth naming: a run must be **stoppable promptly**. An
acquisition step can be seconds long, so the runner waits on an Event rather
than sleeping, and checks for a stop between and *during* steps. A Stop button
that takes ten seconds to take effect is one the operator stops trusting.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

log = logging.getLogger(__name__)

StepKind = Literal["acquire", "move", "settle"]
RunState = Literal["idle", "running", "paused", "done", "stopped", "failed"]


@dataclass
class Step:
    """One instruction. `kind` decides which fields matter."""
    kind: StepKind
    label: str
    #: acquire — exposure in seconds.
    exposure_s: float = 0.05
    #: move — stage destination, in arbitrary units.
    x: float = 0.0
    y: float = 0.0
    #: settle — how long to wait.
    seconds: float = 0.2


@dataclass
class Recipe:
    name: str = "Untitled"
    steps: list[Step] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "steps": [
                {"kind": s.kind, "label": s.label, "exposure_s": s.exposure_s,
                 "x": s.x, "y": s.y, "seconds": s.seconds}
                for s in self.steps
            ],
        }


def default_recipe() -> Recipe:
    """A short 2×2 raster with settles — enough to watch it run."""
    steps: list[Step] = []
    for iy in range(2):
        for ix in range(2):
            steps.append(Step("move", f"Move to ({ix}, {iy})", x=float(ix), y=float(iy)))
            steps.append(Step("settle", "Settle", seconds=0.15))
            steps.append(Step("acquire", f"Acquire ({ix}, {iy})", exposure_s=0.15))
    return Recipe(name="2 × 2 raster", steps=steps)


class Stage:
    """A simulated sample stage. Position feeds the synthetic frames, so moving
    visibly changes what is acquired."""

    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0

    def move_to(self, x: float, y: float) -> None:
        self.x, self.y = float(x), float(y)


class Detector:
    """Synthetic frames that depend on stage position.

    Position-dependent on purpose: a scan whose frames all look the same cannot
    show a stage that never moved, or a runner that acquired every step at the
    same place.
    """

    def __init__(self, shape: tuple[int, int] = (256, 256), seed: int = 0) -> None:
        self.shape = shape
        self._rng = np.random.default_rng(seed)

    def acquire(self, stage: Stage, exposure_s: float) -> np.ndarray:
        h, w = self.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # Two features whose positions track the stage, so a move is visible.
        cy = h * (0.3 + 0.25 * stage.y)
        cx = w * (0.3 + 0.25 * stage.x)
        blob = 2500.0 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (h * 0.08) ** 2)))
        gain = max(exposure_s, 1e-3) / 0.05
        noise = self._rng.poisson(30.0 * gain, size=(h, w)).astype(np.float32)
        frame = blob * gain + noise
        frame[: h // 20, : w // 20] += 2000.0     # fixed orientation marker
        return np.clip(frame, 0, 65535).astype(np.uint16)


class RecipeRunner:
    """Walks a recipe on a daemon thread.

    Callbacks (all fired on the runner thread):
      ``on_frame(frame)``            an acquire step produced an image
      ``on_step(index, state)``      a step started / finished
      ``on_state(state)``            the run's overall state changed
    """

    def __init__(self, recipe: Recipe, stage: Stage, detector: Detector, *,
                 on_frame: Callable[[np.ndarray], None],
                 on_step: Callable[[int, str], None],
                 on_state: Callable[[RunState], None]) -> None:
        self.recipe = recipe
        self.stage = stage
        self.detector = detector
        self._on_frame = on_frame
        self._on_step = on_step
        self._on_state = on_state

        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._thread: threading.Thread | None = None
        self.state: RunState = "idle"
        self.current_step = -1

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _set_state(self, state: RunState) -> None:
        self.state = state
        self._on_state(state)

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._resume.set()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="autopilot-run")
        self._set_state("running")
        self._thread.start()

    def pause(self) -> None:
        if self.running and self.state == "running":
            self._resume.clear()
            self._set_state("paused")

    def resume(self) -> None:
        if self.running and self.state == "paused":
            self._resume.set()
            self._set_state("running")

    def stop(self) -> None:
        """Ask the run to end. Returns immediately — the thread notices at its
        next check, which is at most one settle/exposure away."""
        self._stop.set()
        self._resume.set()          # a paused run must be stoppable too
        self._thread = None

    def _wait(self, seconds: float) -> bool:
        """Sleep, returning False if the run should end. Event-based so a stop
        lands promptly instead of after the full duration."""
        return not self._stop.wait(seconds)

    def _run(self) -> None:
        try:
            for i, step in enumerate(self.recipe.steps):
                if self._stop.is_set():
                    self._set_state("stopped")
                    return
                # Block here while paused, but stay stoppable.
                while not self._resume.wait(0.1):
                    if self._stop.is_set():
                        self._set_state("stopped")
                        return

                self.current_step = i
                self._on_step(i, "running")

                if step.kind == "move":
                    self.stage.move_to(step.x, step.y)
                    if not self._wait(0.05):
                        self._set_state("stopped")
                        return
                elif step.kind == "settle":
                    if not self._wait(step.seconds):
                        self._set_state("stopped")
                        return
                elif step.kind == "acquire":
                    frame = self.detector.acquire(self.stage, step.exposure_s)
                    self._on_frame(frame)

                self._on_step(i, "done")

            self.current_step = -1
            self._set_state("done")
        except Exception:
            # A failed run must SAY so. Dying quietly leaves the UI showing
            # "running" forever with no way to tell it apart from a slow step.
            log.exception("recipe run failed")
            self._set_state("failed")
