"""
camera.py — the camera behind the viewer.

Two implementations behind one interface. `SimulatedCamera` is what runs today
and on any dev machine: it synthesises frames so the UI, the IPC transport and
the acquisition loop can be built and tested with no hardware attached.
`DEServerCamera` is the seam where the real DE Server SDK (`sdk/DEAPI.py` in the
PySide6 app, always 127.0.0.1:13240) plugs in.

Frames are plain numpy arrays held in RAM. There is deliberately no lazy-read
machinery here — a live camera has no file to be lazy about, which is exactly
why this app does not need SpyDE's array cache.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class Camera(Protocol):
    """What the session needs from a camera, and nothing more."""

    @property
    def shape(self) -> tuple[int, int]: ...

    def acquire(self, exposure_s: float) -> np.ndarray:
        """Return one frame. Blocking — callers run it off the main thread."""
        ...


class SimulatedCamera:
    """Synthetic frames: a drifting Airy-ish spot on a noise floor.

    The content is deliberately asymmetric and time-varying, for the same reason
    SpyDE's synthetic movie is: a static or symmetric test pattern hides stale
    frames, mirrored axes and dropped updates, and those are precisely the bugs a
    live viewer gets wrong.
    """

    def __init__(self, shape: tuple[int, int] = (512, 512), seed: int = 0) -> None:
        self._shape = shape
        self._rng = np.random.default_rng(seed)
        self._frame_index = 0
        self._t0 = time.monotonic()

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def frame_index(self) -> int:
        return self._frame_index

    def acquire(self, exposure_s: float = 0.05) -> np.ndarray:
        h, w = self._shape
        t = time.monotonic() - self._t0

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # Beam drifts on a slow Lissajous so successive frames differ in a way
        # the eye can actually follow.
        cy = h * (0.5 + 0.18 * np.sin(t * 0.7))
        cx = w * (0.5 + 0.18 * np.cos(t * 0.4))
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        beam = 3000.0 * np.exp(-r2 / (2.0 * (min(h, w) * 0.06) ** 2))

        # Longer exposure = more signal AND less relative noise, so the exposure
        # control visibly does something.
        gain = max(exposure_s, 1e-3) / 0.05
        noise = self._rng.poisson(40.0 * gain, size=(h, w)).astype(np.float32)

        # A fixed corner block: an unmistakable orientation marker. If the
        # renderer ever flips an axis, this moves and you see it instantly.
        frame = beam * gain + noise
        frame[: h // 16, : w // 16] += 2500.0

        self._frame_index += 1
        return np.clip(frame, 0, 65535).astype(np.uint16)


class DEServerCamera:
    """Real hardware, via the DE Server SDK on 127.0.0.1:13240.

    Not implemented yet — this is the seam. The PySide6 Ground Crew talks to the
    server through `sdk/DEAPI.py`; porting it means constructing the client here
    and mapping `acquire` onto its image grab. Everything above this line (the
    session, the IPC, the viewer) is already agnostic to which camera it holds.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 13240) -> None:
        self._host, self._port = host, port
        raise NotImplementedError(
            "DEServerCamera is not wired up yet — run with the simulated camera. "
            "Porting it means bringing sdk/DEAPI.py over from the PySide6 app."
        )


class AcquisitionLoop:
    """Free-runs a camera on a daemon thread, handing each frame to a callback.

    The callback is invoked on THIS thread, so whatever it does must be safe off
    the main thread — the session's callback marshals onto the asyncio loop via
    `SessionBase._dispatch_to_main` before touching a figure.

    `stop()` is idempotent and does not block on the in-flight exposure: a long
    exposure would otherwise hold up shutdown for its whole duration.
    """

    def __init__(self, camera: Camera, on_frame, exposure_s: float = 0.05) -> None:
        self._camera = camera
        self._on_frame = on_frame
        self._exposure_s = exposure_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_exposure(self, exposure_s: float) -> None:
        self._exposure_s = max(float(exposure_s), 1e-3)

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="groundcrew-acquire")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._camera.acquire(self._exposure_s)
                self._on_frame(frame)
            except Exception:
                # A single bad exposure must not kill the acquisition thread —
                # the user would see a viewer that silently stopped updating
                # with no way to restart it short of relaunching.
                log.exception("frame acquisition failed; continuing")
            # Wait rather than sleep so stop() takes effect promptly.
            self._stop.wait(self._exposure_s)

    def stop(self) -> None:
        self._stop.set()
        self._thread = None
