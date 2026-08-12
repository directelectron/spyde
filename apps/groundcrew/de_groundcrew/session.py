"""
session.py — Ground Crew's backend coordinator.

A thin app layer on `de_shell.session.SessionBase`: the base owns the window
registry, main-thread marshalling and the settings file, and
`de_shell.plotting.figure.FigureView` owns the image pane. What is left here is
the DE Server connection and the acquisition loop.

## The viewer is PULL, not push

The image pane is not fed frames. `DeapiTileBackend` is handed to
`FigureView.enable_tile`, and from then on anyplotlib asks the server for the
region on screen at the resolution it needs — zoom and pan go straight down the
wire as `get_result` arguments. Live acquisition therefore does not paint; it
calls :meth:`_refresh` , which tells anyplotlib to re-read the current view. The
user's zoom survives every new frame, and only the visible pixels cross the
wire. On a 4096² detector that is the difference between usable and not.

`FrameStream` is still here, but for the REFRESH, not for pixels: it collapses a
burst of "new frame available" notifications into one re-read, newest wins. A
server slower than the frame rate falls behind by one refresh rather than
queueing an unbounded backlog.

Actions arrive from the renderer as `{"type": "action", "action": …, "payload":
…}` and are dispatched by name through `_ACTIONS`. That is deliberately a plain
dict rather than SpyDE's YAML-driven registry — Ground Crew has a handful of
fixed controls, and the staged-wizard machinery would be scaffolding around
nothing. If it grows, `de_shell.actions.registry` is there to adopt.
"""
from __future__ import annotations

import logging
import threading
import time

from de_shell.ipc import emit, emit_error, emit_status
from de_shell.plotting.figure import FigureView
from de_shell.session import SessionBase

from de_groundcrew.instrument import Instrument
from de_groundcrew.tile import DeapiTileBackend

log = logging.getLogger(__name__)

#: Properties the header reads. Deliberately short: every name here is a round
#: trip on the one connection, and the simulator answers only a fraction of
#: them (see `test_the_simulator_lacks_most_status_properties`). Unsupported
#: names come back None and the UI shows them as unavailable rather than
#: inventing a value.
HEADER_PROPS = (
    "Frames Per Second",
    "Exposure Time (seconds)",
    "Image Size X (pixels)",
    "Image Size Y (pixels)",
    "Binning X",
    "Temperature - Detector (Celsius)",
    "Temperature - Detector Status",
    "Camera Position Status",
)


class GroundCrewSession(SessionBase):
    """One instance per app lifetime."""

    def __init__(self, settings_dir: str) -> None:
        super().__init__(settings_dir=settings_dir)

        self.instrument = Instrument()
        self.info: dict = {}
        self.tiles: DeapiTileBackend | None = None

        self._viewer = FigureView(self.next_window_id(), title="Live view")
        self.register_window_controller(self._viewer.window_id, self._viewer)

        self._live = threading.Event()
        self._live_thread: threading.Thread | None = None
        self._refresh_pending = False
        self._levels: tuple[float, float] | None = None

    # ── Startup ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Connect, then open the viewer once the sensor size is known.

        The connection is a blocking socket dial, so it runs on the io thread
        and the window opens from its completion callback. Opening first with a
        guessed shape would make the pane resize the moment the real size
        arrives, which reads as the window jumping.
        """
        emit_status("Connecting to DE Server…")
        fut = self.instrument.connect()
        fut.add_done_callback(
            lambda f: self._dispatch_to_main(lambda: self._on_connected(f)))

    def _on_connected(self, fut) -> None:
        try:
            self.info = fut.result()
        except Exception as e:
            log.exception("connection failed")
            emit_error(f"Could not reach the DE Server: {e}")
            emit({"type": "connection", "connected": False, "error": str(e)})
            return

        h, w = int(self.info["height"]), int(self.info["width"])
        self._viewer.open((h, w))
        self.tiles = DeapiTileBackend(self.instrument, (h, w))

        if not self._viewer.enable_tile(self.tiles):
            emit_error("Tiled display unavailable — the viewer needs anyplotlib "
                       "with tile support")
            return

        emit({
            "type": "connection", "connected": True,
            "camera": self.info.get("camera"), "server": self.info.get("server"),
            "width": w, "height": h, "fake": self.info.get("fake", False),
        })
        emit_status(f"{self.info.get('camera') or 'Camera'} ready — {w}×{h}")
        self._refresh()          # first read, so the pane is not left empty
        self._emit_properties()

    # ── Display ───────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        """Re-read the current view from the server. **Main thread only.**

        Coalescing is by a single boolean rather than a queue: while a refresh
        is outstanding, further requests are dropped instead of stacking up.
        Newest-wins on a pull source needs nothing more — the next read returns
        the latest frame anyway, so a dropped request loses nothing.
        """
        if not self._viewer.is_tiled or self._refresh_pending:
            return
        self._refresh_pending = True
        try:
            self._viewer.refresh_tile()
            self._apply_levels()
            self._emit_stats()
        finally:
            self._refresh_pending = False

    def _apply_levels(self) -> None:
        """Track the display range to the scene, without flickering.

        A tiled plot keeps whatever range its placeholder established unless
        told otherwise, so this is not optional — left alone a live frame
        renders as a flat white panel.

        Re-applied only when the range moves by more than a fifth of its own
        span: setting it every frame would let shot noise visibly pump the
        contrast on a live view, while never re-setting it would leave the
        display wrong after an exposure change.
        """
        levels = (self.tiles.last_stats or {}).get("levels") if self.tiles else None
        if not levels:
            return
        lo, hi = float(levels[0]), float(levels[1])
        if self._levels is not None:
            span = max(self._levels[1] - self._levels[0], 1e-9)
            if (abs(lo - self._levels[0]) < 0.2 * span
                    and abs(hi - self._levels[1]) < 0.2 * span):
                return
        if self._viewer.set_clim(lo, hi):
            self._levels = (lo, hi)

    def _emit_stats(self) -> None:
        """The stats strip under the image.

        Free: these came back with the pixels on the same `get_result`, so the
        strip costs no extra round trip.
        """
        if self.tiles is None or not self.tiles.last_stats:
            return
        emit({"type": "frame_stats", **self.tiles.last_stats})

    def _emit_properties(self) -> None:
        fut = self.instrument.properties(list(HEADER_PROPS))
        fut.add_done_callback(
            lambda f: self._dispatch_to_main(lambda: self._on_properties(f)))

    def _on_properties(self, fut) -> None:
        try:
            props = fut.result()
        except Exception as e:
            log.debug("property poll failed: %s", e)
            return
        emit({"type": "properties", "values": props})

    # ── Acquisition ───────────────────────────────────────────────────────────

    def _act_start(self, payload: dict) -> None:
        """Free-run: acquire continuously, refreshing the view as frames land."""
        if self._live.is_set():
            return
        self._live.set()
        self._live_thread = threading.Thread(
            target=self._live_loop, daemon=True, name="groundcrew-live")
        self._live_thread.start()
        emit_status("Acquiring")
        self._emit_acq_state()

    def _live_loop(self) -> None:
        """Poll the server for new frames and ask the main thread to re-read.

        Not a paint loop — see the module docstring. It runs off the io thread
        deliberately: `instrument.call` serialises onto the connection, so this
        interleaves with the tile reads rather than blocking them.
        """
        try:
            self.instrument.call(lambda c: c.start_acquisition(0)).result(timeout=30)
        except Exception as e:
            log.exception("start_acquisition failed")
            self._live.clear()
            self._dispatch_to_main(lambda: emit_error(f"Could not start: {e}"))
            self._dispatch_to_main(self._emit_acq_state)
            return

        while self._live.is_set():
            self._dispatch_to_main(self._refresh)
            time.sleep(0.05)

        try:
            self.instrument.call(lambda c: c.stop_acquisition()).result(timeout=30)
        except Exception as e:
            log.debug("stop_acquisition failed: %s", e)

    def _act_stop(self, payload: dict) -> None:
        self._live.clear()
        emit_status("Stopped")
        self._emit_acq_state()

    def _act_single(self, payload: dict) -> None:
        """One exposure, off the main thread so a long one cannot freeze the UI."""
        if self._live.is_set():
            emit_status("Stop the live view before a single acquisition")
            return

        def _done(fut) -> None:
            def _apply() -> None:
                try:
                    fut.result()
                except Exception as e:
                    log.exception("single acquisition failed")
                    emit_error(f"Acquisition failed: {e}")
                    return
                self._refresh()
            self._dispatch_to_main(_apply)

        def _grab(c) -> None:
            c.start_acquisition(1)
            while c.acquiring:
                time.sleep(0.01)

        emit_status("Acquiring one frame…")
        self.instrument.call(_grab).add_done_callback(_done)

    # ── Property writes ───────────────────────────────────────────────────────

    def _act_set_property(self, payload: dict) -> None:
        """Generic property write — the whole camera control surface.

        One action rather than one per control: `deapi` addresses everything by
        name, so a per-property action would be a table of identical bodies.
        """
        name = payload.get("name")
        if not name:
            emit_error("set_property without a name")
            return
        value = payload.get("value")

        def _done(fut) -> None:
            def _apply() -> None:
                try:
                    fut.result()
                except Exception as e:
                    emit_error(f"Could not set {name}: {e}")
                    return
                emit_status(f"{name} = {value}")
                self._emit_properties()
            self._dispatch_to_main(_apply)

        self.instrument.set(str(name), value).add_done_callback(_done)

    def _act_refresh_properties(self, payload: dict) -> None:
        self._emit_properties()

    def _act_set_colormap(self, payload: dict) -> None:
        self._viewer.set_colormap(str(payload.get("name", "gray")))

    def _emit_acq_state(self) -> None:
        emit({"type": "acq_state", "running": self._live.is_set()})

    # ── Actions ───────────────────────────────────────────────────────────────

    def dispatch_action(self, msg: dict) -> None:
        """Route `{"type": "action", ...}` from the renderer. Called by
        `de_shell.app`'s loop on the main thread."""
        name = str(msg.get("action", ""))
        handler = _ACTIONS.get(name)
        if handler is None:
            log.warning("unknown action: %s", name)
            return
        handler(self, msg.get("payload") or {})

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        # Acquisition first: it is the only thing still producing work, and
        # stopping it before the viewer means no refresh can be requested for a
        # window that has already gone.
        self._live.clear()
        if self._live_thread is not None:
            self._live_thread.join(timeout=5)
        try:
            self._viewer.close()
        except Exception as e:
            log.debug("viewer close failed: %s", e)
        # The connection last — the live loop's stop_acquisition needs it.
        self.instrument.close()
        super().shutdown()


_ACTIONS = {
    "start_acquisition": GroundCrewSession._act_start,
    "stop_acquisition": GroundCrewSession._act_stop,
    "single_acquisition": GroundCrewSession._act_single,
    "set_property": GroundCrewSession._act_set_property,
    "refresh_properties": GroundCrewSession._act_refresh_properties,
    "set_colormap": GroundCrewSession._act_set_colormap,
}
