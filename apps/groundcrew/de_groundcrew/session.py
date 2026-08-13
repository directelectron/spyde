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
import os
import threading
import time

from de_shell.ipc import emit, emit_error, emit_status
from de_shell.plotting.figure import FigureView
from de_shell.session import SessionBase

from de_groundcrew import instrument as instrument_mod
from de_groundcrew import status
from de_groundcrew.instrument import Instrument
from de_groundcrew.tile import DeapiTileBackend

log = logging.getLogger(__name__)

#: How long the live loop waits for one display refresh before giving up on it.
#: Generous — a cold read of a large frame over a slow link is legitimately
#: seconds — but finite, so a wedged server stops the loop instead of leaving a
#: thread parked forever.
REFRESH_TIMEOUT_S = 15.0

#: How long a status read may take before the board reports the camera as
#: unresponsive instead of waiting. Short on purpose: an engineer opening this
#: screen has usually done so BECAUSE something is wrong.
STATUS_DEADLINE_S = 6.0

#: A call already in flight for longer than this means the connection is stuck,
#: so a new read would only queue behind it.
STALL_S = 4.0

#: Properties the instrument sidebar and top bar read. Deliberately short:
#: every name is a round trip on the one connection, and the simulator answers
#: only a fraction of them. Unsupported names come back None and the UI shows
#: them as unavailable rather than inventing a value.
HEADER_PROPS = (
    # Acquisition — the instrument sidebar's editable fields.
    "Frames Per Second",
    "Exposure Time (seconds)",
    "Frame Count",
    "Binning X",
    "Hardware Binning X",
    "Image Size X (pixels)",
    "Image Size Y (pixels)",
    "Autosave Directory",
    # Hardware state — the top bar. Absent on the simulator, which is the
    # point: the controls disable rather than vanish.
    "Temperature - Detector (Celsius)",
    "Temperature - Cool Down Setpoint (Celsius)",
    "Camera Position Status",
)


def _port_from_env() -> int | None:
    """`GROUNDCREW_PORT`, or None to let the Instrument decide.

    None is not the same as the default: in fake mode an unpinned port becomes
    a free one, so a leftover simulated server from an earlier run cannot make
    the next launch fail to connect.
    """
    raw = os.environ.get(instrument_mod.PORT_ENV)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        log.warning("ignoring non-numeric %s=%r", instrument_mod.PORT_ENV, raw)
        return None


class GroundCrewSession(SessionBase):
    """One instance per app lifetime."""

    def __init__(self, settings_dir: str) -> None:
        super().__init__(settings_dir=settings_dir)

        self.instrument = Instrument(port=_port_from_env())
        self.info: dict = {}
        self.tiles: DeapiTileBackend | None = None

        self._viewer = FigureView(self.next_window_id(), title="Live view")
        self.register_window_controller(self._viewer.window_id, self._viewer)

        self._live = threading.Event()
        #: A single exposure in flight. Separate from `_live` so Stop can end
        #: either, and so the UI can say which is running.
        self._single = threading.Event()
        self._live_thread: threading.Thread | None = None
        self._refresh_pending = False
        self._levels: tuple[float, float] | None = None
        #: True once the user typed a range; stops the histogram overriding it.
        self._clim_manual = False
        #: Motion mode's controller — see the `motion` property.
        self._motion = None

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

    def _refresh_and_signal(self, done: threading.Event) -> None:
        """`_refresh`, then release the live loop. Signals even on failure —
        a loop waiting on an event that a raised exception skipped would hang
        until the timeout on every frame."""
        try:
            self._refresh()
        finally:
            done.set()

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
        if self._clim_manual:
            return
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
        stats = dict(self.tiles.last_stats)
        # Report the range that is actually APPLIED. `last_stats["levels"]` is
        # the histogram's robust range, which is only the display range while
        # the range is automatic — once someone has set one by hand, echoing
        # the auto value would snap the histogram handles back under them on
        # the next frame.
        if self._levels is not None:
            stats["levels"] = list(self._levels)
        emit({"type": "frame_stats", **stats})

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
            # Wait for each refresh to LAND before asking for another.
            #
            # Not politeness — correctness. `_refresh` runs on the main thread
            # and blocks it for a whole server round trip, so posting on a
            # fixed 50 ms tick enqueues work faster than it can be drained
            # whenever the server is slower than the tick. The backlog outlives
            # the acquisition: after Stop, the main loop is still working
            # through queued refreshes and every other message — status reports,
            # property reads — waits behind them. That is what it looked like,
            # too: the status board sat on "Reading camera status…" but only
            # when a live view had run first.
            done = threading.Event()
            self._dispatch_to_main(lambda ev=done: self._refresh_and_signal(ev))
            if not done.wait(timeout=REFRESH_TIMEOUT_S):
                log.warning("display refresh did not complete within %ss",
                            REFRESH_TIMEOUT_S)
                break
            time.sleep(0.01)

        # Out-of-band UDP, off the io thread — see Instrument.stop_acquisition.
        self.instrument.stop_acquisition()

    def _act_stop(self, payload: dict) -> None:
        """Stop whichever acquisition is running — live OR single.

        A single exposure can be long (a 60 s integration is ordinary), so it
        needs a way out just as much as a live view does. Both stop the same
        way, because it IS the same thing to the camera: one out-of-band UDP
        command. The only difference is which local loop notices.
        """
        if not (self._live.is_set() or self._single.is_set()):
            return
        self._live.clear()
        self._single.clear()
        self.instrument.stop_acquisition()
        emit_status("Stopped")
        self._emit_acq_state()

    def _act_single(self, payload: dict) -> None:
        """One exposure, off the main thread so a long one cannot freeze the UI."""
        if self._live.is_set() or self._single.is_set():
            emit_status("An acquisition is already running")
            return

        self._single.set()
        self._emit_acq_state()

        def _done(fut) -> None:
            def _apply() -> None:
                self._single.clear()
                self._emit_acq_state()
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
            # Watch the STOP FLAG as well as the camera. `stop_acquisition` is
            # a UDP command whose reply the simulator never sends, so waiting
            # only on `c.acquiring` would leave this loop spinning on a server
            # that never noticed — and this loop holds the io thread.
            while c.acquiring and self._single.is_set():
                time.sleep(0.01)

        emit_status("Acquiring one frame…")
        self.instrument.call(_grab, label="single_acquisition").add_done_callback(_done)

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

    def _act_refresh_status(self, payload: dict) -> None:
        """Read the status board's properties in ONE batch and judge them.

        The read is batched because every name is a round trip on the single
        connection; the judging is a pure function, so it is unit-tested
        without a server.
        """
        # A board that spins forever is the worst outcome: the one moment an
        # engineer needs it is when the camera has stopped answering. So the
        # read is raced against a deadline, and whichever resolves first wins.
        sent = threading.Event()

        def _publish(values: dict, link: dict) -> None:
            if sent.is_set():
                return
            sent.set()
            report = status.build_report(values, link=link)
            log.info("status: %s, %d of %d reporting", report["summary"]["overall"],
                     report["summary"]["reporting"], report["summary"]["total"])
            emit({"type": "status_report", **report})

        stalled = self.instrument.pending()
        if stalled and stalled[1] > STALL_S:
            # Do not even queue the read — it would sit behind the stuck call.
            # This verdict needs no device access, which is exactly why the
            # board can still answer when nothing else can.
            _publish({}, {"state": status.BAD,
                          "detail": f"no response to {stalled[0]} for {stalled[1]:.0f}s"})
            return

        log.info("status: reading %d properties", len(status.STATUS_PROPS))
        fut = self.instrument.properties(list(status.STATUS_PROPS))

        def _on_read(f) -> None:
            def _apply() -> None:
                try:
                    values = f.result()
                except Exception as e:
                    log.debug("status poll failed: %s", e)
                    _publish({}, {"state": status.BAD, "detail": f"read failed: {e}"})
                    return
                _publish(values, {"state": status.OK, "detail": "responding"})
            self._dispatch_to_main(_apply)

        def _on_deadline() -> None:
            def _apply() -> None:
                p = self.instrument.pending()
                where = f" (waiting on {p[0]})" if p else ""
                _publish({}, {"state": status.BAD,
                              "detail": f"no response within {STATUS_DEADLINE_S:.0f}s{where}"})
            self._dispatch_to_main(_apply)

        fut.add_done_callback(_on_read)
        # A one-shot timer rather than a polling thread: this fires only while a
        # status read is outstanding, and is cancelled the moment one lands.
        timer = threading.Timer(STATUS_DEADLINE_S, _on_deadline)
        timer.daemon = True
        timer.start()
        fut.add_done_callback(lambda _f: timer.cancel())

    # ── Motion correction ─────────────────────────────────────────────────────

    @property
    def motion(self):
        """The Motion controller, created on first use.

        Lazy because it opens two figures and most sessions never leave
        Imaging — building it at startup would put two empty panes into every
        run and load numpy/scipy machinery nobody asked for.
        """
        if self._motion is None:
            from de_groundcrew.motion_session import MotionController
            self._motion = MotionController(self)
        return self._motion

    def _act_motion_state(self, payload: dict) -> None:
        self.motion._emit_state()

    def _act_motion_open_stack(self, payload: dict) -> None:
        path = str(payload.get("path", ""))
        if not path:
            emit_error("No movie file given")
            return
        self.motion.open_stack(path)

    def _act_motion_load_test_stack(self, payload: dict) -> None:
        self.motion.load_test_stack(
            n_frames=int(payload.get("n_frames", 6)),
            size=int(payload.get("size", 128)))

    def _act_motion_open_gain(self, payload: dict) -> None:
        path = str(payload.get("path", ""))
        if not path:
            emit_error("No gain file given")
            return
        self.motion.open_gain(path)

    def _act_motion_validate_gain(self, payload: dict) -> None:
        self.motion.validate_gain()

    def _act_motion_set_orientation(self, payload: dict) -> None:
        try:
            self.motion.orientation = int(payload["index"])
        except (KeyError, TypeError, ValueError):
            emit_error(f"Invalid gain orientation: {payload!r}")
            return
        self.motion._emit_state()

    def _act_motion_align(self, payload: dict) -> None:
        try:
            self.motion.align(
                mode=str(payload.get("mode", "fast")),
                throw=int(payload.get("throw", 0)),
                local=bool(payload.get("local", False)),
                patch_size=int(payload.get("patch_size", 512)),
                bin_factor=int(payload.get("bin_factor", 2)),
                apix=float(payload.get("apix", 1.0)))
        except (TypeError, ValueError) as e:
            emit_error(f"Invalid alignment settings: {e}")

    def _act_motion_stop(self, payload: dict) -> None:
        self.motion.cancel()
        emit_status("Stopping…")

    def _act_motion_save(self, payload: dict) -> None:
        path = str(payload.get("path", ""))
        if not path:
            emit_error("No output path given")
            return
        self.motion.save(path)

    def _act_motion_set_frame(self, payload: dict) -> None:
        self.motion.set_frame(int(payload.get("frame", 0)))

    def _act_motion_set_view(self, payload: dict) -> None:
        self.motion.set_view(str(payload.get("view", "raw")))

    def _act_set_colormap(self, payload: dict) -> None:
        self._viewer.set_colormap(str(payload.get("name", "gray")))

    def _act_set_clim(self, payload: dict) -> None:
        """Pin the display range by hand.

        Once set explicitly it STAYS set: `_apply_levels` only re-derives from
        the histogram while the range is automatic. Someone who typed a range
        to compare two frames does not want shot noise moving it back.
        """
        try:
            lo, hi = float(payload["low"]), float(payload["high"])
        except (KeyError, TypeError, ValueError):
            emit_error(f"Invalid display range: {payload!r}")
            return
        if hi <= lo:
            emit_error(f"Display range must increase: {lo} … {hi}")
            return
        if self._viewer.set_clim(lo, hi):
            self._levels = (lo, hi)
            self._clim_manual = True
            emit_status(f"Display range {lo:g} … {hi:g}")

    def _act_auto_clim(self, payload: dict) -> None:
        """Hand the display range back to the histogram."""
        self._clim_manual = False
        self._levels = None
        self._apply_levels()
        emit_status("Display range: auto")

    def _emit_acq_state(self) -> None:
        live, single = self._live.is_set(), self._single.is_set()
        emit({"type": "acq_state", "running": live or single,
              "mode": "live" if live else "single" if single else None})

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
        if self._motion is not None:
            self._motion.close()
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
    "refresh_status": GroundCrewSession._act_refresh_status,
    "set_colormap": GroundCrewSession._act_set_colormap,
    "set_clim": GroundCrewSession._act_set_clim,
    "motion_state": GroundCrewSession._act_motion_state,
    "motion_open_stack": GroundCrewSession._act_motion_open_stack,
    "motion_load_test_stack": GroundCrewSession._act_motion_load_test_stack,
    "motion_open_gain": GroundCrewSession._act_motion_open_gain,
    "motion_validate_gain": GroundCrewSession._act_motion_validate_gain,
    "motion_set_orientation": GroundCrewSession._act_motion_set_orientation,
    "motion_align": GroundCrewSession._act_motion_align,
    "motion_stop": GroundCrewSession._act_motion_stop,
    "motion_save": GroundCrewSession._act_motion_save,
    "motion_set_frame": GroundCrewSession._act_motion_set_frame,
    "motion_set_view": GroundCrewSession._act_motion_set_view,
    "auto_clim": GroundCrewSession._act_auto_clim,
}
