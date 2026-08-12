"""
instrument.py — the DE Server connection, and the thread that owns it.

`deapi` is **one client, one connection**, and essentially every call blocks
until the server answers. So the honest model is a single connection owned by a
single thread, with every caller submitting work to it and getting a future
back. That is not a lock bolted on afterwards; it is the shape the library
already has.

Two connections — a read-only one for status and rendering, a read/write one for
acquisition — is the better arrangement where the server supports it, and
`Client.connect(..., read_only=True)` exists for exactly that. It is
deliberately NOT built here yet: `deapi`'s bundled `FakeServer` accepts exactly
one connection (a single `accept()` then serve-until-disconnect), so a second
client sits unanswered in the listen backlog until it times out. Since the fake
server is the whole dev and test path, an untestable second connection would be
a liability. :func:`Instrument.open_reader` marks where it goes.

FIFO ordering on one thread is also what a device wants: a property write and
the acquisition that depends on it cannot overtake each other.
"""
from __future__ import annotations

import atexit
import logging
import os
import pathlib
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 13240

#: Set to 1 to spawn deapi's simulated server and talk to that instead of real
#: hardware. What `npm run dev` and the e2e suite use.
FAKE_ENV = "GROUNDCREW_FAKE_SERVER"

#: Override the port. Real hardware only — in fake mode a free port is chosen.
PORT_ENV = "GROUNDCREW_PORT"


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class FakeServerProcess:
    """deapi's simulated DE Server, as a child process.

    Started the way deapi's own test suite starts it: run
    ``simulated_server/initialize_server.py <port>`` unbuffered and wait for the
    ``started`` banner. Unbuffered matters — with stdout buffered the banner can
    sit in the pipe long enough to look like a hung server.
    """

    def __init__(self, port: int) -> None:
        self.port = port
        self._proc: subprocess.Popen | None = None

    def start(self, timeout: float = 60.0) -> None:
        import deapi
        script = pathlib.Path(deapi.__file__).parent / "simulated_server" / "initialize_server.py"
        self._proc = subprocess.Popen(
            [sys.executable, "-u", str(script), str(self.port)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        atexit.register(self.stop)

        opening: list[str] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._proc.stdout.readline() if self._proc.stdout else ""
            if line:
                opening.append(line.rstrip())
            if not line and self._proc.poll() is not None:
                # Quote the child's own output. "exited with 1" says nothing,
                # and the usual cause — the port already held by a stray server
                # from an earlier run — is stated plainly in the traceback the
                # child prints.
                tail = "\n".join(opening[-8:]) or "(no output)"
                raise RuntimeError(
                    f"fake DE server on port {self.port} exited with "
                    f"{self._proc.returncode}:\n{tail}")
            if "started" in line.lower():
                log.info("fake DE server listening on %d", self.port)
                # Drain the rest on a daemon thread: the pipe has a finite
                # buffer, and a server whose stdout fills up blocks mid-request.
                threading.Thread(target=self._drain, daemon=True,
                                 name="fake-de-drain").start()
                return
        raise TimeoutError(f"fake DE server did not start within {timeout}s")

    def _drain(self) -> None:
        try:
            for line in self._proc.stdout:          # type: ignore[union-attr]
                log.debug("[fake-de] %s", line.rstrip())
        except Exception:
            pass

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()


class Instrument:
    """One DE Server connection, and the single thread that may touch it.

    Every public method returns a ``Future``. Nothing here blocks the caller,
    and nothing but the io thread ever touches the client.
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int | None = None) -> None:
        self.host = host
        #: A pinned port is honoured verbatim; the default is only a default,
        #: and in fake mode is replaced by a free one (see `_connect`).
        self._port_pinned = port is not None
        self.port = DEFAULT_PORT if port is None else port
        self.client: Any = None
        self._fake: FakeServerProcess | None = None
        self._closed = False
        # ONE worker: this is the connection's owner, not a pool.
        self._io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="de-io")
        self._io_thread_id: int | None = None
        self._prop_names: set | None = None
        #: (label, started_at) for the call currently on the io thread.
        self._pending: tuple[str, float] | None = None
        #: One outstanding out-of-band stop at a time (see `stop_acquisition`).
        self._stopping = threading.Event()
        self._io.submit(self._mark_thread).result(timeout=10)

    def _mark_thread(self) -> None:
        self._io_thread_id = threading.get_ident()

    @property
    def on_io_thread(self) -> bool:
        """True when the caller is already the connection's owner.

        Load-bearing for :meth:`call`: submitting from the io thread and then
        waiting on the future would deadlock against the single worker.
        """
        return threading.get_ident() == self._io_thread_id

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self, *, fake: bool | None = None) -> Future:
        """Connect, optionally spawning the simulated server first."""
        if fake is None:
            fake = os.environ.get(FAKE_ENV) == "1"
        return self._io.submit(self._connect, bool(fake))

    def _connect(self, fake: bool) -> dict:
        from deapi import Client

        if fake:
            # In fake mode the port is OUR implementation detail — we spawn the
            # server and we are its only client — so take a free one unless the
            # caller pinned it. A fixed default meant one leftover server from
            # an earlier run made every subsequent launch fail to connect, which
            # presents as "Offline" with nothing obviously wrong.
            if not self._port_pinned:
                self.port = _free_port()
            self._fake = FakeServerProcess(self.port)
            self._fake.start()

        client = Client()
        # Shared memory is a same-machine Windows optimisation; off keeps the
        # transport identical across platforms and against the fake server.
        client.usingMmf = False
        client.connect(host=self.host, port=self.port)
        self.client = client

        cameras = client.list_cameras()
        if cameras:
            client.set_current_camera(cameras[0])
        return {
            "cameras": list(cameras),
            "camera": client.get_current_camera() if cameras else None,
            "width": int(client["Image Size X (pixels)"]),
            "height": int(client["Image Size Y (pixels)"]),
            "server": str(client["Server Software Version"]),
            "fake": fake,
        }

    def open_reader(self) -> None:
        """Where the second, read-only connection goes.

        Not implemented: deapi's FakeServer serves exactly one connection, so
        this cannot be exercised anywhere the app is currently tested. Add it
        once there is a server that answers two clients, and move status polling
        and `get_result` onto it — acquisition keeps the read/write one.
        """
        raise NotImplementedError(
            "second connection deferred — deapi's FakeServer accepts only one; "
            "see this module's docstring"
        )

    def close(self) -> None:
        """Disconnect and stop the io thread. Idempotent."""
        if self._closed:
            return
        self._closed = True

        def _disconnect() -> None:
            if self.client is not None:
                try:
                    self.client.disconnect()
                except Exception as e:
                    log.debug("disconnect failed: %s", e)
                self.client = None

        try:
            self._io.submit(_disconnect).result(timeout=10)
        except Exception as e:
            log.debug("disconnect did not complete cleanly: %s", e)
        self._io.shutdown(wait=False, cancel_futures=True)
        if self._fake is not None:
            self._fake.stop()

    # ── Submitting work ───────────────────────────────────────────────────────

    def call(self, fn: Callable[[Any], Any], *, label: str = "") -> Future:
        """Run ``fn(client)`` on the io thread. Returns a Future.

        Calling from the io thread itself runs INLINE and returns a completed
        future — submitting and waiting there would deadlock against the single
        worker, which is exactly what a tile request nested inside another DE
        call would do.
        """
        if self._closed:
            fut: Future = Future()
            fut.set_exception(RuntimeError("instrument is closed"))
            return fut
        if self.on_io_thread:
            fut = Future()
            try:
                fut.set_result(fn(self.client))
            except Exception as e:
                fut.set_exception(e)
            return fut
        return self._io.submit(self._run, fn, label or getattr(fn, "__name__", "call"))

    def _run(self, fn: Callable[[Any], Any], label: str) -> Any:
        """Run one call, recording that it is in flight.

        The record is what lets the rest of the app tell "the camera is slow"
        from "the camera has stopped answering". This is a single connection
        owned by a single thread, so ONE call that never returns freezes every
        other feature, and the status board is the one screen that must still
        answer when that happens.

        (It was written for a hang I had caused myself by putting
        :meth:`stop_acquisition` on this thread — see that method. It is kept
        because a device call that genuinely hangs is still possible, not
        because that original case survives.)
        """
        self._pending = (label, time.monotonic())
        try:
            return fn(self.client)
        finally:
            self._pending = None

    def pending(self) -> tuple[str, float] | None:
        """``(label, seconds_in_flight)`` for the call in flight, or None.

        Safe to read from any thread: the field is only ever rebound, never
        mutated, so a reader sees either the old tuple or the new one.
        """
        p = self._pending
        if p is None:
            return None
        return p[0], time.monotonic() - p[1]

    def stop_acquisition(self) -> None:
        """Send the acquisition-stop command. **Never on the io thread.**

        `stop_acquisition` does not touch the TCP client socket at all — it
        opens its OWN UDP socket, sends ``PyClientStopAcq`` and waits for a
        reply:

            sock = socket.socket(AF_INET, SOCK_DGRAM)
            sock.sendto(b"PyClientStopAcq", (host, port))
            respond = sock.recv(32)          # no timeout

        So it needs no serialising, and deapi's own docstring says it may be
        called from another thread while `get_result` blocks — that is the
        point of it being out-of-band.

        Putting it on the io thread was a mistake that cost a day. That `recv`
        has no timeout, and deapi's `FakeServer` answers only TCP, so the call
        never returns there — and because the io thread owns the connection,
        one parked stop made every later read queue behind it. The connection
        was fine throughout: with the stop blocked on its own thread, property
        reads and a full acquisition both still succeed.

        At most one stop is outstanding: they are idempotent, and if one is
        already parked on an unanswered `recv`, spawning more threads that will
        also never return achieves nothing.
        """
        if self._closed or self.client is None or self._stopping.is_set():
            return
        self._stopping.set()

        def _send() -> None:
            try:
                self.client.stop_acquisition()
            except Exception as e:
                log.debug("stop_acquisition failed: %s", e)
            finally:
                self._stopping.clear()

        threading.Thread(target=_send, daemon=True, name="de-stop").start()

    def call_nowait(self, fn: Callable[[Any], Any], label: str = "") -> None:
        """Submit and do not wait, swallowing the result.

        For a command whose completion the UI must not depend on — Stop being
        the one that matters. The user asked for the live view to stop; that
        must happen whether or not the camera acknowledges.
        """
        try:
            self.call(fn, label=label).add_done_callback(lambda f: f.exception())
        except Exception as e:
            log.debug("submitting %s failed: %s", label, e)

    def get(self, prop: str) -> Future:
        return self.call(lambda c: c[prop])

    def set(self, prop: str, value: Any) -> Future:
        return self.call(lambda c: c.__setitem__(prop, value))

    def property_names(self) -> Future:
        """The camera's property names, cached after the first read."""
        def _list(c: Any) -> set:
            if self._prop_names is None:
                self._prop_names = set(c.list_properties() or ())
            return self._prop_names
        return self.call(_list)

    def properties(self, names: list[str]) -> Future:
        """Read several properties in ONE trip to the io thread.

        A status board wants a dozen at once; a dozen separate futures would
        interleave with acquisition work and take a dozen queue slots.

        Unknown names come back as ``None``, checked against the camera's own
        property list rather than by inspecting the value. That distinction
        matters: ``client[unknown]`` returns ``False`` — the library's
        command-failed sentinel — and ``False`` is also a perfectly good value
        for a boolean property, so a status board that treated it as "missing"
        would silently mis-report every switch that happens to be off.
        """
        def _read(c: Any) -> dict:
            if self._prop_names is None:
                try:
                    self._prop_names = set(c.list_properties() or ())
                except Exception as e:
                    log.debug("listing properties failed: %s", e)
                    self._prop_names = set()
            known = self._prop_names
            out: dict[str, Any] = {}
            for n in names:
                if known and n not in known:
                    out[n] = None
                    continue
                try:
                    out[n] = c[n]
                except Exception as e:
                    log.debug("reading %s failed: %s", n, e)
                    out[n] = None
            return out
        return self.call(_read)
