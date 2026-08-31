"""
Qt-free pytest fixtures for the Electron/anyplotlib backend.

Replaces the old pytest-qt conftest. Each fixture builds a real Session (with
Dask skipped via SPYDE_NO_DASK) and synthetic data, captures every PLOTAPP
message both spyde and anyplotlib emit, and yields a dict mirroring the old
fixtures' shape: ``window`` (the Session), ``signal_trees``, ``plots``, and
``messages`` (the captured emit list).
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
from typing import Iterator

import numpy as np
import hyperspy.api as hs
import pytest

os.environ.setdefault("SPYDE_NO_DASK", "1")
# Point settings.json at a throwaway dir for the whole test session (the
# mechanism Session documents for exactly this). Several tests assert on
# DEFAULTS — first_run, update_channel — and a Session reads the real
# ~/.spyde/settings.json at construction, so those tests fail on any machine
# that has actually RUN the app: a dismissed welcome tour persists
# tutorial_seen, an rc build persists update_channel=beta. A fresh dir per
# pytest process also stops a test that persists a setting from poisoning the
# next run. setdefault, so a developer can still point it somewhere on purpose.
os.environ.setdefault(
    "SPYDE_SETTINGS_DIR", tempfile.mkdtemp(prefix="spyde-test-settings-"))


@pytest.fixture
def captured_messages(monkeypatch):
    """Capture both PLOTAPP channels (spyde.ipc and anyplotlib._electron)."""
    import de_shell.ipc as ipc
    import anyplotlib._electron as ael

    msgs: list[dict] = []

    def cap(obj):
        msgs.append(obj)

    monkeypatch.setattr(ipc, "emit", cap)
    monkeypatch.setattr(ael, "emit", cap)
    # session.py binds `emit` at module import (`from ...ipc import emit`), so
    # the ipc patch above doesn't reach it — patch that binding too.
    import spyde.backend.session as sess_mod
    if hasattr(sess_mod, "emit"):
        monkeypatch.setattr(sess_mod, "emit", cap)
    return msgs


def _make_session():
    from spyde.backend.session import Session
    session = Session(n_workers=1, threads_per_worker=1)
    _attach_main_loop(session)
    return session


def _attach_main_loop(session) -> None:
    """Give the session the asyncio loop the real app gives it.

    Without one, ``Session._dispatch_to_main`` falls back to running the
    callback INLINE — on whichever worker thread produced the result. The app
    never does that: ``backend/app.py`` registers a loop, so every worker
    result is ``call_soon_threadsafe``'d and they run one at a time, on one
    thread, interleaved with nothing.

    So a suite with no loop does not test the app's concurrency; it tests a
    more concurrent program that does not exist. Two callbacks that can only
    run in sequence in the app run simultaneously here, and a callback races
    the test body itself. That has cost real time in both directions: races
    reported as failures that no user could hit, and — the expensive
    direction — a genuine race that only reproduces on a loaded CI runner,
    where the reflex is to call it flaky.

    The loop runs on its own thread because the test body owns the main one.
    That leaves ONE difference from the app: a test calls action handlers
    directly rather than through the loop, so handler-vs-callback interleaving
    is still possible where the app has none. Closing that would mean routing
    every call site through the loop; the win here is removing the
    callback-vs-callback concurrency, which is what actually bit.
    """
    loop = asyncio.new_event_loop()
    running = threading.Event()

    def _run():
        asyncio.set_event_loop(loop)
        loop.call_soon(running.set)
        loop.run_forever()

    thread = threading.Thread(target=_run, name="spyde-test-loop", daemon=True)
    thread.start()
    if not running.wait(10.0):
        raise RuntimeError("the test event loop never started")
    session.set_main_loop(loop)
    session._test_loop, session._test_loop_thread = loop, thread


def _teardown(session) -> None:
    """Shut the session down, then the loop it marshals onto — in that order.

    Backwards, and ``shutdown``'s own callbacks land on a dead loop: they
    raise inside ``_dispatch_to_main``, which swallows them and runs the work
    inline instead, back on the very thread this exists to keep work off.
    """
    from spyde.tests.migrated._async import drain_loop

    session.shutdown()
    drain_loop(session, timeout=5.0)
    loop = getattr(session, "_test_loop", None)
    thread = getattr(session, "_test_loop_thread", None)
    if loop is None:
        return
    session._main_loop = None
    loop.call_soon_threadsafe(loop.stop)
    if thread is not None:
        thread.join(timeout=5.0)
    loop.close()


def _settled(session) -> bool:
    """Have the selector debounce/settle timers fired and their updates landed?"""
    from spyde.drawing.selectors import base_selector as bs

    # A queued nav update is still to run (the dispatcher is latest-wins, so a
    # non-empty pending map means work is outstanding).
    if bs._nav_dispatcher._pending:
        return False
    # Any selector with a live settle timer is still going to re-fire.
    for _wid, sel in getattr(session, "_nav_selectors", {}).items():
        for s in (sel, getattr(sel, "selector", None)):
            if s is not None and getattr(s, "_settle_timer", None) is not None:
                return False
    # And every signal plot has actually painted something.
    plots = [p for p in session._plots if not getattr(p, "is_navigator", False)]
    return bool(plots) and all(
        getattr(p, "current_data", None) is not None for p in plots)


def _settle(session, timeout: float = 3.0) -> None:
    """Wait for the selector debounce instead of sleeping a flat 0.8s.

    This was `time.sleep(0.8)`, and the fixtures below are used ~930 times
    across the suite — about 12 minutes of the run spent sleeping. The timer
    it was waiting on is armed for `max(0.12, live_delay + 0.1)`
    (base_selector._arm_settle), so 0.8s was ~6x the margin actually needed.

    The floor matters as much as the condition: at the instant _add_signal
    returns, the timer may not be armed yet and no plot has painted, so a
    pure condition poll could return immediately and hand the test a session
    that has not settled at all. Never return before one full settle period
    has passed, then wait for the real state. Falls through after `timeout`
    rather than hanging, so a fixture can still fail loudly in its test.
    """
    start = time.monotonic()
    floor = start + 0.13          # one settle period; the timer cannot beat it
    deadline = start + timeout
    while time.monotonic() < deadline:
        if time.monotonic() >= floor and _settled(session):
            return
        time.sleep(0.005)


def _load(session, signal):
    session._add_signal(signal, source_path=None)
    _settle(session)


def _bright_disk_4d(nav, sig=(16, 16)):
    data = np.zeros(nav + sig, dtype=np.float32)
    yy, xx = np.mgrid[0:sig[0], 0:sig[1]]
    disk = ((xx - sig[1] // 2) ** 2 + (yy - sig[0] // 2) ** 2 <= 9).astype(np.float32)
    it = np.ndindex(*nav)
    for k, idx in enumerate(it):
        data[idx] = disk * (k + 1)
    return data


@pytest.fixture
def window(captured_messages):
    """Empty session — no data loaded."""
    session = _make_session()
    yield {"window": session, "signal_trees": session.signal_trees,
           "plots": session._plots, "messages": captured_messages}
    _teardown(session)


@pytest.fixture
def tem_2d_dataset(captured_messages):
    """2-D image (no navigation) → one signal window."""
    session = _make_session()
    s = hs.signals.Signal2D(np.random.RandomState(0).rand(32, 32).astype(np.float32))
    _load(session, s)
    yield {"window": session, "signal_trees": session.signal_trees,
           "plots": session._plots, "messages": captured_messages}
    _teardown(session)


@pytest.fixture
def stem_4d_dataset(captured_messages):
    """4-D STEM (2-D nav, 2-D signal) → navigator + signal windows."""
    session = _make_session()
    s = hs.signals.Signal2D(_bright_disk_4d((4, 5)))
    s.set_signal_type("electron_diffraction")
    _load(session, s)
    yield {"window": session, "signal_trees": session.signal_trees,
           "plots": session._plots, "messages": captured_messages}
    _teardown(session)


def _movie_stack(n_frames=8, frame=(32, 32)):
    """A lazy in-situ movie: nav-dim 1 (time) stack of 2-D image frames.
    Each frame is a moving bright blob so successive frames differ."""
    import dask.array as da
    data = np.zeros((n_frames,) + frame, dtype=np.float32)
    yy, xx = np.mgrid[0:frame[0], 0:frame[1]]
    for t in range(n_frames):
        cy = int((t / max(1, n_frames - 1)) * (frame[0] - 1))
        data[t] = np.exp(-((yy - cy) ** 2 + (xx - frame[1] // 2) ** 2) / 8.0)
    # Chunk one frame per block (mimics a large-frame movie's storage layout).
    return da.from_array(data, chunks=(1,) + frame)


@pytest.fixture
def movie_dataset(captured_messages):
    """In-situ movie: nav-dim 1 (time), 2-D image signal → 1-D time navigator."""
    session = _make_session()
    s = hs.signals.Signal2D(_movie_stack()).as_lazy()
    # A calibrated time axis (what the DE-MRC reader gives an in-situ movie).
    tax = s.axes_manager.navigation_axes[0]
    tax.name, tax.units, tax.scale = "time", "sec", 0.1
    s.set_signal_type("insitu")   # gates the Play/Fast Forward toolbar buttons
    _load(session, s)
    yield {"window": session, "signal_trees": session.signal_trees,
           "plots": session._plots, "messages": captured_messages}
    _teardown(session)
