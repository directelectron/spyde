"""
Qt-free pytest fixtures for the Electron/anyplotlib backend.

Replaces the old pytest-qt conftest. Each fixture builds a real Session (with
Dask skipped via SPYDE_NO_DASK) and synthetic data, captures every PLOTAPP
message both spyde and anyplotlib emit, and yields a dict mirroring the old
fixtures' shape: ``window`` (the Session), ``signal_trees``, ``plots``, and
``messages`` (the captured emit list).
"""
from __future__ import annotations

import os
import time
from typing import Iterator

import numpy as np
import hyperspy.api as hs
import pytest

os.environ.setdefault("SPYDE_NO_DASK", "1")


@pytest.fixture
def captured_messages(monkeypatch):
    """Capture both PLOTAPP channels (spyde.ipc and anyplotlib._electron)."""
    import spyde.backend.ipc as ipc
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
    return Session(n_workers=1, threads_per_worker=1)


def _load(session, signal):
    session._add_signal(signal, source_path=None)
    time.sleep(0.8)  # let selector debounce timers fire


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
    session.shutdown()


@pytest.fixture
def tem_2d_dataset(captured_messages):
    """2-D image (no navigation) → one signal window."""
    session = _make_session()
    s = hs.signals.Signal2D(np.random.RandomState(0).rand(32, 32).astype(np.float32))
    _load(session, s)
    yield {"window": session, "signal_trees": session.signal_trees,
           "plots": session._plots, "messages": captured_messages}
    session.shutdown()


@pytest.fixture
def stem_4d_dataset(captured_messages):
    """4-D STEM (2-D nav, 2-D signal) → navigator + signal windows."""
    session = _make_session()
    s = hs.signals.Signal2D(_bright_disk_4d((4, 5)))
    s.set_signal_type("electron_diffraction")
    _load(session, s)
    yield {"window": session, "signal_trees": session.signal_trees,
           "plots": session._plots, "messages": captured_messages}
    session.shutdown()


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
    session.shutdown()
