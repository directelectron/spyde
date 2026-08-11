"""example_download.py — IPC progress + cancel for the Examples-menu downloads.

em-database fetches through pooch, and pooch accepts an arbitrary tqdm-like
object as its ``progressbar`` — that one hook gives us both a byte-level
progress stream and a cancellation point:

- :class:`_IpcProgress` implements pooch's progress protocol (assignable
  ``total``, ``update(n)``, ``reset()``, ``close()``) and emits throttled
  ``download_progress`` messages; ``update`` raises :class:`DownloadCancelled`
  when the flag is set, which aborts pooch's stream (pooch downloads to a temp
  file and deletes it on error, so no partial file is left in the cache; the
  exception is not a ``ValueError``/requests error, so pooch's retry loop does
  NOT re-attempt a cancelled download).
- :func:`example_progress` scopes one download and emits ``download_done``.
- ``download_cancel`` (a staged action, wired in ``actions/registry.py``) sets
  the cancel flag for a token — the renderer's toast Cancel button sends it.

Messages (renderer: ``DownloadToasts.tsx``):
    download_progress {token, label, done, total}     # bytes; total 0 = unknown
    download_done     {token, ok, cancelled}
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager

from de_shell.ipc import emit

log = logging.getLogger(__name__)

# token -> cancel flag for in-flight downloads (Examples menu: one at a time in
# practice, but keyed so concurrent loads stay independent).
_CANCELS: dict[str, threading.Event] = {}
_LOCK = threading.Lock()

_EMIT_INTERVAL_S = 0.2          # progress-message throttle


class DownloadCancelled(Exception):
    """Raised inside pooch's download stream when the user hit Cancel."""


def request_cancel(token: str) -> bool:
    """Flag the download for *token* to stop. Returns True if it was in flight."""
    with _LOCK:
        ev = _CANCELS.get(token)
    if ev is None:
        return False
    ev.set()
    return True


def download_cancel(session, plot, payload) -> None:
    """Staged action: the renderer toast's Cancel button."""
    token = str((payload or {}).get("token", ""))
    if request_cancel(token):
        log.info("[download] cancel requested for %s", token)
    else:
        log.debug("[download] cancel for unknown/finished token %s", token)


class _IpcProgress:
    """pooch-compatible progress object → throttled ``download_progress`` emits.

    pooch drives it per FILE: assigns ``total``, calls ``update(chunk)`` per
    chunk, then ``reset()`` + ``update(total)`` + ``close()`` at the end. The
    ``total`` setter starts a fresh file (an example may fetch several)."""

    def __init__(self, token: str, label: str, cancel: threading.Event):
        self.token = token
        self.label = label
        self._cancel = cancel
        self._total = 0
        self._done = 0
        self._last_emit = 0.0
        self._closing = False
        self.started = False        # any bytes actually flowed (cache miss)?

    # pooch assigns `progress.total = content_length` before streaming a file.
    @property
    def total(self):
        return self._total

    @total.setter
    def total(self, value):
        self._total = int(value or 0)
        self._done = 0
        self._closing = False
        self._emit(force=True)

    def update(self, n) -> None:
        if self._cancel.is_set():
            raise DownloadCancelled(self.token)
        self.started = True
        self._done += int(n)
        if self._total:
            self._done = min(self._done, self._total)
        self._emit(force=self._closing)

    def reset(self) -> None:
        # pooch's end-of-file sequence is reset() → update(total) → close();
        # mark it so the final update emits un-throttled (bar reaches 100%).
        self._done = 0
        self._closing = True

    def close(self) -> None:
        self._closing = False

    def _emit(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_emit) < _EMIT_INTERVAL_S:
            return
        self._last_emit = now
        emit({"type": "download_progress", "token": self.token,
              "label": self.label, "done": int(self._done),
              "total": int(self._total)})


@contextmanager
def example_progress(token: str, label: str):
    """Scope the progress/cancel monitor around one example download.

    Yields the monitor to hand straight to
    ``em_database.DownloadableDataset.download(progressbar=…)``. That method
    forwards whatever it is given to ``pooch.HTTPDownloader``, and pooch's
    ``progressbar`` contract is "any tqdm-like object" — so :class:`_IpcProgress`
    plugs in with no interception at all.

    (This used to swap out ``pyxem.data._data``'s view of the ``pooch`` module
    to reach the same hook, because pyxem's loaders construct the downloader
    themselves and take no progress argument. em-database exposing the
    parameter is what let that monkey-patch go.)

    Progress messages only flow when pooch actually streams — a file already on
    disk never constructs a progress bar — so the toast simply never appears
    for cached data, and ``download_done`` is emitted iff a download started.
    """
    cancel = threading.Event()
    with _LOCK:
        _CANCELS[token] = cancel
    monitor = _IpcProgress(token, label, cancel)
    ok = False
    cancelled = False
    try:
        yield monitor
        ok = True
    except DownloadCancelled:
        cancelled = True
        raise
    finally:
        with _LOCK:
            _CANCELS.pop(token, None)
        if monitor.started:
            emit({"type": "download_done", "token": token,
                  "ok": ok, "cancelled": cancelled})
