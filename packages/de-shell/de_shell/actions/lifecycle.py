"""
lifecycle.py — the shared basis set for interactive actions.

Every heavy action in every shell app repeats the same wiring: run the compute
on a daemon thread and marshal the UI apply back to the asyncio main thread,
guard against superseded runs (React StrictMode double-mount, rapid re-tune),
swap a controller/overlay for a newer one, and narrate progress. These helpers
are the single implementation of those idioms.

What is here is only the part that knows nothing about the data. SpyDE's
`spyde/actions/lifecycle.py` re-exports all of it alongside its own
domain lifecycle (the find-vectors attach gap, painting a tree's signal plots,
the progressive shared-memory fill), so an action keeps importing one module and
does not have to know which half a given helper came from.

THREADING CONTRACT: UI/figure updates happen on the asyncio main thread only.
Workers marshal via ``session._dispatch_to_main``; ``de_shell.ipc.emit*`` is safe
from any thread.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

log = logging.getLogger(__name__)


# ── worker-thread marshal ─────────────────────────────────────────────────────

def run_on_worker(session, work: Callable[[], Any], *, name: str,
                  on_done: Callable[[Any], None] | None = None,
                  on_error: Callable[[Exception], None] | None = None) -> None:
    """Run ``work()`` on a daemon thread and marshal ``on_done(result)`` back
    onto the asyncio main thread via ``session._dispatch_to_main``.

    ``on_error(exc)`` runs on the worker thread (it typically just
    ``emit_error``\\s, which is thread-safe). When *session* can't marshal
    (``None`` or a bare test stub without ``_dispatch_to_main``) everything
    runs inline synchronously, so handler tests see the result immediately.
    """
    dispatch = getattr(session, "_dispatch_to_main", None)
    if dispatch is None:
        try:
            result = work()
        except Exception as e:
            log.exception("%s failed", name)
            if on_error is not None:
                on_error(e)
            return
        if on_done is not None:
            on_done(result)
        return

    def _worker():
        try:
            result = work()
        except Exception as e:
            log.exception("%s failed", name)
            if on_error is not None:
                on_error(e)
            return
        if on_done is not None:
            dispatch(lambda: on_done(result))

    threading.Thread(target=_worker, daemon=True, name=name).start()


# ── cancellation (a superseded compute is stopped, not ignored) ───────────────

class ComputeHandle:
    """The cancellation handle for one dispatched compute.

    ``flag`` is the ``[False]`` stop token the work polls; ``future`` is the
    future it runs as, when there is one. Constructing a handle registers both
    on the signal tree, so closing the tree stops the compute.

    A superseded or abandoned compute must be CANCELLED, not left running so its
    result can be discarded. The generation guard is not a substitute: it drops
    the result on arrival while the pass keeps reading the dataset, and a pass
    over the dataset is the most expensive thing the app does.

    A compute with no interruption point — one library call over an array
    already in memory — can still take a handle. The flag then stops it before
    it starts and drops a late result, which is all that is available.
    """

    __slots__ = ("flag", "future", "_tree")

    def __init__(self, tree, future=None):
        self.flag: list = [False]
        self.future = future
        self._tree = tree
        register = getattr(tree, "register_cancel", None)
        if register is not None:
            register(flag=self.flag, future=future)

    @property
    def stopped(self) -> bool:
        return bool(self.flag[0])

    def attach(self, future) -> None:
        """Adopt a future created after the handle, registering it too."""
        self.future = future
        register = getattr(self._tree, "register_cancel", None)
        if register is not None and future is not None:
            register(future=future)

    def cancel(self) -> None:
        """Stop the compute and drop it from the tree's registry."""
        self.flag[0] = True
        if self.future is not None:
            try:
                if not self.future.done():
                    self.future.cancel()
            except Exception as e:
                log.debug("cancelling a superseded compute failed: %s", e)
        self._unregister()

    def retire(self) -> None:
        """Drop a finished compute's registration, without marking it stopped.

        Required, or the registry gains an entry per run and a long-lived tree
        accumulates one for every interaction.
        """
        self._unregister()

    def _unregister(self) -> None:
        unregister = getattr(self._tree, "unregister_cancel", None)
        if unregister is None:
            return
        try:
            unregister(flag=self.flag, future=self.future)
        except Exception as e:                               # pragma: no cover
            log.debug("unregistering a compute failed: %s", e)


def supersede(prior: "ComputeHandle | None", tree, future=None) -> ComputeHandle:
    """Cancel *prior* and return the handle for the compute replacing it."""
    if prior is not None:
        prior.cancel()
    return ComputeHandle(tree, future)


# ── generation guard (latest-wins / StrictMode double-mount) ──────────────────

def bump_generation(owner, key: str) -> int:
    """Bump and return ``owner.<key>`` (an int generation counter).

    The run/stop generation contract: a wizard's *open* handler bumps its
    ``_<key>_run_gen`` synchronously BEFORE spawning any worker, and every
    deferred build checks ``is_current`` on arrival; the *close* handler bumps
    unconditionally FIRST, cancelling any in-flight open. This closes the React
    StrictMode mount→cleanup→remount race (open, close, open fired synchronously
    before any worker lands) that otherwise builds two live controllers. Also
    used per-controller for latest-wins recomputes.
    """
    gen = int(getattr(owner, key, 0)) + 1
    setattr(owner, key, gen)
    return gen


def is_current(owner, key: str, gen: int) -> bool:
    """True if *gen* is still ``owner.<key>``'s current generation."""
    return getattr(owner, key, None) == gen


# ── controller / overlay replacement ──────────────────────────────────────────

def replace_tree_attr(owner, attr: str, factory: Callable[[], Any] | None):
    """Replace ``owner.<attr>`` (an overlay/controller) with ``factory()``,
    removing the prior one first so re-running an action never stacks markers.
    Pass ``factory=None`` to just remove. Returns the new value (None on a
    failed attach — logged, not raised)."""
    old = getattr(owner, attr, None)
    if old is not None and hasattr(old, "remove"):
        try:
            old.remove()
        except Exception as e:
            log.debug("removing prior %s failed: %s", attr, e)
    setattr(owner, attr, None)
    if factory is None:
        return None
    try:
        new = factory()
    except Exception as e:
        log.debug("attaching %s failed: %s", attr, e)
        new = None
    setattr(owner, attr, new)
    return new


# ── progress narration ────────────────────────────────────────────────────────

def progress_emitter(prefix: str, *, min_interval: float = 0.5) -> Callable[[int, int], None]:
    """A throttled ``progress(done, total)`` callback that emits
    ``"{prefix} {pct}%"`` status lines (always emits the 100% line)."""
    from de_shell.ipc import emit_status
    last = [0.0]

    def progress(done, total):
        if not total:
            return
        now = time.monotonic()
        if done < total and now - last[0] < min_interval:
            return
        last[0] = now
        emit_status(f"{prefix} {int(100 * done / total)}%")

    return progress


# ── per-window "Calculating…" overlay ─────────────────────────────────────────

class window_computing:
    """Context manager: emit ``window_computing`` start/stop around a long
    compute that paints into ``window_id`` — drives the renderer's floating
    translucent "Calculating…" chip centered on that plot window.

    ALWAYS emits the matching stop, even on exception — the ``__exit__`` runs
    unconditionally, so a cancelled or failed compute cannot leave the overlay
    stuck. ``window_id=None`` is a silent no-op both ways (mirrors
    ``emit_window_computing``'s own guard) so call sites don't need to
    special-case an unattached plot.

    Usage::

        with window_computing(nav_plot.window_id):
            ...long fill...

    or, when the start/stop don't naturally bracket a single call (e.g. a
    background thread that outlives this function), call ``.start()`` /
    ``.stop()`` directly and put ``.stop()`` in the thread's own
    ``try/finally``.
    """

    def __init__(self, window_id: int | None):
        self.window_id = window_id

    def start(self) -> None:
        from de_shell.ipc import emit_window_computing
        emit_window_computing(self.window_id, True)

    def stop(self) -> None:
        from de_shell.ipc import emit_window_computing
        emit_window_computing(self.window_id, False)

    def __enter__(self) -> "window_computing":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False
