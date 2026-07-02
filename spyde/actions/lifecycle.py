"""
lifecycle.py — the shared basis set for interactive actions.

Every heavy action repeats the same lifecycle wiring: run the compute on a
daemon thread and marshal the UI apply back to the asyncio main thread, guard
against superseded runs (React StrictMode double-mount, rapid re-tune), wait
out the find-vectors attach gap, swap an overlay for a newer one, paint a
result onto a tree's signal plots, narrate progress, and poll a progressive
shared-memory fill. These helpers are the single implementation; actions must
use them instead of re-rolling the idioms (see ``spyde/actions/README.md``).

THREADING CONTRACT (CLAUDE.md): UI/figure updates happen on the asyncio main
thread only. Workers marshal via ``session._dispatch_to_main``; ``ipc.emit*``
is safe from any thread.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np

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


# ── generation guard (latest-wins / StrictMode double-mount) ──────────────────

def bump_generation(owner, key: str) -> int:
    """Bump and return ``owner.<key>`` (an int generation counter).

    The run/stop generation contract: a wizard's *open* handler bumps its
    tree's ``_<key>_run_gen`` synchronously BEFORE spawning any worker, and
    every deferred build checks ``is_current`` on arrival; the *close* handler
    bumps unconditionally FIRST, cancelling any in-flight open. This closes the
    React StrictMode mount→cleanup→remount race (open, close, open fired
    synchronously before any worker lands) that otherwise builds two live
    controllers. Also used per-controller for latest-wins recomputes.
    """
    gen = int(getattr(owner, key, 0)) + 1
    setattr(owner, key, gen)
    return gen


def is_current(owner, key: str, gen: int) -> bool:
    """True if *gen* is still ``owner.<key>``'s current generation."""
    return getattr(owner, key, None) == gen


# ── the find-vectors attach gap ───────────────────────────────────────────────

def resolve_vectors(session, plot):
    """Resolve ``(tree, diffraction_vectors)`` for an action.

    Prefers the plot's own tree; falls back to ANY tree carrying vectors (the
    caret's window may resolve to a sibling plot of the vectors tree, e.g. the
    count-map navigator)."""
    tree = getattr(plot, "signal_tree", None) if plot is not None else None
    vecs = getattr(tree, "diffraction_vectors", None) if tree is not None else None
    if vecs is None:
        for cand in getattr(session, "signal_trees", []) or []:
            if getattr(cand, "diffraction_vectors", None) is not None:
                return cand, cand.diffraction_vectors
    return tree, vecs


def fv_batch_running(session) -> bool:
    """True while a Find-Vectors batch is in flight on any tree (the
    ``_fv_batch_running`` flag set by ``find_vectors_action``)."""
    for cand in getattr(session, "signal_trees", []) or []:
        if getattr(cand, "_fv_batch_running", False):
            return True
    return False


def wait_for_vectors(session, plot, then: Callable[[], None], *, what: str,
                     grace: float = 6.0, timeout: float = 300.0,
                     status_every: float = 5.0) -> bool:
    """Wait out the find-vectors attach gap, then re-dispatch.

    Find Vectors attaches ``tree.diffraction_vectors`` only when its batch
    finalizes (on a worker thread); a vector-dependent action can fire in the
    gap. This polls on a worker thread and re-dispatches ``then`` via
    ``_dispatch_to_main`` when the vectors land. While a batch is running it
    waits up to *timeout* (with a periodic status ping); with nothing running
    it gives the brief post-attach window *grace* seconds, then errors.

    Returns True if a wait was started (the caller must return immediately);
    False when there is no event loop to wait on (bare test stubs) — the
    caller should emit its own error then.
    """
    from spyde.backend.ipc import emit_error, emit_status
    if getattr(session, "_dispatch_to_main", None) is None:
        return False

    def _wait():
        waited = 0.0
        status_at = 0.0
        while True:
            _, v = resolve_vectors(session, plot)
            if v is not None:
                session._dispatch_to_main(then)
                return
            running = fv_batch_running(session)
            if not running and waited >= grace:
                emit_error(f"{what} needs a Find Vectors result (no diffraction vectors).")
                return
            if running and waited - status_at >= status_every:
                emit_status("Waiting for diffraction vectors to finish computing…")
                status_at = waited
            if waited >= timeout:
                emit_error(f"{what} timed out waiting for diffraction vectors.")
                return
            time.sleep(0.1)
            waited += 0.1

    threading.Thread(target=_wait, daemon=True, name="wait-vectors").start()
    return True


# ── overlays / painting ───────────────────────────────────────────────────────

def replace_tree_attr(tree, attr: str, factory: Callable[[], Any] | None):
    """Replace ``tree.<attr>`` (an overlay/controller) with ``factory()``,
    removing the prior one first so re-running an action never stacks markers.
    Pass ``factory=None`` to just remove. Returns the new value (None on a
    failed attach — logged, not raised)."""
    old = getattr(tree, attr, None)
    if old is not None and hasattr(old, "remove"):
        try:
            old.remove()
        except Exception as e:
            log.debug("removing prior %s failed: %s", attr, e)
    setattr(tree, attr, None)
    if factory is None:
        return None
    try:
        new = factory()
    except Exception as e:
        log.debug("attaching %s failed: %s", attr, e)
        new = None
    setattr(tree, attr, new)
    return new


def paint_signal_plots(tree, data, *, levels: tuple[float, float] | None = None) -> None:
    """Paint *data* onto every signal plot of *tree*. With *levels* the plot's
    contrast is locked to that range; otherwise it re-auto-levels."""
    for sp in list(getattr(tree, "signal_plots", []) or []):
        try:
            if levels is not None:
                sp.needs_auto_level = False
                sp.set_clim(float(levels[0]), float(levels[1]))
            else:
                sp.needs_auto_level = True
            sp.set_data(data)
        except Exception as e:
            log.debug("painting signal plot failed: %s", e)


def progress_emitter(prefix: str, *, min_interval: float = 0.5) -> Callable[[int, int], None]:
    """A throttled ``progress(done, total)`` callback that emits
    ``"{prefix} {pct}%"`` status lines (always emits the 100% line)."""
    from spyde.backend.ipc import emit_status
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


# ── progressive shared-memory fill ────────────────────────────────────────────

def live_fill_poller(shape: Sequence[int], shm_name: str | None,
                     paint: Callable[[np.ndarray], None], *,
                     interval: float = 0.35, name: str = "live-fill") -> Callable[[], None]:
    """Poll a progressive shared-memory buffer and hand each snapshot to
    ``paint(arr)`` until stopped. The buffer is NaN where unfilled; *paint*
    owns the slicing/display (and any ``isfinite`` gating). Returns a
    ``stop()`` callable — call it when the compute finishes so the final paint
    owns the plot. A ``None`` *shm_name* (buffer allocation failed) is a no-op.
    """
    stop_flag = [False]

    def stop():
        stop_flag[0] = True

    if shm_name is None:
        return stop

    from spyde.drawing.update_functions import read_live_buffer

    def _poller():
        while not stop_flag[0]:
            try:
                paint(read_live_buffer(tuple(shape), shm_name))
            except Exception as e:
                log.debug("live-fill poll paint failed: %s", e)
            time.sleep(interval)

    threading.Thread(target=_poller, daemon=True, name=name).start()
    return stop
