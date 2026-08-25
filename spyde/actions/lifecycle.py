"""
lifecycle.py — SpyDE's lifecycle basis set for interactive actions.

Every heavy action repeats the same lifecycle wiring: run the compute on a
daemon thread and marshal the UI apply back to the asyncio main thread, guard
against superseded runs (React StrictMode double-mount, rapid re-tune), wait
out the find-vectors attach gap, swap an overlay for a newer one, paint a
result onto a tree's signal plots, narrate progress, and poll a progressive
shared-memory fill. These helpers are the single implementation; actions must
use them instead of re-rolling the idioms (see ``spyde/actions/README.md``).

**Half of this module now lives in the shell.** The idioms that know nothing
about the data — the worker marshal, the generation guard, attribute
replacement, progress narration, the computing overlay — are
``de_shell.actions.lifecycle``, shared with de-groundcrew and de-autopilot. They
are RE-EXPORTED here, deliberately: this module is SpyDE's lifecycle API, and an
action should import one module rather than having to know which half a given
helper came from. What stays below is the part that is genuinely about
diffraction vectors, signal trees and the progressive fill.

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

from de_shell.actions.lifecycle import (  # noqa: F401  (re-exported API)
    run_on_worker, bump_generation, is_current, replace_tree_attr,
    progress_emitter, window_computing, ComputeHandle, supersede,
)

log = logging.getLogger(__name__)


# ── the find-vectors attach gap ───────────────────────────────────────────────

def attach_container(tree, store, *, name: str):
    """Attach a ragged result container to *tree* under attribute *name* —
    THE seam between a batch compute finalizing and the tree carrying its
    result (``tree.diffraction_vectors`` today; ``tree.particles`` next).

    setattr + provenance stamp: a container carrying no ``provenance`` record
    of its own inherits the tree's commit provenance (the dict
    ``commit._stamp_provenance`` stores as ``tree._commit_provenance``), so a
    saved container self-describes the way a committed tree does. Readers are
    unchanged: :func:`resolve_vectors`, the ``requires_vectors`` toolbar gate
    and the Save hook all read the attribute this helper sets.

    Later-PR design (recorded here, deliberately NOT implemented in this PR):
    toolbar YAML grows a generic ``requires_container: <name>`` key with
    ``requires_vectors`` kept as its alias, and the ``commit.*`` entry points
    (``open_result_tree`` / ``commit_result_tree``) grow a ``container=``
    kwarg routed through this helper — one attach/gate/commit seam for every
    ragged family instead of each action hand-rolling the setattr.

    Returns *store*.
    """
    setattr(tree, name, store)
    try:
        if getattr(store, "provenance", None) is None:
            prov = getattr(tree, "_commit_provenance", None)
            if prov:
                store.provenance = dict(prov)
    except Exception as e:
        log.debug("stamping container provenance failed: %s", e)
    return store


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
                     strict: bool = False, grace: float = 6.0,
                     timeout: float = 300.0, status_every: float = 5.0) -> bool:
    """Wait out the find-vectors attach gap, then re-dispatch.

    Find Vectors attaches ``tree.diffraction_vectors`` only when its batch
    finalizes (on a worker thread); a vector-dependent action can fire in the
    gap. This polls on a worker thread and re-dispatches ``then`` via
    ``_dispatch_to_main`` when the vectors land. While a batch is running it
    waits up to *timeout* (with a periodic status ping); with nothing running
    it gives the brief post-attach window *grace* seconds, then errors.

    ``strict=True`` fires only when the CLICKED plot's own tree gets vectors —
    required when the caller's gate checks that tree specifically (Vector VI,
    Vector OM), otherwise vectors on a *different* tree would fire ``then``
    into the same gate and re-wait forever. The default (any-tree fallback)
    matches handlers that resolve via :func:`resolve_vectors` (Strain).

    Returns True if a wait was started (the caller must return immediately);
    False when there is no event loop to wait on (bare test stubs) — the
    caller should emit its own error then.
    """
    from de_shell.ipc import emit_error, emit_status
    if getattr(session, "_dispatch_to_main", None) is None:
        return False

    def _poll():
        if strict:
            t = getattr(plot, "signal_tree", None) if plot is not None else None
            return getattr(t, "diffraction_vectors", None) if t is not None else None
        return resolve_vectors(session, plot)[1]

    def _wait():
        from de_shell.timing import reliable_sleep
        waited = 0.0
        status_at = 0.0
        while True:
            v = _poll()
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
            reliable_sleep(0.1)
            waited += 0.1

    threading.Thread(target=_wait, daemon=True, name="wait-vectors").start()
    return True


def wait_for_particles(session, plot, then: Callable[[], None], *, what: str,
                       grace: float = 6.0, timeout: float = 600.0,
                       status_every: float = 5.0) -> bool:
    """Wait out the segmentation attach gap, then re-dispatch.

    The particle-tree twin of :func:`wait_for_vectors`, and it exists for the
    same reason: ``seg_run`` opens its result window EARLY with a placeholder
    count trace and attaches ``tree.particles`` only when the batch finalizes on
    a worker thread, so a particle-dependent action (track, export, per-particle
    DP) can fire in the gap and find ``None`` on a tree that gets it seconds
    later. Polls on a worker thread and re-dispatches *then* via
    ``_dispatch_to_main`` once the particles land.

    Unlike the vectors version there is **no ``strict`` switch** — and that is
    deliberate. ``wait_for_vectors`` needs one because vectors attach to the tree
    the user clicked, so an any-tree fallback could satisfy the wait from an
    unrelated tree and re-dispatch forever into a tree-specific gate. Particles
    live on their OWN tree (plan §0.6: segmentation spawns a new tree rather than
    decorating the source), so the only sensible question is whether *this* tree
    has them. There is no ambiguity to resolve, so there is no knob.

    The default *timeout* is longer than the vectors one (600 s vs 300 s):
    segmenting thousands of frames is the plan's stated target scale, and a run
    that legitimately takes eight minutes must not be abandoned at five.

    Returns True if a wait was started (the caller must return immediately);
    False when there is no event loop to wait on (bare test stubs) — the caller
    should emit its own error then.
    """
    from de_shell.ipc import emit_error, emit_status
    if getattr(session, "_dispatch_to_main", None) is None:
        return False

    def _tree():
        return getattr(plot, "signal_tree", None) if plot is not None else None

    def _wait():
        from de_shell.timing import reliable_sleep
        waited = 0.0
        status_at = 0.0
        while True:
            tree = _tree()
            if tree is not None and getattr(tree, "particles", None) is not None:
                session._dispatch_to_main(then)
                return
            running = seg_batch_running(session)
            if not running and waited >= grace:
                emit_error(f"{what} needs a segmentation result (no particles).")
                return
            if running and waited - status_at >= status_every:
                emit_status("Waiting for particle segmentation to finish…")
                status_at = waited
            if waited >= timeout:
                emit_error(f"{what} timed out waiting for particles.")
                return
            reliable_sleep(0.1)
            waited += 0.1

    threading.Thread(target=_wait, daemon=True, name="wait-particles").start()
    return True


def seg_batch_running(session) -> bool:
    """True while any tree has a segmentation batch in flight.

    Mirrors :func:`fv_batch_running`. The flag lives on the tree
    (``_seg_batch_running``) so ``BaseSignalTree.close()`` clears it along with
    everything else, per the ownership map in ``actions/README.md`` §3.
    """
    for tree in getattr(session, "signal_trees", []) or []:
        if getattr(tree, "_seg_batch_running", False):
            return True
    return False


# ── overlays / painting ───────────────────────────────────────────────────────

def show_tree_node(plot, tree, new_signal) -> None:
    """Switch *plot* to display *new_signal* and re-slice from the navigator so
    the new frame shows immediately — ``add_transformation`` only REGISTERS the
    new PlotState; nothing else repaints until a selector event. Then re-emit
    the signal tree so the Workflow panel reflects the change.

    This is what makes a TransformAction (Rebin, Center Zero Beam, …) visibly
    take effect without the user having to nudge the crosshair."""
    # Interactive per-node widgets (the Crop box; the CZB search box /
    # crosshair / found markers) describe the node being LEFT — a node switch
    # keeps their caret mounted (no <key>_close fires), so hide them here
    # rather than leave them painted over the new node's data. Handlers that
    # add markers AFTER switching (czb_run / czb_pick call _display first,
    # then _czb_show_found) are unaffected; imports are lazy so this stays
    # cycle-free.
    try:
        from spyde.actions.base import _crop_remove_widget
        from spyde.actions.center_zero_beam import _czb_clear_widgets
        _crop_remove_widget(tree)
        _czb_clear_widgets(tree)
    except Exception as e:
        log.debug("node-switch widget teardown failed: %s", e)
    try:
        plot.set_plot_state(new_signal)
    except Exception as e:
        log.debug("switching plot to new node failed: %s", e)
    npm = getattr(tree, "navigator_plot_manager", None)
    if npm is not None:
        for sels in getattr(npm, "navigation_selectors", {}).values():
            for sel in sels:
                try:
                    sel.delayed_update_data(force=True)
                except Exception as e:
                    log.debug("re-slicing navigator after transform failed: %s", e)
    session = getattr(plot, "session", None)
    if session is not None and hasattr(session, "_reemit_signal_tree"):
        try:
            session._reemit_signal_tree(plot)
        except Exception as e:
            log.debug("re-emitting signal tree after transform failed: %s", e)


def paint_signal_plots(tree, data, *, levels: tuple[float, float] | None = None) -> int:
    """Paint *data* onto every signal plot of *tree*. With *levels* the plot's
    contrast is locked to that range; otherwise it re-auto-levels.

    Returns the number of plots whose ``set_data`` SUCCEEDED.  A failed paint
    is still swallowed (a progressive compute's per-chunk callback must never
    fail the compute) and logged at DEBUG, but the count makes the swallow
    observable — callers that ignore the return are unaffected, and a caller
    that must know whether pixels actually landed (the live signal preview,
    and the tests that used to infer it from counters that incremented even
    when the paint raised) can assert on it."""
    painted = 0
    for sp in list(getattr(tree, "signal_plots", []) or []):
        try:
            if levels is not None:
                sp.needs_auto_level = False
                sp.set_clim(float(levels[0]), float(levels[1]))
            else:
                sp.needs_auto_level = True
            sp.set_data(data)
            painted += 1
        except Exception as e:
            log.debug("painting signal plot failed: %s", e)
    return painted


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

    from de_shell.timing import reliable_sleep
    from spyde.drawing.update_functions import read_live_buffer

    def _poller():
        # Skip a paint when the buffer has not CHANGED since the last one.
        #
        # Measured on a real 256x256-nav fill: ~195 paints for ~26 distinct
        # buffer states — 7x redundant. Two painters run concurrently and
        # neither knows about the other (this poller on its interval, plus the
        # per-chunk relay), and during the long "0%" stretch before the first
        # chunk lands they were repainting an ALL-NaN array several times a
        # second. Every one of those pays `_set_array` + levels + histogram +
        # a binary push, on the same main thread that has to submit and collect
        # the compute they are waiting for.
        #
        # The digest is over the raw bytes, so NaN-vs-NaN compares equal (unlike
        # `==`) and an unfilled buffer reads as "unchanged" rather than as a
        # change every tick. ~50 us for a 256 KB nav — three orders below the
        # paint it avoids.
        import hashlib
        last = [None]
        while not stop_flag[0]:
            try:
                arr = read_live_buffer(tuple(shape), shm_name)
                digest = hashlib.blake2b(
                    np.ascontiguousarray(arr).view(np.uint8),
                    digest_size=16).digest()
                if digest != last[0]:
                    last[0] = digest
                    paint(arr)
            except Exception as e:
                log.debug("live-fill poll paint failed: %s", e)
            reliable_sleep(interval)

    threading.Thread(target=_poller, daemon=True, name=name).start()
    return stop
