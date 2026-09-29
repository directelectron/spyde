"""
live_signal.py — the SIGNAL half of a progressively-filled result window.

A progressive action opens its result window EARLY (``commit.open_result_tree``)
and fills the NAVIGATOR block by block while the batch runs. Where that window
is a navigator **and** a signal plot (Find Vectors: a vector count-map navigator
plus a rendered-disks signal plot) only half of it used to come alive: the count
map filled in, while the signal plot sat on its placeholder zeros — black for the
whole run — and dragging the crosshair over a region whose vectors were already
sitting on the client showed nothing.

:class:`ProgressiveSignalPreview` drives that signal plot from the SAME per-block
results the navigator fill uses:

  (a) **live during the fill** — each completed block paints ONE sample position's
      frame, so the signal plot visibly updates alongside the navigator;
  (b) **computed regions are readable immediately**: a reader pinned on the
      result tree renders any ALREADY-computed position on demand, so dragging
      over a filled region shows that position's real frame without waiting for
      the batch. An un-computed position returns ``None``, so the last good
      frame stays up (no flash) exactly like the expensive-tier nav read
      (CLAUDE.md Live-Display §3).

**(b) BEATS (a), permanently.** The first time the user TOUCHES this window's
navigator, ``_user_owns`` latches and the auto-sample flash stops for the rest of
the run — the signal panel belongs to them from then on. It is not a timed hold:
a hold resumes the flash the moment the user pauses to look at what they
navigated to, which is exactly when it must not. The navigator count map keeps
filling visibly either way; only the sample paint stops.

**The latch reads the POINTER EVENT, not the read.** It rides
``BaseSelector.interaction_hooks``, fired by the widget handlers before anything
is queued, so it is set the instant the crosshair is grabbed — whatever the read
then does. Inferring it from the reads instead ("a read arrived at a new index,
so the user must have dragged") has holes that all surface as the same symptom:
a landing block repainting the panel just as the user lets go. ``_run_update``
short-circuits a repeat position, so a grab that ends where it began produces no
changed index at all; a drag entirely over not-yet-computed data produces only
declines; and the dispatcher coalesces, so intermediate positions never reach the
preview. The changed-index check is KEPT below as a backstop for a crosshair
moved without a pointer event (a programmatic set, a linked selector) — it can
only latch later than the hook, never instead of it.

**The result tree is LOCKED while it fills** (``lifecycle.lock_tree``): no
actions and no new nodes until the result attaches, because the root is a
placeholder the batch is about to replace and a node hung off it would describe
data that does not exist yet.

The action supplies only ``render(index) -> ndarray | None``; readiness tracking,
sampling, throttling, the thread marshal and the handover to the final display
live here so every progressive action gets identical behaviour.

**"Random" is deterministic per block.** The sample position is drawn from a
generator seeded by the block's own nav origin/extent, so it is pseudo-random
across the grid (you get a different position from each block, not always the
corner) but reproducible regardless of the order blocks land in — which is what
makes it testable.

THREADING CONTRACT (CLAUDE.md): the feeds (:meth:`note_block`,
:meth:`note_ready_mask`) run on whatever thread the compute's per-chunk callback
uses — a Dask done-callback thread or a poller. ``render`` runs there too, so it
must be cheap and must never touch a ``Plot``; the paint is marshalled onto the
asyncio main thread via ``session._dispatch_to_main``. The reader runs on the
``_NavDispatcher`` thread like every other navigator read.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np

log = logging.getLogger(__name__)

#: Minimum seconds between two auto-sample paints (a fast cluster lands many
#: blocks per second; painting each one is pure transport churn).
SAMPLE_MIN_INTERVAL = 0.45

#: Minimum seconds between two INFO narration lines (the paints are faster).
LOG_MIN_INTERVAL = 2.0


def _block_sample_index(nav_slices: Sequence[slice]) -> tuple[int, ...]:
    """A deterministic pseudo-random nav index inside the block *nav_slices*.

    Seeded from the block's own bounds, so every block yields a different
    position but the SAME block always yields the same one — independent of the
    order blocks complete in (which on a cluster is arbitrary). That is what
    makes the "random position from each block" contract testable.
    """
    bounds = []
    for sl in nav_slices:
        start = int(sl.start or 0)
        stop = sl.stop
        bounds.append((start, start + 1 if stop is None else int(stop)))
    seed = 0
    for lo, hi in bounds:
        seed = (seed * 1000003 + lo * 31 + hi) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    return tuple(int(rng.integers(lo, max(lo + 1, hi))) for lo, hi in bounds)


def _hook_targets(selector) -> list:
    """Every object whose pointer events count as "the user touched *selector*".

    Normally just the selector. A composite (the 2-D crosshair + rectangle
    pair, or the 1-D line + span pair) is the exception: it owns no widget of its own
    and delegates unknown attributes to whichever sub-selector is ACTIVE, so a
    hook registered through it would reach exactly one of the two — and would
    silently stop being reachable the moment the user toggled Integrate. Each
    sub-selector has its own widget and its own hook list, so register on both.
    """
    subs = [getattr(selector, name, None)
            for name in ("_crosshair_selector", "_rect_selector",
                         "_inf_line_selector", "_linear_region_selector")]
    subs = [s for s in subs if s is not None]
    return subs or [selector]


class ProgressiveSignalPreview:
    """Live signal-plot preview for one progressively-filled result tree.

    It is the reader the result tree's signal plots read through while the
    batch runs: ``read_frame`` answers a computed position and returns None for
    one the batch has not reached.

    Build it with :func:`attach_signal_preview` (which no-ops on a window that
    has no navigator, e.g. the IPF-only Orientation result). Feed it blocks as
    the compute produces them and :meth:`close` it when the batch finalizes —
    ``close`` never clobbers a final display the action pinned in the meantime
    (it unpins only while the pinned reader is still this one).
    """

    def __init__(self, session, tree, *, render: Callable[[tuple], Any],
                 nav_shape: Sequence[int],
                 sample_interval: float = SAMPLE_MIN_INTERVAL,
                 name: str = "live-signal"):
        self.session = session
        self.tree = tree
        self.render = render
        self.nav_shape = tuple(int(s) for s in nav_shape)
        self.sample_interval = float(sample_interval)
        self.name = name

        self.n_positions = int(np.prod(self.nav_shape))
        self._ready = np.zeros(self.nav_shape, dtype=bool)
        self._lock = threading.Lock()
        self._closed = False
        self._last_paint = 0.0
        #: LATCH: set the first time the user TOUCHES this window's navigator,
        #: and never cleared. From then on the signal panel is theirs — a
        #: landing block may no longer paint a sample frame over what they are
        #: looking at, and letting go over a position the batch has not reached
        #: leaves their last good frame up rather than a sample of somewhere
        #: else. The navigator COUNT MAP keeps filling regardless; only the
        #: auto-sample "flash" stops. (This was a 2-second hold, which meant the
        #: flash resumed and stole the panel back the moment the user paused.)
        self._user_owns = False
        #: The last nav index a navigator read asked for, so a genuine MOVE can
        #: be told from a re-fire at the SAME position. Our own re-fires (the
        #: parked-position refresh, the selector's settle timer) are forced
        #: updates at an unchanged index and must NOT latch — otherwise the
        #: first block landing under a resting crosshair would end auto-sampling
        #: without the user having touched anything. BACKSTOP only: the pointer
        #: hook (`_on_user_interaction`) is the primary latch.
        self._last_index: tuple[int, ...] | None = None
        # ONE bound method per preview: `self._on_user_interaction` builds a
        # fresh bound object on every attribute access, and close() must be able
        # to remove the exact object watch_navigators() appended.
        self.interaction_hook = self._on_user_interaction
        #: the selectors carrying that hook, so close() can take it off again
        self._hooked: list = []
        self._last_log = 0.0
        self._last_serve_log = 0.0
        self._last_decline_log = 0.0
        #: counters the tests (and the log lines) assert on
        self.blocks_seen = 0
        #: (a) auto-sample paints driven by a landing block
        self.frames_painted = 0
        #: of those, paints whose set_data actually SUCCEEDED on >=1 signal
        #: plot (paint_signal_plots' return) — frames_painted counts attempts,
        #: so a swallowed set_data failure is visible as landed < painted.
        self.frames_landed = 0
        #: (b) navigator-driven reads answered from the already-computed region
        self.frames_served = 0
        #: navigator-driven reads that landed on a position the batch has not
        #: reached yet (returned None → the last good frame stays up)
        self.reads_declined = 0

    # ── the readiness feed ───────────────────────────────────────────────────

    def note_block(self, nav_slices: Sequence[slice]) -> None:
        """Mark the block *nav_slices* computed and show something from it.

        If the navigator is PARKED inside this block the user's own position
        wins — the selector is re-fired so the position they are already
        pointing at fills in the moment its data lands (part of "dragging over a
        computed region shows its value": you can also arrive first and wait).
        Otherwise a deterministic sample position from the block is painted.

        Safe to call from any thread and never raises — a progressive compute's
        per-chunk callback must not be able to fail the compute.
        """
        if self._closed:
            return
        try:
            sl = tuple(nav_slices)
            with self._lock:
                self._ready[sl] = True
                self.blocks_seen += 1
            if self._refresh_parked_position(sl):
                return
            self._maybe_paint(_block_sample_index(sl))
        except Exception as e:
            log.debug("[%s] note_block(%r) failed: %s", self.name, nav_slices, e)

    def is_ready(self, index: Sequence[int]) -> bool:
        """Has the position *index* (a full nav index tuple) been computed?"""
        try:
            idx = tuple(int(v) for v in index)
            if len(idx) != len(self.nav_shape):
                return False
            if any(not (0 <= v < n) for v, n in zip(idx, self.nav_shape)):
                return False
            with self._lock:
                return bool(self._ready[idx])
        except Exception:
            return False

    @property
    def ready_count(self) -> int:
        with self._lock:
            return int(self._ready.sum())

    # ── (a) the auto-sample paint ────────────────────────────────────────────

    def _maybe_paint(self, index: tuple[int, ...]) -> None:
        now = time.monotonic()
        if now - self._last_paint < self.sample_interval:
            return
        if self._user_owns:
            return          # the user drove the navigator — the panel is theirs
        frame = None
        try:
            frame = self.render(index)
        except Exception as e:
            log.debug("[%s] render%s failed: %s", self.name, index, e)
        if frame is None:
            return
        self._last_paint = now
        self.frames_painted += 1
        # Narrate at INFO but throttled well below the paint rate — a long batch
        # paints a couple of frames a second and this line goes to the user's Log
        # panel. (It is also how the e2e spec proves the fill was progressive:
        # a line whose `ready` is short of `total` can only have been painted
        # mid-run.)
        if self.frames_painted == 1 or now - self._last_log >= LOG_MIN_INTERVAL:
            self._last_log = now
            log.info("[live-signal] %s: preview frame at %s (%d/%d positions ready)",
                     self.name, index, self.ready_count, self.n_positions)
        self._paint(frame)

    def _paint(self, frame) -> None:
        """Marshal the paint onto the asyncio main thread (CLAUDE.md: plots are
        touched there only). Without a loop to marshal to (handler tests, a bare
        stub session) paint inline so tests see the frame immediately."""
        from spyde.actions.lifecycle import paint_signal_plots
        tree = self.tree

        def _apply():
            # Re-checked here: the grab can land between the sampler's check on
            # the compute thread and this paint reaching the main thread.
            if not self._closed and not self._user_owns:
                if paint_signal_plots(tree, frame) > 0:
                    self.frames_landed += 1

        dispatch = getattr(self.session, "_dispatch_to_main", None)
        if dispatch is None:
            _apply()
            return
        try:
            dispatch(_apply)
        except Exception as e:
            log.debug("[%s] dispatching preview paint failed: %s", self.name, e)

    # ── (b) the on-demand read ───────────────────────────────────────────────

    def _navigation_index(self, indices):
        """A selector's reported indices as a full navigation index tuple, the
        same preparation the base read makes: the spatial pair in data order,
        a crosshair's point cloud reduced to one point, and a region reduced to
        its centre (the action's own final display owns real region
        integration; the preview shows one position)."""
        from spyde.drawing.update_functions import _prepare_nav_indices

        prepared = _prepare_nav_indices(self.tree.root, indices, integrating=False)
        if prepared is None:
            return None
        return tuple(int(v) for v in np.atleast_1d(np.asarray(prepared)).ravel())

    def _refresh_parked_position(self, nav_slices: Sequence[slice]) -> bool:
        """Re-fire any selector whose current position sits in *nav_slices*.

        Goes through ``delayed_update_data`` — i.e. the ``_NavDispatcher``, the
        one legal way to drive a navigator read (CLAUDE.md Live-Display §2) —
        never a direct paint from this callback thread.
        """
        hit = False
        manager = getattr(self.tree, "navigator_plot_manager", None)
        for selector in getattr(manager, "all_navigation_selectors", None) or ():
            try:
                index = self._navigation_index(selector.current_indices)
            except Exception:
                continue
            if index is None or len(index) != len(nav_slices):
                continue
            inside = all(
                int(sl.start or 0) <= v < (int(sl.stop) if sl.stop is not None
                                           else int(sl.start or 0) + 1)
                for v, sl in zip(index, nav_slices)
            )
            if not inside:
                continue
            hit = True
            try:
                selector.delayed_update_data(force=True)
            except Exception as e:
                log.debug("[%s] re-firing parked selector failed: %s", self.name, e)
        return hit

    def _decline(self, index, now: float):
        """Count + narrate a navigator read the preview could NOT answer.

        Narrated at INFO (throttled like the serve line, first one always) so a
        drag that serves NOTHING is distinguishable in the log from a drag that
        never reached the backend at all — the e2e spec's served-count went 0→0
        once and the log could not say whether the reads were declined (drag
        over uncomputed data) or never ran. The index says WHERE the drag
        actually read; the cumulative counts make the line parseable the same
        way as the serve line."""
        self.reads_declined += 1
        if self.reads_declined == 1 or now - self._last_decline_log >= LOG_MIN_INTERVAL:
            self._last_decline_log = now
            log.info("[live-signal] %s: navigator read declined at %s — position "
                     "not yet computed (%d/%d positions ready, %d served / "
                     "%d declined)", self.name, index, self.ready_count,
                     self.n_positions, self.frames_served, self.reads_declined)
        return None

    def _take_ownership(self, why: str) -> None:
        """Hand the signal panel to the user, permanently. Idempotent."""
        if self._user_owns:
            return
        self._user_owns = True
        log.info("[live-signal] %s: navigator %s — the signal panel is the "
                 "user's for the rest of this run (auto-sampling stops; the "
                 "count map keeps filling)", self.name, why)

    def _on_user_interaction(self, selector) -> None:
        """THE latch — a pointer event on a navigator this preview watches
        on (``BaseSelector.interaction_hooks``).

        Fires on the event thread before the read is even queued, so ownership
        is decided by the TOUCH and not by what the read returns. That is the
        whole point: a drag over data the batch has not reached yet returns
        nothing at all, and if ownership waited for a served frame the sampler
        would still be live when the user let go and the next landing block
        would repaint the panel out from under them.
        """
        self._take_ownership("grabbed")

    def _note_read_position(self, index) -> None:
        """Backstop latch: a navigator read arrived at a NEW position.

        Covers a crosshair moved WITHOUT a pointer event (a programmatic set, a
        linked selector driving this one). Our own re-fires — the parked-position
        refresh, the selector's settle timer — are forced reads at an UNCHANGED
        index and must not latch, or the first block landing under a resting
        crosshair would end auto-sampling with nobody having touched anything.
        """
        if index is None:
            return
        if self._last_index is not None and index != self._last_index:
            self._take_ownership(f"driven to {index}")
        self._last_index = index

    def read_frame(self, indices):
        """Render an already-computed position on demand.

        Runs on the ``_NavDispatcher`` thread. Returns ``None`` for a position
        the batch has not reached yet, which the navigator read treats as
        nothing to paint, so the last good frame stays up.
        """
        now = time.monotonic()
        if self._closed:
            return None
        try:
            index = tuple(int(v) for v in np.atleast_1d(np.asarray(indices)).ravel())
            self._note_read_position(index)
            if not self.is_ready(index):
                return self._decline(index, now)
            frame = self.render(index)
            if frame is None:
                return self._decline(index, now)
            self.frames_served += 1
            # Narrate the READ path separately from the auto-sample paint above:
            # this line is the only direct evidence that dragging the navigator
            # over an ALREADY-COMPUTED region returned that position's real
            # frame (as opposed to the display merely being repainted by a
            # landing block). The e2e spec asserts on it, because a pixel
            # signature alone cannot tell those two causes apart.
            if self.frames_served == 1 or now - self._last_serve_log >= LOG_MIN_INTERVAL:
                self._last_serve_log = now
                log.info("[live-signal] %s: navigator read served at %s from the "
                         "already-computed region (%d/%d positions ready, "
                         "%d served / %d not yet computed)",
                         self.name, index, self.ready_count, self.n_positions,
                         self.frames_served, self.reads_declined)
            return frame
        except Exception as e:
            log.debug("[%s] preview read failed: %s", self.name, e)
            return None

    def region_frame(self, points):
        """A region shows its centre position, or nothing when that position
        has not been computed.

        Integrating a region over a half-finished result would have to wait for
        every position in it; the action's own final display owns real region
        integration, and the preview shows one position."""
        grid = np.asarray(points).reshape(-1, np.shape(points)[-1])
        centre = np.mean(grid, axis=0).astype(int)
        return self.read_frame(tuple(int(v) for v in centre))

    @property
    def frame_bytes(self) -> int:
        """One frame of the result window, for a caller sizing a cache."""
        data = getattr(self.tree.root, "data", None)
        shape = tuple(getattr(data, "shape", ()) or ())
        if len(shape) <= len(self.nav_shape):
            return 0
        return int(np.prod(shape[len(self.nav_shape):])) * data.dtype.itemsize

    def watch_navigators(self) -> None:
        """Latch on the TOUCH of any of this window's navigators, and seed the
        backstop latch's reference position.

        The hook goes on every navigation selector of the tree (both halves of
        a composite, see :func:`_hook_targets`), and :meth:`close` removes it.

        The seed is where the crosshair ALREADY sits, so a programmatic move is
        recognised as a move (with no seed the first read only establishes the
        baseline and the backstop would trail one position behind). It is
        prepared exactly as the read prepares the index it hands
        :meth:`read_frame`, so an unmoved crosshair compares equal."""
        from spyde.drawing.update_functions import (
            _NAVIGATION_AXES, _prepare_nav_indices)

        manager = getattr(self.tree, "navigator_plot_manager", None)
        for selector in getattr(manager, "all_navigation_selectors", None) or ():
            for target in _hook_targets(selector):
                hooks = getattr(target, "interaction_hooks", None)
                if hooks is not None and self.interaction_hook not in hooks:
                    hooks.append(self.interaction_hook)
                    self._hooked.append(target)
            if self._last_index is not None:
                continue
            try:
                prepared = _prepare_nav_indices(
                    self.tree.root, selector.current_indices,
                    selector.is_integrating, data=_NAVIGATION_AXES)
                if prepared is None:
                    continue
                if np.ndim(prepared) > 1:
                    # A region reads through region_frame, i.e. its centre.
                    prepared = np.mean(np.asarray(prepared), axis=0).astype(int)
                self._last_index = tuple(
                    int(v) for v in np.atleast_1d(np.asarray(prepared)).ravel())
            except Exception as e:
                log.debug("[%s] seeding the latch position failed: %s",
                          self.name, e)

    # ── teardown ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop previewing and hand the signal plot back.

        The reader is unpinned ONLY while it is still ours: the action's
        finalize runs first and pins the real display, and unpinning that would
        paint the finished window black.
        """
        if self._closed:
            return
        self._closed = True
        signal = getattr(self.tree, "root", None)
        try:
            if (signal is not None
                    and self.tree.reader_override_for(signal) is self):
                self.tree.set_reader_override(signal, None)
        except Exception as e:
            log.debug("[%s] releasing the preview reader failed: %s", self.name, e)
        for target in self._hooked:
            try:
                if self.interaction_hook in target.interaction_hooks:
                    target.interaction_hooks.remove(self.interaction_hook)
            except Exception as e:
                log.debug("[%s] removing interaction hook failed: %s",
                          self.name, e)
        self._hooked = []
        if getattr(self.tree, "_live_signal_preview", None) is self:
            self.tree._live_signal_preview = None

    # aliases so the preview can ride the generic teardown paths
    remove = close


def attach_signal_preview(session, tree, *, render: Callable[[tuple], Any],
                          nav_shape: Sequence[int], name: str = "live-signal",
                          **kwargs) -> ProgressiveSignalPreview | None:
    """Attach a :class:`ProgressiveSignalPreview` to *tree* and pin it as the
    reader the tree's signal plots read through.

    Returns the preview, or ``None`` when *tree* is not a navigator + signal
    window — the Orientation / EBSD IPF result windows are a single 2-D plot
    with no navigator, so there is no signal plot to preview into and this is a
    documented no-op rather than an error. The preview is stored as
    ``tree._live_signal_preview`` (the ownership map in ``actions/README.md``:
    per-run state lives on the tree, so ``BaseSignalTree.close()`` tears it down).
    """
    try:
        if (not getattr(tree, "signal_plots", None)
                or getattr(tree, "navigator_plot_manager", None) is None):
            log.debug("[%s] no navigator to signal link on this tree; "
                      "live signal preview skipped", name)
            return None
        preview = ProgressiveSignalPreview(session, tree, render=render,
                                           nav_shape=nav_shape, name=name,
                                           **kwargs)
        tree.set_reader_override(tree.root, preview)
        for plot in list(tree.signal_plots):
            plot.needs_auto_level = True
        preview.watch_navigators()
    except Exception as e:
        log.debug("attaching live signal preview failed: %s", e)
        return None
    tree._live_signal_preview = preview
    return preview
