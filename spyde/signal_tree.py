from __future__ import annotations

import logging
import threading
import time
from functools import partial
from typing import TYPE_CHECKING, Iterator, List, Union

import numpy as np
import dask.array as da
from psygnal import Signal
from hyperspy.signal import BaseSignal

from spyde.signal_node import SignalNode

if TYPE_CHECKING:
    from spyde.drawing.plots.plot import Plot
    from spyde.drawing.plots.plot_window import PlotWindow
    from spyde.drawing.plots.plot_states import PlotState
    from spyde.drawing.plots.multiplot_manager import MultiplotManager
    from spyde.backend.session import Session

logger = logging.getLogger(__name__)

#: Chunks that must still be OUTSTANDING for the threaded navigator fill to hand
#: over to the cluster mid-flight. Handing over recomputes the chunks already
#: painted, so with only a handful left that costs more than it saves; with a
#: movie's worth left it is the difference between seconds and minutes. The
#: check runs per chunk, so in practice this fires within the first few.
_NAV_HANDOVER_MIN_CHUNKS = 8



def _materialise_signal(signal: BaseSignal, array: np.ndarray) -> None:
    """Swap a LAZY signal's dask data for an in-RAM array, the way hyperspy's own
    ``LazySignal.compute`` does it (``data`` + ``_lazy`` + ``_assign_subclass``).

    Leaving ``_lazy`` set with numpy data would route every navigator read down
    the LAZY branch of ``update_from_navigation_selection``, which needs a dask
    array (``.compute``/``.chunks``) and would fall through to a
    ``_get_cache_dask_chunk`` that has nothing to read."""
    try:
        signal.data = array
        if getattr(signal, "_lazy", False):
            signal._lazy = False
            signal._assign_subclass()
    except Exception as e:
        logger.debug("materialising %r after the navigator fill failed: %s",
                     signal, e)


class BaseSignalTree:
    """
    A class to manage the signal tree — the DAG of signal transformations.

    Each node is a HyperSpy BaseSignal with associated Plot(s).
    Non-breaking transformations update the current plot in-place; breaking
    transformations create new branches.

    Parameters
    ----------
    root_signal : BaseSignal
    session : Session
    distributed_client : distributed.Client, optional
    """

    def __init__(
        self,
        root_signal: BaseSignal,
        session: "Session",
        distributed_client=None,
        selector_type=None,
        navigator_override: BaseSignal = None,
        source_path: "str | None" = None,
    ):
        self.root = root_signal
        self.session = session

        self.navigator_signals: dict[str, BaseSignal] = {}
        self.root_node = SignalNode(signal=root_signal, name="root", parent=None)
        self._client_override = distributed_client
        self._selector_type = selector_type
        self._pending_nav_dask: da.Array | None = None
        # True when _pending_nav_dask is the DEEP (…lead…, y, x) nav-sum of a
        # 5-D+ dataset rather than an already-reduced navigator image — that
        # array feeds BOTH navigators in one pass (see
        # _start_progressive_nav_compute's recursive branch).
        self._pending_nav_deep: bool = False
        # Set by _compute_navigator when a DEEP navigator was served from the
        # sidecar cache, so _preprocess_navigator can hand the cached array to
        # the child navigator's signal too. Consumed immediately.
        self._nav_deep_cached: np.ndarray | None = None
        # On-disk origin of the root signal (None for derived/test trees) —
        # enables the navigator sidecar cache (spyde.nav_sidecar).
        self.source_path = source_path

        if navigator_override is not None:
            navigator = self._preprocess_navigator(navigator_override)
        else:
            # Only the BASE navigator (the root signal's own sum) may be served
            # from / saved to the sidecar — an override (e.g. a vectors count
            # map) or a later add_navigator_signal can share the nav shape but
            # holds a DIFFERENT quantity.
            self._sidecar_eligible = True
            try:
                navigator = self._initialize_navigator(root_signal)
            finally:
                self._sidecar_eligible = False
        self.navigator_signals["base"] = navigator

        self.signal_plots: list[Plot] = []
        self.navigator_plot_manager: "MultiplotManager | None" = None

        # Cancellation registry: heavy actions register a stopped_flag (a 1-elem
        # [False] list the compute cores already poll) and/or a live dask Future
        # here so close() can STOP in-flight compute instead of letting it run to
        # completion on the cluster. See _cancel_all_compute / register_cancel.
        self._cancel_flags: list[list] = []
        self._cancel_futures: list = []

        self._initialize_initial_plots()
        logger.debug("Created signal tree with root %s", self.root)

    # ── Compute cancellation ──────────────────────────────────────────────────

    def register_cancel(self, *, flag: "list | None" = None, future=None):
        """Register an in-flight compute for cancellation on tree close.

        ``flag`` is a 1-element ``[False]`` list the compute core polls (set to
        ``True`` to request stop). ``future`` is a dask Future (or a
        ``.cancel()``-able) to cancel outright. Returns ``flag`` (creating one
        if not given) so callers can do ``flag = tree.register_cancel()`` and
        pass it straight into the core. Already-closed trees flip the flag
        immediately so a late registration can't outlive the tree."""
        if flag is None:
            flag = [False]
        if getattr(self, "_spyde_closed", False):
            # Tree already torn down — don't let this compute keep running.
            flag[0] = True
            if future is not None:
                try:
                    future.cancel()
                except Exception as e:
                    logger.debug("cancelling late-registered future failed: %s", e)
            return flag
        self._cancel_flags.append(flag)
        if future is not None:
            self._cancel_futures.append(future)
        return flag

    def unregister_cancel(self, *, flag=None, future=None) -> None:
        """Drop a finished compute's flag/future so the registry doesn't grow
        without bound across many runs on a long-lived tree."""
        if flag is not None:
            try:
                self._cancel_flags.remove(flag)
            except ValueError:
                pass
        if future is not None:
            try:
                self._cancel_futures.remove(future)
            except ValueError:
                pass

    def _cancel_all_compute(self) -> None:
        """Flip every registered stopped_flag and cancel every registered
        future. Called FIRST in close() so the compute cores bail out on their
        next poll and the cluster stops working on a tree that's going away."""
        for flag in list(self._cancel_flags):
            try:
                flag[0] = True
            except Exception as e:
                logger.debug("setting cancel flag on close failed: %s", e)
        client = self.client
        try:
            from dask.distributed import Future as _Future
        except Exception:
            _Future = ()
        for fut in list(self._cancel_futures):
            try:
                # `client.cancel(x)` runs futures_of(x), which silently returns
                # [] for anything that isn't a dask Future — so a duck-typed
                # handle (the navigator fill's _AssembledFuture, which cancels
                # the whole dispatch) would never be cancelled at all. Route it
                # to its own .cancel().
                if client is not None and isinstance(fut, _Future):
                    client.cancel(fut)
                elif hasattr(fut, "cancel"):
                    fut.cancel()
            except Exception as e:
                logger.debug("cancelling registered future on close failed: %s", e)
        self._cancel_flags.clear()
        self._cancel_futures.clear()

    @property
    def client(self):
        """The Dask distributed client — read LIVE from the session's
        DaskManager so a tree created *before* the cluster finished starting
        still picks it up (the cluster takes ~10 s; examples load sooner). Falls
        back to the override passed at construction."""
        mgr = getattr(self.session, "dask_manager", None) if self.session else None
        live = getattr(mgr, "client", None) if mgr is not None else None
        return live if live is not None else self._client_override

    def open(self) -> None:
        """Called by Session after construction to open MDI windows."""
        # _initialize_initial_plots already ran in __init__; this is a hook
        # for Session to register us and send window descriptors to Electron.
        pass

    # ── Plot initialisation ────────────────────────────────────────────────────

    def _initialize_initial_plots(self) -> None:
        from spyde.drawing.plots.multiplot_manager import MultiplotManager

        if self.root.axes_manager.navigation_dimension > 0:
            self.navigator_plot_manager = MultiplotManager(
                session=self.session,
                signal_tree=self,
                selector_type=self._selector_type,
            )
            if self._pending_nav_dask is not None:
                self._start_nav_compute_after_first_frame()
        else:
            self.navigator_plot_manager = None
            self.add_signal_plot()

    def add_signal_plot(self) -> None:
        pw = self.session.add_plot_window(
            is_navigator=False, signal_tree=self, plot_manager=None
        )
        plot = pw.add_new_plot()
        self.create_plot_states(plot=plot)
        plot.set_plot_state(list(plot.plot_states.keys())[0])
        self.signal_plots.append(plot)

        signal = self.root
        if signal._lazy and self.client is not None:
            # Tracked here (not via a context manager) because the compute
            # outlives this function — it resolves later on the
            # PlotUpdateWorker's poll thread, which marshals the apply onto
            # the main thread as Session._on_plot_ready. That callback (and
            # its "drop a superseded/errored future" branches) is the one
            # place that reliably knows the future is done, so it owns the
            # matching stop — see the finally there.
            from spyde.actions.lifecycle import window_computing
            window_computing(getattr(plot, "window_id", None)).start()
            # NOT a bare `client.compute`: a lazy .tif/.hspy graph holds an open
            # file handle and cannot be pickled to a worker, so that raised here
            # — between the start() above and the _on_plot_ready that stops it —
            # and left the window black on a stuck "Calculating…". See
            # compute_display_future.
            from spyde.drawing.update_functions import compute_display_future
            future = compute_display_future(signal.data, self.client,
                                            label="signal")
            plot.update_data(future)
        else:
            plot.update()

    # ── Progressive navigator compute ─────────────────────────────────────────

    # How long to let the signal plot's FIRST frame win the disk before the
    # navigator fill starts anyway. The frame read is ~0.1–2 s even cold; the
    # timeout only matters if that read never lands (then the fill proceeds).
    _FIRST_FRAME_WAIT_S = 6.0

    def _start_nav_compute_after_first_frame(self) -> None:
        """Start the progressive navigator fill only after the signal plot's
        FIRST frame has painted (bounded by _FIRST_FRAME_WAIT_S).

        On a cold large file the DISTRIBUTED fill reads the WHOLE dataset
        across Dask worker processes — which `_interactive_activity` cannot
        throttle (it only yields the in-process threaded fill) — while the
        initial frame's own direct read was submitted to the dispatcher just
        milliseconds earlier. That one frame loses the disk to hundreds of
        chunk-sum reads and the signal panel sits black well into the fill
        ("black until you move the navigator"). Reading it FIRST costs the fill
        at most a couple of seconds; the frame gets the disk to itself.

        The pending nav dask is consumed NOW (not at fire time) so a stash made
        during the wait (e.g. add_navigator_signal) can't be mistaken for the
        base navigator."""
        nav_dask = self._pending_nav_dask
        deep = self._pending_nav_deep
        self._pending_nav_dask = None
        self._pending_nav_deep = False
        if nav_dask is None:
            return
        stop = threading.Event()
        self._nav_defer_stop = stop

        def _wait_then_start():
            deadline = time.monotonic() + self._FIRST_FRAME_WAIT_S
            while time.monotonic() < deadline and not stop.is_set():
                if any(isinstance(p.current_data, np.ndarray)
                       for p in self.signal_plots):
                    break
                time.sleep(0.05)
            if not stop.is_set():
                try:
                    self._start_progressive_nav_compute(nav_dask, deep=deep)
                except Exception:
                    # Session may be tearing down mid-wait; a blank navigator
                    # beats an unhandled thread exception.
                    logger.exception("deferred navigator fill failed to start")

        threading.Thread(target=_wait_then_start, daemon=True,
                         name="nav-defer").start()

    def _start_progressive_nav_compute(self, nav_dask: "da.Array | None" = None,
                                       deep: "bool | None" = None) -> None:
        """
        Replace the single-future nav compute with a per-chunk progressive
        compute that live-updates the navigator image as chunks finish.

        ``nav_dask`` is normally handed over by the deferral
        (_start_nav_compute_after_first_frame, which consumed the stash up
        front); falling back to the stash keeps direct calls working.

        ``deep`` marks ``nav_dask`` as the DEEP ``(…lead…, y, x)`` nav-sum of a
        5-D+ dataset, which drives BOTH of that dataset's navigators from ONE
        pass over the data — see the recursive branch below.
        """
        from spyde.drawing.update_functions import (
            compute_with_live_buffer,
            ensure_live_buffer,
            graph_can_reach_workers,
            read_live_buffer,
            _interactive_activity,
        )

        if nav_dask is None:
            nav_dask = self._pending_nav_dask
            if deep is None:
                deep = self._pending_nav_deep
            self._pending_nav_dask = None
            self._pending_nav_deep = False
        if nav_dask is None:
            return
        deep = bool(deep)

        nav_signals = self.navigator_signals.get("base")
        nav_plot_windows = list(self.navigator_plot_manager.plot_windows.keys())
        if not nav_plot_windows:
            return
        nav_pw = nav_plot_windows[0]
        nav_plots = self.navigator_plot_manager.plots.get(nav_pw, [])
        if not nav_plots:
            return
        nav_plot = nav_plots[0]

        # ── RECURSIVE (5-D+) fill: ONE pass feeds BOTH navigators ────────────
        # A 5-D stack opens TWO navigators — a 2-D real-space image and a 1-D
        # time line. They are not independent: the real-space image at time t IS
        # the deep nav-sum's plane ``deep[t]``, and the time line is that plane
        # summed. So the fill computes the DEEP array progressively and derives
        # the top navigator from the planes it already has, instead of a second
        # full pass over the dataset (which is what stashing the already-reduced
        # 1-D sum used to force — one whole-dataset read per point of a 4-point
        # line, so the fill looked frozen and the real-space navigator got no
        # progressive fill at all).
        #
        # If the child navigator can't be resolved, reduce the deep array here
        # and take the ordinary single-navigator path (the old behaviour).
        deep_targets = self._deep_nav_targets() if deep else None
        if deep and deep_targets is None:
            logger.debug("deep navigator fill: child navigator not resolvable; "
                         "falling back to the single-navigator sum")
            nav_dask = nav_dask.sum(axis=(-2, -1))
            deep = False
        nav_shape = tuple(nav_dask.shape)

        # A lazy .tif/.hspy graph holds an open file handle, so it cannot be
        # pickled to a worker at ALL (see update_functions.compute_display_future
        # for the full story). Such a fill can only run in this process — which
        # is exactly the threaded branch below, already the proven path for "the
        # cluster isn't up yet".
        #
        # Memoised, because the handover check inside that loop asks the same
        # question every chunk (and must, or it would hand the unsendable graph
        # to the dispatcher the moment the cluster appears). Sendability is a
        # property of the graph, so one probe answers for the whole fill; the
        # LIVE `self.client is not None` half stays outside the memo, since a
        # cluster arriving mid-fill is the case the handover exists for.
        _sendable: list[bool] = []

        def _graph_sendable(_dask=nav_dask) -> bool:
            if not _sendable:
                _sendable.append(graph_can_reach_workers(_dask))
                if not _sendable[0]:
                    logger.info("navigator fill: this graph cannot be serialized "
                                "to the workers (lazy .tif/.hspy hold an open "
                                "file handle) — filling in-process instead")
            return _sendable[0]

        can_distribute = self.client is not None and _graph_sendable()

        logger.debug(
            "NAV-DEBUG _start_progressive_nav_compute: path=%s deep=%s "
            "nav_shape=%s chunks=%s client=%s",
            "DISTRIBUTED" if can_distribute else "THREADED",
            deep, nav_shape, nav_dask.chunks, type(self.client).__name__,
        )

        # No cluster yet (it takes ~10 s to start; examples load sooner, and a
        # huge MRC's navigator sum can take minutes): compute the navigator on a
        # BACKGROUND thread with the threaded scheduler so the already-displayed
        # window stays interactive (crosshair works) while it fills in.
        #
        # Compute PER NAV-CHUNK and paint after each, so the navigator fills
        # PROGRESSIVELY (top-to-bottom) instead of staying blank until the whole
        # multi-GB sum finishes — that "blank navigator that never fills" was the
        # symptom on the large Windows scan.
        if not can_distribute:
            import itertools

            if not deep:
                # DEEP: nav_shape is the (…lead…, y, x) accumulator's shape, which
                # is NOT the top navigator's — its placeholder/levels are owned by
                # _paint_deep_nav instead.
                nav_plot.current_data = np.full(nav_shape, np.nan, dtype=np.float32)
            # Stop flag so the thread bails out cleanly on tree/session shutdown
            # instead of painting onto a torn-down plot.
            stop = threading.Event()
            self._nav_stop = stop

            def _bg_nav(_dask=nav_dask, _plot=nav_plot, _sig=nav_signals,
                        _shape=nav_shape, _stop=stop, _deep=deep,
                        _targets=deep_targets, _sendable_probe=_graph_sendable):
                # window_computing brackets the WHOLE fill (both early-return
                # paths on _stop AND the exception handler below) so a
                # cancelled/failed fill still clears the renderer's overlay —
                # see lifecycle.window_computing.
                from spyde.actions.lifecycle import window_computing
                computing = window_computing(getattr(_plot, "window_id", None))
                computing.start()
                try:
                    acc = np.full(_shape, np.nan, dtype=np.float32)
                    levels = [None]
                    # Walk the navigation chunk grid; compute + paint each block.
                    axes_ranges = []
                    for axis_chunks in _dask.chunks[: len(_shape)]:
                        pos, start = [], 0
                        for size in axis_chunks:
                            pos.append((start, size))
                            start += size
                        axes_ranges.append(pos)
                    total_chunks = int(np.prod([len(r) for r in axes_ranges]))
                    done_chunks = 0
                    for combo in itertools.product(*axes_ranges):
                        if _stop.is_set():
                            return
                        # HAND OVER to the cluster the moment it exists.
                        #
                        # This branch was chosen because `self.client` was None
                        # at the START — the cluster takes ~10 s and a file
                        # opened right after launch beats it. Without this check
                        # that decision was PERMANENT: the whole fill then ran
                        # single-threaded, one chunk at a time, however many
                        # workers turned up a second later. On a 977-frame movie
                        # that is the difference between seconds and minutes,
                        # and it looked exactly like "the cluster is idle".
                        #
                        # It bit movies and not 4D-STEM scans purely by timing:
                        # a scan goes through the nav-shape prompt (a human
                        # round trip) so the cluster is always up by the time
                        # its tree is built, while a movie opens straight
                        # through.
                        #
                        # Re-enter rather than switch in place: the distributed
                        # branch below owns cancellation, the sidecar save and
                        # the final repaint, and duplicating any of that here is
                        # how the two paths drift apart. The chunks already
                        # painted are recomputed, which is why the threshold
                        # exists — a handover is only worth it with real work
                        # left, and this fires within the first few chunks.
                        # `_sendable()` as well as a live client: a graph that
                        # cannot be pickled to a worker must never be handed
                        # over, however many workers turn up.
                        if (self.client is not None
                                and _sendable_probe()
                                and (total_chunks - done_chunks)
                                > _NAV_HANDOVER_MIN_CHUNKS):
                            logger.info(
                                "navigator fill: cluster came up after %d/%d "
                                "chunks — handing the rest to the dispatcher",
                                done_chunks, total_chunks)
                            self._start_progressive_nav_compute(_dask, deep=_deep)
                            return
                        # Yield the disk to active scrubbing: for a large movie
                        # this per-chunk sum reads the whole file, which otherwise
                        # starves the crosshair's own frame read (the signal plot
                        # freezes while the navigator fills). Pause briefly while
                        # the user is actively moving; resume when they settle.
                        # Under a sustained drag this advances ~one chunk per the
                        # wait cap (interaction wins, but the fill still finishes).
                        # `stop` aborts the wait promptly on teardown.
                        _interactive_activity.wait_if_active(stop=_stop)
                        nav_slices = tuple(slice(s, s + n) for s, n in combo)
                        logger.debug("NAV-DEBUG threaded nav chunk %s computing", nav_slices)
                        # scheduler="threads" EXPLICITLY. A bare .compute() uses
                        # dask's default, and a live distributed Client registers
                        # ITSELF as that default — so this "threaded" branch
                        # silently shipped the chunk to the cluster whenever one
                        # existed. Harmless while the branch only ran with no
                        # client; now that an unsendable graph is routed here with
                        # a cluster up, it is the difference between filling and
                        # raising "cannot pickle 'BufferedReader'".
                        block = np.asarray(
                            _dask[nav_slices].compute(scheduler="threads")
                        ).astype(np.float32)
                        acc[nav_slices] = block
                        done_chunks += 1
                        self._emit_nav_progress(done_chunks / max(1, total_chunks))
                        if _stop.is_set():
                            return
                        if _deep:
                            # Recursive paint: the real-space plane the child
                            # navigator is showing, then the top navigator DERIVED
                            # from the planes finished so far.
                            self._paint_deep_nav(acc, _targets)
                            continue
                        finite = acc[np.isfinite(acc)]
                        if finite.size:
                            # Robust percentile levels (2–98%), computed over ALL
                            # painted-so-far data — NOT raw min/max, which a single
                            # bright outlier in one chunk yanks around so the
                            # contrast (and apparent chunk-boundary brightness)
                            # jumps as each chunk lands. Percentiles keep the
                            # stretch stable across the progressive fill.
                            lo, hi = np.percentile(finite, (2.0, 98.0))
                            levels[0] = (float(lo), float(hi) if hi > lo else float(lo) + 1)
                        _plot.set_data(acc.copy(), levels=levels[0])
                    # Final uniform repaint: now that every chunk is in, set the
                    # definitive levels over the whole image so no transient
                    # per-chunk stretch remains visible at a boundary, and emit a
                    # histogram so the navigator gets a Plot-Control histogram /
                    # contrast handles (set_data with explicit levels otherwise
                    # skips _emit_histogram, leaving the navigator histogram-less).
                    if not _stop.is_set():
                        if _deep:
                            self._paint_deep_nav(acc, _targets, final=True)
                        else:
                            finite = acc[np.isfinite(acc)]
                            if finite.size:
                                lo, hi = np.percentile(finite, (2.0, 98.0))
                                lvl = (float(lo), float(hi) if hi > lo else float(lo) + 1)
                                _plot.set_data(acc.copy(), levels=lvl)
                                try:
                                    _plot._emit_histogram(acc, lvl[0], lvl[1])
                                except Exception as e:
                                    logger.debug("navigator histogram emit failed: %s", e)
                    if _sig:
                        if _deep:
                            self._commit_deep_nav(acc, _sig)
                        else:
                            _sig[0].data = acc
                    if not _stop.is_set():
                        self._save_nav_sidecar(acc)
                except Exception:
                    # Primary (threaded) navigator load — a failure here leaves a
                    # blank navigator, so surface the traceback rather than hide it.
                    logger.exception("threaded navigator compute failed")
                finally:
                    computing.stop()
            threading.Thread(target=_bg_nav, daemon=True, name="nav-threaded").start()
            return

        # Cancel the single monolithic future submitted by _preprocess_navigator
        if nav_signals:
            old_future = nav_signals[0].data
            from dask.distributed import Future as _Future
            if isinstance(old_future, _Future):
                try:
                    self.client.cancel(old_future)
                except Exception as e:
                    logger.debug("cancelling prior navigator future failed: %s", e)

        shm_name = f"spyde_nav_{id(nav_plot)}"

        shm = ensure_live_buffer(nav_shape, shm_name)
        self._nav_shm = shm

        if not deep:
            # DEEP: nav_shape belongs to the (…lead…, y, x) accumulator, not to
            # this plot — _paint_deep_nav owns both navigators' pixels.
            nav_plot.current_data = np.full(nav_shape, np.nan, dtype=np.float32)
            nav_plot.needs_auto_level = True
            nav_plot.update()

        # Psygnal relay — emitted from the Dask callback thread, slots run on
        # calling thread (safe for anyplotlib's _push which is GIL-protected).
        class _NavChunkRelay:
            chunk_ready = Signal(object, object)

        relay = _NavChunkRelay()
        self._nav_relay = relay

        def _write_chunk(chunk_result, nav_slices, _shm=shm, _shape=nav_shape):
            try:
                buf = np.ndarray(_shape, dtype=np.float32, buffer=_shm.buf)
                buf[nav_slices] = chunk_result.astype(np.float32)
            except Exception as e:
                logger.debug("writing navigator chunk %r to shm failed: %s",
                             nav_slices, e)

        relay.chunk_ready.connect(_write_chunk)

        def _on_chunk(chunk_result, nav_slices):
            relay.chunk_ready.emit(chunk_result, nav_slices)

        # Register the fill for cancellation so a tree close mid-fill stops it
        # on the cluster instead of letting the whole dataset sum run to
        # completion. This is now ONE handle (compute_dispatch owns the
        # per-chunk futures and cancels them from its own stopped_flag) rather
        # than one entry per nav chunk — on a 977-frame movie that was 978
        # registrations. Tracked locally too so the poll loop can unregister
        # when the fill finishes normally.
        nav_futures: list = []

        def _register_nav_future(fut, _futs=nav_futures):
            _futs.append(fut)
            self.register_cancel(future=fut)

        future = compute_with_live_buffer(
            nav_dask, nav_shape, self.client, shm_name,
            on_chunk_done=_on_chunk, on_future=_register_nav_future,
        )
        self._nav_futures = nav_futures

        if not deep:
            # DEEP: this future resolves to the (…lead…, y, x) array, which is
            # neither navigator's displayed image and must never reach
            # PlotUpdateWorker via current_data. The poll loop below paints both
            # navigators from the shm buffer, and _commit_deep_nav hands the
            # finished array to the signals at the end.
            if nav_signals:
                nav_signals[0].data = future
            nav_plot.current_data = future
            nav_plot.needs_auto_level = True

        # Periodic poll: update the displayed image as chunks arrive
        _nav_levels: list = [None]
        _stop = threading.Event()
        self._nav_stop = _stop

        def _paint_from_buffer():
            arr = read_live_buffer(nav_shape, shm_name)
            finite = arr[np.isfinite(arr)]
            self._emit_nav_progress(finite.size / max(1, arr.size))
            if finite.size == 0:
                return
            if deep:
                self._paint_deep_nav(arr, deep_targets)
                return
            lo, hi = float(finite.min()), float(finite.max())
            if _nav_levels[0] is None:
                _nav_levels[0] = (lo, hi if hi > lo else lo + 1)
            elif hi > _nav_levels[0][1]:
                _nav_levels[0] = (_nav_levels[0][0], hi)
            nav_plot.set_data(arr, levels=_nav_levels[0])

        def _poll_loop():
            # window_computing brackets the whole poll (incl. the early return
            # on _stop) via try/finally, so a cancelled fill (tree close, a
            # newer navigator superseding this one) still clears the
            # renderer's overlay — see lifecycle.window_computing.
            from spyde.actions.lifecycle import window_computing
            computing = window_computing(getattr(nav_plot, "window_id", None))
            computing.start()
            try:
                # Paint, THEN check done — and ALWAYS do a final paint after the
                # loop. The old code broke on future.done() at the top of the loop,
                # so the last chunks that landed in the shm buffer between the
                # final 0.1 s poll and completion were never painted → HOLES in the
                # navigator. (The virtual-image progressive poll never had holes
                # precisely because it does this final read; see
                # stream_progressive_to_plot.)
                while not _stop.is_set():
                    try:
                        _paint_from_buffer()
                    except Exception as e:
                        logger.debug("navigator poll paint failed: %s", e)
                    if future.done():
                        break
                    time.sleep(0.1)
                # The nav fill is over (done or stopped) — drop its futures from
                # the cancel registry so it doesn't grow across reloads on a long
                # tree.
                for _f in list(nav_futures):
                    self.unregister_cancel(future=_f)
                if _stop.is_set():
                    return
                # Final repaint of the COMPLETED navigator. `future.result()` is
                # the dispatcher's client-side assembly of every chunk, so it is
                # complete by construction — whereas the shm buffer is written
                # from a psygnal relay whose last emits can still be in flight,
                # and painting that would leave holes. The navigator is small
                # (nav-shaped, ~MB), so paint the AUTHORITATIVE result directly.
                # Fall back to the buffer if the result isn't fetchable.
                try:
                    res = future.result()
                    arr = np.asarray(res, dtype=np.float32)
                    if arr.shape == tuple(nav_shape):
                        if deep:
                            self._paint_deep_nav(arr, deep_targets, final=True)
                            self._commit_deep_nav(arr, nav_signals)
                        else:
                            finite = arr[np.isfinite(arr)]
                            if finite.size > 0:
                                lo, hi = float(finite.min()), float(finite.max())
                                lv = (_nav_levels[0] if _nav_levels[0] is not None
                                      else (lo, hi if hi > lo else lo + 1))
                                nav_plot.set_data(arr, levels=lv)
                        self._save_nav_sidecar(arr)
                    else:
                        _paint_from_buffer()
                except Exception as e:
                    logger.debug("navigator final result paint failed (%s); "
                                 "falling back to buffer", e)
                    try:
                        _paint_from_buffer()
                    except Exception as e2:
                        logger.debug("navigator final buffer paint failed: %s", e2)
            finally:
                computing.stop()

        t = threading.Thread(target=_poll_loop, daemon=True, name="nav-poll")
        t.start()
        self._nav_poll_thread = t

    # ── Recursive (5-D+) two-navigator fill ───────────────────────────────────
    #
    # A 5-D dataset (t, y, x | ky, kx) opens TWO navigators: the top window is
    # the 1-D time line and its selector drives a child 2-D real-space image
    # (MultiplotManager's ``navigation_depth == 2`` branch). Both are reductions
    # of ONE array — the deep nav-sum ``deep[t, y, x] = data[t, y, x].sum()``:
    #
    #     child (real space) = deep[t_selected]        ← a plane
    #     top   (time line)  = deep.sum(axis=(-2, -1)) ← that plane, summed
    #
    # so the fill computes ``deep`` progressively and DERIVES the top navigator
    # from the planes it already has. Before this, ``_compute_navigator`` stashed
    # the already-reduced 1-D sum, which threw the real-space plane away: the
    # dataset was read once per point of a 4-point line (each "chunk" of the fill
    # being a whole 4-D member — no visible progress), and the real-space
    # navigator got no progressive fill at all.

    def _deep_nav_targets(self):
        """``(top_plot, child_plot, top_selector)`` for a DEEP (5-D+) navigator,
        or ``None`` when this tree has no child navigator to fill.

        The top navigator is the tree's only top-level navigator plot window;
        the child is the plot its selector drives that is itself a navigator
        (the third window in the chain is the diffraction pattern, not a
        navigator, and is driven by the child's own selector)."""
        mgr = self.navigator_plot_manager
        if mgr is None:
            return None
        top_windows = list(mgr.plot_windows.keys())
        if not top_windows:
            return None
        top_pw = top_windows[0]
        top_plots = mgr.plots.get(top_pw) or []
        if not top_plots:
            return None
        for selector in mgr.navigation_selectors.get(top_pw) or []:
            for child in selector.children:
                if getattr(child, "is_navigator", False):
                    return top_plots[0], child, selector
        return None

    @staticmethod
    def _reduce_deep_nav(acc: np.ndarray) -> np.ndarray:
        """Reduce a DEEP nav array ``(…lead…, y, x)`` to the top navigator's
        ``(…lead…,)`` by summing each real-space plane.

        A plane that is not FULLY filled yet stays NaN: a partially-summed time
        point is a WRONG value, not a dim one, and the chunk walk completes one
        lead index at a time (``itertools.product`` iterates the leading axis
        outermost), so the line grows point-by-point instead of dipping and
        jumping as each chunk lands."""
        acc = np.asarray(acc)
        flat = acc.reshape(acc.shape[: acc.ndim - 2] + (-1,))
        finite = np.isfinite(flat)
        out = np.where(finite, flat, 0.0).sum(axis=-1, dtype=np.float64)
        out = out.astype(np.float32)
        out[~finite.all(axis=-1)] = np.nan
        return out

    @staticmethod
    def _robust_levels(arr: np.ndarray):
        """Robust 2–98% display levels over the FULL accumulated finite data —
        the same stretch the 4-D fill uses, so a bright outlier in one chunk
        can't yank the contrast around as the fill progresses."""
        finite = np.asarray(arr)[np.isfinite(arr)]
        if not finite.size:
            return None
        lo, hi = np.percentile(finite, (2.0, 98.0))
        return (float(lo), float(hi) if hi > lo else float(lo) + 1)

    def _deep_nav_plane(self, acc: np.ndarray, selector) -> np.ndarray:
        """The real-space plane the CHILD navigator is currently showing — the
        mean over whichever lead (time) positions the top selector covers, so an
        integrating span reads the same as the selector's own frame."""
        acc = np.asarray(acc)
        lead_ndim = acc.ndim - 2
        rows = [(0,) * lead_ndim]
        try:
            idx = np.atleast_2d(np.asarray(selector.get_selected_indices()))
            idx = idx[:, :lead_ndim].astype(int)
            if lead_ndim >= 2:
                # A 2-D top navigator reports widget (x, y); the innermost pair
                # must be swapped to data order — same rule as
                # update_functions._prepare_nav_indices.
                idx[:, -2:] = idx[:, -2:][:, ::-1]
            limits = np.asarray(acc.shape[:lead_ndim]) - 1
            rows = list(dict.fromkeys(
                tuple(int(v) for v in np.clip(row, 0, limits)) for row in idx))
        except Exception as e:
            logger.debug("resolving the deep navigator's lead position failed: %s", e)
        planes = np.stack([acc[row] for row in rows]) if len(rows) > 1 else acc[rows[0]]
        if planes.ndim == 2:
            return planes
        finite = np.isfinite(planes)
        count = finite.sum(axis=0)
        total = np.where(finite, planes, 0.0).sum(axis=0)
        return np.divide(total, count, where=count > 0,
                         out=np.full(count.shape, np.nan, dtype=np.float64)
                         ).astype(np.float32)

    def _paint_deep_nav(self, acc: np.ndarray, targets, *, final: bool = False) -> None:
        """Paint BOTH navigators from the deep accumulator.

        The child navigator is only repainted when the fill's plane holds MORE
        finite pixels than what is already displayed (or on the final pass), so
        a half-filled plane can never clobber a complete frame the child's own
        selector read just painted. The top navigator has no other writer."""
        if targets is None:
            return
        top_plot, child_plot, selector = targets
        try:
            plane = self._deep_nav_plane(acc, selector)
            n_new = int(np.isfinite(plane).sum())
            current = getattr(child_plot, "current_data", None)
            n_cur = (int(np.isfinite(current).sum())
                     if isinstance(current, np.ndarray)
                     and current.shape == plane.shape else -1)
            if n_new > 0 and (final or n_new > n_cur):
                levels = self._robust_levels(plane)
                child_plot.set_data(plane.copy(), levels=levels)
                if final and levels is not None:
                    child_plot._emit_histogram(plane, levels[0], levels[1])
        except Exception as e:
            logger.debug("deep navigator child paint failed: %s", e)
        try:
            curve = self._reduce_deep_nav(acc)
            levels = self._robust_levels(curve)
            if levels is not None:
                top_plot.set_data(curve.copy(), levels=levels)
                if final:
                    top_plot._emit_histogram(curve, levels[0], levels[1])
        except Exception as e:
            logger.debug("deep navigator top paint failed: %s", e)

    def _commit_deep_nav(self, acc: np.ndarray, nav_signals) -> None:
        """Hand the COMPLETED deep array to the signals it backs.

        The child navigator's ``(…lead…| y, x)`` signal becomes the in-RAM array
        so a later time-slider move is a numpy slice instead of another
        whole-time-slice dask read; the top navigator gets the derived curve."""
        if not nav_signals:
            return
        acc = np.asarray(acc, dtype=np.float32)
        if len(nav_signals) > 1:
            _materialise_signal(nav_signals[1], acc)
        try:
            nav_signals[0].data = self._reduce_deep_nav(acc)
        except Exception as e:
            logger.debug("committing the derived top navigator failed: %s", e)
        logger.info(
            "navigator fill complete (recursive): deep %s → real-space %s + "
            "derived %s", acc.shape, acc.shape[-2:], acc.shape[:-2])

    # ── Navigator processing ───────────────────────────────────────────────────

    def _save_nav_sidecar(self, arr) -> None:
        """Persist the COMPLETED navigator beside the source file (best-effort)
        so the next open of this dataset skips the whole-file navigator read.
        Called from the fill threads once the array is authoritative."""
        path = self.source_path
        if not path:
            return
        try:
            from spyde.nav_sidecar import save_nav_sidecar
            if save_nav_sidecar(path, np.asarray(arr)):
                from de_shell.ipc import emit_status
                emit_status("Navigator ready (cached for the next open)")
        except Exception as e:
            logger.debug("navigator sidecar save failed: %s", e)

    def _emit_nav_progress(self, frac: float) -> None:
        """Throttled status-bar progress for the navigator fill ("Computing
        navigator… 35%") — a large file's fill reads the whole dataset (minutes)
        and without a live number it reads as a hang. Emits on ≥5% steps only."""
        try:
            pct = int(max(0.0, min(1.0, frac)) * 100)
            last = getattr(self, "_nav_progress_pct", -5)
            if pct >= last + 5 or (pct >= 100 and last < 100):
                self._nav_progress_pct = pct
                from de_shell.ipc import emit_status
                emit_status(f"Computing navigator… {pct}%")
        except Exception as e:
            logger.debug("navigator progress emit failed: %s", e)

    def _preprocess_navigator(self, signal: BaseSignal) -> List[BaseSignal]:
        heavy_workers = getattr(self.session.dask_manager, "heavy_workers", None)
        if (
            signal.axes_manager.navigation_shape + signal.axes_manager.signal_shape
        ) != self.root.axes_manager.navigation_shape:
            raise ValueError(
                "Navigator signal must have the same total number of dimensions "
                "as the root signal and the same shape."
            )

        if signal.axes_manager.signal_dimension == 0:
            signal = signal.T

        if (
            signal.axes_manager.signal_dimension > 0
            and signal.axes_manager.navigation_dimension > 0
        ):
            navigator = signal.sum(signal.axes_manager.signal_axes).T
            if navigator._lazy:
                navigator.data = self._compute_navigator(navigator.data, heavy_workers)
            return [navigator, signal]

        elif signal.axes_manager.signal_dimension > 2:
            # DEEP navigator (5-D+): `signal` is the (…lead…| y, x) nav-sum that
            # backs the CHILD real-space navigator, `navigator` its reduction over
            # real space — the TOP (time) navigator. Hand the progressive fill the
            # DEEP array: one pass over the data fills both (the child is a plane
            # of it, the top the sum of that plane). Stashing the already-reduced
            # `navigator.data` instead threw the real-space plane away, so the
            # child navigator got no progressive fill at all and the top one
            # advanced a point per whole-dataset read.
            signal = signal.transpose(2)
            navigator = signal.sum(signal.axes_manager.signal_axes).T
            if navigator._lazy:
                self._nav_deep_cached = None
                navigator.data = self._compute_navigator(
                    navigator.data, heavy_workers, deep_nav_dask=signal.data)
                cached, self._nav_deep_cached = self._nav_deep_cached, None
                if cached is not None:
                    # Sidecar hit: the deep array is already in RAM, so the child
                    # navigator reads it directly (no compute at all).
                    _materialise_signal(signal, cached)
            return [navigator, signal]

        if signal._lazy:
            signal.data = self._compute_navigator(signal.data, heavy_workers)
        return [signal]

    def _compute_navigator(self, nav_dask: da.Array, heavy_workers,
                           deep_nav_dask: "da.Array | None" = None):
        """Stash the navigator sum for the progressive compute — NEVER blocking.

        The display must not wait on the navigator compute (tree ``__init__``
        runs on the load thread and emits the windows right after this), so we
        only stash ``nav_dask`` and return a NaN placeholder. The single
        authoritative compute is owned by ``_start_progressive_nav_compute``
        (per-chunk progressive, for BOTH the distributed and threaded paths).

        Do NOT submit a monolithic ``client.compute(nav_dask)`` here: it would
        start the full navigator sum on the cluster, then
        ``_start_progressive_nav_compute`` immediately cancels it and resubmits
        the same sum per-chunk — and the cancel races the already-running
        compute, so the sum runs TWICE on the cluster (visible as a duplicate
        task graph on the Dask dashboard). Deferring to the progressive path is
        the one-and-only compute.

        A matching navigator SIDECAR (saved beside the source file by a prior
        fill — see spyde.nav_sidecar) short-circuits all of this: the whole-file
        read is skipped and the cached array IS the navigator. Base navigator
        only (``_sidecar_eligible``) — overrides/extra navigators can share the
        shape but hold different quantities.

        ``deep_nav_dask`` (5-D+ only) is the UNREDUCED ``(…lead…, y, x)`` nav-sum.
        When given it — not ``nav_dask`` — is what the progressive fill computes
        and what the sidecar caches, because that one array drives BOTH of the
        dataset's navigators; the returned placeholder still has ``nav_dask``'s
        (top-navigator) shape, and a sidecar hit is handed back to the caller via
        ``_nav_deep_cached`` so the child navigator's signal gets it too.
        """
        source = nav_dask if deep_nav_dask is None else deep_nav_dask
        if getattr(self, "_sidecar_eligible", False) and self.source_path:
            from spyde.nav_sidecar import load_nav_sidecar
            cached = load_nav_sidecar(self.source_path, tuple(source.shape))
            if cached is not None:
                logger.info("navigator loaded from sidecar for %s (shape=%s) — "
                            "skipping the full-dataset compute",
                            self.source_path, cached.shape)
                cached = cached.astype(np.float32, copy=False)
                if deep_nav_dask is None:
                    return cached
                self._nav_deep_cached = cached
                return self._reduce_deep_nav(cached)
        self._pending_nav_dask = source
        self._pending_nav_deep = deep_nav_dask is not None
        logger.debug(
            "NAV-DEBUG _compute_navigator: stashed nav sum (%s, deep=%s); "
            "progressive compute owns it. shape=%s chunks=%s heavy_workers=%s",
            "DISTRIBUTED" if self.client is not None else "THREADED",
            deep_nav_dask is not None,
            tuple(source.shape), source.chunks, heavy_workers,
        )
        return np.full(tuple(nav_dask.shape), np.nan, dtype=np.float32)

    def add_navigator_signal(self, name: str, signal: BaseSignal) -> None:
        signal = self._preprocess_navigator(signal)
        self.navigator_signals[name] = signal
        self.navigator_plot_manager.add_plot_states_for_navigation_signals(signal)
        # Refresh the navigator chip strip (appears once there are ≥2).
        try:
            from spyde.actions.navigator_views import emit_navigator_options
            emit_navigator_options(self)
        except Exception as e:
            logger.debug("navigator options emit failed: %s", e)

    def _initialize_navigator(self, signal: BaseSignal):
        if signal.axes_manager.navigation_dimension == 0:
            return
        if signal._lazy and signal.navigator is not None:
            navigation_signal = signal.navigator
        else:
            navigation_signal = signal.sum(signal.axes_manager.signal_axes)
        if not isinstance(navigation_signal, BaseSignal):
            navigation_signal = BaseSignal(navigation_signal)
        return self._preprocess_navigator(navigation_signal)

    # ── Tree traversal & mutation ──────────────────────────────────────────────

    def walk(self) -> Iterator[SignalNode]:
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def signals(self) -> List[BaseSignal]:
        return [node.signal for node in self.walk()]

    def create_plot_states(self, plot: "Plot" = None) -> dict:
        for signal in self.signals():
            dynamic = signal.axes_manager.navigation_dimension > 0
            plot.add_plot_state(
                signal=signal,
                dynamic=dynamic,
                dimensions=signal.axes_manager.signal_dimension,
            )
        return {}

    def update_plot_states(self, new_signal: BaseSignal) -> None:
        from spyde.drawing.plots.plot_states import PlotState

        dynamic = new_signal.axes_manager.navigation_dimension > 0
        for plot in self.signal_plots:
            if new_signal not in plot.plot_states:
                plot.plot_states[new_signal] = PlotState(
                    signal=new_signal, plot=plot, dynamic=dynamic
                )

    @property
    def nav_dim(self) -> int:
        return self.root.axes_manager.navigation_dimension

    @property
    def plot_windows(self) -> list["PlotWindow"]:
        if self.navigator_plot_manager is None:
            return []
        return list(self.navigator_plot_manager.plot_windows.keys())

    def get_nested_attr(self, attr_path: str):
        if not attr_path:
            return self
        current_obj = self
        for attr in (p for p in attr_path.split(".") if p):
            current_obj = getattr(current_obj, attr, None)
            if current_obj is None:
                return None
        return current_obj

    def get_node(self, signal) -> SignalNode | None:
        for node in self.walk():
            if node.signal is signal:
                return node
        return None

    def resolve_locality(self, signal) -> bool:
        """Is ``signal``'s node ArrayCache-eligible for the fast per-frame
        path? See spyde/array_cache/locality.py — memoized on the node, walks
        ancestry only on first call. A signal not found in this tree resolves
        opaque (False) — the same fail-safe default as an untagged node."""
        from spyde.array_cache.locality import resolve_locality

        node = self.get_node(signal)
        if node is None:
            return False
        return resolve_locality(node)

    def add_node(
        self,
        parent_signal,
        new_signal,
        transformation: str,
        *,
        local: bool | None = None,
    ) -> None:
        parent_node = self.get_node(parent_signal)
        if parent_node is None:
            raise ValueError("Parent node not found in the tree.")
        final_name = transformation
        if final_name in parent_node.children:
            count = 1
            while f"{transformation}_{count}" in parent_node.children:
                count += 1
            final_name = f"{transformation}_{count}"
        parent_node.children[final_name] = SignalNode(
            signal=new_signal,
            name=final_name,
            parent=parent_node,
            transformation=transformation,
            local=local,
        )

    def add_transformation(
        self,
        parent_signal,
        method: str = None,
        function: callable = None,
        node_name: str = None,
        *args,
        local: bool | None = None,
        **kwargs,
    ) -> BaseSignal | None:
        from de_shell.ipc import emit_error

        if method is not None:
            try:
                new_signal = getattr(parent_signal, method)(*args, **kwargs)
            except Exception as e:
                emit_error(
                    f"Transformation '{method}' failed: {e}"
                )
                return None
        else:
            new_signal = function(parent_signal, *args, **kwargs)

        parent_node = self.get_node(parent_signal)
        if parent_node is None:
            raise ValueError("Parent signal not found in the tree.")

        transformation_name = method if method is not None else function.__name__
        if node_name is None:
            node_name = transformation_name

        final_name = node_name
        if final_name in parent_node.children:
            count = 1
            while f"{node_name}_{count}" in parent_node.children:
                count += 1
            final_name = f"{node_name}_{count}"

        parent_node.children[final_name] = SignalNode(
            signal=new_signal,
            name=final_name,
            parent=parent_node,
            transformation=transformation_name,
            args=args,
            kwargs=kwargs,
            local=local,
        )
        self.update_plot_states(new_signal)
        return new_signal

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release tree-held resources. Plot windows are torn down by Session."""
        # Re-entrancy guard: closing the strain controller below closes its
        # reference window, whose last-window teardown can re-enter close().
        if getattr(self, "_spyde_closed", False):
            return
        self._spyde_closed = True
        # STOP in-flight compute FIRST: flip every registered stopped_flag and
        # cancel every registered future so heavy actions (Find Vectors, dense
        # OM, VOM, strain) and the distributed navigator/VI fills stop working
        # on the cluster instead of running to completion after the tree closes.
        self._cancel_all_compute()
        if hasattr(self, "_nav_stop"):
            self._nav_stop.set()
        if hasattr(self, "_nav_defer_stop"):
            self._nav_defer_stop.set()
        # Release the progressive-navigator shared-memory segment (created in
        # _start_progressive_nav_compute via ensure_live_buffer). Without this it
        # leaks for the lifetime of the process on every tree close.
        nav_shm = getattr(self, "_nav_shm", None)
        if nav_shm is not None:
            try:
                nav_shm.close()
            except Exception as e:
                logger.debug("closing navigator shm on close failed: %s", e)
            try:
                nav_shm.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.debug("unlinking navigator shm on close failed: %s", e)
            self._nav_shm = None
        # Interactive action state living on the tree: controllers and overlays
        # own windows / navigator hooks — give them a real teardown; results,
        # caches and back-references just drop so nothing leaks past the tree.
        ctrl = getattr(self, "_strain_controller", None)
        if ctrl is not None:
            try:
                ctrl.remove()
            except Exception as e:
                logger.debug("removing strain controller on tree close failed: %s", e)
            self._strain_controller = None
        # `_result_vector_overlays` is a LIST (one per signal plot on a vectors
        # result window — "Add Selector" opens more than one); the rest are single
        # overlays. Both shapes tear down here so none outlives the tree.
        for attr in ("_fv_preview", "_vector_overlay", "_result_vector_overlay",
                     "_result_vector_overlays",
                     "_orientation_overlay", "_particle_overlay"):
            ov = getattr(self, attr, None)
            for one in (ov if isinstance(ov, (list, tuple)) else [ov]):
                if one is not None and hasattr(one, "remove"):
                    try:
                        one.remove()
                    except Exception as e:
                        logger.debug("removing %s on tree close failed: %s", attr, e)
            if hasattr(self, attr):
                setattr(self, attr, None)
        for wiz_attr in ("_om_wizard", "_vom_wizard", "_ebsd_wizard",
                         "_seg_wizard", "_drift_wizard"):
            wiz = getattr(self, wiz_attr, None)
            if wiz is not None and hasattr(wiz, "remove"):
                try:
                    wiz.remove()
                except Exception as e:
                    logger.debug("removing %s on tree close failed: %s", wiz_attr, e)
            if hasattr(self, wiz_attr):
                setattr(self, wiz_attr, None)
        # On-plot interactive widgets (the Crop box, the CZB search box and
        # Manual crosshair: widgets have no remove(), only hide()) and the CZB
        # found-centre marker groups (real remove()). All harmless when absent.
        for attr in ("_crop_widget", "_czb_region_mg", "_czb_cross"):
            wdg = getattr(self, attr, None)
            if wdg is not None and hasattr(wdg, "hide"):
                try:
                    wdg.hide()
                except Exception as e:
                    logger.debug("hiding %s on tree close failed: %s", attr, e)
            if hasattr(self, attr):
                setattr(self, attr, None)
        for mg in (getattr(self, "_czb_found_mgs", None) or []):
            if hasattr(mg, "remove"):
                try:
                    mg.remove()
                except Exception as e:
                    logger.debug("removing czb found marker on tree close failed: %s", e)
        for attr in ("_czb_found_mgs", "_crop_widget_handler", "_czb_region_handler"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        self.signal_plots = []
        self.navigator_signals = {}
        self.navigator_plot_manager = None
        # The IPF EXPLORER window (window 2) is a BARE figure window — it has no
        # Plot, so it is not in the window_ids _close_tree collected and would be
        # left orphaned on screen. Route it through the real teardown
        # (controller close + figure eviction + window_closed to the renderer).
        ipf_win = getattr(self, "_ipf_window", None)
        if ipf_win is not None:
            try:
                self.session._forget_window(getattr(ipf_win, "window_id", None))
            except Exception as e:
                logger.debug("closing the IPF explorer window on tree close "
                             "failed: %s", e)
                try:
                    ipf_win.close()
                except Exception as e2:
                    logger.debug("IPF window fallback close failed: %s", e2)
        # `source_node` / `source_tree` are the reason this list matters as much
        # as the teardown above: a particle tree holds a back-reference to the
        # movie it was segmented FROM, so leaving them set keeps the source
        # signal — a lazy multi-GB array — alive for as long as the result tree
        # is referenced anywhere. Closing the source would then free nothing.
        for attr in ("diffraction_vectors", "orientation_map", "vector_orientation",
                     "_vom_field", "_ipf_result", "_ipf_p3d", "_ipf_picker",
                     "_ipf_window", "_ipf_pick_fn", "_render_frame_fn",
                     "particles", "_seg_pending_particles", "particle_events",
                     "particle_edits", "nav_traces", "drift",
                     "source_node", "source_tree", "nav_map"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception as e:
                    logger.debug("clearing tree attr %r on close failed: %s", attr, e)
