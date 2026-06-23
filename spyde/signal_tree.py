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
    ):
        self.root = root_signal
        self.session = session

        self.navigator_signals: dict[str, BaseSignal] = {}
        self.root_node = SignalNode(signal=root_signal, name="root", parent=None)
        self._client_override = distributed_client
        self._selector_type = selector_type
        self._pending_nav_dask: da.Array | None = None

        if navigator_override is not None:
            navigator = self._preprocess_navigator(navigator_override)
        else:
            navigator = self._initialize_navigator(root_signal)
        self.navigator_signals["base"] = navigator

        self.signal_plots: list[Plot] = []
        self.navigator_plot_manager: "MultiplotManager | None" = None

        self._initialize_initial_plots()
        logger.debug("Created signal tree with root %s", self.root)

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
                self._start_progressive_nav_compute()
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
            future = self.client.compute(signal.data)
            plot.update_data(future)
        else:
            plot.update()

    # ── Progressive navigator compute ─────────────────────────────────────────

    def _start_progressive_nav_compute(self) -> None:
        """
        Replace the single-future nav compute with a per-chunk progressive
        compute that live-updates the navigator image as chunks finish.
        """
        from spyde.drawing.update_functions import (
            compute_with_live_buffer,
            ensure_live_buffer,
            read_live_buffer,
        )

        nav_dask = self._pending_nav_dask
        self._pending_nav_dask = None
        if nav_dask is None:
            return

        nav_signals = self.navigator_signals.get("base")
        nav_plot_windows = list(self.navigator_plot_manager.plot_windows.keys())
        if not nav_plot_windows:
            return
        nav_pw = nav_plot_windows[0]
        nav_plots = self.navigator_plot_manager.plots.get(nav_pw, [])
        if not nav_plots:
            return
        nav_plot = nav_plots[0]
        nav_shape = tuple(nav_dask.shape)


        logger.debug(
            "NAV-DEBUG _start_progressive_nav_compute: path=%s nav_shape=%s "
            "chunks=%s client=%s",
            "THREADED (no client)" if self.client is None else "DISTRIBUTED",
            nav_shape, nav_dask.chunks, type(self.client).__name__,
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
        if self.client is None:
            import itertools

            placeholder = np.full(nav_shape, np.nan, dtype=np.float32)
            nav_plot.current_data = placeholder
            # Stop flag so the thread bails out cleanly on tree/session shutdown
            # instead of painting onto a torn-down plot.
            stop = threading.Event()
            self._nav_stop = stop

            def _bg_nav(_dask=nav_dask, _plot=nav_plot, _sig=nav_signals,
                        _shape=nav_shape, _stop=stop):
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
                    for combo in itertools.product(*axes_ranges):
                        if _stop.is_set():
                            return
                        nav_slices = tuple(slice(s, s + n) for s, n in combo)
                        logger.debug("NAV-DEBUG threaded nav chunk %s computing", nav_slices)
                        block = np.asarray(_dask[nav_slices].compute()).astype(np.float32)
                        acc[nav_slices] = block
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
                        if _stop.is_set():
                            return
                        _plot.set_data(acc.copy(), levels=levels[0])
                    # Final uniform repaint: now that every chunk is in, set the
                    # definitive levels over the whole image so no transient
                    # per-chunk stretch remains visible at a boundary, and emit a
                    # histogram so the navigator gets a Plot-Control histogram /
                    # contrast handles (set_data with explicit levels otherwise
                    # skips _emit_histogram, leaving the navigator histogram-less).
                    if not _stop.is_set():
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
                        _sig[0].data = acc
                except Exception:
                    # Primary (threaded) navigator load — a failure here leaves a
                    # blank navigator, so surface the traceback rather than hide it.
                    logger.exception("threaded navigator compute failed")
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

        future = compute_with_live_buffer(
            nav_dask, nav_shape, self.client, shm_name, on_chunk_done=_on_chunk
        )

        if nav_signals:
            nav_signals[0].data = future
        nav_plot.current_data = future
        nav_plot.needs_auto_level = True

        # Periodic poll: update the displayed image as chunks arrive
        _nav_levels: list = [None]
        _stop = threading.Event()
        self._nav_stop = _stop

        def _poll_loop():
            while not _stop.is_set():
                if future.done():
                    break
                try:
                    arr = read_live_buffer(nav_shape, shm_name)
                    finite = arr[np.isfinite(arr)]
                    if finite.size > 0:
                        if _nav_levels[0] is None:
                            lo, hi = float(finite.min()), float(finite.max())
                            _nav_levels[0] = (lo, hi if hi > lo else lo + 1)
                        else:
                            lo, hi = float(finite.min()), float(finite.max())
                            if hi > _nav_levels[0][1]:
                                _nav_levels[0] = (_nav_levels[0][0], hi)
                        nav_plot.set_data(arr, levels=_nav_levels[0])
                except Exception as e:
                    logger.debug("navigator poll paint failed: %s", e)
                time.sleep(0.1)

        t = threading.Thread(target=_poll_loop, daemon=True, name="nav-poll")
        t.start()
        self._nav_poll_thread = t

    # ── Navigator processing ───────────────────────────────────────────────────

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
            signal = signal.transpose(2)
            navigator = signal.sum(signal.axes_manager.signal_axes).T
            if navigator._lazy:
                navigator.data = self._compute_navigator(navigator.data, heavy_workers)
            return [navigator, signal]

        if signal._lazy:
            signal.data = self._compute_navigator(signal.data, heavy_workers)
        return [signal]

    def _compute_navigator(self, nav_dask: da.Array, heavy_workers):
        """Kick off computing the (small) navigator image — NEVER blocking.

        The display must not wait on the navigator compute (tree ``__init__``
        runs on the load thread and emits the windows right after this). So we
        always stash ``nav_dask`` and return immediately:

        * with a distributed client → submit a Future (progressive per-chunk
          compute replaces it for the live navigator-on-load);
        * without a client (cluster still starting, or headless) → return a NaN
          placeholder; ``_start_progressive_nav_compute`` computes it on a
          BACKGROUND thread (threaded scheduler) so init isn't blocked.
        """
        self._pending_nav_dask = nav_dask
        if self.client is not None:
            logger.debug(
                "NAV-DEBUG _compute_navigator: DISTRIBUTED path (client=%s) "
                "nav_dask.shape=%s chunks=%s heavy_workers=%s",
                getattr(self.client, "scheduler", None) and self.client.scheduler.address,
                tuple(nav_dask.shape), nav_dask.chunks, heavy_workers,
            )
            return self.client.compute(nav_dask, priority=-10, workers=heavy_workers)
        logger.debug(
            "NAV-DEBUG _compute_navigator: NO CLIENT -> threaded background path "
            "will run; nav_dask.shape=%s chunks=%s", tuple(nav_dask.shape), nav_dask.chunks,
        )
        return np.full(tuple(nav_dask.shape), np.nan, dtype=np.float32)

    def add_navigator_signal(self, name: str, signal: BaseSignal) -> None:
        signal = self._preprocess_navigator(signal)
        self.navigator_signals[name] = signal
        self.navigator_plot_manager.add_plot_states_for_navigation_signals(signal)

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

    def add_node(self, parent_signal, new_signal, transformation: str) -> None:
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
        )

    def add_transformation(
        self,
        parent_signal,
        method: str = None,
        function: callable = None,
        node_name: str = None,
        *args,
        **kwargs,
    ) -> BaseSignal | None:
        from spyde.backend.ipc import emit_error

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
        )
        self.update_plot_states(new_signal)
        return new_signal

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release tree-held resources. Plot windows are torn down by Session."""
        if hasattr(self, "_nav_stop"):
            self._nav_stop.set()
        self.signal_plots = []
        self.navigator_signals = {}
        self.navigator_plot_manager = None
        for attr in ("diffraction_vectors", "orientation_map", "vector_orientation"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception as e:
                    logger.debug("clearing tree attr %r on close failed: %s", attr, e)
