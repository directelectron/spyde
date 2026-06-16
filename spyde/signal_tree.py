from __future__ import annotations

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
        self.client = distributed_client
        self._selector_type = selector_type
        self._pending_nav_dask: da.Array | None = None

        if navigator_override is not None:
            print("Using navigator override for root signal:", navigator_override)
            navigator = self._preprocess_navigator(navigator_override)
        else:
            print("Initializing navigator for root signal:", root_signal)
            navigator = self._initialize_navigator(root_signal)
        print("Navigator initialized:", navigator)
        self.navigator_signals["base"] = navigator

        self.signal_plots: list[Plot] = []
        self.navigator_plot_manager: "MultiplotManager | None" = None

        self._initialize_initial_plots()
        print("Created Signal Tree with root signal:", self.root)

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
        if nav_dask is None or self.client is None:
            return

        # Cancel the single monolithic future submitted by _preprocess_navigator
        nav_signals = self.navigator_signals.get("base")
        if nav_signals:
            old_future = nav_signals[0].data
            from dask.distributed import Future as _Future
            if isinstance(old_future, _Future):
                try:
                    self.client.cancel(old_future)
                except Exception:
                    pass

        nav_plot_windows = list(self.navigator_plot_manager.plot_windows.keys())
        if not nav_plot_windows:
            return
        nav_pw = nav_plot_windows[0]
        nav_plots = self.navigator_plot_manager.plots.get(nav_pw, [])
        if not nav_plots:
            return
        nav_plot = nav_plots[0]

        nav_shape = tuple(nav_dask.shape)
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
            except Exception:
                pass

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
                except Exception:
                    pass
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
                nav_dask = navigator.data
                self._pending_nav_dask = nav_dask
                navigator.data = self.client.compute(
                    nav_dask, priority=-10, workers=heavy_workers
                )
            return [navigator, signal]

        elif signal.axes_manager.signal_dimension > 2:
            signal = signal.transpose(2)
            navigator = signal.sum(signal.axes_manager.signal_axes).T
            if navigator._lazy:
                nav_dask = navigator.data
                self._pending_nav_dask = nav_dask
                navigator.data = self.client.compute(
                    nav_dask, priority=-10, workers=heavy_workers
                )
            return [navigator, signal]

        if signal._lazy:
            nav_dask = signal.data
            self._pending_nav_dask = nav_dask
            signal.data = self.client.compute(
                nav_dask, priority=-10, workers=heavy_workers
            )
        return [signal]

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
                except Exception:
                    pass
