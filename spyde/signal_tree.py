from functools import partial

import numpy as np
import dask.array as da
from hyperspy.signal import BaseSignal

from typing import TYPE_CHECKING, Union, List, Iterator

from spyde.signal_node import SignalNode

from spyde.drawing.plots.plot_window import PlotWindow

if TYPE_CHECKING:
    from spyde.drawing.plots.plot_states import PlotState
    from spyde.__main__ import MainWindow

from spyde.drawing.plots.plot import Plot
from spyde.drawing.plots.multiplot_manager import MultiplotManager


class BaseSignalTree:
    """
    A class to manage the signal tree. This class manages the tree of different signals
    after some transformation has been applied.  The idea is that you can toggle between
    the different signals to see the effects of transformations such as filtering, centering
    the direct beam, azimuthal integration, etc.

    For example, you might have a tree like this:
                                              -/-> [FEM Variance]
                                            /
               --> [denoise filter] --> [centered] --> [azimuthal integration]
             /
    [root signal]
             \
              --> [centered] --> [get_diffraction_vectors] --> [strain matrix] -/-> [strain maps]
                        \
                         -/-> [get_virtual_image]

    -----------------------------------------------------------------------------------------------

                                           -/-> [Bright-field](toggle visible/not visible)
                                         /
       [navigator] --> [signal] --> [Centered]
                                         \
                                          -/-> [Dark-field] (toggle visible/not visible)

    -----------------------------------------------------------------------------------------------
    -----------------------------------------------------------------------------------------------
    Then you can select the different steps in the tree to see the data computed at that point.

    The idea is that a lot of the `map` like transformations are non-breaking.  These are transformations
    where the navigator is still valid. For example.

    In contrast, a non-breaking function will
    just update the current "signal" plot with the new data. Toggling back to the previous fork in the tree will
    allow you to see the data along the way.


    Each

    Parameters
    ----------
    root_signal : BaseSignal
        The root signal of the tree.
    main_window : MainWindow
        The main window of the application.
    distributed_client : distributed.Client, optional
        The Dask client to use for computations.

    """

    def __init__(
        self,
        root_signal: BaseSignal,
        main_window: "MainWindow",
        distributed_client=None,
        selector_type=None,
        navigator_override: BaseSignal = None,
    ):

        # The root signal of the tree
        self.root = root_signal  # type: BaseSignal
        self.main_window = main_window  # type: MainWindow

        # There is only one navigator.  Currently, having more than one navigator makes things very complicated.
        # I want to try to minimize the number of plots.  If for example someone has a 5D STEM signal, and they want
        # to play with the time axis then they should use the virtual imaging tools and do it that way.
        self.navigator_signals = dict()  # type: dict[str:BaseSignal] # only 1

        # The tree structure. This defines the relationship between signals.
        # i.e. parent -> child.  Broken transformations create new seeds and
        # spawn new trees.
        self.root_node = SignalNode(signal=root_signal, name="root", parent=None)
        self.client = distributed_client
        self._selector_type = selector_type
        # Lazy dask array for the nav signal, saved before submitting as a
        # single future so _start_progressive_nav_compute can use it.
        self._pending_nav_dask = None  # type: da.Array | None

        # set up the navigator plots:
        if navigator_override is not None:
            # Caller supplies a ready-made navigator (e.g. a vector count map)
            # — skip the sum-over-signal-axes compute of the full dataset.
            print("Using navigator override for root signal: ", navigator_override)
            navigator = self._preprocess_navigator(navigator_override)
        else:
            print("Initializing navigator for root signal: ", root_signal)
            navigator = self._initialize_navigator(root_signal)
        print("Navigator initialized: ", navigator)
        self.navigator_signals["base"] = navigator

        self.signal_plots = []  # type: Union[List[Plot], None]

        self.navigator_plot_manager = None  # type: Union[MultiplotManager, None]
        # setting up plots

        self._initialize_initial_plots()
        print("Created Signal Tree with root signal: ", self.root)

    def _initialize_initial_plots(self):
        """
        Initialize the initial plots based on the root signal.
        """
        if (
            self.root.axes_manager.navigation_dimension > 0
        ):  # pass to NavigationPlotManager
            self.navigator_plot_manager = MultiplotManager(
                main_window=self.main_window, signal_tree=self,
                selector_type=self._selector_type,
            )
            if self._pending_nav_dask is not None:
                self._start_progressive_nav_compute()
        else:
            self.navigator_plot_manager = None
            self.add_signal_plot()

    def add_signal_plot(self):
        """
        Add a new signal plot for the root signal.
        """
        pw = self.main_window.add_plot_window(
            is_navigator=False, signal_tree=self, plot_manager=None
        )

        plot = pw.add_new_plot()
        self.create_plot_states(plot=plot)
        plot.set_plot_state(list(plot.plot_states.keys())[0])
        self.signal_plots.append(plot)

        signal = self.root
        if signal._lazy and self.client is not None:
            # Submit as a distributed future so PlotUpdateWorker can track it
            # and the image fills in progressively rather than blocking.
            future = self.client.compute(signal.data)
            plot.update_data(future)
        else:
            plot.update()

    def _start_progressive_nav_compute(self):
        """
        Replace the single-future nav compute with a per-chunk progressive
        compute that live-updates the navigator image as chunks finish.

        Mirrors the virtual-imaging compute_with_live_buffer pattern exactly.
        Called after MultiplotManager (and its nav plots) are fully constructed.
        """
        from PySide6 import QtCore as _QtCore
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
        # so it doesn't waste resources duplicating what compute_with_live_buffer does.
        nav_signals = self.navigator_signals.get("base")
        if nav_signals:
            old_future = nav_signals[0].data
            from dask.distributed import Future as _Future
            if isinstance(old_future, _Future):
                try:
                    self.client.cancel(old_future)
                except Exception:
                    pass

        # Find the top-level navigator plot window and its first plot
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

        # Pre-fill shared memory with NaN so the plot shows empty immediately
        shm = ensure_live_buffer(nav_shape, shm_name)
        # keep shm alive on the signal tree
        self._nav_shm = shm

        # Show NaN-filled image now (correct extent, no data yet)
        nav_plot.current_data = np.full(nav_shape, np.nan, dtype=np.float32)
        nav_plot.needs_auto_level = True
        nav_plot.update()

        # Chunk relay: Dask callback thread → GUI thread → shared memory
        from PySide6 import QtCore as _QC

        class _NavChunkRelay(_QC.QObject):
            chunk_ready = _QC.Signal(object, object)

        relay = _NavChunkRelay(self.main_window)
        self._nav_relay = relay  # keep alive

        def _gui_write_chunk(chunk_result, nav_slices, _shm=shm, _shape=nav_shape):
            try:
                buf = np.ndarray(_shape, dtype=np.float32, buffer=_shm.buf)
                buf[nav_slices] = chunk_result.astype(np.float32)
            except Exception:
                pass

        relay.chunk_ready.connect(_gui_write_chunk)

        def _on_chunk(chunk_result, nav_slices):
            relay.chunk_ready.emit(chunk_result, nav_slices)

        future = compute_with_live_buffer(
            nav_dask, nav_shape, self.client, shm_name, on_chunk_done=_on_chunk
        )

        # Overwrite the single future that _preprocess_navigator stored so
        # PlotUpdateWorker still fires on_plot_future_ready when done.
        nav_signals = self.navigator_signals.get("base")
        if nav_signals:
            nav_signals[0].data = future
        nav_plot.current_data = future
        nav_plot.needs_auto_level = True

        # Poll shared buffer every 100 ms to show intermediate progress
        poll_timer = _QtCore.QTimer(self.main_window)
        poll_timer.setInterval(100)
        self._nav_poll_timer = poll_timer  # keep alive

        _nav_levels = [None]

        def _poll():
            if future.done():
                poll_timer.stop()
                return
            arr = read_live_buffer(nav_shape, shm_name)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return
            if _nav_levels[0] is None:
                lo, hi = float(finite.min()), float(finite.max())
                _nav_levels[0] = (lo, hi if hi > lo else lo + 1)
            else:
                lo, hi = float(finite.min()), float(finite.max())
                if hi > _nav_levels[0][1]:
                    _nav_levels[0] = (_nav_levels[0][0], hi)
            nav_plot.image_item.setImage(arr, autoLevels=False, levels=_nav_levels[0])
            nav_plot.update_range()

        poll_timer.timeout.connect(_poll)
        poll_timer.start()

    @property
    def plot_windows(self) -> List["PlotWindow"]:
        """Return all plot windows in the signal tree."""
        if self.navigator_plot_manager is None:
            return []
        return self.navigator_plot_manager.plot_windows



    def _preprocess_navigator(self, signal: BaseSignal) -> List[BaseSignal]:
        """
        Preprocess the navigator signal.

        The navigator is a small 2D image (sum over signal axes) — that gets
        computed and held in memory.  The raw signal data STAYS LAZY; we never
        pull hundreds of GB of 4D STEM data into RAM.

        Returns a list of [nav_signal] or [nav_signal, signal] depending on
        whether the input already has separate nav/signal dims.
        """
        if (
            signal.axes_manager.navigation_shape + signal.axes_manager.signal_shape
        ) != self.root.axes_manager.navigation_shape:
            raise ValueError(
                "Navigator signal must have the same total number of dimensions as the root signal "
                "and the same shape."
            )

        if signal.axes_manager.signal_dimension == 0:
            signal = signal.T

        if (
            signal.axes_manager.signal_dimension > 0
            and signal.axes_manager.navigation_dimension > 0
        ):
            # Navigator: sum over signal axes → small 2D image, compute it.
            # Signal: leave lazy — do NOT call client.compute on hundreds of GB.
            navigator = signal.sum(signal.axes_manager.signal_axes).T
            if navigator._lazy:
                nav_dask = navigator.data
                self._pending_nav_dask = nav_dask
                navigator.data = self.client.compute(
                    nav_dask, priority=-10,
                    workers=self.main_window.dask_manager.heavy_workers,
                )
            print("Preprocessing navigator: ", navigator, signal)
            return [navigator, signal]

        elif signal.axes_manager.signal_dimension > 2:
            signal = signal.transpose(2)
            navigator = signal.sum(signal.axes_manager.signal_axes).T
            if navigator._lazy:
                nav_dask = navigator.data
                self._pending_nav_dask = nav_dask
                navigator.data = self.client.compute(
                    nav_dask, priority=-10,
                    workers=self.main_window.dask_manager.heavy_workers,
                )
            print("Preprocessing navigator: ", navigator, signal)
            return [navigator, signal]

        # signal_dimension == 0 after .T above: signal IS the navigator.
        # It's already small (2D nav image), compute it.
        if signal._lazy:
            nav_dask = signal.data
            self._pending_nav_dask = nav_dask
            signal.data = self.client.compute(
                nav_dask, priority=-10,
                workers=self.main_window.dask_manager.heavy_workers,
            )
        return [signal]

    def add_navigator_signal(self, name: str, signal: BaseSignal):
        """
        This adds a navigator plot to the signal tree.  The idea being that a signal tree can have multiple
        navigator signals but only 1 navigator plot (or multi-plot).  Eventually it would be nice to add the
        ability to multi-plex navigator plots. For example temp and time in an in-situ experiment.

        Parameters
        ----------
        name : str
            The name of the navigator signal.
        signal : BaseSignal
            The navigator signal to add.
        """
        signal = self._preprocess_navigator(signal)
        self.navigator_signals[name] = signal
        self.navigator_plot_manager.add_plot_states_for_navigation_signals(signal)

    def _initialize_navigator(self, signal: BaseSignal):
        """
        Populate the navigator plots based on the root signal.

        Recursively create navigator plots to account for dimensions greater than 2. Eventually this should
        support things like EELS line spectra.

        Parameters
        ----------
        nav_signal : BaseSignal
            The signal to populate the navigator plots for.
        """
        if signal.axes_manager.navigation_dimension == 0:
            # single image or spectrum — no navigator needed
            return
        else:
            # Sum over signal axes to get the navigator image.
            # For lazy signals this produces a lazy dask array; _preprocess_navigator
            # will submit it to the distributed client (small 2D result, not the raw data).
            if signal._lazy and signal.navigator is not None:
                navigation_signal = signal.navigator
                # navigator may itself be lazy — leave it; _preprocess_navigator handles it
            else:
                navigation_signal = signal.sum(signal.axes_manager.signal_axes)
            if not isinstance(navigation_signal, BaseSignal):
                navigation_signal = BaseSignal(navigation_signal)

        # handle lazy computation and setting up the axes properly...
        navigation_signal = self._preprocess_navigator(navigation_signal)
        return navigation_signal

    def walk(self) -> Iterator[SignalNode]:
        """Depth-first generator over all nodes."""
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def signals(self) -> List[BaseSignal]:
        """Return a list of all signals in the tree, including the root."""
        return [node.signal for node in self.walk()]

    def create_plot_states(self, plot: "Plot" = None) -> dict:
        """
        Create plot states for each signal plot in the tree.
        """

        plot_states = {}
        for signal in self.signals():
            if signal.axes_manager.navigation_dimension == 0:
                plot.add_plot_state(
                    signal=signal,
                    dynamic=False,
                    dimensions=signal.axes_manager.signal_dimension,
                )
            else:
                plot.add_plot_state(
                    signal=signal,
                    dynamic=True,
                    dimensions=signal.axes_manager.signal_dimension,
                )
        return plot_states

    def update_plot_states(self, new_signal: BaseSignal):
        """
        Update all plot states in the signal tree.
        """
        from spyde.drawing.plots.plot_states import PlotState

        for plot in self.signal_plots:
            if new_signal not in plot.plot_states:
                if new_signal.axes_manager.navigation_dimension == 0:
                    plot.plot_states[new_signal] = PlotState(
                        signal=new_signal, plot=plot, dynamic=False
                    )
                else:
                    plot.plot_states[new_signal] = PlotState(
                        signal=new_signal, plot=plot, dynamic=True
                    )

    @property
    def nav_dim(self) -> int:
        """
        The number of navigation dimensions in the root signal.
        """
        return self.root.axes_manager.navigation_dimension

    def create_navigator_plots(self):
        """
        Create navigator plots based on the root signal.
        """
        self.navigator_plot_manager = MultiplotManager(
            main_window=self.main_window, signal_tree=self
        )

    def get_nested_attr(self, attr_path: str):
        """
        Get a nested attribute from `self` following a dot-separated path.

        Parameters
        ----------
        attr_path : str
            Dot-separated path of attributes (e.g., "root.axes_manager.navigation_shape").

        Returns
        -------
        Any
            The resolved attribute value or `None` if any segment is missing or `None`.
        """
        if not attr_path:
            return self
        attrs = [p for p in attr_path.split(".") if p]
        current_obj = self
        for attr in attrs:
            current_obj = getattr(current_obj, attr, None)
            if current_obj is None:
                return None
        return current_obj

    def get_node(self, signal) -> SignalNode | None:
        """Get the node in the tree corresponding to the given signal."""
        for node in self.walk():
            if node.signal is signal:
                return node
        return None

    def add_node(self, parent_signal, new_signal, transformation: str):
        """Add a new node to the signal tree."""
        parent_node = self.get_node(parent_signal)
        if parent_node is None:
            raise ValueError("Parent node not found in the tree.")
        # Handle name collision
        final_name = transformation
        if final_name in parent_node.children:
            count = 1
            candidate = f"{transformation}_{count}"
            while candidate in parent_node.children:
                count += 1
                candidate = f"{transformation}_{count}"
            final_name = candidate
        child = SignalNode(
            signal=new_signal,
            name=final_name,
            parent=parent_node,
            transformation=transformation,
        )
        parent_node.children[final_name] = child

    def add_transformation(
        self,
        parent_signal,
        method: str = None,
        function: callable = None,
        node_name: str = None,
        *args,
        **kwargs,
    ):
        """
        Add a transformation to the tree.

        Parameters
        ----------
        parent_signal : Signal
            The parent signal to which the transformation is applied.
        method : str, optional
            The method of the signal to call as a method.
        function : callable, optional
            The function to apply to the parent signal.
        node_name : str, optional
            The name of the new node in the tree.
        *args
            Positional arguments to pass to the function or method.
        **kwargs
            Keyword arguments to pass to the function or method.
        """
        if method is not None:
            try:
                new_signal = getattr(parent_signal, method)(*args, **kwargs)
            except Exception as e:
                from PySide6 import QtWidgets
                QtWidgets.QMessageBox.critical(
                    self.main_window,
                    "Transformation error",
                    f"An error occurred while applying transformation "
                    f"'{method or (function.__name__ if function else '')}':\n{e}",
                )
                return
        else:
            new_signal = function(parent_signal, *args, **kwargs)

        parent_node = self.get_node(parent_signal)
        if parent_node is None:
            raise ValueError("Parent signal not found in the tree.")

        transformation_name = method if method is not None else function.__name__
        if node_name is None:
            node_name = transformation_name

        # Handle name collision
        final_name = node_name
        if final_name in parent_node.children:
            count = 1
            candidate = f"{node_name}_{count}"
            while candidate in parent_node.children:
                count += 1
                candidate = f"{node_name}_{count}"
            final_name = candidate

        child = SignalNode(
            signal=new_signal,
            name=final_name,
            parent=parent_node,
            transformation=transformation_name,
            args=args,
            kwargs=kwargs,
        )
        parent_node.children[final_name] = child
        print(f"Added transformation '{final_name}' to the tree under parent signal.")
        self.update_plot_states(new_signal)
        return new_signal

    def close(self):
        """Release references held by the tree. The plot WINDOWS (and their
        toolbars/selectors) are torn down by MDIManager.close_signal_tree, which
        is the authority; this just drops the tree's own bookkeeping so nothing
        lingers. Safe to call once the windows are closed."""
        self.signal_plots = []
        self.navigator_signals = {}
        self.navigator_plot_manager = None
        # drop the attached results so they can be GC'd
        for attr in ("diffraction_vectors", "orientation_map",
                     "vector_orientation"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
