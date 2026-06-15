from __future__ import annotations
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QObject, Slot, Qt

from spyde.drawing.colormaps import COLORMAPS
from spyde.drawing.signal_tree_presenter import (
    build_axes_groups, build_metadata_groups)
from spyde.external.pyqtgraph.histogram_widget import HistogramLUTWidget, HistogramLUTItem
from spyde.live.camera_control_widget import CameraControlWidget
from spyde.live.control_dock_widget import ControlDockWidget
from spyde.live.particle_scanning import ParticleScanControlWidget

if TYPE_CHECKING:
    from spyde.__main__ import MainWindow
    from spyde.drawing.plots.plot import Plot
    from spyde.drawing.plots.plot_window import PlotWindow


class DockManager(QObject):
    """Owns Plot Control and Instrument Control dock construction and updates."""

    def __init__(self, main_window: "MainWindow", parent=None):
        super().__init__(parent)
        self.main_window = main_window
        # _histogram_image_item is stored on main_window so tests can reset it via win._histogram_image_item = None

        self.dock_widget: QtWidgets.QDockWidget | None = None
        self.control_widget: ControlDockWidget | None = None
        self.histogram: HistogramLUTWidget | None = None
        self.cmap_selector: QtWidgets.QComboBox | None = None
        self.metadata_layout: QtWidgets.QHBoxLayout | None = None
        self.axes_layout: QtWidgets.QVBoxLayout | None = None
        self.selectors_layout: QtWidgets.QVBoxLayout | None = None
        self.btn_auto: QtWidgets.QPushButton | None = None
        self.btn_reset: QtWidgets.QPushButton | None = None

        # Signal Type panel state
        self.signal_class_label: QtWidgets.QLabel | None = None
        self.signal_type_combo: QtWidgets.QComboBox | None = None
        self.btn_set_signal_type: QtWidgets.QPushButton | None = None
        self._signal_type_plot: "Plot" | None = None

        self._build_plot_control_dock()
        self._build_instrument_control_dock()

    def _build_plot_control_dock(self) -> None:
        mw = self.main_window
        self.dock_widget = QtWidgets.QDockWidget("Plot Control", mw)
        self.dock_widget.setObjectName("plotControlDock")
        self.dock_widget.setFeatures(
            self.dock_widget.features()
            & ~QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.dock_widget.setBaseSize(mw.width() // 6, mw.height() // 6)

        main_widget = QtWidgets.QWidget()
        main_widget.setAutoFillBackground(True)
        main_widget.setStyleSheet("background-color: #141414;")
        layout = QtWidgets.QVBoxLayout(main_widget)

        display_group = QtWidgets.QGroupBox("Plot Display Controls")
        display_group.setMaximumHeight(250)
        display_layout = QtWidgets.QVBoxLayout(display_group)

        self.histogram = HistogramLUTWidget(
            orientation="horizontal", autoLevel=False, constantLevel=True,
            show_gradient=False,
        )
        self.histogram.setMinimumWidth(200)
        self.histogram.setMinimumHeight(100)
        self.histogram.setMaximumHeight(150)
        self.histogram.item.sigLevelChangeFinished.connect(self._on_histogram_levels_finished)
        self.histogram.item.sigGammaChanged.connect(self._on_histogram_gamma_changed)
        display_layout.addWidget(self.histogram)

        self.cmap_selector = QtWidgets.QComboBox()
        self.cmap_selector.addItems(list(COLORMAPS.keys()))
        self.cmap_selector.setCurrentText("gray")
        self.cmap_selector.currentTextChanged.connect(self._on_cmap_changed)
        cmap_layout = QtWidgets.QHBoxLayout()
        cmap_layout.addWidget(QtWidgets.QLabel("Colormap"))
        cmap_layout.addWidget(self.cmap_selector, 1)
        display_layout.addLayout(cmap_layout)
        layout.addWidget(display_group)

        buttons_layout = QtWidgets.QHBoxLayout()
        from spyde.qt.style import make_button
        self.btn_auto = make_button("Auto")
        self.btn_reset = make_button("Reset")
        self.btn_auto.clicked.connect(self._on_contrast_auto)
        self.btn_reset.clicked.connect(self._on_contrast_reset)
        buttons_layout.addWidget(self.btn_auto)
        buttons_layout.addWidget(self.btn_reset)
        display_layout.addLayout(buttons_layout)

        # Signal Type: shows the clicked signal's hyperspy class and lets the
        # user retype it (set_signal_type), which re-gates the toolbar actions
        # (e.g. Virtual Imaging / Orientation Mapping need Diffraction2D).
        signal_group = QtWidgets.QGroupBox("Signal Type")
        signal_layout = QtWidgets.QVBoxLayout(signal_group)
        self.signal_class_label = QtWidgets.QLabel("(no signal selected)")
        signal_layout.addWidget(self.signal_class_label)
        type_row = QtWidgets.QHBoxLayout()
        self.signal_type_combo = QtWidgets.QComboBox()
        self.btn_set_signal_type = make_button("Set Type")
        self.btn_set_signal_type.clicked.connect(self._on_set_signal_type)
        type_row.addWidget(self.signal_type_combo, 1)
        type_row.addWidget(self.btn_set_signal_type)
        signal_layout.addLayout(type_row)
        signal_group.setEnabled(False)
        self._signal_type_group = signal_group
        layout.addWidget(signal_group)

        metadata_group = QtWidgets.QGroupBox("Metadata")
        self.metadata_layout = QtWidgets.QHBoxLayout(metadata_group)
        layout.addWidget(metadata_group)

        axes_group = QtWidgets.QGroupBox("Plot Axes")
        self.axes_layout = QtWidgets.QVBoxLayout(axes_group)
        layout.addWidget(axes_group)

        selectors_group = QtWidgets.QGroupBox("Selector Controls")
        self.selectors_layout = QtWidgets.QVBoxLayout(selectors_group)
        layout.addWidget(selectors_group)

        self.dock_widget.setWidget(main_widget)
        mw.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock_widget)

    def _build_instrument_control_dock(self) -> None:
        mw = self.main_window
        self.control_widget = ControlDockWidget()
        self.control_widget.setVisible(False)
        mw.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.control_widget)
        self.control_widget.add_widget(CameraControlWidget())
        self.control_widget.add_widget(ParticleScanControlWidget())

    def toggle_plot_control(self) -> None:
        if self.dock_widget is not None:
            self.dock_widget.setVisible(not self.dock_widget.isVisible())

    def toggle_instrument_control(self) -> None:
        if self.control_widget is not None:
            self.control_widget.setVisible(not self.control_widget.isVisible())

    @Slot(object)
    def on_active_plot_changed(self, window: "PlotWindow") -> None:
        """Called when a new PlotWindow is activated."""
        if window is None:
            return
        from spyde.drawing.plots.plot_window import PlotWindow
        if not isinstance(window, PlotWindow):
            return
        plot = window.current_plot_item
        if plot is None:
            return
        plot_state = getattr(plot, "plot_state", None)

        # Histogram binding
        img_item = getattr(plot, "image_item", None)
        if (
            self.histogram is not None
            and img_item is not None
            and img_item is not self.main_window._histogram_image_item
        ):
            try:
                self.histogram.setImageItem(img_item)
                self.main_window._histogram_image_item = img_item
                if plot_state is not None:
                    self.histogram.setLevels(plot_state.min_level, plot_state.max_level)
                    self.histogram.item.set_gamma(
                        getattr(plot_state, "gamma", 1.0)
                    )
                self.histogram.item.autoHistogramRange()
            except Exception:
                pass

        # Colormap selector sync
        if plot_state is not None and self.cmap_selector is not None:
            self.cmap_selector.setCurrentText(plot_state.colormap)

        # Metadata and axes panels
        st = getattr(window, "signal_tree", None)
        if st is not None:
            self._update_metadata_panel(plot)
        self._update_axes_panel(plot)
        self._update_signal_type_panel(plot)

    # ── Signal Type panel ────────────────────────────────────────────────────

    _GENERIC_TYPE_LABEL = "(generic)"

    @staticmethod
    def _available_signal_types(signal) -> list[str]:
        """
        Signal-type strings the given signal can be converted to.

        Sourced from hyperspy's extension registry (hyperspy itself plus any
        installed extensions like pyxem/exspy), filtered to classes with the
        same signal dimension and dtype family as the current signal.
        """
        import numpy as np
        from hyperspy.extensions import ALL_EXTENSIONS

        sig_dim = signal.axes_manager.signal_dimension
        is_complex = np.issubdtype(signal.data.dtype, np.complexfloating)
        wanted_dtype = "complex" if is_complex else "real"

        types: set[str] = set()
        for info in ALL_EXTENSIONS["signals"].values():
            if info.get("lazy"):
                continue  # lazy/eager pairs share a signal_type
            if info.get("signal_dimension") != sig_dim:
                continue
            if info.get("dtype", "real") != wanted_dtype:
                continue
            types.add(info.get("signal_type", ""))
        types.add("")  # plain SignalxD is always available
        return sorted(types)

    def _update_signal_type_panel(self, plot: "Plot") -> None:
        """Refresh the Signal Type group for the clicked/active plot."""
        if self.signal_type_combo is None:
            return
        signal = getattr(getattr(plot, "plot_state", None), "current_signal", None)
        if signal is None:
            self._signal_type_plot = None
            self._signal_type_group.setEnabled(False)
            self.signal_class_label.setText("(no signal selected)")
            self.signal_type_combo.clear()
            return

        self._signal_type_plot = plot
        self._signal_type_group.setEnabled(True)
        self.signal_class_label.setText(f"Class: {type(signal).__name__}")

        current = signal._signal_type or self._GENERIC_TYPE_LABEL
        self.signal_type_combo.blockSignals(True)
        self.signal_type_combo.clear()
        for st in self._available_signal_types(signal):
            self.signal_type_combo.addItem(st or self._GENERIC_TYPE_LABEL)
        self.signal_type_combo.setCurrentText(current)
        self.signal_type_combo.blockSignals(False)

    def _on_set_signal_type(self) -> None:
        """Retype the active plot's signal and rebuild its toolbars so
        signal-class-gated actions appear/disappear accordingly."""
        plot = self._signal_type_plot
        if plot is None or self.signal_type_combo is None:
            return
        plot_state = getattr(plot, "plot_state", None)
        signal = getattr(plot_state, "current_signal", None)
        if signal is None:
            return

        chosen = self.signal_type_combo.currentText()
        signal_type = "" if chosen == self._GENERIC_TYPE_LABEL else chosen
        if signal_type == (signal._signal_type or ""):
            return  # no change

        try:
            # in-place: hyperspy reassigns the signal's class, so the signal
            # tree and every plot referencing it see the new type
            signal.set_signal_type(signal_type)
        except Exception as e:
            self.main_window.statusBar().showMessage(
                f"Could not set signal type {chosen!r}: {e}", 5000
            )
            return

        plot_state.rebuild_toolbars()
        self._update_signal_type_panel(plot)
        self._update_metadata_panel(plot)
        self.main_window.statusBar().showMessage(
            f"Signal type set to {type(signal).__name__}", 5000
        )

    def _update_metadata_panel(self, plot: "Plot") -> None:
        if self.metadata_layout is None:
            return
        while self.metadata_layout.count():
            item = self.metadata_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            else:
                del item
        if not hasattr(plot, "signal_tree") or plot.signal_tree is None:
            return
        # Themed, editable metadata groups (same EditableLabel widget as the
        # Plot Axes panel) — consistent with the rest of the sidebar.
        for group in build_metadata_groups(plot.signal_tree):
            self.metadata_layout.addWidget(group)

    def _update_axes_panel(self, plot: "Plot") -> None:
        if self.axes_layout is None:
            return
        while self.axes_layout.count():
            item = self.axes_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            else:
                del item
        if not hasattr(plot, "signal_tree") or plot.signal_tree is None:
            return
        plot_state = getattr(plot, "plot_state", None)
        current_signal = plot_state.current_signal if plot_state else None
        groups = build_axes_groups(plot.signal_tree, current_signal, plot)
        for group in groups:
            self.axes_layout.addWidget(group)

    def _on_cmap_changed(self, cmap_name: str) -> None:
        sub = self.main_window.mdi_area.activeSubWindow()
        if sub is None:
            return
        if hasattr(sub, "set_colormap"):
            sub.set_colormap(cmap_name)

    def _on_contrast_auto(self) -> None:
        w = self.main_window._active_plot()
        if w is None or not hasattr(w, "plot_state") or w.plot_state is None:
            return
        if getattr(w.plot_state, "dimensions", 0) == 2:
            mn, mx = self.histogram.percentile2levels(0.00, 99.0)
            self.histogram.setLevels(mn, mx)

    def _on_contrast_reset(self) -> None:
        w = self.main_window._active_plot()
        if w is None or not hasattr(w, "plot_state") or w.plot_state is None:
            return
        if getattr(w.plot_state, "dimensions", 0) == 2:
            mn, mx = w.image_item.quickMinMax()
            self.histogram.setLevels(mn, mx)

    def _on_histogram_gamma_changed(self, item: HistogramLUTItem) -> None:
        """Apply the histogram's gamma line to the active plot's LUT."""
        w = self.main_window._active_plot()
        if w is None or not hasattr(w, "set_gamma"):
            return
        w.set_gamma(item.gamma)

    def _on_histogram_levels_finished(self, signal: HistogramLUTItem) -> None:
        if (
            signal is None
            or getattr(signal, "bins", None) is None
            or getattr(signal, "counts", None) is None
        ):
            return
        percentiles = signal.get_percentile_levels()
        levels = signal.getLevels()
        w = self.main_window._active_plot()
        if w is None or not hasattr(w, "plot_state") or w.plot_state is None:
            return
        w.plot_state.max_level = levels[1]
        w.plot_state.min_level = levels[0]
        w.plot_state.max_percentile = percentiles[1]
        w.plot_state.min_percentile = percentiles[0]
