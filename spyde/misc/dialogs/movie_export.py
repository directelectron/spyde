"""
Export some Signal as a movie.

This dialog allows the user to select parameters for exporting a movie from a selected signal.
"""

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QPushButton,
    QDialogButtonBox,
    QCheckBox,
    QGroupBox,
    QFileDialog,
    QWidget,
    QProgressDialog,
)
from PySide6.QtCore import QCoreApplication
import numpy as np

from distributed import Future

import imageio.v2 as imageio
from spyde.external.pyqtgraph.scale_bar import OutlinedScaleBar
from spyde.misc.utils import get_nice_length

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spyde.drawing.plots.plot import Plot

import pyqtgraph as pg


class MovieExportDialog(QDialog):
    """
    Dialog to export a movie from a selected Plot's underlying HyperSpy Signal.

    Features:
    - Choose navigation axis to animate
    - Set frame start/end
    - Optional ROI (x, y, width, height)
    - Add scale bar
    - Show/hide axes
    - Control FPS and output path
    """

    def __init__(self, plot: "Plot", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Movie")
        self._plot = plot  # type: Plot
        self._signal = self._plot.plot_state.current_signal
        if self._signal is None:
            raise ValueError(
                "MovieExportDialog: Could not find a HyperSpy Signal in the provided Plot."
            )

        self._output_path = None

        self._build_ui()
        self._populate_axes()
        self._update_axis_limits()
        self._populate_roi_defaults()

    # ---------------- UI ----------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Info
        sig_shape = self._signal.axes_manager.signal_shape
        nav_shape = self._signal.axes_manager.navigation_shape
        info = QLabel(f"Signal shape: {sig_shape}, Navigation shape: {nav_shape}")
        layout.addWidget(info)

        # Axis select + frame range
        axis_row = QHBoxLayout()
        axis_row.addWidget(QLabel("Navigation axis:"))
        self.axis_combo = QComboBox()
        self.axis_combo.currentIndexChanged.connect(self._update_axis_limits)
        axis_row.addWidget(self.axis_combo)

        axis_row.addWidget(QLabel("Start:"))
        self.start_spin = QSpinBox()
        self.start_spin.setMinimum(0)
        axis_row.addWidget(self.start_spin)

        axis_row.addWidget(QLabel("End:"))
        self.end_spin = QSpinBox()
        self.end_spin.setMinimum(0)
        axis_row.addWidget(self.end_spin)

        axis_row_widget = QWidget()
        axis_row_widget.setLayout(axis_row)
        layout.addWidget(axis_row_widget)

        # ROI group
        roi_group = QGroupBox("ROI (Region of Interest)")
        roi_v = QVBoxLayout(roi_group)
        self.roi_enable = QCheckBox("Enable ROI")
        self.roi_enable.setChecked(False)
        self.roi_enable.stateChanged.connect(self._toggle_roi_enabled)
        roi_v.addWidget(self.roi_enable)

        roi_grid = QHBoxLayout()
        roi_grid.addWidget(QLabel("x:"))
        self.roi_x = QSpinBox()
        self.roi_x.setMinimum(0)
        self.roi_x.valueChanged.connect(self._roi_bounds_guard)
        roi_grid.addWidget(self.roi_x)

        roi_grid.addWidget(QLabel("y:"))
        self.roi_y = QSpinBox()
        self.roi_y.setMinimum(0)
        self.roi_y.valueChanged.connect(self._roi_bounds_guard)
        roi_grid.addWidget(self.roi_y)

        roi_grid.addWidget(QLabel("w:"))
        self.roi_w = QSpinBox()
        self.roi_w.setMinimum(1)
        self.roi_w.valueChanged.connect(self._roi_bounds_guard)
        roi_grid.addWidget(self.roi_w)

        roi_grid.addWidget(QLabel("h:"))
        self.roi_h = QSpinBox()
        self.roi_h.setMinimum(1)
        self.roi_h.valueChanged.connect(self._roi_bounds_guard)
        roi_grid.addWidget(self.roi_h)

        roi_grid_widget = QWidget()
        roi_grid_widget.setLayout(roi_grid)
        roi_v.addWidget(roi_grid_widget)
        layout.addWidget(roi_group)

        # Options
        opts_row = QHBoxLayout()
        self.cb_show_axes = QCheckBox("Show axes")
        self.cb_show_axes.setChecked(False)
        opts_row.addWidget(self.cb_show_axes)
        self.cb_scale_bar = QCheckBox("Add scale bar")
        self.cb_scale_bar.setChecked(False)
        opts_row.addWidget(self.cb_scale_bar)

        opts_row.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(10)
        opts_row.addWidget(self.fps_spin)

        opts_row_widget = QWidget()
        opts_row_widget.setLayout(opts_row)
        layout.addWidget(opts_row_widget)

        # Output file
        out_row = QHBoxLayout()
        self.out_label = QLabel("Output: (not selected)")
        out_row.addWidget(self.out_label)
        self.choose_btn = QPushButton("Choose…")
        self.choose_btn.clicked.connect(self._choose_output)
        out_row.addWidget(self.choose_btn)
        out_row_widget = QWidget()
        out_row_widget.setLayout(out_row)
        layout.addWidget(out_row_widget)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._toggle_roi_enabled()

    # ---------------- Helpers ----------------

    def _populate_axes(self):
        self.axis_combo.clear()
        am = self._signal.axes_manager
        if am.navigation_dimension == 0:
            # No navigation dims, still populate a dummy axis
            self.axis_combo.addItem("(none)")
            self.axis_combo.setEnabled(False)
        else:
            for ax in am.navigation_axes:
                name = ax.name or "nav"
                self.axis_combo.addItem(f"{name} (size={ax.size})")

    def _update_axis_limits(self):
        am = self._signal.axes_manager
        if am.navigation_dimension == 0:
            self.start_spin.setRange(0, 0)
            self.end_spin.setRange(0, 0)
            self.start_spin.setValue(0)
            self.end_spin.setValue(0)
            return
        idx = self.axis_combo.currentIndex()
        size = am.navigation_axes[idx].size
        self.start_spin.setRange(0, max(0, size - 1))
        self.end_spin.setRange(0, max(0, size - 1))
        self.start_spin.setValue(0)
        self.end_spin.setValue(size - 1)

    def _populate_roi_defaults(self):
        sy, sx = self._signal.axes_manager.signal_shape
        self.roi_x.setRange(0, max(0, sx - 1))
        self.roi_y.setRange(0, max(0, sy - 1))
        self.roi_w.setRange(1, max(1, sx))
        self.roi_h.setRange(1, max(1, sy))
        self.roi_x.setValue(0)
        self.roi_y.setValue(0)
        self.roi_w.setValue(sx)
        self.roi_h.setValue(sy)

    def _toggle_roi_enabled(self):
        enabled = self.roi_enable.isChecked()
        for w in (self.roi_x, self.roi_y, self.roi_w, self.roi_h):
            w.setEnabled(enabled)

    def _roi_bounds_guard(self):
        sy, sx = self._signal.axes_manager.signal_shape
        x = min(self.roi_x.value(), max(0, sx - 1))
        y = min(self.roi_y.value(), max(0, sy - 1))
        self.roi_x.blockSignals(True)
        self.roi_y.blockSignals(True)
        self.roi_x.setValue(x)
        self.roi_y.setValue(y)
        self.roi_x.blockSignals(False)
        self.roi_y.blockSignals(False)

        self.roi_w.setMaximum(max(1, sx - x))
        self.roi_h.setMaximum(max(1, sy - y))

    def _choose_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Movie",
            "",
            "MP4 Video (*.mp4);;GIF (*.gif);;All Files (*)",
        )
        if path:
            self._output_path = path
            self.out_label.setText(f"Output: {path}")

    # ---------------- Export ----------------

    def _on_accept(self):
        if not self._output_path:
            self._choose_output()
            if not self._output_path:
                return

        try:
            self._export_movie()
        except Exception as e:
            # Minimal user feedback; in a real app, show a message box
            print(f"Movie export failed: {e}")
            raise
        self.accept()

    def _export_movie(self):
        s = self._signal

        # Determine navigation indexing
        am = s.axes_manager
        nav_dim = am.navigation_dimension

        axis_idx = self.axis_combo.currentIndex()
        size = am.navigation_axes[axis_idx].size
        start = self.start_spin.value()
        end = self.end_spin.value()
        if start > end:
            start, end = end, start
        start = max(0, min(start, size - 1))
        end = max(0, min(end, size - 1))
        frame_count = end - start + 1

        # ROI
        sy, sx = am.signal_shape
        if self.roi_enable.isChecked():
            x = self.roi_x.value()
            y = self.roi_y.value()
            w = self.roi_w.value()
            h = self.roi_h.value()
        else:
            x, y, w, h = 0, 0, sx, sy

        # Writer
        fps = self.fps_spin.value()
        path = self._output_path
        is_gif = path.lower().endswith(".gif")
        writer_kw = {}
        if is_gif:
            writer_kw["duration"] = 1.0 / max(fps, 1)
        else:
            if not path.lower().endswith(".mp4"):
                path = f"{path}.mp4"
                self._output_path = path
                try:
                    self.out_label.setText(f"Output: {path}")
                except Exception:
                    pass
            writer_kw.update(
                {
                    "fps": fps,
                    "format": "FFMPEG",
                    "codec": "libx264",
                    "ffmpeg_params": [
                        "-crf",
                        "18",
                        "-preset",
                        "medium",
                        "-pix_fmt",
                        "yuv420p",
                    ],
                }
            )

        # Render via pyqtgraph: ImageItem + optional axes + scale bar, then export to QImage
        plot_item = pg.PlotItem()

        self._export_glw = pg.GraphicsLayoutWidget()
        self._export_glw.setBackground("w")
        self._export_glw.resize(int(w), int(h))
        self._export_glw.addItem(plot_item)
        vb = plot_item.getViewBox()
        vb.setAspectLocked(True)

        # Axes visibility
        if not self.cb_show_axes.isChecked():
            plot_item.hideAxis("left")
            plot_item.hideAxis("bottom")
        else:
            plot_item.showAxis("left")
            plot_item.showAxis("bottom")

        # crop the signal:
        s_cropped = s.inav[start : end + 1].isig[y : y + h, x : x + w]

        img_item = pg.ImageItem(image=np.zeros((h, w)))
        plot_item.addItem(img_item)

        # Scale bar
        if self.cb_scale_bar.isChecked():
            nice_length, units = get_nice_length(s, is_navigator=False)
            sb = OutlinedScaleBar(
                nice_length,
                suffix=units,
                pen=pg.mkPen(0, 0, 0, 200),
                brush=pg.mkBrush(255, 255, 255, 180),
            )
            sb.setParentItem(vb)
            sb.anchor((1, 1), (1, 1), offset=(-12, -12))
        with imageio.get_writer(path, **writer_kw) as writer:
            from pyqtgraph.exporters import ImageExporter

            exporter = ImageExporter(plot_item)
            params = exporter.parameters()
            scale = max(1, int(round(self._export_glw.devicePixelRatioF())))
            params.param("antialias").setValue(True)
            params.param("background").setValue(QColor(255, 255, 255))
            params.param("width").setValue(int(w * scale))
            params.param("height").setValue(int(h * scale))
            progress = QProgressDialog("Exporting movie...", None, 0, frame_count, self)
            progress.setWindowTitle("Exporting")
            progress.setAutoClose(True)
            progress.setAutoReset(True)
            progress.setValue(0)
            progress.setCancelButtonText("Cancel")
            progress.show()

            for i in range(frame_count):
                if progress.wasCanceled():
                    break

                if s_cropped._lazy:
                    s_frame = s_cropped._get_cache_dask_chunk((i,), get_result=True)
                    if isinstance(s_frame, Future):
                        s_frame = s_frame.result()
                else:
                    s_frame = s_cropped.inav[i].data

                img_item.setImage(s_frame)
                qimg = exporter.export(toBytes=True)

                # Convert to RGB numpy array for imageio
                img = pg.functions.ndarray_from_qimage(qimg)
                writer.append_data(img)

                # Explicitly delete large objects to free memory
                del s_frame, qimg, img

                progress.setValue(i + 1)
                QCoreApplication.processEvents()

            progress.close()
