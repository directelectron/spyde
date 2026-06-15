from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING

from PySide6 import QtWidgets
from PySide6.QtCore import Qt

if TYPE_CHECKING:
    from spyde.signal_tree import BaseSignalTree
    from spyde.drawing.plots.plot import Plot
from hyperspy.signal import BaseSignal
from spyde.external.qt.labels import EditableLabel
from spyde import METADATA_WIDGET_CONFIG


def _on_axis_field_edit(
    signal_tree: "BaseSignalTree",
    signal: BaseSignal,
    axis,
    field: str,
    line_edit: QtWidgets.QLineEdit,
    is_nav: bool,
    text: str = "",
):
    """Update an axis field on edit. If is_nav, updates all signals in the tree."""
    if is_nav:
        for sig in signal_tree.signals():
            index = sig.axes_manager._axes.index(axis)
            sig.axes_manager._axes[index].__setattr__(field, line_edit.text())
        for plot in signal_tree.navigator_plot_manager.plots.values():
            plot.update_image_rectangle()
    else:
        index = signal.axes_manager._axes.index(axis)
        signal.axes_manager._axes[index].__setattr__(field, line_edit.text())
        for plot in signal_tree.signal_plots:
            if plot.plot_state.current_signal is signal:
                plot.update_image_rectangle()


def build_axes_groups(
    signal_tree: "BaseSignalTree",
    signal: BaseSignal | None,
    plot: "Plot",
) -> list[QtWidgets.QGroupBox]:
    """Build Navigation Axes + Signal Axes QGroupBoxes with editable fields."""
    groups: list[QtWidgets.QGroupBox] = []

    def _make_group(title: str, axes_list, is_nav=False) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(title)
        group.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setMaximumHeight(160)

        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)

        from spyde.qt.style import PANEL_HEADER_QSS
        for col, label in enumerate(["Name", "Scale", "Offset", "Units"]):
            h = QtWidgets.QLabel(label)
            h.setStyleSheet(PANEL_HEADER_QSS)
            if col == 0:
                h.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(h, 0, col)

        for row, axis in enumerate(axes_list, start=1):
            name_edit = EditableLabel(str(axis.name))
            scale_edit = EditableLabel(str(axis.scale))
            offset_edit = EditableLabel(str(axis.offset))
            units_edit = EditableLabel(str(axis.units))

            for w in (name_edit, scale_edit, offset_edit, units_edit):
                w.setFixedWidth(72)
                w.setFixedHeight(18)

            for w, field in zip(
                (name_edit, scale_edit, offset_edit, units_edit),
                ("name", "scale", "offset", "units"),
            ):
                w.editingFinished.connect(
                    partial(_on_axis_field_edit, signal_tree, signal, axis, field, w, is_nav)
                )

            grid.addWidget(name_edit, row, 0)
            grid.addWidget(scale_edit, row, 1)
            grid.addWidget(offset_edit, row, 2)
            grid.addWidget(units_edit, row, 3)

        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 1)

        scroll.setWidget(container)
        v = QtWidgets.QVBoxLayout(group)
        v.setContentsMargins(4, 4, 4, 4)
        v.addWidget(scroll)
        return group

    groups.append(
        _make_group(
            "Navigation Axes",
            signal_tree.root.axes_manager.navigation_axes,
            is_nav=True,
        )
    )
    if signal is not None and not plot.is_navigator:
        groups.append(_make_group("Signal Axes", signal.axes_manager.signal_axes))
    return groups


def build_metadata_dict(signal_tree: "BaseSignalTree") -> dict[str, dict[str, str]]:
    """Return metadata as a plain dict. Callers turn this into widgets."""
    subsections: dict[str, dict[str, str]] = {}
    for subsection, props in METADATA_WIDGET_CONFIG["metadata_widget"].items():
        subsections[subsection] = {}
        for prop, value in props.items():
            current_value, _ = _read_metadata_prop(signal_tree, value)
            subsections[subsection][prop] = (
                f"{current_value} {value.get('units', '')}".strip()
            )
    return subsections


def _read_metadata_prop(signal_tree: "BaseSignalTree", value: dict):
    """Resolve one metadata-config entry to (value, key). `key` is the metadata
    item path if the prop is writable, else None (attr/function props are
    derived and read-only)."""
    if "key" in value:
        return (
            signal_tree.root.metadata.get_item(
                item_path=value["key"], default=value.get("default", "--")
            ),
            value["key"],
        )
    if "attr" in value:
        return signal_tree.get_nested_attr(value["attr"]), None
    if "function" in value:
        fun = signal_tree.get_nested_attr(value["function"])
        return (fun() if callable(fun) else "--"), None
    return "--", None


def _on_metadata_edit(signal_tree, key, units, line_edit, text=""):
    """Write an edited metadata field back onto the tree root's metadata. The
    units suffix is display-only, so strip it before storing the raw value."""
    raw = line_edit.text().strip()
    if units and raw.endswith(units):
        raw = raw[: -len(units)].strip()
    signal_tree.root.metadata.set_item(key, raw)


def build_metadata_groups(signal_tree: "BaseSignalTree") -> list[QtWidgets.QGroupBox]:
    """Build themed Metadata QGroupBoxes whose values reuse the same editable,
    selectable EditableLabel widget as the Plot Axes panel. `key`-backed props
    are editable (write straight back to metadata); derived attr/function props
    render as plain (non-editable) values so they read consistently but can't be
    mangled."""
    from spyde.qt.style import PANEL_HEADER_QSS
    groups: list[QtWidgets.QGroupBox] = []
    for subsection, props in METADATA_WIDGET_CONFIG["metadata_widget"].items():
        group = QtWidgets.QGroupBox(str(subsection))
        group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed
        )
        v = QtWidgets.QVBoxLayout(group)
        v.setContentsMargins(4, 4, 4, 4)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setMaximumHeight(160)
        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)
        grid.setContentsMargins(4, 2, 4, 2)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        for row, (prop, cfg) in enumerate((props or {}).items()):
            current_value, key = _read_metadata_prop(signal_tree, cfg)
            units = cfg.get("units", "")
            display = f"{current_value} {units}".strip()

            key_label = QtWidgets.QLabel(f"{prop}:")
            key_label.setStyleSheet(PANEL_HEADER_QSS)
            key_label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            grid.addWidget(key_label, row, 0)

            value_widget = EditableLabel(display)
            value_widget.setFixedHeight(18)
            if key is not None:
                value_widget.editingFinished.connect(
                    partial(_on_metadata_edit, signal_tree, key, units, value_widget)
                )
            else:
                # derived / read-only: keep the themed label look but don't
                # offer an edit affordance.
                value_widget._label.clicked.disconnect()
            grid.addWidget(value_widget, row, 1)

        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        scroll.setWidget(container)
        v.addWidget(scroll)
        groups.append(group)
    return groups
