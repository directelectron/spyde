from typing import TYPE_CHECKING, Optional
from PySide6 import QtCore, QtGui, QtWidgets
from pyqtgraph import RectROI
import numpy as np


if TYPE_CHECKING:
    from spyde.drawing.toolbars.toolbar import RoundedToolBar


class FileDropWidget(QtWidgets.QWidget):
    """A widget that accepts drag-and-drop or browse-selected files."""

    filesChanged = QtCore.Signal(list)  # emits list of file paths

    def __init__(self, extensions=None, parent=None):
        super().__init__(parent)
        self._extensions = [e.lower() for e in (extensions or [])]
        self._files = []

        self.setAcceptDrops(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        ext_hint = ", ".join(self._extensions) if self._extensions else "any"
        self._label = QtWidgets.QLabel(f"Drop {ext_hint} file(s) here", self)
        self._label.setWordWrap(True)
        self._label.setStyleSheet("color: rgba(255,255,255,150); font-size: 10px;")
        layout.addWidget(self._label)

        from spyde.qt.style import make_button
        self._browse_btn = make_button("Browse…", self)
        self._browse_btn.clicked.connect(self._on_browse)
        layout.addWidget(self._browse_btn)

    def get_files(self):
        return list(self._files)

    def _on_browse(self):
        ext_filter = " ".join(f"*{e}" for e in self._extensions) if self._extensions else "*"
        filter_name = f"{', '.join(self._extensions).upper()} files" if self._extensions else "All files"
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, f"Select file(s)", "", f"{filter_name} ({ext_filter})"
        )
        if paths:
            self._set_files(paths)

    def _set_files(self, paths):
        self._files = list(paths)
        names = [p.split("/")[-1].split("\\")[-1] for p in paths]
        self._label.setText(", ".join(names))
        self.filesChanged.emit(self._files)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if not self._extensions or any(
                url.toLocalFile().lower().endswith(tuple(self._extensions))
                for url in urls
            ):
                event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if not self._extensions
            or url.toLocalFile().lower().endswith(tuple(self._extensions))
        ]
        if paths:
            self._set_files(paths)


class CaretGroup(QtWidgets.QGroupBox):
    """
    A polygonal QGroupBox with a centered triangular caret on one side,
    styled to match RoundedToolBar:
    - Smooth rounded corners
    - Translucent dark fill
    - Thin cosmetic light outline
    side: one of "top", "bottom", "left", "right"
    """

    def __init__(
        self,
        title: str = "",
        parent=None,
        side: str = "auto",
        radius: int = 8,
        caret_base: int = 14,
        caret_depth: int = 8,
        border_width: int = 1,
        padding: int = 15,
        use_mask: bool = False,
        *,
        toolbar: Optional["RoundedToolBar"] = None,
        action_name: Optional[str] = None,
        auto_attach: bool = False,
    ):
        # Optionally derive side from toolbar position when requested.
        def _opposite(pos: str) -> str:
            return {
                "left": "right",
                "right": "left",
                "top": "bottom",
                "bottom": "top",
            }.get(pos, "bottom")

        if toolbar is not None and (side is None or side == "auto"):
            print(
                "Setting CaretGroup side opposite to toolbar position:",
                toolbar.position,
            )
            side = _opposite(getattr(toolbar, "position", "right"))

        super().__init__(title, parent)

        # Store context (optional)
        self.toolbar = toolbar  # type: Optional["RoundedToolBar"]
        self._action_name = action_name  # type: Optional[str]

        self._side = side  # type: str
        self._radius = float(radius)
        self._carrot_base = int(caret_base)
        self._carrot_depth = int(caret_depth)
        self._border_width = float(border_width)
        self._padding = int(padding)
        self._use_mask = bool(use_mask)

        # Visuals to match RoundedToolBar
        self._bg_color = QtGui.QColor(50, 50, 50, 200)
        self._pen_color = QtGui.QColor(255, 255, 255, 60)

        # Transparent background, no default frame
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFlat(True)
        self.setStyleSheet(
            "QGroupBox { border: none; color: white; } "
            "QLabel { background-color: transparent; color: white; } "
            "QLineEdit, QPushButton { color: white; }"
        )
        # Ensure this group has a vertical layout (like the ad-hoc code)
        if self.layout() is None:
            vlay = QtWidgets.QVBoxLayout()
            vlay.setContentsMargins(
                0, 0, 0, 0
            )  # layout padding handled in _update_margins
            self.setLayout(vlay)

        # Do NOT auto-insert into any parent layout; keep free-floating.
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        # Only store references; do not register widgets in the toolbar here.
        # RoundedToolBar.add_action_widget will handle parenting, tracking and binding.
        # ...existing code...
        self._update_margins()
        self._update_mask()

    @staticmethod
    def make_tab_stack(
        tab_labels: list,
        parent=None,
        width: int = 240,
        on_tab_changed=None,
    ):
        """
        Build the standard step-button bar + QStackedWidget used by wizard-style actions.

        Parameters
        ----------
        tab_labels : list of str
            Labels for each tab button (e.g. ``["Auto", "Manual"]``).
        parent : QWidget, optional
            Parent widget for both the bar and the stack.
        width : int
            Fixed pixel width for both widgets. Default 240.
        on_tab_changed : callable(index) | None
            Called whenever the active tab changes.

        Returns
        -------
        (step_bar, stack, select_fn)
            step_bar  — QWidget containing the horizontal button row
            stack     — QStackedWidget; call stack.addWidget(page) for each tab
            select_fn — callable(index) that switches tabs and calls on_tab_changed
        """
        from spyde.qt.style import SmoothButton

        step_bar = QtWidgets.QWidget(parent)
        step_bar.setFixedWidth(width)
        sb_h = QtWidgets.QHBoxLayout(step_bar)
        sb_h.setContentsMargins(0, 0, 0, 0)
        sb_h.setSpacing(3)

        step_btns = []
        for label in tab_labels:
            # Checkable SmoothButton: antialiased rounded corners + accent
            # checked fill, same as every other button in the app.
            b = SmoothButton(label, step_bar, font_px=9)
            b.setCheckable(True)
            b.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            sb_h.addWidget(b)
            step_btns.append(b)

        stack = QtWidgets.QStackedWidget(parent)
        stack.setFixedWidth(width)
        _current_idx = [-1]  # mutable so select_fn closure can track previous tab

        def select_fn(idx: int):
            for i, b in enumerate(step_btns):
                b.setChecked(i == idx)
            stack.setCurrentIndex(idx)
            if callable(on_tab_changed) and idx != _current_idx[0]:
                _current_idx[0] = idx
                on_tab_changed(idx)

        for i, b in enumerate(step_btns):
            b.clicked.connect(lambda _, i=i: select_fn(i))

        return step_bar, stack, select_fn

    def set_side(self, side: str):
        if side not in ("top", "bottom", "left", "right"):
            return
        if self._side != side:
            self._side = side
            self._update_margins()
            self._update_mask()
            self.update()

    def set_use_mask(self, enabled: bool):
        """Enable only if you need precise hit‑testing; it will look more aliased."""
        self._use_mask = bool(enabled)
        self._update_mask()
        self.update()

    def sizeHint(self) -> QtCore.QSize:
        base = super().sizeHint()
        if self._side in ("top", "bottom"):
            return QtCore.QSize(base.width(), base.height() + self._carrot_depth)
        else:
            return QtCore.QSize(base.width() + self._carrot_depth, base.height())

    def finalize_layout(self) -> None:
        """
        Recalculate margins/mask and let Qt recompute the size from children.

        Call this once *after* all child widgets/rows are added, but *before*
        the CaretGroup is positioned/shown by the toolbar. This ensures the
        initial sizeHint already accounts for all content and caret padding,
        preventing first‑render clipping (e.g. for virtual imaging popouts).
        """
        self._update_margins()
        self._update_mask()
        self.adjustSize()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_mask()

    def _update_margins(self):
        # Exclude caret region from the content area via widget contents margins.
        l = r = t = b = self._padding
        if self._side == "top":
            t += self._carrot_depth
        elif self._side == "bottom":
            b += self._carrot_depth
        elif self._side == "left":
            l += self._carrot_depth
        else:  # right
            r += self._carrot_depth
        self.setContentsMargins(l, t, r, b)

        # Apply inner padding only inside the bubble, not in the caret band.
        lay = self.layout()
        if isinstance(lay, QtWidgets.QLayout):
            il = ir = it = ib = self._padding
            if self._side == "top":
                it = 0
            elif self._side == "bottom":
                ib = 0
            elif self._side == "left":
                il = 0
            else:  # right
                ir = 0
            lay.setContentsMargins(int(il), int(it), int(ir), int(ib))

    def _bubble_rect(self) -> QtCore.QRectF:
        # Sub‑pixel align for crisp 1px pen
        bw = self._border_width
        rect = QtCore.QRectF(self.rect()).adjusted(
            bw / 2.0, bw / 2.0, -bw / 2.0, -bw / 2.0
        )
        if self._side == "top":
            rect.adjust(0, self._carrot_depth, 0, 0)
        elif self._side == "bottom":
            rect.adjust(0, 0, 0, -self._carrot_depth)
        elif self._side == "left":
            rect.adjust(self._carrot_depth, 0, 0, 0)
        else:  # right
            rect.adjust(0, 0, -self._carrot_depth, 0)
        return rect

    def _caret_polygon(self, bubble: QtCore.QRectF) -> QtGui.QPolygonF:
        base = float(self._carrot_base)
        depth = float(self._carrot_depth)
        if self._side in ("top", "bottom"):
            cx = bubble.center().x()
            x1 = cx - base / 2.0
            x2 = cx + base / 2.0
            if self._side == "top":
                y = bubble.top()
                return QtGui.QPolygonF(
                    [
                        QtCore.QPointF(x1, y),
                        QtCore.QPointF(x2, y),
                        QtCore.QPointF(cx, y - depth),
                    ]
                )
            else:
                y = bubble.bottom()
                return QtGui.QPolygonF(
                    [
                        QtCore.QPointF(x1, y),
                        QtCore.QPointF(x2, y),
                        QtCore.QPointF(cx, y + depth),
                    ]
                )
        else:
            cy = bubble.center().y()
            y1 = cy - base / 2.0
            y2 = cy + base / 2.0
            if self._side == "left":
                x = bubble.left()
                return QtGui.QPolygonF(
                    [
                        QtCore.QPointF(x, y1),
                        QtCore.QPointF(x, y2),
                        QtCore.QPointF(x - depth, cy),
                    ]
                )
            else:
                x = bubble.right()
                return QtGui.QPolygonF(
                    [
                        QtCore.QPointF(x, y1),
                        QtCore.QPointF(x, y2),
                        QtCore.QPointF(x + depth, cy),
                    ]
                )

    def _path(self) -> QtGui.QPainterPath:
        bubble = self._bubble_rect()
        path = QtGui.QPainterPath()
        path.addRoundedRect(bubble, self._radius, self._radius)
        path.addPolygon(self._caret_polygon(bubble))
        return path.simplified()

    def _update_mask(self):
        # Mask is binary and causes aliasing; keep it off for smooth visuals.
        if self._use_mask:
            path = self._path()
            region = QtGui.QRegion(path.toFillPolygon().toPolygon())
            self.setMask(region)
        else:
            self.clearMask()

    def paintEvent(self, event: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)

        path = self._path()

        # Fill to match RoundedToolBar
        p.setBrush(QtGui.QBrush(self._bg_color))

        # 1px cosmetic pen with round joins/caps for smoother edges
        pen = QtGui.QPen(self._pen_color)
        pen.setWidthF(self._border_width)
        pen.setCosmetic(True)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        p.setPen(pen)

        p.drawPath(path)


class CaretParams(CaretGroup):
    """
    A Caret Group specialized for parameter controls and submitting parameters for some action.

    This is built from a parameters definition dictionary, which specifies the parameters to create,
    their types, default values, and optional display conditions based on other parameters.
    """

    def __init__(
        self,
        title: str = "",
        parent=None,
        side: str = None,
        radius: int = 8,
        caret_base: int = 14,
        caret_depth: int = 8,
        border_width: int = 1,
        padding: int = 8,
        use_mask: bool = False,
        parameters: dict = None,
        function: callable = None,
        *,
        toolbar: Optional["RoundedToolBar"] = None,
        action_name: Optional[str] = None,
        auto_attach: bool = False,
    ):
        super().__init__(
            title,
            parent,
            side,
            radius,
            caret_base,
            caret_depth,
            border_width,
            padding,
            use_mask,
            toolbar=toolbar,
            action_name=action_name,
            auto_attach=auto_attach,
        )
        self.setStyleSheet(
            "QGroupBox { border: none; } QLabel { background-color: transparent; }"
        )

        # Storage
        self.kwargs = {}  # key -> editor widget
        self._rows = {}  # key -> QWidget containing the row
        self._conditions = {}  # key -> {'parameter': ..., 'value': ...}
        self._param_types = {}  # key -> dtype string
        self._dependents = {}  # controller_key -> [dependent_keys]
        self._parameters_def = parameters or {}
        self._rect_rois = {}  # key -> RectROI (for RectangleSelector)

        print("Creating CaretParams with parameters:", parameters)

        self._params_layout = self.layout()

        # Build rows (always create; visibility controlled by conditions)
        for key, item in (self._parameters_def or {}).items():
            dtype = item.get("type", "str")
            name = item.get("name", key)
            default = item.get("default", "")
            self._param_types[key] = dtype

            # Row container so we can hide/show the entire line
            row_widget = QtWidgets.QWidget(self)
            h_layout = QtWidgets.QHBoxLayout(row_widget)
            h_layout.setContentsMargins(0, 0, 0, 0)
            h_layout.setSpacing(4)

            label = QtWidgets.QLabel(name, row_widget)
            label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            label.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Minimum,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )

            # Editor
            if dtype == "int":
                editor = QtWidgets.QLineEdit(str(default), row_widget)
                editor.setValidator(QtGui.QIntValidator())
            elif dtype == "float":
                editor = QtWidgets.QLineEdit(str(default), row_widget)
                editor.setValidator(QtGui.QDoubleValidator())
            elif dtype == "enum":
                editor = QtWidgets.QComboBox(row_widget)
                editor.setSizeAdjustPolicy(
                    QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
                )
                editor.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Expanding,
                    QtWidgets.QSizePolicy.Policy.Preferred,
                )
                options = item.get("options", [])
                editor.addItems([str(opt) for opt in options])
                if default in options:
                    editor.setCurrentText(str(default))
            elif dtype == "button":
                from spyde.qt.style import make_button as _make_btn
                editor = _make_btn(item.get("label", name), row_widget)
                callback = item.get("callback")
                if callable(callback):
                    editor.clicked.connect(callback)
            elif dtype == "button_row":
                # Two buttons side-by-side in one row; no label.
                # "buttons": [{"key": k, "label": l, "callback": fn}, ...]
                buttons_def = item.get("buttons", [])
                from spyde.qt.style import make_button as _make_row_btn
                for btn_def in buttons_def:
                    btn = _make_row_btn(btn_def.get("label", btn_def.get("key", "")), row_widget)
                    cb = btn_def.get("callback")
                    if callable(cb):
                        btn.clicked.connect(cb)
                    h_layout.addWidget(btn, 1)
                    bkey = btn_def.get("key", btn_def.get("label", ""))
                    self.kwargs[bkey] = btn
                    self._param_types[bkey] = "button"
                self._rows[key] = row_widget
                self._params_layout.addWidget(row_widget)
                continue
            elif dtype == "RectangleSelector":
                # Add a rectangle ROI to the associated plot; visible only when action toggled.
                editor = QtWidgets.QLabel("Use selection on plot", row_widget)
                editor.setEnabled(False)

                print("Adding RectangleSelector ROI for parameter:", key)
                print("Toolbar available:", self.toolbar is not None)
                print(
                    "toolbar.plot available:",
                    (
                        getattr(self.toolbar, "plot", None) is not None
                        if self.toolbar
                        else "N/A"
                    ),
                )

                if (
                    self.toolbar is not None
                    and getattr(self.toolbar, "plot_state", None) is not None
                ):
                    # Create a modest default rectangle; users can reposition/resize.
                    current_signal = self.toolbar.plot_state.current_signal
                    extent = current_signal.axes_manager.signal_extent
                    print("Current signal extent:", extent)

                    center = (extent[1] + extent[0]) / 2, (extent[3] + extent[2]) / 2
                    left, right, bottom, top = extent
                    print("Computed signal center:", center)
                    width = np.abs(extent[1] - extent[0])
                    print("Computed signal width:", width)

                    size = item.get("size", 0.1)
                    position = item.get("position", "centered")

                    if isinstance(size, (tuple, list)) and len(size) == 2:
                        frac_w, frac_h = float(size[0]), float(size[1])
                    else:
                        frac_w = frac_h = float(size)

                    roi_w = max(1.0, width * frac_w if frac_w <= 1.0 else frac_w)
                    roi_h = max(1.0, width * frac_h if frac_h <= 1.0 else frac_h)
                    print("Computed ROI size:", roi_w, roi_h)

                    if position == "centered":
                        po = QtCore.QPointF(
                            center[0] - roi_w / 2.0, center[1] - roi_h / 2.0
                        )
                    elif position == "top-left":
                        po = QtCore.QPointF(left, top)
                    elif position == "top-right":
                        po = QtCore.QPointF(right - roi_w, top)
                    elif position == "bottom-left":
                        po = QtCore.QPointF(left, bottom - roi_h)
                    elif position == "bottom-right":
                        po = QtCore.QPointF(right - roi_w, bottom - roi_h)
                    else:
                        po = QtCore.QPointF(
                            center[0] - roi_w / 2.0, center[1] - roi_h / 2.0
                        )

                    print("Computed ROI position:", po)
                    print("Adding RectangleSelector ROI to plot.")
                    roi = RectROI(pos=po, size=(roi_w, roi_h))

                    # Ensure it starts hidden; toolbar will toggle visibility with the action.
                    roi.setVisible(True)
                    # Add to the plot and register with toolbar for toggle and cleanup.
                    # self.toolbar.plot.plot_item.addItem(roi)
                    if self._action_name:
                        self.toolbar.register_action_plot_item(self._action_name, roi)
                    self._rect_rois[key] = roi
                else:
                    print(
                        "No toolbar/plot available; RectangleSelector ROI not created."
                    )
            elif dtype == "file_drop":
                extensions = item.get("extensions", [])
                editor = FileDropWidget(extensions=extensions, parent=row_widget)
            else:  # default to string
                editor = QtWidgets.QLineEdit(str(default), row_widget)

            h_layout.addWidget(label)
            h_layout.addWidget(editor, 1)

            # Store
            self.kwargs[key] = editor
            self._rows[key] = row_widget

            # Capture conditional visibility, if any
            display_condition = item.get("display_condition")
            if display_condition and isinstance(display_condition, dict):
                self._conditions[key] = {
                    "parameter": display_condition.get("parameter"),
                    "value": display_condition.get("value"),
                }
                controller = self._conditions[key]["parameter"]
                if controller:
                    self._dependents.setdefault(controller, []).append(key)

            # Add row to main layout
            self._params_layout.addWidget(row_widget)

        # Submit button — same shared theme as every other popout button
        from spyde.qt.style import make_button
        self.submit_button = make_button("Submit", parent=self)
        self._params_layout.addWidget(self.submit_button)

        # Layout/style wiring
        layout = self.layout()
        layout.setContentsMargins(2, 2, 2, 2)
        self.submit_button.clicked.connect(self._on_submit_clicked)
        from spyde.qt.style import INPUT_QSS
        self.setStyleSheet(
            "QGroupBox { border: none; color: white; } "
            "QLabel { background-color: transparent; color: white; } "
            + INPUT_QSS
            + " QPushButton { color: white; }"
        )
        self.toolbar = toolbar
        self.function = function

        # Connect signals for controllers that affect dependents
        self._connect_visibility_triggers()
        # Apply initial visibility
        self._update_all_visibility()

        # Important: ensure sizeHint accounts for all rows + caret before first show
        self.finalize_layout()

        self._tab_stack = None      # QStackedWidget, set by add_extra_tab
        self._tab_select_fn = None  # callable(idx), set by add_extra_tab
        self._on_first_show_cb = None  # callable(), invoked once on first showEvent

    def add_extra_tab(self, label: str, widget: QtWidgets.QWidget, on_tab_changed=None):
        """
        Add a named tab alongside the existing Auto parameters page.

        On the first call this restructures the layout: the existing parameter
        rows and submit button are moved into a container widget that becomes
        page 0 ("Auto"), and the supplied widget becomes page 1.  Subsequent
        calls add further pages.

        Parameters
        ----------
        label : str
            Tab button label for the new page.
        widget : QWidget
            Content widget for the new page.
        on_tab_changed : callable(index) | None
            Called whenever the active tab changes (receives page index).
        """
        outer_layout = self.layout()

        if self._tab_stack is None:
            # First extra tab: restructure layout around a tab bar + stacked widget.
            # Collect all current top-level children (rows + submit button).
            children = []
            while outer_layout.count():
                item = outer_layout.takeAt(0)
                if item.widget():
                    children.append(item.widget())

            # Wrap them in a container that becomes page 0.
            auto_page = QtWidgets.QWidget(self)
            auto_layout = QtWidgets.QVBoxLayout(auto_page)
            auto_layout.setContentsMargins(0, 0, 0, 0)
            auto_layout.setSpacing(2)
            for child in children:
                child.setParent(auto_page)
                auto_layout.addWidget(child)

            step_bar, stack, select_fn = CaretGroup.make_tab_stack(
                ["Auto", label],
                parent=self,
                width=self.width() or 220,
                on_tab_changed=on_tab_changed,
            )
            stack.addWidget(auto_page)
            stack.addWidget(widget)

            outer_layout.addWidget(step_bar)
            outer_layout.addWidget(stack)

            self._tab_stack = stack
            self._tab_select_fn = select_fn
        else:
            # Subsequent extra tabs: just add more pages.
            old_labels = [
                self._tab_stack.widget(i).objectName() or f"Tab {i}"
                for i in range(self._tab_stack.count())
            ]
            step_bar, stack, select_fn = CaretGroup.make_tab_stack(
                old_labels + [label],
                parent=self,
                width=self.width() or 220,
                on_tab_changed=on_tab_changed,
            )
            # Re-parent pages
            pages = [self._tab_stack.widget(i) for i in range(self._tab_stack.count())]
            for page in pages:
                stack.addWidget(page)
            stack.addWidget(widget)

            # Swap the old stack for the new one in the layout
            outer_layout.replaceWidget(self._tab_stack, stack)
            self._tab_stack.setParent(None)
            self._tab_stack = stack
            self._tab_select_fn = select_fn

        self.finalize_layout()

    def showEvent(self, event):
        super().showEvent(event)
        if self._on_first_show_cb is not None:
            cb = self._on_first_show_cb
            self._on_first_show_cb = None  # fire exactly once
            cb()

    def _connect_visibility_triggers(self):
        from functools import partial

        for controller_key, dependents in self._dependents.items():
            controller = self.kwargs.get(controller_key)
            if controller is None:
                continue
            if isinstance(controller, QtWidgets.QComboBox):
                controller.currentTextChanged.connect(
                    partial(self._on_controller_changed, controller_key)
                )
            else:
                # QLineEdit (str/int/float)
                if hasattr(controller, "textChanged"):
                    controller.textChanged.connect(
                        partial(self._on_controller_changed, controller_key)
                    )

    def get_parameter_widget(self, key: str):
        return self.kwargs.get(key)

    def _on_controller_changed(self, controller_key, *args):
        # Update only rows depending on this controller
        for dep_key in self._dependents.get(controller_key, []):
            cond = self._conditions.get(dep_key)
            row = self._rows.get(dep_key)
            if row is None or cond is None:
                continue
            row.setVisible(self._evaluate_display_condition(cond))

    def _update_all_visibility(self):
        for key, cond in self._conditions.items():
            row = self._rows.get(key)
            if row is None:
                continue
            row.setVisible(self._evaluate_display_condition(cond))
        self._update_margins()

    def _evaluate_display_condition(self, cond: dict) -> bool:
        controller_key = cond.get("parameter")
        target_value = cond.get("value")
        if controller_key not in self.kwargs:
            return False

        controller_type = self._param_types.get(controller_key, "str")
        current_value = self._get_param_value(controller_key, controller_type)

        # Coerce target to controller type
        try:
            if controller_type == "int":
                target_value = int(target_value)
            elif controller_type == "float":
                target_value = float(target_value)
            elif controller_type == "bool":
                target_value = target_value != "False"
            else:
                target_value = str(target_value)
        except Exception:
            # If coercion fails, compare as strings
            target_value = str(target_value)
            current_value = str(current_value)

        return current_value == target_value

    def _get_param_value(self, key: str, dtype: str):
        # RectangleSelector returns bounding box in pixel coords (x, y, width, height).
        if dtype == "RectangleSelector":
            roi = self._rect_rois.get(key)

            print(self.toolbar)
            print(self.toolbar.plot)
            if (
                roi is None
                or self.toolbar is None
                or getattr(self.toolbar, "plot", None) is None
            ):
                return None
            try:
                lower_left = roi.pos()
                size = roi.size()
                # Map ROI geometry to pixel coordinates using the image item's transform.
                inv_transform, _ = self.toolbar.plot.image_item.transform().inverted()
                ll_px = inv_transform.map(lower_left)
                sz_px = inv_transform.map(size) - inv_transform.map(
                    QtCore.QPointF(0, 0)
                )
                x = int(round(ll_px.x()))
                y = int(round(ll_px.y()))
                w = int(round(sz_px.x()))
                h = int(round(sz_px.y()))
                return (x, y, w, h)
            except Exception as e:
                print("Failed to compute RectangleSelector value:", e)
                return None

        if dtype == "file_drop":
            w = self.kwargs.get(key)
            return w.get_files() if w is not None else []

        w = self.kwargs.get(key)
        if w is None:
            return None
        if isinstance(w, QtWidgets.QComboBox):
            return w.currentText()
        # QLineEdit and QLabel
        txt = w.text() if hasattr(w, "text") else ""
        try:
            if dtype == "int":
                return int(txt)
            if dtype == "float":
                return float(txt)
        except Exception:
            pass
        return txt

    def _on_submit_clicked(self):
        if self.function is not None:
            params = {}
            for key, widget in self.kwargs.items():
                row = self._rows.get(key)
                if row is None or not row.isVisibleTo(self):
                    continue
                dtype = self._param_types.get(key, "str")
                if dtype == "RectangleSelector":
                    params[key] = self._get_param_value(key, dtype)
                    continue
                if hasattr(widget, "validator") and isinstance(
                    widget.validator(), QtGui.QDoubleValidator
                ):
                    params[key] = float(widget.text())
                elif hasattr(widget, "validator") and isinstance(
                    widget.validator(), QtGui.QIntValidator
                ):
                    params[key] = int(widget.text())
                elif isinstance(widget, QtWidgets.QComboBox):
                    params[key] = widget.currentText()
                elif dtype == "file_drop":
                    params[key] = widget.get_files()
                else:
                    params[key] = widget.text()

            new_signal = self.function(toolbar=self.toolbar, **params)
            if (
                new_signal is not None
                and self.toolbar is not None
                and hasattr(self.toolbar, "plot")
            ):
                self.toolbar.plot.set_plot_state(new_signal)
