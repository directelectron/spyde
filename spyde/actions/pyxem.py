import threading
import numpy as np
from typing import Tuple, List, Optional

from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
from PySide6.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph import RectROI, CircleROI, mkPen

from spyde.drawing.toolbars.toolbar import RoundedToolBar
from spyde.drawing.toolbars.plot_control_toolbar import resolve_icon_path
from spyde.external.pyqtgraph.ring_roi import RingROI
from diffsims.generators.simulation_generator import SimulationGenerator
from orix.sampling import get_sample_reduced_fundamental


def roi_to_mask(roi, signal) -> np.ndarray:
    """Convert a PyQtGraph ROI to a float32 mask matching signal.data's last two axes.

    HyperSpy Signal2D stores data as (..., slow_k, fast_k) where:
      signal_axes[0] = fast_k (innermost, kx)
      signal_axes[1] = slow_k (next-innermost, ky)

    pyqtgraph renders ImageItem in col-major order: data[i, j] lands at scene
    position (x=i, y=j).  With data shape (ky_size, kx_size), axis 0 (ky) maps
    to scene-x and axis 1 (kx) maps to scene-y.

    Therefore ROI coordinates from the scene have:
      roi.pos().x() / roi.size().x()  → ky coordinate (scene-x = ky)
      roi.pos().y() / roi.size().y()  → kx coordinate (scene-y = kx)

    The mask returned has shape (ky_size, kx_size) = (slow_k, fast_k), matching
    data[..., ky, kx] so that da.tensordot(..., axes=([-2,-1],[0,1])) is correct.
    """
    sig_axes = signal.axes_manager.signal_axes
    # sig_axes[0] = fast/innermost = kx;  sig_axes[1] = slow/next = ky
    ky_axis = sig_axes[1]
    kx_axis = sig_axes[0]

    ky_size = ky_axis.size
    kx_size = kx_axis.size

    # Build coordinate grids over (ky_size, kx_size) — matching data shape
    ky_pixels = np.arange(ky_size)
    kx_pixels = np.arange(kx_size)
    # kx_grid[ky_idx, kx_idx] = kx_idx,  ky_grid[ky_idx, kx_idx] = ky_idx
    kx_grid, ky_grid = np.meshgrid(kx_pixels, ky_pixels)

    # Scene-x = ky coordinate;  scene-y = kx coordinate  (pyqtgraph col-major)
    scene_x = ky_grid * ky_axis.scale + ky_axis.offset
    scene_y = kx_grid * kx_axis.scale + kx_axis.offset

    if isinstance(roi, RingROI):
        outer_roi = roi.rois[1]
        outer_pos = outer_roi.pos()
        outer_size = outer_roi.size()
        cx = outer_pos.x() + outer_size.x() / 2
        cy = outer_pos.y() + outer_size.y() / 2
        inner_r = roi.rois[0].size().x() / 2
        outer_r = outer_size.x() / 2
        dist2 = (scene_x - cx) ** 2 + (scene_y - cy) ** 2
        mask_bool = (dist2 >= inner_r ** 2) & (dist2 <= outer_r ** 2)

    elif isinstance(roi, CircleROI):
        pos = roi.pos()
        size = roi.size()
        cx = pos.x() + size.x() / 2
        cy = pos.y() + size.y() / 2
        r = size.x() / 2
        dist2 = (scene_x - cx) ** 2 + (scene_y - cy) ** 2
        mask_bool = dist2 <= r ** 2

    elif isinstance(roi, RectROI):
        pos = roi.pos()
        size = roi.size()
        x0, x1 = pos.x(), pos.x() + size.x()
        y0, y1 = pos.y(), pos.y() + size.y()
        # Use half-open interval [x0, x1) so the ROI covers exactly
        # the pixels whose left edges fall within the rectangle, matching
        # pyqtgraph's convention that pos+size is the exclusive right edge.
        mask_bool = (
            (scene_x >= x0) & (scene_x < x1) &
            (scene_y >= y0) & (scene_y < y1)
        )

    else:
        raise TypeError(f"Unsupported ROI type: {type(roi)}")

    return mask_bool.astype(np.float32)


def _roi_metadata(roi) -> dict:
    """Extract ROI geometry as a plain dict for signal metadata storage."""
    if isinstance(roi, RingROI):
        return {
            "type": "ring",
            "center": (roi.rois[1].pos().x() + roi.rois[1].size().x() / 2,
                       roi.rois[1].pos().y() + roi.rois[1].size().y() / 2),
            "inner_radius": roi.rois[0].size().x() / 2,
            "outer_radius": roi.rois[1].size().x() / 2,
        }
    elif isinstance(roi, CircleROI):
        return {
            "type": "disk",
            "center": (roi.pos().x() + roi.size().x() / 2,
                       roi.pos().y() + roi.size().y() / 2),
            "radius": roi.size().x() / 2,
        }
    elif isinstance(roi, RectROI):
        return {
            "type": "rectangle",
            "pos": (roi.pos().x(), roi.pos().y()),
            "size": (roi.size().x(), roi.size().y()),
        }
    return {}


def _start_progress_poll(future, indicator, client, timer_holder: list,
                          n_chunks: int = None):
    """
    Track compute progress and update the indicator.

    If *n_chunks* is provided the indicator is initialised with that total and
    a QTimer polls ``future.done()`` every 200 ms — incrementing by one tick
    per poll so the arc visibly moves even when chunk callbacks are unavailable.

    When *n_chunks* is None the old scheduler_info path is used as a fallback
    (kept for callers that don't know the chunk count).
    """
    from PySide6 import QtCore as _QtCore

    if n_chunks is not None:
        total = max(n_chunks, 1)
        indicator.set_computing(total_tasks=total)
        completed = [0]

        timer = _QtCore.QTimer()
        timer.setInterval(200)
        timer_holder.append(timer)

        def _poll():
            if future.done():
                timer.stop()
                indicator.set_done()
                return
            # Advance one tick per poll so the ring visibly fills
            completed[0] = min(completed[0] + 1, total - 1)
            indicator.update_progress(completed[0])

        timer.timeout.connect(_poll)
        timer.start()

    else:
        # Legacy path: introspect dask graph and poll scheduler_info
        try:
            graph = dict(future.__dask_graph__()) if hasattr(future, '__dask_graph__') else {}
            task_keys = list(graph.keys())
        except Exception:
            task_keys = []

        total = max(len(task_keys), 1)
        indicator.set_computing(total_tasks=total)

        timer = _QtCore.QTimer()
        timer.setInterval(200)
        timer_holder.append(timer)

        def _poll():
            if future.done():
                timer.stop()
                indicator.set_done()
                return
            if not task_keys:
                return
            try:
                info = client.scheduler_info()
                all_tasks = info.get("tasks", {})
                completed = sum(
                    1 for k in task_keys
                    if all_tasks.get(str(k), {}).get("state") in ("memory", "released", "forgotten")
                )
                indicator.update_progress(completed)
            except Exception:
                pass

        timer.timeout.connect(_poll)
        timer.start()


_CZB_BUILT_TOOLBARS: set = set()


def center_zero_beam_setup(
    toolbar: RoundedToolBar,
    action_name: str = "Center Zero Beam",
    *args,
    **kwargs,
):
    """
    Augment the YAML-created CaretParams with a Manual tab on first toggle.

    The CaretParams (with its RectangleSelector ROI) is created at PlotState
    init time from the YAML parameters.  This function, called on first trigger,
    adds a Manual tab to it and wires the map window show/hide logic.
    """
    tid = id(toolbar)
    if tid in _CZB_BUILT_TOOLBARS:
        return
    _CZB_BUILT_TOOLBARS.add(tid)

    caret_params = toolbar.action_widgets.get(action_name, {}).get("widget")
    if caret_params is None:
        return

    main_window = toolbar.plot.main_window
    manual_state = {"map_window": [None], "toolbar_ref": toolbar}

    def _on_czb_tab_changed(idx):
        mw = manual_state["map_window"][0]
        if idx == 1:  # Manual tab selected
            if mw is None:
                mw = _czb_create_map_window(toolbar, manual_state, main_window)
            mw.show()
            _czb_install_click_hook(toolbar, manual_state)
        else:
            if mw is not None:
                mw.hide()
            _czb_remove_click_hook(manual_state)

    manual_page = _czb_build_manual_page(caret_params, toolbar, manual_state, 220)
    caret_params.add_extra_tab("Manual", manual_page, on_tab_changed=_on_czb_tab_changed)

    # Store state on toolbar to prevent GC
    toolbar._czb_state = manual_state

    # Wire action toggle: hide map window when the action is closed
    action = toolbar._find_action(action_name)
    if action is not None:
        action.toggled.connect(
            lambda checked: _czb_on_action_toggled(checked, manual_state)
        )

    # Reposition after Qt has processed the resize from finalize_layout.
    pos_fn = toolbar.action_widgets.get(action_name, {}).get("position_fn")
    if pos_fn is not None:
        QtCore.QTimer.singleShot(0, pos_fn)


def _czb_on_action_toggled(checked: bool, manual_state: dict):
    """When the action is closed, hide the map window and remove the click hook."""
    if not checked:
        mw = manual_state["map_window"][0]
        if mw is not None:
            mw.hide()
        _czb_remove_click_hook(manual_state)


def _czb_create_map_window(toolbar, manual_state, main_window):
    """Create the floating X-shift / Y-shift map using a styled PlotWindow."""
    import pyqtgraph as _pg
    from spyde.drawing.plots.plot_window import PlotWindow
    from PySide6.QtCore import Qt as _Qt

    pw = PlotWindow(main_window=main_window)
    pw.setWindowTitle("Manual Beam Centers")
    side = main_window.screen_size.height() // 6  # half the default plot height
    pw.resize(side * 2, side)
    main_window.mdi_area.addSubWindow(pw)
    try:
        pw.setWindowFlags(pw.windowFlags() | _Qt.WindowType.FramelessWindowHint)
        pw.setStyleSheet("QMdiSubWindow { border: none; }")
    except Exception:
        pass

    glw = pw.plot_widget

    x_plot = glw.addPlot(row=0, col=0, title="X Shift (px)")
    x_plot.setAspectLocked(True)
    x_img = _pg.ImageItem(colorMap="CET-D1")
    x_plot.addItem(x_img)
    x_scatter = _pg.ScatterPlotItem(
        size=10, symbol="+",
        pen=_pg.mkPen("y", width=2), brush=_pg.mkBrush(None)
    )
    x_scatter.setZValue(10)
    x_plot.addItem(x_scatter)
    x_plot.getViewBox().setMenuEnabled(False)

    y_plot = glw.addPlot(row=0, col=1, title="Y Shift (px)")
    y_plot.setAspectLocked(True)
    y_img = _pg.ImageItem(colorMap="CET-D1")
    y_plot.addItem(y_img)
    y_scatter = _pg.ScatterPlotItem(
        size=10, symbol="+",
        pen=_pg.mkPen("y", width=2), brush=_pg.mkBrush(None)
    )
    y_scatter.setZValue(10)
    y_plot.addItem(y_scatter)
    y_plot.getViewBox().setMenuEnabled(False)

    x_scatter.sigClicked.connect(
        lambda sc, pts: _czb_on_scatter_clicked(pts, manual_state)
    )
    y_scatter.sigClicked.connect(
        lambda sc, pts: _czb_on_scatter_clicked(pts, manual_state)
    )

    manual_state["x_img"] = x_img
    manual_state["y_img"] = y_img
    manual_state["x_scatter"] = x_scatter
    manual_state["y_scatter"] = y_scatter

    # Install Delete key filter on the map window's graphics viewport
    class _MapKeyFilter(QtCore.QObject):
        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.Type.KeyPress:
                if event.key() == Qt.Key.Key_Delete:
                    fn = manual_state.get("delete_fn")
                    if fn:
                        fn()
                    return True
            return False

    key_filt = _MapKeyFilter(pw)
    # Install on the plot_widget viewport so it catches keys when the map has focus
    glw.viewport().installEventFilter(key_filt)
    manual_state["map_key_filter"] = key_filt

    pw.show()
    manual_state["map_window"][0] = pw
    # Draw any points already recorded before the map window was opened
    _czb_refresh(toolbar, manual_state)
    return pw


class _ShiftClickViewFilter(QtCore.QObject):
    """
    Event filter installed on the QGraphicsView that owns the signal plot.

    Intercepts Shift+Left click (MouseButtonPress with ShiftModifier) before
    pyqtgraph processes it, so ROIs and other items cannot consume it first.
    Returns True to consume the event so pyqtgraph does not also act on it.
    """

    def __init__(self, view, callback, parent=None):
        super().__init__(parent)
        self._view = view
        self._callback = callback

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if (event.button() == Qt.MouseButton.LeftButton and
                    event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                # event.pos() is in viewport coordinates; map to scene.
                scene_pos = self._view.mapToScene(event.position().toPoint())
                self._callback(scene_pos)
                return True  # consume — prevent pyqtgraph from panning/selecting
        return False


def _czb_install_click_hook(toolbar, manual_state):
    """Install viewport event filter for Shift+click and wire navigator-change crosshair."""
    _czb_remove_click_hook(manual_state)
    plot = toolbar.plot
    if plot is None:
        return

    scene = plot.vb.scene() if plot.vb is not None else None
    if scene is None:
        return
    views = scene.views()
    if not views:
        return
    view = views[0]
    viewport = view.viewport()

    def _on_shift_click(scene_pos):
        vb = plot.vb
        view_pos = vb.mapSceneToView(scene_pos)
        img_transform, ok = plot.image_item.transform().inverted()
        if not ok:
            return
        local_pos = img_transform.map(view_pos)
        cx_px = local_pos.x()
        cy_px = local_pos.y()

        nav_idx = _czb_get_nav_indices(plot)
        if nav_idx is None:
            return
        nav_x, nav_y = nav_idx

        # Replace existing point at this nav position, or append a new one.
        pts = manual_state.get("control_points", [])
        for i, (nx, ny, *_) in enumerate(pts):
            if nx == nav_x and ny == nav_y:
                pts[i] = (nav_x, nav_y, cx_px, cy_px)
                break
        else:
            pts.append((nav_x, nav_y, cx_px, cy_px))
        manual_state["control_points"] = pts
        manual_state["selected_idx"] = None
        _czb_update_signal_crosshair(plot, manual_state)
        _czb_refresh(toolbar, manual_state)

    filt = _ShiftClickViewFilter(view, _on_shift_click)
    viewport.installEventFilter(filt)
    manual_state["click_filter"] = filt
    manual_state["click_view"] = viewport

    # Wire axes_manager indices_changed — only once per signal.
    # _czb_remove_click_hook disconnects the old handler before we reconnect.
    signal = plot.plot_state.current_signal if plot.plot_state else None
    if signal is not None and manual_state.get("nav_signal") is not signal:
        def _on_nav_changed():
            _czb_update_signal_crosshair(plot, manual_state)
            current_nav = _czb_get_nav_indices(plot)
            _czb_refresh_scatter(manual_state, current_nav)

        signal.axes_manager.events.indices_changed.connect(_on_nav_changed, kwargs=[])
        manual_state["nav_handler"] = _on_nav_changed
        manual_state["nav_signal"] = signal

    # Show crosshair for current position immediately
    _czb_update_signal_crosshair(plot, manual_state)


def _czb_remove_click_hook(manual_state):
    filt = manual_state.get("click_filter")
    viewport = manual_state.get("click_view")
    if filt is not None and viewport is not None:
        try:
            viewport.removeEventFilter(filt)
        except Exception:
            pass
    manual_state["click_filter"] = None
    manual_state["click_view"] = None

    # Disconnect navigator change handler
    handler = manual_state.get("nav_handler")
    nav_signal = manual_state.get("nav_signal")
    if handler is not None and nav_signal is not None:
        try:
            nav_signal.axes_manager.events.indices_changed.disconnect(handler)
        except Exception:
            pass
    manual_state["nav_handler"] = None
    manual_state["nav_signal"] = None

    # Remove signal-plot crosshair
    _czb_remove_signal_crosshair(manual_state)


def _czb_update_signal_crosshair(plot, manual_state):
    """
    Remove any existing crosshair lines, then place fresh draggable InfiniteLine
    crosshairs if the current navigator position has a recorded control point.

    Two InfiniteLine objects (H + V) are used instead of CrosshairROI because
    InfiniteLine is always visible regardless of zoom level and is always centered
    on the specified position.
    """
    import pyqtgraph as _pg

    _czb_remove_signal_crosshair(manual_state)

    nav_idx = _czb_get_nav_indices(plot)
    if nav_idx is None:
        return

    nav_x, nav_y = nav_idx

    cx_px, cy_px = None, None
    for nx, ny, cx, cy in manual_state.get("control_points", []):
        if nx == nav_x and ny == nav_y:
            cx_px, cy_px = cx, cy
            break

    if cx_px is None:
        return

    # Map image pixel coords → scene/data coords via image_item transform.
    img_transform = plot.image_item.transform()
    data_pos = img_transform.map(QtCore.QPointF(cx_px, cy_px))
    dx, dy = data_pos.x(), data_pos.y()

    pen = _pg.mkPen("y", width=2)
    h_line = _pg.InfiniteLine(pos=dy, angle=0, pen=pen, movable=True)
    v_line = _pg.InfiniteLine(pos=dx, angle=90, pen=pen, movable=True)
    plot.addItem(h_line)
    plot.addItem(v_line)
    manual_state["signal_crosshair"] = (h_line, v_line)

    bound_nav_x, bound_nav_y = nav_x, nav_y

    def _on_line_moved():
        if manual_state.get("signal_crosshair") is not (h_line, v_line):
            return
        # Current intersection in data coords
        cur_dx = v_line.value()
        cur_dy = h_line.value()
        inv, ok = plot.image_item.transform().inverted()
        if not ok:
            return
        px_pos = inv.map(QtCore.QPointF(cur_dx, cur_dy))
        pts = manual_state.get("control_points", [])
        for i, (nx, ny, *_) in enumerate(pts):
            if nx == bound_nav_x and ny == bound_nav_y:
                pts[i] = (nx, ny, px_pos.x(), px_pos.y())
                break
        _czb_refresh(manual_state["toolbar_ref"], manual_state)

    h_line.sigPositionChanged.connect(_on_line_moved)
    v_line.sigPositionChanged.connect(_on_line_moved)


def _czb_remove_signal_crosshair(manual_state):
    crosshair = manual_state.get("signal_crosshair")
    plot = manual_state.get("toolbar_ref").plot if manual_state.get("toolbar_ref") else None
    if crosshair is not None and plot is not None:
        h_line, v_line = crosshair
        for line in (h_line, v_line):
            try:
                plot.removeItem(line)
            except Exception:
                pass
    manual_state["signal_crosshair"] = None


def _czb_get_nav_indices(plot) -> Optional[tuple]:
    """Return (nav_x_idx, nav_y_idx) from the navigator crosshair selector.

    axes_manager.indices is NOT updated by navigator crosshair movement in SpyDE —
    the selector owns the current position. We read it from the signal_tree's
    navigator_plot_manager selectors.
    """
    if plot is None:
        return None
    signal_tree = getattr(plot, "signal_tree", None)
    if signal_tree is None:
        return None
    npm = getattr(signal_tree, "navigator_plot_manager", None)
    if npm is None:
        return None
    selectors = npm.all_navigation_selectors
    if not selectors:
        return None

    def _read_indices(sel):
        """Extract current_indices from a selector, handling IntegratingSelector2D."""
        # IntegratingSelector2D wraps a CrosshairSelector
        inner = getattr(sel, "_crosshair_selector", None)
        if inner is not None:
            return getattr(inner, "current_indices", None)
        return getattr(sel, "current_indices", None)

    for sel in reversed(selectors):
        indices = _read_indices(sel)
        if indices is None:
            continue
        flat = np.asarray(indices).flatten()
        if len(flat) >= 2:
            return (int(flat[0]), int(flat[1]))
        if len(flat) == 1:
            return (int(flat[0]), 0)
    return None


def _czb_on_scatter_clicked(points, manual_state):
    if not points:
        manual_state["selected_idx"] = None
    else:
        pos = points[0].pos()
        px, py = pos.x(), pos.y()
        for i, (nx, ny, *_) in enumerate(manual_state.get("control_points", [])):
            if abs(nx - px) < 0.5 and abs(ny - py) < 0.5:
                manual_state["selected_idx"] = i
                break
    _czb_refresh_scatter(manual_state)


def _czb_fit_plane(control_points, nav_shape):
    nav_y_size, nav_x_size = nav_shape
    ys, xs = np.mgrid[0:nav_y_size, 0:nav_x_size]
    if not control_points:
        return np.zeros(nav_shape), np.zeros(nav_shape)
    pts = np.array(control_points)
    nx, ny, cx, cy = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    A = np.column_stack([nx, ny, np.ones(len(nx))])
    ax, *_ = np.linalg.lstsq(A, cx, rcond=None)
    ay, *_ = np.linalg.lstsq(A, cy, rcond=None)
    x_plane = ax[0] * xs + ax[1] * ys + ax[2]
    y_plane = ay[0] * xs + ay[1] * ys + ay[2]
    return x_plane, y_plane


def _czb_refresh(toolbar, manual_state):
    count_lbl = manual_state.get("count_label")
    if count_lbl is not None:
        count_lbl.setText(f"Points: {len(manual_state.get('control_points', []))}")

    x_img = manual_state.get("x_img")
    y_img = manual_state.get("y_img")
    if x_img is None or y_img is None:
        return

    plot = toolbar.plot
    if plot is None or plot.plot_state is None:
        return
    signal = plot.plot_state.current_signal
    if signal is None:
        return
    nav_axes = signal.axes_manager.navigation_axes
    if len(nav_axes) < 2:
        return
    nav_shape = (nav_axes[1].size, nav_axes[0].size)
    x_plane, y_plane = _czb_fit_plane(manual_state.get("control_points", []), nav_shape)
    x_img.setImage(x_plane.T, autoLevels=True)
    y_img.setImage(y_plane.T, autoLevels=True)
    current_nav = _czb_get_nav_indices(plot)
    _czb_refresh_scatter(manual_state, current_nav)


def _czb_refresh_scatter(manual_state, current_nav=None):
    """Redraw scatter dots on both maps. Yellow = control point, cyan = current nav position."""
    x_scatter = manual_state.get("x_scatter")
    y_scatter = manual_state.get("y_scatter")
    if x_scatter is None:
        return
    pts = manual_state.get("control_points", [])
    if not pts:
        x_scatter.setData([], [])
        y_scatter.setData([], [])
        return
    import pyqtgraph as _pg
    arr = np.array(pts)
    nx, ny = arr[:, 0], arr[:, 1]
    brushes = []
    for i, (px, py, *_) in enumerate(pts):
        if current_nav is not None and px == current_nav[0] and py == current_nav[1]:
            brushes.append(_pg.mkBrush("c"))   # cyan = current position
        else:
            brushes.append(_pg.mkBrush("y"))   # yellow = other control point
    x_scatter.setData(x=nx, y=ny, brush=brushes)
    y_scatter.setData(x=nx, y=ny, brush=brushes)


def _czb_build_manual_page(caret, toolbar, manual_state, width):
    """Build the Manual tab page widget."""
    manual_state.setdefault("control_points", [])
    manual_state.setdefault("selected_idx", None)

    page = QtWidgets.QWidget(caret)
    page.setFixedWidth(width)
    vlay = QtWidgets.QVBoxLayout(page)
    vlay.setContentsMargins(4, 4, 4, 4)
    vlay.setSpacing(4)

    info = QtWidgets.QLabel(
        "Shift+click on the diffraction pattern\n"
        "to record the beam center at the current\n"
        "navigator position.\n"
        "Click a point on the map, then click\n"
        "'Delete Selected' to remove it."
    )
    info.setStyleSheet("color: white; font-size: 10px;")
    info.setWordWrap(True)
    vlay.addWidget(info)

    count_lbl = QtWidgets.QLabel("Points: 0")
    count_lbl.setStyleSheet("color: white; font-size: 10px;")
    vlay.addWidget(count_lbl)
    manual_state["count_label"] = count_lbl

    def _delete_selected():
        idx = manual_state.get("selected_idx")
        pts = manual_state.get("control_points", [])
        if idx is not None and 0 <= idx < len(pts):
            pts.pop(idx)
            manual_state["selected_idx"] = None
            _czb_refresh(toolbar, manual_state)

    from spyde.qt.style import make_button as _czb_btn
    delete_btn = _czb_btn("Delete Selected")
    delete_btn.clicked.connect(_delete_selected)
    vlay.addWidget(delete_btn)

    clear_btn = _czb_btn("Clear All Points")

    def _clear():
        manual_state["control_points"] = []
        manual_state["selected_idx"] = None
        _czb_refresh(toolbar, manual_state)

    clear_btn.clicked.connect(_clear)
    vlay.addWidget(clear_btn)

    submit_btn = _czb_btn("Submit")
    submit_btn.clicked.connect(lambda: _czb_submit(toolbar, manual_state))
    vlay.addWidget(submit_btn)

    # Store delete function so it can also be triggered from a key filter on the map viewport
    manual_state["delete_fn"] = _delete_selected

    return page


def _czb_submit(toolbar, manual_state):
    pts = manual_state.get("control_points", [])
    if not pts:
        print("No control points defined; nothing to apply.")
        return
    import hyperspy.api as hs
    signal = toolbar.plot.plot_state.current_signal
    if signal is None:
        return
    signal.set_signal_type("electron_diffraction")
    nav_axes = signal.axes_manager.navigation_axes
    nav_shape = (nav_axes[1].size, nav_axes[0].size)
    x_plane, y_plane = _czb_fit_plane(pts, nav_shape)
    data = np.stack([x_plane, y_plane], axis=-1)
    shifts = hs.signals.Signal1D(data)
    for i, ax in enumerate(nav_axes):
        out_ax = shifts.axes_manager.navigation_axes[i]
        out_ax.scale = ax.scale
        out_ax.offset = ax.offset
        out_ax.units = ax.units
        out_ax.name = ax.name
    new_signal = toolbar.plot.signal_tree.add_transformation(
        parent_signal=signal,
        node_name="Centered (Manual)",
        method="center_direct_beam",
        shifts=shifts,
        inplace=False,
    )
    new_signal.calibration.center = None
    toolbar.plot.set_plot_state(new_signal)


def center_zero_beam(
    toolbar: RoundedToolBar,
    make_flat_field: bool = False,
    method: str = "com",
    signal_slice: Tuple[int, int, int, int] = None,
    action_name: str = "Center zero-beam",
    *args,
    **kwargs,
):
    """
    Center the zero-beam of a 4D STEM dataset by a couple of different methods.

    Parameters
    ----------
    toolbar : spyde.plugins.toolbar.Toolbar
        The toolbar instance from which to get the current signal.
    selector : spyde.plugins.selector.Selector
        The selector instance from which to get the current signal.

    """

    print("Centering zero-beam...")
    print("arguments:", make_flat_field, method)
    print("kwargs", kwargs)
    print("args", args)

    signal = toolbar.plot.plot_state.current_signal
    if signal is None:
        print("No signal selected.")
        return

    signal.set_signal_type("electron_diffraction")

    sl = (
        signal_slice[0],
        signal_slice[0] + signal_slice[2],
        signal_slice[1],
        signal_slice[1] + signal_slice[3],
    )

    shifts = signal.get_direct_beam_position(method=method, signal_slice=sl, **kwargs)

    print(make_flat_field)
    if make_flat_field:
        if shifts._lazy:
            shifts.compute()
        shifts.get_linear_plane()

    new_signal = toolbar.plot.signal_tree.add_transformation(
        parent_signal=signal,
        node_name="Centered",
        method="center_direct_beam",
        shifts=shifts,
        inplace=False,
    )
    new_signal.calibration.center = None
    toolbar.plot.set_plot_state(new_signal)




def virtual_imaging(*args, **kwargs):
    print("Virtual imaging action triggered.")


def add_virtual_image(
    toolbar: RoundedToolBar, action_name: str = "Add Virtual Image", *args, **kwargs
):
    """
    Add a virtual image from a 4D STEM dataset by integrating over a specified region.

    Parameters
    ----------
    toolbar : spyde.plugins.toolbar.Toolbar
        The toolbar instance from which to get the current signal.

    """
    colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
    print("Adding virtual image...")

    icon_path = resolve_icon_path("drawing/toolbars/icons/virtual_imaging.svg")
    num = toolbar.num_actions()
    color = colors[num % len(colors)]

    base_icon = QIcon(icon_path)
    icon_size = toolbar.iconSize()
    dpr = getattr(toolbar, "devicePixelRatioF", lambda: 1.0)()
    req_w = max(1, int(icon_size.width() * dpr))
    req_h = max(1, int(icon_size.height() * dpr))
    base_pixmap = base_icon.pixmap(req_w, req_h)

    colored_pixmap = QPixmap(base_pixmap.size())
    colored_pixmap.setDevicePixelRatio(dpr)
    colored_pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(colored_pixmap)
    painter.drawPixmap(0, 0, base_pixmap)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(colored_pixmap.rect(), QColor(color))
    painter.end()

    pen = mkPen(color=color, width=6)
    icon = QIcon()
    icon.addPixmap(colored_pixmap)

    action_name = f"Virtual Image ({color})"

    # Per-VI mutable state.
    # Live recompute stays OFF: launching a large dask graph on every ROI move
    # makes the app appear frozen — the user computes on demand via the
    # title-bar Compute button instead.
    _live_enabled = [False]
    _cached_mask = [None]
    _cached_roi = [None]
    _timer_holder = []  # progress poll timer
    _generation = [0]

    def _on_compute_clicked():
        # Title-bar Compute button fires this VI regardless of live flag.
        # _compute_one is defined below; this defers the lookup so it works.
        _compute_one(action_name)

    params = {
        "type": {
            "name": "Detector Type",
            "type": "enum",
            "default": "disk",
            "options": ["annular", "disk", "rectangle", "multiple_disks"],
        },
        "calculation": {
            "name": "Calculation",
            "type": "enum",
            "default": "mean",
            "options": ["mean", "FEM Omega", "COM"],
        },
    }

    action, params_caret_box = toolbar.add_action(
        name=action_name,
        icon_path=icon,
        function=compute_virtual_image,
        toggle=True,
        parameters=params,
    )

    # Compute/Commit live on the preview window's title bar; the caret only
    # holds detector parameters, so its Submit button is redundant.
    try:
        params_caret_box.submit_button.hide()
        if hasattr(params_caret_box, "finalize_layout"):
            params_caret_box.finalize_layout()
    except Exception:
        pass

    type_widget = params_caret_box.kwargs["type"]

    # Get signal/client/GPU info
    plot = toolbar.parent_toolbar.plot
    signal = plot.plot_state.current_signal
    main_window = plot.main_window
    client = main_window.dask_manager.client
    gpu_worker = main_window.dask_manager.gpu_worker_address

    from spyde.qt.compute_status_indicator import ComputeStatusIndicator

    virtual_plot_window = main_window.add_plot_window(
        is_navigator=False,
        signal_tree=plot.signal_tree,
    )
    virtual_plot_window.owner_plot_window = plot.plot_window
    main_window._auto_position_near_owner(virtual_plot_window)
    virtual_plot = virtual_plot_window.add_new_plot()
    if virtual_plot.image_item not in virtual_plot.items:
        virtual_plot.addItem(virtual_plot.image_item)

    indicator = ComputeStatusIndicator(color=color)
    virtual_plot_window.set_compute_indicator(indicator)
    # Compute sits beside Commit on the preview window's title bar.
    virtual_plot_window.set_compute_fn(_on_compute_clicked)

    toolbar.parent_toolbar.register_action_plot_window(
        action_name="Virtual Imaging",
        plot_window=virtual_plot_window,
        key=action_name,
    )

    # ── Shared batch-recompute timer ────────────────────────────────────────
    # All VIs on this parent toolbar share a single 150 ms debounce timer stored
    # in action_widgets["Virtual Imaging"]["_batch_timer"].  Any ROI move restarts
    # the timer; when it fires one dask submit covers all active VIs together.
    vi_entry = toolbar.parent_toolbar.action_widgets.setdefault("Virtual Imaging", {})
    if "_batch_timer" not in vi_entry:
        from PySide6.QtCore import QTimer as _QTimer
        # Parent the timer to the toolbar so Qt manages its lifetime on the GUI
        # thread and it is never destroyed from a Dask worker thread.
        _batch_timer = _QTimer(toolbar.parent_toolbar)
        _batch_timer.setInterval(150)
        _batch_timer.setSingleShot(True)
        vi_entry["_batch_timer"] = _batch_timer
        vi_entry["_vi_registry"] = {}  # action_name → state dict
    batch_timer: "QTimer" = vi_entry["_batch_timer"]
    vi_registry: dict = vi_entry["_vi_registry"]

    # Register this VI's mutable state so the batch callback can reach it
    vi_registry[action_name] = {
        "roi_ref": [None],       # holds the current roi object (updated on type change)
        "virtual_plot": virtual_plot,
        "virtual_plot_window": virtual_plot_window,
        "indicator": indicator,
        "cached_mask": _cached_mask,
        "cached_roi": _cached_roi,
        "generation": _generation,
        "timer_holder": _timer_holder,
        "signal": signal,
        "live_enabled": _live_enabled,
    }

    def _compute_one(name: str):
        """Compute a single VI: writes each nav chunk into shared memory as it finishes."""
        from spyde.drawing.update_functions import (
            compute_with_live_buffer, ensure_live_buffer, read_live_buffer,
            compute_virtual_image_kernel,
        )
        from PySide6 import QtCore as _QtCore
        entry = vi_registry.get(name)
        if entry is None:
            return
        current_roi = entry["roi_ref"][0]
        if current_roi is None:
            return
        try:
            mask = roi_to_mask(current_roi, entry["signal"])
        except Exception:
            return
        entry["cached_mask"][0] = mask
        entry["cached_roi"][0] = current_roi
        entry["generation"][0] += 1
        my_gen = entry["generation"][0]
        vp = entry["virtual_plot"]
        vpw = entry["virtual_plot_window"]
        ind = entry["indicator"]
        th = entry["timer_holder"]
        sig = entry["signal"]

        nav_shape = tuple(sig.axes_manager.navigation_shape[::-1])
        shm_name = f"spyde_vi_{id(vp)}"

        # Create/reset shared buffer; keep reference so it isn't GC'd
        shm = ensure_live_buffer(nav_shape, shm_name)
        entry["_shm"] = shm  # keep alive

        # Show NaN-filled image immediately so the plot has correct extent
        vp.current_data = np.full(nav_shape, np.nan, dtype=np.float32)
        vp.needs_auto_level = True
        vp.update()

        th.clear()
        vpw.set_commit_enabled(False)
        ind.set_computing()

        # Build the nav-reduced lazy array
        import dask as _dask
        data = sig.data
        if gpu_worker:
            with _dask.annotate(resources={"GPU": 1}):
                result_lazy = (data * mask).sum(axis=(-2, -1))
        else:
            result_lazy = (data * mask).sum(axis=(-2, -1))

        # Relay: chunk results arrive in Dask callback threads → emit to GUI thread
        # so the GUI thread writes them into shm (which it owns).
        class _VIChunkRelay(_QtCore.QObject):
            chunk_ready = _QtCore.Signal(object, object)

        old_relay = entry.get("_vi_relay")
        if old_relay is not None:
            try:
                old_relay.chunk_ready.disconnect()
            except Exception:
                pass
        vi_relay = _VIChunkRelay(toolbar.parent_toolbar)
        entry["_vi_relay"] = vi_relay

        def _gui_write_chunk(chunk_result, nav_slices,
                             _gen=my_gen, _shm=shm, _shape=nav_shape):
            if entry["generation"][0] != _gen:
                return
            try:
                buf = np.ndarray(_shape, dtype=np.float32, buffer=_shm.buf)
                buf[nav_slices] = chunk_result.astype(np.float32)
            except Exception:
                pass

        vi_relay.chunk_ready.connect(_gui_write_chunk)

        def _on_chunk_vi(chunk_result, nav_slices, _gen=my_gen):
            if entry["generation"][0] != _gen:
                return
            vi_relay.chunk_ready.emit(chunk_result, nav_slices)

        future = compute_with_live_buffer(result_lazy, nav_shape, client, shm_name,
                                          on_chunk_done=_on_chunk_vi)

        # Set current_data to the Future so PlotUpdateWorker tracks it and fires
        # on_plot_future_ready when done (existing contract for tests).
        vp.current_data = future
        vp._progressive_future = future
        vp.needs_auto_level = True  # auto-level on first real data

        # Poll shared buffer every 100 ms to show intermediate progress
        poll_timer = _QtCore.QTimer(toolbar.parent_toolbar)
        poll_timer.setInterval(100)
        th.append(poll_timer)

        _vi_levels = [None]

        def _poll_buffer(_gen=my_gen):
            if entry["generation"][0] != _gen:
                poll_timer.stop()
                return
            if future.done():
                poll_timer.stop()
                return
            arr = read_live_buffer(nav_shape, shm_name)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return
            if _vi_levels[0] is None:
                lo, hi = float(finite.min()), float(finite.max())
                _vi_levels[0] = (lo, hi if hi > lo else lo + 1)
            else:
                lo, hi = float(finite.min()), float(finite.max())
                if hi > _vi_levels[0][1]:
                    _vi_levels[0] = (_vi_levels[0][0], hi)
            vp.image_item.setImage(arr, autoLevels=False, levels=_vi_levels[0])

        poll_timer.timeout.connect(_poll_buffer)
        poll_timer.start()

        # Count nav chunks for the progress indicator
        import math as _math
        n_chunks = _math.prod(len(c) for c in result_lazy.chunks)
        _start_progress_poll(future, ind, client, th, n_chunks=n_chunks)

        # Stop button cancels the future and the poll timer
        def _stop_compute():
            entry["generation"][0] += 1  # invalidate current gen
            poll_timer.stop()
            try:
                client.cancel(future)
            except Exception:
                pass
            vpw.hide_stop_button()
            vpw.set_commit_enabled(False)
            ind.set_done()

        vpw.set_stop_fn(_stop_compute)

        def _on_preview_done(fut, _gen=my_gen, _entry=entry, _vpw=vpw):
            if _entry["generation"][0] != _gen:
                return
            _QtCore.QMetaObject.invokeMethod(
                _vpw, "hide_stop_button",
                _QtCore.Qt.ConnectionType.QueuedConnection,
            )
            _QtCore.QMetaObject.invokeMethod(
                _vpw, "set_commit_enabled",
                _QtCore.Qt.ConnectionType.QueuedConnection,
                _QtCore.Q_ARG(bool, True),
            )

        future.add_done_callback(_on_preview_done)

    def _batch_recompute_all():
        """Recompute every registered VI whose live flag is on, in one pass."""
        for name, entry in list(vi_registry.items()):
            if not entry["live_enabled"][0]:
                continue
            _compute_one(name)

    batch_timer.timeout.connect(_batch_recompute_all)

    def _schedule_batch_recompute(force: bool = False):
        """Restart the shared debounce timer (or fire immediately if force=True)."""
        if force:
            batch_timer.stop()
            _batch_recompute_all()
        else:
            batch_timer.start()  # restart debounce window

    def _cleanup():
        """Remove ROI from plot, de-register from batch registry, remove action."""
        entry = vi_registry.pop(action_name, None)
        # Wait for any in-flight shm write to complete before dropping the
        # shm reference — if the Dask worker is mid-write when the shm is
        # GC'd, Windows raises an access violation in the worker subprocess.
        if entry is not None:
            fut = entry.get("virtual_plot") and getattr(
                entry.get("virtual_plot"), "_progressive_future", None
            )
            if fut is not None and not fut.done():
                import time as _time
                deadline = _time.monotonic() + 1.0
                while not fut.done() and _time.monotonic() < deadline:
                    _time.sleep(0.02)
            # Explicitly close the shm before GC so we control the timing
            shm = entry.get("_shm")
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass
                entry["_shm"] = None
        try:
            toolbar.parent_toolbar.unregister_action_plot_item(
                action_name="Virtual Imaging", key=action_name
            )
        except Exception:
            pass
        try:
            toolbar.remove_action(action_name)
        except Exception:
            pass

    _orig_close = virtual_plot_window.close_window
    def _close_with_cleanup():
        _cleanup()
        _orig_close()
    virtual_plot_window.close_window = _close_with_cleanup

    # Build the initial ROI
    center, inner_rad, outer_rad = plot.get_annular_roi_parameters()
    if params_caret_box.kwargs["type"].currentText() == "annular":
        roi = RingROI(center=center, inner_rad=inner_rad, outer_rad=outer_rad, pen=pen)
    elif params_caret_box.kwargs["type"].currentText() == "disk":
        roi = CircleROI(center, inner_rad, pen=pen)
    else:
        roi = RectROI(center, inner_rad, pen=pen)

    vi_registry[action_name]["roi_ref"][0] = roi

    toolbar.parent_toolbar.register_action_plot_item(
        action_name="Virtual Imaging", item=roi, key=action_name
    )

    def arrange_widgets_on_move():
        rois = list(
            toolbar.parent_toolbar.action_widgets["Virtual Imaging"].get("plot_items", {}).values()
        )
        sizes = [r.size().x() for r in rois]
        sorted_index = np.argsort(sizes)
        for i, idx in enumerate(sorted_index[::-1]):
            r = rois[idx]
            r.setZValue(10 + i)

    def _on_roi_finished(_roi=None):
        arrange_widgets_on_move()
        _schedule_batch_recompute()

    roi.sigRegionChangeFinished.connect(_on_roi_finished)

    def _do_commit():
        if _cached_mask[0] is None:
            return
        from spyde.drawing.update_functions import compute_virtual_image_kernel
        from pyxem.signals import VirtualDarkFieldImage
        from PySide6 import QtCore as _QtCore

        virtual_plot_window.set_commit_enabled(False)
        indicator.set_computing()
        future = compute_virtual_image_kernel(signal.data, _cached_mask[0], client, gpu_worker)

        def _on_done(fut):
            try:
                result = fut.result()
            except Exception as e:
                print(f"Commit failed: {e}")
                _QtCore.QMetaObject.invokeMethod(
                    virtual_plot_window, "set_commit_enabled",
                    _QtCore.Qt.ConnectionType.QueuedConnection,
                    _QtCore.Q_ARG(bool, True),
                )
                return
            vdf = VirtualDarkFieldImage(result)
            nav_axes = list(signal.axes_manager.navigation_axes)
            sig_axes = list(vdf.axes_manager.signal_axes)
            for i, ax in enumerate(nav_axes):
                if i < len(sig_axes):
                    sig_axes[i].scale = ax.scale
                    sig_axes[i].offset = ax.offset
                    sig_axes[i].units = ax.units
                    sig_axes[i].name = ax.name
            vdf.metadata.Signal.virtual_detector = _roi_metadata(_cached_roi[0] or roi)
            main_window._pending_signal_queue.append(vdf)
            _QtCore.QMetaObject.invokeMethod(
                main_window, "_flush_pending_signals",
                _QtCore.Qt.ConnectionType.QueuedConnection,
            )
            _QtCore.QMetaObject.invokeMethod(
                virtual_plot_window, "set_commit_enabled",
                _QtCore.Qt.ConnectionType.QueuedConnection,
                _QtCore.Q_ARG(bool, True),
            )

        future.add_done_callback(_on_done)

    virtual_plot_window.set_commit_fn(_do_commit)

    def on_type_change(new_type: str) -> None:
        old_roi = toolbar.parent_toolbar.unregister_action_plot_item(
            action_name="Virtual Imaging", key=action_name
        )
        pos = old_roi.pos()
        size = old_roi.size()
        inner_r = min(size) / 2.0
        outer_r = inner_r * 2.0
        nonlocal roi
        if new_type == "annular":
            roi = RingROI(center=pos, inner_rad=inner_r, outer_rad=outer_r, pen=pen)
        elif new_type == "disk":
            roi = CircleROI(pos=pos, size=size, pen=pen)
        else:
            roi = RectROI(pos=pos, size=size, pen=pen)
        vi_registry[action_name]["roi_ref"][0] = roi
        toolbar.parent_toolbar.register_action_plot_item(
            action_name="Virtual Imaging", item=roi, key=action_name
        )
        roi.sigRegionChangeFinished.connect(_on_roi_finished)

    if hasattr(type_widget, "currentTextChanged"):
        type_widget.currentTextChanged.connect(on_type_change)


def compute_virtual_image(
    toolbar: RoundedToolBar, action_name: str = "Compute Virtual Image", *args, **kwargs
):
    """
    Compute the virtual image from a 4D STEM dataset.

    Parameters
    ----------
    toolbar : spyde.plugins.toolbar.Toolbar
        The toolbar instance from which to get the current signal.

    """
    print("Computing virtual image...")
    pass


def _generate_library_from_phases(phases, accelerating_voltage, resolution,
                                   minimum_intensity, reciprocal_radius,
                                   max_excitation_error=0.1):
    """Generate a diffsims Simulation2D library from a list of orix Phase objects."""
    generator = SimulationGenerator(
        accelerating_voltage, minimum_intensity=minimum_intensity
    )
    rotations = [
        get_sample_reduced_fundamental(
            resolution=resolution, point_group=phase.point_group
        )
        for phase in phases
    ]
    sim = generator.calculate_diffraction2d(
        phases if len(phases) > 1 else phases[0],
        rotation=rotations if len(rotations) > 1 else rotations[0],
        max_excitation_error=max_excitation_error,
        reciprocal_radius=reciprocal_radius,
        with_direct_beam=False,
    )
    return sim


def _extract_orientation_outputs(orientation_map, nav_axes, n_phases=1):
    """
    Extract result signals from an OrientationMap.

    OrientationMap.data has shape (nav..., n_best, 4) with column_names
    ['index', 'correlation', 'rotation', 'factor'].  Best match = n_best index 0.

    Returns list of (BaseSignal, title_str) tuples.
    """
    import hyperspy.api as hs

    def _copy_nav_axes(sig, nav_axes):
        for i, ax in enumerate(nav_axes):
            if i < sig.axes_manager.navigation_dimension:
                out_ax = sig.axes_manager.navigation_axes[i]
                out_ax.scale = ax.scale
                out_ax.offset = ax.offset
                out_ax.units = ax.units
                out_ax.name = ax.name
        return sig

    results = []

    # Orientation map itself (contains IPF + crystal map info)
    results.append((orientation_map, "Orientation Map"))

    # Extract best-match correlation (column index 1) at n_best=0
    try:
        col_names = getattr(orientation_map, "column_names", [])
        corr_col = col_names.index("correlation") if "correlation" in col_names else 1
        corr_data = orientation_map.data[..., 0, corr_col]  # (nav_y, nav_x)
        corr = hs.signals.Signal2D(corr_data)
        _copy_nav_axes(corr, nav_axes)
        corr.metadata.General.title = "Correlation Score"
        results.append((corr, "Correlation Score"))
    except Exception as e:
        print(f"Could not extract Correlation Score: {e}")

    # Phase map: column index 0 ('index') for multi-phase
    if n_phases > 1:
        try:
            idx_col = col_names.index("index") if "index" in col_names else 0
            phase_data = orientation_map.data[..., 0, idx_col].astype(float)
            phase_map = hs.signals.Signal2D(phase_data)
            _copy_nav_axes(phase_map, nav_axes)
            phase_map.metadata.General.title = "Phase Map"
            results.append((phase_map, "Phase Map"))
        except Exception as e:
            print(f"Could not extract Phase Map: {e}")

    return results


def _filter_sim_by_radius(coords, intensities, max_radius):
    """Return coords and intensities for spots within max_radius."""
    r = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    mask = r <= max_radius
    return coords[mask], intensities[mask]


def _get_current_nav_indices(plot):
    """Return current navigation indices as a flat tuple of ints (HyperSpy axis order).

    Tries get_selected_indices() first (BaseSelector subclasses), then falls back to
    _get_selected_indices() (IntegratingSelectorMixin subclasses like IntegratingSelector2D).
    """
    selector = getattr(plot, "parent_selector", None)
    if selector is None:
        selector = getattr(getattr(plot, "plot_window", None), "parent_selector", None)
    if selector is not None:
        # Prefer the clipped method to keep indices in bounds.
        # IntegratingSelector2D only has _get_selected_indices (unclipped),
        # so clip manually using the signal's navigation shape.
        for method_name in ("get_selected_indices", "_get_selected_indices"):
            method = getattr(selector, method_name, None)
            if method is not None:
                try:
                    indices = np.asarray(method())
                    flat = np.mean(indices, axis=0).ravel()
                    # CrosshairSelector returns [[col, row]] (x, y in image space).
                    # Clip col to x-nav size and row to y-nav size.
                    nav_axes = plot.plot_state.current_signal.axes_manager.navigation_axes
                    # nav_axes[0]=x(col, innermost), nav_axes[1]=y(row, outermost)
                    clipped = tuple(
                        int(np.clip(flat[i], 0, nav_axes[i].size - 1))
                        for i in range(len(flat))
                    )
                    return clipped
                except Exception:
                    continue
    nav_axes = plot.plot_state.current_signal.axes_manager.navigation_axes
    return tuple(ax.size // 2 for ax in nav_axes)


def _update_refine_pattern(refine_plot, signal, nav_indices):
    """Load the diffraction pattern at nav_indices into refine_plot."""
    idx = tuple(int(i) for i in reversed(nav_indices))
    pattern_data = np.array(signal.data[idx])
    refine_plot.update_data(pattern_data)


def _build_matching_cache(signal, sim):
    """
    Pre-compute slices and templates that only depend on signal geometry and library.
    With numba cache=True in pyxem, get_slices2d is ~0.1 s and JIT compiles are
    skipped after the first run, so this is cheap after a fresh install.
    """
    from pyxem.utils.indexation_utils import _get_integrated_polar_templates, _norm_rows

    NR, NA = 100, 360

    slices, factors, factors_slice, radial_range = signal.calibration.get_slices2d(NR, NA)

    r0, r1 = float(radial_range[0]), float(radial_range[1])
    radial_axis = r0 + (r1 - r0) / NR * np.arange(NR)
    azim_axis   = np.linspace(-np.pi, np.pi, NA, endpoint=False)

    r_templates, theta_templates, intensities_templates = sim.polar_flatten_simulations(
        radial_axes=radial_axis, azimuthal_axes=azim_axis,
    )
    integrated = _get_integrated_polar_templates(NR, r_templates, intensities_templates, True)
    intensities_raw  = intensities_templates.copy().astype(float)
    intensities_norm = _norm_rows(intensities_raw.copy())

    return {
        "slices": slices,
        "factors": factors,
        "factors_slice": factors_slice,
        "r_templates": r_templates,
        "theta_templates": theta_templates,
        "intensities_norm": intensities_norm,
        "intensities_raw": intensities_raw,
        "integrated": integrated,
        "NR": NR, "NA": NA,
    }


def _ipf_xy_for_rotation(rotation, phases):
    """Return stereographic (x, y) of rotation*z projected into the fundamental sector."""
    from orix.vector import Vector3d
    from orix.projections import StereographicProjection
    # phases may be a single Phase or a list — always use first
    phase = phases[0] if hasattr(phases, '__len__') else phases
    vec = rotation * Vector3d.zvector()
    vec = vec.in_fundamental_sector(phase.point_group)
    s = StereographicProjection()
    x, y = s.vector2xy(vec)
    return float(np.atleast_1d(x)[0]), float(np.atleast_1d(y)[0])


def _ipf_triangle_xy(phase):
    """Return (xy_edges, label_xy, label_texts) for the IPF fundamental sector outline."""
    from orix.projections import StereographicProjection
    from pyxem.signals.indexation_results import _closed_edges_in_hemisphere, _get_ipf_axes_labels
    s = StereographicProjection()
    sector = phase.point_group.fundamental_sector
    edges = _closed_edges_in_hemisphere(sector.edges, sector)
    ex, ey = s.vector2xy(edges)
    xy_edges = np.vstack((ex, ey)).T
    try:
        raw_labels = _get_ipf_axes_labels(sector.vertices, symmetry=phase.point_group)
        # Strip LaTeX delimiters ($) — pyqtgraph can't render them
        labels = [l.replace("$", "") for l in raw_labels]
        lx, ly = s.vector2xy(sector.vertices)
        lx = np.atleast_1d(np.array(lx, dtype=float))
        ly = np.atleast_1d(np.array(ly, dtype=float))
        center_x = float(np.mean(lx))
        center_y = float(np.mean(ly))
        # Displace each label away from the triangle centroid
        DISPLACE = 0.25
        label_xy = np.vstack([
            lx + DISPLACE * (lx - center_x),
            ly + DISPLACE * (ly - center_y),
        ]).T
    except Exception:
        labels, label_xy = [], np.empty((0, 2))
    return xy_edges, label_xy, labels


def _get_best_fit_spots(signal, sim, nav_indices, gamma, max_radius, min_intensity=0.0,
                        scale_override=None, matching_cache=None, pattern_override=None,
                        normalize_templates=False, rot_mask=None):
    """
    Run orientation matching on a single diffraction pattern and return
    (coords_data, intensities, ipf_xy) for the best-match simulation spots.

    coords_data : ndarray shape (N, 2) in Å⁻¹ centered at the direct beam.
    intensities : ndarray shape (N,)
    ipf_xy      : (x, y) tuple in stereographic coords for the best-match rotation.

    matching_cache : dict from _build_matching_cache — holds pre-computed slices
        and templates so this function only does ~5 ms of work per call.
        If None, falls back to the slow HyperSpy path (~7 s per call).
    """
    original_scale = signal.axes_manager.signal_axes[0].scale
    from pyxem.utils.indexation_utils import _mixed_matching_lib_to_polar
    from pyxem.utils._azimuthal_integrations import _slice_radial_integrate

    if pattern_override is not None:
        pattern_data = np.asarray(pattern_override).astype(float)
    else:
        # nav_indices are (x, y) HyperSpy order (innermost first).
        # Reverse to (y, x) numpy order for data indexing.
        idx = tuple(int(i) for i in reversed(nav_indices))
        pattern_data = np.array(signal.data[idx]).astype(float)

    if matching_cache is not None:
        # Fast path: raw numpy, no HyperSpy Signal overhead
        slices        = matching_cache["slices"]
        factors       = matching_cache["factors"]
        factors_slice = matching_cache["factors_slice"]
        r_tmpl        = matching_cache["r_templates"]
        theta_tmpl    = matching_cache["theta_templates"]
        int_norm      = matching_cache["intensities_norm"]
        integrated    = matching_cache["integrated"]
        NR, NA        = matching_cache["NR"], matching_cache["NA"]

        polar = _slice_radial_integrate(
            pattern_data, factors, factors_slice, slices, NR, NA, mean=True
        )
        # Apply gamma, convert to (azim, radial) for _mixed_matching_lib_to_polar
        polar = np.nan_to_num(polar ** gamma).T.astype(float)  # (NA, NR)

        int_templates = int_norm if normalize_templates else matching_cache["intensities_raw"]

        # Apply rotation mask: subset templates to only those inside IPF mask circles
        if rot_mask is not None and rot_mask.any():
            mask_idx = np.where(rot_mask)[0]
            integrated_use    = integrated[mask_idx]
            r_tmpl_use        = r_tmpl[mask_idx]
            theta_tmpl_use    = theta_tmpl[mask_idx]
            int_templates_use = int_templates[mask_idx]
        else:
            mask_idx          = None
            integrated_use    = integrated
            r_tmpl_use        = r_tmpl
            theta_tmpl_use    = theta_tmpl
            int_templates_use = int_templates

        n_templates = integrated_use.shape[0]
        result = _mixed_matching_lib_to_polar(
            polar,
            integrated_templates=integrated_use,
            r_templates=r_tmpl_use,
            theta_templates=theta_tmpl_use,
            intensities_templates=int_templates_use,
            n_keep=None, frac_keep=1.0, n_best=n_templates, transpose=False,
        )
        # result shape (n_templates, 4): [library_index, correlation, rotation_idx, mirror]
        # lib_idx values are indices into the *subset* if mask_idx is set; remap to global.
        if mask_idx is not None:
            result[:, 0] = mask_idx[result[:, 0].astype(int)].astype(result.dtype)

        # Best match is result[0] (sorted descending by correlation).
        row = result[0]
        lib_idx = int(row[0])
        rot_idx = int(row[2])
        mirror  = float(row[3])

        # Get the raw simulation coordinates for the best-match entry
        _rot, _phase_idx, coords_dv = sim.get_simulation(lib_idx)
        raw_coords  = coords_dv.data[:, :2].copy().astype(float)
        intensities = np.array(coords_dv.intensity, dtype=float)

        # Replicate vectors_from_orientation_map (pyxem/signals/indexation_results.py):
        #   1. flip y, 2. rotate by mirror*angle, 3. negate, 4. mirror*y
        NA_full   = NA
        angle_deg = rot_idx / NA_full * 360.0 - 180.0
        angle_rad = np.deg2rad(mirror * angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rx = raw_coords[:, 0];  ry = -raw_coords[:, 1]
        kx = rx * cos_a - ry * sin_a;  ky = rx * sin_a + ry * cos_a
        kx, ky = -kx, -ky;  ky = mirror * ky
        raw_coords = np.stack([kx, ky], axis=1)

        # IPF heatmap: project every template rotation * z into the fundamental
        # sector, weighted by normalised correlation score.
        try:
            from orix.vector import Vector3d as _V3
            from orix.projections import StereographicProjection as _SP
            _phase = sim.phases if not hasattr(sim.phases, '__len__') else sim.phases
            if hasattr(_phase, '__len__'):
                _phase = _phase[0]
            _sp = _SP()
            # All rotations in the simulation library
            all_rots = sim.rotations if not hasattr(sim.rotations, '__len__') else sim.rotations
            if hasattr(all_rots, '__len__'):
                all_rots = all_rots[0] if len(all_rots) == 1 else all_rots
            vecs = all_rots * _V3.zvector()
            vecs_fs = vecs.in_fundamental_sector(_phase.point_group)
            ipf_xs, ipf_ys = _sp.vector2xy(vecs_fs)
            ipf_xs = np.atleast_1d(np.array(ipf_xs, dtype=float))
            ipf_ys = np.atleast_1d(np.array(ipf_ys, dtype=float))

            # Map result rows back to rotation indices (result[:,0] = lib_idx into sim)
            # Correlations are already sorted best-first; normalise to [0,1]
            all_cors = result[:, 1].astype(float)
            cor_max = float(all_cors[0]) if all_cors[0] > 0 else 1.0
            all_cors_norm = np.clip(all_cors / cor_max, 0.0, 1.0)
            all_lib_idxs = result[:, 0].astype(int)

            # Build per-rotation correlation array (some rotations may not appear
            # in result if frac_keep filtered them; leave those as 0)
            cor_per_rot = np.zeros(len(ipf_xs), dtype=float)
            for i, li in enumerate(all_lib_idxs):
                if 0 <= li < len(cor_per_rot):
                    cor_per_rot[li] = all_cors_norm[i]

            ipf_heatmap = (ipf_xs, ipf_ys, cor_per_rot)
            ipf_xy = _ipf_xy_for_rotation(_rot, sim.phases)
        except Exception as e:
            print(f"IPF heatmap failed: {e}")
            ipf_heatmap = None
            ipf_xy = (0.0, 0.0)

        # Apply scale override before radius filtering: the library was generated
        # assuming the signal's calibrated scale, but the real data may be
        # miscalibrated.  scale_override is the user's corrected Å⁻¹/px value,
        # so the template coords (in Å⁻¹) need to be rescaled by
        # scale_override / original_scale to match where spots actually land.
        if scale_override is not None:
            raw_coords = raw_coords * (scale_override / original_scale)

    else:
        # Slow fallback: full HyperSpy Signal path
        import hyperspy.api as hs
        pat = hs.signals.Signal2D(pattern_data)
        for i, ax in enumerate(signal.axes_manager.signal_axes):
            pat.axes_manager.signal_axes[i].scale = ax.scale
            pat.axes_manager.signal_axes[i].offset = ax.offset
            pat.axes_manager.signal_axes[i].units = ax.units
        pat.set_signal_type("electron_diffraction")
        polar_hs = pat.get_azimuthal_integral2d(npt=100, npt_azim=360, inplace=False, mean=True)
        polar_hs = polar_hs ** gamma
        orientation = polar_hs.get_orientation(sim)
        vectors_signal = orientation.to_vectors(n_best_index=0, return_object=False)
        spot_array = vectors_signal.data[()]
        if spot_array.ndim == 1 and spot_array.dtype == object:
            spot_array = spot_array[0]
        raw_coords  = spot_array[:, :2].astype(float)
        intensities = spot_array[:, 3].astype(float)

        if scale_override is not None:
            raw_coords = raw_coords * (scale_override / original_scale)

        try:
            best_rot = orientation.data[0, 0]
            from orix.quaternion import Rotation as _Rotation
            ipf_xy = _ipf_xy_for_rotation(_Rotation(best_rot), sim.phases)
        except Exception:
            ipf_xy = (0.0, 0.0)
        ipf_heatmap = None  # not available in slow path

    # Normalize intensities to [0, 1] relative to the brightest spot in the
    # full (pre-filter) simulation, then threshold by min_intensity percentage.
    i_max_all = float(np.max(intensities)) if len(intensities) > 0 else 1.0
    i_max_all = i_max_all if i_max_all > 0 else 1.0
    intensities_norm_display = intensities / i_max_all

    coords_filtered, intensities_filtered = _filter_sim_by_radius(
        raw_coords, intensities_norm_display, max_radius
    )

    if min_intensity > 0.0 and len(intensities_filtered) > 0:
        keep = intensities_filtered >= min_intensity
        coords_filtered      = coords_filtered[keep]
        intensities_filtered = intensities_filtered[keep]

    return coords_filtered, intensities_filtered, ipf_xy, ipf_heatmap


def _compute_reciprocal_radius(signal) -> float:
    """Derive max reciprocal radius from signal axes calibration."""
    sig_axes = signal.axes_manager.signal_axes
    half_extents = [ax.scale * ax.size / 2.0 for ax in sig_axes]
    return min(half_extents)


def _make_slider_row(parent, label_text, min_val, max_val, default, decimals=2, suffix=""):
    """Return (row_widget, spinbox, slider) — shared theme implementation."""
    from spyde.qt.style import make_slider_row
    return make_slider_row(parent, label_text, min_val, max_val, default,
                           decimals, suffix)


# Module-level set of toolbar ids that have already had the OM caret built.
_OM_BUILT_TOOLBARS: set = set()


def orientation_mapping(
    toolbar: RoundedToolBar,
    action_name: str = "Orientation Mapping",
    *args,
    **kwargs,
):
    """5-step wizard (tabbed) for template-matching orientation mapping of 4D-STEM data."""
    from PySide6 import QtWidgets as _QW, QtCore as _QC
    from spyde.drawing.toolbars.caret_group import CaretGroup, FileDropWidget

    # Guard: build the caret only once per toolbar instance.
    tid = id(toolbar)
    if tid in _OM_BUILT_TOOLBARS:
        return
    _OM_BUILT_TOOLBARS.add(tid)

    plot = toolbar.plot
    signal = plot.plot_state.current_signal
    main_window = plot.main_window
    sig_ax = signal.axes_manager.signal_axes
    sig_scale = sig_ax[0].scale  # Å⁻¹/px

    # ── State dict ─────────────────────────────────────────────────────────────
    state = {
        "plot": plot,
        "signal": signal,
        "main_window": main_window,
        "phases": [],
        "sim": [None],
        "gamma": [0.5],
        "min_intensity": [0.1],
        "scale_override": [None],
        "max_radius": [_compute_reciprocal_radius(signal)],
        "refit_timer": [None],
        "scatter_item": [None],
        "circle_roi": [None],
        "ipf_widget": [None],
        "run_status": [None],
        "run_btn": [None],
        "normalize_templates": [False],
        "ipf_mask_circles": [],   # list of [cx, cy, r] in IPF stereographic coords
        "active_step": [0],       # current wizard tab; IPF window only shows on 2 (Refine)
        "ipf_user_closed": [False],  # user pressed X on the IPF window; cleared on re-entering Refine
        "matching_cache": [None],  # pre-computed slices+templates from _build_matching_cache
        "refit_generation": [0],   # incremented on each schedule; threads skip stale runs
        "gen_relay": [None],       # kept alive so queued signal survives until GUI thread delivers it
    }

    # ── Build a CaretGroup directly and register it with the toolbar ───────────
    # We do NOT call toolbar.add_action again (that would create a second icon).
    # Instead we build the caret widget and register it via add_action_widget,
    # which wires the existing toggle action's toggled signal to show/hide it.
    toolbar._om_state = state

    caret = CaretGroup(
        title=action_name,
        toolbar=toolbar,
        action_name=action_name,
    )
    toolbar.add_action_widget(action_name, caret, None)
    # add_action_widget/_bind_action_to_widget calls setChecked(False) which hides
    # the caret. Defer restoring the checked state until after the full widget tree
    # is built and finalize_layout() has been called, so position_fn sees real geometry.

    layout = caret.layout()

    # ── Helper builders — shared theme factories (spyde/qt/style.py) ─────────
    W = 240
    from spyde.qt.style import (
        CHECKBOX_QSS as _chk_qss,
        make_label as _lbl,
        make_button as _btn,
        make_double_spin as _make_spin,
    )

    def _hrow(*widgets):
        w = _QW.QWidget()
        h = _QW.QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(4)
        for wi in widgets:
            if isinstance(wi, int):
                h.addStretch(wi)
            else:
                h.addWidget(wi)
        return w

    def _spin(parent, lo, hi, val, dec, suf=""):
        return _make_spin(parent, lo, hi, val, dec, suf)

    # ── CheckButtonGroup-style step selector ───────────────────────────────────
    def _on_om_tab_changed(idx):
        state["active_step"][0] = idx
        on_refine = idx == 2
        if on_refine:
            # Re-entering Refine forgets a previous X-press on the IPF window
            state["ipf_user_closed"][0] = False
        ipf_w = state["ipf_widget"][0]
        if ipf_w is not None:
            ipf_w.show() if on_refine else ipf_w.hide()
        sc = state["scatter_item"][0]
        if sc is not None:
            sc.setVisible(on_refine)
        roi = state["circle_roi"][0]
        if roi is not None:
            roi.setVisible(on_refine)

    step_bar, stack, _select_step = CaretGroup.make_tab_stack(
        ["1 Load", "2 Library", "3 Refine", "4 Run"],
        parent=caret,
        width=W,
        on_tab_changed=_on_om_tab_changed,
    )

    # ── Page 0: Load CIF ──────────────────────────────────────────────────────
    p0 = _QW.QWidget(); v0 = _QW.QVBoxLayout(p0); v0.setContentsMargins(4, 4, 4, 4); v0.setSpacing(4)
    cif_drop = FileDropWidget(extensions=[".cif"], parent=p0)
    phase_lbl = _lbl("Phases: (none loaded)", p0)
    voltage_s = _spin(p0, 60, 300, 200, 0, " kV")
    v0.addWidget(_lbl("CIF file(s):", p0))
    v0.addWidget(cif_drop)
    v0.addWidget(phase_lbl)
    v0.addWidget(_hrow(_lbl("Voltage:", p0), voltage_s))
    stack.addWidget(p0)

    # ── Page 1: Generate Library ──────────────────────────────────────────────
    p1 = _QW.QWidget(); v1 = _QW.QVBoxLayout(p1); v1.setContentsMargins(4, 4, 4, 4); v1.setSpacing(4)
    res_s = _spin(p1, 0.1, 10.0, 1.0, 1, "°")
    min_int_s = _spin(p1, 0.0, 1.0, 0.05, 3)
    gen_btn = _btn("Generate Library", p1, enabled=False)
    lib_lbl = _lbl("", p1)
    v1.addWidget(_hrow(_lbl("Angle density:", p1), res_s))
    v1.addWidget(_hrow(_lbl("Min intensity:", p1), min_int_s))
    v1.addWidget(gen_btn)
    v1.addWidget(lib_lbl)
    stack.addWidget(p1)

    # ── Page 2: Refine ────────────────────────────────────────────────────────
    p2 = _QW.QWidget(); v2 = _QW.QVBoxLayout(p2); v2.setContentsMargins(4, 4, 4, 4); v2.setSpacing(4)
    gamma_row, gamma_s, gamma_sl = _make_slider_row(p2, "Gamma", 0.1, 1.5, 1.0, decimals=2)
    # Min intensity as percentage of brightest spot (0–100%)
    min_i_row, min_i_s, min_i_sl = _make_slider_row(p2, "Min intensity", 0.0, 100.0, 10.0, decimals=1, suffix="%")
    # Scale: start at signal scale, allow ±10%
    sc_lo = round(sig_scale * 0.9, 6)
    sc_hi = round(sig_scale * 1.1, 6)
    sc_step_dec = max(2, -int(np.floor(np.log10(sig_scale * 0.01))) + 1) if sig_scale > 0 else 4
    scale_row, scale_s, scale_sl = _make_slider_row(p2, "Scale", sc_lo, sc_hi, sig_scale, decimals=sc_step_dec)
    norm_chk = _QW.QCheckBox("Normalize templates", p2)
    norm_chk.setChecked(False)
    norm_chk.setStyleSheet(_chk_qss)
    refine_lbl = _lbl("Generate library first.", p2)
    for r in [gamma_row, min_i_row, scale_row]:
        r.setEnabled(False)
    norm_chk.setEnabled(False)
    v2.addWidget(refine_lbl)
    v2.addWidget(gamma_row)
    v2.addWidget(min_i_row)
    v2.addWidget(scale_row)
    v2.addWidget(norm_chk)
    stack.addWidget(p2)

    # ── Page 3: Run (batch compute → orientation map + IPF window) ───────────
    p3 = _QW.QWidget(); v3 = _QW.QVBoxLayout(p3); v3.setContentsMargins(4, 4, 4, 4); v3.setSpacing(4)
    run_lbl = _lbl("", p3)
    nbest_s = _spin(p3, 1, 20, 5, 0)
    run_btn_w = _btn("Compute Map", p3, enabled=False)
    om_save_btn = _btn("Save Orientations…", p3, enabled=False)
    v3.addWidget(_lbl("Run full orientation mapping on the dataset.", p3))
    v3.addWidget(_hrow(_lbl("Best matches kept:", p3), nbest_s))
    v3.addWidget(run_btn_w)
    v3.addWidget(om_save_btn)
    v3.addWidget(run_lbl)
    stack.addWidget(p3)
    state["run_status"][0] = run_lbl
    state["run_btn"][0] = run_btn_w
    state["om_result"] = [None]

    layout.addWidget(step_bar)
    layout.addWidget(stack)
    caret.finalize_layout()
    _select_step(0)

    # Now that the widget tree is fully built, restore the checked state so the
    # caret shows with correct geometry on the first click.
    om_action = toolbar._find_action(action_name)
    if om_action is not None:
        # Scope the refine overlays (scatter, circle ROI, IPF window) to the
        # action: toggling the caret off must hide them, toggling back on
        # restores whatever the active wizard tab needs.
        def _on_om_action_toggled(checked):
            on_refine = checked and state["active_step"][0] == 2
            ipf_w = state["ipf_widget"][0]
            if ipf_w is not None:
                ipf_w.setVisible(on_refine and not state["ipf_user_closed"][0])
            sc = state["scatter_item"][0]
            if sc is not None:
                sc.setVisible(on_refine)
            roi = state["circle_roi"][0]
            if roi is not None:
                roi.setVisible(on_refine)

        om_action.toggled.connect(_on_om_action_toggled)
        om_action.setChecked(True)
    pos_fn = toolbar.action_widgets.get(action_name, {}).get("position_fn")
    if pos_fn is not None:
        pos_fn()

    # ── Wire callbacks ─────────────────────────────────────────────────────────

    def _on_cif_loaded(files):
        from orix.crystal_map import Phase
        state["phases"].clear()
        for path in files:
            try:
                phase = Phase.from_cif(path)
                state["phases"].append(phase)
            except Exception as e:
                print(f"Failed to load CIF {path}: {e}")
        if state["phases"]:
            phase_lbl.setText("Phases: " + ", ".join(p.name for p in state["phases"]))
            gen_btn.setEnabled(True)
        else:
            phase_lbl.setText("Phases: (none loaded)")
            gen_btn.setEnabled(False)

    cif_drop.filesChanged.connect(_on_cif_loaded)

    def _on_generate():
        if not state["phases"]:
            return
        gen_btn.setEnabled(False)
        lib_lbl.setText("Generating library…")

        from PySide6 import QtCore as _QC2
        class _GenRelay(_QC2.QObject):
            done = _QC2.Signal()
        _gen_relay = _GenRelay()
        state["gen_relay"][0] = _gen_relay  # prevent GC before queued signal is delivered

        def _on_done():
            state["gen_relay"][0] = None  # release relay now that signal has been delivered
            lib_lbl.setText("✓ Library ready")
            gen_btn.setText("✓ Regenerate")
            gen_btn.setEnabled(True)
            for r in [gamma_row, min_i_row, scale_row]:
                r.setEnabled(True)
            norm_chk.setEnabled(True)
            refine_lbl.setText("Overlay active. Adjust sliders to refine.")
            run_btn_w.setEnabled(True)
            _activate_overlay()
            _select_step(2)

        _gen_relay.done.connect(_on_done)

        # Capture widget values on the GUI thread before spawning the worker.
        _accel_kv   = voltage_s.value()
        _resolution = res_s.value()
        _min_int    = min_int_s.value()
        _recip_r    = _compute_reciprocal_radius(signal)

        def _do_generate():
            from PySide6 import QtCore as _QC2
            try:
                new_sim = _generate_library_from_phases(
                    phases=state["phases"],
                    accelerating_voltage=_accel_kv,
                    resolution=_resolution,
                    minimum_intensity=_min_int,
                    reciprocal_radius=_recip_r,
                )
                state["sim"][0] = new_sim
                _QC2.QMetaObject.invokeMethod(
                    lib_lbl, "setText",
                    _QC2.Qt.ConnectionType.QueuedConnection,
                    _QC2.Q_ARG(str, "Building matching cache…"),
                )
                state["matching_cache"][0] = _build_matching_cache(signal, new_sim)
                state["refit_generation"][0] = 0
                _gen_relay.done.emit()
            except Exception as e:
                import traceback; traceback.print_exc()
                from PySide6 import QtCore as _QC2
                _QC2.QMetaObject.invokeMethod(
                    lib_lbl, "setText",
                    _QC2.Qt.ConnectionType.QueuedConnection,
                    _QC2.Q_ARG(str, f"Failed: {e}"),
                )
                _QC2.QMetaObject.invokeMethod(
                    gen_btn, "setEnabled",
                    _QC2.Qt.ConnectionType.QueuedConnection,
                    _QC2.Q_ARG(bool, True),
                )

        threading.Thread(target=_do_generate, daemon=True).start()

    gen_btn.clicked.connect(_on_generate)

    def _activate_overlay():
        """Add ScatterPlotItem + CircleROI to the signal plot (data coords)."""
        from pyqtgraph import ScatterPlotItem
        from pyqtgraph import CircleROI as PgCircleROI
        from PySide6.QtCore import QTimer as _QT

        if state["scatter_item"][0] is not None:
            return

        scatter = ScatterPlotItem(size=10, pen=mkPen("r", width=1.5), brush=None)
        plot.addItem(scatter)
        state["scatter_item"][0] = scatter

        # scene-x = -ky (sig_ax[1]),  scene-y = kx (sig_ax[0])
        _cx_scene = sig_ax[1].size / 2.0 * sig_ax[1].scale + sig_ax[1].offset  # ky center → scene-x
        _cy_scene = sig_ax[0].size / 2.0 * sig_ax[0].scale + sig_ax[0].offset  # kx center → scene-y

        r_data = state["max_radius"][0]
        cx_data = sig_ax[1].size / 2.0 * sig_ax[1].scale + sig_ax[1].offset
        cy_data = sig_ax[0].size / 2.0 * sig_ax[0].scale + sig_ax[0].offset
        circle_roi = PgCircleROI(
            pos=(cx_data - r_data, cy_data - r_data),
            size=(2 * r_data, 2 * r_data),
            pen=mkPen("y", width=1),
        )
        plot.addItem(circle_roi)
        state["circle_roi"][0] = circle_roi

        timer = _QT()
        timer.setInterval(50)
        timer.setSingleShot(True)
        state["refit_timer"][0] = timer

        # Relay object used to marshal spot + IPF data from worker thread to GUI thread.
        from PySide6 import QtCore as _QC
        class _SpotRelay(_QC.QObject):
            spots_ready = _QC.Signal(list, float, float, object)  # spots, ipf_x, ipf_y, heatmap
        _relay = _SpotRelay()

        # ── IPF plot window in MDI area ────────────────────────────────────────
        import pyqtgraph as _pg
        from PySide6 import QtWidgets as _QW2

        ipf_plot_window = main_window.add_plot_window(
            is_navigator=False,
            signal_tree=plot.signal_tree,
        )
        ipf_plot_window.setWindowTitle("IPF — Orientation Refinement")
        _mw_size = main_window.size()
        _ipf_side = min(_mw_size.width(), _mw_size.height()) // 4
        ipf_plot_window.resize(_ipf_side, _ipf_side)

        # Scope the IPF window to this action: without owner/controlling_action
        # the MDI 3-state visibility logic force-shows it on every same-tree
        # activation, so it could never stay hidden.
        ipf_plot_window.owner_plot_window = plot.plot_window
        ipf_plot_window.controlling_action = toolbar._find_action(action_name)
        ipf_plot_window.visibility_gate = lambda: (
            state["active_step"][0] == 2 and not state["ipf_user_closed"][0]
        )

        def _on_ipf_close_request():
            # X on the IPF window hides it for the rest of this Refine visit;
            # destroying it would leave a bare title-bar shell when the wizard
            # re-shows it (QMdiSubWindow close hides the inner container).
            state["ipf_user_closed"][0] = True
            ipf_plot_window.hide()

        ipf_plot_window.on_close_request = _on_ipf_close_request

        main_window._auto_position_near_owner(ipf_plot_window)
        state["ipf_widget"][0] = ipf_plot_window
        ipf_plot_window.hide()  # shown by _select_step(2)

        # Replace the GraphicsLayoutWidget with a raw PlotWidget for full control.
        # Remove the existing plot_widget and substitute our IPF PlotWidget.
        container_layout = ipf_plot_window.container.layout()
        ipf_plot_window.plot_widget.setParent(None)
        ipf_pw = _pg.PlotWidget()
        ipf_pw.setBackground("k")
        ipf_pw.hideAxis("bottom")
        ipf_pw.hideAxis("left")
        ipf_pw.setAspectLocked(True)
        container_layout.addWidget(ipf_pw)

        # Pre-compute IPF triangle outline and grid bounds for the phase
        _ipf_img_item = _pg.ImageItem()
        _ipf_img_item.setZValue(-10)
        ipf_pw.addItem(_ipf_img_item)
        _ipf_grid_info = [None]  # will hold (xx, yy, tri, weights, vertices, outside) once set

        try:
            from scipy.spatial import Delaunay as _Delaunay
            from orix.projections import StereographicProjection as _SP2
            from orix.vector import Vector3d as _V3b

            _phase = state["sim"][0].phases
            if hasattr(_phase, '__len__'):
                _phase = _phase[0]
            _tri_xy, _lbl_xy, _lbl_texts = _ipf_triangle_xy(_phase)

            # Draw triangle outline on top
            ipf_pw.plot(_tri_xy[:, 0], _tri_xy[:, 1],
                        pen=_pg.mkPen("w", width=1.5), antialias=True)
            for (lx, ly), txt in zip(_lbl_xy, _lbl_texts):
                ti = _pg.TextItem(txt, color="w", anchor=(0.5, 0.5))
                ti.setPos(lx, ly)
                ipf_pw.addItem(ti)

            # Pre-compute Delaunay interpolation grid over the IPF triangle extent
            GRID_N = 128
            mins = _tri_xy.min(axis=0)
            maxs = _tri_xy.max(axis=0)

            # All rotation vectors projected to stereographic coords
            _sp2 = _SP2()
            _all_rots = state["sim"][0].rotations
            if hasattr(_all_rots, '__len__'):
                _all_rots = _all_rots[0] if len(_all_rots) == 1 else _all_rots
            _vecs = (_all_rots * _V3b.zvector()).in_fundamental_sector(_phase.point_group)
            _rxs, _rys = _sp2.vector2xy(_vecs)
            _rxs = np.atleast_1d(np.array(_rxs, dtype=float))
            _rys = np.atleast_1d(np.array(_rys, dtype=float))
            rot_xy = np.vstack((_rxs, _rys)).T

            # Grid for the image
            gx = np.linspace(mins[0], maxs[0], GRID_N)
            gy = np.linspace(mins[1], maxs[1], GRID_N)
            xx, yy = np.meshgrid(gx, gy)
            flat_xy = np.vstack((xx.ravel(), yy.ravel())).T

            tri = _Delaunay(rot_xy)
            simplex = tri.find_simplex(flat_xy)
            outside = simplex < 0
            vertices = np.take(tri.simplices, simplex, axis=0)
            temp = np.take(tri.transform, simplex, axis=0)
            delta = flat_xy - temp[:, -1]
            bary = np.einsum("njk,nk->nj", temp[:, :-1, :], delta)
            bary_weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

            _ipf_grid_info[0] = (xx, yy, vertices, bary_weights, outside,
                                  mins, maxs, GRID_N, _rxs, _rys)

            # Set ImageItem transform so it spans the triangle extent
            from pyqtgraph import QtGui as _QtGui
            tr = _QtGui.QTransform()
            tr.translate(mins[0], mins[1])
            tr.scale((maxs[0] - mins[0]) / GRID_N, (maxs[1] - mins[1]) / GRID_N)
            _ipf_img_item.setTransform(tr)

        except Exception as e:
            print(f"IPF triangle/grid failed: {e}")

        # Best-match marker on top
        ipf_marker = _pg.ScatterPlotItem(
            size=14, pen=_pg.mkPen("w", width=1.5), brush=_pg.mkBrush("r")
        )
        ipf_pw.addItem(ipf_marker)

        # ── IPF mask circles (click to add, scroll to resize, Del to remove) ──
        # Each entry in state["ipf_mask_circles"] is [cx, cy, r].
        # Drawn as CircleROI-style items on ipf_pw.
        _mask_circle_items = []   # list of pg.CircleROI, parallel to state["ipf_mask_circles"]
        _selected_mask = [None]   # index of currently selected circle (for Del)

        def _redraw_mask_circles():
            for item in _mask_circle_items:
                try:
                    ipf_pw.removeItem(item)
                except Exception:
                    pass
            _mask_circle_items.clear()
            for i, (cx, cy, r) in enumerate(state["ipf_mask_circles"]):
                is_sel = (i == _selected_mask[0])
                pen = _pg.mkPen("y" if is_sel else "c", width=1.5)
                brush = _pg.mkBrush(0, 200, 255, 40 if not is_sel else 70)
                item = _pg.CircleROI(
                    pos=(cx - r, cy - r), size=(2 * r, 2 * r),
                    pen=pen, movable=False, resizable=False,
                )
                item.removeHandle(0)  # remove the default resize handle
                ipf_pw.addItem(item)
                _mask_circle_items.append(item)

        def _add_mask_circle(cx, cy):
            # Default radius = ~10% of the triangle extent
            gi = _ipf_grid_info[0]
            default_r = 0.05
            if gi is not None:
                mins, maxs = gi[5], gi[6]
                default_r = float((maxs - mins).mean()) * 0.08
            state["ipf_mask_circles"].append([cx, cy, default_r])
            _selected_mask[0] = len(state["ipf_mask_circles"]) - 1
            _redraw_mask_circles()
            _schedule()

        def _remove_selected_mask():
            idx = _selected_mask[0]
            if idx is not None and 0 <= idx < len(state["ipf_mask_circles"]):
                state["ipf_mask_circles"].pop(idx)
                _selected_mask[0] = None
                _redraw_mask_circles()
                _schedule()

        def _select_nearest_mask(cx, cy):
            best_i, best_d2 = None, float("inf")
            for i, (mcx, mcy, mr) in enumerate(state["ipf_mask_circles"]):
                d2 = (cx - mcx) ** 2 + (cy - mcy) ** 2
                if d2 < best_d2:
                    best_d2, best_i = d2, i
            _selected_mask[0] = best_i
            _redraw_mask_circles()

        def _on_ipf_click(event):
            from PySide6.QtCore import Qt as _Qt
            if event.button() == _Qt.MouseButton.LeftButton:
                pos = ipf_pw.plotItem.vb.mapSceneToView(event.scenePos())
                cx, cy = float(pos.x()), float(pos.y())
                # If click is near an existing circle centre, select it; else add new
                if state["ipf_mask_circles"]:
                    nearest_i = min(
                        range(len(state["ipf_mask_circles"])),
                        key=lambda i: (state["ipf_mask_circles"][i][0] - cx) ** 2
                                    + (state["ipf_mask_circles"][i][1] - cy) ** 2
                    )
                    mcx, mcy, mr = state["ipf_mask_circles"][nearest_i]
                    if (cx - mcx) ** 2 + (cy - mcy) ** 2 <= mr ** 2:
                        _select_nearest_mask(cx, cy)
                        return
                _add_mask_circle(cx, cy)

        def _on_ipf_scroll(event):
            # Map cursor to data coords (event.position() is in viewport coords)
            scene_pos = ipf_pw.viewport().mapToGlobal(event.position().toPoint())
            scene_pos = ipf_pw.mapFromGlobal(scene_pos)
            pos = ipf_pw.plotItem.vb.mapSceneToView(scene_pos)
            cx_cur, cy_cur = float(pos.x()), float(pos.y())
            # Find which circle the cursor is inside (if any)
            hit_idx = None
            for i, (mcx, mcy, mr) in enumerate(state["ipf_mask_circles"]):
                if (cx_cur - mcx) ** 2 + (cy_cur - mcy) ** 2 <= mr ** 2:
                    hit_idx = i
                    break
            if hit_idx is None:
                return False  # not consumed — let pyqtgraph zoom
            _selected_mask[0] = hit_idx
            delta = event.angleDelta().y()
            factor = 1.1 if delta > 0 else (1.0 / 1.1)
            state["ipf_mask_circles"][hit_idx][2] *= factor
            _redraw_mask_circles()
            _schedule()
            return True  # consumed

        def _on_ipf_key(event):
            from PySide6.QtCore import Qt as _Qt2
            if event.key() == _Qt2.Key.Key_Delete:
                _remove_selected_mask()

        ipf_pw.scene().sigMouseClicked.connect(_on_ipf_click)
        ipf_pw.keyPressEvent = _on_ipf_key
        ipf_pw.setFocusPolicy(_QC.Qt.FocusPolicy.ClickFocus)

        # Use an event filter on the viewport to catch wheel events for mask resize
        class _WheelFilter(_QC.QObject):
            def eventFilter(self, obj, event):
                if event.type() == _QC.QEvent.Type.Wheel:
                    consumed = _on_ipf_scroll(event)
                    return bool(consumed)  # consume only if cursor was inside a circle
                return False
        _wheel_filter = _WheelFilter(ipf_pw)
        ipf_pw.viewport().installEventFilter(_wheel_filter)
        state["_ipf_wheel_filter"] = _wheel_filter  # keep alive

        def _apply_spots(spots, ipf_x, ipf_y, heatmap):
            sc = state["scatter_item"][0]
            if sc is not None:
                sc.setData(spots)
            # Update heatmap image
            if heatmap is not None and _ipf_grid_info[0] is not None:
                _xs, _ys, cor_per_rot = heatmap  # cor_per_rot already indexed by lib_idx
                gi = _ipf_grid_info[0]
                verts, bw, outside, GRID_N = gi[2], gi[3], gi[4], gi[7]

                # Interpolate correlation values onto the regular grid
                safe_verts = np.clip(verts, 0, len(cor_per_rot) - 1)
                grid_vals = np.einsum("nj,nj->n", np.take(cor_per_rot, safe_verts), bw).astype(float)
                grid_vals[outside] = np.nan
                # meshgrid(gx, gy) is row=y, col=x → reshape gives (y, x).
                # pyqtgraph ImageItem expects (x, y), so transpose.
                img = grid_vals.reshape(GRID_N, GRID_N).T

                # inferno-style RGBA
                rgba = np.zeros((GRID_N, GRID_N, 4), dtype=np.uint8)
                valid = ~np.isnan(img)
                c = np.where(valid, img, 0.0)
                rgba[..., 0] = np.clip(255 * np.minimum(1.0, c * 2.0), 0, 255).astype(np.uint8)
                rgba[..., 1] = np.clip(255 * np.maximum(0.0, c * 2.0 - 1.0), 0, 255).astype(np.uint8)
                rgba[..., 2] = np.clip(80 * (1.0 - c), 0, 255).astype(np.uint8)
                rgba[..., 3] = np.where(valid, np.clip(220 * c + 30, 0, 255), 0).astype(np.uint8)
                _ipf_img_item.setImage(rgba, autoLevels=False)
            ipf_marker.setData([{"pos": (ipf_x, ipf_y)}])

        _relay.spots_ready.connect(_apply_spots)

        def _do_refit():
            if state["sim"][0] is None:
                return
            r_now = circle_roi.size().x() / 2.0
            state["max_radius"][0] = r_now
            sc_override = scale_s.value() if abs(scale_s.value() - sig_scale) > 1e-9 else None
            state["scale_override"][0] = sc_override
            nav_idx = _get_current_nav_indices(plot)
            # Grab the currently displayed pattern on the GUI thread before spawning worker
            _current_pattern = plot.current_data
            if _current_pattern is not None:
                _current_pattern = np.asarray(_current_pattern).copy()

            state["refit_generation"][0] += 1
            my_gen = state["refit_generation"][0]

            # Build rotation mask from IPF circles (done on GUI thread; arrays are read-only)
            _rot_mask = None
            gi = _ipf_grid_info[0]
            circles = list(state["ipf_mask_circles"])
            if gi is not None and circles:
                rot_xs, rot_ys = gi[8], gi[9]
                mask = np.zeros(len(rot_xs), dtype=bool)
                for cx, cy, r in circles:
                    dist2 = (rot_xs - cx) ** 2 + (rot_ys - cy) ** 2
                    mask |= dist2 <= r ** 2
                if mask.any():
                    _rot_mask = mask
            # Batch compute (Run page) uses the same mask the refine view shows
            state["rot_mask"] = _rot_mask

            def _run():
                if state["refit_generation"][0] != my_gen:
                    return  # superseded by a newer request
                try:
                    # coords_data are (kx, ky) in Å⁻¹, centered at origin (0,0)
                    coords_data, intensities, ipf_xy, ipf_heatmap = _get_best_fit_spots(
                        signal, state["sim"][0], nav_idx,
                        state["gamma"][0], state["max_radius"][0],
                        min_intensity=state["min_intensity"][0],
                        scale_override=sc_override,
                        matching_cache=state["matching_cache"][0],
                        pattern_override=_current_pattern,
                        normalize_templates=state["normalize_templates"][0],
                        rot_mask=_rot_mask,
                    )
                    if state["refit_generation"][0] != my_gen:
                        return
                    i_max = float(np.max(intensities)) if len(intensities) > 0 else 1.0
                    i_max = i_max if i_max > 0 else 1.0
                    # scene-x = -ky + ky_center,  scene-y = kx + kx_center
                    spots = [
                        {"pos": (-float(coords_data[i][1]) + _cx_scene,
                                  float(coords_data[i][0]) + _cy_scene),
                         "size": int(5 + 10 * float(intensities[i]) / i_max)}
                        for i in range(len(coords_data))
                    ]
                    if state["refit_generation"][0] == my_gen:
                        _relay.spots_ready.emit(spots, float(ipf_xy[0]), float(ipf_xy[1]), ipf_heatmap)
                except Exception as e:
                    print(f"Refit failed: {e}")

            threading.Thread(target=_run, daemon=True).start()

        def _schedule():
            if state["refit_timer"][0] is not None:
                state["refit_timer"][0].start()

        timer.timeout.connect(_do_refit)
        circle_roi.sigRegionChangeFinished.connect(_schedule)

        nav_sel = getattr(plot, "parent_selector", None)
        if nav_sel is None:
            nav_sel = getattr(getattr(plot, "plot_window", None), "parent_selector", None)
        if nav_sel is not None and hasattr(nav_sel, "roi"):
            # sigRegionChangeFinished fires on mouse release; sigRegionChanged fires
            # continuously during drag. Connect both so position updates feel live.
            nav_sel.roi.sigRegionChangeFinished.connect(_schedule)
            nav_sel.roi.sigRegionChanged.connect(_schedule)

        def _on_slider(_v=None):
            state["gamma"][0] = gamma_s.value()
            # slider is 0–100%, convert to 0.0–1.0 fraction for _get_best_fit_spots
            state["min_intensity"][0] = min_i_s.value() / 100.0
            _schedule()

        def _on_norm_chk(checked):
            state["normalize_templates"][0] = bool(checked)
            _schedule()

        gamma_s.valueChanged.connect(_on_slider)
        gamma_sl.valueChanged.connect(_on_slider)
        min_i_s.valueChanged.connect(_on_slider)
        min_i_sl.valueChanged.connect(_on_slider)
        scale_s.valueChanged.connect(_on_slider)
        scale_sl.valueChanged.connect(_on_slider)
        norm_chk.stateChanged.connect(_on_norm_chk)

        _schedule()

    # ── Batch compute: find_vectors-style workflow ────────────────────────────
    # Compute Map → chunked dispatch with the IPF-RGB orientation map painting
    # in live → lightweight result window: orientation map (navigator, left) +
    # per-phase IPFs with the current position's orientations (signal, right).

    class _OMRelay(_QC.QObject):
        done = _QC.Signal(object, object)   # SpyDEOrientationMap, new_tree
        failed = _QC.Signal(str)

    # No Qt parent: state keeps the reference (toolbar may be a test double)
    om_relay = _OMRelay()
    state["om_relay"] = om_relay

    def _om_read_selection(signal_plot, new_tree, nav_shape_2d):
        """
        Current navigator selection driving the result signal plot.

        Returns ("point", iy, ix) for crosshair-style selectors, or
        ("roi", ys, xs) slices when the selector spans a region
        (Integrate mode) — the IPF then shows ALL points in the ROI.
        """
        ny, nx = nav_shape_2d
        try:
            selector = signal_plot.plot_window.parent_selector
            if selector is not None:
                method = getattr(selector, "get_selected_indices", None) \
                    or getattr(selector, "_get_selected_indices", None)
                raw_idx = np.asarray(method())
                if raw_idx.ndim == 2 and len(raw_idx) >= 2:
                    lo = np.floor(raw_idx.min(axis=0)).astype(int)
                    hi = np.ceil(raw_idx.max(axis=0)).astype(int)
                    if int((hi - lo).max()) >= 1:  # region, not a point
                        y0 = int(np.clip(lo[0], 0, ny - 1))
                        y1 = int(np.clip(hi[0], 0, ny - 1))
                        x0 = int(np.clip(lo[1], 0, nx - 1))
                        x1 = int(np.clip(hi[1], 0, nx - 1))
                        return ("roi", slice(y0, y1 + 1), slice(x0, x1 + 1))
                idx = np.mean(np.atleast_2d(raw_idx), axis=0).astype(int)
                iy = int(np.clip(idx[0], 0, ny - 1))
                ix = int(np.clip(idx[1], 0, nx - 1))
                return ("point", iy, ix)
            nav_idx = new_tree.root.axes_manager.indices
            return ("point", int(nav_idx[1]), int(nav_idx[0]))
        except Exception:
            return ("point", 0, 0)

    def _om_letter_icon(text):
        from PySide6 import QtGui as _QG
        pm = _QG.QPixmap(22, 22)
        pm.fill(_QC.Qt.GlobalColor.transparent)
        p = _QG.QPainter(pm)
        p.setPen(_QG.QPen(_QG.QColor("white")))
        f = p.font()
        f.setBold(True)
        f.setPointSize(9 if len(text) > 1 else 11)
        p.setFont(f)
        p.drawText(pm.rect(), _QC.Qt.AlignmentFlag.AlignCenter, text)
        p.end()
        return _QG.QIcon(pm)

    def _build_ipf_result_widget(om):
        """
        Result view for the orientation map's signal window.

        Page 0: side-by-side 2D IPF panels (one per phase) with the current
        position's candidates (best solid, runners-up faded), or all points
        in the ROI when the navigator selector is in Integrate mode.
        Page 1: reduced 3D view — fundamental-sector outline on a minimal
        sphere; the marker is a point PLUS a tangent arrow encoding the
        in-plane rotation the 2D IPF cannot show.

        Exposed API (used by the navigation hook and toolbar actions):
        _update_position(iy, ix), _update_roi(ys, xs),
        _set_direction("x"|"y"|"z"), _set_3d(bool), _gl_available.
        """
        import pyqtgraph as _pg
        from spyde.signals.orientation_map import ipf_triangle_xy

        view_state = {"direction": "z", "pos": (0, 0), "roi": None}

        stacked = _QW.QStackedWidget()

        # ── Page 0: 2D IPF panels ─────────────────────────────────────────────
        page2d = _QW.QWidget()
        h = _QW.QHBoxLayout(page2d)
        h.setContentsMargins(2, 2, 2, 2)
        h.setSpacing(4)

        best_items, rest_items, roi_items = [], [], []
        for i in range(om.n_phases):
            panel = _QW.QWidget()
            pv = _QW.QVBoxLayout(panel)
            pv.setContentsMargins(0, 0, 0, 0)
            pv.setSpacing(2)
            title = _QW.QLabel(om.phases[i].get("name", f"phase {i}"))
            title.setStyleSheet("color: white; font-size: 10px;")
            title.setAlignment(_QC.Qt.AlignmentFlag.AlignHCenter)
            pw = _pg.PlotWidget()
            pw.setBackground("k")
            pw.hideAxis("bottom")
            pw.hideAxis("left")
            pw.setAspectLocked(True)
            try:
                edges, label_xy, labels = ipf_triangle_xy(om.orix_phase(i))
                pw.plot(edges[:, 0], edges[:, 1],
                        pen=_pg.mkPen("w", width=1.5))
                for (lx, ly), txt in zip(label_xy, labels):
                    ti = _pg.TextItem(txt, color="w", anchor=(0.5, 0.5))
                    ti.setPos(float(lx), float(ly))
                    pw.addItem(ti)
            except Exception as exc:
                print(f"IPF triangle for phase {i} failed: {exc}")
            roi_sc = _pg.ScatterPlotItem(
                symbol="o", size=4, pen=None,
                brush=_pg.mkBrush(80, 180, 255, 60),
            )
            rest = _pg.ScatterPlotItem(
                symbol="o", size=6, pen=None,
                brush=_pg.mkBrush(255, 255, 255, 70),
            )
            best = _pg.ScatterPlotItem(
                symbol="o", size=11,
                pen=_pg.mkPen("w", width=1.5),
                brush=_pg.mkBrush(255, 60, 60, 220),
            )
            pw.addItem(roi_sc)
            pw.addItem(rest)
            pw.addItem(best)
            pv.addWidget(title)
            pv.addWidget(pw)
            h.addWidget(panel)
            best_items.append(best)
            rest_items.append(rest)
            roi_items.append(roi_sc)

        stacked.addWidget(page2d)

        # ── Page 1: reduced 3D view (lazy, requires PyOpenGL) ─────────────────
        gl_state = {"view": None, "scatter": None, "tangent": None,
                    "roi_scatter": None}
        try:
            import pyqtgraph.opengl as _gl  # noqa: F401
            gl_available = True
        except Exception:
            gl_available = False

        def _ensure_gl():
            if gl_state["view"] is not None or not gl_available:
                return gl_state["view"]
            import pyqtgraph.opengl as gl
            from orix.projections import StereographicProjection  # noqa
            view = gl.GLViewWidget()
            view.setBackgroundColor("k")
            view.setCameraPosition(distance=3.0)
            # Minimal sphere context: three great circles
            t = np.linspace(0, 2 * np.pi, 120)
            for circle in (
                np.stack([np.cos(t), np.sin(t), 0 * t], axis=1),
                np.stack([np.cos(t), 0 * t, np.sin(t)], axis=1),
                np.stack([0 * t, np.cos(t), np.sin(t)], axis=1),
            ):
                view.addItem(gl.GLLinePlotItem(
                    pos=circle.astype(np.float32),
                    color=(0.35, 0.35, 0.35, 1.0), width=1.0,
                    antialias=True,
                ))
            # Fundamental-sector outline of phase 0 (context for the marker)
            try:
                sector = om.orix_phase(0).point_group.fundamental_sector
                from pyxem.signals.indexation_results import (
                    _closed_edges_in_hemisphere,
                )
                edges3 = _closed_edges_in_hemisphere(sector.edges, sector)
                view.addItem(gl.GLLinePlotItem(
                    pos=edges3.data.reshape(-1, 3).astype(np.float32),
                    color=(1.0, 1.0, 1.0, 1.0), width=2.0, antialias=True,
                ))
            except Exception as exc:
                print(f"3D sector outline failed: {exc}")
            gl_state["roi_scatter"] = gl.GLScatterPlotItem(
                pos=np.zeros((1, 3), np.float32), size=3.0,
                color=(0.3, 0.7, 1.0, 0.45),
            )
            gl_state["roi_scatter"].setVisible(False)
            gl_state["scatter"] = gl.GLScatterPlotItem(
                pos=np.zeros((1, 3), np.float32), size=10.0,
                color=(1.0, 0.25, 0.25, 1.0),
            )
            gl_state["tangent"] = gl.GLLinePlotItem(
                pos=np.zeros((2, 3), np.float32),
                color=(1.0, 0.85, 0.2, 1.0), width=3.0, antialias=True,
            )
            view.addItem(gl_state["roi_scatter"])
            view.addItem(gl_state["scatter"])
            view.addItem(gl_state["tangent"])
            gl_state["view"] = view
            stacked.addWidget(view)
            return view

        # ── Refresh logic ─────────────────────────────────────────────────────
        def _refresh_point():
            iy, ix = view_state["pos"]
            d = view_state["direction"]
            try:
                xy, pidx, _corr = om.ipf_xy(iy, ix, direction=d)
            except Exception:
                return
            for i in range(om.n_phases):
                best_items[i].setData([])
                rest_items[i].setData([])
                roi_items[i].setData([])
            p0 = int(pidx[0])
            best_items[p0].setData([{"pos": (float(xy[0, 0]),
                                             float(xy[0, 1]))}])
            for k in range(1, om.n_best):
                pk = int(pidx[k])
                rest_items[pk].addPoints(
                    [{"pos": (float(xy[k, 0]), float(xy[k, 1]))}]
                )
            if gl_state["view"] is not None:
                try:
                    v, tang, _p = om.ipf_xyz(iy, ix, direction=d)
                    gl_state["scatter"].setData(
                        pos=v[np.newaxis].astype(np.float32))
                    arrow = np.stack([v, v + 0.3 * tang]).astype(np.float32)
                    gl_state["tangent"].setData(pos=arrow)
                    gl_state["scatter"].setVisible(True)
                    gl_state["tangent"].setVisible(True)
                    gl_state["roi_scatter"].setVisible(False)
                except Exception as exc:
                    print(f"3D marker failed: {exc}")

        def _refresh_roi():
            roi = view_state["roi"]
            if roi is None:
                return
            ys, xs = roi
            d = view_state["direction"]
            for i in range(om.n_phases):
                best_items[i].setData([])
                rest_items[i].setData([])
                try:
                    xy, _c = om.ipf_xy_roi(ys, xs, phase=i, best_only=False,
                                           direction=d)
                    roi_items[i].setData(pos=xy)
                except Exception:
                    roi_items[i].setData([])
            if gl_state["view"] is not None:
                try:
                    from orix.quaternion import Rotation
                    from spyde.signals.orientation_map import \
                        _direction_vector
                    q = om.quats[ys, xs].reshape(-1, 4)
                    if len(q) > 20000:
                        q = q[::int(np.ceil(len(q) / 20000))]
                    vec = (Rotation(q) * _direction_vector(d))
                    vec = vec.in_fundamental_sector(
                        om.orix_phase(0).point_group)
                    gl_state["roi_scatter"].setData(
                        pos=vec.unit.data.reshape(-1, 3).astype(np.float32))
                    gl_state["roi_scatter"].setVisible(True)
                    gl_state["scatter"].setVisible(False)
                    gl_state["tangent"].setVisible(False)
                except Exception as exc:
                    print(f"3D ROI scatter failed: {exc}")

        def _refresh():
            if view_state["roi"] is not None:
                _refresh_roi()
            else:
                _refresh_point()

        # ── Public API ────────────────────────────────────────────────────────
        def _update_position(iy, ix):
            view_state["pos"] = (int(iy), int(ix))
            view_state["roi"] = None
            _refresh()

        def _update_roi(ys, xs):
            view_state["roi"] = (ys, xs)
            _refresh()

        def _set_direction(d):
            view_state["direction"] = str(d).lower()
            _refresh()

        def _set_3d(on):
            if on and gl_available:
                _ensure_gl()
                stacked.setCurrentIndex(1)
                _refresh()
            else:
                stacked.setCurrentIndex(0)

        stacked._update_position = _update_position
        stacked._update_roi = _update_roi
        stacked._set_direction = _set_direction
        stacked._set_3d = _set_3d
        stacked._gl_available = gl_available
        return stacked

    def _on_om_map_done(om, new_tree):
        try:
            poll_t = state.get("_om_poll_timer")
            if poll_t is not None:
                poll_t.stop()
            state["om_result"][0] = om
            new_tree.orientation_map = om
            om_save_btn.setEnabled(True)
            nav_shape_2d = om.nav_shape

            # Final authoritative orientation map + machinery-friendly data
            nav_plot_obj = None
            nav_pws = list(new_tree.navigator_plot_manager.plot_windows.keys())
            if nav_pws:
                nav_plots = new_tree.navigator_plot_manager.plots.get(
                    nav_pws[0], [])
                if nav_plots:
                    nav_plot_obj = nav_plots[0]
                    nav_plot_obj.image_item.setImage(
                        om.ipf_color_map("z"), autoLevels=False,
                        levels=(0, 255),
                    )
            try:
                nav_list = new_tree.navigator_signals.get("base")
                if nav_list:
                    nav_list[-1].data = om.correlation_map()
            except Exception:
                pass

            # Final authoritative X/Y/Z panels (replace the live preview)
            xyz_items = state.get("_om_xyz_items", [])
            for di, direction in enumerate(("x", "y", "z")):
                if di < len(xyz_items):
                    try:
                        xyz_items[di].setImage(
                            om.ipf_color_map(direction),
                            autoLevels=False, levels=(0, 255),
                        )
                    except Exception:
                        pass

            # Swap the result signal plot for the per-phase IPF panels
            ipf_widget = _build_ipf_result_widget(om)
            for sp in new_tree.signal_plots:
                pwin = sp.plot_window
                try:
                    pwin.plot_widget.hide()
                    pwin.container.layout().addWidget(ipf_widget)
                except Exception as exc:
                    print(f"IPF widget swap failed: {exc}")

                def _make_hook(orig_ud, signal_plot):
                    def _hooked(new_data, force=False):
                        try:
                            orig_ud(new_data, force=force)
                        except Exception:
                            pass
                        sel = _om_read_selection(signal_plot, new_tree,
                                                 nav_shape_2d)
                        if sel[0] == "roi":
                            ipf_widget._update_roi(sel[1], sel[2])
                        else:
                            ipf_widget._update_position(sel[1], sel[2])
                    return _hooked

                sp.update_data = _make_hook(sp.update_data, sp)

            # ── View controls on the RIGHT toolbar of the IPF window ─────────
            # (convention: processing actions live on the bottom toolbar,
            # view controls on the right one)
            try:
                ps = new_tree.signal_plots[0].plot_state
                tb_r = ps.toolbar_right
                dir_actions = {}
                state_3d = [False]

                def _apply_direction(d):
                    ipf_widget._set_direction(d)
                    try:
                        if nav_plot_obj is not None:
                            nav_plot_obj.image_item.setImage(
                                om.ipf_color_map(d), autoLevels=False,
                                levels=(0, 255),
                            )
                    except Exception:
                        pass
                    for dd, act in dir_actions.items():
                        act.blockSignals(True)
                        act.setChecked(dd == d)
                        act.blockSignals(False)

                for d in ("x", "y", "z"):
                    act, _w = tb_r.add_action(
                        f"IPF direction {d.upper()}",
                        _om_letter_icon(d.upper()),
                        (lambda tb, action_name=None, _d=d:
                         _apply_direction(_d)),
                        False, None, None,
                    )
                    act.setCheckable(True)
                    dir_actions[d] = act
                dir_actions["z"].setChecked(True)

                def _toggle_3d(tb, action_name=None):
                    state_3d[0] = not state_3d[0]
                    try:
                        ipf_widget._set_3d(state_3d[0])
                    except Exception as exc:
                        # GLViewWidget creation can fail even with PyOpenGL
                        # installed (drivers, remote desktop); report instead
                        # of silently doing nothing.
                        state_3d[0] = False
                        run_lbl.setText(f"3D view failed: {exc}")
                    act3d.blockSignals(True)
                    act3d.setChecked(state_3d[0])
                    act3d.blockSignals(False)

                act3d, _w3 = tb_r.add_action(
                    "3D IPF (in-plane rotation)", _om_letter_icon("3D"),
                    _toggle_3d, False, None, None,
                )
                # Checkable purely for visual feedback; _toggle_3d owns state.
                act3d.setCheckable(True)
                if not ipf_widget._gl_available:
                    act3d.setEnabled(False)
                    act3d.setToolTip("Install PyOpenGL for the 3D IPF view")
                tb_r.set_size()
                tb_r.show()
            except Exception as exc:
                print(f"IPF toolbar wiring failed: {exc}")

            ipf_widget._update_position(nav_shape_2d[0] // 2,
                                        nav_shape_2d[1] // 2)
            run_lbl.setText("✓ Done")
        except Exception as exc:
            import traceback as _tb
            _tb.print_exc()
            run_lbl.setText(f"Result display error: {exc}")
        finally:
            run_btn_w.setEnabled(True)

    def _on_om_map_failed(msg):
        poll_t = state.get("_om_poll_timer")
        if poll_t is not None:
            poll_t.stop()
        run_lbl.setText(msg)
        run_btn_w.setEnabled(True)

    om_relay.done.connect(_on_om_map_done)
    om_relay.failed.connect(_on_om_map_failed)

    def _on_compute_map_clicked():
        import hyperspy.api as hs
        import dask.array as da
        from spyde.drawing.update_functions import (
            ensure_live_buffer, read_live_buffer,
        )
        from spyde.drawing.selectors import CrosshairSelector
        from spyde.actions.find_vectors import _copy_nav_axes_to
        from spyde.actions.orientation_compute import _do_compute_orientations

        if state["sim"][0] is None:
            run_lbl.setText("Generate library first.")
            return
        sim_val = state["sim"][0]
        sig_ref = state["signal"]
        nav_dim = sig_ref.axes_manager.navigation_dimension
        if nav_dim != 2:
            run_lbl.setText("2D navigation only (for now).")
            return

        run_btn_w.setEnabled(False)
        run_lbl.setText("Computing…")

        params = dict(
            n_best=int(nbest_s.value()),
            gamma=float(state["gamma"][0]),
            normalize_templates=bool(state["normalize_templates"][0]),
            rot_mask=state.get("rot_mask"),
        )
        nav_shape_2d = tuple(sig_ref.data.shape[:nav_dim])
        shm_name = f"spyde_om_{id(plot)}"
        # 9 channels: live IPF RGB for X, Y, Z stacked channel-wise
        shm = ensure_live_buffer(nav_shape_2d + (9,), shm_name)
        state["_om_shm"] = shm  # keep alive

        # Lightweight result tree: lazy placeholder root + count-free
        # navigator override (no signal copy, no navigator recompute).
        data_shape = sig_ref.data.shape
        nav_chunks = tuple(min(32, int(s)) for s in data_shape[:nav_dim])
        placeholder = da.zeros(
            data_shape, chunks=nav_chunks + tuple(data_shape[nav_dim:]),
            dtype=np.float32,
        )
        new_sig = sig_ref._deepcopy_with_new_data(placeholder)
        if not new_sig._lazy:
            new_sig._lazy = True
            new_sig._assign_subclass()
        new_sig.metadata.General.title = (
            sig_ref.metadata.get_item("General.title", "Signal")
            + " — Orientations"
        )
        nav_sig = hs.signals.BaseSignal(
            np.zeros(nav_shape_2d, dtype=np.float32)
        ).T
        nav_sig.metadata.General.title = "Orientation map"
        _copy_nav_axes_to(sig_ref, nav_sig)

        main_window.add_signal(new_sig, selector_type=CrosshairSelector,
                               navigator_override=nav_sig)
        new_tree = main_window.signal_trees[-1]

        nav_pws = list(new_tree.navigator_plot_manager.plot_windows.keys())
        nav_plot_ref = [None]
        nav_pw_ref = [None]
        if nav_pws:
            nav_pw_ref[0] = nav_pws[0]
            nav_plots = new_tree.navigator_plot_manager.plots.get(
                nav_pws[0], [])
            if nav_plots:
                nav_plot_ref[0] = nav_plots[0]
        if nav_plot_ref[0] is not None:
            nav_plot_ref[0].image_item.setImage(
                np.zeros(nav_shape_2d + (3,), dtype=np.uint8),
                autoLevels=False, levels=(0, 255),
            )

        # ── Live IPF X/Y/Z window: three navigator-linked panels ─────────────
        # All three maps paint in chunk-by-chunk during the compute; the
        # panels share the navigator's coordinate frame, mirror its crosshair
        # and clicking any panel moves the navigation position.
        import pyqtgraph as _pg_xyz
        xyz_win = main_window.add_plot_window(
            is_navigator=False, signal_tree=new_tree,
        )
        xyz_win.setWindowTitle("IPF X / Y / Z")
        xyz_win.owner_plot_window = nav_pw_ref[0]
        _mw_sz = main_window.size()
        _panel = max(180, min(_mw_sz.width(), _mw_sz.height()) // 5)
        xyz_win.resize(3 * _panel, _panel + 60)
        _xyz_glw = _pg_xyz.GraphicsLayoutWidget()
        _xyz_glw.setBackground("k")
        xyz_items = []
        xyz_plots = []
        _nav_tr = None
        if nav_plot_ref[0] is not None:
            try:
                _nav_tr = nav_plot_ref[0].image_item.transform()
            except Exception:
                _nav_tr = None
        for _col, _dname in enumerate(("X", "Y", "Z")):
            _p = _xyz_glw.addPlot(row=0, col=_col, title=_dname)
            _p.hideAxis("bottom")
            _p.hideAxis("left")
            _p.setAspectLocked(True)
            _img = _pg_xyz.ImageItem(
                np.zeros(nav_shape_2d + (3,), dtype=np.uint8))
            if _nav_tr is not None:
                # same pixel→data transform as the navigator so the panels
                # live in navigator coordinates
                _img.setTransform(_nav_tr)
            _p.addItem(_img)
            xyz_items.append(_img)
            xyz_plots.append(_p)
        # pan/zoom the three panels together
        for _p in xyz_plots[1:]:
            _p.vb.setXLink(xyz_plots[0].vb)
            _p.vb.setYLink(xyz_plots[0].vb)
        xyz_win.plot_widget.setParent(None)
        xyz_win.container.layout().addWidget(_xyz_glw)
        main_window._auto_position_near_owner(xyz_win)
        # X hides (never destroys) so the live poll keeps a valid target;
        # a destroyed shell would come back as a bare title bar.
        _xyz_closed = [False]
        xyz_win.visibility_gate = lambda: not _xyz_closed[0]

        def _xyz_close():
            _xyz_closed[0] = True
            xyz_win.hide()

        xyz_win.on_close_request = _xyz_close
        state["_om_xyz_items"] = xyz_items
        state["_om_xyz_win"] = xyz_win

        # Crosshair link: mirror the result tree's navigator selector in every
        # panel, and let a click in any panel move the navigation position
        # (which drives the per-phase IPF result window).
        try:
            _xyz_sel = new_tree.signal_plots[0].plot_window.parent_selector
        except Exception:
            _xyz_sel = None
        if _xyz_sel is not None and hasattr(_xyz_sel, "roi"):
            _markers = []
            for _p in xyz_plots:
                _vl = _pg_xyz.InfiniteLine(
                    angle=90, movable=False,
                    pen=_pg_xyz.mkPen(255, 255, 255, 150, width=1.0))
                _hl = _pg_xyz.InfiniteLine(
                    angle=0, movable=False,
                    pen=_pg_xyz.mkPen(255, 255, 255, 150, width=1.0))
                _vl.setZValue(50)
                _hl.setZValue(50)
                _p.addItem(_vl)
                _p.addItem(_hl)
                _markers.append((_vl, _hl))

            def _sync_xyz_markers(*_a, _sel=_xyz_sel, _mk=_markers):
                try:
                    pos = _sel.roi.pos()
                    size = _sel.roi.size()
                    cx = pos.x() + size[0] / 2.0
                    cy = pos.y() + size[1] / 2.0
                except Exception:
                    return
                for vl, hl in _mk:
                    vl.setPos(cx)
                    hl.setPos(cy)

            _xyz_sel.roi.sigRegionChanged.connect(_sync_xyz_markers)
            _sync_xyz_markers()

            def _on_xyz_click(ev, _sel=_xyz_sel, _plots=xyz_plots):
                if ev.button() != _QC.Qt.MouseButton.LeftButton:
                    return
                for p in _plots:
                    vb = p.vb
                    if vb.sceneBoundingRect().contains(ev.scenePos()):
                        pt = vb.mapSceneToView(ev.scenePos())
                        size = _sel.roi.size()
                        _sel.roi.setPos(pt.x() - size[0] / 2.0,
                                        pt.y() - size[1] / 2.0)
                        break

            _xyz_glw.scene().sigMouseClicked.connect(_on_xyz_click)

        _stopped = [False]
        old_poll = state.get("_om_poll_timer")
        if old_poll is not None:
            old_poll.stop()
            old_poll.deleteLater()
        poll_timer = _QC.QTimer(toolbar)
        poll_timer.setInterval(150)
        state["_om_poll_timer"] = poll_timer

        def _poll():
            nav_p = nav_plot_ref[0]
            if nav_p is None:
                return
            arr = read_live_buffer(nav_shape_2d + (9,), shm_name)
            finite = np.isfinite(arr)
            if not finite.any():
                return
            disp = np.clip(np.nan_to_num(arr), 0, 255).astype(np.uint8)
            # Navigator shows the Z map; the XYZ window shows all three.
            nav_p.image_item.setImage(disp[..., 6:9], autoLevels=False,
                                      levels=(0, 255))
            for di, img in enumerate(state.get("_om_xyz_items", [])):
                img.setImage(disp[..., 3 * di:3 * di + 3],
                             autoLevels=False, levels=(0, 255))
            pct = 100.0 * float(finite[..., 0].mean())
            run_lbl.setText(f"Computing… {pct:.0f}%")

        poll_timer.timeout.connect(_poll)
        poll_timer.start()

        def _stop():
            _stopped[0] = True
            poll_timer.stop()
            run_btn_w.setEnabled(True)
            run_lbl.setText("Stopped.")

        if nav_pw_ref[0] is not None:
            nav_pw_ref[0].set_stop_fn(_stop)

        cache_snapshot = state["matching_cache"][0]

        def _run():
            try:
                om = _do_compute_orientations(
                    sig_ref, sim_val, params, main_window, None,
                    shm_name=shm_name, stopped_flag=_stopped,
                    cache=cache_snapshot,
                )
                if _stopped[0] or om is None:
                    return
                if nav_pw_ref[0] is not None:
                    _QC.QMetaObject.invokeMethod(
                        nav_pw_ref[0], "hide_stop_button",
                        _QC.Qt.ConnectionType.QueuedConnection,
                    )
                om_relay.done.emit(om, new_tree)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                if nav_pw_ref[0] is not None:
                    _QC.QMetaObject.invokeMethod(
                        nav_pw_ref[0], "hide_stop_button",
                        _QC.Qt.ConnectionType.QueuedConnection,
                    )
                om_relay.failed.emit(f"Error: {exc}")

        threading.Thread(target=_run, daemon=True).start()

    def _on_om_save_clicked():
        om = state["om_result"][0]
        if om is None:
            return
        path, _ = _QW.QFileDialog.getSaveFileName(
            caret, "Save Orientation Map", "", "SpyDE Orientations (*.npz)"
        )
        if not path:
            return
        try:
            om.save(path)
            run_lbl.setText(f"Saved to {path}")
        except Exception as exc:
            run_lbl.setText(f"Save failed: {exc}")

    om_save_btn.clicked.connect(_on_om_save_clicked)
    run_btn_w.clicked.connect(_on_compute_map_clicked)


def _do_run_fit(state):
    """Execute the full orientation mapping fit (called from submit or run button)."""
    if state["sim"][0] is None:
        if state["run_status"][0] is not None:
            state["run_status"][0].setText("Generate library first.")
        return

    signal = state["signal"]
    main_window = state["main_window"]
    gamma_val = state["gamma"][0]
    sim_val = state["sim"][0]
    nav_axes = list(signal.axes_manager.navigation_axes)
    n_phases = len(state["phases"]) if state["phases"] else 1

    run_btn = state["run_btn"][0]
    run_lbl = state["run_status"][0]
    if run_btn is not None:
        run_btn.setEnabled(False)
    if run_lbl is not None:
        run_lbl.setText("Running…")

    def _do_fit():
        try:
            polar = signal.get_azimuthal_integral2d(
                npt=100, npt_azim=360, inplace=False, mean=True
            )
            polar = polar ** gamma_val
            orientation_map = polar.get_orientation(sim_val, frac_keep=1)
            for i, ax in enumerate(nav_axes):
                if i < orientation_map.axes_manager.navigation_dimension:
                    out_ax = orientation_map.axes_manager.navigation_axes[i]
                    out_ax.scale = ax.scale; out_ax.offset = ax.offset
                    out_ax.units = ax.units; out_ax.name = ax.name
            results = _extract_orientation_outputs(orientation_map, nav_axes, n_phases)
            for result_signal, title in results:
                result_signal.metadata.General.title = title
                main_window._pending_signal_queue.append(result_signal)
            QtCore.QMetaObject.invokeMethod(
                main_window, "_flush_pending_signals",
                QtCore.Qt.ConnectionType.QueuedConnection,
            )
            if run_lbl is not None:
                QtCore.QMetaObject.invokeMethod(
                    run_lbl, "setText",
                    QtCore.Qt.ConnectionType.QueuedConnection,
                    QtCore.Q_ARG(str, "✓ Done"),
                )
        except Exception as e:
            print(f"Orientation mapping failed: {e}")
            if run_lbl is not None:
                QtCore.QMetaObject.invokeMethod(
                    run_lbl, "setText",
                    QtCore.Qt.ConnectionType.QueuedConnection,
                    QtCore.Q_ARG(str, f"Failed: {e}"),
                )
        finally:
            if run_btn is not None:
                QtCore.QMetaObject.invokeMethod(
                    run_btn, "setEnabled",
                    QtCore.Qt.ConnectionType.QueuedConnection,
                    QtCore.Q_ARG(bool, True),
                )

    threading.Thread(target=_do_fit, daemon=True).start()
