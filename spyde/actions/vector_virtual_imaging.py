"""
Vector Virtual Imaging — build virtual images from a SpyDEDiffractionVectors
result tree (the output of Find Diffraction Vectors).

Mirrors the Diffraction2D ``Virtual Imaging`` action (toolbar toggle + a
submenu "Add Vector Virtual Image", multiple images via repeated adds, each
with its own colored circular ROI on the diffraction plot). The difference is
the compute path: instead of a Dask reduction over the raw 4D dataset, each
image is built in-memory from the CSR flat buffer via
``vecs.virtual_image_from_roi_gpu`` — O(N_frame), so it recomputes live on
every ROI drag with no shared-memory streaming.

Each ROI is a filled disk whose **radius defaults to the detection kernel
radius** and whose contribution is **intensity-weighted** (sums NXCORR peak
scores), matching how the vectors were detected.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
from PySide6.QtGui import Qt
from pyqtgraph import CircleROI, RectROI, mkPen

from spyde.drawing.toolbars.plot_control_toolbar import resolve_icon_path

# Same palette/cycle as pyxem.add_virtual_image so the two tools look uniform.
_COLORS = ["red", "green", "blue", "yellow", "cyan", "magenta"]


def vector_virtual_imaging(*args, **kwargs):
    """Toggle handler for the parent action — the submenu does the work."""
    print("Vector virtual imaging action triggered.")


def _colored_icon(toolbar, color: str) -> QIcon:
    """Tint the virtual-imaging glyph with `color` (same recipe as pyxem)."""
    base_icon = QIcon(resolve_icon_path("drawing/toolbars/icons/virtual_imaging.svg"))
    icon_size = toolbar.iconSize()
    dpr = getattr(toolbar, "devicePixelRatioF", lambda: 1.0)()
    req_w = max(1, int(icon_size.width() * dpr))
    req_h = max(1, int(icon_size.height() * dpr))
    base_pixmap = base_icon.pixmap(req_w, req_h)

    colored = QPixmap(base_pixmap.size())
    colored.setDevicePixelRatio(dpr)
    colored.fill(Qt.GlobalColor.transparent)
    painter = QPainter(colored)
    painter.drawPixmap(0, 0, base_pixmap)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(colored.rect(), QColor(color))
    painter.end()

    icon = QIcon()
    icon.addPixmap(colored)
    return icon


def add_vector_virtual_image(
    toolbar, action_name: str = "Add Vector Virtual Image", *args, **kwargs
):
    """
    Add one vector virtual image: a colored circular ROI on the diffraction
    plot plus a preview window that recomputes live from the vectors as the
    ROI is dragged.
    """
    from PySide6 import QtCore as _QtCore

    plot = toolbar.parent_toolbar.plot
    signal_tree = getattr(plot, "signal_tree", None)
    vecs = getattr(signal_tree, "diffraction_vectors", None)
    if vecs is None:
        print("Vector virtual imaging: no diffraction_vectors on tree.")
        return
    main_window = plot.main_window
    sig_ax = plot.plot_state.current_signal.axes_manager.signal_axes

    num = toolbar.num_actions()
    color = _COLORS[num % len(_COLORS)]
    icon = _colored_icon(toolbar, color)
    pen = mkPen(color=color, width=2)
    action_name = f"Vector Image ({color})"

    # ── Create the action (toggle) so the ROI hides/shows with it ────────────
    action, params_caret_box = toolbar.add_action(
        name=action_name,
        icon_path=icon,
        function=vector_virtual_imaging,
        toggle=True,
        parameters={
            "shape": {
                "name": "Detector shape",
                "type": "enum",
                "default": "disk",
                "options": ["disk", "annulus", "rectangle"],
            },
            "weighting": {
                "name": "Weighting",
                "type": "enum",
                "default": "intensity",
                "options": ["intensity", "count"],
            },
        },
    )
    try:
        params_caret_box.submit_button.hide()
        if hasattr(params_caret_box, "finalize_layout"):
            params_caret_box.finalize_layout()
    except Exception:
        pass

    # ── Preview window ───────────────────────────────────────────────────────
    from spyde.qt.compute_status_indicator import ComputeStatusIndicator

    vi_window = main_window.add_plot_window(
        is_navigator=False, signal_tree=signal_tree,
    )
    vi_window.owner_plot_window = plot.plot_window
    main_window._auto_position_near_owner(vi_window)
    vi_plot = vi_window.add_new_plot()
    if vi_plot.image_item not in vi_plot.items:
        vi_plot.addItem(vi_plot.image_item)

    indicator = ComputeStatusIndicator(color=color)
    vi_window.set_compute_indicator(indicator)

    toolbar.parent_toolbar.register_action_plot_window(
        action_name="Vector Virtual Imaging",
        plot_window=vi_window,
        key=action_name,
    )

    # ── ROI geometry — initial size = a few detection kernel radii ───────────
    r_data = float(vecs.kernel_radius_data)
    r0 = max(r_data * 3.0, r_data + 1e-6)
    if sig_ax is not None and len(sig_ax) >= 2:
        cx0 = float(sig_ax[0].offset + sig_ax[0].scale * sig_ax[0].size / 2)
        cy0 = float(sig_ax[1].offset + sig_ax[1].scale * sig_ax[1].size / 2)
    else:
        cx0 = cy0 = 0.0

    # Scene convention: the diffraction plot shows kx along scene-x, ky along
    # scene-y (see _render_disks_block / plot transform). ROIs live in scene
    # coords, so an ROI at scene (sx, sy) selects detector (kx=sx, ky=sy).
    roi_ref = [None]

    def _make_roi(shape):
        if shape == "rectangle":
            return RectROI(pos=(cx0 - r0, cy0 - r0), size=(2 * r0, 2 * r0),
                           pen=pen)
        # disk and annulus both use a CircleROI; annulus adds an inner cutoff
        # equal to the kernel radius (BF core excluded).
        return CircleROI(pos=(cx0 - r0, cy0 - r0), size=(2 * r0, 2 * r0),
                         pen=pen, removable=False)

    def _shape():
        try:
            return params_caret_box.kwargs["shape"].currentText()
        except Exception:
            return "disk"

    roi = _make_roi(_shape())
    roi_ref[0] = roi
    toolbar.parent_toolbar.register_action_plot_item(
        action_name="Vector Virtual Imaging", item=roi, key=action_name,
    )

    _levels = [None]

    def _current_t():
        if vecs.n_time <= 0:
            return None
        try:
            return int(signal_tree.root.axes_manager.indices[0])
        except Exception:
            return None

    def _recompute():
        r = roi_ref[0]
        if r is None:
            return
        pos = r.pos(); size = r.size()
        # scene-x = kx, scene-y = ky
        shape = _shape()
        try:
            weighted = params_caret_box.kwargs["weighting"].currentText() != "count"
        except Exception:
            weighted = True

        indicator.set_computing()
        if shape == "rectangle":
            x0 = float(pos.x()); y0 = float(pos.y())
            img = vecs.virtual_image_from_rect(
                x0, y0, x0 + float(size.x()), y0 + float(size.y()),
                t=_current_t(), intensity_weighted=weighted)
        else:
            cx = float(pos.x() + size.x() / 2.0)
            cy = float(pos.y() + size.y() / 2.0)
            r_out = float(size.x() / 2.0)
            r_in = min(r_data, r_out * 0.99) if shape == "annulus" else 0.0
            img = vecs.virtual_image_from_roi_gpu(
                cx, cy, r_out, r_in, t=_current_t(), intensity_weighted=weighted)
        finite = img[img > 0]
        if finite.size:
            hi = float(finite.max())
            _levels[0] = (0.0, hi if hi > 0 else 1.0)
        lvl = _levels[0] if _levels[0] is not None else (0.0, 1.0)
        vi_plot.image_item.setImage(img, autoLevels=False, levels=lvl)
        indicator.set_done()

    # Debounce drags at ~30 ms; the compute itself is sub-ms for typical data.
    _timer = _QtCore.QTimer(toolbar.parent_toolbar)
    _timer.setInterval(30)
    _timer.setSingleShot(True)
    _timer.timeout.connect(_recompute)

    def _schedule():
        _timer.start()

    roi.sigRegionChanged.connect(_schedule)

    def _on_shape_change(new_shape):
        """Swap the ROI to the selected shape, keeping its position/size."""
        old = toolbar.parent_toolbar.unregister_action_plot_item(
            action_name="Vector Virtual Imaging", key=action_name)
        pos = old.pos(); size = old.size()
        new_roi = _make_roi(new_shape)
        new_roi.setPos(pos); new_roi.setSize(size)
        roi_ref[0] = new_roi
        toolbar.parent_toolbar.register_action_plot_item(
            action_name="Vector Virtual Imaging", item=new_roi, key=action_name)
        new_roi.sigRegionChanged.connect(_schedule)
        _schedule()

    try:
        sw = params_caret_box.kwargs["shape"]
        if hasattr(sw, "currentTextChanged"):
            sw.currentTextChanged.connect(_on_shape_change)
        ww = params_caret_box.kwargs["weighting"]
        if hasattr(ww, "currentTextChanged"):
            ww.currentTextChanged.connect(lambda _=None: _schedule())
    except Exception:
        pass

    # ── Cleanup when the preview window closes ───────────────────────────────
    def _cleanup():
        _timer.stop()
        try:
            toolbar.parent_toolbar.unregister_action_plot_item(
                action_name="Vector Virtual Imaging", key=action_name,
            )
        except Exception:
            pass
        try:
            toolbar.remove_action(action_name)
        except Exception:
            pass

    _orig_close = vi_window.close_window

    def _close_with_cleanup():
        _cleanup()
        _orig_close()

    vi_window.close_window = _close_with_cleanup

    # First draw
    _recompute()
    return roi
