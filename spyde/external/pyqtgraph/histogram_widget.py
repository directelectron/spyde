"""
GraphicsWidget displaying an image histogram along with gradient editor. Can be used to
adjust the appearance of images.
"""

import weakref
from functools import partial

import numpy as np

from pyqtgraph import debug as debug
from pyqtgraph import functions as fn
from pyqtgraph.Point import Point
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph import AxisItem
from pyqtgraph import BarGraphItem
from pyqtgraph import GradientEditorItem
from pyqtgraph import GraphicsWidget
from pyqtgraph import LinearRegionItem
from pyqtgraph import PlotCurveItem
from pyqtgraph import ViewBox
from pyqtgraph import GraphicsView

__all__ = ["HistogramLUTItem", "HistogramLUTWidget"]


class _GammaCurveItem(PlotCurveItem):
    """Gamma transfer line — drag it vertically to change gamma.

    The drag point (x, y) is normalised inside the current (min, max, height)
    box; gamma solves y_norm = x_norm**gamma at that point, so the curve
    follows the cursor wherever it is grabbed.
    """

    sigGammaDrag = QtCore.Signal(float)  # new gamma value

    def __init__(self, get_levels, get_height, **kwargs):
        super().__init__(**kwargs)
        self._get_levels = get_levels
        self._get_height = get_height
        self.setClickable(True, width=12)
        self.setCursor(QtCore.Qt.CursorShape.SizeVerCursor)

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        ev.accept()
        try:
            mn, mx = self._get_levels()
            h = float(self._get_height())
        except Exception:
            return
        if mx <= mn or h <= 0:
            return
        pos = ev.pos()  # item coords == view (data) coords
        nx = min(max((pos.x() - mn) / (mx - mn), 0.05), 0.95)
        ny = min(max(pos.y() / h, 0.02), 0.98)
        gamma = float(np.log(ny) / np.log(nx))
        self.sigGammaDrag.emit(min(max(gamma, 0.05), 20.0))


class HistogramLUTItem(GraphicsWidget):
    """
    :class:`~pyqtgraph.GraphicsWidget` with controls for adjusting the display of an
    :class:`~pyqtgraph.ImageItem`.

    Includes:

      - Image histogram
      - Movable region over the histogram to select black/white levels
      - Gradient editor to define color lookup table for single-channel images

    Parameters
    ----------
    image : pyqtgraph.ImageItem, optional
        If provided, control will be automatically linked to the image and changes to
        the control will be reflected in the image's appearance. This may also be set
        via :meth:`setImageItem`.
    fillHistogram : bool, optional
        By default, the histogram is rendered with a fill. Performance may be improved
        by disabling the fill. Additional control over the fill is provided by
        :meth:`fillHistogram`.
    levelMode : str, optional
        'mono' (default)
            One histogram with a :class:`~pyqtgraph.LinearRegionItem` is displayed to
            control the black/white levels of the image. This option may be used for
            color images, in which case the histogram and levels correspond to all
            channels of the image.
        'rgba'
            A histogram and level control pair is provided for each image channel. The
            alpha channel histogram and level control are only shown if the image
            contains an alpha channel.
    gradientPosition : str, optional
        Position of the gradient editor relative to the histogram. Must be one of
        {'right', 'left', 'top', 'bottom'}. 'right' and 'left' options should be used
        with a 'vertical' orientation; 'top' and 'bottom' options are for 'horizontal'
        orientation.
    orientation : str, optional
        The orientation of the axis along which the histogram is displayed. Either
        'vertical' (default) or 'horizontal'.
    autoLevel : bool, optional
        If True, the levels will be set automatically based on the image histogram
        whenever the image data changes. Default is True.
    constantLevel: bool, optional
        If True, the relative levels will be maintained when the image data changes.

    Attributes
    ----------
    sigLookupTableChanged : QtCore.Signal
        Emits the HistogramLUTItem itself when the gradient changes
    sigLevelsChanged : QtCore.Signal
        Emits the HistogramLUTItem itself while the movable region is changing
    sigLevelChangeFinished : QtCore.Signal
        Emits the HistogramLUTItem itself when the movable region is finished changing

    See Also
    --------
    :class:`~pyqtgraph.ImageItem`
        HistogramLUTItem is most useful when paired with an ImageItem.
    :class:`~pyqtgraph.ImageView`
        Widget containing a paired ImageItem and HistogramLUTItem.
    :class:`~pyqtgraph.HistogramLUTWidget`
        QWidget containing a HistogramLUTItem for widget-based layouts.
    """

    sigLookupTableChanged = QtCore.Signal(object)
    sigLevelsChanged = QtCore.Signal(object)
    sigLevelChangeFinished = QtCore.Signal(object)
    sigGammaChanged = QtCore.Signal(object)

    def __init__(
        self,
        image=None,
        fillHistogram=True,
        levelMode="mono",
        gradientPosition="right",
        orientation="vertical",
        autoLevel=True,
        constantLevel=False,
        show_gradient=True,
    ):
        GraphicsWidget.__init__(self)
        self.bins = None
        self.counts = None
        self.lut = None
        self._show_gradient = bool(show_gradient)
        # Bar-chart + gamma UI only in the horizontal mono configuration
        # (the Plot Control dock); vertical falls back to the curve look.
        self._use_bars = orientation == "horizontal"
        self.gamma = 1.0
        self.imageItem = lambda: None  # fake a dead weakref
        self.levelMode = levelMode
        self.orientation = orientation
        self.gradientPosition = gradientPosition

        if orientation == "vertical" and gradientPosition not in {"right", "left"}:
            self.gradientPosition = "right"
        elif orientation == "horizontal" and gradientPosition not in {"top", "bottom"}:
            self.gradientPosition = "bottom"

        self.layout = QtWidgets.QGraphicsGridLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(1, 1, 1, 1)
        self.layout.setSpacing(0)

        self.vb = ViewBox(parent=self)
        if self.orientation == "vertical":
            self.vb.setMaximumWidth(152)
            self.vb.setMinimumWidth(45)
            self.vb.setMouseEnabled(x=False, y=True)
        elif self._use_bars:
            # Bar-chart mode: the histogram view is fixed in place — the
            # range is set from the image data on every redraw, never by
            # mouse pan/zoom.
            self.vb.setMaximumHeight(152)
            self.vb.setMinimumHeight(45)
            self.vb.setMouseEnabled(x=False, y=False)
            self.vb.setMenuEnabled(False)
            self.vb.setDefaultPadding(0.0)
        else:
            self.vb.setMaximumHeight(152)
            self.vb.setMinimumHeight(45)
            self.vb.setMouseEnabled(x=True, y=False)

        self.gradient = GradientEditorItem(orientation=self.gradientPosition)
        self.gradient.loadPreset("grey")

        # LinearRegionItem orientation refers to the bounding lines
        regionOrientation = (
            "horizontal" if self.orientation == "vertical" else "vertical"
        )
        self.regions = [
            # single region for mono levelMode
            LinearRegionItem([0, 1], regionOrientation, swapMode="block"),
            # r/g/b/a regions for rgba levelMode
            LinearRegionItem(
                [0, 1],
                regionOrientation,
                swapMode="block",
                pen="r",
                brush=fn.mkBrush((255, 50, 50, 50)),
                span=(0.0, 1 / 3.0),
            ),
            LinearRegionItem(
                [0, 1],
                regionOrientation,
                swapMode="block",
                pen="g",
                brush=fn.mkBrush((50, 255, 50, 50)),
                span=(1 / 3.0, 2 / 3.0),
            ),
            LinearRegionItem(
                [0, 1],
                regionOrientation,
                swapMode="block",
                pen="b",
                brush=fn.mkBrush((50, 50, 255, 80)),
                span=(2 / 3.0, 1.0),
            ),
            LinearRegionItem(
                [0, 1],
                regionOrientation,
                swapMode="block",
                pen="w",
                brush=fn.mkBrush((255, 255, 255, 50)),
                span=(2 / 3.0, 1.0),
            ),
        ]
        self.region = self.regions[0]  # for backward compatibility.
        for region in self.regions:
            region.setZValue(1000)
            self.vb.addItem(region)
            if region is not self.region:
                # arrow markers only on the rgba per-channel regions; the
                # mono region uses clean unadorned drag lines
                region.lines[0].addMarker("<|", 0.5)
                region.lines[1].addMarker("|>", 0.5)
            region.sigRegionChanged.connect(self.regionChanging)
            region.sigRegionChangeFinished.connect(self.regionChanged)

        # Clean styling for the mono min/max drag lines — same orange as the
        # gamma transfer line, with a horizontal-drag cursor.
        try:
            self.region.setBrush(fn.mkBrush(255, 200, 80, 8))
        except Exception:
            pass
        for line in self.region.lines:
            line.setPen(fn.mkPen(255, 200, 80, 220, width=1.5))
            line.setHoverPen(fn.mkPen(255, 225, 130, 255, width=2.0))
            line.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)

        if not self._show_gradient and self.orientation == "horizontal":
            # Clean dock layout: bars on top, slim axis underneath.
            self.axis = AxisItem("bottom", linkView=self.vb,
                                 maxTickLength=-6, parent=self)
            self.layout.addItem(self.vb, 0, 0)
            self.layout.addItem(self.axis, 1, 0)
        else:
            # gradient position to axis orientation
            ax = {"left": "right", "right": "left", "top": "bottom",
                  "bottom": "top"}[self.gradientPosition]
            self.axis = AxisItem(ax, linkView=self.vb, maxTickLength=-10,
                                 parent=self)

            # axis / viewbox / gradient order in the grid
            avg = ((0, 1, 2) if self.gradientPosition in {"right", "bottom"}
                   else (2, 1, 0))
            if self.orientation == "vertical":
                self.layout.addItem(self.axis, 0, avg[0])
                self.layout.addItem(self.vb, 0, avg[1])
                if self._show_gradient:
                    self.layout.addItem(self.gradient, 0, avg[2])
            else:
                self.layout.addItem(self.axis, avg[0], 0)
                self.layout.addItem(self.vb, avg[1], 0)
                if self._show_gradient:
                    self.layout.addItem(self.gradient, avg[2], 0)
        if not self._show_gradient:
            # colormap is owned by the Plot (Colormap combo); keep the
            # gradient object alive for API compatibility but never shown
            self.gradient.hide()

        self.gradient.setFlag(self.gradient.GraphicsItemFlag.ItemStacksBehindParent)
        self.vb.setFlag(self.gradient.GraphicsItemFlag.ItemStacksBehindParent)

        self.gradient.sigGradientChanged.connect(self.gradientChanged)
        self.vb.sigRangeChanged.connect(self.viewRangeChanged)

        comp = QtGui.QPainter.CompositionMode.CompositionMode_Plus
        self.plots = [
            PlotCurveItem(pen=(200, 200, 200, 100)),  # mono
            PlotCurveItem(pen=(255, 0, 0, 100), compositionMode=comp),  # r
            PlotCurveItem(pen=(0, 255, 0, 100), compositionMode=comp),  # g
            PlotCurveItem(pen=(0, 0, 255, 100), compositionMode=comp),  # b
            PlotCurveItem(pen=(200, 200, 200, 100), compositionMode=comp),  # a
        ]
        self.plot = self.plots[0]  # for backward compatibility.
        for plot in self.plots:
            if self.orientation == "vertical":
                plot.setRotation(90)
            self.vb.addItem(plot)

        # Proper bar-chart histogram (mono, horizontal) + gamma transfer line
        self.bar_item = None
        self.gamma_curve = None
        self._disp_hmax = 1.0  # y-scale of the view (outlier-clipped)
        if self._use_bars:
            self.bar_item = BarGraphItem(
                x=[0.0], height=[0.0], width=1.0,
                # neutral bars so the orange level lines / gamma curve (the
                # app accent) stay the interactive layer that pops
                brush=fn.mkBrush(190, 190, 190, 120), pen=None,
            )
            self.bar_item.setZValue(0)
            self.vb.addItem(self.bar_item)

            self.gamma_curve = _GammaCurveItem(
                get_levels=lambda: self.region.getRegion(),
                get_height=self._hist_height,
                pen=fn.mkPen(255, 200, 80, 230, width=1.6), antialias=True,
            )
            self.gamma_curve.setZValue(900)
            self.vb.addItem(self.gamma_curve)
            self.gamma_curve.sigGammaDrag.connect(self._on_gamma_dragged)

        self.fillHistogram(fillHistogram)
        self._showRegions()
        self.autoLevel = autoLevel
        self.constantLevel = constantLevel
        self.autoHistogramRange()

        self._imageChangedDebounceMs = 35
        self._imageChangedTimer = QtCore.QTimer(self)
        self._imageChangedTimer.setSingleShot(True)
        self._imageChangedTimer.setInterval(self._imageChangedDebounceMs)
        self._pendingImageChange = {
            "autoLevel": False,
            "autoRange": False,
            "min_percentile": None,
            "max_percentile": None,
        }
        self._imageChangedTimer.timeout.connect(self._processImageChanged)

        if image is not None:
            self.setImageItem(image)

    def fillHistogram(self, fill=True, level=0.0, color=(100, 100, 200)):
        """Control fill of the histogram curve(s).

        Parameters
        ----------
        fill : bool, optional
            Set whether or not the histogram should be filled.
        level : float, optional
            Set the fill level. See :meth:`PlotCurveItem.setFillLevel
            <pyqtgraph.PlotCurveItem.setFillLevel>`. Only used if ``fill`` is True.
        color : color_like, optional
            Color to use for the fill when the histogram ``levelMode == "mono"``. See
            :meth:`PlotCurveItem.setBrush <pyqtgraph.PlotCurveItem.setBrush>`.
        """
        colors = [
            color,
            (255, 0, 0, 50),
            (0, 255, 0, 50),
            (0, 0, 255, 50),
            (255, 255, 255, 50),
        ]
        for color, plot in zip(colors, self.plots):
            if fill:
                plot.setFillLevel(level)
                plot.setBrush(color)
            else:
                plot.setFillLevel(None)

    def paint(self, p, *args):
        # paint the bounding edges of the region item and gradient item with lines
        # connecting them
        if self.levelMode != "mono" or not self.region.isVisible():
            return
        if not self._show_gradient:
            return

        pen = self.region.lines[0].pen

        mn, mx = self.getLevels()
        vbc = self.vb.viewRect().center()
        gradRect = self.gradient.mapRectToParent(self.gradient.gradRect.rect())
        if self.orientation == "vertical":
            p1mn = self.vb.mapFromViewToItem(self, Point(vbc.x(), mn)) + Point(0, 5)
            p1mx = self.vb.mapFromViewToItem(self, Point(vbc.x(), mx)) - Point(0, 5)
            if self.gradientPosition == "right":
                p2mn = gradRect.bottomLeft()
                p2mx = gradRect.topLeft()
            else:
                p2mn = gradRect.bottomRight()
                p2mx = gradRect.topRight()
        else:
            p1mn = self.vb.mapFromViewToItem(self, Point(mn, vbc.y())) - Point(5, 0)
            p1mx = self.vb.mapFromViewToItem(self, Point(mx, vbc.y())) + Point(5, 0)
            if self.gradientPosition == "bottom":
                p2mn = gradRect.topLeft()
                p2mx = gradRect.topRight()
            else:
                p2mn = gradRect.bottomLeft()
                p2mx = gradRect.bottomRight()

        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        for pen in [fn.mkPen((0, 0, 0, 100), width=3), pen]:
            p.setPen(pen)

            # lines from the linear region item bounds to the gradient item bounds
            p.drawLine(p1mn, p2mn)
            p.drawLine(p1mx, p2mx)

            # lines bounding the edges of the gradient item
            if self.orientation == "vertical":
                p.drawLine(gradRect.topLeft(), gradRect.topRight())
                p.drawLine(gradRect.bottomLeft(), gradRect.bottomRight())
            else:
                p.drawLine(gradRect.topLeft(), gradRect.bottomLeft())
                p.drawLine(gradRect.topRight(), gradRect.bottomRight())

    def setHistogramRange(self, mn, mx, padding=0.1):
        """Set the X/Y range on the histogram plot, depending on the orientation. This disables auto-scaling."""
        if self.orientation == "vertical":
            self.vb.enableAutoRange(self.vb.YAxis, False)
            self.vb.setYRange(mn, mx, padding)
        else:
            self.vb.enableAutoRange(self.vb.XAxis, False)
            self.vb.setXRange(mn, mx, padding)

    def getHistogramRange(self):
        """Returns range on the histogram plot."""
        if self.orientation == "vertical":
            return self.vb.viewRange()[1]
        else:
            return self.vb.viewRange()[0]

    def autoHistogramRange(self):
        """Enable auto-scaling on the histogram plot."""
        self.vb.enableAutoRange(self.vb.XYAxes)

    def disableAutoHistogramRange(self):
        """Disable auto-scaling on the histogram plot."""
        self.vb.disableAutoRange(self.vb.XYAxes)

    def setImageItem(self, img, min_percentile=None, max_percentile=None):
        """Set an ImageItem to have its levels and LUT automatically controlled by this
        HistogramLUTItem.
        """
        self.imageItem = weakref.ref(img)
        if hasattr(img, "sigImageChanged"):
            image_changed_funct = partial(self.imageChanged, autoLevel=self.autoLevel)
            img.sigImageChanged.connect(image_changed_funct)
        self._setImageLookupTable()
        self.imageChanged(
            autoLevel=False,
            min_percentile=min_percentile,
            max_percentile=max_percentile,
        )

    @QtCore.Slot()
    def viewRangeChanged(self):
        self.update()

    @QtCore.Slot()
    def gradientChanged(self):
        if self.imageItem() is not None:
            self._setImageLookupTable()

        self.lut = None
        self.sigLookupTableChanged.emit(self)

    def _setImageLookupTable(self):
        if not self._show_gradient:
            # The Plot owns the LUT (Colormap combo + gamma); clearing it
            # here would stomp the colormap on every image rebind.
            return
        if self.gradient.isLookupTrivial():
            self.imageItem().setLookupTable(None)
        else:
            self.imageItem().setLookupTable(self.getLookupTable)

    def getLookupTable(self, img=None, n=None, alpha=None):
        """Return a lookup table from the color gradient defined by this
        HistogramLUTItem.
        """
        if self.levelMode != "mono":
            return None
        if n is None:
            if img.dtype == np.uint8:
                n = 256
            else:
                n = 512
        if self.lut is None:
            self.lut = self.gradient.getLookupTable(n, alpha=alpha)
        return self.lut

    @QtCore.Slot()
    def regionChanged(self):
        if self.imageItem() is not None:
            self.imageItem().setLevels(self.getLevels())
        self._update_gamma_curve()
        self.sigLevelChangeFinished.emit(self)

    @QtCore.Slot()
    def regionChanging(self):
        if self.imageItem() is not None:
            self.imageItem().setLevels(self.getLevels())
        self._update_gamma_curve()
        self.update()
        self.sigLevelsChanged.emit(self)

    @QtCore.Slot()
    def imageChanged(
        self,
        autoLevel=False,
        autoRange=False,
        min_percentile=None,
        max_percentile=None,
    ):
        if self.imageItem() is None:
            return

        # Store latest args and restart debounce timer
        self._pendingImageChange.update(
            {
                "autoLevel": autoLevel,
                "autoRange": autoRange,
                "min_percentile": min_percentile,
                "max_percentile": max_percentile,
            }
        )
        self._imageChangedTimer.start()

    def _processImageChanged(self):

        if self.imageItem() is None:
            return

        autoLevel = self._pendingImageChange.get("autoLevel", False)
        autoRange = self._pendingImageChange.get("autoRange", False)
        min_percentile = self._pendingImageChange.get("min_percentile", None)
        max_percentile = self._pendingImageChange.get("max_percentile", None)

        # getHistogram indexes image.shape[1]; a non-2D image (e.g. a plot's
        # signal was swapped/retyped under the histogram, leaving a stale 1-D
        # image) would raise IndexError. Skip until a 2-D image is bound.
        _img = self.imageItem().image if self.imageItem() is not None else None
        if _img is None or getattr(_img, "ndim", 0) < 2:
            return

        if self.levelMode == "mono":
            for plt in self.plots[1:]:
                plt.setVisible(False)
            # plot one histogram for all image data
            profiler = debug.Profiler()
            if self.bar_item is not None:
                # modest bin count so the bars read as a bar chart, not as
                # gappy needles
                h = self.imageItem().getHistogram(bins=128)
            else:
                h = self.imageItem().getHistogram()
            self.bins = h[0]
            self.counts = h[1]
            profiler("get histogram")
            if h[0] is None:
                return
            if self.bar_item is not None:
                self.plots[0].setVisible(False)
                bins = np.asarray(h[0], dtype=float)
                counts = np.asarray(h[1], dtype=float)
                binw = float(bins[1] - bins[0]) if len(bins) > 1 else 1.0
                self.bar_item.setOpts(x=bins + binw / 2.0, height=counts,
                                      width=binw)
                # min/max drag lines can never leave the image's data range
                lo = float(bins[0])
                hi = float(bins[-1]) + binw
                self.region.setBounds([lo, hi])
                # Fixed view, refit on every image redraw. The top 0.5 % of
                # bar heights are treated as outliers so one dominant bin
                # (e.g. a zero peak) doesn't squash the rest of the chart.
                if counts.size:
                    ymax = float(np.percentile(counts, 99.5))
                    if ymax <= 0:
                        ymax = float(counts.max()) or 1.0
                else:
                    ymax = 1.0
                self._disp_hmax = ymax
                self.vb.setXRange(lo, hi, padding=0.01)
                self.vb.setYRange(0.0, ymax * 1.05, padding=0.0)
            else:
                self.plots[0].setVisible(True)
                self.plot.setData(*h)
            profiler("set plot")
            if autoLevel:
                mn = h[0][0]
                mx = h[0][-1]
                self.region.setRegion([mn, mx])
                profiler("set region")
            elif min_percentile is not None and max_percentile is not None:
                mn, mx = self.percentile2levels(min_percentile, max_percentile)
                self.region.setRegion([mn, mx])
                profiler("set region by percentile")
            else:
                mn, mx = self.imageItem().getLevels()
                self.region.setRegion([float(mn), float(mx)])
            self._sanitize_levels()
            self._update_gamma_curve()
        else:
            # plot one histogram for each channel
            self.plots[0].setVisible(False)
            ch = self.imageItem().getHistogram(perChannel=True)
            if ch[0] is None:
                return
            for i in range(1, 5):
                if len(ch) >= i:
                    h = ch[i - 1]
                    self.plots[i].setVisible(True)
                    self.plots[i].setData(*h)
                    if autoLevel:
                        mn = h[0][0]
                        mx = h[0][-1]
                        self.regions[i].setRegion([mn, mx])
                else:
                    # hide channels not present in image data
                    self.plots[i].setVisible(False)
            # make sure we are displaying the correct number of channels
            self._showRegions()

    def percentile2levels(self, min_percentile, max_percentile):
        """Get levels based on percentiles of the image histogram.

        Parameters
        ----------
        min_percentile : float
            Minimum percentile (0-100).
        max_percentile : float
            Maximum percentile (0-100).

        Returns
        -------
        levels : tuple
            (min, max) levels corresponding to the requested percentiles.
        """
        if self.bins is None or self.counts is None:
            return None

        cumsum = np.cumsum(self.counts)
        total = cumsum[-1]
        if total <= 0:
            return None
        low_idx = np.searchsorted(cumsum, (min_percentile / 100.0) * total)
        high_idx = np.searchsorted(cumsum, (max_percentile / 100.0) * total)

        min_level = self.bins[low_idx] if low_idx < len(self.bins) else self.bins[-1]
        max_level = self.bins[high_idx] if high_idx < len(self.bins) else self.bins[-1]

        return (min_level, max_level)

    def get_percentile_levels(self):
        """From the current levels, get the corresponding percentiles."""
        if self.bins is None or self.counts is None:
            return None

        mn, mx = self.getLevels()
        cumsum = np.cumsum(self.counts)
        total = cumsum[-1]

        low_idx = np.searchsorted(self.bins, mn)
        high_idx = np.searchsorted(self.bins, mx)

        min_percentile = (
            (cumsum[low_idx] / total) * 100.0 if low_idx < len(cumsum) else 100.0
        )
        max_percentile = (
            (cumsum[high_idx] / total) * 100.0 if high_idx < len(cumsum) else 100.0
        )

        return (min_percentile, max_percentile)

    def _sanitize_levels(self):
        """Reset the min/max lines when the current levels are useless.

        Stale levels (e.g. the PlotState default 0..1 against 0..4000 data,
        or levels carried over from another image) end up clamped into a
        sliver at the edge of the new range — the lines look compressed at
        the bottom and the image is washed out. Detect that and fall back to
        robust auto levels (min → 99.5th percentile, bright outliers like the
        direct beam excluded).
        """
        if self.bar_item is None or self.bins is None or len(self.bins) < 2:
            return
        lo = float(self.bins[0])
        hi = float(self.bins[-1]) + float(self.bins[1] - self.bins[0])
        span = hi - lo
        if span <= 0:
            return
        try:
            mn, mx = self.region.getRegion()
        except Exception:
            return
        # Reset only clearly-broken levels: degenerate, fully outside the
        # data range, or a <0.1% sliver (the stale-default case). A user's
        # deliberately narrow contrast window stays untouched.
        usable = (
            mx > mn
            and mx > lo
            and mn < hi
            and (mx - mn) >= 0.001 * span
        )
        if usable:
            return
        levels = self.percentile2levels(0.0, 99.5)
        if levels is None:
            return
        mn2, mx2 = float(levels[0]), float(levels[1])
        if mx2 <= mn2:
            mn2, mx2 = lo, hi
        self.region.setRegion([mn2, mx2])
        if self.imageItem() is not None:
            self.imageItem().setLevels((mn2, mx2))

    # ── Gamma transfer line ───────────────────────────────────────────────────

    def _hist_height(self):
        """Display height the gamma curve spans (outlier-clipped y-scale)."""
        h = float(getattr(self, "_disp_hmax", 1.0))
        return h if h > 0 else 1.0

    def _update_gamma_curve(self):
        """Redraw the gamma transfer line between the current min/max levels.

        For gamma == 1 this is the diagonal from (min, 0) to (max, top);
        other gammas bow the line. Drag the line vertically to set gamma.
        """
        # getattr: region signals can fire during __init__ before the gamma
        # items are constructed
        if getattr(self, "gamma_curve", None) is None or self.counts is None:
            return
        try:
            mn, mx = self.region.getRegion()
        except Exception:
            return
        mn, mx = float(mn), float(mx)
        if mx <= mn:
            return
        hmax = self._hist_height()
        x = np.linspace(mn, mx, 128)
        y = ((x - mn) / (mx - mn)) ** self.gamma * hmax
        self.gamma_curve.setData(x, y)

    def _on_gamma_dragged(self, gamma):
        self.gamma = float(gamma)
        self._update_gamma_curve()
        self.sigGammaChanged.emit(self)

    def set_gamma(self, gamma, emit=False):
        """Set the display gamma (clamped) and update the transfer line."""
        self.gamma = float(min(max(float(gamma), 0.05), 20.0))
        self._update_gamma_curve()
        if emit:
            self.sigGammaChanged.emit(self)

    def getLevels(self):
        """Return the min and max levels.

        For rgba mode, this returns a list of the levels for each channel.
        """
        if self.levelMode == "mono":
            return self.region.getRegion()
        else:
            nch = self.imageItem().channels()
            if nch is None:
                nch = 3
            return [r.getRegion() for r in self.regions[1 : nch + 1]]

    def setLevels(self, min=None, max=None, rgba=None):
        """Set the min/max (bright and dark) levels.

        Parameters
        ----------
        min : float, optional
            Minimum level.
        max : float, optional
            Maximum level.
        rgba : list, optional
            Sequence of (min, max) pairs for each channel for 'rgba' mode.
        """
        if None in {min, max} and (rgba is None or None in rgba[0]):
            raise ValueError("Must specify min and max levels")

        if self.levelMode == "mono":
            if min is None:
                min, max = rgba[0]
            self.region.setRegion((min, max))
        else:
            if rgba is None:
                rgba = 4 * [(min, max)]
            for levels, region in zip(rgba, self.regions[1:]):
                region.setRegion(levels)

    def setLevelMode(self, mode):
        """Set the method of controlling the image levels offered to the user.

        Options are 'mono' or 'rgba'.
        """
        if mode not in {"mono", "rgba"}:
            raise ValueError(
                f"Level mode must be one of {{'mono', 'rgba'}}, got {mode}"
            )

        if mode == self.levelMode:
            return

        oldLevels = self.getLevels()
        self.levelMode = mode
        self._showRegions()

        # do our best to preserve old levels
        if mode == "mono":
            levels = np.array(oldLevels).mean(axis=0)
            self.setLevels(*levels)
        else:
            levels = [oldLevels] * 4
            self.setLevels(rgba=levels)

        # force this because calling self.setLevels might not set the imageItem
        # levels if there was no change to the region item
        if self.imageItem() is not None:
            self.imageItem().setLevels(self.getLevels())

        self.imageChanged(autoLevel=self.autoLevel)
        self.update()

    def _showRegions(self):
        for i in range(len(self.regions)):
            self.regions[i].setVisible(False)

        if self.levelMode == "rgba":
            nch = 4
            if self.imageItem() is not None:
                # Only show rgb channels if connected image lacks alpha.
                nch = self.imageItem().channels()
                if nch is None:
                    nch = 3
            xdif = 1.0 / nch
            for i in range(1, nch + 1):
                self.regions[i].setVisible(True)
                self.regions[i].setSpan((i - 1) * xdif, i * xdif)
            self.gradient.hide()
        elif self.levelMode == "mono":
            self.regions[0].setVisible(True)
            self.gradient.show()
        else:
            raise ValueError(f"Unknown level mode {self.levelMode}")

    def saveState(self):
        return {
            "gradient": self.gradient.saveState(),
            "levels": self.getLevels(),
            "mode": self.levelMode,
        }

    def restoreState(self, state):
        if "mode" in state:
            self.setLevelMode(state["mode"])
        self.gradient.restoreState(state["gradient"])
        self.setLevels(*state["levels"])


"""
Widget displaying an image histogram along with gradient editor. Can be used to adjust
the appearance of images. This is a wrapper around HistogramLUTItem
"""


class HistogramLUTWidget(GraphicsView):
    """QWidget wrapper for :class:`~pyqtgraph.HistogramLUTItem`.

    All parameters are passed along in creating the HistogramLUTItem.
    """

    def __init__(self, parent=None, *args, **kargs):
        background = kargs.pop("background", "default")
        GraphicsView.__init__(self, parent, useOpenGL=False, background=background)
        self.item = HistogramLUTItem(*args, **kargs)
        self.setCentralItem(self.item)

        self.orientation = kargs.get("orientation", "vertical")
        if self.orientation == "vertical":
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Preferred,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            self.setMinimumWidth(95)
        else:
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
            self.setMinimumHeight(95)

    def sizeHint(self):
        if self.orientation == "vertical":
            return QtCore.QSize(115, 200)
        else:
            return QtCore.QSize(200, 115)

    def __getattr__(self, attr):
        return getattr(self.item, attr)
