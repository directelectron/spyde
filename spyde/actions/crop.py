"""
crop.py — Crop: cut a dataset down on any of its axes, as a new dataset.

Every axis of the displayed signal gets a range — the image or detector, the
scan, the time axis of a movie — and every plot that draws one of those axes
gets a box on it: a rectangle on the signal plot and on a scan navigator, a
span on a time navigator. The panel's sliders and fields, and those on-plot
boxes, are views of the ONE set of ranges a :class:`CropController` holds, so
moving any of them moves the rest.

The result opens as a NEW dataset rather than as a node of this one. The
source's navigators draw the source's scan and time extent; a node with a
smaller scan or fewer frames would be navigated on a grid it does not have,
and positions outside it would show the nearest edge frame. A new dataset gets
navigators of its own extent. Slicing is lazy (hyperspy ``inav`` / ``isig``),
so cropping reads nothing.

Ranges are axis INDICES, ``[start, stop)``, keyed ``nav<i>`` / ``sig<i>`` in
hyperspy's display order — the order ``inav`` and ``isig`` take.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from de_shell.ipc import emit, emit_error, emit_status

from spyde.actions.context import current_signal, src_plot_tree

log = logging.getLogger(__name__)

CROP_COLOR = "#f9e2af"

# ── Cropping a signal ────────────────────────────────────────────────────────

def clamp_range(start, stop, size: int) -> tuple[int, int]:
    """``[start, stop)`` inside ``[0, size)`` and at least one index wide."""
    start = max(0, min(int(start), size - 1))
    stop = max(start + 1, min(int(stop), size))
    return start, stop


def crop_signal(signal, ranges: dict):
    """*signal* cut to *ranges* (``{"nav0": (start, stop), ...}``).

    An axis with no entry, or one whose range covers it whole, is kept whole.
    Returns *signal* itself when nothing is cut, so a caller can tell a crop
    from a no-op by identity.
    """
    am = signal.axes_manager
    navigation = [slice(None)] * am.navigation_dimension
    signal_slices = [slice(None)] * am.signal_dimension
    for key, (start, stop) in ranges.items():
        slices = navigation if key.startswith("nav") else signal_slices
        index = int(key[3:])
        size = int((am.navigation_axes if key.startswith("nav")
                    else am.signal_axes)[index].size)
        start, stop = clamp_range(start, stop, size)
        if (start, stop) != (0, size):
            slices[index] = slice(start, stop)
    cut = signal
    if any(each != slice(None) for each in navigation):
        cut = cut.inav[tuple(navigation)]
    if any(each != slice(None) for each in signal_slices):
        cut = cut.isig[tuple(signal_slices)]
    return cut


def time_axis_index(signal) -> int | None:
    """Which navigation axis a time navigator draws: the only one of a movie,
    the slowest of a time series of scans. ``None`` for a scan (two axes)."""
    count = signal.axes_manager.navigation_dimension
    if count == 1:
        return 0
    if count >= 3:
        return count - 1
    return None


# ── The axes a crop has ──────────────────────────────────────────────────────

@dataclass
class CropAxis:
    key: str            # "nav0" / "sig1": the inav / isig index, display order
    axis: object        # the hyperspy axis it crops

    @property
    def size(self) -> int:
        return int(self.axis.size)

    def describe(self, start: int, stop: int) -> dict:
        from spyde.drawing.plots.plot import _clean_units
        from spyde.metadata_extract import _clean
        units = str(getattr(self.axis, "units", "") or "")
        units = "" if units in ("<undefined>", "px") else units
        return {
            "key": self.key, "navigate": self.key.startswith("nav"),
            "name": _clean(getattr(self.axis, "name", "")), "size": self.size,
            "start": start, "stop": stop,
            "scale": float(getattr(self.axis, "scale", 1.0) or 1.0),
            "offset": float(getattr(self.axis, "offset", 0.0) or 0.0),
            "units": _clean_units(units) if units else "",
        }


def crop_axes(signal) -> list[CropAxis]:
    """One :class:`CropAxis` per axis of *signal*, in the axes manager's own
    order — the order, and the names, the dock's Axes table shows."""
    am = signal.axes_manager
    navigation, signal_axes = list(am.navigation_axes), list(am.signal_axes)

    def key(axis) -> str:
        for prefix, axes in (("nav", navigation), ("sig", signal_axes)):
            for index, each in enumerate(axes):
                if each is axis:
                    return f"{prefix}{index}"
        raise ValueError(f"{axis} is not an axis of the signal")

    return [CropAxis(key(axis), axis) for axis in am._axes]


def shape_text(signal) -> str:
    """``nav 6 · sig 2048 × 2048`` — the dock's dataset badge, display order."""
    am = signal.axes_manager
    parts = []
    if am.navigation_dimension:
        parts.append("nav " + " × ".join(str(int(n)) for n in am.navigation_shape))
    parts.append("sig " + " × ".join(str(int(n)) for n in am.signal_shape))
    return " · ".join(parts)


def _cropped_shape_text(signal, ranges: dict) -> str:
    am = signal.axes_manager

    def sizes(prefix, count):
        return [ranges[f"{prefix}{index}"][1] - ranges[f"{prefix}{index}"][0]
                for index in range(count)]

    parts = []
    if am.navigation_dimension:
        parts.append("nav " + " × ".join(
            str(n) for n in sizes("nav", am.navigation_dimension)))
    parts.append("sig " + " × ".join(
        str(n) for n in sizes("sig", am.signal_dimension)))
    return " · ".join(parts)


# ── Plot coordinates ─────────────────────────────────────────────────────────

@dataclass
class _EdgeMap:
    """Plot coordinate of crop edge *e*: ``offset + scale * e``.

    Edge ``e`` sits between index ``e - 1`` and ``e``. anyplotlib puts integer
    coordinates at pixel CENTRES, so an image edge is at ``e - 0.5``; a line
    plot's edge is half a step before the value at index ``e``.
    """
    offset: float
    scale: float

    def to_plot(self, edge: float) -> float:
        return self.offset + self.scale * edge

    def to_edge(self, coordinate: float) -> float:
        return (coordinate - self.offset) / self.scale


def _through_root(node_axis, root_axis, plot_map: _EdgeMap) -> _EdgeMap | None:
    """A navigator draws the ROOT's navigation grid. Carry a node edge onto
    it through the calibration, which hyperspy keeps across ``inav`` and
    ``rebin`` — the same translation the navigator read makes."""
    if node_axis is root_axis:
        return plot_map
    if not (getattr(node_axis, "is_uniform", False)
            and getattr(root_axis, "is_uniform", False) and root_axis.scale):
        return None
    ratio = node_axis.scale / root_axis.scale
    root_edge_at_zero = 0.5 + (node_axis.offset - 0.5 * node_axis.scale
                               - root_axis.offset) / root_axis.scale
    return _EdgeMap(plot_map.offset + plot_map.scale * root_edge_at_zero,
                    plot_map.scale * ratio)


def _image_map() -> _EdgeMap:
    return _EdgeMap(-0.5, 1.0)


def _line_map(plot, size: int) -> _EdgeMap:
    """Edge map of a line plot from the x values it actually draws."""
    try:
        x, _units, _label = plot._axes_info_1d(np.zeros(size))
    except Exception:
        x = None
    if x is None or len(x) < 1:
        return _EdgeMap(-0.5, 1.0)
    step = float(x[1] - x[0]) if len(x) > 1 else 1.0
    return _EdgeMap(float(x[0]) - 0.5 * step, step)


class _Box:
    """One on-plot crop box: a rectangle over two axes, or a span over one."""

    def __init__(self, plot, keys, maps, widget):
        self.plot = plot
        self.keys = keys
        self.maps = maps
        self.widget = widget
        self.handler = None
        self.placed = None      # the ranges it was last put at

    def read(self) -> dict:
        """The box's geometry as edges, ``{key: (low, high)}``."""
        widget = self.widget
        if len(self.keys) == 2:
            spans = ((float(widget.x), float(widget.x) + float(widget.w)),
                     (float(widget.y), float(widget.y) + float(widget.h)))
        else:
            spans = ((float(widget.x0), float(widget.x1)),)
        edges = {}
        for key, edge_map, (low, high) in zip(self.keys, self.maps, spans):
            low, high = sorted((edge_map.to_edge(low), edge_map.to_edge(high)))
            edges[key] = (low, high)
        return edges

    def place(self, ranges: dict) -> None:
        wanted = tuple(ranges[key] for key in self.keys)
        if wanted == self.placed:
            return
        self.placed = wanted
        coordinates = [(edge_map.to_plot(ranges[key][0]),
                        edge_map.to_plot(ranges[key][1]))
                       for key, edge_map in zip(self.keys, self.maps)]
        if len(self.keys) == 2:
            (x0, x1), (y0, y1) = coordinates
            self.widget.set(_notify=False, x=x0, y=y0, w=x1 - x0, h=y1 - y0)
        else:
            (x0, x1), = coordinates
            self.widget.set(_notify=False, x0=x0, x1=x1)
        # A targeted widget move updates what the figure DRAWS but not the
        # panel state it keeps; the next time it reloads that state (a click
        # in the plot does) the box jumps back to where it was created, and
        # a drag starts from there. Re-sending the panel keeps the two equal.
        owner = getattr(self.widget, "_plot", None)
        if owner is not None:
            owner._push()

    def hide(self) -> None:
        try:
            self.widget.hide()
        except Exception as e:
            log.debug("hiding a crop box failed: %s", e)


# ── The controller ───────────────────────────────────────────────────────────

class CropController:
    """The ranges of one Crop panel and the boxes that show them.

    Parked on the tree as ``_crop_controller``; one per tree.
    """

    def __init__(self, session, plot, tree, signal):
        self.session = session
        self.plot = plot
        self.tree = tree
        self.signal = signal
        self.axes = crop_axes(signal)
        self.ranges = {axis.key: (0, axis.size) for axis in self.axes}
        self.boxes: list[_Box] = []
        self._placing = False
        self._closed = False

    # -- state --

    def _axis(self, key) -> CropAxis | None:
        return next((axis for axis in self.axes if axis.key == key), None)

    def set_ranges(self, ranges: dict) -> bool:
        """Apply ``{key: (start, stop)}``; returns whether anything moved."""
        changed = False
        for key, (start, stop) in ranges.items():
            axis = self._axis(key)
            if axis is None:
                continue
            new = clamp_range(round(float(start)), round(float(stop)), axis.size)
            if new != self.ranges[key]:
                self.ranges[key] = new
                changed = True
        return changed

    def reset(self) -> None:
        self.ranges = {axis.key: (0, axis.size) for axis in self.axes}

    @property
    def is_crop(self) -> bool:
        return any(self.ranges[axis.key] != (0, axis.size) for axis in self.axes)

    def state(self) -> dict:
        return {
            "type": "crop_state",
            "window_id": getattr(self.plot, "window_id", None),
            "axes": [axis.describe(*self.ranges[axis.key]) for axis in self.axes],
            "before": shape_text(self.signal),
            "after": _cropped_shape_text(self.signal, self.ranges),
            "is_crop": self.is_crop,
            "boxes": len(self.boxes),
        }

    def emit_state(self) -> None:
        emit(self.state())

    # -- boxes --

    def open(self) -> None:
        for plot in self._signal_plots():
            self._add_signal_box(plot)
        for plot in self._navigator_plots():
            self._add_navigator_box(plot)
        self.place_boxes()
        self.emit_state()

    def _signal_plots(self) -> list:
        """Every plot of this tree showing the cropped signal, the panel's own
        first."""
        plots = [self.plot] + [
            plot for plot in self.session._plots
            if plot is not self.plot and not plot.is_navigator
            and getattr(plot, "signal_tree", None) is self.tree
            and current_signal(plot) is self.signal]
        return [plot for plot in plots if plot is not None]

    def _navigator_plots(self) -> list:
        return [plot for plot in self.session._plots
                if plot.is_navigator and getattr(plot, "signal_tree", None) is self.tree]

    def _add_signal_box(self, plot) -> None:
        keys = [axis.key for axis in self.axes if axis.key.startswith("sig")]
        keys.sort()
        if len(keys) == 2 and getattr(plot, "_plot2d", None) is not None:
            self._add_box(plot, keys, [_image_map(), _image_map()])
        elif len(keys) == 1 and getattr(plot, "_plot1d", None) is not None:
            axis = self._axis(keys[0])
            self._add_box(plot, keys, [_line_map(plot, axis.size)])

    def _add_navigator_box(self, plot) -> None:
        root = getattr(self.tree, "root", None)
        if root is None or (root.axes_manager.navigation_dimension
                            != self.signal.axes_manager.navigation_dimension):
            return
        node_axes = list(self.signal.axes_manager.navigation_axes)
        root_axes = list(root.axes_manager.navigation_axes)
        if getattr(plot, "_plot2d", None) is not None and len(node_axes) >= 2:
            indices = (0, 1)
            plot_maps = [_image_map(), _image_map()]
        elif getattr(plot, "_plot1d", None) is not None:
            index = time_axis_index(self.signal)
            if index is None:
                return
            indices = (index,)
            plot_maps = [_line_map(plot, int(root_axes[index].size))]
        else:
            return
        maps = [_through_root(node_axes[index], root_axes[index], plot_map)
                for index, plot_map in zip(indices, plot_maps)]
        if any(edge_map is None for edge_map in maps):
            return
        self._add_box(plot, [f"nav{index}" for index in indices], maps)

    def _add_box(self, plot, keys, maps) -> None:
        from spyde.drawing.selectors.base_selector import event_handler_fn
        try:
            if len(keys) == 2:
                widget = plot._plot2d.add_rectangle_widget(
                    x=0.0, y=0.0, w=1.0, h=1.0, color=CROP_COLOR,
                    show_handles=True)
            else:
                axis = self._axis(keys[0])
                edges = [maps[0].to_plot(edge) for edge in range(axis.size + 1)]
                widget = plot._plot1d.add_range_widget(
                    edges[0], edges[-1], color=CROP_COLOR, snap_values=edges)
        except Exception as e:
            log.debug("adding a crop box to %s failed: %s", plot, e)
            return
        box = _Box(plot, keys, maps, widget)
        box.handler = event_handler_fn(lambda event, _box=box: self._on_drag(_box, event))
        widget.add_event_handler(box.handler, "pointer_move", "pointer_up")
        self.boxes.append(box)

    def place_boxes(self, skip: _Box | None = None) -> None:
        self._placing = True
        try:
            for box in self.boxes:
                if box is not skip:
                    try:
                        box.place(self.ranges)
                    except Exception as e:
                        log.debug("placing a crop box failed: %s", e)
        finally:
            self._placing = False

    def _kept_inside(self, edges: dict) -> dict:
        """A box MOVED past the end of its axis stops there at its own size,
        rather than being cut down by the clamp; a resize is left alone."""
        inside = {}
        for key, (low, high) in edges.items():
            axis = self._axis(key)
            start, stop = self.ranges.get(key, (low, high))
            if axis is not None and abs((high - low) - (stop - start)) < 1e-6:
                if low < 0:
                    low, high = 0.0, high - low
                elif high > axis.size:
                    low, high = low - (high - axis.size), float(axis.size)
            inside[key] = (low, high)
        return inside

    def _on_drag(self, box: _Box, event) -> None:
        if self._closed or self._placing:
            return
        try:
            edges = box.read()
        except Exception as e:
            log.debug("reading a crop box failed: %s", e)
            return
        box.placed = None
        changed = self.set_ranges(self._kept_inside(edges))
        released = getattr(event, "event_type", "") == "pointer_up"
        # Leave the box under the cursor alone until it is let go, then snap
        # it onto whole pixels / frames with everything else.
        self.place_boxes(skip=None if released else box)
        if changed or released:
            self.emit_state()

    # -- result --

    def run(self):
        """Open the cropped dataset; returns its tree, or ``None`` for a no-op."""
        if not self.is_crop:
            emit_status("Crop: every range covers its whole axis — nothing to cut")
            return None
        from spyde.actions.commit import open_result_tree

        cropped = crop_signal(self.signal, self.ranges)
        title = f"{_signal_title(self.signal)} (cropped)"
        before = shape_text(self.signal)
        tree = open_result_tree(
            self.session, title=title, signal=cropped,
            navigator_override=cropped_navigator(self.tree, self.signal,
                                                 cropped, self.ranges),
            provenance={"action": "Crop", "source": _signal_title(self.signal),
                        "ranges": {key: list(value)
                                   for key, value in self.ranges.items()}})
        self.close()
        emit_status(f"Cropped {before} → {shape_text(cropped)}")
        return tree

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for box in self.boxes:
            box.hide()
        self.boxes = []


def _signal_title(signal) -> str:
    title = signal.metadata.get_item("General.title", default="") or ""
    title = str(title)
    return "Signal" if title in ("", "<undefined>") else title


def cropped_navigator(tree, source, cropped, ranges: dict):
    """The source's navigator cut to the crop, or ``None`` to compute one.

    Only when it is exact: the source is the tree's root (its navigator is the
    root's), the detector is kept whole (the navigator sums over it), and the
    navigator is finished (no NaN from a fill still running). Then the cut is
    a slice of an array already in memory, where computing it would read the
    whole cropped dataset.
    """
    am = source.axes_manager
    count = am.navigation_dimension
    if count == 0 or source is not getattr(tree, "root", None):
        return None
    if any(ranges[f"sig{index}"] != (0, int(axis.size))
           for index, axis in enumerate(am.signal_axes)):
        return None
    navigators = (getattr(tree, "navigator_signals", None) or {}).get("base")
    if not navigators:
        return None
    full = getattr(navigators[-1], "data", None)
    data_shape = tuple(int(n) for n in am.navigation_shape[::-1])
    if (not isinstance(full, np.ndarray) or full.shape != data_shape
            or not np.isfinite(full).all()):
        return None
    # ranges are display order (fastest axis first); the array is data order.
    slices = tuple(slice(*ranges[f"nav{index}"]) for index in range(count))[::-1]
    import hyperspy.api as hs
    kind = {1: hs.signals.Signal1D, 2: hs.signals.Signal2D}.get(
        count, hs.signals.BaseSignal)
    navigator = kind(np.ascontiguousarray(full[slices]))
    for target, axis in zip(navigator.axes_manager.signal_axes,
                            cropped.axes_manager.navigation_axes):
        for attribute in ("name", "scale", "offset", "units"):
            try:
                setattr(target, attribute, getattr(axis, attribute))
            except Exception as e:
                log.debug("copying navigator %s failed: %s", attribute, e)
    return navigator


# ── Staged handlers (crop_*) ─────────────────────────────────────────────────

def crop(ctx, action_name: str = "Crop", **kwargs):
    """Parent toolbar action — a no-op; the Electron toolbar opens the Crop
    panel, which drives the ``crop_*`` handlers."""
    return None


def _controller(session, plot) -> CropController | None:
    _src, tree = src_plot_tree(session, plot)
    controller = getattr(tree, "_crop_controller", None) if tree is not None else None
    return None if controller is None or controller._closed else controller


def crop_open(session, plot, payload) -> None:
    """Panel mounted: put a box on every plot of the displayed signal's axes
    and send the ranges."""
    src, tree = src_plot_tree(session, plot)
    signal = current_signal(src)
    if src is None or tree is None or signal is None:
        emit_error("Crop: no dataset to crop")
        return
    close_crop(tree)
    controller = CropController(session, src, tree, signal)
    tree._crop_controller = controller
    controller.open()


def crop_close(session, plot, payload=None) -> None:
    _src, tree = src_plot_tree(session, plot)
    close_crop(tree)


def crop_set_range(session, plot, payload) -> None:
    """A field or slider edit: ``{key, start, stop}`` or ``{ranges: {...}}``."""
    controller = _controller(session, plot)
    if controller is None:
        return
    ranges = dict(payload.get("ranges") or {})
    if "key" in payload:
        ranges[str(payload["key"])] = (payload.get("start", 0),
                                       payload.get("stop", 0))
    controller.set_ranges(ranges)
    controller.place_boxes()
    controller.emit_state()


def crop_reset(session, plot, payload=None) -> None:
    controller = _controller(session, plot)
    if controller is None:
        return
    controller.reset()
    controller.place_boxes()
    controller.emit_state()


def crop_run(session, plot, payload=None) -> None:
    controller = _controller(session, plot)
    if controller is None:
        emit_error("Crop: open the Crop panel first")
        return
    controller.run()


def close_crop(tree) -> None:
    """Take down *tree*'s crop boxes (harmless when there are none)."""
    controller = getattr(tree, "_crop_controller", None) if tree is not None else None
    if controller is not None:
        controller.close()
        tree._crop_controller = None


def follow_node(tree, plot) -> None:
    """The window switched node: rebuild an open panel for the new signal,
    since its ranges and boxes described the one it left."""
    controller = getattr(tree, "_crop_controller", None) if tree is not None else None
    if controller is None or controller._closed:
        return
    session = controller.session
    close_crop(tree)
    crop_open(session, plot or controller.plot, {})
