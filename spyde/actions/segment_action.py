"""
segment_action.py — the Segment caret: paint, train, preview, run.

The caret opens on a 2-D window and puts an anyplotlib brush on it. Shift-drag
paints a stroke in the active class (particle / background / boundary); a
plain drag still pans. Train fits :class:`~spyde.segmentation.PixelClassifier`
on the strokes and paints its particle mask over the image; from then on every
new stroke retrains and repaints. Run labels every field — one image, every
frame of a movie, or the real-space navigator of a 4-D scan — measures the
instances, and opens the result as a new tree carrying ``tree.regions``.

Staged verbs (``registry.STAGED_HANDLERS``): ``seg_open``, ``seg_close``,
``seg_tune`` (brush class / radius / eraser and the split parameters),
``seg_clear``, ``seg_train``, ``seg_run``.

What a field is, and where the strokes land, is decided once in
:meth:`SegmentWizard.resolve` — the source plot is either a signal window
(image or movie: fields are frames, pixels are signal pixels) or a navigator
window (one field, pixels are scan positions). Everything after that is the
same code.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from de_shell.actions.wizard import WizardController
from de_shell.ipc import emit, emit_error, emit_progress, emit_status

from spyde.actions.context import current_signal as _current_signal
from spyde.actions.context import src_plot_tree as _src_plot_tree
from spyde.actions.lifecycle import attach_container, bump_generation
from spyde.segmentation import (
    CLASSES, PARTICLE, Labels, PixelClassifier, field_source, measure_instances,
    split_instances,
)
from spyde.signals.regions import NAVIGATION_SPACE, SIGNAL_SPACE, Regions

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "radius": 4.0,             # brush radius, image pixels
    "active_class": PARTICLE,
    "erase": False,
    "min_size": 20,            # pixels
    "split_touching": True,
    "min_separation": 3,       # pixels
}

SPLIT_KEYS = ("min_size", "split_touching", "min_separation")

#: The particle mask overlay.
MASK_COLOUR = CLASSES[PARTICLE].colour
MASK_ALPHA = 0.35

#: Result label maps are shown through a rainbow so neighbouring instances
#: read as different bodies; 0 (background) sits at the dark end.
LABEL_COLORMAP = "turbo"

#: The device the classifier runs on; unset means the best available. Tests
#: pin ``cpu`` because torch-CUDA work segfaults under pytest on Windows.
DEVICE_VARIABLE = "SPYDE_SEGMENT_DEVICE"


class SegmentWizard(WizardController):
    """The caret's state: the fields, the strokes, the classifier, the brush."""

    key = "seg"
    parameters = {
        "radius": {"name": "Brush (px)", "type": "float", "default": DEFAULTS["radius"],
                   "min": 1.0, "max": 64.0, "step": 1.0},
        "active_class": {"name": "Class", "type": "int",
                         "default": DEFAULTS["active_class"], "min": 0, "max": 2},
        "erase": {"name": "Erase", "type": "bool", "default": DEFAULTS["erase"]},
        "min_size": {"name": "Min size (px)", "type": "int",
                     "default": DEFAULTS["min_size"], "min": 0, "max": 100000},
        "split_touching": {"name": "Split touching", "type": "bool",
                           "default": DEFAULTS["split_touching"]},
        "min_separation": {"name": "Min separation (px)", "type": "int",
                           "default": DEFAULTS["min_separation"], "min": 1, "max": 100},
    }

    def __init__(self, session, tree, src_plot) -> None:
        super().__init__(session, tree)
        self.src_plot = src_plot
        self.params: dict[str, Any] = dict(DEFAULTS)
        self.source = None
        self.space = SIGNAL_SPACE
        self.scale, self.units = 1.0, "px"
        self.labels: Labels | None = None
        self.classifier: PixelClassifier | None = None
        self.result_tree = None
        self.running = False
        self._brush = None
        self._brush_handler = None
        self._brush_state = None
        self._strokes_seen = 0
        self._selectors: list = []
        self._preview_busy = False
        self._preview_again = False
        self._stopped: list | None = None

    # -- the dataset ---------------------------------------------------------

    @property
    def window_id(self):
        return getattr(self.src_plot, "window_id", None)

    def resolve(self) -> None:
        """Decide what the fields are. Raises ``TypeError`` with the reason."""
        if getattr(self.src_plot, "is_navigator", False):
            image = np.asarray(getattr(self.src_plot, "current_data", None))
            if image.ndim != 2:
                raise TypeError("the navigator is not a 2-D image yet")
            self.source = field_source(image)
            self.space = NAVIGATION_SPACE
            axes = self.tree.root.axes_manager.navigation_axes
            axis = axes[-1] if axes else None
        else:
            signal = self.signal()
            self.source = field_source(signal)
            self.space = SIGNAL_SPACE
            axis = signal.axes_manager.signal_axes[0]
        self.scale = float(getattr(axis, "scale", 1.0) or 1.0)
        self.units = str(getattr(axis, "units", "") or "px")
        if self.labels is not None and self.labels.shape != self.source.shape:
            self.labels = None

    def signal(self):
        """The displayed node: segmenting a rebinned or cropped view segments
        what the user is looking at."""
        return _current_signal(self.src_plot) or self.tree.root

    def field_index(self) -> int:
        """Where the movie's navigator is sitting; 0 for a single field."""
        if self.source is None or not self.source.is_movie:
            return 0
        for selector in self._navigator_selectors():
            indices = getattr(selector, "current_indices", None)
            if indices is not None:
                return int(np.atleast_1d(np.asarray(indices)).ravel()[0])
        return 0

    def current_field(self) -> np.ndarray:
        """The field on screen: the plot's own pixels when it shows one, else read."""
        shown = getattr(self.src_plot, "current_data", None)
        if isinstance(shown, np.ndarray) and shown.shape == self.source.shape:
            return shown
        return np.asarray(self.source.get(self.field_index()))

    def label_store(self) -> Labels:
        if self.labels is None:
            self.labels = Labels(self.source.shape)
        return self.labels

    def set_params(self, payload: dict | None, *, merge: bool = True) -> None:
        base = dict(self.params) if merge else dict(DEFAULTS)
        for name, default in DEFAULTS.items():
            if payload and name in payload:
                base[name] = type(default)(payload[name])
        self.params = base

    def split_params(self) -> dict[str, Any]:
        return {name: self.params[name] for name in SPLIT_KEYS}

    # -- the brush -----------------------------------------------------------

    def attach_brush(self) -> bool:
        plot2d = getattr(self.src_plot, "_plot2d", None)
        if plot2d is None or not hasattr(plot2d, "add_brush_widget"):
            return False
        if self._brush is not None:
            self.sync_brush()
            return True
        from spyde.drawing.selectors.base_selector import event_handler_fn
        self._brush = plot2d.add_brush_widget(
            radius=float(self.params["radius"]),
            colors=[label_class.colour for label_class in CLASSES],
            class_id=int(self.params["active_class"]),
            erase=bool(self.params["erase"]), alpha=0.55, active=True)
        # The widget reports a finished stroke once, on pointer_up; nothing
        # fires mid-stroke. The wrapper must be kept — the registration is weak.
        self._brush_handler = event_handler_fn(lambda event: self._on_stroke())
        self._brush.add_event_handler(self._brush_handler, "pointer_up")
        self._brush_state = (int(self.params["active_class"]), float(self.params["radius"]),
                             bool(self.params["erase"]))
        self._strokes_seen = 0
        return True

    def sync_brush(self) -> None:
        """Push the class, radius and eraser onto the widget.

        ``Widget.set`` sends a targeted update, and the renderer reads a
        stroke's class from the PANEL state at stroke start — which only a
        full panel push refreshes. So a changed state is followed by one full
        push; an unchanged one (every other tune) costs nothing.
        """
        if self._brush is None:
            return
        state = (int(self.params["active_class"]), float(self.params["radius"]),
                 bool(self.params["erase"]))
        if state == self._brush_state:
            return
        self._brush_state = state
        try:
            self._brush.set(_notify=False, class_id=state[0], radius=state[1], erase=state[2])
            push = getattr(getattr(self.src_plot, "_plot2d", None), "_push", None)
            if push is not None:
                push()
        except Exception as exc:
            log.debug("[seg] syncing the brush failed: %s", exc)

    def detach_brush(self) -> None:
        brush, self._brush, self._brush_handler = self._brush, None, None
        self._brush_state = None
        if brush is not None:
            try:
                brush.remove()
            except Exception as exc:
                log.debug("[seg] removing the brush failed: %s", exc)

    def _on_stroke(self) -> None:
        """A finished stroke: label its pixels, then retrain if already trained."""
        if self._closed or self._brush is None:
            return
        strokes = list(getattr(self._brush, "strokes", None) or ())
        fresh, self._strokes_seen = strokes[self._strokes_seen:], len(strokes)
        if not fresh:
            return
        # Python is the authority for the class: the strip's choice reached
        # ``params`` before the stroke finished, and the widget's own tag can
        # lag a targeted push. anyplotlib reports ``[x, y]`` in image pixels;
        # the store works in ``(y, x)``.
        field = self.field_index()
        radius = float(self.params["radius"])
        store = self.label_store()
        for stroke in fresh:
            points = [[float(point[1]), float(point[0])] for point in stroke if len(point) >= 2]
            if not points:
                continue
            if self.params["erase"]:
                store.erase(field, points, radius=radius)
            else:
                store.paint(field, points, int(self.params["active_class"]), radius=radius)
        _emit_state(self)
        if self.classifier is not None:
            _train(self)

    # -- the preview ---------------------------------------------------------

    def preview(self) -> None:
        """Paint the classifier's particle mask over the field on screen.

        One preview in flight at a time; a request that arrives while one runs
        re-fires once it lands, so a scrub through a movie shows the newest
        frame rather than every frame in order.
        """
        if self._closed or self.classifier is None or self.source is None:
            return
        if self._preview_busy:
            self._preview_again = True
            return
        self._preview_busy = True
        classifier = self.classifier
        field = self.current_field()
        generation = self.current_generation()

        def _work():
            particle, _boundary = classifier.foreground(field)
            return particle > 0.5

        def _done(mask):
            self._preview_busy = False
            if self._closed or not self.still(generation):
                return
            if self.classifier is classifier:
                self.src_plot.set_overlay_mask(mask, color=MASK_COLOUR, alpha=MASK_ALPHA)
            if self._preview_again:
                self._preview_again = False
                self.preview()

        def _fail(exc):
            self._preview_busy = False
            log.debug("[seg] preview failed: %s", exc)

        self.run_on_worker(_work, name="seg-preview", on_done=_done, on_error=_fail)

    def clear_preview(self) -> None:
        try:
            self.src_plot.set_overlay_mask(None)
        except Exception as exc:
            log.debug("[seg] clearing the mask failed: %s", exc)

    def current_generation(self) -> int:
        return int(getattr(self.tree, self._gen_key, 0))

    # -- following the navigator ---------------------------------------------

    def _navigator_selectors(self) -> list:
        manager = getattr(self.tree, "navigator_plot_manager", None)
        if manager is None:
            return []
        return list(manager.all_navigation_selectors)

    def wire_navigator(self) -> None:
        if self.source is None or not self.source.is_movie:
            return
        self._selectors = self._navigator_selectors()
        for selector in self._selectors:
            selector.index_hooks.append(self._on_indices)

    def unwire_navigator(self) -> None:
        for selector in self._selectors:
            if self._on_indices in selector.index_hooks:
                selector.index_hooks.remove(self._on_indices)
        self._selectors = []

    def _on_indices(self, _indices) -> None:
        """Runs on the navigator dispatcher thread: marshal, touch nothing."""
        if self._closed or self.classifier is None:
            return
        dispatch = getattr(self.session, "_dispatch_to_main", None)
        if dispatch is None:
            self.preview()
        else:
            dispatch(self.preview)

    # -- teardown ------------------------------------------------------------

    def remove(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stopped is not None:
            self._stopped[0] = True
        self.unwire_navigator()
        self.detach_brush()
        self.clear_preview()
        if getattr(self.tree, "_seg_wizard", None) is self:
            self.tree._seg_wizard = None


# ── messages ─────────────────────────────────────────────────────────────────

def _emit(wizard: SegmentWizard, message: dict) -> None:
    message.setdefault("window_id", wizard.window_id)
    emit(message)


def _emit_state(wizard: SegmentWizard) -> None:
    counts = wizard.labels.counts() if wizard.labels is not None else {}
    report = wizard.classifier.report if wizard.classifier is not None else {}
    _emit(wizard, {
        "type": "seg_state",
        "n_fields": int(wizard.source.count) if wizard.source is not None else 0,
        "field": wizard.field_index(),
        "space": wizard.space,
        "units": wizard.units,
        "classes": [{"id": label_class.id, "name": label_class.name,
                     "colour": label_class.colour,
                     "pixels": int(counts.get(label_class.id, 0))}
                    for label_class in CLASSES],
        "painted_fields": wizard.labels.painted_fields() if wizard.labels is not None else [],
        "trained": wizard.classifier is not None,
        "train_accuracy": float(report.get("train_accuracy", 0.0)),
        "running": wizard.running,
        "params": dict(wizard.params),
    })


def _wizard(session, plot) -> SegmentWizard | None:
    _src, tree = _src_plot_tree(session, plot)
    wizard = getattr(tree, "_seg_wizard", None) if tree is not None else None
    return wizard if wizard is not None and not wizard._closed else None


def _device():
    return os.environ.get(DEVICE_VARIABLE) or None


# ── the staged handlers ──────────────────────────────────────────────────────

def seg_open(session, plot, payload) -> None:
    """Caret mounted: put the brush on the window and report the state."""
    src, tree = _src_plot_tree(session, plot)
    if src is None or tree is None:
        emit_error("Segment: no active dataset")
        return
    existing = getattr(tree, "_seg_wizard", None)
    if existing is not None and not existing._closed:
        existing.set_params(payload)
        existing.sync_brush()
        _emit_state(existing)
        return
    wizard = SegmentWizard(session, tree, src)
    try:
        wizard.resolve()
    except TypeError as exc:
        emit_error(f"Segment: {exc}")
        return
    wizard.set_params(payload)
    wizard.guard()                      # BEFORE anything deferred (StrictMode)
    tree._seg_wizard = wizard
    wizard.wire_navigator()
    if wizard.attach_brush():
        emit_status("Shift+drag on the image to paint a class; plain drag still pans")
    else:
        emit_error("Segment: this window cannot take a brush")
    _emit_state(wizard)


def seg_close(session, plot, payload=None) -> None:
    """Caret unmounted: invalidate in-flight work FIRST, then tear down."""
    _src, tree = _src_plot_tree(session, plot)
    if tree is None:
        return
    bump_generation(tree, "_seg_run_gen")
    wizard = getattr(tree, "_seg_wizard", None)
    if wizard is not None:
        wizard.remove()


def seg_tune(session, plot, payload) -> None:
    """Brush class / radius / eraser, and the split parameters."""
    wizard = _wizard(session, plot)
    if wizard is None:
        return
    wizard.set_params(payload)
    wizard.sync_brush()


def seg_clear(session, plot, payload=None) -> None:
    """Forget every stroke and the classifier trained on them."""
    wizard = _wizard(session, plot)
    if wizard is None:
        return
    wizard.guard()
    if wizard.labels is not None:
        wizard.labels.clear()
    wizard.classifier = None
    if wizard._brush is not None:
        try:
            wizard._brush.clear_strokes()
        except Exception as exc:
            log.debug("[seg] clearing the brush failed: %s", exc)
        wizard._strokes_seen = 0
    wizard.clear_preview()
    _emit_state(wizard)


def seg_train(session, plot, payload=None) -> None:
    wizard = _wizard(session, plot)
    if wizard is None:
        emit_error("Segment: the caret is not open")
        return
    if wizard.labels is None or len(wizard.labels) == 0:
        emit_error("Segment: paint a particle and some background first")
        return
    _train(wizard)


def _train(wizard: SegmentWizard) -> None:
    """Fit on a snapshot of the strokes, then preview. Latest fit wins."""
    snapshot = Labels.from_dict(wizard.labels.to_dict())
    source = wizard.source
    generation = wizard.guard()

    def _work():
        classifier = PixelClassifier(device=_device())
        report = classifier.fit(snapshot, source.get)
        return classifier, report

    def _done(result):
        classifier, report = result
        if wizard._closed or not wizard.still(generation):
            return
        wizard.classifier = classifier
        emit_status(f"Segment: trained on {report['n_pixels']} px, "
                    f"accuracy {report['train_accuracy']:.3f}")
        _emit_state(wizard)
        wizard.preview()

    def _fail(exc):
        emit_error(f"Segment: training failed — {exc}")

    wizard.run_on_worker(_work, name="seg-train", on_done=_done, on_error=_fail)


def seg_run(session, plot, payload=None) -> None:
    """Label and measure every field on a worker, then open the result tree."""
    wizard = _wizard(session, plot)
    if wizard is None:
        emit_error("Segment: the caret is not open")
        return
    if wizard.classifier is None:
        emit_error("Segment: train first")
        return
    if wizard.running:
        return
    wizard.set_params(payload)
    classifier, source = wizard.classifier, wizard.source
    split, scale = wizard.split_params(), wizard.scale
    generation = wizard.guard()
    stopped = wizard.tree.register_cancel()
    wizard._stopped = stopped
    wizard.running = True
    wizard.tree._seg_batch_running = True       # read by lifecycle.wait_for_regions
    _emit_state(wizard)
    emit_status(f"Segmenting {source.count} field{'s' if source.count > 1 else ''}…")

    def _work():
        tables = []
        for index in range(source.count):
            if stopped[0]:
                break
            field = np.asarray(source.get(index))
            particle, boundary = classifier.foreground(field)
            labels = split_instances(particle, boundary, **split)
            tables.append(measure_instances(labels, field, scale=scale))
            emit_progress(index + 1, source.count, "Segmenting")
        return tables

    def _finish():
        wizard.running = False
        wizard.tree._seg_batch_running = False
        wizard.tree.unregister_cancel(flag=stopped)
        if wizard._stopped is stopped:
            wizard._stopped = None

    def _done(tables):
        _finish()
        if wizard._closed or not wizard.still(generation):
            return
        cancelled = len(tables) < source.count
        tables += [None] * (source.count - len(tables))
        regions = Regions.from_tables(
            tables, field_shape=source.shape, scale=scale, units=wizard.units,
            space=wizard.space, params=dict(split),
            provenance={"action": "Segment", "params": dict(wizard.params),
                        "labels": wizard.labels.to_dict() if wizard.labels else {},
                        "classifier": classifier.report})
        try:
            wizard.result_tree = _open_result(session, wizard, regions)
        except Exception as exc:
            emit_error(f"Segment: opening the result failed — {exc}")
            log.exception("segment result")
            return
        _emit(wizard, {"type": "seg_result", "n_regions": regions.n_regions,
                       "n_fields": regions.n_fields, "cancelled": cancelled})
        _emit_state(wizard)
        emit_status(f"Found {regions.n_regions} region"
                    f"{'s' if regions.n_regions != 1 else ''} in "
                    f"{regions.n_fields} field{'s' if regions.n_fields != 1 else ''}"
                    + (" (stopped early)" if cancelled else ""))

    def _fail(exc):
        _finish()
        emit_error(f"Segment failed: {exc}")
        log.exception("segment run")

    wizard.run_on_worker(_work, name="seg-run", on_done=_done, on_error=_fail)


def seg_stop(session, plot, payload=None) -> None:
    wizard = _wizard(session, plot)
    if wizard is not None and wizard._stopped is not None:
        wizard._stopped[0] = True


# ── the result tree ──────────────────────────────────────────────────────────

def _open_result(session, wizard: SegmentWizard, regions: Regions):
    """A new tree: the label map(s) as its signal, the regions attached.

    One door for every shape (``open_result_tree``): an image or a navigator
    commits one eager label map; a movie commits a LAZY label movie whose
    navigator is the count per frame.
    """
    from spyde.actions.commit import open_result_tree

    classifier, source, split = wizard.classifier, wizard.source, wizard.split_params()
    title = f"Segmented {_title(wizard.tree)}"
    if source.is_movie:
        signal = label_movie(wizard.signal(), classifier, split)
        navigator = count_navigator(wizard.signal(), regions)
    else:
        signal = label_image(wizard, classifier, split)
        navigator = None
    signal.metadata.General.title = title
    tree = open_result_tree(session, title=title, signal=signal, signal_type="regions",
                            navigator_override=navigator, provenance=regions.provenance)
    attach_container(tree, regions, name="regions")
    tree.source_tree = wizard.tree
    tree.segmentation_classifier = classifier
    highest = int(regions.count_series().max()) if regions.n_regions else 1
    for plot in getattr(tree, "signal_plots", []) or []:
        plot.set_colormap(LABEL_COLORMAP)
        plot.set_clim(0.0, float(max(highest, 1)))
    return tree


def label_field(field: np.ndarray, classifier: PixelClassifier, split: dict) -> np.ndarray:
    particle, boundary = classifier.foreground(field)
    return split_instances(particle, boundary, **split)


def label_image(wizard: SegmentWizard, classifier: PixelClassifier, split: dict):
    """The one field's label map as a 2-D signal on the field's own axes:
    the image's signal axes, or the scan's navigation axes for a navigator."""
    import hyperspy.api as hs
    labels = label_field(np.asarray(wizard.source.get(0)), classifier, split)
    signal = hs.signals.Signal2D(labels)
    if wizard.space == NAVIGATION_SPACE:
        source_axes = wizard.tree.root.axes_manager.navigation_axes
    else:
        source_axes = wizard.signal().axes_manager.signal_axes
    for axis, source_axis in zip(signal.axes_manager.signal_axes, source_axes):
        axis.name, axis.units = source_axis.name, source_axis.units
        axis.scale, axis.offset = float(source_axis.scale), float(source_axis.offset)
    return signal


def label_movie(signal, classifier: PixelClassifier, split: dict):
    """The label movie as a LAZY view of the source: a frame is labelled when
    it is asked for, the way a drift-corrected node is warped, so nothing is
    stored and the memory-safety rule holds for a movie of any length."""
    import dask.array as da
    data = signal.data
    if not hasattr(data, "dask"):
        data = da.from_array(data, chunks=(1,) + tuple(int(size) for size in data.shape[1:]))

    def _block(block):
        out = np.empty(block.shape, np.int32)
        for index in range(block.shape[0]):
            out[index] = label_field(np.asarray(block[index]), classifier, split)
        return out

    labelled = da.map_blocks(_block, data, dtype=np.int32,
                             meta=np.zeros((0, 0, 0), np.int32))
    new = signal._deepcopy_with_new_data(labelled)
    if not new._lazy:
        new._lazy = True
        new._assign_subclass()
    return new


def count_navigator(signal, regions: Regions):
    """Regions per frame as the result's navigator, on the source's time axis."""
    import hyperspy.api as hs
    navigator = hs.signals.Signal1D(regions.count_series().astype(np.float32))
    source_axis = signal.axes_manager.navigation_axes[0]
    axis = navigator.axes_manager.signal_axes[0]
    axis.name, axis.units = source_axis.name, source_axis.units
    axis.scale, axis.offset = float(source_axis.scale), float(source_axis.offset)
    navigator.metadata.General.title = "regions per frame"
    return navigator


def _title(tree) -> str:
    try:
        return str(tree.root.metadata.General.title) or "dataset"
    except Exception:
        return "dataset"


def segment(ctx, action_name: str = "Segment", **params):
    """Toolbar entry — a no-op parent; the Electron toolbar opens the staged
    caret, which drives the ``seg_*`` handlers (README §4)."""
    return None
