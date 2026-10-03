"""
find_vectors_adapt.py — teach the neural disk detector on this dataset.

While the Find Vectors caret is open, a double-click on the diffraction pattern
marks it:

* on a found peak's circle → "this is not a disk" (an orange ✕);
* on empty pattern → "a disk is here" (a green ring);
* on a mark → removes that mark.

About a second after the last mark (or on Adapt) a copy of the model is
fine-tuned on every mark so far (``spyde.models.adapt``) and the wizard
switches its Model to it, so the preview repaints with the adapted detector and
Compute uses it. The fit is an unsaved draft until the user names it; a named
model is kept and offered on every dataset, where it can be adapted again (its
parents recorded as a chain). Revert goes back to the model the marks were made
against; an unsaved draft is also discarded then, and when the caret closes.

The marks, the fit and the saving are the generic parts in ``spyde.teach``;
this module is the Find-Vectors wiring: reading the marked patterns, the
double-click hit test against the live preview, and the marks overlay.

State lives on the source tree as ``tree._fv_adapt``; it survives the caret
closing (reopening resumes with the same marks and model) and goes with the
tree.
"""
from __future__ import annotations

import logging
import os

import numpy as np

from de_shell.ipc import emit, emit_error, emit_status

from spyde import teach
from spyde.actions.context import current_signal as _current_signal
from spyde.actions.context import src_plot_tree as _src_plot_tree

log = logging.getLogger(__name__)

#: Seconds without a new mark before the fit starts.
FIT_DELAY = 1.0
#: A double-click this close (pixels) to a mark removes it.
MARK_HIT_PX = 6.0
#: Marks colours: a disk the detector missed / a detection that is wrong.
DISK_COLOUR, NOT_DISK_COLOUR = "#40d070", "#ff9a3c"


def dataset_scope(tree) -> str:
    """What a taught model is FOR: the file the tree was opened from, or, for
    data that never came from a file, its title in this session."""
    path = getattr(tree, "source_path", None)
    if path:
        return os.path.normcase(os.path.abspath(str(path)))
    title = tree.root.metadata.get_item("General.title", "untitled") if tree.root is not None else "untitled"
    return f"unsaved:{title}"


# ── the marks overlay ────────────────────────────────────────────────────────

def mark_shapes(*, position, marks: dict, radius: float) -> dict:
    """The marks at one navigation position: rings for "disk here", ✕ for "not a
    disk", in image pixels."""
    rows = np.asarray(marks.get(tuple(position), []), dtype=np.float64).reshape(-1, 3)
    disk = rows[rows[:, 2] == 1]
    wrong = rows[rows[:, 2] == 0]
    arm = 0.7 * radius
    crosses = np.concatenate([
        np.stack([wrong[:, [1, 0]] + [-arm, -arm], wrong[:, [1, 0]] + [arm, arm]], 1),
        np.stack([wrong[:, [1, 0]] + [-arm, arm], wrong[:, [1, 0]] + [arm, -arm]], 1)]) \
        if len(wrong) else np.zeros((0, 2, 2))
    return {"disk": disk[:, [1, 0]].astype(np.float32), "not_disk": crosses.astype(np.float32)}


class FindVectorsAdapt:
    """The adapt state of one source tree."""

    def __init__(self, session, tree) -> None:
        self.session, self.tree = session, tree
        self.marks = teach.PointMarks()
        self.base_model_id: str | None = None     # what the marks were made against
        self.model_id: str | None = None          # the taught model, when there is one
        self.entry: dict | None = None
        self.params: dict = {}
        self.active = False
        self.plot = None
        self.overlay = None
        self._wired: set[int] = set()
        self._stop: list | None = None              # the running fit's cancel flag
        self.fitter = teach.DebouncedFit(
            session, snapshot=self._snapshot, fit=self._fit, on_done=self._fitted,
            on_error=self._failed, on_start=lambda: self.emit_state("Adapting…"),
            delay=FIT_DELAY, name="fv-adapt")

    # -- caret life ------------------------------------------------------------

    def open(self, plot, params: dict) -> None:
        """The caret opened (or re-opened) on ``plot``."""
        self.plot, self.active = plot, True
        self.set_params(params)
        self._wire(plot)
        self._attach_overlay()
        self._warm_up()
        self.emit_state()

    def close(self) -> None:
        """The caret closed: stop listening and drop the overlay. Marks and the
        taught model stay for the next time it opens."""
        self.active = False
        self._cancel_fit()
        if self.entry is not None and self.entry.get("unsaved"):
            self._discard_draft()
            self.model_id, self.entry = None, None
            self.marks.clear()
        from spyde.actions.vector_overlay import clear_tree_overlay
        clear_tree_overlay(self.tree, "_fv_adapt_overlay")
        self.overlay = None

    def set_params(self, params: dict) -> None:
        self.params = dict(params)
        if self.base_model_id is None or not len(self.marks):
            chosen = params.get("model_id") or None
            if chosen != self.model_id:
                self.base_model_id = chosen

    # -- input -----------------------------------------------------------------

    def _wire(self, plot) -> None:
        # anyplotlib has no remove_event_handler: wire once per figure and
        # gate on ``active``.
        plot2d = getattr(plot, "_plot2d", None)
        if plot2d is None or id(plot2d) in self._wired:
            return
        from spyde.drawing.selectors.base_selector import event_handler_fn
        handler = event_handler_fn(self.on_double_click)
        try:
            plot2d.add_event_handler(handler, "double_click")
            self._wired.add(id(plot2d))
            self._handler = handler                 # keep it alive
        except Exception as error:
            log.debug("[fv-adapt] wiring the double-click failed: %s", error)

    def on_double_click(self, event=None) -> None:
        if not self.active or event is None or self.plot is None:
            log.info("[fv-adapt] double-click ignored: caret %s", "open" if self.active else "closed")
            return
        try:
            x, y = float(event.xdata), float(event.ydata)
        except Exception:
            log.info("[fv-adapt] double-click ignored: no data position (%r)", event)
            return
        from spyde.actions.vector_overlay import DetectorPixels
        source = _current_signal(self.plot) or self.tree.root
        px, py = DetectorPixels.from_signal(source).to_pixels([[x, y]])[0]
        drawn = self.plot.last_overlay_value(getattr(self.tree, "_fv_preview", None)) or {}
        index = drawn.get("index")
        if index is None:
            log.info("[fv-adapt] double-click ignored: the preview has not drawn on this window yet")
            return
        peaks = drawn.get("peaks") or {}
        self.toggle(tuple(index), float(py), float(px),
                    np.asarray(peaks.get("data", np.zeros((0, 2))), np.float64).reshape(-1, 2),
                    float(peaks.get("radius", 5.0)))

    def toggle(self, index: tuple, y: float, x: float, peaks_xy: np.ndarray, radius: float) -> None:
        """One double-click at ``(y, x)`` pixels on the pattern at ``index``."""
        from spyde.models.adapt import DISK, NOT_DISK
        if self.marks.remove_near(index, y, x, MARK_HIT_PX):
            what = "removed a mark"
        else:
            d = np.hypot(peaks_xy[:, 1] - y, peaks_xy[:, 0] - x) if len(peaks_xy) else np.zeros(0)
            if len(d) and d.min() <= max(radius, MARK_HIT_PX):
                hit = peaks_xy[int(np.argmin(d))]
                self.marks.add(index, hit[1], hit[0], NOT_DISK)
                what = "not a disk"
            else:
                self.marks.add(index, y, x, DISK)
                what = "disk here"
        log.info("[fv-adapt] double-click at %s px (%.1f, %.1f): %s", index, y, x, what)
        self._refresh_overlay()
        self.emit_state()
        if len(self.marks):
            self.fitter.request()
        else:
            self._cancel_fit()               # the last mark went: nothing to fit

    # -- fitting ---------------------------------------------------------------

    def _cancel_fit(self) -> None:
        self.fitter.cancel()
        if self._stop is not None:
            self._stop[0] = True

    def adapt_now(self) -> None:
        if not len(self.marks):
            emit_error("Adapt: double-click wrong detections or missed disks first")
            return
        self.fitter.run_now()

    def _snapshot(self) -> dict:
        source = _current_signal(self.plot) or self.tree.root if self.plot is not None else self.tree.root
        # Closing the tree stops the fit between optimiser steps.
        register = getattr(self.tree, "register_cancel", None)
        self._stop = register() if register is not None else [False]
        return dict(marks=self.marks.copy(), params=dict(self.params), source=source, stop=self._stop,
                    base=self.base_model_id, scope=dataset_scope(self.tree),
                    title=self.tree.root.metadata.get_item("General.title", "dataset"),
                    default_name=default_model_name(self.tree))

    def _fit(self, snap: dict) -> dict:
        """Worker: read the marked patterns, fine-tune, save. Returns the entry."""
        import torch
        from spyde import models
        from spyde.models.adapt import adapt, write_icon
        marks: teach.PointMarks = snap["marks"]
        fields = marks.fields()
        frames = [_read_frame(snap["source"], field) for field in fields]
        base, device = models.get_model(snap["base"])
        try:
            model, report = adapt(base, device, frames, [marks.at(f) for f in fields], snap["params"],
                                  stop=snap["stop"])
        finally:
            unregister = getattr(self.tree, "unregister_cancel", None)
            if unregister is not None:
                unregister(flag=snap["stop"])
        parent = models.registry._entry(snap["base"]) or {}
        chain = list(parent.get("chain") or []) + [
            {"id": parent.get("id", snap["base"]), "label": parent.get("label", parent.get("id"))}]
        arch = dict(parent.get("arch") or {"base": int(model.enc[0][0].out_channels),
                                           "levels": int(model.levels), "in_ch": 1})

        def write(path: str) -> None:
            torch.save({"state_dict": model.state_dict(), "base": arch.get("base"),
                        "levels": arch.get("levels"), "in_ch": arch.get("in_ch", 1)}, path)

        entry = teach.save_user_model(
            models.registry.user_models_dir(), kind=models.registry.TAUGHT_KIND, scope=snap["scope"],
            write_weights=write, write_icon=lambda path: write_icon(model, "cpu", path),
            entry=dict(label=f"Unsaved: {snap['default_name']}", default_name=snap["default_name"],
                       version=1, arch=arch,
                       parent={"id": parent.get("id", snap["base"]), "sha256": parent.get("sha256")},
                       chain=chain, trained_on={"title": snap["title"], "scope": snap["scope"]},
                       notes=f"Fine-tuned from {chain[-1]['label']} on marks made on {snap['title']}.",
                       marks=marks.to_dict(),
                       calibration={k: snap["params"].get(k) for k in
                                    ("spot_radius", "min_distance", "bg_sigma", "threshold")},
                       hyperparameters={k: report[k] for k in
                                        ("steps", "learning_rate", "replay_weight", "distill_weight",
                                         "offset_weight", "peak_hold", "method")},
                       report=report))
        models.registry.reload_manifest()
        return entry

    def _fitted(self, entry: dict) -> None:
        from spyde import models
        old = self.model_id
        self.model_id, self.entry = entry["id"], entry
        self._use_model(entry["id"])
        if old and old != entry["id"]:
            models.registry.forget_model(old)
        report = entry["report"]
        self.emit_state(f"Adapted in {report['seconds']:.1f} s — "
                        f"{report['not_disk_marks']} wrong, {report['disk_marks']} missed")
        self.emit_models()

    def _failed(self, error: Exception) -> None:
        from spyde.models.adapt import Cancelled
        if isinstance(error, Cancelled):
            return
        emit_error(f"Adapt failed: {error}")
        self.emit_state("Adapt failed")

    def revert(self) -> None:
        """Back to the model the marks were made against. The marks go, and so
        does the fitted model unless the user named it (then it is theirs)."""
        self._cancel_fit()
        self._use_model(self.base_model_id or "")
        self._discard_draft()
        self.model_id, self.entry = None, None
        self.marks.clear()
        self._refresh_overlay()
        self.emit_state("Back to the original model")
        self.emit_models()

    def _discard_draft(self) -> None:
        """Delete the current fit if it was never named."""
        from spyde import models
        if not self.model_id or self.entry is None or not self.entry.get("unsaved"):
            return
        teach.remove_user_model(models.registry.user_models_dir(), self.model_id)
        models.registry.forget_model(self.model_id)
        models.registry.reload_manifest()

    def name_model(self, name: str) -> dict:
        """Name the current fit: it becomes the user's own model."""
        if not self.model_id:
            raise RuntimeError("adapt first, then name the model")
        entry = rename_model(self.model_id, name)
        self.entry = entry
        self.emit_state(f"Saved as \u201c{entry['name']}\u201d")
        self.emit_models()
        return entry

    def model_deleted(self, model_id: str) -> None:
        """A model was deleted from the picker: stop using it if this tree was."""
        if model_id == self.model_id:
            self.model_id, self.entry = None, None
        if model_id == self.base_model_id:
            self.base_model_id = None
        if model_id == self.params.get("model_id"):
            self._use_model(self.base_model_id or "")
        self.emit_state()

    def _use_model(self, model_id: str) -> None:
        """Point the live preview at ``model_id`` now, rather than after the caret's
        round trip: the model it was using may be about to be deleted. The caret
        adopts the same id from ``fv_adapt_state`` for Compute and later tunes."""
        self.params["model_id"] = model_id
        preview = getattr(self.tree, "_fv_preview", None)
        if preview is None or not preview.attached:
            return
        from spyde.actions.vector_overlay import overlay_static
        params = dict(overlay_static(preview).get("params", {}), model_id=model_id)
        self.tree.replace_overlay_static(preview, params=params)

    def provenance(self, model_id: str | None) -> dict | None:
        """The taught model's record, when ``model_id`` is it."""
        if not model_id or model_id != self.model_id or self.entry is None:
            return None
        e = self.entry
        record = teach.provenance(kind=e["kind"], model_id=e["id"], parent=e["parent"],
                                  marks=teach.PointMarks.from_dict(e["marks"]),
                                  hyperparameters=e["hyperparameters"], report=e["report"],
                                  scope=e["scope"])
        record.update(name=e.get("name"), chain=e.get("chain", []), trained_on=e.get("trained_on"))
        return record

    def _warm_up(self) -> None:
        """Initialise autograd on this (dispatch) thread and the device kernels on a
        worker, so the first Adapt does not pay ~2 s of one-off setup."""
        if getattr(self.tree, "_fv_adapt_warm", False):
            return
        self.tree._fv_adapt_warm = True
        from spyde.torch_device import warmup_autograd
        warmup_autograd()

        def work():
            import torch
            from spyde import models
            from spyde.device_lock import accelerator_lock
            model, device = models.get_model(self.params.get("model_id") or None)
            with accelerator_lock(device):
                trial = torch.zeros(1, 1, 64, 64, device=device, requires_grad=True)
                model(trial)[0].sum().backward()

        from spyde.actions.lifecycle import run_on_worker
        run_on_worker(self.session, work, name="fv-adapt-warmup")

    # -- display ---------------------------------------------------------------

    def _attach_overlay(self) -> None:
        if self.plot is None:
            return
        from spyde.actions.vector_overlay import _add_overlay, clear_tree_overlay
        from spyde.drawing.overlay_node import NavigationPosition
        clear_tree_overlay(self.tree, "_fv_adapt_overlay")
        source = _current_signal(self.plot) or self.tree.root
        radius = max(3.0, float(self.params.get("kernel_radius", 5)))
        self.overlay = _add_overlay(
            self.tree, source, mark_shapes, name="fv_marks", source=False,
            groups={"disk": ("circles", {"radius": radius, "edgecolors": DISK_COLOUR,
                                         "facecolors": None, "linewidths": 2.0, "alpha": 1.0}),
                    "not_disk": ("lines", {"edgecolors": NOT_DISK_COLOUR, "linewidths": 2.5})},
            iterating={"position": NavigationPosition()},
            static={"marks": self._marks_by_field(), "radius": radius})
        self.tree._fv_adapt_overlay = self.overlay

    def _marks_by_field(self) -> dict:
        return {field: self.marks.at(field).tolist() for field in self.marks.fields()}

    def _refresh_overlay(self) -> None:
        if self.overlay is not None and self.overlay.attached:
            self.tree.replace_overlay_static(self.overlay, marks=self._marks_by_field())

    def emit_state(self, status: str | None = None) -> None:
        from spyde.models.adapt import DISK, NOT_DISK
        window_id = getattr(self.plot, "window_id", None)
        message = {"type": "fv_adapt_state", "window_id": window_id,
                   "marks": len(self.marks), "disk": self.marks.count(DISK),
                   "not_disk": self.marks.count(NOT_DISK), "busy": self.fitter.busy,
                   "model_id": self.model_id, "base_model_id": self.base_model_id}
        if self.entry is not None:
            message["unsaved"] = bool(self.entry.get("unsaved"))
            message["name"] = self.entry.get("name") or ""
            message["default_name"] = self.entry.get("default_name") or ""
            message["original_f1"] = float(self.entry["report"]["original_f1"])
            message["original_f1_base"] = float(self.entry["report"]["original_f1_base"])
        if status:
            message["status"] = status
        emit(message)

    def emit_models(self) -> None:
        from spyde import models
        message = {"type": "fv_models", "window_id": getattr(self.plot, "window_id", None)}
        message.update(models.available_models(scope=dataset_scope(self.tree)))
        emit(message)


def _read_frame(signal, field) -> np.ndarray:
    """One pattern at a navigation index (a single small read, never the dataset).
    ``field`` is the index in array order, as the overlay reports it."""
    frame = signal.data[tuple(int(v) for v in field)]
    if hasattr(frame, "compute"):
        frame = frame.compute()
    return np.asarray(frame, dtype=np.float32)


def controller(session, tree, *, create: bool = False) -> FindVectorsAdapt | None:
    current = getattr(tree, "_fv_adapt", None)
    if current is None and create:
        current = FindVectorsAdapt(session, tree)
        tree._fv_adapt = current
    return current


# ── staged handlers ─────────────────────────────────────────────────────────

def fv_adapt(session, plot, payload) -> None:
    """Adapt now, without waiting for the pause after the last mark."""
    _src, tree = _src_plot_tree(session, plot)
    adapt = controller(session, tree) if tree is not None else None
    if adapt is None:
        emit_error("Adapt: open Find Vectors first")
        return
    adapt.adapt_now()


def fv_adapt_revert(session, plot, payload) -> None:
    """Back to the original model; the marks and the taught model are dropped."""
    _src, tree = _src_plot_tree(session, plot)
    adapt = controller(session, tree) if tree is not None else None
    if adapt is not None:
        adapt.revert()
        emit_status("Find Vectors: back to the original model")


def default_model_name(tree) -> str:
    """``<file stem> adapted``, or the dataset title for data with no file."""
    path = getattr(tree, "source_path", None)
    stem = os.path.splitext(os.path.basename(str(path)))[0] if path else (
        tree.root.metadata.get_item("General.title", "dataset") if tree.root is not None else "dataset")
    return f"{stem} adapted"


def rename_model(model_id: str, name: str) -> dict:
    from spyde import models
    entry = teach.name_user_model(models.registry.user_models_dir(), model_id, name)
    models.registry.reload_manifest()
    return entry


def fv_model_name(session, plot, payload) -> None:
    """Name the current fit, or rename a taught model (``model_id``)."""
    payload = payload or {}
    name = str(payload.get("name") or "").strip()
    model_id = payload.get("model_id") or None
    _src, tree = _src_plot_tree(session, plot)
    adapt = controller(session, tree) if tree is not None else None
    try:
        if adapt is not None and (model_id is None or model_id == adapt.model_id):
            adapt.name_model(name)
        else:
            rename_model(model_id, name)
            if adapt is not None:
                adapt.emit_models()
    except (ValueError, KeyError, RuntimeError) as error:
        emit_error(f"Naming the model failed: {error}")


def fv_model_delete(session, plot, payload) -> None:
    """Delete a taught model (never a vendored one)."""
    from spyde import models
    model_id = (payload or {}).get("model_id")
    entry = models.registry._entry(model_id) if model_id else None
    if entry is None or not entry.get("kind"):
        emit_error("Only a model you taught can be deleted")
        return
    teach.remove_user_model(models.registry.user_models_dir(), model_id)
    models.registry.forget_model(model_id)
    models.registry.reload_manifest()
    _src, tree = _src_plot_tree(session, plot)
    adapt = controller(session, tree) if tree is not None else None
    if adapt is not None:
        adapt.model_deleted(model_id)
        adapt.emit_models()
    emit_status(f"Deleted the model \u201c{entry.get('name') or entry.get('label')}\u201d")
