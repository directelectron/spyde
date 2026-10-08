"""centre_adapt_action.py — "Adapt to this scan" for the Find Vectors centre stage.

The Find Vectors wizard's Adapt button sends ``fv_adapt_centre`` with the tuned
parameters. On a worker thread this reads the scan a chunk of rows at a time,
runs the detector (exactly as Find Vectors would) and the chosen centre
network, and fine-tunes a copy of the network on the scan's own symmetry
(:mod:`spyde.models.centre_adapt`).

Messages to the wizard:

* ``fv_adapt_progress`` ``{stage, done, total}`` while it runs;
* ``fv_models`` again when a copy was accepted, so the Centre dropdown lists it;
* ``fv_adapt_result`` ``{accepted, declined, cancelled, report, model_id, label}``.

``fv_adapt_cancel`` stops it; closing the tree does too. ``fv_adapt_save`` keeps
an accepted copy for later sessions (it is otherwise pruned when the next
session opens Find Vectors).
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

from de_shell.ipc import emit, emit_error, emit_status
from spyde.actions.context import current_signal as _current_signal
from spyde.actions.context import src_plot_tree as _src_plot_tree

log = logging.getLogger(__name__)

#: Identifies this backend session in adapted-refiner manifest entries.
SESSION = f"{os.getpid()}-{int(time.time())}"

#: Payload keys that override :class:`~spyde.models.centre_adapt.AdaptSettings`.
_SETTING_KEYS = ("budget_seconds", "max_steps", "primary_ratio", "other_ratio",
                 "max_refined_drop")


def _settings(payload: dict):
    from spyde.models.centre_adapt import AdaptSettings

    settings = AdaptSettings()
    budget = os.environ.get("SPYDE_ADAPT_BUDGET_SECONDS")
    if budget:
        settings.budget_seconds = float(budget)
    for key in _SETTING_KEYS:
        value = (payload or {}).get(f"adapt_{key}")
        if value is not None:
            setattr(settings, key, int(value) if key == "max_steps" else float(value))
    return settings


def _row_chunks(signal):
    """``(frames, rows)`` a slab of scan rows at a time: the stored row chunk
    for lazy data, so each read is whole storage chunks."""
    data = signal.data
    ny, nx, height, width = data.shape
    if hasattr(data, "chunks") and not isinstance(data, np.ndarray):
        step = int(data.chunks[0][0])
    else:
        step = max(1, 2048 // nx)
    for top in range(0, ny, step):
        block = data[top:top + step]
        if hasattr(block, "compute"):
            block = block.compute()
        block = np.asarray(block, np.float32)
        yield block.reshape(-1, height, width), np.repeat(np.arange(top, top + len(block)), nx)


def _detector(params: dict):
    """The neural detector Find Vectors runs, decode centres, as ``frames ->
    [(n, 2)]``."""
    from spyde.actions.find_vectors_neural import _neural_block

    def detect(frames):
        block = _neural_block(
            frames[None], float(params["threshold"]), int(params["min_distance"]), True, None,
            params.get("model_id") or None, float(params.get("bg_sigma") or 12.0), False,
            float(params.get("spot_radius") or 0.0) or None, centre_refiner=None)[0]
        return [rows[np.isfinite(rows[:, 0]), :2] for rows in block]

    return detect


def _emit_result(window_id, **fields):
    emit({"type": "fv_adapt_result", "window_id": window_id, **fields})


def fv_adapt_centre(session, plot, payload) -> None:
    from spyde.actions.find_vectors_action import _coerce, fv_models
    from spyde.actions.lifecycle import run_on_worker
    from spyde.models import registry
    from spyde.models.centre_adapt import adapt_centre_refiner

    window_id = (payload or {}).get("window_id", getattr(plot, "window_id", None))
    src, tree = _src_plot_tree(session, plot)
    if src is None or tree is None:
        emit_error("Adapt: no active dataset")
        return
    signal = _current_signal(src) or tree.root
    params = _coerce(payload or {})
    parent_id = params.get("centre_refiner")
    if params["method"] != "neural" or signal.axes_manager.navigation_dimension != 2 \
            or signal.axes_manager.signal_dimension != 2:
        _emit_result(window_id, ok=False, accepted=False,
                     declined="Adapting needs the neural method on a 4D-STEM scan.")
        return
    try:
        base = registry.get_refiner(parent_id)
    except Exception:
        _emit_result(window_id, ok=False, accepted=False,
                     declined="Pick a centre network to adapt (not the decode or the mask centroid).")
        return
    radius = float(params.get("spot_radius") or 0.0)
    if radius < base.min_spot_radius:
        _emit_result(window_id, ok=False, accepted=False,
                     declined=f"The spot size ({radius:g} px) is under the {base.min_spot_radius:g} px "
                              f"this network refines, so there is nothing to adapt.")
        return

    cancel = tree.register_cancel()
    tree._fv_adapt_cancel = cancel
    dataset = str(tree.root.metadata.get_item("General.title", "") or "this scan")
    last_emit = [0.0]

    def progress(stage, done, total):
        now = time.monotonic()
        if now - last_emit[0] > 0.5:
            last_emit[0] = now
            emit({"type": "fv_adapt_progress", "window_id": window_id,
                  "stage": stage, "done": done, "total": total})

    def work():
        try:
            emit_status("Adapting the centre network to this scan…")
            adapted, report = adapt_centre_refiner(
                _row_chunks(signal), radius, base, _detector(params),
                tuple(signal.axes_manager.signal_shape[::-1]), _settings(payload),
                progress=progress, cancel=cancel)
            fields = dict(ok=True, accepted=report.accepted, declined=report.declined,
                          cancelled=report.cancelled, report=report.to_dict(), parent=parent_id)
            if adapted is not None:
                entry = registry.register_adapted(adapted, parent_id, dataset, report.to_dict(),
                                                  session=SESSION)
                fields.update(model_id=entry["id"], label=entry["label"])
                fv_models(session, plot, {"window_id": window_id})
            _emit_result(window_id, **fields)
            emit_status("Adapting: " + ("accepted" if report.accepted else
                                        "cancelled" if report.cancelled else
                                        "declined" if report.declined else "not adopted"))
        except Exception as error:
            log.exception("[fv-adapt] failed")
            _emit_result(window_id, ok=False, accepted=False, declined=f"Adapting failed: {error}")
        finally:
            tree.unregister_cancel(flag=cancel)

    run_on_worker(session, work, name="fv-adapt-centre")


def fv_adapt_cancel(session, plot, payload) -> None:
    src, tree = _src_plot_tree(session, plot)
    cancel = getattr(tree, "_fv_adapt_cancel", None) if tree is not None else None
    if cancel is not None:
        cancel[0] = True


def fv_adapt_save(session, plot, payload) -> None:
    from spyde.models import registry

    model_id = (payload or {}).get("model_id")
    if model_id and registry.keep_adapted(model_id):
        emit_status("Saved the adapted centre network for later sessions.")
    else:
        emit_error(f"Adapt: no adapted network {model_id!r} to save")
