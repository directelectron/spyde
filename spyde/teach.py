"""
teach.py — what every "mark it, fit it, look again" feature shares.

A user marks a few places on a field, a small model is fitted to those marks on
a worker, the field repaints with it, and the result is kept as a model of
their own. Find Vectors' adapt step is the first user; the trainable
segmentation's scribbles are meant to be the second. Nothing here knows about
diffraction or about any particular model:

* :class:`PointMarks` — the marks, kept per field (a navigation position).
* :class:`DebouncedFit` — fit a snapshot of the marks once the user pauses,
  one fit at a time, the newest request winning.
* :func:`save_user_model` / :func:`name_user_model` / :func:`user_models` /
  :func:`remove_user_model` — a fitted model kept on disk beside the downloaded
  ones, in a manifest of its own (the downloaded models' manifest is overwritten
  by every refresh). A fit is an unsaved draft until the user names it.
* :func:`provenance` — the record a result made with a taught model carries.

torch is never imported here.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Iterable

import numpy as np

log = logging.getLogger(__name__)

#: The manifest of user-taught models, beside the downloaded models' manifest.
USER_MANIFEST = "taught.json"
#: Subdirectory the taught weights are written to.
USER_WEIGHTS_DIR = "taught"


# ── marks ─────────────────────────────────────────────────────────────────────

class PointMarks:
    """Point labels, per field.

    A field is any hashable key; Find Vectors uses the navigation index tuple.
    Each mark is ``(y, x, label)`` in the field's pixel coordinates, the label a
    small integer whose meaning belongs to the caller (Find Vectors: 1 = a disk
    is here, 0 = this detection is wrong).
    """

    def __init__(self) -> None:
        self._fields: dict[Any, list[tuple[float, float, int]]] = {}

    def add(self, field, y: float, x: float, label: int) -> None:
        self._fields.setdefault(_key(field), []).append((float(y), float(x), int(label)))

    def remove_near(self, field, y: float, x: float, radius: float) -> bool:
        """Remove the mark nearest ``(y, x)`` if it lies within ``radius``.
        Returns True when one was removed."""
        marks = self._fields.get(_key(field))
        if not marks:
            return False
        distances = [np.hypot(my - y, mx - x) for my, mx, _ in marks]
        nearest = int(np.argmin(distances))
        if distances[nearest] > radius:
            return False
        marks.pop(nearest)
        if not marks:
            del self._fields[_key(field)]
        return True

    def at(self, field) -> np.ndarray:
        """``(N, 3)`` float array of ``[y, x, label]`` for one field."""
        marks = self._fields.get(_key(field), [])
        return np.asarray(marks, dtype=np.float64).reshape(-1, 3)

    def fields(self) -> list:
        return list(self._fields)

    def count(self, label: int | None = None) -> int:
        return sum(1 for marks in self._fields.values() for *_, lab in marks
                   if label is None or lab == label)

    def __len__(self) -> int:
        return self.count()

    def clear(self) -> None:
        self._fields.clear()

    def to_dict(self) -> dict:
        """JSON-safe: ``{"fields": [{"field": [...], "marks": [[y, x, label], ...]}]}``."""
        return {"fields": [{"field": list(field) if isinstance(field, tuple) else field,
                            "marks": [list(mark) for mark in marks]}
                           for field, marks in self._fields.items()]}

    @classmethod
    def from_dict(cls, state: dict) -> "PointMarks":
        marks = cls()
        for entry in state.get("fields", []):
            for y, x, label in entry["marks"]:
                marks.add(entry["field"], y, x, label)
        return marks

    def copy(self) -> "PointMarks":
        return PointMarks.from_dict(self.to_dict())


def _key(field):
    if isinstance(field, (list, tuple, np.ndarray)):
        return tuple(int(v) for v in field)
    return field


# ── the debounced fit ─────────────────────────────────────────────────────────

class DebouncedFit:
    """Fit once the user pauses, on a worker, one fit at a time.

    ``request()`` (re)starts a ``delay``-second timer; when it runs out the fit
    starts. ``run_now()`` skips the wait. ``snapshot()`` is taken on the main
    thread at the moment the fit starts, so marks added during a fit belong to
    the next one; ``fit(snapshot)`` runs on a worker and ``on_done(result)`` back
    on the main thread. A request that arrives while a fit runs waits for it and
    then starts one more fit. ``cancel()`` drops the timer and makes any running
    fit's result be ignored.
    """

    def __init__(self, session, *, snapshot: Callable[[], Any], fit: Callable[[Any], Any],
                 on_done: Callable[[Any], None], on_error: Callable[[Exception], None] | None = None,
                 on_start: Callable[[], None] | None = None, delay: float = 1.0,
                 name: str = "teach-fit") -> None:
        self.session = session
        self.snapshot, self.fit, self.on_done = snapshot, fit, on_done
        self.on_error, self.on_start = on_error, on_start
        self.delay, self.name = float(delay), name
        self.generation = 0
        self._timer: threading.Timer | None = None
        self._busy = False
        self._again = False
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self._busy

    def request(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.delay, self._elapsed)
            self._timer.daemon = True
            self._timer.start()

    def run_now(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self._start()

    def cancel(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self.generation += 1
            self._again = False

    def _elapsed(self) -> None:
        with self._lock:
            self._timer = None
        dispatch = getattr(self.session, "_dispatch_to_main", None)
        if dispatch is not None:
            dispatch(self._start)
        else:
            self._start()

    def _start(self) -> None:
        if self._busy:
            self._again = True
            return
        self.generation += 1
        generation = self.generation
        self._busy = True
        try:
            snapshot = self.snapshot()
        except Exception as error:      # noqa: BLE001 - reported, never raised into a timer
            self._busy = False
            self._report(error)
            return
        if self.on_start is not None:
            self.on_start()
        from de_shell.actions.lifecycle import run_on_worker

        def done(result) -> None:
            self._busy = False                 # before on_done, which reports the state
            try:
                if generation == self.generation:
                    self.on_done(result)
            finally:
                self._next()

        def failed(error: Exception) -> None:
            self._busy = False
            if generation == self.generation:
                self._report(error)
            dispatch = getattr(self.session, "_dispatch_to_main", None)
            if dispatch is not None:
                dispatch(self._next)
            else:
                self._next()

        run_on_worker(self.session, lambda: self.fit(snapshot), name=self.name,
                      on_done=done, on_error=failed)

    def _next(self) -> None:
        if self._again:
            self._again = False
            self._start()

    def _report(self, error: Exception) -> None:
        if self.on_error is not None:
            self.on_error(error)
        else:
            log.warning("[teach] %s failed: %s", self.name, error)


# ── the user's own models ─────────────────────────────────────────────────────

def _manifest_path(directory: str) -> str:
    return os.path.join(directory, USER_MANIFEST)


def user_models(directory: str, kind: str | None = None) -> list[dict]:
    """The taught models recorded in ``directory``, optionally of one kind."""
    path = _manifest_path(directory)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            entries = json.load(handle).get("models", [])
    except FileNotFoundError:
        return []
    except Exception as error:      # noqa: BLE001 - a broken manifest must not break the app
        log.warning("[teach] %s unreadable (%s); ignoring it", path, error)
        return []
    return [entry for entry in entries if kind is None or entry.get("kind") == kind]


def _write_manifest(directory: str, entries: Iterable[dict]) -> None:
    os.makedirs(directory, exist_ok=True)
    path = _manifest_path(directory)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump({"models": list(entries)}, handle, indent=1)
    os.replace(temporary, path)


def save_user_model(directory: str, *, kind: str, scope: str,
                    write_weights: Callable[[str], None], entry: dict,
                    write_icon: Callable[[str], None] | None = None) -> dict:
    """Write a fitted model's weights (and icon) and record it as an UNSAVED draft
    of ``scope``, replacing that scope's previous draft. A draft becomes a model
    of the user's own when it is named (:func:`name_user_model`); a draft nobody
    names is discarded by the caller, and drafts left by a crash are pruned here
    after a day.

    ``write_weights(path)`` writes the weight file. The id is derived from the
    file's sha256, so a model is immutable once recorded and a new fit is a new
    id — a process that cached the old one by id can never be handed new weights
    under it. ``write_icon(path)`` writes a small PNG; a failure leaves the model
    without one. ``entry`` holds whatever the kind needs to load and describe it;
    ``id``, ``kind``, ``scope``, ``sha256``, ``source``, ``icon``, ``unsaved``
    and ``created`` are added. Returns the recorded entry."""
    weights_dir = os.path.join(directory, USER_WEIGHTS_DIR)
    os.makedirs(weights_dir, exist_ok=True)
    temporary = os.path.join(weights_dir, f".{kind}-{os.getpid()}-{threading.get_ident()}.part")
    write_weights(temporary)
    digest = hashlib.sha256()
    with open(temporary, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    sha256 = digest.hexdigest()
    model_id = f"{kind}-taught-{sha256[:12]}"
    filename = f"{model_id}.pt"
    os.replace(temporary, os.path.join(weights_dir, filename))
    icon = None
    if write_icon is not None:
        try:
            write_icon(os.path.join(weights_dir, f"{model_id}.png"))
            icon = f"{USER_WEIGHTS_DIR}/{model_id}.png"
        except Exception as error:      # noqa: BLE001 - an icon is decoration
            log.info("[teach] no icon for %s: %s", model_id, error)
    recorded = {**entry, "id": model_id, "kind": kind, "scope": scope, "sha256": sha256,
                "source": {"type": "taught", "file": f"{USER_WEIGHTS_DIR}/{filename}"},
                "icon": icon, "unsaved": True, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    entries = user_models(directory)
    now = time.time()
    replaced = [e for e in entries if e.get("unsaved") and e.get("id") != model_id
                and e.get("kind") == kind
                and (e.get("scope") == scope or now - _created(e) > STALE_DRAFT_SECONDS)]
    kept = [e for e in entries if e not in replaced and e.get("id") != model_id]
    _write_manifest(directory, kept + [recorded])
    for old in replaced:
        _remove_files(directory, old)
    return recorded


#: An unsaved draft older than this belongs to a session that is gone.
STALE_DRAFT_SECONDS = 24 * 3600


def _created(entry: dict) -> float:
    try:
        return time.mktime(time.strptime(entry.get("created", ""), "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return 0.0


def name_user_model(directory: str, model_id: str, name: str) -> dict:
    """Name (or rename) a model: a named model is the user's own, kept, and
    offered for every dataset. Returns the updated entry."""
    name = " ".join(str(name).split())
    if not name:
        raise ValueError("a model needs a name")
    entries = user_models(directory)
    for entry in entries:
        if entry.get("id") == model_id:
            entry.update(name=name, label=name, unsaved=False)
            _write_manifest(directory, entries)
            return entry
    raise KeyError(model_id)


def remove_user_model(directory: str, model_id: str) -> bool:
    """Forget a taught model and delete its files. Returns True when found."""
    entries = user_models(directory)
    gone = [e for e in entries if e.get("id") == model_id]
    if not gone:
        return False
    _write_manifest(directory, [e for e in entries if e.get("id") != model_id])
    for entry in gone:
        _remove_files(directory, entry)
    return True


def user_model_path(directory: str, entry: dict) -> str:
    return os.path.join(directory, entry["source"]["file"])


def _remove_files(directory: str, entry: dict) -> None:
    for relative in (entry["source"]["file"], entry.get("icon")):
        if not relative:
            continue
        try:
            os.remove(os.path.join(directory, relative))
        except OSError as error:
            log.debug("[teach] removing %s failed: %s", relative, error)


# ── provenance ────────────────────────────────────────────────────────────────

def provenance(*, kind: str, model_id: str, parent: dict, marks: PointMarks,
               hyperparameters: dict, report: dict, scope: str) -> dict:
    """The record a result made with a taught model carries: what it was taught
    from (the parent model and the marks), how, and how it scored."""
    return {"kind": kind, "model_id": model_id, "parent": dict(parent), "scope": scope,
            "marks": marks.to_dict(), "hyperparameters": dict(hyperparameters),
            "report": dict(report)}
