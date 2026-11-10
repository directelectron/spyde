"""An Autopilot run folder, opened as a lazy signal.

Autopilot's "Open in SpyDE" hands SpyDE the run folder: ``run.json`` and,
beside it, ``dataset.json``, the record of the files DE-Server wrote and which
scan point each frame was taken at. Only files the record marks complete are
read — a run still going opens as far as its closed files reach — and each is
checked against the record before it is mapped (see ``dataset_reader``).

The signal is the latest frame taken at every scan point, read from disk only
as it is looked at; a point with no complete frame yet is zeros. A grid scan
gets its grid as the navigation, a list of points one navigation axis.
"""
from __future__ import annotations

import os

import numpy as np

from spyde.autopilot.dataset_reader import RECORD, DatasetReader

#: How many bytes of frames one block of the lazy signal holds.
BLOCK_BYTES = 64 << 20


def run_folder(path: str) -> str | None:
    """The run folder *path* names — the folder, or its ``dataset.json`` —
    or None when it is not one."""
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    if os.path.isdir(path) or os.path.basename(path) == RECORD:
        if os.path.isfile(os.path.join(folder, RECORD)):
            return folder
    return None


def load_run(path: str, *, detector: str = "camera"):
    """The run folder at *path* as a lazy HyperSpy ``Signal2D``."""
    import dask.array as da
    import hyperspy.api as hs

    folder = run_folder(path)
    if folder is None:
        raise ValueError(f"{path} is not an Autopilot run folder (no {RECORD})")
    reader = DatasetReader(folder)
    complete = [file for file in reader.files(detector) if file["complete"]]
    if not complete:
        note = reader.record["detectors"][detector].get("note") or "no file is complete yet"
        raise ValueError(f"{os.path.basename(folder)} has no frames to open: {note}")
    stacks = []
    for file in complete:
        frames = reader.open(file)
        per_block = max(1, BLOCK_BYTES // max(1, frames[0].nbytes))
        stacks.append(da.from_array(frames, chunks=(per_block, -1, -1), name=False))
    movie = da.concatenate(stacks)
    blank = da.zeros((1, *movie.shape[1:]), movie.dtype)
    latest = reader.latest(detector)
    rows = np.where(latest < 0, len(movie), latest)
    data = da.concatenate([movie, blank])[rows]
    data = data.reshape(*reader.navigation_shape, *movie.shape[1:])
    signal = hs.signals.Signal2D(data).as_lazy()
    signal.metadata.General.title = os.path.basename(folder)
    signal.metadata.set_item("Autopilot.run_folder", folder)
    signal.metadata.set_item("Autopilot.running", reader.running)
    signal.metadata.set_item("Autopilot.missing_points", int((latest < 0).sum()))
    return signal
