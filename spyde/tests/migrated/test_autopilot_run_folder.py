"""An Autopilot run folder — what Autopilot's "Open in SpyDE" hands over —
opens through Autopilot's own dataset reader as a lazy scan.

The folder is written here as Autopilot and DE-Server leave it: ``dataset.json``
beside ``run.json``, a movie saved in two files of DE-Server's MRC layout, and
an index per complete file of each frame's scan point.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

from spyde.autopilot import dataset_reader
from spyde.autopilot.run_folder import load_run, run_folder

#: A frame's pixels: small and not square, so a transposed read shows.
SHAPE = (4, 6)
#: The scan: 2 rows of 3 points, two passes.
NAVIGATION = (2, 3)


def _de_mrc(path, frames):
    """An MRC as DE-Server writes a movie: a 1024-byte header (nx, ny, nz,
    mode 2, the grid, MAP, a little-endian stamp), then float32 frames."""
    header = np.zeros(256, np.int32)
    nz, ny, nx = frames.shape
    header[[0, 1, 2, 3, 7, 8, 9]] = [nx, ny, nz, 2, nx, ny, nz]
    header[52], header[53] = 542130509, 17476
    path.write_bytes(header.tobytes() + frames.astype(np.float32).tobytes())
    return str(path)


def _run_folder(tmp_path, *, first_complete=True, last_complete=True):
    """Two raster passes of the scan, frame *n* filled with *n*, saved as
    frames 0-5 and 6-11, each complete or not."""
    points = int(np.prod(NAVIGATION))
    movie = np.repeat(np.arange(2 * points, dtype=np.float32), np.prod(SHAPE))
    movie = movie.reshape(-1, *SHAPE)
    arrays = tmp_path / dataset_reader.ARRAYS
    arrays.mkdir()
    rows, columns = np.mgrid[0 : NAVIGATION[0], 0 : NAVIGATION[1]]
    np.save(arrays / "positions.npy", np.stack([rows.ravel(), columns.ravel()], axis=1))
    files = []
    for first, complete in ((0, first_complete), (points, last_complete)):
        path = _de_mrc(tmp_path / f"run_{first}_movie.mrc", movie[first : first + points])
        entry = {"path": path, "first": first, "frames": points, "complete": complete,
                 "format": "MRC", "shape": list(SHAPE), "dtype": "<f4"}
        if complete:
            index = np.zeros(points, list(dataset_reader.INDEX_FIELDS))
            index["point"] = np.arange(points)
            index["frame"] = first + np.arange(points)
            index["pass"] = first // points
            np.save(arrays / f"camera_{first}.npy", index)
            entry.update(index=f"dataset/camera_{first}.npy", bytes=os.path.getsize(path))
        files.append(entry)
    record = {
        "format": dataset_reader.FORMAT,
        "version": dataset_reader.VERSION,
        "running": not last_complete,
        "scan": {"type": "Raster", "field": list(NAVIGATION), "roi": None,
                 "navigation_shape": list(NAVIGATION), "repeats": 2,
                 "positions": "dataset/positions.npy"},
        "coordinates": [],
        "timeline": {"frames": "timeline/frames.csv"},
        "detectors": {"camera": {"files": files, "note": ""}},
        "derived": [],
        "vectors": [],
        "calibrations_used": [],
    }
    (tmp_path / dataset_reader.RECORD).write_text(json.dumps(record), encoding="utf-8")
    (tmp_path / "run.json").write_text("{}", encoding="utf-8")
    return movie


def test_a_run_folder_is_recognised_by_its_record(tmp_path):
    _run_folder(tmp_path)
    assert run_folder(str(tmp_path)) == str(tmp_path)
    assert run_folder(str(tmp_path / dataset_reader.RECORD)) == str(tmp_path)
    assert run_folder(str(tmp_path / "run_0_movie.mrc")) is None
    assert run_folder(str(tmp_path / "dataset")) is None


def test_a_finished_run_opens_as_its_latest_frame_at_every_point(tmp_path):
    movie = _run_folder(tmp_path)
    signal = load_run(str(tmp_path))
    assert signal._lazy
    assert tuple(signal.data.shape) == (*NAVIGATION, *SHAPE)
    # the second pass: frame 6 + point at every point
    expected = movie[6:].reshape(*NAVIGATION, *SHAPE)
    np.testing.assert_array_equal(np.asarray(signal.data.compute()), expected)
    assert signal.metadata.Autopilot.missing_points == 0


def test_a_run_still_going_opens_only_its_complete_files(tmp_path):
    movie = _run_folder(tmp_path, last_complete=False)
    signal = load_run(str(tmp_path))
    np.testing.assert_array_equal(
        np.asarray(signal.data.compute()), movie[:6].reshape(*NAVIGATION, *SHAPE)
    )
    assert signal.metadata.Autopilot.running


def test_a_file_that_disagrees_with_the_record_is_refused(tmp_path):
    movie = _run_folder(tmp_path)
    _de_mrc(tmp_path / "run_6_movie.mrc", movie[6:11])  # five frames, not six
    with pytest.raises(dataset_reader.Mismatch, match="frames 5"):
        load_run(str(tmp_path))


def test_the_session_opens_a_run_folder_without_asking_its_scan_shape(window, tmp_path):
    _run_folder(tmp_path)
    session = window["window"]
    session.open_file(str(tmp_path))
    deadline = time.time() + 60.0
    while not session.signal_trees and time.time() < deadline:
        time.sleep(0.2)
    assert getattr(session, "_pending_load", None) is None
    root = session.signal_trees[0].root
    assert tuple(root.axes_manager.navigation_shape) == NAVIGATION[::-1]


def test_a_complete_file_is_read_from_its_own_place_in_the_movie(tmp_path):
    """Only the second file complete: its frames are movie frames 6-11, not
    the first six of anything."""
    movie = _run_folder(tmp_path, first_complete=False)
    signal = load_run(str(tmp_path))
    np.testing.assert_array_equal(
        np.asarray(signal.data.compute()), movie[6:].reshape(*NAVIGATION, *SHAPE)
    )


def test_the_copy_of_autopilots_reader_is_unedited():
    """The copy is Autopilot's, byte for byte after its header: a fix made
    here would be lost at the next sync, and the two apps would disagree."""
    from spyde.autopilot import _sync

    assert not _sync.drifted(), "re-run spyde.autopilot._sync rather than editing the copy"
