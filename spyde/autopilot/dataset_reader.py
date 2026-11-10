# Copied from directelectron/de_autopilot de_automate/dataset_reader.py at a062374 by _sync.py; edit it there.
"""A run's dataset, read: the files DE-Server wrote, found through the index
Autopilot kept beside them, and the derived signals made from them.

A run folder holds ``run.json`` and, beside it, ``dataset.json``: the record
of what the run acquired, by reference (``de_automate/CLAUDE.md``, "Datasets
and recording"):

- per detector, the files the server wrote (path, format, the movie index of
  their first frame, frame count, bytes, frame shape and type), each marked
  ``complete`` once the server closed it, and for each complete file its
  index (``dataset/<detector>_<first>.npy``): every frame's navigation point,
  server frame number, pass and any series coordinates;
- the run's timeline files (``timeline/*.csv``): ``frames.csv`` labels each
  range of server frame numbers with how many of its images were taken while
  the instrument was still changing, joined to the index by ``frame``
  (:meth:`DatasetReader.frame_ranges`);
- the scan (type, field, navigation shape) and every navigation point's
  position in the field (``dataset/positions.npy``);
- the derived signals and saved vectors, with the parameters that made them;
- ``calibrations_used``.

This module is the one reader of that record, for Autopilot and SpyDE alike,
so it imports nothing but numpy and the standard library: SpyDE keeps a
byte-identical copy rather than installing Autopilot's backend.

- **Only complete files are read.** A file the index does not mark complete
  is refused (:class:`NotComplete`): the server may still be writing it, and
  an MRC's frame count is written as it closes.
- **Verified before trusted.** On open, a file's size, frame count, frame
  shape and type must be what the index says, or it is reported
  (:class:`Mismatch`), never read around.
- **Read lazily**, through a read-only memory map.
- **The open part comes from memory.** Mid-run, frames not yet in a complete
  file come from the app's preview (binned, the latest pattern per point),
  marked as a preview, never from the file being written.
- **On this machine only.** A run whose server was elsewhere recorded no files.
"""

from __future__ import annotations

import bisect
import csv
import json
import os
import struct
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

__all__ = [
    "ARRAYS",
    "FORMAT",
    "INDEX_FIELDS",
    "RECORD",
    "VERSION",
    "DatasetReader",
    "Frame",
    "Mismatch",
    "MovieFile",
    "NotComplete",
    "OrientationMap",
    "Unreadable",
    "movie_file",
]

#: The record's file, beside ``run.json``, and the folder of its arrays.
RECORD = "dataset.json"
ARRAYS = "dataset"
#: What the record says it is, and the version of its layout.
FORMAT = "de_autopilot.dataset"
VERSION = 1
#: The fields every index has, a row per movie frame: the navigation point
#: (-1 off the scan), the server's frame number (-1 unknown) and the pass.
#: Series coordinates follow as float64 fields of their own names. Whether a
#: frame was settled is the timeline's (:meth:`DatasetReader.frame_ranges`),
#: not repeated here.
INDEX_FIELDS = (("point", "<i8"), ("frame", "<i8"), ("pass", "<i4"))
#: The columns of the timeline's ``frames.csv`` this reads.
_FRAME_RANGE = np.dtype(
    [
        ("first_frame", "<i8"),
        ("last_frame", "<i8"),
        ("step", "<i8"),
        ("first_pass", "<i8"),
        ("last_pass", "<i8"),
        ("unsettled", "<i8"),
    ]
)

#: The MRC modes DE-Server writes a movie in, by its output data format.
#: Mode 0 is its unsigned 8-bit output (RAW8U), whatever MRC2014 says of
#: mode 0's sign.
_MRC_MOVIE_MODES = {0: "u1", 1: "i2", 2: "f4", 6: "u2"}
#: MRC2014's fixed header, and the machine stamp's first byte for each order.
_MRC_HEADER_BYTES = 1024
_MRC_LITTLE, _MRC_BIG = 0x44, 0x11
#: An EDAX UP2: version 3, 16-bit patterns after the header.
_UP2_VERSION = 3
_UP2_HEADER = struct.Struct("<4I")


class Unreadable(ValueError):
    """A movie this does not read: another format, or files that disagree
    with what their headers say."""


class NotComplete(Unreadable):
    """A file the index does not mark complete: it may still be written."""


class Mismatch(Unreadable):
    """A file that is not what the index says it is."""


# ── the server's files ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class MovieFile:
    """One file of a saved movie: its frames' type, shape and count, and the
    byte the first starts at."""

    path: str
    dtype: np.dtype
    offset: int
    frames: int
    shape: tuple[int, int]

    @property
    def frame_bytes(self) -> int:
        return int(np.prod(self.shape)) * self.dtype.itemsize

    @property
    def format(self) -> str:
        """``MRC`` or ``UP2``, by its extension."""
        return os.path.splitext(self.path)[1].lstrip(".").upper()


def movie_file(path: str) -> MovieFile:
    """*path*'s layout, as DE-Server writes a movie: MRC (``File/FileWriterMRC``)
    or EDAX's UP2 (``File/FileWriterUP2``). Any other format is
    :class:`Unreadable`, as is a file whose size disagrees with its header."""
    extension = os.path.splitext(path)[1].lower()
    if extension == ".mrc":
        found = _mrc(path)
    elif extension == ".up2":
        found = _up2(path)
    else:
        raise Unreadable(f"{path}: recovery reads MRC and UP2 movies, not {extension or path}")
    size = os.path.getsize(path)
    if size != found.offset + found.frames * found.frame_bytes:
        raise Unreadable(
            f"{path}: {size} bytes, where {found.frames} frames of {found.shape} take"
            f" {found.offset + found.frames * found.frame_bytes}"
        )
    return found


def _mrc(path: str) -> MovieFile:
    """The server writes the header, and with it the frame count, as it
    closes the file. Read here rather than through ``de_automate.mrc`` so
    this module stands alone."""
    with open(path, "rb") as fh:
        raw = fh.read(_MRC_HEADER_BYTES)
    if len(raw) < _MRC_HEADER_BYTES:
        raise Unreadable(f"{path}: {len(raw)} bytes is too short to be an MRC header")
    stamp = raw[212]
    endian = {_MRC_LITTLE: "<", _MRC_BIG: ">"}.get(stamp)
    if endian is None:  # no stamp: whichever byte order gives a known mode
        endian = next(
            (e for e in "<>" if struct.unpack(e + "i", raw[12:16])[0] in _MRC_MOVIE_MODES), None
        )
    if endian is None:
        raise Unreadable(f"{path}: not an MRC file — no machine stamp and no readable mode")
    nx, ny, nz, mode = struct.unpack(endian + "4i", raw[:16])
    (extended,) = struct.unpack(endian + "i", raw[92:96])
    if mode not in _MRC_MOVIE_MODES:
        raise Unreadable(f"{path}: MRC mode {mode} is not one DE-Server writes a movie in")
    if min(nx, ny) < 1 or nz < 0:
        raise Unreadable(f"{path}: not an MRC file — dimensions {nx}×{ny}×{nz}")
    return MovieFile(
        path,
        np.dtype(endian + _MRC_MOVIE_MODES[mode]),
        _MRC_HEADER_BYTES + max(0, extended),
        nz,
        (ny, nx),
    )


def _up2(path: str) -> MovieFile:
    """Version, width, height and the first pattern's byte, then uint16
    patterns row by row; the count is what the file holds."""
    with open(path, "rb") as fh:
        head = fh.read(_UP2_HEADER.size)
    if len(head) < _UP2_HEADER.size:
        raise Unreadable(f"{path}: too short to be a UP2")
    version, width, height, offset = _UP2_HEADER.unpack(head)
    if version != _UP2_VERSION or min(width, height) < 1:
        raise Unreadable(f"{path}: not a 16-bit UP2 (version {version}, {width}×{height})")
    frame_bytes = width * height * 2
    frames, extra = divmod(os.path.getsize(path) - offset, frame_bytes)
    if frames < 0 or extra:
        raise Unreadable(f"{path}: not a whole number of {width}×{height} patterns")
    return MovieFile(path, np.dtype("<u2"), offset, frames, (height, width))


def _mapped(found: MovieFile) -> np.memmap:
    """*found*'s frames, read-only and mapped."""
    return np.memmap(
        found.path, found.dtype, "r", offset=found.offset, shape=(found.frames, *found.shape)
    )


# ── derived signals ─────────────────────────────────────────────────────────


@dataclass
class OrientationMap:
    """An EBSD scan's orientations, point by point: exact for a sparse or
    non-grid scan, since every point carries its own position and whether it
    was measured.

    Each array has a row per navigation point: *euler*, Bunge angles in
    radians in the scan's frame; *phase*, -1 where nothing was indexed;
    *confidence*, the indexer's match score; *fit*, when the indexer reports
    one (else None); *measured*, indexed in this run; and *positions*,
    ``(row, column)`` in the scan field in whole scan pixels. *phases* names
    each phase number; *step_um* is the scan's step on the specimen when it
    is known.
    """

    euler: np.ndarray
    phase: np.ndarray
    confidence: np.ndarray
    measured: np.ndarray
    positions: np.ndarray
    navigation_shape: tuple[int, ...]
    phases: list[str] = field(default_factory=list)
    fit: np.ndarray | None = None
    step_um: float | None = None

    def save(self, path: str) -> str:
        """Write it as ``.npz``, atomically: a reader mid-run sees the last
        complete map, never a torn one."""
        arrays = {
            "euler": np.asarray(self.euler, np.float32),
            "phase": np.asarray(self.phase, np.int16),
            "confidence": np.asarray(self.confidence, np.float32),
            "measured": np.asarray(self.measured, bool),
            "positions": np.asarray(self.positions, np.int64),
            "navigation_shape": np.asarray(self.navigation_shape, np.int64),
            "about": np.asarray(json.dumps({"phases": self.phases, "step_um": self.step_um})),
        }
        if self.fit is not None:
            arrays["fit"] = np.asarray(self.fit, np.float32)
        temporary = path + ".tmp"
        with open(temporary, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(temporary, path)
        return path

    @classmethod
    def load(cls, path: str) -> OrientationMap:
        with np.load(path, allow_pickle=False) as found:
            about = json.loads(str(found["about"]))
            return cls(
                euler=found["euler"],
                phase=found["phase"],
                confidence=found["confidence"],
                measured=found["measured"],
                positions=found["positions"],
                navigation_shape=tuple(int(n) for n in found["navigation_shape"]),
                phases=list(about["phases"]),
                fit=found["fit"] if "fit" in found else None,
                step_um=about["step_um"],
            )


# ── reading a run ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Frame:
    """The data at one navigation point.

    *preview* is True when it came from the app's in-memory preview (binned
    by *binning*: the frames not yet in a complete file), False when read
    from a complete file. *movie* is its index in the saved movie (-1 when
    the preview does not know it); *path* the file it was read from.
    """

    data: np.ndarray
    point: int
    movie: int
    preview: bool = False
    binning: int = 1
    path: str | None = None


class Preview(Protocol):
    """What the reader asks of the app's preview: the latest pattern at a
    point with its movie index, or None; and how much it is binned."""

    binning: int

    def latest(self, point: int) -> tuple[np.ndarray, int] | None: ...


class DatasetReader:
    """A run folder's dataset, read through its index.

    *preview*, mid-run, is the app's in-memory preview of the frames not yet
    in a complete file; without one (SpyDE, or after the run) only complete
    files are read. :meth:`refresh` takes in what was recorded since.
    """

    def __init__(self, folder: str | os.PathLike, *, preview: Preview | None = None) -> None:
        self.folder = os.fspath(folder)
        self.preview = preview
        self._maps: dict[str, np.memmap] = {}
        self._indexed: dict[str, set[int]] = {}
        #: Per detector, the movie index of the latest frame at each point
        #: that is in a complete file; -1 where there is none.
        self._latest: dict[str, np.ndarray] = {}
        self.refresh()

    # ── the record ──
    def refresh(self) -> None:
        """Read the record again, and the index of every file completed since."""
        path = os.path.join(self.folder, RECORD)
        try:
            with open(path, encoding="utf-8") as fh:
                record = json.load(fh)
        except FileNotFoundError:
            raise Unreadable(f"{self.folder}: no {RECORD}; not an Autopilot run folder") from None
        if record.get("format") != FORMAT or int(record.get("version", 0)) > VERSION:
            raise Unreadable(
                f"{path}: a {record.get('format')} record of version {record.get('version')};"
                f" this reads {FORMAT} up to version {VERSION}"
            )
        self.record = record
        self.positions = np.load(self._here(record["scan"]["positions"]))
        for detector, entry in record["detectors"].items():
            latest = self._latest.setdefault(detector, np.full(len(self.positions), -1, np.int64))
            seen = self._indexed.setdefault(detector, set())
            for file in entry["files"]:
                if file["complete"] and file["first"] not in seen:
                    point = self.index(file)["point"]
                    on = np.flatnonzero(point >= 0)
                    np.maximum.at(latest, point[on], file["first"] + on)
                    seen.add(file["first"])

    @property
    def running(self) -> bool:
        """Whether the run was still going when the record was last written."""
        return bool(self.record.get("running"))

    @property
    def scan(self) -> dict:
        """The scan: ``type``, ``field`` (rows, columns), ``roi`` and
        ``navigation_shape``."""
        return dict(self.record["scan"])

    @property
    def navigation_shape(self) -> tuple[int, ...]:
        return tuple(int(n) for n in self.record["scan"]["navigation_shape"])

    @property
    def calibrations_used(self) -> list[dict]:
        return list(self.record.get("calibrations_used", []))

    @property
    def derived(self) -> list[dict]:
        """The derived signals: name, path (relative to the run folder),
        kind and the parameters that made each."""
        return list(self.record.get("derived", []))

    @property
    def vectors(self) -> list[dict]:
        """The saved vectors, as :attr:`derived`."""
        return list(self.record.get("vectors", []))

    def files(self, detector: str = "camera") -> list[dict]:
        """*detector*'s files, in movie order, complete or not."""
        return list(self.record["detectors"][detector]["files"])

    def index(self, file: dict) -> np.ndarray:
        """A complete *file*'s index: a row per frame, :data:`INDEX_FIELDS`
        then any series coordinates."""
        if not file["complete"]:
            raise NotComplete(f"{file['path']}: not complete, so not indexed yet")
        return np.load(self._here(file["index"]))

    def frame_ranges(self, detector: str = "camera") -> np.ndarray | None:
        """*detector*'s rows of the timeline's ``frames.csv``: each range of
        server frame numbers (``first_frame`` to ``last_frame``, ``step``
        apart), its passes, and how many of its images were ``unsettled``.
        An index row belongs to the range its ``frame`` falls in. None while
        the run has not written it (the timeline is labelled as it finishes)."""
        relative = (self.record.get("timeline") or {}).get("frames")
        path = self._here(relative) if relative else None
        if path is None or not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8", newline="") as fh:
            rows = [row for row in csv.DictReader(fh) if row["detector"] == detector]
        ranges = np.zeros(len(rows), _FRAME_RANGE)
        for at, row in enumerate(rows):
            ranges[at] = tuple(row[name] for name in _FRAME_RANGE.names)
        return ranges

    # ── the data ──
    def point(self, at: int | Sequence[int]) -> int:
        """*at* as a flat navigation point: a point number, or coordinates on
        the navigation grid (``(row, column)``)."""
        if np.ndim(at) == 0:
            return int(at)
        return int(np.ravel_multi_index(tuple(int(n) for n in at), self.navigation_shape))

    def latest(self, detector: str = "camera") -> np.ndarray:
        """Per navigation point, the movie index of the latest frame taken
        there that is in a complete file; -1 where there is none."""
        return self._latest[detector].copy()

    def frame(self, at: int | Sequence[int], *, detector: str = "camera") -> Frame | None:
        """The latest frame taken at *at*: from a complete file, or from the
        preview when it holds a later one (or the only one). None when
        neither has a frame there."""
        point = self.point(at)
        movie = int(self._latest[detector][point]) if detector in self._latest else -1
        preview = self.preview.latest(point) if self.preview is not None else None
        if preview is not None and (movie < 0 or preview[1] > movie):
            pattern, number = preview
            return Frame(pattern, point, int(number), preview=True, binning=self.preview.binning)
        if movie < 0:
            return None
        file = self._file_of(detector, movie)
        return Frame(self.read(file, movie - file["first"]), point, movie, path=file["path"])

    def read(self, file: dict, number: int) -> np.ndarray:
        """Frame *number* of *file*, counted from its first, read lazily."""
        return self.open(file)[number]

    def open(self, file: dict) -> np.memmap:
        """*file*'s frames, read-only and mapped: refused unless the index
        marks it complete, and only once its header and size agree with what
        the index says."""
        path = file["path"]
        if path in self._maps:
            return self._maps[path]
        if not file["complete"]:
            raise NotComplete(f"{path}: the server may still be writing it, so it is not opened")
        if not os.path.isfile(path):
            raise Unreadable(f"{path}: not on this machine")
        found = movie_file(path)
        expected = {
            "frames": file["frames"],
            "bytes": file["bytes"],
            "shape": tuple(file["shape"]),
            "type": np.dtype(file["dtype"]),
        }
        actual = {
            "frames": found.frames,
            "bytes": os.path.getsize(path),
            "shape": found.shape,
            "type": found.dtype,
        }
        wrong = [
            f"{key} {actual[key]}, where the index says {expected[key]}"
            for key in expected
            if actual[key] != expected[key]
        ]
        if wrong:
            raise Mismatch(f"{path}: " + "; ".join(wrong))
        self._maps[path] = frames = _mapped(found)
        return frames

    def signal(self, name: str) -> Any:
        """A derived signal by name: an orientation map as an
        :class:`OrientationMap`, a ``.npy`` as its array, an MRC image as its
        array (mapped, read-only)."""
        entry = next((d for d in self.derived if d["name"] == name), None)
        if entry is None:
            raise KeyError(f"no derived signal {name!r}")
        path = self._here(entry["path"])
        if entry.get("kind") == "orientation_map":
            return OrientationMap.load(path)
        if path.endswith(".npy"):
            return np.load(path, mmap_mode="r")
        frames = _mapped(movie_file(path))
        return frames[0] if len(frames) == 1 else frames

    def close(self) -> None:
        """Let go of every mapped file."""
        self._maps.clear()

    # ── helpers ──
    def _here(self, relative: str) -> str:
        return os.path.join(self.folder, relative)

    def _file_of(self, detector: str, movie: int) -> dict:
        files = [f for f in self.files(detector) if f["complete"]]
        file = files[bisect.bisect_right([f["first"] for f in files], movie) - 1]
        if not file["first"] <= movie < file["first"] + file["frames"]:
            raise Unreadable(f"movie frame {movie} is in no complete file")
        return file
