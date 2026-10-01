"""Direct Electron movie sidecars — the REAL per-frame time base.

A DE acquisition writes, beside ``<name>_movie.mrc``:

* ``<name>_movie_timestamps.csv`` — ``Frame Index, Timestamp (s), Electrons``,
  one row per SAVED frame. ``Timestamp (s)`` is a free-running monotonic camera
  clock (values in the ~10^6 s range: uptime, not an epoch), so it says exactly
  *when each frame happened relative to every other frame* and nothing about
  where that sits in wall-clock time.
* ``<name>_info.txt`` — ``key = value`` acquisition metadata, including
  ``Timestamp (seconds since Epoch)``, the single absolute anchor available.

Two reasons this matters more than it looks:

**The uniform axis is a guess, and here it is a wrong one.** RosettaSciIO's MRC
reader derives the time scale from the metadata as
``1 / (Frames Per Second * Autosave Movie Sum Count)`` (``rsciio/mrc/_api.py``).
Summing ``N`` camera frames into one saved frame makes the saved period
``N / fps``, not ``1 / (fps*N)`` — so a 2-frame sum at 61.05 fps is calibrated
0.00819 s/frame when the timestamps say 0.03276 s/frame, a factor of ``N**2``
too fast. The CSV is ground truth and does not need the formula to be right.

**The epoch anchor is coarse and LATE.** ``Timestamp (seconds since Epoch)`` has
1 s resolution and is written when the info file is written — after the
acquisition stops (the same file reports ``Acquisition Status = Stopped`` and an
``Acquisition Counter`` already incremented past this dataset's). So it dates
the END of the movie, give or take the flush, and it is a UTC epoch while the
instrument log it must be matched against is normally naive local time. Treat it
as a coarse hint, never as a precise sync — which is why
:mod:`spyde.insitu.align` prefers to match on span and keeps the absolute route
as a cross-check.
"""
from __future__ import annotations

import csv
import datetime as dt
import glob
import logging
import os
import re
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

TIMESTAMPS_SUFFIX = "_movie_timestamps.csv"
INFO_SUFFIX = "_info.txt"

# The CSV column header, normalised (lowercased, stripped). Kept loose because
# the units suffix has changed between DE server versions.
_FRAME_COL = "frame index"
_TIME_COL = "timestamp"
_ELECTRONS_COL = "electrons"


@dataclass
class MovieClock:
    """The per-frame time base of one DE movie, plus its acquisition metadata.

    ``t_camera`` is the raw monotonic camera clock in seconds — only DIFFERENCES
    of it are meaningful. ``t`` is the same thing rebased to zero at the first
    saved frame, which is what you want as the movie's own time axis.
    """

    frame_index: np.ndarray
    t_camera: np.ndarray
    electrons: np.ndarray | None = None
    info: dict[str, str] = field(default_factory=dict)
    timestamps_path: str | None = None
    info_path: str | None = None

    @property
    def n_frames(self) -> int:
        return int(self.t_camera.size)

    @property
    def t(self) -> np.ndarray:
        """Frame times in seconds since the first saved frame."""
        if self.t_camera.size == 0:
            return self.t_camera
        return self.t_camera - self.t_camera[0]

    @property
    def duration(self) -> float:
        """First-frame-to-last-frame span in seconds (one period short of the
        total exposed time — see :attr:`frame_period`)."""
        if self.t_camera.size < 2:
            return 0.0
        return float(self.t_camera[-1] - self.t_camera[0])

    @property
    def frame_period(self) -> float:
        """Median saved-frame period in seconds (robust to a dropped frame)."""
        if self.t_camera.size < 2:
            return 0.0
        return float(np.median(np.diff(self.t_camera)))

    @property
    def epoch_utc(self) -> float | None:
        """``Timestamp (seconds since Epoch)`` from the info file, or None.

        Coarse (1 s) and stamped at acquisition STOP — see the module docstring.
        """
        raw = self.info.get("Timestamp (seconds since Epoch)")
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            log.warning("de_movie: uninterpretable epoch timestamp %r", raw)
            return None

    def epoch_datetime(self, tz: dt.tzinfo | None = dt.timezone.utc) -> dt.datetime | None:
        """:attr:`epoch_utc` as an aware datetime, or None."""
        ep = self.epoch_utc
        if ep is None:
            return None
        return dt.datetime.fromtimestamp(ep, dt.timezone.utc).astimezone(tz)

    def dropped_frames(self, tol: float = 0.5) -> np.ndarray:
        """Indices ``i`` where the gap to frame ``i+1`` exceeds ``1 + tol``
        periods — i.e. where the camera lost frames."""
        if self.t_camera.size < 3:
            return np.empty(0, dtype=int)
        gaps = np.diff(self.t_camera)
        period = self.frame_period
        if period <= 0:
            return np.empty(0, dtype=int)
        return np.flatnonzero(gaps > period * (1.0 + tol))

    @property
    def reader_frame_period(self) -> float | None:
        """The period RosettaSciIO's MRC reader would derive from the metadata.

        Exposed so a caller can SEE the discrepancy rather than guess at it;
        compare against :attr:`frame_period`.
        """
        try:
            fps = float(self.info["Frames Per Second"])
            n_sum = float(self.info.get("Autosave Movie Sum Count", 1) or 1)
        except (KeyError, TypeError, ValueError):
            return None
        if fps <= 0 or n_sum <= 0:
            return None
        return 1.0 / (fps * n_sum)


def _strip_suffix(path: str) -> str:
    """``…_movie.mrc`` / ``…_movie_timestamps.csv`` / ``…_info.txt`` → ``…``."""
    base = path
    for suffix in (TIMESTAMPS_SUFFIX, INFO_SUFFIX):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    base = os.path.splitext(base)[0]
    if base.endswith("_movie"):
        base = base[: -len("_movie")]
    return base


def find_movie_sidecars(path: str) -> tuple[str | None, str | None]:
    """Locate ``(timestamps_csv, info_txt)`` for *path* (the .mrc, or either
    sidecar). Returns None for whichever is absent.

    DE names the info file after the ACQUISITION while the movie file also
    carries a per-autosave-session number (``…_88071_run1_2616_movie.mrc`` vs
    ``…_88071_info.txt``), so an exact stem match is not enough — we widen to a
    glob in the same directory and take the longest common prefix.
    """
    stem = _strip_suffix(path)
    ts = stem + TIMESTAMPS_SUFFIX
    info = stem + INFO_SUFFIX
    found_ts = ts if os.path.exists(ts) else None
    found_info = info if os.path.exists(info) else None

    directory = os.path.dirname(os.path.abspath(path)) or "."
    name = os.path.basename(stem)
    for suffix, current in ((TIMESTAMPS_SUFFIX, found_ts), (INFO_SUFFIX, found_info)):
        if current is not None:
            continue
        best, best_len = None, 0
        for cand in glob.glob(os.path.join(glob.escape(directory), "*" + suffix)):
            cand_stem = os.path.basename(cand)[: -len(suffix)]
            shared = len(os.path.commonprefix([name, cand_stem]))
            # Require a real shared prefix, not just a leading date.
            if shared > best_len and shared >= min(8, len(cand_stem)):
                best, best_len = cand, shared
        if suffix == TIMESTAMPS_SUFFIX:
            found_ts = best
        else:
            found_info = best
    return found_ts, found_info


def read_info(path: str) -> dict[str, str]:
    """Parse a DE ``*_info.txt`` into ``{key: value}`` (both stripped)."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            key, sep, value = line.partition("=")
            if not sep:
                continue
            out[key.strip()] = value.strip()
    return out


def read_timestamps(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Parse a DE ``*_movie_timestamps.csv`` → ``(frame_index, t_camera, electrons)``.

    Column order is taken from the header rather than assumed, and a trailing
    torn row (acquisition killed mid-write) is dropped rather than raising.
    """
    frames: list[int] = []
    times: list[float] = []
    electrons: list[float] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path}: empty timestamps file") from None
        norm = [c.strip().lower() for c in header]

        def _col(want: str) -> int | None:
            for i, c in enumerate(norm):
                if c.startswith(want):
                    return i
            return None

        i_frame, i_time, i_el = _col(_FRAME_COL), _col(_TIME_COL), _col(_ELECTRONS_COL)
        if i_time is None:
            raise ValueError(f"{path}: no 'Timestamp' column in header {header!r}")
        for row_no, row in enumerate(reader, start=2):
            if not row or not row[0].strip():
                continue
            try:
                t = float(row[i_time])
                f = int(float(row[i_frame])) if i_frame is not None else len(times)
                e = float(row[i_el]) if i_el is not None and i_el < len(row) else np.nan
            except (IndexError, ValueError):
                log.warning("de_movie: dropping unparseable row %d of %s", row_no, path)
                continue
            frames.append(f)
            times.append(t)
            electrons.append(e)

    t = np.asarray(times, dtype=float)
    idx = np.asarray(frames, dtype=np.int64)
    el = np.asarray(electrons, dtype=float)
    return idx, t, (None if el.size == 0 or np.all(np.isnan(el)) else el)


def read_movie_clock(path: str) -> MovieClock:
    """Read the frame time base for the movie at *path*.

    *path* may be the ``.mrc``, the timestamps ``.csv`` or the ``_info.txt`` —
    the siblings are located automatically. Raises ``FileNotFoundError`` if no
    timestamps file can be found, since without it there is no real time base.
    """
    ts_path, info_path = find_movie_sidecars(path)
    if ts_path is None:
        raise FileNotFoundError(
            f"no '*{TIMESTAMPS_SUFFIX}' sidecar found beside {path!r}; "
            "the movie has no per-frame time base without it"
        )
    idx, t, electrons = read_timestamps(ts_path)
    info = read_info(info_path) if info_path else {}

    if t.size and np.any(np.diff(t) <= 0):
        log.warning("de_movie: %s is not strictly increasing", ts_path)

    clock = MovieClock(
        frame_index=idx,
        t_camera=t,
        electrons=electrons,
        info=info,
        timestamps_path=ts_path,
        info_path=info_path,
    )
    reader_period = clock.reader_frame_period
    if reader_period and clock.frame_period > 0:
        ratio = clock.frame_period / reader_period
        if abs(ratio - 1.0) > 0.01:
            log.info(
                "de_movie: %s measured frame period %.6f s vs reader-derived "
                "%.6f s (x%.3g) — using the timestamps",
                os.path.basename(ts_path), clock.frame_period, reader_period, ratio,
            )
    return clock


_FRAMES_WRITTEN_KEYS = ("Autosave Movie Frames Written", "Number of Frames Processed")


def info_matches_movie(clock: MovieClock) -> bool:
    """True when the info file's frame count agrees with the timestamps file.

    DE writes ONE info file per acquisition but a new movie file per autosave
    session, so a mismatched count means the info file (and therefore its epoch
    stamp) belongs to a DIFFERENT session and must not be used as this movie's
    absolute anchor.
    """
    for key in _FRAMES_WRITTEN_KEYS:
        raw = clock.info.get(key)
        if raw is None:
            continue
        try:
            return int(float(raw)) == clock.n_frames
        except ValueError:
            continue
    return False


def _info_float(info: dict, key: str) -> float | None:
    raw = info.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def spatial_calibration(info: dict) -> tuple[float, float, str] | None:
    """``(scale_y, scale_x, units)`` for a DE frame, or None if undeterminable.

    DE records the real-space pixel size as ``Specimen Pixel Size X/Y
    (nanometers)`` and the reciprocal one as ``Diffraction Pixel Size X/Y``,
    with ``-1`` meaning "not applicable". Which pair applies is decided by
    ``Instrument Project Camera Length (centimeters)``: a genuine diffraction
    exposure has a POSITIVE camera length.

    RosettaSciIO gets that test wrong — it asks ``camera_length != -1``, so the
    ``0`` an imaging exposure records reads as "has a camera length" and the
    frame is calibrated as diffraction. The reciprocal pixel size is then the
    unset ``-1``, which survives its own ``== -1`` guard (the value is a string
    at that point), and a TEM image ends up at ``-1.00 nm^-1`` per pixel with
    the correct ``1.14786 nm`` sitting unused in the same file.

    So the sentinel test here is ``> 0``, applied to both candidates: a scale
    is only used if it is a real positive number.
    """
    camera_length = _info_float(info, "Instrument Project Camera Length (centimeters)")
    diff_x = _info_float(info, "Diffraction Pixel Size X")
    diff_y = _info_float(info, "Diffraction Pixel Size Y")
    spec_x = _info_float(info, "Specimen Pixel Size X (nanometers)")
    spec_y = _info_float(info, "Specimen Pixel Size Y (nanometers)")

    diffracting = bool(camera_length and camera_length > 0)
    if diffracting and diff_x and diff_x > 0 and diff_y and diff_y > 0:
        return diff_y, diff_x, "nm^-1"
    if spec_x and spec_x > 0 and spec_y and spec_y > 0:
        return spec_y, spec_x, "nm"
    # Imaging exposure with no specimen pixel size, or a diffraction one with
    # no diffraction pixel size — nothing trustworthy to say.
    if diff_x and diff_x > 0 and diff_y and diff_y > 0:
        return diff_y, diff_x, "nm^-1"
    return None


def parse_de_datetime(text: str) -> dt.datetime | None:
    """Parse the ``YYYYMMDD`` prefix DE puts on dataset names, for a sanity date."""
    m = re.match(r"(\d{4})(\d{2})(\d{2})", text.strip())
    if not m:
        return None
    try:
        return dt.datetime(int(m[1]), int(m[2]), int(m[3]))
    except ValueError:
        return None
