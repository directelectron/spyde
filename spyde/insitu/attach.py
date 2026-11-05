"""Attach aligned instrument channels to a loaded movie's signal tree.

This is the glue between the readers in this package and the app: it finds the
instrument record that belongs to a movie, solves the clock offset, resamples
the channels onto the frame times, and registers the interesting ones as named
navigator signals so they appear as chips beside the movie's own navigator.

Registering them as navigators rather than inventing a display is deliberate.
A named navigator already gets a chip, already stacks onto one shared time
cursor when several are selected (``navigator_views.select_navigator``), and
already drives the movie when dragged — all of which is written and tested. An
E/t trace is an ordinary line, so the generic stacked builder draws it
correctly; nothing here needs its own figure code.

Two entry points, one path underneath:

* :func:`discover_and_attach` — the automatic one, run when a movie loads. It
  scans the movie's own folder, and stays silent unless it finds a record that
  really does explain the movie (:func:`~spyde.insitu.align.match_runs`
  scores on duration agreement).
* :func:`attach_ec_file` — the manual one, behind File ▸ Load In-Situ Data…,
  for a record that lives somewhere else or that the scorer passed over.

Only the potential and current become chips. The full resampled table is left
on the tree as ``insitu_channels``, because a chip strip holding eleven
entries — most of them flags — makes the two that matter harder to find, not
easier.
"""
from __future__ import annotations

import logging
import os

import numpy as np

from spyde.insitu.align import align_clocks, match_runs, resample_to_frames
from spyde.insitu.de_movie import MovieClock, read_movie_clock
from spyde.insitu.eclab import EcRun, find_ec_runs, read_ec_file

log = logging.getLogger(__name__)

POTENTIAL_LANE = "Ewe (V)"
CURRENT_LANE = "I"

# Below this the current reads better in µA than in mA — a 0.0001 axis is
# unreadable. The lane NAME carries whichever unit was chosen, so the scaling
# is never silent.
_MICROAMP_CUTOFF_MA = 1.0


class AttachResult:
    """What an attach attempt did, for the caller's status line."""

    def __init__(self, run=None, alignment=None, lanes=(), reason=""):
        self.run = run
        self.alignment = alignment
        self.lanes = tuple(lanes)
        self.reason = reason

    def __bool__(self) -> bool:
        return bool(self.lanes)

    def describe(self) -> str:
        if not self:
            return self.reason or "no in-situ data attached"
        al = self.alignment
        return (
            f"Attached {self.run.technique} from {self.run.name} — "
            f"{self.run.n_points} samples aligned by {al.method}, "
            f"{abs(al.duration_mismatch_s):.2f} s over {self.run.duration:.1f} s, "
            f"{al.covered_fraction * 100:.0f}% of frames covered"
        )


def movie_clock_for(tree, path: str | None = None) -> MovieClock | None:
    """The tree's frame time base — cached on the tree by the loader, else read."""
    clock = getattr(tree, "insitu_clock", None)
    if clock is not None:
        return clock
    source = path or getattr(tree, "source_path", None)
    if not source:
        return None
    try:
        clock = read_movie_clock(source)
    except (FileNotFoundError, ValueError, OSError) as exc:
        log.debug("insitu: no frame time base for %s (%s)", source, exc)
        return None
    tree.insitu_clock = clock
    return clock


def _current_lane(values: np.ndarray) -> tuple[str, np.ndarray]:
    """Name and scale the current lane so its axis is readable."""
    finite = values[np.isfinite(values)]
    peak = float(np.max(np.abs(finite))) if finite.size else 0.0
    if 0 < peak < _MICROAMP_CUTOFF_MA:
        return f"{CURRENT_LANE} (µA)", values * 1e3
    return f"{CURRENT_LANE} (mA)", values


def _register_lanes(tree, columns: dict[str, np.ndarray]) -> list[str]:
    """Ensure the potential and current exist as named navigator signals.

    A navigator signal must be exactly as long as the movie's navigation axis
    (``_preprocess_navigator`` enforces it), and NaN would poison the display
    levels — so frames outside the instrument record are filled with the
    nearest in-record value here. The authoritative NaN-bearing arrays stay on
    ``tree.insitu_channels``; this is a display copy.

    Attaching twice is an ordinary thing to do — auto-discovery runs on open
    and the user may then pick a record by hand — so a lane that already
    exists has its VALUES replaced in place rather than being skipped or
    registered again. Re-registering would add a second plot state for the
    same name; skipping would leave the old run's trace on screen while the
    tree claims the new one. Mutating the existing signal's array keeps one
    plot state and one truth.

    Returns every lane now present, whether it was created or updated — so the
    caller can tell "this run has no E/I channel" (empty) from "nothing left
    to do" (non-empty), which is not the same outcome.
    """
    # The movie's OWN navigation calibration has to land on each lane's signal
    # axis. A 1-D selector turns a widget position into a frame index with
    # ``(x - offset) / scale`` read from the navigator plot's own signal axis,
    # so an uncalibrated lane is indexed in seconds against a scale of 1 — the
    # stacked view's x-axis reads 0…7913 instead of 0…259 s, and dragging its
    # cursor resolves to the wrong frame. ``calibrated_nav_signal`` is the
    # existing helper for exactly this; imported lazily so this package stays
    # usable without the backend.
    from spyde.backend._session_files import calibrated_nav_signal

    lanes: list[tuple[str, np.ndarray]] = []
    potential = next(
        (columns[k] for k in ("Ewe/V", "<Ewe>/V", "|E|/V") if k in columns), None
    )
    if potential is not None:
        lanes.append((POTENTIAL_LANE, potential))
    current = next(
        (columns[k] for k in ("<I>/mA", "I/mA", "|I|/mA") if k in columns), None
    )
    if current is not None:
        lanes.append(_current_lane(current))

    present: list[str] = []
    existing = getattr(tree, "navigator_signals", {})
    for name, values in lanes:
        display = _fill_edges(values).astype(np.float32)
        try:
            if name in existing:
                _replace_lane_data(existing[name], display)
            else:
                lane = calibrated_nav_signal(display, tree.root)
                lane.metadata.General.title = name
                tree.add_navigator_signal(name, lane)
            present.append(name)
        except Exception as exc:
            # Warning, not debug: a lane that silently fails to register looks
            # from the outside exactly like "this run has no E/I channel", and
            # chasing that took a round trip through the real app.
            log.warning("insitu: registering navigator %r failed: %s", name, exc)
    return present


def _replace_lane_data(entry, values: np.ndarray) -> None:
    """Overwrite an already-registered lane's samples in place.

    ``add_navigator_signal`` stores whatever ``_preprocess_navigator`` returned,
    and that is ALWAYS a list — ``[signal]`` for a plain trace, or
    ``[navigator, signal]`` for a navigated one — never the bare signal handed
    in. So unwrap before touching ``.data``, and write through the entries
    whose shape matches rather than rebinding the list.
    """
    targets = entry if isinstance(entry, (list, tuple)) else [entry]
    updated = 0
    for signal in targets:
        data = getattr(signal, "data", None)
        if data is None:
            continue
        shape = np.asarray(data).shape
        if shape != values.shape:
            continue
        signal.data = values.reshape(shape)
        updated += 1
    if not updated:
        shapes = [getattr(getattr(s, "data", None), "shape", None) for s in targets]
        raise ValueError(
            f"no registered lane array matches {values.shape} (found {shapes})"
        )


def _fill_edges(values: np.ndarray) -> np.ndarray:
    """Replace NaN with the nearest finite value (display copy only)."""
    out = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(out)
    if not finite.any():
        return np.zeros_like(out)
    idx = np.arange(out.size)
    out[~finite] = np.interp(idx[~finite], idx[finite], out[finite])
    return out


def _store(tree, clock, run, alignment, columns) -> None:
    tree.insitu_clock = clock
    tree.insitu_run = run
    tree.insitu_alignment = alignment
    tree.insitu_channels = columns


def attach_run(tree, clock: MovieClock, run: EcRun, alignment) -> AttachResult:
    """Resample *run* onto *clock*'s frames and register the display lanes."""
    columns = resample_to_frames(clock, run, alignment)
    nav = _nav_size(tree)
    if nav is not None and nav != clock.n_frames:
        return AttachResult(
            reason=(
                f"the movie has {nav} frames but "
                f"{os.path.basename(clock.timestamps_path or '?')} lists "
                f"{clock.n_frames} — cannot map instrument samples to frames"
            )
        )
    _store(tree, clock, run, alignment, columns)
    lanes = _register_lanes(tree, columns)
    if not lanes:
        return AttachResult(
            run, alignment, (),
            f"{run.name} has no potential or current channel — it records "
            f"{', '.join(sorted(run.channels)) or 'nothing'}",
        )
    return AttachResult(run, alignment, lanes)


def _nav_size(tree) -> int | None:
    try:
        shape = tree.root.axes_manager.navigation_shape
        return int(shape[0]) if len(shape) == 1 else None
    except Exception:
        return None


def discover_and_attach(tree, path: str) -> AttachResult:
    """Look beside *path* for an instrument record that explains this movie.

    Silent by design when nothing matches: an in-situ movie sitting in a folder
    of unrelated records is the common case, and a false attach is worse than
    none. :func:`~spyde.insitu.align.match_runs` only returns runs whose
    duration agrees with the movie's, so "found nothing" here means "found
    nothing that could plausibly have been recorded during this movie".
    """
    clock = movie_clock_for(tree, path)
    if clock is None:
        return AttachResult(reason="no frame timestamps beside this movie")
    directory = os.path.dirname(os.path.abspath(path)) or "."
    try:
        runs = find_ec_runs(directory)
    except OSError as exc:
        return AttachResult(reason=f"could not scan {directory}: {exc}")
    if not runs:
        return AttachResult(reason="no EC-Lab records in this folder")
    ranked = match_runs(clock, runs)
    if not ranked:
        return AttachResult(
            reason=(
                f"{len(runs)} EC-Lab record(s) beside this movie, none matching "
                f"its {clock.duration:.1f} s duration"
            )
        )
    run, alignment = ranked[0]
    return attach_run(tree, clock, run, alignment)


def attach_ec_file(tree, ec_path: str, *, movie_path: str | None = None,
                   method: str = "auto", lag_s: float | None = None,
                   ec_utc_offset_hours: float | None = None) -> AttachResult:
    """Attach one explicitly chosen instrument record to *tree*."""
    clock = movie_clock_for(tree, movie_path)
    if clock is None:
        return AttachResult(
            reason=("this dataset has no per-frame time base — a DE movie needs "
                    "its *_movie_timestamps.csv beside it")
        )
    try:
        run = read_ec_file(ec_path)
    except (ValueError, OSError) as exc:
        return AttachResult(reason=f"could not read {os.path.basename(ec_path)}: {exc}")
    if run.n_points < 2:
        return AttachResult(reason=f"{run.name} has no samples")
    try:
        alignment = align_clocks(
            clock, run, method=method, lag_s=lag_s,
            ec_utc_offset_hours=ec_utc_offset_hours,
        )
    except ValueError as exc:
        return AttachResult(reason=f"could not align {run.name}: {exc}")
    return attach_run(tree, clock, run, alignment)
