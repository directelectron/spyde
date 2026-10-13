"""Put a movie's frames and an instrument's samples on one time axis.

The two recordings share an experiment but not a clock. The camera stamps
frames with a free-running monotonic counter; the potentiostat stamps samples
with seconds since its own acquisition start, anchored to the naive local wall
clock of a different PC. Neither knows about the other, and the sampling rates
differ (here: 30.5 frame/s against 12.5 sample/s). Aligning them means finding
one number — the instrument-clock time of movie frame 0 — after which
everything else is interpolation.

Three ways to find that number, in descending order of trustworthiness:

``span``
    The two records have the same duration, so they were started and stopped
    together; align first sample to first frame. This needs no clock agreement
    at all, which is exactly why it is the default when it applies: it is
    immune to unsynchronised PCs, timezones and DST. A duration agreement to
    well under a second across several minutes is not a coincidence.

``absolute``
    Convert both to UTC and subtract. Requires the caller to say what timezone
    the instrument PC was in (nothing in an EC-Lab file records it) and leans
    on the camera's epoch stamp, which has 1 s resolution and is written at
    acquisition stop rather than start. Good for picking WHICH run pairs with
    which movie; too coarse to trust for the final offset when ``span`` is
    available.

``manual``
    The caller knows better — a trigger wire, a lab notebook, a feature both
    records saw. Always available as an override.

When both ``span`` and ``absolute`` are possible, the result reports the UTC
offset the span solution *implies* (:attr:`Alignment.implied_utc_offset_hours`).
That is the honest way to surface clock skew: if it lands near a whole number of
hours the two PCs simply disagreed about the timezone; if it does not, one of
their clocks was genuinely wrong, and you can see by how much.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import numpy as np

from spyde.insitu.de_movie import MovieClock, info_matches_movie
from spyde.insitu.eclab import DISCRETE_CHANNELS, EcRun

log = logging.getLogger(__name__)

# Durations this close (relative, or absolute seconds — whichever is looser)
# count as "started and stopped together".
SPAN_REL_TOL = 0.02
SPAN_ABS_TOL = 1.0


@dataclass
class Alignment:
    """The solved offset between a movie's clock and an instrument's clock.

    :attr:`lag_s` is the whole answer: the instrument-clock time (seconds since
    the instrument's acquisition start, the same origin as ``EcRun.time_s``) at
    which movie frame 0 was exposed. Everything else on this object is
    diagnostics for judging whether to believe it.
    """

    method: str
    lag_s: float
    duration_mismatch_s: float = 0.0
    overlap_s: float = 0.0
    covered_fraction: float = 0.0
    implied_utc_offset_hours: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        """True when the instrument covers essentially the whole movie."""
        return self.covered_fraction > 0.99

    def describe(self) -> str:
        lines = [
            f"method={self.method}  lag={self.lag_s:.3f} s  "
            f"duration mismatch={self.duration_mismatch_s:+.3f} s  "
            f"overlap={self.overlap_s:.1f} s  "
            f"frames covered={self.covered_fraction * 100:.1f}%"
        ]
        if self.implied_utc_offset_hours is not None:
            off = self.implied_utc_offset_hours
            nearest = round(off)
            lines.append(
                f"implied instrument-PC UTC offset={off:+.4f} h "
                f"({(off - nearest) * 3600:+.1f} s from UTC{nearest:+d})"
            )
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def _span_lag(clock: MovieClock, run: EcRun) -> float:
    """Instrument-clock time of frame 0 under the co-started assumption."""
    return float(run.time_s[0])


def _absolute_lag(clock: MovieClock, run: EcRun, utc_offset_hours: float,
                  notes: list[str]) -> float | None:
    """Instrument-clock time of frame 0 from the two absolute stamps."""
    epoch = clock.epoch_utc
    if epoch is None or run.start is None:
        return None
    if not info_matches_movie(clock):
        notes.append(
            "the info file's frame count does not match this movie, so its "
            "epoch stamp belongs to a different autosave session"
        )
    # The epoch stamp is written at acquisition STOP, so it dates the LAST
    # frame; walk back over the movie to reach frame 0.
    frame0_utc = epoch - clock.duration
    ec_start_utc = run.start_utc(utc_offset_hours)
    assert ec_start_utc is not None
    return float(frame0_utc - ec_start_utc.timestamp())


def _implied_utc_offset(clock: MovieClock, run: EcRun, lag_s: float) -> float | None:
    """The instrument PC's UTC offset implied by *lag_s*, in hours."""
    epoch = clock.epoch_utc
    if epoch is None or run.start is None:
        return None
    frame0_utc = epoch - clock.duration
    # Frame 0 in instrument-local wall clock:
    frame0_local = run.start + dt.timedelta(seconds=lag_s)
    naive_utc = frame0_local.replace(tzinfo=dt.timezone.utc).timestamp()
    return (naive_utc - frame0_utc) / 3600.0


def _coverage(clock: MovieClock, run: EcRun, lag_s: float) -> tuple[float, float]:
    """``(overlap_seconds, fraction_of_frames_with_instrument_data)``."""
    if clock.n_frames == 0 or run.n_points == 0:
        return 0.0, 0.0
    t = clock.t + lag_s
    lo, hi = float(run.time_s[0]), float(run.time_s[-1])
    inside = (t >= lo) & (t <= hi)
    overlap = max(0.0, min(t[-1], hi) - max(t[0], lo))
    return float(overlap), float(np.count_nonzero(inside) / t.size)


def align_clocks(
    clock: MovieClock,
    run: EcRun,
    *,
    method: str = "auto",
    lag_s: float | None = None,
    ec_utc_offset_hours: float | None = None,
) -> Alignment:
    """Solve the movie↔instrument clock offset.

    *method* is ``"auto"`` (span if the durations agree, else absolute),
    ``"span"``, ``"absolute"`` or ``"manual"``. ``"manual"`` requires *lag_s*;
    ``"absolute"`` requires *ec_utc_offset_hours* (the instrument PC's offset
    from UTC — no EC-Lab file records it).
    """
    if clock.n_frames == 0:
        raise ValueError("movie clock has no frames")
    if run.n_points == 0:
        raise ValueError(f"{run.name}: instrument run has no samples")

    notes: list[str] = []
    mismatch = run.duration - clock.duration
    span_tol = max(SPAN_ABS_TOL, SPAN_REL_TOL * max(run.duration, clock.duration))
    spans_agree = abs(mismatch) <= span_tol

    if method == "manual":
        if lag_s is None:
            raise ValueError("method='manual' requires lag_s")
        chosen, chosen_lag = "manual", float(lag_s)
    elif method == "span":
        chosen, chosen_lag = "span", _span_lag(clock, run)
        if not spans_agree:
            notes.append(
                f"durations differ by {mismatch:+.2f} s (tolerance {span_tol:.2f} s) "
                "— the records may not be co-extensive"
            )
    elif method == "absolute":
        if ec_utc_offset_hours is None:
            raise ValueError("method='absolute' requires ec_utc_offset_hours")
        got = _absolute_lag(clock, run, ec_utc_offset_hours, notes)
        if got is None:
            raise ValueError(
                "cannot align by absolute time: need both the movie's epoch "
                "stamp (from *_info.txt) and the run's acquisition start"
            )
        chosen, chosen_lag = "absolute", got
    elif method == "auto":
        if spans_agree:
            chosen, chosen_lag = "span", _span_lag(clock, run)
            notes.append(
                f"durations agree to {abs(mismatch):.3f} s over {run.duration:.1f} s "
                "— treating the two records as co-started"
            )
        elif ec_utc_offset_hours is not None:
            got = _absolute_lag(clock, run, ec_utc_offset_hours, notes)
            if got is None:
                raise ValueError("no usable absolute stamps for method='auto'")
            chosen, chosen_lag = "absolute", got
            notes.append(
                f"durations differ by {mismatch:+.2f} s, so the records are not "
                "co-extensive; fell back to the absolute clocks"
            )
        else:
            raise ValueError(
                f"durations differ by {mismatch:+.2f} s (tolerance {span_tol:.2f} s), "
                "so the records are not co-extensive. Pass ec_utc_offset_hours to "
                "align by absolute time, or lag_s with method='manual'."
            )
    else:
        raise ValueError(f"unknown alignment method {method!r}")

    overlap, covered = _coverage(clock, run, chosen_lag)
    if covered < 1.0:
        notes.append(
            f"{(1 - covered) * 100:.1f}% of frames fall outside the instrument "
            "record and will resample to NaN"
        )
    return Alignment(
        method=chosen,
        lag_s=chosen_lag,
        duration_mismatch_s=float(mismatch),
        overlap_s=overlap,
        covered_fraction=covered,
        implied_utc_offset_hours=_implied_utc_offset(clock, run, chosen_lag),
        notes=notes,
    )


def ec_time_for_frames(clock: MovieClock, alignment: Alignment) -> np.ndarray:
    """Each movie frame's time on the instrument clock (seconds since the
    instrument's acquisition start)."""
    return clock.t + alignment.lag_s


def resample_to_frames(
    clock: MovieClock,
    run: EcRun,
    alignment: Alignment,
    channels: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Resample the instrument's channels onto the movie's frame times.

    Returns one float64 array per channel, each ``clock.n_frames`` long, plus
    ``"time/s"`` (the instrument-clock time of every frame). Frames outside the
    instrument record are **NaN**, never clamped — a movie that ran on past the
    end of the sweep must not show the last potential held flat, because that
    is a measurement that was never made.

    Continuous channels are linearly interpolated. Channels that step rather
    than vary (:data:`~spyde.insitu.eclab.DISCRETE_CHANNELS` — cycle number,
    I Range, the mode/ox-red flags) take the nearest sample instead, since an
    interpolated cycle 1.5 or a current range between two ranges never existed.
    """
    t_frames = ec_time_for_frames(clock, alignment)
    t_ec = run.time_s
    order = np.argsort(t_ec)
    t_sorted = t_ec[order]
    inside = (t_frames >= t_sorted[0]) & (t_frames <= t_sorted[-1])

    wanted = channels if channels is not None else list(run.channels)
    out: dict[str, np.ndarray] = {"time/s": t_frames}
    for name in wanted:
        values = run.channels.get(name)
        if values is None:
            log.warning("align: no channel %r in %s", name, run.name)
            continue
        y = np.asarray(values, dtype=float)[order]
        if name in DISCRETE_CHANNELS or np.asarray(values).dtype.kind in "biu":
            idx = np.searchsorted(t_sorted, t_frames)
            idx = np.clip(idx, 1, t_sorted.size - 1)
            left = t_frames - t_sorted[idx - 1] <= t_sorted[idx] - t_frames
            resampled = y[np.where(left, idx - 1, idx)]
        else:
            resampled = np.interp(t_frames, t_sorted, y)
        out[name] = np.where(inside, resampled, np.nan)
    return out


def frame_for_ec_sample(
    clock: MovieClock, run: EcRun, alignment: Alignment
) -> np.ndarray:
    """The nearest movie frame index for each instrument sample.

    The inverse mapping of :func:`resample_to_frames`, for going the other way:
    "show me the frame where the current peaked". Samples falling outside the
    movie get ``-1``.
    """
    t_frames = ec_time_for_frames(clock, alignment)
    idx = np.searchsorted(t_frames, run.time_s)
    idx = np.clip(idx, 1, max(1, t_frames.size - 1))
    left = run.time_s - t_frames[idx - 1] <= t_frames[idx] - run.time_s
    nearest = np.where(left, idx - 1, idx)
    outside = (run.time_s < t_frames[0]) | (run.time_s > t_frames[-1])
    return np.where(outside, -1, nearest).astype(np.int64)


def match_runs(
    clock: MovieClock,
    runs: list[EcRun],
    *,
    ec_utc_offset_hours: float | None = None,
) -> list[tuple[EcRun, Alignment]]:
    """Rank instrument runs by how well each explains this movie.

    The scoring is duration agreement: of all the techniques in a session, the
    one whose record is the same length as the movie is the one that ran
    *during* it. Runs that cannot be aligned at all are dropped. Best first.
    """
    scored: list[tuple[float, EcRun, Alignment]] = []
    for run in runs:
        if run.n_points < 2 or clock.duration <= 0:
            continue
        try:
            alignment = align_clocks(
                clock, run, method="auto", ec_utc_offset_hours=ec_utc_offset_hours
            )
        except ValueError as exc:
            log.debug("align: %s does not match this movie (%s)", run.name, exc)
            continue
        penalty = abs(alignment.duration_mismatch_s) / max(clock.duration, 1e-9)
        scored.append((penalty - alignment.covered_fraction, run, alignment))
    scored.sort(key=lambda item: item[0])
    return [(run, alignment) for _, run, alignment in scored]
