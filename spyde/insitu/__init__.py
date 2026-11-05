"""In-situ auxiliary channels — the non-image time series recorded *alongside* a
movie, and the clock arithmetic that puts them on the same axis as its frames.

An in-situ experiment produces two independent recordings of the same event:
the camera's frame stack, and whatever the stimulus instrument logged (a
potentiostat's E/I, a heating holder's temperature, a gas cell's pressure).
They come off two machines with two unsynchronised clocks and two sampling
rates, so "what was the potential in frame 4210?" is not a lookup — it is an
alignment problem. This package answers it in three pieces:

* :mod:`~spyde.insitu.de_movie` reads the Direct Electron sidecars that give a
  movie its REAL per-frame time base — ``*_movie_timestamps.csv`` (one row per
  saved frame) and ``*_info.txt``. Without them a movie only has the reader's
  uniform ``1/fps`` guess.
* :mod:`~spyde.insitu.eclab` reads BioLogic EC-Lab potentiostat records —
  ``.mpr`` (binary), ``.mpt``/``.txt`` (ASCII export) and ``.mps`` (settings).
* :mod:`~spyde.insitu.align` finds the offset between the two clocks and
  resamples the instrument channels onto the frame times.

The alignment is deliberately kept separate from both readers: it takes plain
time vectors, so a temperature log or any other per-time channel can reuse it
without either reader being involved.
"""
from __future__ import annotations

from spyde.insitu.align import (
    Alignment,
    align_clocks,
    ec_time_for_frames,
    frame_for_ec_sample,
    match_runs,
    resample_to_frames,
)
from spyde.insitu.de_movie import MovieClock, find_movie_sidecars, read_movie_clock
from spyde.insitu.eclab import EcRun, find_ec_runs, read_ec_file, read_mps

__all__ = [
    "Alignment",
    "EcRun",
    "MovieClock",
    "align_clocks",
    "ec_time_for_frames",
    "find_ec_runs",
    "find_movie_sidecars",
    "frame_for_ec_sample",
    "match_runs",
    "read_ec_file",
    "read_movie_clock",
    "read_mps",
    "resample_to_frames",
]
