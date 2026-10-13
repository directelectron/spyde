"""In-situ auxiliary channels: DE frame timestamps, EC-Lab records, clock align.

Everything here is synthesised into ``tmp_path`` — no instrument files on disk —
but the synthesis follows the real byte layout, so the ``.mpr`` builder below
doubles as the format's specification. The numbers in
:class:`TestRealWorldShape` are taken from a real paired acquisition (a DE
Apollo movie and a BioLogic SP-200 cyclic voltammogram recorded together) and
pin the two behaviours that make the alignment work at all: matching on span
rather than on the two disagreeing wall clocks, and refusing to invent
instrument data outside the record.
"""
from __future__ import annotations

import datetime as dt
import struct

import numpy as np
import pytest

from spyde.insitu import align as align_mod
from spyde.insitu import de_movie, eclab

# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

MPR_HEADER = eclab.MPR_MAGIC + b" " * 25 + b"\x00" * 4  # 52 bytes, as EC-Lab writes
_DATA_START = 1007  # padding before the records, EC-Lab v11


def _module(short: str, long: str, version: int, body: bytes) -> bytes:
    """One ``MODULE`` block in the 64-bit-length layout."""
    return (
        b"MODULE"
        + short.encode().ljust(10)
        + long.encode().ljust(25)
        + struct.pack("<I", 0xFFFFFFFF)
        + struct.pack("<Q", len(body))
        + struct.pack("<I", version)
        + b"11/17/25"
        + body
    )


def _log_module(start: dt.datetime) -> bytes:
    """A ``VMP LOG`` carrying the acquisition start as an OLE date."""
    body = bytearray(8010)
    days = (start - dt.datetime(1899, 12, 30)).total_seconds() / 86400.0
    struct.pack_into("<d", body, 585, days)
    return _module("VMP LOG", "VMP LOG", 10, bytes(body))


def build_mpr(
    *,
    ids: tuple[int, ...],
    records: np.ndarray,
    start: dt.datetime = dt.datetime(2025, 11, 17, 14, 50, 59, 687000),
    settings_text: bytes = b"Cyclic Voltammetry\x00",
) -> bytes:
    """Assemble a minimal but structurally faithful ``.mpr``."""
    npts = len(records)
    header = struct.pack("<I", npts) + struct.pack("<H", len(ids))
    header += struct.pack(f"<{len(ids)}H", *ids)
    body = header + b"\x00" * (_DATA_START - len(header)) + records.tobytes()
    return (
        MPR_HEADER
        + _module("VMP Set", "VMP settings", 10, settings_text.ljust(6654, b"\x00"))
        + _module("VMP data", "VMP data", 11, body)
        + _log_module(start)
    )


# A real CV record's column set: 5 flags packed into one u1, then 8 columns.
CV_IDS = (1, 2, 3, 21, 65, 4, 19, 6, 11, 24, 434, 39)
CV_DTYPE = np.dtype([
    ("flags", "<u1"), ("time/s", "<f8"), ("control/V", "<f4"), ("Ewe/V", "<f4"),
    ("<I>/mA", "<f8"), ("cycle number", "<f8"), ("(Q-Qo)/C", "<f4"),
    ("I Range", "<u2"),
])


def cv_records(n=100, t0=31.56, dt_s=0.08) -> np.ndarray:
    rec = np.zeros(n, dtype=CV_DTYPE)
    t = t0 + dt_s * np.arange(n)
    rec["time/s"] = t
    rec["Ewe/V"] = 0.02 * (t - t0)          # a 20 mV/s ramp
    rec["control/V"] = rec["Ewe/V"]
    rec["<I>/mA"] = 1e-4 * np.sin(t)
    rec["cycle number"] = np.where(np.arange(n) < n // 2, 1.0, 2.0)
    rec["I Range"] = np.where(np.arange(n) < n // 2, 52, 53)
    rec["flags"] = np.where(np.arange(n) % 2, 0x04, 0x00) | 0x02  # ox/red + mode
    return rec


def write_movie(tmp_path, *, stem="20251117_88071", n=200, period=0.03276,
                t0=1133858.28439, epoch=None, info_extra=None, frames_written=None):
    """Write a DE timestamps CSV (+ info.txt) and return the .mrc path."""
    csv_path = tmp_path / f"{stem}{de_movie.TIMESTAMPS_SUFFIX}"
    lines = ["Frame Index, Timestamp (s), Electrons"]
    for i in range(n):
        lines.append(f"{i}, {t0 + i * period:.6f}, {31000000 + i}")
    csv_path.write_text("\n".join(lines) + "\n")

    info = {
        "Frames Per Second": "61.05006",
        "Autosave Movie Sum Count": "2",
        "Autosave Movie Frames Written": str(n if frames_written is None else frames_written),
        "Acquisition Status": "Stopped",
    }
    if epoch is not None:
        info["Timestamp (seconds since Epoch)"] = str(epoch)
    info.update(info_extra or {})
    info_path = tmp_path / f"{stem}{de_movie.INFO_SUFFIX}"
    info_path.write_text(
        "\n".join(f"{k:<60} = {v}" for k, v in info.items()) + "\n"
    )
    mrc = tmp_path / f"{stem}_movie.mrc"
    mrc.write_bytes(b"")
    return str(mrc)


EC_START = dt.datetime(2025, 11, 17, 14, 50, 59, 687000)


def _eu(value: float, fmt: str) -> str:
    """A number as EC-Lab writes it on a decimal-comma locale."""
    return format(value, fmt).replace(".", ",")


def write_mpt(tmp_path, name="run.txt", *, absolute=False, n=50):
    """An EC-Lab ASCII export.

    ``Nb header lines`` counts every line up to AND INCLUDING the column
    header, so it is derived from the header list rather than hard-coded —
    getting that off by one is the classic way to misread these files.
    """
    header = [
        "EC-Lab ASCII FILE",
        "Nb header lines : {n}",
        "",
        "Cyclic Voltammetry",
        "",
        "Acquisition started on : 11/17/2025 14:50:59.687",
        "Device : SP-200 (SN 1593)",
        "",
        "ox/red\ttime/s\tcontrol/V\tEwe/V\t<I>/mA\t",
    ]
    header[1] = header[1].format(n=len(header))

    rows = []
    for i in range(n):
        t = 31.56 + 0.08 * i
        ramp = 0.02 * (t - 31.56)
        # A datetime keeps its dot; only the numbers take a decimal comma.
        when = (EC_START + dt.timedelta(seconds=t)).strftime("%m/%d/%Y %H:%M:%S.%f")[:-2]
        t_col = when if absolute else _eu(t, ".15E")
        rows.append("\t".join([
            str(i % 2), t_col, _eu(ramp, ".7E"), _eu(ramp, ".7E"),
            _eu(1e-4 * np.sin(t), ".15E"),
        ]))

    path = tmp_path / name
    path.write_text("\n".join(header + rows) + "\n", encoding="latin1")
    return str(path)


# --------------------------------------------------------------------------
# DE movie sidecars
# --------------------------------------------------------------------------

class TestMovieClock:
    def test_reads_frame_times(self, tmp_path):
        mrc = write_movie(tmp_path, n=200)
        clock = de_movie.read_movie_clock(mrc)
        assert clock.n_frames == 200
        assert clock.frame_period == pytest.approx(0.03276, abs=1e-9)
        assert clock.duration == pytest.approx(199 * 0.03276, abs=1e-6)
        assert clock.t[0] == 0.0

    def test_finds_sidecars_from_any_member(self, tmp_path):
        mrc = write_movie(tmp_path)
        for probe in (mrc, mrc.replace("_movie.mrc", de_movie.TIMESTAMPS_SUFFIX),
                      mrc.replace("_movie.mrc", de_movie.INFO_SUFFIX)):
            ts, info = de_movie.find_movie_sidecars(probe)
            assert ts is not None and info is not None

    def test_finds_sidecar_when_movie_carries_a_session_number(self, tmp_path):
        """DE names the info file per ACQUISITION but the movie per autosave
        session, so the stems do not match exactly."""
        write_movie(tmp_path, stem="20251117_88071")
        movie = tmp_path / "20251117_88071_run1_2616_movie.mrc"
        movie.write_bytes(b"")
        ts, info = de_movie.find_movie_sidecars(str(movie))
        assert ts is not None and info is not None

    def test_reader_derived_period_disagrees_with_the_timestamps(self, tmp_path):
        """RosettaSciIO derives 1/(fps*sum) where the true saved period is
        sum/fps — a factor of sum**2. The CSV is ground truth; this pins that
        the discrepancy is visible rather than silent."""
        clock = de_movie.read_movie_clock(write_movie(tmp_path))
        assert clock.reader_frame_period == pytest.approx(1 / (61.05006 * 2))
        assert clock.frame_period / clock.reader_frame_period == pytest.approx(4.0, rel=1e-3)

    def test_epoch_is_read_as_utc(self, tmp_path):
        clock = de_movie.read_movie_clock(write_movie(tmp_path, epoch=1763387769))
        assert clock.epoch_utc == 1763387769.0
        assert clock.epoch_datetime().strftime("%Y-%m-%d %H:%M:%S") == "2025-11-17 13:56:09"

    def test_info_frame_count_gates_the_epoch_anchor(self, tmp_path):
        """A DE acquisition writes ONE info file but a movie per autosave
        session; a count mismatch means the epoch stamp is another session's."""
        good = de_movie.read_movie_clock(write_movie(tmp_path, n=50))
        assert de_movie.info_matches_movie(good)
        other = tmp_path / "b"
        other.mkdir()
        bad = de_movie.read_movie_clock(
            write_movie(other, n=50, frames_written=3005)
        )
        assert not de_movie.info_matches_movie(bad)

    def test_detects_dropped_frames(self, tmp_path):
        mrc = write_movie(tmp_path, n=10)
        csv = mrc.replace("_movie.mrc", de_movie.TIMESTAMPS_SUFFIX)
        lines = open(csv).read().splitlines()
        # Blow a hole in the middle: frame 5 arrives three periods late.
        rows = [lines[0]]
        for i, line in enumerate(lines[1:]):
            idx, t, e = line.split(",")
            rows.append(f"{idx},{float(t) + (0.0655 if i >= 5 else 0):.6f},{e}")
        open(csv, "w").write("\n".join(rows) + "\n")
        clock = de_movie.read_movie_clock(mrc)
        assert clock.dropped_frames().tolist() == [4]

    def test_missing_timestamps_raises(self, tmp_path):
        (tmp_path / "x_movie.mrc").write_bytes(b"")
        with pytest.raises(FileNotFoundError):
            de_movie.read_movie_clock(str(tmp_path / "x_movie.mrc"))

    def test_torn_row_is_dropped_not_fatal(self, tmp_path):
        mrc = write_movie(tmp_path, n=10)
        csv = mrc.replace("_movie.mrc", de_movie.TIMESTAMPS_SUFFIX)
        with open(csv, "a") as fh:
            fh.write("10, \n")  # acquisition killed mid-write
        assert de_movie.read_movie_clock(mrc).n_frames == 10


# --------------------------------------------------------------------------
# EC-Lab records
# --------------------------------------------------------------------------

class TestMpr:
    def test_round_trips_columns_and_flags(self, tmp_path):
        rec = cv_records(n=100)
        path = tmp_path / "r_02_CV_C01.mpr"
        path.write_bytes(build_mpr(ids=CV_IDS, records=rec))
        run = eclab.read_mpr(str(path))

        assert run.n_points == 100
        assert run.time_s == pytest.approx(rec["time/s"])
        assert run.channels["Ewe/V"] == pytest.approx(rec["Ewe/V"])
        assert run.channels["<I>/mA"] == pytest.approx(rec["<I>/mA"])
        assert run.channels["I Range"].tolist() == rec["I Range"].tolist()
        # flags share one u1: ox/red alternates, mode is the low 2 bits
        assert run.channels["ox/red"].tolist() == [bool(i % 2) for i in range(100)]
        assert set(np.unique(run.channels["mode"])) == {2}
        assert run.unknown_columns == ()

    def test_reads_the_acquisition_start(self, tmp_path):
        path = tmp_path / "r_02_CV_C01.mpr"
        path.write_bytes(build_mpr(ids=CV_IDS, records=cv_records()))
        run = eclab.read_mpr(str(path))
        assert run.start == dt.datetime(2025, 11, 17, 14, 50, 59, 687000)

    def test_technique_falls_back_to_the_filename_code(self, tmp_path):
        path = tmp_path / "r_02_CV_C01.mpr"
        path.write_bytes(build_mpr(ids=CV_IDS, records=cv_records(),
                                   settings_text=b"\x00"))
        assert eclab.read_mpr(str(path)).technique == "Cyclic Voltammetry"

    def test_one_unknown_column_is_pinned_by_the_record_size(self, tmp_path):
        """An unrecognised ID is readable when the byte arithmetic leaves only
        one possible width."""
        dtype = np.dtype(CV_DTYPE.descr + [("mystery", "<f8")])
        rec = np.zeros(60, dtype=dtype)
        rec["time/s"] = np.arange(60) * 0.08
        rec["mystery"] = np.arange(60) * 3.0
        path = tmp_path / "r_02_CV_C01.mpr"
        path.write_bytes(build_mpr(ids=CV_IDS + (9999,), records=rec))
        run = eclab.read_mpr(str(path))
        assert run.unknown_columns == (9999,)
        assert run.channels["unknown_9999"] == pytest.approx(rec["mystery"])

    def test_two_unknown_columns_refuse_rather_than_guess(self, tmp_path):
        dtype = np.dtype(CV_DTYPE.descr + [("a", "<f8"), ("b", "<f4")])
        rec = np.zeros(60, dtype=dtype)
        path = tmp_path / "r_02_CV_C01.mpr"
        path.write_bytes(build_mpr(ids=CV_IDS + (9998, 9999), records=rec))
        with pytest.raises(eclab.UnsupportedColumns, match="9998"):
            eclab.read_mpr(str(path))

    def test_rejects_a_non_mpr(self, tmp_path):
        path = tmp_path / "nope.mpr"
        path.write_bytes(b"not a biologic file")
        with pytest.raises(ValueError, match="not a BioLogic"):
            eclab.read_mpr(str(path))


class TestMpt:
    def test_reads_decimal_comma_and_elapsed_time(self, tmp_path):
        run = eclab.read_mpt(write_mpt(tmp_path, n=50))
        assert run.n_points == 50
        assert run.technique == "Cyclic Voltammetry"
        assert run.start == dt.datetime(2025, 11, 17, 14, 50, 59, 687000)
        assert run.time_s[0] == pytest.approx(31.56)
        assert run.sample_period == pytest.approx(0.08, abs=1e-6)
        assert run.channels["Ewe/V"][0] == pytest.approx(0.0, abs=1e-9)

    def test_absolute_time_column_becomes_elapsed_seconds(self, tmp_path):
        """EC-Lab can export time as a wall-clock stamp; both conventions must
        land on the same elapsed axis."""
        elapsed = eclab.read_mpt(write_mpt(tmp_path, "e.txt", n=30))
        absolute = eclab.read_mpt(
            write_mpt(tmp_path, "a.txt", absolute=True, n=30)
        )
        assert absolute.time_s == pytest.approx(elapsed.time_s, abs=1e-3)

    def test_dispatch_reads_either_format(self, tmp_path):
        mpr = tmp_path / "r_02_CV_C01.mpr"
        mpr.write_bytes(build_mpr(ids=CV_IDS, records=cv_records()))
        assert eclab.read_ec_file(str(mpr)).n_points == 100
        assert eclab.read_ec_file(write_mpt(tmp_path, n=12)).n_points == 12

    def test_find_ec_runs_dedups_a_run_exported_twice(self, tmp_path):
        """The same technique as .mpr plus two ASCII exports is ONE run."""
        rec = cv_records(n=50)
        (tmp_path / "r_02_CV_C01.mpr").write_bytes(build_mpr(ids=CV_IDS, records=rec))
        write_mpt(tmp_path, "r_02_CV_C01-et.txt", n=50)
        write_mpt(tmp_path, "r_02_CV_C01 (2).txt", n=50)
        runs = eclab.find_ec_runs(str(tmp_path))
        assert len(runs) == 1
        assert runs[0].path.endswith(".mpr")  # binary preferred


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------

def _paired(tmp_path, *, n_frames=200, period=0.03276, n_ec=None, ec_dt=0.08,
            epoch=None):
    """A movie and an EC run of matching duration (co-started)."""
    mrc = write_movie(tmp_path, n=n_frames, period=period, epoch=epoch)
    clock = de_movie.read_movie_clock(mrc)
    if n_ec is None:
        # Round UP so the record spans the whole movie — a co-started pair.
        n_ec = int(np.ceil(clock.duration / ec_dt)) + 1
    path = tmp_path / "r_02_CV_C01.mpr"
    path.write_bytes(build_mpr(ids=CV_IDS, records=cv_records(n=n_ec, dt_s=ec_dt)))
    return clock, eclab.read_mpr(str(path))


class TestAlign:
    def test_span_match_needs_no_clock_agreement(self, tmp_path):
        clock, run = _paired(tmp_path)
        al = align_mod.align_clocks(clock, run)
        assert al.method == "span"
        assert al.lag_s == pytest.approx(run.time_s[0])
        assert al.covered_fraction == pytest.approx(1.0, abs=0.01)
        assert abs(al.duration_mismatch_s) < 0.1

    def test_mismatched_durations_refuse_to_span_match(self, tmp_path):
        clock, run = _paired(tmp_path, n_ec=30)  # EC far shorter than the movie
        with pytest.raises(ValueError, match="not co-extensive"):
            align_mod.align_clocks(clock, run)

    def test_absolute_method_uses_the_epoch_and_a_stated_utc_offset(self, tmp_path):
        clock, run = _paired(tmp_path, n_ec=30, epoch=1763387769)
        al = align_mod.align_clocks(
            clock, run, method="absolute", ec_utc_offset_hours=1.0
        )
        assert al.method == "absolute"
        # frame 0 = epoch - duration; EC start = 14:50:59.687 CET = 13:50:59.687 UTC
        expected = (1763387769 - clock.duration) - dt.datetime(
            2025, 11, 17, 13, 50, 59, 687000, tzinfo=dt.timezone.utc
        ).timestamp()
        assert al.lag_s == pytest.approx(expected, abs=1e-6)

    def test_absolute_method_requires_an_offset(self, tmp_path):
        clock, run = _paired(tmp_path, epoch=1763387769)
        with pytest.raises(ValueError, match="ec_utc_offset_hours"):
            align_mod.align_clocks(clock, run, method="absolute")

    def test_manual_override(self, tmp_path):
        clock, run = _paired(tmp_path)
        al = align_mod.align_clocks(clock, run, method="manual", lag_s=12.5)
        assert (al.method, al.lag_s) == ("manual", 12.5)

    def test_implied_utc_offset_exposes_clock_skew(self, tmp_path):
        """The span solution implies what the instrument PC's UTC offset must
        have been — the honest way to surface two disagreeing clocks."""
        clock, run = _paired(tmp_path, epoch=1763387769)
        al = align_mod.align_clocks(clock, run)
        assert al.implied_utc_offset_hours is not None
        assert round(al.implied_utc_offset_hours) == 1  # instrument PC on UTC+1


class TestResample:
    def test_channels_land_on_the_frame_time_base(self, tmp_path):
        clock, run = _paired(tmp_path)
        al = align_mod.align_clocks(clock, run)
        cols = align_mod.resample_to_frames(clock, run, al)
        assert cols["Ewe/V"].shape == (clock.n_frames,)
        assert cols["time/s"][0] == pytest.approx(run.time_s[0])
        # The synthetic Ewe is an exact linear ramp, so interpolation is exact.
        expected = 0.02 * (cols["time/s"] - run.time_s[0])
        assert cols["Ewe/V"] == pytest.approx(expected, abs=1e-6)

    def test_frames_outside_the_record_are_nan_not_clamped(self, tmp_path):
        """A movie that outlives the sweep must not show the last potential
        held flat — that is a measurement nobody made."""
        clock, run = _paired(tmp_path, n_ec=30)
        al = align_mod.align_clocks(clock, run, method="span")
        cols = align_mod.resample_to_frames(clock, run, al)
        ewe = cols["Ewe/V"]
        assert np.isfinite(ewe[0])
        assert np.isnan(ewe[-1])
        assert not al.trustworthy

    def test_discrete_channels_take_the_nearest_sample(self, tmp_path):
        """Interpolating a cycle number would invent cycle 1.5."""
        clock, run = _paired(tmp_path)
        al = align_mod.align_clocks(clock, run)
        cols = align_mod.resample_to_frames(clock, run, al)
        for name in ("cycle number", "I Range", "ox/red"):
            finite = cols[name][np.isfinite(cols[name])]
            assert set(np.unique(finite)) <= set(
                np.unique(np.asarray(run.channels[name], dtype=float))
            ), f"{name} was interpolated between samples"

    def test_requested_channel_subset(self, tmp_path):
        clock, run = _paired(tmp_path)
        al = align_mod.align_clocks(clock, run)
        cols = align_mod.resample_to_frames(clock, run, al, channels=["Ewe/V"])
        assert set(cols) == {"time/s", "Ewe/V"}

    def test_inverse_map_returns_frames_for_samples(self, tmp_path):
        clock, run = _paired(tmp_path)
        al = align_mod.align_clocks(clock, run)
        frames = align_mod.frame_for_ec_sample(clock, run, al)
        assert frames.shape == run.time_s.shape
        assert frames[0] == 0
        inside = frames >= 0
        # every mapped frame is within half a frame period of its sample
        t_frames = align_mod.ec_time_for_frames(clock, al)
        err = np.abs(t_frames[frames[inside]] - run.time_s[inside])
        assert err.max() <= clock.frame_period / 2 + 1e-9


class TestMatchRuns:
    def test_picks_the_run_whose_duration_matches(self, tmp_path):
        """Of several techniques in a session, the one that ran DURING the
        movie is the one with the movie's duration."""
        mrc = write_movie(tmp_path, n=200, epoch=1763387769)
        clock = de_movie.read_movie_clock(mrc)
        n_match = int(np.ceil(clock.duration / 0.08)) + 1
        for name, n_ec in (("a_02_CV_C01.mpr", 30),
                           ("b_02_CV_C01.mpr", n_match),
                           ("c_02_CV_C01.mpr", 12)):
            (tmp_path / name).write_bytes(
                build_mpr(ids=CV_IDS, records=cv_records(n=n_ec))
            )
        runs = eclab.find_ec_runs(str(tmp_path))
        ranked = align_mod.match_runs(clock, runs)
        assert ranked, "no run matched the movie"
        best, al = ranked[0]
        assert best.path.endswith("b_02_CV_C01.mpr")
        assert al.method == "span"
        assert al.trustworthy

    def test_no_candidates_returns_empty(self, tmp_path):
        clock, run = _paired(tmp_path, n_ec=30)
        assert align_mod.match_runs(clock, [run]) == []


class TestRealWorldShape:
    """Numbers from a real paired DE Apollo + BioLogic SP-200 acquisition.

    The point is the *relationship*: 7914 frames at 30.525 fps and 3256 samples
    at 12.5 Hz cover the same 259.2 s, while the two PCs' wall clocks disagree
    by very nearly an hour. Span matching gets the right answer without ever
    consulting those clocks.
    """

    def test_span_beats_a_one_hour_clock_skew(self, tmp_path):
        mrc = write_movie(tmp_path, n=7914, period=0.03276, epoch=1763387769)
        clock = de_movie.read_movie_clock(mrc)
        assert clock.duration == pytest.approx(259.23, abs=0.01)

        path = tmp_path / "floating_02_CV_C01.mpr"
        path.write_bytes(
            build_mpr(ids=CV_IDS, records=cv_records(n=3243, t0=31.56, dt_s=0.08))
        )
        run = eclab.read_ec_file(str(path))

        al = align_mod.align_clocks(clock, run)
        assert al.method == "span"
        assert abs(al.duration_mismatch_s) < 1.0
        assert al.covered_fraction == pytest.approx(1.0, abs=1e-3)
        # The camera's own epoch stamp puts the movie an hour off the EC clock;
        # span matching is immune to that, and reports the skew instead.
        assert round(al.implied_utc_offset_hours) == 1

        cols = align_mod.resample_to_frames(clock, run, al)
        assert np.isfinite(cols["Ewe/V"]).all()
        assert cols["Ewe/V"].shape == (7914,)
