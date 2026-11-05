"""BioLogic EC-Lab potentiostat records — ``.mpr``, ``.mpt``/``.txt``, ``.mps``.

An EC-Lab experiment is one *settings* file (``.mps``, the recipe) plus one
*record* per linked technique, written twice over: a binary ``.mpr`` and,
whenever the operator remembers to export it, an ASCII ``.mpt`` (often saved as
``.txt``). Both carry the same samples; only the ``.mpr`` is guaranteed to
exist, so it is the primary path here and the ASCII reader is the cross-check.

Time in an EC-Lab record is ``time/s``, seconds since the ACQUISITION started —
which is *before* the technique started, because a linked sequence keeps one
clock across all its techniques. A second technique therefore begins at a
``time/s`` of tens or hundreds of seconds, not at zero. The absolute origin for
that clock lives in the ``VMP LOG`` module as an OLE date (and in the ASCII
header as ``Acquisition started on``); it is the EC-Lab PC's **naive local wall
clock**, with no timezone recorded anywhere in the file.

Format notes (reverse-engineered; cross-checked against an ASCII export of the
same run, see ``spyde/tests/migrated/test_eclab.py``):

* A module header is ``b"MODULE"`` + 10-byte short name + 25-byte long name,
  then either a 32-bit length, or — when that field reads ``0xFFFFFFFF`` — a
  64-bit length following it. Both layouts appear in the wild.
* The ``VMP data`` body is ``npts:u4, ncols:u2, ids:u2[ncols]`` and then a
  fixed block of padding before the records. Rather than hard-code that padding
  (it is version-dependent), the record start is DERIVED as
  ``len(body) - npts * record_size``, which self-checks the column layout: if
  the assumed widths were wrong the subtraction would not land on a plausible
  offset over a run of zero bytes.
* Several column IDs are *flags* packed into a single leading ``u1`` rather than
  columns of their own.

Coverage: the column table below holds the IDs seen in real CV/OCV records. An
unrecognised ID is not guessed at — a wrong width silently shifts every later
column and corrupts the whole record — except in the one case where exactly one
ID is unknown and the byte arithmetic pins its width unambiguously. Anything
else raises :class:`UnsupportedColumns`, which names the IDs and points at the
ASCII export as the way through.
"""
from __future__ import annotations

import datetime as dt
import glob
import logging
import os
import re
import struct
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

MPR_MAGIC = b"BIO-LOGIC MODULAR FILE\x1a"
MPT_MAGIC = "EC-Lab ASCII FILE"

_OLE_EPOCH = dt.datetime(1899, 12, 30)
# Candidate byte offsets of the acquisition-start OLE date inside VMP LOG.
# Which one is populated varies between EC-Lab versions; the first that decodes
# to a sane calendar date wins.
_LOG_DATE_OFFSETS = (585, 465, 469, 473, 993)
_OLE_MIN, _OLE_MAX = 36525.0, 73050.0  # 2000-01-01 .. 2100-01-01

# Column IDs that are BIT FLAGS sharing one leading u1 byte, not columns.
_FLAG_IDS: dict[int, tuple[str, int]] = {
    1: ("mode", 0x03),
    2: ("ox/red", 0x04),
    3: ("error", 0x08),
    21: ("control changes", 0x10),
    31: ("Ns changes", 0x20),
    65: ("counter inc.", 0x80),
}

# Column ID -> (name, numpy dtype). Verified against an ASCII export.
_COLUMN_IDS: dict[int, tuple[str, str]] = {
    4: ("time/s", "<f8"),
    5: ("control/V/mA", "<f4"),
    6: ("Ewe/V", "<f4"),
    11: ("<I>/mA", "<f8"),
    19: ("control/V", "<f4"),
    24: ("cycle number", "<f8"),
    39: ("I Range", "<u2"),
    174: ("<Ewe>/V", "<f4"),
    434: ("(Q-Qo)/C", "<f4"),
    438: ("step time/s", "<f8"),
}

_WIDTH_TO_DTYPE = {1: "<u1", 2: "<u2", 4: "<f4", 8: "<f8"}

# Channels that step rather than vary continuously — interpolating them would
# invent values that never occurred (cycle 1.5, an I Range between two ranges).
DISCRETE_CHANNELS = frozenset({
    "cycle number", "I Range", "Ns", "mode", "ox/red", "error",
    "control changes", "Ns changes", "counter inc.", "half cycle",
})

_TECHNIQUES = (
    "Cyclic Voltammetry", "Open Circuit Voltage", "Chronoamperometry",
    "Chronopotentiometry", "Galvanostatic Cycling with Potential Limitation",
    "Potentio Electrochemical Impedance Spectroscopy",
    "Galvano Electrochemical Impedance Spectroscopy",
    "Linear Sweep Voltammetry", "Loop", "Wait", "Modulo Bat",
    "Constant Current", "Constant Voltage",
)

# EC-Lab's filename technique codes (``…_02_CV_C01.mpr``). Expanded to the same
# full names the ASCII header uses, so a run read from the binary and the same
# run read from its export report the same technique — which is what lets
# :func:`find_ec_runs` recognise them as one run.
_TECHNIQUE_CODES = {
    "CV": "Cyclic Voltammetry",
    "LSV": "Linear Sweep Voltammetry",
    "OCV": "Open Circuit Voltage",
    "CA": "Chronoamperometry",
    "CP": "Chronopotentiometry",
    "GCPL": "Galvanostatic Cycling with Potential Limitation",
    "PEIS": "Potentio Electrochemical Impedance Spectroscopy",
    "GEIS": "Galvano Electrochemical Impedance Spectroscopy",
    "MB": "Modulo Bat",
    "CC": "Constant Current",
    "CV_": "Constant Voltage",
    "WAIT": "Wait",
    "LOOP": "Loop",
}

# The data module carries a lone marker byte at this offset inside the padding
# that precedes the records; EC-Lab sets it in some versions and not others.
# It is not a column, so it must not trip the padding integrity check.
_PADDING_MARKER = 1006


class UnsupportedColumns(ValueError):
    """A ``.mpr`` uses column IDs whose byte widths are unknown."""


@dataclass
class EcRun:
    """One EC-Lab technique record.

    ``time_s`` is seconds since ``start`` (the ACQUISITION start, shared across
    the linked techniques of one sequence — so it need not begin near zero).
    ``start`` is naive local wall clock as recorded by the EC-Lab PC; see
    :meth:`start_utc` for turning it into an absolute instant.
    """

    path: str
    technique: str
    start: dt.datetime | None
    time_s: np.ndarray
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    settings: dict[str, str] = field(default_factory=dict)
    unknown_columns: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.time_s = np.asarray(self.time_s, dtype=float)

    @property
    def n_points(self) -> int:
        return int(self.time_s.size)

    @property
    def duration(self) -> float:
        """First-to-last sample span in seconds."""
        if self.time_s.size < 2:
            return 0.0
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def sample_period(self) -> float:
        """Median sampling period in seconds."""
        if self.time_s.size < 2:
            return 0.0
        return float(np.median(np.diff(self.time_s)))

    @property
    def name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    def start_utc(self, utc_offset_hours: float) -> dt.datetime | None:
        """:attr:`start` reinterpreted as being *utc_offset_hours* ahead of UTC.

        The file records no timezone, so the offset has to come from the caller
        (or from :func:`spyde.insitu.align.align`'s span match, which sidesteps
        the question entirely).
        """
        if self.start is None:
            return None
        return (self.start - dt.timedelta(hours=utc_offset_hours)).replace(
            tzinfo=dt.timezone.utc
        )

    def potential(self) -> np.ndarray | None:
        """The working-electrode potential, whichever column carries it."""
        for key in ("Ewe/V", "<Ewe>/V", "Ewe-Ece/V", "|E|/V"):
            if key in self.channels:
                return self.channels[key]
        return None

    def current(self) -> np.ndarray | None:
        """The current in mA, whichever column carries it."""
        for key in ("<I>/mA", "I/mA", "|I|/mA"):
            if key in self.channels:
                return self.channels[key]
        return None

    def describe(self) -> str:
        when = self.start.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if self.start else "?"
        return (
            f"{self.name}: {self.technique}, {self.n_points} pts, "
            f"{self.time_s[0]:.2f}–{self.time_s[-1]:.2f} s "
            f"({self.duration:.2f} s @ {self.sample_period * 1e3:.1f} ms), "
            f"acq start {when}"
        ) if self.n_points else f"{self.name}: {self.technique}, empty"


# --------------------------------------------------------------------------
# .mpr (binary)
# --------------------------------------------------------------------------

@dataclass
class _Module:
    offset: int
    short: str
    long: str
    version: int
    date: str
    body: bytes


def _read_modules(buf: bytes) -> list[_Module]:
    mods: list[_Module] = []
    i = buf.find(b"MODULE")
    while i >= 0:
        short = buf[i + 6 : i + 16].decode("latin1").strip()
        long = buf[i + 16 : i + 41].decode("latin1").strip()
        (first,) = struct.unpack_from("<I", buf, i + 41)
        if first == 0xFFFFFFFF:  # 64-bit length follows the sentinel
            (length,) = struct.unpack_from("<Q", buf, i + 45)
            (version,) = struct.unpack_from("<I", buf, i + 53)
            date = buf[i + 57 : i + 65].decode("latin1")
            start = i + 65
        else:
            length = first
            (version,) = struct.unpack_from("<I", buf, i + 45)
            date = buf[i + 49 : i + 57].decode("latin1")
            start = i + 57
        body = buf[start : start + length]
        if len(body) != length:
            log.warning("eclab: module %r truncated (%d of %d bytes)",
                        short, len(body), length)
        mods.append(_Module(i, short, long, version, date, body))
        i = buf.find(b"MODULE", start + len(body))
    return mods


def _decode_ole_date(value: float) -> dt.datetime | None:
    if not (_OLE_MIN < value < _OLE_MAX):
        return None
    try:
        return _OLE_EPOCH + dt.timedelta(days=float(value))
    except (OverflowError, ValueError):
        return None


def _log_start_date(mod: _Module | None) -> dt.datetime | None:
    if mod is None:
        return None
    for off in _LOG_DATE_OFFSETS:
        if off + 8 > len(mod.body):
            continue
        (raw,) = struct.unpack_from("<d", mod.body, off)
        when = _decode_ole_date(raw)
        if when is not None:
            return when
    return None


def _technique_from(settings: _Module | None, path: str) -> str:
    """Technique name from the settings module's embedded strings, else the
    filename's EC-Lab suffix (``…_02_CV_C01.mpr`` → ``CV``)."""
    if settings is not None:
        text = settings.body.decode("latin1", errors="replace")
        for name in _TECHNIQUES:
            if name in text:
                return name
    m = re.search(r"_\d+_([A-Za-z]+)_C\d+", os.path.basename(path))
    if not m:
        return "unknown"
    code = m.group(1).upper()
    return _TECHNIQUE_CODES.get(code, code)


def _layout(ids: tuple[int, ...]) -> tuple[list[tuple[str, str]], dict[str, int], list[int]]:
    """``ids`` → (numpy dtype fields, flag masks, unknown ids).

    Flag IDs collapse into a single leading ``flags`` field; everything else
    becomes one field, in the order the IDs appear.
    """
    fields: list[tuple[str, str]] = []
    flags: dict[str, int] = {}
    unknown: list[int] = []
    for cid in ids:
        if cid in _FLAG_IDS:
            name, mask = _FLAG_IDS[cid]
            if not flags:
                fields.append(("flags", "<u1"))
            flags[name] = mask
            continue
        if cid in _COLUMN_IDS:
            name, dtype = _COLUMN_IDS[cid]
            fields.append((name, dtype))
        else:
            unknown.append(cid)
            fields.append((f"unknown_{cid}", ""))  # width filled in later
    return fields, flags, unknown


def _padding_is_clean(body: bytes, first: int, last: int) -> bool:
    """True when ``body[first:last]`` is the all-zero run that precedes the
    records (bar the lone marker byte EC-Lab sets in some versions)."""
    padding = bytearray(body[first:last])
    marker = _PADDING_MARKER - first
    if 0 <= marker < len(padding):
        padding[marker] = 0
    return not any(padding)


def _resolve_unknown(fields: list[tuple[str, str]], body: bytes, npts: int,
                     header_end: int, unknown: list[int],
                     path: str) -> list[tuple[str, str]]:
    """Pin the width of a single unknown column from the record size.

    The records sit flush against the END of the data module behind a run of
    zero padding, so a candidate width is right only if it puts the first
    record where the padding stops. That is the whole test: try each plausible
    width and keep the one whose implied start lands on clean padding.

    With two or more unknowns there is nothing to solve — several width
    combinations would satisfy the same total — and a wrong width silently
    shifts every column after it, so we refuse rather than guess.
    """
    if not unknown:
        return fields
    if len(unknown) > 1:
        raise UnsupportedColumns(
            f"{os.path.basename(path)}: unknown EC-Lab column IDs {unknown} — "
            "cannot determine their byte widths. Export the run as ASCII from "
            "EC-Lab (File > Export as text) and load the .mpt/.txt instead."
        )
    known = sum(np.dtype(d).itemsize for _, d in fields if d)
    for width, dtype in _WIDTH_TO_DTYPE.items():
        start = len(body) - npts * (known + width)
        if start >= header_end and _padding_is_clean(body, header_end, start):
            log.warning(
                "eclab: %s has unrecognised column ID %d; reading it as a "
                "%d-byte 'unknown_%d' pinned by the record size",
                os.path.basename(path), unknown[0], width, unknown[0],
            )
            return [(n, d or dtype) for n, d in fields]
    raise UnsupportedColumns(
        f"{os.path.basename(path)}: unknown EC-Lab column ID {unknown[0]} — no "
        f"byte width places {npts} records inside a {len(body)} B data module. "
        "Export the run as ASCII from EC-Lab and load the .mpt/.txt instead."
    )


def read_mpr(path: str) -> EcRun:
    """Read a BioLogic ``.mpr`` binary record."""
    with open(path, "rb") as fh:
        buf = fh.read()
    if not buf.startswith(MPR_MAGIC):
        raise ValueError(f"{path}: not a BioLogic .mpr file")

    mods = _read_modules(buf)
    by_short = {m.short: m for m in mods}
    data_mod = next((m for m in mods if m.short.startswith("VMP data")), None)
    if data_mod is None:
        raise ValueError(f"{path}: no 'VMP data' module")

    body = data_mod.body
    (npts,) = struct.unpack_from("<I", body, 0)
    (ncols,) = struct.unpack_from("<H", body, 4)
    ids = struct.unpack_from(f"<{ncols}H", body, 6)

    if npts <= 0:
        raise ValueError(f"{path}: data module declares {npts} points")
    header_end = 6 + 2 * ncols

    fields, flag_masks, unknown = _layout(ids)
    fields = _resolve_unknown(fields, body, npts, header_end, unknown, path)
    dtype = np.dtype([(n, d) for n, d in fields])
    record_size = dtype.itemsize

    data_start = len(body) - npts * record_size
    if data_start < header_end:
        raise ValueError(
            f"{path}: {npts} records of {record_size} B do not fit in a "
            f"{len(body)} B data module — column layout is wrong"
        )
    if not _padding_is_clean(body, header_end, data_start):
        log.warning("eclab: %s has non-zero padding before the records; the "
                    "column layout may be misread", os.path.basename(path))

    records = np.frombuffer(body, dtype=dtype, count=npts, offset=data_start)

    channels: dict[str, np.ndarray] = {}
    for name in dtype.names:
        if name == "flags":
            continue
        channels[name] = np.asarray(records[name])
    if "flags" in dtype.names:
        raw = np.asarray(records["flags"])
        for name, mask in flag_masks.items():
            value = raw & mask
            # 'mode' is a 2-bit field; the rest are single bits read as bool.
            channels[name] = value if mask == 0x03 else value.astype(bool)

    time_s = channels.pop("time/s", np.arange(npts, dtype=float))
    return EcRun(
        path=path,
        technique=_technique_from(by_short.get("VMP Set"), path),
        start=_log_start_date(by_short.get("VMP LOG")),
        time_s=time_s,
        channels=channels,
        settings=_mpr_settings_text(by_short.get("VMP Set")),
        unknown_columns=tuple(unknown),
    )


def _mpr_settings_text(mod: _Module | None) -> dict[str, str]:
    """Best-effort ``key : value`` pairs from the settings module's ASCII."""
    if mod is None:
        return {}
    text = mod.body.decode("latin1", errors="replace")
    out: dict[str, str] = {}
    for line in re.split(r"[\r\n\x00]+", text):
        key, sep, value = line.partition(" : ")
        if sep and 0 < len(key.strip()) < 60 and key.strip().isprintable():
            out[key.strip()] = value.strip()
    return out


# --------------------------------------------------------------------------
# .mpt / .txt (ASCII export)
# --------------------------------------------------------------------------

_ACQ_START_RE = re.compile(
    r"Acquisition started on\s*:\s*(\d{1,2}/\d{1,2}/\d{4}\s+[\d:.]+)"
)
_ABS_TIME_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}\s")


def _parse_ec_datetime(text: str) -> dt.datetime | None:
    for fmt in ("%m/%d/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S",
                "%d/%m/%Y %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S"):
        try:
            return dt.datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    return None


def _to_float(token: str) -> float:
    """EC-Lab writes in the exporting PC's locale — decimal comma is common."""
    token = token.strip()
    if not token:
        return np.nan
    try:
        return float(token)
    except ValueError:
        try:
            return float(token.replace(",", "."))
        except ValueError:
            return np.nan


def read_mpt(path: str) -> EcRun:
    """Read an EC-Lab ASCII export (``.mpt``, often saved as ``.txt``).

    Handles both time conventions EC-Lab can export: ``time/s`` as elapsed
    seconds, or as an absolute ``MM/DD/YYYY HH:MM:SS.ffff`` stamp (which is
    converted back to elapsed seconds so both readers agree).
    """
    with open(path, encoding="latin1") as fh:
        lines = fh.read().splitlines()
    if not lines or MPT_MAGIC not in lines[0]:
        raise ValueError(f"{path}: not an EC-Lab ASCII file")

    n_header = 3
    for line in lines[:10]:
        m = re.search(r"Nb header lines\s*:\s*(\d+)", line)
        if m:
            n_header = int(m.group(1))
            break
    header = lines[: n_header - 1]
    technique = next(
        (t for line in header for t in _TECHNIQUES if line.strip() == t), "unknown"
    )
    start = None
    for line in header:
        m = _ACQ_START_RE.search(line)
        if m:
            start = _parse_ec_datetime(m.group(1))
            break

    columns = [c.strip() for c in lines[n_header - 1].split("\t") if c.strip()]
    rows = [line.split("\t") for line in lines[n_header:] if line.strip()]
    if not rows:
        raise ValueError(f"{path}: no data rows after {n_header} header lines")

    time_idx = next((i for i, c in enumerate(columns) if c.startswith("time/s")), None)
    absolute_time = (
        time_idx is not None
        and len(rows[0]) > time_idx
        and bool(_ABS_TIME_RE.match(rows[0][time_idx].strip()))
    )

    values: dict[str, list[float]] = {c: [] for c in columns}
    stamps: list[dt.datetime] = []
    for row in rows:
        for i, name in enumerate(columns):
            token = row[i] if i < len(row) else ""
            if i == time_idx and absolute_time:
                when = _parse_ec_datetime(token)
                stamps.append(when or dt.datetime.min)
                values[name].append(np.nan)
            else:
                values[name].append(_to_float(token))

    channels = {c: np.asarray(v, dtype=float) for c, v in values.items()}
    if absolute_time:
        origin = start or (stamps[0] if stamps else None)
        time_s = np.asarray([(s - origin).total_seconds() for s in stamps], dtype=float)
        channels.pop(columns[time_idx], None)
        if start is None:
            start = origin
    else:
        time_s = channels.pop(columns[time_idx]) if time_idx is not None else np.arange(
            len(rows), dtype=float
        )

    settings = {}
    for line in header:
        key, sep, value = line.partition(" : ")
        if sep:
            settings[key.strip()] = value.strip()

    return EcRun(path=path, technique=technique, start=start, time_s=time_s,
                 channels=channels, settings=settings)


# --------------------------------------------------------------------------
# .mps (settings) + discovery
# --------------------------------------------------------------------------

def read_mps(path: str) -> dict:
    """Parse an EC-Lab ``.mps`` settings file into ``{header, techniques}``.

    The ``.mps`` holds no samples — it is the recipe. Useful for knowing which
    techniques a sequence was *meant* to run, and in what order.
    """
    with open(path, encoding="latin1") as fh:
        lines = fh.read().splitlines()
    header: dict[str, str] = {}
    techniques: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in lines:
        stripped = line.strip()
        m = re.match(r"Technique\s*:\s*(\d+)", stripped)
        if m:
            current = {"index": m.group(1), "name": ""}
            techniques.append(current)
            continue
        if current is not None and not current["name"] and stripped:
            current["name"] = stripped
            continue
        key, sep, value = stripped.partition(" : ")
        if sep:
            (current if current is not None else header)[key.strip()] = value.strip()
    return {"path": path, "header": header, "techniques": techniques}


def read_ec_file(path: str) -> EcRun:
    """Read any EC-Lab record, dispatching on content rather than extension
    (the ASCII export is routinely saved as ``.txt``)."""
    with open(path, "rb") as fh:
        head = fh.read(len(MPR_MAGIC))
    if head.startswith(MPR_MAGIC):
        return read_mpr(path)
    return read_mpt(path)


def find_ec_runs(directory: str) -> list[EcRun]:
    """Read every EC-Lab record in *directory*, oldest acquisition first.

    Prefers the ``.mpr`` when a run has both a binary and an ASCII export, so
    the same technique is not returned twice. Identity is the run's *sample
    window* — acquisition start plus point count plus first/last timestamp —
    rather than the filename, because EC-Lab exports pick up suffixes
    (``-et``, ``(2)``) and the same run can be exported more than once.
    Unreadable files are logged and skipped rather than failing the whole scan.
    """
    seen: dict[tuple, EcRun] = {}
    candidates = sorted(glob.glob(os.path.join(glob.escape(directory), "*")))
    binaries = [p for p in candidates if p.lower().endswith(".mpr")]
    ascii_files = [p for p in candidates if p.lower().endswith((".mpt", ".txt"))]
    for path in binaries + ascii_files:
        try:
            run = read_ec_file(path)
        except (ValueError, OSError, struct.error) as exc:
            log.debug("eclab: skipping %s (%s)", os.path.basename(path), exc)
            continue
        key = (
            run.start,
            run.n_points,
            round(float(run.time_s[0]), 3) if run.n_points else 0.0,
            round(float(run.time_s[-1]), 3) if run.n_points else 0.0,
        )
        seen.setdefault(key, run)
    runs = list(seen.values())
    runs.sort(key=lambda r: (r.start or dt.datetime.min, r.time_s[0] if r.n_points else 0))
    return runs
