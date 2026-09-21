"""
model.py — :class:`MultiAngleModel`, the small serialisable result of aligning a
multi-angle acquisition.

A multi-angle acquisition is N separate 4-D datasets ("members") recorded at
different tilt magnitudes and azimuths about one centre point. Members sharing a
tilt magnitude form a **shell**; an acquisition may have several shells and they
need not hold the same number of members, so nothing here assumes one global
tilt angle or an even split.

Aligning the members produces nothing but this object — two small integer offset
arrays and the metadata to interpret them. That is the design, not an economy:

* **Alignment is an index REMAP, never a resample and never a copy.** Both
  offsets are whole numbers of pixels, so a member is READ at shifted indices
  rather than interpolated into a new array. Sub-pixel alignment would mean
  resampling every diffraction pattern of every member, which is precisely what
  a loader for tens of gigabytes cannot afford to do.
* The sub-pixel remainder that rounding discarded is REPORTED
  (:attr:`nav_residuals`, :attr:`dp_residuals`) rather than silently dropped, so
  a caller can tell "aligned to the nearest pixel" from "happened to land on
  one".

Sign convention
---------------
``nav_offsets[i]`` and ``dp_offsets[i]`` are **corrections**: the ``(dy, dx)``
you ADD to member *i* to bring it onto the reference member. This is exactly the
convention of :mod:`spyde.drift.model`, which is the authority — read the sign
section of its module docstring before changing anything here. The offsets are
the output of :func:`spyde.drift.solve_translation` rounded to integers, so they
cannot mean anything else without the rounding step lying about its input.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np


#: Tilt magnitudes closer than this (degrees) are the same shell. Nominal tilts
#: come from the acquisition and are usually exact, but a value read back from an
#: instrument log carries rounding, so grouping on exact equality would split a
#: shell in two and silently halve every per-shell average.
DEFAULT_SHELL_TOLERANCE = 0.01


def assign_shells(tilts, *, tolerance: float = DEFAULT_SHELL_TOLERANCE) -> np.ndarray:
    """Group tilt magnitudes into shells, numbered from the smallest tilt up.

    Returns ``(N,)`` int64 shell ids. Members within *tolerance* degrees of the
    first member of a shell join it; the comparison is against that opening tilt
    and not against the previous member, so a long run of slightly increasing
    tilts cannot chain itself into one enormous shell.
    """
    values = np.asarray(tilts, dtype=np.float64).ravel()
    shell_ids = np.empty(values.size, dtype=np.int64)
    current = -1
    opening = 0.0
    for position in np.argsort(values, kind="stable"):
        tilt = float(values[position])
        if current < 0 or tilt - opening > float(tolerance):
            current += 1
            opening = tilt
        shell_ids[position] = current
    return shell_ids


def _as_integer_offsets(values, name: str, n_members: int) -> np.ndarray:
    """``(N, 2)`` int64, rejecting anything that is not already whole pixels.

    Rejects rather than rounds. A caller handing over 1.5 px either forgot to
    round or is trying to express an alignment this model cannot represent, and
    both deserve an exception rather than a quietly different answer.
    """
    array = np.asarray(values)
    if array.ndim != 2 or array.shape != (n_members, 2):
        raise ValueError(f"{name} must be ({n_members}, 2); got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")
        if not np.all(array == np.round(array)):
            raise ValueError(
                f"{name} must be whole pixels — alignment is an index remap, "
                "not a resample (see the module docstring)"
            )
    elif not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a numeric array; got dtype {array.dtype}")
    return np.ascontiguousarray(array, dtype=np.int64)


def _extent(shape, name: str, what: str) -> tuple[int, int]:
    """Validate a 2-D ``(height, width)`` extent."""
    values = tuple(int(v) for v in shape)
    if len(values) != 2:
        raise ValueError(
            f"{name} must be the 2-D {what} shape (height, width); got {shape!r}")
    return values


def _region_origin(offsets) -> tuple[int, int]:
    """Start of the region every member covers, in reference coordinates.

    A member covers reference coordinates ``[offset, offset + extent)``, so the
    region common to all of them starts at the LARGEST offset.
    """
    origin = offsets.max(axis=0)
    return int(origin[0]), int(origin[1])


def _region_shape(offsets, extent) -> tuple[int, int]:
    """Size of that common region: the extent less the spread of the offsets."""
    spread = offsets.max(axis=0) - offsets.min(axis=0)
    return (max(0, extent[0] - int(spread[0])),
            max(0, extent[1] - int(spread[1])))


def _region_slices(offsets, member_index: int, extent) -> tuple[slice, slice]:
    """The common region expressed in ONE member's own coordinates.

    Reference coordinate ``r`` is member coordinate ``r - offset``, which is
    what makes the start non-negative and the stop within the extent for every
    member: the origin is the largest offset, so no member is asked for a
    position it does not have.
    """
    height, width = _region_shape(offsets, extent)
    origin_y, origin_x = _region_origin(offsets)
    start_y = origin_y - int(offsets[member_index, 0])
    start_x = origin_x - int(offsets[member_index, 1])
    return (slice(start_y, start_y + height), slice(start_x, start_x + width))


@dataclass
class MultiAngleModel:
    """How the members of a multi-angle acquisition line up with one another.

    Parameters
    ----------
    paths
        One source path per member, in member order. Length fixes N.
    tilts
        ``(N,)`` tilt (precession) magnitude in degrees.
    azimuths
        ``(N,)`` azimuth about the centre, in degrees.
    shell_ids
        ``(N,)`` int — which shell each member belongs to. Build it with
        :func:`assign_shells` unless the acquisition already states it.
    nav_offsets
        ``(N, 2)`` int — the ``(dy, dx)`` in SCAN positions to add to a member to
        bring its real-space image onto the reference (see the module docstring
        on sign).
    dp_offsets
        ``(N, 2)`` int — the same, in DETECTOR pixels, bringing the direct beams
        together.
    reference
        Index of the member everything is aligned to. Its two offsets are
        ``(0, 0)`` by construction.
    nav_residuals, dp_residuals
        Optional ``(N, 2)`` float — the sub-pixel remainder rounding discarded.
        A residual near 0.5 px means the integer answer is the worse of two
        roughly equal choices, which is worth surfacing rather than hiding.
    provenance
        Free-form record of how the model was made.
    """

    paths: list[str]
    tilts: np.ndarray
    azimuths: np.ndarray
    shell_ids: np.ndarray
    nav_offsets: np.ndarray
    dp_offsets: np.ndarray
    reference: int = 0
    nav_residuals: np.ndarray | None = None
    dp_residuals: np.ndarray | None = None
    provenance: dict[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        self.paths = [str(p) for p in self.paths]
        n_members = len(self.paths)
        if n_members == 0:
            raise ValueError("a multi-angle model needs at least one member")

        for name in ("tilts", "azimuths"):
            array = np.ascontiguousarray(getattr(self, name), dtype=np.float64)
            if array.shape != (n_members,):
                raise ValueError(
                    f"{name} must be ({n_members},); got {array.shape}")
            setattr(self, name, array)

        self.shell_ids = np.ascontiguousarray(self.shell_ids, dtype=np.int64)
        if self.shell_ids.shape != (n_members,):
            raise ValueError(
                f"shell_ids must be ({n_members},); got {self.shell_ids.shape}")

        self.nav_offsets = _as_integer_offsets(
            self.nav_offsets, "nav_offsets", n_members)
        self.dp_offsets = _as_integer_offsets(
            self.dp_offsets, "dp_offsets", n_members)

        self.reference = int(self.reference)
        if not 0 <= self.reference < n_members:
            raise ValueError(
                f"reference {self.reference} outside 0..{n_members - 1}")

        for name in ("nav_residuals", "dp_residuals"):
            residuals = getattr(self, name)
            if residuals is None:
                continue
            residuals = np.ascontiguousarray(residuals, dtype=np.float32)
            if residuals.shape != (n_members, 2):
                raise ValueError(
                    f"{name} must be ({n_members}, 2); got {residuals.shape}")
            setattr(self, name, residuals)

    # ── basic properties ─────────────────────────────────────────────────────

    @property
    def n_members(self) -> int:
        return len(self.paths)

    @property
    def n_shells(self) -> int:
        return int(np.unique(self.shell_ids).size)

    @property
    def shells(self) -> dict[int, tuple[int, ...]]:
        """Member indices per shell id, shells in ascending id order.

        Member order within a shell is preserved, so a caller can pair a shell's
        members with their azimuths without sorting anything again.
        """
        grouped: dict[int, list[int]] = {}
        for index, shell in enumerate(self.shell_ids):
            grouped.setdefault(int(shell), []).append(index)
        return {shell: tuple(members)
                for shell, members in sorted(grouped.items())}

    @property
    def max_abs_nav_offset(self) -> int:
        """Largest single-axis scan-position correction. Sizes the lost border."""
        return int(np.max(np.abs(self.nav_offsets)))

    @property
    def max_abs_dp_offset(self) -> int:
        """Largest single-axis detector correction."""
        return int(np.max(np.abs(self.dp_offsets)))

    # ── the two remaps ───────────────────────────────────────
    #
    # Scan positions and detector pixels are remapped by exactly the same
    # arithmetic on different offsets, so the geometry lives in the three
    # module-level helpers and these methods only choose which offsets to hand
    # them. Writing it out twice is how the two drift apart.

    @property
    def overlap_origin(self) -> tuple[int, int]:
        """Where the common scan region starts, in the REFERENCE member's grid.

        Needed to calibrate the composed navigation axes: the composed scan does
        not start at the reference's own origin unless every offset is positive.
        """
        return _region_origin(self.nav_offsets)

    def overlap_shape(self, member_shape) -> tuple[int, int]:
        """Scan shape that survives the real-space remap, given a member's own.

        Every member covers the same extent but at a different offset, so the
        region present in ALL of them is smaller by the spread of the offsets on
        each axis. Returns ``(0, 0)`` components where the members do not overlap
        at all rather than a negative size.
        """
        return _region_shape(
            self.nav_offsets, _extent(member_shape, "member_shape", "scan"))

    def nav_slices(self, member_index: int, member_shape) -> tuple[slice, slice]:
        """The slice of member *member_index* that lands on the common region.

        This IS the alignment: reading each member through its own pair of
        slices puts them all on the same scan grid, with no data moved and no
        value changed.
        """
        self._check_member(member_index)
        return _region_slices(
            self.nav_offsets, int(member_index),
            _extent(member_shape, "member_shape", "scan"))

    @property
    def detector_origin(self) -> tuple[int, int]:
        """Where the common detector region starts, in the REFERENCE member's
        detector. Shifts the signal axes' offset, so the reciprocal-space origin
        still lands on the pixel the direct beam actually occupies."""
        return _region_origin(self.dp_offsets)

    def detector_shape(self, member_detector_shape) -> tuple[int, int]:
        """Detector shape that survives the reciprocal remap.

        The detector is CROPPED to the region every member covers, exactly as
        the scan is. Keeping the full extent and padding the uncovered edge
        instead would leave a border where fewer than N members contribute, and
        a summed pattern whose edge is dimmer than its middle by a step is worse
        than a slightly smaller one: it is the detector edge that vector finding
        and strain read, and a step there is indistinguishable from real
        structure. The offsets are a few pixels of beam centring, so the crop
        costs a thin border.
        """
        return _region_shape(
            self.dp_offsets,
            _extent(member_detector_shape, "member_detector_shape", "detector"))

    def detector_slices(self, member_index: int,
                        member_detector_shape) -> tuple[slice, slice]:
        """The slice of member *member_index*'s DETECTOR that lands on the common
        region — the reciprocal-space counterpart of :meth:`nav_slices`."""
        self._check_member(member_index)
        return _region_slices(
            self.dp_offsets, int(member_index),
            _extent(member_detector_shape, "member_detector_shape", "detector"))

    def _check_member(self, member_index: int) -> None:
        if not 0 <= int(member_index) < self.n_members:
            raise ValueError(
                f"member_index {member_index} outside 0..{self.n_members - 1}")

    # ── serialisation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"MultiAngleModel(n_members={self.n_members}, "
            f"n_shells={self.n_shells}, reference={self.reference}, "
            f"max_abs_nav_offset={self.max_abs_nav_offset} px, "
            f"max_abs_dp_offset={self.max_abs_dp_offset} px)"
        )


def model_from_metadata(signal):
    """The model a composed signal RECORDS, or None if it records none.

    A recipe is a runtime object: it is attached when an acquisition is
    composed and does not survive being written to a file. Everything the
    model holds is in the metadata precisely so a reopened dataset still
    knows what it is — without this, a saved acquisition re-expands into its
    nodes and then has no angles to draw a ring from.
    """
    import numpy as np

    from spyde.signals.multiangle import MULTIANGLE_METADATA

    metadata = getattr(signal, "metadata", None)
    if metadata is None or not metadata.has_item(MULTIANGLE_METADATA):
        return None
    read = lambda key, default=None: metadata.get_item(
        f"{MULTIANGLE_METADATA}.{key}", default)
    try:
        members = int(read("n_members"))
    except (TypeError, ValueError):
        return None
    if members < 1:
        return None
    zeros = [[0, 0]] * members
    try:
        return MultiAngleModel(
            paths=[str(path) for path in
                   read("paths", [f"member{i}" for i in range(members)])],
            tilts=np.asarray([float(v) for v in
                              read("tilts", [0.0] * members)]),
            azimuths=np.asarray([float(v) for v in
                                 read("azimuths", [0.0] * members)]),
            shell_ids=np.asarray([int(v) for v in
                                  read("shell_ids", [0] * members)],
                                 dtype=np.int64),
            nav_offsets=np.asarray(read("nav_offsets", zeros), dtype=np.int64),
            dp_offsets=np.asarray(read("dp_offsets", zeros), dtype=np.int64),
            reference=int(read("reference", 0)),
        )
    except Exception:
        # A file with half a record is a file to open plainly, not to fail on.
        return None
