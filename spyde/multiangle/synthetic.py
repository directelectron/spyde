"""
synthetic.py — multi-angle members with KNOWN planted offsets.

Alignment is the kind of thing that looks right whichever way round you get the
sign, so the fixture plants the answer instead of leaving it to be eyeballed:
every member is built from one shared scene and one shared diffraction pattern,
displaced by an offset the generator returns. A solver either reproduces those
numbers or it does not.

Two properties of the construction are load-bearing for that to mean anything.

**The planted offset is the CORRECTION**, in the sense
:mod:`spyde.multiangle.model` defines: member *i*'s real-space window is cropped
from the scene at an origin displaced by exactly ``nav_offsets[i]``, and its
direct beam is drawn at exactly ``-dp_offsets[i]`` from the detector centre, so
ADDING the offset is what restores the reference. Invert the sign anywhere and
the recovered offsets come back negated, loudly.

**The real-space scene has structure and the two offset sets are not
proportional.** Phase correlation cannot register featureless noise, so the
scene is irregular blobs of mixed width; and the detector offsets are derived
from the azimuth with a different phase and handedness from the scan offsets, so
a solver that swapped the two spaces — or swapped ``dy`` for ``dx`` — cannot
accidentally pass.

Sizes are deliberately tiny (a few tens of MB for the default ten members) so a
test suite runs in seconds. Scale with ``scan_shape`` / ``detector_shape``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spyde.multiangle.model import DEFAULT_SHELL_TOLERANCE, assign_shells

#: Azimuth stagger between shells, degrees. Irregular on purpose: concentric
#: shells whose members share azimuths would put several members at the same
#: offset direction, and a direction-dependent bug could then hide.
_SHELL_STAGGER = 17.0

#: Extra scene border beyond the largest scan offset, so no member's crop runs
#: off the edge.
_SCENE_MARGIN = 2


@dataclass
class SyntheticMultiAngle:
    """Generated members plus the ground truth they were built from."""

    members: list[np.ndarray]
    tilts: np.ndarray
    azimuths: np.ndarray
    shell_ids: np.ndarray
    nav_offsets: np.ndarray
    dp_offsets: np.ndarray
    reference: int
    images: list[np.ndarray]
    patterns: list[np.ndarray]

    @property
    def n_members(self) -> int:
        return len(self.members)

    @property
    def scan_shape(self) -> tuple[int, int]:
        return self.members[0].shape[0], self.members[0].shape[1]

    @property
    def detector_shape(self) -> tuple[int, int]:
        return self.members[0].shape[2], self.members[0].shape[3]

    @property
    def paths(self) -> list[str]:
        """Stand-in source paths, so a model can be built from this fixture."""
        return [f"synthetic://member-{i:02d}" for i in range(self.n_members)]

    def navigators(self) -> np.ndarray:
        """``(N, h, w)`` — each member's real-space image, as a loader makes it."""
        return np.stack([member.sum(axis=(-2, -1), dtype=np.float64)
                         for member in self.members])

    def mean_patterns(self) -> np.ndarray:
        """``(N, ky, kx)`` — each member's mean diffraction pattern."""
        return np.stack([member.mean(axis=(0, 1), dtype=np.float64)
                         for member in self.members])


def _scene(shape: tuple[int, int], seed: int) -> np.ndarray:
    """An irregular, non-periodic, strictly positive scene.

    Irregular so a sign error or an axis swap lands on visibly different
    content; non-periodic so there is no second lattice translation for the
    correlation to lock onto; and it mixes wide blobs with sharp ones, because a
    scene of only wide blobs changes so little over a few pixels that a wrong
    answer still looks close.
    """
    height, width = shape
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float64)
    image = np.full((height, width), 0.12)
    for row_fraction, column_fraction, amplitude, sigma in (
        (0.27, 0.21, 1.00, 6.0),
        (0.63, 0.46, 0.70, 9.0),
        (0.39, 0.74, 0.85, 3.0),
        (0.81, 0.67, 0.55, 5.5),
        (0.16, 0.59, 0.65, 2.0),
        (0.52, 0.13, 0.45, 2.5),
        (0.72, 0.88, 0.60, 1.8),
    ):
        centre_row = row_fraction * height
        centre_column = column_fraction * width
        image += amplitude * np.exp(
            -((rows - centre_row) ** 2 + (columns - centre_column) ** 2)
            / (2.0 * sigma ** 2)
        )
    rng = np.random.default_rng(seed)
    image += 0.02 * rng.standard_normal((height, width))
    return np.clip(image, 0.02, None) / image.max()


def _diffraction_pattern(shape: tuple[int, int], offset,
                         beam_radius: float) -> np.ndarray:
    """A direct-beam disc plus a few Bragg spots, displaced by ``-offset``.

    Drawn at its displaced position rather than rolled, so nothing wraps around
    the detector edge and the pattern's total intensity is the same for every
    member.
    """
    height, width = shape
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float64)
    centre_row = (height - 1) / 2.0 - float(offset[0])
    centre_column = (width - 1) / 2.0 - float(offset[1])

    radius = np.hypot(rows - centre_row, columns - centre_column)
    # Soft edge: a hard threshold aliases, which puts a sub-pixel bias into the
    # very quantity the reciprocal solve is measuring.
    pattern = 0.5 * (1.0 - np.tanh((radius - beam_radius) / 0.8))

    spacing = max(4.0, 0.22 * min(height, width))
    for row_order, column_order, amplitude in (
        (-1, 0, 0.45), (1, 0, 0.30), (0, -1, 0.38), (0, 1, 0.22),
        (-1, -1, 0.18), (1, 1, 0.12),
    ):
        spot_row = centre_row + row_order * spacing
        spot_column = centre_column + column_order * spacing
        pattern += amplitude * np.exp(
            -((rows - spot_row) ** 2 + (columns - spot_column) ** 2)
            / (2.0 * 1.2 ** 2)
        )
    return pattern


def _planted_offsets(tilts: np.ndarray, azimuths: np.ndarray,
                     radius: float, azimuth_phase: float,
                     handedness: float) -> np.ndarray:
    """Offsets that grow with tilt and point along the azimuth.

    A member sits further from the centre the further it is tilted, which is the
    physical reason the members need aligning at all — so the planted answer is
    that shape rather than an arbitrary set of numbers.
    """
    largest = float(np.max(tilts)) if tilts.size else 0.0
    fraction = tilts / largest if largest > 0 else np.zeros_like(tilts)
    angle = np.radians(azimuths + azimuth_phase)
    offsets = np.stack([
        radius * fraction * np.cos(angle),
        handedness * radius * fraction * np.sin(angle),
    ], axis=1)
    return np.rint(offsets).astype(np.int64)


def make_multiangle(
    *,
    shells=((0.0, 1), (1.0, 6)),
    scan_shape: tuple[int, int] = (32, 36),
    detector_shape: tuple[int, int] = (32, 32),
    reference: int = 0,
    nav_offsets=None,
    dp_offsets=None,
    max_nav_offset: int = 3,
    max_dp_offset: int = 2,
    azimuths=None,
    noise: float = 0.01,
    counts: float = 4000.0,
    beam_radius: float = 2.5,
    seed: int = 0,
    dtype=np.uint16,
    shell_tolerance: float = DEFAULT_SHELL_TOLERANCE,
) -> SyntheticMultiAngle:
    """Build N synthetic 4-D members with known offsets.

    Parameters
    ----------
    shells
        ``((tilt_degrees, member_count), ...)``. Tilts need not be evenly spaced
        and shells need not hold the same number of members — the two things a
        multi-angle acquisition is not allowed to assume.
    scan_shape, detector_shape
        Per-member scan grid and detector. Keep both small; every member is a
        dense ``scan × detector`` array.
    reference
        The member offsets are expressed relative to. It gets ``(0, 0)``, which
        is what a solver returns for it.
    nav_offsets, dp_offsets
        Plant explicit ``(N, 2)`` integer offsets instead of deriving them from
        tilt and azimuth. Still taken relative to the reference member.
    max_nav_offset, max_dp_offset
        Offset magnitude at the largest tilt, in scan positions / detector
        pixels.
    azimuths
        Explicit ``(N,)`` degrees, for testing uneven azimuthal spacing.
    noise
        Relative per-detector-pixel noise. ``0`` gives a deterministic fixture.

    Returns
    -------
    SyntheticMultiAngle
        ``members`` are ``(scan_y, scan_x, ky, kx)`` arrays; ``nav_offsets`` and
        ``dp_offsets`` are the planted truth, in the correction sense of
        :mod:`spyde.multiangle.model`.
    """
    tilt_values: list[float] = []
    azimuth_values: list[float] = []
    for shell_index, (tilt, count) in enumerate(shells):
        count = int(count)
        if count < 1:
            raise ValueError(f"shell {shell_index} has {count} members")
        for member in range(count):
            tilt_values.append(float(tilt))
            azimuth_values.append(
                (360.0 * member / count + shell_index * _SHELL_STAGGER) % 360.0)

    tilts = np.asarray(tilt_values, dtype=np.float64)
    if azimuths is None:
        azimuths_array = np.asarray(azimuth_values, dtype=np.float64)
    else:
        azimuths_array = np.asarray(azimuths, dtype=np.float64).ravel()
        if azimuths_array.shape != tilts.shape:
            raise ValueError(
                f"azimuths must be ({tilts.size},); got {azimuths_array.shape}")
    n_members = tilts.size
    if not 0 <= int(reference) < n_members:
        raise ValueError(f"reference {reference} outside 0..{n_members - 1}")
    reference = int(reference)

    if nav_offsets is None:
        nav = _planted_offsets(tilts, azimuths_array, float(max_nav_offset),
                               0.0, 1.0)
    else:
        nav = np.asarray(nav_offsets, dtype=np.int64).reshape(n_members, 2)
    if dp_offsets is None:
        # A different phase and the opposite handedness, so the two offset sets
        # are never proportional to one another.
        dp = _planted_offsets(tilts, azimuths_array, float(max_dp_offset),
                              37.0, -1.0)
    else:
        dp = np.asarray(dp_offsets, dtype=np.int64).reshape(n_members, 2)
    nav = nav - nav[reference]
    dp = dp - dp[reference]

    shell_ids = assign_shells(tilts, tolerance=shell_tolerance)

    scan_height, scan_width = (int(v) for v in scan_shape)
    margin = int(np.max(np.abs(nav))) + _SCENE_MARGIN
    scene = _scene((scan_height + 2 * margin, scan_width + 2 * margin), seed)

    rng = np.random.default_rng(seed + 1)
    members: list[np.ndarray] = []
    images: list[np.ndarray] = []
    patterns: list[np.ndarray] = []
    for index in range(n_members):
        # Crop origin displaced by exactly the planted correction — see the
        # module docstring.
        start_row = margin + int(nav[index, 0])
        start_column = margin + int(nav[index, 1])
        image = scene[start_row:start_row + scan_height,
                      start_column:start_column + scan_width]
        pattern = _diffraction_pattern(
            (int(detector_shape[0]), int(detector_shape[1])),
            dp[index], beam_radius)

        member = (counts * image.astype(np.float32)[:, :, None, None]
                  * pattern.astype(np.float32)[None, None, :, :])
        if noise:
            member *= 1.0 + float(noise) * rng.standard_normal(
                member.shape, dtype=np.float32)
        if np.issubdtype(np.dtype(dtype), np.integer):
            limit = np.iinfo(np.dtype(dtype)).max
            member = np.clip(np.rint(member), 0, limit)
        members.append(member.astype(dtype))
        images.append(np.ascontiguousarray(image))
        patterns.append(pattern)

    return SyntheticMultiAngle(
        members=members,
        tilts=tilts,
        azimuths=azimuths_array,
        shell_ids=shell_ids,
        nav_offsets=nav,
        dp_offsets=dp,
        reference=reference,
        images=images,
        patterns=patterns,
    )
