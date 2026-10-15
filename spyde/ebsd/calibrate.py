"""calibrate.py — solve an EBSD scan's pattern centre from its own patterns.

The pattern centre (where the beam spot sits relative to the screen, and how
far away the screen is) is rarely known well enough to index with: a few
percent off and every orientation is a few degrees wrong. The patterns
themselves pin it down. de_ebsd's ``determine_pattern_center`` fits ONE
centre shared by several patterns together with each pattern's orientation —
a single pattern cannot separate a centre shift from a small rotation, but
several patterns at different orientations can.

The steps, all on de_ebsd:

1. remove each pattern's background (the smooth backscatter gradient);
2. seed an orientation per pattern by dictionary indexing against the
   phase's dynamical master pattern (a 3° grid over its fundamental zone);
3. fit the shared centre, refining the orientations alongside.

The detector geometry is the viewer's (:mod:`spyde.ebsd.microscope`), which
agrees with de_ebsd's ``EBSDDetector`` exactly (both follow kikuchipy).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

#: Patterns read from the scan: a grid this many positions on a side, spread
#: across it, so the orientations differ (what separates centre from rotation).
CALIBRATION_GRID = 5
#: Orientation step of the seeding dictionary, in degrees. The centre fit
#: refines the orientations itself, so a seed only has to be close. A
#: spherical index at a cheap bandwidth was tried first and is NOT close
#: enough on small noisy patterns: its seeds matched no better than chance,
#: and the fit, with nothing to start from, did not move.
SEED_STEP_DEG = 3.0
#: Patterns are binned to at most this many pixels on a side. The centre is
#: measured in fractions of the screen, so binning does not change it, and
#: the seeding dictionary's cost grows with every pixel.
MAX_SIDE = 160
#: The mean per-tile correlation below which a solved centre is not trusted.
#: Patterns too small or noisy to pin the centre still "improve" the fit by
#: wandering off along the noise: kikuchipy's 60×60 nickel scan reaches 0.08
#: while moving 0.05 away from its recorded centre. Clean synthetic patterns
#: reach 0.8.
MIN_FIT = 0.2


@dataclass
class Calibration:
    pc: tuple[float, float, float]
    rotations: np.ndarray        # (N, 3, 3) crystal → sample, one per pattern
    fit: float                   # mean per-tile NCC at the solved geometry
    fit_before: float            # … and at the starting one
    positions: list              # the navigation indices the patterns came from

    @property
    def trusted(self) -> bool:
        return self.fit >= MIN_FIT


def calibration_positions(nav_shape, grid: int = CALIBRATION_GRID) -> list[tuple]:
    """Navigation indices spread over a scan — the centre one first.

    *nav_shape* is hyperspy's navigation shape (x first). A scan with no
    navigation gives the single position ``()``.
    """
    if not nav_shape:
        return [()]
    axes = [np.unique(np.linspace(0, n - 1, min(grid, n)).round().astype(int))
            for n in nav_shape]
    positions = [tuple(int(v) for v in combo)
                 for combo in np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1).T]
    centre = tuple(n // 2 for n in nav_shape)
    positions.sort(key=lambda p: sum((a - c) ** 2 for a, c in zip(p, centre)))
    return positions


def read_patterns(signal, positions) -> np.ndarray:
    """The frames at *positions*, as ``(N, rows, cols)`` float32, binned to
    at most :data:`MAX_SIDE` on a side.

    Reads one frame at a time — never the whole scan (CLAUDE.md, memory
    safety): ``inav`` of a lazy signal is a lazy one-frame slice.
    """
    frames = []
    for position in positions:
        frame = signal.inav[position] if position else signal
        frames.append(_bin(np.asarray(frame.data, dtype=np.float32)))
    return np.stack(frames)


def _bin(frame: np.ndarray) -> np.ndarray:
    factor = int(np.ceil(max(frame.shape) / MAX_SIDE))
    if factor <= 1:
        return frame
    rows, cols = (frame.shape[0] // factor) * factor, (frame.shape[1] // factor) * factor
    return frame[:rows, :cols].reshape(rows // factor, factor, cols // factor, factor).mean((1, 3))


def solve_pattern_centre(patterns, master, setup, point_group, *,
                         device=None, positions=()) -> Calibration:
    """Fit the pattern centre shared by *patterns* ``(N, rows, cols)``.

    *master* is a de_ebsd ``MasterPattern`` of the phase at the scan's beam
    energy; *setup* the SEM :class:`~spyde.ebsd.microscope.Setup`, whose
    angles are held and whose centre is the starting guess. *point_group* is
    the phase's (orix), which bounds the seeding dictionary.
    """
    import torch
    import de_ebsd
    from de_ebsd.index import (DictionaryIndexer, determine_pattern_center,
                               remove_background)

    patterns = np.asarray(patterns, np.float32)
    detector = de_ebsd.EBSDDetector(
        shape=tuple(patterns.shape[-2:]),
        pc=(setup["pc_x"], setup["pc_y"], setup["pc_z"]),
        sample_tilt=setup["sample_tilt"], tilt=setup["detector_tilt"],
        azimuthal=setup["azimuthal"], twist=setup["twist"])
    device = device or master.data.device
    master = master.to(device)
    clean = remove_background(patterns, dynamic=True, device=device)
    seeds, _quality = DictionaryIndexer(master, detector, point_group=point_group,
                                        resolution=SEED_STEP_DEG,
                                        device=device).index(clean)
    pc, rotations, info = determine_pattern_center(
        clean, master, detector, seeds, return_orientations=True, device=device)
    pc = tuple(float(v) for v in torch.as_tensor(pc).detach().cpu().numpy().reshape(3))
    # orix rotations here are Bunge matrices (sample → crystal); the viewer
    # keeps crystal → sample.
    bunge = np.asarray(rotations.to_matrix()).reshape(-1, 3, 3)
    return Calibration(pc=pc, rotations=np.transpose(bunge, (0, 2, 1)),
                       fit=float(info["ncc"]), fit_before=float(info["ncc_seed"]),
                       positions=list(positions))
