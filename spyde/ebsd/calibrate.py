"""calibrate.py — solve an EBSD scan's pattern centre from its own patterns.

The pattern centre (where the beam spot sits relative to the screen, and how
far away the screen is) is rarely known well enough to index with: a few
percent off and every orientation is a few degrees wrong. The patterns
themselves pin it down. de_ebsd's ``determine_pattern_center`` fits ONE
centre shared by several patterns together with each pattern's orientation —
a single pattern cannot separate a centre shift from a small rotation, but
several patterns at different orientations can.

The steps, all on de_ebsd:

1. remove the detector's fixed intensity profile (the mean of the patterns)
   and each pattern's own background (the smooth backscatter gradient);
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
#: kikuchipy's real Ni calibration patterns (nine, 480×480) solve to within
#: 0.002 of kikuchipy's own refined centre from starts up to 0.05 off, at a
#: fit of 0.35–0.46. A failed solve "improves" the fit by wandering off along
#: the noise and stops near 0.08: those patterns with bad seeds, or
#: kikuchipy's 60×60 nickel scan, which moves 0.05 from its recorded centre.
MIN_FIT = 0.2
#: Fewer patterns than this do not average their bands away, so no static
#: background is estimated from them.
MIN_PATTERNS_FOR_STATIC = 9

# ── The search, for a centre not known to within the fit's reach ──
#: The fit above only polishes: it moves the centre a few hundredths at most,
#: so from the usual defaults (0.5, 0.5, 0.55) it cannot reach a real centre
#: like kikuchipy's Ni scan's (0.42, 0.21, 0.50) or de-twin's (0.5, 0.3, 0.6),
#: and stops at a fit of about 0.1. The search first scores candidate centres
#: on a grid, by how far each pattern's best dictionary match stands above
#: that pattern's other matches (in standard deviations). The raw best
#: correlation is NOT that score: it rises steadily with the screen distance
#: (a far screen sees a smoother, easier-to-match pattern) and peaks at the
#: edge of any grid. Measured on both scans, the grid's best cell is within
#: 0.05 of the centre, and the fit from it lands within 0.0001 (Ni) and 0.004
#: (de-twin) of it.
SEARCH_PC_Y = np.round(np.arange(0.1, 0.91, 0.1), 3)
SEARCH_PC_Z = np.round(np.arange(0.4, 1.01, 0.1), 3)
#: Finer grids around the best candidates, each half the last step.
SEARCH_REFINE_STEPS = (0.05, 0.025)
#: Candidates carried from one grid to the next, and finally fitted.
SEARCH_KEPT = 2
#: The scoring dictionary: coarse and matched binned, since it only has to
#: rank centres a grid step apart. Its orientations are sampled once for the
#: whole search: sampling the fundamental zone is 0.8 s on the CPU, eight
#: times the projection of a centre's dictionary, and it was being redone for
#: each of ~160 centres.
SEARCH_STEP_DEG = 5.0
SEARCH_BINNING = 2
#: Orientations projected at a time, which bounds the device memory a
#: dictionary takes while it is simulated.
SEARCH_CHUNK = 4096


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
    # a line scan gets as many patterns as a map, enough for a static background
    per_axis = grid if len(nav_shape) > 1 else grid * grid
    axes = [np.unique(np.linspace(0, n - 1, min(per_axis, n)).round().astype(int))
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


def static_background(patterns) -> np.ndarray | None:
    """The detector's fixed intensity profile, as the mean of *patterns*.

    Patterns at different orientations average their bands away, leaving the
    profile every frame shares. Without it the seeds go wrong: on kikuchipy's
    real Ni calibration patterns, dictionary seeds were right for 5 of 9
    patterns at the true centre (0 of 9 from 0.03 off) with only the
    per-pattern background removed, and 9 of 9 (8 of 9) with this subtracted
    too — as good as the background recorded with the scan. Too few patterns
    still carry their own bands, so then there is none.
    """
    if len(patterns) < MIN_PATTERNS_FOR_STATIC:
        return None
    return np.asarray(patterns, np.float32).mean(0)


def solve_pattern_centre(patterns, master, setup, point_group, *,
                         device=None, positions=(), orientations=None) -> Calibration:
    """Fit the pattern centre shared by *patterns* ``(N, rows, cols)``.

    *master* is a de_ebsd ``MasterPattern`` of the phase at the scan's beam
    energy; *setup* the SEM :class:`~spyde.ebsd.microscope.Setup`, whose
    angles are held and whose centre is the starting guess. *point_group* is
    the phase's (orix), which bounds the seeding dictionary; *orientations*
    are that dictionary's (:func:`seed_orientations`), passed in when several
    starts are fitted, since sampling them is most of a seed's cost.
    """
    import torch
    import de_ebsd
    from de_ebsd.index import determine_pattern_center, remove_background

    patterns = np.asarray(patterns, np.float32)
    detector = _detector(setup, (setup["pc_x"], setup["pc_y"], setup["pc_z"]),
                         patterns.shape[-2:])
    device = device or master.data.device
    master = master.to(device)
    clean = remove_background(patterns, static_bg=static_background(patterns),
                              dynamic=True, device=device)
    if orientations is None:
        orientations = seed_orientations(point_group)
    seeds = _best_matches(clean, master, detector, orientations, device)
    pc, rotations, info = determine_pattern_center(
        clean, master, detector, seeds, return_orientations=True, device=device)
    pc = tuple(float(v) for v in torch.as_tensor(pc).detach().cpu().numpy().reshape(3))
    # orix rotations here are Bunge matrices (sample → crystal); the viewer
    # keeps crystal → sample.
    bunge = np.asarray(rotations.to_matrix()).reshape(-1, 3, 3)
    return Calibration(pc=pc, rotations=np.transpose(bunge, (0, 2, 1)),
                       fit=float(info["ncc"]), fit_before=float(info["ncc_seed"]),
                       positions=list(positions))


def _detector(setup, pc, shape):
    import de_ebsd

    return de_ebsd.EBSDDetector(
        shape=tuple(shape), pc=tuple(float(v) for v in pc),
        sample_tilt=setup["sample_tilt"], tilt=setup["detector_tilt"],
        azimuthal=setup["azimuthal"], twist=setup["twist"])


def grid_search(score, start_x: float, *, kept: int = SEARCH_KEPT) -> list[tuple]:
    """The *kept* best centres by *score* ``(pc) -> float``: the coarse
    (pcy, pcz) grid at *start_x*, then finer grids in all three around the
    best so far. Each centre is scored once."""
    scores: dict[tuple, float] = {}

    def ranked(candidates):
        for pc in candidates:
            if pc not in scores:
                scores[pc] = float(score(pc))
        return sorted(set(candidates), key=scores.__getitem__, reverse=True)[:kept]

    best = ranked([(round(float(start_x), 4), float(y), float(z))
                   for y in SEARCH_PC_Y for z in SEARCH_PC_Z])
    for step in SEARCH_REFINE_STEPS:
        offsets = (-step, 0.0, step)
        best = ranked([tuple(round(c + d, 4) for c, d in zip(centre, (dx, dy, dz)))
                       for centre in best
                       for dx in offsets for dy in offsets for dz in offsets])
    return best


def search_pattern_centre(patterns, master, setup, point_group, *, device=None,
                          positions=(), progress=None) -> Calibration:
    """:func:`solve_pattern_centre` from wherever the centre is: a
    :func:`grid_search` for the candidates, then the fit from each, the best
    fit kept. *setup*'s centre gives only the grid's pcx. About 25 s: some
    160 centres are scored; *progress* ``(done, total)`` is called after each,
    and after each fit."""
    import de_ebsd  # noqa: F401  (fail early, before the slow part)
    from de_ebsd.index import remove_background

    patterns = np.asarray(patterns, np.float32)
    device = device or master.data.device
    master = master.to(device)
    clean = remove_background(patterns, static_bg=static_background(patterns),
                              dynamic=True, device=device)
    score = _lead_scorer(clean, master, setup, point_group, device)
    expected = (len(SEARCH_PC_Y) * len(SEARCH_PC_Z)
                + len(SEARCH_REFINE_STEPS) * SEARCH_KEPT * 27 + SEARCH_KEPT)
    done = 0

    def tick():
        nonlocal done
        done += 1
        if progress is not None:
            progress(min(done, expected), expected)

    def contrast(pc):
        lead = score(pc)
        tick()
        return lead

    candidates = grid_search(contrast, setup["pc_x"])
    log.info("pattern centre search: candidates %s", candidates)
    # Every candidate is fitted: the best-scored is not always the best fit.
    orientations = seed_orientations(point_group)
    fits = []
    for pc in candidates:
        start = dict(setup) if isinstance(setup, dict) else dict(setup.values)
        start.update(zip(("pc_x", "pc_y", "pc_z"), pc))
        fits.append(solve_pattern_centre(patterns, master, start, point_group, device=device,
                                         positions=positions, orientations=orientations))
        tick()
    return max(fits, key=lambda calibration: calibration.fit)


def _normalized_rows(x):
    x = x - x.mean(-1, keepdim=True)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _lead_scorer(patterns, master, setup, point_group, device):
    """The search's score of a centre, ``(pc) -> float``: how far each of
    *patterns*' best match in the centre's dictionary stands above its other
    matches, in standard deviations, averaged over the patterns. The
    dictionary is de_ebsd's ``DictionaryIndexer``'s, binned by
    `SEARCH_BINNING`, with the orientations and the binned patterns made once
    for every centre scored."""
    import torch
    import torch.nn.functional as F
    from orix.sampling import get_sample_fundamental

    rotations = get_sample_fundamental(resolution=SEARCH_STEP_DEG, point_group=point_group)
    quaternions = torch.as_tensor(np.asarray(rotations.data), dtype=torch.float32)
    binned = torch.as_tensor(patterns).to(device=device, dtype=torch.float32)
    binned = F.avg_pool2d(binned[:, None], SEARCH_BINNING)[:, 0]
    shape = tuple(binned.shape[-2:])
    binned = _normalized_rows(binned.reshape(len(binned), -1))

    def score(pc) -> float:
        detector = _detector(setup, pc, shape)
        with torch.no_grad():
            parts = []
            for start in range(0, len(quaternions), SEARCH_CHUNK):
                simulated = master.project(detector, quaternions[start:start + SEARCH_CHUNK],
                                           rescale=False)
                simulated = _normalized_rows(simulated.reshape(len(simulated), -1).float())
                parts.append(binned @ simulated.T)
            scores = torch.cat(parts, dim=1)
            lead = (scores.max(1).values - scores.mean(1)) / scores.std(1)
        return float(lead.mean())

    return score


def seed_orientations(point_group):
    """The seeding dictionary's orientations, `SEED_STEP_DEG` apart over
    *point_group*'s fundamental zone, as ``(N, 4)`` quaternions."""
    from orix.sampling import get_sample_fundamental

    rotations = get_sample_fundamental(resolution=SEED_STEP_DEG, point_group=point_group)
    return np.asarray(rotations.data, np.float32)


def _best_matches(patterns, master, detector, orientations, device):
    """Each of *patterns*' best match among *orientations* projected from
    *master* onto *detector*: de_ebsd's ``DictionaryIndexer.index``, with the
    orientations given rather than sampled again."""
    import torch
    from orix.quaternion import Rotation

    quaternions = torch.as_tensor(orientations)
    flat = torch.as_tensor(patterns).to(device=device, dtype=torch.float32)
    flat = _normalized_rows(flat.reshape(len(flat), -1))
    best = torch.full((len(flat),), -torch.inf, device=flat.device)
    at = torch.zeros(len(flat), dtype=torch.long, device=flat.device)
    with torch.no_grad():
        for start in range(0, len(quaternions), SEARCH_CHUNK):
            simulated = master.project(detector, quaternions[start:start + SEARCH_CHUNK],
                                       rescale=False)
            simulated = _normalized_rows(simulated.reshape(len(simulated), -1).float())
            score, index = (flat @ simulated.T).max(1)
            better = score > best
            best = torch.where(better, score, best)
            at = torch.where(better, index + start, at)
    return Rotation(orientations[at.cpu().numpy()])
