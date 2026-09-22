"""Drive the vendored quantem ACOM matcher from SpyDE's vectors and phases.

Everything SpyDE-shaped lives here so that ``spyde/external/quantem`` stays a
mechanical copy of upstream: this module converts an orix ``Phase`` into the
``Crystal`` their library generation needs, presents a
:class:`~spyde.signals.diffraction_vectors.SpyDEDiffractionVectors` as the peak
container their matcher reads, and decodes the result into the same
``VectorOrientationResult`` our own fit returns.

The live preview under the crosshair is driven from here
(:class:`SinglePatternFitter`, via ``vector_overlay``) and so is the whole-field
fit (:func:`compute_vector_orientation_quantem`).
``spyde/tests/benchmark_quantem_orientation.py`` times it on real data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from spyde.device_lock import accelerator_lock
from spyde.signals.orientation_map import VectorOrientationResult
from spyde.signals.diffraction_vectors import COL_KX, COL_KY, COL_INTENSITY

# Re-exported rather than copied, so they cannot drift from the matcher they
# describe. (The defaults module imports nothing, so this costs no torch.)
#: How near a simulated reflection has to be to count as the same peak, Å⁻¹.
#: How weak a reflection may be, relative to the strongest, and still be one a
#: detector would register.
from spyde.external.quantem.diffraction.defaults import (  # noqa: E402
    MIN_SIM_INTENSITY_REL, PAIR_DISTANCE,
)

log = logging.getLogger(__name__)

#: Their matcher reads these three columns by name; the order is the order of
#: the columns in the arrays :class:`PeaksAdapter` hands back.
PEAK_FIELDS = ("qx", "qy", "intensity")

#: Upstream's ``match_orientations(min_detector_fraction=...)`` default. A zone
#: axis whose template has less than this fraction of its weight on the detector
#: scores zero, because most of what would confirm it was never measured.
#:
#: Unlike the two above, upstream keeps this one in a signature rather than in
#: its defaults module, so this IS a copy and can drift. It has to be passed
#: explicitly and it has to stay non-zero, because it is the seam the zone mask
#: rides (:meth:`SinglePatternFitter.set_zone_mask`) — at zero, every mask
#: would silently do nothing. ``test_quantem_adapter`` reads the signature and
#: fails if the two stop agreeing.
MIN_DETECTOR_FRACTION = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# orix Phase → quantem Crystal
# ─────────────────────────────────────────────────────────────────────────────

def phase_to_crystal(phase, name: Optional[str] = None, verbose: bool = False,
                     **crystal_kwargs):
    """Build a quantem ``Crystal`` from an orix ``Phase``.

    Deliberately not ``Crystal.from_cif``: that reads the file through ase,
    whose cif parser rejects spacegroup spellings orix accepts (our own Silver
    fixture says ``F m 3 m``, which ase will not resolve). Going through the
    ``Phase`` also means this works for every phase the sample-phases UI can
    produce, whether or not it came from a file.

    The structure arrives already expanded to the full cell with fractional
    coordinates, which is exactly what ``Crystal`` reads off the ase atoms.
    """
    from ase import Atoms
    from ase.data import atomic_numbers

    from spyde.external.quantem.diffraction.crystal import Crystal

    structure = phase.structure
    lattice = structure.lattice
    # diffpy writes the element with its charge state ("Ag2+"); ase wants the
    # bare symbol.
    symbols = ["".join(c for c in str(atom.element) if c.isalpha()).capitalize()
               for atom in structure]
    unknown = sorted({s for s in symbols if s not in atomic_numbers})
    if unknown:
        raise ValueError(f"phase {phase.name!r} has unrecognised elements: {unknown}")

    atoms = Atoms(
        symbols=symbols,
        scaled_positions=np.array([atom.xyz for atom in structure], dtype=float),
        cell=[lattice.a, lattice.b, lattice.c,
              lattice.alpha, lattice.beta, lattice.gamma],
        pbc=True,
    )
    occupancy = np.array([float(getattr(atom, "occupancy", 1.0)) for atom in structure])
    if not np.allclose(occupancy, 1.0):
        atoms.set_array("occupancy", occupancy)
    return Crystal(atoms, name=name or str(phase.name), verbose=verbose,
                   **crystal_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# SpyDEDiffractionVectors → their peak container
# ─────────────────────────────────────────────────────────────────────────────

class _Cell:
    """One scan position's peaks, as their matcher expects to find them."""

    __slots__ = ("array",)

    def __init__(self, array: np.ndarray):
        self.array = array


class PeaksAdapter:
    """Present SpyDE's CSR vectors as the peak container their matcher reads.

    Their ``OrientationMap`` touches a small read-only slice of upstream's
    ragged ``Vector``: ``.shape``, ``.fields``, ``.metadata``,
    ``peaks[row, column].array`` and ``.select_fields(...).flatten()``. Meeting
    that here, rather than converting into their container, is what keeps the
    vendored files unmodified — they only ever duck-type.

    Coordinates are converted to Å⁻¹ once, on construction. Our vectors are
    stored in the detector's own units so they draw on the pattern they were
    found in, while the template library is always Å⁻¹; this is the same join
    ``vector_orientation.measured_in_inverse_angstrom`` makes for our own fit.

    The per-position arrays are materialised up front because the matcher reads
    them repeatedly — once to select valid positions, once per batch to build
    polar images, and again in every refinement round.
    """

    fields = list(PEAK_FIELDS)

    def __init__(self, vectors, inverse_angstrom_factor: float = 1.0,
                 t: Optional[int] = None, rotation_ccw_deg: float = 0.0):
        ny, nx = vectors.nav_shape
        self.shape = (ny, nx)
        self.metadata = {"rotation_ccw_deg": float(rotation_ccw_deg)}
        use_time = t is not None and getattr(vectors, "n_time", 0) > 0
        factor = float(inverse_angstrom_factor)

        self._cells = []
        for row in range(ny):
            for column in range(nx):
                peaks = (vectors.at_t(row, column, t) if use_time
                         else vectors.at(row, column))
                array = np.empty((len(peaks), 3), dtype=np.float64)
                if len(peaks):
                    array[:, 0] = peaks[:, COL_KX] * factor
                    array[:, 1] = peaks[:, COL_KY] * factor
                    array[:, 2] = peaks[:, COL_INTENSITY]
                self._cells.append(_Cell(array))

    def __getitem__(self, index) -> _Cell:
        row, column = index
        return self._cells[row * self.shape[1] + column]

    def select_fields(self, *names) -> "PeaksAdapter":
        if tuple(names) != PEAK_FIELDS:
            raise NotImplementedError(
                f"the adapter carries {PEAK_FIELDS}, not {names}")
        return self

    def flatten(self) -> np.ndarray:
        """Every peak in the scan as one ``(K, 3)`` array.

        Used only by ``detector_q_max="auto"``, which measures the detector
        footprint from the peaks themselves.
        """
        return np.vstack([cell.array for cell in self._cells])

    @property
    def peak_counts(self) -> np.ndarray:
        return np.array([len(cell.array) for cell in self._cells], dtype=int)


class _RowBand:
    """A horizontal band of a scan's peaks, as the matcher reads a scan.

    The match is per position, so matching the scan a band at a time against
    the one plan gives the same answer as matching it whole — and lets the
    band's orientations be shown, and the count reported, while the rest is
    still running. The refinement is NOT banded: its neighbour rescue looks
    across positions, so it runs on the whole reassembled map.
    """

    fields = list(PEAK_FIELDS)

    def __init__(self, peaks: PeaksAdapter, row_start: int, row_stop: int):
        self._peaks = peaks
        self._row_start = int(row_start)
        self.shape = (int(row_stop) - int(row_start), peaks.shape[1])
        self.metadata = peaks.metadata

    def __getitem__(self, index) -> _Cell:
        row, column = index
        return self._peaks[row + self._row_start, column]

    def select_fields(self, *names) -> "_RowBand":
        if tuple(names) != PEAK_FIELDS:
            raise NotImplementedError(
                f"the adapter carries {PEAK_FIELDS}, not {names}")
        return self

    def flatten(self) -> np.ndarray:
        rows, columns = self.shape
        return np.vstack([self[row, column].array
                          for row in range(rows) for column in range(columns)])

    @property
    def peak_counts(self) -> np.ndarray:
        rows, columns = self.shape
        return np.array([len(self[row, column].array)
                         for row in range(rows) for column in range(columns)],
                        dtype=int)


#: Rows matched per band by :func:`compute_vector_orientation_quantem`. Small
#: enough that the map visibly fills, large enough that a band is still a
#: batch: 8 rows of a 64-wide scan is 512 patterns, four of the matcher's
#: own batches.
BAND_ROWS = 8


def band_schedule(rows: int, band_rows: int, ramp: bool = True) -> list:
    """``[(row_start, row_stop), ...]`` covering *rows*. With *ramp* the bands
    start at one row and double up to *band_rows*, so the first rows are on
    screen after one row's match rather than after eight — on a wide scan the
    difference between seeing the map start and staring at grey."""
    band_rows = max(1, int(band_rows or rows))
    bands, start, size = [], 0, (1 if ramp else band_rows)
    while start < rows:
        stop = min(rows, start + min(size, band_rows))
        bands.append((start, stop))
        start, size = stop, size * 2
    return bands

#: The tensors :meth:`OrientationMap.match_orientations` writes, in the order
#: they are reassembled from the bands.
_MATCH_FIELDS = ("quats", "corr", "corr_second", "reliability", "mirror")

#: How many times the neighbour rescue may run over the field. Upstream runs
#: it once; a mis-indexed patch wider than one position needs a pass per
#: position of depth, and each pass stops early when it changes nothing.
RESCUE_PASSES = 3


# ─────────────────────────────────────────────────────────────────────────────
# Strain
# ─────────────────────────────────────────────────────────────────────────────

def reciprocal_affine(orientation_map, match: int = 0,
                      pair_distance: Optional[float] = None,
                      sigma_excitation: Optional[float] = None,
                      min_pairs: int = 4, device: str = "cpu",
                      chunk: int = 2048):
    """Per-position ``A`` mapping simulated to measured peaks, in reciprocal
    space — ``(ny, nx, 2, 2)`` and the pair count, NaN where too few paired.

    Separate from :func:`strain_from_orientation_map` because the overlay wants
    the map itself, to draw the simulated pattern where the fit actually puts
    it rather than where an unstrained crystal would.
    """
    import torch

    from spyde.external.quantem.diffraction.rotations import quat_to_matrix

    plan = orientation_map.metadata.get("plan", {}) or {}
    refine = orientation_map.metadata.get("refine", {}) or {}
    delta = float(pair_distance if pair_distance is not None
                  else refine.get("pair_distance", plan.get("pair_distance", 0.05)))
    sigma = float(sigma_excitation if sigma_excitation is not None
                  else refine.get("sigma_excitation", plan.get("sigma_excitation", 0.04)))

    device = torch.device(device)
    dtype = torch.float64
    peaks = orientation_map.peaks
    rows, columns = orientation_map.quats.shape[:2]
    field_index = [peaks.fields.index(f) for f in PEAK_FIELDS]

    cells = [peaks[r, c].array for r, c in np.ndindex(rows, columns)]
    counts = np.array([cell.shape[0] for cell in cells])
    max_peaks = max(1, int(counts.max()))
    measured = np.full((len(cells), max_peaks, 2), 1e6)
    weights = np.zeros((len(cells), max_peaks))
    for i, cell in enumerate(cells):
        n = cell.shape[0]
        if n == 0:
            continue
        measured[i, :n] = cell[:, field_index[:2]]
        weights[i, :n] = np.clip(cell[:, field_index[2]], 0.0, None)
    measured = torch.as_tensor(measured, dtype=dtype, device=device)
    weights = torch.as_tensor(weights, dtype=dtype, device=device)

    reflections = orientation_map.crystal.g_vec.to(device=device, dtype=dtype)
    wavelength = orientation_map.wavelength
    quats = orientation_map.quats[..., match, :].reshape(-1, 4).to(
        device=device, dtype=dtype)
    eye = torch.eye(2, dtype=dtype, device=device)

    affine = torch.full((quats.shape[0], 2, 2), float("nan"), dtype=dtype, device=device)
    pair_count = torch.zeros(quats.shape[0], dtype=torch.long, device=device)

    for start in range(0, quats.shape[0], chunk):
        stop = min(start + chunk, quats.shape[0])
        g = torch.einsum("bij,gj->bgi", quat_to_matrix(quats[start:stop]), reflections)
        gz, g2 = g[..., 2], (g ** 2).sum(dim=-1)
        excitation = (2 * gz - wavelength * g2) / (2 - 2 * wavelength * gz)
        simulated = g[..., :2]
        distance = torch.cdist(simulated, measured[start:stop])
        nearest_distance, nearest = distance.min(dim=-1)
        paired = (torch.abs(excitation) < 2 * sigma) & (nearest_distance < delta)
        weight = torch.gather(weights[start:stop], 1, nearest) * (
            1 - nearest_distance / delta).clamp_min(0) * paired
        target = torch.gather(
            measured[start:stop], 1, nearest[..., None].expand(-1, -1, 2))

        # A = (sum w q_meas q_simᵀ)(sum w q_sim q_simᵀ)⁻¹
        m1 = torch.einsum("bp,bpi,bpj->bij", weight, target, simulated)
        m2 = torch.einsum("bp,bpi,bpj->bij", weight, simulated, simulated)
        solved = m1 @ torch.linalg.inv(m2 + 1e-12 * eye)
        # Enough pairs, and pairs that span the plane: four peaks along one
        # row of reflections make m2 rank one, which the regularised inverse
        # turns into a huge A rather than a refusal — strains of 60 on a
        # real scan, all from positions whose pairing could not have
        # determined the other direction.
        enough = (paired.sum(dim=1) >= min_pairs) & well_conditioned(m2)
        affine[start:stop][enough] = solved[enough]
        pair_count[start:stop] = paired.sum(dim=1)

    return (affine.reshape(rows, columns, 2, 2),
            pair_count.reshape(rows, columns))


#: The weakest direction of a pairing must carry at least this fraction of
#: the strongest (the ratio of the normal matrix's eigenvalues) for the
#: deformation to be solved from it.
CONDITION_FLOOR = 0.05

#: Beyond this magnitude a solved "strain" is a pairing failure, not a
#: strain: no crystal in a microscope is stretched by half.
MAX_STRAIN = 0.5


def well_conditioned(normal, floor: float = CONDITION_FLOOR):
    """``(B,)`` bool — whether each ``(B, 2, 2)`` symmetric normal matrix
    ``sum w q qᵀ`` has its paired peaks spanning both directions of the
    plane, i.e. its smaller eigenvalue is at least *floor* of the larger."""
    import torch

    trace = normal.diagonal(dim1=-2, dim2=-1).sum(-1)
    determinant = torch.linalg.det(normal)
    half_gap = (trace * trace / 4 - determinant).clamp_min(0).sqrt()
    smallest, largest = trace / 2 - half_gap, trace / 2 + half_gap
    return (largest > 0) & (smallest > floor * largest)


def strain_from_orientation_map(orientation_map, match: int = 0,
                                pair_distance: Optional[float] = None,
                                sigma_excitation: Optional[float] = None,
                                min_pairs: int = 4, device: str = "cpu",
                                chunk: int = 2048, reciprocal: bool = False):
    """Per-position strain, from measured against simulated peak positions.

    Their orientation refinement deliberately does not fit strain: peak
    positions carry almost no out-of-plane information, so the pose is refined
    first and the deformation solved afterwards, on the pairing the refined
    orientation implies. This is that solve —
    ``A = (sum w q_meas q_simᵀ)(sum w q_sim q_simᵀ)⁻¹`` — batched over the scan
    rather than looped per position as upstream's ``calculate_strain`` does.

    The strain is referenced to the crystal's own ideal lattice, so unlike
    lattice-vector strain mapping it is absolute, with no reference region to
    pick.

    ``A`` maps simulated to measured in RECIPROCAL space. Real space is its
    inverse transpose, so an expanded lattice reads as positive strain the way
    a microscopist expects; ``reciprocal=True`` returns the uninverted quantity
    instead, which is what our own pose fit reports and is therefore what a
    comparison against it needs.

    Returns ``(strain, pair_count)`` with strain ``(ny, nx, 3)`` holding
    ``[exx, eyy, exy]`` and NaN wherever too few peaks paired.
    """
    import torch

    affine, pair_count = reciprocal_affine(
        orientation_map, match=match, pair_distance=pair_distance,
        sigma_excitation=sigma_excitation, min_pairs=min_pairs, device=device,
        chunk=chunk)
    rows, columns = affine.shape[:2]
    affine = affine.reshape(-1, 2, 2)
    if not reciprocal:
        affine = real_space_deformation(affine)
    strain = _symmetric_strain(affine).reshape(rows, columns, 3).cpu().numpy()
    return strain.astype(np.float32), pair_count.cpu().numpy()


def real_space_deformation(affine):
    """``(B, 2, 2)`` reciprocal-space maps → their real-space deformations, the
    inverse transpose, NaN where a map is missing or singular.

    Inverting the batch in one call raised on a single singular member — a
    position whose paired peaks all lie on one line through the origin has a
    rank-one map — and took the whole scan's strain down with it ("batch
    element 1552 … The input matrix is singular"). One bad position is one
    NaN, like a position with too few pairs.
    """
    import torch

    finite = torch.isfinite(affine).all(dim=(-2, -1))
    clean = torch.nan_to_num(affine)
    # Singular relative to the map's own magnitude, so a well-conditioned map
    # of small entries is not mistaken for a degenerate one.
    scale = clean.abs().amax(dim=(-2, -1)).clamp_min(1e-12) ** 2
    invertible = finite & (torch.linalg.det(clean).abs() > 1e-8 * scale)
    out = torch.full_like(affine, float("nan"))
    if bool(invertible.any()):
        out[invertible] = torch.linalg.inv(affine[invertible]).transpose(-1, -2)
    return out


def _symmetric_strain(affine):
    """``(B, 2, 2)`` deformation → ``(B, 3)`` ``[exx, eyy, exy]``.

    Polar-decomposed rather than symmetrised: a fit is free to split the total
    transform between rotation and stretch, so ``0.5(A + Aᵀ) − I`` still holds
    whatever rotation the fit parked in ``A``. Taking the stretch factor of
    ``A = R S`` leaves the rotation with the orientation, where it belongs, and
    only the physical deformation in the strain — the same reasoning as our own
    ``vector_orientation.strain_from_pose``.
    """
    import torch

    finite = torch.isfinite(affine).all(dim=(-2, -1))
    out = torch.full(affine.shape[:-2] + (3,), float("nan"),
                     dtype=affine.dtype, device=affine.device)
    if not bool(finite.any()):
        return out
    good = affine[finite]
    _u, singular, vh = torch.linalg.svd(good)
    stretch = (vh.transpose(-1, -2) * singular[..., None, :]) @ vh
    identity = torch.eye(2, dtype=affine.dtype, device=affine.device)
    strain = stretch - identity
    components = torch.stack(
        (strain[..., 0, 0], strain[..., 1, 1], strain[..., 0, 1]), dim=-1)
    # A pairing that solved to a stretch of more than half is a failed
    # pairing that happened to be invertible; reported, it would set the
    # colour scale of the whole map.
    plausible = components.abs().amax(dim=-1) <= MAX_STRAIN
    components[~plausible] = float("nan")
    out[finite] = components
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Orientation convention
# ─────────────────────────────────────────────────────────────────────────────

#: Both sides use scalar-first unit quaternions and compose the same way (an
#: in-plane spin about the beam, applied to a zone-axis orientation), but their
#: rotation runs the opposite direction to the one orix reads — so the bridge
#: is the conjugate. Measured, not derived: on a real sped_ag region their
#: field matches ours to 100% of pixels in IPF-Z and 89/92% in IPF-X/Y under
#: ``"conjugate"``, versus 0/3/0% under ``"identity"``. IPF colour is the
#: metric that settles it, because both results render through the same
#: ``SpyDEOrientationMap.ipf_color_map`` and so cannot agree by coincidence of
#: convention. ``benchmark_quantem_orientation`` still sweeps both and prints
#: the winner; keep that sweep as the regression guard.
QUAT_CONVENTIONS = ("identity", "conjugate")


def quantem_quats_to_orix(quats: np.ndarray, convention: str = "conjugate"
                          ) -> np.ndarray:
    """Reinterpret their ``(..., 4)`` quaternions in orix's convention."""
    quats = np.asarray(quats, dtype=np.float64)
    if convention == "identity":
        out = quats
    elif convention == "conjugate":
        out = quats.copy()
        out[..., 1:] *= -1.0
    else:
        raise ValueError(f"unknown convention {convention!r}; "
                         f"expected one of {QUAT_CONVENTIONS}")
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Zone-angle table cache
# ─────────────────────────────────────────────────────────────────────────────

#: (zone axes, symmetry ops) → their pairwise symmetry-reduced angles, keyed by
#: identity. The tensors themselves are held so an id can never be recycled
#: onto a different object, and only a couple of plans are ever live at once.
_ZONE_ANGLE_CACHE: dict = {}
_ZONE_ANGLE_CACHE_LIMIT = 4
_zone_angle_cache_installed = False


def install_zone_angle_cache() -> None:
    """Stop the matcher rebuilding its zone-angle table on every call.

    ``match_orientations`` computes the pairwise symmetry-reduced angle between
    every pair of zone axes each time it runs, to place the exclusion ball for
    the second-best match. The table depends only on the plan's zone axes and
    the crystal's symmetry, so for a scan matched in one call it is a rounding
    error — but it dominates a per-pattern fit, which is what a live refine
    under the crosshair is. Measured on a 1127-zone Ag plan: a single-pattern
    match plus refine goes from 184.5 ms to 44.6 ms, i.e. 5.4 to 22.4 fits per
    second, which is the difference between unusable and interactive.

    Patching rather than editing, because ``spyde/external/quantem`` is a
    mechanical copy of upstream — the same shape as
    ``heavy_imports._patch_cached_dask_client``. Delete this the day upstream
    caches the table itself; nothing else depends on it.
    """
    global _zone_angle_cache_installed
    if _zone_angle_cache_installed:
        return
    from spyde.external.quantem.diffraction import orientation as _orientation

    original = _orientation.symmetry_reduced_zone_angles

    def cached(zone_axes, symmetry_ops):
        key = (id(zone_axes), id(symmetry_ops))
        hit = _ZONE_ANGLE_CACHE.get(key)
        if hit is None:
            # Hold the inputs so their ids stay meaningful for the entry's life.
            hit = (zone_axes, symmetry_ops, original(zone_axes, symmetry_ops))
            if len(_ZONE_ANGLE_CACHE) >= _ZONE_ANGLE_CACHE_LIMIT:
                _ZONE_ANGLE_CACHE.pop(next(iter(_ZONE_ANGLE_CACHE)))
            _ZONE_ANGLE_CACHE[key] = hit
        return hit[2]

    _orientation.symmetry_reduced_zone_angles = cached
    _zone_angle_cache_installed = True


# ─────────────────────────────────────────────────────────────────────────────
# Driving the matcher
# ─────────────────────────────────────────────────────────────────────────────

class GpuRefineOrientationMap:
    """Mixin that runs the pairwise refinement on an accelerator.

    Upstream's ``_refine_batched`` is already torch — what pins it to the CPU
    is that it allocates in float64 and walks the scan 64 positions at a time,
    so a full scan is over a thousand launches of small kernels. That is the
    cost, not numpy: on the full sped_ag scan the refinement is 90 s against
    20 s for the correlation it follows.

    This is a transcription of that method with the device, the dtype and the
    chunk size lifted out, so the same arithmetic runs in one pass per large
    chunk. It lives here rather than in the vendored file because
    ``spyde/external/quantem`` is a mechanical copy of upstream — see that
    package's docstring. Delete this the day the refinement takes a device
    upstream; ``benchmark_quantem_orientation --refine-parity`` is the gate
    that says the two still agree.

    Illumination averaging (a precession angle or convergence semiangle
    recorded on the map) routes through numpy inside upstream's envelope, so
    those maps fall back to the vendored implementation rather than silently
    changing what is computed.
    """

    #: Poses refined per pass. Sized so the ``(chunk, reflections, peaks)``
    #: distance matrix stays a few hundred MB.
    refine_chunk = 2048
    refine_device = "cpu"
    refine_dtype = None

    def _gpu_refine_active(self) -> bool:
        """False when the vendored implementation has to run instead."""
        import torch

        precession = float(self.metadata.get("precession_deg", 0.0) or 0.0)
        convergence = float(self.metadata.get("semiconv_mrad", 0.0) or 0.0)
        return (torch.device(self.refine_device).type != "cpu"
                and precession <= 0 and convergence <= 0)

    def _peak_table(self, device, dtype):
        """The measured peaks as one padded table, transferred once.

        Returns ``(coordinates, weights, counts)`` where the first two are
        ``(positions, max_peaks, ...)`` device tensors. Unused slots hold a
        coordinate far outside any pairing distance, exactly as upstream pads.
        """
        import torch

        rows, columns = self.peaks.shape[0], self.peaks.shape[1]
        field_index = [self.peaks.fields.index(f) for f in PEAK_FIELDS]
        cells = [self.peaks[r, c].array for r, c in np.ndindex(rows, columns)]
        counts = np.array([cell.shape[0] for cell in cells])
        max_peaks = max(1, int(counts.max()))
        coordinates = np.full((len(cells), max_peaks, 2), 1e6)
        weights = np.zeros((len(cells), max_peaks))
        for i, cell in enumerate(cells):
            n = cell.shape[0]
            if n == 0:
                continue
            coordinates[i, :n] = cell[:, field_index[:2]]
            w = np.clip(cell[:, field_index[2]], 0.0, None)
            weights[i, :n] = w / max(float(w.max()), 1e-12)
        return (torch.as_tensor(coordinates, dtype=dtype, device=device),
                torch.as_tensor(weights, dtype=dtype, device=device),
                counts)

    def _refine_poses(self, q, active, batch_measured, batch_weights, settings):
        """Refine a batch of (position, starting orientation) pairs.

        ``q`` holds ``(K, 4)`` starting orientations and the peak rows are
        already gathered to match them, so the caller decides what a pose is:
        the main pass hands one per scan position, the rescue pass hands one
        per (outlier, neighbour candidate). Returns the refined orientations
        and their pairing scores; entries outside ``active`` come back
        untouched.
        """
        import torch

        from spyde.external.quantem.diffraction.rotations import (
            qmult, quat_from_axis_angle, quat_to_matrix,
        )

        device, dtype = settings["device"], settings["dtype"]
        delta, sigma, sigma_env = settings["delta"], settings["sigma"], settings["sigma_env"]
        tg, tilt_cap, power_env = settings["tg"], settings["tilt_cap"], settings["power_env"]
        min_pairs, refine_zone = settings["min_pairs"], settings["refine_zone"]
        reflections, intensities = settings["reflections"], settings["intensities"]
        wavelength = self.wavelength
        n_tilt = tg.shape[0]

        batch = q.shape[0]
        tilt_total = torch.zeros((batch, 2), dtype=dtype, device=device)
        score = torch.zeros(batch, dtype=dtype, device=device)
        for _ in range(settings["num_iterations"]):
            rotation = quat_to_matrix(q)
            g = torch.einsum("bij,gj->bgi", rotation, reflections)
            gz, g2 = g[..., 2], (g ** 2).sum(dim=-1)
            excitation = (2 * gz - wavelength * g2) / (2 - 2 * wavelength * gz)
            excited = torch.abs(excitation) < 2 * sigma
            distance = torch.cdist(g[..., :2], batch_measured)
            nearest_distance, nearest = distance.min(dim=-1)
            paired = excited & (nearest_distance < delta)
            pair_weight = torch.gather(batch_weights, 1, nearest) * (
                1 - nearest_distance / delta).clamp_min(0)
            pair_weight = pair_weight * paired
            ok = active & (paired.sum(dim=1) >= min_pairs)
            if not bool(ok.any()):
                break
            score = torch.where(ok, pair_weight.sum(dim=1), score)

            # in-plane rotation, closed form
            target = torch.gather(
                batch_measured, 1, nearest[..., None].expand(-1, -1, 2))
            residual = target - g[..., :2]
            tangent = torch.stack((-g[..., 1], g[..., 0]), dim=-1)
            numerator = (pair_weight[..., None] * tangent * residual).sum(dim=(1, 2))
            denominator = (pair_weight[..., None] * tangent * tangent).sum(dim=(1, 2))
            spin = torch.where(ok, numerator / denominator.clamp_min(1e-12),
                               torch.zeros_like(numerator))
            half = spin / 2
            zeros = torch.zeros_like(half)
            delta_q = torch.stack(
                (torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)
            q = torch.where(ok[:, None], qmult(delta_q, q), q)

            if not refine_zone:
                continue
            # zone-axis tilt from the intensity envelope, sparse over the
            # paired reflections only
            batch_index, reflection_index = torch.nonzero(paired, as_tuple=True)
            excitation_paired = excitation[batch_index, reflection_index]
            gy = g[batch_index, reflection_index, 1]
            gx = g[batch_index, reflection_index, 0]
            structure_factor = intensities[reflection_index]
            weight_paired = pair_weight[batch_index, reflection_index]
            shifted = (excitation_paired[:, None, None]
                       + tg[None, :, None] * gy[:, None, None]
                       - tg[None, None, :] * gx[:, None, None])
            predicted = (structure_factor[:, None, None]
                         * torch.exp(-(shifted ** 2) / (2 * sigma_env ** 2))
                         ).clamp_min(0) ** power_env
            envelope_numerator = torch.zeros(
                (batch, n_tilt, n_tilt), dtype=dtype, device=device
            ).index_add_(0, batch_index,
                         (weight_paired ** power_env)[:, None, None] * predicted)
            envelope_denominator = torch.zeros(
                (batch, n_tilt, n_tilt), dtype=dtype, device=device
            ).index_add_(0, batch_index, predicted ** 2)
            envelope = envelope_numerator / envelope_denominator.sqrt().clamp_min(1e-12)

            best = envelope.reshape(batch, -1).argmax(dim=1)
            best_x, best_y = best // n_tilt, best % n_tilt
            step = float(tg[1] - tg[0])
            tilt_x, tilt_y = tg[best_x].clone(), tg[best_y].clone()
            batch_range = torch.arange(batch, device=device)
            for axis, index, tilt in ((0, best_x, tilt_x), (1, best_y, tilt_y)):
                interior = (index > 0) & (index < n_tilt - 1)
                if not bool(interior.any()):
                    continue
                if axis == 0:
                    low = envelope[batch_range, (index - 1).clamp(0), best_y]
                    mid = envelope[batch_range, index, best_y]
                    high = envelope[batch_range, (index + 1).clamp(max=n_tilt - 1), best_y]
                else:
                    low = envelope[batch_range, best_x, (index - 1).clamp(0)]
                    mid = envelope[batch_range, best_x, index]
                    high = envelope[batch_range, best_x, (index + 1).clamp(max=n_tilt - 1)]
                curvature = 2 * mid - low - high
                tilt += torch.where(
                    interior & (curvature.abs() > 1e-12),
                    0.5 * (high - low) / curvature * step,
                    torch.zeros_like(mid))

            # trust region on the cumulative tilt from the start
            proposed = tilt_total + torch.stack((tilt_x, tilt_y), dim=-1)
            magnitude = torch.linalg.norm(proposed, dim=-1)
            proposed = proposed * torch.where(
                magnitude > tilt_cap, tilt_cap / magnitude.clamp_min(1e-12),
                torch.ones_like(magnitude))[:, None]
            tilt_step = torch.where(ok[:, None], proposed - tilt_total,
                                    torch.zeros_like(proposed))
            tilt_total = torch.where(ok[:, None], proposed, tilt_total)
            tilt_angle = torch.linalg.norm(tilt_step, dim=-1)
            turning = tilt_angle > 1e-10
            if bool(turning.any()):
                axis_vector = torch.zeros((batch, 3), dtype=dtype, device=device)
                axis_vector[turning, 0] = tilt_step[turning, 0] / tilt_angle[turning]
                axis_vector[turning, 1] = tilt_step[turning, 1] / tilt_angle[turning]
                q[turning] = qmult(
                    quat_from_axis_angle(axis_vector[turning], tilt_angle[turning]),
                    q[turning])
        return q, score

    def _refine_batched(self, scores, delta, sigma, sigma_env, tg, tilt_cap,
                        num_iterations, min_pairs, refine_zone, progress_bar,
                        chunk=None, power_env=None):
        import torch

        from spyde.external.quantem.diffraction.defaults import POWER_INTENSITY

        if power_env is None:
            power_env = POWER_INTENSITY
        if not self._gpu_refine_active():
            return super()._refine_batched(
                scores, delta=delta, sigma=sigma, sigma_env=sigma_env, tg=tg,
                tilt_cap=tilt_cap, num_iterations=num_iterations,
                min_pairs=min_pairs, refine_zone=refine_zone,
                progress_bar=progress_bar, power_env=power_env,
                **({} if chunk is None else {"chunk": chunk}))

        device = torch.device(self.refine_device)
        dtype = self.refine_dtype or torch.float32
        chunk = chunk or self.refine_chunk
        rows, columns, num_matches = self.quats.shape[:3]
        measured, weights, counts = self._peak_table(device, dtype)
        settings = dict(
            device=device, dtype=dtype, delta=delta, sigma=sigma,
            sigma_env=sigma_env, tg=tg.to(device=device, dtype=dtype),
            tilt_cap=tilt_cap, power_env=power_env, min_pairs=min_pairs,
            refine_zone=refine_zone, num_iterations=num_iterations,
            reflections=self.crystal.g_vec.to(device=device, dtype=dtype),
            intensities=self.crystal.struct_factors_int.to(device=device, dtype=dtype),
        )

        n_positions = rows * columns
        quats = self.quats.reshape(n_positions, num_matches, 4).to(
            device=device, dtype=dtype)
        correlation = self.corr.reshape(n_positions, num_matches).to(device)
        position_valid = torch.as_tensor(counts >= min_pairs, device=device)
        scores_flat = scores.reshape(-1)

        for start in range(0, n_positions, chunk):
            stop = min(start + chunk, n_positions)
            for match in range(num_matches):
                active = position_valid[start:stop] & (correlation[start:stop, match] > 0)
                if not bool(active.any()):
                    continue
                q, score = self._refine_poses(
                    quats[start:stop, match].clone(), active,
                    measured[start:stop], weights[start:stop], settings)
                quats[start:stop, match] = torch.where(
                    active[:, None], q, quats[start:stop, match])
                if match == 0:
                    scores_flat[start:stop] = torch.where(
                        active, score, scores_flat[start:stop].to(device)
                    ).to(torch.float64).cpu()

        self.quats = quats.reshape(rows, columns, num_matches, 4).to(
            device="cpu", dtype=torch.float64)
        # refine_orientations keeps these local, and the rescue pass needs them
        self._refine_state = dict(settings=settings, measured=measured,
                                  weights=weights, scores=scores)

    def refine_orientations(self, *args, **kwargs):
        """Upstream's refinement with the neighbour-rescue pass batched too.

        The rescue re-refines every position whose orientation disagrees with
        all of its neighbours, starting from each distinct neighbour
        orientation and keeping the best-scoring result. Upstream runs it as a
        Python loop over the single-position refinement, which on the full
        sped_ag scan costs 54 s against 1.3 s for everything else — 976
        outliers times up to eight candidates is 7808 refinements one at a
        time. They are just more (position, starting orientation) pairs, so
        they go through :meth:`_refine_poses` in one batch.

        One difference is deliberate and cannot be batched away. Upstream
        sweeps the outliers in order and writes each result straight back into
        ``quats``, so a later outlier reads whichever of its neighbours have
        already been rescued; a batch necessarily works from the snapshot
        taken before the pass. On a real scan that changes roughly 2 percent
        of positions, all inside the outlier set and all cases where several
        neighbour orientations score within the pass's own 2 percent
        acceptance margin. Refereed on that score the two come out level, so
        the speedup costs no accuracy — but it is a semantic difference, not
        rounding, and ``benchmark_quantem_orientation --refine-parity``
        reports it as one.
        """
        rescue = kwargs.pop("neighbor_rescue", True)
        passes = int(kwargs.pop("rescue_passes", 1))
        batched = kwargs.get("batched", True)
        refine_tilt = kwargs.get("refine_tilt", False)
        # Falling back leaves upstream wholly in charge, rescue included.
        if (not self._gpu_refine_active() or not rescue or not batched
                or refine_tilt or len(args) > 10):
            return super().refine_orientations(
                *args, neighbor_rescue=rescue, **kwargs)

        import torch

        self._refine_state = None
        result = super().refine_orientations(*args, neighbor_rescue=False, **kwargs)
        if self._refine_state is not None:
            threshold = float(kwargs.get("rescue_threshold_deg", 2.0))
            # One pass rescues a position from the neighbours it has NOW; a
            # position two steps into a mis-indexed patch only gets a good
            # neighbour once the first pass has fixed the one between. Repeat
            # until a pass changes nothing, up to ``rescue_passes``.
            for _ in range(max(1, passes)):
                before = self.quats.clone()
                self._rescue_batched(threshold)
                if torch.equal(before, self.quats):
                    break
            self.metadata["refine"]["neighbor_rescue"] = True
            self.metadata["refine"]["rescue_passes"] = int(max(1, passes))
        return result

    def _rescue_batched(self, threshold_deg: float) -> None:
        import torch

        from spyde.external.quantem.diffraction.rotations import (
            misorientation_angle_deg,
        )

        state = self._refine_state
        settings = state["settings"]
        device, dtype = settings["device"], settings["dtype"]
        rows, columns = self.quats.shape[:2]
        symmetry = self.crystal.sym_quats
        best_of_matches = self.quats[..., 0, :]

        # positions whose orientation disagrees with every neighbour
        closest = torch.full((rows, columns), torch.inf, dtype=torch.float64)
        for row_step, column_step in ((0, 1), (1, 0)):
            here = best_of_matches[: rows - row_step, : columns - column_step]
            there = best_of_matches[row_step:, column_step:]
            angle = misorientation_angle_deg(
                here.reshape(-1, 4), there.reshape(-1, 4), symmetry
            ).reshape(rows - row_step, columns - column_step)
            closest[: rows - row_step, : columns - column_step] = torch.minimum(
                closest[: rows - row_step, : columns - column_step], angle)
            closest[row_step:, column_step:] = torch.minimum(
                closest[row_step:, column_step:], angle)
        outliers = torch.nonzero(closest > threshold_deg)
        if outliers.shape[0] == 0:
            return

        offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                   if (dr, dc) != (0, 0)]
        n_offsets = len(offsets)
        n_outliers = outliers.shape[0]
        outlier_row, outlier_column = outliers[:, 0], outliers[:, 1]
        neighbour_row = outlier_row[:, None] + torch.tensor([d[0] for d in offsets])
        neighbour_column = outlier_column[:, None] + torch.tensor([d[1] for d in offsets])
        inside = ((neighbour_row >= 0) & (neighbour_row < rows)
                  & (neighbour_column >= 0) & (neighbour_column < columns))
        candidates = best_of_matches[neighbour_row.clamp(0, rows - 1),
                                     neighbour_column.clamp(0, columns - 1)]

        # Upstream keeps a neighbour only if it differs from every candidate
        # already kept by more than half a degree. That greedy scan is eight
        # sequential steps, so it vectorises across positions.
        separation = misorientation_angle_deg(
            candidates[:, :, None, :], candidates[:, None, :, :], symmetry)
        keep = torch.zeros((n_outliers, n_offsets), dtype=torch.bool)
        for j in range(n_offsets):
            distinct = torch.ones(n_outliers, dtype=torch.bool)
            for i in range(j):
                distinct &= ~keep[:, i] | (separation[:, j, i] > 0.5)
            keep[:, j] = inside[:, j] & distinct

        position = (outlier_row * columns + outlier_column)[:, None].expand(
            -1, n_offsets).reshape(-1).to(device)
        refined, refined_score = self._refine_poses(
            candidates.reshape(-1, 4).to(device=device, dtype=dtype),
            keep.reshape(-1).to(device),
            state["measured"][position], state["weights"][position], settings)
        refined = refined.reshape(n_outliers, n_offsets, 4)
        refined_score = refined_score.reshape(n_outliers, n_offsets)

        # Upstream accepts a candidate only if it beats the incumbent by 2%,
        # updating as it goes, so the order of the offsets is part of the
        # result — replay it rather than taking a plain maximum.
        scores_flat = state["scores"].reshape(-1)
        flat_position = (outlier_row * columns + outlier_column)
        best_quat = best_of_matches[outlier_row, outlier_column].to(
            device=device, dtype=dtype)
        best_score = scores_flat[flat_position].to(device=device, dtype=dtype)
        keep_device = keep.to(device)
        for j in range(n_offsets):
            take = keep_device[:, j] & (refined_score[:, j] > best_score * 1.02)
            best_score = torch.where(take, refined_score[:, j], best_score)
            best_quat = torch.where(take[:, None], refined[:, j], best_quat)

        quats = self.quats.clone()
        quats[outlier_row, outlier_column, 0] = best_quat.to(
            device="cpu", dtype=torch.float64)
        self.quats = quats
        scores_flat[flat_position] = best_score.to(
            device="cpu", dtype=torch.float64)


def orientation_map_class(refine_device: str = "cpu", refine_chunk: int = 2048,
                          refine_dtype=None):
    """The ``OrientationMap`` to build, given where the refinement should run.

    ``"cpu"`` returns upstream's class untouched, so the default path is
    exactly the vendored code.
    """
    from spyde.external.quantem.diffraction.orientation import OrientationMap

    if str(refine_device) == "cpu":
        return OrientationMap

    class _GpuRefine(GpuRefineOrientationMap, OrientationMap):
        pass

    _GpuRefine.__name__ = "GpuRefineOrientationMap"
    _GpuRefine.refine_device = str(refine_device)
    _GpuRefine.refine_chunk = int(refine_chunk)
    _GpuRefine.refine_dtype = refine_dtype
    return _GpuRefine


def build_orientation_map(peaks: PeaksAdapter, crystal, energy_ev: float,
                          k_max: float, angle_step_zone_axis_deg: float = 1.0,
                          angle_step_in_plane_deg: float = 5.0,
                          device: str = "cpu", verbose: bool = False,
                          refine_device: str = "cpu", refine_chunk: int = 2048,
                          refine_dtype=None, **plan_kwargs):
    """Structure factors + polar correlation plan, ready to match.

    ``k_max`` is the outer reciprocal radius in Å⁻¹, i.e. the same quantity our
    own library build calls ``reciprocal_radius`` / ``r_max``. Reflections
    beyond it cannot be measured, so generating them only costs plan size.
    """
    install_zone_angle_cache()
    if crystal.g_vec is None:
        crystal.calculate_structure_factors(k_max=k_max)
    map_class = orientation_map_class(refine_device, refine_chunk, refine_dtype)
    orientation_map = map_class.from_vectors(peaks, crystal, energy_ev=energy_ev)
    orientation_map.build_plan(
        angle_step_zone_axis_deg=angle_step_zone_axis_deg,
        angle_step_in_plane_deg=angle_step_in_plane_deg,
        device=device, verbose=verbose, **plan_kwargs)
    return orientation_map


def combine_phases(orientation_maps, phases_meta: list,
                   convention: str = "conjugate",
                   strains: Optional[list] = None,
                   params: Optional[dict] = None) -> VectorOrientationResult:
    """Pick, at each position, the phase whose pattern the peaks best support.

    Every phase needs its own plan — the polar shells ARE that crystal's
    reciprocal lattice radii, so nothing about a plan is shareable and the
    maps are matched independently. What makes combining them afterwards
    meaningful is that their correlation is a normalised cosine similarity:
    the library slices are unit vectors and the pattern is divided by its own
    norm, so a score is comparable across crystals and not only across
    patterns. A larger-cell phase does not win by having more reflections.

    ``strains`` is one array per map, in the same order, or None.
    """
    maps = list(orientation_maps)
    if not maps:
        raise ValueError("combine_phases needs at least one orientation map")

    correlation = np.stack(
        [np.asarray(m.corr[..., 0].cpu().numpy(), float) for m in maps])
    winner = correlation.argmax(axis=0)                     # (ny, nx)
    best_correlation = np.take_along_axis(
        correlation, winner[None], axis=0)[0].astype(np.float32)
    ny, nx = winner.shape

    quats = np.stack([
        quantem_quats_to_orix(m.quats[..., 0, :].cpu().numpy(), convention)
        for m in maps])                                     # (P, ny, nx, 4)
    chosen_quats = np.take_along_axis(
        quats, winner[None, ..., None], axis=0)[0].astype(np.float32)

    theta = np.stack([
        np.deg2rad(m.in_plane_angle_deg().cpu().numpy()) for m in maps])
    chosen_theta = np.take_along_axis(theta, winner[None], axis=0)[0].astype(np.float32)

    if strains is None:
        chosen_strain = np.full((ny, nx, 3), np.nan, np.float32)
    else:
        stacked = np.stack([np.asarray(s, np.float32) for s in strains])
        chosen_strain = np.take_along_axis(
            stacked, winner[None, ..., None], axis=0)[0].astype(np.float32)

    valid = best_correlation > 0
    counts = maps[0].peaks.peak_counts.reshape(ny, nx)
    n_matched = np.clip(counts, 0, np.iinfo(np.int16).max).astype(np.int16)

    result = VectorOrientationResult(
        quats=chosen_quats,
        phase_idx=np.where(valid, winner, 0).astype(np.int16),
        theta=chosen_theta,
        strain=np.where(valid[..., None], chosen_strain, np.nan).astype(np.float32),
        residual=np.full((ny, nx), np.nan, np.float32),
        friedel_asym=np.full((ny, nx), np.nan, np.float32),
        n_matched=n_matched,
        coarse_score=np.where(valid, best_correlation, 0.0).astype(np.float32),
        phases_meta=phases_meta,
        nav_shape=(ny, nx),
        params=dict(params or {}),
    )
    # Their discriminability metric, which our own path has no field for: how
    # far the best orientation beat the best one well away from it. This is
    # what a grain-boundary or amorphous mask is thresholded on.
    reliability = np.stack(
        [np.asarray(m.reliability.cpu().numpy(), float) for m in maps])
    result.reliability = np.take_along_axis(
        reliability, winner[None], axis=0)[0].astype(np.float32)
    return result


def to_result(orientation_map, phases_meta: list,
              convention: str = "conjugate",
              params: Optional[dict] = None,
              strain: Optional[np.ndarray] = None) -> VectorOrientationResult:
    """Decode a matched ``OrientationMap`` into our result container.

    ``strain`` comes from :func:`strain_from_orientation_map`, which is a
    separate pass because their refinement deliberately does not fit it; pass
    it in, or leave it None to get a NaN strain field. ``friedel_asym`` stays
    NaN — it is our own diagnostic, computed from the pose residual of ±g
    pairs, and has no counterpart on their side.

    ``coarse_score`` carries their correlation, which unlike ours is a
    normalised cosine similarity in [0, 1] and so is comparable across
    patterns. ``reliability`` (best minus best-outside-an-exclusion-ball) is
    attached as an attribute — the container has no field for it, and it is
    the discriminability metric our own path lacks entirely.
    """
    return combine_phases(
        [orientation_map], phases_meta, convention=convention,
        strains=None if strain is None else [strain], params=params)


# ─────────────────────────────────────────────────────────────────────────────
# The whole field
# ─────────────────────────────────────────────────────────────────────────────

def _band_winner(parts, convention: str = "conjugate"):
    """The best phase and its orientation at every position of the band just
    matched — ``parts[phase]`` is that phase's match tensors for the band.

    The same comparison :func:`combine_phases` makes for the finished scan;
    here it is made early so the band can be shown. A position no phase
    matched gets phase ``-1``.
    """
    correlation = np.stack(
        [np.asarray(part["corr"][..., 0].numpy(), float) for part in parts])
    winner = correlation.argmax(axis=0)
    best = np.take_along_axis(correlation, winner[None], axis=0)[0]
    quats = np.stack([
        quantem_quats_to_orix(part["quats"][..., 0, :].numpy(), convention)
        for part in parts])
    chosen = np.take_along_axis(quats, winner[None, ..., None], axis=0)[0]
    return chosen.astype(np.float32), np.where(best > 0, winner, -1).astype(np.int16)


def compute_vector_orientation_quantem(
        vectors, phases, energy_ev: float, k_max: float,
        inverse_angstrom_factor: float = 1.0,
        angle_step_zone_axis_deg: float = 1.0,
        angle_step_in_plane_deg: float = 5.0,
        device: str = "cuda", t: Optional[int] = None,
        progress=None, stopped_flag=None,
        params: Optional[dict] = None,
        on_band=None, band_rows: int = BAND_ROWS, ramp: bool = True,
        stage=None,
        ) -> Optional[VectorOrientationResult]:
    """Orientation, phase and strain for every position, by correlation match.

    One plan per phase, each matched and refined over the whole scan and then
    combined per position on the correlation — see :func:`combine_phases` for
    why comparing them is legitimate.

    The match runs in bands of ``band_rows`` scan rows against the plans built
    once up front, and ``on_band(row_start, row_stop, quats, phase_index)`` is
    called as each band lands with its best orientation per position, so the
    map can be shown filling in. The match is per position, so this is the
    same answer the whole scan gives in one call; the refinement, whose
    neighbour rescue looks across positions, runs on the reassembled map.

    The correlation runs on ``device`` and the refinement on the same one
    through the override in this module; both are worth accelerating at scan
    scale, which is the opposite of the single-pattern case, where a batch of
    one is launch-overhead bound and the CPU wins.

    ``progress(done, total)`` counts positions: each is matched once and
    refined once per phase; ``stage(text)`` is told which stage is running
    (plan, match, refine), since the count stands still through the plan
    build and the refinement and a still count reads as a hang.
    ``stopped_flag`` is polled between bands and stages, so closing the tree
    stops the run rather than leaving a scan's compute to finish into nothing.
    With *ramp* the first bands are one, two, four rows — see
    :func:`band_schedule`.
    """
    import torch

    def _stage(text: str) -> None:
        log.info("[vom] %s", text)
        if stage is not None:
            stage(text)

    peaks = PeaksAdapter(vectors, inverse_angstrom_factor=inverse_angstrom_factor,
                         t=t)
    rows, columns = peaks.shape
    total = rows * columns
    total_units = 2 * max(1, len(phases)) * total
    done = [0]

    def _advance(positions: int):
        done[0] += int(positions)
        if progress is not None:
            progress(done[0], total_units)

    def _stopped() -> bool:
        return bool(stopped_flag is not None and stopped_flag[0])

    # The Refine tab's settings, so the map is of what the preview showed.
    settings = dict(params or {})
    pair_distance = settings.get("pair_distance")
    sigma_excitation = settings.get("sigma_excitation")
    rescue_passes = int(settings.get("rescue_passes", RESCUE_PASSES))

    maps = []
    for index, phase in enumerate(phases):
        if _stopped():
            return None
        _stage(f"building the correlation plan for phase {index + 1} of "
               f"{len(phases)} ({getattr(phase, 'name', '') or 'phase'})…")
        crystal = phase_to_crystal(phase)
        with accelerator_lock(torch.device(device)):
            maps.append(build_orientation_map(
                peaks, crystal, energy_ev=energy_ev, k_max=k_max,
                angle_step_zone_axis_deg=angle_step_zone_axis_deg,
                angle_step_in_plane_deg=angle_step_in_plane_deg,
                device=device, refine_device=device))
    if not maps:
        return None

    parts = [[] for _ in maps]
    for row_start, row_stop in band_schedule(rows, band_rows, ramp):
        if _stopped():
            return None
        _stage(f"matching rows {row_start + 1}-{row_stop} of {rows}…")
        band = _RowBand(peaks, row_start, row_stop)
        for orientation_map, phase_parts in zip(maps, parts):
            orientation_map.peaks = band
            with accelerator_lock(torch.device(device)):
                orientation_map.match_orientations(progress_bar=False)
            phase_parts.append({name: getattr(orientation_map, name)
                                for name in _MATCH_FIELDS})
            _advance(band.shape[0] * columns)
        if on_band is not None:
            quats, phase_index = _band_winner(
                [phase_parts[-1] for phase_parts in parts])
            on_band(row_start, row_stop, quats, phase_index)

    strains = []
    for index, (orientation_map, phase_parts) in enumerate(zip(maps, parts)):
        if _stopped():
            return None
        _stage(f"refining phase {index + 1} of {len(maps)} and solving strain…")
        orientation_map.peaks = peaks
        for name in _MATCH_FIELDS:
            setattr(orientation_map, name,
                    torch.cat([part[name] for part in phase_parts], dim=0))
        with accelerator_lock(torch.device(device)):
            # Upstream's own refinement (the CPU path) rescues once and takes
            # no pass count; the batched override does.
            passes = ({"rescue_passes": rescue_passes}
                      if isinstance(orientation_map, GpuRefineOrientationMap) else {})
            orientation_map.refine_orientations(
                progress_bar=False, pair_distance=pair_distance,
                sigma_excitation=sigma_excitation, **passes)
            strain, _pairs = strain_from_orientation_map(
                orientation_map, device=device, pair_distance=pair_distance,
                sigma_excitation=sigma_excitation)
        strains.append(strain)
        _advance(total)

    if _stopped():
        return None
    from spyde.signals.orientation_map import phase_to_dict

    result = combine_phases(maps, [phase_to_dict(p) for p in phases],
                            strains=strains, params=dict(params or {}))
    if progress is not None:
        progress(total_units, total_units)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# One pattern at a time
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SinglePatternFit:
    """What one pattern's fit gives the overlay and the caret readout."""

    quat: np.ndarray            #: (4,) orientation, orix convention
    phase_index: int
    #: Cosine similarity against the matched template. Normalised to about
    #: [0, 1] and comparable across patterns and crystals, but not bounded by
    #: 1: the square-detector correction divides by a per-rotation template
    #: norm that is clamped away from zero, so a pattern fitting its template
    #: almost exactly can land slightly over (1.02 measured on clean silver).
    correlation: float
    reliability: float          #: best minus best outside the exclusion ball
    strain: np.ndarray          #: (3,) [exx, eyy, exy], NaN if too few pairs
    spots: np.ndarray           #: (M, 2) simulated peaks in Å⁻¹, as fitted
    intensities: np.ndarray     #: (M,) their relative intensity
    #: Median distance, Å⁻¹, from each MEASURED peak to the nearest drawn spot.
    #: Measured in that direction because a simulated pattern always carries
    #: reflections the peak finder did not detect, so scoring the spots instead
    #: would penalise a good fit for being complete.
    residual: float
    n_matched: int              #: measured peaks with a drawn spot on them


class SinglePatternFitter:
    """Fit one pattern against a plan built once, as the crosshair moves.

    The whole-field fit takes tens of seconds, so it cannot follow a navigator;
    one pattern against an existing plan takes about 40 ms, which can. This is
    the unit the live overlay and the Refine readout consume — the plan, the
    structure factors and the zone-angle table are all built in ``__init__``
    and reused for every position.

    Runs on the CPU by default, and that is measured rather than conservative:
    at a batch of one the GPU is launch-overhead bound, and the refinement is
    12 ms on the CPU against 39 ms on CUDA. The whole-field path wants CUDA;
    this one does not.

    One map per phase, each with its own plan, picked per pattern on the
    correlation — the same comparison :func:`combine_phases` makes for a scan.

    ``peaks`` is the WHOLE scan's adapter, and it is required even though only
    one position is ever fitted: the plan measures the detector footprint from
    the peaks (``detector_q_max="auto"``) so it can renormalise by the template
    norm actually on the detector at each in-plane angle. Built from a single
    position — or worse, none — that measurement is wrong or silently skipped,
    and every correlation comes out low and not comparable with the batch
    path's. Measured on sped_ag: median correlation 0.425 against 0.944.
    """

    def __init__(self, crystals, peaks, energy_ev: float, k_max: float,
                 angle_step_zone_axis_deg: float = 1.0,
                 angle_step_in_plane_deg: float = 5.0,
                 device: str = "cpu", min_peaks: int = 5,
                 convention: str = "conjugate",
                 inverse_angstrom_factor: float = 1.0, **plan_kwargs):
        self.min_peaks = int(min_peaks)
        self.convention = convention
        self.device = device
        #: Remembered so the overlay can hand over raw rows and not restate the
        #: detector's units at every position.
        self.inverse_angstrom_factor = float(inverse_angstrom_factor)
        #: The two knobs worth exposing live. Both are arguments of the
        #: REFINEMENT, not of the plan, so changing them re-fits the pattern
        #: under the crosshair without the ~2 s plan rebuild that the zone and
        #: in-plane steps would force. None means "whatever the plan was built
        #: with".
        #:
        #: ``pair_distance`` — how near a simulated reflection has to be to
        #: count as the same peak, Å⁻¹. Too small and real peaks go unpaired;
        #: too large and neighbouring reflections are claimed by one spot.
        #: ``sigma_excitation`` — how far off the Ewald sphere a reflection may
        #: sit and still be treated as excited, Å⁻¹.
        self.pair_distance: Optional[float] = None
        self.sigma_excitation: Optional[float] = None
        self._plan_pair_distance = float(
            plan_kwargs.get('corr_kernel_size', PAIR_DISTANCE))
        self._metadata = dict(getattr(peaks, "metadata", {}) or {})
        self._maps = [
            build_orientation_map(
                    peaks, crystal, energy_ev=energy_ev,
                k_max=k_max, angle_step_zone_axis_deg=angle_step_zone_axis_deg,
                angle_step_in_plane_deg=angle_step_in_plane_deg,
                device=device, **plan_kwargs)
            for crystal in crystals
        ]
        #: The untouched on-detector template fraction per plan, kept so a zone
        #: mask can be recomputed from scratch each time rather than composed
        #: onto an already-masked tensor.
        self._full_frac_shift = [
            None if m.plan_frac_shift is None else m.plan_frac_shift.clone()
            for m in self._maps
        ]

    def set_zone_mask(self, keep_per_phase) -> None:
        """Restrict the match to a subset of each phase's zone axes.

        ``keep_per_phase`` is one ``(Z,)`` boolean per phase, or None for a
        phase where every zone competes; ``None`` for the whole argument clears
        the restriction.

        This rides upstream's OWN suppression rather than reimplementing the
        match: ``match_orientations`` already zeroes the correlation of any
        (zone, in-plane angle) whose template mostly falls off the detector,
        by comparing ``plan_frac_shift`` against ``min_detector_fraction``. A
        masked-out zone is simply one with no template weight on the detector,
        so setting its fraction to zero makes the matcher skip it — exact, and
        it leaves the 240-line method alone.
        """
        import torch

        keep_per_phase = list(keep_per_phase or [None] * len(self._maps))
        for index, orientation_map in enumerate(self._maps):
            full = self._full_frac_shift[index]
            keep = keep_per_phase[index] if index < len(keep_per_phase) else None
            if full is None:
                # No detector aperture was measured, so upstream never consults
                # plan_frac_shift and there is nothing to ride.
                if keep is not None and not bool(np.all(keep)):
                    log.warning(
                        "zone mask ignored for phase %d: the plan has no "
                        "detector aperture correction to suppress through",
                        index)
                continue
            if keep is not None:
                keep = np.asarray(keep, bool).reshape(-1)
                if not keep.any():
                    # Every zone masked out leaves the argmax to pick
                    # arbitrarily among equal zeros, which would read as a
                    # random orientation rather than as "nothing selected".
                    log.debug("empty zone mask for phase %d ignored", index)
                    keep = None
                elif bool(keep.all()):
                    keep = None

            # A double-click arrives on its own thread while the navigator may
            # be correlating on this device — see spyde.device_lock. A null
            # context off MPS.
            with accelerator_lock(orientation_map.device):
                if keep is None:
                    orientation_map.plan_frac_shift = full.clone()
                else:
                    allowed = torch.as_tensor(
                        keep, device=full.device).to(full.dtype)      # (Z,)
                    orientation_map.plan_frac_shift = \
                        full * allowed[None, :, None]

    @property
    def zone_counts(self) -> list:
        """How many zone axes each phase samples — the mask's expected width."""
        return [int(m.zone_axes.shape[0]) for m in self._maps]

    def zone_correlations(self, rows, inverse_angstrom_factor: Optional[float] = None
                          ) -> Optional[list]:
        """How well every sampled orientation explains this pattern.

        One array of correlations per phase, in that phase's zone-axis order —
        the surface the matcher picks its answer off, rather than the single
        number it picked. That is what makes an IPF heat map worth looking at:
        a confident position is one bright spot, an ambiguous one has several,
        and a wrong phase is uniformly dim.

        This mirrors the correlation inside ``match_orientations``, which
        computes exactly this and then keeps only the best. Upstream has no
        entry point that returns it, and re-running the matcher would not help;
        so the contraction is repeated here, against the same plan.
        """
        import torch

        if inverse_angstrom_factor is None:
            inverse_angstrom_factor = self.inverse_angstrom_factor
        peaks = _OnePosition(_rows_to_peaks(rows, inverse_angstrom_factor),
                             metadata=self._metadata)
        if peaks.peak_counts[0] < self.min_peaks:
            return None

        out = []
        for orientation_map in self._maps:
            # NOT `orientation_map.peaks = peaks`. This runs inline on the
            # navigator thread while `fit` may be running on the overlay lane
            # against the SAME plans, and `fit` re-reads `self.peaks` at every
            # stage (match, refine, strain) — so assigning it here rebinds the
            # pattern out from under a fit in progress. Nothing here needs it:
            # `_polar_images` takes its arrays as an argument.
            # Every torch submission on this device is serialised, or none of
            # them are — see spyde.device_lock. A null context off MPS.
            with accelerator_lock(orientation_map.device):
                image = orientation_map._polar_images(
                    [peaks[0, 0].array], [0, 1, 2]
                ).to(orientation_map.dtype).to(orientation_map.device)
                norm = torch.linalg.norm(
                    image.reshape(1, -1), dim=1).clamp_min(1e-12)
                spectrum = torch.fft.fft(image, dim=-1)
                channels = [torch.einsum(
                    "zsg,bsg->bzg", orientation_map.plan_fft, spectrum)]
                channels.append(torch.einsum(
                    "zsg,bsg->bzg", orientation_map.plan_fft, torch.conj(spectrum)))
                correlation = torch.fft.ifft(
                    torch.stack(channels, dim=1), dim=-1
                ).real / norm[:, None, None, None]
                if orientation_map.plan_norm_shift is not None:
                    channels_n = correlation.shape[1]
                    correlation = correlation / orientation_map.plan_norm_shift[
                        None, :channels_n].clamp_min(1e-3)
                    # The same suppression match_orientations applies, so the
                    # triangle shows the surface the matcher actually chooses
                    # off — including any zone the user has masked out, which
                    # is set to zero weight on the detector.
                    correlation = correlation.masked_fill(
                        orientation_map.plan_frac_shift[None, :channels_n]
                        < MIN_DETECTOR_FRACTION, 0.0)
                # best over the mirror channel and the in-plane angle: what is
                # left is one number per zone axis, which the triangle shows.
                out.append(
                    correlation.amax(dim=(1, 3))[0].detach().cpu().numpy())
        return out

    def fit(self, rows, inverse_angstrom_factor: Optional[float] = None
            ) -> Optional[SinglePatternFit]:
        """Fit one position's vector rows, or None if there are too few peaks.

        ``rows`` is the ``(N, 6)`` flat-buffer slice the overlay already holds,
        in the detector's own units; the conversion to Å⁻¹ happens here.
        """
        import torch

        if inverse_angstrom_factor is None:
            inverse_angstrom_factor = self.inverse_angstrom_factor
        peaks = _OnePosition(_rows_to_peaks(rows, inverse_angstrom_factor),
                             metadata=self._metadata)
        if peaks.peak_counts[0] < self.min_peaks:
            return None

        best = None
        for index, orientation_map in enumerate(self._maps):
            orientation_map.peaks = peaks
            # This runs on the overlay lane, concurrently with whatever else is
            # submitting to the device, so it is serialised like every other
            # torch call site — see spyde.device_lock. Null context off MPS.
            with accelerator_lock(orientation_map.device):
                # min_detector_fraction is passed rather than left to default
                # because set_zone_mask masks BY it: a masked zone is one whose
                # on-detector fraction has been set to zero, so a caller that
                # lowered this to 0 would silently unmask everything.
                orientation_map.match_orientations(
                    progress_bar=False, min_number_peaks=self.min_peaks,
                    min_detector_fraction=MIN_DETECTOR_FRACTION)
                orientation_map.refine_orientations(
                    progress_bar=False, neighbor_rescue=False,
                    pair_distance=self.pair_distance,
                    sigma_excitation=self.sigma_excitation)
            correlation = float(orientation_map.corr[0, 0, 0])
            if best is None or correlation > best[0]:
                best = (correlation, index, orientation_map)
        correlation, phase_index, orientation_map = best
        if not np.isfinite(correlation) or correlation <= 0:
            return None

        with accelerator_lock(orientation_map.device):
            affine, pairs = reciprocal_affine(
                orientation_map, device=self.device,
                pair_distance=self.pair_distance,
                sigma_excitation=self.sigma_excitation)
            deformation = affine[0, 0]
            strain = _symmetric_strain(
                real_space_deformation(deformation[None]))[0].cpu().numpy()
            pattern = orientation_map.generate_pattern(0, 0)

        spots = torch.stack((pattern["qx"], pattern["qy"]), dim=1).to(torch.float64)
        intensity = pattern["intensity"]
        # A simulated pattern carries every reflection inside its excitation
        # tolerance, including ones no detector would register. Drawn unfiltered
        # they bury the ones that matter — upstream's own threshold for what is
        # observable is a fraction of the strongest reflection.
        if intensity.numel():
            observable = intensity >= MIN_SIM_INTENSITY_REL * intensity.max()
            spots, intensity = spots[observable], intensity[observable]
        if bool(torch.isfinite(deformation).all()):
            # Draw the simulated pattern where the FIT puts it, not where an
            # unstrained crystal would: the overlay is how a user judges the
            # fit, so its spots have to be the fit's own prediction.
            spots = spots @ deformation.to(spots.dtype).T

        drawn = spots.cpu().numpy().astype(np.float32)
        measured = peaks[0, 0].array[:, :2]
        if drawn.size and measured.size:
            nearest = np.linalg.norm(
                drawn[:, None, :] - measured[None, :, :], axis=-1).min(axis=0)
            residual = float(np.median(nearest))
            n_matched = int((nearest < (self.pair_distance
                                        or self._plan_pair_distance)).sum())
        else:
            residual, n_matched = float("nan"), 0

        return SinglePatternFit(
            quat=quantem_quats_to_orix(
                orientation_map.quats[0, 0, 0].cpu().numpy(), self.convention),
            phase_index=phase_index,
            correlation=correlation,
            reliability=float(orientation_map.reliability[0, 0]),
            strain=np.asarray(strain, np.float32),
            spots=drawn,
            intensities=intensity.cpu().numpy().astype(np.float32),
            residual=residual,
            n_matched=n_matched,
        )


def _rows_to_peaks(rows, inverse_angstrom_factor: float) -> np.ndarray:
    """``(N, 6)`` vector rows → the ``(N, 3)`` qx/qy/intensity their matcher
    reads, in Å⁻¹."""
    rows = np.asarray(rows)
    out = np.empty((len(rows), 3), dtype=np.float64)
    if len(rows):
        out[:, 0] = rows[:, COL_KX] * float(inverse_angstrom_factor)
        out[:, 1] = rows[:, COL_KY] * float(inverse_angstrom_factor)
        out[:, 2] = rows[:, COL_INTENSITY]
    return out


class _OnePosition:
    """The peak protocol over a single pattern — a scan of one position."""

    fields = list(PEAK_FIELDS)
    shape = (1, 1)

    def __init__(self, array: np.ndarray, metadata: Optional[dict] = None):
        self.metadata: dict = dict(metadata or {})
        self._cell = _Cell(np.asarray(array, dtype=np.float64))

    def __getitem__(self, index) -> _Cell:
        return self._cell

    def select_fields(self, *names) -> "_OnePosition":
        return self

    def flatten(self) -> np.ndarray:
        return self._cell.array

    @property
    def peak_counts(self) -> np.ndarray:
        return np.array([self._cell.array.shape[0]], dtype=int)
