"""Drive the vendored quantem ACOM matcher from SpyDE's vectors and phases.

Everything SpyDE-shaped lives here so that ``spyde/external/quantem`` stays a
mechanical copy of upstream: this module converts an orix ``Phase`` into the
``Crystal`` their library generation needs, presents a
:class:`~spyde.signals.diffraction_vectors.SpyDEDiffractionVectors` as the peak
container their matcher reads, and decodes the result into the same
``VectorOrientationResult`` our own fit returns.

Nothing here is wired into an action yet. It exists to be measured against
``vector_orientation_gpu`` on real data — see
``spyde/tests/benchmark_quantem_orientation.py``.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from spyde.actions.vector_orientation import VectorOrientationResult
from spyde.signals.diffraction_vectors import COL_KX, COL_KY, COL_INTENSITY

log = logging.getLogger(__name__)

#: Their matcher reads these three columns by name; the order is the order of
#: the columns in the arrays :class:`PeaksAdapter` hands back.
PEAK_FIELDS = ("qx", "qy", "intensity")


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
        batched = kwargs.get("batched", True)
        refine_tilt = kwargs.get("refine_tilt", False)
        # Falling back leaves upstream wholly in charge, rescue included.
        if (not self._gpu_refine_active() or not rescue or not batched
                or refine_tilt or len(args) > 10):
            return super().refine_orientations(
                *args, neighbor_rescue=rescue, **kwargs)

        self._refine_state = None
        result = super().refine_orientations(*args, neighbor_rescue=False, **kwargs)
        if self._refine_state is not None:
            self._rescue_batched(float(kwargs.get("rescue_threshold_deg", 2.0)))
            self.metadata["refine"]["neighbor_rescue"] = True
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
    if crystal.g_vec is None:
        crystal.calculate_structure_factors(k_max=k_max)
    map_class = orientation_map_class(refine_device, refine_chunk, refine_dtype)
    orientation_map = map_class.from_vectors(peaks, crystal, energy_ev=energy_ev)
    orientation_map.build_plan(
        angle_step_zone_axis_deg=angle_step_zone_axis_deg,
        angle_step_in_plane_deg=angle_step_in_plane_deg,
        device=device, verbose=verbose, **plan_kwargs)
    return orientation_map


def to_result(orientation_map, phases_meta: list,
              convention: str = "conjugate",
              params: Optional[dict] = None) -> VectorOrientationResult:
    """Decode a matched ``OrientationMap`` into our result container.

    Two fields have no counterpart on their side and stay NaN:

    ``strain``
        Their orientation refinement deliberately does not fit strain — peak
        positions carry almost no out-of-plane information, so they refine the
        pose and solve the deformation afterwards in ``calculate_strain``,
        which is not vendored.
    ``friedel_asym``
        Our own diagnostic, computed from the pose residual of ±g pairs.

    ``coarse_score`` carries their correlation, which unlike ours is a
    normalised cosine similarity in [0, 1] and so is comparable across
    patterns. Their ``reliability`` (best minus best-outside-an-exclusion-ball)
    has no field on this container and is returned separately by the caller.
    """
    quats = quantem_quats_to_orix(
        orientation_map.quats[..., 0, :].cpu().numpy(), convention)
    ny, nx = quats.shape[:2]
    corr = orientation_map.corr[..., 0].cpu().numpy().astype(np.float32)
    theta = np.deg2rad(
        orientation_map.in_plane_angle_deg().cpu().numpy()).astype(np.float32)
    valid = corr > 0

    n_matched = np.zeros((ny, nx), np.int16)
    counts = orientation_map.peaks.peak_counts.reshape(ny, nx)
    n_matched[:] = np.clip(counts, 0, np.iinfo(np.int16).max)

    return VectorOrientationResult(
        quats=quats,
        phase_idx=np.zeros((ny, nx), np.int16),
        theta=theta,
        strain=np.full((ny, nx, 3), np.nan, np.float32),
        residual=np.full((ny, nx), np.nan, np.float32),
        friedel_asym=np.full((ny, nx), np.nan, np.float32),
        n_matched=n_matched,
        coarse_score=np.where(valid, corr, 0.0).astype(np.float32),
        phases_meta=phases_meta,
        nav_shape=(ny, nx),
        params=dict(params or {}),
    )
