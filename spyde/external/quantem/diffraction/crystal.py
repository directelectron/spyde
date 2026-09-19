"""Crystal structures and kinematical diffraction for orientation mapping.

A Crystal wraps an ase.Atoms object and computes the reciprocal lattice,
kinematical structure factors, symmetry operators (via spglib), and simulated
diffraction patterns for arbitrary orientations. All numerical state is stored
as torch tensors (float64) so downstream matching and refinement can run on
GPU and differentiate through the calculation.

Conventions
-----------
- Real lattice vectors are rows of `lat_real` (Angstroms).
- Reciprocal lattice vectors are rows of `lat_recip` (1/Angstroms, no 2*pi).
- Structure factors follow F_hkl = (1/V) * sum_n f_n * exp(-2*pi*i * hkl.p_n),
  so intensities have units of scattering amplitude per unit volume.
- Orientations are unit quaternions rotating crystal Cartesian vectors into
  the lab frame (see quantem.diffraction.rotations).
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols

from .defaults import SIGMA_EXCITATION
from .rotations import qrotate, symmetry_quaternions

# unicode combining overline, applies to the preceding character
_B = "\u0305"


def direction_indices(
    lat_real: torch.Tensor | np.ndarray, d, max_multiple: int = 12
) -> np.ndarray | None:
    """Smallest integer [uvw] along a Cartesian direction, or None if the
    direction is not a lattice direction with indices up to max_multiple."""
    A_T_inv = np.linalg.inv(np.asarray(lat_real, dtype=float).T)
    v = A_T_inv @ np.asarray(d, dtype=float)
    v = v / np.abs(v).max()
    for m in range(1, max_multiple + 1):
        w = v * m
        if np.allclose(w, np.round(w), atol=2e-3):
            ints = np.round(w).astype(int)
            g = np.gcd.reduce(np.abs(ints))
            return ints // max(g, 1)
    return None


def format_direction(uvw, hexagonal: bool = False, mathtext: bool = True) -> str:
    """Direction label such as [011] or [10-10], with overlines on negative
    indices (mathtext for figures, combining overlines for text)."""
    if uvw is None:
        return ""
    ks = miller_to_miller_bravais(uvw) if hexagonal else np.asarray(uvw)
    ks = np.atleast_1d(ks)
    if mathtext:
        body = "".join(str(k) if k >= 0 else "$\\bar{%d}$" % -k for k in ks)
    else:
        body = "".join(str(k) if k >= 0 else "%d%s" % (-k, _B) for k in ks)
    return "[" + body + "]"


def miller_to_miller_bravais(uvw: np.ndarray) -> np.ndarray:
    """Convert 3-index [u'v'w'] direction indices to 4-index [u v t w].

    u = (2u' - v') / 3, v = (2v' - u') / 3, t = -(u + v), w = w', cleared to
    the smallest integer form.
    """
    uvw = np.atleast_2d(np.asarray(uvw, dtype=float))
    u = (2 * uvw[:, 0] - uvw[:, 1]) / 3
    v = (2 * uvw[:, 1] - uvw[:, 0]) / 3
    out = np.stack([u, v, -(u + v), uvw[:, 2]], axis=1)
    # clear fractions and common factors
    out = out * 3
    gcd = np.gcd.reduce(np.abs(np.round(out)).astype(int), axis=1)
    gcd[gcd == 0] = 1
    out = out / gcd[:, None]
    return out.astype(int).squeeze()


def miller_bravais_to_miller(uvtw: np.ndarray) -> np.ndarray:
    """Convert 4-index [u v t w] direction indices to 3-index [u'v'w'].

    u' = 2u + v, v' = 2v + u, w' = w (t is redundant: t = -(u + v)).
    """
    uvtw = np.atleast_2d(np.asarray(uvtw, dtype=float))
    out = np.stack([2 * uvtw[:, 0] + uvtw[:, 1], 2 * uvtw[:, 1] + uvtw[:, 0], uvtw[:, 3]], axis=1)
    gcd = np.gcd.reduce(np.abs(np.round(out)).astype(int), axis=1)
    gcd[gcd == 0] = 1
    return (out / gcd[:, None]).astype(int).squeeze()


# point group -> Laue class
_LAUE_CLASS = {
    "1": "-1",
    "-1": "-1",
    "2": "2/m",
    "m": "2/m",
    "2/m": "2/m",
    "222": "mmm",
    "mm2": "mmm",
    "mmm": "mmm",
    "4": "4/m",
    "-4": "4/m",
    "4/m": "4/m",
    "422": "4/mmm",
    "4mm": "4/mmm",
    "-42m": "4/mmm",
    "4/mmm": "4/mmm",
    "3": "-3",
    "-3": "-3",
    "32": "-3m",
    "3m": "-3m",
    "-3m": "-3m",
    "6": "6/m",
    "-6": "6/m",
    "6/m": "6/m",
    "622": "6/mmm",
    "6mm": "6/mmm",
    "-6m2": "6/mmm",
    "6/mmm": "6/mmm",
    "23": "m-3",
    "m-3": "m-3",
    "432": "m-3m",
    "-43m": "m-3m",
    "m-3m": "m-3m",
}


def _load_lobato_params() -> dict[str, np.ndarray]:
    with resources.files(__package__).joinpath("data/lobato.json").open() as f:
        raw = json.load(f)
    return {sym: np.array(p) for sym, p in raw.items()}


_LOBATO: dict[str, np.ndarray] | None = None


def electron_scattering_factor(numbers: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Lobato & Van Dyck (2014) electron scattering factors.

    Parameters
    ----------
    numbers : torch.Tensor
        Atomic numbers (N,).
    g : torch.Tensor
        Scattering vector magnitudes (M,) in 1/Angstroms.

    Returns
    -------
    torch.Tensor
        f_e(g) of shape (N, M) in Angstroms.
    """
    global _LOBATO
    if _LOBATO is None:
        _LOBATO = _load_lobato_params()
    g2 = (g**2)[None, :, None]  # (1, M, 5)
    a = torch.stack(
        [
            torch.as_tensor(_LOBATO[chemical_symbols[int(z)]][0], dtype=g.dtype, device=g.device)
            for z in numbers
        ]
    )[:, None, :]  # (N, 1, 5)
    b = torch.stack(
        [
            torch.as_tensor(_LOBATO[chemical_symbols[int(z)]][1], dtype=g.dtype, device=g.device)
            for z in numbers
        ]
    )[:, None, :]
    return (a * (2.0 + b * g2) / (1.0 + b * g2) ** 2).sum(dim=-1)


class Crystal:
    """A crystal structure with kinematical diffraction methods.

    Build with `from_ase` or `from_cif`, then call
    `calculate_structure_factors` before generating patterns or orientation
    plans.
    """

    def __init__(
        self,
        atoms: Atoms,
        name: str | None = None,
        symprec: float = 1e-4,
        pseudo_symmetry_tol: float | None = 0.01,
        pseudo_symmetry_intensity_tol: float = 0.05,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        symprec : float, default=1e-4
            spglib tolerance (Angstroms) for the cell's own symmetry.
        pseudo_symmetry_tol : float | None, default=0.01
            Dimensionless distance tolerance for the symmetry used in
            orientation matching: a fraction of the shortest lattice vector
            within which atoms and lattice vectors are allowed to deviate
            from a higher-symmetry parent (a 4 A cell with an atom at
            (0.5, 0.5, 0.50001) is body centered at any tolerance above
            1e-5). Cells within it are matched with the parent group, so
            variants no experiment can separate are never sampled as
            distinct orientations; the library builders warn when the
            matching group differs from the cell's own. None matches with
            the exact symmetry.
        pseudo_symmetry_intensity_tol : float, default=0.05
            Dimensionless intensity tolerance of the same decision: the
            extra operations of the parent group must map the kinematical
            intensities of the reflections they relate onto each other
            within this fraction of the strongest reflection, otherwise the
            patterns are distinguishable and the parent group is rejected.
        """
        self.atoms = atoms
        self.name = name if name is not None else atoms.get_chemical_formula()
        self._pseudo_symmetry_tol = pseudo_symmetry_tol
        self._pseudo_symmetry_intensity_tol = float(pseudo_symmetry_intensity_tol)
        self.pseudo_symmetry_report: dict = {}
        self._wedge_cache: torch.Tensor | None | str = "unset"

        self.lat_real = torch.as_tensor(atoms.cell[:], dtype=torch.float64)
        self.positions_frac = torch.as_tensor(atoms.get_scaled_positions(), dtype=torch.float64)
        self.numbers = torch.as_tensor(atoms.numbers, dtype=torch.long)
        occupancy = atoms.arrays.get("occupancy", np.ones(len(atoms)))
        self.occupancy = torch.as_tensor(np.asarray(occupancy, dtype=float))

        self._setup_symmetry(symprec, pseudo_symmetry_tol, pseudo_symmetry_intensity_tol)
        if verbose:
            print(self.symmetry_summary())

        # populated by calculate_structure_factors
        self.k_max: float | None = None
        self.hkl: torch.Tensor | None = None
        self.g_vec: torch.Tensor | None = None
        self.g_len: torch.Tensor | None = None
        self.struct_factors: torch.Tensor | None = None
        self.struct_factors_int: torch.Tensor | None = None

    @classmethod
    def from_ase(cls, atoms: Atoms, name: str | None = None, **kwargs) -> "Crystal":
        return cls(atoms, name=name, **kwargs)

    @classmethod
    def from_cif(cls, file_path: str | Path, name: str | None = None, **kwargs) -> "Crystal":
        from ase.io import read

        atoms = read(file_path)
        assert isinstance(atoms, Atoms)
        return cls(atoms, name=name, **kwargs)

    @property
    def volume(self) -> float:
        return float(torch.abs(torch.linalg.det(self.lat_real)))

    @property
    def lat_recip(self) -> torch.Tensor:
        """Reciprocal lattice vectors as rows, no 2*pi factor."""
        return torch.linalg.inv(self.lat_real).T

    def _quick_intensities(self, k_max: float = 1.2) -> tuple[torch.Tensor, torch.Tensor]:
        """Kinematical |F|^2 of every reflection with |g| <= k_max (hkl, I),
        for the pseudo-symmetry intensity check; no thermal factors."""
        recip = self.lat_recip
        k_len = torch.linalg.norm(recip, dim=1)
        n_max = torch.ceil(k_max / k_len * 2).to(torch.long)
        ranges = [torch.arange(-int(n), int(n) + 1) for n in n_max]
        hkl = torch.cartesian_prod(*ranges).to(torch.float64)
        g_vec = hkl @ recip
        g_len = torch.linalg.norm(g_vec, dim=1)
        keep = (g_len <= k_max) & (g_len > 0)
        hkl, g_len = hkl[keep], g_len[keep]
        f_e = electron_scattering_factor(self.numbers, g_len)
        phase = torch.exp(-2j * np.pi * (self.positions_frac @ hkl.T))
        F = (f_e * self.occupancy[:, None] * phase).sum(dim=0) / self.volume
        return hkl.to(torch.long), torch.abs(F) ** 2

    def _setup_symmetry(
        self, symprec: float, pseudo_symmetry_tol: float | None, intensity_tol: float
    ) -> None:
        """Detect the true symmetry group, and optionally a pseudo-symmetry group.

        The true group (at `symprec`) is stored for reporting and refinement.
        The pseudo-symmetry group is detected at a distance tolerance of
        `pseudo_symmetry_tol` times the shortest lattice vector and kept
        only if its extra operations relate reflections of equal kinematical
        intensity to within `intensity_tol` of the strongest reflection:
        two orientations are merged only when no experiment could tell
        their patterns apart, in position or in intensity. Matching uses
        that group, so nearly-degenerate cells are idealized to their
        higher-symmetry parent.
        """
        import spglib

        from .rotations import quat_to_matrix

        cell = (
            self.lat_real.numpy(),
            self.positions_frac.numpy(),
            self.numbers.numpy(),
        )
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        self.spacegroup: str = f"{dataset.international} ({dataset.number})"
        pg = spglib.get_pointgroup(dataset.rotations)[0].strip()
        self.pointgroup: str = pg
        self.laue_group: str = _LAUE_CLASS.get(pg, "-1")
        self.sym_quats = symmetry_quaternions(dataset.rotations, self.lat_real.numpy())

        self.pointgroup_matching = pg
        self.laue_group_matching = self.laue_group
        self.sym_quats_matching = self.sym_quats
        if pseudo_symmetry_tol is None:
            return
        a_min = float(torch.linalg.norm(self.lat_real, dim=1).min())
        symprec_pseudo = float(pseudo_symmetry_tol) * a_min
        self.pseudo_symmetry_report = {"distance_A": symprec_pseudo}
        if symprec_pseudo <= symprec:
            return
        try:
            ds_pseudo = spglib.get_symmetry_dataset(cell, symprec=symprec_pseudo)
        except Exception:
            ds_pseudo = None
        if ds_pseudo is None:
            return
        pg_pseudo = spglib.get_pointgroup(ds_pseudo.rotations)[0].strip()
        quats_pseudo = symmetry_quaternions(ds_pseudo.rotations, self.lat_real.numpy())
        if quats_pseudo.shape[0] <= self.sym_quats.shape[0]:
            return

        # intensity check on the extra operations: |F|^2 of every reflection
        # against |F|^2 of its image, relative to the strongest reflection
        hkl, inten = self._quick_intensities()
        lut = {tuple(h): i for i, h in enumerate(hkl.tolist())}
        g = hkl.to(torch.float64) @ self.lat_recip
        i_max = float(inten.max())
        Rs = quat_to_matrix(quats_pseudo)
        Rs_true = quat_to_matrix(self.sym_quats)
        worst = 0.0
        for R in Rs:
            if any(float((R - Rt).abs().max()) < 1e-6 for Rt in Rs_true):
                continue
            g_img = g @ R.T
            hkl_img = torch.round(g_img @ self.lat_real.T).to(torch.long)
            idx = torch.tensor([lut.get(tuple(h), -1) for h in hkl_img.tolist()])
            ok = idx >= 0
            diff = (inten[ok] - inten[idx[ok]]).abs() / i_max
            worst = max(worst, float(diff.max()) if ok.any() else 0.0)
        self.pseudo_symmetry_report["intensity_mismatch"] = worst
        self.pseudo_symmetry_report["candidate"] = pg_pseudo
        if worst > intensity_tol:
            self.pseudo_symmetry_report["rejected"] = True
            return
        self.pointgroup_matching = pg_pseudo
        self.laue_group_matching = _LAUE_CLASS.get(pg_pseudo, "-1")
        self.sym_quats_matching = quats_pseudo

    def zone_axis_wedge(self) -> torch.Tensor | None:
        """Fundamental zone-axis wedge corners (3, 3) Cartesian, or None.

        Built from the symmetry operations actually used for matching (the
        pseudo-symmetry group when one was found), so the wedge is right
        for every crystal setting. None means the Laue class (-1 or 2/m)
        has no 3-corner wedge and libraries sample the full hemisphere.
        """
        if isinstance(self._wedge_cache, str):
            from .rotations import fundamental_zone_axis_wedge

            self._wedge_cache = fundamental_zone_axis_wedge(self.sym_quats_matching)
        return self._wedge_cache

    @property
    def hexagonal_matching(self) -> bool:
        """Whether the matching Laue class uses 4-index direction symbols."""
        return self.laue_group_matching in ("6/m", "6/mmm", "-3", "-3m")

    def zone_axis_wedge_labels(self, mathtext: bool = True) -> list[str] | None:
        """Direction labels of the wedge corners (4-index for hexagonal and
        trigonal crystals), indexed from the corner directions themselves."""
        corners = self.zone_axis_wedge()
        if corners is None:
            return None
        return [
            format_direction(
                direction_indices(self.lat_real, c.numpy()),
                hexagonal=self.hexagonal_matching,
                mathtext=mathtext,
            )
            for c in corners
        ]

    def matching_symmetry_warning(self) -> str | None:
        """Message when the matching (pseudo) symmetry differs from the
        cell's own symmetry, or None when they agree."""
        if self.pointgroup_matching == self.pointgroup:
            return None
        n_extra = self.sym_quats_matching.shape[0] // max(self.sym_quats.shape[0], 1)
        return (
            f"{self.name}: orientation libraries are built with the "
            f"pseudo-symmetry point group {self.pointgroup_matching} (Laue "
            f"class {self.laue_group_matching}, found at pseudo_symmetry_tol = "
            f"{self._pseudo_symmetry_tol:g} of the shortest lattice vector, "
            f"intensities matching within {self.pseudo_symmetry_report.get('intensity_mismatch', 0.0):.1%}), "
            f"while the cell's own symmetry "
            f"is {self.pointgroup} (Laue class {self.laue_group}). Orientations "
            f"related by the extra operations give the same library entry, so "
            f"the {n_extra} variants they generate are reported as one and the "
            f"distortion between them is not resolved. To match with the exact "
            f"symmetry, build the Crystal with pseudo_symmetry_tol=None (or a "
            f"tolerance below the distortion)."
        )

    def symmetry_summary(self) -> str:
        """Human-readable symmetry report, including any pseudo-symmetry."""
        import re

        # subscript the space group screw/glide digits: P6_3/mmc -> P6[sub3]/mmc
        subs = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        sg = re.sub(r"_(\d)", lambda m: m.group(1).translate(subs), self.spacegroup)
        lines = [
            f"{self.name}",
            f"  space group      {sg}",
            f"  point group      {self.pointgroup}   (Laue class {self.laue_group})",
        ]
        if self.pointgroup_matching != self.pointgroup:
            lines += [
                f"  pseudo-symmetry  {self.pointgroup_matching} "
                f"(Laue class {self.laue_group_matching}) "
                "-- used for orientation matching",
            ]
        elif self._pseudo_symmetry_tol is not None:
            rep = self.pseudo_symmetry_report
            if rep.get("rejected"):
                lines += [
                    f"  pseudo-symmetry  {rep['candidate']} within {self._pseudo_symmetry_tol:g} of "
                    f"the lattice, rejected: intensities differ by "
                    f"{rep['intensity_mismatch']:.1%} (tol {self._pseudo_symmetry_intensity_tol:.0%})",
                ]
            else:
                lines += [
                    f"  pseudo-symmetry  none found at tol = {self._pseudo_symmetry_tol:g} "
                    f"({rep.get('distance_A', 0.0):.3f} A)",
                ]
        else:
            lines += ["  pseudo-symmetry  not checked (set pseudo_symmetry_tol)"]
        # matching line reflects the symmetry actually used, after any
        # pseudo-symmetry reduction
        labels = self.zone_axis_wedge_labels(mathtext=False)
        wedge_txt = (
            f"zone axis wedge {labels[0]}, {labels[1]}, {labels[2]}"
            if labels is not None
            else "full hemisphere"
        )
        lines += [
            f"  matching         {self.sym_quats_matching.shape[0]} proper rotations, {wedge_txt}"
        ]
        return "\n".join(lines)

    def calculate_structure_factors(
        self,
        k_max: float = 1.5,
        tol_structure_factor: float = 1e-4,
        thermal_sigma: float | dict[str, float] | None = None,
    ) -> "Crystal":
        """Kinematical structure factors for all reflections with |g| <= k_max.

        Parameters
        ----------
        k_max : float, default=1.5
            Maximum scattering vector magnitude, 1/Angstroms.
        tol_structure_factor : float, default=1e-4
            Discard reflections with |F| below this threshold.
        thermal_sigma : float | dict[str, float] | None
            RMS thermal displacement (Angstroms), scalar or per-element,
            applied as a Debye-Waller factor.

        Returns
        -------
        Crystal
            self, for chaining.
        """
        self.k_max = float(k_max)
        recip = self.lat_recip

        # index range: project k_max onto each reciprocal cell direction
        k_len = torch.linalg.norm(recip, dim=1)
        n_max = torch.ceil(k_max / k_len * 2).to(torch.long)
        ranges = [torch.arange(-int(n), int(n) + 1) for n in n_max]
        hkl = torch.cartesian_prod(*ranges).to(torch.float64)
        g_vec = hkl @ recip
        g_len = torch.linalg.norm(g_vec, dim=1)
        keep = (g_len <= k_max) & (g_len > 0)
        hkl, g_vec, g_len = hkl[keep], g_vec[keep], g_len[keep]

        f_e = electron_scattering_factor(self.numbers, g_len)  # (N_atoms, N_g)

        if thermal_sigma is not None:
            if isinstance(thermal_sigma, dict):
                sigma = torch.tensor(
                    [thermal_sigma[chemical_symbols[int(z)]] for z in self.numbers],
                    dtype=torch.float64,
                )
            else:
                sigma = torch.full((len(self.numbers),), float(thermal_sigma))
            dwf = torch.exp(-0.5 * (2 * np.pi * sigma[:, None] * g_len[None, :]) ** 2)
            f_e = f_e * dwf

        phase = torch.exp(-2j * np.pi * (self.positions_frac @ hkl.T))  # (N_atoms, N_g)
        F = (f_e * self.occupancy[:, None] * phase).sum(dim=0) / self.volume

        keep = torch.abs(F) > tol_structure_factor
        self.hkl = hkl[keep].to(torch.long)
        self.g_vec = g_vec[keep]
        self.g_len = g_len[keep]
        self.struct_factors = F[keep]
        self.struct_factors_int = torch.abs(F[keep]) ** 2
        return self

    def calculate_dynamical_structure_factors(
        self,
        energy_ev: float,
        thermal_sigma: float | dict[str, float] = 0.05,
        k_max: float | None = None,
        include_core: bool = True,
        include_phonon: bool = True,
    ) -> "Crystal":
        """Absorptive structure factors for Bloch wave calculations.

        Uses the Weickenmeier-Kohl parameterization (Acta Cryst. A47, 590
        (1991)): the elastic part is Debye-Waller damped, and the imaginary
        (absorptive) part includes core-loss and phonon/TDS contributions.
        The returned factors are relativistically corrected and already carry
        the 1/pi convention of the Bloch structure matrix, i.e. they are the
        U_g of De Graef ch. 5 after division by the unit cell volume.

        All reflections up to k_max are kept, including kinematically
        forbidden ones (their U_g can be nonzero through absorption and they
        are required as coupling vectors g - h).

        Parameters
        ----------
        energy_ev : float
            Beam energy in eV.
        thermal_sigma : float | dict[str, float], default=0.05
            RMS thermal displacement (Angstroms), scalar or per-element.
        k_max : float | None
            Maximum |g| of stored factors; defaults to the kinematical k_max.
            For Bloch calculations with beams out to k, this should be 2k so
            every coupling vector is covered.
        """
        from .wk_scattering_factors import compute_WK_factor

        if k_max is None:
            if self.k_max is None:
                raise RuntimeError("Provide k_max or run calculate_structure_factors.")
            k_max = self.k_max
        recip = self.lat_recip
        k_len = torch.linalg.norm(recip, dim=1)
        n_max = torch.ceil(k_max / k_len * 2).to(torch.long)
        ranges = [torch.arange(-int(n), int(n) + 1) for n in n_max]
        hkl = torch.cartesian_prod(*ranges).to(torch.float64)
        g_vec = hkl @ recip
        g_len = torch.linalg.norm(g_vec, dim=1)
        keep = g_len <= k_max
        hkl, g_len = hkl[keep], g_len[keep]

        g_np = g_len.numpy()
        if isinstance(thermal_sigma, dict):
            sigma_per_atom = np.array(
                [thermal_sigma[chemical_symbols[int(z)]] for z in self.numbers]
            )
        else:
            sigma_per_atom = np.full(len(self.numbers), float(thermal_sigma))

        # one WK evaluation per unique (Z, sigma) pair
        f_atoms = np.zeros((len(self.numbers), g_np.size), dtype=np.complex128)
        cache: dict[tuple[int, float], np.ndarray] = {}
        for i, (z, sig) in enumerate(zip(self.numbers.tolist(), sigma_per_atom)):
            key = (int(z), float(sig))
            if key not in cache:
                cache[key] = compute_WK_factor(
                    g_np,
                    int(z),
                    energy_ev,
                    thermal_sigma=float(sig),
                    include_core=include_core,
                    include_phonon=include_phonon,
                )
            f_atoms[i] = cache[key]

        phase = np.exp(-2j * np.pi * (self.positions_frac.numpy() @ hkl.numpy().T))
        occ = self.occupancy.numpy()[:, None]
        U = (f_atoms * occ * phase).sum(axis=0) / self.volume

        self.hkl_dyn = hkl.to(torch.long)
        self.g_len_dyn = g_len
        self.U_dyn = torch.as_tensor(U, dtype=torch.complex128)
        self.dyn_energy_ev = float(energy_ev)
        self.dyn_k_max = float(k_max)
        return self

    def generate_pattern(
        self,
        orientation: torch.Tensor,
        energy_ev: float = 300e3,
        sigma_excitation: float = SIGMA_EXCITATION,
        tol_excitation_mult: float = 3.0,
        k_max: float | None = None,
        precession_deg: float = 0.0,
        semiconv_mrad: float = 0.0,
        excitation_model: str = "gaussian",
        thickness_A: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Kinematical diffraction pattern for one orientation.

        The intensity of each reflection is |F_g|^2 times a Gaussian
        excitation envelope of width sigma_excitation, averaged exactly over
        the illumination when a precession angle or a convergence
        semiangle is given (quantem.diffraction.illumination): the
        precession ring sweeps the excitation error of reflection g by
        +- a_g = r |g_xy| / |K - g_z| about its central value c_g, and the
        averaged envelope is the Bessel transform G(c_g, a_g, b_g; sigma).

        Parameters
        ----------
        orientation : torch.Tensor
            Unit quaternion (4,) rotating crystal vectors into the lab frame.
        energy_ev : float, default=300e3
            Beam energy in eV.
        sigma_excitation : float, default=0.02
            Excitation error tolerance (1/Angstroms) in the shape-factor
            envelope exp(-s_g^2 / 2 sigma^2).
        tol_excitation_mult : float, default=3.0
            Include reflections with |s_g| below this multiple of sigma.
        k_max : float | None
            Optionally trim the pattern below the structure-factor k_max.
        precession_deg, semiconv_mrad : float
            Precession semi-angle (degrees) and convergence semiangle
            (mrad) of the illumination the intensities are averaged over.
        excitation_model : {"gaussian", "slab"}
            "gaussian" is the empirical envelope of width sigma_excitation
            used by the orientation library. "slab" is the finite-thickness
            first Born rocking curve, (pi |U_g| z / k0)^2 sinc(s_g z)^2
            with U_g = gamma_rel F_g / pi, averaged over the illumination
            the same way; it needs thickness_A and is the kinematical limit
            of the Bloch wave calculation for thin crystals.
        thickness_A : float | None
            Thickness for the slab model (Angstroms).

        Returns
        -------
        dict with 'qx', 'qy', 'intensity', 'hkl', 's_g' (the central
        excitation error), 'a' and 'b' (ring and disk sweep amplitudes).
        """
        if self.g_vec is None:
            raise RuntimeError("Run calculate_structure_factors first.")
        from .illumination import (
            averaged_gaussian_intensity_envelope,
            excitation_coefficients,
            slab_envelope,
        )

        g = qrotate(orientation, self.g_vec)
        if excitation_model == "slab":
            if thickness_A is None:
                raise ValueError("the slab excitation model needs thickness_A")
            c, a, b = excitation_coefficients(g, energy_ev, precession_deg, semiconv_mrad)
            # the sinc^2 tails are algebraic: keep everything whose main
            # lobe (width 1/z) plus illumination sweep is within the tolerance
            width = tol_excitation_mult / float(thickness_A)
            c_t = torch.as_tensor(c, dtype=torch.float64)
            a_t = torch.as_tensor(a, dtype=torch.float64)
            b_t = torch.as_tensor(b, dtype=torch.float64)
            keep = torch.abs(c_t) < a_t + b_t + width
            if k_max is not None:
                keep &= self.g_len <= k_max
            env = slab_envelope(
                c[keep.numpy()], a[keep.numpy()], b[keep.numpy()], float(thickness_A)
            )
            from .._compat import electron_wavelength_angstrom

            lam = electron_wavelength_angstrom(energy_ev)
            gamma_rel = 1.0 + float(energy_ev) / 510998.95
            u_abs = torch.abs(self.struct_factors[keep]) * (gamma_rel / np.pi)
            intensity = (np.pi * u_abs * float(thickness_A) * lam) ** 2 * torch.as_tensor(
                env, dtype=torch.float64
            )
        else:
            env, c, a, b = averaged_gaussian_intensity_envelope(
                g, energy_ev, sigma_excitation, precession_deg, semiconv_mrad
            )
            c_t = torch.as_tensor(c, dtype=torch.float64)
            a_t = torch.as_tensor(a, dtype=torch.float64)
            b_t = torch.as_tensor(b, dtype=torch.float64)
            # the full illumination support enters the selection, not only
            # the central excitation error
            keep = torch.abs(c_t) < a_t + b_t + sigma_excitation * tol_excitation_mult
            if k_max is not None:
                keep &= self.g_len <= k_max
            intensity = (
                self.struct_factors_int[keep] * torch.as_tensor(env, dtype=torch.float64)[keep]
            )
        return {
            "qx": g[keep, 0],
            "qy": g[keep, 1],
            "intensity": intensity,
            "hkl": self.hkl[keep],
            "s_g": c_t[keep],
            "a": a_t[keep],
            "b": b_t[keep],
        }

    def __repr__(self) -> str:
        return (
            f"Crystal({self.name}, {len(self.numbers)} atoms, "
            f"spacegroup {self.spacegroup}, pointgroup {self.pointgroup})"
        )
