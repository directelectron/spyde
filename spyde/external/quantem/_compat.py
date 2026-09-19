"""The three ``quantem.core`` names the vendored diffraction code imports.

Vendoring ``quantem.core`` as well would pull its dataset containers, zarr
serialisation and plotting for the sake of one formula and one base class. The
import rewrites in ``_sync.py`` point at this module instead.
"""
from __future__ import annotations

import math


class AutoSerialize:
    """Upstream's zarr-backed save/load base class, reduced to nothing.

    It contributes only serialisation, which SpyDE does through its own signal
    tree and result containers. Keeping the name means the vendored class
    statements stay untouched.
    """


def electron_wavelength_angstrom(E_eV: float) -> float:
    """Relativistic electron wavelength in Angstroms. Verbatim from upstream —
    it sets the Ewald sphere curvature that every excitation error is measured
    against, so it must agree with the code it was vendored alongside."""
    m = 9.109383 * 10**-31
    e = 1.602177 * 10**-19
    c = 299792458
    h = 6.62607 * 10**-34

    lam = h / math.sqrt(2 * m * e * E_eV) / math.sqrt(1 + e * E_eV / 2 / m / c**2) * 10**10
    return lam


class Vector:
    """Upstream's ragged peak container — constructors only, and unsupported.

    ``OrientationMap`` reads its peaks through a small read-only protocol
    (``.shape``, ``.fields``, ``.metadata``, ``peaks[row, column].array`` and
    ``.select_fields(...).flatten()``), which the adapter satisfies directly
    over SpyDE's CSR vector buffer with no copy. Only ``match_residual`` builds
    a *new* container, so only that path lands here.
    """

    _UNSUPPORTED = (
        "quantem's Vector container is not vendored. Only OrientationMap."
        "match_residual constructs one; SpyDE passes its own peaks adapter "
        "instead (spyde/actions/vector_orientation_quantem.py)."
    )

    @classmethod
    def from_shape(cls, *args, **kwargs):
        raise NotImplementedError(cls._UNSUPPORTED)

    @classmethod
    def from_data(cls, *args, **kwargs):
        raise NotImplementedError(cls._UNSUPPORTED)
