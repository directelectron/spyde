"""
DenseDiffractionVectors — a HyperSpy signal type that *stores* a diffraction-
vectors result as a dense flat buffer so it round-trips through ``.zspy``/``.hspy``.

Motivation
----------
pyxem's ``DiffractionVectors2D`` is a *ragged* (object-array) signal — one
variable-length ``(N_i, 2)`` array per navigation position. Ragged arrays
serialize poorly (object dtype, no chunking, slow save/load). This is the dense
alternative: the **whole scan's vectors live in one contiguous ``(N_total, 6)``
float32 buffer** — the same CSR flat buffer ``SpyDEDiffractionVectors`` keeps in
memory — and the navigation grouping is carried by a row-pointer (offsets) array
in metadata. No padding (unlike a ``(ny, nx, max_n, 6)`` dense block), no object
dtype, exact mirror of the in-memory structure.

Layout
------
``.data``                : (N_total, 6) float32 — the flat buffer
                           columns [nav_x, nav_y, kx, ky, time, intensity]
                           (see ``spyde.signals.diffraction_vectors`` COL_*).
``metadata.SpyDE.DiffractionVectors``:
    nav_offsets        : list of int64 row-pointer arrays (outermost-first CSR;
                         only the innermost is strictly needed — the rest rebuild
                         from the flat buffer, but storing them is cheap and exact).
    full_nav_shape     : list[int] — all nav dims outermost-first.
    sig_shape          : list[int] — signal (detector) shape.
    sig_axes           : list[dict(scale, offset, size, units, name)] — calibration.
    kernel_radius_px   : float — detection kernel radius in pixels.
    kernel_radius_data : float — same in calibrated units.
    params             : dict — the find-vectors parameters used.

The ``signal_type`` (``spyde_dense_diffraction_vectors``) is registered as a
HyperSpy extension (``spyde/hyperspy_extension.yaml``) so ``set_signal_type`` and
``hs.load`` round-trip the class. On load, :func:`from_spyde_vectors` /
:meth:`to_spyde_vectors` convert between this carrier and the in-memory
``SpyDEDiffractionVectors`` used by the render / virtual-imaging / orientation
code.

This is intentionally a *storage/serialization* type — distinct from
``DiffractionVectorsImage`` (the rendered-disk *display* view). Save writes this;
the result window still shows ``DiffractionVectorsImage`` frames rendered on
demand from the reconstructed ``SpyDEDiffractionVectors``.
"""
from __future__ import annotations

import numpy as np
from hyperspy.signals import BaseSignal

from spyde.signals.diffraction_vectors import COLUMN_NAMES

# Where the reconstruction metadata lives on the signal.
META_ROOT = "SpyDE.DiffractionVectors"
SIGNAL_TYPE = "spyde_dense_diffraction_vectors"


class DenseDiffractionVectors(BaseSignal):
    """Dense flat-buffer storage of a diffraction-vectors result.

    Not ragged: ``.data`` is one ``(N_total, 6)`` buffer for the whole scan, with
    the per-position grouping carried by the CSR offsets in metadata. Convert to
    the in-memory :class:`SpyDEDiffractionVectors` with :meth:`to_spyde_vectors`.
    """

    _signal_type = SIGNAL_TYPE
    _signal_dimension = 1   # rows are 6-vectors; nav axis is "vector index"

    def to_spyde_vectors(self):
        """Reconstruct the in-memory :class:`SpyDEDiffractionVectors`."""
        return from_dense_signal(self)


def _axes_meta(sig_axes) -> list[dict]:
    """Serialise signal-axis calibration to a list of plain dicts."""
    out = []
    for ax in sig_axes:
        out.append(dict(
            scale=float(ax.scale),
            offset=float(ax.offset),
            size=int(ax.size),
            units=str(getattr(ax, "units", "") or ""),
            name=str(getattr(ax, "name", "") or ""),
        ))
    return out


def to_dense_signal(vecs) -> DenseDiffractionVectors:
    """Pack a :class:`SpyDEDiffractionVectors` into a savable
    :class:`DenseDiffractionVectors` (flat buffer + metadata).

    The result saves losslessly to ``.zspy``/``.hspy``; reload with
    :func:`from_dense_signal` (or ``hs.load`` + ``.to_spyde_vectors()``)."""
    # A contiguous, plain-numpy copy so dask/zarr write a clean chunked array.
    flat = np.ascontiguousarray(np.asarray(vecs.flat_buffer, dtype=np.float32))
    # Zarr can't chunk a zero-length axis (ceildiv by 0). For an empty result,
    # write a single zero sentinel row and record the true count so load restores
    # the empty (0, 6) buffer.
    n_vectors = int(flat.shape[0])
    if n_vectors == 0:
        flat = np.zeros((1, 6), dtype=np.float32)
    sig = DenseDiffractionVectors(flat)
    sig.metadata.set_item(f"{META_ROOT}.n_vectors", n_vectors)
    # Positional column semantics so an external reader (no SpyDE) can interpret
    # the dense (N, 6) buffer — mirrors pyxem's VectorMetadata.column_names.
    sig.metadata.set_item(f"{META_ROOT}.column_names", list(COLUMN_NAMES))
    sig.metadata.set_item(f"{META_ROOT}.nav_offsets",
                          [np.asarray(o, dtype=np.int64) for o in vecs.nav_offsets])
    sig.metadata.set_item(f"{META_ROOT}.full_nav_shape",
                          [int(s) for s in vecs.full_nav_shape])
    sig.metadata.set_item(f"{META_ROOT}.sig_shape",
                          [int(s) for s in vecs.sig_shape])
    sig.metadata.set_item(f"{META_ROOT}.sig_axes", _axes_meta(vecs.sig_axes))
    sig.metadata.set_item(f"{META_ROOT}.nav_axes",
                          _axes_meta(getattr(vecs, "nav_axes", []) or []))
    sig.metadata.set_item(f"{META_ROOT}.kernel_radius_px",
                          float(vecs.kernel_radius_px))
    sig.metadata.set_item(f"{META_ROOT}.kernel_radius_data",
                          float(vecs.kernel_radius_data))
    sig.metadata.set_item(f"{META_ROOT}.params", dict(vecs.params or {}))
    sig.metadata.General.title = "Diffraction Vectors"
    return sig


def from_dense_signal(sig) -> "object":
    """Reconstruct a :class:`SpyDEDiffractionVectors` from a loaded
    :class:`DenseDiffractionVectors` (or any signal carrying the
    ``SpyDE.DiffractionVectors`` metadata + an ``(N, 6)`` flat buffer)."""
    from spyde.signals.diffraction_vectors import (
        SpyDEDiffractionVectors, _AxisLite,
    )

    if not sig.metadata.has_item(META_ROOT):
        raise ValueError(
            "signal has no SpyDE.DiffractionVectors metadata — not a "
            "DenseDiffractionVectors file"
        )
    # The metadata node is a hyperspy DictionaryTreeBrowser; flatten to a plain
    # dict so nested lists/arrays come back as themselves (not browser nodes).
    md = sig.metadata.get_item(META_ROOT).as_dictionary()

    flat_buffer = np.ascontiguousarray(np.asarray(sig.data, dtype=np.float32))
    if flat_buffer.ndim != 2 or flat_buffer.shape[1] != 6:
        raise ValueError(
            f"DenseDiffractionVectors data must be (N, 6); got {flat_buffer.shape}"
        )
    # An empty result was written with a single sentinel row (see to_dense_signal);
    # n_vectors records the true count so we restore the genuine (0, 6) buffer.
    n_vectors = int(md.get("n_vectors", flat_buffer.shape[0]))
    flat_buffer = flat_buffer[:n_vectors]

    full_nav_shape = tuple(int(s) for s in md["full_nav_shape"])
    sig_shape = tuple(int(s) for s in md["sig_shape"])
    sig_axes = [_AxisLite(**dict(a)) for a in md["sig_axes"]]
    nav_axes = [_AxisLite(**dict(a)) for a in md.get("nav_axes", []) or []]

    # from_arrays rebuilds nav_offsets from the (already-sorted) flat buffer, so
    # we don't strictly need the stored offsets — but they're exact, so this is a
    # cheap, deterministic reconstruction either way.
    return SpyDEDiffractionVectors.from_arrays(
        flat_buffer=flat_buffer,
        full_nav_shape=full_nav_shape,
        sig_shape=sig_shape,
        sig_axes=sig_axes,
        kernel_radius_px=float(md["kernel_radius_px"]),
        kernel_radius_data=float(md["kernel_radius_data"]),
        params=dict(md.get("params", {}) or {}),
        nav_axes=nav_axes,
    )


def is_dense_vectors_signal(sig) -> bool:
    """True if ``sig`` carries the SpyDE dense-vectors metadata (so it should be
    reconstructed into a vectors result tree rather than opened as image data)."""
    try:
        return sig.metadata.has_item(META_ROOT)
    except Exception:
        return False
