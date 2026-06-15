"""
DiffractionVectorsImage — a HyperSpy signal type for the *rendered-disk image*
view of a diffraction-vectors result tree.

The Find-Vectors result tree's root is a lazy image where each navigation
position is rendered from the stored vectors (flat disks at each peak, see
`SpyDEDiffractionVectors.to_rendered_dask`). It *displays* like a diffraction
pattern (2 signal dims), but it is NOT raw 4D-STEM data — it is a vectors
result. We give it a distinct signal type so the toolbar gating naturally keeps
the dense diffraction actions (Virtual Imaging / Orientation Mapping / Find
Diffraction Vectors) off it, while the vector actions (Vector Virtual Imaging,
Vector Orientation Mapping) attach to it.

Subclasses pyxem's Diffraction2D so it keeps all the diffraction display
behaviour; only the signal_type differs. Registered as a HyperSpy extension
(see spyde/hyperspy_extension.yaml) so `set_signal_type` and save/load work.

(Longer term this belongs upstream in pyxem alongside DiffractionVectors2D.)
"""
from __future__ import annotations

from pyxem.signals.diffraction2d import Diffraction2D, LazyDiffraction2D

SIGNAL_TYPE = "spyde_diffraction_vectors_image"


class DiffractionVectorsImage(Diffraction2D):
    """Rendered-disk image of a diffraction-vectors result (eager)."""
    _signal_type = SIGNAL_TYPE


class LazyDiffractionVectorsImage(LazyDiffraction2D):
    """Rendered-disk image of a diffraction-vectors result (lazy)."""
    _signal_type = SIGNAL_TYPE
