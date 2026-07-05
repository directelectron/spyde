"""SpyDE's self-contained result/signal containers.

These classes are the currency of the three-host parity contract
(NOTEBOOK_PARITY_PLAN.md): the app's actions, `spyde.api` scripts, and future
notebook wizards all return THE SAME objects, which import no backend/drawing
code and are constructible + saveable standalone.
"""
from spyde.signals.diffraction_vectors import SpyDEDiffractionVectors
from spyde.signals.orientation_map import SpyDEOrientationMap

__all__ = ["SpyDEDiffractionVectors", "SpyDEOrientationMap"]
