"""spyde.spectroscopy — EELS and EDS (0.3.0 Wave 2, GitHub #60).

Sits on top of :mod:`spyde.fitting`: a composition becomes a ``ModelSpec``,
which the batched engine fits.

**exspy owns the spectroscopy, this package owns the plumbing.** Edge and line
energies, GOS-based edge shapes, family relationships and quantification
factors all come from exspy so SpyDE's numbers match published work. What is
here is the uniform API, the range pruning, and the bridge to the batched
engine.

Needs the ``eels`` extra (``pip install "spyde[eels]"``). Importing this
package is safe without it — the functions raise
:class:`~spyde.spectroscopy.composition.MissingExtra` with the install line,
and toolbar actions are hidden by ``requires_package`` so a user never clicks
into the error.
"""
from __future__ import annotations

from spyde.spectroscopy.composition import (
    MissingExtra,
    model_for_composition,
    prune_to_range,
)

from spyde.spectroscopy.quantify import (
    element_intensity_maps,
    quantify,
    quantify_result,
)
from spyde.spectroscopy.tabulate import onset_energies, tabulate_model

__all__ = ["model_for_composition", "prune_to_range", "MissingExtra",
           "tabulate_model", "onset_energies",
           "element_intensity_maps", "quantify", "quantify_result"]
