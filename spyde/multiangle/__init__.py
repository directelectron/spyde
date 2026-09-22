"""
spyde.multiangle — aligning a multi-angle 4D-STEM acquisition.

A multi-angle (MAPED-style) acquisition is N separate 4-D datasets, the
"members", recorded at different tilt magnitudes and azimuths about one centre.
Members sharing a tilt magnitude form a **shell**; there can be several shells
and they need not hold equal numbers of members, so nothing here assumes a
single global precession angle or an even split.

The two load-bearing constraints:

* **Alignment is an index REMAP, never a resample and never a copy.** Both the
  real-space alignment (scan positions) and the reciprocal one (detector pixels)
  are INTEGER. The output of a solve is a small :class:`MultiAngleModel` — two
  ``(N, 2)`` integer arrays — which a reader applies by shifting the indices it
  reads at. Nothing is interpolated and no member is written out. The sub-pixel
  remainder rounding discards is reported, not hidden.
* **Nothing materialises a member.** The solvers register already-reduced
  images: per-member navigators for real space, per-member mean diffraction
  patterns for reciprocal space. They read those one frame at a time through
  :func:`spyde.drift.frames.frame_source` and never touch a member's 4-D array.

Both solves are :func:`spyde.drift.solve_translation` with the chosen member as
a fixed reference, because a second implementation of phase correlation is a
second thing to get the sign of wrong. :mod:`spyde.drift.model` is the authority
on that sign; :mod:`spyde.multiangle.model` restates it once and points there.

Public API::

    from spyde.multiangle import (
        MultiAngleModel, assign_shells, solve_real_space, solve_reciprocal)

    nav_offsets, nav_residuals = solve_real_space(navigators, reference=0)
    dp_offsets, dp_residuals = solve_reciprocal(mean_patterns, reference=0)
    model = MultiAngleModel(
        paths=paths, tilts=tilts, azimuths=azimuths,
        shell_ids=assign_shells(tilts),
        nav_offsets=nav_offsets, dp_offsets=dp_offsets, reference=0,
        nav_residuals=nav_residuals, dp_residuals=dp_residuals,
    )
"""
from __future__ import annotations

from spyde.multiangle.align import solve_real_space, solve_reciprocal
from spyde.multiangle.model import (
    DEFAULT_SHELL_TOLERANCE,
    MultiAngleModel,
    assign_shells,
)
from spyde.multiangle.synthetic import SyntheticMultiAngle, make_multiangle

__all__ = [
    "MultiAngleModel",
    "assign_shells",
    "solve_real_space",
    "solve_reciprocal",
    "make_multiangle",
    "SyntheticMultiAngle",
    "DEFAULT_SHELL_TOLERANCE",
    "",
]
