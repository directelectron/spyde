"""
spyde.drift — drift correction for image stacks and in-situ movies.

See ``DRIFT_AND_PARTICLES_PLAN.md`` (repo root) for the full design. The two
load-bearing constraints, both from CLAUDE.md:

* **Nothing materialises the stack.** The target is thousands of frames at
  2048²–4096² (tens of GB). Every solver here STREAMS: read a frame, transform
  it, discard it. The output of a solve is a small :class:`DriftModel` — an
  ``(N, 2)`` shift array, not an aligned copy of the movie.
* **The corrected movie is a LAZY VIEW.** ``DriftModel`` describes the
  correction; applying it is a per-frame numpy operation
  (:func:`spyde.drift.warp.shift_frame`) wired into the signal tree as a lazy
  transformation node. Nothing is ever written out.

Public API::

    from spyde.drift import DriftModel, solve_translation, shift_frame

    model = solve_translation(signal, upsample=8, max_shift=32)
    aligned_frame = shift_frame(raw_frame, model.shifts[i])
"""
from __future__ import annotations

from spyde.drift.frames import frame_source
from spyde.drift.model import DriftModel
from spyde.drift.translation import solve_translation
from spyde.drift.warp import coverage_mask, shift_frame

__all__ = [
    "DriftModel",
    "solve_translation",
    "shift_frame",
    "coverage_mask",
    "frame_source",
]
