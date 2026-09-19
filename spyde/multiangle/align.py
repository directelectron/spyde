"""
align.py — the two alignments of a multi-angle acquisition, both integer-only.

They are the same measurement on different images, and each is one call to
:func:`spyde.drift.solve_translation`:

* :func:`solve_real_space` correlates the members' real-space images (in
  practice each member's navigator) so the scan grids overlap.
* :func:`solve_reciprocal` correlates the members' mean diffraction patterns so
  the direct beams coincide.

There is deliberately no second implementation of phase correlation here. The
drift solver already streams one frame at a time, already refines the peak to a
fraction of a pixel, already rejects an implausible shift, and already has a
numerical test suite; these two functions name the chosen member as its fixed
reference and round the answer. Everything they add over that is the rounding
and its residual.

**Both solves take a FIXED reference — the chosen member — not a running
average.** A running average is right for a drift movie, where every frame is
the same scene; it is wrong here, because members sit at different tilts and so
genuinely differ, and an average of them is nobody's image. Each member's offset
has to be measured against the one member the rest will be read relative to.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from spyde.drift.frames import frame_source
from spyde.drift.translation import solve_translation


def _solve_offsets(images, reference: int, solver_kwargs: dict[str, Any]):
    """Integer offsets and their discarded remainder, one per member."""
    # Only for the member count and the shape check — the solver reads the
    # frames itself, one at a time.
    n_members, _, _ = frame_source(images)
    try:
        reference = int(reference)
    except (TypeError, ValueError):
        # `reference` here is a member INDEX, not one of the solver's reference
        # modes; a caller reaching for "running" deserves to be told so.
        raise ValueError(
            f"reference must be a member index; got {reference!r}") from None
    if not 0 <= reference < n_members:
        raise ValueError(
            f"reference member {reference} outside 0..{n_members - 1}")

    model = solve_translation(
        images, reference=f"fixed:{reference}", **solver_kwargs)

    shifts = np.asarray(model.shifts, dtype=np.float64)
    if not np.all(np.isfinite(shifts)):
        raise ValueError(
            "the translation solve did not produce a shift for every member "
            "(it was cancelled, or a member could not be registered)"
        )

    offsets = np.rint(shifts).astype(np.int64)
    residuals = (shifts - offsets).astype(np.float32)
    return offsets, residuals


def solve_real_space(images, *, reference: int = 0, **solver_kwargs):
    """Align the members' real-space images. Returns ``(offsets, residuals)``.

    Parameters
    ----------
    images
        ``(N, h, w)`` — one real-space image per member, member order, anything
        :func:`spyde.drift.frames.frame_source` accepts. In practice these are
        the members' navigators, which are already reduced; nothing here reads a
        member's 4-D array.
    reference
        Index of the member the others are aligned to. It gets ``(0, 0)``.
    solver_kwargs
        Passed through to :func:`spyde.drift.solve_translation` — ``max_shift``,
        ``upsample``, ``roi`` and the rest. ``roi`` is worth reaching for when
        only part of the field of view is common to every member.

    Returns
    -------
    offsets
        ``(N, 2)`` int64 — the scan-position ``(dy, dx)`` to ADD to each member
        (:mod:`spyde.multiangle.model` states the sign convention).
    residuals
        ``(N, 2)`` float32 — what rounding threw away. Report it; a component
        near 0.5 px means the integer answer was a coin toss.
    """
    return _solve_offsets(images, reference, solver_kwargs)


def solve_reciprocal(patterns, *, reference: int = 0, **solver_kwargs):
    """Align the members' diffraction patterns. Returns ``(offsets, residuals)``.

    Identical in every respect to :func:`solve_real_space` except for what it is
    given: ``patterns`` is ``(N, ky, kx)``, one mean diffraction pattern per
    member, and the feature being registered is the direct beam. The returned
    offsets are DETECTOR pixels.
    """
    return _solve_offsets(patterns, reference, solver_kwargs)
