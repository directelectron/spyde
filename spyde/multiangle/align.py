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


def _local_contrast(image, size: int = 25):
    """Intensity CHANGE, relative to the neighbourhood it sits in.

    The members are recorded at different tilts, so the same position is not
    obliged to be equally bright in all of them: a tilt changes which
    reflections are excited, and with them the contrast. Registering on
    intensity assumes they are comparable. This is blind to a slow offset and
    to a slow gain, which is what the difference between two tilts mostly is.

    *size* is in scan pixels and wants to be larger than the features being
    registered and smaller than the image. Too small and it removes the
    features along with the background; too large and it is a global mean and
    does nothing.
    """
    from scipy.ndimage import uniform_filter

    values = np.asarray(image, dtype=np.float64)
    local = uniform_filter(values, size)
    # A floor, so vacuum or a beam-stop shadow cannot divide by nothing and
    # hand the solver a field of spikes to register on.
    floor = 0.05 * float(np.mean(np.abs(local))) + 1e-9
    return (values - local) / np.maximum(local, floor)


def _high_pass(image, sigma: float = 8.0):
    """The same idea without the division — safer where the image is dark."""
    from scipy.ndimage import gaussian_filter

    values = np.asarray(image, dtype=np.float64)
    return values - gaussian_filter(values, sigma)


#: What the images are registered ON. Raw intensity is what a drift solver
#: assumes and is the weakest assumption here for the reason `_local_contrast`
#: gives. Measured on a four-member acquisition of a layered specimen, by how
#: much structure survived the sum and how strong the layers' own periodicity
#: was: raw 1.16 / 294, high pass 1.48 / 269, local contrast 1.49 / 295 — and
#: ALONG the layers, where the specimen is uniform and the registration has
#: least to hold on to, the two derived ones were 28% better than raw.
REGISTRATION_PREFILTERS = (
    ("local contrast", _local_contrast),
    ("high pass", _high_pass),
    ("raw intensity", None),
)

#: How far ahead a candidate must be before it is preferred to the settings
#: :func:`solve_real_space` chooses for itself. The scores of answers that are
#: a pixel apart differ by well under a percent, so a bare "highest wins" turns
#: that noise into a different answer; a real improvement on a real image is
#: worth tens of percent.
DECISIVE_MARGIN = 0.10

#: Registration settings tried when the answer is chosen by evidence rather
#: than assumed. The first is :func:`spyde.drift.solve_translation`'s own, which
#: is tuned for drift-correcting an in-situ movie; the others matter because a
#: virtual image of a layered specimen is not that. Whitening amplifies noise in
#: a low-contrast image and sharpens every fringe of a periodic structure
#: equally; apodizing tapers away the very features that sit near the edges.
REGISTRATION_CANDIDATES = (
    {"apodize": True, "normalize": True},
    {"apodize": False, "normalize": False},
    {"apodize": True, "normalize": False},
    {"apodize": False, "normalize": True},
)


def best_real_space(images, *, reference=0, score, **solver_kwargs):
    """Real-space offsets chosen by what they do to the sum, not by trust.

    Phase correlation answers a question with one maximum. A specimen of
    repeating layers does not have one: its correlation peak repeats with the
    layers, so the solver can sit a whole layer out and still report a clean
    sub-pixel residual — the number that is supposed to say the answer is firm.
    Whatever settings it runs with, the way to know is to look at the sum.

    So every candidate is SCORED: *score* takes offsets and returns how much
    sharper the members summed at them are than summed unaligned, which is the
    ratio the loader already shows.

    A score only overrules the solver's own settings by :data:`DECISIVE_MARGIN`,
    because it cannot resolve the last pixel on a small or low-contrast image —
    measured on a 24x28 acquisition whose true offsets are known, the whole
    field of candidates spans 0.65% and the one that scores highest is a pixel
    wrong. On images with something to see the winner is ahead by tens of
    percent. Below the margin the number is noise and the answer is to leave
    the default alone; above it, it is real.

    Returns ``(offsets, residuals, report)``; *report* lists what was tried.
    """
    attempts = []
    best = best_gain = best_residuals = None
    default_gain = None
    for label, prefilter in REGISTRATION_PREFILTERS:
        prepared = (images if prefilter is None
                    else np.stack([prefilter(image) for image in images]))
        for candidate in REGISTRATION_CANDIDATES:
            settings = {**solver_kwargs, **candidate}
            named = {"on": label, **candidate}
            try:
                offsets, residuals = solve_real_space(
                    prepared, reference=reference, **settings)
            except Exception as e:                # one bad combination is data
                attempts.append((named, None, str(e)))
                continue
            # Scored on the REAL images, never on the filtered ones: the
            # prefilter is how the offsets are found, not what they are for.
            gain = float(score(offsets))
            attempts.append((named, gain, None))
            if prefilter is None and candidate is REGISTRATION_CANDIDATES[0]:
                default = np.asarray(offsets, dtype=np.int64)
                default_gain, default_residuals = gain, residuals
            if best_gain is None or gain > best_gain:
                best, best_gain, best_residuals = (
                    np.asarray(offsets, dtype=np.int64), gain, residuals)

    if best is None:
        raise ValueError("no registration candidate produced offsets")

    decisive = (default_gain is None
                or best_gain > default_gain * (1.0 + DECISIVE_MARGIN))
    if not decisive:
        best, best_gain, best_residuals = default, default_gain, default_residuals

    return best, best_residuals, {"attempts": attempts, "gain": best_gain,
                                  "decisive": bool(decisive)}
