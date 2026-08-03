"""A label image for tests that need one, without an engine.

Most of the particle tests do not care HOW a frame became labels — they are
testing measurement, tracking, the CSR store, the overlay or the tree, and they
just need a plausible ``int32`` label image to feed them. They used to call
``segment_frame``, which went with the classical engine.

Otsu + :func:`split_instances` is exactly what that engine did, so nothing about
these tests changes. The difference is where it lives: a fixed global threshold
is a perfectly reasonable way to fabricate a label image from a *synthetic*
fixture built to have two clean modes, and a completely unreasonable way to
segment real low-contrast in-situ data — which is why it is a test helper now
and not a shipped engine. See :mod:`spyde.particles.instances`.

A test that genuinely wants to exercise the real path should train a
``ScribbleClassifier`` instead (``test_particles_scribble.py``).
"""
from __future__ import annotations

import numpy as np


def labels_from(frame, *, invert: bool = False, blur: float = 0.0,
                sensitivity: float = 0.5, **params) -> np.ndarray:
    """``int32`` labels for *frame*: prepare → otsu → the shared instance split.

    Reproduces the deleted ``segment_frame`` exactly for the arguments these
    tests used, so their asserted counts and ground-truth matches are unchanged:
    NaN fill (polarity following *invert*, because a drift-corrected frame
    carries a NaN border and filling it the wrong way segments the padding as
    one enormous edge-hugging particle), optional gaussian pre-blur, invert,
    then otsu at ``sensitivity=0.5`` — which contributed a zero offset, so it
    drops out.

    *blur* is the old ``gaussian``; *invert* selects DARK objects;
    *sensitivity* is the old 0..1 control, an additive offset scaled by the
    image's robust 5–95 spread (0.5 = plain otsu, 1.0 lowers the threshold by
    half the spread). It is kept ONLY so
    ``TestNoGlobalThresholdFindsTheFaintProbes`` can sweep it and show that no
    setting rescues a global threshold — the measurement that justifies having
    deleted the engine. Remaining keyword arguments go to
    :class:`~spyde.particles.instances.SegmentParams`.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_otsu

    from spyde.particles import SegmentParams, split_instances

    # Caught HERE rather than four frames down inside SegmentParams, whose own
    # TypeError names the field but not the fix. `gaussian=` is the natural
    # thing to write when porting a call from the deleted `segment_frame`.
    # (`invert` and `sensitivity` are real arguments of this function, so they
    # never reach **params.)
    if "gaussian" in params:
        raise TypeError("labels_from() has no 'gaussian'; it is 'blur' here")
    gone = {"threshold", "rb_kernel", "local_size"} & set(params)
    if gone:
        raise TypeError(
            f"labels_from() has no {sorted(gone)} — they belonged to the "
            f"deleted classical engine (see spyde.particles.instances). Build "
            f"the foreground you want and call split_instances directly.")

    img = np.asarray(frame, dtype=np.float32)
    bad = ~np.isfinite(img)
    if bad.any():
        finite = img[~bad]
        img = img.copy()
        img[bad] = (finite.max() if invert else finite.min()) if finite.size else 0.0
    if blur > 0:
        img = gaussian_filter(img, float(blur))
    if invert:
        img = -img

    offset = 0.0
    if sensitivity != 0.5:
        finite = img[np.isfinite(img)]
        if finite.size:
            lo, hi = np.percentile(finite, [5, 95])
            offset = -(float(sensitivity) - 0.5) * float(hi - lo)
    return split_instances(img > (threshold_otsu(img) + offset),
                           SegmentParams(**params))
