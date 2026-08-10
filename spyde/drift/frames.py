"""
frames.py — one streaming accessor for every kind of frame stack we accept.

The whole point of this module is the Memory-Safety rule (CLAUDE.md): a drift
solve on a 3000 × 4096² movie must never hold more than a couple of frames at
once. So callers get a ``(n_frames, get_frame, frame_shape)`` triple and read
frames one at a time — they never touch ``.data`` directly and therefore cannot
accidentally ``.compute()`` the whole array.

Accepted inputs, in the order they are tried:

* a HyperSpy signal (lazy or eager) with 1-D navigation and 2-D signal axes
* a dask array of shape ``(n, h, w)`` — sliced and computed per frame
* a numpy array of shape ``(n, h, w)`` — already resident, just indexed
* any sequence of 2-D arrays
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)


def _is_dask(obj) -> bool:
    """True for a dask array, WITHOUT importing dask when it isn't already in."""
    return type(obj).__module__.startswith("dask.array")


#: Warn once per process, not once per frame — a thousand identical warnings is
#: how you train people to ignore the log.
_warned_distributed = False


def _region_slice(stack, i: int, region, step: int = 1):
    """Slice frame *i*, optionally only the ``(y0, x0, h, w)`` sub-region.

    **Slice before reading, not after.** The ROI preview used to read whole
    frames and throw most of each away — on a 2048² movie aligning a 1024² box
    that is 4x the bytes it needs. On a memmap-backed .mrc (a real in-situ movie:
    no decode, a frame IS a slice) the read cost is close to linear in the region
    actually touched, so pushing the crop down to the reader is most of the cost.

    Rows are contiguous, so a row-restricted slice is the part that genuinely
    pays; narrowing columns still reads whole rows off disk. That asymmetry is
    real and is why the win is closer to h/H than to (h·w)/(H·W).

    *step* decimates on the SAME principle. The check images are thumbnails, and
    reading a whole 2048² frame only to keep every 4th pixel pays full price for
    a sixteenth of the data; ``[::step]`` on a memmap touches only the rows it
    keeps.
    """
    step = max(1, int(step))
    if region is None:
        return stack[i, ::step, ::step] if step > 1 else stack[i]
    y0, x0, h, w = (int(v) for v in region)
    return stack[i, y0:y0 + h:step, x0:x0 + w:step]


def _compute_one_frame(slice_) -> np.ndarray:
    """Compute ONE frame slice on the LOCAL threaded scheduler.

    **The explicit ``scheduler=`` is the whole point of this function.** A bare
    ``.compute()`` resolves the scheduler from context, and in the running app
    that context contains a process-global ``distributed.Client`` — so every
    frame becomes a scheduler round-trip: serialise the graph, hand it to a
    worker process, read there, ship the frame back over IPC. At 2048² that is
    ~8 MB back per frame and hundreds of milliseconds each, for data this
    process is going to consume locally anyway.

    Measured consequence, and it is the whole reason this function exists: the
    drift caret reads 64 frames for its check image, ~20 for the ROI preview and
    N for the solve, all through here. At a few hundred ms per frame that is the
    "the ROI spawns a drift check image which takes 60 seconds" report — one
    unqualified ``.compute()`` accounting for three separate symptoms.

    This is the SAME failure the navigator already fixed, one layer down: CLAUDE.md
    Live-Display §3 pins ``CachedDaskArray._client = None`` precisely so the
    per-frame read takes hyperspy's synchronous branch instead of the distributed
    one, and records that leaving the ambient client in place was a silent
    perf-only bug for a long time. This module bypassed that machinery entirely
    by holding a raw dask array, and so re-acquired the bug it had already been
    fixed. Anything here that reads frames one at a time must go through this
    function.
    """
    try:
        return np.asarray(slice_.compute(scheduler="threads"))
    except Exception as exc:
        # A graph whose data lives on the workers (a `.persist()`ed array) genuinely
        # cannot be computed locally. Fall back rather than fail the solve — but say
        # so, because on that path every frame costs a round-trip and the caller
        # deserves to know why their drift solve is slow.
        global _warned_distributed
        if not _warned_distributed:
            _warned_distributed = True
            log.warning(
                "[drift] local frame read failed (%s); falling back to the ambient "
                "scheduler. Per-frame reads now cost a cluster round-trip each — "
                "expect the drift check, preview and solve to be much slower.", exc)
        return np.asarray(slice_.compute())


def frame_source(data) -> tuple[int, Callable[[int], np.ndarray], tuple[int, int]]:
    """Return ``(n_frames, get_frame, frame_shape)`` for *data*.

    ``get_frame(i)`` returns frame ``i`` as a **numpy** 2-D array. For a lazy
    (dask) backing it computes exactly that one frame — never the whole stack.

    Raises
    ------
    TypeError
        If *data* is not a recognised stack, or is not 3-D / not a sequence of
        2-D frames. Failing loudly here is deliberate: a silently-wrong axis
        order would produce a plausible but meaningless drift curve.
    """
    # ── HyperSpy signal ──────────────────────────────────────────────────────
    # Duck-typed on axes_manager so this module never imports hyperspy (import
    # cost at backend startup) and so test doubles work.
    if hasattr(data, "axes_manager") and hasattr(data, "data"):
        am = data.axes_manager
        nav = int(am.navigation_dimension)
        sig = int(am.signal_dimension)
        if sig != 2:
            raise TypeError(
                f"drift needs 2-D signal axes (images); got signal_dimension={sig}"
            )
        if nav != 1:
            raise TypeError(
                "drift needs a 1-D navigation axis (a frame stack / movie); got "
                f"navigation_dimension={nav}. Reduce a higher-dimensional "
                "dataset to a movie first (e.g. a virtual image)."
            )
        return frame_source(data.data)

    # ── dask / numpy 3-D array ───────────────────────────────────────────────
    if _is_dask(data) or isinstance(data, np.ndarray):
        if data.ndim != 3:
            raise TypeError(f"expected a 3-D (n, h, w) stack; got shape {data.shape}")
        n, h, w = data.shape
        if _is_dask(data):
            def get_frame(i: int, region=None, step: int = 1, _d=data) -> np.ndarray:
                # ONE frame, computed LOCALLY. Never `_d.compute()`.
                return _compute_one_frame(_region_slice(_d, int(i), region, step))
        else:
            def get_frame(i: int, region=None, step: int = 1, _d=data) -> np.ndarray:
                return np.asarray(_region_slice(_d, int(i), region, step))
        return int(n), get_frame, (int(h), int(w))

    # ── sequence of 2-D frames ───────────────────────────────────────────────
    try:
        n = len(data)
    except TypeError as exc:
        raise TypeError(
            f"cannot read frames from {type(data).__name__}: expected a HyperSpy "
            "signal, a 3-D array, or a sequence of 2-D arrays"
        ) from exc
    if n == 0:
        raise TypeError("empty frame stack")
    first = np.asarray(data[0])
    if first.ndim != 2:
        raise TypeError(f"sequence elements must be 2-D frames; got ndim={first.ndim}")

    def get_frame(i: int, _d=data) -> np.ndarray:
        return np.asarray(_d[int(i)])

    return int(n), get_frame, (int(first.shape[0]), int(first.shape[1]))
