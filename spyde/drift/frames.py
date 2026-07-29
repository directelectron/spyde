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

from typing import Callable

import numpy as np


def _is_dask(obj) -> bool:
    """True for a dask array, WITHOUT importing dask when it isn't already in."""
    return type(obj).__module__.startswith("dask.array")


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
            def get_frame(i: int, _d=data) -> np.ndarray:
                # ONE frame. Never `_d.compute()`.
                return np.asarray(_d[int(i)].compute())
        else:
            def get_frame(i: int, _d=data) -> np.ndarray:
                return np.asarray(_d[int(i)])
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
