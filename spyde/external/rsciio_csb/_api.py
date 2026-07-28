"""file_reader for Direct Electron CSB centroid-streaming files.

A CSB file is **not a frame stack** — it is a compressed sparse block stream of
detected electron events with a frame cadence. There is no dense array to read;
an image only exists once you choose a time window and integrate the events in
it. So "opening" a CSB movie means choosing an exposure, and the reader's job
is to make that choice cheap to change and cheap to scrub.

The lazy path is the point
--------------------------
``lazy=True`` returns a dask array with **one time plane per block**, each block
integrating only its own window. That shape is load-bearing, not incidental:
SpyDE's navigator scrub rests on ``data[i].compute()`` touching only plane
``i``, and the sibling ``rosettasciio/mrc`` patch documents a 40x scrub
regression from a graph that read a whole 134 MB chunk to yield one frame. A
plane here is an integration over its own frames and nothing else.

Two more consequences of the format:

* **The navigator is free.** A per-plane total-counts overview comes from the
  block table alone — ``counts`` summed per frame — reading zero payload bytes.
  Handing it over matters: without it a viewer builds the overview by reducing
  every plane, i.e. by integrating the entire movie, which is the one thing
  this reader exists to avoid.
* **The accumulator is not concurrency-safe.** ``reset()`` + ``add_frames()``
  mutate one shared device/host buffer, so concurrent planes would corrupt each
  other. Integrations take a lock rather than getting an accumulator each: at
  8192² one accumulator's buffer is 268 MB, so per-thread copies would cost
  gigabytes to win back parallelism the GPU serialises anyway.
"""
from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict

import numpy as np

from ._core import CSBFile
from ._sparse import SparseCSB

_logger = logging.getLogger(__name__)

#: Planes to aim for when the caller names no exposure. Enough that the time
#: slider has somewhere to go, few enough that each plane has real signal in it
#: — a single frame of a sparse stream is mostly empty pixels.
DEFAULT_PLANES = 50

#: Longest plane edge to integrate at by default. A CSB frame can be 8192², and
#: at float32 that is a 268 MB readback per plane — which IS the cost of a
#: scrub (accumulating the events is ~10 ms of it). de-csb's own README says
#: the same thing under "Readout is a separate cost": the full image caps you
#: at ~11 Hz while a 4x-binned one runs at ~84 Hz, so display from the binned
#: one and pull the full image only when something is actually measured.
#:
#: Binning is a SUM, so no counts are lost — and on data this sparse (the test
#: movie is ~1.9 counts/px for the entire 125.8M-event movie) binning is what
#: makes a plane visible at all. Pass ``bin=1`` for the full grid.
DEFAULT_MAX_EDGE = 2048

# One integration at a time (see the module docstring).
_ACC_LOCK = threading.Lock()

# Open datasets, keyed by (path, backend). The graph carries this KEY rather
# than the SparseCSB itself, for two reasons:
#
#   * `dask.delayed(pure=True)` tokenizes its arguments, and a SparseCSB holds
#     the memory-mapped payload — so passing it made dask hash the whole file
#     once per plane. On the 252 MB test movie that was 70 SECONDS to build a
#     graph that reads nothing.
#   * a key is picklable and a memmap-backed reader is not, so the same graph
#     can go to a distributed worker, which resolves its own dataset here.
#
# One dataset per key also means one accumulator per key (SparseCSB caches it),
# which is what keeps device memory from being reallocated on every scrub.
_DATASETS: dict[tuple, SparseCSB] = {}
_DATASETS_LOCK = threading.Lock()


class _TorchSparseCSB(SparseCSB):
    """A SparseCSB whose integrations run on torch-CUDA.

    ``_core``'s GPU lane needs CuPy, which SpyDE does not install — its GPU
    stack is torch. Subclassing rather than editing ``_sparse`` keeps both
    vendored modules diffable against upstream de-csb.
    """

    def _accumulator(self):
        if self._acc is None:
            from ._torch import TorchAccumulator
            self._acc = TorchAccumulator(self.csb)
        return self._acc

    def _sum_frames(self, f0, f1, bin_factor, dtype):
        """Integrate, binning ON THE DEVICE before the readback.

        ``_sparse``'s version pulls the full image and bins on the host, so
        asking for bin=4 costs MORE than bin=1 — it pays the full 268 MB
        transfer and then bins it. Binning first is the entire reason
        ``preview()`` exists: it moves 1/16th of the bytes, which is where a
        scrub's time actually goes.
        """
        acc = self._accumulator()
        acc.reset()
        acc.add_frames(int(f0), int(f1))
        acc.synchronize()
        if bin_factor and int(bin_factor) > 1:
            return acc.preview(int(bin_factor), dtype)
        return acc.image(dtype)


def _dataset(path: str, backend: str) -> SparseCSB:
    """The process-wide dataset for one file, opened once.

    ``backend="auto"`` prefers torch-CUDA when there is a device, because that
    is the GPU SpyDE actually ships; it falls back to ``_core``'s own auto
    (CuPy if somehow present, else numba, else numpy).
    """
    key = (os.path.abspath(path), backend)
    with _DATASETS_LOCK:
        ds = _DATASETS.get(key)
        if ds is None:
            csb = CSBFile(path)
            use_torch = backend == "torch"
            if backend == "auto":
                from ._core import gpu_available
                from ._torch import torch_gpu_available
                use_torch = not gpu_available() and torch_gpu_available()
            if use_torch:
                ds = _TorchSparseCSB(csb, backend=backend)
                _logger.debug("CSB %s: integrating on torch-CUDA",
                              os.path.basename(path))
            else:
                ds = SparseCSB(csb, backend=backend)
            _DATASETS[key] = ds
        return ds


def _resolve_step(ds: SparseCSB, step, frames_per_plane) -> float:
    """The exposure per output plane, in seconds.

    ``frames_per_plane`` is offered because a frame is the atomic exposure and
    "8 frames" is a more natural thing to ask for than "3.12 ms". Whichever is
    given, the result is floored at one frame — a step finer than the cadence
    cannot produce more planes, only empty ones.
    """
    dt = ds.frame_duration
    if dt <= 0:
        raise ValueError(
            "this CSB file declares no frame cadence (microsec_per_frame is 0), "
            "so it has no time axis to integrate over")
    if frames_per_plane:
        return max(1, int(frames_per_plane)) * dt
    if step:
        return max(float(step), dt)
    return max(ds.duration / DEFAULT_PLANES, dt)


def _resolve_bin(ds: SparseCSB, requested) -> int:
    """Spatial binning per plane. ``None``/0 picks one for the frame size.

    The default keeps a plane's longest edge at or under
    :data:`DEFAULT_MAX_EDGE` by the next power of two, so an 8192² movie
    integrates 2048² planes (16 MB, ~36 ms) instead of 8192² ones (268 MB,
    ~180 ms). An explicit ``bin`` — including ``bin=1`` — is always honoured.
    """
    if requested:
        return max(1, int(requested))
    edge = max(ds.shape)
    factor = 1
    while edge // factor > DEFAULT_MAX_EDGE and factor < 16:
        factor *= 2
    return factor


def plane_counts(ds: SparseCSB, bounds) -> np.ndarray:
    """Total events per plane, straight from the block table.

    The navigator, for free — no payload byte is read. ``counts`` is one entry
    per block and blocks tile each frame, so summing within a frame and then
    across a plane's frames gives its total intensity.
    """
    bpf = int(ds.csb.blocks_per_frame)
    per_frame = np.asarray(ds.csb.counts, np.int64).reshape(-1, bpf).sum(1)
    return np.array([per_frame[f0:f1].sum() for f0, f1 in bounds], np.float64)


#: Bytes of integrated planes to keep. Scrubbing revisits planes constantly —
#: a drag goes back and forth over the same handful — and re-integrating one is
#: a fresh readback, which is the whole cost (see integrate_plane). Sized to
#: hold a decent run of binned planes (16 MB each at 2048²) or a couple of
#: full-resolution ones.
PLANE_CACHE_BYTES = 512 << 20

_plane_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_plane_cache_bytes = 0
_PLANE_CACHE_LOCK = threading.Lock()


def _cache_get(key):
    with _PLANE_CACHE_LOCK:
        hit = _plane_cache.get(key)
        if hit is not None:
            _plane_cache.move_to_end(key)          # LRU
        return hit


def _cache_put(key, arr: np.ndarray) -> None:
    global _plane_cache_bytes
    nbytes = int(arr.nbytes)
    if nbytes > PLANE_CACHE_BYTES:
        return                                     # one plane over budget
    with _PLANE_CACHE_LOCK:
        if key in _plane_cache:
            return
        _plane_cache[key] = arr
        _plane_cache_bytes += nbytes
        while _plane_cache_bytes > PLANE_CACHE_BYTES and len(_plane_cache) > 1:
            _k, old = _plane_cache.popitem(last=False)
            _plane_cache_bytes -= int(old.nbytes)


def clear_plane_cache() -> None:
    """Drop every cached plane (a different exposure invalidates nothing —
    the key carries the window — so this is only for tests and teardown)."""
    global _plane_cache_bytes
    with _PLANE_CACHE_LOCK:
        _plane_cache.clear()
        _plane_cache_bytes = 0


def integrate_plane(path: str, backend: str, f0: int, f1: int,
                    bin_factor: int, dtype) -> np.ndarray:
    """One plane: integrate frames ``[f0, f1)`` -> ``(1, H, W)``.

    Takes the file PATH, not an open dataset — see :data:`_DATASETS`. Every
    argument is small and hashable, so the graph is cheap to build, it can be
    shipped to a worker, and it doubles as the cache key.

    Cached because a scrub is not a linear pass: dragging the slider walks back
    and forth over the same few planes, and each repeat would otherwise pay the
    full readback again. Accumulating the events is ~10 ms; pulling the result
    back is the rest.
    """
    key = (os.path.abspath(path), backend, int(f0), int(f1), int(bin_factor),
           np.dtype(dtype).str)
    hit = _cache_get(key)
    if hit is not None:
        return hit

    ds = _dataset(path, backend)
    with _ACC_LOCK:
        img = ds._sum_frames(f0, f1, bin_factor, dtype)
    out = np.asarray(img, dtype)[None]
    _cache_put(key, out)
    return out


def lazy_stack(path: str, bounds, shape, bin_factor: int = 1,
               dtype=np.float32, backend: str = "auto"):
    """A dask array of integrated planes, one plane per block.

    Each block is an independent ``[f0, f1)`` integration, so computing plane
    ``i`` reads that window's events and nothing else — the property the
    navigator scrub depends on.
    """
    import dask
    import dask.array as da

    h, w = shape[0] // bin_factor, shape[1] // bin_factor
    delayed = dask.delayed(integrate_plane, pure=True)
    blocks = [
        da.from_delayed(delayed(path, backend, int(f0), int(f1), bin_factor, dtype),
                        shape=(1, h, w), dtype=dtype)
        for f0, f1 in bounds
    ]
    return da.concatenate(blocks, axis=0)


def _axes(ds: SparseCSB, bounds, bin_factor: int):
    """Time / y / x axes, calibrated. The time axis is named so a viewer can
    recognise this as a movie rather than a scan."""
    dt = ds.frame_duration
    times = np.array([f0 * dt for f0, _ in bounds], float)
    # Plane spacing, not frame spacing: consecutive planes are `step` apart.
    scale = float(times[1] - times[0]) if len(times) > 1 else float(dt)
    ang = float(getattr(ds.csb, "ang_per_pix", 0.0) or 0.0) * bin_factor
    px = {"scale": ang, "units": "Å"} if ang > 0 else {"scale": 1.0, "units": "px"}
    h = ds.csb.frame_height // bin_factor
    w = ds.csb.frame_width // bin_factor
    return [
        {"name": "time", "size": len(bounds), "navigate": True,
         "offset": float(times[0]), "scale": scale, "units": "s"},
        {"name": "y", "size": h, "navigate": False, "offset": 0.0, **px},
        {"name": "x", "size": w, "navigate": False, "offset": 0.0, **px},
    ]


def file_reader(filename, lazy=False, step=None, frames_per_plane=None,
                start=0.0, stop=None, bin=None, backend="auto", **kwds):
    """Read a CSB centroid stream as a stack of integrated time planes.

    Parameters
    ----------
    lazy : bool
        With ``True`` (strongly preferred) each plane integrates on demand, so
        opening the file costs only the header and block table. Eager reads
        integrate every plane up front — at 8192² that is 268 MB per plane.
    step : float, optional
        Exposure per plane in seconds. Defaults to the whole movie split into
        :data:`DEFAULT_PLANES`.
    frames_per_plane : int, optional
        The same thing in frames, which is usually how it is thought about.
        Takes precedence over *step*.
    start, stop : float
        Time range to cover, in seconds. Defaults to the whole movie.
    bin : int, optional
        Integer spatial binning, summed. Defaults to whatever keeps a plane's
        longest edge at :data:`DEFAULT_MAX_EDGE` — see :func:`_resolve_bin`,
        and note that binning is what makes a scrub interactive AND what makes
        sparse counted data visible. Pass ``1`` for the full pixel grid.
    backend : {"auto", "gpu", "cpu", "cpu-numba", "cpu-numpy"}
        Integration backend; ``"auto"`` takes the GPU when there is one.
    """
    if kwds:
        _logger.debug("CSB reader ignoring unknown kwargs: %s", sorted(kwds))
    path = str(filename)
    ds = _dataset(path, backend)
    bin_factor = _resolve_bin(ds, bin)
    exposure = _resolve_step(ds, step, frames_per_plane)
    bounds = ds._time_bounds(exposure, float(start),
                             ds.duration if stop is None else float(stop))
    if not bounds:
        raise ValueError(
            f"no planes: a {exposure:g} s exposure over "
            f"[{start}, {stop if stop is not None else ds.duration}) s of this "
            f"{ds.duration:g} s movie selects no frames")

    dtype = np.float32
    if lazy:
        data = lazy_stack(path, bounds, ds.shape, bin_factor, dtype, backend)
    else:
        data = np.concatenate([integrate_plane(path, backend, f0, f1,
                                               bin_factor, dtype)
                               for f0, f1 in bounds])

    info = ds.csb.info()
    return [{
        "data": data,
        "axes": _axes(ds, bounds, bin_factor),
        "metadata": {
            "General": {
                "title": os.path.splitext(os.path.basename(str(filename)))[0],
                "original_filename": os.path.basename(str(filename)),
            },
            "Signal": {"signal_type": ""},
            "Acquisition_instrument": {
                "TEM": {
                    "magnification": info.get("mag"),
                    "defocus": info.get("defocus_um"),
                    "Detector": {"camera_serial": info.get("camera_sn")},
                },
            },
        },
        "original_metadata": {
            "csb": {
                **info,
                # What this particular read chose — without it a plane's
                # intensity is uninterpretable (it is events per exposure, and
                # the exposure was a load-time decision).
                "exposure_s": exposure,
                "frames_per_plane": [int(f1 - f0) for f0, f1 in bounds],
                "plane_frame_bounds": [[int(f0), int(f1)] for f0, f1 in bounds],
                "bin": bin_factor,
                "backend": backend,
                # The free navigator (see plane_counts). A reader cannot hand
                # HyperSpy a navigator directly, and the signal dict's
                # `attributes` key does NOT reach the signal object — so it
                # travels as ordinary original_metadata, which does, and a
                # viewer picks it up from there. Without this the overview is
                # built by reducing every plane, i.e. by integrating the whole
                # movie: the one thing this reader exists to avoid.
                "plane_counts": plane_counts(ds, bounds).tolist(),
            },
        },
    }]
