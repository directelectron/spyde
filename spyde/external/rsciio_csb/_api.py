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

import numpy as np

from ._core import CSBFile
from ._sparse import SparseCSB

_logger = logging.getLogger(__name__)

#: Planes to aim for when the caller names no exposure. Enough that the time
#: slider has somewhere to go, few enough that each plane has real signal in it
#: — a single frame of a sparse stream is mostly empty pixels.
DEFAULT_PLANES = 50

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


def _dataset(path: str, backend: str) -> SparseCSB:
    """The process-wide dataset for one file, opened once."""
    key = (os.path.abspath(path), backend)
    with _DATASETS_LOCK:
        ds = _DATASETS.get(key)
        if ds is None:
            ds = SparseCSB(CSBFile(path), backend=backend)
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


def plane_counts(ds: SparseCSB, bounds) -> np.ndarray:
    """Total events per plane, straight from the block table.

    The navigator, for free — no payload byte is read. ``counts`` is one entry
    per block and blocks tile each frame, so summing within a frame and then
    across a plane's frames gives its total intensity.
    """
    bpf = int(ds.csb.blocks_per_frame)
    per_frame = np.asarray(ds.csb.counts, np.int64).reshape(-1, bpf).sum(1)
    return np.array([per_frame[f0:f1].sum() for f0, f1 in bounds], np.float64)


def integrate_plane(path: str, backend: str, f0: int, f1: int,
                    bin_factor: int, dtype) -> np.ndarray:
    """One plane: integrate frames ``[f0, f1)`` -> ``(1, H, W)``.

    Takes the file PATH, not an open dataset — see :data:`_DATASETS`. Every
    argument is small and hashable, so the graph is cheap to build and can be
    shipped to a worker.
    """
    ds = _dataset(path, backend)
    with _ACC_LOCK:
        img = ds._sum_frames(f0, f1, bin_factor, dtype)
    return np.asarray(img, dtype)[None]


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
                start=0.0, stop=None, bin=1, backend="auto", **kwds):
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
    bin : int
        Integer spatial binning, summed.
    backend : {"auto", "gpu", "cpu", "cpu-numba", "cpu-numpy"}
        Integration backend; ``"auto"`` takes the GPU when there is one.
    """
    if kwds:
        _logger.debug("CSB reader ignoring unknown kwargs: %s", sorted(kwds))
    path = str(filename)
    ds = _dataset(path, backend)
    bin_factor = max(1, int(bin))
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
