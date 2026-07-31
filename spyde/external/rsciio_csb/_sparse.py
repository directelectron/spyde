"""Time-indexed sparse view over a CSB centroid-streaming file.

This is the layer a viewer sits on. :class:`SparseCSB` wraps the fast
:class:`csb.CSBFile` / :class:`csb.CSBAccumulator` core and adds:

* a physical **time axis** (frames carry a ``microsec_per_frame`` cadence), so
  you address the movie in seconds, not frame indices;
* :meth:`SparseCSB.integrate` - one dense image summed over a time window
  ``[t0, t1)``, the primitive a slider-driven viewer calls on every drag;
* :meth:`SparseCSB.to_frames` - a dense stack, one plane per fixed time step,
  the whole movie rebinned to a chosen exposure;
* :meth:`SparseCSB.events` / :meth:`SparseCSB.events_in_time` - the decoded
  ``(y, x)`` event lists (a COO-style sparse view) for an event-based renderer;
* :meth:`SparseCSB.to_hyperspy` and the module-level :func:`file_reader` - a
  RosettaSciIO-compatible signal dictionary, so the result drops straight into
  HyperSpy with calibrated navigation (time) and signal (y, x) axes.

The numeric core returns plain NumPy arrays and has **no** hard HyperSpy / dask
dependency; those are imported lazily only inside :meth:`to_hyperspy` /
:func:`file_reader` and only when you ask for a lazy signal.

Design note - why seconds map to whole frames
----------------------------------------------
A CSB frame is the atomic exposure; there is no sub-frame timing in the format.
So a time window ``[t0, t1)`` is realised as the half-open *frame* range whose
frames start within it. :meth:`time_to_frames` is the single place that mapping
lives, and every method routes through it, so the rounding rule is defined once.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np

from ._core import CSBFile, CSBAccumulator, bin_image

__all__ = ["SparseCSB", "load", "file_reader"]


class SparseCSB:
    """A lazy, time-indexed sparse dataset over one CSB file.

    Parameters
    ----------
    source : str or CSBFile
        A path to a ``.csb`` file, or an already-open :class:`csb.CSBFile`.
    backend : {"auto", "gpu", "cpu", "cpu-numba", "cpu-numpy"}
        Compute backend for the integrations; forwarded to
        :class:`csb.CSBAccumulator`. ``"auto"`` uses the GPU when available.

    Notes
    -----
    Opening is cheap - only the header and block table are read. The payload is
    memory-mapped and touched lazily, per integration, so a multi-gigabyte movie
    costs nothing until you actually integrate a window of it.

    A single accumulator is created once and reused (``reset`` + ``add_frames``)
    across calls, so repeated slider drags don't reallocate device memory.

    Examples
    --------
    >>> ds = SparseCSB("movie.csb")                     # doctest: +SKIP
    >>> img = ds.integrate(0.0, 0.1)                    # 0 - 0.1 s   # doctest: +SKIP
    >>> stack = ds.to_frames(0.05)                      # 50 ms/plane # doctest: +SKIP
    >>> y, x = ds.events_in_time(0.0, 0.01)             # COO view    # doctest: +SKIP
    """

    def __init__(self, source: Union[str, CSBFile], backend: str = "auto"):
        self.csb = source if isinstance(source, CSBFile) else CSBFile(source)
        self.backend = backend
        self._acc: Optional[CSBAccumulator] = None      # created on first use

    # -- time axis --------------------------------------------------------
    @property
    def frame_count(self) -> int:
        return self.csb.frame_count

    @property
    def frame_duration(self) -> float:
        """Seconds per frame (from the header ``microsec_per_frame``)."""
        return self.csb.microsec_per_frame * 1e-6

    @property
    def frame_rate(self) -> float:
        """Frames per second; 0.0 if the header has no cadence."""
        dt = self.frame_duration
        return 1.0 / dt if dt > 0 else 0.0

    @property
    def duration(self) -> float:
        """Total acquisition time in seconds (``frame_count * frame_duration``)."""
        return self.frame_count * self.frame_duration

    @property
    def time_axis(self) -> np.ndarray:
        """Start time (s) of every frame: ``[0, dt, 2dt, ...]``, length frame_count."""
        return np.arange(self.frame_count, dtype=np.float64) * self.frame_duration

    @property
    def shape(self) -> Tuple[int, int]:
        """(height, width) of a frame in pixels."""
        return self.csb.frame_height, self.csb.frame_width

    def _require_timed(self) -> None:
        if self.frame_duration <= 0:
            raise ValueError(
                "this file declares no frame cadence (microsec_per_frame is 0); "
                "address it in frame indices via integrate_frames() instead of "
                "seconds")

    def time_to_frames(self, t0: float, t1: float) -> Tuple[int, int]:
        """Map a half-open time window ``[t0, t1)`` in seconds to frames ``[f0, f1)``.

        A frame belongs to the window when *its start time* lies in ``[t0, t1)``.
        ``t1 = None``-style "to the end" is expressed by passing ``self.duration``
        (or any value past it); results are always clamped to the movie.
        """
        self._require_timed()
        if t1 <= t0:
            raise ValueError(f"empty time window [{t0}, {t1}) s")
        dt = self.frame_duration
        # Frame f starts at f*dt; it is in [t0, t1) iff t0 <= f*dt < t1.
        f0 = max(0, math.ceil(t0 / dt - 1e-9))
        f1 = min(self.frame_count, math.ceil(t1 / dt - 1e-9))
        if f1 <= f0:
            raise ValueError(
                f"time window [{t0}, {t1}) s selects no frames "
                f"(cadence {dt*1e3:.4g} ms/frame); window is narrower than one frame")
        return f0, f1

    # -- accumulator plumbing --------------------------------------------
    def _accumulator(self) -> CSBAccumulator:
        if self._acc is None:
            self._acc = CSBAccumulator(self.csb, backend=self.backend)
        return self._acc

    def _sum_frames(self, f0: int, f1: int, bin_factor: int,
                    dtype) -> np.ndarray:
        acc = self._accumulator()
        acc.reset()
        acc.add_frames(f0, f1)
        acc.synchronize()
        img = acc.image()
        if bin_factor > 1:
            img = bin_image(img, bin_factor)
        return img.astype(dtype) if dtype is not None else img

    # -- integration (seconds) -------------------------------------------
    def integrate(self, t0: float, t1: Optional[float] = None,
                  bin: int = 1, dtype=None) -> np.ndarray:
        """Sum all events in the time window ``[t0, t1)`` into one dense image.

        This is the viewer primitive: give it the current slider range in
        seconds and it returns the corresponding summed frame.

        Parameters
        ----------
        t0, t1 : float
            Window start/stop in seconds. ``t1=None`` means "to the end".
        bin : int
            Integer output binning (summed), applied after integration.
        dtype :
            Output dtype; ``None`` keeps the accumulator's integer counts.

        Returns
        -------
        ndarray
            2-D image, shape ``(H//bin, W//bin)``.
        """
        if t1 is None:
            t1 = self.duration
        f0, f1 = self.time_to_frames(t0, t1)
        return self._sum_frames(f0, f1, bin, dtype)

    def integrate_frames(self, f0: int, f1: Optional[int] = None,
                         bin: int = 1, dtype=None) -> np.ndarray:
        """Frame-indexed sibling of :meth:`integrate` (no cadence needed)."""
        if f1 is None:
            f1 = self.frame_count
        if not (0 <= f0 < f1 <= self.frame_count):
            raise ValueError(f"bad frame range [{f0}, {f1})")
        return self._sum_frames(f0, f1, bin, dtype)

    # -- dense stacks -----------------------------------------------------
    def _time_bounds(self, step: float, start: float,
                     stop: float) -> list:
        """Frame boundaries for consecutive `step`-second planes over [start, stop).

        Returns a **contiguous partition**: each plane ends exactly where the
        next begins, so every frame in the covered range lands in exactly one
        plane - no gaps, no overlaps - even when ``step`` is not a whole number
        of frames. (Mapping each plane's time window independently, as an
        earlier version did, let floating-point rounding at the boundaries put a
        frame in two adjacent planes or in neither.)

        A plane's frames are those whose *start time* falls in that plane's time
        window; the shared edge between plane i and i+1 is a single frame index,
        computed once, so it cannot be double-assigned. Planes that would be
        empty (a time step narrower than the frame cadence leaves some with no
        frame start) are dropped rather than emitted as zero-frame planes.
        """
        self._require_timed()
        if step <= 0:
            raise ValueError("step must be > 0 seconds")
        dt = self.frame_duration
        f_start = max(0, math.ceil(start / dt - 1e-9))
        f_stop = min(self.frame_count, math.ceil(stop / dt - 1e-9))
        if f_stop <= f_start:
            return []
        n_planes = int(math.ceil((stop - start) / step - 1e-9))
        # The frame index at each plane's leading time edge, clamped and made
        # monotone; consecutive edges form the [f0, f1) of each plane.
        edge_f = []
        for i in range(n_planes + 1):
            t_edge = start + i * step
            fe = math.ceil(t_edge / dt - 1e-9)
            edge_f.append(min(max(fe, f_start), f_stop))
        bounds = []
        for i in range(n_planes):
            f0, f1 = edge_f[i], edge_f[i + 1]
            if f1 > f0:                                 # drop empty planes
                bounds.append((f0, f1))
        return bounds

    def to_frames(self, step: float, start: float = 0.0,
                  stop: Optional[float] = None, bin: int = 1,
                  dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
        """Rebin the movie into a dense stack, one plane per ``step`` seconds.

        Parameters
        ----------
        step : float
            Exposure per output plane, in seconds (e.g. ``0.1`` -> 100 ms planes).
        start, stop : float
            Time range to cover; defaults to the whole movie.
        bin : int
            Integer spatial binning (summed) applied per plane.
        dtype :
            Output dtype (default float32, HyperSpy/MRC-friendly).

        Returns
        -------
        stack : ndarray
            3-D array ``(n_planes, H//bin, W//bin)``.
        plane_times : ndarray
            Start time (s) of each plane, length ``n_planes`` - the calibrated
            navigation axis for a viewer or for HyperSpy.
        """
        if stop is None:
            stop = self.duration
        bounds = self._time_bounds(step, start, stop)
        if not bounds:
            raise ValueError(
                f"no planes: step {step}s over [{start}, {stop})s selects nothing")
        h = self.csb.frame_height // bin
        w = self.csb.frame_width // bin
        stack = np.empty((len(bounds), h, w), dtype=dtype)
        for i, (f0, f1) in enumerate(bounds):
            stack[i] = self._sum_frames(f0, f1, bin, dtype)
        plane_times = np.array([f0 * self.frame_duration for f0, _ in bounds],
                               dtype=np.float64)
        return stack, plane_times

    # -- sparse / event (COO) view ---------------------------------------
    def events(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        """Decoded ``(y, x)`` int32 arrays for a single frame (COO-style view).

        Every returned pair is one detected event; repeated coordinates mean
        repeated counts. This is the raw material for an event-based renderer
        that draws points rather than an integrated image.
        """
        if not (0 <= frame < self.frame_count):
            raise ValueError(f"frame {frame} out of range [0, {self.frame_count})")
        return self.csb.frame_events(frame)

    def events_in_time(self, t0: float, t1: Optional[float] = None
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decoded ``(y, x, frame)`` for every event whose frame is in ``[t0, t1)``.

        Returns three parallel arrays: pixel ``y``, pixel ``x``, and the source
        frame index of each event (so a renderer can colour or fade by time).
        Concatenates per-frame decodes; for very wide windows prefer
        :meth:`integrate`, which never materialises the event list.
        """
        if t1 is None:
            t1 = self.duration
        f0, f1 = self.time_to_frames(t0, t1)
        ys, xs, fs = [], [], []
        for fr in range(f0, f1):
            y, x = self.csb.frame_events(fr)
            if y.size:
                ys.append(y)
                xs.append(x)
                fs.append(np.full(y.size, fr, dtype=np.int32))
        if not ys:
            z = np.empty(0, np.int32)
            return z, z, z
        return (np.concatenate(ys), np.concatenate(xs), np.concatenate(fs))

    # -- HyperSpy / RosettaSciIO bridge ----------------------------------
    def to_signal_dict(self, step: float, start: float = 0.0,
                       stop: Optional[float] = None, bin: int = 1,
                       dtype=np.float32) -> dict:
        """Build a RosettaSciIO signal dictionary for a ``to_frames`` stack.

        The returned dict has the standard keys - ``data``, ``axes``,
        ``metadata``, ``original_metadata`` - and can be handed to
        ``hyperspy.signals.Signal2D(**d)`` or returned from a rsciio reader.

        Axes: a navigation *time* axis (seconds) and two signal *y*/*x* axes
        calibrated in angstrom from the header ``ang_per_pix`` (scaled by bin).
        """
        stack, plane_times = self.to_frames(step, start, stop, bin, dtype)
        return self._signal_dict(stack, plane_times, step, bin)

    def _signal_dict(self, stack: np.ndarray, plane_times: np.ndarray,
                     step: float, bin: int) -> dict:
        cs = self.csb
        apix = cs.ang_per_pix * bin
        nav_scale = step if step > 0 else (self.frame_duration or 1.0)
        nav_offset = float(plane_times[0]) if plane_times.size else 0.0
        axes = [
            {"name": "time", "size": stack.shape[0], "index_in_array": 0,
             "scale": float(nav_scale), "offset": nav_offset,
             "units": "s", "navigate": True},
            {"name": "y", "size": stack.shape[1], "index_in_array": 1,
             "scale": float(apix) if apix > 0 else 1.0, "offset": 0.0,
             "units": "Å" if apix > 0 else "px", "navigate": False},
            {"name": "x", "size": stack.shape[2], "index_in_array": 2,
             "scale": float(apix) if apix > 0 else 1.0, "offset": 0.0,
             "units": "Å" if apix > 0 else "px", "navigate": False},
        ]
        info = cs.info()
        metadata = {
            "General": {
                "title": _basename_no_ext(cs.path),
                "original_filename": _basename(cs.path),
            },
            "Signal": {"signal_type": ""},
        }
        if info.get("datetime"):
            date, _, tm = info["datetime"].partition(" ")
            metadata["General"]["date"] = date
            metadata["General"]["time"] = tm
        return {
            "data": stack,
            "axes": axes,
            "metadata": metadata,
            "original_metadata": {"csb_header": info},
        }

    def to_hyperspy(self, step: float, start: float = 0.0,
                    stop: Optional[float] = None, bin: int = 1,
                    dtype=np.float32, lazy: bool = False):
        """Return a HyperSpy ``Signal2D`` of the ``to_frames`` stack.

        Requires ``hyperspy``. With ``lazy=True`` the data is wrapped in a dask
        array (requires ``dask``); the stack is still computed eagerly here -
        true per-plane lazy compute is a future refinement (see module docstring).
        """
        try:
            import hyperspy.api as hs
        except Exception as ex:                          # pragma: no cover
            raise ImportError(
                "to_hyperspy() needs hyperspy installed "
                "(`uv add hyperspy` or `pip install hyperspy`)") from ex
        d = self.to_signal_dict(step, start, stop, bin, dtype)
        if lazy:
            import dask.array as da
            d = dict(d)
            d["data"] = da.from_array(d["data"], chunks="auto")
            return hs.signals.Signal2D(**d).as_lazy()
        return hs.signals.Signal2D(**d)

    def __repr__(self) -> str:
        dt = self.frame_duration
        cad = f"{dt*1e3:.4g} ms/frame" if dt > 0 else "no cadence"
        return (f"<SparseCSB {self.csb.frame_width}x{self.csb.frame_height} "
                f"{self.frame_count}f  {self.duration*1e3:.4g} ms  {cad}  "
                f"{self.csb.n_events:,} events  backend={self.backend}>")


# --------------------------------------------------------------------------
#  Module-level helpers
# --------------------------------------------------------------------------
def _basename(path: str) -> str:
    import os
    return os.path.basename(path)


def _basename_no_ext(path: str) -> str:
    import os
    return os.path.splitext(os.path.basename(path))[0]


def load(source: Union[str, CSBFile], backend: str = "auto") -> SparseCSB:
    """Open a CSB file as a :class:`SparseCSB` (thin constructor alias)."""
    return SparseCSB(source, backend=backend)


def file_reader(filename: str, lazy: bool = False, step: Optional[float] = None,
                bin: int = 1, backend: str = "auto", dtype=np.float32,
                **kwds) -> list:
    """RosettaSciIO-style reader: return a list with one signal dictionary.

    Parameters
    ----------
    filename : str
        Path to the ``.csb`` file.
    lazy : bool
        If True, wrap ``data`` in a dask array (needs ``dask``).
    step : float, optional
        Seconds per output plane. Defaults to one plane per frame
        (``step = frame_duration``), i.e. the movie at native cadence.
    bin : int
        Integer spatial binning (summed) per plane.
    backend : str
        Compute backend forwarded to the accumulator.

    Returns
    -------
    list of dict
        A single-element list holding the signal dictionary, per the
        RosettaSciIO contract.
    """
    ds = SparseCSB(filename, backend=backend)
    if step is None:
        step = ds.frame_duration if ds.frame_duration > 0 else 1.0
    d = ds.to_signal_dict(step, bin=bin, dtype=dtype)
    if lazy:
        import dask.array as da
        d["data"] = da.from_array(d["data"], chunks="auto")
    return [d]
