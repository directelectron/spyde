"""Fast reader and frame accumulator for Direct Electron CSB centroid-streaming files.

Drop-in-compatible with the reference ``de_csb.py`` for the summation path, but
built for real-time ingest. Verified to reproduce the reference reader's output
bit-identically (0 differing pixels of 67,108,864 on the Apollo XS test movie).

Real-time context
-----------------
At 390 us/frame and ~314,546 events/frame the acquisition rate is roughly
807 M events/s (1.61 GB/s of payload). Measured on an RTX 3050 (a deliberately
weak proxy for production hardware):

    reference de_csb.py .................     0.5 M ev/s    0.0006x real-time
    this module, CPU (numpy) backend .....    19-28 M ev/s   0.02-0.04x
    this module, CPU (numpy, block-local)    50-70 M ev/s   0.06-0.09x
    this module, CPU (numba, multithread)   >800 M ev/s    real-time-class*
    this module, GPU, payload in host RAM   ~2,800 M ev/s    3.5x  (PCIe-bound)
    this module, GPU, payload in VRAM ....  13,140 M ev/s   16.3x

The host-RAM path is limited by PCIe transfer (measured 5.65 GB/s pinned), not
by the kernel, so handing this class device-resident data is worth roughly 5x.

* The numba CPU path scales with core count and clears the real-time bar on a
  many-core host once the ~1 s JIT warm-up is paid (on first ``add_frames``, or
  reloaded from numba's on-disk cache). It is the path to use when no GPU is
  present but sustained ingest matters; for a single one-shot conversion the
  warm-up can dominate and the block-local numpy path is the better default.
  All three CPU paths are bit-identical to one another and to the GPU path.

File format
-----------
Little-endian throughout. A 108-byte metadata header (padded out to
``csb_data_offset``), then one uint16 per event, then a footer of one uint16
event-count per block.

A frame is tiled by ``csb_block_width`` x ``csb_block_height`` blocks. Each
event is a single uint16 giving its raster position *inside* its own block::

    y = block_y_origin + word // csb_block_width
    x = block_x_origin + word %  csb_block_width

Block ordering is given by the ``csb_order`` header field (0 = column-major,
1 = row-major). Events arrive in readout (time) order, which is ascending
raster order within a readout pass; a block typically spans one to four passes,
so its words form a few ascending runs rather than one sorted list. Nothing
here depends on that ordering.

Why it is fast
--------------
Because the payload word is *already* the raster index inside its block, an
accumulator stored BLOCK-MAJOR - one contiguous ``block_w*block_h`` tile per
block - needs no coordinate arithmetic at all: the scatter target is simply
``acc[block_index * tile_size + word]``. Every atomic issued by one thread
block then lands inside a single contiguous tile (256 KB for 2048x32), which
stays resident in L2. One transpose at readout restores normal image layout.

See README.md for the measured tuning study behind ``FRAMES_PER_GROUP`` and
``THREADS_PER_BLOCK``.
"""

from __future__ import annotations

import math
import os
import struct
from contextlib import nullcontext as _nullcontext
from typing import Optional, Sequence, Tuple

import numpy as np

try:                                                  # optional GPU backend
    import cupy as _cp
except Exception:                                     # pragma: no cover
    _cp = None

try:                                                  # optional JIT CPU backend
    import numba as _numba
except Exception:                                     # pragma: no cover
    _numba = None

__all__ = ["CSBFile", "CSBAccumulator", "sum_file", "write_fraction_stack",
           "bin_image", "gpu_available", "numba_available"]

CSB_MAGIC = 13240
_HEADER_FIELDS_BYTES = 108


def gpu_available() -> bool:
    """True if a working CuPy/CUDA device is present."""
    if _cp is None:
        return False
    try:
        _cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def numba_available() -> bool:
    """True if Numba is importable (used by the fast multithreaded CPU path)."""
    return _numba is not None


# --------------------------------------------------------------------------
#  Header / container
# --------------------------------------------------------------------------
class CSBFile:
    """Parses a CSB file's header and block table and memory-maps the payload.

    Nothing is read eagerly beyond the header and the block-length table, so
    opening a multi-gigabyte movie is cheap.
    """

    def __init__(self, path: str):
        self.path = path
        self.size = os.path.getsize(path)
        if self.size < _HEADER_FIELDS_BYTES + 2:
            raise ValueError(f"{path}: too short to be a CSB file")

        with open(path, "rb") as fh:
            h = fh.read(_HEADER_FIELDS_BYTES)
        if len(h) < _HEADER_FIELDS_BYTES:
            raise ValueError(f"{path}: truncated header")

        u16 = lambda o: struct.unpack_from("<H", h, o)[0]      # noqa: E731
        u32 = lambda o: struct.unpack_from("<I", h, o)[0]      # noqa: E731
        u64 = lambda o: struct.unpack_from("<Q", h, o)[0]      # noqa: E731
        f32 = lambda o: struct.unpack_from("<f", h, o)[0]      # noqa: E731

        self.file_specifier = u16(0)
        if self.file_specifier != CSB_MAGIC:
            raise ValueError(f"{path}: not a CSB file "
                             f"(specifier {self.file_specifier}, expected {CSB_MAGIC})")
        self.file_version = u16(2)
        self.frame_width = u16(4)
        self.frame_height = u16(6)
        self.frame_count = u32(8)
        self.ang_per_pix = f32(12)
        self.microsec_per_frame = f32(16)
        self.csb_block_width = u16(20)
        self.csb_block_height = u16(22)
        self.csb_data_offset = u64(24)
        self.csb_lengths_offset = u64(32)
        self.csb_order = u16(40)              # 0 = column-major, 1 = row-major
        self.camera_sn = u16(42)
        self.firmware = tuple(u16(44 + 2 * i) for i in range(4))
        self.software = tuple(u16(52 + 2 * i) for i in range(4))
        self.superres_factor = u16(60)
        self.microscope_kv = u16(62)
        self.microscope_mode = u16(64)
        self.microscope_spot = u16(66)
        self.microscope_mag = f32(68)
        self.microscope_defocus = f32(72)
        self.position = (f32(76), f32(80), f32(84))
        self.stage_tilt = f32(88)
        self.beam_shift = (f32(92), f32(96))
        self.timestamp = u64(100)

        for name, val in (("frame_width", self.frame_width),
                          ("frame_height", self.frame_height),
                          ("frame_count", self.frame_count),
                          ("csb_block_width", self.csb_block_width),
                          ("csb_block_height", self.csb_block_height)):
            if val < 1:
                raise ValueError(f"{path}: invalid {name} ({val})")

        # Block grid. Edge blocks may be partial; they keep the full stride, so
        # the accumulator is allocated for the padded frame and cropped at readout.
        self.blocks_per_width = math.ceil(self.frame_width / self.csb_block_width)
        self.blocks_per_height = math.ceil(self.frame_height / self.csb_block_height)
        self.blocks_per_frame = self.blocks_per_width * self.blocks_per_height
        self.tile_size = self.csb_block_width * self.csb_block_height
        self.padded_width = self.blocks_per_width * self.csb_block_width
        self.padded_height = self.blocks_per_height * self.csb_block_height

        n_table = self.frame_count * self.blocks_per_frame
        need = self.csb_lengths_offset + n_table * 2
        if need > self.size:
            raise ValueError(f"{path}: block table is incomplete "
                             f"(needs {need} bytes, file is {self.size})")

        self.counts = np.asarray(
            np.memmap(path, dtype="<u2", mode="r",
                      offset=self.csb_lengths_offset, shape=(n_table,)),
            dtype=np.int64)
        self.n_events = int(self.counts.sum())

        payload_words = (self.csb_lengths_offset - self.csb_data_offset) // 2
        if self.n_events != payload_words:
            raise ValueError(
                f"{path}: block table sums to {self.n_events:,} events but the "
                f"payload region holds {payload_words:,} words")

        self.events = np.memmap(path, dtype="<u2", mode="r",
                                offset=self.csb_data_offset, shape=(self.n_events,))
        self._ends = np.cumsum(self.counts)
        self.starts = (self._ends - self.counts).astype(np.int64)

    # -- geometry ---------------------------------------------------------
    def block_origin(self, block_index: int) -> Tuple[int, int]:
        """(y, x) pixel origin of a block index within a frame."""
        if self.csb_order == 1:                       # row-major
            by, bx = divmod(block_index, self.blocks_per_width)
        else:                                         # column-major (default)
            bx, by = divmod(block_index, self.blocks_per_height)
        return by * self.csb_block_height, bx * self.csb_block_width

    def frame_slice(self, f0: int, f1: int) -> Tuple[int, int]:
        """(first_word, n_words) of the payload for frames [f0, f1)."""
        a = f0 * self.blocks_per_frame
        b = f1 * self.blocks_per_frame
        s = int(self.starts[a])
        e = int(self._ends[b - 1])
        return s, e - s

    def events_in_range(self, f0: int, f1: int) -> int:
        """Number of events in frames [f0, f1)."""
        return int(self.counts[f0 * self.blocks_per_frame:
                               f1 * self.blocks_per_frame].sum())

    def frame_events(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        """Decoded (y, x) int32 arrays for one frame, in readout order.

        Convenience/reference path - not the fast path.
        """
        a = frame * self.blocks_per_frame
        b = a + self.blocks_per_frame
        s, e = int(self.starts[a]), int(self._ends[b - 1])
        w = np.asarray(self.events[s:e], dtype=np.int32)
        cnt = self.counts[a:b]
        oy = np.empty(self.blocks_per_frame, np.int32)
        ox = np.empty(self.blocks_per_frame, np.int32)
        for i in range(self.blocks_per_frame):
            oy[i], ox[i] = self.block_origin(i)
        y = np.repeat(oy, cnt) + (w // self.csb_block_width)
        x = np.repeat(ox, cnt) + (w % self.csb_block_width)
        return y, x

    def info(self) -> dict:
        ts = self.timestamp / 1000.0
        import datetime
        return {
            "path": self.path,
            "frame_width": self.frame_width, "frame_height": self.frame_height,
            "frame_count": self.frame_count, "ang_per_pix": self.ang_per_pix,
            "microsec_per_frame": self.microsec_per_frame,
            "csb_block_width": self.csb_block_width,
            "csb_block_height": self.csb_block_height,
            "csb_order": "row-major" if self.csb_order == 1 else "column-major",
            "blocks_per_frame": self.blocks_per_frame,
            "n_events": self.n_events,
            "camera_sn": self.camera_sn, "superres_factor": self.superres_factor,
            "microscope_kv": self.microscope_kv, "mag": self.microscope_mag,
            "defocus_um": self.microscope_defocus,
            "timestamp": self.timestamp,
            "datetime": datetime.datetime.fromtimestamp(ts).isoformat(sep=" "),
        }

    def __repr__(self):
        return (f"<CSBFile {os.path.basename(self.path)} "
                f"{self.frame_width}x{self.frame_height} {self.frame_count}f "
                f"{self.n_events:,} events "
                f"blocks {self.csb_block_width}x{self.csb_block_height}>")


# --------------------------------------------------------------------------
#  GPU kernel
# --------------------------------------------------------------------------
# blockIdx.x is scheduled fastest, so it MUST be the frame group and
# blockIdx.y the tile. Concurrent thread blocks then share one tile, which
# stays hot in L2. Putting the tile on .x instead costs ~6x (measured).
_KERNEL_SRC = r'''
extern "C" __global__
void csb_accumulate(const unsigned short* __restrict__ words,
                    long long word_base,
                    const long long*      __restrict__ starts,
                    const int*            __restrict__ counts,
                    int* acc, int blocks_per_frame, int first_frame,
                    int n_frames, int frames_per_group, int tile_size)
{
    const int b  = blockIdx.y;
    const int g0 = blockIdx.x * frames_per_group;
    const int g1 = min(g0 + frames_per_group, n_frames);
    int* tile = acc + (long long)b * tile_size;

    for (int f = g0; f < g1; ++f) {
        const int k = (first_frame + f) * blocks_per_frame + b;
        const int n = counts[k];
        if (n <= 0) continue;
        // starts[] hold absolute file word offsets; word_base rebases them
        // onto `words`. Done here rather than in Python so no temporary array
        // is created on a different stream than the one this kernel runs on.
        const unsigned short* p = words + (starts[k] - word_base);
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            atomicAdd(&tile[p[i]], 1);
    }
}
'''

# Measured optimum on an RTX 3050 for 2048x32 blocks; these two interact
# strongly (at 25 frames/group the best thread count is 1024, not 256), so
# re-tune them together on new hardware. See README.md.
FRAMES_PER_GROUP = 4
THREADS_PER_BLOCK = 256

_kernel_cache = {}


def _get_kernel():
    if "k" not in _kernel_cache:
        if _cp is None:
            raise RuntimeError("CuPy is not installed; GPU backend unavailable")
        _kernel_cache["k"] = _cp.RawModule(code=_KERNEL_SRC).get_function(
            "csb_accumulate")
    return _kernel_cache["k"]


# --------------------------------------------------------------------------
#  Numba CPU kernel
# --------------------------------------------------------------------------
# Same block-major idea as the GPU kernel, mapped onto CPU threads. Each thread
# owns a disjoint set of spatial blocks and walks them through every frame,
# writing only into that block's own tile. Because tiles are disjoint slices of
# `acc`, threads never touch the same memory - no atomics, no privatisation, no
# reduction. The scatter for one block stays inside its `tile_size`-word tile
# (256 KB for 2048x32), which is L2-resident, mirroring the GPU's L2 trick.
#
# Built lazily so `import csb` costs nothing when Numba is absent, and so the
# ~1 s JIT warm-up is paid once, on first use, not at import.
_numba_kernel_cache = {}


def _get_numba_kernel():
    if "k" not in _numba_kernel_cache:
        if _numba is None:
            raise RuntimeError("Numba is not installed; numba backend unavailable")
        from numba import njit, prange

        @njit(parallel=True, cache=True, nogil=True, fastmath=False)
        def _accumulate(words, base_word, starts, counts, acc,
                        blocks_per_frame, first_frame, n_frames, tile_size):
            # One parallel iteration per spatial block. Blocks map to disjoint
            # tiles in `acc`, so iterations are independent - safe to run on
            # separate threads with no synchronisation.
            for b in prange(blocks_per_frame):
                base = b * tile_size
                for f in range(n_frames):
                    k = (first_frame + f) * blocks_per_frame + b
                    n = counts[k]
                    if n <= 0:
                        continue
                    p = starts[k] - base_word
                    for i in range(n):
                        acc[base + words[p + i]] += 1

        _numba_kernel_cache["k"] = _accumulate
    return _numba_kernel_cache["k"]


# --------------------------------------------------------------------------
#  Accumulator
# --------------------------------------------------------------------------
class CSBAccumulator:
    """Accumulates CSB events into a summed image.

    The accumulator lives in block-major layout for the whole lifetime of the
    object and is only rearranged into image layout when you ask for it, so a
    live view can keep adding frames without ever paying for a full readout.

    Parameters
    ----------
    csb : CSBFile
    backend : {"auto", "gpu", "cpu", "cpu-numba", "cpu-numpy"}
        "auto" selects the GPU when CuPy and a device are available, otherwise
        the fastest CPU path. "cpu" uses the Numba kernel when Numba is
        installed and falls back to pure NumPy otherwise. "cpu-numba" and
        "cpu-numpy" force one specific CPU path (mainly for benchmarking and
        tests); all three CPU paths produce bit-identical results.
    dtype :
        Accumulator element type. int32 is plenty for counting (the busiest
        pixel in a 400-frame Apollo XS movie sees ~100 events).

    Examples
    --------
    >>> f = CSBFile("movie.csb")                        # doctest: +SKIP
    >>> acc = CSBAccumulator(f)                         # doctest: +SKIP
    >>> acc.add_frames(0, f.frame_count)                # doctest: +SKIP
    >>> img = acc.image()                               # doctest: +SKIP
    """

    def __init__(self, csb: CSBFile, backend: str = "auto", dtype=np.int32):
        self.csb = csb
        self.dtype = dtype
        if backend == "auto":
            backend = "gpu" if gpu_available() else "cpu"
        if backend == "cpu":                          # pick the best CPU path
            backend = "cpu-numba" if numba_available() else "cpu-numpy"
        if backend == "gpu" and not gpu_available():
            raise RuntimeError("GPU backend requested but no CUDA device is available")
        if backend == "cpu-numba" and not numba_available():
            raise RuntimeError("cpu-numba backend requested but Numba is not installed")
        if backend not in ("gpu", "cpu-numba", "cpu-numpy"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        # True for any CPU path; the GPU path is the sole special case elsewhere.
        self._on_gpu = backend == "gpu"

        n = csb.blocks_per_frame * csb.tile_size
        if backend == "gpu":
            self._acc = _cp.zeros(n, dtype=dtype)
            self._d_counts = _cp.asarray(csb.counts.astype(np.int32))
            self._d_starts = _cp.asarray(csb.starts)
            self._kernel = _get_kernel()
        elif backend == "cpu-numba":
            self._acc = np.zeros(n, dtype=np.int64)
            self._kernel = _get_numba_kernel()        # triggers JIT on first use
        else:
            self._acc = np.zeros(n, dtype=np.int64)   # bincount output dtype
        # Device staging buffers for host payloads, held until the work that
        # reads them has completed. Without this the buffer could be returned
        # to CuPy's pool and reused while an async copy/kernel is still pending.
        self._pending: list = []
        self.events_added = 0

    # -- ingest -----------------------------------------------------------
    def add_frames(self, f0: int, f1: int, payload=None, stream=None) -> None:
        """Accumulate frames [f0, f1).

        Parameters
        ----------
        payload :
            ``None``  - read from the file's memory map (offline use).
            ndarray   - host uint16 array holding *exactly* the payload for
                        these frames. Use pinned memory for best transfer rate.
            cupy array- device-resident payload for these frames; no transfer
                        is performed. This is the fastest path by ~5x.
        stream :
            Optional CuPy stream. Work is enqueued and NOT synchronised, so you
            can overlap successive calls; call :meth:`synchronize` before
            reading results.
        """
        if not (0 <= f0 < f1 <= self.csb.frame_count):
            raise ValueError(f"bad frame range [{f0}, {f1}) for a "
                             f"{self.csb.frame_count}-frame movie")
        base_word, n_words = self.csb.frame_slice(f0, f1)

        if payload is None:
            payload = np.asarray(self.csb.events[base_word:base_word + n_words])
        if len(payload) != n_words:
            raise ValueError(f"payload has {len(payload):,} words, expected "
                             f"{n_words:,} for frames [{f0}, {f1})")

        if self._on_gpu:
            self._add_gpu(f0, f1, payload, base_word, stream)
        elif self.backend == "cpu-numba":
            self._add_numba(f0, f1, payload, base_word)
        else:
            self._add_cpu(f0, f1, payload, base_word)
        self.events_added += int(n_words)

    def _add_gpu(self, f0, f1, payload, base_word, stream):
        cs = self.csb
        # Everything - staging allocation, transfer and kernel - must be issued
        # inside the same stream context, or work lands on the default stream
        # and races with the kernel.
        ctx = stream if stream is not None else _nullcontext()
        with ctx:
            if isinstance(payload, np.ndarray):        # host -> device
                d_words = _cp.empty(len(payload), dtype=_cp.uint16)
                if stream is not None:
                    _cp.cuda.runtime.memcpyAsync(
                        d_words.data.ptr, payload.ctypes.data, payload.nbytes,
                        _cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
                else:
                    d_words.set(payload)
                self._pending.append(d_words)          # released by synchronize()
            else:                                      # already device-resident
                d_words = payload

            n_frames = f1 - f0
            grid = ((n_frames + FRAMES_PER_GROUP - 1) // FRAMES_PER_GROUP,
                    cs.blocks_per_frame)
            self._kernel(grid, (THREADS_PER_BLOCK,), (
                d_words, np.int64(base_word), self._d_starts, self._d_counts,
                self._acc, np.int32(cs.blocks_per_frame), np.int32(f0),
                np.int32(n_frames), np.int32(FRAMES_PER_GROUP),
                np.int32(cs.tile_size)))

    def _add_cpu(self, f0, f1, payload, base_word):
        """Pure-NumPy block-local histogram.

        The event word is already the intra-block raster index, so each spatial
        block's events histogram directly into that block's own `tile_size`-word
        slice of the accumulator. Doing it per block keeps every ``bincount``
        working set inside one L2-resident tile and, crucially, never builds the
        full-accumulator-sized index array or temporary the naive one-shot
        ``bincount`` needs (that temp is ~536 MB for an 8192^2 frame and is
        touched three times per call regardless of event count).

        All of a block's events across the whole chunk are gathered once so the
        per-block Python overhead is paid ``blocks_per_frame`` times, not once
        per (block, frame).
        """
        cs = self.csb
        bpf = cs.blocks_per_frame
        tile_size = cs.tile_size
        n_frames = f1 - f0
        counts = cs.counts[f0 * bpf:f1 * bpf].reshape(n_frames, bpf)
        # Per-block start offset into `payload` for every (frame, block), from a
        # single cumulative sum of the counts in payload (readout) order.
        ends = np.cumsum(counts.reshape(-1))
        starts = ends - counts.reshape(-1)
        starts = starts.reshape(n_frames, bpf)
        ends = ends.reshape(n_frames, bpf)
        words = np.asarray(payload)
        acc = self._acc
        for b in range(bpf):
            # Gather this spatial block's events from every frame in the chunk.
            if n_frames == 1:
                seg = words[starts[0, b]:ends[0, b]]
            else:
                parts = [words[starts[f, b]:ends[f, b]] for f in range(n_frames)]
                seg = np.concatenate(parts) if parts else words[:0]
            if seg.size:
                acc[b * tile_size:(b + 1) * tile_size] += np.bincount(
                    seg, minlength=tile_size)

    def _add_numba(self, f0, f1, payload, base_word):
        """Multithreaded JIT histogram; see :func:`_get_numba_kernel`."""
        cs = self.csb
        words = np.ascontiguousarray(payload, dtype=np.uint16)
        # `starts` are absolute file word offsets; the kernel rebases them onto
        # `words` with `base_word`, matching the GPU kernel exactly.
        self._kernel(words, np.int64(base_word), cs.starts, cs.counts, self._acc,
                     np.int64(cs.blocks_per_frame), np.int64(f0),
                     np.int64(f1 - f0), np.int64(cs.tile_size))

    def synchronize(self, stream=None) -> None:
        """Wait for outstanding GPU work and release staging buffers.

        Pass the stream(s) you enqueued on, or nothing to wait on the whole
        device. Staging buffers for host payloads are only freed here, so call
        it periodically in a long-running ingest loop.
        """
        if not self._on_gpu:
            return
        if stream is not None:
            stream.synchronize()
        else:
            _cp.cuda.Device().synchronize()
        self._pending.clear()

    def reset(self) -> None:
        self.synchronize()
        self._acc.fill(0)
        self.events_added = 0

    # -- readout ----------------------------------------------------------
    def _untile(self):
        """Block-major accumulator -> padded 2-D image, still on-device."""
        cs = self.csb
        bh, bw = cs.csb_block_height, cs.csb_block_width
        if cs.csb_order == 1:                              # row-major
            v = self._acc.reshape(cs.blocks_per_height, cs.blocks_per_width, bh, bw)
            v = v.transpose(0, 2, 1, 3)
        else:                                              # column-major
            v = self._acc.reshape(cs.blocks_per_width, cs.blocks_per_height, bh, bw)
            v = v.transpose(1, 2, 0, 3)
        return v.reshape(cs.padded_height, cs.padded_width)

    def image(self, dtype=None) -> np.ndarray:
        """The summed image as a host ndarray, cropped to the true frame size.

        Costs a full device-to-host copy (~90 ms for 8192^2 on an RTX 3050).
        For a live view prefer :meth:`preview`, which is ~5x cheaper.
        """
        cs = self.csb
        v = self._untile()[:cs.frame_height, :cs.frame_width]
        if self._on_gpu:
            out = _cp.asnumpy(_cp.ascontiguousarray(v))
        else:
            out = np.ascontiguousarray(v)
        return out if dtype is None else out.astype(dtype)

    def preview(self, bin_factor: int = 4, dtype=np.float32) -> np.ndarray:
        """Binned preview for live display; the binning happens on-device.

        At bin_factor=4 an 8192^2 accumulator returns 2048^2 in ~18 ms,
        i.e. a sustainable ~55 Hz display rate.
        """
        if bin_factor < 1:
            raise ValueError("bin_factor must be >= 1")
        cs = self.csb
        h = (cs.frame_height // bin_factor) * bin_factor
        w = (cs.frame_width // bin_factor) * bin_factor
        v = self._untile()[:h, :w]
        xp = _cp if self._on_gpu else np
        v = xp.ascontiguousarray(v).reshape(h // bin_factor, bin_factor,
                                            w // bin_factor, bin_factor)
        out = v.sum(axis=(1, 3), dtype=xp.int64).astype(dtype)
        return _cp.asnumpy(out) if self._on_gpu else out

    def apply_corrections(self, image: np.ndarray,
                          gain: Optional[np.ndarray] = None,
                          defects: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply gain and defect masking to a summed image.

        Both are exactly equivalent to the reference reader's per-event
        handling and far cheaper: gain and the defect mask depend only on
        (y, x), so scaling or zeroing the *sum* gives an identical result to
        scaling or skipping each event.

        Note this deliberately does NOT reproduce ``de_csb.py``'s defect
        *filling*, which synthesises random events on flagged pixels at the
        local block sparsity. That fabricates data and has no place in a
        quantitative path; do it in the display layer if you want it.
        """
        out = image
        if gain is not None:
            if gain.shape != out.shape:
                raise ValueError(f"gain shape {gain.shape} != image {out.shape}")
            out = out.astype(np.float32) * gain.astype(np.float32)
        if defects is not None:
            if defects.shape != out.shape:
                raise ValueError(f"defects shape {defects.shape} != image {out.shape}")
            out = out.copy()
            out[defects.astype(bool)] = 0
        return out


# --------------------------------------------------------------------------
#  Convenience
# --------------------------------------------------------------------------
def bin_image(img: np.ndarray, factor: int) -> np.ndarray:
    """Bin an image by an integer factor, summing counts within each block.

    Summing (not averaging) is correct for count data: a physical pixel's count
    is the sum of the super-resolution sub-pixels it contains. Edge rows/columns
    that don't fill a full block are cropped, matching the readout convention.
    """
    if factor < 1:
        raise ValueError("bin factor must be >= 1")
    if factor == 1:
        return img
    h = (img.shape[0] // factor) * factor
    w = (img.shape[1] // factor) * factor
    return img[:h, :w].reshape(h // factor, factor,
                               w // factor, factor).sum(axis=(1, 3))


def write_fraction_stack(csb, out_path: str, frames_per_fraction: int,
                         frames: Optional[Sequence[int]] = None,
                         bin_factor: int = 1,
                         gain: Optional[np.ndarray] = None,
                         defects: Optional[np.ndarray] = None,
                         backend: str = "auto") -> list:
    """Write an MRC stack of dose-fraction sums, one plane per group of frames.

    Each output plane is the sum of ``frames_per_fraction`` consecutive CSB
    frames. This is the movie you hand to an external motion estimator
    (cryoSPARC / MotionCor / RELION); the trajectories it returns drive the
    motion-correction step.

    Output is at native (super-resolution) sampling by default. ``bin_factor``
    reduces it as a final step only - super-res is the point of centroid data.

    Parameters
    ----------
    csb : CSBFile or str
    out_path : str
        Destination MRC. Written as a 3-D float32 stack, streamed plane by
        plane, with ``voxel_size`` set from the header (scaled by bin_factor).
    frames_per_fraction : int
        CSB frames summed into each output plane.
    frames : (start, end), optional
        Frame range; default all frames. A trailing partial fraction is kept.
    bin_factor : int
        Integer output binning, summed. 1 = native super-resolution (default).
    gain, defects : ndarray, optional
        Applied per plane, at full resolution, before binning.

    Returns
    -------
    list of (f0, f1)
        Frame boundaries for each output plane, in order. Piece 2 needs these
        to map returned trajectories back onto individual frames.
    """
    import mrcfile

    if not isinstance(csb, CSBFile):
        csb = CSBFile(csb)
    if frames_per_fraction < 1:
        raise ValueError("frames_per_fraction must be >= 1")
    if bin_factor < 1:
        raise ValueError("bin_factor must be >= 1")

    f0, f1 = (0, csb.frame_count) if frames is None else frames
    f0 = max(0, f0)
    f1 = csb.frame_count if f1 < 0 else min(f1, csb.frame_count)
    if f1 <= f0:
        raise ValueError(f"empty frame range [{f0}, {f1})")

    bounds = [(a, min(a + frames_per_fraction, f1))
              for a in range(f0, f1, frames_per_fraction)]
    h_out = (csb.frame_height // bin_factor)
    w_out = (csb.frame_width // bin_factor)

    acc = CSBAccumulator(csb, backend=backend)
    apix = csb.ang_per_pix * bin_factor
    with mrcfile.new_mmap(out_path, shape=(len(bounds), h_out, w_out),
                          mrc_mode=2, overwrite=True) as m:
        for i, (a, b) in enumerate(bounds):
            acc.reset()
            acc.add_frames(a, b)
            acc.synchronize()
            img = acc.image(dtype=np.float32)
            if gain is not None or defects is not None:
                img = acc.apply_corrections(img, gain=gain, defects=defects)
            m.data[i] = bin_image(img, bin_factor).astype(np.float32)
        if apix > 0:
            m.voxel_size = apix
        label = (f"CSB dose-fraction stack: {frames_per_fraction} frames/plane"
                 + (f", bin {bin_factor}" if bin_factor > 1 else ""))
        m.header.nlabl = 1
        m.header.label[0] = label.encode("ascii")[:80]
    return bounds


def sum_file(path: str, frames: Optional[Sequence[int]] = None,
             backend: str = "auto", chunk_frames: int = 20,
             n_streams: int = 3) -> np.ndarray:
    """Sum a whole CSB file (or a frame range) into an image.

    On the GPU backend this stages through pinned host buffers across several
    CUDA streams so the file read overlaps the transfer and the kernel. Reading
    from the memory map straight into a device buffer instead is ~8x slower.
    """
    csb = CSBFile(path)
    f0, f1 = (0, csb.frame_count) if frames is None else frames
    f0 = max(0, f0)
    f1 = csb.frame_count if f1 < 0 else min(f1, csb.frame_count)
    if f1 <= f0:
        raise ValueError(f"empty frame range [{f0}, {f1})")
    acc = CSBAccumulator(csb, backend=backend)

    bounds = [(a, min(a + chunk_frames, f1)) for a in range(f0, f1, chunk_frames)]
    if not acc._on_gpu:                       # any CPU path: plain sequential add
        for a, b in bounds:
            acc.add_frames(a, b)
        return acc.image()

    import threading
    from concurrent.futures import ThreadPoolExecutor

    nmax = max(csb.frame_slice(a, b)[1] for a, b in bounds)
    streams = [_cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)]

    # A ring of pinned buffers, deeper than the stream count so a reader rarely
    # has to wait. Reads run on worker threads (readinto releases the GIL), so
    # file I/O overlaps the transfer and the kernel. NVMe needs several
    # concurrent readers to reach full bandwidth.
    ring = max(6, 2 * n_streams)
    ring = min(ring, len(bounds))
    staging = []
    for _ in range(ring):
        mem = _cp.cuda.alloc_pinned_memory(nmax * 2)
        staging.append(np.frombuffer(mem, dtype=np.uint16, count=nmax))
    slot_ready = [None] * ring          # event: prior GPU use of this slot done
    local = threading.local()

    def fetch(j):
        slot = j % ring
        ev = slot_ready[slot]
        if ev is not None:
            ev.synchronize()
        if not hasattr(local, "fh"):
            local.fh = open(path, "rb", buffering=0)
        s, n = csb.frame_slice(*bounds[j])
        local.fh.seek(csb.csb_data_offset + 2 * s)
        local.fh.readinto(memoryview(staging[slot])[:n].cast("B"))
        return n

    n_readers = min(6, max(2, ring // 2))
    ex = ThreadPoolExecutor(max_workers=n_readers)
    try:
        futs = {j: ex.submit(fetch, j) for j in range(min(ring, len(bounds)))}
        for j, (a, b) in enumerate(bounds):
            n = futs.pop(j).result()
            slot, sl = j % ring, j % n_streams
            acc.add_frames(a, b, payload=staging[slot][:n], stream=streams[sl])
            ev = _cp.cuda.Event()
            ev.record(streams[sl])
            slot_ready[slot] = ev
            nxt = j + ring
            if nxt < len(bounds):
                futs[nxt] = ex.submit(fetch, nxt)
    finally:
        ex.shutdown(wait=True)
    for st in streams:
        st.synchronize()
    acc.synchronize()
    return acc.image()


def _load_mrc(path):
    import mrcfile
    with mrcfile.open(path, permissive=True) as m:
        return np.asarray(m.data)


def _main(argv=None):
    import argparse
    import time
    p = argparse.ArgumentParser(
        description="Convert a CSB file to an MRC image or dose-fraction stack.")
    p.add_argument("input_csb")
    p.add_argument("output_mrc", nargs="?")
    p.add_argument("--frames", type=int, nargs=2, default=[0, -1],
                   metavar=("START", "END"), help="frame range [start end)")
    p.add_argument("--fractions", type=int, metavar="N",
                   help="write an MRC stack, summing N CSB frames per plane "
                        "(for feeding to a motion estimator)")
    p.add_argument("--bin", type=int, default=1, metavar="B",
                   help="integer output binning, summed (default 1 = native "
                        "super-resolution)")
    p.add_argument("--backend",
                   choices=["auto", "gpu", "cpu", "cpu-numba", "cpu-numpy"],
                   default="auto")
    p.add_argument("--gain", help="MRC gain reference")
    p.add_argument("--defects", help="MRC defect map (non-zero = bad pixel)")
    p.add_argument("--info", action="store_true", help="print header and exit")
    a = p.parse_args(argv)

    csb = CSBFile(a.input_csb)
    for k, v in csb.info().items():
        print(f"  {k}: {v}")
    if a.info:
        return 0

    f0 = max(0, a.frames[0])
    f1 = csb.frame_count if a.frames[1] < 0 else min(a.frames[1], csb.frame_count)
    gain = _load_mrc(a.gain) if a.gain else None
    defects = _load_mrc(a.defects) if a.defects else None

    # --- dose-fraction stack -------------------------------------------------
    if a.fractions:
        if not a.output_mrc:
            p.error("--fractions requires an output_mrc path")
        t0 = time.perf_counter()
        bounds = write_fraction_stack(csb, a.output_mrc, a.fractions,
                                      frames=[f0, f1], bin_factor=a.bin,
                                      gain=gain, defects=defects,
                                      backend=a.backend)
        dt = time.perf_counter() - t0
        print(f"\n  wrote {len(bounds)}-plane stack "
              f"({a.fractions} frames/plane, bin {a.bin}) to {a.output_mrc} "
              f"in {dt*1000:.0f} ms")
        return 0

    # --- single summed image -------------------------------------------------
    t0 = time.perf_counter()
    img = sum_file(a.input_csb, [f0, f1], backend=a.backend)
    dt = time.perf_counter() - t0
    n_ev = csb.events_in_range(f0, f1)
    acq = (f1 - f0) * csb.microsec_per_frame * 1e-6
    print(f"\n  frames [{f0}, {f1}) : summed {n_ev:,} events in {dt*1000:.1f} ms "
          f"({n_ev/dt/1e6:,.0f} M events/s)")
    if acq > 0:
        print(f"  acquisition was {acq*1000:.0f} ms -> {acq/dt:.2f}x real-time "
              f"(includes file read and one-time CUDA/context setup)")

    if gain is not None or defects is not None:
        acc = CSBAccumulator(csb, backend=a.backend)
        img = acc.apply_corrections(img, gain, defects)
    img = bin_image(img, a.bin)

    if a.output_mrc:
        import mrcfile
        with mrcfile.new(a.output_mrc, overwrite=True) as m:
            m.set_data(img.astype(np.float32))
            if csb.ang_per_pix > 0:
                m.voxel_size = csb.ang_per_pix * a.bin
        print(f"  wrote {a.output_mrc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
