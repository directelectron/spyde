"""
Benchmark: what a navigator scrub pays to cross a zarr chunk boundary.

Run it directly against a real ``.zspy``; it is slow and it needs a file, so it
is not a test::

    SPYDE_ZSPY_BENCH=D:\\path\\to\\movie.zspy SPYDE_NO_DASK=1 \\
        uv run --no-sync python -m spyde.tests.benchmark_zspy_scrub

Without ``SPYDE_ZSPY_BENCH`` it prints how to run it and exits.

What it measures
----------------
A blosc-compressed chunk is atomic: reading one frame out of it decodes the
whole chunk. Dwelling inside a decoded chunk is a numpy slice, so the only
costly reads a drag makes are the ones that enter a chunk nobody has decoded
yet. Those reads are the subject.

The file is opened through a real ``Session`` (no dask cluster, the way the
tests build one) and the drags run on a worker thread, because that is where
the navigator reads: ``numcodecs`` decides whether to use blosc's thread pool
from the identity of the calling thread, so a decode on the main thread and the
same decode on any other thread are different measurements.

Two drags:

* ``x-stride-7`` steps seven positions along x per read. With eight-wide
  navigation chunks nearly every read lands in a new chunk, so it is the case
  where a read-ahead has almost no time to help.
* ``diagonal`` steps one position along both axes. It crosses a boundary every
  eighth read and leaves the read-ahead the seven reads in between, so it is
  the case where a read-ahead either lands or does not.

Each drag runs twice. The first pass leaves the file's compressed bytes in the
operating system's page cache; the decoded-block and frame caches are cleared
between the passes, so the second pass measures decode and not disk. Both are
reported: ``cold`` includes the disk read, ``warm`` is decode alone.

Reads go through ``get_local_frame``, and each read primes the block prefetcher
exactly as ``_direct_read_frame`` does, so the read-ahead under test is the one
the app runs.

It also times the two batch computations that share the decoder with the
navigator: the navigator sum over a square of chunks, and a Find Vectors DoG
batch over the same square, both on dask's threaded scheduler with eight
threads. Serialising blosc decode makes those the side that can regress. Each
batch runs twice and the second run is reported, so neither the page cache nor
the first CUDA context is in the number.
"""
from __future__ import annotations

import os
import statistics
import threading
import time

import numpy as np

PATH_VARIABLE = "SPYDE_ZSPY_BENCH"

# Milliseconds of pause between reads, standing in for the time the painter and
# the display take in a real drag. It is the whole budget a read-ahead has, so
# a benchmark that reads back to back would measure a read-ahead that can never
# win. Not counted in any reported read time.
THINK_MILLISECONDS = float(os.environ.get("SPYDE_ZSPY_BENCH_THINK_MS", "16"))

READS_PER_DRAG = 60
BATCH_CHUNKS_PER_AXIS = 4
BATCH_THREADS = 8


def _percentile(samples, fraction):
    ordered = sorted(samples)
    if not ordered:
        return float("nan")
    position = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[position]


def _navigation_chunk_shape(signal):
    """Per-navigation-axis storage chunk lengths of the signal's zarr array."""
    from spyde.array_cache.readers.source_array import find_source_array

    source = find_source_array(signal.data)
    if source is None:
        raise SystemExit("the signal is not backed directly by a zarr array")
    navigation_dimension = int(signal.axes_manager.navigation_dimension)
    return tuple(int(c) for c in source.chunks[:navigation_dimension])


def _x_stride_points(navigation_shape, stride=7, row=128):
    width = navigation_shape[1]
    return [(row, (4 + stride * step) % width) for step in range(READS_PER_DRAG)]


def _diagonal_points(navigation_shape, start=10):
    return [(min(start + step, navigation_shape[0] - 1),
             min(start + step, navigation_shape[1] - 1))
            for step in range(READS_PER_DRAG)]


class DragResult:
    def __init__(self, name, pass_name):
        self.name = name
        self.pass_name = pass_name
        self.milliseconds: list[float] = []
        self.crossings = 0
        self.crossing_hits = 0
        self.decode_milliseconds: list[float] = []

    @property
    def crossing_decodes(self):
        return len(self.decode_milliseconds)

    def line(self):
        # The per-read median sits wherever the decode/hit split falls, so the
        # median of the reads that DID decode is reported next to it: that is
        # the cost of one crossing, and it does not move with the split.
        decode_median = (statistics.median(self.decode_milliseconds)
                         if self.decode_milliseconds else float("nan"))
        return (f"{self.name:<12} {self.pass_name:<5} "
                f"median {statistics.median(self.milliseconds):7.1f} ms  "
                f"p95 {_percentile(self.milliseconds, 0.95):7.1f} ms  "
                f"crossings {self.crossings:3d}  "
                f"decodes {self.crossing_decodes:3d} "
                f"(median {decode_median:6.1f} ms)  "
                f"hits {self.crossing_hits:3d}")


def _chunk_resident(plot, signal, point):
    """Would this read be a numpy slice out of an already decoded block?

    Asks the reader serving the signal, which answers at chunk granularity and
    without touching the cache's ordering. False before any reader exists.
    """
    reader = plot._local_transform_readers.get(id(signal))
    probe = getattr(reader, "is_chunk_resident", None) if reader is not None else None
    return bool(probe(point)) if probe is not None else False


def _run_drag(plot, signal, points, chunk_shape, name, pass_name):
    from spyde.array_cache import get_local_frame
    from spyde.drawing.update_functions import _block_prefetcher

    data = signal.data
    result = DragResult(name, pass_name)
    previous_point = None
    previous_chunk = None
    for point in points:
        chunk = tuple(coordinate // length
                      for coordinate, length in zip(point, chunk_shape))
        crossed = previous_chunk is not None and chunk != previous_chunk
        resident = _chunk_resident(plot, signal, point) if crossed else False

        started = time.perf_counter()
        frame = get_local_frame(plot, signal, data, np.asarray(point))
        elapsed = (time.perf_counter() - started) * 1e3
        if frame is None:
            raise SystemExit(f"the array cache declined the read at {point}")

        result.milliseconds.append(elapsed)
        if crossed:
            result.crossings += 1
            if resident:
                result.crossing_hits += 1
            else:
                result.decode_milliseconds.append(elapsed)

        _block_prefetcher.prime(plot, signal, data, previous_point, point, 2)
        previous_point, previous_chunk = point, chunk
        if THINK_MILLISECONDS > 0:
            time.sleep(THINK_MILLISECONDS / 1e3)
    return result


def _run_drags_on_worker(plot, signal, chunk_shape):
    """Run both drags twice on one worker thread and return the four results.

    One thread for all four passes: the app reads on a single dispatcher
    thread, and blosc's behaviour depends on the thread identity.
    """
    navigation_shape = tuple(int(v) for v in signal.data.shape[:2])
    drags = (("x-stride-7", _x_stride_points(navigation_shape)),
             ("diagonal", _diagonal_points(navigation_shape)))
    results: list[DragResult] = []
    failure: list[BaseException] = []

    def read_all():
        try:
            for name, points in drags:
                results.append(_run_drag(plot, signal, points, chunk_shape,
                                         name, "cold"))
                plot._block_cache.clear()
                plot._array_cache.clear()
                results.append(_run_drag(plot, signal, points, chunk_shape,
                                         name, "warm"))
                plot._block_cache.clear()
                plot._array_cache.clear()
        except BaseException as error:       # reported, not swallowed
            failure.append(error)

    worker = threading.Thread(target=read_all, name="bench-dispatch")
    worker.start()
    worker.join()
    if failure:
        raise failure[0]
    return results


def _batch_region(signal, chunk_shape):
    """A square of ``BATCH_CHUNKS_PER_AXIS`` chunks per navigation axis, taken
    from the far corner so the drags have not warmed it."""
    navigation_shape = tuple(int(v) for v in signal.data.shape[:2])
    extents = []
    for axis in range(2):
        length = chunk_shape[axis] * BATCH_CHUNKS_PER_AXIS
        start = max(0, navigation_shape[axis] - length)
        extents.append(slice(start, start + length))
    return signal.data[tuple(extents)]


def _time_twice(compute):
    """Run ``compute`` twice and return the second wall time in milliseconds.

    The first run pays the disk read and, for the detector, the first CUDA
    context. Neither belongs in a decoder comparison.
    """
    for _ in range(2):
        started = time.perf_counter()
        compute()
        elapsed = (time.perf_counter() - started) * 1e3
    return elapsed


def _time_navigator_sum(region):
    total = region.sum(axis=(-2, -1))
    return _time_twice(lambda: total.compute(scheduler="threads",
                                             num_workers=BATCH_THREADS))


def _time_find_vectors_dog(region):
    from spyde.actions.find_vectors.chunk import _find_vectors_chunk_dog
    from spyde.actions.find_vectors.detectors import (
        DEFAULT_DOG_SIGMA1, DEFAULT_DOG_SIGMA2, DEFAULT_DOG_THRESHOLD,
    )
    from spyde.actions.find_vectors.gpu_runtime import MAX_PEAKS

    peaks = region.map_blocks(
        _find_vectors_chunk_dog, 0, 2, 0.0,
        DEFAULT_DOG_SIGMA1, DEFAULT_DOG_SIGMA2, DEFAULT_DOG_THRESHOLD, 3,
        False, None,
        dtype=np.float32,
        chunks=(region.chunks[0], region.chunks[1], (MAX_PEAKS,), (3,)))
    return _time_twice(lambda: peaks.compute(scheduler="threads",
                                             num_workers=BATCH_THREADS))


def _silence_messages():
    """Drop the session's outgoing PLOTAPP messages.

    They go to stdout, which is where this benchmark's numbers go. The same
    three bindings the test fixture replaces, because ``session`` imports
    ``emit`` by name.
    """
    import anyplotlib._electron
    import de_shell.ipc
    import spyde.backend.session

    for module in (de_shell.ipc, anyplotlib._electron, spyde.backend.session):
        if hasattr(module, "emit"):
            module.emit = lambda message: None


def run(path):
    from spyde.tests.migrated.conftest import close_session, make_session

    _silence_messages()
    session = make_session()
    try:
        session._load_file_thread(path)
        if not session.signal_trees:
            raise SystemExit(f"nothing opened from {path}")
        signal = session.signal_trees[0].root
        plot = next((p for p in session._plots
                     if not p.is_navigator and p.plot_state is not None), None)
        if plot is None:
            raise SystemExit("the session opened no signal plot")

        chunk_shape = _navigation_chunk_shape(signal)
        print(f"file           {path}")
        print(f"shape          {tuple(int(v) for v in signal.data.shape)} "
              f"{signal.data.dtype}")
        print(f"nav chunks     {chunk_shape}")
        print(f"blosc threads  {os.environ.get('SPYDE_BLOSC_THREADS', 'unset')}")
        print(f"think time     {THINK_MILLISECONDS:.0f} ms between reads")
        print()

        for result in _run_drags_on_worker(plot, signal, chunk_shape):
            print(result.line())
        print()

        region = _batch_region(signal, chunk_shape)
        print(f"batch region   {tuple(int(v) for v in region.shape)} "
              f"on {BATCH_THREADS} threads")
        print(f"navigator sum  {_time_navigator_sum(region):8.0f} ms")
        print(f"find vectors   {_time_find_vectors_dog(region):8.0f} ms  (DoG)")
    finally:
        close_session(session)


def main():
    path = os.environ.get(PATH_VARIABLE)
    if not path:
        print(f"set {PATH_VARIABLE} to a .zspy dataset to run this benchmark, "
              f"for example:\n"
              f"    {PATH_VARIABLE}=D:\\data\\movie.zspy SPYDE_NO_DASK=1 "
              f"uv run --no-sync python -m spyde.tests.benchmark_zspy_scrub")
        return
    if not os.path.exists(path):
        raise SystemExit(f"{PATH_VARIABLE} points at {path}, which does not exist")
    run(path)


if __name__ == "__main__":
    main()
