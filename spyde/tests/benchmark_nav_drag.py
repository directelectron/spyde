"""
benchmark_nav_drag.py
=====================
End-to-end benchmark for navigator-drag performance on a real 4D-STEM MRC file.

Simulates a user dragging the crosshair across the full navigation space and
measures the latency from "position requested" to "data ready to display",
distinguishing:

  * In-chunk hits   — position is inside the already-cached chunk (numpy lookup, <1 ms)
  * Out-of-chunk misses — position crosses a chunk boundary (disk read required)
  * Cancelled futures — stale surrounding-block fetches dropped on fast drag

This mirrors the exact code path that runs in production:
  selector.delayed_update_data()
    -> update_from_navigation_selection()
      -> signal._get_cache_dask_chunk(..., return_future=True)
      -> client.submit(write_shared_array, chunk_future, shm_name, priority=10)

Run:
    .venv/Scripts/python spyde/tests/benchmark_nav_drag.py
    .venv/Scripts/python spyde/tests/benchmark_nav_drag.py --path D:/data/file.mrc
    .venv/Scripts/python spyde/tests/benchmark_nav_drag.py --workers 4 --drag-speed fast
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import time
import statistics
import threading
from multiprocessing import shared_memory
from typing import NamedTuple

import numpy as np
import hyperspy.api as hs
from dask.distributed import Client, LocalCluster

from spyde.drawing.update_functions import write_shared_array, read_shared_array


# ---------------------------------------------------------------------------
# Tiny stub objects so we can call update_from_navigation_selection without
# a full Qt stack. The production function only calls:
#   child.plot_state.current_signal
#   child.shared_memory   (lazy property)
#   child._pending_shm_future
#   child.main_window.dask_manager.client
# ---------------------------------------------------------------------------

class _FakeDaskManager:
    def __init__(self, client):
        self.client = client


class _FakeMainWindow:
    def __init__(self, client):
        self.dask_manager = _FakeDaskManager(client)


class _FakePlotState:
    def __init__(self, signal):
        self.current_signal = signal


class _FakePlot:
    """Minimal stand-in for spyde.drawing.plots.plot.Plot."""

    def __init__(self, signal, client, shm_name: str):
        self.plot_state = _FakePlotState(signal)
        self.main_window = _FakeMainWindow(client)
        self._shm_name = shm_name
        self._shm_obj = None
        self._pending_shm_future = None

    @property
    def shared_memory(self):
        if self._shm_obj is None:
            sig_shape = self.plot_state.current_signal.axes_manager.signal_shape
            nbytes = int(np.prod(sig_shape)) * 4 + 256  # float32 + header
            try:
                self._shm_obj = shared_memory.SharedMemory(
                    name=self._shm_name, create=True, size=nbytes
                )
            except FileExistsError:
                self._shm_obj = shared_memory.SharedMemory(
                    name=self._shm_name, create=False
                )
        return self._shm_obj

    def cleanup(self):
        if self._shm_obj is not None:
            try:
                self._shm_obj.close()
                self._shm_obj.unlink()
            except Exception:
                pass
            self._shm_obj = None


class _FakeSelector:
    """Minimal selector stub — is_integrating=False means indices are averaged."""
    is_integrating = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class DragResult(NamedTuple):
    position: tuple          # (y, x) nav index
    chunk_hit: bool          # True = numpy cache hit, no disk I/O
    t_submit_ms: float       # time to call get_cache_dask_chunk + submit shm write
    t_ready_ms: float        # time from submit to future.done()
    t_total_ms: float        # t_submit + t_ready
    cancelled_surroundings: int  # surrounding futures cancelled before this request


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def benchmark_drag(
    path: str,
    n_workers: int = 3,
    threads_per_worker: int = 4,
    cache_padding: int = 1,
    drag_step: int = 1,         # nav pixels between drag positions
    drag_speed_ms: float = 16,  # ms between positions (16ms ≈ 60 Hz drag)
    max_positions: int = 500,   # cap total positions (None = full dataset)
):
    print(f"\n{'='*70}")
    print(f"Nav-drag benchmark")
    print(f"  File:        {path}")
    print(f"  Workers:     {n_workers} x {threads_per_worker} threads")
    print(f"  Cache pad:   {cache_padding}")
    print(f"  Drag step:   every {drag_step} nav pixel(s)")
    print(f"  Drag speed:  {drag_speed_ms:.0f} ms between positions")
    print(f"{'='*70}\n")

    # --- Load signal ---
    t0 = time.perf_counter()
    print("Loading signal (lazy)...", end=" ", flush=True)
    sig = hs.load(path, lazy=True)
    print(f"done in {(time.perf_counter()-t0)*1000:.0f} ms")
    print(f"  Signal shape: {sig.data.shape}  chunks: {sig.data.chunks}")
    nav_shape = sig.axes_manager.navigation_shape   # (ny, nx)
    sig_shape = sig.axes_manager.signal_shape       # (ky, kx)
    ny, nx = nav_shape
    print(f"  Nav: {nav_shape}  Signal: {sig_shape}")

    # --- Start cluster ---
    t0 = time.perf_counter()
    print(f"\nStarting LocalCluster (1 worker → scale to {n_workers})...", end=" ", flush=True)
    cluster = LocalCluster(n_workers=1, threads_per_worker=threads_per_worker)
    client = Client(cluster)
    t_client_ready = time.perf_counter() - t0
    print(f"client ready in {t_client_ready*1000:.0f} ms")
    if n_workers > 1:
        cluster.scale(n_workers)
        # Wait for workers to come up (measure separately)
        t0 = time.perf_counter()
        client.wait_for_workers(n_workers)
        t_scale = time.perf_counter() - t0
        print(f"  All {n_workers} workers ready in additional {t_scale*1000:.0f} ms")
    print(f"  Dashboard: {client.dashboard_link}")

    # --- Set global client so CachedDaskArray.client property finds it ---
    # (Production does this automatically via get_client())
    sig._lazy = True  # ensure lazy

    # Trigger CachedDaskArray creation with the global distributed client
    sig.cache_pad = cache_padding
    # Warm up: CachedDaskArray is created lazily on first call to _get_cache_dask_chunk

    # --- Shared memory ---
    shm_name = "spyde_drag_bench"
    plot = _FakePlot(sig, client, shm_name)
    selector = _FakeSelector()

    # --- Build drag path: row-major scan across nav space ---
    positions = []
    for y in range(0, ny, drag_step):
        row = list(range(0, nx, drag_step))
        if y // drag_step % 2 == 1:
            row = row[::-1]   # boustrophedon — alternating row direction like a real drag
        for x in row:
            positions.append((y, x))
            if max_positions and len(positions) >= max_positions:
                break
        if max_positions and len(positions) >= max_positions:
            break

    print(f"\nDrag path: {len(positions)} positions "
          f"({ny}x{nx} nav, step={drag_step}, boustrophedon)")

    # --- Simulate drag ---
    results: list[DragResult] = []
    prev_chunk_ind = None

    # Import here to avoid circular import at module level
    from spyde.drawing.update_functions import update_from_navigation_selection

    print("\nRunning drag simulation...\n")
    print(f"  {'pos':>12}  {'type':>8}  {'submit':>8}  {'ready':>8}  {'total':>8}  {'cancelled':>9}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}")

    for i, (y, x) in enumerate(positions):
        # Count cancelled surroundings before this request
        cached_arr = getattr(sig, "cached_dask_array", None)
        n_cancelled = 0
        if cached_arr is not None:
            try:
                from dask.distributed import Future as _DFuture
                n_cancelled = sum(
                    1 for f in cached_arr.surrounding_cached_blocks
                    if isinstance(f, _DFuture) and not f.done()
                )
            except Exception:
                pass

            # Determine current chunk index (to classify hit vs miss)
        nav_dim = len(sig.axes_manager.navigation_shape)
        if cached_arr is not None:
            try:
                indices_arr = np.array([[y, x]]) if nav_dim == 2 else np.array([[y]])
                from hyperspy.misc.array_tools import _get_navigation_dimension_chunk_slice
                core_ind, _, _ = _get_navigation_dimension_chunk_slice(
                    indices_arr, sig.data.chunks, cache_padding
                )
                chunk_ind = tuple(core_ind[0]) if core_ind else None
            except Exception:
                chunk_ind = None
        else:
            chunk_ind = None

        chunk_hit = (chunk_ind is not None and chunk_ind == prev_chunk_ind)

        # Time the submit
        t_submit_start = time.perf_counter()
        indices = np.array([[y, x]])
        current_img = update_from_navigation_selection(
            selector, plot, indices,
            get_result=False, cache_in_shared_memory=True,
        )
        t_submit_ms = (time.perf_counter() - t_submit_start) * 1000

        # Time until future is done
        t_wait_start = time.perf_counter()
        from dask.distributed import Future as _DFut
        if isinstance(current_img, _DFut):
            # Poll — mirrors what PlotUpdateWorker does every 2ms
            while not current_img.done():
                time.sleep(0.001)
            t_ready_ms = (time.perf_counter() - t_wait_start) * 1000
        else:
            # Already a numpy array (shouldn't happen in this path, but handle it)
            t_ready_ms = 0.0

        t_total_ms = t_submit_ms + t_ready_ms

        prev_chunk_ind = chunk_ind

        result = DragResult(
            position=(y, x),
            chunk_hit=chunk_hit,
            t_submit_ms=t_submit_ms,
            t_ready_ms=t_ready_ms,
            t_total_ms=t_total_ms,
            cancelled_surroundings=n_cancelled,
        )
        results.append(result)

        # Print every 10th position to show progress without flooding
        if i % 10 == 0 or not chunk_hit:
            kind = "HIT " if chunk_hit else "MISS"
            print(f"  ({y:4d},{x:4d})    {kind}   "
                  f"{t_submit_ms:6.1f}ms  {t_ready_ms:6.1f}ms  "
                  f"{t_total_ms:6.1f}ms  {n_cancelled:>9d}")

        # Simulate real drag timing (don't actually sleep the full interval —
        # we want to stress the system with back-to-back requests)
        if drag_speed_ms > 0 and i < len(positions) - 1:
            time.sleep(drag_speed_ms / 1000)

    # --- Summary ---
    hits   = [r for r in results if r.chunk_hit]
    misses = [r for r in results if not r.chunk_hit]

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Total positions:      {len(results)}")
    print(f"  In-chunk hits:        {len(hits)} ({100*len(hits)/len(results):.0f}%)")
    print(f"  Out-of-chunk misses:  {len(misses)} ({100*len(misses)/len(results):.0f}%)")

    if hits:
        hit_totals = [r.t_total_ms for r in hits]
        print(f"\n  In-chunk hit latency (submit + numpy lookup):")
        print(f"    median  {statistics.median(hit_totals):6.1f} ms")
        print(f"    p95     {sorted(hit_totals)[int(0.95*len(hit_totals))]:6.1f} ms")
        print(f"    max     {max(hit_totals):6.1f} ms")

    if misses:
        miss_totals   = [r.t_total_ms for r in misses]
        miss_submit   = [r.t_submit_ms for r in misses]
        miss_ready    = [r.t_ready_ms for r in misses]
        print(f"\n  Out-of-chunk miss latency (disk read + TCP transfer):")
        print(f"                    submit       ready      total")
        print(f"    median      {statistics.median(miss_submit):6.1f} ms   "
              f"{statistics.median(miss_ready):6.1f} ms   "
              f"{statistics.median(miss_totals):6.1f} ms")
        print(f"    p95         {sorted(miss_submit)[int(0.95*len(miss_submit))]:6.1f} ms   "
              f"{sorted(miss_ready)[int(0.95*len(miss_ready))]:6.1f} ms   "
              f"{sorted(miss_totals)[int(0.95*len(miss_totals))]:6.1f} ms")
        print(f"    max         {max(miss_submit):6.1f} ms   "
              f"{max(miss_ready):6.1f} ms   "
              f"{max(miss_totals):6.1f} ms")

    total_cancelled = sum(r.cancelled_surroundings for r in results)
    print(f"\n  Total surrounding futures cancelled: {total_cancelled}")
    print(f"  (these were stale prefetch tasks freed before each new request)")

    # Chunk boundary crossings
    chunk_crossings = len(misses)
    if chunk_crossings > 0:
        nav_dim = len(sig.axes_manager.navigation_shape)
        nav_chunk_y = sig.data.chunks[0][0]
        nav_chunk_x = sig.data.chunks[1][0] if nav_dim >= 2 else None
        chunk_str = f"({nav_chunk_y}, {nav_chunk_x})" if nav_chunk_x else f"({nav_chunk_y},)"
        print(f"\n  Nav chunk size: {chunk_str} — "
              f"miss every ~{nav_chunk_y // drag_step} drag steps")

    # Cleanup
    plot.cleanup()
    try:
        client.close()
        cluster.close()
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Navigator drag latency benchmark")
    parser.add_argument(
        "--path",
        default=r"D:/data/sample.mrc",
        help="Path to 4D-STEM MRC file (~60 GB)",
    )
    parser.add_argument("--workers", type=int, default=3,
                        help="Number of dask workers (default: 3)")
    parser.add_argument("--threads", type=int, default=4,
                        help="Threads per worker (default: 4)")
    parser.add_argument("--cache-padding", type=int, default=1,
                        help="CachedDaskArray cache_padding (default: 1)")
    parser.add_argument("--drag-step", type=int, default=1,
                        help="Nav pixels between drag positions (default: 1 = every pixel)")
    parser.add_argument("--drag-speed", type=float, default=16,
                        help="ms between drag positions (default: 16 ≈ 60 Hz). 0 = no sleep.")
    parser.add_argument("--max-positions", type=int, default=300,
                        help="Max nav positions to visit (default: 300, 0 = full dataset)")
    args = parser.parse_args()

    benchmark_drag(
        path=args.path,
        n_workers=args.workers,
        threads_per_worker=args.threads,
        cache_padding=args.cache_padding,
        drag_step=args.drag_step,
        drag_speed_ms=args.drag_speed,
        max_positions=args.max_positions or None,
    )


if __name__ == "__main__":
    main()
