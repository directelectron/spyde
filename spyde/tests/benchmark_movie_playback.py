"""
benchmark_movie_playback.py
===========================
Phase-0 risk-reduction benchmark for the in-situ movie viewer (see
``docs`` / the movie-viewer plan). Measures **where the per-frame time actually
goes** when scrubbing a large in-situ movie, so we commit to the playback
rewrite (direct-read path bypassing dask, binary transport, GPU renderer)
KNOWING the bottleneck ordering rather than guessing.

The target dataset is a real Direct-Electron in-situ movie: a stack of large
2-D image frames over a time axis (nav-dim 1, 2-D signal), e.g.
``20251117_88074_run1_9104_movie.mrc`` = (3618, 4096, 4096) uint8 — a
3618-frame movie of 4k×4k images (16 MB/frame). This is the case the current
4D-STEM-oriented display path was NOT built for.

It times these per-frame stages for a scrub of N frames and prints a markdown
table + the ordering:

  * ``memmap``      np.memmap direct read of one frame ``mm[t]`` (+ force into RAM)
  * ``compute``     dask ``raw[t].compute()`` on the lazy signal (threaded scheduler)
  * ``getinds``     hyperspy CachedDaskArray get_index (the current live-display call),
                    threaded (no distributed cluster stood up here)
  * ``normalize``   _normalize_image → uint8 (the anyplotlib set_data step)
  * ``b64``         base64 encode of the uint8 frame (the transport payload today)
  * ``json``        json.dumps of the PLOTAPP line carrying that b64 (transport)

The A/B that confirms the dask suspicion is ``memmap`` vs ``compute`` vs
``getinds``. The transport blow-up is ``normalize`` → ``b64`` → ``json`` (and
the printed payload MB per frame).

Run (NOT under pytest):

    .venv/Scripts/python spyde/tests/benchmark_movie_playback.py
    .venv/Scripts/python spyde/tests/benchmark_movie_playback.py --frames 30
    .venv/Scripts/python spyde/tests/benchmark_movie_playback.py --path "C:\\...\\movie.mrc"
    .venv/Scripts/python spyde/tests/benchmark_movie_playback.py --synthetic 8192

``--synthetic K`` skips disk and benchmarks the transport/render stages on a
synthetic K×K frame (for the pure 8k×8k transport numbers on any machine).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import time

import numpy as np

# Candidate real in-situ movies on this dev box (first that exists wins).
_CANDIDATES = [
    r"C:\Users\CarterFrancis\Downloads\20251117_88074_run1_9104_movie.mrc",
    r"C:\Users\CarterFrancis\Downloads\20251117_88075_run3 some growth_1236_movie.mrc",
    r"C:\Users\CarterFrancis\Downloads\20241002_07954_movie.mrc",
]


def _default_path() -> "str | None":
    for p in _CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1e3:8.2f}"


def _time_stage(fn, indices, *, warmup=1, repeat=1):
    """Call ``fn(t)`` for each t in indices (after ``warmup`` untimed frames);
    return (mean_ms, min_ms, max_ms) over the timed frames."""
    idx = list(indices)
    for t in idx[:warmup]:
        try:
            fn(t)
        except Exception:
            pass
    samples = []
    for t in idx[warmup:]:
        best = None
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn(t)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        samples.append(best)
    if not samples:
        return (float("nan"),) * 3
    a = np.asarray(samples)
    return (float(a.mean()), float(a.min()), float(a.max()))


# --------------------------------------------------------------------------- #
# Transport stages (shared by the real and synthetic paths)
# --------------------------------------------------------------------------- #
def _normalize_image(frame: np.ndarray) -> np.ndarray:
    """Mirror anyplotlib._utils._normalize_image: scale to uint8 by frame min/max."""
    img = frame.astype(np.float64, copy=False)
    vmin = float(np.nanmin(img))
    vmax = float(np.nanmax(img))
    if vmax > vmin:
        buf = np.empty_like(img)
        np.subtract(img, vmin, out=buf)
        np.divide(buf, vmax - vmin, out=buf)
        np.multiply(buf, 255.0, out=buf)
        return buf.astype(np.uint8)
    return np.zeros(frame.shape, dtype=np.uint8)


def _bench_transport(get_frame, indices, results, *, warmup=1):
    """Time normalize → b64 → json on frames from ``get_frame(t)``."""
    def norm(t):
        _bench_transport._u8 = _normalize_image(get_frame(t))

    def b64(t):
        _bench_transport._u8 = _normalize_image(get_frame(t))
        _bench_transport._b = base64.b64encode(_bench_transport._u8.tobytes()).decode("ascii")

    def js(t):
        u8 = _normalize_image(get_frame(t))
        b = base64.b64encode(u8.tobytes()).decode("ascii")
        _bench_transport._j = json.dumps(
            {"type": "state_update", "key": "image_b64", "value": b,
             "image_width": int(u8.shape[1]), "image_height": int(u8.shape[0])}
        )

    results["normalize"] = _time_stage(norm, indices, warmup=warmup)
    results["b64"] = _time_stage(b64, indices, warmup=warmup)
    results["json"] = _time_stage(js, indices, warmup=warmup)
    # Payload size of the last json line (bytes).
    last = getattr(_bench_transport, "_j", "")
    results["_payload_mb"] = len(last.encode("utf-8")) / 1e6 if last else float("nan")


# --------------------------------------------------------------------------- #
# Real-file path
# --------------------------------------------------------------------------- #
def _read_mrc_header_offset(path: str) -> int:
    """MRC data starts after a 1024-byte header + NSYMBT extended header bytes
    (words 24 at byte offset 92). Return the byte offset of frame data."""
    with open(path, "rb") as fh:
        head = fh.read(1024)
    nsymbt = struct.unpack("<i", head[92:96])[0]
    return 1024 + int(nsymbt)


def _mrc_dtype_shape(path: str):
    """Parse (nx, ny, nz, dtype) from the MRC header (mode word at byte 12)."""
    with open(path, "rb") as fh:
        head = fh.read(1024)
    nx, ny, nz, mode = struct.unpack("<iiii", head[0:16])
    mode_map = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16, 12: np.float16,
                101: np.uint8}  # 101 = 4-bit packed handled elsewhere; treat 0/uint8
    # DE writes uint8 movies as mode 0 (signed int8) or 101; hyperspy resolves it.
    dt = mode_map.get(mode, np.int8)
    return nx, ny, nz, dt


def run_real(path: str, frames: int) -> None:
    import hyperspy.api as hs

    print(f"# Movie playback benchmark\n\nDataset: `{os.path.basename(path)}`")
    size_gb = os.path.getsize(path) / 1e9
    print(f"File size: {size_gb:.2f} GB\n")

    # Lazy signal (the app's load path). Used for compute() + get_index timing.
    s = hs.load(path, lazy=True)
    if isinstance(s, list):
        s = s[0]
    am = s.axes_manager
    nav_dim = am.navigation_dimension
    sig_shape = tuple(int(x) for x in am.signal_shape[::-1])  # (h, w)
    n_time = int(s.data.shape[0])
    dtype = s.data.dtype
    frame_bytes = int(np.prod(sig_shape)) * dtype.itemsize
    print(f"Shape: {s.data.shape} dtype={dtype} nav_dim={nav_dim} "
          f"signal={sig_shape} frame={frame_bytes/1e6:.1f} MB\n")
    print(f"Reader auto-chunks (nav axis): {s.data.chunks[0]}\n")

    if nav_dim != 1:
        print("> NOTE: nav_dim != 1 — this is not a plain image-stack movie; the "
              "leading-axis timings still illustrate the per-frame read cost.\n")

    n = min(frames, n_time)
    # Sample frames spread across the stack (cross chunk boundaries), plus warmup.
    indices = list(np.linspace(0, n_time - 1, n + 1).astype(int))

    results: dict = {}

    # --- memmap direct read (the proposed playback path) ---
    offset = _read_mrc_header_offset(path)
    mm = np.memmap(path, dtype=dtype, mode="r", offset=offset,
                   shape=(n_time,) + sig_shape)

    def memmap_read(t):
        # Force the page into RAM (a bare view is lazy) — mirrors what the
        # renderer needs: an actual contiguous frame.
        memmap_read._f = np.array(mm[int(t)])

    results["memmap"] = _time_stage(memmap_read, indices, warmup=1)

    # --- dask raw[t].compute() (threaded scheduler) ---
    raw = s.data

    def dask_compute(t):
        dask_compute._f = np.asarray(raw[int(t)].compute())

    results["compute"] = _time_stage(dask_compute, indices, warmup=1)

    # --- hyperspy _get_cache_dask_chunk (the EXACT current live-display call in
    # update_from_navigation_selection) — threaded (no distributed client pinned
    # here, so this is the synchronous compute branch). ---
    def getinds(t):
        # indices is (npoints, nav_ndim); one nav point for a crosshair.
        getinds._f = np.asarray(
            s._get_cache_dask_chunk(np.array([[int(t)]]), get_result=True)
        )
    try:
        results["getinds"] = _time_stage(getinds, indices, warmup=1)
    except Exception as e:
        print(f"> get_index timing failed: {type(e).__name__}: {e}\n")
        results["getinds"] = (float("nan"),) * 3

    # --- transport stages (feed from memmap, the fastest read) ---
    _bench_transport(lambda t: np.array(mm[int(t)]), indices, results, warmup=1)

    _print_table(results, frame_bytes)


# --------------------------------------------------------------------------- #
# Synthetic path (pure transport/render numbers on any machine)
# --------------------------------------------------------------------------- #
def run_synthetic(k: int, frames: int) -> None:
    print(f"# Movie playback benchmark (synthetic {k}×{k})\n")
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(k, k), dtype=np.uint8)
    frame_bytes = k * k  # uint8

    # Vary per frame so caches/b64 don't dedupe.
    def get_frame(t):
        f = base.copy()
        f[0, 0] = int(t) & 0xFF
        return f

    indices = list(range(frames + 1))
    results: dict = {"memmap": (float("nan"),) * 3, "compute": (float("nan"),) * 3}
    _bench_transport(get_frame, indices, results, warmup=1)
    _print_table(results, frame_bytes)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
_STAGE_ORDER = ["memmap", "compute", "getinds", "normalize", "b64", "json"]
_STAGE_DESC = {
    "memmap": "np.memmap mm[t] -> RAM (proposed playback read)",
    "compute": "dask raw[t].compute() (threaded)",
    "getinds": "hyperspy get_index (current live-display call, threaded)",
    "normalize": "-> uint8 (anyplotlib set_data)",
    "b64": "+ base64 encode (transport payload)",
    "json": "+ json.dumps PLOTAPP line (transport)",
}


def _print_table(results: dict, frame_bytes: int) -> None:
    print("| stage | mean ms | min ms | max ms | what |")
    print("|---|---:|---:|---:|---|")
    for k in _STAGE_ORDER:
        if k not in results:
            continue
        mean, lo, hi = results[k]
        print(f"| `{k}` | {_fmt_ms(mean/1)} | {_fmt_ms(lo)} | {_fmt_ms(hi)} | "
              f"{_STAGE_DESC[k]} |")
    print()
    pay = results.get("_payload_mb", float("nan"))
    print(f"Frame size in RAM: **{frame_bytes/1e6:.1f} MB**  ·  "
          f"transport payload (b64-in-JSON): **{pay:.1f} MB/frame**\n")

    # Ordering summary — the headline.
    timed = [(k, results[k][0]) for k in _STAGE_ORDER
             if k in results and not np.isnan(results[k][0])]
    timed.sort(key=lambda kv: kv[1])
    if timed:
        line = "  <  ".join(f"{k} ({v*1e3:.1f}ms)" for k, v in timed)
        print(f"**Per-frame ordering (fast->slow):** {line}\n")
    # The dask suspicion, quantified.
    if "memmap" in results and "compute" in results \
            and not np.isnan(results["memmap"][0]) and not np.isnan(results["compute"][0]):
        mm_ms = results["memmap"][0] * 1e3
        cp_ms = results["compute"][0] * 1e3
        if mm_ms > 0:
            print(f"**Dask overhead:** `compute()` is **{cp_ms/mm_ms:.1f}x** the raw "
                  f"memmap read ({cp_ms:.1f}ms vs {mm_ms:.1f}ms).\n")


def main() -> None:
    # Windows console defaults to cp1252; force UTF-8 so the markdown table
    # (and any stray unicode) prints without a UnicodeEncodeError.
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=None, help="a .mrc/.hspy in-situ movie")
    ap.add_argument("--frames", type=int, default=20, help="frames to time")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="skip disk; benchmark transport on a K×K synthetic frame")
    args = ap.parse_args()

    if args.synthetic:
        run_synthetic(args.synthetic, args.frames)
        return

    path = args.path or _default_path()
    if not path or not os.path.exists(path):
        print("No movie file found. Pass --path <file.mrc> or --synthetic 8192.")
        print("Tried:", *(f"\n  {p}" for p in _CANDIDATES))
        return
    run_real(path, args.frames)


if __name__ == "__main__":
    main()
