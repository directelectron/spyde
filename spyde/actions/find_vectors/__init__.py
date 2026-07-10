"""
find_vectors — Find Diffraction Vectors compute package for SpyDE.

The Qt-free compute core for Find Vectors on 4D/5D-STEM data: nav-space Gaussian
blur, disk-template NXCORR (or DoG band-pass), peak detection with subpixel
refinement, and a batch pipeline that adds a DiffractionVectors node to the
signal tree. The interactive UI (tuning carets, live preview overlay) lives in
``find_vectors_action`` / ``vector_overlay``; this package is pure compute.

Nav-space Gaussian blur uses two paths:
  - Live preview: NavBlurCache — async per-chunk blur that piggybacks on
    CachedDaskArray's already-resident chunk data (O(1) pattern access when warm,
    ~1ms single-frame fallback when cold).
  - Batch compute: dask.array.map_overlap with ghost zones (depth=ceil(3σ)) so
    chunk boundaries are handled correctly.

This module was split out of the former monolithic ``find_vectors.py`` into a
package; the submodules are:
  - ``gpu_runtime`` — numba-CUDA / CuPy runtime infra (pools, streams, latches)
  - ``kernels``     — numba @cuda.jit kernels + CuPy/cuFFT NXCORR
  - ``detectors``   — per-frame algorithm cores (NXCORR, DoG, beam-stop, …)
  - ``chunk``       — ghost-block chunk pipeline (GPU + CPU + DoG paths)
  - ``orchestrate`` — dask/distributed batch orchestration (_do_compute_vectors)

The names re-exported below form this package's public API — the surface that
external modules (vector_overlay, find_vectors_action, find_vectors_torch,
find_vectors_neural, benchmarks, tests) import via
``from spyde.actions.find_vectors import X``.
"""

from __future__ import annotations

# ── Public API re-exports (see module docstring) ──────────────────────────────
from spyde.actions.find_vectors.gpu_runtime import (  # noqa: E402,F401
    MAX_PEAKS,
    _cupy_available,
    _gpu_task_allowed,
    _reset_gpu_state,
)
from spyde.actions.find_vectors.detectors import (  # noqa: E402,F401
    DEFAULT_DOG_SIGMA1,
    DEFAULT_DOG_SIGMA2,
    DEFAULT_DOG_THRESHOLD,
    DEFAULT_NEURAL_THRESHOLD,
    METHOD_DOG,
    METHOD_NEURAL,
    METHOD_NXCORR,
    NavBlurCache,
    _auto_beamstop_from_signal,
    _auto_params,
    _dilate_mask,
    _disk_mean_intensity,
    _estimate_disk_radius,
    _find_peaks_single_frame,
    _find_vectors_single_frame,
    _find_vectors_single_frame_dog,
    _get_disk_fft,
    _make_disk,
    _sample_raw_bilinear,
    _subpixel_parabola,
    _with_raw_intensity,
    detect_beamstop,
)
from spyde.actions.find_vectors.chunk import (  # noqa: E402,F401
    _dog_block,
    _find_vectors_chunk,
    _find_vectors_chunk_dog,
    _find_vectors_chunk_gpu,
    _find_vectors_chunk_gpu_impl,
    _nav_blur_trim,
)
from spyde.actions.find_vectors.orchestrate import (  # noqa: E402,F401
    _balanced_nav_chunks,
    _compute_chunks_with_live_counts,
    _copy_nav_axes_to,
    _dispatch_chunks_gpu_aware,
    _do_compute_vectors,
    _nav_chunk_size,
)
