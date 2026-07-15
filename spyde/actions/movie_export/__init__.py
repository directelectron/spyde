"""
movie_export — MOVIE EXPORT for in-situ (time-series) signals.

An in-situ movie is a dataset with a 1-D time navigation axis and 2-D image
signal (see :mod:`spyde.signals.insitu`). This package renders that stack out to
a shareable video (H.264 mp4 — pastes straight into PowerPoint — or a small GIF)
with a colormapped LUT, a burnt-in timestamp + scale bar, time-gated
annotations, and optional 1-D "trace" insets (a signal loaded in the session,
plotted with a moving time cursor).

Layout (mirrors the Find-Vectors compute-package split — pure compute layers,
interactive wiring on top):

* :mod:`encoder`  — the writer seam (``open_writer(path, fps, size)`` →
  ``.append(rgb) / .close()``). The one place that knows about ffmpeg / GIF.
* :mod:`pipeline` — the memory-safe frame loop: per-frame lazy slice → LUT →
  PIL draw → trace inset → encoder. NEVER computes the full dataset.
* :mod:`traces`   — the 1-D trace capture + ``np.interp`` resample helpers.
* :mod:`handlers` — the staged ``mvx_*`` wizard handlers (registered in
  :data:`spyde.actions.registry.STAGED_HANDLERS`, wizard key ``mvx``).
"""
from __future__ import annotations

from spyde.actions.movie_export.handlers import PARAMETERS  # noqa: F401
