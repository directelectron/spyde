"""
movie_export — the MOVIE RENDER ENGINE for in-situ (time-series) signals.

An in-situ movie is a dataset with a 1-D time navigation axis and 2-D image
signal (see :mod:`spyde.signals.insitu`). This package renders that stack out to
a shareable video (H.264 mp4 — pastes straight into PowerPoint — or a small GIF)
with a colormapped LUT, a burnt-in timestamp + scale bar, time-gated
annotations, crop, freeze holds, 1-D-signal-as-text overlays, and optional 1-D
"trace" insets.

It is the shared render engine behind the MOVIE BLOCK
(:mod:`spyde.actions.report.movie`) — the editable, persistent movie cell in the
report/presentation document. (It formerly also backed a per-plot ``mvx_*`` caret
wizard, now removed; the movie block replaced it.)

Layout (mirrors the Find-Vectors compute-package split — pure compute layers,
interactive wiring on top):

* :mod:`encoder`  — the writer seam (``open_writer(path, fps, size)`` →
  ``.append(rgb) / .close()``). The one place that knows about ffmpeg / GIF.
* :mod:`pipeline` — the memory-safe frame loop: per-frame lazy slice → crop →
  LUT → PIL draw → overlays → encoder. NEVER computes the full dataset. Also
  exposes ``render_single_frame`` for a poster bake.
* :mod:`traces`   — the 1-D trace / text-overlay capture + ``np.interp`` resample.
"""
from __future__ import annotations
