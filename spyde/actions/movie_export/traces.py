"""
traces.py — 1-D trace capture + resample for the movie's trace inset.

A "trace" is a 1-D signal loaded in the session (dragged onto the wizard's trace
slot) plotted below/over the movie frames with a moving time cursor. We capture
its ``(x, y)`` NOW (at add time — a snapshot, like the report's figure cells) so
a later close of the source window can't break the export, then resample the
whole trace onto the movie's own time base at render time with ``np.interp``.

The v1 source is a 1-D plot window's current signal. A future source —
per-frame values baked into a movie's ``original_metadata`` (e.g. a temperature
column shipped by the DE-MRC reader) — is a documented seam: :func:`from_metadata`
below is intentionally NOT implemented, only sketched, so the trace inset can
grow that source without reshaping this module.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

# Default colour cycle for successive traces (matplotlib-ish, readable on white).
_TRACE_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd",
                 "#ff7f0e", "#17becf", "#8c564b", "#e377c2")

_id_counter = itertools.count(1)


@dataclass
class TraceSpec:
    """A captured 1-D trace to overlay on the movie's trace inset.

    ``x``/``y`` are the trace's own sample coordinates AT CAPTURE TIME (numpy
    arrays; a snapshot — the source window may close afterward). ``x`` is in the
    trace's own axis units; :func:`resample` maps ``y`` onto the movie's time
    base. ``label`` names it in the legend; ``color`` is the plot colour;
    ``units`` labels the trace's x-axis (informational)."""

    id: str
    label: str
    color: str
    units: str
    x: np.ndarray
    y: np.ndarray

    def state(self) -> dict:
        """The pixel-free descriptor for ``mvx_state`` (renderer contract)."""
        return {"id": self.id, "label": self.label, "color": self.color,
                "units": self.units}

    def resample(self, movie_times: np.ndarray) -> np.ndarray:
        """Interpolate ``y`` onto *movie_times* (the frame time base, seconds).

        ``np.interp`` clamps outside the captured range (holds the first/last
        value) — a trace shorter than the movie simply flat-lines past its end,
        never extrapolates wild values."""
        mt = np.asarray(movie_times, dtype=float)
        if self.x.size == 0 or self.y.size == 0:
            return np.zeros_like(mt)
        # np.interp needs increasing xp; sort defensively (a captured axis is
        # normally already monotonic).
        order = np.argsort(self.x)
        return np.interp(mt, np.asarray(self.x)[order], np.asarray(self.y)[order])


def new_trace_id() -> str:
    return f"tr{next(_id_counter)}"


def color_for_index(i: int) -> str:
    return _TRACE_COLORS[i % len(_TRACE_COLORS)]


def capture_from_plot(plot, *, color: str | None = None) -> "TraceSpec | None":
    """Capture a :class:`TraceSpec` from a live 1-D ``Plot``.

    ``y`` = the plot's displayed 1-D data (``current_data``); ``x`` = its signal
    axis coordinate array; ``label`` from the plot title / signal quantity;
    ``units`` from the signal axis. Returns None when the plot is not a paintable
    1-D signal (caller emits the error)."""
    data = getattr(plot, "current_data", None)
    if not isinstance(data, np.ndarray) or data.ndim != 1 or data.size == 0:
        return None
    y = np.asarray(data, dtype=float)

    # x-axis: the current signal's single signal axis coordinate array.
    x = np.arange(y.shape[0], dtype=float)
    units = ""
    label = ""
    try:
        sig = plot.plot_state.current_signal
        ax = sig.axes_manager.signal_axes[0]
        xa = np.asarray(ax.axis, dtype=float)
        if xa.shape[0] == y.shape[0]:
            x = xa
        u = str(getattr(ax, "units", "") or "")
        units = "" if u in ("<undefined>", "px", "") else u
    except Exception as e:
        log.debug("trace axis read failed: %s", e)

    # label: plot title, then the signal's quantity metadata, then a default.
    try:
        label = plot._plot_title() or ""
    except Exception:
        label = ""
    if not label:
        try:
            sig = plot.plot_state.current_signal
            q = sig.metadata.get_item("Signal.quantity", default="")
            label = str(q or "")
        except Exception:
            label = ""
    if not label:
        label = "trace"

    return TraceSpec(id=new_trace_id(), label=str(label),
                     color=str(color or _TRACE_COLORS[0]), units=units,
                     x=x, y=y)


def from_insitu_channel(tree, channel: str, *, color: str | None = None):
    """Capture a :class:`TraceSpec` from a movie's ATTACHED instrument channel.

    This is the second trace source (``capture_from_plot`` is the first): a
    per-frame column that :mod:`spyde.insitu` already aligned and resampled
    onto this movie's frame times, sitting on the tree as ``insitu_channels``.
    It needs no 1-D plot window to exist — an electrochemistry potential lives
    on the tree from the moment the movie opens, so it can be burnt into the
    frame directly.

    The x base is the movie's OWN time axis in seconds (index × scale), not the
    instrument clock: an overlay is resampled against ``movie_times`` at render
    time, and the instrument clock has a lag offset that would shift every
    value by that lag. The values are already per-frame, so this is an identity
    resample — but going through the same path keeps one code path for both
    sources.

    Frames outside the instrument record are NaN in ``insitu_channels`` and
    stay NaN here; :func:`~spyde.actions.movie_export.pipeline._draw_text_overlays`
    paints a dash for those rather than a made-up number.

    Returns None when the tree has no such channel.
    """
    channels = getattr(tree, "insitu_channels", None) or {}
    values = channels.get(channel)
    if values is None:
        log.debug("traces: no in-situ channel %r on this tree", channel)
        return None
    y = np.asarray(values, dtype=float)
    if y.size == 0:
        return None
    try:
        ax = tree.root.axes_manager.navigation_axes[0]
        scale, offset = float(ax.scale), float(ax.offset)
    except Exception:
        scale, offset = 1.0, 0.0
    x = np.arange(y.size, dtype=float) * scale + offset

    label, units = _split_channel_name(channel)
    return TraceSpec(id=new_trace_id(), label=label,
                     color=str(color or _TRACE_COLORS[0]), units=units, x=x, y=y)


def _split_channel_name(channel: str) -> tuple[str, str]:
    """``"Ewe/V"`` → ``("Ewe", "V")``; ``"<I>/mA"`` → ``("I", "mA")``.

    EC-Lab names a column ``quantity/unit`` and brackets an averaged one, which
    reads badly burnt into a frame ("<I>/mA = 0.0002"). Split it so the overlay
    can render "I = 0.0002 mA" like every other trace.
    """
    name, _, unit = str(channel).rpartition("/")
    if not name:
        name, unit = str(channel), ""
    return name.strip().strip("<>") or str(channel), unit.strip()


def insitu_channel_options(tree) -> list[dict]:
    """The instrument channels on *tree* worth offering as an overlay.

    Flags and bookkeeping columns (``mode``, ``error``, ``counter inc.``) are
    excluded — they are booleans that read as "0"/"1" burnt into a frame. The
    result is renderer-facing: ``[{"channel", "label", "units"}]``.
    """
    channels = getattr(tree, "insitu_channels", None) or {}
    out: list[dict] = []
    for name in channels:
        if name == "time/s" or name in _SKIP_CHANNELS:
            continue
        values = np.asarray(channels[name], dtype=float)
        if values.size == 0 or not np.isfinite(values).any():
            continue
        label, units = _split_channel_name(name)
        out.append({"channel": name, "label": label, "units": units})
    return out


# Per-frame columns that carry no meaning as a burnt-in number.
_SKIP_CHANNELS = frozenset({
    "mode", "error", "ox/red", "control changes", "Ns changes", "counter inc.",
})
