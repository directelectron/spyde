"""
pipeline.py — the memory-safe movie-export frame loop.

MEMORY-SAFETY CONTRACT (CLAUDE.md, same rule as find_vectors):
    NEVER compute the full dataset. For a lazy dask movie we slice ONE frame at
    a time — ``raw[t, ...].compute()`` — and materialise only that small 2-D
    frame. ``raw`` itself is never ``.compute()``-d. For a numpy movie the array
    is already in RAM, so a plain slice suffices.

Per frame ``t`` in the selected range (with temporal ``stride``):

    frame = np.asarray(raw[t, ...].compute())      # lazy → one-frame slice ONLY
          → strided box-mean spatial downsample (factor k)
          → 256-entry LUT built ONCE from (cmap, clim): lut[clip((f-lo)/(hi-lo)*255)]
          → PIL RGB image
          → time-gated annotations (kind dispatch; skip when t_sec ∉ time_range)
          → timestamp text  ("t = 12.34 s", top-left)
          → scale bar        (bottom-right, physical length; skip if uncalibrated)
          → paste trace inset (matplotlib Agg rendered ONCE; per-frame cursor + dots)
          → writer.append(rgb)

Even dimensions: H.264 needs even W/H — we crop 1px off an odd edge after
downsample and pass ``macro_block_size=1`` to the writer (see encoder.py).

The loop is driven by :func:`export_movie` on ``lifecycle.run_on_worker`` with a
generation guard, a per-frame cancel check, and partial-file cleanup on
cancel/failure (all wired in ``handlers.mvx_run``).
"""
from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger(__name__)

# Timestamp / scalebar / annotation styling.
_TS_COLOR = (255, 255, 255)
_TS_SHADOW = (0, 0, 0)
_SB_COLOR = (255, 255, 255)
_SB_SHADOW = (0, 0, 0)
# Trace inset width as a fraction of the frame width.
_INSET_W_FRAC = 0.28


# ── LUT ────────────────────────────────────────────────────────────────────────

def build_lut(cmap: str) -> np.ndarray:
    """A ``(256, 3)`` uint8 RGB lookup table for *cmap*, built ONCE per export
    (never per frame). Uses matplotlib's colormap table; falls back to grayscale
    for an unknown name."""
    try:
        import matplotlib
        cm = matplotlib.colormaps.get_cmap(cmap)
        ramp = np.linspace(0.0, 1.0, 256)
        rgba = cm(ramp)                      # (256, 4) float in 0..1
        return (rgba[:, :3] * 255.0 + 0.5).astype(np.uint8)
    except Exception as e:
        log.debug("LUT build for %r failed (%s) → grayscale", cmap, e)
        g = np.arange(256, dtype=np.uint8)
        return np.stack([g, g, g], axis=1)


def apply_lut(frame: np.ndarray, lut: np.ndarray,
              lo: float, hi: float) -> np.ndarray:
    """Map a 2-D *frame* to an ``(H, W, 3)`` uint8 RGB image via *lut* and the
    ``[lo, hi]`` contrast window. No matplotlib here — pure numpy indexing."""
    f = np.asarray(frame, dtype=np.float32)
    span = float(hi) - float(lo)
    if span <= 0:
        span = 1.0
    idx = np.clip((f - float(lo)) * (255.0 / span), 0, 255).astype(np.uint8)
    return lut[idx]


# ── spatial downsample ──────────────────────────────────────────────────────────

def downsample(frame: np.ndarray, k: int) -> np.ndarray:
    """Strided box-mean spatial downsample by integer factor *k* (k<=1 → no-op).

    Crops to a multiple of k, reshapes into k×k blocks and means them — a cheap,
    numpy-only anti-aliased shrink (no scipy/PIL resample needed)."""
    k = int(k)
    if k <= 1:
        return frame
    h, w = frame.shape[:2]
    hh, ww = (h // k) * k, (w // k) * k
    if hh == 0 or ww == 0:
        return frame
    f = np.asarray(frame, dtype=np.float32)[:hh, :ww]
    f = f.reshape(hh // k, k, ww // k, k).mean(axis=(1, 3))
    return f


def even_crop(rgb: np.ndarray) -> np.ndarray:
    """Crop 1px off an odd height/width so H.264 gets even dimensions."""
    h, w = rgb.shape[:2]
    return rgb[: h - (h % 2), : w - (w % 2)]


def _fit_frame(rgb: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Force an ``(H, W, 3)`` RGB frame to exactly ``(out_h, out_w, 3)`` for the
    fixed-size writer: crop any dimension that's too big, then zero-pad (letterbox,
    top-left origin) any that's too small. Keeps the writer's frame shape uniform
    even when a ragged source frame shrinks below the first frame's size."""
    rgb = rgb[:out_h, :out_w]
    h, w = rgb.shape[:2]
    if h == out_h and w == out_w:
        return rgb
    out = np.zeros((out_h, out_w, rgb.shape[2]), dtype=rgb.dtype)
    out[:h, :w] = rgb
    return out


# ── frame reading (MEMORY-SAFE) ──────────────────────────────────────────────────

def read_frame(raw, t: int) -> np.ndarray:
    """Read a SINGLE 2-D frame ``raw[t]`` as numpy.

    Lazy dask → ``raw[t, ...].compute()`` materialises ONLY that frame (never the
    whole array — the memory-safety contract). Numpy → a plain slice."""
    sl = raw[t, ...]
    comp = getattr(sl, "compute", None)
    if comp is not None:
        sl = comp()
    return np.asarray(sl)


# ── font / drawing helpers ───────────────────────────────────────────────────────

def _load_font(px: int):
    """A TrueType font at *px* (matplotlib's bundled DejaVuSans), PIL default as
    last resort."""
    from PIL import ImageFont
    try:
        import matplotlib.font_manager as fm
        path = fm.findfont("DejaVu Sans", fallback_to_default=True)
        return ImageFont.truetype(path, size=max(8, int(px)))
    except Exception as e:
        log.debug("truetype font load failed (%s) → PIL default", e)
        return ImageFont.load_default()


def _text_with_shadow(draw, xy, text, font, fill, shadow):
    x, y = xy
    draw.text((x + 1, y + 1), text, font=font, fill=shadow)
    draw.text((x, y), text, font=font, fill=fill)


def _draw_timestamp(img, t_sec: float, font):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    _text_with_shadow(draw, (6, 4), f"t = {t_sec:.2f} s", font,
                      _TS_COLOR, _TS_SHADOW)


def _draw_scalebar(img, scale_x: float, units: str, font):
    """Bottom-right physical scale bar. *scale_x* is the signal-x axis step
    (physical units per displayed pixel). A 'nice' bar length ~1/5 of the width
    is chosen from a 1/2/5 sequence."""
    from PIL import ImageDraw
    w, h = img.size
    if scale_x <= 0:
        return
    target_px = w * 0.2
    target_units = target_px * scale_x
    # snap to a 1/2/5 × 10^n length
    import math
    exp = math.floor(math.log10(target_units)) if target_units > 0 else 0
    base = target_units / (10 ** exp)
    nice = 1 if base < 1.5 else (2 if base < 3.5 else (5 if base < 7.5 else 10))
    bar_units = nice * (10 ** exp)
    bar_px = int(round(bar_units / scale_x))
    if bar_px <= 0 or bar_px > w:
        return
    draw = ImageDraw.Draw(img)
    margin = max(6, w // 60)
    x1 = w - margin
    x0 = x1 - bar_px
    yb = h - margin - max(3, h // 200)
    thick = max(2, h // 150)
    # shadow then bar
    draw.rectangle([x0 + 1, yb + 1, x1 + 1, yb + thick + 1], fill=_SB_SHADOW)
    draw.rectangle([x0, yb, x1, yb + thick], fill=_SB_COLOR)
    label = _format_length(bar_units, units)
    tw = draw.textlength(label, font=font)
    _text_with_shadow(draw, (x1 - tw, yb - thick - _font_height(font) - 2),
                      label, font, _SB_COLOR, _SB_SHADOW)


def _font_height(font) -> int:
    try:
        bbox = font.getbbox("Ag")
        return int(bbox[3] - bbox[1])
    except Exception:
        return 12


def _format_length(v: float, units: str) -> str:
    s = f"{v:g}"
    return f"{s} {units}" if units else s


# ── annotations (kind dispatch, time-gated) ──────────────────────────────────────

def _in_time_range(t_sec: float, ann: dict) -> bool:
    tr = ann.get("time_range")
    if not tr:
        return True
    try:
        t0, t1 = float(tr[0]), float(tr[1])
    except Exception:
        return True
    return t0 <= t_sec <= t1


def _draw_annotations(img, anns, t_sec: float, k: int):
    """Draw each time-gated annotation. Coordinates are in the ORIGINAL image's
    pixel space; we divide by the downsample factor *k* so they land correctly on
    the shrunk frame."""
    from PIL import ImageDraw
    if not anns:
        return
    draw = ImageDraw.Draw(img, "RGBA")

    def sc(v):
        return float(v) / max(1, k)

    for ann in anns:
        if not isinstance(ann, dict) or not _in_time_range(t_sec, ann):
            continue
        kind = str(ann.get("kind", ""))
        color = _color(ann.get("color", "#ffcc00"))
        try:
            if kind == "text":
                xy = ann.get("xy", [0, 0])
                font = _load_font(int(ann.get("size", 16)))
                _text_with_shadow(draw, (sc(xy[0]), sc(xy[1])),
                                  str(ann.get("text", "")), font,
                                  color, (0, 0, 0, 220))
            elif kind == "circle":
                xy = ann.get("xy", [0, 0])
                r = sc(ann.get("radius", 10))
                cx, cy = sc(xy[0]), sc(xy[1])
                draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                             outline=color, width=max(1, int(sc(ann.get("width", 3)))))
            elif kind == "rect":
                xy = ann.get("xy", [0, 0])
                wh = ann.get("wh", [10, 10])
                x0, y0 = sc(xy[0]), sc(xy[1])
                draw.rectangle([x0, y0, x0 + sc(wh[0]), y0 + sc(wh[1])],
                               outline=color, width=max(1, int(sc(ann.get("width", 3)))))
            elif kind == "arrow":
                xy = ann.get("xy", [0, 0])         # tail
                xy2 = ann.get("xy2", [10, 10])     # head
                _draw_arrow(draw, (sc(xy[0]), sc(xy[1])),
                            (sc(xy2[0]), sc(xy2[1])), color,
                            max(1, int(sc(ann.get("width", 3)))))
        except Exception as e:
            log.debug("annotation %r draw failed: %s", kind, e)


def _draw_arrow(draw, tail, head, color, width):
    draw.line([tail, head], fill=color, width=width)
    # arrowhead
    dx, dy = head[0] - tail[0], head[1] - tail[1]
    import math
    ang = math.atan2(dy, dx)
    hl = 8 + width * 2
    for da in (math.radians(150), math.radians(-150)):
        hx = head[0] + hl * math.cos(ang + da)
        hy = head[1] + hl * math.sin(ang + da)
        draw.line([head, (hx, hy)], fill=color, width=width)


def _color(c):
    """Normalise a color spec (hex string or (r,g,b[,a])) to an RGBA tuple."""
    if isinstance(c, (list, tuple)):
        vals = [int(v) for v in c]
        return tuple(vals + [255])[:4] if len(vals) < 4 else tuple(vals[:4])
    try:
        from PIL import ImageColor
        rgb = ImageColor.getrgb(str(c))
        return (rgb[0], rgb[1], rgb[2], 255)
    except Exception:
        return (255, 204, 0, 255)


# ── trace inset (matplotlib Agg once + per-frame cursor) ─────────────────────────

class _TraceInset:
    """The trace plot rendered ONCE to an RGBA ndarray, plus the data→pixel maps
    captured from the Agg axes so each frame can PIL-draw a moving cursor + dots
    without re-rendering matplotlib."""

    def __init__(self, rgba, x_to_px, y_to_px, resampled, times, colors):
        self.rgba = rgba                 # (h, w, 4) uint8 base plot
        self._x_to_px = x_to_px          # data-x (s) → pixel-x within the inset
        self._y_to_px = y_to_px          # data-y → pixel-y within the inset
        self._resampled = resampled      # list of y-on-movie-time arrays
        self._times = times              # movie time base (s)
        self._colors = colors

    @property
    def size(self):
        return self.rgba.shape[1], self.rgba.shape[0]   # (w, h)

    def frame_image(self, frame_i: int):
        """A PIL RGBA image of the inset for movie-frame index *frame_i*: the base
        plot + a vertical time cursor + a dot at each trace's current value."""
        from PIL import Image, ImageDraw
        img = Image.fromarray(self.rgba, mode="RGBA").copy()
        draw = ImageDraw.Draw(img)
        t = float(self._times[frame_i])
        cx = self._x_to_px(t)
        h = img.size[1]
        draw.line([(cx, 0), (cx, h)], fill=(60, 60, 60, 200), width=1)
        for yr, col in zip(self._resampled, self._colors):
            cy = self._y_to_px(float(yr[frame_i]))
            rgb = _color(col)
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=rgb)
        return img


def render_trace_inset(traces, movie_times, width_px, x_units: str):
    """Render the trace inset base ONCE with matplotlib Agg (dark-on-white,
    labelled). Returns a :class:`_TraceInset` or None when there are no traces.

    All traces are resampled onto *movie_times* (the frame time base). The Agg
    data→display transform is captured so per-frame cursor/dot placement is a
    cheap PIL draw, not a matplotlib re-render."""
    if not traces:
        return None
    times = np.asarray(movie_times, dtype=float)
    resampled = [tr.resample(times) for tr in traces]
    colors = [tr.color for tr in traces]

    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    w_in = max(1.6, width_px / 100.0)
    h_in = w_in * 0.6
    fig = Figure(figsize=(w_in, h_in), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.18, 0.22, 0.78, 0.72])
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    for tr, yr, col in zip(traces, resampled, colors):
        ax.plot(times, yr, color=col, lw=1.4, label=tr.label)
    ax.set_xlabel(f"time ({x_units})" if x_units else "time", fontsize=8, color="#222")
    ax.tick_params(labelsize=7, colors="#222")
    for spine in ax.spines.values():
        spine.set_color("#888")
    if any(tr.label for tr in traces):
        ax.legend(fontsize=7, loc="best", framealpha=0.7)
    ax.set_xlim(float(times.min()), float(times.max()))

    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())         # (h, w, 4) uint8
    rgba = np.array(buf, copy=True)

    # Capture the data→pixel transform (Agg display coords have origin at
    # BOTTOM-LEFT; PIL/image rows are top-down → flip y).
    h_px = rgba.shape[0]
    trans = ax.transData

    def x_to_px(xdata):
        return float(trans.transform((xdata, 0.0))[0])

    def y_to_px(ydata):
        return float(h_px - trans.transform((0.0, ydata))[1])

    return _TraceInset(rgba, x_to_px, y_to_px, resampled, times, colors)


# ── the export driver ────────────────────────────────────────────────────────────

def frame_indices(n_frames: int, t_start: int, t_end: int, stride: int):
    """The list of source frame indices to render (inclusive t_start..t_end with
    stride), clamped to the dataset."""
    t0 = max(0, int(t_start))
    t1 = min(int(n_frames) - 1, int(t_end))
    step = max(1, int(stride))
    return list(range(t0, t1 + 1, step))


def export_movie(raw, *, path: str, params: dict, n_frames: int,
                 scale_s: float, sig_scale_x: float, sig_units: str,
                 traces=None, should_cancel=None, progress=None) -> int:
    """Render the movie to *path*. Returns the number of frames written.

    MEMORY-SAFE: reads one frame per iteration via :func:`read_frame` (a lazy
    slice ``raw[t].compute()`` / numpy slice) — the full dataset is NEVER
    materialised.

    *should_cancel* is a 0-arg predicate polled per frame (returns True → stop and
    raise :class:`_Cancelled`). *progress(done, total)* narrates. The caller owns
    partial-file cleanup; on cancel we raise so the handler removes the file.
    """
    from spyde.actions.movie_export.encoder import open_writer

    p = params
    k = int(p.get("downsample", 1) or 1)
    fps = float(p.get("fps", 10.0) or 10.0)
    cmap = str(p.get("cmap", "gray") or "gray")
    clim = p.get("clim")
    timestamp = bool(p.get("timestamp", True))
    scalebar = bool(p.get("scalebar", True)) and sig_scale_x > 0
    anns = p.get("annotations") or []

    idxs = frame_indices(n_frames, p.get("t_start", 0),
                         p.get("t_end", n_frames - 1), p.get("stride", 1))
    total = len(idxs)
    if total == 0:
        raise ValueError("Movie export: empty frame range.")

    lut = build_lut(cmap)

    # Cancel BEFORE the first-frame (contrast probe) read so a cancel requested
    # during setup takes effect at the earliest possible moment. The in-flight dask
    # read itself cannot be interrupted mid-compute (a single-frame slice is small,
    # so this is acceptable) — but polling on each side keeps the window between a
    # cancel request and stopping as tight as possible (finding: uncancellable probe).
    if should_cancel is not None and should_cancel():
        raise _Cancelled()

    # Auto contrast from the FIRST rendered frame when clim is unset (robust 2-98%).
    first = downsample(read_frame(raw, idxs[0]), k)

    # Cancel immediately AFTER the probe read too (a cancel that arrived while the
    # read was in flight stops us before we open the writer / encode any frame).
    if should_cancel is not None and should_cancel():
        raise _Cancelled()

    if clim and len(clim) == 2 and clim[0] is not None and clim[1] is not None:
        lo, hi = float(clim[0]), float(clim[1])
    else:
        finite = first[np.isfinite(first)]
        if finite.size:
            lo, hi = (float(np.percentile(finite, 2)),
                      float(np.percentile(finite, 98)))
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0

    # Even-crop the first frame to fix the output size; scale bar / downsample
    # factor drive annotation coordinate scaling.
    rgb0 = even_crop(apply_lut(first, lut, lo, hi))
    out_h, out_w = rgb0.shape[:2]

    # The physical x-scale on the DOWNSAMPLED frame (each output px spans k input px).
    sb_scale = sig_scale_x * k

    # Movie time base (seconds) for each rendered frame — annotation gating +
    # trace resampling both use this.
    times = np.array([i * scale_s for i in idxs], dtype=float)

    inset = None
    if traces:
        inset_w = max(120, int(out_w * _INSET_W_FRAC))
        inset = render_trace_inset(traces, times, inset_w, sig_units or "s")

    ts_font = _load_font(max(12, out_h // 28))
    sb_font = _load_font(max(11, out_h // 32))

    writer = open_writer(path, fps, (out_w, out_h))
    written = 0
    try:
        for fi, t in enumerate(idxs):
            if should_cancel is not None and should_cancel():
                raise _Cancelled()
            frame = first if fi == 0 else downsample(read_frame(raw, t), k)
            rgb = even_crop(apply_lut(frame, lut, lo, hi))
            # Fit every frame to the writer's fixed (out_h, out_w): a LARGER frame is
            # cropped, a SMALLER one (a ragged frame that shrank after downsample /
            # even-crop) is letterbox-PADDED with zeros — the shrink-only slice
            # couldn't pad, so a smaller frame reached the writer with a mismatched
            # shape and raised (finding: ragged-frame writer mismatch).
            if rgb.shape[:2] != (out_h, out_w):
                rgb = _fit_frame(rgb, out_h, out_w)
            img = _pil_rgb(rgb)

            _draw_annotations(img, anns, times[fi], k)
            if timestamp:
                _draw_timestamp(img, times[fi], ts_font)
            if scalebar:
                _draw_scalebar(img, sb_scale, sig_units, sb_font)
            if inset is not None:
                _paste_inset(img, inset, fi)

            writer.append(np.asarray(img.convert("RGB")))
            written += 1
            if progress is not None:
                progress(written, total)
        return written
    finally:
        writer.close()


class _Cancelled(Exception):
    """Raised inside :func:`export_movie` when the cancel predicate trips."""


def _pil_rgb(rgb: np.ndarray):
    from PIL import Image
    return Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), mode="RGB")


def _paste_inset(img, inset: _TraceInset, frame_i: int):
    """Paste the trace inset (bottom-left, above any scale bar area) with alpha."""
    tile = inset.frame_image(frame_i)      # RGBA
    w, h = img.size
    iw, ih = tile.size
    x = max(4, w // 60)
    y = h - ih - max(4, h // 60)
    if y < 0:
        y = 0
    img.paste(tile.convert("RGB"), (x, y), tile.split()[-1])
