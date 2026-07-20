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


# ── crop (source pixel space, applied BEFORE downsample) ──────────────────────────

def apply_crop(frame: np.ndarray, crop) -> np.ndarray:
    """Crop a 2-D *frame* to ``[x0, y0, x1, y1]`` (source pixel space, the same
    coordinate system annotations use). ``None`` / a malformed / degenerate rect →
    the full frame unchanged. Clamped to the frame bounds so a stale crop from a
    larger source can never index out of range."""
    if not crop:
        return frame
    try:
        x0, y0, x1, y1 = (int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3]))
    except (TypeError, ValueError, IndexError):
        return frame
    h, w = frame.shape[:2]
    x0 = max(0, min(x0, w))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h))
    y1 = max(0, min(y1, h))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return frame
    return frame[y0:y1, x0:x1]


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


def _draw_annotations(img, anns, t_sec: float, k: int, crop_origin=(0, 0)):
    """Draw each time-gated annotation. Coordinates are in the ORIGINAL (uncropped)
    source pixel space; we subtract the crop origin (so a cropped export places them
    correctly) then divide by the downsample factor *k* to land on the shrunk
    frame."""
    from PIL import ImageDraw
    if not anns:
        return
    draw = ImageDraw.Draw(img, "RGBA")
    ox, oy = float(crop_origin[0]), float(crop_origin[1])

    def sc(v, o=0.0):
        return (float(v) - o) / max(1, k)

    for ann in anns:
        if not isinstance(ann, dict) or not _in_time_range(t_sec, ann):
            continue
        kind = str(ann.get("kind", ""))
        color = _color(ann.get("color", "#ffcc00"))
        try:
            if kind == "text":
                xy = ann.get("xy", [0, 0])
                font = _load_font(int(ann.get("size", 16)))
                _text_with_shadow(draw, (sc(xy[0], ox), sc(xy[1], oy)),
                                  str(ann.get("text", "")), font,
                                  color, (0, 0, 0, 220))
            elif kind == "circle":
                xy = ann.get("xy", [0, 0])
                r = sc(ann.get("radius", 10))
                cx, cy = sc(xy[0], ox), sc(xy[1], oy)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                             outline=color, width=max(1, int(sc(ann.get("width", 3)))))
            elif kind == "rect":
                xy = ann.get("xy", [0, 0])
                wh = ann.get("wh", [10, 10])
                x0, y0 = sc(xy[0], ox), sc(xy[1], oy)
                draw.rectangle([x0, y0, x0 + sc(wh[0]), y0 + sc(wh[1])],
                               outline=color, width=max(1, int(sc(ann.get("width", 3)))))
            elif kind == "arrow":
                xy = ann.get("xy", [0, 0])         # tail
                xy2 = ann.get("xy2", [10, 10])     # head
                _draw_arrow(draw, (sc(xy[0], ox), sc(xy[1], oy)),
                            (sc(xy2[0], ox), sc(xy2[1], oy)), color,
                            max(1, int(sc(ann.get("width", 3)))))
        except Exception as e:
            log.debug("annotation %r draw failed: %s", kind, e)


def _draw_text_overlays(img, overlays, values_at_t, t_sec: float, k: int,
                        crop_origin=(0, 0)):
    """Draw each 1-D-signal-as-text overlay: a live value formatted as text (e.g.
    ``"T = 812.3 °C"``). *overlays* is the list of overlay dicts; *values_at_t* is
    a parallel list of the current numeric value for each (already resampled onto
    the movie time base and indexed at this frame) — ``None`` for an overlay with
    no resolved signal, which then paints its label + a dash. Coordinates are in
    ORIGINAL (uncropped) source pixels; subtract the crop origin then divide by the
    downsample factor *k*."""
    from PIL import ImageDraw
    if not overlays:
        return
    draw = ImageDraw.Draw(img, "RGBA")
    ox, oy = float(crop_origin[0]), float(crop_origin[1])

    def sc(v, o=0.0):
        return (float(v) - o) / max(1, k)

    for ov, val in zip(overlays, values_at_t):
        if not isinstance(ov, dict) or not _in_time_range(t_sec, ov):
            continue
        try:
            xy = ov.get("xy", [8, 8])
            font = _load_font(int(ov.get("size", 18)))
            color = _color(ov.get("color", "#ffffff"))
            label = str(ov.get("label", "") or "")
            units = str(ov.get("units", "") or "")
            fmt = str(ov.get("fmt", "") or "")
            if val is None:
                text = f"{label} = —" if label else "—"
            elif fmt:
                # A user format string; fall back to a plain render if it's malformed.
                try:
                    text = fmt.format(label=label, value=float(val), units=units)
                except (KeyError, IndexError, ValueError):
                    text = f"{label} = {float(val):.2f} {units}".strip()
            else:
                text = f"{label} = {float(val):.2f} {units}".strip()
            _text_with_shadow(draw, (sc(xy[0], ox), sc(xy[1], oy)), text, font,
                              color, (0, 0, 0, 220))
        except Exception as e:
            log.debug("text overlay draw failed: %s", e)


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


def frame_indices_with_freezes(n_frames: int, t_start: int, t_end: int,
                               stride: int, freezes, fps: float):
    """The render sequence WITH freeze holds expanded in.

    Starts from :func:`frame_indices`, then for every ``freeze`` ``{t, hold_s}``
    whose ``t`` lands ON a rendered index, that index is REPEATED
    ``round(hold_s * fps)`` extra times (a hold — the frame lingers on screen for
    ``hold_s`` seconds at the export fps). A freeze on a ``t`` not in the sequence
    (outside the range / skipped by stride) is ignored. Order is preserved: the
    repeats are inserted immediately after the frame's first occurrence.

    Returns a plain list of source indices (with duplicates for the holds) — the
    export loop renders each in turn, so a frozen frame is simply re-encoded, and
    the time base still advances by the frame's OWN time (a freeze shows the same
    picture while the timestamp holds)."""
    base = frame_indices(n_frames, t_start, t_end, stride)
    if not freezes:
        return base
    # Map source index -> extra repeat count from the freezes.
    extra: dict[int, int] = {}
    for fz in freezes:
        try:
            t = int(fz.get("t"))
            hold_s = float(fz.get("hold_s", 0.0) or 0.0)
        except (TypeError, ValueError, AttributeError):
            continue
        reps = int(round(hold_s * float(fps)))
        if reps > 0:
            extra[t] = extra.get(t, 0) + reps
    if not extra:
        return base
    out: list[int] = []
    for t in base:
        out.append(t)
        for _ in range(extra.get(t, 0)):
            out.append(t)
    return out


def frame_indices_with_speed(n_frames: int, t_start: int, t_end: int, stride: int,
                             speed_segments, fps: float, scale_s: float,
                             freezes=None):
    """The render sequence resampled through VARIABLE-SPEED segments.

    ``speed_segments`` is a list of ``{"time_range":[s0,s1] (seconds), "speed":m}``.
    Source time inside a segment advances at ``m×`` (0 = hold, <1 slow-mo → more
    output frames, >1 fast-forward → fewer). Outside any segment the speed is 1×.

    The output is one source index per OUTPUT frame (fixed 1/fps output steps): a
    source-time cursor walks ``t_start..t_end`` at the local speed; each output step
    emits the nearest source index. A very slow / hold segment repeats indices; a
    fast one skips them. Falls back to :func:`frame_indices_with_freezes` when there
    are no speed segments (so legacy freezes still work)."""
    if not speed_segments:
        return frame_indices_with_freezes(n_frames, t_start, t_end, stride,
                                          freezes or [], fps)
    t0 = max(0, int(t_start))
    t1 = min(int(n_frames) - 1, int(t_end))
    if t1 < t0:
        return []
    sc = float(scale_s) if scale_s and scale_s > 0 else 1.0
    out_dt = 1.0 / max(1.0, float(fps))          # output seconds per frame

    def speed_at(src_frame: float) -> float:
        t_sec = src_frame * sc
        for seg in speed_segments:
            try:
                tr = seg.get("time_range")
                if tr and float(tr[0]) <= t_sec <= float(tr[1]):
                    return max(0.0, float(seg.get("speed", 1.0)))
            except (TypeError, ValueError, AttributeError):
                continue
        return 1.0

    out: list[int] = []
    cur = float(t0)
    # A hard cap so a 0×/tiny-speed segment spanning the whole movie can't loop
    # forever — 10 minutes of output at this fps.
    cap = int(600 * fps)
    while cur <= t1 + 0.5 and len(out) < cap:
        out.append(int(round(min(cur, t1))))
        m = speed_at(cur)
        # advance the source cursor by (speed * out_dt) SECONDS → frames.
        adv = (m * out_dt) / sc
        # A 0× (hold) segment still needs to eventually escape: nudge by a tiny
        # epsilon so a hold at the very end doesn't wedge (it emits the same frame
        # repeatedly while inside the segment's time window, which is the point).
        if adv <= 0:
            # Peek: if we're inside a hold segment, emit holds for its DURATION.
            held = _hold_frames(speed_segments, cur * sc, out_dt, fps)
            for _ in range(max(0, held - 1)):
                if len(out) >= cap:
                    break
                out.append(int(round(min(cur, t1))))
            # Then step past the segment end.
            cur = _segment_end_frame(speed_segments, cur * sc, sc, t1) + (stride or 1)
        else:
            cur += adv
    return out


def _hold_frames(segments, t_sec, out_dt, fps) -> int:
    """How many output frames a 0× hold at *t_sec* should last (its duration × fps)."""
    for seg in segments:
        try:
            tr = seg.get("time_range")
            if tr and float(seg.get("speed", 1)) == 0 and float(tr[0]) <= t_sec <= float(tr[1]):
                return max(1, int(round((float(tr[1]) - float(tr[0])) * fps)))
        except (TypeError, ValueError, AttributeError):
            continue
    return 1


def _segment_end_frame(segments, t_sec, sc, t1) -> float:
    """The source frame index at the end of the hold segment containing *t_sec*."""
    for seg in segments:
        try:
            tr = seg.get("time_range")
            if tr and float(tr[0]) <= t_sec <= float(tr[1]):
                return min(t1, float(tr[1]) / sc)
        except (TypeError, ValueError, AttributeError):
            continue
    return t_sec / sc


def _base_frame(raw, t: int, crop, k: int) -> np.ndarray:
    """Read source frame *t*, apply the CROP (source px, before downsample), then
    the spatial downsample. Memory-safe: one :func:`read_frame` slice."""
    return downsample(apply_crop(read_frame(raw, t), crop), k)


def _resolve_clim(first: np.ndarray, clim):
    """The ``(lo, hi)`` contrast window: an explicit ``clim`` [lo, hi] when set,
    else a robust 2–98% auto-window from the FIRST rendered frame."""
    if clim and len(clim) == 2 and clim[0] is not None and clim[1] is not None:
        return float(clim[0]), float(clim[1])
    finite = first[np.isfinite(first)]
    if finite.size:
        lo, hi = (float(np.percentile(finite, 2)),
                  float(np.percentile(finite, 98)))
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _compose_frame(frame, lut, lo, hi, out_h, out_w, *, t_sec, k,
                   anns, text_overlays, text_values, timestamp, scalebar,
                   sb_scale, sig_units, ts_font, sb_font, inset=None, inset_i=0,
                   overlay=None, raw_over=None, src_t=0, crop_origin=(0, 0)):
    """LUT + fit + the whole overlay stack for ONE already-read (cropped +
    downsampled) *frame* → a PIL RGB image. Shared by :func:`export_movie` and the
    single-frame preview so the editor preview is byte-for-byte what exports.

    ``crop_origin`` (the crop's ``(x0, y0)`` in source px, ``(0,0)`` when uncropped)
    shifts the annotation / text-overlay coordinates so they land correctly on a
    cropped frame. A 2nd-image *overlay* (with its lazy array *raw_over*) is
    alpha/screen-composited onto the base BEFORE the drawn overlays (so
    annotations/timestamp sit on top)."""
    rgb = even_crop(apply_lut(frame, lut, lo, hi))
    # Fit every frame to the writer's fixed (out_h, out_w): a LARGER frame is
    # cropped, a SMALLER (ragged / shrunk) one is letterbox-PADDED with zeros.
    if rgb.shape[:2] != (out_h, out_w):
        rgb = _fit_frame(rgb, out_h, out_w)
    if overlay is not None and raw_over is not None:
        rgb = _composite_overlay(rgb, overlay, raw_over, src_t, k, t_sec, out_h, out_w)
    img = _pil_rgb(rgb)
    _draw_annotations(img, anns, t_sec, k, crop_origin)
    if text_overlays:
        _draw_text_overlays(img, text_overlays, text_values, t_sec, k, crop_origin)
    if timestamp:
        _draw_timestamp(img, t_sec, ts_font)
    if scalebar:
        _draw_scalebar(img, sb_scale, sig_units, sb_font)
    if inset is not None:
        _paste_inset(img, inset, inset_i)
    return img


def render_single_frame(raw, t: int, *, params: dict, n_frames: int,
                        scale_s: float, sig_scale_x: float, sig_units: str,
                        text_overlays=None, text_values=None, overlay=None):
    """Render ONE composed frame at source index *t* → a PIL RGB image. The
    editor's live preview path — MEMORY-SAFE (a single :func:`read_frame` slice),
    no writer, no full sequence. Uses the SAME LUT + overlay stack as the export so
    the preview matches. Auto-contrast (when ``clim`` is unset) is computed from
    THIS frame."""
    p = params
    k = int(p.get("downsample", 1) or 1)
    crop = p.get("crop")
    cmap = str(p.get("cmap", "gray") or "gray")
    timestamp = bool(p.get("timestamp", True))
    scalebar = bool(p.get("scalebar", True)) and sig_scale_x > 0
    anns = p.get("annotations") or []
    lut = build_lut(cmap)
    frame = _base_frame(raw, int(t), crop, k)
    lo, hi = _resolve_clim(frame, p.get("clim"))
    rgb0 = even_crop(apply_lut(frame, lut, lo, hi))
    out_h, out_w = rgb0.shape[:2]
    sb_scale = sig_scale_x * k
    ts_font = _load_font(max(12, out_h // 28))
    sb_font = _load_font(max(11, out_h // 32))
    t_sec = int(t) * scale_s
    crop_origin = (int(crop[0]), int(crop[1])) if crop and len(crop) == 4 else (0, 0)
    raw_over = overlay.get("raw") if overlay else None
    return _compose_frame(
        frame, lut, lo, hi, out_h, out_w, t_sec=t_sec, k=k,
        anns=anns, text_overlays=(text_overlays or []),
        text_values=(text_values or []), timestamp=timestamp, scalebar=scalebar,
        sb_scale=sb_scale, sig_units=sig_units, ts_font=ts_font, sb_font=sb_font,
        crop_origin=crop_origin, overlay=overlay, raw_over=raw_over, src_t=int(t))


def _composite_overlay(base_rgb: np.ndarray, overlay, raw_over, t: int, k: int,
                       t_sec: float, out_h: int, out_w: int) -> np.ndarray:
    """Alpha/screen-composite a SECOND image over *base_rgb* (an (H,W,3) uint8).

    *overlay* is the resolved 2nd-image spec ``{raw, alpha, cmap, clim, blend,
    time_range, downsample}``; *raw_over* is its lazy/numpy array. A 3-D source is a
    TIME STACK (read one frame at the current index); a 2-D source is a STATIC image
    (same on every frame). Time-gated by ``time_range`` (seconds). Returns the
    blended RGB. On any failure returns *base_rgb* unchanged (overlay is cosmetic)."""
    try:
        if overlay is None or raw_over is None:
            return base_rgb
        tr = overlay.get("time_range")
        if tr and not (float(tr[0]) <= t_sec <= float(tr[1])):
            return base_rgb
        ov_k = int(overlay.get("downsample", k) or k)
        # A 3-D overlay is a time stack → read the frame at the current index; a 2-D
        # overlay is a static image → use it directly (memory-safe single slice).
        if getattr(raw_over, "ndim", 2) >= 3:
            ti = min(int(t), raw_over.shape[0] - 1)
            over_frame = read_frame(raw_over, ti)
        else:
            over_frame = np.asarray(raw_over)
        of = downsample(over_frame, ov_k)
        olut = build_lut(str(overlay.get("cmap", "magma") or "magma"))
        oclim = overlay.get("clim")
        olo, ohi = _resolve_clim(of, oclim)
        orgb = even_crop(apply_lut(of, olut, olo, ohi))
        if orgb.shape[:2] != (out_h, out_w):
            orgb = _fit_frame(orgb, out_h, out_w)
        alpha = float(overlay.get("alpha", 0.5) or 0.5)
        blend = str(overlay.get("blend", "over") or "over")
        b = base_rgb.astype(np.float32)
        o = orgb.astype(np.float32)
        if blend == "screen":
            mixed = 255.0 - (255.0 - b) * (255.0 - o) / 255.0
            out = b * (1.0 - alpha) + mixed * alpha
        else:  # "over" — straight alpha
            out = b * (1.0 - alpha) + o * alpha
        return np.clip(out, 0, 255).astype(np.uint8)
    except Exception as e:
        log.debug("overlay-image composite failed: %s", e)
        return base_rgb


def export_movie(raw, *, path: str, params: dict, n_frames: int,
                 scale_s: float, sig_scale_x: float, sig_units: str,
                 traces=None, text_overlays=None, overlay=None,
                 should_cancel=None, progress=None) -> int:
    """Render the movie to *path*. Returns the number of frames written.

    MEMORY-SAFE: reads one frame per iteration via :func:`read_frame` (a lazy
    slice ``raw[t].compute()`` / numpy slice) — the full dataset is NEVER
    materialised.

    Applies the ``crop`` (source px) before downsample, expands freeze holds into
    the render sequence, and draws the annotation / 1-D-signal-as-text / trace-inset
    overlays. *should_cancel* is a 0-arg predicate polled per frame (returns True →
    stop and raise :class:`_Cancelled`). *progress(done, total)* narrates. The
    caller owns partial-file cleanup; on cancel we raise so the handler removes the
    file.
    """
    from spyde.actions.movie_export.encoder import open_writer

    p = params
    k = int(p.get("downsample", 1) or 1)
    fps = float(p.get("fps", 10.0) or 10.0)
    cmap = str(p.get("cmap", "gray") or "gray")
    clim = p.get("clim")
    crop = p.get("crop")
    timestamp = bool(p.get("timestamp", True))
    scalebar = bool(p.get("scalebar", True)) and sig_scale_x > 0
    anns = p.get("annotations") or []
    text_overlays = list(text_overlays or [])

    # Variable-speed segments (slow / fast-forward / hold) resample source→output
    # time; when absent this falls back to legacy freeze-holds. A hold segment (0×)
    # subsumes the freeze.
    idxs = frame_indices_with_speed(
        n_frames, p.get("t_start", 0), p.get("t_end", n_frames - 1),
        p.get("stride", 1), p.get("speed_segments") or [], fps, scale_s,
        freezes=p.get("freezes") or [])
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
    first = _base_frame(raw, idxs[0], crop, k)

    # Cancel immediately AFTER the probe read too (a cancel that arrived while the
    # read was in flight stops us before we open the writer / encode any frame).
    if should_cancel is not None and should_cancel():
        raise _Cancelled()

    lo, hi = _resolve_clim(first, clim)

    # Even-crop the first frame to fix the COMPOSE size; scale bar / downsample
    # factor drive annotation coordinate scaling.
    rgb0 = even_crop(apply_lut(first, lut, lo, hi))
    out_h, out_w = rgb0.shape[:2]

    # Optional final OUTPUT size (spec.out_size [w, h]) — the composed frame is
    # resized to it (even-dimensioned for H.264) as the very last step. None → the
    # frame's own (crop + downsample) size is the output.
    os_spec = p.get("out_size")
    final_wh = None
    if os_spec and len(os_spec) == 2 and os_spec[0] and os_spec[1]:
        fw2, fh2 = int(os_spec[0]) - (int(os_spec[0]) % 2), int(os_spec[1]) - (int(os_spec[1]) % 2)
        if fw2 >= 2 and fh2 >= 2 and (fw2, fh2) != (out_w, out_h):
            final_wh = (fw2, fh2)

    # The physical x-scale on the DOWNSAMPLED frame (each output px spans k input px).
    sb_scale = sig_scale_x * k

    # Movie time base (seconds) for each rendered frame — annotation gating +
    # trace/text resampling all use this. (A freeze repeats an index, so the same
    # t_sec appears consecutively — the timestamp holds, correct for a freeze.)
    times = np.array([i * scale_s for i in idxs], dtype=float)

    inset = None
    if traces:
        inset_w = max(120, int(out_w * _INSET_W_FRAC))
        inset = render_trace_inset(traces, times, inset_w, sig_units or "s")

    # Resample each 1-D-signal-as-text overlay onto the movie time base ONCE.
    text_resampled = _resample_text_overlays(text_overlays, times, src_indices=idxs)

    ts_font = _load_font(max(12, out_h // 28))
    sb_font = _load_font(max(11, out_h // 32))

    raw_over = overlay.get("raw") if overlay else None
    crop_origin = (int(crop[0]), int(crop[1])) if crop and len(crop) == 4 else (0, 0)

    writer_wh = final_wh if final_wh is not None else (out_w, out_h)
    writer = open_writer(path, fps, writer_wh)
    written = 0
    try:
        for fi, t in enumerate(idxs):
            if should_cancel is not None and should_cancel():
                raise _Cancelled()
            frame = first if fi == 0 else _base_frame(raw, t, crop, k)
            text_values = [(r[fi] if r is not None else None) for r in text_resampled]
            img = _compose_frame(
                frame, lut, lo, hi, out_h, out_w, t_sec=times[fi], k=k,
                anns=anns, text_overlays=text_overlays, text_values=text_values,
                timestamp=timestamp, scalebar=scalebar, sb_scale=sb_scale,
                sig_units=sig_units, ts_font=ts_font, sb_font=sb_font,
                inset=inset, inset_i=fi,
                overlay=overlay, raw_over=raw_over, src_t=t,
                crop_origin=crop_origin)
            if final_wh is not None:
                from PIL import Image as _Image
                img = img.resize(final_wh, _Image.LANCZOS)
            writer.append(np.asarray(img.convert("RGB")))
            written += 1
            if progress is not None:
                progress(written, total)
        return written
    finally:
        writer.close()


def _resample_text_overlays(text_overlays, times, src_indices=None):
    """For each 1-D-signal-as-text overlay carrying a captured trace (``_trace`` —
    a :class:`TraceSpec`), produce its per-rendered-frame values. When the trace
    has ONE point per source frame (a per-frame column, e.g. a temperature log) and
    ``src_indices`` (the source frame index per rendered frame) is given, read it by
    FRAME INDEX (index-aligned); otherwise resample by physical time on *times*. An
    overlay with no trace yields ``None`` (paints label + dash). Returns a list
    parallel to *text_overlays*."""
    out = []
    n_src = None
    if src_indices is not None:
        idx = np.asarray(src_indices, dtype=int)
    for ov in text_overlays:
        tr = ov.get("_trace") if isinstance(ov, dict) else None
        if tr is not None and hasattr(tr, "resample"):
            try:
                y = np.asarray(getattr(tr, "y", None))
                if (src_indices is not None and y is not None
                        and y.size and int(y.size) == int(_src_span(src_indices))):
                    # Per-source-frame column → index-align to the rendered frames.
                    out.append(y[np.clip(idx, 0, y.size - 1)].astype(float))
                    continue
                out.append(np.asarray(tr.resample(times), dtype=float))
                continue
            except Exception as e:
                log.debug("text-overlay resample failed: %s", e)
        out.append(None)
    return out


def _src_span(src_indices) -> int:
    """The number of DISTINCT source frames a per-frame column would have (max
    index + 1) — used to decide whether a trace is index-aligned to the movie."""
    a = np.asarray(src_indices, dtype=int)
    return int(a.max()) + 1 if a.size else 0


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
