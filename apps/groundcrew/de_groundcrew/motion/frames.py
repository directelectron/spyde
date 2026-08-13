"""
frames.py — movie stacks, gain references, and the power-spectrum display.

Ported from the old Ground Crew's `workers/motion_correction_worker.py`
(recovered from commit ``36377f7^``; see MOTION.md). The algorithms are
unchanged — only the Qt transport is gone, so each former `QThread.run()` is a
plain function that can be tested without a GUI.

Two things here are less obvious than they look:

**Gain is matched, not resized.** A super-resolution gain reference is an
integer multiple of the frame it corrects, so it is BINNED down by summing and
dividing — never interpolated. A non-integer ratio raises: silently resampling
a gain reference corrupts every frame it touches, and quietly.

**The power spectrum is SerialEM's, not a plain log.** An electron-microscope
FFT is dominated by the DC peak; `log(|F|)` alone renders as a white dot on
black. `log_fft` reproduces SerialEM's ProcFFT + FindPctStretch — edge taper,
Nyquist-normalised log scaling, then a two-anchor stretch — because that is
what makes Thon rings visible, and matching it means an engineer reads the same
picture in both applications.
"""
from __future__ import annotations

import os

import numpy as np

#: The eight orientations a gain reference can be stored in, in the order the
#: old app's combo box listed them — the index IS the stored setting, so it
#: must not be reordered.
ORIENTATION_LABELS = (
    "Identity", "Rot90", "Rot180", "Rot270",
    "FlipH", "FlipV", "Transpose", "Transverse",
)


def apply_orientation(img: np.ndarray, idx: int) -> np.ndarray:
    """One of the eight orientation transforms, by index into ORIENTATION_LABELS."""
    if idx == 0: return img
    if idx == 1: return np.rot90(img, 1)
    if idx == 2: return np.rot90(img, 2)
    if idx == 3: return np.rot90(img, 3)
    if idx == 4: return np.fliplr(img)
    if idx == 5: return np.flipud(img)
    if idx == 6: return img.T
    if idx == 7: return np.rot90(img, 2).T
    return img


def bin_image(img: np.ndarray, factor: int) -> np.ndarray:
    """Bin by SUMMING factor×factor blocks. Trailing partial blocks are dropped."""
    h, w = img.shape
    nh = (h // factor) * factor
    nw = (w // factor) * factor
    return img[:nh, :nw].reshape(
        nh // factor, factor, nw // factor, factor
    ).sum(axis=(1, 3))


def match_gain_to_frame(gain: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
    """Bring a gain reference to the frame's dimensions.

    Handles the super-resolution case — an Apollo gain can be 2× the data in
    each axis — by binning and converting the sum back to an average. Any ratio
    that is not a clean integer in both axes RAISES rather than resampling:
    a silently interpolated gain corrupts every frame it multiplies.
    """
    gh, gw = gain.shape
    if gh == frame_h and gw == frame_w:
        return gain
    if gh == frame_h * 2 and gw == frame_w * 2:
        return bin_image(gain, 2) / 4.0                    # sum → average
    ratio_h, ratio_w = gh // frame_h, gw // frame_w
    if (ratio_h == ratio_w and ratio_h >= 2
            and gh == frame_h * ratio_h and gw == frame_w * ratio_w):
        return bin_image(gain, ratio_h) / float(ratio_h * ratio_w)
    raise ValueError(
        f"Gain dimensions ({gh}×{gw}) don't match frame ({frame_h}×{frame_w}). "
        "Expected the same size or an integer multiple.")


# ── Loading ───────────────────────────────────────────────────────────────────

def load_movie_stack(path: str) -> tuple[np.ndarray, dict]:
    """Load a movie stack from MRC or TIFF. Returns ``(stack, metadata)``.

    The stack is always 3-D ``(n_frames, h, w)``; a single-frame file loads as
    ``(1, h, w)`` rather than 2-D, so every caller downstream can assume the
    frame axis exists.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mrc", ".mrcs"):
        import mrcfile
        with mrcfile.open(path, permissive=True) as mrc:
            stack = np.asarray(mrc.data)
    elif ext in (".tif", ".tiff"):
        import tifffile
        stack = np.asarray(tifffile.imread(path))
    else:
        raise ValueError(f"Unsupported movie format: {ext or path!r} "
                         "(expected .mrc, .mrcs, .tif or .tiff)")

    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    if stack.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D stack, got shape {stack.shape}")

    n, h, w = stack.shape
    return stack, {"n_frames": int(n), "height": int(h), "width": int(w),
                   "filename": os.path.basename(path), "path": path,
                   "dtype": str(stack.dtype)}


def load_gain(path: str) -> np.ndarray:
    """Load a gain reference from MRC or TIFF, as a 2-D array."""
    stack, _ = load_movie_stack(path)
    gain = stack[0] if stack.shape[0] == 1 else stack.mean(axis=0)
    return np.asarray(gain, dtype=np.float32)


def validate_gain_orientation(frame: np.ndarray,
                              gain: np.ndarray) -> list[tuple[float, str, int]]:
    """Score all eight gain orientations against a frame, best first.

    The right orientation is the one that FLATTENS the frame: a correctly
    oriented gain divides out the detector's fixed pattern, so the corrected
    frame has the lowest relative spread. A wrong orientation superimposes the
    pattern twice and the spread goes up.

    Returns ``[(score, label, index), …]`` where a LOWER score is better.
    """
    fh, fw = frame.shape
    f = np.asarray(frame, dtype=np.float32)
    results: list[tuple[float, str, int]] = []
    for idx, label in enumerate(ORIENTATION_LABELS):
        try:
            g = match_gain_to_frame(apply_orientation(gain, idx), fh, fw)
        except ValueError:
            continue                       # this orientation cannot even fit
        corrected = f * np.asarray(g, dtype=np.float32)
        mean = float(np.mean(corrected))
        if not np.isfinite(mean) or abs(mean) < 1e-9:
            continue
        results.append((float(np.std(corrected) / abs(mean)), label, idx))
    results.sort(key=lambda r: r[0])
    return results


# ── Power spectrum ────────────────────────────────────────────────────────────

def log_fft(image: np.ndarray, *, log_base: float = 5.0,
            trunc_diam: float = 0.01, bkgd_gray: float = 4.0) -> np.ndarray:
    """SerialEM-style power spectrum, as a 0–255 float32 image.

    Pipeline, following SerialEM's ProcFFT + FindPctStretch:

    1. cosine edge-taper (2.5%) so the frame edges do not ring across the FFT
    2. FFT → magnitude
    3. normalise the log scale by the mean magnitude at the Nyquist edge
    4. ``log(logScale · mag + 1)``, scaled against the brightest non-DC peak
    5. two-anchor stretch — white point from a small ring near the centre
       (clips the bright core), black point chosen so the high-frequency
       background lands on *bkgd_gray*

    A plain ``log(|F|)`` renders an EM power spectrum as a white dot on black;
    this is what makes Thon rings visible, and it is the same transform
    SerialEM shows, so the two applications agree.
    """
    h, w = image.shape[-2:]

    # 1. Edge taper — fade the border to the mean, not to zero.
    taper_frac = 0.025
    img = np.asarray(image, dtype=np.float64)
    ty = int(max(h * taper_frac, 1))
    tx = int(max(w * taper_frac, 1))
    ramp_y = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, ty)))
    ramp_x = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, tx)))
    mean_val = float(np.mean(img))
    t = img - mean_val
    t[:ty, :] *= ramp_y[:, None]
    t[-ty:, :] *= ramp_y[::-1, None]
    t[:, :tx] *= ramp_x[None, :]
    t[:, -tx:] *= ramp_x[None, ::-1]
    t += mean_val

    # 2. FFT → magnitude
    mag = np.abs(np.fft.fftshift(np.fft.fft2(t)))
    cy, cx = h // 2, w // 2

    # 3. Normalise the log scale by the Nyquist-edge mean
    border = np.concatenate([mag[0, :], mag[-1, :], mag[:, 0], mag[:, -1]])
    border_mean = float(np.mean(border)) or 1.0
    if border_mean < 1e-12:
        border_mean = 1.0
    log_scale = log_base / border_mean

    # 4. Log scaling, referenced to the brightest peak near the centre that is
    #    NOT the DC term.
    r_search = max(min(h, w) // 10, 5)
    patch = mag[cy - r_search:cy + r_search, cx - r_search:cx + r_search].copy()
    patch[r_search, r_search] = 0.0
    cen_max = float(np.max(patch))
    if cen_max < 1e-12:
        cen_max = 1.0
    scale = 32000.0 / np.log(log_scale * cen_max + 1.0)
    ps = (scale * np.log(log_scale * mag + 1.0)).astype(np.float32)
    ps[cy, cx] = 32000.0

    # 5. Two-anchor stretch
    ring_rad = max(np.sqrt(h * w) * trunc_diam / 2.0, 3.0)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2.0 + (x - cx) ** 2.0)
    ring = (r >= 0.7 * ring_rad) & (r <= ring_rad + 1.0)
    max_scale = float(np.mean(ps[ring])) if np.any(ring) else float(
        np.percentile(ps, 99.9))

    edge_mean = float(np.mean(np.concatenate(
        [ps[0, :], ps[-1, :], ps[1:-1, 0], ps[1:-1, -1]])))
    target = bkgd_gray / 255.0
    min_scale = ((edge_mean - max_scale * target) / (1.0 - target)
                 if abs(1.0 - target) > 1e-6 else edge_mean)

    denom = max_scale - min_scale
    if denom < 1e-6:
        denom = 1.0
    return np.clip((ps - min_scale) / denom * 255.0, 0, 255).astype(np.float32)


def save_image(image: np.ndarray, path: str) -> str:
    """Write a result image to MRC or TIFF, chosen by extension."""
    ext = os.path.splitext(path)[1].lower()
    arr = np.asarray(image, dtype=np.float32)
    if ext in (".mrc", ".mrcs"):
        import mrcfile
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(arr)
    elif ext in (".tif", ".tiff"):
        import tifffile
        tifffile.imwrite(path, arr)
    else:
        raise ValueError(f"Unsupported output format: {ext or path!r}")
    return path
