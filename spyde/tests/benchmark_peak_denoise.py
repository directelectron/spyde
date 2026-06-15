"""
Navigation-dimension denoiser bake-off for peak finding: Gaussian blur vs TV
denoising applied ACROSS the scan (nav) axes, not within the detector frame.

Adjacent probe positions see nearly the same diffraction pattern, so averaging a
detector pixel's value across neighbouring scan positions suppresses per-frame
Poisson noise before NXCORR peak finding. The pipeline currently does this with a
nav-space Gaussian (NavBlurCache: gaussian_filter(sigma=(s,s,0,0))). Question
(user, 2026-06-15): does nav-space TV denoising beat nav-space Gaussian for peak
detection — especially where the pattern changes sharply across the scan (grain
boundaries), where Gaussian smears two orientations together but TV shouldn't?

Run:
    python -m spyde.tests.benchmark_peak_denoise

Synthetic 4D dataset: a scan with two grains (a sharp boundary) whose diffraction
spots sit at different positions, plus per-frame Poisson noise. We denoise across
the nav axes (Gaussian vs TV vs none), run NXCORR peak finding per frame, and
score detection vs the KNOWN per-position spots. Reports precision/recall/F1 and
sub-pixel position error, separately for interior vs boundary positions.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from spyde.actions.find_vectors import _find_vectors_single_frame


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic 4D dataset: (ny, nx, H, W) with a grain boundary across the scan
# ─────────────────────────────────────────────────────────────────────────────

def make_4d(ny=20, nx=20, H=64, W=64, radius=3, n_spots=10,
            peak_counts=12.0, bg_counts=2.0, seed=0):
    """Return (noisy 4D float32, true_xy per position list-of-(M,2)).

    Two grains split at x = nx//2: each grain's spots sit at a fixed set of
    detector positions (different orientation), so across the scan the patterns
    are piecewise-constant with one sharp boundary. Within a grain there is a
    gentle sub-pixel drift (lattice strain) so positions aren't bit-identical.
    Per-frame Poisson noise at `peak_counts` dose.
    """
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:H, 0:W]

    def _spots(theta, scale):
        out = []
        for i in range(n_spots):
            ang = 2 * np.pi * i / n_spots + theta
            rr = scale * 0.32 * min(H, W)
            out.append((H / 2 + rr * np.sin(ang), W / 2 + rr * np.cos(ang)))
        return np.array(out)

    grainA = _spots(0.0, 1.0)
    grainB = _spots(np.deg2rad(18.0), 1.0)   # second orientation

    data = np.empty((ny, nx, H, W), np.float32)
    true_xy = [[None] * nx for _ in range(ny)]
    for iy in range(ny):
        for ix in range(nx):
            base = grainA if ix < nx // 2 else grainB
            # gentle intra-grain drift (strain), sub-pixel, smooth in space
            drift = 0.6 * np.array([np.sin(iy / ny * np.pi),
                                    np.cos(ix / nx * np.pi)])
            spots = base + drift
            clean = np.full((H, W), bg_counts, np.float64)
            for (cy, cx) in spots:
                amp = peak_counts * rng.uniform(0.6, 1.0)
                # Gaussian spot (real diffraction spots are peaked, not flat
                # disks — flat disks give NXCORR a plateau and spurious NMS hits)
                clean += amp * np.exp(
                    -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (radius * 0.7) ** 2))
            data[iy, ix] = rng.poisson(clean).astype(np.float32)
            true_xy[iy][ix] = spots
    return data, true_xy


# ─────────────────────────────────────────────────────────────────────────────
# Nav-dimension denoisers (operate on axes 0,1; leave detector axes 2,3 alone)
# ─────────────────────────────────────────────────────────────────────────────

def nav_none(data):
    return data


def nav_gaussian(data, sigma=1.0):
    """Gaussian blur across the two nav axes only (the current pipeline)."""
    return gaussian_filter(data, sigma=(sigma, sigma, 0, 0))


def nav_tv(data, weight=0.5):
    """TV denoise across the nav axes only. Treat each detector pixel's
    (ny, nx) map as an image and TV-denoise it; vectorised over detector pixels
    via reshape. Piecewise-constant prior should keep grain boundaries sharp."""
    from skimage.restoration import denoise_tv_chambolle
    ny, nx, H, W = data.shape
    out = np.empty_like(data)
    # denoise each detector pixel's nav-map; normalise per-pixel for stable weight
    for ky in range(H):
        for kx in range(W):
            m = data[:, :, ky, kx].astype(np.float32)
            lo, hi = float(m.min()), float(m.max())
            span = (hi - lo) or 1.0
            d = denoise_tv_chambolle((m - lo) / span, weight=weight)
            out[:, :, ky, kx] = d * span + lo
    return out


DENOISERS = {
    "none": nav_none,
    "gaussian": lambda d: nav_gaussian(d, sigma=1.0),
    "tv": lambda d: nav_tv(d, weight=0.5),
}


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score(detected_xy, true_xy, tol=2.0):
    if len(true_xy) == 0 or len(detected_xy) == 0:
        return (0.0, 0.0, 0.0, np.nan)
    d = np.sqrt(((detected_xy[:, None, :] - true_xy[None, :, :]) ** 2).sum(-1))
    md, mt, errs = set(), set(), []
    order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    for di, ti in order:
        if d[di, ti] > tol:
            break
        if di in md or ti in mt:
            continue
        md.add(int(di)); mt.add(int(ti)); errs.append(d[di, ti])
    tp = len(mt)
    prec = tp / len(detected_xy)
    rec = tp / len(true_xy)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return (prec, rec, f1, float(np.median(errs)) if errs else np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run(radius=3, threshold=0.4, min_distance=4):
    doses = [("medium", 30.0), ("low", 10.0), ("very_low", 4.0)]
    print(f"nav-dim denoise: radius={radius} thr={threshold} "
          f"min_dist={min_distance}")
    for dose_name, pc in doses:
        data, true_xy = make_4d(peak_counts=pc, seed=0)
        ny, nx, H, W = data.shape
        bcol = nx // 2
        for name, fn in DENOISERS.items():
            den = fn(data)
            # interior = away from the boundary column; boundary = adjacent to it
            res = {"interior": [], "boundary": []}
            for iy in range(ny):
                for ix in range(nx):
                    _cm, _rc, peaks = _find_vectors_single_frame(
                        den[iy, ix].astype(np.float32), radius, threshold,
                        min_distance, subpixel=True)
                    det = peaks[:, :2] if len(peaks) else np.zeros((0, 2))
                    sc = score(det, true_xy[iy][ix])
                    key = "boundary" if abs(ix - bcol) <= 1 else "interior"
                    res[key].append(sc)
            line = []
            for region in ("interior", "boundary"):
                a = np.array(res[region], float)
                line.append(f"{region[:3]} F1 {np.nanmean(a[:,2]):.3f} "
                            f"err {np.nanmedian(a[:,3]):.3f}")
            print(f"  dose={dose_name:8s} {name:9s}: " + " | ".join(line))


if __name__ == "__main__":
    run()
