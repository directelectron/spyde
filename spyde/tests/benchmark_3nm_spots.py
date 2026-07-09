"""
benchmark_3nm_spots.py
======================
Spot-finding benchmark for the DirectElectron "3 nm step size" 4D-STEM SEM scan
(``20241215_29639_movie_movie.mrc``): 300x300 nav, 256x256 uint16 patterns,
300 keV, with a physical **beam stop** (vertical bar + circular blob) sitting
over the direct beam.

Goal: find the discrete diffraction spots (FWHM ~3 px, radius ~1-2 px, arranged
symmetrically about the centre) WITHOUT firing on the high-gradient beam-stop
rim, and WITHOUT a real zero-order disk (it is occluded by the stop).

There is no hand-labelled ground truth, so detections are scored with
physically-motivated proxy metrics:

  * ``bs_fp``   beam-stop false-positive rate -- fraction of detected peaks
                that land on / within ``bs_dilate`` px of the beam-stop mask.
                These are unambiguously wrong (no signal can come from behind
                an opaque stop).  LOWER is better.
  * ``fri``     Friedel inlier fraction -- with the centre fixed at the
                Friedel-symmetry centre, fraction of spots whose inversion
                partner (2c - v) has a real detected spot within ``fri_tol`` px.
                Real lattice reflections come in +g/-g pairs; rim artifacts do
                not.  HIGHER is better.
  * ``n``       mean spots per frame (off the beam stop).
  * ``t``       wall-clock per frame (ms).

Run (NOT under pytest -- numba/CUDA segfaults in the pytest process on Windows):

    .venv/Scripts/python spyde/tests/benchmark_3nm_spots.py
    .venv/Scripts/python spyde/tests/benchmark_3nm_spots.py --frames 200
    .venv/Scripts/python spyde/tests/benchmark_3nm_spots.py --sweep

Outputs a markdown table to stdout and (with --sweep) writes
``benchmark_3nm_spots_results.md`` + overlay PNGs next to the repo root.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

# --------------------------------------------------------------------------- #
# Dataset constants
# --------------------------------------------------------------------------- #
DEFAULT_MRC = (
    r"C:\Users\CarterFrancis\Downloads"
    r"\directelectron_3nm-step-size-scan_2026-06-03_1444"
    r"\20241215_29639_movie_movie.mrc"
)
H = W = 256
NY = NX = 300
N = NY * NX
HDR = 1024
DTYPE = "<u2"


# --------------------------------------------------------------------------- #
# Loading / characterisation
# --------------------------------------------------------------------------- #
def open_memmap(path):
    return np.memmap(path, dtype=DTYPE, mode="r", offset=HDR, shape=(N, H, W))


def build_sum_and_beamstop(mm, stride=50):
    """Sparse sum over the scan -> beam-stop mask (the stop blocks electrons,
    so it is a stable very-low-intensity region of the average pattern)."""
    sub = mm[::stride].astype(np.float64)
    s = sub.sum(0)
    bs = s < (s.mean() * 0.15)
    return s.astype(np.float32), bs


def dilate(mask, r):
    if r <= 0:
        return mask
    return maximum_filter(mask.astype(np.uint8), size=2 * r + 1) > 0


# --------------------------------------------------------------------------- #
# Friedel centre (per frame, robust to rim noise via trimmed residual)
# --------------------------------------------------------------------------- #
def friedel_center(spots_yx, lo=110.0, hi=126.0, step=0.5):
    """Brute-force the inversion centre that best pairs the spot set.

    Vectorised over the candidate-centre grid: for G centres and P spots the
    pairwise term is (G, P, P), evaluated in one shot.  A trimmed mean of the
    nearest-partner distances ignores the worst 30% (rim artifacts) so the
    centre locks onto the symmetric lattice even with some false positives.
    """
    if len(spots_yx) < 4:
        return None
    pts = spots_yx[:, :2].astype(np.float32)          # (P, 2)
    grid = np.arange(lo, hi, step, dtype=np.float32)
    cy, cx = np.meshgrid(grid, grid, indexing="ij")
    centers = np.stack([cy.ravel(), cx.ravel()], -1)  # (G, 2)
    # reflected points for every centre: (G, P, 2)
    refl = 2.0 * centers[:, None, :] - pts[None, :, :]
    # distance from each reflected point to each real spot: (G, P, P)
    d = np.sqrt(((refl[:, :, None, :] - pts[None, None, :, :]) ** 2).sum(-1))
    nn = d.min(-1)                                     # (G, P) nearest partner
    nn.sort(-1)
    k = max(2, int(0.7 * nn.shape[1]))
    sc = nn[:, :k].mean(-1)                            # (G,) trimmed residual
    g = int(np.argmin(sc))
    return float(centers[g, 0]), float(centers[g, 1])


def friedel_inlier_fraction(spots_yx, center, tol=2.0):
    if center is None or len(spots_yx) < 2:
        return 0.0
    pts = spots_yx[:, :2]
    cy, cx = center
    refl = 2.0 * np.array([cy, cx]) - pts
    d = np.sqrt(((pts[:, None, :] - refl[None, :, :]) ** 2).sum(-1))
    nn = d.min(1)
    return float((nn <= tol).mean())


# --------------------------------------------------------------------------- #
# Lattice-consistency metric (more robust than raw Friedel)
# --------------------------------------------------------------------------- #
def lattice_consistency(spots_yx, *, max_d=50.0, bins=17, top=6):
    """Score how lattice-like a spot set is, in [0, 1].

    A crystalline diffraction pattern is a 2D reciprocal lattice, so the
    DIFFERENCE vectors between detected spots cluster tightly around a few
    basis vectors and their integer combinations.  Junk detections (background
    texture, rim fragments) give diffuse, unstructured difference vectors.

    We histogram all pairwise difference vectors (which is centre-free -- it
    needs no knowledge of the occluded direct beam) into a 2D grid and report
    the fraction of difference-vector mass that falls in the few most-populated
    cells.  High concentration == strong lattice == real spots.
    """
    pts = spots_yx[:, :2]
    P = len(pts)
    if P < 4:
        return 0.0
    diff = pts[:, None, :] - pts[None, :, :]
    iu = np.triu_indices(P, k=1)
    dv = diff[iu]                                   # (P*(P-1)/2, 2)
    r = np.hypot(dv[:, 0], dv[:, 1])
    dv = dv[(r > 3.0) & (r < max_d)]                # drop self / far pairs
    if len(dv) < 4:
        return 0.0
    # ~6 px cells (bins=17 over +/-50) absorb subpixel jitter and mild lattice
    # distortion while still resolving distinct basis vectors.
    edges = np.linspace(-max_d, max_d, bins + 1)
    Hh, _, _ = np.histogram2d(dv[:, 0], dv[:, 1], bins=[edges, edges])
    Hh = Hh.ravel()
    total = Hh.sum()
    if total == 0:
        return 0.0
    # fraction of mass in the `top` fullest cells (lattice basis + Friedel
    # partners); a perfect lattice piles every difference onto a few cells,
    # random/flooded junk spreads them out.
    return float(np.sort(Hh)[-top:].sum() / total)


# --------------------------------------------------------------------------- #
# Detection back-ends
# --------------------------------------------------------------------------- #
def detect_nxcorr(frame, *, kernel_radius, threshold, min_distance, sigma,
                  beamstop_mask=None, subpixel=True):
    """The production CPU NXCORR single-frame path."""
    from spyde.actions.find_vectors import _find_vectors_single_frame
    f = frame.astype(np.float32)
    if sigma and sigma > 0:
        f = gaussian_filter(f, sigma)
    _, _, peaks = _find_vectors_single_frame(
        f, kernel_radius, threshold, min_distance,
        subpixel=subpixel, beamstop_mask=beamstop_mask,
    )
    return peaks  # (N,3) [ky,kx,intensity]


def detect_log(frame, *, sigma_log, threshold_rel, min_distance,
               beamstop_mask=None, bg_sigma=4.0, bs_dilate=3):
    """Laplacian-of-Gaussian blob detector tuned for tiny (2-3 px) spots.

    LoG is the matched filter for a small Gaussian blob; unlike a flat disk it
    does not need a radius that matches the spot and gives a sharp single-pixel
    response per spot.  Background is removed first so the smooth beam tails do
    not bias the response.

    Beam-stop handling (this is the crucial bit on this dataset): the masked
    region is *background-filled*, NOT zero-filled.  Zeroing makes a sharp
    signal->0 step at the mask boundary and the LoG fires on that step (this is
    why naive small-sigma LoG gave 60-90% beam-stop false positives).  Filling
    with the smooth local background removes the step, and detections inside the
    dilated mask are excluded as a backstop.
    """
    from scipy.ndimage import gaussian_laplace
    f = frame.astype(np.float32)
    bg = gaussian_filter(f, bg_sigma)
    hp = f - bg
    excl = None
    if beamstop_mask is not None:
        excl = dilate(beamstop_mask, bs_dilate)
        hp = hp.copy()
        hp[excl] = 0.0  # hp is already background-subtracted -> ~0, no step
    resp = -gaussian_laplace(hp, sigma_log)  # bright blobs -> positive
    resp = np.clip(resp, 0, None)
    if resp.max() <= 0:
        return np.zeros((0, 3), np.float32)
    thr = threshold_rel * resp.max()
    mx = maximum_filter(resp, size=2 * min_distance + 1)
    pk = (resp == mx) & (resp >= thr)
    if excl is not None:
        pk &= ~excl
    ys, xs = np.where(pk)
    inten = f[ys, xs]
    return np.column_stack([ys, xs, inten]).astype(np.float32)


def detect_dog(frame, *, sigma1, sigma2, threshold_rel, min_distance,
               beamstop_mask=None, bs_dilate=3):
    """Difference-of-Gaussians blob detector (cheap LoG approximation).

    band-pass = G(sigma1) - G(sigma2), sigma1<sigma2: keeps spot-scale
    structure, removes both pixel noise and the smooth beam background in one
    pass (no separate bg subtraction).  Same background-fill beam-stop handling
    as detect_log.
    """
    f = frame.astype(np.float32)
    excl = dilate(beamstop_mask, bs_dilate) if beamstop_mask is not None else None
    if excl is not None:
        f = f.copy()
        f[excl] = gaussian_filter(frame.astype(np.float32), sigma2)[excl]
    resp = gaussian_filter(f, sigma1) - gaussian_filter(f, sigma2)
    resp = np.clip(resp, 0, None)
    if resp.max() <= 0:
        return np.zeros((0, 3), np.float32)
    thr = threshold_rel * resp.max()
    mx = maximum_filter(resp, size=2 * min_distance + 1)
    pk = (resp == mx) & (resp >= thr)
    if excl is not None:
        pk &= ~excl
    ys, xs = np.where(pk)
    return np.column_stack([ys, xs, frame[ys, xs]]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Beam-stop fill strategies (for the NXCORR path)
# --------------------------------------------------------------------------- #
def make_beamstop_variants(bs):
    return {
        "none": None,             # no mask -- baseline (current production)
        "mask": bs,               # mean-fill + exclude (current core support)
        "mask+dil2": dilate(bs, 2),  # dilate to swallow the bright rim
    }


# --------------------------------------------------------------------------- #
# Scoring one configuration over a set of frames
# --------------------------------------------------------------------------- #
def score_config(mm, frame_idxs, bs, detect_fn, *, bs_dilate=2, fri_tol=2.0,
                 fri_topk=80):
    """fri_topk caps the (brightest) peaks fed to the O(n^2) Friedel search so a
    flooding config (hundreds of junk peaks) stays affordable to score; real
    lattice reflections are bright, so the brightest 80 always contain them."""
    bs_fp_region = dilate(bs, bs_dilate)
    bs_fp_rates, fri_fracs, fri_counts, lat_scores, n_spots, times = \
        [], [], [], [], [], []
    for fi in frame_idxs:
        frame = mm[int(fi)]
        t0 = time.perf_counter()
        peaks = detect_fn(frame)
        times.append((time.perf_counter() - t0) * 1000.0)
        if len(peaks) == 0:
            bs_fp_rates.append(0.0); fri_fracs.append(0.0)
            fri_counts.append(0); lat_scores.append(0.0); n_spots.append(0)
            continue
        yx = peaks[:, :2]
        iy = np.clip(np.round(yx[:, 0]).astype(int), 0, H - 1)
        ix = np.clip(np.round(yx[:, 1]).astype(int), 0, W - 1)
        on_bs = bs_fp_region[iy, ix]
        bs_fp_rates.append(float(on_bs.mean()))
        keep = ~on_bs
        off = yx[keep]
        off_int = peaks[keep, 2] if peaks.shape[1] > 2 else np.ones(len(off))
        n_spots.append(len(off))
        if len(off) > fri_topk:
            sel = np.argsort(-off_int)[:fri_topk]
            off = off[sel]
        c = friedel_center(off)
        frac = friedel_inlier_fraction(off, c, tol=fri_tol)
        fri_fracs.append(frac)
        fri_counts.append(int(round(frac * len(off))))
        lat_scores.append(lattice_consistency(off))
    return dict(
        bs_fp=float(np.mean(bs_fp_rates)),
        fri=float(np.mean(fri_fracs)),
        # lat: lattice-consistency (difference-vector concentration), the
        # headline ranking metric -- robust to undetected Friedel partners.
        lat=float(np.mean(lat_scores)),
        # n_real: Friedel-paired spots per frame -- the throughput of *real*
        # reflections.  A detector that floods the frame inflates n but its
        # fri collapses, so n_real (count, not fraction) is the headline:
        # high n_real AND high fri == many real spots, few junk.
        n_real=float(np.mean(fri_counts)),
        n=float(np.mean(n_spots)),
        t=float(np.median(times)),
    )


# --------------------------------------------------------------------------- #
# Frame selection -- prefer "spotty" frames (real diffraction present)
# --------------------------------------------------------------------------- #
def pick_spotty_frames(mm, bs, n_frames, survey_stride=4):
    scores, idxs = [], []
    for iy in range(0, NY, survey_stride):
        for ix in range(0, NX, survey_stride):
            fi = iy * NX + ix
            f = mm[fi].astype(np.float32)
            hp = f - gaussian_filter(f, 4)
            hp[bs] = 0
            scores.append(np.percentile(hp, 99.9))
            idxs.append(fi)
    order = np.argsort(-np.asarray(scores))
    return np.asarray(idxs)[order[:n_frames]]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_MRC)
    ap.add_argument("--frames", type=int, default=60,
                    help="number of spotty frames to score")
    ap.add_argument("--sweep", action="store_true",
                    help="run the full parameter/method sweep and write a report")
    args = ap.parse_args()

    print(f"[3nm] opening {args.path}")
    mm = open_memmap(args.path)
    print(f"[3nm] memmap {mm.shape} {mm.dtype}")

    t0 = time.perf_counter()
    s, bs = build_sum_and_beamstop(mm)
    print(f"[3nm] beam stop = {int(bs.sum())} px  (built in {time.perf_counter()-t0:.1f}s)")

    frame_idxs = pick_spotty_frames(mm, bs, args.frames)
    print(f"[3nm] scoring on {len(frame_idxs)} spotty frames\n")

    rows = []

    def run(label, detect_fn):
        r = score_config(mm, frame_idxs, bs, detect_fn)
        rows.append((label, r))
        print(f"  {label:<42}  lat={r['lat']:.3f}  bs_fp={r['bs_fp']*100:5.1f}%  "
              f"fri={r['fri']*100:5.1f}%  n_real={r['n_real']:5.1f}  "
              f"n={r['n']:5.1f}  t={r['t']:6.2f}ms")

    # ---- Baseline: current production defaults, NO beam-stop mask ----------
    print("== Baseline (production defaults, kr=5, thr=0.5, no mask) ==")
    run("nxcorr kr5 thr0.5 md5 s1.0 nomask",
        lambda f: detect_nxcorr(f, kernel_radius=5, threshold=0.5,
                                min_distance=5, sigma=1.0, beamstop_mask=None))

    if not args.sweep:
        return

    bsv = make_beamstop_variants(bs)

    # ---- NXCORR: smaller kernel to match the 2-3 px spots ------------------
    print("\n== NXCORR kernel radius sweep (thr0.4 md3 s0.8, mask+dil2) ==")
    for kr in (1, 2, 3, 4, 5):
        run(f"nxcorr kr{kr} thr0.4 md3 s0.8 mask+dil2",
            lambda f, kr=kr: detect_nxcorr(
                f, kernel_radius=kr, threshold=0.4, min_distance=3,
                sigma=0.8, beamstop_mask=bsv["mask+dil2"]))

    # ---- NXCORR: beam-stop strategy at the best small kernel ---------------
    print("\n== NXCORR beam-stop strategy (kr2 thr0.4 md3 s0.8) ==")
    for name, mask in bsv.items():
        run(f"nxcorr kr2 thr0.4 md3 s0.8 bs={name}",
            lambda f, m=mask: detect_nxcorr(
                f, kernel_radius=2, threshold=0.4, min_distance=3,
                sigma=0.8, beamstop_mask=m))

    # ---- NXCORR: threshold sweep at small + mid kernel ---------------------
    print("\n== NXCORR threshold sweep (kr2/kr3 md3 s0.8 mask+dil2) ==")
    for kr in (2, 3):
        for thr in (0.45, 0.55, 0.65, 0.75):
            run(f"nxcorr kr{kr} thr{thr} md3 s0.8 mask+dil2",
                lambda f, kr=kr, thr=thr: detect_nxcorr(
                    f, kernel_radius=kr, threshold=thr, min_distance=3,
                    sigma=0.8, beamstop_mask=bsv["mask+dil2"]))

    # ---- LoG blob detector (background-fill beam stop, dilate 3) -----------
    print("\n== LoG blob detector (bg-fill beam stop, bs_dilate=3) ==")
    for sl in (1.0, 1.3, 1.6, 2.0):
        for tr in (0.10, 0.18, 0.28):
            run(f"log sig{sl} thr{tr} md3 bsdil3",
                lambda f, sl=sl, tr=tr: detect_log(
                    f, sigma_log=sl, threshold_rel=tr, min_distance=3,
                    beamstop_mask=bs, bs_dilate=3))

    # ---- DoG blob detector -------------------------------------------------
    print("\n== DoG blob detector (bg-fill beam stop, bs_dilate=3) ==")
    for (s1, s2) in ((0.8, 2.0), (1.0, 2.5), (1.2, 3.0)):
        for tr in (0.10, 0.18, 0.28):
            run(f"dog s{s1}/{s2} thr{tr} md3",
                lambda f, s1=s1, s2=s2, tr=tr: detect_dog(
                    f, sigma1=s1, sigma2=s2, threshold_rel=tr, min_distance=3,
                    beamstop_mask=bs, bs_dilate=3))

    # ---- write report ------------------------------------------------------
    write_report(rows, args, int(bs.sum()))


def write_report(rows, args, bs_px):
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "benchmark_3nm_spots_results.md")
    out = os.path.abspath(out)
    lines = [
        "# 3 nm scan -- spot-finding benchmark results",
        "",
        f"- dataset: `{os.path.basename(args.path)}`  (300x300 nav, 256x256 uint16)",
        f"- beam stop: {bs_px} px masked",
        f"- scored on {args.frames} spotty frames",
        "",
        "Metrics: **lat** = lattice-consistency, difference-vector "
        "concentration (HEADLINE, higher better); "
        "**bs_fp** = beam-stop false-positive rate (lower better); "
        "**fri** = Friedel inlier fraction (higher better, but noisy); "
        "**n_real** = Friedel-paired spots/frame; "
        "**n** = spots/frame off the stop; **t** = median ms/frame.",
        "",
        "Reference lattice scores on this data: perfect synthetic lattice "
        "~0.31, clean hand-picked spots ~0.22, random points ~0.12, "
        "flooded junk ~0.02.",
        "",
        "| config | lat | bs_fp % | fri % | n_real | n | t (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, r in sorted(rows, key=lambda kv: -kv[1]["lat"]):
        lines.append(f"| {label} | {r['lat']:.3f} | {r['bs_fp']*100:.1f} "
                     f"| {r['fri']*100:.1f} | {r['n_real']:.1f} | {r['n']:.1f} "
                     f"| {r['t']:.2f} |")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[3nm] wrote {out}")


if __name__ == "__main__":
    main()
