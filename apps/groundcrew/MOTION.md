# Motion correction — ported from Ground Crew

**The compute is VENDORED, not reimplemented.** `de_groundcrew/external/gc_motion/`
holds Ground Crew's motion modules copied verbatim from
`CSSFrancis/de_ground_crew@e9e21de`; `de_groundcrew/motion/driver.py` is the
thin Qt-free layer that calls them. See the vendor package's docstring for the
reasoning and MANIFEST.md for provenance.

## How this went wrong first, and why the arrangement is what it is

The first attempt hand-ported from `36377f7^` — a snapshot that had been
**deleted** from `main`. Two mistakes compounded:

1. **I read history instead of fetching.** The local clone sat on branch
   `clean-up`, well behind `origin/main`. Everything concluded from it was a
   year-stale view.
2. **I transcribed numerics.** Even had the snapshot been current, re-typing
   externally-validated code is how you inherit bugs silently.

What that cost, concretely: the port reproduced a CPU-shift no-op that upstream
had **already fixed**, missed the entire **v3** aligner (a coarse-to-fine
pyramid co-designed against cryoSPARC and RELION 5.0.1 that replaced the 2-pass
aligner I copied), missed **dose weighting**, missed **runaway-motion
detection**, missed the power spectrum's **crop-to-square**, and substituted a
naive gain-orientation metric for one calibrated on 219 real gain pairings.

It also asserted CTF "was never implemented". There is a **15,827-line faithful
CTFFIND5 port** on `main`, held to the reference binary to ±0.0005 µm.

Upstream's own QThread-removal spec rejects a parallel package — *"duplicates
code, invites drift"*. Vendoring is what that leaves.

## 1. Feature inventory (against `origin/main`, not the deleted snapshot)

| Feature | Upstream | Here |
|---|---|---|
| **Global alignment** | `motion_correct_v3` — coarse-to-fine binning pyramid, accumulating residual update, adaptive schedule in Å, DoG+Hann prep, B-spline smooth solve via L-BFGS, MAD outlier-step rejection, coarse seed with prominence z-score | ✅ vendored |
| Fast / fine mode | `_MODE_PRESETS` — refine to 6 Å or 3 Å | ✅ |
| Throw | discard N leading frames | ✅ driver |
| Runaway-motion detection | `_assess_confidence` → `low_confidence` + `failure_reason` | ✅ surfaced as an error + a banner |
| Dose weighting | `dose_weighting.py`, additive `dw_sum` | ✅ vendored, plumbed (no UI yet) |
| **Local motion** | patch grid, per-patch CC, MAD + spline smoothing, degree-3 polynomial field, cosine-blend compositing | ✅ vendored |
| Gain: 8 orientations | `_image_ops.ORIENTATION_LABELS` | ✅ |
| Gain: auto-detect | `rank_gain_orientations` — `row_std + col_std` on a binned frame | ✅ |
| Gain: fit tiers | `classify_gain_tier` — thresholds from 219 real pairings | ✅ ok / weak / **fail** all surfaced |
| Gain: size matching | super-res gain binned to the frame | ✅ |
| Power spectrum | SerialEM-style, **cropped to a centred square** so Thon rings stay circular | ✅ |
| Truncated-MRC rejection | clear error, not a `NoneType` crash | ✅ driver |
| Load / save MRC + TIFF | | ✅ |
| View: raw / unaligned / aligned / local | | ✅ |
| Frame scrubber, drift plot, per-frame table | | ✅ |
| GPU (CuPy) | optional, with CPU fallback | ➖ CPU only — see §3 |
| Quiver overlay of the local field | `ui/stack_plots.py` | ❌ not yet |
| CTF (CTFFIND5 port) | `ctffind5/`, 15,827 lines, oracle-gated | ❌ separate port |

## 2. What this app owns

Only `motion/driver.py` and the UI. The driver is glue: it swaps `QThread` +
`Signal` for `progress` / `should_cancel` callbacks — exactly the contract
upstream's own QThread-removal spec describes — and does two things that are
load-bearing.

**The sign reconciliation.** v3 returns `shifts_y` in its internal convention
but `shifts_x` negated (MotionCor3's). The rest of the pipeline wants both
internal, so x is un-negated on the way out. Get it wrong and everything runs
while the drift plot is mirrored in x. `test_the_x_sign_is_reconciled` pins the
direction.

**Fail-loud passes through.** `low_confidence` is a *result*, not an exception:
the sum is still displayed so it can be inspected, but it is announced as an
error and banner so nobody uses it by accident.

The old Bin and Reference controls are **gone** from the UI — v3 owns its own
schedule, so those knobs would do nothing.

## 3. Deliberate omissions

**CuPy.** Upstream is GPU-accelerated with a CPU fallback; this app has no GPU
dependency and the boundary test enforces it. The vendored files' `_GPU` flag
is simply False, which selects the fallback they already have — no code was
changed to achieve it. Note this matters for the **~20 s ship gate** upstream
cites for 8k×8k, 30–50 frames: that budget assumes the GPU path.

**Memory-mapped stacks.** A 50 × 8192² float32 stack is 13 GB and is loaded
whole, same as upstream.

## 4. Re-syncing

`MANIFEST.md` records the commit and a hash per file;
`test_motion.py::TestVendorIsPristine` re-checks them, normalising the import
rewrites first. Editing a vendored file is therefore a test failure. To take
upstream changes: re-copy, re-apply the import rewrites, update the manifest,
re-run. Do not merge by hand.
