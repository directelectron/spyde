# Motion correction — what the old app did, and what was ported

The old PySide6 Ground Crew called this **S.T.A.C.K.** — *Single-stack Tool for
Alignment, CTF and k-space*. It is not in the app you can run today: commit
`36377f7` ("Rename project to de_ground_crew, remove S.T.A.C.K./CTF from main")
**deleted** it, and `ui/main_window.py` still carries the commented-out stubs
(`# self._stack_panel = StackPanel()  # Phase 2/3 — excluded from this build`).

Recovered from `36377f7^`:

| File | Lines | What it is |
|---|---|---|
| `ui/stack_panel.py` | 1263 | the panel |
| `workers/motion_correction_worker.py` | 1095 | loading, global alignment, local motion driver, saving |
| `workers/local_motion.py` | 592 | patch-based local motion |
| `CONTEXT_motion_correction.md` | 311 | pre-implementation briefing |
| `PHASE3_CTF_PLAN.md` | 315 | CTF plan — **never implemented** |

Phases 1 and 2 were finished and working. Phase 3 (CTF) is a plan document
only: there is no defocus, astigmatism or Thon-ring code anywhere in the
recovered implementation.

---

## 1. Feature inventory

### 1.1 Loading

| Feature | Old implementation | Ported |
|---|---|---|
| Movie stack from TIFF | `LoadMovieStackWorker` via `tifffile` | ✅ |
| Movie stack from MRC | same, via `mrcfile` | ✅ |
| Stack metadata | `n_frames`, `height`, `width`, `filename` | ✅ |
| Gain reference (TIFF/MRC) | `LoadGainWorker` | ✅ |
| Gain orientation | 8-way: Identity, Rot90, Rot180, Rot270, FlipH, FlipV, Transpose, Transverse | ✅ |
| Gain auto-detect | `GainValidationWorker` — scores all 8, returns best-first | ✅ |
| Gain size matching | `_match_gain_to_frame` — super-resolution gain vs binned frame | ✅ |

### 1.2 Global (whole-frame) alignment — Phase 1

`MotionCorrectionWorker`. Follows **Unblur** (Grant & Grigorieff 2015) and
**MotionCor2** (Zheng et al. 2017).

| Feature | Detail | Ported |
|---|---|---|
| Cross-correlation | **not** phase correlation — preserves amplitude/SNR | ✅ |
| Subpixel refinement | upsampled DFT (Guizar-Sicairos 2008), `upsample_factor=20` | ✅ |
| Bandpass | `low_freq=0.005`, `high_freq=0.5` | ✅ |
| Two-pass | pass 1 against an initial reference → build refined reference → pass 2 re-align | ✅ |
| Reference choice | Central frame / First frame / Average | ✅ |
| Correlation binning | 1 / 2 / 4 / 8 | ✅ |
| Throw | discard N initial frames (MotionCor2 `-Throw`), 0–50 | ✅ |
| Temporal smoothing | cubic spline over the shift trajectory | ✅ |
| Shift application | Fourier phase ramp — exact subpixel, no interpolation blur | ✅ |
| Outputs | `aligned_sum`, `unaligned_sum`, `aligned_fft`, raw **and** smoothed X/Y shifts | ✅ |
| Cancellable | `cancel()` checked in the frame loop | ✅ |
| Progress | per-frame `"Pass 1: frame i/N dy=… dx=…"` | ✅ |

### 1.3 Local (patch-based) motion — Phase 2

`LocalMotionWorker` + `local_motion.py`. Runs **after** Phase 1, on the
globally-aligned frames.

| Feature | Detail | Ported |
|---|---|---|
| Patch grid | `generate_patch_grid`, patch size 256 / 512 / 1024 | ✅ |
| Cosine blend weights | `overlap_frac=0.5`, so patches composite without seams | ✅ |
| Per-patch correlation | `correlate_patches`, `upsample_factor=10` | ✅ |
| Patch shift smoothing | `smooth_patch_shifts` | ✅ |
| Motion field fit | polynomial surface, Vandermonde, **degree 3** | ✅ |
| Field evaluation | `evaluate_motion_field` on arbitrary points | ✅ |
| Compositing | `apply_local_shifts` — full-res, cosine-blended | ✅ |
| Outputs | `corrected_sum`, `corrected_fft`, `coefficients`, patch centres, timings | ✅ |

### 1.4 Display

| Feature | Old implementation | Ported |
|---|---|---|
| View selector | Raw Frame / Unaligned Sum / Aligned Sum | ✅ |
| Frame scrubber | slider + "Frame N/M" | ✅ |
| Display binning | 1× / 2× / 4× / 8× / 16× | ➖ anyplotlib tiles handle this |
| Contrast | mean ± Nσ, σ ∈ [0.5, 10] | ➖ SpyDE histogram handles this |
| Low-pass | Gaussian, σ ∈ [0.5, 5.0], toggle | ✅ |
| FFT panel | log power spectrum, side by side, toggleable | ✅ |
| FFT pan-lock | centred, no panning | ➖ anyplotlib |
| FFT Fourier-crop | centre-crop rather than block-average when binning | ✅ (in `log_fft`) |
| Drift plot | X/Y vs frame, raw **and** smoothed overlaid | ✅ |
| Results table | Frame │ dX (px) │ dY (px) │ \|shift\| | ✅ |
| Save result | MRC or TIFF | ✅ |
| GPU | CuPy optional, numpy fallback | ✅ (torch/CuPy-free; numpy only — see §3) |

➖ = deliberately not ported because the new app already has a better
equivalent; noted rather than dropped silently.

### 1.5 Not implemented in the old app either

- **CTF / defocus / astigmatism / Thon rings** — `PHASE3_CTF_PLAN.md` is a
  plan. No code exists. The mockup's "Motion gains … a CTF fit" is therefore a
  *new* feature, not a port, and is out of scope here.
- **Dose weighting** — discussed in the briefing, never built.
- **Per-patch quiver display** — `ImageViewer` has a `_quiver_arrows`
  renderer, but nothing in the app ever calls it. Dead code.

---

## 2. What changed in the port, and why

**Qt is gone.** Each `QThread` worker became a plain function taking
`progress(str)` and `should_cancel() -> bool` callbacks. The algorithms were
already pure numpy inside `run()`; only the transport changed. This is what
makes them testable without a GUI, which the originals were not.

**`_`-private helpers became public.** `_cross_correlate`, `_compute_log_fft`
and `_apply_orientation` were imported across module boundaries by the panel
despite the leading underscore. They are the API; they are named as such.

**Reference choice is an enum, not a UI string.** The old worker did
`params.get("reference", "central")` while the combo box held
`"Central frame"`, and the mapping lived in the panel. Now the mapping is in
one place and an unknown value raises rather than silently falling back to
central.

---

## 3. Two defects found while porting

The old implementation had **no tests** — everything ran inside a
`QThread.run()` reachable only through the GUI. Writing the first tests found
two things.

### The CPU shift did nothing

```python
phase = np.exp(np.float32(-2j * np.pi) * (fy * dy + fx * dx))
#              ^^^^^^^^^^ casts a COMPLEX constant to a float
```

NumPy 2 raises on this; NumPy 1 truncated it to the real part, `0.0`, making
the phase ramp `exp(0) == 1` — **an exact no-op**. Every CPU-path shift, in
both the refined-reference build and the final aligned sum, silently did
nothing. The GPU branch used `_cp.complex64` and was correct, so on the CUDA
machine it was developed on the bug was invisible; anyone running without an
NVIDIA card got an "aligned" sum identical to the unaligned one.

Fixed to `np.complex64`. `test_alignment_sharpens_the_sum` fails without it.

### The outermost pixel ring composites to black

In local motion, the frame border is covered by exactly one patch, and the Hann
window is zero at its edge — so the composite divides ~0 by ~0 and that ring
comes out black. One pixel wide regardless of frame size (0.01% of an 8192
frame), so it is **pinned by a test rather than changed**: the fix would be a
behaviour change, and it is not worth diverging from the original for a single
row of pixels. `test_the_outermost_pixel_ring_is_lost_to_the_hann_taper`.

While there: the claim that "Hann windows at 50% overlap sum to 1" is false for
`np.hanning`, which is the *symmetric* window — only the periodic variant
satisfies COLA. What actually flattens the composite is dividing by the
accumulated weight map. Both the source comment and the test now say so, so
nobody deletes the division on the strength of the window's reputation.

## 4. Deliberate omissions

**CuPy.** The old app used CuPy for FFTs with a numpy fallback. This app has no
GPU dependency at all and the boundary test enforces that
(`packages/de-shell/tests/test_boundary.py`). The numpy path is the one the old
app used on every machine without an NVIDIA card, including the Macs deapi is
tested on. If GPU is wanted, the seam is `align.py`'s FFT calls — one module,
not scattered.

**Cropping the valid region.** A Fourier phase ramp is a CIRCULAR shift:
content pushed off one edge wraps round to the other, so the outer few pixels
of the aligned sum are a blend of opposite edges. MotionCor2 crops to the
region all frames cover; the old app did not, and neither does this. The band
is as wide as the largest shift — a few pixels for normal drift — but it is
real, and it is at the edge where an FFT taper is already fighting for
cleanliness.

**Memory-mapped stacks.** The briefing suggested `mrcfile.mmap()` for stacks
that exceed RAM; the implementation never did it, and neither does this.
A 50 × 8192² float32 stack is 13 GB, so this is a real limit worth knowing.
