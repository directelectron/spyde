# Particle segmentation, round 2 — strokes as structures, and an interactive loop

**Status: proposal. Nothing implemented.** Scoped 2026-08-05 against
`feat/drift-particles` @ `d4b448b` (PR #117, open, 44 commits ahead of `main`).

This picks up the user's critique of the shipped scribble engine:

> I think right now we are using the individual points from the scribble when we
> should instead use the scribble as connected components. Then from the scribble
> → a noise profile of the structures as well as contrast and variations. That
> defines a particle […] The actual convolution should be very small, we want
> something extremely small.

The critique is correct about the code, and §1 below reframes where the time
actually goes — because it is not where either of us assumed.

---

## 0. The diagnosis is right: stroke identity is destroyed at storage time

`LabelStore` stores a frame's labels as two parallel flat arrays and nothing else
(`spyde/particles/scribble.py:225`):

```python
#: frame index -> (flat indices int64, class ids int16)
_frames: dict[int, tuple[np.ndarray, np.ndarray]]
```

`paint_stroke` (`scribble.py:322`) densifies the polyline to half-pixel steps,
dabs a disc at each, `np.unique`s the result, and merges it into that frame's one
global pool via `_dedup_last_wins`. **After the call returns, there is no record
that those pixels came from one stroke.** Then `ScribbleClassifier.fit`
(`scribble.py:697`) samples the 34-channel feature vector at each pixel
*independently* and trains an MLP on `(pixel → class)` i.i.d.

Two concrete consequences, both of which match what the user is seeing:

- **A long stroke outvotes a short one.** The class weighting is
  `counts = torch.bincount(y); weight = N / (n_classes · counts)` — balancing is
  **per class**, so within a class, pixel count is influence. Painting more of
  one particle silently reweights the model toward that particle.
- **Nothing can be learned that is a property of a *region*.** A noise floor, an
  intra-structure variance, a contrast *relative to its own local background* —
  none of these are functions of a single pixel, so none of them are expressible
  in the current training set no matter how many channels are added.

---

## 1. The reframe: the convolution is not the problem — the refit is

Measured on this box (TITAN X Pascal, CUDA, torch 2.6.0+cu124), one 1024² frame,
~1800 labelled pixels across two classes, one full cycle
*(retrain → predict → split)*:

| spec | featurise | **fit** | predict | split | **cycle** |
|---|---:|---:|---:|---:|---:|
| default `(0.5,1,2,4,8)`, 34 ch | 68 ms | **1938 ms** | 98 ms | 358 ms | **2487 ms** |
| `(0.5,1,2)`, 22 ch | 38 ms | 613 ms | 56 ms | 289 ms | 997 ms |
| `(0.5,1)` no rank/hessian, 8 ch | 31 ms | 661 ms | 43 ms | 287 ms | 1023 ms |

**The convolution is 2.7% of the cycle. The 300-epoch full-batch Adam refit is
78%.** Shrinking the kernels is worth having (below) but on its own it cannot
make this interactive — it is optimising the third-smallest term.

### 1a. Small kernels are still worth it — for a different reason

Feature-stack cost alone, same box:

| spec | ch | halo | CPU 512² | CUDA 1024² |
|---|---:|---:|---:|---:|
| default `(0.5,1,2,4,8)` | 34 | **33** | 41.3 ms | 36.1 ms |
| `(0.5,1,2)` | 22 | 9 | 25.6 ms | 50.1 ms ⚠ |
| `(0.5,1)` | 16 | 5 | 24.9 ms | 14.3 ms |
| `(0.5,1)` no rank | 12 | 5 | 12.6 ms | 12.9 ms |
| `(0.5,1)` no rank/hessian | 8 | 5 | 15.9 ms | 10.9 ms |

⚠ the 22-channel CUDA cell is out of order versus its neighbours — one sample,
allocator noise. Don't build on the exact ratios; the CPU column and the halo
column are the structural results.

The **halo** column is the one that matters and it is not visible in a
single-frame timing. `sigma=8` at `truncate=4` gives a 33-row halo, and every row
band re-featurises its halo above and below: `features.py:113` records **36% of
the work being halo** at 4096² with 184-row bands. At halo 5 that overhead
effectively disappears. So the win from small kernels at full frame size is
larger than the 3.3× measured here — but it is a win on a term worth 68 ms.

The real payoff of going small is **locality** — see §1b, which supersedes the
"cache the stack" idea this section originally led to.

For reference, full-resolution stack sizes:

| | 1024² | 2048² | 4096² |
|---|---:|---:|---:|
| 34 ch | 136 MiB | 544 MiB | **2176 MiB** |
| 8 ch | 32 MiB | 128 MiB | **512 MiB** |

### 1b. The stack never needs to exist at 4096² — a small receptive field makes it tileable

A small receptive field does not just make the kernels cheaper, it makes the
computation **local**: a tile's features depend only on that tile plus a halo. So
you can compute exactly the tile you need and throw it away. The halo is what
decides whether that is affordable.

Wasted work per tile (halo re-featurised on every side):

| tile | halo=33 (σ up to 8) | halo=5 (σ up to 1) |
|---:|---:|---:|
| 64² | 4.13× | 1.34× |
| 128² | 2.30× | 1.16× |
| 256² | 1.58× | **1.08×** |
| 512² | 1.27× | 1.04× |

At 256² tiles the big stack recomputes **58% of its work as halo**; the small one
recomputes 8%. **A large receptive field cannot be tiled finely at all** — which
is exactly why the current design had to band the whole frame and then wonder
about caching it.

Measured, one tile, CUDA:

| tile | 34 ch, halo 33 | 8 ch, halo 5 | speedup |
|---:|---:|---:|---:|
| 128² | 14.31 ms | 4.77 ms | 3.00× |
| 256² | 15.73 ms | 4.84 ms | 3.25× |
| 512² | 19.03 ms | 5.94 ms | 3.21× |

Note the big stack barely gets cheaper going from 512² to 128² (19.03 → 14.31 ms)
— at small tiles it is nearly *all* halo. The small stack scales the way you would
want.

So the answer to "do we need the channels at 4096²?" is **no, and we do not need
to cache them either**:

| what you actually need | cost (8 ch, halo 5) | touched |
|---|---:|---|
| **train**: featurise 6 stroke boxes (120² each) | **25.6 ms** | 0.60% of a 4096² frame |
| **preview**: featurise the visible 1 MP tile + head | **18.7 ms** | 32.6 MiB, freed after |
| *(commit/batch: the whole 4096² frame, banded)* | *104.7 ms* | *— once, not per stroke* |

(the same three at 34 ch / halo 33: 82.3 ms, 49.5 ms, 417.3 ms.)

**This deletes a whole subsystem from the proposal.** No resident feature cache,
no budget, no invalidation on frame change, no eviction policy, no contention with
the display's own caches on a 12 GB card. Recomputing the tile you need is cheaper
than the bookkeeping to avoid recomputing it. It also lines up with the display
architecture that already exists: anyplotlib owns viewport tiling for frames
≥1024 px (`Plot._maybe_tile_signal` → `enable_tile`/`set_detail`), so the preview
can follow the viewport the renderer is already tracking.

> ### The trap that makes or breaks this
>
> **A tile featurised standalone is NOT equal to that region of the full frame,
> and the difference is not subtle.** `FeatureSpec.normalize_frame` standardises
> by robust per-array median/IQR, and `prepare_frame` computes those from
> whatever array it is handed — so a tile normalises by *its own* statistics.
>
> Measured on a deliberately inhomogeneous 1024² frame (right half brighter, i.e.
> an ordinary illumination gradient), a 128² tile from the dark half against the
> same region of the full-frame stack:
>
> ```
> max abs diff  = 5.536        <- on features whose full-frame range is [-1.399, 0.954]
> mean abs diff = 0.587
> normalize_frame=False:  max abs diff = 0   (bit-identical)
> ```
>
> The tile's features land **4× outside the range of the real ones**. Uncorrected,
> a tiled preview is a different computation from the committed run — the *exact*
> failure already on the record for the classical engine, where the preview's otsu
> threshold was 120.0 and the full frame's was 146.0, so "the preview literally did
> not control the run" (`particles_action.py:65`).
>
> **Fix:** compute the normalisation statistics **once per frame** (already
> subsampled to ≤10⁶ px, `_STAT_SAMPLE_MAX`) and pass them into every tile's
> `prepare_frame`. Cheap, but it must be plumbed as an explicit argument — the
> current signature makes the wrong thing the default. Pin it with a test that
> asserts tile == full-frame region **bit-for-bit**, on an inhomogeneous frame
> (a uniform fixture cannot fail this).

---

## 1c. Full-frame 4096² at TV rates — measured on the real growth movie

Requirement: segment the **whole** frame (not a viewport) at ~24–30 fps for
4096². Prototyped in `fastseg.py` and run end-to-end on
`InSituElectrochemGrowth` (245 × 4096² uint8, DE-Artemis counting mode,
0.45448 nm/px, 0.26208 s/frame). All numbers TITAN X (Pascal, 2015), fp32.

| config | ch | rf | full frame | fps | train acc |
|---|---:|---:|---:|---:|---:|
| σ=(1), no background ref | 5 | 13 px | **37.6 ms** | **26.6** | — |
| σ=(2), no background ref | 5 | 25 px | 47.4 ms | 21.1 | 0.66 |
| σ=(2) + bg ref | 8 | 321 px | 66.3 ms | 15.1 | 0.66 |
| σ=(1,2,4) + bg ref + logit smoothing | 12 | 339 px | 165 ms | 6.1 | 0.71 |

**So TV rate at 4096² is reachable — 37.6 ms / 26.6 fps — but only with a lean
feature set, and this dataset needs more than the lean set to work at all.**
That is the real trade-off, not a tuning detail.

Three structural findings got it from ~240 ms to 37.6 ms:

1. **A wide head is unaffordable at full frame.** `hidden=64` materialises a
   (1,64,4096,4096) fp32 activation — **4.3 GB**, ~22 ms of DRAM traffic before
   any arithmetic. Measured head cost alone: 113 ms (`Linear` on a reshaped
   `(H·W, C)` matrix) / 74 ms (`Conv2d` 1×1). Going to `hidden=0` (a plain linear
   head, i.e. 1×1 conv) is 2.8× on the whole pipeline. The epoch sweep in §3
   already showed the scribbles are near-separable, so width was not buying
   separability.
2. **Produce K filter planes in two multi-output conv passes**, not K
   one-channel separable blurs — cuDNN is poor at 1-channel work (one separable
   blur at 4096² measured 9.3 ms against a ~1 ms bandwidth floor).
3. **Ship uint8, convert on device.** float32 H2D of a 4096² frame is 11.6 ms;
   pinned uint8 + `.float()` on device is 1.9 ms, and the mask back is 1.3 ms.

Two levers that did **not** work on this card, so don't plan around them:
`torch.compile`/inductor **refuses to run on Pascal** ("too old"), and
hand-rolled shift-and-accumulate convolution is *slower* than cuDNN (0.42–0.58×).
Both would likely pay off on Ampere or newer, where fp16/TF32 also becomes
available — this GPU has neither, so treat 37.6 ms as a floor for a 2015 card,
not a ceiling for the approach.

### The finding that overturns "extremely small" — for this data

The small particles are **6.8 nm median equivalent diameter (~15 px)** and their
contrast to *local* surroundings is **1.07 counts at per-pixel CNR 0.169**. At
that CNR a single pixel carries almost no information; the whole 15-px particle
carries at best CNR ≈ 2.3. So:

- A receptive field matched to the particle (σ≈2–4) is the *matched filter* and
  is what the data actually wants. σ=1 alone is too small here.
- **A local background reference is mandatory**, and it must be large. The field
  has large-scale thickness variation, so "darker than the frame" ≠ "darker than
  its surroundings". Both of my first two stroke-siting attempts compared
  intensities **globally** and both produced garbage — one put particle strokes
  on the electrolyte and returned an **inverted overlay**.

These two pull in opposite directions, and the resolution is the useful part:
**small kernels at full resolution, plus a coarse background reference computed
on a decimated image** (`FeatureBank.bg_sigma`, decimate 4× → blur → resample).
A σ=25 reference costs ~1/16 of what it would at full resolution, so the
receptive field is 321 px while the *cost* stays near the small-kernel budget.
"Extremely small" is right for the discriminative kernels and wrong for the
reference channel; a pyramid gets both.

Also worth keeping: **smooth the logits, not the image.** At CNR 0.17 an
independent per-pixel decision shatters a real particle into specks and inflates
the count (7095 → 4842 instances at the same threshold once the decision variable
is blurred with σ=3). It is one separable pass on 2 channels and it is a matched
filter on the quantity actually being thresholded.

### What it measured

Trained on **31 strokes on ONE frame = 0.277% of the pixels**, then applied to all
245 frames:

| | t = 0 | t = 244 (63.9 s) |
|---|---:|---:|
| particles | 4842 | 3670 |
| mean particle area | 94 nm² | 133 nm² |
| total deposited area | 1.03 µm² | 1.32 µm² |

Count down, mean size up, total up — coalescence during growth, which is the
expected signature. Throughput over the whole movie: **184 ms/frame segmentation,
79 ms zarr decode, 219 ms instance labelling on CPU** — i.e. `scipy.ndimage.label`
becomes the bottleneck and is the obvious next thing to move to the GPU.

> **Process note.** Three of the first four runs produced confident, wrong
> results, and only the guard caught them: an inverted overlay, and a "growth
> curve" that was the classifier flipping at t≈200. The check that worked was
> **verifying every stroke against its own surroundings before training** and
> refusing to run if they don't separate. Train accuracy is the tell — 0.55 is
> chance, and any pipeline that reports a number without it will happily publish
> a growth curve made of noise. Whatever ships should surface the CNR and the
> train accuracy in the caret (§2d), not bury them.

---

## 2. The core change: strokes are regions, and regions have statistics

**One enabling change, then three things it unlocks.**

### 2a. Enabling: keep the stroke id

`LabelStore._frames[t]` becomes `(flat_idx int64, class_id int16, stroke_id int32)`,
with a monotonic per-store stroke counter. `_dedup_last_wins` carries the third
column. `to_dict`/`from_dict` take a version bump; a v1 file loads with every
pixel assigned `stroke_id = -1` (meaning "unknown provenance"), and every consumer
below degrades to current behaviour on `-1`. Erase drops pixels as it does now.

This is small and mechanical, and **nothing else in §2 is possible without it.**

### 2b. Per-stroke statistics — `StrokeStats`

For each stroke, computed once at paint time (a few thousand pixels — microseconds):

- **level**: median and IQR of raw intensity along the stroke
- **noise**: std of `raw − gaussian(raw, σ≈1)` within the stroke — the
  within-structure fluctuation, i.e. the noise floor *of that structure*, which
  is exactly the "noise profile" asked for and is not estimable from a point
- **texture**: std/IQR of the small-σ responses within the stroke
- **scale**: stroke length, pixel area, bbox, and area/length ≈ the width of the
  thing painted across

### 2c. Three uses, in descending order of confidence

**(i) Per-stroke sample weighting — highest confidence, ~5 lines.**
Weight each pixel `1 / (n_strokes_in_class · n_px_in_its_stroke)` instead of
`1 / n_px_in_class`. Every stroke then counts once, so painting more of the same
particle stops biasing the model. This is a strict improvement over per-class
balancing and is worth doing on its own merits.

**(ii) Contrast-relative feature normalisation — this is the "changes in
intensity" fix.**
`FeatureSpec.normalize_frame` currently standardises by whole-frame robust
median/IQR (`features.py`). Replace that with a standardisation derived from the
*class strokes*: centre on the background strokes' level, scale by the
foreground-vs-background level difference. The head then sees **contrast relative
to what the user called background**, not absolute intensity. If the film
brightens across the field or drifts over the movie, absolute intensity moves and
the contrast does not — which is the failure the user anticipated.

**(iii) Data-driven scale selection — the principled version of "extremely
small".**
The stroke width tells you the structure scale. Choose `sigmas` from it
(roughly `σ ∈ {0.5, 1, 2} · width/4`, capped) instead of hard-coding
`(0.5, 1, 2, 4, 8)`. The kernels end up small *because the structures are small*,
and the app can know that from the scribble rather than being told.

> **Guard.** `features.py:88` records a measured sensitivity: coarsening the
> sigma set inflates radius error (MAE 13.4% at `(0.5,1,2,4,8)` → 25.5% at
> `(4,8)`; the r=3 probe goes −12% → −44%). That measurement is about raising the
> **floor**. Dropping the **ceiling** (8, 4) is a different change and is
> *unmeasured*. It must be re-run before the default moves — a found particle
> measured 44% too small enters the size distribution and is worse than an honest
> miss.

### 2d. The noise profile as a *verdict*, not just a feature

From (b) you get, for free, `Δμ / σ_noise` between any two classes. That is a
contrast-to-noise ratio for the actual segmentation the user is attempting, and
it is available **the moment the second stroke is painted** — before any training.

Show it. `CNR ≈ 1.3 — marginal` next to the class list is worth more than any
slider, because it is the number that says *no classifier will fix this, go get
better data*. This is the "otsu on low-contrast in-situ is unrescuable by any
knob" lesson made quantitative and moved to the front of the interaction instead
of being discovered after a batch run.

---

## 3. Making it interactive

Target: **stroke → updated mask in under ~60 ms**, so the user can see whether an
addition was enough. Three changes, none of them speculative:

**(i) Featurise only what you need, and don't cache it.** Per §1b: training needs
the stroke neighbourhoods (**25.6 ms**, 0.6% of a 4096² frame) and the preview
needs the visible tile (**18.7 ms**, 32.6 MiB freed immediately). Today `fit`
re-featurises *every labelled frame in full* on every retrain. The fix is not a
cache — it is to stop computing 99.4% of a frame nobody asked for.

> An earlier draft of this plan proposed a resident per-frame feature cache
> (512 MiB at 8 ch / 4096²). **Superseded.** With halo 5 the recompute is cheaper
> than the bookkeeping, and it avoids competing for VRAM with the display's own
> caches. Keep the resampling number in mind only if profiling later says
> otherwise: 0.061 ms to resample 900 px from an already-resident stack.

**(ii) Warm-start the fit instead of retraining from scratch.**

| epochs | fit | train acc | IoU vs 300-epoch |
|---:|---:|---:|---:|
| 300 | 697 ms | 1.0000 | 1.0000 |
| 100 | 211 ms | 1.0000 | 0.8884 |
| 50 | 119 ms | 1.0000 | 0.8743 |
| 20 | **43 ms** | 1.0000 | 0.8494 |
| 10 | 30 ms | 1.0000 | 0.8174 |

Training accuracy is **1.0 at every epoch count** — the scribbles are linearly
separable almost immediately. The other 280 epochs only sharpen the boundary in
*unlabelled* space. So the answer is not "fewer epochs" (IoU 0.85 is a visibly
different mask) but **cumulative** epochs: keep the previous weights, run ~20 more
per added stroke. The user converges on the 300-epoch answer over the course of
the interaction instead of paying for it on every keystroke. A "Refine" button
(or committing) runs the full fit for the authoritative result.

**(iii) The live preview shows the MASK, not the instance split.** Splitting is
287–358 ms at 1024² — once (i) and (ii) land, it becomes the dominant term. But
"is my addition enough?" is a question about the *classification*, not about
instance identity. Show foreground probability live; split on demand and on
commit. (If splitting live is wanted anyway, the boundary route is 5.4× faster
than watershed and more accurate — `benchmarks.md:928`.)

### Projected budget — one stroke added, 4096² frame, 1 MP visible

| step | cost |
|---|---:|
| featurise the new stroke's box (120²+halo) | ~4 ms |
| warm-start fit, ~20 epochs | ~43 ms |
| featurise the visible tile + head predict | ~19 ms |
| **stroke → visible mask** | **~66 ms** |
| *(split, on demand)* | *~290 ms @1024²* |

Against **2487 ms** today at a *quarter* of the frame size. Note the frame size
barely enters: the cost is set by the stroke box and the viewport, not by the
data. That is the whole point of §1b — a 16 MP frame and a 1 MP frame cost the
same to scribble on.

---

## 4. The UI: `[+ Add segment]`, named and renameable

The backend is most of the way there and the renderer is not there at all.

**Already exists:** `LabelStore.add_class(name, colour, particle=, boundary=)`
and `remove_class(cid)` (`scribble.py:241`, `:255` — the latter correctly takes
its pixels with it). `ScribbleClass.name` is a plain field, so renaming is a field
write.

**Missing:** action verbs `seg_add_class` / `seg_remove_class` / `seg_rename_class`
/ `seg_set_class_role`, and the renderer UI. `SegmentWizard.tsx:724` has an
explicit comment that there is no add-class button because the backend has no
verb. This is a small, well-shaped piece of work.

### The one trap: defaulting to two classes has a hidden cost

The user asked to start with `[Foreground] [Background]`. Today's default is four
(`scribble.py:123`): `particle`, `support film`, `vacuum`, **`boundary`**.

**The `boundary` class is load-bearing for performance and accuracy.** When a
boundary class carries trained weight, `split_instances` takes a
connected-components route instead of watershed — measured at 4096²
(`benchmarks.md:928`): **1.29 s vs 2.73 s end-to-end (5.4× on the split itself),
and *more* accurate** — 162 bodies against 162 ground truth, versus 173 for the
watershed.

So shipping a two-class default silently routes every user onto the slower, less
accurate path. Recommendation:

- Default to **`[Particle]` `[Background]`** — the user's ask, and the right
  starting point.
- Make the **role an explicit per-class dropdown** (`particle` / `background` /
  `boundary`) rather than today's hidden pair of booleans. Renaming a class then
  cannot accidentally change what it *means*.
- Offer the boundary class rather than pre-creating it: when the preview shows
  merged instances, surface *"particles are touching — add a Boundary segment and
  paint the seams"* with a one-click add. That teaches the feature at the moment
  it is needed instead of presenting four classes to someone who wanted two.

---

## 5. Click-to-select vs scribble — resolved, with a reason

The user proposed clicking points, then reconsidered ("I guess actually the
scribble might be better as you might have changes in intensity"). **The
reconsideration is right, and §2 is why:** a noise profile, an intra-structure
variance and a width estimate are all properties of an *extent*. None of them can
be computed from a single pixel. The statistical design the user is asking for
*requires* strokes as its input.

So: **keep the stroke as the primitive, and treat a click as a one-dab stroke.**
One code path, and a click still works — it just contributes a weaker
`StrokeStats` (level only, no noise or scale), which is honest, because it
genuinely carries less information.

Note this is **not** plan B4 (`DRIFT_AND_PARTICLES_PLAN.md:624`), which proposed
click-to-segment via EfficientSAM. That remains a reasonable separate bootstrap,
but it needs a 40 MB model download and the plan's own §0.4 table notes a
natural-photography prior is out of distribution on EM contrast — it
"over-segments film texture, merges touching particles". The proposal here needs
no model and adapts to the data by construction.

---

## 6. What could invalidate this

- **(2c-ii) could hurt on data where absolute intensity is the signal.** Gate it
  on the existing `TestRandomForestParity` fixture plus a real in-situ movie
  before it becomes the default; keep whole-frame normalisation selectable.
- **Dropping the sigma ceiling is unmeasured** (§2c guard). The docstring's own
  argument for keeping `σ=8` is that it "gives the head a local-background
  reference" — which is precisely the job (2c-ii) proposes to give to the stroke
  statistics instead. If that substitution does not hold, the ceiling has to stay
  and the memory argument in §3(i) weakens with it. **Measure this first — it is
  the load-bearing assumption of the whole proposal.**
- **Warm-starting introduces history dependence.** The model becomes a function of
  the order strokes were painted, which `TestDeterminism` currently forbids. The
  full-fit-on-commit path must stay the authoritative, order-independent one, and
  the determinism test should pin *that*, not the live preview.
- All numbers here are one box, one synthetic frame, `n=2` timing repeats. They
  are the right order of magnitude and the ratios are large enough to act on, but
  re-measure on a real in-situ movie before tuning to them.

---

## 7. Suggested order

Each stage is independently useful and independently reviewable.

1. **Stroke ids + `StrokeStats`** (§2a, §2b) — no behaviour change; pure enabling.
2. **Per-stroke weighting** (§2c-i) — smallest change with a real quality payoff.
3. **CNR readout** (§2d) — no algorithm change, high user value, needs 1 + 2.
4. **Interactive loop** (§3) — local featurising (§1b), warm start, mask-only
   preview. The biggest win (2487 → ~66 ms, and now independent of frame size)
   and independent of §2c. **Land the frame-level normalisation statistics and
   its bit-exactness test first** — everything else here rests on tile == frame.
5. **Class management UI** (§4) — add / rename / role dropdown, two-class default.
6. **Contrast-relative normalisation** (§2c-ii) — behind a flag, A/B'd.
7. **Data-driven scale selection** (§2c-iii) — only after the §6 measurement.

Stages 4 and 5 are what the user will *feel*; 1–3 are what make the engine
defensible. 6 and 7 are the parts that need evidence before they ship.
