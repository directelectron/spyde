# Neural (SpotUNet) Integration Plan

Status of the neural disk-detector integration and the phased plan to make it
complete. Written 2026-07-15 after an audit of `spyde/models/`,
`spyde/actions/find_vectors*`, the wizard UI, and the packaging config.

> **Status 2026-07-16: Phases 0 and 1 are IMPLEMENTED** (G1, G2, G3, G7 fixed;
> the concurrent-download half of G5 fixed via client-side `ensure_local`).
> **Phase 2 partially done**: the SPYDE_FV_GPU policy now governs the neural
> path (neural unset-default "all" = today's behaviour; flipping it needs the
> multi-worker benchmark, still open) and `load_model` uses the CUDA→MPS→CPU
> chain with a load-time MPS smoke test (needs a Mac to validate for real).
> **Phase 3 partially done**: `persistence` is wired end-to-end behind a
> default-off "Neighbor refine" wizard toggle (refine.py is live code now);
> the eval promotion gate is still open. Phase 4 not started.
> Verified: pytest (`test_model_registry.py`, `test_find_vectors_neural.py`)
> + Playwright (`fv_neural_calibration.spec.ts` — screenshots show High-pass σ
> auto-calibrating 12→4 on si-grains, live preview re-tuning, model-list
> refresh, and the vectors window opening). Full migrated suite green.

## Where we are

The core integration is in good shape:

- `spyde/models/` is a self-contained vendored copy of the yoloDiffraction
  detector (U-Net + preprocess + GPU decode + refine), so SpyDE ships without
  the research repo.
- The **model registry** (`spyde/models/registry.py`) merges bundled < user
  manifests, caches loaded models, and falls back to the bundled default on any
  failure — the wizard can never crash offline. Two bundled models ship
  (`spotunet-production-v2` default).
- **"neural" is the default find-vectors method**: wizard Model dropdown
  (populated by `fv_models`), per-chunk batched torch forward pass on GPU with a
  per-frame CPU fallback, beam-stop rejection and disk-mean intensity parity
  with NXCORR/DoG.
- Wiring tests (`test_find_vectors_neural.py`, `test_neural_detect.py`) and a
  real-scale benchmark (`benchmark_neural_spots.py`, sped_ag 13k patterns)
  exist. `spyde/models/RELEASING.md` documents the model-upgrade workflow.

## The gaps (audit findings)

Ranked by how much they break a documented or advertised behaviour.

### G1 — The remote-model upgrade path is inert (documented but cannot work)

`RELEASING.md` path A ("ship a new model WITHOUT re-releasing SpyDE") relies on
two things that don't exist:

- **`huggingface_hub` is not a dependency** — absent from `pyproject.toml` and
  `uv.lock`. `registry._resolve_hf` and `refresh_remote_registry` import it
  lazily and swallow the `ImportError`, so in a shipped app every remote
  resolve/refresh silently no-ops and the bundled default is used forever.
- **`fv_refresh_models` does not exist.** `RELEASING.md:27` says the wizard's
  Model-dropdown refresh calls it; there is no such action in
  `actions/registry.py` and no refresh control in `FindVectorsWizard.tsx`.
  `registry.refresh_remote_registry()` has zero callers in app code.

Net effect: a newly trained model uploaded to HF can never reach a user.

### G2 — Auto-calibration is dead code (the "parameter-free" promise)

`calibrate_neural` (`find_vectors_neural.py:246`) — the one-shot optimiser for
`bg_sigma` (diffuse/beam-stop backgrounds) and a lowered threshold (faint-peak
data) — **is never called**. `orchestrate.py:281` reads `params["bg_sigma"]`
but nothing sets it (the wizard has no such field), so every run uses the
default `bg_sigma=12.0`, `thresh=0.3`. The registry notes advertise
"parameter-free with auto-calibration"; only the disk-size auto-scale actually
runs.

### G3 — Preview/batch parameter divergence

The live-preview dispatch `_find_peaks_single_frame` (`detectors.py:697`)
forwards `model_id` but **not `bg_sigma`** to the single-frame neural detector.
Once calibration (G2) is wired, the preview would silently use different
parameters than the batch run. Fix together with G2.

### G4 — The refine/persistence stage is fully dead

`models/refine.py` and the propose-then-refine machinery
(`_persistence_filter` / `_refine_block`, `find_vectors_neural.py:100-157`)
are vendored, parameterised (`persistence=` on `_neural_block` /
`_find_vectors_chunk_neural`) — and never enabled: `chunk.py:221` doesn't pass
`persistence`, no wizard control exists. Either wire it (it encodes real
physics: scan-neighbour persistence + Friedel) or delete it.

### G5 — Neural GPU use ignores the cluster GPU policy

The numba NXCORR path gates GPU use per worker (`_gpu_task_allowed`,
`SPYDE_FV_GPU`, default = worker "1" only) so CPU workers keep contributing.
`_neural_block` checks only `torch_gpu_device() is not None` — so on a
multi-process cluster **every worker builds a CUDA context and pushes batches
at the same GPU** (context VRAM × N workers, kernel contention). Relatedly,
each worker resolves the model itself: an HF-sourced `model_id` would be
concurrently downloaded by every worker into the same directory (G1 makes this
theoretical today, real after G1 ships).

### G6 — No MPS: Macs run the model on CPU while taking the "GPU" branch

`infer.load_model` picks `cuda`-else-`cpu` (`infer.py:27`), but
`torch_gpu_device()` supports MPS — so on Apple Silicon the batch path is taken
("GPU" branch) with a CPU-resident model. Works, but mislabeled and leaves the
Mac GPU idle.

### G7 — Checkpoint loading is neither hardened nor verified

`torch.load(ckpt_path, map_location=device)` (`infer.py:28`) without
`weights_only=True`: arbitrary-code-execution risk for downloaded checkpoints
on torch < 2.6, and a behaviour flip (potential load failure) on torch ≥ 2.6
where the default changed. Registry entries carry no `sha256`, so a corrupted
or tampered download is undetectable. Downloads also have no progress/status
surfacing — a first use of a remote model blocks wherever `get_model` was
called.

### G8 — No path from SpyDE usage back to training data

`RELEASING.md` says the detector "is meant to be revised indefinitely", trained
in yoloDiffraction — but there is no way to get labelled examples out of SpyDE.
Every user scan where the detector under- or over-fires is training signal we
currently discard.

## The plan

Each phase is independently shippable. Phases 0–1 fix documented-but-broken
behaviour; 2–4 extend. Per CLAUDE.md discipline, anything touching the live
compute path changes default behaviour only with a benchmark on real-scale data
(`benchmark_neural_spots.py` on sped_ag) and gets an Electron/Playwright
screenshot verification, not just pytest.

### Phase 0 — Make the model-upgrade loop real (G1, G7) — DONE 2026-07-15

1. Add `huggingface_hub` to `pyproject.toml` dependencies + re-lock (decision
   2026-07-15: hard dep, not an optional extra — it's a small pure-Python
   package and the registry code already treats it as the download backend).
2. Implement **`fv_refresh_models`**: staged verb that runs
   `registry.refresh_remote_registry()` via `run_on_worker` (network — never on
   the main loop), then re-emits `fv_models`. Wizard: a small ↻ button beside
   the Model dropdown + "checked — N models" status text.
3. **Resolve weights once, client-side**: `registry.ensure_local(model_id)` +
   `is_cached(model_id)`, called in the `fv_open` preview worker and at the top
   of the batch worker (`_start_batch._work`) before any compute is submitted;
   a "downloading model…" status line is emitted only when the file isn't
   cached. Workers then only ever read a locally-present file (also removes the
   N-way concurrent-download race, G5b). Per-byte `emit_progress` can be added
   later if model files grow beyond a few MB.
4. Harden loading: `torch.load(..., weights_only=True)` (verify both bundled
   checkpoints load — they store plain state dicts + scalar hyperparams);
   optional `sha256` field per registry entry, verified after download, with a
   clear log + fallback-to-bundled on mismatch.
5. Tests: `fv_refresh_models` emits the payload (hub monkeypatched); merge +
   checksum + `ensure_local` unit tests, all offline. E2E: wizard shows the
   refresh control (screenshot).

### Phase 1 — Wire auto-calibration end-to-end (G2, G3) — DONE 2026-07-15

1. When the wizard is opened with (or switched to) the neural method, run
   `calibrate_neural` on a handful of sample frames — reuse the NavBlurCache
   chunk plus a few frames spread across the scan — on a worker thread,
   cancellable via the tree cancel registry (wizard close must kill it).
2. Emit `fv_calibration {bg_sigma, thresh, confidence, scale_factor}`; the
   wizard shows the values as auto-filled (user-overridable) fields and includes
   them in `params` for **both** preview and run.
3. Thread `bg_sigma` through `_find_peaks_single_frame` so preview == batch
   (fixes G3 independently of the UI).
4. Stamp `model_id` + the effective calibrated params into the committed
   vectors' provenance dict (mechanism already exists in `commit.py`).
5. Tests: calibration-dispatch parity unit test (preview and chunk fn receive
   identical params); an e2e run on `load_test_data_si_grains` asserting the
   calibration payload arrives and the run completes ("Found N diffraction
   vectors").

Decision (2026-07-15): calibration **auto-runs on wizard-open** (parameter-free
is the selling point) with visible values + override; it's ~8 forward passes on
~4 frames.

### Phase 2 — Runtime parity + platform reach (G5, G6) — PARTIAL 2026-07-16

1. ~~Wire the policy~~ DONE: `_gpu_task_allowed(default_mode=...)` is consulted
   by `_neural_block` with a neural-specific unset-default of **"all"**
   (preserves today's behaviour exactly; `SPYDE_FV_GPU=off/one/N` now governs
   neural too). OPEN: the multi-process-cluster benchmark
   (all-workers-GPU vs. one-GPU-worker + CPU-rest, extend
   `benchmark_neural_spots.py`) that decides whether the unset-default should
   move to "one"/"N".
2. ~~MPS~~ DONE (code): `load_model` picks cuda → mps → cpu and smoke-tests one
   forward on MPS at load, degrading to CPU if an op is unsupported. OPEN:
   validate on real Apple-Silicon hardware.
3. OPEN (benchmark-gated): move `detect_batch`'s per-frame CPU zoom+normalize
   loop into batched torch ops if the CPU preprocess dominates GPU batch time.

### Phase 3 — Quality: propose-then-refine + a promotion gate (G4) — PARTIAL 2026-07-16

1. ~~Wire persistence~~ DONE: `persistence` flows wizard → `_coerce` →
   orchestrate → chunk → `_refine_block` (refine.py is live code now), behind
   the default-off "Neighbor refine" checkbox (neural only, batch-only — the
   preview has no scan neighbours). NB `refine()` normalises the col-2 value
   internally, so the raw-intensity column works as the relative-confidence
   term; the hard `min_persist=0.5` floor does the real filtering.
2. OPEN: add an **eval mode** to `benchmark_neural_spots.py`: precision/recall
   against synthetic ground truth + persistence-consensus pseudo-labels on
   sped_ag. Reference it from `RELEASING.md` as the required promotion gate —
   and use it to decide whether "Neighbor refine" should default ON.

### Phase 4 — Training-data flywheel (G8)

1. **"Export training data"** action on a vectors result: sample N frames +
   their detected labels into the yoloDiffraction dataset format (npz/json
   patches + scan fingerprint/metadata). v1 needs no UI beyond a menu entry and
   closes the loop RELEASING.md assumes.
2. Later: in-app correction (add/remove spots on the vector overlay) exporting
   hard examples — the highest-value labels are exactly the frames users had to
   fix.

## Out of scope (noted for later)

Other NN surfaces (denoising, segmentation, learned orientation) would reuse
the same registry/HF pattern — the registry is already model-agnostic apart
from the `SpotUNet` constructor in `get_model`; generalising the `arch` field
to a model-class key is a small refactor to do when a second model type
actually lands, not before.
