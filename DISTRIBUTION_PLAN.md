# SpyDE Distribution, Installer & Auto-Update Plan

Scope: make SpyDE easy to **install**, **update**, and **run with a correctly
set-up GPU**, with `uv` as the engine. This is a design doc — no code yet.

## 1. Where we are today

| Concern | Current state |
|---|---|
| Bundling | **PyCrucible**: one self-extracting exe with an embedded `uv` (0.9.21) + `uv.lock`; on first run uv resolves deps into a venv and launches `main.py`. |
| Release | Tag-triggered CI (`release.yml`): builds `SpyDE.exe`→zip (Win), `.dmg` (mac), `.AppImage` (Linux); publishes a GitHub Release. |
| Install (Win) | Zip + `create_shortcut.ps1` run manually. **No installer, no Add/Remove Programs entry, no fixed install dir.** |
| Updates | **None.** No version check, no channel, no in-app update. |
| GPU | `torch` is a dep with `[tool.uv] torch-backend = "auto"`; `vector_orientation_gpu.select_device()` picks CUDA→MPS→CPU at runtime. **No first-run validation, no diagnostics surfaced, no driver guidance.** |
| Version | Duplicated: `pyproject.toml` and `spyde/__init__.py` (`0.0.1`). |

The PyCrucible + uv foundation is good — uv already does runtime dependency
resolution, which is exactly what an update mechanism can reuse. The gaps are
**install UX**, **update flow**, **GPU readiness**, and **version hygiene**.

## 2. Goals & non-goals

**Goals**
1. A real **installer** per platform (Win: MSI/NSIS w/ Add-Remove entry, Start-Menu + optional desktop shortcut, per-user default; mac: signed/notarized `.dmg`→`.app`; Linux: AppImage + optional `.deb`).
2. **Check for updates** (in-app, manual + optional auto-check on startup) against GitHub Releases, with a clear "update available → download/apply" flow.
3. **GPU readiness**: first-run (and on-demand) detection of CUDA/MPS, the right `torch` wheel installed, a diagnostics panel, and graceful CPU fallback with an explanation — never a silent slow path.
4. `uv`-powered throughout: dependency resolution, the GPU-correct wheel selection, and (where feasible) the update install step.
5. Robust: signed where possible, atomic updates (no half-written installs), offline-friendly first run optional, reproducible via `uv.lock`.

**Non-goals (this round)**
- An app store / winget / Homebrew cask (can layer on later).
- Delta/binary-patch updates (full-artifact updates are fine at this size cadence).
- Background silent auto-update without consent (we prompt).

## 3. Design decisions (the important ones)

### 3a. Two viable architectures — pick per the size/speed tradeoff

**Option A — "uv-managed app" (recommended).** The installer lays down a small
launcher + the bundled `uv` + the project (`pyproject.toml`/`uv.lock`). First
launch runs `uv sync` into a managed venv next to the install. Updates =
fetch the new `pyproject.toml`/`uv.lock` (or a versioned source bundle) and
`uv sync` again — fast, incremental, and **this is where uv shines**:
`uv sync` only changes what the lock changed, and `--torch-backend=auto`
fetches the right GPU wheel.
- *Pros:* tiny installer, fast incremental updates, GPU wheel handled by uv, no
  3 GB torch baked into the installer, reproducible from the lock.
- *Cons:* first run needs network (mitigate: optional "offline bundle" build
  that pre-seeds the uv cache); venv lives on the user's disk.

**Option B — "fully bundled".** Keep PyCrucible's single self-contained exe
(torch baked in) and just wrap it in an installer. Updates replace the whole exe.
- *Pros:* fully offline, one file.
- *Cons:* ~3 GB artifact per platform/GPU variant, slow updates (re-download
  everything), GPU variant matrix explodes (cu121/cu124/cpu/mps).

**Recommendation: Option A.** It's the natural fit for "powered by uv", keeps
artifacts small, makes GPU-correct installs automatic, and makes updates cheap.
Keep Option B's single-exe as a fallback "portable" download.

### 3b. Update transport: GitHub Releases as the channel
- A small `latest.json` manifest published per release (`version`, per-platform
  artifact URLs, sha256, min-supported-version, release notes URL, channel).
- App checks `https://github.com/<org>/spyde/releases/latest` (or the manifest)
  → compares semver → prompts.
- Channels: `stable` (tags `vX.Y.Z`) and optional `beta` (tags `vX.Y.Z-rc.N`).

### 3c. Versioning: single source of truth
- Make `spyde/__init__.__version__` the source; have `pyproject.toml` read it
  dynamically (`[tool.setuptools.dynamic]` or hatch), OR generate both from a
  git tag at build (`uv version` / `hatch-vcs`). Update checks compare
  `__version__` to the manifest.

### 3d. GPU readiness as a first-class step
- A `spyde/gpu_setup.py` module: detect platform + NVIDIA driver (via `nvidia-smi`
  / `torch.cuda`), decide the correct backend, and verify a real torch op runs.
- Installer/first-run: run `uv sync --torch-backend=auto` so the matching wheel
  is fetched (cu12x on Win/Linux+NVIDIA, MPS wheel on mac arm64, CPU otherwise).
- In-app **GPU diagnostics** panel (Help → GPU Status): device name, torch
  build, CUDA/MPS availability, a "re-run GPU setup" button, and the
  `gpu_unavailable_reason()` string we already expose. Surfaces driver-missing /
  CPU-only situations instead of the silent slow path the vector-OM work hit.

## 4. Concrete deliverables (phased)

### Phase 0 — Foundations (low risk, do first)
- [ ] Single-source version (`__version__` → pyproject dynamic).
- [ ] `spyde/_build_info.py` written at build time (version, git sha, channel,
      build date) for the About box + update checks.
- [ ] `tools/release.py` (uv-run) to cut a release: bump version, tag, push.

### Phase 1 — GPU readiness
- [ ] `spyde/gpu_setup.py`: `detect()`, `ensure_backend()` (wraps
      `uv sync --torch-backend=auto`), `diagnostics()`.
- [ ] Help → "GPU Status…" dialog (reuses `vector_orientation_gpu.select_device`
      / `gpu_unavailable_reason`).
- [ ] First-run check: if an accelerated device exists but torch is CPU-only,
      offer to fetch the GPU wheel via uv.
- [ ] Tests: `gpu_setup` detection logic (mock `nvidia-smi`/torch), subprocess-
      isolated as the existing GPU tests are.

### Phase 2 — uv-managed install + update core
- [ ] `spyde/updater.py`: `check_for_updates(channel)` → parse manifest,
      compare semver; `download_and_stage()`; `apply_update()` (atomic swap of
      the source bundle + `uv sync`, then relaunch).
- [ ] Help → "Check for Updates…" (manual) + optional startup check with a
      "remind me later / skip this version" choice (persisted in QSettings).
- [ ] `latest.json` manifest generation added to `release.yml`.
- [ ] Robustness: verify sha256, stage to a temp dir, swap on success only,
      keep the previous version for rollback, single-instance guard during apply.

### Phase 3 — Real installers
- [ ] **Windows**: NSIS or WiX MSI — installs launcher+uv+source under
      `%LOCALAPPDATA%\Programs\SpyDE` (per-user, no admin), Start-Menu + desktop
      shortcuts, Add/Remove Programs entry, file associations (`.hspy`, `.zspy`,
      `.mrc`), uninstaller. First-launch runs `uv sync`.
- [ ] **macOS**: `.app` in a signed+notarized `.dmg` (needs Apple Developer ID;
      flag as a prerequisite/cost). Sparkle-style update or our updater.
- [ ] **Linux**: keep AppImage; add optional `.deb`. AppImage self-update via
      `appimageupdate` or our updater.
- [ ] CI: extend `release.yml` to produce installers + manifest; matrix stays
      per-OS. Add code-signing secrets (Win EV cert, Apple ID) as a follow-up.

### Phase 4 — Polish
- [ ] About box (version, sha, GPU summary, licenses).
- [ ] Offline-install bundle variant (pre-seeded uv cache) for air-gapped labs
      (microscope PCs are often offline — worth it for this audience).
- [ ] Crash/first-run telemetry opt-in (optional).

## 5. Risks & mitigations
- **First-run network dependency (Option A).** Mitigate with an optional
  offline bundle and a clear progress UI during the initial `uv sync`.
- **Code signing.** Unsigned Win/mac apps trigger SmartScreen/Gatekeeper.
  Win EV cert (~$300/yr) and Apple Developer ($99/yr) are prerequisites for a
  clean install UX — call out as a decision/cost, ship unsigned in the interim
  with install instructions.
- **GPU wheel size/index.** `--torch-backend=auto` needs the PyTorch index
  reachable; pin a known-good torch version in the lock so updates are
  deterministic.
- **Update atomicity.** Never overwrite the running install in place — stage +
  swap + relaunch, keep N-1 for rollback.
- **Microscope-PC constraints.** Often locked-down/offline/older GPUs — per-user
  install (no admin), offline bundle, and CPU fallback all matter here.

## 6. Decisions (locked 2026-06-15)

| Question | Decision |
|---|---|
| Architecture | **Both: uv-managed (primary) + portable single-exe (offline fallback).** Build the uv-managed installer as the main path; keep PyCrucible's self-contained exe as a "portable/offline" download for air-gapped microscope PCs. |
| Windows installer | **NSIS** (.exe installer) — per-user, no-admin, Start-Menu + desktop shortcuts, Add/Remove entry, uninstaller. |
| Code signing | **Ship unsigned for now**; document SmartScreen/Gatekeeper click-through. Wire signing into CI later when certs are procured (Win EV, Apple Developer). |
| Update checking | **Startup check + manual.** Auto-check on launch with "skip this version" / "remind me later" (persisted in QSettings), plus Help → Check for Updates. Single `stable` channel to start; beta channel deferred. |
| Offline bundle | In scope (Phase 4) — the "portable single-exe" doubles as the offline path; optionally also a uv-cache-seeded bundle. |

### Resulting build matrix
- **uv-managed installer** (primary): Win NSIS `.exe`, mac `.dmg` (.app launcher
  + uv), Linux AppImage/`.deb`. Tiny; first run / updates = `uv sync`.
- **Portable single-exe** (fallback): the existing PyCrucible artifacts, renamed
  `SpyDE-portable-<os>` in the release.

## 7. Recommended starting point
Given "ship unsigned now" + "uv-managed", the lowest-risk first slice is
**Phase 0 + Phase 1 + the update *check*** (not yet auto-apply):
single-source version, `_build_info`, `gpu_setup.py` + GPU Status dialog, and a
Help → Check for Updates that compares `__version__` to the GitHub latest release
and links the download. That delivers visible value (version hygiene, GPU
diagnostics, update awareness) with no installer/signing dependencies, and the
NSIS installer + auto-apply land in Phases 2–3.
