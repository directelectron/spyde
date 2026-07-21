# Packaging SpyDE (Electron + uv Python sidecar)

SpyDE is an Electron shell that spawns the `spyde` Python backend as a
**sidecar** over stdio (the `PLOTAPP:` JSON protocol). Packaging follows
`DISTRIBUTION_PLAN.md` **Option A (uv-managed)**: ship a tiny installer that
materialises the Python env on first run with `uv`, so the GPU-correct torch
wheel is fetched per machine and updates are a cheap incremental `uv sync`.

## Pieces

| File | Role |
|---|---|
| `electron-builder.yml` | electron-builder config: NSIS (Win) / dmg (mac) / AppImage (Linux); ships the staged Python payload as an `extraResource`. |
| `scripts/bundle-python.mjs` | Stages the sidecar payload into `resources/python`: `pyproject.toml` + `uv.lock` + the `spyde/` source (tests excluded) + the `uv` binary. |
| `src/main/pythonEnv.ts` | At launch, `resolvePythonEnv()`: dev → `uv run` from the repo; packaged → `uv sync --frozen` into a venv in `app.getPath('userData')/python-env` (first run / when `uv.lock` changes), then launch that venv's python. |
| `vendor/uv/<platform>/uv[.exe]` | The pinned `uv` binary per OS (CI populates; falls back to `uv` on PATH locally). |

## Build

```bash
cd electron
npm install            # picks up electron-builder + electron-updater
npm run dist           # build renderer/main → stage python → electron-builder
# or, unpacked (faster, for smoke-testing the bundle):
npm run dist:dir
# CI publishes with:
npm run dist -- --publish always
```

Artifacts land in `electron/dist/`. The installer is small (~tens of MB) — it
does **not** contain torch/numpy; those are fetched by `uv sync` on first run.

### First-run behaviour (packaged)

1. App starts, shows the window, status "Setting up Python environment…".
2. `pythonEnv.ts` runs the bundled `uv sync --frozen --no-dev` with
   `UV_PROJECT_ENVIRONMENT` pointing at `…/userData/python-env` (the bundle is
   read-only, so the venv must live in the writable user dir). `torch-backend =
   "auto"` (from `pyproject.toml [tool.uv]`) selects the right wheel
   (CUDA / MPS / CPU). uv output streams to the app log panel.
3. On success a `.spyde-lock-hash` stamp is written; later launches skip the
   sync unless `uv.lock` changed (→ an update re-syncs only what changed).

## Auto-update + beta channel

Wired via `electron-updater` (`electron/src/main/updater.ts`):

- **Channel**: `vX.Y.Z` tags publish to `stable`; any prerelease suffix
  (`-rc.N`/`-beta.N`/`-alpha.N`) publishes to `beta` and is marked as a GitHub
  prerelease (`.github/workflows/release.yml`'s `channel` job). The user picks
  stable/beta in Help → Check for Updates…; the choice is persisted to
  `userData/update-channel.json` (Electron side) and mirrored into
  `~/.spyde/settings.json` via the `set_update_channel` staged action (Python
  side, for visibility/debugging).
- **Flow**: startup check (~5s after launch) + manual "Check Now". Downloads
  are NOT automatic — `autoDownload = false`, so the user clicks "Download"
  then "Restart to Install" once it's ready. No silent background updates.
- **GPU Status**: Help → GPU Status… surfaces the existing
  `vector_orientation_gpu.select_device()`/`gpu_unavailable_reason()`
  diagnostics via a `get_gpu_status` staged action — no new detection logic,
  just a UI window onto what already existed.
- **CI**: `release.yml`'s `build` job runs `electron-builder --publish always`
  against a draft release `prepare-release` creates up front (avoids
  electron-builder's concurrent-matrix-publish duplicate-draft race), so
  `latest.yml`/`latest-mac.yml`/`latest-linux.yml` are generated and attached.
  `finalize` un-drafts the release once all 3 platform legs succeed.

## What's intentionally deferred (follow-ups)

- **Code signing / notarization.** Shipping unsigned for now (Gatekeeper /
  SmartScreen will warn — see `DISTRIBUTION_PLAN.md §5`). Wire Apple Developer
  ID + Windows EV cert into CI when procured; `mac.identity: null` removes the
  ad-hoc signing attempt meanwhile. **macOS sign+notarize is a ready-to-follow
  checklist in [`SIGNING.md`](SIGNING.md)** (certs → GitHub secrets →
  electron-builder.yml + release.yml diffs → verify); the entitlements file it
  needs already exists at `build/entitlements.mac.plist`.
- **Offline bundle.** The "portable" PyCrucible single-exe (torch baked in)
  remains the air-gapped fallback per the locked decision; this uv-managed
  installer is the primary path.
- **First-run progress UI.** Currently surfaced via the status text + log panel;
  a dedicated splash/progress bar during the (potentially multi-minute) initial
  `uv sync` is polish.

## Notes

- Dev is unchanged: with no `resources/python` payload, `resolvePythonEnv()`
  returns `uv run python -m spyde` from the repo root.
- The staged `resources/python/` and `vendor/` and `dist/` are git-ignored
  (generated / large).
