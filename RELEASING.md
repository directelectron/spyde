# Releasing SpyDE

The single checklist for cutting a SpyDE release. The version is **derived from
the git tag** (`setuptools_scm`), so tagging is the act that sets the version —
there is no version string to bump by hand.

## Versioning model

- `pyproject.toml` declares `dynamic = ["version"]` with `[tool.setuptools_scm]`;
  `spyde/__init__.py` reads `importlib.metadata.version("spyde")`.
- A clean checkout **at tag `vX.Y.Z`** builds version `X.Y.Z`. Any other commit
  builds a dev version like `0.0.postN.devN+g<sha>` — that is expected off-tag and
  is **not** a release.
- Therefore the build environment must have the git history **and tags**. CI uses
  `actions/checkout@v4` with `fetch-depth: 0` (see `.github/workflows/{build,release}.yml`);
  a shallow clone has no tags and will produce a `0.0.postN.devN` wheel.

## Update channel: tag shape controls stable vs beta

`release.yml`'s `channel` job derives the channel from the pushed tag —
this is what determines who electron-updater's `autoUpdater` offers the
release to (see `electron/PACKAGING.md` "Auto-update + beta channel"):

| Tag shape | Channel | GitHub prerelease flag |
|---|---|---|
| `vX.Y.Z` (e.g. `v0.2.0`) | `stable` | no |
| `vX.Y.Z-rc.N` / `-beta.N` / `-alpha.N` (e.g. `v0.2.0-rc.1`) | `beta` | yes |

Users opt into the beta channel from Help → Check for Updates… in the app;
stable-channel users are never offered a beta build.

## Pre-release checklist

Run from a clean checkout of the commit you intend to tag.

1. **Lockfile is in sync** (a drifted `uv.lock` ships an installer that only
   fails on the user's first launch):
   ```bash
   uv lock --check
   ```
   If it fails, `uv lock`, review the diff, and commit it.

2. **Git dependencies are pinned** — `pyproject.toml`'s git deps must reference
   explicit commit SHAs (or tags), never moving branches:
   ```bash
   grep -nE "@git\+|git\+https" pyproject.toml   # each must end in @<40-char-sha> or a tag
   ```
   (`anyplotlib` is now a normal PyPI dependency. `hyperspy` / `rosettasciio` are
   pinned to SHAs.)

3. **Python test suite green** (Qt-free migrated suite):
   ```bash
   uv run --extra tests pytest spyde/tests/migrated -q
   ```

4. **TypeScript + build green:**
   ```bash
   cd electron && npm ci && npm run typecheck && npm run build
   ```

5. **Electron e2e** (the UI's only automated gate):
   ```bash
   cd electron && npm test          # default 'electron' project (synthetic data)
   ```

6. **Manual checks the automated suite cannot cover** (do these on a real
   machine — CI has no GPU/display and the migrated suite forces `SPYDE_NO_DASK=1`):
   - **Distributed navigator path** — open a multi-GB 4D-STEM scan, drag the
     navigator; the diffraction pattern must track without freezing. If the
     `repro_*.py` distributed scripts are present, run them directly
     (`uv run python -m spyde.tests.repro_<name>`); they spin a real
     `LocalCluster` and won't run under pytest.
   - **GPU find-vectors / orientation** — on a CUDA box, run Find Vectors and
     Orientation Mapping on a real dataset (e.g. `pyxem.data.sped_ag()`); confirm
     results look right and nothing segfaults. (The numba subpixel kernel is unit-
     tested for arithmetic, but the live CUDA path needs a real run.)
   - **No-GPU fallback** — on a machine without `numba`/CUDA, the app must still
     launch and Find Vectors must fall back to CPU. (Guarded by
     `test_find_vectors_no_numba.py`, but verify the packaged app.)
   - **IPF colour-key legend** — open an Orientation map; the colour-key triangle
     pins in the corner on the 2-D map and hides on the 3-D view.
   - **Clean shutdown** — quit the app; confirm no orphaned `python.exe` / Dask
     worker processes remain (Task Manager / `ps`).

7. **Versions agree** — `electron/package.json` `version` should match the tag you
   are about to cut (it is set by hand; keep it in step with the Python tag).
   `release.yml`'s `build` job now enforces this and fails the release if they
   drift, but fix it before tagging rather than relying on that as your check.

## Cutting the release

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

The tag triggers `.github/workflows/release.yml`, which builds the Electron app +
uv-managed Python sidecar on Windows / macOS / Linux in parallel and publishes a
single GitHub Release once all three legs pass (`fail-fast`, so users never see a
partial release). See `electron/PACKAGING.md` and `DISTRIBUTION_PLAN.md` for how
the installer stages `uv` + the locked sources and builds the venv on first launch.

## Notes

- Installers are currently **unsigned** (Windows SmartScreen / macOS Gatekeeper
  will warn). Signing is tracked separately.
- The neural disk-detector model upgrade path is documented in
  `spyde/models/RELEASING.md`.
