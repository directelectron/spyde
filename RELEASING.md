# Releasing SpyDE

The single checklist for cutting a SpyDE release. The Python version is
**derived from the git tag** (`setuptools_scm`), so tagging is the act that sets
it. The one hand-maintained version string is `electron/package.json` — and the
**Prepare Release** workflow (`.github/workflows/prepare_release.yml`) bumps it
for you, so nothing is edited by hand in practice.

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

7. **Versions agree** — `electron/package.json` `version` must match the tag you
   are about to cut. Don't bump it by hand: run the **Prepare Release** workflow
   (below), which bumps it in a reviewable PR. `release.yml`'s `build` job
   enforces the match and fails the release if they drift (this is exactly how
   the `v0.1.0-rc.3` release died: tag `-rc.3`, package.json `0.1.0`).

## Cutting the release

1. **Run the Prepare Release workflow** (Actions tab → *Prepare Release* → *Run
   workflow*). The bump options (and the `beta` checkbox) mirror anyplotlib's
   Prepare Release; only the pre-release *suffix* differs — this repo emits
   semver `-rc.N` where anyplotlib emits PEP 440 `bN`, because the version of
   record is `electron/package.json` (npm/electron-builder), which is
   semver-only. Pick the bump:
   - `major` / `minor` / `bugfix` — standard bumps (optionally cut as an
     `-rc.1` by ticking the `beta` checkbox).
   - `pre-release` — next `-rc.N` on the current base (always a beta; the
     `beta` checkbox is implied).
   - `finalize` — drop the `-rc.N` suffix to promote the current release
     candidate to stable (e.g. `0.2.0-rc.2` → `0.2.0`).

   From an `-rc.N` base, `major`/`minor`/`bugfix` are rejected (they'd skip the
   in-progress version) — use `finalize` to ship it, or `pre-release` for the
   next candidate. It bumps `electron/package.json` (+ lockfile), runs the
   lockfile/pin pre-flight checks, and opens a `release/vX.Y.Z` PR.

   > Exception: if `electron/package.json` already equals the version you want
   > to tag (no bump needed), skip straight to tagging.

2. **Review and merge the PR** — CI runs the full matrix on it.

3. **Tag the merge commit with exactly the tag named in the PR** and push:

   ```bash
   git fetch origin
   git tag vX.Y.Z origin/main
   git push origin vX.Y.Z
   ```

The tag triggers `.github/workflows/release.yml`, which builds the Electron app +
uv-managed Python sidecar on Windows / macOS / Linux in parallel and publishes a
single GitHub Release once all three legs pass (`fail-fast`, so users never see a
partial release). See `electron/PACKAGING.md` and `DISTRIBUTION_PLAN.md` for how
the installer stages `uv` + the locked sources and builds the venv on first launch.

4. **Confirm the release actually has installers** before announcing it:

   ```bash
   gh release view vX.Y.Z --repo CSSFrancis/spyde --json assets --jq '.assets[].name'
   ```

   Expect the three installers (`SpyDE Setup *.exe`, `SpyDE-*.dmg`,
   `SpyDE-*.AppImage`), their `.blockmap`s, and the `latest*.yml` update feeds.
   `release.yml`'s `finalize` job now refuses to un-draft an asset-less release,
   so a green run with an empty release should no longer be possible — but check.

## ⚠️ Golden rule: NEVER create or edit the GitHub Release by hand

`release.yml` owns the entire release object. It creates a **draft**, the three
build legs upload installers **into that draft**, and only then does `finalize`
un-draft it and stamp the beta/stable flag. Do **not**:

- run `gh release create` / click "Draft a new release" for a release tag, or
- create the release before pushing the tag, or
- flip a release's draft/prerelease flags manually mid-run.

**Why this is load-bearing (this bit us on `v0.2.0-rc.1`):** electron-builder's
GitHub publisher only uploads into an existing release when it is still a
**draft**. If it finds a **non-draft** release for the tag, it *skips every
asset* (`"existing type not compatible with publishing type"`) — and, because
that skip is non-fatal, the run goes **green with an empty release**. A
hand-made release (created ~30 min before the workflow ran) is exactly what
stranded `v0.2.0-rc.1` with zero installers. The workflow now force-drafts the
release and hard-fails on a skipped publish, but the simplest guarantee is: push
the tag and let CI do everything.

### If a release ends up empty anyway

1. Delete the empty release (keep the tag):
   `gh release delete vX.Y.Z --repo CSSFrancis/spyde --yes`
2. Cut the **next** rc through Prepare Release (`bump=pre-release`) rather than
   reusing the stranded tag — cleaner provenance. Re-pushing a deleted-release
   tag also works (`git push origin :refs/tags/vX.Y.Z && git push origin vX.Y.Z`)
   once no release object exists for it.

## Notes

- Installers are currently **unsigned** (Windows SmartScreen / macOS Gatekeeper
  will warn). Signing is tracked separately.
- The neural disk-detector model upgrade path is documented in
  `spyde/models/RELEASING.md`.
