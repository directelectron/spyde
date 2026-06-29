SpyDE
=====

SpyDE is a desktop application for visualizing and analyzing electron microscopy
data (TEM, STEM, Cryo EM, 4D STEM, EELS). It is built on top of
[HyperSpy](https://hyperspy.org) and [PyXEM](https://pyxem.org), with original
support from Direct Electron (DE) for DE cameras and other EM detectors.

Architecture: an **Electron** frontend (React/TypeScript) drives a **Python**
backend (the `spyde` package) that runs as a sidecar over stdio. Heavy compute
is parallelised with Dask and GPU-accelerated with PyTorch where available.


Download
--------

Pre-built installers are on the [Releases](https://github.com/directelectron/spyde/releases) page.

| Platform | Download | Notes |
|----------|----------|-------|
| Windows  | `SpyDE Setup *.exe` | NSIS installer (per-user, no admin) |
| macOS    | `SpyDE-*.dmg`       | Open and drag SpyDE to Applications |
| Linux    | `SpyDE-*.AppImage`  | `chmod +x SpyDE-*.AppImage && ./SpyDE-*.AppImage` |

**First launch** sets up the Python analysis environment (including the
GPU-correct PyTorch wheel) with `uv`. This needs a network connection and may
take a few minutes; progress shows in the app's log panel. Subsequent launches
are instant. See [`electron/PACKAGING.md`](electron/PACKAGING.md) for how this works.

> The installers are currently **unsigned** — Windows SmartScreen / macOS
> Gatekeeper will warn on first run. On Windows: "More info → Run anyway"; on
> macOS: right-click the app → Open.


Development
-----------

You need [Node.js](https://nodejs.org) (18+) and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Frontend deps (also runs `npm install` in electron/ via postinstall).
npm install

# 2. Python backend env (resolves the git-pinned hyperspy/rosettasciio/anyplotlib
#    forks + the right torch wheel from uv.lock).
uv sync --extra tests

# 3. Run the app in dev mode (electron-vite dev + the Python sidecar via `uv run`).
npm run dev
```

Useful scripts (run from the repo root):

| Command | What it does |
|---------|--------------|
| `npm run dev`   | Launch the app (hot-reloading renderer + Python backend) |
| `npm run build` | Build the Electron frontend (`electron/out/`) |
| `npm run test`  | Playwright UI tests (`electron/tests/`) |
| `uv run pytest spyde/tests/migrated` | Python backend test suite |

To debug the **backend alone** (it speaks the `PLOTAPP:` JSON protocol on stdio):

```bash
uv run python -m spyde
```

### Building installers locally

```bash
cd electron
npm run dist        # build → stage Python sidecar → electron-builder
npm run dist:dir    # unpacked (faster, for smoke-testing the bundle)
```

Artifacts land in `electron/dist/`. See [`electron/PACKAGING.md`](electron/PACKAGING.md).


Releasing a new version
------------------------

Push a version tag to trigger the CI build matrix (Windows, macOS, Linux) and
publish a GitHub Release with all three installers:

```bash
git tag v0.1.0
git push origin v0.1.0
```

See [`.github/workflows/release.yml`](.github/workflows/release.yml).


License
-------

SpyDE is licensed under the **GNU General Public License v3.0 or later**
(GPL-3.0-or-later); see [`LICENSE`](LICENSE) for the full text. SpyDE builds on
[HyperSpy](https://hyperspy.org) and [PyXEM](https://pyxem.org), which are
themselves distributed under the GPL, so the combined work is GPL-licensed.
