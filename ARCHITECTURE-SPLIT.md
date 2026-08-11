# Splitting the shell out of SpyDE

Three products share one substrate:

| App | Layout | Domain |
|---|---|---|
| **SpyDE** | MDI workspace | Offline EM data analysis (HyperSpy, Dask, lazy multi-GB scans) |
| **de-groundcrew** | Fixed panes | Live camera/hardware control via the DE Server SDK (today: PySide6) |
| **de-autopilot** | Fixed panes | Automated acquisition sequencing |

Ground Crew and Autopilot are **live, in-memory** applications. They must not pull
in HyperSpy, Dask, RosettaSciIO, pyxem, or any of the lazy-read machinery. That is
the single constraint that fixes the boundary.

## The boundary

**In the shell** — anything that answers *"how do I be a desktop app with a Python
brain and pictures in it?"*

**In the app** — anything that answers *"what is the data and what do you do to it?"*

The `array_cache` tiering, `signal_tree`, the navigator read path, `compute_backend`'s
distributed branch, and every `spyde/actions/*` handler are answers to the second
question. They stay in SpyDE.

## Layout

```
packages/                     → becomes the `de-shell` repo
  shell-main/       @de/shell-main       Electron main process kernel
  shell-preload/    @de/shell-preload    contextBridge core surface
  shell-renderer/   @de/shell-renderer   React kernel, shell UI, layouts
  shell-testing/    @de/shell-testing    Playwright harness
  de-shell/         de_shell (Python)    backend kernel

apps/
  groundcrew/       de-groundcrew
  autopilot/        de-autopilot

electron/ + spyde/            SpyDE, at its current paths (see "Deferred" below)
```

Root `package.json` declares npm workspaces; root `pyproject.toml` declares a uv
workspace. Each package is independently publishable, so the eventual split into
four repos is a `git filter-repo` per directory, not a redesign.

## Python: `de_shell`

| Module | From | Notes |
|---|---|---|
| `de_shell/ipc.py` | `backend/ipc.py` | verbatim — `emit`, `emit_status/error/progress`, PLOTBIN frames, stdin reader |
| `de_shell/log_stream.py` | `backend/log_stream.py` | area tagging is rule-driven; apps supply their own `_AREA_RULES` |
| `de_shell/process_guard.py` | `backend/process_guard.py` | verbatim |
| `de_shell/debug_flags.py` | `backend/debug_flags.py` | generic env-flag reader |
| `de_shell/app.py` | `backend/app.py` | asyncio loop; prewarm hooks become a pluggable list instead of hardcoded HyperSpy/RosettaSciIO warmups |
| `de_shell/session.py` | `backend/session.py` | `SessionBase`: action dispatch, `set_main_loop`/`_dispatch_to_main`, plot registry, shutdown. SpyDE's `Session` subclasses it and keeps the trees/Dask/file I/O |
| `de_shell/windows.py` | `backend/_session_windows.py` | window/figure registry, minus the signal-tree teardown |
| `de_shell/actions/` | `actions/{registry,context,wizard,lifecycle,figure_registry}.py` | the staged-wizard framework. `action.py`'s `TransformAction`/`RegionAction` import HyperSpy and stay in SpyDE |
| `de_shell/plotting/` | `drawing/plots/{plot,plot_window}.py`, `colormaps.py`, `selectors/`, `toolbars/` | `Plot` is already an anyplotlib wrapper with no HyperSpy import |
| `de_shell/compute.py` | new | `ThreadCompute`: the `concurrent.futures` half of `ComputeBackend`. SpyDE's `ComputeBackend` subclasses it and adds the distributed branch — core never imports Dask |
| `de_shell/testing/` | `backend/_session_testharness.py` | the app-agnostic half |

Staying in SpyDE: `signal_tree`, `signal_node`, `array_cache/`, `drawing/update_functions.py`,
`dask_manager`, `workers/`, `signals/`, `models/`, `ebsd/`, `drift/`, `fitting/`,
`spectroscopy/`, `atoms/`, `external/`, and all of `actions/` bar the framework files.

## TypeScript

### `@de/shell-main`
`runner.ts` (sidecar spawn/supervise + PLOTAPP parsing), `pythonEnv.ts` + `envProgress.ts`
(first-run `uv sync`), `updater.ts` + `updater_errors.ts`, the figure custom-protocol
(scheme name becomes a parameter, not the hardcoded `spyde-fig`), window creation +
the sleep/resume lifecycle diagnostics, and the generic IPC handlers: file/folder/save
pickers, clipboard PNG, `open-external`, `open-path`, update control, GPU triage.

Composition root: `createShellApp({ appId, protocolScheme, pythonModule, menu, ipc })`.
SpyDE's `main/index.ts` shrinks to a menu spec plus its own handlers (report export,
zarr folder open).

### `@de/shell-renderer`
- `kernel/protocol.ts` — the **core** message union: `ready`, `dask_ready`, `status`,
  `error`, `progress`, `env_setup`, `backend_exited`, `figure`, `toolbar_config`,
  `window_*`, `state_update`(+binary), `action_active`, `sub_item`, `log*`,
  `download_*`, `wizard_event`, `selector_info`, `loading`, `open_path`. Apps extend
  the union; the existing `MsgBase` index signature already makes that additive.
- `kernel/ShellContext.tsx` — window/figure registry, toolbar model, status, the log
  ring, `sendAction`. Apps inject extra reducer cases through an extension hook.
- `ui/` — MenuBar, StatusBar, LogPanel, Dropdown, AnchoredMenu, Pill, CaretBox,
  WizardShell + wizardHooks, FloatingToolbar, ThemePanel, UpdateCard/Dialog/Gate,
  GpuStatus/GpuHelp, EnvSetupOverlay, DownloadToasts, Tour + guideDriver, and the
  figure `<iframe>` host.
- `layout/` — **`MdiLayout`** (today's `MDIArea`/`SubWindow`) and **`FixedLayout`**
  (named slots in a fixed grid). Both consume the same window registry, so a figure
  the backend opens lands in an MDI subwindow in SpyDE and in a named pane in
  Ground Crew with no backend change.
- `sidebar/SidebarHost` — panel registration. SpyDE registers PlotControlDock /
  ReportSidebar / MetadataPanel; Ground Crew registers its control panel.

Staying in SpyDE: report/presentation (`ReportFigureCell` 2320 lines, `ReportSidebar`,
`PresentMode`, `MovieEditor`, all report cells), every wizard (FindVectors, Orientation,
Strain, Drift, Ebsd, Fit, Crop, CenterZeroBeam), `PeriodicTable`, `CompositionPanel`,
`CodPicker`, `ChunkViewer`, `DaskMonitor`, `MetadataPanel`, `NavShapeDialog`,
`StackDialog`, `WindowContent`.

### `@de/shell-testing`
`launchApp` parameterized by app dir + ready signals, `waitForLog`/`waitForMessage`,
`waitForWindowCount`, `countColorPixels`, `assertNoJsErrors`. SpyDE keeps its
domain helpers (`loadTestVectors`, `navWindow`, `dragCrosshair`, `waitForVectorActions`).

## Status

Branch `refactor/split-shell-monorepo`.

**Landed and verified**

- Monorepo skeleton: npm workspaces (`packages/*`, `electron`, `apps/*`) and a uv
  workspace with `spyde` depending on the local `de-shell`.
- `de_shell`: `ipc`, `log_stream`, `process_guard`, `debug_flags`,
  `plotting/colormaps`, `plotting/selectors/utils`, and a new dask-free
  `compute.ThreadCompute`. 73 files of imports rewritten; no shims.
- `@de/shell-main`: `backendProcess` (was `runner.ts`), `pythonEnv`,
  `envProgress`, `updater`, `updaterErrors`, plus `config.ts`. SpyDE's
  `main/index.ts` now calls `configureShell({appId:'spyde', …})` and imports the
  package through a tsconfig path + vite alias (raw TS, no build step between
  editing the shell and running the app).
- Gates: main + web typecheck clean, `npm run build` clean, 14 shell unit tests,
  **2593 pytest passed / 0 failed**, and `tests/shell_split_smoke.spec.ts` drives
  the real app — 4 windows, 410k painted pixels, 15.7k overlay pixels, the
  status line, no import errors.

**Known pre-existing failure, not caused by this work:** at ~94% the full pytest
run dies with a native `Fatal Python error: Aborted` inside pyxem's numba matcher
(`_get_full_correlations`) on a worker thread. `test_vector_orientation_om.py`
passes in isolation (2 passed, 8 s); this is the concurrent-matcher crash the
global `PYXEM_LOCK` was added for, surfacing under accumulated full-suite state.

**Not started**

- The rest of `de_shell`: `SessionBase`, the window/figure registry, the actions
  framework, and the plotting layer. Each needs its app back-reference turned
  into a hook first — the ones found so far are `lifecycle` →
  `compute_dispatch` / `actions.base` / `actions.center_zero_beam` /
  `update_functions`, `base_selector` → `actions.overlay`,
  `plot_control_toolbar` → `spyde.TOOLBAR_ACTIONS`, `figure_registry` →
  `actions.views`, `registry` → `import spyde`. The `log_stream` area rules are
  the worked example of the pattern (`register_area_rules`, called from
  `spyde/__init__.py`).
- `@de/shell-preload`, `@de/shell-renderer`, `@de/shell-testing`.
- The figure custom-protocol, `createWindow` + lifecycle diagnostics, and the
  generic IPC handlers are still inline in `electron/src/main/index.ts`.
- `apps/groundcrew` and `apps/autopilot` are empty directories.

**Note on the renderer.** Splitting `SpyDEContext.tsx` (2012 lines) and
`protocol.ts` (1079) is the one part where a clean break is the wrong tool: the
generic half (window/figure registry, toolbars, status, log) and the SpyDE half
(report, movie, drift, vectors, composition) are interleaved through one reducer.
Recommend carving it incrementally behind the extension hook, with the
groundcrew app as the forcing function for what the core actually needs.

## Deferred on purpose

SpyDE's own move to `apps/spyde/` is a **pure directory rename** with no code change,
and it invalidates paths in CI, `electron-builder.yml`, `crucible.toml`, `.idea/`, and
CLAUDE.md. Doing it in the same change as the extraction would make the diff
unreviewable and every test failure ambiguous between "wrong boundary" and "wrong
path". It is a separate, mechanical commit once the boundary is proven.

## How we know the boundary is right

`apps/groundcrew` is the test. It is a real Electron app on `FixedLayout` +
`SidebarHost`, backed by a `de_shell` Session with no HyperSpy or Dask installed in
its environment. If it boots, shows a live image, and streams logs, the shell is
genuinely app-agnostic. If it needs a shim back into `spyde/`, the boundary moved and
we fix it there rather than papering over it.
