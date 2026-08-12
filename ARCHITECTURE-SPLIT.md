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

**Second pass — `SessionBase`, `de_shell.app`, and de-groundcrew**

- `de_shell.session.SessionBase`: window/plot registry, main-loop marshalling,
  the settings store. SpyDE's `Session` inherits it; `WindowManagerMixin` keeps
  only the teardown that knows about signal trees and selectors. Lifted
  verbatim, including the non-obvious `unregister_plot` semantics (identity, all
  occurrences — `register_plot` does not dedupe).
- `de_shell.app.run()`: a generic asyncio loop parameterised by a session
  factory, with `on_message`/`on_ready` hooks. **SpyDE still uses its own
  `backend/app.py`** — adopting this is a follow-up, not done here.
- `@de/shell-testing`: `launchApp` parameterised by app dir + ready signals.
  `countColorPixels` now THROWS on an unknown `kind` — the SpyDE version
  silently counted nothing, which is how a bad assertion passes vacuously.
- **`apps/groundcrew` — a real, working second app.** Fixed panes (control
  sidebar, viewer, stats strip, status bar, collapsible log), its own
  `de_groundcrew` Python package, a simulated camera with a free-running
  acquisition thread, and newest-wins painting. Its whole dependency set is
  `de-shell` + numpy. `packages/de-shell/tests/test_boundary.py` enforces that
  in a subprocess and is mutation-tested (adding `import dask` to a de_shell
  module makes it fail).

**Four things the shell/app contract requires that were not obvious**

Each of these cost a debug cycle building the second app, and each is a
candidate for the shell to own rather than have every app rediscover:

1. `_electron.register(fig)` must be called and its RETURNED fig_id used. A
   made-up id leaves the figure unregistered, so no trait observers attach and
   `set_data` emits nothing — the figure mounts, sizes and titles correctly and
   then never updates, silently.
2. The main process must supply an `onBinary` handler. The shell turns on
   `APL_BINARY_TRANSPORT`, so every pixel update is a PLOTBIN frame; without the
   handler the runner parses them and drops them.
3. A `srcdoc` figure iframe inherits the PARENT page's CSP, so `script-src` needs
   `blob:` or anyplotlib's ESM boot fails with "Failed to fetch dynamically
   imported module".
4. State updates that arrive before the iframe mounts are lost — the backend
   only sends changes, so the first frame must be retained and replayed on load.

**Third pass — the actions framework and a shared figure**

- `de_shell.actions.lifecycle` — `run_on_worker`, `bump_generation` /
  `is_current`, `replace_tree_attr`, `progress_emitter`, `window_computing`.
- `de_shell.actions.registry` — the dispatch MECHANISM (lazy dotted-path
  resolution, wizard-schema lookup, the WindowController protocol). The tables
  stay in SpyDE and are registered at import. `spyde.actions.registry`
  rebinds `STAGED_HANDLERS` to the shell's dict afterwards, so there is ONE
  authority: otherwise a `register_staged()` call lands in one dict and is read
  from the other.
- `de_shell.actions.context` (`ActionContext`), `de_shell.actions.wizard`
  (`WizardController`), `de_shell.actions.figure_registry`,
  `de_shell.timing` (`reliable_sleep`).
- `de_shell.plotting.figure` — `FigureView` + `robust_levels`. **Ground Crew now
  uses it**; its bespoke viewer is deleted.

Two API notes worth knowing:

*Re-exports are deliberate, and are not compatibility shims.*
`spyde.actions.lifecycle` re-exports the shell primitives alongside its own
domain lifecycle, and `spyde.actions.context` / `registry` do the same. An
action imports one module; it should not have to know which half a helper came
from. The alternative — splitting imports across ~58 call sites — spreads that
knowledge everywhere for no benefit.

*`figure_registry` no longer reaches into the app.* It used to lazily import
SpyDE's `actions.views` to evict per-window state. Apps now call
`register_evictor(fn)`; `views.py` does so at import.

**Fourth pass — `@de/shell-renderer`**

The renderer kernel, taken from the top of the shopping list rather than
wholesale. What landed is the part BOTH apps had already written twice:

- `figureBridge.ts` — the iframe registry, state retention and replay. Lifted
  from SpyDE's implementation, which was the better of the two: it already
  carried the three subtleties Ground Crew's simpler copy lacked (replay takes
  an explicit target; binary frames stash per PANEL by `header.geom`; replay
  sends a COPY because postMessage transfers and detaches). Plain TypeScript,
  no React, so it unit-tests without a renderer — 13 tests, one per shipped bug.
- `FigureFrame.tsx` — the iframe host: registration, replay-on-load, resize
  reporting, and `srcdoc` vs `src`.
- `protocol.ts` — the core message union. Apps extend it additively, which works
  because every variant carries an index signature.

**Both apps use it.** Ground Crew's bridge and hand-rolled iframe are deleted.
SpyDE DELEGATES: the bridge exposes its maps as `{current}` boxes, so
`iframeRefs` / `latestStates` / `latestBinaryStates` keep exactly the
`MutableRefObject<Map<…>>` shape they had, and none of the seven components that
thread them as props needed touching.

The `grid_present` spec is the evidence that the subtle behaviour survived: its
diagnostic shows the presented slide retaining TWO panels
(`panel_…_geom::image_b64` twice), which is precisely what breaks if the
per-panel stash key is wrong.

**Fifth pass — the context carve (`shellState.ts`)**

The chrome slice of the reducer, which both apps had written independently:
status, the busy indicator, the stdout/log rings, first-run env setup, the
backend-death latch, per-window computing overlays, and toolbar action state.

Composition rather than a second context: an app's state EXTENDS `ShellState`
and its reducer delegates in its `default:` branch. `shellReducer` is generic
over the state type and returns it UNTOUCHED (same object identity) for an
action it does not own, so the delegation is safe in either order and does not
re-render on foreign actions. `toShellAction(msg)` maps the chrome messages and
returns null for everything else, so an app writes
`if (a) { dispatch(a); return }` and its own switch carries only its own cases.

SpyDE's `State` now extends `ShellState`, 13 reducer cases are gone, and
`LogEntry` / `LOG_MAX` / `SubItem` / `EnvPhase` / `EnvSetupState` are re-exported
from the shell so the components importing them from the context keep working.
Ground Crew moved from ad-hoc `useState` to the same reducer.

Deliberately NOT extracted: the **window/figure registry**. SpyDE's is entangled
with its named-view/chip system (`view`, `view_label`, strain components) and
Ground Crew has no window registry at all — one fixed pane. Extracting it now
would be generalising from a single consumer. It waits for a second app that
needs it.

Also still SpyDE's: the report/movie state and the ~50 domain message cases.

**Not started**

- SpyDE's `Plot` still does not use `FigureView`. It carries the array cache,
  the tiered navigator read, overlay layers and tile mode, so adopting the
  shared core means layering those on top — a real piece of work, and riskier
  than anything done so far because the navigator read path is the app's most
  performance-sensitive code (see the Live-Display section of CLAUDE.md).
  `FigureView` is deliberately the in-memory core so that layering is possible.
- `spyde/drawing/selectors/` and `toolbars/`: `base_selector` still reaches into
  `actions.overlay`, and `plot_control_toolbar` into `spyde.TOOLBAR_ACTIONS`.
  Both want the `register_*` hook treatment.
- `@de/shell-preload` and `@de/shell-renderer`. Ground Crew's preload and its
  `useFigureBridge` are a second copy of SpyDE's shapes — that duplication is
  the argument for extracting them, and now there are two call sites to
  generalise from instead of one.
- The figure custom-protocol, `createWindow` + lifecycle diagnostics, and the
  generic IPC handlers are still inline in `electron/src/main/index.ts`.
- `apps/autopilot` is an empty directory.

**Known defect in the groundcrew scaffold:** the figure does not fit its pane —
the resize round-trip (renderer `ResizeObserver` → `groundcrew:resize` →
`sendResize` → `de_shell.app._resize_figure`) never reaches Python, so the
figure renders at anyplotlib's default size and overflows. Cosmetic, and
deliberately left rather than chased further in a scaffold.

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
