# SpyDE Actions — how they work and how to add one

Everything a toolbar button, caret, or wizard does goes through this package.
This document is the contract: the action shapes, the two dispatch paths, the
lifecycle every action follows, and the step-by-step for adding a new one.
Copyable skeletons live in [`_template_action.py`](_template_action.py) (kept
compiling by `test_template_skeletons.py`). These same actions are designed to
run from scripts and Jupyter notebooks too (function ↔ anywidget ↔ toolbar
parity): every host returns the same `spyde.signals` result objects and resolves
wizard parameter schemas via `registry.wizard_parameters(key)` — see
`spyde/api.py` and `spyde/signals/`.

## 1. The action taxonomy

| Shape | Contract | Base / pattern | Examples |
|---|---|---|---|
| **View action** | one-shot UI command, no tree change | plain `fn(ctx, …)` | zoom, reset, `tile_views` |
| **TransformAction** | signal + params → a **new node in the SAME tree** | `action.TransformAction` | Rebin (`Rebin2DAction`), CZB apply |
| **RegionAction** | interactive ROI → a **linked live output plot** | `action.RegionAction` | Virtual Imaging, FFT, Line Profile, Vector VI |
| **Wizard** | staged caret: open → tune → run → commit → close | `wizard.WizardController` + staged handlers | Find Vectors, Orientation, Vector-OM, EBSD, Strain, CZB, DPC |
| **Commit** | promote a live/finished result to a **NEW SignalTree** | `commit.commit_result_tree` | strain Commit, VOM result windows |

Deciding: does it need an ROI? → RegionAction. Does it produce a new node of
the same dataset? → TransformAction. Does it need staged interaction (preview,
heavy compute, result windows)? → Wizard. Is the output a standalone dataset?
→ it goes through `commit.py` (either door).

## 2. The two dispatch paths (never invent a third)

Every renderer click arrives as `{action, payload, window_id}` at
`Session.dispatch_action` (`spyde/backend/_session_actions.py`).

**Path 1 — YAML toolbar actions.** Declared in `spyde/toolbars.yaml`;
`_dispatch_toolbar_action` resolves the dotted `function`, builds an
`ActionContext`, and calls either `ActionSubclass(ctx).run(**params)` or
`fn(ctx, action_name=…, **params)`. A returned selector is tracked in
`session._action_artifacts[(window_id, name)]` (with the Action instance)
so deselecting the action closes its outputs and `update_vi` caret edits
reach `update_live_params`.

YAML gate keys: `signal_types` / `exclude_signal_types` (match the signal's
`_signal_type` string — covers lazy+eager), `signal_class` (isinstance),
`requires_vectors` (hidden until `tree.diffraction_vectors` attaches),
`requires_package` (a module name or list of them — hidden until the optional
extra is installed; resolved with `find_spec`, so it never imports the package
and never pays its import cost. `install_hint(pkg)` gives the `pip install
"spyde[eels]"` line. Add BOTH filter paths' gates — see below),
`plot_dim` (`[1, 2]`), `navigation` (navigator vs signal window), `toggle`,
`submenu`/`subfunctions`, `parameters` (rendered by the Electron param panel;
supports `type`, `default`, `min`/`max`/`step`, `options`, `tab`,
`display_condition`, `file`).

**Path 2 — staged actions** (the wizard protocol). Registered in
[`registry.py`](registry.py) `STAGED_HANDLERS` as `"module.function"` (lazy
import), all with the uniform signature `fn(session, plot, payload)`.
`payload["window_id"]` is always injected so handlers can resolve bare-figure
windows.

Naming convention — `<key>` is the wizard's prefix:

    <key>_open           caret mounted → start live preview / controller
    <key>_close          caret unmounted → tear everything down
    <key>_tune           debounced live re-tune
    <key>_set_<param>    discrete parameter change
    <key>_run            heavy compute stage (may open a result tree)
    <key>_commit         snapshot the live result into a NEW SignalTree

Extra stages keep the prefix (`om_generate_library`, `vom_refine`, `czb_pick`).

## 3. The lifecycle

```
click ──► toolbar gate (plot_control_toolbar filters)
      ──► dispatch (Path 1 or 2)
      ──► worker (lifecycle.run_on_worker: daemon thread; UI apply marshalled
           back via session._dispatch_to_main — NEVER touch plots/figures from
           the worker; generation guard drops superseded runs)
      ──► result (commit.open_result_tree for progressive windows /
           lifecycle.paint_signal_plots / lifecycle.replace_tree_attr overlays)
      ──► Commit (commit.commit_result_tree: primary map + chip views +
           provenance)
      ──► teardown (Session._forget_window → controller.close() → figure
           eviction; tree.close() removes all interactive action state)
```

**Ownership map** — where state lives:

- **on the tree**: results (`diffraction_vectors`, `orientation_map`,
  `vector_orientation`), wizard controllers (`_om_wizard`, `_vom_wizard`,
  `_strain_controller`), overlays (`_vector_overlay`, `_fv_preview`, …),
  run generations (`_<key>_run_gen`), batch flags (`_fv_batch_running`).
  `BaseSignalTree.close()` tears all of it down.
- **on the Session**: `_action_artifacts` (RegionAction selectors/outputs),
  `_window_controllers` (bare-figure window controllers), `signal_trees`.
- **on the Plot**: `_vi_items` (VI chips).
- **module-level**: only `figure_registry._FIGS` (per-window keep-alive,
  evicted by `_forget_window`). Do NOT add module-level mutable state.

**The basis set** ([`lifecycle.py`](lifecycle.py)) — use these, never re-roll:
`run_on_worker`, `bump_generation`/`is_current`, `resolve_vectors`,
`wait_for_vectors`, `replace_tree_attr`, `paint_signal_plots`,
`progress_emitter`, `live_fill_poller`.

**The two tree-spawning doors** ([`commit.py`](commit.py)):
`open_result_tree` (window opens early, blank, progressively filled — FV count
map, OM live IPF) and `commit_result_tree` (data-ready snapshot — the Commit
action: primary map as the signal plot, extra maps as chip-selectable views,
locked symmetric levels for signed components, `attrs` on the tree,
provenance stamped on `tree._commit_provenance` +
`metadata.General.spyde_provenance`).

**The progressive window's SIGNAL half** ([`live_signal.py`](live_signal.py)):
an `open_result_tree` window that is a navigator **and** a signal plot only
half-fills — the navigator fills block by block while the signal plot sits on
its placeholder. `attach_signal_preview(session, tree, render=…,
nav_shape=…)` drives the signal plot from the same per-block results: each
landing block paints one deterministic sample position, and the
navigator→signal slice function is swapped for one that renders any
already-computed position on demand (an un-computed one returns `None`, so
the last good frame stays up). Feed it `preview.note_block(nav_slices)` from
the compute's per-chunk callback and `preview.close()` when the batch
finalizes — close never clobbers a final display installed in the meantime.
Returns `None` on a window with no navigator (the Orientation / EBSD IPF map
is a single 2-D plot); that is a documented no-op, not an error.

## 4. Adding an action, step by step

**TransformAction / RegionAction** (copy from `_template_action.py`):
1. Subclass in a new module under `spyde/actions/`.
2. Add the `toolbars.yaml` entry pointing `function:` at the class.
3. Test: `session._dispatch_toolbar_action(plot, "Name", params)` + assert on
   `messages` (see `test_template_actions.py`); a gating test via
   `get_toolbar_actions_for_plot` on a fake plot (see
   `test_vector_vvi_action.py::TestVectorVVIGating`).

**Wizard**:
1. Subclass `WizardController` (set `key`), write the staged handlers
   (`_template_action.py` shows the full open/close/commit set with every
   guard in place).
2. **Declare the parameter schema** — a `parameters` classattr on the
   controller (same dict spec as the YAML `parameters:` blocks) and an entry
   in `registry._WIZARD_SCHEMAS`, so any host (the Electron caret, a notebook
   form, `spyde.api` docs) resolves it via `registry.wizard_parameters(key)`.
   One source of truth; drift against handler `DEFAULTS` is caught by
   `test_wizard_schemas.py`.
3. Register the stages in `registry.STAGED_HANDLERS`.
3. Add the caret component on the renderer (see §5) and, if it's opened from
   a toolbar button, a `toolbars.yaml` entry whose `function` is a no-op
   parent (the Electron toolbar opens the wizard; see
   `vector_orientation_mapping`).
4. Tests: call the handlers directly (`fn(session, plot, payload)`) and poll
   with a `_wait(pred)` helper (see `test_find_vectors_wizard.py`); add a
   double-fire test (open, close, open → exactly one live controller — see
   `test_wizard_double_fire.py`).

**A long pass over a lazy dataset should STREAM, not block.** Build the lazy
graph, hand it to `ComputeBackend.compute_chunks_progressive(graph, nav_ndim,
on_chunk_done)`, and repaint from the partial array as chunks land (`dpc_action.
_measure_progressive` is a compact example; find-vectors is the large one). Two
things make it work: the graph must keep the dataset's own nav chunking, so a
streamed chunk is a STORAGE chunk (Live-Display §1) — check for `rechunk` layers
if unsure; and every stage downstream must tolerate `NaN` for not-yet-computed
positions, including the DISPLAY, or one unfilled pixel blanks the whole map.

**Compute-heavy actions**: keep the compute in its own module (or package —
`find_vectors/` is the model: pure compute layers + `__init__` re-export;
interactive wiring stays outside). NEVER `.compute()` the full dataset
(CLAUDE.md memory-safety rule). DPC is the small version of the same split:
`dpc.py` (pure physics) / `dpc_display.py` (figures) / `dpc_action.py`
(interaction). Read `dpc.py` before touching any beam-shift code — it documents
the two conventions pyxem does not make obvious (a shift is `centre − beam`,
i.e. the NEGATIVE of the deflection; and a colour wheel has to be 90° off its
map to read true) and derives its legend from the map's own colour rule rather
than copying that offset.

**Legends that float over a plot** — an IPF triangle, a DPC direction wheel —
are `Plot2D.add_key` overlays, NOT `Figure.add_inset`. An inset is a draggable
window with a title bar and its own canvas stack: right for a live sub-plot,
wrong for a legend, which should read as part of the figure the way a scale bar
does. Keys are also included in a PNG export. `hover_only=True` suits a legend
for a map that is readable without it (the IPF triangle); leave it False when
the map is meaningless without its key (the DPC hue wheel).

**Explanatory prose does not belong on a caret face.** Carets are ~240 px wide
and every paragraph pushes a control off the visible area. Put the sentence
behind an `ⓘ` — `DpcWizard.tsx`'s `Info` is a copyable one: a click-toggled
popover, absolutely positioned so it costs no layout, and opting out of the
`nowrap` that `Field` labels set.

## 5. The renderer side (wizard carets)

Shared pieces in `electron/src/renderer/src/components/`:
- `WizardShell.tsx` — chrome (header/tabs/status) + form controls.
- `wizardHooks.tsx` — the lifecycle:
  - `useWizardLifecycle({openAction, closeAction, …})` — mount→open,
    unmount→close. **StrictMode rule**: React dev-mode mounts every wizard
    twice synchronously (mount→cleanup→remount). The hook defers the open one
    tick so the first mount's open is cancelled; the backend's run/stop
    generation guard (`WizardController.guard()`/`cancel_inflight()`) is the
    belt-and-braces. Every wizard whose open spawns a worker MUST use both.
  - `useDebouncedAction` — the `<key>_tune` sender (cancelled on unmount).
  - `useWizardEvent` — `spyde:*` CustomEvents (re-broadcast from PLOTAPP
    messages in `SpyDEContext.tsx`), filtered per window.
  - `CommitButton` — sends `<key>_commit`; testid `<key>-commit`. Add it to
    any wizard whose controller implements `commit()`.
- Register the wizard name in `FloatingToolbar.tsx` `WIZARD_ACTIONS` and
  render it in the caret switch.

Toggle-state sync: the renderer highlights actions from `action_active`
messages only. The backend emits `active:false` on deselect, on window
teardown (`_forget_window`), and when a dispatched action raises.

## 6. Pitfalls (each of these was a real bug)

- **The vectors gap**: `tree.diffraction_vectors` attaches only when the
  find-vectors batch finalizes. A vector-dependent handler must
  `wait_for_vectors(…, strict=True)` (strict = only the clicked plot's tree
  counts — the any-tree fallback would re-dispatch forever into a
  tree-specific gate).
- **Two filter paths, one gate**: `get_toolbar_actions_for_plot` (resolves the
  function) and `_action_matches_plot` (used by `get_toolbar_config_for_plot`,
  imports nothing) apply the SAME gates in two places. Add a new gate key to
  both — one alone renders a button that never dispatches, or vice versa.
  `test_requires_package_gate.py` asserts this for `requires_package`.
- **Bare-figure windows**: a window emitted as a raw `figure` message is NOT
  a registered Plot — `_plot_by_window_id` returns None and generic dispatch
  silently no-ops. Register a controller (`own_window`) and resolve via
  `session.controller_by_window_id`; keep figures alive with
  `figure_registry.keep_alive(window_id, fig)`.
- **Thread marshal**: plots/figures/IPC state may only be touched on the
  asyncio main thread. Workers hand results to `on_done` (marshalled);
  `emit_status`/`emit_error` are safe from any thread.
- **Latest-wins**: any recompute that can be superseded needs a generation
  guard; teardown paths bump the generation FIRST.
- **Never materialise the dataset** — see the CLAUDE.md memory-safety rule.
- **Don't touch the live-display core** (NavDispatcher, shm display path,
  chunking) — see CLAUDE.md; actions call it, never restructure it.

## 7. Testing your action

**Start with `test_action_conformance.py` — it already covers your action.** It
is driven by the registry, not a hand-written list, so a new staged action is
checked the moment it registers. It enforces the seams that fail SILENTLY:
handlers resolve and take `(session, plot, payload)`; `_open` and `_close` come
as a pair; the schema is well formed and matches the module `DEFAULTS`; the
caret is in `WIZARD_ACTIONS`, rendered in the switch, and named in
toolbars.yaml; every `useWizardEvent('spyde:X')` has a re-broadcast case; caret
`DEFAULTS` match the backend's; and — at runtime — messages are addressed to the
CARET's window, open/close/open leaves one controller, and close leaves nothing
registered.

Add your wizard to `RUNTIME_FIXTURES` (a loader + an open payload) to get the
runtime contracts, or to `RUNTIME_EXEMPT` with a reason if opening it needs a
CIF/library/movie. `test_every_wizard_is_either_covered_or_exempt` fails if you
do neither, so the gap cannot go unnoticed.

When it fails for a new action, wire the seam it names — don't add an exemption.



- Fixtures: `spyde/tests/migrated/conftest.py` (`window`, `tem_2d_dataset`,
  `stem_4d_dataset`) — real Qt-free `Session`s with captured `messages`.
- Staged handlers: call directly, poll with `_wait(pred)`.
- Toolbar path: `session._dispatch_toolbar_action(plot, name, params)`.
- Gating: fake plot + `get_toolbar_actions_for_plot`.
- Threaded marshal: the `_LoopSession` harness (`test_lifecycle.py`,
  `test_strain_threaded.py`).
- GPU paths: force CPU in wiring tests (`monkeypatch gpu_available → False`);
  real GPU correctness runs in a subprocess (CLAUDE.md).
- **UI features are verified by RUNNING THE APP** (Playwright harness
  `electron/tests/_harness.cjs`) and looking at screenshots — headless tests
  + a clean typecheck are NOT verification (CLAUDE.md).
