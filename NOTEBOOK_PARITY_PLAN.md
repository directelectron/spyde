# Three-Host Action Parity — Design Plan (script ↔ Jupyter ↔ SpyDE)

Status: proposed (2026-07-02)
Owner: Carter Francis
Related: `spyde/actions/README.md` (the action framework this builds on),
`VECTOR_ORIENTATION_MAPPING_PLAN.md`, `DIFFRACTION_VECTORS_PLAN.md`

## 1. Goal & the definition of parity

Every scientific action should run **equally** from:

1. a plain Python **script** — `spyde.api.find_vectors(signal, …) → vectors`,
2. a **Jupyter notebook** — `nb.find_vectors(signal)` with a REAL interactive
   wizard (parameter form, live peak preview on an inline anyplotlib figure,
   Compute button) whose result is the same object the script gets,
3. **SpyDE** — the toolbar action, unchanged.

**Parity means: same handlers, same payloads, same protocol messages** — not
three parallel implementations. Concretely:

- The **compute contract** is one pure function per action, shared by all
  three hosts, returning a self-contained result object.
- The **wizard contract** is the existing staged-action protocol
  (`<key>_open/_tune/_set_*/_run/_commit/_close`, `spyde/actions/registry.py`)
  driven through `Session.dispatch_action` — the notebook sends the *same
  payloads* the React carets send, so generation guards, `wait_for_vectors`,
  param coercion, and every existing handler test apply verbatim.
- The **call shape** is unified: `action(signal, **params)`. Script returns
  the result object; the notebook additionally displays the live wizard UI;
  SpyDE additionally does window/tree bookkeeping. In Jupyter there is **no
  visible Session/SignalTree machinery** — signal in, widget/result out
  ("the tree is implied by Jupyter").

Out of scope for the first pass: view/window-management actions (zoom, tile,
signal-tree navigation) — SpyDE-only by nature. In scope (the scientific
core): **Find Diffraction Vectors, Orientation Mapping, Vector Orientation
Mapping, Strain Mapping, Virtual Imaging (incl. Vector VI), Center Zero
Beam.**

## 2. Background — what already exists (audited 2026-07-02)

The gap is much smaller than it looks; almost all of it is presentation.

**Compute cores are already headless.** Verified per action:

| Core | Signature (essentials) | Returns | Session use |
|---|---|---|---|
| `spyde/actions/strain_mapping.py::compute_strain_field` | `(vecs, ref_yx=None, *, ref_vectors=None, tol=None)` | `StrainField(exx,eyy,exy,omega,coverage)` | none — pure numpy/scipy |
| `spyde/actions/find_vectors/orchestrate.py::_do_compute_vectors` | `(signal, params, main_window, signal_tree, shm_name=None, …)` | `SpyDEDiffractionVectors` | `main_window`/`signal_tree` = dask-client lookup ONLY; `None` → `compute(scheduler="threads")` |
| `spyde/actions/orientation_compute.py::_do_compute_orientations` | `(signal, sim, params, main_window, signal_tree, …)` | `SpyDEOrientationMap` | same — client lookup only |
| `spyde/actions/vector_orientation.py::compute_vector_orientation` (+`_chunked`, `vector_orientation_gpu`) | `(vectors, lib, params, …)` | `VectorOrientationResult` | serial path fully pure; chunked = client lookup only |
| Center Zero Beam | pyxem `get_direct_beam_position` / `center_direct_beam` | signal | none |
| Virtual Imaging / Vector VI | mask×sum (`virtual_image.py`) / methods on `SpyDEDiffractionVectors` | array/signal | none |

**Result objects are self-contained.** `SpyDEDiffractionVectors`,
`SpyDEOrientationMap` (`spyde/signals/`), `VectorOrientationResult`,
`StrainField` import zero backend/drawing code, are constructible from
scripts, saveable, and carry their own numpy viz outputs (`.ipf_color_map()`,
`.strain_map()`, `.count_map()`, `.render_frame()`).

**A real headless `Session` already runs in every pytest**
(`spyde/tests/migrated/conftest.py` builds `Session(1,1)` under
`SPYDE_NO_DASK=1` with `ipc.emit` captured). The whole wizard test suite is
already "SpyDE in a kernel without Electron" — so the notebook host is a
**presentation-seam problem, not a re-implementation problem**.

**The Electron coupling is one idiom in four places**:
`anyplotlib._electron.register(fig)` + `finalize_figure_html(fig, fig_id)` +
`ipc.emit({"type":"figure", "html": …})` — in
`spyde/drawing/plots/plot.py` (`_ensure_figure`, `set_view_tag`),
`spyde/actions/views.py` (`emit_view_figure`, `emit_tiled_figure`),
`spyde/actions/strain_display.py` (`build_strain_figure` consumer), and the
IPF builders (`ipf_view.py`, `ipf_refine_render.py`, `ipf_density.py`).

**Dormant notebook affordances**: `Action.for_plot(plot)`
(`spyde/actions/action.py:57`) is the documented notebook entry point but is
used nowhere; `spyde/signals/__init__.py` is empty; `spyde/__init__.py`
exports only config dicts; there are no example notebooks. The
`Action.parameters` / `toolbars.yaml parameters:` schema explicitly promises
"a host (Electron panel or an ipywidgets form) can render an input form" —
never cashed.

**anyplotlib is host-agnostic already** (v0.1.0, PyPI-pinned, locked):
- ONE `anywidget.AnyWidget` subclass (`Figure`); everything else (Plot1D/2D/3D,
  widgets, markers) is plain Python mutating its synced traits.
- The front-end is a single **self-contained 263 KB `figure_esm.js`**
  (hand-written Canvas2D + native WebGPU; no three.js/d3/regl; **zero CDN or
  network fetches** — grep-verified).
- The interactive API SpyDE uses (crosshair/rect/circle/annular widgets,
  MarkerGroups, `add_event_handler`, scatter3d, pcolormesh) works identically
  in both hosts: Jupyter events flow over anywidget comms, Electron events
  over the standalone-HTML postMessage→PLOTAPP bridge, and **both converge on
  the same `Figure._dispatch_event`** — Python-side handlers fire identically.
- A bare `fig` displayed in a notebook cell already works today.

## 3. Architecture decision — real `Session` + a two-part host seam

**Decision: `NotebookSession(Session)` — a thin subclass of the real Session —
plus a host seam at the drawing/ipc layer.** The staged handlers,
`WizardController`, `lifecycle.py`, and `commit.py` run **byte-for-byte
unchanged**; only where figures and protocol messages go changes.

Rejected alternatives (for the record):

- *Duck-typed session + lightweight tree*: fails true parity — handlers reach
  into real internals: `StrainController._attach_reference_selector` calls
  `tree.navigator_plot_manager.add_navigation_selector_and_signal_plot(...)`;
  `find_vectors_action._install_render_display` rewrites live navigator
  selector children and stashes `tree._render_frame_fn`; CZB calls
  `tree.add_transformation` + `session._reemit_signal_tree`. A fake tree
  would end up re-implementing `BaseSignalTree`/`MultiplotManager`, forked.
- *A `Host` parameter injected under actions*: the actions are already
  host-agnostic given a Session; injecting a new parameter churns 20+ handler
  signatures for zero gain. The host concept is right but belongs at the two
  concrete Electron touchpoints (figure presentation, message stream).
- *Three thin frontends over pure functions only*: script parity but no
  wizard parity — no live tune loop, no reference-crosshair strain, no commit
  flow in the notebook. Out per the chosen ambition.

**The exact Session surface the actions package touches** (grep-verified —
everything inherited for free by a subclass): `dispatch_action`,
`_dispatch_to_main`, `_add_signal`, `add_plot_window`, `signal_trees`,
`_plots`, `next_window_id`, `register_window_controller` /
`controller_by_window_id`, `_forget_window`, `_close_signal_plot`,
`_reemit_signal_tree`, `_action_artifacts`, `dask_manager.client`.

**`NotebookSession` is internal, not the public face.** The Jupyter API is
**signal-first** — `action(signal)` — with no visible Session/SignalTree
machinery. A module-level singleton is created lazily on first use
(`spyde.notebook._session()`); each signal-first call does the
`_add_signal`/tree bookkeeping under the hood and keeps a signal→tree map so
repeated calls on the same signal reuse the same hidden tree.
`set_main_loop(asyncio.get_event_loop())` on the ipykernel loop makes
`_dispatch_to_main` marshal worker results onto the comm-processing thread —
the exact analogue of the Electron main thread, so the threading contract in
`lifecycle.py` holds without modification. Power-user escape hatches
(documented, not headline API): `spyde.notebook.session()`,
`spyde.notebook.close()` (shutdown + worker-thread teardown), and dask
configuration on first touch (`configure(dask="none"|"local"|Client)`).

## 4. The host seam (two parts)

### 4a. Figure presenter — new `spyde/drawing/host.py`

```python
def set_host(name: Literal["electron", "notebook", "headless"]) -> None
def get_host() -> DisplayHost

class DisplayHost(Protocol):
    def present_figure(self, fig, *, window_id, title, is_navigator=False,
                       aspect=None, view_label=None, view_kind=None,
                       extra: dict | None = None) -> str      # fig_id
    def close_window(self, window_id: int) -> None
```

- `ElectronHost.present_figure` = a pure extraction of today's
  register+finalize+emit idiom. Migrating the four call sites onto it is a
  behavior-neutral refactor that also deletes the duplication.
- `NotebookHost.present_figure` **skips `_electron.register` and HTML
  finalization entirely** — the `apl.Figure` *is* an anywidget; it records
  `fig_id → fig` and hands `(window_id, fig, meta)` to the
  `NotebookWindowManager` for display.
- `HeadlessHost` records metadata and presents nothing — today's test
  behavior made explicit (tests keep monkeypatching `emit` and keep passing).

### 4b. Message sink — `ipc.set_sink(fn)` in `spyde/backend/ipc.py`

Default sink = today's `PLOTAPP:` stdout writer. The notebook sink also
captures `anyplotlib._electron.emit` (the same patching pattern
`redirect_stray_stdout()` and the test conftest already use) and routes:

- `status` / `progress` / `error` → a status-bar widget / log output,
- `window_closed` → `NotebookWindowManager.remove(window_id)`,
- `fv_auto_params` → seed the wizard form's sliders (same round-trip the
  React caret does),
- `toolbar_config` / `action_active` / `histogram` / `metadata` → dropped
  (surfaceable later).

**The PLOTAPP protocol becomes the notebook event bus** — the notebook
frontend consumes the same messages the Electron renderer consumes,
in-process. That is the strongest parity guarantee available.

### 4c. Window-concept mapping

| Electron concept | Notebook realization |
|---|---|
| PlotWindow / bare-figure window | `FigurePanel` = `VBox([title, figure_or_tab])`, keyed by `window_id` in `NotebookWindowManager` |
| A tree's windows (nav + signal + outputs) | a panel container that grows live as windows open (VI output, strain map, strain-reference DP appear as panels) |
| chip views (`register_views` + `emit_view_figure`) | `present_figure` calls carrying `view_label` for an existing window → children of an `ipywidgets.Tab` inside that window's panel (one `apl.Figure` per view; arrays already stashed in `views._VIEW_DATA`) |
| ⌘-click tiled compare | `spyde.notebook.tile(...)` reusing the existing `views.build_tiled_figure` (linked crosshairs work — host-agnostic widgets) |
| `window_closed` | panel removed; existing `figure_registry` eviction unchanged (`_forget_window` already does it) |
| resize / `figure_event` routing | unnecessary — anywidget comms deliver events straight into `Figure._dispatch_event`; sizing via widget layout |

`Plot._ensure_figure` change is minimal: build fig+plots exactly as now, then
`self.fig_id = get_host().present_figure(...)`. `set_view_tag` likewise.

## 5. `spyde.api` — the script layer

Typed, documented wrappers over the existing cores; **`spyde.api` must never
import `spyde.backend` or `spyde.drawing`** (enforce with a trivial
import-graph test):

```python
find_vectors(signal, *, method="nxcorr", threshold=None, sigma=1.0,
             kernel_radius=5, min_distance=5, subpixel=True,
             dog_sigma1=0.8, dog_sigma2=2.0, beamstop_auto=False,
             client=None) -> SpyDEDiffractionVectors
orientation_map(signal, phases, *, resolution=1.0, accelerating_voltage=200.0,
                minimum_intensity=1e-4, n_best=5, gamma=1.0,
                client=None) -> SpyDEOrientationMap
vector_orientation_map(vectors, phases_or_library, *, gpu="auto",
                       strain_cap=0.05, **params) -> VectorOrientationResult
strain_map(vectors, *, ref_yx=None, ref_vectors=None, cif=None) -> StrainField
center_zero_beam(signal, *, method="center_of_mass", half_square_width=0,
                 plane_fit=False, inplace=False) -> Signal2D
virtual_image(signal, *, cx, cy, r=None, r_inner=None, kind="disk",
              calculation="sum") -> Signal2D
vector_virtual_image(vectors, *, cx, cy, r, kind="disk",
                     intensity_weighted=True) -> np.ndarray
```

Notes:
- `strain_map(cif=…)` routes through `_zero_beam_filtered` +
  `snap_reference_to_cif(… cif_g_families(Phase.from_cif(cif)))` — the same
  physics the wizard uses, so script and wizard agree numerically.
- Replace the `main_window=` shim on the two batch cores with an explicit
  `client=None` kwarg (short-circuits the lookup at
  `orchestrate.py` / `orientation_compute.py`); the in-app callers pass
  `client=session.dask_manager.client`.
- `spyde/signals/__init__.py` re-exports `SpyDEDiffractionVectors`,
  `SpyDEOrientationMap`; `spyde.api` re-exports those plus
  `VectorOrientationResult` and `StrainField`.
- **Provenance**: add an optional `provenance: dict | None` field to the four
  result dataclasses; every `spyde.api` function stamps
  `{"action": <name>, "params": {...}, "spyde_version": __version__}` — the
  same dict convention as `commit._stamp_provenance`, so scripted results and
  committed trees carry interchangeable records.

## 6. `spyde/notebook` — session, windows, forms, wizards

New package: `session.py` (the internal `NotebookSession` + singleton),
`windows.py` (`FigurePanel`, `NotebookWindowManager`), `forms.py`,
`wizards.py`, `__init__.py` (the public signal-first API).

### 6a. Forms from the existing parameter schema

```python
def build_form(schema: dict[str, dict], *, on_change, debounce_ms=200) -> FormWidget
```

Same dict spec as `toolbars.yaml parameters:` / `Action.parameters`:
`int→IntSlider(min/max/step)`, `float→FloatSlider`, `bool→Checkbox`,
`enum→Dropdown(options)`, `file→Text(path)` (FileUpload later), `tab→Tab`
grouping, `display_condition→observe + visibility`. Debounce mirrors the
renderer's `useDebouncedAction` (a restarted `threading.Timer`).
`Action._resolved_params` already implements the merge order, so the runtime
side exists.

**Schema-completion work item (real gap found by audit): Strain, Vector OM
and Center Zero Beam declare NO parameter schema** — their forms are
hard-coded in the React carets (`StrainWizard.tsx` etc.). Add `parameters`
classattrs to the wizard controllers (strain: component enum, method enum
region/cif, cif_path file, match_radius_px float; VOM: voltage/resolution/
min-intensity + refine sliders; CZB: method enum, half_square_width int,
make_flat_field bool, manual tab) as the single source of truth. FV and OM
reuse their existing YAML blocks (via `spyde.TOOLBAR_ACTIONS`). The Electron
carets can migrate onto the schema later but do not have to.

### 6b. The generic notebook wizard

```python
class NotebookWizard(ipywidgets.VBox):
    """form + status + Run/Commit buttons + adopted result panels."""
    # on display  -> session.dispatch_action({"action": f"{key}_open",
    #                "payload": {...}, "window_id": src_plot.window_id})
    # form change -> debounced <key>_tune / mapped <key>_set_* dispatch,
    #                payload IDENTICAL to the React caret's
    # Run/Commit   -> <key>_run / <key>_commit
    # close()      -> <key>_close
```

Everything goes through `session.dispatch_action` — the same entry as
Electron, so `payload["window_id"]` injection, `_coerce`, the run/stop
generation guards, and `wait_for_vectors` behave identically. Re-running a
wizard cell fires open twice without close — **the existing StrictMode
generation guards are exactly the right defense** (a direct reuse win).
Result windows created after `_open` (the strain map, the reference DP, the
FV result panels) are adopted into the wizard's own container so one cell
shows form + live figures.

### 6c. The public signal-first API (the headline)

```python
from spyde import notebook as nb

vecs = nb.find_vectors(sig)              # wizard: live peak preview on an
                                         # inline DP, tune sliders, Compute;
                                         # .result -> SpyDEDiffractionVectors
om   = nb.orientation_map(sig, cif=...)  # om_generate_library / om_refine / om_run
res  = nb.vector_orientation_map(vecs)   # vom_* stages
sf   = nb.strain(vecs_or_sig)            # runs Find Vectors first on a raw
                                         # signal (the in-app wait-for-vectors
                                         # flow); reference crosshair panel,
                                         # component toggle, Commit
ctr  = nb.center_zero_beam(sig)          # auto tab -> czb_run; manual -> czb_open/czb_pick
vi   = nb.virtual_image(sig, kind="disk")  # live ROI via Action.for_plot —
                                           # finally exercising the dormant entry
nb.show(sig)                             # plain browse: navigator + DP crosshair
```

Each call hides tree creation, displays its panels as the cell output, and
returns/resolves to the **same result objects `spyde.api` returns**, so
notebook code continues script-style afterwards. Form toolkit:
**ipywidgets** — anywidget subclasses `ipywidgets.DOMWidget` and already
requires it, so forms add zero Python deps and zero new JS (a pure-anywidget
form would mean authoring new JS, against the asset-minimization goal).

## 7. The three-way contract table

Headline: **the same call shape `action(signal, **params)` works in all three
hosts** — script returns the result object; notebook adds the live wizard UI;
SpyDE adds windows/tree bookkeeping.

| Action | Pure core (shared) | Script (`spyde.api`) | Notebook (`spyde.notebook`) | SpyDE (toolbar + staged verbs) | Shared machinery |
|---|---|---|---|---|---|
| Find Vectors | `_do_compute_vectors` | `api.find_vectors(sig, **p)` | `nb.find_vectors(sig)` — form←YAML schema, debounce→`fv_tune`, Compute→`fv_run`, `fv_auto_params`→sliders | YAML params panel; `fv_open/tune/run/close` | `_coerce`, preview overlay, `open_result_tree`, gen guards |
| Orientation Mapping | `_do_compute_orientations` (+ library gen) | `api.orientation_map(sig, phases, **p)` | `nb.orientation_map(sig, cif=…)` — stages→`om_generate_library/om_refine/om_run` | YAML params; `om_*` | library cache, live IPF shm fill, commit |
| Vector OM | `compute_vector_orientation(+_chunked/_gpu)` | `api.vector_orientation_map(vecs, lib)` | `nb.vector_orientation_map(vecs)` — `vom_*` stages | caret (schema to add, §6a); `vom_*` | `commit_result_tree` + chip views |
| Strain | `compute_strain_field` | `api.strain_map(vecs, ref_yx=…\|cif=…)` | `nb.strain(vecs)` — ref-crosshair panel, component dropdown→`strain_set_component`, method→`strain_set_method`, radius→`strain_set_match_radius`, Commit→`strain_commit` | caret (schema to add); `strain_*` | `StrainController` unchanged; `commit_result_tree` (εyy/εxy/ω as Tab views) |
| Virtual Imaging | mask×sum (`virtual_image.reduce`) | `api.virtual_image(sig, cx, cy, r, …)` | `nb.virtual_image(sig, …)` — live ROI on the inline figure via `Action.for_plot` | YAML subfunction params; RegionAction | `RegionAction.run/update_live_params`, selectors |
| Center Zero Beam | pyxem beam-centering methods | `api.center_zero_beam(sig, …)` | `nb.center_zero_beam(sig)` — auto→`czb_run`, manual crosshair→`czb_open`/`czb_pick` | caret (schema to add); `czb_*` | `tree.add_transformation`, `_display` re-slice |

Shared by every row: the result dataclasses + provenance dict,
`Session.dispatch_action`, the `lifecycle.py` basis set, the `commit.py`
doors, and the anyplotlib `Figure` (host-presented).

## 8. anywidget / JS-asset posture (verified feasibility)

1. **Nothing to slim in anyplotlib itself**: one dependency-free 263 KB ESM,
   no CDN, no runtime JS libraries. Displaying figures requires **no network
   access** in either host.
2. **anywidget does NOT dedupe `_esm`** (verified in anywidget 0.11.0):
   although `Figure._esm` is a class-level string, `AnyWidget.__init__` adds
   it as a **per-instance synced trait** — every figure ships its own 263 KB
   copy over the comm — and the frontend creates a fresh Blob URL per model,
   so V8's code cache never reuses the parse. This is the same drag SpyDE
   already fixed for Electron with the shared `file://` ESM
   (`plot.py::_shared_esm_url`).
3. **Pathlib `_esm` is NOT a size win** — anywidget still syncs the file
   *contents*; the only gain is dev hot-reload. Do not pursue it for size.
4. **The real dedupe lever: href `_esm`** — anywidget's loader
   short-circuits on a URL (`import(url)`) so the browser module cache
   dedupes across all figures and the comm carries only a URL. Proposal: an
   optional upstream anyplotlib feature (`apl.set_esm_url(...)`, served via
   the Jupyter server), **off by default** — the zero-network/air-gapped
   property is worth more than 263 KB × N figures for typical N.
5. **Notebook-file weight**: Jupyter's "Save Widget State" embeds the full
   `_esm` per figure into the `.ipynb` — document that it stays off for
   SpyDE notebooks.
6. **Packaging**: SpyDE gains NO Jupyter dependencies. Notebook support needs
   only what anyplotlib already pulls (`anywidget → ipywidgets, psygnal,
   typing-extensions`). Define a `spyde[jupyter]` extra (jupyterlab) purely
   as a convenience; the Electron installer never includes it. The ~1.5 MB of
   labextension JS (anywidget ~584 KB + ipywidgets ~909 KB) belongs to the
   user's local Jupyter install — served locally, cached, never CDN-fetched.
7. **Policy**: `figure_esm.js` stays single-file and dependency-free; any
   heavy future feature (e.g. volumetric 3-D) goes behind a second,
   lazily-loaded widget class rather than growing the shared bundle.

## 9. Implementation phasing (later sessions; each phase lands green)

1. **Script layer** — `spyde/api.py`, `spyde/signals/__init__` re-exports,
   provenance fields on the result dataclasses, the `client=` kwarg on the
   two batch cores. Lowest risk, immediately useful, unblocks everything.
2. **Host seam refactor** (Electron-neutral) — `spyde/drawing/host.py` +
   `ipc.set_sink`, migrate the four register+finalize+emit sites. The
   Playwright suite is the regression gate.
3. **NotebookSession + window manager** — the hidden singleton, panels,
   `nb.show(signal)` browse (inline navigation crosshair working).
4. **Exemplar wizards** — **Strain first** (it exercises everything:
   controller registry, bare-figure window→panel, additive reference
   selector→second panel, `set_*` verbs, worker marshal + gen guards,
   `commit_result_tree`→Tab chip views), then **Find Vectors** (adds the live
   tune/debounce loop, `open_result_tree` progressive fill, and the
   `fv_auto_params` round-trip), then OM, VOM, CZB.
5. **Polish** — example notebooks in `examples/`, the `[jupyter]` extra,
   docs page, optional href-ESM upstream work in anyplotlib.

## 10. Testing strategy per host

- **Headless (pytest)**: the existing handler tests already ARE the parity
  tests (real Session, same dispatch). Add NotebookSession variants of the
  conftest fixtures — ipywidgets/anywidget widgets construct fine without a
  browser (comms simply never open), so wizard tests set form values and
  `_wait(pred)` on tree state, exactly like today's staged-handler tests.
- **Notebook smoke (CI)**: `nbclient` executes one example notebook
  kernel-side (no front-end needed).
- **Electron**: the existing Playwright suite gates the Phase-2 seam refactor
  (`tests/README.md` documents the tiers).
- **Import-graph test**: `spyde.api` imports neither `spyde.backend` nor
  `spyde.drawing`.

## 11. Risks & open questions

- **Kernel-loop marshaling**: long-running cells delay `_dispatch_to_main`
  applies (the analogue of a busy Electron main thread). Document;
  `run_on_worker`'s inline fallback covers loop-less contexts.
- **Worker-thread lifetime in a kernel**: `PlotUpdateWorker`'s 5 ms poll
  thread needs `nb.close()`/`atexit` teardown.
- **stdout hygiene**: any emitter bypassing the sink would print `PLOTAPP:`
  lines into cell outputs — reuse the `redirect_stray_stdout` patching
  pattern and audit with a smoke notebook.
- **Re-running wizard cells** fires `_open` twice without `_close` — covered
  by the existing run/stop generation guards (the StrictMode defense).
- **Dask client ownership**: user-provided client vs an internal
  LocalCluster — resolved by `configure(dask=…)`; default `"none"`
  (threaded) keeps notebooks dependency- and process-light.
- **Windows SharedMemory names** across two kernels: names already include
  `id(plot)` — low risk, note it.
- **The frozen-timer pathology is Electron-spawn-specific** (see the
  2026-07-02 investigation): a Jupyter kernel is a normal console-class
  process and is unaffected; the diagnostics (`dump_dask_state`,
  `[fv-batch]` timings) work there too if ever needed.
- Open: how `nb.strain(raw_signal)`'s implicit Find Vectors surfaces its
  parameters (accept defaults + a "re-tune" affordance vs chaining two
  wizards); whether `nb.show` should support the 5-D stack navigator on day
  one; CIF file pickers in a browser-notebook context (path Text first,
  FileUpload later).
