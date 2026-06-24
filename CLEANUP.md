# SpyDE cleanup runbook — fully remove Qt + kill error-hiding `except`s

A **mechanical, verifiable** plan to (1) delete the now-dead Qt/pyqtgraph layer,
(2) drop the Qt dependencies, and (3) eliminate the `except: pass` anti-pattern.
Each step has an exact command, a verification gate, and a rollback. Work in
**small commits** (one logical step each) so any regression is caught immediately
and `git revert`/`git reset` is trivial.

> Status legend used below: ✅ proven this session · ⚠️ verify before acting.

---

## 0. Why this is safe (proven facts)

The Electron app's Python backend is launched as `uv run python -m spyde`
(`electron/src/main/runner.ts`) → `spyde/__main__.py:main()` → `spyde.backend.app.run`.

These were **measured** (not assumed) this session:

- ✅ Importing `spyde.backend.session` + `spyde.backend.app` pulls **zero**
  `PySide6` / `pyqtgraph` modules.
- ✅ Importing the **entire live action + figure surface** (all 7 staged-handler
  modules + the IPF/figure path + `spyde.drawing.plots.plot`) pulls **zero** Qt.
- ✅ The only real importers of `spyde.actions.pyxem` are `line_profile.py` and
  `vector_orientation_action.py` — both themselves dead. Nothing live imports it.
- ✅ `spyde/backend/app.py` and `spyde/backend/session.py` mention Qt only in
  **comments/docstrings** (no import). `vector_orientation_gpu.py` likewise
  (a comment about `QApplication.processEvents`). Leave these as-is.

**Conclusion:** because the live surface is Qt-free, *no live module imports any
Qt module*. Therefore every file that imports `PySide6`/`pyqtgraph` is dead with
respect to the Electron app — **with one exception**: `find_vectors.py` is live
but lazy-imports Qt inside a few legacy widget functions (trim those, keep the
module). See §4.

The 7 live staged-handler modules (from `spyde/backend/session.py`):
`center_zero_beam`, `composition`, `find_vectors_action`, `ipf_view`,
`orientation_action`, `vector_orientation_om`, `views`.

---

## 1. Safety harness — run after EVERY step

Define these three invariants. A step is only "done" when all three pass.

**INV-1 — live surface stays Qt-free** (catches an accidental live coupling):

```bash
uv run python - <<'PY'
import sys, importlib
for m in ['spyde.backend.app','spyde.backend.session','spyde.actions.center_zero_beam',
          'spyde.actions.composition','spyde.actions.find_vectors_action','spyde.actions.ipf_view',
          'spyde.actions.orientation_action','spyde.actions.vector_orientation_om',
          'spyde.actions.views','spyde.actions.ipf_density','spyde.actions.ipf_refine',
          'spyde.actions.ipf_refine_render','spyde.actions.orientation_compute',
          'spyde.actions.find_vectors','spyde.actions.vector_overlay',
          'spyde.actions.find_vectors_torch','spyde.drawing.plots.plot']:
    importlib.import_module(m)
qt=[m for m in sys.modules if 'PySide6' in m or m=='pyqtgraph' or m.startswith('pyqtgraph.')]
assert not qt, f'LIVE SURFACE PULLED QT: {qt[:6]}'
print('INV-1 OK: live surface Qt-free')
PY
```

> Note: the harness intentionally uses a heredoc with `uv run python`. If a
> tooling hook blocks heredocs, paste the body into `scripts/_inv1.py` and run
> `uv run python scripts/_inv1.py` instead.

**INV-2 — the Qt-free test suite is green** (THE liveness oracle — if a deletion
broke something live, a migrated test goes red):

```bash
uv run pytest spyde/tests/migrated/ -q -p no:cacheprovider   # ~2–3 min, must be all-pass
```

> ⚠️ Live actions have **two** entry points, not one — trust INV-2 over static
> grep: (a) the 7 string handlers in `session.py`; (b) the **YAML-wired actions**
> in `spyde/toolbars.yaml`, `spyde/actions/hyper_signal_actions/*.yaml`,
> `spyde/actions/plot_actions/*.yaml` (modules: `base`, `center_zero_beam`,
> `fft_action`, `find_vectors_action`, `line_profile_action`, `orientation_action`,
> `vector_orientation_om`, `vector_virtual_imaging`, `virtual_image`). A module
> reachable from either — including via a **lazy** import inside an action that
> only fires at runtime — is LIVE even though INV-1 (import-only) shows it Qt-free.
> This is how `line_profile` was caught.

**INV-3 — the backend boots** (import the entry, don't spawn a UI):

```bash
uv run python -c "import spyde.__main__; import spyde.backend.app; print('INV-3 OK')"
```

**E2E smoke (run at phase boundaries, not every step — it's heavy, one at a time):**

```bash
cd electron && npm run build && npx playwright test tests/orientation_lazy.spec.ts \
  tests/spyde.spec.ts --workers=1 --reporter=line
```

**Baseline:** before starting, run INV-1/2/3 + the E2E smoke and record that they
pass. If the baseline isn't green, fix that first — do not start deleting.

**Branch:** `git checkout -b chore/remove-qt` off `main`. One step = one commit.

---

## 2. The KEEP set (never delete)

- `spyde/backend/**`, `spyde/signals/**`, `spyde/workers/**`
- `spyde/__init__.py`, `spyde/__main__.py`
- `spyde/actions/**` **except** the dead Qt actions in §3 (and trim §4)
- `spyde/drawing/plots/plot.py`, `spyde/drawing/selectors/**`,
  `spyde/drawing/update_functions.py`, `spyde/drawing/colormaps.py`,
  `spyde/drawing/__init__.py`
- `spyde/*.yaml`, `spyde/actions/**/*.yaml`
- `spyde/tests/migrated/**`, `spyde/conftest.py`, `spyde/qt/shared.py`'s
  replacement helpers ⚠️ (see §5 — `shared.py` itself is Qt; check whether any
  migrated test imports `open_window`/`create_data`/`wait_until` from it and, if
  so, move those Qt-free helpers into `spyde/tests/migrated/_helpers.py` first).

Everything in §3 is dead and goes.

---

## 3. Delete the dead Qt packages (the bulk of the work)

These all import `PySide6`/`pyqtgraph` and nothing live imports them. Delete in
the batches below; after **each batch** run INV-1, INV-2, INV-3.

**Batch 3a — pure Qt UI packages (zero live refs):**

```bash
git rm -r spyde/qt spyde/live spyde/misc spyde/external/pyqtgraph spyde/external/qt
```
(If `spyde/external/` is now empty except `__init__.py`, remove it too.)

**Batch 3b — the legacy Qt MainWindow + scratch + Qt-only top-level modules:**

```bash
git rm spyde/_qt_main_legacy.py spyde/qt_scrapper.py spyde/_conftest_legacy.py \
       spyde/dock_manager.py
```
> ✅ Verified: `dock_manager.py` is imported only by the two legacy files (dead).
> ⚠️ Do **NOT** delete `spyde/mdi_manager.py` or `spyde/metadata_extract.py` —
> both are **live and Qt-free**. `session.__init__` instantiates `MDIManager`
> (the Qt-free window abstraction; `PlotWindow` *replaces* `QMdiSubWindow`, it is
> not Qt), and `session` calls `metadata_extract.build_metadata_dict/_axes_list`
> for the dock. (`metadata_extract`'s only "PySide6" is a docstring; importing it
> pulls no Qt.)

**Batch 3c — the Qt drawing layer (toolbars + presenter):**

```bash
git rm spyde/drawing/toolbars/caret_group.py spyde/drawing/toolbars/toolbar.py \
       spyde/drawing/toolbars/popout_toolbar.py spyde/drawing/toolbars/stylized_toolbar.py \
       spyde/drawing/toolbars/floating_button_trees.py spyde/drawing/toolbars/utils.py
git rm spyde/drawing/signal_tree_presenter.py
```
> ⚠️ **`spyde/drawing/toolbars/` is NOT all dead.** Do **NOT** `git rm -r` the
> whole dir — it also holds LIVE, Qt-free assets that the running app needs:
> `plot_control_toolbar.py` (`get_toolbar_config_for_plot` — builds the per-window
> toolbar action list emitted by `plot_states._send_toolbar_config`; without it
> EVERY toolbar action button vanishes), `__init__.py`, and `icons/*.svg`
> (referenced by `toolbars.yaml`). Delete only the six Qt modules above. (Both
> were over-deleted once and restored — neither headless nor the smoke E2Es catch
> it, because migrated tests don't render the toolbar and the smoke specs don't
> click toolbar action buttons. A `vector_*_lazy` / `strain_lazy` E2E that clicks
> an `action-btn-*` is the guard.)
⚠️ Then verify the rest of `spyde/drawing/plots/` is Qt-free and live:
`grep -rn "PySide6\|pyqtgraph\|QMdiSubWindow\|QWidget" spyde/drawing/plots/`.
If `plot_window.py` / `multiplot_manager.py` / `plot_states.py` import Qt **and**
no migrated test imports them, `git rm` them too; if a migrated test imports
them, they're live — keep and open a follow-up to de-Qt them. (INV-2 will tell
you: delete, run it, and if a test errors on import, revert that one file.)

**Batch 3d — the dead Qt action modules:**

```bash
git rm spyde/actions/vector_orientation_action.py   # the dead pyqtgraph caret
```
> ✅ Verified dead: `vector_orientation_action` is only named in comments; the
> live path is `vector_orientation_om.py`.
> ⚠️ **`pyxem.py`, `line_profile.py`, `line_profile_action.py` are NOT dead.**
> `line_profile_action` is wired in `spyde/toolbars.yaml` (a LIVE entry point —
> see §1) and lazy-imports the Qt `line_profile.py`, which imports
> `pyxem._start_progress_poll` + pyqtgraph + PySide6. Deleting them turns
> `test_template_actions::test_line_profile_opens_output_window` red. These are
> **Phase-6 de-Qt targets** (§4b), not deletions.

After 3a–3d: run INV-1/2/3, then the **E2E smoke**. Commit each batch separately,
e.g. `git commit -m "chore(qt): remove dead Qt UI packages (qt/ live/ misc/ external/)"`.

> Rollback for any batch: `git restore --staged --worktree <paths>` (before
> commit) or `git revert <sha>` (after).

---

## 4. Trim the dead Qt tail of `find_vectors.py` (live module)

`find_vectors.py` is **live** (the Find-Vectors compute) but its tail
(≈ lines 2780→EOF) is the legacy Qt overlay widget — it lazy-imports
`pyqtgraph` / `PySide6` inside functions, so it doesn't pull Qt at import.

1. Identify the Qt functions: `grep -n "pyqtgraph\|PySide6\|QtCore\|CircleROI\|ScatterPlotItem\|ImageItem" spyde/actions/find_vectors.py`.
2. For each such function, confirm **no live caller**:
   `grep -rn "<func_name>" spyde --include=*.py | grep -v /tests/` (and check it
   isn't referenced by a YAML action or a session handler). The Electron overlay
   is `spyde/actions/vector_overlay.py` (Qt-free) — the Qt versions here are the
   old pyqtgraph ones.
3. Delete those functions. Keep everything above the Qt tail (the compute).
4. Gate: `grep -c "pyqtgraph\|PySide6" spyde/actions/find_vectors.py` → **0**.
   Run INV-1/2/3 + `uv run pytest spyde/tests/migrated/test_find_vectors_port.py -q`.

Repeat the same liveness check for any other module flagged by
`grep -rln "PySide6\|pyqtgraph" spyde/actions --include=*.py` that turns out to be
live-with-lazy-Qt.

### 4b. De-Qt the live line-profile action (`line_profile.py` + `pyxem.py`)

`line_profile_action.py` (live, YAML-wired) lazy-imports `line_profile.py`, which
is a **pyqtgraph LineROI widget** that also pulls `pyxem._start_progress_poll` and
`spyde.qt.compute_status_indicator`. This is the last live action still on Qt.

1. **Extract the Qt-free helper:** `pyxem.py` is ~3.2k lines of mostly-dead Qt
   UI; the only thing live code needs from it is `_start_progress_poll`. Move that
   (and any pure-compute helpers `line_profile.py` uses) into a Qt-free module
   (e.g. `spyde/actions/_progress.py`), repoint the import. Confirm with
   `uv run python -c "import sys,importlib; importlib.import_module('spyde.actions.line_profile_action'); ..."`
   style INV-1.
2. **Port the line-profile UI to the action template** the way the other actions
   were (Electron/anyplotlib `RegionAction`), so `line_profile.py` no longer needs
   pyqtgraph/PySide6/`spyde.qt`. The migrated test
   `test_template_actions::test_line_profile_opens_output_window` is the gate.
3. Once `grep -c "PySide6\|pyqtgraph\|spyde.qt" spyde/actions/line_profile.py` → 0
   and `find_vectors.py` is trimmed (§4), **`pyxem.py` is fully dead** — delete it
   — and **`spyde/qt/` has no importer** — delete it:
   ```bash
   git rm spyde/actions/pyxem.py
   git rm -r spyde/qt
   ```
   Gate: INV-1/2/3 + E2E + `grep -rn "PySide6\|pyqtgraph" spyde --include=*.py | grep -v /tests/` → empty.

---

## 5. Remove the legacy `pytest-qt` test suite

`spyde/tests/` (root, ~28 files) tests the **old Qt app** (`qtbot`, the legacy
`MainWindow`, `from spyde.qt …`). The Qt-free suite is `spyde/tests/migrated/`.

1. List them: `grep -rln "PySide6\|qtbot\|_qt_main_legacy\|from spyde.qt\|MainWindow\|_conftest_legacy" spyde/tests/*.py`.
2. ⚠️ **Before deleting**, rescue anything still referenced by migrated tests:
   - `grep -rn "from spyde.qt.shared import\|import spyde.qt.shared" spyde/tests/migrated/`
     — if migrated tests use `open_window`/`create_data`/`wait_until`, copy those
     **Qt-free** helpers into `spyde/tests/migrated/_helpers.py` and repoint imports.
   - Keep data fixtures used by migrated tests (e.g. `Silver__0011135.cif`,
     `*.hspy` test inputs) — `grep -rn "Silver__0011135\|<fixture>" spyde/tests/migrated/`.
3. `git rm` the legacy test files (and `spyde/tests/<legacy>conftest*.py` that only
   serves them). Run INV-2 (now only migrated runs) — must stay green.
4. `git rm` any now-orphaned legacy fixtures not referenced by migrated tests.

---

## 6. Drop the Qt dependencies (the acceptance that Qt is gone)

1. Edit `pyproject.toml`: remove `PySide6` and `pyqtgraph` from `dependencies`
   (and any Qt entry in `[project.optional-dependencies]`/`[tool.*]`). Also remove
   stale Qt packaging if unused: the root `spyde.spec` (PyInstaller) and any
   `[tool.pycrucible]` block — confirm the Electron build (`npm run build`) is the
   only shipping path first.
2. Recreate the environment **without** Qt to prove nothing needs it:
   ```bash
   uv sync --reinstall            # or: uv lock && uv sync
   uv pip list | grep -iE "pyside6|pyqtgraph"   # must print NOTHING
   ```
3. Gate: INV-1/2/3 + full E2E. If anything imports Qt now, it will fail loudly —
   that's the point. Fix by porting (rare) or deleting the offending import.
4. Final Qt grep — must be **empty**:
   ```bash
   grep -rn "PySide6\|import pyqtgraph\|from pyqtgraph" spyde --include=*.py | grep -v /tests/
   ```

---

## 7. Eliminate error-hiding `except`s

> **STATUS: DONE (2026-06).** All silent `except … : pass` in live source are
> gone — **176 → 0** (AST finder below = 0). Converted module-by-module, one
> commit each, compute/data-path modules first and UI-glue last, exactly as the
> process prescribes. Every handler now either logs (`log.debug`, or
> `log.warning`/`log.exception` where a swallow would otherwise erase a
> user-visible failure — e.g. `write_shared_array`, the threaded navigator load),
> let-raises, or is a narrow membership guard (the `except ValueError: pass` on
> `list.remove` became `if x in lst`). Two latent bugs surfaced and were handled:
>
> - **`log` vs `logger` shadow** — `ipf_density.build_ipf_density_figure` takes a
>   `log: bool` param that shadows a module-level `log`, so `log.debug(...)` would
>   crash *only when the except fired*. Module loggers in plotting modules are now
>   named `logger`. Guard added: `tools/scan_logger_shadow.py` (AST check, 0 hits). See
>   the `logger-name-shadow` memory.
> - **anyplotlib `Axes.set_title` doesn't exist** — several multi-panel titles
>   (`ipf_density`, `ipf_refine_render`, `views`) were wrapped in `except: pass`
>   and so *silently never rendered*. Now logged; the real fix is to add
>   `set_title` to anyplotlib (tracked in §9).

228 of 457 `except` handlers (49%) are silent `except … : pass` — they hide real
failures. Fix systematically; **never** leave a bare swallow.

**Find them (AST-accurate, prints file:line):**

```bash
uv run python - <<'PY'
import ast, pathlib
for p in pathlib.Path('spyde').rglob('*.py'):
    if '/tests/' in str(p): continue
    try: t = ast.parse(p.read_text())
    except SyntaxError: continue
    for n in ast.walk(t):
        if isinstance(n, ast.ExceptHandler) and len(n.body)==1 and isinstance(n.body[0], ast.Pass):
            typ = ast.unparse(n.type) if n.type else 'BARE'
            print(f'{p}:{n.lineno}: except {typ}: pass')
PY
```

**Remediation rule (apply per occurrence):**

1. **Never** `except:` or `except BaseException:` — narrow to the specific
   exception(s) actually expected.
2. **Never** `except Exception: pass`. Choose one:
   - *Genuinely optional / best-effort* (e.g. a cosmetic UI nicety, an optional
     metadata field): keep going but **log it** —
     `except SpecificError as e: log.debug("…: %s", e)` (module-level
     `log = logging.getLogger(__name__)`). It must be traceable.
   - *Could mask a real bug* (compute, data, anything in a hot/analysis path):
     **let it raise** — delete the `try/except`, or re-raise after logging.
   - *Control flow* (`except ImportError`, `except KeyError` with a default,
     `except (FileNotFoundError, ...)`): keep, but narrowed and commented with
     *why* it's safe to continue.
3. Add `import logging` + `log = logging.getLogger(__name__)` to any module that
   gains a logged handler.

**Process:** one module per commit (start with the live compute modules —
`orientation_compute.py`, `ipf_refine.py`, `find_vectors*.py`, `vector_*` — where
hidden errors are most dangerous; UI-glue modules last). After each module:
`uv run pytest spyde/tests/migrated/ -q` + INV-1. Re-run the finder; the count
must monotonically drop. Target: **0** silent `except: pass` in non-test code,
and `grep -rn "except:" spyde --include=*.py` (bare) = 0.

> While here: the audit also found **142 bare `print()`** in non-test source
> (`grep -rn "^\s*print(" spyde --include=*.py | grep -v /tests/`). Convert to
> `logging` in the same per-module passes (optional but recommended).

---

## 8. Acceptance criteria (the cleanup is "done" when ALL hold)

- [x] `grep -rn "PySide6\|import pyqtgraph\|from pyqtgraph" spyde --include=*.py | grep -v /tests/` → only 2 *comments*, no imports *(done 2026-06)*
- [x] `uv pip list | grep -iE "pyside6|pyqtgraph"` → **empty** (deps removed) *(done 2026-06)*
- [x] AST finder (§7) → **0** silent `except: pass`; `grep -rn "except:" spyde --include=*.py` → **0** bare *(done 2026-06)*
- [x] INV-1, INV-2 (320 tests), INV-3 all green *(done 2026-06)*
- [~] E2E: migrated suite (320) + `app_log` spec green this round; re-run the full Playwright set (`orientation_lazy`/`spyde`/`ipf_*`/`composition`) before release
- [x] `spyde/tests/` contains only the migrated (Qt-free) suite — legacy retired *(done 2026-06)*
- [x] App still launches (Playwright launches the real backend each run) *(done 2026-06)*

---

## 9. Deferred / related follow-ups (track separately, not blockers)

- **Port the IPF-refine panel off matplotlib** (`ipf_refine_render.py` still uses
  a matplotlib-Agg raster). The anyplotlib prerequisite is already merged
  (1-D / PlotXY `double_click` now reports `xdata`/`ydata`). Plan: `PlotXY` +
  `pcolormesh(clip_path=…)` (mirror `ipf_density.py`) with live `marker.set`
  repaint per navigator move; the double-click mask uses the new data-coord event.
- **Split the 3.8k-line `find_vectors.py`** once its Qt tail is gone (compute core
  vs action glue).
- **Broader dead-code sweep** (non-Qt): use INV-2 + E2E as the oracle — delete a
  candidate, run the suite, keep iff green. Do this *after* Qt removal so the dead
  Qt files don't confuse the graph.
- **Distribution story** *(resolved 2026-06)*: the locked decision in
  `DISTRIBUTION_PLAN.md` is **PyCrucible + uv** (self-extracting exe with embedded
  uv) as the portable/offline path, with a uv-managed installer as the primary.
  So pycrucible config stays. Deleted the genuinely-stale **PyInstaller**
  `spyde.spec` (untracked, referenced the removed `spyde.qt` icons, used by
  nothing) and gitignored `*.spec` so it can't recur. The remaining distribution
  work is the phased plan in `DISTRIBUTION_PLAN.md` (installer / GPU-readiness /
  auto-update), tracked there, not here.
