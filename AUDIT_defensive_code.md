# SpyDE backend audit — bug-hiding except / dead legacy / over-defensive getattr

**Scope:** `spyde/` Python backend (not tests, not electron, not anyplotlib), extra depth on `spyde/actions/report/*.py`. **Read-only** — no edits applied. Every CONFIRMED finding below was re-verified against source after the auditors reported.

**Confidence tags:** `CONFIRMED` = I traced it to source and the smell holds. `SUSPECTED` = real but lower blast-radius or more defensible.

---

## TL;DR — what's actually worth fixing

The backend is in good shape on the two things you were most worried about. There is **no dual-format serialization, no `_migrate_*` shims, no old-vs-new parse branch** anywhere — the report format is a single never-yet-shipped `SCHEMA_VERSION = 1`. And the report's *top-level* save/open/export all fail loud (`emit_error`). The genuine issues cluster into three buckets:

1. **Two real data-loss swallows in the report load/save path** (P1) — a malformed field silently voids a whole figure spec on open; a bake failure silently drops a figure's pixels from a "successful" save. Both are `except Exception` + `log.debug`/silent in a path where the failure means corruption.
2. **~25 `getattr(cell/spec/panel/doc, 'field', default)` on guaranteed dataclass fields** (P2) — redundant defensiveness that would mask a "this isn't a Cell" bug instead of raising. Directly in line with the split-cell design goal: let a wrong object fail loud.
3. **A pile of dead Qt-migration vestiges** (P3, mechanical) — no-op stub methods/classes/aliases from the pyqtgraph era, zero callers. Safe deletions, mostly in `spyde/drawing/selectors/`.

Broad `except Exception` scope is **mostly correct-by-design** here (worker boundaries, best-effort emits, anyplotlib build). Only a handful have a clearly-narrower intended error type.

---

## P1 — Bug-hiding swallows in the report data path (fix these)

### 1. CONFIRMED — one malformed field silently voids a whole figure spec on open
`spyde/actions/report/model.py:1547-1551` and `:1576-1579` (both in `read_report`)
```python
try:
    c.spec = FigureSpec.from_yaml(zf.read(spec_name).decode("utf-8"))
except Exception:
    c.spec = None
```
`FigureSpec.from_yaml → from_dict → PanelSpec.from_dict → LayerSpec.from_dict`, and **none of the inner parsers has a try/except**. `LayerSpec.from_dict` does `float(d.get("alpha", 1.0))` / `float(linewidth)`; `PanelSpec.from_dict` does `list(...)`/`float(...)`. So one bad scalar in a `figures/<id>.yaml` (a hand-edit, a truncated write, `alpha: "--"`) raises `ValueError`/`TypeError` — and this catch throws away the **entire** `FigureSpec`, not just the bad field.

**What it hides:** the loud `emit_error` in `report_open` (handlers.py:1832) is never reached — the exception was already eaten *inside* `read_report`. The cell is pushed onto `_offline` (handlers.py:1861 skips it since `c.spec is None`), the user sees the baked PNG with **no indication their spec was corrupted**, and edit/refresh/re-slice are silently dead for that figure forever. On the next save the spec is gone from disk. If there's no baked PNG either, the figure is fully lost — with not even a log line (this is a silent `pass`-equivalent: no `log` call at all).

**Fix:** (a) narrow to `except (yaml.YAMLError, ValueError, TypeError, KeyError)` so a real bug in `from_dict` (a typo'd attr → `AttributeError`) surfaces instead of masquerading as "corrupt file"; (b) at minimum `log.warning` with the cell id + exception; (c) ideally mark the cell "spec-parse-failed" and surface a per-cell warning so the user knows that figure lost its editability.

### 2. CONFIRMED — a bake failure silently drops a figure's pixels from a "successful" save
`spyde/actions/report/handlers.py:797-803` (`assemble_assets`)
```python
if arr is not None and c.spec is not None:
    try:
        png = _bake_primary_snapshot(c, arr, max_edge=1200)
    except Exception as e:
        log.debug("asset bake failed for cell %s: %s", c.id, e)
if not png:
    png = self._baked.get(c.id)
```
On save/export, when the renderer harvest returned no PNG (headless save, slow/absent renderer reply past the 3 s timeout, unmounted window), this bakes the held snapshot. If `_bake_primary_snapshot` raises (bad dtype/shape, an mpl/agg error, a missing colormap the spec references) it's swallowed to `log.debug`; if `_baked` also has nothing (a freshly-added figure never harvested), `png` stays falsy and the cell is **omitted from `assets`** — while `write_report` still writes an `assets/<id>.png` image ref into `report.md`. Result: a **dangling image ref** on next open, the figure's pixels silently lost, and the save reports success (`report_saved`).

The code even comments the dangling-ref hazard for the empty-harvest case (handlers.py:787-790), but the bake-failure branch re-opens it.

**Fix:** keep the fallback chain, but if the final `png` is still empty for a non-scene3d figure cell, collect those cell ids and `emit_error`/warn ("Report saved, but N figures could not be rendered and were omitted") instead of silently writing dangling refs. At minimum raise the bake failure from `log.debug` to `log.warning`.

### 3. SUSPECTED — `count_map()` failure → blank-black navigator in the embedded vectors explorer
`spyde/actions/report/vectors_embed.py:214-218` — `except Exception → cm = np.zeros(...)`. If the CSR count-map aggregation raises, the embedded explorer renders a blank navigator the user may read as "no vectors here" rather than "the count map crashed." Degraded-not-corrupt. **Fix:** narrow the except, `log.warning`, and ideally mark "count map unavailable." Low priority.

### 4. SUSPECTED — a 3-D IPF panel silently vanishes on render failure
`spyde/actions/report/figure_builder.py:334-337` — `except Exception → return None`, and the caller skips a None panel. scene3d is genuinely fragile (Agg can't render 3-D, hence the separate path), so this is a defensible best-effort, but a panel the user asked for disappears invisibly. **Fix:** keep the fallback, raise the log to `warning`, ideally draw a "3-D panel unavailable" placeholder tile. Borderline.

### 5. SUSPECTED (minor) — example-data calibration silently wrong
`spyde/backend/_session_files.py:92-97` — applying the hardcoded scale/offset for a built-in example dataset under `except Exception → log.debug`. Scoped to example data (known dict, axes known to exist), so a failure here is really a regression being hidden. `except Exception` is too broad for a should-always-succeed op. Same file: the metadata/axis carry-over swallows at `:441, :562, :570` silently degrade calibration on stacked/reshaped derived signals (`log.debug` best-effort, primary data still loads — lower severity). **Fix:** narrow or let it propagate in dev.

---

## P2 — Over-defensive `getattr`/defaults on GUARANTEED dataclass fields

Every field of `Cell`, `ReportDoc`, `FigureSpec`, `PanelSpec`, `LayerSpec` is **default-valued** (model.py:711-724 Cell, 736-742 ReportDoc, 520-531 PanelSpec, 590-594 FigureSpec) → **always present on a real instance**. So `getattr(cell, 'field', default)` can only fire its default if `cell` isn't the dataclass it's typed as (None / a dict / wrong type) — i.e. it masks a contract violation. The tell throughout: the *same block* reads `c.id`, `c.spec`, `c.cell_type`, `c.source` as plain attributes but wraps sibling fields in `getattr`. Recommended fix everywhere: **drop the default, access the attribute directly**, so a wrong object fails loud (exactly the split-cell design goal).

### CONFIRMED — report state / serialization (a wrong object here corrupts the document)

| # | Site | Code | Fix |
|---|---|---|---|
| 1 | `handlers.py:302-310` | `getattr(c, "slide_break"/"live_action"/"slide_kind"/"slide_style"/"notes", …)` in `state()` | `c.slide_break` etc. |
| 2 | `handlers.py:319, 323, 330` | `getattr(c, "split_layout", "text-left")`, `getattr(c, "image_ext", "")` | `c.split_layout`, `c.image_ext` |
| 3 | `handlers.py:372, 2024, 2040` | `getattr(self.doc/mgr.doc, "doc_type", "report")`, `getattr(c, "split_layout", …)` in template save | direct |
| 4 | `handlers.py:2355` | `getattr(cell, "slide_break", False)` in `report_toggle_slide_break` (cell already non-None) | `not cell.slide_break` |
| 5 | `model.py:942-965, 873` | `getattr(c, …)` for `slide_break/slide_kind/slide_style/notes/live_action/split_layout` in `serialize_report_md` + `move_slide` | direct (serialization twins of #1/#2) |
| 6 | `model.py:804, 823-825, 834` | `getattr(c, "cell_type", "")`, `getattr(first, "slide_kind"/"slide_style"/"notes", "")` in `slide_columns`/`slide_meta`/`slide_notes` | keep the `first is None` guard, drop the attr-absent default: `first.slide_kind if first else ""` |
| 7 | `export_html.py:174, 239, 287` | `getattr(cell, "image_ext")`, `getattr(cell.spec, "vectors_mode")` (spec already non-None), `getattr(cell, "split_layout")` | direct |
| 8 | `handlers.py:425` | `getattr(spec, "vectors_mode", "")` (spec non-None, guarded above) | `spec.vectors_mode` |

### CONFIRMED — `PanelSpec.kind` read via `getattr` (redundant even inside `str(...)`)
`kind` is a guaranteed `PanelSpec` field (default `"image"`). The codebase uses `str(panel.kind)` in most places and `str(getattr(panel, "kind", ""))` in these — the getattr form is the smell:
- `handlers.py:68, 80` (`_is_scene3d_panel`/`_is_line_panel`), `figure_builder.py:454, 456`, `export_html.py:777` (in `report_paste_cell` — `spec` was *just built* by `FigureSpec.from_dict` two lines up, so `kind` is guaranteed). → `str(panel.kind)`.

### SUSPECTED — `_is_*`/`_cell` gate helpers
`handlers.py:63-64` (`_is_figure_like`), `:75` (`_is_scene3d_cell`), `compose.py:84-85` (`_cell`), `overlay_embed.py:100`, `vectors_embed.py:917-918` — `getattr(cell, "cell_type"/"spec", …)`. The `cell is None` guard is legitimate; the attr-absent default is not (in `_is_figure_like` the None check is on the line above, so `getattr(cell, "cell_type", "")` is pure redundancy). Fix: after the None check, use `cell.cell_type`/`cell.spec`.

**NOT flagged (correct):** ~450 other `getattr` across `spyde/` read genuinely external/dynamic shapes — hyperspy axes (`scale/offset/units/size`), dask internals (`_lazy`, `chunksize`, `client`), set-late plot/session/tree attrs (`_plot2d`, `plot_state`, `_om_wizard`), torch/cupy device probes. Those are correct defensive reads. Also correct: all the `*.from_dict` `d.get(key, default)` — those parse external YAML with genuinely-optional fields.

---

## P3 — Broad `except Exception` where narrow is intended

The report area's `except Exception` is **overwhelmingly correct-by-design** (event-handler wiring that must never crash compute, anyplotlib build/emit best-effort, snapshot harvest, save/export top-level — all log). Only these have a clearly-narrower intended error:

| Site | Code | Narrow to | Hides |
|---|---|---|---|
| `handlers.py:2524-2527` | `count = int(len(vecs.flat_buffer))` → `except Exception: count = 0` | `(TypeError, AttributeError)` or drop | a renamed/broken `flat_buffer` silently reports `count=0` forever |
| `handlers.py:1435-1438, 1577-1580` | `clim = [float(lv[0]), float(lv[1])]` → `except Exception: clim = None` | `(TypeError, ValueError, IndexError)` (matches `compose.py:1048` which does it right) | an unrelated error becomes "no clim" |
| `handlers.py:1338-1349, 1376-1386` | per-field `str()/float()` of renderer line-style state → `except Exception: pass` | `(TypeError, ValueError)` | an `AttributeError` from a mistyped dict access |
| `model.py:1188-1190` | `yaml.safe_load(m_live.group("payload"))` → `except Exception` | `yaml.YAMLError` | a regex-group `AttributeError` (low sev — external marker text) |

**NOT flagged (correct broad catches):** `_wire_*`/`_make_*_handler` bodies, `build_cell_figure`, all `figure_builder.py` anyplotlib calls, `export_html` interactive→static degradation, report open/save/export top-level (all `emit_error`), the atomic `write_report` cleanup (`model.py:1420` — swallows only the cleanup OSError and **re-raises** the real error, textbook-correct), and every compute/dask/worker/GPU boundary CLAUDE.md mandates.

---

## P4 — Dead legacy / Qt vestiges (mechanical, safe deletions)

No serialization-format legacy exists. What's dead is pyqtgraph/QWidget scaffolding left from the Qt→Electron migration — no-op stubs with **zero callers** (all grep-verified repo-wide). None touches a format, so none can reintroduce a split-cell-style edge bug.

**Group A — the `spyde/drawing/selectors/` dead subsystem (one sweep):**
- `selectors/selection_selector2d.py` — **entire file** dead (`SelectionSelector2D`, "kept for import compatibility", zero importers). Delete the file.
- `selectors/utils.py:32-49` — `no_return_update_function`, `create_linked_rect_roi`, `create_linked_linear_region`, `create_linked_infinite_line` (pyqtgraph linked-ROI stubs, zero callers). Keep `broadcast_rows_cartesian` (live). Delete the four + the misleading "pyqtgraph signal chaining" comment.
- `base_selector.py:124-128, 186` — `_StubWidget` + `self.widget = _StubWidget()` — nothing ever *reads* `selector.widget`. Delete both.
- `selector2d.py:602-603` — `IntegratingSSelector2D = IntegratingSSelector2D` (self-assignment no-op). Delete.
- `selector2d.py:469-470` + `selectors/__init__.py:4` — `LineSelector = LineProfileSelector` alias, no consumer uses the name. Delete + drop from `__init__` export.
- `add_linked_roi(self, plot): pass` stubs across `selector2d.py` (70,166,214,254,424,598), `selector1d.py` (81,179,317), `base_selector.py` (489) — zero call sites. Delete all.

**Group B — Qt method-name no-op shims on Plot/PlotWindow:**
- `plot_window.py` — `set_graphics_layout_widget` (182), `_build_new_layout` (186), `setGraphicsEffect` (143), `raise_` (118), `lower` (122), `setGeometry` (136), `previous_subplots_pos`/`previous_subplot_added` (72-73), and the `x/y/width/height` lambda properties (189-204) — QMdiSubWindow/QWidget shims, no callers. (Keep `show`/`hide`/`isVisible`/`move`/`resize`/`close` — those have real Electron-emit bodies.)
- `plot.py:1222-1235` — `addItem`, `removeItem`, `normalize_axes`, `update_range` (all `pass`, pyqtgraph PlotItem names, zero callers). Delete + the section header.
- *Risk note:* CONFIRMED-dead within the Python backend; these are Python object methods, not IPC verbs, so the electron boundary is unaffected — but a 10-second grep of the electron side for these exact names before deleting is cheap insurance.

**Group C — the one true dead compat branch in the report/overlay path:**
- `spyde/actions/overlay.py:413-417` — `_apply_pending_layer_frames` tolerates a "legacy (handle, frame) 2-tuple", but the only writer (`_enqueue_layer_push`) always builds a 3-tuple `(layer, layer.handle, frame)`. The `else` is unreachable. Collapse to `layer, handle, frame = entry`.

**Group D — low-value dead kwargs:**
- `multiplot_manager.py:48-52` — `main_window=None` "legacy" kwarg; the sole construction site passes `session=`. Drop the param + `or main_window`. (The `main_window=` *parameter name* on the compute functions is a live `Session` carrier — a rename, not dead code; leave it.)
- `update_functions.py:926, 943-947` — `cache_in_shared_memory` DEPRECATED no-op kwarg; only a benchmark passes it. Drop once you confirm no electron caller sets it.

**Housekeeping (not code):**
- `mdi_manager.py:4` references `_qt_main_legacy.py ("Phase 4 reference")` — **that file doesn't exist**. Stale comment; drop the reference. (`MDIManager` itself is fully live.)
- No Qt imports remain anywhere in `spyde/` — clean. Only migration *comments* reference it.

---

## Explicitly NOT dead / NOT a smell (traced — don't waste time removing)

The report code's docstrings say "back-compat"/"legacy" in several places that are actually **correct optional-field tolerance within the single v1 schema**, or live fallbacks still triggered by current inputs:
- The whole `model.py` "SCHEMA_VERSION stays 1 / older files → default" family (`LayerSpec.tint/color/linewidth`, `PanelSpec.scene/text_sizes`, `FigureSpec.vectors_mode`, `Cell.slide_break/…/split_layout`, `ReportDoc.doc_type`) — features added *within* v1 with emit-when-set + `.get()`. No prior format ever shipped. **Correct new-optional-field design** — the opposite of the split-cell bug.
- `read_report` `.png`-without-yaml → image promotion — the current, intentional figure-vs-photo disambiguation. Both are live cell types.
- `_normalize_doc_type` "movie" — reserved-*forward* value, forward-compat not backward.
- `compose.py` "legacy edge-of-grid placement" (289,401,427,443) — live fallback when the renderer omits `target_panel_id`.
- `export_html.py` "backward compatible" token-omission (591,643,693) — optional field in the *live* backend↔renderer IPC (they ship together).
- `session.py:447`, `playback.py DEFAULT_FPS`, `live_overlay.py "sync"`, `_session_files.py _EXAMPLE_CALIBRATION`, `spotunet-base16-v1` weights — all live; "legacy" there is lineage-describing.
- `diffraction_vectors.py offsets` "legacy single-level" (226,253,347,748,1014) — a redundant alias of `nav_offsets[-1]` that's still *produced and consumed*; a de-dup *simplification* opportunity, not dead code.

---

## Suggested order

1. **P1 #1 + #2** — the two data-loss swallows. Highest stakes, smallest diff. Narrow the except + escalate to a user-visible warning.
2. **P2** — the ~25 guaranteed-field `getattr` sites. Mechanical, and it makes wrong-object bugs fail loud (your stated split-cell goal). One focused pass over `handlers.py state()/serialize`, `model.py serialize/slide_*`, `export_html.py`, `figure_builder.py`, `compose.py`.
3. **P4 Group A** — the selectors dead-subsystem sweep. Pure deletion, removes the most "looks like it still does something" misdirection.
4. **P3** + **P4 B/C/D** — narrow the four excepts; delete the remaining Qt shims + the overlay 2-tuple branch + the two dead kwargs.

---

## Execution outcome (all four applied)

All four buckets were applied, with per-site verification. Two auditor claims were **corrected during execution** by re-checking callers myself:

- **`_StubWidget` / `self.widget` in `base_selector.py` is NOT dead — kept.** The auditor grepped only within the selectors package and concluded nothing reads `selector.widget`. It missed `plot.py:1125` `sel.widget.hide()`, a live call during node-switching. `_StubWidget` provides that `.hide()` no-op; removing it would raise. Left in place.
- **`raise_` / `lower` / `setGeometry` / `setGraphicsEffect` on `PlotWindow` are functional IPC emitters, not no-op shims.** The auditor listed them among dead no-op shims; they actually emit `window_raise`/`window_lower`/etc. They have no current callers but are a real API surface, so I left them (deleting uncalled-but-working emitters is a judgment call, not "dead code"). Only the genuine `pass`-body shims (`set_graphics_layout_widget`, `_build_new_layout`, the `x/y/width/height` fake-geometry properties, `previous_subplot*` attrs) were removed.

Everything else was applied as reported:
- **P1:** narrowed both `read_report` excepts (+ new `Cell.spec_error`, user warning on open), narrowed the `assemble_assets` bake except (+ `_dropped_assets` tracking, user warning on save), and the three SUSPECTED (count_map / scene3d / example-calibration) narrowed + bumped to `log.warning`. 7 new regression tests.
- **P2:** all ~25 guaranteed-field `getattr` sites → direct attribute access across `handlers.py`, `model.py`, `export_html.py`, `figure_builder.py`, `compose.py`, `overlay_embed.py`, `vectors_embed.py`.
- **P3:** narrowed the `flat_buffer` count, both `clim` conversions, the six line-style per-field conversions, and the `yaml.safe_load` marker parse.
- **P4:** deleted `selection_selector2d.py`, the 4 `create_linked_*`/`no_return_update_function` stubs, all 9 `add_linked_roi` stubs, the `LineSelector`/`IntegratingSSelector2D` aliases, the `Plot` axis no-op shims, the `PlotWindow` no-op shims, the overlay 2-tuple dead branch, the `main_window` + `cache_in_shared_memory` dead kwargs, and the stale `_qt_main_legacy.py` doc reference.
