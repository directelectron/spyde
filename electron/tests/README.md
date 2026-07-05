# SpyDE Playwright suite — the map (read this before writing or debugging a spec)

Run: `npx playwright test --project=electron --reporter=line --retries=0`
(from `electron/`; `npm run build` first if you changed renderer/main code —
the harness launches `out/main/index.js`, NOT the dev server).
One worker, serialized on purpose: one Electron + one Dask cluster at a time.

## The two speed tiers — pick the right one

| Tier | Data path | Startup | When |
|---|---|---|---|
| **fast** (`dask:false`, `SPYDE_NO_DASK`) | `loadTestVectors(page)` / `load_test_data*` | seconds | ANYTHING downstream of vectors (strain, vector VI, vector OM), UI wiring, wizards, layout |
| **slow** (`dask:true`, real `LocalCluster`) | `load_test_data_si_grains` + run the batch | ~60 s cluster + batch | ONLY what genuinely needs distributed: the find-vectors batch itself, the requires_vectors gate timing, nav-drag under distributed |

Rule: **do not pay the distributed batch to test something that isn't the
batch.** `loadTestVectors` gives you a finished Find-Vectors result tree
(vectors attached, vector actions unlocked) in seconds.

## What each spec answers

Fast (no dask):
- `spyde.spec.ts` — renderer↔backend contract via injected messages: toolbars,
  carets, all five wizards dispatch the right staged actions (`fv_open`,
  `strain_commit`, `czb_run`…). The first place a wizard-wiring bug shows.
- `strain_lazy.spec.ts` — the full strain lifecycle on `loadTestVectors`:
  caret → live map + component chips → **Commit → new "Strain" tree** →
  toggle-off teardown (committed tree survives) → reopen idempotency.
- `vector_vi_lazy` / `vector_om_lazy` / `om_wizard_lazy` / `vi_lazy` /
  `orientation_lazy` — each downstream action's wiring on the fast data paths.
- `find_vectors_result`, `vector_overlay`, `view_bar`, `ipf_*`, `visual`,
  `selector`, `mdi_layout`, `window_placement`, `nav_shape`, `loading`,
  `signal_type`, `composition`, `center_zero_beam`, `app_log`, `tour`,
  `update_gpu_dialogs` — focused single-feature checks.

Slow (real dask):
- `actions_lifecycle.spec.ts` — THE batch journey: `requires_vectors` gate
  (vector actions hidden while computing, appear on attach), then commit +
  caret teardown on the real result.
- `find_vectors_workflow.spec.ts` — live preview markers + batch launch.
- `orientation_workflow`, `virtual_imaging_workflow`, `nav_drag_distributed`
  — their features under a real cluster.

Diagnostic (opt-in, never in the default run):
- `_probe_fv_stall.spec.ts` — `SPYDE_PROBE=1` — zero-interaction batch health
  with periodic `dump_dask_state` snapshots. Use when a compute "hangs".

Real-data tier: `*.real.spec.ts` + `guide_screenshots` via
`--project=electron-real` (downloads pyxem datasets; nightly/local only).

## How to read a failure (in order)

1. **The screenshots.** `test-results/**/test-failed-1.png` + the spec's own
   per-stage shots (`electron/<name>_shots/NN-*.png`). The screenshot is the
   test — a black frame is a launch/placeholder failure, not "close enough".
2. **`error-context.md`** next to it — an accessibility snapshot of the whole
   page at failure; tells you which windows/buttons actually existed.
3. **`ctx.backend.logBuffer`** — the captured backend stderr. Specs dump the
   tail in `afterAll`. Backend `emit`/`emit_status`/`emit_error` do NOT appear
   here (they're the PLOTAPP stdout protocol) — set `SPYDE_LOG_LEVEL=INFO` in
   `launchApp({env})` to see `logging` lines (e.g. `[fv-batch]` timings).
4. Compute looks stuck? `await dumpDaskState(page)` and read the
   `[dask-state]` lines: task-state histogram + per-worker load + call stacks.

## Hard-won rules (each one was a real debugging session)

- **Signal-based waits only.** Never `waitForTimeout` as a completion wait.
  The "vectors attached" signal is `waitForVectorActions` (the
  requires_vectors-gated buttons appearing) — NOT a log grep for "Found".
- **Never select windows by text negatives.** `hasNotText` is case-insensitive
  substring over the WHOLE subwindow (wizard text included: the FV wizard hint
  contains "navigator", its title contains "Vectors"). Select by what the
  window HAS: `filter({ has: page.getByTestId('action-btn-<Name>') })`.
- **Park, hover, then click.** Toolbars reveal on hover and the hovered window
  auto-raises above siblings — but the PREVIOUS window's still-revealed
  toolbar can cover your target and intercept the pointer. The reliable idiom
  when switching windows: `page.mouse.move(640, 12)` (neutral top bar) +
  `waitForTimeout(600)` (> the 350 ms toolbar-hide delay), then `hover()` the
  target titlebar, then `click()`.
- **Kill strays before a dask run**: leftover electron/python processes cause
  port contention that looks like app flakiness
  (`Get-Process electron | Stop-Process -Force`). Don't kill python while a
  pytest run is live.
- **Batch time budgets:** cluster spawn ~60 s on this box; the si-grains batch
  a further ~10-60 s. Give attach polls 120 s+ and put the wait on the REAL
  signal so a healthy fast run finishes fast.
- New wizard? Give every control a `data-testid`, verify its staged dispatch
  in `spyde.spec.ts`, its lifecycle in a FAST spec, and only touch the slow
  tier if the feature is distributed-specific. Backend contract:
  `spyde/actions/README.md`.
