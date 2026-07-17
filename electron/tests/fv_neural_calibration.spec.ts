/**
 * fv_neural_calibration.spec.ts — neural model registry + auto-calibration UI.
 *
 * Covers the Phase 0/1 neural-integration wiring (NEURAL_INTEGRATION_PLAN.md):
 *   - the wizard shows the neural High-pass σ slider + the ↻ refresh-models
 *     button beside the Model dropdown,
 *   - opening the wizard runs the one-shot auto-calibration on the backend
 *     (asserted via the backend log line — PLOTAPP messages don't reach stdout),
 *   - clicking ↻ refreshes the model list (offline-safe) and reports in the
 *     status line,
 *   - Compute still opens the vectors window with the new bg_sigma param in the
 *     payload.
 *
 * Real Dask + bundled si-grains, matching find_vectors_workflow.spec.ts.
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  // INFO logs tee to stderr so backend.waitForLog can see the calibration line.
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  const { page } = ctx
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(180_000)

test('neural wizard: bg-σ control, auto-calibration, model refresh, compute', async () => {
  const { page, backend } = ctx

  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()

  // Minimal neural pane (user decision 2026-07-16): Spot size + Threshold
  // sliders only — nav blur / min distance / subpixel / high-pass are hidden
  // (blur is never applied; the high-pass is auto-calibrated invisibly).
  await expect(page.getByTestId('fv-spot-size')).toBeVisible()
  await expect(page.getByTestId('fv-threshold')).toBeVisible()
  await expect(page.getByTestId('fv-sigma')).toHaveCount(0)
  await expect(page.getByTestId('fv-mindist')).toHaveCount(0)
  await expect(page.getByTestId('fv-subpixel')).toHaveCount(0)
  await expect(page.getByTestId('fv-bg-sigma')).toHaveCount(0)
  await expect(page.getByTestId('fv-model')).toBeVisible()
  await expect(page.getByTestId('fv-refresh-models')).toBeVisible()
  await page.screenshot({ path: 'fv_neural_shots/01-wizard-open.png' })

  // The themed Model dropdown opens with the menubar look (screenshot check).
  await page.getByTestId('fv-model').click()
  await expect(page.getByTestId('fv-model-opt-spotunet-production-v2')).toBeVisible()
  await page.screenshot({ path: 'fv_neural_shots/01b-model-dropdown.png' })
  await page.keyboard.press('Escape')

  // Auto-calibration ran on wizard-open (backend log; the emitted fv_calibration
  // is only adopted in the UI when it differs from the defaults, so the log is
  // the reliable signal that the pipeline executed).
  await backend.waitForLog('neural calibration:', 60_000)

  // ↻ refresh: offline-safe — with or without reachable HF the backend re-emits
  // the merged list and the wizard reports it.
  await page.getByTestId('fv-refresh-models').click()
  await expect(page.getByTestId('fv-status')).toContainText(/Model list refreshed/i, {
    timeout: 30_000,
  })
  await page.screenshot({ path: 'fv_neural_shots/02-models-refreshed.png' })

  // Compute (params now include spot_radius; nav blur forced off) → vectors
  // result window opens and the batch runs to completion.
  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('fv-compute').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 120_000, message: 'vectors result window never opened',
  }).toBeGreaterThan(before)

  // The LIVE compute HUD (backend dask_stats sampler → StatusBar DaskMonitor)
  // is flowing during the batch — the real end-to-end telemetry check.
  await expect(page.getByTestId('dask-monitor')).toBeVisible({ timeout: 15_000 })
  await page.getByTestId('dask-monitor').click()
  await expect(page.getByTestId('dask-monitor-popover')).toBeVisible()
  await page.screenshot({ path: 'fv_neural_shots/02b-compute-monitor.png' })
  await page.getByTestId('dask-monitor').click()   // close the popover again

  // WAIT for the batch to finish before afterAll closes the app: closing
  // mid-batch wedges teardown on Windows (the Electron stdin tick that keeps
  // the hidden backend scheduled stops during shutdown — see the fv-batch
  // stall note in CLAUDE.md). This also asserts the sigma=0 / spot-size
  // compute path actually completes.
  await backend.waitForLog('[fv-batch] finalized', 120_000)
  await page.screenshot({ path: 'fv_neural_shots/03-vectors-window.png' })

  ctx.assertNoJsErrors()
})
