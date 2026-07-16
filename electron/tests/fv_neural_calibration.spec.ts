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

  // New neural controls: High-pass σ slider + refresh button by the dropdown.
  await expect(page.getByTestId('fv-bg-sigma')).toBeVisible()
  await expect(page.getByTestId('fv-model')).toBeVisible()
  await expect(page.getByTestId('fv-refresh-models')).toBeVisible()
  await page.screenshot({ path: 'fv_neural_shots/01-wizard-open.png' })

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

  // Compute (params now include bg_sigma) → vectors result window opens.
  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('fv-compute').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 120_000, message: 'vectors result window never opened',
  }).toBeGreaterThan(before)
  await page.screenshot({ path: 'fv_neural_shots/03-vectors-window.png' })

  ctx.assertNoJsErrors()
})
