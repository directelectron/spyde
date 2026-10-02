/**
 * fv_neural_calibration.spec.ts — neural model registry + auto-calibration UI.
 *
 * Covers the neural-integration wiring (remote model registry + auto-calibration):
 *   - the wizard shows the neural High-pass σ slider + the ↻ refresh-models
 *     button beside the Model dropdown,
 *   - opening the wizard runs the one-shot auto-calibration on the backend
 *     (asserted via the backend log line — PLOTAPP messages don't reach stdout),
 *   - clicking ↻ refreshes the model list (offline-safe) and reports in the
 *     status line,
 *   - Compute is held until the automatic estimates have landed, then runs
 *     with exactly the parameters the caret shows, and opens the vectors
 *     window.
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
  const { page, backend, app } = ctx
  // Record what the renderer sends, beside the real handler that forwards it.
  await app.evaluate(({ ipcMain }) => {
    ;(globalThis as any).__fvSent = []
    ipcMain.on('spyde:action', (_e, action, payload) => {
      ;(globalThis as any).__fvSent.push({ action, payload })
    })
  })

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

  // Compute waits for the estimates, then sends what the caret shows (params
  // include spot_radius; nav blur forced off) → the vectors result window
  // opens and the batch runs to completion.
  await expect(page.getByTestId('fv-compute')).toBeEnabled({ timeout: 60_000 })
  // The number beside each slider, not the slider itself: a range input
  // snaps its value to the step, and a calibrated threshold need not sit on it.
  const label = (testid: string) => page.getByTestId(testid)
    .locator('xpath=following-sibling::span').textContent()
  const shown = {
    spot: Number(await label('fv-spot-size')),
    threshold: Number(await label('fv-threshold')),
  }
  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('fv-compute').click()
  const sent = await app.evaluate(() => (globalThis as any).__fvSent)
  const run = sent.find((s: any) => s.action === 'fv_run')
  expect(run, 'Compute sent no fv_run').toBeTruthy()
  expect(run.payload).toMatchObject({ spot_radius: shown.spot, kernel_radius: shown.spot })
  expect(run.payload.threshold).toBeCloseTo(shown.threshold, 2)
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
