/**
 * fv_centre_refiner.spec.ts — the neural method's Centre dropdown.
 *
 * Find Vectors (neural) with Centre = Mask centroid on the bundled Si grains:
 * the dropdown lists the built-in choices, picking one re-runs the live preview
 * with it (backend log), and Compute runs the batch with it to completion.
 *
 * Real Dask + bundled si-grains, matching fv_neural_calibration.spec.ts.
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

let ctx: Awaited<ReturnType<typeof launchApp>>
const SHOTS = 'fv_centre_shots'

test.beforeAll(async () => {
  // INFO logs tee to stderr so backend.waitForLog sees the tune / batch lines.
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  await backendAction(ctx.page, 'load_test_data_si_grains')
  await waitForSubwindowCount(ctx.page, 2, 120_000)
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(240_000)

test('neural wizard: Centre = Mask centroid previews and computes', async () => {
  const { page, backend } = ctx

  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()
  await backend.waitForLog('neural calibration:', 60_000)

  const centre = page.getByTestId('fv-centre')
  await expect(centre).toBeVisible()
  await centre.click()
  await expect(page.getByTestId('fv-centre-opt-decode')).toBeVisible()
  await expect(page.getByTestId('fv-centre-opt-mask-centroid')).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/01-centre-dropdown.png` })

  await page.getByTestId('fv-centre-opt-mask-centroid').click()
  await expect(centre).toContainText('Mask centroid')
  await backend.waitForLog('centre=mask-centroid', 60_000)
  await page.screenshot({ path: `${SHOTS}/02-mask-centroid-preview.png` })
  await sig.screenshot({ path: `${SHOTS}/02b-pattern-with-preview.png` })

  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('fv-compute').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 120_000, message: 'vectors result window never opened',
  }).toBeGreaterThan(before)
  // Wait for the batch: closing mid-batch wedges teardown on Windows.
  await backend.waitForLog('[fv-batch] finalized', 180_000)
  await page.screenshot({ path: `${SHOTS}/03-vectors-window.png` })

  // Reopening the caret restores the choice.
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('fv-centre')).toContainText('Mask centroid')

  ctx.assertNoJsErrors()
})
