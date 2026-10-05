/**
 * fv_symmetry_stage.spec.ts — the neural method's Sample and Symmetry dropdowns.
 *
 * Sample: Crystalline shows the Symmetry dropdown (Off / Friedel pairs); picking
 * Friedel re-runs the live preview with it (backend log) and Compute runs the
 * batch with it to completion. Sample: Amorphous hides the Symmetry dropdown.
 *
 * Real Dask + bundled si-grains, matching fv_centre_refiner.spec.ts.
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

let ctx: Awaited<ReturnType<typeof launchApp>>
const SHOTS = 'fv_symmetry_shots'

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  await backendAction(ctx.page, 'load_test_data_si_grains')
  await waitForSubwindowCount(ctx.page, 2, 120_000)
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(240_000)

test('neural wizard: Sample and Symmetry dropdowns', async () => {
  const { page, backend } = ctx

  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()
  await backend.waitForLog('neural calibration:', 60_000)

  await expect(page.getByTestId('fv-sample-type')).toContainText('Crystalline')
  const symmetry = page.getByTestId('fv-symmetry')
  await expect(symmetry).toContainText('Off')
  await symmetry.click()
  await expect(page.getByTestId('fv-symmetry-opt-off')).toBeVisible()
  await expect(page.getByTestId('fv-symmetry-opt-friedel')).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/01-symmetry-dropdown.png` })

  await page.getByTestId('fv-symmetry-opt-friedel').click()
  await expect(symmetry).toContainText('Friedel pairs')
  await backend.waitForLog('symmetry=friedel', 60_000)

  await page.getByTestId('fv-sample-type').click()
  await expect(page.getByTestId('fv-sample-type-opt-amorphous')).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/02-sample-dropdown.png` })
  await page.getByTestId('fv-sample-type-opt-amorphous').click()
  await expect(page.getByTestId('fv-symmetry')).toHaveCount(0)
  await backend.waitForLog('symmetry=off', 60_000)
  await page.screenshot({ path: `${SHOTS}/03-amorphous-hides-symmetry.png` })

  await page.getByTestId('fv-sample-type').click()
  await page.getByTestId('fv-sample-type-opt-crystalline').click()
  await expect(page.getByTestId('fv-symmetry')).toContainText('Friedel pairs')

  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('fv-compute').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 120_000, message: 'vectors result window never opened',
  }).toBeGreaterThan(before)
  // Wait for the batch: closing mid-batch wedges teardown on Windows.
  await backend.waitForLog('[fv-batch] finalized', 180_000)
  await page.screenshot({ path: `${SHOTS}/04-vectors-window.png` })

  ctx.assertNoJsErrors()
})
