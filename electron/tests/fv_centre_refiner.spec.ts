/**
 * fv_centre_refiner.spec.ts — the neural method's centre steps.
 *
 * Find Vectors (neural) always refines each disk's centre; the "Friedel
 * partner" checkbox, on by default, adds the step on top that also reads each
 * disk's Friedel mirror. Unticking and re-ticking it re-runs the live preview
 * with the switch (backend log), and Compute runs the batch with it on.
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

test('neural wizard: the Friedel-partner step is a checkbox, on by default', async () => {
  const { page, backend } = ctx

  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  const wizard = page.getByTestId('find-vectors-wizard')
  await expect(wizard).toBeVisible()
  await backend.waitForLog('neural calibration:', 60_000)

  // One switch, and no choice of refine step.
  const friedel = page.getByTestId('fv-friedel')
  await expect(friedel).toBeVisible()
  await expect(friedel).toBeChecked()
  await expect(page.getByTestId('fv-centre')).toHaveCount(0)
  await wizard.screenshot({ path: `${SHOTS}/01-controls.png` })

  await friedel.click()
  await expect(friedel).not.toBeChecked()
  await backend.waitForLog('refine=True friedel=False', 60_000)
  await page.screenshot({ path: `${SHOTS}/02-friedel-off-preview.png` })

  await friedel.click()
  await expect(friedel).toBeChecked()
  await backend.waitForLog('refine=True friedel=True', 60_000)
  await page.screenshot({ path: `${SHOTS}/03-friedel-on-preview.png` })
  await sig.screenshot({ path: `${SHOTS}/03b-pattern-with-preview.png` })
  // Both bundled networks loaded (a step that cannot load warns and is skipped).
  expect(backend.logBuffer.filter((line: string) => line.includes('unavailable'))).toEqual([])

  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('fv-compute').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 120_000, message: 'vectors result window never opened',
  }).toBeGreaterThan(before)
  // Wait for the batch: closing mid-batch wedges teardown on Windows.
  await backend.waitForLog('[fv-batch] finalized', 180_000)
  await page.screenshot({ path: `${SHOTS}/04-vectors-window.png` })

  // Reopening the caret restores the switch.
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('fv-friedel')).toBeChecked()

  ctx.assertNoJsErrors()
})
