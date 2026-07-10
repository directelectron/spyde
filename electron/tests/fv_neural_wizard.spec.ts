/**
 * fv_neural_wizard.spec.ts — the neural (SpotUNet) method in the FV wizard.
 *
 * Asserts the reconstructed neural wiring end-to-end in the real app:
 *   - the wizard's Method dropdown DEFAULTS to 'neural',
 *   - the Model dropdown appears and is populated from the backend registry
 *     (the fv_models payload → registry default selected),
 *   - the live preview draws peak markers with the neural detector,
 *   - Compute with the neural method opens a vectors result window.
 *
 * Harness-based (real Dask + bundled Si-grains) — see find_vectors_workflow.spec.ts.
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, backendAction, waitForSubwindowCount, countColorPixels, sigWindow,
} = require('./_harness.cjs')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true })
  const { page } = ctx
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(180_000)

test('neural is the default method with a populated Model dropdown', async () => {
  const { page } = ctx

  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()

  // Method defaults to neural.
  await expect(page.getByTestId('fv-method')).toHaveValue('neural')

  // Model dropdown populated from the registry (fv_models round trip): it
  // renders only once options arrive, and the registry default is selected.
  const model = page.getByTestId('fv-model')
  await expect(model).toBeVisible({ timeout: 15_000 })
  const selected = await model.inputValue()
  expect(selected.length).toBeGreaterThan(0)
  const optionCount = await model.locator('option').count()
  expect(optionCount).toBeGreaterThan(0)

  // Live preview draws red peak markers via the neural detector.
  await expect.poll(() => countColorPixels(page, 'red'), {
    timeout: 60_000, message: 'neural live preview drew no peak markers',
  }).toBeGreaterThan(0)

  await page.screenshot({ path: 'fv_neural_shots/01-wizard-neural-default.png' })

  // Compute with the neural method → a vectors result window opens.
  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('fv-compute').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 120_000, message: 'neural vectors result window never opened',
  }).toBeGreaterThan(before)
  await page.screenshot({ path: 'fv_neural_shots/02-neural-vectors-window.png' })

  ctx.assertNoJsErrors()
})
