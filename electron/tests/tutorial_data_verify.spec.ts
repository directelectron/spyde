/**
 * tutorial_data_verify.spec.ts — Phase-1 verification-only smoke test (not part
 * of the durable suite; a throwaway drive to confirm `tutorial_load` actually
 * opens windows in the real app). Dispatches `tutorial_load {name:'navigation'}`
 * (eager, no-dask, no download — pyxem's generate_4d_data 10x10x50x50) and
 * confirms a navigator + signal subwindow both appear, then screenshots.
 */
const { test, expect } = require('@playwright/test')
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

test('tutorial_load(navigation) opens navigator + signal windows', async () => {
  const { app, page, assertNoJsErrors } = await launchApp({ dask: false })
  try {
    await page.waitForTimeout(1500)
    await backendAction(page, 'tutorial_load', { name: 'navigation' })
    await waitForSubwindowCount(page, 2, 30_000)
    await page.waitForTimeout(1500)  // let the navigator/signal frames paint
    await page.screenshot({ path: 'tutorial_data_shots/01-navigation.png' })

    const count = await page.locator('[data-testid="subwindow"]').count()
    expect(count).toBeGreaterThanOrEqual(2)
    assertNoJsErrors()
  } finally {
    await app.close()
  }
})
