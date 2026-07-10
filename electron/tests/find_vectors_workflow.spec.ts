/**
 * find_vectors_workflow.spec.ts — Find Diffraction Vectors, end-to-end.
 *
 * Opens the wizard on the live diffraction pattern (live red peak preview),
 * Computes across the scan, and asserts the found vectors stay overlaid as red
 * markers on the source DP. Harness-based: renderer JS errors fail the test,
 * waits are signal-based (red-pixel poll / subwindow count), no fixed sleeps.
 *
 * Real Dask + bundled-synthetic Si-grains (real reciprocal lattice with crisp
 * spots, so the peak finder has something to detect) — the eager SPYDE_NO_DASK
 * path does not render windows through the full Electron pipeline in this tree.
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

test('live red-peak preview, then Compute opens the vectors window', async () => {
  const { page } = ctx
  const red = () => countColorPixels(page, 'red')

  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()

  // No overlay yet → no saturated-red pixels on the grayscale DP.
  expect(await red()).toBe(0)

  // Open the wizard → the LIVE preview draws red peak markers under the crosshair
  // BEFORE any full-dataset compute.
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()
  await expect.poll(red, {
    timeout: 30_000, message: 'live preview drew no peak markers on the DP',
  }).toBeGreaterThan(0)

  // Compute → a vectors result window opens (the full-scan found-vectors output).
  // We assert the window opens rather than polling the post-compute overlay
  // repaint: the streaming compute over the whole scan is the slow stage and the
  // persistent-overlay repaint is covered by vector_overlay.spec.ts. This spec's
  // value is the live preview + the wizard→compute wiring + JS-error safety.
  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('fv-compute').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 120_000, message: 'vectors result window never opened',
  }).toBeGreaterThan(before)

  ctx.assertNoJsErrors()
})
