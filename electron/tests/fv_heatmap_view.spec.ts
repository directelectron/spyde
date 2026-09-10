/**
 * fv_heatmap_view.spec.ts: the Find Vectors "Show heatmap" view is an overlay
 * group of kind `transform`: while it is on, the painter shows the detector's
 * response image instead of the raw pattern and never flashes the raw frame
 * on a navigator move; turning it off restores the pattern without a move.
 *
 * Real Dask + the lazy 3×3-chunk synthetic scan. Screenshots land in
 * electron/fv_heatmap_shots/ and are the check: read them.
 */
import { test, expect } from '@playwright/test'
import * as fs from 'fs'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, countColorPixels, sigWindow,
  navWindow, dragCrosshair,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'fv_heatmap_shots')
let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  fs.mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  const { page } = ctx
  await backendAction(page, 'load_test_data_lazy_chunked')
  await waitForSubwindowCount(page, 2, 120_000)
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(240_000)

test('the heatmap view replaces the pattern and comes back off without a move', async () => {
  const { page } = ctx
  const sig = sigWindow(page)
  const nav = navWindow(page)
  const red = () => countColorPixels(page, 'red')

  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()
  await expect.poll(red, { timeout: 60_000 }).toBeGreaterThan(0)
  await page.screenshot({ path: `${SHOTS}/01-preview.png` })

  // Heatmap on: the signal panel shows the response image.
  const before = await sig.screenshot()
  await page.getByTestId('fv-show-transform').click()
  await page.waitForTimeout(1_500)
  const heat = await sig.screenshot({ path: `${SHOTS}/02-heatmap-on.png` })
  expect(Buffer.compare(before, heat), 'the panel did not change with the heatmap on').not.toBe(0)
  await page.screenshot({ path: `${SHOTS}/02b-heatmap-on-full.png` })

  // Drag with the heatmap on: it must keep showing the response, not the raw frame.
  await dragCrosshair(page, nav, { dx: 6, dy: 4, steps: 6, settleMs: 150 })
  await page.waitForTimeout(800)
  await sig.screenshot({ path: `${SHOTS}/03-heatmap-after-drag.png` })
  expect(await red()).toBeGreaterThan(0)

  // Heatmap off: the raw pattern comes back with no navigator move.
  await page.getByTestId('fv-show-transform').click()
  await page.waitForTimeout(1_500)
  const back = await sig.screenshot({ path: `${SHOTS}/04-heatmap-off.png` })
  expect(Buffer.compare(back, heat), 'the panel did not change back with the heatmap off').not.toBe(0)

  ctx.assertNoJsErrors()
})
