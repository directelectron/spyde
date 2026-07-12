/**
 * virtual_imaging_workflow.spec.ts — Virtual Imaging, end-to-end.
 *
 * Loads lazy 4D data, opens the Virtual Imaging sub-toolbar on the DP, adds a
 * detector-ROI virtual image, and asserts the output window paints non-black
 * (the "VI is just black" bug the eager tests don't catch). Harness-based:
 * renderer JS errors fail the test, waits are signal-based, no fixed sleeps.
 *
 * Real Dask + synthetic lazy 4D — the eager SPYDE_NO_DASK path does not render
 * windows through the full Electron pipeline in this tree.
 */
import { test, expect } from '@playwright/test'
const { launchApp, backendAction, waitForSubwindowCount, sigWindow } = require('./_harness.cjs')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true })
  const { page } = ctx
  await backendAction(page, 'load_test_data_lazy')
  await waitForSubwindowCount(page, 2, 60_000)
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(180_000)

test('virtual image output window paints non-black', async () => {
  const { page } = ctx
  // Breadcrumb-chip picker — the literal word "Navigator" is gone from titles,
  // so hasNotText:'Navigator' matched BOTH windows and .first() grabbed the nav.
  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()   // reveal toolbar
  // Wait for the toolbar button to actually be revealed before clicking — on a
  // loaded CI runner the hover-reveal lags, and clicking immediately times out.
  const viBtn = sig.getByTestId('action-btn-Virtual Imaging')
  await expect(viBtn).toBeVisible({ timeout: 30_000 })
  await viBtn.click()
  await expect(page.getByTestId('sub-toolbar')).toBeVisible({ timeout: 15_000 })
  await page.getByTestId('subaction-add_virtual_image').click()

  await expect(page.getByTestId('vi-icon-Virtual Image 1 (red)'))
    .toBeVisible({ timeout: 30_000 })

  const viWin = page.getByTestId('subwindow').last()
  const maxPix = async () => {
    const h = await viWin.locator('iframe').first().elementHandle()
    const frame = h ? await h.contentFrame() : null
    if (!frame) return 0
    return frame.evaluate(() => {
      const cs = Array.from(document.querySelectorAll('canvas')) as HTMLCanvasElement[]
      if (!cs.length) return 0
      const c = cs.sort((a, b) => b.width * b.height - a.width * a.height)[0]
      const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data
      let mx = 0
      for (let p = 0; p < d.length; p += 4) mx = Math.max(mx, d[p] + d[p + 1] + d[p + 2])
      return mx
    })
  }
  await expect.poll(maxPix, { timeout: 30_000, message: 'VI output stayed black' })
    .toBeGreaterThan(30)

  ctx.assertNoJsErrors()
})
