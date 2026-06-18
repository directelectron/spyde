/**
 * vi_lazy.spec.ts — the VIRTUAL IMAGE must display on the LAZY + real-Dask path
 * (the "VI is just black" bug the no-Dask eager tests don't catch). Launches the
 * real backend WITH Dask, loads lazy synthetic 4D data, adds a VI, and asserts
 * the output window's canvas is NOT the black placeholder.
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'

let app: ElectronApplication
let page: Page

test.beforeAll(async () => {
  app = await electron.launch({
    args: [join(__dirname, '..', 'out', 'main', 'index.js')],
    env: { ...process.env },   // NO SPYDE_NO_DASK → a real LocalCluster + client
  })
  let daskReady = false
  app.process().stdout?.on('data', (d: Buffer) => {
    if (String(d).includes('Dask cluster ready')) daskReady = true
  })
  page = await app.firstWindow()
  await page.waitForLoadState('domcontentloaded')
  for (let i = 0; i < 80 && !daskReady; i++) await page.waitForTimeout(500)  // ≤40s
  await page.evaluate(() => window.electron.action('load_test_data_lazy', {}))
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
    { timeout: 60_000 },
  )
  await page.waitForTimeout(2500)
})

test.afterAll(async () => { await app?.close() })

test('lazy virtual image displays non-black (real Dask client)', async () => {
  const sig = page.getByTestId('subwindow').filter({ hasNotText: 'Navigator' }).first()
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()   // toolbar reveals on hover
  await sig.getByTestId('action-btn-Virtual Imaging').click()
  await expect(page.getByTestId('sub-toolbar')).toBeVisible()
  await page.getByTestId('subaction-add_virtual_image').click()

  await expect(page.getByTestId('vi-icon-Virtual Image 1 (red)')).toBeVisible({ timeout: 30_000 })

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
})
