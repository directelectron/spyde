/**
 * find_vectors_result.spec.ts — once Find Vectors has computed, the RESULT
 * window must DISPLAY nicely (Qt parity): navigating its count map renders each
 * position's vectors as flat disks AND draws red found-vector markers over them.
 *
 * Regression guard for two bugs that left the result window looking broken:
 *   1. the signal plot's CachedDaskArray captured the placeholder zeros, so the
 *      window stayed all-black after the data was swapped for rendered disks;
 *   2. no marker overlay was attached to the result window (markers were only on
 *      the source DP).
 *
 * Scopes the pixel scan to the vectors SIGNAL window's own iframe so red pixels
 * from the source-DP overlay can't satisfy the assertion.
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'

let app: ElectronApplication
let page: Page

test.beforeAll(async () => {
  app = await electron.launch({
    args: [join(__dirname, '..', 'out', 'main', 'index.js')],
    env: { ...process.env },   // real LocalCluster + client
  })
  let daskReady = false
  app.process().stdout?.on('data', (d: Buffer) => {
    if (String(d).includes('Dask cluster ready')) daskReady = true
  })
  page = await app.firstWindow()
  await page.waitForLoadState('domcontentloaded')
  for (let i = 0; i < 80 && !daskReady; i++) await page.waitForTimeout(500)
  await page.evaluate(() => window.electron.action('load_test_vectors', {}))
  // source nav+sig (2) + vectors nav+sig (2) = 4 windows.
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 4,
    { timeout: 60_000 },
  )
  await page.waitForTimeout(2500)
})

test.afterAll(async () => { await app?.close() })

test('computed vectors window renders disks + red vector markers (Qt parity)', async () => {
  // The SIGNAL vectors window (not its count-map navigator) is the one carrying
  // the Vector Virtual Imaging action — raise it to the front.
  const vsig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('action-btn-Vector Virtual Imaging') }).first()
  await vsig.getByTestId('subwindow-titlebar').click()
  await page.waitForTimeout(1000)

  // Scan ONLY this window's figure iframe: bright (rendered-disk) grayscale
  // pixels AND saturated-red (#ff3030) marker pixels must both be present.
  const ifel = vsig.locator('iframe').first()
  await expect(ifel).toBeVisible()
  const frame = await (await ifel.elementHandle())!.contentFrame()
  expect(frame).not.toBeNull()

  const scan = async () => frame!.evaluate(() => {
    let bright = 0, red = 0
    for (const c of Array.from(document.querySelectorAll('canvas')) as HTMLCanvasElement[]) {
      const ctx = c.getContext('2d')
      if (!ctx || !c.width || !c.height) continue
      const d = ctx.getImageData(0, 0, c.width, c.height).data
      for (let p = 0; p < d.length; p += 4) {
        const r = d[p], g = d[p + 1], b = d[p + 2]
        if (r > 120 && g < 80 && b < 80) red++
        else if (r > 100 && g > 100 && b > 100) bright++
      }
    }
    return { bright, red }
  })

  // Rendered disks: the window is NOT an all-black placeholder.
  await expect.poll(async () => (await scan()).bright, {
    timeout: 20_000, message: 'result window rendered no disks (still placeholder zeros)',
  }).toBeGreaterThan(0)
  // Found-vector markers overlaid on the rendered disks.
  await expect.poll(async () => (await scan()).red, {
    timeout: 20_000, message: 'no found-vector markers on the result window',
  }).toBeGreaterThan(0)
})
