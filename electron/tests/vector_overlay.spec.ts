/**
 * vector_overlay.spec.ts — the Find-Diffraction-Vectors WIZARD must show a LIVE
 * found-peaks preview as you open it, then OVERLAY the final found vectors on
 * the live diffraction pattern after Compute (Qt parity). Launches the real
 * backend, loads lazy 4D data with a bright central disk, opens the wizard
 * (live preview → red markers), Computes, and asserts saturated-RED marker
 * pixels are present on the SOURCE DP canvas (the grayscale image has none).
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
  for (let i = 0; i < 80 && !daskReady; i++) await page.waitForTimeout(500)  // ≤40s
  await page.evaluate(() => window.electron.action('load_test_data_lazy', {}))
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
    { timeout: 60_000 },
  )
  await page.waitForTimeout(2500)
})

test.afterAll(async () => { await app?.close() })

// Count strongly-red pixels (R high, G/B low) across every canvas in EVERY
// figure iframe — the overlay markers are #ff3030; the grayscale DP/navigator
// never are. Scanning all frames is robust to which frame is the DP (a
// per-window `locator('iframe')` can resolve to a stale/empty contentFrame).
const COUNT_RED = () => {
  let r = 0
  for (const c of Array.from(document.querySelectorAll('canvas')) as HTMLCanvasElement[]) {
    const ctx = c.getContext('2d')
    if (!ctx || !c.width || !c.height) continue
    const d = ctx.getImageData(0, 0, c.width, c.height).data
    for (let p = 0; p < d.length; p += 4) {
      if (d[p] > 120 && d[p + 1] < 80 && d[p + 2] < 80) r++
    }
  }
  return r
}

async function redPixels(): Promise<number> {
  let total = 0
  for (const frame of page.frames()) {
    try { total += await frame.evaluate(COUNT_RED) } catch { /* detached frame */ }
  }
  return total
}

test('live preview + found-vector overlay as red markers on the diffraction pattern', async () => {
  const sig = page.getByTestId('subwindow').filter({ hasNotText: 'Navigator' }).first()
  await sig.getByTestId('subwindow-title').click()

  // No overlay yet → no saturated-red pixels on the DP.
  expect(await redPixels()).toBe(0)

  // Open the Find Vectors wizard (toolbar reveals on hover). Opening it starts
  // the LIVE preview → red peak markers appear under the crosshair BEFORE any
  // full-dataset compute.
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()
  await expect.poll(() => redPixels(), {
    timeout: 30_000, message: 'live preview drew no peak markers on the DP',
  }).toBeGreaterThan(0)

  // Compute → a new vectors window opens and the persistent overlay stays on
  // the source DP.
  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('fv-compute').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 60_000, message: 'vectors window never opened',
  }).toBeGreaterThan(before)
  await expect.poll(() => redPixels(), {
    timeout: 30_000, message: 'no vector markers overlaid on the DP after compute',
  }).toBeGreaterThan(0)
  const afterCompute = await redPixels()   // source overlay + result-window overlay

  // Deselecting the action (closing the caret) HIDES the SOURCE-DP overlay. The
  // vectors RESULT window keeps its OWN markers (its display, independent of the
  // source action), so total red DROPS but need not reach 0. Measure relative.
  await page.getByTestId('fv-close').click()
  await expect.poll(() => redPixels(), {
    timeout: 15_000, message: 'source overlay did not hide when the action was deselected',
  }).toBeLessThan(afterCompute)
  const afterHide = await redPixels()

  // Reselecting the action shows the source overlay again → red rises back.
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()
  await expect.poll(() => redPixels(), {
    timeout: 20_000, message: 'source overlay did not reappear when reselected',
  }).toBeGreaterThan(afterHide)
})
