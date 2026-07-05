/**
 * vector_overlay.spec.ts — the Find-Diffraction-Vectors WIZARD must show a LIVE
 * found-peaks preview as you open it. Compute collapses the caret, drops the
 * preview, and leaves the SOURCE DP clean (the persistent overlay attaches
 * hidden); the RESULT window draws its own markers. Reopening the caret brings
 * the live preview back (a single marker set — no duplicated peaks). Launches
 * the real backend, loads lazy 4D data with a bright central disk, and asserts
 * on saturated-RED marker pixels (the grayscale image has none).
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

  // Compute → the wizard caret COLLAPSES back into the toolbar button, the
  // live preview is dropped, and a new vectors window opens. The SOURCE DP is
  // left clean (the persistent overlay attaches hidden); the RESULT window
  // draws its own markers once the vectors attach.
  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('fv-compute').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeHidden()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 60_000, message: 'vectors window never opened',
  }).toBeGreaterThan(before)
  await expect.poll(() => redPixels(), {
    timeout: 60_000, message: 'no vector markers on the RESULT window after compute',
  }).toBeGreaterThan(0)
  const resultOnly = await redPixels()   // result-window markers only

  // Reopening the caret restores a live preview on the source DP (a SINGLE
  // marker set — the preview supersedes the batch overlay, so peaks are never
  // double-drawn) → red rises above the result-window-only level. Raise the
  // source window first: the result window opened focused ON TOP of its
  // toolbar (toolbar shares the window z-level by design).
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()
  await expect.poll(() => redPixels(), {
    timeout: 20_000, message: 'live preview did not reappear when reselected',
  }).toBeGreaterThan(resultOnly)
})
