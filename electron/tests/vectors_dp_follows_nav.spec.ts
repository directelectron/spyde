/**
 * vectors_dp_follows_nav.spec.ts — the vectors RESULT window's diffraction
 * pattern must repaint as the navigator moves.
 *
 * Reported from the app: "the red circles track the position of the cursor but
 * the image doesn't respond". The overlay rides `index_hooks` on the active
 * sub-selector, while the rendered frame comes from that sub-selector's
 * `children` mapping — and the composite navigator selector used to keep a
 * SEPARATE mapping, which is the one every installer wrote its render function
 * into. So the circles moved over a diffraction pattern that was still slicing
 * the lazy zero placeholder.
 *
 * Only a 5-D result gets a composite here (MultiplotManager ignores
 * `selector_type` below nav level 3), which is why this is a stack test.
 *
 * Asserted against PIXELS, not state: hash the DP canvas across a real crosshair
 * drag. A window that paints once at finalize and never again passes every
 * structural check there is.
 *
 * NB the bundled `load_test_vectors` fixture CANNOT be used for this — it plants
 * the same four spots at every scan position, so every rendered frame is
 * byte-identical and a stuck display is indistinguishable from a working one.
 * `load_test_data_5d` gives each position its own bright pixel.
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, raiseWindow, backendAction, waitForSubwindowCount, dragCrosshair,
} = require('./_harness.cjs')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  const { page } = ctx
  // `spots: true` — a DETECTABLE disk whose position moves with the scan
  // position. Without it every frame is one central disk that differs only in
  // brightness, which per-frame auto-levelling erases: a stuck display would be
  // pixel-identical to a working one and this spec could never fail.
  await backendAction(page, 'load_test_data_5d',
                      { frames: 6, nav: 24, sig: 32, spots: true })
  await waitForSubwindowCount(page, 3, 120_000)
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(300_000)

/** A content hash of the biggest canvas in a window's figure iframe. */
async function frameHash(win: any): Promise<string> {
  const ifel = await win.locator('iframe').first().elementHandle()
  if (!ifel) return 'no-iframe'
  const frame = await ifel.contentFrame()
  if (!frame) return 'no-frame'
  return frame.evaluate(() => {
    let best: HTMLCanvasElement | null = null
    for (const c of Array.from(document.querySelectorAll('canvas'))) {
      const cv = c as HTMLCanvasElement
      if (!cv.width || !cv.height) continue
      if (!best || cv.width * cv.height > best.width * best.height) best = cv
    }
    if (!best) return 'no-canvas'
    const g = best.getContext('2d')
    if (!g) return 'no-ctx'
    const d = g.getImageData(0, 0, best.width, best.height).data
    let h = 2166136261 >>> 0
    let sum = 0, nz = 0
    for (let i = 0; i < d.length; i += 41) {
      h = Math.imul((h ^ d[i]) >>> 0, 16777619) >>> 0
      sum += d[i]
      if (d[i] > 12) nz++
    }
    return `${h}:${nz}:${Math.round(sum / 1000)}`
  })
}

test('5-D: the DP repaints when the scan position moves', async () => {
  const { page } = ctx
  const sigs = page.getByTestId('subwindow').filter({
    has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
  const src = sigs.first()
  await src.getByTestId('subwindow-title').click()
  await src.getByTestId('subwindow-titlebar').hover()
  await src.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()

  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('fv-compute').click()
  await expect.poll(() => page.getByTestId('subwindow').count(),
                    { timeout: 180_000 }).toBeGreaterThan(before)
  await ctx.backend.waitForLog('[fv-batch] finalized', 240_000)
  await page.waitForTimeout(2500)

  const resSig = page.getByTestId('subwindow').filter({
    has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
    .filter({ hasText: 'Vectors' }).first()
  const nav2d = page.getByTestId('subwindow').filter({
    has: page.getByTestId('window-breadcrumb').filter({ hasText: /^N-/ }) })
    .filter({ hasText: 'Vectors' }).last()

  const h0 = await frameHash(resSig)
  // Raise it first: a drag gets no actionability check, so a navigator sitting
  // under another window yields "moved 0" rather than a timeout.
  await raiseWindow(nav2d)
  const d = await dragCrosshair(page, nav2d, { dx: 46, dy: 34, steps: 4,
                                               settleMs: 500 })
  expect(d.moved, 'the 2-D navigator crosshair never moved').toBeGreaterThan(5)
  await page.waitForTimeout(1800)
  const h1 = await frameHash(resSig)
  console.log('5D DP hash across a scan move:', h0, '->', h1)
  expect(h1, 'the DP did not repaint when the scan position moved — the '
    + 'render function is installed somewhere the update path never reads')
    .not.toBe(h0)

  ctx.assertNoJsErrors()
})
