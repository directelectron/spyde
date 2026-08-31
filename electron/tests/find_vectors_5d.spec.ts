/**
 * find_vectors_5d.spec.ts — Find Diffraction Vectors on a 5-D STACK.
 *
 * A 5-D result window has THREE plots: a 1-D stack navigator → a 2-D real-space
 * navigator → the diffraction pattern. Everything here is about those three
 * showing the right thing, which is exactly what headless tests could not see:
 *
 *  - the REAL-SPACE navigator must show the count map, NOT a diffraction
 *    pattern. It used to draw DPs because MultiplotManager filed the
 *    intermediate navigator in `tree.signal_plots`, so Find-Vectors installed
 *    the DP's render-frame slice function on the stack selector.
 *  - the 1-D stack navigator must be filled with per-slice vector totals. It
 *    was matched by `current_data.shape`, which is None until the first async
 *    paint lands — the miss then pushed a 2-D count map onto the 1-D line plot
 *    and left it a flat zero forever.
 *  - scrubbing the stack axis must repaint the count map for that slice.
 *
 * Screenshots land in electron/fv5d_shots/ — a blank/black nav is a failure.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
import { join } from 'path'
const {
  launchApp, raiseWindow, backendAction, waitForSubwindowCount, sigWindow, navWindows,
  dragCrosshair,
} = require('./_harness.cjs')

let ctx: Awaited<ReturnType<typeof launchApp>>
const SHOTS = join(__dirname, '..', 'fv5d_shots')

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  const { page } = ctx
  // 6 stack slices × 24×24 real space × 32×32 signal. Small enough that the
  // batch finishes in the test budget, big enough to have a real 3-level chain.
  await backendAction(page, 'load_test_data_5d', { frames: 6, nav: 24, sig: 32 })
  // 1-D stack nav + 2-D real-space nav + DP.
  await waitForSubwindowCount(page, 3, 120_000)
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(300_000)

/** Mean luminance + a crude "has structure" measure per canvas in a window. */
async function canvasStats(win: any) {
  const ifel = await win.locator('iframe').first().elementHandle()
  if (!ifel) return null
  const frame = await ifel.contentFrame()
  if (!frame) return null
  return frame.evaluate(() => {
    let best: any = null
    for (const c of Array.from(document.querySelectorAll('canvas'))) {
      const g = (c as HTMLCanvasElement).getContext('2d')
      const cv = c as HTMLCanvasElement
      if (!g || !cv.width || !cv.height) continue
      const d = g.getImageData(0, 0, cv.width, cv.height).data
      let sum = 0, n = 0, nonBlack = 0
      for (let p = 0; p < d.length; p += 4) {
        if (d[p + 3] < 20) continue
        const v = (d[p] + d[p + 1] + d[p + 2]) / 3
        sum += v; n++
        if (v > 24) nonBlack++
      }
      if (!n) continue
      const st = { w: cv.width, h: cv.height, mean: sum / n, frac: nonBlack / n, n }
      if (!best || st.n > best.n) best = st
    }
    return best
  })
}

test('5-D result: real-space navigator shows the count map, stack nav fills', async () => {
  const { page } = ctx

  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await page.screenshot({ path: join(SHOTS, '01-loaded.png'), fullPage: true })

  // Run the full-scan batch through the wizard (the path a user takes).
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()

  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('fv-compute').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 180_000, message: 'vectors result window never opened',
  }).toBeGreaterThan(before)

  await ctx.backend.waitForLog('[fv-batch] finalized', 240_000)
  await page.waitForTimeout(1500)
  await page.screenshot({ path: join(SHOTS, '02-result.png'), fullPage: true })

  // ── The 1-D stack navigator got REAL per-slice totals ────────────────────
  // A line plot's values can't be read back from a screenshot; _finalize logs
  // them at INFO precisely so this assertion is possible.
  const log = (ctx.backend.logBuffer as string[]).join('\n')
  const painted = /\[fv-5d\] time-nav painted: (\d+) slices, totals \[([^\]]*)\]/
    .exec(log)
  expect(painted, `time navigator never painted. log tail:\n${log.slice(-4000)}`)
    .not.toBeNull()
  const totals = painted![2].split(',').map((s) => Number(s.trim()))
  expect(Number(painted![1])).toBe(6)
  expect(totals.some((v) => v > 0),
         `per-slice totals are all zero: ${painted![2]}`).toBe(true)

  // ── The REAL-SPACE navigator is a count map, not a diffraction pattern ───
  // The result tree's nav windows: 1-D stack line, then the 24×24 count map.
  // A count map over a scan where every position found vectors is broadly
  // filled; a DP render is a few small disks on black (frac << 0.3).
  const navs = navWindows(page)
  const nNav = await navs.count()
  const stats: any[] = []
  for (let i = 0; i < nNav; i++) {
    stats.push({ i, ...(await canvasStats(navs.nth(i))) })
  }
  console.log('nav window canvas stats:', JSON.stringify(stats))
  const filled = stats.filter((s) => s && s.frac > 0.3)
  expect(filled.length,
         `no navigator looks like a filled count map: ${JSON.stringify(stats)}`)
    .toBeGreaterThan(0)

  // ── Scrubbing the stack axis repaints the count map for THAT slice ───────
  // The stack selector's child is the real-space navigator; its slice function
  // is now `_count_fn` (count_map_at_t) instead of the DP renderer. Click the
  // result window's 1-D navigator at ~80% width to jump slices.
  const resultNav1d = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^N-/ }) })
    .filter({ hasText: 'Vectors' }).first()
  // A 1-D navigator's marker is a VLine: it must be DRAGGED, not clicked (a
  // bare click only focuses the window — the readout updates, the marker does
  // not). dragCrosshair grabs the green marker and moves it.
  // Raise it first: a drag gets no actionability check, so a navigator sitting
  // under another window yields a short/zero move rather than a timeout.
  await raiseWindow(resultNav1d)
  const drag = await dragCrosshair(page, resultNav1d,
                                   { dx: 120, dy: 0, steps: 4, settleMs: 350 })
  expect(drag.moved, 'the stack marker never moved').toBeGreaterThan(10)
  await page.waitForTimeout(2000)
  await page.screenshot({ path: join(SHOTS, '03-scrubbed.png'), fullPage: true })

  const log2 = (ctx.backend.logBuffer as string[]).join('\n')
  const slices = [...log2.matchAll(/\[fv-5d\] count map -> slice (\d+)/g)]
    .map((m) => Number(m[1]))
  expect(slices.length,
         `the stack move never reached the count map. log tail:\n${log2.slice(-3000)}`)
    .toBeGreaterThan(0)
  expect(Math.max(...slices), 'stack move stayed on slice 0').toBeGreaterThan(0)

  // The count map must still BE a count map after the move (not black, not a DP).
  const after = []
  for (let i = 0; i < nNav; i++) after.push({ i, ...(await canvasStats(navs.nth(i))) })
  console.log('nav stats after scrub:', JSON.stringify(after))
  expect(after.filter((s: any) => s && s.frac > 0.3).length,
         `count map went blank after scrubbing: ${JSON.stringify(after)}`)
    .toBeGreaterThan(0)

  ctx.assertNoJsErrors()
})
