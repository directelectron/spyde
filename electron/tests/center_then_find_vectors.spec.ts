/**
 * center_then_find_vectors.spec.ts: an overlay belongs to the node on screen.
 *
 * Centre the diffraction pattern (a mapped child node), THEN open Find Vectors:
 * the live preview and, after Compute, the found-vectors circles must draw on
 * the centred pattern the window shows, and the navigator must keep dragging at
 * its usual per-move cost while the preview is attached. Before this contract,
 * Find Vectors computed on the tree root and its circles missed the centred
 * disks; the preview also computed a whole dask block per move on lazy data.
 *
 * Real Dask + the lazy 3×3-chunk synthetic scan (the eager si-grains dataset
 * skips every cache and measures nothing, CLAUDE.md, region drag trap 1).
 * The per-move numbers come from the backend's own [NAV-PROFILE] lines
 * (SPYDE_NAV_PROFILE=1). Screenshots land in electron/center_fv_shots/.
 */
import { test, expect } from '@playwright/test'
import * as fs from 'fs'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, countColorPixels, sigWindow,
  navWindow, dragCrosshair,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'center_fv_shots')
let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  fs.mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({
    dask: true,
    env: { SPYDE_LOG_LEVEL: 'INFO', SPYDE_NAV_PROFILE: '1' },
  })
  const { page } = ctx
  await backendAction(page, 'load_test_data_lazy_chunked')
  await waitForSubwindowCount(page, 2, 120_000)
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(300_000)

function profileStats(lines: string[]) {
  const durations: number[] = []
  for (const l of lines) {
    const m = l.match(/total=([\d.]+)ms/)
    if (m) durations.push(parseFloat(m[1]))
  }
  durations.sort((a, b) => a - b)
  const med = durations.length ? durations[Math.floor(durations.length / 2)] : NaN
  const p95 = durations.length
    ? durations[Math.min(durations.length - 1, Math.floor(durations.length * 0.95))]
    : NaN
  return { n: durations.length, med, p95 }
}

test('centre the pattern, then Find Vectors draws on the centred node', async () => {
  const { page, backend } = ctx
  const red = () => countColorPixels(page, 'red')
  const sig = sigWindow(page)
  const nav = navWindow(page)
  await page.screenshot({ path: `${SHOTS}/01-loaded.png` })

  // ---- Center Zero Beam → a "Centered" child node is displayed ----------
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Center Zero Beam').click()
  await expect(page.getByTestId('center-zero-beam-wizard')).toBeVisible()
  await page.getByTestId('czb-center').click()
  // The "Centered" node in the Workflow panel is the completion signal; the
  // status text is overwritten by the navigator recompute that follows.
  await expect(page.getByTestId(/^tree-node-Centered/).first())
    .toBeVisible({ timeout: 90_000 })
  await page.getByTestId('czb-close').click()
  await page.waitForTimeout(500)
  await page.screenshot({ path: `${SHOTS}/02-centred.png` })

  // ---- a drag with NO overlay: the baseline per-move cost ----------------
  const before = backend.logBuffer.length
  await dragCrosshair(page, nav, { dx: 6, dy: 4, steps: 8, settleMs: 120 })
  const baseline = profileStats(
    backend.logBuffer.slice(before).filter((l: string) => l.includes('[NAV-PROFILE]')))
  console.log(`[center-fv] baseline drag: n=${baseline.n} med=${baseline.med}ms p95=${baseline.p95}ms`)

  // ---- Find Vectors preview on the CENTRED node ---------------------------
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()
  await expect.poll(red, {
    timeout: 60_000, message: 'live preview drew no peak markers on the centred DP',
  }).toBeGreaterThan(0)
  await page.screenshot({ path: `${SHOTS}/03-preview-on-centred.png` })

  // ---- a drag WITH the preview attached: the same per-move cost -----------
  const mid = backend.logBuffer.length
  await dragCrosshair(page, nav, { dx: -6, dy: -4, steps: 8, settleMs: 120 })
  const withPreview = profileStats(
    backend.logBuffer.slice(mid).filter((l: string) => l.includes('[NAV-PROFILE]')))
  console.log(`[center-fv] drag with preview: n=${withPreview.n} med=${withPreview.med}ms p95=${withPreview.p95}ms`)
  await page.screenshot({ path: `${SHOTS}/04-preview-after-drag.png` })
  expect(await red()).toBeGreaterThan(0)
  // No move went to the async tier and nothing computed a dask block for the
  // preview: the profile lines say so, and a block compute is not a nav read.
  const asyncLines = backend.logBuffer.slice(mid).filter((l: string) => /async-submit/.test(l))
  expect(asyncLines, 'a preview move was routed async').toEqual([])
  if (baseline.n && withPreview.n) {
    // Generous: the preview's own work runs on its worker thread, so the base
    // read should not move; allow noise, catch a block-per-move regression
    // (that is ~100 ms, an order of magnitude above any noise).
    expect(withPreview.med).toBeLessThan(Math.max(baseline.med * 3, baseline.med + 20))
  }

  // ---- Compute → the batch runs on the centred node --------------------------
  // The wizard attaches the source overlay HIDDEN after Compute (the DP stays
  // clean until the caret is reopened), so the red pixels counted here are the
  // vectors window's circles; the source overlay's geometry on the centred
  // frame is pinned by test_overlay_follows_displayed_node.py.
  const windows = await page.getByTestId('subwindow').count()
  await page.getByTestId('fv-compute').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 120_000, message: 'vectors result window never opened',
  }).toBeGreaterThan(windows)
  await backend.waitForLog('[fv-batch] finalized', 120_000)
  await sig.getByTestId('subwindow-title').click()
  await expect.poll(red, {
    timeout: 30_000, message: 'the vectors window drew no circles after Compute',
  }).toBeGreaterThan(0)
  await page.screenshot({ path: `${SHOTS}/05-found-on-centred.png` })

  const after = backend.logBuffer.length
  await dragCrosshair(page, nav, { dx: 6, dy: 0, steps: 6, settleMs: 120 })
  const withOverlay = profileStats(
    backend.logBuffer.slice(after).filter((l: string) => l.includes('[NAV-PROFILE]')))
  console.log(`[center-fv] drag with found-vectors overlay: n=${withOverlay.n} med=${withOverlay.med}ms p95=${withOverlay.p95}ms`)
  await page.screenshot({ path: `${SHOTS}/06-found-after-drag.png` })
  expect(await red()).toBeGreaterThan(0)

  ctx.assertNoJsErrors()
})
