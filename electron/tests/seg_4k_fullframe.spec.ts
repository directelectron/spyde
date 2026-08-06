/**
 * seg_4k_fullframe.spec.ts — the WHOLE 4096² frame, on the real in-situ movie.
 *
 * The preview used to crop a 4k frame to its middle megapixel, because a full
 * preview cost 8.36 s (7.6 s of it the watershed) and a drawn box had to admit
 * that 15/16 of the frame had not been looked at. The mask-only path does not
 * split and does not measure, so a whole 4096² frame is ~262 ms and the crop is
 * obsolete — `_PREVIEW_PIXEL_BUDGET_MASK` covers it.
 *
 * This drives the real dataset (`InSituElectrochemGrowth`: 245 × 4096² uint8,
 * DE-Artemis hardware counting, 0.45448 nm/px) rather than a fixture, because
 * that is the frame size and the contrast the change exists for — per-pixel CNR
 * on those particles is 0.169, and no synthetic fixture reproduces that.
 *
 * Proves: the preview covers the FULL frame (no crop box), it lands in a time a
 * human would call interactive, and the mask is neither empty nor the whole
 * frame.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
import { openCaret, trainFromGroundTruth, windowIdOf } from './_seg'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = 'seg_4k_shots'
let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(900_000)

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  await ctx.page.waitForTimeout(1500)
})

test.afterAll(async () => { await ctx?.close?.() })

test('a whole 4096² frame previews without a crop', async () => {
  const { page } = ctx
  // 2.44 GB zspy, already in ~/em_database. Lazy-loaded, one frame per chunk.
  await backendAction(page, 'load_example', { name: 'InSituElectrochemGrowth' })
  await waitForSubwindowCount(page, 2, 600_000)
  // A 4096² lazy frame has to decode before anything can be painted or trained.
  await page.waitForTimeout(20_000)
  const sig = sigWindow(page)
  await page.screenshot({ path: `${SHOTS}/01-loaded-4k.png` })

  await openCaret(page, sig)
  await trainFromGroundTruth(page, await windowIdOf(sig), { timeout: 600_000 })
  await page.waitForTimeout(3000)
  await page.screenshot({ path: `${SHOTS}/02-4k-mask.png` })

  const stats = page.getByTestId('seg-preview-stats')
  // THE assertion: no crop. `data-cropped` is set from `preview_box`, which the
  // backend sends only when the frame did not fit the budget. If this is 'true'
  // the caret is showing the middle sixteenth and the change did not take.
  await expect(stats, 'the 4k frame was still cropped to a preview window')
    .toHaveAttribute('data-cropped', 'false')

  const coverage = Number(await stats.getAttribute('data-coverage'))
  expect(coverage, 'the mask is empty').toBeGreaterThan(0.001)
  expect(coverage, 'the mask covers everything — it learnt "all film"')
    .toBeLessThan(0.9)
  console.log(`[spec] full 4096² mask preview: coverage ` +
    `${(coverage * 100).toFixed(2)}%`)

  expect(backendErrorLines(ctx.backend)).toEqual([])
})
