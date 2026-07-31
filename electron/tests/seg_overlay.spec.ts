/**
 * seg_overlay.spec.ts — the live segmentation overlay, on a TILED frame.
 *
 * The bug this exists for: the caret reported "106 particles on this frame" and
 * nothing was drawn on a real 4096² movie. `_preview` was calling
 * `set_overlay_mask` and the mask WAS being pushed (`[plot] overlay mask set:
 * N px` in the backend log) — but a signal frame at or above 1024 px goes
 * through anyplotlib's GPU tile display, whose base image is drawn by WebGPU,
 * and the mask composites onto the Canvas2D context underneath it. Invisible.
 * Vector markers draw over the GPU base correctly, which is why the brush
 * strokes showed up in the same screenshot that had no overlay.
 *
 * So the frame size is the whole point of this spec. `load_test_data_particles`
 * defaults to 96×112 — BELOW the tile threshold — which is exactly why the
 * existing `segment_wizard.spec.ts` never caught this. Here it is loaded at
 * 1200² so the tiled path is the one under test.
 *
 * What it proves:
 *   1. After a preview finds particles, outlines are actually DRAWN (green
 *      pixels appear on the figure canvas, and only after the preview lands).
 *   2. The outlines FOLLOW THE NAVIGATOR — scrolling to another frame
 *      re-previews and repaints, without the caret being touched.
 *   3. Closing the caret takes the overlay with it.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow,
  countColorPixels, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = 'seg_overlay_shots'
let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(420_000)

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  // 1200² is ABOVE anyplotlib's 1024 tile threshold — the display path the 4k
  // dataset uses, and the one the 96×112 default never exercises.
  await backendAction(page, 'load_test_data_particles', { frames: 6, size: [1200, 1200] })
  await waitForSubwindowCount(page, 2, 180_000)
  await page.waitForTimeout(3000)
})

test.afterAll(async () => {
  await ctx?.app?.close()
})

async function openCaret() {
  const { page } = ctx
  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Segment Particles').click()
  await expect(page.getByTestId('segment-wizard')).toBeVisible()
  return sig
}

/** The caret's own monotonic preview counter — the reliable "it re-ran" signal.
 *  The COUNT is not: two frames can legitimately find the same number. */
async function previewSeq(): Promise<number> {
  const attr = await ctx.page.getByTestId('seg-preview-stats').getAttribute('data-seq')
  return Number(attr ?? 0)
}
async function previewCount(): Promise<number> {
  const attr = await ctx.page.getByTestId('seg-preview-stats').getAttribute('data-count')
  return Number(attr ?? -1)
}

test('outlines are drawn on a tiled frame once the preview lands', async () => {
  const { page } = ctx

  // BEFORE the caret exists there is no overlay, so this is the baseline the
  // "it drew something" assertion is measured against rather than a bare >0.
  const greenBefore = await countColorPixels(page, 'green')
  await page.screenshot({ path: `${SHOTS}/01-before-caret.png` })

  await openCaret()
  await expect.poll(previewCount, {
    timeout: 180_000, message: 'seg_preview never reached the caret',
  }).toBeGreaterThan(0)
  // The push is a figure update marshalled onto the main loop; give it a beat
  // to reach the canvas after the count line has updated.
  await expect.poll(() => countColorPixels(page, 'green'), {
    timeout: 60_000,
    message: 'the preview found particles but drew no outlines — the overlay '
      + 'never reached the figure (this is the GPU-tile bug)',
  }).toBeGreaterThan(greenBefore + 200)

  await page.screenshot({ path: `${SHOTS}/02-outlines.png` })
  await sigWindow(page).screenshot({ path: `${SHOTS}/03-outlines-window.png` })
})

test('the outlines follow the navigator', async () => {
  const { page } = ctx

  const seq0 = await previewSeq()
  expect(seq0, 'no preview to follow').toBeGreaterThan(0)

  // Drive the NAVIGATOR, not the caret — this is the "scroll through the
  // dataset and watch it update" path, and nothing subscribed to it before.
  // `test_nav_drag` moves the real navigation SELECTOR, which is where the
  // wizard's index hook lives; clicking the navigator canvas would not reach it
  // anyway (the canvas is inside a nested figure iframe).
  await backendAction(page, 'test_nav_drag', { targets: [[3, 0]] })

  await expect.poll(previewSeq, {
    timeout: 180_000,
    message: 'moving the navigator did not re-preview — the overlay is stuck '
      + 'on whichever frame was showing when the caret was opened',
  }).toBeGreaterThan(seq0)

  // And it still draws after following.
  expect(await countColorPixels(page, 'green'),
    'the overlay vanished after the navigator moved').toBeGreaterThan(200)

  await page.screenshot({ path: `${SHOTS}/04-followed-navigator.png` })
  await sigWindow(page).screenshot({ path: `${SHOTS}/05-followed-window.png` })

  const errors = backendErrorLines(ctx.backend)
  expect(errors, `backend errors:\n${errors.join('\n')}`).toEqual([])
  ctx.assertNoJsErrors()
})

test('closing the caret removes the overlay', async () => {
  const { page } = ctx
  const before = await countColorPixels(page, 'green')
  expect(before, 'nothing drawn to remove').toBeGreaterThan(200)

  await page.getByTestId('seg-close').click()
  await expect(page.getByTestId('segment-wizard')).toHaveCount(0)

  await expect.poll(() => countColorPixels(page, 'green'), {
    timeout: 60_000,
    message: 'the outlines outlived the caret that owns them',
  }).toBeLessThan(200)

  await page.screenshot({ path: `${SHOTS}/06-closed.png` })
  ctx.assertNoJsErrors()
})
