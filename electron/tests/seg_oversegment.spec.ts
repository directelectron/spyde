/**
 * seg_oversegment.spec.ts — the reported failure, reproduced and pinned.
 *
 * Reported with a screenshot on a real in-situ movie: "14028 particles in this
 * region", the whole preview window painted a flat sheet of green, the renderer
 * hung, and the caret unusable because the image you have to paint on was under
 * that sheet.
 *
 * Three separate defects produced that one screenshot, and every existing spec
 * was green throughout because the bundled fixture is small, clean and
 * high-contrast:
 *
 *   1. The raster overlay was UNREACHABLE on a tiled frame. `_set_raster_overlay`
 *      reduced the mask to the overview grid (what the renderer wants, since it
 *      checks `bytes.length === (base_width||image_width) * …`), anyplotlib's
 *      `set_overlay_mask` validated that against `image_width` — the FULL native
 *      frame in tile mode — and raised. SpyDE logged the ValueError at DEBUG and
 *      fell back to one filled polygon per instance. At 14028 instances that is
 *      the hang, and thousands of overlapping translucent fills are the green
 *      sheet. So the path added to avoid N polygons could never run on the only
 *      frames big enough to need it.
 *   2. Nothing capped the polygon fallback.
 *   3. Segmenting the support film was reported as a RESULT. It was otsu landing
 *      inside the noise then; that engine is deleted, but the failure is not
 *      engine-specific — an under-trained head that learnt "film is particle"
 *      produces the same thing, which is why the verdict tests the measured
 *      OUTPUT (instance count AND coverage) rather than anything about a
 *      threshold. "14028 particles" reads as a bad answer rather than as no
 *      answer, and sends the user to sliders that cannot fix it.
 *
 * `noise: 0.35` is what makes the fixture fail this way; at its default 0.015 it
 * is clean and the head trains cleanly on it. 1200² is above anyplotlib's 1024
 * tile threshold, so the tiled path is the one under test — that pairing is the
 * entire point, and either one alone reproduces nothing.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
import { trainFromGroundTruth, windowIdOf } from './_seg'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow, countColorPixels,
} = require('./_harness.cjs')

const SHOTS = 'seg_oversegment_shots'
let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(420_000)

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  // 1024², and BOTH halves of that number matter:
  //   * `>= 1024` on an edge is anyplotlib's tile threshold (`_GPU_TILE_MIN_EDGE`),
  //     so this is the tiled display path — the one the raster-overlay bug lived on.
  //   * `1024*1024` is exactly `_PREVIEW_PIXEL_BUDGET`, so the preview segments the
  //     WHOLE frame rather than a centred crop.
  // That second point is load-bearing here and was not obvious: `particle_movie`
  // does NOT scale its particles with `shape` — they stay at their original
  // 16..102 px coordinates with 3-9 px radii. At 1200² they therefore sit in a
  // corner OUTSIDE the centred 1024² preview crop, the trained head correctly
  // finds nothing in the crop, and the spec fails for a reason that has nothing
  // to do with what it is testing.
  await backendAction(page, 'load_test_data_particles',
    { frames: 4, size: [1024, 1024], noise: 0.35 })
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

/** Open, MIS-label (film taught as particle) and Train — the realistic way a
 *  user produces the over-segmentation the coverage verdict is for. */
async function openAndTrain() {
  const { page } = ctx
  const sig = await openCaret()
  await trainFromGroundTruth(page, await windowIdOf(sig), { mislabel: true })
  return sig
}

test('a noise frame over-segments, and the caret NAMES it instead of counting it', async () => {
  const { page } = ctx
  await openAndTrain()

  const stats = page.getByTestId('seg-preview-stats')
  await expect.poll(async () => Number(await stats.getAttribute('data-count')), {
    timeout: 180_000, message: 'seg_preview never reached the caret',
  }).toBeGreaterThan(0)

  const count = Number(await stats.getAttribute('data-count'))
  const coverage = Number(await stats.getAttribute('data-coverage'))
  // The head has to actually FAIL or this spec proves nothing — the whole
  // reason the bug shipped is that the clean fixture never got here.
  expect(coverage,
    `coverage ${(100 * coverage).toFixed(0)}%: the mislabelled head did not `
    + 'return the film, so this spec is not exercising the reported failure')
    .toBeGreaterThan(0.6)
  // ...and note the SHAPE. A bad classifier merges the film into a few hundred
  // large blobs; the deleted threshold engine shattered it into thousands of
  // small ones. The verdict has to catch both, and a rule tuned on the count
  // alone caught only the second — measured: 228 instances over ~the whole
  // frame sailed under a 500-instance bar while looking like a green sheet.
  expect(count, 'no instances at all').toBeGreaterThan(0)

  // The verdict, not just the number. "14028 particles" reads as an answer.
  await expect(stats).toHaveAttribute('data-failed', 'true')
  const notice = page.getByTestId('seg-threshold-failed')
  await expect(notice).toBeVisible()
  await expect(notice).toContainText('support film')
  // ...and it says the ONE thing that fixes it. Size and shape sliders
  // cannot: over-segmented film is small AND round.
  await expect(notice).toContainText('re-train')

  await page.screenshot({ path: `${SHOTS}/01-threshold-failed.png` })
  await page.getByTestId('segment-wizard').screenshot({
    path: `${SHOTS}/02-caret.png` })
  ctx.assertNoJsErrors()
})

test('the overlay is ONE mask, not thousands of polygons, and does not blanket the frame', async () => {
  const { page } = ctx

  // THE regression guard for defect 1. This line is emitted at WARNING by
  // `_set_raster_overlay`'s except branch, so it reaches the harness's stderr
  // buffer with SPYDE_LOG_LEVEL=INFO. Before the fix it fired on every preview
  // of this frame — and at DEBUG, where nobody would ever see it.
  const failed = (ctx.backend.logBuffer as string[])
    .filter((l) => l.includes('raster overlay failed'))
  expect(failed, `the raster overlay fell back to polygons:\n${failed.join('\n')}`)
    .toEqual([])

  // The cap must not have fired either — reaching it means the raster path was
  // unavailable, which is the bug wearing a seatbelt rather than the bug fixed.
  const capped = (ctx.backend.logBuffer as string[])
    .filter((l) => l.includes('exceeds the'))
  expect(capped, `the outline draw cap fired, so the mask never drew:\n`
    + capped.join('\n')).toEqual([])

  // And the visual half: an overlay is drawn, but it is NOT the solid sheet of
  // the screenshot. A mask that covers essentially the whole window carries no
  // information — you cannot see the data under it, which is what made the
  // Scribble tab unusable.
  const green = await countColorPixels(page, 'green')
  const sig = sigWindow(page)
  const box = await sig.boundingBox()
  const area = box ? box.width * box.height : 1
  expect(green, 'no overlay was drawn at all').toBeGreaterThan(100)
  expect(green / area,
    `the overlay covers ${(100 * green / area).toFixed(0)}% of the window — `
    + 'that is the solid-green sheet, not an overlay').toBeLessThan(0.5)

  await page.screenshot({ path: `${SHOTS}/03-overlay-not-a-sheet.png` })
  ctx.assertNoJsErrors()
})

test('closing the caret takes the RASTER overlay with it', async () => {
  const { page } = ctx

  // `seg_overlay.spec.ts` already pins close-clears-overlay, but on a frame with
  // few enough instances to be drawn as OUTLINES. This is the other drawing
  // route, and it has its own teardown (`_clear_raster_overlay`) that was
  // missed once already: `show_preview_window` cleared the vector group and
  // left the mask, so a dead result stayed on the image. Above
  // `_RASTER_ABOVE` the mask IS the whole drawing, so a teardown that forgets
  // it leaves the frame unusable rather than merely untidy.
  const before = await countColorPixels(page, 'green')
  expect(before, 'no raster overlay to clear — this test needs one drawn')
    .toBeGreaterThan(100)

  await page.getByTestId('seg-close').click()
  await expect(page.getByTestId('segment-wizard')).toHaveCount(0)

  await expect.poll(() => countColorPixels(page, 'green'), {
    timeout: 30_000,
    message: 'the raster mask outlived the caret that drew it',
  }).toBeLessThan(2000)

  await page.screenshot({ path: `${SHOTS}/04-closed-clean.png` })
  ctx.assertNoJsErrors()
})
