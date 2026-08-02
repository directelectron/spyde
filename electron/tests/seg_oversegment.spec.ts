/**
 * seg_oversegment.spec.ts — the reported failure, reproduced and pinned.
 *
 * Reported with a screenshot on a real in-situ movie: "14028 particles in this
 * region", the whole preview window painted a flat sheet of green, the renderer
 * hung, and the Scribble tab unusable because the image you have to paint on was
 * under that sheet.
 *
 * THREE separate defects produced that one screenshot, and every existing spec
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
 *   3. A failed threshold was reported as a result. Otsu on a low-contrast frame
 *      has no bimodal histogram to find, lands inside the noise, and the split
 *      shatters the support film — but the caret said "14028 particles", which
 *      reads as a bad answer rather than as no answer, and sends the user to
 *      sliders that cannot fix it.
 *
 * `noise: 0.35` is what makes the fixture fail this way; at its default 0.015 it
 * is clean and a global threshold works fine on it. 1200² is above anyplotlib's
 * 1024 tile threshold, so the tiled path is the one under test — that pairing is
 * the entire point, and either one alone reproduces nothing.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
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
  await backendAction(page, 'load_test_data_particles',
    { frames: 4, size: [1200, 1200], noise: 0.35 })
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

test('a noise frame over-segments, and the caret NAMES it instead of counting it', async () => {
  const { page } = ctx
  await openCaret()

  const stats = page.getByTestId('seg-preview-stats')
  await expect.poll(async () => Number(await stats.getAttribute('data-count')), {
    timeout: 180_000, message: 'seg_preview never reached the caret',
  }).toBeGreaterThan(0)

  const count = Number(await stats.getAttribute('data-count'))
  // The fixture has to actually FAIL or this spec proves nothing — the whole
  // reason the bug shipped is that the clean fixture never got here.
  expect(count, `only ${count} instances; noise:0.35 did not over-segment, so `
    + 'this spec is not exercising the reported failure').toBeGreaterThan(200)

  // The verdict, not just the number. "14028 particles" reads as an answer.
  await expect(stats).toHaveAttribute('data-failed', 'true')
  const notice = page.getByTestId('seg-threshold-failed')
  await expect(notice).toBeVisible()
  await expect(notice).toContainText('landed inside the noise')
  // ...and it points at the engine that DOES work on this data (plan §0.9),
  // rather than leaving the user on sliders that cannot fix a bad threshold.
  await expect(notice).toContainText('Scribble')

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

test('Scribble stays usable: the image is not buried under the failed preview', async () => {
  const { page } = ctx
  await page.getByTestId('seg-tab-scribble').click()
  await expect(page.getByTestId('seg-class-strip')).toBeVisible({ timeout: 30_000 })

  // The reported symptom, in one number: you cannot paint on an image you
  // cannot see. The untrained Scribble engine has produced no result, so the
  // previous engine's drawing must be GONE — not merely thinner. It used to
  // survive because `show_preview_window` cleared the vector outlines and left
  // the raster mask, and above 100 instances the mask is the whole drawing.
  await expect.poll(() => countColorPixels(page, 'green'), {
    timeout: 30_000,
    message: 'the classical result is still drawn on the Scribble tab, over '
      + 'the image the user has to paint on',
  }).toBeLessThan(2000)

  await page.screenshot({ path: `${SHOTS}/04-scribble-usable.png` })
  ctx.assertNoJsErrors()
})
