/**
 * csb_movie.spec.ts — a real Direct Electron CSB event stream, in the app.
 *
 * Opens the 8192², 400-frame, 125.8M-event test movie, checks it arrives as a
 * scrubbable in-situ movie, drags the time slider (which integrates a fresh
 * window per position — the whole point of the format) and runs To Frames.
 *
 * Skips itself when the file is absent, so this is not a CI gate; it is the
 * "look at the pixels" check CLAUDE.md requires for anything UI.
 */
import { existsSync } from 'fs'
import { createHash } from 'crypto'
import { test, expect } from '@playwright/test'
const {
  launchApp, backendAction, waitForSubwindowCount, countColorPixels, sigWindow,
  navWindow,
} = require('./_harness.cjs')

const CSB = 'C:\\Users\\CarterFrancis\\Downloads'
  + '\\directelectron_csb-data-for-testing-de_csb-py_2026-07-22_1739'
  + '\\20240604_00001_ces_movie_234.csb'
const SHOTS = 'csb_shots'

let ctx: Awaited<ReturnType<typeof launchApp>>

test.skip(!existsSync(CSB), 'CSB test movie not present on this machine')
test.setTimeout(600_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  await backendAction(ctx.page, 'open_file', { path: CSB })
  // 8192² planes and a first-touch numba/CUDA warm-up — give it room.
  await waitForSubwindowCount(ctx.page, 2, 300_000)
})

test.afterAll(async () => {
  await ctx?.app?.close()
})

test('a CSB stream opens as a scrubbable in-situ movie', async () => {
  const { page } = ctx
  await page.screenshot({ path: `${SHOTS}/01-opened.png` })

  // The reader names the nav axis "time" in seconds, which is what tags it
  // insitu — and that is what turns on Play / Fast-Forward.
  await expect(navWindow(page)).toBeVisible({ timeout: 60_000 })
  await expect(sigWindow(page)).toBeVisible()
  await expect.poll(() => countColorPixels(page, 'bright'), {
    timeout: 240_000, message: 'the integrated plane never painted',
  }).toBeGreaterThan(0)
  await sigWindow(page).screenshot({ path: `${SHOTS}/02-plane.png` })
  ctx.assertNoJsErrors()
})

/** A cheap signature of what the signal window is actually showing.
 *
 *  Counting bright pixels page-wide is NOT enough — it stays non-zero no
 *  matter which plane is displayed (and the navigator's own canvas keeps it
 *  non-zero regardless), which is exactly how "every scrub position showed
 *  plane 0" passed a green test. This samples the displayed pixels so two
 *  different planes cannot produce the same number. */
async function planeSignature(page): Promise<string> {
  // A hash of the SIGNAL window as composited.
  //
  // Two cheaper things were tried and both lied. Counting bright pixels
  // page-wide never changes with the plane. Reading canvas pixels via
  // getImageData returns all zeros here, because the image is drawn on a
  // WebGPU canvas — the same trap the vectors-embed work hit, where buffer
  // asserts "pass" against an overlay canvas that holds nothing. A screenshot
  // is the composited truth, which is what "the image doesn't move" means.
  try {
    const buf = await sigWindow(page).screenshot()
    return createHash('sha1').update(buf).digest('hex')
  } catch {
    return ''
  }
}

test('dragging the time slider shows a DIFFERENT integrated plane', async () => {
  const { page } = ctx
  const nav = navWindow(page)
  const box = await nav.boundingBox()
  if (!box) throw new Error('no navigator window')

  await page.screenshot({ path: `${SHOTS}/03-before-scrub.png` })
  const before = await planeSignature(page)
  expect(before, 'no plane was on screen to begin with').not.toBe('')

  // Move the navigator with test_nav_drag, not the mouse. Clicking the plot
  // moves the CURSOR, not the VLine widget — the readout showed x:0.0297 while
  // the selector marker sat at 0, so a mouse-driven check "passed" on an
  // unrelated repaint. This drives the same path a real drag ends in.
  const goTo = async (plane: number) => {
    await backendAction(page, 'test_nav_drag', { targets: [[plane, 0]] })
  }
  await goTo(30)
  await expect.poll(() => planeSignature(page), {
    timeout: 60_000,
    message: 'the displayed plane did not change when the slider moved — every '
      + 'position is resolving to the same index',
  }).not.toBe(before)

  const moved = await planeSignature(page)
  await page.screenshot({ path: `${SHOTS}/04-after-scrub.png` })
  await sigWindow(page).screenshot({ path: `${SHOTS}/05-scrubbed-plane.png` })

  // And a third position differs from the second, so it is tracking rather
  // than flipping between two cached frames.
  await goTo(12)
  await expect.poll(() => planeSignature(page), {
    timeout: 60_000, message: "the plane stopped tracking after one move",
  }).not.toBe(moved)
  await sigWindow(page).screenshot({ path: `${SHOTS}/05b-third-position.png` })
  ctx.assertNoJsErrors()
})

test('Sum frames widens the point selector without moving it', async () => {
  const { page } = ctx
  await backendAction(page, 'test_nav_drag', { targets: [[25, 0]] })
  const one = await planeSignature(page)

  const dd = page.getByTestId('selector-sum-frames')
  await expect(dd, 'no Sum-frames control on a 1-D navigator').toBeVisible({
    timeout: 30_000,
  })
  await page.screenshot({ path: `${SHOTS}/09-sum-frames-control.png` })

  // Themed dropdown: click the trigger, then the option (selectOption is dead
  // on it — see Dropdown.tsx).
  await dd.click()
  await page.getByTestId('selector-sum-frames-opt-8').click()

  await expect.poll(() => planeSignature(page), {
    timeout: 60_000,
    message: 'summing 8 frames did not change the image',
  }).not.toBe(one)

  // The button must REPORT the width back. The image changing is not enough:
  // the backend applied the width fine while its confirming emit raised and
  // was swallowed, so the pointer read 8 frames and the button still said 1.
  await expect(page.getByTestId('selector-crosshair'),
    'the Point button does not show the width it is reading')
    .toContainText('8f')
  await page.screenshot({ path: `${SHOTS}/10-summed-8.png` })
  await sigWindow(page).screenshot({ path: `${SHOTS}/11-summed-plane.png` })
  ctx.assertNoJsErrors()
})

test('To Frames adds an in-situ dataset to the signal tree', async () => {
  const { page } = ctx
  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()

  const button = sig.getByTestId('action-btn-To Frames')
  await expect(button, 'the To Frames action never appeared — the '
    + 'requires_original_metadata gate did not match').toBeVisible({ timeout: 30_000 })
  await page.screenshot({ path: `${SHOTS}/06-toolbar.png` })

  const before = await page.getByTestId('subwindow').count()
  await button.click()
  await page.screenshot({ path: `${SHOTS}/07-to-frames-params.png` })
  // The param popout's Run.
  const run = page.getByRole('button', { name: /^Run$/ }).first()
  if (await run.isVisible().catch(() => false)) await run.click()

  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 300_000, message: 'To Frames never opened a new dataset',
  }).toBeGreaterThan(before)
  await page.screenshot({ path: `${SHOTS}/08-to-frames-result.png` })
  ctx.assertNoJsErrors()
})

test('the Point width can drop BELOW a plane to one raw camera frame', async () => {
  const { page } = ctx
  await backendAction(page, 'test_nav_drag', { targets: [[25, 0]] })
  const dd = page.getByTestId('selector-sum-frames')
  await expect(dd).toBeVisible({ timeout: 30_000 })

  // Back to one plane first, so the comparison is plane-vs-raw.
  await dd.click()
  await page.getByTestId('selector-sum-frames-opt-1').click()
  await page.waitForTimeout(2_000)
  const plane = await planeSignature(page)

  // The raw option only exists when the source streams finer than it loaded.
  await dd.click()
  const rawOpt = page.getByTestId('selector-sum-frames-opt-0')
  await expect(rawOpt, 'no raw-frame option on a CSB stream').toBeVisible()
  await expect(rawOpt).toContainText(/raw frame/)
  await rawOpt.click()

  await expect.poll(() => planeSignature(page), {
    timeout: 60_000,
    message: 'one raw frame looks identical to the 8-frame plane',
  }).not.toBe(plane)

  // And the button says which it is reading.
  await expect(page.getByTestId('selector-crosshair')).toContainText('raw')
  await page.screenshot({ path: `${SHOTS}/12-raw-frame.png` })
  await sigWindow(page).screenshot({ path: `${SHOTS}/13-raw-frame-plane.png` })

  // Going back to a plane restores the integrated view.
  await dd.click()
  await page.getByTestId('selector-sum-frames-opt-1').click()
  await expect.poll(() => planeSignature(page), {
    timeout: 60_000, message: 'leaving raw mode did not restore the plane',
  }).toBe(plane)
  ctx.assertNoJsErrors()
})
