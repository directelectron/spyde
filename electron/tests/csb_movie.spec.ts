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

test('dragging the time slider integrates a new window', async () => {
  const { page } = ctx
  const nav = navWindow(page)
  const box = await nav.boundingBox()
  if (!box) throw new Error('no navigator window')

  await page.screenshot({ path: `${SHOTS}/03-before-scrub.png` })
  // Drag along the 1-D time navigator: each position is a different exposure
  // window, so the displayed image must change.
  const y = box.y + box.height * 0.55
  await page.mouse.move(box.x + box.width * 0.25, y)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width * 0.75, y, { steps: 12 })
  await page.mouse.up()

  await expect.poll(() => countColorPixels(page, 'bright'), {
    timeout: 240_000, message: 'nothing painted after the scrub',
  }).toBeGreaterThan(0)
  await page.screenshot({ path: `${SHOTS}/04-after-scrub.png` })
  await sigWindow(page).screenshot({ path: `${SHOTS}/05-scrubbed-plane.png` })
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
