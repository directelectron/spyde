/**
 * seg_second_window.spec.ts — close a signal window, segment a DIFFERENT one.
 *
 * Reported: "When I close a signal plot and go back to try to segment a
 * different signal plot it doesn't work."
 *
 * The BACKEND is fine — driving `seg_open`/`seg_autolabel`/`seg_train` against
 * a second tree after closing the first passes in python. So whatever breaks is
 * renderer state that outlives the closed window: the per-window tuning store,
 * the caret's figure-id set, the brush strip, or the lifecycle hook's
 * open/close bookkeeping. This drives the real thing.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
import { trainFromGroundTruth, windowIdOf } from './_seg'
const { launchApp, backendAction, waitForSubwindowCount, sigWindow } =
  require('./_harness.cjs')

const SHOTS = 'seg_second_window_shots'
let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(420_000)

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  await ctx.page.waitForTimeout(1500)
})

test.afterAll(async () => { await ctx?.app?.close() })

/** Every `S-` signal subwindow currently open. */
function sigWindows() {
  const { page } = ctx
  return page.getByTestId('subwindow').filter({
    has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }),
  })
}

/** A window by its BACKEND id.
 *
 *  Not `.nth(i)`: a Playwright locator is lazy and re-evaluates at use, so an
 *  index captured before a close silently addresses a different window (or
 *  nothing) afterwards — which is the whole subject of this spec, and it cost
 *  a run to notice. `data-window-id` is stable across the close. */
function winById(id: number) {
  return ctx.page.locator(`[data-testid="subwindow"][data-window-id="${id}"]`)
}

async function openCaretOn(win: ReturnType<typeof sigWindows>) {
  const { page } = ctx
  await win.getByTestId('subwindow-title').click()
  await win.getByTestId('subwindow-titlebar').hover()
  await win.getByTestId('action-btn-Segment Particles').click()
  await expect(page.getByTestId('segment-wizard')).toBeVisible({ timeout: 30_000 })
}

test('segment a second signal window after closing the first', async () => {
  const { page } = ctx

  // TWO datasets, so a second signal window genuinely exists.
  await backendAction(page, 'load_test_data_particles', { frames: 4 })
  await waitForSubwindowCount(page, 2, 120_000)
  await page.waitForTimeout(2000)
  await backendAction(page, 'load_test_data_particles', { frames: 4 })
  await waitForSubwindowCount(page, 4, 120_000)
  await page.waitForTimeout(2500)

  expect(await sigWindows().count(), 'expected two signal windows').toBe(2)
  const idFirst = await windowIdOf(sigWindows().nth(0))
  const idSecond = await windowIdOf(sigWindows().nth(1))
  expect(idFirst).not.toBe(idSecond)
  const first = winById(idFirst)
  const second = winById(idSecond)

  // Use the FIRST one properly, so there is real state to leak.
  await openCaretOn(first)
  await trainFromGroundTruth(page, idFirst)
  await page.screenshot({ path: `${SHOTS}/01-first-trained.png` })

  // Close it — the caret goes with the window.
  await first.getByTestId('close-btn').click()
  await expect(page.getByTestId('segment-wizard')).toHaveCount(0, { timeout: 30_000 })
  await page.waitForTimeout(1500)
  await page.screenshot({ path: `${SHOTS}/02-first-closed.png` })

  // ...and now the OTHER one. This is the reported step.
  await openCaretOn(second)
  await page.screenshot({ path: `${SHOTS}/03-second-caret.png` })

  // It must come up UNTRAINED with its own empty class list — not carrying the
  // closed window's trained head or its labelled-pixel counts.
  await expect(page.getByTestId('seg-scribble-note')).toBeVisible()
  await expect(page.getByTestId('seg-class-pixels-0')).toHaveText(/^!?\s*0$/)
  await expect(page.getByTestId('seg-class-strip')).toBeVisible({ timeout: 30_000 })

  // And it must actually WORK.
  await trainFromGroundTruth(page, idSecond)
  await expect(page.getByTestId('seg-trained-note')).toContainText(/Trained on \d+ px/)
  await expect.poll(
    async () => Number(await page.getByTestId('seg-preview-stats')
      .getAttribute('data-count')),
    { timeout: 120_000, message: 'the second window never previewed' },
  ).toBeGreaterThan(0)

  await page.screenshot({ path: `${SHOTS}/04-second-trained.png` })
  ctx.assertNoJsErrors()
})

test('after a full RUN and a close, a third window still segments', async () => {
  const { page } = ctx

  // The realistic session: actually run the batch, which opens a RESULT window
  // and leaves `_seg_batch_running` / `result_tree` on the source tree, THEN
  // close things and try again. The plain close path above is clean, so if the
  // report is reproducible the state that survives a RUN is the next suspect.
  await page.getByTestId('seg-run').click()
  await expect(page.getByTestId('seg-status')).toHaveText(/Segmenting the movie/)
  await expect
    .poll(() => page.getByTestId('action-btn-Particle Overlay').count(),
      { timeout: 180_000, message: 'the batch never finalized' })
    .toBeGreaterThan(0)
  await page.waitForTimeout(1500)
  await page.screenshot({ path: `${SHOTS}/06-after-run.png` })

  // Close EVERY signal window that now exists — source and result alike.
  for (let guard = 0; guard < 8; guard++) {
    const n = await sigWindows().count()
    if (n === 0) break
    await sigWindows().first().getByTestId('close-btn').click()
    await page.waitForTimeout(600)
  }
  await page.waitForTimeout(1500)
  await page.screenshot({ path: `${SHOTS}/07-all-closed.png` })

  // A fresh dataset, from nothing.
  await backendAction(page, 'load_test_data_particles', { frames: 4 })
  await expect.poll(() => sigWindows().count(),
    { timeout: 120_000, message: 'the fresh dataset never opened' })
    .toBeGreaterThan(0)
  await page.waitForTimeout(2500)

  const id = await windowIdOf(sigWindows().first())
  await openCaretOn(winById(id))
  await expect(page.getByTestId('seg-scribble-note')).toBeVisible()
  await expect(page.getByTestId('seg-class-strip')).toBeVisible({ timeout: 30_000 })
  await trainFromGroundTruth(page, id)
  await expect.poll(
    async () => Number(await page.getByTestId('seg-preview-stats')
      .getAttribute('data-count')),
    { timeout: 120_000, message: 'a fresh window after a run + close never previewed' },
  ).toBeGreaterThan(0)

  await page.screenshot({ path: `${SHOTS}/08-third-window.png` })
  ctx.assertNoJsErrors()
})

test('a REAL Shift+drag paints on the second window too', async () => {
  const { page } = ctx
  const win = sigWindows().first()          // only one left after the close
  const box = await win.locator('iframe').first().boundingBox()
  expect(box).toBeTruthy()

  const read = async () => {
    const t = await page.getByTestId('seg-class-pixels-0').textContent()
    return Number((t ?? '').replace(/[^\d]/g, '')) || 0
  }
  const before = await read()

  const y = box!.y + box!.height * 0.5
  await page.keyboard.down('Shift')
  await page.mouse.move(box!.x + box!.width * 0.3, y)
  await page.mouse.down()
  for (let i = 1; i <= 8; i++) {
    await page.mouse.move(box!.x + box!.width * (0.3 + 0.035 * i), y)
    await page.waitForTimeout(30)
  }
  await page.mouse.up()
  await page.keyboard.up('Shift')

  await expect.poll(read, {
    timeout: 30_000,
    message: 'Shift+drag painted nothing on the second window — the brush is '
      + 'armed on the closed window, or not armed at all',
  }).toBeGreaterThan(before)

  await page.screenshot({ path: `${SHOTS}/05-second-painted.png` })
  ctx.assertNoJsErrors()
})
