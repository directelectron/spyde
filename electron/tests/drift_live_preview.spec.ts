/**
 * drift_live_preview.spec.ts — the alignment box tracks the cursor, on pixels.
 *
 * Headless timings cannot see this. `benchmark_drift_latency.py` says a preview
 * step is 13 ms, but a step that is fast and never fires, or fires and paints the
 * same image every time, produces exactly the same number. The claim here is
 * visual and behavioural: DRAG the box, and both the preview panels and the gain
 * readout change WHILE the pointer is down — not on release.
 *
 * Screenshots are written per drag stage so the sequence can be read, because
 * "it updates" is a statement about a sequence of frames rather than about any
 * one of them.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = 'drift_live_shots'
let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(900_000)

test.beforeAll(async () => {
  test.setTimeout(900_000)          // hooks carry their own timeout
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({
    dask: true,
    env: { SPYDE_LOG_LEVEL: 'INFO', SPYDE_ACTION_PROFILE: '1' },
  })
  const { page } = ctx
  await page.waitForTimeout(1500)
  // The PARTICLE movie, because it has real stamped drift. `load_test_data_movie`
  // has none — its frames differ by a moving index band, not a translation — so a
  // correct solve returns zero shift there and the gain is exactly 1.000 for every
  // box. That makes it impossible to tell a live preview from a dead one, which is
  // precisely what this spec has to distinguish. Sized 1024² so the fixed 512 box
  // has room to be dragged over genuinely different content.
  await backendAction(page, 'load_test_data_particles',
    { frames: 12, size: [1024, 1024] })
  await waitForSubwindowCount(page, 2, 300_000)
  await page.waitForTimeout(2000)
})

test.afterAll(async () => {
  await ctx?.app?.close()
})

test('the gain readout tracks the box WHILE it is dragged', async () => {
  const { page } = ctx
  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Drift Correction').click()
  await expect(page.getByTestId('drift-wizard')).toBeVisible()
  await waitForSubwindowCount(page, 3, 300_000)

  const gain = () => page.getByTestId('drift-roi-readout').getAttribute('data-gain')
  await expect.poll(gain, { timeout: 300_000, message: 'no first preview' }).toBeTruthy()
  await page.waitForTimeout(1200)
  await page.screenshot({ path: `${SHOTS}/01-open.png` })

  const box = await sig.locator('iframe').first().boundingBox()
  expect(box).toBeTruthy()
  const cx = box!.x + box!.width / 2
  const cy = box!.y + box!.height / 2

  // Drag WITHOUT releasing, sampling the readout as we go. The assertion is that
  // it changes mid-drag; a settle-on-release design would hold one value here.
  await page.mouse.move(cx, cy)
  await page.mouse.down()
  const seen: string[] = []
  for (let i = 1; i <= 10; i++) {
    await page.mouse.move(cx - i * 9, cy - i * 6)
    await page.waitForTimeout(120)
    const g = await gain()
    if (g) seen.push(g)
    if (i === 5) await page.screenshot({ path: `${SHOTS}/02-mid-drag.png` })
  }
  await page.screenshot({ path: `${SHOTS}/03-late-drag.png` })
  await page.mouse.up()
  await page.waitForTimeout(1500)
  await page.screenshot({ path: `${SHOTS}/04-released.png` })

  const distinct = new Set(seen)
  expect(seen.length, 'no readings taken during the drag').toBeGreaterThan(4)
  expect(distinct.size,
    `the gain never changed while dragging (values: ${[...distinct].join(', ')}) — ` +
    'the preview is not live').toBeGreaterThan(1)

  const errors = backendErrorLines(ctx.backend)
  expect(errors, `backend errors:\n${errors.join('\n')}`).toEqual([])
  ctx.assertNoJsErrors()
})
