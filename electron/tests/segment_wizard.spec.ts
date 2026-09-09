/**
 * segment_wizard.spec.ts — the Segment caret, end-to-end on the bundled
 * synthetic particle movie.
 *
 * What this proves that tsc + headless tests cannot:
 *   1. A REAL Shift+drag on the figure paints: the anyplotlib brush is armed on
 *      the window and its strokes reach the backend's label store (the per-class
 *      pixel counts on the caret move). This is the only path a user has, and
 *      it is the wiring most likely to break silently.
 *   2. Train paints the particle mask over the frame (screenshot).
 *   3. Run opens the result tree: a label movie with a count-per-frame
 *      navigator (window count + screenshot).
 *
 * The strokes are placed in IMAGE pixels through the renderer's own letterbox
 * math (anyplotlib `PLOT_PADDING` + contain-fit), so they land on the fixture's
 * particles rather than wherever the iframe happens to be.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = 'segment_wizard_shots'
let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

/** The fixture: `particle_movie(shape=(96, 112))`, particles at fixed pixels. */
const IMAGE = { width: 112, height: 96 }
/** anyplotlib `_PanelMixin.PLOT_PADDING` (left, right, top, bottom), CSS px. */
const PAD = { left: 58, right: 12, top: 12, bottom: 42 }

/** Image pixel → page coordinates, mirroring the renderer's contain-fit. */
function imageToPage(box: { x: number; y: number; width: number; height: number },
                     ix: number, iy: number): { x: number; y: number } {
  const areaW = box.width - PAD.left - PAD.right
  const areaH = box.height - PAD.top - PAD.bottom
  const scale = Math.min(areaW / IMAGE.width, areaH / IMAGE.height)
  const x0 = box.x + PAD.left + (areaW - IMAGE.width * scale) / 2
  const y0 = box.y + PAD.top + (areaH - IMAGE.height * scale) / 2
  return { x: x0 + (ix + 0.5) * scale, y: y0 + (iy + 0.5) * scale }
}

async function paint(page: any, box: any, from: [number, number], to: [number, number]) {
  const a = imageToPage(box, from[1], from[0]), b = imageToPage(box, to[1], to[0])
  await page.keyboard.down('Shift')
  await page.mouse.move(a.x, a.y)
  await page.mouse.down()
  for (let i = 1; i <= 8; i++) {
    await page.mouse.move(a.x + (b.x - a.x) * i / 8, a.y + (b.y - a.y) * i / 8)
    await page.waitForTimeout(20)
  }
  await page.mouse.up()
  await page.keyboard.up('Shift')
}

const pixelsOf = (page: any, classId: number) =>
  page.getByTestId(`seg-class-${classId}`).getAttribute('data-pixels')

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_particles', { frames: 12 })
  await waitForSubwindowCount(page, 2, 120_000)
  // The window appears before the navigator has painted; clicking the toolbar
  // through that leaves the caret unopened.
  await page.waitForTimeout(4000)
})

test.afterAll(async () => {
  await ctx?.app?.close()
})

test('the caret opens with three classes and an armed brush', async () => {
  const { page } = ctx
  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Segment').click()
  await expect(page.getByTestId('seg-wizard')).toBeVisible()
  await expect(page.getByTestId('seg-class-0')).toBeVisible({ timeout: 30_000 })
  for (const id of [0, 1, 2]) expect(await pixelsOf(page, id)).toBe('0')
  await expect(page.getByTestId('seg-train')).toBeDisabled()
  await expect(page.getByTestId('seg-advanced')).toHaveCount(0)
  await page.screenshot({ path: `${SHOTS}/01-caret-open.png` })
  ctx.assertNoJsErrors()
})

test('a REAL Shift+drag paints, and the class picked on the caret is the one painted', async () => {
  const { page } = ctx
  const sig = sigWindow(page)
  const box = await sig.locator('iframe').first().boundingBox()
  expect(box, 'no figure iframe to paint on').toBeTruthy()

  // Particle class (the default): through the two bright anchors.
  await paint(page, box!, [24, 16], [24, 28])
  await paint(page, box!, [70, 24], [70, 36])
  await expect.poll(() => pixelsOf(page, 0), {
    timeout: 30_000,
    message: 'a Shift+drag over the figure painted nothing — no brush is armed, '
      + 'or its strokes never reach the label store',
  }).not.toBe('0')

  // Background: pick the class on the caret, then paint the film.
  await page.getByTestId('seg-class-1').click()
  await paint(page, box!, [4, 4], [4, 100])
  await paint(page, box!, [50, 55], [62, 55])
  await expect.poll(() => pixelsOf(page, 1), { timeout: 30_000 }).not.toBe('0')
  expect(await pixelsOf(page, 2)).toBe('0')
  await expect(page.getByTestId('seg-train')).toBeEnabled()
  await page.waitForTimeout(500)
  await sig.screenshot({ path: `${SHOTS}/02-strokes-on-the-frame.png` })
  ctx.assertNoJsErrors()
})

test('Train paints the particle mask over the frame', async () => {
  const { page } = ctx
  const sig = sigWindow(page)
  await page.getByTestId('seg-train').click()
  await expect.poll(
    async () => await page.getByTestId('seg-trained').getAttribute('data-accuracy'),
    { timeout: 120_000, message: 'training never reported back' },
  ).not.toBe('')
  const accuracy = Number(await page.getByTestId('seg-trained').getAttribute('data-accuracy'))
  expect(accuracy).toBeGreaterThan(0.9)
  await page.waitForTimeout(2000)
  await sig.screenshot({ path: `${SHOTS}/03-mask-preview.png` })
  await page.getByTestId('seg-wizard').screenshot({ path: `${SHOTS}/04-caret-trained.png` })
  await expect(page.getByTestId('seg-run')).toBeEnabled()
  ctx.assertNoJsErrors()
})

test('Run opens the label movie with a count navigator', async () => {
  const { page } = ctx
  await page.getByTestId('seg-run').click()
  await expect(page.getByTestId('seg-result')).toBeVisible({ timeout: 180_000 })
  const regions = Number(await page.getByTestId('seg-result').getAttribute('data-regions'))
  expect(regions, 'the run found nothing').toBeGreaterThan(20)
  // The result is its own tree: a label-movie window and its count navigator.
  await waitForSubwindowCount(page, 4, 120_000)
  await expect(page.getByTestId('status-text')).toContainText(/Found \d+ regions in 12 fields/)
  await page.waitForTimeout(2500)
  await page.screenshot({ path: `${SHOTS}/05-result.png` })
  const errors = backendErrorLines(ctx.backend)
  expect(errors, `backend errors:\n${errors.join('\n')}`).toEqual([])
  ctx.assertNoJsErrors()
})
