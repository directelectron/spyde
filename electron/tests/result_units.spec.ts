/**
 * result_units.spec.ts — a committed result map says what its numbers mean.
 *
 * DPC is the only result type that ever computed a unit (`MV/cm` electric,
 * `mrad` magnetic), and it lived as a string on a dataclass that reached a chip
 * label and nothing else: no colorbar was drawn anywhere in SpyDE, and the
 * committed signal carried neither the unit nor the scan's spatial calibration.
 *
 * This drives a real DPC run to Commit and then LOOKS at the figure — the
 * colorbar strip and its label are drawn inside the anyplotlib iframe, so a
 * metadata assertion would prove nothing about what the user sees.
 */
import { test, expect } from '@playwright/test'
import * as fs from 'fs'
import * as path from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = path.join(__dirname, '..', 'result_units_shots')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(240_000)

test.beforeAll(async () => {
  fs.mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  await backendAction(ctx.page, 'load_test_data_dpc', { nav: 24, sig: 40 })
  await waitForSubwindowCount(ctx.page, 2, 120_000)
  await ctx.page.waitForTimeout(2000)
})

test.afterAll(async () => {
  try {
    const bad = ctx?.backend ? backendErrorLines(ctx.backend) : []
    expect(bad, `backend errors:\n${bad.join('\n')}`).toEqual([])
    ctx?.assertNoJsErrors()
  } finally {
    await ctx?.app?.close()
  }
})

/**
 * Does any figure iframe draw a colorbar strip?
 *
 * anyplotlib paints the strip on its own tall, narrow canvas next to the image.
 * Asserting on the FIGURE STATE would prove only that a flag was set — this
 * project's history is explicit that buffer-level asserts pass while the screen
 * is black — so this looks for a canvas at least 3× taller than it is wide whose
 * pixels are not all one colour, i.e. a rendered gradient.
 */
async function colorbarStripsOnScreen(page: any): Promise<number> {
  let found = 0
  for (const frame of page.frames()) {
    try {
      found += await frame.evaluate(() => {
        let n = 0
        for (const c of Array.from(document.querySelectorAll('canvas'))) {
          const cv = c as HTMLCanvasElement
          if (!cv.width || !cv.height) continue
          if (cv.height < cv.width * 3) continue        // not a vertical strip
          const ctx = cv.getContext('2d')
          if (!ctx) continue
          const d = ctx.getImageData(0, 0, cv.width, cv.height).data
          const seen = new Set<number>()
          for (let p = 0; p < d.length; p += 4) {
            if (d[p + 3] === 0) continue
            seen.add((d[p] << 16) | (d[p + 1] << 8) | d[p + 2])
            if (seen.size > 8) break
          }
          if (seen.size > 8) n++          // a real gradient, not a blank strip
        }
        return n
      })
    } catch { /* cross-origin or torn-down frame */ }
  }
  return found
}

test('a committed DPC map draws a colorbar labelled with its units', async () => {
  const { page } = ctx
  const sig = sigWindow(page)

  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-DPC').click()
  await expect(page.getByTestId('dpc-wizard')).toBeVisible()

  // Wait for the measure-once pass to produce a result.
  await expect.poll(() => page.getByTestId('dpc-centering').getAttribute('data-centered'),
    { timeout: 90_000, message: 'the descan readout never arrived' }).toBe('false')
  await page.waitForTimeout(3000)
  await page.screenshot({ path: path.join(SHOTS, '01-dpc-open.png') })

  // Commit: freezes the field into a new SignalTree with per-component views.
  const before = await page.getByTestId('subwindow').count()
  await backendAction(page, 'dpc_commit', {})
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 60_000, message: 'Commit opened no window',
  }).toBeGreaterThan(before)
  await page.waitForTimeout(4000)
  await page.screenshot({ path: path.join(SHOTS, '02-committed.png') })

  // The committed tree's SCALAR component views are what carry a unit; switch
  // to one so a scalar map is on screen (the primary is the RGB direction map,
  // which correctly has no colorbar).
  // NOT .first() — that is the "B direction" chip, the RGB primary, which
  // correctly has no colorbar. Pick a chip whose own label names a unit.
  const chip = page.locator('[data-testid^="view-chip-"]')
    .filter({ hasText: /\(mrad\)|\(MV\/cm\)/ }).first()
  await expect(chip, 'no scalar component chip on the committed window')
    .toBeVisible({ timeout: 30_000 })
  await chip.click()
  await page.waitForTimeout(3500)
  await page.screenshot({ path: path.join(SHOTS, '03-component-view.png') })

  await expect.poll(() => colorbarStripsOnScreen(page), {
    timeout: 20_000,
    message: 'no figure drew a colorbar strip — the unit never reached the figure',
  }).toBeGreaterThan(0)

  // …and the committed map keeps the SCAN's spatial calibration, not pixels.
  // The Axes dock reads it straight off the displayed signal.
  const axisUnits = await page.locator('[data-testid="axes-table"] tbody tr')
    .filter({ hasText: /sig/ }).first().innerText()
  expect(axisUnits, `committed map axes read "${axisUnits}" — expected a real unit`)
    .toMatch(/nm|µm|um|Å/)
})
