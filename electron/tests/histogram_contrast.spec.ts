/**
 * histogram_contrast.spec.ts — the dock's contrast widget on REAL Poisson data.
 *
 * The reported symptom is visual (bars crushed into a sliver at the left, handles
 * with nowhere to grab), so this drives the real app and screenshots the dock:
 * headless assertions cannot see a squished histogram.
 *
 * Covers the three things that make it controllable — bins around the display
 * range rather than min–max, Auto/Reset, and a handle dragged off the drawn range
 * staying recoverable.
 */
import { test, expect } from '@playwright/test'
const { launchApp, backendAction } = require('./_harness.cjs')

const SHOTS = 'histogram_shots'
let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ env: { SPYDE_LOG_LEVEL: 'INFO' } })
  // Si grains: bundled, eager, and a real reciprocal lattice — a bright central
  // beam over a Poisson background, which is exactly the distribution that
  // squashed the histogram.
  await backendAction(ctx.page, 'load_test_data_si_grains')
  await expect(ctx.page.getByTestId('subwindow').first()).toBeVisible({ timeout: 60_000 })
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(120_000)

/** The clim labels, parsed back to numbers (they render exponential above 1e3). */
async function clim(): Promise<[number, number]> {
  const { page } = ctx
  const lo = Number(await page.getByTestId('clim-min').textContent())
  const hi = Number(await page.getByTestId('clim-max').textContent())
  return [lo, hi]
}

test('the bars use the width of the widget, not a sliver of it', async () => {
  const { page } = ctx
  const hist = page.getByTestId('histogram')
  await expect(hist).toBeVisible({ timeout: 30_000 })

  // Measure what the complaint was about: how far across the widget the drawn
  // bars actually reach. Bars are the <rect>s after the tint rect.
  const spread = await hist.evaluate((svg: SVGSVGElement) => {
    const W = Number(svg.getAttribute('width'))
    // Bars are the count rects: the selected-range tint carries an opacity and
    // the handle grips are pink, so neither is one.
    const bars = Array.from(svg.querySelectorAll('rect')).filter((r) =>
      !r.hasAttribute('opacity') &&
      ['#89b4fa', '#585b70'].includes(r.getAttribute('fill') ?? '') &&
      Number(r.getAttribute('height')) > 0)
    if (!bars.length) return 0
    const right = Math.max(...bars.map(
      (r) => Number(r.getAttribute('x')) + Number(r.getAttribute('width'))))
    return right / W
  })
  expect(spread, 'the histogram bars still occupy a sliver of the widget')
    .toBeGreaterThan(0.5)

  await page.getByTestId('histogram-section').screenshot({ path: `${SHOTS}/01-histogram.png` })
  ctx.assertNoJsErrors()
})

test('Reset widens to the full data range and Auto brings it back', async () => {
  const { page } = ctx
  const [autoLo0, autoHi0] = await clim()

  await page.getByTestId('clim-reset').click()
  await expect.poll(async () => (await clim())[1],
    { message: 'Reset did not widen the display range' })
    .toBeGreaterThan(autoHi0)
  const [, fullHi] = await clim()
  await page.getByTestId('histogram-section').screenshot({ path: `${SHOTS}/02-reset.png` })

  await page.getByTestId('clim-auto').click()
  await expect.poll(async () => (await clim())[1],
    { message: 'Auto did not restore the robust range' })
    .toBeLessThan(fullHi)
  const [autoLo1, autoHi1] = await clim()
  expect(autoLo1).toBeCloseTo(autoLo0, 5)
  expect(autoHi1).toBeCloseTo(autoHi0, 5)
  await page.getByTestId('histogram-section').screenshot({ path: `${SHOTS}/03-auto.png` })
  ctx.assertNoJsErrors()
})

test('the handles are a hover affordance, not permanent furniture', async () => {
  const { page } = ctx
  const hist = page.getByTestId('histogram')
  // Grip caps mark an "armed" handle: pink 6px-wide rects at the top/bottom.
  const grips = () => hist.evaluate((svg: SVGSVGElement) =>
    Array.from(svg.querySelectorAll('rect')).filter((r) =>
      r.getAttribute('fill') === '#f38ba8' && r.getAttribute('width') === '6').length)

  // At rest: hairlines only. Park the pointer somewhere else in the dock first —
  // an earlier test may have left it over the widget.
  await page.getByTestId('plot-control-dock').hover({ position: { x: 5, y: 5 } })
  await expect.poll(grips, { message: 'grip caps are drawn without hovering' }).toBe(0)
  await page.getByTestId('histogram-section').screenshot({ path: `${SHOTS}/05-rest.png` })
  // Whole dock: Auto/Reset must read as the same control as Point/Integrate.
  await page.getByTestId('plot-control-dock').screenshot({ path: `${SHOTS}/05b-dock.png` })

  await hist.hover()
  await expect.poll(grips, { message: 'hovering did not arm the handles' }).toBe(4)
  await page.getByTestId('histogram-section').screenshot({ path: `${SHOTS}/06-hover.png` })
  ctx.assertNoJsErrors()
})

test('a handle dragged off the drawn range stays grabbable', async () => {
  const { page } = ctx
  const hist = page.getByTestId('histogram')
  const box = (await hist.boundingBox())!

  // Drag the min handle far past the left edge of the widget.
  const handle = page.getByTestId('hist-min-handle')
  const hb = (await handle.boundingBox())!
  await page.mouse.move(hb.x + hb.width / 2, hb.y + hb.height / 2)
  await page.mouse.down()
  await page.mouse.move(box.x - 200, hb.y + hb.height / 2, { steps: 10 })
  await page.mouse.up()

  // It must still be ON the widget (pinned at the edge, with the off-range
  // arrow), not drawn off-canvas where it can never be grabbed again.
  await expect(page.getByTestId('hist-min-offscreen')).toBeVisible()
  const pinned = (await page.getByTestId('hist-min-handle').boundingBox())!
  expect(pinned.x + pinned.width).toBeGreaterThan(box.x)
  expect(pinned.x).toBeLessThan(box.x + box.width)
  await page.getByTestId('histogram-section').screenshot({ path: `${SHOTS}/04-min-offscreen.png` })

  // …and dragging it back in recovers.
  await page.mouse.move(pinned.x + pinned.width / 2, pinned.y + pinned.height / 2)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width * 0.3, pinned.y + pinned.height / 2, { steps: 10 })
  await page.mouse.up()
  await expect(page.getByTestId('hist-min-offscreen')).toBeHidden()
  ctx.assertNoJsErrors()
})
