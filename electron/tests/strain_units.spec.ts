/**
 * strain_units.spec.ts — the Strain window says how much, and over what.
 *
 * The strain fit is fractional; nothing on screen used to say whether a red
 * patch was 0.2 % or 2 %, and the map's axes were scan pixels. This drives the
 * real workflow — bundled vectors → Strain Mapping → component toggle → Commit
 * → a chip view — and LOOKS at each figure: the colorbar strip is drawn on a
 * canvas inside the anyplotlib iframe, so a metadata assertion would prove
 * nothing about what the user sees. Screenshots land in strain_units_shots/.
 *
 * Runs without a Dask cluster, like strain_lazy.spec.ts: strain's own logic
 * does not care, and a real LocalCluster is slow to spawn on this box.
 */
import { test, expect } from '@playwright/test'
import * as fs from 'fs'
import * as path from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = path.join(__dirname, '..', 'strain_units_shots')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(180_000)

test.beforeAll(async () => {
  fs.mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ dask: false, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  await ctx.page.waitForTimeout(1500)
  await backendAction(ctx.page, 'load_test_vectors')
  await waitForSubwindowCount(ctx.page, 4, 60_000)
  await ctx.page.waitForTimeout(2500)
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
 * Count the colorbar strips a window's figure iframes draw.
 *
 * anyplotlib paints the strip on its own tall, narrow canvas beside the image;
 * a real strip is a gradient, so it holds many distinct colours. A blank
 * canvas of the same shape (or none at all) means the unit never reached the
 * figure.
 */
async function colorbarStrips(win: any): Promise<number> {
  let found = 0
  for (const iframe of await win.locator('iframe').elementHandles()) {
    const frame = await iframe.contentFrame()
    if (!frame) continue
    try {
      found += await frame.evaluate(() => {
        let n = 0
        for (const c of Array.from(document.querySelectorAll('canvas'))) {
          const cv = c as HTMLCanvasElement
          if (!cv.width || !cv.height || cv.height < cv.width * 3) continue
          const g = cv.getContext('2d')
          if (!g) continue
          const d = g.getImageData(0, 0, cv.width, cv.height).data
          const seen = new Set<number>()
          for (let p = 0; p < d.length; p += 4) {
            if (d[p + 3] === 0) continue
            seen.add((d[p] << 16) | (d[p + 1] << 8) | d[p + 2])
            if (seen.size > 8) break
          }
          if (seen.size > 8) n++
        }
        return n
      })
    } catch { /* torn-down frame */ }
  }
  return found
}

test('the live Strain window draws a percent colorbar over a calibrated map', async () => {
  const { page } = ctx
  const vsig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('action-btn-Strain Mapping') }).first()
  await vsig.getByTestId('subwindow-titlebar').click()
  await vsig.getByTestId('subwindow-titlebar').hover()
  const btn = vsig.getByTestId('action-btn-Strain Mapping')
  await expect(btn).toBeVisible({ timeout: 15_000 })

  const before = await page.getByTestId('subwindow').count()
  await btn.click()
  await expect(page.getByTestId('strain-wizard')).toBeVisible({ timeout: 15_000 })
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 60_000, message: 'strain map window never opened',
  }).toBeGreaterThan(before)
  const swin = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId(/^strain-toggle-/) }).first()
  await expect(swin.getByTestId(/^strain-comp-exx-/)).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(2500)
  await page.screenshot({ path: path.join(SHOTS, '01-strain-exx.png') })

  await expect.poll(() => colorbarStrips(swin), {
    timeout: 20_000, message: 'the live strain map drew no colorbar strip',
  }).toBeGreaterThan(0)

  // Rotation is degrees, not radians — the toggle relabels the same figure.
  await swin.getByTestId(/^strain-comp-omega-/).click()
  await page.waitForTimeout(2000)
  await page.screenshot({ path: path.join(SHOTS, '02-strain-omega.png') })
  expect(await colorbarStrips(swin)).toBeGreaterThan(0)
})

test('Commit keeps the units: the tree and its chip views are labelled too', async () => {
  const { page } = ctx
  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('strain-commit').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 60_000, message: 'Commit opened no window',
  }).toBeGreaterThan(before)
  await page.waitForTimeout(3000)
  await page.screenshot({ path: path.join(SHOTS, '03-committed.png') })

  const committed = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId(/^view-chip-εyy-/) }).first()
  await expect(committed).toBeVisible({ timeout: 15_000 })
  await expect.poll(() => colorbarStrips(committed), {
    timeout: 20_000, message: 'the committed εxx map drew no colorbar strip',
  }).toBeGreaterThan(0)

  // …and the committed map is on the scan's scale, not pixels: the Axes dock
  // reads the displayed signal's own axes.
  await committed.getByTestId('subwindow-title').click()
  await page.waitForTimeout(800)
  const axisRow = await page.locator('[data-testid="axes-table"] tbody tr')
    .filter({ hasText: /sig/ }).first().innerText()
  expect(axisRow, `committed map axes read "${axisRow}" — expected nm`).toMatch(/nm/)

  await committed.getByTestId(/^view-chip-εyy-/).click()
  await page.waitForTimeout(3000)
  await page.screenshot({ path: path.join(SHOTS, '04-chip-eyy.png') })
  await expect.poll(() => colorbarStrips(committed), {
    timeout: 20_000, message: 'the εyy chip view drew no colorbar strip',
  }).toBeGreaterThan(0)
})

test('the committed map keeps its diverging colours and a zero-centred range', async () => {
  const { page } = ctx
  const committed = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId(/^view-chip-εyy-/) }).first()
  await committed.getByTestId('subwindow-title').click()
  await page.waitForTimeout(800)

  // The dock's picker says what the figure shows — it was never touched here.
  await expect(page.getByTestId('colormap-select')).toContainText('coolwarm', { timeout: 10_000 })

  // Drag the UPPER handle inward: the lower one must mirror it, so zero stays
  // white in the middle of the map.
  const hist = (await page.getByTestId('histogram').boundingBox())!
  const handle = page.getByTestId('hist-max-handle')
  await page.getByTestId('histogram').hover()
  const hb = (await handle.boundingBox())!
  await page.mouse.move(hb.x + hb.width / 2, hb.y + hb.height / 2)
  await page.mouse.down()
  await page.mouse.move(hist.x + hist.width * 0.7, hb.y + hb.height / 2, { steps: 8 })
  await page.mouse.up()
  await page.waitForTimeout(800)
  const lo = Number(await page.getByTestId('clim-min').textContent())
  const hi = Number(await page.getByTestId('clim-max').textContent())
  expect(hi, 'the upper handle did not move').toBeGreaterThan(0)
  expect(lo, `range ${lo}..${hi} is not centred on zero`).toBeCloseTo(-hi, 6)
  await page.screenshot({ path: path.join(SHOTS, '05-symmetric-handles.png') })
})

/**
 * Page coordinates of the HEAD of the pink (scan-x) arrow drawn on a window's
 * figure. The head is the pink pixel farthest from the gold (scan-y) arrow:
 * both arrows share a tail, so the x head is the point farthest from the y
 * arrow's line. Returns null when no arrows are drawn.
 */
async function pinkArrowHead(win: any): Promise<{ x: number; y: number } | null> {
  const iframe = await win.locator('iframe').first().elementHandle()
  const frame = iframe && await iframe.contentFrame()
  if (!frame) return null
  const box = (await win.locator('iframe').first().boundingBox())!
  const local = await frame.evaluate(() => {
    for (const c of Array.from(document.querySelectorAll('canvas'))) {
      const g = c.getContext('2d')
      if (!g || !c.width || !c.height) continue
      const d = g.getImageData(0, 0, c.width, c.height).data
      const pink: number[][] = [], gold: number[][] = []
      for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) {
        const p = (y * c.width + x) * 4
        const r = d[p], gr = d[p + 1], b = d[p + 2], a = d[p + 3]
        if (a < 60) continue
        if (r > 200 && gr < 150 && b > 150) pink.push([x, y])
        else if (r > 200 && gr > 170 && b < 120) gold.push([x, y])
      }
      if (pink.length < 20 || gold.length < 20) continue
      let best = pink[0], bestD = -1
      for (const [px, py] of pink) {
        let near = Infinity
        for (let i = 0; i < gold.length; i += 3) {
          const dx = px - gold[i][0], dy = py - gold[i][1]
          const dd = dx * dx + dy * dy
          if (dd < near) near = dd
        }
        if (near > bestD) { bestD = near; best = [px, py] }
      }
      const rect = c.getBoundingClientRect()
      return { x: rect.left + (best[0] + 0.5) * (rect.width / c.width),
               y: rect.top + (best[1] + 0.5) * (rect.height / c.height) }
    }
    return null
  })
  return local && { x: box.x + local.x, y: box.y + local.y }
}

test('dragging the x arrow on the reference pattern turns the frame', async () => {
  const { page } = ctx
  const ref = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('subwindow-title').filter({ hasText: 'Strain Reference' }) })
    .first()
  await expect(ref).toBeVisible({ timeout: 15_000 })
  await ref.getByTestId('subwindow-title').click()
  await page.waitForTimeout(600)
  const head = await pinkArrowHead(ref)
  expect(head, 'no x/y arrows drawn on the reference pattern').not.toBeNull()
  await page.screenshot({ path: path.join(SHOTS, '07-axes-glyph.png') })

  // Drag the head straight up: the scan's x now points up-and-right on the
  // detector, an angle between the two axes — and the caret's slider says so.
  await page.mouse.move(head!.x, head!.y)
  await page.mouse.down()
  await page.mouse.move(head!.x, head!.y - 40, { steps: 8 })
  await page.mouse.up()
  await page.waitForTimeout(1200)
  const deg = Number(await page.getByTestId('strain-rotation').inputValue())
  expect(deg, `slider reads ${deg}° after the drag`).toBeGreaterThan(10)
  expect(deg).toBeLessThan(80)
  await page.screenshot({ path: path.join(SHOTS, '08-axes-glyph-dragged.png') })

  // No DPC run is open in this session, and the picker says so.
  await expect(page.getByTestId('strain-rotation-from-dpc')).toContainText('none open')
})

test('the Rotation control turns the tensor into the scan frame', async () => {
  const { page } = ctx
  const rotation = page.getByTestId('strain-rotation')
  await expect(rotation).toBeVisible()
  await expect(page.getByTestId('strain-flip')).toBeVisible()
  await expect(page.getByTestId('strain-rotation-from-dpc')).toBeVisible()
  await expect(page.getByTestId('strain-dpc-refresh')).toBeVisible()
  // 90°: what was εxx on the detector is εyy on the scan — the live window
  // repaints the same figure, no re-fit, no new window.
  const before = await page.getByTestId('subwindow').count()
  await rotation.fill('90')
  await page.waitForTimeout(1500)
  expect(await page.getByTestId('subwindow').count()).toBe(before)
  await page.screenshot({ path: path.join(SHOTS, '06-rotated-90.png') })
})
