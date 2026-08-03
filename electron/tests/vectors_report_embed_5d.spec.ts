/**
 * vectors_report_embed_5d.spec.ts — the embedded-vectors explorer for a 5-D
 * STACK, in a REAL browser (plain chromium over file://, no app / backend /
 * network).
 *
 * A stack used to collapse in the report: the packer refused to build a
 * run-length index for anything deeper than a 2-D scan, so the page fell back to
 * matching on (iy, ix) — which hits EVERY slice's vectors at that position. The
 * explorer showed all slices piled onto one pattern, with nothing to scrub.
 *
 * The fixture (spyde/tests/gen_vectors_embed.py `synthetic_vectors_5d`) makes
 * that failure impossible to miss: slice t's vectors sit at kx swept left→right
 * across the stack, and each position holds t+1 of them. So the CURRENT slice is
 * identified twice over — which side of the DP lights up, and how many vectors
 * the readout reports. The collapsed page shows both sides and 10 vectors
 * (1+2+3+4) at every position.
 */
import { test, expect, chromium, Browser, Page } from '@playwright/test'
import { execFileSync } from 'child_process'
import { existsSync, mkdirSync } from 'fs'
import { join } from 'path'
import { tmpdir } from 'os'

let browser: Browser
let page: Page
const htmlPath = join(tmpdir(), 'spyde-vectors-embed-5d-test.html')
const SHOTS = join(__dirname, '..', 'vectors_embed_5d_shots')
const NT = 4

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  const root = join(__dirname, '..', '..')
  const candidates = [
    join(root, '.venv', 'Scripts', 'python.exe'),
    join(root, '.venv', 'bin', 'python'),
  ]
  const py = candidates.find((p) => existsSync(p))
    ?? (process.platform === 'win32' ? 'python' : 'python3')
  execFileSync(py, ['-m', 'spyde.tests.gen_vectors_embed', htmlPath, '--5d'],
    { cwd: root })
  browser = await chromium.launch()
  page = await browser.newPage()
  await page.goto('file:///' + htmlPath.replace(/\\/g, '/'))
  await page.waitForSelector('#vx-root[data-ready="1"]', { timeout: 30_000 })
})

test.afterAll(async () => { await browser?.close() })

const stats = () => page.evaluate(() => (window as any).__vx.stats)

test('the stack slider selects one slice, and only that slice', async () => {
  // The page knows it is a stack.
  expect(await page.evaluate(() => (window as any).__vx.nSlices)).toBe(NT)

  await page.evaluate(() => (window as any).__vx.setPointer({ ix: 4, iy: 4 }))

  for (let t = 0; t < NT; t++) {
    await page.evaluate((tt) => (window as any).__vx.setSlice(tt), t)
    // Exactly this slice's vectors — not every slice's, and not another's.
    await expect.poll(async () => (await stats()).hit, {
      timeout: 5_000,
      message: `slice ${t} should hold ${t + 1} vectors at one position`,
    }).toBe(t + 1)

    const s = await stats()
    // kx sweeps left→right across the stack, so the DP energy must follow.
    if (t === 0) {
      expect(s.leftMean, 'slice 0 must render on the LEFT of the DP')
        .toBeGreaterThan(5 * Math.max(0.001, s.rightMean))
    }
    if (t === NT - 1) {
      expect(s.rightMean, 'the last slice must render on the RIGHT of the DP')
        .toBeGreaterThan(5 * Math.max(0.001, s.leftMean))
    }
    await expect(page.locator('#vx-readout')).toContainText(`slice ${t}/${NT - 1}`)
    await page.screenshot({ path: join(SHOTS, `0${t + 1}-slice-${t}.png`) })
  }
})

test('the stack gets its own navigator PANEL, and its vline drives the page', async () => {
  // App parity: a 5-D vectors window is stack navigator → real-space navigator
  // → DP, so the embed is three panels, not two-plus-a-slider.
  expect(await page.evaluate(() => (window as any).__vx.hasStackPanel())).toBe(true)
  const ids = await page.evaluate(() => {
    const h = (window as any).__vx._h()
    return { nav: h.NAV_ID, dp: h.DP_ID, stack: h.STACK_ID,
             mounted: Boolean(h.stackPanel) }
  })
  expect(ids.stack, 'no stack panel id').toBeTruthy()
  expect(new Set([ids.nav, ids.dp, ids.stack]).size,
         'the three panels must be distinct').toBe(3)
  expect(ids.mounted, 'the stack panel never mounted').toBe(true)
  // The slider IS shipped (it is the phone layout's slice control) but must
  // stay hidden wherever the stack panel mounted — two competing controls for
  // one axis is worse than either alone.
  expect(await page.getByTestId('vx-slice').count()).toBe(1)
  await expect(page.getByTestId('vx-slice')).toBeHidden()

  // Drag the stack panel's ORANGE vline to the right → a later slice.
  await page.evaluate(() => (window as any).__vx.setSlice(0))
  await page.waitForTimeout(300)
  const line = await page.evaluate(() => {
    // The vline is drawn #f6c177 on a widget-overlay canvas. Search ONLY the
    // stack panel's own canvases — the navigator's crosshair is the same colour
    // and has far more pixels, so a figure-wide search grabs that instead.
    const host = (window as any).__vx._h().stackPanel?.plotCanvas?.parentElement
    if (!host) return null
    let best: any = null
    for (const c of Array.from(host.querySelectorAll('canvas'))) {
      const cv = c as HTMLCanvasElement
      const g = cv.getContext('2d')
      if (!g || !cv.width || !cv.height) continue
      const d = g.getImageData(0, 0, cv.width, cv.height).data
      const cols = new Int32Array(cv.width)
      let total = 0
      for (let y = 0; y < cv.height; y++) {
        for (let x = 0; x < cv.width; x++) {
          const p = (y * cv.width + x) * 4
          // #f6c177 ≈ (246, 193, 119): warm, red>green>blue, all high-ish.
          if (d[p + 3] > 40 && d[p] > 200 && d[p + 1] > 150 && d[p + 1] < 225
              && d[p + 2] > 70 && d[p + 2] < 165) { cols[x]++; total++ }
        }
      }
      if (!total || (best && total <= best.total)) continue
      let bx = 0
      for (let x = 1; x < cv.width; x++) if (cols[x] > cols[bx]) bx = x
      const r = cv.getBoundingClientRect()
      best = { total,
               x: r.left + (bx / cv.width) * r.width,
               y: r.top + r.height / 2,
               right: r.left + r.width }
    }
    return best
  })
  expect(line, 'the stack vline was not found on any canvas').not.toBeNull()

  await page.mouse.move(line.x, line.y)
  await page.mouse.down()
  await page.mouse.move(line.x + (line.right - line.x) * 0.75, line.y,
                        { steps: 8 })
  await page.mouse.up()
  await expect.poll(() => page.evaluate(() => (window as any).__vx.slice()), {
    timeout: 5_000, message: 'dragging the stack vline changed no slice',
  }).toBeGreaterThan(0)
  await page.screenshot({ path: join(SHOTS, '06-vline-drag.png') })
})

test('integrate mode sums within the current slice only', async () => {
  await page.evaluate(() => (window as any).__vx.setSlice(1))
  await page.evaluate(() => (window as any).__vx.setMode(true))
  await page.evaluate(() => (window as any).__vx.setRegion(
    { x: 0, y: 0, w: 4, h: 4 }))
  // 4x4 positions x 2 vectors each in slice 1 — a region that leaked across
  // slices would report 4*4*10 = 160.
  await expect.poll(async () => (await stats()).hit, { timeout: 5_000 })
    .toBe(4 * 4 * 2)
  await page.screenshot({ path: join(SHOTS, '05-integrate-slice-1.png') })
  await page.evaluate(() => (window as any).__vx.setMode(false))
})
