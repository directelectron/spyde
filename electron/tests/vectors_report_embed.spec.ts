/**
 * vectors_report_embed.spec.ts — the embedded-vectors explorer in an exported
 * HTML report, driven in a REAL browser (plain chromium over file://; the page
 * must work with no app, no backend, no network — that's its whole point).
 *
 * The explorer mirrors the MDI vector view: ONE anyplotlib figure with TWO
 * panels — a NAVIGATOR (count map) with a draggable CROSSHAIR (pointer mode) /
 * RECTANGLE (integrate mode), and a DIFFRACTION-PATTERN panel that shows the
 * pointed position's vectors as intensity DISKS (rendered client-side from the
 * embedded POINTS — no frames are shipped). The fixture
 * (spyde/tests/gen_vectors_embed.py) plants cluster A at k=(-0.5,0) only in the
 * LEFT nav half and cluster B at k=(+0.5,0) only in the RIGHT half. Because
 * kx ↔ DP column, a position in the left nav half renders disks in the LEFT of
 * the DP; a right-half position renders in the RIGHT. Geometry is driven through
 * window.__vx (the same code path the widget events call), plus one REAL pointer
 * drag on the anyplotlib crosshair as a smoke test that events actually flow.
 */
import { test, expect, chromium, Browser, Page } from '@playwright/test'
import { execFileSync } from 'child_process'
import { join } from 'path'
import { tmpdir } from 'os'

let browser: Browser
let page: Page
const htmlPath = join(tmpdir(), 'spyde-vectors-embed-test.html')

test.beforeAll(async () => {
  // Cross-platform venv interpreter (Windows Scripts/ vs POSIX bin/), falling
  // back to whatever python is on PATH (CI layouts vary).
  const root = join(__dirname, '..', '..')
  const { existsSync } = require('fs')
  const candidates = [
    join(root, '.venv', 'Scripts', 'python.exe'),
    join(root, '.venv', 'bin', 'python'),
  ]
  const py = candidates.find((p) => existsSync(p))
    ?? (process.platform === 'win32' ? 'python' : 'python3')
  execFileSync(py, ['-m', 'spyde.tests.gen_vectors_embed', htmlPath],
    { cwd: root })
  browser = await chromium.launch()
  page = await browser.newPage()
  await page.goto('file:///' + htmlPath.replace(/\\/g, '/'))
  await page.waitForSelector('#vx-root[data-ready="1"]', { timeout: 30_000 })
})

test.afterAll(async () => { await browser?.close() })

const stats = () => page.evaluate(() => (window as any).__vx.stats)

// The DP overlay canvas: brightest per-pixel mean over the figure's canvases
// (the DP disks push here — a broken pixel push would leave it dark).
const dpBrightness = () => page.evaluate(() => {
  let best = 0
  for (const c of document.querySelectorAll('#vx-fig canvas')) {
    const ctx = (c as HTMLCanvasElement).getContext('2d')
    if (!ctx || !(c as HTMLCanvasElement).width) continue
    const d = ctx.getImageData(0, 0, (c as HTMLCanvasElement).width,
                               (c as HTMLCanvasElement).height).data
    let sum = 0
    for (let i = 0; i < d.length; i += 4) sum += d[i]
    best = Math.max(best, sum / (d.length / 4))
  }
  return best
})

test('pointer + integrate render the DP from embedded points', async () => {
  // ONE figure, TWO panels → several canvases mounted.
  expect(await page.locator('#vx-fig canvas').count()).toBeGreaterThan(0)

  // POINTER on cluster A (left nav half) → disks in the LEFT of the DP.
  await page.evaluate(() => (window as any).__vx.setPointer({ ix: 1, iy: 0 }))
  await expect.poll(async () => (await stats()).hit).toBe(3)
  let s = await stats()
  expect(s.leftMean).toBeGreaterThan(5 * Math.max(0.001, s.rightMean))
  // The DP canvas must actually light up (stats mirror the compute, but a broken
  // pixel push once left the canvas black while stats passed).
  await expect.poll(dpBrightness, { timeout: 5_000 }).toBeGreaterThan(5)
  await expect(page.locator('#vx-readout')).toContainText('3 vectors')
  await page.screenshot({ path: 'vectors_embed_shots/01-pointer-left.png' })

  // POINTER on cluster B (right nav half) → flips to the RIGHT of the DP.
  await page.evaluate((nx) =>
    (window as any).__vx.setPointer({ ix: nx - 2, iy: 0 }), 16)
  await expect.poll(async () => (await stats()).rightMean)
    .toBeGreaterThan(0)
  s = await stats()
  expect(s.rightMean).toBeGreaterThan(5 * Math.max(0.001, s.leftMean))
  await page.screenshot({ path: 'vectors_embed_shots/02-pointer-right.png' })

  // INTEGRATE the whole LEFT nav half (8 x 16 positions x 3 vectors = 384).
  await page.evaluate(() => (window as any).__vx.setMode(true))
  await page.evaluate(() =>
    (window as any).__vx.setRegion({ x: 0, y: 0, w: 8, h: 16 }))
  await expect.poll(async () => (await stats()).hit).toBe(384)
  await expect(page.locator('#vx-readout')).toContainText('384 vectors summed')
  // Integrating many positions is much brighter than a single pointer frame.
  expect((await stats()).max).toBeGreaterThan(1000)
  await page.screenshot({ path: 'vectors_embed_shots/03-integrate.png' })

  // A 1x1 integrate region equals the pointer at that position (2x2=12 here).
  await page.evaluate(() =>
    (window as any).__vx.setRegion({ x: 0, y: 0, w: 2, h: 2 }))
  await expect.poll(async () => (await stats()).hit).toBe(12)

  // Smoke: a REAL pointer drag on the anyplotlib crosshair flows through onEvent
  // into the glue. Back to pointer mode, then grab the crosshair at its actual
  // SCREEN position (computed from the nav panel fit rect) so the drag lands on
  // the handle, not near it.
  await page.evaluate(() => (window as any).__vx.setMode(false))
  const before = await page.evaluate(() => ({ ...(window as any).__vx.cross }))
  const crossPos = await page.evaluate(() => {
    const h = (window as any).__vx._h()
    const nav = h.navPanel
    const pj = JSON.parse(h.H.get(h.navKey))
    const w = (pj.overlay_widgets || []).find((x: any) => x.type === 'crosshair')
    const host = nav.plotCanvas.getBoundingClientRect()
    const scale = Math.min(nav.imgW / pj.image_width, nav.imgH / pj.image_height)
    const offX = (nav.imgW - pj.image_width * scale) / 2
    const offY = (nav.imgH - pj.image_height * scale) / 2
    return { sx: host.x + offX + (w.cx + 0.5) * scale,
             sy: host.y + offY + (w.cy + 0.5) * scale }
  })
  await page.mouse.move(crossPos.sx, crossPos.sy)
  await page.mouse.down()
  await page.mouse.move(crossPos.sx + 20, crossPos.sy + 15, { steps: 5 })
  await page.mouse.up()
  const after = await page.evaluate(() => ({ ...(window as any).__vx.cross }))
  expect(after.ix !== before.ix || after.iy !== before.iy).toBe(true)
})
