/**
 * vectors_report_embed.spec.ts — the embedded-vectors explorer in an exported
 * HTML report, driven in a REAL browser (plain chromium over file://; the page
 * must work with no app, no backend, no network — that's its whole point).
 *
 * The explorer is built from real anyplotlib figures: a circle/annulus widget
 * on the k-space density selects the detector, a rectangle widget on the
 * virtual image selects a real-space region that filters the k view. The
 * fixture (spyde/tests/gen_vectors_embed.py) plants cluster A at k=(-0.5,0)
 * only in the LEFT nav half and cluster B at k=(+0.5,0) only in the RIGHT
 * half — moving the detector between clusters must flip which half lights up.
 * Geometry is driven through window.__vx (the same code path the widget
 * events call), plus one REAL pointer drag on the anyplotlib widget as a
 * smoke test that events actually flow.
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

// Fixture geometry: k extent [-1,1] over a 256-bin density image →
// k=-0.5 is px 63.75, k=+0.5 is px 191.25; r=0.2 in k = 25.5 px.
const PX = (k: number) => ((k + 1) / 2) * 255

test('detector position selects which nav half lights up (disk + annulus + region)', async () => {
  // Both anyplotlib figures mounted (each renders at least one canvas).
  expect(await page.locator('#vx-figk canvas').count()).toBeGreaterThan(0)
  expect(await page.locator('#vx-figvi canvas').count()).toBeGreaterThan(0)

  // Disk on cluster A → LEFT half bright.
  await page.evaluate(({ cx, cy, r }) =>
    (window as any).__vx.setDetector({ cx, cy, r }),
    { cx: PX(-0.5), cy: PX(0), r: 25.5 })
  await expect.poll(async () => (await stats()).leftMean).toBeGreaterThan(50)
  let s = await stats()
  expect(s.leftMean).toBeGreaterThan(10 * Math.max(1, s.rightMean))
  expect(s.hit).toBeGreaterThan(0)
  // The RENDERED canvas must light up too — the stats mirror the compute,
  // but a broken pixel push once left the canvas black while stats passed.
  await expect.poll(async () => page.evaluate(() => {
    let best = 0
    for (const c of document.querySelectorAll('#vx-figvi canvas')) {
      const ctx = (c as HTMLCanvasElement).getContext('2d')
      if (!ctx || !(c as HTMLCanvasElement).width) continue
      const d = ctx.getImageData(0, 0, (c as HTMLCanvasElement).width,
                                 (c as HTMLCanvasElement).height).data
      let sum = 0
      for (let i = 0; i < d.length; i += 4) sum += d[i]
      best = Math.max(best, sum / (d.length / 4))
    }
    return best
  }), { timeout: 5_000, message: 'VI canvas never painted' }).toBeGreaterThan(15)
  await expect(page.locator('#vx-readout')).toContainText('of 768 vectors')
  await page.screenshot({ path: 'vectors_embed_shots/01-left-cluster.png' })

  // Disk on cluster B → flips to the RIGHT half.
  await page.evaluate(({ cx, cy, r }) =>
    (window as any).__vx.setDetector({ cx, cy, r }),
    { cx: PX(0.5), cy: PX(0), r: 25.5 })
  await expect.poll(async () => (await stats()).rightMean).toBeGreaterThan(50)
  s = await stats()
  expect(s.rightMean).toBeGreaterThan(10 * Math.max(1, s.leftMean))

  // ANNULUS with the ring straddling cluster B stays bright; a ring whose
  // inner radius EXCLUDES the cluster goes dark (r in px: cluster ±0.02 k).
  await page.locator('input[name=vx-shape][value=annular]').check()
  await page.evaluate(({ cx, cy }) =>
    (window as any).__vx.setDetector({ cx, cy, rIn: 2, rOut: 26 }),
    { cx: PX(0.5), cy: PX(0) })
  await expect.poll(async () => (await stats()).rightMean).toBeGreaterThan(50)
  await page.evaluate(({ cx, cy }) =>
    (window as any).__vx.setDetector({ cx, cy, rIn: 15, rOut: 26 }),
    { cx: PX(0.5), cy: PX(0) })
  await expect.poll(async () => (await stats()).hit).toBe(0)
  await page.screenshot({ path: 'vectors_embed_shots/02-annulus.png' })

  // REAL-SPACE region filter: back to a disk on cluster A, then restrict the
  // region to the RIGHT nav half — cluster A vanishes from the k view, so the
  // detector catches nothing.
  await page.locator('input[name=vx-shape][value=circle]').check()
  await page.evaluate(({ cx, cy, r }) =>
    (window as any).__vx.setDetector({ cx, cy, r }),
    { cx: PX(-0.5), cy: PX(0), r: 25.5 })
  await expect.poll(async () => (await stats()).hit).toBeGreaterThan(0)
  await page.evaluate((nx) =>
    (window as any).__vx.setRegion({ x: nx / 2, y: 0, w: nx / 2, h: 16 }), 16)
  // The VI itself is detector-driven (unchanged), but the readout notes the
  // region filter and the k view now only holds cluster B.
  await expect(page.locator('#vx-readout')).toContainText('region-filtered')
  await page.screenshot({ path: 'vectors_embed_shots/03-region.png' })

  // Smoke: a REAL pointer drag on the anyplotlib circle widget flows through
  // onEvent into the glue (det changes from where we left it). Park a big
  // disk at the image centre so the widget sits under the canvas centre.
  await page.locator('input[name=vx-shape][value=circle]').check()
  await page.evaluate(() =>
    (window as any).__vx.setDetector({ cx: 127.5, cy: 127.5, r: 40 }))
  const before = await page.evaluate(() => ({ ...(window as any).__vx.det }))
  const box = (await page.locator('#vx-figk canvas').first().boundingBox())!
  const startX = box.x + box.width * 0.5, startY = box.y + box.height * 0.5
  await page.mouse.move(startX, startY)
  await page.mouse.down()
  await page.mouse.move(startX + 30, startY + 20, { steps: 5 })
  await page.mouse.up()
  const after = await page.evaluate(() => ({ ...(window as any).__vx.det }))
  expect(after.cx !== before.cx || after.cy !== before.cy).toBe(true)
})
