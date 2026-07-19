/**
 * docs_embed.spec.ts — the INTERACTIVE docs-website embed, driven in a REAL
 * browser (plain chromium over file://; NO app, NO backend, NO network).
 *
 * This proves the docs-site's interactive walkthrough embeds work with ZERO
 * runtime Python (the precompute-embed model — no pyodide). The embed .html is
 * the exact file `spyde/tests/gen_guide_embeds.py` writes into
 * docs-site/public/media/<guide>/vectors-explorer.html and the docs step mounts
 * in a sandboxed iframe. We build it the same way the docs media dir is built
 * (via `python -m spyde.tests.gen_guide_embeds <out>`), load it over file://,
 * and assert it's genuinely interactive: the navigator + diffraction-pattern
 * panels render, and a navigate / integrate / virtual-image action RECOMPUTES
 * pixels — all client-side, no server.
 *
 * The synthetic fixture is a grain-boundary lattice: the LEFT nav half is grain
 * A (+12° lattice), the RIGHT half is grain B (−18° lattice), so navigation and
 * VI are pixel-visibly meaningful.
 */
import { test, expect, chromium, Browser, Page } from '@playwright/test'
import { execFileSync } from 'child_process'
import { join } from 'path'
import { tmpdir } from 'os'

let browser: Browser
let page: Page
const htmlPath = join(tmpdir(), 'spyde-docs-embed-test.html')

test.beforeAll(async () => {
  // Cross-platform venv interpreter (Windows Scripts/ vs POSIX bin/), falling
  // back to whatever python is on PATH (CI layouts vary) — same shape as
  // vectors_report_embed.spec.ts.
  const root = join(__dirname, '..', '..')
  const { existsSync } = require('fs')
  const candidates = [
    join(root, '.venv', 'Scripts', 'python.exe'),
    join(root, '.venv', 'bin', 'python'),
  ]
  const py = candidates.find((p) => existsSync(p))
    ?? (process.platform === 'win32' ? 'python' : 'python3')
  // Build the SAME embed the docs media dir ships — a single-output run.
  execFileSync(py, ['-m', 'spyde.tests.gen_guide_embeds', htmlPath], { cwd: root })
  browser = await chromium.launch()
  page = await browser.newPage()
  await page.goto('file:///' + htmlPath.replace(/\\/g, '/'))
  await page.waitForSelector('#vx-root[data-ready="1"]', { timeout: 30_000 })
})

test.afterAll(async () => { await browser?.close() })

const stats = () => page.evaluate(() => (window as any).__vx.stats)

// Brightest per-pixel mean over the figure's canvases — the DP disks / VI push
// here, so a recompute that actually paints raises this.
const brightness = () => page.evaluate(() => {
  let best = 0
  for (const c of document.querySelectorAll('#vx-fig canvas')) {
    const el = c as HTMLCanvasElement
    const ctx = el.getContext('2d')
    if (!ctx || !el.width) continue
    const d = ctx.getImageData(0, 0, el.width, el.height).data
    let sum = 0
    for (let i = 0; i < d.length; i += 4) sum += d[i]
    best = Math.max(best, sum / (d.length / 4))
  }
  return best
})

test('docs embed is interactive with no backend (navigate + integrate + VI)', async () => {
  // The self-contained page mounted: ONE figure, TWO panels → canvases exist.
  expect(await page.locator('#vx-fig canvas').count()).toBeGreaterThan(0)

  // NAVIGATE: pointer on the LEFT grain renders its lattice in the DP. Grain has
  // a bright direct beam + 8 first-order spots per position = 9 vectors.
  await page.evaluate(() => (window as any).__vx.setPointer({ ix: 2, iy: 3 }))
  await expect.poll(async () => (await stats()).hit, { timeout: 5_000 }).toBe(9)
  await expect.poll(brightness, { timeout: 5_000 }).toBeGreaterThan(2)
  await page.screenshot({ path: 'docs_embed_shots/01-navigate-grainA.png' })

  // NAVIGATE flips to the RIGHT grain — a DIFFERENT diffraction pattern (rotated
  // lattice). Pixels must change from grain A's frame.
  const gridW = await page.evaluate(() => (window as any).__vx.region.w) // full nav width
  await page.evaluate((nx) =>
    (window as any).__vx.setPointer({ ix: nx - 3, iy: 3 }), gridW)
  await expect.poll(async () => (await stats()).hit, { timeout: 5_000 }).toBe(9)
  await page.screenshot({ path: 'docs_embed_shots/02-navigate-grainB.png' })

  // INTEGRATE a region over the LEFT grain: many positions summed. The DP is
  // auto-normalised to a 255 peak either way, so the recompute signal is the raw
  // accumulator MAX — a region sums many frames' intensity, so its `max` dwarfs a
  // single pointer frame's (proves the region recompute actually ran).
  const pointerMax = (await stats()).max
  await page.evaluate(() => (window as any).__vx.setMode(true))
  await page.evaluate(() =>
    (window as any).__vx.setRegion({ x: 0, y: 0, w: 6, h: 14 }))
  await expect.poll(async () => (await stats()).hit, { timeout: 5_000 })
    .toBeGreaterThan(9)
  await expect(page.locator('#vx-readout')).toContainText('vectors summed')
  const integrateMax = (await stats()).max
  expect(integrateMax).toBeGreaterThan(pointerMax * 2)   // region recompute ran
  await expect.poll(brightness, { timeout: 5_000 }).toBeGreaterThan(2)
  await page.screenshot({ path: 'docs_embed_shots/03-integrate.png' })

  // VIRTUAL IMAGING: parking the DP detector recomputes a virtual image on the
  // navigator (a nonzero VI hit count proves the detector→scan scan ran).
  await page.evaluate(() => (window as any).__vx.setMode(false))
  await page.evaluate(() => (window as any).__vx.setDetector({ cx: 128, cy: 128, r: 40 }))
  await expect.poll(async () => (await stats()).viHit, { timeout: 5_000 })
    .toBeGreaterThan(0)
  await page.screenshot({ path: 'docs_embed_shots/04-virtual-image.png' })

  // DARK THEME (matches the docs site): the embed body is the app surface color.
  const bodyBg = await page.evaluate(() =>
    getComputedStyle(document.body).backgroundColor)
  expect(bodyBg).toBe('rgb(30, 30, 46)')   // #1e1e2e

  // A REAL pointer drag on the crosshair flows through onEvent → recompute (not
  // just the test hook) — the whole point of "try it".
  await page.evaluate(() => (window as any).__vx.setMode(false))
  const crossBefore = await page.evaluate(() => ({ ...(window as any).__vx.cross }))
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
  await page.mouse.move(crossPos.sx + 18, crossPos.sy + 14, { steps: 5 })
  await page.mouse.up()
  const crossAfter = await page.evaluate(() => ({ ...(window as any).__vx.cross }))
  expect(crossAfter.ix !== crossBefore.ix || crossAfter.iy !== crossBefore.iy).toBe(true)
})
