/**
 * orientation_report_embed.spec.ts — the embedded IPF explorer in an exported
 * HTML report, driven in a REAL browser (plain chromium over file://; the page
 * must work with no app, no backend and no network — that is its whole point).
 *
 * The explorer mirrors the app's two orientation windows in one figure: the IPF
 * MAP with a crosshair, the fundamental-sector TRIANGLE, and the unit SPHERE.
 * The fixture (spyde/tests/gen_orientation_embed.py) puts ONE orientation in the
 * left half of the nav grid and a very different one in the right half, so a
 * pick on each side must land in a different place on the triangle AND swing the
 * sphere's camera. If either stays put, the pick is not driving the view.
 *
 * Geometry goes through window.__ox — the SAME setPick/setDirection the
 * crosshair and the direction buttons call — plus one REAL pointer drag on the
 * anyplotlib crosshair as a smoke test that widget events actually flow.
 */
import { test, expect, chromium, Browser, Page } from '@playwright/test'
import { execFileSync } from 'child_process'
import { existsSync, mkdirSync } from 'fs'
import { join } from 'path'
import { tmpdir } from 'os'

let browser: Browser
let page: Page
const htmlPath = join(tmpdir(), 'spyde-orientation-embed-test.html')
const SHOTS = join(__dirname, '..', 'orientation_embed_shots')

test.describe.configure({ mode: 'serial' })
test.setTimeout(120_000)

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  const root = join(__dirname, '..', '..')
  const candidates = [
    join(root, '.venv', 'Scripts', 'python.exe'),
    join(root, '.venv', 'bin', 'python'),
  ]
  const py = candidates.find((p) => existsSync(p))
    ?? (process.platform === 'win32' ? 'python' : 'python3')
  execFileSync(py, ['-m', 'spyde.tests.gen_orientation_embed', htmlPath],
    { cwd: root })
  // Plain launch, exactly like vectors_report_embed — the default headless
  // shell, which is what CI has and what a report reader's browser stands in
  // for here. NOT `channel: 'chromium'` + --enable-unsafe-webgpu: that gets
  // headless a real navigator.gpu, and it paints fine locally, but on the CI
  // runner it produced three correctly-sized panels with ZERO pixels in any of
  // them, the plain 2-D map included. The 3-D panel does not need WebGPU
  // (anyplotlib falls back to Canvas2D) and every assertion here is about
  // pixels rather than which path drew them.
  browser = await chromium.launch()
  page = await browser.newPage({ viewport: { width: 1400, height: 700 } })
  await page.goto('file:///' + htmlPath.replace(/\\/g, '/'))
  await page.waitForSelector('#ox-root[data-ready="1"]', { timeout: 60_000 })
  // Wait for the PIXELS, not for a guess at how long they take: `data-ready`
  // only says the script finished, and a fixed sleep is what turns a slow
  // runner into "nothing painted".
  await expect.poll(async () => (await canvasInk()).filter((c) => c.ink > 200).length,
    { timeout: 60_000, message: 'the panels never painted' }).toBeGreaterThanOrEqual(3)
})

test.afterAll(async () => { await browser?.close() })

const oxState = () => page.evaluate(() => (window as any).__ox.state())

/** Per-canvas ink: how many pixels in each figure canvas are neither
 *  transparent nor the page background. A panel that drew nothing scores 0. */
const canvasInk = () => page.evaluate(() => {
  const out: Array<{ w: number; h: number; ink: number; colorful: number }> = []
  for (const el of document.querySelectorAll('#ox-fig canvas')) {
    const c = el as HTMLCanvasElement
    const ctx = c.getContext('2d')
    if (!ctx || !c.width || !c.height) { out.push({ w: c.width, h: c.height, ink: -1, colorful: -1 }); continue }
    let d: Uint8ClampedArray
    try { d = ctx.getImageData(0, 0, c.width, c.height).data } catch { continue }
    let ink = 0, colorful = 0
    for (let i = 0; i < d.length; i += 4) {
      if (d[i + 3] === 0) continue
      const r = d[i], g = d[i + 1], b = d[i + 2]
      if (r + g + b > 40) ink++
      if (Math.max(r, g, b) - Math.min(r, g, b) > 40) colorful++
    }
    out.push({ w: c.width, h: c.height, ink, colorful })
  }
  return out
})

test('1) the page mounts all three panels and every one of them paints', async () => {
  const s = await oxState()
  expect(s.dir).toBe('z')
  expect(s.marker, 'the triangle has no pick marker').toBeTruthy()
  expect(s.highlight, 'the sphere has no highlight').toBeTruthy()

  await page.screenshot({ path: join(SHOTS, '01-mounted.png'), fullPage: true })
  const inks = await canvasInk()
  console.log('[ox] canvas ink =', JSON.stringify(inks))
  // Three panels' worth of drawn content. Panels have several stacked canvases
  // each, so assert on the TOTAL and on how many carry real ink rather than
  // trying to name which canvas belongs to which panel.
  const painted = inks.filter((c) => c.ink > 200)
  expect(painted.length, 'fewer than three canvases drew anything').toBeGreaterThanOrEqual(3)
  // The IPF map and the triangle are both strongly chromatic — a greyscale
  // result would mean the colours never made it out of the packed blob.
  expect(inks.reduce((a, c) => a + Math.max(0, c.colorful), 0),
    'nothing chromatic on the page — the IPF colours did not arrive')
    .toBeGreaterThan(500)
})

test('2) picking the two nav halves gives two different orientations', async () => {
  // Left half → orientation A, right half → orientation B (the fixture's whole
  // design). Both the triangle marker and the sphere highlight must move.
  await page.evaluate(() => (window as any).__ox.setPick(8, 2))
  await page.waitForTimeout(600)
  const left = await oxState()
  await page.screenshot({ path: join(SHOTS, '02-pick-left.png'), fullPage: true })

  await page.evaluate(() => (window as any).__ox.setPick(8, 13))
  await page.waitForTimeout(600)
  const right = await oxState()
  await page.screenshot({ path: join(SHOTS, '03-pick-right.png'), fullPage: true })

  console.log('[ox] left  =', JSON.stringify(left))
  console.log('[ox] right =', JSON.stringify(right))

  expect(left.ix).toBe(2)
  expect(right.ix).toBe(13)

  const dMarker = Math.hypot(left.marker[0] - right.marker[0],
                             left.marker[1] - right.marker[1])
  expect(dMarker, 'the triangle marker did not move between the two halves')
    .toBeGreaterThan(0.01)

  const dHi = Math.hypot(left.highlight.x - right.highlight.x,
                         left.highlight.y - right.highlight.y,
                         left.highlight.z - right.highlight.z)
  expect(dHi, 'the sphere highlight did not move between the two halves')
    .toBeGreaterThan(0.05)

  // …and the camera turned to face it (the app's "picking rotates the sphere").
  expect(Math.abs(left.azimuth - right.azimuth)
    + Math.abs(left.elevation - right.elevation),
    'the sphere camera did not turn towards the picked orientation')
    .toBeGreaterThan(1.0)
})

test('2b) the picked orientation ends up CENTRED on the sphere', async () => {
  // The camera must FACE the pick, not merely move. `atan2(vy, vx) - 90°` aims
  // the same direction 180° out and parks the highlight on the sphere's far
  // edge — which every "did the state change" assertion happily passes. So find
  // the white highlight disk in the pixels and require it near the middle.
  const centroid = async () => {
    const box = (await page.locator('#ox-fig').boundingBox())!
    const shot = await page.screenshot({ clip: box })
    return await page.evaluate(async (b64: string) => {
      const img = await new Promise<HTMLImageElement>((res, rej) => {
        const i = new Image(); i.onload = () => res(i); i.onerror = rej
        i.src = 'data:image/png;base64,' + b64
      })
      const cv = document.createElement('canvas')
      cv.width = img.width; cv.height = img.height
      const cx = cv.getContext('2d')!
      cx.drawImage(img, 0, 0)
      const d = cx.getImageData(0, 0, cv.width, cv.height).data
      // The sphere is the RIGHT third of the figure.
      const x0 = Math.floor(cv.width * (2 / 3))
      let sx = 0, sy = 0, n = 0
      for (let y = 0; y < cv.height; y++) {
        for (let x = x0; x < cv.width; x++) {
          const i = 4 * (y * cv.width + x)
          if (d[i] > 240 && d[i + 1] > 240 && d[i + 2] > 240) { sx += x; sy += y; n++ }
        }
      }
      return n ? { fx: (sx / n - x0) / (cv.width - x0), fy: sy / n / cv.height, n }
        : { fx: -1, fy: -1, n: 0 }
    }, shot.toString('base64'))
  }

  for (const [iy, ix] of [[8, 2], [8, 13], [1, 15]] as const) {
    await page.evaluate(([a, b]) => (window as any).__ox.setPick(a, b), [iy, ix])
    await page.waitForTimeout(900)
    const c = await centroid()
    console.log(`[ox] highlight centroid at (${iy},${ix}) =`, JSON.stringify(c))
    expect(c.n, `no white highlight visible at (${iy},${ix})`).toBeGreaterThan(30)
    // Within the middle ~40% of the sphere panel in both axes.
    expect(Math.abs(c.fx - 0.5), `highlight off-centre horizontally at (${iy},${ix})`)
      .toBeLessThan(0.2)
    expect(Math.abs(c.fy - 0.5), `highlight off-centre vertically at (${iy},${ix})`)
      .toBeLessThan(0.2)
  }
  await page.screenshot({ path: join(SHOTS, '03b-centred.png'), fullPage: true })
})

test('3) the sphere really redraws when the camera turns', async () => {
  // The state says the camera moved; this says the PIXELS did. Without it a
  // panel that accepted the azimuth and never repainted would pass test 2.
  const shoot = async (name: string) => {
    const box = await page.locator('#ox-fig').boundingBox()
    return await page.screenshot({ clip: box!, path: join(SHOTS, name) })
  }
  await page.evaluate(() => (window as any).__ox.setPick(8, 2))
  await page.waitForTimeout(900)
  const a = await shoot('04-sphere-left.png')
  await page.evaluate(() => (window as any).__ox.setPick(8, 13))
  await page.waitForTimeout(900)
  const b = await shoot('05-sphere-right.png')

  const changed = await page.evaluate(async ([p, q]: [string, string]) => {
    const load = (s: string) => new Promise<HTMLImageElement>((res, rej) => {
      const i = new Image(); i.onload = () => res(i); i.onerror = rej
      i.src = 'data:image/png;base64,' + s
    })
    const [ia, ib] = await Promise.all([load(p), load(q)])
    const cv = document.createElement('canvas')
    cv.width = ia.width; cv.height = ia.height
    const cx = cv.getContext('2d')!
    cx.drawImage(ia, 0, 0)
    const da = cx.getImageData(0, 0, cv.width, cv.height).data
    cx.clearRect(0, 0, cv.width, cv.height); cx.drawImage(ib, 0, 0)
    const db = cx.getImageData(0, 0, cv.width, cv.height).data
    let n = 0
    for (let i = 0; i < da.length; i += 4) {
      if (Math.abs(da[i] - db[i]) + Math.abs(da[i + 1] - db[i + 1])
        + Math.abs(da[i + 2] - db[i + 2]) > 30) n++
    }
    return { n, total: da.length / 4 }
  }, [a.toString('base64'), b.toString('base64')])
  console.log('[ox] pick-to-pick pixel diff =', JSON.stringify(changed))
  expect(changed.n / changed.total,
    'the figure looks identical at two very different orientations')
    .toBeGreaterThan(0.005)
})

test('4) X / Y / Z re-colours every panel from the packed blob', async () => {
  const before = await page.screenshot({ clip: (await page.locator('#ox-fig').boundingBox())!, path: join(SHOTS, '06-dir-z.png') })
  await page.evaluate(() => (window as any).__ox.setDirection('x'))
  await page.waitForTimeout(1_200)
  const s = await oxState()
  expect(s.dir).toBe('x')
  const after = await page.screenshot({ clip: (await page.locator('#ox-fig').boundingBox())!, path: join(SHOTS, '07-dir-x.png') })

  const changed = await page.evaluate(async ([p, q]: [string, string]) => {
    const load = (t: string) => new Promise<HTMLImageElement>((res, rej) => {
      const i = new Image(); i.onload = () => res(i); i.onerror = rej
      i.src = 'data:image/png;base64,' + t
    })
    const [ia, ib] = await Promise.all([load(p), load(q)])
    const cv = document.createElement('canvas')
    cv.width = ia.width; cv.height = ia.height
    const cx = cv.getContext('2d')!
    cx.drawImage(ia, 0, 0)
    const da = cx.getImageData(0, 0, cv.width, cv.height).data
    cx.clearRect(0, 0, cv.width, cv.height); cx.drawImage(ib, 0, 0)
    const db = cx.getImageData(0, 0, cv.width, cv.height).data
    let n = 0
    for (let i = 0; i < da.length; i += 4) {
      if (Math.abs(da[i] - db[i]) + Math.abs(da[i + 1] - db[i + 1])
        + Math.abs(da[i + 2] - db[i + 2]) > 30) n++
    }
    return { n, total: da.length / 4 }
  }, [before.toString('base64'), after.toString('base64')])
  console.log('[ox] Z→X pixel diff =', JSON.stringify(changed))
  expect(changed.n / changed.total, 'switching to IPF-X changed nothing')
    .toBeGreaterThan(0.005)
  // Still painted afterwards — a re-colour that blanks a panel is a regression
  // the diff above would happily call a pass.
  const inks = await canvasInk()
  expect(inks.filter((c) => c.ink > 200).length).toBeGreaterThanOrEqual(3)
  await page.evaluate(() => (window as any).__ox.setDirection('z'))
  await page.waitForTimeout(1_000)
})

test('4b) …including the SPHERE, whose points ride a separate channel', async () => {
  // Test 4's whole-figure diff passes on the map and the triangle alone, and
  // that is exactly what happened first: the 3-D cloud lives in
  // `panel_<id>_geom`, not in the view json, so writing `vertices` into the
  // view json re-coloured nothing and no assertion noticed. Crop the sphere.
  const sphereShot = async (name: string) => {
    const b = (await page.locator('#ox-fig').boundingBox())!
    return await page.screenshot({
      clip: { x: b.x + b.width * (2 / 3), y: b.y, width: b.width / 3, height: b.height },
      path: join(SHOTS, name),
    })
  }
  await page.evaluate(() => (window as any).__ox.setPick(8, 13))
  await page.waitForTimeout(900)
  const z = await sphereShot('09-sphere-dir-z.png')
  await page.evaluate(() => (window as any).__ox.setDirection('y'))
  await page.waitForTimeout(1_500)
  const y = await sphereShot('10-sphere-dir-y.png')

  const changed = await page.evaluate(async ([p, q]: [string, string]) => {
    const load = (t: string) => new Promise<HTMLImageElement>((res, rej) => {
      const i = new Image(); i.onload = () => res(i); i.onerror = rej
      i.src = 'data:image/png;base64,' + t
    })
    const [ia, ib] = await Promise.all([load(p), load(q)])
    const cv = document.createElement('canvas')
    cv.width = ia.width; cv.height = ia.height
    const cx = cv.getContext('2d')!
    cx.drawImage(ia, 0, 0)
    const da = cx.getImageData(0, 0, cv.width, cv.height).data
    cx.clearRect(0, 0, cv.width, cv.height); cx.drawImage(ib, 0, 0)
    const db = cx.getImageData(0, 0, cv.width, cv.height).data
    let n = 0
    for (let i = 0; i < da.length; i += 4) {
      if (Math.abs(da[i] - db[i]) + Math.abs(da[i + 1] - db[i + 1])
        + Math.abs(da[i + 2] - db[i + 2]) > 30) n++
    }
    return { n, total: da.length / 4 }
  }, [z.toString('base64'), y.toString('base64')])
  console.log('[ox] sphere Z→Y diff =', JSON.stringify(changed))
  expect(changed.n / changed.total,
    'the sphere is identical for IPF-Z and IPF-Y — its geometry channel never updated')
    .toBeGreaterThan(0.01)
  await page.evaluate(() => (window as any).__ox.setDirection('z'))
  await page.waitForTimeout(1_000)
})

test('5) a REAL crosshair drag moves the pick', async () => {
  // Everything above drives __ox directly. This proves the WIDGET EVENTS reach
  // it: grab anyplotlib's own crosshair on the map panel and drag it across the
  // nav grid. The crosshair sits where it was mounted (the nav centre) — setPick
  // moves the marker and the highlight, not the widget — so press there and
  // drag to a position in the other half.
  const before = await oxState()
  const from = await page.evaluate(() => (window as any).__ox.navToPage(8, 8))
  const to = await page.evaluate(() => (window as any).__ox.navToPage(3, 13))
  expect(from, 'no map panel geometry to drag on').toBeTruthy()

  await page.mouse.move(from.x, from.y)
  await page.mouse.down()
  await page.mouse.move(to.x, to.y, { steps: 12 })
  await page.mouse.up()
  await page.waitForTimeout(900)
  const after = await oxState()
  await page.screenshot({ path: join(SHOTS, '08-after-drag.png'), fullPage: true })
  console.log('[ox] drag', JSON.stringify(before), '→', JSON.stringify(after))
  expect(after.pick, 'dragging the crosshair did not move the pick')
    .not.toBe(before.pick)
  // …and it landed where the pointer was let go, not just somewhere else.
  expect(after.ix).toBe(13)
  expect(after.iy).toBe(3)
})
