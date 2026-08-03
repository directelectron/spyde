/**
 * orientation_report_embed.spec.ts — the embedded IPF explorer in an exported
 * HTML report, driven in a REAL browser (plain chromium over file://; the page
 * must work with no app, no backend and no network — that is its whole point).
 *
 * The page is the app's two orientation windows: the IPF MAP with a crosshair,
 * and ONE explorer view chosen by the app's two independent toggle pairs,
 * [2D | 3D] and [Points | Heatmap]. The fixture
 * (spyde/tests/gen_orientation_embed.py) puts ONE orientation in the left half
 * of the nav grid and a very different one in the right half, so a pick on each
 * side must land in a different place on the triangle AND swing the sphere's
 * camera. If either stays put, the pick is not driving the view.
 *
 * Geometry goes through window.__ox — the SAME setPick/select/setDirection the
 * crosshair and the toggles call — plus one REAL pointer drag on the anyplotlib
 * crosshair as a smoke test that widget events actually flow.
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
test.setTimeout(180_000)

/** Per-canvas ink across the WHOLE page (map + whichever explorer view is up):
 *  how many pixels are neither transparent nor near-black. A panel that drew
 *  nothing scores 0. */
const canvasInk = () => page.evaluate(() => {
  const out: Array<{ w: number; h: number; ink: number; colorful: number }> = []
  for (const el of document.querySelectorAll('#ox-root canvas')) {
    const c = el as HTMLCanvasElement
    if (!c.offsetParent && c.parentElement?.closest('div[data-view]')
      && !c.parentElement.closest('div[data-view].ox-on')) continue   // hidden view
    const ctx = c.getContext('2d')
    if (!ctx || !c.width || !c.height) {
      out.push({ w: c.width, h: c.height, ink: -1, colorful: -1 }); continue
    }
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

const painted = async () => (await canvasInk()).filter((c) => c.ink > 200).length

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
  // shell, which is what CI has. NOT `channel: 'chromium'` +
  // --enable-unsafe-webgpu: that gets headless a real navigator.gpu, and it
  // paints fine locally, but on the CI runner it produced correctly-sized
  // panels with ZERO pixels in any of them, the plain 2-D map included. The
  // 3-D views do not need WebGPU (anyplotlib falls back to Canvas2D) and every
  // assertion here is about pixels rather than which path drew them.
  browser = await chromium.launch()
  page = await browser.newPage({ viewport: { width: 1400, height: 700 } })
  await page.goto('file:///' + htmlPath.replace(/\\/g, '/'))
  await page.waitForSelector('#ox-root[data-ready="1"]', { timeout: 60_000 })
  // Wait for the PIXELS, not for a guess at how long they take: `data-ready`
  // only says the script finished, and a fixed sleep is what turns a loaded
  // runner into "nothing painted".
  await expect.poll(painted, { timeout: 60_000, message: 'the panels never painted' })
    .toBeGreaterThanOrEqual(2)
})

test.afterAll(async () => { await browser?.close() })

const oxState = () => page.evaluate(() => (window as any).__ox.state())
const figBox = async () => (await page.locator('#ox-root .ox-row').boundingBox())!

/** Fraction of pixels that differ between two screenshots of the same clip. */
async function diffFraction(a: Buffer, b: Buffer) {
  return await page.evaluate(async ([p, q]: [string, string]) => {
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
    return n / (da.length / 4)
  }, [a.toString('base64'), b.toString('base64')])
}

test('1) the map and the default 2-D scatter both paint', async () => {
  const s = await oxState()
  expect(s.dir).toBe('z')
  expect(s.view).toBe('2d-points')
  expect(s.marker, 'the scatter has no pick marker').toBeTruthy()
  // Only the selected view is mounted — the other three are paid for when asked
  // for, not up front.
  expect(s.mounted, 'more than the selected view was mounted').toEqual(['2d-points'])

  await page.screenshot({ path: join(SHOTS, '01-mounted.png'), fullPage: true })
  const inks = await canvasInk()
  console.log('[ox] canvas ink =', JSON.stringify(inks))
  expect(await painted()).toBeGreaterThanOrEqual(2)
  expect(inks.reduce((a, c) => a + Math.max(0, c.colorful), 0),
    'nothing chromatic — the IPF colours did not arrive').toBeGreaterThan(500)
})

test('2) the scatter is IPF-COLOURED, not one flat colour', async () => {
  // The whole point of the triangle: every point wears its own IPF key colour.
  // The fixture is two very different orientations, so the drawn cloud must
  // carry at least two distinct fill colours.
  const distinct = await page.evaluate(() => {
    const st = (window as any).__ox
    const h = (st._h().H)['2d-points']
    const pid = st._h().FIGS['2d-points'].panels[0]
    const pj = JSON.parse(h.get('panel_' + pid + '_json'))
    const g = (pj.markers || []).filter((m: any) => m.type === 'points')[0]
    const fc = g && g.fill_color
    return Array.isArray(fc) ? new Set(fc).size : (fc ? 1 : 0)
  })
  console.log('[ox] distinct cloud colours =', distinct)
  expect(distinct, 'the IPF triangle is a single flat colour').toBeGreaterThan(1)
})

test('3) picking the two nav halves gives two different orientations', async () => {
  await page.evaluate(() => (window as any).__ox.setPick(8, 2))
  await page.waitForTimeout(500)
  const left = await oxState()
  await page.screenshot({ path: join(SHOTS, '02-pick-left.png'), fullPage: true })

  await page.evaluate(() => (window as any).__ox.setPick(8, 13))
  await page.waitForTimeout(500)
  const right = await oxState()
  await page.screenshot({ path: join(SHOTS, '03-pick-right.png'), fullPage: true })

  console.log('[ox] left  =', JSON.stringify(left.marker),
    ' right =', JSON.stringify(right.marker))
  expect(left.ix).toBe(2)
  expect(right.ix).toBe(13)
  const d = Math.hypot(left.marker[0] - right.marker[0],
                       left.marker[1] - right.marker[1])
  expect(d, 'the marker did not move between the two halves').toBeGreaterThan(0.01)
})

test('4) all four [2D|3D] x [Points|Heatmap] states render', async () => {
  const states: Array<[string, string, string]> = [
    ['2d', 'points', '04-2d-points.png'],
    ['2d', 'heatmap', '05-2d-heatmap.png'],
    ['3d', 'points', '06-3d-points.png'],
    ['3d', 'heatmap', '07-3d-heatmap.png'],
  ]
  const shots: Record<string, Buffer> = {}
  for (const [dim, style, file] of states) {
    await page.getByTestId(`ox-dim-${dim}`).click()
    await page.getByTestId(`ox-style-${style}`).click()
    // Each view mounts on FIRST selection, so the first visit to a state has a
    // mount + a first draw to do. Wait for the pixels, not for a fixed guess.
    await expect.poll(painted, { timeout: 30_000, message: `${dim}/${style} never painted` })
      .toBeGreaterThanOrEqual(2)
    const s = await oxState()
    expect(s.view).toBe(`${dim}-${style === 'points' ? 'points' : 'heat'}`)
    shots[s.view] = await page.screenshot({ clip: await figBox(), path: join(SHOTS, file) })
    const inks = await canvasInk()
    console.log(`[ox] ${dim}/${style} painted=${inks.filter((c) => c.ink > 200).length}`,
      'colorful=', inks.reduce((a, c) => a + Math.max(0, c.colorful), 0))
  }
  // The four are genuinely DIFFERENT pictures, not the same panel relabelled.
  const names = Object.keys(shots)
  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      const f = await diffFraction(shots[names[i]], shots[names[j]])
      expect(f, `${names[i]} and ${names[j]} render identically`).toBeGreaterThan(0.005)
    }
  }
})

test('5) the picked orientation ends up CENTRED on the sphere', async () => {
  // The camera must FACE the pick, not merely move. `atan2(vy, vx) - 90` aims
  // the same direction 180 out and parks the highlight on the sphere's far
  // edge — which every "did the state change" assertion happily passes. So find
  // the white highlight disk in the pixels and require it near the middle.
  await page.getByTestId('ox-dim-3d').click()
  await page.getByTestId('ox-style-points').click()
  await expect.poll(painted, { timeout: 30_000 }).toBeGreaterThanOrEqual(2)

  // Clip to the view's OWN box, read off the page — a guessed fraction of the
  // row measures the wrong pixels and still passes (it did: an eyeballed 0.62
  // put the panel centre at 0.35 and a ±0.22 tolerance waved it through).
  const centroid = async () => {
    const box = await page.evaluate(() => (window as any).__ox.viewRect())
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
      let sx = 0, sy = 0, n = 0
      for (let y = 0; y < cv.height; y++) {
        for (let x = 0; x < cv.width; x++) {
          const i = 4 * (y * cv.width + x)
          if (d[i] > 240 && d[i + 1] > 240 && d[i + 2] > 240) { sx += x; sy += y; n++ }
        }
      }
      return n ? { fx: sx / n / cv.width, fy: sy / n / cv.height, n }
        : { fx: -1, fy: -1, n: 0 }
    }, shot.toString('base64'))
  }

  const shots: Buffer[] = []
  for (const [iy, ix] of [[8, 2], [8, 13], [1, 15]] as const) {
    await page.evaluate(([a, b]) => (window as any).__ox.setPick(a, b), [iy, ix])
    await page.waitForTimeout(900)
    const c = await centroid()
    console.log(`[ox] highlight centroid at (${iy},${ix}) =`, JSON.stringify(c))
    expect(c.n, `no white highlight visible at (${iy},${ix})`).toBeGreaterThan(30)
    expect(Math.abs(c.fx - 0.5), `off-centre horizontally at (${iy},${ix})`)
      .toBeLessThan(0.12)
    expect(Math.abs(c.fy - 0.5), `off-centre vertically at (${iy},${ix})`)
      .toBeLessThan(0.12)
    shots.push(await page.screenshot({
      clip: await page.evaluate(() => (window as any).__ox.viewRect()),
    }))
  }
  // The highlight staying put is the POINT (the camera centres it), so it
  // cannot also be the evidence that the pick landed. The rest of the sphere
  // must move: three very different orientations, three different pictures.
  expect(await diffFraction(shots[0], shots[1]),
    'the sphere is identical at two very different orientations')
    .toBeGreaterThan(0.01)
  expect(await diffFraction(shots[1], shots[2]),
    'the sphere is identical at two very different orientations')
    .toBeGreaterThan(0.01)
  await page.screenshot({ path: join(SHOTS, '08-centred.png'), fullPage: true })
})

test('6) X / Y / Z re-colours the MAP, the scatter and the sphere', async () => {
  // Each of the three has a different update path — an overlay canvas, the
  // marker json, the geometry channel — so each gets its own check. The scatter
  // one is the regression that shipped: `facecolors` is the PYTHON kwarg name,
  // the wire calls it `fill_color`, and writing the wrong one moved the points
  // while leaving the previous direction's colours on them.
  await page.getByTestId('ox-dim-2d').click()
  await page.getByTestId('ox-style-points').click()
  await expect.poll(painted, { timeout: 30_000 }).toBeGreaterThanOrEqual(2)
  await page.evaluate(() => (window as any).__ox.setDirection('z'))
  await page.waitForTimeout(900)

  const colours = () => page.evaluate(() => {
    const st = (window as any).__ox
    const h = st._h().H['2d-points']
    const pid = st._h().FIGS['2d-points'].panels[0]
    const pj = JSON.parse(h.get('panel_' + pid + '_json'))
    const g = (pj.markers || []).filter((m: any) => m.type === 'points')[0]
    return (g.fill_color || []).slice(0, 200).join(',')
  })
  const beforeCols = await colours()
  const beforeShot = await page.screenshot({ clip: await figBox(), path: join(SHOTS, '09-dir-z.png') })

  await page.getByTestId('ox-dir-x').click()
  await page.waitForTimeout(1_500)
  const afterCols = await colours()
  const afterShot = await page.screenshot({ clip: await figBox(), path: join(SHOTS, '10-dir-x.png') })

  expect((await oxState()).dir).toBe('x')
  expect(afterCols, 'the scatter kept the previous direction\'s IPF colours')
    .not.toBe(beforeCols)
  const f = await diffFraction(beforeShot, afterShot)
  console.log('[ox] Z→X pixel diff =', f)
  expect(f, 'switching to IPF-X changed nothing on screen').toBeGreaterThan(0.005)
  expect(await painted()).toBeGreaterThanOrEqual(2)
})

test('6b) …including the SPHERE, whose points ride a separate channel', async () => {
  // Test 6's whole-figure diff passes on the map and the scatter alone, and
  // that is exactly what happened first: a 3-D panel's cloud lives in
  // `panel_<id>_geom`, not in the view json, so writing `vertices` into the
  // view json re-coloured nothing and no assertion noticed.
  await page.getByTestId('ox-dim-3d').click()
  await page.getByTestId('ox-style-points').click()
  await expect.poll(painted, { timeout: 30_000 }).toBeGreaterThanOrEqual(2)
  await page.evaluate(() => (window as any).__ox.setPick(8, 13))
  await page.evaluate(() => (window as any).__ox.setDirection('z'))
  await page.waitForTimeout(1_200)

  const clip = async () => await page.evaluate(() => (window as any).__ox.viewRect())
  const z = await page.screenshot({ clip: await clip(), path: join(SHOTS, '11-sphere-z.png') })
  await page.getByTestId('ox-dir-y').click()
  await page.waitForTimeout(1_800)
  const y = await page.screenshot({ clip: await clip(), path: join(SHOTS, '12-sphere-y.png') })
  const f = await diffFraction(z, y)
  console.log('[ox] sphere Z→Y diff =', f)
  expect(f, 'the sphere is identical for IPF-Z and IPF-Y — its geometry channel never updated')
    .toBeGreaterThan(0.01)
  await page.getByTestId('ox-dir-z').click()
  await page.waitForTimeout(1_000)
})

test('7) a REAL crosshair drag moves the pick', async () => {
  // Everything above drives __ox directly. This proves the WIDGET EVENTS reach
  // it: grab anyplotlib's own crosshair on the map and drag it across the nav
  // grid. The crosshair sits where it was mounted (the nav centre) — setPick
  // moves the marker, not the widget — so press there and drag elsewhere.
  const before = await oxState()
  const from = await page.evaluate(() => (window as any).__ox.navToPage(8, 8))
  const to = await page.evaluate(() => (window as any).__ox.navToPage(3, 13))
  expect(from, 'no map geometry to drag on').toBeTruthy()

  await page.mouse.move(from.x, from.y)
  await page.mouse.down()
  await page.mouse.move(to.x, to.y, { steps: 12 })
  await page.mouse.up()
  await page.waitForTimeout(900)
  const after = await oxState()
  await page.screenshot({ path: join(SHOTS, '13-after-drag.png'), fullPage: true })
  console.log('[ox] drag pick', before.pick, '→', after.pick,
    `(${after.iy}, ${after.ix})`)
  expect(after.pick, 'dragging the crosshair did not move the pick')
    .not.toBe(before.pick)
  expect(after.ix).toBe(13)
  expect(after.iy).toBe(3)
})
