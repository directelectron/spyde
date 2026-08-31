/**
 * ipf_two_window.spec.ts — the TWO-WINDOW orientation result, on real pixels.
 *
 * An orientation run must now open TWO windows:
 *   window 1  the IPF-X / IPF-Y / IPF-Z projection MAPS (chip strip, ⌘-tile),
 *   window 2  the inverse pole figure itself, with two INDEPENDENT toggles —
 *             [2D | 3D] and [Points | Heatmap] — and a crosshair-driven
 *             orientation marker that ROTATES the sphere.
 *
 * Driven with real Dask + the bundled synthetic Si-grains scan via the
 * test-only `run_test_orientation` (the cheapest deterministic dense OM — the
 * same recipe orientation_workflow / report_ipf3d use; there is no bundled OM
 * loader).
 *
 * Pixel proof: WebGPU canvases refuse getImageData, so every probe decodes a
 * composited page SCREENSHOT (the gpu_image_parity / report_ipf3d pattern) and
 * classifies pixels. A blank/black panel fails. Each of the four toggle states
 * is captured to ipf_two_window_shots/ and read by the author.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import { mkdirSync } from 'fs'
const { launchApp, raiseWindow, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'ipf_two_window_shots')

let ctx: Awaited<ReturnType<typeof launchApp>>
let ipfId = ''        // window 2 — owns the toggles
let mapId = ''        // window 1 — the projection maps

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)
  await page.waitForTimeout(2500)
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

/** Decode a screenshot PNG in-page → {colorful, dark, total} pixel counts.
 *  Works for WebGPU content because the input is the COMPOSITED screenshot. */
async function shotStats(page: any, buf: Buffer):
    Promise<{ colorful: number; dark: number; total: number }> {
  return await page.evaluate(async (b64: string) => {
    const img = await new Promise<HTMLImageElement>((res, rej) => {
      const i = new Image(); i.onload = () => res(i); i.onerror = rej
      i.src = 'data:image/png;base64,' + b64
    })
    const cv = document.createElement('canvas')
    cv.width = img.width; cv.height = img.height
    const c2 = cv.getContext('2d')!
    c2.drawImage(img, 0, 0)
    const d = c2.getImageData(0, 0, cv.width, cv.height).data
    let colorful = 0, dark = 0
    for (let p = 0; p < d.length; p += 4) {
      const r = d[p], g = d[p + 1], b = d[p + 2]
      if (Math.max(r, g, b) - Math.min(r, g, b) > 40) colorful++
      if (0.3 * r + 0.59 * g + 0.11 * b < 40) dark++
    }
    return { colorful, dark, total: cv.width * cv.height }
  }, buf.toString('base64'))
}

const ipfWin = (page: any) => page.getByTestId('subwindow')
  .filter({ has: page.getByTestId(`ipf-view-toggle-${ipfId}`) }).first()

/** The state of every key-overlay canvas (anyplotlib draws keys on their own
 *  canvas at z-index 7) across all figure iframes.
 *
 *  Exists because "the key did not appear" has three very different causes and
 *  a pixel diff cannot tell them apart: NO canvas means `add_key` never reached
 *  the figure; `display:none` means the figure declares no key; `painted:0`
 *  means the key IS declared but either the hover flag never flipped or the key
 *  image has not decoded (anyplotlib decodes key images asynchronously and
 *  redraws from onload, so a still-decoding key draws nothing). */
async function keyCanvases(page: any) {
  const out: any[] = []
  for (const fr of page.frames()) {
    try {
      const r = await fr.evaluate(() => {
        const cs = Array.from(document.querySelectorAll('canvas')) as HTMLCanvasElement[]
        return cs.filter(c => c.style.zIndex === '7').map(c => {
          let painted = -1
          try {
            const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data
            painted = 0
            for (let i = 3; i < d.length; i += 4) if (d[i] > 0) painted++
          } catch { /* tainted or no 2-D context */ }
          return { w: c.width, h: c.height, display: c.style.display, painted }
        })
      })
      out.push(...r)
    } catch { /* a frame can detach mid-walk */ }
  }
  return out
}

/** Screenshot window 2 and return its pixel stats. */
async function shootIpf(name: string) {
  const { page } = ctx
  const bb = await ipfWin(page).boundingBox()
  expect(bb, `window 2 has no box for ${name}`).not.toBeNull()
  const shot = await page.screenshot({ clip: bb!, path: join(SHOTS, name) })
  return await shotStats(page, shot)
}

test('1) an OM run opens BOTH windows: the maps and the IPF explorer', async () => {
  const { page } = ctx
  const before = await page.getByTestId('subwindow').count()
  await backendAction(page, 'run_test_orientation')

  // Signal-based: the explorer window is the one that owns the toggle group.
  const toggle = page.getByTestId(/^ipf-view-toggle-/).first()
  await expect(toggle).toBeAttached({ timeout: 200_000 })
  ipfId = (await toggle.getAttribute('data-testid'))!.replace('ipf-view-toggle-', '')

  // The MAP window is the one carrying the IPF-X / IPF-Y / IPF-Z chip strip.
  const chip = page.getByTestId(/^view-chip-IPF-X-/).first()
  await expect(chip).toBeAttached({ timeout: 60_000 })
  mapId = (await chip.getAttribute('data-testid'))!.replace('view-chip-IPF-X-', '')
  expect(mapId, 'the maps and the IPF must be SEPARATE windows').not.toBe(ipfId)

  // Two new windows, not one.
  expect(await page.getByTestId('subwindow').count()).toBeGreaterThanOrEqual(before + 2)
  await page.waitForTimeout(2000)
  await page.screenshot({ path: join(SHOTS, '01-both-windows.png'), fullPage: false })
  ctx.assertNoJsErrors()
})

test('2) window 1 shows the X, Y and Z projections', async () => {
  const { page } = ctx
  const mapWin = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId(`view-chip-IPF-X-${mapId}`) }).first()
  for (const d of ['X', 'Y', 'Z']) {
    await expect(page.getByTestId(`view-chip-IPF-${d}-${mapId}`)).toBeVisible()
  }
  // Another window may have opened over this one since the chips appeared;
  // raise it so the chip clicks land (see raiseWindow).
  await raiseWindow(mapWin)
  // Each projection on its own …
  for (const d of ['X', 'Y', 'Z']) {
    await page.getByTestId(`view-chip-IPF-${d}-${mapId}`).click()
    await page.waitForTimeout(1200)
    const bb = (await mapWin.boundingBox())!
    const shot = await page.screenshot({ clip: bb, path: join(SHOTS, `02-map-ipf-${d.toLowerCase()}.png`) })
    const st = await shotStats(page, shot)
    console.log(`[two-window] IPF-${d} map stats =`, JSON.stringify(st))
    expect(st.colorful, `IPF-${d} map is not an orientation map`).toBeGreaterThan(200)
  }
  // … and all three tiled side by side (⌘-click, the app's compare idiom).
  await page.getByTestId(`view-chip-IPF-X-${mapId}`).click()
  await page.getByTestId(`view-chip-IPF-Y-${mapId}`).click({ modifiers: ['Meta'] })
  await page.getByTestId(`view-chip-IPF-Z-${mapId}`).click({ modifiers: ['Meta'] })
  await page.waitForTimeout(3000)
  const bb = (await mapWin.boundingBox())!
  const shot = await page.screenshot({ clip: bb, path: join(SHOTS, '03-map-xyz-tiled.png') })
  const st = await shotStats(page, shot)
  console.log('[two-window] tiled XYZ stats =', JSON.stringify(st))
  expect(st.colorful, 'the tiled X/Y/Z comparison is blank').toBeGreaterThan(200)
  ctx.assertNoJsErrors()
})

test('2b) the IPF colour key appears on HOVER, inside the map figure', async () => {
  // The key used to be a separate figure floated over the window by the
  // renderer; it is now an anyplotlib key overlay owned by the map figure
  // (Plot2D.add_key, hover_only). So there is no DOM node to assert on — the
  // only honest check is that hovering the panel changes the pixels, and that
  // moving away puts them back.
  const { page } = ctx
  await page.getByTestId(`view-chip-IPF-Z-${mapId}`).click()
  await page.waitForTimeout(1500)
  const mapWin = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId(`view-chip-IPF-X-${mapId}`) }).first()
  const bb = (await mapWin.boundingBox())!

  await page.mouse.move(5, 5)                       // pointer well away
  await page.waitForTimeout(600)
  const cold = await page.screenshot({ clip: bb, path: join(SHOTS, '11-key-cold.png') })
  console.log('[two-window] key canvases COLD =', JSON.stringify(await keyCanvases(page)))

  // Onto the panel. Aim at the CENTRE of the map window, not the bottom-right
  // quadrant: `mouseenter` fires off the panel's OVERLAY canvas, which anyplotlib
  // sizes to the IMAGE rect alone (_resizePanelDOM), not to the whole panel — so
  // a corner-ward point sits inside the letterbox margin on any aspect ratio but
  // the one this was eyeballed at. The key is pinned bottom-right regardless of
  // where the pointer is, so the centre proves the same thing with no geometry
  // assumption. Two moves: the first lands in the panel, the second keeps the
  // pointer there and gives Chromium a second hit-test to derive enter from.
  await page.mouse.move(bb.x + bb.width * 0.5, bb.y + bb.height * 0.55)
  await page.mouse.move(bb.x + bb.width * 0.5, bb.y + bb.height * 0.56)
  await page.waitForTimeout(2000)                   // + async key-image decode
  const hot = await page.screenshot({ clip: bb, path: join(SHOTS, '12-key-hover.png') })
  console.log('[two-window] key canvases HOT =', JSON.stringify(await keyCanvases(page)))

  const changed = await page.evaluate(async ([a, b]) => {
    const load = (d: string) => new Promise<HTMLImageElement>(res => {
      const i = new Image(); i.onload = () => res(i); i.src = 'data:image/png;base64,' + d
    })
    const [ia, ib] = await Promise.all([load(a), load(b)])
    const c = document.createElement('canvas')
    c.width = ia.width; c.height = ia.height
    const cx = c.getContext('2d')!
    cx.drawImage(ia, 0, 0)
    const pa = cx.getImageData(0, 0, c.width, c.height).data
    cx.clearRect(0, 0, c.width, c.height); cx.drawImage(ib, 0, 0)
    const pb = cx.getImageData(0, 0, c.width, c.height).data
    let n = 0
    for (let i = 0; i < pa.length; i += 4) {
      if (Math.abs(pa[i] - pb[i]) + Math.abs(pa[i + 1] - pb[i + 1])
          + Math.abs(pa[i + 2] - pb[i + 2]) > 30) n++
    }
    return { changed: n, total: pa.length / 4 }
  }, [cold.toString('base64'), hot.toString('base64')])

  // Park the pointer OUTSIDE the map before asserting: these tests share one
  // app instance serially, and leaving the cursor hovering a panel would carry
  // a hover state (and a drawn key) into the crosshair test below.
  await page.mouse.move(5, 5)
  await page.waitForTimeout(300)

  console.log('[two-window] hover-key diff =', JSON.stringify(changed))
  // The key occupies ~0.26 of the short edge, so it is a small but unmistakable
  // fraction of the panel. A zero here means hover_only never drew.
  expect(changed.changed, 'hovering the map did not reveal the IPF colour key')
    .toBeGreaterThan(300)
  ctx.assertNoJsErrors()
})

test('3) window 2 has BOTH toggle pairs and renders all four states', async () => {
  const { page } = ctx
  for (const t of ['ipf-view-2d', 'ipf-view-3d', 'ipf-style-points', 'ipf-style-heatmap']) {
    await expect(page.getByTestId(`${t}-${ipfId}`)).toBeVisible()
  }
  const states: Array<[string, string, string]> = [
    ['2d', 'points', '04-2d-points.png'],
    ['2d', 'heatmap', '05-2d-heatmap.png'],
    ['3d', 'points', '06-3d-points.png'],
    ['3d', 'heatmap', '07-3d-heatmap.png'],
  ]
  for (const [dim, style, file] of states) {
    await page.getByTestId(`ipf-view-${dim}-${ipfId}`).click({ force: true })
    await page.getByTestId(`ipf-style-${style}-${ipfId}`).click({ force: true })
    await page.waitForTimeout(3500)          // 3-D needs the GPU probe + a draw
    const st = await shootIpf(file)
    console.log(`[two-window] ${dim}/${style} stats =`, JSON.stringify(st))
    expect(st.colorful, `${dim} · ${style} rendered nothing chromatic`)
      .toBeGreaterThan(120)
  }
  ctx.assertNoJsErrors()
})

test('4) the crosshair rotates the sphere', async () => {
  const { page } = ctx
  await page.getByTestId(`ipf-view-3d-${ipfId}`).click({ force: true })
  await page.getByTestId(`ipf-style-points-${ipfId}`).click({ force: true })
  await page.waitForTimeout(3000)

  await backendAction(page, 'test_ipf_pick', { iy: 0, ix: 0 })
  await page.waitForTimeout(2500)
  const bb = (await ipfWin(page).boundingBox())!
  const a = await page.screenshot({ clip: bb, path: join(SHOTS, '08-rotate-before.png') })

  // A far-away scan pixel → a different grain → a different crystal direction
  // → a visibly different camera. si_grains is 6x6 nav, so (5,5) is the
  // opposite corner.
  await backendAction(page, 'test_ipf_pick', { iy: 5, ix: 5 })
  await page.waitForTimeout(2500)
  const b = await page.screenshot({ clip: bb, path: join(SHOTS, '09-rotate-after.png') })

  // The two frames must DIFFER — a static sphere means set_view never landed.
  const diff = await page.evaluate(async ([b1, b2]: [string, string]) => {
    const load = (s: string) => new Promise<HTMLImageElement>((res, rej) => {
      const i = new Image(); i.onload = () => res(i); i.onerror = rej
      i.src = 'data:image/png;base64,' + s
    })
    const [i1, i2] = await Promise.all([load(b1), load(b2)])
    const px = (im: HTMLImageElement) => {
      const c = document.createElement('canvas')
      c.width = im.width; c.height = im.height
      const g = c.getContext('2d')!; g.drawImage(im, 0, 0)
      return g.getImageData(0, 0, c.width, c.height).data
    }
    const d1 = px(i1), d2 = px(i2)
    let changed = 0
    for (let p = 0; p < Math.min(d1.length, d2.length); p += 4) {
      if (Math.abs(d1[p] - d2[p]) + Math.abs(d1[p + 1] - d2[p + 1])
        + Math.abs(d1[p + 2] - d2[p + 2]) > 40) changed++
    }
    return { changed, total: Math.min(d1.length, d2.length) / 4 }
  }, [a.toString('base64'), b.toString('base64')])
  console.log('[two-window] rotate diff =', JSON.stringify(diff))
  expect(diff.changed / diff.total,
    'the sphere did not move when the crosshair picked a new orientation')
    .toBeGreaterThan(0.01)
  ctx.assertNoJsErrors()
})

test('5) X/Y/Z re-colours the explorer window', async () => {
  const { page } = ctx
  await page.getByTestId(`ipf-view-2d-${ipfId}`).click({ force: true })
  await page.getByTestId(`ipf-style-points-${ipfId}`).click({ force: true })
  await page.waitForTimeout(2500)
  const before = await shootIpf('10-dir-z.png')
  await page.getByTestId(`ipf-dir-x-${ipfId}`).click({ force: true })
  await page.waitForTimeout(4000)
  const after = await shootIpf('11-dir-x.png')
  console.log('[two-window] dir z→x =', JSON.stringify(before), JSON.stringify(after))
  expect(after.colorful, 'the IPF went blank after switching to X').toBeGreaterThan(120)
  ctx.assertNoJsErrors()
})
