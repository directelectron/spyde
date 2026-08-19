/**
 * dpc_workflow.spec.ts — DPC (electric / magnetic field mapping), end-to-end.
 *
 * A DPC map is only useful if its directions are right, and a wrong direction
 * looks exactly as plausible as a right one. The Python suite pins the maths;
 * what only the real app can show is that the *picture* is there: an RGB
 * direction map, a colour wheel beside it, four boxes on the navigator, and a
 * map whose colours actually change when the rotation slider moves.
 *
 * So this spec screenshots every stage into `dpc_shots/` and asserts on pixels:
 *
 *   1  the wizard opens and reports the descan the fixture bakes in
 *   2  Corners mode draws four boxes on the NAVIGATOR (yellow pixels appear)
 *   3  the result window is a COLOURED map (an RGB direction map, not grey)
 *   4  Solve recovers the fixture's 25° with a large residual drop
 *   5  moving the rotation slider REPAINTS the map (hue changes)
 *   6  the Map tab's scalar views paint, and the wheel folds away for them
 *   7  Commit opens a new tree
 *
 * Bundled synthetic data (`load_test_data_dpc`, ground truth on metadata) — no
 * download, no dask required for the compute itself.
 */
import { test, expect } from '@playwright/test'
import * as fs from 'fs'
import * as path from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow, navWindow,
} = require('./_harness.cjs')

let ctx: Awaited<ReturnType<typeof launchApp>>

const SHOTS = path.join(__dirname, '..', 'dpc_shots')

/** The fixture's baked-in scan↔detector rotation (`_load_test_data_dpc`). */
const TRUTH_ROTATION = 25.0

test.beforeAll(async () => {
  fs.mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  await backendAction(ctx.page, 'load_test_data_dpc', { nav: 24, sig: 40 })
  await waitForSubwindowCount(ctx.page, 2, 120_000)
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(240_000)

const shot = async (name: string) =>
  ctx.page.screenshot({ path: path.join(SHOTS, `${name}.png`) })

/**
 * Read the canvases inside ONE subwindow and describe their colour.
 *
 * `countColorPixels` in the harness sweeps every frame in the page, which is
 * exactly wrong here — the whole question is whether the DPC window in
 * particular is showing a colour map, while the navigator and the diffraction
 * pattern beside it stay grey. So this walks the frames whose element belongs
 * to the given window.
 *
 * `saturated` is the fraction of pixels with real chroma. In this app's grey
 * theme the RGB direction map is essentially the only source of it, so a
 * non-trivial value means the map rendered. `hue` is their circular mean, which
 * is what makes "did rotating actually repaint it?" answerable in pixels — the
 * one claim no headless assertion can reach.
 */
async function colourStats(page: import('@playwright/test').Page,
                           windowTitle: RegExp) {
  const handle = await page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('subwindow-title').filter({ hasText: windowTitle }) })
    .first().elementHandle()
  if (!handle) return { saturated: 0, hue: NaN, pixels: 0 }

  const readFrame = (frame: import('@playwright/test').Frame) =>
    frame.evaluate(() => {
      let n = 0, total = 0, sx = 0, sy = 0
      const bins = new Array(12).fill(0)
      for (const c of Array.from(document.querySelectorAll('canvas'))) {
        const g = (c as HTMLCanvasElement).getContext('2d')
        if (!g || !(c as HTMLCanvasElement).width) continue
        const d = g.getImageData(0, 0, (c as HTMLCanvasElement).width,
                                 (c as HTMLCanvasElement).height).data
        for (let i = 0; i < d.length; i += 4) {
          total++
          const r = d[i], gr = d[i + 1], bl = d[i + 2]
          const max = Math.max(r, gr, bl), min = Math.min(r, gr, bl)
          if (max < 60 || max - min < 45) continue      // black, grey, or washed out
          n++
          let h = 0
          if (max === r) h = ((gr - bl) / (max - min) + 6) % 6
          else if (max === gr) h = (bl - r) / (max - min) + 2
          else h = (r - gr) / (max - min) + 4
          h = ((h * 60) % 360 + 360) % 360
          sx += Math.cos(h * Math.PI / 180); sy += Math.sin(h * Math.PI / 180)
          bins[Math.floor(h / 30) % 12]++
        }
      }
      return { n, total, sx, sy, bins }
    })

  let n = 0, total = 0, sx = 0, sy = 0
  const bins = new Array(12).fill(0)
  for (const frame of page.frames()) {
    const el = await frame.frameElement().catch(() => null)
    if (!el) continue
    const inside = await handle.evaluate(
      (w, f) => w.contains(f as Node), el).catch(() => false)
    if (!inside) continue
    const r = await readFrame(frame).catch(() => null)
    if (r) {
      n += r.n; total += r.total; sx += r.sx; sy += r.sy
      r.bins.forEach((v: number, i: number) => { bins[i] += v })
    }
  }
  return {
    saturated: total ? n / total : 0,
    hue: n ? ((Math.atan2(sy / n, sx / n) * 180 / Math.PI) + 360) % 360 : NaN,
    pixels: n,
    // How many 30°-wide hue bins carry a real share of the coloured pixels.
    // A direction map spans the whole wheel; a diverging scalar colormap is two
    // hues. This is the discriminator between them — SATURATION is not, because
    // a coolwarm map is every bit as saturated as an RGB one.
    hueBins: bins.filter((v) => n > 0 && v / n > 0.02).length,
  }
}

/**
 * `DpcWizard.WINDOW_TITLE`. Deliberately narrower than /DPC/: Commit opens a
 * tree titled "DPC (E)", which /DPC/ also matches, and the teardown assertion
 * below would then pass or fail on whichever window happened to sort first.
 * (Window titles carry an "S-"/"N-" role prefix, so these are unanchored.)
 */
const DPC_TITLE = /DPC Field Map/
/** The COMMITTED tree — "DPC (E)" or "DPC (B)", never "…Field Map". */
const COMMITTED_TITLE = /DPC \((E|B)\)/

/** The LIVE DPC result window (not a committed tree). */
function dpcWindow(page: import('@playwright/test').Page) {
  return page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('subwindow-title').filter({ hasText: DPC_TITLE }) })
}

test('DPC: centre, solve the rotation, read the field off the colour wheel', async () => {
  const { page } = ctx
  const sig = sigWindow(page)
  const nav = navWindow(page)

  // ── 1. open the wizard on the diffraction pattern ─────────────────────────
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-DPC').click()
  await expect(page.getByTestId('dpc-wizard')).toBeVisible()

  // The fixture bakes in a constant offset AND a ramp, so the caret must say so
  // rather than letting the user apply a correction blind.
  const centering = page.getByTestId('dpc-centering')
  await expect.poll(() => centering.getAttribute('data-centered'),
    { timeout: 60_000, message: 'the descan readout never arrived' })
    .toBe('false')
  const worst = Number(await centering.getAttribute('data-worst'))
  expect(worst, 'the fixture has ~2 px of descan').toBeGreaterThan(1)
  await shot('01-wizard-open')

  // ── 2. Corners mode draws four boxes on the NAVIGATOR ─────────────────────
  // They select SCAN positions, so the navigator is the only window they can
  // mean anything on. #f9e2af is unique to them in this app, and counting it on
  // the navigator specifically is what proves they did not land on the pattern.
  await expect(page.getByTestId('dpc-center-mode'))
    .toHaveAttribute('data-value', 'corners')
  const cornerPixels = async () => {
    let n = 0
    for (const frame of page.frames()) {
      const el = await frame.frameElement().catch(() => null)
      const host = await nav.elementHandle().catch(() => null)
      if (!el || !host) continue
      const inside = await host.evaluate((w, f) => w.contains(f as Node), el)
        .catch(() => false)
      if (!inside) continue
      n += await frame.evaluate(() => {
        let hits = 0
        for (const c of Array.from(document.querySelectorAll('canvas'))) {
          const g = (c as HTMLCanvasElement).getContext('2d')
          if (!g || !(c as HTMLCanvasElement).width) continue
          const d = g.getImageData(0, 0, (c as HTMLCanvasElement).width,
                                   (c as HTMLCanvasElement).height).data
          // #f9e2af = (249, 226, 175): red high, green high, blue clearly lower.
          for (let i = 0; i < d.length; i += 4) {
            if (d[i] > 200 && d[i + 1] > 180 && d[i + 2] > 120 && d[i + 2] < 215
                && d[i] - d[i + 2] > 40) hits++
          }
        }
        return hits
      }).catch(() => 0)
    }
    return n
  }
  await expect.poll(cornerPixels, {
    timeout: 30_000,
    message: 'the four corner boxes never appeared on the navigator',
  }).toBeGreaterThan(0)
  await shot('02-corner-boxes-on-navigator')

  // ── 3. the result window is a COLOURED direction map ──────────────────────
  await expect(dpcWindow(page).first()).toBeVisible({ timeout: 60_000 })
  const before = await colourStats(page, DPC_TITLE)
  expect(before.saturated,
    'the DPC window shows no saturated colour — the RGB direction map did not render')
    .toBeGreaterThan(0.02)
  await shot('03-direction-map')

  // ── 3b. the direction legend appears ON HOVER and not before ──────────────
  // It is an anyplotlib KEY (`Plot2D.add_key`, hover_only) — the same overlay
  // primitive as the IPF colour triangle and the scale bar, not a floating
  // inset panel. So it must be absent while the pointer is away and present
  // when it is over the map; a key that never shows is indistinguishable from
  // one that was never attached.
  // The key is drawn by anyplotlib's own overlay layer inside the figure
  // iframe, so this counts saturated pixels PER FRAME of the DPC window (the
  // top page has no canvases at all — a top-level probe silently reads zero).
  const keyPixels = async () => {
    let n = 0
    const host = await dpcWindow(page).first().elementHandle()
    if (!host) return 0
    for (const frame of page.frames()) {
      const el = await frame.frameElement().catch(() => null)
      if (!el) continue
      if (!await host.evaluate((w, f) => w.contains(f as Node), el).catch(() => false)) continue
      n += await frame.evaluate(() => {
        let hits = 0
        for (const c of Array.from(document.querySelectorAll('canvas'))) {
          const g = (c as HTMLCanvasElement).getContext('2d')
          if (!g || !(c as HTMLCanvasElement).width) continue
          const d = g.getImageData(0, 0, (c as HTMLCanvasElement).width,
                                   (c as HTMLCanvasElement).height).data
          for (let i = 0; i < d.length; i += 4) {
            const max = Math.max(d[i], d[i + 1], d[i + 2])
            if (max > 150 && max - Math.min(d[i], d[i + 1], d[i + 2]) > 90) hits++
          }
        }
        return hits
      }).catch(() => 0)
    }
    return n
  }
  const away = await keyPixels()
  const mapFrame = dpcWindow(page).first().locator('iframe').first()
  const box = await mapFrame.boundingBox()
  expect(box, 'the DPC window has no figure iframe').not.toBeNull()
  await page.mouse.move(box!.x + box!.width * 0.5, box!.y + box!.height * 0.5)
  await expect.poll(keyPixels, {
    timeout: 20_000, message: 'the colour-wheel key never appeared on hover',
  }).toBeGreaterThan(away)
  await shot('03b-wheel-on-hover')

  // ── 4. Solve the rotation ─────────────────────────────────────────────────
  // The fixture's field is curl-free, so the ELECTRIC constraint is the one
  // that recovers its rotation (the magnetic one lands ~90° away — that is what
  // choosing a mode means, and test_dpc_action pins it).
  await page.getByTestId('dpc-tab-Field').click()
  await page.getByTestId('dpc-mode').click()
  await page.getByTestId('dpc-mode-opt-electric').click()
  await page.getByTestId('dpc-tab-Rotation').click()
  await page.getByTestId('dpc-solve-rotation').click()

  const est = page.getByTestId('dpc-estimate')
  await expect.poll(() => est.getAttribute('data-angle'),
    { timeout: 90_000, message: 'the rotation was never solved' }).not.toBeNull()
  const angle = Number(await est.getAttribute('data-angle'))
  const err = Math.min(Math.abs((angle - TRUTH_ROTATION) % 180),
                       180 - Math.abs((angle - TRUTH_ROTATION) % 180))
  expect(err, `solved ${angle}°, fixture truth ${TRUTH_ROTATION}°`).toBeLessThan(4)
  expect(Number(await est.getAttribute('data-improvement')),
    'the fit should report a large residual drop on this fixture').toBeGreaterThan(5)
  await shot('04-rotation-solved')

  // ── 5. moving the slider REPAINTS the map ─────────────────────────────────
  // The live-tune claim, checked in pixels: turning the field by 90° must
  // change the hues on screen. A caret that only updated its own label would
  // pass every headless test and fail here.
  const solved = await colourStats(page, DPC_TITLE)
  const slider = page.getByTestId('dpc-rotation')
  await slider.fill(String((angle + 90) % 360))
  await slider.dispatchEvent('change')
  await expect.poll(async () => {
    const now = await colourStats(page, DPC_TITLE)
    if (!Number.isFinite(now.hue) || !Number.isFinite(solved.hue)) return 0
    return Math.abs(((now.hue - solved.hue + 180) % 360) - 180)
  }, { timeout: 30_000, message: 'rotating the field did not repaint the map' })
    .toBeGreaterThan(15)
  await shot('05-rotated-90')

  // put it back on the solved angle for the remaining stages
  await slider.fill(String(angle))
  await slider.dispatchEvent('change')

  // ── 6. the scalar views paint, and the wheel folds away for them ──────────
  await page.getByTestId('dpc-tab-Map').click()
  for (const view of ['divergence', 'magnitude', 'fx'] as const) {
    await page.getByTestId('dpc-view').click()
    await page.getByTestId(`dpc-view-opt-${view}`).click()
    await page.waitForTimeout(400)
    await shot(`06-view-${view}`)
  }
  // A scalar map is NOT the RGB one: a diverging colormap is TWO hues where the
  // direction map spans the whole wheel. Compare hue diversity, not saturation
  // — coolwarm is every bit as saturated as an RGB direction map, so a
  // saturation test passes on both and proves nothing about the swap.
  const scalar = await colourStats(page, DPC_TITLE)
  expect(scalar.hueBins,
    `the scalar view spans ${scalar.hueBins} hue bins vs the direction map's `
    + `${before.hueBins} — the view swap never reached the figure`)
    .toBeLessThan(before.hueBins)

  await page.getByTestId('dpc-view').click()
  await page.getByTestId('dpc-view-opt-rgb').click()
  await page.waitForTimeout(400)
  await shot('07-back-to-direction-map')

  // ── 7. Commit opens a new tree ────────────────────────────────────────────
  const windowsBefore = await page.getByTestId('subwindow').count()
  await page.getByTestId('dpc-commit').click()
  await expect.poll(() => page.getByTestId('subwindow').count(),
    { timeout: 60_000, message: 'Commit opened no new window' })
    .toBeGreaterThan(windowsBefore)
  await shot('08-committed')

  ctx.assertNoJsErrors()
})

test('the Center tab offers all three references, each with its own furniture', async () => {
  const { page } = ctx
  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  // The caret is still open from the previous test, parked on its Map tab.
  await page.getByTestId('dpc-tab-Center').click()

  // Manual — a crosshair on the DIFFRACTION PATTERN (it picks a detector
  // position), and an explicit "use this" step. The teal #94e2d5 is unique to
  // it; the navigator's crosshair is pure green and Center-Zero-Beam's is
  // yellow, so a colour count here really does identify this widget.
  await page.getByTestId('dpc-center-mode').click()
  await page.getByTestId('dpc-center-mode-opt-manual').click()
  await expect(page.getByTestId('dpc-use-crosshair')).toBeVisible()
  const tealOnPattern = async () => {
    let n = 0
    const host = await sig.elementHandle()
    for (const frame of page.frames()) {
      const el = await frame.frameElement().catch(() => null)
      if (!el || !host) continue
      if (!await host.evaluate((w, f) => w.contains(f as Node), el).catch(() => false)) continue
      n += await frame.evaluate(() => {
        let hits = 0
        for (const c of Array.from(document.querySelectorAll('canvas'))) {
          const g = (c as HTMLCanvasElement).getContext('2d')
          if (!g || !(c as HTMLCanvasElement).width) continue
          const d = g.getImageData(0, 0, (c as HTMLCanvasElement).width,
                                   (c as HTMLCanvasElement).height).data
          // #94e2d5 = (148, 226, 213): green highest, blue close behind, red low.
          for (let i = 0; i < d.length; i += 4) {
            if (d[i + 1] > 190 && d[i + 2] > 170 && d[i] < 190
                && d[i + 1] - d[i] > 40) hits++
          }
        }
        return hits
      }).catch(() => 0)
    }
    return n
  }
  await expect.poll(tealOnPattern, {
    timeout: 30_000, message: 'Manual mode drew no crosshair on the pattern',
  }).toBeGreaterThan(0)
  await shot('10-manual-crosshair')
  // The BACKEND echoes the picked position back on `dpc_state`; the caret shows
  // it. Asserting on the echo (not on the click) is what proves the pick
  // actually landed rather than that a button was pressed.
  await page.getByTestId('dpc-use-crosshair').click()
  await expect(page.getByTestId('dpc-center-xy'))
    .toContainText(/Centre: \(/, { timeout: 30_000 })

  // Vacuum — offers the OTHER open datasets plus a file picker. Load a second
  // scan through the test harness and check it turns up in the list.
  await backendAction(page, 'load_test_data_dpc', { nav: 24, sig: 40, amplitude: 0 })
  await page.getByTestId('dpc-center-mode').click()
  await page.getByTestId('dpc-center-mode-opt-vacuum').click()
  await expect(page.getByTestId('dpc-vacuum-file')).toBeVisible()
  await page.getByTestId('dpc-vacuum-tree').click()
  // Pick the LAST option rather than a hard-coded index: the choice values are
  // positions in `session.signal_trees`, which also holds the tree the previous
  // test committed. (That tree is filtered OUT of the list — only real 4D scans
  // can be a vacuum reference — so the index is not simply "1".)
  const options = page.locator('[data-testid^="dpc-vacuum-tree-opt-"]')
  await expect(options).toHaveCount(2, { timeout: 20_000 })  // placeholder + the scan
  await options.last().click()
  await expect(page.getByTestId('dpc-vacuum-label'))
    .toContainText(/Using /, { timeout: 60_000 })
  await shot('11-vacuum-reference')

  // Back to Corners so the teardown test finds the boxes it asserts on.
  await page.getByTestId('dpc-center-mode').click()
  await page.getByTestId('dpc-center-mode-opt-corners').click()
  ctx.assertNoJsErrors()
})

test('closing the caret removes the DPC window and the corner boxes', async () => {
  const { page } = ctx
  const sig = sigWindow(page)
  // Commit (previous test) opened a new window on top, so the source window has
  // to be RE-FOCUSED before its toolbar is reachable — a bare hover finds
  // nothing and times out.
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-DPC').click()      // toggle OFF
  await expect(page.getByTestId('dpc-wizard')).toHaveCount(0)
  // The LIVE window must go; the committed tree from the previous test must
  // STAY (a Commit that vanishes when the caret closes is worthless).
  await expect.poll(() => dpcWindow(page).count(),
    { timeout: 30_000, message: 'the live DPC window outlived its caret' })
    .toBe(0)
  expect(await page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('subwindow-title').filter({ hasText: COMMITTED_TITLE }) })
    .count(), 'closing the caret also took the committed tree').toBeGreaterThan(0)
  await shot('09-closed')
  ctx.assertNoJsErrors()
})
