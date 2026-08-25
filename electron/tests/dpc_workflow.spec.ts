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

/**
 * The SOURCE diffraction-pattern window — the fixture's own signal window.
 *
 * NOT the harness's `sigWindow`, which takes the first window whose breadcrumb
 * starts "S-". That is unambiguous only until DPC opens windows of its own:
 * "S-DPC Field Map" and the committed "S-DPC (E)" also match, and `.first()`
 * then silently returns whichever sorts first. It bit a pixel probe here — the
 * map window's colour wheel is teal, the exact colour the beam-region check
 * counts.
 */
function sourceWindow(page: import('@playwright/test').Page) {
  return page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('subwindow-title').filter({ hasText: /S-Synthetic DPC$/ }) })
    .first()
}

/** Count a widget's colour inside ONE window's figure frames. */
async function colorPixelsIn(page: import('@playwright/test').Page,
                             host: import('@playwright/test').Locator,
                             match: (r: number, g: number, b: number) => boolean) {
  const handle = await host.elementHandle()
  if (!handle) return 0
  let n = 0
  for (const frame of page.frames()) {
    const el = await frame.frameElement().catch(() => null)
    if (!el) continue
    if (!await handle.evaluate((w, f) => w.contains(f as Node), el).catch(() => false)) continue
    n += await frame.evaluate((src) => {
      // eslint-disable-next-line no-new-func
      const test = new Function('r', 'g', 'b', `return (${src})(r,g,b)`)
      let hits = 0
      for (const c of Array.from(document.querySelectorAll('canvas'))) {
        const g2 = (c as HTMLCanvasElement).getContext('2d')
        if (!g2 || !(c as HTMLCanvasElement).width) continue
        const d = g2.getImageData(0, 0, (c as HTMLCanvasElement).width,
                                  (c as HTMLCanvasElement).height).data
        for (let i = 0; i < d.length; i += 4) {
          if (test(d[i], d[i + 1], d[i + 2])) hits++
        }
      }
      return hits
    }, match.toString()).catch(() => 0)
  }
  return n
}

/**
 * #ff3030 — the four corner boxes. Red, and deliberately not the teal beam
 * region or Center Zero Beam's yellow: `r - g > 60` is what separates it from
 * that yellow (#f9e2af sits at r - g = 23), so keep that clause if you retune.
 */
const IS_CORNER = (r: number, g: number, b: number) =>
  r > 200 && g < 190 && r - g > 60 && r - b > 40
/** #94e2d5 — the beam region (circle / ring). */
const IS_BEAM = (r: number, g: number, b: number) =>
  g > 190 && b > 170 && r < 190 && g - r > 40

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
  // mean anything on. #ff3030 is unique to them here, and counting it on the
  // navigator specifically is what proves they did not land on the pattern.
  await expect(page.getByTestId('dpc-center-mode'))
    .toHaveAttribute('data-value', 'corners')
  const cornerPixels = () => colorPixelsIn(page, nav, IS_CORNER)
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

  // ── 3b. the direction legend is up WITHOUT hovering ───────────────────────
  // It is an anyplotlib KEY (`Plot2D.add_key`) — the same overlay primitive as
  // the IPF colour triangle and the scale bar, not a floating inset panel — and
  // it is shown always, because a direction map's hues are meaningless without
  // it. Counting its saturated pixels with the pointer AWAY is what proves it
  // is not hover-gated.
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
  // Park the pointer well away from the map first, so nothing can be attributed
  // to hover.
  await page.mouse.move(5, 5)
  await expect.poll(keyPixels, {
    timeout: 20_000,
    message: 'the colour-wheel key is not visible without hovering the map',
  }).toBeGreaterThan(0)
  await shot('03b-wheel-always-on')

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
  const sig = sourceWindow(page)
  await sig.getByTestId('subwindow-title').click()
  // The caret is still open from the previous test, parked on its Map tab.
  await page.getByTestId('dpc-tab-Center').click()

  // The beam region FIRST — Manual below depends on it being on, since the
  // region's centre is what Manual adopts.
  //
  // One draggable shape that BOTH masks the centre of mass and marks the zero
  // beam. Toggling Circle→Ring swaps the anyplotlib widget type, so this
  // checks the shape actually changed on the PATTERN, not just in the caret.
  const beamPixels = () => colorPixelsIn(page, sourceWindow(page), IS_BEAM)
  expect(await beamPixels(), 'the region should start off').toBe(0)

  await page.getByTestId('dpc-beam-circle').click()
  await expect(page.getByTestId('dpc-beam-r')).toBeVisible()
  await expect.poll(beamPixels, {
    timeout: 30_000, message: 'the beam circle never appeared on the pattern',
  }).toBeGreaterThan(0)
  await expect(page.getByTestId('dpc-beam-readout'))
    .toHaveAttribute('data-brightness', /\d/, { timeout: 30_000 })
  await shot('10-beam-circle')

  // Ring — an inner radius appears, and the widget becomes an annulus.
  await page.getByTestId('dpc-beam-ring').click()
  await expect(page.getByTestId('dpc-beam-r-inner')).toBeVisible()
  await expect.poll(beamPixels, {
    timeout: 30_000, message: 'the ring never replaced the circle',
  }).toBeGreaterThan(0)
  await shot('11-beam-ring')

  // The ⓘ affordance: the explanation is available but costs one glyph until
  // it is asked for. (The prose used to sit inline under every control.)
  await expect(page.getByTestId('dpc-info-beam-text')).toHaveCount(0)
  await page.getByTestId('dpc-info-beam').click()
  await expect(page.getByTestId('dpc-info-beam-text')).toBeVisible()
  await shot('12-info-expanded')
  await page.getByTestId('dpc-info-beam').click()
  await expect(page.getByTestId('dpc-info-beam-text')).toHaveCount(0)

  await page.getByTestId('dpc-beam-off').click()
  await expect.poll(beamPixels, {
    timeout: 30_000, message: 'turning the region off left its widget behind',
  }).toBe(0)

  // Manual — the beam region IS the marker, so this is one click, not a
  // separate crosshair to place. Turn the region back on so there is a centre
  // to adopt.
  await page.getByTestId('dpc-beam-circle').click()
  await expect.poll(beamPixels, { timeout: 30_000 }).toBeGreaterThan(0)
  await page.getByTestId('dpc-center-mode').click()
  await page.getByTestId('dpc-center-mode-opt-manual').click()
  await expect(page.getByTestId('dpc-use-crosshair')).toBeVisible()
  // The BACKEND echoes the adopted position back on `dpc_state`; the caret
  // shows it. Asserting on the echo (not on the click) is what proves the pick
  // actually landed rather than that a button was pressed.
  await page.getByTestId('dpc-use-crosshair').click()
  await expect(page.getByTestId('dpc-center-xy'))
    .toContainText(/Centre: \(/, { timeout: 30_000 })
  await shot('13-manual-from-region')

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

test('a lazy scan streams the beam-shift pass through the real cluster', async () => {
  const { page } = ctx
  // The threaded ComputeBackend and the DISTRIBUTED one take different branches
  // of `compute_chunks_progressive`; the Python suite covers the threaded one,
  // so this is the branch only the real app reaches. Storage-aligned chunks
  // (whole signal frames per chunk) so a "chunk" here is a real storage chunk.
  // Wait for the NEW windows specifically. A bare `count() > 2` is already true
  // — several tests' windows are still open — so it returns instantly and every
  // locator below then resolves against the OLD dataset.
  const before = await page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('subwindow-title').filter({ hasText: /S-Synthetic DPC$/ }) })
    .count()
  await backendAction(page, 'load_test_data_dpc',
                      { nav: 24, sig: 40, lazy: true, nav_chunk: 8 })
  const lazySigAll = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('subwindow-title').filter({ hasText: /S-Synthetic DPC$/ }) })
  await expect.poll(() => lazySigAll.count(),
    { timeout: 180_000, message: 'the lazy dataset never opened a window' })
    .toBe(before + 1)
  const lazySig = lazySigAll.last()
  await lazySig.getByTestId('subwindow-title').click()
  await lazySig.getByTestId('subwindow-titlebar').hover()
  await lazySig.getByTestId('action-btn-DPC').click()
  // Carets are parented to their own window, so with the earlier dataset's
  // caret still open a page-wide `getByTestId('dpc-wizard')` matches two.
  // Scope to this window.
  await expect(lazySig.getByTestId('dpc-wizard')).toBeVisible()

  // It has to actually finish and produce a field — a stream that dispatches
  // but never assembles would leave the descan readout empty forever.
  await expect.poll(
    () => lazySig.getByTestId('dpc-centering').getAttribute('data-worst'),
    { timeout: 180_000, message: 'the streamed pass never produced a field' })
    .toMatch(/\d/)
  await expect.poll(() => colourStats(page, DPC_TITLE).then(s => s.saturated),
    { timeout: 60_000, message: 'the streamed pass painted no map' })
    .toBeGreaterThan(0.02)
  await shot('14-lazy-streamed')

  // No NaN holes left behind: every scan position must have been written.
  const stats = await colourStats(page, DPC_TITLE)
  expect(stats.hueBins, 'the streamed field looks incomplete')
    .toBeGreaterThan(4)

  // Clean up after ourselves. The teardown test below asserts that closing a
  // caret leaves NO live DPC window, which only holds if this one's is gone.
  // Re-focus first: the result window opened on top, and a toolbar is only
  // reachable on the focused window.
  await lazySig.getByTestId('subwindow-title').click()
  await lazySig.getByTestId('subwindow-titlebar').hover()
  await lazySig.getByTestId('action-btn-DPC').click()
  await expect(lazySig.getByTestId('dpc-wizard')).toHaveCount(0)
  ctx.assertNoJsErrors()
})

test('closing the caret removes the DPC window and the corner boxes', async () => {
  const { page } = ctx
  const sig = sourceWindow(page)
  // Commit (previous test) opened a new window on top, so the source window has
  // to be RE-FOCUSED before its toolbar is reachable — a bare hover finds
  // nothing and times out.
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-DPC').click()      // toggle OFF
  // Scoped to THIS window: the lazy-streaming test above leaves its own caret
  // open on a different window, and carets are per-window.
  await expect(sig.getByTestId('dpc-wizard')).toHaveCount(0)
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
