/**
 * report_text_styling.spec.ts — Report Builder double-click TEXT SIZING, e2e.
 *
 * The just-shipped feature (unit-tested; this spec confirms it in the REAL app):
 *   • Double-clicking an axis tick/label zone / title / colorbar / a 1-D legend
 *     inside a report figure emits a `double_click` carrying a `target`
 *     ('x_ticks' / 'x_label' / 'y_ticks' / 'y_label' / 'title' /
 *     'colorbar_label' / 'legend'); ReportFigureCell opens a TextSizePopover
 *     (a NumBox 6..96 + preset dots) that sends
 *     `repfig_set_text_size {cell_id, panel_id, target, size}`.
 *   • The backend persists PanelSpec.text_sizes (ticks share one 'ticks' key)
 *     and live-pushes the size in place (no rebuild). Works OUTSIDE edit mode.
 *   • A plain plot-area double_click (no `target`) is IGNORED — the popover
 *     stays closed.
 *
 * Data: bundled synthetic Si-grains 4-D STEM (`load_test_data_si_grains`, real
 * Dask — same fixture report_callouts/report_annotations use). Drop the SIGNAL
 * window → a 2-D image panel with a CALIBRATED axis gutter (si_grains ships
 * signal-axis units "$\AA^{-1}$" + scale 0.01, so `Plot._axes_info` returns
 * real axes and anyplotlib draws tick labels). `load_test_data`'s synthetic
 * disk fixture is intentionally left uncalibrated (scale=1, units="") to catch
 * a different bug class (see its docstring / the lazy fixture's calibration
 * comment) — anyplotlib._axes_info treats units in ("<undefined>","px","") as
 * "uncalibrated" and returns axes=None, so NO ticks are ever drawn on that
 * panel and the original version of this test's gutter-pixel assertion could
 * never pass (0 tick pixels before AND after — not a feature bug, a fixture
 * choice). Switching to a calibrated dataset makes the visual assertion real.
 *
 * Interaction paths:
 *   - figcell chrome mounts on a bubbling `mouseover` (the OOPIF eats real hover).
 *   - report_state (window._spyde_test_report) is authoritative — poll it.
 *   - The x-axis-gutter double_click uses the CUSTOM-EVENT FALLBACK: the figure
 *     iframe is an OOPIF whose gutter canvas grabs pointer capture, so a synthetic
 *     dblclick can't reliably land on the (display:none-until-shown, class-less)
 *     xAxisCanvas. This spec dispatches the `spyde:figure_event` CustomEvent with
 *     the exact double_click {target:'x_ticks'} payload anyplotlib's xAxisCanvas
 *     handler emits (figure_esm.js ~6696; SpyDEContext mirrors every awi_event as
 *     spyde:figure_event; ReportFigureCell.onFigEvent opens the popover off it).
 *     Renderer-only — it does NOT hit the backend until the popover's control is
 *     used. Called out inline; the plot-CENTER negative test uses the SAME path
 *     with NO target (proving that branch is ignored).
 *
 * Screenshots to report_styling_shots/ — each Read by the author.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_styling_shots')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(180_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')   // calibrated 4-D STEM
  await waitForSubwindowCount(page, 2, 120_000)  // navigator + signal
  await page.waitForTimeout(2500)               // let the DP paint
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

// ── shared helpers (proven shapes from report_slimbar / report_edit2) ───────────

async function dragToBody(page: any, srcSel: string) {
  await page.evaluate(({ srcSel }: any) => {
    const src = document.querySelector(srcSel) as HTMLElement
    const dst = document.querySelector('[data-testid="report-body"]') as HTMLElement
    if (!src || !dst) throw new Error('drag src/report-body not found')
    const dt = new DataTransfer()
    const fire = (target: HTMLElement, type: string) => {
      const r = target.getBoundingClientRect()
      const ev = new DragEvent(type, {
        bubbles: true, cancelable: true,
        clientX: r.left + r.width / 2, clientY: r.top + r.height / 2,
      })
      Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
      target.dispatchEvent(ev)
    }
    fire(src, 'dragstart')
    fire(dst, 'dragenter'); fire(dst, 'dragover'); fire(dst, 'drop'); fire(src, 'dragend')
  }, { srcSel })
}

async function figCellId(page: any): Promise<string> {
  const figCell = page.locator('[data-testid^="report-figcell-c"]').first()
  return await figCell.evaluate((el: HTMLElement) =>
    (el.getAttribute('data-testid') || '').replace('report-figcell-', ''))
}

async function reportFigId(page: any): Promise<string | null> {
  return await page.evaluate(() => {
    const cell = document.querySelector('[data-testid^="report-figcell-c"]')
    const ifr = cell?.querySelector('iframe[data-testid^="figure-"]') as HTMLIFrameElement | null
    if (!ifr) return null
    return (ifr.getAttribute('data-testid') || '').replace('figure-', '')
  })
}

async function docCell(page: any, cellId: string): Promise<any> {
  return await page.evaluate((cid: string) => {
    const d = (window as any)._spyde_test_report?.()
    return d?.cells?.find((c: any) => c.id === cid) ?? null
  }, cellId)
}

/** The persisted text_sizes dict of the cell's first panel (or {} / null). */
async function panelTextSizes(page: any, cellId: string): Promise<Record<string, number> | null> {
  const cell = await docCell(page, cellId)
  const ts = cell?.figure?.panels?.[0]?.text_sizes
  return ts ?? null
}

/**
 * Count pixels inside the report figure iframe whose colour matches the axis
 * gutter TEXT (light-grey tick labels on the dark #1e1e2e gutter), restricted to
 * the BOTTOM strip (the x-axis gutter) of the iframe. Larger ticks → more grey
 * text pixels in that strip. A crude but robust before/after signal for "the
 * tick font grew". Uses a full-page screenshot (captures WebGPU or Canvas2D).
 */
async function xGutterTextPixels(page: any, figId: string): Promise<number> {
  const iframe = page.locator(`iframe[data-testid="figure-${figId}"]`)
  if (!(await iframe.count())) return -1
  const box = await iframe.boundingBox()
  if (!box) return -1
  let buf: Buffer
  try { buf = await page.screenshot() } catch { return -1 }
  return await page.evaluate(async ({ b64, box }: { b64: string; box: any }) => {
    const img = await new Promise<HTMLImageElement>((res, rej) => {
      const i = new Image(); i.onload = () => res(i); i.onerror = rej
      i.src = 'data:image/png;base64,' + b64
    })
    const cv = document.createElement('canvas')
    cv.width = img.width; cv.height = img.height
    const c2 = cv.getContext('2d')!
    c2.drawImage(img, 0, 0)
    const W = cv.width, H = cv.height
    const d = c2.getImageData(0, 0, W, H).data
    const dpr = W / window.innerWidth
    const x0 = box.x * dpr, y0 = box.y * dpr
    const bw = box.width * dpr, bh = box.height * dpr
    // Bottom ~18% strip of the iframe = the x-axis gutter (ticks + label).
    const stripTop = y0 + bh * 0.82
    let n = 0
    for (let p = 0; p < d.length; p += 4) {
      const idx = p / 4
      const px = idx % W, py = Math.floor(idx / W)
      if (px < x0 || px >= x0 + bw || py < stripTop || py >= y0 + bh) continue
      const r = d[p], g = d[p + 1], b = d[p + 2]
      // Light-grey tick text (~#9399b2 / #cdd6f4) over the dark gutter: all
      // channels bright-ish and near-neutral, clearly above the ~30 background.
      if (r > 110 && g > 110 && b > 120 && Math.abs(r - g) < 45 && Math.abs(g - b) < 50) n++
    }
    return n
  }, { b64: buf.toString('base64'), box })
}

async function assertNoBackendErrors(tag: string) {
  const errs = backendErrorLines(ctx.backend)
    .filter((l: string) => /report|repfig|panel|figure|text.?size|tick/i.test(l))
  if (errs.length) console.log(`[${tag}] backend error lines:\n` + errs.join('\n'))
  expect(errs, 'report-related Python tracebacks/errors in backend log').toEqual([])
}

/**
 * Dispatch the double_click CustomEvent anyplotlib would post for a gutter/plot
 * dblclick. `target` null → the plot-area (no-target) case the renderer ignores.
 * Uses the spec panel id for panel_id (repfig_set_text_size resolves a spec id
 * OR a dispatch id; the real event carries the dispatch id — both persist).
 */
async function dispatchDblClick(page: any, panelId: string,
                                target: string | null, fx: number, fy: number) {
  const fid = (await reportFigId(page))!
  await page.evaluate(({ f, pid, t, x, y }: any) => {
    const ev: Record<string, unknown> = {
      event_type: 'double_click', panel_id: pid, x, y, img_x: 10, img_y: 10,
    }
    if (t != null) ev.target = t
    window.dispatchEvent(new CustomEvent('spyde:figure_event', {
      detail: { figId: f, event: ev },
    }))
  }, { f: fid, pid: panelId, t: target, x: fx, y: fy })
}

let cellId = ''
let panelId = ''

// ── 1: drop 2-D data → figure cell; a plot-CENTER double_click is IGNORED ──────

test('1) figure cell mounts; a plot-center double_click does NOT open the popover', async () => {
  const { page } = ctx
  if (!(await page.getByTestId('report-sidebar').count())) {
    await page.getByTestId('toggle-report').click()
  }
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await backendAction(page, 'report_new')
  await expect(page.getByTestId('report-body')).toBeVisible({ timeout: 10_000 })
  await expect(page.locator('[data-testid^="report-figcell-c"]')).toHaveCount(0)

  const sig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
    .first()
  await expect(sig).toBeVisible({ timeout: 10_000 })
  await sig.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-ts-sig', '1'))
  await dragToBody(page, '[data-ts-sig="1"]')
  await sig.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.removeAttribute('data-ts-sig'))

  const figCell = page.locator('[data-testid^="report-figcell-c"]').first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(2500)
  cellId = await figCellId(page)
  const cell = await docCell(page, cellId)
  panelId = cell?.figure?.panels?.[0]?.id
  expect(panelId, 'no panel id in the report doc').toBeTruthy()
  await page.screenshot({ path: join(SHOTS, '10-figure-cell.png') })

  // A plot-CENTER double_click (NO target) must NOT open any text-size popover.
  await dispatchDblClick(page, panelId, null, 0.5, 0.5)
  await page.waitForTimeout(600)
  await expect(page.locator('[data-testid^="figcell-text-size-"]'),
    'plot-center double_click (no target) wrongly opened a text-size popover')
    .toHaveCount(0)
  await page.screenshot({ path: join(SHOTS, '11-center-dblclick-no-popover.png') })
  ctx.assertNoJsErrors()
  await assertNoBackendErrors('text-1')
})

// ── 2: x-axis gutter double_click → TextSizePopover → size 16 persists ─────────

test('2) x-axis double_click opens the popover; size 16 persists to text_sizes.ticks', async () => {
  const { page } = ctx

  // Baseline (no text_sizes yet) + baseline gutter-text pixel count.
  expect((await panelTextSizes(page, cellId))?.ticks ?? null,
    'ticks size already set before the test').toBeNull()
  const figIdBefore = (await reportFigId(page))!
  const gutterBefore = await xGutterTextPixels(page, figIdBefore)
  console.log('[text] x-gutter text pixels BEFORE =', gutterBefore)

  // Open the popover via an x-axis-gutter double_click (target 'x_ticks').
  // FALLBACK PATH (called out): CustomEvent injection — see the file header.
  const popover = page.getByTestId('figcell-text-size-x_ticks')
  for (let attempt = 0; attempt < 3; attempt++) {
    await dispatchDblClick(page, panelId, 'x_ticks', 0.5, 0.95)
    try { await expect(popover).toBeVisible({ timeout: 3_000 }); break }
    catch { /* stale figId during a swap — retry */ }
  }
  await expect(popover).toBeVisible({ timeout: 4_000 })
  await expect(page.getByTestId('figcell-text-size-input')).toBeVisible()
  // READ THIS SHOT: a "X ticks" popover with a size NumBox + preset dots, anchored
  // over the figure's lower edge.
  await page.screenshot({ path: join(SHOTS, '12-xticks-popover-open.png') })

  // Set size 16 via the NumBox (commits on change → repfig_set_text_size).
  const input = page.getByTestId('figcell-text-size-input')
  await input.click()
  await input.fill('16')
  await input.blur()

  // The size persists into the panel's text_sizes under the 'ticks' key (both
  // axes share it) — poll the authoritative report doc.
  await expect.poll(async () => (await panelTextSizes(page, cellId))?.ticks ?? null, {
    timeout: 10_000, message: 'text_sizes.ticks did not persist to 16',
  }).toBe(16)
  console.log('[text] text_sizes after set =', JSON.stringify(await panelTextSizes(page, cellId)))

  // The tick font grew on screen: the live in-place push (tick_size, no rebuild)
  // repaints bigger tick labels → more grey text pixels in the x gutter strip.
  await page.waitForTimeout(2000)
  const figIdAfter = (await reportFigId(page))!
  await expect.poll(async () => await xGutterTextPixels(page, figIdAfter), {
    timeout: 10_000, message: 'x-gutter tick text did not visibly grow after size 16',
  }).toBeGreaterThan(gutterBefore + 20)
  const gutterAfter = await xGutterTextPixels(page, figIdAfter)
  console.log('[text] x-gutter text pixels AFTER =', gutterAfter,
    '(Δ', gutterAfter - gutterBefore, ')')
  await page.screenshot({ path: join(SHOTS, '13-xticks-bigger.png') })
  ctx.assertNoJsErrors()
  await assertNoBackendErrors('text-2')
})

test('3) final: no text-size / report Python tracebacks in the backend log', async () => {
  await assertNoBackendErrors('text-final')
})
