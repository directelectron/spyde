/**
 * report_line_panel.spec.ts — Report Builder 1-D LINE panels, end-to-end.
 *
 * The just-shipped feature (unit-tested in test_report_line_panels.py; this spec
 * confirms it in the REAL Electron app):
 *   • Dropping a 1-D signal window into a report creates a kind="line" panel
 *     (curve + calibrated x axis + styling) — NOT an image panel.
 *   • The slim bar's LayerEdit row for a line panel shows a color swatch + preset
 *     dots + a width NumBox + a label input, each sending repfig_set_layer
 *     {color|linewidth|label}.
 *   • Setting a label renders a legend; double-clicking the legend region emits a
 *     double_click with target 'legend' → the TextSizePopover, whose size
 *     persists into the panel's text_sizes.legend.
 *
 * Data: a NEW test-only backend loader `load_test_data_line` (512-pt calibrated
 * synthetic spectrum, two Gaussian peaks) — no existing loader yields a 1-D
 * signal window. It builds ONE signal window (no navigator), so no dask is
 * needed → SPYDE_NO_DASK fast launch (matches report_text_ui's config).
 *
 * Interaction paths (proven shapes from report_edit2 / report_slimbar):
 *   - figcell chrome only mounts on a bubbling `mouseover` dispatch (the OOPIF
 *     iframe eats real hover) — never .hover().
 *   - report_state (window._spyde_test_report) is authoritative — poll it.
 *   - LINE color/width/label controls are REAL DOM interactions on the slim bar.
 *   - The legend double-click uses the CUSTOM-EVENT FALLBACK: a 1-D panel is a
 *     SINGLE overlay canvas whose legend is hit-tested by pixel geometry inside
 *     the OOPIF (no separate legend DOM element to dblclick reliably), so this
 *     spec dispatches the `spyde:figure_event` CustomEvent with a hand-built
 *     double_click {target:'legend'} payload (the exact shape SpyDEContext mirrors
 *     from an awi_event; see ReportFigureCell's onFigEvent). Called out inline.
 *
 * Screenshots to report_styling_shots/ — each Read by the author (a blank line
 * panel is a failure even when selectors pass).
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
  ctx = await launchApp({ env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1200)
  await backendAction(page, 'load_test_data_line')
  await waitForSubwindowCount(page, 1, 60_000)   // one signal window, no navigator
  await page.waitForTimeout(2500)                // let the 1-D curve paint
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

// ── shared helpers (proven shapes from report_edit2 / report_slimbar) ───────────

/** Native HTML5 drag src→report body, one shared DataTransfer. */
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

/** The single figure cell's id (exact regex — the prefix match snags inner testids). */
async function figCellId(page: any): Promise<string> {
  const figCell = page.locator('[data-testid^="report-figcell-c"]').first()
  return await figCell.evaluate((el: HTMLElement) =>
    (el.getAttribute('data-testid') || '').replace('report-figcell-', ''))
}

/** The anyplotlib figId of the (single) report figure iframe. */
async function reportFigId(page: any): Promise<string | null> {
  return await page.evaluate(() => {
    const cell = document.querySelector('[data-testid^="report-figcell-c"]')
    const ifr = cell?.querySelector('iframe[data-testid^="figure-"]') as HTMLIFrameElement | null
    if (!ifr) return null
    return (ifr.getAttribute('data-testid') || '').replace('figure-', '')
  })
}

/** The report doc's figure cell (panels + layers) via the test hook. */
async function docCell(page: any, cellId: string): Promise<any> {
  return await page.evaluate((cid: string) => {
    const d = (window as any)._spyde_test_report?.()
    return d?.cells?.find((c: any) => c.id === cid) ?? null
  }, cellId)
}

/** Report-scoped backend-error assertion. */
async function assertNoBackendErrors(tag: string) {
  const errs = backendErrorLines(ctx.backend)
    .filter((l: string) => /report|repfig|annotation|panel|figure|layer|line/i.test(l))
  if (errs.length) console.log(`[${tag}] backend error lines:\n` + errs.join('\n'))
  expect(errs, 'report-related Python tracebacks/errors in backend log').toEqual([])
}

/**
 * Count pixels of a colour class inside the report figure's OWN iframe canvases.
 * A 1-D curve is drawn on the overlay/plot canvas, so a recoloured line is
 * pixel-visible here. kind:
 *   'blue'  — the anyplotlib Plot1D default line #4fc3f7 (79,195,247)
 *   'red'   — a recolour to the #f38ba8 preset (243,139,168): pinkish-red
 *   'any'   — any non-background pixel (blank/black frame = failure)
 */
async function figurePixels(page: any, figId: string, kind: 'blue' | 'red' | 'any'): Promise<number> {
  const el = await page.$(`iframe[data-testid="figure-${figId}"]`)
  const frame = el ? await el.contentFrame() : null
  if (!frame) return -1
  return await frame.evaluate((k: string) => {
    let n = 0
    for (const c of Array.from(document.querySelectorAll('canvas'))) {
      const g = (c as HTMLCanvasElement).getContext('2d')
      if (!g || !(c as HTMLCanvasElement).width || !(c as HTMLCanvasElement).height) continue
      let d: Uint8ClampedArray
      try { d = g.getImageData(0, 0, (c as HTMLCanvasElement).width, (c as HTMLCanvasElement).height).data }
      catch { continue }
      for (let p = 0; p < d.length; p += 4) {
        const r = d[p], gg = d[p + 1], b = d[p + 2], a = d[p + 3]
        if (a < 16) continue
        // #4fc3f7 ≈ (79,195,247): blue high, green mid-high, red low.
        if (k === 'blue' && b > 180 && gg > 140 && r < 140 && (b - r) > 60) n++
        // #f38ba8 ≈ (243,139,168): red high, green mid, blue mid; red clearly
        // dominant and NOT the blue default (b < r).
        if (k === 'red' && r > 190 && gg > 90 && gg < 190 && b < 210 && (r - gg) > 40 && (r - b) > 25) n++
        // Any non-near-background pixel (bg is ~#1e1e2e = (30,30,46)).
        if (k === 'any' && (r > 55 || gg > 55 || b > 70)) n++
      }
    }
    return n
  }, kind)
}

// Shared across the serial tests.
let cellId = ''
let panelId = ''
let baseLayerId = ''

// ── 1: drop a 1-D window → a kind="line" panel renders a curve ─────────────────

test('1) dropping a 1-D signal window creates a line panel with a calibrated curve', async () => {
  const { page } = ctx
  // Open the report sidebar + a fresh report.
  if (!(await page.getByTestId('report-sidebar').count())) {
    await page.getByTestId('toggle-report').click()
  }
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await backendAction(page, 'report_new')
  await expect(page.getByTestId('report-body')).toBeVisible({ timeout: 10_000 })
  await expect(page.locator('[data-testid^="report-figcell-c"]')).toHaveCount(0)

  // Drag the (only) 1-D signal window's breadcrumb pill into the report body.
  const sig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
    .first()
  await expect(sig).toBeVisible({ timeout: 10_000 })
  await sig.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-line-sig', '1'))
  await dragToBody(page, '[data-line-sig="1"]')
  await sig.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.removeAttribute('data-line-sig'))

  const figCell = page.locator('[data-testid^="report-figcell-c"]').first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(2500)
  cellId = await figCellId(page)

  // report_state: the panel is kind="line", a single base layer, calibrated axes.
  const cell = await docCell(page, cellId)
  const panel = cell?.figure?.panels?.[0]
  expect(panel, 'no panel in the report doc').toBeTruthy()
  expect(panel.kind, `panel kind is ${panel.kind}, expected "line"`).toBe('line')
  panelId = panel.id
  baseLayerId = panel.layers?.[0]?.id
  expect(baseLayerId, 'no base layer id on the line panel').toBeTruthy()
  // The calibrated x axis (eV) rode along in the panel's axes dict.
  expect(panel.axes?.x_axis?.length, 'line panel carries no calibrated x_axis').toBe(512)
  expect(String(panel.axes?.units || '')).toContain('eV')

  // The curve actually painted (blue default line pixels present, not a blank frame).
  const figId = (await reportFigId(page))!
  await expect.poll(async () => await figurePixels(page, figId, 'any'), {
    timeout: 10_000, message: 'line panel rendered a blank/black frame',
  }).toBeGreaterThan(500)
  const bluePix = await figurePixels(page, figId, 'blue')
  console.log('[line] default #4fc3f7 line pixels =', bluePix, 'figId=', figId)
  expect(bluePix, 'no default-blue line pixels — curve did not paint').toBeGreaterThan(30)

  // READ THIS SHOT: a blue sine-plus-two-peaks curve on a calibrated eV x-axis.
  await page.screenshot({ path: join(SHOTS, '20-line-panel-curve.png') })
  ctx.assertNoJsErrors()
  await assertNoBackendErrors('line-1')
})

// ── 2: edit mode → line LayerEdit sets color (preset) + width; curve recolors ──

test('2) line LayerEdit: preset color + width 4 persist + the curve recolors', async () => {
  const { page } = ctx
  const figCell = page.locator(`[data-testid="report-figcell-${cellId}"]`)
  await figCell.dispatchEvent('mouseover', { bubbles: true })
  const toggle = page.getByTestId(`report-figcell-edit-toggle-${cellId}`)
  await expect(toggle).toBeVisible()
  await toggle.click()
  await expect(page.getByTestId(`figcell-edit-${cellId}`)).toBeVisible({ timeout: 10_000 })
  await page.waitForTimeout(1500)   // edit rebuild paints

  // The LINE LayerEdit row shows the line-styling controls (NOT the image
  // cmap/tint controls). Assert the line-specific controls exist.
  const colorSwatch = page.getByTestId(`figcell-layer-color-${baseLayerId}`)
  const widthBox = page.getByTestId(`figcell-layer-width-${baseLayerId}`)
  const labelInput = page.getByTestId(`figcell-layer-label-${baseLayerId}`)
  await expect(colorSwatch).toBeVisible({ timeout: 10_000 })
  await expect(widthBox).toBeVisible()
  await expect(labelInput).toBeVisible()
  // No image-only cmap control on a line layer.
  await expect(page.getByTestId(`figcell-layer-cmap-${baseLayerId}`)).toHaveCount(0)
  await page.screenshot({ path: join(SHOTS, '21-line-layeredit-row.png') })

  const layerColor = async (): Promise<string | null> => {
    const cell = await docCell(page, cellId)
    const l = cell?.figure?.panels?.[0]?.layers?.[0]
    return l ? (l.color != null ? String(l.color).toLowerCase() : null) : null
  }
  const layerWidth = async (): Promise<number | null> => {
    const cell = await docCell(page, cellId)
    const l = cell?.figure?.panels?.[0]?.layers?.[0]
    return l && l.linewidth != null ? Number(l.linewidth) : null
  }

  const figIdBefore = (await reportFigId(page))!

  // Click the RED preset dot (#f38ba8) → repfig_set_layer {color}. The line
  // panel presets use the shared PRESET_COLORS; #f38ba8 is the warm-red dot.
  await page.getByTestId(`figcell-layer-color-${baseLayerId}-preset-f38ba8`).click()
  await expect.poll(layerColor, {
    timeout: 10_000, message: 'preset dot did not persist layer.color=#f38ba8',
  }).toBe('#f38ba8')

  // Width 4 via the NumBox (type + Enter — the NumBox commits on Enter/blur).
  await widthBox.click()
  await widthBox.fill('4')
  await widthBox.press('Enter')
  await expect.poll(layerWidth, {
    timeout: 10_000, message: 'width NumBox did not persist linewidth=4',
  }).toBeCloseTo(4, 1)

  // A layer-styling change rebuilds the figure — wait for the new iframe to
  // promote, then assert the curve pixels turned RED (and the blue default is
  // gone). Scoped to the report figure's own iframe.
  await page.waitForTimeout(2500)
  const figIdAfter = (await reportFigId(page))!
  console.log('[line] figId before/after recolor =', figIdBefore, figIdAfter)
  await expect.poll(async () => await figurePixels(page, figIdAfter, 'red'), {
    timeout: 12_000, message: 'curve did not recolor to a warm-red on screen',
  }).toBeGreaterThan(30)
  const bluePixAfter = await figurePixels(page, figIdAfter, 'blue')
  console.log('[line] blue pixels after recolor =', bluePixAfter, '(should be ~0)')

  // READ THIS SHOT: a THICKER, warm-RED curve (was thin blue).
  await page.screenshot({ path: join(SHOTS, '22-line-recolored-thick.png') })
  ctx.assertNoJsErrors()
  await assertNoBackendErrors('line-2')
})

// ── 3: set a label → legend renders; dbl-click the legend → text-size popover ──

test('3) label → legend renders; legend double_click → text-size popover persists', async () => {
  const { page } = ctx

  const layerLabel = async (): Promise<string | null> => {
    const cell = await docCell(page, cellId)
    const l = cell?.figure?.panels?.[0]?.layers?.[0]
    return l && l.label != null ? String(l.label) : null
  }

  // Type a label into the line LayerEdit's label input + commit (Enter → blur).
  const labelInput = page.getByTestId(`figcell-layer-label-${baseLayerId}`)
  await expect(labelInput).toBeVisible({ timeout: 10_000 })
  await labelInput.click()
  await labelInput.fill('spectrum')
  await labelInput.press('Enter')
  await expect.poll(layerLabel, {
    timeout: 10_000, message: 'label input did not persist layer.label="spectrum"',
  }).toBe('spectrum')

  // The label set → the figure rebuilds with a legend. Wait for the new iframe
  // + a settle, then screenshot for the legend (human-eyes: a "spectrum" legend
  // box appears on the curve).
  await page.waitForTimeout(2500)
  const figId = (await reportFigId(page))!
  await expect.poll(async () => await figurePixels(page, figId, 'any'), {
    timeout: 10_000, message: 'legend rebuild rendered a blank frame',
  }).toBeGreaterThan(500)
  await page.screenshot({ path: join(SHOTS, '23-line-legend.png') })

  // ── legend double-click → TextSizePopover ────────────────────────────────
  // FALLBACK PATH (called out per the harness memo): a 1-D panel is a SINGLE
  // overlay canvas whose legend is hit-tested by pixel geometry inside the OOPIF
  // — there is no discrete legend DOM node to dblclick reliably. Dispatch the
  // `spyde:figure_event` CustomEvent with the exact double_click payload
  // anyplotlib emits for a legend hit (see figure_esm.js line ~7012 and
  // ReportFigureCell.onFigEvent's double_click branch). Renderer-only: it does
  // NOT touch the backend (no awi_event postMessage), it only opens the popover.
  // The real anyplotlib double_click carries the panel's DISPATCH id in panel_id.
  // repfig_set_text_size resolves EITHER a dispatch id OR a spec panel id (see
  // its docstring), so we inject the spec panelId here — the backend resolves it
  // the same way, and it's the id we can name from the report doc. (Injecting
  // panel_id:null instead reproduces "panel not found" — an unrealistic event.)
  const dispatchLegendDblClick = async () => {
    const fid = (await reportFigId(page))!
    await page.evaluate(({ f, pid }: { f: string; pid: string }) => {
      window.dispatchEvent(new CustomEvent('spyde:figure_event', {
        detail: {
          figId: f,
          event: {
            event_type: 'double_click', target: 'legend',
            panel_id: pid, x: 0.8, y: 0.15, xdata: 0, ydata: 0,
          },
        },
      }))
    }, { f: fid, pid: panelId })
  }

  const popover = page.getByTestId('figcell-text-size-legend')
  for (let attempt = 0; attempt < 3; attempt++) {
    await dispatchLegendDblClick()
    try { await expect(popover).toBeVisible({ timeout: 3_000 }); break }
    catch { /* stale figId during a swap — re-read + retry */ }
  }
  await expect(popover).toBeVisible({ timeout: 4_000 })
  await expect(page.getByTestId('figcell-text-size-input')).toBeVisible()
  await page.screenshot({ path: join(SHOTS, '24-legend-text-size-popover.png') })

  // Set 14 via the preset dot → repfig_set_text_size {target:'legend', size:14}
  // → persisted into the panel's text_sizes.legend.
  await page.getByTestId('figcell-text-size-preset-14').click()
  await expect.poll(async () => {
    const cell = await docCell(page, cellId)
    const ts = cell?.figure?.panels?.[0]?.text_sizes
    return ts && ts.legend != null ? Number(ts.legend) : null
  }, { timeout: 10_000, message: 'text_sizes.legend did not persist to 14' }).toBe(14)

  await page.waitForTimeout(1500)
  await page.screenshot({ path: join(SHOTS, '25-line-legend-sized.png') })
  ctx.assertNoJsErrors()
  await assertNoBackendErrors('line-3')
})

test('4) final: no line/report-related Python tracebacks in the backend log', async () => {
  await assertNoBackendErrors('line-final')
})
