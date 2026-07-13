/**
 * report_annotations.spec.ts — Report Builder edit-mode annotations, selection,
 * and figure-level annotations, end-to-end in the real app (Spec C).
 *
 * Verifies the newly-implemented edit-panel behaviours:
 *   1. Coordinate fix — "+ Text" places the annotation at the panel's IMAGE
 *      CENTER (was upper-left). Asserted on the accent-color pixels drawn in the
 *      centre band vs the corner band of the report figure iframe.
 *   2. Drag persistence — a widget pointer_up (injected via the proven
 *      awi_event postMessage path) moves an annotation; the spec/DOM row reflects
 *      the moved value; exiting edit mode re-renders the marker at the new spot.
 *   3. Selection — a panel pointer_down (injected) selects the panel → the edit
 *      dock shows the chips row + only that panel's controls; a figure-background
 *      pointer_down deselects → the figure-level section (grid summary, figure
 *      annotation palette).
 *   4. Figure-level annotation — figure-level "+ Text" draws a centered marker +
 *      adds a row; a figure-marker pointer_up drag persists the new fraction.
 *
 * Interaction paths: the figure iframe is an out-of-process OOPIF whose canvas
 * does setPointerCapture, so a synthetic Playwright mouse cannot reliably grab a
 * tiny widget handle or land a click inside it. This spec therefore INJECTS the
 * exact widget / panel / figure-marker events the anyplotlib figure would post,
 * via `window.postMessage({type:'awi_event', figId, data})` (the same path
 * selector.spec.ts uses — routed through SpyDEContext's window 'message' listener
 * → window.electron.figureEvent → the backend's Figure._dispatch_event). The
 * event JSON shapes are copied from test_report_edit_mode.py (the migrated unit
 * tests). Palette-button clicks + chips are REAL DOM clicks.
 *
 * Real Dask + bundled si_grains. Screenshots to report_edit_shots/. Each test
 * ends with assertNoJsErrors + a backend traceback scan.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_edit_shots')
const FIG_MIME = 'application/x-spyde-figure'
const ACCENT = { r: 0xff, g: 0x98, b: 0x00 }   // #ff9800 annotation accent

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(240_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)   // navigator + signal
  await page.waitForTimeout(2500)                 // let the DP paint
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

// ── shared helpers ────────────────────────────────────────────────────────────

/** Native HTML5 drag src→body, one shared DataTransfer (report_sidebar pattern). */
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

/** The single figure cell's id. */
async function figCellId(page: any): Promise<string> {
  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  return await figCell.evaluate((el: HTMLElement) =>
    (el.getAttribute('data-testid') || '').replace('report-figcell-', ''))
}

/** The anyplotlib figId of the report figure iframe for the (single) cell. */
async function reportFigId(page: any): Promise<string | null> {
  return await page.evaluate(() => {
    const cell = document.querySelector('[data-testid^="report-figcell-"]')
    const ifr = cell?.querySelector('iframe[data-testid^="figure-"]') as HTMLIFrameElement | null
    if (!ifr) return null
    return (ifr.getAttribute('data-testid') || '').replace('figure-', '')
  })
}

/** Post an awi_event JSON blob to a figure (the exact selector.spec.ts path). */
async function figureEvent(page: any, figId: string, ev: Record<string, unknown>) {
  await page.evaluate(
    ({ fid, data }: any) => window.postMessage(
      { type: 'awi_event', figId: fid, data }, '*'),
    { fid: figId, data: JSON.stringify(ev) },
  )
}

/** The report figure's overlay widgets (from the test hook), grouped by panel. */
async function reportWidgets(page: any, figId: string): Promise<Array<{
  panel_id: string; id: string; type: string; data: Record<string, unknown>
}>> {
  return await page.evaluate((fid: string) => (window as any)._spyde_test_widgets(fid), figId)
}

/**
 * Open the figure cell's edit dock (✎ toggle). Returns {cellId, figId}. Idempotent
 * on the toggle (only clicks if the dock isn't already open).
 */
async function openEdit(page: any): Promise<{ cellId: string; figId: string }> {
  const cellId = await figCellId(page)
  const figCell = page.locator(`[data-testid="report-figcell-${cellId}"]`)
  await figCell.dispatchEvent('mouseover', { bubbles: true })
  const toggle = page.getByTestId(`report-figcell-edit-toggle-${cellId}`)
  await expect(toggle).toBeVisible()
  if (!(await page.getByTestId(`figcell-edit-${cellId}`).count())) await toggle.click()
  await expect(page.getByTestId(`figcell-edit-${cellId}`)).toBeVisible()
  const figId = (await reportFigId(page))!
  expect(figId, 'report figure has no figId').toBeTruthy()
  return { cellId, figId }
}

/**
 * Classify accent-colored (#ff9800) pixels inside the report figure iframe into
 * CENTER-band vs CORNER-band counts. The center band is the middle 40% × 40% of
 * the iframe; the corners are the outer 25% × 25% squares. A center-placed marker
 * yields center >> corners. Returns {center, cornerTL, cornerAll}.
 *
 * The report figure canvas may render via WebGPU (getImageData returns nothing on
 * a WebGPU context), so we SCREENSHOT the iframe element (captures WebGPU or
 * Canvas2D alike, per the report_export.spec.ts pattern) and decode the PNG.
 */
async function accentBands(page: any, figId: string): Promise<any> {
  const iframe = page.locator(`iframe[data-testid="figure-${figId}"]`)
  if (!(await iframe.count())) return { center: 0, cornerTL: 0, cornerAll: 0 }
  const box = await iframe.boundingBox()
  if (!box) return { center: 0, cornerTL: 0, cornerAll: 0 }
  // FULL-PAGE screenshot (proven to capture the WebGPU-composited orange marker;
  // a clipped shot of the OOPIF misses it), then classify pixels RESTRICTED to the
  // iframe's bounding-box region (CSS px → device px via devicePixelRatio).
  let buf: Buffer
  try {
    buf = await page.screenshot()
  } catch { return { center: 0, cornerTL: 0, cornerAll: 0 } }
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
    // Map the iframe's CSS-px rect to screenshot (device-px) coords.
    const dpr = W / window.innerWidth
    const x0 = box.x * dpr, y0 = box.y * dpr
    const bw = box.width * dpr, bh = box.height * dpr
    // #ff9800 = (255,152,0): warm ORANGE — high red, mid green, low blue. Robust
    // to matplotlib anti-aliasing over a dark/white base.
    const near = (r: number, g: number, b: number) =>
      r > 150 && g > 60 && g < 200 && b < 110 && (r - b) > 70 && (r - g) > 40 && (g - b) > 10
    let center = 0, cornerTL = 0, cornerAll = 0, upperLeft = 0, total = 0
    let sumFx = 0, sumFy = 0
    for (let p = 0; p < d.length; p += 4) {
      if (!near(d[p], d[p + 1], d[p + 2])) continue
      const idx = p / 4
      const px = idx % W, py = Math.floor(idx / W)
      if (px < x0 || px >= x0 + bw || py < y0 || py >= y0 + bh) continue
      const fx = (px - x0) / bw, fy = (py - y0) / bh
      total++; sumFx += fx; sumFy += fy
      if (fx > 0.3 && fx < 0.7 && fy > 0.3 && fy < 0.7) center++
      if (fx < 0.25 && fy < 0.25) cornerTL++
      if ((fx < 0.25 || fx > 0.75) && (fy < 0.25 || fy > 0.75)) cornerAll++
      if (fx < 0.5 && fy < 0.5) upperLeft++
    }
    const cx = total ? sumFx / total : -1
    const cy = total ? sumFy / total : -1
    return { center, cornerTL, cornerAll, upperLeft, total, cx, cy }
  }, { b64: buf.toString('base64'), box })
}

/** A fresh single-panel figure cell in a new report. Returns {cellId, figId}. */
async function makeFigureCell(page: any): Promise<{ cellId: string; figId: string }> {
  await page.getByTestId('toggle-report').click().catch(() => {})
  // Ensure the sidebar is open; if a report is already open, start fresh.
  if (!(await page.getByTestId('report-sidebar').count())) {
    await page.getByTestId('toggle-report').click()
  }
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await backendAction(page, 'report_new')
  await expect(page.getByTestId('report-body')).toBeVisible({ timeout: 10_000 })
  await expect(page.locator('[data-testid^="report-figcell-"]')).toHaveCount(0)

  const sig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
    .first()
  await sig.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-ann-sig', '1'))
  await dragToBody(page, '[data-ann-sig="1"]')

  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(2500)   // let the figure paint
  const cellId = await figCellId(page)
  const figId = (await reportFigId(page))!
  return { cellId, figId }
}

// Verify si_grains embed worked (the drag stamps the figure MIME).
async function assertNoBackendErrors() {
  const errs = backendErrorLines(ctx.backend)
    .filter((l: string) => /report|repfig|annotation|panel|figure/i.test(l))
  if (errs.length) console.log('[annotations] backend error lines:\n' + errs.join('\n'))
  expect(errs, 'report-related Python tracebacks/errors in backend log').toEqual([])
}

// ── 1 + 2: coordinate fix + drag persistence ───────────────────────────────────

test('coordinate fix: "+ Text" places the annotation at the image CENTER', async () => {
  const { page } = ctx
  await makeFigureCell(page)
  void FIG_MIME

  // Open edit → the single panel's PanelEdit shows the "+ Text" palette. In edit
  // mode we select the panel first so the panel-level palette is shown (a fresh
  // cell defaults to figure-level view). Select the only panel via its chip.
  // NB: toggling edit mode REBUILDS the figure (new figId), so always re-read the
  // CURRENT figId right before reading pixels.
  await openEdit(page)
  const chip = page.locator(`[data-testid^="figcell-chip-"]`).first()
  await expect(chip).toBeVisible()
  // The panel chips are single letters (A, B…); the "Figure" chip has a distinct
  // testid. Click the first LETTER chip (panel A).
  const panelChip = page.locator(`[data-testid^="figcell-chip-p"]`).first()
  await expect(panelChip).toBeVisible({ timeout: 10_000 })
  await panelChip.click()

  // Discover the panel id from the panel block that appears.
  const panelBlock = page.locator(`[data-testid^="figcell-panel-"]`).first()
  await expect(panelBlock).toBeVisible({ timeout: 10_000 })
  const panelId = await panelBlock.evaluate((el: HTMLElement) =>
    (el.getAttribute('data-testid') || '').replace('figcell-panel-', ''))

  // Baseline accent pixels (should be ~0 — no annotation yet). Re-read figId.
  const before = await accentBands(page, (await reportFigId(page))!)
  console.log('[annotations] accent bands BEFORE +Text =', JSON.stringify(before))

  // Click "+ Text" → the backend appends a text annotation at the panel center +
  // rebuilds the figure. Wait for the annotation ROW to appear.
  await page.getByTestId(`figcell-add-text-${panelId}`).click()
  await expect(page.locator(`[data-testid^="figcell-annotation-${panelId}-"]`).first())
    .toBeVisible({ timeout: 10_000 })
  await page.waitForTimeout(2500)   // let the rebuilt figure paint the marker

  await page.screenshot({ path: join(SHOTS, 'C-01-text-centered.png') })

  const after = await accentBands(page, (await reportFigId(page))!)
  console.log('[annotations] accent bands AFTER +Text =', JSON.stringify(after))

  // The marker drew accent pixels, and they cluster in the CENTER band — NOT the
  // corners. (The old bug placed it at the upper-left corner.)
  expect(after.center, 'no accent-colored pixels in the center band after +Text')
    .toBeGreaterThan(0)
  expect(after.center, 'text annotation did not land in the center (still corner?)')
    .toBeGreaterThan(after.cornerAll + 2)
  ctx.assertNoJsErrors()
  await assertNoBackendErrors()
})

test('drag persistence: a widget pointer_up moves the annotation + persists', async () => {
  const { page } = ctx
  const edit = await openEdit(page)
  const cellId = edit.cellId
  let figId = edit.figId   // re-read after any rebuild (edit-mode toggles rebuild)

  // The panel + its (one) text annotation from the previous test are live. In
  // edit mode the annotation renders as a draggable WIDGET — discover it.
  const widgets = await reportWidgets(page, figId)
  console.log('[annotations] widgets =', JSON.stringify(widgets.map(w => ({ t: w.type, p: w.panel_id }))))
  const labelW = widgets.find(w => w.type === 'label')
  expect(labelW, `no label widget in edit mode; got ${JSON.stringify(widgets.map(w => w.type))}`)
    .toBeTruthy()

  // The widget position is the PIXEL index of the annotation center. Read the
  // spec's DATA-coord offset BEFORE the drag from the report doc (the persisted
  // truth; the widget's pixel x is separate). The panel is calibrated, so the
  // stored offset differs from the pixel index.
  const panelId = labelW!.panel_id
  const xPxBefore = Number(labelW!.data.x)   // pixel index of the widget center
  console.log('[annotations] label widget before (px) =', xPxBefore, Number(labelW!.data.y))

  // The report-doc reader for this cell's first panel's first annotation offset.
  const specOffsetX = async (): Promise<number | null> =>
    await page.evaluate((cid: string) => {
      const d = (window as any)._spyde_test_report?.()
      const cell = d?.cells?.find((c: any) => c.id === cid)
      const panel = cell?.figure?.panels?.[0]
      const ann = panel?.annotations?.[0]
      const off = ann?.offsets?.[0]
      return off ? Number(off[0]) : null
    }, cellId)
  const dataXBefore = await specOffsetX()
  console.log('[annotations] annotation data-offset x BEFORE =', dataXBefore)
  expect(dataXBefore, 'no persisted annotation offset before drag').not.toBeNull()

  // Inject a pointer_up at a NEW pixel position (toward the top-left quadrant of
  // the image) — the drag-end the anyplotlib label widget posts. Shape from
  // test_report_edit_mode.py::_dispatch_up. A smaller pixel x → smaller data x.
  const newXpx = Math.max(4, xPxBefore * 0.25)
  const newYpx = Math.max(4, Number(labelW!.data.y) * 0.25)
  await figureEvent(page, figId, {
    panel_id: panelId, widget_id: labelW!.id, event_type: 'pointer_up',
    x: newXpx, y: newYpx,
  })

  // The persisted spec offset must move (no rebuild — the widget moved JS-side,
  // the backend persisted the new geometry into panel.annotations). Poll the doc.
  await expect.poll(async () => await specOffsetX(), {
    timeout: 10_000, message: 'annotation data offset did not move after pointer_up drag',
  }).toBeLessThan((dataXBefore as number) - 0.01)

  await page.waitForTimeout(800)
  await page.screenshot({ path: join(SHOTS, 'C-02a-annotation-dragged-editmode.png') })

  // Exit edit mode → the annotation re-renders as a STATIC marker AT THE MOVED
  // position (top-left quadrant now), not the center. Toggle the ✎ off.
  await page.getByTestId(`report-figcell-edit-toggle-${cellId}`).click()
  await expect(page.getByTestId(`figcell-edit-${cellId}`)).toHaveCount(0, { timeout: 10_000 })
  await page.waitForTimeout(2000)   // static-marker rebuild + paint

  figId = (await reportFigId(page))!   // rebuilt on edit-mode OFF
  const bands = await accentBands(page, figId)
  console.log('[annotations] accent bands after drag+exit =', JSON.stringify(bands))
  await page.screenshot({ path: join(SHOTS, 'C-02b-annotation-moved-static.png') })
  // The static marker re-rendered AT THE MOVED position: accent pixels exist and
  // their centroid is in the UPPER-LEFT quadrant (fx<0.5, fy<0.5) — it was dragged
  // there from the image center. (The text is drawn to the right of its anchor, so
  // assert on the quadrant centroid, not the extreme-corner band.)
  expect(bands.total, 'moved marker drew no accent pixels after exiting edit mode')
    .toBeGreaterThan(0)
  expect(bands.cx, 'moved marker centroid is not left-of-center').toBeLessThan(0.5)
  expect(bands.cy, 'moved marker centroid is not above center').toBeLessThan(0.5)
  expect(bands.upperLeft, 'most accent pixels are not in the upper-left quadrant')
    .toBeGreaterThan(bands.total / 2)

  // Re-enter edit → the widget is rebuilt from the persisted spec AT THE MOVED
  // spot (its pixel x is now smaller than before the drag). Poll for the rebuilt
  // figure's widget state to arrive at the renderer (figId changed → new state).
  await openEdit(page)
  await expect.poll(async () => {
    const fid = await reportFigId(page)
    if (!fid) return NaN
    const ws = await reportWidgets(page, fid)
    const lw = ws.find(w => w.type === 'label')
    return lw ? Number(lw.data.x) : NaN
  }, { timeout: 15_000, message: 'label widget missing / not at moved spot after re-entering edit' })
    .toBeLessThan(xPxBefore - 1)
  ctx.assertNoJsErrors()
  await assertNoBackendErrors()
})

// ── 3: selection UI (panel select / figure-background deselect) ─────────────────

test('selection: panel pointer_down selects (chips + panel controls); background deselects', async () => {
  const { page } = ctx
  const { cellId, figId } = await openEdit(page)

  // Discover the base panel's DISPATCH id (the panel_id widgets carry — it's the
  // panel plot's dispatch id, not the spec id "p1"). Any widget on the base panel
  // carries it; or read the report figure panel map via a widget.
  const widgets = await reportWidgets(page, figId)
  expect(widgets.length, 'no widgets in edit mode — cannot resolve a panel dispatch id')
    .toBeGreaterThan(0)
  const panelDispatchId = widgets[0].panel_id
  console.log('[annotations] panel dispatch id =', panelDispatchId)

  // Inject a genuine panel click (misses widgets) → pointer_down on the panel
  // plot → backend report_panel_selected → renderer selects the panel. Shape from
  // test_report_edit_mode.py::test_panel_pointer_down_selects_and_outlines.
  await figureEvent(page, figId, {
    panel_id: panelDispatchId, event_type: 'pointer_down',
  })

  // The dock's chips row shows + a panel LETTER chip becomes active, and the
  // panel block (panel-specific controls: Layers / Annotations) is shown.
  const chipsRow = page.getByTestId(`figcell-chips-${cellId}`)
  await expect(chipsRow).toBeVisible({ timeout: 10_000 })
  // A panel-specific block (figcell-panel-<specId>) must be present when a panel
  // is selected (figure-level view has NO figcell-panel- block).
  await expect.poll(async () =>
    page.locator(`[data-testid="figcell-edit-${cellId}"] [data-testid^="figcell-panel-"]`).count(),
    { timeout: 10_000, message: 'panel controls did not appear after panel pointer_down' })
    .toBeGreaterThan(0)
  // The Figure chip is NOT the active one now (a panel is selected).
  await page.waitForTimeout(400)
  await page.screenshot({ path: join(SHOTS, 'C-03a-panel-selected.png') })

  // Now inject a figure-BACKGROUND pointer_down → deselect → figure-level view.
  await figureEvent(page, figId, {
    panel_id: '', event_type: 'pointer_down', figure_background: true,
  })
  // The figure-level section shows (grid summary + figure annotation palette);
  // the panel-specific block is gone.
  await expect(page.getByTestId(`figcell-figure-edit-${cellId}`)).toBeVisible({ timeout: 10_000 })
  await expect(page.getByTestId(`figcell-grid-summary-${cellId}`)).toBeVisible()
  await expect.poll(async () =>
    page.locator(`[data-testid="figcell-edit-${cellId}"] [data-testid^="figcell-panel-"]`).count(),
    { timeout: 10_000, message: 'panel controls did not clear after figure-background click' })
    .toBe(0)
  await page.waitForTimeout(400)
  await page.screenshot({ path: join(SHOTS, 'C-03b-figure-deselected.png') })
  ctx.assertNoJsErrors()
  await assertNoBackendErrors()
})

// ── 4: figure-level annotation (add + drag persist) ─────────────────────────────

test('figure-level annotation: "+ Text" draws + a marker drag persists', async () => {
  const { page } = ctx
  const edit = await openEdit(page)
  const cellId = edit.cellId
  let figId = edit.figId

  // Ensure figure-level view (click the Figure chip).
  await page.getByTestId(`figcell-chip-figure-${cellId}`).click()
  await expect(page.getByTestId(`figcell-figure-edit-${cellId}`)).toBeVisible({ timeout: 10_000 })

  // Add a FIGURE-level Text annotation → repfig_add_fig_annotation → the figure
  // rebuilds with a figure-marker; a fig-annotation ROW appears. Count ONLY the
  // row container (testid `figcell-fig-annotation-<index>` = prefix + digits),
  // NOT its -text/-remove sub-elements (which share the prefix).
  const rowRe = /^figcell-fig-annotation-\d+$/
  const countFigRows = async () => await page.locator('[data-testid^="figcell-fig-annotation-"]')
    .evaluateAll((els, re: string) =>
      els.filter(e => new RegExp(re).test(e.getAttribute('data-testid') || '')).length,
      rowRe.source)
  const rowsBefore = await countFigRows()
  await page.getByTestId(`figcell-add-fig-text-${cellId}`).click()
  await expect.poll(countFigRows,
    { timeout: 10_000, message: 'figure-level +Text did not add a row' })
    .toBe(rowsBefore + 1)
  await page.waitForTimeout(2500)   // rebuilt figure paints the figure marker
  figId = (await reportFigId(page))!   // rebuilt on add
  await page.screenshot({ path: join(SHOTS, 'C-04a-fig-annotation-added.png') })

  // The figure-marker drew accent pixels centered over the whole figure (default
  // fraction 0.5, 0.5). Assert some accent pixels landed in the center band.
  const bands = await accentBands(page, figId)
  console.log('[annotations] fig-annotation accent bands =', JSON.stringify(bands))
  expect(bands.center, 'figure-level text annotation drew no centered accent pixels')
    .toBeGreaterThan(0)

  // Resolve the backend-assigned figure-annotation id from the report doc via the
  // additive test hook (window._spyde_test_report). The id is needed to inject a
  // figure-marker drag; it never surfaces in the DOM.
  const markerId = await page.evaluate((cid: string) => {
    const doc = (window as any)._spyde_test_report?.()
    const cell = doc?.cells?.find((c: any) => c.id === cid)
    const anns = cell?.figure?.annotations ?? []
    const a = anns[anns.length - 1]
    return (a?.id as string) ?? null
  }, cellId)
  console.log('[annotations] figure-marker id =', markerId)
  expect(markerId, 'figure annotation id not found in report doc (test hook)').toBeTruthy()

  // Inject a figure-marker pointer_up drag (shape from
  // test_report_edit_mode.py::TestFigureMarkerDrag). anyplotlib merges the moved
  // fraction fields into fig.figure_markers before firing pointer_up; the backend
  // persists by marker_id. Move it toward the lower-left (0.2, 0.75).
  await figureEvent(page, figId, {
    panel_id: '', event_type: 'pointer_up', figure_marker: true,
    marker_id: markerId, x: 0.2, y: 0.75,
  })

  // The persisted fraction must update in the report doc (no rebuild — the marker
  // already moved JS-side). Poll the test hook.
  await expect.poll(async () => {
    const val = await page.evaluate(({ cid, mid }: { cid: string; mid: string }) => {
      const d = (window as any)._spyde_test_report?.()
      const cell = d?.cells?.find((c: any) => c.id === cid)
      const anns = cell?.figure?.annotations ?? []
      const a = anns.find((x: any) => x.id === mid) ?? anns[anns.length - 1]
      return a ? Number(a.x) : null
    }, { cid: cellId, mid: markerId })
    return val
  }, { timeout: 10_000, message: 'figure marker x fraction did not persist after drag' })
    .toBeCloseTo(0.2, 1)
  await page.waitForTimeout(1500)
  await page.screenshot({ path: join(SHOTS, 'C-04b-fig-annotation-dragged.png') })
  ctx.assertNoJsErrors()
  await assertNoBackendErrors()
})

test('final: no Python tracebacks in the backend log', async () => {
  const errs = backendErrorLines(ctx.backend)
  if (errs.length) console.log('[annotations] backend error lines:\n' + errs.join('\n'))
  expect(errs, 'Python tracebacks/errors in backend log').toEqual([])
})
