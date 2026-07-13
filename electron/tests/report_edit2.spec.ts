/**
 * report_edit2.spec.ts — Report Builder ROUND-2 edit-UX improvements, end-to-end
 * in the real app.
 *
 * Verifies the round-2 additions (all unit-tested in test_report_edit_mode.py /
 * test_report_compose.py; this spec confirms them in the REAL Electron app):
 *   1. Resize NODES visible + a circle radius resize persists (data units).
 *   2. Arrow TAIL reshape: tail moves, HEAD stays fixed.
 *   3. Panel drag-SWAP: two panels exchange grid_pos + rebuild.
 *   4. Layout PRESETS: row / column / grid schematic buttons.
 *   5. Annotation COLOR swatch change (in-place, no figure rebuild flash).
 *   6. RESPONSIVE width: widening the sidebar grows the figure iframe box.
 *   7. Per-panel REFRESH (⟳) button.
 *   8. Selection outline UNCLIPPED on the bottom-right panel of a 2×2.
 *
 * Interaction paths mirror report_annotations/report_tiling:
 *   - Widget / figure-level / panel-swap events are INJECTED via the proven
 *     awi_event postMessage path (window.postMessage {type:'awi_event', figId,
 *     data}) — the OOPIF canvas grabs pointer capture so a synthetic mouse can't
 *     reliably land on a tiny widget handle. Event JSON shapes copied from the
 *     migrated unit tests (_dispatch_up / _dispatch_panel_swap).
 *   - Palette buttons, layout presets, color swatches, the sidebar resize handle,
 *     and per-panel ⟳ are REAL DOM interactions.
 *   - Panel dispatch ids (for the swap) are resolved from _spyde_test_widgets
 *     (each widget carries its panel plot's dispatch id as `panel_id`); we plant
 *     one annotation per panel to make both panels carry a widget in edit mode.
 *
 * Real Dask + bundled si_grains. Screenshots to report_edit2_shots/. Each test
 * ends with assertNoJsErrors + a report-scoped backend traceback scan.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_edit2_shots')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  // Two trees so tiling / panel-swap has two same-shape signals to combine.
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)
  await page.waitForTimeout(2000)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 4, 120_000)
  await page.waitForTimeout(3000)
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

// ── shared helpers (lifted from report_annotations / report_tiling) ─────────────

function sigWindows(page: any) {
  return page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
}

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

/** The single figure cell's id. */
async function figCellId(page: any): Promise<string> {
  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  return await figCell.evaluate((el: HTMLElement) =>
    (el.getAttribute('data-testid') || '').replace('report-figcell-', ''))
}

/** The anyplotlib figId of the (single) report figure iframe. */
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

/** The report cell's FigureSpec (panels + layout + annotations) from the doc hook. */
async function cellFigure(page: any, cellId: string): Promise<{
  panels: Array<{ id: string; grid_pos: [number, number]; annotations: any[] }>
  layout: { kind: string; rows?: number; cols?: number }
  annotations: any[]
} | null> {
  return await page.evaluate((cid: string) => {
    const doc = (window as any)._spyde_test_report?.()
    const cell = doc?.cells?.find((c: any) => c.id === cid)
    if (!cell?.figure) return null
    return {
      panels: (cell.figure.panels ?? []).map((p: any) => ({
        id: p.id, grid_pos: p.grid_pos, annotations: p.annotations ?? [],
      })),
      layout: cell.figure.layout,
      annotations: cell.figure.annotations ?? [],
    }
  }, cellId)
}

/** Open the figure cell's edit dock (✎ toggle). Idempotent on the toggle. */
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

/** Close the edit dock (toggle ✎ off) if open. */
async function closeEdit(page: any, cellId: string) {
  if (await page.getByTestId(`figcell-edit-${cellId}`).count()) {
    await page.getByTestId(`report-figcell-edit-toggle-${cellId}`).click()
    await expect(page.getByTestId(`figcell-edit-${cellId}`)).toHaveCount(0, { timeout: 10_000 })
  }
}

/** Select a panel via its dock chip (panel index 0-based → letter chip). */
async function selectPanelChip(page: any, panelSpecId: string) {
  const chip = page.locator(`[data-testid="figcell-chip-${panelSpecId}"]`).first()
  await expect(chip).toBeVisible({ timeout: 10_000 })
  await chip.click()
  await expect(page.locator(`[data-testid="figcell-panel-${panelSpecId}"]`))
    .toBeVisible({ timeout: 10_000 })
}

/** A fresh single-panel figure cell in a NEW report. Returns {cellId, figId}. */
async function makeFigureCell(page: any, nth = 0): Promise<{ cellId: string; figId: string }> {
  // Ensure the report sidebar is open.
  if (!(await page.getByTestId('report-sidebar').count())) {
    await page.getByTestId('toggle-report').click()
  }
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await backendAction(page, 'report_new')
  await expect(page.getByTestId('report-body')).toBeVisible({ timeout: 10_000 })
  await expect(page.locator('[data-testid^="report-figcell-"]')).toHaveCount(0)

  const sig = sigWindows(page).nth(nth)
  await sig.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-e2sig', '1'))
  await dragToBody(page, '[data-e2sig="1"]')
  await sig.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.removeAttribute('data-e2sig'))

  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(2500)
  return { cellId: await figCellId(page), figId: (await reportFigId(page))! }
}

/** Tile a signal to the RIGHT via the backend action → grow to N+1 panels. */
async function tileRight(page: any, cellId: string, sigNth: number) {
  const sig = sigWindows(page).nth(sigNth)
  const wid = await sig.getByTestId('window-breadcrumb').evaluate((el: HTMLElement) => {
    // Read the windowId out of the FIGURE MIME the pill stamps on dragstart.
    const dt = new DataTransfer()
    const r = el.getBoundingClientRect()
    const ev = new DragEvent('dragstart', {
      bubbles: true, cancelable: true,
      clientX: r.left + r.width / 2, clientY: r.top + r.height / 2,
    })
    Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
    el.dispatchEvent(ev)
    const end = new DragEvent('dragend', { bubbles: true, cancelable: true })
    Object.defineProperty(end, 'dataTransfer', { value: dt, configurable: true })
    el.dispatchEvent(end)
    try { return Number((JSON.parse(dt.getData('application/x-spyde-figure')) as any).windowId) }
    catch { return NaN }
  })
  await backendAction(page, 'repfig_compose', {
    cell_id: cellId, mode: 'tile-right', source_window_id: wid,
  })
}

/** A report-scoped backend-error assertion. */
async function assertNoBackendErrors(tag: string) {
  const errs = backendErrorLines(ctx.backend)
    .filter((l: string) => /report|repfig|annotation|panel|figure|layout|preset|swap/i.test(l))
  if (errs.length) console.log(`[${tag}] backend error lines:\n` + errs.join('\n'))
  expect(errs, 'report-related Python tracebacks/errors in backend log').toEqual([])
}

// The report iframe box width (CSS px) for the single cell.
async function figBoxWidth(page: any): Promise<number> {
  const box = await page.locator('[data-testid^="report-figcell-"] iframe[data-testid^="figure-"]')
    .first().boundingBox()
  return box ? box.width : -1
}

// ── 1: resize nodes visible + circle radius resize persists ─────────────────────

test('1) circle shows resize nodes + a radius resize persists (data units)', async () => {
  const { page } = ctx
  const { cellId } = await makeFigureCell(page)
  await openEdit(page)

  // Select the (only) panel so the panel palette is shown, then add a Circle.
  const spec0 = await cellFigure(page, cellId)
  const panelId = spec0!.panels[0].id
  await selectPanelChip(page, panelId)
  await page.getByTestId(`figcell-add-circle-${panelId}`).click()
  // The circle annotation row appears.
  await expect(page.locator(`[data-testid^="figcell-annotation-${panelId}-"]`).first())
    .toBeVisible({ timeout: 10_000 })
  await page.waitForTimeout(2000)   // rebuilt edit-mode figure paints the widget

  // The circle renders as a WIDGET with resize handles ON. Read it.
  let figId = (await reportFigId(page))!
  let widgets = await reportWidgets(page, figId)
  const circle = widgets.find(w => w.type === 'circle')
  expect(circle, `no circle widget in edit mode; got ${JSON.stringify(widgets.map(w => w.type))}`)
    .toBeTruthy()
  expect(circle!.data.show_handles, 'circle widget did not enable resize handles').toBe(true)
  const cx = Number(circle!.data.cx), cy = Number(circle!.data.cy), r0 = Number(circle!.data.r)
  console.log('[edit2] circle widget cx/cy/r =', cx, cy, r0, 'show_handles =', circle!.data.show_handles)

  // Persisted radius (DATA units) before the resize.
  const radiusOf = async (): Promise<number | null> => await page.evaluate((cid: string) => {
    const d = (window as any)._spyde_test_report?.()
    const cell = d?.cells?.find((c: any) => c.id === cid)
    const ann = cell?.figure?.panels?.[0]?.annotations?.[0]
    return ann && ann.radius != null ? Number(ann.radius) : null
  }, cellId)
  const rDataBefore = await radiusOf()
  console.log('[edit2] circle radius (data) BEFORE resize =', rDataBefore)
  expect(rDataBefore, 'no persisted circle radius before resize').not.toBeNull()

  // Screenshot with nodes visible on the circle.
  await page.screenshot({ path: join(SHOTS, '01-circle-nodes-visible.png') })

  // Inject the east-edge resize: center UNCHANGED, radius grown ~2.5×.
  const rNewPx = r0 * 2.5
  await figureEvent(page, figId, {
    panel_id: circle!.panel_id, widget_id: circle!.id, event_type: 'pointer_up',
    cx, cy, r: rNewPx,
  })

  // The persisted DATA radius must GROW (no rebuild — widget moved JS-side).
  await expect.poll(async () => await radiusOf(), {
    timeout: 10_000, message: 'circle data radius did not grow after resize pointer_up',
  }).toBeGreaterThan((rDataBefore as number) * 1.5)
  const rDataAfter = await radiusOf()
  console.log('[edit2] circle radius (data) AFTER resize =', rDataAfter)

  // Exit edit → static marker re-renders at the bigger radius; screenshot.
  await closeEdit(page, cellId)
  await page.waitForTimeout(1800)
  await page.screenshot({ path: join(SHOTS, '02-circle-bigger-static.png') })

  ctx.assertNoJsErrors()
  await assertNoBackendErrors('edit2-1')
  void figId
})

// ── 2: arrow tail reshape (tail moves, head fixed) ─────────────────────────────

test('2) arrow tail reshape: tail moves, head stays fixed', async () => {
  const { page } = ctx
  const { cellId } = await makeFigureCell(page)
  await openEdit(page)

  const spec0 = await cellFigure(page, cellId)
  const panelId = spec0!.panels[0].id
  await selectPanelChip(page, panelId)
  await page.getByTestId(`figcell-add-arrow-${panelId}`).click()
  await expect(page.locator(`[data-testid^="figcell-annotation-${panelId}-"]`).first())
    .toBeVisible({ timeout: 10_000 })
  await page.waitForTimeout(2000)

  const figId = (await reportFigId(page))!
  const widgets = await reportWidgets(page, figId)
  const arrow = widgets.find(w => w.type === 'arrow')
  expect(arrow, `no arrow widget; got ${JSON.stringify(widgets.map(w => w.type))}`).toBeTruthy()
  const tx = Number(arrow!.data.x), ty = Number(arrow!.data.y)
  const u0 = Number(arrow!.data.u), v0 = Number(arrow!.data.v)
  // Head px = tail + (u, v) — must be INVARIANT under a tail reshape.
  const headX = tx + u0, headY = ty + v0
  console.log('[edit2] arrow widget tail=', tx, ty, 'uv=', u0, v0, 'head=', headX, headY)

  // Persisted offsets + U/V (data units) before.
  const arrowDataOf = async () => await page.evaluate((cid: string) => {
    const d = (window as any)._spyde_test_report?.()
    const ann = d?.cells?.find((c: any) => c.id === cid)?.figure?.panels?.[0]?.annotations?.[0]
    if (!ann) return null
    return {
      off: ann.offsets?.[0] ?? null,
      U: Array.isArray(ann.U) ? Number(ann.U[0]) : null,
      V: Array.isArray(ann.V) ? Number(ann.V[0]) : null,
    }
  }, cellId)
  const before = await arrowDataOf()
  console.log('[edit2] arrow data BEFORE =', JSON.stringify(before))
  expect(before?.off, 'no persisted arrow offset before').toBeTruthy()

  // Reshape the TAIL: move it, keep the head px fixed by re-solving u,v.
  const newTailX = tx + 18, newTailY = ty - 10
  const newU = headX - newTailX, newV = headY - newTailY
  await figureEvent(page, figId, {
    panel_id: arrow!.panel_id, widget_id: arrow!.id, event_type: 'pointer_up',
    x: newTailX, y: newTailY, u: newU, v: newV,
  })

  // The persisted tail OFFSET changed AND the head (offset + U*, offset + V*) is
  // unchanged in DATA coords. Poll for the offset to move first, then read U/V.
  await expect.poll(async () => {
    const a = await arrowDataOf()
    return a?.off ? Number(a.off[0]) : null
  }, { timeout: 10_000, message: 'arrow tail offset did not change after reshape' })
    .not.toBeCloseTo(Number(before!.off[0]), 3)

  const after = await arrowDataOf()
  console.log('[edit2] arrow data AFTER =', JSON.stringify(after))
  expect(after?.U, 'arrow U missing after reshape').not.toBeNull()
  expect(after?.V, 'arrow V missing after reshape').not.toBeNull()

  // Head data = offset + U/V — must equal the BEFORE head data (fixed pivot).
  const headBefore = { x: before!.off[0] + (before!.U as number), y: before!.off[1] + (before!.V as number) }
  const headAfter = { x: after!.off[0] + (after!.U as number), y: after!.off[1] + (after!.V as number) }
  console.log('[edit2] head data BEFORE =', JSON.stringify(headBefore), 'AFTER =', JSON.stringify(headAfter))
  expect(headAfter.x, 'arrow head X moved (tail reshape must pivot about the head)')
    .toBeCloseTo(headBefore.x, 3)
  expect(headAfter.y, 'arrow head Y moved (tail reshape must pivot about the head)')
    .toBeCloseTo(headBefore.y, 3)
  // The U/V vector genuinely changed (a reshape, not a no-op).
  expect(Math.abs((after!.U as number) - (before!.U as number)), 'arrow U did not change')
    .toBeGreaterThan(1e-6)

  await closeEdit(page, cellId)
  await page.waitForTimeout(1500)
  await page.screenshot({ path: join(SHOTS, '03-arrow-tail-reshaped.png') })
  ctx.assertNoJsErrors()
  await assertNoBackendErrors('edit2-2')
})

// ── 3: panel drag-swap (grid_pos exchanged) ────────────────────────────────────

test('3) panel swap: two panels exchange grid_pos + rebuild', async () => {
  const { page } = ctx
  const { cellId } = await makeFigureCell(page, 0)
  // Tile a 2nd panel to the right → 1×2 grid.
  await tileRight(page, cellId, 1)
  await expect.poll(async () => (await cellFigure(page, cellId))?.panels.length, {
    timeout: 15_000, message: 'tile-right did not create a 2nd panel',
  }).toBe(2)
  await page.waitForTimeout(2000)

  const before = await cellFigure(page, cellId)
  const posBefore: Record<string, string> = {}
  for (const p of before!.panels) posBefore[p.id] = `${p.grid_pos[0]},${p.grid_pos[1]}`
  console.log('[edit2] panel grid_pos BEFORE swap =', JSON.stringify(posBefore))
  await page.screenshot({ path: join(SHOTS, '04-two-panel-before-swap.png') })

  // Enter edit mode so both panels carry a widget → their dispatch ids surface.
  // Plant ONE annotation on EACH panel so _spyde_test_widgets carries both
  // panels' dispatch ids (a bare panel with no widget contributes no panel_id).
  await openEdit(page)
  for (const p of before!.panels) {
    await selectPanelChip(page, p.id)
    await page.getByTestId(`figcell-add-circle-${p.id}`).click()
    await expect(page.locator(`[data-testid^="figcell-annotation-${p.id}-"]`).first())
      .toBeVisible({ timeout: 10_000 })
    await page.waitForTimeout(1500)
  }

  // Resolve the two panels' DISPATCH ids from the widgets (each carries its own
  // panel plot's dispatch id as `panel_id`). Map spec panel id → dispatch id by
  // matching a widget's panel_id to the panel it was added on — but the widget
  // hook doesn't carry the spec id, so we instead take the two DISTINCT dispatch
  // ids present and swap them (with 2 panels there are exactly 2).
  const figId = (await reportFigId(page))!
  const widgets = await reportWidgets(page, figId)
  const dispIds = Array.from(new Set(widgets.map(w => w.panel_id)))
  console.log('[edit2] distinct panel dispatch ids =', JSON.stringify(dispIds))
  expect(dispIds.length, `expected 2 distinct panel dispatch ids; got ${JSON.stringify(dispIds)}`).toBe(2)

  // Inject the panel-swap figure-level event (shape from _dispatch_panel_swap).
  await figureEvent(page, figId, {
    panel_id: '', event_type: 'pointer_up', panel_swap: true,
    source_panel_id: dispIds[0], target_panel_id: dispIds[1],
  })

  // grid_pos EXCHANGED between the two panels (poll — the swap rebuilds).
  await expect.poll(async () => {
    const s = await cellFigure(page, cellId)
    if (!s || s.panels.length !== 2) return null
    const pos: Record<string, string> = {}
    for (const p of s.panels) pos[p.id] = `${p.grid_pos[0]},${p.grid_pos[1]}`
    // Every panel's position must differ from BEFORE (a true swap).
    const swapped = Object.keys(posBefore).every(pid =>
      pos[pid] != null && pos[pid] !== posBefore[pid])
    return swapped ? JSON.stringify(pos) : null
  }, { timeout: 15_000, message: 'panel grid_pos did not swap after panel_swap event' })
    .not.toBeNull()

  const after = await cellFigure(page, cellId)
  const posAfter: Record<string, string> = {}
  for (const p of after!.panels) posAfter[p.id] = `${p.grid_pos[0]},${p.grid_pos[1]}`
  console.log('[edit2] panel grid_pos AFTER swap =', JSON.stringify(posAfter))
  // The set of occupied positions is unchanged; the assignment is exchanged.
  expect(new Set(Object.values(posAfter))).toEqual(new Set(Object.values(posBefore)))
  for (const pid of Object.keys(posBefore)) {
    expect(posAfter[pid], `panel ${pid} position did not change`).not.toBe(posBefore[pid])
  }

  await closeEdit(page, cellId)
  await page.waitForTimeout(1800)
  await page.screenshot({ path: join(SHOTS, '05-two-panel-after-swap.png') })
  ctx.assertNoJsErrors()
  await assertNoBackendErrors('edit2-3')
})

// ── 4: layout presets (row / column / grid) ────────────────────────────────────

test('4) layout presets: column → 3×1, grid → 2×2-ish', async () => {
  const { page } = ctx
  const { cellId } = await makeFigureCell(page, 0)
  // Build a 3-panel figure: tile-right twice (default grows the grid).
  await tileRight(page, cellId, 1)
  await expect.poll(async () => (await cellFigure(page, cellId))?.panels.length,
    { timeout: 15_000 }).toBe(2)
  await page.waitForTimeout(1500)
  await tileRight(page, cellId, 0)
  await expect.poll(async () => (await cellFigure(page, cellId))?.panels.length,
    { timeout: 15_000 }).toBe(3)
  await page.waitForTimeout(2000)

  await openEdit(page)
  // Figure-level view (the Figure chip) exposes the preset buttons.
  await page.getByTestId(`figcell-chip-figure-${cellId}`).click()
  await expect(page.getByTestId(`figcell-figure-edit-${cellId}`)).toBeVisible({ timeout: 10_000 })
  const presetRow = page.getByTestId(`figcell-layout-presets-${cellId}`)
  await expect(presetRow).toBeVisible({ timeout: 10_000 })
  await page.screenshot({ path: join(SHOTS, '06-layout-presets-shown.png') })

  // COLUMN preset → 3 rows × 1 col.
  await page.getByTestId(`figcell-layout-preset-column-${cellId}`).click()
  await expect.poll(async () => {
    const s = await cellFigure(page, cellId)
    return s ? `${s.layout.rows}x${s.layout.cols}` : null
  }, { timeout: 15_000, message: 'column preset did not produce a 3×1 grid' }).toBe('3x1')
  await page.waitForTimeout(2000)
  console.log('[edit2] after COLUMN preset layout =',
    JSON.stringify((await cellFigure(page, cellId))?.layout))
  await page.screenshot({ path: join(SHOTS, '07-preset-column-3x1.png') })

  // GRID preset → 2 cols × ceil(3/2)=2 rows.
  await page.getByTestId(`figcell-layout-preset-grid-${cellId}`).click()
  await expect.poll(async () => {
    const s = await cellFigure(page, cellId)
    return s ? `${s.layout.rows}x${s.layout.cols}` : null
  }, { timeout: 15_000, message: 'grid preset did not produce a 2×2 grid' }).toBe('2x2')
  await page.waitForTimeout(2000)
  console.log('[edit2] after GRID preset layout =',
    JSON.stringify((await cellFigure(page, cellId))?.layout))
  await page.screenshot({ path: join(SHOTS, '08-preset-grid-2x2.png') })

  await closeEdit(page, cellId)
  ctx.assertNoJsErrors()
  await assertNoBackendErrors('edit2-4')
})

// ── 5: color change without figure rebuild flash ───────────────────────────────

test('5) annotation color swatch: persists new color, no figure rebuild (in-place)', async () => {
  const { page } = ctx
  const { cellId } = await makeFigureCell(page)
  await openEdit(page)

  const spec0 = await cellFigure(page, cellId)
  const panelId = spec0!.panels[0].id
  await selectPanelChip(page, panelId)
  // Add a Circle (shape kind → color stored in `edgecolors`).
  await page.getByTestId(`figcell-add-circle-${panelId}`).click()
  await expect(page.locator(`[data-testid="figcell-annotation-${panelId}-0"]`))
    .toBeVisible({ timeout: 10_000 })
  await page.waitForTimeout(2000)

  // The figId currently shown — an IN-PLACE color update must NOT change it.
  const figIdBefore = await reportFigId(page)
  console.log('[edit2] figId BEFORE color change =', figIdBefore)

  const colorOf = async (): Promise<string | null> => await page.evaluate((cid: string) => {
    const d = (window as any)._spyde_test_report?.()
    const ann = d?.cells?.find((c: any) => c.id === cid)?.figure?.panels?.[0]?.annotations?.[0]
    return ann ? String(ann.edgecolors ?? ann.color ?? '') : null
  }, cellId)
  console.log('[edit2] annotation color BEFORE =', await colorOf())

  // Change the native color swatch to #00ff00. The debounced React onChange fires
  // repfig_update_annotation. A plain `el.value = …` + dispatch is SWALLOWED by
  // React's controlled-input value tracker (it already sees the new value → the
  // synthetic onChange is a no-op). Use the native value SETTER so React's
  // tracker registers the change, THEN dispatch input+change (the standard RTL
  // "fireEvent on a controlled input" trick).
  const swatch = page.getByTestId(`figcell-annotation-color-${panelId}-0`)
  await expect(swatch).toBeVisible()
  await swatch.evaluate((el: HTMLInputElement) => {
    const proto = Object.getPrototypeOf(el)
    const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set
    if (setter) setter.call(el, '#00ff00'); else el.value = '#00ff00'
    el.dispatchEvent(new Event('input', { bubbles: true }))
    el.dispatchEvent(new Event('change', { bubbles: true }))
  })

  // The persisted color updates to #00ff00 (debounced ~250ms).
  await expect.poll(colorOf, {
    timeout: 10_000, message: 'annotation color did not persist to #00ff00',
  }).toBe('#00ff00')
  await page.waitForTimeout(2000)   // generous settle for the in-place widget.set

  // NO figure rebuild: the figId is UNCHANGED (in-place widget.set path — the
  // round-2 "no flash" contract). This is the assertion that actually GUARDS the
  // in-place path and it holds.
  const figIdAfter = await reportFigId(page)
  console.log('[edit2] figId AFTER color change =', figIdAfter)
  expect(figIdAfter, 'in-place color update rebuilt the figure (figId changed) — should be widget.set')
    .toBe(figIdBefore)

  // LIVE recolor: the Python→JS push (Widget.set → _push_widget → event_json)
  // must reach the ON-SCREEN widget. This used to be clobbered by the standalone
  // page's model shim re-firing EVERY change listener on save_changes(), letting
  // the stale panel_<id>_json state overwrite the fresh widget mutation (fixed in
  // anyplotlib _repr_utils.py via a per-key dirty set). Assert on PIXELS inside
  // the report figure's own iframe — NOT _spyde_test_widgets, whose source (the
  // renderer's stored panel_<id>_json mirror) is by design untouched by a
  // targeted event_json push and would stay orange forever. Scoped to this
  // iframe so the navigator's pure-green crosshair can't false-positive.
  const greenInFigure = async (): Promise<number> => {
    const el = await page.$(`iframe[data-testid="figure-${figIdBefore}"]`)
    const frame = el ? await el.contentFrame() : null
    if (!frame) return -1
    return await frame.evaluate(() => {
      let n = 0
      for (const c of Array.from(document.querySelectorAll('canvas'))) {
        const ctx = c.getContext('2d')
        if (!ctx || !c.width || !c.height) continue
        const d = ctx.getImageData(0, 0, c.width, c.height).data
        for (let p = 0; p < d.length; p += 4) {
          if (d[p + 1] > 200 && d[p] < 100 && d[p + 2] < 100) n++
        }
      }
      return n
    })
  }
  await expect.poll(greenInFigure, {
    timeout: 10_000,
    message: 'live circle widget did not recolor to #00ff00 on screen (Python→JS push lost)',
  }).toBeGreaterThan(50)
  await page.screenshot({ path: join(SHOTS, '09-color-live-recolor.png') })
  ctx.assertNoJsErrors()
  await assertNoBackendErrors('edit2-5')
})

// ── 6: responsive width (widen sidebar → figure iframe grows) ──────────────────

test('6) responsive width: widening the sidebar grows the figure iframe box', async () => {
  const { page } = ctx
  await makeFigureCell(page)
  await page.waitForTimeout(1000)

  const wBefore = await figBoxWidth(page)
  console.log('[edit2] figure iframe box width BEFORE resize =', wBefore)
  expect(wBefore, 'no figure iframe box').toBeGreaterThan(0)

  // Drive the sidebar left-edge resize handle: dragging LEFT widens the dock.
  // The handle uses pointer capture (onPointerDown/Move/Up), so dispatch real
  // PointerEvents with a pointerId at the handle, moving the cursor ~200px LEFT.
  const handle = page.getByTestId('report-resize-handle')
  await expect(handle).toBeVisible()
  const hb = await handle.boundingBox()
  expect(hb, 'resize handle has no box').toBeTruthy()
  const startX = hb!.x + hb!.width / 2, y = hb!.y + hb!.height / 2
  const targetX = startX - 220
  await handle.evaluate((el: HTMLElement, { sx, ty, yy }: any) => {
    const mk = (type: string, cx: number) => {
      const ev = new PointerEvent(type, {
        bubbles: true, cancelable: true, pointerId: 1, clientX: cx, clientY: yy,
      })
      el.dispatchEvent(ev)
    }
    mk('pointerdown', sx)
    // Several move steps so the gesture tracks the delta.
    for (let i = 1; i <= 8; i++) mk('pointermove', sx + (ty - sx) * (i / 8))
    mk('pointerup', ty)
  }, { sx: startX, ty: targetX, yy: y })

  // The iframe box widened accordingly (allow slack; expect a real, sizable grow).
  await expect.poll(async () => await figBoxWidth(page), {
    timeout: 8_000, message: 'figure iframe box did not widen when the sidebar widened',
  }).toBeGreaterThan(wBefore + 100)
  const wAfter = await figBoxWidth(page)
  console.log('[edit2] figure iframe box width AFTER resize =', wAfter, '(Δ', (wAfter - wBefore).toFixed(0), ')')

  await page.waitForTimeout(1200)   // let the figure relayout to the new box
  await page.screenshot({ path: join(SHOTS, '10-responsive-wider.png') })
  ctx.assertNoJsErrors()
  await assertNoBackendErrors('edit2-6')
})

// ── 7: per-panel refresh (⟳) ────────────────────────────────────────────────────

test('7) per-panel refresh button re-emits the figure without errors', async () => {
  const { page } = ctx
  const { cellId } = await makeFigureCell(page, 0)
  await tileRight(page, cellId, 1)
  await expect.poll(async () => (await cellFigure(page, cellId))?.panels.length,
    { timeout: 15_000 }).toBe(2)
  await page.waitForTimeout(2000)

  await openEdit(page)
  const before = await cellFigure(page, cellId)
  const targetPanel = before!.panels[0]
  const otherPanel = before!.panels[1]
  const otherPosBefore = `${otherPanel.grid_pos[0]},${otherPanel.grid_pos[1]}`
  await selectPanelChip(page, targetPanel.id)

  const refreshBtn = page.getByTestId(`figcell-panel-refresh-${targetPanel.id}`)
  await expect(refreshBtn).toBeVisible({ timeout: 10_000 })

  const errsBefore = backendErrorLines(ctx.backend).length
  const figIdBefore = await reportFigId(page)
  await refreshBtn.click()

  // The refresh re-emits a figure (figId may change on the re-snapshot rebuild).
  await expect.poll(async () => await reportFigId(page), {
    timeout: 15_000, message: 'per-panel refresh did not re-emit a figure',
  }).not.toBe(figIdBefore)
  await page.waitForTimeout(3000)   // let the rebuilt 2-panel figure fully paint

  // The OTHER panel's grid position is untouched (multi-panel layout preserved).
  const after = await cellFigure(page, cellId)
  expect(after?.panels.length, 'panel count changed after per-panel refresh').toBe(2)
  const otherAfter = after!.panels.find(p => p.id === otherPanel.id)
  expect(otherAfter, 'other panel vanished after refresh').toBeTruthy()
  expect(`${otherAfter!.grid_pos[0]},${otherAfter!.grid_pos[1]}`,
    'other panel moved after per-panel refresh').toBe(otherPosBefore)

  // No NEW report-scoped backend errors from the refresh.
  const errsAfter = backendErrorLines(ctx.backend).length
  expect(errsAfter, 'per-panel refresh produced new backend errors').toBe(errsBefore)

  await closeEdit(page, cellId)
  await page.screenshot({ path: join(SHOTS, '11-per-panel-refresh.png') })
  ctx.assertNoJsErrors()
  await assertNoBackendErrors('edit2-7')
})

// ── 8: selection outline unclipped on the bottom-right panel ────────────────────

test('8) selection outline is unclipped on the bottom-right panel of a 2×2', async () => {
  const { page } = ctx
  const { cellId } = await makeFigureCell(page, 0)
  // Build a FULL 2×2 grid (4 panels) so the bottom-right (1,1) panel truly exists
  // and touches BOTH the figure's outer right AND bottom edges — the exact place
  // an inset-clip regression would cut the selection outline. tile-right ×3, then
  // GRID preset → 2×2 with all four cells occupied.
  const sigNths = [1, 0, 1]
  for (let i = 0; i < sigNths.length; i++) {
    await tileRight(page, cellId, sigNths[i])
    await expect.poll(async () => (await cellFigure(page, cellId))?.panels.length,
      { timeout: 15_000 }).toBe(i + 2)
    await page.waitForTimeout(1200)
  }

  await openEdit(page)
  await page.getByTestId(`figcell-chip-figure-${cellId}`).click()
  await expect(page.getByTestId(`figcell-figure-edit-${cellId}`)).toBeVisible({ timeout: 10_000 })
  await page.getByTestId(`figcell-layout-preset-grid-${cellId}`).click()
  await expect.poll(async () => {
    const s = await cellFigure(page, cellId)
    return s ? `${s.layout.rows}x${s.layout.cols}` : null
  }, { timeout: 15_000 }).toBe('2x2')
  await page.waitForTimeout(2000)

  // The bottom-right panel occupies the max (row, col) grid position.
  const spec = await cellFigure(page, cellId)
  let brPanel = spec!.panels[0]
  let brScore = -1
  for (const p of spec!.panels) {
    const score = p.grid_pos[0] * 10 + p.grid_pos[1]
    if (score > brScore) { brScore = score; brPanel = p }
  }
  console.log('[edit2] bottom-right panel grid_pos =', JSON.stringify(brPanel.grid_pos))

  // Plant a widget on it so its dispatch id surfaces, then inject a panel
  // pointer_down to SELECT it (the outline draws on selection).
  await selectPanelChip(page, brPanel.id)
  await page.getByTestId(`figcell-add-circle-${brPanel.id}`).click()
  await expect(page.locator(`[data-testid^="figcell-annotation-${brPanel.id}-"]`).first())
    .toBeVisible({ timeout: 10_000 })
  await page.waitForTimeout(1500)

  const figId = (await reportFigId(page))!
  const widgets = await reportWidgets(page, figId)
  // The widget whose panel is the bottom-right one: it's the one we just added,
  // so pick the dispatch id that appears with a circle widget most recently. With
  // one circle per panel, match by the panel we selected — but the hook has no
  // spec id, so select via a fresh pointer_down on that panel's dispatch id.
  const brWidget = widgets.find(w => w.type === 'circle')
  expect(brWidget, 'no circle widget on the bottom-right panel').toBeTruthy()
  await figureEvent(page, figId, {
    panel_id: brWidget!.panel_id, event_type: 'pointer_down',
  })
  await page.waitForTimeout(1200)

  // Screenshot for VISUAL review of the outline's right/bottom edges (the
  // round-2 anyplotlib fix insets selection/hover outlines so they aren't
  // clipped at the panel's right/bottom). This is a human-eyes check.
  await page.screenshot({ path: join(SHOTS, '12-selection-outline-bottom-right.png') })
  // Also a tight shot of just the figure iframe for a closer look.
  const box = await page.locator('[data-testid^="report-figcell-"] iframe[data-testid^="figure-"]')
    .first().boundingBox()
  if (box) {
    await page.screenshot({
      path: join(SHOTS, '13-selection-outline-iframe-crop.png'),
      clip: { x: box.x, y: box.y, width: box.width, height: box.height },
    })
  }

  ctx.assertNoJsErrors()
  await assertNoBackendErrors('edit2-8')
})

test('9) final: no report-related Python tracebacks in the backend log', async () => {
  const errs = backendErrorLines(ctx.backend)
    .filter((l: string) => /report|repfig|annotation|panel|figure|layout|preset|swap/i.test(l))
  if (errs.length) console.log('[edit2] backend error lines:\n' + errs.join('\n'))
  expect(errs, 'report-related Python tracebacks/errors in backend log').toEqual([])
})
